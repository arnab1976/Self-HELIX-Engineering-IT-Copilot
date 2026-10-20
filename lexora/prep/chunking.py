"""Structure-aware chunking.

Code: one chunk per function / method / class header, with path, symbol, line range, imports used and
called names (for code-graph edges). Python uses the standard-library `ast` (exact line numbers);
other languages use tree-sitter when installed (`tree-sitter-language-pack`), else a regex fallback.

Docs: heading-aware chunks packed to a token budget with overlap (default 600 tokens, 15%).
"""
from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field

from common.textutils import estimate_tokens
from lexora.ingest.base import Document


@dataclass
class Chunk:
    repo: str
    path: str
    start: int
    end: int
    symbol: str
    kind: str                     # function | method | class | module | doc | ticket
    text: str
    sha: str = ""
    imports: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    licence: str | None = None
    repo_class: str = "internal"
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.repo}:{self.path}:{self.start}-{self.end}"

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)

    def citation(self) -> str:
        return f"{self.path}:{self.start}-{self.end}@{self.sha}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["id"] = self.id
        return d


# ------------------------------------------------------------------------------------------------ Python
def _called_names(node: ast.AST) -> list[str]:
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return sorted(names)


def chunk_python(doc: Document) -> list[Chunk]:
    try:
        tree = ast.parse(doc.text)
    except SyntaxError:
        return [_whole_file(doc)]
    lines = doc.text.splitlines()
    imports = [ast.get_source_segment(doc.text, n) or "" for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    out: list[Chunk] = []

    def used_imports(text: str) -> list[str]:
        res = []
        for imp in imports:
            names = re.findall(r"import\s+(.+)$", imp)
            tokens = [t.strip().split(" as ")[-1] for t in names[0].split(",")] if names else []
            if any(re.search(rf"\b{re.escape(t)}\b", text) for t in tokens if t):
                res.append(imp)
        return res

    def emit(node, symbol: str, kind: str, end: int | None = None, calls: list[str] | None = None):
        start = node.lineno - (len(node.decorator_list) if getattr(node, "decorator_list", None) else 0)
        end = end or node.end_lineno
        text = "\n".join(lines[start - 1 : end])
        out.append(Chunk(doc.repo or "", doc.path, start, end, symbol, kind, text, doc.sha, used_imports(text),
                         _called_names(node) if calls is None else calls))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            emit(node, node.name, "function")
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            header_end = (methods[0].lineno - 1 - len(methods[0].decorator_list)) if methods else node.end_lineno
            while header_end > node.lineno and not lines[header_end - 1].strip():
                header_end -= 1
            header_calls = sorted({n for stmt in node.body if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                                   for n in _called_names(stmt)})
            emit(node, node.name, "class", header_end, header_calls)   # methods carry their own call edges
            for m in methods:
                emit(m, f"{node.name}.{m.name}", "method")
    if not out:
        out.append(_whole_file(doc))
    return out


# ------------------------------------------------------------------------------------------------ other languages
_FN = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|func\s+(?:\([^)]*\)\s*)?(\w+)|(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\([^;]*\)\s*\{)")


def chunk_generic(doc: Document) -> list[Chunk]:
    ts = _chunk_tree_sitter(doc)
    if ts:
        return ts
    lines = doc.text.splitlines()
    starts = [(i, next(g for g in m.groups() if g)) for i, ln in enumerate(lines) if (m := _FN.match(ln))]
    if not starts:
        return [_whole_file(doc)]
    out = []
    for j, (i, name) in enumerate(starts):
        end = starts[j + 1][0] if j + 1 < len(starts) else len(lines)
        text = "\n".join(lines[i:end]).rstrip()
        out.append(Chunk(doc.repo or "", doc.path, i + 1, i + text.count("\n") + 1, name, "function", text, doc.sha))
    return out


def _chunk_tree_sitter(doc: Document) -> list[Chunk] | None:
    lang = {"ts": "typescript", "js": "javascript", "go": "go", "java": "java"}.get(doc.meta.get("language", ""))
    if not lang:
        return None
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore
    except ImportError:
        return None
    parser = get_parser(lang)
    tree = parser.parse(doc.text.encode())
    wanted = {"function_declaration", "method_declaration", "method_definition", "class_declaration"}
    out = []

    def walk(n):
        if n.type in wanted:
            name_node = n.child_by_field_name("name")
            name = name_node.text.decode() if name_node else n.type
            out.append(Chunk(doc.repo or "", doc.path, n.start_point[0] + 1, n.end_point[0] + 1, name,
                             "function", n.text.decode(), doc.sha))
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return out or None


def _whole_file(doc: Document) -> Chunk:
    n = doc.text.count("\n") + 1
    return Chunk(doc.repo or "", doc.path, 1, n, doc.path.rsplit("/", 1)[-1], "module", doc.text, doc.sha)


# ------------------------------------------------------------------------------------------------ docs
def chunk_markdown(doc: Document, max_tokens: int = 600, overlap: float = 0.15) -> list[Chunk]:
    """Split on headings; pack paragraphs into <= max_tokens; carry `overlap` of the previous chunk forward."""
    lines = doc.text.splitlines()
    sections: list[tuple[str, int, list[str]]] = []
    title, start, buf = doc.path, 1, []
    for i, ln in enumerate(lines, 1):
        m = re.match(r"^(#{1,6})\s+(.*)", ln)
        if m and buf:
            sections.append((title, start, buf))
            buf = []
        if m:
            title, start = m.group(2).strip(), i
        buf.append(ln)
    if buf:
        sections.append((title, start, buf))

    out: list[Chunk] = []
    ov_chars = int(max_tokens * overlap * 4)
    for title, start, body in sections:
        text = "\n".join(body).strip()
        if not text:
            continue
        if estimate_tokens(text) <= max_tokens:
            out.append(Chunk(doc.repo or "", doc.path, start, start + len(body) - 1, f"{doc.path}#{title}", "doc", text, doc.sha))
            continue
        step = max_tokens * 4 - ov_chars
        for k in range(0, len(text), step):
            piece = text[k : k + max_tokens * 4]
            out.append(Chunk(doc.repo or "", doc.path, start, start + len(body) - 1, f"{doc.path}#{title}", "doc",
                             (f"# {title}\n" if k else "") + piece, doc.sha, meta={"part": k // step}))
    return out


def chunk_document(doc: Document, **kw) -> list[Chunk]:
    if doc.kind in {"ticket", "log"}:
        n = doc.text.count("\n") + 1
        return [Chunk(doc.repo or "", doc.path, 1, n, doc.path, doc.kind, doc.text, doc.sha)]
    if doc.kind == "doc" or doc.path.endswith(".md"):
        return chunk_markdown(doc, **kw)
    if doc.path.endswith(".py"):
        return chunk_python(doc)
    return chunk_generic(doc)
