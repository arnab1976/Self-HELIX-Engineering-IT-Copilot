"""Log template mining with the Drain algorithm.

Uses the `drain3` library when installed (streaming, persistence, masking config in configs/drain3.ini);
`DrainMiner` below is a compact implementation of the same fixed-depth parse tree for teaching and tests.

  mask variables -> route by token count -> route by leading tokens (depth) -> leaf: best template by
  positional token similarity -> merge if sim >= st (differing positions become <*>) else create.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MASKS = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"), ""),        # drop timestamps
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<UUID>"),
    (re.compile(r"(?<==)[^\s,]+"), "<*>"),                                          # key=value values
    (re.compile(r"(?<![\w.])\d+(?![\w.])"), "<NUM>"),
]


def mask(line: str) -> list[str]:
    for rx, rep in MASKS:
        line = rx.sub(rep, line)
    return line.split()


@dataclass
class Cluster:
    id: int
    template: list[str]
    size: int = 0
    lines: list[int] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.template)


class DrainMiner:
    def __init__(self, sim_th: float = 0.5, depth: int = 4, max_children: int = 100):
        self.sim_th, self.depth, self.max_children = sim_th, depth, max_children
        self.root: dict = {}
        self.clusters: list[Cluster] = []

    @staticmethod
    def _sim(tpl: list[str], toks: list[str]) -> float:
        same = sum(1 for a, b in zip(tpl, toks) if a == b or a == "<*>")
        return same / max(1, len(toks))

    def add(self, line: str, line_no: int = 0) -> tuple[Cluster, str]:
        toks = mask(line)
        node = self.root.setdefault(len(toks), {})
        for t in toks[: self.depth - 2]:
            key = "<*>" if any(c.isdigit() for c in t) or t.startswith("<") else t
            if key not in node and len(node) >= self.max_children:
                key = "<*>"
            node = node.setdefault(key, {})
        leaf: list[Cluster] = node.setdefault("__clusters__", [])
        best, best_sim = None, -1.0
        for c in leaf:
            s = self._sim(c.template, toks)
            if s > best_sim:
                best, best_sim = c, s
        if best is not None and best_sim >= self.sim_th:
            new_tpl = [a if a == b else "<*>" for a, b in zip(best.template, toks)]
            change = "template_changed" if new_tpl != best.template else "none"
            best.template = new_tpl
            best.size += 1
            best.lines.append(line_no)
            return best, change
        c = Cluster(len(self.clusters) + 1, toks, 1, [line_no])
        self.clusters.append(c)
        leaf.append(c)
        return c, "cluster_created"


def mine_templates(lines: list[str], sim_th: float = 0.5, prefer_drain3: bool = True) -> list[dict]:
    """Returns [{id, template, count, first_line}] sorted by count desc."""
    if prefer_drain3:
        try:
            from drain3 import TemplateMiner  # type: ignore
            from drain3.template_miner_config import TemplateMinerConfig  # type: ignore

            from common.config import ROOT

            conf = TemplateMinerConfig()
            conf.load(str(ROOT / "configs" / "drain3.ini"))
            conf.drain_sim_th = sim_th
            conf.profiling_enabled = False
            tm = TemplateMiner(config=conf)
            first: dict[int, int] = {}
            for i, ln in enumerate(lines):
                body = MASKS[0][0].sub("", ln).strip()
                r = tm.add_log_message(body)
                first.setdefault(r["cluster_id"], i)
            res = [{"id": c.cluster_id, "template": c.get_template(), "count": c.size, "first_line": first.get(c.cluster_id, 0)}
                   for c in tm.drain.clusters]
            return sorted(res, key=lambda r: -r["count"])
        except ImportError:
            pass
    miner = DrainMiner(sim_th)
    for i, ln in enumerate(lines):
        miner.add(ln, i)
    return sorted([{"id": c.id, "template": c.text, "count": c.size, "first_line": c.lines[0]} for c in miner.clusters],
                  key=lambda r: -r["count"])


def new_templates(baseline: list[str], window: list[str], sim_th: float = 0.5) -> list[dict]:
    """Templates present in `window` but not in `baseline` - a strong root-cause signal after a deploy."""
    base = {t["template"] for t in mine_templates(baseline, sim_th, prefer_drain3=False)}
    return [t for t in mine_templates(window, sim_th, prefer_drain3=False) if t["template"] not in base]
