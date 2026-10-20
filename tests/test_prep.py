"""LEXORA data hygiene: secrets, PII, licence, dedupe, chunking, Drain, FIM."""
import random

from lexora.ingest.base import Document
from lexora.prep import dedupe, fim, licence, pii, secrets
from lexora.prep.chunking import chunk_markdown, chunk_python
from lexora.prep.logs import DrainMiner, mine_templates


def test_secret_scan_and_redact():
    src = 'AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"\nDB_PASSWORD = "Summer2026!"\nREGION = "ap-south-1"'
    red, found = secrets.redact(src)
    assert {f.type for f in found} == {"AWS access key", "Password literal"}
    assert "<AWS_KEY>" in red and "<PASSWORD>" in red and "AKIA" not in red
    assert "ap-south-1" in red


def test_entropy_separates_keys_from_words():
    from common.textutils import entropy

    assert entropy("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY") > 4.0
    assert entropy("password") < 3.0


def test_pii_emails_get_stable_surrogates():
    a, _ = pii.anonymise("owner: priya.raman@acme-corp.com")
    b, _ = pii.anonymise("contact priya.raman@acme-corp.com today")
    tag = a.split("owner: ")[1]
    assert tag.startswith("<EMAIL:") and tag in b


def test_licence_copyleft_excluded_from_fim_only():
    r = licence.detect("# SPDX-License-Identifier: GPL-3.0-only\nx = 1", "vendor/x.py")
    assert r.copyleft and not r.allow_fim and r.allow_rag


def test_minhash_lsh_finds_near_duplicates():
    base = "def get_client():\n    return boto3.client('s3', region_name=REGION)\n" * 3 + "REGION = 'ap-south-1'\n"
    near = base.replace("ap-south-1", "ap-south-2")
    kept, dropped = dedupe.deduplicate([("a", base), ("b", near), ("c", "completely different text about cats and dogs")], threshold=0.8)
    assert kept == ["a", "c"] and "b" in dropped
    b, r = dedupe.lsh_params(128, 0.8)
    assert b * r == 128 and (1 / b) ** (1 / r) <= 0.8


def test_python_chunking_has_symbols_lines_and_calls():
    code = "import re\n\nclass V:\n    def check(self, x):\n        return re.match('a', x)\n\ndef helper():\n    return V().check('a')\n"
    chunks = chunk_python(Document("git", "v.py", code, "code", "r/x"))
    syms = {c.symbol: c for c in chunks}
    assert {"V", "V.check", "helper"} <= set(syms)
    assert syms["V.check"].start == 4 and syms["V.check"].end == 5
    assert "check" in syms["helper"].calls
    assert syms["V.check"].imports == ["import re"]


def test_markdown_chunks_by_heading():
    md = "# Title\nintro\n## A\ntext a\n## B\ntext b\n"
    ch = chunk_markdown(Document("kb", "k.md", md, "doc", "d"))
    assert [c.symbol for c in ch] == ["k.md#Title", "k.md#A", "k.md#B"]


def test_drain_groups_variable_lines():
    lines = [f"2026-10-07 10:42:{i:02d} ERROR checkout-svc E1043 payment token decode failed user={1000 + i}" for i in range(20)]
    lines += [f"2026-10-07 10:43:{i:02d} INFO request id={i} path=/pay" for i in range(10)]
    t = mine_templates(lines, prefer_drain3=False)
    assert len(t) == 2 and t[0]["count"] == 20 and "E1043" in t[0]["template"]
    m = DrainMiner()
    c1, ch1 = m.add("a b c 1")
    c2, ch2 = m.add("a b c 2")
    assert ch1 == "cluster_created" and c1 is c2


def test_fim_sample_shapes():
    rng = random.Random(0)
    modes = {fim.make_sample("def f(x):\n    return x + 1\n" * 3, rng, fim_rate=1.0).mode for _ in range(20)}
    assert modes <= {"psm", "spm"} and modes
    s = fim.make_sample("def f(x):\n    return x + 1\n" * 3, random.Random(1), fim_rate=1.0, spm_rate=0.0)
    assert s.text.startswith("<|fim_prefix|>") and "<|fim_middle|>" in s.text
