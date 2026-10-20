"""pass@k harness for L4/L5: run n generated solutions per task against unit tests in a sandboxed subprocess.

Production runs each sample in gVisor/Firecracker with no network; the MVP uses a subprocess with a timeout
and an empty environment. Tasks: {"prompt", "tests"}; generate(prompt, n) -> list of completions.

    python -m lexora.eval.passk_harness --demo
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable

from lexora.eval.metrics import pass_at_k

DEMO_TASKS = [
    {"id": "luhn", "prompt": "def luhn_ok(number: str) -> bool:\n",
     "tests": "assert luhn_ok('4539578763621486')\nassert not luhn_ok('4539578763621487')\n"},
    {"id": "discount", "prompt": "def apply_discount(total: float, pct: float) -> float:\n",
     "tests": "assert apply_discount(100, 10) == 90.0\nassert apply_discount(80, 0) == 80.0\n"},
]
DEMO_SAMPLES = {
    "luhn": ["    d=[int(c) for c in number][::-1]\n    return (sum(d[0::2])+sum(sum(divmod(2*x,10)) for x in d[1::2]))%10==0\n",
             "    return sum(int(c) for c in number) % 10 == 0\n",
             "    d=[int(c) for c in number][::-1]\n    return (sum(d[0::2])+sum(sum(divmod(2*x,10)) for x in d[1::2]))%10==0\n",
             "    return True\n"],
    "discount": ["    return round(total * (1 - pct / 100), 2)\n", "    return total - pct\n",
                 "    return round(total * (1 - pct / 100), 2)\n", "    return total * pct\n"],
}


def run_sample(code: str, tests: str, timeout: float = 5.0) -> bool:
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py")
        with open(p, "w") as f:
            f.write(code + "\n" + tests)
        try:
            r = subprocess.run([sys.executable, "-I", p], capture_output=True, timeout=timeout, env={}, cwd=d)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def evaluate(tasks: list[dict], generate: Callable[[str, int], list[str]], n: int = 4, ks=(1, 2)) -> dict:
    per = []
    for t in tasks:
        samples = generate(t["prompt"], n)
        c = sum(run_sample(t["prompt"] + s, t["tests"]) for s in samples)
        per.append({"task": t["id"], "n": n, "c": c, **{f"pass@{k}": pass_at_k(n, c, k) for k in ks}})
    agg = {f"pass@{k}": sum(p[f"pass@{k}"] for p in per) / len(per) for k in ks}
    return {"tasks": per, **agg}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.parse_args()
    lookup = {t["prompt"]: t["id"] for t in DEMO_TASKS}
    print(json.dumps(evaluate(DEMO_TASKS, lambda prompt, n: DEMO_SAMPLES[lookup[prompt]][:n]), indent=1))
