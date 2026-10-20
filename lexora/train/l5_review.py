"""L5 code chat & review adapter: QLoRA SFT on review threads, then DPO on accepted vs dismissed comments.

SFT rows   {"messages": [system, user(diff + context), assistant(JSON comments)]}
DPO rows   {"prompt", "chosen", "rejected"}  exported from NEXUS feedback (`/v1/feedback` on review comments)

    python -m lexora.train.l5_review --dry-run                  # build both datasets from local data/feedback
    python -m lexora.train.l5_review --stage sft                # GPU
    python -m lexora.train.l5_review --stage dpo --adapter var/artifacts/l5_review/sft/adapter
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import load_yaml, settings
from common.store import store
from lexora.train.qlora import write_jsonl

SYSTEM = ("You are the HELIX code reviewer. Return JSON {\"comments\": [{path, line, severity, category, message, suggestion}]}. "
          "Only comment on changed lines. Follow the engineering handbook.")


def sft_rows() -> list[dict]:
    diff = (Path(settings().data_dir) / "sample" / "pr" / "pr_4182.diff").read_text()
    from nexus.workflows.pr_review import rule_review  # reuse the validator-confirmed seed reviewer as a teacher

    comments = rule_review(diff)
    rows = [{"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": diff},
                          {"role": "assistant", "content": json.dumps({"comments": comments})}]}]
    # accepted comments captured in production become additional SFT targets
    for f in store().query("SELECT payload FROM feedback WHERE target='review_comment' AND signal='accept'"):
        p = json.loads(f["payload"])
        rows.append({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p.get("context", "")},
                                  {"role": "assistant", "content": json.dumps({"comments": [p.get("comment", {})]})}]})
    return rows


def dpo_rows() -> list[dict]:
    """Pair accepted and dismissed comments on the same PR: chosen = accepted, rejected = dismissed."""
    fb = store().query("SELECT signal, payload FROM feedback WHERE target='review_comment'")
    by_pr: dict[str, dict[str, list]] = {}
    for f in fb:
        p = json.loads(f["payload"])
        by_pr.setdefault(p.get("pr", "?"), {"accept": [], "dismiss": []})[f["signal"]].append(p)
    rows = []
    for pr, d in by_pr.items():
        for acc, rej in zip(d["accept"], d["dismiss"]):
            rows.append({"prompt": acc.get("context", f"PR {pr}"), "chosen": json.dumps(acc.get("comment")),
                         "rejected": json.dumps(rej.get("comment"))})
    return rows


def main(argv=None):
    rec = load_yaml("configs/train/recipes.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sft", "dpo"], default="sft")
    ap.add_argument("--base", default=rec["l5_sft"]["base_model"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    root = Path(settings().state_dir) / "datasets"
    sft_p = write_jsonl(root / "l5_sft.jsonl", sft_rows())
    dpo_r = dpo_rows()
    dpo_p = write_jsonl(root / "l5_dpo.jsonl", dpo_r) if dpo_r else None
    print(f"L5 SFT rows -> {sft_p}; DPO pairs: {len(dpo_r)}")
    if a.dry_run:
        return
    from lexora.train.qlora import dpo, sft  # pragma: no cover - GPU path

    out = Path(settings().state_dir) / "artifacts" / "l5_review"
    if a.stage == "sft":
        lc = rec["l5_sft"]["lora"]
        sft(a.base, sft_p, out / "sft", lc["r"], lc["alpha"], rec["l5_sft"]["learning_rate"], rec["l5_sft"]["epochs"])
    else:
        if not dpo_p:
            raise SystemExit("no DPO pairs yet: accept/dismiss review comments in the portal first")
        dpo(a.base, Path(a.adapter) if a.adapter else None, dpo_p, out / "dpo", rec["l5_dpo"]["beta"],
            rec["l5_dpo"]["learning_rate"], rec["l5_dpo"]["epochs"])


if __name__ == "__main__":
    main()
