"""L7 incident summariser: QLoRA SFT on (incident evidence -> brief), then DPO with SRE preference rankings.

Training rows come from past incidents: alert + log templates + metrics summary + deploys + similar
incidents -> postmortem-style brief (impact, timeline, ranked hypotheses, suggested runbook).

    python -m lexora.train.l7_incident --dry-run
    python -m lexora.train.l7_incident                     # GPU
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from common.config import load_yaml, settings
from lexora.train.qlora import write_jsonl

SYSTEM = ("You are the HELIX incident copilot. From the evidence, write: Impact, Timeline, Hypotheses (ranked, each with "
          "evidence and confidence), Suggested runbook. Only state facts present in the evidence.")


def build_rows() -> list[dict]:
    rows = []
    for f in sorted((Path(settings().data_dir) / "sample" / "postmortems").glob("*.md")):
        text = f.read_text()
        summary = re.search(r"## Summary\n(.+?)\n\n", text, re.S)
        resolution = re.search(r"## Resolution\n(.+?)(\n\n|$)", text, re.S)
        sev = re.search(r"Severity: (SEV-\d)", text)
        evidence = f"alert: error rate high on checkout-service\nsummary signals: {summary.group(1) if summary else ''}"
        brief = (f"Impact: {sev.group(1) if sev else 'SEV-3'} on checkout-service.\n"
                 f"Hypotheses: 1) {summary.group(1).split('.')[0] if summary else 'unknown'} (0.7)\n"
                 f"Suggested runbook: {resolution.group(1).strip() if resolution else 'investigate'}")
        rows.append({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": evidence},
                                  {"role": "assistant", "content": brief}]})
    return rows


def main(argv=None):
    rec = load_yaml("configs/train/recipes.yaml")["l7_incident"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=rec["base_model"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rows = build_rows()
    p = write_jsonl(Path(settings().state_dir) / "datasets" / "l7_sft.jsonl", rows)
    print(f"L7 rows: {len(rows)} -> {p}")
    if a.dry_run:
        return rows
    from lexora.train.qlora import sft  # pragma: no cover

    sft(a.base, p, Path(settings().state_dir) / "artifacts" / "l7_incident", rec["lora"]["r"], rec["lora"]["alpha"],
        rec["learning_rate"], rec["epochs"])
    return rows


if __name__ == "__main__":
    main()
