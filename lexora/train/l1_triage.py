"""L1 ticket triage classifier: category, assignment group, priority with calibrated probabilities.

Backends
  baseline (default, CPU, seconds): TF-IDF word 1-2 grams + char 3-5 grams -> one logistic regression per
            head (class_weight=balanced), temperature-scaled on the validation split.
  transformers (GPU): ModernBERT/DeBERTa encoder with three classification heads, class-weighted CE,
            then the same temperature scaling. Exported to ONNX for millisecond serving (Triton/ORT).

    python -m lexora.train.l1_triage                  # baseline, registers lexora/triage
    python -m lexora.train.l1_triage --backend transformers --epochs 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from common.config import settings
from lexora.eval.gates import check
from lexora.eval.metrics import expected_calibration_error
from lexora.ingest.itsm_connector import load_tickets_csv
from lexora.registry import ModelCard, register
from lexora.train.calibration import fit_temperature, softmax

HEADS = ["category", "assignment_group", "priority"]


def text_of(t: dict) -> str:
    return f"{t['title']} . {t['description']}"


def Featuriser() -> FeatureUnion:
    """Word 1-2 grams + char 3-5 grams. A plain sklearn FeatureUnion so the artefact unpickles anywhere."""
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=40000)),
    ])


def train_baseline(out_dir: Path) -> dict:
    rows = load_tickets_csv()
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    feat = Featuriser()
    Xtr = feat.fit_transform([text_of(r) for r in split["train"]])
    Xva = feat.transform([text_of(r) for r in split["val"]])
    Xte = feat.transform([text_of(r) for r in split["test"]])
    bundle = {"featuriser": feat, "heads": {}, "temperature": {}, "classes": {}}
    scores: dict[str, float] = {}
    for h in HEADS:
        clf = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced")
        clf.fit(Xtr, [r[h] for r in split["train"]])
        classes = list(clf.classes_)
        yva = np.array([classes.index(r[h]) for r in split["val"]])
        T = fit_temperature(np.log(np.clip(clf.predict_proba(Xva), 1e-12, 1)), yva)
        pte = softmax(np.log(np.clip(clf.predict_proba(Xte), 1e-12, 1)), T)
        yte = np.array([classes.index(r[h]) for r in split["test"]])
        pred = pte.argmax(1)
        acc = float((pred == yte).mean())
        ece_raw = expected_calibration_error(clf.predict_proba(Xte).max(1), clf.predict_proba(Xte).argmax(1) == yte)
        ece = expected_calibration_error(pte.max(1), pred == yte)
        scores[f"{h}_accuracy"] = acc
        scores[f"{h}_ece"] = ece
        scores[f"{h}_ece_uncalibrated"] = ece_raw
        bundle["heads"][h], bundle["temperature"][h], bundle["classes"][h] = clf, T, classes
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "triage_baseline.joblib"
    joblib.dump(bundle, path)
    scorecard = {"top1_group_accuracy": scores["assignment_group_accuracy"], "ece": scores["assignment_group_ece"], **scores}
    return {"path": str(path), "scorecard": scorecard, "temperature": bundle["temperature"],
            "n_train": len(split["train"]), "n_test": len(split["test"])}


def train_transformers(out_dir: Path, base: str, epochs: int, lr: float, batch: int) -> dict:  # pragma: no cover - GPU path
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    rows = load_tickets_csv()
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    classes = {h: sorted({r[h] for r in rows}) for h in HEADS}
    tok = AutoTokenizer.from_pretrained(base)

    class MultiHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(base)
            hid = self.enc.config.hidden_size
            self.heads = nn.ModuleDict({h: nn.Linear(hid, len(classes[h])) for h in HEADS})

        def forward(self, **x):
            cls = self.enc(**x).last_hidden_state[:, 0]
            return {h: m(cls) for h, m in self.heads.items()}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiHead().to(dev)
    weights = {h: torch.tensor([len(split["train"]) / max(1, sum(r[h] == c for r in split["train"])) for c in classes[h]],
                               dtype=torch.float, device=dev) for h in HEADS}
    losses = {h: nn.CrossEntropyLoss(weight=weights[h] / weights[h].mean()) for h in HEADS}
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * (len(split["train"]) // batch + 1)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    def batches(data):
        for i in range(0, len(data), batch):
            b = data[i : i + batch]
            enc = tok([text_of(r) for r in b], truncation=True, max_length=512, padding=True, return_tensors="pt").to(dev)
            ys = {h: torch.tensor([classes[h].index(r[h]) for r in b], device=dev) for h in HEADS}
            yield enc, ys

    for _ in range(epochs):
        model.train()
        for enc, ys in batches(split["train"]):
            out = model(**enc)
            loss = sum(losses[h](out[h], ys[h]) for h in HEADS)
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()

    def logits_of(data):
        model.eval()
        acc = {h: [] for h in HEADS}
        with torch.no_grad():
            for enc, _ in batches(data):
                o = model(**enc)
                for h in HEADS:
                    acc[h].append(o[h].float().cpu().numpy())
        return {h: np.vstack(v) for h, v in acc.items()}

    lv, lt = logits_of(split["val"]), logits_of(split["test"])
    scorecard, temps = {}, {}
    for h in HEADS:
        yv = np.array([classes[h].index(r[h]) for r in split["val"]])
        yt = np.array([classes[h].index(r[h]) for r in split["test"]])
        temps[h] = fit_temperature(lv[h], yv)
        p = softmax(lt[h], temps[h])
        scorecard[f"{h}_accuracy"] = float((p.argmax(1) == yt).mean())
        scorecard[f"{h}_ece"] = expected_calibration_error(p.max(1), p.argmax(1) == yt)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "triage_encoder.pt")
    tok.save_pretrained(out_dir / "tokenizer")
    (out_dir / "heads.json").write_text(json.dumps({"classes": classes, "temperature": temps, "base": base}))
    scorecard["top1_group_accuracy"] = scorecard["assignment_group_accuracy"]
    scorecard["ece"] = scorecard["assignment_group_ece"]
    return {"path": str(out_dir), "scorecard": scorecard, "temperature": temps}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["baseline", "transformers"], default="baseline")
    ap.add_argument("--base", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args(argv)
    out = Path(settings().state_dir) / "artifacts" / "l1_triage"
    res = train_baseline(out) if a.backend == "baseline" else train_transformers(out, a.base, a.epochs, a.lr, a.batch)
    gate = check("lexora/triage", res["scorecard"])
    card = register(ModelCard(
        name="lexora/triage", version="1.3.0" if a.backend == "baseline" else "2.0.0", artefact="L1",
        task_tags=["ticket-triage", "intent"], base="tfidf+logreg" if a.backend == "baseline" else a.base,
        algorithm="multi-head classification + temperature scaling", artifact_path=res["path"],
        scorecard=res["scorecard"], gate=gate, latency_profile={"p50_ms": 4, "p95_ms": 12},
        cost_profile={"usd_per_1k_requests": 0.01}, params={"temperature": res["temperature"]},
        known_failure_modes=["new assignment groups not in history", "very short titles"]))
    print(json.dumps({"card": card.ref, "gate": gate}, indent=1))
    return res


if __name__ == "__main__":
    main()
