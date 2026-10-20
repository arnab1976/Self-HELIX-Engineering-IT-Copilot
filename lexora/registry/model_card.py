"""Model cards (spec 4.5): task tags, scorecard, gate result, latency/cost, data classes, failure modes, rollback.

Stored as JSON under var/registry/<name>/<version>.json with a `production` pointer. When MLflow is
installed and MLFLOW_TRACKING_URI is set, the card is also logged as a run with the artefact attached.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common.config import settings


@dataclass
class ModelCard:
    name: str                                   # e.g. lexora/triage
    version: str
    artefact: str                               # L1..L9
    task_tags: list[str]
    base: str
    algorithm: str
    artifact_path: str
    scorecard: dict[str, float] = field(default_factory=dict)
    gate: dict = field(default_factory=dict)
    latency_profile: dict = field(default_factory=dict)
    cost_profile: dict = field(default_factory=dict)
    data_classes_permitted: list[str] = field(default_factory=lambda: ["public", "internal", "confidential", "restricted"])
    dataset_version: str = ""
    known_failure_modes: list[str] = field(default_factory=list)
    rollback_target: str | None = None
    params: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


def _root() -> Path:
    p = Path(settings().state_dir) / "registry"
    p.mkdir(parents=True, exist_ok=True)
    return p


def register(card: ModelCard, promote_if_passed: bool = True) -> ModelCard:
    d = _root() / card.name.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    prod = d / "production"
    if prod.exists():
        card.rollback_target = card.rollback_target or prod.read_text().strip()
    (d / f"{card.version}.json").write_text(json.dumps(asdict(card), indent=1), encoding="utf-8")
    if promote_if_passed and card.gate.get("passed", False):
        prod.write_text(card.version, encoding="utf-8")
    _maybe_mlflow(card)
    return card


def get_card(name: str, version: str | None = None) -> ModelCard | None:
    d = _root() / name.replace("/", "__")
    if version is None:
        if not (d / "production").exists():
            files = sorted(d.glob("*.json")) if d.exists() else []
            if not files:
                return None
            version = files[-1].stem
        else:
            version = (d / "production").read_text().strip()
    f = d / f"{version}.json"
    return ModelCard(**json.loads(f.read_text())) if f.exists() else None


def list_cards() -> list[dict]:
    out = []
    for d in sorted(_root().iterdir()):
        if d.is_dir():
            prod = (d / "production").read_text().strip() if (d / "production").exists() else None
            for f in sorted(d.glob("*.json")):
                c = json.loads(f.read_text())
                c["is_production"] = c["version"] == prod
                out.append(c)
    return out


def _maybe_mlflow(card: ModelCard) -> None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow  # type: ignore
    except ImportError:
        return
    mlflow.set_experiment(card.name)
    with mlflow.start_run(run_name=card.ref):
        mlflow.log_params({k: str(v) for k, v in card.params.items()})
        mlflow.log_metrics({k: float(v) for k, v in card.scorecard.items() if isinstance(v, (int, float))})
        mlflow.log_dict(asdict(card), "model_card.json")
        if Path(card.artifact_path).exists():
            mlflow.log_artifact(card.artifact_path)
