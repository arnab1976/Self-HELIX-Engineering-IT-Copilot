"""Metrics, calibration, release gates and the trained CPU artefacts."""
import numpy as np

from lexora.eval.metrics import expected_calibration_error, mrr, ndcg_at_k, pass_at_k, recall_at_k, roc_auc
from lexora.registry import get_card
from lexora.train.calibration import fit_temperature, softmax
from lexora.train.qlora import lora_param_count


def test_pass_at_k_matches_combinatorial_formula():
    from math import comb

    for n, c, k in [(20, 6, 1), (20, 6, 5), (10, 0, 3), (5, 5, 2)]:
        exp = 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
        assert abs(pass_at_k(n, c, k) - exp) < 1e-9


def test_ranking_metrics():
    ranked = ["a", "x", "b", "y"]
    assert recall_at_k(ranked, {"a", "b", "c"}, 3) == 2 / 3
    assert mrr(["x", "a"], {"a"}) == 0.5
    assert abs(ndcg_at_k(["a", "b"], {"a": 1, "b": 1}) - 1.0) < 1e-9


def test_auc_and_ece():
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert expected_calibration_error(np.array([0.9] * 10), np.array([1] * 9 + [0])) < 1e-9


def test_temperature_preserves_argmax_and_reduces_overconfidence():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 4)) * 4
    y = np.where(rng.random(500) < 0.7, logits.argmax(1), rng.integers(0, 4, 500))
    T = fit_temperature(logits, y)
    assert T > 1.0
    assert (softmax(logits, T).argmax(1) == logits.argmax(1)).all()


def test_lora_parameter_count_for_8b():
    dims = {"q": (4096, 4096), "k": (4096, 1024), "v": (4096, 1024), "o": (4096, 4096),
            "gate": (4096, 14336), "up": (4096, 14336), "down": (14336, 4096)}
    assert lora_param_count(dims, 16, 32) == 41_943_040


def test_cpu_artefacts_registered_and_pass_gates():
    for name in ("lexora/triage", "lexora/duplicates", "lexora/change-risk"):
        card = get_card(name)
        assert card is not None and card.gate["passed"], name


def test_triage_predicts_calibrated_labels():
    from nexus.ml.triage import get_triage

    p = get_triage().predict("Need MFA reset after changing phone", "Okta Verify push never arrives")
    assert p["category"]["label"] == "Access > Password/MFA"
    assert abs(sum(x["p"] for x in p["category"]["top"]) - 1) < 0.2


def test_change_risk_contributions_add_up():
    from nexus.ml.change_risk import score

    r = score({"services_touched": 8, "lines_changed": 3000, "test_coverage": 20, "author_failure_rate": 25, "dependency_fanout": 25,
               "peak_window": 1, "has_rollback_plan": 0, "change_type": "emergency"})
    z = r["base_log_odds"] + sum(f["contribution"] for f in r["factors"])
    assert abs(1 / (1 + np.exp(-z)) - r["probability"]) < 0.01
    assert r["band"] == "high"
