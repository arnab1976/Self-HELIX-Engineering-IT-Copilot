"""Dev-only API for LEXORA Agent 05 · L1 Triage + Calibration.

Labs: load/split tickets → train multi-head baseline → calibrate (T, ECE) →
thresholded route decision. Models stay in-process for the lab session.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import Counter
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from common.config import settings
from lexora.eval.metrics import expected_calibration_error
from lexora.train.calibration import fit_temperature, softmax

l1_lab_api = APIRouter()
log = logging.getLogger("nexus.api.l1_lab")

HEADS = ["category", "assignment_group", "priority"]
_MODELS: dict[str, dict[str, Any]] = {}

# Compact lab corpus so Agent 05 works without generated/tickets.csv.
_SAMPLE_TICKETS: list[dict] = [
    # Access > Password/MFA
    {"title": "Cannot reset MFA authenticator", "description": "Phone lost, need MFA reset for Okta login", "category": "Access > Password/MFA", "assignment_group": "IAM Service Desk", "priority": "P2"},
    {"title": "Locked out after password change", "description": "AD password changed, MFA prompts fail", "category": "Access > Password/MFA", "assignment_group": "IAM Service Desk", "priority": "P3"},
    {"title": "OTP codes not arriving", "description": "Authenticator app not generating codes for VPN", "category": "Access > Password/MFA", "assignment_group": "IAM Service Desk", "priority": "P2"},
    {"title": "Password expired and MFA stuck", "description": "Need temporary password and MFA re-enrolment", "category": "Access > Password/MFA", "assignment_group": "IAM Service Desk", "priority": "P3"},
    {"title": "Reset MFA for contractor account", "description": "Contractor phone wiped, cannot pass step-up MFA", "category": "Access > Password/MFA", "assignment_group": "IAM Service Desk", "priority": "P3"},
    {"title": "Forgot password after leave", "description": "Returning from PTO, password reset loop with MFA", "category": "Access > Password/MFA", "assignment_group": "IAM Service Desk", "priority": "P4"},
    # Access > VPN
    {"title": "GlobalProtect disconnects every hour", "description": "VPN drops on WiFi, reconnects with error 100", "category": "Access > VPN", "assignment_group": "Network Access", "priority": "P3"},
    {"title": "VPN cannot reach internal git", "description": "Connected but checkout-service repo unreachable", "category": "Access > VPN", "assignment_group": "Network Access", "priority": "P2"},
    {"title": "VPN portal certificate warning", "description": "Browser warns on portal.company.com VPN login", "category": "Access > VPN", "assignment_group": "Network Access", "priority": "P3"},
    {"title": "Split tunnel blocking Teams", "description": "When on VPN, Microsoft Teams call quality collapses", "category": "Access > VPN", "assignment_group": "Network Access", "priority": "P3"},
    {"title": "VPN login timeout overseas", "description": "Traveling APAC, GlobalProtect times out at auth", "category": "Access > VPN", "assignment_group": "Network Access", "priority": "P2"},
    {"title": "Need VPN for new hire laptop", "description": "Fresh Intune device cannot enrol VPN profile", "category": "Access > VPN", "assignment_group": "Network Access", "priority": "P4"},
    # Access > Request
    {"title": "Access to payments GitHub org", "description": "Need read access to payments/checkout-service", "category": "Access > Request", "assignment_group": "Access Governance", "priority": "P3"},
    {"title": "Confluence space permission", "description": "Request write to Engineering Handbook space", "category": "Access > Request", "assignment_group": "Access Governance", "priority": "P4"},
    {"title": "ServiceNow fulfill group add", "description": "Add me to IT Ops fulfillers for change tickets", "category": "Access > Request", "assignment_group": "Access Governance", "priority": "P3"},
    {"title": "SharePoint folder for HR project", "description": "Need folder ACL for compensation-portal docs", "category": "Access > Request", "assignment_group": "Access Governance", "priority": "P4"},
    {"title": "S3 bucket read for analytics", "description": "Request read-only role on analytics lake bucket", "category": "Access > Request", "assignment_group": "Access Governance", "priority": "P3"},
    {"title": "Jira project role change", "description": "Move from viewer to developer on HELIX board", "category": "Access > Request", "assignment_group": "Access Governance", "priority": "P4"},
    # Software > Install
    {"title": "Install Visual Studio Code", "description": "Need VS Code via Intune company portal", "category": "Software > Install", "assignment_group": "Endpoint Engineering", "priority": "P4"},
    {"title": "Docker Desktop blocked by policy", "description": "Cannot install Docker, WSL2 required for local lab", "category": "Software > Install", "assignment_group": "Endpoint Engineering", "priority": "P3"},
    {"title": "Python 3.12 missing on laptop", "description": "Company image has 3.10 only, need 3.12 for HELIX", "category": "Software > Install", "assignment_group": "Endpoint Engineering", "priority": "P3"},
    {"title": "Request JetBrains licence", "description": "Need PyCharm licence for backend work", "category": "Software > Install", "assignment_group": "Endpoint Engineering", "priority": "P4"},
    {"title": "Chrome extension allowlist", "description": "Need uBlock Origin approved for research", "category": "Software > Install", "assignment_group": "Endpoint Engineering", "priority": "P4"},
    {"title": "Install Node 20 LTS", "description": "Frontend build fails without Node 20", "category": "Software > Install", "assignment_group": "Endpoint Engineering", "priority": "P3"},
    # Hardware > Laptop
    {"title": "Laptop battery drains in 90 minutes", "description": "Dell Latitude battery health critical", "category": "Hardware > Laptop", "assignment_group": "Desktop Support", "priority": "P3"},
    {"title": "Keyboard keys sticking", "description": "Several keys stick after spill, need swap", "category": "Hardware > Laptop", "assignment_group": "Desktop Support", "priority": "P3"},
    {"title": "Broken hinge on lid", "description": "Laptop lid hinge cracked, screen wobbles", "category": "Hardware > Laptop", "assignment_group": "Desktop Support", "priority": "P3"},
    {"title": "Docking station not detected", "description": "USB-C dock no displays, power LED off", "category": "Hardware > Laptop", "assignment_group": "Desktop Support", "priority": "P4"},
    {"title": "Request loaner laptop", "description": "Primary device in repair, need loaner for sprint", "category": "Hardware > Laptop", "assignment_group": "Desktop Support", "priority": "P2"},
    {"title": "Trackpad cursor jumps", "description": "Cursor jumps while typing, drivers updated", "category": "Hardware > Laptop", "assignment_group": "Desktop Support", "priority": "P4"},
    # Email & Collaboration
    {"title": "Outlook cannot send email", "description": "Messages stuck in Outbox, O365 online works", "category": "Email & Collaboration", "assignment_group": "Messaging Ops", "priority": "P3"},
    {"title": "Teams meeting audio missing", "description": "Participants cannot hear me in Teams calls", "category": "Email & Collaboration", "assignment_group": "Messaging Ops", "priority": "P3"},
    {"title": "Shared mailbox permissions", "description": "Need send-as on finance@company.com", "category": "Email & Collaboration", "assignment_group": "Messaging Ops", "priority": "P4"},
    {"title": "OneDrive sync conflict loop", "description": "Desktop folder sync stuck at 99%", "category": "Email & Collaboration", "assignment_group": "Messaging Ops", "priority": "P4"},
    {"title": "Calendar invites delayed", "description": "Meeting invites arrive hours late", "category": "Email & Collaboration", "assignment_group": "Messaging Ops", "priority": "P3"},
    {"title": "Teams chat history missing", "description": "1:1 chat history blank after tenant move", "category": "Email & Collaboration", "assignment_group": "Messaging Ops", "priority": "P3"},
    # Application > Incident
    {"title": "Checkout API returning 500", "description": "Production payments checkout-service 500 spike", "category": "Application > Incident", "assignment_group": "App Support · Payments", "priority": "P1"},
    {"title": "Card vault token decode errors", "description": "Tokeniser failures in card-vault after deploy", "category": "Application > Incident", "assignment_group": "App Support · Payments", "priority": "P1"},
    {"title": "Latency on address validation", "description": "checkout address.py p95 over SLO", "category": "Application > Incident", "assignment_group": "App Support · Payments", "priority": "P2"},
    {"title": "Compensation portal blank page", "description": "HR compensation-portal white screen for managers", "category": "Application > Incident", "assignment_group": "App Support · HR", "priority": "P2"},
    {"title": "Auth middleware rejecting tokens", "description": "Intermittent 401 on API gateway after cert rotate", "category": "Application > Incident", "assignment_group": "App Support · Platform", "priority": "P1"},
    {"title": "Duplicate ticket creation storm", "description": "ITSM creating duplicate incidents for VPN outage", "category": "Application > Incident", "assignment_group": "App Support · Platform", "priority": "P2"},
]


def _require_dev() -> None:
    if not settings().is_dev:
        raise HTTPException(status_code=404, detail="Not found.")


def _text_of(t: dict) -> str:
    return f"{t.get('title', '')} . {t.get('description', '')}"


def _featuriser() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                    max_features=20000,
                ),
            ),
        ]
    )


def _load_csv_tickets(limit: int = 240) -> list[dict] | None:
    try:
        from lexora.ingest.itsm_connector import load_tickets_csv

        rows = load_tickets_csv()
        if not rows:
            return None
        by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": [], "other": []}
        for r in rows:
            if not all(r.get(h) for h in HEADS):
                continue
            item = {
                "title": r.get("title") or "",
                "description": r.get("description") or "",
                "category": r["category"],
                "assignment_group": r["assignment_group"],
                "priority": r["priority"],
                "split": r.get("split") or "",
            }
            sp = item["split"] if item["split"] in {"train", "val", "test"} else "other"
            by_split[sp].append(item)
        if by_split["train"] and by_split["val"] and by_split["test"]:
            n_tr = max(8, int(limit * 0.70))
            n_va = max(2, int(limit * 0.15))
            n_te = max(2, limit - n_tr - n_va)
            out = by_split["train"][:n_tr] + by_split["val"][:n_va] + by_split["test"][:n_te]
            return out or None
        # Flat list — re-split later
        flat = by_split["train"] + by_split["val"] + by_split["test"] + by_split["other"]
        return flat[:limit] or None
    except Exception as exc:  # noqa: BLE001 — lab falls back to embedded sample
        log.info("L1 lab CSV unavailable (%s); using embedded sample", exc)
        return None


def _assign_splits(rows: list[dict]) -> list[dict]:
    """Keep train/val/test when all three exist; otherwise assign 70/15/15."""
    counts = Counter(r.get("split") for r in rows)
    if counts.get("train") and counts.get("val") and counts.get("test"):
        return rows
    n = len(rows)
    out = []
    for i, r in enumerate(rows):
        if i < int(0.70 * n):
            split = "train"
        elif i < int(0.85 * n):
            split = "val"
        else:
            split = "test"
        out.append({**r, "split": split})
    return out


def _ece_bins(probs: np.ndarray, correct: np.ndarray, bins: int = 5) -> list[dict]:
    probs = np.asarray(probs, float)
    correct = np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (probs >= lo) & (probs < hi if b < bins - 1 else probs <= hi)
        n = int(mask.sum())
        if not n:
            rows.append({"bin": b + 1, "lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        rows.append(
            {
                "bin": b + 1,
                "lo": round(float(lo), 2),
                "hi": round(float(hi), 2),
                "n": n,
                "confidence": round(float(probs[mask].mean()), 3),
                "accuracy": round(float(correct[mask].mean()), 3),
            }
        )
    return rows


class L1LoadRequest(BaseModel):
    tickets: list[dict] = Field(default_factory=list, max_length=500)
    chunks: list[dict] = Field(default_factory=list, max_length=500)
    prefer_csv: bool = False
    allow_lab_fallback: bool = False
    require_journey: bool = True
    dataset_version: str = Field(default="", max_length=120)
    primary_source: str = Field(default="", max_length=64)
    source_label: str = Field(default="", max_length=120)
    adapt_choice: str = Field(default="", max_length=32)
    collection_preview: str = Field(default="", max_length=8_000)
    # Full Agent 01 multi-doc handoff text (===== path ===== sections)
    handoff_text: str = Field(default="", max_length=150_000)
    # Manual portal Path A may have a small labelled set (demo); allow below the usual 12-row floor
    allow_small_corpus: bool = False


class L1TrainRequest(BaseModel):
    tickets: list[dict] = Field(..., min_length=12, max_length=500)
    dataset_version: str = Field(default="", max_length=120)


class L1CalibrateRequest(BaseModel):
    model_id: str = Field(..., min_length=4, max_length=80)


class L1PredictRequest(BaseModel):
    model_id: str = Field(..., min_length=4, max_length=80)
    title: str = Field(..., min_length=1, max_length=400)
    description: str = Field(default="", max_length=4000)
    threshold: float = Field(default=0.85, ge=0.05, le=0.99)
    head: str = Field(default="assignment_group", max_length=40)


def _clean_ticket_row(r: dict) -> dict | None:
    if not all(r.get(h) for h in HEADS):
        return None
    if not (r.get("title") or r.get("description")):
        return None
    return {
        "title": str(r.get("title") or "")[:400],
        "description": str(r.get("description") or "")[:4000],
        "category": str(r.get("category") or ""),
        "assignment_group": str(r.get("assignment_group") or ""),
        "priority": str(r.get("priority") or ""),
        "split": str(r.get("split") or ""),
    }


_KV = re.compile(r"(?im)^([a-z_][a-z0-9_]*)\s*[:=]\s*(.+?)\s*$")
_SECTION = re.compile(r"(?m)^===== (.+?) =====\s*$")


def _parse_kv_block(block: str) -> dict:
    fields: dict[str, str] = {}
    for match in _KV.finditer(block or ""):
        fields[match.group(1).lower()] = match.group(2).strip()
    # Common ITSM aliases from ServiceNow-style exports
    if "short_description" in fields and "title" not in fields:
        fields["title"] = fields["short_description"]
    if "number" in fields and "id" not in fields:
        fields["id"] = fields["number"]
    return fields


def _tickets_from_handoff_text(text: str) -> list[dict]:
    """Parse Agent 01 ticket handoff sections into labelled L1 rows (not tickets.csv)."""
    if not (text or "").strip():
        return []
    matches = list(_SECTION.finditer(text))
    blocks: list[str] = []
    if matches:
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            blocks.append(text[start:end])
    else:
        # Fallback: split on blank lines if no section markers
        blocks = re.split(r"\n\s*\n", text)

    out: list[dict] = []
    seen: set[str] = set()
    for block in blocks:
        fields = _parse_kv_block(block)
        row = {
            "title": fields.get("title") or "",
            "description": fields.get("description") or "",
            "category": fields.get("category") or "",
            "assignment_group": fields.get("assignment_group") or "",
            "priority": fields.get("priority") or "",
            "split": fields.get("split") or "",
        }
        cleaned = _clean_ticket_row(row)
        if not cleaned:
            continue
        key = f"{cleaned['title']}|{cleaned['category']}|{cleaned['assignment_group']}|{cleaned['priority']}"
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _tickets_from_chunks(chunks: list[dict]) -> list[dict]:
    """Promote labelled ticket-like chunks from the Step 02 handoff into L1 rows."""
    out: list[dict] = []
    seen: set[str] = set()
    for c in chunks:
        meta = c.get("meta") if isinstance(c.get("meta"), dict) else {}
        fields = _parse_kv_block(str(c.get("text") or ""))
        row = {
            "title": c.get("title") or meta.get("title") or fields.get("title") or c.get("symbol") or "",
            "description": c.get("description") or meta.get("description") or fields.get("description") or "",
            "category": c.get("category") or meta.get("category") or fields.get("category") or "",
            "assignment_group": c.get("assignment_group")
            or meta.get("assignment_group")
            or fields.get("assignment_group")
            or "",
            "priority": c.get("priority") or meta.get("priority") or fields.get("priority") or "",
            "split": c.get("split") or meta.get("split") or fields.get("split") or "",
        }
        # If description empty but body has prose, keep a short slice as description aid
        if not row["description"] and c.get("text") and not fields:
            row["description"] = str(c.get("text") or "")[:500]
        cleaned = _clean_ticket_row(row)
        if not cleaned:
            continue
        key = f"{cleaned['title']}|{cleaned['category']}|{cleaned['assignment_group']}|{cleaned['priority']}"
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


@l1_lab_api.post("/lexora/l1/load", tags=["LEXORA L1 lab"])
def l1_load(request: L1LoadRequest) -> dict:
    """Load L1 training tickets strictly from the Agent 01→02 journey handoff.

    Prefer parsing the multi-doc tickets handoff text. tickets.csv is only used when
    the caller explicitly opts into lab fallback (dev Unlock).
    """
    _require_dev()
    t0 = time.perf_counter()
    primary = (request.primary_source or "").lower().strip()
    handoff_text = (request.handoff_text or request.collection_preview or "").strip()
    has_journey = bool(
        request.dataset_version or request.chunks or handoff_text or primary
    )

    if request.require_journey and not has_journey and not request.allow_lab_fallback:
        raise HTTPException(
            400,
            "No Agent 01/02 handoff. Run Collect -> Chunk -> Adapt (fine-tune), then Load again. "
            "L1 will not invent a disconnected tickets.csv corpus.",
        )
    if request.require_journey and primary and primary != "tickets" and not request.allow_lab_fallback:
        raise HTTPException(
            400,
            f"Agent 05 L1 triage needs the tickets source from Agent 01. "
            f"Current journey primary_source is '{primary or 'unknown'}'. "
            "Re-run Collect with tickets, or Unlock and enable lab fallback for development.",
        )

    lineage = {
        "agent_01_dataset": request.dataset_version or "",
        "primary_source": primary or "",
        "source_label": request.source_label or "",
        "adapt_choice": request.adapt_choice or "",
        "chunk_count_in": len(request.chunks),
        "handoff_chars": len(handoff_text),
    }

    source = "handoff"
    rows: list[dict] = []

    # 1) Agent 01 multi-doc tickets handoff — authoritative journey corpus (never silent CSV)
    if handoff_text:
        from_text = _tickets_from_handoff_text(handoff_text)
        min_handoff = 1 if request.allow_small_corpus else 12
        if len(from_text) >= min_handoff:
            rows = from_text
            source = "journey_handoff"

    # 2) Explicit labelled tickets in the request (rare / tests)
    if len(rows) < 12:
        for r in request.tickets:
            cleaned = _clean_ticket_row(r)
            if cleaned:
                rows.append(cleaned)
        if len(rows) >= 12:
            source = "request_tickets"

    # 3) Step 02 chunks that still carry labelled ticket fields
    if len(rows) < 12 and request.chunks:
        from_chunks = _tickets_from_chunks(request.chunks)
        if from_chunks:
            rows = from_chunks
            source = "journey_chunks"

    # 4) Opt-in lab fallback only (dev Unlock) — never silent tickets.csv on the journey path
    if len(rows) < 12 and request.allow_lab_fallback:
        csv_rows = _load_csv_tickets()
        if csv_rows and len(csv_rows) >= 12:
            rows = csv_rows
            source = "lab_fallback_csv"
        else:
            rows = [dict(r) for r in _SAMPLE_TICKETS]
            source = "lab_fallback_embedded"

    cleaned = [item for r in rows if (item := _clean_ticket_row(r))]

    min_rows = 1 if request.allow_small_corpus else 12
    if len(cleaned) < min_rows:
        raise HTTPException(
            400,
            "Not enough labelled tickets in the Agent 01 handoff for L1 "
            f"(need ≥{min_rows}, got {len(cleaned)}). Re-run Collect with tickets so the handoff "
            "includes category / assignment_group / priority fields, or use Lab fallback after Unlock.",
        )

    demo_replicated = 0
    if request.allow_small_corpus and 0 < len(cleaned) < 12:
        # Path A demo: expand a small manual portal set so train/val/test splits exist
        seed = [dict(r) for r in cleaned]
        while len(cleaned) < 12:
            clone = dict(seed[len(cleaned) % len(seed)])
            clone["split"] = ""
            cleaned.append(clone)
            demo_replicated += 1

    cleaned = _assign_splits(cleaned)
    split_counts = Counter(r["split"] for r in cleaned)
    label_counts = {h: dict(Counter(r[h] for r in cleaned).most_common(8)) for h in HEADS}
    ds = request.dataset_version or f"l1-{source}"
    base_note = (
        f"Parsed labelled ticket(s) from Agent 01 handoff text (dataset {ds})."
        if source == "journey_handoff"
        else "Rows taken from Step 02 labelled chunks."
        if source == "journey_chunks"
        else "Lab fallback corpus (not journey-linked)."
        if source.startswith("lab_fallback")
        else "Rows supplied in the handoff payload."
    )
    if demo_replicated:
        base_note += (
            f" Demo Path A expanded +{demo_replicated} replicate(s) so train/val/test splits work "
            "(replace with more real portal tickets for production)."
        )
    return {
        "dataset_version": ds,
        "source": source,
        "lineage": lineage,
        "ticket_count": len(cleaned),
        "split_counts": dict(split_counts),
        "label_counts": label_counts,
        "heads": HEADS,
        "tickets": cleaned,
        "preview": cleaned[:3],
        "demo_replicated": demo_replicated,
        "seconds": round(time.perf_counter() - t0, 3),
        "next": "Train the multi-head baseline on the train split (Lab B).",
        "note": base_note,
    }


@l1_lab_api.post("/lexora/l1/train", tags=["LEXORA L1 lab"])
def l1_train(request: L1TrainRequest) -> dict:
    _require_dev()
    t0 = time.perf_counter()
    rows = _assign_splits(
        [
            {
                "title": str(r.get("title") or "")[:400],
                "description": str(r.get("description") or "")[:4000],
                "category": str(r.get("category") or ""),
                "assignment_group": str(r.get("assignment_group") or ""),
                "priority": str(r.get("priority") or ""),
                "split": str(r.get("split") or ""),
            }
            for r in request.tickets
            if all(r.get(h) for h in HEADS)
        ]
    )
    if len(rows) < 12:
        raise HTTPException(400, "Need at least 12 labelled tickets across category / assignment_group / priority.")

    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    if len(split["train"]) < 6:
        raise HTTPException(400, "Train split too small; reload the lab dataset.")
    if not split["val"]:
        # Steal a few from train for calibration
        split["val"] = split["train"][-max(2, len(split["train"]) // 5) :]
    if not split["test"]:
        split["test"] = split["val"]

    feat = _featuriser()
    Xtr = feat.fit_transform([_text_of(r) for r in split["train"]])
    Xva = feat.transform([_text_of(r) for r in split["val"]])
    Xte = feat.transform([_text_of(r) for r in split["test"]])

    bundle: dict[str, Any] = {"featuriser": feat, "heads": {}, "temperature": {}, "classes": {}}
    scores: dict[str, float] = {}
    for h in HEADS:
        y_train = [r[h] for r in split["train"]]
        if len(set(y_train)) < 2:
            raise HTTPException(400, f"Head '{h}' needs at least two classes in the train split.")
        clf = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")
        clf.fit(Xtr, y_train)
        classes = list(clf.classes_)
        yva = np.array([classes.index(r[h]) if r[h] in classes else 0 for r in split["val"]])
        raw_va = np.clip(clf.predict_proba(Xva), 1e-12, 1)
        T = fit_temperature(np.log(raw_va), yva)
        raw_te = np.clip(clf.predict_proba(Xte), 1e-12, 1)
        p_te = softmax(np.log(raw_te), T)
        yte = np.array([classes.index(r[h]) if r[h] in classes else 0 for r in split["test"]])
        pred = p_te.argmax(1)
        acc = float((pred == yte).mean()) if len(yte) else 0.0
        ece_raw = float(expected_calibration_error(raw_te.max(1), raw_te.argmax(1) == yte)) if len(yte) else 0.0
        ece = float(expected_calibration_error(p_te.max(1), pred == yte)) if len(yte) else 0.0
        scores[f"{h}_accuracy"] = round(acc, 4)
        scores[f"{h}_ece"] = round(ece, 4)
        scores[f"{h}_ece_uncalibrated"] = round(ece_raw, 4)
        bundle["heads"][h], bundle["temperature"][h], bundle["classes"][h] = clf, T, classes

    model_id = "l1-" + hashlib.sha1(
        (request.dataset_version + "|" + str(len(rows)) + "|" + str(time.time())).encode()
    ).hexdigest()[:10]
    _MODELS[model_id] = {
        "bundle": bundle,
        "scores": scores,
        "split_counts": {k: len(v) for k, v in split.items()},
        "dataset_version": request.dataset_version or "l1-lab",
        "n_tickets": len(rows),
        "Xte": Xte,
        "split_test": split["test"],
    }

    return {
        "model_id": model_id,
        "dataset_version": request.dataset_version or "l1-lab",
        "backend": "tfidf+logreg",
        "heads": HEADS,
        "split_counts": {k: len(v) for k, v in split.items()},
        "scorecard": {
            "top1_group_accuracy": scores["assignment_group_accuracy"],
            "ece": scores["assignment_group_ece"],
            **scores,
        },
        "temperature": {h: round(float(bundle["temperature"][h]), 3) for h in HEADS},
        "class_counts": {h: len(bundle["classes"][h]) for h in HEADS},
        "seconds": round(time.perf_counter() - t0, 3),
        "next": "Inspect calibration (Lab C): compare raw vs temperature-scaled ECE.",
    }


@l1_lab_api.post("/lexora/l1/calibrate", tags=["LEXORA L1 lab"])
def l1_calibrate(request: L1CalibrateRequest) -> dict:
    _require_dev()
    t0 = time.perf_counter()
    stored = _MODELS.get(request.model_id)
    if not stored:
        raise HTTPException(404, "Unknown model_id — run Lab B train first in this API process.")

    bundle = stored["bundle"]
    Xte = stored["Xte"]
    test_rows = stored["split_test"]
    head = "assignment_group"
    clf, T, classes = bundle["heads"][head], bundle["temperature"][head], bundle["classes"][head]
    raw = np.clip(clf.predict_proba(Xte), 1e-12, 1)
    cal = softmax(np.log(raw), T)
    yte = np.array([classes.index(r[head]) if r[head] in classes else 0 for r in test_rows])
    raw_pred = raw.argmax(1)
    cal_pred = cal.argmax(1)
    # Temperature never changes argmax
    argmax_unchanged = bool(np.all(raw_pred == cal_pred))
    reliability = _ece_bins(cal.max(1), cal_pred == yte, bins=5)
    reliability_raw = _ece_bins(raw.max(1), raw_pred == yte, bins=5)

    return {
        "model_id": request.model_id,
        "head": head,
        "temperature": {h: round(float(bundle["temperature"][h]), 3) for h in HEADS},
        "argmax_unchanged": argmax_unchanged,
        "ece_uncalibrated": stored["scores"][f"{head}_ece_uncalibrated"],
        "ece_calibrated": stored["scores"][f"{head}_ece"],
        "accuracy": stored["scores"][f"{head}_accuracy"],
        "reliability_raw": reliability_raw,
        "reliability_calibrated": reliability,
        "scorecard": stored["scores"],
        "seconds": round(time.perf_counter() - t0, 3),
        "next": "Set an auto-route threshold and predict a live ticket (Lab D).",
    }


@l1_lab_api.post("/lexora/l1/predict", tags=["LEXORA L1 lab"])
def l1_predict(request: L1PredictRequest) -> dict:
    _require_dev()
    t0 = time.perf_counter()
    stored = _MODELS.get(request.model_id)
    if not stored:
        raise HTTPException(404, "Unknown model_id — run Lab B train first in this API process.")

    bundle = stored["bundle"]
    head = request.head if request.head in HEADS else "assignment_group"
    X = bundle["featuriser"].transform([f"{request.title} . {request.description}"])
    prediction: dict[str, Any] = {}
    for h in HEADS:
        clf, T, classes = bundle["heads"][h], bundle["temperature"][h], bundle["classes"][h]
        raw = clf.predict_proba(X)[0]
        p = softmax(np.log(np.clip(raw, 1e-12, 1))[None, :], T)[0]
        order = np.argsort(-p)[:3]
        prediction[h] = {
            "label": str(classes[order[0]]),
            "p": round(float(p[order[0]]), 3),
            "p_uncalibrated": round(float(raw[order[0]]), 3),
            "temperature": round(float(T), 3),
            "top": [{"label": str(classes[i]), "p": round(float(p[i]), 3)} for i in order],
        }

    conf = float(prediction[head]["p"])
    decision = "auto_route" if conf >= request.threshold else "agent_queue"
    return {
        "model_id": request.model_id,
        "dataset_version": stored["dataset_version"],
        "title": request.title,
        "description": request.description,
        "threshold": request.threshold,
        "decision_head": head,
        "confidence": conf,
        "decision": decision,
        "decision_label": "Auto-route" if decision == "auto_route" else "Agent queue",
        "prediction": prediction,
        "scorecard": stored["scores"],
        "temperature": {h: round(float(bundle["temperature"][h]), 3) for h in HEADS},
        "seconds": round(time.perf_counter() - t0, 3),
        "next": (
            "Confidence meets threshold — safe to auto-route in the triage workflow."
            if decision == "auto_route"
            else "Below threshold — keep a human in the loop (agent queue)."
        ),
        "handoff": {
            "model_id": request.model_id,
            "artefact": "L1",
            "name": "lexora/triage",
            "dataset_version": stored["dataset_version"],
            "threshold": request.threshold,
            "ece": stored["scores"].get("assignment_group_ece"),
            "accuracy": stored["scores"].get("assignment_group_accuracy"),
            "temperature": {h: round(float(bundle["temperature"][h]), 3) for h in HEADS},
            "sample_decision": decision,
            "sample_confidence": conf,
        },
    }
