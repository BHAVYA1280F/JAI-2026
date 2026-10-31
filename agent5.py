"""
Agent 5: Safety Auditor (DermaGuard)

Reads the reports of A1-A4 and decides whether AUTO-TRIAGE is safe.
It NEVER looks at the image and NEVER sets the final decision; it only
vetoes automatic release and explains why. A6 (Coordinator) must honour it.

Usage (plain function call from the pipeline):

    from agent5 import safety_audit
    audit = safety_audit(a1, a2, a3, a4)      # a1..a4 = dicts from the agents
    audit.veto            # True  -> A6 must not AUTO-TRIAGE (-> REFER)
    audit.triggers        # ["MALIGNANT_TOP_CLASS", ...]
    audit.conflicts       # cross-agent disagreements
    audit.justification   # readable sentence for the traceable rationale
    audit.model_dump()    # JSON-ready dict

Optional FastAPI route:

    @app.post("/agent5")
    def agent5_route(payload: dict):
        return safety_audit(payload["a1"], payload["a2"],
                            payload.get("a3"), payload.get("a4")).model_dump()

Field names are matched through the aliases in the _g(...) calls below.
If your teammates use different names, add them there (one place).
Thresholds live in DEFAULT_CONFIG; tune_agent5.py writes a tuned
agent5_config.json that load_config() picks up.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

from pydantic import BaseModel, Field

CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
MALIGNANT = {"mel", "bcc", "akiec"}
# HAM10000 'localization' values treated as higher-risk sites
HIGH_RISK_SITES = {"face", "scalp", "ear", "neck", "acral", "foot", "hand", "genital"}
MISSING_TOKENS = {"", "unknown", "nan", "none", "null"}

DEFAULT_CONFIG: dict[str, float] = {
    "min_confidence": 0.80,     # A2 top-1 probability below this -> veto
    "min_margin": 0.30,         # top1 - top2 below this -> veto
    "max_entropy": 0.90,        # entropy (nats, max ln7 = 1.946) above this -> veto
    "min_tta_agreement": 0.75,  # A2 TTA agreement below this -> veto
    "max_p_malignant": 0.20,    # P(mel+bcc+akiec) above this -> veto
    "susp_conflict": 0.60,      # A3 suspicion at/above this vs benign label -> conflict
    "min_quality_score": 0.60,  # A1 score below this AND confidence < borderline_conf
    "borderline_conf": 0.90,
    "high_age": 60,             # age at/above this counts as higher risk
    "context_conf": 0.92,       # benign call on high-risk context needs >= this
    "missing_min": 2,           # this many missing context fields ...
    "missing_conf": 0.90,       # ... needs confidence >= this
}


def load_config(path: str = "agent5_config.json") -> dict[str, float]:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


class SafetyAudit(BaseModel):
    veto: bool = Field(description="True = block AUTO-TRIAGE (A6 should REFER)")
    veto_action: str = Field(description="'REFER' when veto, else 'NONE'")
    a1_failed: bool = Field(description="A1 failed quality; A6 should RE-IMAGE (A1 owns that)")
    conflicts: list[str] = []
    triggers: list[str] = []
    justification: str
    evidence: dict[str, Any] = {}


# ---------------------------------------------------------------- helpers
def _g(d: Optional[dict], *names: str, default=None):
    """First present, non-None key among aliases."""
    if not d:
        return default
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v).strip().lower() in MISSING_TOKENS


def _entropy(probs: list[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


def _truthy_reliable(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v) >= 0.5  # A3 reports reliability as a 0-1 number
    if isinstance(v, str):
        return v.strip().lower() not in {"low", "unreliable", "false", "no", "poor"}
    return bool(v)


# ------------------------------------------------------------- main audit
def safety_audit(
    a1: Optional[dict],
    a2: dict,
    a3: Optional[dict] = None,
    a4: Optional[dict] = None,
    cfg: Optional[dict] = None,
) -> SafetyAudit:
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    triggers: list[str] = []
    conflicts: list[str] = []
    reasons: list[str] = []

    def fire(tid: str, msg: str, conflict: bool = False):
        triggers.append(tid)
        reasons.append(msg)
        if conflict:
            conflicts.append(msg)

    # ---- A2: diagnosis signals
    probs_d = _g(a2, "probabilities", "probs") or {}
    probs = [float(probs_d.get(c, 0.0)) for c in CLASSES] if probs_d else []
    label = str(_g(a2, "label", "top_label", "pred_class", "predicted_label", default="")).lower()
    if probs and not label:
        label = CLASSES[max(range(7), key=lambda i: probs[i])]
    if probs:
        srt = sorted(probs, reverse=True)
        conf = srt[0]
        margin = srt[0] - srt[1]
        entropy = _entropy(probs)
        pmal = sum(float(probs_d.get(c, 0.0)) for c in MALIGNANT)
    else:  # fall back to whatever A2 reported
        conf = float(_g(a2, "confidence", default=0.0))
        margin = float(_g(a2, "margin", default=1.0))
        entropy = float(_g(a2, "entropy", default=0.0))
        pmal = float(_g(a2, "p_malignant", default=0.0))
    tta = _g(a2, "tta_agreement")
    is_mal = label in MALIGNANT

    # ---- A1: quality
    status = str(_g(a1, "severity", "status", "verdict", default="PASS")).upper()
    q_score = _g(a1, "quality_factor")           # A1 repo: 0-1
    if q_score is None:
        q_score = _g(a1, "usability_score", "score")
        if q_score is not None and float(q_score) > 1.0:
            q_score = float(q_score) / 100.0     # A1 'score' is 0-100
    a1_failed = status in {"FAIL", "FAILED", "POOR", "BAD", "REJECT", "REJECTED", "RE_IMAGE", "RE-IMAGE"}

    # ---- A3: lesion evidence
    susp = _g(a3, "suspicion", "suspicion_score")
    a3_reliable = _truthy_reliable(_g(a3, "reliable", "reliability"))

    # ---- A4: context
    used = _g(a4, "used", default={}) or {}      # A4 repo nests age/sex/location in "used"
    age = _g(a4, "age", default=used.get("age"))
    sex = _g(a4, "sex", "gender", default=used.get("sex"))
    site = _g(a4, "site", "localization", "location", default=used.get("location"))
    missing = _g(a4, "missing", "missing_fields")
    if missing is None:
        missing = [n for n, v in (("age", age), ("sex", sex), ("site", site)) if _is_missing(v)]
    missing = [m for m in missing if m != "location_unrecognised"]
    n_missing = len(missing)
    risk = _g(a4, "risk_level", "risk_indicator", "risk")
    if risk is not None:
        risk_high = str(risk).lower() in {"high", "elevated"}  # UNKNOWN/MEDIUM/LOW -> not high
    else:  # fallback computed from age / site
        age_high = (not _is_missing(age)) and float(age) >= cfg["high_age"]
        site_high = (not _is_missing(site)) and str(site).lower() in HIGH_RISK_SITES
        risk_high = age_high or site_high

    # ---- Rules
    if a1_failed:
        fire("A1_QUALITY_FAIL", f"A1 failed image quality ({status}); A6 should RE-IMAGE.")

    if is_mal:
        fire("MALIGNANT_TOP_CLASS", f"A2 predicts a malignant/pre-malignant class ({label}, {conf:.2f}).")
    if pmal > cfg["max_p_malignant"]:
        fire("HIGH_P_MALIGNANT", f"Combined malignant probability {pmal:.2f} exceeds {cfg['max_p_malignant']:.2f}.")

    if (not is_mal) and susp is not None and a3_reliable and float(susp) >= cfg["susp_conflict"]:
        fire("DIAG_LESION_CONFLICT",
             f"A2 predicts benign '{label}' ({conf:.2f}) but A3 lesion suspicion is {float(susp):.2f}.",
             conflict=True)
    if (not is_mal) and susp is not None and not a3_reliable:
        reasons.append("A3 flagged its own evidence as unreliable; suspicion score ignored.")

    if conf < cfg["min_confidence"]:
        fire("LOW_CONFIDENCE", f"A2 confidence {conf:.2f} is below {cfg['min_confidence']:.2f}.")
    if margin < cfg["min_margin"]:
        fire("SMALL_MARGIN", f"Top-2 probability margin {margin:.2f} is below {cfg['min_margin']:.2f}.")
    if entropy > cfg["max_entropy"]:
        fire("HIGH_ENTROPY", f"A2 probability entropy {entropy:.2f} exceeds {cfg['max_entropy']:.2f}.")
    if tta is not None and float(tta) < cfg["min_tta_agreement"]:
        fire("LOW_TTA_AGREEMENT", f"A2 TTA agreement {float(tta):.2f} is below {cfg['min_tta_agreement']:.2f}.")

    if (q_score is not None and float(q_score) < cfg["min_quality_score"]
            and conf < cfg["borderline_conf"]):
        fire("BORDERLINE_QUALITY_VS_CONFIDENCE",
             f"A1 quality score {float(q_score):.2f} is marginal and A2 confidence only {conf:.2f}.",
             conflict=True)

    if (not is_mal) and risk_high and conf < cfg["context_conf"]:
        fire("CONTEXT_RISK",
             f"Benign call on a higher-risk context (age={age}, site={site}) with confidence {conf:.2f}.",
             conflict=True)
    if n_missing >= cfg["missing_min"] and conf < cfg["missing_conf"]:
        fire("MISSING_CONTEXT",
             f"{n_missing} context fields missing ({', '.join(map(str, missing))}) and confidence only {conf:.2f}.")

    veto = len([t for t in triggers if t != "A1_QUALITY_FAIL"]) > 0 or a1_failed
    if veto:
        justification = " ".join(reasons) + " Automatic triage is unsafe."
    else:
        justification = (
            f"No safety concern: A2 predicts {label} with confidence {conf:.2f} "
            f"(margin {margin:.2f}, malignant prob {pmal:.2f}), no conflicts with A3/A4, quality acceptable."
        )

    return SafetyAudit(
        veto=veto,
        veto_action="REFER" if veto else "NONE",
        a1_failed=a1_failed,
        conflicts=conflicts,
        triggers=triggers,
        justification=justification,
        evidence={
            "label": label, "confidence": round(conf, 4), "margin": round(margin, 4),
            "entropy": round(entropy, 4), "p_malignant": round(pmal, 4), "tta_agreement": tta,
            "quality_score": q_score, "a3_suspicion": susp, "risk_high": risk_high,
            "missing_fields": list(missing),
        },
    )


# ------------------------------------------------------------------- demo
if __name__ == "__main__":
    def mk(label, conf):
        rest = (1 - conf) / 6
        return {"label": label, "probabilities": {c: (conf if c == label else rest) for c in CLASSES}}

    cases = {
        "clearly safe": (
            {"status": "PASS", "score": 0.95}, mk("nv", 0.96),
            {"suspicion": 0.12}, {"age": 25, "sex": "female", "site": "lower extremity"}),
        "conflict (nv vs suspicious)": (
            {"status": "PASS", "score": 0.91}, mk("nv", 0.85),
            {"suspicion": 0.81}, {"age": 68, "sex": "male", "site": "back"}),
        "malignant class": (
            {"status": "PASS", "score": 0.90}, mk("mel", 0.73),
            {"suspicion": 0.81}, {"age": 60, "sex": "male", "site": "back"}),
        "bad photo": (
            {"status": "FAIL", "score": 0.30, "reasons": ["blur"]}, mk("nv", 0.55), None, None),
    }
    for name, (a1, a2, a3, a4) in cases.items():
        r = safety_audit(a1, a2, a3, a4)
        print(f"\n[{name}] veto={r.veto} triggers={r.triggers}\n  {r.justification}")
