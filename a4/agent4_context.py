"""
Agent 4 - Reasoning / Context Agent
Uses age, sex and lesion location (HAM10000 metadata) to produce:
  context evidence, a risk indicator, and missing-field flags.
Missing metadata is never guessed: it lowers context_reliability and is flagged.
All priors are heuristic and meant to be tuned on the validation set.
"""
import math
import numpy as np

MALIGNANT = {"mel", "bcc", "akiec"}

# Rough site priors (0..1). 'unknown' is neutral on purpose.
SITE_RISK = {
    "face": 0.70, "scalp": 0.70, "ear": 0.70, "neck": 0.65,        # chronic sun exposure
    "back": 0.60, "trunk": 0.60,                                   # common melanoma sites
    "hand": 0.55, "foot": 0.60, "acral": 0.60, "genital": 0.60,    # often found late / acral melanoma
    "lower extremity": 0.50, "upper extremity": 0.45,
    "chest": 0.45, "abdomen": 0.40,
    "unknown": None,
}
AGE_X = [0, 20, 40, 60, 80]
AGE_Y = [0.00, 0.10, 0.35, 0.70, 1.00]

W_AGE, W_SEX, W_SITE = 0.60, 0.10, 0.30
LEVEL_MED, LEVEL_HIGH = 0.35, 0.60

_MISSING = {None, "", "nan", "none", "unknown", "n/a", "na"}


def _clean_age(age):
    try:
        a = float(age)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(a) or a < 0 or a > 120 else a


def _clean_str(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    return None if s in _MISSING else s


def context_agent(age=None, sex=None, location=None, predicted_class=None):
    """
    Returns the Agent-4 output dict.
    predicted_class (optional, from Agent 2, e.g. 'nv'/'mel') enables plausibility checks.
    """
    a, s, loc = _clean_age(age), _clean_str(sex), _clean_str(location)
    missing = [n for n, v in (("age", a), ("sex", s), ("location", loc)) if v is None]
    site_val = SITE_RISK.get(loc) if loc else None
    if loc and loc not in SITE_RISK:
        missing.append("location_unrecognised"); site_val = None

    evidence, parts = [], []   # parts: (weight, value)
    if a is not None:
        v = float(np.interp(a, AGE_X, AGE_Y))
        parts.append((W_AGE, v))
        evidence.append(f"age {int(a)}: {'higher' if v >= 0.6 else 'moderate' if v >= 0.3 else 'lower'} age-related risk")
    if s in ("male", "female"):
        parts.append((W_SEX, 0.55 if s == "male" else 0.45))
        if s == "male":
            evidence.append("male: slightly higher melanoma incidence")
    if site_val is not None:
        parts.append((W_SITE, site_val))
        evidence.append(f"site '{loc}': prior {site_val:.2f}")

    if parts:   # re-normalise over the fields we actually have
        tw = sum(w for w, _ in parts)
        risk = sum(w * v for w, v in parts) / tw
    else:
        risk = None

    level = "UNKNOWN" if risk is None else "HIGH" if risk >= LEVEL_HIGH else "MEDIUM" if risk >= LEVEL_MED else "LOW"
    context_rel = round(1 - len([m for m in missing if m != "location_unrecognised"]) / 3, 3)

    # plausibility of the Diagnosis Agent's label given the context
    flags = []
    pc = _clean_str(predicted_class)
    if pc and a is not None:
        if pc in ("bcc", "akiec") and a < 30:
            flags.append("atypical_for_age:" + pc)
        if pc == "mel" and a < 15:
            flags.append("atypical_for_age:mel")
        if pc == "nv" and a >= 70 and level in ("HIGH",):
            flags.append("benign_label_in_high_risk_context")
    if pc in MALIGNANT and level == "LOW":
        flags.append("malignant_label_in_low_risk_context")
    if pc == "nv" and level == "HIGH":
        flags.append("benign_label_in_high_risk_context") if "benign_label_in_high_risk_context" not in flags else None

    return {
        "agent": "A4",
        "risk_score": None if risk is None else round(float(risk), 3),
        "risk_level": level,
        "context_reliability": context_rel,
        "missing_fields": missing,
        "context_flags": flags,
        "evidence": evidence,
        "used": {"age": a, "sex": s, "location": loc},
    }
