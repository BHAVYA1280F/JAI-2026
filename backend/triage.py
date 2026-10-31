"""Triage Agent: combines the Image Quality and Diagnosis payloads into one decision.

Rules follow the DermaGuard draft, in this order:
  1. quality FAIL                                   -> RE_IMAGE (cites the defects)
  2. p_malignant > 0.20 or top class mel/bcc/akiec  -> REFER
  3. low confidence, high entropy or low TTA agreement -> REFER
  4. otherwise (benign, confident, certain, quality ok) -> AUTO_TRIAGE
Image quality modulates confidence: adjusted = confidence * quality_factor.
"""

MALIGNANT = ("mel", "bcc", "akiec")
P_MALIGNANT_REFER = 0.20          # from the DermaGuard draft rules

# DRAFT values: the spec gives no numbers. Tune them on outputs/val_logits.npy.
MIN_CONFIDENCE = 0.60
MAX_ENTROPY = 1.00
MIN_TTA_AGREEMENT = 0.75


def _p_malignant(d):
    if d.get("p_malignant") is not None:
        return float(d["p_malignant"])
    probs = {k.lower(): v for k, v in (d.get("probabilities") or {}).items()}
    return float(sum(probs.get(c, 0.0) for c in MALIGNANT))


def run_triage(quality, diagnosis):
    trace = [{"agent": "image_quality", "severity": quality["severity"], "score": quality["score"],
              "contribution": ("Quality " + quality["severity"] +
                               (": " + "; ".join(quality["reasons"]) if quality["reasons"] else ""))}]

    def done(decision, rationale):
        trace.append({"agent": "triage", "decision": decision, "contribution": rationale})
        return {"decision": decision, "rationale": rationale, "agent_trace": trace}

    # 1. quality gate
    if quality["severity"] == "FAIL":
        fails = [f"{d['type']} ({d['detail']})" for d in quality["defects"] if d["severity"] == "FAIL"]
        return done("RE_IMAGE", "Image quality is too low for a reliable result: " +
                    "; ".join(fails or quality["reasons"]) + ". Retake the photo.")

    if not diagnosis:
        return done("REFER", "No diagnosis is available, so the image cannot be auto-triaged.")

    label = str(diagnosis.get("label", "")).lower()
    conf = float(diagnosis.get("confidence", 0.0))
    adj = conf * quality["quality_factor"]
    p_mal = _p_malignant(diagnosis)
    trace.append({"agent": "diagnosis", "label": label, "confidence": conf,
                  "adjusted_confidence": round(adj, 4), "p_malignant": round(p_mal, 4),
                  "contribution": f"Top class {label} at {conf:.0%} confidence "
                                  f"({adj:.0%} after quality adjustment); p_malignant {p_mal:.0%}"})

    # 2. risk-asymmetric malignancy rule
    if p_mal > P_MALIGNANT_REFER or label in MALIGNANT:
        why = (f"top class is {label}" if label in MALIGNANT else "") + \
              (" and " if label in MALIGNANT and p_mal > P_MALIGNANT_REFER else "") + \
              (f"p_malignant is {p_mal:.0%} (above {P_MALIGNANT_REFER:.0%})" if p_mal > P_MALIGNANT_REFER else "")
        return done("REFER", f"Possible malignant or pre-malignant lesion: {why}.")

    # 3. uncertainty rule (a missing signal is skipped)
    problems = []
    if adj < MIN_CONFIDENCE:
        problems.append(f"quality-adjusted confidence {adj:.0%} is below {MIN_CONFIDENCE:.0%}")
    if diagnosis.get("entropy") is not None and diagnosis["entropy"] > MAX_ENTROPY:
        problems.append(f"entropy {diagnosis['entropy']:.2f} is above {MAX_ENTROPY:.2f}")
    if diagnosis.get("tta_agreement") is not None and diagnosis["tta_agreement"] < MIN_TTA_AGREEMENT:
        problems.append(f"TTA agreement {diagnosis['tta_agreement']:.0%} is below {MIN_TTA_AGREEMENT:.0%}")
    if problems:
        return done("REFER", "The prediction is not reliable enough to auto-triage: " + "; ".join(problems) + ".")

    note = " Quality has minor warnings." if quality["severity"] == "WARN" else ""
    return done("AUTO_TRIAGE", f"Benign class ({label}) with high confidence, low uncertainty and acceptable image quality.{note}")