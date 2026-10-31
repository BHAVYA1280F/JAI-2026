"""
Agent 6 - Coordinator Agent  (Agentic DermaGuard, Track 2 Healthcare)

Receives the outputs of Agents 1-5, applies a documented decision policy and
returns: final predicted label, triage decision (AUTO_TRIAGE / REFER / RE_IMAGE)
and a traceable rationale (ordered list of rules that fired).

Integration (3 lines):

    from coordinator_agent import CoordinatorAgent
    coord = CoordinatorAgent()
    result = coord.decide(a1=a1_out, a2=a2_out, a3=a3_out, a4=a4_out, a5=a5_out)
    # result.final_label, result.triage, result.rationale, result.to_dict()

Each a*_out may be a dict, dataclass, or any object with attributes. Common key
aliases are accepted (see _pick), so you normally do not need to change your
earlier agents. If your keys differ, add aliases in the ALIASES table below.

Decision policy (evaluated in this order, first hit wins for the triage):
  R1  Image quality FAIL / score < quality_fail            -> RE_IMAGE
  R2  Safety agent hard veto asking for re-image           -> RE_IMAGE
  R3  Safety agent flag / veto                             -> REFER
  R4  Predicted class is malignant/pre-malignant (mel,bcc,akiec)
      OR P(malignant classes) >= malignant_prob_refer      -> REFER
  R5  Low confidence / small top-2 margin                  -> REFER
      (the confidence bar is raised when image quality is marginal)
  R6  Lesion Analysis suspicion >= suspicion_refer         -> REFER
  R7  Context risk HIGH                                    -> REFER
  R9  Otherwise (benign class, confident, consistent)      -> AUTO_TRIAGE
      (rule number R8 is intentionally unused)
Every image always receives a final label and a triage decision.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CLASSES: Tuple[str, ...] = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
MALIGNANT = frozenset({"mel", "bcc", "akiec"})  # malignant or pre-malignant
BENIGN = frozenset(set(CLASSES) - MALIGNANT)

AUTO_TRIAGE, REFER, RE_IMAGE = "AUTO_TRIAGE", "REFER", "RE_IMAGE"

# Aliases so agents with slightly different output keys still plug in.
ALIASES = {
    "q_status": ("quality_status", "status", "quality", "verdict"),
    "q_usable": ("usable", "is_usable", "passed", "pass", "ok"),
    "q_score": ("quality_score", "score", "q_score"),
    "q_reasons": ("reasons", "issues", "quality_reasons", "problems"),
    "d_label": ("predicted_class", "pred_class", "label", "prediction", "diagnosis", "class"),
    "d_probs": ("probabilities", "class_probabilities", "class_probs", "probs", "proba"),
    "d_conf": ("confidence", "conf", "max_prob", "score"),
    "l_susp": ("suspicion_score", "suspicion", "score", "risk_score"),
    "l_flags": ("reliability_flags", "flags", "reliability"),
    "l_evidence": ("feature_evidence", "evidence", "features"),
    "c_risk": ("risk_indicator", "risk", "risk_level", "context_risk"),
    "c_missing": ("missing_fields", "missing", "missing_flags"),
    "s_flag": ("safety_flag", "flag", "veto", "unsafe", "is_unsafe", "conflict"),
    "s_trigger": ("trigger", "triggers", "reason", "reasons"),
    "s_just": ("justification", "explanation", "rationale", "message"),
    "s_action": ("action", "recommended_action", "recommendation", "triage"),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if dataclasses.is_dataclass(obj):
        return getattr(obj, key, default)
    return getattr(obj, key, default)


def _pick(obj: Any, alias_key: str, default: Any = None) -> Any:
    for k in ALIASES[alias_key]:
        v = _get(obj, k, None)
        if v is not None:
            return v
    return default


def _f(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(x)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _truthy(x: Any) -> bool:
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "flag", "flagged", "veto", "unsafe",
                                      "conflict", "fail", "failed", "high", "block"}
    if isinstance(x, (list, tuple, set, dict)):
        return len(x) > 0
    return bool(x)


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x] if x else []
    if isinstance(x, dict):
        return [f"{k}={v}" for k, v in x.items() if v]
    try:
        return [str(i) for i in x]
    except TypeError:
        return [str(x)]


def _norm_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    return s if s else None


def _softmax_if_needed(p: Dict[str, float]) -> Dict[str, float]:
    tot = sum(p.values())
    if tot <= 0:
        return p
    if abs(tot - 1.0) > 1e-3:  # logits or unnormalised
        if min(p.values()) < 0 or tot > 1.5:
            m = max(p.values())
            e = {k: math.exp(v - m) for k, v in p.items()}
            z = sum(e.values())
            return {k: v / z for k, v in e.items()}
        return {k: v / tot for k, v in p.items()}
    return p


# --------------------------------------------------------------------------- #
# normalised views of each agent output
# --------------------------------------------------------------------------- #
@dataclass
class QualityView:
    usable: bool = True
    marginal: bool = False
    score: Optional[float] = None
    reasons: List[str] = field(default_factory=list)


@dataclass
class DiagnosisView:
    label: Optional[str] = None
    probs: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    margin: float = 0.0
    second: Optional[str] = None
    p_malignant: float = 0.0


@dataclass
class LesionView:
    suspicion: Optional[float] = None
    unreliable: bool = False
    flags: List[str] = field(default_factory=list)


@dataclass
class ContextView:
    risk: str = "unknown"  # low / medium / high / unknown
    missing: List[str] = field(default_factory=list)


@dataclass
class SafetyView:
    flagged: bool = False
    wants_reimage: bool = False
    trigger: str = ""
    justification: str = ""


def parse_quality(a1: Any) -> QualityView:
    v = QualityView()
    if a1 is None:
        return v
    status = _pick(a1, "q_status")
    usable = _pick(a1, "q_usable")
    score = _f(_pick(a1, "q_score"))
    v.score = score
    v.reasons = _as_list(_pick(a1, "q_reasons"))
    if isinstance(status, str):
        s = status.strip().lower()
        if s in {"fail", "failed", "poor", "bad", "unusable", "reject", "re_image", "reimage", "not usable"}:
            v.usable = False
        elif s in {"marginal", "borderline", "warn", "warning", "acceptable", "medium", "fair"}:
            v.marginal = True
        elif s in {"pass", "passed", "good", "ok", "usable", "high"}:
            v.usable = True
    if usable is not None and not isinstance(usable, str):
        v.usable = bool(usable) if v.usable else False
    return v


def parse_diagnosis(a2: Any) -> DiagnosisView:
    v = DiagnosisView()
    if a2 is None:
        return v
    probs = _pick(a2, "d_probs")
    if isinstance(probs, dict) and probs:
        p = {str(k).lower(): _f(val, 0.0) for k, val in probs.items()}
        p = _softmax_if_needed(p)
    elif isinstance(probs, (list, tuple)) and len(probs) == len(CLASSES):
        p = _softmax_if_needed({c: _f(x, 0.0) for c, x in zip(CLASSES, probs)})
    else:
        p = {}
    label = _norm_label(_pick(a2, "d_label"))
    conf = _f(_pick(a2, "d_conf"))
    if p:
        ranked = sorted(p.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_p = ranked[0]
        label = label or top_label
        conf = p.get(label, top_p) if conf is None else conf
        v.second = ranked[1][0] if len(ranked) > 1 else None
        v.margin = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else ranked[0][1]
        v.p_malignant = sum(val for k, val in p.items() if k in MALIGNANT)
    else:
        v.margin = conf or 0.0
        if label in MALIGNANT:
            v.p_malignant = conf or 0.0
    v.label, v.probs, v.confidence = label, p, float(conf or 0.0)
    return v


def parse_lesion(a3: Any) -> LesionView:
    v = LesionView()
    if a3 is None:
        return v
    v.suspicion = _f(_pick(a3, "l_susp"))
    if v.suspicion is not None and v.suspicion > 1.0:  # 0-100 scale
        v.suspicion /= 100.0
    flags = _as_list(_pick(a3, "l_flags"))
    v.flags = flags
    low = " ".join(flags).lower()
    v.unreliable = any(t in low for t in ("unreliable", "low", "poor", "uncertain", "unstable"))
    return v


def parse_context(a4: Any) -> ContextView:
    v = ContextView()
    if a4 is None:
        return v
    risk = _pick(a4, "c_risk")
    if isinstance(risk, (int, float)):
        r = float(risk)
        r = r / 100 if r > 1 else r
        v.risk = "high" if r >= 0.66 else "medium" if r >= 0.33 else "low"
    elif isinstance(risk, str):
        s = risk.strip().lower()
        v.risk = "high" if s in {"high", "elevated", "increased"} else \
                 "medium" if s in {"medium", "moderate", "med"} else \
                 "low" if s in {"low", "average", "normal"} else "unknown"
    v.missing = _as_list(_pick(a4, "c_missing"))
    return v


def parse_safety(a5: Any) -> SafetyView:
    v = SafetyView()
    if a5 is None:
        return v
    v.flagged = _truthy(_pick(a5, "s_flag", False))
    action = str(_pick(a5, "s_action", "") or "").upper()
    if "RE_IMAGE" in action or "REIMAGE" in action:
        v.wants_reimage, v.flagged = True, True
    elif "REFER" in action:
        v.flagged = True
    v.trigger = "; ".join(_as_list(_pick(a5, "s_trigger")))
    v.justification = str(_pick(a5, "s_just", "") or "")
    return v


# --------------------------------------------------------------------------- #
# configuration & result
# --------------------------------------------------------------------------- #
@dataclass
class CoordinatorPolicy:
    quality_fail: float = 0.30            # quality score below this => RE_IMAGE
    quality_marginal: float = 0.50        # below this => marginal
    min_confidence: float = 0.80          # AUTO_TRIAGE needs top prob >= this
    min_margin: float = 0.40              # and top1-top2 gap >= this
    marginal_min_confidence: float = 0.90  # stricter when quality is marginal
    malignant_prob_refer: float = 0.20    # P(mel+bcc+akiec) >= this => REFER
    mel_prob_refer: float = 0.10          # P(mel) >= this => REFER
    suspicion_refer: float = 0.60         # lesion suspicion >= this => REFER
    refer_on_high_context_risk: bool = True
    refer_on_unreliable_lesion: bool = True
    auto_classes: Tuple[str, ...] = tuple(sorted(BENIGN))
    version: str = "1.0"


@dataclass
class CoordinatorResult:
    final_label: str
    triage: str
    confidence: float
    rationale: List[str]
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def rationale_text(self) -> str:
        return " | ".join(self.rationale)

    def __str__(self) -> str:
        return f"[{self.triage}] {self.final_label} ({self.confidence:.2f}) :: {self.rationale_text}"


# --------------------------------------------------------------------------- #
# the agent
# --------------------------------------------------------------------------- #
class CoordinatorAgent:
    """Agent 6: applies a documented policy over Agents 1-5."""

    name = "CoordinatorAgent"

    def __init__(self, policy: Optional[CoordinatorPolicy] = None, default_label: str = "nv"):
        self.policy = policy or CoordinatorPolicy()
        self.default_label = default_label

    # -- main entry point ---------------------------------------------------
    def decide(self, a1: Any = None, a2: Any = None, a3: Any = None,
               a4: Any = None, a5: Any = None) -> CoordinatorResult:
        p = self.policy
        q, d = parse_quality(a1), parse_diagnosis(a2)
        l, c, s = parse_lesion(a3), parse_context(a4), parse_safety(a5)

        label = d.label if d.label in CLASSES else (d.label or self.default_label)
        notes: List[str] = [
            f"A2 diagnosis: {label} (conf {d.confidence:.2f}, margin {d.margin:.2f}, "
            f"P(malignant-group) {d.p_malignant:.2f})"
        ]
        if q.score is not None or q.reasons:
            notes.append(f"A1 quality: {'usable' if q.usable else 'NOT usable'}"
                         f"{' (marginal)' if q.marginal else ''}"
                         f"{'' if q.score is None else f', score {q.score:.2f}'}"
                         f"{'; ' + ', '.join(q.reasons) if q.reasons else ''}")
        if l.suspicion is not None:
            notes.append(f"A3 lesion suspicion {l.suspicion:.2f}"
                         f"{'; flags: ' + ', '.join(l.flags) if l.flags else ''}")
        if c.risk != "unknown" or c.missing:
            notes.append(f"A4 context risk {c.risk}"
                         f"{'; missing: ' + ', '.join(c.missing) if c.missing else ''}")
        if s.flagged:
            notes.append(f"A5 SAFETY FLAG: {s.trigger or 'unspecified'}"
                         f"{' - ' + s.justification if s.justification else ''}")

        triage, why = self._policy(q, d, l, c, s, label)
        notes.append(f"Decision: {triage} - {why}")

        evidence = {
            "quality": dataclasses.asdict(q), "diagnosis": dataclasses.asdict(d),
            "lesion": dataclasses.asdict(l), "context": dataclasses.asdict(c),
            "safety": dataclasses.asdict(s), "policy_version": p.version,
        }
        return CoordinatorResult(label, triage, round(d.confidence, 4), notes, evidence)

    __call__ = decide

    # -- policy -------------------------------------------------------------
    def _policy(self, q: QualityView, d: DiagnosisView, l: LesionView,
                c: ContextView, s: SafetyView, label: str) -> Tuple[str, str]:
        p = self.policy
        if not q.usable or (q.score is not None and q.score < p.quality_fail):
            return RE_IMAGE, "R1 image quality inadequate for a reliable reading"
        if s.wants_reimage:
            return RE_IMAGE, "R2 safety agent requested a new image"
        if s.flagged:
            return REFER, "R3 safety agent veto/flag overrides automation"
        p_mel = d.probs.get("mel", 0.0)
        if label in MALIGNANT:
            return REFER, f"R4 predicted class '{label}' is malignant/pre-malignant"
        if d.p_malignant >= p.malignant_prob_refer or p_mel >= p.mel_prob_refer:
            return REFER, (f"R4 non-trivial malignant probability "
                           f"(group {d.p_malignant:.2f}, mel {p_mel:.2f})")
        if label not in p.auto_classes:
            return REFER, f"R4 class '{label}' not eligible for automatic screening"
        need_conf = p.marginal_min_confidence if q.marginal or \
            (q.score is not None and q.score < p.quality_marginal) else p.min_confidence
        if d.confidence < need_conf:
            return REFER, f"R5 confidence {d.confidence:.2f} < required {need_conf:.2f}"
        if d.margin < p.min_margin:
            return REFER, f"R5 top-2 margin {d.margin:.2f} < {p.min_margin:.2f}"
        if l.suspicion is not None and l.suspicion >= p.suspicion_refer:
            return REFER, f"R6 lesion analysis suspicion {l.suspicion:.2f} >= {p.suspicion_refer:.2f}"
        if p.refer_on_unreliable_lesion and l.unreliable and (l.suspicion or 0) >= 0.4:
            return REFER, "R6 lesion cues unreliable but moderately suspicious"
        if p.refer_on_high_context_risk and c.risk == "high":
            return REFER, "R7 high contextual risk (age/site/sex)"
        return AUTO_TRIAGE, (f"R9 benign class '{label}', confident, consistent with "
                             f"quality, lesion, context and safety checks")

    # -- batch utilities ----------------------------------------------------
    def run_batch(self, records: Iterable[Dict[str, Any]]) -> List[CoordinatorResult]:
        """records: dicts with keys a1..a5 (any may be missing)."""
        return [self.decide(r.get("a1"), r.get("a2"), r.get("a3"), r.get("a4"), r.get("a5"))
                for r in records]

    @staticmethod
    def evaluate(results: Sequence[CoordinatorResult], truth: Sequence[str]) -> Dict[str, Any]:
        """Hackathon counts: correct/incorrect AUTO_TRIAGE, plus overall metrics."""
        n = len(results)
        truth = [_norm_label(t) for t in truth]
        counts = {AUTO_TRIAGE: 0, REFER: 0, RE_IMAGE: 0}
        correct_auto = wrong_auto = correct_all = missed_malignant_auto = 0
        for r, t in zip(results, truth):
            counts[r.triage] += 1
            ok = r.final_label == t
            correct_all += ok
            if r.triage == AUTO_TRIAGE:
                correct_auto += ok
                wrong_auto += (not ok)
                missed_malignant_auto += (t in MALIGNANT)
        return {
            "n": n, "label_accuracy": correct_all / max(n, 1),
            "triage_counts": counts,
            "correct_auto_triage": correct_auto, "incorrect_auto_triage": wrong_auto,
            "auto_triage_precision": correct_auto / max(counts[AUTO_TRIAGE], 1),
            "malignant_in_auto_triage": missed_malignant_auto,
        }

    def tune(self, records: Sequence[Dict[str, Any]], truth: Sequence[str],
             max_wrong_rate: float = 0.03, max_malignant_auto: int = 0) -> CoordinatorPolicy:
        """Grid-search thresholds on the labelled validation set.

        Maximises number of correct AUTO_TRIAGE subject to:
          incorrect/auto <= max_wrong_rate and malignant slipped into AUTO <= max_malignant_auto.
        Sets and returns the tuned policy. Never tune on hidden test images.
        """
        import itertools
        base = self.policy
        best, best_score = base, (-1, 0.0)
        grid = itertools.product(
            (0.6, 0.7, 0.8, 0.85, 0.9, 0.95),   # min_confidence
            (0.2, 0.3, 0.4, 0.5, 0.6),          # min_margin
            (0.05, 0.1, 0.15, 0.2, 0.3),        # malignant_prob_refer
            (0.03, 0.05, 0.1, 0.15),            # mel_prob_refer
        )
        for conf, marg, mal, mel in grid:
            pol = dataclasses.replace(base, min_confidence=conf, min_margin=marg,
                                      malignant_prob_refer=mal, mel_prob_refer=mel,
                                      marginal_min_confidence=max(conf, base.marginal_min_confidence))
            self.policy = pol
            m = self.evaluate(self.run_batch(records), truth)
            n_auto = m["correct_auto_triage"] + m["incorrect_auto_triage"]
            wrong_rate = m["incorrect_auto_triage"] / max(n_auto, 1)
            if wrong_rate <= max_wrong_rate and m["malignant_in_auto_triage"] <= max_malignant_auto:
                score = (m["correct_auto_triage"], -wrong_rate)
                if score > best_score:
                    best, best_score = pol, score
        self.policy = best
        return best


# --------------------------------------------------------------------------- #
# CLI:  python coordinator_agent.py outputs.jsonl [--truth-key label] [--tune]
# each line: {"id":..., "a1":{...}, "a2":{...}, "a3":{...}, "a4":{...}, "a5":{...}, "label": "nv"}
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--truth-key", default="label")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--out", default="coordinator_results.jsonl")
    args = ap.parse_args()

    with open(args.jsonl) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    agent = CoordinatorAgent()
    truth = [r.get(args.truth_key) for r in rows]
    has_truth = all(t is not None for t in truth)
    if args.tune and has_truth:
        print("Tuned policy:", agent.tune(rows, truth))
    res = agent.run_batch(rows)
    with open(args.out, "w") as fh:
        for r, out in zip(rows, res):
            fh.write(json.dumps({"id": r.get("id"), **out.to_dict()}) + "\n")
    if has_truth:
        print(json.dumps(agent.evaluate(res, truth), indent=2))
    print(f"Wrote {len(res)} decisions to {args.out}")
