"""Image Quality Agent (rule-based, OpenCV).

    engine = QualityEngine()
    payload = engine.analyze_bytes(file_bytes)   # uploaded image
    payload = engine.analyze(frame_bgr)          # webcam frame (numpy BGR array)

Agent payload (agent contract): agent, score (0-100), severity (PASS|WARN|FAIL),
reasons[], defects[], quality_factor (0-1). Detail fields used by the UI
(quality_analysis, decision, recommendations, ...) are kept alongside.
"""
import cv2
import numpy as np

from .config import QualityConfig
from . import checks

FEEDBACK = {
    "clarity": "Hold the camera steady and let it focus",
    "illumination": "Adjust the lighting (not too dark, not too bright)",
    "glare": "Change the camera angle or move away from direct light to remove reflections",
    "obstruction": "Move hair, fingers or objects away from the lesion",
    "resolution": "Use a higher-resolution camera or move closer",
    "framing": "Centre the whole lesion in the frame",
}

SOFT_TIPS = {
    "clarity": "Sharpness is acceptable but could be better; steady your hand and tap to focus",
    "illumination": "Lighting is acceptable; even, diffuse light would improve it",
    "glare": "Minor reflections are present; tilt the camera slightly",
    "obstruction": "Some hair or thin objects are visible; move or trim them if possible",
    "resolution": "Resolution is acceptable; a closer or higher-resolution capture would help",
    "framing": "Framing is acceptable; centre the lesion a little better",
}


def grade(score):
    return ("EXCELLENT" if score >= 90 else "GOOD" if score >= 80
            else "POOR" if score >= 60 else "VERY_POOR")


def defect_type(key, result, cfg):
    """Machine-readable defect name, so downstream agents can cite the exact problem."""
    if key == "illumination":
        mean = result["metrics"]["mean_gray"]
        return "low_illumination" if mean < sum(cfg.mean_ok_range) / 2 else "overexposure"
    return {"clarity": "blur", "glare": "glare", "obstruction": "hair_obstruction",
            "resolution": "low_resolution", "framing": "poor_framing"}[key]


class QualityEngine:
    def __init__(self, config: QualityConfig | None = None):
        self.cfg = config or QualityConfig()

    def analyze_bytes(self, data: bytes, image_id: str = "IMG"):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return self._invalid(image_id, "File is not a readable image")
        return self.analyze(img, image_id)

    def analyze(self, bgr: np.ndarray, image_id: str = "IMG"):
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.size == 0:
            return self._invalid(image_id, "Image must be a 3-channel colour image")
        cfg = self.cfg
        oh, ow = bgr.shape[:2]

        scale = cfg.analysis_size / max(oh, ow)
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        results = {
            "clarity": checks.check_clarity(gray, cfg),
            "illumination": checks.check_illumination(gray, cfg),
            "glare": checks.check_glare(small, cfg),
            "obstruction": checks.check_obstruction(gray, cfg),
            "resolution": checks.check_resolution(oh, ow, cfg),
            "framing": checks.check_framing(small, cfg),
        }

        overall = sum(cfg.weights[k] * results[k]["score"] for k in results)
        failed = [k for k in cfg.critical if not results[k]["passed"]]   # can reject on their own
        approved = overall >= cfg.approve_threshold and not failed

        decision_reasons = [results[k]["reason"] for k in failed]
        if not failed and not approved:
            decision_reasons.append(f"Overall quality {overall:.0f} is below {cfg.approve_threshold:.0f}")

        # ---- defects (every problem found, FAIL ones first) ----
        defects = []
        for k, r in results.items():
            if not r["passed"]:
                sev, detail = ("FAIL" if k in cfg.critical else "WARN"), r["reason"]
            elif r["score"] < cfg.warn_score:
                sev, detail = "WARN", f"{k} score is {r['score']:.0f}/100"
            else:
                continue
            defects.append({"type": defect_type(k, r, cfg), "check": k, "severity": sev,
                            "score": r["score"], "detail": detail})
        defects.sort(key=lambda d: d["severity"] != "FAIL")

        severity = "FAIL" if not approved else ("WARN" if defects else "PASS")
        reasons = [d["detail"] for d in defects]
        if not failed and not approved:
            reasons.append(decision_reasons[-1])

        # ---- recommendations ----
        recommendations = []
        for k, r in results.items():
            if not r["passed"] and k in cfg.critical:
                recommendations.append({"check": k, "severity": "MUST_FIX",
                                        "problem": r["reason"], "action": FEEDBACK[k]})
            elif not r["passed"]:
                recommendations.append({"check": k, "severity": "SUGGESTED",
                                        "problem": r["reason"], "action": FEEDBACK[k]})
            elif r["score"] < cfg.warn_score:
                recommendations.append({"check": k, "severity": "SUGGESTED",
                                        "problem": f"{k} score is {r['score']:.0f}/100",
                                        "action": SOFT_TIPS[k]})
        if not recommendations:
            recommendations.append({"check": "overall", "severity": "INFO", "problem": "",
                                    "action": "Image quality is good; no changes needed"})

        return {
            # ---- agent contract ----
            "agent": "image_quality",
            "score": round(overall, 1),
            "severity": severity,
            "reasons": reasons,
            "defects": defects,
            "quality_factor": round(overall / 100, 3),
            # ---- detail ----
            "model": "ImageQualityAgent",
            "image_id": image_id,
            "quality_analysis": results,
            "overall_quality_score": round(overall, 1),
            "quality_grade": grade(overall),
            "recommendations": recommendations,
            "decision": {
                "approved": approved,
                "status": "APPROVED" if approved else "REJECTED",
                "reasons": decision_reasons,
                "feedback": [FEEDBACK[k] for k in failed],
            },
            "next_step": "SEND_TO_MODEL_2" if approved else "REQUEST_NEW_IMAGE",
        }

    def _invalid(self, image_id, reason):
        return {"agent": "image_quality", "score": 0.0, "severity": "FAIL", "reasons": [reason],
                "defects": [{"type": "invalid_image", "check": "input", "severity": "FAIL",
                             "score": 0.0, "detail": reason}],
                "quality_factor": 0.0,
                "model": "ImageQualityAgent", "image_id": image_id, "quality_analysis": {},
                "overall_quality_score": 0.0, "quality_grade": "VERY_POOR",
                "recommendations": [{"check": "input", "severity": "MUST_FIX", "problem": reason,
                                     "action": "Upload a valid JPG or PNG image"}],
                "decision": {"approved": False, "status": "REJECTED",
                             "reasons": [reason], "feedback": ["Upload a valid JPG or PNG image"]},
                "next_step": "REQUEST_NEW_IMAGE"}


class StabilityTracker:
    """Webcam helper: unlock capture only after N consecutive approved frames."""

    def __init__(self, required_frames: int = 5):
        self.required, self.streak = required_frames, 0

    def update(self, report) -> bool:
        self.streak = self.streak + 1 if report["decision"]["approved"] else 0
        return self.streak >= self.required