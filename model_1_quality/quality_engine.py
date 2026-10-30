"""Model 1: Image Quality Agent (rule-based, OpenCV).

    engine = QualityEngine()
    report = engine.analyze_bytes(file_bytes)   # uploaded image
    report = engine.analyze(frame_bgr)          # webcam frame (numpy BGR array)

Agent contract (read by the Triage Agent):
    report["agent"]               "image_quality"
    report["score"]               0-100 overall quality score
    report["status"]              "PASS" | "WARN" | "FAIL"
    report["reasons"][]           human-readable defects / warnings
    report["defects"]             per-defect severity (blur, glare, low_illumination, ...)
    report["confidence_modifier"] 0.0-1.0, multiply the diagnosis confidence by this
    report["recommended_action"]  "PROCEED" | "PROCEED_WITH_CAUTION" | "RE_IMAGE"
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


class QualityEngine:
    def __init__(self, config: QualityConfig | None = None):
        self.cfg = config or QualityConfig()

    # ------------------------------------------------------------------ input
    def analyze_bytes(self, data: bytes, image_id: str = "IMG"):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return self._invalid(image_id, "File is not a readable image")
        return self.analyze(img, image_id)

    def _severity(self, key, r):
        if not r["passed"]:
            return "FAIL" if key in self.cfg.critical else "WARN"
        return "WARN" if r["score"] < self.cfg.warn_score else "PASS"

    # --------------------------------------------------------------- analysis
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
        for k, r in results.items():
            r["severity"] = self._severity(k, r)

        overall = sum(cfg.weights[k] * results[k]["score"] for k in results)
        failed = [k for k, r in results.items() if r["severity"] == "FAIL"]
        warned = [k for k, r in results.items() if r["severity"] == "WARN"]
        approved = overall >= cfg.approve_threshold and not failed

        reasons = [results[k]["reason"] for k in failed]
        if not failed and not approved:
            reasons.append(f"Overall quality {overall:.0f} is below {cfg.approve_threshold:.0f}")
        warnings = [results[k]["reason"] if not results[k]["passed"] else SOFT_TIPS[k] for k in warned]

        status = "FAIL" if not approved else ("WARN" if warned else "PASS")

        recommendations = []
        for k, r in results.items():
            if r["severity"] == "FAIL":
                recommendations.append({"check": k, "severity": "MUST_FIX",
                                        "problem": r["reason"], "action": FEEDBACK[k]})
            elif r["severity"] == "WARN":
                if not r["passed"]:
                    recommendations.append({"check": k, "severity": "SUGGESTED",
                                            "problem": r["reason"], "action": FEEDBACK[k]})
                else:
                    recommendations.append({"check": k, "severity": "SUGGESTED",
                                            "problem": f"{k} score is {r['score']:.0f}/100",
                                            "action": SOFT_TIPS[k]})
        if not recommendations:
            recommendations.append({"check": "overall", "severity": "INFO", "problem": "",
                                    "action": "Image quality is good; no changes needed"})

        # Quality actively modulates diagnosis confidence downstream.
        if approved:
            modifier = round(min(1.0, overall / cfg.full_confidence_score), 3)
        else:
            modifier = 0.0
        action = "RE_IMAGE" if not approved else ("PROCEED_WITH_CAUTION" if warned else "PROCEED")

        under = results["illumination"]["metrics"].get("underexposed", 0.0) > 0.5
        ill_sev = results["illumination"]["severity"]
        defects = {
            "blur": results["clarity"]["severity"],
            "glare": results["glare"]["severity"],
            "low_illumination": ill_sev if under else "PASS",
            "overexposure": "PASS" if under else ill_sev,
            "hair_obstruction": results["obstruction"]["severity"],
            "low_resolution": results["resolution"]["severity"],
            "framing": results["framing"]["severity"],
        }

        return {
            # --- agent contract ---
            "agent": "image_quality",
            "model": "ImageQualityAgent",
            "image_id": image_id,
            "score": round(overall, 1),
            "status": status,
            "reasons": reasons + warnings,
            "defects": defects,
            "confidence_modifier": modifier,
            "recommended_action": action,
            # --- detail ---
            "quality_analysis": results,
            "overall_quality_score": round(overall, 1),
            "quality_grade": grade(overall),
            "recommendations": recommendations,
            "decision": {
                "approved": approved,
                "status": "APPROVED" if approved else "REJECTED",
                "reasons": reasons,
                "warnings": warnings,
                "feedback": [FEEDBACK[k] for k in failed],
            },
            "next_step": "SEND_TO_MODEL_2" if approved else "REQUEST_NEW_IMAGE",
        }

    def _invalid(self, image_id, reason):
        return {
            "agent": "image_quality", "model": "ImageQualityAgent", "image_id": image_id,
            "score": 0.0, "status": "FAIL", "reasons": [reason], "defects": {},
            "confidence_modifier": 0.0, "recommended_action": "RE_IMAGE",
            "quality_analysis": {}, "overall_quality_score": 0.0, "quality_grade": "VERY_POOR",
            "recommendations": [{"check": "input", "severity": "MUST_FIX", "problem": reason,
                                 "action": "Upload a valid JPG or PNG image"}],
            "decision": {"approved": False, "status": "REJECTED", "reasons": [reason],
                         "warnings": [], "feedback": ["Upload a valid JPG or PNG image"]},
            "next_step": "REQUEST_NEW_IMAGE",
        }


class StabilityTracker:
    """Webcam helper: unlock capture only after N consecutive approved frames."""

    def __init__(self, required_frames: int = 5):
        self.required, self.streak = required_frames, 0

    def update(self, report) -> bool:
        self.streak = self.streak + 1 if report["decision"]["approved"] else 0
        return self.streak >= self.required