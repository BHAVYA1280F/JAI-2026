"""Individual quality checks. Each returns:
{"score": 0-100 (higher = better), "passed": bool, "reason": str, "metrics": {...}}
"""
import cv2
import numpy as np

from .config import QualityConfig


def _lin(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0) * 100.0)


def _result(score, passed, reason, **metrics):
    return {"score": round(float(score), 1), "passed": bool(passed),
            "reason": reason, "metrics": {k: round(float(v), 4) for k, v in metrics.items()}}


def check_clarity(gray, cfg: QualityConfig):
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    passed = var >= cfg.blur_min_var
    return _result(_lin(var, 0, cfg.blur_good_var), passed,
                   "Image is sufficiently sharp" if passed else "Image is blurry",
                   laplacian_var=var)


def check_illumination(gray, cfg: QualityConfig):
    mean = float(gray.mean())
    dark = float((gray < 20).mean())
    bright = float((gray > 245).mean())
    lo, hi = cfg.mean_ok_range
    hlo, hhi = cfg.mean_hard_range
    if mean < lo:
        mean_score = _lin(mean, 20, lo)
    elif mean > hi:
        mean_score = 100 - _lin(mean, hi, 245)
    else:
        mean_score = 100.0
    score = max(0.0, mean_score - (dark + bright) * 150)

    if mean < hlo or dark > cfg.max_clipped_fraction:
        reason, passed = "Image is too dark (low illumination)", False
    elif mean > hhi or bright > cfg.max_clipped_fraction:
        reason, passed = "Image is too bright (overexposed)", False
    else:
        reason, passed = "Lighting is within the acceptable range", True
    # 1.0 when the problem (if any) is on the dark side, 0.0 for the bright side
    underexposed = float(mean < (lo + hi) / 2)
    return _result(score, passed, reason, mean_gray=mean, dark_fraction=dark,
                   bright_fraction=bright, underexposed=underexposed)


def check_glare(bgr, cfg: QualityConfig):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 2] >= cfg.glare_v_min) & (hsv[..., 1] <= cfg.glare_s_max)).astype(np.uint8)
    total = float(mask.mean())
    largest = 0.0
    if mask.any():
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            largest = float(stats[1:, cv2.CC_STAT_AREA].max()) / mask.size
    passed = total <= cfg.glare_max_fraction and largest <= cfg.glare_max_blob_fraction
    score = 100 - min(100.0, total * 100 * 8 + largest * 100 * 10)
    return _result(score, passed,
                   "No significant reflective obstruction detected" if passed
                   else "Glare or reflection covers part of the image",
                   glare_fraction=total, largest_glare_blob=largest)


def check_obstruction(gray, cfg: QualityConfig):
    """HEURISTIC: detects thin dark structures (hair). Does NOT detect fingers,
    objects or other occluders; use a segmentation model for that."""
    k = max(9, int(min(gray.shape) * 0.03) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(cv2.GaussianBlur(gray, (3, 3), 0), cv2.MORPH_BLACKHAT, kernel)
    hair = float((blackhat > 25).mean())
    passed = hair <= cfg.hair_max_fraction
    return _result(100 - min(100.0, hair * 100 * 6), passed,
                   "No major obstruction detected" if passed
                   else "Hair or thin objects may be covering the lesion",
                   hair_fraction=hair)


def check_resolution(orig_h, orig_w, cfg: QualityConfig):
    side = min(orig_h, orig_w)
    passed = side >= cfg.min_side_px
    return _result(_lin(side, cfg.min_side_px / 2, cfg.good_side_px), passed,
                   "Resolution is sufficient" if passed
                   else f"Resolution too low (shorter side {side}px, need {cfg.min_side_px}px)",
                   width=orig_w, height=orig_h, pixels=orig_w * orig_h)


def check_framing(bgr, cfg: QualityConfig):
    """HEURISTIC (advisory): assumes the lesion differs in colour from surrounding skin.
    Background colour is estimated from the image border; the lesion is the largest
    region that differs from it. Unreliable on vignetted dermoscopy images, so a
    failure here only produces a WARN, never a rejection."""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(2, int(min(h, w) * 0.05))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    dist = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return _result(0, False, "No lesion found; point the camera at the lesion",
                       lesion_area=0, centroid_x=0, centroid_y=0)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[i]
    area_frac = area / (h * w)
    cx, cy = cents[i][0] / w, cents[i][1] / h

    problems, score = [], 100.0
    lo, hi = cfg.lesion_area_range
    if area_frac < lo:
        problems.append("Lesion is too small; move the camera closer"); score -= 50
    elif area_frac > hi:
        problems.append("Lesion fills the whole frame; move the camera back"); score -= 50
    mx, my = cfg.border_margin * w, cfg.border_margin * h
    if x <= mx or y <= my or x + bw >= w - mx or y + bh >= h - my:
        problems.append("Lesion is cut off at the edge; reposition the camera"); score -= 40
    if abs(cx - 0.5) > cfg.centre_tolerance or abs(cy - 0.5) > cfg.centre_tolerance:
        problems.append("Lesion is not centred"); score -= 20
    return _result(max(0, score), not problems,
                   "Lesion is well framed" if not problems else "; ".join(problems),
                   lesion_area=area_frac, centroid_x=cx, centroid_y=cy)