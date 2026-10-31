"""
Agent 3 - Lesion Analysis Agent
Looks for visual cues (ABCD-style): asymmetry, border irregularity, colour variation, size.
Output: feature evidence, suspicion score (0-1), reliability (0-1).

Rule-based by default. Call fit_weights() on the labelled validation set to
learn the weights instead of using the hand-set ones.
"""
from dataclasses import dataclass, asdict
import cv2
import numpy as np

# Hand-set weights / thresholds (tune on the 2,000 validation images)
WEIGHTS = {"asymmetry": 0.30, "border": 0.30, "color": 0.30, "size": 0.10}
THRESH_LOW = 0.33          # score < this        -> LOW
THRESH_SUSPICIOUS = 0.55   # score >= this       -> SUSPICIOUS, in between -> MODERATE
WORK_SIZE = 256


@dataclass
class LesionFeatures:
    asymmetry: float        # 0 symmetric .. ~0.5 very asymmetric
    compactness: float      # 1 = perfect circle
    solidity: float         # lesion area / convex hull area
    border_irregularity: float  # 0..1
    n_colors: int
    color_std: float
    area_frac: float        # lesion area / image area
    contrast: float         # lesion vs surrounding skin (Lab distance)
    touches_border: float   # fraction of lesion contour on image edge


# ----------------------------------------------------------------- segmentation
def _remove_hair(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    _, hair = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(bgr, hair, 3, cv2.INPAINT_TELEA)


def segment_lesion(bgr):
    """Returns (mask uint8 0/1, cleaned small BGR image, lab float image, skin_lab)."""
    h, w = bgr.shape[:2]
    s = WORK_SIZE / max(h, w)
    small = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    clean = _remove_hair(small)
    lab = cv2.cvtColor(cv2.GaussianBlur(clean, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)

    H, W = lab.shape[:2]
    b = max(4, int(0.05 * min(H, W)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    skin = np.median(border, axis=0)

    dist = np.linalg.norm(lab - skin, axis=2)
    d8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((H, W), np.uint8), clean, lab, skin
    cx0, cy0 = W / 2, H / 2

    def score(c):  # big and near the centre wins
        M = cv2.moments(c)
        if M["m00"] == 0:
            return 0
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        d = np.hypot(cx - cx0, cy - cy0) / np.hypot(cx0, cy0)
        return M["m00"] * (1 - 0.7 * d)

    best = max(cnts, key=score)
    mask = np.zeros((H, W), np.uint8)
    cv2.drawContours(mask, [best], -1, 1, thickness=-1)  # fills holes
    return mask, clean, lab, skin


# --------------------------------------------------------------------- features
def _asymmetry(mask):
    """Align principal axis, flip across both axes, average (1 - IoU)."""
    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] == 0:
        return 0.0
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    angle = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
    H, W = mask.shape
    R = cv2.getRotationMatrix2D((cx, cy), np.degrees(angle), 1.0)
    R[0, 2] += W / 2 - cx
    R[1, 2] += H / 2 - cy
    a = cv2.warpAffine(mask, R, (W, H), flags=cv2.INTER_NEAREST)
    scores = []
    for flip in (0, 1):
        f = cv2.flip(a, flip)
        inter = np.logical_and(a, f).sum()
        union = np.logical_or(a, f).sum()
        scores.append(1 - inter / union if union else 0)
    return float(np.mean(scores))


def _colors(lab, mask, merge_dist=18.0, min_frac=0.05, k=6):
    px = lab[mask > 0]
    if len(px) < 50:
        return 1, 0.0
    if len(px) > 4000:
        px = px[np.random.RandomState(0).choice(len(px), 4000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(px.astype(np.float32), k, None, crit, 3,
                                    cv2.KMEANS_PP_CENTERS)
    fr = np.bincount(labels.ravel(), minlength=k) / len(labels)
    keep = [centers[i] for i in range(k) if fr[i] >= min_frac]
    merged = []
    for c in keep:  # merge clusters that are perceptually close
        if all(np.linalg.norm(c - m) > merge_dist for m in merged):
            merged.append(c)
    return max(1, len(merged)), float(np.mean(px.std(axis=0)))


def extract_features(bgr):
    mask, clean, lab, skin = segment_lesion(bgr)
    H, W = mask.shape
    area = float(mask.sum())
    if area < 20:
        return None, mask
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    perim = cv2.arcLength(c, True)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    compactness = float(min(1.0, 4 * np.pi * area / (perim ** 2 + 1e-6)))
    solidity = float(area / (hull_area + 1e-6))
    border_irr = float(np.clip(0.5 * (1 - compactness) + 0.5 * np.clip((1 - solidity) * 4, 0, 1), 0, 1))
    n_col, col_std = _colors(lab, mask)
    pts = c.reshape(-1, 2)
    on_edge = np.mean((pts[:, 0] <= 1) | (pts[:, 1] <= 1) | (pts[:, 0] >= W - 2) | (pts[:, 1] >= H - 2))
    inside = np.linalg.norm(lab[mask > 0] - skin, axis=1).mean()
    return LesionFeatures(
        asymmetry=_asymmetry(mask), compactness=compactness, solidity=solidity,
        border_irregularity=border_irr, n_colors=n_col, color_std=col_std,
        area_frac=area / (H * W), contrast=float(inside), touches_border=float(on_edge),
    ), mask


# ---------------------------------------------------------------- score / agent
def _norm_vec(f: LesionFeatures):
    asym = np.clip(f.asymmetry / 0.35, 0, 1)
    border = f.border_irregularity
    color = np.clip(0.6 * (f.n_colors - 1) / 3 + 0.4 * f.color_std / 25, 0, 1)
    size = np.clip((f.area_frac - 0.05) / 0.35, 0, 1)
    return np.array([asym, border, color, size])


def suspicion_score(f: LesionFeatures, weights=None):
    w = weights or WEIGHTS
    v = _norm_vec(f)
    wv = np.array([w["asymmetry"], w["border"], w["color"], w["size"]])
    return float(np.clip(v @ wv / wv.sum(), 0, 1))


def reliability(f: LesionFeatures):
    flags, r = [], 1.0
    if f.area_frac < 0.02:
        r -= 0.5; flags.append("lesion_too_small_or_not_found")
    if f.area_frac > 0.80:
        r -= 0.4; flags.append("segmentation_covers_most_of_image")
    if f.touches_border > 0.15:
        r -= 0.3; flags.append("lesion_cut_off_by_frame")
    if f.contrast < 12:
        r -= 0.3; flags.append("low_lesion_skin_contrast")
    return float(np.clip(r, 0, 1)), flags


def lesion_analysis_agent(image, weights=None):
    """image: BGR ndarray or file path. Returns the Agent-3 output dict."""
    bgr = cv2.imread(image) if isinstance(image, str) else image
    if bgr is None:
        return {"agent": "A3", "ok": False, "symmetry": None, "border": None, "color": None,
                "suspicion": None, "label": "UNKNOWN", "suspicion_score": None,
                "reliability": 0.0, "flags": ["unreadable_image"], "features": {}}
    f, _ = extract_features(bgr)
    if f is None:
        return {"agent": "A3", "ok": False, "symmetry": None, "border": None, "color": None,
                "suspicion": None, "label": "UNKNOWN", "suspicion_score": None,
                "reliability": 0.0, "flags": ["no_lesion_found"], "features": {}}
    s = suspicion_score(f, weights)
    rel, flags = reliability(f)
    label = "SUSPICIOUS" if s >= THRESH_SUSPICIOUS else "MODERATE" if s >= THRESH_LOW else "LOW"
    evidence = []
    if f.asymmetry > 0.2: evidence.append("asymmetric shape")
    if f.border_irregularity > 0.5: evidence.append("irregular border")
    if f.n_colors >= 3: evidence.append(f"{f.n_colors} distinct colours")
    if f.area_frac > 0.25: evidence.append("large lesion")
    v = _norm_vec(f)   # [asymmetry, border, color, size], each 0..1 (higher = more abnormal)
    return {"agent": "A3", "ok": True,
            "symmetry": round(float(1 - v[0]), 3),   # 1 = perfectly symmetric
            "border": round(float(v[1]), 3),         # 1 = very irregular border
            "color": round(float(v[2]), 3),          # 1 = high colour variation
            "suspicion": round(s, 3),                # overall 0..1
            "label": label, "suspicion_score": round(s, 3),
            "reliability": round(rel, 3), "flags": flags, "evidence": evidence,
            "features": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(f).items()}}


# ---------------------------------------------------- optional: tune on val set
def fit_weights(feature_rows, is_malignant):
    """
    feature_rows: list of LesionFeatures from validation images
    is_malignant: list of 0/1 (MEL, BCC, AKIEC = 1)
    Returns a weights dict you can pass to lesion_analysis_agent(weights=...).
    """
    from sklearn.linear_model import LogisticRegression
    X = np.stack([_norm_vec(f) for f in feature_rows])
    clf = LogisticRegression(class_weight="balanced", C=1.0).fit(X, is_malignant)
    w = np.clip(clf.coef_[0], 1e-3, None)
    w = w / w.sum()
    return dict(zip(["asymmetry", "border", "color", "size"], w.tolist()))