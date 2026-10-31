"""Calibrate + evaluate Model 1 on a dataset.

Model 1 is rule-based, so "training" = choosing thresholds that best separate
good from bad images, then measuring how well the gate performs.

    python -m model_1_quality.evaluate --good_dir data/good [--bad_dir data/bad] \
        [--synthesize] [--out evaluation]

  --good_dir     images you consider good quality (e.g. sharp HAM10000 images)
  --bad_dir      (optional) real bad images you labelled by hand
  --synthesize   also create bad copies of the good images (blur, dark, bright,
                 glare, low-res, hair) with KNOWN defect labels; used for calibration
Positive class = BAD image. Sensitivity = bad images correctly rejected.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from .config import QualityConfig
from .quality_engine import QualityEngine

DEFECTS = ["blur", "dark", "bright", "glare", "lowres", "hair"]
CHECK_OF = {"blur": "clarity", "dark": "illumination", "bright": "illumination",
            "glare": "glare", "lowres": "resolution", "hair": "obstruction"}
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def list_images(folder, limit):
    paths = sorted(p for e in EXTS for p in glob.glob(os.path.join(folder, e)))
    return paths[:limit]


def degrade(img, kind, rng):
    h, w = img.shape[:2]
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), rng.uniform(4, 14))
    if kind == "dark":
        return (img.astype(np.float32) * rng.uniform(0.1, 0.35)).clip(0, 255).astype(np.uint8)
    if kind == "bright":
        return cv2.convertScaleAbs(img, alpha=1.0, beta=float(rng.uniform(90, 140)))
    if kind == "glare":
        out = img.copy()
        for _ in range(int(rng.integers(1, 4))):
            c = (int(rng.integers(w // 5, 4 * w // 5)), int(rng.integers(h // 5, 4 * h // 5)))
            cv2.circle(out, c, int(min(h, w) * rng.uniform(0.07, 0.15)), (255, 255, 255), -1)
        return out
    if kind == "lowres":
        f = rng.uniform(60, 200) / min(h, w)
        return cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    if kind == "hair":
        out = img.copy()
        for _ in range(int(rng.integers(40, 90))):
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p2 = (p1[0] + int(rng.integers(-w // 3, w // 3)), p1[1] + int(rng.integers(-h // 3, h // 3)))
            cv2.line(out, p1, p2, (20, 25, 30), int(rng.integers(1, 3)), cv2.LINE_AA)
        return out
    raise ValueError(kind)


def iter_samples(good_paths, bad_paths, synthesize, seed=0):
    """Deterministic generator of (name, image, label_bad, defect)."""
    for i, p in enumerate(good_paths):
        img = cv2.imread(p)
        if img is None:
            continue
        yield os.path.basename(p), img, 0, None
        if synthesize:
            rng = np.random.default_rng(seed + i)
            for kind in DEFECTS:
                yield f"{os.path.basename(p)}#{kind}", degrade(img, kind, rng), 1, kind
    for p in bad_paths:
        img = cv2.imread(p)
        if img is not None:
            yield os.path.basename(p), img, 1, "real_bad"


def run(engine, samples):
    rows = []
    for name, img, label_bad, defect in samples:
        r = engine.analyze(img, name)
        rows.append({"name": name, "label_bad": label_bad, "defect": defect,
                     "approved": r["decision"]["approved"], "overall": r["overall_quality_score"],
                     "checks": r["quality_analysis"]})
    return rows


def _median(xs):
    return round(float(np.median(xs)), 1) if len(xs) else None


def _div(a, b):
    return round(a / b, 4) if b else None


def summarize(rows):
    tp = sum(r["label_bad"] and not r["approved"] for r in rows)   # bad, rejected
    fn = sum(r["label_bad"] and r["approved"] for r in rows)       # bad, approved (dangerous)
    tn = sum(not r["label_bad"] and r["approved"] for r in rows)   # good, approved
    fp = sum(not r["label_bad"] and not r["approved"] for r in rows)  # good, rejected
    prec, sens = _div(tp, tp + fp), _div(tp, tp + fn)
    good = [r for r in rows if not r["label_bad"]]
    bad = [r for r in rows if r["label_bad"]]
    return {
        "samples": {"good": len(good), "bad": len(bad)},
        "confusion_matrix": {"true_positive": tp, "false_negative": fn,
                             "true_negative": tn, "false_positive": fp},
        "sensitivity": sens,
        "specificity": _div(tn, tn + fp),
        "precision": prec,
        "accuracy": _div(tp + tn, len(rows)),
        "f1": _div(2 * prec * sens, prec + sens) if prec and sens else None,
        "false_approval_rate": _div(fn, tp + fn),
        "false_rejection_rate": _div(fp, tn + fp),
        "mean_overall_score_good": round(float(np.mean([r["overall"] for r in good])), 1) if good else None,
        "mean_overall_score_bad": round(float(np.mean([r["overall"] for r in bad])), 1) if bad else None,
        "detection_rate_by_defect": {
            d: _div(sum(not r["approved"] for r in bad if r["defect"] == d),
                    sum(r["defect"] == d for r in bad))
            for d in sorted({r["defect"] for r in bad})},
        "false_rejection_by_check": {
            k: _div(sum(not r["checks"][k]["passed"] for r in good), len(good))
            for k in (good[0]["checks"] if good else {})},
        # Scale check: good images should score high and defective ones below 50
        # (50 = the pass/fail boundary) on the SAME 0-100 scale for every feature.
        "score_by_check": {
            k: {"median_good": _median([r["checks"][k]["score"] for r in good]),
                "median_when_defective": _median([
                    r["checks"][k]["score"] for r in bad
                    if CHECK_OF.get(r["defect"]) == k and k in r["checks"]])}
            for k in (good[0]["checks"] if good else {})},
    }


def best_threshold(good, bad, bad_if_below, sens_weight=1.5):
    """Pick the threshold maximising sens_weight*sensitivity + specificity.
    sens_weight > 1 because approving a bad image is worse than asking for a retake."""
    good, bad = np.asarray(good, float), np.asarray(bad, float)
    vals = np.unique(np.concatenate([good, bad]))
    if len(vals) < 2:
        return None
    best_t, best_j = None, -1e9
    for t in (vals[:-1] + vals[1:]) / 2:
        if bad_if_below:
            sens, spec = (bad < t).mean(), (good >= t).mean()
        else:
            sens, spec = (bad > t).mean(), (good <= t).mean()
        j = sens_weight * sens + spec
        if j > best_j:
            best_t, best_j = float(t), j
    return best_t


def calibrate(rows, cfg, sens_weight=1.5):
    def vals(defect, check, metric):
        return [r["checks"][check]["metrics"][metric] for r in rows
                if r["defect"] == defect and check in r["checks"]]
    good = lambda check, metric: vals(None, check, metric)
    new, notes = QualityConfig(**{**cfg.__dict__}), []

    def need(*lists):
        return all(len(x) >= 5 for x in lists)

    g, b = good("clarity", "laplacian_var"), vals("blur", "clarity", "laplacian_var")
    if need(g, b):
        t = best_threshold(g, b, True, sens_weight)
        new.blur_min_var = round(t, 1)
        new.blur_good_var = round(max(float(np.median(g)), t * 1.5), 1)   # best anchor (score 100)
        new.blur_worst_var = round(max(1.0, min(float(np.median(b)), t * 0.5)), 1)  # worst anchor (score 0)
        notes.append(f"blur anchors: worst={new.blur_worst_var}, boundary={new.blur_min_var}, best={new.blur_good_var}")

    g = good("illumination", "mean_gray")
    d, br = vals("dark", "illumination", "mean_gray"), vals("bright", "illumination", "mean_gray")
    lo, hi = new.mean_hard_range
    if need(g, d):
        lo = best_threshold(g, d, True, sens_weight)
    if need(g, br):
        hi = best_threshold(g, br, False, sens_weight)
    # Safety caps: never let calibration reject normal dark-skin / dark-lesion photos
    # or accept washed-out ones, whatever the (possibly synthetic) data suggests.
    lo, hi = min(lo, 70.0), max(hi, 200.0)
    if (lo, hi) != tuple(new.mean_hard_range) and hi - lo > 80:
        new.mean_hard_range = (round(lo, 1), round(hi, 1))
        new.mean_ok_range = (round(lo + 30, 1), round(hi - 30, 1))
        notes.append(f"mean_hard_range={new.mean_hard_range}")

    for metric, attr in (("glare_fraction", "glare_max_fraction"),
                         ("largest_glare_blob", "glare_max_blob_fraction")):
        g, b = good("glare", metric), vals("glare", "glare", metric)
        if need(g, b):
            setattr(new, attr, round(best_threshold(g, b, False, sens_weight), 5))
            notes.append(f"{attr}={getattr(new, attr)}")

    g, b = good("obstruction", "hair_fraction"), vals("hair", "obstruction", "hair_fraction")
    if need(g, b):
        new.hair_max_fraction = round(best_threshold(g, b, False, sens_weight), 5)
        notes.append(f"hair_max_fraction={new.hair_max_fraction}")
    return new, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--good_dir", required=True)
    ap.add_argument("--bad_dir")
    ap.add_argument("--synthesize", action="store_true")
    ap.add_argument("--max_images", type=int, default=300)
    ap.add_argument("--min_side", type=int, default=224, help="min image side Model 2 needs")
    ap.add_argument("--sens_weight", type=float, default=1.5)
    ap.add_argument("--out", default="evaluation")
    a = ap.parse_args()

    good = list_images(a.good_dir, a.max_images)
    bad = list_images(a.bad_dir, a.max_images) if a.bad_dir else []
    if not good:
        raise SystemExit(f"No images found in {a.good_dir}")
    if not a.synthesize and not bad:
        raise SystemExit("Need --bad_dir and/or --synthesize, otherwise there are no bad images to measure.")
    os.makedirs(a.out, exist_ok=True)
    make = lambda: iter_samples(good, bad, a.synthesize)

    base_cfg = QualityConfig(min_side_px=a.min_side)
    before_rows = run(QualityEngine(base_cfg), make())
    before = summarize(before_rows)

    new_cfg, notes = calibrate(before_rows, base_cfg, a.sens_weight)
    after = summarize(run(QualityEngine(new_cfg), make()))

    new_cfg.to_json("calibrated_config.json")
    result = {"before_calibration": before, "after_calibration": after, "changed": notes,
              "note": "Calibrated on synthetic defects unless --bad_dir was given; "
                      "validate on real, hand-labelled images before trusting these numbers."}
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    for tag, m in (("BEFORE", before), ("AFTER", after)):
        print(f"\n=== {tag} calibration ===")
        for k in ("samples", "confusion_matrix", "sensitivity", "specificity", "precision", "accuracy",
                  "f1", "false_approval_rate", "false_rejection_rate",
                  "mean_overall_score_good", "mean_overall_score_bad"):
            print(f"{k:26s} {m[k]}")
        print("detection by defect:", m["detection_rate_by_defect"])
        print("false rejection by check (good images):", m["false_rejection_by_check"])
        print("median score by check (good vs defective; boundary = 50):")
        for k, v in m["score_by_check"].items():
            print(f"  {k:13s} good={v['median_good']}  defective={v['median_when_defective']}")
    print("\nThresholds changed:", notes)
    print("Saved calibrated_config.json and", os.path.join(a.out, "metrics.json"))


if __name__ == "__main__":
    main()