"""Calibrate + evaluate Model 1 on a dataset.

    python -m model_1_quality.evaluate --good_dir data/good [--bad_dir data/bad] \
        [--synthesize] [--out evaluation]

Positive class = BAD image. Sensitivity = bad images correctly rejected.
"""
import argparse
import copy
import glob
import json
import os
from collections import Counter

import cv2
import numpy as np

from .config import QualityConfig
from .quality_engine import QualityEngine

DEFECTS = ["blur", "dark", "bright", "glare", "lowres", "hair"]
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
                     "status": r["status"], "checks": r["quality_analysis"]})
    return rows


def _div(a, b):
    return round(a / b, 4) if b else None


def summarize(rows):
    tp = sum(r["label_bad"] and not r["approved"] for r in rows)      # bad, rejected
    fn = sum(r["label_bad"] and r["approved"] for r in rows)          # bad, approved (dangerous)
    tn = sum(not r["label_bad"] and r["approved"] for r in rows)      # good, approved
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
        "status_distribution_good": dict(Counter(r["status"] for r in good)),
        "status_distribution_bad": dict(Counter(r["status"] for r in bad)),
        "detection_rate_by_defect": {
            d: _div(sum(not r["approved"] for r in bad if r["defect"] == d),
                    sum(r["defect"] == d for r in bad))
            for d in sorted({r["defect"] for r in bad})},
        "false_rejection_by_check": {
            k: _div(sum(not r["checks"][k]["passed"] for r in good), len(good))
            for k in (good[0]["checks"] if good else {})},
    }


def best_threshold(good, bad, bad_if_below, sens_weight=1.5):
    """Maximise sens_weight*sensitivity + specificity. sens_weight > 1 because
    approving a bad image is worse than asking for a retake."""
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
    new, notes = copy.deepcopy(cfg), []

    def need(*lists):
        return all(len(x) >= 5 for x in lists)

    def apply(attr, t, digits):
        if t is not None:
            setattr(new, attr, round(t, digits))
            notes.append(f"{attr}={getattr(new, attr)}")
            return True
        return False

    g, b = good("clarity", "laplacian_var"), vals("blur", "clarity", "laplacian_var")
    if need(g, b):
        t = best_threshold(g, b, True, sens_weight)
        if t is not None:
            new.blur_min_var = round(t, 1)
            new.blur_good_var = round(max(float(np.median(g)), t * 1.5), 1)
            notes.append(f"blur_min_var={new.blur_min_var}")

    g = good("illumination", "mean_gray")
    d, br = vals("dark", "illumination", "mean_gray"), vals("bright", "illumination", "mean_gray")
    lo, hi = new.mean_hard_range
    if need(g, d):
        t = best_threshold(g, d, True, sens_weight)
        lo = t if t is not None else lo
    if need(g, br):
        t = best_threshold(g, br, False, sens_weight)
        hi = t if t is not None else hi
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
            apply(attr, best_threshold(g, b, False, sens_weight), 5)

    g, b = good("obstruction", "hair_fraction"), vals("hair", "obstruction", "hair_fraction")
    if need(g, b):
        apply("hair_max_fraction", best_threshold(g, b, False, sens_weight), 5)
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
    ap.add_argument("--config_out", default="calibrated_config.json")
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

    new_cfg.to_json(a.config_out)
    result = {"before_calibration": before, "after_calibration": after, "changed": notes,
              "note": "Calibrated on synthetic defects unless --bad_dir was given; "
                      "validate on real, hand-labelled images before trusting these numbers."}
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    for tag, m in (("BEFORE", before), ("AFTER", after)):
        print(f"\n=== {tag} calibration ===")
        for k in ("samples", "confusion_matrix", "sensitivity", "specificity", "precision", "accuracy",
                  "f1", "false_approval_rate", "false_rejection_rate",
                  "mean_overall_score_good", "mean_overall_score_bad",
                  "status_distribution_good", "status_distribution_bad"):
            print(f"{k:26s} {m[k]}")
        print("detection by defect:", m["detection_rate_by_defect"])
        print("false rejection by check (good images):", m["false_rejection_by_check"])
    print("\nThresholds changed:", notes)
    print("Saved", a.config_out, "and", os.path.join(a.out, "metrics.json"))


if __name__ == "__main__":
    main()