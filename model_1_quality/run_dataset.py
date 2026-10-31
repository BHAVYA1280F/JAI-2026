"""Run Model 1 on every image in a folder and report what it did.

    python -m model_1_quality.run_dataset --dir data\\all
    python -m model_1_quality.run_dataset --dir data\\all --make_template 100
    python -m model_1_quality.run_dataset --dir data\\all --labels review_template.csv

Without labels it reports the approval rate (how many images the gate accepts),
NOT accuracy: accuracy needs someone to say which images are really good or bad.
With --labels (a CSV with columns `filename` and `true_label` = good/bad) it also
reports accuracy, sensitivity, specificity and lists the mistakes.
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import shutil
import time
from collections import Counter

import cv2

from .config import QualityConfig
from .quality_engine import QualityEngine

EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def load_engine(path):
    if path and os.path.exists(path):
        print("Using thresholds from", path)
        return QualityEngine(QualityConfig.from_json(path))
    print("Using default thresholds (no calibrated_config.json found)")
    return QualityEngine()


def read_labels(path):
    labels = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            v = (row.get("true_label") or row.get("label") or "").strip().lower()
            if v in ("good", "bad"):
                labels[row["filename"].strip()] = v
    return labels


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder with the images (not searched recursively)")
    ap.add_argument("--config", default="calibrated_config.json")
    ap.add_argument("--labels", help="CSV with filename,true_label (good/bad) to measure accuracy")
    ap.add_argument("--make_template", type=int, metavar="N",
                    help="write review_template.csv with N random images for you to label")
    ap.add_argument("--copy_rejected", action="store_true", help="copy rejected images to <out>/rejected")
    ap.add_argument("--out", default="evaluation_run")
    a = ap.parse_args()

    paths = sorted(p for e in EXTS for p in glob.glob(os.path.join(a.dir, e)))
    if not paths:
        raise SystemExit(f"No images found in {a.dir}")
    os.makedirs(a.out, exist_ok=True)
    engine = load_engine(a.config)

    rows, unreadable, t0 = [], [], time.time()
    for i, p in enumerate(paths, 1):
        img = cv2.imread(p)
        if img is None:
            unreadable.append(os.path.basename(p))
            continue
        r = engine.analyze(img, os.path.basename(p))
        fails = [d["type"] for d in r["defects"] if d["severity"] == "FAIL"]
        warns = [d["type"] for d in r["defects"] if d["severity"] == "WARN"]
        rows.append({"filename": os.path.basename(p), "path": p,
                     "decision": r["decision"]["status"], "severity": r["severity"],
                     "score": r["score"], "failed_defects": ";".join(fails),
                     "warning_defects": ";".join(warns), "reasons": " | ".join(r["reasons"])})
        if i % 100 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)} images done", flush=True)
    secs = time.time() - t0

    cols = ["filename", "decision", "severity", "score", "failed_defects", "warning_defects", "reasons"]
    with open(os.path.join(a.out, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    approved = [r for r in rows if r["decision"] == "APPROVED"]
    rejected = [r for r in rows if r["decision"] == "REJECTED"]
    sev = Counter(r["severity"] for r in rows)
    defects = Counter(d for r in rows for d in r["failed_defects"].split(";") if d)
    scores = sorted(r["score"] for r in rows)

    print(f"\n=== Model 1 on {n} images ({secs:.0f} s) ===")
    if unreadable:
        print(f"Could not read {len(unreadable)} files: {unreadable[:5]}")
    print(f"Approved: {len(approved)} ({pct(len(approved), n)})")
    print(f"Rejected: {len(rejected)} ({pct(len(rejected), n)})")
    print(f"Severity: PASS {sev['PASS']}, WARN {sev['WARN']}, FAIL {sev['FAIL']}")
    print("Rejections by defect (an image can have several):")
    for d, c in defects.most_common():
        print(f"  {d:18s} {c}")
    if scores:
        print(f"Quality score: mean {sum(scores) / n:.1f}, median {scores[n // 2]:.1f}, "
              f"min {scores[0]:.1f}, max {scores[-1]:.1f}")

    summary = {"images": n, "approved": len(approved), "rejected": len(rejected),
               "unreadable": len(unreadable), "seconds": round(secs, 1),
               "severity": {k: sev.get(k, 0) for k in ("PASS", "WARN", "FAIL")},
               "rejections_by_defect": dict(defects.most_common()),
               "score": {"mean": round(sum(scores) / n, 1), "median": scores[n // 2],
                         "min": scores[0], "max": scores[-1]} if n else None,
               "thresholds": a.config if os.path.exists(a.config) else "defaults",
               "accuracy": None}

    if a.copy_rejected and rejected:
        dest = os.path.join(a.out, "rejected")
        os.makedirs(dest, exist_ok=True)
        for r in rejected:
            shutil.copy(r["path"], dest)
        print(f"Copied {len(rejected)} rejected images to {dest} so you can look at them.")

    if a.make_template:
        sample = random.Random(0).sample(rows, min(a.make_template, n))
        path = os.path.join(a.out, "review_template.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "true_label", "model_decision", "score", "failed_defects"])
            for r in sample:
                w.writerow([r["filename"], "", r["decision"], r["score"], r["failed_defects"]])
        print(f"\nWrote {path} with {len(sample)} random images. Open each image, type good or bad "
              f"in the true_label column, save, then run again with --labels {path}")

    if a.labels:
        labels = read_labels(a.labels)
        pairs = [(r, labels[r["filename"]]) for r in rows if r["filename"] in labels]
        if not pairs:
            print("\nNo usable labels found (true_label must be good or bad).")
        else:
            tp = sum(l == "bad" and r["decision"] == "REJECTED" for r, l in pairs)
            fn = sum(l == "bad" and r["decision"] == "APPROVED" for r, l in pairs)
            tn = sum(l == "good" and r["decision"] == "APPROVED" for r, l in pairs)
            fp = sum(l == "good" and r["decision"] == "REJECTED" for r, l in pairs)
            m = len(pairs)
            acc = (tp + tn) / m
            moe = 1.96 * math.sqrt(acc * (1 - acc) / m)
            print(f"\n=== Accuracy on {m} labelled images ===")
            print(f"Correct: {tp + tn} of {m}  ->  accuracy {acc:.1%} (give or take {moe:.1%})")
            print(f"Sensitivity (bad images rejected): {pct(tp, tp + fn)}   [{tp} of {tp + fn} bad]")
            print(f"Specificity (good images approved): {pct(tn, tn + fp)}   [{tn} of {tn + fp} good]")
            print(f"Mistakes: {fn} bad images approved, {fp} good images rejected")
            wrong = [(r["filename"], l, r["decision"], r["failed_defects"]) for r, l in pairs
                     if (l == "bad") != (r["decision"] == "REJECTED")]
            with open(os.path.join(a.out, "mistakes.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["filename", "true_label", "model_decision", "failed_defects"])
                w.writerows(wrong)
            print("Saved the mistakes to", os.path.join(a.out, "mistakes.csv"))
            summary["accuracy"] = {
                "labelled": m, "correct": tp + tn, "accuracy": round(acc, 4), "margin": round(moe, 4),
                "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
                "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
                "true_positive": tp, "false_negative": fn, "true_negative": tn, "false_positive": fp}

    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved", os.path.join(a.out, "results.csv"), "and", os.path.join(a.out, "summary.json"))
    print("Open the Model performance page in the website to see this.")


if __name__ == "__main__":
    main()