import argparse
import os
import numpy as np
import pandas as pd
from agent3_lesion_analysis import lesion_analysis_agent

MAL = {"mel", "bcc", "akiec"}
CLASSES = {"nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"}

ap = argparse.ArgumentParser()
ap.add_argument("--folder", required=True)
ap.add_argument("--meta", default=None, help="optional CSV with image_id and dx columns")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="agent3_results.csv")
a = ap.parse_args()

# collect all images, including subfolders
paths = []
for root, _, files in os.walk(a.folder):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(root, f))
if a.limit:
    paths = paths[:a.limit]
print(len(paths), "images found")

labels = {}
if a.meta:
    m = pd.read_csv(a.meta).drop_duplicates("image_id")
    labels = dict(zip(m["image_id"], m["dx"]))

rows = []
for n, p in enumerate(paths, 1):
    stem = os.path.splitext(os.path.basename(p))[0]
    folder_name = os.path.basename(os.path.dirname(p)).lower()
    dx = labels.get(stem) or (folder_name if folder_name in CLASSES else None)
    r = lesion_analysis_agent(p)
    rows.append({"image": stem, "dx": dx, "a3_label": r["label"],
                 "a3_score": r["suspicion_score"], "reliability": r["reliability"],
                 "flags": ";".join(r["flags"])})
    if n % 50 == 0:
        print(n, "/", len(paths))

df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
print("\nSaved", a.out)
print("Failures:", df.a3_score.isna().sum())
print(df.a3_label.value_counts())

lab = df.dropna(subset=["dx", "a3_score"]).copy()
if len(lab) == 0:
    print("\nNo labels found, so only scores were saved.")
else:
    from sklearn.metrics import (roc_auc_score, confusion_matrix,
                                 precision_recall_fscore_support)
    lab["y"] = lab.dx.isin(MAL).astype(int)
    print("\nMean score per class:")
    print(lab.groupby("dx").a3_score.mean().round(3))
    if lab.y.nunique() == 2:
        print("\nAUROC:", round(roc_auc_score(lab.y, lab.a3_score), 3))
        for thr in (0.33, 0.55):
            pred = (lab.a3_score >= thr).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                lab.y, pred, average="binary", zero_division=0)
            tn, fp, fn, tp = confusion_matrix(lab.y, pred, labels=[0, 1]).ravel()
            print(f"thr {thr}: precision {p:.3f} recall {r:.3f} "
                  f"specificity {tn/(tn+fp):.3f} F1 {f1:.3f}  TP {tp} FP {fp} TN {tn} FN {fn}")
    else:
        print("\nOnly one class (malignant or benign) in the labels, so no AUROC.")