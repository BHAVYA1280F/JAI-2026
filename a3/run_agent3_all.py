"""
Run Agent 3 on any image folder and (if labels exist) print accuracy, precision, recall, etc.

Examples:
  python run_agent3_all.py
        (it will ask you for the folder, and optionally a labels CSV)
  python run_agent3_all.py --data "C:/path/to/images" --meta "C:/path/to/HAM10000_metadata.csv" --limit 100
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix,
                             precision_recall_curve)
from agent3_lesion_analysis import lesion_analysis_agent

ap = argparse.ArgumentParser()
ap.add_argument("--data", help="folder with images (subfolders are included)")
ap.add_argument("--meta", help="optional labels CSV (needs an image-name column and a label column such as dx)")
ap.add_argument("--limit", type=int, default=0, help="only run the first N images")
ap.add_argument("--out", default="agent3_results.csv")
args = ap.parse_args()

clean = lambda s: s.strip().strip('"').strip("'")
GOOD = clean(args.data or input("Path to your image folder: "))
META = clean(args.meta) if args.meta else clean(input("Path to labels CSV (press Enter to auto-search): "))
LIMIT = args.limit

MAL = {"mel", "bcc", "akiec", "melanoma", "basal cell carcinoma", "actinic keratoses",
       "malignant", "1", "1.0", "true", "cancer", "positive"}
ID_COLS = ["image_id", "image", "image_name", "id", "filename", "file", "isic_id"]
LAB_COLS = ["dx", "label", "diagnosis", "class", "target", "category"]

if not os.path.isdir(GOOD):
    sys.exit(f"Folder not found: {GOOD}")

# 1. images
imgs = {}
for root, _, files in os.walk(GOOD):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            imgs[os.path.splitext(f)[0]] = os.path.join(root, f)
print("Images found:", len(imgs))
if not imgs:
    sys.exit("No images (.jpg/.jpeg/.png) in that folder.")


def read_labels(csv_path):
    try:
        c = pd.read_csv(csv_path)
    except Exception as e:
        print("Could not read", csv_path, "->", e)
        return {}
    cols = {x.lower(): x for x in c.columns}
    idc = next((cols[k] for k in ID_COLS if k in cols), None)
    lc = next((cols[k] for k in LAB_COLS if k in cols), None)
    if not (idc and lc):
        print("CSV", csv_path, "has no recognisable id/label columns. Columns:", list(c.columns))
        return {}
    ids = c[idc].astype(str).map(lambda s: os.path.splitext(s)[0])
    print("CSV", csv_path, "-> matches", int(ids.isin(imgs.keys()).sum()), "images")
    return dict(zip(ids, c[lc].astype(str).str.lower().str.strip()))


# 2. labels: given CSV > CSVs near the image folder > class-named subfolders
labels = {}
if META:
    labels = read_labels(META)
else:
    search_roots = {GOOD, os.path.dirname(os.path.abspath(GOOD))}
    for base in search_roots:
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith(".csv") and not f.startswith("agent3") and not labels:
                    got = read_labels(os.path.join(root, f))
                    if any(k in imgs for k in got):
                        labels = got
if not any(k in labels for k in imgs):
    labels = {}
    for k, p in imgs.items():
        if os.path.normcase(os.path.dirname(p)) != os.path.normcase(os.path.abspath(GOOD)):
            labels[k] = os.path.basename(os.path.dirname(p)).lower()
print("Images with a label:", sum(k in labels for k in imgs))

# 3. run Agent 3
keys = list(imgs)[:LIMIT] if LIMIT else list(imgs)
rows = []
for n, k in enumerate(keys, 1):
    r = lesion_analysis_agent(imgs[k])
    rows.append({"image": k, "dx": labels.get(k),
                 "symmetry": r["symmetry"], "color": r["color"], "border": r["border"],
                 "suspicion": r["suspicion"], "a3_label": r["label"],
                 "a3_score": r["suspicion_score"], "reliability": r["reliability"],
                 "flags": ";".join(r["flags"])})
    if n % 50 == 0:
        print(n, "/", len(keys))
df = pd.DataFrame(rows)
df.to_csv(args.out, index=False)
print("\nSaved", args.out, "| failures:", int(df.a3_score.isna().sum()))
print(df[["image", "symmetry", "color", "border", "suspicion"]].head(10).to_string(index=False))
print("\nAverages:", df[["symmetry", "color", "border", "suspicion"]].mean().round(3).to_dict())
print(df.a3_label.value_counts().to_string())

# 4. metrics
d = df.dropna(subset=["dx", "a3_score"]).copy()
if d.empty:
    print("\nNo labels found, so no precision/accuracy. Re-run and give the labels CSV path when asked.")
    sys.exit(0)
d["y"] = d.dx.isin(MAL).astype(int)
print("\nLabelled images:", len(d), "| malignant:", int(d.y.sum()), "| benign:", int((d.y == 0).sum()))
print("\nMean per class:\n", d.groupby("dx")[["symmetry", "color", "border", "suspicion"]].mean().round(3).to_string())
if d.y.nunique() < 2:
    print("\nOnly one class present, so precision/AUROC are not meaningful.")
    sys.exit(0)
print("\nAUROC:", round(roc_auc_score(d.y, d.a3_score), 3))
p, r, t = precision_recall_curve(d.y, d.a3_score)
best = t[int(np.argmax(2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9)))]
for thr in (0.33, 0.55, float(best)):
    pred = (d.a3_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(d.y, pred, labels=[0, 1]).ravel()
    print(f"\n--- threshold {thr:.3f}{' (best F1)' if thr == float(best) else ''} ---")
    print("Accuracy:", round(accuracy_score(d.y, pred), 3),
          "| Balanced acc:", round(balanced_accuracy_score(d.y, pred), 3))
    print("Precision:", round(precision_score(d.y, pred, zero_division=0), 3),
          "| Recall:", round(recall_score(d.y, pred, zero_division=0), 3),
          "| Specificity:", round(tn / (tn + fp), 3) if tn + fp else 0,
          "| F1:", round(f1_score(d.y, pred, zero_division=0), 3))
    print(f"TP {tp}  FP {fp}  TN {tn}  FN {fn}")