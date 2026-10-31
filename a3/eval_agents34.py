"""
Evaluate Agent 3 (lesion analysis) and Agent 4 (context) on a labelled set.
Target: malignant (mel, bcc, akiec) = 1, benign = 0.
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, precision_recall_curve)
from agent3_lesion_analysis import lesion_analysis_agent
from agent4_context import context_agent

MAL = {"mel", "bcc", "akiec"}


def find_images(dirs):
    found = {}
    for d in dirs:
        for f in os.listdir(d):
            stem, ext = os.path.splitext(f)
            if ext.lower() in (".jpg", ".jpeg", ".png"):
                found[stem] = os.path.join(d, f)
    return found


def at_threshold(y, s, thr):
    pred = (s >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return dict(thr=round(float(thr), 3), precision=p, recall=r, specificity=spec, f1=f1,
                TP=tp, FP=fp, TN=tn, FN=fn)


def report(name, y, s, fixed_thr):
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    print(f"\n=== {name}  (n={len(y)}, malignant={int(y.sum())}) ===")
    print(f"AUROC = {roc_auc_score(y, s):.3f}   Average precision = {average_precision_score(y, s):.3f}"
          f"   (random AP = {y.mean():.3f})")
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9)
    best_f1_thr = t[int(np.argmax(f1))]
    ok90 = np.where(r[:-1] >= 0.90)[0]
    thr90 = t[ok90[-1]] if len(ok90) else t[0]
    rows = {"default": fixed_thr, "best-F1": best_f1_thr, "90%-recall": thr90}
    out = pd.DataFrame({k: at_threshold(y, s, v) for k, v in rows.items()}).T
    print(out.to_string(float_format=lambda x: f"{x:.3f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dirs", nargs="+", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="agents34_results.csv")
    a = ap.parse_args()

    meta = pd.read_csv(a.meta).drop_duplicates("image_id").set_index("image_id")
    paths = find_images(a.img_dirs)
    ids = [i for i in meta.index if i in paths]
    if a.limit:
        ids = ids[:a.limit]
    print(f"{len(ids)} labelled images found")

    rows = []
    for n, iid in enumerate(ids, 1):
        m = meta.loc[iid]
        a3 = lesion_analysis_agent(paths[iid])
        a4 = context_agent(m.get("age"), m.get("sex"), m.get("localization"))
        rows.append({
            "image_id": iid, "dx": m["dx"], "malignant": int(m["dx"] in MAL),
            "a3_score": np.nan if a3["suspicion_score"] is None else a3["suspicion_score"],
            "a3_label": a3["label"], "a3_reliability": a3["reliability"],
            "a3_flags": ";".join(a3["flags"]),
            "a4_score": np.nan if a4["risk_score"] is None else a4["risk_score"],
            "a4_level": a4["risk_level"], "a4_missing": ";".join(a4["missing_fields"]),
        })
        if n % 100 == 0:
            print(f"  {n}/{len(ids)}")

    df = pd.DataFrame(rows)
    df["fused_score"] = np.where(df.a4_score.isna(), df.a3_score, 0.7 * df.a3_score + 0.3 * df.a4_score)
    df.to_csv(a.out, index=False)
    print(f"\nSaved per-image results to {a.out}")
    print(f"A3 failures (no lesion / unreadable): {df.a3_score.isna().sum()}")
    print(f"A4 images with missing metadata: {(df.a4_missing != '').sum()}")

    y = df.malignant.values
    report("Agent 3 suspicion score", y, df.a3_score.values.astype(float), 0.55)
    report("Agent 4 risk score", y, df.a4_score.values.astype(float), 0.60)
    report("Fused (0.7*A3 + 0.3*A4)", y, df.fused_score.values.astype(float), 0.50)

    print("\n=== Mean scores per class ===")
    print(df.groupby("dx")[["a3_score", "a4_score", "fused_score"]].mean().round(3).to_string())

    print("\n=== A3 label distribution per class (rows sum to 1) ===")
    print(pd.crosstab(df.dx, df.a3_label, normalize="index").round(3).to_string())


if __name__ == "__main__":
    main()