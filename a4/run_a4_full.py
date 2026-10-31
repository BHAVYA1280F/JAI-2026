"""
Run Agent 4 on the full HAM10000 metadata and print/save all results.

Usage:
  python run_a4_full.py --meta HAM10000_metadata.csv
  python run_a4_full.py --meta HAM10000_metadata.csv --preds outputs/val_predictions.csv [--pred_col NAME]
"""
import argparse, json, os
import pandas as pd
from agent4_context import context_agent, MALIGNANT

ap = argparse.ArgumentParser()
ap.add_argument("--meta", default="HAM10000_metadata.csv")
ap.add_argument("--preds", default=None, help="val_predictions.csv from evaluate.py (Agent 2 output)")
ap.add_argument("--pred_col", default=None, help="name of the predicted-label column in --preds")
ap.add_argument("--out", default="a4_outputs")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

meta = pd.read_csv(args.meta, dtype=str, keep_default_na=False)

# --- choose where predicted_class comes from ---
if args.preds:
    p = pd.read_csv(args.preds, dtype=str, keep_default_na=False)
    col = args.pred_col
    if col is None:
        for c in ["pred", "pred_label", "predicted", "predicted_label", "pred_class", "label_pred", "prediction"]:
            if c in p.columns:
                col = c; break
    if col is None or col not in p.columns:
        raise SystemExit(f"Could not find the predicted-label column. Columns are: {list(p.columns)}. Use --pred_col.")
    key = "image_id" if "image_id" in p.columns else p.columns[0]
    p = p[[key, col]].rename(columns={key: "image_id", col: "predicted_class"})
    meta = meta.merge(p, on="image_id", how="inner")
    print(f"Using Agent 2 predictions from column '{col}': {len(meta)} images matched")
else:
    meta["predicted_class"] = meta["dx"]
    print("No --preds given: using ground-truth dx as predicted_class (tests the flags only, not the real pipeline)")

# --- run A4 on every row ---
rows = []
for _, r in meta.iterrows():
    o = context_agent(age=r["age"], sex=r["sex"], location=r["localization"], predicted_class=r["predicted_class"])
    rows.append({
        "image_id": r["image_id"], "lesion_id": r.get("lesion_id", ""), "dx": r["dx"],
        "predicted_class": r["predicted_class"], "age": r["age"], "sex": r["sex"], "localization": r["localization"],
        "risk_score": o["risk_score"], "risk_level": o["risk_level"],
        "context_reliability": o["context_reliability"],
        "missing_fields": ";".join(o["missing_fields"]),
        "context_flags": ";".join(o["context_flags"]),
        "evidence": " | ".join(o["evidence"]),
    })
res = pd.DataFrame(rows)
res["is_malignant"] = res["dx"].isin(MALIGNANT)
res.to_csv(f"{args.out}/a4_context_output.csv", index=False)

# --- summary ---
lines = []
def log(s=""):
    print(s); lines.append(str(s))

log(f"\nImages processed: {len(res)}")
log("\nMean risk_score by true class:")
log(res.groupby("dx")["risk_score"].mean().round(3).sort_values().to_string())
log("\nTrue class x risk_level:")
log(pd.crosstab(res["dx"], res["risk_level"]).to_string())
log("\nMissing fields (count):")
log(res["missing_fields"].replace("", "none").value_counts().to_string())
log("\nContext flags (count):")
log(res["context_flags"].replace("", "none").str.split(";").explode().value_counts().to_string())
log("\ncontext_reliability distribution:")
log(res["context_reliability"].value_counts().sort_index().to_string())

summary = {"n": int(len(res))}
scored = res.dropna(subset=["risk_score"])
if scored["is_malignant"].nunique() == 2:
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(scored["is_malignant"], scored["risk_score"])
    summary["auroc_risk_score_vs_malignant"] = round(float(auc), 3)
    log(f"\nAUROC of risk_score for malignant (mel+bcc+akiec) vs benign: {auc:.3f}  (0.5 = no signal)")
    for name, lv in [("HIGH only", ["HIGH"]), ("MEDIUM or HIGH", ["MEDIUM", "HIGH"])]:
        flagged = res["risk_level"].isin(lv)
        sens = flagged[res.is_malignant].mean(); frac = flagged.mean()
        summary[f"malignant_recall_{name}"] = round(float(sens), 3)
        log(f"Risk level {name}: catches {sens:.1%} of malignant cases, flags {frac:.1%} of all images")

with open(f"{args.out}/a4_summary.txt", "w") as f:
    f.write("\n".join(lines))
json.dump(summary, open(f"{args.out}/a4_summary.json", "w"), indent=2)
print(f"\nSaved to {args.out}/: a4_context_output.csv, a4_summary.txt, a4_summary.json")
