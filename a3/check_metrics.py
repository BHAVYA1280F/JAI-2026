import sys
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, confusion_matrix)

path = sys.argv[1] if len(sys.argv) > 1 else "agent3_results.csv"
df = pd.read_csv(path).dropna(subset=["dx", "a3_score"])
df["y"] = df.dx.isin(["mel", "bcc", "akiec"]).astype(int)

print("Images used:", len(df), "| malignant:", df.y.sum(), "| benign:", (df.y == 0).sum())
print("AUROC:", round(roc_auc_score(df.y, df.a3_score), 3))

for thr in (0.33, 0.45, 0.55):
    pred = (df.a3_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(df.y, pred, labels=[0, 1]).ravel()
    print(f"\n--- threshold {thr} ---")
    print("Accuracy          :", round(accuracy_score(df.y, pred), 3))
    print("Balanced accuracy :", round(balanced_accuracy_score(df.y, pred), 3))
    print("Precision         :", round(precision_score(df.y, pred, zero_division=0), 3))
    print("Recall            :", round(recall_score(df.y, pred, zero_division=0), 3))
    print("Specificity       :", round(tn / (tn + fp), 3))
    print("F1                :", round(f1_score(df.y, pred, zero_division=0), 3))
    print(f"Confusion matrix  : TP {tp}  FP {fp}  TN {tn}  FN {fn}")