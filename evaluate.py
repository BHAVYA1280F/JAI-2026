"""Run the Diagnosis Agent on the 2,000 validation images; report metrics and cache outputs.

python evaluate.py --val_dir PATH --meta HAM10000_metadata.csv --ckpt outputs/diagnosis_agent.pt
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch

from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--ckpt", default="outputs/diagnosis_agent.pt")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(a.ckpt, map_location=dev)
    model = timm.create_model(ck["model_name"], pretrained=False, num_classes=NUM)
    model.load_state_dict(ck["state_dict"]); model = model.to(dev).to(memory_format=torch.channels_last)
    T = ck["temperature"]

    df = load_split(a.val_dir, a.meta)
    X = load_array(df["path"], out / f"val_cache_{len(df)}.npy")
    y = df["y"].values

    lg = predict_logits(model, X, dev, tta=True)
    np.save(out / "val_logits.npy", lg)          # (views, N, C): re-tune Level-2 thresholds offline
    raw = compute_metrics(y, softmax(lg.mean(0)))
    ag = agent_outputs(lg, T)
    cal = compute_metrics(y, ag["probs"])
    single = compute_metrics(y, softmax(lg[0]))

    print(f"\n{'metric':28s}{'no TTA':>10s}{'TTA':>10s}{'TTA+calib':>11s}")
    for k in ["accuracy", "balanced_accuracy", "macro_f1", "macro_auroc",
              "melanoma_recall", "malignant_sensitivity", "malignant_specificity", "ece"]:
        print(f"{k:28s}{single[k]:10.4f}{raw[k]:10.4f}{cal[k]:11.4f}")
    print("\nPer-class (calibrated):")
    for c in CLASSES:
        r = cal["per_class"][c]
        print(f"  {c:6s} P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1-score']:.3f}  n={int(r['support'])}")

    pred = ag["probs"].argmax(1)
    res = pd.DataFrame({"image_id": df["image_id"], "true": [CLASSES[i] for i in y],
                        "pred": [CLASSES[i] for i in pred], "confidence": ag["probs"].max(1),
                        "entropy": ag["entropy"], "tta_agreement": ag["tta_agreement"],
                        "tta_std": ag["tta_std"], "p_malignant": ag["p_malignant"]})
    for k, c in enumerate(CLASSES):
        res[f"p_{c}"] = ag["probs"][:, k]
    res.to_csv(out / "val_predictions.csv", index=False)
    (out / "val_metrics.json").write_text(json.dumps(dict(no_tta=single, tta=raw, calibrated=cal), indent=1))

    # confusion matrix
    cm = np.array(cal["confusion_matrix"]); cmn = cm / cm.sum(1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(cmn, cmap="Blues")
    ax.set_xticks(range(NUM), CLASSES); ax.set_yticks(range(NUM), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(NUM):
        for j in range(NUM):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)

    # reliability diagram (before vs after calibration)
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot([0, 1], [0, 1], "k--")
    for name, p in [("before", softmax(lg.mean(0))), ("after", ag["probs"])]:
        conf, ok = p.max(1), p.argmax(1) == y
        bins = np.linspace(0, 1, 11); mid, acc = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5: mid.append(conf[m].mean()); acc.append(ok[m].mean())
        ax.plot(mid, acc, "o-", label=f"{name} (ECE {ece_score(p, y):.3f})")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "reliability.png", dpi=150); plt.close(fig)
    print("\nSaved val_predictions.csv, val_metrics.json, val_logits.npy, confusion_matrix.png, reliability.png")


if __name__ == "__main__":
    main()
