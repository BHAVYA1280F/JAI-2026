"""Train the Diagnosis Agent. Augmentation runs on GPU (no DataLoader workers needed).

python train.py --train_dir PATH --val_dir PATH --meta HAM10000_metadata.csv
(val_dir is only used here to confirm paths; it is never trained on or tuned on.)
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from common import *


def gpu_augment(x):
    """x: (B,3,256,256) float in [0,1] -> (B,3,224,224) augmented, normalised."""
    B, dev = x.shape[0], x.device
    r = lambda: torch.rand(B, device=dev)
    ang = (r() * 2 - 1) * math.pi                      # any rotation
    zoom = 0.65 + 0.35 * r()                           # visible fraction (random crop)
    sx, sy = (r() < .5).float() * 2 - 1, (r() < .5).float() * 2 - 1   # random flips
    tx, ty = (r() * 2 - 1) * (1 - zoom), (r() * 2 - 1) * (1 - zoom)
    c, s = torch.cos(ang) * zoom, torch.sin(ang) * zoom
    theta = torch.stack([torch.stack([c * sx, -s * sy, tx], 1),
                         torch.stack([s * sx, c * sy, ty], 1)], 1)
    grid = F.affine_grid(theta, (B, 3, IMG_SIZE, IMG_SIZE), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    j = lambda: (0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=dev))
    x = x * j()                                                        # brightness
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * j() + m         # contrast
    g = x.mean(1, keepdim=True); x = (x - g) * j() + g                 # saturation
    return normalize(x.clamp(0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="debug: use N train images")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")

    df = load_split(a.train_dir, a.meta)
    if a.limit:
        df = df.sample(a.limit, random_state=0).reset_index(drop=True)
    X = load_array(df["path"], out / f"train_cache_{len(df)}.npy")
    y = df["y"].values

    # Internal holdout (10%), grouped by lesion_id so the same lesion never spans both sets.
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=a.seed)
    tr, ho = next(sgkf.split(X, y, groups=df["lesion_id"]))
    (out / "holdout_ids.txt").write_text("\n".join(df.loc[ho, "image_id"]))
    Xtr, ytr, Xho, yho = X[tr], y[tr], X[ho], y[ho]
    counts = np.bincount(ytr, minlength=NUM)
    print("train", len(tr), "holdout", len(ho), "class counts", dict(zip(CLASSES, counts)))

    model = timm.create_model(a.model, pretrained=True, num_classes=NUM, drop_path_rate=0.1)
    model = model.to(dev).to(memory_format=torch.channels_last)
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    head = [p for p in model.parameters() if id(p) in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.lr * 10}],
                            weight_decay=0.05)

    # sqrt inverse-frequency class weights + label smoothing (moderate imbalance handling)
    w = torch.tensor((len(ytr) / (NUM * counts)) ** 0.5, dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)

    steps_per_ep = len(tr) // a.bs
    total, warm = a.epochs * steps_per_ep, steps_per_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    best, bad, hist = -1, 0, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); perm = np.random.permutation(len(tr)); run = 0.0
        for i in range(steps_per_ep):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            xb = torch.from_numpy(Xtr[idx]).to(dev).permute(0, 3, 1, 2).float() / 255
            xb = gpu_augment(xb).contiguous(memory_format=torch.channels_last)
            yb = torch.from_numpy(ytr[idx]).to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = crit(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
        pred = predict_logits(model, Xho, dev, tta=False)[0].argmax(1)
        f1 = f1_score(yho, pred, average="macro")
        bacc = balanced_accuracy_score(yho, pred)
        mel = recall_score(yho == C2I["mel"], pred == C2I["mel"])
        hist.append(dict(epoch=ep + 1, loss=run / steps_per_ep, macro_f1=f1, bal_acc=bacc, mel_recall=mel))
        print(f"ep {ep+1:02d}  loss {run/steps_per_ep:.4f}  holdout macroF1 {f1:.4f}  "
              f"balAcc {bacc:.4f}  melRecall {mel:.3f}  ({time.time()-t0:.0f}s)")
        score = f1 + bacc
        if score > best:
            best, bad = score, 0
            torch.save(model.state_dict(), out / "best_state.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop"); break

    # Calibrate: temperature scaling fitted on TTA logits of the internal holdout
    model.load_state_dict(torch.load(out / "best_state.pt", map_location=dev))
    lg = predict_logits(model, Xho, dev, tta=True)
    T = fit_temperature(lg.mean(0), yho)
    print(f"temperature T = {T:.3f}  ECE {ece_score(softmax(lg.mean(0)), yho):.4f} -> "
          f"{ece_score(softmax(lg.mean(0) / T), yho):.4f}")
    torch.save(dict(state_dict=model.state_dict(), model_name=a.model, temperature=T,
                    classes=CLASSES), out / "diagnosis_agent.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print("saved", out / "diagnosis_agent.pt")


if __name__ == "__main__":
    main()
