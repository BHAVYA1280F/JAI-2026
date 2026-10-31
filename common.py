"""Shared utilities for the DermaGuard Diagnosis Agent."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from tqdm import tqdm

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
C2I = {c: i for i, c in enumerate(CLASSES)}
NUM = len(CLASSES)
MALIGNANT = [C2I[c] for c in ("akiec", "bcc", "mel")]  # malignant / pre-malignant
IMG_SIZE = 224       # network input
CACHE_SIZE = 256     # cached resolution (augmentation crops from this)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- data
def index_images(folder):
    exts = {".jpg", ".jpeg", ".png"}
    return {p.stem: p for p in Path(folder).rglob("*") if p.suffix.lower() in exts}


def load_split(folder, meta_csv):
    """Rows of the metadata CSV whose image file exists in `folder`."""
    meta = pd.read_csv(meta_csv)
    if "lesion_id" not in meta.columns:
        meta["lesion_id"] = meta["image_id"]
    idx = index_images(folder)
    df = meta[meta["image_id"].isin(idx)].copy()
    df["path"] = df["image_id"].map(idx)
    df["y"] = df["dx"].map(C2I)
    print(f"[{folder}] {len(idx)} files, {len(df)} matched to metadata")
    return df.reset_index(drop=True)


def shades_of_gray(img, power=6):
    """Color constancy: removes illumination colour cast (common in dermoscopy)."""
    x = img.astype(np.float32)
    illum = np.power(np.mean(np.power(x, power), axis=(0, 1)), 1.0 / power)
    illum = illum / (np.linalg.norm(illum) + 1e-8)
    x = x / (illum * np.sqrt(3) + 1e-8)
    return np.clip(x, 0, 255).astype(np.uint8)


def prep_image(img: Image.Image) -> np.ndarray:
    """PIL -> uint8 (256,256,3), colour-normalised. Used in training AND inference."""
    img = img.convert("RGB").resize((CACHE_SIZE, CACHE_SIZE), Image.LANCZOS)
    return shades_of_gray(np.asarray(img))


def load_array(paths, cache=None):
    if cache and Path(cache).exists():
        return np.load(cache)

    def f(p):
        with Image.open(p) as im:
            return prep_image(im)

    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(tqdm(ex.map(f, paths), total=len(paths), desc="loading")))
    if cache:
        np.save(cache, arr)
    return arr


# ---------------------------------------------------------------- tensors
def normalize(x):
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def to_input(batch_uint8, device):
    """uint8 (B,256,256,3) numpy -> normalised float (B,3,224,224) on device."""
    x = torch.from_numpy(batch_uint8).to(device).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                      antialias=True, align_corners=False)
    return normalize(x)


@torch.no_grad()
def predict_logits(model, X, device, bs=64, tta=True):
    """Returns logits of shape (views, N, C). TTA = identity + 3 flips."""
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        x = to_input(X[i:i + bs], device).contiguous(memory_format=torch.channels_last)
        views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)] if tta else [x]
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outs.append(torch.stack([model(v).float() for v in views]).cpu())
    return torch.cat(outs, 1).numpy()


# ---------------------------------------------------------------- calibration / uncertainty
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def fit_temperature(logits, y):
    from scipy.optimize import minimize_scalar

    def nll(log_t):
        p = softmax(logits / np.exp(log_t))
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))

    return float(np.exp(minimize_scalar(nll, bounds=(-2, 3), method="bounded").x))


def ece_score(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def agent_outputs(logit_views, T):
    """Everything the downstream Triage agent needs, from TTA logits (V,N,C)."""
    V, N, _ = logit_views.shape
    probs = softmax(logit_views.mean(0) / T)
    top = probs.argmax(1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(1) / np.log(NUM)  # 0..1
    vp = softmax(logit_views / T)                                     # V,N,C
    agree = (vp.argmax(2) == top[None]).mean(0)                       # view agreement
    idx = np.broadcast_to(top[None, :, None], (V, N, 1))
    std = np.take_along_axis(vp, idx, 2)[..., 0].std(0)               # top-class instability
    return dict(probs=probs, entropy=entropy, tta_agreement=agree, tta_std=std,
                p_malignant=probs[:, MALIGNANT].sum(1))


# ---------------------------------------------------------------- metrics
def compute_metrics(y, probs):
    pred = probs.argmax(1)
    rep = classification_report(y, pred, labels=range(NUM), target_names=CLASSES,
                                output_dict=True, zero_division=0)
    auroc = {}
    for k, c in enumerate(CLASSES):
        yk = y == k
        if 0 < yk.sum() < len(y):
            auroc[c] = float(roc_auc_score(yk, probs[:, k]))
    ym, pm = np.isin(y, MALIGNANT), np.isin(pred, MALIGNANT)
    return dict(
        accuracy=float((pred == y).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(rep["macro avg"]["f1-score"]),
        weighted_f1=float(rep["weighted avg"]["f1-score"]),
        macro_auroc=float(np.mean(list(auroc.values()))),
        auroc_per_class=auroc,
        melanoma_recall=float(rep["mel"]["recall"]),
        malignant_sensitivity=float((pm & ym).sum() / max(ym.sum(), 1)),
        malignant_specificity=float((~pm & ~ym).sum() / max((~ym).sum(), 1)),
        malignant_auroc=float(roc_auc_score(ym, probs[:, MALIGNANT].sum(1))),
        ece=ece_score(probs, y),
        per_class=rep,
        confusion_matrix=confusion_matrix(y, pred, labels=range(NUM)).tolist(),
    )
