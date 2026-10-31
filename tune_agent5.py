"""
Tune Agent 5 thresholds on the 2,000 validation images.

    python tune_agent5.py --preds outputs/val_predictions.csv --meta HAM10000_metadata.csv

Inputs
  --preds  A2's val_predictions.csv (true label, 7 probabilities; optional
           tta_agreement, quality_score, suspicion, quality_status columns).
  --meta   HAM10000_metadata.csv (image_id, dx, age, sex, localization, lesion_id).
           Used for A4-style context and for lesion_id-grouped splitting.

Method
  * Split validation 50/50 (grouped by lesion_id): tune on half, CONFIRM on the other.
  * Simulated A6 policy: A1 fail -> RE_IMAGE, A5 veto -> REFER, else AUTO_TRIAGE.
  * Pick thresholds that maximise auto-correct subject to
      - auto-incorrect rate <= --max_err   (of auto-triaged cases)
      - missed-malignant rate <= --max_mal_miss (true mel/bcc/akiec auto-triaged wrongly)
    falling back to a weighted cost if nothing is feasible.
  * Writes agent5_config.json (loaded by agent5.load_config) and
    agent5_val_audit.csv (per-image decision, for demo evidence).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np
import pandas as pd

from agent5 import CLASSES, DEFAULT_CONFIG, HIGH_RISK_SITES, MALIGNANT, MISSING_TOKENS, safety_audit

MAL_IDX = np.array([c in MALIGNANT for c in CLASSES])


# ------------------------------------------------------------------ loading
def find_col(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def load(preds_path, meta_path):
    df = pd.read_csv(preds_path)
    idc = find_col(df, "image_id", "image", "id")
    if idc is None:
        raise SystemExit(f"No image_id column in {preds_path}. Columns: {list(df.columns)}")
    df = df.rename(columns={idc: "image_id"})

    pcols = {}
    for c in CLASSES:
        col = find_col(df, c, f"prob_{c}", f"p_{c}", f"{c}_prob", f"proba_{c}", f"prob_{c}_cal")
        if col is None:
            raise SystemExit(f"Cannot find probability column for class '{c}'. Columns: {list(df.columns)}")
        pcols[c] = col
    print("Probability columns detected:", pcols)

    meta = None
    if meta_path:
        meta = pd.read_csv(meta_path)
        keep = [c for c in ["image_id", "dx", "age", "sex", "localization", "lesion_id"] if c in meta.columns]
        meta = meta[keep].drop_duplicates("image_id")
        df = df.merge(meta, on="image_id", how="left", suffixes=("", "_meta"))

    tcol = find_col(df, "true_label", "true", "y_true", "gt", "dx", "label_true", "label")
    if tcol is None:
        raise SystemExit("No true-label column found (need true_label or metadata dx).")

    P = df[[pcols[c] for c in CLASSES]].to_numpy(float)
    P = P / P.sum(1, keepdims=True)
    srt = np.sort(P, axis=1)[:, ::-1]
    pred_idx = P.argmax(1)

    age_col = find_col(df, "age")
    site_col = find_col(df, "localization", "site")
    sex_col = find_col(df, "sex")
    tta_col = find_col(df, "tta_agreement")
    q_col = find_col(df, "quality_score", "a1_score")
    qs_col = find_col(df, "quality_status", "a1_status")
    s_col = find_col(df, "suspicion", "a3_suspicion")

    age = pd.to_numeric(df[age_col], errors="coerce").to_numpy(float) if age_col else np.full(len(df), np.nan)
    site = df[site_col].astype(str).str.lower().to_numpy() if site_col else np.array(["unknown"] * len(df))
    sex = df[sex_col].astype(str).str.lower().to_numpy() if sex_col else np.array(["unknown"] * len(df))
    nmiss = (np.isnan(age).astype(int)
             + np.isin(site, list(MISSING_TOKENS)).astype(int)
             + np.isin(sex, list(MISSING_TOKENS)).astype(int))

    F = dict(
        image_id=df["image_id"].to_numpy(),
        true=df[tcol].astype(str).str.lower().to_numpy(),
        pred=np.array(CLASSES)[pred_idx],
        is_mal=MAL_IDX[pred_idx],
        conf=srt[:, 0],
        margin=srt[:, 0] - srt[:, 1],
        entropy=-(np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0)).sum(1),
        pmal=P[:, MAL_IDX].sum(1),
        tta=pd.to_numeric(df[tta_col], errors="coerce").to_numpy(float) if tta_col else np.full(len(df), np.nan),
        q=pd.to_numeric(df[q_col], errors="coerce").to_numpy(float) if q_col else np.full(len(df), np.nan),
        a1fail=(df[qs_col].astype(str).str.upper().isin(["FAIL", "FAILED", "POOR"]).to_numpy()
                if qs_col else np.zeros(len(df), bool)),
        susp=pd.to_numeric(df[s_col], errors="coerce").to_numpy(float) if s_col else np.full(len(df), np.nan),
        age=age, site=site, sex=sex, nmiss=nmiss,
        group=(df["lesion_id"].astype(str).to_numpy() if "lesion_id" in df.columns else df["image_id"].to_numpy()),
        probs=P,
    )
    return F


def subset(F, idx):
    return {k: v[idx] for k, v in F.items()}


# --------------------------------------------------------- vectorised rules
def veto_mask(F, cfg):
    """Vectorised twin of agent5.safety_audit (checked by check_consistency)."""
    benign = ~F["is_mal"]
    risk_high = (np.nan_to_num(F["age"], nan=-1) >= cfg["high_age"]) | np.isin(F["site"], list(HIGH_RISK_SITES))
    with np.errstate(invalid="ignore"):
        m = F["is_mal"].copy()
        m |= F["pmal"] > cfg["max_p_malignant"]
        m |= benign & (F["susp"] >= cfg["susp_conflict"])
        m |= F["conf"] < cfg["min_confidence"]
        m |= F["margin"] < cfg["min_margin"]
        m |= F["entropy"] > cfg["max_entropy"]
        m |= F["tta"] < cfg["min_tta_agreement"]
        m |= (F["q"] < cfg["min_quality_score"]) & (F["conf"] < cfg["borderline_conf"])
        m |= benign & risk_high & (F["conf"] < cfg["context_conf"])
        m |= (F["nmiss"] >= cfg["missing_min"]) & (F["conf"] < cfg["missing_conf"])
    return m | F["a1fail"]


def metrics(F, veto):
    n = len(F["conf"])
    reimage = F["a1fail"]
    refer = veto & ~reimage
    auto = ~veto & ~reimage
    correct = F["pred"] == F["true"]
    true_mal = np.isin(F["true"], list(MALIGNANT))
    ac, ai = int((auto & correct).sum()), int((auto & ~correct).sum())
    mal_miss = int((auto & ~correct & true_mal).sum())
    return dict(
        n=n, auto=int(auto.sum()), auto_correct=ac, auto_incorrect=ai,
        mal_auto_incorrect=mal_miss, n_true_mal=int(true_mal.sum()),
        refer=int(refer.sum()), reimage=int(reimage.sum()),
        auto_err_rate=ai / max(1, ac + ai),
        mal_miss_rate=mal_miss / max(1, int(true_mal.sum())),
        accuracy=float(correct.mean()),
    )


def grid(F):
    has_tta = not np.isnan(F["tta"]).all()
    has_susp = not np.isnan(F["susp"]).all()
    axes = {
        "min_confidence": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
        "min_margin": [0.1, 0.3, 0.5],
        "max_entropy": [0.5, 0.7, 0.9, 1.1],
        "max_p_malignant": [0.05, 0.10, 0.20, 0.30],
        "min_tta_agreement": [0.5, 0.75, 1.0] if has_tta else [DEFAULT_CONFIG["min_tta_agreement"]],
        "susp_conflict": [0.5, 0.6, 0.7, 0.8] if has_susp else [DEFAULT_CONFIG["susp_conflict"]],
    }
    keys = list(axes)
    for combo in itertools.product(*axes.values()):
        yield dict(zip(keys, combo))


def tune(F, max_err, max_mal_miss):
    best, best_key = None, None
    for g in grid(F):
        cfg = {**DEFAULT_CONFIG, **g}
        m = metrics(F, veto_mask(F, cfg))
        feasible = m["auto_err_rate"] <= max_err and m["mal_miss_rate"] <= max_mal_miss
        cost = 10 * m["mal_auto_incorrect"] + m["auto_incorrect"] - 0.1 * m["auto_correct"]
        key = (0, -m["auto_correct"], m["auto_incorrect"]) if feasible else (1, cost, 0)
        if best_key is None or key < best_key:
            best, best_key = cfg, key
    return best, best_key[0] == 0


def check_consistency(F, cfg, k=300):
    """Vectorised rules must agree with the real safety_audit()."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(F["conf"]), size=min(k, len(F["conf"])), replace=False)
    mask = veto_mask(F, cfg)
    bad = 0
    for i in idx:
        a1 = {"status": "FAIL" if F["a1fail"][i] else "PASS",
              "score": None if np.isnan(F["q"][i]) else float(F["q"][i])}
        a2 = {"probabilities": dict(zip(CLASSES, map(float, F["probs"][i]))),
              "tta_agreement": None if np.isnan(F["tta"][i]) else float(F["tta"][i])}
        a3 = None if np.isnan(F["susp"][i]) else {"suspicion": float(F["susp"][i])}
        a4 = {"age": None if np.isnan(F["age"][i]) else float(F["age"][i]),
              "sex": F["sex"][i], "site": F["site"][i]}
        if safety_audit(a1, a2, a3, a4, cfg).veto != bool(mask[i]):
            bad += 1
    print(f"Consistency check vs safety_audit(): {bad} mismatches in {len(idx)} rows")
    return bad


def show(title, m):
    print(f"\n{title}")
    print(f"  cases={m['n']}  accuracy={m['accuracy']:.3f}")
    print(f"  AUTO_TRIAGE={m['auto']} (correct {m['auto_correct']}, incorrect {m['auto_incorrect']}, "
          f"err rate {m['auto_err_rate']:.3f})  REFER={m['refer']}  RE_IMAGE={m['reimage']}")
    print(f"  true malignant cases wrongly auto-triaged: {m['mal_auto_incorrect']} / {m['n_true_mal']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--max_err", type=float, default=0.05)
    ap.add_argument("--max_mal_miss", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_config", default="agent5_config.json")
    ap.add_argument("--out_audit", default="agent5_val_audit.csv")
    a = ap.parse_args()

    F = load(a.preds, a.meta)
    if np.isnan(F["susp"]).all():
        print("NOTE: no A3 suspicion column; conflict rule untuned. Add 'suspicion' to the CSV to tune it.")
    if np.isnan(F["tta"]).all():
        print("NOTE: no tta_agreement column; that rule untuned.")

    # grouped 50/50 split
    groups = np.unique(F["group"])
    rng = np.random.default_rng(a.seed)
    rng.shuffle(groups)
    tune_groups = set(groups[: len(groups) // 2])
    is_tune = np.array([g in tune_groups for g in F["group"]])
    Ft, Fc = subset(F, np.where(is_tune)[0]), subset(F, np.where(~is_tune)[0])
    print(f"\nTune half: {len(Ft['conf'])}   Confirm half: {len(Fc['conf'])}")

    cfg, feasible = tune(Ft, a.max_err, a.max_mal_miss)
    if not feasible:
        print("WARNING: no config met the constraints on the tune half; used weighted-cost fallback.")
    tuned = {k: cfg[k] for k in cfg}

    no_a5 = np.zeros(len(F["conf"]), bool) | F["a1fail"]
    for name, S in (("TUNE half", Ft), ("CONFIRM half (honest numbers)", Fc)):
        base = metrics(S, np.zeros(len(S["conf"]), bool) | S["a1fail"])
        res = metrics(S, veto_mask(S, cfg))
        show(f"{name}: WITHOUT A5 (auto-triage everything usable)", base)
        show(f"{name}: WITH A5", res)
        print(f"  >> A5 prevented {base['auto_incorrect'] - res['auto_incorrect']} incorrect auto-triages "
              f"({base['mal_auto_incorrect'] - res['mal_auto_incorrect']} of them true malignant), "
              f"costing {base['auto_correct'] - res['auto_correct']} auto-correct cases sent to REFER")

    check_consistency(F, cfg)

    with open(a.out_config, "w") as f:
        json.dump(tuned, f, indent=2)
    full = veto_mask(F, cfg)
    pd.DataFrame({
        "image_id": F["image_id"], "true": F["true"], "pred": F["pred"],
        "confidence": F["conf"].round(4), "a5_veto": full,
        "sim_decision": np.where(F["a1fail"], "RE_IMAGE", np.where(full, "REFER", "AUTO_TRIAGE")),
        "tune_half": is_tune,
    }).to_csv(a.out_audit, index=False)
    print(f"\nSaved {a.out_config} and {a.out_audit}")
    print("Tuned thresholds:", json.dumps(tuned, indent=2))


if __name__ == "__main__":
    main()
