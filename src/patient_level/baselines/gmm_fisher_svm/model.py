
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    auc as sk_auc,
    roc_auc_score,
)

from common.patient_paper import patient_table, patient_labels, write_json
from common.patient_baselines import stable_seed

EXCLUDE_COLS = {
    "label", "labels", "target", "class", "diagnosis", "y", "disease",
    "patient_id", "patient", "case_id", "case", "sample_id",
    "xuhao", "event_id", "eventid", "event", "index", "time", "nix",
}
NORMAL_LABELS = {"normal", "negative", "neg", "control", "healthy", "benign", "non-neoplastic", "non_neoplastic", "nonneoplastic"}


def read_header(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        line = f.readline()
    return next(csv.reader([line]))


def marker_columns_from_paths(paths: list[str], header_scan_limit: int = 2000) -> list[str]:
    inter = None
    scanned = 0
    for p in paths[:header_scan_limit]:
        try:
            cols = [c for c in read_header(p) if str(c).strip().lower() not in EXCLUDE_COLS]
        except Exception:
            continue
        if not cols:
            continue
        s = set(cols)
        inter = s if inter is None else inter & s
        scanned += 1
    markers = sorted(inter or [])
    if len(markers) < 2:
        raise ValueError(f"too few aligned marker columns after scanning {scanned} files: {markers}")
    return markers


def choose_positive_label(classes: Iterable[str]) -> str:
    classes = [str(c) for c in classes]
    for c in classes:
        if c.lower() not in NORMAL_LABELS and "normal" not in c.lower():
            return c
    return classes[-1]


def load_cells_for_patient(rows: pd.DataFrame, patient_id: str, markers: list[str], max_cells_per_patient: int, seed: int) -> np.ndarray:
    hit = rows[rows["_patient_id"].astype(str) == str(patient_id)]
    chunks = []
    for p in hit["_csv_path"].astype(str):
        try:
            header = read_header(p)
            avail = [m for m in markers if m in header]
            if not avail:
                continue
            df = pd.read_csv(p, usecols=avail, low_memory=False)
            # Reindex in case a later file lacks a scanned marker; fill handled later.
            df = df.reindex(columns=markers)
            arr = df.to_numpy(dtype=np.float32, copy=False)
            arr = arr[np.isfinite(arr).all(axis=1)]
            if len(arr):
                chunks.append(arr)
        except Exception as exc:
            print(f"[WARN] failed reading {p}: {exc}", flush=True)
    if not chunks:
        raise ValueError(f"no cells loaded for patient {patient_id}")
    x = np.vstack(chunks).astype(np.float32, copy=False)
    if max_cells_per_patient > 0 and len(x) > max_cells_per_patient:
        rng = np.random.default_rng(stable_seed(seed, "gmm_patient", patient_id))
        x = x[rng.choice(len(x), max_cells_per_patient, replace=False)]
    return x


def make_patient_cells(rows: pd.DataFrame, markers: list[str], max_cells_per_patient: int, seed: int):
    labels = patient_labels(rows)
    ids = labels["patient_id"].astype(str).tolist()
    y = labels["label"].astype(str).to_numpy()
    cells = []
    kept_ids, kept_y = [], []
    for pid, lab in zip(ids, y):
        try:
            cells.append(load_cells_for_patient(rows, pid, markers, max_cells_per_patient, seed))
            kept_ids.append(pid); kept_y.append(lab)
        except Exception as exc:
            print(f"[WARN] skip patient {pid}: {exc}", flush=True)
    return np.asarray(kept_ids), np.asarray(kept_y), cells


def sample_train_cells(cells: list[np.ndarray], max_total: int, max_per_patient: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for x in cells:
        take = min(len(x), max_per_patient)
        if take < len(x):
            x = x[rng.choice(len(x), take, replace=False)]
        parts.append(x)
    allx = np.vstack(parts).astype(np.float32, copy=False)
    if max_total > 0 and len(allx) > max_total:
        allx = allx[rng.choice(len(allx), max_total, replace=False)]
    return allx


def fit_preprocess(train_cells: list[np.ndarray], clip_quantile: float, max_gmm_events: int, max_gmm_events_per_patient: int, seed: int):
    train_sample = sample_train_cells(train_cells, max_gmm_events, max_gmm_events_per_patient, seed)
    q = float(clip_quantile)
    lo = np.nanquantile(train_sample, q, axis=0).astype(np.float32)
    hi = np.nanquantile(train_sample, 1.0 - q, axis=0).astype(np.float32)
    hi = np.where(hi <= lo, lo + 1e-6, hi).astype(np.float32)
    clipped = np.clip(train_sample, lo, hi)
    scaler = StandardScaler().fit(clipped)
    return lo, hi, scaler


def transform_cells(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    x = np.clip(x, lo, hi)
    return scaler.transform(x).astype(np.float32, copy=False)


def fit_gmm(train_cells, lo, hi, scaler, k: int, max_gmm_events: int, max_gmm_events_per_patient: int, seed: int):
    sampled = sample_train_cells(train_cells, max_gmm_events, max_gmm_events_per_patient, seed)
    sampled = transform_cells(sampled, lo, hi, scaler)
    gmm = GaussianMixture(
        n_components=int(k),
        covariance_type="diag",
        max_iter=150,
        n_init=1,
        reg_covar=1e-6,
        random_state=seed,
        verbose=0,
    )
    gmm.fit(sampled)
    return gmm


def fisher_vector_one(x: np.ndarray, gmm: GaussianMixture) -> np.ndarray:
    # Diagonal-covariance Fisher Vector: first and second-order deviations.
    if len(x) == 0:
        return np.zeros(2 * gmm.n_components * gmm.means_.shape[1], dtype=np.float32)
    gamma = gmm.predict_proba(x).astype(np.float64)  # T x K
    w = np.maximum(gmm.weights_.astype(np.float64), 1e-12)
    mu = gmm.means_.astype(np.float64)
    sigma = np.sqrt(np.maximum(gmm.covariances_.astype(np.float64), 1e-12))
    T = float(len(x))
    diff = (x[:, None, :] - mu[None, :, :]) / sigma[None, :, :]
    g_mu = (gamma[:, :, None] * diff).sum(axis=0) / (T * np.sqrt(w)[:, None])
    g_sig = (gamma[:, :, None] * (diff ** 2 - 1.0)).sum(axis=0) / (T * np.sqrt(2.0 * w)[:, None])
    fv = np.concatenate([g_mu.ravel(), g_sig.ravel()]).astype(np.float32)
    # Standard Fisher Vector post-processing: signed sqrt + L2 norm.
    fv = np.sign(fv) * np.sqrt(np.abs(fv) + 1e-12)
    norm = np.linalg.norm(fv)
    if norm > 0:
        fv = fv / norm
    return np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fisher_matrix(cells: list[np.ndarray], lo, hi, scaler, gmm) -> np.ndarray:
    feats = []
    for x in cells:
        z = transform_cells(x, lo, hi, scaler)
        feats.append(fisher_vector_one(z, gmm))
    return np.vstack(feats).astype(np.float32)


def binary_metrics(y_true, y_pred, proba, classes, n_bootstrap: int, seed: int):
    pos = choose_positive_label(classes)
    labels = [pos] + [c for c in classes if c != pos]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if len(labels) == 2:
        tp, fn = int(cm[0,0]), int(cm[0,1])
        fp, tn = int(cm[1,0]), int(cm[1,1])
    else:
        tp=fn=fp=tn=0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = f1_score(y_true, y_pred, labels=[pos], average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    bacc = (sens + spec) / 2.0
    pos_idx = list(classes).index(pos)
    y_bin = (np.asarray(y_true) == pos).astype(int)
    score = proba[:, pos_idx]
    roc = roc_auc_score(y_bin, score) if len(np.unique(y_bin)) == 2 else np.nan
    pr, rc, _ = precision_recall_curve(y_bin, score)
    prauc = sk_auc(rc, pr) if len(np.unique(y_bin)) == 2 else np.nan
    auc_ci = bootstrap_ci(y_bin, score, metric="auc", n=n_bootstrap, seed=seed)
    pr_ci = bootstrap_ci(y_bin, score, metric="prauc", n=n_bootstrap, seed=seed+1)
    return {
        "accuracy": acc, "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
        "f1": float(f1), "bacc": bacc, "auc": float(roc), "auc_ci_low": auc_ci[0], "auc_ci_high": auc_ci[1],
        "prauc": float(prauc), "prauc_ci_low": pr_ci[0], "prauc_ci_high": pr_ci[1],
        "support": int(len(y_true)), "positive_label": pos, "confusion_matrix": cm.tolist(),
    }


def bootstrap_ci(y_bin, score, metric: str, n: int, seed: int):
    if n <= 0 or len(np.unique(y_bin)) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals=[]; y_bin=np.asarray(y_bin); score=np.asarray(score)
    for _ in range(n):
        idx = rng.integers(0, len(y_bin), len(y_bin))
        yy = y_bin[idx]; ss = score[idx]
        if len(np.unique(yy)) < 2:
            continue
        try:
            if metric == "auc":
                vals.append(roc_auc_score(yy, ss))
            else:
                pr, rc, _ = precision_recall_curve(yy, ss)
                vals.append(sk_auc(rc, pr))
        except Exception:
            pass
    if not vals:
        return (np.nan, np.nan)
    return tuple(float(x) for x in np.percentile(vals, [2.5, 97.5]))


def multiclass_tables(y_true, y_pred, proba, classes, n_bootstrap: int, seed: int):
    classes=list(classes)
    cm_all = confusion_matrix(y_true, y_pred, labels=classes)
    rows=[]
    for i,c in enumerate(classes):
        yb=(np.asarray(y_true)==c).astype(int)
        pb=(np.asarray(y_pred)==c).astype(int)
        tp=int(((yb==1)&(pb==1)).sum()); fn=int(((yb==1)&(pb==0)).sum())
        fp=int(((yb==0)&(pb==1)).sum()); tn=int(((yb==0)&(pb==0)).sum())
        sens=tp/(tp+fn) if tp+fn else 0.0; spec=tn/(tn+fp) if tn+fp else 0.0
        ppv=tp/(tp+fp) if tp+fp else 0.0; npv=tn/(tn+fn) if tn+fn else 0.0
        f1=f1_score(yb,pb,zero_division=0)
        bacc=(sens+spec)/2
        score=proba[:,i]
        roc=roc_auc_score(yb,score) if len(np.unique(yb))==2 else np.nan
        pr,rc,_=precision_recall_curve(yb,score)
        prauc=sk_auc(rc,pr) if len(np.unique(yb))==2 else np.nan
        auc_ci=bootstrap_ci(yb,score,"auc",n_bootstrap,seed+i*10)
        pr_ci=bootstrap_ci(yb,score,"prauc",n_bootstrap,seed+i*10+1)
        rows.append({"label":c,"accuracy":accuracy_score(y_true,y_pred),"sensitivity":sens,"specificity":spec,
                     "ppv":ppv,"npv":npv,"precision":ppv,"recall":sens,"f1":f1,"bacc":bacc,
                     "auc":roc,"auc_ci_low":auc_ci[0],"auc_ci_high":auc_ci[1],"prauc":prauc,
                     "prauc_ci_low":pr_ci[0],"prauc_ci_high":pr_ci[1],"support":int(yb.sum()),
                     "tp":tp,"fp":fp,"tn":tn,"fn":fn})
    per=pd.DataFrame(rows)
    overall={
        "accuracy": float(accuracy_score(y_true,y_pred)),
        "sensitivity": float(per["sensitivity"].mean()),
        "specificity": float(per["specificity"].mean()),
        "ppv": float(per["ppv"].mean()),
        "npv": float(per["npv"].mean()),
        "macro_f1": float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
        "weighted_f1": float(f1_score(y_true,y_pred,average="weighted",zero_division=0)),
        "bacc": float(per["bacc"].mean()),
        "auc": float(per["auc"].mean()),
        "auc_ci_low": float(per["auc_ci_low"].mean()),
        "auc_ci_high": float(per["auc_ci_high"].mean()),
        "prauc": float(per["prauc"].mean()),
        "prauc_ci_low": float(per["prauc_ci_low"].mean()),
        "prauc_ci_high": float(per["prauc_ci_high"].mean()),
        "support": int(len(y_true)),
        "confusion_matrix": cm_all.tolist(),
    }
    return overall, per


def save_outputs(out: Path, ids, y_true, pred, proba, classes, n_bootstrap: int, seed: int):
    out.mkdir(parents=True, exist_ok=True)
    pred_df=pd.DataFrame({"patient_id":ids,"true_label":y_true,"pred_label":pred,"correct":np.asarray(y_true)==np.asarray(pred)})
    pred_df.to_csv(out/"predictions.csv", index=False)
    prob_df=pd.DataFrame({"patient_id":ids,"true_label":y_true})
    for i,c in enumerate(classes):
        prob_df[f"prob_{c}"]=proba[:,i]
    prob_df.to_csv(out/"probabilities.csv", index=False)
    # compatibility with existing paper baselines
    legacy=pd.DataFrame({"patient_id":ids,"label":y_true,"prediction":pred})
    for i,c in enumerate(classes): legacy[f"prob_{c}"]=proba[:,i]
    legacy.to_csv(out/"test_predictions.csv", index=False)
    if len(classes)==2:
        m=binary_metrics(y_true,pred,proba,classes,n_bootstrap,seed)
        overall={k:v for k,v in m.items() if k not in {"positive_label","confusion_matrix"}}
        pd.DataFrame([{**overall, "positive_label":m["positive_label"], "confusion_matrix":json.dumps(m["confusion_matrix"],ensure_ascii=False)}]).to_csv(out/"overall_metrics.csv",index=False)
        # Per-class rows via multiclass helper for compatibility
        _, per=multiclass_tables(y_true,pred,proba,classes,n_bootstrap,seed)
        per.to_csv(out/"metrics_per_class.csv", index=False)
        return m
    overall, per = multiclass_tables(y_true,pred,proba,classes,n_bootstrap,seed)
    pd.DataFrame([{**{k:v for k,v in overall.items() if k!="confusion_matrix"}, "confusion_matrix":json.dumps(overall["confusion_matrix"],ensure_ascii=False)}]).to_csv(out/"overall_metrics.csv",index=False)
    per.to_csv(out/"metrics_per_class.csv", index=False)
    return overall
