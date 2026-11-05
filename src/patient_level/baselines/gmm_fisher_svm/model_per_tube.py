
from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from common.patient_paper import patient_labels
from common.patient_baselines import stable_seed
from patient_level.baselines.gmm_fisher_svm.model import (
    EXCLUDE_COLS,
    read_header,
    fisher_vector_one,
)


def marker_intersection(paths, header_scan_limit=2000):
    # Keep only real numeric marker channels. 09_Z2JW_LMD CSV files contain
    # a string metadata column named case_group (AML/normal/etc.); if it is
    # treated as a marker, every file fails when converted to float.
    extra_exclude = {
        "case_group", "casegroup", "group", "subtype", "type", "filename",
        "file", "csv_path", "path", "split", "tube", "tube_id",
    }
    exclude = set(EXCLUDE_COLS) | extra_exclude
    inter=None; scanned=0
    for p in list(paths)[:header_scan_limit]:
        try:
            header=[c for c in read_header(str(p)) if str(c).strip().lower() not in exclude]
            if not header:
                continue
            sample=pd.read_csv(p, usecols=header, nrows=64, low_memory=False)
            numeric=[]
            for c in header:
                s=pd.to_numeric(sample[c], errors='coerce')
                if s.notna().any():
                    numeric.append(c)
            cols=numeric
        except Exception:
            continue
        if not cols:
            continue
        inter=set(cols) if inter is None else inter & set(cols)
        scanned += 1
    markers=sorted(inter or [])
    if len(markers) < 1:
        raise ValueError(f"no aligned markers for tube after scanning {scanned} files")
    return markers, scanned


def load_one_file(path, markers, max_events, seed):
    try:
        header=set(read_header(str(path)))
        avail=[m for m in markers if m in header]
        if not avail:
            return None
        df=pd.read_csv(path, usecols=avail, low_memory=False)
        df=df.reindex(columns=markers)
        arr=df.to_numpy(dtype=np.float32, copy=False)
        arr=arr[np.isfinite(arr).all(axis=1)]
        if len(arr)==0:
            return None
        if max_events > 0 and len(arr) > max_events:
            rng=np.random.default_rng(seed)
            arr=arr[rng.choice(len(arr), max_events, replace=False)]
        return arr.astype(np.float32, copy=False)
    except Exception as exc:
        print(f"[WARN] failed file {path}: {exc}", flush=True)
        return None


def patient_ids_labels(rows):
    labels=patient_labels(rows)
    return labels['patient_id'].astype(str).tolist(), labels['label'].astype(str).to_numpy()


def fit_preprocess_from_arrays(arrays, clip_quantile, max_gmm_events, seed):
    rng=np.random.default_rng(seed)
    parts=[a for a in arrays if a is not None and len(a)]
    if not parts:
        raise ValueError('no arrays for preprocessing')
    x=np.vstack(parts)
    if max_gmm_events>0 and len(x)>max_gmm_events:
        x=x[rng.choice(len(x), max_gmm_events, replace=False)]
    q=float(clip_quantile)
    lo=np.nanquantile(x,q,axis=0).astype(np.float32)
    hi=np.nanquantile(x,1-q,axis=0).astype(np.float32)
    hi=np.where(hi<=lo,lo+1e-6,hi).astype(np.float32)
    x=np.clip(x,lo,hi)
    scaler=StandardScaler().fit(x)
    return lo,hi,scaler


def transform(x, lo, hi, scaler):
    return scaler.transform(np.clip(x,lo,hi)).astype(np.float32, copy=False)


def fit_tube_pipeline(train_rows, tube, train_ids, k, args, seed_offset=0):
    tube_rows=train_rows[train_rows['_tube_id'].astype(str)==str(tube)]
    markers, scanned=marker_intersection(tube_rows['_csv_path'].astype(str).tolist(), args.header_scan_limit)
    print(f"[TUBE] {tube} markers({len(markers)})={markers}", flush=True)
    arrays=[]
    for pid in train_ids:
        hit=tube_rows[tube_rows['_patient_id'].astype(str)==str(pid)]
        chunks=[]
        for p in hit['_csv_path'].astype(str):
            a=load_one_file(p,markers,args.max_events_per_file,args.seed+seed_offset)
            if a is not None:
                chunks.append(a)
        if chunks:
            x=np.vstack(chunks)
            if args.max_events_per_patient_tube>0 and len(x)>args.max_events_per_patient_tube:
                rng=np.random.default_rng(stable_seed(args.seed+seed_offset, "gmm_patient_tube", pid, tube))
                x=x[rng.choice(len(x),args.max_events_per_patient_tube,replace=False)]
            arrays.append(x)
        else:
            arrays.append(None)
    fit_arrays=[a for a in arrays if a is not None]
    lo,hi,scaler=fit_preprocess_from_arrays(fit_arrays,args.clip_quantile,args.max_gmm_events,args.seed+seed_offset)
    sample=np.vstack([transform(a,lo,hi,scaler) for a in fit_arrays])
    if args.max_gmm_events>0 and len(sample)>args.max_gmm_events:
        rng=np.random.default_rng(args.seed+seed_offset)
        sample=sample[rng.choice(len(sample),args.max_gmm_events,replace=False)]
    gmm=GaussianMixture(n_components=int(k),covariance_type='diag',max_iter=150,n_init=1,reg_covar=1e-6,random_state=args.seed+seed_offset)
    gmm.fit(sample)
    dim=2*int(k)*len(markers)
    return {'tube':str(tube),'markers':markers,'lo':lo,'hi':hi,'scaler':scaler,'gmm':gmm,'dim':dim}


def transform_tube(rows, tube, ids, pipe, args, seed_offset=0):
    tube_rows=rows[rows['_tube_id'].astype(str)==str(tube)]
    feats=[]
    markers=pipe['markers']; lo=pipe['lo']; hi=pipe['hi']; scaler=pipe['scaler']; gmm=pipe['gmm']
    for pid in ids:
        hit=tube_rows[tube_rows['_patient_id'].astype(str)==str(pid)]
        chunks=[]
        for p in hit['_csv_path'].astype(str):
            a=load_one_file(p,markers,args.max_events_per_file,args.seed+seed_offset)
            if a is not None:
                chunks.append(a)
        if not chunks:
            feats.append(np.zeros(pipe['dim'], dtype=np.float32)); continue
        x=np.vstack(chunks)
        if args.max_events_per_patient_tube>0 and len(x)>args.max_events_per_patient_tube:
            rng=np.random.default_rng(stable_seed(args.seed+seed_offset, "gmm_patient_tube", pid, tube))
            x=x[rng.choice(len(x),args.max_events_per_patient_tube,replace=False)]
        z=transform(x,lo,hi,scaler)
        feats.append(fisher_vector_one(z,gmm))
    return np.vstack(feats).astype(np.float32)


def fit_feature_pipeline(train_rows, train_ids, eval_rows, eval_ids, k, args, seed_offset=0):
    tubes=sorted(train_rows['_tube_id'].astype(str).unique(), key=lambda x: (0, int(x), x) if x.isdigit() else (1, x))
    pipes=[]; Xtr_parts=[]; Xev_parts=[]
    for ti,tube in enumerate(tubes):
        pipe=fit_tube_pipeline(train_rows,tube,train_ids,k,args,seed_offset+ti*1000)
        pipes.append(pipe)
        Xtr_parts.append(transform_tube(train_rows,tube,train_ids,pipe,args,seed_offset+ti*1000+100))
        Xev_parts.append(transform_tube(eval_rows,tube,eval_ids,pipe,args,seed_offset+ti*1000+200))
    return np.hstack(Xtr_parts).astype(np.float32), np.hstack(Xev_parts).astype(np.float32), pipes
