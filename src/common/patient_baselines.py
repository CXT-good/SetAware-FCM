"""Shared, reproducible I/O, features, models, and evaluation for patient tasks."""
from __future__ import annotations
import hashlib
import json, random
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from dataio.paths import remap_manifest_path

PATH_COLS = ('csv_path', 'path', 'file_path', 'filepath')
LABEL_COLS = ('label', 'labels', 'diagnosis', 'class', 'target', 'y', 'disease')
PATIENT_COLS = ('patient_id', 'patient', 'individual', 'case_id', 'case', 'sample_id')

def pick_column(columns: Iterable[str], candidates: Iterable[str], required=True):
    lower = {str(c).lower().strip(): c for c in columns}
    for name in candidates:
        if name in lower: return lower[name]
    for key, original in lower.items():
        if any(name in key for name in candidates): return original
    if required: raise ValueError(f'Cannot infer required column from {list(columns)}; expected one of {tuple(candidates)}')
    return None

def read_split_book(split_xlsx: str, split: str) -> pd.DataFrame:
    requested = split.strip().casefold()
    with pd.ExcelFile(split_xlsx) as workbook:
        sheets = workbook.sheet_names
        wanted = next((s for s in sheets if s.strip().casefold() == requested), None)
        if wanted is None and len(sheets) != 1:
            raise ValueError(f'Split {split!r} is missing from workbook sheets {sheets}')
        sheet = wanted if wanted is not None else sheets[0]
        columns = pd.read_excel(workbook, sheet_name=sheet, nrows=0).columns
        # Preserve Excel string identities such as "01" alongside numeric tube IDs.
        tube_dtypes = {c: object for c in columns if str(c).strip().casefold() == 'tube_type'}
        df = pd.read_excel(workbook, sheet_name=sheet, dtype=tube_dtypes)
    split_col = None
    if len(sheets) == 1:
        normalized_columns = {str(c).strip().casefold(): c for c in df.columns}
        split_col = next((normalized_columns[name] for name in ('split', 'set', 'fold') if name in normalized_columns), None)
    partitioned = split_col is not None
    if wanted is None and not partitioned:
        raise ValueError(f'Split {split!r} is missing and the single sheet has no split column')
    if partitioned:
        # A single sheet's explicit partitions take precedence over its sheet name.
        df = df[df[split_col].astype(str).str.strip().str.casefold() == requested].copy()
        if df.empty:
            raise ValueError(f'Split {split!r} has no rows in the single-sheet partition')
    path_col = pick_column(df.columns, PATH_COLS)
    label_col = pick_column(df.columns, LABEL_COLS)
    patient_col = pick_column(df.columns, PATIENT_COLS, required=False)
    df = df.dropna(subset=[path_col, label_col]).copy()
    if partitioned and df.empty:
        raise ValueError(f'Split {split!r} has no rows with a CSV path and label')
    df['_csv_path'] = df[path_col].astype(str).map(remap_manifest_path)
    df['_label'] = df[label_col].astype(str)
    df['_patient_id'] = (df[patient_col].astype(str) if patient_col else df['_csv_path'].map(lambda x: Path(x).stem.split('_')[0]))
    missing = [p for p in df['_csv_path'] if not Path(p).is_file()]
    if missing: raise FileNotFoundError(f'{len(missing)} CSV paths do not exist; first: {missing[0]}')
    return df

def stable_seed(seed: int, *identities: str) -> int:
    """Derive a reproducible RNG seed from logical identities, never file paths."""
    payload = json.dumps([int(seed), *map(str, identities)], ensure_ascii=False, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest()[:8], 'little')

def choose_split(split_xlsx: str, preferred: str, fallback: str = 'test') -> str:
    sheets = pd.ExcelFile(split_xlsx).sheet_names
    lower = {s.lower(): s for s in sheets}
    if preferred.lower() in lower:
        return lower[preferred.lower()]
    if fallback.lower() in lower:
        return lower[fallback.lower()]
    return preferred

def numeric_events(path: str, max_events=12000) -> np.ndarray:
    df = pd.read_csv(path, nrows=max_events, low_memory=False)
    arr = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32, copy=False)
    if arr.ndim != 2 or arr.shape[1] == 0: raise ValueError(f'No numeric marker columns in {path}')
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) == 0: raise ValueError(f'No finite events in {path}')
    if len(arr) > max_events:
        rng = np.random.default_rng(abs(hash(path)) % (2**32)); arr = arr[rng.choice(len(arr), max_events, replace=False)]
    return arr

def summary_features(path: str, max_events=12000) -> np.ndarray:
    x = numeric_events(path, max_events)
    qs = np.quantile(x, [0.01, .05, .25, .5, .75, .95, .99], axis=0)
    return np.concatenate([x.mean(0), x.std(0), qs.reshape(-1), [np.log1p(len(x))]]).astype('float32')

def group_patient_features(rows: pd.DataFrame, feature_fn, **kwargs):
    sample = [feature_fn(p, **kwargs) for p in rows['_csv_path']]
    # Different panels have different marker counts: pad then mean/max aggregate.
    width = max(map(len, sample)); mat = np.full((len(sample), width), np.nan, np.float32)
    for i, f in enumerate(sample): mat[i, :len(f)] = f
    rows = rows.copy(); rows['_feature'] = list(mat)
    records=[]
    for pid, g in rows.groupby('_patient_id', sort=False):
        a=np.vstack(g['_feature'].to_numpy()); records.append((pid, g['_label'].mode().iat[0], np.nanmean(a,0), np.nanmax(a,0)))
    ids, y, mean, mx = zip(*records)
    X=np.nan_to_num(np.hstack([np.vstack(mean),np.vstack(mx)]), nan=0., posinf=0., neginf=0.)
    return np.asarray(ids), np.asarray(y), X

def save_model(out_dir, model, encoder, metadata):
    import joblib
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model':model,'encoder':encoder,'metadata':metadata}, out/'model.joblib')
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')

def train_rf(train_rows, val_rows, out_dir, feature_fn=summary_features, method='baseline', **kwargs):
    _, ytr, Xtr=group_patient_features(train_rows,feature_fn,**kwargs); _, yv, Xv=group_patient_features(val_rows,feature_fn,**kwargs)
    enc=LabelEncoder().fit(ytr); model=RandomForestClassifier(n_estimators=500,class_weight='balanced',n_jobs=-1,random_state=42)
    model.fit(Xtr,enc.transform(ytr)); pred=enc.inverse_transform(model.predict(Xv))
    metadata={'method':method,'classes':enc.classes_.tolist(),'validation_accuracy':float(accuracy_score(yv,pred)),'validation_macro_f1':float(f1_score(yv,pred,average='macro')),'feature_kwargs':kwargs}
    save_model(out_dir,model,enc,metadata); return metadata

def evaluate_rf(rows, model_dir, output_csv, feature_fn=summary_features, **kwargs):
    import joblib
    saved=joblib.load(Path(model_dir)/'model.joblib'); ids,y,X=group_patient_features(rows,feature_fn,**kwargs)
    pred=saved['encoder'].inverse_transform(saved['model'].predict(X)); proba=saved['model'].predict_proba(X)
    result=pd.DataFrame({'patient_id':ids,'label':y,'prediction':pred})
    for i,c in enumerate(saved['encoder'].classes_): result[f'prob_{c}']=proba[:,i]
    output=Path(output_csv); output.parent.mkdir(parents=True,exist_ok=True); result.to_csv(output,index=False)
    metrics={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'report':classification_report(y,pred,output_dict=True,zero_division=0)}
    output.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8'); return metrics
