"""Per-tube UMAP reducers and patient histogram features."""
from __future__ import annotations

import numpy as np

from common.patient_baselines import numeric_events
from common.patient_paper import marker_frame, scaled_values


def import_umap():
    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP-RF requires umap-learn") from exc
    return umap


def tube_order(rows):
    return sorted(rows["_tube_id"].astype(str).unique(), key=lambda x: (0, int(x), x) if x.isdigit() else (1, x))


def pad_width(x: np.ndarray, width: int) -> np.ndarray:
    if x.shape[1] == width:
        return x
    if x.shape[1] > width:
        return x[:, :width]
    out = np.zeros((x.shape[0], width), dtype=x.dtype)
    out[:, : x.shape[1]] = x
    return out


def balanced_tube_events(rows, tube: str, max_total_events: int, max_events_per_file: int, seed: int):
    rng = np.random.default_rng(seed)
    tube_rows = rows[rows["_tube_id"].astype(str) == tube].copy()
    labels = sorted(tube_rows["_label"].astype(str).unique())
    per_label = max(1, max_total_events // max(len(labels), 1))
    chunks = []
    for label in labels:
        subset = tube_rows[tube_rows["_label"].astype(str) == label]
        per_file = max(1, per_label // max(len(subset), 1))
        for path in subset["_csv_path"]:
            frame = marker_frame(path, max_events_per_file)
            values = scaled_values(frame, list(frame.columns)).astype(np.float32)
            take = min(len(values), per_file)
            if take < len(values):
                values = values[rng.choice(len(values), take, replace=False)]
            chunks.append(values)
    width = max(x.shape[1] for x in chunks)
    values = np.vstack([pad_width(x, width) for x in chunks])
    if len(values) > max_total_events:
        values = values[rng.choice(len(values), max_total_events, replace=False)]
    return values, width


def patient_hist(rows, patient_id: str, tube: str, reducer, width: int, bins: int, max_events: int) -> np.ndarray:
    hit = rows[(rows["_patient_id"].astype(str) == str(patient_id)) & (rows["_tube_id"].astype(str) == tube)]
    if hit.empty:
        return np.zeros(bins * bins, dtype=np.float32)
    frame = marker_frame(hit["_csv_path"].iloc[0], max_events)
    values = scaled_values(frame, list(frame.columns)).astype(np.float32)
    z = reducer.transform(pad_width(values, width))
    hist, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=bins)
    hist = hist.astype(np.float32)
    return (hist / (hist.sum() + 1e-8)).ravel()


def feature_matrix(rows, patient_ids, tubes, reducers, widths, bins, max_events):
    features = []
    for pid in patient_ids:
        parts = [patient_hist(rows, pid, tube, reducers[tube], widths[tube], bins, max_events) for tube in tubes]
        features.append(np.concatenate(parts))
    return np.vstack(features).astype(np.float32)


def _pad_events(events, width):
    if events.shape[1] == width:
        return events
    if events.shape[1] > width:
        return events[:, :width]
    out = np.zeros((events.shape[0], width), dtype=events.dtype)
    out[:, :events.shape[1]] = events
    return out

def fit_umap(paths, max_events_per_file=3000, max_total_events=200000, random_state=42):
    try: import umap
    except ImportError as e: raise ImportError('UMAP-RF requires umap-learn: pip install umap-learn') from e
    chunks=[numeric_events(p,max_events_per_file) for p in paths]
    width=max(x.shape[1] for x in chunks)
    chunks=[_pad_events(x,width) for x in chunks]
    x=np.vstack(chunks); x=x[np.random.default_rng(random_state).choice(len(x),min(len(x),max_total_events),replace=False)]
    reducer=umap.UMAP(n_components=2,n_neighbors=15,min_dist=.1,random_state=random_state).fit(x)
    reducer.fcm_ml_input_width_=width
    return reducer
def umap_features(path, reducer, max_events=12000, bins=32):
    x=numeric_events(path,max_events)
    x=_pad_events(x,getattr(reducer,'fcm_ml_input_width_',x.shape[1]))
    z=reducer.transform(x); h,_,_=np.histogram2d(z[:,0],z[:,1],bins=bins,range=((-15,15),(-15,15)))
    return (h/(h.sum()+1e-8)).ravel().astype('float32')
