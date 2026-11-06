"""Marker-pair histogram features and CNN architecture for EnsembleCNN."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from common.patient_baselines import numeric_events
from common.patient_paper import EXCLUDE_NUMERIC_COLS, SmallHistogramCNN, hist2d, marker_frame


def marker_columns(path: str, max_events: int) -> list[str]:
    frame = pd.read_csv(path, nrows=max_events, low_memory=False)
    numeric = frame.select_dtypes(include=[np.number])
    return [c for c in numeric.columns if str(c).lower() not in EXCLUDE_NUMERIC_COLS]


def pair_specs(rows: pd.DataFrame, max_events: int, max_pairs: int) -> list[tuple[str, str, str]]:
    specs = []
    for tube_id, group in rows.groupby("_tube_id", sort=True):
        cols = marker_columns(group["_csv_path"].iloc[0], max_events)
        for x_col, y_col in itertools.combinations(cols, 2):
            specs.append((str(tube_id), x_col, y_col))
    if max_pairs > 0:
        specs = specs[:max_pairs]
    return specs


def image_for_patient(rows: pd.DataFrame, patient_id: str, spec: tuple[str, str, str], bins: int, max_events: int) -> np.ndarray:
    tube_id, x_col, y_col = spec
    hit = rows[(rows["_patient_id"].astype(str) == str(patient_id)) & (rows["_tube_id"].astype(str) == tube_id)]
    if hit.empty:
        return np.zeros((bins, bins), dtype=np.float32)
    frame = marker_frame(hit["_csv_path"].iloc[0], max_events)
    if x_col not in frame.columns or y_col not in frame.columns:
        return np.zeros((bins, bins), dtype=np.float32)
    return hist2d(frame, x_col, y_col, bins)


def build_images(rows: pd.DataFrame, patient_ids: list[str], spec: tuple[str, str, str], bins: int, max_events: int) -> np.ndarray:
    return np.stack([image_for_patient(rows, pid, spec, bins, max_events) for pid in patient_ids]).astype(np.float32)


def ensemble_features(path,max_events=12000,bins=16,max_pairs=24):
    x=numeric_events(path,max_events); var=np.argsort(np.var(x,axis=0))[::-1][:8]; chunks=[]
    for i,a in enumerate(var):
      for b in var[i+1:]:
        lo1,hi1=np.quantile(x[:,a],[.01,.99]); lo2,hi2=np.quantile(x[:,b],[.01,.99])
        h,_,_=np.histogram2d(x[:,a],x[:,b],bins=bins,range=((lo1,hi1),(lo2,hi2))); chunks.append((h/(h.sum()+1e-8)).ravel())
        if len(chunks)>=max_pairs: return np.concatenate(chunks).astype('float32')
    return np.concatenate(chunks).astype('float32')
