"""SOM density features and the trained FlowCat CNN architecture."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common.patient_baselines import numeric_events
from common.patient_paper import marker_frame, scaled_values


def som_density(frame, grid_size: int, max_som_events: int, seed: int) -> np.ndarray:
    values = scaled_values(frame, list(frame.columns)).astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(values) > max_som_events:
        values = values[rng.choice(len(values), max_som_events, replace=False)]
    width = values.shape[1]
    weights = rng.random((grid_size, grid_size, width), dtype=np.float32)
    coords = np.stack(np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij"), axis=-1).astype(np.float32)
    steps = max(200, min(2000, len(values)))
    samples = values[rng.choice(len(values), steps, replace=len(values) < steps)]
    for t, x in enumerate(samples):
        diff = weights - x
        dist = np.sum(diff * diff, axis=-1)
        bmu = np.unravel_index(int(np.argmin(dist)), dist.shape)
        frac = 1.0 - t / max(steps - 1, 1)
        lr = 0.3 * frac + 0.02
        sigma = max(0.75, grid_size * 0.35 * frac)
        d2 = np.sum((coords - np.asarray(bmu, dtype=np.float32)) ** 2, axis=-1)
        influence = np.exp(-d2 / (2 * sigma * sigma)).astype(np.float32)
        weights += lr * influence[..., None] * (x - weights)
    flat = weights.reshape(-1, width)
    dist = ((values[:, None, :] - flat[None, :, :]) ** 2).sum(axis=-1)
    bmu = np.argmin(dist, axis=1)
    hist = np.bincount(bmu, minlength=grid_size * grid_size).reshape(grid_size, grid_size).astype(np.float32)
    return hist / (hist.sum() + 1e-8)


def tube_order(rows) -> list[str]:
    return sorted(rows["_tube_id"].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)


def patient_tensor(rows, patient_id: str, tubes: list[str], grid_size: int, max_events: int, max_som_events: int, seed: int) -> np.ndarray:
    maps = []
    for tube in tubes:
        hit = rows[(rows["_patient_id"].astype(str) == str(patient_id)) & (rows["_tube_id"].astype(str) == tube)]
        if hit.empty:
            maps.append(np.zeros((grid_size, grid_size), dtype=np.float32))
            continue
        frame = marker_frame(hit["_csv_path"].iloc[0], max_events)
        maps.append(som_density(frame, grid_size, max_som_events, seed + hash((patient_id, tube)) % 100000))
    return np.stack(maps).astype(np.float32)


class TensorDataset(Dataset):
    def __init__(self, tensors: np.ndarray, labels: np.ndarray) -> None:
        self.x = torch.from_numpy(tensors.astype(np.float32))
        self.y = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class FlowCatCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def predict(model, tensors, device, batch_size):
    loader = DataLoader(TensorDataset(tensors, np.zeros(len(tensors), dtype=np.int64)), batch_size=batch_size)
    parts = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            parts.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
    return np.vstack(parts)


def flowcat_features(path, max_events=12000, grid_size=10):
    x=numeric_events(path,max_events)
    # Stable marker-pair selection: highest variance two channels for each panel.
    order=np.argsort(np.nanvar(x,axis=0))[::-1]; a,b=order[:2] if x.shape[1]>1 else (0,0)
    u=x[:,a]; v=x[:,b]
    def scale(z):
        lo,hi=np.quantile(z,[.01,.99]); return np.clip((z-lo)/(hi-lo+1e-8),0,1)
    h,_,_=np.histogram2d(scale(u),scale(v),bins=grid_size,range=((0,1),(0,1)))
    return (h/h.sum()).ravel().astype('float32')
