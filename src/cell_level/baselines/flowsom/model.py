"""Small, dependency-light FlowSOM implementation for cytometry baselines."""
from __future__ import annotations
import numpy as np
from sklearn.cluster import MiniBatchKMeans


class FlowSOM:
    def __init__(self, xdim=10, ydim=10, sigma=2.0, learning_rate=0.5, random_state=42):
        self.xdim, self.ydim = xdim, ydim
        self.sigma, self.learning_rate = sigma, learning_rate
        self.rng = np.random.default_rng(random_state)
        self.codebook = None
        self.meta_model = None
        self.node_to_meta = None
        self.meta_to_class = None

    @property
    def n_nodes(self): return self.xdim * self.ydim

    def _grid_distance2(self):
        g = np.stack(np.meshgrid(np.arange(self.xdim), np.arange(self.ydim), indexing="ij"), -1).reshape(-1, 2)
        return ((g[:, None] - g[None, :]) ** 2).sum(-1)

    def fit(self, x, epochs=10, batch_size=4096):
        x = np.asarray(x, dtype=np.float32)
        if len(x) < self.n_nodes: raise ValueError("sample smaller than SOM grid")
        self.codebook = x[self.rng.choice(len(x), self.n_nodes, replace=False)].copy()
        gdist2 = self._grid_distance2().astype(np.float32)
        for epoch in range(epochs):
            order = self.rng.permutation(len(x))
            lr = self.learning_rate * (1 - epoch / max(epochs, 1)) + 0.02
            radius = self.sigma * (1 - epoch / max(epochs, 1)) + 0.25
            for start in range(0, len(x), batch_size):
                z = x[order[start:start + batch_size]]
                d2 = ((z[:, None, :] - self.codebook[None, :, :]) ** 2).sum(-1)
                bmu = d2.argmin(1)
                h = np.exp(-gdist2[bmu] / (2 * radius * radius)).astype(np.float32)
                denom = h.sum(0)[:, None] + 1e-8
                target = (h.T @ z) / denom
                self.codebook += lr * (target - self.codebook)
        return self

    def nodes(self, x, batch_size=200000):
        out=[]
        for s in range(0, len(x), batch_size):
            z=np.asarray(x[s:s+batch_size], dtype=np.float32)
            out.append(((z[:,None,:]-self.codebook[None,:,:])**2).sum(-1).argmin(1))
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def fit_meta(self, n_clusters=30, random_state=42):
        self.meta_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=20, batch_size=256)
        self.node_to_meta = self.meta_model.fit_predict(self.codebook)
        return self

    def assign_meta(self, x): return self.node_to_meta[self.nodes(x)]

    def fit_label_mapping(self, x, y, n_classes):
        m=self.assign_meta(x); counts=np.zeros((int(self.node_to_meta.max())+1, n_classes), dtype=np.int64)
        np.add.at(counts, (m, y.astype(int)), 1)
        self.meta_to_class=counts.argmax(1); self.meta_class_counts=counts
        return self

    def predict(self, x):
        if self.meta_to_class is None: raise RuntimeError("fit_label_mapping first")
        return self.meta_to_class[self.assign_meta(x)]

    def state_dict(self):
        return {k:getattr(self,k) for k in ("xdim","ydim","sigma","learning_rate","codebook","node_to_meta","meta_to_class","meta_class_counts")}

    @classmethod
    def from_state_dict(cls, d):
        o=cls(d["xdim"],d["ydim"],d["sigma"],d["learning_rate"])
        for k,v in d.items(): setattr(o,k,v)
        return o
