"""FAISS feature index used for retrieval-based timbre matching."""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

from ..paths import replace_file

log = logging.getLogger(__name__)

MAX_VECTORS_BEFORE_KMEANS = 200_000
KMEANS_CENTROIDS = 10_000


def build_index(features: np.ndarray, path: str | Path, seed: int = 0) -> Path:
    """Train an IVF index on ``features [N, dim]`` and write it to ``path``."""
    path = Path(path)
    features = np.ascontiguousarray(features, dtype=np.float32)
    features = features[np.isfinite(features).all(axis=1)]
    if len(features) < 2:
        raise ValueError("Not enough feature frames to build an index.")
    rng = np.random.default_rng(seed)
    features = features[rng.permutation(len(features))]

    if len(features) > MAX_VECTORS_BEFORE_KMEANS:
        log.info("Reducing %d vectors to %d centroids", len(features), KMEANS_CENTROIDS)
        kmeans = faiss.Kmeans(
            features.shape[1], KMEANS_CENTROIDS, niter=20, seed=seed, verbose=False
        )
        kmeans.train(features)
        features = np.ascontiguousarray(kmeans.centroids, dtype=np.float32)

    n_lists = int(min(16 * np.sqrt(len(features)), len(features) // 39))
    n_lists = max(n_lists, 1)
    index = faiss.index_factory(features.shape[1], f"IVF{n_lists},Flat")
    faiss.extract_index_ivf(index).nprobe = 1
    index.train(features)
    for start in range(0, len(features), 8192):
        index.add(features[start : start + 8192])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    faiss.write_index(index, str(tmp))
    replace_file(tmp, path)
    return path


class FeatureIndex:
    """Loaded index plus the stored vectors used for blending."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.index = faiss.read_index(str(self.path))
        self.vectors = self.index.reconstruct_n(0, self.index.ntotal)
        self.dim = self.index.d

    def blend(self, feats: np.ndarray, rate: float, k: int = 8) -> np.ndarray:
        """Replace ``rate`` of each frame with its inverse-distance weighted
        neighbours from the training data."""
        if rate <= 0 or self.index.ntotal == 0:
            return feats
        k = min(k, self.index.ntotal)
        queries = np.ascontiguousarray(feats, dtype=np.float32)
        distances, ids = self.index.search(queries, k)
        valid = ids >= 0
        # Missing neighbours come back as id -1 with a huge distance.
        distances = np.clip(distances.astype(np.float64), 1e-6, 1e30)
        weights = np.where(valid, 1.0 / np.square(distances), 0.0)
        totals = weights.sum(axis=1, keepdims=True)
        has_match = totals[:, 0] > 0
        weights = np.divide(
            weights, totals, out=np.zeros_like(weights), where=totals > 0
        )
        retrieved = np.einsum(
            "nk,nkd->nd", weights, self.vectors[np.where(valid, ids, 0)]
        )
        out = feats.copy()
        out[has_match] = rate * retrieved[has_match] + (1 - rate) * feats[has_match]
        return out
