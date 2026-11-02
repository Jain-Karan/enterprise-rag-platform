"""Exact cosine-similarity index (vectors are pre-normalised). Fine to ~1M chunks per tenant;
swap for pgvector / Azure AI Search behind the same interface beyond that."""

from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.matrix = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def add(self, vectors: np.ndarray) -> None:
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self.matrix = np.vstack([self.matrix, vectors.astype(np.float32)])

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not len(self):
            return []
        sims = self.matrix @ query_vec.astype(np.float32)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(int(i), float(sims[i])) for i in top]
