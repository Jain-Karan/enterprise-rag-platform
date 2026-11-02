"""Per-tenant index storage and retrieval.

Isolation model: every tenant gets its own directory, its own BM25 + vector index, and its own
chunk table. There is no shared index with a tenant filter, so a missing `WHERE tenant = ?`
cannot leak data across tenants. Retrieval also re-checks `chunk.tenant_id` on the way out.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import RetrievalConfig
from .documents import Chunk
from .embeddings import Embedder
from .index import BM25Index, VectorIndex, reciprocal_rank_fusion
from .rerank import CoverageReranker, Reranker


class TenantIsolationError(RuntimeError):
    pass


@dataclass
class Hit:
    chunk: Chunk
    score: float


class TenantIndex:
    def __init__(self, tenant_id: str, embedder: Embedder) -> None:
        self.tenant_id = tenant_id
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.bm25 = BM25Index()  # chunk bodies
        self.bm25_head = BM25Index()  # breadcrumbs: title > section path
        self.vectors = VectorIndex(embedder.dim)

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], batch_size: int = 256) -> None:
        for c in chunks:
            if c.tenant_id != self.tenant_id:
                raise TenantIsolationError(
                    f"chunk {c.chunk_id} belongs to {c.tenant_id!r}, not {self.tenant_id!r}"
                )
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.chunks.extend(batch)
            self.bm25.add([c.body for c in batch])
            self.bm25_head.add([c.heading_text for c in batch])
            self.vectors.add(self.embedder.embed([c.text for c in batch]))

    def _bm25(self, query: str, cfg: RetrievalConfig) -> list[tuple[int, float]]:
        """Two-field BM25: body score + heading_weight * breadcrumb score."""
        scores: dict[int, float] = {}
        for i, s in self.bm25.search(query, len(self.chunks)):
            scores[i] = s
        if cfg.heading_weight:
            for i, s in self.bm25_head.search(query, len(self.chunks)):
                scores[i] = scores.get(i, 0.0) + cfg.heading_weight * s
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[: cfg.candidate_k]

    def search(
        self, query: str, cfg: RetrievalConfig, reranker: Reranker | None = None
    ) -> list[Hit]:
        if not self.chunks:
            return []
        lists = []
        if cfg.mode in ("hybrid", "bm25"):
            lists.append(self._bm25(query, cfg))
        if cfg.mode in ("hybrid", "vector"):
            qv = self.embedder.embed([query])[0]
            lists.append(self.vectors.search(qv, cfg.candidate_k))
        fused = (
            reciprocal_rank_fusion(lists, k=cfg.rrf_k, limit=cfg.candidate_k)
            if len(lists) > 1
            else lists[0]
        )
        candidates = [(self.chunks[i], s) for i, s in fused]
        if cfg.rerank:
            candidates = (reranker or CoverageReranker()).rerank(query, candidates)
        hits = [Hit(c, s) for c, s in candidates[: cfg.top_k]]
        for h in hits:  # defence in depth
            if h.chunk.tenant_id != self.tenant_id:
                raise TenantIsolationError(h.chunk.chunk_id)
        return hits

    # -- persistence ------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / ".tmp"
        tmp.mkdir(exist_ok=True)
        with (tmp / "chunks.jsonl").open("w") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        (tmp / "bm25.json").write_text(json.dumps(self.bm25.to_dict()))
        (tmp / "bm25_head.json").write_text(json.dumps(self.bm25_head.to_dict()))
        np.save(tmp / "vectors.npy", self.vectors.matrix)
        (tmp / "manifest.json").write_text(
            json.dumps(
                {
                    "tenant_id": self.tenant_id,
                    "chunks": len(self.chunks),
                    "dim": self.embedder.dim,
                    "built_at": time.time(),
                }
            )
        )
        # swap files in only after everything is written, so a crash mid-save
        # never leaves a half-written index behind
        for p in tmp.iterdir():
            p.replace(directory / p.name)
        tmp.rmdir()

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> TenantIndex:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["dim"] != embedder.dim:
            raise ValueError(
                f"index built with dim={manifest['dim']}, embedder has dim={embedder.dim}"
            )
        idx = cls(manifest["tenant_id"], embedder)
        with (directory / "chunks.jsonl").open() as f:
            idx.chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        idx.bm25 = BM25Index.from_dict(json.loads((directory / "bm25.json").read_text()))
        head = directory / "bm25_head.json"
        if head.exists():
            idx.bm25_head = BM25Index.from_dict(json.loads(head.read_text()))
        else:  # index written before the heading field existed
            idx.bm25_head.add([c.heading_text for c in idx.chunks])
        idx.vectors.matrix = np.load(directory / "vectors.npy")
        return idx
