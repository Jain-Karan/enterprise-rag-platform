"""Reciprocal Rank Fusion: combine ranked lists without having to calibrate BM25 and cosine
scores against each other."""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]], k: int = 60, limit: int | None = None
) -> list[tuple[int, float]]:
    fused: dict[int, float] = defaultdict(float)
    for ranking in ranked_lists:
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] += 1.0 / (k + rank + 1)
    out = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return out[:limit] if limit else out
