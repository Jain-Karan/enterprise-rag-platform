"""Reciprocal Rank Fusion: combine ranked lists without having to calibrate BM25 and cosine
scores against each other."""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
    limit: int | None = None,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """Weighted RRF. Equal weights treat every retriever as equally trustworthy, which lets a
    weak retriever (e.g. an offline embedder on domain jargon) drag down a strong one."""
    weights = weights or [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("one weight per ranked list")
    fused: dict[int, float] = defaultdict(float)
    for w, ranking in zip(weights, ranked_lists, strict=True):
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] += w / (k + rank + 1)
    out = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return out[:limit] if limit else out
