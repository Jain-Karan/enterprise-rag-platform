"""Second-stage rerankers applied to the fused candidate set.

`CoverageReranker` is a fast, deterministic default: it rewards candidates that cover more of
the query's distinct terms and that cover them close together (a minimal-window proximity
score). `CrossEncoderReranker` uses a sentence-transformers cross-encoder when installed.
"""

from __future__ import annotations

import math
from typing import Protocol

from .documents import Chunk
from .text import tokenize


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]: ...


def _min_window(positions: dict[str, list[int]]) -> int | None:
    """Smallest token span containing at least one occurrence of every term in `positions`."""
    events = sorted((p, t) for t, ps in positions.items() for p in ps)
    need = len(positions)
    if need == 0:
        return None
    counts: dict[str, int] = {}
    best: int | None = None
    left = 0
    for pos, term in events:
        counts[term] = counts.get(term, 0) + 1
        while len(counts) == need:
            span = pos - events[left][0] + 1
            best = span if best is None else min(best, span)
            lt = events[left][1]
            counts[lt] -= 1
            if counts[lt] == 0:
                del counts[lt]
            left += 1
    return best


class CoverageReranker:
    """Rewards candidates that cover the query's *distinguishing* terms, close together.

    Term weights are IDF computed over the candidate set itself: a term that every candidate
    contains ("wire" when all candidates come from wire runbooks) says nothing about which
    candidate is right, while a term only a few contain ("international") says a lot.
    """

    def __init__(self, prior_weight: float = 0.35) -> None:
        self.prior_weight = prior_weight

    @staticmethod
    def _weights(q: set[str], docs: list[set[str]]) -> dict[str, float]:
        n = len(docs)
        w = {}
        for t in q:
            df = sum(1 for d in docs if t in d)
            if df:  # terms no candidate contains can't separate candidates
                w[t] = math.log(1 + n / df)
        return w

    def score(self, query: str, text: str, weights: dict[str, float] | None = None) -> float:
        q = set(tokenize(query))
        if not q:
            return 0.0
        weights = weights if weights is not None else dict.fromkeys(q, 1.0)
        total = sum(weights.values())
        if not total:
            return 0.0
        toks = tokenize(text)
        positions: dict[str, list[int]] = {}
        for i, t in enumerate(toks):
            if t in weights:
                positions.setdefault(t, []).append(i)
        coverage = sum(weights[t] for t in positions) / total
        window = _min_window(positions)
        proximity = (len(positions) / window) if window else 0.0
        return 0.75 * coverage + 0.25 * proximity

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        q = set(tokenize(query))
        weights = self._weights(q, [set(tokenize(c.text)) for c, _ in candidates])
        top_prior = max(s for _, s in candidates) or 1.0
        scored = [
            (
                c,
                (1 - self.prior_weight) * self.score(query, c.text, weights)
                + self.prior_weight * s / top_prior,
            )
            for c, s in candidates
        ]
        return sorted(scored, key=lambda cs: -cs[1])


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder  # optional dependency

        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        scores = self.model.predict([(query, c.text) for c, _ in candidates])
        return sorted(
            ((c, float(s)) for (c, _), s in zip(candidates, scores, strict=True)),
            key=lambda cs: -cs[1],
        )
