"""Okapi BM25 over an inverted index."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from ..text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.avgdl = 0.0

    def __len__(self) -> int:
        return len(self.doc_len)

    def add(self, texts: list[str]) -> None:
        for text in texts:
            idx = len(self.doc_len)
            tf = Counter(tokenize(text))
            self.doc_len.append(sum(tf.values()))
            for term, c in tf.items():
                self.postings[term].append((idx, c))
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

    def _idf(self, term: str) -> float:
        n = len(self.doc_len)
        df = len(self.postings.get(term, ()))
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf(term)
            for idx, tf in self.postings.get(term, ()):
                dl = self.doc_len[idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[idx] += idf * tf * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    # -- persistence ----------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_len": self.doc_len,
            "postings": {t: p for t, p in self.postings.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> BM25Index:
        idx = cls(d["k1"], d["b"])
        idx.doc_len = list(d["doc_len"])
        idx.postings = defaultdict(
            list, {t: [tuple(x) for x in p] for t, p in d["postings"].items()}
        )
        idx.avgdl = sum(idx.doc_len) / len(idx.doc_len) if idx.doc_len else 0.0
        return idx
