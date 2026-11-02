"""Embedders.

`HashingEmbedder` is a deterministic, dependency-free embedder (signed feature hashing over
word unigrams, bigrams and character trigrams). It lets CI and the benchmark run offline.
`LangChainEmbedder` wraps any LangChain `Embeddings` (Azure OpenAI, Bedrock, Voyage, ...) for
real deployments.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


class HashingEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        toks = tokenize(text)
        feats = [f"w:{t}" for t in toks]
        feats += [f"b:{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        for t in toks:
            padded = f"#{t}#"
            feats += [f"c:{padded[i : i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for f in self._features(text):
                h = _h(f)
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            # sublinear scaling, then L2 normalise for cosine similarity
            v = np.sign(out[row]) * np.log1p(np.abs(out[row]))
            norm = np.linalg.norm(v)
            out[row] = v / norm if norm else v
        return out


class LangChainEmbedder:
    def __init__(self, embeddings, dim: int) -> None:  # noqa: ANN001 - any LangChain Embeddings
        self._e = embeddings
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(self._e.embed_documents(texts), dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return v / norms
