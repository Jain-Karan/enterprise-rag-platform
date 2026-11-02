"""Tokenization shared by BM25, the hashing embedder, and the reranker."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "does",
        "do",
        "did",
        "can",
        "our",
        "we",
        "you",
        "your",
        "their",
        "there",
        "than",
        "then",
        "them",
        "they",
        "i",
        "me",
        "my",
    ]
)


def _norm(tok: str) -> str:
    # light plural folding so "limits" matches "limit"; avoids a stemmer dependency
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [_norm(t) for t in toks]
