"""Retrieval benchmark run against a tenant's golden set.

Golden set format (JSONL), one question per line:
    {"id": "q1", "question": "...", "answer": "exact answer string", "doc_id": "source:path"}

Metrics
- answer_recall@k : share of questions where a top-k chunk contains the answer string
- doc_recall@k    : share of questions where a top-k chunk comes from the right document
- mrr             : mean reciprocal rank of the first chunk containing the answer
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .store import Hit

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.replace("|", " ")).strip().lower()


@dataclass
class BenchResult:
    questions: int
    k: int
    answer_recall: float
    doc_recall: float
    mrr: float
    misses: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def load_golden(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(retrieve, golden: list[dict], k: int) -> BenchResult:  # noqa: ANN001
    ans_hits = doc_hits = 0
    rr_total = 0.0
    misses: list[str] = []
    for g in golden:
        hits: list[Hit] = retrieve(g["question"])[:k]
        target = _norm(g["answer"])
        rank = next((i for i, h in enumerate(hits, 1) if target in _norm(h.chunk.text)), None)
        if rank:
            ans_hits += 1
            rr_total += 1.0 / rank
        else:
            misses.append(g["id"])
        if any(h.chunk.doc_id == g["doc_id"] for h in hits):
            doc_hits += 1
    n = len(golden) or 1
    return BenchResult(
        questions=len(golden),
        k=k,
        answer_recall=ans_hits / n,
        doc_recall=doc_hits / n,
        mrr=rr_total / n,
        misses=misses,
    )
