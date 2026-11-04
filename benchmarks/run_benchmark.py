"""Compare chunking strategies and retrieval modes on every benchmark tenant.

python benchmarks/generate_corpus.py
python benchmarks/run_benchmark.py              # prints a Markdown table, writes results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ragplatform.bench import evaluate, load_golden
from ragplatform.config import load_config
from ragplatform.platform import Platform

HERE = Path(__file__).parent
CONFIGS = [
    # (label, chunking strategy, retrieval mode, rerank)
    ("fixed window + BM25", "fixed", "bm25", False),
    ("fixed window + vector", "fixed", "vector", False),
    ("fixed window + hybrid + rerank", "fixed", "hybrid", True),
    ("layout-aware + BM25", "layout", "bm25", False),
    ("layout-aware + vector", "layout", "vector", False),
    ("layout-aware + hybrid", "layout", "hybrid", False),
    ("layout-aware + hybrid + rerank", "layout", "hybrid", True),
]


def main() -> int:
    base = load_config(HERE.parent / "config" / "bench.yaml")
    rows, results = [], []
    for tenant in base.tenants:
        golden = load_golden(tenant.benchmark)
        for label, strategy, mode, rerank in CONFIGS:
            cfg = base.model_copy(deep=True)
            t = cfg.tenant(tenant.tenant_id)
            t.chunking.strategy = strategy
            t.retrieval.mode = mode
            t.retrieval.rerank = rerank
            p = Platform(cfg)
            rep = p.ingest(t.tenant_id, persist=False)
            res = evaluate(
                lambda q, p=p, tid=t.tenant_id: p.retrieve(tid, q), golden, k=t.retrieval.top_k
            )
            results.append(
                {
                    "tenant": t.tenant_id,
                    "config": label,
                    "chunks": rep.chunks,
                    **{k: v for k, v in res.as_dict().items() if k != "misses"},
                }
            )
            rows.append(
                f"| {t.tenant_id} | {label} | {rep.chunks} | {res.answer_recall:.1%} | "
                f"{res.attributable_recall:.1%} | {res.doc_recall:.1%} | {res.mrr:.3f} |"
            )
    k = base.tenants[0].retrieval.top_k
    print(f"| tenant | config | chunks | answer recall@{k} | attributable@{k} | doc recall | MRR |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
