"""CI gate: the default tenant configuration must hold its retrieval quality on the synthetic
benchmark. Regenerates the corpus (seeded, deterministic) so the gate needs no checked-in data."""

from pathlib import Path

import pytest

from ragplatform.bench import evaluate, load_golden
from ragplatform.config import load_config
from ragplatform.platform import Platform

ROOT = Path(__file__).resolve().parents[1]
FLOORS = {  # measured values minus a small margin; raise them when retrieval improves
    "acme-insurance": {"answer_recall": 0.97, "attributable_recall": 0.97, "mrr": 0.92},
    "northwind-bank": {"answer_recall": 0.98, "attributable_recall": 0.98, "mrr": 0.95},
}


@pytest.fixture(scope="module")
def bench_config():
    import sys

    sys.path.insert(0, str(ROOT / "benchmarks"))
    import generate_corpus

    generate_corpus.build()
    return load_config(ROOT / "config" / "bench.yaml")


@pytest.mark.parametrize("tenant_id", sorted(FLOORS))
def test_default_config_meets_floor(bench_config, tenant_id):
    t = bench_config.tenant(tenant_id)
    p = Platform(bench_config)
    p.ingest(tenant_id, persist=False)
    res = evaluate(
        lambda q: p.retrieve(tenant_id, q), load_golden(t.benchmark), k=t.retrieval.top_k
    )
    for metric, floor in FLOORS[tenant_id].items():
        assert getattr(res, metric) >= floor, f"{tenant_id} {metric}={getattr(res, metric):.3f}"
