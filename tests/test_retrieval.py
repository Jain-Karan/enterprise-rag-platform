import numpy as np

from ragplatform.embeddings import HashingEmbedder
from ragplatform.index import BM25Index, VectorIndex, reciprocal_rank_fusion
from ragplatform.rerank import CoverageReranker, _min_window
from ragplatform.text import tokenize


def test_tokenize_folds_plurals_and_drops_stopwords():
    assert tokenize("The limits for Adjusters") == ["limit", "adjuster"]


def test_bm25_prefers_matching_doc():
    idx = BM25Index()
    idx.add(["lodging limit tokyo", "meal allowance paris", "lodging paris"])
    top = idx.search("tokyo lodging", 3)
    assert top[0][0] == 0
    assert BM25Index.from_dict(idx.to_dict()).search("tokyo lodging", 3) == top


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(dim=256)
    a, b = e.embed(["settlement authority"]), e.embed(["settlement authority"])
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a[0]) - 1) < 1e-5


def test_vector_index_ranks_by_cosine():
    e = HashingEmbedder(dim=256)
    idx = VectorIndex(256)
    idx.add(e.embed(["claims settlement authority", "travel meals", "it access review"]))
    assert idx.search(e.embed(["settlement authority for claims"])[0], 2)[0][0] == 0


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[(1, 9.0), (2, 5.0)], [(2, 0.9), (3, 0.8)]])
    assert fused[0][0] == 2


def test_min_window():
    assert _min_window({"a": [0, 10], "b": [9]}) == 2
    assert _min_window({}) is None


def test_coverage_reranker_scores_full_coverage_higher():
    r = CoverageReranker()
    assert r.score("senior adjuster limit", "Senior Adjuster limit is $5") > r.score(
        "senior adjuster limit", "limit for travel"
    )


def test_long_answer_section_beats_short_boilerplate():
    """Regression: breadcrumb terms used to be length-normalised with the body, so every short
    boilerplate chunk of the right document outranked the long section holding the answer."""
    from ragplatform.chunking import LayoutAwareChunker
    from ragplatform.config import RetrievalConfig
    from ragplatform.documents import Document
    from ragplatform.store import TenantIndex

    filler = "Owners review this document every year and record any exception. " * 6
    rows = "\n".join(f"| Control {i} | ${i}0,000 |" for i in range(12))
    md = (
        "# Payments Runbook: Card Settlement\n\n## Training\n\nStaff train yearly.\n\n"
        "## Definitions\n\n- Limit: a cap.\n\n"
        f"## Cutoff and Limits\n\n{filler}\n\n| Control | Value |\n|---|---|\n"
        f"| Single payment limit | $2,337,065 |\n{rows}\n\n## Document Control\n\nOwned by ops."
    )
    doc = Document("s:card.md", "s", "Payments Runbook: Card Settlement", md)
    idx = TenantIndex("t", HashingEmbedder(64))
    idx.add(LayoutAwareChunker().chunk(doc, "t"))
    cfg = RetrievalConfig(mode="bm25", rerank=False, top_k=1)
    hits = idx.search("single payment limit for card settlement payments", cfg)
    assert "$2,337,065" in hits[0].chunk.text


def test_weighted_rrf_down_weights_a_retriever():
    strong = [(1, 9.0), (2, 8.0)]
    weak = [(2, 0.9), (3, 0.8)]
    assert reciprocal_rank_fusion([strong, weak])[0][0] == 2  # equal weights: agreement wins
    # a near-zero weight leaves the strong retriever's order intact
    assert reciprocal_rank_fusion([strong, weak], weights=[1.0, 0.01])[0][0] == 1


def test_rrf_weight_count_must_match():
    import pytest

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[(1, 1.0)]], weights=[1.0, 2.0])
