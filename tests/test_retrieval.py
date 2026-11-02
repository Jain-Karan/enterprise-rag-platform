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
