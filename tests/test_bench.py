from ragplatform.bench import evaluate
from ragplatform.documents import Chunk
from ragplatform.store import Hit


def _hit(text: str, doc: str = "d1") -> Hit:
    return Hit(Chunk(f"{doc}#0", doc, "t", "s", "T", text), 1.0)


def test_evaluate_metrics():
    golden = [
        {"id": "a", "question": "q1", "answer": "$5,000", "doc_id": "d1"},
        {"id": "b", "question": "q2", "answer": "12 days", "doc_id": "d2"},
    ]
    results = {
        "q1": [_hit("nothing"), _hit("| Manager | $5,000 |")],
        "q2": [_hit("other", doc="d3")],
    }
    r = evaluate(lambda q: results[q], golden, k=5)
    assert r.answer_recall == 0.5
    assert r.doc_recall == 0.5
    assert r.mrr == 0.25
    assert r.misses == ["b"]
