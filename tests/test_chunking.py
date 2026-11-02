import pytest

from ragplatform.chunking import FixedWindowChunker, LayoutAwareChunker, n_tokens
from ragplatform.documents import Document


def _doc(md: str) -> Document:
    return Document("src:doc.md", "src", "Claims Standard: Auto", md)


def test_layout_chunks_follow_sections():
    md = (
        "## Authority\n\nIntro.\n\n| Role | Limit |\n|---|---|\n| Adjuster | $10 |\n\n"
        "## SLAs\n\nFast."
    )
    chunks = LayoutAwareChunker(max_tokens=100).chunk(_doc(md), "t1")
    assert [c.heading_path for c in chunks] == [("Authority",), ("SLAs",)]
    assert chunks[0].text.startswith("Claims Standard: Auto > Authority")
    assert "| Adjuster | $10 |" in chunks[0].text
    assert all(c.tenant_id == "t1" for c in chunks)


def test_oversize_table_splits_on_rows_and_repeats_header():
    rows = "\n".join(f"| Role {i} | ${i * 1000} limit for this role |" for i in range(40))
    md = f"## Authority\n\n| Role | Limit |\n|---|---|\n{rows}\n"
    chunks = LayoutAwareChunker(max_tokens=60).chunk(_doc(md), "t1")
    assert len(chunks) > 1
    for c in chunks:
        assert "| Role | Limit |" in c.text
        assert n_tokens(c.text) <= 60
        # no row is cut in half
        for line in c.text.splitlines():
            if line.startswith("| Role "):
                assert line.endswith("|")


def test_nested_headings_build_path():
    md = "## A\n\n### B\n\ntext b\n\n## C\n\ntext c"
    chunks = LayoutAwareChunker().chunk(_doc(md), "t")
    assert [c.heading_path for c in chunks] == [("A", "B"), ("C",)]


def test_fixed_window_overlap():
    words = " ".join(f"w{i}" for i in range(100))
    chunks = FixedWindowChunker(max_tokens=40, overlap_tokens=10).chunk(
        Document("d", "s", "t", words, content_type="text/plain"), "t"
    )
    assert [len(c.text.split()) for c in chunks] == [40, 40, 40]
    assert chunks[1].text.split()[0] == "w30"


def test_fixed_window_rejects_bad_overlap():
    with pytest.raises(ValueError):
        FixedWindowChunker(10, 10)


def test_h1_matching_title_is_not_repeated_in_breadcrumb():
    md = "# Claims Standard: Auto\n\n## Authority\n\n| Role | Limit |\n|---|---|\n| A | $1 |"
    chunks = LayoutAwareChunker().chunk(_doc(md), "t")
    assert chunks[0].heading_path == ("Authority",)
    assert chunks[0].text.splitlines()[0] == "Claims Standard: Auto > Authority"


def test_h1_different_from_title_is_kept():
    md = "# Appendix\n\ntext"
    chunks = LayoutAwareChunker().chunk(_doc(md), "t")
    assert chunks[0].heading_path == ("Appendix",)
