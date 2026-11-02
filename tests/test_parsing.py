from ragplatform.documents import BlockKind, Document
from ragplatform.parsing import parse_document, parse_html, parse_markdown


def test_markdown_blocks():
    md = (
        "# Title\n\nIntro text\nstill intro.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "- one\n- two\n\n```\ncode\n```\n"
    )
    blocks = parse_markdown(md)
    kinds = [b.kind for b in blocks]
    assert kinds == [
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.TABLE,
        BlockKind.LIST,
        BlockKind.CODE,
    ]
    assert blocks[0].level == 1
    assert "---" not in blocks[2].text  # separator row dropped
    assert blocks[2].text.count("\n") == 1


def test_table_directly_after_paragraph_is_split_out():
    blocks = parse_markdown("Some text\n| a | b |\n| 1 | 2 |")
    assert [b.kind for b in blocks] == [BlockKind.PARAGRAPH, BlockKind.TABLE]


def test_html_blocks_and_skips_scripts():
    html = (
        "<html><head><title>T</title><script>var x=1</script></head><body>"
        "<h2>Limits</h2><p>Read  this.</p>"
        "<table><tr><th>Role</th><th>Limit</th></tr><tr><td>Manager</td><td>$5,000</td></tr></table>"
        "<ul><li>first</li><li>second</li></ul></body></html>"
    )
    blocks = parse_html(html)
    assert [b.kind for b in blocks] == [
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.TABLE,
        BlockKind.LIST,
    ]
    assert blocks[1].text == "Read this."
    assert blocks[2].text.splitlines()[1] == "| Manager | $5,000 |"
    assert "var x" not in " ".join(b.text for b in blocks)


def test_plain_text_paragraphs():
    doc = Document("d", "s", "t", "one\n\ntwo\n", content_type="text/plain")
    assert [b.text for b in parse_document(doc)] == ["one", "two"]
