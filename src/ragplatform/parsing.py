"""Turn Markdown / HTML / plain text into layout blocks (headings, paragraphs, tables, lists, code).

Keeping layout structure is what lets the chunker avoid splitting a table in half or
separating a paragraph from the heading that gives it meaning.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from .documents import Block, BlockKind, Document

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def parse_markdown(text: str) -> list[Block]:
    blocks: list[Block] = []
    buf: list[str] = []
    buf_kind: BlockKind | None = None
    in_code = False

    def flush() -> None:
        nonlocal buf, buf_kind
        if buf and buf_kind is not None:
            body = "\n".join(buf).strip()
            if body:
                blocks.append(Block(buf_kind, body))
        buf, buf_kind = [], None

    for line in text.splitlines():
        if line.strip().startswith("```"):
            if in_code:
                buf.append(line)
                flush()
                in_code = False
            else:
                flush()
                in_code = True
                buf, buf_kind = [line], BlockKind.CODE
            continue
        if in_code:
            buf.append(line)
            continue

        m = _MD_HEADING.match(line)
        if m:
            flush()
            blocks.append(Block(BlockKind.HEADING, m.group(2).strip(), level=len(m.group(1))))
            continue
        if not line.strip():
            flush()
            continue
        if line.lstrip().startswith("|"):
            kind = BlockKind.TABLE
        elif _MD_LIST.match(line):
            kind = BlockKind.LIST
        elif buf_kind == BlockKind.LIST and line.startswith(("  ", "\t")):
            kind = BlockKind.LIST  # continuation of a list item
        else:
            kind = BlockKind.PARAGRAPH
        if buf_kind is not None and kind != buf_kind:
            flush()
        if kind == BlockKind.TABLE and _MD_TABLE_SEP.match(line):
            continue  # drop |---|---| separator rows; they carry no content
        buf_kind = kind
        buf.append(line)
    flush()
    return blocks


class _HTMLBlockParser(HTMLParser):
    _HEADINGS = {f"h{i}": i for i in range(1, 7)}
    _SKIP = {"script", "style", "head", "nav", "footer"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._text: list[str] = []
        self._mode: str | None = None  # heading|p|li|pre|cell
        self._heading_level = 0
        self._rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._list_items: list[str] = []
        self._list_depth = 0
        self._table_depth = 0
        self._skip_depth = 0

    # -- helpers --------------------------------------------------------------
    def _take_text(self) -> str:
        t = re.sub(r"\s+", " ", "".join(self._text)).strip()
        self._text = []
        return t

    def _flush_paragraph(self) -> None:
        t = self._take_text()
        if t:
            self.blocks.append(Block(BlockKind.PARAGRAPH, t))

    # -- HTMLParser hooks ---------------------------------------------------------
    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in self._SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self._HEADINGS:
            self._flush_paragraph()
            self._mode, self._heading_level = "heading", self._HEADINGS[tag]
        elif tag == "table":
            self._flush_paragraph()
            self._table_depth += 1
            if self._table_depth == 1:
                self._rows = []
        elif tag == "tr" and self._table_depth:
            self._row = []
        elif tag in ("td", "th") and self._table_depth:
            self._text = []
            self._mode = "cell"
        elif tag in ("ul", "ol"):
            if self._list_depth == 0:
                self._flush_paragraph()
                self._list_items = []
            self._list_depth += 1
        elif tag == "li":
            self._text = []
            self._mode = "li"
        elif tag == "pre":
            self._flush_paragraph()
            self._mode = "pre"
        elif tag == "p" and not self._table_depth and not self._list_depth:
            self._flush_paragraph()
            self._mode = "p"
        elif tag == "br":
            self._text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in self._HEADINGS and self._mode == "heading":
            t = self._take_text()
            if t:
                self.blocks.append(Block(BlockKind.HEADING, t, level=self._heading_level))
            self._mode = None
        elif tag in ("td", "th") and self._row is not None:
            self._row.append(self._take_text())
            self._mode = None
        elif tag == "tr" and self._row is not None:
            if any(c for c in self._row):
                self._rows.append(self._row)
            self._row = None
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1
            if self._table_depth == 0 and self._rows:
                body = "\n".join("| " + " | ".join(r) + " |" for r in self._rows)
                self.blocks.append(Block(BlockKind.TABLE, body))
                self._rows = []
        elif tag == "li" and self._mode == "li":
            t = self._take_text()
            if t:
                self._list_items.append("- " + t)
            self._mode = None
        elif tag in ("ul", "ol") and self._list_depth:
            self._list_depth -= 1
            if self._list_depth == 0 and self._list_items:
                self.blocks.append(Block(BlockKind.LIST, "\n".join(self._list_items)))
                self._list_items = []
        elif tag == "pre" and self._mode == "pre":
            t = "".join(self._text).strip()
            self._text = []
            if t:
                self.blocks.append(Block(BlockKind.CODE, t))
            self._mode = None
        elif tag == "p" and self._mode == "p":
            self._flush_paragraph()
            self._mode = None

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._text.append(data)

    def close(self) -> None:
        super().close()
        self._flush_paragraph()


def parse_html(text: str) -> list[Block]:
    p = _HTMLBlockParser()
    p.feed(text)
    p.close()
    return p.blocks


def parse_document(doc: Document) -> list[Block]:
    ct = doc.content_type.lower()
    if "html" in ct or "xml" in ct:
        return parse_html(doc.content)
    if "markdown" in ct:
        return parse_markdown(doc.content)
    # plain text: blank-line separated paragraphs
    return [
        Block(BlockKind.PARAGRAPH, p.strip())
        for p in re.split(r"\n\s*\n", doc.content)
        if p.strip()
    ]
