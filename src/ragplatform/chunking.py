"""Chunkers.

`FixedWindowChunker` is the common baseline: slide a token window over the flattened text.
`LayoutAwareChunker` packs whole layout blocks within a section, never splits a table or
list mid-row unless the block alone exceeds the budget (then splits on row/item boundaries
and repeats the table header), and prefixes every chunk with its heading path so a table
row like "| International | $350 |" still says which policy and which section it belongs to.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol

from .documents import Block, BlockKind, Chunk, Document
from .parsing import parse_document

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def n_tokens(text: str) -> int:
    """Whitespace token count. Cheap, deterministic, and close enough for budget packing."""
    return len(text.split())


class Chunker(Protocol):
    def chunk(self, doc: Document, tenant_id: str) -> list[Chunk]: ...


def _make_chunk(
    doc: Document,
    tenant_id: str,
    i: int,
    text: str,
    path: tuple[str, ...] = (),
    prefix: str = "",
) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}#{i}",
        doc_id=doc.doc_id,
        tenant_id=tenant_id,
        source=doc.source,
        title=doc.title,
        text=text,
        heading_path=path,
        metadata=dict(doc.metadata),
        prefix=prefix,
    )


class FixedWindowChunker:
    def __init__(self, max_tokens: int = 220, overlap_tokens: int = 30) -> None:
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap must be smaller than the window")
        self.max_tokens = max_tokens
        self.overlap = overlap_tokens

    def chunk(self, doc: Document, tenant_id: str) -> list[Chunk]:
        blocks = parse_document(doc)
        words = " ".join(b.text for b in blocks).split()
        step = self.max_tokens - self.overlap
        out = []
        for i, start in enumerate(range(0, max(len(words), 1), step)):
            window = words[start : start + self.max_tokens]
            if not window:
                break
            out.append(_make_chunk(doc, tenant_id, i, " ".join(window)))
            if start + self.max_tokens >= len(words):
                break
        return out


class LayoutAwareChunker:
    def __init__(self, max_tokens: int = 220, overlap_tokens: int = 0) -> None:
        self.max_tokens = max_tokens
        # overlap is intentionally unused: section prefixes give context without duplication
        self.overlap = overlap_tokens

    # -- splitting oversize blocks on structural boundaries -----------------------------
    def _split_block(self, block: Block, budget: int) -> list[str]:
        if n_tokens(block.text) <= budget:
            return [block.text]
        if block.kind == BlockKind.TABLE:
            rows = block.text.splitlines()
            header, body = rows[0], rows[1:]
            return self._pack(body, budget - n_tokens(header), prefix=header)
        if block.kind in (BlockKind.LIST, BlockKind.CODE):
            return self._pack(block.text.splitlines(), budget)
        return self._pack(_SENTENCE.split(block.text), budget, joiner=" ")

    def _pack(
        self, units: Iterable[str], budget: int, prefix: str | None = None, joiner: str = "\n"
    ) -> list[str]:
        parts: list[str] = []
        cur: list[str] = []
        size = 0
        for u in units:
            t = n_tokens(u)
            if cur and size + t > budget:
                parts.append(joiner.join(cur))
                cur, size = [], 0
            cur.append(u)
            size += t
        if cur:
            parts.append(joiner.join(cur))
        if prefix:
            parts = [prefix + "\n" + p for p in parts]
        return parts

    def chunk(self, doc: Document, tenant_id: str) -> list[Chunk]:
        blocks = parse_document(doc)
        out: list[Chunk] = []
        stack: list[tuple[int, str]] = []  # (level, heading)
        pending: list[str] = []
        pending_tokens = 0

        def path() -> tuple[str, ...]:
            return tuple(h for _, h in stack)

        def header_for(p: tuple[str, ...]) -> str:
            return " > ".join((doc.title, *p))

        def emit() -> None:
            nonlocal pending, pending_tokens
            if pending:
                p = path()
                body = "\n\n".join(pending)
                head = header_for(p)
                out.append(_make_chunk(doc, tenant_id, len(out), f"{head}\n\n{body}", p, head))
            pending, pending_tokens = [], 0

        title_key = doc.title.strip().casefold()
        for b in blocks:
            if b.kind == BlockKind.HEADING:
                emit()  # a new section always starts a new chunk
                while stack and stack[-1][0] >= b.level:
                    stack.pop()
                # The H1 usually *is* the document title, which the breadcrumb already starts
                # with. Pushing it again doubles the title's term weight in every chunk, and
                # short boilerplate chunks then outrank the section that holds the answer.
                if b.text.strip().casefold() != title_key:
                    stack.append((b.level, b.text))
                continue
            budget = self.max_tokens - n_tokens(header_for(path()))
            for piece in self._split_block(b, max(budget, 20)):
                t = n_tokens(piece)
                if pending and pending_tokens + t > budget:
                    emit()
                pending.append(piece)
                pending_tokens += t
        emit()
        return out


def make_chunker(strategy: str, max_tokens: int, overlap_tokens: int) -> Chunker:
    if strategy == "layout":
        return LayoutAwareChunker(max_tokens, overlap_tokens)
    if strategy == "fixed":
        return FixedWindowChunker(max_tokens, overlap_tokens)
    raise ValueError(f"unknown chunking strategy {strategy!r}")
