"""Core data types shared by connectors, parsers, chunkers, and indexes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    CODE = "code"


@dataclass(frozen=True)
class Block:
    """A layout unit extracted from a document (a heading, a paragraph, a whole table...)."""

    kind: BlockKind
    text: str
    level: int = 0  # heading depth; 0 for non-headings


@dataclass
class Document:
    """A raw document pulled from a source, before parsing."""

    doc_id: str
    source: str
    title: str
    content: str
    content_type: str = "text/markdown"  # text/markdown | text/html | text/plain
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit. `text` is what gets indexed; `heading_path` is kept for citations."""

    chunk_id: str
    doc_id: str
    tenant_id: str
    source: str
    title: str
    text: str
    heading_path: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "tenant_id": self.tenant_id,
            "source": self.source,
            "title": self.title,
            "text": self.text,
            "heading_path": list(self.heading_path),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            tenant_id=d["tenant_id"],
            source=d["source"],
            title=d["title"],
            text=d["text"],
            heading_path=tuple(d.get("heading_path", ())),
            metadata=dict(d.get("metadata", {})),
        )
