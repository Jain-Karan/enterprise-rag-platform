from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..documents import Document
from .base import content_type_for, title_from


class LocalConnector:
    """Reads Markdown / HTML / text files from a directory tree. Used for fixtures, benchmarks,
    and file-share exports."""

    def __init__(self, name: str, root: str) -> None:
        self.name = name
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"source {name!r}: {self.root} is not a directory")

    def fetch(self) -> Iterator[Document]:
        for path in sorted(self.root.rglob("*")):
            ct = content_type_for(path.name)
            if not ct or not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            yield Document(
                doc_id=f"{self.name}:{rel}",
                source=self.name,
                title=title_from(rel, text, ct),
                content=text,
                content_type=ct,
                metadata={"path": rel},
            )
