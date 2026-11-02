from __future__ import annotations

import posixpath
from collections.abc import Iterator
from typing import Protocol

from ..documents import Document

SUPPORTED_EXT = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
}


def content_type_for(name: str) -> str | None:
    return SUPPORTED_EXT.get(posixpath.splitext(name.lower())[1])


def title_from(name: str, content: str, content_type: str) -> str:
    """First heading if there is one, otherwise the file stem."""
    if content_type == "text/markdown":
        for line in content.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    elif content_type == "text/html":
        lower = content.lower()
        for tag in ("<title>", "<h1>"):
            i = lower.find(tag)
            if i >= 0:
                j = lower.find("</", i)
                if j > i:
                    return content[i + len(tag) : j].strip()
    stem = posixpath.splitext(posixpath.basename(name))[0]
    return stem.replace("-", " ").replace("_", " ").strip().title()


class Connector(Protocol):
    name: str

    def fetch(self) -> Iterator[Document]: ...
