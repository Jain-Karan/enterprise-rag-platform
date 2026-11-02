"""Source connectors. Adding a source to a tenant is a YAML entry; adding a new *kind* of source
is one class implementing `fetch() -> Iterator[Document]` plus a line in `build_connector`."""

from __future__ import annotations

from ..config import SourceConfig
from .base import Connector
from .confluence import ConfluenceConnector
from .local import LocalConnector
from .s3 import S3Connector
from .sharepoint import SharePointConnector

_REGISTRY = {
    "local": LocalConnector,
    "s3": S3Connector,
    "confluence": ConfluenceConnector,
    "sharepoint": SharePointConnector,
}


def build_connector(src: SourceConfig, **overrides) -> Connector:
    try:
        cls = _REGISTRY[src.type]
    except KeyError as e:
        raise ValueError(f"unknown source type {src.type!r}") from e
    return cls(name=src.name, **{**src.params, **overrides})


__all__ = [
    "Connector",
    "ConfluenceConnector",
    "LocalConnector",
    "S3Connector",
    "SharePointConnector",
    "build_connector",
]
