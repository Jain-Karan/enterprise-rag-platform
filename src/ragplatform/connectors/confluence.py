from __future__ import annotations

from collections.abc import Iterator

import httpx

from ..documents import Document


class ConfluenceConnector:
    """Pages from one Confluence Cloud space via the REST API, in storage (XHTML) format."""

    def __init__(
        self,
        name: str,
        base_url: str,
        space_key: str,
        email: str,
        api_token: str,
        page_size: int = 50,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name = name
        self.space_key = space_key
        self.page_size = page_size
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            auth=(email, api_token),
            timeout=30,
            transport=transport,
        )

    def fetch(self) -> Iterator[Document]:
        start = 0
        while True:
            r = self.client.get(
                "/wiki/rest/api/content",
                params={
                    "spaceKey": self.space_key,
                    "type": "page",
                    "status": "current",
                    "expand": "body.storage,version",
                    "limit": self.page_size,
                    "start": start,
                },
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            for page in results:
                yield Document(
                    doc_id=f"{self.name}:{page['id']}",
                    source=self.name,
                    title=page.get("title", ""),
                    content=page["body"]["storage"]["value"],
                    content_type="text/html",
                    metadata={
                        "page_id": str(page["id"]),
                        "version": str(page.get("version", {}).get("number", "")),
                    },
                )
            if not results or "next" not in data.get("_links", {}):
                return
            start += len(results)
