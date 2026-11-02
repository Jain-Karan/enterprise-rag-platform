from __future__ import annotations

from collections.abc import Iterator

import httpx

from ..documents import Document
from .base import content_type_for, title_from

GRAPH = "https://graph.microsoft.com/v1.0"


class SharePointConnector:
    """Files from a SharePoint document library through Microsoft Graph, using an Entra ID app
    registration (client-credentials flow, `Sites.Read.All`)."""

    def __init__(
        self,
        name: str,
        site_id: str,
        tenant: str,
        client_id: str,
        client_secret: str,
        folder: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name = name
        self.site_id = site_id
        self.folder = folder.strip("/")
        self._token_req = (tenant, client_id, client_secret)
        self.client = httpx.Client(timeout=30, transport=transport)
        self._token: str | None = None

    def _auth(self) -> dict[str, str]:
        if self._token is None:
            tenant, cid, secret = self._token_req
            r = self.client.post(
                f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": cid,
                    "client_secret": secret,
                    "scope": "https://graph.microsoft.com/.default",
                },
            )
            r.raise_for_status()
            self._token = r.json()["access_token"]
        return {"Authorization": f"Bearer {self._token}"}

    def _children_url(self, item_path: str) -> str:
        if item_path:
            return f"{GRAPH}/sites/{self.site_id}/drive/root:/{item_path}:/children"
        return f"{GRAPH}/sites/{self.site_id}/drive/root/children"

    def _walk(self, item_path: str) -> Iterator[tuple[str, dict]]:
        url: str | None = self._children_url(item_path)
        while url:
            r = self.client.get(url, headers=self._auth())
            r.raise_for_status()
            data = r.json()
            for item in data.get("value", []):
                path = f"{item_path}/{item['name']}" if item_path else item["name"]
                if "folder" in item:
                    yield from self._walk(path)
                else:
                    yield path, item
            url = data.get("@odata.nextLink")

    def fetch(self) -> Iterator[Document]:
        for path, item in self._walk(self.folder):
            ct = content_type_for(item["name"])
            if not ct:
                continue
            r = self.client.get(
                f"{GRAPH}/sites/{self.site_id}/drive/items/{item['id']}/content",
                headers=self._auth(),
                follow_redirects=True,
            )
            r.raise_for_status()
            text = r.text
            yield Document(
                doc_id=f"{self.name}:{item['id']}",
                source=self.name,
                title=title_from(item["name"], text, ct),
                content=text,
                content_type=ct,
                metadata={"path": path, "web_url": item.get("webUrl", "")},
            )
