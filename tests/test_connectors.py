import json

import boto3
import httpx
import pytest
from moto import mock_aws

from ragplatform.config import SourceConfig
from ragplatform.connectors import (
    ConfluenceConnector,
    LocalConnector,
    S3Connector,
    SharePointConnector,
    build_connector,
)


def test_local_connector(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "policy.md").write_text("# Expense Policy\n\nbody")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    docs = list(LocalConnector("files", str(tmp_path)).fetch())
    assert [d.doc_id for d in docs] == ["files:a/policy.md"]
    assert docs[0].title == "Expense Policy"


def test_local_connector_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalConnector("x", str(tmp_path / "nope"))


@mock_aws
def test_s3_connector_paginates_and_filters():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="kb-docs")
    for i in range(3):
        s3.put_object(
            Bucket="kb-docs", Key=f"policies/p{i}.md", Body=f"# Policy {i}\n\ntext".encode()
        )
    s3.put_object(Bucket="kb-docs", Key="policies/scan.pdf", Body=b"%PDF")
    s3.put_object(Bucket="kb-docs", Key="other/x.md", Body=b"# Other")
    docs = list(S3Connector("s3", "kb-docs", "policies/", client=s3).fetch())
    assert sorted(d.title for d in docs) == ["Policy 0", "Policy 1", "Policy 2"]
    assert all(d.metadata["etag"] for d in docs)


def test_confluence_connector_follows_pagination():
    pages = [
        {
            "id": str(i),
            "title": f"Page {i}",
            "version": {"number": 3},
            "body": {"storage": {"value": f"<h2>Section</h2><p>body {i}</p>"}},
        }
        for i in range(5)
    ]
    seen_starts = []

    def handler(req: httpx.Request) -> httpx.Response:
        start = int(req.url.params["start"])
        limit = int(req.url.params["limit"])
        seen_starts.append(start)
        assert req.url.params["spaceKey"] == "OPS"
        assert req.headers["authorization"].startswith("Basic ")
        batch = pages[start : start + limit]
        links = {"next": "/next"} if start + limit < len(pages) else {}
        return httpx.Response(200, json={"results": batch, "_links": links})

    c = ConfluenceConnector(
        "wiki",
        "https://x.atlassian.net",
        "OPS",
        "me@x.com",
        "tok",
        page_size=2,
        transport=httpx.MockTransport(handler),
    )
    docs = list(c.fetch())
    assert [d.title for d in docs] == [f"Page {i}" for i in range(5)]
    assert seen_starts == [0, 2, 4]
    assert docs[0].content_type == "text/html"


def test_sharepoint_connector_recurses_folders_and_pages():
    calls = {"token": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if "login.microsoftonline.com" in url:
            calls["token"] += 1
            return httpx.Response(200, json={"access_token": "abc"})
        assert req.headers["authorization"] == "Bearer abc"
        if url.endswith("/drive/root/children"):
            return httpx.Response(
                200,
                json={
                    "value": [{"id": "f1", "name": "Policies", "folder": {}}],
                    "@odata.nextLink": "https://graph.microsoft.com/v1.0/page2",
                },
            )
        if url.endswith("/page2"):
            return httpx.Response(200, json={"value": [{"id": "i3", "name": "readme.txt"}]})
        if "root:/Policies:/children" in url:
            return httpx.Response(
                200,
                json={
                    "value": [
                        {"id": "i1", "name": "travel.md", "webUrl": "https://sp/travel.md"},
                        {"id": "i2", "name": "deck.pptx"},
                    ]
                },
            )
        if url.endswith("/items/i1/content"):
            return httpx.Response(200, text="# Travel\n\nbody")
        if url.endswith("/items/i3/content"):
            return httpx.Response(200, text="plain")
        return httpx.Response(404, text=url)

    c = SharePointConnector(
        "sp", "site-1", "tenant-1", "cid", "secret", transport=httpx.MockTransport(handler)
    )
    docs = list(c.fetch())
    assert [d.metadata["path"] for d in docs] == ["Policies/travel.md", "readme.txt"]
    assert docs[0].title == "Travel"
    assert calls["token"] == 1


def test_build_connector_unknown_type():
    src = SourceConfig.model_construct(name="x", type="ftp", params={})
    with pytest.raises(ValueError):
        build_connector(src)


def test_build_connector_passes_params(tmp_path):
    src = SourceConfig(name="f", type="local", params={"root": str(tmp_path)})
    assert build_connector(src).root == tmp_path
    json.dumps(src.model_dump())  # config stays serialisable
