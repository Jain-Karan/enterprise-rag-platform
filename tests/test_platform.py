import pytest
from fastapi.testclient import TestClient

from ragplatform.api import create_app
from ragplatform.documents import Chunk
from ragplatform.generation import cited_indices
from ragplatform.platform import Platform
from ragplatform.store import TenantIndex, TenantIsolationError


def test_retrieval_is_tenant_isolated(platform):
    q = "lodging daily limit Japan"
    acme = " ".join(h.chunk.text for h in platform.retrieve("acme", q))
    globex = " ".join(h.chunk.text for h in platform.retrieve("globex", q))
    assert "41,000" in acme and "77,700" not in acme
    assert "77,700" in globex and "41,000" not in globex


def test_index_rejects_foreign_chunks(platform):
    foreign = Chunk("x#0", "x", "globex", "s", "t", "text")
    with pytest.raises(TenantIsolationError):
        platform.index("acme").add([foreign])


def test_persisted_index_reloads(two_tenant_config, platform):
    fresh = Platform(two_tenant_config)
    hits = fresh.retrieve("acme", "meals daily limit")
    assert any("9,300" in h.chunk.text for h in hits)
    assert isinstance(fresh.index("acme"), TenantIndex)


def test_answer_cites_sources(platform):
    a = platform.answer("acme", "How many nights before VP approval is needed?")
    assert "12 nights" in a.answer
    assert a.citations and a.citations[0]["title"] == "Travel Policy: Japan"


def test_cited_indices_drops_out_of_range():
    assert cited_indices("a [2] b [9] c [2] [1]", 3) == [2, 1]


def test_api_tenant_comes_from_key(platform):
    client = TestClient(create_app(platform))
    assert client.post("/v1/search", json={"question": "lodging"}).status_code == 401
    r = client.post(
        "/v1/search", json={"question": "lodging limit"}, headers={"X-API-Key": "key-globex"}
    )
    assert r.status_code == 200
    assert "77,700" in r.json()[0]["text"]
    r = client.post(
        "/v1/query",
        json={"question": "What is the lodging daily limit?"},
        headers={"X-API-Key": "key-acme"},
    )
    body = r.json()
    assert body["tenant_id"] == "acme" and "41,000" in body["answer"]
    assert body["citations"][0]["section"]


def test_api_ingest(platform):
    client = TestClient(create_app(platform))
    r = client.post("/v1/ingest", headers={"X-API-Key": "key-acme"})
    assert r.status_code == 200 and r.json()["documents"] == 1


def test_empty_api_key_is_never_valid(two_tenant_config):
    two_tenant_config.tenants[0].api_keys = [""]
    p = Platform(two_tenant_config)
    assert p.tenant_for_key("") is None
    assert p.tenant_for_key(None) is None
