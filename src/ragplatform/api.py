"""HTTP API. The tenant is derived from the API key, never from the request body, so a caller
cannot ask for another tenant's data by changing a field."""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import load_config
from .platform import Platform


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    n: int
    doc_id: str
    title: str
    section: str
    source: str


class QueryResponse(BaseModel):
    tenant_id: str
    answer: str
    citations: list[Citation]


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    score: float
    text: str


def create_app(platform: Platform) -> FastAPI:
    app = FastAPI(title="EnterpriseRAGPlatform", version="0.1.0")

    def tenant(x_api_key: str | None = Header(default=None)) -> str:
        t = platform.tenant_for_key(x_api_key)
        if t is None:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
        return t

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "tenants": len(platform.config.tenants)}

    @app.post("/v1/ingest")
    def ingest(tenant_id: str = Depends(tenant)) -> dict:
        r = platform.ingest(tenant_id)
        return {
            "tenant_id": r.tenant_id,
            "documents": r.documents,
            "chunks": r.chunks,
            "per_source": r.per_source,
            "seconds": round(r.seconds, 3),
        }

    @app.post("/v1/search", response_model=list[SearchHit])
    def search(req: QueryRequest, tenant_id: str = Depends(tenant)) -> list[SearchHit]:
        return [
            SearchHit(
                chunk_id=h.chunk.chunk_id,
                doc_id=h.chunk.doc_id,
                title=h.chunk.title,
                section=" > ".join(h.chunk.heading_path),
                score=round(h.score, 4),
                text=h.chunk.text,
            )
            for h in platform.retrieve(tenant_id, req.question)
        ]

    @app.post("/v1/query", response_model=QueryResponse)
    def query(req: QueryRequest, tenant_id: str = Depends(tenant)) -> QueryResponse:
        a = platform.answer(tenant_id, req.question)
        return QueryResponse(
            tenant_id=tenant_id, answer=a.answer, citations=[Citation(**c) for c in a.citations]
        )

    return app


def app_from_env() -> FastAPI:
    """Entry point for `uvicorn ragplatform.api:app_from_env --factory`."""
    return create_app(Platform(load_config(os.environ.get("RAGP_CONFIG", "config/tenants.yaml"))))
