"""Orchestrates ingest and query for every configured tenant."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import make_chunker
from .config import PlatformConfig, TenantConfig
from .connectors import build_connector
from .embeddings import Embedder, HashingEmbedder
from .generation import build_messages, cited_indices, make_chat_model
from .rerank import CoverageReranker, Reranker
from .store import Hit, TenantIndex


@dataclass
class IngestReport:
    tenant_id: str
    documents: int = 0
    chunks: int = 0
    per_source: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class Answer:
    answer: str
    citations: list[dict]
    hits: list[Hit]


class Platform:
    def __init__(
        self,
        config: PlatformConfig,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        connector_overrides: dict[str, dict] | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder or HashingEmbedder()
        self.reranker = reranker or CoverageReranker()
        self.connector_overrides = connector_overrides or {}
        self._indexes: dict[str, TenantIndex] = {}
        self._key_to_tenant: dict[str, str] = {}
        for t in config.tenants:
            for k in t.api_keys:
                if not k:
                    continue  # an unset ${ENV} reference must not become a valid empty key
                if k in self._key_to_tenant:
                    raise ValueError("API key reused across tenants")
                self._key_to_tenant[k] = t.tenant_id

    # -- tenancy ----------------------------------------------------------------------
    def tenant_for_key(self, api_key: str | None) -> str | None:
        return self._key_to_tenant.get(api_key or "")

    def _dir(self, tenant_id: str) -> Path:
        return Path(self.config.data_dir) / tenant_id

    def index(self, tenant_id: str) -> TenantIndex:
        if tenant_id not in self._indexes:
            d = self._dir(tenant_id)
            if (d / "manifest.json").exists():
                self._indexes[tenant_id] = TenantIndex.load(d, self.embedder)
            else:
                self._indexes[tenant_id] = TenantIndex(tenant_id, self.embedder)
        return self._indexes[tenant_id]

    # -- ingest -----------------------------------------------------------------------
    def ingest(self, tenant_id: str, persist: bool = True) -> IngestReport:
        t0 = time.perf_counter()
        tenant: TenantConfig = self.config.tenant(tenant_id)
        chunker = make_chunker(
            tenant.chunking.strategy, tenant.chunking.max_tokens, tenant.chunking.overlap_tokens
        )
        report = IngestReport(tenant_id)
        idx = TenantIndex(tenant_id, self.embedder)  # full rebuild, swapped in at the end
        for src in tenant.sources:
            conn = build_connector(src, **self.connector_overrides.get(src.name, {}))
            n = 0
            for doc in conn.fetch():
                chunks = chunker.chunk(doc, tenant_id)
                idx.add(chunks)
                report.documents += 1
                report.chunks += len(chunks)
                n += 1
            report.per_source[src.name] = n
        if persist:
            idx.save(self._dir(tenant_id))
        self._indexes[tenant_id] = idx
        report.seconds = time.perf_counter() - t0
        return report

    # -- query ------------------------------------------------------------------------
    def retrieve(self, tenant_id: str, question: str) -> list[Hit]:
        tenant = self.config.tenant(tenant_id)
        return self.index(tenant_id).search(question, tenant.retrieval, self.reranker)

    def answer(self, tenant_id: str, question: str) -> Answer:
        tenant = self.config.tenant(tenant_id)
        hits = self.retrieve(tenant_id, question)
        if not hits:
            return Answer("I don't know. No documents are indexed for this tenant yet.", [], [])
        model = make_chat_model(tenant.llm)
        reply = model.invoke(build_messages(question, hits))
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        citations = [
            {
                "n": i,
                "doc_id": hits[i - 1].chunk.doc_id,
                "title": hits[i - 1].chunk.title,
                "section": " > ".join(hits[i - 1].chunk.heading_path),
                "source": hits[i - 1].chunk.source,
            }
            for i in cited_indices(text, len(hits))
        ]
        return Answer(text, citations, hits)
