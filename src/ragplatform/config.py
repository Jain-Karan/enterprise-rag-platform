"""Tenant configuration. Onboarding a tenant or a new source is a YAML change, not a code change."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

_ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)\}")
_TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")


class SourceConfig(BaseModel):
    name: str
    type: Literal["local", "s3", "confluence", "sharepoint"]
    params: dict[str, Any] = Field(default_factory=dict)


class ChunkingConfig(BaseModel):
    strategy: Literal["layout", "fixed"] = "layout"
    max_tokens: int = 220
    overlap_tokens: int = 30


class RetrievalConfig(BaseModel):
    mode: Literal["hybrid", "bm25", "vector"] = "hybrid"
    top_k: int = 5
    candidate_k: int = 30
    rrf_k: int = 60
    rerank: bool = True
    # BM25 scores the breadcrumb (title > section) as a separate field with this weight.
    # Folding it into the body lets length normalisation favour short boilerplate chunks.
    heading_weight: float = 1.0
    # RRF weights per retriever. Lower vector_weight when the embedder is weak on the
    # tenant's vocabulary; raise it with a strong domain embedding model.
    bm25_weight: float = 1.0
    vector_weight: float = 0.3  # tuned for the offline HashingEmbedder; see README


class LLMConfig(BaseModel):
    provider: Literal["anthropic", "azure_foundry", "fake"] = "fake"
    model: str = "claude-sonnet-4-5"
    temperature: float = 0.0
    max_tokens: int = 700
    params: dict[str, Any] = Field(default_factory=dict)


class TenantConfig(BaseModel):
    tenant_id: str
    display_name: str = ""
    api_keys: list[str] = Field(default_factory=list)
    sources: list[SourceConfig] = Field(default_factory=list)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    benchmark: str | None = None  # path to a golden-set file run on onboarding

    @field_validator("tenant_id")
    @classmethod
    def _valid_tenant_id(cls, v: str) -> str:
        # tenant_id becomes a directory name, so keep it path-safe
        if not _TENANT_ID.match(v):
            raise ValueError(f"invalid tenant_id {v!r}: use lowercase letters, digits and dashes")
        return v


class PlatformConfig(BaseModel):
    data_dir: str = ".ragdata"
    tenants: list[TenantConfig]

    def tenant(self, tenant_id: str) -> TenantConfig:
        for t in self.tenants:
            if t.tenant_id == tenant_id:
                return t
        raise KeyError(tenant_id)


def _expand_env(value: Any) -> Any:
    """Resolve ${VAR} references so secrets never live in the YAML file itself."""
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_config(path: str | Path) -> PlatformConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    cfg = PlatformConfig.model_validate(_expand_env(raw))
    # relative paths in the config are relative to the config file, not the cwd
    base = path.parent
    if not Path(cfg.data_dir).is_absolute():
        cfg.data_dir = str((base / cfg.data_dir).resolve())
    for t in cfg.tenants:
        if t.benchmark and not Path(t.benchmark).is_absolute():
            t.benchmark = str((base / t.benchmark).resolve())
        for s in t.sources:
            root = s.params.get("root")
            if s.type == "local" and root and not Path(root).is_absolute():
                s.params["root"] = str((base / root).resolve())
    ids = [t.tenant_id for t in cfg.tenants]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate tenant ids: {sorted(dupes)}")
    return cfg
