from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from ragplatform.config import load_config
from ragplatform.platform import Platform

ACME_DOC = textwrap.dedent(
    """\
    # Travel Policy: Japan

    ## Purpose

    This policy covers business travel to Japan.

    ## Daily Limits

    | Category | Daily limit |
    |---|---|
    | Lodging | 41,000 yen |
    | Meals | 9,300 yen |

    ## Approvals

    Trips longer than 12 nights need VP approval.
    """
)

GLOBEX_DOC = textwrap.dedent(
    """\
    # Travel Policy: Japan

    ## Daily Limits

    | Category | Daily limit |
    |---|---|
    | Lodging | 77,700 yen |
    """
)


@pytest.fixture
def two_tenant_config(tmp_path: Path):
    """Two tenants with near-identical documents but different numbers, to catch leaks."""
    for tenant, doc in (("acme", ACME_DOC), ("globex", GLOBEX_DOC)):
        d = tmp_path / "docs" / tenant
        d.mkdir(parents=True)
        (d / "travel-japan.md").write_text(doc)
    cfg = {
        "data_dir": "data",
        "tenants": [
            {
                "tenant_id": tid,
                "api_keys": [f"key-{tid}"],
                "sources": [{"name": "files", "type": "local", "params": {"root": f"docs/{tid}"}}],
            }
            for tid in ("acme", "globex")
        ],
    }
    path = tmp_path / "tenants.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return load_config(path)


@pytest.fixture
def platform(two_tenant_config) -> Platform:
    p = Platform(two_tenant_config)
    p.ingest("acme")
    p.ingest("globex")
    return p
