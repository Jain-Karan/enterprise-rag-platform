import pytest
import yaml

from ragplatform.config import load_config


def _write(tmp_path, cfg):
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_env_expansion_and_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("ACME_KEY", "secret-1")
    (tmp_path / "docs").mkdir()
    cfg = load_config(
        _write(
            tmp_path,
            {
                "tenants": [
                    {
                        "tenant_id": "acme",
                        "api_keys": ["${ACME_KEY}"],
                        "sources": [{"name": "f", "type": "local", "params": {"root": "docs"}}],
                    }
                ]
            },
        )
    )
    t = cfg.tenant("acme")
    assert t.api_keys == ["secret-1"]
    assert t.sources[0].params["root"] == str((tmp_path / "docs").resolve())
    assert cfg.data_dir == str((tmp_path / ".ragdata").resolve())


@pytest.mark.parametrize("bad", ["../etc", "Acme", "a", "has space"])
def test_rejects_unsafe_tenant_ids(tmp_path, bad):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, {"tenants": [{"tenant_id": bad}]}))


def test_rejects_duplicate_tenants(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_config(_write(tmp_path, {"tenants": [{"tenant_id": "ab"}, {"tenant_id": "ab"}]}))
