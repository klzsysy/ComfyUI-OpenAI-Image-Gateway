"""Tests for gateway configuration loading, aliases, and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.config import load_config, save_config
from gateway.models import GatewayError

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def test_load_example_config():
    cfg = load_config(PLUGIN_ROOT, explicit_path=PLUGIN_ROOT / "config.example.yaml", env={})
    assert cfg.default_model == "qwen-image-2.1-turbo-edit"
    assert "qwen-image-2.1-turbo-edit" in cfg.models
    assert cfg.resolve_model("gpt-image-1").id == "qwen-image-2.1-turbo-edit"
    assert cfg.resolve_model("unknown-model").id == "qwen-image-2.1-turbo-edit"
    # By default expose_aliases is False so /v1/models lists only primary configured names.
    assert cfg.listed_model_ids() == ["qwen-image-2.1-turbo-edit"]
    assert cfg.default_response_format == "b64_json"


def test_save_and_reload_config_roundtrip(tmp_path: Path):
    cfg = load_config(PLUGIN_ROOT, explicit_path=PLUGIN_ROOT / "config.example.yaml", env={})
    saved = save_config(cfg, tmp_path, target_path=tmp_path / "config.local.yaml")
    reloaded = load_config(tmp_path, explicit_path=saved, env={})
    assert reloaded.default_model == cfg.default_model
    assert list(reloaded.models) == list(cfg.models)
    assert reloaded.expose_aliases is False


def test_strict_model_resolution_without_fallback(tmp_path: Path):
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "SaveImage", "inputs": {}}}), encoding="utf-8")
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "default_model": "my-edit",
                "fallback_to_default_model": False,
                "expose_aliases": False,
                "models": {
                    "my-edit": {
                        "workflow": str(wf),
                        "aliases": ["gpt-image-1"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(tmp_path, explicit_path=cfg_file, env={})
    assert cfg.listed_model_ids() == ["my-edit"]
    assert cfg.resolve_model("gpt-image-1").id == "my-edit"
    with pytest.raises(GatewayError) as exc_info:
        cfg.resolve_model("nonexistent")
    assert exc_info.value.status == 404
    assert exc_info.value.code == "model_not_found"


def test_env_overrides(tmp_path: Path):
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "SaveImage", "inputs": {}}}), encoding="utf-8")
    cfg_file = tmp_path / "config.local.json"
    cfg_file.write_text(
        json.dumps({"models": {"m1": {"workflow": "wf.json"}}}),
        encoding="utf-8",
    )

    cfg = load_config(
        tmp_path,
        env={
            "COMFY_OPENAI_GATEWAY_API_KEYS": "sk-one, sk-two",
            "COMFY_OPENAI_GATEWAY_CORS_ORIGINS": "https://app.example.com",
            "COMFY_OPENAI_GATEWAY_RESPONSE_FORMAT": "url",
            "COMFY_OPENAI_GATEWAY_TIMEOUT": "120",
            "COMFY_OPENAI_GATEWAY_KEEP_INPUT_FILES": "true",
        },
    )
    assert cfg.api_keys == ("sk-one", "sk-two")
    assert cfg.cors_origins == "https://app.example.com"
    assert cfg.default_response_format == "url"
    assert cfg.timeout == 120.0
    assert cfg.keep_input_files is True


def test_invalid_binding_rejected(tmp_path: Path):
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "SaveImage", "inputs": {}}}), encoding="utf-8")
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "models": {
                    "bad": {
                        "workflow": "wf.json",
                        "bindings": {"prompt": "1.wrong.text"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid binding path"):
        load_config(tmp_path, explicit_path=cfg_file, env={})


def test_empty_config_when_no_file_exists(tmp_path: Path):
    cfg = load_config(tmp_path, env={})
    assert cfg.models == {}
    assert cfg.config_path is None
    with pytest.raises(GatewayError) as exc_info:
        cfg.resolve_model("gpt-image-1")
    assert exc_info.value.status == 503
