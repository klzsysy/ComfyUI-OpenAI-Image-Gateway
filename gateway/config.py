"""Configuration loading, validation, and atomic persistence."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

from .models import (
    GatewayConfig,
    GatewayError,
    ImageInputSpec,
    ImageTargetSpec,
    LoadImageNodeSlot,
    MaskInputSpec,
    ModelConfig,
)

log = logging.getLogger("comfyui_openai_image_gateway")

CONFIG_ENV_VAR = "COMFY_OPENAI_GATEWAY_CONFIG"
DEFAULT_CONFIG_FILENAMES = (
    "config.local.yaml",
    "config.local.yml",
    "config.yaml",
    "config.yml",
    "config.local.json",
    "config.json",
)
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
_SLUG_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def validate_model_id(model_id: str) -> str:
    cleaned = (model_id or "").strip()
    if not cleaned or not _MODEL_ID_RE.match(cleaned) or ".." in cleaned:
        raise GatewayError(
            "Invalid model name. Use 1-128 letters, digits, or `._:@-` (starting with a letter or digit).",
            status=400,
            param="id",
            code="invalid_model_id",
        )
    return cleaned


def safe_workflow_filename(model_id: str) -> str:
    slug = _SLUG_UNSAFE_RE.sub("_", model_id).strip("._") or "workflow"
    return f"{slug}.json"


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def _parse_api_keys(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = [str(part).strip() for part in value]
    else:
        raise ValueError("api_keys must be a list of strings or a comma-separated string")
    return tuple(item for item in items if item)


def _read_config_file(path: Path) -> dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        parsed = json.loads(raw_text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                f"PyYAML is required to load {path.name}; install pyyaml or use a .json config file."
            ) from exc
        parsed = yaml.safe_load(raw_text)

    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Config root in {path} must be a mapping/object")
    return parsed


def discover_config_path(plugin_root: Path, explicit_path: str | Path | None = None) -> Path | None:
    """Find the active configuration file."""
    candidate = explicit_path or os.environ.get(CONFIG_ENV_VAR)
    if candidate:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (plugin_root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configured gateway config file not found: {path}")
        return path

    for name in DEFAULT_CONFIG_FILENAMES:
        path = plugin_root / name
        if path.is_file():
            return path
    return None


def parse_bindings(raw_bindings: Any, model_id: str) -> dict[str, tuple[str, ...]]:
    if not raw_bindings:
        return {}
    if not isinstance(raw_bindings, dict):
        raise ValueError(f"models.{model_id}.bindings must be a mapping")
    parsed: dict[str, tuple[str, ...]] = {}
    for key, value in raw_bindings.items():
        if isinstance(value, str):
            targets = (value.strip(),) if value.strip() else ()
        elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            targets = tuple(v.strip() for v in value if v.strip())
        else:
            raise ValueError(
                f"models.{model_id}.bindings.{key} must be a string or list of strings"
            )
        if not targets:
            continue
        for target in targets:
            parts = target.split(".", 2)
            if len(parts) != 3 or parts[1] != "inputs" or not parts[0] or not parts[2]:
                raise ValueError(
                    f"Invalid binding path `{target}` in models.{model_id}.bindings.{key}; "
                    "expected `<node_id>.inputs.<field_name>`"
                )
        parsed[str(key)] = targets
    return parsed


def parse_image_inputs(raw: Any, model_id: str) -> ImageInputSpec | None:
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"models.{model_id}.image_inputs must be a mapping")

    mode = str(raw.get("mode", "dynamic_load_image")).strip()
    if mode not in ("dynamic_load_image", "load_image_nodes"):
        raise ValueError(
            f"models.{model_id}.image_inputs.mode must be `dynamic_load_image` or `load_image_nodes`"
        )

    max_images = int(raw.get("max_images", 10))
    if max_images < 0:
        raise ValueError(f"models.{model_id}.image_inputs.max_images must be >= 0")

    targets: list[ImageTargetSpec] = []
    for item in raw.get("targets") or ():
        if not isinstance(item, dict) or not item.get("node_id"):
            raise ValueError(f"Invalid target entry in models.{model_id}.image_inputs.targets")
        targets.append(
            ImageTargetSpec(
                node_id=str(item["node_id"]),
                input_prefix=str(item.get("input_prefix", "images.image_")),
                start_index=int(item.get("start_index", 1)),
                clear_existing_prefix=_parse_bool(item.get("clear_existing_prefix"), True),
                prune_if_empty=_parse_bool(item.get("prune_if_empty"), False),
            )
        )

    nodes: list[LoadImageNodeSlot] = []
    for item in raw.get("nodes") or ():
        if isinstance(item, (str, int)):
            nodes.append(LoadImageNodeSlot(node_id=str(item)))
        elif isinstance(item, dict) and item.get("node_id"):
            nodes.append(
                LoadImageNodeSlot(
                    node_id=str(item["node_id"]),
                    input_name=str(item.get("input", item.get("input_name", "image"))),
                )
            )
        else:
            raise ValueError(f"Invalid node entry in models.{model_id}.image_inputs.nodes")

    if mode == "dynamic_load_image" and not targets:
        raise ValueError(f"models.{model_id}.image_inputs requires non-empty `targets`")
    if mode == "load_image_nodes" and not nodes:
        raise ValueError(f"models.{model_id}.image_inputs requires non-empty `nodes`")

    return ImageInputSpec(
        mode=mode,
        max_images=max_images if mode == "dynamic_load_image" else len(nodes),
        targets=tuple(targets),
        nodes=tuple(nodes),
        prune_unused=_parse_bool(raw.get("prune_unused"), True),
    )


def parse_mask_input(raw: Any, model_id: str) -> MaskInputSpec | None:
    if not raw:
        return None
    if isinstance(raw, (str, int)):
        return MaskInputSpec(node_id=str(raw))
    if not isinstance(raw, dict) or not raw.get("node_id"):
        raise ValueError(f"models.{model_id}.mask_input must specify `node_id`")
    return MaskInputSpec(
        node_id=str(raw["node_id"]),
        input_name=str(raw.get("input", raw.get("input_name", "image"))),
        remove_if_unused=_parse_bool(raw.get("remove_if_unused"), False),
    )


def _resolve_workflow_path(raw_path: str, base_dir: Path, plugin_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.is_file():
        return candidate
    fallback = (plugin_root / path).resolve()
    if fallback.is_file():
        return fallback
    return candidate


def _relative_workflow_str(workflow_path: Path, base_dir: Path) -> str:
    try:
        return workflow_path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(workflow_path)


def serialize_model_config(model: ModelConfig, *, base_dir: Path) -> dict[str, Any]:
    """Serialize a ModelConfig into a clean YAML/JSON-friendly dict."""
    out: dict[str, Any] = {
        "workflow": _relative_workflow_str(model.workflow_path, base_dir),
    }
    if model.source_workflow:
        out["source_workflow"] = model.source_workflow
    if model.description:
        out["description"] = model.description
    if model.aliases:
        out["aliases"] = list(model.aliases)
    if model.output_node is not None:
        out["output_node"] = model.output_node
    out["prompt_required"] = model.prompt_required
    out["allow_no_image"] = model.allow_no_image
    if model.default_width is not None:
        out["default_width"] = model.default_width
    if model.default_height is not None:
        out["default_height"] = model.default_height
    out["size_multiple"] = model.size_multiple
    if model.max_n is not None:
        out["max_n"] = model.max_n
    if model.timeout is not None:
        out["timeout"] = model.timeout
    if model.defaults:
        out["defaults"] = dict(model.defaults)
    if model.remove_nodes:
        out["remove_nodes"] = list(model.remove_nodes)
    if model.image_inputs is not None:
        img = model.image_inputs
        if img.mode == "dynamic_load_image":
            out["image_inputs"] = {
                "mode": "dynamic_load_image",
                "max_images": img.max_images,
                "targets": [
                    {
                        "node_id": t.node_id,
                        "input_prefix": t.input_prefix,
                        "start_index": t.start_index,
                        "clear_existing_prefix": t.clear_existing_prefix,
                        "prune_if_empty": t.prune_if_empty,
                    }
                    for t in img.targets
                ],
            }
        else:
            out["image_inputs"] = {
                "mode": "load_image_nodes",
                "prune_unused": img.prune_unused,
                "nodes": [{"node_id": n.node_id, "input": n.input_name} for n in img.nodes],
            }
    if model.mask_input is not None:
        out["mask_input"] = {
            "node_id": model.mask_input.node_id,
            "input": model.mask_input.input_name,
            "remove_if_unused": model.mask_input.remove_if_unused,
        }
    if model.bindings:
        out["bindings"] = {
            k: (v[0] if len(v) == 1 else list(v))
            for k, v in model.bindings.items()
        }
    return out


def serialize_config(config: GatewayConfig, *, base_dir: Path) -> dict[str, Any]:
    """Serialize GatewayConfig to a clean dict for persistence."""
    data: dict[str, Any] = {
        "default_model": config.default_model,
        "fallback_to_default_model": config.fallback_to_default_model,
        "expose_aliases": config.expose_aliases,
        "api_keys": list(config.api_keys),
        "cors_origins": config.cors_origins,
        "default_response_format": config.default_response_format,
        "timeout": config.timeout,
        "max_n": config.max_n,
        "max_image_mb": round(config.max_image_bytes / (1024 * 1024), 2),
        "allow_remote_image_urls": config.allow_remote_image_urls,
        "remote_url_timeout": config.remote_url_timeout,
        "keep_input_files": config.keep_input_files,
        "input_subfolder": config.input_subfolder,
        "models": {
            model_id: serialize_model_config(model, base_dir=base_dir)
            for model_id, model in config.models.items()
        },
    }
    if config.public_base_url:
        data["public_base_url"] = config.public_base_url
    return data


def save_config(config: GatewayConfig, plugin_root: Path, *, target_path: Path | None = None) -> Path:
    """Persist GatewayConfig atomically to YAML (default config.local.yaml) or JSON."""
    path = target_path or config.config_path or (plugin_root / "config.local.yaml")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_config(config, base_dir=path.parent)

    if path.suffix.lower() == ".json":
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    else:
        import yaml

        text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False)

    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def save_managed_workflow(plugin_root: Path, model_id: str, prompt_graph: dict[str, Any]) -> Path:
    """Atomically save a converted API-format workflow snapshot under workflows/local/."""
    local_dir = (plugin_root / "workflows" / "local").resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    target = (local_dir / safe_workflow_filename(model_id)).resolve()
    if os.path.commonpath((str(local_dir), str(target))) != str(local_dir):
        raise GatewayError("Invalid managed workflow path.", status=400, param="id")
    tmp = target.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(prompt_graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_config(
    plugin_root: Path,
    *,
    explicit_path: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> GatewayConfig:
    """Load gateway configuration from YAML/JSON and apply environment overrides."""
    environ = os.environ if env is None else env
    config_path = discover_config_path(plugin_root, explicit_path or environ.get(CONFIG_ENV_VAR))
    raw: dict[str, Any] = {}
    base_dir = plugin_root
    if config_path is not None:
        raw = _read_config_file(config_path)
        base_dir = config_path.parent

    models: dict[str, ModelConfig] = {}
    alias_map: dict[str, str] = {}
    raw_models = raw.get("models") or {}
    if not isinstance(raw_models, dict):
        raise ValueError("`models` must be a mapping of model_id -> model config")

    for raw_id, entry in raw_models.items():
        model_id = str(raw_id).strip()
        if not model_id:
            raise ValueError("Model id must be a non-empty string")
        if not isinstance(entry, dict):
            raise ValueError(f"models.{model_id} must be a mapping")
        wf_raw = entry.get("workflow")
        if not wf_raw or not isinstance(wf_raw, str):
            raise ValueError(f"models.{model_id}.workflow is required")
        wf_path = _resolve_workflow_path(wf_raw, base_dir, plugin_root)
        if not wf_path.is_file():
            raise FileNotFoundError(
                f"Workflow file for model `{model_id}` not found: {wf_path}"
            )

        aliases = tuple(str(a).strip() for a in (entry.get("aliases") or ()) if str(a).strip())
        for alias in aliases:
            if alias in alias_map and alias_map[alias] != model_id:
                raise ValueError(
                    f"Duplicate alias `{alias}` mapped to both `{alias_map[alias]}` and `{model_id}`"
                )
            alias_map[alias] = model_id

        remove_nodes = tuple(str(n).strip() for n in (entry.get("remove_nodes") or ()) if str(n).strip())
        defaults = dict(entry.get("defaults") or {})

        models[model_id] = ModelConfig(
            id=model_id,
            workflow_path=wf_path,
            source_workflow=str(entry.get("source_workflow") or "").strip(),
            aliases=aliases,
            description=str(entry.get("description", "")),
            output_node=str(entry["output_node"]) if entry.get("output_node") is not None else None,
            prompt_required=_parse_bool(entry.get("prompt_required"), True),
            allow_no_image=_parse_bool(entry.get("allow_no_image"), True),
            default_width=int(entry["default_width"]) if entry.get("default_width") is not None else None,
            default_height=int(entry["default_height"]) if entry.get("default_height") is not None else None,
            size_multiple=max(1, int(entry.get("size_multiple", 16))),
            max_n=int(entry["max_n"]) if entry.get("max_n") is not None else None,
            timeout=float(entry["timeout"]) if entry.get("timeout") is not None else None,
            defaults=defaults,
            bindings=parse_bindings(entry.get("bindings"), model_id),
            image_inputs=parse_image_inputs(entry.get("image_inputs"), model_id),
            mask_input=parse_mask_input(entry.get("mask_input"), model_id),
            remove_nodes=remove_nodes,
        )

    default_model = raw.get("default_model")
    if default_model is not None:
        default_model = str(default_model).strip() or None
    if default_model and default_model not in models:
        canonical = alias_map.get(default_model)
        if canonical:
            default_model = canonical
        else:
            raise ValueError(f"default_model `{default_model}` is not defined in `models`")
    if default_model is None and models:
        default_model = next(iter(models))

    api_keys_raw = environ.get("COMFY_OPENAI_GATEWAY_API_KEYS")
    api_keys = _parse_api_keys(api_keys_raw) if api_keys_raw is not None else _parse_api_keys(raw.get("api_keys"))

    cors_origins = environ.get("COMFY_OPENAI_GATEWAY_CORS_ORIGINS")
    if cors_origins is None:
        cors_origins = str(raw.get("cors_origins", "*"))

    default_response_format = str(
        environ.get("COMFY_OPENAI_GATEWAY_RESPONSE_FORMAT")
        or raw.get("default_response_format", "b64_json")
    ).strip().lower()
    if default_response_format not in ("b64_json", "url"):
        raise ValueError("default_response_format must be `b64_json` or `url`")

    public_base_url = environ.get("COMFY_OPENAI_GATEWAY_PUBLIC_BASE_URL")
    if public_base_url is None:
        public_base_url = raw.get("public_base_url")
    if public_base_url is not None:
        public_base_url = str(public_base_url).strip().rstrip("/") or None

    timeout = float(environ.get("COMFY_OPENAI_GATEWAY_TIMEOUT") or raw.get("timeout", 600.0))
    max_n = max(1, int(environ.get("COMFY_OPENAI_GATEWAY_MAX_N") or raw.get("max_n", 4)))
    max_image_mb = float(environ.get("COMFY_OPENAI_GATEWAY_MAX_IMAGE_MB") or raw.get("max_image_mb", 25))
    max_image_bytes = max(1024, int(max_image_mb * 1024 * 1024))
    allow_remote_urls = _parse_bool(
        environ.get("COMFY_OPENAI_GATEWAY_ALLOW_REMOTE_URLS"),
        _parse_bool(raw.get("allow_remote_image_urls"), True),
    )
    remote_url_timeout = float(raw.get("remote_url_timeout", 30.0))
    keep_input_files = _parse_bool(
        environ.get("COMFY_OPENAI_GATEWAY_KEEP_INPUT_FILES"),
        _parse_bool(raw.get("keep_input_files"), False),
    )
    input_subfolder = str(raw.get("input_subfolder", "openai_gateway")).strip().strip("/\\") or "openai_gateway"

    return GatewayConfig(
        models=models,
        alias_map=alias_map,
        default_model=default_model,
        fallback_to_default_model=_parse_bool(raw.get("fallback_to_default_model"), True),
        expose_aliases=_parse_bool(raw.get("expose_aliases"), False),
        api_keys=api_keys,
        cors_origins=cors_origins,
        default_response_format=default_response_format,
        public_base_url=public_base_url,
        timeout=timeout,
        max_n=max_n,
        max_image_bytes=max_image_bytes,
        allow_remote_image_urls=allow_remote_urls,
        remote_url_timeout=remote_url_timeout,
        keep_input_files=keep_input_files,
        input_subfolder=input_subfolder,
        config_path=config_path,
    )
