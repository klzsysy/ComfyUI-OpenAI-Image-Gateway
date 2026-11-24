"""HTTP route handlers and orchestration for the OpenAI-compatible Image Gateway."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode, urlparse

from aiohttp import web

from .analyzer import analyze_workflow
from .auth import apply_cors_headers, preflight_response, verify_api_key
from .backend import ComfyBackend
from .config import (
    parse_bindings,
    parse_image_inputs,
    parse_mask_input,
    save_config,
    save_managed_workflow,
    serialize_model_config,
    validate_model_id,
)
from .images import convert_output_image, decode_image_bytes, decode_image_source, parse_size
from .models import DecodedImage, GatewayConfig, GatewayError, ModelConfig
from .workflow import build_prompt_graph, load_workflow_template, validate_workflow_dict

log = logging.getLogger("comfyui_openai_image_gateway")

_MULTIPART_IMAGE_FIELD_RE = re.compile(r"^(?:image|images)(?:\[\])?$|^image_?\d+$")


def _parse_int(value: Any, *, name: str, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise GatewayError(f"`{name}` must be an integer.", status=400, param=name)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise GatewayError(f"`{name}` must be an integer.", status=400, param=name) from exc


def _parse_float(value: Any, *, name: str, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise GatewayError(f"`{name}` must be a number.", status=400, param=name)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise GatewayError(f"`{name}` must be a number.", status=400, param=name) from exc


def _parse_bool_strict(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def _collect_json_image_sources(body: dict[str, Any]) -> list[Any]:
    """Collect image inputs from common OpenAI/relay JSON keys."""
    sources: list[Any] = []
    for key in ("image", "images", "reference_images"):
        if key not in body or body[key] is None:
            continue
        val = body[key]
        if isinstance(val, list):
            sources.extend(val)
        else:
            sources.append(val)
    return sources


async def _read_part_bytes_limited(part: Any, *, max_bytes: int, param: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await part.read_chunk(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise GatewayError(
                f"Uploaded `{param}` exceeds maximum allowed size of {max_bytes // (1024 * 1024)} MiB.",
                status=413,
                param=param,
                code="image_too_large",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _is_binary_part(part: Any) -> bool:
    if getattr(part, "filename", None):
        return True
    ctype = (part.headers.get("Content-Type") or "").lower()
    return ctype.startswith("image/") or ctype == "application/octet-stream"


def _verify_admin_same_origin(request: web.Request) -> None:
    """Reject cross-site browser requests against internal ComfyUI Settings admin routes."""
    sec_fetch = request.headers.get("Sec-Fetch-Site", "").strip().lower()
    if sec_fetch == "cross-site":
        raise GatewayError(
            "Cross-site requests to gateway admin endpoints are forbidden.",
            status=403,
            error_type="permission_error",
            code="forbidden_cross_origin",
        )
    origin = request.headers.get("Origin", "").strip()
    host = request.headers.get("Host", "").strip().lower()
    if origin and host:
        origin_netloc = urlparse(origin).netloc.lower()
        if origin_netloc and origin_netloc != host:
            raise GatewayError(
                "Cross-origin requests to gateway admin endpoints are forbidden.",
                status=403,
                error_type="permission_error",
                code="forbidden_cross_origin",
            )


def _rebuild_alias_map(models: dict[str, ModelConfig]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for model_id, model in models.items():
        for alias in model.aliases:
            if alias in alias_map and alias_map[alias] != model_id:
                raise GatewayError(
                    f"Duplicate alias `{alias}` is already assigned to `{alias_map[alias]}`.",
                    status=400,
                    param="aliases",
                    code="duplicate_alias",
                )
            alias_map[alias] = model_id
    return alias_map


class GatewayService:
    """Coordinates configuration, template caching, input staging, and ComfyUI execution."""

    def __init__(
        self,
        config: GatewayConfig,
        backend: ComfyBackend,
        *,
        plugin_root: Path | None = None,
        config_loader: Callable[[], GatewayConfig] | None = None,
    ) -> None:
        self._config = config
        self._backend = backend
        self._plugin_root = (plugin_root or Path(__file__).resolve().parent.parent).resolve()
        self._config_loader = config_loader
        self._templates: dict[str, dict[str, Any]] = {}
        self._created_ts = int(time.time())
        self._load_templates(config)

    @property
    def config(self) -> GatewayConfig:
        return self._config

    @property
    def plugin_root(self) -> Path:
        return self._plugin_root

    def _load_templates(self, config: GatewayConfig) -> None:
        loaded: dict[str, dict[str, Any]] = {}
        for model_id, model in config.models.items():
            loaded[model_id] = load_workflow_template(model.workflow_path)
        self._templates = loaded

    def reload(self) -> GatewayConfig:
        if self._config_loader is None:
            raise GatewayError("Hot reload is not configured.", status=501, error_type="server_error")
        new_cfg = self._config_loader()
        self._load_templates(new_cfg)
        self._config = new_cfg
        log.info(
            "[OpenAI-Image-Gateway] Reloaded config (%s): %d model(s)",
            new_cfg.config_path or "none",
            len(new_cfg.models),
        )
        return new_cfg

    def _commit_config(self, new_cfg: GatewayConfig, new_templates: dict[str, dict[str, Any]]) -> GatewayConfig:
        saved_path = save_config(new_cfg, self._plugin_root)
        final_cfg = replace(new_cfg, config_path=saved_path)
        self._templates = new_templates
        self._config = final_cfg
        log.info(
            "[OpenAI-Image-Gateway] Saved config to %s (%d model(s), listed=%s)",
            saved_path,
            len(final_cfg.models),
            ",".join(final_cfg.listed_model_ids()) or "<none>",
        )
        return final_cfg

    def _resolve_base_url(self, request: web.Request) -> str:
        if self._config.public_base_url:
            return self._config.public_base_url
        scheme = request.headers.get("X-Forwarded-Proto") or request.scheme or "http"
        host = request.headers.get("X-Forwarded-Host") or request.host
        return f"{scheme}://{host}".rstrip("/")

    def _build_view_url(self, request: web.Request, *, filename: str, subfolder: str, file_type: str) -> str:
        params: dict[str, str] = {"filename": filename, "type": file_type}
        if subfolder:
            params["subfolder"] = subfolder
        return f"{self._resolve_base_url(request)}/view?{urlencode(params)}"

    def _serialize_admin_state(self, request: web.Request) -> dict[str, Any]:
        cfg = self._config
        base_dir = cfg.config_path.parent if cfg.config_path else self._plugin_root
        models_out: list[dict[str, Any]] = []
        for model_id, model in cfg.models.items():
            entry = serialize_model_config(model, base_dir=base_dir)
            entry["id"] = model_id
            entry["is_default"] = model_id == cfg.default_model
            entry["capabilities"] = model.capabilities
            entry["max_images"] = model.max_images
            models_out.append(entry)

        return {
            "base_url": f"{self._resolve_base_url(request)}/v1",
            "config_path": str(cfg.config_path) if cfg.config_path else str(self._plugin_root / "config.local.yaml"),
            "listed_models": cfg.listed_model_ids(),
            "settings": {
                "default_model": cfg.default_model,
                "fallback_to_default_model": cfg.fallback_to_default_model,
                "expose_aliases": cfg.expose_aliases,
                "default_response_format": cfg.default_response_format,
                "api_keys": list(cfg.api_keys),
                "cors_origins": cfg.cors_origins,
                "timeout": cfg.timeout,
                "max_n": cfg.max_n,
                "max_image_mb": round(cfg.max_image_bytes / (1024 * 1024), 2),
                "allow_remote_image_urls": cfg.allow_remote_image_urls,
                "keep_input_files": cfg.keep_input_files,
            },
            "models": models_out,
        }

    # ---------- OpenAI Public Endpoints ----------

    async def execute_image_request(
        self,
        request: web.Request,
        *,
        model_name: str | None,
        prompt: str | None,
        images: list[DecodedImage],
        mask: DecodedImage | None,
        n: int,
        size: str | None,
        response_format: str | None,
        output_format: str | None,
        output_compression: int | None,
        negative_prompt: str | None,
        seed: int | None,
        steps: int | None,
        cfg: float | None,
        denoise: float | None,
    ) -> dict[str, Any]:
        config = self._config
        model: ModelConfig = config.resolve_model(model_name)
        template = self._templates[model.id]

        prompt_clean = (prompt or "").strip() if prompt is not None else ""
        if model.prompt_required and not prompt_clean:
            raise GatewayError(
                "`prompt` is required and cannot be empty.",
                status=400,
                param="prompt",
                code="missing_prompt",
            )

        max_n = min(config.max_n, model.max_n) if model.max_n is not None else config.max_n
        if n < 1 or n > max_n:
            raise GatewayError(
                f"`n` must be between 1 and {max_n}.",
                status=400,
                param="n",
                code="invalid_n",
            )

        resp_fmt = (response_format or config.default_response_format).strip().lower()
        if resp_fmt not in ("b64_json", "url"):
            raise GatewayError(
                f"Unsupported `response_format` `{response_format}`. Expected `b64_json` or `url`.",
                status=400,
                param="response_format",
                code="invalid_response_format",
            )

        parsed_size = parse_size(size, multiple=model.size_multiple)
        req_width = parsed_size[0] if parsed_size else None
        req_height = parsed_size[1] if parsed_size else None

        req_id = uuid.uuid4().hex[:12]
        staged_paths: list[str] = []
        staged_image_paths: list[str] = []
        staged_mask_path: str | None = None
        should_cleanup = not config.keep_input_files

        try:
            for idx, img in enumerate(images, start=1):
                annotated = await self._backend.save_input_image(
                    img.data,
                    f"{req_id}_img{idx}.{img.ext}",
                    config.input_subfolder,
                )
                staged_paths.append(annotated)
                staged_image_paths.append(annotated)

            if mask is not None and model.mask_input is not None:
                staged_mask_path = await self._backend.save_input_image(
                    mask.data,
                    f"{req_id}_mask.{mask.ext}",
                    config.input_subfolder,
                )
                staged_paths.append(staged_mask_path)

            timeout = model.timeout if model.timeout is not None else config.timeout
            data_items: list[dict[str, Any]] = []
            final_format: str | None = None
            final_size: str | None = None

            for run_idx in range(n):
                prompt_graph, _ = build_prompt_graph(
                    model,
                    template,
                    prompt=prompt_clean if prompt is not None else None,
                    negative_prompt=negative_prompt,
                    image_filenames=staged_image_paths,
                    mask_filename=staged_mask_path,
                    width=req_width,
                    height=req_height,
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    denoise=denoise,
                    run_index=run_idx,
                )
                rendered = await self._backend.run_workflow(
                    prompt_graph,
                    output_node=model.output_node,
                    timeout=timeout,
                    client_id=f"openai-gw-{req_id}-{run_idx}",
                )
                for item in rendered:
                    if len(data_items) >= n:
                        break
                    if resp_fmt == "url":
                        url = self._build_view_url(
                            request,
                            filename=item.filename,
                            subfolder=item.subfolder,
                            file_type=item.file_type,
                        )
                        data_items.append({"url": url})
                        final_format = item.ext
                        if item.width and item.height:
                            final_size = f"{item.width}x{item.height}"
                    else:
                        out_bytes, out_fmt, out_w, out_h = convert_output_image(
                            item.data,
                            output_format=output_format,
                            compression=output_compression,
                        )
                        data_items.append(
                            {"b64_json": base64.b64encode(out_bytes).decode("ascii")}
                        )
                        final_format = out_fmt
                        if out_w and out_h:
                            final_size = f"{out_w}x{out_h}"
                if len(data_items) >= n:
                    break

            payload: dict[str, Any] = {
                "created": int(time.time()),
                "data": data_items,
            }
            if final_format:
                payload["output_format"] = final_format
            if final_size:
                payload["size"] = final_size
            return payload
        except asyncio.CancelledError:
            should_cleanup = False
            raise
        finally:
            if should_cleanup and staged_paths:
                await self._backend.cleanup_input_files(staged_paths)

    async def handle_list_models(self, request: web.Request) -> web.Response:
        verify_api_key(request, self._config.api_keys)
        data = [
            {
                "id": model_id,
                "object": "model",
                "created": self._created_ts,
                "owned_by": "comfyui",
            }
            for model_id in self._config.listed_model_ids()
        ]
        return web.json_response({"object": "list", "data": data})

    async def handle_get_model(self, request: web.Request) -> web.Response:
        verify_api_key(request, self._config.api_keys)
        requested = request.match_info.get("model_id", "").strip()
        if requested not in self._config.listed_model_ids():
            raise GatewayError(
                f"The model `{requested}` does not exist.",
                status=404,
                param="model",
                code="model_not_found",
            )
        return web.json_response(
            {
                "id": requested,
                "object": "model",
                "created": self._created_ts,
                "owned_by": "comfyui",
            }
        )

    async def _parse_json_image_request(
        self,
        request: web.Request,
        *,
        require_image: bool,
    ) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception as exc:
            raise GatewayError(
                f"Invalid JSON request body: {exc}",
                status=400,
                code="invalid_json",
            ) from exc
        if not isinstance(body, dict):
            raise GatewayError("Request body must be a JSON object.", status=400, code="invalid_json")

        raw_sources = _collect_json_image_sources(body)
        if require_image and not raw_sources:
            raise GatewayError(
                "At least one `image` is required for `/v1/images/edits`.",
                status=400,
                param="image",
                code="missing_image",
            )

        images: list[DecodedImage] = []
        for src in raw_sources:
            images.append(
                await decode_image_source(
                    src,
                    max_bytes=self._config.max_image_bytes,
                    allow_remote_urls=self._config.allow_remote_image_urls,
                    remote_timeout=self._config.remote_url_timeout,
                    param="image",
                )
            )

        mask: DecodedImage | None = None
        if body.get("mask") is not None:
            mask = await decode_image_source(
                body["mask"],
                max_bytes=self._config.max_image_bytes,
                allow_remote_urls=self._config.allow_remote_image_urls,
                remote_timeout=self._config.remote_url_timeout,
                param="mask",
            )

        return await self.execute_image_request(
            request,
            model_name=body.get("model"),
            prompt=body.get("prompt"),
            images=images,
            mask=mask,
            n=_parse_int(body.get("n"), name="n", default=1) or 1,
            size=body.get("size"),
            response_format=body.get("response_format"),
            output_format=body.get("output_format"),
            output_compression=_parse_int(body.get("output_compression"), name="output_compression"),
            negative_prompt=body.get("negative_prompt"),
            seed=_parse_int(body.get("seed"), name="seed"),
            steps=_parse_int(body.get("steps"), name="steps"),
            cfg=_parse_float(body.get("cfg"), name="cfg"),
            denoise=_parse_float(body.get("denoise"), name="denoise"),
        )

    async def _parse_multipart_edit_request(self, request: web.Request) -> dict[str, Any]:
        reader = await request.multipart()
        fields: dict[str, str] = {}
        images: list[DecodedImage] = []
        mask: DecodedImage | None = None

        while True:
            part = await reader.next()
            if part is None:
                break
            name = (part.name or "").strip()
            if not name:
                continue

            if _MULTIPART_IMAGE_FIELD_RE.match(name):
                if _is_binary_part(part):
                    raw = await _read_part_bytes_limited(
                        part, max_bytes=self._config.max_image_bytes, param=name
                    )
                    images.append(
                        decode_image_bytes(raw, max_bytes=self._config.max_image_bytes, param=name)
                    )
                else:
                    text = (await part.text()).strip()
                    if text:
                        images.append(
                            await decode_image_source(
                                text,
                                max_bytes=self._config.max_image_bytes,
                                allow_remote_urls=self._config.allow_remote_image_urls,
                                remote_timeout=self._config.remote_url_timeout,
                                param=name,
                            )
                        )
            elif name == "mask":
                if _is_binary_part(part):
                    raw = await _read_part_bytes_limited(
                        part, max_bytes=self._config.max_image_bytes, param="mask"
                    )
                    mask = decode_image_bytes(raw, max_bytes=self._config.max_image_bytes, param="mask")
                else:
                    text = (await part.text()).strip()
                    if text:
                        mask = await decode_image_source(
                            text,
                            max_bytes=self._config.max_image_bytes,
                            allow_remote_urls=self._config.allow_remote_image_urls,
                            remote_timeout=self._config.remote_url_timeout,
                            param="mask",
                        )
            else:
                fields[name] = await part.text()

        if not images:
            raise GatewayError(
                "At least one `image` (or `image[]`) file is required for `/v1/images/edits`.",
                status=400,
                param="image",
                code="missing_image",
            )

        return await self.execute_image_request(
            request,
            model_name=fields.get("model"),
            prompt=fields.get("prompt"),
            images=images,
            mask=mask,
            n=_parse_int(fields.get("n"), name="n", default=1) or 1,
            size=fields.get("size"),
            response_format=fields.get("response_format"),
            output_format=fields.get("output_format"),
            output_compression=_parse_int(fields.get("output_compression"), name="output_compression"),
            negative_prompt=fields.get("negative_prompt"),
            seed=_parse_int(fields.get("seed"), name="seed"),
            steps=_parse_int(fields.get("steps"), name="steps"),
            cfg=_parse_float(fields.get("cfg"), name="cfg"),
            denoise=_parse_float(fields.get("denoise"), name="denoise"),
        )

    async def handle_images_edits(self, request: web.Request) -> web.Response:
        verify_api_key(request, self._config.api_keys)
        ctype = (request.content_type or "").lower()
        if ctype.startswith("multipart/"):
            payload = await self._parse_multipart_edit_request(request)
        elif ctype == "application/json":
            payload = await self._parse_json_image_request(request, require_image=True)
        elif ctype == "application/x-www-form-urlencoded":
            post_data = await request.post()
            raw_sources: list[Any] = []
            fields: dict[str, str] = {}
            for key, value in post_data.items():
                if _MULTIPART_IMAGE_FIELD_RE.match(key):
                    if isinstance(value, str) and value.strip():
                        raw_sources.append(value.strip())
                elif isinstance(value, str):
                    fields[key] = value
            if not raw_sources:
                raise GatewayError(
                    "At least one `image` (or `image[]`) file is required for `/v1/images/edits`.",
                    status=400,
                    param="image",
                    code="missing_image",
                )
            images = [
                await decode_image_source(
                    src,
                    max_bytes=self._config.max_image_bytes,
                    allow_remote_urls=self._config.allow_remote_image_urls,
                    remote_timeout=self._config.remote_url_timeout,
                    param="image",
                )
                for src in raw_sources
            ]
            payload = await self.execute_image_request(
                request,
                model_name=fields.get("model"),
                prompt=fields.get("prompt"),
                images=images,
                mask=None,
                n=_parse_int(fields.get("n"), name="n", default=1) or 1,
                size=fields.get("size"),
                response_format=fields.get("response_format"),
                output_format=fields.get("output_format"),
                output_compression=_parse_int(fields.get("output_compression"), name="output_compression"),
                negative_prompt=fields.get("negative_prompt"),
                seed=_parse_int(fields.get("seed"), name="seed"),
                steps=_parse_int(fields.get("steps"), name="steps"),
                cfg=_parse_float(fields.get("cfg"), name="cfg"),
                denoise=_parse_float(fields.get("denoise"), name="denoise"),
            )
        else:
            raise GatewayError(
                f"Unsupported Content-Type `{request.content_type}`. Expected `multipart/form-data` or `application/json`.",
                status=415,
                code="unsupported_media_type",
            )
        return web.json_response(payload)

    async def handle_images_generations(self, request: web.Request) -> web.Response:
        verify_api_key(request, self._config.api_keys)
        ctype = (request.content_type or "").lower()
        if ctype.startswith("multipart/"):
            payload = await self._parse_multipart_edit_request(request)
        else:
            payload = await self._parse_json_image_request(request, require_image=False)
        return web.json_response(payload)

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "configured": bool(self._config.models),
                "default_model": self._config.default_model,
                "models": self._config.listed_model_ids(),
                "auth_enabled": bool(self._config.api_keys),
            }
        )

    async def handle_reload(self, request: web.Request) -> web.Response:
        verify_api_key(request, self._config.api_keys)
        try:
            cfg = self.reload()
        except GatewayError:
            raise
        except Exception as exc:
            raise GatewayError(
                f"Failed to reload gateway configuration: {exc}",
                status=500,
                error_type="server_error",
                code="reload_failed",
            ) from exc
        return web.json_response(
            {
                "status": "reloaded",
                "default_model": cfg.default_model,
                "models": cfg.listed_model_ids(),
            }
        )

    # ---------- Internal ComfyUI Settings UI Admin Endpoints (Same-Origin Only) ----------

    async def handle_admin_get_config(self, request: web.Request) -> web.Response:
        _verify_admin_same_origin(request)
        return web.json_response(self._serialize_admin_state(request))

    async def handle_admin_analyze(self, request: web.Request) -> web.Response:
        _verify_admin_same_origin(request)
        try:
            body = await request.json()
        except Exception as exc:
            raise GatewayError(f"Invalid JSON body: {exc}", status=400, code="invalid_json") from exc
        wf = body.get("workflow_prompt") or body.get("workflow") if isinstance(body, dict) else None
        try:
            validated = validate_workflow_dict(wf, label="uploaded workflow")
        except ValueError as exc:
            raise GatewayError(str(exc), status=400, param="workflow_prompt", code="invalid_workflow") from exc
        return web.json_response(analyze_workflow(validated))

    async def handle_admin_upsert_model(self, request: web.Request) -> web.Response:
        _verify_admin_same_origin(request)
        try:
            body = await request.json()
        except Exception as exc:
            raise GatewayError(f"Invalid JSON body: {exc}", status=400, code="invalid_json") from exc
        if not isinstance(body, dict):
            raise GatewayError("Request body must be a JSON object.", status=400, code="invalid_json")

        model_id = validate_model_id(str(body.get("id") or ""))
        previous_id = str(body.get("previous_id") or "").strip()
        existing = self._config.models.get(previous_id or model_id) or self._config.models.get(model_id)

        workflow_prompt = body.get("workflow_prompt")
        if workflow_prompt is not None:
            try:
                template = validate_workflow_dict(workflow_prompt, label=model_id)
            except ValueError as exc:
                raise GatewayError(str(exc), status=400, param="workflow_prompt", code="invalid_workflow") from exc
            wf_path = save_managed_workflow(self._plugin_root, model_id, template)
            auto_analysis = analyze_workflow(template)
        elif body.get("workflow"):
            raw_wf = str(body["workflow"]).strip()
            candidate = Path(raw_wf).expanduser()
            if not candidate.is_absolute():
                candidate = (self._plugin_root / candidate).resolve()
            if not candidate.is_file():
                raise GatewayError(
                    f"Workflow file not found: {raw_wf}",
                    status=400,
                    param="workflow",
                    code="workflow_not_found",
                )
            try:
                template = load_workflow_template(candidate)
            except ValueError as exc:
                raise GatewayError(str(exc), status=400, param="workflow", code="invalid_workflow") from exc
            wf_path = candidate
            auto_analysis = analyze_workflow(template)
        elif existing is not None:
            wf_path = existing.workflow_path
            template = self._templates.get(existing.id) or load_workflow_template(wf_path)
            auto_analysis = analyze_workflow(template)
        else:
            raise GatewayError(
                "Provide `workflow_prompt` (API-format workflow JSON) or an existing `workflow` path.",
                status=400,
                param="workflow_prompt",
                code="missing_workflow",
            )

        use_auto = _parse_bool_strict(body.get("auto_analyze"), True)
        fallback_base: dict[str, Any] = auto_analysis if use_auto else {}

        raw_aliases = body.get("aliases")
        if raw_aliases is None:
            aliases = existing.aliases if existing is not None else ()
        elif isinstance(raw_aliases, str):
            aliases = tuple(a.strip() for a in raw_aliases.split(",") if a.strip())
        elif isinstance(raw_aliases, list):
            aliases = tuple(str(a).strip() for a in raw_aliases if str(a).strip())
        else:
            raise GatewayError("`aliases` must be a list or comma-separated string.", status=400, param="aliases")

        try:
            bindings_raw = body.get("bindings")
            if bindings_raw is not None:
                bindings = parse_bindings(bindings_raw, model_id)
            elif workflow_prompt is not None or existing is None:
                bindings = parse_bindings(fallback_base.get("bindings"), model_id)
            else:
                bindings = existing.bindings

            image_inputs_raw = body.get("image_inputs") if "image_inputs" in body else None
            if "image_inputs" in body:
                image_inputs = parse_image_inputs(image_inputs_raw, model_id)
            elif workflow_prompt is not None or existing is None:
                image_inputs = parse_image_inputs(fallback_base.get("image_inputs"), model_id)
            else:
                image_inputs = existing.image_inputs

            if "mask_input" in body:
                mask_input = parse_mask_input(body.get("mask_input"), model_id)
            elif workflow_prompt is not None or existing is None:
                mask_input = parse_mask_input(fallback_base.get("mask_input"), model_id)
            else:
                mask_input = existing.mask_input
        except ValueError as exc:
            raise GatewayError(str(exc), status=400, code="invalid_model_config") from exc

        if "output_node" in body:
            out_val = body.get("output_node")
            output_node = str(out_val).strip() or None if out_val is not None else None
        elif workflow_prompt is not None or existing is None:
            output_node = fallback_base.get("output_node")
        else:
            output_node = existing.output_node

        if "remove_nodes" in body:
            remove_nodes = tuple(str(n).strip() for n in (body.get("remove_nodes") or ()) if str(n).strip())
        elif workflow_prompt is not None or existing is None:
            remove_nodes = tuple(str(n) for n in (fallback_base.get("remove_nodes") or ()))
        else:
            remove_nodes = existing.remove_nodes

        source_workflow = (
            str(body.get("source_workflow") or "").strip()
            if "source_workflow" in body
            else (existing.source_workflow if existing is not None else "")
        )
        description = (
            str(body.get("description") or "").strip()
            if "description" in body
            else (existing.description if existing is not None else "")
        )
        prompt_required = _parse_bool_strict(
            body.get("prompt_required"),
            fallback_base.get("prompt_required", existing.prompt_required if existing else True),
        )
        allow_no_image = _parse_bool_strict(
            body.get("allow_no_image"),
            fallback_base.get("allow_no_image", existing.allow_no_image if existing else True),
        )
        size_multiple = max(
            1,
            _parse_int(
                body.get("size_multiple"),
                name="size_multiple",
                default=int(fallback_base.get("size_multiple", existing.size_multiple if existing else 16)),
            )
            or 16,
        )
        default_width = (
            _parse_int(body.get("default_width"), name="default_width")
            if "default_width" in body
            else (existing.default_width if existing else 1024)
        )
        default_height = (
            _parse_int(body.get("default_height"), name="default_height")
            if "default_height" in body
            else (existing.default_height if existing else 1024)
        )
        timeout = (
            _parse_float(body.get("timeout"), name="timeout")
            if "timeout" in body
            else (existing.timeout if existing else None)
        )

        new_model = ModelConfig(
            id=model_id,
            workflow_path=wf_path,
            source_workflow=source_workflow,
            aliases=aliases,
            description=description,
            output_node=output_node,
            prompt_required=prompt_required,
            allow_no_image=allow_no_image,
            default_width=default_width,
            default_height=default_height,
            size_multiple=size_multiple,
            max_n=existing.max_n if existing else None,
            timeout=timeout,
            defaults=dict(existing.defaults) if existing else {},
            bindings=bindings,
            image_inputs=image_inputs,
            mask_input=mask_input,
            remove_nodes=remove_nodes,
        )

        new_models = dict(self._config.models)
        new_templates = dict(self._templates)
        if previous_id and previous_id != model_id and previous_id in new_models:
            del new_models[previous_id]
            new_templates.pop(previous_id, None)
        new_models[model_id] = new_model
        new_templates[model_id] = template

        alias_map = _rebuild_alias_map(new_models)
        set_default = _parse_bool_strict(body.get("set_default"), False)
        default_model = self._config.default_model
        if set_default or not default_model or default_model not in new_models or default_model == previous_id:
            default_model = model_id

        updated_cfg = replace(
            self._config,
            models=new_models,
            alias_map=alias_map,
            default_model=default_model,
        )
        self._commit_config(updated_cfg, new_templates)
        return web.json_response(
            {
                "status": "saved",
                "model_id": model_id,
                "analysis": auto_analysis,
                "state": self._serialize_admin_state(request),
            }
        )

    async def handle_admin_delete_model(self, request: web.Request) -> web.Response:
        _verify_admin_same_origin(request)
        model_id = request.match_info.get("model_id", "").strip()
        if model_id not in self._config.models:
            raise GatewayError(
                f"Model `{model_id}` does not exist.",
                status=404,
                param="model_id",
                code="model_not_found",
            )

        removed = self._config.models[model_id]
        new_models = {k: v for k, v in self._config.models.items() if k != model_id}
        new_templates = {k: v for k, v in self._templates.items() if k != model_id}
        alias_map = _rebuild_alias_map(new_models)
        default_model = self._config.default_model
        if default_model == model_id:
            default_model = next(iter(new_models), None)

        updated_cfg = replace(
            self._config,
            models=new_models,
            alias_map=alias_map,
            default_model=default_model,
        )
        self._commit_config(updated_cfg, new_templates)

        # Clean up managed snapshot under workflows/local/ if no other model uses it.
        managed_dir = (self._plugin_root / "workflows" / "local").resolve()
        still_used = any(m.workflow_path.resolve() == removed.workflow_path.resolve() for m in new_models.values())
        if not still_used:
            try:
                wf_real = removed.workflow_path.resolve()
                if os.path.commonpath((str(managed_dir), str(wf_real))) == str(managed_dir) and wf_real.is_file():
                    wf_real.unlink(missing_ok=True)
            except OSError:
                pass

        return web.json_response(
            {
                "status": "deleted",
                "model_id": model_id,
                "state": self._serialize_admin_state(request),
            }
        )

    async def handle_admin_update_settings(self, request: web.Request) -> web.Response:
        _verify_admin_same_origin(request)
        try:
            body = await request.json()
        except Exception as exc:
            raise GatewayError(f"Invalid JSON body: {exc}", status=400, code="invalid_json") from exc
        if not isinstance(body, dict):
            raise GatewayError("Request body must be a JSON object.", status=400, code="invalid_json")

        cfg = self._config
        default_model = cfg.default_model
        if "default_model" in body:
            dm = str(body.get("default_model") or "").strip() or None
            if dm is not None and dm not in cfg.models:
                raise GatewayError(
                    f"default_model `{dm}` is not in configured models.",
                    status=400,
                    param="default_model",
                )
            default_model = dm

        resp_fmt = cfg.default_response_format
        if "default_response_format" in body:
            resp_fmt = str(body.get("default_response_format") or "b64_json").strip().lower()
            if resp_fmt not in ("b64_json", "url"):
                raise GatewayError(
                    "default_response_format must be `b64_json` or `url`.",
                    status=400,
                    param="default_response_format",
                )

        api_keys = cfg.api_keys
        if "api_keys" in body:
            raw_keys = body.get("api_keys")
            if isinstance(raw_keys, str):
                api_keys = tuple(k.strip() for k in raw_keys.split(",") if k.strip())
            elif isinstance(raw_keys, list):
                api_keys = tuple(str(k).strip() for k in raw_keys if str(k).strip())
            elif raw_keys is None:
                api_keys = ()
            else:
                raise GatewayError("`api_keys` must be a list or comma-separated string.", status=400, param="api_keys")

        timeout = cfg.timeout
        if "timeout" in body:
            t_val = _parse_float(body.get("timeout"), name="timeout")
            if t_val is not None and t_val >= 0:
                timeout = t_val

        max_n = cfg.max_n
        if "max_n" in body:
            mn = _parse_int(body.get("max_n"), name="max_n")
            if mn is not None and mn >= 1:
                max_n = mn

        max_image_bytes = cfg.max_image_bytes
        if "max_image_mb" in body:
            mb = _parse_float(body.get("max_image_mb"), name="max_image_mb")
            if mb is not None and mb > 0:
                max_image_bytes = max(1024, int(mb * 1024 * 1024))

        updated_cfg = replace(
            cfg,
            default_model=default_model,
            fallback_to_default_model=_parse_bool_strict(
                body.get("fallback_to_default_model"), cfg.fallback_to_default_model
            ),
            expose_aliases=_parse_bool_strict(body.get("expose_aliases"), cfg.expose_aliases),
            default_response_format=resp_fmt,
            api_keys=api_keys,
            cors_origins=str(body.get("cors_origins")) if "cors_origins" in body and body["cors_origins"] is not None else cfg.cors_origins,
            timeout=timeout,
            max_n=max_n,
            max_image_bytes=max_image_bytes,
            allow_remote_image_urls=_parse_bool_strict(
                body.get("allow_remote_image_urls"), cfg.allow_remote_image_urls
            ),
            keep_input_files=_parse_bool_strict(body.get("keep_input_files"), cfg.keep_input_files),
        )
        self._commit_config(updated_cfg, dict(self._templates))
        return web.json_response({"status": "saved", "state": self._serialize_admin_state(request)})


def _wrap_handler(
    service: GatewayService,
    handler: Callable[[web.Request], Any],
    *,
    with_cors: bool = True,
) -> Callable[[web.Request], Any]:
    async def wrapped(request: web.Request) -> web.StreamResponse:
        try:
            resp = await handler(request)
        except GatewayError as err:
            resp = web.json_response(err.to_dict(), status=err.status)
        except web.HTTPException as http_err:
            err = GatewayError(
                http_err.reason or "HTTP error",
                status=http_err.status,
                error_type="invalid_request_error" if http_err.status < 500 else "server_error",
            )
            resp = web.json_response(err.to_dict(), status=http_err.status)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("[OpenAI-Image-Gateway] Unhandled error on %s %s", request.method, request.path)
            err = GatewayError(
                f"Internal server error: {exc}",
                status=500,
                error_type="server_error",
                code="internal_error",
            )
            resp = web.json_response(err.to_dict(), status=500)
        if with_cors:
            return apply_cors_headers(request, resp, service.config.cors_origins)
        return resp

    return wrapped


def register_routes(app: web.Application, service: GatewayService) -> int:
    """Register OpenAI-compatible image routes and internal Settings UI admin routes."""
    async def _options(request: web.Request) -> web.Response:
        return preflight_response(request, service.config.cors_origins)

    route_specs = [
        # OpenAI-compatible public routes
        ("GET", "/v1/models", _wrap_handler(service, service.handle_list_models, with_cors=True)),
        ("OPTIONS", "/v1/models", _options),
        ("GET", "/v1/models/{model_id}", _wrap_handler(service, service.handle_get_model, with_cors=True)),
        ("OPTIONS", "/v1/models/{model_id}", _options),
        ("POST", "/v1/images/edits", _wrap_handler(service, service.handle_images_edits, with_cors=True)),
        ("OPTIONS", "/v1/images/edits", _options),
        ("POST", "/v1/images/generations", _wrap_handler(service, service.handle_images_generations, with_cors=True)),
        ("OPTIONS", "/v1/images/generations", _options),
        ("GET", "/v1/gateway/health", _wrap_handler(service, service.handle_health, with_cors=True)),
        ("OPTIONS", "/v1/gateway/health", _options),
        ("POST", "/v1/gateway/reload", _wrap_handler(service, service.handle_reload, with_cors=True)),
        ("OPTIONS", "/v1/gateway/reload", _options),
        # ComfyUI Settings UI internal admin routes (same-origin only, no wildcard CORS)
        ("GET", "/api/openai-gateway/admin/config", _wrap_handler(service, service.handle_admin_get_config, with_cors=False)),
        ("POST", "/api/openai-gateway/admin/analyze", _wrap_handler(service, service.handle_admin_analyze, with_cors=False)),
        ("POST", "/api/openai-gateway/admin/models", _wrap_handler(service, service.handle_admin_upsert_model, with_cors=False)),
        ("DELETE", "/api/openai-gateway/admin/models/{model_id}", _wrap_handler(service, service.handle_admin_delete_model, with_cors=False)),
        ("PUT", "/api/openai-gateway/admin/settings", _wrap_handler(service, service.handle_admin_update_settings, with_cors=False)),
    ]

    registered = 0
    for method, path, handler in route_specs:
        try:
            app.router.add_route(method, path, handler)
            registered += 1
        except (RuntimeError, ValueError) as exc:
            log.warning("[OpenAI-Image-Gateway] Skipped route %s %s: %s", method, path, exc)
    return registered
