"""End-to-end HTTP integration tests for all OpenAI-compatible gateway endpoints."""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from gateway.backend import ComfyBackend
from gateway.config import load_config
from gateway.models import GatewayConfig, RenderedImage
from gateway.routes import GatewayService, register_routes

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def _png_bytes(size=(64, 48), color=(30, 120, 220)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class FakeComfyBackend(ComfyBackend):
    def __init__(self, output_bytes: bytes | None = None) -> None:
        self.output_bytes = output_bytes or _png_bytes((96, 128), (200, 100, 50))
        self.saved_inputs: dict[str, bytes] = {}
        self.cleaned_inputs: list[str] = []
        self.submitted_prompts: list[dict[str, Any]] = []

    async def save_input_image(self, data: bytes, filename: str, subfolder: str) -> str:
        rel = f"{subfolder}/{filename}" if subfolder else filename
        self.saved_inputs[rel] = data
        return rel

    async def cleanup_input_files(self, annotated_paths: list[str]) -> None:
        self.cleaned_inputs.extend(annotated_paths)

    async def run_workflow(
        self,
        prompt: dict[str, Any],
        *,
        output_node: str | None = None,
        timeout: float = 600.0,
        client_id: str = "test",
    ) -> list[RenderedImage]:
        self.submitted_prompts.append(prompt)
        with Image.open(io.BytesIO(self.output_bytes)) as img:
            w, h = img.size
        return [
            RenderedImage(
                data=self.output_bytes,
                filename=f"out_{len(self.submitted_prompts)}.png",
                subfolder="",
                file_type="output",
                ext="png",
                width=w,
                height=h,
            )
        ]


def _make_app(
    *,
    api_keys: str = "",
    cors_origins: str = "*",
    default_response_format: str = "b64_json",
) -> tuple[web.Application, FakeComfyBackend, GatewayService]:
    env = {
        "COMFY_OPENAI_GATEWAY_API_KEYS": api_keys,
        "COMFY_OPENAI_GATEWAY_CORS_ORIGINS": cors_origins,
        "COMFY_OPENAI_GATEWAY_RESPONSE_FORMAT": default_response_format,
    }
    cfg = load_config(PLUGIN_ROOT, explicit_path=PLUGIN_ROOT / "config.example.yaml", env=env)
    backend = FakeComfyBackend()
    service = GatewayService(cfg, backend, config_loader=lambda: cfg)
    app = web.Application()
    register_routes(app, service)
    return app, backend, service


def test_models_health_cors_and_auth():
    async def _run():
        app, _, _ = _make_app(api_keys="sk-secret")
        async with TestClient(TestServer(app)) as client:
            # Preflight CORS + Private Network Access
            opt = await client.options(
                "/v1/images/edits",
                headers={
                    "Origin": "https://tryon.example.com",
                    "Access-Control-Request-Private-Network": "true",
                },
            )
            assert opt.status == 204
            assert opt.headers["Access-Control-Allow-Origin"] == "*"
            assert opt.headers["Access-Control-Allow-Private-Network"] == "true"

            # Health does not require auth
            h = await client.get("/v1/gateway/health")
            assert h.status == 200
            h_json = await h.json()
            assert h_json["configured"] is True
            assert h_json["auth_enabled"] is True

            # Unauthorized /v1/models returns OpenAI 401 envelope with CORS headers
            unauth = await client.get("/v1/models")
            assert unauth.status == 401
            assert unauth.headers["Access-Control-Allow-Origin"] == "*"
            err = await unauth.json()
            assert err["error"]["type"] == "authentication_error"

            # Authorized /v1/models lists configured primary model name cleanly
            ok = await client.get("/v1/models", headers={"Authorization": "Bearer sk-secret"})
            assert ok.status == 200
            body = await ok.json()
            ids = [item["id"] for item in body["data"]]
            assert ids == ["qwen-image-2.1-turbo-edit"]

            # Single model retrieval works with X-API-Key too
            m = await client.get("/v1/models/qwen-image-2.1-turbo-edit", headers={"X-API-Key": "sk-secret"})
            assert m.status == 200
            assert (await m.json())["id"] == "qwen-image-2.1-turbo-edit"

            # Hot reload works
            rl = await client.post("/v1/gateway/reload", headers={"Authorization": "Bearer sk-secret"})
            assert rl.status == 200
            assert (await rl.json())["status"] == "reloaded"

    asyncio.run(_run())


def test_images_edits_multipart_multi_image_brackets():
    async def _run():
        app, backend, _ = _make_app()
        img1 = _png_bytes((64, 96), (255, 0, 0))
        img2 = _png_bytes((80, 80), (0, 255, 0))

        form = aiohttp.FormData()
        form.add_field("model", "gpt-image-1")
        form.add_field("prompt", "Virtual try-on: person from image 1 wearing dress from image 2")
        form.add_field("image[]", img1, filename="person.png", content_type="image/png")
        form.add_field("image[]", img2, filename="dress.png", content_type="image/png")
        form.add_field("quality", "high")  # standard OpenAI field accepted gracefully
        form.add_field("output_format", "jpeg")

        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/v1/images/edits", data=form)
            assert resp.status == 200
            body = await resp.json()
            assert body["output_format"] == "jpeg"
            assert body["size"] == "96x128"
            assert len(body["data"]) == 1
            raw_out = base64.b64decode(body["data"][0]["b64_json"])
            with Image.open(io.BytesIO(raw_out)) as im:
                assert im.format == "JPEG"
                assert im.size == (96, 128)

        # Verify both images were staged, wired into TextEncodeQwenImage21, and cleaned up!
        assert len(backend.saved_inputs) == 2
        assert len(backend.cleaned_inputs) == 2
        assert len(backend.submitted_prompts) == 1
        submitted = backend.submitted_prompts[0]
        enc = submitted["459:474"]["inputs"]
        assert enc["images.image_1"] == ["__gw_load_image_1", 0]
        assert enc["images.image_2"] == ["__gw_load_image_2", 0]
        assert submitted["459:468"]["inputs"]["switch"] is False

    asyncio.run(_run())


def test_images_generations_json_multi_image_and_url_response():
    async def _run():
        app, backend, _ = _make_app()
        img1_b64 = base64.b64encode(_png_bytes((32, 32), (10, 20, 30))).decode("ascii")
        img2_b64 = base64.b64encode(_png_bytes((32, 32), (40, 50, 60))).decode("ascii")

        payload = {
            "model": "gpt-image-2",
            "prompt": "Combine both references",
            "image": [f"data:image/png;base64,{img1_b64}", img2_b64],
            "size": "1024x1536",
            "n": 2,
            "seed": 100,
            "response_format": "url",
        }

        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/v1/images/generations", json=payload)
            assert resp.status == 200
            body = await resp.json()
            assert len(body["data"]) == 2
            assert "/view?filename=out_1.png" in body["data"][0]["url"]
            assert "/view?filename=out_2.png" in body["data"][1]["url"]

        assert len(backend.submitted_prompts) == 2
        assert backend.submitted_prompts[0]["459:504"]["inputs"]["noise_seed"] == 100
        assert backend.submitted_prompts[1]["459:504"]["inputs"]["noise_seed"] == 101
        assert backend.submitted_prompts[0]["459:468"]["inputs"]["switch"] is True
        assert backend.submitted_prompts[0]["459:456"]["inputs"]["width"] == 1024
        assert backend.submitted_prompts[0]["459:456"]["inputs"]["height"] == 1536

    asyncio.run(_run())


def test_images_edits_validation_errors():
    async def _run():
        app, _, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            # Missing image in multipart
            form = aiohttp.FormData()
            form.add_field("prompt", "hello")
            r1 = await client.post("/v1/images/edits", data=form)
            assert r1.status == 400
            assert (await r1.json())["error"]["code"] == "missing_image"

            # Invalid image bytes
            form2 = aiohttp.FormData()
            form2.add_field("prompt", "hello")
            form2.add_field("image", b"not-an-image", filename="bad.png", content_type="image/png")
            r2 = await client.post("/v1/images/edits", data=form2)
            assert r2.status == 400
            assert (await r2.json())["error"]["code"] == "invalid_image"

            # Missing prompt
            form3 = aiohttp.FormData()
            form3.add_field("image[]", _png_bytes(), filename="ok.png", content_type="image/png")
            r3 = await client.post("/v1/images/edits", data=form3)
            assert r3.status == 400
            assert (await r3.json())["error"]["code"] == "missing_prompt"

    asyncio.run(_run())


def test_unconfigured_gateway_returns_503_gracefully():
    async def _run():
        cfg = GatewayConfig()
        backend = FakeComfyBackend()
        service = GatewayService(cfg, backend)
        app = web.Application()
        register_routes(app, service)

        async with TestClient(TestServer(app)) as client:
            h = await client.get("/v1/gateway/health")
            assert h.status == 200
            assert (await h.json())["configured"] is False

            m = await client.get("/v1/models")
            assert m.status == 200
            assert (await m.json())["data"] == []

            r = await client.post("/v1/images/generations", json={"prompt": "test"})
            assert r.status == 503
            assert (await r.json())["error"]["code"] == "gateway_not_configured"

    asyncio.run(_run())


def test_comfyui_custom_node_entrypoint_loads_with_mangled_module_name():
    import importlib.util
    import sys

    sys_name = str(PLUGIN_ROOT).replace(".", "_x_")
    spec = importlib.util.spec_from_file_location(sys_name, PLUGIN_ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[sys_name] = mod
    try:
        spec.loader.exec_module(mod)
        assert hasattr(mod, "NODE_CLASS_MAPPINGS")
        assert isinstance(mod.NODE_CLASS_MAPPINGS, dict)
        assert hasattr(mod, "NODE_DISPLAY_NAME_MAPPINGS")
        assert getattr(mod, "WEB_DIRECTORY", None) == "./web"
    finally:
        sys.modules.pop(sys_name, None)


def test_admin_multi_workflow_lifecycle_and_routing(tmp_path: Path):
    async def _run():
        backend = FakeComfyBackend()
        service = GatewayService(GatewayConfig(), backend, plugin_root=tmp_path)
        app = web.Application()
        register_routes(app, service)

        wf_edit = {
            "461": {"class_type": "SaveImageAdvanced", "inputs": {"images": ["10", 0]}},
            "470": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
            "474": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": "", "negative_prompt": "", "images.image_1": ["470", 0]}},
            "504": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
        }
        wf_t2i = {
            "3": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 7.0, "positive": ["6", 0], "negative": ["7", 0]}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
        }

        async with TestClient(TestServer(app)) as client:
            # Cross-site browser request to admin endpoint must be blocked
            blocked = await client.get("/api/openai-gateway/admin/config", headers={"Sec-Fetch-Site": "cross-site"})
            assert blocked.status == 403

            # Cross-origin Origin header (without Sec-Fetch-Site) must also be blocked
            blocked2 = await client.post(
                "/api/openai-gateway/admin/models",
                json={"id": "evil", "workflow_prompt": wf_t2i},
                headers={"Origin": "https://evil.example.com", "X-Forwarded-Host": "comfy.example.com"},
            )
            assert blocked2.status == 403

            # Reverse-proxied same-origin request (Host rewritten, X-Forwarded-Host matches Origin) must succeed
            proxied_ok = await client.get(
                "/api/openai-gateway/admin/config",
                headers={
                    "Origin": "https://comfy.example.com",
                    "X-Forwarded-Host": "comfy.example.com:443",
                    "X-Forwarded-Proto": "https",
                },
            )
            assert proxied_ok.status == 200
            assert (await proxied_ok.json())["base_url"] == "https://comfy.example.com:443/v1"

            # Reverse-proxied request with Sec-Fetch-Site: same-origin (even without X-Forwarded-Host) must succeed
            sec_same_ok = await client.get(
                "/api/openai-gateway/admin/config",
                headers={
                    "Origin": "https://comfy.example.com",
                    "Sec-Fetch-Site": "same-origin",
                },
            )
            assert sec_same_ok.status == 200

            # Add workflow 1 under custom name "my-qwen-edit"
            r1 = await client.post(
                "/api/openai-gateway/admin/models",
                json={
                    "id": "my-qwen-edit",
                    "workflow_prompt": wf_edit,
                    "source_workflow": "edit.json",
                    "set_default": True,
                },
            )
            assert r1.status == 200
            d1 = await r1.json()
            assert d1["analysis"]["output_node"] == "461"
            assert d1["state"]["listed_models"] == ["my-qwen-edit"]

            # Add workflow 2 under custom name "my-sdxl-t2i"
            r2 = await client.post(
                "/api/openai-gateway/admin/models",
                json={
                    "id": "my-sdxl-t2i",
                    "workflow_prompt": wf_t2i,
                    "source_workflow": "t2i.json",
                },
            )
            assert r2.status == 200
            assert (await r2.json())["state"]["listed_models"] == ["my-qwen-edit", "my-sdxl-t2i"]

            # GET /v1/models lists ONLY the two user-defined workflow names
            m_resp = await client.get("/v1/models")
            assert m_resp.status == 200
            assert [m["id"] for m in (await m_resp.json())["data"]] == ["my-qwen-edit", "my-sdxl-t2i"]

            # Calling model="my-qwen-edit" executes wf_edit; model="my-sdxl-t2i" executes wf_t2i!
            form = aiohttp.FormData()
            form.add_field("model", "my-qwen-edit")
            form.add_field("prompt", "edit dress")
            form.add_field("image[]", _png_bytes(), filename="a.png", content_type="image/png")
            e_resp = await client.post("/v1/images/edits", data=form)
            assert e_resp.status == 200
            assert "474" in backend.submitted_prompts[-1]
            assert backend.submitted_prompts[-1]["474"]["inputs"]["prompt"] == "edit dress"

            g_resp = await client.post(
                "/v1/images/generations",
                json={"model": "my-sdxl-t2i", "prompt": "a mountain lake", "size": "768x1024"},
            )
            assert g_resp.status == 200
            assert "6" in backend.submitted_prompts[-1]
            assert backend.submitted_prompts[-1]["6"]["inputs"]["text"] == "a mountain lake"
            assert backend.submitted_prompts[-1]["5"]["inputs"]["width"] == 768

            # Rename model 1 in place -> preserves order, migrates managed workflow file, updates /v1/models
            rn_resp = await client.post(
                "/api/openai-gateway/admin/models",
                json={
                    "previous_id": "my-qwen-edit",
                    "id": "my-qwen-edit-bf16",
                    "auto_analyze": False,
                },
            )
            assert rn_resp.status == 200
            rn_body = await rn_resp.json()
            assert rn_body["state"]["listed_models"] == ["my-qwen-edit-bf16", "my-sdxl-t2i"]
            assert (tmp_path / "workflows" / "local" / "my-qwen-edit-bf16.json").is_file()
            assert not (tmp_path / "workflows" / "local" / "my-qwen-edit.json").exists()

            # Update settings (disable fallback) and delete first model
            s_resp = await client.put(
                "/api/openai-gateway/admin/settings",
                json={"fallback_to_default_model": False, "default_model": "my-sdxl-t2i"},
            )
            assert s_resp.status == 200
            assert (await s_resp.json())["state"]["settings"]["fallback_to_default_model"] is False

            del_resp = await client.delete("/api/openai-gateway/admin/models/my-qwen-edit-bf16")
            assert del_resp.status == 200
            assert (await del_resp.json())["state"]["listed_models"] == ["my-sdxl-t2i"]

            # Verify config.local.yaml was persisted on disk
            assert (tmp_path / "config.local.yaml").is_file()

    asyncio.run(_run())

