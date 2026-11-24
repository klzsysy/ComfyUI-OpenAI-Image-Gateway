"""ComfyUI custom-node entrypoint for ComfyUI-OpenAI-Image-Gateway."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parent

if __package__:
    from .gateway.backend import InProcessComfyBackend
    from .gateway.config import load_config
    from .gateway.models import GatewayConfig
    from .gateway.routes import GatewayService, register_routes
else:
    if str(_PLUGIN_ROOT) not in sys.path:
        sys.path.insert(0, str(_PLUGIN_ROOT))
    from gateway.backend import InProcessComfyBackend
    from gateway.config import load_config
    from gateway.models import GatewayConfig
    from gateway.routes import GatewayService, register_routes

log = logging.getLogger("comfyui_openai_image_gateway")

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

_SERVICE: GatewayService | None = None


def _bootstrap() -> None:
    global _SERVICE
    try:
        from server import PromptServer
    except ImportError:
        return

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None or getattr(prompt_server, "app", None) is None:
        return

    def _loader() -> GatewayConfig:
        return load_config(_PLUGIN_ROOT)

    try:
        config = _loader()
    except Exception as exc:  # noqa: BLE001
        log.error("[OpenAI-Image-Gateway] Failed to load config: %s", exc)
        config = GatewayConfig()

    backend = InProcessComfyBackend(prompt_server)
    service = GatewayService(config, backend, plugin_root=_PLUGIN_ROOT, config_loader=_loader)
    count = register_routes(prompt_server.app, service)
    _SERVICE = service

    if config.models:
        log.info(
            "[OpenAI-Image-Gateway] Registered %d route(s) on ComfyUI server (models=%s, default=%s, auth=%s)",
            count,
            ",".join(config.listed_model_ids()),
            config.default_model,
            "on" if config.api_keys else "off",
        )
    else:
        log.info(
            "[OpenAI-Image-Gateway] Registered %d route(s); no workflows configured yet. "
            "Open ComfyUI Settings -> OpenAI Image Gateway to add workflows.",
            count,
        )


_bootstrap()
