"""Core data models and OpenAI-formatted error types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class GatewayError(Exception):
    """Structured error serialized as an OpenAI API error envelope."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.error_type = error_type
        self.param = param
        self.code = code

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


@dataclass(frozen=True)
class ImageTargetSpec:
    """Target node receiving dynamically created LoadImage links."""

    node_id: str
    input_prefix: str = "images.image_"
    start_index: int = 1
    clear_existing_prefix: bool = True
    prune_if_empty: bool = False


@dataclass(frozen=True)
class LoadImageNodeSlot:
    """Fixed LoadImage node slot in a workflow template."""

    node_id: str
    input_name: str = "image"


@dataclass(frozen=True)
class ImageInputSpec:
    """How incoming reference/edit images are wired into a workflow."""

    mode: str = "dynamic_load_image"  # "dynamic_load_image" | "load_image_nodes"
    max_images: int = 10
    targets: tuple[ImageTargetSpec, ...] = ()
    nodes: tuple[LoadImageNodeSlot, ...] = ()
    prune_unused: bool = True


@dataclass(frozen=True)
class MaskInputSpec:
    """Optional mask node binding for inpainting workflows."""

    node_id: str
    input_name: str = "image"
    remove_if_unused: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """Configuration mapping an OpenAI model name to a ComfyUI API workflow."""

    id: str
    workflow_path: Path
    source_workflow: str = ""
    aliases: tuple[str, ...] = ()
    description: str = ""
    output_node: str | None = None
    prompt_required: bool = True
    allow_no_image: bool = True
    default_width: int | None = None
    default_height: int | None = None
    size_multiple: int = 16
    max_n: int | None = None
    timeout: float | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    bindings: dict[str, tuple[str, ...]] = field(default_factory=dict)
    image_inputs: ImageInputSpec | None = None
    mask_input: MaskInputSpec | None = None
    remove_nodes: tuple[str, ...] = ()

    @property
    def max_images(self) -> int:
        if self.image_inputs is None:
            return 0
        if self.image_inputs.mode == "load_image_nodes":
            return len(self.image_inputs.nodes)
        return max(0, self.image_inputs.max_images)

    @property
    def capabilities(self) -> list[str]:
        caps: list[str] = []
        if self.allow_no_image:
            caps.append("text-to-image")
        if self.image_inputs is not None:
            caps.append("image-to-image")
        return caps


@dataclass(frozen=True)
class GatewayConfig:
    """Top-level gateway settings and model registry."""

    models: dict[str, ModelConfig] = field(default_factory=dict)
    alias_map: dict[str, str] = field(default_factory=dict)
    default_model: str | None = None
    fallback_to_default_model: bool = True
    expose_aliases: bool = False
    api_keys: tuple[str, ...] = ()
    cors_origins: str = "*"
    default_response_format: str = "b64_json"
    public_base_url: str | None = None
    timeout: float = 600.0
    max_n: int = 4
    max_image_bytes: int = 25 * 1024 * 1024
    allow_remote_image_urls: bool = True
    remote_url_timeout: float = 30.0
    keep_input_files: bool = False
    input_subfolder: str = "openai_gateway"
    config_path: Path | None = None

    def resolve_model(self, requested: str | None) -> ModelConfig:
        """Resolve a requested model name or alias to a configured ModelConfig."""
        if not self.models:
            raise GatewayError(
                "No workflows are configured in ComfyUI-OpenAI-Image-Gateway. "
                "Open ComfyUI Settings -> OpenAI Image Gateway to add a workflow, or create config.local.yaml.",
                status=503,
                error_type="server_error",
                code="gateway_not_configured",
            )

        name = (requested or "").strip()
        if not name:
            if self.default_model and self.default_model in self.models:
                return self.models[self.default_model]
            return next(iter(self.models.values()))

        if name in self.models:
            return self.models[name]

        canonical = self.alias_map.get(name)
        if canonical and canonical in self.models:
            return self.models[canonical]

        if self.fallback_to_default_model and self.default_model and self.default_model in self.models:
            return self.models[self.default_model]

        raise GatewayError(
            f"The model `{name}` does not exist.",
            status=404,
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
        )

    def listed_model_ids(self) -> list[str]:
        """Return model IDs to advertise on GET /v1/models."""
        seen: set[str] = set()
        ids: list[str] = []
        for model in self.models.values():
            if model.id not in seen:
                seen.add(model.id)
                ids.append(model.id)
            if self.expose_aliases:
                for alias in model.aliases:
                    if alias not in seen:
                        seen.add(alias)
                        ids.append(alias)
        return ids


@dataclass(frozen=True)
class DecodedImage:
    """Validated input image ready to be saved for ComfyUI."""

    data: bytes
    ext: str
    mime: str
    width: int
    height: int


@dataclass(frozen=True)
class RenderedImage:
    """Image produced by a ComfyUI execution."""

    data: bytes
    filename: str
    subfolder: str = ""
    file_type: str = "output"
    ext: str = "png"
    width: int = 0
    height: int = 0
