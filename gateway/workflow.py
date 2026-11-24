"""Workflow template loading, validation, and dynamic multi-image graph patching."""

from __future__ import annotations

import copy
import json
import secrets
from pathlib import Path
from typing import Any

from .models import GatewayError, ImageInputSpec, ModelConfig

_MAX_SAFE_SEED = (1 << 53) - 1


def validate_workflow_dict(data: Any, *, label: str = "workflow") -> dict[str, Any]:
    """Validate an in-memory ComfyUI API-format workflow JSON dict."""
    if not isinstance(data, dict):
        raise ValueError(f"Workflow `{label}` must contain a JSON object")

    if isinstance(data.get("nodes"), list) and isinstance(data.get("links"), list):
        raise ValueError(
            f"Workflow `{label}` is in ComfyUI UI format, not API format. "
            "In ComfyUI, use Workflow -> Export (API) or add it from Settings -> OpenAI Image Gateway."
        )

    if not data:
        raise ValueError(f"Workflow `{label}` is empty")

    for node_id, node in data.items():
        if not isinstance(node, dict) or "class_type" not in node:
            raise ValueError(
                f"Workflow `{label}` has invalid node `{node_id}`: expected an object with `class_type`."
            )
    return data


def load_workflow_template(path: Path) -> dict[str, Any]:
    """Load a ComfyUI API-format workflow JSON file and validate its top-level shape."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return validate_workflow_dict(data, label=str(path))


def remove_nodes(graph: dict[str, Any], node_ids: set[str]) -> int:
    """Remove nodes from a prompt graph and prune dangling link references to them."""
    removed = 0
    for node_id in list(node_ids):
        if node_id in graph:
            del graph[node_id]
            removed += 1

    if not removed:
        return 0

    for node in graph.values():
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        to_delete = [
            key
            for key, val in inputs.items()
            if isinstance(val, list) and len(val) == 2 and str(val[0]) in node_ids
        ]
        for key in to_delete:
            del inputs[key]
    return removed


def _next_synthetic_node_id(graph: dict[str, Any], index: int) -> str:
    candidate = f"__gw_load_image_{index}"
    suffix = 0
    while candidate in graph:
        suffix += 1
        candidate = f"__gw_load_image_{index}_{suffix}"
    return candidate


def _wire_dynamic_load_images(
    graph: dict[str, Any],
    spec: ImageInputSpec,
    image_filenames: list[str],
) -> None:
    nodes_to_prune: set[str] = set()
    for target in spec.targets:
        node = graph.get(target.node_id)
        if node is None:
            raise GatewayError(
                f"Configured image target node `{target.node_id}` was not found in workflow.",
                status=500,
                error_type="server_error",
                code="invalid_workflow_config",
            )
        inputs = node.setdefault("inputs", {})
        if target.clear_existing_prefix:
            for key in [k for k in inputs if k.startswith(target.input_prefix)]:
                del inputs[key]
        if not image_filenames and target.prune_if_empty:
            nodes_to_prune.add(target.node_id)

    if nodes_to_prune:
        remove_nodes(graph, nodes_to_prune)

    if not image_filenames:
        return

    synth_ids: list[str] = []
    for idx, filename in enumerate(image_filenames, start=1):
        synth_id = _next_synthetic_node_id(graph, idx)
        graph[synth_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": filename},
        }
        synth_ids.append(synth_id)

    for target in spec.targets:
        if target.node_id not in graph:
            continue
        inputs = graph[target.node_id].setdefault("inputs", {})
        for offset, synth_id in enumerate(synth_ids):
            key = f"{target.input_prefix}{target.start_index + offset}"
            inputs[key] = [synth_id, 0]


def _wire_fixed_load_image_nodes(
    graph: dict[str, Any],
    spec: ImageInputSpec,
    image_filenames: list[str],
) -> None:
    unused_nodes: set[str] = set()
    for idx, slot in enumerate(spec.nodes):
        if idx < len(image_filenames):
            node = graph.get(slot.node_id)
            if node is None:
                raise GatewayError(
                    f"Configured LoadImage node `{slot.node_id}` was not found in workflow.",
                    status=500,
                    error_type="server_error",
                    code="invalid_workflow_config",
                )
            node.setdefault("inputs", {})[slot.input_name] = image_filenames[idx]
        elif spec.prune_unused:
            unused_nodes.add(slot.node_id)

    if unused_nodes:
        remove_nodes(graph, unused_nodes)


def resolve_seed(requested_seed: int | None, run_index: int = 0) -> int:
    """Resolve a positive JS/ComfyUI-safe seed for a single generation pass."""
    if requested_seed is None or requested_seed < 0:
        return secrets.randbelow(_MAX_SAFE_SEED) + 1
    return (int(requested_seed) + run_index) % _MAX_SAFE_SEED


def _apply_binding(graph: dict[str, Any], path: str, value: Any) -> None:
    node_id, section, field = path.split(".", 2)
    node = graph.get(node_id)
    if node is None:
        return
    target = node.setdefault(section, {})
    if isinstance(target, dict):
        target[field] = value


def build_prompt_graph(
    model: ModelConfig,
    template: dict[str, Any],
    *,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    image_filenames: list[str] | None = None,
    mask_filename: str | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    denoise: float | None = None,
    filename_prefix: str | None = None,
    run_index: int = 0,
) -> tuple[dict[str, Any], int]:
    """Patch a workflow template with request parameters and input images."""
    graph = copy.deepcopy(template)
    images = list(image_filenames or [])

    if images and model.image_inputs is None:
        raise GatewayError(
            f"Model `{model.id}` does not accept input images.",
            status=400,
            param="image",
            code="images_not_supported",
        )
    if images and len(images) > model.max_images:
        raise GatewayError(
            f"Model `{model.id}` accepts at most {model.max_images} input image(s), got {len(images)}.",
            status=400,
            param="image",
            code="too_many_images",
        )
    if not images and not model.allow_no_image:
        raise GatewayError(
            f"Model `{model.id}` requires at least one input image.",
            status=400,
            param="image",
            code="image_required",
        )

    if model.remove_nodes:
        remove_nodes(graph, set(model.remove_nodes))

    if model.image_inputs is not None:
        if model.image_inputs.mode == "dynamic_load_image":
            _wire_dynamic_load_images(graph, model.image_inputs, images)
        else:
            _wire_fixed_load_image_nodes(graph, model.image_inputs, images)

    if model.mask_input is not None:
        if mask_filename is not None:
            node = graph.get(model.mask_input.node_id)
            if node is None:
                raise GatewayError(
                    f"Configured mask node `{model.mask_input.node_id}` was not found in workflow.",
                    status=500,
                    error_type="server_error",
                    code="invalid_workflow_config",
                )
            node.setdefault("inputs", {})[model.mask_input.input_name] = mask_filename
        elif model.mask_input.remove_if_unused:
            remove_nodes(graph, {model.mask_input.node_id})

    used_seed = resolve_seed(seed, run_index=run_index)
    explicit_size = width is not None and height is not None
    eff_width = width if width is not None else model.default_width
    eff_height = height if height is not None else model.default_height

    values: dict[str, Any] = dict(model.defaults)
    updates = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": eff_width,
        "height": eff_height,
        "seed": used_seed,
        "steps": steps,
        "cfg": cfg,
        "denoise": denoise,
        "filename_prefix": filename_prefix,
    }
    for key, val in updates.items():
        if val is not None:
            values[key] = val

    if "use_custom_size" in model.bindings:
        values["use_custom_size"] = explicit_size or (len(images) == 0)

    for key, paths in model.bindings.items():
        if key not in values or values[key] is None:
            continue
        for path in paths:
            _apply_binding(graph, path, values[key])

    return graph, used_seed
