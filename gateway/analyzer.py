"""Automatic ComfyUI API-workflow analyzer for zero-config model registration."""

from __future__ import annotations

from typing import Any

_SAVE_NODE_PRIORITY = (
    "SaveImageAdvanced",
    "SaveImage",
    "SaveImageWebsocket",
    "PreviewImage",
)
_QWEN_EDIT_CLASSES = {
    "TextEncodeQwenImage21",
    "TextEncodeQwenImageEdit",
    "TextEncodeQwenImageEditPlus",
}
_LATENT_SIZE_CLASSES = {
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyFlux2LatentImage",
    "EmptyHunyuanLatentVideo",
}
_POSITIVE_PROMPT_CLASSES = {
    "CLIPTextEncode",
    "CLIPTextEncodeFlux",
    "CLIPTextEncodeSDXL",
    "CLIPTextEncodeSD3",
    "TextEncodeQwenImage21",
    "TextEncodeQwenImageEdit",
    "TextEncodeQwenImageEditPlus",
}
_STRING_PASSTHROUGH_CLASSES = {
    "PreviewAny",
    "ShowText",
    "ShowText|pysssss",
    "StringFunction",
}
_STRING_PRIMITIVE_CLASSES = {
    "PrimitiveString",
    "PrimitiveStringMultiline",
    "String",
    "Text",
    "Text Multiline",
}
_TEXT_GEN_CLASSES = {
    "TextGenerate",
    "TextGenerateLTX2",
}


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)


def _node_inputs(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    node = graph.get(node_id)
    if not isinstance(node, dict):
        return {}
    inputs = node.get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def _node_class(graph: dict[str, Any], node_id: str) -> str:
    node = graph.get(node_id)
    return str(node.get("class_type", "")) if isinstance(node, dict) else ""


def _sort_node_ids(ids: list[str]) -> list[str]:
    def _key(nid: str) -> tuple[int, str]:
        tail = nid.rsplit(":", 1)[-1]
        return (int(tail), nid) if tail.isdigit() else (10**9, nid)

    return sorted(ids, key=_key)


def _trace_string_bindings(
    graph: dict[str, Any],
    node_id: str,
    field: str,
    visited: set[tuple[str, str]] | None = None,
) -> list[str]:
    """Trace an input field upstream through switches/previews/TextGenerate to editable string slots."""
    if visited is None:
        visited = set()
    key = (node_id, field)
    if key in visited or node_id not in graph:
        return []
    visited.add(key)

    inputs = _node_inputs(graph, node_id)
    val = inputs.get(field)
    if isinstance(val, str):
        return [f"{node_id}.inputs.{field}"]
    if not _is_link(val):
        return []

    src_id = str(val[0])
    src_class = _node_class(graph, src_id)
    src_inputs = _node_inputs(graph, src_id)

    if src_class in _STRING_PASSTHROUGH_CLASSES:
        for cand in ("source", "text", "string", "value"):
            if cand in src_inputs:
                return _trace_string_bindings(graph, src_id, cand, visited)
        return []

    if src_class == "ComfySwitchNode":
        found: list[str] = []
        for branch in ("on_false", "on_true"):
            if branch in src_inputs:
                found.extend(_trace_string_bindings(graph, src_id, branch, visited))
        return found

    if src_class in _TEXT_GEN_CLASSES:
        if "prompt" in src_inputs:
            return _trace_string_bindings(graph, src_id, "prompt", visited)
        return []

    if src_class in _STRING_PRIMITIVE_CLASSES:
        for cand in ("value", "string", "text"):
            if cand in src_inputs:
                return _trace_string_bindings(graph, src_id, cand, visited)
        return []

    return []


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _detect_output_node(graph: dict[str, Any]) -> str | None:
    for cls_name in _SAVE_NODE_PRIORITY:
        matches = _sort_node_ids([nid for nid in graph if _node_class(graph, nid) == cls_name])
        if matches:
            return matches[-1]
    return None


def _detect_image_inputs(
    graph: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str], bool, int]:
    """Return (image_inputs_dict, remove_nodes, allow_no_image, size_multiple)."""
    remove_nodes: list[str] = [
        nid for nid in _sort_node_ids(list(graph)) if _node_class(graph, nid) == "ImageCompare"
    ]

    qwen_nodes = _sort_node_ids([nid for nid in graph if _node_class(graph, nid) in _QWEN_EDIT_CLASSES])
    if qwen_nodes:
        primary = qwen_nodes[0]
        primary_class = _node_class(graph, primary)
        prefix = "images.image_" if primary_class == "TextEncodeQwenImage21" else "image"
        targets: list[dict[str, Any]] = [
            {
                "node_id": primary,
                "input_prefix": prefix,
                "start_index": 1,
                "clear_existing_prefix": True,
                "prune_if_empty": False,
            }
        ]

        # Wire BatchImagesNode too when present (e.g. Qwen 2.1 Prompt Enhancer subgraph).
        batch_nodes = _sort_node_ids([nid for nid in graph if _node_class(graph, nid) == "BatchImagesNode"])
        for bnode in batch_nodes:
            targets.append(
                {
                    "node_id": bnode,
                    "input_prefix": "images.image",
                    "start_index": 0,
                    "clear_existing_prefix": True,
                    "prune_if_empty": True,
                }
            )

        # Remove template LoadImage nodes that feed these dynamic image targets.
        target_ids = {t["node_id"] for t in targets}
        for tid in target_ids:
            for val in _node_inputs(graph, tid).values():
                if _is_link(val):
                    src_id = str(val[0])
                    if _node_class(graph, src_id) == "LoadImage":
                        remove_nodes.append(src_id)

        image_inputs = {
            "mode": "dynamic_load_image",
            "max_images": 10 if primary_class == "TextEncodeQwenImage21" else 4,
            "targets": targets,
        }
        return image_inputs, _dedupe(remove_nodes), True, 32

    load_image_nodes = _sort_node_ids([nid for nid in graph if _node_class(graph, nid) == "LoadImage"])
    if load_image_nodes:
        image_inputs = {
            "mode": "load_image_nodes",
            "prune_unused": len(load_image_nodes) > 1,
            "nodes": [{"node_id": nid, "input": "image"} for nid in load_image_nodes],
        }
        return image_inputs, _dedupe(remove_nodes), False, 16

    return None, _dedupe(remove_nodes), True, 16


def _detect_mask_input(graph: dict[str, Any]) -> dict[str, Any] | None:
    mask_nodes = _sort_node_ids([nid for nid in graph if _node_class(graph, nid) == "LoadImageMask"])
    if mask_nodes:
        return {"node_id": mask_nodes[0], "input": "image", "remove_if_unused": False}
    return None


def _detect_prompt_bindings(graph: dict[str, Any]) -> tuple[list[str], list[str]]:
    positive: list[str] = []
    negative: list[str] = []

    # Priority 1: Qwen unified text encoders (contain both prompt and negative_prompt).
    qwen_nodes = _sort_node_ids([nid for nid in graph if _node_class(graph, nid) in _QWEN_EDIT_CLASSES])
    if qwen_nodes:
        qnode = qwen_nodes[0]
        positive.extend(_trace_string_bindings(graph, qnode, "prompt"))
        negative.extend(_trace_string_bindings(graph, qnode, "negative_prompt"))
        return _dedupe(positive), _dedupe(negative)

    # Priority 2: Trace positive/negative conditioning inputs from samplers or guiders.
    sampler_classes = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "CFGGuider", "BasicGuider"}
    pos_cond_nodes: list[str] = []
    neg_cond_nodes: list[str] = []

    def _collect_text_encodes(start_id: str, out: list[str], seen: set[str]) -> None:
        if start_id in seen or start_id not in graph:
            return
        seen.add(start_id)
        cls = _node_class(graph, start_id)
        if cls in _POSITIVE_PROMPT_CLASSES:
            out.append(start_id)
            return
        for k, v in _node_inputs(graph, start_id).items():
            if _is_link(v) and k in ("conditioning", "positive", "negative", "cond"):
                _collect_text_encodes(str(v[0]), out, seen)

    for nid in _sort_node_ids(list(graph)):
        if _node_class(graph, nid) not in sampler_classes:
            continue
        inputs = _node_inputs(graph, nid)
        for pkey in ("positive", "conditioning"):
            if _is_link(inputs.get(pkey)):
                _collect_text_encodes(str(inputs[pkey][0]), pos_cond_nodes, set())
        if _is_link(inputs.get("negative")):
            _collect_text_encodes(str(inputs["negative"][0]), neg_cond_nodes, set())

    for nid in _dedupe(pos_cond_nodes):
        for field in ("text", "clip_l", "t5xxl", "prompt"):
            if field in _node_inputs(graph, nid):
                positive.extend(_trace_string_bindings(graph, nid, field))
    for nid in _dedupe(neg_cond_nodes):
        if nid in pos_cond_nodes:
            continue
        for field in ("text", "clip_l", "t5xxl", "prompt"):
            if field in _node_inputs(graph, nid):
                negative.extend(_trace_string_bindings(graph, nid, field))

    # Fallback: standalone CLIPTextEncode nodes if sampler tracing didn't resolve them.
    if not positive:
        enc_nodes = _sort_node_ids([nid for nid in graph if _node_class(graph, nid) in _POSITIVE_PROMPT_CLASSES])
        if enc_nodes:
            for field in ("text", "prompt"):
                if field in _node_inputs(graph, enc_nodes[0]):
                    positive.extend(_trace_string_bindings(graph, enc_nodes[0], field))
                    break
            if len(enc_nodes) > 1:
                for field in ("text", "prompt"):
                    if field in _node_inputs(graph, enc_nodes[1]):
                        negative.extend(_trace_string_bindings(graph, enc_nodes[1], field))
                        break

    return _dedupe(positive), _dedupe(negative)


def _detect_seed_bindings(graph: dict[str, Any]) -> list[str]:
    seeds: list[str] = []
    for nid in _sort_node_ids(list(graph)):
        cls = _node_class(graph, nid)
        inputs = _node_inputs(graph, nid)
        if cls in ("RandomNoise", "KSamplerAdvanced", "SamplerCustom") and isinstance(
            inputs.get("noise_seed"), (int, float)
        ):
            seeds.append(f"{nid}.inputs.noise_seed")
        elif cls == "KSampler" and isinstance(inputs.get("seed"), (int, float)):
            seeds.append(f"{nid}.inputs.seed")
        elif cls in _TEXT_GEN_CLASSES:
            if isinstance(inputs.get("sampling_mode.seed"), (int, float)):
                seeds.append(f"{nid}.inputs.sampling_mode.seed")
            elif isinstance(inputs.get("seed"), (int, float)):
                seeds.append(f"{nid}.inputs.seed")
    return _dedupe(seeds)


def _detect_size_and_sampler_bindings(graph: dict[str, Any]) -> dict[str, str]:
    bindings: dict[str, str] = {}

    latent_nodes = _sort_node_ids([nid for nid in graph if _node_class(graph, nid) in _LATENT_SIZE_CLASSES])
    if latent_nodes:
        lnode = latent_nodes[0]
        linputs = _node_inputs(graph, lnode)
        if isinstance(linputs.get("width"), (int, float)) and isinstance(linputs.get("height"), (int, float)):
            bindings["width"] = f"{lnode}.inputs.width"
            bindings["height"] = f"{lnode}.inputs.height"

        # Detect Qwen 2.1 auto-size vs custom-size ComfySwitchNode.
        for nid in _sort_node_ids(list(graph)):
            if _node_class(graph, nid) != "ComfySwitchNode":
                continue
            sinputs = _node_inputs(graph, nid)
            on_true = sinputs.get("on_true")
            on_false = sinputs.get("on_false")
            if (
                _is_link(on_true)
                and str(on_true[0]) == lnode
                and _is_link(on_false)
                and _node_class(graph, str(on_false[0])) in _QWEN_EDIT_CLASSES
            ):
                bindings["use_custom_size"] = f"{nid}.inputs.switch"
                break

    for nid in _sort_node_ids(list(graph)):
        cls = _node_class(graph, nid)
        inputs = _node_inputs(graph, nid)
        if "cfg" not in bindings and cls in ("CFGGuider", "KSampler", "KSamplerAdvanced", "SamplerCustom"):
            if isinstance(inputs.get("cfg"), (int, float)):
                bindings["cfg"] = f"{nid}.inputs.cfg"
        if "steps" not in bindings and cls in ("KSampler", "KSamplerAdvanced", "BasicScheduler"):
            if isinstance(inputs.get("steps"), (int, float)):
                bindings["steps"] = f"{nid}.inputs.steps"
        if "denoise" not in bindings and cls in ("KSampler", "BasicScheduler"):
            if isinstance(inputs.get("denoise"), (int, float)):
                bindings["denoise"] = f"{nid}.inputs.denoise"

    return bindings


def analyze_workflow(graph: dict[str, Any]) -> dict[str, Any]:
    """Inspect a ComfyUI API-format workflow and infer gateway bindings and capabilities."""
    output_node = _detect_output_node(graph)
    image_inputs, remove_nodes, allow_no_image, size_multiple = _detect_image_inputs(graph)
    mask_input = _detect_mask_input(graph)
    pos_prompts, neg_prompts = _detect_prompt_bindings(graph)
    seeds = _detect_seed_bindings(graph)
    extra_bindings = _detect_size_and_sampler_bindings(graph)

    bindings: dict[str, Any] = {}
    if pos_prompts:
        bindings["prompt"] = pos_prompts[0] if len(pos_prompts) == 1 else pos_prompts
    if neg_prompts:
        bindings["negative_prompt"] = neg_prompts[0] if len(neg_prompts) == 1 else neg_prompts
    if seeds:
        bindings["seed"] = seeds[0] if len(seeds) == 1 else seeds
    bindings.update(extra_bindings)

    capabilities: list[str] = []
    if allow_no_image:
        capabilities.append("text-to-image")
    if image_inputs is not None:
        capabilities.append("image-to-image")

    max_images = 0
    if image_inputs is not None:
        if image_inputs["mode"] == "load_image_nodes":
            max_images = len(image_inputs["nodes"])
        else:
            max_images = int(image_inputs.get("max_images", 10))

    return {
        "output_node": output_node,
        "remove_nodes": remove_nodes,
        "image_inputs": image_inputs,
        "mask_input": mask_input,
        "bindings": bindings,
        "prompt_required": bool(pos_prompts),
        "allow_no_image": allow_no_image,
        "size_multiple": size_multiple,
        "capabilities": capabilities,
        "max_images": max_images,
        "node_count": len(graph),
    }
