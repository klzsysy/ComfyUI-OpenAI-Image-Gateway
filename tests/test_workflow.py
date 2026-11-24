"""Tests for workflow template validation and multi-image graph patching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.config import load_config
from gateway.models import (
    GatewayError,
    ImageInputSpec,
    ImageTargetSpec,
    LoadImageNodeSlot,
    ModelConfig,
)
from gateway.workflow import build_prompt_graph, load_workflow_template

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def test_rejects_ui_format_workflow(tmp_path: Path):
    ui_wf = tmp_path / "ui.json"
    ui_wf.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="ComfyUI UI format"):
        load_workflow_template(ui_wf)


def test_dynamic_multi_image_wiring_and_bindings():
    cfg = load_config(PLUGIN_ROOT, explicit_path=PLUGIN_ROOT / "config.example.yaml", env={})
    model = cfg.resolve_model("gpt-image-1")
    template = load_workflow_template(model.workflow_path)

    graph, seed = build_prompt_graph(
        model,
        template,
        prompt="Put the blue jacket from image 2 on the person in image 1",
        image_filenames=["openai_gateway/a.png", "openai_gateway/b.png", "openai_gateway/c.png"],
        width=None,
        height=None,
        seed=42,
    )

    assert seed == 42
    assert graph["459:474"]["inputs"]["prompt"] == "Put the blue jacket from image 2 on the person in image 1"
    assert graph["459:504"]["inputs"]["noise_seed"] == 42
    # Auto size with reference images -> switch=False (follow first reference image)
    assert graph["459:468"]["inputs"]["switch"] is False

    enc_inputs = graph["459:474"]["inputs"]
    assert enc_inputs["images.image_1"] == ["__gw_load_image_1", 0]
    assert enc_inputs["images.image_2"] == ["__gw_load_image_2", 0]
    assert enc_inputs["images.image_3"] == ["__gw_load_image_3", 0]
    assert graph["__gw_load_image_1"]["class_type"] == "LoadImage"
    assert graph["__gw_load_image_1"]["inputs"]["image"] == "openai_gateway/a.png"
    assert graph["__gw_load_image_3"]["inputs"]["image"] == "openai_gateway/c.png"


def test_dynamic_explicit_size_and_zero_images_t2i():
    cfg = load_config(PLUGIN_ROOT, explicit_path=PLUGIN_ROOT / "config.example.yaml", env={})
    model = cfg.resolve_model("qwen-image-2.1-turbo-edit")
    template = load_workflow_template(model.workflow_path)

    # Zero images -> use_custom_size=True, no synthetic LoadImage nodes
    graph_t2i, _ = build_prompt_graph(
        model,
        template,
        prompt="A cinematic lighthouse at dusk",
        image_filenames=[],
        width=1024,
        height=1536,
    )
    assert graph_t2i["459:468"]["inputs"]["switch"] is True
    assert graph_t2i["459:456"]["inputs"]["width"] == 1024
    assert graph_t2i["459:456"]["inputs"]["height"] == 1536
    assert not any(k.startswith("__gw_load_image_") for k in graph_t2i)
    assert not any(k.startswith("images.image_") for k in graph_t2i["459:474"]["inputs"])


def test_dynamic_prunes_empty_batch_node_and_supports_dotted_field_names(tmp_path: Path):
    template = {
        "10": {"class_type": "TextEncodeQwenImage21", "inputs": {"images.image_1": ["99", 0]}},
        "11": {"class_type": "BatchImagesNode", "inputs": {"images.image0": ["99", 0]}},
        "12": {
            "class_type": "TextGenerate",
            "inputs": {"prompt": "", "sampling_mode.seed": 1, "image": ["11", 0]},
        },
        "99": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
        "100": {"class_type": "ImageCompare", "inputs": {"image_a": ["99", 0]}},
    }
    model = ModelConfig(
        id="edit-pe",
        workflow_path=tmp_path / "dummy.json",
        remove_nodes=("99", "100"),
        image_inputs=ImageInputSpec(
            mode="dynamic_load_image",
            max_images=4,
            targets=(
                ImageTargetSpec(node_id="10", input_prefix="images.image_", start_index=1),
                ImageTargetSpec(node_id="11", input_prefix="images.image", start_index=0, prune_if_empty=True),
            ),
        ),
        bindings={
            "prompt": ("12.inputs.prompt",),
            "seed": ("12.inputs.sampling_mode.seed",),
        },
    )

    # With 2 images -> both 10 and 11 wired, 99 and 100 removed
    g2, _ = build_prompt_graph(model, template, prompt="edit", image_filenames=["a.png", "b.png"], seed=77)
    assert "99" not in g2 and "100" not in g2
    assert g2["10"]["inputs"]["images.image_1"] == ["__gw_load_image_1", 0]
    assert g2["10"]["inputs"]["images.image_2"] == ["__gw_load_image_2", 0]
    assert g2["11"]["inputs"]["images.image0"] == ["__gw_load_image_1", 0]
    assert g2["11"]["inputs"]["images.image1"] == ["__gw_load_image_2", 0]
    assert g2["12"]["inputs"]["sampling_mode.seed"] == 77

    # With 0 images -> node 11 pruned and link in 12.inputs.image removed
    g0, _ = build_prompt_graph(model, template, prompt="t2i", image_filenames=[], seed=88)
    assert "11" not in g0
    assert "image" not in g0["12"]["inputs"]
    assert g0["12"]["inputs"]["sampling_mode.seed"] == 88


def test_fixed_load_image_nodes_and_pruning(tmp_path: Path):
    template = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "placeholder1.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "placeholder2.png"}},
        "3": {
            "class_type": "MultiRefNode",
            "inputs": {"img1": ["1", 0], "img2": ["2", 0], "text": ""},
        },
    }
    model = ModelConfig(
        id="fixed-edit",
        workflow_path=tmp_path / "dummy.json",
        allow_no_image=False,
        image_inputs=ImageInputSpec(
            mode="load_image_nodes",
            max_images=2,
            nodes=(LoadImageNodeSlot("1"), LoadImageNodeSlot("2")),
            prune_unused=True,
        ),
        bindings={"prompt": ("3.inputs.text",)},
    )

    g1, _ = build_prompt_graph(model, template, prompt="restyle", image_filenames=["x.png"])
    assert g1["1"]["inputs"]["image"] == "x.png"
    assert "2" not in g1
    assert "img2" not in g1["3"]["inputs"]
    assert g1["3"]["inputs"]["img1"] == ["1", 0]

    with pytest.raises(GatewayError) as exc_info:
        build_prompt_graph(model, template, prompt="restyle", image_filenames=["a.png", "b.png", "c.png"])
    assert exc_info.value.code == "too_many_images"

    with pytest.raises(GatewayError) as exc_info:
        build_prompt_graph(model, template, prompt="restyle", image_filenames=[])
    assert exc_info.value.code == "image_required"
