"""Tests for automatic ComfyUI API-workflow analysis."""

from __future__ import annotations

from gateway.analyzer import analyze_workflow


def test_analyze_qwen_2_1_edit_with_pe_and_image_compare():
    graph = {
        "461": {"class_type": "SaveImageAdvanced", "inputs": {"images": ["459:457", 0]}},
        "470": {"class_type": "LoadImage", "inputs": {"image": "ballet.png"}},
        "472": {"class_type": "ImageCompare", "inputs": {"image_a": ["470", 0], "image_b": ["461", 0]}},
        "459:456": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "459:468": {
            "class_type": "ComfySwitchNode",
            "inputs": {"switch": False, "on_false": ["459:474", 2], "on_true": ["459:456", 0]},
        },
        "459:474": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "prompt": ["459:480", 0],
                "negative_prompt": "",
                "resolution": 0,
                "images.image_1": ["470", 0],
            },
        },
        "459:480": {"class_type": "PreviewAny", "inputs": {"source": ["459:484", 0]}},
        "459:484": {
            "class_type": "ComfySwitchNode",
            "inputs": {"switch": False, "on_false": "original prompt", "on_true": ["459:500", 0]},
        },
        "459:485": {"class_type": "BatchImagesNode", "inputs": {"images.image0": ["470", 0]}},
        "459:500": {
            "class_type": "TextGenerate",
            "inputs": {"prompt": "original prompt", "sampling_mode.seed": 123, "image": ["459:485", 0]},
        },
        "459:504": {"class_type": "RandomNoise", "inputs": {"noise_seed": 123}},
        "459:507": {"class_type": "CFGGuider", "inputs": {"cfg": 1.0, "positive": ["459:474", 0]}},
    }

    info = analyze_workflow(graph)
    assert info["output_node"] == "461"
    assert set(info["remove_nodes"]) == {"470", "472"}
    assert info["allow_no_image"] is True
    assert info["size_multiple"] == 32
    assert info["max_images"] == 10
    assert info["image_inputs"]["mode"] == "dynamic_load_image"
    assert len(info["image_inputs"]["targets"]) == 2
    assert info["image_inputs"]["targets"][0]["node_id"] == "459:474"
    assert info["image_inputs"]["targets"][1]["node_id"] == "459:485"
    assert info["image_inputs"]["targets"][1]["prune_if_empty"] is True
    assert info["bindings"]["prompt"] == ["459:484.inputs.on_false", "459:500.inputs.prompt"]
    assert info["bindings"]["negative_prompt"] == "459:474.inputs.negative_prompt"
    assert info["bindings"]["seed"] == ["459:500.inputs.sampling_mode.seed", "459:504.inputs.noise_seed"]
    assert info["bindings"]["width"] == "459:456.inputs.width"
    assert info["bindings"]["height"] == "459:456.inputs.height"
    assert info["bindings"]["use_custom_size"] == "459:468.inputs.switch"
    assert info["bindings"]["cfg"] == "459:507.inputs.cfg"


def test_analyze_classic_sdxl_t2i_and_multi_load_image():
    t2i = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 1,
                "steps": 20,
                "cfg": 7.0,
                "denoise": 1.0,
                "positive": ["6", 0],
                "negative": ["7", 0],
            },
        },
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }
    info_t2i = analyze_workflow(t2i)
    assert info_t2i["output_node"] == "9"
    assert info_t2i["image_inputs"] is None
    assert info_t2i["allow_no_image"] is True
    assert info_t2i["capabilities"] == ["text-to-image"]
    assert info_t2i["bindings"]["prompt"] == "6.inputs.text"
    assert info_t2i["bindings"]["negative_prompt"] == "7.inputs.text"
    assert info_t2i["bindings"]["seed"] == "3.inputs.seed"
    assert info_t2i["bindings"]["steps"] == "3.inputs.steps"
    assert info_t2i["bindings"]["cfg"] == "3.inputs.cfg"
    assert info_t2i["bindings"]["denoise"] == "3.inputs.denoise"

    i2i = dict(t2i)
    i2i["10"] = {"class_type": "LoadImage", "inputs": {"image": "a.png"}}
    i2i["11"] = {"class_type": "LoadImage", "inputs": {"image": "b.png"}}
    i2i["12"] = {"class_type": "LoadImageMask", "inputs": {"image": "m.png"}}
    info_i2i = analyze_workflow(i2i)
    assert info_i2i["image_inputs"]["mode"] == "load_image_nodes"
    assert [n["node_id"] for n in info_i2i["image_inputs"]["nodes"]] == ["10", "11"]
    assert info_i2i["image_inputs"]["prune_unused"] is True
    assert info_i2i["mask_input"]["node_id"] == "12"
    assert info_i2i["allow_no_image"] is False
