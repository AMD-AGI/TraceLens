"""Tests for GLM-5.3-Flash modeling source resolution."""

from __future__ import annotations

import pytest

from visualizer.config_resolve import normalize_config
from visualizer.loader import load_model_spec
from visualizer.source import _config_model_types, resolve_source_files


def test_config_model_types_includes_wrapper_after_normalization():
    raw = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "model_type": "glm5_next_text",
            "hidden_size": 4096,
            "num_hidden_layers": 45,
        },
    }
    normalized = normalize_config(raw, source_label="test")
    assert normalized["model_type"] == "glm5_next_text"
    assert _config_model_types(normalized) == ["glm5_next", "glm5_next_text"]


def test_glm53_flash_resolves_transformers_modeling_source():
    pytest.importorskip("huggingface_hub")
    from visualizer.extract import _resolve_checkpoint
    from visualizer.render import _ffn_label

    config, _ = _resolve_checkpoint(checkpoint="zai-org/GLM-5.3-Flash", github=None)
    files, labels = resolve_source_files("zai-org/GLM-5.3-Flash", config)
    assert files
    assert any("modeling_glm5_next.py" in str(path) for path in files)
    assert any(label.startswith("github://huggingface/transformers") for label in labels)

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    assert _ffn_label(spec)[0] == "Glm5NextTextMoE / Glm5NextTextMLP"
