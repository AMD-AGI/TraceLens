###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for modular Hugging Face config discovery."""

from TraceLens.ModelUtils.config_resolve import (
    discover_remote_config,
    normalize_config,
    _score_config_content,
)


def test_normalize_qwen3_vl_text_config():
    raw = {
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
        },
    }
    normalized = normalize_config(raw, source_label="test")
    assert normalized["hidden_size"] == 5120
    assert normalized["num_hidden_layers"] == 64
    assert normalized["_wrapper_model_type"] == "qwen3_vl"


def test_score_prefers_text_encoder():
    text_cfg = {"model_type": "qwen3_vl", "text_config": {"num_hidden_layers": 64}}
    vae_cfg = {"_class_name": "AutoencoderKLMiniMaxH3Audio"}
    assert _score_config_content(
        text_cfg, "FL2VA/text_encoder/config.json"
    ) > _score_config_content(vae_cfg, "FL2VA/audio_vae/config.json")


def test_discover_minimax_h3_config():
    config, label = discover_remote_config("MiniMaxAI/MiniMax-H3")
    assert "text_encoder/config.json" in label
    assert config["num_hidden_layers"] == 64
    assert config["num_key_value_heads"] == 8
