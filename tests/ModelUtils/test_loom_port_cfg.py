###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the Loom loom-port-llm cfg.json architecture-facts module and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from TraceLens.ModelUtils.extract import parse_architecture
from TraceLens.ModelUtils.loom_port_cfg import (
    build_loom_arch_facts,
    coverage_notes,
    diff_against_existing,
)
from TraceLens.ModelUtils import loom_port_cfg_cli

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

_DENSE_GQA_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 8192,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
}


# --------------------------------------------------------------------------- #
# build_loom_arch_facts: core geometry
# --------------------------------------------------------------------------- #


def test_dense_gqa_model_derives_core_geometry_and_rope():
    spec = parse_architecture(_DENSE_GQA_CONFIG, "fixture")
    facts = build_loom_arch_facts(spec, name="meta-llama/Llama-3.1-8B")

    assert facts["name"] == "meta-llama/Llama-3.1-8B"
    assert facts["arch"] == "llama"
    assert facts["hidden"] == 2048
    assert facts["layers"] == 16
    assert facts["heads"] == 32
    assert facts["kv_heads"] == 8
    assert facts["head_dim"] == 2048 // 32
    assert facts["ffn"] == 8192
    assert facts["vocab"] == 128256
    assert facts["context_length"] == 131072
    assert facts["tied_embeddings"] is False
    assert facts["rms_eps"] == 1e-05
    assert facts["rope_theta"] == 500000.0
    # No layer_types/qk_norm/sliding declared -- nothing guessed for them.
    assert "sliding_layers" not in facts
    assert "qk_norm" not in facts
    assert "recurrent_layers" not in facts


def test_name_falls_back_to_spec_name_when_not_given():
    spec = parse_architecture(_DENSE_GQA_CONFIG, "fixture")
    facts = build_loom_arch_facts(spec)
    assert facts["name"] == spec.name


def test_rope_theta_defaults_when_rope_present_without_explicit_theta():
    config = dict(_DENSE_GQA_CONFIG)
    del config["rope_theta"]
    config["rope_scaling"] = {"type": "linear", "factor": 2.0}
    spec = parse_architecture(config, "fixture")
    facts = build_loom_arch_facts(spec)
    assert facts["rope_theta"] == 10000.0


def test_mqa_model_defaults_kv_heads_to_heads_when_absent():
    config = dict(_DENSE_GQA_CONFIG)
    del config["num_key_value_heads"]
    spec = parse_architecture(config, "fixture")
    facts = build_loom_arch_facts(spec)
    assert facts["kv_heads"] == facts["heads"] == 32


def test_head_dim_falls_back_to_hidden_over_heads_when_undeclared():
    spec = parse_architecture(_DENSE_GQA_CONFIG, "fixture")
    assert spec.head_dim is None  # config never states head_dim explicitly
    facts = build_loom_arch_facts(spec)
    assert facts["head_dim"] == 2048 // 32


# --------------------------------------------------------------------------- #
# qk_norm / partial rotary
# --------------------------------------------------------------------------- #


def test_qk_norm_flag_is_read_through_either_spelling():
    for key in ("qk_norm", "use_qk_norm"):
        config = dict(_DENSE_GQA_CONFIG)
        config[key] = True
        spec = parse_architecture(config, "fixture")
        facts = build_loom_arch_facts(spec)
        assert facts["qk_norm"] is True


def test_partial_rotary_factor_derives_rope_dims():
    config = dict(_DENSE_GQA_CONFIG)
    config["head_dim"] = 256
    config["partial_rotary_factor"] = 0.25
    spec = parse_architecture(config, "fixture")
    facts = build_loom_arch_facts(spec)
    assert facts["rope_dims"] == 64


def test_full_rotary_factor_omits_rope_dims():
    config = dict(_DENSE_GQA_CONFIG)
    config["partial_rotary_factor"] = 1.0
    spec = parse_architecture(config, "fixture")
    facts = build_loom_arch_facts(spec)
    assert "rope_dims" not in facts


# --------------------------------------------------------------------------- #
# sliding_layers / recurrent_layers from layer_types
# --------------------------------------------------------------------------- #


def test_layer_types_sliding_window_gives_explicit_per_layer_indices():
    config = dict(_DENSE_GQA_CONFIG)
    config["layer_types"] = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
    ]
    config["num_hidden_layers"] = 4
    config["sliding_window"] = 4096
    spec = parse_architecture(config, "fixture")
    facts = build_loom_arch_facts(spec)
    assert facts["sliding_layers"] == [0, 1, 3]
    assert facts["sliding_window"] == 4096


def test_layer_types_all_full_attention_reports_confident_empty_sliding_layers():
    config = dict(_DENSE_GQA_CONFIG)
    config["layer_types"] = ["full_attention", "full_attention"]
    config["num_hidden_layers"] = 2
    spec = parse_architecture(config, "fixture")
    facts = build_loom_arch_facts(spec)
    assert facts["sliding_layers"] == []


def test_max_window_layers_fallback_when_layer_types_absent():
    config = dict(_DENSE_GQA_CONFIG)
    config["num_hidden_layers"] = 10
    config["use_sliding_window"] = True
    config["max_window_layers"] = 6
    config["sliding_window"] = 2048
    spec = parse_architecture(config, "fixture")
    facts = build_loom_arch_facts(spec)
    assert facts["sliding_layers"] == list(range(6))
    assert facts["sliding_window"] == 2048


def test_recurrent_layers_from_non_attention_layer_types():
    config = dict(_DENSE_GQA_CONFIG)
    config["layer_types"] = [
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
    ]
    config["num_hidden_layers"] = 4
    spec = parse_architecture(config, "fixture")
    facts = build_loom_arch_facts(spec)
    assert facts["recurrent_layers"] == [0, 1, 3]
    assert any("recurrent_layers inferred" in note for note in coverage_notes(spec))


def test_sliding_only_layer_types_does_not_trigger_a_recurrent_layers_note():
    # Regression: decoder_type == "Hybrid" is extract.py's own loose label for
    # *any* layer_types list, including a pure sliding/full mix with no
    # recurrent layers at all -- that must not spuriously ask for gdn by hand.
    config = dict(_DENSE_GQA_CONFIG)
    config["layer_types"] = ["sliding_attention", "full_attention"]
    config["num_hidden_layers"] = 2
    spec = parse_architecture(config, "fixture")
    assert spec.decoder_type == "Hybrid"
    facts = build_loom_arch_facts(spec)
    assert "recurrent_layers" not in facts
    assert not any("recurrent_layers" in note for note in coverage_notes(spec))


# --------------------------------------------------------------------------- #
# coverage_notes: MoE / MLA / no_rope_layers
# --------------------------------------------------------------------------- #


def test_moe_model_gets_a_coverage_note():
    config = dict(_DENSE_GQA_CONFIG)
    config["num_experts"] = 8
    config["num_experts_per_tok"] = 2
    spec = parse_architecture(config, "fixture")
    assert spec.decoder_type == "Sparse MoE"
    notes = coverage_notes(spec)
    assert any("Sparse MoE" in note for note in notes)


def test_mla_model_gets_a_coverage_note():
    config = dict(_DENSE_GQA_CONFIG)
    config["kv_lora_rank"] = 512
    spec = parse_architecture(config, "fixture")
    assert spec.attention_type == "MLA"
    notes = coverage_notes(spec)
    assert any("Multi-head Latent Attention" in note for note in notes)


def test_no_rope_layers_key_flags_a_manual_review_note_without_guessing():
    config = dict(_DENSE_GQA_CONFIG)
    config["no_rope_layers"] = [1, 1, 0, 1]
    spec = parse_architecture(config, "fixture")
    facts = build_loom_arch_facts(spec)
    assert "nope_layers" not in facts
    notes = coverage_notes(spec)
    assert any("no_rope_layers" in note for note in notes)


# --------------------------------------------------------------------------- #
# diff_against_existing
# --------------------------------------------------------------------------- #


def test_diff_flags_a_value_mismatch():
    derived = {"hidden": 2048, "heads": 32}
    existing = {"hidden": 4096, "heads": 32}
    lines = diff_against_existing(derived, existing)
    assert len(lines) == 1
    assert "hidden" in lines[0]


def test_diff_flags_a_missing_key_as_an_addition():
    derived = {"hidden": 2048, "qk_norm": True}
    existing = {"hidden": 2048}
    lines = diff_against_existing(derived, existing)
    assert len(lines) == 1
    assert "qk_norm" in lines[0]


def test_diff_compares_lists_order_independently():
    derived = {"sliding_layers": [0, 1, 3]}
    existing = {"sliding_layers": [3, 1, 0]}
    assert diff_against_existing(derived, existing) == []


def test_diff_never_flags_a_field_tracelens_does_not_derive():
    derived = {"hidden": 2048}
    existing = {"hidden": 2048, "attn_gate": "fused_q", "rope_type": "neox"}
    assert diff_against_existing(derived, existing) == []


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_end_to_end_writes_json(tmp_path, capsys):
    output_path = tmp_path / "cfg.json"
    exit_code = loom_port_cfg_cli.main(
        [
            str(FIXTURES / "custom_model"),
            "--no-meta-shapes",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Wrote Loom architecture facts" in captured.out
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["hidden"] == 4096
    assert payload["layers"] == 32
    assert payload["kv_heads"] == 8
    # custom_model is MLA + Sparse MoE -- both notes should surface on stderr.
    assert "Multi-head Latent Attention" in captured.err
    assert "Sparse MoE" in captured.err


def test_cli_prints_json_to_stdout_by_default(capsys):
    exit_code = loom_port_cfg_cli.main(
        [str(FIXTURES / "custom_model"), "--no-meta-shapes"]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["arch"] == "custom_mla_moe"


def test_cli_requires_a_checkpoint():
    with pytest.raises(SystemExit):
        loom_port_cfg_cli.main([])


def test_cli_reports_an_architecture_load_failure_without_a_traceback(capsys):
    exit_code = loom_port_cfg_cli.main(
        ["/nonexistent/path/for/testing", "--no-meta-shapes"]
    )
    assert exit_code == 1
    assert "Error loading architecture" in capsys.readouterr().err


def test_cli_check_mode_reports_no_mismatches_against_its_own_output(tmp_path):
    output_path = tmp_path / "cfg.json"
    assert (
        loom_port_cfg_cli.main(
            [
                str(FIXTURES / "custom_model"),
                "--no-meta-shapes",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    exit_code = loom_port_cfg_cli.main(
        [
            str(FIXTURES / "custom_model"),
            "--no-meta-shapes",
            "--check",
            str(output_path),
        ]
    )
    assert exit_code == 0


def test_cli_check_mode_flags_a_mismatch(tmp_path, capsys):
    existing_path = tmp_path / "existing_cfg.json"
    existing_path.write_text(json.dumps({"hidden": 1, "heads": 32}), encoding="utf-8")
    exit_code = loom_port_cfg_cli.main(
        [
            str(FIXTURES / "custom_model"),
            "--no-meta-shapes",
            "--check",
            str(existing_path),
        ]
    )
    assert exit_code == 1
    assert "hidden" in capsys.readouterr().out
