###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Coverage for architecture extraction: config inference, AST merge, exports."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import TraceLens.ModelUtils.extract as extract
from TraceLens.ModelUtils.ast_analyze import ClassStructure, analyze_sources
from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.blocks import BlockComponent, CodeAnalysis
from TraceLens.ModelUtils.extract import (
    ArchitectureSpec,
    _apply_uniform_ffn_component,
    _as_bool,
    _build_export_block_trees,
    _build_highlights,
    _code_rotates_positions,
    _default_attention_class,
    _estimate_kv_cache,
    _estimate_param_hint,
    _finalize_layer_repeat_lines,
    _first,
    _human_bytes,
    _infer_layer_variants,
    _merge_code_analysis,
    _rebuild_stack_components,
    _refine_positional_from_code,
    _resolve_attention_from_config_lists,
    _resolve_checkpoint,
    _resolve_ffn_for_layer,
    dump_model_ast,
    find_vision_tower,
    load_architecture,
    parse_architecture,
    vision_tower_component,
)


def _structure(name: str, **kwargs) -> ClassStructure:
    return ClassStructure(
        name=name,
        node=ast.parse(f"class {name}:\n    pass").body[0],
        init_assignments=dict(kwargs.pop("assignments", {}) or {}),
        init_details={},
        forward_calls=list(kwargs.pop("calls", []) or []),
        norm_before=list(kwargs.pop("norm_before", []) or []),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def test_primitive_helper_edges():
    assert _first(None, None, None) is None
    assert _as_bool(5) is True
    assert _as_bool(0) is False
    assert _human_bytes(512) == "512 B"
    assert _human_bytes(2048).endswith("KiB")


# --------------------------------------------------------------------------- #
# _infer_ffn_and_moe / _infer_norm branches
# --------------------------------------------------------------------------- #


def test_hybrid_and_moe_note_branches():
    spec = parse_architecture(
        {
            "model_type": "custom",
            "num_hidden_layers": 8,
            "hidden_act": "silu",
            "hybrid_block_types": ["a", "b"],
            "first_k_dense_replace": 2,
            "moe_layer_start_index": 3,
            "moe_layer_interval": 2,
            "mlp_only_layers": [0, 1],
        },
        "fixture",
    )
    # num_experts absent but hybrid_block_types present -> Hybrid (line 322)
    assert spec.decoder_type == "Hybrid"
    assert any("First 2 layers are dense" in note for note in spec.moe_notes)
    assert any("MoE from layer 3" in note for note in spec.moe_notes)
    assert any("Dense FFN on layer indices" in note for note in spec.moe_notes)


def test_layer_types_counter_and_gelu_gated():
    spec = parse_architecture(
        {
            "model_type": "custom",
            "num_hidden_layers": 4,
            "hidden_act": "gelu",
            "gated_ffn": True,
            "layer_types": ["full", "full", "sliding", "sliding"],
        },
        "fixture",
    )
    assert spec.decoder_type == "Hybrid"
    assert spec.ffn_type == "GeGLU"
    assert "2 full" in spec.layer_mix


def test_norm_placement_olmo_and_post_norm():
    olmo = parse_architecture(
        {"model_type": "olmo2", "layer_norm_eps": 1e-5}, "fixture"
    )
    assert olmo.norm_placement.startswith("Post-Norm")
    assert "Sandwich / post-norm variant" in olmo.norm_notes

    post = parse_architecture(
        {"model_type": "custom", "post_norm": True, "rms_norm_eps": 1e-6}, "fixture"
    )
    assert post.norm_placement == "Post-Norm"


def test_total_params_hint_from_config():
    spec = parse_architecture(
        {"model_type": "custom", "total_params": "70B"}, "fixture"
    )
    assert spec.total_params_hint == "70B"


# --------------------------------------------------------------------------- #
# _estimate_kv_cache / _estimate_param_hint early exits
# --------------------------------------------------------------------------- #


def test_estimate_kv_cache_missing_heads_returns():
    spec = ArchitectureSpec(name="x", model_type="x", num_hidden_layers=2)
    _estimate_kv_cache(spec, {})
    assert spec.kv_cache_per_token_bf16 is None


def test_estimate_param_hint_early_return_when_present():
    spec = ArchitectureSpec(
        name="x",
        model_type="x",
        total_params_hint="1B",
        hidden_size=64,
        num_hidden_layers=2,
        vocab_size=100,
    )
    _estimate_param_hint({}, spec)
    assert spec.total_params_hint == "1B"


# --------------------------------------------------------------------------- #
# _build_highlights custom block branch
# --------------------------------------------------------------------------- #


def test_build_highlights_includes_custom_block():
    spec = ArchitectureSpec(
        name="x",
        model_type="x",
        attention_type="MHA",
        decoder_type="Dense",
        positional_encoding="RoPE",
        custom_blocks=["MambaMixer"],
    )
    _build_highlights(spec)
    assert "MambaMixer" in spec.highlights


# --------------------------------------------------------------------------- #
# _resolve_attention_from_config_lists edge branches
# --------------------------------------------------------------------------- #


def test_resolve_attention_from_config_lists_edges():
    decoder = _structure("Decoder")
    decoder.init_assignment_options = {"self_attn": ["SlidingAttention"]}
    registry = {"Decoder": decoder}

    # No class_registry -> None (394)
    assert (
        _resolve_attention_from_config_lists(
            0, {"layer_types": ["full"]}, class_registry=None, decoder_class="Decoder"
        )
        is None
    )
    # decoder not a ClassStructure -> None (399)
    assert (
        _resolve_attention_from_config_lists(
            0,
            {"layer_types": ["full"]},
            class_registry={"Decoder": object()},
            decoder_class="Decoder",
        )
        is None
    )
    # decoder has no self_attn options -> None (402)
    empty = _structure("Empty")
    assert (
        _resolve_attention_from_config_lists(
            0,
            {"layer_types": ["full"]},
            class_registry={"Empty": empty},
            decoder_class="Empty",
        )
        is None
    )
    # block_type matches neither marker -> fallback options[0] (417)
    assert (
        _resolve_attention_from_config_lists(
            0,
            {"layer_types": ["sliding"]},
            class_registry=registry,
            decoder_class="Decoder",
        )
        == "SlidingAttention"
    )


def test_default_attention_class_no_match_returns_none():
    decoder = _structure("Decoder", assignments={"mlp": "DenseMLP"})
    assert _default_attention_class({"Decoder": decoder}, "Decoder") is None


# --------------------------------------------------------------------------- #
# _resolve_ffn_for_layer fallbacks
# --------------------------------------------------------------------------- #


def test_resolve_ffn_for_layer_moe_and_default_fallbacks():
    # config moe layer true, no moe-role rule -> "block_sparse_moe", None (514)
    assert _resolve_ffn_for_layer(
        4,
        [("mlp", "DenseMLP", "never")],
        {"num_experts": 8},
    ) == ("block_sparse_moe", None)

    # not a moe layer, no ffn-role rule -> "mlp", None (520)
    assert _resolve_ffn_for_layer(
        0,
        [("side", "Weird", "never")],
        {},
    ) == ("mlp", None)


# --------------------------------------------------------------------------- #
# _apply_uniform_ffn_component continue / already-matching branches
# --------------------------------------------------------------------------- #


def test_apply_uniform_ffn_component_skip_and_noop():
    spec = ArchitectureSpec(
        name="x",
        model_type="x",
        block_components=[
            BlockComponent("attn", "Attn", "attention", "Attn", 0),
            BlockComponent("mlp", "DenseMLP", "ffn", "FFN", 1),
        ],
    )
    # first component skipped (attr mismatch -> continue, 543); second already matches
    # ffn_class -> return without replacing (545)
    _apply_uniform_ffn_component(spec, ffn_attr="mlp", ffn_class="DenseMLP")
    assert spec.block_components[1].class_name == "DenseMLP"


# --------------------------------------------------------------------------- #
# _infer_layer_variants early returns
# --------------------------------------------------------------------------- #


def test_infer_layer_variants_no_layers_returns():
    spec = ArchitectureSpec(name="x", model_type="x", num_hidden_layers=None)
    _infer_layer_variants({}, spec)
    assert spec.layer_variants == []


def test_infer_layer_variants_config_layer_lists_only_returns():
    spec = ArchitectureSpec(name="x", model_type="x", num_hidden_layers=4)
    # layer_types present, but no ffn variation / conditionals / per-layer typing (589)
    _infer_layer_variants({"layer_types": ["full", "full", "full", "full"]}, spec)
    assert spec.layer_variants == []


# --------------------------------------------------------------------------- #
# positional refinement helpers
# --------------------------------------------------------------------------- #


def test_code_rotates_positions_variants():
    # positional component in stack_pre (992-993)
    pre_analysis = CodeAnalysis(
        stack_pre=[BlockComponent("rope", "RoPE", "positional", "RoPE", 0)]
    )
    assert _code_rotates_positions(pre_analysis)

    # class with a positional synthetic forward call, class name not positional (997-998)
    synth = _structure("Mixer", calls=["@positional_l3_apply_rotary_emb"])
    assert _code_rotates_positions(CodeAnalysis(class_registry={"Mixer": synth}))

    # nothing rotates -> False (999)
    plain = _structure("Plain", calls=["proj"])
    assert not _code_rotates_positions(CodeAnalysis(class_registry={"Plain": plain}))


def test_refine_positional_from_code_downgrades_and_skips():
    # non-RoPE spec returns early (1013)
    spec_nope = ArchitectureSpec(name="x", model_type="x", positional_encoding="NoPE")
    _refine_positional_from_code(spec_nope, CodeAnalysis())
    assert spec_nope.positional_encoding == "NoPE"

    # RoPE claim contradicted by code -> downgraded to NoPE (1016-1020)
    spec = ArchitectureSpec(
        name="x",
        model_type="x",
        positional_encoding="RoPE",
        attention_notes=["RoPE theta=10000", "GQA group size ≈ 4"],
    )
    _refine_positional_from_code(spec, CodeAnalysis())
    assert spec.positional_encoding == "NoPE"
    assert not any(n.startswith("RoPE theta=") for n in spec.attention_notes)
    assert "No positional encoding applied in modeling code" in spec.layer_notes


# --------------------------------------------------------------------------- #
# _rebuild_stack_components / _finalize edge cases
# --------------------------------------------------------------------------- #


def test_rebuild_stack_components_empty_registry_returns():
    spec = ArchitectureSpec(name="x", model_type="x")
    _rebuild_stack_components(spec, CodeAnalysis())
    assert spec.stack_pre == [] and spec.stack_tail == []


def test_rebuild_stack_components_no_stack_model_or_causal_lm():
    # registry non-empty but nothing looks like a model/causal-lm/decoder stack.
    norm = _structure("PlainNorm", calls=["proj"])
    analysis = CodeAnalysis(class_registry={"PlainNorm": norm})
    spec = ArchitectureSpec(name="x", model_type="x")
    _rebuild_stack_components(spec, analysis)
    assert spec.stack_pre == [] and spec.stack_tail == []


def test_finalize_layer_repeat_lines_empty_returns():
    spec = ArchitectureSpec(name="x", model_type="x", layer_repeat_lines=[])
    _finalize_layer_repeat_lines(spec)
    assert spec.layer_repeat_lines == []


# --------------------------------------------------------------------------- #
# End-to-end AST merge via analyze_sources + parse_architecture
# --------------------------------------------------------------------------- #

_MODEL_SOURCE = """
def apply_rotary_emb(x, freqs):
    return x

class RMSNorm:
    def forward(self, x):
        return x

class FancyAttention:
    def __init__(self, config):
        self.q_proj = Linear()
        self.k_proj = Linear()
        self.v_proj = Linear()
        self.o_proj = Linear()
    def forward(self, hidden_states, freqs):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = apply_rotary_emb(q, freqs)
        return self.o_proj(eager_attention_forward(q, k, v))

class GatedMLP:
    def __init__(self, config):
        self.gate_proj = Linear()
        self.up_proj = Linear()
        self.down_proj = Linear()
    def forward(self, x):
        return self.down_proj(self.gate_proj(x))

class SparseMoeBlock:
    def __init__(self, config):
        self.gate = Linear()
        self.experts = ModuleList([Expert() for _ in range(config.num_experts)])
    def moe_infer(self, x, weights):
        return (x * weights).sum(dim=1)
    def forward(self, hidden_states):
        scores = self.gate(hidden_states)
        expert = self.experts[0]
        routed = expert(hidden_states)
        return self.moe_infer(routed, scores)

class CustomMixer:
    def __init__(self, config):
        self.proj = Linear()
    def forward(self, x):
        return self.proj(x)

class DecoderLayer:
    def __init__(self, config, layer_idx):
        self.input_layernorm = RMSNorm()
        self.self_attn = FancyAttention(config)
        self.custom = CustomMixer(config)
        self.post_attention_layernorm = RMSNorm()
        self.mlp = SparseMoeBlock(config)
    def forward(self, hidden_states, freqs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, freqs)
        hidden_states = self.custom(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

class TextModel:
    def __init__(self, config):
        self.embed_tokens = Embedding()
        self.layers = ModuleList([DecoderLayer(config, i) for i in range(config.depth)])
        self.rotary_emb = RotaryEmbedding()
        self.norm = RMSNorm()
    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)

class TextForCausalLM:
    def __init__(self, config):
        self.model = TextModel(config)
        self.lm_head = Linear()
    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))
"""


def _analysis(all_tensor_ops: bool = False) -> CodeAnalysis:
    return analyze_sources(
        {Path("modeling_custom.py"): _MODEL_SOURCE},
        config={"num_experts": 8, "depth": 4},
        all_tensor_ops=all_tensor_ops,
    )


def test_merge_code_analysis_populates_spec():
    analysis = _analysis()
    spec = ArchitectureSpec(name="x", model_type="custom", num_hidden_layers=4)
    _merge_code_analysis(spec, analysis)

    assert spec.decoder_class == "DecoderLayer"
    assert spec.block_components  # populated from AST
    # a custom "other" block produces a layer note (981)
    assert any("Custom block" in note for note in spec.layer_notes)
    # moe module notes recorded (973-977)
    assert any("AST module" in note for note in spec.moe_notes)


def test_parse_architecture_with_code_analysis_end_to_end():
    analysis = _analysis()
    config = {
        "model_type": "custom",
        "architectures": ["TextForCausalLM"],
        "num_hidden_layers": 4,
        "hidden_size": 64,
        "vocab_size": 1000,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "num_experts": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
    }
    spec = parse_architecture(config, "fixture", code_analysis=analysis)
    assert spec.decoder_class == "DecoderLayer"
    assert spec.positional_encoding == "RoPE"  # rotary present in code
    assert spec.ffn_type


# --------------------------------------------------------------------------- #
# Vision tower helpers + export block trees
# --------------------------------------------------------------------------- #


def test_vision_tower_detection_and_export_tree(monkeypatch):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    tower = _structure(
        "FooVisionModel",
        assignments={"patch_embed": "PatchEmbed"},
        calls=["patch_embed"],
    )
    wrapper = _structure("Wrapper", assignments={"visual": "FooVisionModel"})
    registry = {"FooVisionModel": tower, "Wrapper": wrapper}

    # Detection is config-driven: a nested `vision_config` block whose model_type maps
    # (via transformers) to FooVisionConfig -> FooVisionModel, confirmed in the registry.
    monkeypatch.setitem(CONFIG_MAPPING_NAMES, "foo_vision", "FooVisionConfig")
    spec = ArchitectureSpec(
        name="x",
        model_type="x",
        class_registry=registry,
        stack_model_class="Wrapper",
        block_components=[BlockComponent("mlp", "GatedMLP", "ffn", "FFN", 1)],
        raw_config={"vision_config": {"model_type": "foo_vision"}},
    )
    assert find_vision_tower(spec) == ("visual", "FooVisionModel")
    component = vision_tower_component(spec)
    assert component is not None and component.role == "vision"

    basic_ops = BasicOpFilter.for_detailed()
    _build_export_block_trees(spec, basic_ops)
    assert any(tree.attr_name == "visual" for _label, tree in spec.export_block_trees)


def test_build_export_block_trees_no_registry():
    spec = ArchitectureSpec(name="x", model_type="x")
    _build_export_block_trees(spec, BasicOpFilter.for_detailed())
    assert spec.export_block_trees == []


# --------------------------------------------------------------------------- #
# _resolve_checkpoint from GitHub config + dump_model_ast
# --------------------------------------------------------------------------- #


def test_resolve_checkpoint_github_and_missing(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"_class_name": "GhModel", "num_layers": 2}), encoding="utf-8"
    )
    ref = SimpleNamespace(display="org/repo@main")
    monkeypatch.setattr(extract, "parse_github_url", lambda _url: ref)
    monkeypatch.setattr(extract, "fetch_github_source", lambda _ref: tmp_path)
    monkeypatch.setattr(extract, "github_config_path", lambda _root: config_path)

    config, label = _resolve_checkpoint(checkpoint=None, github="github:org/repo")
    assert config["num_hidden_layers"] == 2
    assert label == "github-config://org/repo@main"

    monkeypatch.setattr(extract, "github_config_path", lambda _root: None)
    with pytest.raises(FileNotFoundError, match="No checkpoint provided"):
        _resolve_checkpoint(checkpoint=None, github="github:org/repo")

    with pytest.raises(ValueError, match="Provide a Hugging Face checkpoint"):
        _resolve_checkpoint(checkpoint=None, github=None)


def test_dump_model_ast_success_and_missing(monkeypatch, tmp_path):
    modeling = tmp_path / "modeling_x.py"
    modeling.write_text("class A:\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(
        extract, "_resolve_checkpoint", lambda **_kwargs: ({"model_type": "x"}, "label")
    )
    monkeypatch.setattr(
        extract, "resolve_source_files", lambda *a, **k: ([modeling], ["label"])
    )
    dumped = dump_model_ast(checkpoint="x")
    assert "ClassDef" in dumped

    monkeypatch.setattr(extract, "resolve_source_files", lambda *a, **k: ([], []))
    with pytest.raises(FileNotFoundError, match="No modeling source file"):
        dump_model_ast(checkpoint="x")


# --------------------------------------------------------------------------- #
# load_architecture end-to-end from a local checkpoint + code_path
# --------------------------------------------------------------------------- #


def test_load_architecture_local_checkpoint_detailed(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "custom",
                "architectures": ["TextForCausalLM"],
                "num_hidden_layers": 4,
                "hidden_size": 64,
                "vocab_size": 1000,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "num_experts": 8,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000,
            }
        ),
        encoding="utf-8",
    )
    modeling = root / "modeling_custom.py"
    modeling.write_text(_MODEL_SOURCE, encoding="utf-8")

    spec = load_architecture(
        source=root,
        code_path=modeling,
        detailed=True,
    )
    assert spec.decoder_class == "DecoderLayer"
    assert spec.export_block_trees
    assert spec.basic_ops is not None
