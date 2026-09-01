###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Focused unit coverage for architecture extraction and block-tree helpers."""

from __future__ import annotations

import ast
import json
import re

import pytest

from visualizer.ast_analyze import ClassStructure, SideInputSpec
from visualizer.basic_ops import (
    BasicOpFilter,
    introspect_is_modeling_operation,
    is_fused_silu_mul_class,
    keep_detail_graph_node,
    resolve_is_basic,
    show_in_detail_graph,
)
from visualizer.block_tree import (
    BlockNode,
    CombineSegment,
    FanOutSegment,
    ResidualAddSegment,
    SeqSegment,
    SideCombineSegment,
    SideFeedSegment,
    TensorPortsSegment,
    block_purpose,
    build_block_node,
    collect_computation_segments,
    collect_graph_segments,
    collect_method_wrappers,
    collect_parallel_gate_wrappers,
    components_from_registry,
    gated_norm_activation,
    gated_norm_tile_label,
    inline_block_frame_label,
    inline_composite_steps,
    is_basic_op_tile,
    is_inline_expandable_module,
    is_method_wrapper,
    is_simple_modeled_tile,
    is_single_function_tree,
    is_straight_line_module,
    partition_detail_trees,
    side_producer_has_activation,
    spine_expanded_frame_label,
    straight_line_steps,
    tile_display_labels,
    wrapper_bullet,
    wrapper_panel_line,
    wrapper_skips_comment,
)
from visualizer.blocks import (
    BlockComponent,
    CodeAnalysis,
    collect_norm_module_pairs,
    input_sources_from_forward_sequence,
    ordered_components,
    upstream_input_sources,
)
from visualizer.config_resolve import (
    _paths_from_model_index,
    _score_config_content,
    _score_config_path,
    discover_local_config,
    discover_remote_config,
    load_checkpoint_config,
    normalize_config,
)
from visualizer.extract import (
    ArchitectureSpec,
    _as_bool,
    _as_int,
    _config_has_ffn_layer_variation,
    _config_moe_layer,
    _estimate_kv_cache,
    _finalize_layer_repeat_lines,
    _first,
    _format_params,
    _human_bytes,
    _resolve_checkpoint,
    architecture_section_trees,
    parse_architecture,
)
from visualizer.layer_repeat_simplify import (
    _compact_ranges,
    _format_layer_index_set,
    _parse_layer_index_set,
    layer_condition_matches,
    simplify_layer_repeat_lines,
)


def node(
    attr: str,
    class_name: str = "Linear",
    *,
    role: str = "other",
    label: str | None = None,
    basic: bool = True,
    children: list[BlockNode] | None = None,
    details: list[str] | None = None,
) -> BlockNode:
    return BlockNode(
        attr_name=attr,
        class_name=class_name,
        role=role,
        label=label or class_name,
        is_basic=basic,
        children=list(children or []),
        details=list(details or []),
    )


def structure(
    name: str,
    *,
    assignments: dict[str, str] | None = None,
    calls: list[str] | None = None,
    norm_before: list[str] | None = None,
) -> ClassStructure:
    return ClassStructure(
        name=name,
        node=ast.parse(f"class {name}:\n    pass").body[0],
        init_assignments=dict(assignments or {}),
        init_details={},
        forward_calls=list(calls or []),
        norm_before=list(norm_before or []),
    )


def component(
    attr: str,
    order: int | None,
    *,
    role: str = "ffn",
    label: str | None = None,
) -> BlockComponent:
    return BlockComponent(
        attr_name=attr,
        class_name="Linear",
        role=role,
        label=label or attr.title(),
        forward_order=order,
    )


def test_basic_op_filter_cli_and_patterns():
    custom = BasicOpFilter.from_cli(
        remove=[r"(?i)^aten_"], add=[r"(?i)^CustomLeaf$", re.compile("Extra")]
    )

    assert custom.is_basic("aten.add")
    assert custom.is_basic("CustomLeaf")
    assert custom.is_basic("", "ExtraOp")
    assert not custom.is_basic("aten_fake")
    assert r"(?i)^CustomLeaf$" in custom.pattern_strings()
    assert BasicOpFilter.for_detailed().basic_only
    assert BasicOpFilter.for_detailed().is_basic("Linear", "proj")
    assert not BasicOpFilter.for_detailed().is_basic("Embedding")


@pytest.mark.parametrize(
    ("class_name", "attr_name", "details", "expected"),
    [
        ("SituAndMul", "act", None, True),
        ("Anything", "@functional_relu", None, False),
        ("Anything", "@attention", None, True),
        ("Linear", "g_proj", None, False),
        ("CustomGate", "gate", None, True),
        ("Thing", "x", ["kernel: scan"], True),
        ("Thing", "x", ["method `helper()`"], False),
        ("Embedding", "embed", None, False),
    ],
)
def test_modeling_operation_classification(class_name, attr_name, details, expected):
    assert introspect_is_modeling_operation(class_name, attr_name, details) is expected


def test_basic_op_resolution_and_detail_visibility():
    basic_filter = BasicOpFilter.for_detailed()
    linear = node("proj")
    method = node("helper", "helper", details=["method `helper()`"])
    modeled = node("scan", "KernelOp", basic=False)

    assert is_fused_silu_mul_class("FusedSiluAndMul")
    assert not is_fused_silu_mul_class(None)
    assert resolve_is_basic("Linear", "proj", basic_filter, in_registry=True)
    assert not resolve_is_basic("KernelOp", "scan", basic_filter)
    assert show_in_detail_graph(linear, basic_only=True)
    assert not show_in_detail_graph(method, basic_only=True)
    assert keep_detail_graph_node(synthetic="@input", basic_only=True)
    assert keep_detail_graph_node(synthetic="@combine", label="+", basic_only=True)
    assert keep_detail_graph_node(block=modeled, label="SiLU", basic_only=True)
    assert not keep_detail_graph_node(label="Opaque", basic_only=True)
    assert keep_detail_graph_node(label="anything", basic_only=False)


def test_component_order_norm_pairs_and_sources():
    comps = [
        component("late", 3),
        component("norm", 1, role="norm", label="Norm"),
        component("attn", 2, role="attention", label="Attention"),
        component("parallel", 2, label="Parallel"),
        component("unknown", None),
    ]

    assert [item.attr_name for item in ordered_components(comps)] == [
        "norm",
        "attn",
        "parallel",
        "late",
        "unknown",
    ]
    assert [
        (a.attr_name, b.attr_name) for a, b in collect_norm_module_pairs(comps)
    ] == [
        ("norm", "attn"),
        ("norm", "parallel"),
    ]
    assert upstream_input_sources(comps) == {
        "attn": "Norm",
        "parallel": "Norm",
        "late": "Attention",
    }
    source_comps = [
        component("norm", 0, role="norm", label="Norm"),
        component("attn", 1, role="attention", label="Attention"),
        component("parallel", 2, label="Parallel"),
        component("late", 3),
    ]
    assert input_sources_from_forward_sequence(
        source_comps, ["norm", "attn", "parallel", "late"]
    ) == {"attn": "Norm", "parallel": "Attention", "late": "Parallel"}


def test_code_analysis_presence():
    assert not CodeAnalysis().has_block_graph()
    assert CodeAnalysis(forward_sequence=["self_attn"]).has_block_graph()


def test_config_scoring_and_model_index_paths():
    assert _score_config_path("README.md") == -100
    assert _score_config_path("language_model/config.json") > _score_config_path(
        "tokenizer/config.json"
    )
    assert _score_config_content(
        {
            "model_type": "x",
            "architectures": ["XForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 16,
        },
        "config.json",
    ) > _score_config_content({"_class_name": "PipelineThing"}, "vae/config.json")
    assert _paths_from_model_index(
        {
            "text_encoder": ["library", "Class", {"subfolder": "text"}],
            "prior": "prior/model_index.json",
            "nested": {"decoder": "decoder/model_index.json"},
            "_ignored": ["library", "Class", {"subfolder": "ignored"}],
        }
    ) == [
        "text/config.json",
        "prior/config.json",
        "decoder/config.json",
    ]


def test_normalize_config_flattens_wrapper_and_fallback_fields():
    normalized = normalize_config(
        {
            "model_type": "wrapper",
            "architectures": ["Wrapper"],
            "vision_config": {"model_type": "vision"},
            "text_config": {
                "_class_name": "TextModel",
                "num_layers": 3,
                "ffn_hidden_size": 128,
            },
        },
        source_label="fixture",
    )

    assert normalized["model_type"] == "wrapper"
    assert normalized["architectures"] == ["Wrapper"]
    assert normalized["num_hidden_layers"] == 3
    assert normalized["intermediate_size"] == 128
    assert normalized["_wrapper_model_type"] == "wrapper"
    assert normalized["_has_vision_tower"]
    assert normalized["_config_source"] == "fixture"
    fallback = normalize_config({"_class_name": "TextModel"})
    assert fallback["model_type"] == "textmodel"
    assert fallback["architectures"] == ["TextModel"]


def test_local_config_discovery_file_direct_and_ranked(tmp_path):
    standalone = tmp_path / "standalone.json"
    standalone.write_text(json.dumps({"model_type": "file"}), encoding="utf-8")
    config, label = discover_local_config(standalone)
    assert config["model_type"] == "file"
    assert label == str(standalone.resolve())

    direct_root = tmp_path / "direct"
    direct_root.mkdir()
    direct = direct_root / "config.json"
    direct.write_text(json.dumps({"model_type": "direct"}), encoding="utf-8")
    assert discover_local_config(direct_root)[0]["model_type"] == "direct"

    ranked_root = tmp_path / "ranked"
    (ranked_root / "tokenizer").mkdir(parents=True)
    (ranked_root / "language_model").mkdir()
    (ranked_root / "tokenizer" / "config.json").write_text("{}", encoding="utf-8")
    best = ranked_root / "language_model" / "config.json"
    best.write_text(
        json.dumps({"model_type": "best", "num_hidden_layers": 4}),
        encoding="utf-8",
    )
    assert discover_local_config(ranked_root)[1] == str(best.resolve())

    with pytest.raises(FileNotFoundError, match="No config.json"):
        discover_local_config(tmp_path / "missing")


def test_load_checkpoint_config_explicit_paths(tmp_path):
    root = tmp_path / "checkpoint"
    nested = root / "nested"
    nested.mkdir(parents=True)
    config_path = nested / "config.json"
    config_path.write_text(json.dumps({"model_type": "nested"}), encoding="utf-8")

    config, label = load_checkpoint_config(root, config_path="nested/config.json")
    assert config["model_type"] == "nested"
    assert label == str(config_path.resolve())
    assert load_checkpoint_config(root, config_path=str(config_path))[0] == config
    with pytest.raises(FileNotFoundError, match="Config path not found"):
        load_checkpoint_config(root, config_path="absent.json")


def test_remote_config_discovery_ranks_downloaded_candidates(monkeypatch, tmp_path):
    payloads = {
        "model_index.json": {
            "text_encoder": ["pkg", "Text", {"subfolder": "text_encoder"}]
        },
        "config.json": {"_class_name": "Pipeline"},
        "text_encoder/config.json": {
            "model_type": "text",
            "architectures": ["TextForCausalLM"],
            "hidden_size": 32,
        },
    }

    def fake_download(_model_id, path):
        if path not in payloads:
            raise OSError(path)
        target = tmp_path / path.replace("/", "_")
        target.write_text(json.dumps(payloads[path]), encoding="utf-8")
        return target

    monkeypatch.setattr("visualizer.config_resolve._download_config", fake_download)
    monkeypatch.setattr(
        "visualizer.config_resolve._list_repo_config_paths",
        lambda _model_id: ["config.json", "text_encoder/config.json"],
    )

    config, label = discover_remote_config("org/model")
    assert config["model_type"] == "text"
    assert label == "hf://org/model/text_encoder/config.json"


def test_remote_config_discovery_errors(monkeypatch):
    monkeypatch.setattr(
        "visualizer.config_resolve._download_config",
        lambda *_args: (_ for _ in ()).throw(OSError("offline")),
    )
    monkeypatch.setattr(
        "visualizer.config_resolve._list_repo_config_paths", lambda _model_id: []
    )
    with pytest.raises(FileNotFoundError, match="Could not load any config.json"):
        discover_remote_config("org/empty")

    monkeypatch.setattr(
        "visualizer.config_resolve._list_repo_config_paths",
        lambda _model_id: (_ for _ in ()).throw(OSError("denied")),
    )
    with pytest.raises(FileNotFoundError, match="Could not list files"):
        discover_remote_config("org/error")


def test_extract_primitive_helpers():
    assert _first(None, None, 3) == 3
    assert _as_int("4") == 4
    assert _as_int("bad") is None
    assert _as_bool("YES") is True
    assert _as_bool("no") is False
    assert _human_bytes(512) == "512 B"
    assert _human_bytes(1024) == "1 KiB"
    assert _format_params(999) == "999"
    assert _format_params(2_000_000) == "2M"
    assert _format_params(1_500_000_000) == "1.5B"
    assert _format_params(2_500_000_000_000) == "2.50T"


def test_parse_architecture_infers_gqa_sliding_moe_and_estimates():
    spec = parse_architecture(
        {
            "model_type": "custom",
            "architectures": "CustomForCausalLM",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 128,
            "vocab_size": 1000,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_position_embeddings": 4096,
            "use_sliding_window": True,
            "sliding_window": 256,
            "max_window_layers": 3,
            "attention_bias": True,
            "qk_norm": True,
            "hidden_act": "gelu",
            "gated_ffn": True,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_shared_experts": 1,
            "moe_intermediate_size": 96,
            "rms_norm_eps": 1e-6,
        },
        "fixture",
    )

    assert spec.name == "CustomForCausalLM"
    assert spec.architectures == ["CustomForCausalLM"]
    assert spec.attention_type == "GQA"
    assert spec.layer_mix == "3 sliding-window + 1 global"
    assert "Attention projections use bias" in spec.attention_notes
    assert "QK-Norm inside attention" in spec.norm_notes
    assert spec.ffn_type == "GeGLU"
    assert spec.decoder_type == "Sparse MoE"
    assert spec.kv_cache_per_token_bf16 == "256 B"
    assert spec.total_params_hint
    assert spec.active_params_hint
    assert spec.highlights[0] == "GQA"


@pytest.mark.parametrize(
    ("config", "attention", "position"),
    [
        (
            {
                "model_type": "deepseek",
                "kv_lora_rank": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
            },
            "MLA",
            "RoPE",
        ),
        ({"model_type": "x", "alibi": True}, "MHA", "ALiBi"),
        (
            {"model_type": "x", "position_embedding_type": "absolute"},
            "MHA",
            "Learned absolute",
        ),
        ({"model_type": "gpt2", "layer_norm_eps": 1e-5}, "MHA", "Learned absolute"),
        ({"model_type": "nope_model", "rope_theta": 10_000}, "MHA", "NoPE"),
    ],
)
def test_parse_architecture_attention_and_position_branches(
    config, attention, position
):
    spec = parse_architecture(config, "fixture")
    assert spec.attention_type == attention
    assert spec.positional_encoding == position


def test_kv_cache_and_moe_layer_helpers():
    mla = ArchitectureSpec(
        name="x",
        model_type="x",
        num_hidden_layers=2,
        num_attention_heads=4,
        attention_type="MLA",
    )
    _estimate_kv_cache(mla, {"kv_lora_rank": 8})
    assert mla.kv_cache_per_token_bf16 == "128 B"

    assert _config_moe_layer(2, {"num_experts": 8, "moe_layer_freq": [0, 1, 1]})
    assert not _config_moe_layer(3, {"num_experts": 8, "moe_layer_freq": [0, 1, 1]})
    assert _config_moe_layer(
        4, {"num_experts": 8, "first_k_dense_replace": 2, "moe_layer_interval": 2}
    )
    assert _config_has_ffn_layer_variation({"mlp_layer_types": ["dense", "sparse"]})
    assert _config_has_ffn_layer_variation(
        {"num_experts": 8, "first_k_dense_replace": 1}
    )


def test_layer_repeat_finalization_and_architecture_trees():
    tree = node("mlp", "MLP", basic=False)
    spec = ArchitectureSpec(
        name="x",
        model_type="x",
        num_hidden_layers=3,
        layer_repeat_lines=["N × DecoderLayer (layer_idx in range(config.layers))"],
        raw_config={"layers": 3},
        export_block_trees=[("MLP", tree)],
    )
    _finalize_layer_repeat_lines(spec)

    assert spec.layer_repeat_lines == ["3 × DecoderLayer (layer_idx in range(3))"]
    assert architecture_section_trees(spec) == [("MLP", tree)]


def test_resolve_checkpoint_requires_source():
    with pytest.raises(ValueError, match="Provide a Hugging Face checkpoint"):
        _resolve_checkpoint(checkpoint=None, github=None)


@pytest.mark.parametrize(
    ("layer", "condition", "config", "expected"),
    [
        (2, "if layer_idx >= 2", {}, True),
        (3, "elif layer_idx % 2 == 0", {}, False),
        (1, "layer_idx not in [0, 2]", {}, True),
        (0, "not (layer_idx == 0)", {}, False),
        (2, "(layer_idx + 1) in config.linear_layers", {"linear_layers": [1, 3]}, True),
        (
            1,
            "config.is_linear_layer(layer_idx)",
            {"linear_layers": [1, 3]},
            False,
        ),
        (0, "config.is_mla and layer_idx == 0", {"kv_lora_rank": 8}, True),
        (0, "unknown_call(layer_idx)", {}, False),
        (8, "else", {}, True),
    ],
)
def test_layer_condition_matches(layer, condition, config, expected):
    assert layer_condition_matches(layer, condition, config) is expected


def test_repeat_line_simplification_and_ranges():
    lines = [
        "LinearAttention (if config.is_linear_layer(layer_idx) and True)",
        "DenseAttention (elif config.is_mla)",
        "Fallback (else)",
        "N × Decoder (layer_idx in range(config.num_hidden_layers))",
        "unchanged",
    ]
    simplified = simplify_layer_repeat_lines(
        lines,
        {
            "linear_layers": [1, 2, 3, 7],
            "kv_lora_rank": 8,
            "num_hidden_layers": 8,
        },
    )

    assert simplified == [
        "LinearAttention (if (layer_idx + 1) in [1–3, 7])",
        "DenseAttention (else)",
        "Fallback (else)",
        "N × Decoder (layer_idx in range(8))",
        "unchanged",
    ]
    assert _parse_layer_index_set("[1–3, 7, bad]") == {1, 2, 3, 7}
    assert _parse_layer_index_set("not-a-list") == set()
    assert _compact_ranges([1, 2, 3, 5]) == "1–3, 5"
    assert "13 layers" in _format_layer_index_set(list(range(1, 14)))


def test_wrapper_labels_comments_and_purpose():
    method = node(
        "_update_state",
        "_update_state",
        label="_update state",
        details=["method `_update_state()`"],
    )
    assert is_method_wrapper(method)
    assert wrapper_bullet(method) == "update state (_update_state)"
    assert wrapper_panel_line(method) == "update state (_update_state)"

    ffn = node("mlp", "MLP", role="ffn", label="MLP", basic=False)
    assert block_purpose(ffn) == "Position-wise feed-forward transform"
    assert wrapper_panel_line(ffn) == "mlp — Position-wise feed-forward transform"
    assert wrapper_skips_comment(node("self_attn", "Attention", role="attention"))
    assert wrapper_skips_comment(node("residual_add", "Add"))
    assert not wrapper_skips_comment(ffn)


def test_gated_norm_and_tile_helpers():
    gated = node(
        "norm",
        "FusedRMSNormGated",
        role="norm",
        label="Norm",
        basic=False,
        details=["SiLU"],
    )
    assert gated_norm_tile_label(gated) == "RMSNorm"
    assert gated_norm_activation(gated) == "SiLU"
    assert (
        gated_norm_activation(node("norm", "SomeNormGated", role="norm", basic=False))
        == "Sigmoid"
    )
    assert is_simple_modeled_tile(gated)
    assert is_basic_op_tile(gated)
    assert tile_display_labels(gated, spec_label="Norm")[0] == "Norm"
    assert tile_display_labels(None, spec_label="Unknown") == ("Unknown", None)


def test_collect_straight_line_steps_and_partitioning():
    first = node("first")
    second = node("second")
    pipeline = node(
        "pipeline",
        "Pipeline",
        basic=False,
        children=[first, second],
    )

    assert is_straight_line_module(pipeline)
    assert is_inline_expandable_module(pipeline)
    assert straight_line_steps(pipeline) == [first, second]
    steps, wrapper = inline_composite_steps(pipeline)
    assert steps == [first, second]
    assert wrapper is pipeline
    assert not is_single_function_tree(pipeline)
    assert partition_detail_trees(
        [
            ("tokenizer", node("tokenizer")),
            ("pipeline", pipeline),
            ("kept", node("opaque", "Opaque", basic=False)),
        ]
    ) == [("kept", node("opaque", "Opaque", basic=False))]


def test_kernel_and_fused_inline_frame_labels():
    kernel = node(
        "@attn_pipeline",
        "KernelPipeline",
        label="KDA pipeline",
        basic=False,
        children=[node("stage", "KernelOp", basic=False)],
    )
    fused = node(
        "act_fn",
        "SituAndMul",
        basic=False,
        children=[node("activation")],
    )
    assert inline_block_frame_label(kernel) == "KDA pipeline"
    assert inline_block_frame_label(fused) == "SituAndMul"
    assert inline_composite_steps(kernel) == ([kernel.children[0]], kernel)


def test_computation_segments_for_sequence_tensor_ports_and_fanout():
    sequential = node(
        "parent",
        "Parent",
        basic=False,
        children=[node("one"), node("two")],
    )
    assert all(
        isinstance(segment, SeqSegment)
        for segment in collect_computation_segments(sequential)
    )

    tensor = node(
        "kernel",
        "KernelPipeline",
        basic=False,
        children=[node("stage", "KernelOp", basic=False)],
    )
    tensor.tensor_input_labels = ["q", "k"]
    tensor.tensor_step_targets = {"q": "stage"}
    segment = collect_computation_segments(tensor)[0]
    assert isinstance(segment, TensorPortsSegment)
    assert segment.labels == ["q", "k"]

    q = node("q_proj")
    k = node("k_proj")
    merge = node("@attention", "AttentionOp", basic=False)
    attention = node(
        "attention",
        "Attention",
        basic=False,
        children=[q, k, merge],
    )
    attention.attention_inputs = {"q": ["q_proj"], "k": ["k_proj"]}
    fanout = collect_computation_segments(attention)[0]
    assert isinstance(fanout, FanOutSegment)
    assert [branch.label for branch in fanout.branches] == ["q", "k"]


def test_computation_segments_for_side_inputs_and_parallel_gate():
    producer = node("gate")
    consumer = node("norm", "RMSNorm", role="norm")
    parent = node(
        "parent",
        "Parent",
        basic=False,
        children=[producer, consumer],
    )
    parent.side_inputs = {
        "norm": [
            SideInputSpec(
                arg_name="gate",
                port_label="gate",
                source_chain=["gate"],
                source_kind="prior_step",
            )
        ]
    }
    side = collect_computation_segments(parent)[0]
    assert isinstance(side, SideFeedSegment)
    assert side.side_producer_nodes == {"gate": producer}

    residual_parent = node(
        "parent",
        "Parent",
        basic=False,
        children=[consumer],
    )
    residual_parent.side_inputs = {
        "norm": [
            SideInputSpec(
                arg_name="residual",
                port_label="residual",
                source_chain=[],
                source_kind="forward_input",
            )
        ]
    }
    assert isinstance(
        collect_computation_segments(residual_parent)[0], ResidualAddSegment
    )

    method = node("combine", "combine", details=["method `combine()`", "add"])
    method_parent = node(
        "parent",
        "Parent",
        basic=False,
        children=[producer, method],
    )
    method_parent.side_inputs = {
        "combine": [SideInputSpec("gate", "gate", ["gate"], source_kind="prior_step")]
    }
    assert isinstance(
        collect_computation_segments(method_parent)[-1], SideCombineSegment
    )

    merge = node("@attention", "AttentionOp", basic=False)
    gate = node("g_proj", "OutputGate", role="gate", basic=False)
    output = node("o_proj")
    gated_attention = node(
        "attention",
        "Attention",
        basic=False,
        children=[merge, gate, output],
    )
    gated_attention.parallel_gates = ["g_proj"]
    assert isinstance(collect_computation_segments(gated_attention)[-1], CombineSegment)


def test_method_and_parallel_gate_collection():
    method = node("helper", "helper", details=["method `helper()`"])
    gate = node("gate", "Linear")
    root = node(
        "root",
        "Root",
        basic=False,
        children=[method, gate],
    )
    root.parallel_gates = ["gate"]

    assert collect_method_wrappers(root) == [method]
    assert collect_parallel_gate_wrappers(root) == [gate]

    wrapped_gate = node(
        "gate",
        "OutputGate",
        role="gate",
        basic=False,
        children=[node("@gate_activation", "ActivationOp", basic=False)],
    )
    assert side_producer_has_activation(wrapped_gate)


def test_build_block_node_handles_registry_leaf_recursion_and_methods():
    basic_filter = BasicOpFilter.for_detailed()
    registry = {
        "Parent": structure(
            "Parent",
            assignments={"proj": "Linear", "child": "Child"},
            calls=["proj", "helper", "child"],
        ),
        "Child": structure("Child", assignments={"parent": "Parent"}, calls=["parent"]),
    }
    registry["Parent"].forward_step_details["helper"] = ["custom detail"]

    built = build_block_node(
        attr_name="layer",
        class_name="Parent",
        registry=registry,
        basic_ops=basic_filter,
    )

    assert [child.attr_name for child in built.children] == [
        "proj",
        "helper",
        "child",
    ]
    assert built.children[0].is_basic
    assert built.children[1].details == ["custom detail"]
    assert built.children[2].children[0].details == ["recursive reference"]
    assert (
        build_block_node(
            attr_name="external",
            class_name="External",
            registry=registry,
            basic_ops=basic_filter,
        ).is_basic
        is False
    )
    assert components_from_registry("missing", registry) == []


def test_graph_segments_and_spine_labels():
    norm = node("norm", "RMSNorm", role="norm")
    attn = node("attn", "Attention", role="attention", basic=False)
    mlp = node("mlp", "MLP", role="ffn", basic=False)

    segments = collect_graph_segments([norm, attn, mlp], ["attn"], use_residual=True)
    assert segments == [("sublayer", norm, attn), ("seq", mlp)]
    assert collect_graph_segments([norm], [], use_residual=False) == [("seq", norm)]

    positional = BlockComponent(
        "rotary",
        "RotaryEmbedding",
        "positional",
        "RoPE",
    )
    assert (
        spine_expanded_frame_label(positional, positional_encoding="RoPE")
        == "Positional (RoPE) (rotary)"
    )
