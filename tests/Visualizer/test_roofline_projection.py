###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the static, trace-free roofline projection library and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.blocks import BlockComponent, LayerVariant
from TraceLens.ModelUtils.extract import load_architecture
from TraceLens.Visualizer.model_explorer_export.merge import (
    _stamp_variant_ffn_identity,
)
from TraceLens.Visualizer.model_explorer_export import roofline_cli
from TraceLens.Visualizer.model_explorer_export.roofline_projection import (
    DimResolutionError,
    RooflineOp,
    RooflineProjection,
    _RT,
    _attention_boundary_cost,
    _attention_extra_ops,
    _dense_ffn_cost,
    _mtp_stack_ops,
    _attr_text,
    _classify_normalization,
    _conv_cost,
    _dispatch_op_cost,
    _embedding_lookup_cost,
    _is_leaf_computation,
    _parse_conv_details,
    _engine_boundary_cost,
    _engine_head_cost,
    _introspect_weight_bpe,
    _finish_op,
    _graph_contains_draft,
    _moe_cost,
    _moe_extra_ops,
    _parse_op_node,
    _persistent_state_ops,
    _populate_introspection,
    _project_op,
    _LinearWidths,
    _adopt_linear_width,
    _shard,
    _shard_heads,
    _split_shape_text,
    build_projection_dims,
    conv_ops_from_meta,
    project_roofline,
    resolve_dim,
    resolve_shape,
)
from TraceLens.Visualizer.model_explorer_export.shapes import parse_shape_dims

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

_ARCH = {
    "name": "MI300X",
    "mem_bw_gbps": 5300,
    "max_achievable_tflops": {
        "matrix_fp16": 654,
        "matrix_bf16": 708,
        "vector_fp16": 163,
        "vector_bf16": 163,
    },
}


def test_persistent_state_ops_cost_traced_reads_and_writes_independently():
    nodes = [
        {
            "id": "state_input",
            "namespace": "4x_decoder",
            "attrs": [
                {"key": "state_read", "value": "recurrent_state"},
                {"key": "output_shape", "value": "[B, H, D] float32"},
            ],
        },
        {
            "id": "layers.0.recurrence",
            "namespace": "4x_decoder",
            "attrs": [
                {"key": "state_write", "value": "recurrent_state"},
                {"key": "state_write", "value": "auxiliary_state"},
                {"key": "output_shape", "value": "[B, H, 2*D] bfloat16"},
            ],
        },
    ]

    ops = _persistent_state_ops(nodes, {"B": 2, "H": 8, "D": 16}, _ARCH)

    assert [op.computation for op in ops] == [
        "persistent state read (recurrent_state)",
        "persistent state write (recurrent_state)",
        "persistent state write (auxiliary_state)",
    ]
    assert [op.bytes for op in ops] == [2 * 8 * 16 * 4, 2 * 8 * 32 * 2, 2 * 8 * 32 * 2]
    assert all(op.instances == 4 for op in ops)
    assert all(op.flops == 0 for op in ops)
    assert all(op.roofline_bound == "MEMORY_BOUND" for op in ops)
    assert all(
        op.roofline_time_us == pytest.approx(op.bytes / 5300e9 * 1e6) for op in ops
    )


def test_persistent_state_ops_skip_unresolved_or_untyped_state():
    nodes = [
        {
            "id": "unknown-shape",
            "namespace": "",
            "attrs": [
                {"key": "details", "value": "stores: state"},
                {"key": "state_write", "value": "state"},
                {"key": "output_shape", "value": "[unknown] bfloat16"},
            ],
        },
        {
            "id": "unknown-type",
            "namespace": "",
            "attrs": [
                {"key": "details", "value": "stores: state"},
                {"key": "state_write", "value": "state"},
                {"key": "output_shape", "value": "[4] opaque"},
            ],
        },
    ]

    assert _persistent_state_ops(nodes, {}, _ARCH) == []


# resolve_dim / resolve_shape


def test_resolve_dim_int_passthrough():
    assert resolve_dim(4096, {}) == 4096


def test_resolve_dim_negative_int_is_unresolved():
    with pytest.raises(DimResolutionError):
        resolve_dim(-1, {})


def test_resolve_dim_numeric_string():
    assert resolve_dim("128", {}) == 128


def test_resolve_dim_symbol_lookup():
    assert resolve_dim("B", {"B": 2}) == 2


def test_resolve_dim_product_and_sum():
    dims = {"B": 2, "S": 10, "past_len": 5}
    assert resolve_dim("B*S", dims) == 20
    assert resolve_dim("past_len+S", dims) == 15
    assert resolve_dim("B*S+past_len", dims) == 25


def test_resolve_dim_unresolved_symbol_raises():
    with pytest.raises(DimResolutionError):
        resolve_dim("B", {})  # unset symbol maps to itself in ShapeContext.dims


def test_resolve_dim_missing_symbol_raises():
    with pytest.raises(DimResolutionError):
        resolve_dim("unknown_dim", {})


def test_resolve_dim_self_mapped_symbol_raises():
    # ShapeContext.dims maps a genuinely-unset symbol to itself rather than
    # omitting it; this is a different unresolved shape than a name dims never
    # mentions at all (test_resolve_dim_missing_symbol_raises above).
    with pytest.raises(DimResolutionError):
        resolve_dim("B", {"B": "B"})


def test_resolve_shape():
    assert resolve_shape(["B", "S", 4096], {"B": 2, "S": 10}) == (2, 10, 4096)


# _attr_text / _parse_op_node
def test_attr_text_returns_none_for_non_list_attrs():
    assert _attr_text(None, "op_type") is None
    assert _attr_text("not-a-list", "op_type") is None


def test_parse_op_node_skips_a_collective_boundary_node():
    node = {
        "id": "allreduce@comm:0",
        "attrs": [{"key": "op_type", "value": "Collective"}],
    }
    assert _parse_op_node(node) is None


def test_parse_op_node_skips_a_node_without_op_type():
    node = {"id": "n1", "attrs": [{"key": "class_name", "value": "Linear"}]}
    assert _parse_op_node(node) is None


def test_parse_op_node_skips_a_node_without_output_shape():
    node = {
        "id": "n1",
        "attrs": [
            {"key": "op_type", "value": "GEMM"},
            {"key": "class_name", "value": "Linear"},
        ],
    }
    assert _parse_op_node(node) is None


def test_parse_op_node_classifies_an_engine_parallel_linear_as_linear():
    # vLLM/SGLang split-aware linear subclasses (QKVParallelLinear, ...) carry
    # their own op_type label but cost exactly like a plain Linear GEMM.
    node = {
        "id": "n1",
        "namespace": "decoder/0x_layer",
        "attrs": [
            {"key": "op_type", "value": "QKVParallelLinear"},
            {"key": "class_name", "value": "QKVParallelLinear"},
            {"key": "output_shape", "value": "[B, S, H]"},
        ],
    }
    parsed = _parse_op_node(node)
    assert parsed is not None
    (
        _section,
        computation,
        class_name,
        _operation,
        _output_text,
        _input_texts,
        _role,
        _details,
    ) = parsed
    assert computation == "Linear"
    assert class_name == "QKVParallelLinear"


# _classify_normalization


@pytest.mark.parametrize(
    "computation,expected_kind,expected_bias,expected_channel_first",
    [
        ("RMSNorm", "rmsnorm", False, False),
        ("LlamaRMSNorm", "rmsnorm", False, False),
        ("CustomRMSNorm", "rmsnorm", False, False),
        ("LayerNorm", "generic", True, False),
        ("Qwen2LayerNorm", "generic", True, False),
        ("GroupNorm", "generic", True, True),
        ("BatchNorm2d", "generic", True, True),
        ("InstanceNorm1d", "generic", True, True),
    ],
)
def test_classify_normalization(
    computation, expected_kind, expected_bias, expected_channel_first
):
    assert _classify_normalization(computation) == (
        expected_kind,
        expected_bias,
        expected_channel_first,
    )


def test_classify_normalization_unrecognized_returns_none():
    assert _classify_normalization("Softmax") is None
    assert _classify_normalization("Linear") is None


# _dispatch_op_cost: synthetic per-family unit tests


def test_dispatch_gemm_linear():
    category, maf_type, flops, bytes_moved = _dispatch_op_cost(
        "Linear", [(2, 10, 4096)], (2, 10, 1024), 2
    )
    assert category == "GEMM"
    assert maf_type == "matrix"
    assert flops == 2 * 20 * 1024 * 4096
    assert bytes_moved is not None


def test_dispatch_gemm_linear_without_an_input_shape_is_unmodeled():
    assert _dispatch_op_cost("Linear", [], (2, 10, 1024), 2) is None
    assert _dispatch_op_cost("Linear", [None], (2, 10, 1024), 2) is None


def test_dispatch_gemm_functional_matmul():
    category, _, flops, _ = _dispatch_op_cost(
        "MatMul", [(2, 10, 64), (64, 128)], (2, 10, 128), 2
    )
    assert category == "GEMM"
    assert flops == 2 * 20 * 128 * 64


def test_dispatch_gemm_matmul_with_a_missing_operand_is_unmodeled():
    assert _dispatch_op_cost("MatMul", [(2, 10, 64)], (2, 10, 128), 2) is None
    assert _dispatch_op_cost("MatMul", [(2, 10, 64), None], (2, 10, 128), 2) is None


def test_dispatch_sdpa():
    category, maf_type, flops, bytes_moved = _dispatch_op_cost(
        "scaled_dot_product_attention",
        [(1, 32, 2048, 128), (1, 8, 2048, 128), (1, 8, 2048, 128)],
        (1, 32, 2048, 128),
        2,
    )
    assert category == "SDPA"
    assert maf_type == "matrix"
    assert flops > 0
    assert bytes_moved is not None


# CONV: geometry is threaded on the node's ``details`` attr by the meta pass.


def test_parse_conv_details():
    assert _parse_conv_details(
        "conv_weight=[64, 3, 7, 7]; conv_bias=1; conv_transposed=0"
    ) == ((64, 3, 7, 7), True, False)
    # Flags default to False when absent; weight is required.
    assert _parse_conv_details("conv_weight=[32, 32, 3, 3]") == (
        (32, 32, 3, 3),
        False,
        False,
    )
    assert _parse_conv_details("conv_transposed=1") is None
    assert _parse_conv_details("") is None


def test_conv_cost_matches_perf_model():
    # 3->64, 7x7, stride-2 conv on 224x224 -> 112x112 output, with bias.
    x, out = (1, 3, 224, 224), (1, 64, 112, 112)
    details = "conv_weight=[64, 3, 7, 7]; conv_bias=1; conv_transposed=0"
    category, maf_type, flops, bytes_moved = _conv_cost(details, [x], out, 2)
    assert category == "CONV"
    assert maf_type == "matrix"
    # prod(out) * 2 * prod(weight[1:]) + bias term.
    assert flops == 112 * 112 * 64 * 2 * (3 * 7 * 7) + (112 * 112 * 64)
    assert bytes_moved is not None


def test_conv_ops_from_meta_costs_every_instance():
    # Two live conv instances a meta forward would hand back: the ConvNeXt stem
    # (regular, biased) and a depthwise 7x7 block conv (groups == channels).
    meta_convs = [
        {
            "path": "convnext.embeddings.patch_embeddings",
            "input_shape": [1, 3, 224, 224],
            "output_shape": [1, 128, 56, 56],
            "weight_shape": [128, 3, 4, 4],
            "bias": True,
            "transposed": False,
        },
        {
            "path": "convnext.encoder.stages.0.layers.0.dwconv",
            "input_shape": [1, 128, 56, 56],
            "output_shape": [1, 128, 56, 56],
            "weight_shape": [128, 1, 7, 7],
            "bias": True,
            "transposed": False,
        },
    ]
    ops = conv_ops_from_meta(meta_convs, dtype="float32")
    assert len(ops) == 2
    assert all(op.modeled and op.category == "CONV" for op in ops)
    stem, dwconv = ops
    # prod(out) * 2 * prod(weight[1:]) + bias term, per CONV.flops_func.
    assert stem.flops == 128 * 56 * 56 * 2 * (3 * 4 * 4) + (128 * 56 * 56)
    assert dwconv.flops == 128 * 56 * 56 * 2 * (1 * 7 * 7) + (128 * 56 * 56)
    assert stem.output_elements == 128 * 56 * 56
    # The checkpoint's declared dtype drives the compute spec (fp32 here).
    assert stem.compute_spec == "matrix_fp32"
    # Entries missing geometry/shapes are skipped, not silently miscosted.
    assert conv_ops_from_meta([{"path": "x", "output_shape": [1, 8, 4, 4]}]) == []
    assert conv_ops_from_meta(None) == []


def test_dispatch_conv_routes_and_needs_geometry():
    x, out = (1, 3, 224, 224), (1, 64, 112, 112)
    details = "conv_weight=[64, 3, 7, 7]; conv_bias=0; conv_transposed=0"
    cost = _dispatch_op_cost("Conv2d", [x], out, 2, details)
    assert cost is not None and cost[0] == "CONV"
    # Without threaded geometry a conv stays unmodeled (not silently costed).
    assert _dispatch_op_cost("Conv2d", [x], out, 2, "") is None


def test_dispatch_sdpa_insufficient_operands_is_unmodeled():
    assert (
        _dispatch_op_cost(
            "scaled_dot_product_attention",
            [(1, 32, 2048, 128)],
            (1, 32, 2048, 128),
            2,
        )
        is None
    )


def test_dispatch_sdpa_requires_four_dimensional_operands():
    # A flattened (seq, head_dim) operand -- e.g. a batch/head axis already
    # folded away -- cannot be read as [B, heads, seq, head_dim].
    assert (
        _dispatch_op_cost(
            "scaled_dot_product_attention",
            [(2048, 128), (2048, 128), (2048, 128)],
            (1, 32, 2048, 128),
            2,
        )
        is None
    )


def test_dispatch_normalization_rmsnorm_subclass():
    category, maf_type, flops, bytes_moved = _dispatch_op_cost(
        "LlamaRMSNorm", [(2, 10, 4096)], (2, 10, 4096), 2
    )
    assert category == "Normalization (LlamaRMSNorm)"
    assert maf_type == "vector"
    assert flops > 0
    assert bytes_moved is not None


def test_dispatch_normalization_generic_layernorm_has_bias_cost():
    category, maf_type, flops, bytes_moved = _dispatch_op_cost(
        "LayerNorm", [(2, 10, 4096)], (2, 10, 4096), 2
    )
    assert category == "Normalization (LayerNorm)"
    assert maf_type == "vector"
    assert flops > 0
    assert bytes_moved is not None


def test_dispatch_normalization_zero_channels_is_unmodeled():
    assert _dispatch_op_cost("RMSNorm", [(2, 10, 0)], (2, 10, 0), 2) is None


def test_dispatch_softmax():
    category, maf_type, flops, bytes_moved = _dispatch_op_cost(
        "Softmax", [(2, 32, 10, 10)], (2, 32, 10, 10), 2
    )
    assert category == "Softmax"
    assert maf_type == "vector"
    assert flops > 0


def test_dispatch_softmax_without_output_shape_is_unmodeled():
    assert _dispatch_op_cost("Softmax", [(2, 32, 10, 10)], (), 2) is None


def test_dispatch_unary_elementwise():
    category, _, flops, bytes_moved = _dispatch_op_cost(
        "Silu", [(2, 10, 4096)], (2, 10, 4096), 2
    )
    assert category == "UnaryElementwise"
    assert flops == 2 * 10 * 4096


def test_dispatch_unary_elementwise_without_output_shape_is_unmodeled():
    assert _dispatch_op_cost("Silu", [(2, 10, 4096)], (), 2) is None


def test_dispatch_binary_elementwise():
    category, _, flops, _ = _dispatch_op_cost(
        "elementwise_mul", [(2, 10, 4096), (2, 10, 4096)], (2, 10, 4096), 2
    )
    assert category == "BinaryElementwise"
    assert flops == 2 * 10 * 4096


def test_dispatch_binary_elementwise_scalar_operand_defaults_to_one_element():
    category, _, _, bytes_moved = _dispatch_op_cost(
        "multiply", [(2, 10, 4096)], (2, 10, 4096), 2
    )
    assert category == "BinaryElementwise"
    assert bytes_moved is not None


def test_dispatch_reduce():
    category, _, flops, bytes_moved = _dispatch_op_cost(
        "sum", [(2, 10, 4096)], (2, 10), 2
    )
    assert category == "Reduce"
    assert flops > 0
    assert bytes_moved is not None


def test_dispatch_reduce_without_an_input_shape_is_unmodeled():
    assert _dispatch_op_cost("sum", [], (2, 10), 2) is None
    assert _dispatch_op_cost("sum", [None], (2, 10), 2) is None


def test_dispatch_unmapped_computation_returns_none():
    assert _dispatch_op_cost("Reshape", [(2, 10, 4096)], (2, 10, 4096), 2) is None


# _engine_head_cost / _moe_cost / _engine_boundary_cost: vLLM/SGLang engine
# boundaries (ParallelLMHead, FusedMoE) that shape_inference has no rule for,
# costed from the architecture's own dims instead of a node shape.


def test_engine_head_cost_vocab_projection_gemm():
    dims = {"B": 2, "S": 10, "H": 4096, "V": 32000}
    category, maf_type, flops, bytes_moved, output_elements = _engine_head_cost(
        dims, bpe=2
    )
    assert category == "GEMM"
    assert maf_type == "matrix"
    assert flops == 2 * 20 * 32000 * 4096
    assert bytes_moved is not None
    assert output_elements == 20 * 32000


def test_engine_head_cost_missing_vocab_dim_returns_none():
    assert _engine_head_cost({"B": 2, "S": 10, "H": 4096}, bpe=2) is None


def test_moe_cost_scales_by_experts_per_tok_not_num_experts():
    dims = {"B": 1, "S": 10, "H": 4096}
    category, maf_type, flops, bytes_moved, output_elements = _moe_cost(
        dims, top_k=2, intermediate_size=1024, bpe=2
    )
    assert category == "MoE"
    assert maf_type == "matrix"
    active_tokens = 10 * 2
    expected_gate_up = 2 * active_tokens * (2 * 1024) * 4096
    expected_down = 2 * active_tokens * 4096 * 1024
    assert flops == expected_gate_up + expected_down
    assert bytes_moved is not None
    assert output_elements == 10 * 4096


def test_moe_cost_missing_top_k_or_intermediate_returns_none():
    dims = {"B": 1, "S": 10, "H": 4096}
    assert _moe_cost(dims, top_k=None, intermediate_size=1024, bpe=2) is None
    assert _moe_cost(dims, top_k=2, intermediate_size=None, bpe=2) is None


def test_introspect_weight_bpe_reads_quant_config():
    # MXFP4 (quark) -> 4-bit element + e8m0 scale per 32.
    assert (
        _introspect_weight_bpe(
            {
                "quant_method": "quark",
                "global_quant_config": {"weight": {"dtype": "fp4"}},
            }
        )
        == 0.53125
    )
    assert _introspect_weight_bpe({"quant_method": "fp8", "fmt": "e4m3"}) == 1.0
    assert (
        _introspect_weight_bpe(
            {
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
            }
        )
        == 0.5
    )
    assert _introspect_weight_bpe(None) is None
    assert _introspect_weight_bpe({"quant_method": "unknown"}) is None


def test_moe_cost_uses_introspected_weight_bpe():
    dims = {"B": 1, "S": 10, "H": 4096}
    _RT.reset()
    try:
        _, _, _, bytes_bf16, _ = _moe_cost(dims, top_k=2, intermediate_size=1024, bpe=2)
        _RT.weight_bpe = 0.5  # as if MXFP4 weights were introspected
        _, _, _, bytes_fp4, _ = _moe_cost(dims, top_k=2, intermediate_size=1024, bpe=2)
    finally:
        _RT.reset()
    # Weight reads dominate the GEMM, so a smaller weight dtype cuts total bytes.
    assert bytes_fp4 < bytes_bf16


def test_moe_cost_grouped_gemm_scales_weight_bytes_by_experts_touched():
    # active_tokens = B*S*top_k = 1*10*2 = 20.
    dims = {"B": 1, "S": 10, "H": 4096}
    _RT.reset()
    try:
        # No num_experts -> G falls back to 1 (prior single-weight cost).
        _, _, flops_g1, bytes_g1, _ = _moe_cost(dims, 2, 1024, bpe=2)
        # num_experts > active_tokens -> G = active_tokens (20) distinct weights read.
        _RT.moe_num_experts = 256
        _, _, flops_g, bytes_g, _ = _moe_cost(dims, 2, 1024, bpe=2)
    finally:
        _RT.reset()
    # FLOPs are unchanged (only the selected token-expert pairs compute).
    assert flops_g == flops_g1
    # Reading G=20 expert weight matrices moves far more than a single one.
    assert bytes_g > bytes_g1


def test_moe_cost_adds_dispatch_combine_allreduce_only_when_ep():
    dims = {"B": 1, "S": 10, "H": 4096}
    _RT.reset()
    try:
        _RT.moe_num_experts = 256
        _, _, flops_ep1, bytes_ep1, _ = _moe_cost(dims, 2, 1024, bpe=2)
        _RT.ep = 8
        _, _, flops_ep8, bytes_ep8, _ = _moe_cost(dims, 2, 1024, bpe=2)
    finally:
        _RT.reset()
    # One GPU owns floor(routed_rows / EP) of the expert GEMMs. 20 rows
    # across 8 ranks is 2 rows here, not a fractional 2.5.
    local_tokens = (10 * 2) // 8
    assert flops_ep8 == pytest.approx(flops_ep1 * local_tokens / (10 * 2))
    assert bytes_ep8 > 2 * local_tokens * 4096 * 2


def test_attention_boundary_cost_honors_sliding_window_mix():
    dims = {"B": 1, "S": 1, "past_len": 4096}
    base = dict(attn_heads=8, attn_kv_heads=2, attn_head_dim=64, n_layers=4)
    _RT.reset()
    try:
        # All-global (no sliding window) vs mixed (3 of 4 layers windowed at 512).
        for k, v in base.items():
            setattr(_RT, k, v)
        _RT.sliding_window = None
        _RT.n_local_layers = 0
        cat, maf, flops_global, bytes_global = _attention_boundary_cost(dims, bpe=2)
        _RT.sliding_window = 512
        _RT.n_local_layers = 3
        _, _, flops_windowed, bytes_windowed = _attention_boundary_cost(dims, bpe=2)
    finally:
        _RT.reset()
    assert cat == "Attention" and maf == "matrix"
    # Windowing the local layers shrinks the effective KV length -> fewer FLOPs/bytes.
    assert flops_windowed < flops_global
    assert bytes_windowed < bytes_global


def test_moe_cost_missing_hidden_dim_returns_none():
    assert _moe_cost({"B": 1, "S": 10}, top_k=2, intermediate_size=1024, bpe=2) is None


def test_engine_boundary_cost_dispatches_by_class_name_marker():
    dims = {"B": 1, "S": 10, "H": 4096, "V": 32000}
    assert _engine_boundary_cost("ParallelLMHead", dims, 2, 1024, 2) is not None
    assert _engine_boundary_cost("FusedMoEFactory", dims, 2, 1024, 2) is not None


def test_engine_boundary_cost_unrelated_class_returns_none():
    dims = {"B": 1, "S": 10, "H": 4096, "V": 32000}
    assert _engine_boundary_cost("Linear", dims, 2, 1024, 2) is None
    assert _engine_boundary_cost("", dims, 2, 1024, 2) is None


# _split_shape_text / _project_op
def test_split_shape_text_without_brackets_has_no_dtype():
    # A bracketless shape text is unexpected from the real graph builder, but
    # must not be mistaken for one with a trailing dtype suffix.
    dims, dtype = _split_shape_text("B, S, H")
    assert dims == parse_shape_dims("B, S, H")
    assert dtype is None


def test_project_op_uses_the_engine_boundary_cost_when_the_class_name_matches():
    dims = {"B": 2, "S": 10, "H": 4096, "V": 32000}
    op = _project_op(
        "n1",
        "section",
        "Linear",
        "ParallelLMHead",
        "nn_module",
        "[B, S, H] bfloat16",
        [],
        dims,
        instances=1,
        gpu_arch=None,
        moe_top_k=2,
        moe_intermediate_size=1024,
    )
    assert op.modeled is True
    assert op.category == "GEMM"
    assert op.output_elements == 20 * 32000


def test_project_op_records_an_unresolved_output_dim():
    op = _project_op(
        "n1",
        "section",
        "Linear",
        "Linear",
        "nn_module",
        "[B, S, unknown_dim] bfloat16",
        [["B", "S", "H"]],
        {"B": 2, "S": 10, "H": 4096},
        instances=1,
        gpu_arch=None,
        moe_top_k=None,
        moe_intermediate_size=None,
    )
    assert op.modeled is False
    assert "unresolved output dim" in op.reason


def test_project_op_records_an_unresolved_input_dim():
    op = _project_op(
        "n1",
        "section",
        "Linear",
        "Linear",
        "nn_module",
        "[B, S, H] bfloat16",
        [["B", "S", "unknown_dim"]],
        {"B": 2, "S": 10, "H": 4096},
        instances=1,
        gpu_arch=None,
        moe_top_k=None,
        moe_intermediate_size=None,
    )
    assert op.modeled is False
    assert "unresolved input dim" in op.reason


def test_project_op_costs_an_embedding_lookup_as_gathered_rows():
    op = _project_op(
        "embed_tokens",
        "embed_tokens",
        "Embedding",
        "Embedding",
        "nn_module",
        "[B, S, H] bfloat16",
        [["B", "S"]],
        {"B": 2, "S": 4, "H": 8},
        instances=1,
        gpu_arch=None,
        moe_top_k=None,
        moe_intermediate_size=None,
    )
    assert op.modeled is True
    assert op.category == "DataMovement"
    assert op.flops == 0
    # 8 tokens, hidden 8, bf16 rows read and written, plus int64 indices.
    assert op.bytes == 8 * 8 * 2 + 8 * 8 * 2 + 8 * 8


@pytest.mark.parametrize(
    "computation",
    [
        "aten::embedding",
        "EmbeddingBag",
        "Token Embedding",
        "Indexing",
        "Index select",
        "aten::index",
    ],
)
def test_project_op_costs_common_embedding_labels(computation):
    op = _project_op(
        "embed",
        "embed",
        computation,
        computation,
        "aten_op",
        "[B, S, H] bfloat16",
        [["B", "S"]],
        {"B": 2, "S": 4, "H": 8},
        instances=1,
        gpu_arch=None,
        moe_top_k=None,
        moe_intermediate_size=None,
    )
    assert op.modeled is True
    assert op.category == "DataMovement"
    assert op.flops == 0
    assert op.bytes == 8 * 8 * 2 + 8 * 8 * 2 + 8 * 8


def test_project_op_costs_indexing_a_wider_tail_as_one_row():
    # ``x[idx]`` on a rank-3 table keeps the tail, so one index selects H*D.
    op = _project_op(
        "idx",
        "idx",
        "Indexing",
        "Indexing",
        "aten_op",
        "[B, S, H, D] bfloat16",
        [["B", "S"]],
        {"B": 2, "S": 4, "H": 8, "D": 4},
        instances=1,
        gpu_arch=None,
        moe_top_k=None,
        moe_intermediate_size=None,
    )
    assert op.modeled is True
    assert op.flops == 0
    assert op.bytes == 2 * 8 * 32 * 2 + 8 * 8


def test_project_op_leaves_a_general_gather_unmodeled():
    op = _project_op(
        "g",
        "g",
        "Gather",
        "Gather",
        "aten_op",
        "[B, S, H] bfloat16",
        [["B", "S"]],
        {"B": 2, "S": 4, "H": 8},
        instances=1,
        gpu_arch=None,
        moe_top_k=None,
        moe_intermediate_size=None,
    )
    assert op.modeled is False


def test_embedding_lookup_is_a_leaf_the_opaque_tile_must_not_recost():
    assert _is_leaf_computation("Embedding") is True
    assert _is_leaf_computation("Indexing") is True


def test_embedding_lookup_without_a_width_is_unmodeled():
    op = _project_op(
        "embed",
        "embed",
        "Embedding",
        "Embedding",
        "nn_module",
        "[B, S, H]",
        [["B", "S"]],
        {"B": 2, "S": 4, "H": 8},
        instances=1,
        gpu_arch=None,
        moe_top_k=None,
        moe_intermediate_size=None,
    )
    assert op.modeled is False
    assert _embedding_lookup_cost([(2, 4)], (), 2) is None


def test_embedding_lookup_falls_back_when_no_input_is_an_index_vector():
    # A missing input, a tensor at least as wide as the output, and an input
    # that does not divide the output are not index vectors. The row count is
    # then the output with its last axis removed.
    op = _project_op(
        "idx",
        "idx",
        "Indexing",
        "Indexing",
        "aten_op",
        "[B, H] bfloat16",
        [[], ["B", "H", "D"], ["K"]],
        {"B": 2, "H": 4, "D": 8, "K": 3},
        instances=1,
        gpu_arch=None,
        moe_top_k=None,
        moe_intermediate_size=None,
    )
    assert op.modeled is True
    assert op.bytes == 2 * 2 * 4 * 2 + 2 * 8

    vector = _project_op(
        "idx",
        "idx",
        "Indexing",
        "Indexing",
        "aten_op",
        "[N] bfloat16",
        [["N"]],
        {"N": 8},
        instances=1,
        gpu_arch=None,
        moe_top_k=None,
        moe_intermediate_size=None,
    )
    assert vector.bytes == 2 * 8 * 2 + 8


def test_project_op_records_an_unmapped_computation():
    op = _project_op(
        "n1",
        "section",
        "Reshape",
        "Reshape",
        "nn_module",
        "[B, S, H] bfloat16",
        [["B", "S", "H"]],
        {"B": 2, "S": 10, "H": 4096},
        instances=1,
        gpu_arch=None,
        moe_top_k=None,
        moe_intermediate_size=None,
    )
    assert op.modeled is False
    assert "no PerfModel mapping" in op.reason


# RooflineProjection.totals()


def test_totals_aggregates_modeled_and_unmodeled():
    projection = RooflineProjection(batch_size=1, seq_len=10)
    projection.ops = [
        RooflineOp(
            name="a",
            section="s",
            computation="Linear",
            operation="nn_module",
            category="GEMM",
            modeled=True,
            reason=None,
            output_elements=100,
            instances=2,
            flops=1000.0,
            bytes=500,
            roofline_time_us=10.0,
            roofline_bound="COMPUTE_BOUND",
        ),
        RooflineOp(
            name="b",
            section="s",
            computation="Embedding",
            operation="nn_module",
            category=None,
            modeled=False,
            reason="no PerfModel mapping",
            output_elements=50,
            instances=1,
        ),
    ]
    totals = projection.totals()
    assert totals["total_flops"] == 2000.0
    assert totals["total_bytes"] == 1000
    assert totals["total_roofline_time_us"] == 20.0
    assert totals["compute_bound_time_us"] == 20.0
    assert totals["memory_bound_time_us"] == 0
    assert totals["modeled_op_count"] == 1
    assert totals["unmodeled_op_count"] == 1
    # 2 instances * 100 elems modeled vs. 1 instance * 50 elems unmodeled.
    assert totals["modeled_output_element_share"] == pytest.approx(200 / 250)


def test_totals_empty_projection_has_no_coverage_share():
    assert RooflineProjection().totals()["modeled_output_element_share"] is None


# Integration: project_roofline against the custom_model fixture


def _load_fixture_spec():
    return load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.for_roofline(),
    )


def test_build_projection_dims_overrides_batch_and_seq():
    spec = _load_fixture_spec()
    dims = build_projection_dims(spec, batch_size=4, seq_len=128, extra_dims={"foo": 7})
    assert dims["B"] == 4
    assert dims["S"] == 128
    assert dims["foo"] == 7


def test_project_roofline_fixture_coverage():
    spec = _load_fixture_spec()
    projection = project_roofline(spec, batch_size=1, seq_len=2048, gpu_arch=_ARCH)

    modeled_names = {op.name for op in projection.modeled_ops}
    assert any(name.endswith("q_proj:q_proj:0") for name in modeled_names)
    assert any(name.endswith("kv_proj:kv_proj:0") for name in modeled_names)
    assert any(name.endswith("router:router:0") for name in modeled_names)
    assert any(name.endswith("shared_expert:shared_expert:0") for name in modeled_names)
    assert "lm_head" in modeled_names
    # RMSNorm is drawn under its literal subclass name (CustomRMSNorm), not
    # the base class name -- this is the substring-matching case.
    rmsnorm_ops = [
        op
        for op in projection.modeled_ops
        if op.category and "Normalization" in op.category
    ]
    assert rmsnorm_ops

    embed = next(op for op in projection.modeled_ops if op.name == "embed_tokens")
    assert embed.category == "DataMovement"
    assert embed.flops == 0
    assert embed.bytes > 0

    totals = projection.totals()
    assert totals["total_flops"] > 0
    assert totals["total_roofline_time_us"] > 0
    assert 0 < totals["modeled_output_element_share"] <= 1


def test_project_roofline_scales_decoder_sections_by_num_hidden_layers():
    spec = _load_fixture_spec()
    projection = project_roofline(spec, batch_size=1, seq_len=2048)
    decoder_op = next(
        op for op in projection.ops if op.name.endswith("q_proj:q_proj:0")
    )
    assert decoder_op.instances == (spec.num_hidden_layers or 1)
    # lm_head sits outside the repeated decoder stack, so it runs once --
    # not once per layer the way a flat num_hidden_layers scale-up would.
    lm_head_op = next(op for op in projection.ops if op.name == "lm_head")
    assert lm_head_op.instances == 1
    embedding_op = next(op for op in projection.ops if op.name == "embed_tokens")
    assert embedding_op.instances == 1


def test_project_roofline_without_gpu_arch_has_no_roofline_time():
    spec = _load_fixture_spec()
    projection = project_roofline(spec, batch_size=1, seq_len=2048)
    for op in projection.modeled_ops:
        assert op.roofline_time_us is None
        assert op.flops is not None


# CLI


def test_parse_dim_overrides():
    assert roofline_cli.parse_dim_overrides(["past_len=1024", "foo=2"]) == {
        "past_len": 1024,
        "foo": 2,
    }


def test_parse_dim_overrides_rejects_malformed_entry():
    with pytest.raises(ValueError):
        roofline_cli.parse_dim_overrides(["not_a_kv_pair"])


def test_parse_dim_overrides_rejects_non_int_value():
    with pytest.raises(ValueError):
        roofline_cli.parse_dim_overrides(["foo=bar"])


def test_cli_end_to_end_writes_csv_and_json(tmp_path, capsys):
    csv_path = tmp_path / "roofline.csv"
    json_path = tmp_path / "roofline.json"
    exit_code = roofline_cli.main(
        [
            str(FIXTURES / "custom_model"),
            "--title",
            "Custom MLA MoE",
            "--no-meta-shapes",
            "--gpu-arch-name",
            "MI300X",
            "--output-csv",
            str(csv_path),
            "--output-json",
            str(json_path),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Roofline time" in captured.out
    assert csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["totals"]["total_flops"] > 0
    assert any(row["category"] == "GEMM" for row in payload["ops"])


def test_cli_unknown_gpu_arch_name_errors(capsys):
    with pytest.raises(SystemExit):
        roofline_cli.main(
            [
                str(FIXTURES / "custom_model"),
                "--no-meta-shapes",
                "--gpu-arch-name",
                "not_a_real_gpu",
            ]
        )
    assert "Unknown --gpu-arch-name" in capsys.readouterr().err


def test_cli_requires_a_checkpoint():
    with pytest.raises(SystemExit):
        roofline_cli.main([])


def test_cli_rejects_a_malformed_dim_override(capsys):
    with pytest.raises(SystemExit):
        roofline_cli.main(
            [
                str(FIXTURES / "custom_model"),
                "--no-meta-shapes",
                "--dim",
                "not_a_kv_pair",
            ]
        )
    assert "not_a_kv_pair" in capsys.readouterr().err


def test_cli_gpu_arch_json_path_takes_precedence(tmp_path):
    arch_path = tmp_path / "arch.json"
    arch_path.write_text(json.dumps(_ARCH), encoding="utf-8")
    exit_code = roofline_cli.main(
        [
            str(FIXTURES / "custom_model"),
            "--no-meta-shapes",
            "--gpu-arch-json-path",
            str(arch_path),
            "--seq-len",
            "64",
        ]
    )
    assert exit_code == 0


def test_cli_reports_an_architecture_load_failure_without_a_traceback(capsys):
    exit_code = roofline_cli.main(["/nonexistent/path/for/testing", "--no-meta-shapes"])
    assert exit_code == 1
    assert "Error loading architecture" in capsys.readouterr().err


def test_cli_tp_ep_pp_flags_wire_a_parallel_layout_and_warn_when_undeclared(capsys):
    # custom_model declares no base_model_tp_plan/base_model_ep_plan, so the
    # flags are expected to wire through to a real ParallelLayout build (not
    # just be accepted by argparse and ignored) and surface its "undeclared"
    # notes as warnings, the same as TraceLens_visualize_model_in_explorer.
    exit_code = roofline_cli.main(
        [
            str(FIXTURES / "custom_model"),
            "--title",
            "Custom MLA MoE",
            "--no-meta-shapes",
            "--tp",
            "2",
            "--ep",
            "2",
        ]
    )
    assert exit_code == 0
    stderr = capsys.readouterr().err
    assert (
        "Warning: No Hugging Face parallel plan was found in the model source."
        in stderr
    )


def test_cli_tp_1_does_not_build_a_parallel_layout(monkeypatch):
    # The default (every axis == 1) must not pay for / warn about a layout
    # build at all.
    called = []
    original = roofline_cli.build_parallel_layout
    monkeypatch.setattr(
        roofline_cli,
        "build_parallel_layout",
        lambda spec, degree: called.append(degree) or original(spec, degree),
    )
    exit_code = roofline_cli.main([str(FIXTURES / "custom_model"), "--no-meta-shapes"])
    assert exit_code == 0
    assert called == []


def test_cli_num_predicted_tokens_is_accepted():
    exit_code = roofline_cli.main(
        [
            str(FIXTURES / "custom_model"),
            "--no-meta-shapes",
            "--num-predicted-tokens",
            "2",
            "--seq-len",
            "8",
        ]
    )
    assert exit_code == 0


def test_cli_rejects_unknown_engine(capsys):
    with pytest.raises(SystemExit):
        roofline_cli.main(
            [
                str(FIXTURES / "custom_model"),
                "--no-meta-shapes",
                "--engine",
                "not_a_real_engine",
            ]
        )
    assert "invalid choice" in capsys.readouterr().err


# Custom-layer introspection: packed attention projections, relative attention,
# and tensor-/expert-parallel sharding of the role-based analytical costs.


@pytest.fixture
def _reset_rt():
    """Restore the module-global introspection singleton after a test mutates it."""
    yield
    _RT.reset()


def test_shard_divides_only_above_degree_one():
    assert _shard(64, 8) == 8
    assert _shard(64, 1) == 64
    assert _shard(8, 8) == 1
    # Never below one GPU's single head, even when the count is smaller than TP.
    assert _shard(3, 8) == 1


def test_attention_extra_ops_costs_packed_qkv_and_o_projections(_reset_rt):
    _RT.reset()
    _RT.attn_heads = 64
    _RT.attn_kv_heads = 8
    _RT.attn_head_dim = 128
    _RT.hidden_size = 6144
    ops = _attention_extra_ops(
        "decoder",
        "decoder/attn",
        instances=1,
        dims={"B": 1, "S": 1},
        bpe=2,
        dtype="bfloat16",
        gpu_arch=None,
    )
    names = {op.computation for op in ops}
    assert names == {"qkv_proj", "o_proj"}  # no d_rel -> no relative rows
    qkv = next(op for op in ops if op.computation == "qkv_proj")
    # Fused Q + K + V width: h_q*d + 2*h_kv*d, contracted over hidden.
    assert qkv.flops == 2 * (1 * (64 * 128 + 2 * 8 * 128) * 6144)
    assert all(op.modeled and op.category == "GEMM" for op in ops)


def test_attention_extra_ops_uses_latent_ranks_when_declared(_reset_rt):
    _RT.reset()
    _RT.attn_heads = 8
    _RT.attn_kv_heads = 1
    _RT.attn_head_dim = 128
    _RT.hidden_size = 1024
    _RT.q_lora_rank = 256
    _RT.kv_lora_rank = 512
    ops = _attention_extra_ops(
        "decoder",
        "decoder/attn",
        instances=1,
        dims={"B": 1, "S": 1},
        bpe=2,
        dtype="bfloat16",
        gpu_arch=None,
    )
    names = {op.computation for op in ops}
    assert {"q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj", "o_proj"} <= names
    assert "qkv_proj" not in names
    _RT.qk_nope_head_dim = 16
    _RT.qk_rope_head_dim = 16
    _RT.v_head_dim = 16
    latent = _attention_extra_ops(
        "decoder",
        "decoder/attn",
        instances=1,
        dims={"B": 1, "S": 1},
        bpe=2,
        dtype="bfloat16",
        gpu_arch=None,
    )
    q_b = next(op for op in latent if op.computation == "q_b_proj")
    # Query width is nope + rope, not the unrelated head_dim.
    assert q_b.flops == 2 * 1 * (8 * (16 + 16)) * 256


def test_index_topk_does_not_replace_a_layer_window(_reset_rt):
    _RT.reset()
    _RT.attn_heads = 4
    _RT.attn_kv_heads = 4
    _RT.attn_head_dim = 16
    _RT.n_layers = 1
    _RT.index_topk = 4
    dims = {"B": 1, "S": 1, "past_len": 32}
    windowed = _attention_boundary_cost(dims, bpe=2, window=8)[2]
    full = _attention_boundary_cost(dims, bpe=2, window=None)[2]
    assert windowed < full
    _RT.sliding_window = None
    _RT.variant_windows = ()
    indexed = _attention_boundary_cost(dims, bpe=2)[2]
    _RT.index_topk = None
    uncapped = _attention_boundary_cost(dims, bpe=2)[2]
    assert indexed < uncapped


def test_linear_widths_stay_with_their_owner():
    widths = _LinearWidths()
    widths.by_owner[("Attention", "proj")] = (8, 16)
    widths.by_owner[("MLP", "proj")] = (32, 64)
    assert widths.get("proj", "Attention") == (8, 16)
    assert widths.get("proj", "MLP") == (32, 64)
    assert widths.get("proj") is None


def test_constructor_width_keeps_the_formula_shard(_reset_rt):
    _RT.reset()
    _RT.tp = 4
    k, n = _adopt_linear_width(64, 128, 64, 32, "")
    assert (k, n) == (64, _shard(128, 4))
    k, n = _adopt_linear_width(128, 256, 32, 256, "")
    assert (k, n) == (_shard(128, 4), 256)
    k, n = _adopt_linear_width(64, 128, 64, 128, "ColumnParallelLinear")
    assert (k, n) == (64, _shard(128, 4))


def test_attention_boundary_uses_model_causal_policy_for_cold_prefill(_reset_rt):
    _RT.reset()
    _RT.attn_heads = 8
    _RT.attn_kv_heads = 8
    _RT.attn_head_dim = 64
    _RT.n_layers = 1
    _RT.causal_attention = True
    causal = _attention_boundary_cost({"B": 1, "S": 16}, bpe=2)[2]
    _RT.causal_attention = False
    bidirectional = _attention_boundary_cost({"B": 1, "S": 16}, bpe=2)[2]
    assert causal < bidirectional


def test_attention_extra_ops_emits_relative_attention_rows_when_declared(_reset_rt):
    _RT.reset()
    _RT.attn_heads = 64
    _RT.attn_kv_heads = 8
    _RT.attn_head_dim = 128
    _RT.hidden_size = 6144
    _RT.d_rel = 16
    _RT.rel_extent = 1024
    ops = _attention_extra_ops(
        "decoder",
        "decoder/attn",
        instances=1,
        dims={"B": 1, "S": 1},
        bpe=2,
        dtype="bfloat16",
        gpu_arch=None,
    )
    names = {op.computation for op in ops}
    assert {"rel_r_proj", "rel_bias"} <= names
    r = next(op for op in ops if op.computation == "rel_r_proj")
    assert r.flops == 2 * (1 * (64 * 16) * 6144)  # hidden -> h_q * d_rel


def test_attention_extra_ops_needs_attention_dims(_reset_rt):
    _RT.reset()  # no attn_heads/head_dim/hidden -> nothing to cost
    assert (
        _attention_extra_ops("s", "n", 1, {"B": 1, "S": 1}, 2, "bfloat16", None) == []
    )


def test_moe_cost_shards_intermediate_by_tp_and_tokens_by_ep(_reset_rt):
    dims = {"B": 1, "S": 8, "H": 4096}
    _RT.reset()
    whole = _moe_cost(dims, top_k=2, intermediate_size=1024, bpe=2)[2]
    _RT.tp = 8  # column-shard each expert's intermediate width
    tp_only = _moe_cost(dims, top_k=2, intermediate_size=1024, bpe=2)[2]
    assert tp_only == pytest.approx(whole / 8)
    _RT.reset()
    _RT.ep = 2  # expert-parallel halves the per-GPU routed-token work
    ep_only = _moe_cost(dims, top_k=2, intermediate_size=1024, bpe=2)[2]
    assert ep_only == pytest.approx(whole / 2)


def test_dense_ffn_costs_every_token_at_the_dense_width(_reset_rt):
    _RT.reset()
    _RT.dense_intermediate = 24576
    dims = {"B": 1, "S": 1, "H": 6144}
    flops = _dense_ffn_cost(dims, bpe=2)[2]
    # SwiGLU: gate_up is 2·intermediate wide, down projects back. Every token,
    # not a top-k subset of them.
    assert flops == 2 * (2 * 24576) * 6144 + 2 * 6144 * 24576


def test_unparsed_dense_mlp_is_not_costed_as_moe(_reset_rt):
    # Both MLP classes are absent from the registry, so each is one opaque tile.
    # The dense tile must use the dense width over every token; the MoE tile
    # keeps the routed-expert formula.
    _RT.reset()
    _RT.moe_top_k = 6
    _RT.moe_intermediate = 3072
    _RT.dense_intermediate = 24576
    _RT.hidden_size = 6144
    dims = {"B": 1, "S": 1, "H": 6144}
    dense = _project_op(
        "decoder/mlp",
        "decoder",
        "InklingDenseMLP",
        "InklingDenseMLP",
        "",
        "[B, S, H] bfloat16",
        [],
        dims,
        instances=2,
        gpu_arch=None,
        moe_top_k=6,
        moe_intermediate_size=3072,
        role="ffn",
    )
    moe = _project_op(
        "decoder/mlp",
        "decoder",
        "InklingMoE",
        "InklingMoE",
        "",
        "[B, S, H] bfloat16",
        [],
        dims,
        instances=64,
        gpu_arch=None,
        moe_top_k=6,
        moe_intermediate_size=3072,
        role="ffn",
    )
    assert dense.modeled and dense.category == "GEMM"
    assert moe.modeled and moe.category == "MoE"
    assert dense.flops == _dense_ffn_cost(dims, 2)[2]
    assert dense.flops != moe.flops


def test_moe_role_is_costed_without_an_ffn_alias(_reset_rt):
    _RT.reset()
    _RT.moe_top_k = 2
    _RT.moe_intermediate = 1024
    _RT.hidden_size = 4096
    op = _project_op(
        "decoder/experts",
        "decoder",
        "SparseMoeBlock",
        "SparseMoeBlock",
        "",
        "[B, S, H] bfloat16",
        [],
        {"B": 1, "S": 1, "H": 4096},
        instances=1,
        gpu_arch=None,
        moe_top_k=2,
        moe_intermediate_size=1024,
        role="moe",
    )
    assert op.modeled and op.category == "MoE"


def test_short_conv_leaf_uses_the_declared_kernel(_reset_rt):
    _RT.reset()
    _RT.sconv_kernel = 4
    _RT.hidden_size = 6144
    op = _project_op(
        "decoder/attn_sconv",
        "decoder",
        "InklingShortConv",
        "InklingShortConv",
        "",
        "[B, S, 768] bfloat16",
        [],
        {"B": 1, "S": 2, "H": 6144},
        instances=1,
        gpu_arch=None,
        moe_top_k=None,
        moe_intermediate_size=None,
    )
    assert op.modeled and op.category == "Conv1d"
    # Depthwise: 2 * tokens * channels * kernel, channels from the node shape.
    assert op.flops == 2 * (1 * 2) * 768 * 4


def test_fused_attention_adds_the_packed_kv_short_convs(_reset_rt):
    _RT.reset()
    _RT.attn_heads = 64
    _RT.attn_kv_heads = 8
    _RT.attn_head_dim = 128
    _RT.hidden_size = 6144
    _RT.sconv_kernel = 4
    ops = _attention_extra_ops(
        "decoder",
        "decoder/attn",
        instances=1,
        dims={"B": 1, "S": 1},
        bpe=2,
        dtype="bfloat16",
        gpu_arch=None,
    )
    by_name = {op.computation: op for op in ops}
    assert {"k_sconv", "v_sconv"} <= set(by_name)
    # KV channels are the (unsharded) kv heads times head dim.
    assert by_name["k_sconv"].flops == 2 * 1 * (8 * 128) * 4


def test_mtp_stack_is_one_dense_depth_per_predicted_token(_reset_rt):
    _RT.reset()
    _RT.attn_heads = 64
    _RT.attn_kv_heads = 8
    _RT.attn_head_dim = 128
    _RT.hidden_size = 6144
    _RT.dense_intermediate = 24576
    _RT.sconv_kernel = 4
    _RT.mtp_layers = 8
    _RT.n_layers = 1
    dims = {"B": 1, "S": 1, "H": 6144, "V": 1024}
    ops = _mtp_stack_ops(
        dims, num_predicted_tokens=3, gpu_arch=None, activation=(2, "bfloat16")
    )
    assert {op.section for op in ops} == {"mtp/0", "mtp/1", "mtp/2"}
    assert any(op.computation == "mlp" and op.category == "GEMM" for op in ops)
    assert any(op.computation == "input_proj" for op in ops)
    assert not any(op.category == "MoE" for op in ops)
    capped = _mtp_stack_ops(
        dims, num_predicted_tokens=100, gpu_arch=None, activation=(2, "bfloat16")
    )
    assert {op.section for op in capped} == {f"mtp/{index}" for index in range(8)}
    assert _mtp_stack_ops(dims, 0, None, (2, "bfloat16")) == []
    assert {op.cost_source for op in ops} == {"fallback"}


_DRAFT_MOE_SOURCE = """
class RMSNorm:
    def __init__(self, hidden_size):
        pass

class Attention:
    def __init__(self, config):
        pass

class FusedMoE:
    def __init__(self, num_experts):
        pass

class DecoderLayer:
    def __init__(self, config):
        self.self_attn = Attention(config)
        self.mlp = FusedMoE(num_experts=8)

class MultiTokenPredictorLayer:
    def __init__(self, config):
        self.enorm = RMSNorm(config.hidden_size)
        self.hnorm = RMSNorm(config.hidden_size)
        self.eh_proj = Linear(config.hidden_size * 2, config.hidden_size)
        self.mtp_block = DecoderLayer(config)

class Linear:
    def __init__(self, in_features, out_features):
        pass
"""

_DRAFT_DENSE_SOURCE = """
class RMSNorm:
    def __init__(self, hidden_size):
        pass

class Attention:
    def __init__(self, config):
        pass

class DenseMLP:
    def __init__(self, config):
        pass

class Linear:
    def __init__(self, in_features, out_features):
        pass

class NextNLayer:
    def __init__(self, config):
        self.enorm = RMSNorm(config.hidden_size)
        self.hnorm = RMSNorm(config.hidden_size)
        self.eh_proj = Linear(config.hidden_size * 2, config.hidden_size)
        self.self_attn = Attention(config)
        self.mlp = DenseMLP(config)
"""


def _draft_spec(source: str):
    from types import SimpleNamespace

    from TraceLens.ModelUtils.ast_analyze import analyze_source

    analysis = analyze_source(source, config={"hidden_size": 6144})
    return SimpleNamespace(class_registry=analysis.class_registry, layer_variants=[])


def test_speculative_class_pattern_matches_engine_draft_names():
    from TraceLens.ModelUtils.ast_analyze import SPECULATIVE_CLASS_RE

    assert SPECULATIVE_CLASS_RE.search("DeepseekV3MultiTokenPredictorLayer")
    assert SPECULATIVE_CLASS_RE.search("NextNLayer")
    assert SPECULATIVE_CLASS_RE.search("EagleDraft")
    assert SPECULATIVE_CLASS_RE.search("SpeculativeDecoder")
    assert not SPECULATIVE_CLASS_RE.search("DecoderLayer")


def test_mtp_stack_follows_a_parsed_moe_draft_layer(_reset_rt):
    _RT.reset()
    _RT.attn_heads = 64
    _RT.attn_kv_heads = 8
    _RT.attn_head_dim = 128
    _RT.hidden_size = 6144
    _RT.dense_intermediate = 24576
    _RT.moe_intermediate = 1024
    _RT.moe_top_k = 2
    _RT.moe_num_experts = 8
    _RT.n_layers = 1
    dims = {"B": 1, "S": 1, "H": 6144, "V": 1024}
    ops = _mtp_stack_ops(
        dims,
        num_predicted_tokens=1,
        gpu_arch=None,
        activation=(2, "bfloat16"),
        spec=_draft_spec(_DRAFT_MOE_SOURCE),
    )
    by_name = {op.computation: op for op in ops}
    assert {"enorm", "hnorm", "eh_proj"} <= set(by_name)
    assert by_name["mlp"].category == "MoE"
    assert by_name["mlp"].cost_source == "config"
    assert "input_proj" not in by_name
    assert "router" in by_name


def test_mtp_stack_follows_a_parsed_dense_draft_layer(_reset_rt):
    _RT.reset()
    _RT.attn_heads = 64
    _RT.attn_kv_heads = 8
    _RT.attn_head_dim = 128
    _RT.hidden_size = 6144
    _RT.dense_intermediate = 24576
    _RT.moe_intermediate = 1024
    _RT.moe_top_k = 2
    _RT.n_layers = 1
    dims = {"B": 1, "S": 1, "H": 6144, "V": 1024}
    ops = _mtp_stack_ops(
        dims,
        num_predicted_tokens=1,
        gpu_arch=None,
        activation=(2, "bfloat16"),
        spec=_draft_spec(_DRAFT_DENSE_SOURCE),
    )
    mlp = next(op for op in ops if op.computation == "mlp")
    assert mlp.category == "GEMM"
    assert mlp.cost_source == "config"
    assert not any(op.category == "MoE" for op in ops)


def test_mtp_draft_reuses_the_backbone_tail_variant_when_the_ffn_is_ambiguous(
    _reset_rt,
):
    from types import SimpleNamespace

    _RT.reset()
    _RT.attn_heads = 8
    _RT.attn_kv_heads = 8
    _RT.attn_head_dim = 64
    _RT.hidden_size = 1024
    _RT.dense_intermediate = 2048
    _RT.moe_intermediate = 512
    _RT.moe_top_k = 2
    _RT.moe_num_experts = 4
    _RT.n_layers = 1
    decoder = SimpleNamespace(
        name="DecoderLayer",
        init_assignments={"self_attn": "Attention", "mlp": "FusedMoE"},
        init_assignment_options={"mlp": ["DenseMLP", "FusedMoE"]},
    )
    draft = SimpleNamespace(
        name="MultiTokenPredictorLayer",
        init_assignments={
            "enorm": "RMSNorm",
            "hnorm": "RMSNorm",
            "eh_proj": "Linear",
            "mtp_block": "DecoderLayer",
        },
        init_assignment_options={},
    )
    spec = SimpleNamespace(
        class_registry={
            "MultiTokenPredictorLayer": draft,
            "DecoderLayer": decoder,
        },
        layer_variants=[
            LayerVariant(
                label="dense",
                count=2,
                attention_label="Attn",
                ffn_label="DenseMLP",
                ffn_class="DenseMLP",
                ffn_attr="mlp",
                layer_indices=[0, 1],
            ),
            LayerVariant(
                label="moe",
                count=2,
                attention_label="Attn",
                ffn_label="FusedMoE",
                ffn_class="FusedMoE",
                ffn_attr="mlp",
                layer_indices=[2, 3],
            ),
        ],
    )
    ops = _mtp_stack_ops(
        {"B": 1, "S": 1, "H": 1024, "V": 128},
        num_predicted_tokens=1,
        gpu_arch=None,
        activation=(2, "bfloat16"),
        spec=spec,
    )
    mlp = next(op for op in ops if op.computation == "mlp")
    assert mlp.category == "MoE"
    assert mlp.cost_source == "config"


def test_attention_boundary_cost_shards_heads_by_tp(_reset_rt):
    _RT.reset()
    _RT.attn_heads = 64
    _RT.attn_kv_heads = 8
    _RT.attn_head_dim = 128
    _RT.n_layers = 1
    whole = _attention_boundary_cost({"B": 1, "S": 1}, bpe=2)[2]
    _RT.tp = 8
    sharded = _attention_boundary_cost({"B": 1, "S": 1}, bpe=2)[2]
    assert sharded == pytest.approx(whole / 8)


def test_unparsed_dense_variant_tile_is_named_for_the_dense_class():
    component = BlockComponent("mlp", "InklingMoE", "ffn", "MoE")
    variant = LayerVariant(
        label="dense",
        count=2,
        attention_label="Attn",
        ffn_label="InklingDenseMLP",
        ffn_class="InklingDenseMLP",
        ffn_attr="mlp",
    )
    node = {
        "id": "mlp",
        "label": "MoE",
        "attrs": [
            {"key": "class_name", "value": "InklingMoE"},
            {"key": "role", "value": "ffn"},
        ],
    }
    _stamp_variant_ffn_identity(node, component, variant)
    attrs = {item["key"]: item["value"] for item in node["attrs"]}
    assert node["label"] == "InklingDenseMLP"
    assert attrs["class_name"] == "InklingDenseMLP"
    assert attrs["role"] == "ffn"


def test_shard_heads_keeps_the_remainder_on_the_busiest_gpu():
    assert _shard_heads(64, 8) == 8
    assert _shard_heads(20, 8) == 3
    assert _shard_heads(3, 8) == 1
    assert _shard(20, 8) == 2


def test_latent_attention_uses_one_kv_vector(_reset_rt):
    _RT.reset()
    _RT.attn_heads = 8
    _RT.attn_kv_heads = 8
    _RT.attn_head_dim = 128
    _RT.n_layers = 1
    full = _attention_boundary_cost({"B": 1, "S": 4}, bpe=2)[2]
    _RT.kv_lora_rank = 64
    _RT.qk_nope_head_dim = 32
    _RT.qk_rope_head_dim = 16
    _RT.v_head_dim = 32
    latent = _attention_boundary_cost({"B": 1, "S": 4}, bpe=2)[2]
    assert latent < full


def test_indexer_rows_cover_query_key_score_and_topk(_reset_rt):
    _RT.reset()
    _RT.attn_heads = 4
    _RT.attn_kv_heads = 4
    _RT.attn_head_dim = 32
    _RT.hidden_size = 64
    _RT.index_n_heads = 2
    _RT.index_head_dim = 8
    _RT.index_topk = 4
    ops = _attention_extra_ops(
        "decoder",
        "decoder/attn",
        instances=1,
        dims={"B": 1, "S": 8},
        bpe=2,
        dtype="bfloat16",
        gpu_arch=None,
    )
    names = {op.computation for op in ops}
    assert {"indexer_q", "indexer_k", "indexer_score", "indexer_topk"} <= names


def test_variant_windows_are_costed_separately(_reset_rt):
    _RT.reset()
    _RT.attn_heads = 4
    _RT.attn_kv_heads = 4
    _RT.attn_head_dim = 16
    _RT.n_layers = 2
    _RT.sliding_window = 2
    _RT.variant_windows = (("sliding_attention", 1, 2), ("full_attention", 1, None))
    dims = {"B": 1, "S": 1, "past_len": 16}
    sliding = _attention_boundary_cost(dims, bpe=2, window=2)[2]
    full = _attention_boundary_cost(dims, bpe=2, window=None)[2]
    mixed = _attention_boundary_cost(dims, bpe=2)[2]
    assert sliding < full
    assert mixed == pytest.approx((sliding + full) / 2)


def test_functional_matmul_charges_weight_bytes_on_the_parameter(_reset_rt):
    _RT.reset()
    _RT.weight_bpe = 0.5
    _category, _maf, _flops, narrow = _dispatch_op_cost(
        "MatMul",
        [(2, 10, 64), (64, 128)],
        (2, 10, 128),
        2,
        weight_bpe=0.5,
    )
    _category, _maf, _flops, wide = _dispatch_op_cost(
        "MatMul",
        [(2, 10, 64), (2, 64, 128)],
        (2, 10, 128),
        2,
        weight_bpe=0.5,
    )
    assert narrow < wide
    _category, _maf, _flops, left = _dispatch_op_cost(
        "MatMul",
        [(64, 128), (2, 128, 10)],
        (2, 64, 10),
        2,
        weight_bpe=0.5,
    )
    _category, _maf, _flops, both = _dispatch_op_cost(
        "MatMul",
        [(2, 64, 128), (2, 128, 10)],
        (2, 64, 10),
        2,
        weight_bpe=0.5,
    )
    assert left < both


def test_linear_bias_flag_adds_the_bias_flops():
    plain = _dispatch_op_cost("Linear", [(2, 10, 16)], (2, 10, 8), 2)[2]
    biased = _dispatch_op_cost(
        "Linear", [(2, 10, 16)], (2, 10, 8), 2, details="bias=1"
    )[2]
    assert biased == plain + 20 * 8


def test_weight_peak_does_not_apply_to_attention(_reset_rt):
    _RT.reset()
    _RT.weight_bpe = 0.53125
    op = RooflineOp(
        name="attn",
        section="decoder",
        computation="attn",
        operation="",
        category=None,
        modeled=False,
        reason=None,
        output_elements=1,
        instances=1,
    )
    finished = _finish_op(
        op,
        "Attention",
        "matrix",
        100.0,
        10.0,
        "bfloat16",
        {"max_achievable_tflops": {"matrix_fp4": 1000, "matrix_bf16": 100}},
    )
    assert finished.compute_spec == "matrix_bf16"


def test_moe_extras_skip_leaves_already_in_the_graph(_reset_rt):
    _RT.reset()
    _RT.moe_num_experts = 4
    _RT.moe_top_k = 2
    _RT.moe_intermediate = 32
    _RT.num_shared_experts = 1
    _RT.hidden_size = 16
    _RT.dense_intermediate = 32
    ops = _moe_extra_ops(
        "decoder",
        "decoder/mlp",
        1,
        {"B": 1, "S": 1, "H": 16},
        2,
        "bfloat16",
        None,
        present=frozenset({"router", "shared"}),
    )
    names = {op.computation for op in ops}
    assert "router" not in names
    assert "shared_expert" not in names
    assert "combine" in names


def test_missing_engine_source_is_recorded(_reset_rt):
    from types import SimpleNamespace

    _populate_introspection(
        SimpleNamespace(
            engine="vllm",
            class_registry={},
            code_paths=[],
            raw_config={},
            num_experts_per_tok=None,
            moe_intermediate_size=None,
            intermediate_size=None,
            num_experts=None,
            num_shared_experts=None,
            num_attention_heads=None,
            num_key_value_heads=None,
            head_dim=None,
            hidden_size=None,
            num_hidden_layers=None,
            positional_encoding=None,
            architectures=[],
            model_type="",
            parallel_layout=None,
        )
    )
    assert _RT.engine_miss
    assert "vllm" in _RT.engine_miss


def test_draft_widths_come_from_the_draft_class(_reset_rt):
    from types import SimpleNamespace

    _RT.reset()
    _RT.attn_heads = 4
    _RT.attn_kv_heads = 4
    _RT.attn_head_dim = 16
    _RT.hidden_size = 64
    _RT.dense_intermediate = 128
    _RT.moe_intermediate = 1024
    _RT.moe_top_k = 2
    _RT.moe_num_experts = 4
    _RT.n_layers = 1
    _RT.module_dims = SimpleNamespace(
        scalar_by_class={"DecoderLayer": {"moe_intermediate_size": 32}}
    )
    decoder = SimpleNamespace(
        name="DecoderLayer",
        init_assignments={"self_attn": "Attention", "mlp": "FusedMoE"},
        init_assignment_options={},
    )
    draft = SimpleNamespace(
        name="MultiTokenPredictorLayer",
        init_assignments={
            "enorm": "RMSNorm",
            "hnorm": "RMSNorm",
            "eh_proj": "Linear",
            "mtp_block": "DecoderLayer",
        },
        init_assignment_options={},
    )
    spec = SimpleNamespace(
        class_registry={
            "MultiTokenPredictorLayer": draft,
            "DecoderLayer": decoder,
            "FusedMoE": SimpleNamespace(name="FusedMoE", init_assignments={}),
        },
        layer_variants=[],
    )
    narrow = _mtp_stack_ops(
        {"B": 1, "S": 1, "H": 64, "V": 32},
        num_predicted_tokens=1,
        gpu_arch=None,
        activation=(2, "bfloat16"),
        spec=spec,
    )
    _RT.module_dims = None
    wide = _mtp_stack_ops(
        {"B": 1, "S": 1, "H": 64, "V": 32},
        num_predicted_tokens=1,
        gpu_arch=None,
        activation=(2, "bfloat16"),
        spec=spec,
    )
    narrow_mlp = next(op for op in narrow if op.computation == "mlp")
    wide_mlp = next(op for op in wide if op.computation == "mlp")
    assert narrow_mlp.flops < wide_mlp.flops
    assert _RT.moe_intermediate == 1024


def test_graph_draft_class_suppresses_the_synthetic_stack():
    assert _graph_contains_draft(
        [{"label": "MultiTokenPredictorLayer", "namespace": "model", "attrs": []}]
    )
    assert not _graph_contains_draft(
        [{"label": "DecoderLayer", "namespace": "model/layers", "attrs": []}]
    )


def test_offline_draft_models_project_from_their_parsed_layers():
    deepseek = load_architecture(
        FIXTURES / "vllm_deepseek_like",
        name="DeepSeek-like",
        detailed=True,
        basic_ops=BasicOpFilter.for_roofline(),
    )
    deepseek.engine = "vllm"
    projected = project_roofline(
        deepseek, batch_size=1, seq_len=4, num_predicted_tokens=2
    )
    # The graph draws eh_proj and q_proj. The inlined attention core, indexer,
    # and expert GEMMs are still costed, and the drawn leaves are not repeated.
    missing = [op for op in projected.ops if op.section.startswith("mtp/")]
    assert {op.section for op in missing} == {"mtp/0"}
    assert any(op.computation == "attn" for op in missing)
    assert (
        next(op for op in missing if op.computation == "kv_proj").cost_source
        == "source"
    )
    assert (
        next(op for op in missing if op.computation == "attn").cost_source == "config"
    )
    assert any(op.computation == "indexer_score" for op in missing)
    assert any(op.computation == "mlp" and op.category == "MoE" for op in missing)
    assert not any(op.computation in {"eh_proj", "q_proj", "router"} for op in missing)
    assert any(
        "eh_proj" in op.name and not op.section.startswith("mtp/")
        for op in projected.ops
    )
    assert any(op.section.endswith("sliding_attention") for op in projected.ops)
    assert any(op.section.endswith("full_attention") for op in projected.ops)
    assert projected.notes == []

    nextn = load_architecture(
        FIXTURES / "sglang_nextn_like",
        name="Qwen-NextN",
        detailed=True,
        basic_ops=BasicOpFilter.for_roofline(),
    )
    nextn.engine = "sglang"
    projected = project_roofline(nextn, batch_size=1, seq_len=4, num_predicted_tokens=1)
    missing = [op for op in projected.ops if op.section.startswith("mtp/")]
    assert any(op.computation == "attn" for op in missing)
    assert not any(op.computation == "mlp" for op in missing)
    draft = [op for op in projected.ops if op.section.startswith("nextn")]
    assert draft and all(op.category != "MoE" for op in draft)
    assert any(op.category == "GEMM" for op in draft)
