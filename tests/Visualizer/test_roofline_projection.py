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
from TraceLens.ModelUtils.extract import load_architecture
from TraceLens.Visualizer.model_explorer_export import roofline_cli
from TraceLens.Visualizer.model_explorer_export.roofline_projection import (
    DimResolutionError,
    RooflineOp,
    RooflineProjection,
    _RT,
    _attention_boundary_cost,
    _attr_text,
    _classify_normalization,
    _conv_cost,
    _dispatch_op_cost,
    _parse_conv_details,
    _engine_boundary_cost,
    _engine_head_cost,
    _introspect_weight_bpe,
    _moe_cost,
    _parse_op_node,
    _project_op,
    _split_shape_text,
    build_projection_dims,
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
        _, _, _, bytes_ep1, _ = _moe_cost(dims, 2, 1024, bpe=2)
        _RT.ep = 8
        _, _, _, bytes_ep8, _ = _moe_cost(dims, 2, 1024, bpe=2)
    finally:
        _RT.reset()
    # EP>1 adds the dispatch+combine all-to-all: 2 * active_tokens * hidden * bpe.
    active_tokens = 10 * 2
    assert bytes_ep8 - bytes_ep1 == 2 * active_tokens * 4096 * 2


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

    unmodeled_names = {op.name for op in projection.unmodeled_ops}
    assert "embed_tokens" in unmodeled_names

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
