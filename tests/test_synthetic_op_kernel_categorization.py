###############################################################################
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for categorizing TreePerf synthetic ops by underlying GPU kernel name."""

import warnings

import pytest

from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    resolve_perf_model_class,
)

MOE_GEMM_KERNEL = (
    "_ZN2ck15kernel_moe_gemmINS_25GridwiseMoeGemmBlockScaleINS_13tensor_layout4gemm"
)
GEMM_KERNEL = "_ZN2ck59kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffleINS_58GridwiseGemm"
QUANT_KERNEL = "_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDhDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii"
ATEN_ELEMENTWISE_KERNEL = "void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl"


def _synthetic_row(kernel_name, parent="hipGraphLaunch"):
    return {
        "name": f"{parent}->{kernel_name} (Synthetic Op)",
        "kernel_details": [{"name": kernel_name}],
    }


def test_hipgraph_moe_gemm_categorizes_as_moe_unfused():
    assert categorize_torch_op(_synthetic_row(MOE_GEMM_KERNEL)) == "MoE_unfused"


def test_hipgraph_ck_gemm_categorizes_as_gemm():
    assert categorize_torch_op(_synthetic_row(GEMM_KERNEL)) == "GEMM"


def test_hipgraph_quant_kernel_categorizes_as_groupquant():
    assert categorize_torch_op(_synthetic_row(QUANT_KERNEL)) == "GroupQuant"


def test_hipgraph_aten_native_elementwise():
    row = _synthetic_row(ATEN_ELEMENTWISE_KERNEL)
    assert categorize_torch_op(row) == "elementwise"


def test_synthetic_name_without_kernel_details_uses_parsed_kernel():
    row = {
        "name": f"hipGraphLaunch->{MOE_GEMM_KERNEL} (Synthetic Op)",
        "kernel_details": [],
    }
    assert categorize_torch_op(row) == "MoE_unfused"


def test_graph_parent_name_alone_would_be_other():
    row = {"name": "hipGraphLaunch", "kernel_details": []}
    assert categorize_torch_op(row) == "other"


def test_fwd_grouped_kernel_stage1_categorizes_as_sdpa():
    row = _synthetic_row("_fwd_grouped_kernel_stage1")
    assert categorize_torch_op(row) == "SDPA_fwd"


def test_grouped_topk_categorizes_as_moe_aux():
    row = _synthetic_row(
        "void aiter::grouped_topk_kernel<float, float __vector(4), 1, true, true, false>"
    )
    assert categorize_torch_op(row) == "MoE_aux"


def test_store_kvcache_categorizes_as_inference_attention():
    row = _synthetic_row(
        "void (anonymous namespace)::store_kvcache<256l, 1, false, long>"
    )
    assert categorize_torch_op(row) == "InferenceAttention"


def test_triton_poi_fused_categorizes_as_elementwise():
    row = _synthetic_row(
        "triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0"
    )
    assert categorize_torch_op(row) == "elementwise"


def test_cuda_graph_launch_prefix():
    row = _synthetic_row(GEMM_KERNEL, parent="cudaGraphLaunch")
    assert categorize_torch_op(row) == "GEMM"


def _synthetic_event(kernel_name, parent="aten::mm", input_dims=None):
    args = {}
    if input_dims is not None:
        m, k, n = input_dims
        args = {
            "Input Dims": [[m, k], [k, n]],
            "Input type": ["c10::BFloat16", "c10::BFloat16"],
            "Input Strides": [[k, 1], [1, k]],
        }
    return {
        "name": f"{parent}->{kernel_name} (Synthetic Op)",
        "args": args,
        "kernel_details": [{"name": kernel_name}],
        "gpu_events": [],
    }


def test_resolve_perf_model_class_named_op():
    event = {"name": "aten::mm", "args": {"Input Dims": [[4, 8], [8, 2]]}}
    assert resolve_perf_model_class(event) is perf_model.aten_mm


def test_synthetic_gemm_resolves_aten_mm_with_input_dims():
    event = _synthetic_event(GEMM_KERNEL, input_dims=(128, 64, 32))
    assert resolve_perf_model_class(event) is perf_model.aten_mm
    pm = perf_model.aten_mm(event=event)
    assert pm.flops() == 2 * 128 * 32 * 64


def test_synthetic_gemm_without_input_dims_has_no_perf_model():
    event = _synthetic_event(GEMM_KERNEL, parent="hipGraphLaunch")
    assert resolve_perf_model_class(event) is None


def test_synthetic_moe_gemm_resolves_grouped_gemm_with_input_dims():
    event = _synthetic_event(
        MOE_GEMM_KERNEL, parent="aten::bmm", input_dims=(8, 64, 32)
    )
    assert resolve_perf_model_class(event) is perf_model.primus_turbo_grouped_gemm
