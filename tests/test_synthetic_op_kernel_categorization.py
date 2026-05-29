###############################################################################
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for categorizing TreePerf synthetic ops by underlying GPU kernel name."""

import warnings

import pytest

from TraceLens.PerfModel.torch_op_mapping import categorize_torch_op

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


def test_unmapped_synthetic_kernel_falls_back_to_other():
    row = _synthetic_row("_fwd_grouped_kernel_stage1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert categorize_torch_op(row) == "other"


def test_cuda_graph_launch_prefix():
    row = _synthetic_row(GEMM_KERNEL, parent="cudaGraphLaunch")
    assert categorize_torch_op(row) == "GEMM"
