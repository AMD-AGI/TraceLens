###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for the registry-based torch op categorizer.
"""

import pytest

from TraceLens.PerfModel.torch_op_mapping import (
    OP_CATEGORY_REGISTRY,
    build_sheet_category_to_op_names,
    categorize_torch_op,
    op_to_perf_model_class_map,
    register_op_categories,
)


def _row(name, kernel_details=None):
    return {"name": name, "kernel_details": kernel_details or []}


@pytest.mark.parametrize(
    "name,expected",
    [
        ("aten::mm", "GEMM"),
        ("aten::addmm", "GEMM"),
        ("aten::convolution", "CONV_fwd"),
        ("aten::miopen_convolution", "CONV_fwd"),
        ("aten::cudnn_convolution", "CONV_fwd"),
        ("aten::convolution_backward", "CONV_bwd"),
        ("ConvBias_Backward", "CONV_bwd"),
        ("aten::layer_norm", "NORM_fwd"),
        ("aten::native_layer_norm_backward", "NORM_bwd"),
        ("FlashAttnFunc", "SDPA_fwd"),
        ("FlashAttnFuncBackward", "SDPA_bwd"),
        ("FusedAttnFuncBackward", "SDPA_bwd"),
        ("flash_attn::_flash_attn_backward", "SDPA_bwd"),
        ("aten::_scaled_dot_product_flash_attention_backward", "SDPA_bwd"),
        ("primus_turbo::grouped_gemm", "GroupedGEMM_fwd"),
        ("primus_turbo_cpp_extension::ck_grouped_gemm", "GroupedGEMM_fwd"),
        ("MambaSplitConv1dScanCombinedFn", "SSM_fwd"),
        ("MambaSplitConv1dScanCombinedFnBackward", "SSM_bwd"),
        ("DaoAILab::_causal_conv1d_bwd_cpp", "SSM_bwd"),
        ("MoEDispatch", "MoE_comm_fwd"),
        ("MoEDispatchBackward", "MoE_comm_bwd"),
        ("TokenPermuteMaskMap", "MoE_comm_fwd"),
        ("TokenPermuteMaskMapBackward", "MoE_comm_bwd"),
        ("FusedRoPEFunc", "RoPE_fwd"),
        ("FusedRoPEFuncBackward", "RoPE_bwd"),
        ("CrossEntropyFunction", "CrossEntropy_fwd"),
        ("CrossEntropyFunctionBackward", "CrossEntropy_bwd"),
        ("aiter::topk_softmax", "MoE_aux"),
        ("_C_cache_ops::reshape_and_cache_flash", "InferenceAttention"),
        ("aiter::silu_and_mul", "elementwise"),
        ("aten::sum", "reduce"),
        ("triton_per_token_quant_fp8", "triton"),
        ("record_param_comms_alltoall_base", "record_param_comms"),
        ("completely::unknown::op", "other"),
    ],
)
def test_categorize_torch_op_expected_categories(name, expected):
    assert categorize_torch_op(_row(name)) == expected


@pytest.mark.parametrize(
    "kernel_name,expected",
    [
        ("void at::native::elementwise_kernel<8, 4>(...)", "elementwise"),
        ("void at::native::reduce_kernel<512, 4>(...)", "reduce"),
        (
            "void at::native::multi_tensor_apply_kernel<...>(...)",
            "multi_tensor_apply",
        ),
        ("void at::native::unrelated_kernel(...)", "other"),
        ("some_user_kernel(...)", "other"),
    ],
)
def test_kernel_name_fallback(kernel_name, expected):
    row = _row("unknown_op_name", kernel_details=[{"name": kernel_name}])
    assert categorize_torch_op(row) == expected


def test_registry_covers_every_perf_model_op():
    missing = [
        name for name in op_to_perf_model_class_map if name not in OP_CATEGORY_REGISTRY
    ]
    assert missing == []


def test_sheet_category_to_op_names_keeps_legacy_sheet_view():
    sheet_category_to_op_names = build_sheet_category_to_op_names(
        op_to_perf_model_class_map
    )

    assert "aten::mm" in sheet_category_to_op_names["GEMM"]
    assert "primus_turbo::grouped_gemm" in sheet_category_to_op_names["GroupedGEMM"]
    assert "aten::convolution" in sheet_category_to_op_names["CONV"]
    assert "aten::convolution_backward" in sheet_category_to_op_names["CONV"]
    assert "FlashAttnFuncBackward" not in sheet_category_to_op_names["SDPA"]
    assert "TokenPermuteMaskMap" not in sheet_category_to_op_names["MoE_comm"]


def test_register_op_category_extension_updates_registry_only():
    registry = {}

    register_op_categories(
        {"MyCategoryOnlyBackward": "SDPA_bwd"},
        registry,
    )

    assert registry["MyCategoryOnlyBackward"] == "SDPA_bwd"
