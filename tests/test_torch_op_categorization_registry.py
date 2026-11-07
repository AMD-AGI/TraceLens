###############################################################################
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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
    registered_op_for_kernel,
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


@pytest.mark.parametrize(
    "name,expected",
    [
        ("hipGraphLaunch->Cijk_Alik_Bljk_MT16x16x256 (Synthetic Op)", "other"),
        (
            "hipGraphLaunch->void ck::kernel_gemm_xdl_cshuffle_v3"
            "<ck::tensor_operation::element_wise::PassThrough> (Synthetic Op)",
            "other",
        ),
        (
            "hipGraphLaunch->aiter::mla_a8w8_qh64_qseqlen1_gqaratio64_nm (Synthetic Op)",
            "other",
        ),
        (
            "hipGraphLaunch->void aiter::mhc_pre_big_fuse_rmsnorm_kernel<std::bfloat16_t> (Synthetic Op)",
            "mHC",
        ),
        ("hipGraphLaunch->aiter::mhc_post (Synthetic Op)", "mHC"),
        (
            "hipGraphLaunch->void at::native::elementwise_kernel<8, 4>() (Synthetic Op)",
            "elementwise",
        ),
        ("hipGraphLaunch->totally_custom_kernel (Synthetic Op)", "other"),
        (
            "hipModuleLaunchKernel->_fwd_kernel_stage2_asm (Synthetic Op)",
            "InferenceAttention",
        ),
    ],
)
def test_synthetic_graph_row_uses_the_perf_report_categorizer(name, expected):
    """A graph-replay row is categorized by the perf-report categorizer.

    The launch's kernel list often starts with an ATen elementwise kernel.
    That list must not relabel the kernel this row owns. A kernel takes a
    registered op's category when its name contains that op at a kernel
    boundary. An unrecognized kernel stays other.
    """
    row = _row(
        name,
        kernel_details=[{"name": "void at::native::elementwise_kernel<8, 4>()"}],
    )
    assert categorize_torch_op(row) == expected


def test_registered_op_for_kernel_uses_the_longest_bounded_key():
    registry = {
        "aiter::mhc_pre": "mHC",
        "aiter::mhc_pre_big_fuse_rmsnorm": "mHC",
        "aiter::mhc_post": "mHC",
        "aiter::gemm_a8w8": "GEMM",
        "aiter::gemm_a8w8_blockscale_ck": "GEMM",
    }
    assert (
        registered_op_for_kernel(
            "void aiter::mhc_pre_big_fuse_rmsnorm_kernel<std::bfloat16_t>",
            registry,
        )
        == "aiter::mhc_pre_big_fuse_rmsnorm"
    )
    assert (
        registered_op_for_kernel(
            "void aiter::mhc_post_kernel<std::bfloat16_t, 4>",
            registry,
        )
        == "aiter::mhc_post"
    )
    assert (
        registered_op_for_kernel(
            "void aiter::gemm_a8w8_blockscale_ck_kernel<float>",
            registry,
        )
        == "aiter::gemm_a8w8_blockscale_ck"
    )
    assert registered_op_for_kernel("Cijk_Alik_Bljk_MT16x16x256", registry) is None
    assert registered_op_for_kernel("aiter::mla_a8w8_qh64", registry) is None
    assert registered_op_for_kernel("mfma_moe1_silu_mul", registry) is None


def test_registry_covers_every_perf_model_op():
    missing = [
        name for name in op_to_perf_model_class_map if name not in OP_CATEGORY_REGISTRY
    ]
    assert missing == []


def test_every_perf_model_class_declares_category():
    """Every perf model class must declare a non-None ``category`` attribute.

    Guards against new perf-model PRs that forget to set ``category`` (or
    inherit it from a base that does), which would otherwise raise
    ``ValueError`` from ``build_op_category_registry`` at import time with
    no indication of which class caused it.
    """
    missing = []
    for op_name, perf_model_class in op_to_perf_model_class_map.items():
        category = getattr(perf_model_class, "category", None)
        if category is None:
            missing.append((op_name, perf_model_class.__name__))
    assert missing == [], (
        "perf model classes missing a non-None 'category' attribute: "
        f"{missing}. Declare 'category' on the class (or on a base class) "
        "so OP_CATEGORY_REGISTRY can resolve it."
    )


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
