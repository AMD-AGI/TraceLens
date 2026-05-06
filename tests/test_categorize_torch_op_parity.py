###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Parity test for :func:`categorize_torch_op_v2` (registry-based) against
:func:`categorize_torch_op` (legacy if/elif chain).

PR A introduces v2 alongside v1 without changing v1's behavior. The single
acceptance criterion is that v2 produces the same output as v1 for every
reachable input. PR B will delete v1; that PR's safety relies on this test.
"""

import pytest

from TraceLens.PerfModel.torch_op_mapping import (
    OP_CATEGORY_REGISTRY,
    categorize_torch_op,
    categorize_torch_op_v2,
    op_to_perf_model_class_map,
)


def _row(name, kernel_details=None):
    return {"name": name, "kernel_details": kernel_details or []}


# ---------------------------------------------------------------------------
# Coverage set 1: every name reachable through perf-model registration
# (op_to_perf_model_class_map, including extension-contributed entries).
# ---------------------------------------------------------------------------
PERF_MODEL_NAMES = sorted(op_to_perf_model_class_map.keys())


# ---------------------------------------------------------------------------
# Coverage set 2: names hardcoded inside the legacy categorize_torch_op
# if/elif chain. These are the names PR A had to thread into
# LEGACY_CATEGORIZE_EXTRAS to preserve parity.
# ---------------------------------------------------------------------------
LEGACY_HARDCODED_NAMES = [
    # CONV_fwd
    "aten::convolution",
    "aten::miopen_convolution",
    "aten::cudnn_convolution",
    "ConvBias_",
    "ConvBiasReLU_",
    # CONV_bwd
    "aten::convolution_backward",
    "ConvBias_Backward",
    "ConvBiasReLU_Backward",
    # SDPA backward names (some perf-modeled, some unreachable in v1)
    "FlashAttnFuncBackward",
    "FusedAttnFuncBackward",
    "flash_attn::_flash_attn_backward",
    "flash_attn::_flash_attn_varlen_backward",
    "aten::_scaled_dot_product_cudnn_attention_backward",
    "aten::_scaled_dot_product_efficient_attention_backward",
    "aten::_scaled_dot_product_flash_attention_backward",
    "aiter::_flash_attn_backward",
    "aiter::wrapper_fmha_v3_bwd",
    "aiter::mha_bwd",
    # SSM
    "MambaSplitConv1dScanCombinedFn",
    "MambaSplitConv1dScanCombinedFnBackward",
    "DaoAILab::_causal_conv1d_bwd_cpp",
    # MoE_comm
    "TokenPermuteMaskMap",
    "_OperationFuserAutogradFunction",
    "MoEDispatchBackward",
    "MoECombineBackward",
    "TokenPermuteMaskMapBackward",
    "_OperationFuserAutogradFunctionBackward",
    # RoPE / CrossEntropy backward
    "FusedRoPEFuncBackward",
    "CrossEntropyFunctionBackward",
    # MoE auxiliary
    "aiter::moe_sorting_fwd",
    "aiter::moe_sorting_opus_fwd",
    "aiter::moe_align_block_size",
    "_moe_C::moe_align_block_size",
    "aiter::fused_moe_->_fused_dynamic_mxfp4_quant_moe_sort_kernel (Synthetic Op)",
    "aiter::moe_sum",
    "aiter::topk_softmax",
    "aiter::topk_softmax_asm",
    "aiter::topk_sigmoid",
    "aiter::biased_grouped_topk_hip",
    "aiter::grouped_topk",
    "aiter::moe_fused_gate",
    # InferenceAttention extras
    "_C_cache_ops::reshape_and_cache_flash",
    "_C_cache_ops::concat_and_cache_mla",
]


# ---------------------------------------------------------------------------
# Coverage set 3: names matched by patterns (triton, record_param_comms).
# ---------------------------------------------------------------------------
PATTERN_NAMES = [
    "triton",
    "triton_softmax_kernel",
    "triton_per_token_quant_fp8",
    "record_param_comms",
    "record_param_comms_alltoall_base",
]


# ---------------------------------------------------------------------------
# Coverage set 4: unknown names that should resolve to "other" (or to
# a kernel-name fallback).
# ---------------------------------------------------------------------------
OTHER_NAMES = [
    "completely::unknown::op",
    "no_match_at_all",
    "Some::Random::Op",
]


# ---------------------------------------------------------------------------
# Coverage set 5: kernel-name fallback inputs.
# ---------------------------------------------------------------------------
KERNEL_FALLBACK_CASES = [
    # (kernel_name, expected_category)
    ("void at::native::elementwise_kernel<8, 4>(...)", "elementwise"),
    ("void at::native::reduce_kernel<512, 4>(...)", "reduce"),
    ("void at::native::multi_tensor_apply_kernel<...>(...)", "multi_tensor_apply"),
    # Native kernel that matches no needle -> "other"
    ("void at::native::unrelated_kernel(...)", "other"),
    # Non-native kernel -> "other"
    ("some_user_kernel(...)", "other"),
]


# ---------------------------------------------------------------------------
# Parametrized parity assertions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", PERF_MODEL_NAMES)
def test_parity_perf_model_names(name):
    row = _row(name)
    assert categorize_torch_op_v2(row) == categorize_torch_op(row)


@pytest.mark.parametrize("name", LEGACY_HARDCODED_NAMES)
def test_parity_legacy_hardcoded_names(name):
    row = _row(name)
    assert categorize_torch_op_v2(row) == categorize_torch_op(row)


@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_parity_pattern_names(name):
    row = _row(name)
    assert categorize_torch_op_v2(row) == categorize_torch_op(row)


@pytest.mark.parametrize("name", OTHER_NAMES)
def test_parity_other_names(name):
    row = _row(name)
    assert categorize_torch_op_v2(row) == categorize_torch_op(row)


@pytest.mark.parametrize("kernel_name,expected", KERNEL_FALLBACK_CASES)
def test_parity_kernel_fallback(kernel_name, expected):
    row = _row("unknown_op_name", kernel_details=[{"name": kernel_name}])
    v1 = categorize_torch_op(row)
    v2 = categorize_torch_op_v2(row)
    assert v1 == v2 == expected, f"v1={v1!r}, v2={v2!r}, expected={expected!r}"


# ---------------------------------------------------------------------------
# Sanity checks on the registry shape
# ---------------------------------------------------------------------------
def test_registry_built_at_import():
    assert isinstance(OP_CATEGORY_REGISTRY, dict)
    assert len(OP_CATEGORY_REGISTRY) > 0


def test_registry_contains_expected_entries():
    # Auto-derived from a perf-model base class
    assert OP_CATEGORY_REGISTRY["aten::mm"] == "GEMM"
    # Auto-derived NORM split
    assert OP_CATEGORY_REGISTRY["aten::layer_norm"] == "NORM_fwd"
    assert OP_CATEGORY_REGISTRY["aten::layer_norm_backward"] == "NORM_bwd"
    # Auto-derived SDPA fwd vs explicit-list bwd
    assert OP_CATEGORY_REGISTRY["FlashAttnFunc"] == "SDPA_fwd"
    assert OP_CATEGORY_REGISTRY["aiter::wrapper_fmha_v3_bwd"] == "SDPA_bwd"
    # Extras-only entries
    assert OP_CATEGORY_REGISTRY["aten::miopen_convolution"] == "CONV_fwd"
    assert OP_CATEGORY_REGISTRY["DaoAILab::_causal_conv1d_bwd_cpp"] == "SSM_bwd"
    assert OP_CATEGORY_REGISTRY["aiter::topk_softmax"] == "MoE_aux"


def test_registry_covers_every_perf_model_op():
    # The builder must produce a category for every perf-modeled op (i.e. no
    # silent drops). PR B relies on this invariant.
    missing = [n for n in op_to_perf_model_class_map if n not in OP_CATEGORY_REGISTRY]
    assert missing == [], f"perf-modeled ops missing from registry: {missing}"
