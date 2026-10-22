###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Registry mappings for DSv4 Pro kernel names already covered by existing classes.

Shapes follow each class's documented Input Dims. No MI455X capture contains
these kernel names, so the events are synthetic.
"""

import math

import pytest

from TraceLens.PerfModel.extensions.perf_model_extensions import (
    gemm_a16w16_asm,
    gemm_a8w8_blockscale,
    mhc_fused_post_pre_gemm_sqrsum,
    mhc_post,
    mhc_pre_big_fuse_rmsnorm,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    resolve_perf_model_class,
)

MX128_NAME = "gemm_a8w8_mx128_bf16_128x256"
BATCHED_MX128_NAME = "batched_gemm_a8w8_mx128_bf16_128x256"
A16W16_NAME = "gemm_a16w16_clusterlaunch_tdm_splitk_ws_kernel_gfx1250"

_BF16 = "c10::BFloat16"


def _event(name, dims, types):
    return {"name": name, "args": {"Input Dims": dims, "Input type": types}}


# gemm_a8w8_blockscale: [[M, K], [N, K], [x_scale], [w_scale], ...]
MX128_EVENT = _event(
    MX128_NAME,
    [[128, 256], [512, 256], [128, 2], [4, 2]],
    ["signed char", "signed char", "float", "float"],
)

# gemm_a16w16_asm: [[M, K], [N, K], [M, N], [split-K workspace], ...]
# Index 3 is a non-tensor workspace (dtype "unsigned int").
A16W16_EVENT = _event(
    A16W16_NAME,
    [[128, 256], [512, 256], [128, 512], []],
    [_BF16, _BF16, _BF16, "unsigned int"],
)

# mhc_post(out, x, residual, post_layer_mix, comb_res_mix, ...)
MHC_POST_EVENT = _event(
    "mhc_post_kernel",
    [[4, 2, 128], [4, 128], [4, 2, 128], [4, 2, 1], [4, 2, 2]],
    [_BF16, _BF16, _BF16, "float", "float"],
)

# mhc_pre_big_fuse_rmsnorm: residual is Input index 7, hc_base is index 6.
MHC_PRE_FUSE_EVENT = _event(
    "mhc_pre_big_fuse_rmsnorm_kernel",
    [
        [4, 2, 1],  # post_mix
        [4, 2, 2],  # comb_mix
        [4, 128],  # out (M, C)
        [2, 4, 8],  # gemm_out_mul (split_k, M, hc3)
        [2, 4],  # gemm_out_sqrsum
        [8],  # hc_scale
        [8],  # hc_base
        [4, 2, 128],  # residual
        [128],  # norm_weight
    ],
    ["float", "float", _BF16, "float", "float", "float", "float", _BF16, _BF16],
)

# mhc_fused_post_pre_gemm_sqrsum: layer_input is index 3, fn is index 7.
MHC_FUSED_EVENT = _event(
    "mhc_fused_post_pre_gemm_sqrsum_kernel",
    [
        [2, 4, 8],  # gemm_out_mul
        [2, 4],  # gemm_out_sqrsum
        [4, 2, 128],  # next_residual
        [4, 128],  # layer_input
        [4, 2, 128],  # residual_in
        [4, 2, 1],  # post_layer_mix
        [4, 2, 2],  # comb_res_mix
        [8, 256],  # fn (hc3, K)
    ],
    ["float", "float", _BF16, _BF16, _BF16, "float", "float", "float"],
)


@pytest.mark.parametrize(
    "name,expected",
    [
        (MX128_NAME, gemm_a8w8_blockscale),
        ("void gemm_a8w8_mx128_kernel<128>(float)", gemm_a8w8_blockscale),
        (A16W16_NAME, gemm_a16w16_asm),
        ("mhc_post_kernel", mhc_post),
        ("mhc_pre_big_fuse_rmsnorm_kernel", mhc_pre_big_fuse_rmsnorm),
        ("mhc_fused_post_pre_gemm_sqrsum_kernel", mhc_fused_post_pre_gemm_sqrsum),
        ("aiter::mhc_post", mhc_post),
    ],
)
def test_resolve_perf_model_class(name, expected):
    assert resolve_perf_model_class(name) is expected


def test_batched_mx128_is_not_matched():
    assert resolve_perf_model_class(BATCHED_MX128_NAME) is None
    assert resolve_perf_model_class(f"prefix_{BATCHED_MX128_NAME}_suffix") is None


@pytest.mark.parametrize(
    "name,expected",
    [
        (MX128_NAME, "GEMM"),
        (A16W16_NAME, "GEMM"),
        ("mhc_post_kernel", "mHC"),
        ("mhc_pre_big_fuse_rmsnorm_kernel", "mHC"),
        ("mhc_fused_post_pre_gemm_sqrsum_kernel", "mHC"),
        ("aiter::mhc_post", "mHC"),
    ],
)
def test_categorize_mapped_names(name, expected):
    assert categorize_torch_op({"name": name}) == expected


def test_batched_mx128_is_not_categorized_as_gemm():
    assert categorize_torch_op({"name": BATCHED_MX128_NAME}) == "other"


@pytest.mark.parametrize(
    "name,event",
    [
        (MX128_NAME, MX128_EVENT),
        (A16W16_NAME, A16W16_EVENT),
        ("mhc_post_kernel", MHC_POST_EVENT),
        ("mhc_pre_big_fuse_rmsnorm_kernel", MHC_PRE_FUSE_EVENT),
        ("mhc_fused_post_pre_gemm_sqrsum_kernel", MHC_FUSED_EVENT),
    ],
)
def test_synthetic_event_has_finite_flops_and_bytes(name, event):
    model = resolve_perf_model_class(name)(event)
    flops = model.flops()
    nbytes = model.bytes()
    assert flops is not None and math.isfinite(flops) and flops > 0
    assert nbytes is not None and math.isfinite(nbytes) and nbytes > 0
