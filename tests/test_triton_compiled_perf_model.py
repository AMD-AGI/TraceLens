###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for TritonCompiledPerfModel (V2 trace-intrinsic path).

Start with RMSNorm (reduction kernel, clear expected FLOPs/bytes).
More cases (V1 wrapper-parsing, multi-word ops, fallback) in follow-ups.
"""

from TraceLens.PerfModel.triton_compiled_perf_model import (
    TritonCompiledPerfModel,
    _meta_from_trace_args,
    _parse_kernel_name,
)

# ---------------------------------------------------------------------------
# RMSNorm kernel from a TransformerBlock trace (bf16, batch=8, seq=4096, dim=2048)
# triton_red_fused_add_mean_mul_pow_rsqrt_0
# ---------------------------------------------------------------------------
_RMSNORM_EVENT = {
    "name": "triton_red_fused_add_mean_mul_pow_rsqrt_0",
    "args": {
        "Concrete Inputs": ["", "", "", "32768", "2048"],
        "Input Dims": [[8, 4096, 2048], [2048], [8, 4096, 2048], [], []],
        "Input type": [
            "c10::BFloat16",
            "c10::BFloat16",
            "c10::BFloat16",
            "Scalar",
            "Scalar",
        ],
    },
}


def test_parse_kernel_name_rmsnorm():
    ops = _parse_kernel_name("triton_red_fused_add_mean_mul_pow_rsqrt_0")
    assert ops == [
        "aten.add",
        "aten.mean",
        "aten.mul",
        "aten.pow",
        "aten.rsqrt",
    ]


def test_parse_kernel_name_no_match():
    assert _parse_kernel_name("some_other_kernel") == []


def test_meta_from_trace_args_rmsnorm():
    meta = _meta_from_trace_args(_RMSNORM_EVENT)
    assert meta is not None
    assert meta["xnumel"] == 32768
    assert meta["rnumel"] == 2048
    assert meta["aten_ops"] == [
        "aten.add",
        "aten.mean",
        "aten.mul",
        "aten.pow",
        "aten.rsqrt",
    ]


def test_meta_from_trace_args_missing_fields():
    event = {"name": "triton_red_fused_add_0", "args": {}}
    assert _meta_from_trace_args(event) is None


def test_flops_rmsnorm():
    model = TritonCompiledPerfModel(_RMSNORM_EVENT)
    # add=1, mean=1, mul=1, pow=2, rsqrt=2 => 7 flops/elem
    # 7 * 32768 * 2048 = 469,762,048
    assert model.flops() == 7 * 32768 * 2048


def test_bytes_rmsnorm():
    model = TritonCompiledPerfModel(_RMSNORM_EVENT)
    # Three bf16 tensors: [8,4096,2048]*2 + [2048]*2 + [8,4096,2048]*2
    # = 134,217,728 + 4,096 + 134,217,728 = 268,439,552
    expected = (8 * 4096 * 2048 * 2) + (2048 * 2) + (8 * 4096 * 2048 * 2)
    assert model.bytes() == expected
