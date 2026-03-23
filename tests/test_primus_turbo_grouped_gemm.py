###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from TraceLens.PerfModel.perf_model import (
    primus_turbo_grouped_gemm,
    primus_turbo_grouped_gemm_variable_k,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)

# ---------------------------------------------------------------------------
# Mapping and categorization
# ---------------------------------------------------------------------------


def test_fixed_k_ops_map_to_primus_turbo_grouped_gemm():
    fixed_k_ops = [
        "primus_turbo::grouped_gemm",
        "primus_turbo::grouped_gemm_impl",
        "primus_turbo_cpp_extension::grouped_gemm",
    ]
    for op in fixed_k_ops:
        assert op_to_perf_model_class_map[op] is primus_turbo_grouped_gemm, op
        row = {"name": op, "kernel_details": []}
        assert categorize_torch_op(row) == "GEMM", op


def test_variable_k_ops_map_to_primus_turbo_grouped_gemm_variable_k():
    variable_k_ops = [
        "primus_turbo::grouped_gemm_variable_k",
        "primus_turbo::grouped_gemm_variable_k_impl",
        "primus_turbo_cpp_extension::grouped_gemm_variable_k",
    ]
    for op in variable_k_ops:
        assert (
            op_to_perf_model_class_map[op] is primus_turbo_grouped_gemm_variable_k
        ), op
        row = {"name": op, "kernel_details": []}
        assert categorize_torch_op(row) == "GEMM", op


# ---------------------------------------------------------------------------
# primus_turbo_grouped_gemm (fixed-K) — flops and bytes
# ---------------------------------------------------------------------------


def test_fixed_k_impl_format_gkn():
    """Compact _impl format: [[M,K], [G,K,N]]."""
    event = {
        "name": "primus_turbo::grouped_gemm_impl",
        "args": {
            "Input Dims": [[24576, 2048], [8, 2048, 2816]],
            "Input type": ["c10::BFloat16", "c10::BFloat16"],
        },
    }
    model = primus_turbo_grouped_gemm(event)
    M, K, N, G = 24576, 2048, 2816, 8
    assert model.flops() == 2 * M * K * N
    # GroupedGemm.bytes: (M*K + G*K*N)*bpe_in + M*N*bpe_out  (bf16 = 2 bytes)
    assert model.bytes() == (M * K + G * K * N) * 2 + M * N * 2


def test_fixed_k_impl_format_gnk():
    """Compact _impl format with transposed weight layout: [[M,K], [G,N,K]]."""
    event = {
        "name": "primus_turbo::grouped_gemm_impl",
        "args": {
            "Input Dims": [[24576, 2048], [8, 1408, 2048]],
            "Input type": ["c10::BFloat16", "c10::BFloat16"],
        },
    }
    model = primus_turbo_grouped_gemm(event)
    M, K, N, G = 24576, 2048, 1408, 8
    assert model.flops() == 2 * M * K * N
    assert model.bytes() == (M * K + G * K * N) * 2 + M * N * 2


def test_fixed_k_zipped_format():
    """Zipped format: uniform K and N across all groups."""
    event = {
        "name": "primus_turbo::grouped_gemm",
        "args": {
            "Input Dims": [
                [[4, 8], [5, 8]],
                [[8, 16], [8, 16]],
            ],
            "Input type": ["c10::BFloat16", "c10::BFloat16"],
        },
    }
    model = primus_turbo_grouped_gemm(event)
    M_total, K, N, G = 9, 8, 16, 2
    assert model.flops() == 2 * M_total * K * N
    assert model.bytes() == (M_total * K + G * K * N) * 2 + M_total * N * 2


# ---------------------------------------------------------------------------
# primus_turbo_grouped_gemm_variable_k — flops and bytes
# ---------------------------------------------------------------------------


def test_variable_k_impl_format():
    """Compact variable-K _impl format: [[M,K], [M,N]] — aggregate view, single group."""
    event = {
        "name": "primus_turbo::grouped_gemm_variable_k_impl",
        "args": {
            "Input Dims": [[24576, 1408], [24576, 2048]],
            "Input type": ["c10::BFloat16", "c10::BFloat16"],
        },
    }
    model = primus_turbo_grouped_gemm_variable_k(event)
    M, K, N = 24576, 1408, 2048
    assert model.flops() == 2 * M * K * N
    # Per-group bytes with G=1: (M*K + 1*K*N)*bpe_in + M*N*bpe_out
    assert model.bytes() == (M * K + K * N) * 2 + M * N * 2


def test_variable_k_zipped_format():
    """Zipped format with variable K per group."""
    event = {
        "name": "primus_turbo::grouped_gemm_variable_k",
        "args": {
            "Input Dims": [
                [[4, 8], [5, 6]],
                [[8, 16], [6, 12]],
            ],
            "Input type": ["c10::BFloat16", "c10::BFloat16"],
        },
    }
    model = primus_turbo_grouped_gemm_variable_k(event)
    # group1: (4,8)@(8,16) → 2*4*8*16 = 1024 flops
    # group2: (5,6)@(6,12) → 2*5*6*12 = 720 flops
    assert model.flops() == 1744
    # group1 bytes (G=1): (4*8 + 8*16)*2 + 4*16*2 = (32+128)*2+128 = 448
    # group2 bytes (G=1): (5*6 + 6*12)*2 + 5*12*2 = (30+72)*2+120 = 324
    assert model.bytes() == 448 + 324
