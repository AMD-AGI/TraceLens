###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for the Triton fused-MoE expert GEMM perf model + categorization.

Regression coverage for the gap where the sglang/vLLM Triton ``fused_moe_kernel``
(op ``..._invoke_fused_moe_kernel``) fell into the catch-all ``other`` category
with no perf model -> no roofline efficiency, no optimization candidate, even
though it was the single largest editable GPU kernel.
"""

import pytest

from TraceLens.PerfModel.extensions.moe_perf_model_extensions import (
    moe_triton_fused_gemm,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    resolve_perf_model_class,
)

# Real op name from a Qwen3-30B-A3B sglang trace; the trailing index is an
# unstable registration counter, hence pattern matching rather than exact name.
FUSED_MOE_OP = (
    "sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel_427"
)

# Gate/up (w1) projection launch: A[T,K], B=[E,N,K], C[T*topk, N].
W1_EVENT = {
    "name": FUSED_MOE_OP,
    "args": {
        "Input Dims": [
            (16384, 2048),
            (128, 1536, 2048),
            (),
            (131072, 1536),
            (),
            (),
            (),
            (16384, 8),
            (16384, 8),
            (139199,),
            (2175,),
            (1,),
            (),
        ],
        "Input type": [
            "c10::BFloat16",
            "c10::BFloat16",
            "",
            "c10::BFloat16",
            "",
            "",
            "",
            "float",
            "int",
            "int",
            "int",
            "int",
            "",
        ],
    },
}

# Down (w2) projection launch: A[T*topk, inter], B=[E, hidden, inter],
# C[T, topk, hidden] (3-D output).
W2_EVENT = {
    "name": FUSED_MOE_OP,
    "args": {
        "Input Dims": [
            (131072, 768),
            (128, 2048, 768),
            (),
            (16384, 8, 2048),
            (),
            (),
            (),
            (16384, 8),
            (16384, 8),
            (139199,),
            (2175,),
            (1,),
            (),
        ],
        "Input type": [
            "c10::BFloat16",
            "c10::BFloat16",
            "",
            "c10::BFloat16",
            "",
            "",
            "",
            "float",
            "int",
            "int",
            "int",
            "int",
            "",
        ],
    },
}


# ---------------------------------------------------------------------------
# Categorization + perf-model resolution (pattern-based, suffix-agnostic)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        FUSED_MOE_OP,
        "sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel",
        "vllm::fused_moe::invoke_fused_moe_kernel",
        "fused_moe::invoke_fused_moe_kernel_9",
    ],
)
def test_fused_moe_op_resolves_perf_model_and_category(name):
    assert resolve_perf_model_class(name) is moe_triton_fused_gemm, name
    # Was previously 'other'; now a roofline-capable MoE category.
    row = {"name": name, "kernel_details": [{"name": "fused_moe_kernel"}]}
    assert categorize_torch_op(row) == "MoE_fused", name


def test_non_fused_moe_ops_unaffected():
    # Sanity: unrelated ops keep their categories (no perf-model fallback hit).
    assert categorize_torch_op({"name": "aten::mm", "kernel_details": []}) == "GEMM"
    assert (
        categorize_torch_op({"name": "completely::unknown::op", "kernel_details": []})
        == "other"
    )


# ---------------------------------------------------------------------------
# Roofline: FLOPs / bytes for both expert GEMMs
# ---------------------------------------------------------------------------


def test_w1_gate_up_flops_and_bytes():
    model = moe_triton_fused_gemm(W1_EVENT)
    M, K, N, G = 131072, 2048, 1536, 128  # M = num_tokens(16384) * top_k(8)
    assert (model.M, model.K, model.N, model.G) == (M, K, N, G)
    assert model.flops() == 2 * M * K * N
    # GroupedGemm.bytes: (M*K + G*K*N)*bpe_in + M*N*bpe_out  (bf16 -> 2 bytes)
    assert model.bytes() == (M * K + G * K * N) * 2 + M * N * 2
    assert model.get_compute_precision() == "bf16"
    assert model.get_maf_type() == "matrix"


def test_w2_down_flops_and_bytes_with_3d_output():
    model = moe_triton_fused_gemm(W2_EVENT)
    M, K, N, G = 131072, 768, 2048, 128
    assert (model.M, model.K, model.N, model.G) == (M, K, N, G)
    assert model.flops() == 2 * M * K * N
    assert model.bytes() == (M * K + G * K * N) * 2 + M * N * 2


def test_contraction_dim_taken_from_activation_for_packed_weights():
    """W4A16/FP8 weights store a packed K; the logical K comes from activation A."""
    event = {
        "name": FUSED_MOE_OP,
        "args": {
            # A keeps the full K=2048; packed weight stores K=256 (2048 / 8).
            "Input Dims": [
                (16384, 2048),
                (128, 1536, 256),
                (),
                (131072, 1536),
                (),
                (),
                (),
                (16384, 8),
            ],
            "Input type": [
                "c10::BFloat16",
                "unsigned char",
                "",
                "c10::BFloat16",
                "",
                "",
                "",
                "int",
            ],
        },
    }
    model = moe_triton_fused_gemm(event)
    assert model.K == 2048  # logical contraction dim, not the packed 256
    assert model.N == 1536
    assert model.M == 131072
    assert model.flops() == 2 * 131072 * 2048 * 1536


def test_backward_not_supported():
    model = moe_triton_fused_gemm(W1_EVENT)
    with pytest.raises(NotImplementedError):
        model.flops_bwd()
    with pytest.raises(NotImplementedError):
        model.bytes_bwd()


def test_missing_weight_tensor_raises():
    event = {"name": FUSED_MOE_OP, "args": {"Input Dims": [(16384, 2048)], "Input type": ["c10::BFloat16"]}}
    with pytest.raises(ValueError):
        moe_triton_fused_gemm(event)
