###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for primus_turbo FP8 ops (#626):
  - primus_turbo_cpp_extension::hipblaslt_gemm_fp8 -> GEMM
  - primus_turbo_cpp_extension::quantize_fp8_tensorwise -> UnaryElementwise
"""

from TraceLens.PerfModel.perf_model import (
    hipblaslt_gemm_fp8,
    primus_turbo_quantize_fp8,
    GEMM,
    UnaryElementwise,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fp8_gemm_event(
    A_shape,
    B_shape,
    trans_a=False,
    trans_b=True,
    trans_c=False,
    dtype="c10::Float8_e4m3fnuz",
):
    """
    Build a hipblaslt_gemm_fp8 event with explicit transpose flags.

    Defaults reflect the typical Flux 12B weight layout:
      A is row-major (M,K), B is stored as (N,K) so transB=True logically
      means B is treated as (K,N) by the kernel.
    """
    return {
        "name": "primus_turbo_cpp_extension::hipblaslt_gemm_fp8",
        "args": {
            "Input Dims": [A_shape, (), B_shape, (), (), (), (), (), ()],
            "Input type": [
                dtype,
                "float",
                dtype,
                "float",
                "Scalar",
                "Scalar",
                "Scalar",
                "Scalar",
                "",
            ],
            "Concrete Inputs": [
                "",
                "",
                "",
                "",
                "8",
                "True" if trans_a else "False",
                "True" if trans_b else "False",
                "True" if trans_c else "False",
                "",
            ],
        },
    }


def _quantize_event(M=20480, N=12288, dtype_in="c10::BFloat16"):
    return {
        "name": "primus_turbo_cpp_extension::quantize_fp8_tensorwise",
        "args": {
            "Input Dims": [(M, N), (), ()],
            "Input type": [dtype_in, "Scalar", ""],
        },
    }


# ===========================================================================
# hipblaslt_gemm_fp8
# ===========================================================================


def test_hipblaslt_gemm_fp8_mapping():
    ops = [
        "primus_turbo_cpp_extension::hipblaslt_gemm_fp8",
        "primus_turbo::hipblaslt_gemm_fp8",
    ]
    for op in ops:
        assert op in op_to_perf_model_class_map, op
        assert op_to_perf_model_class_map[op] is hipblaslt_gemm_fp8, op


def test_hipblaslt_gemm_fp8_categorizes_as_gemm():
    for op in [
        "primus_turbo_cpp_extension::hipblaslt_gemm_fp8",
        "primus_turbo::hipblaslt_gemm_fp8",
    ]:
        row = {"name": op, "kernel_details": []}
        assert categorize_torch_op(row) == "GEMM", op


def test_hipblaslt_gemm_fp8_inherits_gemm():
    assert issubclass(hipblaslt_gemm_fp8, GEMM)


def test_hipblaslt_gemm_fp8_flops_double_stream_mlp_up_proj():
    """Flux 12B double-stream MLP up-proj: A=(20480,3072) x B=(12288,3072), transB=True."""
    M, K, N = 20480, 3072, 12288
    model = hipblaslt_gemm_fp8(_fp8_gemm_event((M, K), (N, K), trans_b=True))
    assert (model.M, model.N, model.K) == (M, N, K)
    assert model.flops() == 2 * M * N * K


def test_hipblaslt_gemm_fp8_bytes_double_stream_mlp_up_proj():
    M, K, N = 20480, 3072, 12288
    model = hipblaslt_gemm_fp8(_fp8_gemm_event((M, K), (N, K), trans_b=True))
    bpe_in = 1  # FP8
    bpe_out = 2  # FP16 output for FP8 input
    expected = M * K * bpe_in + K * N * bpe_in + M * N * bpe_out
    assert model.bytes() == expected


def test_hipblaslt_gemm_fp8_single_stream_attn_out_proj():
    """Flux 12B single-stream attn out-proj: A=(10240,3072) x B=(3072,3072)."""
    M, K, N = 10240, 3072, 3072
    model = hipblaslt_gemm_fp8(_fp8_gemm_event((M, K), (N, K), trans_b=True))
    assert (model.M, model.N, model.K) == (M, N, K)
    assert model.flops() == 2 * M * N * K
    assert model.bytes() == M * K * 1 + K * N * 1 + M * N * 2


def test_hipblaslt_gemm_fp8_weight_grad_shape():
    """NT layout weight grad: (3072, 20480) x (12288, 20480), transB=True."""
    M, K, N = 3072, 20480, 12288
    model = hipblaslt_gemm_fp8(_fp8_gemm_event((M, K), (N, K), trans_b=True))
    assert (model.M, model.N, model.K) == (M, N, K)
    assert model.flops() == 2 * M * N * K


def test_hipblaslt_gemm_fp8_no_transpose():
    """A=(M,K), B=(K,N), no transpose flags - A and B both row-major."""
    M, K, N = 1024, 512, 768
    model = hipblaslt_gemm_fp8(
        _fp8_gemm_event((M, K), (K, N), trans_a=False, trans_b=False)
    )
    assert (model.M, model.N, model.K) == (M, N, K)
    assert model.flops() == 2 * M * N * K


def test_hipblaslt_gemm_fp8_trans_a():
    """transA=True: A is stored as (K,M), kernel reads M from A_shape[1]."""
    M, K, N = 1024, 512, 768
    model = hipblaslt_gemm_fp8(
        _fp8_gemm_event((K, M), (K, N), trans_a=True, trans_b=False)
    )
    assert (model.M, model.N, model.K) == (M, N, K)
    assert model.flops() == 2 * M * N * K


def test_hipblaslt_gemm_fp8_trans_c_swap():
    """transC=True swaps A/B and inverts transA/transB (mirrors the C++ binding)."""
    M, K, N = 1024, 512, 768
    base = hipblaslt_gemm_fp8(
        _fp8_gemm_event((M, K), (K, N), trans_a=False, trans_b=False, trans_c=False)
    )
    swapped = hipblaslt_gemm_fp8(
        _fp8_gemm_event((K, N), (M, K), trans_a=True, trans_b=True, trans_c=True)
    )
    assert (base.M, base.N, base.K) == (swapped.M, swapped.N, swapped.K)


def test_hipblaslt_gemm_fp8_missing_concrete_inputs_defaults_to_no_transpose():
    """Older traces without Concrete Inputs default to transA=transB=False."""
    M, K, N = 1024, 512, 768
    event = {
        "name": "primus_turbo_cpp_extension::hipblaslt_gemm_fp8",
        "args": {
            "Input Dims": [(M, K), (), (K, N), (), (), (), (), (), ()],
            "Input type": [
                "c10::Float8_e4m3fnuz",
                "float",
                "c10::Float8_e4m3fnuz",
                "float",
                "Scalar",
                "Scalar",
                "Scalar",
                "Scalar",
                "",
            ],
        },
    }
    model = hipblaslt_gemm_fp8(event)
    assert (model.M, model.N, model.K) == (M, N, K)


def test_hipblaslt_gemm_fp8_null_concrete_inputs_defaults_to_no_transpose():
    """Some kineto exporters write Concrete Inputs as None (not absent); must not raise."""
    M, K, N = 1024, 512, 768
    event = {
        "name": "primus_turbo_cpp_extension::hipblaslt_gemm_fp8",
        "args": {
            "Input Dims": [(M, K), (), (K, N), (), (), (), (), (), ()],
            "Input type": [
                "c10::Float8_e4m3fnuz",
                "float",
                "c10::Float8_e4m3fnuz",
                "float",
                "Scalar",
                "Scalar",
                "Scalar",
                "Scalar",
                "",
            ],
            "Concrete Inputs": None,
        },
    }
    model = hipblaslt_gemm_fp8(event)
    assert (model.M, model.N, model.K) == (M, N, K)


# ===========================================================================
# quantize_fp8_tensorwise
# ===========================================================================


def test_quantize_fp8_mapping():
    ops = [
        "primus_turbo_cpp_extension::quantize_fp8_tensorwise",
        "primus_turbo::quantize_fp8_tensorwise",
    ]
    for op in ops:
        assert op in op_to_perf_model_class_map, op
        assert op_to_perf_model_class_map[op] is primus_turbo_quantize_fp8, op


def test_quantize_fp8_categorizes_as_elementwise():
    row = {
        "name": "primus_turbo_cpp_extension::quantize_fp8_tensorwise",
        "kernel_details": [],
    }
    assert categorize_torch_op(row) == "elementwise"


def test_quantize_fp8_inherits_unary():
    assert issubclass(primus_turbo_quantize_fp8, UnaryElementwise)


def test_quantize_fp8_bytes():
    M, N = 20480, 12288
    model = primus_turbo_quantize_fp8(_quantize_event(M=M, N=N))
    bpe_in = 2  # BF16
    bpe_out = 1  # FP8
    expected = M * N * bpe_in + M * N * bpe_out
    assert model.bytes() == expected


def test_quantize_fp8_flops():
    M, N = 20480, 12288
    model = primus_turbo_quantize_fp8(_quantize_event(M=M, N=N))
    assert model.flops() == M * N
