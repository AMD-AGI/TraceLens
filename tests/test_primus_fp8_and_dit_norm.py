###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for primus_turbo FP8 ops (#626) and primus::fused_ln_modulate (#627).
"""

from TraceLens.PerfModel.perf_model import (
    hipblaslt_gemm_fp8,
    primus_turbo_quantize_fp8,
    FusedLnModulate,
    FusedLnModulateBackward,
    GEMM,
    UnaryElementwise,
    Normalization,
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


def _fused_ln_fwd_event(T=256, B=40, H=3072, dtype="c10::BFloat16"):
    return {
        "name": "primus::fused_ln_modulate",
        "args": {
            "Input Dims": [(T, B, H), (B, H), (B, H), ()],
            "Input type": [dtype, dtype, dtype, "Scalar"],
        },
    }


def _fused_ln_bwd_event(T=256, B=40, H=3072, dtype="c10::BFloat16"):
    return {
        "name": "primus::fused_ln_modulate_backward",
        "args": {
            "Input Dims": [
                (T, B, H),
                (T, B, H),
                (B * T,),
                (B * T,),
                (B, H),
            ],
            "Input type": [dtype, dtype, "float", "float", dtype],
        },
    }


# ===========================================================================
# #626 — hipblaslt_gemm_fp8
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


# ===========================================================================
# #626 — quantize_fp8_tensorwise
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


# ===========================================================================
# #627 — fused_ln_modulate forward
# ===========================================================================


def test_fused_ln_modulate_mapping():
    assert "primus::fused_ln_modulate" in op_to_perf_model_class_map
    assert op_to_perf_model_class_map["primus::fused_ln_modulate"] is FusedLnModulate


def test_fused_ln_modulate_categorizes_as_norm_fwd():
    row = {"name": "primus::fused_ln_modulate", "kernel_details": []}
    assert categorize_torch_op(row) == "NORM_fwd"


def test_fused_ln_modulate_inherits_normalization():
    assert issubclass(FusedLnModulate, Normalization)


def test_fused_ln_modulate_flops():
    T, B, H = 256, 40, 3072
    model = FusedLnModulate(_fused_ln_fwd_event(T=T, B=B, H=H))
    assert model.flops() == T * B * H * 12


def test_fused_ln_modulate_bytes():
    T, B, H = 256, 40, 3072
    bpe = 2  # BF16
    model = FusedLnModulate(_fused_ln_fwd_event(T=T, B=B, H=H))
    read_x = T * B * H * bpe
    write_y = T * B * H * bpe
    read_scale_shift = 2 * B * H * bpe
    write_mean_rstd = 2 * T * B * 4  # float32, one per (T,B) row
    expected = read_x + write_y + read_scale_shift + write_mean_rstd
    assert model.bytes() == expected


def test_fused_ln_modulate_double_stream():
    """T=512 double-stream block."""
    T, B, H = 512, 40, 3072
    bpe = 2
    model = FusedLnModulate(_fused_ln_fwd_event(T=T, B=B, H=H))
    assert model.flops() == T * B * H * 12
    expected_bytes = 2 * T * B * H * bpe + 2 * B * H * bpe + 2 * T * B * 4
    assert model.bytes() == expected_bytes


# ===========================================================================
# #627 — fused_ln_modulate_backward
# ===========================================================================


def test_fused_ln_modulate_bwd_mapping():
    assert "primus::fused_ln_modulate_backward" in op_to_perf_model_class_map
    assert (
        op_to_perf_model_class_map["primus::fused_ln_modulate_backward"]
        is FusedLnModulateBackward
    )


def test_fused_ln_modulate_bwd_categorizes_as_norm_bwd():
    row = {"name": "primus::fused_ln_modulate_backward", "kernel_details": []}
    assert categorize_torch_op(row) == "NORM_bwd"


def test_fused_ln_modulate_bwd_inherits_normalization():
    assert issubclass(FusedLnModulateBackward, Normalization)


def test_fused_ln_modulate_bwd_flops():
    T, B, H = 256, 40, 3072
    model = FusedLnModulateBackward(_fused_ln_bwd_event(T=T, B=B, H=H))
    assert model.flops() == T * B * H * 15


def test_fused_ln_modulate_bwd_bytes():
    T, B, H = 256, 40, 3072
    bpe = 2
    model = FusedLnModulateBackward(_fused_ln_bwd_event(T=T, B=B, H=H))
    read_grad_x_norm = 2 * T * B * H * bpe
    read_mean_rstd = 2 * B * T * 4  # float32
    write_x_grad = T * B * H * bpe
    write_scale_shift_grad = 2 * B * H * bpe
    read_mod_grad = B * H * bpe
    expected = (
        read_grad_x_norm
        + read_mean_rstd
        + write_x_grad
        + write_scale_shift_grad
        + read_mod_grad
    )
    assert model.bytes() == expected


def test_fused_ln_modulate_bwd_double_stream():
    """T=512 double-stream block, BF16."""
    T, B, H = 512, 40, 3072
    bpe = 2
    model = FusedLnModulateBackward(_fused_ln_bwd_event(T=T, B=B, H=H))
    assert model.flops() == T * B * H * 15
    expected = (
        2 * T * B * H * bpe
        + 2 * B * T * 4
        + T * B * H * bpe
        + 2 * B * H * bpe
        + B * H * bpe
    )
    assert model.bytes() == expected
