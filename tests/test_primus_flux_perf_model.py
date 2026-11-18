###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for Primus Turbo / Flux MXFP4 ops:
  - primus_flux::a6w4_mm_qa -> GEMM
  - primus_flux::a6w4_mm_qa_prequant_w -> GEMM
  - primus_flux::quantize_mxfp4_h16_dual -> UnaryElementwise
"""

from TraceLens.PerfModel.perf_model import GEMM
from TraceLens.PerfModel.extensions.primus_flux_perf_model_extensions import (
    a6w4_mm_qa,
    a6w4_mm_qa_prequant_w,
    quantize_mxfp4_h16_dual,
    _QuantizeMxfp4Base,
)
from TraceLens.PerfModel.torch_op_mapping import op_to_perf_model_class_map
from TraceLens.PerfModel.utils import MXFP4_GROUP_SIZE, name2bpe

# ---------------------------------------------------------------------------
# Helpers — synthetic trace events
# ---------------------------------------------------------------------------


def _a6w4_mm_qa_event(M=3072, N=3072, K=3072, has_bias=True):
    scale_elems = M * (K // MXFP4_GROUP_SIZE)
    return {
        "name": "primus_flux::a6w4_mm_qa",
        "args": {
            "Input Dims": [
                [M, K],  # aq (u8 activation)
                [scale_elems],  # sa (activation scales)
                [N, K],  # b (BF16 weight)
                [N] if has_bias else [],  # bias
            ],
            "Input type": [
                "unsigned char",
                "unsigned char",
                "c10::BFloat16",
                "c10::BFloat16" if has_bias else "",
            ],
            "Input Strides": [[], [], [], []],
            "Concrete Inputs": ["", "", "", ""],
        },
    }


def _a6w4_mm_qa_prequant_w_event(M=3072, N=3072, K=3072, has_bias=True):
    K_packed = K // 2
    K_scale = K // MXFP4_GROUP_SIZE
    scale_elems = M * K_scale
    return {
        "name": "primus_flux::a6w4_mm_qa_prequant_w",
        "args": {
            "Input Dims": [
                [M, K],  # aq (u8 activation)
                [scale_elems],  # sa (activation scales)
                [N, K_packed],  # wq (FP4 packed weight)
                [N, K_scale],  # sb (weight scales)
                [],  # Scalar (n)
                [N] if has_bias else [],  # bias
            ],
            "Input type": [
                "unsigned char",
                "unsigned char",
                "unsigned char",
                "unsigned char",
                "Scalar",
                "c10::BFloat16" if has_bias else "",
            ],
            "Input Strides": [[], [], [], [], [], []],
            "Concrete Inputs": ["", "", "", "", "", ""],
        },
    }


def _quantize_mxfp4_event(M=3072, N=3072):
    return {
        "name": "primus_flux::quantize_mxfp4_h16_dual",
        "args": {
            "Input Dims": [[M, N], [], []],
            "Input type": ["c10::BFloat16", "Scalar", "Scalar"],
            "Input Strides": [[N, 1], [], []],
            "Concrete Inputs": ["", "", ""],
        },
    }


# ===========================================================================
# Op registration
# ===========================================================================


def test_a6w4_mm_qa_registered():
    assert "primus_flux::a6w4_mm_qa" in op_to_perf_model_class_map
    assert op_to_perf_model_class_map["primus_flux::a6w4_mm_qa"] is a6w4_mm_qa


def test_a6w4_mm_qa_prequant_w_registered():
    assert "primus_flux::a6w4_mm_qa_prequant_w" in op_to_perf_model_class_map
    assert (
        op_to_perf_model_class_map["primus_flux::a6w4_mm_qa_prequant_w"]
        is a6w4_mm_qa_prequant_w
    )


def test_quantize_mxfp4_h16_dual_registered():
    assert "primus_flux::quantize_mxfp4_h16_dual" in op_to_perf_model_class_map
    assert (
        op_to_perf_model_class_map["primus_flux::quantize_mxfp4_h16_dual"]
        is quantize_mxfp4_h16_dual
    )


# ===========================================================================
# a6w4_mm_qa
# ===========================================================================


def test_a6w4_mm_qa_inherits_gemm():
    assert issubclass(a6w4_mm_qa, GEMM)


def test_a6w4_mm_qa_flops():
    M, N, K = 3072, 3072, 3072
    model = a6w4_mm_qa(_a6w4_mm_qa_event(M, N, K, has_bias=False))
    assert model.flops() == 2 * M * N * K


def test_a6w4_mm_qa_param_details():
    M, N, K = 4096, 2048, 6144
    model = a6w4_mm_qa(_a6w4_mm_qa_event(M, N, K))
    assert (model.M, model.N, model.K) == (M, N, K)


def test_a6w4_mm_qa_bytes():
    M, N, K = 3072, 3072, 3072
    model = a6w4_mm_qa(_a6w4_mm_qa_event(M, N, K, has_bias=True))
    bpe_u8 = 1
    bpe_bf16 = 2
    scale_elems = M * (K // MXFP4_GROUP_SIZE)
    expected = (
        M * K * bpe_u8  # activation
        + N * K * bpe_bf16  # weight
        + M * N * bpe_bf16  # output
        + N * bpe_bf16  # bias
        + scale_elems * bpe_u8  # activation scales
    )
    assert model.bytes() == expected


def test_a6w4_mm_qa_no_bias():
    M, N, K = 3072, 3072, 3072
    model = a6w4_mm_qa(_a6w4_mm_qa_event(M, N, K, has_bias=False))
    bpe_u8 = 1
    bpe_bf16 = 2
    scale_elems = M * (K // MXFP4_GROUP_SIZE)
    expected = (
        M * K * bpe_u8 + N * K * bpe_bf16 + M * N * bpe_bf16 + scale_elems * bpe_u8
    )
    assert model.bytes() == expected


# ===========================================================================
# a6w4_mm_qa_prequant_w
# ===========================================================================


def test_a6w4_mm_qa_prequant_w_inherits_gemm():
    assert issubclass(a6w4_mm_qa_prequant_w, GEMM)


def test_a6w4_mm_qa_prequant_w_flops():
    M, N, K = 3072, 3072, 3072
    model = a6w4_mm_qa_prequant_w(_a6w4_mm_qa_prequant_w_event(M, N, K, has_bias=False))
    assert model.flops() == 2 * M * N * K


def test_a6w4_mm_qa_prequant_w_param_details():
    M, N, K = 4096, 2048, 6144
    model = a6w4_mm_qa_prequant_w(_a6w4_mm_qa_prequant_w_event(M, N, K))
    assert (model.M, model.N, model.K) == (M, N, K)


def test_a6w4_mm_qa_prequant_w_bytes():
    M, N, K = 3072, 3072, 3072
    model = a6w4_mm_qa_prequant_w(_a6w4_mm_qa_prequant_w_event(M, N, K, has_bias=True))
    bpe_u8 = 1
    bpe_bf16 = 2
    expected = (
        M * K * bpe_u8  # activation (u8)
        + N * K * bpe_u8  # weight (u8 packed, bpe from trace = unsigned char = 1)
        + M * N * bpe_bf16  # output
        + N * bpe_bf16  # bias
    )
    assert model.bytes() == expected


# ===========================================================================
# quantize_mxfp4_h16_dual
# ===========================================================================


def test_quantize_mxfp4_dual_inherits_base():
    assert issubclass(quantize_mxfp4_h16_dual, _QuantizeMxfp4Base)


def test_quantize_mxfp4_dual_flops():
    M, N = 3072, 3072
    model = quantize_mxfp4_h16_dual(_quantize_mxfp4_event(M, N))
    assert model.flops() == 4 * M * N


def test_quantize_mxfp4_dual_bytes():
    M, N = 3072, 3072
    model = quantize_mxfp4_h16_dual(_quantize_mxfp4_event(M, N))
    bpe_in = 2  # BF16
    bpe_fp4 = 0.5
    bpe_scale = 1
    expected = (
        M * N * bpe_in  # read
        + 2 * M * N * bpe_fp4  # write FP4 (dual)
        + 2 * max(1, M * N // MXFP4_GROUP_SIZE) * bpe_scale  # write scales (dual)
    )
    assert model.bytes() == expected


# ===========================================================================
# name2bpe for FP4 types
# ===========================================================================


def test_name2bpe_fp4_returns_half():
    assert name2bpe("fp4") == 0.5


def test_name2bpe_float4_e2m1fn_x2_returns_one():
    assert name2bpe("c10::Float4_e2m1fn_x2") == 1


def test_name2bpe_e8m0_scale():
    assert name2bpe("c10::Float8_e8m0fnu") == 1


def test_mxfp4_group_size_is_32():
    assert MXFP4_GROUP_SIZE == 32


def test_a6w4_mm_qa_compute_precision():
    from TraceLens.PerfModel.utils import torch_dtype_map

    model = a6w4_mm_qa(_a6w4_mm_qa_event())
    assert model.get_compute_precision() == torch_dtype_map("mxfp4")


def test_a6w4_mm_qa_prequant_w_compute_precision():
    from TraceLens.PerfModel.utils import torch_dtype_map

    model = a6w4_mm_qa_prequant_w(_a6w4_mm_qa_prequant_w_event())
    assert model.get_compute_precision() == torch_dtype_map("mxfp4")


def test_quantize_mxfp4_base_flops_single():
    """_QuantizeMxfp4Base.flops() returns 2 * nelems for single quantization."""
    model = quantize_mxfp4_h16_dual(_quantize_mxfp4_event(M=16, N=32))
    base_flops = _QuantizeMxfp4Base.flops(model)
    assert base_flops == 2 * 16 * 32


def test_quantize_mxfp4_base_bytes_single():
    """_QuantizeMxfp4Base.bytes() returns read + single FP4 write + single scale write."""
    model = quantize_mxfp4_h16_dual(_quantize_mxfp4_event(M=32, N=64))
    base_bytes = _QuantizeMxfp4Base.bytes(model)
    bpe_in = 2
    bpe_fp4 = 0.5
    bpe_scale = 1
    expected = 32 * 64 * bpe_in + 32 * 64 * bpe_fp4 + max(1, 32 * 64 // 32) * bpe_scale
    assert base_bytes == expected
