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

from TraceLens.PerfModel.perf_model import GEMM, SDPA
from TraceLens.PerfModel.extensions.primus_flux_perf_model_extensions import (
    flux_flydsl_attn_fwd,
    flux_flydsl_attn_fwd_split,
    flux_flydsl_attn_bwd,
    flux_flydsl_attn_bwd_split,
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
    K_packed = K // 2
    expected = (
        M * K * bpe_u8  # activation (u8)
        + N
        * K_packed
        * bpe_u8  # weight (FP4 packed as unsigned char, traced shape [N, K//2])
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


# ===========================================================================
# FlydSL attention helpers
# ===========================================================================


def _flydsl_attn_fwd_event(B=32, S=512, H=24, D=128):
    shape = [B, S, H, D]
    stride = [S * H * D, H * D, D, 1]
    return {
        "name": "flux_flydsl::attn_fwd",
        "args": {
            "Input Dims": [shape, shape, shape],
            "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
            "Input Strides": [stride, stride, stride],
            "Concrete Inputs": ["", "", ""],
        },
    }


def _flydsl_attn_fwd_split_event(B=32, S=512, H=24, D=128, split=256):
    shape = [B, S, H, D]
    stride = [S * H * D, H * D, D, 1]
    return {
        "name": "flux_flydsl::attn_fwd_split",
        "args": {
            "Input Dims": [shape, shape, shape, []],
            "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16", "Scalar"],
            "Input Strides": [stride, stride, stride, []],
            "Concrete Inputs": ["", "", "", str(split)],
        },
    }


def _flydsl_attn_bwd_event(B=32, S=512, H=24, D=128):
    shape = [B, S, H, D]
    stride = [S * H * D, H * D, D, 1]
    lse_shape = [B, H, S]
    lse_stride = [H * S, S, 1]
    return {
        "name": "flux_flydsl::attn_bwd",
        "args": {
            "Input Dims": [shape, shape, shape, shape, shape, lse_shape],
            "Input type": [
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "float",
            ],
            "Input Strides": [stride, stride, stride, stride, stride, lse_stride],
            "Concrete Inputs": ["", "", "", "", "", ""],
        },
    }


def _flydsl_attn_bwd_split_event(B=32, S=512, H=24, D=128):
    half_S = S // 2
    full_shape = [B, S, H, D]
    half_shape = [B, half_S, H, D]
    full_stride = [S * H * D, H * D, D, 1]
    half_stride = [half_S * H * D, H * D, D, 1]
    lse_shape = [B, H, S]
    lse_stride = [H * S, S, 1]
    return {
        "name": "flux_flydsl::attn_bwd_split",
        "args": {
            "Input Dims": [
                half_shape,
                half_shape,  # dOut_x, dOut_img
                full_shape,
                full_shape,
                full_shape,  # K, V, Out
                half_shape,
                half_shape,  # Q_x, Q_img
                lse_shape,  # LSE
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "float",
            ],
            "Input Strides": [
                half_stride,
                half_stride,
                full_stride,
                full_stride,
                full_stride,
                half_stride,
                half_stride,
                lse_stride,
            ],
            "Concrete Inputs": [""] * 8,
        },
    }


# ===========================================================================
# FlydSL attention registration
# ===========================================================================


def test_flydsl_attn_fwd_registered():
    assert "flux_flydsl::attn_fwd" in op_to_perf_model_class_map
    assert op_to_perf_model_class_map["flux_flydsl::attn_fwd"] is flux_flydsl_attn_fwd


def test_flydsl_attn_fwd_split_registered():
    assert "flux_flydsl::attn_fwd_split" in op_to_perf_model_class_map


def test_flydsl_attn_bwd_registered():
    assert "flux_flydsl::attn_bwd" in op_to_perf_model_class_map


def test_flydsl_attn_bwd_split_registered():
    assert "flux_flydsl::attn_bwd_split" in op_to_perf_model_class_map


# ===========================================================================
# FlydSL attn_fwd
# ===========================================================================


def test_flydsl_attn_fwd_inherits_sdpa():
    assert issubclass(flux_flydsl_attn_fwd, SDPA)


def test_flydsl_attn_fwd_category():
    assert flux_flydsl_attn_fwd.category == "SDPA_fwd"


def test_flydsl_attn_fwd_params():
    model = flux_flydsl_attn_fwd(_flydsl_attn_fwd_event(B=32, S=512, H=24, D=128))
    assert (
        model.B,
        model.N_Q,
        model.H_Q,
        model.N_KV,
        model.H_KV,
        model.d_h_qk,
        model.d_h_v,
    ) == (32, 512, 24, 512, 24, 128, 128)


def test_flydsl_attn_fwd_flops():
    B, S, H, D = 32, 512, 24, 128
    model = flux_flydsl_attn_fwd(_flydsl_attn_fwd_event(B, S, H, D))
    expected = 4 * B * H * S * S * D  # non-causal
    assert model.flops() == expected


def test_flydsl_attn_fwd_bytes():
    B, S, H, D = 32, 512, 24, 128
    model = flux_flydsl_attn_fwd(_flydsl_attn_fwd_event(B, S, H, D))
    bpe = 2  # BF16
    expected = (3 * B * S * H * D + B * S * H * D) * bpe  # Q+K+V read + O write
    assert model.bytes() == expected


# ===========================================================================
# FlydSL attn_fwd_split
# ===========================================================================


def test_flydsl_attn_fwd_split_same_flops_as_fwd():
    B, S, H, D = 32, 512, 24, 128
    fwd = flux_flydsl_attn_fwd(_flydsl_attn_fwd_event(B, S, H, D))
    split = flux_flydsl_attn_fwd_split(_flydsl_attn_fwd_split_event(B, S, H, D))
    assert fwd.flops() == split.flops()


def test_flydsl_attn_fwd_split_same_bytes_as_fwd():
    B, S, H, D = 32, 512, 24, 128
    fwd = flux_flydsl_attn_fwd(_flydsl_attn_fwd_event(B, S, H, D))
    split = flux_flydsl_attn_fwd_split(_flydsl_attn_fwd_split_event(B, S, H, D))
    assert fwd.bytes() == split.bytes()


# ===========================================================================
# FlydSL attn_bwd
# ===========================================================================


def test_flydsl_attn_bwd_inherits_sdpa():
    assert issubclass(flux_flydsl_attn_bwd, SDPA)


def test_flydsl_attn_bwd_category():
    assert flux_flydsl_attn_bwd.category == "SDPA_bwd"


def test_flydsl_attn_bwd_params():
    model = flux_flydsl_attn_bwd(_flydsl_attn_bwd_event(B=32, S=512, H=24, D=128))
    assert (model.B, model.N_Q, model.H_Q, model.N_KV) == (32, 512, 24, 512)


def test_flydsl_attn_bwd_flops():
    B, S, H, D = 32, 512, 24, 128
    model = flux_flydsl_attn_bwd(_flydsl_attn_bwd_event(B, S, H, D))
    fwd_flops = 4 * B * H * S * S * D
    expected = int(fwd_flops * 2.5)  # bwd = 5/2 * fwd (5 matmuls with flash recompute)
    assert model.flops() == expected


def test_flydsl_attn_bwd_bytes():
    B, S, H, D = 32, 512, 24, 128
    model = flux_flydsl_attn_bwd(_flydsl_attn_bwd_event(B, S, H, D))
    bpe = 2
    expected = 7 * B * S * H * D * bpe  # Q,K,V,dO reads + dQ,dK,dV writes
    assert model.bytes() == expected


# ===========================================================================
# FlydSL attn_bwd_split
# ===========================================================================


def test_flydsl_attn_bwd_split_category():
    assert flux_flydsl_attn_bwd_split.category == "SDPA_bwd"


def test_flydsl_attn_bwd_split_uses_full_seq():
    model = flux_flydsl_attn_bwd_split(
        _flydsl_attn_bwd_split_event(B=32, S=512, H=24, D=128)
    )
    assert model.N_Q == 512
    assert model.N_KV == 512


def test_flydsl_attn_bwd_split_same_flops_as_bwd():
    B, S, H, D = 32, 512, 24, 128
    bwd = flux_flydsl_attn_bwd(_flydsl_attn_bwd_event(B, S, H, D))
    split = flux_flydsl_attn_bwd_split(_flydsl_attn_bwd_split_event(B, S, H, D))
    assert bwd.flops() == split.flops()
