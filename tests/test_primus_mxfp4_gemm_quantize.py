###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for primus_turbo and aiter MXFP4 ops (#637, #644):
  - primus_turbo_cpp_extension::hipblaslt_gemm_fp4 -> GEMM
  - primus_turbo::gemm_fp4_impl -> GEMM (Flux trace leaf)
  - primus_turbo_cpp_extension::quantize_mxfp4_dual -> UnaryElementwise
  - aiter::gemm_a4w4 / aiter::_gemm_a4w4_asm -> GEMM (AITER FP4 backend)
"""

import pytest

from TraceLens.PerfModel.perf_model import (
    aiter_gemm_a4w4,
    gemm_fp4_impl,
    hipblaslt_gemm_fp4,
    primus_turbo_quantize_mxfp4_dual,
    GEMM,
    UnaryElementwise,
    FP4GemmMXFunction,
    FP4GemmMXFunctionBackward,
    FP4GroupedMLPMXFunc,
    FP4GroupedMLPMXFuncBackward,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)
from TraceLens.PerfModel.utils import MXFP4_GROUP_SIZE, name2bpe, torch_dtype_map

BF16 = "c10::BFloat16"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fp4_gemm_event(
    A_shape,
    B_shape,
    trans_a=False,
    trans_b=True,
    trans_c=False,
    dtype="c10::Float4_e2m1fn_x2",
    scale_dtype="c10::Float8_e8m0fnu",
    scaleA_shape=(),
    scaleB_shape=(),
    granularity_str="MX_BLOCKWISE",
):
    """
    Build a hipblaslt_gemm_fp4 event. Shape semantics mirror the C++ binding:
      A has shape (M, K_packed) in the default NT layout where K_packed = K/2.
    """
    return {
        "name": "primus_turbo_cpp_extension::hipblaslt_gemm_fp4",
        "args": {
            "Input Dims": [
                A_shape,
                scaleA_shape,
                B_shape,
                scaleB_shape,
                (),
                (),
                (),
                (),
                (),
            ],
            "Input type": [
                dtype,
                scale_dtype,
                dtype,
                scale_dtype,
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
                "15",  # out_dtype enum (BF16)
                "True" if trans_a else "False",
                "True" if trans_b else "False",
                "True" if trans_c else "False",
                granularity_str,
            ],
        },
    }


def _mxfp4_quantize_event(M=16384, N=12288, dtype_in="c10::BFloat16"):
    return {
        "name": "primus_turbo_cpp_extension::quantize_mxfp4_dual",
        "args": {
            "Input Dims": [(M, N)] + [()] * 11,
            "Input type": [dtype_in] + ["Scalar"] * 11,
            "Concrete Inputs": ["", "45"] + ["False"] * 10,
        },
    }


# ===========================================================================
# name2bpe — new FP4 / E8M0 dtypes
# ===========================================================================


def test_name2bpe_float4_e2m1fn_x2():
    # Packed FP4 pair: one byte per element of the traced tensor.
    assert name2bpe("c10::Float4_e2m1fn_x2") == 1
    # One logical value, for callers that have already unpacked K.
    assert name2bpe("fp4") == 0.5


def test_name2bpe_e8m0_scale():
    # MXFP4 scale factor: 1 byte per scale element.
    assert name2bpe("c10::Float8_e8m0fnu") == 1


# ===========================================================================
# hipblaslt_gemm_fp4
# ===========================================================================


def test_hipblaslt_gemm_fp4_mapping():
    ops = [
        "primus_turbo_cpp_extension::hipblaslt_gemm_fp4",
        "primus_turbo::hipblaslt_gemm_fp4",
    ]
    for op in ops:
        assert op in op_to_perf_model_class_map, op
        assert op_to_perf_model_class_map[op] is hipblaslt_gemm_fp4, op


def test_hipblaslt_gemm_fp4_categorizes_as_gemm():
    for op in [
        "primus_turbo_cpp_extension::hipblaslt_gemm_fp4",
        "primus_turbo::hipblaslt_gemm_fp4",
    ]:
        row = {"name": op, "kernel_details": []}
        assert categorize_torch_op(row) == "GEMM", op


def test_hipblaslt_gemm_fp4_inherits_gemm():
    assert issubclass(hipblaslt_gemm_fp4, GEMM)


def test_hipblaslt_gemm_fp4_flops_double_stream_mlp_up_proj():
    """Flux 12B double-stream MLP up-proj observed in trace:
    A=(16384, 1536) x B=(12288, 1536), K_packed=1536 -> K=3072."""
    M, K, N = 16384, 3072, 12288
    K_packed = K // 2
    model = hipblaslt_gemm_fp4(
        _fp4_gemm_event(
            (M, K_packed),
            (N, K_packed),
            trans_b=True,
            scaleA_shape=(M, K // 32),
            scaleB_shape=(N, K // 32),
        )
    )
    assert (model.M, model.N, model.K) == (M, N, K)
    assert model.flops() == 2 * M * N * K


def test_hipblaslt_gemm_fp4_bytes_includes_block_scales():
    """Bytes must include packed FP4 operands, BF16 output, and E8M0 scales."""
    M, K, N = 16384, 3072, 12288
    K_packed = K // 2
    model = hipblaslt_gemm_fp4(
        _fp4_gemm_event(
            (M, K_packed),
            (N, K_packed),
            trans_b=True,
            scaleA_shape=(M, K // 32),
            scaleB_shape=(N, K // 32),
        )
    )
    bpe = name2bpe("c10::Float4_e2m1fn_x2")
    bytes_A = M * (K // 2) * bpe
    bytes_B = N * (K // 2) * bpe
    bytes_scaleA = M * (K // 32)
    bytes_scaleB = N * (K // 32)
    bytes_out = M * N * 2  # BF16 output
    assert model.bytes() == bytes_A + bytes_B + bytes_scaleA + bytes_scaleB + bytes_out


def test_hipblaslt_gemm_fp4_weight_grad_shape():
    """Flux 12B weight-grad pattern: A=(3072, 8192) x B=(12288, 8192),
    K_packed=8192 -> K=16384."""
    M, K, N = 3072, 16384, 12288
    K_packed = K // 2
    model = hipblaslt_gemm_fp4(
        _fp4_gemm_event((M, K_packed), (N, K_packed), trans_b=True)
    )
    assert (model.M, model.N, model.K) == (M, N, K)
    assert model.flops() == 2 * M * N * K


def test_hipblaslt_gemm_fp4_trans_c_swap():
    """transC=True swaps A/B and inverts transA/transB (mirrors the C++ binding)."""
    M, K, N = 1024, 512, 768
    K_packed = K // 2
    base = hipblaslt_gemm_fp4(
        _fp4_gemm_event(
            (M, K_packed),
            (N, K_packed),
            trans_a=False,
            trans_b=True,
            trans_c=False,
        )
    )
    # After C++ swap of (A↔B, transA↔¬transB, transB↔¬transA), reconstructing
    # the same logical (M, K, N) from transposed inputs must yield identical
    # post-swap shapes.
    swapped = hipblaslt_gemm_fp4(
        _fp4_gemm_event(
            (N, K_packed),
            (M, K_packed),
            trans_a=False,
            trans_b=True,
            trans_c=True,
        )
    )
    assert (base.M, base.N, base.K) == (swapped.M, swapped.N, swapped.K)


def test_hipblaslt_gemm_fp4_bool_parser_accepts_bool_and_lowercase_strings():
    M, K, N = 1024, 512, 768
    K_packed = K // 2
    event = _fp4_gemm_event((K_packed, M), (K_packed, N), trans_b=False)
    event["args"]["Concrete Inputs"][5] = True
    event["args"]["Concrete Inputs"][6] = "false"
    event["args"]["Concrete Inputs"][7] = ""

    model = hipblaslt_gemm_fp4(event)

    assert (model.M, model.N, model.K) == (M, N, K)


def test_hipblaslt_gemm_fp4_missing_concrete_inputs_defaults_to_no_transpose():
    """Older traces without Concrete Inputs default to transA=transB=False.
    K_packed in the trace dim is still unpacked via ×2."""
    M, K, N = 1024, 512, 768
    K_packed = K // 2
    event = {
        "name": "primus_turbo_cpp_extension::hipblaslt_gemm_fp4",
        "args": {
            "Input Dims": [
                (M, K_packed),
                (),
                (K_packed, N),
                (),
                (),
                (),
                (),
                (),
                (),
            ],
            "Input type": [
                "c10::Float4_e2m1fn_x2",
                "c10::Float8_e8m0fnu",
                "c10::Float4_e2m1fn_x2",
                "c10::Float8_e8m0fnu",
                "Scalar",
                "Scalar",
                "Scalar",
                "Scalar",
                "",
            ],
        },
    }
    model = hipblaslt_gemm_fp4(event)
    assert (model.M, model.N, model.K) == (M, N, K)


def test_hipblaslt_gemm_fp4_null_concrete_inputs_defaults_to_no_transpose():
    """Some kineto exporters write Concrete Inputs as None (not absent)."""
    M, K, N = 1024, 512, 768
    K_packed = K // 2
    event = {
        "name": "primus_turbo_cpp_extension::hipblaslt_gemm_fp4",
        "args": {
            "Input Dims": [
                (M, K_packed),
                (),
                (K_packed, N),
                (),
                (),
                (),
                (),
                (),
                (),
            ],
            "Input type": [
                "c10::Float4_e2m1fn_x2",
                "c10::Float8_e8m0fnu",
                "c10::Float4_e2m1fn_x2",
                "c10::Float8_e8m0fnu",
                "Scalar",
                "Scalar",
                "Scalar",
                "Scalar",
                "",
            ],
            "Concrete Inputs": None,
        },
    }
    model = hipblaslt_gemm_fp4(event)
    assert (model.M, model.N, model.K) == (M, N, K)


def test_hipblaslt_gemm_fp4_k_unpacked_factor_of_2():
    """K_actual must always be 2× the traced K dimension (FP4 packing)."""
    M, K_packed, N = 16384, 1536, 12288
    model = hipblaslt_gemm_fp4(
        _fp4_gemm_event((M, K_packed), (N, K_packed), trans_b=True)
    )
    assert model.K == K_packed * 2


# ===========================================================================
# quantize_mxfp4_dual
# ===========================================================================


def test_quantize_mxfp4_dual_mapping():
    ops = [
        "primus_turbo_cpp_extension::quantize_mxfp4_dual",
        "primus_turbo::quantize_mxfp4_dual",
        "primus::quantize_mxfp4_dual",
    ]
    for op in ops:
        assert op in op_to_perf_model_class_map, op
        assert op_to_perf_model_class_map[op] is primus_turbo_quantize_mxfp4_dual, op


def test_quantize_mxfp4_dual_categorizes_as_elementwise():
    row = {
        "name": "primus_turbo_cpp_extension::quantize_mxfp4_dual",
        "kernel_details": [],
    }
    assert categorize_torch_op(row) == "elementwise"


def test_quantize_mxfp4_dual_inherits_unary():
    assert issubclass(primus_turbo_quantize_mxfp4_dual, UnaryElementwise)


def test_quantize_mxfp4_dual_bytes_dual_output():
    """Reads BF16 input once; writes both row- and col-packed FP4 outputs and
    both 1-D E8M0 block-scale tensors (1 byte per 32-element block)."""
    M, N = 16384, 12288
    model = primus_turbo_quantize_mxfp4_dual(_mxfp4_quantize_event(M=M, N=N))
    bpe_in = 2  # BF16
    read_in = M * N * bpe_in
    write_rowwise_fp4 = (M * N + 1) // 2
    write_colwise_fp4 = (N * M + 1) // 2
    write_rowwise_scale = M * (N // 32)
    write_colwise_scale = N * (M // 32)
    expected = (
        read_in
        + write_rowwise_fp4
        + write_colwise_fp4
        + write_rowwise_scale
        + write_colwise_scale
    )
    assert model.bytes() == expected


def test_quantize_mxfp4_dual_bytes_scales_use_ceil_div():
    """Block-scale count is ceil(N/32), not floor — covers shapes where N
    is not an exact multiple of MXFP4_BLOCK_SIZE."""
    M, N = 8, 33  # N=33 → ceil(33/32)=2 scales per row
    model = primus_turbo_quantize_mxfp4_dual(_mxfp4_quantize_event(M=M, N=N))
    bpe_in = 2
    expected = (
        M * N * bpe_in
        + (M * N + 1) // 2
        + (N * M + 1) // 2
        + M * 2  # ceil(33/32)
        + N * 1  # ceil(8/32)
    )
    assert model.bytes() == expected


def test_quantize_mxfp4_dual_bytes_odd_element_count():
    """Ceil-div for packed FP4 writes matters when M*N is odd."""
    M, N = 1, 33
    model = primus_turbo_quantize_mxfp4_dual(_mxfp4_quantize_event(M=M, N=N))
    bpe_in = 2
    expected = (
        M * N * bpe_in
        + (M * N + 1) // 2
        + (N * M + 1) // 2
        + M * ((N + 32 - 1) // 32)
        + N * ((M + 32 - 1) // 32)
    )
    assert model.bytes() == expected


def test_quantize_mxfp4_dual_flops():
    """No real arithmetic; the model counts the input element count, mirroring
    the FP8 quantize counterpart."""
    M, N = 16384, 12288
    model = primus_turbo_quantize_mxfp4_dual(_mxfp4_quantize_event(M=M, N=N))
    assert model.flops() == M * N


# ===========================================================================
# aiter::gemm_a4w4 (AITER MXFP4 backend — issue #644)
# ===========================================================================


def _aiter_gemm_a4w4_event(
    A_shape,
    B_shape,
    scaleA_shape=(),
    scaleB_shape=(),
    bpreshuffle=True,
    name="aiter::gemm_a4w4",
):
    """Build an aiter::gemm_a4w4 event. The aiter binding only supports NT
    (transA=False, transB=True); A has shape (M, K_packed), B has shape
    (N, K_packed). Slot ordering differs from hipblaslt_gemm_fp4:
    (A, B, scaleA, scaleB) instead of (A, scaleA, B, scaleB)."""
    return {
        "name": name,
        "args": {
            "Input Dims": [
                A_shape,
                B_shape,
                scaleA_shape,
                scaleB_shape,
                (),
                (),
                (),
                (),
                (),
            ],
            "Input type": [
                "c10::Float4_e2m1fn_x2",
                "c10::Float4_e2m1fn_x2",
                "c10::Float8_e8m0fnu",
                "c10::Float8_e8m0fnu",
                "",
                "Scalar",
                "Scalar",
                "Scalar",
                "Scalar",
            ],
            "Concrete Inputs": [
                "",
                "",
                "",
                "",
                "",
                "15",  # dtype enum (BF16)
                "1.",  # alpha
                "0.",  # beta
                "True" if bpreshuffle else "False",
            ],
        },
    }


def _aiter_gemm_a4w4_asm_event(
    A_shape,
    B_shape,
    scaleA_shape=(),
    scaleB_shape=(),
):
    """Build an aiter::_gemm_a4w4_asm event. 11 slots: same first 4 tensors
    as the public entry, plus an output tensor at slot 4 (BF16), plus 6
    trailing scalars."""
    M = A_shape[0]
    N = B_shape[0]
    return {
        "name": "aiter::_gemm_a4w4_asm",
        "args": {
            "Input Dims": [
                A_shape,
                B_shape,
                scaleA_shape,
                scaleB_shape,
                (M, N),
                (),
                (),
                (),
                (),
                (),
                (),
            ],
            "Input type": [
                "c10::Float4_e2m1fn_x2",
                "c10::Float4_e2m1fn_x2",
                "c10::Float8_e8m0fnu",
                "c10::Float8_e8m0fnu",
                "c10::BFloat16",
                "",
                "",
                "Scalar",
                "Scalar",
                "Scalar",
                "Scalar",
            ],
            "Concrete Inputs": [
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "1.",
                "0.",
                "1",
                "0",
            ],
        },
    }


def test_aiter_gemm_a4w4_mapping():
    for op in ["aiter::gemm_a4w4", "aiter::_gemm_a4w4_asm"]:
        assert op in op_to_perf_model_class_map, op
        assert op_to_perf_model_class_map[op] is aiter_gemm_a4w4, op


def test_aiter_gemm_a4w4_categorizes_as_gemm():
    for op in ["aiter::gemm_a4w4", "aiter::_gemm_a4w4_asm"]:
        row = {"name": op, "kernel_details": []}
        assert categorize_torch_op(row) == "GEMM", op


def test_aiter_gemm_a4w4_inherits_gemm_directly():
    """Inherits directly from GEMM (per the op-registry's single-base
    convention). The bytes() implementation is shared with hipblaslt_gemm_fp4
    via delegation — see test_aiter_gemm_a4w4_bytes_match_hipblaslt_fp4."""
    assert aiter_gemm_a4w4.__bases__ == (GEMM,)


def test_aiter_gemm_a4w4_flops_double_stream_mlp_up_proj():
    """Same Flux 12B shape as the hipBLASLt sibling test, dispatched via aiter:
    A=(16384, 1536) x B=(12288, 1536), K_packed=1536 -> K=3072."""
    M, K, N = 16384, 3072, 12288
    K_packed = K // 2
    model = aiter_gemm_a4w4(
        _aiter_gemm_a4w4_event(
            (M, K_packed),
            (N, K_packed),
            scaleA_shape=(M, K // 32),
            scaleB_shape=(N, K // 32),
        )
    )
    assert (model.M, model.N, model.K) == (M, N, K)
    assert model.flops() == 2 * M * N * K


def test_aiter_gemm_a4w4_bytes_match_hipblaslt_fp4():
    """The two FP4 GEMM paths touch the same bytes (different memory layout,
    same byte count). The shared `bytes()` implementation must produce the
    same result for matching M, N, K and scale shapes."""
    M, K, N = 16384, 3072, 12288
    K_packed = K // 2
    aiter_model = aiter_gemm_a4w4(
        _aiter_gemm_a4w4_event(
            (M, K_packed),
            (N, K_packed),
            scaleA_shape=(M, K // 32),
            scaleB_shape=(N, K // 32),
        )
    )
    hipblaslt_model = hipblaslt_gemm_fp4(
        _fp4_gemm_event(
            (M, K_packed),
            (N, K_packed),
            trans_b=True,
            scaleA_shape=(M, K // 32),
            scaleB_shape=(N, K // 32),
        )
    )
    assert aiter_model.bytes() == hipblaslt_model.bytes()


def test_aiter_gemm_a4w4_asm_event_uses_same_class():
    """The inner asm dispatch event (the leaf that launches the kernel) has
    11 slots — same first 4 tensors plus an output, then 6 scalars — and
    must yield the same M/N/K as the public entry."""
    M, K, N = 16384, 3072, 9216
    K_packed = K // 2
    model = aiter_gemm_a4w4(
        _aiter_gemm_a4w4_asm_event(
            (M, K_packed),
            (N, K_packed),
            scaleA_shape=(M, K // 32),
            scaleB_shape=(N, K // 32),
        )
    )
    assert (model.M, model.N, model.K) == (M, N, K)
    assert model.flops() == 2 * M * N * K


def test_aiter_gemm_a4w4_k_unpacked_factor_of_2():
    """FP4 packing means K_logical = 2 * K_traced for the aiter path too."""
    M, K_packed, N = 16384, 1536, 9216
    model = aiter_gemm_a4w4(_aiter_gemm_a4w4_event((M, K_packed), (N, K_packed)))
    assert model.K == K_packed * 2


def test_aiter_gemm_a4w4_nt_layout_assumption():
    """The aiter binding only supports NT. The model should derive M from
    A_shape[0] and N from B_shape[0] (transB=True), regardless of the
    bpreshuffle flag value — bpreshuffle is a layout knob, not a shape knob."""
    M, K, N = 32768, 3072, 12288
    K_packed = K // 2
    for bpreshuffle in (True, False):
        model = aiter_gemm_a4w4(
            _aiter_gemm_a4w4_event(
                (M, K_packed), (N, K_packed), bpreshuffle=bpreshuffle
            )
        )
        assert (model.M, model.N, model.K) == (M, N, K)


def _gemm_fp4_impl_event(M=3072, N=3072, K=3072, has_bias=False):
    """Build a primus_turbo::gemm_fp4_impl event matching the Flux trace layout.
    In Flux traces this op is the leaf (no inner hipblaslt/aiter children)."""
    K_packed = K // 2
    K_scale = K // MXFP4_GROUP_SIZE
    dims = [
        [M, K_packed],  # A (FP4 packed)
        [M, K_scale],  # A scales (E8M0)
        [],  # Scalar
        [N, K_packed],  # B (FP4 packed)
        [N, K_scale],  # B scales (E8M0)
        [],  # Scalar
        [],  # out_dtype
        [],  # bias_flag
        [],  # granularity
        [],  # backend
        [],  # preshuffled
        [N] if has_bias else [],  # bias
        [],  # out_scale_A (optional)
        [],  # out_scale_B (optional)
    ]
    types = [
        "c10::Float4_e2m1fn_x2",
        "c10::Float8_e8m0fnu",
        "Scalar",
        "c10::Float4_e2m1fn_x2",
        "c10::Float8_e8m0fnu",
        "Scalar",
        "Scalar",
        "Scalar",
        "Scalar",
        "Scalar",
        "Scalar",
        "c10::BFloat16" if has_bias else "",
        "",
        "",
    ]
    return {
        "name": "primus_turbo::gemm_fp4_impl",
        "args": {
            "Input Dims": dims,
            "Input type": types,
            "Input Strides": [[] for _ in dims],
            "Concrete Inputs": ["", "", "", "", "", "", "15"] + [""] * (len(dims) - 7),
        },
    }


def test_gemm_fp4_impl_registered():
    assert "primus_turbo::gemm_fp4_impl" in op_to_perf_model_class_map
    assert op_to_perf_model_class_map["primus_turbo::gemm_fp4_impl"] is gemm_fp4_impl


def test_gemm_fp4_impl_inherits_gemm():
    assert issubclass(gemm_fp4_impl, GEMM)


def test_gemm_fp4_impl_flops():
    M, N, K = 3072, 3072, 3072
    model = gemm_fp4_impl(_gemm_fp4_impl_event(M, N, K))
    assert model.flops() == 2 * M * N * K


def test_gemm_fp4_impl_param_details():
    M, N, K = 4096, 3072, 6144
    model = gemm_fp4_impl(_gemm_fp4_impl_event(M, N, K))
    assert (model.M, model.N, model.K) == (M, N, K)


def test_gemm_fp4_impl_k_unpacked_from_packed_dim():
    """K_logical = K_packed * 2 (Float4_e2m1fn_x2 packs two values per byte)."""
    M, N, K = 1024, 2048, 4096
    model = gemm_fp4_impl(_gemm_fp4_impl_event(M, N, K))
    assert model.K == K
    assert model.K == model.param_details["a_scale_shape"][1] * MXFP4_GROUP_SIZE


def test_gemm_fp4_impl_bytes_includes_scales():
    M, N, K = 3072, 3072, 3072
    model = gemm_fp4_impl(_gemm_fp4_impl_event(M, N, K))
    bpe_fp4_packed = 1  # c10::Float4_e2m1fn_x2 = 1 byte per packed pair
    bpe_scale = 1
    bpe_out = 2
    K_packed = K // 2
    K_scale = K // MXFP4_GROUP_SIZE
    expected = (
        M * K_packed * bpe_fp4_packed  # A
        + N * K_packed * bpe_fp4_packed  # B
        + M * N * bpe_out  # Y
        + M * K_scale * bpe_scale  # A scales
        + N * K_scale * bpe_scale  # B scales
    )
    assert model.bytes() == expected


def test_gemm_fp4_impl_bytes_with_bias():
    M, N, K = 3072, 3072, 3072
    model = gemm_fp4_impl(_gemm_fp4_impl_event(M, N, K, has_bias=True))
    bpe_fp4_packed = 1
    bpe_scale = 1
    bpe_out = 2
    K_packed = K // 2
    K_scale = K // MXFP4_GROUP_SIZE
    expected = (
        M * K_packed * bpe_fp4_packed
        + N * K_packed * bpe_fp4_packed
        + M * N * bpe_out
        + N * bpe_out  # bias (BF16)
        + M * K_scale * bpe_scale  # A scales
        + N * K_scale * bpe_scale  # B scales
    )
    assert model.bytes() == expected


def test_gemm_fp4_impl_compute_precision():
    from TraceLens.PerfModel.utils import torch_dtype_map

    model = gemm_fp4_impl(_gemm_fp4_impl_event())
    assert model.get_compute_precision() == torch_dtype_map("mxfp4")


# =========================================================================== #
# gpt-oss / primus-megatron MXFP4 composite autograd ops
#   FP4GemmMXFunction / FP4GemmMXFunctionBackward
#   FP4GroupedMLPMXFunc / FP4GroupedMLPMXFuncBackward
# Covers get_param_details, flops()/bytes(), the Sequence-number fwd/bwd cache
# (hit + miss -> None), the two-GEMM backward roll-up, get_compute_precision,
# and the NotImplementedError guards on the forward GEMM class.
# =========================================================================== #
def _fp4mx_gemm_event(M, N, K, seq=None, dtypes=(BF16, BF16)):
    args = {
        "Input Dims": [(M, K), (N, K), (), (), (), ()],
        "Input type": [dtypes[0], dtypes[1], "", "", "", ""],
    }
    if seq is not None:
        args["Sequence number"] = seq
    return {"name": "FP4GemmMXFunction", "args": args}


def _fp4mx_gemm_bwd_event(M, N, seq=None, dtype=BF16):
    args = {"Input Dims": [(M, N)], "Input type": [dtype]}
    if seq is not None:
        args["Sequence number"] = seq
    return {"name": "FP4GemmMXFunctionBackward", "args": args}


def _fp4mx_grouped_event(M, N, K, G, seq=None):
    inter = N // 2
    args = {
        "Input Dims": [(M, K), (M,), (G, N, K), (G, K, inter), (), (), (), (G,), (G + 1,)],
        "Input type": [BF16, "", BF16, BF16, "", "", "", "", ""],
    }
    if seq is not None:
        args["Sequence number"] = seq
    return {"name": "FP4GroupedMLPMXFunc", "args": args}


def _fp4mx_grouped_bwd_event(M, K, seq=None):
    args = {"Input Dims": [(M, K)], "Input type": [BF16]}
    if seq is not None:
        args["Sequence number"] = seq
    return {"name": "FP4GroupedMLPMXFuncBackward", "args": args}


# --------------------------------------------------------------------------- #
# FP4GemmMXFunction (forward)
# --------------------------------------------------------------------------- #
def test_fp4mx_gemm_param_details_and_flops():
    M, N, K = 32768, 2880, 4096
    m = FP4GemmMXFunction(_fp4mx_gemm_event(M, N, K))
    assert (m.M, m.N, m.K) == (M, N, K)
    assert m.param_details["bias"] is False
    assert m.param_details["dtype_A_B"] == (BF16, BF16, "c10::BFloat16")
    assert m.flops() == 2 * M * N * K


def test_fp4mx_gemm_bytes_is_bf16_roofline():
    M, N, K = 64, 128, 256
    m = FP4GemmMXFunction(_fp4mx_gemm_event(M, N, K))
    bpe = name2bpe(BF16)
    assert m.bytes() == (M * K + N * K + M * N) * bpe


def test_fp4mx_gemm_compute_precision_is_mxfp4():
    m = FP4GemmMXFunction(_fp4mx_gemm_event(16, 16, 32))
    assert m.get_compute_precision() == torch_dtype_map("mxfp4")


def test_fp4mx_gemm_backward_methods_raise_not_implemented():
    m = FP4GemmMXFunction(_fp4mx_gemm_event(16, 16, 32))
    with pytest.raises(NotImplementedError):
        m.flops_bwd()
    with pytest.raises(NotImplementedError):
        m.bytes_bwd()


def test_fp4mx_gemm_caches_by_sequence_number():
    FP4GemmMXFunction.fwd_pass_cache.clear()
    FP4GemmMXFunction(_fp4mx_gemm_event(100, 200, 300, seq=42))
    assert FP4GemmMXFunction.fwd_pass_cache[42]["K"] == 300


def test_fp4mx_gemm_without_sequence_number_not_cached():
    FP4GemmMXFunction.fwd_pass_cache.clear()
    FP4GemmMXFunction(_fp4mx_gemm_event(10, 20, 30))
    assert FP4GemmMXFunction.fwd_pass_cache == {}


# --------------------------------------------------------------------------- #
# FP4GemmMXFunctionBackward
# --------------------------------------------------------------------------- #
def test_fp4mx_gemm_backward_linked_recovers_dims():
    FP4GemmMXFunction.fwd_pass_cache.clear()
    M, N, K = 32768, 5120, 2880
    FP4GemmMXFunction(_fp4mx_gemm_event(M, N, K, seq=7))
    b = FP4GemmMXFunctionBackward(_fp4mx_gemm_bwd_event(M, N, seq=7))
    assert b.param_details["linked_fwd"] is True
    assert (b.M, b.N, b.K) == (M, N, K)
    assert b.flops() == 4 * M * N * K
    assert b.bytes() is not None and b.bytes() > 0
    assert b.get_compute_precision() == torch_dtype_map("mxfp4")


def test_fp4mx_gemm_backward_cache_miss_returns_none():
    FP4GemmMXFunction.fwd_pass_cache.clear()
    b = FP4GemmMXFunctionBackward(_fp4mx_gemm_bwd_event(32768, 5120, seq=999))
    assert b.param_details["linked_fwd"] is False
    assert b.K is None
    assert b.flops() is None
    assert b.bytes() is None


def test_fp4mx_gemm_backward_scalar_grad_out_gives_none_n():
    FP4GemmMXFunction.fwd_pass_cache.clear()
    b = FP4GemmMXFunctionBackward(
        {"args": {"Input Dims": [(4096,)], "Input type": [BF16]}}
    )
    assert b.param_details["N"] is None
    assert b.flops() is None
    assert b.bytes() is None


# --------------------------------------------------------------------------- #
# FP4GroupedMLPMXFunc (forward)
# --------------------------------------------------------------------------- #
def test_fp4mx_grouped_param_details_and_flops():
    M, N, K, G = 131072, 5760, 2880, 32
    m = FP4GroupedMLPMXFunc(_fp4mx_grouped_event(M, N, K, G))
    assert (m.M, m.N, m.K, m.G) == (M, N, K, G)
    assert m.flops() == m.flops_func(M, K, N) + m.flops_func(M, N // 2, K)


def test_fp4mx_grouped_bytes_nonzero_and_precision():
    M, N, K, G = 1024, 512, 256, 8
    m = FP4GroupedMLPMXFunc(_fp4mx_grouped_event(M, N, K, G))
    assert m.bytes() > 0
    assert m.get_compute_precision() == torch_dtype_map("mxfp4")


def test_fp4mx_grouped_caches_by_sequence_number():
    FP4GroupedMLPMXFunc.fwd_pass_cache.clear()
    FP4GroupedMLPMXFunc(_fp4mx_grouped_event(128, 64, 32, 4, seq=5))
    assert 5 in FP4GroupedMLPMXFunc.fwd_pass_cache


def test_fp4mx_grouped_forward_bwd_rollup_matches_backward_class():
    # flops_bwd/bytes_bwd on the FORWARD class drive the category-level backward
    # roll-up. They must model the two-GEMM backward (gate_up + down), i.e. match
    # FP4GroupedMLPMXFuncBackward, not the one-GEMM GroupedGemm base.
    FP4GroupedMLPMXFunc.fwd_pass_cache.clear()
    M, N, K, G = 131072, 5760, 2880, 32
    fwd = FP4GroupedMLPMXFunc(_fp4mx_grouped_event(M, N, K, G, seq=21))
    bwd = FP4GroupedMLPMXFuncBackward(_fp4mx_grouped_bwd_event(M, K, seq=21))

    # two-GEMM sums (exercises the flops_bwd/bytes_bwd overrides directly)
    assert fwd.flops_bwd() == fwd.flops_bwd_func(M, K, N) + fwd.flops_bwd_func(
        M, N // 2, K
    )
    assert fwd.bytes_bwd() == fwd.bytes_bwd_func(
        M, K, N, G, fwd.bpe_in, fwd.bpe_out
    ) + fwd.bytes_bwd_func(M, N // 2, K, G, fwd.bpe_in, fwd.bpe_out)

    # roll-up path (forward.flops_bwd/bytes_bwd) equals the dedicated backward op
    assert fwd.flops_bwd() == bwd.flops()
    assert fwd.bytes_bwd() == bwd.bytes()
    # and is strictly larger than a single-GEMM backward would give
    assert fwd.flops_bwd() > fwd.flops_bwd_func(M, K, N)


# --------------------------------------------------------------------------- #
# FP4GroupedMLPMXFuncBackward
# --------------------------------------------------------------------------- #
def test_fp4mx_grouped_backward_linked():
    FP4GroupedMLPMXFunc.fwd_pass_cache.clear()
    M, N, K, G = 131072, 5760, 2880, 32
    FP4GroupedMLPMXFunc(_fp4mx_grouped_event(M, N, K, G, seq=11))
    b = FP4GroupedMLPMXFuncBackward(_fp4mx_grouped_bwd_event(M, K, seq=11))
    assert b.param_details["linked_fwd"] is True
    assert b.flops() == b.flops_bwd_func(M, K, N) + b.flops_bwd_func(M, N // 2, K)
    assert b.bytes() is not None and b.bytes() > 0
    assert b.get_compute_precision() == torch_dtype_map("mxfp4")


def test_fp4mx_grouped_backward_cache_miss_returns_none():
    FP4GroupedMLPMXFunc.fwd_pass_cache.clear()
    b = FP4GroupedMLPMXFuncBackward(_fp4mx_grouped_bwd_event(131072, 2880, seq=404))
    assert b.param_details["linked_fwd"] is False
    assert b.flops() is None
    assert b.bytes() is None
