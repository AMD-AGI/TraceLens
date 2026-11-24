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
    primus_turbo_grouped_gemm_fp4,
    primus_turbo_grouped_gemm_fp4_glu,
    primus_turbo_grouped_gemm_fp4_dglu,
    primus_turbo_grouped_gemm_variable_k_fp4,
    primus_turbo_grouped_gemm_variable_k_accum_fp4,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)
from TraceLens.PerfModel.utils import MXFP4_GROUP_SIZE, name2bpe, torch_dtype_map

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


# --------------------------------------------------------------------------- #
# FP4 MoE-MLP grouped GEMM leaf ops (primus_turbo::grouped_gemm_fp4_*_impl)
# Expected byte counts are hand-written from the trace slot shapes and the
# primus_turbo register_fake output shapes (without tile-alignment padding),
# independently of the model formulas. Group tables are not modelled.
# --------------------------------------------------------------------------- #
FP4 = "c10::Float4_e2m1fn_x2"
E8M0 = "c10::Float8_e8m0fnu"
LONG = "long int"

# gpt-oss trace shapes: M tokens, K hidden (padded 2880 -> 2944), G experts
M_T, KP_T, G_T = 131072, 1472, 32


def _ev(name, dims, types):
    return {"name": name, "args": {"Input Dims": dims, "Input type": types}}


def _fixed_k_event(name, N, extra_dims=(), extra_types=()):
    dims = [[M_T, KP_T], [G_T, N, KP_T], [M_T, 92], [G_T, N, 92], [G_T], [G_T + 1]]
    types = [FP4, FP4, E8M0, E8M0, LONG, LONG]
    return _ev(name, dims + list(extra_dims), types + list(extra_types))


def _fixed_k_reads(N, extra=0):
    return (
        M_T * KP_T  # a (fp4, 1 B per packed element)
        + G_T * N * KP_T  # b
        + M_T * 92  # a_scales (e8m0)
        + G_T * N * 92  # b_scales
        + extra
    )


def test_fp4_grouped_ops_map_to_fp4_subclasses():
    from TraceLens.PerfModel.extensions.pseudo_ops_perf_utils import (
        get_pseudo_op_mappings,
    )

    m = get_pseudo_op_mappings()
    assert m["primus_turbo::grouped_gemm_fp4_impl"] is primus_turbo_grouped_gemm_fp4
    assert (
        m["primus_turbo::grouped_gemm_fp4_glu_impl"]
        is primus_turbo_grouped_gemm_fp4_glu
    )
    assert (
        m["primus_turbo::grouped_gemm_fp4_dglu_impl"]
        is primus_turbo_grouped_gemm_fp4_dglu
    )
    assert (
        m["primus_turbo::grouped_gemm_fp4_variable_k_impl"]
        is primus_turbo_grouped_gemm_variable_k_fp4
    )
    assert (
        m["primus_turbo::grouped_gemm_fp4_variable_k_accum_impl"]
        is primus_turbo_grouped_gemm_variable_k_accum_fp4
    )
    # autograd-level wrappers are not modelled; their leaf cpu_ops are
    for dropped in (
        "FP4GemmMXFunction",
        "FP4GemmMXFunctionBackward",
        "FP4GroupedMLPMXFunc",
        "FP4GroupedMLPMXFuncBackward",
        "LinearWithGradAccumulationAndAsyncCommunication",
        "LinearWithGradAccumulationAndAsyncCommunicationBackward",
    ):
        assert dropped not in m


def test_fp4_grouped_gemm_impl_flops_and_bytes():
    N = 2880  # fc2 down
    m = primus_turbo_grouped_gemm_fp4(
        _fixed_k_event("primus_turbo::grouped_gemm_fp4_impl", N), None
    )
    assert (m.M, m.K, m.N, m.G) == (M_T, 2 * KP_T, N, G_T)  # K unpacked
    assert m.flops() == 2 * M_T * 2944 * N
    assert m.bytes() == _fixed_k_reads(N) + M_T * N * 2  # out[M, N] bf16
    assert m.get_compute_precision() == torch_dtype_map("c10::Float4_e2m1fn_x2")


def test_fp4_grouped_gemm_glu_writes_intermediate_and_quantized_activation():
    N = 5760  # gate||up, I = 2880
    probs = M_T * name2bpe("float")
    m = primus_turbo_grouped_gemm_fp4_glu(
        _fixed_k_event(
            "primus_turbo::grouped_gemm_fp4_glu_impl", N, [[M_T]], ["float"]
        ),
        None,
    )
    assert m.flops() == 2 * M_T * 2944 * N
    intermediate = M_T * N * 2  # [M, N] bf16
    # activation [M, I=2880], row + col MXFP4: 2 x (M*I/2 data + M*I/32 scales)
    act = 2 * (131072 * 2880 // 2 + 131072 * 2880 // 32)
    assert act == 401080320
    assert m.bytes() == _fixed_k_reads(N, probs) + intermediate + act


def test_fp4_grouped_gemm_dglu_writes_grad_probs_and_quantized_grad_l1():
    I = 2880  # dglu b is [G, I, K/2]
    extra = M_T * name2bpe("float") + M_T * 2 * I * 2  # probs + intermediate bf16
    m = primus_turbo_grouped_gemm_fp4_dglu(
        _fixed_k_event(
            "primus_turbo::grouped_gemm_fp4_dglu_impl",
            I,
            [[M_T], [M_T, 2 * I]],
            ["float", "c10::BFloat16"],
        ),
        None,
    )
    assert m.flops() == 2 * M_T * 2944 * I
    grad_probs = M_T * 4  # [M] fp32
    # grad_l1 [M, 2I=5760], row + col MXFP4: 2 x (M*2I/2 data + M*2I/32 scales)
    act = 2 * (131072 * 5760 // 2 + 131072 * 5760 // 32)
    assert act == 802160640
    assert m.bytes() == _fixed_k_reads(I, extra) + grad_probs + act


def _wgrad_dims(out, inn, with_out_buffer):
    dims = [[out, 69632], [inn, 69632], [out, 4352], [inn, 4352], [G_T], [G_T + 1]]
    types = [FP4, FP4, E8M0, E8M0, LONG, LONG]
    if with_out_buffer:
        dims += [[], [], [], [], [], [], [], [G_T, out, inn]]
        types += ["Scalar"] * 7 + ["c10::BFloat16"]
    return dims, types


def _wgrad_reads(out, inn):
    return (out + inn) * 69632 + (out + inn) * 4352  # fp4 operands + e8m0 scales


def test_fp4_variable_k_accum_reads_and_writes_grad_w():
    out, inn = 5760, 2880  # fc1 wgrad
    dims, types = _wgrad_dims(out, inn, with_out_buffer=True)
    m = primus_turbo_grouped_gemm_variable_k_accum_fp4(
        _ev("primus_turbo::grouped_gemm_fp4_variable_k_accum_impl", dims, types), None
    )
    # M=out, N=in, K=tokens (139264 = 131072 + 32*256 padding)
    assert (m.M, m.N, m.K, m.G) == (out, inn, 139264, G_T)
    assert m.flops() == 2 * out * 139264 * inn
    grad_w = G_T * out * inn * 2
    # accum: out += ... (beta=1) -> grad_W is read and written
    assert m.bytes() == _wgrad_reads(out, inn) + grad_w + grad_w


def test_fp4_variable_k_non_accum_writes_grad_w():
    out, inn = 2880, 2880  # fc2 wgrad
    dims, types = _wgrad_dims(out, inn, with_out_buffer=False)
    m = primus_turbo_grouped_gemm_variable_k_fp4(
        _ev("primus_turbo::grouped_gemm_fp4_variable_k_impl", dims, types), None
    )
    assert m.flops() == 2 * out * 139264 * inn
    assert m.bytes() == _wgrad_reads(out, inn) + G_T * out * inn * 2  # write only


@pytest.mark.parametrize(
    "cls",
    [
        primus_turbo_grouped_gemm_fp4,
        primus_turbo_grouped_gemm_fp4_glu,
        primus_turbo_grouped_gemm_fp4_dglu,
    ],
)
def test_fp4_fixed_k_backward_methods_raise(cls):
    m = cls(_fixed_k_event("primus_turbo::grouped_gemm_fp4_impl", 2880), None)
    with pytest.raises(NotImplementedError):
        m.flops_bwd()
    with pytest.raises(NotImplementedError):
        m.bytes_bwd()


@pytest.mark.parametrize(
    "cls, accum",
    [
        (primus_turbo_grouped_gemm_variable_k_fp4, False),
        (primus_turbo_grouped_gemm_variable_k_accum_fp4, True),
    ],
)
def test_fp4_variable_k_backward_methods_raise(cls, accum):
    dims, types = _wgrad_dims(2880, 2880, with_out_buffer=accum)
    m = cls(_ev("primus_turbo::grouped_gemm_fp4_variable_k_impl", dims, types), None)
    with pytest.raises(NotImplementedError):
        m.flops_bwd()
    with pytest.raises(NotImplementedError):
        m.bytes_bwd()
