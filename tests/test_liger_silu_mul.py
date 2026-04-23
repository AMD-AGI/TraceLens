###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from TraceLens.PerfModel.perf_model import liger_silu_mul_function, BinaryElementwise
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Shapes from a real OpenFold3 trace event:
# a, b: [Batch=1, N_tokens=2911, hidden=256], dtype float32
_A_SHAPE = [1, 2911, 256]
_B_SHAPE = [1, 2911, 256]
_A_STRIDES = [745216, 256, 1]
_B_STRIDES = [745216, 256, 1]
_DTYPE = "float"


def _event(dtype=None):
    """Build an event using the real trace shapes and strides (_A_SHAPE / _A_STRIDES).
    Use _simple_event() when you need arbitrary shapes with dummy unit strides."""
    dt = dtype or _DTYPE
    return {
        "name": "LigerSiLUMulFunction",
        "args": {
            "Input Dims": [_A_SHAPE, _B_SHAPE],
            "Input type": [dt, dt],
            "Input Strides": [_A_STRIDES, _B_STRIDES],
            "Concrete Inputs": ["", ""],
        },
    }


def _simple_event(a_shape, b_shape, dtype=_DTYPE):
    return {
        "name": "LigerSiLUMulFunction",
        "args": {
            "Input Dims": [a_shape, b_shape],
            "Input type": [dtype, dtype],
            "Input Strides": [
                [1] * len(a_shape),
                [1] * len(b_shape),
            ],
            "Concrete Inputs": ["", ""],
        },
    }


# ---------------------------------------------------------------------------
# Mapping and categorization
# ---------------------------------------------------------------------------


def test_liger_silu_mul_is_mapped():
    assert op_to_perf_model_class_map["LigerSiLUMulFunction"] is liger_silu_mul_function


def test_liger_silu_mul_categorizes_as_elementwise():
    row = {"name": "LigerSiLUMulFunction", "kernel_details": []}
    assert categorize_torch_op(row) == "elementwise"


def test_liger_silu_mul_inherits_binary_elementwise():
    assert issubclass(liger_silu_mul_function, BinaryElementwise)


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------


def test_liger_silu_mul_param_details():
    evt = _event()
    m = liger_silu_mul_function(evt)
    assert m.nelems_in1 == 1 * 2911 * 256
    assert m.nelems_in2 == 1 * 2911 * 256
    assert m.nelems_out == 1 * 2911 * 256


def test_liger_silu_mul_dtype_float32():
    """float dtype should resolve to 4 bytes per element."""
    evt = _event()
    m = liger_silu_mul_function(evt)
    assert m.bpe_in1 == 4
    assert m.bpe_in2 == 4
    assert m.bpe_out == 4


def test_liger_silu_mul_dtype_bfloat16():
    evt = _simple_event(_A_SHAPE, _B_SHAPE, dtype="c10::BFloat16")
    m = liger_silu_mul_function(evt)
    assert m.bpe_in1 == 2
    assert m.bpe_in2 == 2
    assert m.bpe_out == 2


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def test_liger_silu_mul_flops():
    # BinaryElementwise convention: 1 FLOP per output element
    expected = 1 * 2911 * 256
    evt = _event()
    m = liger_silu_mul_function(evt)
    assert m.flops() == expected


def test_liger_silu_mul_flops_scales_with_tokens():
    """Doubling the token count should double FLOPs."""
    evt1 = _simple_event([1, 1000, 256], [1, 1000, 256])
    evt2 = _simple_event([1, 2000, 256], [1, 2000, 256])
    assert (
        liger_silu_mul_function(evt2).flops()
        == 2 * liger_silu_mul_function(evt1).flops()
    )


# ---------------------------------------------------------------------------
# Bytes
# ---------------------------------------------------------------------------


def test_liger_silu_mul_bytes_float32():
    # 3 tensors (a in, b in, c out), each 1*2911*256 elements * 4 bytes
    expected = 3 * 1 * 2911 * 256 * 4
    evt = _event()
    m = liger_silu_mul_function(evt)
    assert m.bytes() == expected


def test_liger_silu_mul_bytes_bfloat16():
    # same layout, 2 bpe → half the bytes
    evt = _simple_event(_A_SHAPE, _B_SHAPE, dtype="c10::BFloat16")
    m = liger_silu_mul_function(evt)
    assert m.bytes() == 3 * 1 * 2911 * 256 * 2


def test_liger_silu_mul_bytes_scales_with_hidden():
    """Doubling hidden dim should double bytes."""
    evt1 = _simple_event([1, 512, 128], [1, 512, 128])
    evt2 = _simple_event([1, 512, 256], [1, 512, 256])
    assert (
        liger_silu_mul_function(evt2).bytes()
        == 2 * liger_silu_mul_function(evt1).bytes()
    )
