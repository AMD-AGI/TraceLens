###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for DeepEP Expert-Parallel communication performance models.

Trace shape reference (DeepSeek V2 Lite, EP=8):
  DeepEPDispatch        Input Dims[0] = (16384, 2048)   – local tokens sent out
  DeepEPCombine         Input Dims[0] = (131072, 2048)  – dispatched tokens combined
  DeepEPDispatchBackward Input Dims[0] = (131072, 2048) – gradient of dispatched tokens
  DeepEPCombineBackward  Input Dims[0] = (16384, 2048)  – gradient of local tokens
"""

from TraceLens.PerfModel.perf_model import (
    EPComm,
    deepep_dispatch,
    deepep_combine,
    deepep_dispatch_backward,
    deepep_combine_backward,
)
from TraceLens.PerfModel.torch_op_mapping import (
    op_to_perf_model_class_map,
    categorize_torch_op,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dispatch_event(num_tokens=16384, hidden=2048, dtype="c10::BFloat16"):
    """Synthetic DeepEPDispatch event matching real trace layout."""
    return {
        "name": "DeepEPDispatch",
        "args": {
            "Input Dims": [
                [num_tokens, hidden],
                [num_tokens, 6],
                [num_tokens, 6],
                [],
                [],
                [],
                [],
            ],
            "Input type": [
                dtype,
                "long int",
                "float",
                "Scalar",
                "Scalar",
                "Scalar",
                "Scalar",
            ],
            "Input Strides": [[hidden, 1], [6, 1], [6, 1], [], [], [], []],
            "Concrete Inputs": ["", "", "", "64", "True", "True", "131072"],
        },
    }


def _combine_event(num_tokens=131072, hidden=2048, dtype="c10::BFloat16"):
    """Synthetic DeepEPCombine event matching real trace layout."""
    return {
        "name": "DeepEPCombine",
        "args": {
            "Input Dims": [[num_tokens, hidden], [], []],
            "Input type": [dtype, "Scalar", "Scalar"],
            "Input Strides": [[hidden, 1], [], []],
            "Concrete Inputs": ["", "True", "True"],
        },
    }


def _dispatch_bwd_event(num_tokens=131072, hidden=2048, dtype="c10::BFloat16"):
    """Synthetic DeepEPDispatchBackward event matching real trace layout."""
    return {
        "name": "DeepEPDispatchBackward",
        "args": {
            "Input Dims": [[num_tokens, hidden], [], [num_tokens, 6], [], []],
            "Input type": [dtype, "", "float", "", ""],
            "Input Strides": [[hidden, 1], [], [6, 1], [], []],
            "Concrete Inputs": ["", "", "", "", ""],
        },
    }


def _combine_bwd_event(num_tokens=16384, hidden=2048, dtype="c10::BFloat16"):
    """Synthetic DeepEPCombineBackward event matching real trace layout."""
    return {
        "name": "DeepEPCombineBackward",
        "args": {
            "Input Dims": [[num_tokens, hidden]],
            "Input type": [dtype],
            "Input Strides": [[hidden, 1]],
            "Concrete Inputs": [""],
        },
    }


# ---------------------------------------------------------------------------
# Mapping tests
# ---------------------------------------------------------------------------


def test_deepep_dispatch_mapped():
    assert op_to_perf_model_class_map["DeepEPDispatch"] is deepep_dispatch


def test_deepep_combine_mapped():
    assert op_to_perf_model_class_map["DeepEPCombine"] is deepep_combine


def test_deepep_dispatch_backward_mapped():
    assert (
        op_to_perf_model_class_map["DeepEPDispatchBackward"] is deepep_dispatch_backward
    )


def test_deepep_combine_backward_mapped():
    assert (
        op_to_perf_model_class_map["DeepEPCombineBackward"] is deepep_combine_backward
    )


# ---------------------------------------------------------------------------
# Categorization tests
# ---------------------------------------------------------------------------


def test_deepep_dispatch_category():
    row = {"name": "DeepEPDispatch", "kernel_details": []}
    assert categorize_torch_op(row) == "EP_Communication"


def test_deepep_combine_category():
    row = {"name": "DeepEPCombine", "kernel_details": []}
    assert categorize_torch_op(row) == "EP_Communication"


def test_deepep_dispatch_backward_category():
    row = {"name": "DeepEPDispatchBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "EP_Communication"


def test_deepep_combine_backward_category():
    row = {"name": "DeepEPCombineBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "EP_Communication"


# ---------------------------------------------------------------------------
# Performance model tests
# ---------------------------------------------------------------------------


def test_deepep_dispatch_flops_zero():
    model = deepep_dispatch(_dispatch_event())
    assert model.flops() == 0


def test_deepep_dispatch_bytes():
    num_tokens, hidden = 16384, 2048
    model = deepep_dispatch(_dispatch_event(num_tokens=num_tokens, hidden=hidden))
    expected = num_tokens * hidden * 2  # BF16 = 2 bytes
    assert model.bytes() == expected


def test_deepep_combine_bytes():
    num_tokens, hidden = 131072, 2048
    model = deepep_combine(_combine_event(num_tokens=num_tokens, hidden=hidden))
    expected = num_tokens * hidden * 2
    assert model.bytes() == expected


def test_deepep_dispatch_backward_bytes():
    num_tokens, hidden = 131072, 2048
    model = deepep_dispatch_backward(
        _dispatch_bwd_event(num_tokens=num_tokens, hidden=hidden)
    )
    expected = num_tokens * hidden * 2
    assert model.bytes() == expected


def test_deepep_combine_backward_bytes():
    num_tokens, hidden = 16384, 2048
    model = deepep_combine_backward(
        _combine_bwd_event(num_tokens=num_tokens, hidden=hidden)
    )
    expected = num_tokens * hidden * 2
    assert model.bytes() == expected


def test_deepep_combine_bytes_larger_than_dispatch():
    """Combine operates on all dispatched tokens; dispatch only on local tokens."""
    dispatch = deepep_dispatch(_dispatch_event(num_tokens=16384, hidden=2048))
    combine = deepep_combine(_combine_event(num_tokens=131072, hidden=2048))
    assert combine.bytes() > dispatch.bytes()


def test_deepep_dispatch_fp32_dtype():
    event = _dispatch_event(dtype="float")
    model = deepep_dispatch(event)
    expected = 16384 * 2048 * 4  # float32 = 4 bytes
    assert model.bytes() == expected


def test_deepep_unknown_dtype_returns_none():
    """bytes() must return None rather than a silent wrong estimate for unknown dtypes."""
    event = _dispatch_event(dtype="unknown_dtype")
    model = deepep_dispatch(event)
    assert model.bytes() is None


def test_deepep_all_inherit_epcomm():
    for cls in (
        deepep_dispatch,
        deepep_combine,
        deepep_dispatch_backward,
        deepep_combine_backward,
    ):
        assert issubclass(cls, EPComm)
