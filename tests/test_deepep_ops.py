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

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from TraceLens.PerfModel.perf_model import EPComm
from TraceLens.PerfModel.torch_op_mapping import (
    op_to_perf_model_class_map,
    categorize_torch_op,
    dict_cat2names,
)
from example_megatron_extension import (
    deepep_dispatch,
    deepep_combine,
    deepep_dispatch_backward,
    deepep_combine_backward,
    perf_model_extension,
    dict_cat2names_extension,
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
# Extension registration tests
# ---------------------------------------------------------------------------


def test_deepep_dispatch_in_extension():
    assert perf_model_extension["DeepEPDispatch"] is deepep_dispatch


def test_deepep_combine_in_extension():
    assert perf_model_extension["DeepEPCombine"] is deepep_combine


def test_deepep_dispatch_backward_in_extension():
    assert perf_model_extension["DeepEPDispatchBackward"] is deepep_dispatch_backward


def test_deepep_combine_backward_in_extension():
    assert perf_model_extension["DeepEPCombineBackward"] is deepep_combine_backward


def test_deepep_not_in_core_mapping():
    for name in [
        "DeepEPDispatch",
        "DeepEPCombine",
        "DeepEPDispatchBackward",
        "DeepEPCombineBackward",
    ]:
        assert (
            name not in op_to_perf_model_class_map
        ), f"{name} should be in extension, not core"


# ---------------------------------------------------------------------------
# Categorization tests
# ---------------------------------------------------------------------------


def test_deepep_in_extension_category():
    ep_ops = dict_cat2names_extension.get("EP_Communication", [])
    for name in [
        "DeepEPDispatch",
        "DeepEPCombine",
        "DeepEPDispatchBackward",
        "DeepEPCombineBackward",
    ]:
        assert name in ep_ops


def test_deepep_categorization_with_extension_loaded():
    """After merging extension into core dict_cat2names, categorize_torch_op works."""
    for name, ep_names in dict_cat2names_extension.items():
        dict_cat2names[name].extend(ep_names)
    try:
        for op_name in [
            "DeepEPDispatch",
            "DeepEPCombine",
            "DeepEPDispatchBackward",
            "DeepEPCombineBackward",
        ]:
            row = {"name": op_name, "kernel_details": []}
            assert categorize_torch_op(row) == "EP_Communication"
    finally:
        for name in dict_cat2names_extension:
            for ep_name in dict_cat2names_extension[name]:
                if ep_name in dict_cat2names[name]:
                    dict_cat2names[name].remove(ep_name)


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


def test_deepep_dispatch_flops_bwd_zero():
    model = deepep_dispatch(_dispatch_event())
    assert model.flops_bwd() == 0


def test_deepep_dispatch_bytes_bwd():
    num_tokens, hidden = 16384, 2048
    model = deepep_dispatch(_dispatch_event(num_tokens=num_tokens, hidden=hidden))
    assert model.bytes_bwd() == model.bytes()
