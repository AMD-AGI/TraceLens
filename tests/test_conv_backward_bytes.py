###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for backward conv byte estimates."""

from TraceLens.PerfModel.perf_model import (
    CONV,
    ConvBias_,
    ConvBias_Backward,
    ConvBiasReLU_,
    ConvBiasReLU_Backward,
)


def _conv_bias_fwd_event(seq_num=42):
    return {
        "args": {
            "Sequence number": seq_num,
            "Input Dims": [
                (1, 1, 4, 4),
                (1, 1, 3, 3),
                (1,),
            ],
            "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
            "Concrete Inputs": ["", "", "", "1", "0"],
        }
    }


def _conv_bias_bwd_event(seq_num=42):
    return {
        "args": {
            "Sequence number": seq_num,
            "Input Dims": [(1, 1, 2, 2)],
            "Input type": ["c10::BFloat16"],
        }
    }


def _conv_bias_relu_fwd_event(seq_num=43):
    return {
        "args": {
            "Sequence number": seq_num,
            "Input Dims": [
                (1, 1, 4, 4),
                (1, 1, 3, 3),
                (1,),
            ],
            "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
            "Concrete Inputs": ["", "", "", "1", "0"],
        }
    }


def _conv_bias_relu_bwd_event(seq_num=43):
    return _conv_bias_bwd_event(seq_num)


def _expected_bytes_bwd():
    x_shape = (1, 1, 4, 4)
    w_shape = (1, 1, 3, 3)
    out_shape = CONV.get_output_shape(
        x_shape, w_shape, (1, 1), (0, 0), (1, 1), False, (0, 0)
    )
    return CONV.bytes_bwd_func(x_shape, w_shape, out_shape, True, 2)


def _expected_bytes_fwd():
    x_shape = (1, 1, 4, 4)
    w_shape = (1, 1, 3, 3)
    out_shape = CONV.get_output_shape(
        x_shape, w_shape, (1, 1), (0, 0), (1, 1), False, (0, 0)
    )
    return CONV.bytes_func(x_shape, w_shape, out_shape, True, 2)


def test_conv_bias_backward_bytes_uses_bwd_formula():
    ConvBias_.fwd_pass_cache.clear()
    ConvBias_(_conv_bias_fwd_event())

    bwd = ConvBias_Backward(_conv_bias_bwd_event())
    expected_bwd = _expected_bytes_bwd()
    expected_fwd = _expected_bytes_fwd()

    assert bwd.bytes() == expected_bwd
    assert bwd.bytes_bwd() == expected_bwd
    assert expected_bwd != expected_fwd
    assert expected_bwd == 121
    assert expected_fwd == 60


def test_conv_bias_relu_backward_bytes_uses_bwd_formula():
    ConvBiasReLU_.fwd_pass_cache.clear()
    ConvBiasReLU_(_conv_bias_relu_fwd_event())

    bwd = ConvBiasReLU_Backward(_conv_bias_relu_bwd_event())
    expected_bwd = _expected_bytes_bwd()

    assert bwd.bytes() == expected_bwd
    assert bwd.bytes_bwd() == expected_bwd
    assert expected_bwd == 121
