###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for backward conv byte estimates."""

from TraceLens.PerfModel.perf_model import (
    CONV,
    aten_conv_bwd,
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
    assert expected_bwd == 126
    assert expected_fwd == 60


def test_conv_bias_relu_backward_bytes_uses_bwd_formula():
    ConvBiasReLU_.fwd_pass_cache.clear()
    ConvBiasReLU_(_conv_bias_relu_fwd_event())

    bwd = ConvBiasReLU_Backward(_conv_bias_relu_bwd_event())
    expected_bwd = _expected_bytes_bwd()

    assert bwd.bytes() == expected_bwd
    assert bwd.bytes_bwd() == expected_bwd
    assert expected_bwd == 126


def _aten_conv_bwd_event(bias_sizes, output_mask, dtype="c10::BFloat16"):
    # Layout of aten::convolution_backward as recorded by the PyTorch profiler
    # with record_shapes=True: bias_sizes and output_mask are not tensors, so
    # they only show up in Concrete Inputs.
    return {
        "args": {
            "Input Dims": [
                [2, 64, 16, 16],
                [2, 64, 16, 16],
                [64, 64, 3, 3],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ],
            "Input type": [dtype, dtype, dtype] + ["ScalarList"] * 8,
            "Concrete Inputs": [
                "",
                "",
                "",
                bias_sizes,
                "[1, 1]",
                "[1, 1]",
                "[1, 1]",
                "False",
                "[0, 0]",
                "1",
                output_mask,
            ],
        }
    }


def test_aten_conv_bwd_reads_bias_grad_from_output_mask():
    with_bias = aten_conv_bwd(_aten_conv_bwd_event("[64]", "[True, True, True]"))
    no_bias = aten_conv_bwd(_aten_conv_bwd_event("[0]", "[True, True, False]"))

    assert with_bias.param_details["bias"] is True
    assert no_bias.param_details["bias"] is False
    # grad_output is (2, 64, 16, 16): the bias gradient reads it once and
    # writes 64 values.
    assert with_bias.flops() - no_bias.flops() == 2 * 64 * 16 * 16
    assert with_bias.bytes() - no_bias.bytes() == (2 * 64 * 16 * 16 + 64) * 2


def test_conv_bwd_bias_grad_bytes_scale_with_dtype():
    x_shape = (2, 64, 16, 16)
    w_shape = (64, 64, 3, 3)
    out_shape = CONV.get_output_shape(
        x_shape, w_shape, (1, 1), (1, 1), (1, 1), False, (0, 0)
    )
    for bpe in (1, 2, 4):
        with_bias = CONV.bytes_bwd_func(x_shape, w_shape, out_shape, True, bpe)
        no_bias = CONV.bytes_bwd_func(x_shape, w_shape, out_shape, False, bpe)
        assert with_bias - no_bias == (2 * 64 * 16 * 16 + 64) * bpe
