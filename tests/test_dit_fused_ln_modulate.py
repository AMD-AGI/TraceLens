###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for primus::fused_ln_modulate{,_backward} (#627):
  Fused LayerNorm + scale/shift modulation for MM-DiT / DiT-family blocks.
"""

from TraceLens.PerfModel.perf_model import (
    FusedLnModulate,
    FusedLnModulateBackward,
    Normalization,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fused_ln_fwd_event(T=256, B=40, H=3072, dtype="c10::BFloat16"):
    """
    Build a fused_ln_modulate forward event.
    Inputs: x:(T,B,H), scale:(B,H), shift:(B,H), eps (scalar).
    """
    return {
        "name": "primus::fused_ln_modulate",
        "args": {
            "Input Dims": [(T, B, H), (B, H), (B, H), ()],
            "Input type": [dtype, dtype, dtype, "Scalar"],
        },
    }


def _fused_ln_bwd_event(T=256, B=40, H=3072, dtype="c10::BFloat16"):
    """
    Build a fused_ln_modulate_backward event.
    Inputs: grad_out:(T,B,H), x_norm:(T,B,H), mean:(T*B,), rstd:(T*B,), mod_grad:(B,H).
    Note: mean/rstd are per-row (one per (t,b) sample), stored as fp32.
    """
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
# fused_ln_modulate forward
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
    """
    fwd bytes = read_x + write_y + read_scale_shift + write_mean_rstd
              = 2*T*B*H*bpe + 2*B*H*bpe + 2*T*B*4
    Mean/rstd are per-row (one per (t,b) sample), stored as fp32.
    """
    T, B, H = 256, 40, 3072
    bpe = 2  # BF16
    model = FusedLnModulate(_fused_ln_fwd_event(T=T, B=B, H=H))
    read_x = T * B * H * bpe
    write_y = T * B * H * bpe
    read_scale_shift = 2 * B * H * bpe
    write_mean_rstd = 2 * T * B * 4
    expected = read_x + write_y + read_scale_shift + write_mean_rstd
    assert model.bytes() == expected


def test_fused_ln_modulate_double_stream():
    """T=512 double-stream MM-DiT block."""
    T, B, H = 512, 40, 3072
    bpe = 2
    model = FusedLnModulate(_fused_ln_fwd_event(T=T, B=B, H=H))
    assert model.flops() == T * B * H * 12
    expected_bytes = 2 * T * B * H * bpe + 2 * B * H * bpe + 2 * T * B * 4
    assert model.bytes() == expected_bytes


# ===========================================================================
# fused_ln_modulate_backward
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
    """
    bwd bytes = read_grad_x_norm + read_mean_rstd + write_x_grad
              + write_scale_shift_grad + read_mod_grad
    """
    T, B, H = 256, 40, 3072
    bpe = 2
    model = FusedLnModulateBackward(_fused_ln_bwd_event(T=T, B=B, H=H))
    read_grad_x_norm = 2 * T * B * H * bpe
    read_mean_rstd = 2 * B * T * 4  # fp32, per row
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
