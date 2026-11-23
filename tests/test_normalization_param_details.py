###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for how the normalization perf models read profiler event arguments."""

import pytest

from TraceLens.PerfModel.perf_model import (
    BatchNorm,
    BatchNormBwd,
    GroupNorm,
    GroupNormBwd,
    InstanceNorm,
    LayerNorm,
    RMSNorm,
    RMSNormBwd,
)

X = [4, 8, 6, 6]
C = [8]
X_LN = [4, 8, 36]
C_LN = [36]


def _event(name, dims, concrete):
    """An event laid out as the profiler records it with record_shapes=True:
    a tensor that is not passed is [] in Input Dims, and bools in Concrete
    Inputs are the strings "True" / "False"."""
    return {
        "name": name,
        "args": {
            "Input Dims": dims,
            "Input type": ["float" if d else "" for d in dims],
            "Input Strides": [[1] * len(d) for d in dims],
            "Concrete Inputs": concrete,
        },
    }


def _batch_norm(affine, training):
    w = C if affine else []
    return _event(
        "aten::native_batch_norm",
        [X, w, w, C, C, [], [], []],
        ["", "", "", "", "", training, "0.1", "1e-05"],
    )


def _batch_norm_bwd(affine, training):
    w = C if affine else []
    return _event(
        "aten::native_batch_norm_backward",
        [X, X, w, C, C, C, C, [], [], []],
        ["", "", "", "", "", "", "", training, "1e-05", "[True, True, True]"],
    )


def _layer_norm(weight, bias):
    return _event(
        "aten::native_layer_norm",
        [X_LN, [], C_LN if weight else [], C_LN if bias else [], []],
        ["", "[36]", "", "", "1e-05"],
    )


def _group_norm(affine):
    w = C if affine else []
    return _event(
        "aten::group_norm",
        [X, [], w, w, [], []],
        ["", "2", "", "", "1e-05", "True"],
    )


def _group_norm_bwd(affine):
    mask = "[True, True, True]" if affine else "[True, False, False]"
    return _event(
        "aten::native_group_norm_backward",
        [X, X, [4, 2], [4, 2], C if affine else [], [], [], [], [], []],
        ["", "", "", "", "", "4", "8", "36", "2", mask],
    )


def _instance_norm(affine, use_input_stats="True"):
    w = C if affine else []
    return _event(
        "aten::instance_norm",
        [X, w, w, [], [], [], [], [], []],
        ["", "", "", "", "", use_input_stats, "0.1", "1e-05", "True"],
    )


def _rms_norm(affine):
    return _event(
        "aten::_fused_rms_norm",
        [X_LN, [], C_LN if affine else [], []],
        ["", "[36]", "", ""],
    )


def _rms_norm_bwd(affine):
    return _event(
        "aten::_fused_rms_norm_backward",
        [X_LN, X_LN, [], [4, 8, 1], C_LN if affine else [], []],
        ["", "", "[36]", "", "", "[True, True]" if affine else "[True, False]"],
    )


@pytest.mark.parametrize(
    "cls,make_event",
    [
        (BatchNorm, lambda affine: _batch_norm(affine, "True")),
        (BatchNormBwd, lambda affine: _batch_norm_bwd(affine, "True")),
        (LayerNorm, lambda affine: _layer_norm(affine, affine)),
        (GroupNorm, _group_norm),
        (GroupNormBwd, _group_norm_bwd),
        (InstanceNorm, _instance_norm),
        (RMSNorm, _rms_norm),
        (RMSNormBwd, _rms_norm_bwd),
    ],
)
def test_is_affine_follows_weight_argument(cls, make_event):
    affine = cls(make_event(True))
    plain = cls(make_event(False))
    assert affine.param_details["is_affine"] is True
    assert plain.param_details["is_affine"] is False
    assert plain.bytes() < affine.bytes()


def test_layer_norm_without_bias():
    both = LayerNorm(_layer_norm(weight=True, bias=True))
    weight_only = LayerNorm(_layer_norm(weight=True, bias=False))
    assert weight_only.param_details["is_affine"] is True
    assert weight_only.param_details["has_bias"] is False
    assert both.bytes() - weight_only.bytes() == C_LN[0] * 4


def test_rms_norm_without_weight_drops_weight_multiply():
    affine = RMSNorm(_rms_norm(True))
    plain = RMSNorm(_rms_norm(False))
    num_elems = 4 * 8 * 36
    assert affine.flops() - plain.flops() == num_elems
    assert affine.bytes() - plain.bytes() == C_LN[0] * 4


@pytest.mark.parametrize("affine", [True, False])
def test_batch_norm_eval_uses_running_stats(affine):
    train = BatchNorm(_batch_norm(affine, "True"))
    eval_ = BatchNorm(_batch_norm(affine, "False"))
    assert train.param_details["is_training"] is True
    assert eval_.param_details["is_training"] is False
    num_elems = 4 * 8 * 6 * 6
    # no mean/var pass over the input and no running stats written
    assert train.flops() - eval_.flops() == 3 * num_elems
    assert train.bytes() - eval_.bytes() == 2 * C[0] * 4


def test_batch_norm_backward_eval():
    train = BatchNormBwd(_batch_norm_bwd(True, "True"))
    eval_ = BatchNormBwd(_batch_norm_bwd(True, "False"))
    assert train.param_details["is_training"] is True
    assert eval_.param_details["is_training"] is False
    assert eval_.flops() == BatchNormBwd.flops_func_bwd(
        True, True, False, 4 * 8 * 6 * 6, 8, [True]
    )


def test_instance_norm_reads_use_input_stats():
    assert InstanceNorm(_instance_norm(True, "True")).param_details["is_training"]
    assert not InstanceNorm(_instance_norm(True, "False")).param_details["is_training"]


@pytest.mark.parametrize(
    "dims, passed",
    [(None, False), ([], False), ([0], False), ([8], True), ([4, 8], True)],
)
def test_optional_tensor_passed(dims, passed):
    from TraceLens.PerfModel.utils import optional_tensor_passed

    assert optional_tensor_passed(dims) is passed
