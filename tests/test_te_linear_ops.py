###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from TraceLens.PerfModel.perf_model import te_layer_norm_fn
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)

# ---------------------------------------------------------------------------
# Mapping and categorization
# ---------------------------------------------------------------------------


def test_te_layer_norm_fn_mapped():
    assert op_to_perf_model_class_map["LayerNormFn"] is te_layer_norm_fn


def test_layer_norm_fn_categorizes_as_norm_fwd():
    row = {"name": "LayerNormFn", "kernel_details": []}
    assert categorize_torch_op(row) == "NORM_fwd"


def test_layer_norm_fn_backward_categorizes_as_norm_bwd():
    row = {"name": "LayerNormFnBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "NORM_bwd"


# ---------------------------------------------------------------------------
# Helpers — synthetic events matching real TE trace format
# ---------------------------------------------------------------------------


def _layer_norm_fn_event(X_shape, gamma_shape, dtype="c10::BFloat16"):
    """LayerNormFn: Input[0]=X, Input[1]=gamma."""
    strides_X = []
    s = 1
    for d in reversed(X_shape):
        strides_X.insert(0, s)
        s *= d
    return {
        "name": "LayerNormFn",
        "args": {
            "Input Dims": [X_shape, gamma_shape, [], [], [], [], [], []],
            "Input type": [
                dtype,
                dtype,
                "",
                "",
                "Scalar",
                "Scalar",
                "Scalar",
                "Scalar",
            ],
            "Input Strides": [strides_X, [1], [], [], [], [], [], []],
            "Concrete Inputs": ["", "", "", "", "1e-05", "256", "False", "True"],
        },
    }


# ---------------------------------------------------------------------------
# te_layer_norm_fn — normalization model
# ---------------------------------------------------------------------------


def test_te_layer_norm_fn_instantiates():
    event = _layer_norm_fn_event(X_shape=[2048, 4, 2048], gamma_shape=[2048])
    model = te_layer_norm_fn(event)
    assert model.num_elems == 2048 * 4 * 2048
    assert model.num_channels == 2048


def test_te_layer_norm_fn_bytes():
    event = _layer_norm_fn_event(X_shape=[2048, 4, 2048], gamma_shape=[2048])
    model = te_layer_norm_fn(event)
    num_elems = 2048 * 4 * 2048
    num_channels = 2048
    bpe = 2  # bf16
    # is_affine=True, is_training=True, has_bias=False
    # num_weight_tensors = 2 (mean+var) + 1 (gamma) + 2 (training) = 5
    activation_bytes = num_elems * bpe + num_elems * bpe
    weight_bytes = 5 * num_channels * bpe
    assert model.bytes() == activation_bytes + weight_bytes
