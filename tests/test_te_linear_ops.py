###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from TraceLens.PerfModel.perf_model import (
    te_linear,
    te_layer_norm_linear,
    te_layer_norm_fn,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)

# ---------------------------------------------------------------------------
# Mapping and categorization
# ---------------------------------------------------------------------------


def test_te_linear_mapped():
    assert op_to_perf_model_class_map["_Linear"] is te_linear


def test_te_layer_norm_linear_mapped():
    assert op_to_perf_model_class_map["_LayerNormLinear"] is te_layer_norm_linear


def test_te_layer_norm_fn_mapped():
    assert op_to_perf_model_class_map["LayerNormFn"] is te_layer_norm_fn


def test_te_linear_categorizes_as_gemm():
    row = {"name": "_Linear", "kernel_details": []}
    assert categorize_torch_op(row) == "GEMM"


def test_te_linear_backward_categorizes_as_gemm():
    row = {"name": "_LinearBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "GEMM"


def test_te_layer_norm_linear_categorizes_as_gemm():
    row = {"name": "_LayerNormLinear", "kernel_details": []}
    assert categorize_torch_op(row) == "GEMM"


def test_te_layer_norm_linear_backward_categorizes_as_gemm():
    row = {"name": "_LayerNormLinearBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "GEMM"


def test_layer_norm_fn_categorizes_as_norm_fwd():
    row = {"name": "LayerNormFn", "kernel_details": []}
    assert categorize_torch_op(row) == "NORM_fwd"


def test_layer_norm_fn_backward_categorizes_as_norm_bwd():
    row = {"name": "LayerNormFnBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "NORM_bwd"


# ---------------------------------------------------------------------------
# Helpers — synthetic events matching real TE trace format
# ---------------------------------------------------------------------------


def _linear_event(W_shape, X_shape, dtype="c10::BFloat16"):
    """_Linear: Input[0]=W, Input[1]=X, then many scalar placeholders."""
    n_placeholders = 30
    dims = [W_shape, X_shape] + [[]] * n_placeholders
    types = [dtype, dtype] + [""] * n_placeholders
    strides_W = [W_shape[1], 1] if len(W_shape) == 2 else []
    strides_X = []
    s = 1
    for d in reversed(X_shape):
        strides_X.insert(0, s)
        s *= d
    strides = [strides_W, strides_X] + [[]] * n_placeholders
    return {
        "name": "_Linear",
        "args": {
            "Input Dims": dims,
            "Input type": types,
            "Input Strides": strides,
        },
    }


def _layer_norm_linear_event(
    X_shape, gamma_shape, W_shape, dtype="c10::BFloat16", has_beta=False
):
    """_LayerNormLinear: Input[0]=X, Input[1]=gamma, Input[2]=beta|empty, Input[3]=W."""
    n_placeholders = 35
    beta_shape = gamma_shape if has_beta else []
    dims = [X_shape, gamma_shape, beta_shape, W_shape] + [[]] * n_placeholders
    types = [dtype, dtype, dtype if has_beta else "", dtype] + [""] * n_placeholders
    strides_X = []
    s = 1
    for d in reversed(X_shape):
        strides_X.insert(0, s)
        s *= d
    strides_W = [W_shape[1], 1]
    strides = [strides_X, [1], [], strides_W] + [[]] * n_placeholders
    return {
        "name": "_LayerNormLinear",
        "args": {
            "Input Dims": dims,
            "Input type": types,
            "Input Strides": strides,
        },
    }


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
# te_linear — flops and bytes
# ---------------------------------------------------------------------------


def test_te_linear_flops():
    """_Linear with W=[4096, 4096], X=[4096, 4, 4096]: M=16384, N=4096, K=4096."""
    event = _linear_event(W_shape=[4096, 4096], X_shape=[4096, 4, 4096])
    model = te_linear(event)
    M, N, K = 16384, 4096, 4096
    assert model.M == M
    assert model.N == N
    assert model.K == K
    assert model.flops() == 2 * M * N * K


def test_te_linear_bytes():
    """bf16 inputs: bpe_in=2, bpe_out=2."""
    event = _linear_event(W_shape=[4096, 4096], X_shape=[4096, 4, 4096])
    model = te_linear(event)
    M, N, K = 16384, 4096, 4096
    expected = (M * K + K * N + M * N) * 2  # all bf16
    assert model.bytes() == expected


def test_te_linear_bytes_mixed_precision():
    """FP8 weights (1 byte) with BF16 activations (2 bytes)."""
    event = _linear_event(
        W_shape=[4096, 4096],
        X_shape=[4096, 4, 4096],
        dtype="c10::BFloat16",
    )
    event["args"]["Input type"][0] = "c10::Float8_e4m3fnuz"  # W dtype = FP8
    model = te_linear(event)
    M, N, K = 16384, 4096, 4096
    bpe_act, bpe_wt, bpe_out = 2, 1, 2  # bf16 activation, fp8 weight, bf16 output
    expected = M * K * bpe_act + K * N * bpe_wt + M * N * bpe_out
    assert model.bytes() == expected


def test_te_layer_norm_linear_bytes_mixed_precision():
    """FP8 weights (1 byte) with BF16 activations (2 bytes)."""
    event = _layer_norm_linear_event(
        X_shape=[4096, 4, 4096],
        gamma_shape=[4096],
        W_shape=[6144, 4096],
        dtype="c10::BFloat16",
    )
    event["args"]["Input type"][3] = "c10::Float8_e4m3fnuz"  # W dtype = FP8
    model = te_layer_norm_linear(event)
    M, N, K = 16384, 6144, 4096
    bpe_act, bpe_wt, bpe_out = 2, 1, 2
    expected = M * K * bpe_act + K * N * bpe_wt + M * N * bpe_out
    assert model.bytes() == expected


def test_te_linear_asymmetric():
    """_Linear with W=[1024, 2048], X=[2048, 4, 2048]: M=8192, N=1024, K=2048."""
    event = _linear_event(W_shape=[1024, 2048], X_shape=[2048, 4, 2048])
    model = te_linear(event)
    assert model.M == 8192
    assert model.N == 1024
    assert model.K == 2048
    assert model.flops() == 2 * 8192 * 1024 * 2048


# ---------------------------------------------------------------------------
# te_layer_norm_linear — flops and bytes
# ---------------------------------------------------------------------------


def test_te_layer_norm_linear_flops():
    """_LayerNormLinear with X=[4096,4,4096], gamma=[4096], W=[6144,4096]."""
    event = _layer_norm_linear_event(
        X_shape=[4096, 4, 4096],
        gamma_shape=[4096],
        W_shape=[6144, 4096],
    )
    model = te_layer_norm_linear(event)
    M, N, K = 16384, 6144, 4096
    assert model.M == M
    assert model.N == N
    assert model.K == K
    assert model.flops() == 2 * M * N * K


def test_te_layer_norm_linear_bytes():
    event = _layer_norm_linear_event(
        X_shape=[4096, 4, 4096],
        gamma_shape=[4096],
        W_shape=[6144, 4096],
    )
    model = te_layer_norm_linear(event)
    M, N, K = 16384, 6144, 4096
    expected = (M * K + K * N + M * N) * 2
    assert model.bytes() == expected


def test_te_layer_norm_linear_with_beta():
    """_LayerNormLinear with beta (has_bias=True) — GEMM dims unchanged."""
    event = _layer_norm_linear_event(
        X_shape=[2048, 4, 1024],
        gamma_shape=[1024],
        W_shape=[4384, 1024],
        has_beta=True,
    )
    model = te_layer_norm_linear(event)
    assert model.M == 8192
    assert model.N == 4384
    assert model.K == 1024
    assert model.flops() == 2 * 8192 * 4384 * 1024


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
