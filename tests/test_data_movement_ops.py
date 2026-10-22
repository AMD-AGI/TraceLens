###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for memory/data-movement perf models: aten::cat, aten::index_select,
aten::gather, aten::_index_put_impl_, aten::masked_scatter_, aten::arange,
and aten::where. Unlike the generic unary/binary/reduce ops, these have
variable arity (cat) or index-tensor-driven output shapes (index_select,
gather, index_put, masked_scatter), so each gets a dedicated class.
"""

from math import prod

from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)

TIER2_OPS = {
    "aten::cat": perf_model.aten_cat,
    "aten::index_select": perf_model.aten_index_select,
    "aten::gather": perf_model.aten_gather,
    "aten::_index_put_impl_": perf_model.aten_index_put_impl,
    "aten::masked_scatter_": perf_model.aten_masked_scatter,
    "aten::arange": perf_model.aten_arange,
    "aten::where": perf_model.aten_where,
}


def test_data_movement_ops_mapped():
    for op, cls in TIER2_OPS.items():
        assert op_to_perf_model_class_map[op] is cls, op


def test_data_movement_ops_categorize():
    for op in [
        "aten::cat",
        "aten::index_select",
        "aten::gather",
        "aten::_index_put_impl_",
        "aten::masked_scatter_",
        "aten::arange",
    ]:
        row = {"name": op, "kernel_details": []}
        assert categorize_torch_op(row) == "data_movement", op
    # where -> elementwise (ternary select)
    row = {"name": "aten::where", "kernel_details": []}
    assert categorize_torch_op(row) == "elementwise"


# ---------------------------------------------------------------------------
# get_param_details / flops / bytes -- real trace payloads captured from
# GR00T / CogACT / DDVLA (robotics VLA) record_shapes=True traces.
# ---------------------------------------------------------------------------


def test_aten_cat_variable_arity_tensorlist():
    event = {
        "name": "aten::cat",
        "args": {
            "Input Dims": [
                [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 2]],
                [],
            ],
            "Input type": ["TensorList", "Scalar"],
            "Concrete Inputs": ["", "-1"],
        },
    }
    model = perf_model.aten_cat(event)
    # 6 tensors of shape [1,1] (1 elem each) + 1 tensor of shape [1,2] (2 elems)
    assert model.num_input_elems == 6 * 1 + 2
    assert model.flops() == 0
    # Per-tensor dtype is unrecoverable from a TensorList arg.
    assert model.bytes() is None


def test_aten_index_select_dim_replaced():
    event = {
        "name": "aten::index_select",
        "args": {
            "Input Dims": [[151680, 2048], [], [1568]],
            "Input type": ["c10::BFloat16", "Scalar", "long int"],
            "Concrete Inputs": ["", "0", ""],
        },
    }
    model = perf_model.aten_index_select(event)
    assert model.num_output_elems == 1568 * 2048
    assert model.num_index_elems == 1568
    assert model.flops() == 0
    expected_bytes = 2 * (1568 * 2048) * 2 + 1568 * 8  # bf16=2B, long=8B
    assert model.bytes() == expected_bytes


def test_aten_gather_output_matches_index_shape():
    event = {
        "name": "aten::gather",
        "args": {
            "Input Dims": [[8, 633, 3584], [], [8, 1, 3584], []],
            "Input type": ["c10::BFloat16", "Scalar", "long int", "Scalar"],
            "Concrete Inputs": ["", "1", "", "False"],
        },
    }
    model = perf_model.aten_gather(event)
    assert model.num_output_elems == prod([8, 1, 3584])
    assert model.flops() == 0
    assert model.bytes() is not None


def test_aten_index_put_impl_uses_values_shape():
    event = {
        "name": "aten::_index_put_impl_",
        "args": {
            "Input Dims": [[1568, 2048], [], [1296, 2048], [], []],
            "Input type": ["c10::BFloat16", "", "c10::BFloat16", "Scalar", "Scalar"],
            "Concrete Inputs": ["", "", "", "False", "False"],
        },
    }
    model = perf_model.aten_index_put_impl(event)
    assert model.num_touched_elems == 1296 * 2048
    assert model.flops() == 0
    assert model.bytes() == 2 * (1296 * 2048) * 2  # bf16 read + write


def test_aten_masked_scatter_upper_bound_approximation():
    event = {
        "name": "aten::masked_scatter_",
        "args": {
            "Input Dims": [[64, 128], [64, 128], [1000]],
            "Input type": ["float", "bool", "float"],
        },
    }
    model = perf_model.aten_masked_scatter(event)
    # Upper-bounded by numel(source), not mask.sum() (data-dependent/unknown).
    assert model.num_touched_elems == 1000
    assert model.flops() == 0


def test_aten_arange_output_size_from_scalars():
    event = {
        "name": "aten::arange",
        "args": {
            "Input Dims": [[], [], [], [], [], []],
            "Input type": ["Scalar", "Scalar", "", "", "", "Scalar"],
            "Concrete Inputs": ["0", "196", "", "", "", "False"],
        },
    }
    model = perf_model.aten_arange(event)
    assert model.num_output_elems == 196
    assert model.flops() == 0
    assert model.bytes() is None


def test_aten_where_ternary_with_scalar_and_tensor():
    event = {
        "name": "aten::where",
        "args": {
            "Input Dims": [[8, 32, 1, 196], [], []],
            "Input type": ["bool", "Scalar", "c10::BFloat16"],
            "Concrete Inputs": ["", "0.", ""],
        },
    }
    model = perf_model.aten_where(event)
    assert model.num_output_elems == prod([8, 32, 1, 196])
    assert model.flops() == model.num_output_elems
    # bool condition is 1 byte/elem (requires the name2bpe "bool" fix).
    assert model.bytes() is not None
