###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Embedding lookup is a gather: zero FLOPs, bytes for the rows plus indices."""

import pytest

from TraceLens.PerfModel.op_projections import op_work
from TraceLens.PerfModel.perf_model import (
    EmbeddingLookup,
    aten_embedding,
    aten_embedding_bag,
    aten_embedding_bag_dense_backward,
    aten_embedding_dense_backward,
    aten_gather,
    aten_index_select,
    aten_index_select_backward,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
    resolve_perf_model_class,
)


def _event(dims, types, concrete=None, name="aten::embedding"):
    args = {"Input Dims": dims, "Input type": types}
    if concrete is not None:
        args["Concrete Inputs"] = concrete
    return {"name": name, "args": args}


def test_bytes_func_counts_row_read_write_and_indices():
    # 4 tokens, hidden 8, bf16 rows, int64 indices.
    assert EmbeddingLookup.bytes_func(4, 8, 2, 8) == 2 * 4 * 8 * 2 + 4 * 8
    assert EmbeddingLookup.flops_func(4, 8) == 0
    assert EmbeddingLookup.bytes_func(4, 8, None, 8) is None
    assert EmbeddingLookup.bytes_func(4, 8, 2, None) is None


def test_aten_embedding_gathers_weight_rows():
    event = _event(
        [[32000, 4096], [2, 128], [], [], []],
        ["c10::BFloat16", "long int", "Scalar", "Scalar", "Scalar"],
    )
    model = aten_embedding(event)
    assert model.flops() == 0
    assert model.param_details["tokens"] == 256
    assert model.param_details["hidden"] == 4096
    assert model.bytes() == 2 * 256 * 4096 * 2 + 256 * 8
    assert model.get_compute_precision() == "bf16"
    assert model.get_maf_type() == "vector"

    work = op_work(model)
    assert work.category == "Embedding"
    assert work.gflops == 0
    assert work.bytes_moved == model.bytes()
    # Dense backward of the same lookup re-reads the updated row.
    assert model.bytes_bwd() == 3 * 256 * 4096 * 2 + 256 * 8


def test_embedding_dense_backward_scatters_grad_rows():
    event = _event(
        [[2, 128, 4096], [2, 128]],
        ["c10::BFloat16", "long int"],
        name="aten::embedding_dense_backward",
    )
    model = aten_embedding_dense_backward(event)
    assert model.flops() == 0
    assert model.flops_bwd() == 0
    assert model.bytes() == 3 * 256 * 4096 * 2 + 256 * 8


def test_embedding_bag_counts_offsets_as_index_traffic():
    event = _event(
        [[100, 16], [10], [3]],
        ["c10::BFloat16", "long int", "long int"],
        name="aten::_embedding_bag",
    )
    model = aten_embedding_bag(event)
    assert model.param_details["tokens"] == 10
    assert model.param_details["hidden"] == 16
    assert model.param_details["index_elems"] == 13
    assert model.bytes() == 2 * 10 * 16 * 2 + 13 * 8


def test_index_select_replaces_one_axis():
    event = _event(
        [[2, 100, 8], [], [4]],
        ["c10::Half", "Scalar", "long int"],
        concrete=["", "1", ""],
        name="aten::index_select",
    )
    model = aten_index_select(event)
    # dim 1: output [2, 4, 8], 32 elements, one index per selected slice.
    assert model.param_details["tokens"] == 4
    assert model.param_details["hidden"] == 16
    assert model.bytes() == 2 * 64 * 2 + 4 * 8

    negative = _event(
        [[2, 100, 8], [], [4]],
        ["c10::Half", "Scalar", "long int"],
        concrete=["", "-1", ""],
        name="aten::index_select",
    )
    along_last = aten_index_select(negative)
    # dim -1: output [2, 100, 4].
    assert along_last.bytes() == 2 * (2 * 100 * 4) * 2 + 4 * 8


def test_index_select_backward_scatters_the_grad():
    event = _event(
        [[2, 4, 8], [4]],
        ["c10::Half", "long int"],
        name="aten::index_select_backward",
    )
    model = aten_index_select_backward(event)
    assert model.bytes() == 3 * (2 * 4 * 8) * 2 + 4 * 8


def test_gather_moves_one_element_per_index():
    # Shape taken from an embedding that lowered to aten::gather: the index
    # is expanded to the output, so each element carries its own index.
    event = _event(
        [[32128, 4096], [], [256, 4096], [], [256, 4096]],
        ["c10::BFloat16", "Scalar", "long int", "Scalar", "c10::BFloat16"],
        name="aten::gather",
    )
    model = aten_gather(event)
    elems = 256 * 4096
    assert model.param_details["tokens"] == elems
    assert model.param_details["hidden"] == 1
    assert model.bytes() == 2 * elems * 2 + elems * 8


def test_unknown_dtype_has_no_byte_count():
    event = _event([[8, 4], [8]], ["not_a_dtype", "long int"])
    model = aten_embedding(event)
    assert model.flops() == 0
    assert model.bytes() is None


def test_missing_index_tensor_is_rejected():
    event = _event([[8, 4]], ["c10::BFloat16"])
    with pytest.raises(ValueError):
        aten_embedding(event)


@pytest.mark.parametrize(
    "name,cls",
    [
        ("aten::embedding", aten_embedding),
        ("aten::embedding_backward", aten_embedding_dense_backward),
        ("aten::embedding_dense_backward", aten_embedding_dense_backward),
        ("aten::embedding_bag", aten_embedding_bag),
        ("aten::_embedding_bag", aten_embedding_bag),
        ("aten::_embedding_bag_forward_only", aten_embedding_bag),
        ("aten::index_select", aten_index_select),
        ("aten::index_select_backward", aten_index_select_backward),
        ("aten::_embedding_bag_dense_backward", aten_embedding_bag_dense_backward),
        ("aten::gather", aten_gather),
    ],
)
def test_common_ops_resolve_to_the_lookup_model(name, cls):
    assert resolve_perf_model_class(name) is cls
    assert op_to_perf_model_class_map[name] is cls
    assert categorize_torch_op({"name": name}) == "Embedding"
