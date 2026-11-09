###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""IndexedCopy is the shared gather formula. An embedding lookup is one caller."""

import pytest

from TraceLens.PerfModel.op_projections import op_work
from TraceLens.PerfModel.perf_model import (
    IndexedCopy,
    aten_embedding,
    aten_embedding_bag,
    aten_embedding_bag_dense_backward,
    aten_embedding_dense_backward,
    aten_gather,
    aten_index,
    aten_index_put,
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
    assert IndexedCopy.bytes_func(4, 8, 2, 8) == 2 * 4 * 8 * 2 + 4 * 8
    assert IndexedCopy.flops_func(4, 8) == 0
    assert IndexedCopy.bytes_func(4, 8, None, 8) is None
    assert IndexedCopy.bytes_func(4, 8, 2, None) is None


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
    assert work.category == "IndexedCopy"
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


def test_aten_index_gathers_rows_from_an_index_vector():
    event = _event(
        [[32000, 4096], [2, 128]],
        ["c10::BFloat16", "long int"],
        name="aten::index",
    )
    model = aten_index(event)
    assert model.flops() == 0
    assert model.param_details["tokens"] == 256
    assert model.param_details["hidden"] == 4096
    assert model.bytes() == 2 * 256 * 4096 * 2 + 256 * 8


def test_aten_index_reads_a_tensor_list_of_indices():
    # Two leading axes, each with its own index. The tail (32,) is the row.
    event = _event(
        [[8, 16, 32], [[4], [4]]],
        ["c10::BFloat16", "TensorList"],
        name="aten::index",
    )
    model = aten_index(event)
    assert model.param_details["tokens"] == 4
    assert model.param_details["hidden"] == 32
    assert model.param_details["index_elems"] == 8
    assert model.bytes() == 2 * 4 * 32 * 2 + 8 * 8


def test_aten_index_with_an_unrecorded_index_list_charges_the_tensor():
    # Kineto records Tensor?[] as an empty placeholder. The indexed tensor is
    # the only shape in the event, so the read and the write are that tensor.
    event = _event(
        [[1, 77, 1280], []],
        ["c10::BFloat16", ""],
        name="aten::index",
    )
    model = aten_index(event)
    assert model.param_details["index_elems"] == 0
    assert model.bytes() == 2 * (1 * 77 * 1280) * 2


def test_index_put_writes_values_and_rereads_when_accumulating():
    event = _event(
        [[100, 16], [4], [4, 16]],
        ["c10::BFloat16", "long int", "c10::BFloat16"],
        concrete=["", "", "", "False"],
        name="aten::index_put",
    )
    model = aten_index_put(event)
    assert model.bytes() == 2 * 4 * 16 * 2 + 4 * 8

    accumulating = _event(
        [[100, 16], [4], [4, 16]],
        ["c10::BFloat16", "long int", "c10::BFloat16"],
        concrete=["", "", "", "True"],
        name="aten::index_put_",
    )
    assert aten_index_put(accumulating).bytes() == 3 * 4 * 16 * 2 + 4 * 8


def test_index_put_of_an_integer_table_uses_the_values_tensor():
    # A cache write records the table as ``int`` and the index list as an empty
    # placeholder. The table is not an index, and the last tensor is the payload.
    event = _event(
        [[3682, 131076], [], [16], []],
        ["int", "", "int", "Scalar"],
        concrete=["", "", "", "False"],
        name="aten::index_put_",
    )
    model = aten_index_put(event)
    assert model.param_details["tokens"] == 16
    assert model.param_details["hidden"] == 1
    assert model.param_details["index_elems"] == 0
    assert model.bytes() == 2 * 16 * 4


def test_index_select_accepts_an_integer_dim_and_an_empty_axis():
    event = _event(
        [[2, 100, 8], [], [4]],
        ["c10::Half", "Scalar", "long int"],
        concrete=["", 1, ""],
        name="aten::index_select",
    )
    along_axis = aten_index_select(event)
    assert along_axis.param_details["hidden"] == 16

    empty_axis = _event(
        [[2, 0, 8], [], [4]],
        ["c10::Half", "Scalar", "long int"],
        concrete=["", "1", ""],
        name="aten::index_select",
    )
    assert aten_index_select(empty_axis).param_details["hidden"] == 0

    # A dim that is not an axis falls back to axis 0.
    out_of_range = _event(
        [[2, 100, 8], [], [4]],
        ["c10::Half", "Scalar", "long int"],
        concrete=["", "9", ""],
        name="aten::index_select",
    )
    assert aten_index_select(out_of_range).param_details["hidden"] == 800


def test_unparsable_shapes_are_skipped():
    event = _event([[32000, "bad"], [4]], ["c10::BFloat16", "long int"])
    with pytest.raises(ValueError):
        aten_embedding(event)

    # A bool is not a dimension, so that tensor is not the table.
    indexed = _event(
        [[True], [8, 4], [2]],
        ["", "c10::BFloat16", "long int"],
        name="aten::index",
    )
    model = aten_index(indexed)
    assert model.param_details["tokens"] == 2
    assert model.param_details["hidden"] == 4


def test_index_without_a_tensor_is_rejected():
    event = _event([[], []], ["", ""], name="aten::index")
    with pytest.raises(ValueError, match="tensor being indexed"):
        aten_index(event)


def test_index_put_with_a_blank_index_dtype_still_counts_the_index():
    event = _event(
        [[8, 16], [4], [4, 16]],
        ["c10::BFloat16", "", "c10::BFloat16"],
        concrete=["", "", "", "False"],
        name="aten::index_put",
    )
    model = aten_index_put(event)
    assert model.param_details["dtype_index"] == "long int"
    assert model.bytes() == 2 * 4 * 16 * 2 + 4 * 8


def test_index_put_accumulate_reads_a_bool_and_defaults_off():
    missing = _event(
        [[100, 16], [4], [4, 16]],
        ["c10::BFloat16", "long int", "c10::BFloat16"],
        name="aten::index_put",
    )
    assert aten_index_put(missing).param_details["value_copies"] == 2

    accumulating = _event(
        [[100, 16], [4], [4, 16]],
        ["c10::BFloat16", "long int", "c10::BFloat16"],
        concrete=["", "", "", True],
        name="aten::index_put_",
    )
    assert aten_index_put(accumulating).param_details["value_copies"] == 3
    assert aten_index_put(accumulating).bytes() == 3 * 4 * 16 * 2 + 4 * 8


def test_tensor_list_skips_slices_and_rejects_a_malformed_list():
    sliced = _event(
        [[8, 16, 32], [[4], None]],
        ["c10::BFloat16", "TensorList"],
        name="aten::index",
    )
    model = aten_index(sliced)
    assert model.param_details["tokens"] == 4
    assert model.param_details["hidden"] == 16 * 32
    assert model.param_details["index_elems"] == 4

    malformed = _event(
        [[8, 16], ["nope", [4]]],
        ["c10::BFloat16", "TensorList"],
        name="aten::index",
    )
    whole = aten_index(malformed)
    assert whole.param_details["index_elems"] == 0
    assert whole.bytes() == 2 * (8 * 16) * 2

    non_numeric = _event(
        [[8, 16], [[4, "x"]]],
        ["c10::BFloat16", "TensorList"],
        name="aten::index",
    )
    assert aten_index(non_numeric).param_details["index_elems"] == 0


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
        ("aten::index", aten_index),
        ("aten::index_put", aten_index_put),
        ("aten::index_put_", aten_index_put),
        ("aten::_index_put_impl_", aten_index_put),
    ],
)
def test_common_ops_resolve_to_the_lookup_model(name, cls):
    assert resolve_perf_model_class(name) is cls
    assert op_to_perf_model_class_map[name] is cls
    assert cls.category == "IndexedCopy"
    assert categorize_torch_op({"name": name}) == "IndexedCopy"
