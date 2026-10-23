###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""What the shape checks do when a node tells them something unusable.

Each check reads a node's own record -- the operation it performed, the shapes
it was handed, the dtype it produced. A record can be absent, truncated, or
malformed, and a check that is going to run over every node of every model has
to walk past all of those without raising and without inventing a warning.

These are the paths where a check declines. They matter as much as the ones
where it speaks: a false warning on a malformed record would send someone
looking for a defect that is not there, and an exception would take the whole
export down over one bad attribute.
"""

from __future__ import annotations

import json

from TraceLens.Visualizer.model_explorer_export.type_check import (
    _counts_provably_differ,
    _dim_factors,
    _in_place_write_warnings,
    _reshape_conservation_warnings,
    _shape_preserving_op_warnings,
    projection_weight_bias_widths,
)


def _node(node_id="n", label="Cast", **attrs):
    return {
        "id": node_id,
        "label": label,
        "attrs": [{"key": key, "value": value} for key, value in attrs.items()],
    }


class TestAShapePreservingOpWithAnUnusableRecord:
    def test_input_shapes_that_are_not_json(self) -> None:
        node = _node(raw_op="float", input_shapes="{not json", output_shape="[B] f32")
        assert _shape_preserving_op_warnings([node]) == []

    def test_input_shapes_that_are_not_a_list(self) -> None:
        node = _node(raw_op="float", input_shapes='"B, S"', output_shape="[B, S] f32")
        assert _shape_preserving_op_warnings([node]) == []

    def test_an_empty_operand(self) -> None:
        node = _node(raw_op="float", input_shapes="[[]]", output_shape="[B, S] f32")
        assert _shape_preserving_op_warnings([node]) == []

    def test_no_output_shape_recorded(self) -> None:
        node = _node(raw_op="float", input_shapes='[["B", "S"]]')
        assert _shape_preserving_op_warnings([node]) == []

    def test_no_input_shapes_recorded(self) -> None:
        assert _shape_preserving_op_warnings([_node(raw_op="float")]) == []


class TestReadingAShapeAsAnElementCount:
    def test_a_product_is_the_same_count_written_differently(self) -> None:
        assert _dim_factors(["B", "S*768"]) == _dim_factors(["B", "S", "768"])

    def test_a_size_one_axis_multiplies_nothing(self) -> None:
        assert _dim_factors(["1", "8192", "1"]) == ["8192"]

    def test_a_placeholder_axis_states_no_count(self) -> None:
        """``-1`` stands for whatever makes the count come out right."""
        assert _dim_factors(["B", "-1"]) is None

    def test_an_empty_token_states_no_count(self) -> None:
        assert _dim_factors(["B", ""]) is None

    def test_counts_that_cancel_do_not_differ(self) -> None:
        assert not _counts_provably_differ(["B", "S", "768"], ["B", "S*768"])

    def test_two_symbolic_remainders_prove_nothing(self) -> None:
        """The same count can be written with different symbols."""
        assert not _counts_provably_differ(["Pv", "1176"], ["Img", "3"])

    def test_literal_remainders_that_disagree(self) -> None:
        assert _counts_provably_differ(["B", "S", "768"], ["B", "S", "1024"])

    def test_a_placeholder_on_either_side_proves_nothing(self) -> None:
        assert not _counts_provably_differ(["B", "-1"], ["B", "S", "768"])
        assert not _counts_provably_differ(["B", "S", "768"], ["B", "-1"])


class TestTheReshapeConservationCheck:
    def test_a_view_that_cannot_hold_its_elements(self) -> None:
        node = _node(
            raw_op="view",
            label="View",
            input_shapes='[["B*S", "33792"]]',
            output_shape="[B, S, 7168] bfloat16",
        )
        (message,) = _reshape_conservation_warnings([node])
        assert "same number of elements" in message

    def test_a_view_that_only_moves_the_axes(self) -> None:
        node = _node(
            raw_op="view",
            label="View",
            input_shapes='[["B*S", "4096"]]',
            output_shape="[B, S, 4096] bfloat16",
        )
        assert _reshape_conservation_warnings([node]) == []

    def test_a_reshape_that_converts_a_dtype(self) -> None:
        node = _node(
            raw_op="view",
            label="View",
            details="shape: *orig_shape",
            input_shapes='[["nnz", "1"]]',
            input_types='["int64"]',
            output_shape="[nnz, 1] bfloat16",
            output_dtype="bfloat16",
        )
        (message,) = _reshape_conservation_warnings([node])
        assert "int64" in message and "bfloat16" in message

    def test_malformed_input_shapes(self) -> None:
        node = _node(raw_op="view", input_shapes="{{", output_shape="[B] f32")
        assert _reshape_conservation_warnings([node]) == []

    def test_a_constant_is_skipped(self) -> None:
        node = _node(
            raw_op="view",
            constant="true",
            input_shapes='[["B*S", "33792"]]',
            output_shape="[B, S, 7168] bfloat16",
        )
        assert _reshape_conservation_warnings([node]) == []

    def test_an_op_that_may_change_the_count(self) -> None:
        """``expand`` really does make more elements than it was given."""
        node = _node(
            raw_op="expand",
            input_shapes='[["B", "1", "S"]]',
            output_shape="[B, 4, S] bfloat16",
        )
        assert _reshape_conservation_warnings([node]) == []

    def test_a_declared_shape_with_no_recorded_dtypes(self) -> None:
        node = _node(
            raw_op="view",
            details="shape: *orig_shape",
            input_shapes='[["B", "S"]]',
            output_shape="[B, S] bfloat16",
        )
        assert _reshape_conservation_warnings([node]) == []

    def test_a_dtype_that_carries_no_concrete_type(self) -> None:
        node = _node(
            raw_op="view",
            details="shape: *orig_shape",
            input_shapes='[["B", "S"]]',
            input_types='["Constant"]',
            output_shape="[B, S] bfloat16",
            output_dtype="bfloat16",
        )
        assert _reshape_conservation_warnings([node]) == []


class TestTheInPlaceWriteCheck:
    def test_a_write_reporting_its_index_instead_of_its_receiver(self) -> None:
        node = _node(
            raw_op="index_add_",
            label="Index add",
            details="mutates: final; loop: 288 iterations",
            input_shapes='[["nnz", "1"], ["nnz", "1"], ["1"]]',
            output_shape="[nnz, 1] int64",
        )
        # Reads what it writes into and reports the same: no complaint.
        assert _in_place_write_warnings([node]) == []

    def test_a_receiver_and_a_result_that_disagree(self) -> None:
        node = _node(
            raw_op="index_add_",
            label="Index add",
            details="mutates: final",
            input_shapes='[["B*S", "4096"], ["nnz", "1"], ["1"]]',
            output_shape="[nnz, 1] int64",
        )
        (message,) = _in_place_write_warnings([node])
        assert "writes into" in message and "'final'" in message

    def test_a_write_that_changes_the_dtype_it_writes_into(self) -> None:
        node = _node(
            raw_op="index_add_",
            label="Index add",
            details="mutates: final",
            input_shapes='[["B*S", "4096"], ["nnz", "1"]]',
            input_types='["bfloat16", "int64"]',
            output_shape="[B*S, 4096] int64",
            output_dtype="int64",
        )
        messages = _in_place_write_warnings([node])
        assert any("keeps the dtype" in m for m in messages)

    def test_a_node_that_mutates_nothing(self) -> None:
        node = _node(
            raw_op="add",
            input_shapes='[["B", "S"]]',
            output_shape="[B, S, 4] f32",
        )
        assert _in_place_write_warnings([node]) == []

    def test_a_constant_is_skipped(self) -> None:
        node = _node(
            raw_op="copy_",
            constant="true",
            details="mutates: x",
            input_shapes='[["B", "S"]]',
            output_shape="[B] f32",
        )
        assert _in_place_write_warnings([node]) == []

    def test_no_shapes_recorded(self) -> None:
        node = _node(raw_op="copy_", details="mutates: x")
        assert _in_place_write_warnings([node]) == []

    def test_an_empty_receiver_record(self) -> None:
        node = _node(
            raw_op="copy_",
            details="mutates: x",
            input_shapes="[[]]",
            output_shape="[B, S] f32",
        )
        assert _in_place_write_warnings([node]) == []

    def test_a_receiver_dtype_that_carries_no_concrete_type(self) -> None:
        node = _node(
            raw_op="copy_",
            details="mutates: x",
            input_shapes='[["B", "S"]]',
            input_types='["Tensor"]',
            output_shape="[B, S] f32",
            output_dtype="f32",
        )
        assert _in_place_write_warnings([node]) == []


class TestTheProjectionWidthCheck:
    def test_a_weight_whose_shape_carries_no_numbers(self) -> None:
        nodes = [
            {
                "id": "m/@op_x:const:weight",
                "label": "weight",
                "attrs": [{"key": "output_shape", "value": "[in, out] bfloat16"}],
            }
        ]
        assert projection_weight_bias_widths(nodes) == []

    def test_a_constant_with_no_recorded_shape(self) -> None:
        nodes = [{"id": "m/@op_x:const:weight", "label": "weight", "attrs": []}]
        assert projection_weight_bias_widths(nodes) == []

    def test_a_constant_that_is_neither_weight_nor_bias(self) -> None:
        nodes = [
            {
                "id": "m/@op_x:const:eps",
                "label": "eps",
                "attrs": [{"key": "output_shape", "value": "[1] bfloat16"}],
            }
        ]
        assert projection_weight_bias_widths(nodes) == []

    def test_a_node_that_is_not_a_constant(self) -> None:
        nodes = [
            {
                "id": "m/@op_x",
                "label": "weight",
                "attrs": [{"key": "output_shape", "value": "[768, 2304] bfloat16"}],
            }
        ]
        assert projection_weight_bias_widths(nodes) == []


class TestEveryCheckSurvivesAnEmptyGraph:
    def test_nothing_to_look_at(self) -> None:
        for check in (
            _shape_preserving_op_warnings,
            _reshape_conservation_warnings,
            _in_place_write_warnings,
            projection_weight_bias_widths,
        ):
            assert check([]) == [], check.__name__

    def test_a_node_with_no_attrs_at_all(self) -> None:
        bare = [{"id": "n", "label": "x"}]
        for check in (
            _shape_preserving_op_warnings,
            _reshape_conservation_warnings,
            _in_place_write_warnings,
            projection_weight_bias_widths,
        ):
            assert check(bare) == [], check.__name__

    def test_input_shapes_holding_a_non_list_entry(self) -> None:
        node = _node(
            raw_op="float",
            input_shapes=json.dumps(["B, S"]),
            output_shape="[B, S] f32",
        )
        assert _shape_preserving_op_warnings([node]) == []
