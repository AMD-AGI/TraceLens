###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An op that cannot change a shape must not be recorded having changed one.

MiniMax's text ``q_norm`` opens with ``x.float()``, and that node recorded::

    raw_op       : float
    input_shapes : [["B", "S", "64", "128"]]
    output_shape : [B, S, 48, 128] float32

The contradiction sat in two adjacent attributes of ONE node and every check
walked past it, because they ask whether a shape RESOLVED -- and 48 is as
resolved as 64. The sibling :func:`_noop_cast_warnings` even opens by stating
this invariant for casts, then only ever compares dtypes.

The wrong number came from the same node id being inferred once WITH its class
(``MiniMaxM3VLAttention`` -> ``[B, S, 64, 128]``) and again by a later root-less
pass that resolved the body against whatever module it reached last. That
overwrite can land on any op, so the invariant is asked of every op it holds
for, not only of the cast where it first showed up.

Membership in that class is decided by torch -- the model's own op name run
against its own callable, across probe shapes and every argument assignment its
own schema allows -- so no operation is named here, and nothing reads what the
node is CALLED. A node labelled "Cast" that records no operation is not checked:
a label is for reading and for naming a producer, not for deciding what an op
does.
"""

from __future__ import annotations

import json

from TraceLens.ModelUtils.shape_inference import TensorSpec
from TraceLens.Visualizer.model_explorer_export.shapes import (
    apply_shape_attrs,
    refine_multi_output_port_shapes,
    widen_broadcast_results,
)
from TraceLens.Visualizer.model_explorer_export.type_check import (
    _shape_preserving_op_warnings,
)


def _node(
    reads: list[str] | None,
    produces: str | None,
    *,
    raw_op: str | None = "float",
    op_type: str | None = "Cast",
    label: str = "Cast",
    **extra,
) -> dict:
    attrs = []
    if op_type is not None:
        attrs.append({"key": "op_type", "value": op_type})
    if raw_op is not None:
        attrs.append({"key": "raw_op", "value": raw_op})
    if reads is not None:
        attrs.append({"key": "input_shapes", "value": json.dumps([reads])})
    if produces is not None:
        attrs.append({"key": "output_shape", "value": produces})
    for key, value in extra.items():
        attrs.append({"key": key, "value": value})
    return {"id": "m/@op_x", "label": label, "attrs": attrs}


class TestTheDefectItCatches:
    def test_the_cast_that_changed_an_axis(self) -> None:
        assert _shape_preserving_op_warnings(
            [_node(["B", "S", "64", "128"], "[B, S, 48, 128] float32")]
        )

    def test_the_message_shows_both_shapes(self) -> None:
        (message,) = _shape_preserving_op_warnings(
            [_node(["B", "S", "64", "128"], "[B, S, 48, 128] float32")]
        )
        assert "'64'" in message and "'48'" in message

    def test_a_dropped_axis_is_caught_too(self) -> None:
        assert _shape_preserving_op_warnings(
            [_node(["B", "S", "128"], "[B, S] float32")]
        )

    def test_the_final_edge_repairs_a_stale_same_rank_shape(self) -> None:
        source = {
            "id": "source",
            "label": "Mean",
        }
        apply_shape_attrs(source, TensorSpec(("B", "S", 1), "bfloat16"))
        target = _node(
            ["B", "S", "1"],
            "[1] bfloat16",
            raw_op="rsqrt",
            op_type="Reciprocal sqrt",
            label="Reciprocal sqrt",
        )
        apply_shape_attrs(target, TensorSpec((1,), "bfloat16"))
        target["incomingEdges"] = [
            {
                "sourceNodeId": "source",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "0",
            }
        ]
        widen_broadcast_results([source, target])
        shape = next(
            attr["value"] for attr in target["attrs"] if attr["key"] == "output_shape"
        )
        assert shape == "[B, S, 1] bfloat16"

    def test_the_final_edge_also_repairs_a_stale_wider_shape(self) -> None:
        source = {"id": "source", "label": "Mean"}
        apply_shape_attrs(source, TensorSpec((1,), "bfloat16"))
        target = _node(
            ["1"],
            "[B, S, 1] bfloat16",
            raw_op="rsqrt",
            op_type="Reciprocal sqrt",
            label="Reciprocal sqrt",
        )
        apply_shape_attrs(target, TensorSpec(("B", "S", 1), "bfloat16"))
        target["incomingEdges"] = [
            {
                "sourceNodeId": "source",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "0",
            }
        ]
        widen_broadcast_results([source, target])
        shape = next(
            attr["value"] for attr in target["attrs"] if attr["key"] == "output_shape"
        )
        assert shape == "[1] bfloat16"


class TestFinalMultiOutputPortShapes:
    def test_chunk_ports_are_slices_of_the_final_whole_shape(self) -> None:
        node = {
            "id": "chunk",
            "label": "Chunk",
            "attrs": [
                {"key": "raw_op", "value": "chunk"},
                {"key": "details", "value": "split_size: 2; dim: -1"},
            ],
            "outputsMetadata": [
                {"id": "0", "attrs": [{"key": "port_label", "value": "left"}]},
                {"id": "1", "attrs": [{"key": "port_label", "value": "right"}]},
            ],
        }
        left = {
            "id": "chunk^@slice_out:0",
            "label": "left",
            "attrs": [{"key": "synthetic", "value": "@slice_out"}],
            "incomingEdges": [
                {
                    "sourceNodeId": "chunk",
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            ],
        }
        apply_shape_attrs(node, TensorSpec(("B", "S", 24, 512), "bfloat16"))
        # ``apply_shape_attrs`` gives every port the whole shape; the final
        # refinement reads the operation's own contract and divides them.
        apply_shape_attrs(left, TensorSpec(("B", "S", 24, 512), "bfloat16"))
        refine_multi_output_port_shapes([node, left])
        refine_multi_output_port_shapes([node, left])
        shapes = [
            next(attr["value"] for attr in port["attrs"] if attr["key"] == "shape")
            for port in node["outputsMetadata"]
        ]
        assert shapes == [
            "[B, S, 24, 256] bfloat16",
            "[B, S, 24, 256] bfloat16",
        ]
        left_shape = next(
            attr["value"] for attr in left["attrs"] if attr["key"] == "output_shape"
        )
        assert left_shape == "[B, S, 24, 256] bfloat16"


class TestTheRestOfTheClass:
    """The same overwrite on an op the Cast-only check could not see."""

    def test_an_activation(self) -> None:
        assert _shape_preserving_op_warnings(
            [
                _node(
                    ["B", "S", "64", "128"],
                    "[B, S, 48, 128] float32",
                    raw_op="silu",
                    op_type=None,
                    label="SiLU",
                )
            ]
        )

    def test_a_dropout(self) -> None:
        assert _shape_preserving_op_warnings(
            [
                _node(
                    ["B", "S", "768"],
                    "[B, S, 1024] float32",
                    raw_op="dropout",
                    op_type=None,
                    label="Dropout",
                )
            ]
        )

    def test_a_layout_copy(self) -> None:
        assert _shape_preserving_op_warnings(
            [
                _node(
                    ["B", "S", "768"],
                    "[B, S, 1024] float32",
                    raw_op="contiguous",
                    op_type=None,
                    label="Contiguous",
                )
            ]
        )


class TestWhatItAccepts:
    def test_a_cast_that_only_changes_dtype(self) -> None:
        assert (
            _shape_preserving_op_warnings(
                [_node(["B", "S", "64", "128"], "[B, S, 64, 128] float32")]
            )
            == []
        )

    def test_a_symbolic_shape_that_matches(self) -> None:
        assert (
            _shape_preserving_op_warnings(
                [_node(["B", "S", "index_n_heads"], "[B, S, index_n_heads] float32")]
            )
            == []
        )

    def test_an_op_that_really_does_change_the_shape(self) -> None:
        """A transpose moving an axis is the op doing its job, not a defect."""
        assert (
            _shape_preserving_op_warnings(
                [
                    _node(
                        ["B", "S", "H", "D"],
                        "[B, H, S, D] float32",
                        raw_op="transpose",
                        op_type=None,
                        label="Transpose",
                    )
                ]
            )
            == []
        )

    def test_a_reduction(self) -> None:
        assert (
            _shape_preserving_op_warnings(
                [
                    _node(
                        ["B", "S", "H"],
                        "[B, S] float32",
                        raw_op="sum",
                        op_type=None,
                        label="Sum",
                    )
                ]
            )
            == []
        )


class TestWhereItSaysNothing:
    def test_a_node_recording_no_operation(self) -> None:
        """Even labelled ``Cast``: the label is not evidence about the op."""
        node = _node(["B", "S", "64"], "[B, S, 48] float32", raw_op=None)
        assert _shape_preserving_op_warnings([node]) == []

    def test_a_constant_cast(self) -> None:
        """A learned weight's conversion loses its operand at render time."""
        node = _node(["B", "S", "64"], "[B, S, 48] float32", constant="true")
        assert _shape_preserving_op_warnings([node]) == []

    def test_more_than_one_recorded_operand(self) -> None:
        """Ambiguous which entry is the tensor being cast."""
        node = _node(["B", "S", "64"], "[B, S, 48] float32")
        node["attrs"] = [
            (
                a
                if a["key"] != "input_shapes"
                else {
                    "key": "input_shapes",
                    "value": json.dumps([["B", "S", "64"], ["1"]]),
                }
            )
            for a in node["attrs"]
        ]
        assert _shape_preserving_op_warnings([node]) == []

    def test_nothing_recorded(self) -> None:
        assert _shape_preserving_op_warnings([_node(None, None)]) == []
