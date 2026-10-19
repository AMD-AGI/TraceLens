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

from TraceLens.ModelUtils.shape_inference import op_cannot_change_shape
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


class TestTorchDecidesTheClass:
    """The membership the check relies on, asked directly."""

    def test_ops_that_cannot_change_a_shape(self) -> None:
        for name in ("float", "dropout", "contiguous", "clone", "detach", "silu"):
            assert op_cannot_change_shape(name), name

    def test_ops_that_can(self) -> None:
        for name in ("transpose", "flatten", "reshape", "sum", "cat", "expand"):
            assert not op_cannot_change_shape(name), name

    def test_an_op_a_small_shape_would_hide(self) -> None:
        """A squeeze needs a size-1 axis, and ``view`` keeps a one-axis shape
        through its dtype overload -- both look like the identity if the probe
        is not wide enough."""
        assert not op_cannot_change_shape("squeeze")
        assert not op_cannot_change_shape("view")

    def test_an_op_torch_does_not_know_is_not_answered_for(self) -> None:
        assert not op_cannot_change_shape("some_custom_kernel")
        assert not op_cannot_change_shape("")
