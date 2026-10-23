###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A reshape that declares N target axes must produce N axes.

The ``shape:`` detail is the model's own argument list, so its LENGTH is a fact
even when the individual dims are not: ``chunk_kv.view(batch, n_windows, ratio,
-1)`` names four axes whatever ``n_windows`` turns out to be. Rank is therefore
checkable where the dims are not, and checking it is what separates "this axis
is still symbolic" from "this is not the same tensor".

Nothing was asking. DeepSeek's compressor view declared four axes and produced
``[B*S, 1024]`` -- two. Every dim in that result is resolved, so the
unresolved-shape check was satisfied; the conservation check folds dims and
declined on the symbols; and the rank simply went unexamined. The rank-4 slices
downstream then addressed an axis the operand did not have, and THAT is how it
finally showed up -- three steps from where it happened.

Only the length is asserted, never the dims: a symbolic axis is normal.
"""

from __future__ import annotations

import json

from TraceLens.Visualizer.model_explorer_export.type_check import (
    _declared_rank_warnings,
    _spans_several_axes,
)


def _node(details, output_shape, op_type="View", node_id="n1"):
    return {
        "id": node_id,
        "attrs": [
            {"key": "op_type", "value": op_type},
            {"key": "details", "value": "; ".join(details)},
            {"key": "output_shape", "value": output_shape},
        ],
    }


class TestTheRankMustMatch:
    def test_fewer_axes_than_declared(self) -> None:
        """DeepSeek's compressor: four declared, two produced."""
        warnings = _declared_rank_warnings(
            [
                _node(
                    ["shape: hidden_states.shape[0], n_windows, ratio, -1"],
                    "[B*S, 1024] bfloat16",
                )
            ]
        )
        assert len(warnings) == 1
        assert "declares 4 target axes" in warnings[0]
        assert "has 2" in warnings[0]

    def test_more_axes_than_declared(self) -> None:
        """MiniMax's repeat_kv: the collapsing reshape never happened."""
        warnings = _declared_rank_warnings(
            [
                _node(
                    ["shape: a, b, c, d"],
                    "[B, 4, 1, S, 128] bfloat16",
                    op_type="Reshape",
                )
            ]
        )
        assert len(warnings) == 1

    def test_a_following_detail_is_not_read_as_more_axes(self) -> None:
        """Details render ``a; b`` on one line; the shape ends at the ``;``."""
        warnings = _declared_rank_warnings(
            [
                _node(
                    [
                        "shape: hidden_states.shape[0], hidden_states.shape[1], -1, 128",
                        "shape_snapshots: hidden_states.shape[0], hidden_states.shape[1]",
                    ],
                    "[B, S, 64, 128] bfloat16",
                )
            ]
        )
        assert warnings == []


class TestWhatIsAccepted:
    def test_the_rank_it_declared(self) -> None:
        assert (
            _declared_rank_warnings(
                [_node(["shape: a, b, -1"], "[B, S, 4096] bfloat16")]
            )
            == []
        )

    def test_a_symbolic_axis_is_nobody_s_business_here(self) -> None:
        """Only the count is asserted -- an unresolved dim is not this check's."""
        assert (
            _declared_rank_warnings(
                [
                    _node(
                        ["shape: batch, n_windows, ratio, -1"], "[B, n_windows, 4, 512]"
                    )
                ]
            )
            == []
        )

    def test_one_argument_standing_for_a_whole_shape(self) -> None:
        """``x.reshape(x.shape)`` says nothing about how many axes come out."""
        assert (
            _declared_rank_warnings([_node(["shape: orig_shape"], "[B, S, H]")]) == []
        )

    def test_an_unpacked_shape_slice(self) -> None:
        """``*x.shape[:2]`` stands for several axes, so the count is not the rank."""
        assert (
            _declared_rank_warnings(
                [_node(["shape: *hidden.shape[:2], -1"], "[B, S, 64, 128]")]
            )
            == []
        )

    def test_a_constant_node_is_skipped(self) -> None:
        node = _node(["shape: a, b, c"], "[B, S]")
        node["attrs"].append({"key": "constant", "value": "true"})
        assert _declared_rank_warnings([node]) == []

    def test_a_node_with_no_declaration(self) -> None:
        assert _declared_rank_warnings([_node(["raw_op: view"], "[B, S]")]) == []

    def test_a_node_with_no_output_shape(self) -> None:
        node = {
            "id": "n",
            "attrs": [
                {"key": "op_type", "value": "View"},
                {"key": "details", "value": "shape: a, b, c"},
            ],
        }
        assert _declared_rank_warnings([node]) == []

    def test_an_empty_graph(self) -> None:
        assert _declared_rank_warnings([]) == []


class TestWhichArgumentsSpanSeveralAxes:
    def test_an_unpack(self) -> None:
        assert _spans_several_axes("*x.shape[:2]") is True

    def test_a_shape_slice(self) -> None:
        assert _spans_several_axes("x.shape[:-1]") is True

    def test_a_scalar_shape_read_is_one_axis(self) -> None:
        assert _spans_several_axes("hidden_states.shape[0]") is False

    def test_a_plain_name_or_literal(self) -> None:
        assert _spans_several_axes("n_windows") is False
        assert _spans_several_axes("-1") is False


def test_json_fixture_shape_matches_the_exporter() -> None:
    """Guards the fixture itself: attrs are rendered strings, not objects."""
    node = _node(["shape: a, b"], "[B, S]")
    assert all(isinstance(a["value"], str) for a in node["attrs"])
    assert json.loads("[]") == []
