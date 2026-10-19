###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An op that declares WHICH axis it changes must have that axis.

``select_dim`` / ``resize_dim`` name an axis by index, read off the model's own
subscript. The index is therefore a fact about the tensor the model slices -- so
an operand with fewer axes than that contradicts it, and one of the two is
wrong: either the operand was mis-wired, or whatever produced it reported the
wrong rank.

This is the check that was missing when Kimi's post-attention
``attn_output[:, :, :, : self.v_head_dim]`` (``resize_dim: 3``) was handed a
RANK-3 operand, because the kernel above it reported the enclosing module's
output rather than its own. Nothing objected: the shape rules wrote the declared
width at whatever axis they could reach, giving ``[128, S, 7168]``, and every
existing check asked only whether the dims RESOLVED. They did -- they were just
not this tensor's.

The same invariant immediately found four more of the same kind in DeepSeek's
compressor, which had gone equally unreported.
"""

from __future__ import annotations

import json

from TraceLens.Visualizer.model_explorer_export.type_check import (
    _declared_axis_warnings,
)


def _node(shapes, details, op_type="Slice", node_id="n1"):
    """A node as the exporter writes one: attrs carry JSON / rendered strings."""
    return {
        "id": node_id,
        "attrs": [
            {"key": "op_type", "value": op_type},
            {"key": "input_shapes", "value": json.dumps(shapes)},
            {"key": "details", "value": "; ".join(details)},
        ],
    }


class TestTheAxisMustExist:
    def test_an_axis_beyond_the_operands_rank(self) -> None:
        """Kimi's case: a rank-4 narrowing handed a rank-3 tensor."""
        warnings = _declared_axis_warnings(
            [_node([["B", "S", "7168"]], ["resize_dim: 3=128"])]
        )
        assert len(warnings) == 1
        assert "rank 3" in warnings[0] and "resize_dim 3" in warnings[0]

    def test_a_select_dim_too(self) -> None:
        warnings = _declared_axis_warnings([_node([["B", "S"]], ["select_dim: 2"])])
        assert len(warnings) == 1

    def test_the_operand_two_ranks_short(self) -> None:
        """DeepSeek's compressor: ``resize_dim: 3`` against ``[B*S, 1024]``."""
        warnings = _declared_axis_warnings(
            [_node([["B*S", "1024"]], ["resize_dim: 3=256"])]
        )
        assert len(warnings) == 1


class TestWhatIsAccepted:
    def test_an_axis_the_operand_has(self) -> None:
        assert (
            _declared_axis_warnings(
                [_node([["B", 96, "S", 128]], ["resize_dim: 3=128"])]
            )
            == []
        )

    def test_the_last_axis_exactly(self) -> None:
        assert _declared_axis_warnings([_node([["B", "S"]], ["select_dim: 1"])]) == []

    def test_a_negative_axis_counts_from_the_end(self) -> None:
        assert (
            _declared_axis_warnings([_node([["B", "S", "H"]], ["resize_dim: -1=64"])])
            == []
        )

    def test_a_negative_axis_still_out_of_range(self) -> None:
        assert len(_declared_axis_warnings([_node([["B"]], ["resize_dim: -3=4"])])) == 1

    def test_the_first_tensor_operand_is_the_one_addressed(self) -> None:
        """A trailing scalar operand carries no axes and must not be read as one."""
        assert (
            _declared_axis_warnings(
                [_node([["B", 96, "S", 128], []], ["resize_dim: 3=128"])]
            )
            == []
        )


class TestWhatItDeclines:
    def test_a_node_with_no_declaration(self) -> None:
        assert _declared_axis_warnings([_node([["B", "S"]], ["raw_op: slice"])]) == []

    def test_a_node_with_no_operand_shapes(self) -> None:
        assert _declared_axis_warnings([_node([], ["resize_dim: 3=128"])]) == []

    def test_a_declaration_that_is_not_an_index(self) -> None:
        """``shape_slice`` narrows by arithmetic and names no axis index."""
        assert (
            _declared_axis_warnings(
                [_node([["B", "S"]], ["shape_slice: x.shape[-1] - 1"])]
            )
            == []
        )

    def test_an_unparseable_axis(self) -> None:
        assert _declared_axis_warnings([_node([["B"]], ["resize_dim: n=4"])]) == []

    def test_an_empty_graph(self) -> None:
        assert _declared_axis_warnings([]) == []
