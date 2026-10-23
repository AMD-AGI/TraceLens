###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The grid a vision tower is handed is a model input, not the image patches.

``Glm5NextVisionModel.forward(self, hidden_states, grid_thw)`` takes two tensors.
The first is the activation; the second the caller passes straight from its own
signature::

    self.visual(pixel_values, grid_thw=image_grid_thw)   # image_grid_thw: LongTensor

Nothing computes ``grid_thw``, so with no boundary of its own it docked onto the
tower's activation ``@input`` -- the flat image patches. The shape was then
correctly inherited from the wrong tensor, so every node downstream agreed and
the whole ``cu_seqlens`` chain ran on the patch axis::

    grid_thw   [Pv, 1176] bfloat16     instead of  [Img, 3] int64
    cu_seqlens [Pv + 1]   bfloat16     instead of  [Img + 1] int64

against a source docstring that says ``(num_segments + 1,) int32``. ``Pv`` counts
patches; one image contributes many.

The width is measured from the columns the readers take -- ``grid[:, 0]``,
``[:, 1]``, ``[:, 2]`` is three wide -- because nothing else in the model states
it. That is what ``select_index`` records.
"""

from __future__ import annotations

import ast

from TraceLens.ModelUtils.ast_analyze import (
    _subscript_select_indices,
    analyze_source,
)


def _slice_expr(source: str) -> ast.AST:
    return ast.parse(source, mode="eval").body.slice


class TestSelectIndex:
    def test_a_column_select(self) -> None:
        assert _subscript_select_indices(_slice_expr("grid[:, 2]")) == [2]

    def test_a_trailing_select_after_ellipsis(self) -> None:
        assert _subscript_select_indices(_slice_expr("x[..., -1]")) == [-1]

    def test_a_bare_index_is_not_a_column_select(self) -> None:
        """``x[0]`` is a pass-through alias, not an axis drop."""
        assert _subscript_select_indices(_slice_expr("x[0]")) == []

    def test_a_range_slice_selects_no_column(self) -> None:
        assert _subscript_select_indices(_slice_expr("x[:, :]")) == []

    def test_every_column_of_a_fan_out(self) -> None:
        assert _subscript_select_indices(_slice_expr("grid[0, 1]")) == [0, 1]


_SOURCE = """
class M(nn.Module):
    def forward(self, grid_thw):
        seqlens = grid_thw[:, 1] * grid_thw[:, 2]
        repeated = torch.repeat_interleave(seqlens, grid_thw[:, 0])
        return F.pad(repeated.cumsum(dim=0), (1, 0), value=0)
"""


class TestTheSelectsAreRecorded:
    def test_each_slice_records_the_column_it_takes(self) -> None:
        """Without the column there is no record of how wide the grid is."""
        analysis = analyze_source(_SOURCE, config={"hidden_size": 8})
        columns = sorted(
            detail
            for op in analysis.class_registry["M"].forward_operations.values()
            for detail in (op.details or [])
            if detail.startswith("select_index:")
        )
        assert columns == [
            "select_index: 0",
            "select_index: 1",
            "select_index: 2",
        ], columns

    def test_the_axis_is_still_recorded_alongside(self) -> None:
        """``select_dim`` drops the axis; ``select_index`` says which column."""
        analysis = analyze_source(_SOURCE, config={"hidden_size": 8})
        for op in analysis.class_registry["M"].forward_operations.values():
            details = op.details or ()
            if any(d.startswith("select_index:") for d in details):
                assert any(d.startswith("select_dim:") for d in details), details
