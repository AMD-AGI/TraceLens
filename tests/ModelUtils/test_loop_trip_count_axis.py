###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A loop over a tensor's rows is named for the axis it walks.

GLM's vision position-ids helper runs once per image::

    for (t, h, w), merge_size in zip(grid_thw.tolist(), spatial_merge_size.tolist()):

The trip count is not a constant, so the frame read ``loop_iterations_repeated``
-- true, and uninformative. The rows of a vision grid descriptor ARE the image
axis: the same wiring that gives that tensor its boundary shapes it
``[Img, columns]``. So the frame can say ``Img``, which is the count.

``.tolist()`` is what makes this worth saying. It moves the value to the host,
so the body walks Python numbers rather than slices of a tensor and there is no
row-select op to carry the count -- but how many times it runs is still a
property of the tensor those numbers came from.

The analyser records WHICH tensor; only a stage that knows that tensor's shape
can name the axis. A loop over anything whose leading axis nothing has named
keeps ``repeated``, because inventing a symbol would assert a count nobody
established.
"""

from __future__ import annotations

import ast
import textwrap

from TraceLens.ModelUtils.ast_analyze import _loop_row_source_name, analyze_source


def _loop(body: str) -> ast.For:
    tree = ast.parse(textwrap.dedent(body))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.For))


class TestWhichTensorTheLoopWalks:
    def test_a_zip_of_host_lists_names_the_first(self) -> None:
        """A ``zip`` walks its sequences in step, so any measures the loop."""
        node = _loop("""
            for (t, h, w), m in zip(grid_thw.tolist(), spatial_merge_size.tolist()):
                pass
            """)
        assert _loop_row_source_name(node) == "grid_thw"

    def test_a_bare_host_list_is_named(self) -> None:
        node = _loop("""
            for row in grid_thw.tolist():
                pass
            """)
        assert _loop_row_source_name(node) == "grid_thw"

    def test_a_loop_over_a_range_names_nothing(self) -> None:
        node = _loop("""
            for i in range(4):
                pass
            """)
        assert _loop_row_source_name(node) is None

    def test_a_loop_over_a_tensor_itself_names_nothing(self) -> None:
        """Without ``.tolist()`` the body slices rows, and the row-select op
        already carries the loop -- there is nothing extra to say."""
        node = _loop("""
            for row in grid_thw:
                pass
            """)
        assert _loop_row_source_name(node) is None


class TestTheAnalyserRecordsIt:
    SOURCE = """
import torch
import torch.nn as nn


class M(nn.Module):
    def forward(self, hidden_states, grid_thw):
        out = []
        for t, h, w in grid_thw.tolist():
            out.append(torch.arange(w))
        return torch.cat(out)
"""

    def _details(self) -> list[str]:
        analysis = analyze_source(self.SOURCE, config={"hidden_size": 8})
        structure = analysis.class_registry["M"]
        return [
            line
            for lines in (structure.forward_step_details or {}).values()
            for line in lines
        ]

    def test_the_row_source_is_recorded(self) -> None:
        assert "loop_rows_of: grid_thw" in self._details(), self._details()

    def test_the_count_is_still_reported_as_unbounded(self) -> None:
        """The analyser does not know the axis -- only which tensor to ask."""
        assert "loop: repeated" in self._details(), self._details()
