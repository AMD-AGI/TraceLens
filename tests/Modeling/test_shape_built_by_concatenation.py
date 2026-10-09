###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A shape assembled by concatenation is still a shape.

A reshape target written as one tuple was already expanded::

    hidden_shape = (*input_shape, -1, self.head_dim)
    x.view(hidden_shape)

GPT-2's ``Conv1D`` builds the same thing by CONCATENATING instead::

    size_out = x.size()[:-1] + (self.nf,)
    x = torch.addmm(...)
    x = x.view(size_out)

which is a binary operation, not a tuple, so nothing expanded it. ``view`` was
handed the single token ``size_out``, the resolver had one name and no axes, and
every GPT-2 projection came out rank-1 ``[size_out]`` -- a shape printing a bare
identifier, which nothing flags because the shape checks look for ``?``, ``[]``
and ``-1``.

``x.size()[:-1]`` also had to be read as the leading-axes slice it is; only the
``.shape`` spelling was, though the two are the same expression.
"""

from __future__ import annotations

from TraceLens.ModelUtils.ast_analyze import analyze_source

TEMPLATE = """
import torch
import torch.nn as nn


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))

    def forward(self, x):
        size_out = {expr}
        x = x.view(size_out)
        return x
"""


def _view_detail(expr: str) -> str:
    source = TEMPLATE.format(expr=expr)
    analysis = analyze_source(source, config={"hidden_size": 768}, all_tensor_ops=True)
    details = analysis.class_registry["Conv1D"].forward_step_details or {}
    for lines in details.values():
        for line in lines:
            if line.startswith("shape: "):
                return line[len("shape: ") :]
    return ""


class TestTheTargetIsExpanded:
    def test_a_concatenated_shape_names_its_axes(self) -> None:
        detail = _view_detail("x.size()[:-1] + (self.nf,)")
        assert detail == "*x.shape[:-1], self.nf", detail

    def test_the_shape_spelling_reads_the_same(self) -> None:
        """``x.size()[:-1]`` and ``x.shape[:-1]`` are one expression."""
        assert _view_detail("x.shape[:-1] + (self.nf,)") == _view_detail(
            "x.size()[:-1] + (self.nf,)"
        )

    def test_several_trailing_axes_are_all_kept(self) -> None:
        detail = _view_detail("x.size()[:-1] + (-1, self.nf)")
        assert detail == "*x.shape[:-1], -1, self.nf", detail

    def test_it_is_not_left_as_one_bare_name(self) -> None:
        """A bare name renders as a shape no check can tell is broken."""
        assert "size_out" not in _view_detail("x.size()[:-1] + (self.nf,)")


class TestWhatIsNotAShape:
    def test_adding_two_tensors_is_not_a_shape(self) -> None:
        assert _view_detail("x + x") == "size_out"

    def test_a_slice_with_no_static_bound_is_not_taken(self) -> None:
        detail = _view_detail("x.size()[:n] + (self.nf,)")
        assert detail == "size_out", detail
