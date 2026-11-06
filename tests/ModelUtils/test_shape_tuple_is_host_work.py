###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Building a shape is host bookkeeping, and ``addmm`` is a projection.

GPT-2 does every projection through ``Conv1D``::

    size_out = x.size()[:-1] + (self.nf,)
    x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    x = x.view(size_out)

Both lines were read wrongly, in opposite directions.

The first line computes a SHAPE. ``x.shape[...]`` was already understood as host
bookkeeping but ``x.size()[...]`` -- the same thing spelled as a call -- was not,
and a tuple display was not either. So the slice became a tensor ``Slice``
claiming to be ``[B, S, 768]``, the concatenation became an ``Add`` that nothing
could consume, and ``self.nf`` was materialised as a ``[1]`` constant tensor.
Ten dead nodes in GPT-2, all of them a shape pretending to be a tensor.

The second line is the actual matrix multiply of the model -- and produced NO
op at all, because ``addmm`` was in none of the tables. GPT-2's QKV projection,
its largest single operation, was drawn as nothing.

``addmm(bias, x, weight)`` also stores its weight ``[in, out]``, the TRANSPOSE
of what ``F.linear`` uses, so the output width is the column axis. Reading it
the other way gives every projection its input width back.
"""

from __future__ import annotations

from TraceLens.ModelUtils.ast_analyze import analyze_source

SOURCE = """
import torch
import torch.nn as nn


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
"""


def _ops(source: str = SOURCE, cls: str = "Conv1D") -> dict[str, list[str]]:
    analysis = analyze_source(source, config={"hidden_size": 768})
    return dict(analysis.class_registry[cls].forward_step_details or {})


class TestBuildingAShapeEmitsNoTensorOp:
    def test_the_slice_of_a_size_call_is_not_a_tensor_slice(self) -> None:
        assert not any("slice" in op for op in _ops()), _ops()

    def test_concatenating_the_tuple_is_not_a_tensor_add(self) -> None:
        """It produced a value nothing consumed -- a dead node per Conv1D."""
        assert not any("_add" in op for op in _ops()), _ops()

    def test_the_call_spelling_matches_the_attribute_spelling(self) -> None:
        """``x.size()[:-1]`` and ``x.shape[:-1]`` are the same statement."""
        attribute_form = SOURCE.replace("x.size()[:-1]", "x.shape[:-1]")
        assert set(_ops(attribute_form)) == set(_ops())


class TestTheProjectionIsVisible:
    def test_addmm_produces_an_op(self) -> None:
        ops = _ops()
        assert ops, "GPT-2's QKV projection was drawn as nothing at all"

    def test_it_records_the_torch_op_it_came_from(self) -> None:
        """So the shape rule can tell its weight layout from ``F.linear``'s."""
        ops = _ops()
        (details,) = ops.values()
        assert "raw_op: addmm" in details, ops


class TestWhatIsStillATensorOp:
    def test_adding_two_tensors_is_untouched(self) -> None:
        source = """
import torch
import torch.nn as nn


class M(nn.Module):
    def forward(self, a, b):
        return a + b
"""
        assert _ops(source, "M"), "a real elementwise add must survive"

    def test_a_tuple_argument_keeps_its_tensor_operands(self) -> None:
        """``torch.cat((a, b))`` reads two tensors; a tuple is not host there."""
        source = """
import torch
import torch.nn as nn


class M(nn.Module):
    def forward(self, a, b):
        return torch.cat((a, b), dim=-1)
"""
        ops = _ops(source, "M")
        assert any("cat" in op or "concat" in op for op in ops), ops
