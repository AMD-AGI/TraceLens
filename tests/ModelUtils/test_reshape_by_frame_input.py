###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A reshape can name the frame's own input, and torch can say what preserves shape.

GPT-2's ``Conv1D`` flattens, projects, then restores::

    size_out = x.size()[:-1] + (self.nf,)
    x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    x = x.view(size_out)

The ``x`` measured on the first line is NOT the ``x`` being reshaped on the
third -- it was rebound in between. Reading the leading axes from the tensor
being reshaped gives ``[B*S, nf]`` where the model has ``[B, S, nf]``: one axis
short, self-consistent, and the ``split(..., dim=2)`` that follows then has no
axis 2.

Two guards matter as much as the fix:

* Only a SNAPSHOT may look elsewhere. An inline ``*x.shape[:-1]`` written in the
  reshape itself names the tensor being reshaped, and reading another gave
  MiniMax's attention a fourth axis and then a second batch axis.
* The trailing axis may only be settled by element conservation when the
  leading axes came from a named tensor. Allowing it generally collapsed GLM's
  ``[B, S, 4096]`` to ``[B*S*4096]``.
"""

from __future__ import annotations

from TraceLens.ModelUtils.shape_inference import TensorSpec, _resolve_view_shape


def _resolve(detail, source, *, named=None, snapshots=(), dims=None):
    return _resolve_view_shape(
        detail,
        TensorSpec(shape=source, dtype="float16"),
        dims or {},
        named,
        frozenset(snapshots),
    )


class TestTheFrameInputIsReadWhenItWasMeasuredEarlier:
    def test_the_leading_axes_come_from_the_named_tensor(self) -> None:
        resolved = _resolve(
            "*x.shape[:-1], self.nf",
            ("B*S", 2304),
            named={"x": ("B", "S", 768)},
            snapshots=("x.shape[:-1]",),
        )
        assert resolved == ("B", "S", 2304), resolved

    def test_the_trailing_axis_is_settled_by_conservation(self) -> None:
        """``self.nf`` is a constructor argument; no config states it."""
        resolved = _resolve(
            "*x.shape[:-1], self.nf",
            ("B*S", 3072),
            named={"x": ("B", "S", 768)},
            snapshots=("x.shape[:-1]",),
        )
        assert resolved == ("B", "S", 3072), resolved


class TestWhatMustStillReadTheSource:
    def test_an_inline_star_names_the_tensor_being_reshaped(self) -> None:
        """Not a snapshot, so the named tensor is NOT consulted."""
        resolved = _resolve(
            "*x.shape[:-1], 128",
            ("B", "S", 8192),
            named={"x": ("B", "S", 64, 128)},
        )
        assert resolved == ("B", "S", 128), resolved

    def test_an_unresolvable_axis_without_a_named_prefix_gives_up(self) -> None:
        """Returning None leaves a fallback that is right; guessing does not."""
        assert _resolve("*x.shape[:-1], self.nf", ("B", "S", 4096)) is None

    def test_a_name_that_is_not_known_gives_up(self) -> None:
        resolved = _resolve(
            "*other.shape[:-1], self.nf",
            ("B*S", 2304),
            named={"x": ("B", "S", 768)},
            snapshots=("other.shape[:-1]",),
        )
        assert resolved is None


class TestTorchSaysWhatPreservesShape:
    """``nn.Dropout`` has no symbolic rule and cannot run on ``[B, S, 768]``.

    Substituting a size per symbol answers the only question that matters --
    does the output have the same shape as the input -- and the real symbolic
    shape is then carried through. The stand-ins are distinct so an op that
    permutes or flattens cannot look shape-preserving by coincidence.
    """

    def test_a_dropout_keeps_its_symbolic_shape(self) -> None:
        import pytest

        pytest.importorskip("torch")
        from TraceLens.ModelUtils.shape_inference import _neutral_scalar_args

        import torch

        args = _neutral_scalar_args(torch, "dropout")
        assert [name for name, _ in args] == ["p", "train"], args
        assert dict(args)["train"] is False

    def test_an_op_the_schema_cannot_describe_yields_nothing(self) -> None:
        import pytest

        pytest.importorskip("torch")
        from TraceLens.ModelUtils.shape_inference import _neutral_scalar_args

        import torch

        assert _neutral_scalar_args(torch, "not_a_real_aten_op") == []
