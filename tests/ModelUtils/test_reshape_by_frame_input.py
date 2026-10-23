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

from TraceLens.ModelUtils.extract import ArchitectureSpec
from TraceLens.ModelUtils.model_graph import ModelGraphNode, NodeKind, OperationKind
from TraceLens.ModelUtils.shape_inference import (
    ShapeContext,
    ShapeInferencer,
    TensorSpec,
    _resolve_expand_shape,
    _resolve_view_shape,
)


def _resolve(detail, source, *, named=None, snapshots=(), dims=None):
    return _resolve_view_shape(
        detail,
        TensorSpec(shape=source, dtype="float16"),
        dims or {},
        named,
        frozenset(snapshots),
    )


def test_expand_resolves_extents_traced_to_shape_reads():
    """Broadcast sizes come from the shape reads the source bound those names to."""
    result = _resolve_expand_shape(
        "batch_size, -1, q_length, kv_length",
        TensorSpec(("S",), "bool"),
        {},
        {"inputs_embeds": ("B", "S", "H"), "attention_mask": ("B", "S")},
        {
            "batch_size": "inputs_embeds.shape[0]",
            "q_length": "inputs_embeds.shape[1]",
            "kv_length": "inputs_embeds.shape[1]|attention_mask.shape[-1]",
        },
    )
    assert result == ("B", 1, "S", "S")


def test_expand_leaves_an_untraced_extent_name_unsized():
    """A bare ``batch_size`` stays a broadcast of the source until a shape read names its axis."""
    result = _resolve_expand_shape(
        "batch_size, -1, q_length, kv_length",
        TensorSpec(("S",), "bool"),
        {},
    )
    assert result == (1, 1, 1, "S")


def test_expand_resolves_the_shape_string_a_mask_builder_emits():
    """Single-read sizes are substituted; a size with two reads stays a name.

    That is the string ``sdpa_mask``'s expand carries once the builder has been
    traced: batch and query each come from one axis of ``inputs_embeds``, and
    the key length is the two reads the cold path can still reach.
    """
    result = _resolve_expand_shape(
        "inputs_embeds.shape[0], -1, inputs_embeds.shape[1], kv_length",
        TensorSpec(("S",), "bool"),
        {},
        {"inputs_embeds": ("B", "S", "H"), "attention_mask": ("B", "S")},
        {"kv_length": "inputs_embeds.shape[1]|attention_mask.shape[-1]"},
    )
    assert result == ("B", 1, "S", "S")


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


def test_an_arithmetic_product_is_one_reshape_axis():
    """``reshape(t, h * w, s, s)`` states four axes even though the source has five.

    A ``*`` inside an axis is multiplication, not ``*x.shape``. Treating every
    star as an unpack skipped the rank the call states and published the
    source shape instead.
    """
    spec = ArchitectureSpec(
        name="Test", model_type="test", hidden_size=32, raw_config={"hidden_size": 32}
    )
    inferred = ShapeInferencer(spec, context=ShapeContext.from_spec(spec))
    node = ModelGraphNode(
        id="@op_reshape",
        label="reshape",
        kind=NodeKind.LEAF,
        operation=OperationKind.TORCH_FUNCTIONAL,
        metadata={
            "details": [
                "shape: grid_t, num_windows_h * num_windows_w, "
                "vit_merger_window_size, vit_merger_window_size"
            ]
        },
    )
    source = TensorSpec(
        shape=(
            "grid_t",
            "num_windows_h",
            "num_windows_w",
            "vit_merger_window_size",
            "vit_merger_window_size",
        ),
        dtype="int64",
    )
    out = inferred._infer_node_output(node, [source], root=None)
    assert out is not None
    assert out.shape == (
        "grid_t",
        "num_windows_h * num_windows_w",
        "vit_merger_window_size",
        "vit_merger_window_size",
    )
