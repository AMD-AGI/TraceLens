###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An index sits on the axis it was written on.

``x[idx]`` and ``x[:, idx]`` arrive downstream as the same thing -- a base and
one index tensor -- so nothing after the parse can tell them apart. The emit
site records ``index_axes`` precisely so the shape rule can place the selection
instead of assuming it lands on the leading axis.

The case this actually fixes is the boolean mask. ``torch`` cannot be asked for
a mask's result shape from dtypes alone: a mask consumes as many axes as it has
and leaves ONE data-dependent length behind. Before this, masks were not
recognised as indices at all (only ``int64`` operands were), so a masked
subscript fell through to a fallback that replaces the last dim with
``EXPERTS_PER_TOK`` -- a MoE-routing guess, which on GLM's attention indexer
produced ``[batch_size, number_of_pools, 8]`` for a tensor whose real answer is
``[batch_size, nnz, 128]``: it had substituted the top-k constant for the head
dim.

What is deliberately NOT changed here: a leading-axis INTEGER index keeps the
answer it has always had. See ``_placed_advanced_index`` for the measurement --
answering it faithfully propagates a wrong index shape out of the MoE expert
loop through an entire MLP.
"""

from __future__ import annotations

import ast

from TraceLens.ModelUtils.ast_analyze import _subscript_index_axes
from TraceLens.ModelUtils.extract import ArchitectureSpec
from TraceLens.ModelUtils.model_graph import ModelGraphNode, NodeKind, OperationKind
from TraceLens.ModelUtils.shape_inference import (
    ShapeContext,
    ShapeInferencer,
    TensorSpec,
)


def _make_inferencer(**dims: int) -> ShapeInferencer:
    spec = ArchitectureSpec(
        name="Test",
        model_type="test",
        hidden_size=dims.get("hidden_size", 4096),
        raw_config={"hidden_size": dims.get("hidden_size", 4096)},
    )
    context = ShapeContext.from_spec(spec)
    for key, value in dims.items():
        context.dims[key] = value
    return ShapeInferencer(spec, context=context)


def _node(label: str, *, details: list[str] | None = None) -> ModelGraphNode:
    meta: dict = {"class_name": label}
    if details:
        meta["details"] = details
    return ModelGraphNode(
        id="n1",
        kind=NodeKind.LEAF,
        label=label,
        operation=OperationKind.TORCH_FUNCTIONAL,
        metadata=meta,
    )


def _axes(expression: str) -> list[int]:
    """The recorded axes for the subscript in *expression*."""
    subscript = ast.parse(expression, mode="eval").body
    assert isinstance(subscript, ast.Subscript)
    return [axis for axis, _operand in _subscript_index_axes(subscript.slice)]


def _infer(inf, node, inputs, positions):
    """Infer with *positions* naming each operand's own argument slot.

    ``inputs`` has had unresolved operands dropped by the time a rule sees it,
    so the real pipeline records which argument each survivor came from. A test
    that skipped this would be exercising a different function.
    """
    inf._last_input_positions = list(positions)
    return inf._infer_node_output(node, inputs, root=None)


class TestWhichAxisWasWritten:
    def test_a_bare_index(self) -> None:
        assert _axes("x[idx]") == [0]

    def test_an_index_after_a_slice(self) -> None:
        assert _axes("x[:, idx]") == [1]

    def test_two_indices_side_by_side(self) -> None:
        assert _axes("x[i, j]") == [0, 1]

    def test_slices_and_ints_are_not_indices(self) -> None:
        assert _axes("x[:, 0, idx]") == [2]

    def test_after_an_ellipsis_the_axis_counts_from_the_back(self) -> None:
        """``x[..., idx]`` indexes the LAST axis whatever the rank is."""
        assert _axes("x[..., idx]") == [-1]
        assert _axes("x[..., idx, :]") == [-2]

    def test_a_subscript_with_no_tensor_index(self) -> None:
        assert _axes("x[:, 1:3]") == []


class TestABooleanMaskLeavesOneDataDependentLength:
    def test_a_mask_on_a_middle_axis(self) -> None:
        """``x[:, mask]`` keeps the axes either side and collapses the masked one."""
        inf = _make_inferencer()
        base = TensorSpec(
            shape=("batch_size", "number_of_pools", 128), dtype="bfloat16"
        )
        mask = TensorSpec(shape=("number_of_pools",), dtype="bool")
        result = _infer(
            inf, _node("Indexing", details=["index_axes: 1"]), [base, mask], [0, 1]
        )
        assert result.shape == ("batch_size", "nnz", 128)
        assert result.dtype == "bfloat16"

    def test_a_mask_on_the_leading_axis(self) -> None:
        inf = _make_inferencer()
        base = TensorSpec(shape=("N", 4096), dtype="float16")
        mask = TensorSpec(shape=("N",), dtype="bool")
        result = _infer(
            inf, _node("Indexing", details=["index_axes: 0"]), [base, mask], [0, 1]
        )
        assert result.shape == ("nnz", 4096)

    def test_a_mask_counted_from_the_back(self) -> None:
        inf = _make_inferencer()
        base = TensorSpec(shape=("B", "S", 64), dtype="float32")
        mask = TensorSpec(shape=(64,), dtype="bool")
        result = _infer(
            inf, _node("Indexing", details=["index_axes: -1"]), [base, mask], [0, 1]
        )
        assert result.shape == ("B", "S", "nnz")

    def test_a_rank_2_mask_consumes_two_axes(self) -> None:
        """A mask is not an index of one axis -- it eats as many as it has."""
        inf = _make_inferencer()
        base = TensorSpec(shape=("B", "S", 128), dtype="bfloat16")
        mask = TensorSpec(shape=("B", "S"), dtype="bool")
        result = _infer(
            inf, _node("Indexing", details=["index_axes: 0"]), [base, mask], [0, 1]
        )
        assert result.shape == ("nnz", 128)


class TestAnIntegerIndexOffTheLeadingAxis:
    def test_it_is_placed_where_it_was_written(self) -> None:
        inf = _make_inferencer()
        base = TensorSpec(shape=("B", "P", 128), dtype="bfloat16")
        index = TensorSpec(shape=(4,), dtype="int64")
        result = _infer(
            inf, _node("Indexing", details=["index_axes: 1"]), [base, index], [0, 1]
        )
        assert result.shape == ("B", 4, 128)


class TestALeadingIntegerIndex:
    def test_it_is_placed_like_any_other(self) -> None:
        """Selecting rows keeps the columns -- the precondition now holds.

        This case was DECLINED when the rule first landed, because the only
        leading-axis index in any model was the MoE expert loop's, and that
        index was itself wrong: it carried the whole routing table, so placing
        it faithfully pushed a wrong shape through an entire MLP. The loop now
        hands over its own expert's tokens, so the index is right and the
        correct answer is an improvement rather than a louder bug.
        """
        inf = _make_inferencer()
        base = TensorSpec(shape=("nnz", 4096), dtype="bfloat16")
        index = TensorSpec(shape=("nnz",), dtype="int64")
        result = _infer(
            inf, _node("Indexing", details=["index_axes: 0"]), [base, index], [0, 1]
        )
        assert result.shape == ("nnz", 4096)


class TestWhenItDeclinesAndLeavesTheOldAnswerAlone:

    def test_more_operands_wired_than_axes_recorded(self) -> None:
        """Three tensors wired against one axis: the structure is not known."""
        inf = _make_inferencer()
        base = TensorSpec(shape=("nnz", 1), dtype="int64")
        index = TensorSpec(shape=(288, 8, "B*S"), dtype="int64")
        extra = TensorSpec(shape=(1,), dtype="int64")
        node = _node("Indexing", details=["index_axes: 1"])
        result = _infer(inf, node, [base, index, extra], [0, 1, 2])
        assert result.shape != ("nnz", 288, 8, "B*S", 1)

    def test_indices_separated_by_a_slice(self) -> None:
        """``x[i, :, j]`` moves the selection to the FRONT; not modelled."""
        inf = _make_inferencer()
        base = TensorSpec(shape=("A", "B", "C"), dtype="float32")
        i = TensorSpec(shape=(4,), dtype="int64")
        j = TensorSpec(shape=(4,), dtype="int64")
        node = _node("Indexing", details=["index_axes: 0,2"])
        result = _infer(inf, node, [base, i, j], [0, 1, 2])
        assert result.shape != ("A", 4, "C")

    def test_an_axis_outside_the_bases_rank(self) -> None:
        inf = _make_inferencer()
        base = TensorSpec(shape=("B", "S"), dtype="float32")
        mask = TensorSpec(shape=("S",), dtype="bool")
        node = _node("Indexing", details=["index_axes: 7"])
        assert _infer(inf, node, [base, mask], [0, 1]).shape != ("B", "S", "nnz")

    def test_no_recorded_axes_at_all(self) -> None:
        """Emitted without a base, so the operands do not line up: old answer."""
        inf = _make_inferencer()
        base = TensorSpec(shape=("B", "P", 128), dtype="bfloat16")
        index = TensorSpec(shape=(4,), dtype="int64")
        result = _infer(inf, _node("Indexing"), [base, index], [0, 1])
        assert result.shape == (4,)

    def test_a_real_torch_gather_is_untouched(self) -> None:
        """``Gather`` never reaches this rule -- its output IS its index's shape."""
        inf = _make_inferencer()
        base = TensorSpec(shape=("B", "S", 4096), dtype="float16")
        index = TensorSpec(shape=("B", "S", 8), dtype="int64")
        node = _node("Gather", details=["dim: -1", "index_axes: 1"])
        result = _infer(inf, node, [base, index], [0, 1])
        assert result.shape == ("B", "S", 8)
