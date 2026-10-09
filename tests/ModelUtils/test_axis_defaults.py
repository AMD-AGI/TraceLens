###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Leaving an axis argument out is a statement about the axis, not a blank.

``torch.cat([a, b])`` joins along axis 0 and ``x.topk(k)`` narrows the last one.
Both rules guessed the TRAILING axis whenever no ``dim`` was recorded, which is
right for ``topk`` and wrong for ``cat`` -- the one op where omitting the
argument changes the answer. And ``k`` narrows whichever axis ``dim`` names: a
gate scoring expert GROUPS takes its top few along axis 1, and narrowing the
last axis instead reported a width that tensor never had.
"""

from __future__ import annotations

from TraceLens.ModelUtils.extract import ArchitectureSpec
from TraceLens.ModelUtils.model_graph import ModelGraphNode, NodeKind, OperationKind
from TraceLens.ModelUtils.shape_inference import (
    ShapeContext,
    ShapeInferencer,
    TensorSpec,
)


def _inferencer() -> ShapeInferencer:
    spec = ArchitectureSpec(
        name="Test",
        model_type="test",
        hidden_size=4096,
        raw_config={"hidden_size": 4096},
    )
    return ShapeInferencer(spec, context=ShapeContext.from_spec(spec))


def _infer(label: str, details: list[str], inputs: list[TensorSpec]) -> TensorSpec:
    node = ModelGraphNode(
        id=f"@op_{label}",
        label=label,
        kind=NodeKind.LEAF,
        operation=OperationKind.TORCH_FUNCTIONAL,
        metadata={"details": details},
    )
    return _inferencer()._infer_node_output(node, inputs, root=None)


class TestConcatDefaultAxis:
    def test_no_dim_joins_along_axis_zero(self) -> None:
        operand = TensorSpec(("B", "S", 8), "float32")
        out = _infer("concat", [], [operand, operand])
        assert out.shape == ("B + B", "S", 8), out.shape

    def test_an_explicit_axis_still_wins(self) -> None:
        operand = TensorSpec(("B", "S", 8), "float32")
        assert _infer("concat", ["dim: -1"], [operand, operand]).shape == (
            "B",
            "S",
            16,
        )

    def test_stack_already_agreed_on_axis_zero(self) -> None:
        operand = TensorSpec(("B", "S", 8), "float32")
        assert _infer("stack", [], [operand, operand]).shape == (2, "B", "S", 8)

    def test_an_axis_named_but_not_a_literal_keeps_the_trailing_guess(self) -> None:
        """``cat(parts, dim=d)`` leaves the axis unknown, not defaulted."""
        operand = TensorSpec(("B", "S", 8), "float32")
        assert _infer("concat", ["dim: d"], [operand, operand]).shape == (
            "B",
            "S",
            16,
        )


class TestTopkAxis:
    def test_k_narrows_the_axis_dim_names(self) -> None:
        out = _infer("topk", ["k: 2", "dim: 1"], [TensorSpec(("B", "S", 257), "f32")])
        assert out.shape == ("B", 2, 257), out.shape

    def test_the_trailing_axis_remains_the_default(self) -> None:
        out = _infer("topk", ["k: 2"], [TensorSpec(("B", "S", 257), "f32")])
        assert out.shape == ("B", "S", 2), out.shape

    def test_the_negative_spelling_names_the_same_axis(self) -> None:
        out = _infer("topk", ["k: 2", "dim: -1"], [TensorSpec(("B", "S", 257), "f32")])
        assert out.shape == ("B", "S", 2), out.shape

    def test_an_axis_beyond_the_rank_falls_back(self) -> None:
        out = _infer("topk", ["k: 2", "dim: 9"], [TensorSpec(("B", "S", 257), "f32")])
        assert out.shape == ("B", "S", 2), out.shape
