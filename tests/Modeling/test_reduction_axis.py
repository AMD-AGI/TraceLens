###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A reduction over an axis the tensor HAS drops that axis.

The rule reduced only when ``dim`` was ``-1`` and returned the source unchanged
for any positive ``dim``, on the grounds that it might name a head or stream axis
the symbolic ``(B, S, H)`` view omits. That is true of an axis beyond the shape's
rank, and false of one inside it -- so ``dim: 2`` and ``dim: -1`` gave different
answers about the very same axis of a rank-3 tensor.

``DeepseekV4IndexerScorer`` returns ``(scores * weights.unsqueeze(-1)).sum(dim=2)``,
documented ``[B, S, T]``. With the reduction skipped it published ``[B, S, H, T]``
-- the tensor from before the sum -- and the indexer, the compressor's mask and
the scatter reading them all inherited the extra axis.
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


def _node(label: str, details: list[str]) -> ModelGraphNode:
    return ModelGraphNode(
        id=f"@op_{label}",
        label=label,
        kind=NodeKind.LEAF,
        operation=OperationKind.TORCH_FUNCTIONAL,
        metadata={"details": details},
    )


class TestReductionAxis:
    def test_a_positive_axis_inside_the_rank_is_reduced(self) -> None:
        out = _inferencer()._infer_node_output(
            _node("sum", ["dim: 2"]),
            [TensorSpec(("B", "S", 8, 4), "float32")],
            root=None,
        )
        assert out.shape == ("B", "S", 4), out.shape

    def test_it_agrees_with_the_negative_spelling_of_the_same_axis(self) -> None:
        inf = _inferencer()
        source = [TensorSpec(("B", "S", 8), "float32")]
        last = inf._infer_node_output(_node("sum", ["dim: -1"]), source, root=None)
        same = inf._infer_node_output(_node("sum", ["dim: 2"]), source, root=None)
        assert last.shape == same.shape == ("B", "S"), (last.shape, same.shape)

    def test_keepdim_collapses_instead_of_dropping(self) -> None:
        out = _inferencer()._infer_node_output(
            _node("sum", ["dim: 1", "keepdim: True"]),
            [TensorSpec(("B", "S", 8), "float32")],
            root=None,
        )
        assert out.shape == ("B", 1, 8), out.shape

    def test_an_axis_beyond_the_rank_is_left_alone(self) -> None:
        """A head or stream axis the symbolic view omits cannot be dropped."""
        out = _inferencer()._infer_node_output(
            _node("sum", ["dim: 9"]),
            [TensorSpec(("B", "S", 8), "float32")],
            root=None,
        )
        assert out.shape == ("B", "S", 8), out.shape

    def test_every_axis_a_tuple_names_is_reduced(self) -> None:
        """``expert_mask.sum(dim=(-1, -2))`` -- which every MoE router runs.

        Reducing only the first axis of the tuple, or none of them, published the
        tensor from before the reduction, and the ``greater``/``nonzero`` reading
        it counted two axes that were already summed away.
        """
        out = _inferencer()._infer_node_output(
            _node("sum", ["dim: (-1, -2)"]),
            [TensorSpec(("B", "S", 8, 4), "float32")],
            root=None,
        )
        assert out.shape == ("B", "S"), out.shape

    def test_a_tuple_of_axes_honours_keepdim(self) -> None:
        out = _inferencer()._infer_node_output(
            _node("sum", ["dim: (1, 2)", "keepdim: True"]),
            [TensorSpec(("B", "S", 8), "float32")],
            root=None,
        )
        assert out.shape == ("B", 1, 1), out.shape

    def test_a_list_spells_the_same_thing(self) -> None:
        out = _inferencer()._infer_node_output(
            _node("sum", ["dim: [1, 2]"]),
            [TensorSpec(("B", "S", 8), "float32")],
            root=None,
        )
        assert out.shape == ("B",), out.shape

    def test_an_axis_the_view_omits_is_skipped_among_others(self) -> None:
        out = _inferencer()._infer_node_output(
            _node("sum", ["dim: (1, 9)"]),
            [TensorSpec(("B", "S", 8), "float32")],
            root=None,
        )
        assert out.shape == ("B", 8), out.shape


class TestPredicateReduction:
    """``any``/``all`` reduce an axis and answer bool, like any other reduction."""

    def test_any_drops_its_axis_and_answers_bool(self) -> None:
        out = _inferencer()._infer_node_output(
            _node("any", ["dim: -1"]),
            [TensorSpec(("B", "S", 257), "float32")],
            root=None,
        )
        assert out.shape == ("B", "S"), out.shape
        assert out.dtype == "bool", out.dtype

    def test_any_keeps_the_axis_when_asked(self) -> None:
        out = _inferencer()._infer_node_output(
            _node("any", ["dim: -1", "keepdim: True"]),
            [TensorSpec(("B", "S", 257), "float32")],
            root=None,
        )
        assert out.shape == ("B", "S", 1), out.shape
        assert out.dtype == "bool", out.dtype

    def test_all_reduces_a_named_inner_axis(self) -> None:
        out = _inferencer()._infer_node_output(
            _node("all", ["dim: 1", "keepdim: True"]),
            [TensorSpec(("B", "S", 257), "float32")],
            root=None,
        )
        assert out.shape == ("B", 1, 257), out.shape
        assert out.dtype == "bool", out.dtype

    def test_an_index_reduction_still_answers_int64(self) -> None:
        out = _inferencer()._infer_node_output(
            _node("argmax", ["dim: -1", "keepdim: True"]),
            [TensorSpec(("B", "S", 257), "float32")],
            root=None,
        )
        assert out.shape == ("B", "S", 1), out.shape
        assert out.dtype == "int64", out.dtype
