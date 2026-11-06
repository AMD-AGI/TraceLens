###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A loop body resolves its shapes after the boundary that feeds it.

GLM's mixture-of-experts accumulates into one tensor across its iterations::

    final = torch.zeros_like(hidden_states)     # [B*S, 4096] bfloat16
    for expert_idx in hit:
        top_k_pos, token_idx = torch.where(mask[expert_idx])
        ...
        final.index_add_(0, token_idx, current.to(final.dtype))
    return final

``index_add_`` returns the tensor it writes INTO, and there is a rule saying so
-- it anchors on operand 0. The operand 0 it was given was ``token_idx``, so
``final`` left the loop as ``[nnz, 1] int64``: the shape of the INDEX, carried
out through the loop boundary and into the ``.view(*orig_shape)`` that follows,
where a view was recorded reading int64 and producing bfloat16.

The receiver was never missing from the graph. Its edge was there, first in
order, from the loop's carried-in boundary -- but that boundary had no shape
yet, and an operand with no shape is dropped rather than held open, so every
later operand shifted down one place and the index became operand 0.

It had no shape because the two carried-value boundaries are joined by a "next
iteration" edge. Counted as an ordering constraint it closes a cycle that a
topological sort cannot resolve, so everything in the loop fell out of the sort
and was appended in whatever order the nodes happened to be in -- here, the body
before the boundary feeding it. The edge is a real dependency and the graph
keeps it; it just does not order anything, because what the body reads on its
first iteration is the seed, which is ordered ahead of it already.
"""

from __future__ import annotations

from TraceLens.ModelUtils.model_graph import (
    GraphEdge,
    ModelGraph,
    ModelGraphNode,
    NodeKind,
)
from TraceLens.ModelUtils.shape_inference import _topological_order


def _node(node_id: str, synthetic: str | None = None) -> ModelGraphNode:
    metadata = {"synthetic": synthetic} if synthetic else {}
    return ModelGraphNode(
        id=node_id, kind=NodeKind.LEAF, label=node_id, metadata=metadata
    )


def _accumulating_loop() -> ModelGraph:
    """seed -> loop in -> body -> loop out, with the back edge that closed it."""
    return ModelGraph(
        title="loop",
        nodes=[
            _node("seed"),
            _node("loop_in", synthetic="@loop_carried"),
            _node("body"),
            _node("loop_out", synthetic="@loop_carried"),
            _node("after"),
        ],
        edges=[
            GraphEdge(source="seed", target="loop_in"),
            GraphEdge(source="loop_in", target="body"),
            GraphEdge(source="body", target="loop_out"),
            GraphEdge(source="loop_out", target="loop_in"),  # next iteration
            GraphEdge(source="loop_out", target="after"),
        ],
    )


class TestTheBoundaryIsResolvedFirst:
    def test_the_body_comes_after_what_feeds_it(self) -> None:
        order = _topological_order(_accumulating_loop())
        assert order.index("loop_in") < order.index("body")

    def test_the_whole_loop_is_ordered(self) -> None:
        """Nothing falls out of the sort to be appended in arbitrary order."""
        order = _topological_order(_accumulating_loop())
        assert order.index("seed") < order.index("loop_in")
        assert order.index("body") < order.index("loop_out")
        assert order.index("loop_out") < order.index("after")

    def test_every_node_is_still_listed_once(self) -> None:
        order = _topological_order(_accumulating_loop())
        assert sorted(order) == ["after", "body", "loop_in", "loop_out", "seed"]


class TestWhatIsStillOrdered:
    def test_an_ordinary_edge_between_two_ops(self) -> None:
        graph = ModelGraph(
            title="chain",
            nodes=[_node("a"), _node("b")],
            edges=[GraphEdge(source="a", target="b")],
        )
        assert _topological_order(graph) == ["a", "b"]

    def test_the_seed_edge_into_a_carried_boundary(self) -> None:
        """Only the edge with a carried tile at BOTH ends is set aside."""
        graph = ModelGraph(
            title="seeded",
            nodes=[_node("seed"), _node("loop_in", synthetic="@loop_carried")],
            edges=[GraphEdge(source="seed", target="loop_in")],
        )
        assert _topological_order(graph) == ["seed", "loop_in"]

    def test_a_cycle_of_real_ops_still_yields_every_node(self) -> None:
        """Breaking one loop edge is not a licence to lose nodes elsewhere."""
        graph = ModelGraph(
            title="cyclic",
            nodes=[_node("a"), _node("b")],
            edges=[
                GraphEdge(source="a", target="b"),
                GraphEdge(source="b", target="a"),
            ],
        )
        assert sorted(_topological_order(graph)) == ["a", "b"]
