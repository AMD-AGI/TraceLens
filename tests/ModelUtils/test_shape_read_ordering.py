###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Reading a tensor's SIZE is a dependency, even with no edge to carry it.

``F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]))`` needs
``kv``'s extent, not its data -- so nothing wires the two together and the
topological sort was free to infer the pad first, with ``kv`` not yet known.
The pad then refused the width and reported its input unchanged.

The op declares the read in its own details, so the constraint is derived from
what the op says it reads. It belongs to inference only: in the emitted graph
``kv`` is not one of the pad's operands, and drawing it as one would state a
tensor flow that does not exist.
"""

from __future__ import annotations

from types import SimpleNamespace

from TraceLens.ModelUtils.model_graph import (
    GraphEdge,
    ModelGraph,
    ModelGraphNode,
    NodeKind,
)
from TraceLens.ModelUtils.shape_inference import ShapeInferencer, _topological_order


def _step(index, attr, details=()):
    """A bare operation step, which names itself on both sides as a real one does."""
    return ModelGraphNode(
        id=f"seq:{index}:{attr}:{attr}:0",
        kind=NodeKind.LEAF,
        label="Op",
        metadata={"details": list(details), "attr_name": attr},
    )


def _id(index, attr):
    return f"seq:{index}:{attr}:{attr}:0"


CONCAT = _id(4, "@op_l823_c17_concat")
PAD = _id(7, "@op_l835_c33_pad")
MASK = _id(2, "@op_l800_c1_view")

# The hazard in miniature: the pad's own operand is ready immediately, while
# the concat is still waiting on its producer. Nothing connects the two, so a
# sort that only follows edges reaches the pad first -- and `kv`, which the
# concat makes, does not exist yet.
GRAPH = ModelGraph(
    title="attention",
    nodes=[
        _step(2, "@op_l800_c1_view"),
        _step(
            7, "@op_l835_c33_pad", ["pad: 0, kv.shape[2] - attention_mask.shape[-1]"]
        ),
        _step(4, "@op_l823_c17_concat"),
    ],
    edges=[GraphEdge(source=MASK, target=CONCAT)],
)

STRUCTURE = SimpleNamespace(
    forward_input_name="hidden_states",
    forward_var_bindings={"kv": [(823, "@op_l823_c17_concat")]},
)


def _inferencer():
    self = ShapeInferencer.__new__(ShapeInferencer)
    self.spec = SimpleNamespace(class_registry={"Attention": STRUCTURE})
    self._owner_classes = {}
    self._owner_class_refs = []
    return self


ROOT = SimpleNamespace(attr_name=None, class_name="Attention", children=[])


class TestTheConstraintADetailDeclares:
    def test_the_producer_of_the_read_name_is_named(self) -> None:
        assert _inferencer()._shape_read_order(GRAPH, ROOT) == [(CONCAT, PAD)]

    def test_without_the_constraint_the_pad_may_come_first(self) -> None:
        order = _topological_order(GRAPH)
        assert order.index(PAD) < order.index(CONCAT)

    def test_with_it_the_producer_comes_first(self) -> None:
        order = _topological_order(GRAPH, [(CONCAT, PAD)])
        assert order.index(CONCAT) < order.index(PAD)

    def test_every_node_is_still_ordered(self) -> None:
        assert sorted(_topological_order(GRAPH, [(CONCAT, PAD)])) == sorted(
            node.id for node in GRAPH.nodes
        )

    def test_a_read_of_a_name_with_no_binding_constrains_nothing(self) -> None:
        graph = ModelGraph(
            title="attention",
            nodes=[_step(7, "@op_l835_c33_pad", ["pad: 0, absent.shape[2]"])],
            edges=[],
        )
        assert _inferencer()._shape_read_order(graph, ROOT) == []

    def test_a_constraint_that_would_close_a_cycle_is_still_complete(self) -> None:
        # The fallback matters more than program order: an order that drops
        # nodes would leave them with no shape at all.
        order = _topological_order(GRAPH, [(CONCAT, PAD), (PAD, CONCAT)])
        assert sorted(order) == sorted(node.id for node in GRAPH.nodes)
