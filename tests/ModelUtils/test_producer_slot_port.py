###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Which port a return slot reads on the op that produces it.

A tuple slot resolved to its own producer usually reads port ``"0"`` -- that
producer has one output, and the CALLER's tuple ordinal names a port on the call,
not on the op, so carrying it over would dangle.

That assumption fails when the producer is itself a multi-output op. GLM's
``expand_kv`` does ``k_nope, value_states = kv.split(...)`` and then ``return
key_states, value_states``, so the split is recorded as ``value_states``'s
producer while being its SECOND slice. Reading port ``"0"`` published ``k_nope``
under the name ``value_states``, and the slice the model really returns was left
with no consumer at all -- a dead node carrying live computation.

An op names its own unpack targets in source order, index-aligned to its ports,
so the slot's own name picks the port out of them.
"""

from __future__ import annotations

from TraceLens.ModelUtils.block_tree import BlockNode
from TraceLens.ModelUtils.computation_graph import (
    ComputationGraph,
    GraphNodeSpec,
    _producer_slot_port,
)


def _graph(*blocks):
    graph = ComputationGraph()
    graph.nodes = [
        GraphNodeSpec(key=f"n{i}", block=block) for i, block in enumerate(blocks)
    ]
    return graph


def _op(attr_name, outputs=()):
    return BlockNode(
        attr_name=attr_name,
        class_name="Op",
        role="operation",
        label=attr_name,
        output_names=list(outputs),
    )


class TestAMultiOutputProducer:
    def test_the_slot_name_picks_its_slice(self) -> None:
        graph = _graph(_op("split", ("k_nope", "value_states")))
        assert _producer_slot_port(graph, 0, "value_states") == "1"

    def test_the_first_slice_too(self) -> None:
        graph = _graph(_op("split", ("k_nope", "value_states")))
        assert _producer_slot_port(graph, 0, "k_nope") == "0"

    def test_a_three_way_unbind(self) -> None:
        graph = _graph(_op("unbind", ("pre", "post", "comb")))
        assert _producer_slot_port(graph, 0, "comb") == "2"


class TestWhenItKeepsTheDefault:
    def test_a_single_output_op(self) -> None:
        """The ordinary case: one output, so the slot reads port 0."""
        graph = _graph(_op("copy", ("key_states",)))
        assert _producer_slot_port(graph, 0, "key_states") == "0"

    def test_an_op_that_names_no_outputs(self) -> None:
        graph = _graph(_op("matmul"))
        assert _producer_slot_port(graph, 0, "attn_output") == "0"

    def test_a_slot_the_producer_does_not_name(self) -> None:
        """Not one of this op's unpack targets, so nothing here selects a port."""
        graph = _graph(_op("split", ("a", "b")))
        assert _producer_slot_port(graph, 0, "c") == "0"

    def test_a_node_index_outside_the_graph(self) -> None:
        graph = _graph(_op("split", ("a", "b")))
        assert _producer_slot_port(graph, 7, "a") == "0"
        assert _producer_slot_port(graph, -1, "a") == "0"

    def test_a_node_with_no_block(self) -> None:
        graph = ComputationGraph()
        graph.nodes = [GraphNodeSpec(key="synthetic", block=None)]
        assert _producer_slot_port(graph, 0, "a") == "0"

    def test_the_caller_chooses_the_default(self) -> None:
        """An empty default lets a caller tell "no port" from port zero."""
        graph = _graph(_op("matmul"))
        assert _producer_slot_port(graph, 0, "x", default="") == ""
