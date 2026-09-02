###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for computation-graph dead code elimination."""

from __future__ import annotations

from visualizer.block_tree import BlockNode
from visualizer.computation_graph import (
    ComputationGraph,
    GraphNodeSpec,
    SYNTHETIC_INPUT,
    _apply_dead_code_elimination,
    _dead_node_indices,
    _live_node_indices_to_fixpoint,
    _strip_dead_nodes,
)


def _block(attr_name: str, *, label: str | None = None) -> BlockNode:
    return BlockNode(
        attr_name=attr_name,
        class_name=label or attr_name,
        role="other",
        label=label or attr_name,
    )


def test_live_node_fixpoint_closes_long_predecessor_chains():
    graph = ComputationGraph(
        nodes=[
            GraphNodeSpec(key="n0", block=_block("out")),
            GraphNodeSpec(key="n1", block=_block("s1")),
            GraphNodeSpec(key="n2", block=_block("s2")),
            GraphNodeSpec(key="n3", block=_block("s3")),
        ],
        links=[(3, 2), (2, 1), (1, 0)],
        primary_output_index=0,
    )

    live = _live_node_indices_to_fixpoint(graph, [0])

    assert live == {0, 1, 2, 3}


def test_live_node_fixpoint_marks_cyclic_dead_components():
    graph = ComputationGraph(
        nodes=[
            GraphNodeSpec(key="live", block=_block("live")),
            GraphNodeSpec(key="spine", block=_block("spine")),
            GraphNodeSpec(key="a", block=_block("a")),
            GraphNodeSpec(key="b", block=_block("b")),
        ],
        links=[(1, 0), (2, 3), (3, 2)],
        primary_output_index=0,
    )

    live = _live_node_indices_to_fixpoint(graph, [0])

    assert live == {0, 1}


def test_dead_code_elimination_removes_isolated_dead_chains():
    root = BlockNode(
        attr_name="hc",
        class_name="HyperConnection",
        role="other",
        label="HC",
        primary_output_step="collapsed",
        multi_return_module=True,
    )
    graph = ComputationGraph(
        nodes=[
            GraphNodeSpec(key="input", label="input", synthetic=SYNTHETIC_INPUT),
            GraphNodeSpec(key="collapsed", block=_block("collapsed")),
            GraphNodeSpec(key="live", block=_block("live")),
            GraphNodeSpec(key="dead_head", block=_block("post")),
            GraphNodeSpec(key="dead_tail", block=_block("tail")),
        ],
        links=[
            (2, 1),
            (1, 0),
            (4, 3),
        ],
        primary_output_index=1,
    )

    first_dead = _dead_node_indices(
        graph,
        root,
        strip_unused_return_branches=True,
    )
    assert first_dead == {3, 4}

    graph.dead_node_indices = first_dead
    once = _strip_dead_nodes(graph)
    assert len(once.nodes) == 3
    assert {node.key for node in once.nodes} == {"input", "collapsed", "live"}

    second_dead = _dead_node_indices(
        once,
        root,
        strip_unused_return_branches=True,
    )
    assert not second_dead

    result = _apply_dead_code_elimination(
        graph,
        root,
        strip_unused_return_branches=True,
    )
    assert len(result.nodes) == 3
    assert not result.dead_node_indices


def test_dead_code_elimination_is_idempotent():
    root = BlockNode(
        attr_name="hc",
        class_name="HyperConnection",
        role="other",
        label="HC",
        primary_output_step="collapsed",
        multi_return_module=True,
    )
    graph = ComputationGraph(
        nodes=[
            GraphNodeSpec(key="input", label="input", synthetic=SYNTHETIC_INPUT),
            GraphNodeSpec(key="collapsed", block=_block("collapsed")),
            GraphNodeSpec(key="live", block=_block("live")),
            GraphNodeSpec(key="dead_head", block=_block("post")),
            GraphNodeSpec(key="dead_tail", block=_block("tail")),
        ],
        links=[
            (2, 1),
            (1, 0),
            (4, 3),
        ],
        primary_output_index=1,
    )

    once = _apply_dead_code_elimination(
        graph,
        root,
        strip_unused_return_branches=True,
    )
    twice = _apply_dead_code_elimination(
        once,
        root,
        strip_unused_return_branches=True,
    )

    assert len(once.nodes) == len(twice.nodes)
    assert set(once.links) == set(twice.links)
    assert not twice.dead_node_indices


def test_dead_code_elimination_keeps_referenced_multi_return_branches():
    root = BlockNode(
        attr_name="hc",
        class_name="HyperConnection",
        role="other",
        label="HC",
        primary_output_step="collapsed",
        referenced_return_producers={"post", "comb", "collapsed"},
        multi_return_module=True,
    )
    graph = ComputationGraph(
        nodes=[
            GraphNodeSpec(key="collapsed", block=_block("collapsed")),
            GraphNodeSpec(key="pre", block=_block("pre")),
            GraphNodeSpec(key="post", block=_block("post")),
            GraphNodeSpec(key="post_setup", block=_block("post_setup")),
            GraphNodeSpec(key="comb", block=_block("comb")),
            GraphNodeSpec(key="sinkhorn", block=_block("sinkhorn")),
            GraphNodeSpec(key="unused", block=_block("unused")),
        ],
        links=[(1, 0), (3, 2), (5, 4)],
        primary_output_index=0,
    )

    dead = _dead_node_indices(graph, root, strip_unused_return_branches=True)

    assert dead == {6}
