###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A tuple-returning submodule hands each slot to the consumer that asked for it.

``topk_idx, topk_weight = self.gate(x)`` then ``self.moe_infer(h, topk_idx,
topk_weight)``: every slot used to resolve to the gate's LAST op, so the
aggregation was handed the routing weights twice while the expert indices never
arrived, and a router published one tensor twice under one id.
"""

from __future__ import annotations

import ast
import textwrap

from TraceLens.ModelUtils import ast_analyze
from TraceLens.ModelUtils.ast_analyze import ClassStructure, SideInputSpec
from TraceLens.ModelUtils.block_tree import BlockNode
from TraceLens.ModelUtils.computation_graph import _side_slot_sources
from TraceLens.Visualizer.model_explorer_export.merge import _fold_same_source_twins


def _class(name: str, source: str) -> ClassStructure:
    node = ast.parse(textwrap.dedent(source)).body[0]
    assert isinstance(node, ast.ClassDef)
    return ClassStructure(
        name=name,
        node=node,
        init_assignments={},
        init_details={},
        forward_calls=[],
        norm_before=[],
    )


def _registry() -> dict[str, ClassStructure]:
    gate = _class(
        "Gate",
        """
        class Gate:
            def forward(self, hidden_states):
                return topk_idx, topk_weight
        """,
    )
    gate.forward_return_order = ["topk_idx", "topk_weight"]
    gate.forward_return_slots = {"topk_idx": "@op_topk", "topk_weight": "@op_mul"}
    caller = _class(
        "Block",
        """
        class Block:
            def forward(self, hidden_states):
                return hidden_states
        """,
    )
    caller.init_assignments = {"gate": "Gate"}
    return {"Gate": gate, "Block": caller}


def test_submodule_publishes_one_producer_per_return_slot() -> None:
    registry = _registry()
    ast_analyze._publish_submodule_return_producers(registry["Block"], registry)
    assert registry["Block"].forward_step_return_producers == {
        "gate": ["@op_topk", "@op_mul"]
    }


def test_single_slot_return_publishes_nothing() -> None:
    """One return slot has no ordinal to disambiguate, so nothing is published."""
    registry = _registry()
    registry["Gate"].forward_return_order = ["topk_idx"]
    registry["Gate"].forward_return_slots = {"topk_idx": "@op_topk"}
    ast_analyze._publish_submodule_return_producers(registry["Block"], registry)
    assert registry["Block"].forward_step_return_producers == {}


def test_each_side_feed_reads_its_own_slot() -> None:
    root = BlockNode(attr_name="block", class_name="Block", role="other", label="Block")
    root.forward_step_return_producers = {"gate": ["@op_topk", "@op_mul"]}
    sides = [
        SideInputSpec(
            arg_name="top_k_index", port_label="top_k_index", source_chain=["gate"]
        ),
        SideInputSpec(
            arg_name="top_k_weights", port_label="top_k_weights", source_chain=["gate"]
        ),
    ]
    attr_last_index = {"gate": 9, "@op_topk": 4, "@op_mul": 7}

    assert _side_slot_sources(root, sides, attr_last_index) == {0: 4, 1: 7}


def test_partial_unpack_keeps_the_module_tail() -> None:
    """One side cannot say WHICH slot it took, so the wiring is left alone."""
    root = BlockNode(attr_name="block", class_name="Block", role="other", label="Block")
    root.forward_step_return_producers = {"gate": ["@op_topk", "@op_mul"]}
    sides = [
        SideInputSpec(
            arg_name="top_k_weights", port_label="top_k_weights", source_chain=["gate"]
        )
    ]

    assert _side_slot_sources(root, sides, {"gate": 9, "@op_mul": 7}) == {}


def _tile(node_id: str, label: str, namespace: str, synthetic: str, sources):
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": synthetic}],
        "incomingEdges": [
            {"sourceNodeId": source, "sourceNodeOutputId": port}
            for source, port in sources
        ],
    }


def test_a_mirror_repeating_the_tile_beside_it_folds_onto_it() -> None:
    nodes = [
        {"id": "producer", "label": "Op", "namespace": "block"},
        _tile("block/@input", "hidden_states", "block", "@input", [("producer", "0")]),
        _tile(
            "block/child/@input_mirror^hidden_states",
            "hidden_states",
            "block",
            "@input_mirror",
            [("producer", "0")],
        ),
        _tile(
            "block/child/@input",
            "hidden_states",
            "block/child",
            "@input",
            [("block/child/@input_mirror^hidden_states", "0")],
        ),
    ]

    _fold_same_source_twins(nodes)

    assert [n["id"] for n in nodes] == [
        "producer",
        "block/@input",
        "block/child/@input",
    ]
    # The child now reads the declaration its mirror was repeating.
    assert nodes[-1]["incomingEdges"][0]["sourceNodeId"] == "block/@input"


def test_a_kernel_keeps_every_operand_port_it_declares() -> None:
    """One ``cu_seqlens`` feeding ports 15/16/17 is still three operand slots."""
    nodes = [
        {"id": "producer", "label": "Op", "namespace": "attn"},
        _tile(
            "attn/@kernel_in:15:cu_seqlens",
            "cu_seqlens",
            "attn",
            "@kernel_port_in",
            [("producer", "0")],
        ),
        _tile(
            "attn/@kernel_in:16:cu_seqlens",
            "cu_seqlens",
            "attn",
            "@kernel_port_in",
            [("producer", "0")],
        ),
    ]

    _fold_same_source_twins(nodes)

    assert len(nodes) == 3


def test_two_slots_of_one_producer_are_not_the_same_tensor() -> None:
    """A producer declaring no ports can still publish several ordinals."""
    nodes = [
        {"id": "split", "label": "Split", "namespace": "block"},
        _tile("block/@output:a", "value", "block", "@output", [("split", "1")]),
        _tile("block/@output:b", "value", "block", "@output", [("split", "2")]),
    ]

    _fold_same_source_twins(nodes)

    assert len(nodes) == 3


def test_a_stale_slot_index_is_replaced_by_the_node_that_owns_the_attr() -> None:
    from types import SimpleNamespace

    from TraceLens.ModelUtils.computation_graph import (
        ComputationGraph,
        GraphNodeSpec,
        _producer_node_index,
    )

    graph = ComputationGraph(
        nodes=[
            GraphNodeSpec(key="tail", block=SimpleNamespace(attr_name="@op_weights")),
            GraphNodeSpec(
                key="indices", block=SimpleNamespace(attr_name="@op_indices")
            ),
        ]
    )
    found = _producer_node_index(graph, {"call\x00slot0": 0}, "call", 0, "@op_indices")
    assert found == 1


def test_expanding_a_host_int_is_not_the_activation() -> None:
    from TraceLens.ModelUtils.ast_analyze import analyze_sources
    from TraceLens.ModelUtils.basic_ops import BasicOpFilter
    from TraceLens.ModelUtils.block_tree import build_block_node
    from TraceLens.ModelUtils.computation_graph import build_computation_graph

    analysis = analyze_sources(
        {"m.py": """
import torch
class Tower(torch.nn.Module):
    def forward(self, pixel_values, grid_thw):
        size = torch.tensor([2]).expand(len(grid_thw))
        return size
"""},
        all_tensor_ops=True,
    )
    root = build_block_node(
        attr_name="tower",
        class_name="Tower",
        registry=analysis.class_registry,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    graph = build_computation_graph(root, basic_ops=BasicOpFilter.for_detailed())
    expand = next(
        spec
        for spec in graph.nodes
        if spec.block is not None and spec.block.attr_name.endswith("_expand")
    )
    index = graph.nodes.index(expand)
    assert expand.constant
    assert not any(target == index for _source, target in graph.links)


def test_an_output_tile_still_leaves_its_parent_box() -> None:
    from TraceLens.Visualizer.model_explorer_export.merge import (
        _route_frame_exits_through_output_tiles,
    )

    inner = "embed/norm/@output:result"
    mirror = {
        "id": "mirror",
        "label": "result",
        "namespace": "",
        "incomingEdges": [
            {"sourceNodeId": inner, "sourceNodeOutputId": "0"},
        ],
    }
    nodes = [
        {
            "id": inner,
            "label": "result",
            "namespace": "embed/norm",
            "attrs": [{"key": "synthetic", "value": "@output"}],
            "incomingEdges": [],
        },
        mirror,
    ]
    _route_frame_exits_through_output_tiles(nodes)
    source = mirror["incomingEdges"][0]["sourceNodeId"]
    assert source == "embed/@output:result"
