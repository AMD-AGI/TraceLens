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

from TraceLens.Visualizer.model_explorer_export.merge import _fold_same_source_twins


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
