###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""One tensor entering several boxes is drawn once, not once per box.

``position_ids`` is computed by a single ``unsqueeze`` at model scope and read by
the decoder stack, the rotary embedding and the mask builder. Each of those
minted its own boundary straight off the producer, so the diagram showed the
unsqueeze fanning out to three unrelated-looking ports -- one of them labelled
``Unsqueeze``, after the producing op rather than the tensor -- while a correctly
named ``position_ids`` tile sat beside them carrying only the decoder's edge.

The producer should feed ONE named tile that every box reads.
"""

from __future__ import annotations

import pytest

from TraceLens.Visualizer.model_explorer_export.merge import (
    _share_one_tile_per_entering_tensor,
)


def _tile(node_id: str, namespace: str, label: str, synthetic: str, sources=()):
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": synthetic}],
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": "0"} for s in sources
        ],
    }


def _op(node_id: str, namespace: str):
    return {"id": node_id, "label": "Unsqueeze", "namespace": namespace, "attrs": []}


def _sources(node) -> list[str]:
    return [str(e["sourceNodeId"]) for e in node.get("incomingEdges") or []]


class TestTheFold:
    def test_every_box_reads_one_shared_tile(self) -> None:
        nodes = [
            _op("op", ""),
            _tile(
                "@input_mirror:position_ids^Nx",
                "",
                "position_ids",
                "@input_mirror",
                ["op"],
            ),
            _tile(
                "rotary/@input:position_ids", "rotary", "position_ids", "@input", ["op"]
            ),
            _tile("mask/@input:Unsqueeze", "mask", "Unsqueeze", "@input", ["op"]),
        ]
        _share_one_tile_per_entering_tensor(nodes)
        shared = "@input_mirror:position_ids^Nx"
        assert _sources(nodes[2]) == [shared]
        assert _sources(nodes[3]) == [shared]
        assert _sources(nodes[1]) == ["op"]

    def test_a_boundary_takes_the_name_of_the_tensor(self) -> None:
        """Not the name of the op that produced it."""
        nodes = [
            _op("op", ""),
            _tile(
                "@input_mirror:position_ids^Nx",
                "",
                "position_ids",
                "@input_mirror",
                ["op"],
            ),
            _tile("mask/@input:Unsqueeze", "mask", "Unsqueeze", "@input", ["op"]),
        ]
        _share_one_tile_per_entering_tensor(nodes)
        assert nodes[2]["label"] == "position_ids"

    def test_a_lone_consumer_is_left_alone(self) -> None:
        nodes = [
            _op("op", ""),
            _tile(
                "rotary/@input:position_ids", "rotary", "position_ids", "@input", ["op"]
            ),
        ]
        _share_one_tile_per_entering_tensor(nodes)
        assert _sources(nodes[1]) == ["op"]

    def test_nothing_is_rerouted_when_no_tile_sits_above_the_producer(self) -> None:
        """Rerouting into a box that the producer is not inside would be a cycle."""
        nodes = [
            _op("outer/op", "outer"),
            _tile("a/@input:x", "a", "x", "@input", ["outer/op"]),
            _tile("b/@input:x", "b", "x", "@input", ["outer/op"]),
        ]
        _share_one_tile_per_entering_tensor(nodes)
        assert _sources(nodes[1]) == ["outer/op"]
        assert _sources(nodes[2]) == ["outer/op"]

    def test_separate_output_ports_are_separate_tensors(self) -> None:
        nodes = [
            _op("op", ""),
            _tile("@input_mirror:cos^Nx", "", "cos", "@input_mirror", ["op"]),
            _tile("box/@input:sin", "box", "sin", "@input", ["op"]),
        ]
        nodes[2]["incomingEdges"][0]["sourceNodeOutputId"] = "1"
        _share_one_tile_per_entering_tensor(nodes)
        assert _sources(nodes[2]) == ["op"]


@pytest.mark.parametrize(
    "model_id,producer",
    [
        ("MiniMaxAI/MiniMax-M3", "@op_l758_c27_unsqueeze"),
        ("deepseek-ai/DeepSeek-V4-Flash", "@op_l1296_c27_unsqueeze"),
    ],
)
def test_position_ids_enters_every_box_through_one_tile(
    model_graph_nodes, model_id, producer
):
    nodes = model_graph_nodes(model_id)
    fanout = [
        n
        for n in nodes
        for e in n.get("incomingEdges") or []
        if producer in str(e.get("sourceNodeId"))
    ]
    assert len(fanout) == 1, [str(n["id"]) for n in fanout]
    assert (fanout[0].get("label") or "") == "position_ids", fanout[0].get("label")
