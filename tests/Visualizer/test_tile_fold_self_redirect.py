###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Folding "one value said twice" must not fold a tile onto itself.

``_fold_same_source_twins`` groups boundary tiles that share a scope, a name and
a producer, keeps the plainest one and redirects the rest to it. The removal step
drops every id in the redirect map -- so when a group contains the same id twice
(one tile reached through two node objects), the tile is redirected to ITSELF,
its id lands in the removal set, and BOTH copies go. The producer, which had
exactly one consumer, is then dead.

Found while routing MiniMax's ``key_states`` through its rotary frame: the
frame's ``k_embed`` output boundary vanished and the concat computing it was
reported as a dead node. The duplicate itself is a separate defect -- this test
only pins that the fold cannot delete a tile by folding it onto itself.
"""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.merge import _fold_same_source_twins


def _tile(node_id: str, label: str, namespace: str, source: str) -> dict:
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": "@output"}],
        "outputsMetadata": [{"id": "0", "attrs": []}],
        "incomingEdges": [{"sourceNodeId": source, "sourceNodeOutputId": "0"}],
    }


def _producer(node_id: str, namespace: str) -> dict:
    return {
        "id": node_id,
        "label": "Concat",
        "namespace": namespace,
        "outputsMetadata": [{"id": "0", "attrs": []}],
        "incomingEdges": [],
    }


class TestSelfRedirect:
    def test_one_id_listed_twice_survives_the_fold(self) -> None:
        concat = _producer("frame/concat", "frame")
        tile = _tile("frame/@output:k_embed", "k_embed", "frame", "frame/concat")
        # The same tile reachable twice in the node list.
        nodes = [concat, tile, dict(tile)]

        _fold_same_source_twins(nodes)

        survivors = [n for n in nodes if n["id"] == "frame/@output:k_embed"]
        assert survivors, "the fold deleted the tile by folding it onto itself"

    def test_the_producer_keeps_a_consumer(self) -> None:
        concat = _producer("frame/concat", "frame")
        tile = _tile("frame/@output:k_embed", "k_embed", "frame", "frame/concat")
        nodes = [concat, tile, dict(tile)]

        _fold_same_source_twins(nodes)

        consumed = {
            str(edge.get("sourceNodeId"))
            for node in nodes
            for edge in node.get("incomingEdges", []) or []
        }
        assert "frame/concat" in consumed, "producer left with nothing reading it"

    def test_two_genuinely_distinct_tiles_still_fold(self) -> None:
        """The pass must keep doing its job: same scope, name and producer.

        The survivor is the one consumers already read -- that keeps every
        reader at the level it was reading from.
        """
        concat = _producer("frame/concat", "frame")
        nodes = [
            concat,
            _tile("frame/@output:k_embed", "k_embed", "frame", "frame/concat"),
            _tile("frame/@output:k_embed#2", "k_embed", "frame", "frame/concat"),
            {
                "id": "reader",
                "label": "reader",
                "namespace": "outer",
                "incomingEdges": [
                    {
                        "sourceNodeId": "frame/@output:k_embed#2",
                        "sourceNodeOutputId": "0",
                    }
                ],
            },
        ]

        _fold_same_source_twins(nodes)

        ids = {n["id"] for n in nodes}
        tiles = [i for i in ids if "@output:k_embed" in i]
        assert len(tiles) == 1, ids
        reader = next(n for n in nodes if n["id"] == "reader")
        assert reader["incomingEdges"][0]["sourceNodeId"] == tiles[0]
