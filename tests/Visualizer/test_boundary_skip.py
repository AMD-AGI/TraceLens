###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A tensor entering a box lands on that box's own boundary.

GLM's ``grid_thw`` is a model input consumed deep inside the vision tower. It was
wired straight from the top level to a tile two levels down, inside
``visual/get_vision_attention_seqlens`` -- so the ``visual`` box drew no
``grid_thw`` input at all, even though the tensor plainly enters it. A reader
opening the tower sees its inputs; one that crosses the wall invisibly is not
among them.

``I5 boundary-skip`` is the standing check. Repeat groups are deliberately not
walls: ``45x_Glm5NextTextDecoderLayer`` renders N identical layers rather than
naming a scope a tensor enters, and its carried values are drawn by the loop
boundary machinery instead.
"""

from __future__ import annotations

import pytest

from TraceLens.Visualizer.model_explorer_export.type_check import (
    integrity_check_graph_nodes,
)

_MODEL = "zai-org/GLM-5.3-Flash"


def _node(node_id: str, namespace: str, sources: list[str] | None = None) -> dict:
    return {
        "id": node_id,
        "label": node_id.rsplit("/", 1)[-1],
        "namespace": namespace,
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": "0"} for s in sources or []
        ],
        "attrs": [],
    }


def _skips(nodes: list[dict]) -> list[str]:
    return [w for w in integrity_check_graph_nodes(nodes) if "I5" in w]


class TestTheCheck:
    def test_an_edge_two_boxes_deep_is_flagged(self) -> None:
        nodes = [
            _node("@in", ""),
            _node("tower/inner/op", "tower/inner", ["@in"]),
        ]
        assert _skips(nodes), "an edge skipping the tower's boundary went unflagged"

    def test_entering_one_box_is_fine(self) -> None:
        nodes = [_node("@in", ""), _node("tower/@input", "tower", ["@in"])]
        assert not _skips(nodes)

    def test_leaving_two_boxes_is_flagged(self) -> None:
        nodes = [
            _node("tower/inner/op", "tower/inner"),
            _node("after", "", ["tower/inner/op"]),
        ]
        assert _skips(nodes)

    def test_a_repeat_group_is_not_a_wall(self) -> None:
        """``45x_...`` renders N identical layers; it is not a scope entered."""
        nodes = [
            _node("@in", ""),
            _node("decoder/45x_Layer/op", "decoder/45x_Layer", ["@in"]),
        ]
        assert not _skips(nodes), "a repeat group was treated as a box wall"

    def test_a_sibling_hop_is_not_a_skip(self) -> None:
        nodes = [
            _node("tower/a/op", "tower/a"),
            _node("tower/b/op", "tower/b", ["tower/a/op"]),
        ]
        assert not _skips(nodes)


@pytest.fixture(scope="module")
def glm_nodes(model_graph_nodes):
    return model_graph_nodes(_MODEL)


class TestTheModelHasNoSkips:
    def test_glm_has_no_boundary_skips(self, glm_nodes) -> None:
        assert _skips(list(glm_nodes)) == []

    def test_the_vision_tower_declares_the_grid_it_is_handed(self, glm_nodes) -> None:
        """The defect itself: the tower showing no ``grid_thw`` input."""
        tiles = [
            n
            for n in glm_nodes
            if (n.get("label") or "") == "grid_thw"
            and str(n.get("namespace") or "") == "visual"
        ]
        assert tiles, "the visual tower has no grid_thw boundary of its own"
