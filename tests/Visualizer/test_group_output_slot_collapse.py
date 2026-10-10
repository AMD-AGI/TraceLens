###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two edges landing on one return slot make one boundary, not two nodes.

``_inject_group_outputs`` emits one Output node per return slot, and the node's
id encodes that slot. When two outgoing edges resolve to the SAME slot it emitted
the node twice under one id -- and a later fold, finding a group that holds that
id twice, redirects it to itself and deletes both copies, orphaning the producer.

It happens whenever a consumer addresses a slot by the CALLEE's ordinal while the
op producing it inside the frame has only output 0: MiniMax's ``key_states`` is
return 1 of ``apply_rotary_pos_emb``, whose ``k_embed`` comes from a single
``cat``. The two edges carry one tensor, so they are one boundary.
"""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.merge import _inject_group_outputs


def _node(node_id: str, namespace: str, **extra) -> dict:
    node = {
        "id": node_id,
        "label": node_id.rsplit("/", 1)[-1],
        "namespace": namespace,
        "incomingEdges": [],
        "outputsMetadata": [{"id": "0", "attrs": []}],
    }
    node.update(extra)
    return node


def _frame_nodes() -> list[dict]:
    """A frame returning two slots, whose second is read at two ordinals."""
    inside = "frame"
    nodes = [
        _node(
            f"{inside}/@input",
            inside,
            attrs=[{"key": "synthetic", "value": "@input"}],
        ),
        _node(f"{inside}/q_cat", inside),
        _node(f"{inside}/k_cat", inside),
        # Two consumers outside the frame, both reading k_cat -- one at the
        # producer's own port 0, one at the callee's return ordinal 1.
        _node("outer/reader_a", "outer"),
        _node("outer/reader_b", "outer"),
    ]
    by_id = {n["id"]: n for n in nodes}
    by_id["outer/reader_a"]["incomingEdges"] = [
        {"sourceNodeId": f"{inside}/k_cat", "sourceNodeOutputId": "0"}
    ]
    by_id["outer/reader_b"]["incomingEdges"] = [
        {"sourceNodeId": f"{inside}/k_cat", "sourceNodeOutputId": "1"},
        {"sourceNodeId": f"{inside}/q_cat", "sourceNodeOutputId": "0"},
    ]
    return nodes


def _slot_names(_prefix: str) -> dict[str, list[str]]:
    return {"q_cat": ["q_embed"], "k_cat": ["k_embed"]}


class TestOneSlotOneBoundary:
    def test_no_two_nodes_share_an_output_id(self) -> None:
        nodes = _frame_nodes()
        _inject_group_outputs(nodes, resolve_slot_names=_slot_names)
        ids = [str(n["id"]) for n in nodes]
        assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]

    def test_both_edges_reach_the_same_boundary(self) -> None:
        nodes = _frame_nodes()
        _inject_group_outputs(nodes, resolve_slot_names=_slot_names)
        by_id = {str(n["id"]): n for n in nodes}
        reaching = {
            str(edge["sourceNodeId"])
            for name in ("outer/reader_a", "outer/reader_b")
            for edge in by_id[name]["incomingEdges"]
        }
        # Every consumer now reads a boundary, not the frame's internals.
        assert not any(
            src.startswith("frame/") and "@output" not in src for src in reaching
        ), reaching

    def test_the_producer_keeps_a_consumer(self) -> None:
        nodes = _frame_nodes()
        _inject_group_outputs(nodes, resolve_slot_names=_slot_names)
        consumed = {
            str(edge.get("sourceNodeId"))
            for node in nodes
            for edge in node.get("incomingEdges", []) or []
        }
        assert "frame/k_cat" in consumed, "the slot's producer was orphaned"
