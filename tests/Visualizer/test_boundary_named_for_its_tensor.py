###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A boundary fed by a tensor's own mirror carries that tensor's name.

A tensor that enters a box without a boundary gets one minted for it, named
after whatever produced it -- which is an OPERATION. When that producer is later
redirected to the tensor's mirror, the tile keeps the operation's name, because
the existing fold only renames a tile when two of them share one producer and a
lone tile never qualified. GLM's ``get_visible_tokens`` drew a port called
``Slice`` for what is plainly ``valid_keys``.

A mirror carries the SAME tensor across a wall, so its name IS the tensor's
name. A boundary fed by an ``@output`` is a different thing -- that one is a
genuine rename across a wall (``@input:hidden_states`` from
``@output:collapsed``) -- and must be left alone.
"""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.merge import (
    _name_a_boundary_for_its_mirror,
)


def _tile(node_id: str, label: str, synthetic: str, source: str | None = None) -> dict:
    return {
        "id": node_id,
        "label": label,
        "namespace": "box",
        "attrs": [{"key": "synthetic", "value": synthetic}],
        "incomingEdges": (
            [{"sourceNodeId": source, "sourceNodeOutputId": "0"}] if source else []
        ),
    }


def _named(nodes: list[dict], node_id: str) -> str:
    return next(str(n["label"]) for n in nodes if n["id"] == node_id)


class TestABoundaryTakesItsTensorsName:
    def test_it_adopts_the_mirror_s_name(self) -> None:
        nodes = [
            _tile(
                "m/@input_mirror:valid_keys^valid_keys", "valid_keys", "@input_mirror"
            ),
            _tile(
                "box/@input:Slice",
                "Slice",
                "@input",
                "m/@input_mirror:valid_keys^valid_keys",
            ),
        ]
        _name_a_boundary_for_its_mirror(nodes)
        assert _named(nodes, "box/@input:Slice") == "valid_keys"

    def test_a_rename_across_a_wall_is_left_alone(self) -> None:
        """``@input:hidden_states`` from ``@output:collapsed`` is deliberate."""
        nodes = [
            _tile("m/@output:collapsed", "collapsed", "@output"),
            _tile(
                "box/@input:hidden_states",
                "hidden_states",
                "@input",
                "m/@output:collapsed",
            ),
        ]
        _name_a_boundary_for_its_mirror(nodes)
        assert _named(nodes, "box/@input:hidden_states") == "hidden_states"

    def test_a_tile_already_named_for_its_tensor_does_not_move(self) -> None:
        nodes = [
            _tile(
                "m/@input_mirror:valid_keys^valid_keys", "valid_keys", "@input_mirror"
            ),
            _tile(
                "box/@input:valid_keys",
                "valid_keys",
                "@input",
                "m/@input_mirror:valid_keys^valid_keys",
            ),
        ]
        _name_a_boundary_for_its_mirror(nodes)
        assert _named(nodes, "box/@input:valid_keys") == "valid_keys"

    def test_a_tile_fed_by_several_producers_is_left_alone(self) -> None:
        """Two producers means this tile does not stand for one named tensor."""
        node = _tile(
            "box/@input:Slice",
            "Slice",
            "@input",
            "m/@input_mirror:valid_keys^valid_keys",
        )
        node["incomingEdges"].append(
            {"sourceNodeId": "m/@input_mirror:other^other", "sourceNodeOutputId": "0"}
        )
        nodes = [
            _tile(
                "m/@input_mirror:valid_keys^valid_keys", "valid_keys", "@input_mirror"
            ),
            _tile("m/@input_mirror:other^other", "other", "@input_mirror"),
            node,
        ]
        _name_a_boundary_for_its_mirror(nodes)
        assert _named(nodes, "box/@input:Slice") == "Slice"
