###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A named tensor a box is handed is drawn as that box's input.

``chunk_kda_pipeline`` declares its operands as ``@tensor`` tiles -- ``q``, ``k``,
``v``, ``g``, ``beta`` -- and they already ARE the box's interface: every edge
entering the box lands on one, and the ``block boundaries`` check counts them as
input boundaries. They just did not look like it. Drawn in the ordinary op gray,
a reader opening ``KimiDeltaAttention`` saw the pipeline's operands as though
they were computed inside it, with nothing marking where its inputs are.

The test is the tile's own dataflow, not its name: a ``@tensor`` tile all of
whose producers lie outside its namespace was entered there.
"""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.merge import (
    _draw_entering_tensor_tiles_as_inputs,
)
from TraceLens.Visualizer.model_explorer_export.styles import input_port_style

_INPUT_FILL = input_port_style()["backgroundColor"]


def _tile(node_id: str, namespace: str, sources: list[str], **extra) -> dict:
    node = {
        "id": node_id,
        "label": node_id.rsplit("/", 1)[-1],
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": "@tensor"}],
        "style": {"backgroundColor": "#cccccc"},
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": "0"} for s in sources
        ],
    }
    node.update(extra)
    return node


def _op(node_id: str, namespace: str) -> dict:
    return {
        "id": node_id,
        "label": node_id.rsplit("/", 1)[-1],
        "namespace": namespace,
        "incomingEdges": [],
    }


def _fill(node: dict) -> str:
    return str((node.get("style") or {}).get("backgroundColor"))


class TestEnteringTilesAreDrawnAsInputs:
    def test_a_tile_fed_from_outside_the_box_is_an_input(self) -> None:
        nodes = [
            _op("attn/Rearrange", "attn"),
            _tile("attn/pipe/q", "attn/pipe", ["attn/Rearrange"]),
        ]
        _draw_entering_tensor_tiles_as_inputs(nodes)
        assert _fill(nodes[1]) == _INPUT_FILL

    def test_a_tile_fed_from_inside_the_box_is_not(self) -> None:
        """An intermediate value keeps the op styling."""
        nodes = [
            _op("attn/pipe/CumSum", "attn/pipe"),
            _tile("attn/pipe/gk", "attn/pipe", ["attn/pipe/CumSum"]),
        ]
        _draw_entering_tensor_tiles_as_inputs(nodes)
        assert _fill(nodes[1]) != _INPUT_FILL

    def test_a_tile_fed_from_a_nested_box_is_not(self) -> None:
        """Deeper is still inside: the tensor did not enter here."""
        nodes = [
            _op("attn/pipe/inner/Op", "attn/pipe/inner"),
            _tile("attn/pipe/out", "attn/pipe", ["attn/pipe/inner/Op"]),
        ]
        _draw_entering_tensor_tiles_as_inputs(nodes)
        assert _fill(nodes[1]) != _INPUT_FILL

    def test_a_constant_is_never_a_box_input(self) -> None:
        nodes = [
            _op("attn/weight", "attn"),
            _tile(
                "attn/pipe/w",
                "attn/pipe",
                ["attn/weight"],
                attrs=[
                    {"key": "synthetic", "value": "@tensor"},
                    {"key": "constant", "value": "true"},
                ],
            ),
        ]
        _draw_entering_tensor_tiles_as_inputs(nodes)
        assert _fill(nodes[1]) != _INPUT_FILL

    def test_a_scalar_operand_docked_beside_one_op_is_not(self) -> None:
        nodes = [
            _op("attn/src", "attn"),
            _tile("attn/pipe/op:external:scale", "attn/pipe", ["attn/src"]),
        ]
        _draw_entering_tensor_tiles_as_inputs(nodes)
        assert _fill(nodes[1]) != _INPUT_FILL

    def test_a_tile_with_no_producer_is_left_alone(self) -> None:
        nodes = [_tile("attn/pipe/q", "attn/pipe", [])]
        _draw_entering_tensor_tiles_as_inputs(nodes)
        assert _fill(nodes[0]) != _INPUT_FILL

    def test_mixed_producers_mean_it_was_not_entered_here(self) -> None:
        """One producer inside is enough: the tile is not purely an entry."""
        nodes = [
            _op("attn/Rearrange", "attn"),
            _op("attn/pipe/Local", "attn/pipe"),
            _tile(
                "attn/pipe/q",
                "attn/pipe",
                ["attn/Rearrange", "attn/pipe/Local"],
            ),
        ]
        _draw_entering_tensor_tiles_as_inputs(nodes)
        assert _fill(nodes[2]) != _INPUT_FILL
