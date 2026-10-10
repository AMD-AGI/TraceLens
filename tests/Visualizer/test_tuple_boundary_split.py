###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A boundary carrying a tuple is drawn as one tile per component.

``position_embeddings`` is one PARAMETER but two tensors. Drawn as a single tile
re-exposing them as two output ports it can publish only one shape for two
tensors, and the reader sees neither name -- which component a port carries is an
ordinal nothing on screen shows.

What must NOT be split matters as much. Several unrelated ops landing on one
boundary (MiniMax's ``block_indices``, fed by an ``Expand``, an
``attention_mask`` and a ``masked_fill``) is a different defect, and inventing
component names for it would paper over that. Nor may a split proceed when a
consumer's port names no component -- guessing would hand a reader the wrong
tensor.
"""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.merge import (
    _split_tuple_boundary_slots,
)


def _tile(
    node_id: str,
    label: str,
    synthetic: str | None = None,
    sources: list[tuple[str, str]] | None = None,
    shape: str | None = None,
) -> dict:
    attrs: list[dict] = []
    if synthetic:
        attrs.append({"key": "synthetic", "value": synthetic})
    if shape:
        attrs.append({"key": "output_shape", "value": shape})
    return {
        "id": node_id,
        "label": label,
        "namespace": "box",
        "attrs": attrs,
        "incomingEdges": [
            {"sourceNodeId": source, "sourceNodeOutputId": port}
            for source, port in (sources or [])
        ],
    }


def _bundle(consumer_ports: list[str]) -> list[dict]:
    """A ``cos``/``sin`` tuple boundary read at the given ports."""
    nodes = [
        _tile("m/@output:cos", "cos", "@output", shape="[B, D] f32"),
        _tile("m/@output:sin", "sin", "@output", shape="[B, D] f32"),
        _tile(
            "box/@input:position_embeddings",
            "position_embeddings",
            "@input",
            [("m/@output:cos", "0"), ("m/@output:sin", "0")],
            shape="[B, S, H] f32",
        ),
    ]
    for index, port in enumerate(consumer_ports):
        nodes.append(
            _tile(
                f"box/@op_reader{index}",
                f"Reader{index}",
                None,
                [("box/@input:position_embeddings", port)],
            )
        )
    return nodes


def _labels(nodes: list[dict]) -> set[str]:
    return {str(node["label"]) for node in nodes}


def _sources_of(nodes: list[dict], node_id: str) -> list[str]:
    node = next(n for n in nodes if n["id"] == node_id)
    return [str(e["sourceNodeId"]) for e in node.get("incomingEdges") or []]


class TestATupleBoundaryBecomesOneTilePerComponent:
    def test_it_splits_and_names_each_component(self) -> None:
        nodes = _bundle(["0", "1"])
        _split_tuple_boundary_slots(nodes)
        assert "position_embeddings.cos" in _labels(nodes)
        assert "position_embeddings.sin" in _labels(nodes)
        assert "position_embeddings" not in _labels(nodes)

    def test_each_component_carries_its_producer_s_shape(self) -> None:
        """Not the one shape a bundle had to pick for two tensors."""
        nodes = _bundle(["0", "1"])
        _split_tuple_boundary_slots(nodes)
        for node in nodes:
            if str(node["label"]).startswith("position_embeddings."):
                shapes = [
                    a["value"] for a in node["attrs"] if a["key"] == "output_shape"
                ]
                assert shapes == ["[B, D] f32"], (node["id"], shapes)

    def test_a_consumer_reads_the_component_it_addressed(self) -> None:
        nodes = _bundle(["0", "1"])
        _split_tuple_boundary_slots(nodes)
        assert _sources_of(nodes, "box/@op_reader0") == [
            "box/@input:position_embeddings.cos"
        ]
        assert _sources_of(nodes, "box/@op_reader1") == [
            "box/@input:position_embeddings.sin"
        ]


class TestWhatIsLeftAlone:
    def test_unrelated_producers_are_not_a_tuple(self) -> None:
        """``block_indices`` fed by three ops is a different defect."""
        nodes = [
            _tile("m/@op_expand", "Expand"),
            _tile("m/@op_fill", "masked_fill"),
            _tile(
                "box/@input:block_indices",
                "block_indices",
                "@input",
                [("m/@op_expand", "0"), ("m/@op_fill", "0")],
            ),
            _tile(
                "box/@op_reader", "Reader", None, [("box/@input:block_indices", "0")]
            ),
        ]
        _split_tuple_boundary_slots(nodes)
        assert "block_indices" in _labels(nodes)

    def test_producers_sharing_a_name_are_not_components(self) -> None:
        """Two producers called ``result`` name no distinct components."""
        nodes = [
            _tile("m/@output:a", "result", "@output"),
            _tile("m/@output:b", "result", "@output"),
            _tile(
                "box/@input:x",
                "x",
                "@input",
                [("m/@output:a", "0"), ("m/@output:b", "0")],
            ),
            _tile("box/@op_reader", "Reader", None, [("box/@input:x", "0")]),
        ]
        _split_tuple_boundary_slots(nodes)
        assert "x" in _labels(nodes)

    def test_a_consumer_port_naming_no_component_aborts(self) -> None:
        """Guessing which component it wanted would hand it the wrong tensor."""
        nodes = _bundle(["mystery"])
        _split_tuple_boundary_slots(nodes)
        assert "position_embeddings" in _labels(nodes)
        assert "position_embeddings.cos" not in _labels(nodes)


class TestAPassThroughTakesEveryComponent:
    def test_a_boundary_forwarding_the_tuple_gets_one_edge_each(self) -> None:
        """It carried one tensor while claiming two downstream."""
        nodes = _bundle([])
        nodes.append(
            _tile(
                "box/inner/@input:position_embeddings",
                "position_embeddings",
                "@input",
                [("box/@input:position_embeddings", "0")],
            )
        )
        # Its own consumers address two components, so it forwards the whole tuple.
        for port in ("0", "1"):
            nodes.append(
                _tile(
                    f"box/inner/@op_use{port}",
                    f"Use{port}",
                    None,
                    [("box/inner/@input:position_embeddings", port)],
                )
            )
        _split_tuple_boundary_slots(nodes)
        # Taking every component makes it a tuple boundary in turn, so the split
        # walks the chain and the inner level is per-component too.
        assert "position_embeddings" not in _labels(nodes)
        assert {"position_embeddings.cos", "position_embeddings.sin"} <= _labels(nodes)
        assert _sources_of(nodes, "box/inner/@op_use0") == [
            "box/inner/@input:position_embeddings.cos"
        ]
        assert _sources_of(nodes, "box/inner/@op_use1") == [
            "box/inner/@input:position_embeddings.sin"
        ]
        # Each inner component traces back to its own outer component.
        assert _sources_of(nodes, "box/inner/@input:position_embeddings.cos") == [
            "box/@input:position_embeddings.cos"
        ]


class TestAddressingVersusForwarding:
    """Which component a consumer wants is read from how it ADDRESSES them.

    A mirror chain forwards a tuple by handing the next mirror its single
    output, sometimes as two parallel edges off the same port. Counting edges,
    or trusting a bare port ``"0"``, reads that as "component 0" and the other
    component reaches nothing -- which is how ``sin`` disappeared behind
    DeepSeek's ``l813`` mirror and MiniMax's attention boundary.
    """

    def _forwarder(self, edges: list[str]) -> list[dict]:
        nodes = _bundle([])
        nodes.append(
            _tile(
                "box/inner/@input:position_embeddings",
                "position_embeddings",
                "@input",
                [("box/@input:position_embeddings", port) for port in edges],
            )
        )
        for port in ("0", "1"):
            nodes.append(
                _tile(
                    f"box/inner/@op_use{port}",
                    f"Use{port}",
                    None,
                    [("box/inner/@input:position_embeddings", port)],
                )
            )
        return nodes

    def test_two_parallel_edges_off_one_port_are_not_an_address(self) -> None:
        nodes = self._forwarder(["0", "0"])
        _split_tuple_boundary_slots(nodes)
        assert _sources_of(nodes, "box/inner/@op_use1") == [
            "box/inner/@input:position_embeddings.sin"
        ]

    def test_a_single_edge_off_port_zero_is_not_an_address(self) -> None:
        nodes = self._forwarder(["0"])
        _split_tuple_boundary_slots(nodes)
        assert _sources_of(nodes, "box/inner/@op_use1") == [
            "box/inner/@input:position_embeddings.sin"
        ]

    def test_addressing_every_component_is_honoured(self) -> None:
        """Edges naming distinct components are mapped, not re-fanned."""
        nodes = self._forwarder(["0", "1"])
        _split_tuple_boundary_slots(nodes)
        assert _sources_of(nodes, "box/inner/@op_use0") == [
            "box/inner/@input:position_embeddings.cos"
        ]
        assert _sources_of(nodes, "box/inner/@op_use1") == [
            "box/inner/@input:position_embeddings.sin"
        ]

    def test_an_op_addressing_one_component_still_gets_that_one(self) -> None:
        """Only a BOUNDARY forwards a tuple; an op reads the part it named."""
        nodes = _bundle(["0"])
        _split_tuple_boundary_slots(nodes)
        assert _sources_of(nodes, "box/@op_reader0") == [
            "box/@input:position_embeddings.cos"
        ]
