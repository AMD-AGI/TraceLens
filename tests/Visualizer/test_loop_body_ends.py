###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A loop shows its body and names both ends; it does not draw a cycle.

``Loop in``/``Loop out`` existed to make the back edge legible, and that edge
was the only cycle the graph allowed. Two extra boxes plus a cycle say less than
the body's own ends do once those are named for the direction they face -- and a
cycle drawn around a body costs every reader who has to work out it is not real
dataflow.

So the ports fold away everywhere: the seed feeds the body directly, the body
feeds its consumer directly, and what remains is ``loop in: <var> iterations:
<count>`` and ``loop out: <var> iterations: <count>`` -- the same tensor named
the same way at both ends, one named pair PER carried value, each saying how
often the body between them runs.

Two things this must not do:

* Claim a recurrence that is not there. ``get_vision_position_ids`` appends to a
  list and concatenates afterwards, so it has an exit and no entry; calling that
  exit a ``loop out`` would invent a back edge the source has not got.
* Lose the recurrence for anything that is not a human. The ports carried the
  ``@loop_carried`` marker, so the body ends carry it now -- invisible in the
  render, still findable by a pass that needs it.
"""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.merge import (
    _fold_loop_ports,
    _node_attr,
)


def _port(prefix: str, kind: str, loop: str, var: str, namespace: str) -> dict:
    return {
        "id": f"{prefix}@loop_carried_{kind}:{loop}:{var}",
        "label": "Loop in" if kind == "in" else "Loop out",
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": "@loop_carried"}],
        "incomingEdges": [],
    }


def _body_tile(prefix: str, kind: str, loop: str, var: str, namespace: str) -> dict:
    return {
        "id": f"{prefix}@body_{kind}:{loop}:{var}",
        "label": var,
        "namespace": namespace,
        "attrs": [
            {"key": "synthetic", "value": "@input" if kind == "in" else "@output"}
        ],
        "incomingEdges": [],
    }


def _loop(extra_inputs: int = 0, carried: tuple[str, ...] = ("final",)) -> list[dict]:
    body = "box/loop_iterations_8"
    nodes: list[dict] = []
    for var in carried:
        nodes += [
            _port("box/", "in", "loop_l1", var, "box"),
            _port("box/", "out", "loop_l1", var, "box"),
            _body_tile("box/", "in", "loop_l1", var, body),
            _body_tile("box/", "out", "loop_l1", var, body),
        ]
    for index in range(extra_inputs):
        nodes.append(
            {
                "id": f"{body}/@input:extra{index}",
                "label": f"extra{index}",
                "namespace": body,
                "attrs": [{"key": "synthetic", "value": "@input"}],
                "incomingEdges": [],
            }
        )
    return nodes


def _ports(nodes: list[dict]) -> list[str]:
    return [str(n["id"]) for n in nodes if "@loop_carried" in str(n["id"])]


def _labels(nodes: list[dict]) -> set[str]:
    return {str(n["label"]) for n in nodes}


def _carried_name(node: dict) -> str:
    """The tensor out of ``loop <side>: <var> iterations: <count>``."""
    return str(node["label"]).split(": ", 1)[1].split(" iterations: ", 1)[0]


class TestEveryLoopFoldsItsPorts:
    def test_a_body_handed_only_what_it_carries(self) -> None:
        nodes = _loop(extra_inputs=0)
        _fold_loop_ports(nodes)
        assert _ports(nodes) == []

    def test_a_body_handed_another_tensor(self) -> None:
        nodes = _loop(extra_inputs=1)
        _fold_loop_ports(nodes)
        assert _ports(nodes) == []

    def test_a_body_handed_several(self) -> None:
        """DeepSeek's expert loop is handed four tensors beside its accumulator."""
        nodes = _loop(extra_inputs=4)
        _fold_loop_ports(nodes)
        assert _ports(nodes) == []


class TestBothEndsAreNamed:
    def test_the_ends_say_which_end_they_are(self) -> None:
        nodes = _loop()
        _fold_loop_ports(nodes)
        assert "loop in: final iterations: 8" in _labels(nodes)
        assert "loop out: final iterations: 8" in _labels(nodes)

    def test_both_ends_name_the_same_tensor(self) -> None:
        """The direction differs; the tensor does not."""
        nodes = _loop()
        _fold_loop_ports(nodes)
        named = [n for n in nodes if str(n["label"]).startswith("loop ")]
        assert {_carried_name(n) for n in named} == {"final"}

    def test_each_carried_value_gets_its_own_pair(self) -> None:
        """A loop carrying two values names two pairs, not one combined tile."""
        nodes = _loop(carried=("final", "running"))
        _fold_loop_ports(nodes)
        assert _ports(nodes) == []
        for var in ("final", "running"):
            assert f"loop in: {var} iterations: 8" in _labels(nodes)
            assert f"loop out: {var} iterations: 8" in _labels(nodes)

    def test_the_marker_survives_for_a_later_pass(self) -> None:
        nodes = _loop()
        _fold_loop_ports(nodes)
        named = [n for n in nodes if str(n["label"]).startswith("loop ")]
        assert named
        for node in named:
            assert _node_attr(node, "loop_carried") == "loop_l1:final", node["id"]


class TestBothEndsSayHowOften:
    def test_a_repeat_group_counts_from_its_own_segment(self) -> None:
        """A ``{N}x_`` group states its trip count where a frame has none."""
        nodes = _loop()
        for node in nodes:
            if "@body_" in str(node["id"]):
                node["namespace"] = "box/12x_Block"
        _fold_loop_ports(nodes)
        assert "loop in: final iterations: 12" in _labels(nodes)

    def test_the_innermost_loop_owns_the_count(self) -> None:
        """An expert loop runs its own count, not the decoder stack's."""
        nodes = _loop()
        for node in nodes:
            if "@body_" in str(node["id"]):
                node["namespace"] = "45x_DecoderLayer/MoE/loop_iterations_288"
        _fold_loop_ports(nodes)
        assert "loop in: final iterations: 288" in _labels(nodes)
        assert "loop out: final iterations: 288" in _labels(nodes)

    def test_a_count_that_is_not_a_number_is_still_a_count(self) -> None:
        """A loop over a tensor's rows is named for the axis it walks."""
        nodes = _loop()
        for node in nodes:
            if "@body_" in str(node["id"]):
                node["namespace"] = "visual/loop_iterations_Img"
        _fold_loop_ports(nodes)
        assert "loop in: final iterations: Img" in _labels(nodes)

    def test_a_frame_with_no_count_says_nothing_it_cannot_know(self) -> None:
        """Two siblings sharing a label fall back to a frame id, which has none."""
        nodes = _loop()
        for node in nodes:
            if "@body_" in str(node["id"]):
                node["namespace"] = "box/loop_op_l12_c8_add"
        _fold_loop_ports(nodes)
        assert "loop in: final" in _labels(nodes)
        assert not any(" iterations: " in label for label in _labels(nodes))


class TestAnAccumulatorIsNotARecurrence:
    def test_an_exit_without_an_entry_keeps_its_plain_name(self) -> None:
        """``position_ids`` is appended per iteration, then concatenated."""
        body = "box/loop_iterations_repeated"
        nodes = [
            _port("box/", "in", "loop_l2", "position_ids", "box"),
            _port("box/", "out", "loop_l2", "position_ids", "box"),
            _body_tile("box/", "out", "loop_l2", "position_ids", body),
        ]
        _fold_loop_ports(nodes)
        assert _ports(nodes) == [], "its ports go too -- the back edge was invented"
        assert "position_ids" in _labels(nodes)
        assert not any(str(n["label"]).startswith("loop ") for n in nodes)

    def test_it_carries_no_recurrence_marker(self) -> None:
        body = "box/loop_iterations_repeated"
        nodes = [
            _port("box/", "in", "loop_l2", "position_ids", "box"),
            _port("box/", "out", "loop_l2", "position_ids", "box"),
            _body_tile("box/", "out", "loop_l2", "position_ids", body),
        ]
        _fold_loop_ports(nodes)
        exit_tile = next(n for n in nodes if "@body_out:" in str(n["id"]))
        assert _node_attr(exit_tile, "loop_carried") is None
