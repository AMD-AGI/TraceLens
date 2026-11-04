###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Where a loop body ends, and how mutually exclusive layer variants meet."""

from __future__ import annotations

# --- test_loop_body_ends.py ---

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


# --- test_layer_variants_converge.py ---

from TraceLens.Visualizer.model_explorer_export.merge import (
    _converge_layer_variants_through_an_output_tile,
)


def _node(node_id, namespace, label="op", sources=()):
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [{"key": "output_shape", "value": "[B, S, H] bfloat16"}],
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": "0", "targetNodeInputId": str(i)}
            for i, s in enumerate(sources)
        ],
    }


STACK = "45x_Decoder"


def _outputs(nodes):
    return [
        n
        for n in nodes
        if any(
            a.get("key") == "synthetic" and a.get("value") == "@output"
            for a in n.get("attrs") or []
        )
    ]


class TestTheArmsConverge:
    def test_three_variants_reach_their_consumer_through_one_output_tile(self) -> None:
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("c", f"{STACK}/3x_C"),
            _node("head", "", label="Mean", sources=["a", "b", "c"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        tiles = _outputs(nodes)
        assert tiles
        assert tiles[0]["label"] == "output"
        head = next(n for n in nodes if n["id"] == "head")
        assert [e["sourceNodeId"] for e in head["incomingEdges"]] == [tiles[0]["id"]]
        assert {e["sourceNodeId"] for e in tiles[0]["incomingEdges"]} == {"a", "b", "c"}

    def test_the_join_is_an_output_tile_for_the_consumer(self) -> None:
        """The consumer reads an output tile, the same port every block leaves through."""
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        attrs = {a["key"]: a["value"] for a in _outputs(nodes)[0]["attrs"]}
        assert attrs["synthetic"] == "@output"
        assert "class_name" not in attrs

    def test_loop_ends_name_the_tile_the_consumer_reads(self) -> None:
        """The variant output tiles already name the tensor. The stack tile keeps it."""
        nodes = [
            _node(
                "a",
                f"{STACK}/48x_A",
                label="loop out: hidden_states - 48 of 64 iterations",
            ),
            _node(
                "b",
                f"{STACK}/16x_B",
                label="loop out: hidden_states - 16 of 64 iterations",
            ),
            _node("head", "norm", label="x", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        tile = _outputs(nodes)[0]
        assert tile["label"] == "hidden_states"
        head = next(n for n in nodes if n["id"] == "head")
        assert [e["sourceNodeId"] for e in head["incomingEdges"]] == [tile["id"]]

    def test_two_consumers_of_the_same_arms_share_the_tile(self) -> None:
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("head", "", sources=["a", "b"]),
            _node("tail", "other", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        tiles = _outputs(nodes)
        assert tiles
        for node_id in ("head", "tail"):
            node = next(n for n in nodes if n["id"] == node_id)
            assert [e["sourceNodeId"] for e in node["incomingEdges"]] == [
                tiles[0]["id"]
            ]

    def test_it_is_drawn_beside_the_stack_that_holds_the_variants(self) -> None:
        """The stack box holds the variants. Their join sits one level up."""
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        tile = _outputs(nodes)[0]
        assert tile["namespace"] == ""
        assert tile["id"] == f"{STACK}/@output:output"

        nested = "decoder/45x_Decoder"
        nodes = [
            _node("a", f"{nested}/31x_A"),
            _node("b", f"{nested}/11x_B"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        tile = _outputs(nodes)[0]
        assert tile["namespace"] == "decoder"
        assert tile["id"] == f"{nested}/@output:output"

    def test_it_carries_the_arms_shape(self) -> None:
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        attrs = {a["key"]: a["value"] for a in _outputs(nodes)[0]["attrs"]}
        assert attrs["output_shape"] == "[B, S, H] bfloat16"

    def test_it_is_drawn_like_every_other_output_tile(self) -> None:
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        assert isinstance(_outputs(nodes)[0].get("style"), dict)

    def test_the_innermost_variant_is_the_one_chosen_between(self) -> None:
        """The stack nests in its own ``45x_`` wrapper; that is not a variant."""
        nodes = [
            _node("a", f"{STACK}/31x_A/inner"),
            _node("b", f"{STACK}/11x_B/inner"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        assert _outputs(nodes)


class TestWhatIsNotAnAlternative:
    def test_two_tensors_out_of_one_variant(self) -> None:
        """Two edges from the SAME group are two tensors, not a choice."""
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/31x_A"),
            _node("op", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        assert _outputs(nodes) == []

    def test_a_uniform_stack(self) -> None:
        """GPT-2's layers are all alike: one group, nothing to choose."""
        nodes = [
            _node("a", "12x_Block"),
            _node("x", ""),
            _node("op", "", sources=["a", "x"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        assert _outputs(nodes) == []

    def test_producers_outside_any_repeat_group(self) -> None:
        nodes = [
            _node("a", "enc"),
            _node("b", "dec"),
            _node("op", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        assert _outputs(nodes) == []

    def test_a_single_operand(self) -> None:
        nodes = [_node("a", f"{STACK}/31x_A"), _node("op", "", sources=["a"])]
        _converge_layer_variants_through_an_output_tile(nodes)
        assert _outputs(nodes) == []

    def test_an_unrelated_operand_is_left_alone(self) -> None:
        """Only the alternatives converge; a genuine second input stays put."""
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("w", "", label="weight"),
            _node("op", "", sources=["a", "b", "w"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        op = next(n for n in nodes if n["id"] == "op")
        sources = {e["sourceNodeId"] for e in op["incomingEdges"]}
        assert "w" in sources and len(sources) == 2

    def test_an_empty_graph(self) -> None:
        nodes: list[dict] = []
        _converge_layer_variants_through_an_output_tile(nodes)
        assert nodes == []
