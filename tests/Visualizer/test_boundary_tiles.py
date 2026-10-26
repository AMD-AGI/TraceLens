###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""How a value crosses a group boundary: the tile, its name, and the shape it carries."""

from __future__ import annotations

# --- test_boundary_named_for_its_tensor.py ---

from TraceLens.Visualizer.model_explorer_export.merge import (
    _name_a_boundary_for_its_mirror,
)


def _tile__boundary_named_for_its_tensor(
    node_id: str, label: str, synthetic: str, source: str | None = None
) -> dict:
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
            _tile__boundary_named_for_its_tensor(
                "m/@input_mirror:valid_keys^valid_keys", "valid_keys", "@input_mirror"
            ),
            _tile__boundary_named_for_its_tensor(
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
            _tile__boundary_named_for_its_tensor(
                "m/@output:collapsed", "collapsed", "@output"
            ),
            _tile__boundary_named_for_its_tensor(
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
            _tile__boundary_named_for_its_tensor(
                "m/@input_mirror:valid_keys^valid_keys", "valid_keys", "@input_mirror"
            ),
            _tile__boundary_named_for_its_tensor(
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
        node = _tile__boundary_named_for_its_tensor(
            "box/@input:Slice",
            "Slice",
            "@input",
            "m/@input_mirror:valid_keys^valid_keys",
        )
        node["incomingEdges"].append(
            {"sourceNodeId": "m/@input_mirror:other^other", "sourceNodeOutputId": "0"}
        )
        nodes = [
            _tile__boundary_named_for_its_tensor(
                "m/@input_mirror:valid_keys^valid_keys", "valid_keys", "@input_mirror"
            ),
            _tile__boundary_named_for_its_tensor(
                "m/@input_mirror:other^other", "other", "@input_mirror"
            ),
            node,
        ]
        _name_a_boundary_for_its_mirror(nodes)
        assert _named(nodes, "box/@input:Slice") == "Slice"


# --- test_boundary_shape_carry.py ---

from TraceLens.Visualizer.model_explorer_export.merge import (
    _carry_producer_shape_to_boundary_tiles,
)


def _tile__boundary_shape_carry(node_id, sources=(), synthetic="@input", **attrs):
    node = {
        "id": node_id,
        "label": node_id.rsplit(":", 1)[-1],
        "attrs": [{"key": "synthetic", "value": synthetic}] if synthetic else [],
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": "0"} for s in sources
        ],
    }
    node["attrs"].extend({"key": k, "value": v} for k, v in attrs.items())
    return node


def _shape(nodes, node_id):
    node = next(n for n in nodes if n["id"] == node_id)
    return next(
        (a["value"] for a in node.get("attrs", []) if a["key"] == "output_shape"), None
    )


class TestTheShapeTravelsDownTheChain:
    def test_one_level(self) -> None:
        nodes = [
            _tile__boundary_shape_carry(
                "mirror", synthetic="@input_mirror", output_shape="[B, S] bool"
            ),
            _tile__boundary_shape_carry(
                "outer/@input:attention_mask", sources=["mirror"]
            ),
        ]
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert _shape(nodes, "outer/@input:attention_mask") == "[B, S] bool"

    def test_every_level_of_a_chain(self) -> None:
        """Run to a fixed point, or only the outermost would be filled."""
        nodes = [
            _tile__boundary_shape_carry(
                "mirror", synthetic="@input_mirror", output_shape="[B, S] bool"
            ),
            _tile__boundary_shape_carry("a/@input:attention_mask", sources=["mirror"]),
            _tile__boundary_shape_carry(
                "a/b/@input:attention_mask", sources=["a/@input:attention_mask"]
            ),
            _tile__boundary_shape_carry(
                "a/b/c/@input:attention_mask", sources=["a/b/@input:attention_mask"]
            ),
        ]
        _carry_producer_shape_to_boundary_tiles(nodes)
        for node_id in (
            "a/@input:attention_mask",
            "a/b/@input:attention_mask",
            "a/b/c/@input:attention_mask",
        ):
            assert _shape(nodes, node_id) == "[B, S] bool", node_id

    def test_the_dtype_comes_with_it(self) -> None:
        nodes = [
            _tile__boundary_shape_carry(
                "producer",
                synthetic="@input_mirror",
                output_shape="[B, S] bool",
                output_dtype="bool",
            ),
            _tile__boundary_shape_carry("outer/@input:mask", sources=["producer"]),
        ]
        _carry_producer_shape_to_boundary_tiles(nodes)
        tile = next(n for n in nodes if n["id"] == "outer/@input:mask")
        assert any(
            a["key"] == "output_dtype" and a["value"] == "bool" for a in tile["attrs"]
        )


class TestWhatIsLeftAlone__boundary_shape_carry:
    def test_a_stale_boundary_shape_is_replaced_by_its_producer(self) -> None:
        nodes = [
            _tile__boundary_shape_carry(
                "producer",
                synthetic="@input_mirror",
                output_shape="[B, 1, S, S] bfloat16",
                output_dtype="bfloat16",
            ),
            _tile__boundary_shape_carry(
                "outer/@input:mask",
                sources=["producer"],
                output_shape="[B, S] int64",
                output_dtype="int64",
            ),
        ]
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert _shape(nodes, "outer/@input:mask") == "[B, 1, S, S] bfloat16"
        tile = next(n for n in nodes if n["id"] == "outer/@input:mask")
        assert (
            next(a["value"] for a in tile["attrs"] if a["key"] == "output_dtype")
            == "bfloat16"
        )

    def test_a_tile_with_several_producers(self) -> None:
        """Not a pass-through; it would have to pick one, so it says nothing."""
        nodes = [
            _tile__boundary_shape_carry(
                "p1", synthetic="@input_mirror", output_shape="[B, S] bool"
            ),
            _tile__boundary_shape_carry(
                "p2", synthetic="@input_mirror", output_shape="[B, S, H] bfloat16"
            ),
            _tile__boundary_shape_carry("outer/@input:mask", sources=["p1", "p2"]),
        ]
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert _shape(nodes, "outer/@input:mask") is None

    def test_a_producer_that_is_not_in_the_graph(self) -> None:
        nodes = [_tile__boundary_shape_carry("outer/@input:mask", sources=["absent"])]
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert _shape(nodes, "outer/@input:mask") is None

    def test_a_producer_with_no_shape_of_its_own(self) -> None:
        nodes = [
            _tile__boundary_shape_carry("producer", synthetic="@input_mirror"),
            _tile__boundary_shape_carry("outer/@input:mask", sources=["producer"]),
        ]
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert _shape(nodes, "outer/@input:mask") is None

    def test_a_tile_with_no_producer(self) -> None:
        nodes = [_tile__boundary_shape_carry("outer/@input:mask")]
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert _shape(nodes, "outer/@input:mask") is None

    def test_a_node_that_is_not_a_boundary(self) -> None:
        """An op computes its own shape; it does not inherit one."""
        nodes = [
            _tile__boundary_shape_carry(
                "producer", synthetic="@input_mirror", output_shape="[B, S] bool"
            ),
            _tile__boundary_shape_carry(
                "outer/@op_add", sources=["producer"], synthetic=None
            ),
        ]
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert _shape(nodes, "outer/@op_add") is None

    def test_an_empty_graph(self) -> None:
        nodes: list[dict] = []
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert nodes == []


# --- test_elided_branch_boundary_name.py ---

from TraceLens.ModelUtils.ast_analyze import PHI_CLASS, PHI_LABEL
from TraceLens.Visualizer.model_explorer_export.merge import (
    _elide_noop_merges_and_casts,
)


def _edge(source: str) -> dict[str, str]:
    return {
        "sourceNodeId": source,
        "sourceNodeOutputId": "0",
        "targetNodeInputId": "0",
    }


def test_an_input_named_for_a_removed_branch_takes_the_variable() -> None:
    nodes = [
        {"id": "expand", "label": "Expand", "namespace": "", "attrs": []},
        {
            "id": "phi",
            "label": PHI_LABEL,
            "namespace": "",
            "attrs": [
                {"key": "class_name", "value": PHI_CLASS},
                {
                    "key": "details",
                    "value": "merge: position_ids is None; merges: position_ids",
                },
            ],
            "incomingEdges": [_edge("expand")],
        },
        {
            "id": "mask/@input:Mutually exclusive branch",
            "label": PHI_LABEL,
            "namespace": "create_causal_mask",
            "attrs": [{"key": "synthetic", "value": "@input"}],
            "incomingEdges": [_edge("phi")],
        },
    ]
    _elide_noop_merges_and_casts(nodes)
    assert [node["id"] for node in nodes] == [
        "expand",
        "mask/@input:Mutually exclusive branch",
    ]
    tile = nodes[1]
    assert tile["label"] == "position_ids"
    assert tile["incomingEdges"][0]["sourceNodeId"] == "expand"


def test_a_sibling_boundary_names_the_tensor_when_the_branch_recorded_none() -> None:
    """The same Expand already enters rotary as ``position_ids``."""
    nodes = [
        {"id": "expand", "label": "Expand", "namespace": "", "attrs": []},
        {
            "id": "phi",
            "label": PHI_LABEL,
            "namespace": "",
            "attrs": [
                {"key": "class_name", "value": PHI_CLASS},
                {"key": "details", "value": "merge: position_ids is None"},
            ],
            "incomingEdges": [_edge("expand")],
        },
        {
            "id": "rotary/@input:position_ids",
            "label": "position_ids",
            "namespace": "rotary_emb",
            "attrs": [{"key": "synthetic", "value": "@input"}],
            "incomingEdges": [_edge("expand")],
        },
        {
            "id": "mask/@input:Mutually exclusive branch",
            "label": PHI_LABEL,
            "namespace": "create_causal_mask",
            "attrs": [{"key": "synthetic", "value": "@input"}],
            "incomingEdges": [_edge("phi")],
        },
    ]
    _elide_noop_merges_and_casts(nodes)
    mask = next(node for node in nodes if node["id"].startswith("mask/"))
    assert mask["label"] == "position_ids"
    assert mask["incomingEdges"][0]["sourceNodeId"] == "expand"


def test_a_branch_that_is_still_drawn_keeps_its_name() -> None:
    nodes = [
        {"id": "add", "label": "Add", "namespace": "create_causal_mask", "attrs": []},
        {
            "id": "inner_expand",
            "label": "Expand",
            "namespace": "create_causal_mask",
            "attrs": [],
        },
        {
            "id": "phi",
            "label": PHI_LABEL,
            "namespace": "create_causal_mask",
            "attrs": [
                {"key": "class_name", "value": PHI_CLASS},
                {"key": "details", "value": "merge: not use_vmap; merges: mask"},
            ],
            "incomingEdges": [_edge("add"), _edge("inner_expand")],
        },
        {
            "id": "create_causal_mask/@output:Mutually exclusive branch",
            "label": PHI_LABEL,
            "namespace": "create_causal_mask",
            "attrs": [{"key": "synthetic", "value": "@output"}],
            "incomingEdges": [_edge("phi")],
        },
    ]
    _elide_noop_merges_and_casts(nodes)
    assert any(node["id"] == "phi" for node in nodes)
    output = next(
        node
        for node in nodes
        if str(node["id"]).endswith("@output:Mutually exclusive branch")
    )
    assert output["label"] == PHI_LABEL


# --- test_entering_tensor_tiles.py ---

from TraceLens.Visualizer.model_explorer_export.merge import (
    _draw_entering_tensor_tiles_as_inputs,
)
from TraceLens.Visualizer.model_explorer_export.styles import input_port_style

_INPUT_FILL = input_port_style()["backgroundColor"]


def _tile__entering_tensor_tiles(
    node_id: str, namespace: str, sources: list[str], **extra
) -> dict:
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


def _op__entering_tensor_tiles(node_id: str, namespace: str) -> dict:
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
            _op__entering_tensor_tiles("attn/Rearrange", "attn"),
            _tile__entering_tensor_tiles(
                "attn/pipe/q", "attn/pipe", ["attn/Rearrange"]
            ),
        ]
        _draw_entering_tensor_tiles_as_inputs(nodes)
        assert _fill(nodes[1]) == _INPUT_FILL

    def test_a_tile_fed_from_inside_the_box_is_not(self) -> None:
        """An intermediate value keeps the op styling."""
        nodes = [
            _op__entering_tensor_tiles("attn/pipe/CumSum", "attn/pipe"),
            _tile__entering_tensor_tiles(
                "attn/pipe/gk", "attn/pipe", ["attn/pipe/CumSum"]
            ),
        ]
        _draw_entering_tensor_tiles_as_inputs(nodes)
        assert _fill(nodes[1]) != _INPUT_FILL

    def test_a_tile_fed_from_a_nested_box_is_not(self) -> None:
        """Deeper is still inside: the tensor did not enter here."""
        nodes = [
            _op__entering_tensor_tiles("attn/pipe/inner/Op", "attn/pipe/inner"),
            _tile__entering_tensor_tiles(
                "attn/pipe/out", "attn/pipe", ["attn/pipe/inner/Op"]
            ),
        ]
        _draw_entering_tensor_tiles_as_inputs(nodes)
        assert _fill(nodes[1]) != _INPUT_FILL

    def test_a_constant_is_never_a_box_input(self) -> None:
        nodes = [
            _op__entering_tensor_tiles("attn/weight", "attn"),
            _tile__entering_tensor_tiles(
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
            _op__entering_tensor_tiles("attn/src", "attn"),
            _tile__entering_tensor_tiles(
                "attn/pipe/op:external:scale", "attn/pipe", ["attn/src"]
            ),
        ]
        _draw_entering_tensor_tiles_as_inputs(nodes)
        assert _fill(nodes[1]) != _INPUT_FILL

    def test_a_tile_with_no_producer_is_left_alone(self) -> None:
        nodes = [_tile__entering_tensor_tiles("attn/pipe/q", "attn/pipe", [])]
        _draw_entering_tensor_tiles_as_inputs(nodes)
        assert _fill(nodes[0]) != _INPUT_FILL

    def test_mixed_producers_mean_it_was_not_entered_here(self) -> None:
        """One producer inside is enough: the tile is not purely an entry."""
        nodes = [
            _op__entering_tensor_tiles("attn/Rearrange", "attn"),
            _op__entering_tensor_tiles("attn/pipe/Local", "attn/pipe"),
            _tile__entering_tensor_tiles(
                "attn/pipe/q",
                "attn/pipe",
                ["attn/Rearrange", "attn/pipe/Local"],
            ),
        ]
        _draw_entering_tensor_tiles_as_inputs(nodes)
        assert _fill(nodes[2]) != _INPUT_FILL


# --- test_shared_entering_tile.py ---

import pytest

from TraceLens.Visualizer.model_explorer_export.merge import (
    _share_one_tile_per_entering_tensor,
)


def _tile__shared_entering_tile(
    node_id: str, namespace: str, label: str, synthetic: str, sources=()
):
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": synthetic}],
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": "0"} for s in sources
        ],
    }


def _op__shared_entering_tile(node_id: str, namespace: str):
    return {"id": node_id, "label": "Unsqueeze", "namespace": namespace, "attrs": []}


def _sources__shared_entering_tile(node) -> list[str]:
    return [str(e["sourceNodeId"]) for e in node.get("incomingEdges") or []]


class TestTheFold:
    def test_every_box_reads_one_shared_tile(self) -> None:
        nodes = [
            _op__shared_entering_tile("op", ""),
            _tile__shared_entering_tile(
                "@input_mirror:position_ids^Nx",
                "",
                "position_ids",
                "@input_mirror",
                ["op"],
            ),
            _tile__shared_entering_tile(
                "rotary/@input:position_ids", "rotary", "position_ids", "@input", ["op"]
            ),
            _tile__shared_entering_tile(
                "mask/@input:Unsqueeze", "mask", "Unsqueeze", "@input", ["op"]
            ),
        ]
        _share_one_tile_per_entering_tensor(nodes)
        shared = "@input_mirror:position_ids^Nx"
        assert _sources__shared_entering_tile(nodes[2]) == [shared]
        assert _sources__shared_entering_tile(nodes[3]) == [shared]
        assert _sources__shared_entering_tile(nodes[1]) == ["op"]

    def test_a_boundary_takes_the_name_of_the_tensor(self) -> None:
        """Not the name of the op that produced it."""
        nodes = [
            _op__shared_entering_tile("op", ""),
            _tile__shared_entering_tile(
                "@input_mirror:position_ids^Nx",
                "",
                "position_ids",
                "@input_mirror",
                ["op"],
            ),
            _tile__shared_entering_tile(
                "mask/@input:Unsqueeze", "mask", "Unsqueeze", "@input", ["op"]
            ),
        ]
        _share_one_tile_per_entering_tensor(nodes)
        assert nodes[2]["label"] == "position_ids"

    def test_a_lone_consumer_is_left_alone(self) -> None:
        nodes = [
            _op__shared_entering_tile("op", ""),
            _tile__shared_entering_tile(
                "rotary/@input:position_ids", "rotary", "position_ids", "@input", ["op"]
            ),
        ]
        _share_one_tile_per_entering_tensor(nodes)
        assert _sources__shared_entering_tile(nodes[1]) == ["op"]

    def test_nothing_is_rerouted_when_no_tile_sits_above_the_producer(self) -> None:
        """Rerouting into a box that the producer is not inside would be a cycle."""
        nodes = [
            _op__shared_entering_tile("outer/op", "outer"),
            _tile__shared_entering_tile("a/@input:x", "a", "x", "@input", ["outer/op"]),
            _tile__shared_entering_tile("b/@input:x", "b", "x", "@input", ["outer/op"]),
        ]
        _share_one_tile_per_entering_tensor(nodes)
        assert _sources__shared_entering_tile(nodes[1]) == ["outer/op"]
        assert _sources__shared_entering_tile(nodes[2]) == ["outer/op"]

    def test_a_sibling_does_not_read_another_call_s_input_mirror(self) -> None:
        """The saved residual and the next norm both read the layer input.

        The gate's mirror is that call's own port. Folding the norm onto it
        draws the norm as though the gate fed it.
        """
        nodes = [
            _op__shared_entering_tile("decoder/@input", "62x_Layer"),
            _tile__shared_entering_tile(
                "decoder/gate/@input_mirror^residual",
                "62x_Layer",
                "residual",
                "@input_mirror",
                ["decoder/@input"],
            ),
            _tile__shared_entering_tile(
                "decoder/norm/@input",
                "62x_Layer/norm",
                "x",
                "@input",
                ["decoder/@input"],
            ),
        ]
        _share_one_tile_per_entering_tensor(nodes)
        assert _sources__shared_entering_tile(nodes[1]) == ["decoder/@input"]
        assert _sources__shared_entering_tile(nodes[2]) == ["decoder/@input"]
        assert nodes[2]["label"] == "x"

    def test_separate_output_ports_are_separate_tensors(self) -> None:
        nodes = [
            _op__shared_entering_tile("op", ""),
            _tile__shared_entering_tile(
                "@input_mirror:cos^Nx", "", "cos", "@input_mirror", ["op"]
            ),
            _tile__shared_entering_tile(
                "box/@input:sin", "box", "sin", "@input", ["op"]
            ),
        ]
        nodes[2]["incomingEdges"][0]["sourceNodeOutputId"] = "1"
        _share_one_tile_per_entering_tensor(nodes)
        assert _sources__shared_entering_tile(nodes[2]) == ["op"]


@pytest.mark.parametrize(
    "model_id",
    ["MiniMaxAI/MiniMax-M3", "deepseek-ai/DeepSeek-V4-Flash"],
)
def test_position_ids_enters_every_box_through_one_tile(model_graph_nodes, model_id):
    nodes = model_graph_nodes(model_id)
    tiles = [n for n in nodes if (n.get("label") or "") == "position_ids"]
    assert len(tiles) == 1, [str(n["id"]) for n in tiles]
    sources = [e["sourceNodeId"] for e in tiles[0].get("incomingEdges") or []]
    assert len(sources) == 1, sources
    source = next(n for n in nodes if n["id"] == sources[0])
    assert (source.get("label") or "") == "Unsqueeze", source.get("label")


# --- test_mirror_name_qualification.py ---

from TraceLens.Visualizer.model_explorer_export.merge import (
    _qualify_colliding_mirrors,
)


def _node__mirror_name_qualification(
    node_id, label, namespace, synthetic=None, sources=()
):
    node = {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "incomingEdges": [
            {"sourceNodeId": source, "sourceNodeOutputId": "0"} for source in sources
        ],
    }
    if synthetic:
        node["attrs"] = [{"key": "synthetic", "value": synthetic}]
    return node


def test_two_output_mirrors_name_the_block_they_left() -> None:
    nodes = [
        _node__mirror_name_qualification(
            "layer/attn_hc/@output:collapsed", "collapsed", "layer/attn_hc", "@output"
        ),
        _node__mirror_name_qualification(
            "layer/ffn_hc/@output:collapsed", "collapsed", "layer/ffn_hc", "@output"
        ),
        _node__mirror_name_qualification(
            "layer/attn_hc/@output:collapsed^collapsed",
            "collapsed",
            "layer",
            "@output_mirror",
            ["layer/attn_hc/@output:collapsed"],
        ),
        _node__mirror_name_qualification(
            "layer/ffn_hc/@output:collapsed^collapsed",
            "collapsed",
            "layer",
            "@output_mirror",
            ["layer/ffn_hc/@output:collapsed"],
        ),
    ]

    _qualify_colliding_mirrors(nodes)

    assert [n["label"] for n in nodes] == [
        "collapsed",
        "collapsed",
        "attn_hc.collapsed",
        "ffn_hc.collapsed",
    ]


def test_input_mirrors_name_the_block_they_enter() -> None:
    nodes = [
        _node__mirror_name_qualification("attn/@input:x", "x", "attn", "@input"),
        _node__mirror_name_qualification(
            "attn/rope_a/@input_mirror:x^x", "x", "attn", "@input_mirror"
        ),
        _node__mirror_name_qualification(
            "attn/rope_a/@input",
            "x",
            "attn/rope_a",
            "@input",
            ["attn/rope_a/@input_mirror:x^x"],
        ),
        _node__mirror_name_qualification(
            "attn/rope_b/@input_mirror:x^x", "x", "attn", "@input_mirror"
        ),
        _node__mirror_name_qualification(
            "attn/rope_b/@input",
            "x",
            "attn/rope_b",
            "@input",
            ["attn/rope_b/@input_mirror:x^x"],
        ),
    ]

    _qualify_colliding_mirrors(nodes)

    assert nodes[0]["label"] == "x", "the block's own declaration keeps the plain name"
    assert nodes[1]["label"] == "rope_a.x"
    assert nodes[3]["label"] == "rope_b.x"


def test_a_name_used_once_is_left_alone() -> None:
    nodes = [
        _node__mirror_name_qualification(
            "layer/attn_hc/@output:collapsed", "collapsed", "layer/attn_hc", "@output"
        ),
        _node__mirror_name_qualification(
            "layer/attn_hc/@output:collapsed^collapsed",
            "collapsed",
            "layer",
            "@output_mirror",
            ["layer/attn_hc/@output:collapsed"],
        ),
    ]

    _qualify_colliding_mirrors(nodes)

    assert [n["label"] for n in nodes] == ["collapsed", "collapsed"]


def test_qualifying_that_would_not_settle_it_is_declined() -> None:
    """Two mirrors of ONE block under one name gain nothing from its name."""
    nodes = [
        _node__mirror_name_qualification(
            "layer/hc/@output:a", "slot", "layer/hc", "@output"
        ),
        _node__mirror_name_qualification(
            "layer/hc/@output:b", "slot", "layer/hc", "@output"
        ),
        _node__mirror_name_qualification(
            "layer/hc/@output:a^a",
            "slot",
            "layer",
            "@output_mirror",
            ["layer/hc/@output:a"],
        ),
        _node__mirror_name_qualification(
            "layer/hc/@output:b^b",
            "slot",
            "layer",
            "@output_mirror",
            ["layer/hc/@output:b"],
        ),
    ]

    _qualify_colliding_mirrors(nodes)

    assert [n["label"] for n in nodes[2:]] == ["slot", "slot"]


def test_the_port_label_follows_the_node_label() -> None:
    mirror = _node__mirror_name_qualification(
        "layer/attn_hc/@output:collapsed^collapsed",
        "collapsed",
        "layer",
        "@output_mirror",
        ["layer/attn_hc/@output:collapsed"],
    )
    mirror["outputsMetadata"] = [
        {"id": "collapsed", "attrs": [{"key": "port_label", "value": "collapsed"}]}
    ]
    nodes = [
        _node__mirror_name_qualification(
            "layer/attn_hc/@output:collapsed", "collapsed", "layer/attn_hc", "@output"
        ),
        _node__mirror_name_qualification(
            "layer/ffn_hc/@output:collapsed", "collapsed", "layer/ffn_hc", "@output"
        ),
        mirror,
        _node__mirror_name_qualification(
            "layer/ffn_hc/@output:collapsed^collapsed",
            "collapsed",
            "layer",
            "@output_mirror",
            ["layer/ffn_hc/@output:collapsed"],
        ),
    ]

    _qualify_colliding_mirrors(nodes)

    # The label a reader sees is qualified; the port ID an edge cites is not.
    assert mirror["outputsMetadata"][0]["attrs"][0]["value"] == "attn_hc.collapsed"
    assert mirror["outputsMetadata"][0]["id"] == "collapsed"


def _op__mirror_name_qualification(node_id, attr_name, namespace):
    return {
        "id": node_id,
        "label": "Linear",
        "namespace": namespace,
        "attrs": [{"key": "attr_name", "value": attr_name}],
        "incomingEdges": [],
    }


def test_kernel_operand_ports_name_the_step_that_fed_them() -> None:
    """Three calls to one helper bind ``x`` three times, on q, k and v."""
    nodes = [
        _op__mirror_name_qualification("attn/q_proj", "q_proj", "attn"),
        _op__mirror_name_qualification("attn/k_proj", "k_proj", "attn"),
        _op__mirror_name_qualification("attn/v_proj", "v_proj", "attn"),
        _node__mirror_name_qualification(
            "attn/@kernel_in:15:x", "x", "attn", "@kernel_port_in", ["attn/q_proj"]
        ),
        _node__mirror_name_qualification(
            "attn/@kernel_in:16:x", "x", "attn", "@kernel_port_in", ["attn/k_proj"]
        ),
        _node__mirror_name_qualification(
            "attn/@kernel_in:17:x", "x", "attn", "@kernel_port_in", ["attn/v_proj"]
        ),
    ]

    _qualify_colliding_mirrors(nodes)

    assert [n["label"] for n in nodes[3:]] == ["q_proj.x", "k_proj.x", "v_proj.x"]


def test_one_tensor_at_several_operand_slots_keeps_its_name() -> None:
    """Ports 15/16/17 all hold the SAME cu_seqlens -- nothing to tell apart."""
    nodes = [
        _op__mirror_name_qualification("attn/unpad", "unpad", "attn"),
        _node__mirror_name_qualification(
            "attn/@kernel_in:15:cu_seqlens",
            "cu_seqlens",
            "attn",
            "@kernel_port_in",
            ["attn/unpad"],
        ),
        _node__mirror_name_qualification(
            "attn/@kernel_in:16:cu_seqlens",
            "cu_seqlens",
            "attn",
            "@kernel_port_in",
            ["attn/unpad"],
        ),
    ]

    _qualify_colliding_mirrors(nodes)

    assert [n["label"] for n in nodes[1:]] == ["cu_seqlens", "cu_seqlens"]


# --- test_section_external_inputs.py ---

from TraceLens.Visualizer.model_explorer_export.merge import (
    _coalesce_duplicate_section_inputs,
    _drop_mismatched_boundary_operands,
    _expose_unfed_section_inputs,
)


def _input(node_id: str, label: str, namespace: str):
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [
            {"key": "synthetic", "value": "@input"},
            {"key": "boundary_input", "value": label},
            {"key": "output_shape", "value": "[R, C] int64"},
        ],
    }


class TestOneBoundaryPerSectionParameter:
    def test_duplicate_requests_are_coalesced(self) -> None:
        first = _input("section/@input:grid", "grid", "section")
        second = _input("section/@input:grid", "grid", "section")
        first["incomingEdges"] = [
            {
                "sourceNodeId": "a",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "grid",
            }
        ]
        second["incomingEdges"] = [
            {
                "sourceNodeId": "b",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "grid",
            }
        ]
        nodes = [first, second]
        _coalesce_duplicate_section_inputs(nodes)
        assert len(nodes) == 1
        assert {edge["sourceNodeId"] for edge in nodes[0]["incomingEdges"]} == {
            "a",
            "b",
        }


class TestAnOuterSectionParameterIsAModelInput:
    def test_an_unfed_boundary_gets_a_root_source(self) -> None:
        target = _input("section/@input:grid", "grid", "section")
        nodes = [target]
        _expose_unfed_section_inputs(nodes)
        source = next(node for node in nodes if node["id"] == "@input:grid")
        assert source["namespace"] == ""
        assert any(
            attr == {"key": "output_shape", "value": "[R, C] int64"}
            for attr in source["attrs"]
        )
        assert target["incomingEdges"][0]["sourceNodeId"] == "@input:grid"

    def test_an_unverified_external_parameter_does_not_inherit_entry_shape(
        self,
    ) -> None:
        class NoObservedParameterShape:
            @staticmethod
            def boundary_input_spec(_param: str, _namespace: str):
                return None

        target = _input("section/@input:grid", "grid", "section")
        target["outputsMetadata"] = [
            {
                "id": "0",
                "attrs": [
                    {"key": "shape", "value": "[B, S, H] bfloat16"},
                    {"key": "dtype", "value": "bfloat16"},
                ],
            }
        ]
        nodes = [target]
        _expose_unfed_section_inputs(nodes, NoObservedParameterShape())
        source = next(node for node in nodes if node["id"] == "@input:grid")
        assert not any(
            attr["key"] == "output_shape" for attr in source.get("attrs", [])
        )
        assert not any(
            attr["key"] == "output_shape" for attr in target.get("attrs", [])
        )
        assert not any(
            attr["key"] == "shape"
            for metadata in target.get("outputsMetadata", [])
            for attr in metadata.get("attrs", [])
        )

    def test_an_unknown_external_shape_is_cleared_through_nested_boundaries(
        self,
    ) -> None:
        class NoObservedParameterShape:
            @staticmethod
            def boundary_input_spec(_param: str, _namespace: str):
                return None

        outer = _input("section/@input:grid", "grid", "section")
        inner = _input("section/helper/@input:grid", "grid", "section/helper")
        inner["incomingEdges"] = [
            {
                "sourceNodeId": "section/@input:grid",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "grid",
            }
        ]
        nodes = [outer, inner]
        _expose_unfed_section_inputs(nodes, NoObservedParameterShape())
        assert not any(attr["key"] == "output_shape" for attr in inner.get("attrs", []))

    def test_an_internal_module_boundary_stays_a_diagnostic(self) -> None:
        target = _input("section/helper/@input:grid", "grid", "section/helper")
        nodes = [target]
        _expose_unfed_section_inputs(nodes)
        assert nodes == [target]
        assert not target.get("incomingEdges")

    def test_an_already_fed_boundary_is_left_alone(self) -> None:
        target = _input("section/@input:grid", "grid", "section")
        target["incomingEdges"] = [
            {
                "sourceNodeId": "producer",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "grid",
            }
        ]
        nodes = [target]
        _expose_unfed_section_inputs(nodes)
        assert nodes == [target]
        assert target["incomingEdges"][0]["sourceNodeId"] == "producer"


class TestAnOpReadsItsDeclaredBoundaryParameter:
    def test_an_overfull_binary_op_drops_the_sibling_parameter(self) -> None:
        left = {"id": "left", "label": "Left", "attrs": []}
        wanted = _input("frame/@input:wanted", "wanted", "frame")
        sibling = _input("frame/@input:sibling", "sibling", "frame")
        target = {
            "id": "frame/equal",
            "label": "Equal",
            "attrs": [
                {"key": "raw_op", "value": "eq"},
                {"key": "boundary_input", "value": "wanted"},
            ],
            "incomingEdges": [
                {"sourceNodeId": "left", "targetNodeInputId": "0"},
                {"sourceNodeId": "frame/@input:wanted", "targetNodeInputId": "1"},
                {"sourceNodeId": "frame/@input:sibling", "targetNodeInputId": "2"},
            ],
        }
        _drop_mismatched_boundary_operands([left, wanted, sibling, target])
        assert {edge["sourceNodeId"] for edge in target["incomingEdges"]} == {
            "left",
            "frame/@input:wanted",
        }


# --- test_tuple_boundary_split.py ---

from TraceLens.Visualizer.model_explorer_export.merge import (
    _split_tuple_boundary_slots,
)


def _tile__tuple_boundary_split(
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
        _tile__tuple_boundary_split(
            "m/@output:cos", "cos", "@output", shape="[B, D] f32"
        ),
        _tile__tuple_boundary_split(
            "m/@output:sin", "sin", "@output", shape="[B, D] f32"
        ),
        _tile__tuple_boundary_split(
            "box/@input:position_embeddings",
            "position_embeddings",
            "@input",
            [("m/@output:cos", "0"), ("m/@output:sin", "0")],
            shape="[B, S, H] f32",
        ),
    ]
    for index, port in enumerate(consumer_ports):
        nodes.append(
            _tile__tuple_boundary_split(
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


class TestWhatIsLeftAlone__tuple_boundary_split:
    def test_unrelated_producers_are_not_a_tuple(self) -> None:
        """``block_indices`` fed by three ops is a different defect."""
        nodes = [
            _tile__tuple_boundary_split("m/@op_expand", "Expand"),
            _tile__tuple_boundary_split("m/@op_fill", "masked_fill"),
            _tile__tuple_boundary_split(
                "box/@input:block_indices",
                "block_indices",
                "@input",
                [("m/@op_expand", "0"), ("m/@op_fill", "0")],
            ),
            _tile__tuple_boundary_split(
                "box/@op_reader", "Reader", None, [("box/@input:block_indices", "0")]
            ),
        ]
        _split_tuple_boundary_slots(nodes)
        assert "block_indices" in _labels(nodes)

    def test_producers_sharing_a_name_are_not_components(self) -> None:
        """Two producers called ``result`` name no distinct components."""
        nodes = [
            _tile__tuple_boundary_split("m/@output:a", "result", "@output"),
            _tile__tuple_boundary_split("m/@output:b", "result", "@output"),
            _tile__tuple_boundary_split(
                "box/@input:x",
                "x",
                "@input",
                [("m/@output:a", "0"), ("m/@output:b", "0")],
            ),
            _tile__tuple_boundary_split(
                "box/@op_reader", "Reader", None, [("box/@input:x", "0")]
            ),
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
            _tile__tuple_boundary_split(
                "box/inner/@input:position_embeddings",
                "position_embeddings",
                "@input",
                [("box/@input:position_embeddings", "0")],
            )
        )
        # Its own consumers address two components, so it forwards the whole tuple.
        for port in ("0", "1"):
            nodes.append(
                _tile__tuple_boundary_split(
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


class TestARopeHelperReusesTheTensorItsCallerAlreadyNamed:
    def test_the_helper_is_not_given_a_second_copy(self) -> None:
        """The helper reads the attention tile. The tile is not drawn again."""
        nodes = _bundle([])
        nodes[2]["namespace"] = "DeepseekV4Attention"
        nodes.append(
            _tile__tuple_boundary_split(
                "attn/seq:6:@positional_l810_apply_rotary_pos_emb:@/@input:position_embeddings",
                "position_embeddings",
                "@input",
                [("box/@input:position_embeddings", "0")],
            )
        )
        nodes[-1][
            "namespace"
        ] = "DeepseekV4Attention/_positional_l810_apply_rotary_pos_emb"
        for port in ("0", "1"):
            reader = _tile__tuple_boundary_split(
                f"rope/@op_use{port}",
                f"Use{port}",
                None,
                [
                    (
                        "attn/seq:6:@positional_l810_apply_rotary_pos_emb:@/@input:position_embeddings",
                        port,
                    )
                ],
            )
            reader["namespace"] = nodes[-1]["namespace"]
            nodes.append(reader)
        _split_tuple_boundary_slots(nodes)
        assert (
            "attn/seq:6:@positional_l810_apply_rotary_pos_emb:@/@input:position_embeddings.cos"
            not in {node["id"] for node in nodes}
        )
        assert _sources_of(nodes, "rope/@op_use0") == [
            "box/@input:position_embeddings.cos"
        ]
        assert _sources_of(nodes, "rope/@op_use1") == [
            "box/@input:position_embeddings.sin"
        ]

    def test_a_helper_two_boxes_in_keeps_its_own_boundary(self) -> None:
        """The caller's tile is two boxes out. The helper still declares its own.

        Reusing the further tile wires the op straight across both boxes, so
        the inner box renders with an input it never declares.
        """
        nodes = _bundle([])
        nodes[2]["namespace"] = "DecoderLayer"
        helper = (
            "DecoderLayer/MiniMaxM3VLAttention/_positional_l810_apply_rotary_pos_emb"
        )
        nodes.append(
            _tile__tuple_boundary_split(
                "attn/seq:6:@positional_l810_apply_rotary_pos_emb:@/@input:position_embeddings",
                "position_embeddings",
                "@input",
                [("box/@input:position_embeddings", "0")],
            )
        )
        nodes[-1]["namespace"] = helper
        for port in ("0", "1"):
            reader = _tile__tuple_boundary_split(
                f"rope/@op_use{port}",
                f"Use{port}",
                None,
                [
                    (
                        "attn/seq:6:@positional_l810_apply_rotary_pos_emb:@/@input:position_embeddings",
                        port,
                    )
                ],
            )
            reader["namespace"] = helper
            nodes.append(reader)
        _split_tuple_boundary_slots(nodes)
        assert _sources_of(nodes, "rope/@op_use0") == [
            "attn/seq:6:@positional_l810_apply_rotary_pos_emb:@/@input:position_embeddings.cos"
        ]

    def test_a_real_nested_module_still_names_its_own_boundary(self) -> None:
        nodes = _bundle([])
        nodes[2]["namespace"] = "DeepseekV4Attention"
        nodes.append(
            _tile__tuple_boundary_split(
                "compressor/@input:position_embeddings",
                "position_embeddings",
                "@input",
                [("box/@input:position_embeddings", "0")],
            )
        )
        nodes[-1]["namespace"] = "DeepseekV4Attention/DeepseekV4CSACompressor"
        for port in ("0", "1"):
            reader = _tile__tuple_boundary_split(
                f"compressor/@op_use{port}",
                f"Use{port}",
                None,
                [("compressor/@input:position_embeddings", port)],
            )
            reader["namespace"] = nodes[-1]["namespace"]
            nodes.append(reader)
        _split_tuple_boundary_slots(nodes)
        assert _sources_of(nodes, "compressor/@op_use0") == [
            "compressor/@input:position_embeddings.cos"
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
            _tile__tuple_boundary_split(
                "box/inner/@input:position_embeddings",
                "position_embeddings",
                "@input",
                [("box/@input:position_embeddings", port) for port in edges],
            )
        )
        for port in ("0", "1"):
            nodes.append(
                _tile__tuple_boundary_split(
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


# --- test_published_port_shapes.py ---

from TraceLens.ModelUtils.extract import ArchitectureSpec
from TraceLens.ModelUtils.parallel_plan import (
    ParallelDegree,
    annotate_parallel_nodes,
    layout_from_plans,
    parse_class_plans,
)
from TraceLens.Visualizer.model_explorer_export.merge import (
    _publish_declared_shapes_as_output_ports,
    publish_stated_shapes_and_check,
)
from TraceLens.Visualizer.model_explorer_export.shapes import stamp_full_shapes


def _node__published_port_shapes(
    node_id, shape=None, dtype=None, ports=None, sources=()
):
    attrs = []
    if shape:
        attrs.append({"key": "output_shape", "value": shape})
    if dtype:
        attrs.append({"key": "output_dtype", "value": dtype})
    node = {
        "id": node_id,
        "label": node_id,
        "namespace": "box",
        "attrs": attrs,
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": p} for s, p in sources
        ],
    }
    if ports is not None:
        node["outputsMetadata"] = ports
    return node


def _ports(node):
    return node.get("outputsMetadata") or []


def _attrs(port):
    return {attr["key"]: attr["value"] for attr in port.get("attrs", [])}


class TestAStatedShapeIsPublished:
    def test_the_shape_reaches_a_port(self) -> None:
        nodes = [
            _node__published_port_shapes(
                "v", shape="[B, S, 96, 128] bfloat16", dtype="bfloat16"
            )
        ]
        _publish_declared_shapes_as_output_ports(nodes)
        assert _attrs(_ports(nodes[0])[0]) == {
            "shape": "[B, S, 96, 128] bfloat16",
            "dtype": "bfloat16",
        }

    def test_a_shape_with_no_dtype_still_publishes(self) -> None:
        nodes = [_node__published_port_shapes("v", shape="[B, S]")]
        _publish_declared_shapes_as_output_ports(nodes)
        assert _attrs(_ports(nodes[0])[0]) == {"shape": "[B, S]"}

    def test_a_node_stating_nothing_publishes_nothing(self) -> None:
        """Inventing a shape is worse than drawing ``?``."""
        nodes = [_node__published_port_shapes("v")]
        _publish_declared_shapes_as_output_ports(nodes)
        assert _ports(nodes[0]) == []

    def test_what_a_node_already_published_is_left_alone(self) -> None:
        """Several ports is where several ports come from: never overwrite."""
        existing = [
            {"id": "cos", "attrs": [{"key": "shape", "value": "[B, S, D]"}]},
            {"id": "sin", "attrs": [{"key": "shape", "value": "[B, S, D]"}]},
        ]
        nodes = [
            _node__published_port_shapes("rotary", shape="[B, S, D]", ports=existing)
        ]
        _publish_declared_shapes_as_output_ports(nodes)
        assert _ports(nodes[0]) == existing

    def test_the_port_takes_the_id_its_readers_dock_on(self) -> None:
        """A tile read at a named slot keeps that name, or nothing matches it."""
        nodes = [
            _node__published_port_shapes("tile", shape="[B, S]"),
            _node__published_port_shapes("reader", sources=[("tile", "result")]),
        ]
        _publish_declared_shapes_as_output_ports(nodes)
        assert _ports(nodes[0])[0]["id"] == "result"

    def test_readers_docking_on_different_slots_fall_back_to_zero(self) -> None:
        nodes = [
            _node__published_port_shapes("tile", shape="[B, S]"),
            _node__published_port_shapes("a", sources=[("tile", "cos")]),
            _node__published_port_shapes("b", sources=[("tile", "sin")]),
        ]
        _publish_declared_shapes_as_output_ports(nodes)
        assert _ports(nodes[0])[0]["id"] == "0"

    def test_a_tile_nothing_reads_publishes_on_zero(self) -> None:
        nodes = [_node__published_port_shapes("tile", shape="[B, S]")]
        _publish_declared_shapes_as_output_ports(nodes)
        assert _ports(nodes[0])[0]["id"] == "0"

    def test_an_empty_graph(self) -> None:
        nodes: list[dict] = []
        _publish_declared_shapes_as_output_ports(nodes)
        assert nodes == []


def test_the_full_shape_is_recorded_even_when_the_graph_is_not_split() -> None:
    node = _node__published_port_shapes(
        "linear",
        shape="[B, S, 8192] bfloat16",
        ports=[
            {"id": "0", "attrs": [{"key": "shape", "value": "[B, S, 8192] bfloat16"}]}
        ],
    )
    stamp_full_shapes([node])
    assert {item["key"]: item["value"] for item in node["attrs"]}["full_shape"] == (
        "[B, S, 8192] bfloat16"
    )
    assert node["outputsMetadata"][0]["attrs"][1] == {
        "key": "full_shape",
        "value": "[B, S, 8192] bfloat16",
    }
    # A second pass must not replace a full shape already saved.
    node["attrs"][0]["value"] = "[B, S, 4096] bfloat16"
    stamp_full_shapes([node])
    assert {item["key"]: item["value"] for item in node["attrs"]}["full_shape"] == (
        "[B, S, 8192] bfloat16"
    )


def test_a_gather_added_after_the_build_publishes_the_shape_it_states() -> None:
    """The edge leaving a gather reads the port. With no port the viewer draws ``?``."""
    source = """
class TextConfig:
    model_type = "demo_text"
    base_model_tp_plan = {
        "layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
    }
    base_model_pp_plan = None
    base_model_ep_plan = None
"""
    spec = ArchitectureSpec(
        name="demo",
        model_type="demo",
        num_hidden_layers=32,
        raw_config={"model_type": "demo", "architectures": []},
    )
    layout = layout_from_plans(
        parse_class_plans(source), spec, ParallelDegree(tp=4), spec.raw_config
    )
    producer = {
        "id": "block/linear_attn/in_proj_qkv",
        "label": "in_proj_qkv",
        "namespace": "",
        "attrs": [
            {"key": "attr_name", "value": "in_proj_qkv"},
            {"key": "output_shape", "value": "[B, S, 8192] bfloat16"},
        ],
        "outputsMetadata": [
            {
                "id": "0",
                "attrs": [{"key": "shape", "value": "[B, S, 8192] bfloat16"}],
            }
        ],
        "incomingEdges": [
            {
                "sourceNodeId": "@input",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "0",
            }
        ],
    }
    graph = {
        "nodes": [
            {
                "id": "@input",
                "label": "hidden_states",
                "namespace": "",
                "attrs": [{"key": "synthetic", "value": "@input"}],
            },
            producer,
            {
                "id": "view",
                "label": "view",
                "namespace": "",
                "attrs": [
                    {"key": "synthetic", "value": "@output"},
                    {"key": "output_shape", "value": "[B, S, 64, 128] bfloat16"},
                ],
                "outputsMetadata": [
                    {
                        "id": "0",
                        "attrs": [
                            {"key": "shape", "value": "[B, S, 64, 128] bfloat16"}
                        ],
                    }
                ],
                "incomingEdges": [
                    {
                        "sourceNodeId": producer["id"],
                        "sourceNodeOutputId": "0",
                        "targetNodeInputId": "0",
                    }
                ],
            },
        ]
    }
    annotate_parallel_nodes(graph, layout)
    gather = next(node for node in graph["nodes"] if node["label"] == "gather")
    assert gather.get("outputsMetadata") is None
    warnings = publish_stated_shapes_and_check(graph, label="parallel")
    port = {
        item["key"]: item["value"] for item in gather["outputsMetadata"][0]["attrs"]
    }
    assert port["shape"] == "[B, S, 8192] bfloat16"
    assert port["full_shape"] == "[B, S, 8192] bfloat16"
    assert gather["outputsMetadata"][0]["id"] == "0"
    assert warnings == []


# --- test_nullary_op_operands.py ---

from TraceLens.Visualizer.model_explorer_export.merge import (
    _dock_nullary_ops_on_what_sizes_them,
)


def _op__nullary_op_operands(
    node_id: str, raw_op: str, sources: list[str], sized_by: str | None = "grid_thw"
) -> dict:
    attrs = [{"key": "raw_op", "value": raw_op}]
    if sized_by:
        attrs.append({"key": "boundary_input", "value": sized_by})
    return {
        "id": node_id,
        "label": raw_op.title(),
        "namespace": node_id.rsplit("/", 1)[0],
        "attrs": attrs,
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": "0"} for s in sources
        ],
    }


def _boundary(node_id: str, label: str) -> dict:
    return {
        "id": node_id,
        "label": label,
        "namespace": node_id.rsplit("/", 1)[0],
        "attrs": [{"key": "synthetic", "value": "@input"}],
        "incomingEdges": [],
    }


def _sources__nullary_op_operands(nodes: list[dict], node_id: str) -> list[str]:
    node = next(n for n in nodes if n["id"] == node_id)
    return [str(e["sourceNodeId"]) for e in node.get("incomingEdges") or []]


class TestANullaryOpReadsWhatSizesIt:
    def test_arange_leaves_the_chain_for_the_grid(self) -> None:
        nodes = [
            _boundary("f/@input:hidden_states", "hidden_states"),
            _boundary("f/@input:grid_thw", "grid_thw"),
            _op__nullary_op_operands(
                "f/@op_arange", "arange", ["f/@input:hidden_states"]
            ),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources__nullary_op_operands(nodes, "f/@op_arange") == [
            "f/@input:grid_thw"
        ]

    def test_it_is_not_left_rootless(self) -> None:
        """A range sized by the model's data DEPENDS on that data."""
        nodes = [
            _boundary("f/@input:hidden_states", "hidden_states"),
            _boundary("f/@input:grid_thw", "grid_thw"),
            _op__nullary_op_operands(
                "f/@op_arange", "arange", ["f/@input:hidden_states"]
            ),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources__nullary_op_operands(nodes, "f/@op_arange")

    def test_a_carrier_one_level_up_is_reached(self) -> None:
        nodes = [
            _boundary("f/@input:grid_thw", "grid_thw"),
            _boundary("f/inner/@input:hidden_states", "hidden_states"),
            _op__nullary_op_operands(
                "f/inner/@op_arange", "arange", ["f/inner/@input:hidden_states"]
            ),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources__nullary_op_operands(nodes, "f/inner/@op_arange") == [
            "f/@input:grid_thw"
        ]

    def test_an_op_naming_nothing_is_left_alone(self) -> None:
        """Without a recorded parameter there is nothing to move it to."""
        nodes = [
            _boundary("f/@input:x", "x"),
            _op__nullary_op_operands(
                "f/@op_arange", "arange", ["f/@input:x"], sized_by=None
            ),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources__nullary_op_operands(nodes, "f/@op_arange") == ["f/@input:x"]


class TestWhatIsLeftAlone__nullary_op_operands:
    def test_an_op_that_does_take_a_tensor_is_untouched(self) -> None:
        nodes = [
            _boundary("f/@input:x", "x"),
            _boundary("f/@input:grid_thw", "grid_thw"),
            _op__nullary_op_operands("f/@op_add", "add", ["f/@input:x"]),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources__nullary_op_operands(nodes, "f/@op_add") == ["f/@input:x"]

    def test_an_edge_from_a_real_producer_survives(self) -> None:
        """Somebody built that dependency; removing it would hide work."""
        nodes = [
            _boundary("f/@input:grid_thw", "grid_thw"),
            _op__nullary_op_operands("f/@op_mul", "mul", []),
            _op__nullary_op_operands("f/@op_arange", "arange", ["f/@op_mul"]),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources__nullary_op_operands(nodes, "f/@op_arange") == ["f/@op_mul"]

    def test_an_op_with_no_known_schema_is_untouched(self) -> None:
        nodes = [
            _boundary("f/@input:x", "x"),
            _boundary("f/@input:grid_thw", "grid_thw"),
            _op__nullary_op_operands(
                "f/@op_custom", "some_custom_kernel", ["f/@input:x"]
            ),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources__nullary_op_operands(nodes, "f/@op_custom") == ["f/@input:x"]

    def test_a_variadic_op_is_untouched(self) -> None:
        """``cat`` takes a tensor LIST; its ceiling is not a real zero."""
        nodes = [
            _boundary("f/@input:x", "x"),
            _boundary("f/@input:grid_thw", "grid_thw"),
            _op__nullary_op_operands("f/@op_cat", "cat", ["f/@input:x"]),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources__nullary_op_operands(nodes, "f/@op_cat") == ["f/@input:x"]


# --- test_tile_fold_self_redirect.py ---

from TraceLens.Visualizer.model_explorer_export.merge import _fold_same_source_twins


def _tile__tile_fold_self_redirect(
    node_id: str, label: str, namespace: str, source: str
) -> dict:
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
        tile = _tile__tile_fold_self_redirect(
            "frame/@output:k_embed", "k_embed", "frame", "frame/concat"
        )
        # The same tile reachable twice in the node list.
        nodes = [concat, tile, dict(tile)]

        _fold_same_source_twins(nodes)

        survivors = [n for n in nodes if n["id"] == "frame/@output:k_embed"]
        assert survivors, "the fold deleted the tile by folding it onto itself"

    def test_the_producer_keeps_a_consumer(self) -> None:
        concat = _producer("frame/concat", "frame")
        tile = _tile__tile_fold_self_redirect(
            "frame/@output:k_embed", "k_embed", "frame", "frame/concat"
        )
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
            _tile__tile_fold_self_redirect(
                "frame/@output:k_embed", "k_embed", "frame", "frame/concat"
            ),
            _tile__tile_fold_self_redirect(
                "frame/@output:k_embed#2", "k_embed", "frame", "frame/concat"
            ),
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


# --- test_group_output_slot_collapse.py ---

from TraceLens.Visualizer.model_explorer_export.merge import _inject_group_outputs


def _node__group_output_slot_collapse(node_id: str, namespace: str, **extra) -> dict:
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
        _node__group_output_slot_collapse(
            f"{inside}/@input",
            inside,
            attrs=[{"key": "synthetic", "value": "@input"}],
        ),
        _node__group_output_slot_collapse(f"{inside}/q_cat", inside),
        _node__group_output_slot_collapse(f"{inside}/k_cat", inside),
        # Two consumers outside the frame, both reading k_cat -- one at the
        # producer's own port 0, one at the callee's return ordinal 1.
        _node__group_output_slot_collapse("outer/reader_a", "outer"),
        _node__group_output_slot_collapse("outer/reader_b", "outer"),
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


# --- test_attention_boundary_operand.py ---


def _node_attr(node: dict, key: str) -> str | None:
    for attr in node.get("attrs", []) or []:
        if attr.get("key") == key:
            return attr.get("value")
    return None


def _ports_of_attention_kernels(nodes: list[dict]) -> list[tuple[str, dict]]:
    """Each attention kernel's operand ports, as ``(kernel id, port node)``."""
    by_id = {node["id"]: node for node in nodes}
    found: list[tuple[str, dict]] = []
    for node in nodes:
        if "@attention" not in node.get("id", ""):
            continue
        for edge in node.get("incomingEdges", []) or []:
            port = by_id.get(str(edge.get("sourceNodeId")))
            if port is not None and _node_attr(port, "synthetic") == "@kernel_port_in":
                found.append((node["id"], port))
    return found


class TestBoundaryOperandIsWired:
    def test_no_attention_kernel_reads_a_port_named_for_the_activation(
        self, kimi_nodes
    ) -> None:
        """``sdpa`` has no ``hidden_states`` parameter, so no port may claim one."""
        offenders = [
            kernel
            for kernel, port in _ports_of_attention_kernels(kimi_nodes)
            if (port.get("label") or "").strip().lower() == "hidden_states"
        ]
        assert offenders == [], offenders

    def test_the_mask_port_is_sourced_from_the_mask_boundary(self, kimi_nodes) -> None:
        by_id = {node["id"]: node for node in kimi_nodes}
        # A mask the module is HANDED has no producer inside the block, so it
        # arrives on the module's own boundary rather than on a port built from
        # a producing chain. Either way the question is the same: does the
        # operand named ``attention_mask`` carry the mask?
        masks = [
            port
            for _kernel, port in _ports_of_attention_kernels(kimi_nodes)
            if (port.get("label") or "").strip().lower() == "attention_mask"
        ]
        masks += [
            by_id[str(edge["sourceNodeId"])]
            for node in kimi_nodes
            if "@attention" in str(node.get("id", ""))
            for edge in node.get("incomingEdges", []) or []
            if str(edge.get("sourceNodeId")) in by_id
            and (by_id[str(edge["sourceNodeId"])].get("label") or "").strip().lower()
            == "attention_mask"
        ]
        assert masks, "no attention_mask operand found"
        for port in masks:
            sources = [
                by_id.get(str(edge.get("sourceNodeId")))
                for edge in port.get("incomingEdges", []) or []
            ]
            labels = {
                (source.get("label") or "").strip().lower()
                for source in sources
                if source is not None
            }
            # The mask's producer is the mask, not the block's activation.
            assert "hidden_states" not in labels, (port["id"], labels)
            assert labels == {"attention_mask"}, (port["id"], labels)


# --- test_boundary_skip.py ---

import pytest

from TraceLens.Visualizer.model_explorer_export.type_check import (
    integrity_check_graph_nodes,
)

_MODEL = "zai-org/GLM-5.3-Flash"


def _node__boundary_skip(
    node_id: str, namespace: str, sources: list[str] | None = None
) -> dict:
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
            _node__boundary_skip("@in", ""),
            _node__boundary_skip("tower/inner/op", "tower/inner", ["@in"]),
        ]
        assert _skips(nodes), "an edge skipping the tower's boundary went unflagged"

    def test_entering_one_box_is_fine(self) -> None:
        nodes = [
            _node__boundary_skip("@in", ""),
            _node__boundary_skip("tower/@input", "tower", ["@in"]),
        ]
        assert not _skips(nodes)

    def test_leaving_two_boxes_is_flagged(self) -> None:
        nodes = [
            _node__boundary_skip("tower/inner/op", "tower/inner"),
            _node__boundary_skip("after", "", ["tower/inner/op"]),
        ]
        assert _skips(nodes)

    def test_a_repeat_group_is_not_a_wall(self) -> None:
        """``45x_...`` renders N identical layers; it is not a scope entered."""
        nodes = [
            _node__boundary_skip("@in", ""),
            _node__boundary_skip("decoder/45x_Layer/op", "decoder/45x_Layer", ["@in"]),
        ]
        assert not _skips(nodes), "a repeat group was treated as a box wall"

    def test_a_sibling_hop_is_not_a_skip(self) -> None:
        nodes = [
            _node__boundary_skip("tower/a/op", "tower/a"),
            _node__boundary_skip("tower/b/op", "tower/b", ["tower/a/op"]),
        ]
        assert not _skips(nodes)


@pytest.fixture(scope="module")
def glm_nodes(model_graph_nodes):
    return model_graph_nodes(_MODEL)


class TestTheModelHasNoSkips:
    def test_the_vision_tower_declares_the_grid_it_is_handed(self, glm_nodes) -> None:
        """The defect itself: the tower showing no ``grid_thw`` input."""
        tiles = [
            n
            for n in glm_nodes
            if (n.get("label") or "") == "grid_thw"
            and str(n.get("namespace") or "") == "visual"
        ]
        assert tiles, "the visual tower has no grid_thw boundary of its own"
