###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A kernel pipeline names what enters it where the box is SEEN.

Read at the ``KimiDeltaAttention`` level, ``chunk_kda_pipeline`` was a box with
five bare arrows going in -- a Linear and four Rearranges -- and nothing saying
which carried ``q`` and which carried ``beta``. The names existed, one level
down, on the tiles inside the box; a reader who has not opened it sees none of
them.

Every other box at that level already reads the other way round: each kernel
INSIDE the pipeline has its operand named beside it (``l2norm_fwd_q.q``), and
``_name_repeat_group_inputs_outside_it`` gives a repeat group the same thing.
So this was the one crossing in the drawing still unlabelled, and the pipeline
now gets the same treatment rather than a new convention.

The operand ``v`` is the reason the naming is taken from the tile the tensor
LANDS on rather than from a list of tiles: ``v`` reaches the pipeline through a
kernel port interposed late (``_port_unnamed_kernel_operands``), not through a
``@tensor`` tile like the other four, and it has to be named all the same.
"""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.merge import (
    _name_kernel_pipeline_inputs_outside_it,
)

PIPELINE = "attn/pipeline"


def _node(node_id, label, *, namespace, synthetic=None, constant=False, sources=()):
    attrs = []
    if synthetic:
        attrs.append({"key": "synthetic", "value": synthetic})
    if constant:
        attrs.append({"key": "constant", "value": "true"})
    attrs.append({"key": "output_shape", "value": "[B, S, H] bfloat16"})
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": attrs,
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": "0"} for s in sources
        ],
    }


def _mirrors(nodes):
    return {
        n["label"]: n
        for n in nodes
        if n["id"].startswith("attn/@input_mirror:") and n.get("namespace") == "attn"
    }


def _source_of(node):
    return node["incomingEdges"][0]["sourceNodeId"]


class TestThePassThatNamesThem:
    def test_each_operand_is_named_outside_the_box(self) -> None:
        nodes = [
            _node("q_src", "Rearrange", namespace="attn"),
            _node("k_src", "Rearrange", namespace="attn"),
            _node("q", "q", namespace=PIPELINE, synthetic="@tensor", sources=["q_src"]),
            _node("k", "k", namespace=PIPELINE, synthetic="@tensor", sources=["k_src"]),
        ]
        _name_kernel_pipeline_inputs_outside_it(nodes)
        mirrors = _mirrors(nodes)
        assert sorted(mirrors) == ["k", "q"]
        assert _source_of(mirrors["q"]) == "q_src"
        # The tile inside now reads its own mirror, not the raw producer.
        assert (
            _source_of(next(n for n in nodes if n["id"] == "q")) == mirrors["q"]["id"]
        )

    def test_an_operand_arriving_on_a_kernel_port_is_named_too(self) -> None:
        """Kimi's ``v``: no ``@tensor`` tile of its own, same crossing."""
        nodes = [
            _node("q_src", "Rearrange", namespace="attn"),
            _node("v_src", "Rearrange", namespace="attn"),
            _node("q", "q", namespace=PIPELINE, synthetic="@tensor", sources=["q_src"]),
            _node(
                "v",
                "v",
                namespace=PIPELINE,
                synthetic="@kernel_port_in",
                sources=["v_src"],
            ),
        ]
        _name_kernel_pipeline_inputs_outside_it(nodes)
        assert sorted(_mirrors(nodes)) == ["q", "v"]

    def test_a_crossing_already_named_at_that_level_is_left_alone(self) -> None:
        nodes = [
            _node("named", "cos", namespace="attn", synthetic="@input"),
            _node("q", "q", namespace=PIPELINE, synthetic="@tensor", sources=["named"]),
        ]
        _name_kernel_pipeline_inputs_outside_it(nodes)
        assert _mirrors(nodes) == {}

    def test_a_learned_weight_is_not_part_of_the_interface(self) -> None:
        nodes = [
            _node("w", "weight", namespace="attn", constant=True),
            _node("q", "q", namespace=PIPELINE, synthetic="@tensor", sources=["w"]),
        ]
        _name_kernel_pipeline_inputs_outside_it(nodes)
        assert _mirrors(nodes) == {}

    def test_one_producer_read_twice_inside_gets_one_tile(self) -> None:
        nodes = [
            _node("src", "Rearrange", namespace="attn"),
            _node("q", "q", namespace=PIPELINE, synthetic="@tensor", sources=["src"]),
            _node("use", "Kernel", namespace=PIPELINE, sources=["src"]),
        ]
        _name_kernel_pipeline_inputs_outside_it(nodes)
        assert len(_mirrors(nodes)) == 1

    def test_two_operands_the_box_calls_the_same_thing_stay_apart(self) -> None:
        """One tile means one tensor; merging them would drop one."""
        nodes = [
            _node("cos_src", "Cosine", namespace="attn"),
            _node("sin_src", "Sine", namespace="attn"),
            _node(
                "a",
                "position_embeddings",
                namespace=PIPELINE,
                synthetic="@tensor",
                sources=["cos_src"],
            ),
            _node(
                "b",
                "position_embeddings",
                namespace=PIPELINE,
                synthetic="@tensor",
                sources=["sin_src"],
            ),
        ]
        _name_kernel_pipeline_inputs_outside_it(nodes)
        assert sorted(_mirrors(nodes)) == ["Cosine", "Sine"]

    def test_a_box_declaring_no_operands_is_not_a_pipeline(self) -> None:
        nodes = [
            _node("src", "Rearrange", namespace="attn"),
            _node("op", "Kernel", namespace=PIPELINE, sources=["src"]),
        ]
        _name_kernel_pipeline_inputs_outside_it(nodes)
        assert _mirrors(nodes) == {}

    def test_an_empty_graph(self) -> None:
        nodes: list[dict] = []
        _name_kernel_pipeline_inputs_outside_it(nodes)
        assert nodes == []


def _pipeline_namespaces(nodes):
    return sorted(
        {
            str(n.get("namespace") or "")
            for n in nodes
            if str(n.get("namespace") or "").endswith("/chunk_kda_pipeline")
        }
    )


class TestKimiDrawsItThatWay:
    """The graph the reports were made against."""

    def test_every_pipeline_operand_is_named_at_the_attention_level(
        self, kimi_nodes
    ) -> None:
        for pipeline in _pipeline_namespaces(kimi_nodes):
            attention = pipeline.rsplit("/", 1)[0]
            inside = {
                str(n["id"])
                for n in kimi_nodes
                if str(n.get("namespace") or "").startswith(pipeline)
            }
            entering = {
                str(edge["sourceNodeId"])
                for n in kimi_nodes
                if str(n["id"]) in inside
                for edge in n.get("incomingEdges", []) or []
                if str(edge["sourceNodeId"]) not in inside
            }
            by_id = {str(n["id"]): n for n in kimi_nodes}
            unnamed = [
                by_id[node_id].get("label")
                for node_id in entering
                if not any(
                    attr["key"] == "synthetic" for attr in by_id[node_id]["attrs"]
                )
            ]
            assert not unnamed, f"{attention} enters {pipeline} unnamed: {unnamed}"

    def test_the_five_operands_it_declares_are_the_five_drawn(self, kimi_nodes) -> None:
        for pipeline in _pipeline_namespaces(kimi_nodes):
            attention = pipeline.rsplit("/", 1)[0]
            drawn = {
                str(n.get("label") or "").rsplit(".", 1)[-1]
                for n in kimi_nodes
                if str(n.get("namespace") or "") == attention
                and n["id"].startswith(f"{attention}/@input_mirror:")
            }
            assert drawn == {"q", "k", "v", "g", "beta"}, pipeline

    def test_the_v_operand_states_the_shape_it_carries(self, kimi_nodes) -> None:
        """It drew ``?`` -- the one report that was about a shape, not a name."""
        ports = [
            n
            for n in kimi_nodes
            if str(n["id"]).endswith("@kernel_in:v")
            or (
                str(n.get("label")) == "v"
                and str(n.get("namespace") or "").endswith("chunk_kda_pipeline")
            )
        ]
        assert ports, "no `v` operand tile in the pipeline"
        for port in ports:
            published = [
                attr["value"]
                for metadata in port.get("outputsMetadata") or []
                for attr in metadata.get("attrs", [])
                if attr["key"] == "shape"
            ]
            assert published == ["[B, S, 96, 128] bfloat16"], port["id"]


def test_no_node_states_a_shape_it_does_not_publish(kimi_nodes) -> None:
    """The invariant, on a real graph: an attr the reader cannot see is ``?``."""
    silent = [
        n["id"]
        for n in kimi_nodes
        if any(attr["key"] == "output_shape" for attr in n.get("attrs", []) or [])
        and not n.get("outputsMetadata")
    ]
    assert silent == []
