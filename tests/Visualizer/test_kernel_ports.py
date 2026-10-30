###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel operand tiles, pipeline ports, and how a return value is wired out."""

from __future__ import annotations

# --- test_kernel_interface_tiles.py ---

from TraceLens.Visualizer.model_explorer_export.merge import (
    _declared_kernel_input_names,
    _port_unnamed_kernel_operands,
)
from TraceLens.Visualizer.model_explorer_export.type_check import (
    _kernel_interface_tile_warnings,
)


def _node__kernel_interface_tiles(
    node_id,
    label,
    *,
    synthetic=None,
    operation=None,
    details=None,
    constant=False,
    raw_op=None,
    namespace="box",
    sources=(),
):
    attrs = []
    if synthetic:
        attrs.append({"key": "synthetic", "value": synthetic})
    if operation:
        attrs.append({"key": "operation", "value": operation})
    if details:
        attrs.append({"key": "details", "value": details})
    if constant:
        attrs.append({"key": "constant", "value": "true"})
    if raw_op:
        attrs.append({"key": "raw_op", "value": raw_op})
    attrs.append({"key": "output_shape", "value": "[B, S] bfloat16"})
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": attrs,
        "incomingEdges": [{"sourceNodeId": s} for s in sources],
    }


def _kernel(sources, details="kernel: k"):
    return _node__kernel_interface_tiles(
        "kern", "k", operation="gpu_kernel", details=details, sources=sources
    )


def _port(node_id, label, sources=()):
    return _node__kernel_interface_tiles(
        node_id, label, synthetic="@kernel_port_in", sources=sources
    )


class TestTheCheck:
    def test_an_operand_with_no_port_of_its_own(self) -> None:
        nodes = [
            _node__kernel_interface_tiles("mask", "attention_mask", synthetic="@input"),
            _port("p:q", "query"),
            _kernel(["p:q", "mask"]),
        ]
        warnings = _kernel_interface_tile_warnings(nodes)
        assert warnings
        assert "attention_mask" in warnings[0]

    def test_every_operand_on_a_port(self) -> None:
        nodes = [_port("p:q", "query"), _port("p:k", "key"), _kernel(["p:q", "p:k"])]
        assert _kernel_interface_tile_warnings(nodes) == []

    def test_a_single_operand_needs_no_port(self) -> None:
        """One unnamed edge into a box is readable; several are not."""
        nodes = [_node__kernel_interface_tiles("x", "Linear"), _kernel(["x"])]
        assert _kernel_interface_tile_warnings(nodes) == []

    def test_a_learned_weight_is_not_part_of_the_interface(self) -> None:
        nodes = [
            _node__kernel_interface_tiles("w", "weight", constant=True),
            _port("p:q", "query"),
            _kernel(["p:q", "w"]),
        ]
        assert _kernel_interface_tile_warnings(nodes) == []

    def test_several_readers_with_no_output_tile(self) -> None:
        nodes = [
            _port("p:q", "query"),
            _kernel(["p:q"]),
            _node__kernel_interface_tiles("a", "Slice", sources=["kern"]),
            _node__kernel_interface_tiles("b", "Slice", sources=["kern"]),
        ]
        warnings = _kernel_interface_tile_warnings(nodes)
        assert warnings
        assert "publishes to 2 readers" in warnings[0]

    def test_several_readers_each_on_an_output_tile(self) -> None:
        nodes = [
            _port("p:q", "query"),
            _kernel(["p:q"]),
            _node__kernel_interface_tiles(
                "o1", "out1", synthetic="@kernel_port_out", sources=["kern"]
            ),
            _node__kernel_interface_tiles(
                "o2", "out2", synthetic="@output", sources=["kern"]
            ),
        ]
        assert _kernel_interface_tile_warnings(nodes) == []

    def test_an_allocation_reader_is_not_a_published_value(self) -> None:
        """``qkv.new_empty(...)`` borrows a kernel's dtype/device as a buffer
        template for a later kernel to fill in; it never reads the kernel's
        value, so two such allocations off one kernel are not two of its
        outputs competing to be told apart."""
        nodes = [
            _port("p:q", "query"),
            _kernel(["p:q"]),
            _node__kernel_interface_tiles(
                "q_buf", "New empty", raw_op="new_empty", sources=["kern"]
            ),
            _node__kernel_interface_tiles(
                "idx_buf", "New empty", raw_op="new_empty", sources=["kern"]
            ),
        ]
        assert _kernel_interface_tile_warnings(nodes) == []

    def test_an_ordinary_op_is_not_a_kernel(self) -> None:
        nodes = [
            _node__kernel_interface_tiles("a", "A"),
            _node__kernel_interface_tiles("b", "B"),
            _node__kernel_interface_tiles(
                "add", "Add", operation="torch_functional", sources=["a", "b"]
            ),
        ]
        assert _kernel_interface_tile_warnings(nodes) == []

    def test_an_empty_graph(self) -> None:
        assert _kernel_interface_tile_warnings([]) == []


class TestThePassThatDrawsThem:
    def test_the_unported_operand_gets_a_port(self) -> None:
        nodes = [
            _node__kernel_interface_tiles(
                "mask", "attention_mask", synthetic="@input", namespace="other"
            ),
            _port("p:q", "query"),
            _kernel(["p:q", "mask"], details="kernel: k; inputs: query,attention_mask"),
        ]
        _port_unnamed_kernel_operands(nodes)
        assert _kernel_interface_tile_warnings(nodes) == []
        added = [n for n in nodes if n["id"].endswith(":@kernel_in:attention_mask")]
        assert added
        assert added[0]["label"] == "attention_mask"

    def test_the_port_takes_the_kernels_own_remaining_name(self) -> None:
        nodes = [
            _node__kernel_interface_tiles("src", "Rearrange", namespace="other"),
            _port("p:q", "query"),
            _kernel(["p:q", "src"], details="kernel: k; inputs: query,value"),
        ]
        _port_unnamed_kernel_operands(nodes)
        added = [n for n in nodes if ":@kernel_in:" in n["id"] and n["id"] != "p:q"]
        assert [n["label"] for n in added] == ["value"]

    def test_the_producers_name_when_the_kernel_declares_none(self) -> None:
        nodes = [
            _node__kernel_interface_tiles("src", "cu_seqlens", namespace="other"),
            _port("p:x", "x"),
            _kernel(["p:x", "src"]),
        ]
        _port_unnamed_kernel_operands(nodes)
        added = [n for n in nodes if ":@kernel_in:" in n["id"] and n["id"] != "p:x"]
        assert [n["label"] for n in added] == ["cu_seqlens"]

    def test_one_kernel_keeps_two_ports_for_one_tensor(self) -> None:
        """``key`` and ``value`` are two operands even when both are ``kv``."""
        nodes = [
            _node__kernel_interface_tiles("kv", "kv", namespace="other"),
            _port("p:q", "query"),
            _node__kernel_interface_tiles(
                "sdpa",
                "sdpa",
                operation="gpu_kernel",
                details="kernel: sdpa; inputs: query,key,value",
                sources=["p:q", "kv", "kv"],
            ),
        ]
        _port_unnamed_kernel_operands(nodes)
        added = [
            node
            for node in nodes
            if ":@kernel_in:" in node["id"] and node["id"] != "p:q"
        ]
        assert {node["label"] for node in added} == {"key", "value"}
        assert added

    def test_one_tensor_read_by_several_kernels_is_one_port(self) -> None:
        """The convention ``_share_one_tile_per_kernel_operand`` already draws."""
        nodes = [
            _node__kernel_interface_tiles("src", "cu_seqlens", namespace="other"),
            _port("p:a", "a"),
            _port("p:b", "b"),
            _node__kernel_interface_tiles(
                "k1", "k", operation="gpu_kernel", sources=["p:a", "src"]
            ),
            _node__kernel_interface_tiles(
                "k2", "k", operation="gpu_kernel", sources=["p:b", "src"]
            ),
        ]
        _port_unnamed_kernel_operands(nodes)
        added = [
            n
            for n in nodes
            if ":@kernel_in:" in n["id"] and n["id"] not in {"p:a", "p:b"}
        ]
        assert added
        assert _kernel_interface_tile_warnings(nodes) == []

    def test_a_tile_the_port_duplicates_is_folded_away(self) -> None:
        """Two tiles of one name in one box; the port keeps the name."""
        nodes = [
            _node__kernel_interface_tiles("up", "Rearrange", namespace="box"),
            _node__kernel_interface_tiles(
                "v", "v", synthetic="@tensor", namespace="box", sources=["up"]
            ),
            _port("p:q", "q"),
            _kernel(["p:q", "v"]),
        ]
        _port_unnamed_kernel_operands(nodes)
        assert [n["id"] for n in nodes if n["id"] == "v"] == []
        port = next(n for n in nodes if n["id"].endswith(":@kernel_in:v"))
        assert port["label"] == "v"
        assert port["incomingEdges"][0]["sourceNodeId"] == "up"

    def test_a_tile_several_consumers_share_is_kept(self) -> None:
        nodes = [
            _node__kernel_interface_tiles("up", "Rearrange", namespace="box"),
            _node__kernel_interface_tiles(
                "v", "v", synthetic="@tensor", namespace="box", sources=["up"]
            ),
            _node__kernel_interface_tiles(
                "other", "Slice", namespace="box", sources=["v"]
            ),
            _port("p:q", "q"),
            _kernel(["p:q", "v"]),
        ]
        _port_unnamed_kernel_operands(nodes)
        assert any(n["id"] == "v" for n in nodes)

    def test_a_kernel_already_fully_ported_is_untouched(self) -> None:
        nodes = [_port("p:q", "query"), _port("p:k", "key"), _kernel(["p:q", "p:k"])]
        before = len(nodes)
        _port_unnamed_kernel_operands(nodes)
        assert len(nodes) == before


class TestTheNamesAKernelDeclares:
    def test_an_inputs_list(self) -> None:
        node = _node__kernel_interface_tiles(
            "k", "k", details="kernel: f; inputs: query,key,value"
        )
        assert _declared_kernel_input_names(node) == ["query", "key", "value"]

    def test_operand_pairs(self) -> None:
        node = _node__kernel_interface_tiles(
            "k", "k", details="kernel: f; operand: stage_a=q, stage_b=gk"
        )
        assert _declared_kernel_input_names(node) == ["q", "gk"]

    def test_a_kernel_that_declares_nothing(self) -> None:
        assert (
            _declared_kernel_input_names(
                _node__kernel_interface_tiles("k", "k", details="kernel: f")
            )
            == []
        )
        assert (
            _declared_kernel_input_names(_node__kernel_interface_tiles("k", "k")) == []
        )


# --- test_kernel_operand_naming.py ---

from TraceLens.Visualizer.model_explorer_export.merge import (
    _share_one_tile_per_kernel_operand,
)


def _node__kernel_operand_naming(node_id, label, namespace, synthetic=None, sources=()):
    node = {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": "0"} for s in sources
        ],
    }
    if synthetic:
        node["attrs"] = [{"key": "synthetic", "value": synthetic}]
    return node


class TestOneTilePerOperand:
    def test_one_tensor_read_by_two_kernels_is_one_node(self) -> None:
        nodes = [
            _node__kernel_operand_naming("pipe/cumsum", "CumSum", "pipe"),
            _node__kernel_operand_naming(
                "pipe/@kernel_in:1:gk", "gk", "pipe", "@kernel_port_in", ["pipe/cumsum"]
            ),
            _node__kernel_operand_naming(
                "pipe/@kernel_in:2:gk", "gk", "pipe", "@kernel_port_in", ["pipe/cumsum"]
            ),
            _node__kernel_operand_naming(
                "pipe/intra", "intra", "pipe", None, ["pipe/@kernel_in:1:gk"]
            ),
            _node__kernel_operand_naming(
                "pipe/delta", "delta", "pipe", None, ["pipe/@kernel_in:2:gk"]
            ),
        ]

        _share_one_tile_per_kernel_operand(nodes)

        ports = [n for n in nodes if "@kernel_in" in n["id"]]
        assert ports, [n["id"] for n in ports]
        kept = ports[0]["id"]
        for consumer in ("pipe/intra", "pipe/delta"):
            node = next(n for n in nodes if n["id"] == consumer)
            assert node["incomingEdges"][0]["sourceNodeId"] == kept

    def test_two_slots_of_one_kernel_stay_two_operands(self) -> None:
        """A kernel binding one tensor twice really does take two operands."""
        nodes = [
            _node__kernel_operand_naming("pipe/src", "CumSum", "pipe"),
            _node__kernel_operand_naming(
                "pipe/@kernel_in:1:a", "x", "pipe", "@kernel_port_in", ["pipe/src"]
            ),
            _node__kernel_operand_naming(
                "pipe/@kernel_in:2:b", "x", "pipe", "@kernel_port_in", ["pipe/src"]
            ),
            _node__kernel_operand_naming(
                "pipe/one",
                "one",
                "pipe",
                None,
                ["pipe/@kernel_in:1:a", "pipe/@kernel_in:2:b"],
            ),
        ]

        _share_one_tile_per_kernel_operand(nodes)

        assert [n for n in nodes if "@kernel_in" in n["id"]]

    def test_a_port_repeating_the_tile_feeding_it_is_dropped(self) -> None:
        """``v`` docking into a second ``v`` says nothing the first did not."""
        nodes = [
            _node__kernel_operand_naming("pipe/v", "v", "pipe", "@tensor"),
            _node__kernel_operand_naming(
                "pipe/@kernel_in:1:v", "v", "pipe", "@kernel_port_in", ["pipe/v"]
            ),
            _node__kernel_operand_naming(
                "pipe/intra", "intra", "pipe", None, ["pipe/@kernel_in:1:v"]
            ),
        ]

        _share_one_tile_per_kernel_operand(nodes)

        assert [n["id"] for n in nodes] == ["pipe/v", "pipe/intra"]
        consumer = next(n for n in nodes if n["id"] == "pipe/intra")
        assert consumer["incomingEdges"][0]["sourceNodeId"] == "pipe/v"

    def test_a_port_naming_something_else_is_kept(self) -> None:
        nodes = [
            _node__kernel_operand_naming("pipe/v", "v", "pipe", "@tensor"),
            _node__kernel_operand_naming(
                "pipe/@kernel_in:1:k", "k", "pipe", "@kernel_port_in", ["pipe/v"]
            ),
            _node__kernel_operand_naming(
                "pipe/intra", "intra", "pipe", None, ["pipe/@kernel_in:1:k"]
            ),
        ]

        _share_one_tile_per_kernel_operand(nodes)

        assert nodes


# --- test_pipeline_box_inputs_named.py ---

from TraceLens.Visualizer.model_explorer_export.merge import (
    _name_kernel_pipeline_inputs_outside_it,
)

PIPELINE = "attn/pipeline"


def _node__pipeline_box_inputs_named(
    node_id, label, *, namespace, synthetic=None, constant=False, sources=()
):
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
            _node__pipeline_box_inputs_named("q_src", "Rearrange", namespace="attn"),
            _node__pipeline_box_inputs_named("k_src", "Rearrange", namespace="attn"),
            _node__pipeline_box_inputs_named(
                "q", "q", namespace=PIPELINE, synthetic="@tensor", sources=["q_src"]
            ),
            _node__pipeline_box_inputs_named(
                "k", "k", namespace=PIPELINE, synthetic="@tensor", sources=["k_src"]
            ),
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
            _node__pipeline_box_inputs_named("q_src", "Rearrange", namespace="attn"),
            _node__pipeline_box_inputs_named("v_src", "Rearrange", namespace="attn"),
            _node__pipeline_box_inputs_named(
                "q", "q", namespace=PIPELINE, synthetic="@tensor", sources=["q_src"]
            ),
            _node__pipeline_box_inputs_named(
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
            _node__pipeline_box_inputs_named(
                "named", "cos", namespace="attn", synthetic="@input"
            ),
            _node__pipeline_box_inputs_named(
                "q", "q", namespace=PIPELINE, synthetic="@tensor", sources=["named"]
            ),
        ]
        _name_kernel_pipeline_inputs_outside_it(nodes)
        assert _mirrors(nodes) == {}

    def test_a_learned_weight_is_not_part_of_the_interface(self) -> None:
        nodes = [
            _node__pipeline_box_inputs_named(
                "w", "weight", namespace="attn", constant=True
            ),
            _node__pipeline_box_inputs_named(
                "q", "q", namespace=PIPELINE, synthetic="@tensor", sources=["w"]
            ),
        ]
        _name_kernel_pipeline_inputs_outside_it(nodes)
        assert _mirrors(nodes) == {}

    def test_one_producer_read_twice_inside_gets_one_tile(self) -> None:
        nodes = [
            _node__pipeline_box_inputs_named("src", "Rearrange", namespace="attn"),
            _node__pipeline_box_inputs_named(
                "q", "q", namespace=PIPELINE, synthetic="@tensor", sources=["src"]
            ),
            _node__pipeline_box_inputs_named(
                "use", "Kernel", namespace=PIPELINE, sources=["src"]
            ),
        ]
        _name_kernel_pipeline_inputs_outside_it(nodes)
        assert _mirrors(nodes)

    def test_two_operands_the_box_calls_the_same_thing_stay_apart(self) -> None:
        """One tile means one tensor; merging them would drop one."""
        nodes = [
            _node__pipeline_box_inputs_named("cos_src", "Cosine", namespace="attn"),
            _node__pipeline_box_inputs_named("sin_src", "Sine", namespace="attn"),
            _node__pipeline_box_inputs_named(
                "a",
                "position_embeddings",
                namespace=PIPELINE,
                synthetic="@tensor",
                sources=["cos_src"],
            ),
            _node__pipeline_box_inputs_named(
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
            _node__pipeline_box_inputs_named("src", "Rearrange", namespace="attn"),
            _node__pipeline_box_inputs_named(
                "op", "Kernel", namespace=PIPELINE, sources=["src"]
            ),
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


# --- test_buffer_operand_first.py ---

from build_graph_in_env import pinned_merged_graph, pinned_model_spec

import pytest

from TraceLens.ModelUtils.shape_inference import ShapeInferencer

from tests.model_pins import pin_for

MODEL__buffer_operand_first = "MiniMaxAI/MiniMax-M3"


@pytest.fixture(scope="module")
def nodes__buffer_operand_first() -> list[dict]:
    pytest.importorskip("huggingface_hub")
    pin = pin_for(MODEL__buffer_operand_first)
    spec = pinned_model_spec(
        MODEL__buffer_operand_first,
        detailed=True,
        revision=pin.revision if pin else None,
    )
    graph = pinned_merged_graph(spec, shape_inferencer=ShapeInferencer(spec))
    return graph["nodes"]


def _attr(node: dict, key: str) -> str | None:
    for item in node.get("attrs", []) or []:
        if item.get("key") == key:
            return str(item.get("value"))
    return None


def _rotary_first_unsqueeze(nodes__buffer_operand_first: list[dict]) -> dict:
    found = [
        node
        for node in nodes__buffer_operand_first
        if "rotary_emb" in str(node.get("id", ""))
        and str(node.get("label")) == "Unsqueeze"
        and "_unsqueeze:" in str(node.get("id", ""))
    ]
    assert found, "expected the rotary embedding's first axis insert"
    return found[0]


class TestTheBufferIsReadFirst:
    def test_its_first_operand_is_the_buffer(
        self, nodes__buffer_operand_first: list[dict]
    ) -> None:
        by_id = {str(n["id"]): n for n in nodes__buffer_operand_first}
        node = _rotary_first_unsqueeze(nodes__buffer_operand_first)
        edges = node.get("incomingEdges", []) or []
        assert edges, node["id"]
        first = by_id.get(str(edges[0]["sourceNodeId"]))
        assert first is not None, edges[0]
        assert _attr(first, "constant") == "true", (
            first.get("id"),
            _attr(first, "constant"),
        )

    def test_it_reports_the_buffer_s_rank_not_the_hidden_state_s(
        self, nodes__buffer_operand_first: list[dict]
    ) -> None:
        shape = (
            _attr(_rotary_first_unsqueeze(nodes__buffer_operand_first), "output_shape")
            or ""
        )
        # ``inv_freq`` is 1-D, so inserting one axis gives rank 2. The hidden
        # state is rank 3, and inheriting it made this rank 4.
        assert shape.count(",") == 1, shape

    def test_the_buffer_is_its_only_operand(
        self, nodes__buffer_operand_first: list[dict]
    ) -> None:
        """``x`` supplies a device, not a tensor, so it is not wired here."""
        by_id = {str(n["id"]): n for n in nodes__buffer_operand_first}
        node = _rotary_first_unsqueeze(nodes__buffer_operand_first)
        sources = [
            by_id.get(str(edge["sourceNodeId"]))
            for edge in node.get("incomingEdges", []) or []
        ]
        assert sources, [s and s.get("id") for s in sources]
        assert _attr(sources[0], "constant") == "true"


# --- test_tuple_parameter_slots.py ---

from build_graph_in_env import pinned_merged_graph, pinned_model_spec

import collections

import pytest

from TraceLens.ModelUtils.shape_inference import ShapeInferencer

from tests.model_pins import pin_for

MODEL__tuple_parameter_slots = "deepseek-ai/DeepSeek-V4-Flash"


@pytest.fixture(scope="module")
def nodes__tuple_parameter_slots() -> list[dict]:
    pytest.importorskip("huggingface_hub")
    pin = pin_for(MODEL__tuple_parameter_slots)
    spec = pinned_model_spec(
        MODEL__tuple_parameter_slots,
        detailed=True,
        revision=pin.revision if pin else None,
    )
    graph = pinned_merged_graph(spec, shape_inferencer=ShapeInferencer(spec))
    return graph["nodes"]


def _ports_read_from(
    nodes__tuple_parameter_slots: list[dict], node_id: str
) -> collections.Counter:
    ports: collections.Counter = collections.Counter()
    for node in nodes__tuple_parameter_slots:
        for edge in node.get("incomingEdges", []) or []:
            if str(edge.get("sourceNodeId")) == node_id:
                ports[str(edge.get("sourceNodeOutputId"))] += 1
    return ports


def _attention_boundaries(nodes__tuple_parameter_slots: list[dict]) -> list[dict]:
    """The attention's ``position_embeddings`` crossings, one per component."""
    found = [
        node
        for node in nodes__tuple_parameter_slots
        if "self_attn/@input:position_embeddings" in str(node.get("id", ""))
    ]
    assert found, "expected the attention's position_embeddings boundaries"
    return found


class TestBothRotarySlotsAreRead:
    def test_each_component_crosses_on_its_own_tile(
        self, nodes__tuple_parameter_slots: list[dict]
    ) -> None:
        """A tuple is two tensors, so it crosses as two tiles, not one with two ports."""
        labels = {
            str(node.get("label"))
            for node in _attention_boundaries(nodes__tuple_parameter_slots)
        }
        assert labels == {
            "position_embeddings.cos",
            "position_embeddings.sin",
        }, labels

    def test_every_component_is_consumed(
        self, nodes__tuple_parameter_slots: list[dict]
    ) -> None:
        """``cos`` was read by nothing; only ``sin`` ever was."""
        for boundary in _attention_boundaries(nodes__tuple_parameter_slots):
            ports = _ports_read_from(nodes__tuple_parameter_slots, str(boundary["id"]))
            assert ports, f"{boundary['id']} reaches no consumer"

    def test_each_rotary_frame_reads_both(
        self, nodes__tuple_parameter_slots: list[dict]
    ) -> None:
        """Every ``apply_rotary_pos_emb`` frame multiplies by cos AND sin.

        Each component arrives on its own tile now, so the two
        ``repeat_interleave`` ops read two DISTINCT producers rather than two
        ports of one.
        """
        frames: dict[str, set[str]] = {}
        by_id = {str(n["id"]): n for n in nodes__tuple_parameter_slots}
        for node in nodes__tuple_parameter_slots:
            node_id = str(node.get("id", ""))
            if "apply_rotary_pos_emb" not in node_id:
                continue
            if "repeat_interleave" not in node_id:
                continue
            frame = node_id.rsplit(":@op_", 1)[0]
            for edge in node.get("incomingEdges", []) or []:
                source = by_id.get(str(edge.get("sourceNodeId")), {})
                label = str(source.get("label") or "")
                if label.startswith("position_embeddings"):
                    frames.setdefault(frame, set()).add(label)
        assert frames, "expected rotary frames reading position_embeddings"
        for frame, components in frames.items():
            assert components == {
                "position_embeddings.cos",
                "position_embeddings.sin",
            }, (frame, components)


# --- test_return_slot_wiring.py ---

from TraceLens.Visualizer.model_explorer_export.merge import _fold_same_source_twins


def _tile(node_id: str, label: str, namespace: str, synthetic: str, sources):
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": synthetic}],
        "incomingEdges": [
            {"sourceNodeId": source, "sourceNodeOutputId": port}
            for source, port in sources
        ],
    }


def test_a_mirror_repeating_the_tile_beside_it_folds_onto_it() -> None:
    nodes = [
        {"id": "producer", "label": "Op", "namespace": "block"},
        _tile("block/@input", "hidden_states", "block", "@input", [("producer", "0")]),
        _tile(
            "block/child/@input_mirror^hidden_states",
            "hidden_states",
            "block",
            "@input_mirror",
            [("producer", "0")],
        ),
        _tile(
            "block/child/@input",
            "hidden_states",
            "block/child",
            "@input",
            [("block/child/@input_mirror^hidden_states", "0")],
        ),
    ]

    _fold_same_source_twins(nodes)

    assert [n["id"] for n in nodes] == [
        "producer",
        "block/@input",
        "block/child/@input",
    ]
    # The child now reads the declaration its mirror was repeating.
    assert nodes[-1]["incomingEdges"][0]["sourceNodeId"] == "block/@input"


def test_a_kernel_keeps_every_operand_port_it_declares() -> None:
    """One ``cu_seqlens`` feeding ports 15/16/17 is still three operand slots."""
    nodes = [
        {"id": "producer", "label": "Op", "namespace": "attn"},
        _tile(
            "attn/@kernel_in:15:cu_seqlens",
            "cu_seqlens",
            "attn",
            "@kernel_port_in",
            [("producer", "0")],
        ),
        _tile(
            "attn/@kernel_in:16:cu_seqlens",
            "cu_seqlens",
            "attn",
            "@kernel_port_in",
            [("producer", "0")],
        ),
    ]

    _fold_same_source_twins(nodes)

    assert nodes


def test_two_slots_of_one_producer_are_not_the_same_tensor() -> None:
    """A producer declaring no ports can still publish several ordinals."""
    nodes = [
        {"id": "split", "label": "Split", "namespace": "block"},
        _tile("block/@output:a", "value", "block", "@output", [("split", "1")]),
        _tile("block/@output:b", "value", "block", "@output", [("split", "2")]),
    ]

    _fold_same_source_twins(nodes)

    assert nodes


def test_an_output_tile_still_leaves_its_parent_box() -> None:
    from TraceLens.Visualizer.model_explorer_export.merge import (
        _route_frame_exits_through_output_tiles,
    )

    inner = "embed/norm/@output:result"
    mirror = {
        "id": "mirror",
        "label": "result",
        "namespace": "",
        "incomingEdges": [
            {"sourceNodeId": inner, "sourceNodeOutputId": "0"},
        ],
    }
    nodes = [
        {
            "id": inner,
            "label": "result",
            "namespace": "embed/norm",
            "attrs": [{"key": "synthetic", "value": "@output"}],
            "incomingEdges": [],
        },
        mirror,
    ]
    _route_frame_exits_through_output_tiles(nodes)
    source = mirror["incomingEdges"][0]["sourceNodeId"]
    assert source == "embed/@output:result"
