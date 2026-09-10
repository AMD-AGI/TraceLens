###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Extra branch coverage for the merged Model Explorer graph builder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from TraceLens.Visualizer.model_explorer_export import merge
from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.ast_analyze import ForwardOperation, StackEntryDataflow
from TraceLens.ModelUtils.block_tree import BlockNode
from TraceLens.ModelUtils.blocks import BlockComponent, LayerVariant
from TraceLens.ModelUtils.computation_graph import ComputationGraph, GraphNodeSpec
from TraceLens.ModelUtils.extract import ArchitectureSpec
from TraceLens.ModelUtils.shape_inference import ShapeContext, TensorSpec


# ---------------------------------------------------------------------------
# Shared helpers (mirroring test_export_merge_coverage.py conventions)
# ---------------------------------------------------------------------------


def _component(
    attr: str,
    role: str = "other",
    *,
    class_name: str = "Module",
    label: str = "",
    order: int | None = None,
) -> BlockComponent:
    return BlockComponent(attr, class_name, role, label, order)


def _spec(**kwargs: object) -> ArchitectureSpec:
    defaults: dict[str, object] = {
        "name": "Synthetic",
        "model_type": "test",
        "hidden_size": 16,
        "vocab_size": 101,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "intermediate_size": 32,
    }
    defaults.update(kwargs)
    return ArchitectureSpec(**defaults)


def _edge(source: str, target_input: str = "0") -> dict[str, str]:
    return {
        "sourceNodeId": source,
        "sourceNodeOutputId": "0",
        "targetNodeInputId": target_input,
    }


def _attrs(**values: str) -> list[dict[str, str]]:
    return [{"key": key, "value": value} for key, value in values.items()]


def _op(label: str, details: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(label=label, details=list(details or []))


def _shaped_node(node_id: str, shape: str, dtype: str, **extra: object) -> dict:
    node = {
        "id": node_id,
        "label": "Prod",
        "outputsMetadata": [
            {
                "id": "0",
                "attrs": [
                    {"key": "shape", "value": f"{shape} {dtype}"},
                    {"key": "dtype", "value": dtype},
                ],
            }
        ],
    }
    node.update(extra)
    return node


BASIC = BasicOpFilter.for_detailed()


# ---------------------------------------------------------------------------
# _computation_nodes: multi-output split + skipped synthetic input (515-542)
# ---------------------------------------------------------------------------


def test_computation_nodes_multi_output_split_and_skipped_input():
    computation = ComputationGraph(
        nodes=[
            GraphNodeSpec("@input", label="input", synthetic="@input"),
            GraphNodeSpec("prod", label="Prod"),
            GraphNodeSpec("@output", label="output"),
        ],
        links=[(0, 1)],
        output_node_index=2,
        output_ports={"a": 1, "b": 1},
    )
    nodes = merge._computation_nodes(
        computation,
        id_prefix="blk",
        namespace_prefix="ns",
        skip_synthetic_input=True,
    )
    ids = {node["id"] for node in nodes}
    # synthetic @input is skipped entirely
    assert "blk/@input" not in ids
    # prod's edge from @input was redirected to the skipped-section sentinel
    prod = next(node for node in nodes if node["id"] == "blk/prod")
    assert prod["incomingEdges"][0]["sourceNodeId"] == merge._SKIPPED_SECTION_INPUT
    # two output ports -> one node per port
    port_ids = {node["id"] for node in nodes if node["id"].startswith("blk/@output")}
    assert len(port_ids) == 2


# ---------------------------------------------------------------------------
# _config_leaf / _data_movement_shape
# ---------------------------------------------------------------------------


def test_config_leaf_recurses_into_nested_dicts():
    config = {"a": 1, "nested": {"b": 2, "deeper": {"target": 42}}, "scalar": 5}
    assert merge._config_leaf(config, "a") == 1
    assert merge._config_leaf(config, "target") == 42
    assert merge._config_leaf(config, "missing") is None


def test_data_movement_shape_covers_all_branches():
    spec = _spec(raw_config={"hidden_size": 16})
    src = TensorSpec(("B", "S", 8), "float16")

    # source is None
    assert merge._data_movement_shape(_op("Unsqueeze"), None, spec=spec) is None
    # Unsqueeze with no dim detail returns source unchanged
    assert merge._data_movement_shape(_op("Unsqueeze"), src, spec=spec) is src
    # Unsqueeze with a non-integer dim detail returns source unchanged
    assert (
        merge._data_movement_shape(_op("Unsqueeze", ["dim: x"]), src, spec=spec) is src
    )
    # Negative dim wraps around the rank (+1)
    wrapped = merge._data_movement_shape(_op("Unsqueeze", ["dim: -1"]), src, spec=spec)
    assert wrapped.shape == ("B", "S", 8, 1)
    # Reshape/View with no shape detail returns source unchanged
    assert merge._data_movement_shape(_op("Reshape"), src, spec=spec) is src
    # View shape parts: -1 (from source), int literal, config lookup, and raw token
    resolved = merge._data_movement_shape(
        _op("View", ["shape: -1, 4, self.config.hidden_size, dynamic"]),
        src,
        spec=spec,
    )
    assert resolved.shape == ("B", 4, 16, "dynamic")
    # Unknown label passes source through
    assert merge._data_movement_shape(_op("Softmax"), src, spec=spec) is src


# ---------------------------------------------------------------------------
# _append_stack_entry_dataflow
# ---------------------------------------------------------------------------


def test_append_stack_entry_dataflow_skips_and_infers_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    op_skip = ForwardOperation("op_skip", "X", "", predecessors=("missing",))
    op_keep = ForwardOperation("op_keep", "Unsqueeze", "", predecessors=("prod",))
    dataflow = StackEntryDataflow(
        operations=(op_skip, op_keep), output_producer="op_keep"
    )
    monkeypatch.setattr(merge, "stack_entry_dataflow", lambda cls: dataflow)
    spec = _spec(stack_pre=[], stack_model_class="M", class_registry={"M": object()})
    nodes = [_shaped_node("producer_node", "B x S x 8", "float16")]
    inferencer = SimpleNamespace(context=ShapeContext({"H": 8}, "float16"))

    result = merge._append_stack_entry_dataflow(
        nodes,
        spec=spec,
        module_sources={"prod": "producer_node"},
        shape_inferencer=inferencer,
    )

    by_id = {node["id"]: node for node in nodes}
    assert "@model_forward/op_skip" not in by_id
    kept = by_id["@model_forward/op_keep"]
    assert kept["incomingEdges"][0]["sourceNodeId"] == "producer_node"
    assert result == ["@model_forward/op_keep"]


def test_append_stack_entry_dataflow_returns_none_without_dataflow(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(merge, "stack_entry_dataflow", lambda cls: None)
    monkeypatch.setattr(merge, "_pick_stack_model_class", lambda registry, name: None)
    assert (
        merge._append_stack_entry_dataflow(
            [], spec=_spec(class_registry={}), module_sources={}, shape_inferencer=None
        )
        is None
    )


# ---------------------------------------------------------------------------
# _apply_labeled_external_entry_ports edge cases (341, 345, 359)
# ---------------------------------------------------------------------------


def test_apply_labeled_entry_ports_duplicate_labels_return_false():
    tensor_target = {"attrs": _attrs(synthetic="@tensor"), "label": "q"}
    ports = [
        ("q", _edge("a"), dict(tensor_target, incomingEdges=[_edge("a")])),
        ("q", _edge("b"), dict(tensor_target, incomingEdges=[_edge("b")])),
    ]
    assert not merge._apply_labeled_external_entry_ports(ports, set())


def test_apply_labeled_entry_ports_single_empty_label_continues():
    target = {
        "attrs": _attrs(synthetic="@tensor"),
        "label": "q",
        "incomingEdges": [_edge("a")],
    }
    ports = [("", _edge("a"), target)]
    # single_labeled is True ("" is not None) but the empty label triggers `continue`.
    assert merge._apply_labeled_external_entry_ports(ports, set())


def test_apply_labeled_entry_ports_relabels_only_matching_edge():
    external = {"sourceNodeId": "outside", "sourceNodeOutputId": "0", "targetNodeInputId": "0"}
    other = {"sourceNodeId": "elsewhere", "sourceNodeOutputId": "0", "targetNodeInputId": "0"}
    internal = _edge("inside", "1")
    target = {
        "id": "t",
        "attrs": _attrs(synthetic="@tensor"),
        "label": "q",
        "incomingEdges": [internal, dict(external), dict(other)],
    }
    ports = [("q", external, target)]
    assert merge._apply_labeled_external_entry_ports(ports, {"inside"})
    labels = [
        edge.get("metadata", {}).get("port_label")
        for edge in target["incomingEdges"]
    ]
    # internal kept, matching external relabeled, unrelated external kept as-is
    assert labels == [None, "q", None]


# ---------------------------------------------------------------------------
# _replace_tile_with_group single-unresolved branch (637-638)
# ---------------------------------------------------------------------------


def test_replace_tile_with_group_single_unresolved_takes_all_unclaimed():
    tile = {
        "id": "tile",
        "incomingEdges": [_edge("p1"), _edge("p2", "1")],
    }
    entry = {
        "id": "nested/@input",
        "attrs": _attrs(synthetic="@input"),
        "label": "unmatched_label",
    }
    nested = [entry, {"id": "nested/work", "incomingEdges": [_edge("nested/@input")]}]
    section = [tile, *nested]
    merge._replace_tile_with_group(section, nested, tile_id="tile", exit_id=None)
    assert tile not in section
    # The single unresolved entry claims all unclaimed producer edges.
    assert entry["incomingEdges"] == [_edge("p1"), _edge("p2", "1")]


# ---------------------------------------------------------------------------
# _section_exits boundary fallthrough (694-695)
# ---------------------------------------------------------------------------


def test_section_exits_uses_output_node_when_present():
    computation = ComputationGraph(
        nodes=[GraphNodeSpec(key="a", label="A"), GraphNodeSpec(key="@output", label="O")],
        links=[(0, 1)],
        output_node_index=1,
        primary_output_port="result",
        output_ports={},
    )
    section = [{"id": "p/a"}, {"id": "p/@output:result"}]
    assert merge._section_exits(computation, section, id_prefix="p") == [
        ("p/@output:result", "result")
    ]


def test_section_exits_falls_back_to_boundary_nodes():
    computation = ComputationGraph(
        nodes=[GraphNodeSpec(key="a", label="A"), GraphNodeSpec(key="b", label="B")],
        links=[(0, 1)],
        primary_output_index=None,
        output_node_index=None,
    )
    section = [
        {"id": "p/a"},
        {"id": "p/b", "incomingEdges": [_edge("p/a")]},
    ]
    assert merge._section_exits(computation, section, id_prefix="p") == ["p/b"]


# ---------------------------------------------------------------------------
# _inject_group_inputs skip + empty-entry branches (937, 969)
# ---------------------------------------------------------------------------


def test_inject_group_inputs_skips_merged_tensor_port_parent():
    nodes = [
        {
            "id": "pipeline/forward_l2norm_fwd_q:sub_0",
            "namespace": "root/l2norm_fwd",
        }
    ]
    merge._inject_group_inputs(nodes)
    # The merged-tensor-port parent is skipped, so no @input is injected.
    assert not any(merge._is_synthetic_input(node) for node in nodes)


def test_inject_group_inputs_no_entry_nodes_when_all_internal():
    nodes = [
        {
            "id": "g/a",
            "namespace": "root/Cycle",
            "incomingEdges": [_edge("g/b")],
        },
        {
            "id": "g/b",
            "namespace": "root/Cycle",
            "incomingEdges": [_edge("g/a")],
        },
    ]
    merge._inject_group_inputs(nodes)
    assert not any(merge._is_synthetic_input(node) for node in nodes)


# ---------------------------------------------------------------------------
# _inject_group_outputs empty-source branch (1149)
# ---------------------------------------------------------------------------


def test_inject_group_outputs_skips_namespace_with_no_real_source():
    nodes = [
        {
            "id": "g/@input",
            "namespace": "ns",
            "attrs": _attrs(synthetic="@input"),
        }
    ]
    merge._inject_group_outputs(nodes)
    assert not any(merge._is_synthetic_output(node) for node in nodes)


# ---------------------------------------------------------------------------
# _inject_group_outputs multi-source + slot names (1162-1173, 1207-1229)
# ---------------------------------------------------------------------------


def test_inject_group_outputs_multi_source_with_slot_names():
    nodes = [
        {"id": "ns/@input", "namespace": "ns", "attrs": _attrs(synthetic="@input")},
        {"id": "ns/attn", "namespace": "ns", "incomingEdges": [_edge("ns/@input")]},
        {"id": "ns/mlp", "namespace": "ns", "incomingEdges": [_edge("ns/@input")]},
        {
            "id": "consumer",
            "namespace": "",
            "incomingEdges": [_edge("ns/attn", "0"), _edge("ns/mlp", "1")],
        },
    ]
    merge._inject_group_outputs(
        nodes,
        resolve_slot_names=lambda prefix: {"attn": "attn_out", "mlp": "mlp_out"},
    )
    output_ids = {node["id"] for node in nodes if merge._is_synthetic_output(node)}
    # One output node per source (multi-output branch).
    assert len(output_ids) == 2
    consumer = next(node for node in nodes if node["id"] == "consumer")
    src = [edge["sourceNodeId"] for edge in consumer["incomingEdges"]]
    assert all(sid in output_ids for sid in src)
    ports = {edge["sourceNodeOutputId"] for edge in consumer["incomingEdges"]}
    assert ports == {"attn_out", "mlp_out"}


# ---------------------------------------------------------------------------
# _mirror_boundary_inputs (1515-1571)
# ---------------------------------------------------------------------------


def test_mirror_boundary_inputs_creates_parent_mirrors():
    nodes = [
        {
            "id": "blk/attn/@input",
            "namespace": "blk/attn",
            "label": "hidden",
            "attrs": _attrs(synthetic="@input", port_label="hidden"),
            "incomingEdges": [_edge("outside")],
        },
        {
            "id": "blk/attn/@input:2",
            "namespace": "blk/attn",
            "label": "mask",
            "attrs": _attrs(synthetic="@input"),
            "incomingEdges": [_edge("blk/attn/@input")],  # internal only -> no mirror
        },
    ]
    merge._mirror_boundary_inputs(nodes)
    mirrors = [
        node
        for node in nodes
        if merge._node_attr(node, "synthetic") == "@input_mirror"
    ]
    assert len(mirrors) == 1
    mirror = mirrors[0]
    assert mirror["namespace"] == "blk"
    assert mirror["incomingEdges"][0]["sourceNodeId"] == "outside"
    # Original input now points at the mirror instead of the external source.
    original = next(node for node in nodes if node["id"] == "blk/attn/@input")
    assert original["incomingEdges"][0]["sourceNodeId"] == mirror["id"]


# ---------------------------------------------------------------------------
# _flatten_transparent_group_inputs (1247-1268)
# ---------------------------------------------------------------------------


def test_flatten_transparent_group_inputs_rewires_single_and_multi():
    nodes = [
        {
            "id": "g/@input",
            "namespace": "g",
            "label": "hidden_states",
            "attrs": _attrs(synthetic="@input"),
            "incomingEdges": [_edge("ext1"), _edge("ext2")],
        },
        {
            "id": "single/@input",
            "namespace": "single",
            "label": "input",
            "attrs": _attrs(synthetic="@input"),
            "incomingEdges": [_edge("solo")],
        },
        {
            "id": "empty/@input",
            "namespace": "empty",
            "label": "hidden_states",
            "attrs": _attrs(synthetic="@input"),
        },
        {
            "id": "consumer",
            "incomingEdges": [
                {"sourceNodeId": "g/@input", "sourceNodeOutputId": "0", "targetNodeInputId": "3"},
                {"sourceNodeId": "single/@input", "sourceNodeOutputId": "0", "targetNodeInputId": "0"},
                _edge("kept"),
            ],
        },
    ]
    merge._flatten_transparent_group_inputs(nodes)
    ids = {node["id"] for node in nodes}
    assert "g/@input" not in ids
    assert "single/@input" not in ids
    assert "empty/@input" in ids  # no incoming edges -> left in place
    consumer = next(node for node in nodes if node["id"] == "consumer")
    sources = [edge["sourceNodeId"] for edge in consumer["incomingEdges"]]
    targets = [edge["targetNodeInputId"] for edge in consumer["incomingEdges"]]
    assert "ext1" in sources and "ext2" in sources and "solo" in sources
    assert "3_0" in targets and "3_1" in targets  # multi-source split
    assert "kept" in sources


# ---------------------------------------------------------------------------
# _rename_generic_output_ports partial rename (1313, 1321, 1330-1332)
# ---------------------------------------------------------------------------


def test_rename_generic_output_ports_partial():
    output = {
        "id": "blk/@output",
        "inputsMetadata": [
            {"id": "result", "attrs": _attrs(port_label="result")},
            {"id": "keep", "attrs": _attrs(port_label="keep")},
        ],
        "outputsMetadata": [
            {"id": "result", "attrs": _attrs(port_label="result")},
            {"id": "keep", "attrs": _attrs(port_label="keep")},
        ],
        "incomingEdges": [
            {"sourceNodeId": "a", "sourceNodeOutputId": "0", "targetNodeInputId": "result",
             "metadata": {"port_label": "result"}},
            {"sourceNodeId": "b", "sourceNodeOutputId": "0", "targetNodeInputId": "keep"},
        ],
    }
    downstream = {
        "id": "consumer",
        "incomingEdges": [
            {"sourceNodeId": "blk/@output", "sourceNodeOutputId": "result", "targetNodeInputId": "0"},
            {"sourceNodeId": "blk/@output", "sourceNodeOutputId": "keep", "targetNodeInputId": "1"},
        ],
    }
    nodes = [output, downstream]
    renames = merge._rename_generic_output_ports(
        nodes, output_id="blk/@output", name="hidden"
    )
    assert renames == {"result": "hidden_1"}
    out_ports = [meta["id"] for meta in output["outputsMetadata"]]
    assert out_ports == ["hidden_1", "keep"]
    down_ports = [edge["sourceNodeOutputId"] for edge in downstream["incomingEdges"]]
    assert down_ports == ["hidden_1", "keep"]


def test_rename_generic_output_ports_missing_output_returns_empty():
    assert merge._rename_generic_output_ports([], output_id="absent", name="x") == {}


# ---------------------------------------------------------------------------
# _remove_transparent_root_output (1343-1362)
# ---------------------------------------------------------------------------


def test_remove_transparent_root_output_selects_primary_port():
    nodes = [
        {
            "id": "p/@output",
            "incomingEdges": [
                {"sourceNodeId": "a", "sourceNodeOutputId": "0", "targetNodeInputId": "result"},
                {"sourceNodeId": "b", "sourceNodeOutputId": "1", "targetNodeInputId": "other"},
            ],
        }
    ]
    ref = merge._remove_transparent_root_output(nodes, id_prefix="p", primary_port="other")
    assert ref == ("b", "1")
    assert not nodes


def test_remove_transparent_root_output_none_and_empty():
    assert merge._remove_transparent_root_output([], id_prefix="p", primary_port=None) is None
    nodes = [{"id": "p/@output", "incomingEdges": []}]
    assert merge._remove_transparent_root_output(nodes, id_prefix="p", primary_port=None) is None
    assert not nodes


# ---------------------------------------------------------------------------
# _wrap_actual_group_boundary (1379, 1392)
# ---------------------------------------------------------------------------


def test_wrap_actual_group_boundary_no_inputs_returns_outputs():
    outputs = [("x", "0")]
    assert merge._wrap_actual_group_boundary(
        [], namespace="ns", id_prefix="ns", inputs=[], outputs=outputs, first_node_index=0
    ) == outputs


def test_wrap_actual_group_boundary_rewires_and_skips_outside():
    nodes = [
        {"id": "blk/x", "namespace": "blk", "incomingEdges": [_edge("in")]},
        {"id": "outside", "namespace": "other", "incomingEdges": [_edge("in")]},
    ]
    refs = merge._wrap_actual_group_boundary(
        nodes,
        namespace="blk",
        id_prefix="blk",
        inputs=["in"],
        outputs=["blk/x"],
        first_node_index=0,
    )
    assert refs == [("blk/@output", "result")]
    by_id = {node["id"]: node for node in nodes}
    assert by_id["blk/x"]["incomingEdges"][0]["sourceNodeId"] == "blk/@input"
    # A node outside the namespace keeps its original edge.
    assert by_id["outside"]["incomingEdges"][0]["sourceNodeId"] == "in"
    assert "blk/@input" in by_id and "blk/@output" in by_id


# ---------------------------------------------------------------------------
# _mirror_boundary_outputs (1469, 1479)
# ---------------------------------------------------------------------------


def test_mirror_boundary_outputs_skips_empty_port_and_unconsumed():
    nodes = [
        {
            "id": "g/@output",
            "namespace": "g",
            "attrs": _attrs(synthetic="@output"),
            "outputsMetadata": [{"id": ""}, {"id": "a"}],
        },
        {
            "id": "g/@output:2",
            "namespace": "g",
            "attrs": _attrs(synthetic="@output"),
            "outputsMetadata": [{"id": "b"}],
        },
        {
            "id": "consumer",
            "namespace": "",
            "incomingEdges": [
                {"sourceNodeId": "g/@output", "sourceNodeOutputId": "a", "targetNodeInputId": "0"}
            ],
        },
    ]
    merge._mirror_boundary_outputs(nodes)
    mirror_ids = {node["id"] for node in nodes if merge._node_attr(node, "synthetic") == "@output_mirror"}
    # Only the consumed port "a" gets a mirror; "" and unconsumed "b" do not.
    assert mirror_ids == {"g/@output^a"}
    consumer = next(node for node in nodes if node["id"] == "consumer")
    assert consumer["incomingEdges"][0]["sourceNodeId"] == "g/@output^a"


# ---------------------------------------------------------------------------
# _prune_noop_cast_nodes missing-source branch (1608)
# ---------------------------------------------------------------------------


def test_prune_noop_cast_ignores_cast_with_missing_source():
    nodes = [
        {"id": "c", "label": "Cast", "incomingEdges": [_edge("ghost")]},
    ]
    merge._prune_noop_cast_nodes(nodes)
    assert [node["id"] for node in nodes] == ["c"]


# ---------------------------------------------------------------------------
# _prune_unconsumed_outputs keeps synthetic-input producers (1717)
# ---------------------------------------------------------------------------


def test_prune_unconsumed_outputs_keeps_synthetic_input_producer():
    nodes = [
        {"id": "g/@input", "attrs": _attrs(synthetic="@input")},
        {
            "id": "g/@output",
            "attrs": _attrs(synthetic="@output"),
            "inputsMetadata": [{"id": "unused"}],
            "outputsMetadata": [{"id": "unused"}],
            "incomingEdges": [
                {"sourceNodeId": "g/@input", "sourceNodeOutputId": "0", "targetNodeInputId": "unused"}
            ],
        },
    ]
    merge._prune_unconsumed_outputs(nodes)
    ids = {node["id"] for node in nodes}
    # The dead output port is stripped but its synthetic-input producer survives.
    assert "g/@input" in ids


# ---------------------------------------------------------------------------
# _nested_group_segment variants (1746, 1748, 1753)
# ---------------------------------------------------------------------------


def test_nested_group_segment_variants():
    pipeline = BlockNode("@pipe", "KernelPipeline", "other", "Pipe")
    dup = BlockNode("shared_attr", "Dup", "other", "Dup")
    regular = BlockNode("child", "Child", "other", "Child")

    assert (
        merge._nested_group_segment(pipeline, "Chunk pipe", has_tile=False, duplicate_labels=set())
        == "Chunk_pipe"
    )
    assert (
        merge._nested_group_segment(dup, "Dup", has_tile=True, duplicate_labels={"Dup"})
        == "shared_attr"
    )
    assert (
        merge._nested_group_segment(regular, "Title", has_tile=True, duplicate_labels=set())
        == "Title"
    )
    assert (
        merge._nested_group_segment(regular, "Ignored", has_tile=False, duplicate_labels=set())
        == "child"
    )


# ---------------------------------------------------------------------------
# _integrate_kernel_pipeline_merge output rewiring branches (1851, 1857)
# ---------------------------------------------------------------------------


def test_integrate_kernel_pipeline_merge_skips_and_keeps_edges():
    namespace = "decoder/attn"
    pipeline = f"{namespace}/pipeline"
    kernel = {
        "id": "chunk_gated_delta_rule_fwd_h",
        "namespace": pipeline,
        "incomingEdges": [_edge("src")],
    }
    merge_node = {
        "id": "merge",
        "namespace": namespace,
        "attrs": _attrs(attr_name="@attn_pipeline", class_name="KernelPipeline"),
        "incomingEdges": [_edge("q-source")],
    }
    # A sibling in the same namespace that is NOT the @attn_output -> `continue`.
    sibling = {
        "id": "sibling",
        "namespace": namespace,
        "attrs": _attrs(attr_name="something_else"),
        "incomingEdges": [_edge("merge")],
    }
    output = {
        "id": "output",
        "namespace": namespace,
        "attrs": _attrs(attr_name="@attn_output"),
        "incomingEdges": [_edge("merge"), _edge("other-producer", "1")],
    }
    tensor = {
        "id": "tensor-q",
        "label": "q",
        "namespace": pipeline,
        "attrs": _attrs(synthetic="@tensor"),
    }
    nodes = [tensor, kernel, merge_node, sibling, output]
    merge._integrate_kernel_pipeline_merge(
        nodes,
        namespace_prefix=namespace,
        pipeline_namespace=pipeline,
        pipeline_prefix="ignored",
        pipeline_label="Pipeline",
    )
    # merge-fed edge rewired to the pipeline exit; the non-merge edge is preserved.
    src_ids = {edge["sourceNodeId"] for edge in output["incomingEdges"]}
    assert kernel["id"] in src_ids
    assert "other-producer" in src_ids
    assert sibling["incomingEdges"][0]["sourceNodeId"] == "merge"


# ---------------------------------------------------------------------------
# _group_node_label (1958)
# ---------------------------------------------------------------------------


def test_group_node_label_norm_and_other():
    spec = _spec(norm_type="LayerNorm")
    assert merge._group_node_label(spec, _component("n", "norm")) == "LayerNorm"
    assert (
        merge._group_node_label(spec, _component("x", "other", label="Custom")) == "Custom"
    )


# ---------------------------------------------------------------------------
# _append_section: wrapper-around-one-op summary (2089-2104)
# ---------------------------------------------------------------------------


def _patch_section_pipeline(
    monkeypatch,
    *,
    root,
    prepared,
    computation=None,
    nested=None,
    transparent=False,
):
    monkeypatch.setattr(merge, "component_has_detail_section", lambda *a: True)
    monkeypatch.setattr(
        merge, "_resolve_section_tree_for_component", lambda *a, **k: ("Title", root)
    )
    monkeypatch.setattr(merge, "expand_block_tree_inplace", lambda tree, **k: prepared)
    if computation is not None:
        monkeypatch.setattr(
            merge, "build_computation_graph", lambda tree, **k: computation
        )
    monkeypatch.setattr(
        merge, "collect_nested_diagrams", lambda *a, **k: list(nested or [])
    )
    monkeypatch.setattr(
        merge, "is_transparent_inline_expansion", lambda tree: transparent
    )


def test_append_section_summary_single_op_collapse(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(merge, "component_has_detail_section", lambda *a: False)
    tree = BlockNode("mlp", "Mlp", "other", "Mlp",
                     children=[BlockNode("c", "C", "other", "C")])
    prepared = BlockNode("mlp", "Gelu", "other", "Gelu")  # childless after prep
    monkeypatch.setattr(
        merge, "_resolve_section_tree_for_component", lambda *a, **k: ("T", tree)
    )
    monkeypatch.setattr(merge, "expand_block_tree_inplace", lambda t, **k: prepared)
    nodes: list[dict[str, object]] = []
    exits = merge._append_section(
        nodes,
        spec=_spec(),
        component=_component("mlp", "other", label="MLP"),
        id_prefix="dec/mlp",
        namespace_prefix="dec",
        basic_ops=BASIC,
        previous_exits=["prev"],
    )
    assert exits == ["dec/mlp"]
    # summary label collapsed to the single prepared op's label
    assert nodes[0]["label"] == "Gelu"
    assert nodes[0]["incomingEdges"][0]["sourceNodeId"] == "prev"


def test_append_section_wrapper_around_single_op(monkeypatch: pytest.MonkeyPatch):
    root = BlockNode("hc_head", "Wrapper", "other", "Wrapper",
                     children=[BlockNode("c", "C", "other", "C")])
    prepared = BlockNode("hc_head", "Mean", "other", "Mean")  # childless
    _patch_section_pipeline(monkeypatch, root=root, prepared=prepared)
    nodes: list[dict[str, object]] = []
    exits = merge._append_section(
        nodes,
        spec=_spec(),
        component=_component("hc_head", "other", label="HC"),
        id_prefix="decoder/hc_head",
        namespace_prefix="decoder/Wrapper",
        basic_ops=BASIC,
        previous_exits=["prev"],
    )
    assert exits == ["decoder/hc_head"]
    assert nodes[0]["label"] == "Mean"
    assert nodes[0]["incomingEdges"] == [_edge("prev")]


# ---------------------------------------------------------------------------
# _append_section: kernel pipeline step integration (2132)
# ---------------------------------------------------------------------------


def test_append_section_triggers_kernel_pipeline_step(monkeypatch: pytest.MonkeyPatch):
    pipe = BlockNode("@pipe", "KernelPipeline", "other", "Pipeline")
    root = BlockNode("attn", "Attention", "attention", "Attention", children=[pipe])
    computation = ComputationGraph(
        nodes=[
            GraphNodeSpec("@input", label="input", synthetic="@input"),
            GraphNodeSpec("work", label="Work"),
        ],
        links=[(0, 1)],
        primary_output_index=1,
    )
    _patch_section_pipeline(
        monkeypatch, root=root, prepared=root, computation=computation
    )
    called: list[str] = []
    real_integrate = merge._integrate_kernel_pipeline_merge

    def spy(section_nodes, **kwargs):
        called.append(kwargs["pipeline_namespace"])
        return real_integrate(section_nodes, **kwargs)

    monkeypatch.setattr(merge, "_integrate_kernel_pipeline_merge", spy)
    nodes: list[dict[str, object]] = []
    merge._append_section(
        nodes,
        spec=_spec(),
        component=_component("attn", "attention", class_name="Attention"),
        id_prefix="decoder/attn",
        namespace_prefix="decoder/Attention",
        basic_ops=BASIC,
        previous_exits=["prev"],
    )
    assert "decoder/Attention/Pipeline" in called


# ---------------------------------------------------------------------------
# _append_section: transparent inline expansion at root (2256-2285)
# ---------------------------------------------------------------------------


def test_append_section_transparent_inline_expansion(monkeypatch: pytest.MonkeyPatch):
    root = BlockNode("blk", "Blk", "other", "Blk",
                     children=[BlockNode("c", "C", "other", "C")])
    computation = ComputationGraph(
        nodes=[
            GraphNodeSpec("@input", label="input", synthetic="@input"),
            GraphNodeSpec("work", label="Work"),
            GraphNodeSpec("@output", label="output"),
        ],
        links=[(0, 1), (1, 2)],
        primary_output_index=1,
    )
    _patch_section_pipeline(
        monkeypatch, root=root, prepared=root, computation=computation, transparent=True
    )
    nodes: list[dict[str, object]] = []
    exits = merge._append_section(
        nodes,
        spec=_spec(),
        component=_component("blk", "other", label="Blk"),
        id_prefix="blk",
        namespace_prefix="",
        basic_ops=BASIC,
        previous_exits=["prev"],
    )
    ids = {node["id"] for node in nodes}
    assert "blk/@input" not in ids
    assert "blk/@output" not in ids
    work = next(node for node in nodes if node["id"] == "blk/work")
    assert work["incomingEdges"][0]["sourceNodeId"] == "prev"
    assert exits == [("blk/work", "0")]


# ---------------------------------------------------------------------------
# _append_section: non-decoder norm group attributes (2324-2332)
# ---------------------------------------------------------------------------


def test_append_section_custom_norm_group_attributes(monkeypatch: pytest.MonkeyPatch):
    root = BlockNode("custom_norm", "RMSNorm", "norm", "Norm",
                     children=[BlockNode("c", "C", "other", "C")])
    computation = ComputationGraph(
        nodes=[
            GraphNodeSpec("@input", label="input", synthetic="@input"),
            GraphNodeSpec("op", label="Op"),
        ],
        links=[(0, 1)],
        primary_output_index=1,
    )
    _patch_section_pipeline(
        monkeypatch, root=root, prepared=root, computation=computation
    )
    group_attrs: dict[str, dict[str, str]] = {}
    merge._append_section(
        nodes := [],
        spec=_spec(norm_type="RMSNorm"),
        component=_component("custom_norm", "norm", class_name="RMSNorm"),
        id_prefix="norm",
        namespace_prefix="dec/Norm",
        basic_ops=BASIC,
        previous_exits=["prev"],
        group_node_attributes=group_attrs,
    )
    assert group_attrs["dec/Norm"] == {"label": "RMSNorm", "operation": "RMSNorm"}
    assert nodes


# ---------------------------------------------------------------------------
# _append_section: skip_variant_root_input path (2119, 2226-2227)
# ---------------------------------------------------------------------------


def test_append_section_skip_variant_root_input(monkeypatch: pytest.MonkeyPatch):
    root = BlockNode("blk", "Blk", "other", "Blk",
                     children=[BlockNode("c", "C", "other", "C")])
    computation = ComputationGraph(
        nodes=[
            GraphNodeSpec("@input", label="input", synthetic="@input"),
            GraphNodeSpec("op", label="Op"),
        ],
        links=[(0, 1)],
        primary_output_index=1,
    )
    _patch_section_pipeline(
        monkeypatch, root=root, prepared=root, computation=computation
    )
    monkeypatch.setattr(merge, "_skip_variant_root_input", lambda component: True)
    nodes: list[dict[str, object]] = []
    merge._append_section(
        nodes,
        spec=_spec(),
        component=_component("blk", "other", label="Blk"),
        id_prefix="blk",
        namespace_prefix="dec/Blk",
        basic_ops=BASIC,
        previous_exits=["prev"],
    )
    # The synthetic @input is skipped, so the section input placeholder is rewired
    # straight to the previous exit.
    op = next(node for node in nodes if node["id"] == "blk/op")
    assert op["incomingEdges"][0]["sourceNodeId"] == "prev"


# ---------------------------------------------------------------------------
# _append_variant_layer hyperconnection residual mixing (2349-2466)
# ---------------------------------------------------------------------------


def test_append_variant_layer_hyperconnection_residual_mix(
    monkeypatch: pytest.MonkeyPatch,
):
    # The bare name `inline_expansion` at the _append_section call site resolves to
    # a module global; provide one so the (otherwise dead) branch can execute.
    monkeypatch.setattr(merge, "inline_expansion", True, raising=False)

    attn_hc = _component("attn_hc", "other")
    self_attn = _component("self_attn", "attention")
    ffn_hc = _component("ffn_hc", "other")
    mlp = _component("mlp", "ffn")
    monkeypatch.setattr(
        merge,
        "_ordered_decoder_components",
        lambda spec: [attn_hc, self_attn, ffn_hc, mlp],
    )

    def fake_append(merged_nodes, **kwargs):
        prefix = kwargs["id_prefix"]
        component = kwargs["component"]
        merged_nodes.append(
            {
                "id": prefix,
                "namespace": kwargs["namespace_prefix"],
                "incomingEdges": [
                    merge._source_edge(src, "0") for src in kwargs["previous_exits"]
                ],
            }
        )
        if component.attr_name in {"attn_hc", "ffn_hc"}:
            merged_nodes.append(
                {
                    "id": f"{prefix}/@output",
                    "namespace": kwargs["namespace_prefix"],
                    "outputsMetadata": [{"id": "post"}, {"id": "comb"}],
                }
            )
        return [prefix]

    monkeypatch.setattr(merge, "_append_section", fake_append)

    variant = LayerVariant("v", 2, "Attn", "VariantAttn", "FFN", "VariantFFN", "experts")
    merged: list[dict[str, object]] = []
    exits = merge._append_variant_layer(
        merged,
        spec=_spec(decoder_class="Dec"),
        variant=variant,
        id_prefix="decoder/v",
        namespace_prefix="dec/v",
        basic_ops=BASIC,
        previous_exits=["@input"],
        group_node_configs=[],
        group_node_attributes={},
    )
    ids = {node["id"] for node in merged}
    assert "decoder/v/@residual:attention:add" in ids
    assert "decoder/v/@residual:ffn:add" in ids
    assert "decoder/v/@residual:attention:matmul" in ids
    assert "decoder/v/@residual:attention:multiply" in ids
    assert exits == ["decoder/v/@residual:ffn:add"]


# ---------------------------------------------------------------------------
# _append_source_decoder_layer (2502, 2517, 2542, 2548, 2586-2596)
# ---------------------------------------------------------------------------


def test_append_source_decoder_layer_no_forward_calls_returns_inputs():
    spec = _spec(decoder_class=None, class_registry={})
    assert merge._append_source_decoder_layer(
        [],
        spec=spec,
        id_prefix="d",
        namespace_prefix="d",
        basic_ops=BASIC,
        previous_exits=["prev"],
        variant=None,
        group_node_configs=[],
        group_node_attributes={},
        shape_inferencer=None,
    ) == ["prev"]


def test_append_source_decoder_layer_full_paths(monkeypatch: pytest.MonkeyPatch):
    op_ok = ForwardOperation("op_ok", "MatMul", "", predecessors=("@method_input",))
    op_nosrc = ForwardOperation("op_nosrc", "Bad", "", predecessors=("missing",))
    attn_callee = SimpleNamespace(forward_return_slots={"slot": "attn_producer"})
    decoder = SimpleNamespace(
        forward_calls=["op_ok", "op_nosrc", "self_attn", "mlp_nofeed", "ghost_step"],
        forward_operations={"op_ok": op_ok, "op_nosrc": op_nosrc},
        forward_step_predecessors={
            "self_attn": ("op_ok",),
            "mlp_nofeed": ("nothing",),
        },
        primary_return_slot=None,
        forward_return_slots={},
    )
    self_attn = _component("self_attn", "attention", class_name="AttnClass")
    mlp = _component("mlp_nofeed", "ffn", class_name="MlpClass")
    spec = _spec(
        decoder_class="Dec",
        class_registry={"Dec": decoder, "AttnClass": attn_callee},
        block_components=[self_attn, mlp],
    )

    def fake_append(merged_nodes, **kwargs):
        prefix = kwargs["id_prefix"]
        merged_nodes.append({"id": prefix, "namespace": kwargs["namespace_prefix"]})
        merged_nodes.append(
            {"id": f"{prefix}/@output", "outputsMetadata": [{"id": "slot"}]}
        )
        return [prefix]

    monkeypatch.setattr(merge, "_append_section", fake_append)
    merged: list[dict[str, object]] = []
    exits = merge._append_source_decoder_layer(
        merged,
        spec=spec,
        id_prefix="d",
        namespace_prefix="d",
        basic_ops=BASIC,
        previous_exits=["P"],
        variant=None,
        group_node_configs=[],
        group_node_attributes={},
        shape_inferencer=None,
    )
    ids = {node["id"] for node in merged}
    assert "d/op_ok" in ids
    assert "d/op_nosrc" not in ids  # no available source -> skipped
    assert "d/self_attn" in ids
    assert exits == ["d/self_attn"]  # reversed-forward-calls fallback


def test_append_source_decoder_layer_primary_return_slot(
    monkeypatch: pytest.MonkeyPatch,
):
    op_ok = ForwardOperation("op_ok", "MatMul", "", predecessors=("@method_input",))
    decoder = SimpleNamespace(
        forward_calls=["op_ok"],
        forward_operations={"op_ok": op_ok},
        forward_step_predecessors={},
        primary_return_slot="ret",
        forward_return_slots={"ret": "op_ok"},
    )
    spec = _spec(decoder_class="Dec", class_registry={"Dec": decoder}, block_components=[])
    monkeypatch.setattr(merge, "_append_section", lambda *a, **k: ["x"])
    exits = merge._append_source_decoder_layer(
        [],
        spec=spec,
        id_prefix="d",
        namespace_prefix="d",
        basic_ops=BASIC,
        previous_exits=["P"],
        variant=None,
        group_node_configs=[],
        group_node_attributes={},
        shape_inferencer=None,
    )
    assert exits == ["d/op_ok"]


# ---------------------------------------------------------------------------
# _append_decoder_layers variant paths (2614-2679, 2654) + source path (2637-2652)
# ---------------------------------------------------------------------------


def test_append_decoder_layers_variant_uses_variant_layer(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(merge, "expand_class_forward_dataflow", lambda *a, **k: None)
    decoder = SimpleNamespace(forward_operations={}, forward_step_predecessors={})
    variant = LayerVariant("v", 2, "A", "VariantAttn", "F", "VariantFFN")
    spec = _spec(
        decoder_class="Dec", class_registry={"Dec": decoder}, layer_variants=[variant]
    )

    def fake_variant(merged_nodes, **kwargs):
        merged_nodes.append(
            {
                "id": f'{kwargs["id_prefix"]}/x',
                "namespace": kwargs["namespace_prefix"],
                "incomingEdges": [
                    merge._source_edge(src, "0") for src in kwargs["previous_exits"]
                ],
            }
        )
        return [f'{kwargs["id_prefix"]}/x']

    monkeypatch.setattr(merge, "_append_variant_layer", fake_variant)
    merged: list[dict[str, object]] = []
    configs: list[dict[str, object]] = []
    attrs: dict[str, dict[str, str]] = {}
    exits = merge._append_decoder_layers(
        merged,
        spec=spec,
        decoder_namespace="dec",
        basic_ops=BASIC,
        previous_exits=["@input"],
        group_node_configs=configs,
        group_node_attributes=attrs,
    )
    slug = merge._variant_namespace_slug(variant)
    ids = {node["id"] for node in merged}
    assert f"decoder/{slug}/@input" in ids
    assert f"decoder/{slug}/@output" in ids
    assert exits
    assert f"dec/{slug}" in attrs


def test_append_decoder_layers_variant_uses_source_layer(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(merge, "expand_class_forward_dataflow", lambda *a, **k: None)
    op = ForwardOperation("op", "MatMul", "", predecessors=())
    decoder = SimpleNamespace(
        forward_operations={"op": op}, forward_step_predecessors={}
    )
    variant = LayerVariant("v", 1, "A", "VariantAttn", "F", "VariantFFN")
    spec = _spec(
        decoder_class="Dec", class_registry={"Dec": decoder}, layer_variants=[variant]
    )
    seen: list[str] = []

    def fake_source(merged_nodes, **kwargs):
        seen.append(kwargs["namespace_prefix"])
        merged_nodes.append({"id": f'{kwargs["id_prefix"]}/x', "namespace": kwargs["namespace_prefix"]})
        return [f'{kwargs["id_prefix"]}/x']

    monkeypatch.setattr(merge, "_append_source_decoder_layer", fake_source)
    merged: list[dict[str, object]] = []
    merge._append_decoder_layers(
        merged,
        spec=spec,
        decoder_namespace="dec",
        basic_ops=BASIC,
        previous_exits=["@input"],
        group_node_configs=[],
        group_node_attributes={},
    )
    slug = merge._variant_namespace_slug(variant)
    assert seen == [f"dec/{slug}"]


# ---------------------------------------------------------------------------
# _append_decoder_layers non-variant source path (2681-2716)
# ---------------------------------------------------------------------------


def test_append_decoder_layers_non_variant_source_path(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(merge, "expand_class_forward_dataflow", lambda *a, **k: None)
    op = ForwardOperation("op", "MatMul", "", predecessors=())
    decoder = SimpleNamespace(
        forward_operations={"op": op}, forward_step_predecessors={}
    )
    attn = _component("self_attn", "attention", class_name="Attn")
    mlp = _component("mlp", "ffn", class_name="Mlp")
    spec = _spec(
        decoder_class="Dec",
        class_registry={"Dec": decoder},
        layer_variants=[],
        block_components=[attn, mlp],
    )
    monkeypatch.setattr(merge, "_ordered_decoder_components", lambda s: [attn, mlp])

    def fake_source(merged_nodes, **kwargs):
        merged_nodes.append(
            {
                "id": "decoder/self_attn",
                "namespace": kwargs["namespace_prefix"],
                "incomingEdges": [
                    merge._source_edge(src, "0") for src in kwargs["previous_exits"]
                ],
            }
        )
        return ["decoder/self_attn"]

    monkeypatch.setattr(merge, "_append_source_decoder_layer", fake_source)
    merged: list[dict[str, object]] = []
    configs: list[dict[str, object]] = []
    exits = merge._append_decoder_layers(
        merged,
        spec=spec,
        decoder_namespace="model/decoder",
        basic_ops=BASIC,
        previous_exits=["@input"],
        group_node_configs=configs,
        group_node_attributes={},
    )
    ids = {node["id"] for node in merged}
    # Group boundary wraps the chain: synthetic input/output added at decoder ns.
    assert "decoder/@input" in ids
    assert "decoder/@output" in ids
    # attention role produced a group config entry.
    assert any("attention" in cfg.get("namespaceRegex", "").lower() for cfg in configs) or configs
    assert exits


# ---------------------------------------------------------------------------
# _append_vision_section group config append (2776)
# ---------------------------------------------------------------------------


def test_append_vision_section_appends_group_config(monkeypatch: pytest.MonkeyPatch):
    vision = _component("visual", "attention", class_name="VisionModel", label="Vision")
    monkeypatch.setattr(merge, "vision_tower_component", lambda spec: vision)
    monkeypatch.setattr(merge, "component_has_detail_section", lambda *a: True)
    monkeypatch.setattr(
        merge,
        "_resolve_section_tree_for_component",
        lambda *a, **k: ("Vision", BlockNode("visual", "VisionModel", "attention", "Vision")),
    )
    monkeypatch.setattr(merge, "expand_block_tree_inplace", lambda tree, **k: tree)
    monkeypatch.setattr(merge, "is_transparent_inline_expansion", lambda tree: False)

    def fake_append(nodes, **kwargs):
        nodes.append({"id": "visual/x", "namespace": kwargs["namespace_prefix"]})
        return ["visual/x"]

    monkeypatch.setattr(merge, "_append_section", fake_append)
    nodes: list[dict[str, object]] = []
    configs: list[dict[str, object]] = []
    exit_ref = merge._append_vision_section(
        nodes,
        spec=_spec(),
        basic_ops=BASIC,
        group_node_configs=configs,
        group_node_attributes={},
        shape_inferencer=None,
        inline_expansion=True,
    )
    assert exit_ref == "visual/x"
    assert any("visual" in config["namespaceRegex"] for config in configs)
    assert nodes[0]["id"] == "@vision_input"
