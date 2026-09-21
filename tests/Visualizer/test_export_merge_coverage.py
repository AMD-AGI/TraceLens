###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Focused coverage for merged export helpers and symbolic shape fallbacks."""

from __future__ import annotations

import ast
import copy
from types import SimpleNamespace

import pytest

from TraceLens.Visualizer.model_explorer_export import labels, merge, overview, shapes, styles
from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.block_tree import BlockNode
from TraceLens.ModelUtils.blocks import BlockComponent, LayerVariant
from TraceLens.ModelUtils.computation_graph import ComputationGraph, GraphNodeSpec
from TraceLens.ModelUtils.extract import ArchitectureSpec
from TraceLens.ModelUtils.model_graph import (
    GraphEdge,
    ModelGraph,
    ModelGraphNode,
    NodeKind,
    OperationKind,
)
from TraceLens.ModelUtils.shape_inference import (
    ModuleDimRegistry,
    ModuleLinearSpec,
    ModuleParameterSpec,
    OperatorRecord,
    ShapeContext,
    ShapeInferencer,
    TensorSpec,
    _broadcast_rank,
    _config_dtype,
    _dedupe_preserve,
    _detail_value,
    _dim_term,
    _heuristic_linear_out_features,
    _int_dim,
    _nested_dim_aliases,
    _parse_module_ctor,
    _parse_tensor_ctor_shape,
    _replace_last_dim,
    _resolve_dim_expr,
    _symbolic_binop,
    _topological_order,
    build_operator_export,
    save_operator_export,
    subgraph_boundary_signature,
)


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


def _model_node(
    label: str,
    *,
    node_id: str = "node",
    operation: OperationKind | None = OperationKind.TORCH_FUNCTIONAL,
    synthetic: str | None = None,
    details: list[str] | None = None,
    external_inputs: list[str] | None = None,
    class_name: str | None = None,
    attr_name: str | None = None,
    kind: NodeKind = NodeKind.LEAF,
) -> ModelGraphNode:
    metadata: dict[str, object] = {}
    if synthetic:
        metadata["synthetic"] = synthetic
    if details:
        metadata["details"] = details
    if external_inputs:
        metadata["external_inputs"] = external_inputs
    if class_name:
        metadata["class_name"] = class_name
    if attr_name:
        metadata["attr_name"] = attr_name
    return ModelGraphNode(node_id, kind, label, operation, metadata)


def test_merge_scalar_helpers_and_input_metadata():
    assert merge._join_namespace("", "leaf") == "leaf"
    assert merge._join_namespace("root", "") == "root"
    assert merge._join_namespace("root", "leaf") == "root/leaf"
    assert merge._merge_node_id("", "n") == "n"
    assert merge._merge_node_id("p", "n") == "p/n"
    assert merge._group_input_id("p") == "p/@input"
    assert merge._group_input_id("p", "q") == "p/@input:q"

    assert merge._is_synthetic_input({"id": "@input"})
    assert merge._is_synthetic_input({"id": "p/@input:q"})
    assert merge._is_synthetic_input({"attrs": _attrs(synthetic="@input")})
    assert not merge._is_synthetic_input({"id": "input"})
    assert merge._node_attr({"attrs": _attrs(answer="42")}, "answer") == "42"
    assert (
        merge._node_attr({"attrs": [{"key": "answer", "value": 42}]}, "answer") is None
    )

    edge = {"metadata": {"port_label": 7}}
    assert merge._edge_port_label(edge) == "7"
    assert merge._edge_port_label({}) is None
    labeled = merge._label_input_edge({}, "q")
    assert labeled == {"metadata": {"port_label": "q"}}

    node = {
        "inputsMetadata": [
            {
                "id": "0",
                "attrs": [
                    {"key": "port_label", "value": "old"},
                    {"key": "keep", "value": "yes"},
                ],
            }
        ]
    }
    merge._set_input_port_metadata(node, "0", "new")
    merge._set_input_port_metadata(node, "1", "k")
    assert node["inputsMetadata"] == [
        {
            "id": "0",
            "attrs": [
                {"key": "keep", "value": "yes"},
                {"key": "port_label", "value": "new"},
            ],
        },
        {"id": "1", "attrs": [{"key": "port_label", "value": "k"}]},
    ]


def test_merge_entry_port_collection_and_label_application_paths():
    external = _edge("outside")
    internal = _edge("inside", "1")
    target = {
        "id": "inside",
        "label": "q",
        "attrs": _attrs(synthetic="@tensor"),
        "incomingEdges": [external, dict(external), internal],
    }
    node_by_id = {"outside": {"id": "outside", "attrs": _attrs(port_label="source")}}
    ports = merge._collect_group_entry_ports([target], {"inside"}, node_by_id)
    assert len(ports) == 1
    assert ports[0][0] == "q"
    assert merge._apply_labeled_external_entry_ports(ports, {"inside"})
    assert target["incomingEdges"][0]["metadata"] == {"port_label": "q"}
    assert "metadata" not in target["incomingEdges"][-1]

    assert not merge._apply_labeled_external_entry_ports([], set())
    unlabeled = [(None, _edge("x"), {"incomingEdges": [_edge("x")]})]
    assert not merge._apply_labeled_external_entry_ports(unlabeled, set())
    duplicate_labels = [
        ("q", _edge("a"), {"incomingEdges": [_edge("a")]}),
        ("q", _edge("b"), {"incomingEdges": [_edge("b")]}),
    ]
    assert not merge._apply_labeled_external_entry_ports(duplicate_labels, set())

    assert (
        merge._infer_entry_port_label(
            {"sourceNodeId": "source", "metadata": {"port_label": "edge"}},
            {},
            {},
        )
        == "edge"
    )
    assert (
        merge._infer_entry_port_label(
            {"sourceNodeId": "source"},
            {"attrs": _attrs(port_label="target")},
            {},
        )
        == "target"
    )
    assert (
        merge._infer_entry_port_label(
            {"sourceNodeId": "source"},
            {},
            {"source": {"label": "v", "attrs": _attrs(synthetic="@tensor")}},
        )
        == "v"
    )


def test_merge_group_input_injection_fallbacks_and_skips():
    nodes = [
        {"id": "outside", "namespace": ""},
        {
            "id": "p:sub_0",
            "namespace": "root/KimiMoEGate",
            "incomingEdges": [_edge("outside")],
        },
        {
            "id": "p:sub_1",
            "namespace": "root/KimiMoEGate",
            "incomingEdges": [_edge("p:sub_0")],
        },
        {
            "id": "orphan",
            "namespace": "root/orphan",
        },
        {
            "id": "existing/@input",
            "namespace": "root/existing",
            "attrs": _attrs(synthetic="@input"),
        },
        {
            "id": "skip",
            "namespace": "root/skip",
        },
    ]
    merge._inject_group_inputs(nodes, skip_namespaces=frozenset({"root/skip"}))
    gate_input = next(
        node
        for node in nodes
        if node.get("namespace") == "root/KimiMoEGate"
        and merge._is_synthetic_input(node)
    )
    assert gate_input["label"] == "hidden_states"
    assert gate_input["incomingEdges"] == [_edge("outside")]
    assert next(node for node in nodes if node["id"] == "p:sub_0")["incomingEdges"] == [
        _edge(gate_input["id"])
    ]
    assert any(node["id"] == "orphan/@input" for node in nodes)
    assert not any(node["id"] == "skip/@input" for node in nodes)

    parent = "root/l2norm_fwd"
    assert labels.skip_merged_tensor_port_parent(
        parent, [{"id": "pipeline/forward_l2norm_fwd_q:sub_0"}]
    )
    assert not labels.skip_merged_tensor_port_parent(
        "root/l2norm_fwd_q", [{"id": "forward_l2norm_fwd_q"}]
    )


def test_merge_boundaries_connections_and_replacements():
    linear = [
        {"id": "a"},
        {"id": "b", "incomingEdges": [_edge("a")]},
        {"id": "c", "incomingEdges": [_edge("b")]},
    ]
    assert merge._boundary_nodes(linear) == (["a"], ["c"])
    cycle = [
        {"id": "a", "incomingEdges": [_edge("b")]},
        {"id": "b", "incomingEdges": [_edge("a")]},
    ]
    assert merge._boundary_nodes(cycle) == (["a"], ["b"])
    assert merge._boundary_nodes([]) == ([], [])

    section = [{"id": "entry"}, {"id": "other"}]
    merge._connect_external_inputs(
        section, namespace_prefix="missing", previous_exits=["x", "y"]
    )
    assert section[0]["incomingEdges"] == [_edge("x", "0"), _edge("y", "1")]
    unchanged = [dict(item) for item in section]
    merge._connect_external_inputs(
        section, namespace_prefix="missing", previous_exits=[]
    )
    assert section == unchanged

    tile = {"id": "tile", "incomingEdges": [_edge("producer")]}
    nested = [{"id": "nested"}]
    section = [tile, {"id": "consumer", "incomingEdges": [_edge("tile")]}, *nested]
    merge._replace_tile_with_group(section, nested, tile_id="absent", exit_id="out")
    assert tile in section
    merge._replace_tile_with_group(section, nested, tile_id="tile", exit_id=None)
    assert tile not in section
    assert nested[0]["incomingEdges"] == [_edge("producer")]


def test_merge_prefix_namespace_and_group_label_helpers():
    assert merge._common_id_prefix([]) == ""
    assert merge._common_id_prefix(["alpha/one", "alpha/two"]) == "alpha"
    assert (
        merge._group_input_prefix(
            ["p:forward_l2norm_fwd_q_sub_0", "p:forward_l2norm_fwd_q_sub_1"]
        )
        == "p:forward_l2norm_fwd_q"
    )
    assert merge._group_input_prefix(["prefix_sub_0/a", "prefix_sub_0/b"]) == "prefix"
    assert merge._namespace_is_descendant("", "")
    assert not merge._namespace_is_descendant("child", "")
    assert merge._namespace_is_descendant("a/b", "a")
    assert not merge._namespace_is_descendant("ab", "a")

    assert merge._infer_group_input_label([], "root/l2norm_fwd_q") == "q"
    assert merge._infer_group_input_label([], "root/KimiMLP") == "x"
    assert merge._infer_group_input_label([], "root/KimiMoEGate") == "hidden_states"
    assert (
        merge._infer_group_input_label(
            [{"attrs": _attrs(port_label="attr")}], "root/other"
        )
        == "attr"
    )
    assert (
        merge._infer_group_input_label([], "root/other", entry_ports=[("edge", {}, {})])
        == "edge"
    )
    assert merge._infer_group_input_label([], "root/other") == "hidden_states"

    inline_nodes = [
        {
            "id": "parent/@input",
            "namespace": "root/KimiMLP",
            "attrs": _attrs(synthetic="@input"),
        }
    ]
    assert merge._skip_nested_inline_frame_input(
        inline_nodes, "root/KimiMLP/SiluAndMul"
    )
    assert not merge._skip_nested_inline_frame_input(inline_nodes, "SituAndMul")
    assert not merge._skip_nested_inline_frame_input(inline_nodes, "root/Other")


def test_merge_section_exits_and_computation_node_filtering():
    computation = ComputationGraph(
        nodes=[
            GraphNodeSpec(key="@input", label="input", synthetic="@input"),
            GraphNodeSpec(key="work", label="Work"),
        ],
        links=[(0, 1)],
        primary_output_index=1,
    )
    nodes = merge._computation_nodes(
        computation,
        id_prefix="p",
        namespace_prefix="ns",
        skip_synthetic_input=True,
    )
    assert [node["id"] for node in nodes] == ["p/work"]
    assert nodes[0]["incomingEdges"][0]["sourceNodeId"] == (
        merge._SKIPPED_SECTION_INPUT
    )
    merge._connect_external_inputs(
        nodes,
        namespace_prefix="ns",
        previous_exits=["previous"],
    )
    assert nodes[0]["incomingEdges"][0]["sourceNodeId"] == "previous"
    assert merge._section_exits(computation, nodes, id_prefix="p") == ["p/work"]
    assert merge._section_exits(
        computation,
        [{"id": "replacement"}],
        id_prefix="p",
        replacements={"p/work": "replacement"},
    ) == ["replacement"]


def test_merge_kernel_pipeline_integration_and_invalid_port():
    namespace = "decoder/attn"
    pipeline = f"{namespace}/pipeline"
    merge_id = "merge"
    tensor = {
        "id": "tensor-q",
        "label": "q",
        "namespace": pipeline,
        "attrs": _attrs(synthetic="@tensor"),
    }
    kernel = {
        "id": "chunk_gated_delta_rule_fwd_h",
        "namespace": pipeline,
        "incomingEdges": [_edge("tensor-q")],
    }
    merge_node = {
        "id": merge_id,
        "namespace": namespace,
        "attrs": _attrs(
            attr_name="@attn_pipeline",
            class_name="KernelPipeline",
            details="detail",
        ),
        "incomingEdges": [_edge("q-source", "0"), _edge("bad", "not-an-int")],
    }
    output = {
        "id": "output",
        "namespace": namespace,
        "attrs": _attrs(attr_name="@attn_output"),
        "incomingEdges": [_edge(merge_id)],
    }
    nodes = [tensor, kernel, merge_node, output]
    group_attrs: dict[str, dict[str, str]] = {}
    skipped: set[str] = set()
    merge._integrate_kernel_pipeline_merge(
        nodes,
        namespace_prefix=namespace,
        pipeline_namespace=pipeline,
        pipeline_prefix="ignored",
        pipeline_label="Pipeline",
        group_node_attributes=group_attrs,
        inject_skip=skipped,
    )
    assert merge_node not in nodes
    assert tensor["incomingEdges"][0]["sourceNodeId"] == "q-source"
    assert tensor["incomingEdges"][0]["metadata"] == {"port_label": "q"}
    assert output["incomingEdges"][0]["sourceNodeId"] == kernel["id"]
    assert group_attrs[pipeline] == {
        "label": "Pipeline",
        "operation": "kernel pipeline",
        "details": "detail",
    }
    assert skipped == {pipeline}

    untouched = [{"id": "x"}]
    merge._integrate_kernel_pipeline_merge(
        untouched,
        namespace_prefix="n",
        pipeline_namespace="n/p",
        pipeline_prefix="p",
        pipeline_label="P",
    )
    assert untouched == [{"id": "x"}]


def test_overview_defaults_order_labels_and_namespaces():
    spec = _spec(
        block_components=[
            _component("late", order=None),
            _component("norm", "norm", label="", order=0),
            _component("head", "head", class_name="Linear", order=1),
        ]
    )
    assert [item.attr_name for item in overview._ordered_decoder_components(spec)] == [
        "norm",
        "head",
        "late",
    ]
    assert overview._display_label(spec.block_components[0], spec) == "late"
    assert overview._display_label(spec.block_components[1], spec) == "RMSNorm"
    assert overview._display_label(spec.block_components[2], spec) == "Linear"
    assert overview._decoder_namespace(spec) == "2x_Dense"
    assert (
        overview._decoder_namespace(
            _spec(num_hidden_layers=None, decoder_class="Layer")
        )
        == "?x_Layer"
    )

    assert overview._stack_pre_components(spec)[0].attr_name == "embed_tokens"
    assert [item.attr_name for item in overview._stack_tail_components(spec)] == [
        "norm",
        "lm_head",
    ]
    assert overview._stack_pre_components(_spec(vocab_size=None)) == []
    assert overview._stack_tail_components(_spec(vocab_size=None, norm_type="")) == []

    variant = LayerVariant(
        "v", 2, "Attn", "VariantAttn", "FFN", "VariantFFN", "experts"
    )
    attention = _component("self_attn", "attention", class_name="Base")
    ffn = _component("experts", "moe", class_name="BaseFFN")
    assert overview._component_uses_variant_attention_class(attention, variant)
    assert overview._component_uses_variant_ffn_class(ffn, variant)
    assert not overview._component_uses_variant_ffn_class(ffn, None)
    assert (
        overview._section_namespace_segment(spec, attention, variant=variant)
        == "VariantAttn"
    )
    assert (
        overview._section_namespace_segment(spec, ffn, variant=variant) == "VariantFFN"
    )
    assert (
        overview._flat_spine_namespace(attention, "decoder", variant=variant)
        == "decoder"
    )
    assert (
        overview._flat_spine_namespace(
            _component("embed", "embedding"), "ignored", variant=None
        )
        == ""
    )


def test_overview_forward_labels_and_build_graph():
    components = [
        _component("input_layernorm", "norm", class_name="RMSNorm", order=0),
        _component("self_attn", "attention", label="Attention", order=1),
    ]
    spec = _spec(
        decoder_class="Decoder",
        block_components=components,
        forward_sequence=["input_layernorm", "self_attn", "unknown_step"],
        stack_pre=[_component("embed", "embedding", label="Embedding")],
        stack_tail=[_component("head", "head", class_name="Linear")],
    )
    assert overview.forward_sequence_display_labels(spec) == [
        "RMSNorm",
        "Attention",
        "unknown step",
    ]
    assert (
        overview.format_forward_sequence(spec, arrow=" -> ")
        == "RMSNorm -> Attention -> unknown step"
    )
    graph = overview.build_overview_graph(
        spec, attr_name_to_graph_id={"self_attn": "attention-detail"}
    )
    by_id = {node["id"]: node for node in graph["nodes"]}
    assert by_id["decoder/self_attn"]["subgraphIds"] == ["attention-detail"]
    assert by_id["head"]["label"] == "Linear"
    assert by_id["embed"]["incomingEdges"][0]["sourceNodeId"] == "@input"
    assert graph["groupNodeAttributes"][""]["forward"].endswith("unknown step")


def test_overview_shared_classes_and_style_fallbacks():
    same_a = _component("a", "attention", class_name="Shared")
    same_b = _component("b", "attention", class_name="Shared")
    spec = _spec(block_components=[same_a, same_b])
    assert overview._shared_decoder_class_attr_names(spec, "Shared") == ["a", "b"]
    assert overview._section_namespace_segment(spec, same_a) == "a"
    assert overview._subgraph_ids("a", {"a": "graph"}) == ["graph"]
    assert overview._subgraph_ids("x", {}) is None
    assert overview._style_for_component(same_a)["backgroundColor"] == "#5dade2"
    assert (
        overview._style_for_component(_component("x"))["backgroundColor"] == "#bdc3c7"
    )
    assert (
        overview._spine_moe_class(
            _spec(block_components=[_component("moe", "moe", class_name="MoE")])
        )
        == "MoE"
    )
    assert (
        overview._spine_moe_class(_spec(layer_variants=[LayerVariant("v", 1, "A")]))
        is None
    )


def test_styles_readability_finalization_and_group_config_ordering():
    original = {"backgroundColor": "#8E44AD", "textColor": "#ffffff"}
    normalized = styles.ensure_readable_text(original)
    assert normalized == {"backgroundColor": "#f5d9d9", "textColor": "#1a1a1a"}
    assert original["backgroundColor"] == "#8E44AD"
    assert (
        styles.ensure_readable_text(
            {"backgroundColor": "#3A4550", "textColor": "wrong"}
        )["textColor"]
        == "#ffffff"
    )
    assert styles.ensure_readable_text({"backgroundColor": "#ffffff"}) == {
        "backgroundColor": "#ffffff"
    }

    nodes = [
        {"style": {"backgroundColor": "#5dade2", "textColor": "wrong"}},
        {"style": "not-a-dict"},
        {"label": "Multiply"},
        {"label": "Unsqueeze"},
        {"label": "Reshape", "style": {"backgroundColor": "#bdc3c7"}},
        {"label": "Split"},
        {"label": "Concat", "style": {"backgroundColor": "#bdc3c7"}},
    ]
    styles.finalize_graph_node_styles(nodes)
    assert nodes[0]["style"]["textColor"] == "#ffffff"
    assert nodes[1]["style"] == "not-a-dict"
    # Unstyled merge-synthesized tiles pick a fill from what the op does.
    assert nodes[2]["style"]["backgroundColor"] == "#bdc3c7"
    assert nodes[3]["style"]["backgroundColor"] == "#ffffff"
    # A layout-only op never keeps the computation gray.
    assert nodes[4]["style"]["backgroundColor"] == "#ffffff"
    assert nodes[5]["style"]["backgroundColor"] == "#ffffff"
    assert nodes[6]["style"]["backgroundColor"] == "#ffffff"

    attrs = {
        "decoder/KimiMoEGate": {"label": "Gate"},
        "decoder/Attention": {"operation": "gpu_kernel"},
        "decoder/KimiDeltaAttention": {"label": "Attention"},
        "": {"title": "model"},
    }
    configs = styles.build_group_node_configs(
        decoder_namespace="decoder",
        group_node_attributes=attrs,
        role_configs=[{"namespaceRegex": "role", "backgroundColor": "#fff"}],
    )
    assert configs[0]["namespaceRegex"].startswith("^decoder/")
    assert any(item["namespaceRegex"] == "^role$" for item in configs)
    assert configs[-1]["namespaceRegex"] == "^decoder$"
    assert (
        next(item for item in configs if "KimiMoEGate" in item["namespaceRegex"])[
            "borderColor"
        ]
        == "#d98888"
    )


def test_output_port_style_matches_input_blue():
    """@output boundary ports carry the same input-blue fill as @input ports.

    Part D: an @output tile is a data boundary, not computation, so it must read
    as an interface port (input blue ``#d9e8f5``) rather than the old distinct
    output color — matching ``input_port_style`` exactly.
    """
    out_style = styles.output_port_style()
    assert out_style["backgroundColor"] == "#d9e8f5"
    assert out_style == styles.input_port_style()


@pytest.mark.parametrize(
    ("synthetic", "label", "operation", "has_children", "background"),
    [
        ("@input", "", OperationKind.UNKNOWN, False, "#d9e8f5"),
        ("@combine", "", OperationKind.UNKNOWN, False, "#bdc3c7"),
        (None, "Add", OperationKind.UNKNOWN, False, "#bdc3c7"),
        (None, "", OperationKind.SYNTHETIC, False, "#ecf0f1"),
        (None, "", OperationKind.GPU_KERNEL, False, "#f5d9d9"),
        (None, "", OperationKind.COMPOSITE, True, "#5dade2"),
        (None, "", OperationKind.COMPOSITE, False, "#bdc3c7"),
    ],
)
def test_detail_tile_style_paths(
    monkeypatch: pytest.MonkeyPatch,
    synthetic: str | None,
    label: str,
    operation: OperationKind,
    has_children: bool,
    background: str,
):
    monkeypatch.setattr(
        "TraceLens.ModelUtils.model_graph.classify_operation", lambda *a, **k: operation
    )
    block = BlockNode("b", "Block", "other", "Block")
    if has_children:
        block.children.append(BlockNode("leaf", "Linear", "other", "Linear"))
    result = styles.detail_tile_style(block, synthetic=synthetic, label=label)
    assert result["backgroundColor"] == background


def test_labels_frame_detection_splitting_and_sanitization():
    node = {
        "id": "pipeline/forward_l2norm_fwd_q:sub_0",
        "attrs": _attrs(attr_name="forward_l2norm_fwd_q_sub_0"),
    }
    assert labels._frame_owner_attr_name(node) == "forward_l2norm_fwd_q"
    assert labels.tensor_port_frame_key(node) == "l2norm_fwd_q"
    assert labels.tensor_port_frame_key({"id": "plain"}) is None
    assert labels.split_tensor_port_namespace("root/l2norm_fwd", "l2norm_fwd_q") == (
        "root/l2norm_fwd_q"
    )
    assert labels.split_tensor_port_namespace("l2norm_fwd", "l2norm_fwd_k") == (
        "l2norm_fwd_k"
    )
    assert labels.split_tensor_port_namespace("root", "not-a-port") == "root"
    assert labels.frame_group_label("l2norm_fwd_q").endswith("(q)")
    assert labels.kernel_subop_display_label(" ÷ ") == "1/x"
    assert labels.kernel_subop_display_label("a × scale → b − c • d") == (
        "a x scale -> b - c - d"
    )
    assert labels.tensor_port_input_label("root/l2norm_fwd_v") == "v"
    assert labels.tensor_port_input_label("root/plain") is None


def test_labels_apply_preserves_existing_group_attrs_and_updates_subops():
    namespace = "root/l2norm_fwd"
    nodes = [
        {
            "id": "pipeline/forward_l2norm_fwd_q:sub_0",
            "label": "×",
            "namespace": namespace,
            "attrs": _attrs(
                attr_name="forward_l2norm_fwd_q_sub_0", class_name="KernelSubOp"
            ),
        },
        {
            "id": "pipeline/forward_l2norm_fwd_q:sub_1",
            "label": 7,
            "namespace": namespace,
            "attrs": _attrs(
                attr_name="forward_l2norm_fwd_q_sub_1", class_name="KernelSubOp"
            ),
        },
    ]
    expected_namespace = "root/l2norm_fwd_q"
    group_attrs = {expected_namespace: {"label": "Custom"}}
    labels.apply_kernel_frame_labels(nodes, group_attrs)
    assert all(node["namespace"] == expected_namespace for node in nodes)
    assert nodes[0]["label"] == "X"
    assert nodes[1]["label"] == 7
    assert group_attrs[expected_namespace] == {"label": "Custom"}


def test_shape_format_annotation_and_empty_shape_paths():
    spec = TensorSpec(("B", "2×H", "N∗D", "λ"), "bfloat16")
    # Human display is bracketed and keeps ``*`` for merged dims; the unicode
    # multiply signs fold to ``*`` and the non-ASCII ``λ`` drops out.
    assert shapes.format_shape(spec) == "[B, 2*H, N*D, ]"
    # tensor_shape (edge labels) now uses the same bracket form as node attrs.
    assert shapes.format_shape_tensor(spec) == "[B, 2*H, N*D, ] bfloat16"
    assert shapes.format_shape_bracket(spec) == "[B, 2×H, N∗D, λ]"

    node = {
        "id": "p/key",
        "attrs": [
            {"key": "output_shape", "value": "old"},
            {"key": "keep", "value": "yes"},
        ],
    }
    shapes.annotate_nodes_with_shapes(
        [node, {"id": "other"}], {"key": TensorSpec((1, 2), "float32")}, id_prefix="p"
    )
    assert node["attrs"] == [
        {"key": "keep", "value": "yes"},
        {"key": "output_shape", "value": "[1, 2] float32"},
        {"key": "output_dtype", "value": "float32"},
    ]
    assert shapes._node_spec(node) == TensorSpec(("1", "2"), "float32")
    empty = {"id": "empty"}
    shapes._apply_shape_attrs(empty, TensorSpec((), "float16"))
    assert empty == {"id": "empty"}
    shapes.annotate_nodes_with_shapes([empty], {}, id_prefix="")


def test_unconsumed_output_port_prunes_its_producer_backwards():
    nodes = [
        {"id": "@input", "attrs": [{"key": "synthetic", "value": "@input"}]},
        {
            "id": "live",
            "incomingEdges": [{"sourceNodeId": "@input", "sourceNodeOutputId": "0"}],
        },
        {
            "id": "dead_setup",
            "incomingEdges": [{"sourceNodeId": "@input", "sourceNodeOutputId": "0"}],
        },
        {
            "id": "dead_result",
            "incomingEdges": [
                {"sourceNodeId": "dead_setup", "sourceNodeOutputId": "0"}
            ],
        },
        {
            "id": "block/@output",
            "attrs": [{"key": "synthetic", "value": "@output"}],
            "inputsMetadata": [{"id": "used"}, {"id": "unused"}],
            "outputsMetadata": [{"id": "used"}, {"id": "unused"}],
            "incomingEdges": [
                {
                    "sourceNodeId": "live",
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "used",
                },
                {
                    "sourceNodeId": "dead_result",
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "unused",
                },
            ],
        },
        {
            "id": "consumer",
            "incomingEdges": [
                {
                    "sourceNodeId": "block/@output",
                    "sourceNodeOutputId": "used",
                }
            ],
        },
        {
            "id": "@output",
            "attrs": [{"key": "synthetic", "value": "@output"}],
            "incomingEdges": [{"sourceNodeId": "consumer", "sourceNodeOutputId": "0"}],
        },
    ]

    merge._prune_unconsumed_outputs(nodes)

    by_id = {node["id"]: node for node in nodes}
    assert "dead_setup" not in by_id
    assert "dead_result" not in by_id
    assert by_id["block/@output"]["outputsMetadata"] == [{"id": "used"}]
    assert {
        edge["targetNodeInputId"] for edge in by_id["block/@output"]["incomingEdges"]
    } == {"used"}


def test_no_consumer_op_pruned_but_chains_sinks_and_loop_carried_kept():
    nodes = [
        {"id": "@input", "attrs": [{"key": "synthetic", "value": "@input"}]},
        # A no-consumer leaf sitting on a *shared* tensor (a Clamp feeding only an
        # index the graph never models) — removing it orphans nothing, so drop it.
        {
            "id": "dead_clamp",
            "label": "Clamp",
            "incomingEdges": [{"sourceNodeId": "@input", "sourceNodeOutputId": "0"}],
        },
        # A no-consumer leaf capping a *dedicated* chain — its producer feeds only
        # it. This mirrors the DSA indexer's TopK/Expand, whose real consumers are
        # index/in-place ops the tracer can't model. Removing the leaf would orphan
        # legitimate compute, so the whole chain is preserved (not unravelled).
        {
            "id": "chain_producer",
            "label": "Split",
            "incomingEdges": [{"sourceNodeId": "@input", "sourceNodeOutputId": "0"}],
        },
        {
            "id": "chain_leaf",
            "label": "Clamp",
            "incomingEdges": [
                {"sourceNodeId": "chain_producer", "sourceNodeOutputId": "0"}
            ],
        },
        # A live op consumed by @output must survive.
        {
            "id": "live",
            "label": "Linear",
            "incomingEdges": [{"sourceNodeId": "@input", "sourceNodeOutputId": "0"}],
        },
        # A loop-carried-in with no consumer yet is wired later — never pruned here.
        {
            "id": "block/@loop_carried_in:loop_l1_c1:h",
            "label": "Loop in - iterations:N",
            "attrs": [{"key": "synthetic", "value": "@loop_carried"}],
            "incomingEdges": [{"sourceNodeId": "@input", "sourceNodeOutputId": "0"}],
        },
        {
            "id": "@output",
            "attrs": [{"key": "synthetic", "value": "@output"}],
            "incomingEdges": [{"sourceNodeId": "live", "sourceNodeOutputId": "0"}],
        },
    ]

    merge._prune_unconsumed_outputs(nodes)

    ids = {node["id"] for node in nodes}
    assert "dead_clamp" not in ids  # shared-input leaf: orphans nothing → pruned
    assert "chain_leaf" in ids  # dedicated chain preserved (indexer-like)
    assert "chain_producer" in ids
    assert "live" in ids
    assert "block/@loop_carried_in:loop_l1_c1:h" in ids  # kept for later wiring
    assert "@output" in ids


def test_fill_repeated_loop_counts_from_repeat_namespace():
    def _lc(node_id, namespace, sublabel):
        return {
            "id": node_id,
            "namespace": namespace,
            "attrs": [
                {"key": "synthetic", "value": "@loop_carried"},
                {"key": "sublabel", "value": sublabel},
            ],
        }

    nodes = [
        # ModuleList loop (no static range bound) inside a 24× repeat group.
        _lc("visual/@loop_carried_in:l1:hidden_states",
            "visual/24x_Glm5NextVisionBlock", "hidden_states · repeated"),
        # Nearest (deepest) repeat segment wins when nested.
        _lc("d/@loop_carried_in:l2:h", "45x_Decoder/2x_Inner", "h · repeated"),
        # Already-counted inner loop is left untouched.
        _lc("d/@loop_carried_in:l3:comb", "45x_Decoder/Loop_19_iterations",
            "comb · 19 iterations"),
        # A loop-carried boundary with no enclosing repeat group stays "repeated".
        _lc("x/@loop_carried_in:l4:y", "x", "y · repeated"),
    ]

    merge._fill_repeated_loop_counts(nodes)

    def _sub(node):
        return next(a["value"] for a in node["attrs"] if a["key"] == "sublabel")

    assert _sub(nodes[0]) == "hidden_states · 24 iterations"
    assert _sub(nodes[1]) == "h · 2 iterations"  # deepest 2x_ wins, not 45x_
    assert _sub(nodes[2]) == "comb · 19 iterations"  # unchanged
    assert _sub(nodes[3]) == "y · repeated"  # no repeat namespace → unchanged


def _synthetic_input(node_id, namespace, source, *, label="hidden_states"):
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": "@input"}],
        "incomingEdges": [{"sourceNodeId": source, "sourceNodeOutputId": "0",
                           "targetNodeInputId": "0"}],
    }


def _synthetic_output(node_id, namespace, source):
    return {
        "id": node_id,
        "label": "Output",
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": "@output"}],
        "incomingEdges": [{"sourceNodeId": source, "sourceNodeOutputId": "result",
                           "targetNodeInputId": "0"}],
    }


def _plain(node_id, namespace, sources, *, shape=None):
    node = {"id": node_id, "namespace": namespace, "attrs": [],
            "incomingEdges": [{"sourceNodeId": s, "sourceNodeOutputId": "0",
                               "targetNodeInputId": str(i)} for i, s in enumerate(sources)]}
    if shape is not None:
        node["attrs"].append({"key": "output_shape", "value": shape})
    return node


def _lc_in_edges(node):
    return [(e["sourceNodeId"], e.get("metadata", {}).get("port_label"))
            for e in node["incomingEdges"]]


def _consumers(nodes, node_id):
    return [n["id"] for n in nodes for e in n.get("incomingEdges", [])
            if e["sourceNodeId"] == node_id]


def test_synthesize_loop_boundary_suppressed_for_heterogeneous_container():
    """A container whose iterations run *different* modules (multiple distinct
    variant exit sources) is NOT wrapped in a loop-carried pair.

    A single loop-carried abstraction would misrepresent a container that is
    really a *sequence* of distinct variant runs (decoder: 31 layers of one
    class, then 11, then 3), so ``_synthesize_repeat_loop_boundaries`` returns
    early and leaves the pre-synthesis direct wiring untouched: every variant
    ``@input`` still reads the external producer and the post-container consumer
    still reads the variant ``@output``s directly. No ``@loop_carried`` tile and
    no back edge are created.
    """
    nodes = [
        _plain("src", "", [], shape="[B, S, 4, 4096] bfloat16"),
        _synthetic_input("dec/3x_A/@input", "45x_Dec/3x_A", "src"),
        _synthetic_input("dec/2x_B/@input", "45x_Dec/2x_B", "src"),
        _synthetic_output("dec/3x_A/@output", "45x_Dec/3x_A", "dec/3x_A/@input"),
        _synthetic_output("dec/2x_B/@output", "45x_Dec/2x_B", "dec/2x_B/@input"),
        _plain("head", "", ["dec/3x_A/@output", "dec/2x_B/@output"]),
        _plain("sink", "", ["head"]),
    ]
    merge._synthesize_repeat_loop_boundaries(nodes)
    by_id = {n["id"]: n for n in nodes}

    # No loop-carried tiles are synthesized for the heterogeneous container.
    assert "dec/@loop_carried_in:dec:hidden_states" not in by_id
    assert "dec/@loop_carried_out:dec:hidden_states" not in by_id
    assert not any(
        merge._node_attr(n, "synthetic") == "@loop_carried" for n in nodes)

    # Direct wiring is preserved: both variant @inputs still read the raw source,
    # and the post-container head still reads the variant @outputs directly.
    assert _lc_in_edges(by_id["dec/3x_A/@input"]) == [("src", None)]
    assert _lc_in_edges(by_id["dec/2x_B/@input"]) == [("src", None)]
    assert [e["sourceNodeId"] for e in by_id["head"]["incomingEdges"]] == [
        "dec/3x_A/@output", "dec/2x_B/@output"]


def test_synthesize_loop_boundary_wraps_single_template_container():
    """A single-template group wraps its lone @output the same way."""
    nodes = [
        _plain("src", "", [], shape="[B, S, 4096] bfloat16"),
        _synthetic_input("dec/@input", "45x_Dec", "src"),
        _synthetic_output("dec/@output", "45x_Dec", "dec/@input"),
        _plain("sink", "", ["dec/@output"]),
    ]
    merge._synthesize_repeat_loop_boundaries(nodes)
    by_id = {n["id"]: n for n in nodes}
    in_id = "dec/@loop_carried_in:dec:hidden_states"
    out_id = "dec/@loop_carried_out:dec:hidden_states"

    assert set(_lc_in_edges(by_id[in_id])) == {("src", None), (out_id, "next iteration")}
    assert _consumers(nodes, in_id) == ["dec/@input"]
    assert set(_lc_in_edges(by_id[out_id])) == {("dec/@output", "updated")}
    assert sorted(_consumers(nodes, out_id)) == [in_id, "sink"]


def test_synthesize_loop_boundary_is_noop_when_already_wrapped():
    """The vision tower's CG-built boundary is left untouched (no double wrap)."""
    nodes = [
        _plain("src", "", [], shape="[Pv, 1176] bfloat16"),
        {
            "id": "visual/@loop_carried_in:l1:hidden_states",
            "namespace": "visual/24x_Block",
            "attrs": [{"key": "synthetic", "value": "@loop_carried"}],
            "incomingEdges": [{"sourceNodeId": "src", "sourceNodeOutputId": "0",
                               "targetNodeInputId": "0"}],
        },
        _synthetic_output("visual/@output", "visual/24x_Block",
                          "visual/@loop_carried_in:l1:hidden_states"),
        _plain("sink", "", ["visual/@output"]),
    ]
    before = len(nodes)
    merge._synthesize_repeat_loop_boundaries(nodes)
    assert len(nodes) == before
    assert sum("@loop_carried_in:" in n["id"] for n in nodes) == 1


def test_synthesize_loop_boundary_skips_multi_source_container():
    """Multiple external inputs (loop-invariant reads) are not yet hoisted -> skip."""
    nodes = [
        _plain("hidden", "", [], shape="[B, S, D] bfloat16"),
        _plain("cos", "", [], shape="[S, D] bfloat16"),
        _synthetic_input("g/24x_Block/@input", "24x_Block", "hidden"),
        _synthetic_input("g/24x_Block/@input:cos", "24x_Block", "cos", label="cos"),
        _synthetic_output("g/24x_Block/@output", "24x_Block", "g/24x_Block/@input"),
        _plain("sink", "", ["g/24x_Block/@output"]),
    ]
    before = len(nodes)
    merge._synthesize_repeat_loop_boundaries(nodes)
    assert len(nodes) == before
    assert not any("@loop_carried" in n["id"] for n in nodes)


def _kernel_in(node_id, namespace, source, *, label):
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": "@kernel_port_in"}],
        "incomingEdges": [{"sourceNodeId": source, "sourceNodeOutputId": "0",
                           "targetNodeInputId": "0"}],
    }


def _input_mirror(node_id, namespace, source, *, label):
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": "@input_mirror"},
                  {"key": "port_label", "value": label}],
        "incomingEdges": [{"sourceNodeId": source, "sourceNodeOutputId": label,
                           "targetNodeInputId": label}],
    }


def _output_tile(node_id, namespace, source, *, label, synth="@output"):
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": synth},
                  {"key": "port_label", "value": label}],
        "incomingEdges": [{"sourceNodeId": source, "sourceNodeOutputId": "0",
                           "targetNodeInputId": "0"}],
    }


def test_collapse_kernel_input_passthrough_drops_redundant_module_input():
    """A kernel port fed only by a same-scope same-name @input tile collapses:
    the untransformed passthrough (cu_seqlens) loses its redundant @input tile."""
    attn = "blk/Attn"
    nodes = [
        _plain("producer", "", [], shape="[Pv, 1176] bfloat16"),
        _input_mirror("blk/@input_mirror:cu_seqlens^cu_seqlens", "blk", "producer",
                      label="cu_seqlens"),
        _synthetic_input("blk/@input:cu_seqlens", attn,
                         "blk/@input_mirror:cu_seqlens^cu_seqlens", label="cu_seqlens"),
        _kernel_in("blk/@kernel_in:9:cu_seqlens", attn, "blk/@input:cu_seqlens",
                   label="cu_seqlens"),
        # Control: a real kernel input (query_states) fed by a computation, not an
        # @input tile -- must be left untouched (the apply_rotary reference shape).
        _plain("rotary_q", attn, [], shape="[Pv, 64] bfloat16"),
        _kernel_in("blk/@kernel_in:9:query_states", attn, "rotary_q",
                   label="query_states"),
        _plain("kernel", attn, ["blk/@kernel_in:9:cu_seqlens",
                                 "blk/@kernel_in:9:query_states"]),
    ]
    merge._collapse_kernel_input_passthroughs(nodes)
    by_id = {n["id"]: n for n in nodes}

    # The redundant module-input tile is gone; the kernel port reads the mirror.
    assert "blk/@input:cu_seqlens" not in by_id
    cu_port = by_id["blk/@kernel_in:9:cu_seqlens"]
    assert [e["sourceNodeId"] for e in cu_port["incomingEdges"]] == [
        "blk/@input_mirror:cu_seqlens^cu_seqlens"
    ]
    # The real kernel input is untouched (its source is a computation, not @input).
    q_port = by_id["blk/@kernel_in:9:query_states"]
    assert [e["sourceNodeId"] for e in q_port["incomingEdges"]] == ["rotary_q"]


def test_collapse_kernel_input_passthrough_keeps_shared_module_input():
    """An @input tile with a second consumer is NOT a pure passthrough -> kept."""
    attn = "blk/Attn"
    nodes = [
        _plain("producer", "", [], shape="[Pv, 1176] bfloat16"),
        _synthetic_input("blk/@input:mask", attn, "producer", label="mask"),
        _kernel_in("blk/@kernel_in:9:mask", attn, "blk/@input:mask", label="mask"),
        # Second consumer of the same @input tile: a real op reads it too.
        _plain("other_op", attn, ["blk/@input:mask"]),
        _plain("kernel", attn, ["blk/@kernel_in:9:mask"]),
    ]
    merge._collapse_kernel_input_passthroughs(nodes)
    by_id = {n["id"]: n for n in nodes}
    # Shared input tile survives; the kernel port still reads it.
    assert "blk/@input:mask" in by_id
    assert [e["sourceNodeId"] for e in by_id["blk/@kernel_in:9:mask"]["incomingEdges"]] == [
        "blk/@input:mask"
    ]


def test_collapse_same_name_boundary_keeps_cross_module_crossing():
    """A same-name ``@output`` -> ``@input`` crossing ENTERS a different module,
    so both boundary tiles are kept.

    ``input_layernorm/@output:hidden_states`` feeding ``self_attn/@input:
    hidden_states`` is a genuine module entry: the consuming module legitimately
    declares its own ``@input`` boundary. Collapsing across the namespace edge
    would delete ``self_attn``'s ``@input`` (defect B); the same-hierarchy guard
    keeps both tiles and leaves the consumer reading its own ``@input``."""
    nodes = [
        _plain("real_op", "mod_a", [], shape="[B, S, H] bfloat16"),
        _output_tile("mod_a/@output:hidden_states", "mod_a", "real_op",
                     label="hidden_states"),
        _synthetic_input("mod_b/@input:hidden_states", "mod_b",
                         "mod_a/@output:hidden_states", label="hidden_states"),
        _plain("consumer", "mod_b", ["mod_b/@input:hidden_states"]),
    ]
    merge._collapse_same_name_boundary_passthroughs(nodes)
    by_id = {n["id"]: n for n in nodes}

    assert "mod_b/@input:hidden_states" in by_id
    assert "mod_a/@output:hidden_states" in by_id
    assert [e["sourceNodeId"] for e in by_id["consumer"]["incomingEdges"]] == [
        "mod_b/@input:hidden_states"
    ]


def test_collapse_same_name_boundary_folds_same_namespace_pair():
    """A same-name ``@output`` -> ``@input`` pair WITHIN one module namespace is a
    redundant tile rendered twice (no hierarchy crossing) and still folds to one.

    Both tiles share the owning namespace ``mod`` (the token before the trailing
    ``/@...`` boundary marker), so this is not a module entry/exit -- the consumer
    ``@input`` tile is dropped and repointed onto the surviving ``@output``."""
    nodes = [
        _plain("mod/real_op", "mod", [], shape="[B, S, H] bfloat16"),
        _output_tile("mod/@output:hidden_states", "mod", "mod/real_op",
                     label="hidden_states"),
        _synthetic_input("mod/@input:hidden_states", "mod",
                         "mod/@output:hidden_states", label="hidden_states"),
        _plain("mod/consumer", "mod", ["mod/@input:hidden_states"]),
    ]
    merge._collapse_same_name_boundary_passthroughs(nodes)
    by_id = {n["id"]: n for n in nodes}

    assert "mod/@input:hidden_states" not in by_id
    assert "mod/@output:hidden_states" in by_id
    assert [e["sourceNodeId"] for e in by_id["mod/consumer"]["incomingEdges"]] == [
        "mod/@output:hidden_states"
    ]


def test_collapse_same_name_boundary_kernel_port_reads_real_op():
    """A same-name ``@output`` -> ``@kernel_port_in`` reduces to the port reading
    the real op: the kernel keeps its declared port but now reads ``expand_kv``'s
    ``Copy`` directly instead of an interposed ``@output`` tile."""
    nodes = [
        _plain("copy_op", "attn", [], shape="[Pv, 64] bfloat16"),
        _output_tile("attn/@output:key_states", "attn", "copy_op", label="key_states"),
        _kernel_in("attn/@kernel_in:key_states", "attn", "attn/@output:key_states",
                   label="key_states"),
        _plain("kernel", "attn", ["attn/@kernel_in:key_states"]),
    ]
    merge._collapse_same_name_boundary_passthroughs(nodes)
    by_id = {n["id"]: n for n in nodes}

    assert "attn/@output:key_states" not in by_id
    assert "attn/@kernel_in:key_states" in by_id
    assert [e["sourceNodeId"] for e in by_id["attn/@kernel_in:key_states"]["incomingEdges"]] == [
        "copy_op"
    ]


def test_collapse_same_name_boundary_reduces_three_tile_chain():
    """An ``@output`` -> ``@output_mirror`` -> ``@kernel_port_in`` chain of the same
    name collapses iteratively to a single edge from the real op."""
    nodes = [
        _plain("copy_op", "attn", [], shape="[Pv, 64] bfloat16"),
        _output_tile("attn/@output:value_states", "attn", "copy_op",
                     label="value_states"),
        _output_tile("attn/@output:value_states^value_states", "attn",
                     "attn/@output:value_states", label="value_states",
                     synth="@output_mirror"),
        _kernel_in("attn/@kernel_in:value_states", "attn",
                   "attn/@output:value_states^value_states", label="value_states"),
        _plain("kernel", "attn", ["attn/@kernel_in:value_states"]),
    ]
    merge._collapse_same_name_boundary_passthroughs(nodes)
    by_id = {n["id"]: n for n in nodes}

    assert "attn/@output:value_states^value_states" not in by_id
    assert "attn/@output:value_states" not in by_id
    assert [e["sourceNodeId"] for e in by_id["attn/@kernel_in:value_states"]["incomingEdges"]] == [
        "copy_op"
    ]


def test_collapse_same_name_boundary_leaves_renamed_crossing_intact():
    """A boundary where the name genuinely changes (``@output:collapsed`` ->
    ``@input:hidden_states``) is a real rename, not a passthrough, and is kept."""
    nodes = [
        _plain("real_op", "mod_a", [], shape="[B, S, H] bfloat16"),
        _output_tile("mod_a/@output:collapsed", "mod_a", "real_op", label="collapsed"),
        _synthetic_input("mod_b/@input:hidden_states", "mod_b",
                         "mod_a/@output:collapsed", label="hidden_states"),
        _plain("consumer", "mod_b", ["mod_b/@input:hidden_states"]),
    ]
    merge._collapse_same_name_boundary_passthroughs(nodes)
    by_id = {n["id"]: n for n in nodes}

    assert "mod_b/@input:hidden_states" in by_id
    assert [e["sourceNodeId"] for e in by_id["consumer"]["incomingEdges"]] == [
        "mod_b/@input:hidden_states"
    ]


def _cast_node(node_id: str, *, source: str, dtype: str, shape: str = "B x S x 4") -> dict:
    """A `Cast` node with one incoming edge and its own inferred output dtype."""
    return {
        "id": node_id,
        "label": "Cast",
        "incomingEdges": [_edge(source)],
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


def _real_node(node_id: str, *, dtype: str, source: str | None = None) -> dict:
    """A non-Cast node with its own inferred output dtype (a trustworthy producer)."""
    node = {
        "id": node_id,
        "label": "Multiply",
        "outputsMetadata": [
            {
                "id": "0",
                "attrs": [
                    {"key": "shape", "value": f"B x S x 4 {dtype}"},
                    {"key": "dtype", "value": dtype},
                ],
            }
        ],
    }
    if source:
        node["incomingEdges"] = [_edge(source)]
    return node


def test_prune_noop_cast_removes_same_dtype_cast_and_rewires_consumer():
    """A `Cast` whose output dtype equals its (real, non-synthetic) input's
    dtype is a no-op — it must be removed and its consumer rewired straight
    to the real producer."""
    nodes = [
        _real_node("producer", dtype="bfloat16"),
        _cast_node("noop_cast", source="producer", dtype="bfloat16"),
        {
            "id": "consumer",
            "incomingEdges": [_edge("noop_cast")],
        },
    ]

    merge._prune_noop_cast_nodes(nodes)

    by_id = {node["id"]: node for node in nodes}
    assert "noop_cast" not in by_id
    assert by_id["consumer"]["incomingEdges"][0]["sourceNodeId"] == "producer"


def test_prune_noop_cast_keeps_genuine_dtype_change():
    """A `Cast` that actually changes dtype must remain in the graph."""
    nodes = [
        _real_node("producer", dtype="bfloat16"),
        _cast_node("real_cast", source="producer", dtype="float32"),
        {
            "id": "consumer",
            "incomingEdges": [_edge("real_cast")],
        },
    ]

    merge._prune_noop_cast_nodes(nodes)

    by_id = {node["id"]: node for node in nodes}
    assert "real_cast" in by_id
    assert by_id["consumer"]["incomingEdges"][0]["sourceNodeId"] == "real_cast"


def test_prune_noop_cast_skips_synthetic_predecessor():
    """A synthetic boundary port's OWN dtype has no independent ground
    truth (it can be back-filled from whatever it feeds) — a same-dtype
    match against one must NOT be treated as proof of a no-op, or a
    genuine cast right after a boundary would be wrongly erased."""
    nodes = [
        {
            "id": "@input:gate",
            "attrs": [{"key": "synthetic", "value": "@input"}],
            "outputsMetadata": [
                {
                    "id": "0",
                    "attrs": [
                        {"key": "shape", "value": "B x S x 4 float32"},
                        {"key": "dtype", "value": "float32"},
                    ],
                }
            ],
        },
        _cast_node("maybe_real_cast", source="@input:gate", dtype="float32"),
        {
            "id": "consumer",
            "incomingEdges": [_edge("maybe_real_cast")],
        },
    ]

    merge._prune_noop_cast_nodes(nodes)

    by_id = {node["id"]: node for node in nodes}
    assert "maybe_real_cast" in by_id
    assert by_id["consumer"]["incomingEdges"][0]["sourceNodeId"] == "maybe_real_cast"


def test_prune_noop_cast_resolves_chained_removals():
    """Two consecutive no-op casts must both be removed, with the final
    consumer wired all the way back to the real producer."""
    nodes = [
        _real_node("producer", dtype="bfloat16"),
        _cast_node("noop_1", source="producer", dtype="bfloat16"),
        _cast_node("noop_2", source="noop_1", dtype="bfloat16"),
        {
            "id": "consumer",
            "incomingEdges": [_edge("noop_2")],
        },
    ]

    merge._prune_noop_cast_nodes(nodes)

    by_id = {node["id"]: node for node in nodes}
    assert "noop_1" not in by_id
    assert "noop_2" not in by_id
    assert by_id["consumer"]["incomingEdges"][0]["sourceNodeId"] == "producer"


def test_shape_fill_and_boundary_multiple_crossings():
    context = ShapeContext({"H": 16, "V": 101}, "float16")
    nodes = [
        {"id": "@input", "label": "input_ids"},
        {
            "id": "embed_tokens",
            "label": "Embedding",
            "incomingEdges": [_edge("@input")],
        },
        {
            "id": "group/a",
            "namespace": "outer/inner",
            "incomingEdges": [_edge("embed_tokens")],
        },
        {
            "id": "group/b",
            "namespace": "outer/inner",
            "incomingEdges": [_edge("group/a")],
        },
        {"id": "outside", "incomingEdges": [_edge("group/b")]},
        {"id": "output", "label": "anything"},
    ]
    shapes.fill_missing_node_shapes(nodes, context=context)
    result = shapes.group_boundary_shapes(nodes)
    assert result["outer"]["input_shape"] == "[B, S, 16] float16"
    assert result["outer"]["output_shape"] == "[B, S, 16] float16"
    assert result["outer/inner"] == result["outer"]
    assert shapes._namespace_chain("/outer//inner/") == ["outer", "outer/inner"]
    store: dict[str, list[str]] = {}
    shapes._record_shape(store, "g", "x")
    shapes._record_shape(store, "g", "x")
    assert store == {"g": ["x"]}


def test_connected_boundary_named_logits_inherits_producer_over_head_shape():
    """A connected ``@input`` tile that merely shares the name ``logits`` with the
    model head must inherit its producer's real shape, not the ``(B, S, vocab)``
    head guess. A genuine leaf ``logits`` (no producer) still gets the head shape.
    """
    context = ShapeContext({"H": 16, "V": 128}, "bfloat16")
    nodes = [
        {
            "id": "gate/producer",
            "label": "Linear",
            "outputsMetadata": [
                {
                    "id": "0",
                    "attrs": [
                        {"key": "shape", "value": "[BS, 256] bfloat16"},
                        {"key": "dtype", "value": "bfloat16"},
                    ],
                }
            ],
        },
        {
            # A router's local ``logits`` variable surfaced as a module @input
            # boundary -- it has a real producer edge and must inherit it.
            "id": "gate/score_fn/@input",
            "label": "logits",
            "attrs": [{"key": "synthetic", "value": "@input"}],
            "incomingEdges": [_edge("gate/producer")],
        },
        {
            # The model head's own logits leaf: no producer, keeps (B, S, V).
            "id": "lm_head_logits",
            "label": "logits",
        },
    ]
    shapes.fill_missing_node_shapes(nodes, context=context)
    by_id = {node["id"]: node for node in nodes}
    boundary_spec = shapes._node_spec(by_id["gate/score_fn/@input"])
    assert boundary_spec is not None
    assert list(boundary_spec.shape) == ["BS", "256"]
    head_spec = shapes._node_spec(by_id["lm_head_logits"])
    assert head_spec is not None
    assert list(head_spec.shape) == ["B", "S", "128"]


def test_group_output_boundary_shape_keeps_dtype():
    """An expandable module's ``output_shape`` layer attribute carries dtype.

    ``annotate_nodes_with_shapes`` rebuilds an ``@output`` boundary's per-port
    metadata from the producer feeding each port. That ``shape`` field must stay
    dtype-qualified (like every other shape-application path) because
    ``group_boundary_shapes`` reads it for the collapsed module's
    ``output_shape`` -- otherwise a collapsed module shows shape without type.
    """
    nodes = [
        {"id": "mod/producer", "label": "RMSNorm"},
        {
            "id": "mod/@output",
            "label": "Output",
            "namespace": "decoder/mod",
            "attrs": [{"key": "synthetic", "value": "@output"}],
            "incomingEdges": [
                {"sourceNodeId": "mod/producer", "targetNodeInputId": "0"}
            ],
            "outputsMetadata": [
                {"id": "0", "attrs": [{"key": "port_label", "value": "hidden_states"}]}
            ],
        },
    ]
    shapes.annotate_nodes_with_shapes(
        nodes,
        {
            "mod/producer": TensorSpec(("B", "S", "H"), "bfloat16"),
            "mod/@output": TensorSpec(("B", "S", "H"), "bfloat16"),
        },
        id_prefix="",
    )
    port = nodes[1]["outputsMetadata"][0]
    port_attrs = {attr["key"]: attr["value"] for attr in port["attrs"]}
    assert port_attrs["shape"] == "[B, S, H] bfloat16"
    assert port_attrs["dtype"] == "bfloat16"

    boundary = shapes.group_boundary_shapes(nodes)
    assert boundary["decoder/mod"]["output_shape"] == "[B, S, H] bfloat16"


def test_fill_missing_node_shapes_cast_resolves_downcast_dtype():
    # A residual-mix `.to(dtype)` cast fed a float32 HyperConnection output must
    # render as a genuine float32 -> float16 downcast, not a float32 no-op.
    context = ShapeContext({"H": 16}, "float16")
    nodes = [
        {
            "id": "comb",
            "label": "Linear",
            "outputsMetadata": [
                {
                    "id": "0",
                    "attrs": [
                        {"key": "shape", "value": "B x S x 4 float32"},
                        {"key": "tensor_shape", "value": "BxSx4 float32"},
                        {"key": "dtype", "value": "float32"},
                    ],
                }
            ],
        },
        {
            "id": "@op_cast",
            "label": "Cast",
            "attrs": [{"key": "detail", "value": "dtype: dtype"}],
            "incomingEdges": [_edge("comb")],
        },
    ]
    shapes.fill_missing_node_shapes(nodes, context=context)
    cast = next(node for node in nodes if node["id"] == "@op_cast")
    shape_attr = next(
        attr
        for meta in cast["outputsMetadata"]
        for attr in meta["attrs"]
        if attr["key"] == "shape"
    )
    assert shape_attr["value"] == "[B, S, 4] float16"


def test_fallback_node_spec_cast_without_detail_keeps_source_dtype():
    source = TensorSpec(("B", "S", 4), "float32")
    node = {"id": "@op_cast", "label": "Cast"}
    result = shapes._fallback_node_spec(
        node, [("0", source)], working_dtype="float16"
    )
    assert result.dtype == "float32"
    assert result.shape == ("B", "S", 4)


def test_fallback_node_spec_conv_reduces_spatial_from_geometry():
    # Vision patch-merger downsample reaches the fallback (never keyed by the
    # block-tree inference); geometry from the constructor collapses 2x2 -> 1x1.
    source = TensorSpec(("B*S/4", 4096, 2, 2), "bfloat16")
    node = {
        "id": "visual/seq:6:downsample:downsample:0",
        "label": "Conv2d",
        "attrs": [{"key": "attr_name", "value": "downsample"}],
    }
    result = shapes._fallback_node_spec(
        node,
        [("0", source)],
        conv_geometry={"downsample": ((2, 2), (2, 2), (0, 0))},
    )
    assert result.shape == ("B*S/4", 4096, 1, 1)


def test_fallback_node_spec_conv_without_geometry_passes_through():
    source = TensorSpec(("B*S/4", 4096, 2, 2), "bfloat16")
    node = {
        "id": "visual/seq:6:downsample:downsample:0",
        "label": "Conv2d",
        "attrs": [{"key": "attr_name", "value": "downsample"}],
    }
    # No geometry recorded -> spatial axes pass through unchanged.
    assert shapes._fallback_node_spec(node, [("0", source)]).shape == (
        "B*S/4",
        4096,
        2,
        2,
    )


def test_fallback_node_spec_unsqueeze_non_integer_dim_defaults_to_zero():
    source = TensorSpec(("B", "S", 4), "float16")
    node = {"id": "u", "label": "Unsqueeze", "attrs": [{"key": "detail", "value": "dim: -1"}]}
    # A negative dim resolves against rank; a non-integer would default to 0.
    assert shapes._fallback_node_spec(node, [("0", source)]).shape == ("B", "S", 4, 1)
    bad = {"id": "u", "label": "Unsqueeze", "attrs": [{"key": "detail", "value": "dim: n"}]}
    assert shapes._fallback_node_spec(bad, [("0", source)]).shape == (1, "B", "S", 4)


def test_shape_context_config_aliases_and_serialization():
    spec = _spec(
        head_dim=None,
        raw_config={
            "torch_dtype": "torch.bfloat16",
            "integer_float": 4.0,
            "ignored_float": 1.5,
            "flag": True,
            "nested_config": {"width": "12", "skip": False},
        },
    )
    context = ShapeContext.from_spec(spec)
    assert context.dtype == "bfloat16"
    assert context.dims["D"] == 4
    assert context.dims["integer_float"] == 4
    assert context.dims["nested_width"] == 12
    assert "flag" not in context.dims
    assert TensorSpec((1, "H"), "float32").to_dict() == {
        "shape": [1, "H"],
        "dtype": "float32",
    }
    record = OperatorRecord(
        "n", "c", "o", ["x"], TensorSpec((1,)), class_name="C", node_id="id"
    )
    assert record.to_dict()["class_name"] == "C"
    assert _config_dtype({"torch_dtype": 7}) == "float16"


def test_shape_module_registry_lookup_and_ambiguity():
    registry = ModuleDimRegistry()
    first = ModuleParameterSpec((2, 3))
    second = ModuleParameterSpec((4, 5))
    registry.parameter[("A", "weight")] = first
    registry.parameter_by_attr["weight"] = second
    registry.ambiguous_parameters.add("weight")
    assert registry.lookup_parameter("weight", "A") == first
    assert registry.lookup_parameter("weight", "B") is None
    assert registry.lookup_parameter("", "A") is None


@pytest.mark.parametrize(
    ("label", "inputs", "details", "expected_shape", "expected_dtype"),
    [
        (
            "view",
            [TensorSpec(("B", "S", 16))],
            ["shape: (-1, 16)"],
            ("B*S", 16),
            "float16",
        ),
        ("reshape", [TensorSpec(("B", "S", 16))], [], ("B", "S", 16), "float16"),
        ("unsqueeze", [TensorSpec((8,), "float32")], [], (1, 8), "float32"),
        ("cast", [TensorSpec((8,), "float16")], ["dtype: float32"], (8,), "float32"),
        ("topk", [TensorSpec(("B", "S", 8))], [], ("B", "S", 2), "int64"),
        (
            "gather",
            [TensorSpec(("B", "S", 8)), TensorSpec(("B", "S", 2), "int64")],
            [],
            ("B", "S", 2),
            "float16",
        ),
        # torch reductions default to keepdim=False -- the reduced axis is
        # dropped, not collapsed to size 1, unless keepdim=True is explicit.
        ("sum", [TensorSpec(("B", "S", 8))], ["dim: -1"], ("B", "S"), "float16"),
        (
            "sum",
            [TensorSpec(("B", "S", 8))],
            ["dim: -1", "keepdim: True"],
            ("B", "S", 1),
            "float16",
        ),
        ("sum", [TensorSpec(("B", "S", 8))], ["dim: 2"], ("B", "S", 8), "float16"),
        (
            "multiply",
            [TensorSpec((8,)), TensorSpec(("B", "S", 16))],
            [],
            ("B", "S", 16),
            "float16",
        ),
    ],
)
def test_shape_operation_specific_inference(
    label: str,
    inputs: list[TensorSpec],
    details: list[str],
    expected_shape: tuple[object, ...],
    expected_dtype: str,
):
    inferencer = ShapeInferencer(
        _spec(), context=ShapeContext({"H": 16, "V": 101, "TopK": 2, "E": 8})
    )
    result = inferencer._infer_node_output(
        _model_node(label, details=details), inputs, root=None
    )
    assert result.shape == expected_shape
    assert result.dtype == expected_dtype


def test_shape_special_nodes_linear_router_and_fallbacks():
    context = ShapeContext({"H": 16, "V": 101, "I": 32, "E": 8, "TopK": 2})
    registry = ModuleDimRegistry(
        linear_by_attr={"proj": ModuleLinearSpec(16, 7)},
        parameter_by_attr={"weight": ModuleParameterSpec((8, 5, 16))},
    )
    inferencer = ShapeInferencer(_spec(), context=context, module_dims=registry)

    assert inferencer._infer_node_output(
        _model_node("input", synthetic="@input"), [], root=None
    ) == TensorSpec(("B", "S"), "int64")
    assert inferencer._infer_node_output(
        _model_node("weight", synthetic="@tensor"), [], root=None
    ) == TensorSpec((8, 5, 16), "float16")
    heuristic_inferencer = ShapeInferencer(
        _spec(), context=context, module_dims=ModuleDimRegistry()
    )
    assert heuristic_inferencer._infer_node_output(
        _model_node("weight", synthetic="@tensor"), [], root=None
    ) == TensorSpec((8, 16), "float16")
    assert inferencer._infer_node_output(
        _model_node("bias", synthetic="@tensor"), [], root=None
    ) == TensorSpec((8,), "float16")
    assert inferencer._infer_node_output(
        _model_node("other", synthetic="@tensor"), [], root=None
    ) == TensorSpec((), "float16")
    assert inferencer._infer_node_output(
        _model_node(
            "Linear",
            operation=OperationKind.NN_MODULE,
            class_name="Linear",
            attr_name="proj",
        ),
        [TensorSpec(("B", "S", 16))],
        root=None,
    ).shape == ("B", "S", 7)
    assert inferencer._infer_node_output(
        _model_node("RouterBlock", operation=OperationKind.UNKNOWN),
        [TensorSpec(("B", "S", 16))],
        root=None,
    ).shape == ("B", "S", 8)
    assert inferencer._infer_node_output(
        _model_node("Kernel", operation=OperationKind.GPU_KERNEL), [], root=None
    ).shape == ("B", "S", 16)
    assert inferencer._infer_node_output(
        _model_node("Unknown", operation=OperationKind.UNKNOWN), [], root=None
    ).shape == ("B", "S", 16)


def test_fx_op_fallback_join_and_occurrence_counting():
    """The per-module FX fallback keys ops by (source line, op, occurrence-on-
    line-within-block) and only fires when a checkpoint was supplied — so a
    genuinely unknown op resolves to its FX-captured ground-truth shape while
    the common path (no checkpoint) stays a no-op."""
    inferencer = ShapeInferencer(
        _spec(), context=ShapeContext({"H": 16}), module_dims=ModuleDimRegistry()
    )

    # Occurrence indexing: two `diff`s on the same source line in the same block
    # instance get 0 and 1; a `diff` in a different block restarts at 0.
    graph = ModelGraph(
        title="t",
        nodes=[
            _model_node("Diff", node_id="blk:0:@op_l500_c4_diff:0"),
            _model_node("Diff", node_id="blk:0:@op_l500_c9_diff:1"),
            _model_node("Diff", node_id="blk:1:@op_l500_c4_diff:0"),
        ],
    )
    inferencer._register_op_line_occurrences(graph)
    keys = [inferencer._op_line_key(node) for node in graph.nodes]
    assert keys == [(500, "diff", 0), (500, "diff", 1), (500, "diff", 0)]

    # No checkpoint retained → fallback is a strict no-op (never traces).
    unknown = _model_node("Diff", node_id="blk:0:@op_l500_c4_diff:0")
    assert inferencer._fx_op_shape(unknown, [TensorSpec(("B", "S", 16))]) is None

    # With a checkpoint and a pre-populated map, an unknown op (`diff` has no
    # symbolic rule) resolves to the FX-captured shape (dtype inherited from the
    # input) instead of passing the input shape through unchanged.
    inferencer._meta_checkpoint = "dummy"
    inferencer._op_fx_shapes = {(500, "diff", 0): ("B", "S", 15)}
    result = inferencer._infer_node_output(
        unknown, [TensorSpec(("B", "S", 16), "float16")], root=None
    )
    assert result == TensorSpec(("B", "S", 15), "float16")


def test_shape_elementwise_forward_input_prefers_wider_activation():
    inferencer = ShapeInferencer(
        _spec(), context=ShapeContext({"H": 16}), module_dims=ModuleDimRegistry()
    )
    narrow = TensorSpec(("B", "S", 2))
    activation = TensorSpec(("B", "S", 16))
    inferencer._forward_input_specs.add(id(activation))
    assert inferencer._elementwise_operand([narrow, activation]) is activation
    assert inferencer._elementwise_operand([activation, narrow]) is activation


def test_shape_graph_order_cycle_and_boundary_signatures():
    graph = ModelGraph(
        "cycle",
        nodes=[
            _model_node("A", node_id="a"),
            _model_node("B", node_id="b"),
            _model_node("Ignored", node_id="ignored"),
        ],
        edges=[
            GraphEdge("a", "b"),
            GraphEdge("b", "a"),
            GraphEdge("missing", "a"),
        ],
    )
    assert _topological_order(graph) == ["ignored", "a", "b"]

    compute = OperatorRecord("op", "Linear", "nn_module", [], TensorSpec((1, 4)))
    output = OperatorRecord("out", "output", "output", [], TensorSpec((1, 8)))
    assert subgraph_boundary_signature([]) is None
    assert subgraph_boundary_signature([compute]) == (
        "Linear",
        (1, 4),
        "float16",
        (1, 4),
        "float16",
        "no_input",
    )
    input_op = OperatorRecord(
        "input", "input", "input", [], TensorSpec((1, 2), "int64")
    )
    assert subgraph_boundary_signature([input_op, compute, output], class_name="C") == (
        "C",
        (1, 2),
        "int64",
        (1, 4),
        "float16",
    )


def test_shape_ast_dimension_and_constructor_helpers():
    context = ShapeContext({"H": 16, "alias": 9})
    config = {"width": 8, "nested": {"depth": 3}}
    resolve = lambda expression: _resolve_dim_expr(  # noqa: E731
        ast.parse(expression, mode="eval").body,
        config=config,
        local_vars={"local": 4},
        context=context,
    )
    assert resolve("2 + 3") == 5
    assert resolve("8 / 2") == 4
    assert resolve("8 / 0") is None
    assert resolve("H * 2") == 32
    assert resolve("(H + 1) * 2") == 34
    assert resolve("config.width") == 8
    assert resolve("config.nested['depth']") == 3
    assert resolve("int(local)") == 4
    assert resolve("getattr(config, 'alias')") == 9
    assert resolve("-local") == -4
    assert resolve("unknown") is None

    linear = _parse_module_ctor(
        ast.parse("Linear(in_features=H, out_features=32)", mode="eval").body,
        config=config,
        local_vars={},
        context=context,
    )
    assert linear == ModuleLinearSpec(16, 32)
    parameter = _parse_module_ctor(
        ast.parse("Parameter(torch.zeros((2, H)))", mode="eval").body,
        config=config,
        local_vars={},
        context=context,
    )
    assert parameter == ModuleParameterSpec((2, 16))
    assert _parse_tensor_ctor_shape(
        ast.parse("torch.full((2, H), 1.0)", mode="eval").body,
        config=config,
        local_vars={},
        context=context,
    ) == (2, 16)
    symbolic_context = ShapeContext({"H": "H"})
    assert (
        _resolve_dim_expr(
            ast.parse("(H + 1) * 2", mode="eval").body,
            config={},
            local_vars={},
            context=symbolic_context,
        )
        == "(H+1)*2"
    )


def test_shape_misc_helper_error_and_fallback_paths():
    assert _dedupe_preserve(["a", "b", "a"]) == ["a", "b"]
    assert _replace_last_dim((), 4) == ("B", "S", 4)
    assert _replace_last_dim((1, 2), 3) == (1, 3)
    assert _int_dim(True) is None
    assert _int_dim(2.0) == 2
    assert _int_dim("12") == 12
    assert _int_dim("x") is None
    assert _detail_value(["other: 1", " dim: -1 "], "dim") == "-1"
    assert _detail_value([], "dim") is None
    assert _symbolic_binop("H", 2, ast.Pow()) is None
    assert _dim_term("H+1", "*") == "(H+1)"
    assert _dim_term("H+1", "+") == "H+1"
    assert _broadcast_rank(TensorSpec(())) == (0, 1.0)
    assert _broadcast_rank(TensorSpec(("B", "H")))[1] == float("inf")
    aliases = dict(
        _nested_dim_aliases("linear_attn_config", {"head_dim": 4, "x": False})
    )
    assert aliases["linear_head_dim"] == 4
    assert "x" not in aliases
    assert _heuristic_linear_out_features("lm_head", ShapeContext({"V": 101})) == 101
    assert _heuristic_linear_out_features("gate_proj", ShapeContext({"I": 32})) == 32
    assert _heuristic_linear_out_features("router", ShapeContext({"E": 8})) == 8
    assert _heuristic_linear_out_features("custom_proj", ShapeContext({"H": 16})) == 16
    assert _heuristic_linear_out_features(None, ShapeContext()) is None


def test_merge_nested_namespace_segment_variants():
    pipeline = BlockNode("@pipeline", "KernelPipeline", "other", "Pipeline")
    synthetic = BlockNode("@generated", "Generated", "other", "Generated")
    regular = BlockNode("child.attr", "Child", "other", "Child")

    assert (
        merge._nested_namespace_segment(pipeline, "Chunk pipeline") == "Chunk_pipeline"
    )
    assert merge._nested_namespace_segment(synthetic, "Friendly title") == (
        "Friendly_title"
    )
    assert merge._nested_namespace_segment(synthetic, "@generated") == "generated"
    assert merge._nested_namespace_segment(regular, "Ignored") == "child.attr"
    assert (
        merge._kernel_pipeline_step(
            BlockNode("root", "Root", "other", "Root", children=[regular, pipeline])
        )
        is pipeline
    )
    assert merge._kernel_pipeline_step(regular) is None
    assert merge._is_tensor_port({"attrs": _attrs(synthetic="@tensor")})
    assert not merge._is_tensor_port({})


def test_merge_section_tree_resolution_variant_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
):
    basic = BasicOpFilter.for_detailed()
    attention_tree = BlockNode(
        "attention", "VariantAttention", "attention", "Attention"
    )
    ffn_tree = BlockNode("experts", "VariantFFN", "moe", "Experts")
    fallback_tree = BlockNode("experts", "FallbackFFN", "moe", "Fallback")
    trees = [
        ("Attention detail", attention_tree),
        ("FFN detail", ffn_tree),
        ("Fallback detail", fallback_tree),
    ]
    monkeypatch.setattr(merge, "architecture_section_trees", lambda spec: trees)
    monkeypatch.setattr(merge, "subgraph_warrants_json_export", lambda *a, **k: True)
    spec = _spec()

    assert merge._resolve_section_tree_by_class(spec, None, basic_ops=basic) is None
    assert (
        merge._resolve_section_tree_by_class(spec, "VariantAttention", basic_ops=basic)
        == trees[0]
    )
    assert (
        merge._resolve_section_tree_by_class(spec, "Missing", basic_ops=basic) is None
    )

    attention = _component("self_attn", "attention", class_name="Base")
    ffn = _component("experts", "moe", class_name="Base", label="Fallback")
    variant = LayerVariant(
        "variant",
        2,
        "Attention",
        "VariantAttention",
        "FFN",
        "VariantFFN",
        "experts",
    )
    assert (
        merge._resolve_section_tree_for_component(
            spec, attention, variant=variant, basic_ops=basic
        )
        == trees[0]
    )
    assert (
        merge._resolve_section_tree_for_component(
            spec, ffn, variant=variant, basic_ops=basic
        )
        == trees[1]
    )

    monkeypatch.setattr(
        merge,
        "_resolve_section_tree_by_class",
        lambda spec, class_name, basic_ops: None,
    )
    variant.ffn_label = "Fallback"
    assert (
        merge._resolve_section_tree_for_component(
            spec, ffn, variant=variant, basic_ops=basic
        )
        == trees[2]
    )


def test_merge_resolve_section_tree_disambiguates_and_uses_largest(
    monkeypatch: pytest.MonkeyPatch,
):
    small = BlockNode(
        "shared",
        "Small",
        "other",
        "Small",
        children=[BlockNode("a", "A", "other", "A")],
    )
    large = BlockNode(
        "shared",
        "Large",
        "other",
        "Large",
        children=[
            BlockNode("a", "A", "other", "A"),
            BlockNode("b", "B", "other", "B"),
        ],
    )
    trees = [("Small section", small), ("Large section", large)]
    monkeypatch.setattr(merge, "architecture_section_trees", lambda spec: trees)
    monkeypatch.setattr(merge, "subgraph_warrants_json_export", lambda *a, **k: True)
    basic = BasicOpFilter.for_detailed()

    assert (
        merge._resolve_section_tree(
            _spec(), "shared", component_label="Small", basic_ops=basic
        )
        == trees[0]
    )
    assert (
        merge._resolve_section_tree(
            _spec(), "shared", component_label="Unknown", basic_ops=basic
        )
        == trees[1]
    )
    assert (
        merge._resolve_section_tree(
            _spec(), "missing", component_label="", basic_ops=basic
        )
        is None
    )


def test_merge_append_section_unresolved_detail_summary(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(merge, "component_has_detail_section", lambda *a: True)
    monkeypatch.setattr(
        merge, "_resolve_section_tree_for_component", lambda *a, **k: None
    )
    nodes: list[dict[str, object]] = []
    exits = merge._append_section(
        nodes,
        spec=_spec(),
        component=_component("attn", "attention", label="Attention"),
        id_prefix="decoder/attn",
        namespace_prefix="decoder/Attention",
        basic_ops=BasicOpFilter.for_detailed(),
        previous_exits=["previous"],
        variant=LayerVariant("v", 1, "A"),
    )

    assert exits == ["decoder/attn"]
    assert nodes[0]["namespace"] == "decoder/Attention"
    assert nodes[0]["incomingEdges"] == [_edge("previous")]


def test_merge_append_section_expands_nested_diagram_and_shapes(
    monkeypatch: pytest.MonkeyPatch,
):
    nested = BlockNode("nested", "NestedBlock", "other", "Nested")
    root = BlockNode("attn", "Attention", "attention", "Attention", children=[nested])
    parent_computation = ComputationGraph(
        nodes=[
            GraphNodeSpec("@input", label="input", synthetic="@input"),
            GraphNodeSpec("tile", block=nested, label="Nested"),
            GraphNodeSpec("after", label="After"),
        ],
        links=[(0, 1), (1, 2)],
        primary_output_index=2,
    )
    nested_computation = ComputationGraph(
        nodes=[
            GraphNodeSpec("@input", label="input", synthetic="@input"),
            GraphNodeSpec("result", label="Result"),
        ],
        links=[(0, 1)],
        primary_output_index=1,
    )
    monkeypatch.setattr(merge, "component_has_detail_section", lambda *a: True)
    monkeypatch.setattr(
        merge,
        "_resolve_section_tree_for_component",
        lambda *a, **k: ("Attention", root),
    )
    monkeypatch.setattr(merge, "expand_block_tree_inplace", lambda tree, **kwargs: tree)
    monkeypatch.setattr(
        merge,
        "build_computation_graph",
        lambda tree, **kwargs: (
            parent_computation if tree is root else nested_computation
        ),
    )
    monkeypatch.setattr(
        merge, "collect_nested_diagrams", lambda *a, **k: [("Nested detail", nested)]
    )
    monkeypatch.setattr(
        merge,
        "infer_block_tree_shapes",
        lambda inferencer, tree, title: {
            "@input": TensorSpec(("B", "S", 16)),
            "result" if tree is nested else "after": TensorSpec(("B", "S", 16)),
        },
    )
    nodes: list[dict[str, object]] = [{"id": "duplicate"}]
    group_attrs: dict[str, dict[str, str]] = {}
    exits = merge._append_section(
        nodes,
        spec=_spec(),
        component=_component(
            "attn", "attention", class_name="Attention", label="Attention"
        ),
        id_prefix="decoder/attn",
        namespace_prefix="decoder/Attention",
        basic_ops=BasicOpFilter.for_detailed(),
        previous_exits=["previous"],
        group_node_attributes=group_attrs,
        shape_inferencer=object(),
    )

    by_id = {node["id"]: node for node in nodes}
    assert "decoder/attn/tile" not in by_id
    assert "decoder/attn/tile/result" in by_id
    nested_output = by_id["decoder/attn/tile/@output"]
    assert nested_output["incomingEdges"][0]["sourceNodeId"] == (
        "decoder/attn/tile/result"
    )
    assert [item["id"] for item in nested_output["outputsMetadata"]] == ["result"]
    assert by_id["decoder/attn/after"]["incomingEdges"][0]["sourceNodeId"] == (
        "decoder/attn/tile/@output"
    )
    assert (
        by_id["decoder/attn/after"]["incomingEdges"][0]["sourceNodeOutputId"]
        == "result"
    )
    assert exits == ["decoder/attn/after"]
    assert group_attrs["decoder/Attention"] == {
        "label": "Attention",
        "operation": "Attention",
    }
    assert by_id["decoder/attn/tile/result"]["outputsMetadata"]


def test_merge_build_graph_expanded_tail_and_shape_boundaries(
    monkeypatch: pytest.MonkeyPatch,
):
    tail = _component("tail", "ffn", class_name="Tail", label="Tail")
    spec = _spec(stack_pre=[], stack_tail=[tail], block_components=[])
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(merge, "_stack_pre_components", lambda spec: [])
    monkeypatch.setattr(merge, "_stack_tail_components", lambda spec: [tail])
    monkeypatch.setattr(merge, "component_has_detail_section", lambda *a: True)
    monkeypatch.setattr(
        merge,
        "_append_decoder_layers",
        lambda nodes, **kwargs: list(kwargs["previous_exits"]),
    )

    def fake_append(nodes, **kwargs):
        calls.append((kwargs["id_prefix"], kwargs["namespace_prefix"]))
        nodes.append(
            {
                "id": kwargs["id_prefix"],
                "label": "Tail",
                "namespace": kwargs["namespace_prefix"],
            }
        )
        return [kwargs["id_prefix"]]

    monkeypatch.setattr(merge, "_append_section", fake_append)
    monkeypatch.setattr(
        merge,
        "fill_missing_node_shapes",
        lambda nodes, context, boundary_spec=None: None,
    )
    monkeypatch.setattr(
        merge,
        "group_boundary_shapes",
        lambda nodes: {"tail": {"input_shape": "B x S x 16"}},
    )
    inferencer = SimpleNamespace(
        context=ShapeContext({"H": 16}),
        boundary_input_spec=lambda *args, **kwargs: None,
    )
    graph = merge.build_merged_model_graph(spec, shape_inferencer=inferencer)

    assert calls == [("tail", "tail")]
    assert graph["groupNodeAttributes"]["tail"]["input_shape"] == "B x S x 16"
    assert any(
        "tail" in config["namespaceRegex"] for config in graph["groupNodeConfigs"]
    )


def test_overview_forward_operation_labels(monkeypatch: pytest.MonkeyPatch):
    operations = {
        "scores": SimpleNamespace(
            label="MatMul", external_inputs=["query", "key_states"]
        ),
        "activation": SimpleNamespace(label="SiLU", class_name=None),
    }
    decoder = SimpleNamespace(forward_operations=operations)
    spec = _spec(
        decoder_class="Decoder",
        class_registry={"Decoder": decoder},
        forward_sequence=["scores", "activation"],
    )
    monkeypatch.setattr(
        "TraceLens.ModelUtils.ast_analyze.classify_matmul_label",
        lambda external_inputs: f"matmul({','.join(external_inputs)})",
    )
    monkeypatch.setattr(
        "TraceLens.ModelUtils.ast_analyze.operation_display_label",
        lambda label, class_name: f"display:{label}",
    )

    assert overview.forward_sequence_display_labels(spec) == [
        "matmul(query,key_states)",
        "display:SiLU",
    ]
    assert overview.forward_sequence_display_labels(_spec(forward_sequence=[])) == []


def test_overview_detail_tree_filtering_and_component_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    direct = BlockNode("direct", "Direct", "attention", "Direct")
    straight_ffn = BlockNode("ffn", "VariantFFN", "ffn", "FFN")
    nested_omit = BlockNode("omit", "Nested", "other", "Omit")
    nested_hide = BlockNode("hide", "Nested", "other", "Hide")
    nested_keep = BlockNode("keep", "Nested", "other", "Keep")
    trees = [("Direct", direct), ("FFN", straight_ffn)]
    spec = _spec(
        export_block_trees=trees,
        layer_variants=[LayerVariant("v", 1, "A", ffn_class="VariantFFN")],
    )
    monkeypatch.setattr(overview, "architecture_section_trees", lambda spec: trees)
    monkeypatch.setattr(
        overview, "prepare_diagram_section_trees", lambda *a, **k: trees
    )
    monkeypatch.setattr(
        overview,
        "collect_nested_diagrams",
        lambda tree, **k: (
            [
                ("Omit", nested_omit),
                ("Hide", nested_hide),
                ("Keep", nested_keep),
            ]
            if tree is direct
            else []
        ),
    )
    monkeypatch.setattr(overview, "expand_block_tree_inplace", lambda tree, **k: tree)
    monkeypatch.setattr(
        overview, "is_straight_line_module", lambda tree: tree is straight_ffn
    )
    monkeypatch.setattr(overview, "subgraph_warrants_export", lambda *a, **k: True)
    monkeypatch.setattr(
        overview,
        "is_single_function_tree",
        lambda tree: tree is nested_omit or tree is nested_hide,
    )
    monkeypatch.setattr(
        overview, "_omit_from_detailed_view", lambda tree: tree is nested_omit
    )
    monkeypatch.setattr(
        overview,
        "_show_single_function_in_diagram",
        lambda tree: tree is not nested_hide,
    )

    assert [tree.attr_name for tree in overview._detail_section_trees(spec)] == [
        "direct",
        "keep",
        "ffn",
    ]
    assert overview.component_has_detail_section(
        _component("direct", "attention"), spec
    )

    monkeypatch.setattr(overview, "_detail_section_trees", lambda spec: [])
    assert overview.component_has_detail_section(_component("ffn", "ffn"), spec)
    monkeypatch.setattr(overview, "subgraph_warrants_export", lambda *a, **k: False)
    assert not overview.component_has_detail_section(_component("ffn", "ffn"), spec)


def test_shape_external_spec_and_empty_combine_fallbacks():
    context = ShapeContext({"H": 16, "E": 8})
    registry = ModuleDimRegistry(
        parameter_by_attr={"matrix": ModuleParameterSpec((3, 4))}
    )
    inferencer = ShapeInferencer(_spec(), context=context, module_dims=registry)

    combine = inferencer._infer_node_output(
        _model_node("Combine", synthetic="@combine"), [], root=None
    )
    assert combine == TensorSpec(("B", "S", 16))
    parameter_view = inferencer._infer_node_output(
        _model_node("view", external_inputs=["self.matrix"]), [], root=None
    )
    assert parameter_view == TensorSpec((3, 4))
    weight_view = inferencer._infer_node_output(
        _model_node("flatten", external_inputs=["expert_weight"]), [], root=None
    )
    assert weight_view == TensorSpec((8, 16), "float16")
    bias_unsqueeze = inferencer._infer_node_output(
        _model_node("unsqueeze", external_inputs=["router_bias"]), [], root=None
    )
    assert bias_unsqueeze == TensorSpec((1, 8))
    default_view = inferencer._infer_node_output(_model_node("reshape"), [], root=None)
    assert default_view == TensorSpec(("B", "S", 16))


def test_shape_build_and_save_operator_export_fallbacks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    payload = {"name": "synthetic", "sections": []}
    monkeypatch.setattr(
        ShapeInferencer, "export_architecture", lambda self, **kwargs: payload
    )
    assert build_operator_export(_spec(), include_model_output=False) is payload

    target = save_operator_export(payload, tmp_path / "nested" / "operators.json")
    assert target.read_text(encoding="utf-8") == (
        '{\n  "name": "synthetic",\n  "sections": []\n}\n'
    )


# ---------------------------------------------------------------------------
# Vision tower surfacing (multimodal wrapper) — torch-free
# ---------------------------------------------------------------------------

from TraceLens.ModelUtils.ast_analyze import ClassStructure
from TraceLens.ModelUtils.extract import find_vision_tower, vision_tower_component


def _class_structure(name: str, init_assignments: dict[str, str] | None = None):
    """Minimal ClassStructure for registry-based vision-tower detection tests."""
    return ClassStructure(
        name=name,
        node=ast.parse(f"class {name}:\n    pass").body[0],
        init_assignments=init_assignments or {},
        init_details={},
        forward_calls=[],
        norm_before=[],
    )


def _config_vlm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_type: str = "foo_vision",
    config_class: str = "FooVisionConfig",
    tower_class: str = "FooVisionModel",
    attr: str = "visual",
) -> ArchitectureSpec:
    """A VLM spec detected via a nested ``vision_config`` block.

    Registers a synthetic ``model_type -> config class`` entry in transformers'
    mapping (version-independent) and returns a spec whose registry holds the
    derived tower class plus a wrapper binding it to ``attr``.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    monkeypatch.setitem(CONFIG_MAPPING_NAMES, model_type, config_class)
    registry = {
        "Wrapper": _class_structure(
            "Wrapper", {attr: "_from_config", "language_model": "_from_config"}
        ),
        "TextModel": _class_structure("TextModel"),
        tower_class: _class_structure(tower_class, {"patch_embed": "PatchEmbed"}),
    }
    return _spec(
        class_registry=registry,
        stack_model_class="TextModel",
        raw_config={"vision_config": {"model_type": model_type}},
    )


def test_find_vision_tower_resolves_from_vision_config(monkeypatch: pytest.MonkeyPatch):
    spec = _config_vlm(monkeypatch)
    # Detection is driven by the nested `vision_config`; the tower class is derived
    # from transformers' model_type->config mapping (FooVisionConfig -> FooVisionModel)
    # and confirmed against the parsed registry. The `visual` attr is recovered from
    # the wrapper's assignments separately (the config does not name it).
    assert find_vision_tower(spec) == ("visual", "FooVisionModel")

    component = vision_tower_component(spec)
    assert component is not None
    assert component.attr_name == "visual"
    assert component.class_name == "FooVisionModel"
    assert component.role == "vision"


def test_find_vision_tower_none_without_vision_config():
    # A registered `*VisionModel` class alone no longer triggers detection: without a
    # `vision_config` block the checkpoint is treated as text-only.
    registry = {
        "TextModel": _class_structure("TextModel"),
        "FooVisionModel": _class_structure("FooVisionModel", {"patch_embed": "PatchEmbed"}),
    }
    spec = _spec(class_registry=registry, stack_model_class="TextModel")
    assert find_vision_tower(spec) is None
    assert vision_tower_component(spec) is None


def test_find_vision_tower_none_when_tower_class_absent_from_registry(
    monkeypatch: pytest.MonkeyPatch,
):
    # `vision_config` present, but no modeling source was parsed for the derived
    # tower class, so there is nothing to build a detail tree from.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    monkeypatch.setitem(CONFIG_MAPPING_NAMES, "foo_vision", "FooVisionConfig")
    spec = _spec(
        class_registry={"TextModel": _class_structure("TextModel")},
        stack_model_class="TextModel",
        raw_config={"vision_config": {"model_type": "foo_vision"}},
    )
    assert find_vision_tower(spec) is None


def test_find_vision_tower_none_for_empty_registry():
    assert find_vision_tower(_spec(class_registry={})) is None


def _port(node_id, key, port_id, shape):
    return {
        "id": node_id,
        key: [
            {
                "id": port_id,
                "attrs": [
                    {"key": "shape", "value": shape},
                    {"key": "tensor_shape", "value": shape.replace(" x ", "x")},
                ],
            }
        ],
    }


def test_reconcile_edge_endpoint_shapes_fills_weak_dim_from_concrete_end():
    # A target port left as a collapsed `-1 x 4096` is rewritten to the source's
    # concrete `B x 4096` when the two ends share a rank.
    source = _port("src", "outputsMetadata", "0", "B x 4096 float16")
    target = {
        **_port("dst", "inputsMetadata", "0", "-1 x 4096 float16"),
        "incomingEdges": [
            {"sourceNodeId": "src", "sourceNodeOutputId": "0", "targetNodeInputId": "0"}
        ],
    }
    nodes = [source, target]
    merge._reconcile_edge_endpoint_shapes(nodes)
    shape = merge._port_shape_attrs(target["inputsMetadata"][0])["value"]
    assert shape == "[B, 4096] float16"


def test_reconcile_edge_endpoint_shapes_preserves_genuine_rank_change():
    # A real flatten (2D vision output → 3D combine input) differs in RANK; the
    # reconciler must leave both ends untouched rather than paper over it.
    source = _port("visual/@output", "outputsMetadata", "0", "BxS x 4096 float16")
    target = {
        **_port("@combine", "inputsMetadata", "image_embeds", "B x S x 4096 float16"),
        "incomingEdges": [
            {
                "sourceNodeId": "visual/@output",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "image_embeds",
            }
        ],
    }
    nodes = [source, target]
    merge._reconcile_edge_endpoint_shapes(nodes)
    assert (
        merge._port_shape_attrs(source["outputsMetadata"][0])["value"]
        == "BxS x 4096 float16"
    )
    assert (
        merge._port_shape_attrs(target["inputsMetadata"][0])["value"]
        == "B x S x 4096 float16"
    )


def test_attach_vision_language_combine_merges_text_and_vision():
    # The combine node is fed by (text embeddings, image embeddings) and becomes
    # the new exit the language stack consumes — the embedding tile keeps its own
    # single input rather than masquerading as the merge point.
    nodes: list[dict] = []
    exits = merge._attach_vision_language_combine(
        nodes,
        vision_exit=("visual/@output", "result"),
        text_exits=[("embed_tokens", "0")],
        shape_inferencer=None,
    )
    assert exits == [("@vision_language_combine", "0")]
    # A dedicated @image_mask boundary is synthesized as the scatter's control
    # input, then the combine node is appended after it.
    mask = nodes[0]
    assert mask["id"] == "@image_mask"
    assert mask["label"] == "image_mask"
    combine = nodes[-1]
    assert combine["id"] == "@vision_language_combine"
    assert combine["label"] == "Masked scatter"
    sources = [edge["sourceNodeId"] for edge in combine["incomingEdges"]]
    assert sources == ["embed_tokens", "@image_mask", "visual/@output"]
    ports = [meta["id"] for meta in combine["inputsMetadata"]]
    assert ports == ["inputs_embeds", "image_mask", "image_embeds"]


def test_ensure_image_mask_node_dedupes_and_stamps_bool_shape():
    nodes: list[dict] = []
    first = merge._ensure_image_mask_node(nodes, namespace="image_inputs", token_id=42)
    # Same boundary is reused, never duplicated (idempotent by id).
    second = merge._ensure_image_mask_node(nodes, namespace="")
    assert first == second == "@image_mask"
    assert len([n for n in nodes if n["id"] == "@image_mask"]) == 1
    mask = nodes[0]
    assert mask["namespace"] == "image_inputs"
    detail = next(a["value"] for a in mask["attrs"] if a["key"] == "detail")
    assert "image_token_id (42)" in detail
    # Boolean [B, S] selector shape, not the combine's [B, S, hidden].
    shape = next(a["value"] for a in mask["attrs"] if a["key"] == "output_shape")
    assert shape == "[B, S] bool"


def test_attach_vision_language_combine_applies_shape_from_context():
    # With a shape inferencer available, the combine carries the (B, S, hidden)
    # embedding shape so downstream nodes resolve their inputs.
    context = SimpleNamespace(dims={merge.Symbol.HIDDEN.value: 4096}, dtype="float16")
    shape_inferencer = SimpleNamespace(context=context)
    nodes: list[dict] = []
    merge._attach_vision_language_combine(
        nodes,
        vision_exit=("visual/@output", "result"),
        text_exits=[("embed_tokens", "0")],
        shape_inferencer=shape_inferencer,
    )
    combine = nodes[-1]
    shape_attr = next(
        attr
        for meta in combine["outputsMetadata"]
        for attr in meta["attrs"]
        if attr["key"] == "shape"
    )
    assert "4096" in shape_attr["value"]
    assert "float16" in shape_attr["value"]


def test_attach_vision_language_combine_noop_without_text_exits():
    nodes: list[dict] = []
    exits = merge._attach_vision_language_combine(
        nodes,
        vision_exit=("visual/@output", "result"),
        text_exits=[],
        shape_inferencer=None,
    )
    assert exits == []
    assert nodes == []


def _vision_merge_monkeypatch(monkeypatch, *, embed, vision):
    monkeypatch.setattr(merge, "_stack_pre_components", lambda spec: [embed])
    monkeypatch.setattr(merge, "_stack_tail_components", lambda spec: [])
    monkeypatch.setattr(
        merge, "_append_decoder_layers", lambda nodes, **kw: list(kw["previous_exits"])
    )
    monkeypatch.setattr(merge, "vision_tower_component", lambda spec: vision)


def test_merge_graph_emits_vision_group_and_visual_language_edge(
    monkeypatch: pytest.MonkeyPatch,
):
    embed = _component("embed_tokens", "embedding", class_name="Embedding", label="Embedding")
    vision = _component("visual", "vision", class_name="VisionModel", label="Vision Tower", order=0)
    spec = _spec(stack_pre=[embed], stack_tail=[], block_components=[])
    _vision_merge_monkeypatch(monkeypatch, embed=embed, vision=vision)

    monkeypatch.setattr(
        merge,
        "component_has_detail_section",
        lambda component, spec: component.attr_name == "visual",
    )
    monkeypatch.setattr(
        merge,
        "_resolve_section_tree_for_component",
        lambda *a, **k: ("Vision Tower", BlockNode("visual", "VisionModel", "vision", "Vision Tower")),
    )
    monkeypatch.setattr(merge, "is_transparent_inline_expansion", lambda tree: False)
    monkeypatch.setattr(
        merge, "expand_block_tree_inplace", lambda tree, basic_ops=None: tree
    )

    def fake_append(nodes, **kw):
        prefix, namespace = kw["id_prefix"], kw["namespace_prefix"]
        if prefix == "visual":
            nodes.append({"id": "visual/patch", "label": "PatchEmbed", "namespace": "visual"})
            nodes.append({"id": "visual/@output", "label": "result", "namespace": "visual"})
            group_attrs = kw.get("group_node_attributes")
            if group_attrs is not None:
                group_attrs["visual"] = {"label": "Vision Tower", "operation": "VisionModel"}
            return [("visual/@output", "result")]
        nodes.append(
            {
                "id": prefix,
                "label": "Embedding",
                "namespace": namespace,
                "incomingEdges": [merge._source_edge(s, "0") for s in kw["previous_exits"]],
            }
        )
        return [prefix]

    monkeypatch.setattr(merge, "_append_section", fake_append)

    graph = merge.build_merged_model_graph(spec)
    node_ids = {node["id"] for node in graph["nodes"]}

    # (a) the vision tower renders as an expandable "visual" namespace group.
    assert "@vision_input" in node_ids
    assert any(node["namespace"] == "visual" for node in graph["nodes"])
    assert graph["groupNodeAttributes"]["visual"]["label"] == "Vision Tower"

    # (b) an explicit combine node merges the text embedding output with the
    # vision output; the embedding tile itself is NOT the merge point.
    embed_node = next(node for node in graph["nodes"] if node["id"] == "embed_tokens")
    assert all(
        edge["sourceNodeId"] != "visual/@output"
        for edge in embed_node.get("incomingEdges", [])
    )
    combine = next(
        node for node in graph["nodes"] if node["id"] == "@vision_language_combine"
    )
    sources = {edge["sourceNodeId"] for edge in combine["incomingEdges"]}
    assert sources == {"embed_tokens", "@image_mask", "visual/@output"}


def _build_vision_graph_with_raw_config(monkeypatch, raw_config):
    """Build a merged vision graph under the standard vision monkeypatch."""
    embed = _component("embed_tokens", "embedding", class_name="Embedding", label="Embedding")
    vision = _component("visual", "vision", class_name="VisionModel", label="Vision Tower", order=0)
    spec = _spec(stack_pre=[embed], stack_tail=[], block_components=[], raw_config=raw_config)
    _vision_merge_monkeypatch(monkeypatch, embed=embed, vision=vision)
    monkeypatch.setattr(
        merge, "component_has_detail_section",
        lambda component, spec: component.attr_name == "visual",
    )
    monkeypatch.setattr(
        merge, "_resolve_section_tree_for_component",
        lambda *a, **k: ("Vision Tower", BlockNode("visual", "VisionModel", "vision", "Vision Tower")),
    )
    monkeypatch.setattr(merge, "is_transparent_inline_expansion", lambda tree: False)
    monkeypatch.setattr(merge, "expand_block_tree_inplace", lambda tree, basic_ops=None: tree)

    def fake_append(nodes, **kw):
        prefix = kw["id_prefix"]
        if prefix == "visual":
            nodes.append({"id": "visual/@output", "label": "result", "namespace": "visual"})
            return [("visual/@output", "result")]
        nodes.append({"id": prefix, "label": "Embedding", "namespace": kw["namespace_prefix"]})
        return [prefix]

    monkeypatch.setattr(merge, "_append_section", fake_append)
    return merge.build_merged_model_graph(spec)


def test_image_mask_colocated_with_image_patches_when_token_key_present(
    monkeypatch: pytest.MonkeyPatch,
):
    # With a general image-token config key, the image patches (@vision_input)
    # and the placeholder mask (@image_mask) render at top level (no separate box)
    # but stay adjacent in sort order, next to each other rather than floating the
    # mask beside its distant masked_scatter consumer.
    graph = _build_vision_graph_with_raw_config(monkeypatch, {"image_token_id": 154854})
    nodes = graph["nodes"]
    by_id = {node["id"]: node for node in nodes}
    assert by_id["@vision_input"]["namespace"] == ""
    assert by_id["@image_mask"]["namespace"] == ""
    assert "image_inputs" not in graph["groupNodeAttributes"]
    # Model inputs lead in forward-signature order: tokenized text (input_ids)
    # before the image patches, and the derived mask immediately after the patches.
    order = [node["id"] for node in nodes]
    assert order.index("@input") < order.index("@vision_input")
    assert order.index("@image_mask") == order.index("@vision_input") + 1
    # The mask carries the resolved token id and a boolean [B, S] selector shape.
    mask = by_id["@image_mask"]
    detail = next(a["value"] for a in mask["attrs"] if a["key"] == "detail")
    assert "154854" in detail
    shape = next(a["value"] for a in mask["attrs"] if a["key"] == "output_shape")
    assert shape == "[B, S] bool"


def test_image_mask_not_grouped_without_token_key(monkeypatch: pytest.MonkeyPatch):
    # No image-token key -> no co-location group; inputs stay at the top level.
    graph = _build_vision_graph_with_raw_config(monkeypatch, {})
    by_id = {node["id"]: node for node in graph["nodes"]}
    assert by_id["@vision_input"]["namespace"] == ""
    assert by_id["@image_mask"]["namespace"] == ""
    assert "image_inputs" not in graph["groupNodeAttributes"]


def test_merge_graph_text_only_spec_has_no_vision_section(
    monkeypatch: pytest.MonkeyPatch,
):
    # (c) a non-VLM spec is untouched: no vision input, group, or edge, and the
    # tokenized-text input is still the first node.
    embed = _component("embed_tokens", "embedding", class_name="Embedding", label="Embedding")
    spec = _spec(stack_pre=[embed], stack_tail=[], block_components=[])
    _vision_merge_monkeypatch(monkeypatch, embed=embed, vision=None)
    monkeypatch.setattr(merge, "component_has_detail_section", lambda component, spec: False)

    graph = merge.build_merged_model_graph(spec)
    node_ids = {node["id"] for node in graph["nodes"]}

    assert "@vision_input" not in node_ids
    assert not any(node["namespace"].startswith("visual") for node in graph["nodes"])
    assert "visual" not in graph["groupNodeAttributes"]
    assert graph["nodes"][0]["id"] == "@input"


def test_build_export_block_trees_appends_vision_tower_section(
    monkeypatch: pytest.MonkeyPatch,
):
    from pathlib import Path

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    from TraceLens.ModelUtils import ast_analyze as aa
    from TraceLens.ModelUtils.extract import _build_export_block_trees

    source = """
class VisionMLP:
    def __init__(self):
        self.fc1 = Linear()
        self.fc2 = Linear()

    def forward(self, x):
        return self.fc2(self.fc1(x))

class FooVisionModel:
    def __init__(self):
        self.patch_embed = PatchEmbed()
        self.mlp = VisionMLP()

    def forward(self, x):
        x = self.patch_embed(x)
        return self.mlp(x)
"""
    analysis = aa.analyze_sources({Path("m.py"): source})
    monkeypatch.setitem(CONFIG_MAPPING_NAMES, "foo_vision", "FooVisionConfig")
    spec = _spec(
        class_registry=dict(analysis.class_registry),
        stack_model_class=None,
        export_block_trees=[],
        raw_config={"vision_config": {"model_type": "foo_vision"}},
    )
    _build_export_block_trees(spec, BasicOpFilter.for_detailed())

    vision = [tree for _title, tree in spec.export_block_trees if tree.attr_name == "visual"]
    assert vision, "vision tower detail tree should be appended alongside the text spine"
    assert vision[0].class_name == "FooVisionModel"


# ---------------------------------------------------------------------------
# Bracket shape display + reshape/view -1 resolution (Phases 1 & 2)
# ---------------------------------------------------------------------------


def test_bracket_shape_display_round_trips():
    spec = TensorSpec(("B", "S", 4096), "bfloat16")
    assert shapes.format_shape(spec) == "[B, S, 4096]"
    assert shapes.format_shape_with_dtype(spec) == "[B, S, 4096] bfloat16"
    # Merged reshape dims keep the ``*`` product for readability.
    assert shapes.format_shape_dims(["B*S", "4096"]) == "[B*S, 4096]"
    assert shapes.format_shape_dims([]) == ""
    # Round-trip: display -> dims recovers the original list.
    assert shapes.parse_shape_dims("[B*S, 4096]") == ["B*S", "4096"]
    assert shapes.parse_shape_dims(shapes.format_shape(spec)) == ["B", "S", "4096"]
    # Legacy `` x `` form is still parseable defensively.
    assert shapes.parse_shape_dims("B x S x 4096") == ["B", "S", "4096"]
    # tensor_shape (edge labels) now uses the bracket form on every edge.
    assert shapes.format_shape_tensor(spec) == "[B, S, 4096] bfloat16"
    # A shape without a dtype omits the suffix in both display forms.
    no_dtype = TensorSpec(("B", "S", 4096), "")
    assert shapes.format_shape_with_dtype(no_dtype) == "[B, S, 4096]"
    assert shapes.format_shape_tensor(no_dtype) == "[B, S, 4096]"


def test_split_and_write_port_shape_bracket_round_trip():
    dims, dtype = merge._split_shape_dtype("[B*S, 4096] bfloat16")
    assert dims == ["B*S", "4096"] and dtype == "bfloat16"
    # Bracketed dims with no dtype suffix.
    assert merge._split_shape_dtype("[B, S, 4096]") == (["B", "S", "4096"], "")
    metadata = {
        "attrs": [
            {"key": "shape", "value": "old"},
            {"key": "tensor_shape", "value": "old"},
        ]
    }
    shape_attr = metadata["attrs"][0]
    merge._write_port_shape(metadata, shape_attr, ["B*S", "4096"], "bfloat16")
    assert shape_attr["value"] == "[B*S, 4096] bfloat16"
    tensor_val = next(
        a["value"] for a in metadata["attrs"] if a["key"] == "tensor_shape"
    )
    # tensor_shape (edge labels) now uses the bracket form on every edge.
    assert tensor_val == "[B*S, 4096] bfloat16"


def test_merge_data_movement_reshape_flatten_and_expand():
    spec = ArchitectureSpec(name="T", model_type="t", raw_config={})

    def _op(label, shape):
        return SimpleNamespace(label=label, details=[f"shape: {shape}"])

    # Reshape (-1, 4096) over (B, S, 4096) -> merged (B*S, 4096), no literal -1.
    src = TensorSpec(("B", "S", 4096), "bfloat16")
    out = merge._data_movement_shape(_op("Reshape", "-1, 4096"), src, spec=spec)
    assert out.shape == ("B*S", 4096)
    assert "-1" not in [str(d) for d in out.shape]
    # Expand keeps the source dim at its ``-1`` slot (broadcast, not flatten).
    exp = merge._data_movement_shape(_op("Expand", "-1, 8"), TensorSpec((4, 1)), spec=spec)
    assert exp.shape == (4, 8)


# ---------------------------------------------------------------------------
# Secondary ModuleList tagging (Deliverable D — vision tower N× groups)
# ---------------------------------------------------------------------------


def _meta_group(path, length, element_class):
    from TraceLens.ModelUtils.meta_trace import MetaModuleGroup

    return MetaModuleGroup(
        path=path,
        length=length,
        element_class=element_class,
        signatures=("()",) * length,
    )


def test_rename_namespace_prefix_rewrites_nodes_attrs_and_configs():
    nodes = [
        {"id": "visual.blocks/x", "namespace": "visual/VisionBlock"},
        {"id": "visual.blocks/y", "namespace": "visual/VisionBlock/Attn"},
        {"id": "other", "namespace": "visual/PatchEmbed"},
    ]
    attrs = {
        "visual/VisionBlock": {"input_shape": "[Pv, 1024]"},
        "visual/VisionBlock/Attn": {"input_shape": "[Pv, 1024]"},
        "visual/PatchEmbed": {"input_shape": "[Pv, 1176]"},
    }
    import re as _re

    configs = [
        {"namespaceRegex": f"^{_re.escape('visual/VisionBlock')}$"},
        {"namespaceRegex": f"^{_re.escape('visual/VisionBlock/Attn')}$"},
        {"namespaceRegex": f"^{_re.escape('visual/PatchEmbed')}$"},
    ]
    merge._rename_namespace_prefix(
        nodes, attrs, configs, "visual/VisionBlock", "visual/24x_VisionBlock"
    )
    # node ids untouched; only namespace rewritten, and only under the prefix.
    assert [n["namespace"] for n in nodes] == [
        "visual/24x_VisionBlock",
        "visual/24x_VisionBlock/Attn",
        "visual/PatchEmbed",
    ]
    assert [n["id"] for n in nodes] == ["visual.blocks/x", "visual.blocks/y", "other"]
    assert set(attrs) == {
        "visual/24x_VisionBlock",
        "visual/24x_VisionBlock/Attn",
        "visual/PatchEmbed",
    }
    assert configs[0]["namespaceRegex"] == f"^{_re.escape('visual/24x_VisionBlock')}$"
    assert configs[1]["namespaceRegex"] == f"^{_re.escape('visual/24x_VisionBlock/Attn')}$"
    assert configs[2]["namespaceRegex"] == f"^{_re.escape('visual/PatchEmbed')}$"


def test_tag_secondary_module_groups_tags_vision_block_not_decoder():
    spec = SimpleNamespace(
        decoder_class="DecoderLayer",
        meta_module_groups=[
            _meta_group("language_model.layers", 45, "DecoderLayer"),
            _meta_group("visual.blocks", 24, "VisionBlock"),
        ],
    )
    nodes = [
        {"id": "decoder/a", "namespace": "45x_DecoderLayer"},
        {"id": "visual.blocks/b", "namespace": "visual/VisionBlock"},
        {"id": "visual.blocks/c", "namespace": "visual/VisionBlock/Attn"},
    ]
    attrs = {"visual/VisionBlock": {"input_shape": "[Pv, 1024]"}}
    configs: list = []
    merge._tag_secondary_module_groups(
        nodes,
        spec=spec,
        group_node_attributes=attrs,
        group_node_configs=configs,
    )
    # Vision block gets the count-bearing namespace + repeat attr; decoder (its
    # element class already the primary banner) is left alone.
    assert {n["namespace"] for n in nodes} == {
        "45x_DecoderLayer",
        "visual/24x_VisionBlock",
        "visual/24x_VisionBlock/Attn",
    }
    assert attrs["visual/24x_VisionBlock"]["repeat"] == "24x_VisionBlock"
    assert attrs["visual/24x_VisionBlock"]["input_shape"] == "[Pv, 1024]"
    assert any(
        c["namespaceRegex"] == "^visual/24x_VisionBlock$"
        and c["borderColor"] == "#c0392b"
        for c in configs
    )


def test_tag_secondary_module_groups_two_disjoint_repeat_namespaces():
    """Two independent secondary ModuleLists -> two disjoint N× namespaces."""
    spec = SimpleNamespace(
        decoder_class="",
        meta_module_groups=[
            _meta_group("audio.blocks", 6, "AudioBlock"),
            _meta_group("visual.blocks", 24, "VisionBlock"),
        ],
    )
    nodes = [
        {"id": "audio/x", "namespace": "audio/AudioBlock"},
        {"id": "visual/y", "namespace": "visual/VisionBlock"},
    ]
    attrs: dict = {}
    configs: list = []
    merge._tag_secondary_module_groups(
        nodes, spec=spec, group_node_attributes=attrs, group_node_configs=configs
    )
    assert {n["namespace"] for n in nodes} == {
        "audio/6x_AudioBlock",
        "visual/24x_VisionBlock",
    }
    assert attrs["audio/6x_AudioBlock"]["repeat"] == "6x_AudioBlock"
    assert attrs["visual/24x_VisionBlock"]["repeat"] == "24x_VisionBlock"


def test_tag_secondary_module_groups_noops_without_match_or_groups():
    # No meta groups -> untouched.
    nodes = [{"id": "n", "namespace": "visual/VisionBlock"}]
    attrs: dict = {}
    configs: list = []
    merge._tag_secondary_module_groups(
        nodes,
        spec=SimpleNamespace(decoder_class="", meta_module_groups=[]),
        group_node_attributes=attrs,
        group_node_configs=configs,
    )
    assert nodes[0]["namespace"] == "visual/VisionBlock"
    assert not configs
    # Group whose element class matches no rendered namespace -> untouched.
    merge._tag_secondary_module_groups(
        nodes,
        spec=SimpleNamespace(
            decoder_class="",
            meta_module_groups=[_meta_group("experts", 128, "ExpertMLP")],
        ),
        group_node_attributes=attrs,
        group_node_configs=configs,
    )
    assert nodes[0]["namespace"] == "visual/VisionBlock"
    assert not configs


def test_annotate_op_input_signatures_records_tensor_and_scalar_inputs():
    """An op node gets profiler-style operand attrs: a tensor input resolved from
    its producer's output port, followed by a scalar arg parsed from ``details``."""
    import json

    producer = {
        "id": "prod",
        "attrs": [
            {"key": "class_name", "value": "GetPos"},
            {"key": "output_shape", "value": "[Pv, 2] int64"},
        ],
        "outputsMetadata": [
            {"id": "0", "attrs": [{"key": "shape", "value": "[Pv, 2] int64"}]}
        ],
    }
    unsqueeze = {
        "id": "op:unsqueeze",
        "label": "Unsqueeze",
        "attrs": [
            {"key": "class_name", "value": "Unsqueeze"},
            {"key": "details", "value": "dim: -1"},
            {"key": "output_shape", "value": "[Pv, 2, 1] int64"},
        ],
        "incomingEdges": [
            {"sourceNodeId": "prod", "sourceNodeOutputId": "0", "targetNodeInputId": "0"}
        ],
    }
    merge._annotate_op_input_signatures([producer, unsqueeze])

    attrs = {a["key"]: a["value"] for a in unsqueeze["attrs"]}
    assert attrs["op_type"] == "Unsqueeze"
    assert json.loads(attrs["input_shapes"]) == [["Pv", "2"], []]
    assert json.loads(attrs["input_types"]) == ["int64", "Scalar"]
    assert json.loads(attrs["concrete_inputs"]) == ["", "-1"]


def test_annotate_op_input_signatures_skips_synthetic_boundaries():
    """Synthetic @input/@output tiles are not operation nodes and get no attrs."""
    boundary = {
        "id": "grp/@input",
        "label": "x",
        "attrs": [{"key": "synthetic", "value": "@input"}],
    }
    merge._annotate_op_input_signatures([boundary])
    assert all(a["key"] not in {"op_type", "input_shapes"} for a in boundary["attrs"])


# --------------------------------------------------------------------------- #
# Render-time constant filter (viewer_page). Constants stay in the JSON as
# first-class ``constant``-tagged nodes; the HTML render drops them behind a
# flag defaulting to drop. These units guard the filter's contract: it removes
# constant nodes + their edges, prunes now-empty group frames, and NEVER mutates
# its input (the persisted JSON must stay complete).
# --------------------------------------------------------------------------- #

from TraceLens.Visualizer.model_explorer_export import viewer_page


def _const_node(node_id, namespace=""):
    return {
        "id": node_id,
        "label": "Const",
        "namespace": namespace,
        "attrs": [{"key": "constant", "value": "true"}],
    }


def _op_with_edges(node_id, sources, namespace=""):
    return {
        "id": node_id,
        "label": "Op",
        "namespace": namespace,
        "incomingEdges": [{"sourceNodeId": s} for s in sources],
    }


def _sample_graph():
    return {
        "nodes": [
            _const_node("w:const", namespace="blk"),
            {"id": "act", "label": "Act", "namespace": "blk"},
            _op_with_edges("mul", ["act", "w:const"], namespace="blk"),
        ],
        "groupNodeAttributes": {"blk": {"attr": "x"}},
    }


def test_graph_without_constants_drops_constant_node_and_its_edge():
    graph = _sample_graph()
    filtered = viewer_page._graph_without_constants(graph)
    ids = [n["id"] for n in filtered["nodes"]]
    assert "w:const" not in ids
    assert set(ids) == {"act", "mul"}
    # The consumer keeps its activation edge, loses only the constant edge.
    mul = next(n for n in filtered["nodes"] if n["id"] == "mul")
    assert [e["sourceNodeId"] for e in mul["incomingEdges"]] == ["act"]


def test_graph_without_constants_does_not_mutate_input():
    graph = _sample_graph()
    before = copy.deepcopy(graph)
    viewer_page._graph_without_constants(graph)
    assert graph == before, "filter must not mutate the source graph (JSON stays complete)"


def test_graph_without_constants_prunes_now_empty_group():
    # A group frame whose only member is a constant must not be drawn empty.
    graph = {
        "nodes": [
            _const_node("only:const", namespace="lonely"),
            {"id": "keep", "label": "Act", "namespace": "kept"},
        ],
        "groupNodeAttributes": {"lonely": {"attr": "x"}, "kept": {"attr": "y"}},
    }
    filtered = viewer_page._graph_without_constants(graph)
    assert "lonely" not in filtered["groupNodeAttributes"]
    assert "kept" in filtered["groupNodeAttributes"]


def test_graph_without_constants_strips_edge_only_incoming():
    # A consumer fed solely by a constant loses its incomingEdges entirely.
    graph = {
        "nodes": [
            _const_node("c"),
            _op_with_edges("sink", ["c"]),
        ]
    }
    filtered = viewer_page._graph_without_constants(graph)
    sink = next(n for n in filtered["nodes"] if n["id"] == "sink")
    assert "incomingEdges" not in sink


def test_payload_without_constants_keeps_source_payload_complete():
    payload = {
        "graphCollections": [
            {"graphs": [_sample_graph()]},
        ]
    }
    before = copy.deepcopy(payload)
    filtered = viewer_page._payload_without_constants(payload)
    # Source payload still carries the constant (JSON artifact is never filtered).
    assert payload == before
    src_ids = [n["id"] for n in payload["graphCollections"][0]["graphs"][0]["nodes"]]
    assert "w:const" in src_ids
    # Filtered overlay has no constants.
    out_ids = [n["id"] for n in filtered["graphCollections"][0]["graphs"][0]["nodes"]]
    assert "w:const" not in out_ids


def test_compose_viewer_html_flag_controls_constant_visibility():
    payload = {
        "graphCollections": [{"graphs": [_sample_graph()]}],
        "label": "model",
    }
    dropped = viewer_page.compose_viewer_html(payload, drop_constants=True)
    kept = viewer_page.compose_viewer_html(payload, drop_constants=False)
    # The constant node id appears in the shown-constants HTML, not the default.
    assert "w:const" not in dropped
    assert "w:const" in kept
