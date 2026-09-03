###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Focused coverage for merged export helpers and symbolic shape fallbacks."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from model_explorer_export import labels, merge, overview, shapes, styles
from visualizer.basic_ops import BasicOpFilter
from visualizer.block_tree import BlockNode
from visualizer.blocks import BlockComponent, LayerVariant
from visualizer.computation_graph import ComputationGraph, GraphNodeSpec
from visualizer.extract import ArchitectureSpec
from visualizer.model_graph import (
    GraphEdge,
    ModelGraph,
    ModelGraphNode,
    NodeKind,
    OperationKind,
)
from visualizer.shape_inference import (
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
    assert overview._display_label(spec.block_components[2], spec) == "Logits"
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
    assert by_id["head"]["label"] == "Logits"
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
        "visualizer.model_graph.classify_operation", lambda *a, **k: operation
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
    assert shapes.format_shape(spec) == "B x 2xH x N*D x "
    assert shapes.format_shape_tensor(spec) == "Bx2xHxN*Dx"
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
        {"key": "output_shape", "value": "1 x 2 float32"},
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
    assert result["outer"]["input_shape"] == "B x S x 16 float16"
    assert result["outer"]["output_shape"] == "B x S x 16 float16"
    assert result["outer/inner"] == result["outer"]
    assert shapes._namespace_chain("/outer//inner/") == ["outer", "outer/inner"]
    store: dict[str, list[str]] = {}
    shapes._record_shape(store, "g", "x")
    shapes._record_shape(store, "g", "x")
    assert store == {"g": ["x"]}


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
        ("sum", [TensorSpec(("B", "S", 8))], ["dim: -1"], ("B", "S", 1), "float16"),
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
    ) == TensorSpec((8, 16), "float32")
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
    monkeypatch.setattr(merge, "fill_missing_node_shapes", lambda nodes, context: None)
    monkeypatch.setattr(
        merge,
        "group_boundary_shapes",
        lambda nodes: {"tail": {"input_shape": "B x S x 16"}},
    )
    inferencer = SimpleNamespace(context=ShapeContext({"H": 16}))
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
        "visualizer.ast_analyze.classify_matmul_label",
        lambda external_inputs: f"matmul({','.join(external_inputs)})",
    )
    monkeypatch.setattr(
        "visualizer.ast_analyze.operation_display_label",
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
    assert weight_view == TensorSpec((8, 16), "float32")
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
