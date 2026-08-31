"""Regression tests for GLM-5.3-Flash linear-attention graph wiring."""

from __future__ import annotations

import pytest

from model_explorer_export.merge import build_merged_model_graph
from visualizer.computation_graph import add_forward_output, build_computation_graph
from visualizer.loader import load_model_spec


def _linear_attn_tree(spec):
    return next((item for item in spec.export_block_trees if "Linear Attn" in item[0]))


def _graph_key(graph, suffix: str) -> str:
    matches = [node.key for node in graph.nodes if node.key.endswith(suffix)]
    assert len(matches) == 1, matches
    return matches[0]


def _linear_attn_variant_prefix(spec) -> str:
    variant = next(v for v in spec.layer_variants if "LinearAttention" in (v.attention_class or ""))
    return f"decoder/{variant.count}x_{variant.attention_class}_{variant.ffn_class}"


def test_glm53_linear_attention_has_single_output_exit():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    title, tree = _linear_attn_tree(spec)
    graph = build_computation_graph(tree, basic_ops=spec.basic_ops)
    source_indices = {src for src, _target in graph.links}
    exits = [
        index
        for index, node in enumerate(graph.nodes)
        if index not in source_indices
        and node.synthetic not in {"@input", "@hidden_states", "@tensor"}
    ]
    assert exits, f"Expected at least one exit for {title}"

    add_forward_output(graph)
    output_sources = [
        src
        for src, tgt in graph.links
        if graph.nodes[tgt].label == "Output"
    ]
    assert len(output_sources) == 1
    assert graph.nodes[output_sources[0]].label == "Linear"


def test_glm53_linear_attention_gate_chain_is_not_short_circuited():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    _title, tree = _linear_attn_tree(spec)
    graph = build_computation_graph(tree, basic_ops=spec.basic_ops)
    keys = {node.key for node in graph.nodes}

    g_a_key = _graph_key(graph, ":g_a_proj")
    g_b_key = _graph_key(graph, ":g_b_proj")
    o_norm_key = _graph_key(graph, ":o_norm:norm")
    o_norm_mul_key = _graph_key(graph, ":o_norm:mul")

    assert g_a_key in keys
    assert g_b_key in keys
    assert o_norm_key in keys

    key_to_index = {node.key: index for index, node in enumerate(graph.nodes)}
    links = set(graph.links)

    assert (key_to_index[g_a_key], key_to_index[g_b_key]) in links
    assert (key_to_index[g_b_key], key_to_index[o_norm_mul_key]) in links
    assert (key_to_index["@input"], key_to_index[g_b_key]) not in links


def test_glm53_spine_hyperconnection_stays_on_variant_namespace():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    prefix = _linear_attn_variant_prefix(spec)
    variant_namespace = next(
        node["namespace"]
        for node in graph["nodes"]
        if node["id"] == f"{prefix}/input_layernorm"
    )

    attn_nodes = [
        node
        for node in graph["nodes"]
        if node["id"].startswith(f"{prefix}/attn_hc/")
    ]
    ffn_nodes = [
        node
        for node in graph["nodes"]
        if node["id"].startswith(f"{prefix}/ffn_hc/")
    ]
    assert attn_nodes and ffn_nodes
    assert all(
        node.get("namespace", "").endswith("/attn_hc")
        for node in attn_nodes
    )
    assert all(
        node.get("namespace", "").endswith("/ffn_hc")
        for node in ffn_nodes
    )
    assert variant_namespace in attn_nodes[0].get("namespace", "")


def test_glm53_hyperconnection_expands_mhc_math():
    pytest.importorskip("huggingface_hub")
    from visualizer.computation_graph import build_computation_graph
    from visualizer.block_tree import build_block_node

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    tree = build_block_node(
        attr_name="attn_hc",
        class_name="Glm5NextTextHyperConnection",
        registry=spec.class_registry,
        basic_ops=spec.basic_ops,
        infer_init_steps=True,
    )
    assert len(tree.children) >= 20
    labels = [child.label for child in tree.children]
    assert labels.index("RMSNorm") < labels.index("Linear")

    graph = build_computation_graph(tree, basic_ops=spec.basic_ops)
    labels = {node.label for node in graph.nodes}
    assert "Sigmoid" in labels
    assert "Softmax" in labels
    assert "Sum" in labels


def test_glm53_ffn_hc_input_norm_precedes_linear():
    pytest.importorskip("huggingface_hub")
    from model_explorer_export.merge import _resolve_section_tree_for_component
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    hc = spec.class_registry["Glm5NextTextHyperConnection"]
    assert hc.forward_calls.index("input_norm") < next(
        index
        for index, step in enumerate(hc.forward_calls)
        if step.startswith("@op_") and hc.forward_operations[step].label == "Linear"
    )
    assert "@functional_linear" not in hc.forward_calls

    tree = build_block_node(
        attr_name="ffn_hc",
        class_name="Glm5NextTextHyperConnection",
        registry=spec.class_registry,
        basic_ops=spec.basic_ops,
        infer_init_steps=True,
    )
    labels = [child.label for child in tree.children]
    assert labels.index("RMSNorm") < labels.index("Linear")
    assert len(tree.children) >= 20

    variant = next(v for v in spec.layer_variants if "LinearAttention" in (v.attention_class or ""))
    ffn_hc = next(c for c in spec.block_components if c.attr_name == "ffn_hc")
    title, section_tree = _resolve_section_tree_for_component(
        spec,
        ffn_hc,
        variant=variant,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    assert title == "FFN"
    section_labels = [child.label for child in section_tree.children]
    assert section_labels.index("RMSNorm") < section_labels.index("Linear")


def test_glm53_ffn_hc_expands_hyperconnection_not_moe():
    pytest.importorskip("huggingface_hub")
    from model_explorer_export.merge import _resolve_section_tree_for_component
    from model_explorer_export.overview import component_has_detail_section
    from visualizer.basic_ops import BasicOpFilter

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    variant = next(v for v in spec.layer_variants if "LinearAttention" in (v.attention_class or ""))
    prefix = _linear_attn_variant_prefix(spec)
    ffn_hc = next(c for c in spec.block_components if c.attr_name == "ffn_hc")

    assert component_has_detail_section(ffn_hc, spec)
    title, _tree = _resolve_section_tree_for_component(
        spec,
        ffn_hc,
        variant=variant,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    assert title == "FFN"

    graph = build_merged_model_graph(spec)
    ffn_nodes = [node for node in graph["nodes"] if node["id"].startswith(f"{prefix}/ffn_hc/")]
    labels = {node.get("label") for node in ffn_nodes}
    assert "RMSNorm" in labels
    assert "Linear" in labels
    assert all("TopkRouter" not in node["id"] for node in ffn_nodes)
    assert all(
        "Glm5NextTextHyperConnection" in node.get("namespace", "")
        or node.get("namespace", "").endswith("/ffn_hc")
        for node in ffn_nodes
        if not node["id"].endswith("/@input")
    )
