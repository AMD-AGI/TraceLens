"""Tests for serializable model graph IR and operation classification."""

from __future__ import annotations

import json
from pathlib import Path

from visualizer.ast_analyze import SYNTHETIC_ATTENTION, analyze_source
from visualizer.basic_ops import BasicOpFilter
from visualizer.block_tree import BlockNode, build_decoder_block_trees
from visualizer.computation_graph import SYNTHETIC_INPUT, build_computation_graph
from visualizer.extract import load_architecture
from visualizer.model_graph import (
    NodeKind,
    OperationKind,
    assert_operations_reduced,
    build_architecture_model_graphs,
    build_model_graph,
    classify_operation,
    collect_non_reduced_operations,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _mla_fixture_root() -> BlockNode:
    def leaf(name: str) -> BlockNode:
        return BlockNode(attr_name=name, class_name="Linear", role="other", label="Linear", is_basic=True)

    return BlockNode(
        attr_name="attn",
        class_name="Attn",
        role="attention",
        label="Attn",
        parallel_gates=["g_proj"],
        children=[
            leaf("q_a_proj"),
            leaf("q_a_layernorm"),
            leaf("q_b_proj"),
            leaf("kv_a_proj_with_mqa"),
            leaf("kv_a_layernorm"),
            leaf("kv_b_proj"),
            BlockNode(
                attr_name=SYNTHETIC_ATTENTION,
                class_name="AttentionOp",
                role="attention",
                label="Attention",
                is_basic=True,
                details=["kernel: flash_attn"],
            ),
            leaf("g_proj"),
            leaf("o_proj"),
        ],
    )


def test_classify_operation_kinds():
    linear = BlockNode(
        attr_name="q_proj",
        class_name="Linear",
        role="other",
        label="Linear",
        is_basic=True,
    )
    assert classify_operation(linear) == OperationKind.NN_MODULE

    functional = BlockNode(
        attr_name="@functional_linear",
        class_name="Linear",
        role="other",
        label="Linear",
        is_basic=True,
    )
    assert classify_operation(functional) == OperationKind.TORCH_FUNCTIONAL

    kernel = BlockNode(
        attr_name=SYNTHETIC_ATTENTION,
        class_name="AttentionOp",
        role="attention",
        label="Attention",
        details=["kernel: chunk_kda"],
    )
    assert classify_operation(kernel) == OperationKind.GPU_KERNEL

    assert classify_operation(None, synthetic=SYNTHETIC_INPUT, label="hidden_states") == OperationKind.SYNTHETIC
    assert classify_operation(None, synthetic=None, label="×") == OperationKind.SYNTHETIC


def test_build_model_graph_matches_computation_graph_topology():
    root = _mla_fixture_root()
    computation = build_computation_graph(root)
    model_graph = build_model_graph(root, title="MLA")

    assert model_graph.title == "MLA"
    assert len(model_graph.nodes) == len(computation.nodes)
    assert len(model_graph.edges) == len(computation.links)

    labels = {node.label for node in model_graph.nodes}
    assert "hidden_states" in labels
    assert "×" in labels
    assert "Attention" in labels

    operations = {node.label: node.operation for node in model_graph.nodes}
    assert operations["Attention"] == OperationKind.GPU_KERNEL


def test_model_graph_json_roundtrip():
    root = _mla_fixture_root()
    model_graph = build_model_graph(root, title="MLA")
    payload = json.loads(model_graph.to_json())

    assert payload["title"] == "MLA"
    assert payload["nodes"]
    assert payload["edges"]
    for node in payload["nodes"]:
        assert {"id", "kind", "label"} <= set(node)
        assert node["kind"] in {kind.value for kind in NodeKind}
    for edge in payload["edges"]:
        assert {"source", "target", "style"} <= set(edge)


def test_assert_operations_reduced_for_mla_fixture():
    root = _mla_fixture_root()
    model_graph = build_model_graph(root, title="MLA")
    assert_operations_reduced(model_graph)


def test_collect_non_reduced_operations_flags_unexpanded_module():
    unexpanded = BlockNode(
        attr_name="self_attn",
        class_name="CustomLatentAttention",
        role="attention",
        label="Latent Attention",
        is_basic=False,
    )
    graph = build_model_graph(unexpanded, title="Latent Attention")
    issues = collect_non_reduced_operations(graph)
    assert issues
    assert issues[0].operation in {OperationKind.UNKNOWN, OperationKind.COMPOSITE}


def test_build_architecture_model_graphs_custom_model():
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    payload = build_architecture_model_graphs(spec)
    assert payload["name"] == "Custom MLA MoE"
    assert payload["sections"]

    for section in payload["sections"]:
        graph = section["graph"]
        assert graph["nodes"]
        assert graph["edges"]
        assert any(node["label"] == "hidden_states" for node in graph["nodes"])


def test_decoder_block_tree_graph_has_nn_and_functional_ops():
    source = (FIXTURES / "custom_model" / "modeling_custom.py").read_text(encoding="utf-8")
    analysis = analyze_source(source, filename="modeling_custom.py")
    basic_ops = BasicOpFilter.from_cli(add=[r"(?i)^Linear$"])
    trees = build_decoder_block_trees(
        analysis.block_components,
        analysis.class_registry,
        basic_ops,
    )
    moe_tree = next(tree for title, tree in trees if "MoE" in title)
    graph = build_model_graph(moe_tree, title="MoE")

    op_kinds = {node.operation for node in graph.nodes if node.operation is not None}
    assert OperationKind.NN_MODULE in op_kinds
    assert OperationKind.SYNTHETIC in op_kinds

    reduced = {
        node.operation
        for node in graph.nodes
        if node.operation not in {OperationKind.SYNTHETIC, None}
    }
    assert reduced <= {
        OperationKind.NN_MODULE,
        OperationKind.TORCH_FUNCTIONAL,
        OperationKind.GPU_KERNEL,
        OperationKind.COMPOSITE,
    }


def test_model_graph_inline_frames_use_node_ids():
    root = BlockNode(
        attr_name="mlp",
        class_name="MLP",
        role="ffn",
        label="MLP",
        children=[
            BlockNode(attr_name="gate_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="up_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="down_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
        ],
    )
    computation = build_computation_graph(root)
    model_graph = build_model_graph(root, title="MLP")

    if computation.inline_frames:
        frame = model_graph.inline_frames[0]
        assert frame.node_ids
        node_ids = {node.id for node in model_graph.nodes}
        assert set(frame.node_ids) <= node_ids


def test_model_graph_edge_styles():
    root = _mla_fixture_root()
    computation = build_computation_graph(root)
    model_graph = build_model_graph(root, title="MLA")

    dashed_pairs = {
        (computation.nodes[src].key or f"node:{src}", computation.nodes[tgt].key or f"node:{tgt}")
        for src, tgt in computation.dashed_links
    }
    model_dashed = {
        (edge.source, edge.target) for edge in model_graph.edges if edge.style == "dashed"
    }
    assert model_dashed == dashed_pairs
    side_pairs = {
        (computation.nodes[src].key or f"node:{src}", computation.nodes[tgt].key or f"node:{tgt}")
        for src, tgt in computation.side_entry_links
    }
    model_side = {
        (edge.source, edge.target) for edge in model_graph.edges if edge.style == "side"
    }
    assert model_side == side_pairs


def test_model_graph_subgraphs_for_nested_composites():
    pipeline = BlockNode(
        attr_name="kernel",
        class_name="KernelPipeline",
        role="attention",
        label="chunk_kda pipeline",
        children=[
            BlockNode(
                attr_name="step0",
                class_name="KernelOp",
                role="other",
                label="l2norm_fwd",
                details=["kernel: chunk_kda"],
            ),
            BlockNode(
                attr_name="out",
                class_name="KernelOutput",
                role="other",
                label="output",
            ),
        ],
    )
    root = BlockNode(
        attr_name="attn",
        class_name="Attn",
        role="attention",
        label="Attn",
        children=[pipeline],
    )
    graph = build_model_graph(root, title="Attn", include_subgraphs=True)
    assert graph.subgraphs or any(node.operation == OperationKind.GPU_KERNEL for node in graph.nodes)
