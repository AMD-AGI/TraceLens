"""Tests for Model Explorer export from TraceLens computation graphs."""

from __future__ import annotations

from pathlib import Path

import pytest

from visualizer.ast_analyze import SYNTHETIC_ATTENTION
from visualizer.basic_ops import BasicOpFilter
from visualizer.block_tree import BlockNode, build_decoder_block_trees
from visualizer.computation_graph import build_computation_graph
from visualizer.extract import load_architecture

from model_explorer_export.adapter import attach_subgraph_links, computation_graph_to_explorer_graph
from model_explorer_export.build import build_model_explorer_payload

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


def test_computation_graph_to_explorer_graph_topology():
    root = _mla_fixture_root()
    computation = build_computation_graph(root)
    graph = computation_graph_to_explorer_graph(computation, graph_id="attn", label="MLA")

    assert graph["id"] == "attn"
    assert len(graph["nodes"]) == len(computation.nodes)

    node_by_id = {node["id"]: node for node in graph["nodes"]}
    assert len(node_by_id) == len(graph["nodes"])

    edge_count = sum(len(node.get("incomingEdges", [])) for node in graph["nodes"])
    assert edge_count == len(computation.links)

    attention = next(node for node in graph["nodes"] if node["label"] == "Attention")
    assert attention["style"]["backgroundColor"] == "#5dade2"
    assert any(attr["key"] == "operation" and attr["value"] == "gpu_kernel" for attr in attention["attrs"])


def test_inline_frames_become_namespaces():
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
    graph = computation_graph_to_explorer_graph(computation, graph_id="mlp")

    if computation.inline_frames:
        namespaced = [node for node in graph["nodes"] if node["namespace"]]
        assert namespaced


def test_attach_subgraph_links():
    main = {
        "id": "decoder",
        "nodes": [
            {
                "id": "self_attn",
                "label": "Attention",
                "namespace": "",
                "attrs": [{"key": "attr_name", "value": "self_attn"}],
            }
        ],
    }
    nested = {"id": "self_attn", "nodes": [{"id": "q_proj", "label": "Linear", "namespace": ""}]}
    attach_subgraph_links([main, nested], attr_name_to_graph_id={"self_attn": "self_attn"})
    assert main["nodes"][0]["subgraphIds"] == ["self_attn"]


def test_merged_sections_include_input_ports():
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    payload = build_model_explorer_payload(spec)
    graph = payload["graphCollections"][0]["graphs"][0]
    input_nodes = [node for node in graph["nodes"] if node["id"].endswith("/@input") or node["id"] == "@input"]
    assert any(node["label"] == "hidden_states" for node in input_nodes)
    attn_input = next(
        node
        for node in input_nodes
        if "/self_attn/@input" in node["id"] or node["id"].endswith("/KimiMLAAttention/@input")
    )
    assert attn_input["incomingEdges"]


def test_inject_group_inputs_adds_namespace_input_port():
    from model_explorer_export.merge import _inject_group_inputs

    section_nodes = [
        {
            "id": "decoder/moe/@input",
            "label": "hidden_states",
            "namespace": "1x_Layer/moe",
            "attrs": [{"key": "synthetic", "value": "@input"}],
        },
        {
            "id": "decoder/moe/seq:0:gate:linear:0",
            "label": "Linear",
            "namespace": "1x_Layer/moe/KimiMoEGate",
            "incomingEdges": [
                {
                    "sourceNodeId": "decoder/moe/@input",
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            ],
        },
        {
            "id": "decoder/moe/seq:0:gate:sigmoid:1",
            "label": "Sigmoid",
            "namespace": "1x_Layer/moe/KimiMoEGate",
            "incomingEdges": [
                {
                    "sourceNodeId": "decoder/moe/seq:0:gate:linear:0",
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            ],
        },
    ]
    _inject_group_inputs(section_nodes)
    gate_input = next(
        node for node in section_nodes if node["id"] == "decoder/moe/seq:0:gate/@input"
    )
    assert gate_input["label"] == "hidden_states"
    assert gate_input["incomingEdges"] == [
        {
            "sourceNodeId": "decoder/moe/@input",
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "0",
        }
    ]


def test_kimi_layer_variants_export_three_decoder_splits():
    from pathlib import Path

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    spec = load_architecture(
        "moonshotai/Kimi-K3",
        code_path=code_path,
        detailed=True,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    payload = build_model_explorer_payload(spec)
    graph = payload["graphCollections"][0]["graphs"][0]
    decoder_ns = "93x_KimiDecoderLayer"
    attrs = graph["groupNodeAttributes"]
    variant_keys = sorted(
        key for key in attrs if key.startswith(f"{decoder_ns}/") and key.count("/") == 1
    )
    assert len(variant_keys) == 3
    joined = " ".join(variant_keys)
    assert "68x_KimiDeltaAttention_KimiSparseMoeBlock" in joined
    assert "24x_KimiMLAAttention_KimiSparseMoeBlock" in joined
    assert "1x_KimiDeltaAttention_KimiMLP" in joined
    delta_prefix = f"{decoder_ns}/68x_KimiDeltaAttention_KimiSparseMoeBlock/KimiDeltaAttention"
    assert any(node["namespace"].startswith(delta_prefix) for node in graph["nodes"])
    mla_prefix = f"{decoder_ns}/24x_KimiMLAAttention_KimiSparseMoeBlock/KimiMLAAttention"
    assert any(node["namespace"].startswith(mla_prefix) for node in graph["nodes"])
    assert not any(node["label"] == "LayerNorm" for node in graph["nodes"])
    assert any(node["label"] == "RMSNorm" for node in graph["nodes"])
    assert not any("_attn_pipeline" in (node.get("namespace") or "") for node in graph["nodes"])
    pipeline_ns = f"{decoder_ns}/68x_KimiDeltaAttention_KimiSparseMoeBlock/KimiDeltaAttention/chunk_kda_pipeline"
    assert graph["groupNodeAttributes"].get(pipeline_ns, {}).get("label") == "chunk_kda pipeline"
    assert not any(
        "merge:0" in node["id"]
        for node in graph["nodes"]
        if (node.get("namespace") or "").startswith(pipeline_ns)
    )
    pipeline_input = (
        f"decoder/68x_KimiDeltaAttention_KimiSparseMoeBlock/self_attn/@attn_pipeline/@input"
    )
    assert any(node["id"] == pipeline_input for node in graph["nodes"])
    q_tensor = next(
        node
        for node in graph["nodes"]
        if node.get("label") == "q" and node.get("namespace") == pipeline_ns
    )
    assert q_tensor["incomingEdges"][0]["sourceNodeId"] == pipeline_input
    assert any(
        "q_conv1d_activation" in edge["sourceNodeId"]
        for edge in next(node for node in graph["nodes"] if node["id"] == pipeline_input)[
            "incomingEdges"
        ]
    )
    l2norm_ns = f"{pipeline_ns}/l2norm_fwd"
    l2norm_labels = {
        node["label"]
        for node in graph["nodes"]
        if node.get("namespace") == l2norm_ns
        and any(a.get("key") == "class_name" and a.get("value") == "KernelSubOp" for a in node.get("attrs", []))
    }
    assert l2norm_labels == {"Sum sq", "Sqrt", "Inv sqrt", "Normalize"}
    assert graph["groupNodeAttributes"][l2norm_ns]["label"] == "L2Norm"
    gate_ns = f"{pipeline_ns}/kda_gate_chunk_cumsum"
    assert graph["groupNodeAttributes"][gate_ns]["label"] == "Gate cumsum"
    gate_labels = {
        node["label"]
        for node in graph["nodes"]
        if node.get("namespace") == gate_ns
        and any(a.get("key") == "class_name" and a.get("value") == "KernelSubOp" for a in node.get("attrs", []))
    }
    assert gate_labels == {"Exp", "Softplus", "Gate mul", "Gate", "Chunk cumsum"}
    assert "Sigmoid" not in {
        node["label"] for node in graph["nodes"] if node.get("namespace") == gate_ns
    }
    mlp_variant_ns = f"{decoder_ns}/1x_KimiDeltaAttention_KimiMLP"
    assert not any(
        node["id"] == f"decoder/1x_KimiDeltaAttention_KimiMLP/{norm_attr}/@input"
        for norm_attr in ("input_layernorm", "post_attention_layernorm")
        for node in graph["nodes"]
    )
    input_norm = next(
        node
        for node in graph["nodes"]
        if node["id"] == "decoder/1x_KimiDeltaAttention_KimiMLP/input_layernorm/input_layernorm"
    )
    assert input_norm["incomingEdges"][0]["sourceNodeId"] == "embed_tokens/embed_tokens"
    assert any(
        node["id"] == "decoder/1x_KimiDeltaAttention_KimiMLP/self_attn/@input"
        for node in graph["nodes"]
    )


def test_build_model_explorer_payload_is_single_merged_graph():
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    payload = build_model_explorer_payload(spec)
    graphs = payload["graphCollections"][0]["graphs"]
    assert len(graphs) == 1
    assert graphs[0]["id"] == "model"
    assert not any(node.get("subgraphIds") for node in graphs[0]["nodes"])
    namespaces = {node["namespace"] for node in graphs[0]["nodes"] if node.get("namespace")}
    assert namespaces
    overview_labels = [node["label"] for node in graphs[0]["nodes"]]
    assert "Tokenized text" in overview_labels


def test_merged_graph_uses_white_text_on_dark_blocks():
    root = BlockNode(
        attr_name="moe",
        class_name="MoE",
        role="moe",
        label="MoE",
        children=[
            BlockNode(
                attr_name="kernel",
                class_name="KernelOp",
                role="other",
                label="Fused dispatch",
                is_basic=True,
                details=["kernel: moe_dispatch"],
            ),
        ],
    )
    graph = computation_graph_to_explorer_graph(build_computation_graph(root), graph_id="moe")
    dark_nodes = [
        node
        for node in graph["nodes"]
        if node.get("style", {}).get("backgroundColor") in {"#8e44ad", "#566573", "#85929e", "#3a4550"}
    ]
    assert dark_nodes
    assert all(node["style"]["textColor"] == "#ffffff" for node in dark_nodes)


def test_fact_sheet_box_in_merged_graph():
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    payload = build_model_explorer_payload(spec)
    graph = payload["graphCollections"][0]["graphs"][0]
    fact_node = next(node for node in graph["nodes"] if node["id"] == "@fact_sheet")
    assert fact_node["namespace"] == ""
    assert fact_node["label"].startswith("Fact sheet\n")
    assert "Model type:" in fact_node["label"]
    assert "\n• " in fact_node["label"]
    assert not any(
        edge["sourceNodeId"] == "@fact_sheet"
        for node in graph["nodes"]
        for edge in node.get("incomingEdges", [])
    )
    assert not fact_node.get("incomingEdges")


def test_build_model_explorer_payload_custom_model():
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    payload = build_model_explorer_payload(spec)

    assert payload["source"] == "tracelens-computation-graph"
    assert payload["graphCollections"]
    graphs = payload["graphCollections"][0]["graphs"]
    assert graphs

    for graph in graphs:
        assert graph["id"]
        assert graph["nodes"]
        incoming = sum(len(node.get("incomingEdges", [])) for node in graph["nodes"])
        assert incoming >= 1

    labels = {node["label"] for node in graphs[0]["nodes"]}
    assert "Linear" in labels
