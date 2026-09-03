###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for Model Explorer export from TraceLens computation graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from visualizer.ast_analyze import SYNTHETIC_ATTENTION
from visualizer.basic_ops import BasicOpFilter
from visualizer.block_tree import BlockNode
from visualizer.computation_graph import build_computation_graph
from visualizer.extract import load_architecture

from model_explorer_export.adapter import (
    attach_subgraph_links,
    computation_graph_to_explorer_graph,
)
from model_explorer_export.build import build_model_explorer_payload

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_output_boundary_exports_duplicate_producers_as_named_ports():
    producer = BlockNode(
        attr_name="producer",
        class_name="Mean",
        role="other",
        label="Mean",
        is_basic=True,
    )
    root = BlockNode(
        attr_name="root",
        class_name="Wrapper",
        role="other",
        label="Wrapper",
        children=[producer],
        input_label="x",
        forward_return_slots={"left": "producer", "right": "producer"},
        forward_return_order=["left", "right"],
        primary_return_slot="right",
        primary_output_step="producer",
        multi_return_module=True,
    )
    graph = build_computation_graph(root)
    exported = computation_graph_to_explorer_graph(graph, graph_id="duplicate")
    output = next(node for node in exported["nodes"] if node["label"] == "Output")

    assert [item["id"] for item in output["inputsMetadata"]] == ["left", "right"]
    assert [item["id"] for item in output["outputsMetadata"]] == ["left", "right"]
    assert {edge["targetNodeInputId"] for edge in output["incomingEdges"]} == {
        "left",
        "right",
    }
    assert len({edge["sourceNodeId"] for edge in output["incomingEdges"]}) == 1


def _mla_fixture_root() -> BlockNode:
    def leaf(name: str) -> BlockNode:
        return BlockNode(
            attr_name=name,
            class_name="Linear",
            role="other",
            label="Linear",
            is_basic=True,
        )

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
    graph = computation_graph_to_explorer_graph(
        computation, graph_id="attn", label="MLA"
    )

    assert graph["id"] == "attn"
    assert len(graph["nodes"]) == len(computation.nodes)

    node_by_id = {node["id"]: node for node in graph["nodes"]}
    assert len(node_by_id) == len(graph["nodes"])

    edge_count = sum(len(node.get("incomingEdges", [])) for node in graph["nodes"])
    assert edge_count == len(computation.links)

    attention = next(node for node in graph["nodes"] if node["label"] == "Attention")
    assert attention["style"]["backgroundColor"] == "#f5d9d9"
    assert attention["style"]["textColor"] == "#1a1a1a"
    assert any(
        attr["key"] == "operation" and attr["value"] == "gpu_kernel"
        for attr in attention["attrs"]
    )


def test_inline_frames_become_namespaces():
    root = BlockNode(
        attr_name="mlp",
        class_name="MLP",
        role="ffn",
        label="MLP",
        children=[
            BlockNode(
                attr_name="gate_proj",
                class_name="Linear",
                role="other",
                label="Linear",
                is_basic=True,
            ),
            BlockNode(
                attr_name="up_proj",
                class_name="Linear",
                role="other",
                label="Linear",
                is_basic=True,
            ),
            BlockNode(
                attr_name="down_proj",
                class_name="Linear",
                role="other",
                label="Linear",
                is_basic=True,
            ),
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
    nested = {
        "id": "self_attn",
        "nodes": [{"id": "q_proj", "label": "Linear", "namespace": ""}],
    }
    attach_subgraph_links(
        [main, nested], attr_name_to_graph_id={"self_attn": "self_attn"}
    )
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
    input_nodes = [
        node
        for node in graph["nodes"]
        if node["id"].endswith("/@input") or node["id"] == "@input"
    ]
    assert any(
        node["label"].split("\n", 1)[0] == "hidden_states" for node in input_nodes
    )
    attn_input = next(
        node for node in input_nodes if node["id"].endswith("/block_sparse_moe/@input")
    )
    assert attn_input["incomingEdges"]


def test_inject_group_inputs_labels_external_tensor_ports_on_edges():
    from model_explorer_export.merge import _inject_group_inputs

    namespace = "decoder/KimiDeltaAttention/chunk_kda_pipeline"
    q_node_id = "decoder/attn/@pipeline:tensor:0"
    k_node_id = "decoder/attn/@pipeline:tensor:1"
    section_nodes = [
        {
            "id": "decoder/attn/q_act",
            "label": "q act",
        },
        {
            "id": "decoder/attn/k_act",
            "label": "k act",
        },
        {
            "id": q_node_id,
            "label": "q",
            "namespace": namespace,
            "attrs": [{"key": "synthetic", "value": "@tensor"}],
            "incomingEdges": [
                {
                    "sourceNodeId": "decoder/attn/q_act",
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            ],
        },
        {
            "id": k_node_id,
            "label": "k",
            "namespace": namespace,
            "attrs": [{"key": "synthetic", "value": "@tensor"}],
            "incomingEdges": [
                {
                    "sourceNodeId": "decoder/attn/k_act",
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            ],
        },
    ]
    _inject_group_inputs(section_nodes)
    assert not any("/@input:" in node.get("id", "") for node in section_nodes)
    q_node = next(node for node in section_nodes if node["id"] == q_node_id)
    k_node = next(node for node in section_nodes if node["id"] == k_node_id)
    assert q_node["incomingEdges"][0]["metadata"] == {"port_label": "q"}
    assert k_node["incomingEdges"][0]["metadata"] == {"port_label": "k"}
    assert q_node["inputsMetadata"] == [
        {"id": "0", "attrs": [{"key": "port_label", "value": "q"}]}
    ]


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


def test_inject_group_inputs_splits_multiple_operands_into_input_blocks():
    from model_explorer_export.merge import _inject_group_inputs

    namespace = "Experts/Loop_repeated"
    section_nodes = [
        {"id": "final", "label": "Zeros like", "namespace": "Experts"},
        {"id": "indices", "label": "Where", "namespace": "Experts"},
        {
            "id": "index-add",
            "label": "Index add",
            "namespace": namespace,
            "attrs": [{"key": "boundary_input", "value": "final"}],
            "incomingEdges": [
                {
                    "sourceNodeId": "final",
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                },
                {
                    "sourceNodeId": "indices",
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "1",
                },
            ],
        },
    ]

    _inject_group_inputs(section_nodes)

    inputs = [
        node
        for node in section_nodes
        if node.get("namespace") == namespace
        and any(
            attr.get("key") == "synthetic" and attr.get("value") == "@input"
            for attr in node.get("attrs", [])
        )
    ]
    assert len(inputs) == 2
    assert all(len(node.get("incomingEdges", [])) == 1 for node in inputs)
    index_add = next(node for node in section_nodes if node["id"] == "index-add")
    assert {edge["sourceNodeId"] for edge in index_add["incomingEdges"]} == {
        node["id"] for node in inputs
    }


def test_replace_tile_with_group_puts_nested_diagram_in_the_tile_slot():
    from model_explorer_export.merge import _replace_tile_with_group

    def edge(source: str) -> dict[str, Any]:
        return {
            "sourceNodeId": source,
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "0",
        }

    producer = "self_attn/fan0-2:q_a_layernorm:q_a_layernorm:0"
    tile = "self_attn/fan0-2:indexer:indexer:0"
    consumer = "self_attn/fan0-2:q_b_proj:q_b_proj:0"
    nested_input = f"{tile}/@input"
    nested_exit = f"{tile}/seq:1:mask:mask:0"
    nested_ns = "Attention/Glm5NextTextIndexer"

    nested_nodes = [
        {
            "id": nested_input,
            "label": "hidden_states",
            "namespace": nested_ns,
            "attrs": [{"key": "synthetic", "value": "@input"}],
        },
        {
            "id": nested_exit,
            "label": "Masked fill",
            "namespace": nested_ns,
            "incomingEdges": [edge(nested_input)],
        },
    ]
    section_nodes = [
        {"id": producer, "label": "RMSNorm", "namespace": "Attention"},
        {
            "id": tile,
            "label": "Glm5NextTextIndexer",
            "namespace": "Attention",
            "incomingEdges": [edge(producer)],
        },
        {
            "id": consumer,
            "label": "Linear",
            "namespace": "Attention",
            "incomingEdges": [edge(tile)],
        },
        *nested_nodes,
    ]

    _replace_tile_with_group(
        section_nodes, nested_nodes, tile_id=tile, exit_id=nested_exit
    )

    by_id = {node["id"]: node for node in section_nodes}
    assert tile not in by_id
    assert [item["sourceNodeId"] for item in by_id[nested_input]["incomingEdges"]] == [
        producer
    ]
    assert [item["sourceNodeId"] for item in by_id[consumer]["incomingEdges"]] == [
        nested_exit
    ]


def test_replace_tile_with_group_distributes_named_inputs_to_nested_ports():
    from model_explorer_export.merge import _replace_tile_with_group

    def edge(source: str, label: str) -> dict[str, Any]:
        return {
            "sourceNodeId": source,
            "sourceNodeOutputId": "0",
            "targetNodeInputId": label,
            "metadata": {"port_label": label},
        }

    tile = "moe/experts"
    namespace = "MoE/Experts"
    nested_inputs = [
        {
            "id": f"{tile}/@input:{label}",
            "label": label,
            "namespace": namespace,
            "attrs": [
                {"key": "synthetic", "value": "@input"},
                {"key": "port_label", "value": label},
            ],
        }
        for label in ("hidden_states", "top_k_index", "top_k_weights")
    ]
    section_nodes = [
        {
            "id": tile,
            "label": "Experts",
            "namespace": "MoE",
            "incomingEdges": [
                edge("tokens", "hidden_states"),
                edge("router-index", "top_k_index"),
                edge("router-weight", "top_k_weights"),
            ],
        },
        *nested_inputs,
    ]

    _replace_tile_with_group(section_nodes, nested_inputs, tile_id=tile)

    by_label = {node["label"]: node for node in nested_inputs}
    assert by_label["hidden_states"]["incomingEdges"][0]["sourceNodeId"] == "tokens"
    assert by_label["top_k_index"]["incomingEdges"][0]["sourceNodeId"] == "router-index"
    assert (
        by_label["top_k_weights"]["incomingEdges"][0]["sourceNodeId"] == "router-weight"
    )


def test_mirror_boundary_inputs_exposes_multi_input_ports_in_parent():
    from model_explorer_export.merge import _mirror_boundary_inputs

    namespace = "MoE/Experts"
    nodes = [
        {"id": "tokens", "label": "tokens", "namespace": "MoE"},
        {"id": "indices", "label": "indices", "namespace": "MoE"},
        *[
            {
                "id": f"experts/@input:{label}",
                "label": label,
                "namespace": namespace,
                "attrs": [
                    {"key": "synthetic", "value": "@input"},
                    {"key": "port_label", "value": label},
                ],
                "incomingEdges": [
                    {
                        "sourceNodeId": source,
                        "sourceNodeOutputId": "0",
                        "targetNodeInputId": label,
                    }
                ],
            }
            for label, source in (
                ("hidden_states", "tokens"),
                ("top_k_index", "indices"),
            )
        ],
    ]

    _mirror_boundary_inputs(nodes)

    mirrors = [
        node
        for node in nodes
        if any(
            attr.get("key") == "synthetic" and attr.get("value") == "@input_mirror"
            for attr in node.get("attrs", [])
        )
    ]
    assert {node["label"] for node in mirrors} == {
        "hidden_states",
        "top_k_index",
    }
    assert all(node["namespace"] == "MoE" for node in mirrors)
    inner = next(node for node in nodes if node["id"] == "experts/@input:top_k_index")
    assert inner["incomingEdges"][0]["sourceNodeId"].endswith(
        "@input_mirror:top_k_index^top_k_index"
    )


def test_inject_group_inputs_treats_nested_ops_as_internal():
    from model_explorer_export.merge import _inject_group_inputs

    moe_input = "decoder/moe/@input"
    mlp_input = "decoder/moe/shared_experts/@input"
    gate = "decoder/moe/shared_experts:gate_proj"
    up = "decoder/moe/shared_experts:up_proj"
    situ = "decoder/moe/shared_experts:situ"
    mul = "decoder/moe/shared_experts:mul"
    down = "decoder/moe/shared_experts:down_proj"
    mlp_ns = "1x_Layer/moe/KimiMLP"
    situ_ns = f"{mlp_ns}/SituAndMul"

    section_nodes = [
        {
            "id": moe_input,
            "label": "hidden_states",
            "namespace": "1x_Layer/moe",
            "attrs": [{"key": "synthetic", "value": "@input"}],
        },
        {
            "id": gate,
            "label": "Linear",
            "namespace": mlp_ns,
            "incomingEdges": [
                {
                    "sourceNodeId": moe_input,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            ],
        },
        {
            "id": up,
            "label": "Linear",
            "namespace": "1x_Layer/moe",
            "incomingEdges": [
                {
                    "sourceNodeId": moe_input,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            ],
        },
        {
            "id": situ,
            "label": "Situ",
            "namespace": situ_ns,
            "incomingEdges": [
                {
                    "sourceNodeId": gate,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            ],
        },
        {
            "id": mul,
            "label": "Multiply",
            "namespace": situ_ns,
            "incomingEdges": [
                {
                    "sourceNodeId": situ,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                },
                {
                    "sourceNodeId": up,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "1",
                },
            ],
        },
        {
            "id": down,
            "label": "Linear",
            "namespace": mlp_ns,
            "incomingEdges": [
                {
                    "sourceNodeId": mul,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            ],
        },
    ]
    _inject_group_inputs(section_nodes)

    mlp_input_node = next(node for node in section_nodes if node["id"] == mlp_input)
    assert mlp_input_node["label"] == "x"
    assert mlp_input_node["incomingEdges"] == [
        {
            "sourceNodeId": moe_input,
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "0",
        }
    ]
    assert not any(node["id"].endswith("/SituAndMul/@input") for node in section_nodes)

    down_node = next(node for node in section_nodes if node["id"] == down)
    assert down_node["incomingEdges"] == [
        {
            "sourceNodeId": mul,
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "0",
        }
    ]

    gate_node = next(node for node in section_nodes if node["id"] == gate)
    assert gate_node["incomingEdges"] == [
        {
            "sourceNodeId": mlp_input,
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
    delta_prefix = (
        f"{decoder_ns}/68x_KimiDeltaAttention_KimiSparseMoeBlock/KimiDeltaAttention"
    )
    assert any(node["namespace"].startswith(delta_prefix) for node in graph["nodes"])
    mla_prefix = (
        f"{decoder_ns}/24x_KimiMLAAttention_KimiSparseMoeBlock/KimiMLAAttention"
    )
    assert any(node["namespace"].startswith(mla_prefix) for node in graph["nodes"])
    q_norm_ns = next(
        node["namespace"]
        for node in graph["nodes"]
        if ":q_a_layernorm:" in node["id"] and node.get("label") == "Power"
    )
    kv_norm_ns = next(
        node["namespace"]
        for node in graph["nodes"]
        if ":kv_a_layernorm:" in node["id"] and node.get("label") == "Power"
    )
    assert q_norm_ns.endswith("/q_a_layernorm")
    assert kv_norm_ns.endswith("/kv_a_layernorm")
    assert q_norm_ns != kv_norm_ns
    assert not any(
        (node.get("namespace") or "").endswith("/KimiRMSNorm")
        for node in graph["nodes"]
        if (node.get("namespace") or "").startswith(mla_prefix)
    )
    assert not any(node["label"] == "LayerNorm" for node in graph["nodes"])
    assert any(node["label"] == "RMSNorm" for node in graph["nodes"])
    assert not any(
        "_attn_pipeline" in (node.get("namespace") or "") for node in graph["nodes"]
    )
    pipeline_ns = f"{decoder_ns}/68x_KimiDeltaAttention_KimiSparseMoeBlock/KimiDeltaAttention/chunk_kda_pipeline"
    assert (
        graph["groupNodeAttributes"].get(pipeline_ns, {}).get("label")
        == "chunk_kda pipeline"
    )
    l2norm_q_ns = f"{pipeline_ns}/l2norm_fwd_q"
    l2norm_k_ns = f"{pipeline_ns}/l2norm_fwd_k"
    assert graph["groupNodeAttributes"][l2norm_q_ns]["label"] == "L2Norm (q)"
    assert graph["groupNodeAttributes"][l2norm_k_ns]["label"] == "L2Norm (k)"
    fused_beta_ns = f"{pipeline_ns}/fused_beta_sigmoid"
    fused_beta_labels = {
        node["label"]
        for node in graph["nodes"]
        if node.get("namespace") == fused_beta_ns
        and any(
            a.get("key") == "class_name" and a.get("value") == "KernelSubOp"
            for a in node.get("attrs", [])
        )
    }
    assert fused_beta_labels == {"Sigmoid", "x scale"}
    assert not any("?" in node.get("label", "") for node in graph["nodes"])
    l2norm_q_cfg = next(
        cfg
        for cfg in graph["groupNodeConfigs"]
        if cfg.get("backgroundColor") == "#f5d9d9"
        and "l2norm_fwd_q" in cfg.get("namespaceRegex", "")
    )
    assert l2norm_q_cfg["textColor"] == "#1a1a1a"
    assert not any(
        "merge:0" in node["id"]
        for node in graph["nodes"]
        if (node.get("namespace") or "").startswith(pipeline_ns)
    )
    q_tensor = next(
        node
        for node in graph["nodes"]
        if node.get("label") == "q" and node.get("namespace") == pipeline_ns
    )
    k_tensor = next(
        node
        for node in graph["nodes"]
        if node.get("label") == "k" and node.get("namespace") == pipeline_ns
    )
    v_tensor = next(
        node
        for node in graph["nodes"]
        if node.get("label") == "v" and node.get("namespace") == pipeline_ns
    )
    g_tensor = next(
        node
        for node in graph["nodes"]
        if node.get("label") == "g" and node.get("namespace") == pipeline_ns
    )
    beta_tensor = next(
        node
        for node in graph["nodes"]
        if node.get("label") == "beta" and node.get("namespace") == pipeline_ns
    )
    assert not any(
        "/@input:" in node.get("id", "")
        and node.get("namespace", "").startswith(pipeline_ns)
        for node in graph["nodes"]
    )
    for tensor, label in (
        (q_tensor, "q"),
        (k_tensor, "k"),
        (v_tensor, "v"),
        (g_tensor, "g"),
        (beta_tensor, "beta"),
    ):
        assert tensor["incomingEdges"][0]["metadata"] == {"port_label": label}
    assert "q_conv1d_activation" in q_tensor["incomingEdges"][0]["sourceNodeId"]
    l2norm_ns = f"{pipeline_ns}/l2norm_fwd_q"
    l2norm_labels = {
        node["label"]
        for node in graph["nodes"]
        if node.get("namespace") == l2norm_ns
        and any(
            a.get("key") == "class_name" and a.get("value") == "KernelSubOp"
            for a in node.get("attrs", [])
        )
    }
    assert l2norm_labels == {"Sum", "Sqrt", "1/x", "X"}
    assert graph["groupNodeAttributes"][l2norm_ns]["label"] == "L2Norm (q)"
    l2norm_k_ns = f"{pipeline_ns}/l2norm_fwd_k"
    assert not any(
        node.get("id", "").endswith("forward_l2norm_fwd_/@input")
        for node in graph["nodes"]
    )

    def _l2norm_multiply(namespace: str) -> dict[str, Any]:
        return next(
            node
            for node in graph["nodes"]
            if node.get("namespace") == namespace and node.get("label") == "X"
        )

    def _l2norm_input(namespace: str, port: str) -> dict[str, Any]:
        return next(
            node
            for node in graph["nodes"]
            if node.get("namespace") == namespace
            and node.get("label") == port
            and any(
                a.get("key") == "synthetic" and a.get("value") == "@input"
                for a in node.get("attrs", [])
            )
        )

    q_multiply = _l2norm_multiply(l2norm_ns)
    k_multiply = _l2norm_multiply(l2norm_k_ns)
    q_input = _l2norm_input(l2norm_ns, "q")
    k_input = _l2norm_input(l2norm_k_ns, "k")
    assert q_input["incomingEdges"][0]["sourceNodeId"].endswith(":tensor:0")
    assert k_input["incomingEdges"][0]["sourceNodeId"].endswith(":tensor:1")
    for multiply, input_node in ((q_multiply, q_input), (k_multiply, k_input)):
        sources = [edge["sourceNodeId"] for edge in multiply["incomingEdges"]]
        assert len(sources) == 2
        assert input_node["id"] in sources
        assert any("sub_2" in source for source in sources)
        assert not any("sub_1" in source for source in sources)
    gate_ns = f"{pipeline_ns}/kda_gate_chunk_cumsum"
    assert graph["groupNodeAttributes"][gate_ns]["label"] == "Gate cumsum"

    def _labeled_entry(namespace: str, label: str) -> dict[str, Any]:
        return next(
            node
            for node in graph["nodes"]
            if node.get("namespace") == namespace
            and any(
                (edge.get("metadata") or {}).get("port_label") == label
                for edge in node.get("incomingEdges", [])
            )
        )

    gate_entry = _labeled_entry(gate_ns, "g")
    fused_beta_entry = _labeled_entry(fused_beta_ns, "beta")
    assert gate_entry["inputsMetadata"][0]["attrs"] == [
        {"key": "port_label", "value": "g"}
    ]
    assert fused_beta_entry["inputsMetadata"][0]["attrs"] == [
        {"key": "port_label", "value": "beta"}
    ]
    gate_labels = {
        node["label"]
        for node in graph["nodes"]
        if node.get("namespace") == gate_ns
        and any(
            a.get("key") == "class_name" and a.get("value") == "KernelSubOp"
            for a in node.get("attrs", [])
        )
    }
    assert gate_labels == {"Exp", "Softplus", "X", "Sigmoid", "CumSum"}
    node_ids = {node["id"] for node in graph["nodes"]}
    for norm_attr in ("input_layernorm", "post_attention_layernorm"):
        prefix = f"decoder/1x_KimiDeltaAttention_KimiMLP/{norm_attr}"
        assert f"{prefix}/@input" in node_ids
        output = next(
            node for node in graph["nodes"] if node["id"] == f"{prefix}/@output"
        )
        assert [item["id"] for item in output["outputsMetadata"]] == ["hidden_states"]
        # One returned tensor, so no mirror is drawn outside the block.
        assert output["label"] == "hidden_states"
        assert f"{prefix}/@output^hidden_states" not in node_ids
    input_norm = next(
        node
        for node in graph["nodes"]
        if node["id"].startswith(
            "decoder/1x_KimiDeltaAttention_KimiMLP/input_layernorm/"
        )
        and node["label"] == "Power"
    )
    input_boundary = next(
        node
        for node in graph["nodes"]
        if node["id"] == "decoder/1x_KimiDeltaAttention_KimiMLP/input_layernorm/@input"
    )
    assert input_norm["incomingEdges"][0]["sourceNodeId"] == input_boundary["id"]
    assert input_boundary["incomingEdges"][0]["sourceNodeId"] == (
        "decoder/1x_KimiDeltaAttention_KimiMLP/@input"
    )
    norm_nodes = [node for node in graph["nodes"] if node["id"].startswith("norm/")]
    assert {"Power", "Mean", "Reciprocal sqrt", "Multiply"} <= {
        node["label"] for node in norm_nodes
    }
    assert {"norm/@input", "norm/@output"} <= node_ids
    assert "norm/@output^hidden_states" not in node_ids
    final_norm_output = next(
        node for node in graph["nodes"] if node["id"] == "norm/@output"
    )
    assert [item["id"] for item in final_norm_output["outputsMetadata"]] == [
        "hidden_states"
    ]
    assert final_norm_output["label"] == "hidden_states"
    assert any(node["id"] == "lm_head" for node in graph["nodes"])
    assert not any(node.get("namespace") == "lm_head" for node in graph["nodes"])
    assert not any(node["id"] == "lm_head/@input" for node in graph["nodes"])
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
    namespaces = {
        node["namespace"] for node in graphs[0]["nodes"] if node.get("namespace")
    }
    assert namespaces
    overview_labels = [node["label"] for node in graphs[0]["nodes"]]
    assert "Tokenized text" in overview_labels


def test_merged_graph_uses_readable_text_on_colored_blocks():
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
    graph = computation_graph_to_explorer_graph(
        build_computation_graph(root), graph_id="moe"
    )
    purple_nodes = [
        node
        for node in graph["nodes"]
        if node.get("style", {}).get("backgroundColor") == "#f5d9d9"
    ]
    assert purple_nodes
    assert all(node["style"]["textColor"] == "#1a1a1a" for node in purple_nodes)


def test_norm_linear_and_multiply_use_basic_op_gray():
    from visualizer.computation_graph import (
        ComputationGraph,
        GraphNodeSpec,
        build_computation_graph,
    )

    norm = BlockNode(
        attr_name="input_layernorm",
        class_name="KimiRMSNorm",
        role="norm",
        label="RMSNorm",
    )
    linear = BlockNode(
        attr_name="q_proj",
        class_name="Linear",
        role="other",
        label="Linear",
        is_basic=True,
    )
    graph = computation_graph_to_explorer_graph(
        build_computation_graph(
            BlockNode(
                attr_name="attn",
                class_name="Attn",
                role="attention",
                label="Attn",
                children=[norm, linear],
            )
        ),
        graph_id="attn",
    )
    norm_node = next(node for node in graph["nodes"] if node["label"] == "RMSNorm")
    linear_node = next(node for node in graph["nodes"] if node["label"] == "Linear")
    assert norm_node["style"] == {"backgroundColor": "#bdc3c7", "textColor": "#1a1a1a"}
    assert linear_node["style"] == {
        "backgroundColor": "#bdc3c7",
        "textColor": "#1a1a1a",
    }

    multiply_graph = computation_graph_to_explorer_graph(
        ComputationGraph(
            nodes=[
                GraphNodeSpec(
                    key="mul",
                    block=BlockNode(
                        attr_name="gate_mul",
                        class_name="Multiply",
                        role="other",
                        label="Multiply",
                    ),
                    label="Multiply",
                )
            ]
        ),
        graph_id="mul",
    )
    multiply_node = multiply_graph["nodes"][0]
    assert multiply_node["style"] == {
        "backgroundColor": "#bdc3c7",
        "textColor": "#1a1a1a",
    }


def test_fact_sheet_panel_in_payload():
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    payload = build_model_explorer_payload(spec)
    fact = payload["tracelensViewer"]["factSheet"]
    assert fact["title"] == "Fact sheet"
    assert "Model type:" in fact["body"]
    assert fact["body"].startswith("- Model type:")
    graph = payload["graphCollections"][0]["graphs"][0]
    assert not any(node["id"] == "@fact_sheet" for node in graph["nodes"])


def test_default_html_output_path_replaces_slashes():
    from model_explorer_export.cli import default_html_output_path

    assert (
        default_html_output_path("moonshotai/Kimi-K3", None).name
        == "moonshotai_Kimi-K3.html"
    )


def test_implicit_cli_output_uses_html_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from model_explorer_export.cli import main

    fixture = FIXTURES / "llama_like"
    monkeypatch.chdir(tmp_path)
    exit_code = main([str(fixture), "--title", "LlamaLike"])
    assert exit_code == 0
    html_files = list(tmp_path.glob("*.html"))
    assert len(html_files) == 1
    assert html_files[0].name == "llama_like.html"
    assert not list(tmp_path.glob("*_model_explorer.json"))
    assert not list(tmp_path.glob("worker.js"))


def test_viewer_url_uses_root_path():
    from model_explorer_export.serve import viewer_url

    assert viewer_url(8765) == "http://127.0.0.1:8765/"


def test_viewer_shell_reserves_fact_sheet_column():
    index_html = (
        Path(__file__).resolve().parents[1]
        / "model_explorer_export"
        / "viewer"
        / "index.html"
    ).read_text(encoding="utf-8")
    app_js = (
        Path(__file__).resolve().parents[1]
        / "model_explorer_export"
        / "viewer"
        / "app.js"
    ).read_text(encoding="utf-8")
    assert "tracelens-fact-sheet" in index_html
    assert "tracelens-fact-sheet-resizer" in index_html
    assert "--tracelens-fact-sheet-width" in index_html
    assert "grid-template-columns" in index_html
    assert "body {" in index_html and "display: grid" in index_html
    assert "tracelens-fact-sheet-body" in app_js
    assert "initFactSheetResize" in app_js
    assert "bodyHtml" in app_js
    assert "factSheet.hidden = false" in app_js
    assert "hideInfoPanel: true" in app_js
    assert "loadEmbeddedPayload" in app_js
    assert "model_explorer_show_on_node_item_types_v2" in app_js
    assert "Op node attributes" in app_js
    assert "output_shape" in app_js


def test_compose_viewer_html_embeds_payload_without_external_json():
    from model_explorer_export.viewer_page import compose_viewer_html

    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    payload = build_model_explorer_payload(spec)
    html = compose_viewer_html(payload, inline_app=True)

    assert 'id="tracelens-payload"' in html
    assert "graphCollections" in html
    assert "tracelensViewer" in html
    assert 'id="tracelens-worker-source"' in html
    assert "./app.js" not in html
    assert "loadEmbeddedPayload" in html
    start_tag = '<script id="tracelens-payload" type="application/json">'
    start = html.index(start_tag) + len(start_tag)
    end = html.index("</script>", start)
    embedded_json = html[start:end]
    json.loads(embedded_json)
    assert "\n" not in embedded_json


def test_save_viewer_html_is_self_contained(tmp_path: Path):
    from model_explorer_export.viewer_page import save_viewer_html

    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    payload = build_model_explorer_payload(spec)
    html_path = save_viewer_html(payload, tmp_path / "custom_model.html")

    assert html_path.exists()
    assert not (tmp_path / "worker.js").exists()
    html = html_path.read_text(encoding="utf-8")
    assert 'id="tracelens-payload"' in html
    assert 'id="tracelens-worker-source"' in html


def test_fact_sheet_group_attributes_in_graph_info():
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    payload = build_model_explorer_payload(spec)
    graph = payload["graphCollections"][0]["graphs"][0]
    attrs = graph["groupNodeAttributes"][""]
    assert "architecture_fact_sheet" in attrs
    assert "Model type:" in attrs["architecture_fact_sheet"]


def test_fact_sheet_forward_sequence_uses_graph_display_labels():
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    payload = build_model_explorer_payload(spec)
    fact_body = payload["tracelensViewer"]["factSheet"]["body"]
    graph_attrs = payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"][""]

    forward_line = next(
        line for line in fact_body.splitlines() if line.startswith("- Forward:")
    )
    assert forward_line == (
        "- Forward: RMSNorm -> CustomLatent Attn -> Add -> RMSNorm -> CustomSharedExpertMoE -> Add"
    )
    assert (
        graph_attrs["forward"]
        == "RMSNorm → CustomLatent Attn → Add → RMSNorm → CustomSharedExpertMoE → Add"
    )


def test_kernel_frame_labels_split_l2norm_q_and_k_and_sanitize_unicode():
    from model_explorer_export.labels import apply_kernel_frame_labels

    nodes = [
        {
            "id": "pipeline/forward_l2norm_fwd_q:sub_0",
            "label": "Sum",
            "namespace": "decoder/KimiDeltaAttention/chunk_kda_pipeline/l2norm_fwd",
            "attrs": [
                {"key": "attr_name", "value": "forward_l2norm_fwd_q_sub_0"},
                {"key": "class_name", "value": "KernelSubOp"},
            ],
        },
        {
            "id": "pipeline/forward_l2norm_fwd_k:sub_0",
            "label": "Sum",
            "namespace": "decoder/KimiDeltaAttention/chunk_kda_pipeline/l2norm_fwd",
            "attrs": [
                {"key": "attr_name", "value": "forward_l2norm_fwd_k_sub_0"},
                {"key": "class_name", "value": "KernelSubOp"},
            ],
        },
        {
            "id": "pipeline/fused_beta:sub_1",
            "label": "× scale",
            "namespace": "decoder/KimiDeltaAttention/chunk_kda_pipeline/fused_beta_sigmoid",
            "attrs": [
                {"key": "attr_name", "value": "forward_fused_beta_sigmoid_beta_sub_1"},
                {"key": "class_name", "value": "KernelSubOp"},
            ],
        },
    ]
    group_attrs: dict[str, dict[str, str]] = {}
    apply_kernel_frame_labels(nodes, group_attrs)
    assert nodes[0]["namespace"].endswith("/l2norm_fwd_q")
    assert nodes[1]["namespace"].endswith("/l2norm_fwd_k")
    assert nodes[0]["label"] == "Sum"
    assert nodes[2]["label"] == "x scale"
    assert group_attrs[nodes[0]["namespace"]]["label"] == "L2Norm (q)"
    assert group_attrs[nodes[1]["namespace"]]["label"] == "L2Norm (k)"


def test_fill_missing_node_shapes_seeds_and_propagates():
    from visualizer.shape_inference import ShapeContext

    from model_explorer_export.shapes import fill_missing_node_shapes

    context = ShapeContext(
        dims={"B": "B", "S": "S", "H": 4096, "V": 32000}, dtype="float16"
    )
    nodes: list[dict[str, Any]] = [
        {"id": "@input", "label": "Tokenized text"},
        {
            "id": "embed_tokens",
            "label": "Token Embedding",
            "incomingEdges": [{"sourceNodeId": "@input", "sourceNodeOutputId": "0"}],
        },
        {
            "id": "decoder/norm",
            "label": "RMSNorm",
            "incomingEdges": [
                {"sourceNodeId": "embed_tokens", "sourceNodeOutputId": "0"}
            ],
        },
        {
            "id": "mlp",
            "label": "MLP",
            "incomingEdges": [
                {"sourceNodeId": "decoder/norm", "sourceNodeOutputId": "0"}
            ],
        },
        {"id": "lm_head", "label": "Logits"},
        {"id": "orphan", "label": "Detached"},
    ]

    fill_missing_node_shapes(nodes, context=context)

    shapes = {
        node["id"]: next(
            attr["value"]
            for attr in node["outputsMetadata"][0]["attrs"]
            if attr["key"] == "shape"
        )
        for node in nodes
    }
    assert shapes == {
        "@input": "B x S int64",
        "embed_tokens": "B x S x 4096 float16",
        "decoder/norm": "B x S x 4096 float16",
        "mlp": "B x S x 4096 float16",
        "lm_head": "B x S x 32000 float16",
        "orphan": "B x S x 4096 float16",
    }


def test_merged_graph_includes_output_shape_attrs():
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    payload = build_model_explorer_payload(spec)
    nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
    shaped_nodes = [
        node
        for node in nodes
        if any(attr.get("key") == "output_shape" for attr in node.get("attrs", []))
    ]
    assert shaped_nodes
    # Every node needs output metadata or Model Explorer labels its edges "?".
    assert all(node.get("outputsMetadata") for node in nodes)
    router = next(
        node for node in shaped_nodes if node["id"].endswith("router:router:0")
    )
    shape_attr = next(attr for attr in router["attrs"] if attr["key"] == "output_shape")
    assert shape_attr["value"] == "B x S x 64 float16"
    assert payload["tracelensViewer"]["dimensions"]["H"] == 4096
    assert payload["tracelensViewer"]["dtype"] == "float16"

    payload_without = build_model_explorer_payload(spec, include_shapes=False)
    assert "dimensions" not in payload_without["tracelensViewer"]
    assert not any(
        any(attr.get("key") == "output_shape" for attr in node.get("attrs", []))
        for node in payload_without["graphCollections"][0]["graphs"][0]["nodes"]
    )


def test_expandable_groups_carry_their_boundary_shapes():
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    graph = build_model_explorer_payload(spec)["graphCollections"][0]["graphs"][0]
    attrs = graph["groupNodeAttributes"]
    namespaces = {node["namespace"] for node in graph["nodes"] if node.get("namespace")}
    assert namespaces
    # Model Explorer leaves edges into collapsed groups unlabeled, so each group states
    # what crosses its boundary.
    for namespace in namespaces:
        assert attrs[namespace]["input_shape"]
        assert attrs[namespace]["output_shape"]

    decoder = next(name for name in attrs if name and "/" not in name)
    assert attrs[decoder]["input_shape"] == "B x S x 4096 float16"
    assert attrs[decoder]["output_shape"] == "B x S x 4096 float16"

    without_shapes = build_model_explorer_payload(spec, include_shapes=False)
    plain_attrs = without_shapes["graphCollections"][0]["graphs"][0][
        "groupNodeAttributes"
    ]
    assert "input_shape" not in plain_attrs.get(decoder, {})


def test_group_boundary_shapes_reads_the_crossing_tensor():
    from model_explorer_export.shapes import group_boundary_shapes

    def node(node_id: str, namespace: str, shape: str, sources: list[str]):
        return {
            "id": node_id,
            "namespace": namespace,
            "outputsMetadata": [
                {
                    "id": "0",
                    "attrs": [
                        {"key": "shape", "value": shape},
                        {"key": "dtype", "value": "float16"},
                    ],
                }
            ],
            "incomingEdges": [{"sourceNodeId": source} for source in sources],
        }

    nodes = [
        node("embed", "", "B x S x 8", []),
        node("mlp/@input", "mlp", "B x S x 8", ["embed"]),
        node("mlp/up", "mlp", "B x S x 32", ["mlp/@input"]),
        node("mlp/down", "mlp", "B x S x 8", ["mlp/up"]),
        node("norm", "", "B x S x 8", ["mlp/down"]),
    ]
    assert group_boundary_shapes(nodes) == {
        "mlp": {
            "input_shape": "B x S x 8 float16",
            "output_shape": "B x S x 8 float16",
        },
    }


def test_build_model_explorer_payload_includes_operator_export():
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    payload = build_model_explorer_payload(spec, include_operator_export=True)
    operator_export = payload["tracelensViewer"]["operatorExport"]
    assert operator_export["model_type"] == spec.model_type
    assert operator_export["sections"]


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

    labels = {node["label"].split("\n", 1)[0] for node in graphs[0]["nodes"]}
    assert "Linear" in labels
