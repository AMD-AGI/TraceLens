###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Graph rules on a hand-written model, not an upstream checkpoint.

The fixture under ``tests/fixtures/glm_patterns`` carries the structures these
tests care about: two decoder variants, a gated norm, a forget-style gate, a
hyperconnection, an expert loop, and a vision tower. Lookups use those module
names. Nothing here is a line number from a transformers modeling file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.extract import load_architecture
from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export.merge import build_merged_model_graph
from TraceLens.Visualizer.model_explorer_export.type_check import (
    integrity_check_graph_nodes,
    type_check_graph_nodes,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "glm_patterns"
LINEAR = "1x_PatternLinearAttention_PatternMLP"
FULL = "3x_PatternFullAttention_PatternMoE"


@pytest.fixture(scope="module")
def pattern():
    spec = load_architecture(
        FIXTURE,
        code_path=FIXTURE / "modeling_pattern.py",
        detailed=True,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    return spec, graph


def _nodes(graph, fragment: str) -> list[dict]:
    return [
        node
        for node in graph["nodes"]
        if fragment in (node.get("namespace") or "") or fragment in node["id"]
    ]


def _labels(nodes: list[dict]) -> set[str]:
    return {node.get("label") for node in nodes}


def _sources(node: dict) -> list[str]:
    return [edge["sourceNodeId"] for edge in node.get("incomingEdges") or []]


def _by_id(nodes: list[dict]) -> dict[str, dict]:
    return {node["id"]: node for node in nodes}


def _assert_acyclic(nodes: list[dict]) -> None:
    outgoing: dict[str, list[str]] = {}
    for node in nodes:
        for edge in node.get("incomingEdges") or []:
            source = edge["sourceNodeId"]
            outgoing.setdefault(source, []).append(node["id"])
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise AssertionError(f"cycle reaches {node_id}")
        if node_id in visited:
            return
        visiting.add(node_id)
        for target in outgoing.get(node_id, []):
            visit(target)
        visiting.remove(node_id)
        visited.add(node_id)

    for node in nodes:
        visit(node["id"])


def test_two_decoder_variants_share_one_stack(pattern):
    spec, graph = pattern
    pairs = {
        (variant.attention_class, variant.ffn_class): variant.count
        for variant in spec.layer_variants
    }
    assert pairs == {
        ("PatternLinearAttention", "PatternMLP"): 1,
        ("PatternFullAttention", "PatternMoE"): 3,
    }
    groups = graph["groupNodeAttributes"]
    assert "4x_PatternDecoderLayer" in groups
    assert f"4x_PatternDecoderLayer/{LINEAR}" in groups
    assert f"4x_PatternDecoderLayer/{FULL}" in groups
    assert not _nodes(graph, f"{LINEAR}/PatternMoE")
    assert _nodes(graph, f"{FULL}/PatternMoE")
    assert _nodes(graph, f"{LINEAR}/PatternMLP")


def test_linear_attention_projections_share_the_masked_input(pattern):
    _spec, graph = pattern
    body = _nodes(graph, f"{LINEAR}/PatternLinearAttention")
    by_id = _by_id(graph["nodes"])
    projections = [
        node
        for node in body
        if node.get("label") == "Linear"
        and any(name in node["id"] for name in (":q_proj:", ":k_proj:", ":v_proj:"))
    ]
    assert {node["id"].rsplit(":", 2)[-2] for node in projections} == {
        "q_proj",
        "k_proj",
        "v_proj",
    }
    producers = []
    for node in projections:
        sources = _sources(node)
        assert sources, node["id"]
        producers.append(by_id[sources[0]]["label"])
    assert producers == ["Multiply", "Multiply", "Multiply"]


def test_linear_attention_gate_is_expanded_and_exits_through_one_linear(pattern):
    _spec, graph = pattern
    body = _nodes(graph, f"{LINEAR}/PatternLinearAttention")
    labels = _labels(body)
    assert {"Concat", "Add", "Sigmoid", "Linear"} <= labels
    concat = next(node for node in body if node.get("label") == "Concat")
    assert len(_sources(concat)) >= 2
    assert any(":f_a_proj:" in node["id"] for node in body)
    assert any(":f_b_proj:" in node["id"] for node in body)
    norm = [
        node for node in body if "PatternGatedNorm" in (node.get("namespace") or "")
    ]
    assert {"Power", "Mean", "Reciprocal sqrt", "Sigmoid"} <= _labels(norm)
    exits = [node for node in body if node["id"].endswith("/self_attn/@output")]
    assert exits
    assert all(node["label"] == "out_proj" for node in exits)


def test_hyperconnection_keeps_the_activation_and_the_residual(pattern):
    _spec, graph = pattern
    block = _nodes(graph, f"{LINEAR}/attn_hc")
    assert any(
        ":mix:" in node["id"] and node.get("label") == "Linear" for node in block
    )
    assert any(node.get("label") == "Add" for node in block)
    assert any(node.get("label") == "residual" for node in block)


def test_expert_loop_stays_inside_the_moe_variant(pattern):
    _spec, graph = pattern
    moe = _nodes(graph, f"{FULL}/PatternMoE")
    assert any("@body_in:" in node["id"] for node in moe)
    assert any("@body_out:" in node["id"] for node in moe)
    assert any("PatternExperts" in (node.get("namespace") or "") for node in moe)
    carried = [
        node["id"]
        for node in graph["nodes"]
        if "@loop_carried" in node["id"] and LINEAR in node["id"]
    ]
    assert carried == []


def test_full_attention_norms_are_separate_and_feed_one_kernel(pattern):
    _spec, graph = pattern
    query_norm = _nodes(graph, f"{FULL}/PatternFullAttention/q_norm")
    key_norm = _nodes(graph, f"{FULL}/PatternFullAttention/k_norm")
    assert query_norm and key_norm
    assert _labels(query_norm) & {"Power", "Mean", "Reciprocal sqrt"}
    assert {node.get("namespace") for node in query_norm}.isdisjoint(
        {node.get("namespace") for node in key_norm}
    )
    kernel = [
        node
        for node in _nodes(graph, f"{FULL}/PatternFullAttention")
        if node.get("label") == "pattern_sdpa"
    ]
    assert kernel
    ports = {
        edge["sourceNodeId"].rsplit(":", 1)[-1]
        for node in kernel
        for edge in node["incomingEdges"]
    }
    assert {"query", "key", "value"} <= ports


def test_vision_tower_wires_qkv_rotary_and_a_counted_loop(pattern):
    _spec, graph = pattern
    visual = _nodes(graph, "visual")
    labels = _labels(visual)
    assert "Unbind" in labels
    slices = [node.get("label") for node in visual if "@slice_out:" in node["id"]]
    assert slices == ["query", "key", "value"]
    query_ns = {
        node.get("namespace")
        for node in visual
        if node.get("namespace", "").endswith("/q_norm")
    }
    key_ns = {
        node.get("namespace")
        for node in visual
        if node.get("namespace", "").endswith("/k_norm")
    }
    assert query_ns and key_ns and query_ns.isdisjoint(key_ns)
    assert {"Cosine", "Sine"} <= labels
    assert any(":gate_proj:" in node["id"] for node in visual)
    assert any(":up_proj:" in node["id"] for node in visual)
    assert any(node.get("label") == "Cumulative sum" for node in visual)
    assert any(
        "@body_in:" in node["id"] and node.get("label", "").startswith("loop in")
        for node in visual
    )
    assert any("iterations: 2" in (node.get("label") or "") for node in visual)


def test_spine_norms_do_not_share_a_namespace(pattern):
    _spec, graph = pattern
    groups = [
        key
        for key in graph["groupNodeAttributes"]
        if key.endswith("/input_layernorm") or key.endswith("/post_attention_layernorm")
    ]
    assert {key.rsplit("/", 1)[-1] for key in groups} == {
        "input_layernorm",
        "post_attention_layernorm",
    }


def test_export_is_acyclic_and_clean(pattern):
    _spec, graph = pattern
    nodes = graph["nodes"]
    _assert_acyclic(nodes)
    assert integrity_check_graph_nodes(nodes) == []
    assert type_check_graph_nodes(nodes) == []
