"""Tests for the PyTorch-based model graph builder."""

from __future__ import annotations

import json

import pytest
import torch

from TraceLens.ModelUtils.torch_trace import (
    _classify_module,
    _fx_trace_module,
    _instantiate_meta,
    _patch_config,
    _symbolise,
    build_graph,
)


# ── Unit tests (no network) ─────────────────────────────────────────────────


class TestSymbolise:
    def test_basic(self):
        assert _symbolise((1, 128, 4096), batch_size=1, seq_len=128) == "B x S x 4096"

    def test_no_match(self):
        assert _symbolise((2, 256, 1024), batch_size=1, seq_len=128) == "2 x 256 x 1024"

    def test_scalar(self):
        assert _symbolise((1,), batch_size=1, seq_len=128) == "B"


class TestClassifyModule:
    def test_linear(self):
        assert _classify_module(torch.nn.Linear(10, 20)) == "linear"

    def test_embedding(self):
        assert _classify_module(torch.nn.Embedding(100, 32)) == "embedding"

    def test_layernorm(self):
        assert _classify_module(torch.nn.LayerNorm(64)) == "norm"

    def test_relu(self):
        assert _classify_module(torch.nn.ReLU()) == "activation"

    def test_silu(self):
        assert _classify_module(torch.nn.SiLU()) == "activation"

    def test_default(self):
        assert _classify_module(torch.nn.Dropout()) == "default"


class TestFxTrace:
    def test_simple_mlp(self):
        """torch.fx should successfully trace a simple MLP."""
        mlp = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
        )
        graph = _fx_trace_module(mlp)
        assert graph is not None
        node_ops = [n.op for n in graph.nodes]
        assert "call_module" in node_ops

    def test_control_flow_fails(self):
        """Modules with control flow should return None."""
        class ConditionalModule(torch.nn.Module):
            def forward(self, x):
                if x.sum() > 0:
                    return x * 2
                return x * 3

        graph = _fx_trace_module(ConditionalModule())
        assert graph is None


# ── Integration tests (require HF Hub) ──────────────────────────────────────


@pytest.mark.network
class TestBuildGraphGLM:
    """Integration tests against THUDM/glm-4-9b-chat."""

    @pytest.fixture(scope="class")
    def glm_payload(self):
        return build_graph("THUDM/glm-4-9b-chat", seq_len=128, batch_size=1)

    def test_payload_structure(self, glm_payload):
        assert "name" in glm_payload
        assert "graphCollections" in glm_payload
        graphs = glm_payload["graphCollections"][0]["graphs"]
        assert len(graphs) == 1

    def test_has_nodes(self, glm_payload):
        nodes = glm_payload["graphCollections"][0]["graphs"][0]["nodes"]
        assert len(nodes) >= 10

    def test_input_node(self, glm_payload):
        nodes = glm_payload["graphCollections"][0]["graphs"][0]["nodes"]
        input_nodes = [n for n in nodes if n["id"] == "@input"]
        assert len(input_nodes) == 1

    def test_embedding_shape(self, glm_payload):
        nodes = glm_payload["graphCollections"][0]["graphs"][0]["nodes"]
        emb = [n for n in nodes if "word_embeddings" in n["id"]]
        assert len(emb) == 1
        shape_attr = [a for a in emb[0]["attrs"] if a["key"] == "output_shape"]
        assert shape_attr
        assert "4096" in shape_attr[0]["value"]

    def test_qkv_shape_multi_query(self, glm_payload):
        """GLM uses multi-query attention: QKV should be 4608, not 12288."""
        nodes = glm_payload["graphCollections"][0]["graphs"][0]["nodes"]
        qkv = [n for n in nodes if "query_key_value" in n["id"]]
        assert len(qkv) == 1
        shape_attr = [a for a in qkv[0]["attrs"] if a["key"] == "output_shape"]
        assert shape_attr
        assert "4608" in shape_attr[0]["value"]

    def test_layer_deduplication(self, glm_payload):
        """40 GLMBlock layers should be deduplicated."""
        nodes = glm_payload["graphCollections"][0]["graphs"][0]["nodes"]
        layer_nodes = [n for n in nodes if "/layers/" in n["id"]]
        # All should be layer 0 (others are deduplicated)
        for n in layer_nodes:
            assert "/layers/0/" in n["id"]

    def test_all_nodes_have_edges(self, glm_payload):
        nodes = glm_payload["graphCollections"][0]["graphs"][0]["nodes"]
        for n in nodes:
            if n["id"] == "@input":
                continue
            assert "incomingEdges" in n, f"Node {n['id']} has no edges"

    def test_mlp_tensor_ops(self, glm_payload):
        """MLP should have torch.fx tensor ops (chunk, silu, mul)."""
        nodes = glm_payload["graphCollections"][0]["graphs"][0]["nodes"]
        op_labels = {n["label"] for n in nodes}
        assert "Chunk" in op_labels
        assert "SiLU" in op_labels
        assert "Multiply" in op_labels

    def test_fact_sheet(self, glm_payload):
        viewer = glm_payload["tracelensViewer"]
        assert "factSheet" in viewer
        assert "hidden_size" in viewer["factSheet"]

    def test_serializable(self, glm_payload):
        """Payload must be JSON-serializable."""
        json.dumps(glm_payload)
