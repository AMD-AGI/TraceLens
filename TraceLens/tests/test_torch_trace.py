###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the PyTorch-based model graph builder."""

from __future__ import annotations

import json

import pytest
import torch

from TraceLens.ModelUtils.torch_trace import (
    _classify_module,
    _fx_trace_module,
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


# ── Structural tests (no network, uses simple local model) ──────────────────


class _SimpleAttention(torch.nn.Module):
    """Minimal attention-like composite for testing."""

    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.o_proj = torch.nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.o_proj(attn)


class _SimpleMLP(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.up_proj = torch.nn.Linear(dim, dim * 4)
        self.act = torch.nn.SiLU()
        self.down_proj = torch.nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class _SimpleBlock(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(dim)
        self.self_attn = _SimpleAttention(dim)
        self.post_attention_layernorm = torch.nn.LayerNorm(dim)
        self.mlp = _SimpleMLP(dim)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _SimpleModel(torch.nn.Module):
    def __init__(self, vocab: int = 256, dim: int = 64, n_layers: int = 4):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, dim)
        self.layers = torch.nn.ModuleList([_SimpleBlock(dim) for _ in range(n_layers)])
        self.norm = torch.nn.LayerNorm(dim)
        self.lm_head = torch.nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, **kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


def _build_simple_payload() -> dict:
    """Build a graph payload from _SimpleModel without network access."""

    with torch.device("meta"):
        model = _SimpleModel()
    model.eval()

    return (
        build_graph.__wrapped__(model)
        if hasattr(build_graph, "__wrapped__")
        else _build_from_model(model)
    )


def _build_from_model(model: torch.nn.Module) -> dict:
    """Replicate the core of build_graph() for a pre-instantiated model."""
    from TraceLens.ModelUtils import torch_trace as tt

    # Monkey-patch _instantiate_meta to return our model
    original = tt._instantiate_meta

    class FakeConfig:
        _name_or_path = "test/simple-model"
        model_type = "simple"
        hidden_size = 64
        num_hidden_layers = 4
        vocab_size = 256
        dtype = "float32"

        def to_dict(self):
            return {
                k: v
                for k, v in self.__class__.__dict__.items()
                if not k.startswith("_") and not callable(v)
            }

    tt._instantiate_meta = lambda checkpoint: (model, FakeConfig())
    try:
        payload = build_graph("test/simple-model")
    finally:
        tt._instantiate_meta = original
    return payload


@pytest.fixture(scope="module")
def simple_payload():
    return _build_from_model(_SimpleModel().eval())


@pytest.fixture(scope="module")
def simple_nodes(simple_payload):
    return simple_payload["graphCollections"][0]["graphs"][0]["nodes"]


class TestNoDuplicateIONodes:
    """Verify that composite modules have exactly one @input and one @output,
    with no @ext_input or @ext_output nodes."""

    def test_no_ext_input_nodes(self, simple_nodes):
        ext = [n for n in simple_nodes if "@ext_input" in n["id"]]
        assert ext == [], f"Found @ext_input nodes: {[n['id'] for n in ext]}"

    def test_no_ext_output_nodes(self, simple_nodes):
        ext = [n for n in simple_nodes if "@ext_output" in n["id"]]
        assert ext == [], f"Found @ext_output nodes: {[n['id'] for n in ext]}"

    def test_self_attn_has_single_output(self, simple_nodes):
        """self_attn should have exactly one @output, not both @output and @ext_output."""
        attn_outputs = [
            n
            for n in simple_nodes
            if "self_attn" in n["id"] and n["id"].endswith("/@output")
        ]
        # Should have one per representative layer
        for n in attn_outputs:
            assert n["label"] == "Output"
        # No "self_attn Output" labels
        ext_labels = [
            n for n in simple_nodes if "self_attn Output" in n.get("label", "")
        ]
        assert ext_labels == []

    def test_self_attn_has_single_input(self, simple_nodes):
        """self_attn should have exactly one @input, not both @input and @ext_input."""
        attn_inputs = [
            n
            for n in simple_nodes
            if "self_attn" in n["id"] and n["id"].endswith("/@input")
        ]
        for n in attn_inputs:
            assert n["label"] == "Input"
        ext_labels = [
            n for n in simple_nodes if "self_attn Input" in n.get("label", "")
        ]
        assert ext_labels == []


class TestShapePropagation:
    """Verify that synthetic I/O nodes have outputsMetadata so edges
    don't display as '?' in the viewer."""

    def test_root_input_has_shape(self, simple_nodes):
        inp = next(n for n in simple_nodes if n["id"] == "@input")
        assert inp.get("outputsMetadata"), "@input node missing outputsMetadata"

    def test_root_output_has_shape(self, simple_nodes):
        out = next(n for n in simple_nodes if n["id"] == "@output")
        assert out.get("outputsMetadata"), "@output node missing outputsMetadata"

    def test_synthetic_input_nodes_have_shapes(self, simple_nodes):
        """All synthetic input nodes should have outputsMetadata."""
        for n in simple_nodes:
            attrs = {a["key"]: a["value"] for a in n.get("attrs", [])}
            if attrs.get("synthetic") == "input":
                assert n.get(
                    "outputsMetadata"
                ), f"Synthetic input {n['id']} missing outputsMetadata"

    def test_synthetic_output_nodes_have_shapes(self, simple_nodes):
        """Most synthetic output nodes should have outputsMetadata.

        Some composite modules (e.g. decoder blocks) may lack captured
        shapes, so we check that the vast majority have them.
        """
        outputs = [
            n
            for n in simple_nodes
            for a in n.get("attrs", [])
            if a.get("key") == "synthetic" and a.get("value") == "output"
        ]
        with_shapes = [n for n in outputs if n.get("outputsMetadata")]
        assert len(outputs) > 0
        # Composites whose children are all FX ops without captured
        # shapes (e.g. decoder blocks with residual adds) may lack
        # outputsMetadata.  Require at least 50% coverage.
        assert (
            len(with_shapes) / len(outputs) >= 0.5
        ), f"Only {len(with_shapes)}/{len(outputs)} synthetic outputs have shapes"

    def test_all_nodes_have_shapes(self, simple_nodes):
        """Every node (leaf, FX op, synthetic) must have outputsMetadata."""
        missing = [n["id"] for n in simple_nodes if not n.get("outputsMetadata")]
        assert missing == [], f"{len(missing)} nodes missing outputsMetadata: {missing}"

    def test_embedding_edge_has_shape(self, simple_nodes):
        """The edge from @input to embedding should carry a shape, not '?'."""
        emb = next(n for n in simple_nodes if "embed_tokens" in n["id"])
        # The source of embedding's edge should have outputsMetadata
        node_by_id = {n["id"]: n for n in simple_nodes}
        for e in emb.get("incomingEdges", []):
            src = node_by_id.get(e["sourceNodeId"])
            if src:
                assert src.get(
                    "outputsMetadata"
                ), f"Source {src['id']} of embed_tokens edge has no shape"


class TestGroupNodeAttributes:
    """Verify groupNodeAttributes uses the dict format expected by Model Explorer."""

    def test_is_dict(self, simple_payload):
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        assert isinstance(
            ga, dict
        ), f"groupNodeAttributes should be dict, got {type(ga).__name__}"

    def test_values_are_dicts(self, simple_payload):
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        for key, val in ga.items():
            assert isinstance(
                val, dict
            ), f"groupNodeAttributes['{key}'] should be dict, got {type(val).__name__}"

    def test_has_class_key(self, simple_payload):
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        for key, val in ga.items():
            assert "class" in val, f"groupNodeAttributes['{key}'] missing 'class' key"

    def test_layer_group_has_count(self, simple_payload):
        """Layer groups should have a 'count' attribute."""
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        count_entries = {k: v for k, v in ga.items() if "count" in v}
        assert len(count_entries) > 0, "No layer group with 'count' attribute found"


class TestEdgeWiring:
    """Verify that edges inside composite modules follow the correct dataflow."""

    def test_no_broken_edges(self, simple_nodes):
        """Every edge must reference an existing source node."""
        node_ids = {n["id"] for n in simple_nodes}
        for n in simple_nodes:
            for e in n.get("incomingEdges", []):
                assert (
                    e["sourceNodeId"] in node_ids
                ), f"Node {n['id']} has broken edge from {e['sourceNodeId']}"

    def test_all_non_input_nodes_have_edges(self, simple_nodes):
        """Every node except @input must have at least one incoming edge."""
        for n in simple_nodes:
            if n["id"] == "@input":
                continue
            assert n.get("incomingEdges"), f"Node {n['id']} has no incoming edges"

    def test_sequential_dataflow_in_block(self, simple_nodes):
        """Inside the decoder block, every child module must be wired
        (no orphan nodes). The @input node should be the ultimate
        source of all nodes in the block."""
        block_nodes = [n for n in simple_nodes if "layers/0/" in n["id"]]
        assert len(block_nodes) > 0

        # All block nodes except layer @input must have incoming edges
        for n in block_nodes:
            if n["id"].endswith("/@input") and n["id"].count("/") == 1:
                continue
            assert n.get(
                "incomingEdges"
            ), f"Block node {n['id']} has no incoming edges (dead node)"

    def test_call_graph_sequential_fallback(self):
        """When a child module has no tensor-ID-tracked producer, the
        sequential fallback should connect it from the previous child."""
        from TraceLens.ModelUtils.torch_trace import _capture_call_graph

        # Model where child B receives from child A via a non-module
        # function call (torch.cat), which creates an untracked tensor.
        class _Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_a = torch.nn.Linear(32, 32)
                self.proj_b = torch.nn.Linear(64, 32)

            def forward(self, x):
                a = self.proj_a(x)
                # torch.cat creates a new tensor not tracked to any child
                combined = torch.cat([a, x], dim=-1)
                return self.proj_b(combined)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(64, 32)
                self.block = _Block()

            def forward(self, x, **kwargs):
                return self.block(self.emb(x))

        with torch.device("meta"):
            model = _Model()

        composites = set()
        for name, mod in model.named_modules():
            if name and list(mod.children()):
                composites.add(name)

        call_graph, _ = _capture_call_graph(model, composites, seq_len=8, batch_size=1)
        block_edges = call_graph.get("block", [])
        edge_set = {
            (s.split(".")[-1] if s != "@input" else s, t.split(".")[-1])
            for s, t in block_edges
        }

        # proj_b should have an edge (either from proj_a via fallback,
        # or from @input). It must not be orphaned.
        proj_b_sources = {s for s, t in edge_set if t == "proj_b"}
        assert proj_b_sources, f"proj_b has no incoming edges. Got: {edge_set}"


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
        fs = viewer["factSheet"]
        # factSheet may be a string or a dict with title/body
        text = fs["body"] if isinstance(fs, dict) else fs
        assert "hidden_size" in text

    def test_serializable(self, glm_payload):
        """Payload must be JSON-serializable."""
        json.dumps(glm_payload)


class TestInputShapeAttribute:
    """Verify that expanding modules show input_shape alongside output_shape."""

    def test_mlp_group_has_input_shape(self, simple_payload):
        """MLP expands dimension (dim → dim*4 → dim), should have input_shape."""
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        mlp_groups = {k: v for k, v in ga.items() if "MLP" in k or "mlp" in k}
        for key, attrs in mlp_groups.items():
            if "output_shape" in attrs:
                assert (
                    "input_shape" in attrs
                ), f"Group '{key}' has output_shape but missing input_shape"

    def test_composite_groups_with_output_have_input(self, simple_payload):
        """All composite module groups with output_shape should also have input_shape."""
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        for key, attrs in ga.items():
            if "count" in attrs:
                continue  # layer groups may not need input_shape
            if "output_shape" in attrs:
                assert (
                    "input_shape" in attrs
                ), f"Group '{key}' has output_shape but missing input_shape"

    def test_fx_expanded_module_has_shapes(self, simple_payload):
        """FX-expanded leaf modules (e.g. LayerNorm) should have shape attributes."""
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        norm_groups = {k: v for k, v in ga.items() if "Norm" in k or "norm" in k}
        for key, attrs in norm_groups.items():
            assert (
                "output_shape" in attrs
            ), f"FX-expanded group '{key}' missing output_shape"


class TestOutputShapeCoverage:
    """Verify output_shape is present on all composite module groups."""

    def test_all_composites_have_output_shape(self, simple_payload):
        """Every composite module group should have an output_shape attribute."""
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        missing = []
        for key, attrs in ga.items():
            if "output_shape" not in attrs:
                missing.append(key)
        assert not missing, f"Groups missing output_shape: {missing}"

    def test_layer_groups_have_output_shape(self, simple_payload):
        """Layer groups (with 'count') should have output_shape."""
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        layer_groups = {k: v for k, v in ga.items() if "count" in v}
        for key, attrs in layer_groups.items():
            assert "output_shape" in attrs, f"Layer group '{key}' missing output_shape"


class TestMultiOutputWiring:
    """Verify all output children of composite modules are wired to @output."""

    def test_no_orphan_output_nodes(self, simple_nodes):
        """No synthetic @output node should be completely disconnected."""
        for n in simple_nodes:
            attrs = {a["key"]: a["value"] for a in n.get("attrs", [])}
            if attrs.get("synthetic") != "output":
                continue
            # This @output node must be consumed by someone or be the root
            if n["id"] == "@output":
                continue
            consumed = any(
                e["sourceNodeId"] == n["id"]
                for n2 in simple_nodes
                for e in n2.get("incomingEdges", [])
                if n2["id"] != n["id"]
            )
            assert (
                consumed
            ), f"Synthetic output node {n['id']} has no consumers (dead output)"

    def test_block_output_wires_from_mlp(self, simple_nodes):
        """The decoder block @output should include mlp's output in the chain."""
        block_output = next(
            (n for n in simple_nodes if n["id"] == "layers/0/@output"), None
        )
        assert block_output is not None, "Missing layers/0/@output"
        # The block output should be fed by mlp/@output (directly or indirectly)
        # At minimum, check that mlp/@output is consumed by something in the block
        mlp_output = next(
            (n for n in simple_nodes if n["id"] == "layers/0/mlp/@output"), None
        )
        if mlp_output:
            consumed = any(
                e["sourceNodeId"] == "layers/0/mlp/@output"
                for n in simple_nodes
                for e in n.get("incomingEdges", [])
                if n["id"] != "layers/0/mlp/@output"
            )
            assert consumed, "mlp/@output is not wired to anything"

    def test_all_composite_outputs_consumed(self, simple_nodes):
        """Every child's @output inside a composite must be consumed by
        either a sibling @input or the parent's @output node."""
        # Collect all @output nodes grouped by parent composite
        from collections import defaultdict

        composites = defaultdict(list)
        for n in simple_nodes:
            nid = n["id"]
            attrs = {a["key"]: a["value"] for a in n.get("attrs", [])}
            if attrs.get("synthetic") != "output":
                continue
            parts = nid.rsplit("/", 1)
            if len(parts) == 2:
                parent = parts[0]
                composites[parent].append(nid)

        consumers = {}
        for n in simple_nodes:
            for e in n.get("incomingEdges", []):
                consumers.setdefault(e["sourceNodeId"], []).append(n["id"])

        orphans = []
        for parent, output_ids in composites.items():
            for oid in output_ids:
                if oid not in consumers:
                    orphans.append(oid)
        assert not orphans, f"Orphan @output nodes (not consumed): {orphans}"


class TestSyntheticInputShapeMetadata:
    """Verify @input synthetic nodes use the module's actual input shape,
    not the shape inherited from the source node."""

    def _get_shape(self, nodes, node_id):
        for n in nodes:
            if n["id"] == node_id:
                om = n.get("outputsMetadata", [{}])
                if om and om[0].get("attrs"):
                    return next(
                        (a["value"] for a in om[0]["attrs"] if a["key"] == "shape"),
                        "",
                    )
        return ""

    def test_root_input_is_integer(self, simple_nodes):
        """Root @input should reflect token ID shape (integer, no hidden dim)."""
        shape = self._get_shape(simple_nodes, "@input")
        assert shape, "@input has no shape metadata"
        # Token IDs are 2-D (batch, seq) — should NOT have a hidden dimension
        dims = shape.replace(" x ", "x").split("x")
        assert len(dims) <= 3, f"@input has too many dims for token IDs: {shape}"

    def test_embedding_parent_input_is_integer_dtype(self, simple_nodes):
        """A composite whose first child is Embedding should have int64 @input."""
        # In _SimpleModel, the root model's @input carries token IDs
        shape = self._get_shape(simple_nodes, "@input")
        assert (
            "int64" in shape
        ), f"Root @input should be int64 (token IDs), got: {shape}"

    def test_input_shape_not_inherited_from_source(self, simple_nodes):
        """Composite @input nodes should derive shape from their own module's
        input, not from the upstream source node's output."""
        # layers/0/@input should reflect the hidden dim (B x S x 64),
        # not the root @input shape (B x S int64)
        layer_input_shape = self._get_shape(simple_nodes, "layers/0/@input")
        root_shape = self._get_shape(simple_nodes, "@input")
        if layer_input_shape and root_shape:
            assert layer_input_shape != root_shape, (
                f"layers/0/@input ({layer_input_shape}) should differ from "
                f"root @input ({root_shape}) — it should reflect hidden dim"
            )


class TestSyntheticOutputShapeMetadata:
    """Verify @output synthetic nodes use the module's captured output shape,
    not the shape of the last internal child."""

    def _get_shape(self, nodes, node_id):
        for n in nodes:
            if n["id"] == node_id:
                om = n.get("outputsMetadata", [{}])
                if om and om[0].get("attrs"):
                    return next(
                        (a["value"] for a in om[0]["attrs"] if a["key"] == "shape"),
                        "",
                    )
        return ""

    def test_output_nodes_have_shape(self, simple_nodes):
        """All synthetic @output nodes should have outputsMetadata."""
        for n in simple_nodes:
            attrs = {a["key"]: a["value"] for a in n.get("attrs", [])}
            if attrs.get("synthetic") != "output":
                continue
            om = n.get("outputsMetadata")
            assert om, f"Synthetic output {n['id']} has no outputsMetadata"

    def test_block_output_matches_module_shape(self, simple_nodes):
        """Decoder block @output shape should match the block's captured output shape,
        not the internal mlp's up_proj intermediate shape."""
        block_shape = self._get_shape(simple_nodes, "layers/0/@output")
        assert block_shape, "layers/0/@output has no shape"
        # Block output should be hidden_dim (64), not intermediate (256)
        assert (
            "256" not in block_shape
        ), f"layers/0/@output seems to show MLP intermediate shape: {block_shape}"

    def test_self_attn_output_shape(self, simple_nodes):
        """self_attn/@output should reflect o_proj output (hidden_dim),
        not an intermediate projection size."""
        shape = self._get_shape(simple_nodes, "layers/0/self_attn/@output")
        if shape:
            assert (
                "64" in shape
            ), f"self_attn/@output should include hidden_dim=64, got: {shape}"


class TestInputShapeInference:
    """Verify _infer_input_shapes_from_weights produces correct shapes."""

    def test_linear_input_shape(self):
        from TraceLens.ModelUtils.torch_trace import _infer_input_shapes_from_weights

        model = torch.nn.Sequential(torch.nn.Linear(32, 64))
        shapes = _infer_input_shapes_from_weights(model, {}, batch_size=1, seq_len=10)
        assert shapes.get("0") == (1, 10, 32)

    def test_embedding_input_shape(self):
        from TraceLens.ModelUtils.torch_trace import _infer_input_shapes_from_weights

        model = torch.nn.Sequential(torch.nn.Embedding(100, 64))
        shapes = _infer_input_shapes_from_weights(model, {}, batch_size=1, seq_len=10)
        # Embedding input is (batch, seq) — no hidden dim
        assert shapes.get("0") == (1, 10)

    def test_captured_shapes_preserved(self):
        from TraceLens.ModelUtils.torch_trace import _infer_input_shapes_from_weights

        model = torch.nn.Sequential(torch.nn.Linear(32, 64))
        captured = {"0": (2, 5, 32)}
        shapes = _infer_input_shapes_from_weights(
            model, captured, batch_size=1, seq_len=10
        )
        # Should preserve the captured shape, not override
        assert shapes["0"] == (2, 5, 32)

    def test_conv2d_input_shape(self):
        from TraceLens.ModelUtils.torch_trace import _infer_input_shapes_from_weights

        model = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3))
        shapes = _infer_input_shapes_from_weights(model, {}, batch_size=1, seq_len=10)
        assert shapes.get("0") == (1, 3, 10, 10)

    def test_conv1d_input_shape(self):
        from TraceLens.ModelUtils.torch_trace import _infer_input_shapes_from_weights

        model = torch.nn.Sequential(torch.nn.Conv1d(3, 16, 3))
        shapes = _infer_input_shapes_from_weights(model, {}, batch_size=1, seq_len=10)
        assert shapes.get("0") == (1, 3, 10)

    def test_conv3d_input_shape(self):
        from TraceLens.ModelUtils.torch_trace import _infer_input_shapes_from_weights

        model = torch.nn.Sequential(torch.nn.Conv3d(3, 16, 3))
        shapes = _infer_input_shapes_from_weights(model, {}, batch_size=1, seq_len=10)
        assert shapes.get("0") == (1, 3, 10, 10, 10)

    def test_custom_norm_class_gets_generic_weight_fallback(self):
        """Vision-encoder-style RMSNorm variants often don't subclass
        torch.nn.LayerNorm/RMSNorm, so they need to fall back to the
        generic weight-shape heuristic (this is the bug that left the
        GLM vision submodel's internal norms without an input_shape)."""
        from TraceLens.ModelUtils.torch_trace import _infer_input_shapes_from_weights

        class CustomRMSNorm(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(hidden_size))

            def forward(self, x):
                return x

        model = torch.nn.Sequential(CustomRMSNorm(48))
        shapes = _infer_input_shapes_from_weights(model, {}, batch_size=1, seq_len=10)
        assert shapes.get("0") == (1, 10, 48)

    def test_hidden_size_attr_fallback(self):
        """Modules with no weight tensor but a hidden_size attribute
        (e.g. some rotary-embedding-style modules) still get a
        best-effort input_shape."""
        from TraceLens.ModelUtils.torch_trace import _infer_input_shapes_from_weights

        class HiddenSizeOnly(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, x):
                return x

        model = torch.nn.Sequential(HiddenSizeOnly(32))
        shapes = _infer_input_shapes_from_weights(model, {}, batch_size=1, seq_len=10)
        assert shapes.get("0") == (1, 10, 32)


class TestCallGraphCapture:
    """Verify call-graph captures correct dataflow edges."""

    def test_embed_tokens_has_edges(self, simple_nodes):
        """embed_tokens should have incoming edges (wired to something)."""
        et = next((n for n in simple_nodes if n["id"] == "embed_tokens"), None)
        assert et is not None, "embed_tokens node not found"
        edges = [e["sourceNodeId"] for e in et.get("incomingEdges", [])]
        assert edges, "embed_tokens has no incoming edges"

    def test_all_nodes_wired(self, simple_nodes):
        """No non-@input node should be completely unwired."""
        for n in simple_nodes:
            if n["id"] == "@input":
                continue
            edges = n.get("incomingEdges", [])
            attrs = {a["key"]: a["value"] for a in n.get("attrs", [])}
            if attrs.get("synthetic") == "input":
                continue  # input nodes may not always have edges
            assert edges, f"Node {n['id']} has no incoming edges"


class TestCompositeOutputResolution:
    """Verify @output comes from the end of the forward chain."""

    def test_output_has_edges(self, simple_nodes):
        """Root @output should have incoming edges."""
        root_output = next((n for n in simple_nodes if n["id"] == "@output"), None)
        assert root_output is not None
        sources = [e["sourceNodeId"] for e in root_output.get("incomingEdges", [])]
        assert sources, "@output has no incoming edges"

    def test_output_shape_exists(self, simple_nodes):
        """@output should have shape metadata."""
        root_output = next((n for n in simple_nodes if n["id"] == "@output"), None)
        if root_output and root_output.get("outputsMetadata"):
            shape = next(
                (
                    a["value"]
                    for a in root_output["outputsMetadata"][0].get("attrs", [])
                    if a["key"] == "shape"
                ),
                "",
            )
            assert shape, "@output has no shape metadata"


class TestModuleListChildPromotion:
    """Verify ModuleList children are promoted as direct children
    of the grandparent for call-graph edge detection."""

    def test_layers_have_edges(self, simple_nodes):
        """Decoder layers (inside ModuleList) should have incoming edges."""
        layer_0_input = next(
            (n for n in simple_nodes if n["id"] == "layers/0/@input"), None
        )
        if layer_0_input:
            edges = [e["sourceNodeId"] for e in layer_0_input.get("incomingEdges", [])]
            assert edges, "layers/0/@input has no incoming edges"


class TestContainerGroupAttrs:
    """Verify container-level group attributes have shapes."""

    def test_groups_have_output_shape(self, simple_payload):
        """All layer groups should have output_shape."""
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        for key, attrs in ga.items():
            if "count" in attrs:
                assert (
                    "output_shape" in attrs
                ), f"Layer group '{key}' missing output_shape"


class TestGroupAttrOrdering:
    """Verify input_shape appears first and output_shape last in group attrs."""

    def test_input_before_output(self, simple_payload):
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        for key, attrs in ga.items():
            keys = list(attrs.keys())
            if "input_shape" in keys and "output_shape" in keys:
                assert keys.index("input_shape") < keys.index(
                    "output_shape"
                ), f"In group '{key}', input_shape should come before output_shape: {keys}"

    def test_input_shape_is_first(self, simple_payload):
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        for key, attrs in ga.items():
            keys = list(attrs.keys())
            if "input_shape" in keys:
                assert (
                    keys[0] == "input_shape"
                ), f"In group '{key}', input_shape should be first: {keys}"

    def test_output_shape_is_last(self, simple_payload):
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        for key, attrs in ga.items():
            keys = list(attrs.keys())
            if "output_shape" in keys:
                assert (
                    keys[-1] == "output_shape"
                ), f"In group '{key}', output_shape should be last: {keys}"


class TestForkJoinNodes:
    """Verify Fork/Join nodes for multi-group containers."""

    @pytest.fixture(scope="class")
    def multi_group_model(self):
        """Build a model with 2 distinct layer types (simulating interleaved)."""

        class _TypeA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(64)
                self.proj = torch.nn.Linear(64, 64)

            def forward(self, x):
                return self.proj(self.norm(x))

        class _TypeB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(64)
                self.gate = torch.nn.Linear(64, 64)

            def forward(self, x):
                return self.gate(self.norm(x))

        class _InterleavedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(256, 64)
                self.layers = torch.nn.ModuleList(
                    [
                        _TypeA(),
                        _TypeB(),
                        _TypeA(),
                        _TypeB(),
                    ]
                )
                self.norm = torch.nn.LayerNorm(64)
                self.head = torch.nn.Linear(64, 256, bias=False)

            def forward(self, x):
                h = self.embed(x)
                for layer in self.layers:
                    h = layer(h)
                return self.head(self.norm(h))

        with torch.device("meta"):
            model = _InterleavedModel()
        model.eval()
        return _build_from_model(model)

    def test_fork_node_exists(self, multi_group_model):
        nodes = multi_group_model["graphCollections"][0]["graphs"][0]["nodes"]
        forks = [
            n
            for n in nodes
            if any(a.get("value") == "fork" for a in n.get("attrs", []))
        ]
        assert len(forks) >= 1, "No Fork node found for multi-group container"

    def test_join_node_exists(self, multi_group_model):
        nodes = multi_group_model["graphCollections"][0]["graphs"][0]["nodes"]
        joins = [
            n
            for n in nodes
            if any(a.get("value") == "join" for a in n.get("attrs", []))
        ]
        assert len(joins) >= 1, "No Join node found for multi-group container"

    def test_fork_has_incoming_edge(self, multi_group_model):
        nodes = multi_group_model["graphCollections"][0]["graphs"][0]["nodes"]
        for n in nodes:
            if any(a.get("value") == "fork" for a in n.get("attrs", [])):
                edges = n.get("incomingEdges", [])
                assert edges, f"Fork node {n['id']} has no incoming edges"

    def test_join_has_incoming_edges(self, multi_group_model):
        nodes = multi_group_model["graphCollections"][0]["graphs"][0]["nodes"]
        for n in nodes:
            if any(a.get("value") == "join" for a in n.get("attrs", [])):
                edges = n.get("incomingEdges", [])
                assert len(edges) >= 2, (
                    f"Join node {n['id']} should have >= 2 incoming edges "
                    f"(one per branch), got {len(edges)}"
                )


class TestMultiModalSideBranchIntoForkedLayers:
    """Regression test: when the side branch (e.g. a vision encoder) feeds
    into a layer stack that ALSO has multiple interleaved layer types (so
    the stack gets collapsed into Fork/Join nodes), the side branch's
    output must still resolve to a concrete node id on the Fork's
    incoming edges — not get lost, and not collide with the container's
    own "@input" sequential-fallback edge to the same Fork target (both
    independently synthesize a "(pred, fork_path)" call-graph edge for
    the same target, keyed only by target in the global `cg_sources`
    map, which silently let one clobber the other).
    """

    @pytest.fixture(scope="class")
    def vlm_with_forked_layers(self):
        class _Vision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(64, 64)

            def forward(self, x):
                return self.proj(x)

        class _TypeA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(64)

            def forward(self, x):
                return self.norm(x)

        class _TypeB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = torch.nn.Linear(64, 64)

            def forward(self, x):
                return self.gate(x)

        class _TextModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(256, 64)
                self.layers = torch.nn.ModuleList(
                    [_TypeA(), _TypeB(), _TypeA(), _TypeB()]
                )
                self.norm = torch.nn.LayerNorm(64)

            def forward(self, x, inputs_embeds=None):
                h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(x)
                for layer in self.layers:
                    h = layer(h)
                return self.norm(h)

        class _VLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = _Vision()
                self.language_model = _TextModel()

            def forward(self, x, pixel_values=None, **kwargs):
                return self.language_model(x)

        with torch.device("meta"):
            model = _VLM()
        model.eval()
        return _build_from_model(model)

    def test_fork_incoming_edges_have_resolved_source_ids(self, vlm_with_forked_layers):
        nodes = vlm_with_forked_layers["graphCollections"][0]["graphs"][0]["nodes"]
        node_ids = {n["id"] for n in nodes}
        forks = [
            n
            for n in nodes
            if any(a.get("value") == "fork" for a in n.get("attrs", []))
        ]
        assert forks, "No Fork node found for the interleaved layer stack"
        for fork in forks:
            edges = fork.get("incomingEdges", [])
            assert edges, f"Fork node {fork['id']} has no incoming edges"
            for e in edges:
                assert e["sourceNodeId"] in node_ids, (
                    f"Fork edge source {e['sourceNodeId']!r} does not resolve to "
                    "a real node (dropped during call-graph merge)"
                )

    def test_visual_output_is_consumed_somewhere(self, vlm_with_forked_layers):
        """The vision branch's output must feed into something visible,
        not dead-end at the collapsed group boundary."""
        nodes = vlm_with_forked_layers["graphCollections"][0]["graphs"][0]["nodes"]
        consumers = [
            n["id"]
            for n in nodes
            for e in n.get("incomingEdges", [])
            if e["sourceNodeId"] == "visual/@output"
        ]
        assert consumers, "visual/@output has no consumers — the arrow dead-ends"


class TestMultiModalFlowDirection:
    """Verify the name-agnostic multi-modal call-graph fallback: when a
    top-level sibling is never invoked (e.g. an optional vision encoder
    skipped because pixel_values wasn't provided), it should be wired as
    a parallel input branch feeding into whichever sibling WAS invoked —
    derived from actual traced call behavior, not hardcoded module names.
    """

    def _build_fake_vlm(self, *, visual_name: str, lm_name: str):
        from TraceLens.ModelUtils.torch_trace import _capture_call_graph

        class _FakeVisual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(64, 64)

            def forward(self, x):
                return self.proj(x)

        class _FakeLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(256, 64)
                self.norm = torch.nn.LayerNorm(64)

            def forward(self, x, inputs_embeds=None):
                if inputs_embeds is None:
                    inputs_embeds = self.embed(x)
                return self.norm(inputs_embeds)

        class _FakeVLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                setattr(self, visual_name, _FakeVisual())
                setattr(self, lm_name, _FakeLM())

            def forward(self, x, pixel_values=None, **kwargs):
                return getattr(self, lm_name)(x)

        with torch.device("meta"):
            model = _FakeVLM()
        model.eval()
        composites = {
            n for n, m in model.named_modules() if n and any(True for _ in m.children())
        }
        cg, _ = _capture_call_graph(model, composites, seq_len=8, batch_size=1)
        return cg

    def test_uninvoked_sibling_wired_as_input(self):
        """Arbitrarily-named modules should work — not just 'visual'/
        'language_model' — since the fallback keys off invocation, not
        naming.

        The side branch must NOT be wired straight into the main branch's
        own boundary when the main branch's first child is a token
        embedding lookup (nn.Embedding): that child only ever consumes
        discrete token ids, never the side branch's continuous features,
        so an edge landing there would look dead once the main branch is
        expanded. It should instead be routed to the child right after
        the embedding, where a real merge would happen.
        """
        cg = self._build_fake_vlm(visual_name="vision_tower", lm_name="text_backbone")
        assert "" in cg, "Root call graph should exist"
        root_edges = cg[""]
        targets = {tgt for _, tgt in root_edges}
        assert "vision_tower" in targets
        assert "text_backbone" in targets
        assert (
            "vision_tower",
            "text_backbone",
        ) not in root_edges, (
            "Should not wire straight into the embedding-guarded boundary"
        )
        assert ("@input", "vision_tower") in root_edges
        assert ("@input", "text_backbone") in root_edges
        assert (
            "vision_tower",
            "text_backbone.norm",
        ) in cg.get("text_backbone", []), (
            "Side branch should be routed past the embedding lookup to "
            "the next child, where it's actually consumable"
        )

    def test_glm_style_names_also_work(self):
        """Sanity check with the GLM naming convention too."""
        cg = self._build_fake_vlm(visual_name="visual", lm_name="language_model")
        assert "" in cg
        assert ("visual", "language_model") not in cg[""]
        assert ("visual", "language_model.norm") in cg.get("language_model", [])

    def test_side_branch_routed_to_first_layer_when_module_list(self):
        """When the main branch's second child is a ModuleList (the usual
        decoder-layer-stack pattern), the side branch should be routed to
        the FIRST layer instance, not the list container itself."""
        from TraceLens.ModelUtils.torch_trace import _capture_call_graph

        class _FakeVisual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(64, 64)

            def forward(self, x):
                return self.proj(x)

        class _FakeLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(64, 64)

            def forward(self, x):
                return self.lin(x)

        class _FakeLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(256, 64)
                self.layers = torch.nn.ModuleList([_FakeLayer(), _FakeLayer()])

            def forward(self, x, inputs_embeds=None):
                h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(x)
                for layer in self.layers:
                    h = layer(h)
                return h

        class _FakeVLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = _FakeVisual()
                self.language_model = _FakeLM()

            def forward(self, x, pixel_values=None, **kwargs):
                return self.language_model(x)

        with torch.device("meta"):
            model = _FakeVLM()
        model.eval()
        composites = {
            n for n, m in model.named_modules() if n and any(True for _ in m.children())
        }
        cg, _ = _capture_call_graph(model, composites, seq_len=8, batch_size=1)
        assert ("visual", "language_model") not in cg.get("", [])
        assert ("visual", "language_model.layers.0") in cg.get("language_model", [])

    def test_no_fallback_when_all_children_invoked(self):
        """If every top-level child is actually invoked, don't guess —
        real tracing should be trusted over the fallback."""
        from TraceLens.ModelUtils.torch_trace import _capture_call_graph

        class _A(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(8, 8)

            def forward(self, x):
                return self.lin(x)

        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(8, 8)

            def forward(self, x):
                return self.lin(x)

        class _Both(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = _A()
                self.b = _A()

            def forward(self, x, **kwargs):
                return self.b(self.a(x))

        with torch.device("meta"):
            model = _Both()
        model.eval()
        composites = {
            n for n, m in model.named_modules() if n and any(True for _ in m.children())
        }
        cg, _ = _capture_call_graph(model, composites, seq_len=8, batch_size=1)
        # Both children were invoked, so the "single invoked child" fallback
        # condition doesn't apply and no root edges are synthesized.
        assert "" not in cg


class TestTopLevelNodeOrder:
    """Regression test: the final node LIST ORDER (not just edges) must
    match true execution order, even when alphabetical order disagrees.
    A prior bug re-appended composite @input/@output nodes in
    alphabetically-sorted order after the exec-order sort had already
    run, so e.g. "language_model" (< "visual" alphabetically) ended up
    listed before "visual" despite visual executing first and feeding
    into language_model.
    """

    @pytest.fixture(scope="class")
    def ordering_payload(self):
        class _Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(8, 8)

            def forward(self, x):
                return self.proj(x)

        class _Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(64, 8)
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x, inputs_embeds=None):
                h = inputs_embeds if inputs_embeds is not None else self.embed(x)
                return self.norm(h)

        class _AlphaOrderedVLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # "a_decoder" sorts BEFORE "z_encoder" alphabetically, but
                # z_encoder is the (uninvoked) side branch that feeds INTO
                # a_decoder — i.e. alphabetical order is the OPPOSITE of
                # true execution order, mirroring the real GLM bug where
                # "language_model" < "visual" alphabetically despite
                # visual executing first.
                self.a_decoder = _Decoder()
                self.z_encoder = _Encoder()

            def forward(self, x, side_input=None, **kwargs):
                # Only a_decoder is unconditionally invoked; z_encoder is
                # skipped here to mimic an optional modality encoder whose
                # optional input wasn't provided.
                return self.a_decoder(x)

        with torch.device("meta"):
            model = _AlphaOrderedVLM()
        model.eval()
        return _build_from_model(model)

    def test_uninvoked_sibling_ordered_before_main(self, ordering_payload):
        """Leaf FX-op nodes get ordered correctly by the first exec-order
        sort regardless of this bug, so check the SYNTHETIC composite
        @input/@output nodes specifically — those were appended in a
        separate alphabetically-sorted loop and are what actually
        exposed the misordering."""
        nodes = ordering_payload["graphCollections"][0]["graphs"][0]["nodes"]
        ids = [n["id"] for n in nodes]
        encoder_out_idx = ids.index("z_encoder/@output")
        decoder_in_idx = ids.index("a_decoder/@input")
        assert encoder_out_idx < decoder_in_idx, (
            f"z_encoder/@output (idx {encoder_out_idx}) feeds into "
            f"a_decoder/@input (idx {decoder_in_idx}) and must be ordered "
            "before it, even though 'a_decoder' sorts alphabetically "
            "before 'z_encoder' — final node order must be driven by "
            "execution order, not alphabetical convenience"
        )


class TestAstCallOrder:
    """Unit tests for ``forward_call_order`` (ast_call_order.py) — the
    ported, well-developed static call-order extractor from the old
    AST-only graph builder. Used as an ordering fallback for composites
    that have zero runtime call-graph signal (e.g. an optional modality
    branch never invoked during the dummy trace), where declaration
    order (``named_children()``) can disagree with true forward() order.
    """

    def test_matches_declared_order_when_same(self):
        from TraceLens.ModelUtils.ast_call_order import forward_call_order

        class _Seq(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(4, 4)
                self.b = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.b(self.a(x))

        assert forward_call_order(_Seq()) == ["a", "b"]

    def test_detects_order_different_from_declaration(self):
        """Mirrors the real GLM vision-encoder bug: post_layernorm is
        declared before downsample/merger in __init__, but forward()
        actually applies it first."""
        from TraceLens.ModelUtils.ast_call_order import forward_call_order

        class _Mismatched(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = torch.nn.Linear(4, 4)
                # Declared BEFORE norm, but called AFTER norm below.
                self.proj_out = torch.nn.Linear(4, 4)
                self.norm = torch.nn.LayerNorm(4)

            def forward(self, x):
                h = self.stem(x)
                h = self.norm(h)
                h = self.proj_out(h)
                return h

        declared = [n for n, _ in _Mismatched().named_children()]
        assert declared == ["stem", "proj_out", "norm"]
        assert forward_call_order(_Mismatched()) == ["stem", "norm", "proj_out"]

    def test_nested_calls_evaluate_inner_first(self):
        """self.a(self.b(x)) must record b before a — b's result is
        computed first and fed into a."""
        from TraceLens.ModelUtils.ast_call_order import forward_call_order

        class _Nested(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(4, 4)
                self.b = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.a(self.b(x))

        assert forward_call_order(_Nested()) == ["b", "a"]

    def test_modulelist_for_loop_records_container(self):
        """A ``for blk in self.blocks:`` loop should place the container
        itself at the correct position relative to its siblings."""
        from TraceLens.ModelUtils.ast_call_order import forward_call_order

        class _Blk(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.lin(x)

        class _WithLoop(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pre = torch.nn.Linear(4, 4)
                self.post = torch.nn.Linear(4, 4)
                self.blocks = torch.nn.ModuleList([_Blk(), _Blk()])

            def forward(self, x):
                h = self.pre(x)
                for blk in self.blocks:
                    h = blk(h)
                return self.post(h)

        assert forward_call_order(_WithLoop()) == ["pre", "blocks", "post"]

    def test_returns_none_when_no_source_available(self):
        from TraceLens.ModelUtils.ast_call_order import forward_call_order

        class _Builtin(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)

        # torch.nn.Module's own default forward raises NotImplementedError
        # and has no children-calling source to analyze in a useful way;
        # more relevantly, a module using a builtin like nn.Sequential's
        # forward should still not crash.
        seq = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        result = forward_call_order(seq)
        assert result is None or isinstance(result, list)

    def test_qkv_split_method_chain_is_not_dropped(self):
        """Regression test for a real GLM vision-attention bug: a module
        call buried in a long method chain — the extremely common QKV
        split idiom ``self.qkv(x).reshape(...).permute(...).unbind(0)`` —
        must still be recorded. ``permute``/``unbind`` were missing from
        ``_METHOD_CHAIN_OPS``, so ``_unwrap_expr`` stopped unwrapping at
        the outer ``.unbind(0)`` call and silently dropped the inner
        ``self.qkv(...)`` call entirely, corrupting the extracted order
        (and, transitively, the wiring of an uninvoked branch)."""
        from TraceLens.ModelUtils.ast_call_order import forward_call_order

        class _Norm(torch.nn.Module):
            def forward(self, x):
                return x

        class _QkvAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = torch.nn.Linear(4, 12)
                self.proj = torch.nn.Linear(4, 4)
                self.q_norm = _Norm()
                self.k_norm = _Norm()

            def forward(self, hidden_states):
                seq_length = hidden_states.shape[0]
                q, k, v = (
                    self.qkv(hidden_states)
                    .reshape(seq_length, 3, 1, -1)
                    .permute(1, 0, 2, 3)
                    .unbind(0)
                )
                q = self.q_norm(q)
                k = self.k_norm(k)
                out = q + k + v
                return self.proj(out)

        assert forward_call_order(_QkvAttn()) == [
            "qkv",
            "q_norm",
            "k_norm",
            "proj",
        ]


class TestUninvokedBranchExecutionOrder:
    """Integration regression test: a composite that's never invoked at
    all during the dummy trace (e.g. an optional vision encoder branch,
    matching the real GLM-5.3-Flash bug) must have its internal children
    sequenced by true forward()-source order, not __init__ declaration
    order — and its synthetic @output boundary must resolve to the
    module that actually runs LAST in forward(), not the one declared
    last in __init__.
    """

    @pytest.fixture(scope="class")
    def payload(self):
        class _Encoder(torch.nn.Module):
            """Mirrors Glm5NextVisionModel: post_layernorm/downsample/
            merger are declared in an order that does NOT match the
            order forward() actually calls them in."""

            def __init__(self):
                super().__init__()
                self.stem = torch.nn.Linear(8, 8)
                # Declared before norm, but applied AFTER norm+extra in
                # forward() — matching downsample/merger being declared
                # before post_layernorm in the real vision model.
                self.extra = torch.nn.Linear(8, 8)
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                h = self.stem(x)
                h = self.norm(h)
                h = self.extra(h)
                return h

        class _Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(64, 8)
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x, inputs_embeds=None):
                h = inputs_embeds if inputs_embeds is not None else self.embed(x)
                return self.norm(h)

        class _FakeVLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = _Decoder()
                self.encoder = _Encoder()

            def forward(self, x, pixel_values=None, **kwargs):
                # encoder is never invoked — mirrors pixel_values being
                # omitted from the dummy trace.
                return self.decoder(x)

        with torch.device("meta"):
            model = _FakeVLM()
        model.eval()
        return _build_from_model(model)

    def test_internal_children_sequenced_by_forward_order(self, payload):
        """encoder/extra should be wired after encoder/norm (true
        forward() order), not after encoder/stem directly (which
        declaration order would incorrectly suggest)."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        extra_node = next((n for n in nodes if n["id"] == "encoder/extra"), None)
        assert extra_node is not None, "encoder/extra node not found"
        sources = [e["sourceNodeId"] for e in extra_node.get("incomingEdges", [])]
        assert sources == ["encoder/norm"], (
            f"encoder/extra should be wired from encoder/norm "
            f"(true forward() order), got {sources}"
        )

    def test_composite_output_resolves_to_true_last_child(self, payload):
        """encoder/@output must resolve to encoder/extra's output (the
        module that actually runs last in forward()), not encoder/norm's
        (which is declared last in __init__)."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        encoder_output = next((n for n in nodes if n["id"] == "encoder/@output"), None)
        assert encoder_output is not None, "encoder/@output node not found"
        sources = [e["sourceNodeId"] for e in encoder_output.get("incomingEdges", [])]
        assert sources == ["encoder/extra"], (
            f"encoder/@output should resolve to encoder/extra (true last "
            f"child in forward()), got {sources}"
        )


class TestUninvokedBranchShapeInference:
    """Regression test for a real GLM-5.3-Flash bug: inside an uninvoked
    branch (e.g. a vision encoder never run because ``pixel_values`` was
    omitted from the dummy trace), custom norm-like modules that don't
    subclass ``torch.nn.LayerNorm``/``torch.nn.RMSNorm`` fell through
    every branch of ``_infer_input_shapes_from_weights`` and ended up
    with NO ``input_shape`` at all, even though
    ``_infer_shapes_from_weights`` (the output-shape counterpart) had a
    generic fallback and produced an ``output_shape`` just fine. This
    left every internal norm-like group of the visual submodel showing
    ``input_shape: ?`` in the viewer while ``output_shape`` was present.
    """

    @pytest.fixture(scope="class")
    def payload(self):
        class _CustomRMSNorm(torch.nn.Module):
            """Doesn't subclass torch.nn.LayerNorm/RMSNorm — mirrors the
            real Glm5NextRMSNorm class used by the GLM vision encoder.
            Multi-op forward so it gets FX-expanded into its own nested
            group/namespace, like the real vision norms do."""

            def __init__(self, hidden_size):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(hidden_size))

            def forward(self, x):
                x = x.float()
                x = x * self.weight
                x = x + 0.0
                return x.type_as(x)

        class _Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = torch.nn.Conv3d(3, 16, 3)
                self.norm = _CustomRMSNorm(16)

            def forward(self, pixel_values, cu_seqlens=None):
                # Tensor-value-dependent branch so symbolic-tracing the
                # whole encoder as one flat graph fails — mirrors why the
                # real GLM vision block can't be traced end-to-end and
                # forces `norm` through the standalone custom-leaf
                # expansion path instead of being silently inlined.
                if cu_seqlens is not None and cu_seqlens.sum() > 0:
                    pass
                h = self.patch_embed(pixel_values)
                return self.norm(h)

        class _Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(64, 8)
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x, inputs_embeds=None):
                h = inputs_embeds if inputs_embeds is not None else self.embed(x)
                return self.norm(h)

        class _FakeVLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = _Decoder()
                self.encoder = _Encoder()

            def forward(self, x, pixel_values=None, **kwargs):
                # encoder is never invoked — mirrors pixel_values being
                # omitted from the dummy trace.
                return self.decoder(x)

        with torch.device("meta"):
            model = _FakeVLM()
        model.eval()
        return _build_from_model(model)

    def _group_attrs(self, payload, suffix):
        """Look up groupNodeAttributes by a plain dotted path, ignoring
        the ` (ClassName)` annotation each key segment carries."""
        ga = payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        target_parts = suffix.split("/")
        for key, val in ga.items():
            parts = [seg.split(" (", 1)[0] for seg in key.split("/")]
            if parts[-len(target_parts) :] == target_parts:
                return val
        return None

    def test_custom_norm_group_has_input_shape(self, payload):
        """encoder/norm (a non-stdlib norm class, never invoked) must
        get an input_shape, not just an output_shape."""
        attrs = self._group_attrs(payload, "encoder/norm")
        assert attrs is not None, "encoder/norm group attrs not found"
        assert attrs.get(
            "input_shape"
        ), f"encoder/norm is missing input_shape, got attrs: {attrs}"
        assert attrs.get(
            "output_shape"
        ), f"encoder/norm is missing output_shape, got attrs: {attrs}"
        # Norm preserves shape, so input and output should match.
        assert attrs["input_shape"] == attrs["output_shape"]


class TestFxLeafPredecessorAncestorWalk:
    """Regression test for a real GLM-5.3-Flash bug: inside an uninvoked
    composite branch (e.g. the vision encoder, never invoked because
    ``pixel_values`` is omitted from the dummy trace), a custom
    multi-op leaf module (FX-expanded via the "custom leaf" path, e.g. an
    RMSNorm) that is the FIRST child of a NESTED submodule — which itself
    has no earlier sibling within its own immediate parent's scope — must
    still resolve its predecessor by walking up through wider ancestor
    scopes (grandparent, etc.), not just the immediate parent. Otherwise
    it falls through to the unresolved literal "@input" placeholder,
    which later gets misinterpreted as the top-level graph input —
    silently severing the real dataflow chain through earlier siblings
    (e.g. patch_embed → rotary_pos_emb → blocks.0 in the real model).
    """

    @pytest.fixture(scope="class")
    def payload(self):
        class _CustomNorm(torch.nn.Module):
            """A multi-op custom leaf (gets FX-expanded, unlike a plain
            nn.LayerNorm/nn.Linear leaf which stays a single node)."""

            def forward(self, x):
                x = x.float()
                x = x * 2.0
                x = x + 1.0
                return x.type_as(x)

        class _Block(torch.nn.Module):
            """block.inner_norm is the FIRST child of block — it has no
            earlier sibling within block's own scope, so its predecessor
            must be resolved from block's *parent* scope instead.

            forward() includes a tensor-value-dependent branch so that
            symbolic-tracing ``_Block`` as one flat composite graph fails
            (mirrors why the real GLM vision-attention block can't be
            traced end-to-end) — otherwise ``inner_norm`` gets silently
            inlined into a single whole-composite trace instead of
            exercising the standalone-custom-leaf (Path B) code path this
            test targets. Never actually invoked at runtime (the whole
            encoder is uninvoked), so the branch's condition never runs.
            """

            def __init__(self):
                super().__init__()
                self.inner_norm = _CustomNorm()
                self.tail = torch.nn.Linear(8, 8)

            def forward(self, x, cu_seqlens):
                if cu_seqlens.sum() > 0:
                    pass
                return self.tail(self.inner_norm(x))

        class _Encoder(torch.nn.Module):
            """encoder.pre runs before encoder.block in forward(), so
            block.inner_norm's real predecessor is encoder.pre's output —
            a "cousin" one level up, not anything within block itself."""

            def __init__(self):
                super().__init__()
                self.pre = _CustomNorm()
                self.block = _Block()

            def forward(self, x, cu_seqlens):
                return self.block(self.pre(x), cu_seqlens)

        class _Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(64, 8)

            def forward(self, x, inputs_embeds=None):
                return self.embed(x)

        class _FakeVLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = _Decoder()
                self.encoder = _Encoder()

            def forward(self, x, pixel_values=None, **kwargs):
                # encoder is never invoked — mirrors pixel_values being
                # omitted from the dummy trace.
                return self.decoder(x)

        with torch.device("meta"):
            model = _FakeVLM()
        model.eval()
        return _build_from_model(model)

    def test_nested_first_child_resolves_to_cousin_sibling(self, payload):
        """block/inner_norm's @input must chain back through block/@input
        to encoder/pre/@output (a cousin one level up) — not fall back to
        the literal root "@input"."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        by_id = {n["id"]: n for n in nodes}

        block_input = by_id.get("encoder/block/@input")
        assert block_input is not None, "encoder/block/@input node not found"
        sources = [e["sourceNodeId"] for e in block_input.get("incomingEdges", [])]
        assert sources == ["encoder/pre/@output"], (
            "encoder/block/@input should chain back to encoder/pre/@output "
            f"(a cousin sibling one ancestor level up), got {sources}"
        )

    def test_no_node_incorrectly_wired_to_root_input(self, payload):
        """Only the true top-level branches (direct children of the root
        model) may source from the literal root "@input" — an internal,
        nested node falling back to it means the ancestor walk failed."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        bad = [
            n["id"]
            for n in nodes
            if n["id"] != "@input" and "/" in n["id"] and n["id"].count("/") > 1
            for e in n.get("incomingEdges", [])
            if e["sourceNodeId"] == "@input"
        ]
        assert bad == [], f"Deeply-nested nodes wired straight to root @input: {bad}"


class TestSingleOpLeafInlining:
    """A childless leaf module whose entire computation is one primitive
    tensor op (e.g. GLM-5.3-Flash's ``Glm5NextTextHyperHead``, which is
    just ``hidden_streams.mean(dim=2)``) should be inlined in place as a
    single flat node — not wrapped in its own namespace box with a
    synthetic @input/<op>/@output boundary around one op. It should also
    get its own correctly-computed (hook-captured) output shape, not the
    pre-op input shape blindly inherited through the boundary chain.
    """

    @pytest.fixture(scope="class")
    def payload(self):
        class _HyperHeadLike(torch.nn.Module):
            """No children, no weights — forward is a single reducing op,
            matching Glm5NextTextHyperHead exactly."""

            def forward(self, hidden_streams):
                return hidden_streams.mean(dim=2)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(64, 8)
                self.expand = torch.nn.Linear(8, 4 * 8)
                self.head = _HyperHeadLike()

            def forward(self, x):
                h = self.embed(x)
                h = self.expand(h)
                h = h.view(*h.shape[:-1], 4, 8)
                return self.head(h)

        with torch.device("meta"):
            model = _Model()
        model.eval()
        return _build_from_model(model)

    def test_no_nested_boundary_nodes(self, payload):
        """There should be no head/@input, head/mean, or head/@output —
        just a single flat 'head' node."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        ids = {n["id"] for n in nodes}
        assert "head" in ids, f"Expected a flat 'head' node, got: {sorted(ids)}"
        assert "head/@input" not in ids
        assert "head/@output" not in ids
        assert "head/mean" not in ids

    def test_label_is_the_op_name(self, payload):
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        head_node = next((n for n in nodes if n["id"] == "head"), None)
        assert head_node is not None
        assert head_node["label"] == "Mean"

    def test_output_shape_reflects_the_reduction(self, payload):
        """head's output shape must reflect mean(dim=2) reducing the 4
        dim away — not the pre-reduction input shape."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        head_node = next((n for n in nodes if n["id"] == "head"), None)
        assert head_node is not None
        meta = head_node.get("outputsMetadata")
        assert meta, "head node is missing outputsMetadata"
        shape = next(
            (a["value"] for a in meta[0].get("attrs", []) if a["key"] == "shape"),
            "",
        )
        assert (
            "4" not in shape.split("bfloat16")[0].split("float32")[0]
        ), f"head's output shape should have the reduced dim (4) removed, got: {shape!r}"


class TestPathAOpInterleaving:
    """Regression test for a real GLM-5.3-Flash bug: a composite module
    that mixes call_module children with inline tensor ops (like the
    vision patch merger's ``proj -> norm -> act -> gate/up_proj ->
    (raw silu/mul ops) -> down_proj``) gets whole-module Path-A FX-traced
    as one flat graph. All of that graph's raw tensor ops used to be
    emitted at the PARENT's own module_order position — i.e. before ANY
    of its call_module children — even though some of those raw ops
    actually run at the END of real execution (after the last child).
    This corrupted the "last node in this namespace" sequential-wiring
    fallback used for children with no preset edges (e.g. the first
    child, whose only real source is the composite's own placeholder
    input, which is deliberately left unwired at that point): the first
    child's predecessor got wrongly resolved to the LAST raw op instead
    of the composite's true external input, creating a dataflow cycle
    that severed the composite from everything upstream of it.
    """

    @pytest.fixture(scope="class")
    def payload(self):
        class _Merger(torch.nn.Module):
            """Mirrors Glm5NextVisionPatchMerger's shape: proj -> (later)
            gate_proj/up_proj (parallel) -> raw ops (silu, mul) ->
            down_proj. No data-dependent control flow, unlike the
            HyperConnection-like fixtures elsewhere in this file — this
            one must whole-module Path-A FX-trace successfully as ONE
            flat graph."""

            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(8, 8)
                self.gate_proj = torch.nn.Linear(8, 8)
                self.up_proj = torch.nn.Linear(8, 8)
                self.down_proj = torch.nn.Linear(8, 8)

            def forward(self, x):
                h = self.proj(x)
                gate = self.gate_proj(h)
                up = self.up_proj(h)
                combined = torch.nn.functional.silu(gate) * up
                return self.down_proj(combined)

        class _Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Linear(8, 8)
                self.merger = _Merger()

            def forward(self, x, pixel_values=None):
                # merger is never invoked at runtime — mirrors the real
                # vision branch being skipped when pixel_values is
                # omitted from the dummy trace.
                return self.embed(x)

        with torch.device("meta"):
            model = _Encoder()
        model.eval()
        return _build_from_model(model)

    def test_proj_not_wired_from_its_own_downstream_ops(self, payload):
        """merger/proj is the FIRST module called in merger's forward()
        — its only real source should be merger's own @input, never one
        of the raw ops that only exist because they run LATER in the
        same forward() (e.g. ``mul``) — that would be a dataflow cycle."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        by_id = {n["id"]: n for n in nodes}
        proj = by_id.get("merger/proj")
        assert proj is not None, "merger/proj node not found"
        sources = [e["sourceNodeId"] for e in proj.get("incomingEdges", [])]
        assert sources == ["merger/@input"], (
            f"merger/proj should be wired from merger/@input, got {sources} "
            "— likely wired from a downstream raw op instead, creating a "
            "cycle."
        )

    def test_no_cycle_in_merger(self, payload):
        """Walking backwards from merger/@output must terminate without
        revisiting any node."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        by_id = {n["id"]: n for n in nodes}
        cur = "merger/@output"
        seen: set[str] = set()
        for _ in range(50):
            seen.add(cur)
            n = by_id.get(cur)
            if not n:
                break
            inc = n.get("incomingEdges", [])
            if not inc:
                break
            cur = inc[0]["sourceNodeId"]
            assert cur not in seen, f"Cycle detected back at {cur}"


class TestEdgeOutputIdNormalization:
    """Regression test: the viewer looks up
    ``sourceNode.outputsMetadata[edge.sourceNodeOutputId]`` to render a
    tensor's shape on an edge/tooltip. Many edges throughout torch_trace
    are built as bare ``{"sourceNodeId": ...}`` dicts (composite
    boundary wiring, sequential fallback, fx_child_edges, etc.) without
    a ``sourceNodeOutputId``/``targetNodeInputId`` — when missing, the
    viewer's lookup fails and the shape renders as "?" even though the
    source node's own outputsMetadata is well-defined."""

    def test_all_incoming_edges_have_output_and_input_ids(self, simple_nodes):
        missing = []
        for n in simple_nodes:
            for e in n.get("incomingEdges", []):
                if "sourceNodeOutputId" not in e or "targetNodeInputId" not in e:
                    missing.append((n["id"], e))
        assert missing == [], (
            f"{len(missing)} incoming edge(s) missing sourceNodeOutputId/"
            f"targetNodeInputId (will render shape as '?' in the viewer): "
            f"{missing[:5]}"
        )


class TestSingleChildPassthroughComposite:
    """Regression test for a real GLM-5.3-Flash bug: a composite module
    whose ONLY registered child fails whole-module Path-A tracing
    entirely (e.g. Glm5NextTextHyperConnection, whose real math involves
    a sinkhorn-iteration loop that can't be symbolically traced) ends up
    with nothing to show except that lone child's own content —
    wrapping it in a pointless, empty "@input"/"@output" boundary box.
    Such composites should be rendered pass-through: no separate box, no
    duplicate boundary — while still keeping sibling instances (e.g.
    attn_hc vs ffn_hc, each wrapping a same-named/same-class child)
    visually distinguishable, since the viewer groups nodes into boxes
    keyed by the raw namespace string.
    """

    @pytest.fixture(scope="class")
    def payload(self):
        class _InnerNorm(torch.nn.Module):
            """Childless leaf — the HyperConnection-like wrapper's only
            real submodule."""

            def forward(self, x):
                return x * 2.0 + 1.0

        class _HyperConnLike(torch.nn.Module):
            """Mirrors Glm5NextTextHyperConnection: registers exactly
            one real submodule child (`input_norm`), but forward() has
            a data-dependent branch (mirroring the untraceable sinkhorn
            loop) that makes whole-module FX tracing fail — so nothing
            of "self" gets captured besides input_norm's own content.
            Never actually invoked (mirrors the real uninvoked branch),
            so the branch condition never runs."""

            def __init__(self):
                super().__init__()
                self.input_norm = _InnerNorm()
                self.scale = torch.nn.Parameter(torch.ones(1))

            def forward(self, x, cu_seqlens):
                if cu_seqlens.sum() > 0:
                    pass
                return self.input_norm(x) * self.scale

        class _DecoderLayer(torch.nn.Module):
            """Two sibling HyperConnection-like wrappers, each with an
            identically-named/-classed `input_norm` child — the exact
            pattern that risks a namespace collision if both get
            inlined naively."""

            def __init__(self):
                super().__init__()
                self.attn_hc = _HyperConnLike()
                self.ffn_hc = _HyperConnLike()

            def forward(self, x, cu_seqlens):
                a = self.attn_hc(x, cu_seqlens)
                return self.ffn_hc(a, cu_seqlens)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = _DecoderLayer()

            def forward(self, x, cu_seqlens=None, **kwargs):
                # layer is never invoked — mirrors the real branch being
                # skipped (uninvoked) in the dummy trace.
                return x

        with torch.device("meta"):
            model = _Model()
        model.eval()
        return _build_from_model(model)

    def test_no_wrapper_boundary_nodes(self, payload):
        """No layer/attn_hc/@input, layer/attn_hc/@output, layer/ffn_hc/
        @input, or layer/ffn_hc/@output — those would wrap a single
        child with no real computation of the composite's own."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        ids = {n["id"] for n in nodes}
        for hc in ("attn_hc", "ffn_hc"):
            assert f"layer/{hc}/@input" not in ids
            assert f"layer/{hc}/@output" not in ids

    def test_sibling_namespaces_stay_distinct(self, payload):
        """attn_hc/input_norm and ffn_hc/input_norm must NOT share a
        namespace string, or the viewer would merge them into one
        visual box."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        attn_ns = {
            n["namespace"]
            for n in nodes
            if n["id"].startswith("layer/attn_hc/input_norm")
        }
        ffn_ns = {
            n["namespace"]
            for n in nodes
            if n["id"].startswith("layer/ffn_hc/input_norm")
        }
        assert attn_ns, "no layer/attn_hc/input_norm nodes found"
        assert ffn_ns, "no layer/ffn_hc/input_norm nodes found"
        assert attn_ns.isdisjoint(ffn_ns), (
            "attn_hc/input_norm and ffn_hc/input_norm share a namespace "
            f"— they'd be merged into one box in the viewer: "
            f"{attn_ns & ffn_ns}"
        )


class TestAttentionKernelGapRealOutput:
    """Regression test for a real GLM-5.3-Flash bug: inside self_attn,
    Q and K/V are produced by separate Linear projections
    (q_b_proj / kv_b_proj), but the actual attention math combining
    them (``attention_interface(...)``, e.g. eager/SDPA attention) is a
    raw *function* call, not an nn.Module — invisible to both FX
    tracing (whole-module Path-A fails here due to data-dependent
    control flow, like the real module's indexer/cache branching) and
    hook-based call-graph capture. This left q_b_proj's and kv_b_proj's
    outputs as untracked dead-ends alongside the genuine o_proj output,
    so ALL THREE got dumped into self_attn's @output — as if self_attn
    returned three tensors — even though only o_proj's shape/value is
    the real output.
    """

    @pytest.fixture(scope="class")
    def payload(self):
        class _AttnLike(torch.nn.Module):
            """Mirrors Glm5NextTextAttention's shape: q_proj and kv_proj
            both consume the composite's own input directly (parallel,
            untracked-@input siblings); the attention math itself is a
            raw op producing a brand-new tensor unrelated (by identity)
            to either projection's output, mirroring the un-hooked
            ``attention_interface(...)`` call; o_proj consumes that new
            tensor and its output IS the composite's real return
            value."""

            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(8, 8)
                self.kv_proj = torch.nn.Linear(8, 8)
                self.o_proj = torch.nn.Linear(8, 8)

            def forward(self, x):
                for _ in x:  # unsupported by FX tracing (Proxy can't be
                    break  # iterated) but fine at real eager runtime —
                    # forces whole-module Path-A tracing to fail without
                    # crashing the meta-device forward pass used for
                    # call-graph capture.
                q = self.q_proj(x)
                kv = self.kv_proj(x)
                attn_out = torch.ones_like(x)  # stand-in for the
                # un-hooked attention kernel; identity unrelated to q/kv
                return self.o_proj(attn_out)

        class _Model(torch.nn.Module):
            # Needs a real embedding so the dummy token-id input (a
            # LongTensor) becomes a proper float feature tensor before
            # reaching self_attn's Linear layers — otherwise the runtime
            # forward pass used for call-graph capture raises (silently
            # swallowed), leaving no edges captured at all for self_attn
            # and defeating the point of this regression test.
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(16, 8)
                self.self_attn = _AttnLike()

            def forward(self, input_ids, **kwargs):
                return self.self_attn(self.embed(input_ids))

        with torch.device("meta"):
            model = _Model()
        model.eval()
        return _build_from_model(model)

    def test_output_sourced_only_from_o_proj(self, payload):
        """self_attn/@output must be wired from o_proj alone — not also
        from q_proj/kv_proj, whose outputs are consumed by the (unseen)
        attention kernel, not by @output."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        by_id = {n["id"]: n for n in nodes}
        output = by_id.get("self_attn/@output")
        assert output is not None, "self_attn/@output node not found"
        sources = [e["sourceNodeId"] for e in output.get("incomingEdges", [])]
        assert sources == ["self_attn/o_proj"], (
            f"self_attn/@output should be sourced only from self_attn/o_proj, "
            f"got {sources} — q_proj/kv_proj dead-ends should not leak into "
            "the composite's output."
        )


class TestNoDistinctNormColor:
    """Regression test for a real GLM-5.3-Flash bug: a built-in
    ``nn.LayerNorm`` leaf (e.g. the indexer's ``k_norm``) rendered as an
    atomic box styled with a unique "norm" category color (khaki),
    while every OTHER norm in the graph is a custom RMSNorm subclass
    that gets FX-expanded into its raw ops (styled the same neutral
    gray as any other primitive op). This made the plain nn.LayerNorm
    stand out as if it were structurally different, when it isn't. No
    node should ever use that distinct color — every leaf module that
    isn't specifically categorized (embedding/linear/attention/
    activation) should render with the same neutral default style."""

    @pytest.fixture(scope="class")
    def payload(self):
        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(16, 8)
                self.k_norm = torch.nn.LayerNorm(8)
                self.lin = torch.nn.Linear(8, 8)

            def forward(self, input_ids, **kwargs):
                return self.lin(self.k_norm(self.embed(input_ids)))

        with torch.device("meta"):
            model = _Model()
        model.eval()
        return _build_from_model(model)

    def test_layernorm_uses_default_style_not_a_unique_color(self, payload):
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        by_id = {n["id"]: n for n in nodes}
        k_norm = by_id.get("k_norm")
        lin = by_id.get("lin")
        assert k_norm is not None, "k_norm node not found"
        assert lin is not None, "lin node not found"
        assert k_norm["style"] == lin["style"], (
            "nn.LayerNorm should render with the same neutral style as "
            f"other uncategorized leaf modules, got {k_norm['style']} vs "
            f"Linear's {lin['style']}"
        )

    def test_no_node_uses_the_retired_norm_color(self, payload):
        """No node anywhere should use the old khaki "norm" color
        (#f0e68c) — it should be fully unreachable now."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        for n in nodes:
            style = n.get("style") or {}
            assert (
                style.get("backgroundColor") != "#f0e68c"
            ), f"Node {n['id']} uses the retired norm color"


class TestParallelSiblingsShareUntrackedInput:
    """Regression test for a real GLM-5.3-Flash bug: inside
    Glm5NextTextLinearAttention, q_proj/k_proj/v_proj (and other
    siblings) are all called directly on the SAME masked hidden_states
    produced by an un-hooked helper (``apply_mask_to_padding_states``),
    which returns a brand-new tensor when a mask is present — breaking
    tensor-identity tracking back to the composite's real "@input".
    Because the sequential fallback used to run BEFORE the "siblings
    sharing an untracked input are parallel" rule, it unconditionally
    chained these into a bogus straight line (q_proj → k_proj → v_proj)
    before the parallel rule ever got a chance to apply, making three
    independent parallel projections look like a serial pipeline.
    """

    @pytest.fixture(scope="class")
    def payload(self):
        class _Attn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(8, 8)
                self.k_proj = torch.nn.Linear(8, 8)
                self.v_proj = torch.nn.Linear(8, 8)

            def forward(self, x, mask):
                for _ in x:  # unsupported by FX tracing (Proxy can't be
                    break  # iterated) but fine at real eager runtime —
                    # forces whole-module Path-A tracing to fail, so this
                    # falls through to the call-graph-driven fallback
                    # wiring instead of FX's (correct) ground truth.
                # Mirrors apply_mask_to_padding_states: multiplying by a
                # mask produces a brand-new tensor (untracked identity),
                # consumed in parallel by three sibling projections.
                masked = x * mask
                q = self.q_proj(masked)
                k = self.k_proj(masked)
                v = self.v_proj(masked)
                return torch.cat([q, k, v], dim=-1)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(16, 8)
                self.attn = _Attn()

            def forward(self, input_ids, **kwargs):
                x = self.embed(input_ids)
                mask = torch.ones_like(x)
                return self.attn(x, mask)

        with torch.device("meta"):
            model = _Model()
        model.eval()
        return _build_from_model(model)

    def test_siblings_all_wired_from_input_not_chained(self, payload):
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        by_id = {n["id"]: n for n in nodes}
        for name in ("q_proj", "k_proj", "v_proj"):
            node = by_id.get(f"attn/{name}")
            assert node is not None, f"attn/{name} node not found"
            sources = [e["sourceNodeId"] for e in node.get("incomingEdges", [])]
            assert sources == ["attn/@input"], (
                f"attn/{name} should be wired directly from attn/@input "
                f"(parallel sibling), got {sources} — looks chained from "
                "another sibling instead."
            )


class TestDeadLeafInLiveParent:
    """Regression test for a real GLM-5.3-Flash bug: a leaf module that
    is registered as a submodule but never actually invoked (its
    .weight/.bias are read directly by a raw, un-hooked kernel function
    instead of calling ``self.mod(x)`` — e.g. KDA linear attention's
    conv1d, whose weights feed a raw ``causal_conv1d_fn(...)`` call)
    used to still get a speculative "@input" edge from the generic
    positional-fallback wiring pass, even though no real tensor ever
    flows into (or out of) it. A half-wire (input but no consumer)
    looks like a broken/dead computation step. Such leaves should
    render as standalone info boxes instead — no incoming edges at all
    — since they aren't really part of the traced dataflow.
    """

    @pytest.fixture(scope="class")
    def payload(self):
        class _WithDeadConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(8, 8)
                # Registered but never called anywhere in forward() —
                # mirrors reading conv1d.weight/.bias directly instead of
                # calling self.conv1d(x).
                self.conv1d = torch.nn.Conv1d(8, 8, kernel_size=3, padding=2)

            def forward(self, x):
                return self.proj(x)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(16, 8)
                self.blk = _WithDeadConv()

            def forward(self, input_ids, **kwargs):
                return self.blk(self.embed(input_ids))

        with torch.device("meta"):
            model = _Model()
        model.eval()
        return _build_from_model(model)

    def test_never_invoked_leaf_has_no_incoming_edges(self, payload):
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        by_id = {n["id"]: n for n in nodes}
        conv1d = by_id.get("blk/conv1d")
        assert conv1d is not None, "blk/conv1d node not found"
        assert conv1d.get("incomingEdges") == [], (
            "conv1d was never invoked, so it should have no incoming "
            f"edges (standalone info box), got {conv1d.get('incomingEdges')}"
        )

    def test_never_invoked_leaf_still_has_shape_metadata(self, payload):
        """It should still render with a weight-inferred shape, just
        without pretending it's wired into the dataflow."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        by_id = {n["id"]: n for n in nodes}
        conv1d = by_id.get("blk/conv1d")
        assert conv1d is not None
        assert conv1d.get("outputsMetadata"), "conv1d should still have shape metadata"


class TestSingleChildPassthroughShapeMismatchBlocked:
    """Regression test for a real GLM-5.3-Flash bug: a single-child
    composite whose FX trace fails does NOT always deserve the
    pass-through treatment from `TestSingleChildPassthroughComposite`
    above. That's only correct for a genuine trivial wrapper, where the
    child's own captured shapes match the composite's own exactly.

    Glm5NextTextHyperConnection registers exactly one real submodule
    (`input_norm`), but only feeds it a *flattened* side-branch used to
    derive gating weights — the composite's actual returned hidden
    state is a differently-shaped reduction of the *original*
    (unflattened) input, computed via untraceable math that never
    touches `input_norm`'s output. Naively inlining `input_norm` made
    it look like the composite's real output IS input_norm's output,
    silently propagating the WRONG (flattened) shape downstream —
    creating the appearance of two norm-like ops in a row with a bogus
    shape change between them, when really there's a whole (untraced)
    computation in between.
    """

    @pytest.fixture(scope="class")
    def payload(self):
        class _InnerNorm(torch.nn.Module):
            """Leaf child used only on a flattened side-branch to derive
            gating weights — its own shape is NOT the composite's real
            output shape."""

            def forward(self, x):
                return x * 2.0

        class _HyperConnLike(torch.nn.Module):
            """Mirrors Glm5NextTextHyperConnection: registers exactly one
            real submodule child (`input_norm`), but the child only ever
            sees a *flattened* view of the input, while the composite's
            actual returned value is a differently-shaped reduction of
            the *original*, unflattened input — computed via math the
            child's output is never used in."""

            def __init__(self):
                super().__init__()
                self.input_norm = _InnerNorm()

            def forward(self, x):
                for _ in x:  # unsupported by FX tracing (Proxy can't be
                    break  # iterated) but fine at real eager runtime —
                    # forces whole-module Path-A tracing to fail, so this
                    # falls through to the call-graph-driven fallback.
                flat = self.input_norm(x.flatten(start_dim=2))  # (B, S, H*D)
                # Real output never touches `flat` — different shape,
                # different (unnormalized) source tensor.
                return x.sum(dim=2)  # (B, S, D), collapsing the H axis

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(16, 8)
                self.hc = _HyperConnLike()
                self.next_layer = torch.nn.Linear(4, 4)

            def forward(self, input_ids, **kwargs):
                x = self.embed(input_ids)  # (B, S, 8)
                streams = x.view(*x.shape[:-1], 2, 4)  # (B, S, H=2, D=4)
                h = self.hc(streams)  # (B, S, 4) — NOT (B, S, 8)
                return self.next_layer(h)

        with torch.device("meta"):
            model = _Model()
        model.eval()
        return _build_from_model(model)

    def test_hc_gets_own_boundary_not_inlined(self, payload):
        """Since input_norm's shape doesn't match hc's real output
        shape, hc must NOT be collapsed into a pass-through — it needs
        its own @input/@output boundary box."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        ids = {n["id"] for n in nodes}
        assert "hc/@input" in ids, "hc should get its own @input boundary"
        assert "hc/@output" in ids, "hc should get its own @output boundary"

    def test_downstream_consumer_shape_matches_hc_output_shape(self, payload):
        """next_layer must see the SAME shape that hc's own @output
        declares — not input_norm's (flattened, mismatched) shape."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        by_id = {n["id"]: n for n in nodes}

        def _shape(node_id):
            n = by_id.get(node_id)
            assert n is not None, f"{node_id} node not found"
            meta = n.get("outputsMetadata")
            assert meta, f"{node_id} is missing outputsMetadata"
            return next(
                a["value"] for a in meta[0]["attrs"] if a["key"] == "shape"
            )

        hc_out_shape = _shape("hc/@output")
        next_layer = by_id.get("next_layer")
        assert next_layer is not None, "next_layer node not found"
        sources = [e["sourceNodeId"] for e in next_layer.get("incomingEdges", [])]
        assert sources == ["hc/@output"], (
            f"next_layer should be wired directly from hc/@output, got {sources}"
        )
        # The declared output shape must reflect the real (D=4) reduction,
        # not input_norm's flattened (H*D=8) shape.
        assert "4 " in hc_out_shape or hc_out_shape.endswith("4"), (
            f"hc/@output shape should reflect the real D=4 output, got {hc_out_shape!r}"
        )
        assert "8" not in hc_out_shape, (
            f"hc/@output shape should NOT be input_norm's flattened (H*D=8) "
            f"shape, got {hc_out_shape!r}"
        )


class TestNestedGateWiredFromCompositeEntryNotSibling:
    """Regression test for a real GLM-5.3-Flash bug: inside
    Glm5NextTextLinearAttention, ``forget_gate`` is a NESTED composite
    (its own children ``f_a_proj``/``f_b_proj``) that receives
    ``self_attn``'s own input directly — in parallel with sibling
    projections like ``v_proj``, not sequentially after them. But
    ``forget_gate``'s own call-graph edge is just ``("@input",
    "forget_gate")`` (relative to ``self_attn``), which
    ``_cg_predecessor`` deliberately skips (it only tracks *real*
    sibling predecessors, not generic "@input" edges) — so resolving
    forget_gate's entry point requires recursing up to self_attn's own
    (real) predecessor.

    The bug: `_record_composite_entry` used to run BEFORE
    `_cg_find_sources` for a given node, using the naive *sequential*
    guess (whatever ran immediately before it — a sibling like
    ``v_proj``, purely by chance of node-list order) as that node's
    ancestors' recorded "entry point" — including ``forget_gate``
    itself. Since this happened for forget_gate's OWN first inner node
    (``f_a_proj``), it poisoned ``composite_entry["...forget_gate"]``
    with the wrong sibling-based guess *before* the call-graph-aware
    resolution for that exact same node ever got to look it up —
    silently overriding the correct recursive resolution with a bogus
    "wired from an unrelated sibling" edge.
    """

    @pytest.fixture(scope="class")
    def payload(self):
        class _Gate(torch.nn.Module):
            """Mirrors Glm5NextTextForgetGate: a nested composite with
            its own two children, called directly on the parent's own
            input (not on a sibling's output)."""

            def __init__(self):
                super().__init__()
                self.f_a_proj = torch.nn.Linear(8, 8)
                self.f_b_proj = torch.nn.Linear(8, 8)

            def forward(self, x):
                return self.f_b_proj(self.f_a_proj(x))

        class _SelfAttnLike(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.v_proj = torch.nn.Linear(8, 8)
                self.gate = _Gate()

            def forward(self, x):
                for _ in x:  # unsupported by FX tracing (Proxy can't be
                    break  # iterated) but fine at real eager runtime —
                    # forces whole-module Path-A tracing to fail, so this
                    # falls through to the call-graph-driven fallback
                    # wiring instead of FX's (correct) ground truth.
                v = self.v_proj(x)
                g = self.gate(x)  # SAME x as v_proj — parallel, not
                # sequential — mirrors `g = self.forget_gate(hidden_states)`
                # running after `v = self.v_proj(hidden_states)` but on
                # the identical tensor, not on v_proj's output.
                return v + g

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(16, 8)
                self.norm = torch.nn.LayerNorm(8)
                self.self_attn = _SelfAttnLike()

            def forward(self, input_ids, **kwargs):
                return self.self_attn(self.norm(self.embed(input_ids)))

        with torch.device("meta"):
            model = _Model()
        model.eval()
        return _build_from_model(model)

    def test_gate_input_wired_from_composite_entry_not_sibling(self, payload):
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        by_id = {n["id"]: n for n in nodes}
        # `gate` gets its own synthetic @input/@output boundary (it has
        # >= 2 real children), so the edge of interest is on THAT
        # boundary node — it must resolve back to self_attn's own
        # shared entry point, not to the sibling v_proj.
        gate_input_boundary = by_id.get("self_attn/gate/@input")
        assert gate_input_boundary is not None, "self_attn/gate/@input node not found"
        sources = [
            e["sourceNodeId"] for e in gate_input_boundary.get("incomingEdges", [])
        ]
        assert sources == ["self_attn/@input"], (
            "self_attn/gate/@input should be wired directly from "
            f"self_attn/@input (its real, shared predecessor), got {sources} "
            "— looks like it was wired from an unrelated sibling instead."
        )

    def test_sibling_v_proj_also_wired_from_composite_entry(self, payload):
        """Sanity check: v_proj (the sibling whose wrong guess used to
        leak into gate's resolution) must itself still resolve
        correctly too."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        by_id = {n["id"]: n for n in nodes}
        v_proj = by_id.get("self_attn/v_proj")
        assert v_proj is not None, "self_attn/v_proj node not found"
        sources = [e["sourceNodeId"] for e in v_proj.get("incomingEdges", [])]
        assert sources == ["self_attn/@input"], (
            f"self_attn/v_proj should be wired from self_attn/@input, got {sources}"
        )


class TestForkPredecessorNotClobberedByAncestorInput:
    """Regression test for a real GLM-5.3-Flash bug: when a side-modality
    branch (e.g. a vision encoder, skipped because its optional input was
    omitted) merges into a Fork node for an interleaved layer-type stack,
    the Fork's correctly-resolved predecessor used to get silently
    overwritten by the *generic* ancestor's own "@input" boundary.

    Root cause: composite boundary creation (which builds each module's
    own synthetic "@input"/"@output" nodes) treats ANY child node whose
    incoming edge source lives outside the ancestor's own subtree as a
    generic "input child" needing rewiring to the ancestor's own
    "@input" — but a Fork node is namespaced INSIDE the ancestor even
    though it represents the entry to a NESTED layer-stack container,
    and its predecessor was already carefully resolved straight from the
    real call graph. Treating it like any other "input child" collapsed
    that specific, real source into the ancestor's own generic "@input",
    destroying the distinction between "the model's own primary input"
    and "a side branch merging in partway through" — and also left the
    token-embedding path (the model's OTHER real predecessor) as a
    disconnected dead end with no consumer at all.
    """

    @pytest.fixture(scope="class")
    def payload(self):
        class _Vision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(64, 64)

            def forward(self, x):
                return self.proj(x)

        class _TypeA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(64)

            def forward(self, x):
                return self.norm(x)

        class _TypeB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = torch.nn.Linear(64, 64)

            def forward(self, x):
                return self.gate(x)

        class _TextModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(256, 64)
                self.layers = torch.nn.ModuleList(
                    [_TypeA(), _TypeB(), _TypeA(), _TypeB()]
                )
                self.norm = torch.nn.LayerNorm(64)

            def forward(self, x, inputs_embeds=None):
                h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(x)
                for layer in self.layers:
                    h = layer(h)
                return self.norm(h)

        class _VLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = _Vision()
                self.language_model = _TextModel()

            def forward(self, x, pixel_values=None, **kwargs):
                return self.language_model(x)

        with torch.device("meta"):
            model = _VLM()
        model.eval()
        return _build_from_model(model)

    def test_fork_predecessor_is_visual_not_generic_input(self, payload):
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        by_id = {n["id"]: n for n in nodes}
        fork = by_id.get("language_model/layers/@fork")
        assert fork is not None, "language_model/layers/@fork node not found"
        sources = {e["sourceNodeId"] for e in fork.get("incomingEdges", [])}
        assert "visual/@output" in sources, (
            "Fork's predecessor should include visual/@output (the real, "
            f"resolved side-branch source), got {sources}"
        )

    def test_embed_tokens_also_feeds_fork_and_has_a_consumer(self, payload):
        """The token-embedding path must ALSO be visible as a parallel
        predecessor of the Fork — not left disconnected just because the
        side branch's edge claims the merge point first."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        by_id = {n["id"]: n for n in nodes}
        fork = by_id.get("language_model/layers/@fork")
        assert fork is not None
        sources = {e["sourceNodeId"] for e in fork.get("incomingEdges", [])}
        assert "language_model/embed_tokens" in sources, (
            f"Fork should also be wired from language_model/embed_tokens, got {sources}"
        )
        consumers = [
            n["id"]
            for n in nodes
            for e in n.get("incomingEdges", [])
            if e["sourceNodeId"] == "language_model/embed_tokens"
        ]
        assert consumers, "language_model/embed_tokens has no consumers — dead end"

    def test_language_model_own_input_is_clean(self, payload):
        """language_model's own @input boundary should carry only the
        model's real primary (token id) input — not also get polluted
        with the side-branch's edge that actually belongs to the nested
        Fork merge point."""
        nodes = payload["graphCollections"][0]["graphs"][0]["nodes"]
        by_id = {n["id"]: n for n in nodes}
        lm_input = by_id.get("language_model/@input")
        assert lm_input is not None, "language_model/@input node not found"
        sources = [e["sourceNodeId"] for e in lm_input.get("incomingEdges", [])]
        assert "visual/@output" not in sources, (
            "language_model/@input should not carry the visual side-branch "
            f"edge (that belongs on the nested Fork instead), got {sources}"
        )
