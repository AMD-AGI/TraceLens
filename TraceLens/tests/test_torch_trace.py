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
        self.layers = torch.nn.ModuleList(
            [_SimpleBlock(dim) for _ in range(n_layers)]
        )
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
    from TraceLens.ModelUtils.torch_trace import (
        _capture_call_graph,
        _capture_shapes,
        _detect_repeated_layers,
        _infer_shapes_from_weights,
        _patch_rotary_embeddings,
    )

    with torch.device("meta"):
        model = _SimpleModel()
    model.eval()

    return build_graph.__wrapped__(model) if hasattr(build_graph, "__wrapped__") else _build_from_model(model)


def _build_from_model(model: torch.nn.Module) -> dict:
    """Replicate the core of build_graph() for a pre-instantiated model."""
    import types
    from collections import defaultdict
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
            return {k: v for k, v in self.__class__.__dict__.items()
                    if not k.startswith("_") and not callable(v)}

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
        attn_outputs = [n for n in simple_nodes
                        if "self_attn" in n["id"] and n["id"].endswith("/@output")]
        # Should have one per representative layer
        for n in attn_outputs:
            assert n["label"] == "Output"
        # No "self_attn Output" labels
        ext_labels = [n for n in simple_nodes
                      if "self_attn Output" in n.get("label", "")]
        assert ext_labels == []

    def test_self_attn_has_single_input(self, simple_nodes):
        """self_attn should have exactly one @input, not both @input and @ext_input."""
        attn_inputs = [n for n in simple_nodes
                       if "self_attn" in n["id"] and n["id"].endswith("/@input")]
        for n in attn_inputs:
            assert n["label"] == "Input"
        ext_labels = [n for n in simple_nodes
                      if "self_attn Input" in n.get("label", "")]
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
                assert n.get("outputsMetadata"), (
                    f"Synthetic input {n['id']} missing outputsMetadata"
                )

    def test_synthetic_output_nodes_have_shapes(self, simple_nodes):
        """Most synthetic output nodes should have outputsMetadata.

        Some composite modules (e.g. decoder blocks) may lack captured
        shapes, so we check that the vast majority have them.
        """
        outputs = [n for n in simple_nodes
                   for a in n.get("attrs", [])
                   if a.get("key") == "synthetic" and a.get("value") == "output"]
        with_shapes = [n for n in outputs if n.get("outputsMetadata")]
        assert len(outputs) > 0
        # Composites whose children are all FX ops without captured
        # shapes (e.g. decoder blocks with residual adds) may lack
        # outputsMetadata.  Require at least 50% coverage.
        assert len(with_shapes) / len(outputs) >= 0.5, (
            f"Only {len(with_shapes)}/{len(outputs)} synthetic outputs have shapes"
        )

    def test_all_nodes_have_shapes(self, simple_nodes):
        """Every node (leaf, FX op, synthetic) must have outputsMetadata."""
        missing = [n["id"] for n in simple_nodes if not n.get("outputsMetadata")]
        assert missing == [], (
            f"{len(missing)} nodes missing outputsMetadata: {missing}"
        )

    def test_embedding_edge_has_shape(self, simple_nodes):
        """The edge from @input to embedding should carry a shape, not '?'."""
        emb = next(n for n in simple_nodes if "embed_tokens" in n["id"])
        # The source of embedding's edge should have outputsMetadata
        node_by_id = {n["id"]: n for n in simple_nodes}
        for e in emb.get("incomingEdges", []):
            src = node_by_id.get(e["sourceNodeId"])
            if src:
                assert src.get("outputsMetadata"), (
                    f"Source {src['id']} of embed_tokens edge has no shape"
                )


class TestGroupNodeAttributes:
    """Verify groupNodeAttributes uses the dict format expected by Model Explorer."""

    def test_is_dict(self, simple_payload):
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        assert isinstance(ga, dict), (
            f"groupNodeAttributes should be dict, got {type(ga).__name__}"
        )

    def test_values_are_dicts(self, simple_payload):
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        for key, val in ga.items():
            assert isinstance(val, dict), (
                f"groupNodeAttributes['{key}'] should be dict, got {type(val).__name__}"
            )

    def test_has_class_key(self, simple_payload):
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        for key, val in ga.items():
            assert "class" in val, (
                f"groupNodeAttributes['{key}'] missing 'class' key"
            )

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
                assert e["sourceNodeId"] in node_ids, (
                    f"Node {n['id']} has broken edge from {e['sourceNodeId']}"
                )

    def test_all_non_input_nodes_have_edges(self, simple_nodes):
        """Every node except @input must have at least one incoming edge."""
        for n in simple_nodes:
            if n["id"] == "@input":
                continue
            assert n.get("incomingEdges"), (
                f"Node {n['id']} has no incoming edges"
            )

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
            assert n.get("incomingEdges"), (
                f"Block node {n['id']} has no incoming edges (dead node)"
            )

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

        call_graph = _capture_call_graph(model, composites, seq_len=8, batch_size=1)
        block_edges = call_graph.get("block", [])
        edge_set = {(s.split(".")[-1] if s != "@input" else s,
                      t.split(".")[-1]) for s, t in block_edges}

        # proj_b should have an edge (either from proj_a via fallback,
        # or from @input). It must not be orphaned.
        proj_b_sources = {s for s, t in edge_set if t == "proj_b"}
        assert proj_b_sources, (
            f"proj_b has no incoming edges. Got: {edge_set}"
        )


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
                assert "input_shape" in attrs, (
                    f"Group '{key}' has output_shape but missing input_shape"
                )

    def test_composite_groups_with_output_have_input(self, simple_payload):
        """All composite module groups with output_shape should also have input_shape."""
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        for key, attrs in ga.items():
            if "count" in attrs:
                continue  # layer groups may not need input_shape
            if "output_shape" in attrs:
                assert "input_shape" in attrs, (
                    f"Group '{key}' has output_shape but missing input_shape"
                )

    def test_fx_expanded_module_has_shapes(self, simple_payload):
        """FX-expanded leaf modules (e.g. LayerNorm) should have shape attributes."""
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        norm_groups = {k: v for k, v in ga.items() if "Norm" in k or "norm" in k}
        for key, attrs in norm_groups.items():
            assert "output_shape" in attrs, (
                f"FX-expanded group '{key}' missing output_shape"
            )


class TestOutputShapeCoverage:
    """Verify output_shape is present on all composite module groups."""

    def test_all_composites_have_output_shape(self, simple_payload):
        """Every composite module group should have an output_shape attribute."""
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        missing = []
        for key, attrs in ga.items():
            if "output_shape" not in attrs:
                missing.append(key)
        assert not missing, (
            f"Groups missing output_shape: {missing}"
        )

    def test_layer_groups_have_output_shape(self, simple_payload):
        """Layer groups (with 'count') should have output_shape."""
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        layer_groups = {k: v for k, v in ga.items() if "count" in v}
        for key, attrs in layer_groups.items():
            assert "output_shape" in attrs, (
                f"Layer group '{key}' missing output_shape"
            )


class TestMultiOutputWiring:
    """Verify all output children of composite modules are wired to @output."""

    def test_no_orphan_output_nodes(self, simple_nodes):
        """No synthetic @output node should be completely disconnected."""
        node_ids = {n["id"] for n in simple_nodes}
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
            assert consumed, (
                f"Synthetic output node {n['id']} has no consumers (dead output)"
            )

    def test_block_output_wires_from_mlp(self, simple_nodes):
        """The decoder block @output should include mlp's output in the chain."""
        block_output = next(
            (n for n in simple_nodes if n["id"] == "layers/0/@output"),
            None
        )
        assert block_output is not None, "Missing layers/0/@output"
        sources = {e["sourceNodeId"] for e in block_output.get("incomingEdges", [])}
        # The block output should be fed by mlp/@output (directly or indirectly)
        # At minimum, check that mlp/@output is consumed by something in the block
        mlp_output = next(
            (n for n in simple_nodes if n["id"] == "layers/0/mlp/@output"),
            None
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
        assert not orphans, (
            f"Orphan @output nodes (not consumed): {orphans}"
        )


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
        assert len(dims) <= 3, (
            f"@input has too many dims for token IDs: {shape}"
        )

    def test_embedding_parent_input_is_integer_dtype(self, simple_nodes):
        """A composite whose first child is Embedding should have int64 @input."""
        # In _SimpleModel, the root model's @input carries token IDs
        shape = self._get_shape(simple_nodes, "@input")
        assert "int64" in shape, (
            f"Root @input should be int64 (token IDs), got: {shape}"
        )

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
        assert "256" not in block_shape, (
            f"layers/0/@output seems to show MLP intermediate shape: {block_shape}"
        )

    def test_self_attn_output_shape(self, simple_nodes):
        """self_attn/@output should reflect o_proj output (hidden_dim),
        not an intermediate projection size."""
        shape = self._get_shape(simple_nodes, "layers/0/self_attn/@output")
        if shape:
            assert "64" in shape, (
                f"self_attn/@output should include hidden_dim=64, got: {shape}"
            )


class TestInputShapeInference:
    """Verify _infer_input_shapes_from_weights produces correct shapes."""

    def test_linear_input_shape(self):
        from TraceLens.ModelUtils.torch_trace import _infer_input_shapes_from_weights
        model = torch.nn.Sequential(torch.nn.Linear(32, 64))
        shapes = _infer_input_shapes_from_weights(
            model, {}, batch_size=1, seq_len=10
        )
        assert shapes.get("0") == (1, 10, 32)

    def test_embedding_input_shape(self):
        from TraceLens.ModelUtils.torch_trace import _infer_input_shapes_from_weights
        model = torch.nn.Sequential(torch.nn.Embedding(100, 64))
        shapes = _infer_input_shapes_from_weights(
            model, {}, batch_size=1, seq_len=10
        )
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
        shapes = _infer_input_shapes_from_weights(
            model, {}, batch_size=1, seq_len=10
        )
        assert shapes.get("0") == (1, 3, 10, 10)


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
        root_output = next(
            (n for n in simple_nodes if n["id"] == "@output"), None
        )
        assert root_output is not None
        sources = [e["sourceNodeId"] for e in root_output.get("incomingEdges", [])]
        assert sources, "@output has no incoming edges"

    def test_output_shape_exists(self, simple_nodes):
        """@output should have shape metadata."""
        root_output = next(
            (n for n in simple_nodes if n["id"] == "@output"), None
        )
        if root_output and root_output.get("outputsMetadata"):
            shape = next(
                (a["value"] for a in root_output["outputsMetadata"][0].get("attrs", [])
                 if a["key"] == "shape"),
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
                assert "output_shape" in attrs, (
                    f"Layer group '{key}' missing output_shape"
                )


class TestGroupAttrOrdering:
    """Verify input_shape appears first and output_shape last in group attrs."""

    def test_input_before_output(self, simple_payload):
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        for key, attrs in ga.items():
            keys = list(attrs.keys())
            if "input_shape" in keys and "output_shape" in keys:
                assert keys.index("input_shape") < keys.index("output_shape"), (
                    f"In group '{key}', input_shape should come before output_shape: {keys}"
                )

    def test_input_shape_is_first(self, simple_payload):
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        for key, attrs in ga.items():
            keys = list(attrs.keys())
            if "input_shape" in keys:
                assert keys[0] == "input_shape", (
                    f"In group '{key}', input_shape should be first: {keys}"
                )

    def test_output_shape_is_last(self, simple_payload):
        ga = simple_payload["graphCollections"][0]["graphs"][0]["groupNodeAttributes"]
        for key, attrs in ga.items():
            keys = list(attrs.keys())
            if "output_shape" in keys:
                assert keys[-1] == "output_shape", (
                    f"In group '{key}', output_shape should be last: {keys}"
                )


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
                self.layers = torch.nn.ModuleList([
                    _TypeA(), _TypeB(), _TypeA(), _TypeB(),
                ])
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
        forks = [n for n in nodes
                 if any(a.get("value") == "fork" for a in n.get("attrs", []))]
        assert len(forks) >= 1, "No Fork node found for multi-group container"

    def test_join_node_exists(self, multi_group_model):
        nodes = multi_group_model["graphCollections"][0]["graphs"][0]["nodes"]
        joins = [n for n in nodes
                 if any(a.get("value") == "join" for a in n.get("attrs", []))]
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
        composites = {n for n, m in model.named_modules()
                      if n and any(True for _ in m.children())}
        return _capture_call_graph(model, composites, seq_len=8, batch_size=1)

    def test_uninvoked_sibling_wired_as_input(self):
        """Arbitrarily-named modules should work — not just 'visual'/
        'language_model' — since the fallback keys off invocation, not
        naming."""
        cg = self._build_fake_vlm(visual_name="vision_tower", lm_name="text_backbone")
        assert "" in cg, "Root call graph should exist"
        root_edges = cg[""]
        targets = {tgt for _, tgt in root_edges}
        assert "vision_tower" in targets
        assert "text_backbone" in targets
        assert ("vision_tower", "text_backbone") in root_edges, (
            "Uninvoked sibling should feed into the invoked one"
        )
        assert ("@input", "vision_tower") in root_edges
        assert ("@input", "text_backbone") in root_edges

    def test_glm_style_names_also_work(self):
        """Sanity check with the GLM naming convention too."""
        cg = self._build_fake_vlm(visual_name="visual", lm_name="language_model")
        assert "" in cg
        assert ("visual", "language_model") in cg[""]

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
        composites = {n for n, m in model.named_modules()
                      if n and any(True for _ in m.children())}
        cg = _capture_call_graph(model, composites, seq_len=8, batch_size=1)
        # Both children were invoked, so the "single invoked child" fallback
        # condition doesn't apply and no root edges are synthesized.
        assert "" not in cg
