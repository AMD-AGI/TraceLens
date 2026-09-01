###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Focused branch coverage for AST analysis and computation graph construction."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from visualizer import ast_analyze as aa
from visualizer import computation_graph as cg
from visualizer.block_tree import (
    BlockNode,
    Branch,
    CombineSegment,
    FanOutSegment,
    ResidualAddSegment,
    SeqSegment,
    SideCombineSegment,
    SideFeedSegment,
    TensorPortsSegment,
)


def _node(
    name: str,
    *,
    class_name: str = "Linear",
    children: list[BlockNode] | None = None,
    **kwargs,
) -> BlockNode:
    return BlockNode(
        attr_name=name,
        class_name=class_name,
        role=kwargs.pop("role", "other"),
        label=kwargs.pop("label", class_name),
        children=children or [],
        **kwargs,
    )


def _function(source: str) -> ast.FunctionDef:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def test_public_name_role_and_kernel_helpers_cover_fallbacks():
    synthetic = aa.positional_synthetic_attr("apply_rotary_emb", 17)
    assert aa.is_positional_synthetic(synthetic)
    assert aa.positional_synthetic_source_pos(synthetic) == (17, 0)
    assert aa.positional_synthetic_source_pos("@positional_invalid") is None
    assert aa.positional_display_label(synthetic) == "Apply rotary emb"
    assert aa.positional_display_label("") == ""

    functional = aa.functional_synthetic_attr("scaled_dot_product_attention")
    assert aa.is_functional_synthetic(functional)
    assert aa.functional_display_label(functional) == "ScaledDotProductAttention"
    assert aa.first_functional_synthetic_index(["x", functional]) == 1
    assert aa.first_functional_synthetic_index(["x"]) is None

    assert aa.combine_op_from_step_details(["note", "combine: Σ"]) == "Σ"
    assert aa.combine_op_from_step_details(["combine: "]) is None
    assert aa.combine_op_from_step_details(None) is None
    assert aa.operation_display_label("", class_name="Multiply") == "Multiply"
    assert aa.operation_display_label("") == "Op"
    assert aa.classify_matmul_label(external_inputs=["weight"]) == "Linear"
    assert aa.classify_matmul_label(external_inputs=[]) == "MatMul"

    roles = {
        aa._classify_role("attn_norm", "Odd"): "norm",
        aa._classify_role("experts", "Odd"): "moe",
        aa._classify_role("tokens", "Embedding"): "other",
        aa._classify_role("lm_head", "Linear"): "head",
        aa._classify_role("rope", "Odd"): "positional",
    }
    assert set(roles.values()) == {"norm", "moe", "other", "head", "positional"}
    assert aa.ffn_role_for_class("mlp", "SparseMoeBlock") == "moe"
    assert aa.ffn_role_for_class("anything", "GatedMLP") == "ffn"
    assert aa.displays_as_pointwise_leaf("proj", "Linear")
    assert aa.displays_as_pointwise_leaf("act", "GELU")
    assert not aa.displays_as_pointwise_leaf("block", None)

    for kernel in ("sdpa", "flash_attn_func", "attention_interface"):
        assert aa.is_standard_attention_kernel(kernel)
    assert not aa.is_standard_attention_kernel(None)
    assert aa.is_torch_native_attention_kernel("torch.nn.attention.flex_attention")
    assert not aa.is_torch_native_attention_kernel("xformers_attention")
    assert aa.attention_kernel_label(["kernel: custom_delta"]) == "custom_delta"
    assert aa.attention_kernel_label([]) == "Attention"
    assert aa.attention_kernel_details(["kernel: eager_attention_forward"]) == []
    assert aa.attention_kernel_details(["kernel: custom"], {"q": ["q_proj"]}) == [
        "kernel: custom",
        "inputs: q",
    ]
    assert aa.kernel_kwarg_ports(
        ["bad", "kwarg: x", "kwarg: q=query", "kwarg: scale=self.scale"]
    ) == {"q": "query"}
    assert aa.tensor_input_label_order(
        ["kwarg: value=v", "kwarg: broken"], {"q": [], "value": []}
    ) == ["value", "q"]


def test_forward_operation_extractor_covers_expressions_and_conditions():
    func = _function(
        """
def forward(self, x, weight, index, flag):
    a = torch.matmul(x, self.weight)
    b = (a + x).reshape(2, -1).float()
    if self.enabled:
        c = b.sum(dim=1, keepdim=True)
    else:
        c = b.gather(1, index)
    c += F.linear(x.to(torch.float16), weight)
    d = c if flag else [a, b][-1]
    d.scatter_(1, index, x)
    return a, d
"""
    )
    analysis = aa._forward_operations_from_forward(
        func,
        self_values={"enabled": aa._UNKNOWN},
        all_tensor_ops=True,
    )
    labels = [op.label for op in analysis.operations]
    assert {"Linear", "Add", "Reshape", "Cast", "Sum", "Gather", "Scatter"} <= set(
        labels
    )
    assert any("condition: self.enabled" in op.details for op in analysis.operations)
    assert any(
        "condition: not (self.enabled)" in op.details for op in analysis.operations
    )
    assert any("dtype: torch.float32" in op.details for op in analysis.operations)
    assert analysis.return_order == ["a", "d"]
    assert analysis.primary_return_slot == "d"
    assert {"weight", "index", "flag"} & set().union(
        *(set(op.param_inputs) for op in analysis.operations)
    )

    compact = aa._forward_operations_from_forward(
        _function("def forward(self, x):\n    return x.reshape(1, -1)\n"),
        self_values={},
        all_tensor_ops=False,
    )
    assert compact.operations == []


def test_config_evaluation_and_assignment_shapes():
    expr = lambda text: ast.parse(text, mode="eval").body
    config = {"kind": "fast", "depth": 4, "enabled": True}
    values = {"copy": 4}
    assert aa._config_value(expr("config.depth"), config, values) == 4
    assert aa._config_value(expr("self.copy"), config, values) == 4
    assert aa._config_value(expr("getattr(config, 'missing', 9)"), config, values) == 9
    assert aa._config_value(expr("not config.enabled"), config, values) is False
    assert aa._config_value(expr("config.enabled and config.depth > 2"), config, values)
    assert aa._config_value(expr("config.missing or True"), config, values) is True
    assert aa._config_value(expr("config.kind != 'slow'"), config, values)
    assert aa._config_value(expr("unknown"), config, values) is aa._UNKNOWN

    assert aa._assignment_class_names(expr("A() if flag else B()")) == ["A", "B"]
    assert aa._assignment_class_names(expr("[Layer() for _ in items]")) == ["Layer"]
    assert aa._assignment_class_names(expr("nn.ModuleList([A(), B()])")) == ["A", "B"]
    assert aa._assignment_class_names(expr("torch.nn.Parameter(x)")) == []
    assert aa._assignment_class_name(expr("getattr(pkg, name)()")) is None
    assert (
        aa._activation_registry_class_name(
            expr("ACT2FN[config.hidden_act]"), {"hidden_act": "gelu_pytorch_tanh"}
        )
        == "GELU"
    )


def test_analyze_rich_synthetic_model_end_to_end():
    source = """
try:
    from kernels.ops import custom_attention as kernel
except ImportError:
    from fallback.ops import custom_attention as kernel

def apply_rotary_emb(x, freqs, inverse=False):
    return x

class FancyAttention:
    def __init__(self):
        self.q_proj = Linear()
        self.k_proj = Linear()
        self.v_proj = Linear()
    def forward(self, hidden_states, mask):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = apply_rotary_emb(q, mask)
        k = apply_rotary_emb(k, mask, inverse=True)
        return kernel(q=q, k=k, v=v, causal=True, scale=self.scale)

class SparseMoeBlock:
    def __init__(self, config):
        self.gate = Linear()
        self.experts = ModuleList([Expert() for _ in range(config.num_experts)])
    def moe_infer(self, x, weights):
        return (x * weights).sum(dim=1)
    def forward(self, hidden_states):
        scores = self.gate(hidden_states)
        expert = self.experts[0]
        routed = expert(hidden_states)
        return self.moe_infer(routed, scores)

class DecoderLayer:
    def __init__(self, config, layer_idx):
        self.input_layernorm = RMSNorm()
        self.self_attn = FancyAttention()
        self.mlp = SparseMoeBlock() if layer_idx > 0 else GatedMLP()
    def forward(self, hidden_states, mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask)
        hidden_states += self.mlp(residual)
        return hidden_states

class Transformer:
    def __init__(self, config):
        self.embed_tokens = Embedding()
        self.layers = ModuleList([DecoderLayer(config, i) for i in range(config.depth)])
        self.rotary_emb = RotaryEmbedding()
        self.norm = RMSNorm()
        self.lm_head = Linear()
    def forward(self, input_ids):
        return self.lm_head(self.norm(self.embed_tokens(input_ids)))
"""
    analysis = aa.analyze_source(
        source,
        filename="synthetic.py",
        config={"num_experts": 8, "depth": 3},
        all_tensor_ops=True,
    )
    assert analysis.decoder_class == "DecoderLayer"
    assert analysis.model_class == "Transformer"
    assert analysis.decoder_type == "Sparse MoE"
    assert analysis.norm_type == "RMSNorm"
    assert analysis.norm_placement == "Pre-Norm"
    assert analysis.positional_helpers == ["apply_rotary_emb"]
    assert analysis.external_imports["kernel"].endswith("#custom_attention")
    assert [component.role for component in analysis.stack_pre] == [
        "embedding",
        "positional",
    ]
    assert [component.role for component in analysis.stack_tail] == ["norm", "head"]
    assert (
        analysis.layer_repeat_lines[0] == "N × DecoderLayer (i in range(config.depth))"
    )

    attention = analysis.class_registry["FancyAttention"]
    assert aa.SYNTHETIC_ATTENTION in attention.forward_calls
    assert set(attention.attention_inputs) == {"q", "k", "v"}
    assert any(
        line.endswith("#custom_attention")
        for line in attention.forward_step_details[aa.SYNTHETIC_ATTENTION]
        if line.startswith("import:")
    )
    assert any(
        details == ["inverse rotation"]
        for details in attention.forward_step_details.values()
    )
    moe = analysis.class_registry["SparseMoeBlock"]
    assert moe.init_assignments["experts"] == "Expert"
    assert moe.forward_step_details["moe_infer"][-1] == "combine: MoE aggregation"

    merged = aa.analyze_sources(
        {Path("model.py"): source, Path("empty.py"): "class Utility: pass"},
        config={"depth": 2},
    )
    assert merged.decoder_class == "DecoderLayer"
    assert {"model.py", "empty.py"} == set(merged.source_files)


def test_registry_finalization_refines_tuple_return_dependencies():
    source = """
class SparseMoeGate:
    def forward(self, x, weight):
        left = F.linear(x, weight)
        right = left.softmax(dim=-1)
        return left, right
class Parent:
    def __init__(self):
        self.split = SparseMoeGate()
        self.proj = Linear()
    def forward(self, x, weight):
        left, right = self.split(x, weight)
        return self.proj(right) + left
"""
    registry = aa.build_class_registry(source)
    aa.finalize_class_registry(registry)
    split = registry["SparseMoeGate"]
    parent = registry["Parent"]
    assert split.referenced_return_producers == set(split.forward_return_slots.values())
    add = next(op for op in parent.forward_operations.values() if op.label == "Add")
    assert set(add.predecessors) & set(split.forward_return_slots.values())


def test_graph_node_helpers_outputs_frames_and_pruning():
    a = _node("a", is_basic=True)
    b = _node("b", is_basic=True)
    root = _node("root", class_name="Pipeline", children=[a, b], input_label="tokens")
    graph = cg.build_computation_graph(root)
    assert graph.nodes[0].label == "tokens"
    assert graph.links == [(0, 1), (1, 2)]
    assert cg.add_forward_output(graph, label="Result") == 3
    assert graph.links[-1] == (2, 3)
    cg.add_root_pipeline_frame(graph, root)
    assert graph.inline_frames[-1].node_indices == [1, 2]

    excluded = cg.ComputationGraph(
        nodes=[cg.GraphNodeSpec(key="only", block=a)],
        excluded_output_indices={0},
    )
    assert cg.add_forward_output(excluded) is None
    assert cg.add_forward_output(cg.ComputationGraph()) is None

    cyclic = cg.ComputationGraph(
        nodes=[
            cg.GraphNodeSpec(key="a", block=a),
            cg.GraphNodeSpec(key="drop", block=b),
            cg.GraphNodeSpec(key="c", block=_node("c")),
        ],
        links=[(0, 1), (1, 2), (1, 1)],
        link_port_labels={(0, 1): "gate"},
        excluded_output_indices={1},
        primary_output_index=2,
    )
    pruned = cg._prune_computation_nodes(cyclic, {1})
    assert len(pruned.nodes) == 2
    assert pruned.links == [(0, 1)]
    assert pruned.link_port_labels == {(0, 1): "gate"}
    assert pruned.primary_output_index == 1


def test_graph_operation_edges_conditions_and_dead_code():
    left = _node("left", forward_order=1)
    right = _node("right", forward_order=2)
    op = _node(
        "@op_l3_c0_add",
        class_name="Add",
        operation_predecessors=["left", "right", aa.FORWARD_METHOD_INPUT],
        forward_order=3,
    )
    root = _node(
        "root",
        class_name="Ops",
        children=[left, right, op],
        primary_output_step=op.attr_name,
        multi_return_module=True,
    )
    graph = cg.build_computation_graph(root, strip_unused_return_branches=True)
    target = next(i for i, spec in enumerate(graph.nodes) if spec.block is op)
    sources = {source for source, dest in graph.links if dest == target}
    assert len(sources) == 3
    assert graph.primary_output_index == target

    conditional = cg.ComputationGraph(
        nodes=[
            cg.GraphNodeSpec(
                key="if",
                block=_node(
                    "@op_l1_c0_add",
                    details=["condition: enabled"],
                    operation_predecessors=["x"],
                ),
            ),
            cg.GraphNodeSpec(
                key="else",
                block=_node(
                    "@op_l2_c0_sub",
                    details=["condition: not (enabled)"],
                    operation_predecessors=["x"],
                ),
            ),
        ]
    )
    assert cg._live_node_indices_to_fixpoint(conditional, [1]) == {0, 1}
    assert conditional.links == [(0, 1)]


def test_segment_building_with_manual_block_graphs(monkeypatch):
    producer = _node("router", role="router")
    consumer = _node("consume", details=["method `consume()`"])
    side = aa.SideInputSpec("weights", "router", ["router"])
    root = _node("root", class_name="Manual", children=[producer, consumer])

    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _root: [
            SeqSegment(producer),
            SideFeedSegment(
                consumer=consumer,
                sides=[side],
                side_producer_nodes={"router": producer},
            ),
            SideCombineSegment(consumer, [side], "+"),
        ],
    )
    graph = cg.build_computation_graph(root)
    assert {"router", "consume"} <= {
        spec.block.attr_name for spec in graph.nodes if spec.block
    }
    assert "Add" in {spec.label for spec in graph.nodes}
    assert len(graph.links) >= 3

    main = _node("main")
    gate = _node("gate")
    after = _node("after")
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _root: [
            SeqSegment(main),
            CombineSegment(gate, after=[after], side_port_label="gate"),
        ],
    )
    combined = cg.build_computation_graph(_node("root2", children=[main, gate, after]))
    assert "Multiply" in [spec.label for spec in combined.nodes]

    kernel = _node(
        "kernel",
        class_name="KernelPipeline",
        children=[
            _node("qk", class_name="KernelOp"),
            _node(
                "out",
                class_name="KernelOutput",
                kernel_second_operand="input",
            ),
        ],
        tensor_input_labels=["q", "v"],
        tensor_step_targets={"q": "qk", "v": "out"},
    )
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _root: [
            TensorPortsSegment(
                labels=["q", "v", "ignored"],
                targets={"q": "kernel", "v": "kernel", "ignored": "missing"},
                steps=[kernel],
            )
        ],
    )
    port_graph = cg.build_computation_graph(
        _node(
            "ports",
            children=[kernel],
            tensor_input_labels=["q", "v"],
            attention_inputs={"q": ["q_proj"], "v": ["value_norm"]},
        )
    )
    ports = [spec for spec in port_graph.nodes if spec.synthetic == cg.SYNTHETIC_TENSOR]
    assert [spec.label for spec in ports] == ["q", "v"]
    assert [spec.sublabel for spec in ports] == ["← Linear", "← RMSNorm"]


def test_fanout_residual_gated_and_kernel_graph_branches(monkeypatch):
    left = _node("left")
    right = _node("right")
    merge = _node("merge", class_name="Merge")
    residual = _node("residual")
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _root: [
            FanOutSegment(
                [Branch("q", [left]), Branch("k", [right], port_style="inline")],
                merge,
            ),
            ResidualAddSegment(residual, []),
        ],
    )
    graph = cg.build_computation_graph(
        _node("fan", children=[left, right, merge, residual])
    )
    merge_index = next(i for i, spec in enumerate(graph.nodes) if spec.block is merge)
    assert len([link for link in graph.links if link[1] == merge_index]) == 2
    assert "Add" in {spec.label for spec in graph.nodes}

    gate = _node("gate")
    up = _node("up")
    activation = _node("act", class_name="Activation", children=[_node("inner")])
    situ = _node("situ", class_name="SituActivation")
    down = _node("down")
    gated = _node("gated", class_name="SituAndMul", children=[gate, up, down])
    import visualizer.block_tree as block_tree

    original_parts = block_tree._situ_gated_mlp_parts
    monkeypatch.setattr(
        block_tree,
        "_situ_gated_mlp_parts",
        lambda _node: (gate, up, activation, situ, down),
    )
    gated_graph = cg.ComputationGraph()
    input_index = cg._add_node(
        gated_graph, key=cg.SYNTHETIC_INPUT, synthetic=cg.SYNTHETIC_INPUT
    )
    indices, tail = cg._add_situ_gated_mlp_chain(
        gated_graph,
        gated,
        key_prefix="gated",
        input_index=input_index,
        create_outer_frame=True,
    )
    assert len(indices) == 5
    assert gated_graph.nodes[tail].block is down
    multiply = next(i for i, spec in enumerate(gated_graph.nodes) if spec.label == "×")
    assert len([link for link in gated_graph.links if link[1] == multiply]) == 2

    pipeline = _node(
        "pipeline",
        class_name="KernelPipeline",
        children=[_node("inner_a"), _node("inner_b")],
    )
    output = _node(
        "output",
        class_name="KernelOutput",
        kernel_predecessors=["inner_b"],
    )
    kernel_graph = cg.ComputationGraph()
    attr_indices: dict[str, int] = {}
    merged, merged_tail = cg._add_kernel_pipeline_merge_chain(
        kernel_graph,
        [pipeline, output],
        key_prefix="kernel",
        attr_last_index=attr_indices,
    )
    assert len(merged) >= 3
    assert kernel_graph.nodes[merged_tail].block is output
    fallback, fallback_tail = cg._add_kernel_pipeline_merge_chain(
        kernel_graph, [left], key_prefix="fallback"
    )
    assert fallback and fallback_tail is not None

    monkeypatch.setattr(block_tree, "_situ_gated_mlp_parts", original_parts)
    norm = _node("norm", class_name="FusedRMSNormGated")
    side = aa.SideInputSpec("gate", "gate", [], source_kind="forward_input")
    monkeypatch.setattr(cg, "is_gated_norm_module", lambda node: node is norm)
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _root: [SideFeedSegment(norm, [side])],
    )
    norm_graph = cg.build_computation_graph(_node("norm_root", children=[norm]))
    assert any(spec.block is norm for spec in norm_graph.nodes)
    assert "Multiply" in {spec.label for spec in norm_graph.nodes}


@pytest.mark.parametrize(
    ("label", "expected"),
    [("+", "Add"), ("×", "Multiply"), ("*", "Multiply"), ("ƒ", "Function"), ("Σ", "Σ")],
)
def test_operation_tile_labels(label, expected):
    assert cg._operation_tile_label(label) == expected
