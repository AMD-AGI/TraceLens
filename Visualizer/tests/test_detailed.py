"""Tests for detailed block tree expansion."""

from pathlib import Path

from visualizer.ast_analyze import analyze_source
from visualizer.basic_ops import BasicOpFilter, introspect_is_modeling_operation, resolve_is_basic
from visualizer.block_tree import (
    build_decoder_block_trees,
    collect_computation_segments,
    is_method_wrapper,
    wrapper_bullet,
    wrapper_module_comment,
    wrapper_panel_line,
    wrapper_skips_comment,
)
from visualizer.extract import load_architecture
from visualizer.render import render_diagram


FIXTURES = Path(__file__).parent / "fixtures"


def test_basic_op_filter_defaults_and_cli_overrides():
    default = BasicOpFilter.from_cli()
    assert default.is_basic("torch.ops.aten.add")
    assert default.is_basic("aten.mm")
    assert not default.is_basic("CustomLatentAttention")

    with_linear = BasicOpFilter.from_cli(add=[r"(?i)^Linear$"])
    assert with_linear.is_basic("Linear")

    without_aten = BasicOpFilter.from_cli(remove=[r"(?i)aten\."])
    assert not without_aten.is_basic("aten.mm")


def test_introspect_modeling_operations_are_not_basic():
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    assert introspect_is_modeling_operation("ShortConvolution", "q_conv1d")
    assert introspect_is_modeling_operation("AttentionMerge", "@attn_merge", ["ports: Q,K,V"])
    assert introspect_is_modeling_operation("KernelPipeline", "@attn_pipeline")
    assert introspect_is_modeling_operation("KernelOutput", "@attn_output", ["kernel: chunk_kda"])
    assert not introspect_is_modeling_operation("Linear", "g_proj")
    assert resolve_is_basic("Linear", "g_proj", basic)
    assert not introspect_is_modeling_operation("Linear", "q_proj")
    assert resolve_is_basic("Linear", "q_proj", basic)
    assert not resolve_is_basic("ShortConvolution", "q_conv1d", basic)
    assert not resolve_is_basic("AttentionMerge", "@attn_merge", basic, details=["ports: Q,K,V"])
    assert resolve_is_basic("Linear", "@functional_linear", basic)


def test_kda_shortconv_and_substeps_are_not_basic_ops():
    from pathlib import Path
    from visualizer.ast_analyze import SYNTHETIC_ATTENTION, analyze_source
    from visualizer.block_tree import block_purpose, build_block_node, is_simple_modeled_tile

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    conv_steps = [child for child in attn.children if child.class_name == "ShortConvolution"]
    assert len(conv_steps) == 3
    assert all(not child.is_basic for child in conv_steps)
    assert all(is_simple_modeled_tile(child) for child in conv_steps)
    assert all(child.label == "Depthwise Conv" for child in conv_steps)
    assert all(block_purpose(child) is None for child in conv_steps)
    act_steps = [child for child in attn.children if child.class_name == "ActivationOp" and child.attr_name.endswith("_activation")]
    assert len(act_steps) == 3
    assert all(is_simple_modeled_tile(step) for step in act_steps)
    from visualizer.sizing import block_sublabel

    assert all(block_sublabel(step) is None for step in act_steps)
    for conv, act in zip(conv_steps, act_steps, strict=True):
        assert act.attr_name == f"{conv.attr_name}_activation"

    pipeline = next(child for child in attn.children if child.class_name == "KernelPipeline")
    output_step = next(child for child in attn.children if child.class_name == "KernelOutput")
    assert is_simple_modeled_tile(output_step)
    assert not is_simple_modeled_tile(pipeline)
    assert len(pipeline.children) >= 3

    linears = [child for child in attn.children if child.class_name == "Linear" and child.is_basic]
    assert linears, "projections should remain basic Linear tiles"


def test_upstream_input_sources_pair_conditional_variants():
    from visualizer.blocks import BlockComponent, upstream_input_sources

    components = [
        BlockComponent("input_layernorm", "RMSNorm", "norm", "RMSNorm", forward_order=0),
        BlockComponent("self_attn", "KimiMLAAttention", "attention", "MLA", forward_order=1),
        BlockComponent("post_attention_layernorm", "RMSNorm", "norm", "RMSNorm", forward_order=2),
        BlockComponent("block_sparse_moe", "KimiSparseMoeBlock", "moe", "KimiSparseMoeBlock", forward_order=3),
        BlockComponent("mlp", "KimiMLP", "ffn", "KimiMLP", forward_order=3),
    ]
    sources = upstream_input_sources(components)
    assert sources["self_attn"] == "RMSNorm"
    assert sources["block_sparse_moe"] == "RMSNorm"
    assert sources["mlp"] == "RMSNorm"


def test_input_source_uses_ast_operation_label_not_attr_name():
    from visualizer.blocks import BlockComponent, input_sources_from_forward_sequence

    components = [
        BlockComponent("input_layernorm", "KimiRMSNorm", "norm", "RMSNorm", forward_order=0),
        BlockComponent("self_attn", "KimiMLAAttention", "attention", "MLA", forward_order=1),
        BlockComponent("post_attention_layernorm", "KimiRMSNorm", "norm", "RMSNorm", forward_order=2),
        BlockComponent("block_sparse_moe", "KimiSparseMoeBlock", "moe", "KimiSparseMoeBlock", forward_order=3),
    ]
    forward_sequence = [
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "block_sparse_moe",
    ]
    sources = input_sources_from_forward_sequence(components, forward_sequence)
    assert sources["self_attn"] == "RMSNorm"
    assert sources["block_sparse_moe"] == "RMSNorm"
    assert "layernorm" not in sources["self_attn"].lower()
    assert "layernorm" not in sources["block_sparse_moe"].lower()


def test_build_decoder_block_trees_for_custom_model():
    source = (FIXTURES / "custom_model" / "modeling_custom.py").read_text(encoding="utf-8")
    analysis = analyze_source(source, filename="modeling_custom.py")
    basic_ops = BasicOpFilter.from_cli(add=[r"(?i)^Linear$"])

    trees = build_decoder_block_trees(
        analysis.block_components, analysis.class_registry, basic_ops
    )
    titles = [title for title, _ in trees]
    assert any("MLA" in title or "Latent" in title for title in titles)
    assert any("MoE" in title for title in titles)

    attn_tree = next(tree for title, tree in trees if "MLA" in title or "Latent" in title)
    # CustomLatentAttention has no forward() — expanded as a leaf, not init internals.
    assert attn_tree.is_basic
    assert "Latent" in attn_tree.class_name or "MLA" in attn_tree.label


def test_forward_extraction_preserves_swiglu_order():
    import ast

    from visualizer.ast_analyze import _parse_forward

    code = '''
class MLP:
    def forward(self, x):
        if self.config.hidden_act == "situ":
            gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
            down_proj = self.down_proj(self.act_fn(gate_up))
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
'''
    func = ast.parse(code).body[0].body[0]
    calls, _, _, _, _ = _parse_forward(func)
    assert calls == ["gate_proj", "up_proj", "act_fn", "down_proj"]


def test_forward_extraction_nested_mla_path():
    import ast

    from visualizer.ast_analyze import _parse_forward

    code = '''
class Attn:
    def forward(self, hidden_states):
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(1, 2)
        attn_output, _ = attention_interface(self, q_states, k_pass, k_pass, None)
        return self.o_proj(attn_output)
'''
    func = ast.parse(code).body[0].body[0]
    calls, _, attention_inputs, _, _ = _parse_forward(func)
    assert calls[:3] == ["q_a_proj", "q_a_layernorm", "q_b_proj"]
    assert "kv_a_proj_with_mqa" in calls
    assert calls.index("kv_a_layernorm") < calls.index("kv_b_proj")
    assert "@attention" in calls
    assert "o_proj" in calls
    assert calls.index("@attention") < calls.index("o_proj")
    assert "q_states" in attention_inputs
    assert attention_inputs["q_states"][:2] == ["q_a_proj", "q_a_layernorm"]


def test_collect_computation_segments_merges_qkv_branches():
    from visualizer.ast_analyze import SYNTHETIC_ATTENTION
    from visualizer.block_tree import BlockNode, FanOutSegment, SeqSegment

    def leaf(name: str) -> BlockNode:
        return BlockNode(attr_name=name, class_name="Linear", role="other", label="Linear", is_basic=True)

    root = BlockNode(
        attr_name="attn",
        class_name="Attn",
        role="attention",
        label="Attn",
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
            ),
            leaf("o_proj"),
        ],
    )
    segments = collect_computation_segments(root)
    assert isinstance(segments[0], FanOutSegment)
    assert segments[0].branches[0].label == "q"
    assert segments[0].branches[1].label == "kv"
    assert isinstance(segments[-1], SeqSegment)
    assert segments[-1].step.attr_name == "o_proj"


def test_mla_attention_inputs_from_kimi_style_forward():
    import ast

    from visualizer.ast_analyze import _parse_forward

    code = '''
class Attn:
    def forward(self, hidden_states):
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(1).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [1, 1], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [1, 1], dim=-1)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(1).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [1, 1], dim=-1)
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)
        attn_output, _ = attention_interface(self, query_states, key_states, value_states, None)
        return self.o_proj(attn_output)
'''
    func = ast.parse(code).body[0].body[0]
    _, _, attention_inputs, _, _ = _parse_forward(func)
    assert attention_inputs["query_states"] == ["q_a_proj", "q_a_layernorm", "q_b_proj"]
    assert attention_inputs["key_states"] == ["kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj"]
    assert attention_inputs["value_states"] == ["kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj"]


def test_tuple_map_assign_preserves_distinct_provenance():
    import ast

    from visualizer.ast_analyze import _parse_forward

    code = '''
class Attn:
    def forward(self, hidden_states):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        q, k = map(lambda x: x.transpose(1, 2), (q, k))
        return attention_kernel(q=q, k=k, v=k)
'''
    func = ast.parse(code).body[0].body[0]
    _, _, attention_inputs, _, _ = _parse_forward(func)
    assert attention_inputs["q"] == ["q_proj"]
    assert attention_inputs["k"] == ["k_proj"]


def test_kernel_merge_detects_multi_input_attention():
    import ast

    from visualizer.ast_analyze import SYNTHETIC_ATTENTION, _parse_forward
    from visualizer.block_tree import FanOutSegment, build_block_node, collect_computation_segments
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.computation_graph import build_computation_graph

    code = '''
import torch.nn as nn

class LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.g_proj = nn.Linear(8, 8)
        self.beta_proj = nn.Linear(8, 8)
        self.out_gate = nn.Linear(8, 8)
        self.o_norm = nn.LayerNorm(8)
        self.o_proj = nn.Linear(8, 8)

    def forward(self, hidden_states):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)
        beta = self.beta_proj(hidden_states)
        attn_out = delta_attn_kernel(q=q, k=k, v=v, g=g, beta=beta)
        gate = self.out_gate(hidden_states)
        normed = self.o_norm(attn_out, gate)
        return self.o_proj(normed)
'''
    from visualizer.ast_analyze import analyze_source

    analysis = analyze_source(code, filename="linear_attn.py")
    cls = analysis.class_registry["LinearAttention"]
    assert SYNTHETIC_ATTENTION in cls.forward_calls
    assert cls.attention_inputs["q"] == ["q_proj"]
    assert cls.attention_inputs["k"] == ["k_proj"]
    assert cls.attention_inputs["v"] == ["v_proj"]

    tree = build_block_node(
        attr_name="attn",
        class_name="LinearAttention",
        registry=analysis.class_registry,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    segments = collect_computation_segments(tree)
    assert isinstance(segments[0], FanOutSegment)
    assert {branch.label for branch in segments[0].branches} >= {"q", "k", "v"}

    graph = build_computation_graph(tree)
    merge_index = next(
        i
        for i, node in enumerate(graph.nodes)
        if node.block and node.block.class_name in {"KernelPipeline", "KernelOp", "KernelOutput", "AttentionOp"}
        and len([src for src, dst in graph.links if dst == i]) >= 3
    )
    incoming = [src for src, dst in graph.links if dst == merge_index]
    assert len(incoming) >= 3
    o_norm_index = next(i for i, node in enumerate(graph.nodes) if node.block and node.block.attr_name == "o_norm")
    side_links = [src for src, dst in graph.links if dst == o_norm_index and (src, dst) in graph.side_entry_links]
    assert side_links


def test_kimi_moe_gate_parses_functional_linear():
    import ast

    from visualizer.ast_analyze import (
        SYNTHETIC_FUNCTIONAL_LINEAR,
        SYNTHETIC_ROUTER_ACTIVATION,
        SYNTHETIC_ROUTER_TOPK,
        _parse_forward,
        _router_forward_step_details,
    )

    code = '''
class KimiMoEGate(nn.Module):
    def forward(self, hidden_states):
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        _, topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
        return topk_idx, topk_weight
'''
    cls = ast.parse(code).body[0]
    func = cls.body[0]
    calls, _, _, _, _ = _parse_forward(func)
    details = _router_forward_step_details(cls.name, func, calls)
    assert SYNTHETIC_FUNCTIONAL_LINEAR in calls
    assert SYNTHETIC_ROUTER_ACTIVATION in calls
    assert SYNTHETIC_ROUTER_TOPK in calls
    assert details[SYNTHETIC_ROUTER_ACTIVATION] == ["Sigmoid"]


def test_functional_ops_render_as_basic_names_without_parentheses():
    from visualizer.ast_analyze import (
        functional_display_label,
        functional_synthetic_attr,
        is_functional_synthetic,
    )
    from visualizer.block_tree import build_block_node, tile_purpose_annotation
    from visualizer.ast_analyze import analyze_source

    assert functional_synthetic_attr("linear") == "@functional_linear"
    assert functional_synthetic_attr("softmax") == "@functional_softmax"
    assert functional_display_label("linear") == "Linear"
    assert functional_display_label("@functional_softmax") == "Softmax"
    assert is_functional_synthetic("@functional_linear")

    code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Gate(nn.Module):
    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight, None)
        return logits.sigmoid()
'''
    analysis = analyze_source(code, filename="gate.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$"])
    tree = build_block_node(
        attr_name="gate",
        class_name="Gate",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    linear = next(child for child in tree.children if is_functional_synthetic(child.attr_name))
    assert linear.label == "Linear"
    assert linear.class_name == "Linear"
    assert not linear.details
    assert tile_purpose_annotation(linear) is None


def test_detail_diagram_omits_tile_purpose_annotations(tmp_path: Path):
    from visualizer.extract import load_architecture
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.render import render_diagram

    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    out = render_diagram(spec, tmp_path / "no_purpose.svg", detailed=True)
    svg = out.read_text(encoding="utf-8")
    for unwanted in (
        "cos, sin from inv_freq",
        "rotate query/key tensors",
        "kernel pipeline ·",
        "Expert bias",
        "Group routing",
        "Sigmoid(linear out)",
    ):
        assert unwanted not in svg


def test_moe_graph_keeps_nested_composite_blocks():
    from visualizer.block_tree import BlockNode, build_block_node, _is_composite_block
    from visualizer.ast_analyze import analyze_source
    from visualizer.computation_graph import build_computation_graph

    code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Gate(nn.Module):
    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight, None)
        return logits.sigmoid(), None

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16)
        self.up_proj = nn.Linear(8, 16)
        self.down_proj = nn.Linear(16, 8)
    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))

class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = Gate()
        self.down = nn.Linear(8, 8)
        self.shared_experts = MLP()
    def forward(self, hidden_states):
        scores, _ = self.gate(hidden_states)
        out = self.down(scores)
        return out + self.shared_experts(hidden_states)
'''
    analysis = analyze_source(code, filename="moe.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$"])
    tree = build_block_node(
        attr_name="moe",
        class_name="MoE",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(tree)
    top_labels = [spec.label for spec in graph.nodes]
    assert "Linear" in top_labels
    assert any(spec.block and spec.block.attr_name == "gate_proj" for spec in graph.nodes)
    assert graph.inline_frames
    assert any("shared_experts" in frame.frame_id for frame in graph.inline_frames)


def test_collect_nested_diagrams_for_moe():
    from visualizer.block_tree import build_block_node, collect_nested_diagrams
    from visualizer.ast_analyze import analyze_source

    code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Gate(nn.Module):
    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight, None)
        return logits.sigmoid(), None

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16)
        self.up_proj = nn.Linear(8, 16)
        self.down_proj = nn.Linear(16, 8)
    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))

class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = Gate()
        self.down = nn.Linear(8, 8)
        self.shared_experts = MLP()
    def forward(self, hidden_states):
        scores, _ = self.gate(hidden_states)
        out = self.down(scores)
        return out + self.shared_experts(hidden_states)
'''
    analysis = analyze_source(code, filename="moe.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$"])
    tree = build_block_node(
        attr_name="moe",
        class_name="MoE",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    nested = collect_nested_diagrams(tree)
    titles = [title for title, _ in nested]
    assert not any("gate" in title for title in titles)
    assert not any("shared_experts" in title for title in titles)


def test_build_computation_graph_mla_structure():
    from visualizer.ast_analyze import SYNTHETIC_ATTENTION
    from visualizer.block_tree import BlockNode
    from visualizer.computation_graph import SYNTHETIC_INPUT, SYNTHETIC_MULTIPLY, build_computation_graph

    def leaf(name: str) -> BlockNode:
        return BlockNode(attr_name=name, class_name="Linear", role="other", label="Linear", is_basic=True)

    root = BlockNode(
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
            ),
            leaf("g_proj"),
            leaf("o_proj"),
        ],
    )
    graph = build_computation_graph(root)
    labels = {spec.label for spec in graph.nodes}
    assert "hidden_states" in labels
    assert "×" in labels
    assert len(graph.links) >= 10
    assert any(spec.synthetic == SYNTHETIC_INPUT for spec in graph.nodes)
    assert any(spec.synthetic == SYNTHETIC_MULTIPLY for spec in graph.nodes)
    gate_spec = next(spec for spec in graph.nodes if spec.block and spec.block.attr_name == "g_proj")
    assert gate_spec.port_label == "Linear"
    input_index = next(i for i, spec in enumerate(graph.nodes) if spec.synthetic == SYNTHETIC_INPUT)
    q_head = next(i for i, spec in enumerate(graph.nodes) if spec.block and spec.block.attr_name == "q_a_proj")
    kv_head = next(i for i, spec in enumerate(graph.nodes) if spec.block and spec.block.attr_name == "kv_a_proj_with_mqa")
    assert (input_index, q_head) in graph.links
    assert (input_index, kv_head) in graph.links
    gate_index = next(i for i, spec in enumerate(graph.nodes) if spec.block and spec.block.attr_name == "g_proj")
    mult_index = next(i for i, spec in enumerate(graph.nodes) if spec.synthetic == SYNTHETIC_MULTIPLY)
    assert (input_index, gate_index) in graph.links
    assert (input_index, gate_index) not in graph.dashed_links
    assert (gate_index, mult_index) in graph.side_entry_links
    assert (gate_index, mult_index) not in graph.dashed_links


def test_dashed_path_draws_each_segment():
    from unittest.mock import MagicMock, patch

    from visualizer.render import _draw_path

    ax = MagicMock()
    points = [(1.0, 5.0), (1.0, 3.0), (4.0, 3.0)]
    with patch("visualizer.render._line") as line_mock:
        _draw_path(ax, points, linestyle="dashed")
    assert line_mock.call_count == 2
    assert line_mock.call_args_list[0].args[1:5] == (1.0, 5.0, 1.0, 3.0)
    assert line_mock.call_args_list[1].args[1:5] == (1.0, 3.0, 4.0, 3.0)


def test_side_entry_combine_connector_enters_multiply_side():
    from visualizer.render import MERGE_RADIUS, _RenderAnchor, _side_entry_combine_connector_points

    gap = 0.04
    gate = _RenderAnchor(cx=7.8, top=19.4, bottom=19.0, left=7.2, right=8.4)
    target_cx = 5.2
    target_cy = 17.3
    points = _side_entry_combine_connector_points(gate, target_cx, target_cy, gap=gap)
    assert points[-1] == (target_cx + MERGE_RADIUS, target_cy)
    assert points[0] == (gate.cx, gate.bottom)


def test_top_entry_combine_connector_enters_operator_top():
    from visualizer.render import MERGE_RADIUS, _RenderAnchor, _top_entry_combine_connector_points

    gap = 0.04
    linear = _RenderAnchor(cx=5.2, top=19.4, bottom=19.0, left=4.6, right=5.8)
    target_cx = 5.2
    target_cy = 17.3
    points = _top_entry_combine_connector_points(linear, target_cx, target_cy, gap=gap)
    assert points[-1] == (target_cx, target_cy + MERGE_RADIUS)
    assert points[0] == (linear.cx, linear.bottom)


def test_inline_block_frame_label_uses_attr_name():
    from visualizer.block_tree import BlockNode, inline_block_frame_label

    gate = BlockNode(
        attr_name="g_proj",
        class_name="OutputGate",
        role="gate",
        label="Output gate",
        is_basic=False,
        children=[
            BlockNode(attr_name="@gate_linear", class_name="Linear", role="other", label="Linear", is_basic=True),
        ],
    )
    linear = BlockNode(attr_name="q_proj", class_name="Linear", role="other", label="Linear", is_basic=True)
    assert inline_block_frame_label(gate) == "g_proj"
    assert inline_block_frame_label(linear) == "q_proj"


def test_tile_display_labels_use_operation_name_for_basic_ops():
    from visualizer.block_tree import BlockNode, tile_display_labels

    block = BlockNode(attr_name="o_proj", class_name="Linear", role="other", label="Linear", is_basic=True)
    assert tile_display_labels(block, spec_label="Linear") == ("Linear", None)
    assert tile_display_labels(None, spec_label="Linear") == ("Linear", None)


def test_tile_display_labels_inline_port_on_basic_op_shows_operation_only():
    from visualizer.block_tree import BlockNode, tile_display_labels

    block = BlockNode(attr_name="g_proj", class_name="Linear", role="other", label="Linear", is_basic=True)
    assert tile_display_labels(block, port_label="g", port_style="inline") == ("Linear", None)
    q = BlockNode(attr_name="q_proj", class_name="Linear", role="other", label="Linear", is_basic=True)
    assert tile_display_labels(q, port_label="Q", port_style="inline") == ("Linear", None)


def test_inline_port_node_on_basic_op_uses_single_line_height():
    from visualizer.block_tree import BlockNode
    from visualizer.computation_graph import GraphNodeSpec, _diagram_size_for_rendered_spec
    from visualizer.sizing import estimate_block_size, single_line_box_height

    gate = BlockNode(attr_name="gate", class_name="Linear", role="router", label="Router", is_basic=True)
    spec = GraphNodeSpec(
        key="gate",
        block=gate,
        label="Router",
        port_label="gate",
        port_style="inline",
    )
    _, rendered_h = _diagram_size_for_rendered_spec(spec)
    _, named_h = estimate_block_size("Router")
    assert abs(rendered_h - named_h) < 1e-6
    assert abs(rendered_h - single_line_box_height()) < 1e-6


def test_inline_dashed_port_connector_enters_gate_top():
    from visualizer.render import _RenderAnchor, _inline_dashed_port_connector_points

    gap = 0.04
    source = _RenderAnchor(cx=3.0, top=10.0, bottom=9.5, left=2.5, right=3.5)
    target = _RenderAnchor(cx=5.0, top=8.0, bottom=7.0, left=4.2, right=5.8)
    points = _inline_dashed_port_connector_points(source, target, gap=gap)
    assert points[-1] == (target.cx, target.top)


def test_combine_op_node_matches_merge_circle_size():
    from visualizer.computation_graph import COMBINE_OP_SIZE, SYNTHETIC_MULTIPLY, GraphNodeSpec, _diagram_size_for_spec
    from visualizer.render import COMBINE_OP_SIZE as RENDER_COMBINE_OP_SIZE

    assert COMBINE_OP_SIZE == RENDER_COMBINE_OP_SIZE
    spec = GraphNodeSpec(key="mul", label="×", synthetic=SYNTHETIC_MULTIPLY)
    width, height = _diagram_size_for_spec(spec)
    assert width == height == COMBINE_OP_SIZE


def test_situ_and_mul_matches_parent_mlp_color():
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.render import _detail_block_facecolor

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    moe = build_block_node(
        attr_name="block_sparse_moe",
        class_name="KimiSparseMoeBlock",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    mlp = next(child for child in moe.children if child.attr_name == "shared_experts")
    situ = next(child for child in mlp.children if child.class_name == "SituAndMul")
    assert _detail_block_facecolor(situ) == _detail_block_facecolor(mlp)


def test_situ_and_mul_is_inlined_not_nested():
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import _is_composite_block, build_block_node, collect_nested_diagrams, is_linear_pipeline_block

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.for_detailed()
    mlp = build_block_node(
        attr_name="shared_experts",
        class_name="KimiMLP",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    act_fn = next(child for child in mlp.children if child.attr_name == "act_fn")
    assert _is_composite_block(act_fn)
    assert is_linear_pipeline_block(mlp)
    nested = collect_nested_diagrams(mlp)
    assert not any("SituAndMul" in title for title, _ in nested)


def test_nested_input_source_for_kimi_mlp():
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node, collect_nested_diagrams, is_linear_pipeline_block

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.for_detailed()
    moe = build_block_node(
        attr_name="block_sparse_moe",
        class_name="KimiSparseMoeBlock",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    mlp = next(child for child in moe.children if child.attr_name == "shared_experts")
    assert is_linear_pipeline_block(mlp)
    nested = collect_nested_diagrams(moe)
    assert not any(tree.class_name == "KimiMLP" for _, tree in nested)


def test_format_input_source_sublabel_splits_in_phrase():
    from visualizer.render import _format_input_source_sublabel

    assert _format_input_source_sublabel("Linear in KimiSparseMoeBlock") == "← Linear"
    assert _format_input_source_sublabel("MoE input") == "← MoE input"


def test_layout_computation_graph_returns_positions():
    from visualizer.computation_graph import GraphNodeSpec, ComputationGraph, layout_computation_graph

    graph = ComputationGraph(
        nodes=[
            GraphNodeSpec(key="a", label="A"),
            GraphNodeSpec(key="b", label="B"),
            GraphNodeSpec(key="c", label="C"),
        ],
        links=[(0, 1), (1, 2)],
    )
    positions, _ = layout_computation_graph(graph, cx=4.0, top_y=10.0, block_w=3.0)
    assert len(positions) == 3
    assert positions[0].top_y >= positions[1].top_y >= positions[2].top_y
    left = 4.0 - 3.0 / 2
    right = 4.0 + 3.0 / 2
    for pos in positions:
        assert left <= pos.cx - pos.width / 2
        assert pos.cx + pos.width / 2 <= right


def test_detail_content_bounds_contain_nodes():
    from visualizer.computation_graph import GraphNodeSpec, LayoutPosition
    from visualizer.render import _detail_content_bounds

    positions = [
        LayoutPosition(
            spec=GraphNodeSpec(key="a", label="A", port_label="Q", port_style="floating"),
            cx=3.0,
            top_y=9.0,
            width=1.0,
            height=0.5,
        ),
        LayoutPosition(
            spec=GraphNodeSpec(key="b", label="B"),
            cx=5.0,
            top_y=8.0,
            width=1.0,
            height=0.5,
        ),
    ]
    left, right, bottom, top = _detail_content_bounds(positions)
    for pos in positions:
        assert left <= pos.cx - pos.width / 2
        assert pos.cx + pos.width / 2 <= right
        assert bottom <= pos.bottom
        assert pos.top_y + 0.08 <= top


def test_is_method_wrapper_and_filtering():
    from visualizer.block_tree import BlockNode

    wrapper = BlockNode(
        attr_name="_forward_attn_residual",
        class_name="_forward_attn_residual",
        role="other",
        label="_forward_attn_residual",
        is_basic=True,
        details=["method `_forward_attn_residual()`"],
    )
    assert is_method_wrapper(wrapper)
    assert wrapper_bullet(wrapper) == "forward attn residual (_forward_attn_residual)"
    assert wrapper_skips_comment(wrapper)
    assert wrapper_module_comment(wrapper) is None

    real = BlockNode(attr_name="q_proj", class_name="Linear", role="other", label="Linear", is_basic=True)
    assert not is_method_wrapper(real)


def test_wrapper_module_comment_for_non_excluded_modules():
    from visualizer.block_tree import BlockNode

    moe = BlockNode(
        attr_name="moe_infer",
        class_name="moe_infer",
        role="other",
        label="moe infer",
        is_basic=True,
        details=["method `moe_infer()`"],
    )
    assert not wrapper_skips_comment(moe)
    assert wrapper_module_comment(moe) is None

    gate = BlockNode(attr_name="gate", class_name="Linear", role="router", label="Router", is_basic=True)
    assert "route" in wrapper_module_comment(gate).lower()

    lm_head = BlockNode(attr_name="lm_head", class_name="Linear", role="other", label="lm head", is_basic=True)
    assert "logits" in wrapper_module_comment(lm_head).lower()

    embed = BlockNode(attr_name="embed_tokens", class_name="Embedding", role="embedding", label="Embedding", is_basic=True)
    assert wrapper_module_comment(embed) is None

    tokenization = BlockNode(attr_name="tokenization", class_name="Tokenizer", role="embedding", label="Tokenizer", is_basic=True)
    assert wrapper_module_comment(tokenization) is None

    assert wrapper_panel_line(moe) == wrapper_bullet(moe)
    assert wrapper_panel_line(embed) == wrapper_bullet(embed)


def test_parallel_gate_wrapper_comment_for_g_proj():
    from visualizer.block_tree import BlockNode, wrapper_bullet, wrapper_module_comment, wrapper_panel_line

    g_proj = BlockNode(
        attr_name="g_proj",
        class_name="OutputGate",
        role="gate",
        label="Output gate",
        is_basic=False,
        details=["Linear", "Sigmoid inside o_norm"],
        children=[
            BlockNode(
                attr_name="g_proj",
                class_name="Linear",
                role="other",
                label="Linear",
                is_basic=True,
            )
        ],
    )
    assert wrapper_bullet(g_proj) == "Output gate (g_proj)"
    assert wrapper_module_comment(g_proj) == "Sigmoid inside o_norm"
    assert "Output gate" in wrapper_panel_line(g_proj)
    assert "g_proj" in wrapper_panel_line(g_proj)
    assert "Sigmoid inside o_norm" in wrapper_panel_line(g_proj)


def test_collect_parallel_gate_wrappers_from_mla():
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node, collect_nested_diagrams, collect_parallel_gate_wrappers

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.for_detailed()
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiMLAAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    gates = collect_parallel_gate_wrappers(attn)
    assert gates == []
    nested = collect_nested_diagrams(attn)
    assert not any("g_proj" in title for title, _ in nested)
    gate = next(child for child in attn.children if child.attr_name == "g_proj")
    assert gate.children
    assert any(child.label == "Sigmoid" for child in gate.children)


def test_mla_output_gate_inlined_in_parent_graph():
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import build_computation_graph

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.for_detailed()
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiMLAAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(attn)
    labels = [spec.label for spec in graph.nodes]
    assert "Linear" in labels
    assert "Sigmoid" in labels
    assert any(frame.frame_id == "g_proj" for frame in graph.inline_frames)


def test_mla_output_gate_matches_kimi_mlp_color():
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.render import _detail_block_facecolor

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.for_detailed()
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiMLAAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    moe = build_block_node(
        attr_name="block_sparse_moe",
        class_name="KimiSparseMoeBlock",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    gate = next(child for child in attn.children if child.attr_name == "g_proj")
    mlp = next(child for child in moe.children if child.attr_name == "shared_experts")
    assert _detail_block_facecolor(gate) == _detail_block_facecolor(mlp)


def test_mla_gate_input_uses_solid_residual_connector(tmp_path: Path):
    from pathlib import Path
    from visualizer.extract import load_architecture
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.computation_graph import SYNTHETIC_INPUT, build_computation_graph
    from visualizer.render import COLORS, render_diagram

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
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"]),
    )
    _title, mla = next(
        (title, tree)
        for title, tree in spec.detailed_block_trees
        if title.startswith("KimiMLA Attn")
    )
    graph = build_computation_graph(mla)
    input_index = next(i for i, spec_node in enumerate(graph.nodes) if spec_node.synthetic == SYNTHETIC_INPUT)
    gate_index = next(i for i, spec_node in enumerate(graph.nodes) if spec_node.block and spec_node.block.attr_name == "g_proj")
    assert (input_index, gate_index) in graph.links
    assert (input_index, gate_index) not in graph.dashed_links

    out = render_diagram(spec, tmp_path / "kimi_detailed.svg", detailed=True)
    svg = out.read_text(encoding="utf-8")
    assert COLORS["flow"] in svg
    assert COLORS["residual"] not in svg or "stroke-dasharray" not in svg.split(COLORS["residual"])[-1][:200]


def test_kimi_mla_attention_feeds_depart_vertically():
    """Same-column spread top-entry ports must drop vertically before shifting horizontally."""
    from collections import defaultdict
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        SYNTHETIC_INPUT,
        _estimate_graph_height,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import (
        COLORS,
        PARALLEL_CONNECTOR_COORD_EPS,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _collect_detail_link_paths,
        _compute_detail_connector_buses,
        _connector_source_bottom_exit_y,
    )
    from visualizer.render_validate import finalize_detail_layout

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiMLAAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(attn, basic_ops=basic)
    fig, ax = plt.subplots(figsize=(16, 13))
    measure_graph_node_sizes(ax, graph)
    positions, links = layout_computation_graph(
        graph,
        cx=2.6,
        top_y=10.0,
        block_w=8.0,
        block_h=_estimate_graph_height(graph),
        content_left=0.6,
    )
    finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=None,
        cx=2.6,
        top_y=10.0,
        detail_fill=COLORS["detail_fill"],
        min_left=0.6,
    )
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
    anchors = _anchors_from_detail_plan(positions, plan)
    incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
    outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for src, tgt in links:
        incoming[tgt].append((src, tgt))
        outgoing[src].append((src, tgt))
    input_index = next(i for i, node in enumerate(graph.nodes) if node.synthetic == SYNTHETIC_INPUT)
    target_bus, source_bus, merge_entry_x, merge_link_bus = _compute_detail_connector_buses(
        graph,
        positions,
        anchors,
        incoming,
        outgoing,
        plan.label_obstacles,
    )
    link_paths = _collect_detail_link_paths(
        graph=graph,
        links=links,
        positions=positions,
        anchors=anchors,
        incoming=incoming,
        label_obstacles=plan.label_obstacles,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_entry_x=merge_entry_x,
        merge_link_bus=merge_link_bus,
        input_index=input_index,
    )
    attn_index = next(i for i, node in enumerate(graph.nodes) if node.label == "Attention")
    try:
        for src, tgt in links:
            if tgt != attn_index:
                continue
            points = link_paths[(src, tgt)]
            assert len(points) >= 2, f"{graph.nodes[src].label} -> Attention missing connector"
            source = anchors[src]
            y_exit = _connector_source_bottom_exit_y(source)
            x1, y1 = points[0]
            x2, y2 = points[1]
            assert abs(y1 - y_exit) < 1e-6, (
                f"{graph.nodes[src].label} must start at source bottom exit"
            )
            assert abs(x1 - x2) < PARALLEL_CONNECTOR_COORD_EPS, (
                f"{graph.nodes[src].label} must depart vertically, not horizontally"
            )
            assert abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS, (
                f"{graph.nodes[src].label} first segment must be vertical"
            )
    finally:
        plt.close(fig)


def test_build_computation_graph_includes_method_wrappers():
    from visualizer.block_tree import BlockNode
    from visualizer.computation_graph import build_computation_graph

    def leaf(name: str) -> BlockNode:
        return BlockNode(attr_name=name, class_name="Linear", role="other", label="Linear", is_basic=True)

    wrapper = BlockNode(
        attr_name="dispatch_tokens",
        class_name="dispatch_tokens",
        role="other",
        label="dispatch tokens",
        is_basic=True,
        details=["method `dispatch_tokens()`"],
    )
    root = BlockNode(
        attr_name="moe",
        class_name="MoE",
        role="moe",
        label="MoE",
        children=[leaf("gate"), wrapper, leaf("down_proj")],
    )
    graph = build_computation_graph(root)
    labels = {spec.label for spec in graph.nodes if spec.block is not None}
    assert "dispatch tokens" in labels
    assert "Linear" in labels
    assert len(graph.links) == 3


def test_kimi_omits_alternate_forward_attn_residual_dispatch():
    from pathlib import Path

    from visualizer.ast_analyze import analyze_source
    from visualizer.extract import load_architecture
    from visualizer.basic_ops import BasicOpFilter

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    decoder = analysis.class_registry["KimiDecoderLayer"]
    assert "_forward_attn_residual" not in decoder.forward_calls
    assert decoder.forward_calls[:2] == ["input_layernorm", "self_attn"]

    spec = load_architecture(
        "moonshotai/Kimi-K3",
        code_path=code_path,
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"]),
    )
    titles = [title for title, _ in spec.detailed_block_trees]
    assert not any("Decoder layer" in title for title in titles)
    component_attrs = {comp.attr_name for comp in spec.block_components}
    assert "_forward_attn_residual" not in component_attrs


def test_kimi_detailed_linear_wrappers_render_in_diagrams(tmp_path: Path):
    from pathlib import Path

    from visualizer.extract import load_architecture
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.render import render_diagram

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
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"]),
    )
    titles = [title for title, _ in spec.detailed_block_trees]
    assert not any("Token embedding" in title for title in titles)
    assert not any("LM head" in title for title in titles)

    out = render_diagram(spec, tmp_path / "kimi_linear_wrappers.svg", detailed=True)
    svg = out.read_text(encoding="utf-8")
    assert "tokenization" not in svg.lower()
    assert "Token Embedding" in svg
    assert "Linear" in svg
    assert "Token embedding (embed_tokens)" not in svg


def test_build_computation_graph_prefixes_attn_residual_on_attention():
    from visualizer.block_tree import BlockNode
    from visualizer.computation_graph import build_computation_graph

    attn_residual = BlockNode(
        attr_name="_forward_attn_residual",
        class_name="_forward_attn_residual",
        role="other",
        label="_forward_attn_residual",
        is_basic=True,
        details=["method `_forward_attn_residual()`"],
    )
    root = BlockNode(
        attr_name="self_attn",
        class_name="KimiMLAAttention",
        role="attention",
        label="MLA",
        children=[
            BlockNode(attr_name="q_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
        ],
    )
    graph = build_computation_graph(root, prefix_steps=[attn_residual])
    labels = [spec.label for spec in graph.nodes if spec.block is not None]
    assert labels[0] == "forward attn residual"
    assert "Linear" in labels


def test_parallel_gates_exclude_sequential_router():
    import ast

    from visualizer.ast_analyze import _ModelAstVisitor, _parse_forward

    code = '''
class MoE:
    def forward(self, hidden_states):
        scores = self.gate(hidden_states)
        out = self.routed_expert_down_proj(scores)
        return self.shared_experts(out)
'''
    tree = ast.parse(code)
    visitor = _ModelAstVisitor()
    visitor.visit(tree)
    moe = visitor.classes["MoE"]
    assert moe.parallel_gates == []


def test_moe_computation_graph_always_shows_input():
    from visualizer.block_tree import BlockNode, build_block_node
    from visualizer.computation_graph import SYNTHETIC_INPUT, build_computation_graph

    root = BlockNode(
        attr_name="block_sparse_moe",
        class_name="MoE",
        role="moe",
        label="MoE",
        input_label="hidden_states",
        children=[
            BlockNode(attr_name="gate", class_name="Linear", role="router", label="Router", is_basic=True),
            BlockNode(attr_name="down", class_name="Linear", role="other", label="Linear", is_basic=True),
        ],
    )
    graph = build_computation_graph(root)
    assert any(spec.synthetic == SYNTHETIC_INPUT for spec in graph.nodes)
    gate_index = next(i for i, spec in enumerate(graph.nodes) if spec.block and spec.block.attr_name == "gate")
    input_index = next(i for i, spec in enumerate(graph.nodes) if spec.synthetic == SYNTHETIC_INPUT)
    assert (input_index, gate_index) in graph.links


def test_computation_graph_layout_respects_box_heights():
    from visualizer.block_tree import BlockNode, build_block_node
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.computation_graph import build_computation_graph, layout_computation_graph, _estimate_graph_height
    from visualizer.sizing import min_vertical_block_gap

    code = '''
class Gate(nn.Module):
    def forward(self, hidden_states):
        return hidden_states, None

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16)
        self.up_proj = nn.Linear(8, 16)
        self.down_proj = nn.Linear(16, 8)
    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))

class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = Gate()
        self.down = nn.Linear(8, 8)
        self.shared_experts = MLP()
    def forward(self, hidden_states):
        scores, _ = self.gate(hidden_states)
        out = self.down(scores)
        return out + self.shared_experts(hidden_states)
'''
    analysis = analyze_source(code, filename="moe.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$"])
    tree = build_block_node(
        attr_name="moe",
        class_name="MoE",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(tree)
    est_h = _estimate_graph_height(graph)
    positions, links = layout_computation_graph(
        graph,
        cx=5.0,
        top_y=10.0,
        block_w=6.0,
        block_h=est_h,
    )
    assert len(links) >= 2
    for src, tgt in links:
        source = positions[src]
        target = positions[tgt]
        gap = source.bottom - target.top_y
        min_gap = min_vertical_block_gap()
        assert gap >= min_gap - 1e-6, (
            f"{source.spec.label} -> {target.spec.label} gap {gap:.3f} < {min_gap:.3f}; "
            f"heights {source.height:.3f}/{target.height:.3f}"
        )


def test_layout_computation_graph_centers_on_cx():
    from visualizer.block_tree import BlockNode, build_block_node
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.computation_graph import (
        _node_content_left,
        _node_content_right,
        build_computation_graph,
        layout_computation_graph,
        _estimate_graph_height,
    )

    def content_mid(positions) -> float:
        min_left = min(_node_content_left(pos) for pos in positions)
        max_right = max(_node_content_right(pos) for pos in positions)
        return (min_left + max_right) / 2

    code = '''
class Gate(nn.Module):
    def forward(self, hidden_states):
        return hidden_states, None

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16)
        self.up_proj = nn.Linear(8, 16)
        self.down_proj = nn.Linear(16, 8)
    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))

class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = Gate()
        self.down = nn.Linear(8, 8)
        self.shared_experts = MLP()
    def forward(self, hidden_states):
        scores, _ = self.gate(hidden_states)
        out = self.down(scores)
        return out + self.shared_experts(hidden_states)
'''
    analysis = analyze_source(code, filename="moe.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$"])
    tree = build_block_node(
        attr_name="moe",
        class_name="MoE",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(tree)
    cx = 5.0
    positions, _ = layout_computation_graph(
        graph,
        cx=cx,
        top_y=10.0,
        block_w=6.0,
        block_h=_estimate_graph_height(graph),
    )
    assert abs(content_mid(positions) - cx) < 0.02

    # Multi-column graph (parallel Q/K/V branches) should also center on cx.
    root = BlockNode(
        attr_name="attn",
        class_name="Attn",
        role="attention",
        label="Attention",
        children=[
            BlockNode(attr_name="q", class_name="Linear", role="other", label="Q", is_basic=True),
            BlockNode(attr_name="k", class_name="Linear", role="other", label="K", is_basic=True),
            BlockNode(attr_name="v", class_name="Linear", role="other", label="V", is_basic=True),
            BlockNode(
                attr_name="@attention",
                class_name="AttentionOp",
                role="attention",
                label="Attention",
                is_basic=True,
            ),
            BlockNode(attr_name="o", class_name="Linear", role="other", label="O", is_basic=True),
        ],
    )
    graph = build_computation_graph(root)
    positions, _ = layout_computation_graph(
        graph,
        cx=cx,
        top_y=10.0,
        block_w=6.0,
        block_h=_estimate_graph_height(graph),
    )
    assert abs(content_mid(positions) - cx) < 0.02


def test_layout_computation_graph_stacks_layers_compactly():
    from pathlib import Path
    from visualizer.extract import load_architecture
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.computation_graph import (
        build_computation_graph,
        layout_computation_graph,
        _estimate_graph_height,
    )
    from visualizer.sizing import min_vertical_block_gap

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
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"]),
    )
    _title, mla = next(
        (title, tree)
        for title, tree in spec.detailed_block_trees
        if title.startswith("KimiMLA Attn")
    )
    graph = build_computation_graph(mla)
    est_h = _estimate_graph_height(graph)
    positions, _ = layout_computation_graph(
        graph,
        cx=5.0,
        top_y=20.0,
        block_w=6.0,
        block_h=est_h,
    )
    content_span = max(pos.top_y for pos in positions) - min(pos.bottom for pos in positions)
    assert content_span <= est_h + 0.05
    assert est_h < 5.0, f"MLA detail block still too tall: {est_h:.2f}"


def test_detail_frame_clear_of_tiles():
    from pathlib import Path
    from visualizer.extract import load_architecture
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.computation_graph import build_computation_graph, layout_computation_graph, _estimate_graph_height
    from visualizer.render import (
        DETAIL_FRAME_GAP,
        _detail_content_bounds,
        _detail_content_extents,
        _detail_frame_edge_pad,
    )

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
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"]),
    )
    edge_pad = _detail_frame_edge_pad()
    for _title, tree in spec.detailed_block_trees:
        graph = build_computation_graph(tree)
        positions, _ = layout_computation_graph(
            graph,
            cx=5.0,
            top_y=20.0,
            block_w=6.0,
            block_h=_estimate_graph_height(graph),
        )
        min_left, max_right, min_bottom, max_top = _detail_content_extents(positions)
        frame_left, frame_right, frame_bottom, frame_top = _detail_content_bounds(
            positions,
        )
        assert min_left - frame_left >= edge_pad - 1e-6
        assert frame_right - max_right >= edge_pad - 1e-6
        assert min_bottom - frame_bottom >= edge_pad - 1e-6
        assert frame_top - max_top >= edge_pad - 1e-6
        assert min_left - frame_left >= DETAIL_FRAME_GAP
        assert frame_top - max_top >= DETAIL_FRAME_GAP


def test_estimate_block_size_fits_attention_label():
    from visualizer.block_tree import BlockNode
    from visualizer.sizing import estimate_block_size, estimate_block_size_for_node

    attn = BlockNode(
        attr_name="@attention",
        class_name="AttentionOp",
        role="attention",
        label="Attention",
        is_basic=True,
    )
    plain_w, plain_h = estimate_block_size("Linear")
    attn_w, attn_h = estimate_block_size_for_node(attn)
    assert attn_w >= plain_w
    assert attn_h == plain_h

    kda = BlockNode(
        attr_name="@attention",
        class_name="AttentionOp",
        role="attention",
        label="Delta attention (KDA)",
        is_basic=False,
        details=["recurrent delta rule (not softmax QKᵀV)"],
    )
    _, kda_h = estimate_block_size_for_node(kda)
    assert kda_h == plain_h


def test_layout_computation_graph_uses_per_node_sizes():
    from visualizer.ast_analyze import SYNTHETIC_ATTENTION
    from visualizer.block_tree import BlockNode
    from visualizer.computation_graph import build_computation_graph, layout_computation_graph

    def leaf(name: str, label: str | None = None) -> BlockNode:
        return BlockNode(attr_name=name, class_name="Linear", role="other", label=label or "Linear", is_basic=True)

    root = BlockNode(
        attr_name="attn",
        class_name="Attn",
        role="attention",
        label="Attn",
        children=[
            leaf("q_proj"),
            BlockNode(
                attr_name=SYNTHETIC_ATTENTION,
                class_name="AttentionOp",
                role="attention",
                label="Attention",
                is_basic=True,
            ),
            leaf("o_proj"),
        ],
    )
    graph = build_computation_graph(root)
    positions, _ = layout_computation_graph(graph, cx=4.0, top_y=10.0, block_w=4.0, block_h=3.0)
    by_key = {pos.spec.key: pos for pos in positions}
    attn = by_key[f"seq:1:{SYNTHETIC_ATTENTION}:{SYNTHETIC_ATTENTION}:0"]
    q_proj = by_key["seq:0:q_proj:q_proj:0"]
    o_proj = by_key["seq:2:o_proj:o_proj:0"]
    assert attn.width >= q_proj.width
    assert q_proj.height == o_proj.height
    assert abs(q_proj.height - attn.height) < 1e-6


def test_detailed_block_trees_include_pipeline_sections():
    from visualizer.computation_graph import SYNTHETIC_INPUT, build_computation_graph
    from visualizer.extract import load_architecture
    from visualizer.basic_ops import BasicOpFilter

    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    titles = [title for title, _ in spec.detailed_block_trees]
    assert not any("Positional" in title for title in titles)
    assert not any("Token embedding" in title for title in titles)
    assert not any("LM head" in title for title in titles)
    assert not any("Tokenization" in title for title in titles)

    for _title, tree in spec.detailed_block_trees:
        graph = build_computation_graph(tree)
        assert any(node.synthetic == SYNTHETIC_INPUT for node in graph.nodes), (
            f"missing input node for {_title}"
        )


def test_detailed_expands_rope_at_top_level(tmp_path: Path):
    from visualizer.extract import load_architecture
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.render import render_diagram

    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    out = render_diagram(spec, tmp_path / "detailed_rope.svg", detailed=True)
    svg = out.read_text(encoding="utf-8")
    assert "Freq computation" in svg
    assert "ApplyRotary" in svg
    assert "cos, sin from inv_freq" not in svg
    assert "rotate query/key tensors" not in svg
    assert "Map token IDs to embeddings" not in svg
    assert "Positional (RoPE)" in svg
    assert "rotary_emb" in svg


def test_partition_detail_trees_skips_spine_and_inlined_modules():
    from visualizer.block_tree import BlockNode, is_single_function_tree, is_straight_line_module, partition_detail_trees

    embed = BlockNode(
        attr_name="embed_tokens",
        class_name="Embedding",
        role="embedding",
        label="Embedding",
        is_basic=True,
    )
    rope = BlockNode(
        attr_name="rotary_emb",
        class_name="RotaryEmbedding",
        role="other",
        label="RoPE",
        children=[
            BlockNode(attr_name="freqs", class_name="RotaryEmbedding", role="other", label="Freq", is_basic=True),
            BlockNode(attr_name="apply", class_name="ApplyRotary", role="other", label="Apply", is_basic=True),
        ],
    )
    gate = BlockNode(
        attr_name="gate",
        class_name="KimiMoEGate",
        role="router",
        label="Router",
        children=[
            BlockNode(
                attr_name="@functional_linear",
                class_name="Linear",
                role="other",
                label="Linear",
                is_basic=True,
            ),
        ],
    )
    assert is_single_function_tree(embed)
    assert not is_single_function_tree(rope)
    assert not is_single_function_tree(gate)
    assert is_straight_line_module(rope)
    assert is_straight_line_module(gate)

    trees = [
        ("Token embedding (embed_tokens)", embed),
        ("Positional (RoPE) (rotary_emb)", rope),
        ("Router (gate)", gate),
    ]
    kept = partition_detail_trees(trees)
    assert kept == []


def test_inline_composite_steps_expands_single_op_wrappers():
    from visualizer.block_tree import BlockNode, inline_composite_steps

    linear = BlockNode(
        attr_name="@functional_linear",
        class_name="Linear",
        role="other",
        label="Linear",
        is_basic=True,
    )
    gate = BlockNode(
        attr_name="gate",
        class_name="KimiMoEGate",
        role="router",
        label="Router",
        children=[linear],
    )
    mlp = BlockNode(
        attr_name="shared_experts",
        class_name="KimiMLP",
        role="ffn",
        label="KimiMLP",
        children=[
            BlockNode(attr_name="gate_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="up_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
        ],
    )

    expanded, wrapper = inline_composite_steps(gate)
    assert expanded == [gate]
    assert wrapper is None

    expanded_mlp, mlp_wrapper = inline_composite_steps(mlp)
    assert mlp_wrapper is mlp
    assert len(expanded_mlp) == 2
    assert expanded_mlp[0].attr_name == "gate_proj"
    assert expanded_mlp[1].attr_name == "up_proj"


def test_router_gate_expands_to_pipeline_steps():
    from visualizer.block_tree import BlockNode, collect_function_steps, inline_composite_steps

    gate = BlockNode(
        attr_name="gate",
        class_name="KimiMoEGate",
        role="router",
        label="Router",
        children=[
            BlockNode(attr_name="@functional_linear", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="@router_activation", class_name="RouterOp", role="other", label="Sigmoid", is_basic=True),
            BlockNode(attr_name="@router_topk", class_name="RouterOp", role="other", label="Top-k experts", is_basic=True),
        ],
    )
    expanded, wrapper = inline_composite_steps(gate)
    assert wrapper is gate
    assert [step.label for step in expanded] == ["Linear", "Sigmoid", "Top-k experts"]


def test_moe_graph_inlines_router_gate():
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node, collect_nested_diagrams
    from visualizer.computation_graph import build_computation_graph

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    moe = build_block_node(
        attr_name="block_sparse_moe",
        class_name="KimiSparseMoeBlock",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(moe)
    labels = [spec.label for spec in graph.nodes]
    assert labels[:9] == [
        "hidden_states",
        "Linear",
        "Sigmoid",
        "Expert bias",
        "Group routing",
        "Top-k experts",
        "Gather weights",
        "Renormalize",
        "Route scaling",
    ]
    assert all(spec.block is None or spec.block.attr_name != "gate" for spec in graph.nodes)
    nested_titles = [title for title, _ in collect_nested_diagrams(moe)]
    assert not any(title.startswith("Router") for title in nested_titles)
    assert not any(title.startswith("KimiMLP") for title in nested_titles)
    assert graph.inline_frames
    assert any(frame.frame_id == "shared_experts" for frame in graph.inline_frames)


def test_kimi_moe_infer_side_inputs_from_gate():
    import ast
    from pathlib import Path

    from visualizer.ast_analyze import _parse_forward

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    tree = ast.parse(code_path.read_text())
    moe_class = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "KimiSparseMoeBlock"
    )
    moe_forward = next(item for item in moe_class.body if isinstance(item, ast.FunctionDef) and item.name == "forward")
    _, _, _, side_inputs, _ = _parse_forward(moe_forward)
    router = side_inputs["moe_infer"]
    assert len(router) == 1
    assert router[0].port_label == "router"
    assert router[0].source_chain == ["gate"]
    assert router[0].source_kind == "prior_step"


def test_kimi_shared_experts_residual_side_input():
    import ast
    from pathlib import Path

    from visualizer.ast_analyze import _parse_forward

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    tree = ast.parse(code_path.read_text())
    moe_class = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "KimiSparseMoeBlock"
    )
    moe_forward = next(item for item in moe_class.body if isinstance(item, ast.FunctionDef) and item.name == "forward")
    _, _, _, side_inputs, _ = _parse_forward(moe_forward)
    residual = side_inputs["shared_experts"]
    assert len(residual) == 1
    assert residual[0].port_label == "identity"
    assert residual[0].source_kind == "forward_input"


def test_moe_infer_combine_op_comes_from_ast():
    import ast
    from pathlib import Path

    from visualizer.ast_analyze import (
        _detect_method_combine_op,
        analyze_source,
        combine_op_from_step_details,
    )

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    tree = ast.parse(code_path.read_text())
    moe_class = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "KimiSparseMoeBlock"
    )
    moe_infer = next(
        item for item in moe_class.body if isinstance(item, ast.FunctionDef) and item.name == "moe_infer"
    )
    assert _detect_method_combine_op(moe_infer) == "MoE aggregation"

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    cls = analysis.class_registry["KimiSparseMoeBlock"]
    assert combine_op_from_step_details(cls.forward_step_details["moe_infer"]) == "MoE aggregation"


def test_moe_infer_graph_dashed_router_side_link():
    from pathlib import Path

    from visualizer.ast_analyze import SYNTHETIC_ROUTER_SCALE, analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import SYNTHETIC_COMBINE, SYNTHETIC_INPUT, build_computation_graph

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    moe = build_block_node(
        attr_name="block_sparse_moe",
        class_name="KimiSparseMoeBlock",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(moe)
    agg_index = next(
        index
        for index, spec in enumerate(graph.nodes)
        if spec.label == "MoE aggregation"
    )
    assert graph.nodes[agg_index].sublabel is None
    assert graph.nodes[agg_index].synthetic is None
    route_scaling_index = next(
        index
        for index, spec in enumerate(graph.nodes)
        if spec.block and spec.block.attr_name == SYNTHETIC_ROUTER_SCALE
    )
    assert (route_scaling_index, agg_index) in graph.links
    assert (route_scaling_index, agg_index) not in graph.dashed_links
    assert (route_scaling_index, agg_index) not in graph.side_entry_links
    assert (route_scaling_index, agg_index) not in graph.link_port_labels
    down_proj_index = next(
        index
        for index, spec in enumerate(graph.nodes)
        if spec.block and spec.block.attr_name == "routed_expert_down_proj"
    )
    input_index = next(index for index, spec in enumerate(graph.nodes) if spec.synthetic == SYNTHETIC_INPUT)
    assert (input_index, down_proj_index) in graph.links
    assert (down_proj_index, agg_index) in graph.links
    assert (route_scaling_index, down_proj_index) not in graph.links


def test_shared_experts_graph_residual_side_link_is_solid():
    import ast

    from visualizer.ast_analyze import _ModelAstVisitor
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import SYNTHETIC_COMBINE, SYNTHETIC_INPUT, build_computation_graph

    code = '''
class MLP:
    def forward(self, hidden_states):
        return hidden_states

class MoE:
    def forward(self, hidden_states):
        identity = hidden_states
        out = self.down_proj(hidden_states)
        return out + self.shared_experts(identity)
'''
    tree = ast.parse(code)
    visitor = _ModelAstVisitor()
    visitor.visit(tree)
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$"])
    moe = build_block_node(
        attr_name="moe",
        class_name="MoE",
        registry=visitor.classes,
        basic_ops=basic,
    )
    graph = build_computation_graph(moe)
    shared_index = next(
        index
        for index, spec in enumerate(graph.nodes)
        if spec.block and spec.block.attr_name == "shared_experts"
    )
    input_index = next(index for index, spec in enumerate(graph.nodes) if spec.synthetic == SYNTHETIC_INPUT)
    plus_index = next(
        index
        for index, spec in enumerate(graph.nodes)
        if spec.synthetic == SYNTHETIC_COMBINE and spec.label == "+"
    )
    assert (input_index, shared_index) in graph.links
    assert (shared_index, plus_index) in graph.side_entry_links
    assert (input_index, shared_index) not in graph.dashed_links
    assert (shared_index, plus_index) not in graph.dashed_links
    assert graph.nodes[plus_index].sublabel is None


def test_render_detailed_diagram(tmp_path: Path):
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    assert spec.detailed_block_trees

    out = render_diagram(spec, tmp_path / "detailed.svg", detailed=True)
    svg = out.read_text(encoding="utf-8")
    assert "Token Embedding" in svg
    assert "tokenization" not in svg.lower()
    assert "expert" in svg.lower() or "logits" in svg.lower()
    assert svg.count("fill: #fff5f4; stroke: #c0392b") >= 1
    assert "getattr" not in svg.lower()
    assert "Parameter" not in svg
    assert "<!-- Q -->" in svg or "DejaVuSans-Bold-51" in svg


def test_infer_kimi_layer_variants():
    from visualizer.extract import load_architecture
    from visualizer.basic_ops import BasicOpFilter

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
    assert spec.attention_type == "Hybrid"
    assert spec.layer_mix is not None
    assert len(spec.layer_variants) == 3
    assert {variant.attention_label for variant in spec.layer_variants} == {
        "KimiDelta Attn",
        "KimiMLA Attn",
    }
    assert sum(variant.count for variant in spec.layer_variants) == spec.num_hidden_layers

    titles = [title for title, _ in spec.detailed_block_trees]
    assert any(title.startswith("KimiDelta Attn") for title in titles)
    assert any(title.startswith("KimiMLA Attn") for title in titles)
    assert any(title.startswith("KimiMLP") for title in titles)
    mlp_tree = next(tree for title, tree in spec.detailed_block_trees if title.startswith("KimiMLP"))
    moe_tree = next(tree for title, tree in spec.detailed_block_trees if title.startswith("KimiSparseMoeBlock"))
    kda_tree = next(tree for title, tree in spec.detailed_block_trees if title.startswith("KimiDelta Attn"))
    mla_tree = next(tree for title, tree in spec.detailed_block_trees if title.startswith("KimiMLA Attn"))
    assert kda_tree.input_source == "RMSNorm"
    assert mla_tree.input_source == "RMSNorm"
    assert mlp_tree.input_source == "RMSNorm"
    assert moe_tree.input_source == "RMSNorm"


def test_kimi_ffn_spine_label_uses_class_names():
    from pathlib import Path
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.extract import load_architecture
    from visualizer.render import _ffn_label

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
    label, sublabel = _ffn_label(spec)
    assert label == "KimiMoE / KimiMLP"
    assert sublabel is None


def test_straight_line_modules_omit_separate_block_internals():
    from pathlib import Path
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.extract import load_architecture
    from visualizer.computation_graph import build_computation_graph

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
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"]),
    )
    titles = [title for title, _ in spec.detailed_block_trees]
    assert any("KimiMLP" in title for title in titles)

    _title, moe = next(
        (title, tree)
        for title, tree in spec.detailed_block_trees
        if title.startswith("KimiSparseMoeBlock")
    )
    graph = build_computation_graph(moe)
    assert any(frame.frame_id == "shared_experts" for frame in graph.inline_frames)


def test_layer_repeat_lines_in_fact_sheet():
    from visualizer.extract import load_architecture
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.render import _fact_lines

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
    assert any(line.startswith("93 × KimiDecoderLayer") for line in spec.layer_repeat_lines)
    assert any(
        "self_attn → KimiDeltaAttention (if (layer_idx + 1) in kda_layers" in line
        for line in spec.layer_repeat_lines
    )
    assert any("self_attn → KimiMLAAttention (else)" in line for line in spec.layer_repeat_lines)
    assert any("block_sparse_moe → KimiSparseMoeBlock (if layer_idx >= 1)" in line for line in spec.layer_repeat_lines)
    assert any("68 × KimiDelta Attn + KimiSparseMoeBlock" in line for line in spec.layer_repeat_lines)

    fact_lines = _fact_lines(spec)
    assert any(line.startswith("Layer repeat: 93 × KimiDecoderLayer") for line in fact_lines)
    assert any(
        line.startswith("    self_attn → KimiDeltaAttention (if (layer_idx + 1) in kda_layers")
        for line in fact_lines
    )
    assert any(line.startswith("    self_attn → KimiMLAAttention (else)") for line in fact_lines)
    assert any(
        line.startswith("    block_sparse_moe → KimiSparseMoeBlock (if layer_idx >= 1)")
        for line in fact_lines
    )
    assert not any(line.startswith("Layer mix:") for line in fact_lines)
    assert not any(line == "Layers: 93" for line in fact_lines)
    assert not any("• •" in line for line in fact_lines)


def test_fact_sheet_height_fits_indented_sublines():
    from visualizer.extract import load_architecture
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.render import (
        FACT_SUBLINE_INDENT,
        PANEL_LINE_HEIGHT,
        PANEL_PAD_BOTTOM,
        PANEL_PAD_TOP,
        _fact_lines,
        _fact_sheet_content_rows,
        _fact_sheet_height,
        _fact_sheet_highlight_rows,
    )

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
    lines = _fact_lines(spec)
    highlight_rows = _fact_sheet_highlight_rows(spec, wrap_width=48)
    rows = _fact_sheet_content_rows(lines, wrap_width=48)
    expected_h = (
        PANEL_PAD_TOP
        + rows * PANEL_LINE_HEIGHT
        + (0.12 if highlight_rows else 0.0)
        + 0.17 * len(highlight_rows)
        + PANEL_PAD_BOTTOM
    )
    assert _fact_sheet_height(spec, wrap_width=48) == max(expected_h, 1.6)
    assert any(line.startswith(FACT_SUBLINE_INDENT) for line in lines)


def test_layer_repeat_condition_simplification():
    from visualizer.layer_repeat_simplify import simplify_layer_repeat_lines

    config = {
        "num_experts": 896,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "kv_lora_rank": 512,
        "linear_attn_config": {"kda_layers": [1, 2, 3, 4, 5]},
    }
    lines = simplify_layer_repeat_lines(
        [
            "N × DemoLayer (layer_idx in range(config.num_hidden_layers))",
            "self_attn → DeltaAttention (if config.is_kda_layer(layer_idx))",
            "self_attn → LatentAttention (elif config.is_mla)",
            (
                "block_sparse_moe → DemoMoE (if config.num_experts is not None and "
                "layer_idx >= config.first_k_dense_replace and "
                "(layer_idx % getattr(config, 'moe_layer_freq', 1) == 0))"
            ),
            "mlp → DemoMLP (else)",
        ],
        config,
    )
    assert lines[0] == "N × DemoLayer (layer_idx in range(config.num_hidden_layers))"
    assert lines[1] == "self_attn → DeltaAttention (if (layer_idx + 1) in [1–5])"
    assert lines[2] == "self_attn → LatentAttention (else)"
    assert lines[3] == "block_sparse_moe → DemoMoE (if layer_idx >= 1)"
    assert lines[4] == "mlp → DemoMLP (else)"


def test_build_layer_repeat_lines_from_minimal_ast():
    from visualizer.ast_analyze import analyze_source

    source = """
import torch.nn as nn

class DemoDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        if layer_idx % 2 == 0:
            self.self_attn = FastAttention(config)
        else:
            self.self_attn = SlowAttention(config)
        if layer_idx >= 4:
            self.block_sparse_moe = DemoMoE(config)
        else:
            self.mlp = DemoMLP(config)

class DemoModel(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DemoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
"""
    analysis = analyze_source(source)
    assert analysis.layer_repeat_lines
    assert analysis.layer_repeat_lines[0].startswith("N × DemoDecoderLayer")
    assert any("FastAttention" in line for line in analysis.layer_repeat_lines)
    assert any("DemoMoE" in line for line in analysis.layer_repeat_lines)
    assert any("DemoMLP (else)" in line for line in analysis.layer_repeat_lines)


def test_repeat_block_label_bulleted_sublists():
    from visualizer.extract import load_architecture
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.render import _repeat_block_label

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
    label = _repeat_block_label(spec)
    assert label.startswith("93 × Transformer block\n")
    assert "• 68 KimiDelta Attn + KimiSparseMoeBlock" in label
    assert "• 24 KimiMLA Attn + KimiSparseMoeBlock" in label
    assert "• 1 KimiDelta Attn + KimiMLP" in label
    assert "N =" not in label


def test_stack_components_from_kimi_ast():
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.extract import parse_architecture, _rebuild_stack_components

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    spec = parse_architecture({}, "Kimi", code_analysis=analysis)
    spec.positional_encoding = "RoPE"
    _rebuild_stack_components(spec, analysis)

    assert [comp.attr_name for comp in spec.stack_pre] == ["embed_tokens", "rotary_emb"]
    assert spec.stack_pre[0].label == "Token Embedding"
    assert [comp.attr_name for comp in spec.stack_tail] == ["norm", "lm_head"]
    assert spec.stack_tail[0].label == "RMSNorm"
    assert spec.stack_tail[1].label == "Linear"


def test_is_linear_pipeline_block():
    from visualizer.block_tree import BlockNode, is_linear_pipeline_block, is_straight_line_module

    linear = BlockNode(
        attr_name="gate_proj",
        class_name="Linear",
        role="other",
        label="Linear",
        is_basic=True,
    )
    mlp = BlockNode(
        attr_name="shared_experts",
        class_name="KimiMLP",
        role="ffn",
        label="KimiMLP",
        children=[linear, linear],
    )
    assert is_linear_pipeline_block(mlp)
    assert is_straight_line_module(mlp)


def test_straight_line_module_general_expansion_rule():
    """Simple straight-line composites expand inline; branching modules stay collapsed."""
    from visualizer.block_tree import (
        BlockNode,
        build_stack_component_tree,
        inline_composite_steps,
        is_straight_line_module,
    )
    from visualizer.blocks import BlockComponent

    output_gate = BlockNode(
        attr_name="g_proj",
        class_name="OutputGate",
        role="gate",
        label="Output gate",
        children=[
            BlockNode(attr_name="g_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="@gate_activation", class_name="ActivationOp", role="other", label="Sigmoid", is_basic=True),
        ],
    )
    assert is_straight_line_module(output_gate)
    expanded, wrapper = inline_composite_steps(output_gate)
    assert wrapper is output_gate
    assert [step.label for step in expanded] == ["Linear", "Sigmoid"]

    branching = BlockNode(
        attr_name="self_attn",
        class_name="Attention",
        role="attention",
        label="Attention",
        children=[
            BlockNode(attr_name="q_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="k_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="v_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="@attention", class_name="AttentionOp", role="attention", label="Attention", is_basic=True),
            BlockNode(attr_name="o_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
        ],
        attention_inputs={"q": ["q_proj"], "k": ["k_proj"], "v": ["v_proj"]},
    )
    assert not is_straight_line_module(branching)

    positional = build_stack_component_tree(
        BlockComponent(
            attr_name="rotary_emb",
            class_name="RotaryEmbedding",
            role="positional",
            label="RoPE",
            forward_order=1,
        ),
        registry={},
        basic_ops=__import__("visualizer.basic_ops", fromlist=["BasicOpFilter"]).BasicOpFilter.for_detailed(),
    )
    assert is_straight_line_module(positional)


def test_moe_graph_records_shared_experts_inline_frame():
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import build_computation_graph

    code = '''
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16)
        self.up_proj = nn.Linear(8, 16)
        self.down_proj = nn.Linear(16, 8)
    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))

class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(8, 8)
        self.shared_experts = MLP()
    def forward(self, hidden_states):
        out = self.down(hidden_states)
        return out + self.shared_experts(hidden_states)
'''
    analysis = analyze_source(code, filename="moe.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$"])
    tree = build_block_node(
        attr_name="moe",
        class_name="MoE",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(tree)
    frame = next(frame for frame in graph.inline_frames if frame.frame_id == "shared_experts")
    assert "shared_experts" in frame.label
    assert len(frame.node_indices) >= 3


def test_moe_gate_inline_frame_spacing_after_finalize():
    from pathlib import Path

    import matplotlib.pyplot as plt

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
        _estimate_graph_height,
    )
    from visualizer.render import COLORS
    from visualizer.render_validate import finalize_detail_layout
    from visualizer.sizing import min_vertical_block_gap

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    moe = build_block_node(
        attr_name="block_sparse_moe",
        class_name="KimiSparseMoeBlock",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    fig, ax = plt.subplots(figsize=(16, 13))
    fig.canvas.draw()
    graph = build_computation_graph(moe)
    measure_graph_node_sizes(ax, graph)
    positions, _ = layout_computation_graph(
        graph,
        cx=2.6,
        top_y=10.0,
        block_w=8.0,
        block_h=_estimate_graph_height(graph),
        content_left=0.6,
    )
    finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=None,
        cx=2.6,
        top_y=10.0,
        detail_fill=COLORS["detail_fill"],
        min_left=0.6,
    )
    gate_frame = next(frame for frame in graph.inline_frames if frame.frame_id == "gate")
    ordered = sorted(gate_frame.node_indices, key=lambda index: -positions[index].top_y)
    min_gap = min_vertical_block_gap()
    for upper, lower in zip(ordered, ordered[1:]):
        gap = positions[upper].bottom - positions[lower].top_y
        assert abs(gap - min_gap) <= 0.02, (
            f"{graph.nodes[upper].label} -> {graph.nodes[lower].label} gap {gap:.4f} != {min_gap:.4f}"
        )


def test_mla_attention_fanout_branch_spacing_after_finalize():
    from pathlib import Path

    import matplotlib.pyplot as plt

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        DETAIL_LAYER_GAP,
        SYNTHETIC_INPUT,
        _estimate_graph_height,
        _fanout_branch_node_groups,
        _node_content_left,
        _node_content_right,
        _ordered_inline_frame_chain,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import (
        CONNECTOR_ATTACHED_BOX_MARGIN,
        CONNECTOR_OBSTACLE_MARGIN,
        COLORS,
    )
    from visualizer.render_validate import finalize_detail_layout
    from visualizer.sizing import min_vertical_block_gap

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    mla = build_block_node(
        attr_name="self_attn",
        class_name="KimiMLAAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    fig, ax = plt.subplots(figsize=(16, 13))
    fig.canvas.draw()
    graph = build_computation_graph(mla)
    measure_graph_node_sizes(ax, graph)
    positions, _ = layout_computation_graph(
        graph,
        cx=2.6,
        top_y=10.0,
        block_w=8.0,
        block_h=_estimate_graph_height(graph),
        content_left=0.6,
    )
    finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=None,
        cx=2.6,
        top_y=10.0,
        detail_fill=COLORS["detail_fill"],
        min_left=0.6,
    )
    min_gap = min_vertical_block_gap()
    min_h_gap = min_gap
    branch_groups = _fanout_branch_node_groups(positions)
    assert len(branch_groups) >= 3
    for indices in branch_groups.values():
        chain = _ordered_inline_frame_chain(graph, list(indices))
        stack_chain = [
            index
            for index in chain
            if graph.nodes[index].label in {"Linear", "RMSNorm"}
        ]
        if len(stack_chain) < 2:
            continue
        column_cx = positions[stack_chain[0]].cx
        for index in stack_chain:
            assert abs(positions[index].cx - column_cx) <= 0.02
        for upper, lower in zip(stack_chain, stack_chain[1:]):
            gap = positions[upper].bottom - positions[lower].top_y
            assert abs(gap - min_gap) <= 0.02, (
                f"{graph.nodes[upper].block.attr_name} -> {graph.nodes[lower].block.attr_name} "
                f"gap {gap:.4f} != {min_gap:.4f}"
            )

    rms_indices = [index for index, spec in enumerate(graph.nodes) if spec.label == "RMSNorm"]
    for left_index in range(len(rms_indices) - 1):
        for right_index in range(left_index + 1, len(rms_indices)):
            left = positions[rms_indices[left_index]]
            right = positions[rms_indices[right_index]]
            if abs(left.top_y - right.top_y) > 0.02:
                continue
            h_gap = _node_content_left(right) - _node_content_right(left)
            if left.cx <= right.cx:
                pass
            else:
                h_gap = _node_content_left(left) - _node_content_right(right)
            assert h_gap >= min_h_gap - 0.02, (
                f"RMSNorm tiles touch on row: h_gap={h_gap:.4f} < {min_h_gap:.4f}"
            )

    input_index = next(
        index for index, spec in enumerate(graph.nodes) if spec.synthetic == SYNTHETIC_INPUT
    )
    targets = [target for source, target in graph.links if source == input_index]
    fanout_gap = DETAIL_LAYER_GAP + CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN
    for target in targets:
        clearance = positions[input_index].bottom - positions[target].top_y
        assert clearance >= fanout_gap - 0.02, (
            f"Input bus clearance {clearance:.4f} < {fanout_gap:.4f}"
        )


def test_inline_frame_labels_are_laid_out_without_overlaps():
    from pathlib import Path

    import matplotlib.pyplot as plt

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        _estimate_graph_height,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import COLORS
    from visualizer.render_validate import (
        VALIDATE_MIN_GAP,
        _frame_caption_belongs_to_nested_frame,
        _inline_frame_member_sets,
        collect_measured_elements,
        finalize_detail_layout,
        validate_render_layout,
    )

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.for_detailed()
    blocks = [
        build_block_node(
            attr_name="self_attn",
            class_name="KimiDeltaAttention",
            registry=analysis.class_registry,
            basic_ops=basic,
        ),
        build_block_node(
            attr_name="self_attn",
            class_name="KimiMLAAttention",
            registry=analysis.class_registry,
            basic_ops=basic,
        ),
        build_block_node(
            attr_name="block_sparse_moe",
            class_name="KimiSparseMoeBlock",
            registry=analysis.class_registry,
            basic_ops=basic,
        ),
    ]
    fig, ax = plt.subplots(figsize=(16, 13))
    fig.canvas.draw()
    for block in blocks:
        graph = build_computation_graph(block, basic_ops=basic)
        measure_graph_node_sizes(ax, graph, input_sublabel=None)
        positions, _ = layout_computation_graph(
            graph,
            cx=2.6,
            top_y=10.0,
            block_w=8.0,
            block_h=_estimate_graph_height(graph),
            content_left=0.6,
        )
        plan = finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=2.6,
            top_y=10.0,
            detail_fill=COLORS["detail_fill"],
            min_left=0.6,
        )
        if graph.inline_frames:
            assert plan.inline_frame_labels
        elements = collect_measured_elements(
            ax,
            graph,
            positions,
            plan,
            detail_fill=COLORS["detail_fill"],
        )
        validate_render_layout(elements, min_gap=VALIDATE_MIN_GAP).raise_if_invalid()
        captions = [element for element in elements if element.kind in {"frame_label", "frame_sublabel"}]
        overlap_obstacles = [
            element
            for element in elements
            if element.kind in {"box", "combine", "inline_frame", "floating_label"}
        ]
        member_sets = _inline_frame_member_sets(elements)
        for caption in captions:
            for obstacle in overlap_obstacles:
                if caption.frame_id and obstacle.frame_id == caption.frame_id:
                    continue
                if obstacle.kind == "inline_frame" and _frame_caption_belongs_to_nested_frame(
                    caption,
                    obstacle,
                    member_sets=member_sets,
                ):
                    continue
                assert not caption.bounds.overlaps(
                    obstacle.bounds,
                    min_gap=VALIDATE_MIN_GAP,
                ), f"{caption.label!r} overlaps {obstacle.label!r}"
            frame_id = caption.frame_id
            frame_element = next(
                (element for element in elements if element.kind == "inline_frame" and element.frame_id == frame_id),
                None,
            )
            if frame_element is None:
                continue
            frame_bounds = frame_element.bounds
            gap_limit = 0.55
            if caption.bounds.bottom > frame_bounds.top + gap_limit:
                pytest.fail(f"{caption.label!r} caption too far above frame")
            if caption.bounds.right < frame_bounds.left - gap_limit:
                pytest.fail(f"{caption.label!r} caption too far left of frame")
            if caption.bounds.left > frame_bounds.right + gap_limit:
                pytest.fail(f"{caption.label!r} caption too far right of frame")


def test_moe_horizontal_span_after_finalize():
    from pathlib import Path

    import matplotlib.pyplot as plt

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        _estimate_graph_height,
        _node_content_left,
        _node_content_right,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import COLORS
    from visualizer.render_validate import finalize_detail_layout

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    moe = build_block_node(
        attr_name="block_sparse_moe",
        class_name="KimiSparseMoeBlock",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    fig, ax = plt.subplots(figsize=(16, 13))
    fig.canvas.draw()
    graph = build_computation_graph(moe)
    measure_graph_node_sizes(ax, graph)
    positions, _ = layout_computation_graph(
        graph,
        cx=2.6,
        top_y=10.0,
        block_w=8.0,
        block_h=_estimate_graph_height(graph),
        content_left=0.6,
    )
    finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=None,
        cx=2.6,
        top_y=10.0,
        detail_fill=COLORS["detail_fill"],
        min_left=0.6,
    )
    left = min(_node_content_left(pos) for pos in positions)
    right = max(_node_content_right(pos) for pos in positions)
    span = right - left
    assert 0.9 <= span <= 10.0, f"MoE horizontal span {span:.3f} outside shrink-wrap range"


def test_moe_finalize_keeps_combine_gap_tight():
    from pathlib import Path

    import matplotlib.pyplot as plt

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        DETAIL_LAYER_GAP,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
        _estimate_graph_height,
    )
    from visualizer.render import COLORS
    from visualizer.render_validate import finalize_detail_layout

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    moe = build_block_node(
        attr_name="block_sparse_moe",
        class_name="KimiSparseMoeBlock",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    fig, ax = plt.subplots(figsize=(16, 13))
    fig.canvas.draw()
    graph = build_computation_graph(moe)
    measure_graph_node_sizes(ax, graph)
    est_h = _estimate_graph_height(graph)
    positions, _ = layout_computation_graph(
        graph,
        cx=2.6,
        top_y=10.0,
        block_w=8.0,
        block_h=est_h,
        content_left=0.6,
    )
    route_index = next(i for i, spec in enumerate(graph.nodes) if spec.label == "Route scaling")
    combine_index = next(i for i, spec in enumerate(graph.nodes) if spec.label == "MoE aggregation")
    gap = positions[route_index].bottom - positions[combine_index].top_y
    assert gap <= DETAIL_LAYER_GAP + 0.02, (
        f"Route scaling -> MoE aggregation gap {gap:.3f} exceeds layer gap {DETAIL_LAYER_GAP:.3f}"
    )
    finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=None,
        cx=2.6,
        top_y=10.0,
        detail_fill=COLORS["detail_fill"],
        min_left=0.6,
    )
    plt.close(fig)


def test_side_entry_combine_connector_enters_at_combine_center_y():
    from visualizer.render import MERGE_RADIUS, _RenderAnchor, _side_entry_combine_connector_points

    gap = 0.04
    source = _RenderAnchor(cx=0.78, top=7.4, bottom=7.2, left=0.5, right=1.0)
    target_cx = 1.1
    target_cy = 6.85
    points = _side_entry_combine_connector_points(
        source,
        target_cx,
        target_cy,
        gap=gap,
    )
    assert points[-1] == (target_cx - MERGE_RADIUS, target_cy)
    assert points[0] == (source.cx, source.bottom)


def _assert_input_clears_direct_consumers(positions, graph, *, min_gap: float | None = None) -> None:
    from visualizer.computation_graph import DETAIL_LAYER_GAP, SYNTHETIC_INPUT, _node_content_left, _node_content_right

    gap = DETAIL_LAYER_GAP if min_gap is None else min_gap
    input_index = next(
        index for index, pos in enumerate(positions) if pos.spec.synthetic == SYNTHETIC_INPUT
    )
    input_pos = positions[input_index]
    targets = [target for source, target in graph.links if source == input_index]
    assert targets, "Synthetic input should feed at least one downstream node"
    for target in targets:
        consumer = positions[target]
        clearance = input_pos.bottom - consumer.top_y
        assert clearance >= gap - 1e-6, (
            f"Input overlaps consumer {consumer.spec.label!r}: clearance {clearance:.4f} < {gap:.4f}"
        )
        horizontal_overlap = (
            _node_content_left(input_pos) + gap <= _node_content_right(consumer)
            and _node_content_left(consumer) + gap <= _node_content_right(input_pos)
        )
        if horizontal_overlap:
            assert clearance >= gap - 1e-6


def _assert_no_layout_overlaps(positions, *, min_gap: float = 0.08) -> None:
    from visualizer.computation_graph import _node_content_left, _node_content_right

    for left_index, left in enumerate(positions):
        for right in positions[left_index + 1 :]:
            left_bounds = (_node_content_left(left), _node_content_right(left), left.bottom, left.top_y)
            right_bounds = (_node_content_left(right), _node_content_right(right), right.bottom, right.top_y)
            horizontal_overlap = left_bounds[1] + min_gap > right_bounds[0] and right_bounds[1] + min_gap > left_bounds[0]
            vertical_overlap = left_bounds[2] - min_gap < right_bounds[3] and right_bounds[2] - min_gap < left_bounds[3]
            if horizontal_overlap and vertical_overlap:
                raise AssertionError(
                    f"Overlapping nodes: {left.spec.label!r} and {right.spec.label!r}"
                )


def test_all_detail_tile_text_fits_boxes(tmp_path: Path):
    """Every measured detail tile must contain its label text (regression guard)."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    from visualizer.basic_ops import BasicOpFilter
    from visualizer.computation_graph import (
        _estimate_graph_height,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.extract import load_architecture
    from visualizer.render import COLORS, DIAGRAM_LEFT_MARGIN, _detail_sections_to_render
    from visualizer.render_validate import (
        VALIDATE_MIN_GAP,
        collect_measured_elements,
        finalize_detail_layout,
        validate_render_layout,
    )

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
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"]),
    )
    detail_min_left = DIAGRAM_LEFT_MARGIN + 0.05
    fig, ax = plt.subplots(figsize=(16, 13))
    fig.canvas.draw()
    try:
        for title, tree, input_sublabel in _detail_sections_to_render(spec):
            graph = build_computation_graph(tree, basic_ops=spec.basic_ops)
            measure_graph_node_sizes(ax, graph, input_sublabel=None)
            positions, _ = layout_computation_graph(
                graph,
                cx=4.083,
                top_y=10.0,
                block_w=18.0,
                block_h=_estimate_graph_height(graph),
                content_left=detail_min_left,
            )
            plan = finalize_detail_layout(
                ax,
                graph,
                positions,
                input_sublabel=input_sublabel,
                cx=4.083,
                top_y=10.0,
                detail_fill=COLORS["detail_fill"],
                min_left=detail_min_left,
            )
            elements = collect_measured_elements(
                ax,
                graph,
                positions,
                plan,
                detail_fill=COLORS["detail_fill"],
            )
            report = validate_render_layout(elements, min_gap=VALIDATE_MIN_GAP)
            overflows = [line for line in report.overlaps if line.startswith("text overflows")]
            assert not overflows, f"{title}: {overflows}"
            assert report.ok, f"{title}: {report.overlaps}"
    finally:
        plt.close(fig)


def test_kimi_detailed_all_sections_render_in_svg(tmp_path: Path):
    """Every detail subsection must render visible tiles, not just section titles."""
    import re
    from pathlib import Path

    import matplotlib.pyplot as plt

    from visualizer.basic_ops import BasicOpFilter
    from visualizer.computation_graph import (
        _estimate_graph_height,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.extract import load_architecture
    from visualizer.render import (
        COLORS,
        DIAGRAM_LEFT_MARGIN,
        _detail_sections_to_render,
        render_diagram,
    )
    from visualizer.render_validate import (
        VALIDATE_MIN_GAP,
        collect_measured_elements,
        finalize_detail_layout,
        validate_render_layout,
    )

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
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"]),
    )
    out = render_diagram(spec, tmp_path / "kimi_all_sections.svg", detailed=True)
    svg = out.read_text(encoding="utf-8")

    section_titles = [title for title, _, _ in _detail_sections_to_render(spec)]
    for title in section_titles:
        assert f"<!-- {title} -->" in svg, f"missing section title {title!r}"

    chunk_kda_blocks = [
        "Gated delta rule h",
        "beta",
        "q",
        "k",
        "v",
        "CumSum",
    ]
    for label in chunk_kda_blocks:
        assert f"<!-- {label} -->" in svg, f"chunk_kda pipeline missing block {label!r}"

    # Tensor ports must stay aligned with their targets (v connects to intra, not orphaned).
    assert "<!-- v -->" in svg
    assert "<!-- chunk_kda_fwd_intra -->" in svg or "<!-- Intra-chunk WY -->" in svg

    frame_x_coords = [
        float(match.group(1))
        for match in re.finditer(
            r'style="fill: #f4f6f7; stroke: #566573[^"]*"\s*/>\s*</g>\s*<g id="patch_\d+">\s*<path d="M ([0-9.]+)',
            svg,
        )
    ]
    if not frame_x_coords:
        frame_x_coords = [float(x) for x in re.findall(r'id="patch_\d+"[^>]*>.*?d="M ([0-9.]+)', svg[:50000])]
    detail_frames = [x for x in frame_x_coords if x < 800.0]
    assert detail_frames, "no detail section frames found in SVG"
    assert max(detail_frames) < 400.0, (
        f"detail sections pushed too far right (max frame x={max(detail_frames):.0f})"
    )

    detail_min_left = DIAGRAM_LEFT_MARGIN + 0.05
    fig, ax = plt.subplots(figsize=(16, 13))
    fig.canvas.draw()
    try:
        for title, tree, input_sublabel in _detail_sections_to_render(spec):
            graph = build_computation_graph(tree, basic_ops=spec.basic_ops)
            measure_graph_node_sizes(ax, graph, input_sublabel=None)
            positions, _ = layout_computation_graph(
                graph,
                cx=4.083,
                top_y=10.0,
                block_w=18.0,
                block_h=_estimate_graph_height(graph),
                content_left=detail_min_left,
            )
            plan = finalize_detail_layout(
                ax,
                graph,
                positions,
                input_sublabel=input_sublabel,
                cx=4.083,
                top_y=10.0,
                detail_fill=COLORS["detail_fill"],
                min_left=detail_min_left,
            )
            elements = collect_measured_elements(
                ax,
                graph,
                positions,
                plan,
                detail_fill=COLORS["detail_fill"],
            )
            report = validate_render_layout(elements, min_gap=VALIDATE_MIN_GAP)
            overflows = [line for line in report.overlaps if line.startswith("text overflows")]
            assert not overflows, f"{title}: {overflows[0]}"
            assert report.ok, f"{title}: {report.overlaps}"
            left = min(pos.cx - pos.width / 2 for pos in positions)
            assert left < detail_min_left + 1.5, f"{title} anchored too far right (left={left:.2f})"
    finally:
        plt.close(fig)


def test_detail_sections_share_left_anchor_after_finalize():
    """Every detail subsection anchors its left content edge at ``min_left`` after finalize."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    from visualizer.basic_ops import BasicOpFilter
    from visualizer.computation_graph import (
        _estimate_graph_height,
        _node_content_left,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.extract import load_architecture
    from visualizer.render import COLORS, DIAGRAM_LEFT_MARGIN, _detail_sections_to_render
    from visualizer.render_validate import finalize_detail_layout

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
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"]),
    )
    detail_min_left = DIAGRAM_LEFT_MARGIN + 0.05
    fig, ax = plt.subplots(figsize=(16, 13))
    fig.canvas.draw()
    try:
        left_edges: list[tuple[str, float]] = []
        for title, tree, input_sublabel in _detail_sections_to_render(spec):
            graph = build_computation_graph(tree, basic_ops=spec.basic_ops)
            measure_graph_node_sizes(ax, graph, input_sublabel=None)
            positions, _ = layout_computation_graph(
                graph,
                cx=4.083,
                top_y=10.0,
                block_w=18.0,
                block_h=_estimate_graph_height(graph),
                content_left=detail_min_left,
            )
            finalize_detail_layout(
                ax,
                graph,
                positions,
                input_sublabel=input_sublabel,
                cx=4.083,
                top_y=10.0,
                detail_fill=COLORS["detail_fill"],
                min_left=detail_min_left,
            )
            left = min(_node_content_left(pos) for pos in positions)
            left_edges.append((title, left))
        assert left_edges, "expected at least one detail section"
        reference = left_edges[0][1]
        for title, left in left_edges:
            assert abs(left - reference) < 0.12, (
                f"{title!r} left={left:.3f} != reference {reference:.3f}"
            )
        assert abs(reference - detail_min_left) < 0.02
    finally:
        plt.close(fig)


def test_kimi_detail_sections_input_sources_and_spacing(tmp_path: Path):
    import matplotlib.pyplot as plt
    from pathlib import Path

    from visualizer.basic_ops import BasicOpFilter
    from visualizer.computation_graph import (
        DETAIL_LAYER_GAP,
        SYNTHETIC_INPUT,
        _estimate_graph_height,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.extract import load_architecture
    from visualizer.render import COLORS, _detail_sections_to_render, _format_input_source_sublabel
    from visualizer.render_validate import finalize_detail_layout

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
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"]),
    )
    expected_sources = {
        "KimiDelta Attn": "RMSNorm",
        "KimiMLA Attn": "RMSNorm",
        "KimiMLP": "RMSNorm",
        "KimiSparseMoeBlock": "RMSNorm",
    }
    for title, tree, sublabel in _detail_sections_to_render(spec):
        for prefix, source in expected_sources.items():
            if title.startswith(prefix):
                assert tree.input_source == source, title
                assert sublabel == _format_input_source_sublabel(source), title

    fig, ax = plt.subplots(figsize=(16, 13))
    fig.canvas.draw()
    try:
        for title, tree, input_sublabel in _detail_sections_to_render(spec):
            if not any(title.startswith(prefix) for prefix in expected_sources):
                continue
            graph = build_computation_graph(tree, basic_ops=spec.basic_ops)
            measure_graph_node_sizes(ax, graph, input_sublabel=None)
            positions, _ = layout_computation_graph(
                graph,
                cx=3.5,
                top_y=10.0,
                block_w=18.0,
                block_h=_estimate_graph_height(graph),
                content_left=0.6,
            )
            finalize_detail_layout(
                ax,
                graph,
                positions,
                input_sublabel=input_sublabel,
                cx=3.5,
                top_y=10.0,
                detail_fill=COLORS["detail_fill"],
                min_left=0.6,
            )
            _assert_input_clears_direct_consumers(positions, graph, min_gap=DETAIL_LAYER_GAP)
            input_index = next(
                index for index, node in enumerate(graph.nodes) if node.synthetic == SYNTHETIC_INPUT
            )
            consumers = [positions[target] for source, target in graph.links if source == input_index]
            _assert_no_layout_overlaps([positions[input_index], *consumers], min_gap=DETAIL_LAYER_GAP / 2)
    finally:
        plt.close(fig)


def test_kda_attention_merge_links_are_unlabeled():
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import build_computation_graph

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(attn)
    merge_index = next(
        i
        for i, node in enumerate(graph.nodes)
        if node.block
        and node.block.class_name in {"KernelPipeline", "KernelOp", "KernelOutput"}
        and len([src for src, dst in graph.links if dst == i]) >= 3
    )
    incoming = [src for src, dst in graph.links if dst == merge_index]
    assert len(incoming) >= 3
    assert not any(dst == merge_index for (_, dst) in graph.link_port_labels)


def test_kda_attention_block_shows_delta_rule_not_sdpa():
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import build_computation_graph

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    pipeline = next(child for child in attn.children if child.class_name == "KernelPipeline")
    output = next(child for child in attn.children if child.class_name == "KernelOutput")
    assert "chunk_gla_fwd_o_gk" in output.label
    assert any("kernel pipeline" in detail for detail in pipeline.details)
    assert any("chunk_gated_delta_rule_fwd_h" in step.label for step in pipeline.children)
    nested = build_computation_graph(pipeline)
    nested_labels = {spec.label for spec in nested.nodes if spec.block is not None}
    frame_labels = {frame.label for frame in nested.inline_frames}
    assert "l2norm_fwd" in frame_labels or any("l2norm_fwd" in label for label in nested_labels)
    assert any(label in {"Sum", "Sigmoid", "CumSum"} for label in nested_labels)


def test_chunk_kda_pipeline_inline_frames_stay_column_aligned():
    """Kernel pipeline inline frames must stack in narrow columns with aligned connectors."""
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from pathlib import Path

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        _dock_single_consumer_tensor_ports,
        _estimate_graph_height,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import (
        COLORS,
        DIAGRAM_LEFT_MARGIN,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _connector_points_for_link,
        _path_hits_obstacles,
    )
    from visualizer.text_measure import box_bounds_at
    from visualizer.render_validate import finalize_detail_layout

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    pipeline = next(child for child in attn.children if child.class_name == "KernelPipeline")
    graph = build_computation_graph(pipeline)

    fig, ax = plt.subplots(figsize=(16, 13))
    try:
        measure_graph_node_sizes(ax, graph)
        positions, _ = layout_computation_graph(
            graph,
            cx=5.0,
            top_y=10.0,
            block_w=18.0,
            block_h=_estimate_graph_height(graph),
        )
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=5.0,
            top_y=10.0,
            detail_fill=COLORS["detail_fill"],
            min_left=DIAGRAM_LEFT_MARGIN + 0.05,
        )
        _dock_single_consumer_tensor_ports(positions, graph)
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
        anchors = _anchors_from_detail_plan(positions, plan)

        for frame in graph.inline_frames:
            cxs = [positions[index].cx for index in frame.node_indices]
            assert max(cxs) - min(cxs) < 1e-3, (
                f"{frame.frame_id} members drifted horizontally: spread={max(cxs) - min(cxs):.3f}"
            )

        content_left = min(pos.cx - pos.width / 2 for pos in positions)
        content_right = max(pos.cx + pos.width / 2 for pos in positions)
        assert content_right - content_left < 7.0, (
            f"chunk_kda pipeline too wide ({content_right - content_left:.2f})"
        )

        v_idx = next(i for i, node in enumerate(graph.nodes) if node.label == "v")
        intra_idx = next(i for i, node in enumerate(graph.nodes) if node.label == "chunk_kda_fwd_intra")

        q_idx = next(i for i, node in enumerate(graph.nodes) if node.label == "q")
        q_target = next(t for s, t in graph.links if s == q_idx)
        beta_idx = next(i for i, node in enumerate(graph.nodes) if node.label == "beta")
        beta_target = next(t for s, t in graph.links if s == beta_idx)

        incoming = defaultdict(list)
        for src, tgt in graph.links:
            incoming[tgt].append((src, tgt))

        from visualizer.render import _compute_shared_target_bus_y

        target_bus: dict[int, float] = {}
        for tgt, link_group in incoming.items():
            if len(link_group) < 2:
                continue
            target_anchor = anchors.get(tgt)
            source_anchors = [anchors[src] for src, _ in link_group if src in anchors]
            if target_anchor is None or len(source_anchors) < 2:
                continue
            involved = {tgt, *(src for src, _ in link_group)}
            route_obstacles = [
                anchor for node_index, anchor in anchors.items() if node_index not in involved
            ]
            target_bus[tgt] = _compute_shared_target_bus_y(
                source_anchors,
                target_anchor,
                route_obstacles,
            )

        misaligned = 0
        for src, tgt in graph.links:
            points = _connector_points_for_link(
                graph=graph,
                positions=positions,
                anchors=anchors,
                src=src,
                tgt=tgt,
                link_key=(src, tgt),
                incoming=incoming,
                label_obstacles=[],
                target_bus=target_bus,
                source_bus={},
                merge_entry_x={},
                merge_link_bus={},
                input_index=None,
            )
            if points is None:
                continue
            start_x, start_y = points[0]
            end_x, end_y = points[-1]
            src_anchor = anchors[src]
            tgt_anchor = anchors[tgt]
            if graph.nodes[src].synthetic == "@tensor":
                start_ok = (
                    abs(start_x - src_anchor.cx) < 0.15
                    and abs(start_y - src_anchor.bottom) < 0.15
                )
            else:
                start_ok = (
                    abs(start_x - src_anchor.right) < 0.15
                    or abs(start_x - src_anchor.left) < 0.15
                    or abs(start_y - src_anchor.top) < 0.15
                    or abs(start_y - src_anchor.bottom) < 0.15
                )
            end_ok = (
                abs(end_x - tgt_anchor.cx) < 0.15
                and end_y <= tgt_anchor.top + 0.15
            ) or (
                abs(end_y - (tgt_anchor.top + tgt_anchor.bottom) / 2) < 0.15
                and (
                    abs(end_x - tgt_anchor.left) < 0.15
                    or abs(end_x - tgt_anchor.right) < 0.15
                )
            )
            if not (start_ok and end_ok):
                misaligned += 1
        assert misaligned == 0, f"{misaligned} connectors detached from tile anchors"

        for port_label, port_idx, tgt_idx in [
            ("beta", beta_idx, beta_target),
            ("q", q_idx, q_target),
            ("v", v_idx, intra_idx),
        ]:
            points = _connector_points_for_link(
                graph=graph,
                positions=positions,
                anchors=anchors,
                src=port_idx,
                tgt=tgt_idx,
                link_key=(port_idx, tgt_idx),
                incoming=incoming,
                label_obstacles=[],
                target_bus=target_bus,
                source_bus={},
                merge_entry_x={},
                merge_link_bus={},
                input_index=None,
            )
            assert points is not None, f"{port_label} connector missing"
            aligned = abs(positions[port_idx].cx - positions[tgt_idx].cx) < 0.08
            if aligned:
                assert abs(points[-1][0] - anchors[tgt_idx].cx) < 0.15
                assert points[-1][1] <= anchors[tgt_idx].top + 0.15
                assert abs(points[0][0] - points[-1][0]) < 0.08, (
                    f"{port_label} should use a vertical feed when docked above its target"
                )
            else:
                side_entry = (
                    abs(points[-1][1] - (anchors[tgt_idx].top + anchors[tgt_idx].bottom) / 2) < 0.15
                    and (
                        abs(points[-1][0] - anchors[tgt_idx].left) < 0.15
                        or abs(points[-1][0] - anchors[tgt_idx].right) < 0.15
                    )
                )
                assert side_entry or len(points) >= 4, f"{port_label} connector should route around obstacles"

        g_frame = next(
            f for f in graph.inline_frames if f.frame_id == "chunk_kda_fwd_kda_gate_chunk_cumsum_g"
        )
        softplus_idx = next(
            i for i in g_frame.node_indices if graph.nodes[i].label == "Softplus"
        )
        softplus_anchor = anchors[softplus_idx]
        beta_points = _connector_points_for_link(
            graph=graph,
            positions=positions,
            anchors=anchors,
            src=beta_idx,
            tgt=beta_target,
            link_key=(beta_idx, beta_target),
            incoming=incoming,
            label_obstacles=[],
            target_bus=target_bus,
            source_bus={},
            merge_entry_x={},
            merge_link_bus={},
            input_index=None,
        )
        assert beta_points is not None
        assert not _path_hits_obstacles(
            beta_points,
            [softplus_anchor],
            margin=0.02,
        ), "beta connector crosses Softplus"

        k_frame = next(f for f in graph.inline_frames if f.frame_id == "forward_l2norm_fwd_k")
        inner_indices = set(k_frame.node_indices) - {k_frame.node_indices[-1]}
        inner_obstacles = [anchors[index] for index in inner_indices if index in anchors]

        through_inner = []
        for src, tgt in graph.links:
            if src in inner_indices and tgt in inner_indices:
                continue
            points = _connector_points_for_link(
                graph=graph,
                positions=positions,
                anchors=anchors,
                src=src,
                tgt=tgt,
                link_key=(src, tgt),
                incoming=incoming,
                label_obstacles=[],
                target_bus=target_bus,
                source_bus={},
                merge_entry_x={},
                merge_link_bus={},
                input_index=None,
            )
            if not points:
                continue
            route_obstacles = [
                anchor
                for index, anchor in anchors.items()
                if index in inner_indices and index not in {src, tgt}
            ]
            if (src, tgt) in graph.inline_binary_operand_links:
                from visualizer.computation_graph import _ordered_inline_frame_chain

                frame = next(
                    f
                    for f in graph.inline_frames
                    if src in f.node_indices and tgt in f.node_indices
                )
                chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
                if src in chain and tgt in chain:
                    skipped = set(chain[chain.index(src) + 1 : chain.index(tgt)])
                    route_obstacles = [
                        anchor
                        for index, anchor in anchors.items()
                        if index in inner_indices
                        and index not in {src, tgt}
                        and index not in skipped
                    ]
            if _path_hits_obstacles(points, route_obstacles, margin=0.02):
                through_inner.append((graph.nodes[src].label, graph.nodes[tgt].label, src))
        assert not through_inner, f"connectors cut through k l2norm ops: {through_inner}"

        g_frame = next(
            f for f in graph.inline_frames if f.frame_id == "chunk_kda_fwd_kda_gate_chunk_cumsum_g"
        )
        multiply_idx = next(
            i for i in g_frame.node_indices if graph.nodes[i].label == "×"
        )
        exp_idx = next(i for i in g_frame.node_indices if graph.nodes[i].label == "Exp")
        softplus_idx = next(i for i in g_frame.node_indices if graph.nodes[i].label == "Softplus")
        assert (softplus_idx, multiply_idx) in graph.links
        assert (exp_idx, multiply_idx) in graph.links
        assert (exp_idx, multiply_idx) in graph.side_entry_links

        exp_to_mul = _connector_points_for_link(
            graph=graph,
            positions=positions,
            anchors=anchors,
            src=exp_idx,
            tgt=multiply_idx,
            link_key=(exp_idx, multiply_idx),
            incoming=incoming,
            label_obstacles=[],
            target_bus=target_bus,
            source_bus={},
            merge_entry_x={},
            merge_link_bus={},
            input_index=None,
        )
        assert exp_to_mul is not None, "Exp should connect to × as second operand"
    finally:
        plt.close(fig)


def test_kda_v_tensor_connector_avoids_chunk_gated_box():
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import build_computation_graph, layout_computation_graph, measure_graph_node_sizes
    from visualizer.render import (
        COLORS,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _compute_shared_target_bus_y,
        _connector_points_for_link,
        _polyline_bounds,
    )
    from visualizer.render_validate import finalize_detail_layout, collect_measured_elements
    from visualizer.text_measure import ContentBounds

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.for_detailed()
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    pipeline = next(child for child in attn.children if child.class_name == "KernelPipeline")
    graph = build_computation_graph(pipeline)

    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        measure_graph_node_sizes(ax, graph)
        positions, _ = layout_computation_graph(graph, cx=5.0, top_y=10.0, block_w=8.0)
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=5.0,
            top_y=10.0,
            detail_fill=COLORS["detail_fill"],
        )
        v_idx = next(i for i, node in enumerate(graph.nodes) if node.label == "v")
        intra_idx = next(i for i, node in enumerate(graph.nodes) if node.label == "chunk_kda_fwd_intra")
        gated_idx = next(i for i, node in enumerate(graph.nodes) if "chunk_gated" in node.label)
        positions[v_idx].cx = positions[intra_idx].cx

        plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
        anchors = _anchors_from_detail_plan(positions, plan)
        incoming = defaultdict(list)
        for src, tgt in graph.links:
            incoming[tgt].append((src, tgt))
        target_bus = {}
        for tgt, link_group in incoming.items():
            if len(link_group) < 2:
                continue
            involved = {tgt, *(src for src, _ in link_group)}
            route_obstacles = [anchors[i] for i in anchors if i not in involved]
            target_bus[tgt] = _compute_shared_target_bus_y(
                [anchors[src] for src, _ in link_group],
                anchors[tgt],
                route_obstacles,
            )

        points = _connector_points_for_link(
            graph=graph,
            positions=positions,
            anchors=anchors,
            src=v_idx,
            tgt=intra_idx,
            link_key=(v_idx, intra_idx),
            incoming=incoming,
            label_obstacles=[],
            target_bus=target_bus,
            source_bus={},
            merge_entry_x={},
            merge_link_bus={},
            input_index=None,
        )
        assert points is not None
        gated = positions[gated_idx]
        gated_bounds = ContentBounds(
            left=gated.cx - gated.width / 2,
            right=gated.cx + gated.width / 2,
            bottom=gated.top_y - gated.height,
            top=gated.top_y,
        )
        connector_bounds = _polyline_bounds(points, half_width=0.04)
        assert not connector_bounds.overlaps(gated_bounds), points
    finally:
        plt.close(fig)


def test_mla_attention_block_keeps_sdpa_label():
    from pathlib import Path
    from visualizer.ast_analyze import SYNTHETIC_ATTENTION, analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiMLAAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    merge = next(child for child in attn.children if child.attr_name == SYNTHETIC_ATTENTION)
    assert merge.label == "Attention"
    assert not any("scaled dot-product" in detail for detail in merge.details)


def test_mlp_situ_and_mul_steps_are_labeled():
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node, block_purpose
    from visualizer.computation_graph import build_computation_graph

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.for_detailed()
    mlp = build_block_node(
        attr_name="shared_experts",
        class_name="KimiMLP",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    situ = next(child for child in mlp.children if child.class_name == "SituAndMul")
    assert block_purpose(situ) == "Situ(gate) × up branch"
    graph = build_computation_graph(mlp)
    labels = [spec.label for spec in graph.nodes]
    assert "Split gate | up" not in labels
    assert "Linear" in labels
    assert "Situ" in labels
    assert "×" in labels
    frame = next(frame for frame in graph.inline_frames if frame.frame_id == "act_fn")
    assert frame.sublabel is None
    mul_index = next(i for i, spec in enumerate(graph.nodes) if spec.label == "×")
    up_index = next(i for i, spec in enumerate(graph.nodes) if spec.block and spec.block.attr_name == "up_proj")
    situ_index = next(i for i, spec in enumerate(graph.nodes) if spec.label == "Situ")
    assert (situ_index, mul_index) in graph.links
    assert (up_index, mul_index) in graph.links
    assert (up_index, mul_index) in graph.side_entry_links
    assert (up_index, mul_index) not in graph.dashed_links


def test_fork_join_branch_layout_is_horizontal():
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
        _estimate_graph_height,
    )
    from visualizer.render import COLORS
    from visualizer.render_validate import finalize_detail_layout

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.for_detailed()
    mlp = build_block_node(
        attr_name="shared_experts",
        class_name="KimiMLP",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(mlp, basic_ops=basic)
    fig, ax = plt.subplots(figsize=(16, 13))
    fig.canvas.draw()
    measure_graph_node_sizes(ax, graph)
    positions, _ = layout_computation_graph(
        graph,
        cx=2.6,
        top_y=10.0,
        block_w=8.0,
        block_h=_estimate_graph_height(graph),
        content_left=0.6,
    )
    finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=None,
        cx=2.6,
        top_y=10.0,
        detail_fill=COLORS["detail_fill"],
        min_left=0.6,
    )

    gate = next(i for i, node in enumerate(graph.nodes) if node.block and node.block.attr_name == "gate_proj")
    up = next(i for i, node in enumerate(graph.nodes) if node.block and node.block.attr_name == "up_proj")
    situ = next(i for i, node in enumerate(graph.nodes) if node.label == "Situ")
    mul = next(i for i, node in enumerate(graph.nodes) if node.label == "×")

    gate_cy = (positions[gate].top_y + positions[gate].bottom) / 2
    up_cy = (positions[up].top_y + positions[up].bottom) / 2
    situ_cy = (positions[situ].top_y + positions[situ].bottom) / 2
    mul_cy = (positions[mul].top_y + positions[mul].bottom) / 2

    assert abs(gate_cy - up_cy) > 0.08
    assert positions[up].cx > positions[mul].cx
    assert abs(positions[situ].cx - positions[mul].cx) < 0.05
    assert abs(up_cy - situ_cy) < 0.05
    assert positions[up].top_y <= positions[gate].top_y + 0.05
    assert positions[up].top_y >= positions[mul].top_y - 0.05
    plt.close()


def test_moe_and_situ_expand_in_basic_only_detailed():
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import build_computation_graph

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.for_detailed()

    mlp = build_block_node(
        attr_name="shared_experts",
        class_name="KimiMLP",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    mlp_graph = build_computation_graph(mlp, basic_ops=basic)
    mlp_labels = [spec.label for spec in mlp_graph.nodes]
    assert "Situ" in mlp_labels
    assert "×" in mlp_labels
    assert any(frame.frame_id == "act_fn" for frame in mlp_graph.inline_frames)

    moe = build_block_node(
        attr_name="block_sparse_moe",
        class_name="KimiSparseMoeBlock",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    moe_graph = build_computation_graph(moe, basic_ops=basic)
    moe_labels = [spec.label for spec in moe_graph.nodes]
    assert moe_labels[:9] == [
        "hidden_states",
        "Linear",
        "Sigmoid",
        "Expert bias",
        "Group routing",
        "Top-k experts",
        "Gather weights",
        "Renormalize",
        "Route scaling",
    ]
    assert "MoE aggregation" in moe_labels
    assert "×" in moe_labels
    assert "Situ" in moe_labels
    gate_frame = next(frame for frame in moe_graph.inline_frames if frame.frame_id == "gate")
    assert gate_frame.label == "KimiMoEGate"
    assert len(gate_frame.node_indices) == 8
    shared_frame = next(frame for frame in moe_graph.inline_frames if frame.frame_id == "shared_experts")
    assert shared_frame.label == "shared_experts"
    assert any(frame.frame_id == "act_fn" for frame in moe_graph.inline_frames)
    act_fn_frame = next(frame for frame in moe_graph.inline_frames if frame.frame_id == "act_fn")
    assert act_fn_frame.label == "SituAndMul"
    assert set(act_fn_frame.node_indices).issubset(set(shared_frame.node_indices))


def test_kda_output_gate_and_gated_norm_expand_in_graph():
    from pathlib import Path

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node, block_purpose
    from visualizer.computation_graph import build_computation_graph

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    gate = next(child for child in attn.children if child.attr_name == "g_proj")
    assert gate.class_name == "Linear"
    assert gate.is_basic
    assert not gate.children

    graph = build_computation_graph(attn, basic_ops=basic)
    labels = [spec.label for spec in graph.nodes]
    assert "Linear" in labels
    assert "Reshape" not in labels
    assert "RMSNorm" in labels
    assert "×" in labels
    assert "Sigmoid" not in labels
    assert not any(frame.frame_id == "g_proj" for frame in graph.inline_frames)
    assert not any(spec.label == "Output gate" for spec in graph.nodes)

    gate_producer_indices = [
        i
        for i, spec in enumerate(graph.nodes)
        if spec.block and spec.block.attr_name == "g_proj"
    ]
    assert gate_producer_indices
    combine_index = next(i for i, spec in enumerate(graph.nodes) if spec.label == "×")
    assert graph.nodes[combine_index].sublabel in (None, "")
    side_into_combine = [
        src
        for src, dst in graph.links
        if dst == combine_index and (src, dst) in graph.side_entry_links
    ]
    assert side_into_combine
    assert all(src in gate_producer_indices for src in side_into_combine)
    assert not any((src, combine_index) in graph.dashed_links for src in side_into_combine)


def test_kda_gated_norm_spine_is_center_aligned():
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        _estimate_graph_height,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import COLORS, DIAGRAM_LEFT_MARGIN
    from visualizer.render_validate import finalize_detail_layout

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.for_detailed()
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(attn, basic_ops=basic)
    fig, ax = plt.subplots()
    measure_graph_node_sizes(ax, graph)
    min_left = DIAGRAM_LEFT_MARGIN + 0.05
    positions, _ = layout_computation_graph(
        graph,
        cx=3.5,
        top_y=10.0,
        block_w=6.0,
        block_h=_estimate_graph_height(graph),
        content_left=min_left,
    )
    plan = finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=None,
        cx=3.5,
        top_y=10.0,
        detail_fill=COLORS["detail_fill"],
        min_left=min_left,
    )
    from visualizer.render import _anchors_from_detail_plan

    anchors = _anchors_from_detail_plan(positions, plan)
    by_attr = {
        pos.spec.block.attr_name: (index, pos)
        for index, pos in enumerate(positions)
        if pos.spec.block is not None
    }
    spine = [
        by_attr[name][1].cx
        for name in ("@attn_output", "o_norm", "o_proj")
        if name in by_attr
    ]
    combine_index = next(
        index for index, pos in enumerate(positions) if pos.spec.label == "×"
    )
    combine = positions[combine_index].cx
    spine.append(combine)
    assert len(spine) == 4
    assert max(spine) - min(spine) < 0.02
    combine_op_x = next(op_x for op_x, _, _, _ in plan.combine_ops)
    assert abs(combine_op_x - combine) < 0.02
    for name in ("@attn_chunk", "o_norm", "o_proj"):
        if name not in by_attr:
            continue
        index, pos = by_attr[name]
        assert abs(anchors[index].cx - pos.cx) < 0.02
    assert abs(anchors[combine_index].cx - combine) < 0.02
    plt.close()


def test_kda_fanout_to_chunk_kda_horizontal_gap_is_tight():
    """Main-spine chunk_kda should sit near parallel fan-out tiles without a reserved column band."""
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        _estimate_graph_height,
        _node_content_left,
        _node_content_right,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import COLORS, DIAGRAM_LEFT_MARGIN
    from visualizer.render_validate import VALIDATE_MIN_GAP, _place_layout_zones, finalize_detail_layout
    from visualizer.sizing import min_horizontal_block_gap

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(attn, basic_ops=basic)
    min_left = DIAGRAM_LEFT_MARGIN + 0.05
    fig, ax = plt.subplots(figsize=(16, 13))
    try:
        measure_graph_node_sizes(ax, graph)
        positions, _ = layout_computation_graph(
            graph,
            cx=3.5,
            top_y=10.0,
            block_w=18.0,
            block_h=_estimate_graph_height(graph),
            content_left=min_left,
        )
        chunk_idx = next(
            index for index, node in enumerate(graph.nodes) if node.label == "chunk_kda pipeline"
        )
        k_idx = next(
            index
            for index, node in enumerate(graph.nodes)
            if node.block and node.block.attr_name == "k_proj"
        )
        _place_layout_zones(
            positions,
            graph,
            cx=3.5,
            min_gap=VALIDATE_MIN_GAP,
            min_left=min_left,
        )
        gap_after_first = _node_content_left(positions[chunk_idx]) - _node_content_right(
            positions[k_idx]
        )
        _place_layout_zones(
            positions,
            graph,
            cx=3.5,
            min_gap=VALIDATE_MIN_GAP,
            min_left=min_left,
        )
        gap_after_second = _node_content_left(positions[chunk_idx]) - _node_content_right(
            positions[k_idx]
        )
        assert abs(gap_after_second - gap_after_first) < 0.02

        positions, _ = layout_computation_graph(
            graph,
            cx=3.5,
            top_y=10.0,
            block_w=18.0,
            block_h=_estimate_graph_height(graph),
            content_left=min_left,
        )
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=3.5,
            top_y=10.0,
            detail_fill=COLORS["detail_fill"],
            min_left=min_left,
        )
        edge_gap = _node_content_left(positions[chunk_idx]) - _node_content_right(
            positions[k_idx]
        )
        cx_gap = abs(positions[chunk_idx].cx - positions[k_idx].cx)
        zone_gap = max(VALIDATE_MIN_GAP * 2, min_horizontal_block_gap())
        assert cx_gap < zone_gap * 12, (
            f"k_proj to chunk_kda cx gap {cx_gap:.3f} too wide (zone_gap={zone_gap:.3f})"
        )
        assert edge_gap < zone_gap * 6 or cx_gap < zone_gap * 6, (
            f"k_proj to chunk_kda layout too wide (edge={edge_gap:.3f}, cx={cx_gap:.3f})"
        )
    finally:
        plt.close(fig)


def test_kda_hidden_states_fanout_uses_shared_source_bus_with_vertical_tees():
    """Parallel linear feeds from hidden_states share one horizontal bus with vertical tees."""
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        _estimate_graph_height,
        _fanout_branch_index,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import (
        COLORS,
        DIAGRAM_LEFT_MARGIN,
        MERGE_RADIUS,
        PARALLEL_CONNECTOR_COORD_EPS,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _collect_detail_link_paths,
        _collect_connector_join_points,
        _compute_detail_connector_buses,
        _connector_target_top_entry_y,
    )
    from visualizer.render_validate import finalize_detail_layout

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.for_detailed()
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(attn, basic_ops=basic)
    input_index = next(i for i, node in enumerate(graph.nodes) if node.synthetic == "@input")
    fig, ax = plt.subplots(figsize=(16, 13))
    try:
        measure_graph_node_sizes(ax, graph)
        min_left = DIAGRAM_LEFT_MARGIN + 0.05
        positions, links = layout_computation_graph(
            graph,
            cx=3.5,
            top_y=10.0,
            block_w=18.0,
            block_h=_estimate_graph_height(graph),
            content_left=min_left,
        )
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=3.5,
            top_y=10.0,
            detail_fill=COLORS["detail_fill"],
            min_left=min_left,
        )
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
        anchors = _anchors_from_detail_plan(positions, plan)
        incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
        outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for src, tgt in links:
            incoming[tgt].append((src, tgt))
            outgoing[src].append((src, tgt))
        target_bus, source_bus, merge_entry_x, merge_link_bus = _compute_detail_connector_buses(
            graph,
            positions,
            anchors,
            incoming,
            outgoing,
            plan.label_obstacles,
        )
        assert input_index in source_bus
        shared_bus_y = source_bus[input_index]
        link_paths = _collect_detail_link_paths(
            graph=graph,
            links=links,
            positions=positions,
            anchors=anchors,
            incoming=incoming,
            label_obstacles=plan.label_obstacles,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_entry_x=merge_entry_x,
            merge_link_bus=merge_link_bus,
            input_index=input_index,
        )
        fanout_links = [
            (src, tgt)
            for src, tgt in links
            if src == input_index
            and (
                _fanout_branch_index(graph.nodes[tgt]) is not None
                or graph.nodes[tgt].key.startswith("sideproducer")
            )
        ]
        assert len(fanout_links) >= 4
        for src, tgt in fanout_links:
            target = anchors[tgt]
            points = link_paths[(src, tgt)]
            end_x, end_y = points[-1]
            assert abs(end_x - target.cx) < 0.08
            assert abs(end_y - _connector_target_top_entry_y(target)) < 0.02
            horiz_ys = [
                y1
                for (x1, y1), (x2, y2) in zip(points, points[1:])
                if abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS
                and abs(x1 - x2) > PARALLEL_CONNECTOR_COORD_EPS
            ]
            assert any(abs(y - shared_bus_y) <= PARALLEL_CONNECTOR_COORD_EPS for y in horiz_ys), (
                f"expected shared source bus at y={shared_bus_y:.3f} for fan-out to {tgt}, got {horiz_ys}"
            )
        join_points = _collect_connector_join_points(
            link_paths,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            graph=graph,
            outgoing=outgoing,
            anchors=anchors,
        )
        input_x = anchors[input_index].cx
        assert not any(
            abs(x - input_x) < 0.12 and abs(y - shared_bus_y) < PARALLEL_CONNECTOR_COORD_EPS
            for x, y in join_points
        )

        g_index = next(
            index for index, node in enumerate(graph.nodes) if node.block and node.block.attr_name == "g_proj"
        )
        combine_index = next(index for index, node in enumerate(graph.nodes) if node.label == "×")
        side_points = link_paths[(g_index, combine_index)]
        assert side_points[-1][0] >= positions[combine_index].cx + MERGE_RADIUS - 0.02
    finally:
        plt.close(fig)


def test_moe_plus_is_spine_aligned_with_sigma():
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        DETAIL_LAYER_GAP,
        _estimate_graph_height,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import COLORS, _anchors_from_detail_plan
    from visualizer.render_validate import finalize_detail_layout

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.for_detailed()
    moe = build_block_node(
        attr_name="block_sparse_moe",
        class_name="KimiSparseMoeBlock",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(moe, basic_ops=basic)
    fig, ax = plt.subplots(figsize=(16, 13))
    fig.canvas.draw()
    measure_graph_node_sizes(ax, graph)
    positions, _ = layout_computation_graph(
        graph,
        cx=2.6,
        top_y=10.0,
        block_w=8.0,
        block_h=_estimate_graph_height(graph),
        content_left=0.6,
    )
    plan = finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=None,
        cx=2.6,
        top_y=10.0,
        detail_fill=COLORS["detail_fill"],
        min_left=0.6,
    )
    anchors = _anchors_from_detail_plan(positions, plan)
    by_label = {spec.label: index for index, spec in enumerate(graph.nodes)}
    up_index = next(
        index
        for index, spec in enumerate(graph.nodes)
        if spec.block and spec.block.attr_name == "routed_expert_up_proj"
    )
    spine = [positions[index].cx for index in (by_label["MoE aggregation"], up_index, by_label["+"])]
    draw = {symbol: x for x, _, symbol, _ in plan.combine_ops}
    assert max(spine) - min(spine) < 0.02
    assert abs(positions[by_label["MoE aggregation"]].cx - spine[0]) < 0.02
    assert abs(draw["+"] - positions[by_label["+"]].cx) < 0.02
    assert abs(anchors[by_label["+"]].cx - positions[by_label["+"]].cx) < 0.02
    gap = positions[up_index].bottom - positions[by_label["+"]].top_y
    assert abs(gap - DETAIL_LAYER_GAP) < 0.02
    plt.close()


def test_top_level_block_centers_attention_under_rmsnorm():
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.extract import load_architecture
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.render import DiagramLayout, _block_content_widths, _layout_component_block

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
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"]),
    )
    fig, ax = plt.subplots(figsize=(11, 13))
    fig.canvas.draw()
    norm_w, inner_w = _block_content_widths(ax, spec)
    spine_cx = 2.0
    layout = DiagramLayout()
    _layout_component_block(
        layout,
        ax,
        cx=spine_cx,
        top_y=10.0,
        block_w=4.0,
        spec=spec,
        norm_w=norm_w,
        inner_w=inner_w,
    )
    by_id = {node.node_id: node for node in layout.nodes}
    norm = by_id["input_layernorm"]
    attn = by_id["self_attn"]
    assert abs(norm.cx - spine_cx) <= 0.02
    assert abs(attn.cx - spine_cx) <= 0.02
    assert abs(norm.cx - attn.cx) <= 0.02
    plt.close(fig)


def test_layout_center_aligns_vertical_chains():
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        _estimate_graph_height,
        build_computation_graph,
        layout_computation_graph,
    )

    code = '''
class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_a_proj = nn.Linear(8, 8)
        self.q_a_layernorm = nn.LayerNorm(8)
        self.q_b_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)
    def forward(self, hidden_states):
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        attn_output = attention_interface(self, q_states, q_states, q_states, None)
        return self.o_proj(attn_output)
'''
    analysis = analyze_source(code, filename="attn.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^LayerNorm$"])
    tree = build_block_node(
        attr_name="attn",
        class_name="Attn",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(tree)
    positions, _ = layout_computation_graph(
        graph,
        cx=5.0,
        top_y=10.0,
        block_w=6.0,
        block_h=_estimate_graph_height(graph),
    )
    by_attr = {
        pos.spec.block.attr_name: pos
        for pos in positions
        if pos.spec.block is not None
    }
    q_chain = [
        by_attr[name].cx
        for name in ("q_a_proj", "q_a_layernorm", "q_b_proj")
        if name in by_attr
    ]
    assert len(q_chain) == 3
    assert max(q_chain) - min(q_chain) < 0.02


def test_layout_fanout_branch_order_reduces_crossings():
    from visualizer.block_tree import BlockNode
    from visualizer.computation_graph import (
        GraphNodeSpec,
        ComputationGraph,
        _count_layout_crossings,
        _optimize_layer_order,
        _topological_layers,
    )

    graph = ComputationGraph(
        nodes=[
            GraphNodeSpec(key="input", label="hidden_states", synthetic="@input"),
            GraphNodeSpec(key="q", block=BlockNode(attr_name="q", class_name="Linear", role="other", label="Q", is_basic=True)),
            GraphNodeSpec(key="k", block=BlockNode(attr_name="k", class_name="Linear", role="other", label="K", is_basic=True)),
            GraphNodeSpec(key="v", block=BlockNode(attr_name="v", class_name="Linear", role="other", label="V", is_basic=True)),
            GraphNodeSpec(key="attn", block=BlockNode(attr_name="@attention", class_name="AttentionOp", role="attention", label="Attention", is_basic=True)),
        ],
        links=[(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)],
    )
    layers = _topological_layers(graph)
    scrambled = [layers[0], [layers[1][2], layers[1][0], layers[1][1]], layers[2]]
    optimized = _optimize_layer_order(scrambled, graph)
    assert _count_layout_crossings(optimized, graph) <= _count_layout_crossings(scrambled, graph)


def test_layout_computation_graph_avoids_box_overlaps():
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        _estimate_graph_height,
        build_computation_graph,
        layout_computation_graph,
    )

    code = '''
class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_a_proj = nn.Linear(8, 8)
        self.q_a_layernorm = nn.LayerNorm(8)
        self.q_b_proj = nn.Linear(8, 8)
        self.kv_a_proj_with_mqa = nn.Linear(8, 8)
        self.kv_a_layernorm = nn.LayerNorm(8)
        self.kv_b_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)
    def forward(self, hidden_states):
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        attn_output = attention_interface(self, q_states, k_pass, k_pass, None)
        return self.o_proj(attn_output)
'''
    analysis = analyze_source(code, filename="attn.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^LayerNorm$"])
    tree = build_block_node(
        attr_name="attn",
        class_name="Attn",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(tree)
    est_h = _estimate_graph_height(graph)
    positions, _links = layout_computation_graph(
        graph,
        cx=5.0,
        top_y=10.0,
        block_w=6.0,
        block_h=est_h,
    )
    _assert_no_layout_overlaps(positions)


def test_fanout_chain_expands_straight_line_wrapper():
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import build_computation_graph

    code = '''
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16)
        self.up_proj = nn.Linear(8, 16)
        self.down_proj = nn.Linear(16, 8)
    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.mlp_branch = MLP()
        self.o_proj = nn.Linear(8, 8)
    def forward(self, hidden_states):
        q = self.q_proj(hidden_states)
        side = self.mlp_branch(hidden_states)
        return self.o_proj(q + side)
'''
    analysis = analyze_source(code, filename="block.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$"])
    tree = build_block_node(
        attr_name="block",
        class_name="Block",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(tree)
    labels = [node.block.attr_name for node in graph.nodes if node.block is not None]
    assert "gate_proj" in labels
    assert "up_proj" in labels
    assert "down_proj" in labels
    assert any(frame.frame_id == "mlp_branch" for frame in graph.inline_frames)


def test_finalize_detail_layout_measures_and_validates(tmp_path: Path):
    import matplotlib.pyplot as plt
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import _estimate_graph_height, build_computation_graph, layout_computation_graph, measure_graph_node_sizes
    from visualizer.render import COLORS, _build_detail_draw_plan
    from visualizer.render_validate import collect_measured_elements, finalize_detail_layout, validate_render_layout

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    tree = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    try:
        graph = build_computation_graph(tree)
        measure_graph_node_sizes(ax, graph, input_sublabel=None)
        est_h = _estimate_graph_height(graph)
        positions, _links = layout_computation_graph(
            graph,
            cx=5.0,
            top_y=20.0,
            block_w=6.0,
            block_h=est_h,
        )
        plan = finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=5.0,
            top_y=20.0,
            detail_fill=COLORS["detail_fill"],
        )
        elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=COLORS["detail_fill"])
        report = validate_render_layout(elements)
        assert report.ok, report.overlaps + report.invisible
        assert plan.node_draws
        assert all(leaf.w > 0 and leaf.h > 0 for leaf, _ in plan.node_draws)
    finally:
        plt.close(fig)


def test_measure_graph_node_sizes_fits_kda_tile_labels():
    import matplotlib.pyplot as plt
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        _estimate_graph_height,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import COLORS, _build_detail_draw_plan
    from visualizer.render_validate import collect_measured_elements, finalize_detail_layout

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    tree = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    try:
        graph = build_computation_graph(tree)
        measure_graph_node_sizes(ax, graph, input_sublabel=None)
        positions, _ = layout_computation_graph(
            graph,
            cx=5.0,
            top_y=20.0,
            block_w=6.0,
            block_h=_estimate_graph_height(graph),
        )
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=5.0,
            top_y=20.0,
            detail_fill=COLORS["detail_fill"],
        )
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
        elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=COLORS["detail_fill"])
        overflows = [element for element in elements if element.kind == "text_overflow"]
        assert not overflows, [element.label for element in overflows]
    finally:
        plt.close(fig)


def test_kda_graph_basic_only_shows_linears_and_norms():
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import build_computation_graph

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.for_detailed()
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(attn, basic_ops=basic)
    labels = {spec.label for spec in graph.nodes if spec.block is not None}
    assert "Linear" in labels
    assert "RMSNorm" in labels
    assert "Depthwise Conv" in labels
    assert "Silu" in labels
    assert "Merge inputs" not in labels
    assert "chunk_kda pipeline" in labels
    assert any("chunk_gla_fwd_o_gk" in label for label in labels)
    assert not any("l2norm_fwd" in label for label in labels)
    assert not any(frame.label == "chunk_kda pipeline" for frame in graph.inline_frames)
    assert "Attention" not in labels
    assert "KDA" not in labels
    assert "×" in {spec.label for spec in graph.nodes}
    pipeline = next(child for child in attn.children if child.class_name == "KernelPipeline")
    pipeline_graph = build_computation_graph(pipeline)
    pipeline_labels = {spec.label for spec in pipeline_graph.nodes if spec.block is not None}
    pipeline_frames = {frame.label for frame in pipeline_graph.inline_frames}
    assert "l2norm_fwd" in pipeline_frames or any("l2norm_fwd" in label for label in pipeline_labels)
    assert any(label in {"Sum", "Sigmoid", "CumSum", "× scale"} for label in pipeline_labels)
    assert "fused_beta_sigmoid" in pipeline_frames or any(
        "fused_beta_sigmoid" in label for label in pipeline_labels
    )
    assert any("kda_gate_chunk_cumsum" in label for label in pipeline_labels) or (
        "kda_gate_chunk_cumsum" in pipeline_frames
    )
    assert any("chunk_gated_delta_rule_fwd_h" in label for label in pipeline_labels)
    assert not any("=" in label for label in pipeline_labels)
    for spec in graph.nodes:
        if spec.block and spec.block.class_name == "Linear":
            assert spec.sublabel in (None, "")
            assert spec.label == "Linear"


def test_kda_tile_labels_fit_when_internals_render_below_fact_sheet():
    import matplotlib.pyplot as plt
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        _estimate_graph_height,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import (
        COLORS,
        DETAIL_MIN_BLOCK_W,
        DIAGRAM_LEFT_MARGIN,
        PANEL_W,
        _build_detail_draw_plan,
        _detail_layout_geometry,
        _fact_sheet_x,
    )
    from visualizer.render_validate import (
        collect_measured_elements,
        finalize_detail_layout,
        measure_detail_tree_content_width,
        measure_max_detail_content_width,
    )

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    tree = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )

    detail_min_left = DIAGRAM_LEFT_MARGIN + 0.05
    fact_x = _fact_sheet_x(5.0)
    fig, ax = plt.subplots(figsize=(16, 13))
    ax.set_xlim(0, 16)
    ax.set_ylim(-3, 5)
    try:
        fig.canvas.draw()
        detail_content_w = measure_max_detail_content_width(
            ax,
            [("KimiDelta Attn", tree)],
            cx=3.5,
            detail_fill=COLORS["detail_fill"],
            min_left=detail_min_left,
        )
        canvas_width, below_fact_sheet = _detail_layout_geometry(
            11.0,
            fact_x=fact_x,
            fact_w=PANEL_W,
            detail_content_width=detail_content_w,
        )
        section_w = measure_detail_tree_content_width(
            ax,
            tree,
            cx=3.5,
            detail_fill=COLORS["detail_fill"],
            min_left=detail_min_left,
        )
        assert below_fact_sheet
        assert 2.0 <= section_w <= 16.0
        ax.set_xlim(0, canvas_width)
        fig.canvas.draw()

        graph = build_computation_graph(tree)
        measure_graph_node_sizes(ax, graph, input_sublabel=None)
        positions, _ = layout_computation_graph(
            graph,
            cx=3.5,
            top_y=3.0,
            block_w=section_w,
            block_h=_estimate_graph_height(graph),
        )
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=3.5,
            top_y=3.0,
            detail_fill=COLORS["detail_fill"],
            min_left=detail_min_left,
            forbidden_regions=None,
        )
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
        elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=COLORS["detail_fill"])
        overflows = [element for element in elements if element.kind == "text_overflow"]
        assert not overflows, [element.label for element in overflows]
    finally:
        plt.close(fig)


def test_uniform_detail_section_width_is_max_shrink_wrap():
    import matplotlib.pyplot as plt
    from pathlib import Path
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.extract import load_architecture
    from visualizer.render import COLORS, DIAGRAM_LEFT_MARGIN, _detail_sections_to_render
    from visualizer.render_validate import (
        measure_detail_tree_content_width,
        measure_uniform_detail_section_width,
    )

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
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"]),
    )
    detail_min_left = DIAGRAM_LEFT_MARGIN + 0.05
    fig, ax = plt.subplots(figsize=(16, 13))
    ax.set_xlim(0, 16)
    ax.set_ylim(-3, 5)
    try:
        fig.canvas.draw()
        cx = 3.5
        individual = [
            measure_detail_tree_content_width(
                ax,
                tree,
                cx=cx,
                detail_fill=COLORS["detail_fill"],
                min_left=detail_min_left,
                input_sublabel=input_sublabel,
            )
            for _title, tree, input_sublabel in _detail_sections_to_render(spec)
        ]
        uniform = measure_uniform_detail_section_width(
            ax,
            spec,
            cx=cx,
            detail_fill=COLORS["detail_fill"],
            min_left=detail_min_left,
        )
        assert individual
        assert uniform == max(individual)
        assert uniform > min(individual)
    finally:
        plt.close(fig)


def test_box_label_size_matches_draw_box_height():
    import matplotlib.pyplot as plt
    from visualizer.sizing import two_line_box_height
    from visualizer.text_measure import box_label_size

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    try:
        _width, height = box_label_size(ax, "Linear", "Output gate — scales normalized output", fontsize=7.6)
        assert height > two_line_box_height() * 0.9
    finally:
        plt.close(fig)


def test_chunk_kda_tensor_ports_include_upstream_hints():
    from pathlib import Path

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import SYNTHETIC_TENSOR, build_computation_graph

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    pipeline = next(child for child in attn.children if child.class_name == "KernelPipeline")
    graph = build_computation_graph(pipeline)

    port_hints = {
        spec.label: spec.sublabel
        for spec in graph.nodes
        if spec.synthetic == SYNTHETIC_TENSOR
    }
    assert port_hints == {
        "q": "← Conv1d",
        "k": "← Conv1d",
        "v": "← Conv1d",
        "g": "← Linear",
        "beta": "← Linear",
    }
