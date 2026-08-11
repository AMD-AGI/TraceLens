"""Tests for detailed block tree expansion."""

from pathlib import Path

from visualizer.ast_analyze import analyze_source
from visualizer.basic_ops import BasicOpFilter
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


def test_build_decoder_block_trees_for_custom_model():
    source = (FIXTURES / "custom_model" / "modeling_custom.py").read_text(encoding="utf-8")
    analysis = analyze_source(source, filename="modeling_custom.py")
    basic_ops = BasicOpFilter.from_cli(add=[r"(?i)^Linear$"])

    trees, standalone_wrappers = build_decoder_block_trees(
        analysis.block_components, analysis.class_registry, basic_ops
    )
    titles = [title for title, _ in trees]
    assert any("MLA" in title or "Latent" in title for title in titles)
    assert any("MoE" in title for title in titles)

    attn_tree = next(tree for title, tree in trees if "self_attn" in title)
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
    assert "Q" in attention_inputs
    assert attention_inputs["Q"][:2] == ["q_a_proj", "q_a_layernorm"]


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
    assert segments[0].branches[0].label == "Q"
    assert segments[0].branches[1].label == "KV"
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
    assert attention_inputs["Q"] == ["q_a_proj", "q_a_layernorm", "q_b_proj"]
    assert attention_inputs["K"] == ["kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj"]
    assert attention_inputs["V"] == ["kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj"]


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
    assert attention_inputs["Q"] == ["q_proj"]
    assert attention_inputs["K"] == ["k_proj"]


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
    assert cls.attention_inputs["Q"] == ["q_proj"]
    assert cls.attention_inputs["K"] == ["k_proj"]
    assert cls.attention_inputs["V"] == ["v_proj"]

    tree = build_block_node(
        attr_name="attn",
        class_name="LinearAttention",
        registry=analysis.class_registry,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    segments = collect_computation_segments(tree)
    assert isinstance(segments[0], FanOutSegment)
    assert {branch.label for branch in segments[0].branches} >= {"Q", "K", "V"}

    graph = build_computation_graph(tree)
    merge_index = next(
        i for i, node in enumerate(graph.nodes) if node.block and node.block.attr_name == SYNTHETIC_ATTENTION
    )
    incoming = [src for src, dst in graph.links if dst == merge_index]
    assert len(incoming) >= 3
    o_norm_index = next(i for i, node in enumerate(graph.nodes) if node.block and node.block.attr_name == "o_norm")
    side_links = [src for src, dst in graph.links if dst == o_norm_index and (src, dst) in graph.dashed_links]
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
    gate_spec = next(spec for spec in graph.nodes if spec.port_style == "inline")
    assert gate_spec.port_label == "gate"
    input_index = next(i for i, spec in enumerate(graph.nodes) if spec.synthetic == SYNTHETIC_INPUT)
    q_head = next(i for i, spec in enumerate(graph.nodes) if spec.port_label == "Q")
    kv_head = next(i for i, spec in enumerate(graph.nodes) if spec.port_label == "K/V")
    assert (input_index, q_head) in graph.links
    assert (input_index, kv_head) in graph.links
    gate_index = next(i for i, spec in enumerate(graph.nodes) if spec.block and spec.block.attr_name == "g_proj")
    mult_index = next(i for i, spec in enumerate(graph.nodes) if spec.synthetic == SYNTHETIC_MULTIPLY)
    assert (input_index, gate_index) in graph.dashed_links
    assert (gate_index, mult_index) in graph.dashed_links


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
    assert points[0] == (gate.cx, gate.bottom - gap)


def test_top_entry_combine_connector_enters_operator_top():
    from visualizer.render import MERGE_RADIUS, _RenderAnchor, _top_entry_combine_connector_points

    gap = 0.04
    linear = _RenderAnchor(cx=5.2, top=19.4, bottom=19.0, left=4.6, right=5.8)
    target_cx = 5.2
    target_cy = 17.3
    points = _top_entry_combine_connector_points(linear, target_cx, target_cy, gap=gap)
    assert points[-1] == (target_cx, target_cy + MERGE_RADIUS + gap)
    assert points[0] == (linear.cx, linear.bottom - gap)


def test_inline_port_node_uses_two_line_height():
    from visualizer.block_tree import BlockNode
    from visualizer.computation_graph import GraphNodeSpec, _diagram_size_for_rendered_spec
    from visualizer.sizing import estimate_block_size, single_line_box_height, two_line_box_height

    gate = BlockNode(attr_name="gate", class_name="Linear", role="router", label="Router", is_basic=True)
    spec = GraphNodeSpec(
        key="gate",
        block=gate,
        label="Router",
        port_label="gate",
        port_style="inline",
    )
    _, rendered_h = _diagram_size_for_rendered_spec(spec)
    _, router_h = estimate_block_size("Router")
    _, gate_h = estimate_block_size("gate", "gate")
    assert abs(rendered_h - gate_h) < 1e-6
    assert rendered_h > single_line_box_height()
    assert abs(rendered_h - two_line_box_height()) < 1e-6
    assert rendered_h > router_h


def test_inline_dashed_port_connector_enters_gate_top():
    from visualizer.render import _RenderAnchor, _inline_dashed_port_connector_points

    gap = 0.04
    source = _RenderAnchor(cx=3.0, top=10.0, bottom=9.5, left=2.5, right=3.5)
    target = _RenderAnchor(cx=5.0, top=8.0, bottom=7.0, left=4.2, right=5.8)
    points = _inline_dashed_port_connector_points(source, target, gap=gap)
    assert points[-1] == (target.cx, target.top + gap)


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

    assert _format_input_source_sublabel("Linear in KimiSparseMoeBlock") == "← Linear\nin KimiSparseMoeBlock"
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
    assert "expert" in wrapper_module_comment(moe).lower()

    gate = BlockNode(attr_name="gate", class_name="Linear", role="router", label="Router", is_basic=True)
    assert "route" in wrapper_module_comment(gate).lower()

    lm_head = BlockNode(attr_name="lm_head", class_name="Linear", role="other", label="lm head", is_basic=True)
    assert "logits" in wrapper_module_comment(lm_head).lower()

    embed = BlockNode(attr_name="embed_tokens", class_name="Embedding", role="embedding", label="Embedding", is_basic=True)
    assert wrapper_module_comment(embed) is None

    tokenization = BlockNode(attr_name="tokenization", class_name="Tokenizer", role="embedding", label="Tokenizer", is_basic=True)
    assert wrapper_module_comment(tokenization) is None

    assert "—" in wrapper_panel_line(moe)
    assert wrapper_panel_line(embed) == wrapper_bullet(embed)


def test_parallel_gate_wrapper_comment_for_g_proj():
    from visualizer.block_tree import BlockNode, wrapper_bullet, wrapper_module_comment, wrapper_panel_line

    g_proj = BlockNode(
        attr_name="g_proj",
        class_name="OutputGate",
        role="gate",
        label="Output gate",
        is_basic=False,
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
    assert wrapper_bullet(g_proj) == "gate (g_proj)"
    assert "output gate" in wrapper_module_comment(g_proj).lower()
    assert "gate (g_proj)" in wrapper_panel_line(g_proj)
    assert "—" in wrapper_panel_line(g_proj)


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


def test_mla_gate_input_uses_residual_dashed_connector(tmp_path: Path):
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
        if title.startswith("MLA (self_attn)")
    )
    graph = build_computation_graph(mla)
    input_index = next(i for i, spec_node in enumerate(graph.nodes) if spec_node.synthetic == SYNTHETIC_INPUT)
    gate_index = next(i for i, spec_node in enumerate(graph.nodes) if spec_node.block and spec_node.block.attr_name == "g_proj")
    assert (input_index, gate_index) in graph.dashed_links

    out = render_diagram(spec, tmp_path / "kimi_detailed.svg", detailed=True)
    svg = out.read_text(encoding="utf-8")
    assert COLORS["residual"] in svg
    assert "stroke-dasharray" in svg


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
    assert not any(node.attr_name == "_forward_attn_residual" for node in spec.detailed_wrapped_modules)


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
    assert spec.detailed_wrapped_modules == []

    out = render_diagram(spec, tmp_path / "kimi_linear_wrappers.svg", detailed=True)
    svg = out.read_text(encoding="utf-8")
    assert "tokenization" not in svg.lower()
    assert "Token Embedding" in svg
    assert "LM head / output projection" in svg
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
                label="Attention (QKᵀV)",
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
        if title.startswith("MLA (self_attn)")
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
    from visualizer.sizing import estimate_block_size, estimate_block_size_for_node, two_line_box_height

    attn = BlockNode(
        attr_name="@attention",
        class_name="AttentionOp",
        role="attention",
        label="Attention (QKᵀV)",
        is_basic=True,
        details=["scaled dot-product attention"],
    )
    plain_w, plain_h = estimate_block_size("Linear")
    attn_w, attn_h = estimate_block_size_for_node(attn)
    assert attn_w > plain_w
    assert attn_h >= two_line_box_height()


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
                label="Attention (QKᵀV)",
                is_basic=True,
                details=["scaled dot-product attention"],
            ),
            leaf("o_proj"),
        ],
    )
    graph = build_computation_graph(root)
    positions, _ = layout_computation_graph(graph, cx=4.0, top_y=10.0, block_w=4.0, block_h=3.0)
    by_label = {pos.spec.label: pos for pos in positions}
    assert by_label["Attention (QKᵀV)"].width > by_label["Linear"].width
    assert by_label["Attention (QKᵀV)"].height > by_label["Linear"].height


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
    wrapped_attrs = {node.attr_name for node in spec.detailed_wrapped_modules}
    assert "embed_tokens" not in wrapped_attrs
    assert "lm_head" not in wrapped_attrs
    assert "tokenization" not in wrapped_attrs

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
    assert "Apply to Q/K" in svg
    assert "Positional (RoPE) (rotary_emb)" in svg
    assert "Block internals" in svg
    assert "<!-- hidden_states -->" not in svg.split("Block internals")[0]


def test_single_function_trees_demoted_to_wrapped_modules():
    from visualizer.block_tree import BlockNode, is_single_function_tree, partition_detail_trees

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
                details=["F.linear(...)"],
            ),
        ],
    )
    assert is_single_function_tree(embed)
    assert not is_single_function_tree(rope)
    assert not is_single_function_tree(gate)

    trees = [
        ("Token embedding (embed_tokens)", embed),
        ("Positional (RoPE) (rotary_emb)", rope),
        ("Router (gate)", gate),
    ]
    kept, wrapped = partition_detail_trees(trees, [])
    assert len(kept) == 3
    assert kept[0][0].startswith("Token embedding")
    assert kept[1][0].startswith("Positional")
    assert kept[2][0].startswith("Router")
    assert wrapped == []


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
    assert expanded == [linear]
    assert wrapper is gate

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
    assert residual[0].port_label == "residual"
    assert residual[0].source_kind == "forward_input"


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
    combine_index = next(
        index
        for index, spec in enumerate(graph.nodes)
        if spec.synthetic == SYNTHETIC_COMBINE and spec.label == "Σ"
    )
    assert graph.nodes[combine_index].sublabel == "∑ w·expert"
    route_scaling_index = next(
        index
        for index, spec in enumerate(graph.nodes)
        if spec.block and spec.block.attr_name == SYNTHETIC_ROUTER_SCALE
    )
    assert (route_scaling_index, combine_index) in graph.links
    assert (route_scaling_index, combine_index) not in graph.dashed_links
    assert (route_scaling_index, combine_index) in graph.side_entry_links
    assert not any(spec.block and spec.block.attr_name == "moe_infer" for spec in graph.nodes)
    down_proj_index = next(
        index
        for index, spec in enumerate(graph.nodes)
        if spec.block and spec.block.attr_name == "routed_expert_down_proj"
    )
    input_index = next(index for index, spec in enumerate(graph.nodes) if spec.synthetic == SYNTHETIC_INPUT)
    assert (input_index, down_proj_index) in graph.links
    assert (down_proj_index, combine_index) in graph.links
    assert (route_scaling_index, down_proj_index) not in graph.links


def test_shared_experts_graph_dashed_residual_side_link():
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
    assert (input_index, shared_index) in graph.dashed_links
    assert (shared_index, plus_index) in graph.dashed_links
    assert graph.nodes[plus_index].sublabel is None


def test_render_detailed_diagram(tmp_path: Path):
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    assert spec.detailed_block_trees or spec.detailed_wrapped_modules

    out = render_diagram(spec, tmp_path / "detailed.svg", detailed=True)
    svg = out.read_text(encoding="utf-8")
    assert "Block internals" in svg
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
    assert {variant.attention_label for variant in spec.layer_variants} == {"KDA", "MLA"}
    assert sum(variant.count for variant in spec.layer_variants) == spec.num_hidden_layers

    titles = [title for title, _ in spec.detailed_block_trees]
    assert any(title.startswith("KDA (self_attn)") for title in titles)
    assert any(title.startswith("MLA (self_attn)") for title in titles)
    assert any("KimiMLP" in title for title in titles)


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
    assert any("self_attn → KimiDeltaAttention" in line for line in spec.layer_repeat_lines)
    assert any("68 × KDA + MoE" in line for line in spec.layer_repeat_lines)

    fact_lines = _fact_lines(spec)
    assert any(line.startswith("Layer repeat: 93 × KimiDecoderLayer") for line in fact_lines)
    assert any(line.startswith("    self_attn → KimiDeltaAttention") for line in fact_lines)
    assert any(line.startswith("    68 × KDA + MoE") for line in fact_lines)
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
    assert "• 68 KDA + MoE" in label
    assert "• 24 MLA + MoE" in label
    assert "• 1 KDA + MLP" in label
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
    assert spec.stack_tail[0].label.startswith("Final")
    assert spec.stack_tail[1].label == "LM head / output projection"


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
            BlockNode(attr_name="@attention", class_name="AttentionOp", role="attention", label="Attention", is_basic=True),
            BlockNode(attr_name="o_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
        ],
        attention_inputs={"Q": ["q_proj"], "K": ["q_proj"], "V": ["q_proj"]},
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


def test_kda_attention_merge_links_are_labeled():
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
        i for i, node in enumerate(graph.nodes) if node.block and node.block.attr_name == "@attention"
    )
    labels = {graph.link_port_labels[(src, merge_index)] for src, tgt in graph.links if tgt == merge_index}
    assert {"Q", "K", "V"}.issubset(labels)


def test_kda_attention_block_shows_delta_rule_not_sdpa():
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
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    merge = next(child for child in attn.children if child.attr_name == SYNTHETIC_ATTENTION)
    assert merge.label == "Delta attention (KDA)"
    assert "QKᵀV" not in merge.label
    assert any("delta rule" in detail for detail in merge.details)
    assert any("S ←" in detail for detail in merge.details)
    assert any("Q,K,V,G,β" in detail for detail in merge.details)


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
    assert merge.label == "Attention (QKᵀV)"
    assert any("scaled dot-product" in detail for detail in merge.details)


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
    assert block_purpose(situ) == "SiLU(gate) × up branch"
    graph = build_computation_graph(mlp)
    labels = [spec.label for spec in graph.nodes]
    assert "Split gate | up" in labels
    assert "SiLU" in labels
    assert "×" in labels
    frame = next(frame for frame in graph.inline_frames if frame.frame_id == "act_fn")
    assert frame.sublabel == "SiLU(gate) × up branch"


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
    purpose = block_purpose(gate)
    assert purpose is not None
    assert "g = g_proj(hidden_states)" in purpose
    assert "reshape" in purpose
    assert "Sigmoid" in purpose or "σ" in purpose
    assert "norm(attn_out)" in purpose

    graph = build_computation_graph(attn)
    labels = [spec.label for spec in graph.nodes]
    assert "Linear" in labels
    assert "Sigmoid" in labels
    assert "RMSNorm" in labels
    assert "×" in labels
    assert any(frame.frame_id == "g_proj" for frame in graph.inline_frames)
    gate_frame = next(frame for frame in graph.inline_frames if frame.frame_id == "g_proj")
    assert gate_frame.sublabel is not None
    assert "g = g_proj(hidden_states)" in gate_frame.sublabel
    assert not any(spec.label == "Output gate" for spec in graph.nodes)

    combine_index = next(i for i, spec in enumerate(graph.nodes) if spec.label == "×")
    dashed_into_combine = [
        src
        for src, dst in graph.links
        if dst == combine_index and (src, dst) in graph.dashed_links
    ]
    assert dashed_into_combine


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
    from visualizer.computation_graph import _estimate_graph_height, build_computation_graph, layout_computation_graph
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
    graph = build_computation_graph(tree)
    est_h = _estimate_graph_height(graph)
    positions, _links = layout_computation_graph(
        graph,
        cx=5.0,
        top_y=20.0,
        block_w=6.0,
        block_h=est_h,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    try:
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


def test_box_label_size_matches_draw_box_height():
    import matplotlib.pyplot as plt
    from visualizer.sizing import two_line_box_height
    from visualizer.text_measure import box_label_size

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    try:
        _width, height = box_label_size(ax, "Linear", "Output gate — scales normalized output", fontsize=7.6)
        assert height == two_line_box_height()
    finally:
        plt.close(fig)

