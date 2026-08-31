"""Tests for detailed block tree expansion."""

from pathlib import Path

import pytest

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


def test_indexer_operation_bypasses_keep_top_entries_without_scalar_sublabels():
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import build_computation_graph

    source = """
class ColumnParallelLinear:
    pass

class Indexer:
    def __init__(self):
        self.weights_proj = ColumnParallelLinear()
        self.softmax_scale = unknown
        self.n_heads = unknown
        self.index_topk = unknown

    def forward(self, x, score, start_pos, end_pos, ratio, offset):
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        score = (score * weights).sum(dim=2)
        topk_idxs = score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            topk_idxs = topk_idxs + offset
        else:
            topk_idxs += offset
        return topk_idxs
"""
    analysis = analyze_source(source, all_tensor_ops=True)
    tree = build_block_node(
        attr_name="indexer",
        class_name="Indexer",
        registry=analysis.class_registry,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    graph = build_computation_graph(tree, basic_ops=BasicOpFilter.for_detailed())
    assert all(node.synthetic != "@combine" for node in graph.nodes)
    assert all(
        node.sublabel is None
        for node in graph.nodes
        if node.label in {"Add", "Multiply", "Power", "TopK"}
    )

    power_index = next(index for index, node in enumerate(graph.nodes) if node.label == "Power")
    assert graph.nodes[power_index].sublabel is None
    # A step reading only configuration still needs an incoming connector, or it
    # is drawn as a tile nothing feeds.
    assert any(target == power_index for _source, target in graph.links)

    projection = next(
        index for index, node in enumerate(graph.nodes) if node.label == "ColumnParallelLinear"
    )
    scaled_weights = next(
        index
        for index, node in enumerate(graph.nodes)
        if node.label == "Multiply"
        and node.block
        and "weights_proj" in node.block.operation_predecessors
    )
    assert (projection, scaled_weights) in graph.links
    bypass_inputs = [
        source_index
        for source_index, target_index in graph.links
        if target_index == scaled_weights
    ]
    assert bypass_inputs
    assert any(source_index != projection for source_index in bypass_inputs)

    topk = next(index for index, node in enumerate(graph.nodes) if node.label == "TopK")
    assert any(
        target == topk and graph.nodes[source_index].label == "Sum"
        for source_index, target in graph.links
    )

    conditional_adds = [
        index
        for index, node in enumerate(graph.nodes)
        if node.label == "Add"
        and node.block
        and any(detail.startswith("condition: ") for detail in node.block.details)
    ]
    assert len(conditional_adds) == 2
    assert (conditional_adds[0], conditional_adds[1]) in graph.links
    assert (topk, conditional_adds[1]) in graph.links


def test_discarded_module_call_on_forward_input_is_a_packed_side_branch():
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        build_computation_graph,
        layout_computation_graph,
        pack_input_fed_inline_frame_branches,
    )
    from visualizer.render import _inline_frame_draw_bounds
    from visualizer.sizing import min_horizontal_block_gap

    source = """
class Linear:
    pass

class Compressor:
    def __init__(self):
        self.in_proj = Linear()
        self.out_proj = Linear()

    def forward(self, x):
        x = self.in_proj(x)
        return self.out_proj(x)

class Indexer:
    def __init__(self):
        self.query = Linear()
        self.compressor = Compressor()
        self.weights = Linear()

    def forward(self, x, qr):
        q = self.query(qr)
        self.compressor(x)
        weights = self.weights(x)
        score = q * weights
        score = score * weights
        return score
"""
    analysis = analyze_source(source, all_tensor_ops=True)
    structure = analysis.class_registry["Indexer"]
    assert [spec.arg_name for spec in structure.side_inputs["compressor"]] == ["x"]
    assert "weights" not in structure.side_inputs

    tree = build_block_node(
        attr_name="indexer",
        class_name="Indexer",
        registry=analysis.class_registry,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    graph = build_computation_graph(tree, basic_ops=BasicOpFilter.for_detailed())
    positions, _ = layout_computation_graph(
        graph,
        cx=3.0,
        top_y=10.0,
        block_w=6.0,
        block_h=8.0,
    )

    frame = next(frame for frame in graph.inline_frames if frame.label == "Compressor")
    assert frame.frame_id in graph.side_effect_frame_ids
    bounds = _inline_frame_draw_bounds(frame, positions, graph)
    members = set(frame.node_indices)
    main_index = next(
        index
        for index, pos in enumerate(positions)
        if index not in members and pos.spec.synthetic is None
    )
    main = positions[main_index]
    main.top_y = bounds.top
    main.cx = bounds.right + 2.0 + main.width / 2
    pack_input_fed_inline_frame_branches(positions, graph)

    packed_bounds = _inline_frame_draw_bounds(frame, positions, graph)
    assert main.cx - main.width / 2 - packed_bounds.right == pytest.approx(
        min_horizontal_block_gap()
    )


def _routing_anchor(cx: float, top: float, *, width: float = 1.0, height: float = 0.32):
    from visualizer.render import _RenderAnchor

    return _RenderAnchor(
        cx=cx,
        top=top,
        bottom=top - height,
        left=cx - width / 2,
        right=cx + width / 2,
    )


class _RoutingGraph:
    """Minimal stand-in for a computation graph during connector routing."""

    def __init__(self, count: int):
        from visualizer.computation_graph import GraphNodeSpec

        self.nodes = [GraphNodeSpec(key=f"n{index}") for index in range(count)]


def test_input_under_foreign_tile_still_uses_top_entry():
    from visualizer.render import _snap_connector_path_endpoints

    source = _routing_anchor(1.0, 2.0)
    target = _routing_anchor(3.0, 0.0)
    graph = _RoutingGraph(3)
    points = _snap_connector_path_endpoints(
        [(source.right, 1.8), (target.left, -0.2)],
        source=source,
        target=target,
        link_key=(0, 1),
        graph=graph,
    )
    assert points[0] == (source.cx, source.bottom)
    assert points[-1][1] == target.top


def test_fan_in_spread_port_remains_on_target_top():
    from visualizer.render import _snap_connector_path_endpoints

    target = _routing_anchor(3.0, 0.0)
    source = _routing_anchor(6.0, 2.0)
    graph = _RoutingGraph(2)
    points = _snap_connector_path_endpoints(
        [(source.cx, source.bottom), (target.cx, target.top)],
        source=source,
        target=target,
        link_key=(0, 1),
        graph=graph,
        merge_entry_x={(0, 1): target.cx - 0.2},
    )
    assert points[-1] == (target.cx - 0.2, target.top)


FIXTURES = Path(__file__).parent / "fixtures"
KIMI_ROUTER_CONFIG = {
    "hidden_size": 7168,
    "num_experts": 896,
    "num_experts_per_token": 16,
    "num_expert_group": 1,
    "topk_group": 1,
    "moe_router_activation_func": "sigmoid",
    "moe_renormalize": True,
    "routed_scaling_factor": 1.0,
}


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

    analysis = analyze_source(
        code_path.read_text(),
        filename="modeling_kimi_linear.py",
        config=KIMI_ROUTER_CONFIG,
    )
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


def test_detail_section_title_matches_overview_attention_tile():
    from visualizer.block_tree import BlockNode
    from visualizer.blocks import LayerVariant
    from visualizer.extract import ArchitectureSpec
    from visualizer.render import _attention_label, _detail_section_title

    attn_tree = BlockNode(
        attr_name="attn",
        class_name="Attention",
        role="attention",
        label="Attn",
    )
    ffn_tree = BlockNode(attr_name="ffn", class_name="MoE", role="ffn", label="MoE")

    single = ArchitectureSpec(name="DeepSeek-like", model_type="deepseek_v4")
    single.attention_type = "MLA"
    single.attention_notes = ["Multi-head Latent Attention (compressed KV)"]
    # The overview tile names this block MLA, so its expanded section must agree.
    assert _attention_label(single).startswith("MLA")
    assert _detail_section_title(single, "Attn", attn_tree) == "MLA"
    assert _detail_section_title(single, "MoE", ffn_tree) == "MoE"

    hybrid = ArchitectureSpec(name="Kimi-like", model_type="kimi_linear")
    hybrid.attention_type = "Hybrid"
    hybrid.layer_variants = [
        LayerVariant(label="a", count=68, attention_label="KimiDelta Attn"),
        LayerVariant(label="b", count=24, attention_label="KimiMLA Attn"),
    ]
    for variant_title in ("KimiDelta Attn", "KimiMLA Attn"):
        assert _detail_section_title(hybrid, variant_title, attn_tree) == variant_title

    hybrid.layer_variants[0].ffn_class = "MiniMaxM3VLSparseMoeBlock"
    moe_tree = BlockNode(
        attr_name="mlp",
        class_name="MiniMaxM3VLSparseMoeBlock",
        role="ffn",
        label="FFN",
    )
    assert _detail_section_title(hybrid, "FFN", moe_tree) == "MiniMaxM3VLSparseMoeBlock"


def test_moe_layer_frequency_list_is_a_per_layer_mask():
    from visualizer.extract import _config_moe_layer

    config = {
        "num_local_experts": 128,
        "moe_layer_freq": [0, 0, 0, 1, 1],
    }
    assert [_config_moe_layer(index, config) for index in range(5)] == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_conditional_submodule_binds_concrete_class_and_expands():
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node, is_inline_expandable_module

    source = """
class Linear:
    pass

class LightningIndexer:
    def __init__(self):
        self.q_proj = Linear()
        self.k_proj = Linear()
        self.top_k = 4

    def forward(self, hidden_states):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        scores = torch.matmul(q, k)
        block_scores = scores.amax(dim=-1)
        return block_scores.topk(self.top_k, dim=-1)

class ActualAttention:
    def __init__(self, config):
        self.indexer = LightningIndexer() if config.use_sparse else None
        self.o_proj = Linear()

    def forward(self, hidden_states):
        if self.indexer is not None:
            indices = self.indexer(hidden_states)
        return self.o_proj(hidden_states)
"""
    analysis = analyze_source(source, filename="modeling_conditional.py")
    attention = analysis.class_registry["ActualAttention"]

    assert attention.init_assignments["indexer"] == "LightningIndexer"
    assert attention.init_assignment_options["indexer"] == ["LightningIndexer"]

    tree = build_block_node(
        attr_name="self_attn",
        class_name="ActualAttention",
        registry=analysis.class_registry,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    indexer = next(child for child in tree.children if child.attr_name == "indexer")
    assert indexer.class_name == "LightningIndexer"
    assert [child.attr_name for child in indexer.children[:2]] == ["q_proj", "k_proj"]
    assert [child.label for child in indexer.children[2:]] == ["MatMul", "Block max", "TopK"]
    assert is_inline_expandable_module(indexer), "a genuinely straight indexer stays inline"


def test_component_sublabel_omits_method_wrapper_metadata():
    from visualizer.blocks import BlockComponent
    from visualizer.render import _component_sublabel

    component = BlockComponent(
        attr_name="mlp",
        class_name="MiniMaxM3VLSparseMoeBlock",
        role="ffn",
        label="SWIGLUOAI",
        details=["method `mlp()`"],
    )
    assert _component_sublabel(component) is None


def _straight_line_tree(class_name: str, step_names: list[str]):
    from visualizer.block_tree import BlockNode

    return BlockNode(
        attr_name="mlp",
        class_name=class_name,
        role="ffn",
        label=class_name,
        children=[
            BlockNode(attr_name=name, class_name="Linear", role="linear", label="Linear")
            for name in step_names
        ],
    )


def test_consecutive_bypasses_stay_inline_but_nested_ones_do_not():
    """One side column serves bypasses in turn; nested ones each need their own."""
    from visualizer.block_tree import BlockNode, _has_overlapping_bypass_spans

    def tree(spans: dict[str, list[str]]):
        return BlockNode(
            attr_name="block",
            class_name="Block",
            role="ffn",
            label="Block",
            children=[
                BlockNode(
                    attr_name=name,
                    class_name="Linear",
                    role="linear",
                    label="Linear",
                    operation_predecessors=list(preds),
                )
                for name, preds in spans.items()
            ],
        )

    consecutive = tree(
        {
            "a": [],
            "b": ["a"],
            "c": ["b"],
            "d": ["c", "a"],
            "e": ["d"],
            "f": ["e"],
            "g": ["f", "d"],
        }
    )
    assert not _has_overlapping_bypass_spans(consecutive)

    nested = tree(
        {
            "a": [],
            "b": ["a"],
            "c": ["b"],
            "d": ["c", "b"],
            "e": ["d", "a"],
        }
    )
    assert _has_overlapping_bypass_spans(nested)


def test_straight_line_component_expands_in_place_instead_of_its_own_section():
    """Sequential components stay inline even when operations have bypass operands."""
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import BlockNode
    from visualizer.extract import ArchitectureSpec
    from visualizer.render import _detail_sections_to_render

    dense = _straight_line_tree("DenseMLP", ["gate_up_proj", "down_proj"])
    nested = BlockNode(
        attr_name="mlp",
        class_name="NestedBypassMLP",
        role="ffn",
        label="NestedBypassMLP",
        children=[
            BlockNode(
                attr_name=name,
                class_name="Linear",
                role="linear",
                label="Linear",
                operation_predecessors=list(preds),
            )
            for name, preds in (
                ("a", []),
                ("b", ["a"]),
                ("c", ["b"]),
                ("d", ["c", "b"]),
                ("e", ["d", "a"]),
            )
        ],
    )
    spec = ArchitectureSpec(name="T", model_type="t", architectures=["T"])
    spec.basic_ops = BasicOpFilter.for_detailed()
    spec.export_block_trees = [("DenseMLP", dense), ("NestedBypassMLP", nested)]

    titles = [title for title, _tree, _sub in _detail_sections_to_render(spec)]
    assert "DenseMLP" not in titles, "straight-line components expand inline"
    assert "NestedBypassMLP" not in titles, "operand bypasses route inside the inline frame"


def test_ffn_the_spine_names_beside_another_variant_gets_its_own_section():
    """A spine tile naming one FFN per layer variant expands each of them."""
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.blocks import LayerVariant
    from visualizer.extract import ArchitectureSpec
    from visualizer.render import _detail_sections_to_render, _ffn_label

    dense = _straight_line_tree("DenseMLP", ["gate_up_proj", "down_proj"])
    moe = _straight_line_tree("SparseMoeBlock", ["gate", "experts"])
    spec = ArchitectureSpec(name="T", model_type="t", architectures=["T"])
    spec.basic_ops = BasicOpFilter.for_detailed()
    spec.export_block_trees = [("DenseMLP", dense), ("FFN", moe)]
    spec.layer_variants = [
        LayerVariant(
            label="dense",
            count=3,
            attention_label="Attn",
            ffn_label="DenseMLP",
            ffn_class="DenseMLP",
            ffn_attr="mlp",
        ),
        LayerVariant(
            label="sparse",
            count=57,
            attention_label="Attn",
            ffn_label="MoE block",
            ffn_class="SparseMoeBlock",
            ffn_attr="mlp",
        ),
    ]

    label, _sublabel = _ffn_label(spec)
    assert label == "DenseMLP / SparseMoeBlock", "the spine tile names both classes"
    titles = [title for title, _tree, _sub in _detail_sections_to_render(spec)]
    assert "DenseMLP" in titles, f"a named variant has to expand somewhere: {titles}"


def test_layer_branch_condition_reads_the_config_it_names():
    """A branch is decided by its config values, not by matching a known phrasing."""
    from visualizer.layer_repeat_simplify import layer_condition_matches

    condition = (
        "if layer_idx not in config.mlp_only_layers and "
        "(config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0)"
    )
    every_layer = {"mlp_only_layers": [], "num_experts": 128, "decoder_sparse_step": 1}
    assert [layer_condition_matches(idx, condition, every_layer) for idx in range(3)] == [
        True,
        True,
        True,
    ]

    every_other = dict(every_layer, mlp_only_layers=[0], decoder_sparse_step=2)
    assert [layer_condition_matches(idx, condition, every_other) for idx in range(4)] == [
        False,
        True,
        False,
        True,
    ]


_CONDITIONAL_FFN_SOURCE = """
class Linear:
    pass

class TinyAttention:
    def __init__(self, config):
        self.q_proj = Linear()

    def forward(self, hidden_states):
        return self.q_proj(hidden_states)

class TinyMLP:
    def __init__(self, config):
        self.gate_proj = Linear()

    def forward(self, hidden_states):
        return self.gate_proj(hidden_states)

class TinySparseMoeBlock:
    def __init__(self, config):
        self.gate = Linear()

    def forward(self, hidden_states):
        return self.gate(hidden_states)

class TinyDecoderLayer:
    def __init__(self, config, layer_idx):
        self.self_attn = TinyAttention(config)
        if layer_idx not in config.mlp_only_layers and config.num_experts > 0:
            self.mlp = TinySparseMoeBlock(config)
        else:
            self.mlp = TinyMLP(config)

    def forward(self, hidden_states):
        return self.mlp(self.self_attn(hidden_states))
"""


def _spec_with_conditional_ffn(config: dict):
    from visualizer.ast_analyze import analyze_source
    from visualizer.blocks import BlockComponent
    from visualizer.extract import ArchitectureSpec, _infer_layer_variants

    analysis = analyze_source(_CONDITIONAL_FFN_SOURCE, filename="modeling_tiny.py")
    spec = ArchitectureSpec(name="Tiny", model_type="tiny", architectures=["Tiny"])
    spec.num_hidden_layers = 4
    spec.raw_config = config
    spec.block_components = [
        BlockComponent(
            attr_name="self_attn",
            class_name="TinyAttention",
            role="attention",
            label="Tiny Attn",
            forward_order=0,
        ),
        # The AST keeps the branch it saw last, which need not be the one this config builds.
        BlockComponent(
            attr_name="mlp",
            class_name="TinyMLP",
            role="ffn",
            label="TinyMLP",
            forward_order=1,
        ),
    ]
    _infer_layer_variants(
        config,
        spec,
        class_registry=analysis.class_registry,
        decoder_class="TinyDecoderLayer",
    )
    return spec


def test_spine_ffn_tile_names_the_branch_the_config_builds():
    """With every layer sparse, the overview tile and its section share one name."""
    from visualizer.ast_analyze import decoder_type_for_components
    from visualizer.block_tree import BlockNode
    from visualizer.render import _detail_section_title, _ffn_label

    spec = _spec_with_conditional_ffn({"mlp_only_layers": [], "num_experts": 128})

    mlp = next(comp for comp in spec.block_components if comp.attr_name == "mlp")
    assert mlp.class_name == "TinySparseMoeBlock"
    assert mlp.role == "moe", "a routed block is not a dense FFN just because it is `self.mlp`"
    assert decoder_type_for_components(spec.block_components) == "Sparse MoE"

    tree = BlockNode(
        attr_name="mlp",
        class_name="TinySparseMoeBlock",
        role="moe",
        label="TinySparseMoeBlock",
    )
    label, _sublabel = _ffn_label(spec)
    assert label == "TinySparseMoeBlock"
    assert _detail_section_title(spec, "TinySparseMoeBlock", tree) == label


def test_spine_ffn_tile_stays_dense_when_the_config_selects_the_dense_branch():
    """A config that routes no layer through the experts keeps the dense FFN tile."""
    from visualizer.render import _ffn_label

    spec = _spec_with_conditional_ffn(
        {"mlp_only_layers": [0, 1, 2, 3], "num_experts": 128}
    )

    mlp = next(comp for comp in spec.block_components if comp.attr_name == "mlp")
    assert (mlp.class_name, mlp.role) == ("TinyMLP", "ffn")
    assert _ffn_label(spec)[0] == spec.ffn_type


def test_section_of_a_module_bypasses_the_step_its_math_skips():
    """A top-level module keeps the dataflow its operations name, bypass and all."""
    from visualizer.block_tree import BlockNode
    from visualizer.computation_graph import build_computation_graph, layout_computation_graph

    def op(attr_name: str, label: str, preds: list[str]) -> BlockNode:
        return BlockNode(
            attr_name=attr_name,
            class_name=label,
            role="operation",
            label=label,
            operation_predecessors=list(preds),
        )

    root = BlockNode(
        attr_name="mlp",
        class_name="DenseMLP",
        role="ffn",
        label="DenseMLP",
        children=[
            BlockNode(attr_name="gate_up_proj", class_name="Linear", role="linear", label="Linear"),
            op("glu", "Multiply", []),
            op("add", "Add", []),
            # (up + 1) * glu reads the multiply two steps back, skipping the add.
            op("scale", "Multiply", ["add", "glu"]),
            BlockNode(attr_name="down_proj", class_name="Linear", role="linear", label="Linear"),
        ],
    )

    graph = build_computation_graph(root)
    labels = [node.label for node in graph.nodes]
    glu = labels.index("Multiply")
    scale = len(labels) - 1 - labels[::-1].index("Multiply")
    add = labels.index("Add")
    assert (glu, scale) in graph.links, f"the bypass is missing from {graph.links}"

    positions, _edges = layout_computation_graph(graph, cx=2.0, top_y=0.0, block_w=4.0)
    assert positions[add].cx != pytest.approx(positions[scale].cx), (
        "the step a bypass skips has to leave the column the bypass runs down"
    )


def test_frame_the_chain_flows_into_takes_the_feeding_column():
    """A frame fed by one step sits in that step's column, so the spine stays straight."""
    from visualizer.computation_graph import (
        ComputationGraph,
        GraphNodeSpec,
        InlineFrameSpec,
        LayoutPosition,
        _center_align_vertical_chains,
    )

    specs = [
        GraphNodeSpec(key="router", label="Router"),
        GraphNodeSpec(key="head", label="Linear"),
        GraphNodeSpec(key="tail", label="Linear"),
    ]
    positions = [
        LayoutPosition(spec=specs[0], cx=1.54, top_y=-17.4, width=0.66, height=0.32),
        LayoutPosition(spec=specs[1], cx=4.25, top_y=-18.2, width=0.63, height=0.32),
        LayoutPosition(spec=specs[2], cx=4.25, top_y=-18.8, width=0.63, height=0.32),
    ]
    graph = ComputationGraph(
        nodes=specs,
        links=[(0, 1), (1, 2)],
        inline_frames=[
            InlineFrameSpec(frame_id="experts", label="Experts", node_indices=[1, 2]),
        ],
    )

    _center_align_vertical_chains(positions, graph)

    assert positions[1].cx == pytest.approx(positions[0].cx), "the frame head left its feeder's column"
    assert positions[2].cx == pytest.approx(positions[0].cx)


def test_input_sits_over_the_single_step_it_feeds():
    """One consumer means the input tile shares its column instead of centering."""
    from visualizer.computation_graph import (
        SYNTHETIC_INPUT,
        ComputationGraph,
        GraphNodeSpec,
        LayoutPosition,
        _center_align_vertical_chains,
    )

    def graph_with(targets: list[int]) -> tuple[ComputationGraph, list[LayoutPosition]]:
        specs = [
            GraphNodeSpec(key="@input", label="hidden_states", synthetic=SYNTHETIC_INPUT),
            GraphNodeSpec(key="left", label="Linear"),
            GraphNodeSpec(key="right", label="Linear"),
        ]
        positions = [
            LayoutPosition(spec=specs[0], cx=2.61, top_y=-13.2, width=1.05, height=0.46),
            LayoutPosition(spec=specs[1], cx=1.54, top_y=-14.1, width=0.63, height=0.32),
            LayoutPosition(spec=specs[2], cx=3.90, top_y=-14.1, width=0.63, height=0.32),
        ]
        return ComputationGraph(nodes=specs, links=[(0, target) for target in targets]), positions

    single, positions = graph_with([1])
    _center_align_vertical_chains(positions, single)
    assert positions[0].cx == pytest.approx(1.54), "a lone consumer leaves nothing to center over"

    fanout, positions = graph_with([1, 2])
    _center_align_vertical_chains(positions, fanout)
    assert positions[0].cx == pytest.approx(2.61), "a fan-out keeps the input centered over its branches"


def test_stacked_inline_frames_share_one_spine_column():
    """Frames that follow one another down the spine must not be split into columns."""
    from visualizer.computation_graph import _group_frame_columns_by_vertical_band
    from visualizer.computation_graph import GraphNodeSpec, LayoutPosition

    def pos(cx: float, top_y: float) -> LayoutPosition:
        return LayoutPosition(
            spec=GraphNodeSpec(key="k", label="Linear"),
            cx=cx,
            top_y=top_y,
            width=1.0,
            height=0.4,
        )

    positions = [pos(1.0, 10.0), pos(1.0, 9.0), pos(1.0, 6.0), pos(1.0, 5.0)]
    stacked = _group_frame_columns_by_vertical_band(positions, [[0, 1], [2, 3]])
    assert len(stacked) == 2, "stacked frames keep their own band, so neither is shifted"

    positions = [pos(1.0, 10.0), pos(1.0, 9.0), pos(3.0, 10.0), pos(3.0, 9.0)]
    side_by_side = _group_frame_columns_by_vertical_band(positions, [[0, 1], [2, 3]])
    assert len(side_by_side) == 1, "frames at the same height compete for one band"


def test_multi_op_helper_expands_but_norm_keeps_its_semantic_type():
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node, expand_block_tree_inplace

    source = """
class MiniMaxM3VLExperts:
    def forward(self, x):
        return self._apply_gate(x)

    def _apply_gate(self, x):
        gate = x * 2
        activated = torch.sigmoid(gate)
        return activated * x

class MiniMaxM3VLRMSNorm:
    def forward(self, x):
        return self._norm(x)

    def _norm(self, x):
        variance = x * x
        return variance + x
"""
    analysis = analyze_source(source, all_tensor_ops=True)
    basic_ops = BasicOpFilter.for_detailed()

    experts = build_block_node(
        attr_name="experts",
        class_name="MiniMaxM3VLExperts",
        registry=analysis.class_registry,
        basic_ops=basic_ops,
    )
    apply_gate = experts.children[0]
    assert apply_gate.label == "apply gate"
    assert [child.label for child in apply_gate.children] == [
        "Multiply",
        "Sigmoid",
        "Multiply",
    ]

    norm = build_block_node(
        attr_name="q_norm",
        class_name="MiniMaxM3VLRMSNorm",
        registry=analysis.class_registry,
        basic_ops=basic_ops,
    )
    prepared_norm = expand_block_tree_inplace(norm, basic_ops=basic_ops)
    assert prepared_norm.class_name == "MiniMaxM3VLRMSNorm"
    assert prepared_norm.label == "RMSNorm"


def test_repeat_summary_uses_the_diagram_module_names():
    from visualizer.blocks import LayerVariant
    from visualizer.extract import ArchitectureSpec
    from visualizer.render import _repeat_block_label

    spec = ArchitectureSpec(name="M3", model_type="m3", architectures=["M3"])
    spec.num_hidden_layers = 60
    spec.layer_variants = [
        LayerVariant(
            label="MiniMaxM3VL Attn + MoE block",
            count=57,
            attention_label="MiniMaxM3VL Attn",
            attention_class="MiniMaxM3VLAttention",
            ffn_label="MoE block",
            ffn_class="MiniMaxM3VLSparseMoeBlock",
            ffn_attr="mlp",
        ),
        LayerVariant(
            label="MiniMaxM3VL Attn + MiniMaxM3VLDenseMLP",
            count=3,
            attention_label="MiniMaxM3VL Attn",
            attention_class="MiniMaxM3VLAttention",
            ffn_label="MiniMaxM3VLDenseMLP",
            ffn_class="MiniMaxM3VLDenseMLP",
            ffn_attr="mlp",
        ),
    ]

    label = _repeat_block_label(spec)
    assert "57 MiniMaxM3VLAttention + MiniMaxM3VLSparseMoeBlock" in label
    assert "3 MiniMaxM3VLAttention + MiniMaxM3VLDenseMLP" in label
    assert "MoE block" not in label


def test_section_content_stays_anchored_when_ports_share_a_consumer():
    """Docking two ports onto one row must not leave a vacant row under the title."""
    import matplotlib.pyplot as plt

    from visualizer.computation_graph import (
        SYNTHETIC_TENSOR,
        ComputationGraph,
        GraphNodeSpec,
        LayoutPosition,
    )
    from visualizer.render import COLORS, _detail_content_extents
    from visualizer.render_validate import finalize_detail_layout

    graph = ComputationGraph()
    graph.nodes.append(GraphNodeSpec(key="kv", label="kv", synthetic=SYNTHETIC_TENSOR))
    graph.nodes.append(GraphNodeSpec(key="topk_idxs", label="topk_idxs", synthetic=SYNTHETIC_TENSOR))
    graph.nodes.append(GraphNodeSpec(key="kernel", label="Sparse attn kernel"))
    graph.nodes.append(GraphNodeSpec(key="contiguous", label="Contiguous"))
    graph.links.extend([(0, 2), (1, 2), (2, 3)])

    top_y = 10.0
    cx = 1.3
    # The layered pass gives each port its own row; docking later collapses them onto one.
    positions = [
        LayoutPosition(spec=graph.nodes[0], cx=cx, top_y=top_y, width=0.85, height=0.51),
        LayoutPosition(spec=graph.nodes[1], cx=cx + 0.9, top_y=top_y - 0.69, width=0.82, height=0.51),
        LayoutPosition(spec=graph.nodes[2], cx=cx, top_y=top_y - 1.39, width=1.39, height=0.32),
        LayoutPosition(spec=graph.nodes[3], cx=cx, top_y=top_y - 1.90, width=0.93, height=0.32),
    ]

    fig, ax = plt.subplots(figsize=(6, 8))
    try:
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=cx,
            top_y=top_y,
            detail_fill=COLORS["detail_fill"],
        )
    finally:
        plt.close(fig)

    _left, _right, _bottom, max_top = _detail_content_extents(positions)
    assert max_top == pytest.approx(top_y, abs=1e-6)


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
    o_norm_inputs = [src for src, dst in graph.links if dst == o_norm_index]
    assert o_norm_inputs


def test_kimi_moe_gate_parses_functional_linear():
    from visualizer.ast_analyze import analyze_source, is_forward_operation

    config = {
        "hidden_size": 7168,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_expert_group": 1,
        "topk_group": 1,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "routed_scaling_factor": 1.0,
    }
    analysis = analyze_source(
        (FIXTURES / "kimi_moe_gate.py").read_text(),
        filename="kimi_moe_gate.py",
        config=config,
    )
    gate = analysis.class_registry["KimiMoEGate"]
    operations = list(gate.forward_operations.values())
    assert [op.label for op in operations] == [
        "Linear",
        "Sigmoid",
        "Add",
        "TopK",
        "Gather",
        "Sum",
        "Add",
        "Divide",
        "Multiply",
    ]
    assert all(is_forward_operation(op.attr_name) for op in operations)
    assert all("Softmax" != op.label for op in operations)
    by_label = {op.label: op for op in operations if op.label not in {"Add"}}
    adds = [op for op in operations if op.label == "Add"]
    assert by_label["TopK"].predecessors == (adds[0].attr_name,)
    assert by_label["Gather"].predecessors == (
        by_label["Sigmoid"].attr_name,
        by_label["TopK"].attr_name,
    )
    assert by_label["Divide"].predecessors == (
        by_label["Gather"].attr_name,
        adds[1].attr_name,
    )


def test_kimi_moe_gate_tensor_op_granularity_and_unresolved_branch():
    source = (FIXTURES / "kimi_moe_gate.py").read_text()
    default_gate = analyze_source(
        source,
        config=KIMI_ROUTER_CONFIG,
    ).class_registry["KimiMoEGate"]
    all_ops_gate = analyze_source(
        source,
        config=KIMI_ROUTER_CONFIG,
        all_tensor_ops=True,
    ).class_registry["KimiMoEGate"]
    default_labels = [op.label for op in default_gate.forward_operations.values()]
    all_labels = [op.label for op in all_ops_gate.forward_operations.values()]
    assert not {"View", "Cast", "Unsqueeze"} & set(default_labels)
    assert {"View", "Cast", "Unsqueeze"} <= set(all_labels)

    unresolved = dict(KIMI_ROUTER_CONFIG)
    unresolved.pop("moe_router_activation_func")
    unresolved_gate = analyze_source(
        source,
        config=unresolved,
    ).class_registry["KimiMoEGate"]
    conditional = [
        op
        for op in unresolved_gate.forward_operations.values()
        if op.label in {"Sigmoid", "Softmax"}
    ]
    assert [op.label for op in conditional] == ["Sigmoid", "Softmax"]
    assert all(any(detail.startswith("condition:") for detail in op.details) for op in conditional)


def test_functional_ops_render_as_basic_names_without_parentheses():
    from visualizer.ast_analyze import (
        functional_display_label,
        functional_synthetic_attr,
        is_forward_operation,
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
    linear = next(child for child in tree.children if is_forward_operation(child.attr_name))
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
    from visualizer.computation_graph import SYNTHETIC_INPUT, build_computation_graph

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
    assert "Multiply" in labels
    assert len(graph.links) >= 10
    assert any(spec.synthetic == SYNTHETIC_INPUT for spec in graph.nodes)
    assert not any(spec.synthetic == "@combine" for spec in graph.nodes)
    gate_spec = next(spec for spec in graph.nodes if spec.block and spec.block.attr_name == "g_proj")
    assert gate_spec.port_label == "Linear"
    input_index = next(i for i, spec in enumerate(graph.nodes) if spec.synthetic == SYNTHETIC_INPUT)
    q_head = next(i for i, spec in enumerate(graph.nodes) if spec.block and spec.block.attr_name == "q_a_proj")
    kv_head = next(i for i, spec in enumerate(graph.nodes) if spec.block and spec.block.attr_name == "kv_a_proj_with_mqa")
    assert (input_index, q_head) in graph.links
    assert (input_index, kv_head) in graph.links
    gate_index = next(i for i, spec in enumerate(graph.nodes) if spec.block and spec.block.attr_name == "g_proj")
    mult_index = next(i for i, spec in enumerate(graph.nodes) if spec.label == "Multiply")
    assert (input_index, gate_index) in graph.links
    assert (input_index, gate_index) not in graph.dashed_links
    assert (gate_index, mult_index) in graph.links
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


def test_combine_connector_is_normalized_to_source_bottom_and_target_top():
    from visualizer.render import _RenderAnchor, _snap_connector_path_endpoints

    gate = _RenderAnchor(cx=7.8, top=19.4, bottom=19.0, left=7.2, right=8.4)
    target = _RenderAnchor(cx=5.2, top=17.5, bottom=17.1, left=4.7, right=5.7)
    graph = _RoutingGraph(2)
    points = _snap_connector_path_endpoints(
        [(gate.right, 19.2), (target.right, 17.3)],
        source=gate,
        target=target,
        link_key=(0, 1),
        graph=graph,
    )
    assert points[0] == (gate.cx, gate.bottom)
    assert points[-1][1] == target.top


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

    experts = BlockNode(
        attr_name="experts",
        class_name="SparseExperts",
        role="other",
        label="Experts",
        is_basic=False,
        children=[
            BlockNode(attr_name="up", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="down", class_name="Linear", role="other", label="Linear", is_basic=True),
        ],
    )
    assert inline_block_frame_label(experts) == "SparseExperts", (
        "a frame holding a whole computation is named by the class implementing it"
    )


def test_nested_inline_frame_borders_nest_without_overlapping():
    """A frame drawn inside another keeps a full pad of clearance from its border."""
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
    from visualizer.render import INLINE_FRAME_PAD, _inline_frame_draw_bounds

    source = """
class Linear:
    pass

class Experts:
    def __init__(self):
        self.up = Linear()
        self.down = Linear()

    def forward(self, x):
        h = self.up(x)
        g = self._apply_gate(h)
        return self.down(g)

    def _apply_gate(self, x):
        gate = x * 2
        activated = torch.sigmoid(gate)
        return activated * x

class MoE:
    def __init__(self):
        self.router = Linear()
        self.experts = Experts()

    def forward(self, x):
        scores = self.router(x)
        return self.experts(scores)
"""
    analysis = analyze_source(source, all_tensor_ops=True)
    basic_ops = BasicOpFilter.for_detailed()
    tree = build_block_node(
        attr_name="mlp",
        class_name="MoE",
        registry=analysis.class_registry,
        basic_ops=basic_ops,
    )
    graph = build_computation_graph(tree, basic_ops=basic_ops)
    outer = next(frame for frame in graph.inline_frames if frame.frame_id == "experts")
    inner = next(frame for frame in graph.inline_frames if frame.frame_id == "_apply_gate")
    assert set(inner.node_indices) < set(outer.node_indices)

    fig, ax = plt.subplots(figsize=(10, 10))
    try:
        measure_graph_node_sizes(ax, graph)
        positions, _links = layout_computation_graph(
            graph,
            cx=5.0,
            top_y=10.0,
            block_w=8.0,
            block_h=_estimate_graph_height(graph),
        )
        outer_bounds = _inline_frame_draw_bounds(outer, positions, graph)
        inner_bounds = _inline_frame_draw_bounds(inner, positions, graph)
        clearances = (
            inner_bounds.left - outer_bounds.left,
            outer_bounds.right - inner_bounds.right,
            outer_bounds.top - inner_bounds.top,
            inner_bounds.bottom - outer_bounds.bottom,
        )
        assert min(clearances) >= INLINE_FRAME_PAD - 1e-6, (
            f"nested frame border sits on its parent's: clearances {clearances}"
        )
        tiles_top = max(positions[index].top_y for index in outer.node_indices)
        assert outer_bounds.top <= tiles_top + 2 * INLINE_FRAME_PAD + 1e-6, (
            "holding a nested frame must not inflate the parent past its own tiles"
        )
    finally:
        plt.close(fig)


def _captioned_frame_fixture():
    """One captioned inline frame of two stacked tiles, plus a feeder above it."""
    from visualizer.computation_graph import (
        ComputationGraph,
        GraphNodeSpec,
        InlineFrameSpec,
        LayoutPosition,
    )

    specs = [
        GraphNodeSpec(key="router", label="Router"),
        GraphNodeSpec(key="head", label="Linear"),
        GraphNodeSpec(key="tail", label="Linear"),
    ]
    positions = [
        LayoutPosition(spec=specs[0], cx=0.6, top_y=6.4, width=0.8, height=0.4),
        LayoutPosition(spec=specs[1], cx=3.0, top_y=5.4, width=0.8, height=0.4),
        LayoutPosition(spec=specs[2], cx=3.0, top_y=4.8, width=0.8, height=0.4),
    ]
    graph = ComputationGraph(
        nodes=specs,
        links=[(0, 1), (1, 2)],
        inline_frames=[
            InlineFrameSpec(frame_id="experts", label="Experts", node_indices=[1, 2]),
        ],
    )
    return graph, positions


def test_shared_bus_clears_the_band_a_frame_caption_sits_in():
    """A bus crossing a frame has to clear its caption, not just its border."""
    from visualizer.render import (
        CONNECTOR_OBSTACLE_MARGIN,
        _inline_frame_caption_band_top,
        _inline_frame_draw_bounds,
        _lift_bus_y_above_inline_frame_interiors,
    )

    graph, positions = _captioned_frame_fixture()
    frame = graph.inline_frames[0]
    bounds = _inline_frame_draw_bounds(frame, positions, graph)
    caption_top = _inline_frame_caption_band_top(frame, bounds)
    assert caption_top > bounds.top, "a labeled frame reserves a band above its border"

    lifted = _lift_bus_y_above_inline_frame_interiors(
        (bounds.top + caption_top) / 2,
        graph=graph,
        positions=positions,
        x_left=0.0,
        x_right=4.0,
    )
    assert lifted >= caption_top + CONNECTOR_OBSTACLE_MARGIN


def test_feed_into_a_frame_head_routes_above_its_caption():
    """The corridor into a frame's first tile runs above the frame's caption."""
    from visualizer.render import (
        CONNECTOR_OBSTACLE_MARGIN,
        _inline_frame_caption_band_top,
        _inline_frame_draw_bounds,
        _inline_frame_top_member_route_y,
        _RenderAnchor,
    )

    graph, positions = _captioned_frame_fixture()
    frame = graph.inline_frames[0]
    bounds = _inline_frame_draw_bounds(frame, positions, graph)
    caption_top = _inline_frame_caption_band_top(frame, bounds)
    # The feeder sits just far enough above the frame that a plain exit stub would
    # drop the corridor into the caption band.
    source = _RenderAnchor(cx=0.6, top=6.15, bottom=5.75, left=0.2, right=1.0)
    target = _RenderAnchor(cx=3.0, top=5.4, bottom=5.0, left=2.6, right=3.4)

    route_y = _inline_frame_top_member_route_y(
        source,
        target,
        frame,
        positions,
        graph,
    )
    assert route_y >= caption_top + CONNECTOR_OBSTACLE_MARGIN - 1e-9
    assert route_y <= source.bottom


def test_measured_overlap_pass_keeps_the_frame_head_row_gap():
    """Re-stacking rows after measurement keeps the room a frame caption needs."""
    from visualizer.computation_graph import (
        DETAIL_LAYER_GAP,
        _frame_head_entry_gap,
        _topological_layers,
    )
    from visualizer.render_validate import resolve_measured_overlaps

    graph, positions = _captioned_frame_fixture()
    resolve_measured_overlaps(positions, graph, top_y=7.0)

    expected = _frame_head_entry_gap(
        graph,
        _topological_layers(graph),
        0,
        min_gap=DETAIL_LAYER_GAP,
    )
    assert expected > DETAIL_LAYER_GAP, "entering a captioned frame asks for extra room"
    gap = positions[0].bottom - positions[1].top_y
    assert gap >= expected - 1e-6, f"row gap {gap} lost the caption clearance {expected}"


def test_layer_siblings_split_apart_once_measurement_drifts_their_rows():
    """Same-layer tiles whose rows drift still get pulled apart horizontally."""
    from visualizer.computation_graph import (
        ComputationGraph,
        GraphNodeSpec,
        LayoutPosition,
    )
    from visualizer.render_validate import (
        MIN_HORIZONTAL_BLOCK_GAP,
        _node_content_left,
        _node_content_right,
        _resolve_same_row_tile_overlaps,
    )

    specs = [
        GraphNodeSpec(key="input", label="hidden_states"),
        GraphNodeSpec(key="experts", label="Experts"),
        GraphNodeSpec(key="router", label="Router"),
    ]
    # The input row grew during measurement, so the two roots no longer share a top_y
    # even though they still sit side by side in the same layer.
    positions = [
        LayoutPosition(spec=specs[0], cx=2.13, top_y=-1.149, width=1.049, height=0.464),
        LayoutPosition(spec=specs[1], cx=2.97, top_y=-0.939, width=1.389, height=0.322),
        LayoutPosition(spec=specs[2], cx=1.60, top_y=-1.867, width=0.629, height=0.322),
    ]
    graph = ComputationGraph(nodes=specs, links=[(0, 2)], inline_frames=[])

    _resolve_same_row_tile_overlaps(
        positions,
        min_gap=MIN_HORIZONTAL_BLOCK_GAP,
        graph=graph,
    )

    gap = _node_content_left(positions[1]) - _node_content_right(positions[0])
    assert gap >= MIN_HORIZONTAL_BLOCK_GAP - 1e-6, (
        f"drifted layer siblings still overlap: gap {gap}"
    )


def test_module_forward_math_lands_between_its_own_projections():
    """An MLP-style module shows the math it runs between its projections."""
    from visualizer.ast_analyze import analyze_source

    source = """
class Linear:
    pass

class RMSNorm:
    pass

class DenseMLP:
    def __init__(self):
        self.gate_up_proj = Linear()
        self.down_proj = Linear()

    def forward(self, hidden_states):
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        glu = gate * torch.sigmoid(gate)
        return self.down_proj((up + 1.0) * glu)

class DecoderLayer:
    def __init__(self):
        self.input_layernorm = RMSNorm()
        self.mlp = DenseMLP()

    def forward(self, hidden_states):
        normed = self.input_layernorm(hidden_states)
        return hidden_states + self.mlp(normed)
"""
    registry = analyze_source(source).class_registry

    mlp_calls = registry["DenseMLP"].forward_calls
    assert mlp_calls[0] == "gate_up_proj"
    assert mlp_calls[-1] == "down_proj"
    assert [call for call in mlp_calls if call.startswith("@op_")], (
        "the module's own activation math has to reach the diagram"
    )

    assert registry["DecoderLayer"].forward_calls == ["input_layernorm", "mlp"], (
        "a container layer leaves the math to its children and its residual to the merge"
    )


def test_frame_column_feeding_past_the_merge_keeps_the_chain_column():
    """The frame whose exit skips ahead packs first, so its skip stays beside its column."""
    from visualizer.computation_graph import (
        ComputationGraph,
        GraphNodeSpec,
        InlineFrameSpec,
        LayoutPosition,
        _sort_frame_columns_for_packing,
    )

    def spec(label: str) -> GraphNodeSpec:
        return GraphNodeSpec(key=label, label=label)

    def pos(cx: float, top_y: float, label: str) -> LayoutPosition:
        return LayoutPosition(spec=spec(label), cx=cx, top_y=top_y, width=1.0, height=0.4)

    positions = [
        pos(1.0, 10.0, "a0"),
        pos(1.0, 9.0, "a1"),
        pos(3.0, 10.0, "b0"),
        pos(3.0, 9.0, "b1"),
        pos(2.0, 8.0, "merge"),
        pos(2.0, 7.0, "tail"),
    ]
    graph = ComputationGraph(
        nodes=[position.spec for position in positions],
        links=[(0, 1), (1, 4), (2, 3), (3, 4), (3, 5), (4, 5)],
        inline_frames=[
            InlineFrameSpec(frame_id="a", label="A", node_indices=[0, 1]),
            InlineFrameSpec(frame_id="b", label="B", node_indices=[2, 3]),
        ],
    )
    columns = [[0, 1], [2, 3]]
    _sort_frame_columns_for_packing(graph, positions, columns, pad=0.1)
    assert columns[0] == [2, 3], "the frame that also feeds past the merge packs first"


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


def test_combine_op_node_uses_regular_rectangular_block_size():
    from visualizer.computation_graph import GraphNodeSpec, _diagram_size_for_spec

    spec = GraphNodeSpec(key="mul", label="Multiply")
    width, height = _diagram_size_for_spec(spec)
    assert width > height


def test_detail_palette_distinguishes_expanded_kernels_and_regular_ops():
    from visualizer.block_tree import BlockNode
    from visualizer.render import COLORS, _detail_block_facecolor, _detail_tile_text_color

    regular = BlockNode(
        attr_name="proj",
        class_name="Linear",
        role="other",
        label="Linear",
        is_basic=True,
    )
    modeled_leaf = BlockNode(
        attr_name="apply_rotary",
        class_name="PositionalOp",
        role="other",
        label="Apply rotary",
    )
    expanded = BlockNode(
        attr_name="attn",
        class_name="Attention",
        role="attention",
        label="Attention",
        children=[regular],
    )
    kernel = BlockNode(
        attr_name="fused",
        class_name="KernelSubOp",
        role="other",
        label="tl.dot",
    )
    torch_exp = BlockNode(
        attr_name="exp",
        class_name="KernelSubOp",
        role="other",
        label="Exp",
    )
    torch_cumsum = BlockNode(
        attr_name="cumsum",
        class_name="KernelSubOp",
        role="other",
        label="CumSum",
    )
    torch_softplus = BlockNode(
        attr_name="softplus",
        class_name="KernelSubOp",
        role="other",
        label="Softplus",
    )
    torch_attention = BlockNode(
        attr_name="@attention",
        class_name="AttentionOp",
        role="attention",
        label="Attention",
        details=["kernel: sdpa_attention_forward"],
    )
    custom_attn = BlockNode(
        attr_name="@attention",
        class_name="AttentionOp",
        role="attention",
        label="chunk_kda",
        details=["kernel: chunk_kda"],
    )

    assert _detail_block_facecolor(regular) == COLORS["basic_op"]
    assert _detail_block_facecolor(modeled_leaf) == COLORS["basic_op"]
    assert _detail_block_facecolor(expanded) == COLORS["attention"]
    assert _detail_block_facecolor(kernel) == COLORS["moe"]
    assert _detail_block_facecolor(torch_exp) == COLORS["basic_op"]
    assert _detail_block_facecolor(torch_cumsum) == COLORS["basic_op"]
    assert _detail_block_facecolor(torch_softplus) == COLORS["basic_op"]
    assert _detail_block_facecolor(torch_attention) == COLORS["basic_op"]
    assert _detail_block_facecolor(custom_attn) == COLORS["moe"]
    assert _detail_tile_text_color(COLORS["basic_op"]) == COLORS["text"]
    assert _detail_tile_text_color(COLORS["attention"]) == "white"
    assert _detail_tile_text_color(COLORS["moe"]) == "white"


def test_top_level_palette_uses_blue_only_when_component_is_expanded():
    from visualizer.blocks import BlockComponent
    from visualizer.block_tree import BlockNode
    from visualizer.extract import ArchitectureSpec
    from visualizer.render import COLORS, _top_level_module_style

    attention = BlockComponent(
        attr_name="self_attn",
        class_name="Attention",
        role="attention",
        label="Attention",
    )
    tree = BlockNode(
        attr_name="self_attn",
        class_name="Attention",
        role="attention",
        label="Attention",
        children=[
            BlockNode(
                attr_name="q_proj",
                class_name="Linear",
                role="other",
                label="Linear",
                is_basic=True,
            )
        ],
    )
    expanded_spec = ArchitectureSpec(
        name="Expanded",
        model_type="test",
        detailed_block_trees=[("Attention", tree)],
    )
    config_only_spec = ArchitectureSpec(name="Config only", model_type="test")

    assert _top_level_module_style(attention, expanded_spec) == (
        COLORS["attention"],
        "white",
        {},
    )
    fill, text, style = _top_level_module_style(attention, config_only_spec)
    assert fill == COLORS["basic_op"]
    assert text == COLORS["text"]
    assert style["edgecolor"] == "#000000"


def test_parallel_head_reference_uses_its_class_name_and_expanded_style():
    from visualizer.blocks import BlockComponent
    from visualizer.block_tree import BlockNode
    from visualizer.extract import ArchitectureSpec
    from visualizer.render import COLORS, _spine_display_label, _spine_module_style

    head = BlockComponent(
        attr_name="head",
        class_name="ParallelHead",
        role="head",
        label="ParallelHead",
    )
    tree = BlockNode(
        attr_name="head",
        class_name="ParallelHead",
        role="head",
        label="ParallelHead",
        children=[
            BlockNode(
                attr_name="hc_head",
                class_name="hc_head",
                role="other",
                label="hc head",
                children=[
                    BlockNode(
                        attr_name="@op",
                        class_name="Multiply",
                        role="other",
                        label="Multiply",
                        is_basic=True,
                    )
                ],
            ),
            BlockNode(
                attr_name="get_logits",
                class_name="Linear",
                role="other",
                label="Linear",
                is_basic=True,
            ),
        ],
    )
    spec = ArchitectureSpec(name="D", model_type="d")
    spec.export_block_trees = [("ParallelHead", tree)]

    assert _spine_display_label(head, spec) == "ParallelHead"
    assert _spine_module_style(head, spec) == (COLORS["attention"], "white", {})


def test_spine_moe_tile_keeps_its_section_when_the_chain_runs_straight():
    from visualizer.blocks import BlockComponent
    from visualizer.block_tree import BlockNode
    from visualizer.extract import ArchitectureSpec
    from visualizer.render import (
        COLORS,
        _detail_sections_to_render,
        _spine_display_label,
        _spine_module_style,
    )

    moe = BlockComponent(
        attr_name="ffn",
        class_name="MoE",
        role="moe",
        label="MoE",
    )
    tree = BlockNode(
        attr_name="ffn",
        class_name="MoE",
        role="moe",
        label="MoE",
        children=[
            BlockNode(
                attr_name="gate",
                class_name="Gate",
                role="router",
                label="Router",
            ),
            BlockNode(
                attr_name="shared_experts",
                class_name="Expert",
                role="expert",
                label="Expert",
                children=[
                    BlockNode(
                        attr_name="w1",
                        class_name="Linear",
                        role="other",
                        label="Linear",
                        is_basic=True,
                    ),
                    BlockNode(
                        attr_name="w2",
                        class_name="Linear",
                        role="other",
                        label="Linear",
                        is_basic=True,
                    ),
                ],
            ),
        ],
    )
    spec = ArchitectureSpec(name="D", model_type="d", block_components=[moe])
    spec.export_block_trees = [("MoE", tree)]

    assert "MoE" in [title for title, _tree, _sublabel in _detail_sections_to_render(spec)]
    assert _spine_display_label(moe, spec) == "MoE"
    assert _spine_module_style(moe, spec) == (COLORS["attention"], "white", {})


def test_module_list_expert_loop_keeps_routed_and_shared_moe_branches():
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import (
        ResidualAddSegment,
        SideFeedSegment,
        build_block_node,
        collect_computation_segments,
    )
    from visualizer.computation_graph import build_computation_graph

    source = """
class Gate:
    def forward(self, x, input_ids):
        return x, input_ids

class Expert:
    def __init__(self):
        self.w1 = Linear()
        self.w2 = Linear()
    def forward(self, x, weights=None):
        y = self.w1(x)
        return self.w2(y)

class MoE:
    def __init__(self):
        self.gate = Gate()
        self.experts = ModuleList([Expert()])
        self.shared_experts = Expert()
    def forward(self, x, input_ids):
        weights, indices = self.gate(x, input_ids)
        y = zeros_like(x)
        for i in range(1):
            expert = self.experts[i]
            idx, top = where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
        y += self.shared_experts(x)
        return y
"""
    analysis = analyze_source(source)
    cls = analysis.class_registry["MoE"]
    assert cls.forward_calls == ["gate", "experts", "shared_experts"]

    tree = build_block_node(
        attr_name="ffn",
        class_name="MoE",
        registry=analysis.class_registry,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    segments = collect_computation_segments(tree)
    routed = next(segment for segment in segments if isinstance(segment, SideFeedSegment))
    shared = next(segment for segment in segments if isinstance(segment, ResidualAddSegment))
    assert routed.consumer.attr_name == "experts"
    assert {side.source_kind for side in routed.sides} == {"forward_input", "prior_step"}
    assert shared.module.attr_name == "shared_experts"

    graph = build_computation_graph(tree, basic_ops=BasicOpFilter.for_detailed())
    assert {frame.frame_id for frame in graph.inline_frames} >= {"experts", "shared_experts"}
    plus = next(
        index
        for index, spec in enumerate(graph.nodes)
        if spec.label in {"Add", "+"}
        and len([source for source, target in graph.links if target == index]) == 2
    )
    incoming = {source for source, target in graph.links if target == plus}
    assert len(incoming) == 2


def test_expanded_pipeline_is_blue_while_contiguous_is_gray():
    from visualizer.block_tree import BlockNode
    from visualizer.render import COLORS, _detail_block_facecolor

    pipeline = BlockNode(
        attr_name="@attn_pipeline",
        class_name="KernelPipeline",
        role="attention",
        label="sparse_attn pipeline",
        children=[
            BlockNode(
                attr_name="@contiguous",
                class_name="KernelOp",
                role="other",
                label="Contiguous",
                details=["kernel: contiguous"],
            )
        ],
    )

    assert _detail_block_facecolor(pipeline) == COLORS["attention"]
    assert _detail_block_facecolor(pipeline.children[0]) == COLORS["basic_op"]


def test_rebinding_forward_input_does_not_invent_parallel_head_residual():
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import (
        ResidualAddSegment,
        build_block_node,
        collect_computation_segments,
    )

    source = """
class ParallelHead:
    def forward(self, x, hc_fn, hc_scale, hc_base, norm):
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        return self.get_logits(norm(x))

    def hc_head(self, x, hc_fn, hc_scale, hc_base):
        return x * hc_scale + hc_base

    def get_logits(self, x):
        return x
"""
    analysis = analyze_source(source, filename="model.py")
    structure = analysis.class_registry["ParallelHead"]
    assert structure.side_inputs == {}

    tree = build_block_node(
        attr_name="head",
        class_name="ParallelHead",
        registry=analysis.class_registry,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    assert not any(
        isinstance(segment, ResidualAddSegment)
        for segment in collect_computation_segments(tree)
    )


def test_computation_figure_output_collects_all_terminal_paths():
    from visualizer.computation_graph import (
        ComputationGraph,
        GraphNodeSpec,
        SYNTHETIC_OUTPUT,
        add_forward_output,
    )

    graph = ComputationGraph(
        nodes=[
            GraphNodeSpec(key="left", label="Left"),
            GraphNodeSpec(key="right", label="Right"),
        ]
    )
    output_index = add_forward_output(graph)

    assert output_index is not None
    assert graph.nodes[output_index].synthetic == SYNTHETIC_OUTPUT
    assert graph.nodes[output_index].label == "Output"
    assert {src for src, tgt in graph.links if tgt == output_index} == {0, 1}


def test_standalone_detail_figure_draws_output_like_its_input():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.block_tree import BlockNode
    from visualizer.render import COLORS, DiagramLayout, _render_laid_out_computation_graph

    tree = BlockNode(
        attr_name="proj",
        class_name="Linear",
        role="other",
        label="Linear",
        is_basic=True,
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    layout = DiagramLayout()
    _render_laid_out_computation_graph(
        layout,
        ax,
        tree,
        cx=3.0,
        top_y=5.0,
        block_w=3.0,
        draw_section_frame=True,
    )

    by_id = {node.node_id: node for node in layout.nodes}
    # Same fill the top-level figure gives "Tokenized text" and its own "Output".
    assert by_id["@input"].facecolor == COLORS["embed"]
    assert by_id["@output"].facecolor == by_id["@input"].facecolor
    assert by_id["@output"].text_color == by_id["@input"].text_color
    plt.close(fig)


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


def test_moe_router_weights_dock_on_the_expert_step_that_consumes_them(tmp_path, monkeypatch):
    """The gate feed enters the multiply that scales the expert, not the frame head."""
    import matplotlib

    matplotlib.use("Agg")

    from visualizer import render as render_module
    from visualizer.computation_graph import SYNTHETIC_INPUT
    from visualizer.loader import build_detailed_basic_ops
    from visualizer.render import (
        CONNECTOR_OBSTACLE_MARGIN,
        PARALLEL_CONNECTOR_CHANNEL_GAP,
        PARALLEL_CONNECTOR_COORD_EPS,
        TOP_ENTRY_PORT_GAP,
        _connector_axis_segments,
        _connector_target_top_entry_y,
    )

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots"
        / "60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py"
    )
    if not code_path.exists():
        pytest.skip("DeepSeek-V4 modeling file not cached locally")

    spec = load_architecture(
        "deepseek-ai/DeepSeek-V4-Flash",
        code_path=code_path,
        detailed=True,
        basic_ops=build_detailed_basic_ops(),
    )
    captured: list[dict] = []
    original = render_module._collect_detail_link_paths

    def capture(**kwargs):
        paths = original(**kwargs)
        captured.append(
            {
                "graph": kwargs["graph"],
                "anchors": dict(kwargs["anchors"]),
                "positions": list(kwargs["positions"]),
                "link_paths": {key: list(points) for key, points in paths.items()},
                "target_bus": dict(kwargs["target_bus"]),
                "merge_link_bus": dict(kwargs["merge_link_bus"]),
            }
        )
        return paths

    monkeypatch.setattr(render_module, "_collect_detail_link_paths", capture)
    render_diagram(spec, tmp_path / "deepseek_sections.svg", detailed=True)
    section = next(
        entry
        for entry in captured
        if any(node.label == "Router" for node in entry["graph"].nodes)
    )
    graph = section["graph"]
    anchors = section["anchors"]
    positions = section["positions"]
    link_paths = section["link_paths"]
    input_index = next(
        index for index, node in enumerate(graph.nodes) if node.synthetic == SYNTHETIC_INPUT
    )
    router_index = next(
        index for index, node in enumerate(graph.nodes) if node.label == "Router"
    )
    router_link = next(link for link in link_paths if link[0] == router_index)
    weights_index = router_link[1]
    # The gate output scales the expert result, so it belongs on that multiply.
    assert graph.nodes[weights_index].label == "Multiply"
    experts_frame = next(
        frame for frame in graph.inline_frames if weights_index in frame.node_indices
    )
    frame_head = max(experts_frame.node_indices, key=lambda index: positions[index].top_y)
    assert weights_index != frame_head
    # A second feed on the frame head would mean the side input was docked on the
    # frame instead of on the step that reads it.
    assert [link for link in link_paths if link[1] == frame_head] == [
        (input_index, frame_head)
    ]

    target = anchors[weights_index]
    entry_y = _connector_target_top_entry_y(target)
    router_points = link_paths[router_link]
    chain_points = link_paths[(input_index, frame_head)]

    assert abs(router_points[-1][1] - entry_y) <= PARALLEL_CONNECTOR_COORD_EPS
    assert target.left < router_points[-1][0] < target.right

    router_source = anchors[router_index]
    for orientation, coord, _lo, _hi, _index in _connector_axis_segments(router_points):
        if orientation != "h":
            continue
        assert abs(router_source.bottom - coord) >= PARALLEL_CONNECTOR_COORD_EPS - 1e-9

    verticals = [
        segment for segment in _connector_axis_segments(chain_points) if segment[0] == "v"
    ]
    horizontals = [
        segment for segment in _connector_axis_segments(router_points) if segment[0] == "h"
    ]
    for _, x, y_lo, y_hi, _index in verticals:
        for _, y, x_lo, x_hi, _other in horizontals:
            assert not (x_lo < x < x_hi and y_lo < y < y_hi), (
                f"router feed crosses the input drop lane at ({x:.3f}, {y:.3f})"
            )

    # The router may not stand over the lane, or the input has to detour around
    # the whole expert column to reach the frame head.
    assert (
        min(x for x, _y in chain_points)
        >= anchors[frame_head].left - CONNECTOR_OBSTACLE_MARGIN
    )

    input_links = [link for link in link_paths if link[0] == input_index]
    source_anchor = anchors[input_index]
    # Each leg leaves on a port of its own: legs stacked on one column draw as a single
    # line that appears to stop where they part.
    ports = {link: link_paths[link][0][0] for link in input_links}
    assert all(
        source_anchor.left <= port <= source_anchor.right for port in ports.values()
    )
    assert len({round(port, 6) for port in ports.values()}) == len(input_links)
    # Ports run in the same order as the columns the legs head for, which is what keeps
    # them from crossing each other on the way out.
    def _departure_x(link):
        points = link_paths[link]
        return next(
            (x for x, _y in points[1:] if abs(x - points[0][0]) > PARALLEL_CONNECTOR_COORD_EPS),
            points[0][0],
        )

    by_departure = sorted(input_links, key=_departure_x)
    assert [ports[link] for link in by_departure] == sorted(
        ports[link] for link in by_departure
    )
    departure_rows = sorted(
        (
            segment[1]
            for link in input_links
            for segment in _connector_axis_segments(link_paths[link])
            if segment[0] == "h"
        ),
    )
    if len(departure_rows) > 1:
        assert all(
            abs(first - second) <= PARALLEL_CONNECTOR_COORD_EPS
            or abs(first - second) >= PARALLEL_CONNECTOR_CHANNEL_GAP - 1e-6
            for first, second in zip(departure_rows, departure_rows[1:])
        ), "departure rows must either be the same row or a clear channel apart"
    for link in input_links:
        target_anchor = anchors[link[1]]
        assert abs(link_paths[link][-1][1] - target_anchor.top) <= PARALLEL_CONNECTOR_COORD_EPS
    right_expert = max(
        (link[1] for link in input_links if graph.nodes[link[1]].label == "Linear"),
        key=lambda index: anchors[index].cx,
    )
    assert abs(link_paths[(input_index, right_expert)][-1][0] - anchors[right_expert].cx) <= 1e-9

    add_index = next(index for index, node in enumerate(graph.nodes) if node.label in {"Add", "+"})
    add_links = [link for link in link_paths if link[1] == add_index]
    assert len(add_links) == 2
    entry_xs = sorted(link_paths[link][-1][0] for link in add_links)
    assert entry_xs[1] - entry_xs[0] >= TOP_ENTRY_PORT_GAP - PARALLEL_CONNECTOR_COORD_EPS
    # A feed standing over its own port drops straight in and needs no approach run at all.
    # The one that has to come across turns above the tile, on a run the other keeps clear of.
    approach_runs = []
    for link in add_links:
        horizontals = [
            segment for segment in _connector_axis_segments(link_paths[link])
            if segment[0] == "h"
        ]
        if horizontals:
            approach_runs.append(horizontals[-1])
    for index, (_, _y, lo, hi, _link) in enumerate(approach_runs):
        for _, _other_y, other_lo, other_hi, _other_link in approach_runs[index + 1 :]:
            assert hi <= other_lo or other_hi <= lo

    from visualizer.render import (
        _find_connector_inline_frame_overlaps,
        _find_connector_node_clearance_violations,
    )

    assert not _find_connector_inline_frame_overlaps(
        link_paths,
        graph=graph,
        positions=positions,
    )
    assert not _find_connector_node_clearance_violations(
        link_paths,
        graph=graph,
        anchors=anchors,
        label_obstacles=[],
        positions=positions,
    )

    indexer = next(
        entry
        for entry in captured
        if any(node.label == "Floor divide" for node in entry["graph"].nodes)
        and sum(node.label == "Apply rotary emb" for node in entry["graph"].nodes) >= 2
    )
    indexer_graph = indexer["graph"]
    indexer_anchors = indexer["anchors"]
    topk = next(
        index for index, node in enumerate(indexer_graph.nodes) if node.label == "TopK"
    )
    topk_links = [link for link in indexer["link_paths"] if link[1] == topk]
    assert sorted(indexer_graph.nodes[source].label for source, _target in topk_links) == [
        "Add",
        "Floor divide",
    ]
    # TopK has two independent operands, not a shared connector trunk. Their
    # topology-assigned approach levels must remain distinct even when their
    # horizontal spans overlap.
    assert topk not in indexer["target_bus"]
    topk_levels = [indexer["merge_link_bus"][link] for link in topk_links]
    assert abs(topk_levels[0] - topk_levels[1]) >= PARALLEL_CONNECTOR_CHANNEL_GAP
    main_rotary = next(
        index
        for index, node in enumerate(indexer_graph.nodes)
        if node.label == "Apply rotary emb"
        and not any(index in frame.node_indices for frame in indexer_graph.inline_frames)
    )
    floor_divide = next(
        index for index, node in enumerate(indexer_graph.nodes) if node.label == "Floor divide"
    )
    assert (
        indexer_anchors[floor_divide].left - indexer_anchors[main_rotary].right
        >= CONNECTOR_OBSTACLE_MARGIN - 1e-9
    )
    assert not _find_connector_inline_frame_overlaps(
        indexer["link_paths"],
        graph=indexer_graph,
        positions=indexer["positions"],
    )


def test_orthogonal_connector_path_collapses_near_duplicates_without_slanting():
    """Merging points closer than the coordinate epsilon must keep segments axis-aligned."""
    from visualizer.render import _ensure_orthogonal_connector_path

    jogged = [
        (0.841, -16.415),
        (0.841, -16.515),
        (0.841, -16.535),
        (1.540, -16.535),
        (1.540, -16.600),
    ]
    fixed = _ensure_orthogonal_connector_path(jogged)

    assert fixed[0] == jogged[0]
    assert fixed[-1] == jogged[-1]
    for (x1, y1), (x2, y2) in zip(fixed, fixed[1:]):
        assert abs(x1 - x2) <= 1e-9 or abs(y1 - y2) <= 1e-9, (
            f"slanted segment ({x1:.3f},{y1:.3f})->({x2:.3f},{y2:.3f})"
        )


_SWIGLU_BIAS_SOURCE = """
class Linear:
    pass

class SwigluMLP:
    def __init__(self, config):
        self.gate_up_proj = Linear()
        self.down_proj = Linear()
        self.swiglu_alpha = 1.702

    def forward(self, hidden_states):
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
        return self.down_proj((up + 1.0) * glu)
"""


def test_swiglu_bias_add_branch_takes_its_own_leg_off_the_projection():
    """`(up + 1.0) * glu` draws the add on its own branch and the join with two feeds."""
    from collections import defaultdict

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
        CONNECTOR_OBSTACLE_MARGIN,
        PARALLEL_CONNECTOR_COORD_EPS,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _collect_detail_link_paths,
        _compute_detail_connector_buses,
        _connector_axis_segments,
        _connector_source_bottom_exit_y,
    )
    from visualizer.render_validate import finalize_detail_layout

    analysis = analyze_source(_SWIGLU_BIAS_SOURCE, all_tensor_ops=True)
    basic = BasicOpFilter.for_detailed()
    tree = build_block_node(
        attr_name="mlp",
        class_name="SwigluMLP",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(tree, basic_ops=basic)
    _fig, ax = plt.subplots(figsize=(16, 13))
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
    input_index = next(
        index for index, node in enumerate(graph.nodes) if node.synthetic == SYNTHETIC_INPUT
    )
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
    try:
        add_index = next(
            index for index, node in enumerate(graph.nodes) if node.label == "Add"
        )
        proj_index = next(src for src, tgt in links if tgt == add_index)
        join_index = next(tgt for src, tgt in links if src == add_index)

        # `(up + 1.0) * glu` reads two values, so the join has two feeds: the add that
        # raised `up`, and the gated multiply that produced `glu`. The projection reaches
        # the join through the add, never beside it.
        join_feeds = sorted(src for src, tgt in links if tgt == join_index)
        glu_index = next(index for index in join_feeds if index != add_index)
        assert join_feeds == sorted((add_index, glu_index)), (
            f"the join reads {len(join_feeds)} values, but the multiply combines two"
        )
        assert graph.nodes[glu_index].label == "Multiply"
        assert [src for src, tgt in links if tgt == add_index] == [proj_index], (
            "the add raises the projection's own output"
        )

        for link_key, points in link_paths.items():
            for (x1, y1), (x2, y2) in zip(points, points[1:]):
                assert abs(x1 - x2) <= 1e-9 or abs(y1 - y2) <= 1e-9, (
                    f"{link_key} has a slanted segment "
                    f"({x1:.3f},{y1:.3f})->({x2:.3f},{y2:.3f})"
                )

        # The projection feeds the gate chain, the gated multiply and the add, so each leg
        # leaves by a port of its own.
        proj = anchors[proj_index]
        exit_y = _connector_source_bottom_exit_y(proj)
        proj_legs = [tgt for src, tgt in links if src == proj_index]
        assert len(proj_legs) >= 3
        exits = []
        for target in proj_legs:
            points = link_paths[(proj_index, target)]
            assert proj.left <= points[0][0] <= proj.right, (
                "every leg must leave by a port on the projection's own bottom edge"
            )
            assert abs(points[0][1] - exit_y) <= PARALLEL_CONNECTOR_COORD_EPS
            exits.append(points[0][0])
        for first, second in zip(sorted(exits), sorted(exits)[1:]):
            assert second - first > PARALLEL_CONNECTOR_COORD_EPS, (
                "legs carrying to different places need ports of their own"
            )

        from visualizer.render import _find_connector_node_clearance_violations

        assert not _find_connector_node_clearance_violations(
            link_paths,
            graph=graph,
            anchors=anchors,
            label_obstacles=plan.label_obstacles,
            positions=positions,
        ), "no leg may reach its target by cutting through a tile"

        # Where one step of the gate chain sits directly under the last with nothing in
        # between, it should drop straight into it rather than step around anything.
        stacked = []
        for src, tgt in links:
            above, below = anchors[src], anchors[tgt]
            if abs(above.cx - below.cx) > PARALLEL_CONNECTOR_COORD_EPS:
                continue
            if any(
                index not in (src, tgt)
                and min(above.bottom, other.top) - max(below.top, other.bottom) > 0
                and min(above.right, other.right) - max(above.left, other.left) > 0
                for index, other in anchors.items()
            ):
                continue
            stacked.append((src, tgt))
        assert stacked, "the gate chain should be stacked in one column"
        for link_key in stacked:
            assert len(link_paths[link_key]) == 2, (
                f"{link_key} is stacked directly with a clear gap, so it should drop "
                f"straight down"
            )

        # Two feeds land on the join's top edge, so they need separate ports.
        join = anchors[join_index]
        add_points = link_paths[(add_index, join_index)]
        glu_points = link_paths[(glu_index, join_index)]
        ports = sorted((add_points[-1][0], glu_points[-1][0]))
        assert ports[1] - ports[0] >= CONNECTOR_OBSTACLE_MARGIN
        for points in (add_points, glu_points):
            assert abs(points[-1][1] - join.top) <= PARALLEL_CONNECTOR_COORD_EPS
            assert join.left <= points[-1][0] <= join.right
            assert abs(points[-1][0] - points[-2][0]) <= PARALLEL_CONNECTOR_COORD_EPS, (
                "a top entry approaches straight down onto its port"
            )
    finally:
        plt.close(_fig)


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


def test_kimi_mla_spread_attention_merge_routes():
    """Cross-column Pad/kv feeders use distinct ports and avoid overlaying Attention."""
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
        CONNECTOR_EXIT_STUB,
        TOP_ENTRY_PORT_GAP,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _collect_detail_link_paths,
        _compute_detail_connector_buses,
        _inline_frame_for_top_member,
        _path_crosses_attached_block_edge_band,
        _spread_merge_horizontal_below_target_corridor,
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
    anchors = _anchors_from_detail_plan(positions, plan, graph)
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
    side_feeders = [
        (src, tgt)
        for src, tgt in links
        if tgt == attn_index and abs(positions[src].cx - positions[attn_index].cx) > 0.08
    ]
    assert len(side_feeders) >= 2
    entry_ports = [link_paths[link][-1][0] for link in side_feeders]
    assert len(set(round(x, 3) for x in entry_ports)) == len(entry_ports)
    for index in range(len(entry_ports) - 1):
        assert entry_ports[index + 1] - entry_ports[index] >= TOP_ENTRY_PORT_GAP - 0.02
    attn_anchor = anchors[attn_index]
    for src, tgt in side_feeders:
        points = link_paths[(src, tgt)]
        assert not _path_crosses_attached_block_edge_band(
            points,
            source=anchors[src],
            target=attn_anchor,
        )
        assert not _spread_merge_horizontal_below_target_corridor(points, attn_anchor)
        assert abs(points[-1][0] - merge_entry_x[(src, tgt)]) < 0.02

    g_proj_index = next(
        i
        for i, node in enumerate(graph.nodes)
        if node.block is not None and node.block.attr_name == "g_proj"
    )
    frame = _inline_frame_for_top_member(graph, g_proj_index)
    assert frame is not None
    assert abs(anchors[g_proj_index].top - positions[g_proj_index].top_y) <= 0.02
    shared_tee = source_bus[input_index]
    g_path = link_paths[(input_index, g_proj_index)]
    assert any(abs(y - shared_tee) < 0.02 for _x, y in g_path)
    assert abs(g_path[-1][1] - positions[g_proj_index].top_y) < 0.02

    pad_index = next(i for i, node in enumerate(graph.nodes) if node.label == "Pad")
    kv_linear_index = next(src for src, tgt in graph.links if tgt == pad_index)
    linear_pad_gap = positions[kv_linear_index].bottom - positions[pad_index].top_y
    assert linear_pad_gap <= min_vertical_block_gap() + 0.02, (
        f"Linear->Pad slack {linear_pad_gap:.3f} should shrinkwrap to the column gap"
    )
    pad_path = link_paths[(pad_index, attn_index)]
    pad_cx = positions[pad_index].cx
    exit_x, exit_y = pad_path[0]
    turn_x, turn_y = pad_path[1]
    assert abs(exit_x - pad_cx) < 0.04 and abs(turn_x - pad_cx) < 0.04, (
        f"Pad exit must leave straight down its own column: {pad_path[:2]}"
    )
    assert exit_y - turn_y >= CONNECTOR_EXIT_STUB - 0.02, (
        f"Pad exit must clear the exit stub before jogging: {pad_path[:2]}"
    )
    plt.close(fig)


def _connector_segment_crossing(seg_a, seg_b):
    """Point where one vertical and one horizontal connector segment intersect."""
    (ax1, ay1), (ax2, ay2) = seg_a
    (bx1, by1), (bx2, by2) = seg_b
    a_vertical = abs(ax1 - ax2) < 0.005
    b_vertical = abs(bx1 - bx2) < 0.005
    if a_vertical == b_vertical:
        return None
    if b_vertical:
        seg_a, seg_b = seg_b, seg_a
        (ax1, ay1), (ax2, ay2) = seg_a
        (bx1, by1), (bx2, by2) = seg_b
    low_y, high_y = sorted((ay1, ay2))
    low_x, high_x = sorted((bx1, bx2))
    if low_y + 0.01 < by1 < high_y - 0.01 and low_x + 0.01 < ax1 < high_x - 0.01:
        return (ax1, by1)
    return None


def _kimi_detail_section_link_paths(tmp_path, monkeypatch, node_label: str):
    """Render the full detailed diagram and capture the section holding node_label."""
    import matplotlib

    matplotlib.use("Agg")

    from visualizer import render as render_module
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
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"]),
    )
    captured: list[dict] = []
    original = render_module._collect_detail_link_paths

    def capture(**kwargs):
        paths = original(**kwargs)
        captured.append(
            {
                "graph": kwargs["graph"],
                "anchors": dict(kwargs["anchors"]),
                "link_paths": {key: list(points) for key, points in paths.items()},
                "merge_entry_x": dict(kwargs["merge_entry_x"]),
                "target_bus": dict(kwargs["target_bus"]),
            }
        )
        return paths

    monkeypatch.setattr(render_module, "_collect_detail_link_paths", capture)
    render_diagram(spec, tmp_path / "kimi_sections.svg", detailed=True)
    return next(
        section
        for section in captured
        if any(node.label == node_label for node in section["graph"].nodes)
    )


def test_kimi_kda_stacked_feeders_enter_gated_delta_rule_without_crossing(
    tmp_path, monkeypatch
):
    """CumSum passes beside Intra-chunk WY instead of crossing its merge bus."""
    from visualizer.render import (
        CONNECTOR_OBSTACLE_MARGIN,
        TOP_ENTRY_PORT_GAP,
        _connector_source_exit_y,
        _connector_turn_before_clearing_source,
    )

    section = _kimi_detail_section_link_paths(
        tmp_path,
        monkeypatch,
        "chunk_gated_delta_rule_fwd_h",
    )
    graph = section["graph"]
    anchors = section["anchors"]
    link_paths = section["link_paths"]
    index_of = {node.label: i for i, node in enumerate(graph.nodes)}
    cumsum = index_of["CumSum"]
    intra = index_of["chunk_kda_fwd_intra"]
    gated = index_of["chunk_gated_delta_rule_fwd_h"]

    intra_anchor = anchors[intra]
    gated_anchor = anchors[gated]
    intra_leg = link_paths[(intra, gated)]
    cumsum_leg = link_paths[(cumsum, gated)]

    for leg in (intra_leg, cumsum_leg):
        assert abs(leg[-1][1] - gated_anchor.top) < 1e-6, "legs must land on the top edge"
        assert abs(leg[-1][0] - leg[-2][0]) < 1e-6, "top entries must be vertical"

    assert abs(intra_leg[-1][0] - gated_anchor.cx) < 1e-6, (
        "the feeder directly above keeps the center port"
    )
    bypass_x = cumsum_leg[-1][0]
    assert abs(bypass_x - gated_anchor.cx) >= TOP_ENTRY_PORT_GAP - 1e-6, (
        f"bypass port {bypass_x:.4f} must stand off the center port at {gated_anchor.cx:.4f}"
    )
    assert gated_anchor.left <= bypass_x <= gated_anchor.right, (
        f"bypass port {bypass_x:.4f} must sit on the top edge it feeds"
    )
    # The bypass takes whichever side its source approaches from, so the side is not the
    # point; what matters is that the run carrying it past Intra-chunk WY stays outside it.
    passed_runs = [
        x1
        for (x1, y1), (x2, y2) in zip(cumsum_leg, cumsum_leg[1:])
        if abs(x1 - x2) < 1e-9
        and min(y1, y2) < intra_anchor.top
        and max(y1, y2) > intra_anchor.bottom
    ]
    assert passed_runs, "the bypass must run past Intra-chunk WY"
    for gutter_x in passed_runs:
        assert (
            gutter_x <= intra_anchor.left - CONNECTOR_OBSTACLE_MARGIN + 1e-6
            or gutter_x >= intra_anchor.right + CONNECTOR_OBSTACLE_MARGIN - 1e-6
        ), (
            f"bypass gutter {gutter_x:.4f} cuts Intra-chunk WY spanning "
            f"[{intra_anchor.left:.4f}, {intra_anchor.right:.4f}]"
        )

    intra_bus = section["target_bus"][intra]
    tee_y = cumsum_leg[1][1]
    assert tee_y < intra_bus, "bypass must tee below the merge bus it would otherwise cross"
    assert tee_y > intra_anchor.top, "bypass must tee above the tile it passes"

    all_segments = [
        (link, list(zip(points, points[1:]))) for link, points in link_paths.items()
    ]
    for index, (link_a, segments_a) in enumerate(all_segments):
        for link_b, segments_b in all_segments[index + 1 :]:
            for seg_a in segments_a:
                for seg_b in segments_b:
                    crossing = _connector_segment_crossing(seg_a, seg_b)
                    assert crossing is None, (
                        f"{link_a} crosses {link_b} at "
                        f"({crossing[0]:.4f}, {crossing[1]:.4f})"
                    )

    for (src, _tgt), points in link_paths.items():
        assert (
            _connector_turn_before_clearing_source(
                points,
                y_exit=_connector_source_exit_y(graph, src, anchors[src]),
                source_cx=anchors[src].cx,
            )
            is None
        ), f"{(src, _tgt)} turns horizontally before clearing its exit stub: {points}"


def test_same_column_bypass_ports_and_corridor():
    """Stacked same-column feeders get ordered ports and a tee below the passed bus."""
    from visualizer.render import (
        CONNECTOR_OBSTACLE_MARGIN,
        _RenderAnchor,
        _same_column_bypass_assignments,
        _same_column_bypass_top_entry_route,
    )

    upper = _RenderAnchor(cx=2.0, top=-1.0, bottom=-1.4, left=1.6, right=2.4)
    middle = _RenderAnchor(cx=2.0, top=-2.0, bottom=-2.4, left=1.5, right=2.5)
    target = _RenderAnchor(cx=2.0, top=-3.0, bottom=-3.4, left=1.3, right=2.7)
    anchors = {0: upper, 1: middle, 2: target}
    positions = [
        type("Pos", (), {"cx": anchor.cx, "bottom": anchor.bottom})()
        for anchor in (upper, middle, target)
    ]
    middle_bus = middle.top + 0.3
    assignments = _same_column_bypass_assignments(
        [(0, 2), (1, 2)],
        target,
        positions=positions,
        anchors=anchors,
        target_bus={1: middle_bus},
    )
    assert list(assignments) == [(0, 2)], "only the feeder above the stack bypasses"
    bypass = assignments[(0, 2)]
    assert bypass.port_x >= middle.right + CONNECTOR_OBSTACLE_MARGIN - 1e-6
    assert bypass.port_x <= target.right
    assert middle.top < bypass.corridor_y < middle_bus
    assert bypass.gutter_x is None, "a port fits beside the passed tile"

    route = _same_column_bypass_top_entry_route(upper, target, bypass)
    assert route == [
        (upper.cx, upper.bottom),
        (upper.cx, bypass.corridor_y),
        (bypass.port_x, bypass.corridor_y),
        (bypass.port_x, target.top),
    ]


def test_bypass_past_a_full_width_tile_enters_beside_the_center():
    """With no port left beyond the passed tile, the bypass returns to the center."""
    from visualizer.render import (
        CONNECTOR_OBSTACLE_MARGIN,
        TOP_ENTRY_PORT_GAP,
        _RenderAnchor,
        _same_column_bypass_assignments,
        _same_column_bypass_top_entry_route,
    )

    upper = _RenderAnchor(cx=2.0, top=-1.0, bottom=-1.4, left=1.6, right=2.4)
    middle = _RenderAnchor(cx=2.0, top=-2.0, bottom=-2.4, left=1.35, right=2.65)
    target = _RenderAnchor(cx=2.0, top=-3.0, bottom=-3.4, left=1.3, right=2.7)
    anchors = {0: upper, 1: middle, 2: target}
    positions = [
        type("Pos", (), {"cx": anchor.cx, "bottom": anchor.bottom})()
        for anchor in (upper, middle, target)
    ]
    bypass = _same_column_bypass_assignments(
        [(0, 2), (1, 2)],
        target,
        positions=positions,
        anchors=anchors,
        target_bus={1: middle.top + 0.3},
    )[(0, 2)]

    assert bypass.port_x == pytest.approx(target.cx + TOP_ENTRY_PORT_GAP)
    assert bypass.gutter_x >= max(middle.right, target.right) + CONNECTOR_OBSTACLE_MARGIN
    assert target.top < bypass.jog_y < middle.bottom

    route = _same_column_bypass_top_entry_route(upper, target, bypass)
    assert route == [
        (upper.cx, upper.bottom),
        (upper.cx, bypass.corridor_y),
        (bypass.gutter_x, bypass.corridor_y),
        (bypass.gutter_x, bypass.jog_y),
        (bypass.port_x, bypass.jog_y),
        (bypass.port_x, target.top),
    ]


def test_merge_legs_from_one_side_get_their_own_corridor():
    """Two feeds arriving from the same side nest instead of sharing one line."""
    from visualizer.render import (
        PARALLEL_CONNECTOR_CHANNEL_GAP,
        _connector_min_bus_y_above_target,
        _nest_same_side_merge_bus_levels,
        _RenderAnchor,
    )

    target = _RenderAnchor(cx=2.73, top=-5.89, bottom=-6.21, left=2.32, right=3.14)
    left = _RenderAnchor(cx=2.13, top=-5.16, bottom=-5.48, left=1.9, right=2.36)
    near = _RenderAnchor(cx=3.26, top=0.65, bottom=0.33, left=3.0, right=3.52)
    far = _RenderAnchor(cx=4.31, top=0.88, bottom=0.56, left=4.05, right=4.57)
    anchors = {0: left, 1: near, 2: far, 3: target}
    positions = [
        type("Pos", (), {"cx": anchor.cx, "bottom": anchor.bottom})()
        for anchor in (left, near, far, target)
    ]
    links = [(0, 3), (1, 3), (2, 3)]
    base = _connector_min_bus_y_above_target(target)
    merge_entry_x = {(0, 3): 2.65, (1, 3): 2.81, (2, 3): 2.89}
    merge_link_bus = {link: base for link in links}

    _nest_same_side_merge_bus_levels(
        links,
        tgt=3,
        positions=positions,
        anchors=anchors,
        merge_entry_x=merge_entry_x,
        merge_link_bus=merge_link_bus,
        obstacles=[],
    )

    assert merge_link_bus[(0, 3)] == pytest.approx(base), "the lone left leg stays put"
    assert merge_link_bus[(2, 3)] == pytest.approx(base), "the outer right leg stays lowest"
    assert merge_link_bus[(1, 3)] >= base + PARALLEL_CONNECTOR_CHANNEL_GAP - 1e-9, (
        "the nearer right leg needs its own level, not the outer leg's line"
    )


def test_shared_bus_classification_requires_link_ownership():
    """An unrelated connector on the same coordinate does not become a bus member."""
    from visualizer.render import _horizontal_segment_is_shared_bus

    assert _horizontal_segment_is_shared_bus(
        (1, 4),
        -2.0,
        target_bus={4: -2.0},
        source_bus={},
    )
    assert _horizontal_segment_is_shared_bus(
        (1, 4),
        -3.0,
        target_bus={},
        source_bus={1: -3.0},
    )
    assert not _horizontal_segment_is_shared_bus(
        (2, 5),
        -2.0,
        target_bus={4: -2.0},
        source_bus={1: -2.0},
    )


def _skip_chain_graph():
    """A framed chain a -> b -> c -> d with a skip from a to d and a dead-end off b."""
    from visualizer.computation_graph import ComputationGraph, GraphNodeSpec, InlineFrameSpec

    labels = ["a", "b", "leaf", "c", "d"]
    return ComputationGraph(
        nodes=[GraphNodeSpec(key=label, label=label) for label in labels],
        links=[(0, 1), (1, 2), (1, 3), (3, 4), (0, 4)],
        inline_frames=[
            InlineFrameSpec(frame_id="f", label="F", node_indices=[0, 1, 2, 3, 4]),
        ],
    )


def test_frame_border_needs_a_row_of_its_own_between_members():
    """A border between two stacked tiles has to fit its padding."""
    from visualizer.computation_graph import (
        ComputationGraph,
        FRAME_BORDER_CLEARANCE,
        GraphNodeSpec,
        InlineFrameSpec,
        _row_gap_rules,
    )
    from visualizer.render import INLINE_FRAME_CAPTION_BAND, INLINE_FRAME_PAD

    graph = ComputationGraph(
        nodes=[GraphNodeSpec(key=name, label=name) for name in ("in", "top", "bottom", "out")],
        links=[(0, 1), (1, 2), (2, 3)],
        inline_frames=[InlineFrameSpec(frame_id="f", label="F", node_indices=[1, 2])],
    )
    required = _row_gap_rules(graph)

    assert required(1, 2) == pytest.approx(0.0), "two members of one frame pack freely"
    assert required(0, 1) == pytest.approx(
        INLINE_FRAME_PAD + INLINE_FRAME_CAPTION_BAND + FRAME_BORDER_CLEARANCE
    ), "the top border and its caption sit between the feeder and the first member"
    assert required(2, 3) == pytest.approx(INLINE_FRAME_PAD + FRAME_BORDER_CLEARANCE), (
        "the bottom border sits between the last member and the step below"
    )


def test_skip_rows_reserve_room_for_the_connector_to_leave_its_column():
    """The rows a skip jogs through are wide enough to hold its horizontal runs."""
    from visualizer.computation_graph import (
        _inline_frame_column_skip_links,
        _row_gap_rules,
    )
    from visualizer.render import CONNECTOR_EXIT_STUB, CONNECTOR_OBSTACLE_MARGIN

    graph = _skip_chain_graph()
    # A jog needs its exit stub plus the margin of the step it passes.
    band = CONNECTOR_EXIT_STUB + CONNECTOR_OBSTACLE_MARGIN
    required = _row_gap_rules(graph)

    assert (0, 4) in _inline_frame_column_skip_links(graph, graph.inline_frames[0]), (
        "a link that passes steps of its own column is a skip"
    )
    assert required(0, 1) >= band, "the skip leaves its column below its source"
    assert required(3, 4) >= band, "and rejoins it above its target"


def test_dead_end_step_steps_out_of_the_column_it_would_block():
    """A step nothing reads gets its own column instead of sitting under the chain."""
    from visualizer.computation_graph import (
        LayoutPosition,
        _dead_end_branch_nodes_among,
        _layout_operation_dag_columns,
        _row_gap_rules,
    )
    from visualizer.render import CONNECTOR_EXIT_STUB, CONNECTOR_OBSTACLE_MARGIN

    graph = _skip_chain_graph()
    members = list(range(len(graph.nodes)))
    assert _dead_end_branch_nodes_among(graph, members) == {2}, "only the leaf is off the flow"

    positions = [
        LayoutPosition(spec=spec, cx=2.0, top_y=10.0 - index, width=1.0, height=0.4)
        for index, spec in enumerate(graph.nodes)
    ]
    _layout_operation_dag_columns(positions, graph, members)
    assert positions[2].cx < positions[1].cx, "the leaf moves aside"
    assert positions[1].cx == pytest.approx(positions[3].cx), "the chain keeps one column"
    assert _row_gap_rules(graph)(1, 2) >= CONNECTOR_EXIT_STUB + CONNECTOR_OBSTACLE_MARGIN, (
        "reaching the column beside the chain needs a run in the row above it"
    )


def test_shift_may_not_close_a_row_reserved_for_a_connector():
    """Clearing a frame border cannot squeeze a row a connector was promised."""
    from visualizer.render_validate import _rows_keep_reserved_gap
    from visualizer.text_measure import box_bounds_at

    box = box_bounds_at(2.0, 10.0, 1.0, 0.4)
    below = box_bounds_at(2.0, 9.4, 1.0, 0.4)
    required = lambda upper, lower: 0.17  # noqa: E731 - a fixed reservation under test

    assert _rows_keep_reserved_gap(
        0,
        1,
        box=box,
        shifted=box_bounds_at(2.0, 9.98, 1.0, 0.4),
        other=below,
        required_row_gap=required,
    ), "a hair of movement leaves the reserved row intact"
    assert not _rows_keep_reserved_gap(
        0,
        1,
        box=box,
        shifted=box_bounds_at(2.0, 9.85, 1.0, 0.4),
        other=below,
        required_row_gap=required,
    ), "dropping onto the reserved row strands the connector that needed it"


def test_connector_turn_before_clearing_source_flags_short_stubs():
    """A jog right below the source bottom counts as a horizontal exit."""
    from visualizer.render import _connector_turn_before_clearing_source

    jogged = [(2.0, -9.03), (2.0, -9.05), (2.1, -9.05), (2.1, -9.22)]
    assert _connector_turn_before_clearing_source(
        jogged,
        y_exit=-9.03,
        source_cx=2.0,
    ) == pytest.approx(-9.05)

    stubbed = [(2.0, -9.03), (2.0, -9.13), (2.1, -9.13), (2.1, -9.22)]
    assert (
        _connector_turn_before_clearing_source(stubbed, y_exit=-9.03, source_cx=2.0)
        is None
    )

    trunk_tee = [(2.0, -9.13), (2.4, -9.13), (2.4, -9.22)]
    assert (
        _connector_turn_before_clearing_source(trunk_tee, y_exit=-9.03, source_cx=2.0)
        is None
    )


def test_ensure_orthogonal_connector_path_squares_sub_eps_offsets():
    """A tiny horizontal offset must not render as a slanted connector."""
    from visualizer.render import _ensure_orthogonal_connector_path

    points = _ensure_orthogonal_connector_path([(2.3516, -12.70), (2.3447, -13.30)])
    assert len(points) == 2
    assert points[0][0] == points[1][0] == pytest.approx(2.3516)


def test_spread_merge_gutter_route_skips_overshoot_on_direct_drop():
    """A drop that already clears the stub must not detour right of the source."""
    from visualizer.render import (
        _RenderAnchor,
        _spread_merge_cross_column_gutter_route,
    )

    source = _RenderAnchor(cx=5.2, top=7.7, bottom=7.5, left=4.9, right=5.5)
    target = _RenderAnchor(cx=2.9, top=7.3, bottom=7.1, left=2.5, right=3.3)
    points = _spread_merge_cross_column_gutter_route(
        source,
        target,
        target.cx,
        7.4,
        [],
    )
    assert len(points) == 4, f"direct drop should be a plain L route: {points}"
    assert max(x for x, _y in points) <= source.cx + 1e-6, (
        f"route must not overshoot right of the source column: {points}"
    )


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
    """The positional block expands into the tensor math its own forward runs."""
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
    assert "Positional (RoPE)" in svg
    assert "rotary_emb" in svg
    for label in ("MatMul", "Concat", "Cosine", "Sine"):
        assert label in svg, f"missing rope operation {label}"
    # Synthetic stand-ins for rope internals must never come back.
    assert "Freq computation" not in svg
    assert "ApplyRotary" not in svg
    assert "cos, sin from inv_freq" not in svg
    assert "rotate query/key tensors" not in svg
    assert "Map token IDs to embeddings" not in svg


def test_expanded_spine_block_stays_on_the_model_axis_and_reports_every_output():
    """A spine block keeps its chain on the axis and hands back all of its outputs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node, expand_block_tree_inplace
    from visualizer.render import DiagramLayout, _render_laid_out_computation_graph

    source = (FIXTURES / "custom_model" / "modeling_custom.py").read_text(encoding="utf-8")
    analysis = analyze_source(source, filename="modeling_custom.py")
    basic = BasicOpFilter.for_detailed()
    tree = expand_block_tree_inplace(
        build_block_node(
            attr_name="rotary_emb",
            class_name="CustomRotaryEmbedding",
            registry=analysis.class_registry,
            basic_ops=basic,
        ),
        basic_ops=basic,
    )

    fig, ax = plt.subplots(figsize=(11, 13))
    fig.canvas.draw()
    spine_cx = 4.5
    layout = DiagramLayout()
    rendered = _render_laid_out_computation_graph(
        layout,
        ax,
        tree,
        cx=spine_cx,
        top_y=10.0,
        block_w=3.4,
        draw_section_frame=False,
        root_frame_label="Positional (RoPE) (rotary_emb)",
        include_input=False,
        align_chain_to_cx=True,
        basic_ops=basic,
    )

    assert rendered.entry is not None
    assert abs(rendered.entry.cx - spine_cx) <= 0.02
    for node in layout.nodes:
        assert abs(node.cx - spine_cx) <= 0.02, f"{node.label} drifted off the spine axis"
    # cos and sin both leave the module, so both have to reach the flow below it.
    assert len(rendered.exits) == 2
    assert all(exit_anchor.bottom > rendered.bottom for exit_anchor in rendered.exits)
    plt.close(fig)


def test_spine_block_side_output_paths_reach_the_downstream_flow():
    """Outputs stacked above the bottom of a spine block route around it, not through it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.render import _RenderAnchor, _connect_spine_block_side_outputs

    upper = _RenderAnchor(cx=4.5, top=-1.0, bottom=-1.4, left=4.0, right=5.0)
    lower = _RenderAnchor(cx=4.5, top=-2.0, bottom=-2.4, left=4.0, right=5.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    _connect_spine_block_side_outputs(
        ax,
        (upper, lower),
        cx=4.5,
        join_y=-2.6,
        corridor_x=5.4,
    )

    paths = [line.get_xydata().tolist() for line in ax.get_lines()]
    assert paths, "the side output was not drawn"
    xs = [x for path in paths for x, _y in path]
    ys = [y for path in paths for _x, y in path]
    # The detour leaves sideways and rejoins the flow below the block, never doubling back up.
    assert max(xs) >= 5.4 - 0.01
    assert min(ys) <= -2.6 + 0.01
    assert all(y <= upper.top + 0.01 for y in ys)
    assert all(x >= lower.right - 0.01 or y <= -2.6 + 0.01 for path in paths for x, y in path)
    plt.close(fig)


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
            BlockNode(attr_name="@op_l1_c0_cosine", class_name="Cosine", role="other", label="Cosine", is_basic=True),
            BlockNode(attr_name="@op_l2_c0_sine", class_name="Sine", role="other", label="Sine", is_basic=True),
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


def test_expand_block_tree_inplace_expands_straight_line_children():
    from visualizer.block_tree import BlockNode, expand_block_tree_inplace

    mlp = BlockNode(
        attr_name="shared_experts",
        class_name="KimiMLP",
        role="ffn",
        label="KimiMLP",
        children=[
            BlockNode(attr_name="gate_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="up_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="down_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
        ],
    )
    expanded = expand_block_tree_inplace(mlp)
    assert expanded.attr_name == "shared_experts"
    assert [child.attr_name for child in expanded.children] == ["gate_proj", "up_proj", "down_proj"]


def test_expand_block_tree_inplace_substitutes_single_op_subgraph():
    from visualizer.block_tree import BlockNode, expand_block_tree_inplace

    linear = BlockNode(
        attr_name="down_proj",
        class_name="Linear",
        role="other",
        label="Linear",
        is_basic=True,
    )
    wrapper = BlockNode(
        attr_name="routed_expert",
        class_name="KimiMLP",
        role="ffn",
        label="Expert",
        input_source="moe scores",
        children=[linear],
    )
    parent = BlockNode(
        attr_name="moe",
        class_name="MoE",
        role="moe",
        label="MoE",
        children=[
            wrapper,
            BlockNode(
                attr_name="merge",
                class_name="Merge",
                role="other",
                label="Merge",
                is_basic=True,
            ),
        ],
    )

    expanded = expand_block_tree_inplace(parent)
    assert len(expanded.children) == 2
    assert expanded.children[0].attr_name == "down_proj"
    assert expanded.children[0].input_source == "moe scores"
    assert expanded.children[1].attr_name == "merge"


def test_prepare_diagram_section_trees_expands_before_render():
    from visualizer.block_tree import BlockNode, prepare_diagram_section_trees

    mlp = BlockNode(
        attr_name="mlp",
        class_name="KimiMLP",
        role="ffn",
        label="KimiMLP",
        children=[
            BlockNode(attr_name="gate_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="up_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
        ],
    )
    prepared = prepare_diagram_section_trees([("KimiMLP", mlp)])
    assert len(prepared) == 1
    _, tree = prepared[0]
    assert [child.attr_name for child in tree.children] == ["gate_proj", "up_proj"]


def test_router_gate_expands_to_pipeline_steps():
    from visualizer.block_tree import BlockNode, collect_function_steps, inline_composite_steps

    gate = BlockNode(
        attr_name="gate",
        class_name="KimiMoEGate",
        role="router",
        label="Router",
        children=[
            BlockNode(attr_name="@op_l1_c0_linear", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="@op_l2_c0_sigmoid", class_name="Sigmoid", role="other", label="Sigmoid", is_basic=True),
            BlockNode(attr_name="@op_l3_c0_topk", class_name="TopK", role="other", label="TopK", is_basic=True),
        ],
    )
    expanded, wrapper = inline_composite_steps(gate)
    assert wrapper is gate
    assert [step.label for step in expanded] == ["Linear", "Sigmoid", "TopK"]


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

    analysis = analyze_source(
        code_path.read_text(),
        filename="modeling_kimi_linear.py",
        config=KIMI_ROUTER_CONFIG,
    )
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    moe = build_block_node(
        attr_name="block_sparse_moe",
        class_name="KimiSparseMoeBlock",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(moe)
    labels = [spec.label for spec in graph.nodes]
    assert labels[:10] == [
        "hidden_states",
        "Linear",
        "Sigmoid",
        "Add",
        "TopK",
        "Gather",
        "Sum",
        "Add",
        "Divide",
        "Multiply",
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

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import SYNTHETIC_INPUT, build_computation_graph

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(
        code_path.read_text(),
        filename="modeling_kimi_linear.py",
        config=KIMI_ROUTER_CONFIG,
    )
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
    multiply_id = next(
        op.attr_name
        for op in analysis.class_registry["KimiMoEGate"].forward_operations.values()
        if op.label == "Multiply"
    )
    route_scaling_index = next(
        index
        for index, spec in enumerate(graph.nodes)
        if spec.block and spec.block.attr_name == multiply_id
    )
    assert (route_scaling_index, agg_index) in graph.links
    assert (route_scaling_index, agg_index) not in graph.dashed_links
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
    from visualizer.computation_graph import SYNTHETIC_INPUT, build_computation_graph

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
        if spec.synthetic is None and spec.label in {"Add", "+"}
    )
    assert (input_index, shared_index) in graph.links
    assert (shared_index, plus_index) in graph.links
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
    assert "<!-- Output -->" in svg
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
    assert label == "KimiSparseMoeBlock / KimiMLP"
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
    assert "• 68 KimiDeltaAttention + KimiSparseMoeBlock" in label
    assert "• 24 KimiMLAAttention + KimiSparseMoeBlock" in label
    assert "• 1 KimiDeltaAttention + KimiMLP" in label
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

    # Kimi's MLA sets `self.rotary_emb = None` and asserts NoPE, so there is no
    # rotary module to draw and none may be invented from the config.
    assert [comp.attr_name for comp in spec.stack_pre] == ["embed_tokens"]
    assert spec.stack_pre[0].label == "Token Embedding"
    assert [comp.attr_name for comp in spec.stack_tail] == ["norm", "lm_head"]
    assert spec.stack_tail[0].label == "RMSNorm"
    assert spec.stack_tail[1].label == "Linear"


def test_positional_encoding_follows_code_not_config():
    """A config carrying rope parameters cannot claim rope the code never applies."""
    from pathlib import Path
    from visualizer.ast_analyze import analyze_source
    from visualizer.extract import parse_architecture

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    spec = parse_architecture({"rope_theta": 50000, "max_position_embeddings": 4096}, "Kimi", code_analysis=analysis)

    assert spec.positional_encoding == "NoPE"
    assert not any(note.startswith("RoPE theta=") for note in spec.attention_notes)


def test_module_level_rope_calls_are_traced_into_the_calling_block():
    """Rope applied by a plain function shows up where the forward applies it."""
    from visualizer.ast_analyze import analyze_source, is_positional_synthetic, positional_display_label

    source = '''
import torch
from torch import nn


def apply_rotary_emb(x, freqs_cis, inverse=False):
    return x


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wq = nn.Linear(8, 8)
        self.wk = nn.Linear(8, 8)
        self.wo = nn.Linear(8, 8)

    def forward(self, x, freqs_cis):
        q = self.wq(x)
        apply_rotary_emb(q, freqs_cis)
        k = self.wk(x)
        apply_rotary_emb(k, freqs_cis)
        o = q + k
        apply_rotary_emb(o, freqs_cis, True)
        return self.wo(o)
'''
    analysis = analyze_source(source, filename="modeling_probe.py")
    attention = analysis.class_registry["Attention"]
    rope_steps = [call for call in attention.forward_calls if is_positional_synthetic(call)]

    # One step per application site: queries, keys, and the inverse on the output.
    assert len(rope_steps) == 3
    assert all(positional_display_label(step) == "Apply rotary emb" for step in rope_steps)
    assert attention.forward_calls.index(rope_steps[0]) > attention.forward_calls.index("wq")
    assert attention.forward_step_details[rope_steps[-1]] == ["inverse rotation"]
    assert analysis.positional_helpers == ["apply_rotary_emb"]


def test_stack_owner_found_by_structure_when_name_is_plain():
    """Inference repos name the stack `Transformer`; its embedding identifies it."""
    from visualizer.ast_analyze import analyze_source

    source = '''
from torch import nn


class ParallelEmbedding(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.weight = nn.Parameter(1)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(1)


class ParallelHead(nn.Module):
    def __init__(self, dim, vocab):
        super().__init__()
        self.weight = nn.Parameter(1)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(8)
        self.attn = nn.Linear(8, 8)

    def forward(self, x):
        return self.attn(self.attn_norm(x))


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = ParallelEmbedding(8, 8)
        self.norm = RMSNorm(8)
        self.head = ParallelHead(8, 8)

    def forward(self, tokens):
        return self.head(self.norm(self.embed(tokens)))
'''
    analysis = analyze_source(source, filename="model.py")

    assert analysis.model_class == "Transformer"
    assert [comp.attr_name for comp in analysis.stack_pre] == ["embed"]
    assert [comp.attr_name for comp in analysis.stack_tail] == ["norm", "head"]


def test_spine_without_modeling_source_comes_from_config_and_says_so():
    """Config-only diagrams show the spine the config implies, flagged as inferred."""
    from visualizer.blocks import BlockComponent
    from visualizer.extract import ArchitectureSpec
    from visualizer.render import (
        _spine_box_style,
        _spine_sublabel,
        _stack_pre_components,
        _stack_tail_components,
    )

    spec = ArchitectureSpec(
        name="Config only",
        model_type="llama",
        vocab_size=151_936,
        norm_type="RMSNorm",
    )

    pre = _stack_pre_components(spec)
    tail = _stack_tail_components(spec)
    assert [comp.role for comp in pre] == ["embedding"]
    assert [comp.role for comp in tail] == ["norm", "head"]
    for comp in pre + tail:
        assert comp.inferred_from_config
        assert _spine_sublabel(comp) == "from config"
        assert _spine_box_style(comp)["linestyle"] == "dashed"

    # Nothing is invented for facts the config never states.
    bare = ArchitectureSpec(name="Bare", model_type="llama", norm_type="")
    assert _stack_pre_components(bare) == []
    assert _stack_tail_components(bare) == []


def test_source_declared_spine_is_never_marked_inferred():
    """A spine read from the modeling source keeps its own tiles and solid styling."""
    from visualizer.blocks import BlockComponent
    from visualizer.extract import ArchitectureSpec
    from visualizer.render import _spine_box_style, _spine_sublabel, _stack_pre_components

    embedding = BlockComponent(
        attr_name="embed",
        class_name="ParallelEmbedding",
        role="embedding",
        label="ParallelEmbedding",
        forward_order=0,
    )
    spec = ArchitectureSpec(
        name="From source",
        model_type="deepseek_v3",
        vocab_size=129_280,
        stack_pre=[embedding],
    )

    assert _stack_pre_components(spec) == [embedding]
    assert _spine_sublabel(embedding) is None
    assert _spine_box_style(embedding) == {"edgecolor": "#000000"}


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

    # A stack component with no parsed source stays one tile; children are never invented.
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
    assert positional.children == []
    assert not is_straight_line_module(positional)


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
    assert frame.label == "MLP", "a frame around a whole module is named by its class"
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

    analysis = analyze_source(
        code_path.read_text(),
        filename="modeling_kimi_linear.py",
        config=KIMI_ROUTER_CONFIG,
    )
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
        upper_fanout = len([target for source, target in graph.links if source == upper]) > 1
        lower_join = len([source for source, target in graph.links if target == lower]) > 1
        if upper_fanout or lower_join:
            assert gap >= min_gap
        else:
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

    analysis = analyze_source(
        code_path.read_text(),
        filename="modeling_kimi_linear.py",
        config=KIMI_ROUTER_CONFIG,
    )
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
    assert 0.9 <= span <= 16.0, f"MoE horizontal span {span:.3f} outside shrink-wrap range"


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

    analysis = analyze_source(
        code_path.read_text(),
        filename="modeling_kimi_linear.py",
        config=KIMI_ROUTER_CONFIG,
    )
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
    route_index = next(
        i
        for i, spec in enumerate(graph.nodes)
        if spec.label == "Multiply"
        and spec.block is not None
        and spec.block.attr_name.startswith("@op_")
    )
    combine_index = next(i for i, spec in enumerate(graph.nodes) if spec.label == "MoE aggregation")
    gap = positions[route_index].bottom - positions[combine_index].top_y
    assert gap <= 2 * DETAIL_LAYER_GAP + 0.02, (
        f"Multiply -> MoE aggregation gap {gap:.3f} exceeds layer gap {DETAIL_LAYER_GAP:.3f}"
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
        assert content_right - content_left < 8.5, (
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
                assert abs(points[-1][1] - anchors[tgt_idx].top) < 0.15, (
                    f"{port_label} connector should enter through the target top"
                )
                assert len(points) >= 4, f"{port_label} connector should route around obstacles"

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
            from visualizer.computation_graph import (
                _inline_frame_column_skip_links,
                _ordered_inline_frame_chain,
            )
            if any(
                (src, tgt) in _inline_frame_column_skip_links(graph, frame)
                for frame in graph.inline_frames
            ):

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
            i
            for i in g_frame.node_indices
            if graph.nodes[i].label in {"Multiply", "×"}
        )
        exp_idx = next(i for i in g_frame.node_indices if graph.nodes[i].label == "Exp")
        softplus_idx = next(i for i in g_frame.node_indices if graph.nodes[i].label == "Softplus")
        assert (softplus_idx, multiply_idx) in graph.links
        assert (exp_idx, multiply_idx) in graph.links

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
        assert abs(exp_to_mul[0][1] - anchors[exp_idx].bottom) < 1e-6
        assert abs(exp_to_mul[-1][1] - anchors[multiply_idx].top) < 1e-6
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
    assert "×" in labels or "Multiply" in labels
    frame = next(frame for frame in graph.inline_frames if frame.frame_id == "act_fn")
    assert frame.sublabel is None
    mul_index = next(
        i for i, spec in enumerate(graph.nodes) if spec.label in {"Multiply", "×"}
    )
    up_index = next(i for i, spec in enumerate(graph.nodes) if spec.block and spec.block.attr_name == "up_proj")
    situ_index = next(i for i, spec in enumerate(graph.nodes) if spec.label == "Situ")
    assert (situ_index, mul_index) in graph.links
    assert (up_index, mul_index) in graph.links
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
    mul = next(i for i, node in enumerate(graph.nodes) if node.label in {"Multiply", "×"})

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

    analysis = analyze_source(
        code_path.read_text(),
        filename="modeling_kimi_linear.py",
        config=KIMI_ROUTER_CONFIG,
    )
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
    assert moe_labels[:10] == [
        "hidden_states",
        "Linear",
        "Sigmoid",
        "Add",
        "TopK",
        "Gather",
        "Sum",
        "Add",
        "Divide",
        "Multiply",
    ]
    assert "MoE aggregation" in moe_labels
    assert "×" in moe_labels
    assert "Situ" in moe_labels
    gate_frame = next(frame for frame in moe_graph.inline_frames if frame.frame_id == "gate")
    assert gate_frame.label == "KimiMoEGate"
    assert len(gate_frame.node_indices) == 9
    shared_frame = next(frame for frame in moe_graph.inline_frames if frame.frame_id == "shared_experts")
    assert shared_frame.label == "KimiMLP"
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
    assert "Multiply" in labels
    assert "Sigmoid" not in labels
    assert not any(frame.frame_id == "g_proj" for frame in graph.inline_frames)
    assert not any(spec.label == "Output gate" for spec in graph.nodes)

    gate_producer_indices = [
        i
        for i, spec in enumerate(graph.nodes)
        if spec.block and spec.block.attr_name == "g_proj"
    ]
    assert gate_producer_indices
    combine_index = next(i for i, spec in enumerate(graph.nodes) if spec.label == "Multiply")
    assert graph.nodes[combine_index].sublabel in (None, "")
    gate_inputs_to_combine = [
        src
        for src, dst in graph.links
        if dst == combine_index and src in gate_producer_indices
    ]
    assert gate_inputs_to_combine
    assert not any((src, combine_index) in graph.dashed_links for src in gate_inputs_to_combine)


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
        index for index, pos in enumerate(positions) if pos.spec.label == "Multiply"
    )
    combine = positions[combine_index].cx
    spine.append(combine)
    assert len(spine) == 4
    assert max(spine) - min(spine) < 0.02
    combine_tile = next(node for node, _ in plan.node_draws if node.label == "Multiply")
    assert abs(combine_tile.cx - combine) < 0.02
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


def test_fanout_exit_feeders_pack_to_outside():
    """Input branches feeding the exit path pack outside; closer consumers sit further out."""
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
        _estimate_graph_height,
        _fanout_branch_node_groups,
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

    def branch_centers(block_class: str) -> dict[str, float]:
        attn = build_block_node(
            attr_name="self_attn",
            class_name=block_class,
            registry=analysis.class_registry,
            basic_ops=basic,
        )
        graph = build_computation_graph(attn)
        fig, ax = plt.subplots(figsize=(16, 13))
        try:
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
            centers: dict[str, float] = {}
            for branch_index, indices in _fanout_branch_node_groups(positions).items():
                labels = [
                    graph.nodes[index].block.attr_name
                    if graph.nodes[index].block
                    else graph.nodes[index].label
                    for index in indices
                ]
                label = labels[0]
                if any(name == "@functional_pad" for name in labels):
                    label = "pad_branch"
                elif label in centers:
                    label = f"{label}_{branch_index}"
                centers[label] = sum(positions[index].cx for index in indices) / len(indices)
            return centers
        finally:
            plt.close(fig)

    kda = branch_centers("KimiDeltaAttention")
    kda_cx = list(kda.values())
    assert abs(kda["b_proj"] - min(kda_cx)) < 0.08, (
        f"b_proj should be the leftmost exit feeder, got cx={kda['b_proj']:.3f}"
    )

    mla = branch_centers("KimiMLAAttention")
    mla_cx = list(mla.values())
    assert abs(mla["pad_branch"] - max(mla_cx)) < 0.08, (
        f"Pad branch should be the rightmost exit feeder, got cx={mla['pad_branch']:.3f}"
    )


def test_kda_hidden_states_fanout_nests_each_leg_on_its_own_row():
    """Parallel linear feeds from hidden_states each get their own exit port and row."""
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
    import itertools

    from visualizer.render import (
        COLORS,
        CONNECTOR_EXIT_STUB,
        DIAGRAM_LEFT_MARGIN,
        PARALLEL_CONNECTOR_CHANNEL_GAP,
        PARALLEL_CONNECTOR_COORD_EPS,
        TOP_ENTRY_PORT_GAP,
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
        source = anchors[input_index]
        rows: list[tuple[float, float, float]] = []
        for src, tgt in fanout_links:
            target = anchors[tgt]
            points = link_paths[(src, tgt)]
            end_x, end_y = points[-1]
            assert target.left <= end_x <= target.right
            assert abs(end_y - _connector_target_top_entry_y(target)) < 0.02
            assert source.left <= points[0][0] <= source.right
            rows.extend(
                (y1, min(x1, x2), max(x1, x2))
                for (x1, y1), (x2, y2) in zip(points, points[1:])
                if abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS
                and abs(x1 - x2) > PARALLEL_CONNECTOR_COORD_EPS
            )
        # Every leg turns below the input, and no two of their runs are drawn along the same
        # row: doubled up they would read as a single line that appears to stop where the
        # legs part.
        assert all(row < source.bottom + 1e-6 for row, _lo, _hi in rows)
        for (row_a, lo_a, hi_a), (row_b, lo_b, hi_b) in itertools.combinations(rows, 2):
            if abs(row_a - row_b) > PARALLEL_CONNECTOR_COORD_EPS:
                continue
            assert min(hi_a, hi_b) - max(lo_a, lo_b) <= PARALLEL_CONNECTOR_COORD_EPS, (
                f"runs at {row_a:.4f} and {row_b:.4f} are drawn on top of each other"
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
        # A fan-out parting company is a split, not a join, so the input's own exit carries
        # no dot however many legs leave it.
        assert not any(
            abs(y - source.bottom) < CONNECTOR_EXIT_STUB for _x, y in join_points
        ), f"fan-out exits must stay undotted, got {sorted(join_points)}"

        g_index = next(
            index for index, node in enumerate(graph.nodes) if node.block and node.block.attr_name == "g_proj"
        )
        combine_index = next(
            index for index, node in enumerate(graph.nodes) if node.label == "Multiply"
        )
        # The gate operand enters the combine tile through its own top port,
        # beside the centered main feed.
        operand_points = link_paths[(g_index, combine_index)]
        operand_x, operand_y = operand_points[-1]
        assert operand_y == pytest.approx(
            _connector_target_top_entry_y(anchors[combine_index]), abs=0.02
        )
        assert operand_x >= positions[combine_index].cx + TOP_ENTRY_PORT_GAP - 0.02
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
    residual_label = "+" if "+" in by_label else "Add"
    up_index = next(
        index
        for index, spec in enumerate(graph.nodes)
        if spec.block and spec.block.attr_name == "routed_expert_up_proj"
    )
    spine = [
        positions[index].cx
        for index in (by_label["MoE aggregation"], up_index, by_label[residual_label])
    ]
    draw = {node.label: node.cx for node, _ in plan.node_draws}
    assert max(spine) - min(spine) < 0.02
    assert abs(positions[by_label["MoE aggregation"]].cx - spine[0]) < 0.02
    assert abs(draw[residual_label] - positions[by_label[residual_label]].cx) < 0.02
    assert abs(anchors[by_label[residual_label]].cx - positions[by_label[residual_label]].cx) < 0.02
    gap = positions[up_index].bottom - positions[by_label[residual_label]].top_y
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
    assert "Multiply" in {spec.label for spec in graph.nodes}
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


def test_input_fanout_into_inline_frame_docks_on_the_tile_without_crossing_siblings():
    """A framed branch tees off the shared input bus rather than crossing it.

    Feeds into the topmost tile of a dotted frame used to be lifted into a
    corridor above the frame even when they already ran along the input's shared
    fan-out bus. That detour dropped back down through the bus its sibling
    branches still followed, and it aimed at the frame envelope rather than the
    tile, so the wire ended on empty frame background.
    """
    from collections import defaultdict
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.computation_graph import (
        SYNTHETIC_INPUT,
        _estimate_graph_height,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import (
        COLORS,
        DIAGRAM_LEFT_MARGIN,
        TOP_ENTRY_PORT_GAP,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _collect_detail_link_paths,
        _compute_detail_connector_buses,
        _connector_target_top_entry_y,
        _detail_sections_to_render,
        _inline_frame_for_top_member,
    )
    from visualizer.render_validate import finalize_detail_layout

    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    spec = load_architecture("moonshotai/Kimi-K3", detailed=True)
    tree = next(
        tree
        for title, tree, _sub in _detail_sections_to_render(spec)
        if title.startswith("KimiMLAAttention")
    )
    graph = build_computation_graph(tree, include_input=True)
    input_index = next(
        i for i, node in enumerate(graph.nodes) if node.synthetic == SYNTHETIC_INPUT
    )
    framed_targets = [
        tgt
        for src, tgt in graph.links
        if src == input_index and _inline_frame_for_top_member(graph, tgt) is not None
    ]
    assert framed_targets, "expected an input feed into a dotted frame's top tile"

    cx, top_y = 3.5, 10.0
    fig, ax = plt.subplots(figsize=(16, 13))
    try:
        measure_graph_node_sizes(ax, graph)
        min_left = DIAGRAM_LEFT_MARGIN + 0.05
        positions, links = layout_computation_graph(
            graph,
            cx=cx,
            top_y=top_y,
            block_w=18.0,
            block_h=_estimate_graph_height(graph),
            content_left=min_left,
        )
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=cx,
            top_y=top_y,
            detail_fill=COLORS["detail_fill"],
            min_left=min_left,
        )
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
        # The renderer builds frame-aware anchors, so the test has to as well.
        anchors = _anchors_from_detail_plan(positions, plan, graph)
        incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
        outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for src, tgt in links:
            incoming[tgt].append((src, tgt))
            outgoing[src].append((src, tgt))
        target_bus, source_bus, merge_entry_x, merge_link_bus = (
            _compute_detail_connector_buses(
                graph, positions, anchors, incoming, outgoing, plan.label_obstacles
            )
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
    finally:
        plt.close(fig)

    for tgt in framed_targets:
        tile = positions[tgt]
        end_x, end_y = link_paths[(input_index, tgt)][-1]
        assert abs(end_x - tile.cx) <= TOP_ENTRY_PORT_GAP, (
            f"feed into framed tile {tgt} docks at x={end_x:.3f}, "
            f"away from the tile centre {tile.cx:.3f}"
        )
        assert end_x >= tile.cx - tile.width / 2 and end_x <= tile.cx + tile.width / 2
        assert abs(end_y - _connector_target_top_entry_y(anchors[tgt])) <= 0.02

    def segments(points):
        return list(zip(points, points[1:]))

    keys = sorted(link_paths)
    crossings = []
    for first in range(len(keys)):
        for second in range(first + 1, len(keys)):
            for seg_a in segments(link_paths[keys[first]]):
                for seg_b in segments(link_paths[keys[second]]):
                    hit = _connector_segment_crossing(seg_a, seg_b)
                    if hit is not None:
                        crossings.append((keys[first], keys[second], hit))
    assert not crossings, "connectors cross: " + ", ".join(
        f"{a}x{b}@({hit[0]:.3f},{hit[1]:.3f})" for a, b, hit in crossings[:4]
    )
