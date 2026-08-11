"""Tests for TraceLens Visualizer (CPU-only)."""

from pathlib import Path

import pytest

from visualizer.ast_analyze import analyze_source, dump_ast
from visualizer.extract import load_architecture, parse_architecture
from visualizer.render import (
    COLORS,
    MERGE_CLEARANCE,
    MERGE_OUTPUT_GAP,
    MERGE_RADIUS,
    RESIDUAL_BRANCH_LIFT,
    _collect_sublayer_pairs,
    _make_node,
    _merge_y_for_module,
    _ordered_block_components,
    _residual_branch_y,
    _residual_merge,
    render_diagram,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_merge_node_sits_below_module_box():
    module_bottom = 4.0
    merge_y = _merge_y_for_module(module_bottom)
    merge_top = merge_y + MERGE_RADIUS
    assert merge_top <= module_bottom - MERGE_CLEARANCE
    merge_connector_top = merge_y + MERGE_RADIUS + MERGE_CLEARANCE
    assert merge_connector_top <= module_bottom - MERGE_OUTPUT_GAP


def test_repeat_label_clears_positional_and_routes_around_bbox():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.render import (
        BLOCK_FRAME_LABEL_PAD_X,
        BLOCK_FRAME_REPEAT_CONNECTOR_MARGIN,
        BLOCK_FRAME_REPEAT_LABEL_GAP,
        BLOCK_FRAME_DECODER_FRAME_GAP,
        BLOCK_FRAME_DECODER_OUTSIDE_GAP,
        BLOCK_FRAME_REPEAT_OUTSIDE_GAP,
        DIAGRAM_LEFT_MARGIN,
        FRAME_PATCH_TOP_OUTSET,
        STACK_BOX_BOTTOM_OUTSET,
        _block_frame_top,
        _block_top_below_repeat_label,
        _block_width_for_repeat_label,
        _decoder_label_bbox,
        _effective_repeat_outside_gap,
        _main_block_width,
        _outside_block_labels_bbox,
        _repeat_label_bbox,
        _text_size_in_axes,
    )
    from visualizer.sizing import FRAME_LABEL_PAD_X, box_width_for_text_width

    fig, ax = plt.subplots(figsize=(11, 13))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13)
    fig.canvas.draw()
    above_bottom = 10.195
    repeat_label = "93 × Transformer block"
    decoder_label = "KimiDecoderLayer"
    block_w = _main_block_width(
        ax,
        repeat_label=repeat_label,
        decoder_label=decoder_label,
        inner_w=3.0,
    )
    repeat_w = _block_width_for_repeat_label(ax, repeat_label)
    assert block_w >= repeat_w
    text_w, _ = _text_size_in_axes(ax, repeat_label, fontsize=10.0, fontweight="bold")
    assert repeat_w == pytest.approx(2 * box_width_for_text_width(text_w, pad_x=FRAME_LABEL_PAD_X))

    cx = DIAGRAM_LEFT_MARGIN + block_w / 2
    block_top = _block_top_below_repeat_label(
        ax,
        cx=cx,
        block_w=block_w,
        above_bottom=above_bottom,
        repeat_label=repeat_label,
        decoder_label=decoder_label,
    )
    frame_top = _block_frame_top(block_top)
    text_x = cx - block_w / 2 + BLOCK_FRAME_LABEL_PAD_X
    outside_gap = _effective_repeat_outside_gap(ax, repeat_label, decoder_label)
    outside_bb = _outside_block_labels_bbox(
        ax,
        text_x,
        frame_top,
        repeat_label,
        decoder_label,
    )
    repeat_bb = _repeat_label_bbox(
        ax,
        text_x,
        frame_top + outside_gap,
        repeat_label,
    )
    decoder_bb = _decoder_label_bbox(
        ax,
        text_x,
        repeat_bb.y0 - BLOCK_FRAME_DECODER_OUTSIDE_GAP,
        decoder_label,
        va="top",
    )

    assert outside_bb.y1 <= above_bottom - STACK_BOX_BOTTOM_OUTSET - BLOCK_FRAME_REPEAT_LABEL_GAP
    assert decoder_bb.y1 <= repeat_bb.y0 - BLOCK_FRAME_DECODER_OUTSIDE_GAP + 1e-6
    assert decoder_bb.y0 >= frame_top + FRAME_PATCH_TOP_OUTSET + BLOCK_FRAME_DECODER_FRAME_GAP - 1e-6
    frame_left = cx - block_w / 2
    assert repeat_bb.x0 >= frame_left + BLOCK_FRAME_LABEL_PAD_X - 1e-6
    label_w = repeat_bb.x1 - repeat_bb.x0
    assert repeat_bb.x1 <= frame_left + BLOCK_FRAME_LABEL_PAD_X + label_w + 1e-6
    assert outside_bb.x1 + BLOCK_FRAME_REPEAT_CONNECTOR_MARGIN <= cx


def test_residual_merge_side_entry_uses_merge_center():
    from unittest.mock import patch

    calls: list[tuple] = []

    def record_line(ax, x1, y1, x2, y2, **kwargs):
        calls.append(("line", x1, y1, x2, y2, kwargs.get("linestyle")))

    def record_arrow(ax, x1, y1, x2, y2, **kwargs):
        calls.append(("arrow", x1, y1, x2, y2, kwargs.get("linestyle")))

    module_bottom = 5.0
    merge_y = _merge_y_for_module(module_bottom)
    spine_x = 3.0
    branch_x = 1.0

    with patch("visualizer.render._line", side_effect=record_line), patch(
        "visualizer.render._arrow", side_effect=record_arrow
    ), patch("visualizer.render._draw_path"), patch("visualizer.render._draw_merge"):
        _residual_merge(
            None,
            module_cx=spine_x,
            module_bottom=module_bottom,
            skip_from_y=6.0,
            spine_x=spine_x,
            branch_x=branch_x,
        )

    dashed_arrows = [call for call in calls if call[0] == "arrow" and call[5] == "--"]
    assert len(dashed_arrows) == 1
    _, x1, y1, x2, y2, _ = dashed_arrows[0]
    assert y1 == merge_y
    assert y2 == merge_y
    assert x2 == spine_x - MERGE_RADIUS
    assert x1 < x2


def test_residual_branch_routes_above_norm():
    norm = _make_node("norm", 5.0, 10.0, 1.35, 0.32, "RMSNorm", COLORS["norm"], text_color=COLORS["text"])
    branch_y = _residual_branch_y(norm.top)
    assert branch_y > norm.top
    assert branch_y - norm.top >= RESIDUAL_BRANCH_LIFT - 1e-6


def test_converging_connectors_share_target_bus():
    from visualizer.render import (
        _RenderAnchor,
        _compute_shared_target_bus_y,
        _orthogonal_path,
    )

    sources = [
        _RenderAnchor(cx=2.0, top=8.0, bottom=7.5, left=1.5, right=2.5),
        _RenderAnchor(cx=4.0, top=8.0, bottom=7.5, left=3.5, right=4.5),
        _RenderAnchor(cx=6.0, top=8.0, bottom=7.5, left=5.5, right=6.5),
    ]
    target = _RenderAnchor(cx=4.0, top=6.0, bottom=5.5, left=3.5, right=4.5)
    bus_y = _compute_shared_target_bus_y(sources, target, obstacles=[])

    for source in sources:
        points = _orthogonal_path(source, target, [], bus_y=bus_y)
        assert points[-2][0] == target.cx
        assert points[-2][1] == bus_y
        assert points[-1] == (target.cx, target.top + 0.04)


def test_fanout_connectors_share_source_bus():
    from visualizer.render import (
        _RenderAnchor,
        _compute_shared_source_bus_y,
        _orthogonal_path,
    )

    source = _RenderAnchor(cx=4.0, top=8.0, bottom=7.5, left=3.5, right=4.5)
    targets = [
        _RenderAnchor(cx=2.0, top=6.0, bottom=5.5, left=1.5, right=2.5),
        _RenderAnchor(cx=4.0, top=6.0, bottom=5.5, left=3.5, right=4.5),
        _RenderAnchor(cx=6.0, top=6.0, bottom=5.5, left=5.5, right=6.5),
    ]
    bus_y = _compute_shared_source_bus_y(source, targets, obstacles=[])

    for target in targets:
        points = _orthogonal_path(source, target, [], bus_near="source", bus_y=bus_y)
        assert points[1] == (source.cx, bus_y)
        if abs(source.cx - target.cx) < 0.06:
            assert points[-1][0] == source.cx
        else:
            assert points[2][1] == bus_y
            assert points[-2][0] == target.cx


def test_min_vertical_block_gap_matches_top_text_inset():
    from visualizer.sizing import BLOCK_PAD_Y, TITLE_LINE_H, min_vertical_block_gap

    assert min_vertical_block_gap() >= BLOCK_PAD_Y
    assert min_vertical_block_gap() == BLOCK_PAD_Y + TITLE_LINE_H / 2


def test_parse_llama_like_config():
    config_path = FIXTURES / "llama_like" / "config.json"
    spec = load_architecture(config_path, name="Test Llama", analyze_code=False)
    assert spec.attention_type == "GQA"
    assert spec.decoder_type == "Dense"
    assert spec.ffn_type == "SwiGLU"
    assert spec.num_hidden_layers == 16
    assert spec.kv_cache_per_token_bf16 is not None


def test_parse_moe_config():
    config = {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "vocab_size": 151936,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "hidden_act": "silu",
    }
    spec = parse_architecture(config, "test", name="Qwen3 MoE")
    assert spec.decoder_type == "Sparse MoE"
    assert spec.num_experts == 128
    assert spec.active_params_hint is not None


def test_render_diagram(tmp_path: Path):
    config_path = FIXTURES / "llama_like" / "config.json"
    spec = load_architecture(config_path, name="Test Llama", analyze_code=False)
    out = render_diagram(spec, tmp_path / "diagram.svg")
    assert out.exists()
    assert out.suffix == ".svg"
    assert out.stat().st_size > 1_000


def test_stroke_white_text_in_svg():
    from visualizer.render import _stroke_white_text_in_svg

    svg = (
        '<g style="fill: #ffffff" transform="translate(1 2) scale(0.076 -0.076)">'
        '<use xlink:href="#A"/>'
        '<use xlink:href="#B" transform="translate(10 0)"/>'
        "</g>"
        '<path style="fill: #ffffff; stroke: #d0d0d0"/>'
    )
    stroked = _stroke_white_text_in_svg(svg)
    assert 'stroke-width: 39.4737' in stroked
    assert stroked.count('stroke: #000000') == 2
    assert 'style="fill: #ffffff; stroke: #d0d0d0"' in stroked


def test_ast_custom_decoder_layer():
    source = (FIXTURES / "custom_model" / "modeling_custom.py").read_text(encoding="utf-8")
    analysis = analyze_source(source, filename="modeling_custom.py")

    assert analysis.decoder_class == "CustomDecoderLayer"
    assert analysis.attention_type == "MLA"
    assert analysis.decoder_type == "Sparse MoE"
    assert analysis.forward_sequence == [
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "block_sparse_moe",
    ]
    assert "CustomSharedExpertMoE" in {comp.class_name for comp in analysis.block_components}


def test_load_architecture_with_local_modeling(tmp_path: Path):
    fixture_dir = FIXTURES / "custom_model"
    spec = load_architecture(fixture_dir, name="Custom MLA MoE")

    assert spec.decoder_class == "CustomDecoderLayer"
    assert spec.attention_type == "MLA"
    assert spec.decoder_type == "Sparse MoE"
    assert len(spec.block_components) >= 4
    assert spec.forward_sequence[1] == "self_attn"

    out = render_diagram(spec, tmp_path / "custom.png")
    assert out.exists()


def test_dump_ast_contains_decoder_class():
    source = (FIXTURES / "custom_model" / "modeling_custom.py").read_text(encoding="utf-8")
    ast_dump = dump_ast(source, filename="modeling_custom.py")
    assert "CustomDecoderLayer" in ast_dump
    assert "CustomLatentAttention" in ast_dump


def _svg_patch_y_range(svg_text: str, style: str) -> tuple[float, float]:
    import re

    pattern = rf'<path d="([^"]+)"[^>]*style="[^"]*{re.escape(style)}'
    match = re.search(pattern, svg_text, flags=re.DOTALL)
    assert match is not None, f"Missing SVG patch with style {style!r}"
    ys = [float(y) for y in re.findall(r"L [\d.]+ ([\d.]+)", match.group(1))]
    assert ys, f"Could not parse SVG path coordinates for style {style!r}"
    return min(ys), max(ys)


def _svg_patch_x_range(svg_text: str, style: str) -> tuple[float, float]:
    import re

    pattern = rf'<path d="([^"]+)"[^>]*style="[^"]*{re.escape(style)}'
    match = re.search(pattern, svg_text, flags=re.DOTALL)
    assert match is not None, f"Missing SVG patch with style {style!r}"
    coords = re.findall(r"[MLQ] ([\d.]+) ([\d.]+)", match.group(1))
    xs = [float(x) for x, _ in coords]
    assert xs, f"Could not parse SVG path x coordinates for style {style!r}"
    return min(xs), max(xs)


def test_fact_sheet_sits_to_the_right_of_transformer_block(tmp_path: Path):
    fixture_dir = FIXTURES / "custom_model"
    spec = load_architecture(fixture_dir, name="Custom MLA MoE")
    pairs = _collect_sublayer_pairs(_ordered_block_components(spec))
    assert len(pairs) >= 2

    out = render_diagram(spec, tmp_path / "custom.svg")
    svg = out.read_text(encoding="utf-8")
    block_left, block_right = _svg_patch_x_range(svg, "fill: #fff5f4; stroke: #c0392b")
    fact_left, _ = _svg_patch_x_range(svg, "fill: #ffffff; stroke: #d0d0d0")

    assert block_left < block_right
    assert fact_left > block_right
