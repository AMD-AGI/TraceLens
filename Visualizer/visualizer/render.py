"""Render Raschka-style LLM architecture block diagrams (CPU-only)."""

from __future__ import annotations

import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

from visualizer.blocks import BlockComponent
from visualizer.ast_analyze import displays_as_linear
from visualizer.basic_ops import BasicOpFilter
from visualizer.blocks import BlockComponent, collect_norm_module_pairs, input_sources_from_forward_sequence, upstream_input_sources
from visualizer.block_tree import (
    BlockNode,
    _is_composite_block,
    _omit_from_detailed_view,
    _show_single_function_in_diagram,
    build_stack_component_tree,
    collect_nested_diagrams,
    is_method_wrapper,
    is_simple_modeled_tile,
    is_single_function_tree,
    is_straight_line_module,
    spine_expanded_frame_label,
    tile_display_labels,
    tile_sublabel,
    wrapper_bullet,
    wrapper_bullet_lines,
    wrapper_module_comment,
)
from visualizer.sizing import (
    BLOCK_PAD_X,
    BLOCK_PAD_Y,
    FRAME_LABEL_PAD_X,
    LABEL_LINE_GAP,
    SUB_LINE_H,
    TITLE_LINE_H,
    block_sublabel,
    box_height_for_content,
    box_text_lines,
    box_width_for_text_width,
    estimate_block_size_for_node,
    BLOCK_PAD_X,
    BLOCK_PAD_Y,
    INPUT_PAD_X,
    INPUT_PAD_Y,
    TENSOR_PORT_PAD_Y,
    min_vertical_block_gap,
    single_line_box_height,
)
from visualizer.text_measure import ContentBounds
from visualizer.computation_graph import (
    SYNTHETIC_HIDDEN,
    SYNTHETIC_COMBINE,
    SYNTHETIC_INPUT,
    SYNTHETIC_MULTIPLY,
    SYNTHETIC_TENSOR,
    LayoutPosition,
    _estimate_graph_height,
    _node_content_left,
    _node_content_right,
    add_root_pipeline_frame,
    build_computation_graph,
    layout_computation_graph,
)
from visualizer.extract import ArchitectureSpec

COLORS = {
    "bg": "#e1e1e1",
    "text": "#1a1a1a",
    "muted": "#555555",
    "embed": "#d9e8f5",
    "pos": "#e8edf2",
    "block_border": "#c0392b",
    "block_fill": "#fff5f4",
    "attention": "#5dade2",
    "ffn": "#3a4550",
    "moe": "#8e44ad",
    "norm": "#f5b041",
    "residual": "#95a5a6",
    "flow": "#2c3e50",
    "head": "#d5dbdb",
    "fact_bg": "#ffffff",
    "fact_border": "#d0d0d0",
    "detail_border": "#566573",
    "detail_fill": "#f4f6f7",
    "basic_op": "#bdc3c7",
}

ROLE_COLORS = {
    "attention": COLORS["attention"],
    "moe": COLORS["moe"],
    "ffn": COLORS["ffn"],
    "norm": COLORS["norm"],
    "router": COLORS["moe"],
    "gate": COLORS["moe"],
    "other": COLORS["moe"],
}

_WHITE_TEXT_GROUP = re.compile(
    r'(<g style="fill: (#fff(?:fff)?)" transform="translate\([^"]+\) scale\(([-\d.]+)\s+([-\d.]+)\)">)'
    r"((?:(?!</g>).)*?)"
    r"(</g>)",
    re.DOTALL,
)
WHITE_TEXT_OUTLINE_PX = 2.0
_BASIC_OP_EDGE = "#000000"
_INPUT_NODE_EDGE = "#000000"


def white_text_has_black_outline_in_svg(svg: str) -> bool:
    """Return True when white diagram labels carry a black stroke in exported SVG."""
    use_groups = _WHITE_TEXT_GROUP.findall(svg)
    if not use_groups:
        return bool(
            re.search(
                rf"fill: #ffffff; stroke: #000000; stroke-width: {WHITE_TEXT_OUTLINE_PX:g}",
                svg,
            )
        )
    for _header, _fill, _sx, _sy, body, _closing in use_groups:
        for use_tag in re.findall(r"<use[^>]*/>", body):
            if "stroke: #000000" not in use_tag:
                return False
    return True


def _default_box_edgecolor(node: Node) -> str:
    """Pick a visible border; basic-op tiles use black instead of matching gray fill."""
    if node.facecolor == COLORS["basic_op"]:
        return _BASIC_OP_EDGE
    return node.facecolor


def _stroke_white_text_in_svg(svg: str) -> str:
    """Add a thick black outline to matplotlib's scaled SVG glyph groups."""
    svg = re.sub(
        r'(<g style=")fill: (#fff(?:fff)?); stroke: #000000; stroke-width: 1px; paint-order: stroke fill(")',
        r"\1fill: \2\3",
        svg,
    )

    def _patch_group(match: re.Match[str]) -> str:
        header, fill, sx, sy, body, closing = match.groups()
        if "<use " not in body:
            return match.group(0)
        scale = min(abs(float(sx)), abs(float(sy)))
        stroke_w = WHITE_TEXT_OUTLINE_PX / scale if scale > 0 else WHITE_TEXT_OUTLINE_PX
        use_style = (
            f'style="fill: {fill}; stroke: #000000; stroke-width: {stroke_w:.4f}; '
            f'paint-order: stroke fill" '
        )

        def _patch_use(use_match: re.Match[str]) -> str:
            tag = use_match.group(0)
            if "style=" in tag:
                return re.sub(r'style="[^"]*"', use_style.strip(), tag)
            return tag.replace("<use ", f"<use {use_style}", 1)

        patched_body = re.sub(r"<use ", _patch_use, body)
        return header + patched_body + closing

    return _WHITE_TEXT_GROUP.sub(_patch_group, svg)


def _finalize_svg_styling(svg: str) -> str:
    """Post-process SVG for readable labels and basic-op borders."""
    svg = _stroke_white_text_in_svg(svg)
    svg = re.sub(
        r"fill: #bdc3c7; stroke: #bdc3c7;",
        "fill: #bdc3c7; stroke: #000000;",
        svg,
    )
    svg = re.sub(
        r"fill: #e8edf2; stroke: #e8edf2;",
        "fill: #e8edf2; stroke: #000000;",
        svg,
    )
    return svg


def _detail_block_facecolor(block: BlockNode) -> str:
    """Face color for nodes drawn inside a detailed block-internals graph."""
    from visualizer.block_tree import is_basic_op_tile

    if is_basic_op_tile(block):
        return COLORS["basic_op"]
    return ROLE_COLORS.get(block.role, ROLE_COLORS["other"])


def _detail_tile_text_color(facecolor: str) -> str:
    """Gray basic-op tiles use dark text; colored tiles keep white labels."""
    return COLORS["text"] if facecolor == COLORS["basic_op"] else "white"

PANEL_W = 5.85
PANEL_WRAP_WIDTH = 64
PANEL_TITLE_FONT = 11
PANEL_TITLE_COLOR = COLORS["text"]
SECTION_TITLE_GAP = 0.10
SECTION_TITLE_HEIGHT = 0.16
BLOCK_FRAME_HEADER_HEIGHT = 0.52
BLOCK_FRAME_LABEL_PAD_X = FRAME_LABEL_PAD_X
BLOCK_FRAME_LABEL_PAD_Y = 0.10
BLOCK_FRAME_REPEAT_LINE_H = 0.20
BLOCK_FRAME_DECODER_LINE_H = 0.18
BLOCK_FRAME_CONTENT_GAP = 0.02
BLOCK_FRAME_REPEAT_OUTSIDE_GAP = 0.11
BLOCK_FRAME_DECODER_OUTSIDE_GAP = 0.04
BLOCK_FRAME_DECODER_FRAME_GAP = 0.05
BLOCK_FRAME_REPEAT_LABEL_GAP = 0.14
BLOCK_FRAME_REPEAT_CONNECTOR_MARGIN = 0.06
BLOCK_FRAME_BOTTOM_INSET = 0.06
# FancyBboxPatch pad/rounding extends the visible border beyond logical bounds.
FRAME_PATCH_TOP_OUTSET = 0.02
FRAME_PATCH_BOTTOM_OUTSET = 0.02
STACK_BOX_BOTTOM_OUTSET = 0.01
MAIN_BLOCK_W = 5.1
DIAGRAM_LEFT_MARGIN = 0.55
DIAGRAM_RIGHT_MARGIN = 0.45
DETAIL_MIN_BLOCK_W = 7.8
FACT_SHEET_GAP = 0.5  # ~1/2 inch in diagram coordinates (axes units match figure inches)
MEASURE_CANVAS_WIDTH = 13.0
PANEL_BODY_FONT = 8.3
PANEL_BODY_COLOR = COLORS["text"]
PANEL_PAD_X = 0.25
PANEL_TITLE_Y = 0.25
PANEL_PAD_TOP = 0.45
PANEL_LINE_HEIGHT = 0.19
PANEL_PAD_BOTTOM = 0.18

MERGE_RADIUS = 0.11
MERGE_CLEARANCE = 0.05
COMBINE_OP_SIZE = 2 * (MERGE_RADIUS + MERGE_CLEARANCE)
MERGE_OUTPUT_GAP = 0.06
RESIDUAL_BRANCH_LIFT = 0.07
FLOW_CONNECTOR_ZORDER = 2
DETAIL_CONNECTOR_ZORDER = 5.5
CONNECTOR_JUNCTION_DOT_RADIUS = 0.025
CONNECTOR_JUNCTION_HALO_RADIUS = 0.038
CONNECTOR_JUNCTION_ZORDER = 5.85
BUS_JUNCTION_Y_EPS = 0.01
PARALLEL_CONNECTOR_CHANNEL_GAP = 0.08
PARALLEL_CONNECTOR_COORD_EPS = 0.025
SHARED_CONNECTOR_BUS_MIN_LINKS = 4
SHARED_SOURCE_BUS_MIN_LINKS = 2
CONNECTOR_EXIT_STUB = 0.10
CONNECTOR_OBSTACLE_MARGIN = 0.06
CONNECTOR_ATTACHED_BOX_MARGIN = 0.01
CONNECTOR_SIDE_ENTRY_GAP = 0.06
FANOUT_SHORT_CHANNEL_MAX = 0.65
FANOUT_SHORT_TEE_FRACTION = 0.5
COMBINE_OP_ZORDER = 6


def _connector_target_top_entry_y(target: _RenderAnchor, *, gap: float = 0.04) -> float:
    """Y coordinate where a downward connector meets the target's top edge."""
    del gap  # connectors attach flush to the tile top border
    return target.top


def _connector_source_bottom_exit_y(source: _RenderAnchor, *, gap: float = 0.04) -> float:
    """Y coordinate where a connector leaves the source bottom edge."""
    del gap  # connectors attach flush to the tile bottom border
    return source.bottom


def _connector_target_side_entry_y(target: _RenderAnchor) -> float:
    """Y coordinate where a side-entry connector meets the target's left/right edge."""
    return (target.top + target.bottom) / 2


def _connector_min_bus_y_above_target(target: _RenderAnchor, *, gap: float = 0.04) -> float:
    """Keep a shared merge bus above a flush top entry and its obstacle margin."""
    return (
        _connector_target_top_entry_y(target, gap=gap)
        + CONNECTOR_OBSTACLE_MARGIN
        + CONNECTOR_ATTACHED_BOX_MARGIN
    )


def _min_bus_y_clearing_horizontal_corridor(
    x1: float,
    x2: float,
    obstacles: list[_RenderAnchor],
    *,
    proposed_y: float,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> float:
    """Raise a source-bus Y only when its horizontal leg would cut through a tile."""
    lo, hi = (x1, x2) if x1 <= x2 else (x2, x1)
    min_y = proposed_y
    for obstacle in obstacles:
        if obstacle.right + margin < lo or obstacle.left - margin > hi:
            continue
        block_bottom = obstacle.bottom - margin
        block_top = obstacle.top + margin + CONNECTOR_ATTACHED_BOX_MARGIN
        if proposed_y >= block_top or proposed_y <= block_bottom:
            continue
        min_y = max(min_y, block_top)
    return min_y

DETAIL_TILE_ROUNDING = 0.08
DETAIL_TILE_BOX_PAD = 0.008
DETAIL_FRAME_BOX_PAD = 0.01
DETAIL_FRAME_ROUNDING = 0.12
DETAIL_FRAME_STROKE = 0.035
DETAIL_FRAME_GAP = 0.025
DETAIL_FRAME_PAD_Y_EXTRA = 0.012
INLINE_FRAME_PAD = 0.10
INLINE_FRAME_CONNECTOR_GUTTER = 0.14
INLINE_FRAME_SIDE_ENTRY_EXTRA_GAP = 0.14
INLINE_FRAME_MULTI_BYPASS_EXTRA_GAP = 0.05
PIPELINE_MERGE_BUS_BELOW_FRAME_GAP = 0.06
INLINE_FRAME_LABEL_GAP = 0.04
INLINE_FRAME_LABEL_CHAR_W = 6.4 * 0.0078
INLINE_FRAME_LABEL_LINE_H = 0.11
# Extra space reserved above an expanded spine block for its dotted-frame label.
SPINE_EXPANDED_BLOCK_TOP_RESERVE = 0.44


@dataclass
class Node:
    """A positioned diagram block."""

    node_id: str
    x: float
    y: float
    w: float
    h: float
    label: str
    facecolor: str
    text_color: str = "white"
    sublabel: str | None = None
    fontsize: float = 9.0
    residual_merge: bool = False
    pad_x: float | None = None
    pad_y: float | None = None

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def top(self) -> float:
        return self.y + self.h

    @property
    def bottom(self) -> float:
        return self.y


@dataclass
class _RenderAnchor:
    cx: float
    top: float
    bottom: float
    left: float
    right: float


@dataclass(frozen=True)
class InlineFrameLabelLine:
    """One measured inline-frame caption line in diagram coordinates."""

    text: str
    x: float
    y: float
    ha: str = "left"
    va: str = "bottom"
    fontsize: float = 6.4
    fontweight: str = "normal"
    style: str | None = None


@dataclass
class InlineFrameLabelPlacement:
    """Resolved caption geometry for a dotted inline frame."""

    frame_id: str
    lines: list[InlineFrameLabelLine] = field(default_factory=list)
    side: str = "top"


@dataclass
class DetailDrawPlan:
    """Pre-rendered box/text descriptors measured before connectors are drawn."""

    input_sublabel: str | None
    node_draws: list[tuple[Node, dict[str, object]]] = field(default_factory=list)
    combine_ops: list[tuple[float, float, str, str | None]] = field(default_factory=list)
    branch_labels: list[tuple[str, float, float, str, str]] = field(default_factory=list)
    label_obstacles: list[_RenderAnchor] = field(default_factory=list)
    inline_frame_labels: dict[str, InlineFrameLabelPlacement] = field(default_factory=dict)


@dataclass
class DiagramLayout:
    nodes: list[Node] = field(default_factory=list)
    height: float = 13.0

    def add(self, node: Node) -> Node:
        self.nodes.append(node)
        return node


def _fit_spine_node_to_label(ax, node: Node) -> None:
    """Grow a main-spine tile so measured label text fits inside the box."""
    from visualizer.text_measure import box_label_size

    width, height = box_label_size(
        ax,
        node.label,
        node.sublabel,
        fontsize=node.fontsize,
        pad_x=node.pad_x,
        pad_y=node.pad_y,
    )
    top = node.top
    node.w = max(node.w, width)
    node.h = max(node.h, height)
    node.x = node.cx - node.w / 2
    node.y = top - node.h


def _draw_box(
    ax,
    node: Node,
    *,
    edgecolor: str | None = None,
    linestyle: str = "solid",
    zorder: float = 5,
) -> None:
    patch = FancyBboxPatch(
        (node.x, node.y),
        node.w,
        node.h,
        boxstyle="round,pad=0.01,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=edgecolor or _default_box_edgecolor(node),
        facecolor=node.facecolor,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    for line in box_text_lines(
        node.top,
        node.h,
        node.label,
        node.sublabel,
        pad_y=node.pad_y,
        title_fontsize=node.fontsize,
    ):
        ax.text(
            node.cx,
            line.y,
            line.text,
            ha="center",
            va=line.va,
            fontsize=line.fontsize,
            color=node.text_color,
            fontweight=line.fontweight,
            zorder=6,
        )


def _arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str | None = None,
    linewidth: float = 1.5,
    linestyle: str = "solid",
    zorder: float = FLOW_CONNECTOR_ZORDER,
) -> None:
    _line(
        ax,
        x1,
        y1,
        x2,
        y2,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )


def _line(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str | None = None,
    linewidth: float = 1.6,
    linestyle: str = "solid",
    zorder: float = FLOW_CONNECTOR_ZORDER,
) -> None:
    ax.plot(
        [x1, x2],
        [y1, y2],
        color=color or COLORS["flow"],
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
        solid_capstyle="butt",
        solid_joinstyle="miter",
    )


def _connect_down(ax, source: Node, target: Node, *, gap: float = 0.06) -> None:
    del gap
    _arrow(ax, source.cx, source.bottom, target.cx, target.top)


def _connect_from_point(ax, x: float, y: float, target: Node, *, gap: float = 0.06) -> None:
    del gap
    _arrow(ax, x, y, target.cx, target.top)


def _combine_uses_box_anchor(target: _RenderAnchor) -> bool:
    """True when a combine node was laid out as a labeled tile rather than a glyph circle."""
    return (target.top - target.bottom) > COMBINE_OP_SIZE * 0.75


def _draw_op_circle(ax, x: float, y: float, symbol: str) -> None:
    circle = Circle(
        (x, y),
        MERGE_RADIUS,
        facecolor=COLORS["bg"],
        edgecolor=COLORS["residual"],
        linewidth=1.6,
        zorder=COMBINE_OP_ZORDER,
    )
    ax.add_patch(circle)
    ax.text(
        x,
        y,
        symbol,
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["residual"],
        zorder=COMBINE_OP_ZORDER + 1,
    )


def _draw_merge(ax, x: float, y: float) -> None:
    _draw_op_circle(ax, x, y, "+")


def _draw_combine_op(
    ax,
    x: float,
    y: float,
    symbol: str,
    *,
    sublabel: str | None = None,
    width: float | None = None,
    height: float | None = None,
) -> None:
    from visualizer.ast_analyze import is_compact_combine_label

    if is_compact_combine_label(symbol):
        _draw_op_circle(ax, x, y, symbol)
        if sublabel:
            ax.text(
                x + MERGE_RADIUS + 0.10,
                y,
                sublabel,
                ha="left",
                va="center",
                fontsize=6.2,
                color=COLORS["muted"],
                zorder=6,
            )
        return

    from visualizer.text_measure import box_label_size

    fontsize = 6.8
    pad_x, pad_y = 0.08, 0.06
    if width is None or height is None:
        width, height = box_label_size(
            ax,
            symbol,
            sublabel,
            fontsize=fontsize,
            pad_x=pad_x,
            pad_y=pad_y,
            white_text_stroke_pad=False,
        )
    top_y = y + height / 2
    leaf = _make_node(
        f"combine:{symbol}",
        x,
        top_y,
        width,
        height,
        symbol,
        COLORS["bg"],
        text_color=COLORS["residual"],
        fontsize=fontsize,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    _draw_box(ax, leaf, edgecolor=COLORS["residual"])


def _is_combine_synthetic(synthetic: str | None) -> bool:
    return synthetic in {SYNTHETIC_COMBINE, SYNTHETIC_MULTIPLY}


def _combine_center_y(target: _RenderAnchor) -> float:
    return (target.top + target.bottom) / 2


def _combine_top_entry_y(target: _RenderAnchor) -> float:
    if _combine_uses_box_anchor(target):
        return target.top
    return _combine_center_y(target) + MERGE_RADIUS


def _combine_bottom_exit_y(source: _RenderAnchor) -> float:
    if _combine_uses_box_anchor(source):
        return source.bottom
    return _combine_center_y(source) - MERGE_RADIUS


def _merge_anchor(cx: float, merge_y: float) -> _RenderAnchor:
    pad = MERGE_RADIUS + MERGE_CLEARANCE
    return _RenderAnchor(cx=cx, top=merge_y + pad, bottom=merge_y - pad, left=cx - pad, right=cx + pad)


def _connect_from_merge(ax, merge_x: float, merge_y: float, target: Node, *, gap: float = 0.06) -> None:
    """Connect downward from a residual merge node to the next block."""
    del gap
    start_y = merge_y - MERGE_RADIUS - MERGE_CLEARANCE
    _arrow(ax, merge_x, start_y, target.cx, target.top)


def _merge_y_for_module(module_bottom: float) -> float:
    """Place merge node fully below the module box with clearance for connectors."""
    return module_bottom - MERGE_OUTPUT_GAP - MERGE_RADIUS - MERGE_CLEARANCE


def _residual_branch_y(skip_from_y: float) -> float:
    """Y level for the horizontal residual bypass, above the norm tile."""
    return skip_from_y + RESIDUAL_BRANCH_LIFT


def _residual_merge(
    ax,
    *,
    module_cx: float,
    module_bottom: float,
    skip_from_y: float,
    spine_x: float,
    branch_x: float,
) -> float:
    """Merge module output with the residual skip. Returns y of the merge node."""
    merge_y = _merge_y_for_module(module_bottom)
    merge_top = merge_y + MERGE_RADIUS + MERGE_CLEARANCE
    branch_y = _residual_branch_y(skip_from_y)
    bus_y = (module_bottom + merge_top) / 2
    merge_left = spine_x - MERGE_RADIUS

    # Residual bypass: route around the left side and enter the merge at its center.
    _line(ax, spine_x, skip_from_y, spine_x, branch_y, color=COLORS["flow"], linestyle="solid")
    _line(ax, spine_x, branch_y, branch_x, branch_y, color=COLORS["flow"], linestyle="solid")
    _line(ax, branch_x, branch_y, branch_x, merge_y, color=COLORS["flow"], linestyle="solid")
    _arrow(
        ax,
        branch_x,
        merge_y,
        merge_left,
        merge_y,
        color=COLORS["flow"],
        linewidth=1.5,
        linestyle="solid",
    )

    # Main path: share the same bus and vertical entry at the merge node.
    if abs(module_cx - spine_x) < 0.06:
        points = [(module_cx, module_bottom), (module_cx, bus_y), (spine_x, merge_top)]
    else:
        points = [
            (module_cx, module_bottom),
            (module_cx, bus_y),
            (spine_x, bus_y),
            (spine_x, merge_top),
        ]
    _draw_path(ax, points, color=COLORS["flow"])
    _draw_connector_junction_dots(
        ax,
        {
            (0, 1): [
                (spine_x, skip_from_y),
                (spine_x, branch_y),
                (branch_x, branch_y),
                (branch_x, merge_y),
            ],
            (1, 2): points,
        },
        obstacles=[],
        combine_ops=[(spine_x, merge_y)],
    )

    _draw_merge(ax, spine_x, merge_y)
    return merge_y


def _residual_branch_x(cx: float, block_w: float, *, inset: float = 0.28) -> float:
    """X coordinate for the residual bypass, inset from the block frame's left edge."""
    return cx - block_w / 2 + inset


def _attention_label(spec: ArchitectureSpec) -> str:
    if spec.layer_variants:
        labels: list[str] = []
        for variant in spec.layer_variants:
            if variant.attention_label not in labels:
                labels.append(variant.attention_label)
        if len(labels) > 1:
            return " / ".join(labels)
        if len(labels) == 1:
            return labels[0]
    attn = spec.attention_type
    if spec.attention_notes:
        return f"{attn}\n{spec.attention_notes[0][:28]}"
    return attn


def _ffn_class_display_name(class_name: str) -> str:
    """Short display names for FFN/MoE classes on the main decoder spine."""
    aliases = {
        "KimiSparseMoeBlock": "KimiMoE",
        "KimiMLP": "KimiMLP",
    }
    return aliases.get(class_name, class_name)


def _ffn_label(spec: ArchitectureSpec) -> tuple[str, str | None]:
    if spec.layer_variants:
        ffn_classes: list[str] = []
        for variant in spec.layer_variants:
            cls = variant.ffn_class or variant.ffn_label
            if cls not in ffn_classes:
                ffn_classes.append(cls)
        if len(ffn_classes) > 1:
            return " / ".join(_ffn_class_display_name(cls) for cls in ffn_classes), None
        if len(ffn_classes) == 1:
            return _ffn_class_display_name(ffn_classes[0]), None
        labels: list[str] = []
        for variant in spec.layer_variants:
            if variant.ffn_label not in labels:
                labels.append(variant.ffn_label)
        if len(labels) > 1:
            return " / ".join(labels), None
        if len(labels) == 1:
            return labels[0], None
    if spec.decoder_type == "Sparse MoE":
        experts = spec.num_experts or "?"
        label = f"MoE ({experts} experts)"
        return label, None
    return spec.ffn_type, None


def _repeat_block_label(spec: ArchitectureSpec) -> str:
    count = spec.num_hidden_layers or "?"
    lines = [f"{count} × Transformer block"]
    bullets: list[str] = []
    if spec.layer_variants:
        bullets.extend(f"{variant.count} {variant.label}" for variant in spec.layer_variants)
    elif spec.layer_mix:
        bullets.extend(part.strip() for part in spec.layer_mix.split(",") if part.strip())
    lines.extend(f"• {bullet}" for bullet in bullets)
    return "\n".join(lines)


def _fact_lines(spec: ArchitectureSpec) -> list[str]:
    lines = [
        f"Model type: {spec.model_type}",
        f"Decoder: {spec.decoder_type}",
        f"Attention: {spec.attention_type}",
        f"Positional: {spec.positional_encoding}",
        f"Norm: {spec.norm_type} ({spec.norm_placement})",
    ]
    if spec.decoder_class:
        lines.append(f"Decoder class: {spec.decoder_class}")
    if spec.checkpoint_source:
        lines.append(f"Checkpoint: {spec.checkpoint_source}")
    if spec.github_source:
        lines.append(f"GitHub code: {spec.github_source}")
    if spec.num_hidden_layers is not None and not spec.layer_repeat_lines:
        lines.append(f"Layers: {spec.num_hidden_layers}")
    if spec.hidden_size is not None:
        lines.append(f"Hidden size: {spec.hidden_size:,}")
    if spec.num_attention_heads is not None:
        kv = spec.num_key_value_heads or spec.num_attention_heads
        lines.append(f"Heads: {spec.num_attention_heads} Q / {kv} KV")
    if spec.vocab_size is not None:
        lines.append(f"Vocab: {spec.vocab_size:,}")
    if spec.max_position_embeddings is not None:
        lines.append(f"Context: {spec.max_position_embeddings:,} tokens")
    if spec.total_params_hint:
        param_line = f"Params (est.): {spec.total_params_hint}"
        if spec.active_params_hint:
            param_line += f" ({spec.active_params_hint} active)"
        lines.append(param_line)
    if spec.kv_cache_per_token_bf16:
        lines.append(f"KV cache / token (bf16 est.): {spec.kv_cache_per_token_bf16}")
    if spec.layer_repeat_lines:
        lines.append(f"Layer repeat: {spec.layer_repeat_lines[0]}")
        lines.extend(f"{FACT_SUBLINE_INDENT}{subline}" for subline in spec.layer_repeat_lines[1:])
    elif spec.layer_mix:
        lines.append(f"Layer mix: {spec.layer_mix}")
    if spec.forward_sequence:
        lines.append("Forward: " + " → ".join(spec.forward_sequence))
    for note in spec.moe_notes[:2]:
        lines.append(f"MoE: {note}")
    for note in spec.layer_notes[:1]:
        lines.append(f"Layers: {note}")
    for note in spec.analysis_notes[:1]:
        lines.append(f"AST: {note}")
    return lines


FACT_SUBLINE_INDENT = "    "


FIGURE_CONTENT_MARGIN = 0.35


def _figure_content_bounds(ax) -> ContentBounds | None:
    """Union of visible artist bounds in data coordinates."""
    fig = ax.figure
    fig.canvas.draw()
    union: ContentBounds | None = None
    for artist in ax.get_children():
        if not artist.get_visible():
            continue
        try:
            display_bb = artist.get_window_extent(fig.canvas.get_renderer())
        except Exception:
            continue
        if display_bb.width <= 1 or display_bb.height <= 1:
            continue
        data_bb = display_bb.transformed(ax.transData.inverted())
        bounds = ContentBounds(
            left=data_bb.x0,
            right=data_bb.x1,
            bottom=data_bb.y0,
            top=data_bb.y1,
        )
        union = bounds if union is None else union.union(bounds)
    return union


def _fit_figure_to_content(
    ax,
    fig,
    *,
    margin: float = FIGURE_CONTENT_MARGIN,
    min_width: float | None = None,
) -> tuple[float, float]:
    """Expand axis limits so every drawn artist fits inside the figure."""
    content = _figure_content_bounds(ax)
    if content is None:
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()
    else:
        x_left = content.left - margin
        x_right = content.right + margin
        y_bottom = content.bottom - margin
        y_top = content.top + margin
        if min_width is not None and x_right - x_left < min_width:
            pad = (min_width - (x_right - x_left)) / 2
            x_left -= pad
            x_right += pad
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    width = x_right - x_left
    height = y_top - y_bottom
    fig.set_size_inches(width, max(height, 1.0))
    return width, height


def _wrap_fact_text(text: str, *, width: int = 48) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [text]


def _inline_frame_label_lines(label: str, frame_width: float) -> list[str]:
    """Wrap an inline-frame caption to stay inside the dotted frame width."""
    if len(label) <= 40:
        return [label]
    usable = max(0.35, frame_width - 0.04)
    wrap_width = max(8, int(usable / INLINE_FRAME_LABEL_CHAR_W))
    if len(label) <= wrap_width:
        return [label]
    return _wrap_fact_text(label, width=wrap_width)


def _fact_sheet_content_rows(bullet_lines: list[str], *, wrap_width: int) -> int:
    """Count wrapped text rows using the same rules as _draw_fact_sheet."""
    total = 0
    for line in bullet_lines:
        is_subline = line.startswith(FACT_SUBLINE_INDENT)
        content = line[len(FACT_SUBLINE_INDENT) :] if is_subline else line
        wrap_width_line = wrap_width - (len(FACT_SUBLINE_INDENT) if is_subline else 2)
        total += len(_wrap_fact_text(content, width=max(20, wrap_width_line)))
    return total


def _fact_sheet_highlight_rows(spec: ArchitectureSpec, *, wrap_width: int) -> list[str]:
    if not spec.highlights:
        return []
    return _wrap_fact_text("Highlights: " + "; ".join(spec.highlights), width=wrap_width)


def _draw_panel_title(ax, x: float, y: float, title: str) -> None:
    ax.text(
        x,
        y,
        title,
        fontsize=PANEL_TITLE_FONT,
        fontweight="bold",
        color=PANEL_TITLE_COLOR,
        zorder=5,
    )


def _fact_sheet_height(spec: ArchitectureSpec, *, wrap_width: int = PANEL_WRAP_WIDTH) -> float:
    bullet_lines = _fact_lines(spec)
    highlight_rows = _fact_sheet_highlight_rows(spec, wrap_width=wrap_width)

    line_height = PANEL_LINE_HEIGHT
    highlight_height = 0.17 * len(highlight_rows) if highlight_rows else 0.0
    padding_top = PANEL_PAD_TOP
    padding_bottom = PANEL_PAD_BOTTOM
    gap_before_highlights = 0.12 if highlight_rows else 0.0

    bullet_count = _fact_sheet_content_rows(bullet_lines, wrap_width=wrap_width)

    fact_h = padding_top + bullet_count * line_height + gap_before_highlights + highlight_height + padding_bottom
    return max(fact_h, 1.6)


def _draw_fact_sheet(
    ax,
    spec: ArchitectureSpec,
    *,
    fact_x: float,
    fact_y: float,
    fact_w: float = PANEL_W,
    wrap_width: int = PANEL_WRAP_WIDTH,
) -> float:
    bullet_lines = _fact_lines(spec)
    highlight_rows = _fact_sheet_highlight_rows(spec, wrap_width=wrap_width)

    line_height = PANEL_LINE_HEIGHT
    fact_h = _fact_sheet_height(spec, wrap_width=wrap_width)
    padding_top = PANEL_PAD_TOP
    gap_before_highlights = 0.12 if highlight_rows else 0.0

    fact_patch = FancyBboxPatch(
        (fact_x, fact_y),
        fact_w,
        fact_h,
        boxstyle="round,pad=0.01,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=COLORS["fact_border"],
        facecolor=COLORS["fact_bg"],
        zorder=1,
    )
    ax.add_patch(fact_patch)
    _draw_panel_title(ax, fact_x + PANEL_PAD_X, fact_y + fact_h - PANEL_TITLE_Y, "Fact sheet")

    cursor_y = fact_y + fact_h - padding_top
    for line in bullet_lines:
        is_subline = line.startswith(FACT_SUBLINE_INDENT)
        content = line[len(FACT_SUBLINE_INDENT) :] if is_subline else line
        wrap_width_line = wrap_width - (len(FACT_SUBLINE_INDENT) if is_subline else 2)
        rows = _wrap_fact_text(content, width=max(20, wrap_width_line))
        for index, row in enumerate(rows):
            if is_subline:
                prefix = FACT_SUBLINE_INDENT if index == 0 else FACT_SUBLINE_INDENT + "  "
            else:
                prefix = "• " if index == 0 else "  "
            ax.text(
                fact_x + PANEL_PAD_X,
                cursor_y,
                f"{prefix}{row}",
                fontsize=PANEL_BODY_FONT,
                color=PANEL_BODY_COLOR,
                va="top",
                zorder=5,
            )
            cursor_y -= line_height

    if highlight_rows:
        cursor_y -= gap_before_highlights
        for row in highlight_rows:
            ax.text(fact_x + 0.25, cursor_y, row, fontsize=8, color=COLORS["muted"], va="top", zorder=5)
            cursor_y -= 0.17

    return fact_h


def _ordered_block_components(spec: ArchitectureSpec) -> list[BlockComponent]:
    if not spec.block_components:
        return []
    return sorted(
        spec.block_components,
        key=lambda comp: (
            comp.forward_order is None,
            comp.forward_order if comp.forward_order is not None else 999,
            comp.attr_name,
        ),
    )


def _collect_sublayer_pairs(sequence: list[BlockComponent]) -> list[tuple[BlockComponent, BlockComponent]]:
    """Pair each norm with the compute module that follows it in forward order."""
    return collect_norm_module_pairs(sequence)


def _default_stack_pre(spec: ArchitectureSpec) -> list[BlockComponent]:
    return [
        BlockComponent(
            attr_name="embed_tokens",
            class_name="Embedding",
            role="embedding",
            label="Token Embedding",
            forward_order=0,
        ),
        BlockComponent(
            attr_name="rotary_emb",
            class_name="RotaryEmbedding",
            role="positional",
            label=spec.positional_encoding,
            forward_order=1,
            details=[f"positional encoding ({spec.positional_encoding})"],
        ),
    ]


def _default_stack_tail(spec: ArchitectureSpec) -> list[BlockComponent]:
    norm_type = spec.norm_type or "RMSNorm"
    return [
        BlockComponent(
            attr_name="norm",
            class_name=norm_type,
            role="norm",
            label=norm_type,
            forward_order=0,
        ),
        BlockComponent(
            attr_name="lm_head",
            class_name="Linear",
            role="head",
            label="Linear",
            forward_order=1,
        ),
    ]


def _stack_pre_components(spec: ArchitectureSpec) -> list[BlockComponent]:
    return spec.stack_pre if spec.stack_pre else _default_stack_pre(spec)


def _stack_tail_components(spec: ArchitectureSpec) -> list[BlockComponent]:
    return spec.stack_tail if spec.stack_tail else _default_stack_tail(spec)


def _spine_color(role: str) -> str:
    return {
        "embedding": COLORS["embed"],
        "positional": COLORS["pos"],
        "norm": COLORS["norm"],
        "head": COLORS["head"],
    }.get(role, COLORS["embed"])


def _basic_component_labels(comp: BlockComponent) -> tuple[str, str | None] | None:
    """Return operation label pair for basic spine/decoder tiles."""
    from visualizer.ast_analyze import _label_for

    if comp.role == "embedding":
        return comp.label, None
    if displays_as_linear(comp.attr_name, comp.class_name):
        return "Linear", None
    if comp.role == "norm":
        return _label_for("norm", comp.class_name, comp.attr_name), None
    return None


def _spine_display_label(component: BlockComponent, spec: ArchitectureSpec) -> str:
    if component.role == "positional":
        return f"Positional ({spec.positional_encoding})"
    basic = _basic_component_labels(component)
    if basic is not None:
        return basic[0]
    return component.label


def _spine_sublabel(component: BlockComponent) -> str | None:
    basic = _basic_component_labels(component)
    if basic is not None:
        return basic[1]
    return None


def _spine_box_height(component: BlockComponent) -> float:
    sublabel = _spine_sublabel(component)
    return box_height_for_content(sublabel) if sublabel else single_line_box_height()


def _module_input_labels(spec: ArchitectureSpec) -> dict[str, str]:
    """Map compute module attr names to the upstream operator that feeds them in the outer block."""
    components = _ordered_block_components(spec)
    if spec.forward_sequence:
        return input_sources_from_forward_sequence(components, spec.forward_sequence)
    return upstream_input_sources(components)


def _block_frame_header_height(repeat_label: str | None = None) -> float:
    """Vertical space reserved inside the frame below the top border."""
    if repeat_label:
        return BLOCK_FRAME_CONTENT_GAP
    return BLOCK_FRAME_LABEL_PAD_Y + BLOCK_FRAME_DECODER_LINE_H


def _block_content_entry_top(block_top: float, repeat_label: str | None = None) -> float:
    """Y for the first submodule row, below the frame header labels."""
    del repeat_label
    frame_top = _block_frame_top(block_top)
    return frame_top - _block_frame_header_height() - BLOCK_FRAME_CONTENT_GAP


def _block_frame_top(block_top: float, repeat_label: str | None = None) -> float:
    """Top edge of the outer block frame."""
    del repeat_label
    return block_top + 0.03


def _text_size_in_axes(
    ax,
    text: str,
    *,
    fontsize: float,
    fontweight: str = "bold",
    va: str = "bottom",
) -> tuple[float, float]:
    from visualizer.text_measure import measure_text_bounds

    bounds = measure_text_bounds(
        ax,
        text,
        0.0,
        0.0,
        fontsize=fontsize,
        fontweight=fontweight,
        va=va,
        ha="left",
    )
    return bounds.width, bounds.height


def _text_width_in_axes(ax, text: str, *, fontsize: float) -> float:
    return _text_size_in_axes(ax, text, fontsize=fontsize)[0]


def _box_label_width(ax, text: str, *, fontsize: float, sublabel: str | None = None, sub_fontsize: float | None = None) -> float:
    from visualizer.text_measure import box_label_size

    width, _height = box_label_size(ax, text, sublabel, fontsize=fontsize, sub_fontsize=sub_fontsize)
    return width


def _repeat_label_bbox(
    ax,
    text_x: float,
    anchor_y: float,
    repeat_label: str,
    *,
    fontsize: float = 10.0,
):
    fig = ax.figure
    if fig.canvas.get_renderer() is None:
        fig.canvas.draw()
    tmp = ax.text(
        text_x,
        anchor_y,
        repeat_label,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        alpha=0.0,
    )
    bb = tmp.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    tmp.remove()
    return bb


def _decoder_label_fontsize(base_fontsize: float = 10.0) -> float:
    return max(8.5, base_fontsize - 0.6)


def _decoder_label_bbox(
    ax,
    text_x: float,
    anchor_y: float,
    decoder_label: str,
    *,
    fontsize: float | None = None,
    va: str = "top",
):
    fs = fontsize if fontsize is not None else _decoder_label_fontsize()
    fig = ax.figure
    if fig.canvas.get_renderer() is None:
        fig.canvas.draw()
    tmp = ax.text(
        text_x,
        anchor_y,
        decoder_label,
        ha="left",
        va=va,
        fontsize=fs,
        fontweight="bold",
        alpha=0.0,
    )
    bb = tmp.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    tmp.remove()
    return bb


def _outside_block_labels_bbox_at_gap(
    ax,
    text_x: float,
    frame_top: float,
    repeat_label: str,
    decoder_label: str,
    *,
    outside_gap: float,
    repeat_fontsize: float = 10.0,
):
    repeat_anchor = frame_top + outside_gap
    repeat_bb = _repeat_label_bbox(ax, text_x, repeat_anchor, repeat_label, fontsize=repeat_fontsize)
    decoder_anchor = repeat_bb.y0 - BLOCK_FRAME_DECODER_OUTSIDE_GAP
    decoder_bb = _decoder_label_bbox(ax, text_x, decoder_anchor, decoder_label, va="top")
    from matplotlib.transforms import Bbox

    return Bbox.from_extents(
        min(repeat_bb.x0, decoder_bb.x0),
        decoder_bb.y0,
        max(repeat_bb.x1, decoder_bb.x1),
        repeat_bb.y1,
    )


def _effective_repeat_outside_gap(
    ax,
    repeat_label: str,
    decoder_label: str,
    *,
    repeat_fontsize: float = 10.0,
) -> float:
    """Minimum gap above frame top so repeat + decoder labels sit fully outside the box."""
    min_clear = FRAME_PATCH_TOP_OUTSET + BLOCK_FRAME_DECODER_FRAME_GAP
    repeat_bb = _repeat_label_bbox(
        ax,
        BLOCK_FRAME_LABEL_PAD_X,
        BLOCK_FRAME_REPEAT_OUTSIDE_GAP,
        repeat_label,
        fontsize=repeat_fontsize,
    )
    decoder_bb = _decoder_label_bbox(
        ax,
        BLOCK_FRAME_LABEL_PAD_X,
        repeat_bb.y0 - BLOCK_FRAME_DECODER_OUTSIDE_GAP,
        decoder_label,
        va="top",
    )
    required = min_clear + BLOCK_FRAME_DECODER_OUTSIDE_GAP + decoder_bb.height
    return max(BLOCK_FRAME_REPEAT_OUTSIDE_GAP, required)


def _outside_block_labels_bbox(
    ax,
    text_x: float,
    frame_top: float,
    repeat_label: str,
    decoder_label: str,
    *,
    repeat_fontsize: float = 10.0,
):
    outside_gap = _effective_repeat_outside_gap(
        ax,
        repeat_label,
        decoder_label,
        repeat_fontsize=repeat_fontsize,
    )
    return _outside_block_labels_bbox_at_gap(
        ax,
        text_x,
        frame_top,
        repeat_label,
        decoder_label,
        outside_gap=outside_gap,
        repeat_fontsize=repeat_fontsize,
    )


def _outside_block_labels_clearance(
    ax,
    repeat_label: str,
    decoder_label: str,
    *,
    repeat_fontsize: float = 10.0,
) -> float:
    """Vertical space needed above the frame top for outside repeat + decoder labels."""
    probe_x = BLOCK_FRAME_LABEL_PAD_X
    outside_gap = _effective_repeat_outside_gap(
        ax,
        repeat_label,
        decoder_label,
        repeat_fontsize=repeat_fontsize,
    )
    bbox = _outside_block_labels_bbox_at_gap(
        ax,
        probe_x,
        0.0,
        repeat_label,
        decoder_label,
        outside_gap=outside_gap,
        repeat_fontsize=repeat_fontsize,
    )
    return (
        bbox.y1
        + BLOCK_FRAME_REPEAT_CONNECTOR_MARGIN
        + BLOCK_FRAME_REPEAT_LABEL_GAP
    )


def _block_frame_bottom(content_bottom: float) -> float:
    """Bottom edge of the outer block frame (logical, before patch outset)."""
    return content_bottom - BLOCK_FRAME_BOTTOM_INSET


def _block_top_below_repeat_label(
    ax,
    *,
    cx: float,
    block_w: float,
    above_bottom: float,
    repeat_label: str,
    decoder_label: str,
) -> float:
    """Choose block_top so outside repeat/decoder labels clear the block above."""
    frame_left = cx - block_w / 2
    text_x = frame_left + BLOCK_FRAME_LABEL_PAD_X
    pos_ceiling = above_bottom - STACK_BOX_BOTTOM_OUTSET - BLOCK_FRAME_REPEAT_LABEL_GAP
    clearance = _outside_block_labels_clearance(ax, repeat_label, decoder_label)
    frame_top = pos_ceiling - clearance + BLOCK_FRAME_REPEAT_LABEL_GAP
    while True:
        bbox = _outside_block_labels_bbox(ax, text_x, frame_top, repeat_label, decoder_label)
        if bbox.y1 <= pos_ceiling:
            return frame_top - 0.03
        frame_top -= 0.03


def _block_width_for_repeat_label(
    ax,
    repeat_label: str,
    *,
    fontsize: float = 10.0,
) -> float:
    """Minimum frame width: 2×(measured text + symmetric horizontal padding).

    The label occupies the left half (padding + text + padding); the right half
    leaves room for the spine at the box center without routing around the text.
    """
    text_w = max(
        _text_width_in_axes(ax, line, fontsize=fontsize) for line in repeat_label.split("\n")
    )
    return 2 * box_width_for_text_width(text_w, pad_x=FRAME_LABEL_PAD_X)


def _main_block_width(
    ax,
    *,
    repeat_label: str | None,
    decoder_label: str,
    inner_w: float,
) -> float:
    label_widths = [
        _box_label_width(ax, decoder_label, fontsize=9.4),
        inner_w + 0.5,
    ]
    if repeat_label:
        label_widths.append(_block_width_for_repeat_label(ax, repeat_label))
        return max(*label_widths)
    return max(MAIN_BLOCK_W, *label_widths)


def _detail_layout_geometry(
    canvas_width: float,
    *,
    fact_x: float,
    fact_w: float = PANEL_W,
    detail_content_width: float | None = None,
) -> tuple[float, bool]:
    """Return ``(canvas_width, below_fact_sheet)`` for detailed diagram placement.

    The fact sheet stays at ``fact_x`` beside the main model. When the widest
    measured detail section is broader than the band left of the fact sheet,
    block internals are rendered below the fact sheet. Individual sections are
    sized to their own content width at render time.
    """
    detail_min_left = DIAGRAM_LEFT_MARGIN + 0.05
    content_w = max(DETAIL_MIN_BLOCK_W, detail_content_width or DETAIL_MIN_BLOCK_W)
    available_beside = max(0.5, fact_x - FACT_SHEET_GAP - detail_min_left)
    below_fact_sheet = content_w > available_beside + 0.15
    required_canvas = max(
        fact_x + fact_w + DIAGRAM_RIGHT_MARGIN,
        detail_min_left + content_w + DIAGRAM_RIGHT_MARGIN,
    )
    canvas_width = max(canvas_width, required_canvas)
    return canvas_width, below_fact_sheet


def _fact_sheet_x(main_model_right: float, *, gap: float = FACT_SHEET_GAP) -> float:
    """Place the fact sheet just to the right of the main model spine."""
    return main_model_right + gap


def _draw_block_frame(
    ax,
    *,
    cx: float,
    block_w: float,
    bottom_y: float,
    top_y: float,
    label: str,
    repeat_label: str | None = None,
    edgecolor: str | None = None,
    facecolor: str | None = None,
    fontsize: float = 10.0,
) -> None:
    block_h = top_y - bottom_y
    frame_left = cx - block_w / 2
    block_patch = FancyBboxPatch(
        (frame_left, bottom_y),
        block_w,
        block_h,
        boxstyle="round,pad=0.01,rounding_size=0.10",
        linewidth=2.0,
        edgecolor=edgecolor or COLORS["block_border"],
        facecolor=facecolor or COLORS["block_fill"],
        zorder=0,
    )
    ax.add_patch(block_patch)

    text_x = frame_left + BLOCK_FRAME_LABEL_PAD_X
    label_color = edgecolor or COLORS["block_border"]
    if repeat_label:
        outside_gap = _effective_repeat_outside_gap(
            ax, repeat_label, label, repeat_fontsize=fontsize
        )
        repeat_anchor = top_y + outside_gap
        ax.text(
            text_x,
            repeat_anchor,
            repeat_label,
            ha="left",
            va="bottom",
            fontsize=fontsize,
            color=label_color,
            fontweight="bold",
            zorder=10,
        )
        repeat_bb = _repeat_label_bbox(ax, text_x, repeat_anchor, repeat_label, fontsize=fontsize)
        ax.text(
            text_x,
            repeat_bb.y0 - BLOCK_FRAME_DECODER_OUTSIDE_GAP,
            label,
            ha="left",
            va="top",
            fontsize=_decoder_label_fontsize(fontsize),
            color=label_color,
            fontweight="bold",
            zorder=10,
        )
    else:
        ax.text(
            text_x,
            top_y - BLOCK_FRAME_LABEL_PAD_Y,
            label,
            ha="left",
            va="top",
            fontsize=fontsize,
            color=label_color,
            fontweight="bold",
            zorder=10,
        )


def _make_node(
    node_id: str,
    cx: float,
    top_y: float,
    w: float,
    h: float,
    label: str,
    facecolor: str,
    *,
    text_color: str = "white",
    sublabel: str | None = None,
    fontsize: float = 9.0,
    pad_x: float | None = None,
    pad_y: float | None = None,
) -> Node:
    return Node(
        node_id=node_id,
        x=cx - w / 2,
        y=top_y - h,
        w=w,
        h=h,
        label=label,
        facecolor=facecolor,
        text_color=text_color,
        sublabel=sublabel,
        fontsize=fontsize,
        pad_x=pad_x,
        pad_y=pad_y,
    )


def _render_sublayer(
    layout: DiagramLayout,
    ax,
    *,
    cx: float,
    branch_x: float,
    top_y: float,
    spine_y: float | None,
    norm_label: str,
    norm_id: str,
    module_label: str,
    module_id: str,
    module_color: str,
    module_sublabel: str | None,
    norm_w: float,
    inner_w: float,
    gap: float,
    entry_from_y: float | None = None,
) -> float:
    """Draw norm -> module -> residual merge. Returns merge y for the next sublayer."""
    norm_h = single_line_box_height()
    sub_h = box_height_for_content(module_sublabel)
    if spine_y is not None:
        top_y = spine_y - MERGE_RADIUS - MERGE_CLEARANCE - gap

    norm_node = _make_node(
        norm_id,
        cx,
        top_y,
        norm_w,
        norm_h,
        norm_label,
        COLORS["norm"],
        text_color=COLORS["text"],
        fontsize=8,
    )
    layout.add(norm_node)
    _fit_spine_node_to_label(ax, norm_node)
    _draw_box(ax, norm_node)
    if spine_y is not None:
        _connect_from_merge(ax, cx, spine_y, norm_node)
    elif entry_from_y is not None:
        _arrow(ax, cx, entry_from_y, norm_node.cx, norm_node.top)

    skip_from_y = norm_node.top

    module_node = _make_node(
        module_id,
        cx,
        norm_node.bottom - gap,
        inner_w,
        sub_h,
        module_label,
        module_color,
        sublabel=module_sublabel,
        fontsize=8.8,
    )
    layout.add(module_node)
    _fit_spine_node_to_label(ax, module_node)
    _draw_box(ax, module_node)
    _connect_down(ax, norm_node, module_node)

    return _residual_merge(
        ax,
        module_cx=module_node.cx,
        module_bottom=module_node.bottom,
        skip_from_y=skip_from_y,
        spine_x=cx,
        branch_x=branch_x,
    )


def _component_sublabel(comp: BlockComponent) -> str | None:
    if _basic_component_labels(comp) is not None:
        return None
    sublabel = comp.class_name if comp.class_name != comp.label else None
    if comp.details:
        sublabel = (sublabel + "\n" if sublabel else "") + comp.details[0][:28]
    return sublabel


def _component_display_label(comp: BlockComponent) -> str:
    basic = _basic_component_labels(comp)
    if basic is not None:
        return basic[0]
    return comp.label


def _block_content_widths(ax, spec: ArchitectureSpec) -> tuple[float, float]:
    """Return (norm_w, inner_w) using the same horizontal padding as RMSNorm boxes."""
    if _ordered_block_components(spec):
        pairs = _collect_sublayer_pairs(_ordered_block_components(spec))
        norm_w = 0.0
        inner_w = 0.0
        for norm_comp, comp in pairs:
            norm_w = max(norm_w, _box_label_width(ax, _component_display_label(norm_comp), fontsize=8))
            inner_w = max(
                inner_w,
                _box_label_width(
                    ax,
                    comp.label,
                    fontsize=8.8,
                    sublabel=_component_sublabel(comp),
                ),
            )
        return norm_w, inner_w

    attn_label = _attention_label(spec)
    ffn_label, ffn_sub = _ffn_label(spec)
    norm_w = _box_label_width(ax, spec.norm_type, fontsize=8)
    inner_w = max(
        _box_label_width(ax, attn_label, fontsize=8.8),
        _box_label_width(ax, ffn_label, fontsize=8.8, sublabel=ffn_sub),
    )
    return norm_w, inner_w


def _connect_into_block(
    ax,
    source: Node | None,
    *,
    cx: float,
    frame_top: float,
    frame_left: float,
    repeat_label: str | None,
    decoder_label: str | None = None,
    source_y: float | None = None,
    source_x: float | None = None,
) -> None:
    if source is not None:
        start_x, start_y = source.cx, source.bottom
    elif source_x is not None and source_y is not None:
        start_x, start_y = source_x, source_y
    else:
        return

    if not repeat_label:
        _arrow(ax, start_x, start_y, cx, frame_top)
        return

    text_x = frame_left + BLOCK_FRAME_LABEL_PAD_X
    decoder = decoder_label or "Transformer block"
    bbox = _outside_block_labels_bbox(ax, text_x, frame_top, repeat_label, decoder)
    margin = BLOCK_FRAME_REPEAT_CONNECTOR_MARGIN
    if cx >= bbox.x1 + margin:
        _arrow(ax, start_x, start_y, cx, frame_top)
        return

    above_text = bbox.y1 + margin
    below_text = bbox.y0 - margin
    _arrow(ax, start_x, start_y, cx, above_text)
    _line(ax, cx, above_text, cx, below_text, color=COLORS["flow"])
    _arrow(ax, cx, below_text, cx, frame_top)


def _layout_component_block(
    layout: DiagramLayout,
    ax,
    *,
    cx: float,
    top_y: float,
    block_w: float,
    spec: ArchitectureSpec,
    norm_w: float,
    inner_w: float,
    repeat_label: str | None = None,
) -> float:
    sequence = _ordered_block_components(spec)
    pairs = _collect_sublayer_pairs(sequence)

    gap = 0.24
    branch_x = _residual_branch_x(cx, block_w)
    entry_top = _block_content_entry_top(top_y, repeat_label)
    spine_y: float | None = None
    merge_ys: list[float] = []

    for norm_comp, comp in pairs:
        color = ROLE_COLORS.get(comp.role, ROLE_COLORS["other"])
        sublabel = _component_sublabel(comp)
        module_label = comp.label
        if comp.role == "attention":
            module_label = _attention_label(spec)
            sublabel = None
        elif comp.role in {"moe", "ffn"}:
            module_label, variant_sub = _ffn_label(spec)
            if variant_sub:
                sublabel = variant_sub

        spine_y = _render_sublayer(
            layout,
            ax,
            cx=cx,
            branch_x=branch_x,
            top_y=entry_top,
            spine_y=spine_y,
            norm_label=_component_display_label(norm_comp),
            norm_id=norm_comp.attr_name,
            module_label=module_label,
            module_id=comp.attr_name,
            module_color=color,
            module_sublabel=sublabel,
            norm_w=norm_w,
            inner_w=inner_w,
            gap=gap,
            entry_from_y=entry_top if spine_y is None else None,
        )
        merge_ys.append(spine_y)

    merge_pad = MERGE_RADIUS + MERGE_CLEARANCE + 0.10
    if merge_ys:
        content_bottom = min(merge_ys) - merge_pad
        frame_bottom = _block_frame_bottom(content_bottom)
    else:
        content_bottom = top_y - 0.2
        frame_bottom = _block_frame_bottom(content_bottom)

    _draw_block_frame(
        ax,
        cx=cx,
        block_w=block_w,
        bottom_y=frame_bottom,
        top_y=_block_frame_top(top_y, repeat_label),
        repeat_label=repeat_label,
        label=spec.decoder_class or "Transformer block",
    )

    if merge_ys:
        return content_bottom
    return content_bottom


def _layout_default_block(
    layout: DiagramLayout,
    ax,
    *,
    cx: float,
    top_y: float,
    block_w: float,
    spec: ArchitectureSpec,
    norm_w: float,
    inner_w: float,
    repeat_label: str | None = None,
) -> float:
    gap = 0.28
    branch_x = _residual_branch_x(cx, block_w)
    entry_top = _block_content_entry_top(top_y, repeat_label)

    attn_label = _attention_label(spec)
    ffn_label, ffn_sub = _ffn_label(spec)
    ffn_color = COLORS["moe"] if spec.decoder_type == "Sparse MoE" else COLORS["ffn"]

    merge1 = _render_sublayer(
        layout,
        ax,
        cx=cx,
        branch_x=branch_x,
        top_y=entry_top,
        spine_y=None,
        norm_label=spec.norm_type,
        norm_id="norm1",
        module_label=attn_label,
        module_id="attn",
        module_color=COLORS["attention"],
        module_sublabel=None,
        norm_w=norm_w,
        inner_w=inner_w,
        gap=gap,
        entry_from_y=entry_top,
    )

    merge2 = _render_sublayer(
        layout,
        ax,
        cx=cx,
        branch_x=branch_x,
        top_y=entry_top,
        spine_y=merge1,
        norm_label=spec.norm_type,
        norm_id="norm2",
        module_label=ffn_label,
        module_id="ffn",
        module_color=ffn_color,
        module_sublabel=ffn_sub,
        norm_w=norm_w,
        inner_w=inner_w,
        gap=gap,
    )

    merge_pad = MERGE_RADIUS + MERGE_CLEARANCE + 0.10
    content_bottom = merge2 - merge_pad
    _draw_block_frame(
        ax,
        cx=cx,
        block_w=block_w,
        bottom_y=_block_frame_bottom(content_bottom),
        top_y=_block_frame_top(top_y, repeat_label),
        repeat_label=repeat_label,
        label=spec.decoder_class or "Transformer block",
    )

    return content_bottom


def _anchor_from_position(pos: LayoutPosition) -> _RenderAnchor:
    return _RenderAnchor(
        cx=pos.cx,
        top=pos.top_y,
        bottom=pos.bottom,
        left=pos.cx - pos.width / 2,
        right=pos.cx + pos.width / 2,
    )


def _anchor_from_node(node: Node) -> _RenderAnchor:
    return _RenderAnchor(
        cx=node.cx,
        top=node.top,
        bottom=node.bottom,
        left=node.x,
        right=node.x + node.w,
    )


def _segment_hits_obstacle(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    obstacles: list[_RenderAnchor],
    *,
    margin: float = 0.05,
) -> bool:
    min_x = min(x1, x2) - margin
    max_x = max(x1, x2) + margin
    min_y = min(y1, y2) - margin
    max_y = max(y1, y2) + margin
    for obs in obstacles:
        if max_x < obs.left or min_x > obs.right or max_y < obs.bottom or min_y > obs.top:
            continue
        return True
    return False


def _anchor_interior(
    anchor: _RenderAnchor,
    *,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> _RenderAnchor | None:
    """Return a shrink-wrapped anchor representing tile interior for hit tests."""
    left = anchor.left + margin
    right = anchor.right - margin
    bottom = anchor.bottom + margin
    top = anchor.top - margin
    if left >= right or bottom >= top:
        return None
    return _RenderAnchor(cx=anchor.cx, top=top, bottom=bottom, left=left, right=right)


def _segment_penetrates_anchor(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    anchor: _RenderAnchor,
    *,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> bool:
    """True when a segment crosses the interior of a tile anchor."""
    interior = _anchor_interior(anchor, margin=margin)
    if interior is None:
        return False
    return _segment_hits_obstacle(x1, y1, x2, y2, [interior], margin=0.0)


def _vertical_segment_crosses_anchor(
    x: float,
    y_a: float,
    y_b: float,
    anchor: _RenderAnchor,
    *,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> bool:
    if x + margin < anchor.left or x - margin > anchor.right:
        return False
    lo, hi = sorted((y_a, y_b))
    return lo < anchor.top - margin and hi > anchor.bottom + margin


def _right_bypass_x_clearing_horizontal_segment(
    x_left: float,
    y: float,
    obstacles: list[_RenderAnchor],
    *,
    initial_bypass_x: float,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> float:
    """Extend bypass_x rightward until a horizontal segment at y from x_left clears obstacles."""
    bypass_x = initial_bypass_x
    changed = True
    while changed:
        changed = False
        for obstacle in obstacles:
            interior = _anchor_interior(obstacle, margin=margin)
            if interior is None:
                continue
            if y + margin < interior.bottom or y - margin > interior.top:
                continue
            if interior.right + margin <= x_left:
                continue
            if interior.left - margin >= bypass_x:
                continue
            new_x = interior.right + margin
            if new_x > bypass_x:
                bypass_x = new_x
                changed = True
    return bypass_x


def _left_bypass_x_clearing_horizontal_segment(
    x_right: float,
    y: float,
    obstacles: list[_RenderAnchor],
    *,
    initial_bypass_x: float,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> float:
    """Extend bypass_x leftward until a horizontal segment at y to x_right clears obstacles."""
    bypass_x = initial_bypass_x
    changed = True
    while changed:
        changed = False
        for obstacle in obstacles:
            interior = _anchor_interior(obstacle, margin=margin)
            if interior is None:
                continue
            if y + margin < interior.bottom or y - margin > interior.top:
                continue
            if interior.left - margin >= x_right:
                continue
            if interior.right + margin <= bypass_x:
                continue
            new_x = interior.left - margin
            if new_x < bypass_x:
                bypass_x = new_x
                changed = True
    return bypass_x


def _right_bypass_x_clearing_vertical_segment(
    y_lo: float,
    y_hi: float,
    obstacles: list[_RenderAnchor],
    *,
    initial_bypass_x: float,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> float:
    """Extend bypass_x rightward until a vertical segment at x clears obstacles."""
    bypass_x = initial_bypass_x
    changed = True
    while changed:
        changed = False
        for obstacle in obstacles:
            if _vertical_segment_crosses_anchor(
                bypass_x,
                y_lo,
                y_hi,
                obstacle,
                margin=margin,
            ):
                new_x = obstacle.right + margin
                if new_x > bypass_x:
                    bypass_x = new_x
                    changed = True
    return bypass_x


def _frame_tail_exit_horiz_y(
    graph,
    positions: list,
    src: int,
) -> float | None:
    """Y of the first horizontal leg when an inline-frame tail exits to a merge bus."""
    from visualizer.computation_graph import _inline_frame_tail_indices

    if src not in _inline_frame_tail_indices(graph):
        return None
    for frame in graph.inline_frames:
        if src not in frame.node_indices or frame.node_indices[-1] != src:
            continue
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        return _frame_exit_horizontal_y(bounds, source_bottom=positions[src].bottom)
    return None


def _pipeline_merge_bus_y(
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    target: int,
    sources: list[int],
) -> float | None:
    """Shared merge-bus Y for tensor-port pipelines fed by inline-frame tails."""
    from visualizer.computation_graph import _graph_has_tensor_ports, _inline_frame_tail_indices

    if not _graph_has_tensor_ports(graph):
        return None
    frame_tails = _inline_frame_tail_indices(graph)
    tail_sources = [src for src in sources if src in frame_tails]
    if len(tail_sources) < 2:
        return None

    exit_horiz_y = float("inf")
    for src in tail_sources:
        horiz_y = _frame_tail_exit_horiz_y(graph, positions, src)
        if horiz_y is not None:
            exit_horiz_y = min(exit_horiz_y, horiz_y)
    if exit_horiz_y == float("inf"):
        return None

    target_anchor = anchors.get(target)
    if target_anchor is None:
        return None
    bus_y = exit_horiz_y - CONNECTOR_EXIT_STUB - PIPELINE_MERGE_BUS_BELOW_FRAME_GAP
    return max(bus_y, _connector_min_bus_y_above_target(target_anchor))


def _all_inline_frame_draw_bounds(
    graph,
    positions: list,
) -> list:
    """Return dotted-frame bounds for every inline frame (routing obstacles)."""
    return [_inline_frame_draw_bounds(frame, positions, graph) for frame in graph.inline_frames]


def _segment_crosses_frame_bounds(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_bounds,
    *,
    margin: float = CONNECTOR_ATTACHED_BOX_MARGIN,
) -> bool:
    """True when a segment cuts through an expanded inline-frame interior."""
    obs = _RenderAnchor(
        cx=(frame_bounds.left + frame_bounds.right) / 2,
        top=frame_bounds.top,
        bottom=frame_bounds.bottom,
        left=frame_bounds.left,
        right=frame_bounds.right,
    )
    return _segment_penetrates_anchor(x1, y1, x2, y2, obs, margin=margin)


def _frame_for_tail_node(graph, src: int):
    for frame in graph.inline_frames:
        if frame.node_indices and frame.node_indices[-1] == src:
            return frame
    return None


def _frame_exit_horizontal_y(
    frame_bounds,
    *,
    source_bottom: float,
    source_cx: float | None = None,
    obstacles: list[_RenderAnchor] | None = None,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> float:
    """Y level for the first horizontal leg leaving a dotted inline frame."""
    below_frame = frame_bounds.bottom - CONNECTOR_EXIT_STUB
    below_source = source_bottom - CONNECTOR_EXIT_STUB
    stub_y = min(below_frame, below_source)
    if source_cx is None or not obstacles:
        return stub_y
    for obstacle in obstacles:
        if abs(obstacle.cx - source_cx) > 0.06:
            continue
        if obstacle.top >= source_bottom - margin:
            continue
        stub_y = max(
            stub_y,
            obstacle.top + margin + CONNECTOR_ATTACHED_BOX_MARGIN + PARALLEL_CONNECTOR_COORD_EPS,
        )
    return stub_y


def _pipeline_frame_exit_x(
    source: _RenderAnchor,
    target: _RenderAnchor,
    frame_bounds,
    obstacles: list[_RenderAnchor],
    *,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> float:
    """Left gutter for frame-exit routing, clear of same-column tiles below the source."""
    exit_x = frame_bounds.left + margin
    for obstacle in obstacles:
        if abs(obstacle.cx - source.cx) > 0.10:
            continue
        if obstacle.top > source.bottom + margin:
            continue
        if obstacle.bottom < target.top - margin:
            continue
        exit_x = min(exit_x, obstacle.left - margin - margin)
    return exit_x


def _inline_frame_bypass_tee_y(
    graph,
    frame,
    src: int,
    tgt: int,
    anchors: dict[int, _RenderAnchor],
) -> float | None:
    """Y on the spine column where a bypass branches toward its first skipped tile."""
    from visualizer.computation_graph import _ordered_inline_frame_chain

    chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
    try:
        src_index = chain.index(src)
        tgt_index = chain.index(tgt)
    except ValueError:
        return None
    if src_index >= tgt_index - 1:
        return None
    intermediate = chain[src_index + 1]
    intermediate_anchor = anchors.get(intermediate)
    if intermediate_anchor is None:
        return None
    return _connector_target_side_entry_y(intermediate_anchor)


def _connector_block_obstacles(
    anchors: dict[int, _RenderAnchor],
    *,
    src: int,
    tgt: int,
    label_obstacles: list[_RenderAnchor],
    graph=None,
    positions: list | None = None,
    link_key: tuple[int, int] | None = None,
) -> list[_RenderAnchor]:
    """Every non-endpoint tile that a connector must not cut through."""
    excluded = {src, tgt}
    if graph is not None:
        for frame in graph.inline_frames:
            members = set(frame.node_indices)
            if tgt in members and src not in members:
                excluded |= members - {tgt}
                break
    obstacles = [
        anchor for node_index, anchor in anchors.items() if node_index not in excluded
    ] + label_obstacles
    if graph is not None and positions is not None:
        obstacles.extend(
            _inline_frame_bounds_obstacles(
                graph,
                positions,
                src=src,
                tgt=tgt,
                exclude_nodes=excluded,
            )
        )
    return obstacles


def _connector_path_violates_inline_frame_bounds(
    points: list[tuple[float, float]],
    graph,
    positions: list,
    *,
    src: int,
    tgt: int,
) -> tuple[str, str] | None:
    """Return frame id and reason when a segment cuts through a dotted frame interior."""
    last_index = len(points) - 2
    for frame in graph.inline_frames:
        members = set(frame.node_indices)
        if src in members and tgt in members:
            continue
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        for index in range(len(points) - 1):
            if tgt in members and index == last_index:
                continue
            if src in members and index == 0:
                continue
            x1, y1 = points[index]
            x2, y2 = points[index + 1]
            if (
                abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS
                and abs(x1 - x2) > 0.06
                and _path_horizontal_segments_overlap_bounds(
                    [(x1, y1), (x2, y2)],
                    bounds,
                )
            ):
                return frame.frame_id, "horizontal segment crosses dotted frame interior"
            if _segment_crosses_frame_bounds(x1, y1, x2, y2, bounds):
                return frame.frame_id, "segment crosses dotted frame interior"
    return None


def _reroute_path_clearing_inline_frames(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    graph,
    positions: list,
    src: int,
    tgt: int,
    obstacles: list[_RenderAnchor],
    anchors: dict[int, _RenderAnchor] | None = None,
    bus_y: float | None = None,
    gap: float = 0.04,
) -> list[tuple[float, float]] | None:
    """Reroute below or around dotted inline frames when a path cuts through one."""
    if _connector_path_violates_inline_frame_bounds(
        points,
        graph,
        positions,
        src=src,
        tgt=tgt,
    ) is None:
        return None

    bypass_y = None
    gutter_x = None
    anchors = anchors or {}
    for frame in graph.inline_frames:
        members = set(frame.node_indices)
        if src in members and tgt in members:
            continue
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        crosses = _path_horizontal_segments_overlap_bounds(points, bounds)
        if not crosses:
            last_index = len(points) - 2
            for index in range(len(points) - 1):
                if tgt in members and index == last_index:
                    continue
                if src in members and index == 0:
                    continue
                x1, y1 = points[index]
                x2, y2 = points[index + 1]
                if _segment_crosses_frame_bounds(x1, y1, x2, y2, bounds):
                    crosses = True
                    break
        if not crosses:
            continue
        candidate_y = bounds.bottom - CONNECTOR_OBSTACLE_MARGIN
        bypass_y = candidate_y if bypass_y is None else min(bypass_y, candidate_y)
        left_gutter = _inline_frame_connector_gutter_width(
            graph,
            frame,
            side="left",
            anchors=anchors,
        )
        gutter_x = bounds.left - INLINE_FRAME_PAD - left_gutter - CONNECTOR_OBSTACLE_MARGIN

    if bypass_y is None:
        return None

    entry_y = _connector_target_top_entry_y(target, gap=gap)
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    route_y = min(bypass_y, y1 - CONNECTOR_EXIT_STUB)
    if bus_y is not None:
        route_y = min(route_y, bus_y - CONNECTOR_OBSTACLE_MARGIN)
    candidates: list[list[tuple[float, float]]] = [
        [
            (source.cx, y1),
            (source.cx, route_y),
            (target.cx, route_y),
            (target.cx, entry_y),
        ]
    ]
    if gutter_x is not None and abs(gutter_x - source.cx) > 0.06:
        candidates.append(
            [
                (source.cx, y1),
                (source.cx, route_y),
                (gutter_x, route_y),
                (gutter_x, entry_y),
                (target.cx, entry_y),
            ]
        )

    for candidate in candidates:
        candidate = _ensure_orthogonal_connector_path(candidate)
        if not _connector_path_clear_of_blocks(
            candidate,
            source=source,
            target=target,
            obstacles=obstacles,
        ):
            continue
        if _connector_path_violates_inline_frame_bounds(
            candidate,
            graph,
            positions,
            src=src,
            tgt=tgt,
        ) is not None:
            continue
        return candidate
    return None


def _assert_connector_path_clear_of_blocks(
    link_key: tuple[int, int],
    points: list[tuple[float, float]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    stage: str,
) -> None:
    """Raise when a connector crosses any intermediate block after layout."""
    src, tgt = link_key
    source = anchors.get(src)
    target = anchors.get(tgt)
    if source is None or target is None or len(points) < 2:
        return
    obstacles = _connector_block_obstacles(
        anchors,
        src=src,
        tgt=tgt,
        label_obstacles=label_obstacles,
        graph=graph,
        positions=positions,
        link_key=link_key,
    )
    if not _connector_path_clear_of_blocks(
        points,
        source=source,
        target=target,
        obstacles=obstacles,
    ):
        raise RuntimeError(
            f"connector {graph.nodes[src].label!r} -> {graph.nodes[tgt].label!r} "
            f"crosses an intermediate block after {stage}"
        )
    if positions is not None:
        violation = _connector_path_violates_inline_frame_bounds(
            points,
            graph,
            positions,
            src=src,
            tgt=tgt,
        )
        if violation is not None:
            frame_id, reason = violation
            raise RuntimeError(
                f"connector {graph.nodes[src].label!r} -> {graph.nodes[tgt].label!r} "
                f"{reason} ({frame_id!r}) after {stage}"
            )


def _graph_requires_strict_connector_validation(graph) -> bool:
    """All detail graphs get runtime block and connector overlap checks."""
    del graph
    return True


def _assert_detail_link_paths_clear_of_blocks(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    stage: str,
) -> None:
    if not _graph_requires_strict_connector_validation(graph):
        return
    for link_key, points in link_paths.items():
        _assert_connector_path_clear_of_blocks(
            link_key,
            points,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            stage=stage,
        )


def _path_penetrates_obstacle_tiles(
    points: list[tuple[float, float]],
    obstacles: list[_RenderAnchor],
    *,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> bool:
    """True when any segment cuts through a tile interior (not just its bbox)."""
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        for obstacle in obstacles:
            if _segment_penetrates_anchor(x1, y1, x2, y2, obstacle, margin=margin):
                return True
    return False


def _ensure_orthogonal_connector_path(
    points: list[tuple[float, float]],
    *,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> list[tuple[float, float]]:
    """Insert axis-aligned legs so no rendered segment is diagonal."""
    if len(points) < 2:
        return points
    fixed: list[tuple[float, float]] = [points[0]]
    for x1, y1 in points[1:]:
        x0, y0 = fixed[-1]
        if abs(x0 - x1) > eps and abs(y0 - y1) > eps:
            fixed.append((x0, y1))
        fixed.append((x1, y1))
    return _dedupe_polyline_points(fixed, eps=eps)


def _connector_uses_target_side_entry(
    target: _RenderAnchor,
    points: list[tuple[float, float]],
    *,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> bool:
    """True when a path terminates on the target's left or right edge."""
    if len(points) < 2:
        return False
    end_x, end_y = points[-1]
    side_y = _connector_target_side_entry_y(target)
    if abs(end_y - side_y) > eps:
        return False
    if abs(end_x - target.left) <= eps or abs(end_x - target.right) <= eps:
        return True
    return abs(end_x - (target.cx - MERGE_RADIUS)) <= eps or abs(
        end_x - (target.cx + MERGE_RADIUS)
    ) <= eps


def _snap_connector_path_endpoints(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    link_key: tuple[int, int],
    graph,
) -> list[tuple[float, float]]:
    """Keep rendered connectors flush with their source/target borders."""
    if len(points) < 2:
        return points
    snapped = list(points)
    nodes = getattr(graph, "nodes", ())
    side_entry_links = getattr(graph, "side_entry_links", set())
    inline_binary_operand_links = getattr(graph, "inline_binary_operand_links", set())
    links = getattr(graph, "links", ())
    if link_key in inline_binary_operand_links:
        y_stub = _connector_source_bottom_exit_y(source) - CONNECTOR_EXIT_STUB
        snapped[0] = (source.cx, y_stub)
        entry_x = target.left if snapped[-2][0] <= target.cx else target.right
        snapped[-1] = (entry_x, _connector_target_side_entry_y(target))
    elif link_key in side_entry_links:
        snapped[0] = (source.cx, _connector_source_bottom_exit_y(source))
        tgt = link_key[1]
        if tgt < len(nodes) and _is_combine_synthetic(nodes[tgt].synthetic):
            combine_cy = _combine_center_y(target)
            entry_x = _side_entry_combine_entry_x(source, target)
            snapped[-1] = (entry_x, combine_cy)
    elif _connector_uses_target_side_entry(target, snapped):
        snapped[0] = (source.cx, _connector_source_bottom_exit_y(source))
    else:
        src = link_key[0]
        if src < len(nodes) and _is_combine_synthetic(nodes[src].synthetic):
            snapped[0] = (source.cx, _combine_bottom_exit_y(source))
        else:
            snapped[0] = (source.cx, _connector_source_bottom_exit_y(source))
        entry_x = snapped[-1][0]
        tgt = link_key[1]
        if tgt < len(nodes) and _is_combine_synthetic(nodes[tgt].synthetic):
            has_side_incoming = any(
                (side_src, tgt) in side_entry_links
                or (side_src, tgt) in inline_binary_operand_links
                for side_src, dst in links
                if dst == tgt
            )
            if has_side_incoming and link_key not in side_entry_links:
                entry_y = _combine_top_entry_y(target)
            else:
                entry_y = _connector_target_top_entry_y(target)
        else:
            entry_y = _connector_target_top_entry_y(target)
        if abs(entry_x - target.cx) < PARALLEL_CONNECTOR_COORD_EPS:
            entry_x = target.cx
        snapped[-1] = (entry_x, entry_y)
    return _ensure_orthogonal_connector_path(snapped)


def _finalize_inline_bypass_spine_tees(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    positions: list,
    inline_binary_bus_x: dict[tuple[int, int], float],
) -> None:
    """Rebuild bypass connectors so they branch from the main spine at skipped tiles."""
    for src, tgt in graph.inline_binary_operand_links:
        link_key = (src, tgt)
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue
        frame = _inline_frame_for_nodes(graph, src, tgt)
        if frame is None:
            continue
        tee_y = _inline_frame_bypass_tee_y(graph, frame, src, tgt, anchors)
        if tee_y is None:
            continue
        frame_bounds = _inline_frame_draw_bounds(frame, positions, graph)
        link_paths[link_key] = _inline_binary_side_entry_connector_points(
            source,
            target,
            bus_x=inline_binary_bus_x.get(link_key),
            frame_bounds=frame_bounds,
            tee_y=tee_y,
        )


def _connector_path_clear_of_blocks(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
) -> bool:
    """Runtime guard: no segment may pass through an unrelated block."""
    if len(points) < 2:
        return True
    if _path_penetrates_obstacle_tiles(points, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN):
        return False
    return not _path_penetrates_attached_boxes(points, source, target)


def _input_to_inline_frame_top_member_route(
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    graph,
    positions: list,
    *,
    src: int,
    tgt: int,
    gap: float = 0.04,
) -> list[tuple[float, float]] | None:
    """Route input to the top tile of a dotted frame above the frame envelope."""
    from visualizer.computation_graph import SYNTHETIC_INPUT

    if graph.nodes[src].synthetic != SYNTHETIC_INPUT:
        return None
    target_frame = next(
        (frame for frame in graph.inline_frames if tgt in frame.node_indices),
        None,
    )
    if target_frame is None or not target_frame.node_indices:
        return None
    if target_frame.node_indices[0] != tgt:
        return None
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    y2 = _connector_target_top_entry_y(target, gap=gap)
    bypass_x = max(source.right, target.right) + CONNECTOR_OBSTACLE_MARGIN
    bypass_x = _right_bypass_x_clearing_horizontal_segment(
        min(source.cx, target.cx),
        y1,
        obstacles,
        initial_bypass_x=bypass_x,
    )
    return _ensure_orthogonal_connector_path(
        [
            (source.cx, y1),
            (bypass_x, y1),
            (target.cx, y1),
            (target.cx, y2),
        ]
    )


def _horizontal_departure_side_bypass_route(
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    bus_y: float | None = None,
    gap: float = 0.04,
) -> list[tuple[float, float]] | None:
    """Leave the source column horizontally before dropping to the target."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    y_stub = y1 - CONNECTOR_EXIT_STUB
    entry_y = _connector_target_top_entry_y(target, gap=gap)
    side_candidates = (
        max(source.right, target.right) + gap + 0.10,
        min(source.left, target.left) - gap - 0.10,
    )
    route_candidates: list[float] = []
    if bus_y is not None:
        route_candidates.append(bus_y)
    route_candidates.extend(
        entry_y + offset for offset in (0.07, 0.0, 0.15, 0.22, 0.30)
    )
    for step in range(16):
        route_candidates.append(y1 - 0.08 * step)
        route_candidates.append(y_stub - 0.08 * step)
    seen: set[float] = set()
    ordered_route_ys: list[float] = []
    for route_y in route_candidates:
        bucket = round(route_y, 4)
        if bucket in seen:
            continue
        seen.add(bucket)
        ordered_route_ys.append(route_y)
    departure_levels = (y1, y_stub)
    for bypass_x in side_candidates:
        if bypass_x >= source.cx:
            bypass_x = _right_bypass_x_clearing_horizontal_segment(
                min(source.cx, target.cx),
                y1,
                obstacles,
                initial_bypass_x=bypass_x,
            )
            bypass_x = _right_bypass_x_clearing_vertical_segment(
                min(entry_y, y_stub),
                max(y1, entry_y),
                obstacles,
                initial_bypass_x=bypass_x,
            )
        else:
            bypass_x = _left_bypass_x_clearing_horizontal_segment(
                max(source.cx, target.cx),
                y1,
                obstacles,
                initial_bypass_x=bypass_x,
            )
        for depart_y in departure_levels:
            for route_y in ordered_route_ys:
                if abs(depart_y - y_stub) <= PARALLEL_CONNECTOR_COORD_EPS:
                    prefix = [
                        (source.cx, y1),
                        (source.cx, y_stub),
                        (bypass_x, y_stub),
                    ]
                else:
                    prefix = [(source.cx, y1), (bypass_x, y1)]
                points = _ensure_orthogonal_connector_path(
                    [
                        *prefix,
                        (bypass_x, route_y),
                        (target.cx, route_y),
                        (target.cx, entry_y),
                    ]
                )
                if (
                    not _path_hits_obstacles(
                        points,
                        obstacles,
                        margin=CONNECTOR_OBSTACLE_MARGIN,
                    )
                    and not _path_penetrates_attached_boxes(points, source, target)
                ):
                    return points
    return None


def _reroute_connector_path_clearing_blocks(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    bus_y: float | None = None,
    bus_near: str = "target",
    graph=None,
    positions: list | None = None,
    link_key: tuple[int, int] | None = None,
    source_bus: dict[int, float] | None = None,
    merge_link_bus: dict[tuple[int, int], float] | None = None,
) -> list[tuple[float, float]]:
    """Last-resort reroute when a laid-out connector still crosses a block."""
    from visualizer.ast_analyze import is_compact_combine_label

    is_side_labeled_combine = (
        graph is not None
        and link_key is not None
        and positions is not None
        and link_key in getattr(graph, "side_entry_links", set())
        and _is_combine_synthetic(positions[link_key[1]].spec.synthetic)
        and not is_compact_combine_label(positions[link_key[1]].spec.label)
    )
    if (
        graph is not None
        and link_key is not None
        and link_key in getattr(graph, "side_entry_links", set())
        and _connector_path_clear_of_blocks(
            points,
            source=source,
            target=target,
            obstacles=obstacles,
        )
        and (
            positions is None
            or _connector_path_violates_inline_frame_bounds(
                points,
                graph,
                positions,
                src=link_key[0],
                tgt=link_key[1],
            )
            is None
        )
    ):
        return points
    if _connector_path_clear_of_blocks(
        points,
        source=source,
        target=target,
        obstacles=obstacles,
    ):
        if (
            graph is not None
            and positions is not None
            and link_key is not None
            and _connector_path_violates_inline_frame_bounds(
                points,
                graph,
                positions,
                src=link_key[0],
                tgt=link_key[1],
            )
            is None
        ):
            return points
        if graph is None or positions is None or link_key is None:
            return points

    frame_violation = None
    if graph is not None and positions is not None and link_key is not None:
        frame_violation = _connector_path_violates_inline_frame_bounds(
            points,
            graph,
            positions,
            src=link_key[0],
            tgt=link_key[1],
        )
        if frame_violation is None and _connector_path_clear_of_blocks(
            points,
            source=source,
            target=target,
            obstacles=obstacles,
        ):
            return points

    candidates: list[list[tuple[float, float]]] = []
    if graph is not None and positions is not None and link_key is not None:
        top_member_route = _input_to_inline_frame_top_member_route(
            source,
            target,
            obstacles,
            graph,
            positions,
            src=link_key[0],
            tgt=link_key[1],
        )
        if top_member_route is not None:
            candidates.append(top_member_route)
        if link_key in getattr(graph, "side_entry_links", set()):
            tgt_spec = positions[link_key[1]].spec
            if _is_combine_synthetic(tgt_spec.synthetic):
                combine_cy = _combine_center_y(target)
                entry_x = _side_entry_combine_entry_x(source, target)
                shared_frame = _inline_frame_for_nodes(graph, link_key[0], link_key[1])
                frame_bounds = (
                    _inline_frame_draw_bounds(shared_frame, positions, graph)
                    if shared_frame is not None
                    else None
                )
                candidates.insert(
                    0,
                    _side_entry_combine_l_route(source, target),
                )
                candidates.append(
                    _side_entry_combine_connector_points(
                        source,
                        target.cx,
                        combine_cy,
                        frame_bounds=frame_bounds,
                        obstacles=obstacles,
                        target=target,
                    )
                )
                crossing_frames = [
                    frame
                    for frame in graph.inline_frames
                    if not (
                        link_key[0] in frame.node_indices
                        and link_key[1] in frame.node_indices
                    )
                    and _path_horizontal_segments_overlap_bounds(
                        points,
                        _inline_frame_draw_bounds(frame, positions, graph),
                    )
                ]
                if shared_frame is not None or crossing_frames:
                    frame_list = (
                        [shared_frame]
                        if shared_frame is not None
                        else crossing_frames
                    )
                    below_y = min(
                        _inline_frame_draw_bounds(frame, positions, graph).bottom
                        - CONNECTOR_OBSTACLE_MARGIN
                        for frame in frame_list
                    )
                    if crossing_frames:
                        below_y -= PARALLEL_CONNECTOR_CHANNEL_GAP
                    below_frame_route = _ensure_orthogonal_connector_path(
                        [
                            (source.cx, _connector_source_bottom_exit_y(source)),
                            (source.cx, below_y),
                            (entry_x, below_y),
                            (entry_x, combine_cy),
                        ]
                    )
                    if is_side_labeled_combine:
                        candidates.insert(0, below_frame_route)
                    else:
                        candidates.append(below_frame_route)
        frame_route = _reroute_path_clearing_inline_frames(
            points,
            source=source,
            target=target,
            graph=graph,
            positions=positions,
            src=link_key[0],
            tgt=link_key[1],
            obstacles=obstacles,
            anchors=None,
            bus_y=bus_y,
        )
        if frame_route is not None:
            candidates.append(frame_route)
    if (
        graph is not None
        and link_key is not None
        and source_bus is not None
        and merge_link_bus is not None
    ):
        tee_y = source_bus.get(link_key[0])
        leg_bus = merge_link_bus.get(link_key)
        if (
            tee_y is not None
            and leg_bus is not None
            and leg_bus < tee_y - PARALLEL_CONNECTOR_COORD_EPS
        ):
            candidates.insert(
                0,
                _fanout_tee_then_entry_column_points(
                    source,
                    target,
                    target.cx,
                    tee_y=tee_y,
                    bus_y=leg_bus,
                ),
            )
    side_bypass = None
    if not is_side_labeled_combine:
        side_bypass = _horizontal_departure_side_bypass_route(
            source,
            target,
            obstacles,
            bus_y=bus_y,
        )
    if side_bypass is not None:
        candidates.insert(0, side_bypass)
    if abs(source.cx - target.cx) < 0.06 and not is_side_labeled_combine:
        candidates.append(
            _same_column_side_gutter_detour(source, target, obstacles)
        )
    if not is_side_labeled_combine:
        candidates.append(
            _orthogonal_path(
                source,
                target,
                obstacles,
                bus_near=bus_near,
                bus_y=bus_y,
            )
        )
    if abs(source.cx - target.cx) < 0.06 and not is_side_labeled_combine:
        candidates.append(
            _same_column_top_entry_detour(source, target, obstacles, bus_y=bus_y)
        )

    for candidate in candidates:
        if not _connector_path_clear_of_blocks(
            candidate,
            source=source,
            target=target,
            obstacles=obstacles,
        ):
            continue
        if (
            graph is not None
            and positions is not None
            and link_key is not None
            and _connector_path_violates_inline_frame_bounds(
                candidate,
                graph,
                positions,
                src=link_key[0],
                tgt=link_key[1],
            )
            is not None
        ):
            continue
        return candidate
    if frame_violation is None:
        return points
    return points


def _reroute_detail_link_paths_clearing_blocks(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float] | None = None,
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Reroute every connector that still crosses an intermediate block."""
    merge_link_bus = merge_link_bus or {}
    rerouted: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for link_key, points in link_paths.items():
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None or len(points) < 2:
            rerouted[link_key] = points
            continue
        obstacles = _connector_block_obstacles(
            anchors,
            src=src,
            tgt=tgt,
            label_obstacles=label_obstacles,
            graph=graph,
            positions=positions,
            link_key=link_key,
        )
        bus_y = _link_routing_bus_y(
            link_key,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
        )
        rerouted[link_key] = _reroute_connector_path_clearing_blocks(
            points,
            source=source,
            target=target,
            obstacles=obstacles,
            bus_y=bus_y,
            graph=graph,
            positions=positions,
            link_key=link_key,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
        )
    return rerouted


def _connector_leave_source_before_horizontal(
    source: _RenderAnchor,
    target: _RenderAnchor,
    bus_x: float,
    *,
    gap: float = 0.04,
) -> list[tuple[float, float]]:
    """Drop below the source tile before the first horizontal bypass segment."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    x1 = source.cx
    y_stub = y1 - CONNECTOR_EXIT_STUB
    return [(x1, y1), (x1, y_stub), (bus_x, y_stub)]


def _connector_leave_source_to_side(
    source: _RenderAnchor,
    target: _RenderAnchor,
    side_x: float,
    *,
    gap: float = 0.04,
) -> list[tuple[float, float]]:
    """Leave the source tile without cutting through a stacked target below."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    x1 = source.cx
    y_stub = y1 - CONNECTOR_EXIT_STUB
    if _vertical_segment_crosses_anchor(
        x1,
        y1,
        y_stub,
        target,
        margin=CONNECTOR_ATTACHED_BOX_MARGIN,
    ):
        return [(x1, y1), (x1, y_stub), (side_x, y_stub)]
    return [(x1, y1), (x1, y_stub), (side_x, y_stub)]


def _segment_is_source_departure_channel(
    source: _RenderAnchor,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    margin: float = CONNECTOR_ATTACHED_BOX_MARGIN,
) -> bool:
    """True for the initial vertical channel leaving a source tile."""
    if abs(x1 - x2) > PARALLEL_CONNECTOR_COORD_EPS:
        return False
    if abs(x1 - source.cx) > PARALLEL_CONNECTOR_COORD_EPS:
        return False
    return max(y1, y2) >= source.bottom - margin


def _path_penetrates_attached_boxes(
    points: list[tuple[float, float]],
    source: _RenderAnchor,
    target: _RenderAnchor,
    *,
    margin: float = CONNECTOR_ATTACHED_BOX_MARGIN,
) -> bool:
    """Return True when any segment cuts through its source or target tile."""
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        if (
            index > 0
            and not _segment_is_source_departure_channel(source, x1, y1, x2, y2, margin=margin)
            and _segment_penetrates_anchor(x1, y1, x2, y2, source, margin=margin)
        ):
            return True
        if index < len(points) - 2 and _segment_penetrates_anchor(
            x1, y1, x2, y2, target, margin=margin
        ):
            return True
    return False


def _same_column_side_gutter_detour(
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    gap: float = 0.04,
    entry_y: float | None = None,
) -> list[tuple[float, float]]:
    """Route via a side gutter when stacked tiles leave no horizontal channel."""
    x2 = target.cx
    y2 = entry_y if entry_y is not None else _connector_target_top_entry_y(target, gap=gap)
    side_candidates = (
        max(source.right, target.right) + gap + 0.10,
        min(source.left, target.left) - gap - 0.10,
    )
    for side_x in side_candidates:
        points = [
            *_connector_leave_source_to_side(source, target, side_x, gap=gap),
            (side_x, y2),
            (x2, y2),
        ]
        if (
            not _path_hits_obstacles(points, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN)
            and not _path_penetrates_attached_boxes(points, source, target)
        ):
            return points
    side_x = side_candidates[0]
    return [
        *_connector_leave_source_to_side(source, target, side_x, gap=gap),
        (side_x, y2),
        (x2, y2),
    ]


def _same_column_top_entry_detour(
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    gap: float = 0.04,
    bus_y: float | None = None,
) -> list[tuple[float, float]]:
    """Route around a target when source and target share a column."""
    x2 = target.cx
    y2 = _connector_target_top_entry_y(target, gap=gap)
    min_bus_y = _connector_min_bus_y_above_target(target, gap=gap)
    max_approach_y = _connector_source_bottom_exit_y(source, gap=gap) - CONNECTOR_OBSTACLE_MARGIN
    approach_y = max(min_bus_y, bus_y if bus_y is not None else y2 + 0.08)
    if approach_y > max_approach_y + 1e-6:
        return _same_column_side_gutter_detour(
            source,
            target,
            obstacles,
            gap=gap,
            entry_y=y2,
        )
    approach_y = min(approach_y, max_approach_y)
    side_candidates = (
        max(source.right, target.right) + gap + 0.10,
        min(source.left, target.left) - gap - 0.10,
    )
    for side_x in side_candidates:
        points = [
            *_connector_leave_source_to_side(source, target, side_x, gap=gap),
            (side_x, approach_y),
            (x2, approach_y),
            (x2, y2),
        ]
        if (
            not _path_hits_obstacles(points, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN)
            and not _path_penetrates_attached_boxes(points, source, target)
        ):
            return points
    return _same_column_side_gutter_detour(
        source,
        target,
        obstacles,
        gap=gap,
        entry_y=y2,
    )


def _path_hits_obstacles(
    points: list[tuple[float, float]],
    obstacles: list[_RenderAnchor],
    *,
    margin: float = 0.05,
) -> bool:
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        if _segment_hits_obstacle(x1, y1, x2, y2, obstacles, margin=margin):
            return True
    return False


def _connector_exit_stub_y(source_bottom: float, *, gap: float = 0.04) -> tuple[float, float]:
    """Return (y_at_bottom_edge, y_after_downward_stub) below a tile."""
    del gap
    y1 = source_bottom
    return y1, y1 - CONNECTOR_EXIT_STUB


def _same_column_straight_connector_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    *,
    gap: float = 0.04,
) -> list[tuple[float, float]]:
    """Direct vertical feed when stacked tiles leave a clear channel."""
    return [
        (source.cx, _connector_source_bottom_exit_y(source, gap=gap)),
        (target.cx, _connector_target_top_entry_y(target, gap=gap)),
    ]


def _orthogonal_path(
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    gap: float = 0.04,
    bus_near: str = "target",
    bus_y: float | None = None,
    graph=None,
    positions: list | None = None,
) -> list[tuple[float, float]]:
    """Build a Manhattan path from source bottom to target top."""
    y1, y_stub = _connector_exit_stub_y(source.bottom, gap=gap)
    x1 = source.cx
    x2, y2 = target.cx, _connector_target_top_entry_y(target, gap=gap)
    min_bus_y = _connector_min_bus_y_above_target(target, gap=gap)

    if abs(x1 - x2) < 0.06:
        if bus_y is None:
            straight = _same_column_straight_connector_points(source, target, gap=gap)
            if (
                not _path_hits_obstacles(straight, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN)
                and not _path_penetrates_attached_boxes(straight, source, target)
            ):
                return straight

            stubbed = [(x1, y1), (x1, y_stub), (x2, y2)]
            if (
                not _path_hits_obstacles(stubbed, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN)
                and not _path_penetrates_attached_boxes(stubbed, source, target)
            ):
                return stubbed

        needs_detour = _vertical_segment_crosses_anchor(
            x1, min(y_stub, y2), max(y_stub, y2), target, margin=CONNECTOR_ATTACHED_BOX_MARGIN
        ) or _vertical_segment_crosses_anchor(
            x1, min(y_stub, y2), max(y_stub, y2), source, margin=CONNECTOR_ATTACHED_BOX_MARGIN
        )
        if bus_y is None and (needs_detour or source.bottom > _connector_target_top_entry_y(target, gap=gap)):
            return _same_column_top_entry_detour(
                source,
                target,
                obstacles,
                gap=gap,
                bus_y=bus_y,
            )
        if bus_y is None:
            for offset in (0.12, -0.12, 0.24, -0.24):
                detour = [
                    (x1, y1),
                    (x1, y_stub),
                    (x1 + offset, y_stub),
                    (x1 + offset, y2),
                    (x2, y2),
                ]
                if (
                    not _path_hits_obstacles(detour, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN)
                    and not _path_penetrates_attached_boxes(detour, source, target)
                ):
                    return detour
            return _same_column_straight_connector_points(source, target, gap=gap)
        if bus_near == "source":
            bus_y = max(min_bus_y, bus_y)
            aligned = [(x1, y1), (x1, bus_y), (x2, y2)]
        else:
            bus_y = max(min_bus_y, bus_y if bus_near == "source" else min(y_stub, bus_y))
            aligned = [(x1, y1), (x1, y_stub), (x1, bus_y), (x2, bus_y), (x2, y2)]
        if (
            not _path_hits_obstacles(aligned, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN)
            and not _path_penetrates_attached_boxes(aligned, source, target)
        ):
            return aligned
        return _same_column_top_entry_detour(source, target, obstacles, gap=gap, bus_y=bus_y)

    if bus_y is None:
        channel = max(y_stub - y2, gap * 4)
        if bus_near == "source":
            bus_y = y_stub - min(0.10, channel * 0.35)
        else:
            bus_y = y2 + min(0.10, channel * 0.25)
        bus_y = min(y_stub - 0.02, max(min_bus_y, bus_y))
    else:
        if (
            bus_near == "source"
            and bus_y is not None
            and bus_y < min_bus_y - PARALLEL_CONNECTOR_COORD_EPS
        ):
            low_bus_points = [(x1, y1), (x1, bus_y), (x2, bus_y), (x2, y2)]
            if (
                not _path_hits_obstacles(
                    low_bus_points,
                    obstacles,
                    margin=CONNECTOR_OBSTACLE_MARGIN,
                )
                and not _path_penetrates_attached_boxes(
                    low_bus_points,
                    source,
                    target,
                )
            ):
                return low_bus_points
            if graph is not None and positions is not None:
                gutter = _input_fanout_gutter_route(
                    source,
                    target,
                    obstacles,
                    graph=graph,
                    positions=positions,
                    bus_y=bus_y,
                )
                if gutter is not None:
                    return gutter
        bus_y = max(min_bus_y, bus_y if bus_near == "source" else min(y_stub, bus_y))

    if bus_near == "source":
        points = [(x1, y1), (x1, bus_y), (x2, bus_y), (x2, y2)]
    else:
        points = [(x1, y1), (x1, y_stub), (x1, bus_y), (x2, bus_y), (x2, y2)]
    for _ in range(6):
        if (
            not _path_hits_obstacles(points, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN)
            and not _path_penetrates_attached_boxes(points, source, target)
        ):
            return points
        bus_y = max(min_bus_y, bus_y - 0.08)
        if bus_near == "source":
            points = [(x1, y1), (x1, bus_y), (x2, bus_y), (x2, y2)]
        else:
            points = [(x1, y1), (x1, y_stub), (x1, bus_y), (x2, bus_y), (x2, y2)]
    if bus_near == "source" and graph is not None and positions is not None:
        gutter = _input_fanout_gutter_route(
            source,
            target,
            obstacles,
            graph=graph,
            positions=positions,
            bus_y=bus_y if bus_y is not None else min_bus_y - CONNECTOR_OBSTACLE_MARGIN,
        )
        if gutter is not None:
            return gutter
    return points


def _input_fanout_gutter_route(
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    graph,
    positions: list,
    bus_y: float,
    gap: float = 0.04,
) -> list[tuple[float, float]] | None:
    """Route dense input fan-out below dotted frames when the shared bus cannot lift."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    y2 = _connector_target_top_entry_y(target, gap=gap)
    gutter_y = bus_y
    gutter_x = min(source.left, target.left) - CONNECTOR_OBSTACLE_MARGIN
    for frame in graph.inline_frames:
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        x_left = min(source.cx, target.cx)
        x_right = max(source.cx, target.cx)
        if bounds.right < x_left - CONNECTOR_OBSTACLE_MARGIN or bounds.left > x_right + CONNECTOR_OBSTACLE_MARGIN:
            continue
        gutter_y = min(gutter_y, bounds.bottom - CONNECTOR_OBSTACLE_MARGIN - CONNECTOR_EXIT_STUB)
        left_gutter = _inline_frame_connector_gutter_width(graph, frame, side="left", anchors={})
        gutter_x = min(gutter_x, bounds.left - INLINE_FRAME_PAD - left_gutter - CONNECTOR_OBSTACLE_MARGIN)

    candidates = [
        [(source.cx, y1), (source.cx, gutter_y), (target.cx, gutter_y), (target.cx, y2)],
    ]
    if abs(gutter_x - source.cx) > 0.06:
        candidates.append(
            [
                (source.cx, y1),
                (source.cx, gutter_y),
                (gutter_x, gutter_y),
                (gutter_x, y2),
                (target.cx, y2),
            ]
        )
    for candidate in candidates:
        candidate = _ensure_orthogonal_connector_path(candidate)
        if _path_hits_obstacles(candidate, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN):
            continue
        if _path_penetrates_attached_boxes(candidate, source, target):
            continue
        if any(
            _path_horizontal_segments_overlap_bounds(candidate, _inline_frame_draw_bounds(frame, positions, graph))
            for frame in graph.inline_frames
        ):
            continue
        return candidate
    return None


def _compute_shared_target_bus_y(
    sources: list[_RenderAnchor],
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    gap: float = 0.04,
) -> float:
    """Pick one horizontal channel so converging links enter the target vertically."""
    y2 = _connector_target_top_entry_y(target, gap=gap)
    y1_min = min(_connector_source_bottom_exit_y(source, gap=gap) for source in sources)
    channel = max(y1_min - y2, gap * 4)
    bus_y = y2 + min(0.10, channel * 0.25)
    bus_y = min(y1_min - 0.02, max(y2 + 0.02, bus_y))
    min_bus_y = _connector_min_bus_y_above_target(target, gap=gap)
    bus_y = max(min_bus_y, bus_y)
    for _ in range(8):
        if all(
            not _path_hits_obstacles(
                [
                    (source.cx, _connector_source_bottom_exit_y(source, gap=gap)),
                    (source.cx, bus_y),
                    (target.cx, bus_y),
                    (target.cx, y2),
                ],
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            )
            for source in sources
        ):
            return bus_y
        bus_y = max(min_bus_y, bus_y - 0.08)
    return max(min_bus_y, bus_y)


def _clamp_bus_y_clearing_obstacles_on_span(
    bus_y: float,
    obstacles: list[_RenderAnchor],
    *,
    x_left: float,
    x_right: float,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> float:
    """Lower a bus so its horizontal span stays below crossing tiles."""
    cleared = bus_y
    for obstacle in obstacles:
        if obstacle.right < x_left - margin or obstacle.left > x_right + margin:
            continue
        if obstacle.bottom - margin < cleared:
            cleared = min(cleared, obstacle.bottom - margin - CONNECTOR_EXIT_STUB)
    return cleared


def _clamp_bus_y_clearing_inline_frames(
    bus_y: float,
    *,
    graph,
    positions: list,
    x_left: float,
    x_right: float,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> float:
    """Lower a horizontal bus so it sits below dotted-frame interiors it would span."""
    cleared = bus_y
    for frame in graph.inline_frames:
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        if x_right < bounds.left - margin or x_left > bounds.right + margin:
            continue
        if bus_y >= bounds.top - margin:
            continue
        cleared = min(cleared, bounds.bottom - margin - CONNECTOR_EXIT_STUB)
    return cleared


def _compute_shared_source_bus_y(
    source: _RenderAnchor,
    targets: list[_RenderAnchor],
    obstacles: list[_RenderAnchor],
    *,
    gap: float = 0.04,
    graph=None,
    positions: list | None = None,
) -> float:
    """Pick one horizontal channel so fan-out links leave the source vertically aligned."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    y2_max = max(_connector_target_top_entry_y(target, gap=gap) for target in targets)
    channel = max(y1 - y2_max, gap * 4)
    bus_y = y1 - min(0.10, channel * 0.35)
    bus_y = min(y1 - 0.02, max(y2_max + 0.02, bus_y))
    for _ in range(8):
        if all(
            not _path_hits_obstacles(
                [
                    (source.cx, y1),
                    (source.cx, bus_y),
                    (target.cx, bus_y),
                    (target.cx, _connector_target_top_entry_y(target, gap=gap)),
                ],
                obstacles,
            )
            for target in targets
        ):
            if graph is not None and positions is not None:
                xs = [source.cx, *(target.cx for target in targets)]
                return _clamp_bus_y_clearing_inline_frames(
                    bus_y,
                    graph=graph,
                    positions=positions,
                    x_left=min(xs),
                    x_right=max(xs),
                )
            return bus_y
        bus_y -= 0.08
    if graph is not None and positions is not None:
        xs = [source.cx, *(target.cx for target in targets)]
        return _clamp_bus_y_clearing_inline_frames(
            bus_y,
            graph=graph,
            positions=positions,
            x_left=min(xs),
            x_right=max(xs),
        )
    return bus_y


def _compute_fanout_split_tee_y(
    source: _RenderAnchor,
    lowest_merge_bus_y: float,
    *,
    gap: float = 0.04,
) -> float:
    """Place a fan-out tee midway along short source-to-merge-bus drops."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    channel = max(y1 - lowest_merge_bus_y, gap * 2)
    if channel <= FANOUT_SHORT_CHANNEL_MAX:
        tee_y = y1 - channel * FANOUT_SHORT_TEE_FRACTION
    else:
        tee_y = y1 - min(0.10, channel * 0.35)
    return max(tee_y, lowest_merge_bus_y + gap * 2)


def _effective_source_bus_y(
    source: _RenderAnchor,
    targets: list[_RenderAnchor],
    proposed_bus_y: float,
    *,
    gap: float = 0.04,
) -> float:
    """Return the horizontal tee Y that fan-out routing will actually use."""
    if not targets:
        return proposed_bus_y
    min_bus_y = max(_connector_min_bus_y_above_target(target, gap=gap) for target in targets)
    _, y_stub = _connector_exit_stub_y(source.bottom, gap=gap)
    bus_y = max(min_bus_y, proposed_bus_y)
    return max(min_bus_y, min(y_stub - 0.02, bus_y))


def _fanout_lowest_target_merge_bus_y(
    graph,
    src: int,
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
) -> float | None:
    """Lowest shared merge-bus Y among a source's target-bus fan-out legs."""
    main_links = _fanout_links_excluding_bypasses(graph, outgoing.get(src, []))
    merge_bus_ys = [target_bus[tgt] for _, tgt in main_links if tgt in target_bus]
    if not merge_bus_ys:
        return None
    return min(merge_bus_ys)


def _should_use_shared_connector_bus(link_count: int) -> bool:
    """Only fan-ins/fan-outs with 4+ main-path links share a merge bus."""
    return link_count >= SHARED_CONNECTOR_BUS_MIN_LINKS


def _should_use_shared_source_bus(link_count: int) -> bool:
    """Fan-outs with 2+ main-path links share a source bus for clean tees."""
    return link_count >= SHARED_SOURCE_BUS_MIN_LINKS


def _fanout_source_bus_y(
    graph,
    src: int,
    link_key: tuple[int, int],
    *,
    positions: list,
    outgoing: dict[int, list[tuple[int, int]]],
    source_bus: dict[int, float],
    target_bus: dict[int, float] | None = None,
    channel_gap: float = 0.07,
) -> float | None:
    """Per-link horizontal level when several outputs leave the same source."""
    base = source_bus.get(src)
    if base is None:
        return None
    main_links = _fanout_links_excluding_bypasses(graph, outgoing.get(src, []))
    if len(main_links) < SHARED_SOURCE_BUS_MIN_LINKS or link_key not in main_links:
        return base
    target_bus = target_bus or {}
    if not _source_fanout_splits_before_target_bus(
        graph,
        src,
        outgoing,
        target_bus,
    ):
        return base
    _, tgt = link_key
    if tgt not in target_bus:
        return base
    ordered = sorted(
        main_links,
        key=lambda link: (-positions[link[1]].top_y, positions[link[1]].cx, link[1]),
    )
    index = ordered.index(link_key)
    return base - index * channel_gap


def _source_fanout_splits_before_target_bus(
    graph,
    src: int,
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
) -> bool:
    """True when one fan-out leg joins a target merge bus and another branches earlier."""
    main_links = _fanout_links_excluding_bypasses(graph, outgoing.get(src, []))
    if len(main_links) < SHARED_SOURCE_BUS_MIN_LINKS:
        return False
    has_target_bus_leg = any(tgt in target_bus for _, tgt in main_links)
    has_other_leg = any(tgt not in target_bus for _, tgt in main_links)
    return has_target_bus_leg and has_other_leg


def _connector_path_respects_tee_before_bus_join(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    tee_y: float,
    merge_bus_y: float,
) -> bool:
    """Fan-out legs must tee on the source column before joining a lower merge bus."""
    if merge_bus_y >= tee_y - PARALLEL_CONNECTOR_COORD_EPS:
        return True
    tee_index: int | None = None
    merge_index: int | None = None
    for index, (x, y) in enumerate(points):
        if abs(x - source.cx) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if abs(y - tee_y) <= PARALLEL_CONNECTOR_COORD_EPS:
            tee_index = index if tee_index is None else tee_index
        if abs(y - merge_bus_y) <= PARALLEL_CONNECTOR_COORD_EPS:
            merge_index = index if merge_index is None else merge_index
    if merge_index is None:
        return True
    if tee_index is None:
        return False
    return tee_index < merge_index


def _assert_connector_tees_precede_bus_joins(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    outgoing: dict[int, list[tuple[int, int]]],
    source_bus: dict[int, float],
    target_bus: dict[int, float],
    stage: str,
) -> None:
    """Runtime check: fan-out tees occur before any leg joins a lower merge bus."""
    offenders: list[str] = []
    for src, tee_y in source_bus.items():
        if not _source_fanout_splits_before_target_bus(graph, src, outgoing, target_bus):
            continue
        source = anchors.get(src)
        if source is None:
            continue
        for _, tgt in outgoing.get(src, []):
            if tgt not in target_bus:
                continue
            link_key = (src, tgt)
            points = link_paths.get(link_key)
            if points is None or len(points) < 2:
                continue
            merge_bus_y = target_bus[tgt]
            if not _connector_path_respects_tee_before_bus_join(
                points,
                source=source,
                tee_y=tee_y,
                merge_bus_y=merge_bus_y,
            ):
                offenders.append(f"{link_key}@{stage}")
    if offenders:
        raise RuntimeError(
            "connector tee must precede merge-bus join: " + ", ".join(offenders)
        )


def _connector_fanout_branch_tee_y(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
) -> float | None:
    """Return the horizontal tee Y when a fan-out leg branches from the source column."""
    if len(points) < 3:
        return None
    y_exit = _connector_source_bottom_exit_y(source)
    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]
    if abs(x0 - source.cx) > PARALLEL_CONNECTOR_COORD_EPS:
        return None
    if abs(y0 - y_exit) > PARALLEL_CONNECTOR_COORD_EPS:
        return None
    if abs(x1 - source.cx) > PARALLEL_CONNECTOR_COORD_EPS:
        return None
    tee_y = y1
    if abs(y2 - tee_y) <= PARALLEL_CONNECTOR_COORD_EPS and abs(x2 - source.cx) > PARALLEL_CONNECTOR_COORD_EPS:
        return tee_y
    return None


def _assert_shared_fanout_branch_tees_aligned(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    outgoing: dict[int, list[tuple[int, int]]],
    source_bus: dict[int, float],
    target_bus: dict[int, float],
    stage: str,
) -> None:
    """Runtime check: branch-style fan-out legs share one horizontal tee level."""
    offenders: list[str] = []
    for src, tee_y in source_bus.items():
        if _source_fanout_splits_before_target_bus(graph, src, outgoing, target_bus):
            continue
        main_links = _fanout_links_excluding_bypasses(graph, outgoing.get(src, []))
        if len(main_links) < SHARED_SOURCE_BUS_MIN_LINKS:
            continue
        source = anchors.get(src)
        if source is None:
            continue
        branch_tees: list[tuple[tuple[int, int], float]] = []
        for link in main_links:
            points = link_paths.get(link)
            if points is None:
                continue
            branch_tee = _connector_fanout_branch_tee_y(points, source=source)
            if branch_tee is not None:
                branch_tees.append((link, branch_tee))
        if (
            len(branch_tees) != len(main_links)
            or len(branch_tees) < SHARED_SOURCE_BUS_MIN_LINKS
        ):
            continue
        if not all(
            abs(branch_tee - tee_y) <= PARALLEL_CONNECTOR_COORD_EPS
            for _, branch_tee in branch_tees
        ):
            continue
        ref_y = branch_tees[0][1]
        for link, branch_tee in branch_tees:
            if abs(branch_tee - ref_y) > PARALLEL_CONNECTOR_COORD_EPS:
                offenders.append(
                    f"{link} tee={branch_tee:.4f} != shared={ref_y:.4f}@{stage}"
                )
    if offenders:
        raise RuntimeError("fan-out branch tees misaligned: " + ", ".join(offenders))


def _assert_fanout_avoids_input_horizontal_departure(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    outgoing: dict[int, list[tuple[int, int]]],
    source_bus: dict[int, float],
    stage: str,
) -> None:
    """Runtime check: fan-out legs leave the input vertically, not sideways at the bottom."""
    offenders: list[str] = []
    for src in source_bus:
        source = anchors.get(src)
        if source is None:
            continue
        main_links = _fanout_links_excluding_bypasses(graph, outgoing.get(src, []))
        if len(main_links) < SHARED_SOURCE_BUS_MIN_LINKS:
            continue
        y_exit = _connector_source_bottom_exit_y(source)
        branch_by_link = {
            link: _connector_fanout_branch_tee_y(
                link_paths.get(link) or [],
                source=source,
            )
            for link in main_links
        }
        branch_tees = [link for link, tee in branch_by_link.items() if tee is not None]
        if (
            len(branch_tees) != len(main_links)
            or len(branch_tees) < SHARED_SOURCE_BUS_MIN_LINKS
        ):
            continue
        tee_y = source_bus[src]
        if not all(
            abs(branch_by_link[link] - tee_y) <= PARALLEL_CONNECTOR_COORD_EPS
            for link in branch_tees
        ):
            continue
        for link in main_links:
            points = link_paths.get(link)
            if points is None or len(points) < 2:
                continue
            x1, y1 = points[0]
            x2, y2 = points[1]
            if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS:
                continue
            if abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS:
                continue
            if abs(y1 - y_exit) > CONNECTOR_OBSTACLE_MARGIN + PARALLEL_CONNECTOR_COORD_EPS:
                continue
            offenders.append(f"{link} horizontal at input bottom y={y1:.4f}@{stage}")
    if offenders:
        raise RuntimeError(
            "fan-out must not depart horizontally from input: " + ", ".join(offenders)
        )


def _assert_source_bus_clears_fanout_targets(
    source_bus: dict[int, float],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    outgoing: dict[int, list[tuple[int, int]]],
    stage: str,
) -> None:
    """Runtime check: shared source-bus Y clears every fan-out target's min merge level."""
    offenders: list[str] = []
    for src, tee_y in source_bus.items():
        main_links = _fanout_links_excluding_bypasses(graph, outgoing.get(src, []))
        if len(main_links) < SHARED_SOURCE_BUS_MIN_LINKS:
            continue
        for _, tgt in main_links:
            target = anchors.get(tgt)
            if target is None:
                continue
            min_bus = _connector_min_bus_y_above_target(target)
            if tee_y + PARALLEL_CONNECTOR_COORD_EPS < min_bus:
                offenders.append(
                    f"({src},{tgt}) source_bus={tee_y:.4f} < min_bus={min_bus:.4f}@{stage}"
                )
    if offenders:
        raise RuntimeError(
            "source_bus below target clearance: " + ", ".join(offenders)
        )


def _assert_detail_fanout_connector_invariants(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    outgoing: dict[int, list[tuple[int, int]]],
    source_bus: dict[int, float],
    target_bus: dict[int, float],
    stage: str,
) -> None:
    """Runtime checks for shared fan-out tee alignment and input departure routing."""
    if not source_bus:
        return
    _assert_source_bus_clears_fanout_targets(
        source_bus,
        graph=graph,
        anchors=anchors,
        outgoing=outgoing,
        stage=stage,
    )
    _assert_shared_fanout_branch_tees_aligned(
        link_paths,
        graph=graph,
        anchors=anchors,
        outgoing=outgoing,
        source_bus=source_bus,
        target_bus=target_bus,
        stage=stage,
    )
    _assert_fanout_avoids_input_horizontal_departure(
        link_paths,
        graph=graph,
        anchors=anchors,
        outgoing=outgoing,
        source_bus=source_bus,
        stage=stage,
    )


def _swap_merge_entry_x_if_crossing(
    ordered_links: list[tuple[int, int]],
    entry_x: dict[tuple[int, int], float],
    positions: list,
) -> None:
    """Keep left sources on left merge ports by swapping entry columns when needed."""
    for index in range(len(ordered_links) - 1):
        for later in range(index + 1, len(ordered_links)):
            left_link = ordered_links[index]
            right_link = ordered_links[later]
            if positions[left_link[0]].cx <= positions[right_link[0]].cx:
                continue
            if entry_x[left_link] < entry_x[right_link]:
                continue
            entry_x[left_link], entry_x[right_link] = entry_x[right_link], entry_x[left_link]


def _assign_spread_merge_entry_x(
    spread_links: list[tuple[int, int]],
    target_anchor: _RenderAnchor,
    target_pos,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    merge_entry_x: dict[tuple[int, int], float],
) -> None:
    """Spread top-edge ports preserving each feeder's relative horizontal order."""
    margin = CONNECTOR_ATTACHED_BOX_MARGIN
    usable_left = target_anchor.left + margin
    usable_right = target_anchor.right - margin
    if usable_right <= usable_left:
        for link in spread_links:
            merge_entry_x[link] = target_anchor.cx
        return

    source_cxs = [positions[link[0]].cx for link in spread_links]
    min_src = min(source_cxs)
    max_src = max(source_cxs)
    if abs(max_src - min_src) <= PARALLEL_CONNECTOR_COORD_EPS:
        for link in spread_links:
            merge_entry_x[link] = target_anchor.cx
        return

    for link in spread_links:
        src_cx = positions[link[0]].cx
        ratio = (src_cx - min_src) / (max_src - min_src)
        merge_entry_x[link] = usable_left + ratio * (usable_right - usable_left)
    _swap_merge_entry_x_if_crossing(spread_links, merge_entry_x, positions)


def _fanout_links_excluding_bypasses(
    graph,
    link_group: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Keep only main-path links when computing shared merge/source buses."""
    return [
        link
        for link in link_group
        if link not in graph.inline_binary_operand_links
    ]


def _plan_inline_binary_bus_x(
    graph,
    links: list[tuple[int, int]],
    anchors: dict[int, _RenderAnchor],
    positions: list | None = None,
    *,
    gap: float = 0.04,
    channel_gap: float = PARALLEL_CONNECTOR_CHANNEL_GAP,
) -> dict[tuple[int, int], float]:
    """Assign distinct in-frame bus columns for inline binary operand connectors."""
    from collections import defaultdict

    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for src, tgt in links:
        link_key = (src, tgt)
        if link_key not in graph.inline_binary_operand_links:
            continue
        frame = _inline_frame_for_nodes(graph, src, tgt)
        if frame is None:
            continue
        grouped[frame.frame_id].append(link_key)

    bus_x_map: dict[tuple[int, int], float] = {}
    frames = getattr(graph, "inline_frames", None) or []
    for frame in frames:
        link_keys = grouped.get(frame.frame_id)
        if not link_keys:
            continue
        left_gutter = _inline_frame_connector_gutter_width(
            graph, frame, side="left", anchors=anchors
        )
        right_gutter = _inline_frame_connector_gutter_width(
            graph, frame, side="right", anchors=anchors
        )
        if positions is not None:
            tile_left, tile_right, _, _ = _inline_frame_tile_envelope(frame, positions)
            frame_left = tile_left - INLINE_FRAME_PAD - left_gutter
            frame_right = tile_right + INLINE_FRAME_PAD + right_gutter
        else:
            source = anchors[link_keys[0][0]]
            target = anchors[link_keys[0][1]]
            frame_left = min(source.left, target.left) - INLINE_FRAME_PAD - left_gutter
            frame_right = max(source.right, target.right) + INLINE_FRAME_PAD + right_gutter
        left_links = [
            link_key
            for link_key in sorted(link_keys)
            if _skip_link_gutter_side(graph, frame, link_key, anchors) == "left"
        ]
        right_links = [
            link_key
            for link_key in sorted(link_keys)
            if _skip_link_gutter_side(graph, frame, link_key, anchors) == "right"
        ]
        for index, link_key in enumerate(left_links):
            bus_x_map[link_key] = frame_left + CONNECTOR_OBSTACLE_MARGIN + index * channel_gap
        for index, link_key in enumerate(right_links):
            bus_x_map[link_key] = frame_right - CONNECTOR_OBSTACLE_MARGIN - index * channel_gap

    for src, tgt in links:
        link_key = (src, tgt)
        if link_key in bus_x_map:
            continue
        if link_key not in graph.inline_binary_operand_links:
            continue
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue
        base_x = min(source.left, target.left) - gap - 0.12
        column_key = round(base_x, 2)
        peers = [
            key
            for key in bus_x_map
            if round(min(anchors[key[0]].left, anchors[key[1]].left) - gap - 0.12, 2) == column_key
        ]
        bus_x_map[link_key] = base_x - len(peers) * channel_gap
    return bus_x_map


def _connector_axis_segments(
    points: list[tuple[float, float]],
) -> list[tuple[str, float, float, float, int]]:
    """Return axis-aligned segments as (orientation, coord, lo, hi, segment_index)."""
    segments: list[tuple[str, float, float, float, int]] = []
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        if abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS:
            lo, hi = sorted((x1, x2))
            segments.append(("h", y1, lo, hi, index))
        elif abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS:
            lo, hi = sorted((y1, y2))
            segments.append(("v", x1, lo, hi, index))
    return segments


def _ranges_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    return hi_a - lo_b > PARALLEL_CONNECTOR_COORD_EPS and hi_b - lo_a > PARALLEL_CONNECTOR_COORD_EPS


def _parallel_coord_bucket(coord: float, *, tol: float = PARALLEL_CONNECTOR_CHANNEL_GAP / 2) -> int:
    return round(coord / tol)


def _connector_segment_pairs_overlap(
    seg_a: tuple[str, float, float, float, int],
    seg_b: tuple[str, float, float, float, int],
    *,
    link_a: tuple[int, int],
    link_b: tuple[int, int],
    coord_tol: float = PARALLEL_CONNECTOR_CHANNEL_GAP / 2,
) -> bool:
    """True when two axis-aligned connector segments occupy the same channel."""
    ori_a, coord_a, lo_a, hi_a, _ = seg_a
    ori_b, coord_b, lo_b, hi_b, _ = seg_b
    if link_a == link_b or ori_a != ori_b:
        return False
    if _parallel_coord_bucket(coord_a, tol=coord_tol) != _parallel_coord_bucket(coord_b, tol=coord_tol):
        return False
    return _ranges_overlap(lo_a, hi_a, lo_b, hi_b)


def _connector_overlap_is_inline_bypass_spine_tee(
    link_a: tuple[int, int],
    seg_a: tuple[str, float, float, float, int],
    link_b: tuple[int, int],
    seg_b: tuple[str, float, float, float, int],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
) -> bool:
    """Bypass paths branch horizontally from the main inline-frame spine."""
    bypass_links = graph.inline_binary_operand_links
    for bypass_link, other_link, bypass_seg, other_seg in (
        (link_a, link_b, seg_a, seg_b),
        (link_b, link_a, seg_b, seg_a),
    ):
        if bypass_link not in bypass_links:
            continue
        spine_x = anchors[bypass_link[0]].cx
        vertical, horizontal = (
            (bypass_seg, other_seg)
            if bypass_seg[0] == "v" and other_seg[0] == "h"
            else (other_seg, bypass_seg)
            if other_seg[0] == "v" and bypass_seg[0] == "h"
            else (None, None)
        )
        if vertical is None or horizontal is None:
            continue
        vx, vy_lo, vy_hi, _ = vertical[1], vertical[2], vertical[3], vertical[4]
        hy, hx_lo, hx_hi, _ = horizontal[1], horizontal[2], horizontal[3], horizontal[4]
        if abs(vx - spine_x) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if (
            hx_lo - PARALLEL_CONNECTOR_COORD_EPS <= vx <= hx_hi + PARALLEL_CONNECTOR_COORD_EPS
            and vy_lo - PARALLEL_CONNECTOR_COORD_EPS <= hy <= vy_hi + PARALLEL_CONNECTOR_COORD_EPS
        ):
            return True
    return False


def _connector_overlap_is_shared_inline_spine_vertical(
    link_a: tuple[int, int],
    seg_a: tuple[str, float, float, float, int],
    link_b: tuple[int, int],
    seg_b: tuple[str, float, float, float, int],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
) -> bool:
    """Bypass connectors reuse the main inline-frame vertical spine column."""
    if seg_a[0] != "v" or seg_b[0] != "v":
        return False
    if abs(seg_a[1] - seg_b[1]) > PARALLEL_CONNECTOR_COORD_EPS:
        return False
    spine_x = seg_a[1]
    bypass_links = graph.inline_binary_operand_links
    for link in (link_a, link_b):
        if link not in bypass_links:
            continue
        if abs(anchors[link[0]].cx - spine_x) <= PARALLEL_CONNECTOR_COORD_EPS:
            return True
    return False


def _connector_overlap_is_fanout_source_tee(
    link_a: tuple[int, int],
    seg_a: tuple[str, float, float, float, int],
    link_b: tuple[int, int],
    seg_b: tuple[str, float, float, float, int],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    outgoing: dict[int, list[tuple[int, int]]],
    source_bus: dict[int, float],
    target_bus: dict[int, float],
) -> bool:
    """Fan-out legs tee horizontally from the source column before a merge bus."""
    if link_a[0] != link_b[0]:
        return False
    src = link_a[0]
    if src not in source_bus:
        return False
    if not _source_fanout_splits_before_target_bus(graph, src, outgoing, target_bus):
        return False
    tee_y = source_bus[src]
    spine_x = anchors[src].cx
    for vertical, horizontal in ((seg_a, seg_b), (seg_b, seg_a)):
        if vertical[0] != "v" or horizontal[0] != "h":
            continue
        vx, vy_lo, vy_hi, _ = vertical[1], vertical[2], vertical[3], vertical[4]
        hy, hx_lo, hx_hi, _ = horizontal[1], horizontal[2], horizontal[3], horizontal[4]
        if abs(vx - spine_x) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if abs(hy - tee_y) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if (
            hx_lo - PARALLEL_CONNECTOR_COORD_EPS <= vx <= hx_hi + PARALLEL_CONNECTOR_COORD_EPS
            and vy_lo - PARALLEL_CONNECTOR_COORD_EPS <= hy <= vy_hi + PARALLEL_CONNECTOR_COORD_EPS
        ):
            return True
    if seg_a[0] == "v" and seg_b[0] == "v":
        if (
            abs(seg_a[1] - spine_x) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(seg_b[1] - spine_x) <= PARALLEL_CONNECTOR_COORD_EPS
            and _ranges_overlap(seg_a[2], seg_a[3], seg_b[2], seg_b[3])
        ):
            return True
    return False


def _connector_overlap_is_shared_source_fanout_horizontal(
    link_a: tuple[int, int],
    seg_a: tuple[str, float, float, float, int],
    link_b: tuple[int, int],
    seg_b: tuple[str, float, float, float, int],
    *,
    anchors: dict[int, _RenderAnchor],
) -> bool:
    """Fan-out legs intentionally share the horizontal departure from a source."""
    if link_a[0] != link_b[0] or seg_a[0] != "h" or seg_b[0] != "h":
        return False
    if abs(seg_a[1] - seg_b[1]) > PARALLEL_CONNECTOR_COORD_EPS:
        return False
    source = anchors.get(link_a[0])
    if source is None:
        return False
    overlap_lo = max(min(seg_a[2], seg_a[3]), min(seg_b[2], seg_b[3]))
    overlap_hi = min(max(seg_a[2], seg_a[3]), max(seg_b[2], seg_b[3]))
    if overlap_hi <= overlap_lo + PARALLEL_CONNECTOR_COORD_EPS:
        return False
    return overlap_lo <= source.cx + 0.15


def _find_connector_inline_frame_overlaps(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    positions: list,
) -> list[tuple[tuple[int, int], str, str]]:
    """Return links whose paths cut through expanded dotted-frame interiors."""
    violations: list[tuple[tuple[int, int], str, str]] = []
    for link_key, points in link_paths.items():
        src, tgt = link_key
        violation = _connector_path_violates_inline_frame_bounds(
            points,
            graph,
            positions,
            src=src,
            tgt=tgt,
        )
        if violation is not None:
            frame_id, reason = violation
            violations.append((link_key, frame_id, reason))
    return violations


def _find_connector_node_clearance_violations(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list,
) -> list[tuple[tuple[int, int], str]]:
    """Return links whose paths sit within the obstacle margin of intermediate nodes."""
    violations: list[tuple[tuple[int, int], str]] = []
    for link_key, points in link_paths.items():
        src, tgt = link_key
        obstacles = _connector_block_obstacles(
            anchors,
            src=src,
            tgt=tgt,
            label_obstacles=label_obstacles,
            graph=graph,
            positions=positions,
            link_key=link_key,
        )
        if _path_hits_obstacles(points, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN):
            violations.append((link_key, "touches intermediate node"))
    return violations


def _find_connector_path_overlaps(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    incoming: dict[int, list[tuple[int, int]]] | None = None,
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
    target_bus: dict[int, float] | None = None,
    source_bus: dict[int, float] | None = None,
    merge_link_bus: dict[tuple[int, int], float] | None = None,
    anchors: dict[int, _RenderAnchor] | None = None,
    graph=None,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return unordered link pairs whose polylines share a non-bus channel."""
    incoming = incoming or {}
    outgoing = outgoing or {}
    target_bus = target_bus or {}
    source_bus = source_bus or {}
    merge_link_bus = merge_link_bus or {}
    anchors = anchors or {}

    segment_refs: list[tuple[tuple[int, int], tuple[str, float, float, float, int]]] = []
    for link_key, points in link_paths.items():
        for segment in _connector_axis_segments(points):
            segment_refs.append((link_key, segment))

    overlaps: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for index, (link_a, seg_a) in enumerate(segment_refs):
        ori_a, coord_a, _, _, _ = seg_a
        for link_b, seg_b in segment_refs[index + 1 :]:
            if not _connector_segment_pairs_overlap(seg_a, seg_b, link_a=link_a, link_b=link_b):
                continue
            if ori_a == "v":
                if _vertical_segment_is_shared_bus(
                    link_a,
                    coord_a,
                    incoming=incoming,
                    outgoing=outgoing,
                    target_bus=target_bus,
                    source_bus=source_bus,
                    anchors=anchors,
                ) and _vertical_segment_is_shared_bus(
                    link_b,
                    seg_b[1],
                    incoming=incoming,
                    outgoing=outgoing,
                    target_bus=target_bus,
                    source_bus=source_bus,
                    anchors=anchors,
                ):
                    continue
            elif _horizontal_segment_is_shared_bus(
                coord_a,
                target_bus=target_bus,
                source_bus=source_bus,
                merge_link_bus=merge_link_bus,
            ) and _horizontal_segment_is_shared_bus(
                seg_b[1],
                target_bus=target_bus,
                source_bus=source_bus,
                merge_link_bus=merge_link_bus,
            ):
                continue
            if graph is not None and _connector_overlap_is_inline_bypass_spine_tee(
                link_a,
                seg_a,
                link_b,
                seg_b,
                graph=graph,
                anchors=anchors,
            ):
                continue
            if graph is not None and _connector_overlap_is_shared_inline_spine_vertical(
                link_a,
                seg_a,
                link_b,
                seg_b,
                graph=graph,
                anchors=anchors,
            ):
                continue
            if graph is not None and _connector_overlap_is_fanout_source_tee(
                link_a,
                seg_a,
                link_b,
                seg_b,
                graph=graph,
                anchors=anchors,
                outgoing=outgoing,
                source_bus=source_bus,
                target_bus=target_bus,
            ):
                continue
            if _connector_overlap_is_shared_source_fanout_horizontal(
                link_a,
                seg_a,
                link_b,
                seg_b,
                anchors=anchors,
            ):
                continue
            if (
                link_a[0] == link_b[0]
                and link_a[0] in source_bus
                and seg_a[0] == "v"
                and seg_b[0] == "v"
            ):
                source = anchors.get(link_a[0])
                if (
                    source is not None
                    and abs(seg_a[1] - source.cx) <= PARALLEL_CONNECTOR_COORD_EPS
                    and abs(seg_b[1] - source.cx) <= PARALLEL_CONNECTOR_COORD_EPS
                ):
                    continue
            pair = tuple(sorted((link_a, link_b)))
            if pair not in seen:
                seen.add(pair)
                overlaps.append(pair)
    return overlaps


def _shift_path_horizontal_levels(
    points: list[tuple[float, float]],
    *,
    delta_y: float,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> list[tuple[float, float]]:
    if abs(delta_y) <= eps:
        return points
    horizontal_levels: set[float] = set()
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        if abs(y1 - y2) <= eps:
            horizontal_levels.add(y1)
    if not horizontal_levels:
        return points
    adjusted: list[tuple[float, float]] = []
    for x, y in points:
        shifted = False
        for level in horizontal_levels:
            if abs(y - level) <= eps:
                adjusted.append((x, y + delta_y))
                shifted = True
                break
        if not shifted:
            adjusted.append((x, y))
    return adjusted


def _shift_path_vertical_levels(
    points: list[tuple[float, float]],
    *,
    delta_x: float,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> list[tuple[float, float]]:
    if abs(delta_x) <= eps:
        return points
    vertical_columns: set[float] = set()
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        if abs(x1 - x2) <= eps:
            vertical_columns.add(x1)
    if not vertical_columns:
        return points
    adjusted: list[tuple[float, float]] = []
    for x, y in points:
        shifted = False
        for column in vertical_columns:
            if abs(x - column) <= eps:
                adjusted.append((x + delta_x, y))
                shifted = True
                break
        if not shifted:
            adjusted.append((x, y))
    return _ensure_orthogonal_connector_path(adjusted)


def _shift_path_resolving_overlap(
    points: list[tuple[float, float]],
    *,
    src: int,
    tgt: int,
    anchors: dict[int, _RenderAnchor],
    delta_y: float,
) -> list[tuple[float, float]]:
    """Shift horizontal bus levels while keeping the path out of third-party tiles."""
    obstacles = [
        anchor for node_index, anchor in anchors.items() if node_index not in {src, tgt}
    ]
    for attempt in (delta_y, -delta_y, 2 * delta_y, -2 * delta_y):
        candidate = _shift_path_horizontal_levels(points, delta_y=attempt)
        if not _path_penetrates_obstacle_tiles(
            candidate,
            obstacles,
            margin=CONNECTOR_OBSTACLE_MARGIN,
        ):
            return candidate
    return _shift_path_horizontal_levels(points, delta_y=delta_y)


def _ensure_connector_paths_non_overlapping(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    anchors: dict[int, _RenderAnchor],
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Offset bypass channels until no non-bus connector segments overlap."""
    cleared = dict(link_paths)
    for _ in range(12):
        overlaps = _find_connector_path_overlaps(
            cleared,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            anchors=anchors,
            graph=graph,
        )
        if not overlaps:
            return cleared
        link_a, link_b = overlaps[0]
        if (
            link_a in graph.inline_binary_operand_links
            or link_b in graph.inline_binary_operand_links
        ):
            continue
        if link_b not in graph.inline_binary_operand_links and link_a in graph.inline_binary_operand_links:
            link_a, link_b = link_b, link_a
        if link_a in graph.inline_binary_operand_links:
            cleared[link_a] = _shift_path_resolving_overlap(
                cleared[link_a],
                src=link_a[0],
                tgt=link_a[1],
                anchors=anchors,
                delta_y=PARALLEL_CONNECTOR_CHANNEL_GAP,
            )
            continue
        if link_b in graph.inline_binary_operand_links:
            cleared[link_b] = _shift_path_resolving_overlap(
                cleared[link_b],
                src=link_b[0],
                tgt=link_b[1],
                anchors=anchors,
                delta_y=PARALLEL_CONNECTOR_CHANNEL_GAP,
            )
            continue
        seg_a = next(
            segment
            for segment in _connector_axis_segments(cleared[link_a])
            if segment[0] == "v"
        )
        seg_b = next(
            segment
            for segment in _connector_axis_segments(cleared[link_b])
            if segment[0] == "v"
        )
        if (
            seg_a[0] == "v"
            and seg_b[0] == "v"
            and abs(seg_a[1] - seg_b[1]) <= PARALLEL_CONNECTOR_COORD_EPS
            and _ranges_overlap(seg_a[2], seg_a[3], seg_b[2], seg_b[3])
        ):
            for delta in (
                PARALLEL_CONNECTOR_CHANNEL_GAP,
                -PARALLEL_CONNECTOR_CHANNEL_GAP,
                2 * PARALLEL_CONNECTOR_CHANNEL_GAP,
                -2 * PARALLEL_CONNECTOR_CHANNEL_GAP,
            ):
                candidate = _shift_path_vertical_levels(
                    cleared[link_b],
                    delta_x=delta,
                )
                if not _find_connector_path_overlaps(
                    {**cleared, link_b: candidate},
                    incoming=incoming,
                    outgoing=outgoing,
                    target_bus=target_bus,
                    source_bus=source_bus,
                    merge_link_bus=merge_link_bus,
                    anchors=anchors,
                    graph=graph,
                ):
                    cleared[link_b] = candidate
                    break
            continue
        cleared[link_b] = _shift_path_resolving_overlap(
            cleared[link_b],
            src=link_b[0],
            tgt=link_b[1],
            anchors=anchors,
            delta_y=PARALLEL_CONNECTOR_CHANNEL_GAP,
        )
    return cleared


def _vertical_segment_is_shared_bus(
    link_key: tuple[int, int],
    coord: float,
    *,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    anchors: dict[int, _RenderAnchor],
) -> bool:
    """True when parallel vertical drops intentionally merge on a shared bus column."""
    src, tgt = link_key
    target = anchors.get(tgt)
    if (
        target is not None
        and tgt in target_bus
        and len(incoming.get(tgt, [])) >= 2
        and abs(coord - target.cx) <= PARALLEL_CONNECTOR_COORD_EPS
    ):
        return True
    source = anchors.get(src)
    if (
        source is not None
        and src in source_bus
        and len(outgoing.get(src, [])) >= 2
        and abs(coord - source.cx) <= PARALLEL_CONNECTOR_COORD_EPS
    ):
        return True
    return False


def _horizontal_segment_is_shared_bus(
    coord: float,
    *,
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
) -> bool:
    """True when a horizontal segment lies on an intentional merge/source bus."""
    shared = set(target_bus.values()) | set(source_bus.values()) | set(merge_link_bus.values())
    return any(abs(coord - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS for bus_y in shared)


def _separate_parallel_connector_paths(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    anchors: dict[int, _RenderAnchor],
    channel_gap: float = PARALLEL_CONNECTOR_CHANNEL_GAP,
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Offset coincident parallel connector segments that are not shared buses."""
    from dataclasses import dataclass

    @dataclass
    class _SegmentRef:
        link_key: tuple[int, int]
        orientation: str
        coord: float
        lo: float
        hi: float

    vertical: dict[float, list[_SegmentRef]] = {}
    horizontal: dict[float, list[_SegmentRef]] = {}

    for link_key, points in link_paths.items():
        for orientation, coord, lo, hi, _index in _connector_axis_segments(points):
            ref = _SegmentRef(link_key=link_key, orientation=orientation, coord=coord, lo=lo, hi=hi)
            bucket = vertical if orientation == "v" else horizontal
            key = _parallel_coord_bucket(coord)
            bucket.setdefault(key, []).append(ref)

    x_offsets: dict[tuple[int, int], dict[float, float]] = {}
    y_offsets: dict[tuple[int, int], dict[float, float]] = {}

    def _assign_offsets(group: list[_SegmentRef], *, vertical_axis: bool) -> None:
        if len(group) < 2:
            return
        remaining = list(group)
        while remaining:
            cluster = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                next_remaining: list[_SegmentRef] = []
                for candidate in remaining:
                    if any(
                        _ranges_overlap(candidate.lo, candidate.hi, member.lo, member.hi)
                        for member in cluster
                    ):
                        cluster.append(candidate)
                        changed = True
                    else:
                        next_remaining.append(candidate)
                remaining = next_remaining
            if len({segment.link_key for segment in cluster}) < 2:
                continue
            if vertical_axis:
                if all(
                    _vertical_segment_is_shared_bus(
                        segment.link_key,
                        segment.coord,
                        incoming=incoming,
                        outgoing=outgoing,
                        target_bus=target_bus,
                        source_bus=source_bus,
                        anchors=anchors,
                    )
                    for segment in cluster
                ):
                    continue
            else:
                if all(
                    _horizontal_segment_is_shared_bus(
                        segment.coord,
                        target_bus=target_bus,
                        source_bus=source_bus,
                        merge_link_bus=merge_link_bus,
                    )
                    for segment in cluster
                ):
                    continue
            base_coord = cluster[0].coord
            for index, segment in enumerate(sorted(cluster, key=lambda item: item.link_key)):
                shifted = base_coord - index * channel_gap if vertical_axis else base_coord - index * channel_gap
                target_map = x_offsets if vertical_axis else y_offsets
                target_map.setdefault(segment.link_key, {})[base_coord] = shifted

    for segments in vertical.values():
        _assign_offsets(segments, vertical_axis=True)
    for segments in horizontal.values():
        _assign_offsets(segments, vertical_axis=False)

    if not x_offsets and not y_offsets:
        return link_paths

    separated: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for link_key, points in link_paths.items():
        link_x = x_offsets.get(link_key, {})
        link_y = y_offsets.get(link_key, {})
        if not link_x and not link_y:
            separated[link_key] = points
            continue
        adjusted: list[tuple[float, float]] = []
        for x, y in points:
            new_x = x
            new_y = y
            for coord, shifted in link_x.items():
                if abs(x - coord) <= PARALLEL_CONNECTOR_COORD_EPS:
                    new_x = shifted
                    break
            for coord, shifted in link_y.items():
                if abs(y - coord) <= PARALLEL_CONNECTOR_COORD_EPS:
                    new_y = shifted
                    break
            adjusted.append((new_x, new_y))
        separated[link_key] = adjusted
    return separated


def _collect_detail_link_paths(
    *,
    graph,
    links: list[tuple[int, int]],
    positions: list,
    anchors: dict[int, _RenderAnchor],
    incoming: dict[int, list[tuple[int, int]]],
    label_obstacles: list[_RenderAnchor],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_entry_x: dict[tuple[int, int], float],
    merge_link_bus: dict[tuple[int, int], float],
    input_index: int | None,
    validate_layout: bool = True,
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    from collections import defaultdict

    outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for src, tgt in links:
        outgoing[src].append((src, tgt))

    link_paths: dict[tuple[int, int], list[tuple[float, float]]] = {}
    inline_binary_bus_x = _plan_inline_binary_bus_x(graph, links, anchors, positions)
    for src, tgt in links:
        link_key = (src, tgt)
        points = _connector_points_for_link(
            graph=graph,
            positions=positions,
            anchors=anchors,
            src=src,
            tgt=tgt,
            link_key=link_key,
            incoming=incoming,
            outgoing=outgoing,
            label_obstacles=label_obstacles,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_entry_x=merge_entry_x,
            merge_link_bus=merge_link_bus,
            input_index=input_index,
            inline_binary_bus_x=inline_binary_bus_x,
        )
        if points is not None and len(points) >= 2:
            source = anchors.get(src)
            target = anchors.get(tgt)
            if source is not None and target is not None:
                obstacles = _connector_block_obstacles(
                    anchors,
                    src=src,
                    tgt=tgt,
                    label_obstacles=label_obstacles,
                    graph=graph,
                    positions=positions,
                    link_key=link_key,
                )
                bus_y = _link_routing_bus_y(
                    link_key,
                    target_bus=target_bus,
                    source_bus=source_bus,
                    merge_link_bus=merge_link_bus,
                )
                points = _reroute_connector_path_clearing_blocks(
                    points,
                    source=source,
                    target=target,
                    obstacles=obstacles,
                    bus_y=bus_y,
                    graph=graph,
                    positions=positions,
                    link_key=link_key,
                    source_bus=source_bus,
                    merge_link_bus=merge_link_bus,
                )
            link_paths[link_key] = points
    separated = _separate_parallel_connector_paths(
        link_paths,
        incoming=incoming,
        outgoing=outgoing,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
        anchors=anchors,
    )
    for link_key, points in separated.items():
        original = link_paths.get(link_key)
        if original is None or points == original:
            continue
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue
        obstacles = _connector_block_obstacles(
            anchors,
            src=src,
            tgt=tgt,
            label_obstacles=label_obstacles,
            graph=graph,
            positions=positions,
            link_key=link_key,
        )
        if _path_hits_obstacles(
            points,
            obstacles,
            margin=CONNECTOR_OBSTACLE_MARGIN,
        ) or not _connector_path_clear_of_blocks(
            points,
            source=source,
            target=target,
            obstacles=obstacles,
        ):
            separated[link_key] = original
    cleared = _clear_detail_link_paths(
        separated,
        anchors,
        label_obstacles,
        graph=graph,
        positions=positions,
    )
    cleared = _reroute_detail_link_paths_clearing_blocks(
        cleared,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
    )
    _assert_detail_link_paths_clear_of_blocks(
        cleared,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
        stage="obstacle clearing",
    )
    cleared = _ensure_connector_paths_non_overlapping(
        cleared,
        graph=graph,
        incoming=incoming,
        outgoing=outgoing,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
        anchors=anchors,
    )
    cleared = _reroute_detail_link_paths_clearing_blocks(
        cleared,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
    )
    _assert_detail_link_paths_clear_of_blocks(
        cleared,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
        stage="overlap separation",
    )
    _finalize_inline_bypass_spine_tees(
        cleared,
        graph=graph,
        anchors=anchors,
        positions=positions,
        inline_binary_bus_x=inline_binary_bus_x,
    )
    cleared = _reroute_detail_link_paths_clearing_blocks(
        cleared,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
    )
    _assert_detail_link_paths_clear_of_blocks(
        cleared,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
        stage="bypass spine finalize",
    )
    validated: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for link_key, points in cleared.items():
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None or len(points) < 2:
            validated[link_key] = points
            continue
        obstacles = _connector_block_obstacles(
            anchors,
            src=src,
            tgt=tgt,
            label_obstacles=label_obstacles,
            graph=graph,
            positions=positions,
            link_key=link_key,
        )
        bus_y = _link_routing_bus_y(
            link_key,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
        )
        routed = points
        if not _connector_path_clear_of_blocks(
            points,
            source=source,
            target=target,
            obstacles=obstacles,
        ):
            routed = _reroute_connector_path_clearing_blocks(
                points,
                source=source,
                target=target,
                obstacles=obstacles,
                bus_y=bus_y,
                graph=graph,
                positions=positions,
                link_key=link_key,
                source_bus=source_bus,
                merge_link_bus=merge_link_bus,
            )
        snapped = _snap_connector_path_endpoints(
            routed,
            source=source,
            target=target,
            link_key=link_key,
            graph=graph,
        )
        if _connector_path_clear_of_blocks(
            snapped,
            source=source,
            target=target,
            obstacles=obstacles,
        ) and (
            graph is None
            or positions is None
            or _connector_path_violates_inline_frame_bounds(
                snapped,
                graph,
                positions,
                src=src,
                tgt=tgt,
            )
            is None
        ):
            routed = snapped
        validated[link_key] = routed
        if validate_layout and _graph_requires_strict_connector_validation(graph):
            _assert_connector_path_clear_of_blocks(
                link_key,
                routed,
                graph=graph,
                anchors=anchors,
                label_obstacles=label_obstacles,
                positions=positions,
                stage="pre-shrinkwrap snap",
            )
    _assert_connector_tees_precede_bus_joins(
        validated,
        graph=graph,
        anchors=anchors,
        outgoing=outgoing,
        source_bus=source_bus,
        target_bus=target_bus,
        stage="final",
    )
    from visualizer.shrinkwrap import SHRINKWRAP_MIN_GAP, shrinkwrap_detail_link_paths

    validated = shrinkwrap_detail_link_paths(
        validated,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
        incoming=incoming,
        outgoing=outgoing,
        min_gap=SHRINKWRAP_MIN_GAP,
    )
    if validate_layout:
        _assert_detail_link_paths_clear_of_blocks(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            stage="shrinkwrap",
        )
    overlap_pairs = _find_connector_path_overlaps(
        validated,
        incoming=incoming,
        outgoing=outgoing,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
        anchors=anchors,
        graph=graph,
    )
    if validate_layout and overlap_pairs and _graph_requires_strict_connector_validation(graph):
        raise RuntimeError(
            "connector overlap after layout: "
            + ", ".join(f"{pair[0]}|{pair[1]}" for pair in overlap_pairs[:4])
        )
    if validate_layout and _graph_requires_strict_connector_validation(graph):
        frame_overlaps = _find_connector_inline_frame_overlaps(
            validated,
            graph=graph,
            positions=positions,
        )
        if frame_overlaps:
            raise RuntimeError(
                "connector crosses dotted frame after layout: "
                + ", ".join(
                    f"{graph.nodes[key[0]].label!r}->{graph.nodes[key[1]].label!r} "
                    f"({frame_id!r}: {reason})"
                    for key, frame_id, reason in frame_overlaps[:4]
                )
            )
        node_clearance = _find_connector_node_clearance_violations(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
        )
        if node_clearance:
            raise RuntimeError(
                "connector touches intermediate node after layout: "
                + ", ".join(
                    f"{graph.nodes[key[0]].label!r}->{graph.nodes[key[1]].label!r} ({reason})"
                    for key, reason in node_clearance[:4]
                )
            )
    _assert_detail_fanout_connector_invariants(
        validated,
        graph=graph,
        anchors=anchors,
        outgoing=outgoing,
        source_bus=source_bus,
        target_bus=target_bus,
        stage="final",
    )
    return validated


def _draw_path(
    ax,
    points: list[tuple[float, float]],
    *,
    color: str | None = None,
    linewidth: float = 1.5,
    linestyle: str = "solid",
    zorder: float = FLOW_CONNECTOR_ZORDER,
) -> None:
    if len(points) < 2:
        return
    stroke = color or COLORS["flow"]
    if linestyle != "solid":
        for index in range(len(points) - 1):
            x1, y1 = points[index]
            x2, y2 = points[index + 1]
            _line(
                ax,
                x1,
                y1,
                x2,
                y2,
                color=stroke,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=zorder,
            )
        return

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.plot(
        xs,
        ys,
        color=stroke,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
        solid_capstyle="butt",
        solid_joinstyle="miter",
    )


def _quantize_connector_point(
    x: float,
    y: float,
    *,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> tuple[float, float]:
    return round(x / eps) * eps, round(y / eps) * eps


def _dedupe_polyline_points(
    points: list[tuple[float, float]],
    *,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped or _quantize_connector_point(*point, eps=eps) != _quantize_connector_point(
            *deduped[-1], eps=eps
        ):
            deduped.append(point)
    return deduped


def _point_on_axis_segment_interior(
    x: float,
    y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> bool:
    if abs(y1 - y2) <= eps:
        if abs(y - y1) > eps:
            return False
        seg_lo, seg_hi = sorted((x1, x2))
        return x > seg_lo + eps and x < seg_hi - eps
    if abs(x1 - x2) <= eps:
        if abs(x - x1) > eps:
            return False
        seg_lo, seg_hi = sorted((y1, y2))
        return y > seg_lo + eps and y < seg_hi - eps
    return False


def _is_connector_path_bend(
    points: list[tuple[float, float]],
    index: int,
    *,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> bool:
    """True when path direction changes between horizontal and vertical at ``index``."""
    if index <= 0 or index >= len(points) - 1:
        return False
    x0, y0 = points[index - 1]
    x1, y1 = points[index]
    x2, y2 = points[index + 1]
    dx_in = abs(x1 - x0)
    dy_in = abs(y1 - y0)
    dx_out = abs(x2 - x1)
    dy_out = abs(y2 - y1)
    in_horiz = dy_in <= eps and dx_in > eps
    out_horiz = dy_out <= eps and dx_out > eps
    in_vert = dx_in <= eps and dy_in > eps
    out_vert = dx_out <= eps and dy_out > eps
    return (in_horiz and out_vert) or (in_vert and out_horiz)


def _segment_orientation(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> str | None:
    if abs(y1 - y2) <= eps and abs(x1 - x2) > eps:
        return "h"
    if abs(x1 - x2) <= eps and abs(y1 - y2) > eps:
        return "v"
    return None


def _orientations_at_path_vertex(
    points: list[tuple[float, float]],
    index: int,
    *,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> tuple[str | None, str | None]:
    x0, y0 = points[index - 1]
    x1, y1 = points[index]
    x2, y2 = points[index + 1]
    return (
        _segment_orientation(x0, y0, x1, y1, eps=eps),
        _segment_orientation(x1, y1, x2, y2, eps=eps),
    )


def _point_on_shared_bus(
    y: float,
    *,
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> bool:
    shared = set(target_bus.values()) | set(source_bus.values()) | set(merge_link_bus.values())
    return any(abs(y - bus_y) <= eps for bus_y in shared)


def _point_is_fanout_split_tee(
    x: float,
    y: float,
    *,
    link_keys: set[tuple[int, int]],
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    graph,
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    anchors: dict[int, _RenderAnchor],
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> bool:
    """True for a single-source tee distributing one tensor onto a shared source bus."""
    sources = {link_key[0] for link_key in link_keys}
    if len(sources) != 1:
        return False
    src = next(iter(sources))
    if src not in source_bus:
        return False
    if not all(link_key[0] == src for link_key in link_keys):
        return False
    source = anchors.get(src)
    if source is None:
        return False
    bus_y = source_bus[src]
    splits_before_merge = _source_fanout_splits_before_target_bus(
        graph,
        src,
        outgoing,
        target_bus,
    )
    if splits_before_merge:
        merge_bus_y = _fanout_lowest_target_merge_bus_y(graph, src, outgoing, target_bus)
        if merge_bus_y is not None and y <= merge_bus_y + eps:
            return False
    for link_key in link_keys:
        path = _dedupe_polyline_points(link_paths.get(link_key, []), eps=eps)
        for index in range(1, len(path) - 1):
            px, py = path[index]
            if abs(px - x) > eps or abs(py - y) > eps:
                continue
            in_ori, out_ori = _orientations_at_path_vertex(path, index, eps=eps)
            if abs(y - bus_y) <= eps and in_ori == "h" and out_ori == "v":
                return True
            if (
                abs(x - source.cx) <= eps
                and abs(y - bus_y) <= eps + CONNECTOR_EXIT_STUB
                and in_ori == "v"
                and out_ori == "h"
            ):
                return True
    return False


def _connector_point_is_bus_t_junction(
    x: float,
    y: float,
    *,
    link_keys: set[tuple[int, int]],
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    graph=None,
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
    anchors: dict[int, _RenderAnchor] | None = None,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> bool:
    """True when two or more links form a T on a shared bus, not a single-link L."""
    if len(link_keys) < 2:
        return False
    if len({link_key[0] for link_key in link_keys}) < 2:
        return False
    outgoing = outgoing or {}
    anchors = anchors or {}
    if (
        graph is not None
        and _point_is_fanout_split_tee(
            x,
            y,
            link_keys=link_keys,
            link_paths=link_paths,
            graph=graph,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            anchors=anchors,
            eps=eps,
        )
    ):
        return False
    if not _point_on_shared_bus(
        y,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
    ):
        return False

    horiz_links: set[tuple[int, int]] = set()
    vert_links: set[tuple[int, int]] = set()
    for link_key in link_keys:
        path = _dedupe_polyline_points(link_paths[link_key], eps=eps)
        for index in range(1, len(path) - 1):
            px, py = path[index]
            if abs(px - x) > eps or abs(py - y) > eps:
                continue
            in_ori, out_ori = _orientations_at_path_vertex(path, index, eps=eps)
            if "h" in (in_ori, out_ori):
                horiz_links.add(link_key)
            if "v" in (in_ori, out_ori):
                vert_links.add(link_key)

    if len(horiz_links) >= 2:
        return True
    return bool(horiz_links and vert_links and len(horiz_links | vert_links) >= 2)


def _collect_cross_link_bus_t_junctions(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    graph=None,
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
    anchors: dict[int, _RenderAnchor] | None = None,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> set[tuple[float, float]]:
    """Find T joins where one link meets the interior of another link's bus segment."""
    joins: set[tuple[float, float]] = set()
    outgoing = outgoing or {}
    anchors = anchors or {}
    link_items = list(link_paths.items())
    for link_a, points_a in link_items:
        path_a = _dedupe_polyline_points(points_a, eps=eps)
        for seg_index in range(len(path_a) - 1):
            x1, y1 = path_a[seg_index]
            x2, y2 = path_a[seg_index + 1]
            if abs(y1 - y2) > eps or abs(x1 - x2) <= eps:
                continue
            if not _point_on_shared_bus(
                y1,
                target_bus=target_bus,
                source_bus=source_bus,
                merge_link_bus=merge_link_bus,
            ):
                continue
            seg_lo, seg_hi = sorted((x1, x2))
            for link_b, points_b in link_items:
                if link_a == link_b:
                    continue
                if link_a[0] == link_b[0]:
                    continue
                path_b = _dedupe_polyline_points(points_b, eps=eps)
                for index in range(1, len(path_b) - 1):
                    bx, by = path_b[index]
                    if abs(by - y1) > BUS_JUNCTION_Y_EPS:
                        continue
                    if not (seg_lo + eps < bx < seg_hi - eps):
                        continue
                    in_ori, out_ori = _orientations_at_path_vertex(path_b, index, eps=eps)
                    if "v" not in (in_ori, out_ori):
                        continue
                    link_keys = {link_a, link_b}
                    if graph is not None and _point_is_fanout_split_tee(
                        bx,
                        by,
                        link_keys=link_keys,
                        link_paths=link_paths,
                        graph=graph,
                        outgoing=outgoing,
                        target_bus=target_bus,
                        source_bus=source_bus,
                        anchors=anchors,
                        eps=eps,
                    ):
                        continue
                    joins.add((bx, by))
    return joins


def _collect_connector_join_points(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
    target_bus: dict[int, float] | None = None,
    source_bus: dict[int, float] | None = None,
    merge_link_bus: dict[tuple[int, int], float] | None = None,
    incoming: dict[int, list[tuple[int, int]]] | None = None,
    anchors: dict[int, _RenderAnchor] | None = None,
    graph=None,
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
) -> list[tuple[float, float]]:
    """Return points where two or more connector paths T-join on a shared bus."""
    del incoming
    target_bus = target_bus or {}
    source_bus = source_bus or {}
    merge_link_bus = merge_link_bus or {}
    anchors = anchors or {}
    outgoing = outgoing or {}

    internal_vertex_links: dict[tuple[float, float], set[tuple[int, int]]] = defaultdict(set)
    representatives: dict[tuple[float, float], tuple[float, float]] = {}

    for link_key, points in link_paths.items():
        path = _dedupe_polyline_points(points, eps=eps)
        if len(path) < 3:
            continue
        for index, (x, y) in enumerate(path):
            if index == 0 or index == len(path) - 1:
                continue
            key = _quantize_connector_point(x, y, eps=eps)
            internal_vertex_links[key].add(link_key)
            representatives.setdefault(key, (x, y))

    joins: set[tuple[float, float]] = set()
    for key, links in internal_vertex_links.items():
        if len(links) < 2 or key not in representatives:
            continue
        x, y = representatives[key]
        if not _connector_point_is_bus_t_junction(
            x,
            y,
            link_keys=links,
            link_paths=link_paths,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            graph=graph,
            outgoing=outgoing,
            anchors=anchors,
            eps=eps,
        ):
            continue
        joins.add((x, y))

    joins.update(
        _collect_cross_link_bus_t_junctions(
            link_paths,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            graph=graph,
            outgoing=outgoing,
            anchors=anchors,
            eps=eps,
        )
    )

    return sorted(joins, key=lambda point: (point[1], point[0]))


def _connector_point_is_bus_junction(
    x: float,
    y: float,
    *,
    link_keys: set[tuple[int, int]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    incoming: dict[int, list[tuple[int, int]]],
    anchors: dict[int, _RenderAnchor],
    link_paths: dict[tuple[int, int], list[tuple[float, float]]] | None = None,
) -> bool:
    """Backward-compatible wrapper around bus T-junction detection."""
    del incoming, anchors
    if link_paths is None:
        return False
    return _connector_point_is_bus_t_junction(
        x,
        y,
        link_keys=link_keys,
        link_paths=link_paths,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
    )


def _junction_dot_fits(
    x: float,
    y: float,
    obstacles: list[_RenderAnchor],
    *,
    halo_radius: float,
    combine_ops: list[tuple[float, float]] | None = None,
) -> bool:
    for obs in obstacles:
        if (
            x + halo_radius > obs.left
            and x - halo_radius < obs.right
            and y + halo_radius > obs.bottom
            and y - halo_radius < obs.top
        ):
            return False
    for cx, cy in combine_ops or []:
        clearance = MERGE_RADIUS + halo_radius + 0.02
        if (x - cx) ** 2 + (y - cy) ** 2 < clearance**2:
            return False
    return True


def _filter_spaced_join_points(
    points: list[tuple[float, float]],
    *,
    min_sep: float | None = None,
) -> list[tuple[float, float]]:
    if min_sep is None:
        min_sep = CONNECTOR_JUNCTION_HALO_RADIUS * 1.6
    kept: list[tuple[float, float]] = []
    min_sep_sq = min_sep * min_sep
    for x, y in points:
        if all((x - other_x) ** 2 + (y - other_y) ** 2 >= min_sep_sq for other_x, other_y in kept):
            kept.append((x, y))
    return kept


def _draw_connector_junction_dot(
    ax,
    x: float,
    y: float,
    *,
    color: str,
    bg_color: str,
    zorder: float = CONNECTOR_JUNCTION_ZORDER,
) -> None:
    """Draw a small halo-backed dot marking a connector join."""
    ax.add_patch(
        Circle(
            (x, y),
            CONNECTOR_JUNCTION_HALO_RADIUS,
            facecolor=bg_color,
            edgecolor="none",
            zorder=zorder,
        )
    )
    ax.add_patch(
        Circle(
            (x, y),
            CONNECTOR_JUNCTION_DOT_RADIUS,
            facecolor=color,
            edgecolor="none",
            linewidth=0.0,
            zorder=zorder + 0.05,
        )
    )


def _draw_connector_junction_dots(
    ax,
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    obstacles: list[_RenderAnchor],
    combine_ops: list[tuple[float, float]] | None = None,
    color: str | None = None,
    bg_color: str | None = None,
    target_bus: dict[int, float] | None = None,
    source_bus: dict[int, float] | None = None,
    merge_link_bus: dict[tuple[int, int], float] | None = None,
    incoming: dict[int, list[tuple[int, int]]] | None = None,
    anchors: dict[int, _RenderAnchor] | None = None,
    graph=None,
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
) -> None:
    """Mark shared merge-bus joins so they are visually distinct from crossings."""
    if ax is None:
        return
    stroke = color or COLORS["flow"]
    fill = bg_color or COLORS["bg"]
    join_points = _filter_spaced_join_points(
        _collect_connector_join_points(
            link_paths,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            incoming=incoming,
            anchors=anchors,
            graph=graph,
            outgoing=outgoing,
        )
    )
    for x, y in join_points:
        if not _junction_dot_fits(
            x,
            y,
            obstacles,
            halo_radius=CONNECTOR_JUNCTION_HALO_RADIUS,
            combine_ops=combine_ops,
        ):
            continue
        _draw_connector_junction_dot(ax, x, y, color=stroke, bg_color=fill)


def _inline_dashed_port_connector_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    *,
    gap: float = 0.04,
    bus_y: float | None = None,
) -> list[tuple[float, float]]:
    """Route a parallel branch into the top center of an inline-port node."""
    entry_x = target.cx
    entry_y = _connector_target_top_entry_y(target, gap=gap)
    y1, y_stub = _connector_exit_stub_y(source.bottom, gap=gap)
    bus_y = entry_y if bus_y is None else bus_y
    return [(source.cx, y1), (source.cx, y_stub), (source.cx, bus_y), (entry_x, bus_y), (entry_x, entry_y)]


def _link_routing_bus_y(
    link_key: tuple[int, int],
    *,
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
) -> float | None:
    """Per-link horizontal routing level used by obstacle reroutes."""
    if link_key in merge_link_bus:
        return merge_link_bus[link_key]
    tgt = link_key[1]
    if tgt in target_bus:
        return target_bus[tgt]
    src = link_key[0]
    if src in source_bus:
        return source_bus[src]
    return None


def _fanout_tee_then_entry_column_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    entry_x: float,
    *,
    tee_y: float,
    bus_y: float,
    gap: float = 0.04,
) -> list[tuple[float, float]]:
    """Fan-out leg tees on the source column, branches, then drops on the target column."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    y2 = _connector_target_top_entry_y(target, gap=gap)
    if bus_y >= tee_y - PARALLEL_CONNECTOR_COORD_EPS:
        return [
            (source.cx, y1),
            (source.cx, tee_y),
            (entry_x, tee_y),
            (entry_x, y2),
        ]
    return [
        (source.cx, y1),
        (source.cx, tee_y),
        (entry_x, tee_y),
        (entry_x, bus_y),
        (entry_x, y2),
    ]


def _shared_merge_bus_connector_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    entry_x: float,
    bus_y: float,
    obstacles: list[_RenderAnchor],
    *,
    exit_x: float | None = None,
    y_stub: float | None = None,
    spine_tee_y: float | None = None,
    gap: float = 0.04,
    graph=None,
    positions: list | None = None,
    link_key: tuple[int, int] | None = None,
    prefer_tee_branch: bool = False,
) -> list[tuple[float, float]]:
    """Route a fan-in branch down to a shared merge bus, across, then into the target."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    y2 = _connector_target_top_entry_y(target, gap=gap)
    check_obstacles = obstacles
    stub_y = y_stub if y_stub is not None else y1 - CONNECTOR_EXIT_STUB

    def _prepend_spine(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if (
            spine_tee_y is None
            or spine_tee_y <= bus_y + PARALLEL_CONNECTOR_COORD_EPS
            or len(path) < 2
        ):
            return path
        if abs(path[0][0] - source.cx) > PARALLEL_CONNECTOR_COORD_EPS:
            return [(source.cx, y1), (source.cx, spine_tee_y), *path]
        if abs(path[1][1] - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS:
            return [path[0], (source.cx, spine_tee_y), *path[1:]]
        return [path[0], (source.cx, spine_tee_y), *path[1:]]

    def _straight_on_bus() -> list[tuple[float, float]]:
        if abs(source.cx - entry_x) < 0.06:
            if bus_y >= y2 + PARALLEL_CONNECTOR_COORD_EPS:
                return _prepend_spine([(source.cx, y1), (source.cx, bus_y), (entry_x, y2)])
            return [(source.cx, y1), (source.cx, y2)]
        return _prepend_spine(
            [(source.cx, y1), (source.cx, bus_y), (entry_x, bus_y), (entry_x, y2)]
        )

    def _via_tee_then_entry_column() -> list[tuple[float, float]] | None:
        """Fan-out leg tees on the source column before dropping to a lower per-leg bus."""
        if (
            spine_tee_y is None
            or spine_tee_y <= bus_y + PARALLEL_CONNECTOR_COORD_EPS
        ):
            return None
        return _fanout_tee_then_entry_column_points(
            source,
            target,
            entry_x,
            tee_y=spine_tee_y,
            bus_y=bus_y,
            gap=gap,
        )

    def _via_gutter(gutter_x: float) -> list[tuple[float, float]]:
        return _prepend_spine(
            [
                (source.cx, y1),
                (source.cx, stub_y),
                (gutter_x, stub_y),
                (gutter_x, bus_y),
                (entry_x, bus_y),
                (entry_x, y2),
            ]
        )

    straight = _straight_on_bus()
    tee_route = _via_tee_then_entry_column()
    prefer_tee = prefer_tee_branch
    if (
        not prefer_tee
        and tee_route is not None
        and graph is not None
        and positions is not None
        and link_key is not None
    ):
        prefer_tee = (
            _connector_path_violates_inline_frame_bounds(
                straight,
                graph,
                positions,
                src=link_key[0],
                tgt=link_key[1],
            )
            is not None
        )
    candidates: list[list[tuple[float, float]]] = []
    if prefer_tee and tee_route is not None:
        candidates.append(tee_route)
    candidates.append(straight)
    if tee_route is not None and not prefer_tee:
        candidates.append(tee_route)
    gutter_x = exit_x
    if gutter_x is None and abs(source.cx - entry_x) > 0.06:
        gutter_x = source.left - CONNECTOR_OBSTACLE_MARGIN
    if gutter_x is not None and abs(gutter_x - source.cx) > 0.06:
        candidates.append(_via_gutter(gutter_x))
    right_gutter = source.right + CONNECTOR_OBSTACLE_MARGIN
    if abs(right_gutter - source.cx) > 0.06 and (
        gutter_x is None or abs(right_gutter - gutter_x) > 0.06
    ):
        candidates.append(_via_gutter(right_gutter))

    for candidate in candidates:
        if not _path_penetrates_obstacle_tiles(
            candidate,
            check_obstacles,
            margin=CONNECTOR_OBSTACLE_MARGIN,
        ):
            return candidate
    return _straight_on_bus()


def _labeled_merge_connector_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    entry_x: float,
    *,
    gap: float = 0.04,
    bus_y: float | None = None,
) -> list[tuple[float, float]]:
    """Route a fan-in branch into a labeled port on the merge node top edge."""
    y1, y_stub = _connector_exit_stub_y(source.bottom, gap=gap)
    entry_y = _connector_target_top_entry_y(target, gap=gap)
    if bus_y is None:
        bus_y = (y_stub + entry_y) / 2
    if abs(source.cx - entry_x) < 0.06:
        if bus_y >= entry_y + PARALLEL_CONNECTOR_COORD_EPS:
            return [
                (source.cx, y1),
                (source.cx, y_stub),
                (source.cx, bus_y),
                (entry_x, entry_y),
            ]
        return [(source.cx, y1), (source.cx, y_stub), (source.cx, entry_y)]
    return [(source.cx, y1), (source.cx, y_stub), (source.cx, bus_y), (entry_x, bus_y), (entry_x, entry_y)]


def _top_entry_combine_connector_points(
    source: _RenderAnchor,
    target_cx: float,
    target_cy: float,
    *,
    gap: float = 0.04,
    target: _RenderAnchor | None = None,
) -> list[tuple[float, float]]:
    """Route the main sequential path into the top of a combine operator."""
    entry_x = target_cx
    if target is not None and _combine_uses_box_anchor(target):
        entry_y = target.top
    else:
        entry_y = target_cy + MERGE_RADIUS
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    if abs(source.cx - target_cx) < 0.08:
        return [(source.cx, y1), (source.cx, entry_y)]
    bus_y = (y1 + entry_y) / 2
    return [(source.cx, y1), (source.cx, bus_y), (entry_x, bus_y), (entry_x, entry_y)]


def _inline_binary_side_entry_connector_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    *,
    gap: float = 0.04,
    bus_x: float | None = None,
    frame_bounds=None,
    tee_y: float | None = None,
) -> list[tuple[float, float]]:
    """Route an earlier inline-frame operand around intermediate tiles into a binary op."""
    target_cy = _connector_target_side_entry_y(target)
    if bus_x is None:
        if frame_bounds is not None:
            bus_x = frame_bounds.left + CONNECTOR_OBSTACLE_MARGIN
        else:
            bus_x = min(source.left, target.left) - gap - 0.12
    if frame_bounds is not None:
        bus_x = _clamp_bus_x_to_frame_interior(bus_x, frame_bounds)
    entry_x = target.left if bus_x <= target.cx else target.right
    entry_y = target_cy
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    if tee_y is not None and tee_y <= y1 - PARALLEL_CONNECTOR_COORD_EPS:
        y_stub = y1 - CONNECTOR_EXIT_STUB
        points: list[tuple[float, float]] = [
            (source.cx, y_stub),
            (bus_x, y_stub),
            (bus_x, tee_y),
        ]
        if abs(entry_y - tee_y) > PARALLEL_CONNECTOR_COORD_EPS:
            points.append((bus_x, entry_y))
        points.append((entry_x, entry_y))
        return points
    return [
        *_connector_leave_source_before_horizontal(source, target, bus_x, gap=gap),
        (bus_x, entry_y),
        (entry_x, entry_y),
    ]


def _side_entry_combine_entry_x(
    source: _RenderAnchor,
    target: _RenderAnchor,
) -> float:
    """Horizontal attach point on the near side of a combine operator."""
    if _combine_uses_box_anchor(target):
        return target.left if source.cx < target.cx else target.right
    if source.cx >= target.cx:
        return target.cx + MERGE_RADIUS
    return target.cx - MERGE_RADIUS


def _side_entry_combine_l_route(
    source: _RenderAnchor,
    target: _RenderAnchor,
    *,
    gap: float = 0.04,
) -> list[tuple[float, float]]:
    """Drop from the source column straight to combine center, then enter from the side."""
    combine_cy = _combine_center_y(target)
    entry_x = _side_entry_combine_entry_x(source, target)
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    return _ensure_orthogonal_connector_path(
        [(source.cx, y1), (source.cx, combine_cy), (entry_x, combine_cy)]
    )


def _side_entry_combine_connector_points(
    source: _RenderAnchor,
    target_cx: float,
    target_cy: float,
    *,
    gap: float = 0.04,
    frame_bounds=None,
    obstacles: list[_RenderAnchor] | None = None,
    target: _RenderAnchor | None = None,
) -> list[tuple[float, float]]:
    """Route a side branch into the center of a combine node without a long horizontal bus."""
    if target is not None and _combine_uses_box_anchor(target):
        entry_x = target.left if source.cx < target.cx else target.right
        if source.cx >= target_cx:
            bus_x = entry_x + gap + 0.08
        else:
            bus_x = entry_x - gap - 0.08
    elif source.cx >= target_cx:
        entry_x = target_cx + MERGE_RADIUS
        bus_x = entry_x + gap + 0.08
    else:
        entry_x = target_cx - MERGE_RADIUS
        bus_x = entry_x - gap - 0.08
    if frame_bounds is not None:
        bus_x = _clamp_bus_x_to_frame_interior(bus_x, frame_bounds)

    source_cy = (source.top + source.bottom) / 2
    if (
        abs(source_cy - target_cy) <= max(source.top - source.bottom, 0.12) * 0.6
        and abs(source.cx - entry_x) <= MERGE_RADIUS + gap + 0.24
    ):
        return [(source.cx, target_cy), (entry_x, target_cy)]

    y1, y_stub = _connector_exit_stub_y(source.bottom, gap=gap)
    if abs(source.cx - bus_x) < 0.06:
        stub_path = [(source.cx, y1), (source.cx, y_stub), (bus_x, target_cy), (entry_x, target_cy)]
    else:
        stub_path = [
            (source.cx, y1),
            (source.cx, y_stub),
            (bus_x, y_stub),
            (bus_x, target_cy),
            (entry_x, target_cy),
        ]
    if obstacles is not None and _path_hits_obstacles(
        stub_path,
        obstacles,
        margin=CONNECTOR_OBSTACLE_MARGIN,
    ):
        l_route = [(source.cx, y1), (source.cx, target_cy), (entry_x, target_cy)]
        if not _path_hits_obstacles(
            l_route,
            obstacles,
            margin=CONNECTOR_OBSTACLE_MARGIN,
        ):
            return l_route
    return stub_path


def _detail_connector_linestyle(
    graph,
    *,
    src: int,
    positions: list[LayoutPosition],
    source: _RenderAnchor,
    target: _RenderAnchor,
) -> str:
    """Detail connectors are always solid; dashed strokes are reserved for expanded inline frames."""
    _ = (graph, src, positions, source, target)
    return "solid"


def _assert_detail_connector_linestyles_are_solid(
    graph,
    *,
    links: list[tuple[int, int]],
    positions: list[LayoutPosition],
    anchors: dict[int, _RenderAnchor],
) -> None:
    """Fail when any detail connector would render with a dashed linestyle."""
    dashed: list[str] = []
    for src, tgt in links:
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue
        if (
            _detail_connector_linestyle(
                graph,
                src=src,
                positions=positions,
                source=source,
                target=target,
            )
            != "solid"
        ):
            dashed.append(f"{graph.nodes[src].label!r}->{graph.nodes[tgt].label!r}")
    if dashed:
        raise RuntimeError(
            "detail connectors must use solid linestyle: " + ", ".join(dashed[:4])
        )


def _polyline_bounds(
    points: list[tuple[float, float]],
    *,
    half_width: float = 0.025,
) -> ContentBounds:
    """Axis-aligned bounds covering an orthogonal connector polyline."""
    from visualizer.text_measure import ContentBounds

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return ContentBounds(
        left=min(xs) - half_width,
        right=max(xs) + half_width,
        bottom=min(ys) - half_width,
        top=max(ys) + half_width,
    )


def _cross_column_side_entry_connector_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    gap: float = 0.04,
    bus_y: float | None = None,
) -> list[tuple[float, float]]:
    """Route a distant column feed into the near side of the target tile."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    y_stub = y1 - CONNECTOR_EXIT_STUB
    target_cy = _connector_target_side_entry_y(target)
    side_gap = CONNECTOR_SIDE_ENTRY_GAP
    if source.cx >= target.cx:
        entry_x = target.right + side_gap
        entry_on_target = target.right
    else:
        entry_x = target.left - side_gap
        entry_on_target = target.left
    if bus_y is None:
        bus_y = (y_stub + target_cy) / 2
    for delta in (0.0, -0.08, -0.16, 0.08, 0.16):
        candidate_bus = bus_y + delta
        points = [
            (source.cx, y1),
            (source.cx, y_stub),
            (source.cx, candidate_bus),
            (entry_x, candidate_bus),
            (entry_x, target_cy),
            (entry_on_target, target_cy),
        ]
        if not _path_hits_obstacles(
            points,
            obstacles,
            margin=CONNECTOR_OBSTACLE_MARGIN,
        ):
            return points
    return [
        (source.cx, y1),
        (source.cx, y_stub),
        (source.cx, bus_y),
        (entry_x, bus_y),
        (entry_x, target_cy),
        (entry_on_target, target_cy),
    ]


def _tensor_port_side_entry_connector_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    *,
    gap: float = 0.04,
) -> list[tuple[float, float]]:
    """Route a tensor port into the side of a distant column with extra clearance."""
    y1, y_stub = _connector_exit_stub_y(source.bottom, gap=gap)
    target_cy = _connector_target_side_entry_y(target)
    side_gap = CONNECTOR_SIDE_ENTRY_GAP
    if source.cx <= target.cx:
        entry_x = target.left - side_gap
        entry_on_target = target.left
    else:
        entry_x = target.right + side_gap
        entry_on_target = target.right
    bus_y = (y_stub + target_cy) / 2
    return [
        (source.cx, y1),
        (source.cx, y_stub),
        (source.cx, bus_y),
        (entry_x, bus_y),
        (entry_x, target_cy),
        (entry_on_target, target_cy),
    ]


def _tensor_port_connector_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    gap: float = 0.04,
) -> list[tuple[float, float]]:
    """Route a modeling tensor port to the kernel step it feeds."""
    return _orthogonal_path(source, target, obstacles, bus_near="target", gap=gap)


def _replace_path_y_level(
    points: list[tuple[float, float]],
    old_y: float,
    new_y: float,
) -> list[tuple[float, float]]:
    return [
        (x, new_y if abs(y - old_y) <= PARALLEL_CONNECTOR_COORD_EPS else y)
        for x, y in points
    ]


def _path_horizontal_bus_levels(points: list[tuple[float, float]]) -> list[float]:
    levels: list[float] = []
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        if abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS and abs(x1 - x2) > 0.06:
            if not any(abs(y1 - existing) <= PARALLEL_CONNECTOR_COORD_EPS for existing in levels):
                levels.append(y1)
    return levels


def _clear_connector_path_obstacles(
    points: list[tuple[float, float]],
    obstacles: list[_RenderAnchor],
    *,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> list[tuple[float, float]]:
    """Shift horizontal bus levels until the path clears all box obstacles."""
    if len(points) < 2 or not obstacles:
        return points
    current = list(points)
    if not _path_penetrates_obstacle_tiles(current, obstacles, margin=margin):
        return current
    for bus_y in sorted(_path_horizontal_bus_levels(current), reverse=True):
        for delta in (0.08, 0.16, 0.24, -0.08, -0.16, -0.24):
            adjusted = _replace_path_y_level(current, bus_y, bus_y + delta)
            if not _path_penetrates_obstacle_tiles(adjusted, obstacles, margin=margin):
                return adjusted
    return current


def _path_horizontal_segments_overlap_bounds(
    points: list[tuple[float, float]],
    bounds,
    *,
    margin: float = CONNECTOR_ATTACHED_BOX_MARGIN,
) -> bool:
    """True when a horizontal segment runs through the interior of axis-aligned bounds."""
    tol = PARALLEL_CONNECTOR_COORD_EPS / 10
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS or abs(x1 - x2) <= 0.06:
            continue
        y = y1
        if y + margin <= bounds.bottom + tol or y - margin >= bounds.top - tol:
            continue
        seg_left = min(x1, x2)
        seg_right = max(x1, x2)
        if seg_right + margin <= bounds.left or seg_left - margin >= bounds.right:
            continue
        return True
    return False


def _path_stays_inside_bounds(
    points: list[tuple[float, float]],
    bounds,
    *,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> bool:
    """Return True when every segment stays inside axis-aligned bounds."""
    tol = PARALLEL_CONNECTOR_COORD_EPS / 10
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        min_x = min(x1, x2) - margin
        max_x = max(x1, x2) + margin
        min_y = min(y1, y2) - margin
        max_y = max(y1, y2) + margin
        if (
            min_x < bounds.left - tol
            or max_x > bounds.right + tol
            or min_y < bounds.bottom - tol
            or max_y > bounds.top + tol
        ):
            return False
    return True


def _clear_detail_link_paths(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    *,
    graph=None,
    positions: list | None = None,
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Ensure every rendered connector segment clears non-endpoint boxes."""
    cleared: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for link_key, points in link_paths.items():
        src, tgt = link_key
        obstacles = _connector_block_obstacles(
            anchors,
            src=src,
            tgt=tgt,
            label_obstacles=label_obstacles,
            graph=graph,
            positions=positions,
            link_key=link_key,
        )
        adjusted = _clear_connector_path_obstacles(points, obstacles)
        if graph is not None and positions is not None:
            shared_frame = _inline_frame_for_nodes(graph, src, tgt)
            if shared_frame is not None and link_key in graph.inline_binary_operand_links:
                frame_bounds = _inline_frame_routing_bounds(shared_frame, positions, graph)
                if not _path_stays_inside_bounds(adjusted, frame_bounds):
                    for bus_y in sorted(_path_horizontal_bus_levels(adjusted), reverse=True):
                        for delta in (0.08, 0.16, -0.08, -0.16):
                            candidate = _replace_path_y_level(adjusted, bus_y, bus_y + delta)
                            if _path_stays_inside_bounds(candidate, frame_bounds) and not _path_hits_obstacles(
                                candidate, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN
                            ):
                                adjusted = candidate
                                break
                        else:
                            continue
                        break
        cleared[link_key] = adjusted
    return _ensure_connector_paths_clear_attached_boxes(cleared, anchors)


def _path_has_horizontal_bus_leg(
    points: list[tuple[float, float]],
    *,
    min_span: float = 0.06,
) -> bool:
    """True when a connector uses a horizontal merge/source bus segment."""
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        if abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS and abs(x1 - x2) >= min_span:
            return True
    return False


def _ensure_connector_paths_clear_attached_boxes(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    anchors: dict[int, _RenderAnchor],
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Re-route any connector that still crosses its source or target tile."""
    cleared: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for link_key, points in link_paths.items():
        if len(points) < 2:
            cleared[link_key] = points
            continue
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            cleared[link_key] = points
            continue
        if _path_penetrates_attached_boxes(points, source, target):
            if _path_has_horizontal_bus_leg(points):
                cleared[link_key] = points
                continue
            route_obstacles = [
                anchor for node_index, anchor in anchors.items() if node_index not in {src, tgt}
            ]
            rerouted = _orthogonal_path(source, target, route_obstacles)
            if not _path_penetrates_attached_boxes(rerouted, source, target):
                cleared[link_key] = rerouted
                continue
            if abs(source.cx - target.cx) < 0.06:
                cleared[link_key] = _same_column_side_gutter_detour(
                    source,
                    target,
                    route_obstacles,
                )
                continue
        cleared[link_key] = points
    return cleared


def _pipeline_frame_exit_connector_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    *,
    exit_x: float,
    bus_y: float,
    frame_bounds=None,
    gap: float = 0.04,
    obstacles: list[_RenderAnchor] | None = None,
    entry_x: float | None = None,
) -> list[tuple[float, float]]:
    """Leave an inline-frame column downward before joining a shared merge bus."""
    route_obstacles = list(obstacles or [])
    x2 = entry_x if entry_x is not None else target.cx
    frame_stub_y = None
    if frame_bounds is not None:
        frame_stub_y = _frame_exit_horizontal_y(
            frame_bounds,
            source_bottom=source.bottom,
            source_cx=source.cx,
            obstacles=route_obstacles,
        )
    return _shared_merge_bus_connector_points(
        source,
        target,
        x2,
        bus_y,
        route_obstacles,
        exit_x=exit_x,
        y_stub=frame_stub_y,
        gap=gap,
    )


def _inline_frame_bounds_obstacles(
    graph,
    positions: list,
    *,
    src: int | None = None,
    tgt: int | None = None,
    exclude_nodes: set[int] | None = None,
) -> list[_RenderAnchor]:
    """Expanded dotted-frame bounds as routing obstacles."""
    del exclude_nodes
    obstacles: list[_RenderAnchor] = []
    for frame in graph.inline_frames:
        members = set(frame.node_indices)
        if src is not None and src in members:
            continue
        if tgt is not None and tgt in members:
            continue
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        obstacles.append(
            _RenderAnchor(
                cx=(bounds.left + bounds.right) / 2,
                top=bounds.top,
                bottom=bounds.bottom,
                left=bounds.left,
                right=bounds.right,
            )
        )
    return obstacles


def _inline_frame_for_nodes(graph, src: int, tgt: int):
    """Return the inline frame containing both node indices, if any."""
    frames = getattr(graph, "inline_frames", None) or []
    for frame in frames:
        members = set(frame.node_indices)
        if src in members and tgt in members:
            return frame
    return None


def _inline_frame_bypass_links(graph, frame) -> list[tuple[int, int]]:
    """Skip/residual operand links routed wholly inside one inline frame."""
    members = set(frame.node_indices)
    return [
        (src, tgt)
        for src, tgt in graph.inline_binary_operand_links
        if src in members and tgt in members
    ]


def _inline_frame_uses_bilateral_skip_gutter(graph, frame) -> bool:
    """Use left and right gutters when skips enter and leave the same frame."""
    bypass = _inline_frame_bypass_links(graph, frame)
    if len(bypass) >= 2:
        return True
    members = set(frame.node_indices)
    for index in members:
        has_in = any(tgt == index for _, tgt in bypass)
        has_out = any(src == index for src, _ in bypass)
        if has_in and has_out:
            return True
    return False


def _bypass_link_preferred_gutter_side(
    source: _RenderAnchor,
    target: _RenderAnchor,
) -> str:
    """Route a bypass horizontally toward the side its target is on."""
    if abs(target.cx - source.cx) <= 0.10:
        return "left"
    return "right" if target.cx > source.cx else "left"


def _skip_link_gutter_side(
    graph,
    frame,
    link_key: tuple[int, int],
    anchors: dict[int, _RenderAnchor],
) -> str:
    """Pick left or right frame gutter for one bypass connector."""
    src, tgt = link_key
    source = anchors.get(src)
    target = anchors.get(tgt)
    if source is None or target is None:
        return "left"
    preferred = _bypass_link_preferred_gutter_side(source, target)

    bypass = _inline_frame_bypass_links(graph, frame)

    def _has_in(index: int) -> bool:
        return any(link_tgt == index for _, link_tgt in bypass)

    def _has_out(index: int) -> bool:
        return any(s == index for s, _ in bypass)

    if _has_in(src) and _has_out(src):
        return "right" if preferred == "left" else "left"
    if not _inline_frame_uses_bilateral_skip_gutter(graph, frame):
        return preferred

    from visualizer.computation_graph import _ordered_inline_frame_chain

    chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
    chain_index = {index: rank for rank, index in enumerate(chain)}
    ordered = sorted(bypass, key=lambda pair: chain_index.get(pair[0], pair[0]))
    preferences = {
        link: _bypass_link_preferred_gutter_side(anchors[link[0]], anchors[link[1]])
        for link in ordered
        if link[0] in anchors and link[1] in anchors
    }
    assigned: dict[tuple[int, int], str] = {}
    for link in ordered:
        side = preferences.get(link, "left")
        used = set(assigned.values())
        if side in used:
            side = "right" if side == "left" else "left"
        assigned[link] = side
    return assigned.get(link_key, preferred)


def _inline_frame_side_entry_link_count(graph, frame) -> int:
    """Count skip/side-entry operand links routed inside one inline frame."""
    return len(_inline_frame_bypass_links(graph, frame))


def _inline_frame_connector_gutter_width(
    graph,
    frame,
    *,
    side: str = "left",
    anchors: dict[int, _RenderAnchor] | None = None,
) -> float:
    """Reserve connector gutter width on one side of an inline frame."""
    bypass = _inline_frame_bypass_links(graph, frame)
    if not bypass:
        return 0.0
    bilateral = _inline_frame_uses_bilateral_skip_gutter(graph, frame)
    if bilateral:
        if anchors:
            link_count = sum(
                1
                for link_key in bypass
                if _skip_link_gutter_side(graph, frame, link_key, anchors) == side
            )
        else:
            link_count = (len(bypass) + 1) // 2 if side == "left" else len(bypass) // 2
        if link_count <= 0:
            return 0.0
        channel_span = link_count * PARALLEL_CONNECTOR_CHANNEL_GAP
        if side == "left":
            return CONNECTOR_OBSTACLE_MARGIN + channel_span
        return channel_span + CONNECTOR_OBSTACLE_MARGIN
    if side != "left":
        return 0.0
    return (
        CONNECTOR_OBSTACLE_MARGIN
        + len(bypass) * PARALLEL_CONNECTOR_CHANNEL_GAP
        + CONNECTOR_OBSTACLE_MARGIN
    )


def _inline_frame_total_connector_gutter_width(graph, frame) -> float:
    """Combined left and right gutter reservation for one inline frame."""
    return _inline_frame_connector_gutter_width(graph, frame, side="left") + _inline_frame_connector_gutter_width(
        graph, frame, side="right"
    )


def _inline_frame_tile_envelope(
    frame,
    positions: list,
) -> tuple[float, float, float, float]:
    """Return (left, right, bottom, top) tile bounds for frame members."""
    frame_positions = [positions[index] for index in frame.node_indices if index < len(positions)]
    if not frame_positions:
        return 0.0, 0.0, 0.0, 0.0
    min_left = min(pos.cx - pos.width / 2 for pos in frame_positions)
    max_right = max(pos.cx + pos.width / 2 for pos in frame_positions)
    min_bottom = min(pos.top_y - pos.height for pos in frame_positions)
    max_top = max(pos.top_y for pos in frame_positions)
    return min_left, max_right, min_bottom, max_top


def _inline_frame_draw_bounds(
    frame,
    positions: list,
    graph,
    *,
    pad: float = INLINE_FRAME_PAD,
):
    """Dotted-frame bounds including an internal connector gutter when needed."""
    from visualizer.text_measure import ContentBounds

    min_left, max_right, min_bottom, max_top = _inline_frame_tile_envelope(frame, positions)
    left_gutter = _inline_frame_connector_gutter_width(graph, frame, side="left")
    right_gutter = _inline_frame_connector_gutter_width(graph, frame, side="right")
    return ContentBounds(
        left=min_left - pad - left_gutter,
        right=max_right + pad + right_gutter,
        bottom=min_bottom - pad,
        top=max_top + pad,
    )


def _inline_frame_routing_bounds(
    frame,
    positions: list,
    graph,
    *,
    pad: float = INLINE_FRAME_PAD,
):
    """Frame bounds for connector routing, including bypass exit-stub clearance."""
    from visualizer.text_measure import ContentBounds

    draw_bounds = _inline_frame_draw_bounds(frame, positions, graph, pad=pad)
    bypass_stub = CONNECTOR_EXIT_STUB if _inline_frame_bypass_links(graph, frame) else 0.0
    if bypass_stub <= 0:
        return draw_bounds
    return ContentBounds(
        left=draw_bounds.left,
        right=draw_bounds.right,
        bottom=draw_bounds.bottom - bypass_stub,
        top=draw_bounds.top,
    )


def _clamp_bus_x_to_frame_interior(
    bus_x: float,
    frame_bounds,
    *,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> float:
    """Keep a vertical connector bus off the dotted frame border."""
    return min(max(bus_x, frame_bounds.left + margin), frame_bounds.right - margin)


def _needs_cross_column_side_entry(
    source: _RenderAnchor,
    target: _RenderAnchor,
    *,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> bool:
    """True when a feed arrives from well outside the target tile column."""
    return (
        source.cx > target.right + margin
        or source.cx < target.left - margin
    )


def _connector_points_for_link(
    *,
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    src: int,
    tgt: int,
    link_key: tuple[int, int],
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
    label_obstacles: list[_RenderAnchor],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_entry_x: dict[tuple[int, int], float],
    merge_link_bus: dict[tuple[int, int], float],
    input_index: int | None,
    inline_binary_bus_x: dict[tuple[int, int], float] | None = None,
) -> list[tuple[float, float]] | None:
    """Return connector polyline points for one graph link (same routing as draw)."""
    source = anchors.get(src)
    target = anchors.get(tgt)
    if source is None or target is None:
        return None

    is_side_link = (
        link_key in graph.side_entry_links or link_key in graph.inline_binary_operand_links
    )
    route_obstacles = [
        anchor for node_index, anchor in anchors.items() if node_index not in {src, tgt}
    ] + label_obstacles
    gap = 0.04
    if input_index is not None and src == input_index:
        target_frame = next(
            (frame for frame in graph.inline_frames if tgt in frame.node_indices),
            None,
        )
        if target_frame is not None:
            y1 = _connector_source_bottom_exit_y(source, gap=gap)
            y2 = _connector_target_top_entry_y(target, gap=gap)
            bypass_x = max(source.right, target.right) + CONNECTOR_OBSTACLE_MARGIN
            if target_frame.node_indices and target_frame.node_indices[0] == tgt:
                bypass_x = _right_bypass_x_clearing_horizontal_segment(
                    min(source.cx, target.cx),
                    y1,
                    route_obstacles,
                    initial_bypass_x=bypass_x,
                )
                return _ensure_orthogonal_connector_path(
                    [
                        (source.cx, y1),
                        (bypass_x, y1),
                        (target.cx, y1),
                        (target.cx, y2),
                    ]
                )
            draw_bounds = _inline_frame_draw_bounds(target_frame, positions, graph)
            gutter_y = draw_bounds.bottom - CONNECTOR_OBSTACLE_MARGIN - CONNECTOR_EXIT_STUB
            bypass_x = _right_bypass_x_clearing_horizontal_segment(
                min(source.cx, target.cx),
                gutter_y,
                route_obstacles,
                initial_bypass_x=bypass_x,
            )
            bypass_x = _right_bypass_x_clearing_horizontal_segment(
                source.cx,
                y1,
                route_obstacles,
                initial_bypass_x=bypass_x,
            )
            bypass_x = _right_bypass_x_clearing_vertical_segment(
                gutter_y,
                y1,
                route_obstacles,
                initial_bypass_x=bypass_x,
            )
            return _ensure_orthogonal_connector_path(
                [
                    (source.cx, y1),
                    (bypass_x, y1),
                    (bypass_x, gutter_y),
                    (target.cx, gutter_y),
                    (target.cx, y2),
                ]
            )
    target_spec = positions[tgt].spec
    inline_port = target_spec.port_style == "inline"
    floating_port = target_spec.port_style == "floating"
    combine_side_entry = (
        _is_combine_synthetic(target_spec.synthetic) and link_key in graph.side_entry_links
    )
    inline_binary_side_entry = link_key in graph.inline_binary_operand_links
    has_side_incoming = any(
        (side_src, tgt) in graph.side_entry_links
        or (side_src, tgt) in graph.inline_binary_operand_links
        for side_src, _ in incoming.get(tgt, [])
    )
    combine_top_entry = (
        _is_combine_synthetic(target_spec.synthetic)
        and not is_side_link
        and has_side_incoming
        and not combine_side_entry
    )
    combine_center_y = (
        (target.top + target.bottom) / 2
        if _is_combine_synthetic(target_spec.synthetic) or inline_binary_side_entry
        else None
    )
    bus_near = "source" if input_index is not None and src == input_index else "target"
    bus_y: float | None = None
    fanout_tee_y: float | None = None
    if outgoing is not None:
        bus_y = _fanout_source_bus_y(
            graph,
            src,
            link_key,
            positions=positions,
            outgoing=outgoing,
            source_bus=source_bus,
            target_bus=target_bus,
        )
        if src in source_bus and outgoing is not None:
            main_links = _fanout_links_excluding_bypasses(graph, outgoing.get(src, []))
            if len(main_links) >= SHARED_SOURCE_BUS_MIN_LINKS:
                fanout_tee_y = source_bus[src]
            else:
                link_bus = merge_link_bus.get(link_key)
                if (
                    link_bus is not None
                    and link_bus < source_bus[src] - PARALLEL_CONNECTOR_COORD_EPS
                ):
                    fanout_tee_y = source_bus[src]
                elif _source_fanout_splits_before_target_bus(
                    graph, src, outgoing, target_bus
                ):
                    fanout_tee_y = source_bus[src]
    prefer_tee_branch = False
    if (
        fanout_tee_y is not None
        and outgoing is not None
        and not _source_fanout_splits_before_target_bus(graph, src, outgoing, target_bus)
    ):
        main_links = _fanout_links_excluding_bypasses(graph, outgoing.get(src, []))
        prefer_tee_branch = len(main_links) >= SHARED_SOURCE_BUS_MIN_LINKS
        if not prefer_tee_branch:
            link_bus = merge_link_bus.get(link_key)
            prefer_tee_branch = (
                link_bus is not None
                and link_bus < fanout_tee_y - PARALLEL_CONNECTOR_COORD_EPS
            )
    if positions[src].spec.synthetic == SYNTHETIC_TENSOR:
        pass
    elif bus_y is None and tgt in target_bus and not combine_side_entry and not combine_top_entry:
        bus_y = target_bus[tgt]
        bus_near = "target"
    elif bus_y is None and src in source_bus and link_key not in graph.inline_binary_operand_links:
        bus_y = source_bus[src]
        bus_near = "source"
    elif bus_y is None and src in source_bus and link_key not in graph.inline_binary_operand_links:
        bus_y = source_bus[src]
        bus_near = "source"

    shared_frame = _inline_frame_for_nodes(graph, src, tgt)
    frame_bounds = (
        _inline_frame_draw_bounds(shared_frame, positions, graph)
        if shared_frame is not None
        else None
    )
    if inline_binary_side_entry and combine_center_y is not None:
        tee_y = None
        if shared_frame is not None:
            tee_y = _inline_frame_bypass_tee_y(graph, shared_frame, src, tgt, anchors)
        return _inline_binary_side_entry_connector_points(
            source,
            target,
            gap=gap,
            bus_x=(inline_binary_bus_x or {}).get(link_key),
            frame_bounds=frame_bounds,
            tee_y=tee_y,
        )
    if combine_side_entry and combine_center_y is not None:
        return _side_entry_combine_connector_points(
            source,
            target.cx,
            combine_center_y,
            gap=gap,
            frame_bounds=frame_bounds,
            obstacles=route_obstacles,
            target=target,
        )
    if combine_top_entry and combine_center_y is not None:
        return _top_entry_combine_connector_points(
            source,
            target.cx,
            combine_center_y,
            gap=gap,
            target=target,
        )
    if inline_port and is_side_link:
        return _inline_dashed_port_connector_points(source, target, gap=gap, bus_y=bus_y)
    if floating_port:
        return _orthogonal_path(source, target, route_obstacles, bus_near=bus_near, bus_y=bus_y)
    if is_side_link and abs(source.cx - target.cx) < 0.08:
        return _orthogonal_path(source, target, route_obstacles, bus_near=bus_near, bus_y=bus_y)
    merge_bus_y = merge_link_bus.get(link_key)
    if merge_bus_y is None and tgt in target_bus:
        merge_bus_y = bus_y
    if (
        tgt in target_bus
        and merge_bus_y is not None
        and not inline_binary_side_entry
        and not combine_side_entry
        and not combine_top_entry
    ):
        from visualizer.computation_graph import (
            _graph_has_tensor_ports,
            _inline_frame_tail_indices,
        )

        if fanout_tee_y is not None:
            entry_x = merge_entry_x.get(link_key, target.cx)
            return _shared_merge_bus_connector_points(
                source,
                target,
                entry_x,
                merge_bus_y,
                route_obstacles,
                gap=gap,
                spine_tee_y=fanout_tee_y,
                graph=graph,
                positions=positions,
                link_key=link_key,
                prefer_tee_branch=prefer_tee_branch,
            )
        is_frame_tail = (
            _graph_has_tensor_ports(graph) and src in _inline_frame_tail_indices(graph)
        )
        if not is_frame_tail:
            entry_x = merge_entry_x.get(link_key, target.cx)
            return _shared_merge_bus_connector_points(
                source,
                target,
                entry_x,
                merge_bus_y,
                route_obstacles,
                gap=gap,
                spine_tee_y=fanout_tee_y,
                graph=graph,
                positions=positions,
                link_key=link_key,
                prefer_tee_branch=prefer_tee_branch,
            )
    if positions[src].spec.synthetic == SYNTHETIC_TENSOR:
        if tgt in target_bus:
            entry_x = merge_entry_x.get(link_key, target.cx)
            return _shared_merge_bus_connector_points(
                source,
                target,
                entry_x,
                merge_link_bus.get(link_key, target_bus[tgt]),
                route_obstacles,
                gap=gap,
            )
        if abs(source.cx - target.cx) > 0.35:
            return _tensor_port_side_entry_connector_points(source, target, gap=gap)
        return _tensor_port_connector_points(source, target, route_obstacles, gap=gap)
    if (
        tgt in target_bus
        and not combine_side_entry
        and not combine_top_entry
        and bus_y is not None
    ):
        from visualizer.computation_graph import (
            _graph_has_tensor_ports,
            _inline_frame_tail_indices,
        )

        if _graph_has_tensor_ports(graph) and src in _inline_frame_tail_indices(graph):
            if fanout_tee_y is None:
                for frame in graph.inline_frames:
                    if src not in frame.node_indices:
                        continue
                    draw_bounds = _inline_frame_draw_bounds(frame, positions, graph)
                    exit_x = _pipeline_frame_exit_x(
                        source,
                        target,
                        draw_bounds,
                        route_obstacles,
                    )
                    return _pipeline_frame_exit_connector_points(
                        source,
                        target,
                        exit_x=exit_x,
                        bus_y=merge_link_bus.get(link_key, bus_y),
                        frame_bounds=draw_bounds,
                        gap=gap,
                        obstacles=route_obstacles,
                        entry_x=merge_entry_x.get(link_key),
                    )
    if link_key in merge_entry_x and tgt in target_bus:
        return _shared_merge_bus_connector_points(
            source,
            target,
            merge_entry_x[link_key],
            merge_link_bus.get(link_key, target_bus[tgt]),
            route_obstacles,
            gap=gap,
            spine_tee_y=fanout_tee_y,
            graph=graph,
            positions=positions,
            link_key=link_key,
            prefer_tee_branch=prefer_tee_branch,
        )
    if (
        link_key in merge_entry_x
        and not inline_binary_side_entry
        and not combine_side_entry
        and not combine_top_entry
    ):
        if abs(source.cx - target.cx) < 0.08:
            straight = _same_column_straight_connector_points(source, target, gap=gap)
            if (
                not _path_hits_obstacles(
                    straight,
                    route_obstacles,
                    margin=CONNECTOR_OBSTACLE_MARGIN,
                )
                and not _path_penetrates_attached_boxes(straight, source, target)
            ):
                return straight
        entry_x = merge_entry_x[link_key]
        spread_bus_y = merge_link_bus.get(link_key, bus_y)
        if spread_bus_y is None:
            y_stub = _connector_source_bottom_exit_y(source, gap=gap) - CONNECTOR_EXIT_STUB
            entry_y = _connector_target_top_entry_y(target, gap=gap)
            spread_bus_y = max(
                _connector_min_bus_y_above_target(target, gap=gap),
                _min_bus_y_clearing_horizontal_corridor(
                    source.cx,
                    entry_x,
                    route_obstacles,
                    proposed_y=(y_stub + entry_y) / 2,
                ),
            )
        elif fanout_tee_y is not None and tgt not in target_bus:
            spread_bus_y = fanout_tee_y
        spread_bus_y = max(
            spread_bus_y,
            _connector_min_bus_y_above_target(target, gap=gap),
            _min_bus_y_clearing_horizontal_corridor(
                source.cx,
                entry_x,
                route_obstacles,
                proposed_y=spread_bus_y,
            ),
        )
        return _shared_merge_bus_connector_points(
            source,
            target,
            entry_x,
            spread_bus_y,
            route_obstacles,
            gap=gap,
            spine_tee_y=fanout_tee_y,
            graph=graph,
            positions=positions,
            link_key=link_key,
            prefer_tee_branch=prefer_tee_branch,
        )
    if (
        not inline_binary_side_entry
        and not combine_side_entry
        and not combine_top_entry
        and positions[src].spec.synthetic != SYNTHETIC_TENSOR
        and abs(source.cx - target.cx) < 0.08
    ):
        straight = _same_column_straight_connector_points(source, target, gap=gap)
        if (
            not _path_hits_obstacles(
                straight,
                route_obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            )
            and not _path_penetrates_attached_boxes(straight, source, target)
        ):
            return straight
    if (
        not inline_binary_side_entry
        and not combine_side_entry
        and not combine_top_entry
        and positions[src].spec.synthetic != SYNTHETIC_TENSOR
        and src not in source_bus
        and link_key not in merge_entry_x
        and _needs_cross_column_side_entry(source, target)
    ):
        return _cross_column_side_entry_connector_points(
            source,
            target,
            route_obstacles,
            bus_y=bus_y,
        )
    if (
        src in source_bus
        and link_key in merge_link_bus
        and link_key not in merge_entry_x
        and not inline_binary_side_entry
        and not combine_side_entry
        and not combine_top_entry
    ):
        return _shared_merge_bus_connector_points(
            source,
            target,
            target.cx,
            merge_link_bus[link_key],
            route_obstacles,
            gap=gap,
            spine_tee_y=fanout_tee_y,
            graph=graph,
            positions=positions,
            link_key=link_key,
            prefer_tee_branch=prefer_tee_branch,
        )
    return _orthogonal_path(
        source,
        target,
        route_obstacles,
        bus_near=bus_near,
        bus_y=bus_y,
        graph=graph,
        positions=positions,
    )


def _compute_detail_connector_buses(
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    label_obstacles: list[_RenderAnchor],
) -> tuple[
    dict[int, float],
    dict[int, float],
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
]:
    """Compute shared merge buses for detail-graph connector routing."""
    target_bus: dict[int, float] = {}
    for tgt, link_group in incoming.items():
        main_links = _fanout_links_excluding_bypasses(graph, link_group)
        if not _should_use_shared_connector_bus(len(main_links)):
            continue
        if _is_combine_synthetic(positions[tgt].spec.synthetic) and any(
            (src, tgt) in graph.side_entry_links for src, _ in main_links
        ):
            continue
        target_anchor = anchors.get(tgt)
        source_anchors = [anchors[src] for src, _ in main_links if src in anchors]
        if target_anchor is None or len(source_anchors) < 2:
            continue
        involved = {tgt, *(src for src, _ in main_links)}
        route_obstacles = [
            anchor for node_index, anchor in anchors.items() if node_index not in involved
        ] + label_obstacles + _inline_frame_bounds_obstacles(
            graph,
            positions,
            exclude_nodes=involved,
        )
        bus_y = _compute_shared_target_bus_y(
            source_anchors,
            target_anchor,
            route_obstacles,
        )
        pipeline_bus = _pipeline_merge_bus_y(
            graph,
            positions,
            anchors,
            tgt,
            [src for src, _ in main_links],
        )
        min_bus_y = _connector_min_bus_y_above_target(target_anchor)
        if pipeline_bus is not None:
            bus_y = max(min_bus_y, min(bus_y, pipeline_bus))
        else:
            bus_y = max(min_bus_y, bus_y)
        target_bus[tgt] = bus_y

    source_bus: dict[int, float] = {}
    for src, link_group in outgoing.items():
        main_links = _fanout_links_excluding_bypasses(graph, link_group)
        if not _should_use_shared_source_bus(len(main_links)):
            continue
        source_anchor = anchors.get(src)
        target_anchors = [anchors[tgt] for _, tgt in main_links if tgt in anchors]
        if source_anchor is None or len(target_anchors) < 2:
            continue
        involved = {src, *(tgt for _, tgt in main_links)}
        route_obstacles = [
            anchor for node_index, anchor in anchors.items() if node_index not in involved
        ] + label_obstacles + _inline_frame_bounds_obstacles(
            graph,
            positions,
            exclude_nodes=involved,
        )
        source_bus[src] = _compute_shared_source_bus_y(
            source_anchor,
            target_anchors,
            route_obstacles,
            graph=graph,
            positions=positions,
        )

    for src in list(source_bus):
        if not _source_fanout_splits_before_target_bus(graph, src, outgoing, target_bus):
            continue
        source_anchor = anchors.get(src)
        merge_bus_y = _fanout_lowest_target_merge_bus_y(
            graph,
            src,
            outgoing,
            target_bus,
        )
        if source_anchor is None or merge_bus_y is None:
            continue
        source_bus[src] = _compute_fanout_split_tee_y(source_anchor, merge_bus_y)

    merge_entry_x: dict[tuple[int, int], float] = {}
    merge_link_bus: dict[tuple[int, int], float] = {}
    for tgt, link_group in incoming.items():
        main_links = _fanout_links_excluding_bypasses(graph, link_group)
        top_main_links = [
            link
            for link in main_links
            if link not in graph.side_entry_links
            and link not in graph.inline_binary_operand_links
        ]
        if len(top_main_links) < SHARED_SOURCE_BUS_MIN_LINKS:
            continue
        target_pos = positions[tgt]
        target_anchor = anchors.get(tgt)
        if target_anchor is None:
            continue
        sorted_links = sorted(top_main_links, key=lambda link: positions[link[0]].cx)
        if tgt in target_bus:
            base_bus = target_bus.get(tgt)
            for link in sorted_links:
                merge_entry_x[link] = target_anchor.cx
                if base_bus is not None:
                    merge_link_bus[link] = base_bus
            continue
        if _should_use_shared_connector_bus(len(main_links)):
            spread_links = [
                link for link in sorted_links if link in graph.link_port_labels
            ]
            if not spread_links:
                spread_links = sorted_links
        else:
            spread_links = sorted_links
        if not spread_links:
            continue
        _assign_spread_merge_entry_x(
            spread_links,
            target_anchor,
            target_pos,
            positions,
            anchors,
            merge_entry_x,
        )
        base_bus = target_bus.get(tgt)
        if base_bus is None and len(spread_links) >= 2:
            spread_sources = [anchors[src] for src, _ in spread_links if src in anchors]
            involved = {tgt, *(src for src, _ in spread_links)}
            route_obstacles = [
                anchor for node_index, anchor in anchors.items() if node_index not in involved
            ] + label_obstacles + _inline_frame_bounds_obstacles(
                graph,
                positions,
                exclude_nodes=involved,
            )
            if len(spread_sources) >= 2:
                base_bus = _compute_shared_target_bus_y(
                    spread_sources,
                    target_anchor,
                    route_obstacles,
                )
                base_bus = max(
                    _connector_min_bus_y_above_target(target_anchor),
                    base_bus,
                )
        if base_bus is not None and len(spread_links) >= 2:
            for link in spread_links:
                merge_link_bus[link] = base_bus

    for src, link_group in outgoing.items():
        if src not in source_bus:
            continue
        main_links = _fanout_links_excluding_bypasses(graph, link_group)
        top_main_links = [
            link
            for link in main_links
            if link not in graph.side_entry_links
            and link not in graph.inline_binary_operand_links
            and link in merge_entry_x
        ]
        for link in top_main_links:
            if link in merge_link_bus:
                continue
            if (
                _source_fanout_splits_before_target_bus(graph, src, outgoing, target_bus)
                and link[1] not in target_bus
            ):
                merge_link_bus[link] = source_bus[src]
                continue
            bus_y = _fanout_source_bus_y(
                graph,
                src,
                link,
                positions=positions,
                outgoing=outgoing,
                source_bus=source_bus,
                target_bus=target_bus,
            )
            if bus_y is not None:
                merge_link_bus[link] = bus_y

    for src in list(source_bus):
        source_anchor = anchors.get(src)
        if source_anchor is None:
            continue
        main_links = _fanout_links_excluding_bypasses(graph, outgoing.get(src, []))
        target_anchors = [anchors[tgt] for _, tgt in main_links if tgt in anchors]
        if not target_anchors:
            continue
        if not _source_fanout_splits_before_target_bus(graph, src, outgoing, target_bus):
            xs = [source_anchor.cx, *(target.cx for target in target_anchors)]
            source_bus[src] = _clamp_bus_y_clearing_inline_frames(
                source_bus[src],
                graph=graph,
                positions=positions,
                x_left=min(xs),
                x_right=max(xs),
            )
        source_bus[src] = _effective_source_bus_y(
            source_anchor,
            target_anchors,
            source_bus[src],
        )

    for src in list(source_bus):
        if _source_fanout_splits_before_target_bus(graph, src, outgoing, target_bus):
            continue
        for _, tgt in outgoing.get(src, []):
            link = (src, tgt)
            if link in graph.inline_binary_operand_links or link in graph.side_entry_links:
                continue
            if tgt in target_bus:
                continue
            target_anchor = anchors.get(tgt)
            if target_anchor is None:
                continue
            merge_link_bus[link] = _connector_min_bus_y_above_target(target_anchor)

    return target_bus, source_bus, merge_entry_x, merge_link_bus


def compute_detail_connector_bounds(
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    links: list[tuple[int, int]],
    *,
    label_obstacles: list[_RenderAnchor] | None = None,
) -> list[ContentBounds]:
    """Estimate connector corridors for layout (matches detail-graph routing)."""
    from collections import defaultdict

    from visualizer.text_measure import ContentBounds

    label_obstacles = list(label_obstacles or [])
    incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
    outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for src, tgt in links:
        incoming[tgt].append((src, tgt))
        outgoing[src].append((src, tgt))

    input_index = next(
        (index for index, spec in enumerate(graph.nodes) if spec.synthetic == SYNTHETIC_INPUT),
        None,
    )

    target_bus, source_bus, merge_entry_x, merge_link_bus = _compute_detail_connector_buses(
        graph,
        positions,
        anchors,
        incoming,
        outgoing,
        label_obstacles,
    )

    link_paths = _collect_detail_link_paths(
        graph=graph,
        links=links,
        positions=positions,
        anchors=anchors,
        incoming=incoming,
        label_obstacles=label_obstacles,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_entry_x=merge_entry_x,
        merge_link_bus=merge_link_bus,
        input_index=input_index,
    )
    return [_polyline_bounds(points) for points in link_paths.values()]


def _draw_graph_connector(
    ax,
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    color: str | None = None,
    linestyle: str = "solid",
    side_entry: bool = False,
    inline_port: bool = False,
    floating_port: bool = False,
    combine_side_entry: bool = False,
    combine_top_entry: bool = False,
    combine_center_y: float | None = None,
    bus_near: str = "target",
    bus_y: float | None = None,
    zorder: float = FLOW_CONNECTOR_ZORDER,
) -> None:
    """Draw an orthogonal connector that routes around node boxes when needed."""
    gap = 0.04
    if combine_side_entry and combine_center_y is not None:
        points = _side_entry_combine_connector_points(
            source,
            target.cx,
            combine_center_y,
            gap=gap,
            obstacles=obstacles,
        )
    elif combine_top_entry and combine_center_y is not None:
        points = _top_entry_combine_connector_points(
            source,
            target.cx,
            combine_center_y,
            gap=gap,
        )
    elif inline_port and side_entry:
        points = _inline_dashed_port_connector_points(source, target, gap=gap, bus_y=bus_y)
    elif floating_port:
        points = _orthogonal_path(source, target, obstacles, bus_near=bus_near, bus_y=bus_y)
    elif side_entry and abs(source.cx - target.cx) < 0.08:
        y_start = _connector_source_bottom_exit_y(source, gap=gap)
        y_end = _connector_target_top_entry_y(target, gap=gap)
        points = [(source.cx, y_start), (source.cx, y_end)]
    else:
        points = _orthogonal_path(source, target, obstacles, bus_near=bus_near, bus_y=bus_y)
    _draw_path(ax, points, color=color, linestyle=linestyle, zorder=zorder)


def _draw_floating_port_label(
    ax,
    label: str,
    x: float,
    y: float,
    *,
    ha: str = "right",
    va: str = "center",
) -> None:
    """Draw a branch port label beside a node with a small background pad."""
    ax.text(
        x,
        y,
        label,
        ha=ha,
        va=va,
        fontsize=7.2,
        color=COLORS["muted"],
        fontweight="bold",
        zorder=10,
        bbox={
            "boxstyle": "round,pad=0.10",
            "facecolor": COLORS["detail_fill"],
            "edgecolor": "none",
            "alpha": 1.0,
        },
    )


def _detail_tile_visual_inset() -> float:
    """Extra draw margin around a tile beyond its layout width/height."""
    return DETAIL_TILE_ROUNDING + DETAIL_TILE_BOX_PAD


def _detail_frame_edge_pad() -> float:
    """Padding from tile visuals to the outer detail-frame rectangle."""
    return DETAIL_FRAME_BOX_PAD + DETAIL_FRAME_STROKE + DETAIL_FRAME_GAP + _detail_tile_visual_inset()


def _detail_content_extents(
    positions: list[LayoutPosition],
) -> tuple[float, float, float, float]:
    """Return visual (left, right, bottom, top) bounds of tiles and port labels."""
    tile_inset = _detail_tile_visual_inset()
    min_left = float("inf")
    max_right = float("-inf")
    min_bottom = float("inf")
    max_top = float("-inf")

    for pos in positions:
        min_left = min(min_left, _node_content_left(pos) - tile_inset)
        max_right = max(max_right, _node_content_right(pos) + tile_inset)
        min_bottom = min(min_bottom, pos.bottom - tile_inset)

        top = pos.top_y + tile_inset
        if pos.spec.port_style == "floating" and pos.spec.port_label:
            label_y = pos.top_y - pos.height / 2
            top = max(top, label_y + 0.12)
            label_x = pos.cx - pos.width / 2 - 0.10
            min_left = min(min_left, label_x - 0.38)
        max_top = max(max_top, top)

    return min_left, max_right, min_bottom, max_top


def _detail_content_bounds(
    positions: list[LayoutPosition],
) -> tuple[float, float, float, float]:
    """Return (left, right, bottom, top) bounds for a detailed block frame."""
    min_left, max_right, min_bottom, max_top = _detail_content_extents(positions)
    pad_x = _detail_frame_edge_pad() + DETAIL_FRAME_ROUNDING
    pad_y = _detail_frame_edge_pad() + DETAIL_FRAME_PAD_Y_EXTRA

    frame_left = min_left - pad_x
    frame_right = max_right + pad_x
    frame_bottom = min_bottom - pad_y
    frame_top = max_top + pad_y
    return frame_left, frame_right, frame_bottom, frame_top


def _format_input_source_sublabel(source: str | None) -> str | None:
    """Format upstream input source as a single-line connector label."""
    if not source:
        return None
    head = source.split(" in ", 1)[0]
    return f"← {head}"


def _resize_input_nodes(
    positions: list[LayoutPosition],
    input_sublabel: str | None,
) -> None:
    """Grow input and tensor port tiles so upstream hints fit inside the box."""
    for pos in positions:
        spec = pos.spec
        if spec.synthetic == SYNTHETIC_INPUT:
            if not input_sublabel and spec.diagram_width is None:
                continue
        elif spec.synthetic == SYNTHETIC_TENSOR:
            if not spec.sublabel and spec.diagram_width is None:
                continue
        else:
            continue
        if spec.diagram_width is None or spec.diagram_height is None:
            continue
        pos.width = spec.diagram_width
        pos.height = spec.diagram_height


def _spine_expanded_block_top_y(cursor_y: float) -> float:
    """Place an expanded spine section below cursor_y, clearing room for its frame label."""
    return cursor_y - SPINE_EXPANDED_BLOCK_TOP_RESERVE


def _render_inline_linear_frames(
    ax,
    graph,
    positions: list,
    *,
    enabled: bool,
    plan: DetailDrawPlan | None = None,
) -> None:
    """Draw dotted frames around steps inlined from straight-line composite sub-blocks."""
    if not enabled or not graph.inline_frames:
        return

    label_placements = plan.inline_frame_labels if plan is not None else {}

    for frame in graph.inline_frames:
        if not frame.node_indices:
            continue
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        frame_left = bounds.left
        frame_bottom = bounds.bottom
        frame_w = bounds.right - bounds.left
        frame_h = bounds.top - bounds.bottom

        patch = FancyBboxPatch(
            (frame_left, frame_bottom),
            frame_w,
            frame_h,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            linewidth=1.2,
            edgecolor=COLORS["detail_border"],
            facecolor="none",
            linestyle=(0, (2, 2)),
            zorder=0.5,
        )
        ax.add_patch(patch)

        placement = label_placements.get(frame.frame_id)
        if placement is not None and placement.lines:
            for line in placement.lines:
                ax.text(
                    line.x,
                    line.y,
                    line.text,
                    ha=line.ha,
                    va=line.va,
                    fontsize=line.fontsize,
                    color=COLORS["muted"],
                    fontweight=line.fontweight,
                    style=line.style or "normal",
                )
            continue

        caption_top = bounds.top + INLINE_FRAME_LABEL_GAP
        label_lines = _inline_frame_label_lines(frame.label, frame_w)
        for line_index, line in enumerate(label_lines):
            ax.text(
                frame_left + 0.02,
                caption_top - line_index * INLINE_FRAME_LABEL_LINE_H,
                line,
                ha="left",
                va="bottom",
                fontsize=6.4,
                color=COLORS["muted"],
            )
        if frame.sublabel:
            sub_lines = [line for line in frame.sublabel.split("\n") if line.strip()]
            for line_index, line in enumerate(sub_lines):
                ax.text(
                    frame_left + 0.02,
                    bounds.top + INLINE_FRAME_LABEL_GAP - 0.11 - line_index * 0.11,
                    line,
                    ha="left",
                    va="bottom",
                    fontsize=5.6,
                    color=COLORS["muted"],
                    style="italic",
                )


def _build_detail_draw_plan(
    positions: list[LayoutPosition],
    graph,
    *,
    input_sublabel: str | None,
) -> DetailDrawPlan:
    """Build node/label draw descriptors without painting (for measurement + validation)."""
    plan = DetailDrawPlan(input_sublabel=input_sublabel)
    inline_frame_members = {
        node_index
        for frame in graph.inline_frames
        for node_index in frame.node_indices
    }

    for index, pos in enumerate(positions):
        spec = pos.spec

        if spec.synthetic == SYNTHETIC_INPUT:
            input_leaf = _make_node(
                spec.key,
                pos.cx,
                pos.top_y,
                pos.width,
                pos.height,
                spec.label,
                COLORS["pos"],
                text_color=COLORS["text"],
                sublabel=input_sublabel,
                fontsize=7.2,
                pad_x=BLOCK_PAD_X,
                pad_y=BLOCK_PAD_Y,
            )
            plan.node_draws.append((input_leaf, {"edgecolor": _INPUT_NODE_EDGE}))
            continue

        if spec.synthetic == SYNTHETIC_TENSOR:
            port_leaf = _make_node(
                spec.key,
                pos.cx,
                pos.top_y,
                pos.width,
                pos.height,
                spec.label,
                COLORS["pos"],
                text_color=COLORS["text"],
                sublabel=spec.sublabel,
                fontsize=7.2,
                pad_x=INPUT_PAD_X,
                pad_y=TENSOR_PORT_PAD_Y,
            )
            plan.node_draws.append((port_leaf, {}))
            continue

        if spec.synthetic == SYNTHETIC_HIDDEN:
            hidden_leaf = _make_node(
                spec.key,
                pos.cx,
                pos.top_y,
                pos.width,
                pos.height,
                spec.label,
                COLORS["bg"],
                text_color=COLORS["residual"],
                fontsize=6.5,
                pad_x=BLOCK_PAD_X,
                pad_y=BLOCK_PAD_Y,
            )
            plan.node_draws.append((hidden_leaf, {}))
            continue

        if _is_combine_synthetic(spec.synthetic):
            center_y = pos.top_y - pos.height / 2
            symbol = spec.label or "×"
            plan.combine_ops.append((pos.cx, center_y, symbol, spec.sublabel))
            continue

        block = spec.block
        if block is not None and is_method_wrapper(block):
            display_label, attr = wrapper_bullet_lines(block)
            leaf = _make_node(
                spec.key,
                pos.cx,
                pos.top_y,
                pos.width,
                pos.height,
                display_label,
                COLORS["detail_fill"],
                sublabel=None,
                fontsize=7.4,
            )
            plan.node_draws.append(
                (
                    leaf,
                    {"edgecolor": COLORS["detail_border"], "linestyle": "dashed"},
                )
            )
            if spec.port_label and spec.port_style == "inline":
                plan.branch_labels.append(
                    (
                        spec.port_label,
                        pos.cx,
                        pos.top_y + 0.14,
                        "center",
                        "bottom",
                    )
                )
            continue

        display_label = spec.label
        sublabel: str | None
        if spec.sublabel is not None:
            sublabel = spec.sublabel or None
        else:
            display_label, sublabel = tile_display_labels(
                block,
                spec_label=spec.label,
                in_inline_frame=index in inline_frame_members,
                port_label=spec.port_label,
                port_style=spec.port_style,
            )

        if spec.port_label:
            if spec.port_style == "floating":
                label_x = pos.cx - pos.width / 2 - 0.10
                label_y = pos.top_y - pos.height / 2
                plan.branch_labels.append(
                    (
                        spec.port_label,
                        label_x,
                        label_y,
                        "right",
                        "center",
                    )
                )
                plan.label_obstacles.append(
                    _RenderAnchor(
                        cx=label_x - 0.18,
                        top=label_y + 0.10,
                        bottom=label_y - 0.10,
                        left=label_x - 0.36,
                        right=label_x + 0.02,
                    )
                )

        facecolor = COLORS["basic_op"]
        if block is not None:
            facecolor = _detail_block_facecolor(block)
        text_color = _detail_tile_text_color(facecolor)

        leaf = _make_node(
            spec.key,
            pos.cx,
            pos.top_y,
            pos.width,
            pos.height,
            display_label,
            facecolor,
            sublabel=sublabel,
            fontsize=7.6,
            text_color=text_color,
        )
        plan.node_draws.append((leaf, {}))

    return plan


def _append_tile_purpose_annotations(
    plan: DetailDrawPlan,
    *,
    graph,
    positions: list[LayoutPosition],
) -> None:
    """Add purpose lines below tiles after layout validation (not measured as obstacles)."""
    from visualizer.block_tree import tile_purpose_annotation

    inline_frame_members = {
        node_index
        for frame in graph.inline_frames
        for node_index in frame.node_indices
    }
    draw_index = 0
    for index, pos in enumerate(positions):
        spec = pos.spec
        if spec.synthetic in {SYNTHETIC_INPUT, SYNTHETIC_TENSOR, SYNTHETIC_HIDDEN}:
            continue
        if _is_combine_synthetic(spec.synthetic):
            continue
        if draw_index >= len(plan.node_draws):
            continue
        _leaf, _ = plan.node_draws[draw_index]
        draw_index += 1
        block = spec.block
        if block is None or spec.sublabel:
            continue
        if index in inline_frame_members and not block.details:
            continue
        purpose = tile_purpose_annotation(block)
        if purpose:
            plan.branch_labels.append(
                (purpose, pos.cx, pos.bottom - 0.04, "center", "top")
            )


def _append_input_source_branch_label(
    plan: DetailDrawPlan,
    *,
    graph,
    links: list[tuple[int, int]],
    anchors: dict[int, _RenderAnchor],
    positions: list[LayoutPosition],
    input_source_label: str | None,
) -> None:
    """Upstream source hints render inside input tiles via ``input_sublabel``."""
    del plan, graph, links, anchors, positions, input_source_label


def _anchors_from_detail_plan(
    positions: list[LayoutPosition],
    plan: DetailDrawPlan,
) -> dict[int, _RenderAnchor]:
    """Build connector anchors from finalized layout positions."""
    del plan  # kept for call-site compatibility; anchors follow positions, not draw order
    anchors: dict[int, _RenderAnchor] = {}
    for index, pos in enumerate(positions):
        spec = pos.spec
        if _is_combine_synthetic(spec.synthetic):
            from visualizer.ast_analyze import is_compact_combine_label

            center_y = pos.top_y - pos.height / 2
            if is_compact_combine_label(spec.label):
                anchors[index] = _merge_anchor(pos.cx, center_y)
            else:
                anchors[index] = _anchor_from_position(pos)
            continue
        anchors[index] = _anchor_from_position(pos)
    return anchors


def _render_laid_out_computation_graph(
    layout: DiagramLayout,
    ax,
    root: BlockNode,
    *,
    cx: float,
    top_y: float,
    block_w: float,
    input_sublabel: str | None = None,
    prefix_steps: list[BlockNode] | None = None,
    inline_linear_frames: bool = True,
    draw_section_frame: bool = True,
    root_frame_label: str | None = None,
    include_input: bool = True,
    min_left: float | None = None,
    forbidden_regions: list | None = None,
    basic_ops: BasicOpFilter | None = None,
) -> tuple[float, float, _RenderAnchor | None]:
    """Lay out a computation graph with graph-layout and draw it."""
    from visualizer.render_validate import LAYOUT_MIN_TOP_Y, finalize_detail_layout

    graph = build_computation_graph(
        root,
        prefix_steps=prefix_steps,
        include_input=include_input,
        basic_ops=basic_ops,
    )
    if not graph.nodes:
        return top_y, cx + block_w / 2, None
    from visualizer.computation_graph import _minimum_graph_layout_width

    block_w = max(block_w, _minimum_graph_layout_width(graph) + 0.12)
    if root_frame_label:
        add_root_pipeline_frame(graph, root, label=root_frame_label)
    from visualizer.computation_graph import measure_graph_node_sizes
    from visualizer.text_measure import ensure_diagram_measure_axes

    ensure_diagram_measure_axes(ax)
    measure_graph_node_sizes(ax, graph, input_sublabel=input_sublabel)
    est_h = _estimate_graph_height(graph)
    layout_seed_y = max(top_y, LAYOUT_MIN_TOP_Y)
    positions, links = layout_computation_graph(
        graph,
        cx=cx,
        top_y=layout_seed_y,
        block_w=block_w,
        block_h=est_h,
        content_left=min_left,
    )
    if not positions:
        return top_y - est_h, cx + block_w / 2, None

    plan = finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=input_sublabel,
        cx=cx,
        top_y=top_y,
        detail_fill=COLORS["detail_fill"],
        min_left=min_left,
        forbidden_regions=forbidden_regions,
    )

    from visualizer.computation_graph import _dock_single_consumer_tensor_ports, _graph_has_tensor_ports

    if _graph_has_tensor_ports(graph):
        _dock_single_consumer_tensor_ports(positions, graph)
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)

    anchors = _anchors_from_detail_plan(positions, plan)
    _append_input_source_branch_label(
        plan,
        graph=graph,
        links=links,
        anchors=anchors,
        positions=positions,
        input_source_label=input_sublabel,
    )
    _append_tile_purpose_annotations(plan, graph=graph, positions=positions)

    frame_left, frame_right, frame_bottom, frame_top = _detail_content_bounds(positions)
    frame_w = frame_right - frame_left
    frame_h = frame_top - frame_bottom
    block_patch = FancyBboxPatch(
        (frame_left, frame_bottom),
        frame_w,
        frame_h,
        boxstyle="round,pad=0.01,rounding_size=0.10",
        linewidth=2.0,
        edgecolor=COLORS["detail_border"],
        facecolor=COLORS["detail_fill"],
        zorder=0,
    )
    if draw_section_frame:
        ax.add_patch(block_patch)
    _render_inline_linear_frames(ax, graph, positions, enabled=inline_linear_frames, plan=plan)

    label_obstacles = list(plan.label_obstacles)

    input_index = next(
        (index for index, spec in enumerate(graph.nodes) if spec.synthetic == SYNTHETIC_INPUT),
        None,
    )

    incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
    outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for link in links:
        src, tgt = link
        incoming[tgt].append(link)
        outgoing[src].append(link)

    target_bus, source_bus, merge_entry_x, merge_link_bus = _compute_detail_connector_buses(
        graph,
        positions,
        anchors,
        incoming,
        outgoing,
        label_obstacles,
    )

    merge_link_labels: list[tuple[str, float, float]] = []

    for leaf, draw_kwargs in plan.node_draws:
        layout.add(leaf)
        _draw_box(ax, leaf, **draw_kwargs)

    link_paths = _collect_detail_link_paths(
        graph=graph,
        links=links,
        positions=positions,
        anchors=anchors,
        incoming=incoming,
        label_obstacles=label_obstacles,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_entry_x=merge_entry_x,
        merge_link_bus=merge_link_bus,
        input_index=input_index,
    )
    for src, tgt in links:
        link_key = (src, tgt)
        port_label = graph.link_port_labels.get(link_key)
        if not (port_label and link_key in merge_entry_x):
            continue
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue
        link_paths[link_key] = _labeled_merge_connector_points(
            source,
            target,
            merge_entry_x[link_key],
            bus_y=merge_link_bus.get(link_key),
        )

    _assert_detail_connector_linestyles_are_solid(
        graph,
        links=links,
        positions=positions,
        anchors=anchors,
    )

    for src, tgt in links:
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue
        link_key = (src, tgt)
        port_label = graph.link_port_labels.get(link_key)
        connector_style = _detail_connector_linestyle(
            graph,
            src=src,
            positions=positions,
            source=source,
            target=target,
        )
        if port_label and link_key in merge_entry_x:
            merge_link_labels.append((port_label, merge_entry_x[link_key], target.top + 0.05))
        points = link_paths.get(link_key)
        if points is None or len(points) < 2:
            continue
        _draw_path(
            ax,
            points,
            linestyle=connector_style,
            zorder=DETAIL_CONNECTOR_ZORDER,
        )

    _draw_connector_junction_dots(
        ax,
        link_paths,
        obstacles=list(anchors.values()) + label_obstacles,
        combine_ops=[(op_x, op_y) for op_x, op_y, _symbol, _sublabel in plan.combine_ops],
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
        incoming=incoming,
        anchors=anchors,
        graph=graph,
        outgoing=outgoing,
    )

    for op_x, op_y, symbol, op_sublabel in plan.combine_ops:
        width = height = None
        for pos in positions:
            if not _is_combine_synthetic(pos.spec.synthetic):
                continue
            center_y = pos.top_y - pos.height / 2
            if abs(pos.cx - op_x) < 1e-6 and abs(center_y - op_y) < 1e-6 and pos.spec.label == symbol:
                width, height = pos.width, pos.height
                break
        _draw_combine_op(
            ax,
            op_x,
            op_y,
            symbol,
            sublabel=op_sublabel,
            width=width,
            height=height,
        )

    for label, x, y, ha, va in plan.branch_labels:
        _draw_floating_port_label(ax, label, x, y, ha=ha, va=va)

    for label, x, y in merge_link_labels:
        _draw_floating_port_label(ax, label, x, y, ha="center", va="bottom")

    input_anchor: _RenderAnchor | None = None
    for index, pos in enumerate(positions):
        if pos.spec.synthetic == SYNTHETIC_INPUT:
            input_anchor = anchors.get(index)
            break
    if input_anchor is None:
        for index, pos in enumerate(positions):
            if pos.spec.synthetic in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}:
                continue
            if _is_combine_synthetic(pos.spec.synthetic):
                continue
            input_anchor = anchors.get(index)
            break

    return frame_bottom, frame_right, input_anchor


def _render_block_tree_node(
    layout: DiagramLayout,
    ax,
    node: BlockNode,
    *,
    cx: float,
    top_y: float,
    block_w: float,
    input_sublabel: str | None = None,
    prefix_steps: list[BlockNode] | None = None,
    inline_linear_frames: bool = True,
    min_left: float | None = None,
    forbidden_regions: list | None = None,
    basic_ops: BasicOpFilter | None = None,
) -> tuple[float, float | None]:
    """Render a block tree node via computation-graph layout (always includes forward input)."""
    if is_method_wrapper(node):
        return top_y, None

    bottom, frame_right, _input_anchor = _render_laid_out_computation_graph(
        layout,
        ax,
        node,
        cx=cx,
        top_y=top_y,
        block_w=block_w - 0.15,
        input_sublabel=input_sublabel,
        prefix_steps=prefix_steps,
        inline_linear_frames=inline_linear_frames,
        min_left=min_left,
        forbidden_regions=forbidden_regions,
        basic_ops=basic_ops,
    )
    return bottom, frame_right


def _detail_sections_to_render(spec: ArchitectureSpec) -> list[tuple[str, BlockNode, str | None]]:
    """Return titled block trees rendered as internal diagram subsections."""
    sections: list[tuple[str, BlockNode, str | None]] = []
    basic_ops = spec.basic_ops or BasicOpFilter.for_detailed()
    for title, tree in spec.detailed_block_trees:
        sections.append((title, tree, _format_input_source_sublabel(tree.input_source)))
        for sub_title, sub_tree in collect_nested_diagrams(tree, basic_ops=basic_ops):
            if is_single_function_tree(sub_tree):
                if _omit_from_detailed_view(sub_tree):
                    continue
                if not _show_single_function_in_diagram(sub_tree):
                    continue
            sections.append((sub_title, sub_tree, _format_input_source_sublabel(sub_tree.input_source)))
    return sections


def _render_diagram_section(
    layout: DiagramLayout,
    ax,
    *,
    cx: float,
    cursor: float,
    block_w: float,
    title: str,
    tree: BlockNode,
    input_sublabel: str | None = None,
    prefix_steps: list[BlockNode] | None = None,
    inline_linear_frames: bool = True,
    min_left: float | None = None,
    forbidden_regions: list | None = None,
    basic_ops: BasicOpFilter | None = None,
) -> float:
    """Render one titled block diagram. Returns y below the diagram."""
    title_y = cursor
    frame_top_pad = _detail_frame_edge_pad()
    diagram_top = cursor - SECTION_TITLE_HEIGHT - SECTION_TITLE_GAP - frame_top_pad
    title_x = min_left if min_left is not None else cx
    ax.text(
        title_x,
        title_y,
        title,
        ha="left" if min_left is not None else "center",
        va="bottom",
        fontsize=PANEL_TITLE_FONT,
        color=PANEL_TITLE_COLOR,
        fontweight="bold",
    )
    diagram_bottom, _frame_right = _render_block_tree(
        layout,
        ax,
        tree,
        cx=cx,
        top_y=diagram_top,
        block_w=block_w - 0.2,
        input_sublabel=input_sublabel,
        prefix_steps=prefix_steps,
        inline_linear_frames=inline_linear_frames,
        min_left=min_left,
        forbidden_regions=forbidden_regions,
        basic_ops=basic_ops,
    )
    return diagram_bottom - 0.5


def _render_block_tree(
    layout: DiagramLayout,
    ax,
    node: BlockNode,
    *,
    cx: float,
    top_y: float,
    block_w: float,
    input_sublabel: str | None = None,
    prefix_steps: list[BlockNode] | None = None,
    inline_linear_frames: bool = True,
    min_left: float | None = None,
    forbidden_regions: list | None = None,
    basic_ops: BasicOpFilter | None = None,
) -> tuple[float, float | None]:
    """Render one recursive block tree from top_y downward."""
    return _render_block_tree_node(
        layout,
        ax,
        node,
        cx=cx,
        top_y=top_y,
        block_w=block_w,
        input_sublabel=input_sublabel,
        prefix_steps=prefix_steps,
        inline_linear_frames=inline_linear_frames,
        min_left=min_left,
        forbidden_regions=forbidden_regions,
        basic_ops=basic_ops,
    )


def _render_detailed_internals(
    layout: DiagramLayout,
    ax,
    spec: ArchitectureSpec,
    *,
    cx: float,
    start_y: float,
    panel_x: float | None = None,
    panel_w: float = PANEL_W,
    wrap_width: int = PANEL_WRAP_WIDTH,
    inline_linear_frames: bool = True,
    forbidden_regions: list | None = None,
    compact_header: bool = False,
) -> float:
    """Render recursive internal block diagrams below the main model."""
    from visualizer.render_validate import measure_detail_tree_content_width

    panel_x = panel_x if panel_x is not None else cx + 3.0
    detail_min_left = DIAGRAM_LEFT_MARGIN + 0.05

    section_drop = 0.22 if compact_header else 0.35
    cursor = start_y - section_drop

    if not spec.detailed_block_trees:
        ax.text(
            cx,
            cursor,
            "Detailed view requires modeling source (omit --config-only)",
            ha="center",
            va="center",
            fontsize=8.5,
            color=COLORS["muted"],
        )
        return cursor - 0.35

    bottom = cursor
    basic_ops = spec.basic_ops or BasicOpFilter.for_detailed()
    for title, tree, input_sublabel in _detail_sections_to_render(spec):
        section_w = measure_detail_tree_content_width(
            ax,
            tree,
            cx=cx,
            detail_fill=COLORS["detail_fill"],
            min_left=detail_min_left,
            input_sublabel=input_sublabel,
            basic_ops=basic_ops,
        )
        cursor = _render_diagram_section(
            layout,
            ax,
            cx=cx,
            cursor=cursor,
            block_w=section_w,
            title=title,
            tree=tree,
            input_sublabel=input_sublabel,
            inline_linear_frames=inline_linear_frames,
            min_left=detail_min_left,
            forbidden_regions=forbidden_regions,
            basic_ops=basic_ops,
        )
        bottom = cursor

    return bottom


def render_diagram(
    spec: ArchitectureSpec,
    output: str | Path,
    *,
    dpi: int = 150,
    title: str | None = None,
    detailed: bool = False,
    inline_linear_frames: bool = True,
) -> Path:
    """Render an architecture diagram to SVG (default), PNG, or PDF."""
    output_path = Path(output).expanduser().resolve()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".svg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas_width = 11.0
    top_y = 12.55
    bottom_margin = 0.35
    fact_gap = FACT_SHEET_GAP
    fact_w = PANEL_W
    wrap_width = PANEL_WRAP_WIDTH

    fact_h = _fact_sheet_height(spec, wrap_width=wrap_width)
    repeat_label = _repeat_block_label(spec)
    decoder_label = spec.decoder_class or "Transformer block"

    fig, ax = plt.subplots(figsize=(MEASURE_CANVAS_WIDTH, 13), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, MEASURE_CANVAS_WIDTH)
    ax.set_ylim(0, 13)
    fig.canvas.draw()

    norm_w, inner_w = _block_content_widths(ax, spec)
    block_w = _main_block_width(
        ax,
        repeat_label=repeat_label,
        decoder_label=decoder_label,
        inner_w=inner_w,
    )
    cx = DIAGRAM_LEFT_MARGIN + block_w / 2
    main_model_right = cx + block_w / 2
    fact_x = _fact_sheet_x(main_model_right, gap=fact_gap)
    canvas_width = max(
        canvas_width,
        fact_x + fact_w + DIAGRAM_RIGHT_MARGIN,
    )
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, 13)
    fig.set_size_inches(canvas_width, 13)
    fig.canvas.draw()
    # Re-measure at the final axis transform so label widths match the saved figure.
    block_w = _main_block_width(
        ax,
        repeat_label=repeat_label,
        decoder_label=decoder_label,
        inner_w=inner_w,
    )
    cx = DIAGRAM_LEFT_MARGIN + block_w / 2
    main_model_right = cx + block_w / 2
    fact_x = _fact_sheet_x(main_model_right, gap=fact_gap)
    ax.axis("off")

    diagram_title = title or spec.name
    ax.text(canvas_width / 2, top_y, diagram_title, ha="center", fontsize=16, fontweight="bold", color=COLORS["text"])
    ax.text(
        canvas_width / 2,
        top_y - 0.4,
        "Generated by TraceLens Visualizer (config + AST, CPU)",
        ha="center",
        fontsize=8.5,
        color=COLORS["muted"],
    )

    layout = DiagramLayout()
    stack_top = top_y - 0.85
    y = stack_top
    spine: list[Node] = []

    stack_h = single_line_box_height()
    linear_sublabel = "Linear"
    linear_stack_h = box_height_for_content(linear_sublabel)
    stack_pre = _stack_pre_components(spec)
    stack_tail = _stack_tail_components(spec)
    stack_labels = ["Tokenized text"] + [_spine_display_label(comp, spec) for comp in stack_pre]
    stack_w = max(_box_label_width(ax, label, fontsize=9.0) for label in stack_labels)
    for comp in stack_pre:
        sublabel = _spine_sublabel(comp)
        if sublabel:
            stack_w = max(
                stack_w,
                _box_label_width(ax, _spine_display_label(comp, spec), fontsize=9.0, sublabel=sublabel),
            )
    tail_w = max(
        _box_label_width(
            ax,
            _spine_display_label(comp, spec),
            fontsize=9.0,
            sublabel=_spine_sublabel(comp),
        )
        if _spine_sublabel(comp)
        else _box_label_width(ax, _spine_display_label(comp, spec), fontsize=9.0)
        for comp in stack_tail
    )

    def place(
        label: str,
        color: str,
        *,
        w: float = stack_w,
        h: float = stack_h,
        text_color: str = COLORS["text"],
        node_id: str | None = None,
        sublabel: str | None = None,
    ) -> Node:
        nonlocal y
        node = _make_node(
            node_id or label,
            cx,
            y,
            w,
            h,
            label,
            color,
            text_color=text_color,
            sublabel=sublabel,
        )
        _fit_spine_node_to_label(ax, node)
        layout.add(node)
        _draw_box(ax, node)
        if spine:
            _connect_down(ax, spine[-1], node)
        spine.append(node)
        y = node.bottom - min_vertical_block_gap()
        return node

    place("Tokenized text", COLORS["embed"], node_id="tokens")

    above_block_bottom = spine[-1].bottom if spine else y

    for comp in stack_pre:
        if detailed:
            stack_tree = build_stack_component_tree(
                comp,
                spec.class_registry,
                BasicOpFilter.for_detailed(),
            )
            if is_straight_line_module(stack_tree):
                frame_title = spine_expanded_frame_label(
                    comp,
                    positional_encoding=spec.positional_encoding,
                )
                expanded_top = _spine_expanded_block_top_y(y)
                expanded_bottom, _, expanded_entry = _render_laid_out_computation_graph(
                    layout,
                    ax,
                    stack_tree,
                    cx=cx,
                    top_y=expanded_top,
                    block_w=max(stack_w + 0.6, 3.4),
                    inline_linear_frames=inline_linear_frames,
                    draw_section_frame=False,
                    root_frame_label=frame_title if inline_linear_frames else None,
                    include_input=comp.role != "positional",
                    basic_ops=spec.basic_ops or BasicOpFilter.for_detailed(),
                )
                if spine and expanded_entry is not None:
                    _arrow(
                        ax,
                        spine[-1].cx,
                        spine[-1].bottom,
                        expanded_entry.cx,
                        expanded_entry.top,
                    )
                above_block_bottom = expanded_bottom
                y = expanded_bottom - min_vertical_block_gap()
                continue

        place(
            _spine_display_label(comp, spec),
            _spine_color(comp.role),
            node_id=comp.attr_name,
            sublabel=_spine_sublabel(comp),
            h=_spine_box_height(comp),
        )
        above_block_bottom = spine[-1].bottom

    block_top = _block_top_below_repeat_label(
        ax,
        cx=cx,
        block_w=block_w,
        above_bottom=above_block_bottom,
        repeat_label=repeat_label,
        decoder_label=decoder_label,
    )
    frame_top = _block_frame_top(block_top, repeat_label)
    frame_left = cx - block_w / 2

    if detailed:
        _connect_into_block(
            ax,
            None,
            cx=cx,
            frame_top=frame_top,
            frame_left=frame_left,
            repeat_label=repeat_label,
            decoder_label=decoder_label,
            source_x=cx,
            source_y=above_block_bottom,
        )
    elif spine:
        _connect_into_block(
            ax,
            spine[-1],
            cx=cx,
            frame_top=frame_top,
            frame_left=frame_left,
            repeat_label=repeat_label,
            decoder_label=decoder_label,
        )

    if _ordered_block_components(spec):
        block_content_bottom = _layout_component_block(
            layout,
            ax,
            cx=cx,
            top_y=block_top,
            block_w=block_w,
            spec=spec,
            norm_w=norm_w,
            inner_w=inner_w,
            repeat_label=repeat_label,
        )
    else:
        block_content_bottom = _layout_default_block(
            layout,
            ax,
            cx=cx,
            top_y=block_top,
            block_w=block_w,
            spec=spec,
            norm_w=norm_w,
            inner_w=inner_w,
            repeat_label=repeat_label,
        )

    frame_bottom = _block_frame_bottom(block_content_bottom)
    tail_cursor = frame_bottom - FRAME_PATCH_BOTTOM_OUTSET - min_vertical_block_gap()
    tail_nodes: list[Node] = []
    for index, comp in enumerate(stack_tail):
        tail_node = _make_node(
            comp.attr_name,
            cx,
            tail_cursor,
            tail_w,
            _spine_box_height(comp),
            _spine_display_label(comp, spec),
            _spine_color(comp.role),
            sublabel=_spine_sublabel(comp),
        )
        layout.add(tail_node)
        _fit_spine_node_to_label(ax, tail_node)
        _draw_box(ax, tail_node)
        if index == 0:
            _arrow(
                ax,
                cx,
                frame_bottom - FRAME_PATCH_BOTTOM_OUTSET,
                tail_node.cx,
                tail_node.top,
            )
        else:
            _connect_down(ax, tail_nodes[-1], tail_node)
        tail_nodes.append(tail_node)
        tail_cursor = tail_node.bottom - min_vertical_block_gap()

    diagram_bottom = tail_nodes[-1].bottom if tail_nodes else frame_bottom

    if detailed:
        from visualizer.render_validate import measure_max_detail_section_width

        detail_min_left = DIAGRAM_LEFT_MARGIN + 0.05
        detail_content_w = measure_max_detail_section_width(
            ax,
            spec,
            cx=cx,
            detail_fill=COLORS["detail_fill"],
            min_left=detail_min_left,
        )
        canvas_width, internals_below_fact_sheet = _detail_layout_geometry(
            canvas_width,
            fact_x=fact_x,
            fact_w=fact_w,
            detail_content_width=detail_content_w,
        )
        ax.set_xlim(0, canvas_width)
        fig.set_size_inches(canvas_width, 13)
        fig.canvas.draw()
    else:
        internals_below_fact_sheet = False
    if detailed and internals_below_fact_sheet:
        fact_y = diagram_bottom
    else:
        fact_y = stack_top - fact_h
    _draw_fact_sheet(ax, spec, fact_x=fact_x, fact_y=fact_y, fact_w=fact_w, wrap_width=wrap_width)

    if detailed:
        fact_sheet_bounds = None
        if not internals_below_fact_sheet:
            fact_sheet_bounds = [
                ContentBounds(
                    left=fact_x - 0.08,
                    right=fact_x + fact_w + 0.08,
                    bottom=fact_y - 0.15,
                    top=fact_y + fact_h + 0.15,
                )
            ]
        if internals_below_fact_sheet:
            internals_start_y = fact_y - 0.25
        else:
            internals_start_y = min(diagram_bottom, fact_y) - 0.25
        diagram_bottom = _render_detailed_internals(
            layout,
            ax,
            spec,
            cx=cx,
            start_y=internals_start_y,
            panel_x=fact_x,
            panel_w=fact_w,
            wrap_width=wrap_width,
            inline_linear_frames=inline_linear_frames,
            forbidden_regions=fact_sheet_bounds,
            compact_header=internals_below_fact_sheet,
        )

    min_canvas_width = canvas_width if detailed else None
    canvas_width, canvas_height = _fit_figure_to_content(
        ax,
        fig,
        margin=bottom_margin,
        min_width=min_canvas_width,
    )

    fmt = output_path.suffix.lstrip(".").lower() or "svg"
    fig.savefig(output_path, format=fmt, bbox_inches="tight", pad_inches=0.08, facecolor=COLORS["bg"])
    plt.close(fig)
    if fmt == "svg":
        svg = output_path.read_text(encoding="utf-8")
        output_path.write_text(_finalize_svg_styling(svg), encoding="utf-8")
    return output_path
