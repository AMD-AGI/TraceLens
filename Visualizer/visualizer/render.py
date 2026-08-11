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
from visualizer.blocks import BlockComponent, collect_norm_module_pairs
from visualizer.block_tree import (
    BlockNode,
    _is_composite_block,
    _omit_from_detailed_view,
    _show_single_function_in_diagram,
    build_stack_component_tree,
    collect_method_wrappers,
    collect_parallel_gate_wrappers,
    collect_nested_diagrams,
    is_method_wrapper,
    is_single_function_tree,
    is_straight_line_module,
    spine_expanded_frame_label,
    wrapper_bullet,
    wrapper_bullet_lines,
    wrapper_module_comment,
    wrapper_panel_line,
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
    box_width_for_text_width,
    estimate_block_size_for_node,
    min_vertical_block_gap,
    single_line_box_height,
)
from visualizer.computation_graph import (
    SYNTHETIC_HIDDEN,
    SYNTHETIC_COMBINE,
    SYNTHETIC_INPUT,
    SYNTHETIC_MULTIPLY,
    LayoutPosition,
    _center_positions_horizontally,
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
    r'(<g style="fill: #ffffff" transform="translate\([^"]+\) scale\(([-\d.]+)\s+([-\d.]+)\)">)'
    r"((?:(?!</g>).)*?)"
    r"(</g>)",
    re.DOTALL,
)
WHITE_TEXT_OUTLINE_PX = 3.0


def _stroke_white_text_in_svg(svg: str) -> str:
    """Add a thick black outline to matplotlib's scaled SVG glyph groups."""
    svg = re.sub(
        r'(<g style=")fill: #ffffff; stroke: #000000; stroke-width: 1px; paint-order: stroke fill(")',
        r"\1fill: #ffffff\2",
        svg,
    )

    def _patch_group(match: re.Match[str]) -> str:
        header, sx, sy, body, closing = match.groups()
        if "<use " not in body:
            return match.group(0)
        scale = min(abs(float(sx)), abs(float(sy)))
        stroke_w = WHITE_TEXT_OUTLINE_PX / scale if scale > 0 else WHITE_TEXT_OUTLINE_PX
        use_style = (
            f'style="fill: #ffffff; stroke: #000000; stroke-width: {stroke_w:.4f}; '
            f'paint-order: stroke fill" '
        )

        def _patch_use(use_match: re.Match[str]) -> str:
            tag = use_match.group(0)
            if "style=" in tag:
                return tag
            return tag.replace("<use ", f"<use {use_style}", 1)

        patched_body = re.sub(r"<use ", _patch_use, body)
        return header + patched_body + closing

    return _WHITE_TEXT_GROUP.sub(_patch_group, svg)


def _detail_block_facecolor(block: BlockNode) -> str:
    """Face color for nodes drawn inside a detailed block-internals graph."""
    if block.is_basic:
        return COLORS["basic_op"]
    return ROLE_COLORS.get(block.role, ROLE_COLORS["other"])

PANEL_W = 4.35
PANEL_WRAP_WIDTH = 48
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

DETAIL_TILE_ROUNDING = 0.08
DETAIL_TILE_BOX_PAD = 0.008
DETAIL_FRAME_BOX_PAD = 0.01
DETAIL_FRAME_ROUNDING = 0.12
DETAIL_FRAME_STROKE = 0.035
DETAIL_FRAME_GAP = 0.025
DETAIL_FRAME_PAD_Y_EXTRA = 0.012
INLINE_FRAME_PAD = 0.10
INLINE_FRAME_LABEL_GAP = 0.04
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


@dataclass
class DetailDrawPlan:
    """Pre-rendered box/text descriptors measured before connectors are drawn."""

    input_sublabel: str | None
    node_draws: list[tuple[Node, dict[str, object]]] = field(default_factory=list)
    combine_ops: list[tuple[float, float, str, str | None]] = field(default_factory=list)
    branch_labels: list[tuple[str, float, float, str, str]] = field(default_factory=list)
    label_obstacles: list[_RenderAnchor] = field(default_factory=list)


@dataclass
class DiagramLayout:
    nodes: list[Node] = field(default_factory=list)
    height: float = 13.0

    def add(self, node: Node) -> Node:
        self.nodes.append(node)
        return node


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
        edgecolor=edgecolor or node.facecolor,
        facecolor=node.facecolor,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    title_y = node.top - BLOCK_PAD_Y - TITLE_LINE_H / 2
    if node.sublabel:
        ax.text(
            node.cx,
            title_y,
            node.label,
            ha="center",
            va="center",
            fontsize=node.fontsize,
            color=node.text_color,
            fontweight="bold",
            zorder=6,
        )
        sub_y = title_y - TITLE_LINE_H / 2 - LABEL_LINE_GAP - SUB_LINE_H / 2
        ax.text(
            node.cx,
            sub_y,
            node.sublabel,
            ha="center",
            va="center",
            fontsize=max(6.5, node.fontsize - 1.5),
            color=node.text_color,
            zorder=6,
        )
    else:
        ax.text(
            node.cx,
            title_y,
            node.label,
            ha="center",
            va="center",
            fontsize=node.fontsize,
            color=node.text_color,
            fontweight="bold",
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
) -> None:
    _line(ax, x1, y1, x2, y2, color=color, linewidth=linewidth, linestyle=linestyle)


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
) -> None:
    ax.plot(
        [x1, x2],
        [y1, y2],
        color=color or COLORS["flow"],
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=2,
        solid_capstyle="round",
    )


def _connect_down(ax, source: Node, target: Node, *, gap: float = 0.06) -> None:
    _arrow(ax, source.cx, source.bottom - gap, target.cx, target.top + gap)


def _connect_from_point(ax, x: float, y: float, target: Node, *, gap: float = 0.06) -> None:
    _arrow(ax, x, y - gap, target.cx, target.top + gap)


def _draw_op_circle(ax, x: float, y: float, symbol: str) -> None:
    circle = Circle(
        (x, y),
        MERGE_RADIUS,
        facecolor=COLORS["bg"],
        edgecolor=COLORS["residual"],
        linewidth=1.6,
        zorder=5,
    )
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha="center", va="center", fontsize=11, color=COLORS["residual"], zorder=6)


def _draw_merge(ax, x: float, y: float) -> None:
    _draw_op_circle(ax, x, y, "+")


def _draw_combine_op(
    ax,
    x: float,
    y: float,
    symbol: str,
    *,
    sublabel: str | None = None,
) -> None:
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


def _is_combine_synthetic(synthetic: str | None) -> bool:
    return synthetic in {SYNTHETIC_COMBINE, SYNTHETIC_MULTIPLY}


def _merge_anchor(cx: float, merge_y: float) -> _RenderAnchor:
    pad = MERGE_RADIUS + MERGE_CLEARANCE
    return _RenderAnchor(cx=cx, top=merge_y + pad, bottom=merge_y - pad, left=cx - pad, right=cx + pad)


def _connect_from_merge(ax, merge_x: float, merge_y: float, target: Node, *, gap: float = 0.06) -> None:
    """Connect downward from a residual merge node to the next block."""
    start_y = merge_y - MERGE_RADIUS - MERGE_CLEARANCE
    _arrow(ax, merge_x, start_y, target.cx, target.top + gap)


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
    _line(ax, spine_x, skip_from_y, spine_x, branch_y, color=COLORS["residual"], linestyle="--")
    _line(ax, spine_x, branch_y, branch_x, branch_y, color=COLORS["residual"], linestyle="--")
    _line(ax, branch_x, branch_y, branch_x, merge_y, color=COLORS["residual"], linestyle="--")
    _arrow(
        ax,
        branch_x,
        merge_y,
        merge_left,
        merge_y,
        color=COLORS["residual"],
        linewidth=1.5,
        linestyle="--",
    )

    # Main path: share the same bus and vertical entry at the merge node.
    if abs(module_cx - spine_x) < 0.06:
        points = [(module_cx, module_bottom - 0.02), (module_cx, bus_y), (spine_x, merge_top)]
    else:
        points = [
            (module_cx, module_bottom - 0.02),
            (module_cx, bus_y),
            (spine_x, bus_y),
            (spine_x, merge_top),
        ]
    _draw_path(ax, points, color=COLORS["flow"])

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


def _ffn_label(spec: ArchitectureSpec) -> tuple[str, str | None]:
    if spec.layer_variants:
        labels: list[str] = []
        for variant in spec.layer_variants:
            if variant.ffn_label not in labels:
                labels.append(variant.ffn_label)
        if len(labels) > 1:
            if "MoE" in labels:
                active = spec.num_experts_per_tok or "?"
                label = f"MoE / {spec.ffn_type or 'FFN'}"
                sub = f"{active} active / token; dense prefix"
                if spec.num_shared_experts:
                    sub += f", {spec.num_shared_experts} shared"
                return label, sub
            return " / ".join(labels), None
    if spec.decoder_type == "Sparse MoE":
        experts = spec.num_experts or "?"
        active = spec.num_experts_per_tok or "?"
        label = f"MoE ({experts} experts)"
        sub = f"{active} active / token"
        if spec.num_shared_experts:
            sub += f", {spec.num_shared_experts} shared"
        return label, sub
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


def _wrap_fact_text(text: str, *, width: int = 48) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [text]


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


def _wrapper_info_height(wrappers: list[BlockNode], *, wrap_width: int = PANEL_WRAP_WIDTH) -> float:
    line_height = PANEL_LINE_HEIGHT
    padding_top = PANEL_PAD_TOP
    padding_bottom = PANEL_PAD_BOTTOM
    total_lines = 0
    for node in wrappers:
        total_lines += len(_wrap_fact_text(wrapper_panel_line(node), width=wrap_width))
    return max(0.9, padding_top + total_lines * line_height + padding_bottom)


def _draw_wrapper_info_box(
    ax,
    wrappers: list[BlockNode],
    *,
    box_x: float,
    box_top: float,
    box_w: float = PANEL_W,
    wrap_width: int = PANEL_WRAP_WIDTH,
) -> float:
    """Draw a bulleted info panel for method wrappers omitted from diagrams."""
    if not wrappers:
        return box_top

    box_h = _wrapper_info_height(wrappers, wrap_width=wrap_width)
    box_y = box_top - box_h
    patch = FancyBboxPatch(
        (box_x, box_y),
        box_w,
        box_h,
        boxstyle="round,pad=0.01,rounding_size=0.08",
        linewidth=1.0,
        edgecolor=COLORS["fact_border"],
        facecolor=COLORS["fact_bg"],
        zorder=1,
    )
    ax.add_patch(patch)
    _draw_panel_title(ax, box_x + PANEL_PAD_X, box_top - PANEL_TITLE_Y, "Simply wrapped modules")

    cursor_y = box_top - PANEL_PAD_TOP
    for node in wrappers:
        rows = _wrap_fact_text(wrapper_panel_line(node), width=wrap_width)
        for index, row in enumerate(rows):
            prefix = "• " if index == 0 else "  "
            ax.text(
                box_x + PANEL_PAD_X,
                cursor_y,
                f"{prefix}{row}",
                fontsize=PANEL_BODY_FONT,
                color=PANEL_BODY_COLOR,
                va="top",
                zorder=5,
            )
            cursor_y -= PANEL_LINE_HEIGHT
    return box_y


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
            label=f"Final {norm_type}",
            forward_order=0,
        ),
        BlockComponent(
            attr_name="lm_head",
            class_name="Linear",
            role="head",
            label="LM head / output projection",
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


def _spine_display_label(component: BlockComponent, spec: ArchitectureSpec) -> str:
    if component.role == "positional":
        return f"Positional ({spec.positional_encoding})"
    return component.label


def _spine_sublabel(component: BlockComponent) -> str | None:
    if displays_as_linear(component.attr_name, component.class_name):
        return "Linear"
    return None


def _spine_box_height(component: BlockComponent) -> float:
    sublabel = _spine_sublabel(component)
    return box_height_for_content(sublabel) if sublabel else single_line_box_height()


def _module_input_labels(spec: ArchitectureSpec) -> dict[str, str]:
    """Map compute module attr names to the norm label that feeds them in the outer block."""
    labels: dict[str, str] = {}
    for norm, comp in _collect_sublayer_pairs(_ordered_block_components(spec)):
        labels[comp.attr_name] = norm.label
    return labels


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
    _draw_box(ax, norm_node)
    if spine_y is not None:
        _connect_from_merge(ax, cx, spine_y, norm_node)
    elif entry_from_y is not None:
        _arrow(ax, cx, entry_from_y - 0.04, norm_node.cx, norm_node.top + 0.04)

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
    sublabel = comp.class_name if comp.class_name != comp.label else None
    if comp.details:
        sublabel = (sublabel + "\n" if sublabel else "") + comp.details[0][:28]
    return sublabel


def _block_content_widths(ax, spec: ArchitectureSpec) -> tuple[float, float]:
    """Return (norm_w, inner_w) using the same horizontal padding as RMSNorm boxes."""
    if _ordered_block_components(spec):
        pairs = _collect_sublayer_pairs(_ordered_block_components(spec))
        norm_w = 0.0
        inner_w = 0.0
        for norm_comp, comp in pairs:
            norm_w = max(norm_w, _box_label_width(ax, norm_comp.label, fontsize=8))
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
        start_x, start_y = source.cx, source.bottom - 0.04
    elif source_x is not None and source_y is not None:
        start_x, start_y = source_x, source_y - 0.04
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
            norm_label=norm_comp.label,
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


def _orthogonal_path(
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    gap: float = 0.04,
    bus_near: str = "target",
    bus_y: float | None = None,
) -> list[tuple[float, float]]:
    """Build a Manhattan path from source bottom to target top."""
    x1, y1 = source.cx, source.bottom - gap
    x2, y2 = target.cx, target.top + gap

    if abs(x1 - x2) < 0.06:
        if bus_y is None:
            straight = [(x1, y1), (x2, y2)]
            if not _path_hits_obstacles(straight, obstacles):
                return straight
            for offset in (0.12, -0.12, 0.24, -0.24):
                detour = [(x1, y1), (x1 + offset, y1), (x1 + offset, y2), (x2, y2)]
                if not _path_hits_obstacles(detour, obstacles):
                    return detour
            return straight
        aligned = [(x1, y1), (x1, bus_y), (x2, y2)]
        if not _path_hits_obstacles(aligned, obstacles):
            return aligned
        for _ in range(6):
            if not _path_hits_obstacles(aligned, obstacles):
                return aligned
            bus_y -= 0.08
            aligned = [(x1, y1), (x1, bus_y), (x2, y2)]
        return aligned

    if bus_y is None:
        channel = max(y1 - y2, gap * 4)
        if bus_near == "source":
            bus_y = y1 - min(0.10, channel * 0.35)
        else:
            bus_y = y2 + min(0.10, channel * 0.25)
        bus_y = min(y1 - 0.02, max(y2 + 0.02, bus_y))

    for _ in range(6):
        points = [(x1, y1), (x1, bus_y), (x2, bus_y), (x2, y2)]
        if not _path_hits_obstacles(points, obstacles):
            return points
        bus_y -= 0.08
    return [(x1, y1), (x1, bus_y), (x2, bus_y), (x2, y2)]


def _compute_shared_target_bus_y(
    sources: list[_RenderAnchor],
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    gap: float = 0.04,
) -> float:
    """Pick one horizontal channel so converging links enter the target vertically."""
    y2 = target.top + gap
    y1_min = min(source.bottom - gap for source in sources)
    channel = max(y1_min - y2, gap * 4)
    bus_y = y2 + min(0.10, channel * 0.25)
    bus_y = min(y1_min - 0.02, max(y2 + 0.02, bus_y))
    for _ in range(8):
        if all(
            not _path_hits_obstacles(
                [
                    (source.cx, source.bottom - gap),
                    (source.cx, bus_y),
                    (target.cx, bus_y),
                    (target.cx, y2),
                ],
                obstacles,
            )
            for source in sources
        ):
            return bus_y
        bus_y -= 0.08
    return bus_y


def _compute_shared_source_bus_y(
    source: _RenderAnchor,
    targets: list[_RenderAnchor],
    obstacles: list[_RenderAnchor],
    *,
    gap: float = 0.04,
) -> float:
    """Pick one horizontal channel so fan-out links leave the source vertically aligned."""
    y1 = source.bottom - gap
    y2_max = max(target.top + gap for target in targets)
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
                    (target.cx, target.top + gap),
                ],
                obstacles,
            )
            for target in targets
        ):
            return bus_y
        bus_y -= 0.08
    return bus_y


def _draw_path(
    ax,
    points: list[tuple[float, float]],
    *,
    color: str | None = None,
    linewidth: float = 1.5,
    linestyle: str = "solid",
) -> None:
    if len(points) < 2:
        return
    stroke = color or COLORS["flow"]
    if linestyle != "solid":
        for index in range(len(points) - 1):
            x1, y1 = points[index]
            x2, y2 = points[index + 1]
            _line(ax, x1, y1, x2, y2, color=stroke, linewidth=linewidth, linestyle=linestyle)
        return

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.plot(
        xs,
        ys,
        color=stroke,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=2,
        solid_capstyle="round",
    )
    last_x, last_y = points[-2]
    end_x, end_y = points[-1]
    _arrow(ax, last_x, last_y, end_x, end_y, color=stroke, linewidth=linewidth, linestyle=linestyle)


def _inline_dashed_port_connector_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    *,
    gap: float = 0.04,
    bus_y: float | None = None,
) -> list[tuple[float, float]]:
    """Route a parallel branch into the top center of an inline-port node."""
    entry_x = target.cx
    entry_y = target.top + gap
    y_start = source.bottom - gap
    bus_y = entry_y if bus_y is None else bus_y
    return [(source.cx, y_start), (source.cx, bus_y), (entry_x, bus_y), (entry_x, entry_y)]


def _labeled_merge_connector_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    entry_x: float,
    *,
    gap: float = 0.04,
    bus_y: float | None = None,
) -> list[tuple[float, float]]:
    """Route a fan-in branch into a labeled port on the merge node top edge."""
    y_start = source.bottom - gap
    entry_y = target.top + gap
    if bus_y is None:
        bus_y = (y_start + entry_y) / 2
    if abs(source.cx - entry_x) < 0.06:
        return [(source.cx, y_start), (source.cx, entry_y)]
    return [(source.cx, y_start), (source.cx, bus_y), (entry_x, bus_y), (entry_x, entry_y)]


def _top_entry_combine_connector_points(
    source: _RenderAnchor,
    target_cx: float,
    target_cy: float,
    *,
    gap: float = 0.04,
) -> list[tuple[float, float]]:
    """Route the main sequential path into the top of a combine operator."""
    entry_x = target_cx
    entry_y = target_cy + MERGE_RADIUS + gap
    y_start = source.bottom - gap
    if abs(source.cx - target_cx) < 0.08:
        return [(source.cx, y_start), (source.cx, entry_y)]
    bus_y = (y_start + entry_y) / 2
    return [(source.cx, y_start), (source.cx, bus_y), (entry_x, bus_y), (entry_x, entry_y)]


def _side_entry_combine_connector_points(
    source: _RenderAnchor,
    target_cx: float,
    target_cy: float,
    *,
    gap: float = 0.04,
) -> list[tuple[float, float]]:
    """Route a parallel gate branch into the side of a combine (×) node, like a residual merge."""
    y_start = source.bottom - gap
    entry_y = target_cy
    if source.cx >= target_cx:
        entry_x = target_cx + MERGE_RADIUS
    else:
        entry_x = target_cx - MERGE_RADIUS
    return [(source.cx, y_start), (source.cx, entry_y), (entry_x, entry_y)]


def _draw_graph_connector(
    ax,
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    color: str | None = None,
    linestyle: str = "solid",
    inline_port: bool = False,
    floating_port: bool = False,
    combine_side_entry: bool = False,
    combine_top_entry: bool = False,
    combine_center_y: float | None = None,
    bus_near: str = "target",
    bus_y: float | None = None,
) -> None:
    """Draw an orthogonal connector that routes around node boxes when needed."""
    gap = 0.04
    if combine_side_entry and combine_center_y is not None:
        points = _side_entry_combine_connector_points(
            source,
            target.cx,
            combine_center_y,
            gap=gap,
        )
    elif combine_top_entry and combine_center_y is not None:
        points = _top_entry_combine_connector_points(
            source,
            target.cx,
            combine_center_y,
            gap=gap,
        )
    elif inline_port and linestyle == "dashed":
        points = _inline_dashed_port_connector_points(source, target, gap=gap, bus_y=bus_y)
    elif floating_port:
        points = _orthogonal_path(source, target, obstacles, bus_near=bus_near, bus_y=bus_y)
    elif linestyle == "dashed" and abs(source.cx - target.cx) < 0.08:
        y_start = source.bottom - gap
        y_end = target.top + gap
        points = [(source.cx, y_start), (source.cx, y_end)]
    else:
        points = _orthogonal_path(source, target, obstacles, bus_near=bus_near, bus_y=bus_y)
    connector_color = color or (COLORS["residual"] if linestyle == "dashed" else None)
    _draw_path(ax, points, color=connector_color, linestyle=linestyle)


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


def _graph_method_wrapper_attrs(
    tree: BlockNode,
    *,
    prefix_steps: list[BlockNode] | None = None,
) -> set[str]:
    graph = build_computation_graph(tree, prefix_steps=prefix_steps)
    return {
        spec.block.attr_name
        for spec in graph.nodes
        if spec.block is not None and is_method_wrapper(spec.block)
    }


def _format_input_source_sublabel(source: str | None) -> str | None:
    """Format a nested block input source on two lines when it includes ' in '."""
    if not source:
        return None
    if " in " in source:
        head, tail = source.split(" in ", 1)
        return f"← {head}\nin {tail}"
    return f"← {source}"


def _resize_input_nodes(
    positions: list[LayoutPosition],
    input_sublabel: str | None,
) -> None:
    """Grow the synthetic input tile when it carries a wrapped source sublabel."""
    if not input_sublabel:
        return
    from visualizer.sizing import estimate_block_size

    for pos in positions:
        if pos.spec.synthetic != SYNTHETIC_INPUT:
            continue
        width, height = estimate_block_size(
            pos.spec.label,
            input_sublabel,
            fontsize=7.2,
            sub_fontsize=6.5,
        )
        pos.width = max(pos.width, width)
        pos.height = max(pos.height, height)


def _spine_expanded_block_top_y(cursor_y: float) -> float:
    """Place an expanded spine section below cursor_y, clearing room for its frame label."""
    return cursor_y - SPINE_EXPANDED_BLOCK_TOP_RESERVE


def _render_inline_linear_frames(
    ax,
    graph,
    positions: list,
    *,
    enabled: bool,
) -> None:
    """Draw dotted frames around steps inlined from straight-line composite sub-blocks."""
    if not enabled or not graph.inline_frames:
        return

    for frame in graph.inline_frames:
        if not frame.node_indices:
            continue
        frame_positions = [positions[index] for index in frame.node_indices if index < len(positions)]
        if not frame_positions:
            continue

        min_left = min(pos.cx - pos.width / 2 for pos in frame_positions)
        max_right = max(pos.cx + pos.width / 2 for pos in frame_positions)
        min_bottom = min(pos.top_y - pos.height for pos in frame_positions)
        max_top = max(pos.top_y for pos in frame_positions)
        pad = INLINE_FRAME_PAD
        frame_left = min_left - pad
        frame_bottom = min_bottom - pad
        frame_w = max_right - min_left + 2 * pad
        frame_h = max_top - min_bottom + 2 * pad

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
        ax.text(
            frame_left + 0.02,
            max_top + pad + INLINE_FRAME_LABEL_GAP,
            frame.label,
            ha="left",
            va="bottom",
            fontsize=6.4,
            color=COLORS["muted"],
        )
        if frame.sublabel:
            ax.text(
                frame_left + 0.02,
                max_top + pad + INLINE_FRAME_LABEL_GAP - 0.11,
                frame.sublabel,
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
            )
            plan.node_draws.append((input_leaf, {}))
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
            )
            plan.node_draws.append(
                (
                    hidden_leaf,
                    {"edgecolor": COLORS["residual"], "linestyle": "dashed"},
                )
            )
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
                sublabel=attr,
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
        sublabel = spec.sublabel or block_sublabel(block)

        if spec.port_label:
            if spec.port_style == "inline":
                display_label = spec.port_label
                if block is not None:
                    sublabel = block.attr_name
            else:
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
        )
        plan.node_draws.append((leaf, {}))

    return plan


def _anchors_from_detail_plan(
    positions: list[LayoutPosition],
    plan: DetailDrawPlan,
) -> dict[int, _RenderAnchor]:
    """Build connector anchors from a validated draw plan."""
    anchors: dict[int, _RenderAnchor] = {}
    draw_index = 0
    for index, pos in enumerate(positions):
        spec = pos.spec
        if _is_combine_synthetic(spec.synthetic):
            center_y = pos.top_y - pos.height / 2
            anchors[index] = _merge_anchor(pos.cx, center_y)
            continue
        if draw_index < len(plan.node_draws):
            anchors[index] = _anchor_from_node(plan.node_draws[draw_index][0])
            draw_index += 1
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
) -> tuple[float, float, _RenderAnchor | None]:
    """Lay out a computation graph with graph-layout and draw it."""
    from visualizer.render_validate import finalize_detail_layout

    graph = build_computation_graph(root, prefix_steps=prefix_steps, include_input=include_input)
    if root_frame_label:
        add_root_pipeline_frame(graph, root, label=root_frame_label)
    est_h = _estimate_graph_height(graph)
    positions, links = layout_computation_graph(
        graph,
        cx=cx,
        top_y=top_y,
        block_w=block_w,
        block_h=est_h,
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
    )
    anchors = _anchors_from_detail_plan(positions, plan)

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
    _render_inline_linear_frames(ax, graph, positions, enabled=inline_linear_frames)

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

    target_bus: dict[int, float] = {}
    for tgt, link_group in incoming.items():
        if len(link_group) < 2:
            continue
        if _is_combine_synthetic(positions[tgt].spec.synthetic) and any(
            (src, tgt) in graph.dashed_links for src, _ in link_group
        ):
            continue
        target_anchor = anchors.get(tgt)
        source_anchors = [anchors[src] for src, _ in link_group if src in anchors]
        if target_anchor is None or len(source_anchors) < 2:
            continue
        involved = {tgt, *(src for src, _ in link_group)}
        route_obstacles = [
            anchor for node_index, anchor in anchors.items() if node_index not in involved
        ] + label_obstacles
        target_bus[tgt] = _compute_shared_target_bus_y(
            source_anchors,
            target_anchor,
            route_obstacles,
        )

    source_bus: dict[int, float] = {}
    for src, link_group in outgoing.items():
        if len(link_group) < 2:
            continue
        source_anchor = anchors.get(src)
        target_anchors = [anchors[tgt] for _, tgt in link_group if tgt in anchors]
        if source_anchor is None or len(target_anchors) < 2:
            continue
        involved = {src, *(tgt for _, tgt in link_group)}
        route_obstacles = [
            anchor for node_index, anchor in anchors.items() if node_index not in involved
        ] + label_obstacles
        source_bus[src] = _compute_shared_source_bus_y(
            source_anchor,
            target_anchors,
            route_obstacles,
        )

    merge_entry_x: dict[tuple[int, int], float] = {}
    merge_link_bus: dict[tuple[int, int], float] = {}
    for tgt, link_group in incoming.items():
        labeled_links = [
            (src, tgt) for src, tgt in link_group if (src, tgt) in graph.link_port_labels
        ]
        if not labeled_links:
            continue
        target_pos = positions[tgt]
        target_anchor = anchors.get(tgt)
        if target_anchor is None:
            continue
        sorted_links = sorted(labeled_links, key=lambda link: positions[link[0]].cx)
        inner_w = max(target_pos.width * 0.72, 0.35 * len(sorted_links))
        for index, link in enumerate(sorted_links):
            merge_entry_x[link] = target_pos.cx - inner_w / 2 + (index + 1) * inner_w / (len(sorted_links) + 1)
        base_bus = target_bus.get(tgt)
        if base_bus is not None and len(sorted_links) >= 2:
            for index, link in enumerate(sorted_links):
                merge_link_bus[link] = base_bus - index * 0.07

    merge_link_labels: list[tuple[str, float, float]] = []

    for src, tgt in links:
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue
        link_key = (src, tgt)
        port_label = graph.link_port_labels.get(link_key)
        if port_label and link_key in merge_entry_x:
            entry_x = merge_entry_x[link_key]
            bus_y = merge_link_bus.get(link_key)
            points = _labeled_merge_connector_points(
                source,
                target,
                entry_x,
                bus_y=bus_y,
            )
            linestyle = "dashed" if link_key in graph.dashed_links else "solid"
            connector_color = COLORS["residual"] if linestyle == "dashed" else None
            _draw_path(ax, points, color=connector_color, linestyle=linestyle)
            merge_link_labels.append((port_label, entry_x, target.top + 0.05))
            continue
        linestyle = "dashed" if link_key in graph.dashed_links else "solid"
        route_obstacles = [
            anchor for node_index, anchor in anchors.items() if node_index not in {src, tgt}
        ] + label_obstacles
        target_spec = positions[tgt].spec
        inline_port = target_spec.port_style == "inline"
        floating_port = target_spec.port_style == "floating"
        combine_side_entry = (
            _is_combine_synthetic(target_spec.synthetic) and (src, tgt) in graph.side_entry_links
        )
        has_dashed_incoming = any(
            (side_src, tgt) in graph.dashed_links for side_src, _ in incoming.get(tgt, [])
        )
        combine_top_entry = (
            _is_combine_synthetic(target_spec.synthetic)
            and linestyle == "solid"
            and has_dashed_incoming
            and not combine_side_entry
        )
        combine_center_y = (
            (target.top + target.bottom) / 2 if _is_combine_synthetic(target_spec.synthetic) else None
        )
        bus_near = "source" if input_index is not None and src == input_index else "target"
        bus_y: float | None = None
        if tgt in target_bus and not combine_side_entry and not combine_top_entry:
            bus_y = target_bus[tgt]
            bus_near = "target"
        elif src in source_bus:
            bus_y = source_bus[src]
            bus_near = "source"
        _draw_graph_connector(
            ax,
            source,
            target,
            route_obstacles,
            linestyle=linestyle,
            inline_port=inline_port,
            floating_port=floating_port,
            combine_side_entry=combine_side_entry,
            combine_top_entry=combine_top_entry,
            combine_center_y=combine_center_y,
            bus_near=bus_near,
            bus_y=bus_y,
        )

    for leaf, draw_kwargs in plan.node_draws:
        layout.add(leaf)
        _draw_box(ax, leaf, **draw_kwargs)

    for op_x, op_y, symbol, op_sublabel in plan.combine_ops:
        _draw_combine_op(ax, op_x, op_y, symbol, sublabel=op_sublabel)

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
    )
    return bottom, frame_right


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
) -> float:
    """Render one titled block diagram. Returns y below the diagram."""
    title_y = cursor
    frame_top_pad = _detail_frame_edge_pad()
    diagram_top = cursor - SECTION_TITLE_HEIGHT - SECTION_TITLE_GAP - frame_top_pad
    ax.text(
        cx,
        title_y,
        title,
        ha="center",
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
    )


def _render_detailed_internals(
    layout: DiagramLayout,
    ax,
    spec: ArchitectureSpec,
    *,
    cx: float,
    start_y: float,
    block_w: float,
    panel_x: float | None = None,
    panel_w: float = PANEL_W,
    wrap_width: int = PANEL_WRAP_WIDTH,
    inline_linear_frames: bool = True,
) -> float:
    """Render recursive internal block diagrams below the main model."""
    panel_x = panel_x if panel_x is not None else cx + block_w / 2 + 0.55

    cursor = start_y - 0.45
    ax.text(
        cx,
        cursor,
        "Block internals",
        ha="center",
        va="center",
        fontsize=PANEL_TITLE_FONT,
        color=PANEL_TITLE_COLOR,
        fontweight="bold",
    )
    cursor -= 0.35
    section_top = cursor

    if not spec.detailed_block_trees and not spec.detailed_wrapped_modules:
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
    wrappers: list[BlockNode] = []
    seen: set[str] = set()
    for node in spec.detailed_wrapped_modules:
        if _omit_from_detailed_view(node):
            continue
        if node.attr_name in seen:
            continue
        seen.add(node.attr_name)
        wrappers.append(node)

    for title, tree in spec.detailed_block_trees:
        input_sublabel = _format_input_source_sublabel(tree.input_source)
        cursor = _render_diagram_section(
            layout,
            ax,
            cx=cx,
            cursor=cursor,
            block_w=block_w,
            title=title,
            tree=tree,
            input_sublabel=input_sublabel,
            inline_linear_frames=inline_linear_frames,
        )
        bottom = cursor
        for sub_title, sub_tree in collect_nested_diagrams(tree):
            if is_single_function_tree(sub_tree):
                if _omit_from_detailed_view(sub_tree):
                    continue
                if _show_single_function_in_diagram(sub_tree):
                    nested_sublabel = _format_input_source_sublabel(sub_tree.input_source)
                    cursor = _render_diagram_section(
                        layout,
                        ax,
                        cx=cx,
                        cursor=cursor,
                        block_w=block_w,
                        title=sub_title,
                        tree=sub_tree,
                        input_sublabel=nested_sublabel,
                        inline_linear_frames=inline_linear_frames,
                    )
                    bottom = cursor
                    continue
                if sub_tree.attr_name not in seen:
                    seen.add(sub_tree.attr_name)
                    wrappers.append(sub_tree)
                continue
            nested_sublabel = _format_input_source_sublabel(sub_tree.input_source)
            cursor = _render_diagram_section(
                layout,
                ax,
                cx=cx,
                cursor=cursor,
                block_w=block_w,
                title=sub_title,
                tree=sub_tree,
                input_sublabel=nested_sublabel,
                inline_linear_frames=inline_linear_frames,
            )
            bottom = cursor

    rendered_in_graph: set[str] = set()
    for title, tree in spec.detailed_block_trees:
        rendered_in_graph.update(_graph_method_wrapper_attrs(tree))
        for _, sub_tree in collect_nested_diagrams(tree):
            if is_single_function_tree(sub_tree):
                continue
            rendered_in_graph.update(_graph_method_wrapper_attrs(sub_tree))

    wrappers = [
        node
        for node in wrappers
        if node.attr_name not in rendered_in_graph
        and not _omit_from_detailed_view(node)
        and not (is_single_function_tree(node) and _show_single_function_in_diagram(node))
    ]

    for _, tree in spec.detailed_block_trees:
        for node in collect_method_wrappers(tree):
            if node.attr_name in seen or node.attr_name in rendered_in_graph:
                continue
            if node.attr_name in tree.side_inputs and is_method_wrapper(node):
                continue
            seen.add(node.attr_name)
            wrappers.append(node)
        for node in collect_parallel_gate_wrappers(tree):
            if node.attr_name in seen:
                continue
            seen.add(node.attr_name)
            wrappers.append(node)

    if wrappers:
        _draw_wrapper_info_box(
            ax,
            wrappers,
            box_x=panel_x,
            box_top=section_top,
            box_w=panel_w,
            wrap_width=wrap_width,
        )

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
    fact_gap = 0.55
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
    canvas_width = max(
        canvas_width,
        cx + block_w / 2 + fact_gap + fact_w + DIAGRAM_RIGHT_MARGIN,
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
        _box_label_width(ax, comp.label, fontsize=9.0, sublabel=_spine_sublabel(comp))
        if _spine_sublabel(comp)
        else _box_label_width(ax, comp.label, fontsize=9.0)
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
                )
                if spine and expanded_entry is not None:
                    _arrow(
                        ax,
                        spine[-1].cx,
                        spine[-1].bottom - 0.06,
                        expanded_entry.cx,
                        expanded_entry.top + 0.06,
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
            comp.label,
            _spine_color(comp.role),
            sublabel=_spine_sublabel(comp),
        )
        layout.add(tail_node)
        _draw_box(ax, tail_node)
        if index == 0:
            _arrow(
                ax,
                cx,
                frame_bottom - FRAME_PATCH_BOTTOM_OUTSET - 0.04,
                tail_node.cx,
                tail_node.top + 0.04,
            )
        else:
            _connect_down(ax, tail_nodes[-1], tail_node)
        tail_nodes.append(tail_node)
        tail_cursor = tail_node.bottom - min_vertical_block_gap()

    diagram_bottom = tail_nodes[-1].bottom if tail_nodes else frame_bottom

    fact_x = cx + block_w / 2 + fact_gap
    fact_y = stack_top - fact_h
    _draw_fact_sheet(ax, spec, fact_x=fact_x, fact_y=fact_y, fact_w=fact_w, wrap_width=wrap_width)

    if detailed:
        diagram_bottom = _render_detailed_internals(
            layout,
            ax,
            spec,
            cx=cx,
            start_y=diagram_bottom - 0.25,
            block_w=block_w,
            panel_x=fact_x,
            panel_w=fact_w,
            wrap_width=wrap_width,
            inline_linear_frames=inline_linear_frames,
        )

    y_min = min(diagram_bottom, fact_y) - bottom_margin
    y_max = top_y + 0.35
    ax.set_ylim(y_min, y_max)

    canvas_height = y_max - y_min
    fig.set_size_inches(canvas_width, canvas_height)

    fmt = output_path.suffix.lstrip(".").lower() or "svg"
    fig.savefig(output_path, format=fmt, bbox_inches="tight", pad_inches=0.08, facecolor=COLORS["bg"])
    plt.close(fig)
    if fmt == "svg":
        svg = output_path.read_text(encoding="utf-8")
        output_path.write_text(_stroke_white_text_in_svg(svg), encoding="utf-8")
    return output_path
