"""Render Raschka-style LLM architecture block diagrams (CPU-only)."""

from __future__ import annotations

import heapq
import itertools
import re
import textwrap
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

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
    expand_block_tree_inplace,
    is_method_wrapper,
    is_simple_modeled_tile,
    is_single_function_tree,
    is_straight_line_module,
    prepare_diagram_section_trees,
    spine_expanded_frame_label,
    subgraph_warrants_export,
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
    INLINE_FRAME_CAPTION_BAND,
    INLINE_FRAME_LABEL_GAP,
    INLINE_FRAME_LABEL_LINE_H,
    INLINE_FRAME_PAD,
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
    SYNTHETIC_INPUT,
    SYNTHETIC_OUTPUT,
    SYNTHETIC_TENSOR,
    LayoutPosition,
    _estimate_graph_height,
    _is_multiply_label,
    _is_summation_label,
    _node_content_left,
    _node_content_right,
    add_forward_output,
    add_root_pipeline_frame,
    build_computation_graph,
    layout_computation_graph,
)
from visualizer.extract import ArchitectureSpec, architecture_section_trees

COLORS = {
    "bg": "#e1e1e1",
    "text": "#1a1a1a",
    "muted": "#555555",
    "embed": "#d9e8f5",
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

_WHITE_TEXT_GROUP = re.compile(
    r'(<g style="fill: (#fff(?:fff)?)" transform="translate\([^"]+\) scale\(([-\d.]+)\s+([-\d.]+)\)">)'
    r"((?:(?!</g>).)*?)"
    r"(</g>)",
    re.DOTALL,
)
WHITE_TEXT_OUTLINE_PX = 2.0
_BASIC_OP_EDGE = "#000000"
_INPUT_NODE_EDGE = "#000000"
# Fills too pale to read as their own border.
_PALE_FILLS = frozenset({COLORS["basic_op"], COLORS["head"]})


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
    """Pick a visible border; pale tiles use black instead of matching their own fill."""
    if node.facecolor in _PALE_FILLS:
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
    return svg


def _detail_block_facecolor(block: BlockNode) -> str:
    """Apply the same semantic palette to every detailed diagram tile."""
    from visualizer.model_graph import OperationKind, classify_operation

    operation = classify_operation(block, label=block.label)
    if operation == OperationKind.GPU_KERNEL:
        return COLORS["moe"]
    if operation == OperationKind.COMPOSITE and block.children:
        return COLORS["attention"]
    return COLORS["basic_op"]


def _detail_tile_text_color(facecolor: str) -> str:
    """Gray tiles use dark text; expanded blocks and kernels use white."""
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

MERGE_OUTPUT_GAP = 0.06
RESIDUAL_BRANCH_LIFT = 0.07
FLOW_CONNECTOR_ZORDER = 2
DETAIL_CONNECTOR_ZORDER = 5.5
INLINE_FRAME_CAPTION_ZORDER = DETAIL_CONNECTOR_ZORDER + 0.1
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
TOP_ENTRY_PORT_GAP = PARALLEL_CONNECTOR_CHANNEL_GAP
# What each kind of connector fault costs the reader, relative to a single crossing.
CONNECTOR_OVERLAP_COST = 4
CONNECTOR_THROUGH_TILE_COST = 3
CONNECTOR_ENTRY_FAULT_COST = 2
TOP_ENTRY_PORT_MAX_CENTER_BAND_FRACTION = 0.45
SAME_COLUMN_BYPASS_CORRIDOR = CONNECTOR_EXIT_STUB + PARALLEL_CONNECTOR_CHANNEL_GAP
FANOUT_SHORT_CHANNEL_MAX = 0.80
FANOUT_SHORT_TEE_FRACTION = 0.5


def _remove_vertical_backtracks(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Drop interior vertices that reverse vertical direction on one column."""
    cleaned = list(points)
    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        for index in range(1, len(cleaned) - 1):
            x0, y0 = cleaned[index - 1]
            x1, y1 = cleaned[index]
            x2, y2 = cleaned[index + 1]
            if abs(x0 - x1) > PARALLEL_CONNECTOR_COORD_EPS or abs(x1 - x2) > PARALLEL_CONNECTOR_COORD_EPS:
                continue
            if (y1 - y0) * (y2 - y1) < 0:
                # Keep a full exit-stub dip before routing back up to a merge bus.
                if (
                    abs(y0 - y1) >= CONNECTOR_EXIT_STUB - PARALLEL_CONNECTOR_COORD_EPS
                    and y1 < min(y0, y2) - PARALLEL_CONNECTOR_COORD_EPS / 2
                ):
                    continue
                del cleaned[index]
                changed = True
                break
    return _ensure_orthogonal_connector_path(cleaned)


def _flatten_upward_connector_steps(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Remove upward vertical jogs by keeping merge corridors monotonically descending."""
    if len(points) < 3:
        return points
    terminal = points[-1]
    cleaned = list(points[:-1])
    for index in range(len(cleaned) - 1):
        _x1, y1 = cleaned[index]
        for later in range(index + 1, len(cleaned)):
            x2, y2 = cleaned[later]
            if y2 <= y1 + PARALLEL_CONNECTOR_COORD_EPS:
                break
            cleaned[later] = (x2, y1)
    return _ensure_orthogonal_connector_path([*cleaned, terminal])


def _restore_target_top_entry_drop(
    points: list[tuple[float, float]],
    *,
    target: _RenderAnchor,
) -> list[tuple[float, float]]:
    """Ensure a spread/bus path ends with a short vertical into the target top."""
    if len(points) < 2:
        return points
    entry_y = _connector_target_top_entry_y(target)
    x_last, y_last = points[-1]
    if abs(y_last - entry_y) <= 1e-9:
        return points
    if len(points) >= 2:
        x_prev, y_prev = points[-2]
        if (
            abs(y_prev - y_last) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(x_prev - x_last) > PARALLEL_CONNECTOR_COORD_EPS
        ):
            completed = [*points, (x_last, entry_y)]
            if abs(completed[-1][1] - completed[-2][1]) > 1e-12:
                return completed
            return points
    if y_last > entry_y + PARALLEL_CONNECTOR_COORD_EPS:
        completed = [*points, (x_last, entry_y)]
        if abs(completed[-1][1] - completed[-2][1]) > 1e-12:
            return completed
    return points


def _collapse_collinear_connector_segments(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Drop interior vertices that sit on the same horizontal run."""
    if len(points) < 3:
        return points
    collapsed = [points[0]]
    for index in range(1, len(points) - 1):
        x_prev, y_prev = collapsed[-1]
        x_mid, y_mid = points[index]
        x_next, y_next = points[index + 1]
        if (
            abs(y_prev - y_mid) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(y_mid - y_next) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(y_prev - y_next) <= PARALLEL_CONNECTOR_COORD_EPS
        ):
            continue
        collapsed.append((x_mid, y_mid))
    collapsed.append(points[-1])
    return _ensure_orthogonal_connector_path(collapsed)


def _horizontal_bus_y_clears_inline_frames(
    bus_y: float,
    graph,
    positions: list,
    *,
    src: int,
    tgt: int,
    x_left: float,
    x_right: float,
) -> bool:
    """True when a horizontal corridor at bus_y avoids every crossed inline frame."""
    for frame in graph.inline_frames:
        members = set(frame.node_indices)
        if tgt in members and src not in members:
            continue
        if src in members and tgt in members:
            continue
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        if x_right < bounds.left - CONNECTOR_OBSTACLE_MARGIN or x_left > bounds.right + CONNECTOR_OBSTACLE_MARGIN:
            continue
        above_frame = max(
            bounds.top + CONNECTOR_OBSTACLE_MARGIN,
            _inline_frame_caption_band_top(frame, bounds) + CONNECTOR_OBSTACLE_MARGIN,
        )
        if bus_y >= above_frame:
            continue
        if bus_y <= bounds.bottom - CONNECTOR_OBSTACLE_MARGIN:
            continue
        return False
    return True


def _path_has_horizontal_backtrack(points: list[tuple[float, float]]) -> bool:
    """True when a horizontal run reverses direction before continuing."""
    for index in range(len(points) - 2):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        x3, y3 = points[index + 2]
        if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS or abs(y2 - y3) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if abs(x2 - x1) <= PARALLEL_CONNECTOR_COORD_EPS or abs(x3 - x2) <= PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if (x2 - x1) * (x3 - x2) < 0:
            return True
    return False


def _prefer_non_backtracking_connector_path(
    original: list[tuple[float, float]],
    candidate: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
) -> list[tuple[float, float]]:
    """Keep a clear route when a rewrite would introduce a horizontal backtrack."""
    del source, target, obstacles
    if candidate == original:
        return candidate
    if not _path_has_horizontal_backtrack(candidate):
        return candidate
    if not _path_has_horizontal_backtrack(original):
        return original
    return candidate


def _repair_horizontal_backtracking_paths(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    label_obstacles: list[_RenderAnchor],
    positions: list,
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_entry_x: dict[tuple[int, int], float],
    merge_link_bus: dict[tuple[int, int], float],
    input_index: int | None,
    inline_bypass_bus_x: dict[tuple[int, int], float] | None,
    initial_link_paths: dict[tuple[int, int], list[tuple[float, float]]] | None = None,
) -> None:
    """Replace connector paths whose horizontal legs reverse direction."""
    originals = initial_link_paths or link_paths
    for link_key, points in list(link_paths.items()):
        if not _path_has_horizontal_backtrack(points):
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
        original = originals.get(link_key)
        if original is not None:
            preferred = _prefer_non_backtracking_connector_path(
                original,
                points,
                source=source,
                target=target,
                obstacles=obstacles,
            )
            if not _path_has_horizontal_backtrack(preferred):
                link_paths[link_key] = preferred
                continue
        fresh = _connector_points_for_link(
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
            inline_bypass_bus_x=inline_bypass_bus_x,
        )
        if fresh is not None and len(fresh) >= 2 and not _path_has_horizontal_backtrack(fresh):
            link_paths[link_key] = fresh


def _vertical_segment_top_at_crossing(
    points: list[tuple[float, float]],
    cx: float,
    cy: float,
) -> float | None:
    """Return the upper Y of the vertical segment involved in a crossing."""
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if abs(x1 - x2) > 0.005:
            continue
        if abs(x1 - cx) > 0.02:
            continue
        low_y, high_y = sorted((y1, y2))
        if low_y + 0.01 < cy < high_y - 0.01:
            return high_y
    return None


def _horizontal_segment_index_at_crossing(
    points: list[tuple[float, float]],
    cx: float,
    cy: float,
) -> int | None:
    """Return the segment index of the horizontal run involved in a crossing."""
    for index, ((x1, y1), (x2, y2)) in enumerate(zip(points, points[1:])):
        if abs(y1 - y2) > 0.005:
            continue
        low_x, high_x = sorted((x1, x2))
        if low_x + 0.01 >= cx or high_x - 0.01 <= cx:
            continue
        if abs(y1 - cy) <= 0.05:
            return index
    return None


def _vertical_segment_bottom_at_crossing(
    points: list[tuple[float, float]],
    cx: float,
    cy: float,
) -> float | None:
    """Return the lower Y of the vertical segment involved in a crossing."""
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if abs(x1 - x2) > 0.005:
            continue
        if abs(x1 - cx) > 0.02:
            continue
        low_y, high_y = sorted((y1, y2))
        if low_y + 0.01 < cy < high_y - 0.01:
            return low_y
    return None


def _connector_pair_crosses(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    link_a: tuple[int, int],
    link_b: tuple[int, int],
) -> bool:
    pair = {link_a, link_b}
    return any({crossing[0], crossing[1]} == pair for crossing in _find_connector_segment_crossings(link_paths))


def _collapse_target_bus_entry_detours_clearing_crossings(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    target_bus: dict[int, float],
    merge_entry_x: dict[tuple[int, int], float],
    max_iters: int = 16,
) -> None:
    """Drop target-bus feeds at the merge center instead of detour verticals that cross siblings."""
    for _ in range(max_iters):
        crossings = _find_connector_segment_crossings(link_paths)
        if not crossings:
            return
        repaired = False
        for link_a, link_b, (cx, _cy) in crossings:
            for v_link in (link_a, link_b):
                tgt = v_link[1]
                if tgt not in target_bus:
                    continue
                target = anchors.get(tgt)
                points = link_paths.get(v_link)
                if target is None or not points or len(points) < 4:
                    continue
                if abs(cx - target.cx) <= PARALLEL_CONNECTOR_COORD_EPS:
                    continue
                detour_index: int | None = None
                bus_y: float | None = None
                for index, ((x1, y1), (x2, y2)) in enumerate(zip(points, points[1:])):
                    if abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS:
                        continue
                    if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS:
                        continue
                    if abs(x2 - cx) > 0.02 or abs(x2 - target.cx) <= PARALLEL_CONNECTOR_COORD_EPS:
                        continue
                    if index + 2 >= len(points):
                        continue
                    vx1, vy1 = points[index + 1]
                    vx2, vy2 = points[index + 2]
                    if abs(vx1 - vx2) > PARALLEL_CONNECTOR_COORD_EPS or abs(vx1 - cx) > 0.02:
                        continue
                    detour_index = index
                    bus_y = y1
                    break
                if detour_index is None or bus_y is None:
                    continue
                entry_y = _connector_target_top_entry_y(target)
                prefix = points[: detour_index + 1]
                candidate = _ensure_orthogonal_connector_path(
                    [*prefix, (target.cx, bus_y), (target.cx, entry_y)]
                )
                trial = {**link_paths, v_link: candidate}
                if _connector_pair_crosses(trial, link_a, link_b):
                    continue
                link_paths[v_link] = candidate
                merge_entry_x[v_link] = target.cx
                repaired = True
                break
            if repaired:
                break
        if not repaired:
            return


def _reroute_gutter_bypass_feeds_clearing_crossings(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    target_bus: dict[int, float],
    incoming: dict[int, list[tuple[int, int]]],
    merge_entry_x: dict[tuple[int, int], float],
    source_bus: dict[int, float],
) -> None:
    """Rebuild Sum gutter bypass feeds so the jog row clears sibling verticals."""
    crossings = _find_connector_segment_crossings(link_paths)
    if not crossings:
        return
    affected = {link for pair in crossings for link in pair[:2]}
    for tgt, link_group in incoming.items():
        target = anchors.get(tgt)
        if target is None:
            continue
        assignments = _same_column_bypass_assignments(
            link_group,
            target,
            positions=positions,
            anchors=anchors,
            target_bus=target_bus,
        )
        for link_key, bypass in assignments.items():
            if (
                bypass.gutter_x is None
                or bypass.jog_y is None
                or link_key not in affected
                or link_key not in link_paths
            ):
                continue
            source = anchors.get(link_key[0])
            if source is None:
                continue
            # The rebuilt route returns to the source column to enter the target, so it
            # only applies to a feed that already shares the target's column.
            if abs(source.cx - target.cx) > PARALLEL_CONNECTOR_COORD_EPS:
                continue
            lo, hi = sorted((bypass.gutter_x, bypass.port_x))
            cleared_jog = _min_bus_y_clearing_vertical_connector_segments(
                lo,
                hi,
                link_paths,
                skip_link=link_key,
                proposed_y=bypass.jog_y,
            )
            if cleared_jog <= bypass.jog_y + PARALLEL_CONNECTOR_COORD_EPS:
                continue
            exit_y = _connector_source_bottom_exit_y(source)
            entry_y = _connector_target_top_entry_y(target)
            tee_y = source_bus.get(link_key[0], bypass.corridor_y)
            link_paths[link_key] = _ensure_orthogonal_connector_path(
                [
                    (source.cx, exit_y),
                    (source.cx, tee_y),
                    (bypass.gutter_x, tee_y),
                    (bypass.gutter_x, cleared_jog),
                    (source.cx, cleared_jog),
                    (source.cx, entry_y),
                ]
            )
            merge_entry_x[link_key] = target.cx


def _lift_horizontal_segments_clearing_crossings(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph=None,
    anchors: dict[int, _RenderAnchor] | None = None,
    incoming: dict[int, list[tuple[int, int]]] | None = None,
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
    target_bus: dict[int, float] | None = None,
    source_bus: dict[int, float] | None = None,
    merge_link_bus: dict[tuple[int, int], float] | None = None,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
    max_iters: int = 64,
) -> None:
    """Raise horizontal connector legs above crossing vertical runs."""
    target_bus = target_bus or {}
    source_bus = source_bus or {}
    merge_link_bus = merge_link_bus or {}
    incoming = incoming or {}
    outgoing = outgoing or {}
    for _ in range(max_iters):
        crossings = _find_connector_segment_crossings(link_paths)
        if not crossings:
            return
        lifted = False
        for link_a, link_b, (cx, cy) in crossings:
            for h_link, v_link in ((link_a, link_b), (link_b, link_a)):
                h_points = link_paths.get(h_link)
                v_points = link_paths.get(v_link)
                if h_points is None or v_points is None:
                    continue
                seg_index = _horizontal_segment_index_at_crossing(h_points, cx, cy)
                v_top = _vertical_segment_top_at_crossing(v_points, cx, cy)
                v_bottom = _vertical_segment_bottom_at_crossing(v_points, cx, cy)
                if seg_index is None or (v_top is None and v_bottom is None):
                    continue
                x1, y1 = h_points[seg_index]
                x2, _y2 = h_points[seg_index + 1]
                # Lifting a run that another connector shares would tear a tee apart, so
                # only runs this connector holds alone may move.
                if _horizontal_run_is_shared(link_paths, h_link, seg_index):
                    continue
                candidate_y_values: list[float] = []
                if v_top is not None:
                    candidate_y_values.append(v_top + margin)
                if v_bottom is not None:
                    candidate_y_values.append(v_bottom - margin)
                lifted_link = False
                for new_y in candidate_y_values:
                    if abs(new_y - y1) <= PARALLEL_CONNECTOR_COORD_EPS:
                        continue
                    updated = list(h_points)
                    updated[seg_index] = (x1, new_y)
                    updated[seg_index + 1] = (x2, new_y)
                    candidate = _ensure_orthogonal_connector_path(updated)
                    if graph is not None and anchors is not None:
                        source = anchors.get(h_link[0])
                        target = anchors.get(h_link[1])
                        if source is None or target is None:
                            continue
                        if _connector_path_has_block_edge_horizontal_jog(
                            candidate,
                            source=source,
                            target=target,
                            link_key=h_link,
                            graph=graph,
                        ):
                            continue
                        if (
                            _connector_turn_before_clearing_source(
                                candidate,
                                y_exit=_connector_source_bottom_exit_y(source),
                                source_cx=source.cx,
                            )
                            is not None
                        ):
                            continue
                        candidate = _repair_connector_source_departure(
                            candidate,
                            source=source,
                            target=target,
                            link_key=h_link,
                            graph=graph,
                        )
                    trial = {**link_paths, h_link: candidate}
                    if _connector_pair_crosses(trial, link_a, link_b):
                        continue
                    if graph is not None and anchors is not None and incoming and outgoing:
                        trial_overlaps = _find_connector_path_overlaps(
                            trial,
                            incoming=incoming,
                            outgoing=outgoing,
                            target_bus=target_bus,
                            source_bus=source_bus,
                            merge_link_bus=merge_link_bus,
                            anchors=anchors,
                            graph=graph,
                        )
                        if any(h_link in overlap for overlap in trial_overlaps):
                            continue
                    link_paths[h_link] = candidate
                    lifted = True
                    lifted_link = True
                    break
                if lifted_link:
                    break
            if lifted:
                break
        if not lifted:
            return


def _horizontal_run_is_shared(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    link_key: tuple[int, int],
    seg_index: int,
) -> bool:
    """True when another connector runs along the same row over the same span."""
    points = link_paths[link_key]
    (x1, y1), (x2, _y2) = points[seg_index], points[seg_index + 1]
    lo, hi = sorted((x1, x2))
    for other, other_points in link_paths.items():
        if other == link_key:
            continue
        for orientation, coord, other_lo, other_hi, _index in _connector_axis_segments(
            other_points
        ):
            if orientation != "h" or abs(coord - y1) > PARALLEL_CONNECTOR_COORD_EPS:
                continue
            if _ranges_overlap(lo, hi, other_lo, other_hi):
                return True
    return False


CONNECTOR_LANE_PAD = 0.03


def _connector_lane_coordinates(
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    *,
    graph=None,
    positions: list | None = None,
) -> tuple[list[float], list[float]]:
    """Vertical and horizontal corridor coordinates that clear tile and frame edges."""
    pad = CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_LANE_PAD
    lane_xs: set[float] = set()
    lane_ys: set[float] = set()
    edges = [*anchors.values(), *label_obstacles]
    if graph is not None and positions is not None:
        edges.extend(_inline_frame_bounds_obstacles(graph, positions))
    for anchor in edges:
        lane_xs.add(anchor.left - pad)
        lane_xs.add(anchor.right + pad)
        lane_ys.add(anchor.bottom - pad)
        lane_ys.add(anchor.top + pad)
    lane_xs.update(_edge_gap_channels(edge for anchor in edges for edge in (anchor.left, anchor.right)))
    lane_ys.update(_edge_gap_channels(edge for anchor in edges for edge in (anchor.bottom, anchor.top)))
    return sorted(lane_xs), sorted(lane_ys)


def _edge_gap_channels(edges: Iterable[float]) -> list[float]:
    """Channels running down the narrow gaps between two tiles' edges.

    The edge-plus-padding lanes miss the sliver between a tile and one offset beside it, and
    that sliver is sometimes the only way past a tile that covers its target's top edge. A
    gap wide enough to hold a channel gets one down its middle.
    """
    channel = PARALLEL_CONNECTOR_CHANNEL_GAP
    ordered = sorted(set(edges))
    return [
        (lower + upper) / 2
        for lower, upper in zip(ordered, ordered[1:])
        if channel <= upper - lower < 3 * channel
    ]


def _connector_sibling_junctions(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    link_key: tuple[int, int],
    points: list[tuple[float, float]],
    *,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    ignore: set[tuple[int, int]] | None = None,
    obstacles: list[_RenderAnchor] | None = None,
) -> frozenset[tuple[tuple[int, int], str, int]]:
    """Rows and columns a path shares with the links it meets at either endpoint.

    Fan-outs branch off a shared stem and merge feeds tee onto a shared row; both show up
    as a collinear run that two paths hold in common. Rerouting has to keep them, or the
    junction the group is drawn around comes apart. Runs that are reported as an overlap
    are the opposite of a junction and are skipped, so a bad route cannot pin itself. A run
    that grazes a tile is skipped for the same reason: it has to move regardless, so
    treating it as a junction would only pin the group to a run nobody can keep.
    """
    src, tgt = link_key
    siblings = {
        other
        for other in (*outgoing.get(src, ()), *incoming.get(tgt, ()))
        if other != link_key and other in link_paths and other not in (ignore or ())
    }
    own = _connector_axis_segments(points)
    junctions: set[tuple[tuple[int, int], str, int]] = set()
    for sibling in siblings:
        for orientation, coord, lo, hi, _index in _connector_axis_segments(link_paths[sibling]):
            for own_orientation, own_coord, own_lo, own_hi, own_index in own:
                if own_orientation != orientation:
                    continue
                if abs(own_coord - coord) > PARALLEL_CONNECTOR_COORD_EPS:
                    continue
                if not _ranges_overlap(own_lo, own_hi, lo, hi):
                    continue
                del own_index
                run = (
                    [(own_coord, own_lo), (own_coord, own_hi)]
                    if orientation == "v"
                    else [(own_lo, own_coord), (own_hi, own_coord)]
                )
                if obstacles and _path_hits_obstacles(
                    run, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN
                ):
                    continue
                junctions.add((sibling, orientation, _parallel_coord_bucket(coord)))
    return frozenset(junctions)


def _polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(
        abs(x2 - x1) + abs(y2 - y1)
        for (x1, y1), (x2, y2) in zip(points, points[1:])
    )


def _candidate_connector_routes(
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    entry_x: float,
    lane_xs: list[float],
    lane_ys: list[float],
) -> list[list[tuple[float, float]]]:
    """Orthogonal source-to-target routes built from clear corridor coordinates."""
    exit_y = _connector_source_bottom_exit_y(source)
    entry_y = _connector_target_top_entry_y(target)
    # A shallow row cannot hold a full exit stub, so the stub shrinks to share the row
    # with the entry drop rather than giving up on routing the link at all.
    stub_y = exit_y - min(CONNECTOR_EXIT_STUB, (exit_y - entry_y) / 2)
    if stub_y <= entry_y + PARALLEL_CONNECTOR_COORD_EPS:
        return []
    approach_ys = [
        lane_y
        for lane_y in (*lane_ys, stub_y, entry_y + TOP_ENTRY_PORT_GAP)
        if entry_y + PARALLEL_CONNECTOR_COORD_EPS < lane_y <= stub_y
    ]
    routes: list[list[tuple[float, float]]] = []
    if abs(source.cx - entry_x) <= PARALLEL_CONNECTOR_COORD_EPS:
        routes.append([(source.cx, exit_y), (source.cx, entry_y)])
    for approach_y in approach_ys:
        routes.append(
            [
                (source.cx, exit_y),
                (source.cx, approach_y),
                (entry_x, approach_y),
                (entry_x, entry_y),
            ]
        )
    for lane_x in lane_xs:
        if abs(lane_x - source.cx) <= PARALLEL_CONNECTOR_COORD_EPS:
            continue
        for approach_y in approach_ys:
            if abs(approach_y - stub_y) <= PARALLEL_CONNECTOR_COORD_EPS:
                continue
            routes.append(
                [
                    (source.cx, exit_y),
                    (source.cx, stub_y),
                    (lane_x, stub_y),
                    (lane_x, approach_y),
                    (entry_x, approach_y),
                    (entry_x, entry_y),
                ]
            )
    return routes


# A corner costs the reader more than a slightly longer run, and a crossing costs more
# again, so the corridor search prices both against the length it saves. What a crossing is
# worth depends on how far around it a connector would have to go, so it is priced as a
# fraction of the corridor lattice's own reach: the first pass will take almost any detour
# to avoid one, and the last ignores crossings and simply returns the shortest route.
CONNECTOR_GRID_BEND_COST = 0.12
CONNECTOR_GRID_CROSSING_COST_FACTORS = (2.0, 0.25, 0.0)
CONNECTOR_GRID_MAX_LANES = 160


def _densified_lane_coordinates(lanes: Sequence[float], *, budget: int) -> list[float]:
    """Fill wide gaps between corridor lanes with extra lanes a channel gap apart.

    Tile edges only ever offer one lane each, so several long runs that all have to pass
    the same wall end up stacked on the one lane outside it. Open space between lanes is
    where those runs can spread out, so it is offered to the search as lanes of its own.
    """
    ordered = _deduplicated_coordinates(lanes)
    if len(ordered) < 2 or budget <= 0:
        return ordered
    gap = PARALLEL_CONNECTOR_CHANNEL_GAP
    extra: list[float] = []
    for low, high in zip(ordered, ordered[1:]):
        lane = low + gap
        while lane < high - gap / 2 and len(extra) < budget:
            extra.append(lane)
            lane += gap
        if len(extra) >= budget:
            break
    return _deduplicated_coordinates([*ordered, *extra])


def _deduplicated_coordinates(values: Sequence[float]) -> list[float]:
    """Sorted coordinates with values a connector cannot tell apart collapsed together."""
    ordered: list[float] = []
    for value in sorted(values):
        if not ordered or value - ordered[-1] > PARALLEL_CONNECTOR_COORD_EPS:
            ordered.append(value)
    return ordered


@dataclass(frozen=True)
class _ConnectorRouteGrid:
    """Corridor lattice a connector may be routed along, and where it may leave it."""

    xs: list[float]
    ys: list[float]
    v_blocked: list[list[bool]]
    h_blocked: list[list[bool]]
    drops: dict[int, list[bool]]
    exit_y: float
    entry_y: float


def _connector_route_grid(
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    entry_xs: Sequence[float],
    lane_xs: Sequence[float],
    lane_ys: Sequence[float],
    obstacles: list[_RenderAnchor],
) -> _ConnectorRouteGrid | None:
    """Corridor lattice for a link, with the steps that cut into an obstacle marked.

    The lattice only depends on the tiles a link may not touch, so it is worth building
    once per link and reusing while the surrounding connectors move around it. Its own
    endpoints count as obstacles too, so a run that dips below the target to get around a
    wall cannot come back up through the tile it is trying to enter. Reaching the port is
    left to a separate drop, which is the one run allowed to touch the target's top edge.
    """
    exit_y = _connector_source_bottom_exit_y(source)
    entry_y = _connector_target_top_entry_y(target)
    stub_y = exit_y - min(CONNECTOR_EXIT_STUB, (exit_y - entry_y) / 2)
    if stub_y <= entry_y + PARALLEL_CONNECTOR_COORD_EPS:
        return None
    margin = CONNECTOR_OBSTACLE_MARGIN
    xs = _densified_lane_coordinates(
        [*lane_xs, source.cx, *entry_xs],
        budget=CONNECTOR_GRID_MAX_LANES // 2,
    )
    ys = _densified_lane_coordinates(
        [
            *(
                lane_y
                for lane_y in (*lane_ys, entry_y + TOP_ENTRY_PORT_GAP)
                if lane_y < stub_y - PARALLEL_CONNECTOR_COORD_EPS
            ),
            stub_y,
        ],
        budget=CONNECTOR_GRID_MAX_LANES // 2,
    )
    if len(xs) < 2 or len(ys) < 2:
        return None
    if len(xs) > CONNECTOR_GRID_MAX_LANES or len(ys) > CONNECTOR_GRID_MAX_LANES:
        return None
    v_blocked = [[False] * (len(ys) - 1) for _ in xs]
    h_blocked = [[False] * (len(xs) - 1) for _ in ys]
    for obs in (*obstacles, source, target):
        y_lo = max(0, bisect_left(ys, obs.bottom - margin) - 1)
        y_hi = min(len(ys) - 2, bisect_right(ys, obs.top + margin) - 1)
        if y_lo <= y_hi:
            for xi in range(
                bisect_left(xs, obs.left - margin), bisect_right(xs, obs.right + margin)
            ):
                v_blocked[xi][y_lo : y_hi + 1] = [True] * (y_hi - y_lo + 1)
        x_lo = max(0, bisect_left(xs, obs.left - margin) - 1)
        x_hi = min(len(xs) - 2, bisect_right(xs, obs.right + margin) - 1)
        if x_lo <= x_hi:
            for yj in range(
                bisect_left(ys, obs.bottom - margin), bisect_right(ys, obs.top + margin)
            ):
                h_blocked[yj][x_lo : x_hi + 1] = [True] * (x_hi - x_lo + 1)
    drops: dict[int, list[bool]] = {}
    for entry_x in _deduplicated_coordinates(entry_xs):
        column = bisect_left(xs, entry_x - PARALLEL_CONNECTOR_COORD_EPS)
        if not 0 <= column < len(xs):
            continue
        drops[column] = [
            lane_y >= entry_y + margin
            and not _segment_hits_obstacle(
                entry_x, lane_y, entry_x, entry_y, obstacles, margin=margin
            )
            for lane_y in ys
        ]
    if not drops:
        return None
    return _ConnectorRouteGrid(
        xs=xs,
        ys=ys,
        v_blocked=v_blocked,
        h_blocked=h_blocked,
        drops=drops,
        exit_y=exit_y,
        entry_y=entry_y,
    )


def _connector_route_grid_crossings(
    xs: list[float],
    ys: list[float],
    other_paths: Sequence[list[tuple[float, float]]],
) -> tuple[list[list[int]], list[list[int]]]:
    """How many existing connector runs each corridor step would cross."""
    v_cross = [[0] * (len(ys) - 1) for _ in xs]
    h_cross = [[0] * (len(xs) - 1) for _ in ys]
    eps = PARALLEL_CONNECTOR_COORD_EPS
    for points in other_paths:
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            if abs(y1 - y2) <= eps and abs(x1 - x2) > eps:
                row = bisect_right(ys, (y1 + y2) / 2) - 1
                if not 0 <= row <= len(ys) - 2:
                    continue
                lo_x, hi_x = sorted((x1, x2))
                for xi in range(bisect_right(xs, lo_x), bisect_left(xs, hi_x)):
                    v_cross[xi][row] += 1
            elif abs(x1 - x2) <= eps and abs(y1 - y2) > eps:
                column = bisect_right(xs, (x1 + x2) / 2) - 1
                if not 0 <= column <= len(xs) - 2:
                    continue
                lo_y, hi_y = sorted((y1, y2))
                for yj in range(bisect_right(ys, lo_y), bisect_left(ys, hi_y)):
                    h_cross[yj][column] += 1
    return v_cross, h_cross


def _vertical_span_crossing_count(
    x: float,
    y_from: float,
    y_to: float,
    horizontal_runs: Sequence[tuple[float, float, float]],
) -> int:
    """Existing horizontal runs a vertical span at one x would cut across."""
    eps = PARALLEL_CONNECTOR_COORD_EPS
    lo_y, hi_y = sorted((y_from, y_to))
    return sum(
        1
        for run_y, run_lo, run_hi in horizontal_runs
        if lo_y + eps < run_y < hi_y - eps and run_lo + eps < x < run_hi - eps
    )


def _grid_connector_routes(
    grid: _ConnectorRouteGrid,
    crossings: tuple[list[list[int]], list[list[int]]],
    horizontal_runs: Sequence[tuple[float, float, float]],
    *,
    source_cx: float,
    entry_x: float,
    crossing_cost: float,
    max_routes: int = 3,
) -> list[list[tuple[float, float]]]:
    """Cheapest corridor routes from a source column down onto one target entry port.

    Searching the lattice rather than a fixed set of route shapes is what lets a connector
    thread past a tile that blocks every straight detour, which is the only way out of a
    column that is walled in on both sides. One search prices every row the port can be
    reached from, so the caller gets alternatives for the rules the lattice cannot see,
    such as the dotted border a frame member's approach has to stay clear of.
    """
    xs, ys = grid.xs, grid.ys
    v_cross, h_cross = crossings
    start_xi = bisect_left(xs, source_cx - PARALLEL_CONNECTOR_COORD_EPS)
    goal_xi = bisect_left(xs, entry_x - PARALLEL_CONNECTOR_COORD_EPS)
    drops = grid.drops.get(goal_xi)
    if drops is None or not 0 <= start_xi < len(xs):
        return []
    start = (start_xi, len(ys) - 1, 0)
    settled: dict[tuple[int, int, int], float] = {}
    tentative: dict[tuple[int, int, int], float] = {start: 0.0}
    previous: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[float, tuple[int, int, int]]] = [(0.0, start)]
    while heap:
        cost, state = heapq.heappop(heap)
        if state in settled:
            continue
        settled[state] = cost
        xi, yj, direction = state
        moves: list[tuple[tuple[int, int, int], float]] = []
        for step in (-1, 1):
            neighbour = yj + step
            if not 0 <= neighbour < len(ys):
                continue
            edge = min(yj, neighbour)
            if grid.v_blocked[xi][edge]:
                continue
            moves.append(
                (
                    (xi, neighbour, 0),
                    abs(ys[neighbour] - ys[yj]) + crossing_cost * v_cross[xi][edge],
                )
            )
        for step in (-1, 1):
            neighbour = xi + step
            if not 0 <= neighbour < len(xs):
                continue
            edge = min(xi, neighbour)
            if grid.h_blocked[yj][edge]:
                continue
            moves.append(
                (
                    (neighbour, yj, 1),
                    abs(xs[neighbour] - xs[xi]) + crossing_cost * h_cross[yj][edge],
                )
            )
        for next_state, weight in moves:
            if next_state in settled:
                continue
            if next_state[2] != direction:
                weight += CONNECTOR_GRID_BEND_COST
            candidate = cost + weight
            if candidate >= tentative.get(next_state, float("inf")):
                continue
            tentative[next_state] = candidate
            previous[next_state] = state
            heapq.heappush(heap, (candidate, next_state))
    priced: list[tuple[float, tuple[int, int, int]]] = []
    for yj, allowed in enumerate(drops):
        if not allowed:
            continue
        drop = ys[yj] - grid.entry_y + crossing_cost * _vertical_span_crossing_count(
            entry_x, ys[yj], grid.entry_y, horizontal_runs
        )
        for direction in (0, 1):
            reached = settled.get((goal_xi, yj, direction))
            if reached is None:
                continue
            priced.append(
                (reached + drop + (CONNECTOR_GRID_BEND_COST if direction else 0.0), (goal_xi, yj, direction))
            )
    routes: list[list[tuple[float, float]]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for _cost, state in sorted(priced):
        walk = [state]
        while walk[-1] != start:
            walk.append(previous[walk[-1]])
        route = _ensure_orthogonal_connector_path(
            _collapse_connector_run_vertices(
                [
                    (source_cx, grid.exit_y),
                    *((xs[xi], ys[row]) for xi, row, _dir in reversed(walk)),
                    (entry_x, grid.entry_y),
                ]
            )
        )
        marker = tuple(route)
        if marker in seen:
            continue
        seen.add(marker)
        routes.append(route)
        if len(routes) >= max_routes:
            break
    return routes


def _collapse_connector_run_vertices(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Drop the interior vertices a straight run picked up from crossing corridor lanes."""
    eps = PARALLEL_CONNECTOR_COORD_EPS
    collapsed: list[tuple[float, float]] = list(points[:1])
    for index in range(1, len(points) - 1):
        x_prev, y_prev = collapsed[-1]
        x_mid, y_mid = points[index]
        x_next, y_next = points[index + 1]
        if abs(x_prev - x_mid) <= eps and abs(x_mid - x_next) <= eps:
            continue
        if abs(y_prev - y_mid) <= eps and abs(y_mid - y_next) <= eps:
            continue
        collapsed.append((x_mid, y_mid))
    collapsed.extend(points[-1:])
    return collapsed


def _admissible_connector_route(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    graph,
    link_key: tuple[int, int],
    positions: list | None = None,
) -> list[tuple[float, float]] | None:
    """Return the route when it satisfies the shared connector departure/clearance rules."""
    normalized = _drop_collinear_connector_vertices(_ensure_orthogonal_connector_path(points))
    if len(normalized) < 2:
        return None
    # The straightened shape is offered first: a jog narrower than a channel is a kink the
    # reader has to look twice at, and it is only kept when straightening it hits something.
    for path in _distinct_connector_routes(
        _straighten_sub_channel_kinks(normalized),
        normalized,
    ):
        if _connector_route_is_admissible(
            path,
            source=source,
            target=target,
            obstacles=obstacles,
            graph=graph,
            link_key=link_key,
            positions=positions,
        ):
            return path
    return None


def _connector_route_no_worse_than(
    points: list[tuple[float, float]],
    current: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    graph,
    link_key: tuple[int, int],
    positions: list | None = None,
) -> list[tuple[float, float]] | None:
    """Return the route when it breaks no rule the one it would replace already breaks.

    Some routes reach the polish passes already touching a tile, usually because the gutter
    they were given is exactly as wide as the clearance they owe. Holding a repair to a
    standard the route never met would leave the overlap it was called to fix.
    """
    admissible = _admissible_connector_route(
        points,
        source=source,
        target=target,
        obstacles=obstacles,
        graph=graph,
        link_key=link_key,
        positions=positions,
    )
    if admissible is not None:
        return admissible
    inherited = _connector_route_faults(
        current,
        source=source,
        target=target,
        obstacles=obstacles,
        graph=graph,
        link_key=link_key,
        positions=positions,
    )
    if not inherited:
        return None
    normalized = _drop_collinear_connector_vertices(_ensure_orthogonal_connector_path(points))
    if len(normalized) < 2:
        return None
    faults = _connector_route_faults(
        normalized,
        source=source,
        target=target,
        obstacles=obstacles,
        graph=graph,
        link_key=link_key,
        positions=positions,
    )
    return normalized if faults <= inherited else None


def _distinct_connector_routes(
    *routes: list[tuple[float, float]],
) -> list[list[tuple[float, float]]]:
    """The given routes with duplicates dropped, in the order offered."""
    kept: list[list[tuple[float, float]]] = []
    for route in routes:
        if all(tuple(route) != tuple(other) for other in kept):
            kept.append(route)
    return kept


def _connector_route_is_admissible(
    path: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    graph,
    link_key: tuple[int, int],
    positions: list | None,
) -> bool:
    """Whether a route satisfies every shared departure, clearance and entry rule."""
    return not _connector_route_faults(
        path,
        source=source,
        target=target,
        obstacles=obstacles,
        graph=graph,
        link_key=link_key,
        positions=positions,
    )


def _connector_route_faults(
    path: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    graph,
    link_key: tuple[int, int],
    positions: list | None,
) -> set[str]:
    """Name every departure, clearance and entry rule a route breaks.

    Naming them rather than answering yes or no lets a repair pass tell a route it has
    spoiled from one that merely inherits a fault the route already had, which it has no
    business being blocked by.
    """
    faults: set[str] = set()
    if not _connector_path_is_orthogonal(path):
        faults.add("diagonal")
    if _connector_route_climbs_back_towards_source(path, source=source, target=target):
        faults.add("climbs back")
    if _connector_route_leaves_own_frame(
        path,
        graph=graph,
        positions=positions,
        link_key=link_key,
    ):
        faults.add("leaves frame")
    if _path_hits_obstacles(path, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN):
        faults.add("touches tile")
    if positions is not None and _find_connector_inline_frame_overlaps(
        {link_key: path},
        graph=graph,
        positions=positions,
    ):
        faults.add("cuts frame")
    if _connector_path_has_block_edge_horizontal_jog(
        path,
        source=source,
        target=target,
        link_key=link_key,
        graph=graph,
    ):
        faults.add("edge jog")
    if (
        _connector_turn_before_clearing_source(
            path,
            y_exit=_connector_source_bottom_exit_y(source),
            source_cx=source.cx,
        )
        is not None
    ):
        faults.add("turns inside exit stub")
    if _connector_path_departs_horizontally_from_source(path, source=source):
        faults.add("departs sideways")
    # A top entry approaches from above, so the run that feeds the port sits in the
    # corridor the target reserves for it. Earlier legs may still dip below the target to
    # get around a wall, provided they keep clear of both tiles the link belongs to.
    if _connector_entry_approach_below_target_corridor(path, target):
        faults.add("approaches from below")
    if _connector_entry_port_off_the_top_edge(path, target):
        faults.add("port off the edge")
    if _spread_merge_horizontal_below_target_corridor(
        path, target
    ) and _connector_path_cuts_endpoint_tiles(path, source=source, target=target):
        faults.add("cuts its own tiles")
    return faults


def _straighten_sub_channel_kinks(
    points: list[tuple[float, float]],
    *,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> list[tuple[float, float]]:
    """Snap a step narrower than a channel onto one straight run.

    A route that shifts sideways by less than the channel gap and carries on in the same
    direction has gained nothing for the offset: it reads as a wobble in a line that was
    meant to be straight. Both runs move onto whichever of the two the longer one used.
    """
    gap = PARALLEL_CONNECTOR_CHANNEL_GAP
    path = list(points)
    for _ in range(len(path)):
        for index in range(1, len(path) - 2):
            if index - 1 < 1 or index + 2 > len(path) - 2:
                continue
            before = path[index - 1]
            start, stop = path[index], path[index + 1]
            after = path[index + 2]
            for axis in (0, 1):
                other = 1 - axis
                step = abs(start[axis] - stop[axis])
                if (
                    abs(before[axis] - start[axis]) > eps
                    or abs(stop[axis] - after[axis]) > eps
                    or abs(start[other] - stop[other]) > eps
                    or not 0.0 < step < gap
                ):
                    continue
                keep = (
                    start[axis]
                    if abs(before[other] - start[other]) >= abs(stop[other] - after[other])
                    else stop[axis]
                )
                for offset in range(-1, 3):
                    point = list(path[index + offset])
                    point[axis] = keep
                    path[index + offset] = (point[0], point[1])
                break
            else:
                continue
            break
        else:
            break
        path = _drop_collinear_connector_vertices(_dedupe_polyline_points(path, eps=eps))
    return path


def _drop_collinear_connector_vertices(
    points: list[tuple[float, float]],
    *,
    eps: float = 1e-9,
) -> list[tuple[float, float]]:
    """Drop vertices that sit mid-run, so a straight feed is drawn as one segment.

    The tolerance is deliberately tight: a vertex only a hair off the run still carries a
    real offset, and dropping it would leave the neighbouring segment slanted.
    """
    if len(points) < 3:
        return list(points)
    kept = [points[0]]
    for (before_x, before_y), (x, y), (after_x, after_y) in zip(
        points, points[1:], points[2:]
    ):
        vertical = abs(before_x - x) <= eps and abs(x - after_x) <= eps
        horizontal = abs(before_y - y) <= eps and abs(y - after_y) <= eps
        if not (vertical or horizontal):
            kept.append((x, y))
    kept.append(points[-1])
    return kept


def _connector_route_climbs_back_towards_source(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
) -> bool:
    """True when a downhill route turns back up towards the row it started from.

    A route reaching a target below its source may dip past it to get around a wall, but
    climbing back to where it left the source reads as a loop that goes nowhere: whatever it
    was avoiding could have been passed on the way down.
    """
    if target.top > source.bottom - PARALLEL_CONNECTOR_COORD_EPS:
        return False
    ceiling = _connector_source_bottom_exit_y(source)
    return any(
        abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS
        and y2 > y1 + PARALLEL_CONNECTOR_COORD_EPS
        and y2 > ceiling + PARALLEL_CONNECTOR_COORD_EPS
        for (x1, y1), (x2, y2) in zip(points, points[1:])
    )


def _connector_route_leaves_own_frame(
    points: list[tuple[float, float]],
    *,
    graph,
    positions: list | None,
    link_key: tuple[int, int],
) -> bool:
    """True when a link inside one dotted frame is drawn outside it.

    A frame draws a submodule, so a connector with both ends inside one belongs to that
    submodule and has to be drawn within its border; stepping outside reads as the value
    leaving the submodule and coming back.
    """
    if positions is None:
        return False
    src, tgt = link_key
    for frame in graph.inline_frames:
        members = set(frame.node_indices)
        if src not in members or tgt not in members:
            continue
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        if not _path_stays_inside_bounds(points, bounds, margin=0.0):
            return True
    return False


def _connector_entry_approach_below_target_corridor(
    points: list[tuple[float, float]],
    target: _RenderAnchor,
) -> bool:
    """True when the run that feeds the entry port sits below the target's own corridor."""
    for index in range(len(points) - 2, -1, -1):
        x1, y1 = points[index]
        x2, _y2 = points[index + 1]
        if abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS:
            continue
        floor_y = _connector_min_bus_y_above_target(target)
        return y1 < floor_y - PARALLEL_CONNECTOR_COORD_EPS
    return False


def _connector_entry_port_off_the_top_edge(
    points: list[tuple[float, float]],
    target: _RenderAnchor,
) -> bool:
    """True when a top-edge drop lands past the corner it is meant to enter through.

    A port that overhangs the corner draws the arrow head on the tile's rounded edge, so it
    reads as a wire brushing past rather than one feeding in.
    """
    if len(points) < 2:
        return False
    end_x, end_y = points[-1]
    if abs(end_y - _connector_target_top_entry_y(target)) > PARALLEL_CONNECTOR_COORD_EPS:
        return False
    eps = 1e-6
    return not (
        target.left + CONNECTOR_ATTACHED_BOX_MARGIN - eps
        <= end_x
        <= target.right - CONNECTOR_ATTACHED_BOX_MARGIN + eps
    )


def _connector_path_docks_on_a_side_edge(
    points: list[tuple[float, float]],
    target: _RenderAnchor,
) -> bool:
    """True when the route ends by running into the left or right edge of its target."""
    if len(points) < 2:
        return False
    (x_prev, _y_prev), (x_end, y_end) = points[-2], points[-1]
    if abs(x_prev - x_end) <= PARALLEL_CONNECTOR_COORD_EPS:
        return False
    return (
        target.bottom - PARALLEL_CONNECTOR_COORD_EPS
        <= y_end
        <= target.top + PARALLEL_CONNECTOR_COORD_EPS
    ) and min(
        abs(x_end - target.left), abs(x_end - target.right)
    ) <= PARALLEL_CONNECTOR_COORD_EPS


def _connector_path_cuts_endpoint_tiles(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
) -> bool:
    """True when a route runs through the tile it leaves or the tile it enters.

    The stub off the source's bottom edge and the drop onto the target's port each touch
    their own tile by design; every leg between them has to stay clear of both.
    """
    return len(points) > 3 and _path_hits_obstacles(
        points[1:-1], [source, target], margin=CONNECTOR_OBSTACLE_MARGIN
    )


def _find_crowded_parallel_connector_runs(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Pairs of connectors whose parallel runs are neither shared nor properly separated.

    Two runs on the same axis either sit on one channel, and read as a single bus, or they
    need the channel gap between them; anything in between reads as a smudge.
    """
    separation = PARALLEL_CONNECTOR_CHANNEL_GAP / 2
    # Runs on one shared channel are assigned the same coordinate, so only float drift
    # counts as collinear here.
    shared = 1e-3
    segments = {
        link_key: _connector_axis_segments(points)
        for link_key, points in link_paths.items()
    }
    crowded: list[tuple[tuple[int, int], tuple[int, int]]] = []
    keys = sorted(segments)
    for first_index, first in enumerate(keys):
        for second in keys[first_index + 1 :]:
            for orientation, coord, lo, hi, _index in segments[first]:
                for other_orientation, other_coord, other_lo, other_hi, _other in segments[
                    second
                ]:
                    if orientation != other_orientation:
                        continue
                    offset = abs(coord - other_coord)
                    if offset <= shared or offset >= separation:
                        continue
                    if not _ranges_overlap(lo, hi, other_lo, other_hi):
                        continue
                    crowded.append((first, second))
                    break
                else:
                    continue
                break
    return crowded


def _connector_path_crossing_count(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    link_key: tuple[int, int],
) -> int:
    """How many other connectors the given link crosses."""
    return sum(
        link_key in (link_a, link_b)
        for link_a, link_b, _point in _find_connector_segment_crossings(link_paths)
    )


_SIDE_ENTRY_LINK_CACHE: dict[int, tuple[object, frozenset[tuple[int, int]]]] = {}


def _side_entry_links(graph) -> frozenset[tuple[int, int]]:
    """Side feeds of a graph, remembered because scoring asks for them thousands of times."""
    from visualizer.computation_graph import _infer_side_entry_links

    cached = _SIDE_ENTRY_LINK_CACHE.get(id(graph))
    if cached is not None and cached[0] is graph:
        return cached[1]
    if len(_SIDE_ENTRY_LINK_CACHE) > 32:
        _SIDE_ENTRY_LINK_CACHE.clear()
    links = frozenset(_infer_side_entry_links(graph))
    _SIDE_ENTRY_LINK_CACHE[id(graph)] = (graph, links)
    return links


def _connector_violation_links(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    only: tuple[int, int] | None = None,
) -> tuple[int, set[tuple[int, int]]]:
    """Weigh crossing, overlap, frame, clearance and entry faults, and name the links.

    The weights are what let the resolver trade one fault for another. A crossing costs the
    reader a moment; a doubled run costs them a whole edge, because there is no way to tell
    which of the two values the line carries, and a run through a tile or a frame is read as
    a connection that is not there. So those outrank a crossing and the search will accept a
    crossing to be rid of one.
    """
    offenders: set[tuple[int, int]] = set()
    score = 0

    def record(*links: tuple[int, int], cost: int = 1) -> None:
        nonlocal score
        if only is not None and only not in links:
            return
        offenders.update(links)
        score += cost

    for link_a, link_b, _point in _find_connector_segment_crossings(link_paths):
        record(link_a, link_b)
    for first, second in _find_connector_path_overlaps(
        link_paths,
        incoming=incoming,
        outgoing=outgoing,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
        anchors=anchors,
        graph=graph,
    ):
        record(first, second, cost=CONNECTOR_OVERLAP_COST)
    if positions is not None:
        for link_key, _frame_id, _reason in _find_connector_inline_frame_overlaps(
            link_paths,
            graph=graph,
            positions=positions,
        ):
            record(link_key, cost=CONNECTOR_THROUGH_TILE_COST)
    for link_key, _reason in _find_connector_node_clearance_violations(
        link_paths,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
    ):
        record(link_key, cost=CONNECTOR_THROUGH_TILE_COST)
    for link_key, _reason in _find_connector_entry_approach_violations(
        link_paths,
        graph=graph,
        anchors=anchors,
    ):
        record(link_key, cost=CONNECTOR_ENTRY_FAULT_COST)
    for link_key, _reason in _find_connector_entry_port_violations(
        link_paths,
        graph=graph,
        anchors=anchors,
    ):
        record(link_key, cost=CONNECTOR_ENTRY_FAULT_COST)
    return score, offenders


def _assert_detail_link_paths_have_no_geometry_violations(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
) -> None:
    """Reject drawn connectors that overlap, cut a dotted frame, or touch a passed tile."""
    from visualizer.computation_graph import _infer_side_entry_links

    overlap_pairs = [
        (first, second)
        for first, second in _find_connector_path_overlaps(
            link_paths,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            allow_shared_buses=True,
            anchors=anchors,
            graph=graph,
        )
        if first[1] != second[0] and second[1] != first[0]
    ]
    if overlap_pairs:
        raise RuntimeError(
            "connector overlap after layout: "
            + ", ".join(f"{pair[0]}|{pair[1]}" for pair in overlap_pairs[:4])
        )
    side_entry_links = set(_infer_side_entry_links(graph))
    frame_overlaps = [
        overlap
        for overlap in _find_connector_inline_frame_overlaps(
            link_paths,
            graph=graph,
            positions=positions,
        )
        if overlap[0] not in side_entry_links
    ]
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
        link_paths,
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


def _admissible_reroutes_for_link(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    link_key: tuple[int, int],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    lane_xs: list[float],
    lane_ys: list[float],
    grid_cache: dict | None = None,
) -> list[list[tuple[float, float]]]:
    """Clear orthogonal routes a link could take instead of the one it holds.

    The corridor search comes first: it prices crossings and corners against length, so its
    routes are the ones most likely to resolve a violation. The fixed detour shapes follow,
    shortest first, so a caller that only wants a handful still gets the tidiest ones.
    """
    src, tgt = link_key
    source = anchors.get(src)
    target = anchors.get(tgt)
    current = link_paths.get(link_key)
    if source is None or target is None or not current:
        return []
    obstacles = _connector_block_obstacles(
        anchors,
        src=src,
        tgt=tgt,
        label_obstacles=label_obstacles,
        graph=graph,
        positions=positions,
        link_key=link_key,
    )
    # The merge stages assign the entry port, and keeping it is what holds a spread of feeds
    # in a readable order, so the current port comes first. Where no corridor works from it,
    # a free port elsewhere on the target's top edge still beats a crossing, as long as it
    # stays clear of its siblings'.
    entry_xs = {current[-1][0]}
    sibling_ports = [
        points[-1][0]
        for other, points in link_paths.items()
        if other != link_key and other[1] == tgt and points
    ]
    entry_xs.update(_free_top_entry_ports(target, sibling_ports))
    admissible: list[list[tuple[float, float]]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()

    def keep(route: list[tuple[float, float]] | None) -> None:
        if route is None:
            return
        path = _admissible_connector_route(
            route,
            source=source,
            target=target,
            obstacles=obstacles,
            graph=graph,
            link_key=link_key,
            positions=positions,
        )
        if path is None or not _top_entry_port_faces_approach(path, target=target):
            return
        marker = tuple(path)
        if marker in seen:
            return
        seen.add(marker)
        admissible.append(path)

    grid_key = (link_key, tuple(sorted(entry_xs)))
    if grid_cache is None or grid_key not in grid_cache:
        grid = _connector_route_grid(
            source=source,
            target=target,
            entry_xs=sorted(entry_xs),
            lane_xs=lane_xs,
            lane_ys=lane_ys,
            obstacles=obstacles,
        )
        if grid_cache is not None:
            grid_cache[grid_key] = grid
    else:
        grid = grid_cache[grid_key]
    if grid is not None:
        other_paths = [points for other, points in link_paths.items() if other != link_key]
        crossings = _connector_route_grid_crossings(grid.xs, grid.ys, other_paths)
        horizontal_runs = [
            ((y1 + y2) / 2, min(x1, x2), max(x1, x2))
            for points in other_paths
            for (x1, y1), (x2, y2) in zip(points, points[1:])
            if abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(x1 - x2) > PARALLEL_CONNECTOR_COORD_EPS
        ]
        reach = (grid.xs[-1] - grid.xs[0]) + (grid.ys[-1] - grid.ys[0])
        for factor in CONNECTOR_GRID_CROSSING_COST_FACTORS:
            for entry_x in sorted(entry_xs):
                for route in _grid_connector_routes(
                    grid,
                    crossings,
                    horizontal_runs,
                    source_cx=source.cx,
                    entry_x=entry_x,
                    crossing_cost=factor * reach,
                ):
                    keep(route)
    searched = len(admissible)
    for entry_x in sorted(entry_xs):
        for route in _candidate_connector_routes(
            source=source,
            target=target,
            entry_x=entry_x,
            lane_xs=lane_xs,
            lane_ys=lane_ys,
        ):
            keep(route)
    detours = admissible[searched:]
    detours.sort(key=lambda path: (len(path), _polyline_length(path)))
    return [*admissible[:searched], *detours]


def _top_entry_port_faces_approach(
    points: list[tuple[float, float]],
    *,
    target: _RenderAnchor,
) -> bool:
    """True when a top entry sits between the target's centre and where the run comes from.

    A port on the far side of the centre makes the connector cut back across the tile it is
    entering, which reads as though it belongs to the feed on the other side.
    """
    if len(points) < 3:
        return True
    entry_x = points[-1][0]
    approach_x = points[-3][0]
    if abs(approach_x - entry_x) <= PARALLEL_CONNECTOR_COORD_EPS:
        return True
    if abs(entry_x - target.cx) <= PARALLEL_CONNECTOR_COORD_EPS:
        return True
    return (entry_x - target.cx) * (approach_x - target.cx) > 0


def _free_top_entry_ports(
    target: _RenderAnchor,
    taken: list[float],
) -> list[float]:
    """Ports on the target's top edge that keep the reader's spacing from those in use."""
    lo = target.left + CONNECTOR_ATTACHED_BOX_MARGIN
    hi = target.right - CONNECTOR_ATTACHED_BOX_MARGIN
    if hi <= lo:
        return []
    ports: list[float] = []
    step = TOP_ENTRY_PORT_GAP
    port_x = lo
    while port_x <= hi + PARALLEL_CONNECTOR_COORD_EPS:
        if all(abs(port_x - other) >= step - PARALLEL_CONNECTOR_COORD_EPS for other in taken):
            ports.append(min(port_x, hi))
        port_x += step
    return ports


def _segment_clearance_shifts(
    lo: float,
    hi: float,
    coord: float,
    obstacles: list[_RenderAnchor],
    *,
    vertical: bool,
    margin: float,
) -> list[float]:
    """Coordinates that move a run just outside the margin band of what it currently touches."""
    shifts: list[float] = []
    for obstacle in obstacles:
        if vertical:
            span_lo, span_hi = obstacle.bottom, obstacle.top
            near_lo, near_hi = obstacle.left, obstacle.right
        else:
            span_lo, span_hi = obstacle.left, obstacle.right
            near_lo, near_hi = obstacle.bottom, obstacle.top
        if not _ranges_overlap(lo, hi, span_lo - margin, span_hi + margin):
            continue
        if not near_lo - margin <= coord <= near_hi + margin:
            continue
        # A run pushed just past the margin can land alongside a neighbouring run, so offer a
        # ladder of clearances and let the caller take the first that works. The gap between
        # the tile edge and the next run is often narrow, so the steps have to be fine.
        for step in range(16):
            clearance = margin + 1e-3 + step * PARALLEL_CONNECTOR_CHANNEL_GAP / 4
            shifts.append(near_lo - clearance)
            shifts.append(near_hi + clearance)
    return sorted(shifts, key=lambda value: abs(value - coord))


def _nudge_connector_runs_clearing_node_margins(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    max_iters: int = 24,
) -> None:
    """Slide interior runs off the tiles they graze, one run at a time.

    The routing stages place a run against a tile edge when they are trading against a
    crossing, and the margin they leave can come out a hair short. Moving just that run to
    the far side of the margin band keeps the rest of the route, so it is the cheapest fix
    available once the shape of the path is settled.
    """
    metrics = {
        "graph": graph,
        "anchors": anchors,
        "label_obstacles": label_obstacles,
        "positions": positions,
        "incoming": incoming,
        "outgoing": outgoing,
        "target_bus": target_bus,
        "source_bus": source_bus,
        "merge_link_bus": merge_link_bus,
    }
    for _ in range(max_iters):
        moved = False
        for link_key, reason in _find_connector_node_clearance_violations(
            link_paths,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
        ):
            del reason
            points = link_paths[link_key]
            if len(points) < 4:
                continue
            baseline, _ = _connector_violation_links(link_paths, only=link_key, **metrics)
            crowded_before = sum(
                link_key in pair
                for pair in _find_crowded_parallel_connector_runs(link_paths)
            )
            obstacles = _connector_block_obstacles(
                anchors,
                src=link_key[0],
                tgt=link_key[1],
                label_obstacles=label_obstacles,
                graph=graph,
                positions=positions,
                link_key=link_key,
            )
            # The end stubs carry the source exit and target entry ports, so only the runs
            # between them are free to move.
            for index in range(1, len(points) - 2):
                (x1, y1), (x2, y2) = points[index], points[index + 1]
                vertical = abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS
                coord = x1 if vertical else y1
                lo, hi = sorted((y1, y2) if vertical else (x1, x2))
                if not _path_hits_obstacles(
                    [points[index], points[index + 1]],
                    obstacles,
                    margin=CONNECTOR_OBSTACLE_MARGIN,
                ):
                    continue
                if not vertical and _horizontal_run_is_shared(link_paths, link_key, index):
                    continue
                for shifted in _segment_clearance_shifts(
                    lo,
                    hi,
                    coord,
                    obstacles,
                    vertical=vertical,
                    margin=CONNECTOR_OBSTACLE_MARGIN,
                ):
                    updated = list(points)
                    if vertical:
                        updated[index] = (shifted, y1)
                        updated[index + 1] = (shifted, y2)
                    else:
                        updated[index] = (x1, shifted)
                        updated[index + 1] = (x2, shifted)
                    candidate = _ensure_orthogonal_connector_path(updated)
                    if candidate[0] != points[0] or candidate[-1] != points[-1]:
                        continue
                    trial = {**link_paths, link_key: candidate}
                    count, _ = _connector_violation_links(trial, only=link_key, **metrics)
                    if count >= baseline:
                        continue
                    # Sliding off a tile edge must not park the run alongside a neighbour,
                    # where the two would smudge into one line.
                    if (
                        sum(
                            link_key in pair
                            for pair in _find_crowded_parallel_connector_runs(trial)
                        )
                        > crowded_before
                    ):
                        continue
                    link_paths[link_key] = candidate
                    moved = True
                    break
                if moved:
                    break
            if moved:
                break
        if not moved:
            return


def _connector_violation_groups(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    offenders: set[tuple[int, int]],
    *,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    anchors: dict[int, _RenderAnchor],
    graph,
    max_group: int,
) -> list[list[tuple[int, int]]]:
    """Group offending links that are tangled with each other, not merely both untidy."""
    parent: dict[tuple[int, int], tuple[int, int]] = {key: key for key in offenders}

    def find(key: tuple[int, int]) -> tuple[int, int]:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(first: tuple[int, int], second: tuple[int, int]) -> None:
        if first not in parent or second not in parent:
            return
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_b] = root_a

    for first, second in _find_connector_path_overlaps(
        link_paths,
        incoming=incoming,
        outgoing=outgoing,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
        anchors=anchors,
        graph=graph,
    ):
        union(first, second)
    offender_list = sorted(offenders)
    for index, first in enumerate(offender_list):
        for second in offender_list[index + 1 :]:
            if _orthogonal_paths_crossings(link_paths[first], link_paths[second]):
                union(first, second)

    grouped: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for key in offender_list:
        grouped[find(key)].append(key)
    return [group for group in grouped.values() if 2 <= len(group) <= max_group]


def _sound_inline_frame_skip_links(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
) -> set[tuple[int, int]]:
    """Frame skip links whose reserved row is still worth holding them to.

    A skip runs on a row its frame set aside for it, and moving it off that row costs the
    frame its clearance, so the repair passes leave it alone. That only holds while the row
    still gets it to its port from above: a skip that ends up under the tile it feeds has
    to climb back through it, and no reserved row is worth that.
    """
    from visualizer.computation_graph import _inline_frame_column_skip_links

    sound: set[tuple[int, int]] = set()
    for frame in graph.inline_frames:
        for link_key in _inline_frame_column_skip_links(graph, frame):
            points = link_paths.get(link_key)
            target = anchors.get(link_key[1])
            if not points or target is None:
                continue
            if _connector_entry_approach_below_target_corridor(points, target):
                continue
            sound.add(link_key)
    return sound


def _reroute_connector_violation_groups(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    max_group: int = 5,
    max_candidates: int = 12,
    max_iters: int = 2,
) -> None:
    """Re-place a whole knot of tangled connectors at once.

    Moving one link at a time cannot untangle a funnel, where several runs have to pass the
    same wall: whichever moves first lands on a neighbour, so every single move scores worse
    than what it replaces and is refused. Re-placing the group in one go lets the steps in
    between be worse, and only the finished arrangement has to beat what it replaced.
    """
    from visualizer.computation_graph import _inline_frame_column_skip_links

    metrics = {
        "graph": graph,
        "anchors": anchors,
        "label_obstacles": label_obstacles,
        "positions": positions,
        "incoming": incoming,
        "outgoing": outgoing,
        "target_bus": target_bus,
        "source_bus": source_bus,
        "merge_link_bus": merge_link_bus,
    }
    protected = _sound_inline_frame_skip_links(
        link_paths, graph=graph, anchors=anchors
    )
    lane_xs, lane_ys = _connector_lane_coordinates(
        anchors,
        label_obstacles,
        graph=graph,
        positions=positions,
    )
    grid_cache: dict = {}
    for _ in range(max_iters):
        score, offenders = _connector_violation_links(link_paths, **metrics)
        if not score:
            return
        groups = _connector_violation_groups(
            link_paths,
            offenders - protected,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            anchors=anchors,
            graph=graph,
            max_group=max_group,
        )
        improved = False
        for group in groups:
            for order in (group, list(reversed(group))):
                trial = dict(link_paths)
                for link_key in order:
                    if anchors.get(link_key[0]) is None or anchors.get(link_key[1]) is None:
                        continue
                    best: list[tuple[float, float]] | None = None
                    best_score = float("inf")
                    for path in _admissible_reroutes_for_link(
                        trial,
                        link_key,
                        graph=graph,
                        anchors=anchors,
                        label_obstacles=label_obstacles,
                        positions=positions,
                        lane_xs=lane_xs,
                        lane_ys=lane_ys,
                        grid_cache=grid_cache,
                    )[:max_candidates]:
                        probe_score, _ = _connector_violation_links(
                            {**trial, link_key: path}, **metrics
                        )
                        if probe_score < best_score:
                            best_score, best = probe_score, path
                    if best is not None:
                        trial[link_key] = best
                trial_score, _ = _connector_violation_links(trial, **metrics)
                if trial_score < score:
                    for link_key in order:
                        link_paths[link_key] = trial[link_key]
                    score = trial_score
                    improved = True
                    break
        if not improved:
            return


def _resolve_connector_violations(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    merge_entry_x: dict[tuple[int, int], float],
    limit_to: set[tuple[int, int]] | None = None,
    max_iters: int = 48,
    max_candidates: int = 240,
) -> None:
    """Move violating connectors onto clear corridors until no candidate improves the layout.

    Only links already involved in a violation are rerouted, so clean sections keep the
    geometry produced by the earlier routing stages, and a reroute has to keep every
    junction its link shares with a sibling. Connectors that carry a shared bus row are
    left alone entirely: their row is agreed with the whole group that tees onto it.
    """
    from visualizer.computation_graph import _inline_frame_column_skip_links

    # A connector that already runs straight down a column is the stem its siblings tee
    # off; trading it for a detour would put a jog in the stem. That only holds while the
    # column is actually free: a straight run through a tile is not a stem worth keeping.
    # Links that bypass an inline frame run on rows the frame reserved for them.
    protected = {
        link_key
        for link_key, points in link_paths.items()
        if all(abs(x - points[0][0]) <= PARALLEL_CONNECTOR_COORD_EPS for x, _y in points)
        and not _path_hits_obstacles(
            points,
            _connector_block_obstacles(
                anchors,
                src=link_key[0],
                tgt=link_key[1],
                label_obstacles=label_obstacles,
                graph=graph,
                positions=positions,
                link_key=link_key,
            ),
            margin=CONNECTOR_OBSTACLE_MARGIN,
        )
    }
    protected.update(
        _sound_inline_frame_skip_links(link_paths, graph=graph, anchors=anchors)
    )
    if limit_to is not None:
        protected.update(set(link_paths) - limit_to)
    metrics = {
        "graph": graph,
        "anchors": anchors,
        "label_obstacles": label_obstacles,
        "positions": positions,
        "incoming": incoming,
        "outgoing": outgoing,
        "target_bus": target_bus,
        "source_bus": source_bus,
        "merge_link_bus": merge_link_bus,
    }
    lane_xs, lane_ys = _connector_lane_coordinates(
        anchors,
        label_obstacles,
        graph=graph,
        positions=positions,
    )
    # The corridor lattice a link may use depends only on the tiles it may not touch, so it
    # survives every reroute of its neighbours and is worth building once.
    grid_cache: dict = {}
    for _ in range(max_iters):
        score, offenders = _connector_violation_links(link_paths, **metrics)
        if not score:
            return
        improved = False
        for link_key in sorted(offenders - protected):
            src, tgt = link_key
            source = anchors.get(src)
            target = anchors.get(tgt)
            current = link_paths.get(link_key)
            if source is None or target is None or not current:
                continue
            baseline, _ = _connector_violation_links(link_paths, only=link_key, **metrics)
            # Two connectors drawn along one run read as a single line, so an overlap costs
            # the reader a whole edge; a reroute has to win clearly before it trades one in.
            if not baseline:
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
            admissible = _admissible_reroutes_for_link(
                link_paths,
                link_key,
                graph=graph,
                anchors=anchors,
                label_obstacles=label_obstacles,
                positions=positions,
                lane_xs=lane_xs,
                lane_ys=lane_ys,
                grid_cache=grid_cache,
            )
            overlapping = {
                second if first == link_key else first
                for first, second in _find_connector_path_overlaps(
                    link_paths,
                    incoming=incoming,
                    outgoing=outgoing,
                    target_bus=target_bus,
                    source_bus=source_bus,
                    merge_link_bus=merge_link_bus,
                    anchors=anchors,
                    graph=graph,
                )
                if link_key in (first, second)
            }
            # A junction is worth keeping while the run it sits on is sound. Where the
            # sibling holding the other side is itself violating, that run is going to have
            # to move anyway, so holding this link to it only freezes both in place.
            ignore = overlapping | (offenders - {link_key})
            junctions = _connector_sibling_junctions(
                link_paths,
                link_key,
                current,
                incoming=incoming,
                outgoing=outgoing,
                ignore=ignore,
                obstacles=obstacles,
            )
            crowded_before = sum(
                link_key in pair
                for pair in _find_crowded_parallel_connector_runs(link_paths)
            )

            def overlap_count(
                paths: dict[tuple[int, int], list[tuple[float, float]]],
            ) -> int:
                return sum(
                    link_key in pair
                    for pair in _find_connector_path_overlaps(
                        paths,
                        incoming=incoming,
                        outgoing=outgoing,
                        target_bus=target_bus,
                        source_bus=source_bus,
                        merge_link_bus=merge_link_bus,
                        anchors=anchors,
                        graph=graph,
                    )
                )

            overlaps_before = overlap_count(link_paths)
            best: list[tuple[float, float]] | None = None
            # Ranking by what the whole section scores, rather than by what this link
            # scores, is what stops a reroute that clears one crossing by handing two to
            # the neighbours from looking like the best move available.
            best_score = score
            for path in admissible[:max_candidates]:
                trial = {**link_paths, link_key: path}
                if not junctions <= _connector_sibling_junctions(
                    link_paths,
                    link_key,
                    path,
                    incoming=incoming,
                    outgoing=outgoing,
                    ignore=ignore,
                ):
                    continue
                # A reroute may inherit crowding the current path already had; it just may
                # not leave this connector running alongside more neighbours than before.
                if (
                    sum(link_key in pair for pair in _find_crowded_parallel_connector_runs(trial))
                    > crowded_before
                ):
                    continue
                # Two connectors drawn along one run read as a single line, so an overlap
                # costs the reader a whole edge and is never worth trading in.
                if overlap_count(trial) > overlaps_before:
                    continue
                count, _ = _connector_violation_links(trial, only=link_key, **metrics)
                if count >= baseline:
                    continue
                trial_score, _ = _connector_violation_links(trial, **metrics)
                if trial_score >= best_score:
                    continue
                best_score = trial_score
                best = path
                if not trial_score:
                    break
            if best is not None:
                link_paths[link_key] = best
                improved = True
                break
        if not improved:
            return


def _polish_connector_violations(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    merge_entry_x: dict[tuple[int, int], float],
    max_iters: int = 24,
    max_candidates: int = 48,
) -> None:
    """Take any reroute that leaves the whole section better than it found it.

    The main resolver holds each connector to its own tally as well as the section's, which
    keeps a link from being handed a crossing so a neighbour can shed two. That caution also
    leaves a knot standing when the way out is for one connector to take a longer way round.
    Here the weighted section score is the only judge, and only a strict improvement is kept,
    so the pass can untie those knots without being able to make the drawing worse. The first
    improvement found is taken rather than the best available, because scoring a candidate
    means scoring the whole section and the loop comes back around anyway.
    """
    metrics = {
        "graph": graph,
        "anchors": anchors,
        "label_obstacles": label_obstacles,
        "positions": positions,
        "incoming": incoming,
        "outgoing": outgoing,
        "target_bus": target_bus,
        "source_bus": source_bus,
        "merge_link_bus": merge_link_bus,
    }
    lane_xs, lane_ys = _connector_lane_coordinates(
        anchors,
        label_obstacles,
        graph=graph,
        positions=positions,
    )
    grid_cache: dict = {}
    for _ in range(max_iters):
        score, offenders = _connector_violation_links(link_paths, **metrics)
        if not score:
            return
        taken: tuple[tuple[int, int], list[tuple[float, float]]] | None = None
        for link_key in sorted(offenders):
            if not link_paths.get(link_key):
                continue
            for path in _admissible_reroutes_for_link(
                link_paths,
                link_key,
                graph=graph,
                anchors=anchors,
                label_obstacles=label_obstacles,
                positions=positions,
                lane_xs=lane_xs,
                lane_ys=lane_ys,
                grid_cache=grid_cache,
            )[:max_candidates]:
                trial_score, _ = _connector_violation_links(
                    {**link_paths, link_key: path}, **metrics
                )
                if trial_score < score:
                    taken = (link_key, path)
                    break
            if taken is not None:
                break
        if taken is None:
            return
        link_key, path = taken
        link_paths[link_key] = path
        merge_entry_x[link_key] = path[-1][0]


def _spread_route_crossing_adjust_skip_links(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    target_bus: dict[int, float],
) -> set[tuple[int, int]]:
    """Links whose spread geometry must survive vertical-crossing repair."""
    from collections import defaultdict

    skip: set[tuple[int, int]] = {
        link_key for link_key in link_paths if link_key[1] in target_bus
    }
    by_target: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for link_key in link_paths:
        by_target[link_key[1]].append(link_key)
    for tgt, links in by_target.items():
        target = anchors.get(tgt)
        if target is None:
            continue
        assignments = _same_column_bypass_assignments(
            links,
            target,
            positions=positions,
            anchors=anchors,
            target_bus=target_bus,
        )
        for link_key, bypass in assignments.items():
            if bypass.gutter_x is None:
                continue
            # A feeder that exists only to reach this tile owns its gutter route; a shared
            # producer's legs are placed as a group and stay adjustable.
            if sum(source == link_key[0] for source, _target in graph.links) == 1:
                skip.add(link_key)
    return skip


def _gutter_bypass_sum_spread_links(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    target_bus: dict[int, float],
    incoming: dict[int, list[tuple[int, int]]],
) -> set[tuple[int, int]]:
    """Sum gutter-bypass feeds whose spread rebuild clears stacked-l2norm crossings."""
    links: set[tuple[int, int]] = set()
    for tgt, link_group in incoming.items():
        target = anchors.get(tgt)
        if target is None:
            continue
        assignments = _same_column_bypass_assignments(
            link_group,
            target,
            positions=positions,
            anchors=anchors,
            target_bus=target_bus,
        )
        for link_key, bypass in assignments.items():
            if (
                link_key in link_paths
                and bypass.gutter_x is not None
                and sum(source == link_key[0] for source, _target in graph.links) > 1
            ):
                links.add(link_key)
    return links

# The overview residual merge is an ordinary tile, like the detailed-graph
# combines: both operands enter through distinct ports on its top edge and the
# sum leaves the bottom edge.
RESIDUAL_ADD_LABEL = "Add"
RESIDUAL_ADD_HEIGHT = single_line_box_height()
RESIDUAL_ADD_HALF_H = RESIDUAL_ADD_HEIGHT / 2
RESIDUAL_ADD_MIN_WIDTH = 0.66
RESIDUAL_ADD_EXIT_GAP = 0.05
# Room above the tile for the residual approach lane plus a separate lane for a
# module output that is not already centered on the tile.
RESIDUAL_ADD_ENTRY_BAND = (
    CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN + PARALLEL_CONNECTOR_CHANNEL_GAP
)


def _connector_target_top_entry_y(target: _RenderAnchor, *, gap: float = 0.04) -> float:
    """Y coordinate where a downward connector meets the target's top edge."""
    del gap  # connectors attach flush to the tile top border
    return target.top


def _connector_source_bottom_exit_y(source: _RenderAnchor, *, gap: float = 0.04) -> float:
    """Y coordinate where a connector leaves the source bottom edge."""
    del gap  # connectors attach flush to the tile bottom border
    return source.bottom


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


def _min_bus_y_clearing_vertical_connector_segments(
    x_left: float,
    x_right: float,
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    skip_link: tuple[int, int] | None,
    proposed_y: float,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> float:
    """Raise a merge bus above vertical connector runs its horizontal leg would cross."""
    lo, hi = (x_left, x_right) if x_left <= x_right else (x_right, x_left)
    cleared = proposed_y
    for link_key, points in link_paths.items():
        if link_key == skip_link:
            continue
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            if abs(x1 - x2) > 0.005:
                continue
            vx = x1
            if vx + margin < lo or vx - margin > hi:
                continue
            v_lo, v_hi = sorted((y1, y2))
            if v_lo - margin <= proposed_y <= v_hi + margin:
                cleared = max(cleared, v_hi + margin)
    return cleared


def _connector_segment_crossing_point(
    seg_a: tuple[tuple[float, float], tuple[float, float]],
    seg_b: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float] | None:
    """Return the intersection of one vertical and one horizontal connector segment."""
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


def _find_connector_segment_crossings(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[float, float]]]:
    """Return link pairs whose orthogonal segments cross in the interior."""
    crossings: list[tuple[tuple[int, int], tuple[int, int], tuple[float, float]]] = []
    items = list(link_paths.items())
    for index, (link_a, points_a) in enumerate(items):
        segments_a = list(zip(points_a, points_a[1:]))
        for link_b, points_b in items[index + 1 :]:
            segments_b = list(zip(points_b, points_b[1:]))
            for seg_a in segments_a:
                for seg_b in segments_b:
                    point = _connector_segment_crossing_point(seg_a, seg_b)
                    if point is not None:
                        crossings.append((link_a, link_b, point))
    return crossings


def _adjust_spread_routes_clearing_vertical_crossings(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    anchors: dict[int, _RenderAnchor],
    merge_entry_x: dict[tuple[int, int], float],
    merge_link_bus: dict[tuple[int, int], float],
    skip_links: set[tuple[int, int]] | None = None,
    only_links: set[tuple[int, int]] | None = None,
) -> None:
    """Raise spread merge horizontals above vertical connector runs they would cross."""
    skip_links = skip_links or set()
    only_links = only_links or set()
    crossings = _find_connector_segment_crossings(link_paths)
    if not crossings:
        return
    affected = {link for pair in crossings for link in pair[:2]}
    for link_key, entry_x in merge_entry_x.items():
        if link_key in skip_links:
            continue
        if only_links and link_key not in only_links:
            continue
        if link_key not in affected or link_key not in link_paths:
            continue
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue
        if abs(source.cx - entry_x) <= PARALLEL_CONNECTOR_COORD_EPS / 2:
            continue
        points = link_paths[link_key]
        horizontals = [
            segment
            for segment in _connector_axis_segments(points)
            if segment[0] == "h"
        ]
        if not horizontals:
            continue
        current_y = max(segment[1] for segment in horizontals)
        cleared_y = current_y
        needs_raise = False
        for segment in horizontals:
            seg_cleared = _min_bus_y_clearing_vertical_connector_segments(
                segment[2],
                segment[3],
                link_paths,
                skip_link=link_key,
                proposed_y=segment[1],
            )
            if seg_cleared > segment[1] + PARALLEL_CONNECTOR_COORD_EPS:
                needs_raise = True
            cleared_y = max(cleared_y, seg_cleared)
        if not needs_raise or cleared_y <= current_y + PARALLEL_CONNECTOR_COORD_EPS:
            continue
        entry_y = _connector_target_top_entry_y(target)
        y1 = _connector_source_bottom_exit_y(source)
        y_stub = y1 - CONNECTOR_EXIT_STUB
        min_bus = entry_y + CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN
        bus_y = merge_link_bus.get(link_key)
        if bus_y is not None:
            min_bus = max(min_bus, bus_y)
        route_y = max(min_bus, cleared_y, min(y_stub, (y1 + entry_y) / 2))
        if route_y <= y_stub + PARALLEL_CONNECTOR_COORD_EPS:
            if route_y < y_stub - PARALLEL_CONNECTOR_COORD_EPS:
                link_paths[link_key] = _ensure_orthogonal_connector_path(
                    [
                        (source.cx, y1),
                        (source.cx, route_y),
                        (entry_x, route_y),
                        (entry_x, entry_y),
                    ]
                )
            else:
                link_paths[link_key] = _ensure_orthogonal_connector_path(
                    [
                        (source.cx, y1),
                        (source.cx, y_stub),
                        (entry_x, y_stub),
                        (entry_x, entry_y),
                    ]
                )
        else:
            link_paths[link_key] = _ensure_orthogonal_connector_path(
                [
                    (source.cx, y1),
                    (source.cx, y_stub),
                    (source.cx, route_y),
                    (entry_x, route_y),
                    (entry_x, entry_y),
                ]
            )


def _resolve_perpendicular_connector_crossings(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list,
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    merge_entry_x: dict[tuple[int, int], float],
    outgoing: dict[int, list[tuple[int, int]]],
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Raise horizontal merge buses above crossing vertical connector runs."""
    cleared = dict(link_paths)
    shift_kwargs = {
        "graph": graph,
        "label_obstacles": label_obstacles,
        "positions": positions,
        "target_bus": target_bus,
        "source_bus": source_bus,
        "merge_link_bus": merge_link_bus,
        "merge_entry_x": merge_entry_x,
        "outgoing": outgoing,
    }
    for _ in range(12):
        crossings = _find_connector_segment_crossings(cleared)
        if not crossings:
            return cleared
        link_a, link_b, _point = crossings[0]
        resolved = False
        for link_key in (link_a, link_b):
            src, tgt = link_key
            source = anchors.get(src)
            target = anchors.get(tgt)
            if source is None or target is None:
                continue
            horizontals = [
                segment
                for segment in _connector_axis_segments(cleared[link_key])
                if segment[0] == "h"
            ]
            if not horizontals:
                continue
            for delta_y in (PARALLEL_CONNECTOR_CHANNEL_GAP, -PARALLEL_CONNECTOR_CHANNEL_GAP):
                candidate = _shift_path_resolving_overlap(
                    cleared[link_key],
                    src=src,
                    tgt=tgt,
                    anchors=anchors,
                    delta_y=delta_y,
                    **shift_kwargs,
                )
                if candidate == cleared[link_key]:
                    continue
                if _connector_path_has_block_edge_horizontal_jog(
                    candidate,
                    source=source,
                    target=target,
                    link_key=link_key,
                    graph=graph,
                ):
                    continue
                trial = {**cleared, link_key: candidate}
                if len(_find_connector_segment_crossings(trial)) < len(crossings):
                    cleared = trial
                    resolved = True
                    break
            if resolved:
                break
        if not resolved:
            break
    return cleared


DETAIL_TILE_ROUNDING = 0.08
DETAIL_TILE_BOX_PAD = 0.008
DETAIL_FRAME_BOX_PAD = 0.01
DETAIL_FRAME_ROUNDING = 0.12
DETAIL_FRAME_STROKE = 0.035
DETAIL_FRAME_GAP = 0.025
DETAIL_FRAME_PAD_Y_EXTRA = 0.012
INLINE_FRAME_CONNECTOR_GUTTER = 0.14
INLINE_FRAME_BYPASS_ROW_GAP = PARALLEL_CONNECTOR_CHANNEL_GAP
PIPELINE_MERGE_BUS_BELOW_FRAME_GAP = 0.14
FRAME_EXIT_LAYOUT_BELOW_GAP = 0.03
INLINE_FRAME_LABEL_CHAR_W = 6.4 * 0.0078
# Shorter captions read better overhanging their frame than broken across lines.
INLINE_FRAME_LABEL_WRAP_MIN = 15
# Advance per character as actually rendered; INLINE_FRAME_LABEL_CHAR_W runs ~1.4x narrow.
INLINE_FRAME_LABEL_RENDERED_CHAR_W = 0.075
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

    center_x = node.cx
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
    node.x = center_x - node.w / 2
    node.y = top - node.h


def _center_spine_node(node: Node, spine_cx: float) -> None:
    """Keep one spine tile centered on the stack column after label resize."""
    node.x = spine_cx - node.w / 2


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


def _draw_arrow_path(ax, points: list[tuple[float, float]]) -> None:
    """Draw a flow polyline whose last segment carries the arrowhead."""
    if len(points) < 2:
        return
    if len(points) > 2:
        _draw_path(ax, points[:-1], color=COLORS["flow"])
    (x1, y1), (x2, y2) = points[-2], points[-1]
    _arrow(ax, x1, y1, x2, y2, color=COLORS["flow"], linestyle="solid")


def _connect_from_merge(ax, merge_x: float, merge_y: float, target: Node, *, gap: float = 0.06) -> None:
    """Connect downward from a residual merge tile to the next block."""
    del gap
    _arrow(ax, merge_x, _merge_edge_below(merge_y), target.cx, target.top)


def _merge_edge_above(merge_y: float) -> float:
    """Top edge of the merge tile, where an incoming connector should land."""
    return merge_y + RESIDUAL_ADD_HALF_H


def _merge_edge_below(merge_y: float) -> float:
    """Bottom edge of the merge tile, where the outgoing connector starts."""
    return merge_y - RESIDUAL_ADD_HALF_H


def _merge_y_for_module(module_bottom: float) -> float:
    """Center the merge tile below the module box, leaving room for both entries."""
    return module_bottom - MERGE_OUTPUT_GAP - RESIDUAL_ADD_ENTRY_BAND - RESIDUAL_ADD_HALF_H


def _residual_merge_ports(spine_x: float, width: float) -> tuple[float, float]:
    """Top-edge ports of the merge tile: (module output, residual bypass).

    The module output keeps the straight spine through the tile center; the
    residual lands beside it, the same port convention the detailed graph uses
    for a bypass arriving next to a centered main feed. The offset grows with the
    tile so the two entries stay legible, but never reaches the rounded corner.
    """
    offset = min(max(TOP_ENTRY_PORT_GAP, width / 4), width / 2 - TOP_ENTRY_PORT_GAP)
    return spine_x, spine_x - offset


def _residual_branch_y(skip_from_y: float) -> float:
    """Y level for the horizontal residual bypass, above the norm tile."""
    return skip_from_y + RESIDUAL_BRANCH_LIFT


def _residual_add_node(node_id: str, spine_x: float, merge_y: float, *, width: float) -> Node:
    return _make_node(
        node_id,
        spine_x,
        _merge_edge_above(merge_y),
        max(width, RESIDUAL_ADD_MIN_WIDTH),
        RESIDUAL_ADD_HEIGHT,
        RESIDUAL_ADD_LABEL,
        COLORS["basic_op"],
        text_color=COLORS["text"],
        fontsize=8,
    )


def _residual_merge(
    ax,
    *,
    merge_id: str,
    module_cx: float,
    module_bottom: float,
    skip_from_y: float,
    spine_x: float,
    branch_x: float,
    width: float,
) -> tuple[float, Node]:
    """Add the module output to the residual skip in a plain tile.

    Returns the tile's center y (the spine level the next sublayer hangs from)
    and the tile itself.
    """
    merge_y = _merge_y_for_module(module_bottom)
    add_node = _residual_add_node(merge_id, spine_x, merge_y, width=width)
    merge_top = add_node.top
    module_port_x, residual_port_x = _residual_merge_ports(spine_x, add_node.w)
    residual_lane_y = _connector_min_bus_y_above_target(_anchor_from_node(add_node))
    module_lane_y = residual_lane_y + PARALLEL_CONNECTOR_CHANNEL_GAP
    branch_y = _residual_branch_y(skip_from_y)

    # Residual bypass: down the left gutter, then into its own top port.
    _line(ax, spine_x, skip_from_y, spine_x, branch_y, color=COLORS["flow"], linestyle="solid")
    _line(ax, spine_x, branch_y, branch_x, branch_y, color=COLORS["flow"], linestyle="solid")
    _draw_arrow_path(
        ax,
        [
            (branch_x, branch_y),
            (branch_x, residual_lane_y),
            (residual_port_x, residual_lane_y),
            (residual_port_x, merge_top),
        ],
    )

    # Module output: straight down the spine into the center port.
    if abs(module_cx - module_port_x) < 1e-6:
        points = [(module_cx, module_bottom), (module_port_x, merge_top)]
    else:
        points = [
            (module_cx, module_bottom),
            (module_cx, module_lane_y),
            (module_port_x, module_lane_y),
            (module_port_x, merge_top),
        ]
    _draw_arrow_path(ax, points)

    _draw_box(ax, add_node, edgecolor=_BASIC_OP_EDGE)
    return merge_y, add_node


def _residual_branch_x(cx: float, block_w: float, *, inset: float = 0.28) -> float:
    """X coordinate for the residual bypass, inset from the block frame's left edge."""
    return cx - block_w / 2 + inset


def _connect_block_frame_boundaries(
    ax,
    *,
    cx: float,
    frame_top: float,
    entry_top: float,
    exit_from_y: float | None,
    frame_bottom: float,
) -> None:
    """Carry the spine across the block frame's reserved header and footer.

    The inbound arrow stops at the frame edge while the first sublayer starts below the
    header, and the last row stops at its own clearance, so both ends of the block
    read as connectors that do not reach.
    """
    if entry_top < frame_top:
        _line(ax, cx, frame_top, cx, entry_top, color=COLORS["flow"])
    if exit_from_y is not None and frame_bottom < exit_from_y:
        _line(ax, cx, exit_from_y, cx, frame_bottom, color=COLORS["flow"])


def _attention_label_base(spec: ArchitectureSpec) -> str:
    """The concrete attention class name shown by the overview and its expansion."""
    labels: list[str] = []
    for variant in spec.layer_variants:
        label = variant.attention_class or variant.attention_label
        if label not in labels:
            labels.append(label)
    if labels:
        return " / ".join(labels)
    for component in _ordered_block_components(spec):
        if component.role == "attention" and component.class_name:
            return component.class_name
    return spec.attention_type


def _attention_label(spec: ArchitectureSpec) -> str:
    base = _attention_label_base(spec)
    if not spec.layer_variants and spec.attention_notes:
        return f"{base}\n{spec.attention_notes[0][:28]}"
    return base


def _ffn_class_display_name(class_name: str) -> str:
    """Use the concrete class name wherever an FFN/MoE block is referenced."""
    return class_name


def _spine_moe_class(spec: ArchitectureSpec) -> str | None:
    """MoE class the spine's FFN tile stands for when every layer builds the same one."""
    if spec.layer_variants:
        return None
    for comp in _ordered_block_components(spec):
        if comp.role == "moe" and comp.class_name:
            return comp.class_name
    return None


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
    moe_class = _spine_moe_class(spec)
    if moe_class:
        # The tile promises the section that expands it, so both carry the same name.
        return _ffn_class_display_name(moe_class), None
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
        for variant in spec.layer_variants:
            ffn = _ffn_class_display_name(variant.ffn_class or variant.ffn_label)
            attention = variant.attention_class or variant.attention_label
            bullets.append(f"{variant.count} {attention} + {ffn}")
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


def _wrap_snake_case_label(label: str, wrap_width: int) -> list[str]:
    """Break a snake_case name into lines at its underscores."""
    words = [word + "_" for word in label.split("_")]
    words[-1] = words[-1][:-1]
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + len(word) > wrap_width:
            lines.append(current)
            current = word
        else:
            current += word
    if current:
        lines.append(current)
    return lines


def _inline_frame_label_lines(label: str, frame_width: float) -> list[str]:
    """Wrap an inline-frame caption to stay inside the dotted frame width."""
    usable = max(0.35, frame_width - 0.04)
    if (
        "_" in label
        and not any(char.isspace() for char in label)
        and len(label) > INLINE_FRAME_LABEL_WRAP_MIN
    ):
        # A snake_case name has no spaces to wrap on, so its underscores are the breaks.
        snake_width = max(8, int(usable / INLINE_FRAME_LABEL_RENDERED_CHAR_W))
        if len(label) > snake_width:
            return _wrap_snake_case_label(label, snake_width)
        return [label]
    if len(label) <= 40:
        return [label]
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


def _config_inferred_stack_pre(spec: ArchitectureSpec) -> list[BlockComponent]:
    """Pre-decoder spine the config implies, for checkpoints that publish no source."""
    if spec.vocab_size is None:
        return []
    return [
        BlockComponent(
            attr_name="embed_tokens",
            class_name="Embedding",
            role="embedding",
            label="Token Embedding",
            forward_order=0,
            inferred_from_config=True,
        )
    ]


def _config_inferred_stack_tail(spec: ArchitectureSpec) -> list[BlockComponent]:
    """Post-decoder spine the config implies, for checkpoints that publish no source."""
    components: list[BlockComponent] = []
    if spec.norm_type:
        components.append(
            BlockComponent(
                attr_name="norm",
                class_name=spec.norm_type,
                role="norm",
                label=spec.norm_type,
                forward_order=0,
                inferred_from_config=True,
            )
        )
    if spec.vocab_size is not None:
        components.append(
            BlockComponent(
                attr_name="lm_head",
                class_name="Linear",
                role="head",
                label="Linear",
                forward_order=1,
                inferred_from_config=True,
            )
        )
    return components


def _stack_pre_components(spec: ArchitectureSpec) -> list[BlockComponent]:
    """Pre-decoder spine, from the modeling source when it declares one."""
    if spec.stack_pre:
        return list(spec.stack_pre)
    return _config_inferred_stack_pre(spec)


def _stack_tail_components(spec: ArchitectureSpec) -> list[BlockComponent]:
    """Post-decoder spine, from the modeling source when it declares one."""
    if spec.stack_tail:
        return list(spec.stack_tail)
    return _config_inferred_stack_tail(spec)


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
    if component.role in {"ffn", "moe"} and spec.layer_variants:
        ffn_classes: list[str] = []
        for variant in spec.layer_variants:
            cls = variant.ffn_class or variant.ffn_label
            if cls not in ffn_classes:
                ffn_classes.append(cls)
        if len(ffn_classes) > 1:
            return " / ".join(_ffn_class_display_name(cls) for cls in ffn_classes)
    if _component_has_detail_section(component, spec) and component.class_name:
        return component.class_name
    basic = _basic_component_labels(component)
    if basic is not None:
        return basic[0]
    return component.label


def _spine_sublabel(component: BlockComponent) -> str | None:
    if component.inferred_from_config:
        return "from config"
    basic = _basic_component_labels(component)
    if basic is not None:
        return basic[1]
    return None


def _spine_fill(component: BlockComponent) -> str:
    """Spine modules use the neutral module style."""
    return COLORS["basic_op"]


def _spine_text_color(component: BlockComponent) -> str:
    if component.inferred_from_config:
        return COLORS["muted"]
    return COLORS["text"]


def _spine_box_style(component: BlockComponent) -> dict[str, str]:
    """Draw kwargs that mark a tile as inferred rather than read from source."""
    if component.inferred_from_config:
        return {"linestyle": "dashed", "edgecolor": COLORS["muted"]}
    return {"edgecolor": _BASIC_OP_EDGE}


def _component_is_expanded(component: BlockComponent, spec: ArchitectureSpec) -> bool:
    """Whether a top-level module has a detailed body figure."""
    for _title, tree in architecture_section_trees(spec):
        if tree.attr_name == component.attr_name or tree.class_name == component.class_name:
            return True
        if component.role in {"attention", "ffn", "moe"} and tree.role == component.role:
            return True
    return False


def _component_has_detail_section(
    component: BlockComponent,
    spec: ArchitectureSpec,
) -> bool:
    """Whether the detailed figure actually renders a section for this component."""
    for _title, tree, _input_sublabel in _detail_sections_to_render(spec):
        if tree.attr_name == component.attr_name or tree.class_name == component.class_name:
            return True
    return False


def _top_level_module_style(
    component: BlockComponent,
    spec: ArchitectureSpec,
) -> tuple[str, str, dict[str, str]]:
    """Return fill, text, and border styles shared by top-level module tiles."""
    if _component_is_expanded(component, spec):
        return COLORS["attention"], "white", {}
    return COLORS["basic_op"], COLORS["text"], {"edgecolor": _BASIC_OP_EDGE}


def _spine_module_style(
    component: BlockComponent,
    spec: ArchitectureSpec,
) -> tuple[str, str, dict[str, str]]:
    """Style a spine tile blue when its class is expanded below the overview."""
    if _component_has_detail_section(component, spec):
        return _top_level_module_style(component, spec)
    return _spine_fill(component), _spine_text_color(component), _spine_box_style(component)


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
    module_text_color: str,
    module_box_style: dict[str, str],
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
        top_y = _merge_edge_below(spine_y) - RESIDUAL_ADD_EXIT_GAP - gap

    norm_node = _make_node(
        norm_id,
        cx,
        top_y,
        norm_w,
        norm_h,
        norm_label,
        COLORS["basic_op"],
        text_color=COLORS["text"],
        fontsize=8,
    )
    layout.add(norm_node)
    _fit_spine_node_to_label(ax, norm_node)
    _center_spine_node(norm_node, cx)
    _draw_box(ax, norm_node, edgecolor=_BASIC_OP_EDGE)
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
        text_color=module_text_color,
        sublabel=module_sublabel,
        fontsize=8.8,
    )
    layout.add(module_node)
    _fit_spine_node_to_label(ax, module_node)
    _center_spine_node(module_node, cx)
    _draw_box(ax, module_node, **module_box_style)
    _connect_down(ax, norm_node, module_node)

    merge_y, add_node = _residual_merge(
        ax,
        merge_id=f"{module_id}_add",
        module_cx=module_node.cx,
        module_bottom=module_node.bottom,
        skip_from_y=skip_from_y,
        spine_x=cx,
        branch_x=branch_x,
        width=norm_w,
    )
    layout.add(add_node)
    return merge_y


def _component_sublabel(comp: BlockComponent) -> str | None:
    if _basic_component_labels(comp) is not None:
        return None
    # The FFN/MoE tile already carries the architecture-facing name. Adding its
    # implementation class below it produced an unexplained second line (for
    # example ``MiniMaxM3VLSparseMoeBlock`` below ``SWIGLUOAI``).
    sublabel = None if comp.role in {"ffn", "moe"} else (
        comp.class_name if comp.class_name != comp.label else None
    )
    # A method-wrapper marker is extraction metadata, not useful architecture
    # text. Rendering it produced labels such as ``method `mlp()` `` beneath
    # otherwise self-explanatory tiles such as SwiGLUOAI.
    if comp.details and not comp.details[0].startswith("method `"):
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


def _connect_spine_block_side_outputs(
    ax,
    exits: Sequence[_RenderAnchor],
    *,
    cx: float,
    join_y: float,
    corridor_x: float,
) -> None:
    """Route the outputs of an expanded spine block that are not on the flow line.

    A framed block stacks its branches on a single column, so every output above the
    bottom of that stack has to leave sideways to reach the flow it merges into.
    """
    if len(exits) < 2:
        return
    downstream = exits[-1]
    for anchor in exits[:-1]:
        if abs(anchor.cx - downstream.cx) > 0.05:
            points = [(anchor.cx, anchor.bottom), (anchor.cx, join_y), (cx, join_y)]
        else:
            side_y = (anchor.top + anchor.bottom) / 2
            points = [
                (anchor.right, side_y),
                (corridor_x, side_y),
                (corridor_x, join_y),
                (cx, join_y),
            ]
        _draw_path(ax, points, color=COLORS["flow"], linewidth=1.5)


def _layout_component_chain(
    layout: DiagramLayout,
    ax,
    *,
    cx: float,
    entry_top: float,
    sequence: list[BlockComponent],
    spec: ArchitectureSpec,
    norm_w: float,
    inner_w: float,
    gap: float,
) -> float | None:
    """Stack the block's own modules when no norm pairs with a compute module.

    Role matching can miss a block's conventions entirely. Falling back to the modules
    the AST actually found keeps the block faithful to the model rather than drawing it
    empty or with generic tiles.
    """
    previous: Node | None = None
    cursor = entry_top
    for comp in sequence:
        is_norm = comp.role == "norm"
        sublabel = None if is_norm else _component_sublabel(comp)
        color, text_color, box_style = _top_level_module_style(comp, spec)
        node = _make_node(
            comp.attr_name,
            cx,
            cursor,
            norm_w if is_norm else inner_w,
            single_line_box_height() if is_norm else box_height_for_content(sublabel),
            _component_display_label(comp),
            color,
            text_color=text_color,
            sublabel=sublabel,
            fontsize=8 if is_norm else 8.8,
        )
        layout.add(node)
        _fit_spine_node_to_label(ax, node)
        _center_spine_node(node, cx)
        _draw_box(ax, node, **box_style)
        if previous is not None:
            _connect_down(ax, previous, node)
        previous = node
        cursor = node.bottom - gap
    return previous.bottom if previous is not None else None


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
        color, text_color, box_style = _top_level_module_style(comp, spec)
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
            module_text_color=text_color,
            module_box_style=box_style,
            module_sublabel=sublabel,
            norm_w=norm_w,
            inner_w=inner_w,
            gap=gap,
            entry_from_y=entry_top if spine_y is None else None,
        )
        merge_ys.append(spine_y)

    chain_bottom: float | None = None
    if not merge_ys:
        chain_bottom = _layout_component_chain(
            layout,
            ax,
            cx=cx,
            entry_top=entry_top,
            sequence=sequence,
            spec=spec,
            norm_w=norm_w,
            inner_w=inner_w,
            gap=gap,
        )

    merge_pad = RESIDUAL_ADD_HALF_H + RESIDUAL_ADD_EXIT_GAP + 0.10
    if merge_ys:
        content_bottom = min(merge_ys) - merge_pad
        exit_from_y: float | None = _merge_edge_below(min(merge_ys))
    elif chain_bottom is not None:
        content_bottom = chain_bottom - 0.10
        exit_from_y = chain_bottom
    else:
        content_bottom = top_y - 0.2
        exit_from_y = None
    frame_bottom = _block_frame_bottom(content_bottom)

    _connect_block_frame_boundaries(
        ax,
        cx=cx,
        frame_top=_block_frame_top(top_y, repeat_label),
        entry_top=entry_top,
        exit_from_y=exit_from_y,
        frame_bottom=frame_bottom,
    )
    _draw_block_frame(
        ax,
        cx=cx,
        block_w=block_w,
        bottom_y=frame_bottom,
        top_y=_block_frame_top(top_y, repeat_label),
        repeat_label=repeat_label,
        label=spec.decoder_class or "Transformer block",
    )

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
    module_color = COLORS["basic_op"]
    module_text_color = COLORS["text"]
    module_box_style = {"edgecolor": _BASIC_OP_EDGE}

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
        module_color=module_color,
        module_text_color=module_text_color,
        module_box_style=module_box_style,
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
        module_color=module_color,
        module_text_color=module_text_color,
        module_box_style=module_box_style,
        module_sublabel=ffn_sub,
        norm_w=norm_w,
        inner_w=inner_w,
        gap=gap,
    )

    merge_pad = RESIDUAL_ADD_HALF_H + RESIDUAL_ADD_EXIT_GAP + 0.10
    content_bottom = merge2 - merge_pad
    _connect_block_frame_boundaries(
        ax,
        cx=cx,
        frame_top=_block_frame_top(top_y, repeat_label),
        entry_top=entry_top,
        exit_from_y=_merge_edge_below(merge2),
        frame_bottom=_block_frame_bottom(content_bottom),
    )
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


def _inline_frame_for_top_member(graph, node_index: int):
    """Return the inline frame when node_index is its topmost member."""
    for frame in getattr(graph, "inline_frames", ()) or ():
        if frame.node_indices and frame.node_indices[0] == node_index:
            return frame
    return None


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


def _left_bypass_x_clearing_vertical_segment(
    y_lo: float,
    y_hi: float,
    obstacles: list[_RenderAnchor],
    *,
    initial_bypass_x: float,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> float:
    """Extend bypass_x leftward until a vertical segment at x clears obstacles."""
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
                new_x = obstacle.left - margin
                if new_x < bypass_x:
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


def _inline_frame_below_exit_y(
    frame_bounds,
    *,
    source_bottom: float | None = None,
) -> float:
    """Y corridor for connector horizontals below a dotted inline-frame border."""
    y = (
        frame_bounds.bottom
        - CONNECTOR_OBSTACLE_MARGIN
        - CONNECTOR_EXIT_STUB
        - FRAME_EXIT_LAYOUT_BELOW_GAP
    )
    if source_bottom is not None:
        y = min(y, source_bottom - CONNECTOR_EXIT_STUB)
    return y


def _inline_frame_outside_gutter_x(
    frame_bounds,
    source: _RenderAnchor,
    *,
    entry_x: float | None = None,
) -> float:
    """X column just outside a dotted frame for vertical drops that avoid the border."""
    left = frame_bounds.left - CONNECTOR_OBSTACLE_MARGIN - CONNECTOR_EXIT_STUB
    right = frame_bounds.right + CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_EXIT_STUB
    if entry_x is not None:
        if entry_x < source.cx - PARALLEL_CONNECTOR_COORD_EPS:
            return left
        if entry_x > source.cx + PARALLEL_CONNECTOR_COORD_EPS:
            return right
    if source.cx - frame_bounds.left <= frame_bounds.right - source.cx:
        return left
    return right


def _vertical_drop_crosses_frame_border(
    x: float,
    y_top: float,
    y_bottom: float,
    frame_bounds,
) -> bool:
    """True when a downward vertical at *x* would cut through the dotted frame border."""
    if y_bottom >= frame_bounds.bottom - PARALLEL_CONNECTOR_COORD_EPS:
        return False
    if (
        x < frame_bounds.left - PARALLEL_CONNECTOR_COORD_EPS
        or x > frame_bounds.right + PARALLEL_CONNECTOR_COORD_EPS
    ):
        return False
    return y_top > frame_bounds.bottom + PARALLEL_CONNECTOR_COORD_EPS


def _should_route_via_outside_gutter_frame_exit(
    frame_bounds,
    source: _RenderAnchor,
    *,
    entry_x: float,
    y_exit: float,
    corridor_y: float,
) -> bool:
    """Use an outside-frame gutter when a source-side vertical would cross the dashed border."""
    if not _vertical_drop_crosses_frame_border(
        source.cx,
        y_exit,
        corridor_y,
        frame_bounds,
    ):
        return False
    frame_width = frame_bounds.right - frame_bounds.left
    if frame_width <= PARALLEL_CONNECTOR_COORD_EPS:
        return False
    rel_x = (source.cx - frame_bounds.left) / frame_width
    return (
        entry_x < source.cx - PARALLEL_CONNECTOR_COORD_EPS
        and rel_x >= 0.5
    )


def _frame_tail_routing_corridor_y(
    frame_bounds,
    source: _RenderAnchor,
    target: _RenderAnchor,
) -> float:
    """Below-frame corridor Y that clears both the dotted border and the target tile."""
    corridor_y = _inline_frame_below_exit_y(
        frame_bounds,
        source_bottom=source.bottom,
    )
    # The horizontal belongs between the frame and the target. Sending it below
    # the target forces the final leg to climb back upward and produces a loop.
    return max(corridor_y, _connector_min_bus_y_above_target(target))


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
    below_frame = _inline_frame_below_exit_y(
        frame_bounds,
        source_bottom=source_bottom,
    )
    below_source = source_bottom - CONNECTOR_EXIT_STUB if source_bottom is not None else below_frame
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
    return (intermediate_anchor.top + intermediate_anchor.bottom) / 2


def _inline_frame_step_above(graph, frame, index: int) -> int | None:
    """The frame step immediately preceding one member on the frame's own chain."""
    from visualizer.computation_graph import _ordered_inline_frame_chain

    chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
    try:
        position = chain.index(index)
    except ValueError:
        return None
    return chain[position - 1] if position > 0 else None


def _crossed_tile_edges(
    positions: list,
    *,
    exclude: set[int],
    x_left: float,
    x_right: float,
) -> list[tuple[float, float]]:
    """(bottom, top) of every tile a horizontal run between two columns would cross."""
    edges = []
    for index, position in enumerate(positions):
        if index in exclude:
            continue
        left, right, bottom, top = _tile_span(position)
        if left < x_right and right > x_left:
            edges.append((bottom, top))
    return edges


def _corridor_y_below_source(
    edges: list[tuple[float, float]],
    *,
    source_bottom: float,
) -> float | None:
    """A level for a horizontal run in the row gap just below a source tile."""
    floor = max(
        (top for _, top in edges if top < source_bottom - PARALLEL_CONNECTOR_COORD_EPS),
        default=None,
    )
    if floor is None:
        return source_bottom - CONNECTOR_EXIT_STUB
    limit = floor + CONNECTOR_OBSTACLE_MARGIN
    level = source_bottom - CONNECTOR_EXIT_STUB
    if level > limit:
        return level
    # The row gap is too shallow for a full exit stub, so split what there is and
    # keep the leg long enough to still render as a corner rather than a slant.
    level = (source_bottom + limit) / 2
    if source_bottom - level <= PARALLEL_CONNECTOR_COORD_EPS:
        return None
    return level


def _corridor_y_above_target(
    edges: list[tuple[float, float]],
    *,
    target: _RenderAnchor,
) -> float | None:
    """A level for a horizontal run in the row gap just above a target tile."""
    low = _connector_min_bus_y_above_target(target)
    ceiling = min(
        (bottom for bottom, _ in edges if bottom > low + PARALLEL_CONNECTOR_COORD_EPS),
        default=None,
    )
    if ceiling is None:
        return low
    high = ceiling - CONNECTOR_OBSTACLE_MARGIN
    if high < low:
        return None
    return (low + high) / 2


def _inline_skip_top_entry_connector_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    *,
    bus_x: float | None,
    entry_x: float | None,
    frame_bounds,
    positions: list,
    exclude: set[int],
    gap: float = 0.04,
) -> list[tuple[float, float]] | None:
    """Carry an in-frame skip down a reserved gutter and into the target's top edge.

    The target is an ordinary step rather than a binary operator, so the operand has
    to arrive vertically on the tile's top border like every other incoming step.
    """
    if bus_x is None:
        bus_x = frame_bounds.left + CONNECTOR_OBSTACLE_MARGIN
    bus_x = _clamp_bus_x_to_frame_interior(bus_x, frame_bounds)
    if entry_x is None or not target.left < entry_x < target.right:
        entry_x = target.cx
    exit_y = _connector_source_bottom_exit_y(source, gap=gap)
    entry_y = _connector_target_top_entry_y(target, gap=gap)
    tee_y = _corridor_y_below_source(
        _crossed_tile_edges(
            positions,
            exclude=exclude,
            x_left=min(bus_x, source.cx),
            x_right=max(bus_x, source.cx),
        ),
        source_bottom=exit_y,
    )
    approach_y = _corridor_y_above_target(
        _crossed_tile_edges(
            positions,
            exclude=exclude,
            x_left=min(bus_x, entry_x),
            x_right=max(bus_x, entry_x),
        ),
        target=target,
    )
    if tee_y is None or approach_y is None or approach_y >= tee_y:
        return None
    return _ensure_orthogonal_connector_path(
        [
            (source.cx, exit_y),
            (source.cx, tee_y),
            (bus_x, tee_y),
            (bus_x, approach_y),
            (entry_x, approach_y),
            (entry_x, entry_y),
        ]
    )


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


def _is_frame_tail_gutter_tee_segment(
    graph,
    positions: list,
    *,
    src: int,
    index: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    bounds,
) -> bool:
    """True for the intentional outside-frame tee from a frame tail into the gutter."""
    from visualizer.computation_graph import _inline_frame_tail_indices

    if index != 1 or src not in _inline_frame_tail_indices(graph):
        return False
    if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS:
        return False
    # The tee is only outside the frame once it clears the bottom border. Higher up
    # it runs along the border instead of below it, which reads as a drawing error.
    if y1 > bounds.bottom - CONNECTOR_OBSTACLE_MARGIN:
        return False
    left_gutter = bounds.left - CONNECTOR_OBSTACLE_MARGIN - CONNECTOR_EXIT_STUB
    right_gutter = bounds.right + CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_EXIT_STUB
    if x2 < x1 - 0.06:
        return (
            abs(x2 - left_gutter) <= PARALLEL_CONNECTOR_COORD_EPS + 0.06
            or x2 < bounds.left + PARALLEL_CONNECTOR_COORD_EPS
        )
    if x2 > x1 + 0.06:
        return (
            abs(x2 - right_gutter) <= PARALLEL_CONNECTOR_COORD_EPS + 0.06
            or x2 > bounds.right - PARALLEL_CONNECTOR_COORD_EPS
        )
    return False


def _horizontal_clears_frame_border_lines(y: float, bounds, *, margin: float) -> bool:
    """True when a horizontal at ``y`` runs along neither dotted border of a frame."""
    return abs(y - bounds.top) >= margin and abs(y - bounds.bottom) >= margin


def _connector_path_violates_inline_frame_bounds(
    points: list[tuple[float, float]],
    graph,
    positions: list,
    *,
    src: int,
    tgt: int,
) -> tuple[str, str] | None:
    """Return frame id and reason when a segment cuts through a dotted frame interior."""
    frame_margin = CONNECTOR_OBSTACLE_MARGIN
    last_index = len(points) - 2
    target_is_output = (
        0 <= tgt < len(graph.nodes)
        and graph.nodes[tgt].synthetic == SYNTHETIC_OUTPUT
    )
    for frame in graph.inline_frames:
        members = set(frame.node_indices)
        if src in members and tgt in members:
            continue
        # An output edge is allowed to leave the frame that owns its terminal node.
        # Block-obstacle validation still prevents it from crossing any enclosed tiles.
        if target_is_output and src in members:
            continue
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        for index in range(len(points) - 1):
            if tgt in members and index == last_index:
                continue
            if src in members and index == 0:
                continue
            x1, y1 = points[index]
            x2, y2 = points[index + 1]
            # A feed into a deeper frame member may cross the dotted border on
            # the penultimate horizontal, then descend into the tile's top edge.
            # The frame is a visual group, not a side-entry port on the tile.
            if (
                tgt in members
                and index == last_index - 1
                and abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS
                and _horizontal_clears_frame_border_lines(
                    y1, bounds, margin=frame_margin
                )
            ):
                continue
            if _is_frame_tail_gutter_tee_segment(
                graph,
                positions,
                src=src,
                index=index,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                bounds=bounds,
            ):
                continue
            if (
                abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS
                and abs(x1 - x2) > 0.06
            ):
                # A run below the frame is outside it, but only once it clears the
                # dotted border by the obstacle margin. Any closer and the run draws
                # on top of the border instead of beside it.
                if y1 <= bounds.bottom - frame_margin:
                    continue
                if _path_horizontal_segments_overlap_bounds(
                    [(x1, y1), (x2, y2)],
                    bounds,
                    margin=frame_margin,
                ):
                    return frame.frame_id, "horizontal segment crosses dotted frame interior"
            if _segment_crosses_frame_bounds(
                x1, y1, x2, y2, bounds, margin=frame_margin
            ):
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

    # Enter a target-owned inline frame through its top edge.  Routing below the
    # frame (the generic obstacle strategy) would force the final leg back through
    # the frame interior, while placing the horizontal turn on the frame border.
    target_frames = [
        frame
        for frame in graph.inline_frames
        if tgt in set(frame.node_indices) and src not in set(frame.node_indices)
    ]
    if target_frames:
        entry_y = _connector_target_top_entry_y(target, gap=gap)
        y1 = _connector_source_bottom_exit_y(source, gap=gap)
        route_y = max(
            _inline_frame_draw_bounds(frame, positions, graph).top
            + CONNECTOR_OBSTACLE_MARGIN
            + 0.02
            for frame in target_frames
        )
        route_y = min(route_y, y1 - CONNECTOR_EXIT_STUB)
        candidates = [
            [
                (source.cx, y1),
                (source.cx, route_y),
                (target.cx, route_y),
                (target.cx, entry_y),
            ],
        ]
        left_gutter = min(
            _inline_frame_draw_bounds(frame, positions, graph).left
            - INLINE_FRAME_PAD
            - _inline_frame_connector_gutter_width(
                graph,
                frame,
                positions,
                side="left",
            )
            - CONNECTOR_OBSTACLE_MARGIN
            for frame in target_frames
        )
        upper_y = max(y1 - CONNECTOR_EXIT_STUB, route_y + CONNECTOR_EXIT_STUB)
        candidates.append(
            [
                (source.cx, y1),
                (source.cx, upper_y),
                (left_gutter, upper_y),
                (left_gutter, route_y),
                (target.cx, route_y),
                (target.cx, entry_y),
            ]
        )
        for candidate in candidates:
            candidate = _ensure_orthogonal_connector_path(candidate)
            if (
                _connector_path_clear_of_blocks(
                    candidate,
                    source=source,
                    target=target,
                    obstacles=obstacles,
                )
                and _connector_path_violates_inline_frame_bounds(
                    candidate,
                    graph,
                    positions,
                    src=src,
                    tgt=tgt,
                )
                is None
            ):
                return candidate

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
        candidate_y = bounds.bottom - CONNECTOR_OBSTACLE_MARGIN - 0.02
        bypass_y = candidate_y if bypass_y is None else min(bypass_y, candidate_y)
        left_gutter = _inline_frame_connector_gutter_width(
            graph,
            frame,
            positions,
            side="left",
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
    path_is_clear = _connector_path_clear_of_blocks(
        points,
        source=source,
        target=target,
        obstacles=obstacles,
    )
    if not path_is_clear:
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
    # Collapsing near-coincident points first: dropping one afterwards leaves the
    # dropped point's coordinate behind on its neighbour and slants that segment.
    collapsed = _dedupe_polyline_points(points, eps=eps)
    fixed: list[tuple[float, float]] = [collapsed[0]]
    for x1, y1 in collapsed[1:]:
        x0, y0 = fixed[-1]
        if abs(x0 - x1) > eps and abs(y0 - y1) > eps:
            fixed.append((x0, y1))
        elif abs(x0 - x1) > 0 and abs(x0 - x1) <= eps and abs(y0 - y1) > eps:
            # A sub-eps horizontal offset would render as a slanted segment.
            x1 = x0
        elif abs(y0 - y1) > 0 and abs(y0 - y1) <= eps and abs(x0 - x1) > eps:
            y1 = y0
        if (x1, y1) != fixed[-1]:
            fixed.append((x1, y1))
    return fixed


def _shared_merge_target_uses_center_entry(
    link_key: tuple[int, int],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    target_bus: dict[int, float],
    source_bus: dict[int, float] | None,
) -> bool:
    """True when a shared merge-bus target should enter at its tile center."""
    tgt = link_key[1]
    if tgt not in target_bus:
        return False
    if (
        source_bus is not None
        and link_key[0] in source_bus
        and abs(source.cx - target.cx) >= 0.08
    ):
        return False
    return True


def _snap_connector_path_endpoints(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    link_key: tuple[int, int],
    graph,
    merge_entry_x: dict[tuple[int, int], float] | None = None,
    target_bus: dict[int, float] | None = None,
    merge_link_bus: dict[tuple[int, int], float] | None = None,
    source_bus: dict[int, float] | None = None,
) -> list[tuple[float, float]]:
    """Keep every detail connector flush with source-bottom and target-top borders."""
    if len(points) < 2:
        return points
    snapped = list(points)
    snapped[0] = (source.cx, _connector_source_bottom_exit_y(source))
    entry_x = snapped[-1][0]
    tgt = link_key[1]
    entry_y = _connector_target_top_entry_y(target)
    target_bus = target_bus or {}
    merge_link_bus = merge_link_bus or {}
    use_center_entry = _shared_merge_target_uses_center_entry(
        link_key,
        source=source,
        target=target,
        target_bus=target_bus,
        source_bus=source_bus,
    )
    if merge_entry_x is not None and link_key in merge_entry_x and not use_center_entry:
        spread_entry_x = merge_entry_x[link_key]
        if not (
            abs(source.cx - target.cx) < 0.08
            and source.bottom >= target.top - (
                CONNECTOR_OBSTACLE_MARGIN + PARALLEL_CONNECTOR_COORD_EPS
            )
            and tgt not in target_bus
            and link_key not in merge_link_bus
            and abs(spread_entry_x - target.cx) < PARALLEL_CONNECTOR_COORD_EPS
        ):
            entry_x = spread_entry_x
            snapped = _snap_spread_top_entry_path(
                snapped,
                entry_x=entry_x,
                entry_y=entry_y,
                min_bus_y=merge_link_bus.get(link_key),
            )
    else:
        entry_x = snapped[-1][0]
        if abs(entry_x - target.cx) < PARALLEL_CONNECTOR_COORD_EPS or use_center_entry:
            entry_x = target.cx
        snapped[-1] = (entry_x, entry_y)
    final_is_horizontal = (
        len(snapped) >= 2
        and abs(snapped[-2][1] - snapped[-1][1]) <= PARALLEL_CONNECTOR_COORD_EPS
        and abs(snapped[-2][0] - snapped[-1][0]) > PARALLEL_CONNECTOR_COORD_EPS
    )
    if abs(snapped[-1][1] - entry_y) > PARALLEL_CONNECTOR_COORD_EPS or final_is_horizontal:
        snapped = _snap_spread_top_entry_path(
            snapped,
            entry_x=entry_x,
            entry_y=entry_y,
            min_bus_y=(merge_link_bus or {}).get(link_key),
        )
    result = _ensure_orthogonal_connector_path(snapped)
    if len(result) >= 2:
        prev_x, prev_y = result[-2]
        end_x, end_y = result[-1]
        if (
            abs(prev_y - end_y) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(prev_x - end_x) > PARALLEL_CONNECTOR_COORD_EPS
        ):
            if (
                len(result) >= 3
                and abs(result[-3][0] - prev_x) <= PARALLEL_CONNECTOR_COORD_EPS
            ):
                approach_y = (
                    entry_y
                    + CONNECTOR_OBSTACLE_MARGIN
                    + CONNECTOR_ATTACHED_BOX_MARGIN
                )
                result[-2] = (prev_x, approach_y)
                result.insert(-1, (end_x, approach_y))
                return _ensure_orthogonal_connector_path(result)
            return _snap_spread_top_entry_path(
                result,
                entry_x=entry_x,
                entry_y=entry_y,
                min_bus_y=(merge_link_bus or {}).get(link_key),
            )
    return result


def _snap_spread_top_entry_path(
    points: list[tuple[float, float]],
    *,
    entry_x: float,
    entry_y: float,
    min_bus_y: float | None = None,
) -> list[tuple[float, float]]:
    """Keep the final drop on a spread top-entry port column."""
    if len(points) < 2:
        return [(entry_x, entry_y)]
    snapped = list(points)
    start_x, start_y = snapped[0]
    end_x, end_y = snapped[-1]
    y_stub = start_y - CONNECTOR_EXIT_STUB
    min_bus = entry_y + CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN
    if min_bus_y is not None:
        min_bus = max(min_bus, min_bus_y)
    y_bus = max(min_bus, min(y_stub, (start_y + entry_y) / 2))

    def _spread_entry_route() -> list[tuple[float, float]]:
        if abs(entry_x - start_x) <= PARALLEL_CONNECTOR_COORD_EPS / 2:
            return [(start_x, start_y), (start_x, entry_y)]
        route_y = max(y_bus, min_bus)
        if route_y <= y_stub + PARALLEL_CONNECTOR_COORD_EPS:
            return [(start_x, start_y), (start_x, y_stub), (entry_x, y_stub), (entry_x, entry_y)]
        return [
            (start_x, start_y),
            (start_x, y_stub),
            (start_x, route_y),
            (entry_x, route_y),
            (entry_x, entry_y),
        ]

    if (
        len(snapped) == 2
        and abs(end_x - start_x) > PARALLEL_CONNECTOR_COORD_EPS / 2
        and abs(end_y - start_y) > PARALLEL_CONNECTOR_COORD_EPS / 2
    ):
        return _spread_entry_route()
    if (
        len(snapped) == 2
        and abs(end_x - start_x) <= PARALLEL_CONNECTOR_COORD_EPS / 2
        and abs(end_y - start_y) > PARALLEL_CONNECTOR_COORD_EPS / 2
    ):
        if abs(entry_x - start_x) <= PARALLEL_CONNECTOR_COORD_EPS / 2:
            return [(start_x, start_y), (entry_x, entry_y)]
        return _spread_entry_route()
    snapped[-1] = (entry_x, entry_y)
    for index in range(len(snapped) - 2, -1, -1):
        x1, y1 = snapped[index]
        x2, y2 = snapped[index + 1]
        if abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS and abs(x1 - x2) > PARALLEL_CONNECTOR_COORD_EPS:
            snapped[index + 1] = (entry_x, y2)
            break
    if len(snapped) >= 2:
        prev_x, prev_y = snapped[-2]
        if abs(prev_x - entry_x) > PARALLEL_CONNECTOR_COORD_EPS:
            if abs(prev_y - entry_y) <= PARALLEL_CONNECTOR_COORD_EPS:
                snapped[-2] = (entry_x, prev_y)
            elif abs(prev_x - start_x) <= PARALLEL_CONNECTOR_COORD_EPS:
                if abs(prev_y - entry_y) > PARALLEL_CONNECTOR_COORD_EPS:
                    if abs(entry_x - start_x) > PARALLEL_CONNECTOR_COORD_EPS:
                        snapped = _spread_entry_route()
                    else:
                        snapped = [(start_x, start_y), (entry_x, entry_y)]
                else:
                    snapped[-2] = (prev_x, entry_y)
                    snapped[-1] = (entry_x, entry_y)
    result = _ensure_orthogonal_connector_path(snapped)
    if len(result) >= 2:
        prev_x, prev_y = result[-2]
        end_x, end_y = result[-1]
        if (
            abs(prev_y - end_y) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(prev_x - end_x) > PARALLEL_CONNECTOR_COORD_EPS
        ):
            return _spread_entry_route()
    if len(result) >= 2:
        x1, y1 = result[0]
        x2, y2 = result[1]
        if (
            abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(y1 - y2) < CONNECTOR_EXIT_STUB - PARALLEL_CONNECTOR_COORD_EPS
            and abs(entry_x - start_x) > PARALLEL_CONNECTOR_COORD_EPS / 2
        ):
            return _spread_entry_route()
    return result


def _enforce_merge_link_bus_floor(
    points: list[tuple[float, float]],
    floor_y: float,
) -> list[tuple[float, float]]:
    """Keep shared merge-bus horizontals above stacked same-column feeds."""
    if len(points) < 2:
        return points
    adjusted = list(points)
    for index in range(len(adjusted) - 1):
        x1, y1 = adjusted[index]
        x2, y2 = adjusted[index + 1]
        if abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS and abs(x1 - x2) > PARALLEL_CONNECTOR_COORD_EPS:
            if y1 < floor_y - PARALLEL_CONNECTOR_COORD_EPS:
                adjusted[index] = (x1, floor_y)
                adjusted[index + 1] = (x2, floor_y)
    return _ensure_orthogonal_connector_path(adjusted)


def _finalize_inline_bypass_spine_tees(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    positions: list,
    inline_bypass_bus_x: dict[tuple[int, int], float],
    merge_entry_x: dict[tuple[int, int], float] | None = None,
) -> None:
    """Rebuild bypass connectors so they branch from the main spine at skipped tiles."""
    frames = getattr(graph, "inline_frames", None) or []
    for frame in frames:
        frame_bounds = _inline_frame_draw_bounds(frame, positions, graph)
        for link_key in _inline_frame_bypass_links(graph, frame):
            src, tgt = link_key
            source = anchors.get(src)
            target = anchors.get(tgt)
            if source is None or target is None:
                continue
            tee_y = _inline_frame_bypass_tee_y(graph, frame, src, tgt, anchors)
            if tee_y is None:
                continue
            bus_x = inline_bypass_bus_x.get(link_key)
            route = _inline_skip_top_entry_connector_points(
                source,
                target,
                bus_x=bus_x,
                entry_x=(merge_entry_x or {}).get(link_key),
                frame_bounds=frame_bounds,
                positions=positions,
                exclude={src, tgt},
            )
            if route is not None:
                link_paths[link_key] = route


def _path_crosses_attached_block_edge_band(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    margin: float = CONNECTOR_ATTACHED_BOX_MARGIN,
) -> bool:
    """True when a horizontal segment runs along a source bottom or target top edge."""
    y_exit = _connector_source_bottom_exit_y(source)
    y_entry = _connector_target_top_entry_y(target)
    edge_eps = PARALLEL_CONNECTOR_COORD_EPS / 2
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS:
            continue
        seg_left = min(x1, x2)
        seg_right = max(x1, x2)
        if abs(y1 - y_exit) <= edge_eps:
            if seg_right > source.left + margin and seg_left < source.right - margin:
                return True
        if abs(y1 - y_entry) <= edge_eps:
            if seg_right > target.left + margin and seg_left < target.right - margin:
                if _segment_is_spread_merge_top_entry_approach(
                    points,
                    index,
                    target,
                    margin=margin,
                ):
                    continue
                return True
    return False


def _connector_path_matches_spread_entry_port(
    points: list[tuple[float, float]],
    *,
    target: _RenderAnchor,
    entry_x: float,
) -> bool:
    """True when a connector ends on the assigned spread top-entry port."""
    if len(points) < 2:
        return False
    end_x, end_y = points[-1]
    return (
        abs(end_x - entry_x) <= PARALLEL_CONNECTOR_COORD_EPS
        and abs(end_y - _connector_target_top_entry_y(target))
        <= PARALLEL_CONNECTOR_COORD_EPS
    )


def _fanout_split_branch_gutter_route(
    source: _RenderAnchor,
    target: _RenderAnchor,
    entry_x: float,
    tee_y: float,
    obstacles: list[_RenderAnchor],
    *,
    gap: float = 0.04,
) -> list[tuple[float, float]]:
    """Detour around stacked tiles before dropping on a spread top-entry port."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    y2 = _connector_target_top_entry_y(target, gap=gap)
    bypass_x = max(source.right, target.right) + CONNECTOR_OBSTACLE_MARGIN
    bypass_x = _right_bypass_x_clearing_vertical_segment(
        tee_y,
        y2,
        obstacles,
        initial_bypass_x=bypass_x,
    )
    route_y = y2 + CONNECTOR_EXIT_STUB
    return _ensure_orthogonal_connector_path(
        [
            (source.cx, y1),
            (source.cx, tee_y),
            (bypass_x, tee_y),
            (bypass_x, route_y),
            (entry_x, route_y),
            (entry_x, y2),
        ]
    )


def _spread_merge_horizontal_below_target_corridor(
    points: list[tuple[float, float]],
    target: _RenderAnchor,
    *,
    gap: float = 0.04,
) -> bool:
    """True when a spread merge horizontal runs below the target's merge corridor."""
    floor_y = _connector_min_bus_y_above_target(target, gap=gap)
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if y1 < floor_y - PARALLEL_CONNECTOR_COORD_EPS:
            return True
    return False


def _spread_merge_cross_column_gutter_route(
    source: _RenderAnchor,
    target: _RenderAnchor,
    entry_x: float,
    bus_y: float,
    obstacles: list[_RenderAnchor],
    *,
    gap: float = 0.04,
) -> list[tuple[float, float]]:
    """Detour around stacked tiles before entering a spread top-entry port."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    y2 = _connector_target_top_entry_y(target, gap=gap)
    y_stub = y1 - CONNECTOR_EXIT_STUB
    bypass_x = max(source.right, target.right) + CONNECTOR_OBSTACLE_MARGIN
    bypass_x = _right_bypass_x_clearing_horizontal_segment(
        source.cx,
        y_stub,
        obstacles,
        initial_bypass_x=bypass_x,
    )
    approach_y = _min_bus_y_clearing_horizontal_corridor(
        min(bypass_x, entry_x),
        max(bypass_x, entry_x),
        obstacles,
        proposed_y=_connector_min_bus_y_above_target(target, gap=gap),
    )
    bypass_x = _right_bypass_x_clearing_vertical_segment(
        min(y_stub, approach_y),
        max(y_stub, approach_y),
        obstacles,
        initial_bypass_x=bypass_x,
    )
    if approach_y <= y_stub + PARALLEL_CONNECTOR_COORD_EPS:
        # The drop already clears the stub level, so the gutter column would only
        # overshoot right of the source and double back along the same corridor.
        return _ensure_orthogonal_connector_path(
            [
                (source.cx, y1),
                (source.cx, approach_y),
                (entry_x, approach_y),
                (entry_x, y2),
            ]
        )
    return _ensure_orthogonal_connector_path(
        [
            (source.cx, y1),
            (source.cx, y_stub),
            (bypass_x, y_stub),
            (bypass_x, approach_y),
            (entry_x, approach_y),
            (entry_x, y2),
        ]
    )


def _fanout_leg_routing_bus_y(
    link_key: tuple[int, int],
    *,
    graph,
    outgoing: dict[int, list[tuple[int, int]]] | None,
    target_bus: dict[int, float] | None,
    merge_link_bus: dict[tuple[int, int], float] | None,
    tee_y: float,
    anchors: dict[int, _RenderAnchor] | None = None,
) -> float:
    """Per-leg bus level for fan-out routing; split branches skip lower merge buses."""
    _src, tgt = link_key
    if (
        graph is not None
        and outgoing is not None
        and target_bus is not None
        and _source_fanout_splits_before_target_bus(graph, _src, outgoing, target_bus)
        and tgt not in target_bus
    ):
        return tee_y
    if merge_link_bus is not None and link_key in merge_link_bus:
        bus = merge_link_bus[link_key]
        if (
            tee_y is not None
            and bus < tee_y - PARALLEL_CONNECTOR_COORD_EPS
            and anchors is not None
        ):
            target = anchors.get(tgt)
            if target is not None:
                return _connector_min_bus_y_above_target(target)
        return bus
    return tee_y


def _repair_connector_target_top_edge_overlap(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    entry_x: float | None = None,
) -> list[tuple[float, float]]:
    """Replace a final horizontal jog along the target top with a bus-above entry drop."""
    if len(points) < 2:
        return points
    y_entry = _connector_target_top_entry_y(target)
    port_x = entry_x if entry_x is not None else points[-1][0]
    if not _path_crosses_attached_block_edge_band(
        points,
        source=source,
        target=target,
    ):
        return points
    for index in range(len(points) - 2, -1, -1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if abs(y1 - y_entry) > PARALLEL_CONNECTOR_COORD_EPS / 2:
            continue
        if abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS:
            continue
        seg_left = min(x1, x2)
        seg_right = max(x1, x2)
        if not (seg_right > target.left and seg_left < target.right):
            continue
        prefix = list(points[: index + 1])
        anchor_x, anchor_y = prefix[-1]
        y_bus = max(
            _spread_top_entry_bus_y(source, target),
            anchor_y if abs(anchor_y - y_entry) > PARALLEL_CONNECTOR_COORD_EPS else _spread_top_entry_bus_y(source, target),
        )
        if index > 0 and abs(anchor_y - y_entry) <= PARALLEL_CONNECTOR_COORD_EPS / 2:
            prev_x, prev_y = prefix[-2]
            if abs(prev_x - anchor_x) <= PARALLEL_CONNECTOR_COORD_EPS:
                y_bus = max(y_bus, prev_y)
                prefix = prefix[:-1]
                anchor_x = prev_x
        repaired = [
            *prefix,
            (anchor_x, y_bus),
            (port_x, y_bus),
            (port_x, y_entry),
        ]
        return _ensure_orthogonal_connector_path(repaired)
    return points


def _coerce_connector_path_for_link(
    points: list[tuple[float, float]],
    *,
    link_key: tuple[int, int],
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    merge_entry_x: dict[tuple[int, int], float] | None = None,
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
) -> list[tuple[float, float]]:
    """Validate a connector path and reroute or repair it when it crosses block edges."""
    src, tgt = link_key
    source = anchors.get(src)
    target = anchors.get(tgt)
    if source is None or target is None or len(points) < 2:
        return points
    obstacles = _connector_block_obstacles(
        anchors,
        src=src,
        tgt=tgt,
        label_obstacles=label_obstacles,
        graph=graph,
        positions=positions,
        link_key=link_key,
    )
    if abs(source.cx - target.cx) < 0.08:
        straight = _same_column_straight_connector_points(source, target)
        if _connector_path_clear_of_blocks(
            straight,
            source=source,
            target=target,
            obstacles=obstacles,
        ):
            return straight
    if _connector_path_clear_of_blocks(
        points,
        source=source,
        target=target,
        obstacles=obstacles,
    ):
        return points
    spread_entry_x = (
        merge_entry_x.get(link_key, target.cx)
        if merge_entry_x is not None
        else target.cx
    )
    repaired = _repair_connector_target_top_edge_overlap(
        points,
        source=source,
        target=target,
        entry_x=spread_entry_x,
    )
    if _connector_path_clear_of_blocks(
        repaired,
        source=source,
        target=target,
        obstacles=obstacles,
    ):
        return repaired
    bus_y = _link_routing_bus_y(
        link_key,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
    )
    rerouted = _reroute_connector_path_clearing_blocks(
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
        merge_entry_x=merge_entry_x,
        outgoing=outgoing,
        target_bus=target_bus,
    )
    if _connector_path_clear_of_blocks(
        rerouted,
        source=source,
        target=target,
        obstacles=obstacles,
    ):
        return rerouted
    repaired = _repair_connector_target_top_edge_overlap(
        rerouted,
        source=source,
        target=target,
        entry_x=spread_entry_x,
    )
    if _connector_path_clear_of_blocks(
        repaired,
        source=source,
        target=target,
        obstacles=obstacles,
    ):
        return repaired
    return rerouted


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
    if _path_crosses_attached_block_edge_band(
        points,
        source=source,
        target=target,
    ):
        return False
    if _path_penetrates_obstacle_tiles(points, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN):
        return False
    return not _path_penetrates_attached_boxes(points, source, target)


def _inline_frame_caption_band_top(frame, frame_bounds) -> float:
    """Top of the band a captioned frame's label occupies above its border."""
    if not (getattr(frame, "label", "") or "").strip():
        return frame_bounds.top
    lines = 1 + len(
        [line for line in (getattr(frame, "sublabel", "") or "").split("\n") if line.strip()]
    )
    return frame_bounds.top + INLINE_FRAME_LABEL_GAP + lines * INLINE_FRAME_LABEL_LINE_H


def _inline_frame_top_member_route_y(
    source: _RenderAnchor,
    target: _RenderAnchor,
    frame,
    positions: list,
    graph,
    *,
    gap: float = 0.04,
) -> float:
    """Pick a horizontal corridor Y that clears the frame envelope interior."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    route_y = y1 - CONNECTOR_EXIT_STUB
    frame_bounds = _inline_frame_draw_bounds(frame, positions, graph)
    if _path_horizontal_segments_overlap_bounds(
        [
            (min(source.cx, target.cx), route_y),
            (max(source.cx, target.cx), route_y),
        ],
        frame_bounds,
        margin=CONNECTOR_OBSTACLE_MARGIN,
    ):
        route_y = max(route_y, frame_bounds.top + CONNECTOR_OBSTACLE_MARGIN)
    # The frame's caption sits in the band right above its border, so a corridor
    # crossing the frame's width has to run above the caption as well.
    above_caption = (
        _inline_frame_caption_band_top(frame, frame_bounds) + CONNECTOR_OBSTACLE_MARGIN
    )
    if route_y < above_caption < y1:
        route_y = above_caption
    return route_y


def _path_enters_target_top_center(
    points: list[tuple[float, float]],
    target: _RenderAnchor,
    *,
    gap: float = 0.04,
) -> bool:
    """True when a route already docks on the middle of the target's top edge."""
    if not points:
        return False
    end_x, end_y = points[-1]
    return (
        abs(end_x - target.cx) <= TOP_ENTRY_PORT_GAP
        and abs(end_y - _connector_target_top_entry_y(target, gap=gap))
        <= CONNECTOR_OBSTACLE_MARGIN
    )


def _same_column_straight_inline_top_feed(
    points: list[tuple[float, float]],
    source: _RenderAnchor,
    target: _RenderAnchor,
    *,
    graph,
    positions: list,
    src: int,
    tgt: int,
    obstacles: list[_RenderAnchor],
) -> bool:
    """True when a same-column vertical feed already enters an inline frame head."""
    target_frame = next(
        (frame for frame in graph.inline_frames if tgt in frame.node_indices),
        None,
    )
    if target_frame is None or src in target_frame.node_indices:
        return False
    top_member = max(
        target_frame.node_indices,
        key=lambda index: positions[index].top_y,
    )
    if tgt != top_member:
        return False
    if abs(source.cx - target.cx) >= 0.08:
        return False
    if len(points) != 2:
        return False
    if not _path_enters_target_top_center(points, target):
        return False
    return not _path_hits_obstacles(
        points,
        obstacles,
        margin=CONNECTOR_OBSTACLE_MARGIN,
    )


def _outside_to_inline_frame_top_member_route(
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    graph,
    positions: list,
    *,
    src: int,
    tgt: int,
    entry_x: float | None = None,
    source_tee_y: float | None = None,
    gap: float = 0.04,
) -> list[tuple[float, float]] | None:
    """Route a feed from outside a dotted frame into its top tile, above the frame."""
    target_frame = next(
        (frame for frame in graph.inline_frames if tgt in frame.node_indices),
        None,
    )
    if target_frame is None or not target_frame.node_indices:
        return None
    if src in target_frame.node_indices:
        return None
    top_member = max(
        target_frame.node_indices,
        key=lambda index: positions[index].top_y,
    )
    if tgt != top_member:
        return None
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    y2 = _connector_target_top_entry_y(target, gap=gap)
    directional_entry_x = (
        target.right - TOP_ENTRY_PORT_GAP
        if source.cx >= target.cx
        else target.left + TOP_ENTRY_PORT_GAP
    )
    preferred_entry_x = (
        entry_x
        if entry_x is not None and target.left < entry_x < target.right
        else directional_entry_x
    )
    alternate_entry_x = (
        target.left + TOP_ENTRY_PORT_GAP
        if preferred_entry_x >= target.cx
        else target.right - TOP_ENTRY_PORT_GAP
    )
    entry_x = preferred_entry_x
    route_y = _inline_frame_top_member_route_y(
        source,
        target,
        target_frame,
        positions,
        graph,
        gap=gap,
    )
    span_left, span_right = sorted((source.cx, entry_x))
    for obstacle in obstacles:
        if obstacle.right < span_left or obstacle.left > span_right:
            continue
        if obstacle.bottom >= y1 - PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if obstacle.top <= y2 + PARALLEL_CONNECTOR_COORD_EPS:
            continue
        route_y = min(
            route_y,
            obstacle.bottom
            - CONNECTOR_OBSTACLE_MARGIN
            - 2 * PARALLEL_CONNECTOR_COORD_EPS,
        )
    for candidate in (preferred_entry_x, alternate_entry_x):
        if not _path_hits_obstacles(
            [(candidate, route_y), (candidate, y2)],
            obstacles,
            margin=CONNECTOR_OBSTACLE_MARGIN,
        ):
            entry_x = candidate
            break
    span_left, span_right = sorted((entry_x, max(source.cx, target.right)))
    for frame in graph.inline_frames:
        if frame is target_frame:
            continue
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        if bounds.right < span_left or bounds.left > span_right:
            continue
        route_y = max(
            route_y,
            bounds.top
            + CONNECTOR_OBSTACLE_MARGIN
            + 2 * PARALLEL_CONNECTOR_COORD_EPS,
        )
    if source_tee_y is not None:
        tee_direct = _ensure_orthogonal_connector_path(
            [
                (source.cx, y1),
                (source.cx, source_tee_y),
                (entry_x, source_tee_y),
                (entry_x, y2),
            ]
        )
        if (
            not _path_hits_obstacles(
                tee_direct,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            )
            and not _path_penetrates_attached_boxes(tee_direct, source, target)
        ):
            return tee_direct
    source_stub_y = (
        source_tee_y
        if source_tee_y is not None
        else y1 - CONNECTOR_EXIT_STUB - 2 * PARALLEL_CONNECTOR_COORD_EPS
    )
    source_gutter = _ensure_orthogonal_connector_path(
        [
            (source.cx, y1),
            (source.cx, source_stub_y),
            (source.cx, route_y),
            (entry_x, route_y),
            (entry_x, y2),
        ]
    )
    if (
        route_y <= source_stub_y + PARALLEL_CONNECTOR_COORD_EPS
        and
        not _path_hits_obstacles(
            source_gutter,
            obstacles,
            margin=CONNECTOR_OBSTACLE_MARGIN,
        )
        and not _path_penetrates_attached_boxes(source_gutter, source, target)
    ):
        return source_gutter
    direct = _ensure_orthogonal_connector_path(
        [
            (source.cx, y1),
            (source.cx, route_y),
            (entry_x, route_y),
            (entry_x, y2),
        ]
    )
    if (
        route_y <= y1 - CONNECTOR_EXIT_STUB
        and not _path_hits_obstacles(
            direct,
            obstacles,
            margin=CONNECTOR_OBSTACLE_MARGIN,
        )
        and not _path_penetrates_attached_boxes(direct, source, target)
    ):
        return direct
    stub_y = (
        source_tee_y
        if source_tee_y is not None
        else y1 - CONNECTOR_EXIT_STUB - 2 * PARALLEL_CONNECTOR_COORD_EPS
    )
    if target.cx < source.cx:
        bypass_x = min(source.left, target.left) - CONNECTOR_OBSTACLE_MARGIN
        bypass_x = _left_bypass_x_clearing_horizontal_segment(
            max(source.cx, target.cx),
            stub_y,
            obstacles,
            initial_bypass_x=bypass_x,
        )
        bypass_x = _left_bypass_x_clearing_vertical_segment(
            stub_y,
            route_y,
            obstacles,
            initial_bypass_x=bypass_x,
        )
    else:
        bypass_x = max(source.right, target.right) + CONNECTOR_OBSTACLE_MARGIN
        bypass_x = _right_bypass_x_clearing_horizontal_segment(
            min(source.cx, target.cx),
            stub_y,
            obstacles,
            initial_bypass_x=bypass_x,
        )
        bypass_x = _right_bypass_x_clearing_vertical_segment(
            stub_y,
            route_y,
            obstacles,
            initial_bypass_x=bypass_x,
        )
    span_left, span_right = sorted((bypass_x, entry_x))
    for frame in graph.inline_frames:
        if frame is target_frame:
            continue
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        if bounds.right < span_left or bounds.left > span_right:
            continue
        route_y = max(
            route_y,
            bounds.top
            + CONNECTOR_OBSTACLE_MARGIN
            + 2 * PARALLEL_CONNECTOR_COORD_EPS,
        )
    frame_bounds = _inline_frame_draw_bounds(target_frame, positions, graph)
    if _path_horizontal_segments_overlap_bounds(
        [(min(source.cx, bypass_x), stub_y), (max(source.cx, bypass_x), stub_y)],
        frame_bounds,
        margin=CONNECTOR_OBSTACLE_MARGIN,
    ):
        stub_y = min(
            stub_y,
            frame_bounds.bottom
            - CONNECTOR_OBSTACLE_MARGIN
            - 2 * PARALLEL_CONNECTOR_COORD_EPS,
        )
    span_left, span_right = sorted((source.cx, bypass_x))
    changed = True
    while changed:
        changed = False
        for obstacle in obstacles:
            if obstacle.right < span_left or obstacle.left > span_right:
                continue
            if (
                obstacle.bottom - CONNECTOR_OBSTACLE_MARGIN
                <= stub_y
                <= obstacle.top + CONNECTOR_OBSTACLE_MARGIN
            ):
                stub_y = (
                    obstacle.bottom
                    - CONNECTOR_OBSTACLE_MARGIN
                    - 2 * PARALLEL_CONNECTOR_COORD_EPS
                )
                changed = True
    route_y = min(route_y, y1 - CONNECTOR_EXIT_STUB)
    if route_y > stub_y + PARALLEL_CONNECTOR_COORD_EPS:
        # The fallback gutter must still descend monotonically. A lower source
        # stub followed by a higher frame-entry corridor creates a visible loop.
        stub_y = route_y
    return _ensure_orthogonal_connector_path(
        [
            (source.cx, y1),
            (source.cx, stub_y),
            (bypass_x, stub_y),
            (bypass_x, route_y),
            (entry_x, route_y),
            (entry_x, y2),
        ]
    )


def _outside_to_inline_frame_inner_member_route(
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    graph,
    positions: list,
    *,
    src: int,
    tgt: int,
    entry_x: float | None = None,
    gap: float = 0.04,
) -> list[tuple[float, float]] | None:
    """Route a feed from outside a dotted frame into a tile below the frame's head.

    The corridor is the row gap directly above the target, so the run crosses the
    dotted border once and docks on the target's own top edge. Dropping to the
    frame's head instead would attach the feed to the wrong step.
    """
    target_frame = next(
        (frame for frame in graph.inline_frames if tgt in frame.node_indices),
        None,
    )
    if target_frame is None or not target_frame.node_indices:
        return None
    if src in target_frame.node_indices:
        return None
    top_member = max(
        target_frame.node_indices,
        key=lambda index: positions[index].top_y,
    )
    if tgt == top_member:
        return None
    frame_bounds = _inline_frame_draw_bounds(target_frame, positions, graph)
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    y2 = _connector_target_top_entry_y(target, gap=gap)
    # The corridor has to sit inside the row gap: clear of the step above and
    # still above the target's entry, or the run draws over a tile.
    route_y = target.top + CONNECTOR_OBSTACLE_MARGIN + PARALLEL_CONNECTOR_COORD_EPS
    step_above = _inline_frame_step_above(graph, target_frame, tgt)
    if step_above is not None:
        ceiling = positions[step_above].bottom - CONNECTOR_OBSTACLE_MARGIN
        if ceiling < route_y:
            return None
    if route_y > y1 - CONNECTOR_EXIT_STUB or route_y <= y2:
        return None
    if not _horizontal_clears_frame_border_lines(
        route_y,
        frame_bounds,
        margin=CONNECTOR_OBSTACLE_MARGIN,
    ):
        return None
    directional_entry_x = (
        target.right - TOP_ENTRY_PORT_GAP
        if source.cx >= target.cx
        else target.left + TOP_ENTRY_PORT_GAP
    )
    preferred_entry_x = (
        entry_x
        if entry_x is not None and target.left < entry_x < target.right
        else directional_entry_x
    )
    alternate_entry_x = (
        target.left + TOP_ENTRY_PORT_GAP
        if preferred_entry_x >= target.cx
        else target.right - TOP_ENTRY_PORT_GAP
    )
    # A drop beside the frame keeps the vertical off the dotted border; the source
    # column itself works whenever it already stands clear of the frame.
    drop_columns: list[float] = []
    if not _vertical_drop_crosses_frame_border(source.cx, y1, route_y, frame_bounds):
        drop_columns.append(source.cx)
    near_gutter = (
        frame_bounds.right + CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_EXIT_STUB
        if source.cx >= frame_bounds.right
        else frame_bounds.left - CONNECTOR_OBSTACLE_MARGIN - CONNECTOR_EXIT_STUB
    )
    drop_columns.append(near_gutter)
    drop_columns.append(
        _inline_frame_outside_gutter_x(frame_bounds, source, entry_x=preferred_entry_x)
    )
    stub_y = y1 - CONNECTOR_EXIT_STUB - 2 * PARALLEL_CONNECTOR_COORD_EPS
    for candidate_entry_x in (preferred_entry_x, alternate_entry_x):
        for drop_x in drop_columns:
            points = (
                [(source.cx, y1), (source.cx, route_y)]
                if abs(drop_x - source.cx) <= PARALLEL_CONNECTOR_COORD_EPS
                else [
                    (source.cx, y1),
                    (source.cx, stub_y),
                    (drop_x, stub_y),
                    (drop_x, route_y),
                ]
            )
            points.extend([(candidate_entry_x, route_y), (candidate_entry_x, y2)])
            candidate = _ensure_orthogonal_connector_path(points)
            if _path_hits_obstacles(
                candidate,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            ):
                continue
            if _path_penetrates_attached_boxes(candidate, source, target):
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


def _backward_top_entry_gutter_route(
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    entry_x: float | None = None,
    channel: int = 0,
) -> list[tuple[float, float]] | None:
    """Route a source below its target through an exterior gutter into the target top."""
    if source.bottom > target.top + PARALLEL_CONNECTOR_COORD_EPS:
        return None
    route_obstacles = [
        obstacle
        for obstacle in obstacles
        if not (
            abs(obstacle.left - source.left) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(obstacle.right - source.right) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(obstacle.top - source.top) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(obstacle.bottom - source.bottom) <= PARALLEL_CONNECTOR_COORD_EPS
        )
        and not (
            abs(obstacle.left - target.left) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(obstacle.right - target.right) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(obstacle.top - target.top) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(obstacle.bottom - target.bottom) <= PARALLEL_CONNECTOR_COORD_EPS
        )
    ]
    entry_x = target.cx if entry_x is None else entry_x
    y1 = _connector_source_bottom_exit_y(source)
    stub_y = y1 - CONNECTOR_EXIT_STUB - 2 * PARALLEL_CONNECTOR_COORD_EPS
    bus_y = (
        target.top
        + CONNECTOR_OBSTACLE_MARGIN
        + 2 * PARALLEL_CONNECTOR_COORD_EPS
    )
    bus_y += channel * PARALLEL_CONNECTOR_CHANNEL_GAP
    left_x = min([source.left, target.left, *(obstacle.left for obstacle in route_obstacles)])
    right_x = max([source.right, target.right, *(obstacle.right for obstacle in route_obstacles)])
    clearance = (
        CONNECTOR_OBSTACLE_MARGIN
        + CONNECTOR_EXIT_STUB
        + channel * PARALLEL_CONNECTOR_CHANNEL_GAP
    )
    candidates = (left_x - clearance, right_x + clearance)
    candidates = tuple(sorted(candidates, key=lambda x: abs(x - source.cx)))
    for gutter_x in candidates:
        candidate_stub_y = stub_y
        span_left, span_right = sorted((source.cx, gutter_x))
        changed = True
        while changed:
            changed = False
            for obstacle in route_obstacles:
                if obstacle.right < span_left or obstacle.left > span_right:
                    continue
                if (
                    obstacle.bottom - CONNECTOR_OBSTACLE_MARGIN
                    <= candidate_stub_y
                    <= obstacle.top + CONNECTOR_OBSTACLE_MARGIN
                ):
                    if obstacle.top < y1:
                        candidate_stub_y = (
                            y1 + obstacle.top + CONNECTOR_OBSTACLE_MARGIN
                        ) / 2
                    else:
                        candidate_stub_y = (
                            obstacle.bottom
                            - CONNECTOR_OBSTACLE_MARGIN
                            - 2 * PARALLEL_CONNECTOR_COORD_EPS
                        )
                    changed = True
        path = _ensure_orthogonal_connector_path(
            [
                (source.cx, y1),
                (source.cx, candidate_stub_y),
                (gutter_x, candidate_stub_y),
                (gutter_x, bus_y),
                (entry_x, bus_y),
                (entry_x, target.top),
            ]
        )
        hits = _path_hits_obstacles(
            path,
            route_obstacles,
            margin=CONNECTOR_OBSTACLE_MARGIN,
        )
        attached = _path_penetrates_attached_boxes(path, source, target)
        if not hits and not attached:
            return path
    return None


def _lanes_flanking_blocked_drop(
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    drop_top: float,
    drop_bottom: float,
    gap: float,
) -> list[float]:
    """Lanes just past each tile that stands in the target's own drop column."""
    clearance = gap + CONNECTOR_OBSTACLE_MARGIN
    lanes: list[float] = []
    for obstacle in obstacles:
        if not (
            obstacle.left - CONNECTOR_OBSTACLE_MARGIN
            <= target.cx
            <= obstacle.right + CONNECTOR_OBSTACLE_MARGIN
        ):
            continue
        if not _ranges_overlap(drop_bottom, drop_top, obstacle.bottom, obstacle.top):
            continue
        lanes.extend((obstacle.left - clearance, obstacle.right + clearance))
    lanes.sort(key=lambda lane: abs(lane - target.cx))
    return lanes


def _horizontal_departure_side_bypass_route(
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    bus_y: float | None = None,
    tee_y: float | None = None,
    gap: float = 0.04,
) -> list[tuple[float, float]] | None:
    """Leave the source column horizontally before dropping to the target."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    y_stub = y1 - CONNECTOR_EXIT_STUB
    entry_y = _connector_target_top_entry_y(target, gap=gap)
    side_candidates = (
        max(source.right, target.right) + gap + 0.10,
        # A lane hugging whatever stands over the target column beats sweeping
        # around the whole group when the wide detour is the only alternative.
        *_lanes_flanking_blocked_drop(
            target,
            obstacles,
            drop_top=y1,
            drop_bottom=entry_y,
            gap=gap,
        ),
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
    departure_levels = [y_stub]
    if tee_y is not None and y1 > tee_y > entry_y + PARALLEL_CONNECTOR_COORD_EPS:
        # Leaving on the fan-out tee keeps this leg's departure with its siblings.
        departure_levels.insert(0, tee_y)
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
                prefix = [
                    (source.cx, y1),
                    (source.cx, depart_y),
                    (bypass_x, depart_y),
                ]
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


def _is_frame_tail_below_border_path(
    graph,
    positions: list,
    *,
    src: int,
    points: list[tuple[float, float]],
) -> bool:
    """True when a connector drops vertically before using a below-frame corridor."""
    from visualizer.computation_graph import _inline_frame_tail_indices

    if src not in _inline_frame_tail_indices(graph) or len(points) < 3:
        return False
    frame = _frame_for_tail_node(graph, src)
    if frame is None:
        return False
    x0, y0 = points[0]
    x1, y1 = points[1]
    if abs(x0 - x1) > PARALLEL_CONNECTOR_COORD_EPS or y1 >= y0 - PARALLEL_CONNECTOR_COORD_EPS:
        return False
    bounds = _inline_frame_draw_bounds(frame, positions, graph)
    corridor_y = _inline_frame_below_exit_y(
        bounds,
        source_bottom=positions[src].bottom,
    )
    return any(y <= corridor_y + PARALLEL_CONNECTOR_COORD_EPS for _x, y in points[1:])


def _reroute_connector_path_clearing_blocks(
    points: list[tuple[float, float]],
    **kwargs,
) -> list[tuple[float, float]]:
    """Reroute a connector clear of blocks, guaranteeing the result renders square.

    The detours below pick columns and rows independently, so two of them can land within
    the tolerance that counts as one column without being equal, which draws as a slant.
    """
    return _ensure_orthogonal_connector_path(
        _detour_connector_path_clearing_blocks(points, **kwargs)
    )


def _detour_connector_path_clearing_blocks(
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
    merge_entry_x: dict[tuple[int, int], float] | None = None,
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
    target_bus: dict[int, float] | None = None,
) -> list[tuple[float, float]]:
    """Last-resort reroute when a laid-out connector still crosses a block."""
    if (
        graph is not None
        and positions is not None
        and link_key is not None
        and _is_frame_tail_below_border_path(
            graph,
            positions,
            src=link_key[0],
            points=points,
        )
        and not _path_penetrates_obstacle_tiles(
            points,
            obstacles,
            margin=CONNECTOR_OBSTACLE_MARGIN,
        )
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
    if (
        merge_entry_x is not None
        and link_key is not None
        and link_key in merge_entry_x
        and abs(source.cx - target.cx) < 0.08
    ):
        candidates.append(
            _same_column_spread_top_entry_connector_points(
                source,
                target,
                merge_entry_x[link_key],
            )
        )
    elif (
        merge_entry_x is not None
        and link_key is not None
        and link_key in merge_entry_x
        and merge_link_bus is not None
        and link_key in merge_link_bus
        and abs(source.cx - target.cx) >= 0.08
        and source.bottom <= target.top + 0.5
    ):
        candidates.insert(
            0,
            _spread_merge_cross_column_gutter_route(
                source,
                target,
                merge_entry_x[link_key],
                merge_link_bus[link_key],
                obstacles,
            ),
        )
    if graph is not None and positions is not None and link_key is not None:
        use_shared_input_fanout = (
            source_bus is not None
            and link_key[0] in source_bus
            and len(
                _fanout_links_excluding_bypasses(
                    graph,
                    [(src, tgt) for src, tgt in graph.links if src == link_key[0]],
                )
            )
            >= SHARED_SOURCE_BUS_MIN_LINKS
        )
        if use_shared_input_fanout:
            target_frame = next(
                (
                    frame
                    for frame in graph.inline_frames
                    if link_key[1] in frame.node_indices
                ),
                None,
            )
            if target_frame is not None:
                tee_y = source_bus[link_key[0]]
                frame_bounds = _inline_frame_draw_bounds(target_frame, positions, graph)
                if (
                    not _requires_shared_input_source_bus(graph, link_key[1])
                    and tee_y < frame_bounds.top - PARALLEL_CONNECTOR_COORD_EPS
                    and _path_horizontal_segments_overlap_bounds(
                        [
                            (min(source.cx, target.cx), tee_y),
                            (max(source.cx, target.cx), tee_y),
                        ],
                        frame_bounds,
                        margin=CONNECTOR_OBSTACLE_MARGIN,
                    )
                ):
                    use_shared_input_fanout = False
        if not use_shared_input_fanout:
            top_member_route = _outside_to_inline_frame_top_member_route(
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
            inner_member_route = _outside_to_inline_frame_inner_member_route(
                source,
                target,
                obstacles,
                graph,
                positions,
                src=link_key[0],
                tgt=link_key[1],
            )
            if inner_member_route is not None:
                candidates.append(inner_member_route)
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
        spread_entry_x = (
            merge_entry_x.get(link_key, target.cx)
            if merge_entry_x is not None and link_key in merge_entry_x
            else target.cx
        )
        if (
            tee_y is not None
            and leg_bus is not None
            and leg_bus < tee_y - PARALLEL_CONNECTOR_COORD_EPS
            and not (
                graph is not None
                and outgoing is not None
                and target_bus is not None
                and _source_fanout_splits_before_target_bus(
                    graph,
                    link_key[0],
                    outgoing,
                    target_bus,
                )
                and link_key[1] not in target_bus
            )
        ):
            candidates.insert(
                0,
                _fanout_tee_then_entry_column_points(
                    source,
                    target,
                    spread_entry_x,
                    tee_y=tee_y,
                    bus_y=leg_bus,
                ),
            )
        if (
            tee_y is not None
            and graph is not None
            and outgoing is not None
            and target_bus is not None
            and _source_fanout_splits_before_target_bus(
                graph,
                link_key[0],
                outgoing,
                target_bus,
            )
        ):
            split_bus_y = _fanout_leg_routing_bus_y(
                link_key,
                graph=graph,
                outgoing=outgoing,
                target_bus=target_bus,
                merge_link_bus=merge_link_bus,
                tee_y=tee_y,
                anchors={link_key[1]: target},
            )
            candidates.insert(
                0,
                _fanout_tee_then_entry_column_points(
                    source,
                    target,
                    spread_entry_x,
                    tee_y=tee_y,
                    bus_y=split_bus_y,
                ),
            )
            candidates.insert(
                0,
                _tee_branch_avoiding_vertical_obstacles(
                    source,
                    target,
                    spread_entry_x,
                    tee_y,
                    obstacles,
                ),
            )
            candidates.insert(
                0,
                _fanout_split_branch_gutter_route(
                    source,
                    target,
                    spread_entry_x,
                    tee_y,
                    obstacles,
                ),
            )
        if merge_entry_x is not None and link_key in merge_entry_x and leg_bus is not None:
            candidates.insert(
                0,
                _shared_merge_bus_connector_points(
                    source,
                    target,
                    spread_entry_x,
                    leg_bus,
                    obstacles,
                    graph=graph,
                    positions=positions,
                    link_key=link_key,
                ),
            )
    side_bypass = None
    skip_horizontal_side_bypass = False
    fanout_leg_below_tee = False
    if (
        graph is not None
        and link_key is not None
        and source_bus is not None
        and merge_link_bus is not None
    ):
        tee_level = source_bus.get(link_key[0])
        leg_level = merge_link_bus.get(link_key)
        fanout_leg_below_tee = (
            tee_level is not None
            and leg_level is not None
            and leg_level < tee_level - PARALLEL_CONNECTOR_COORD_EPS
        )
    if graph is not None and link_key is not None:
        from visualizer.computation_graph import _inline_frame_tail_indices

        skip_horizontal_side_bypass = link_key[0] in _inline_frame_tail_indices(graph) or (
            merge_link_bus is not None
            and link_key in merge_link_bus
            and merge_entry_x is not None
            and link_key in merge_entry_x
            and abs(source.cx - target.cx) < 0.08
        )
    if fanout_leg_below_tee:
        skip_horizontal_side_bypass = True
    if not skip_horizontal_side_bypass:
        side_bypass = _horizontal_departure_side_bypass_route(
            source,
            target,
            obstacles,
            bus_y=bus_y,
            tee_y=source_bus.get(link_key[0]) if source_bus and link_key else None,
        )
    if side_bypass is not None:
        candidates.insert(0, side_bypass)
    if abs(source.cx - target.cx) < 0.06:
        candidates.append(
            _same_column_side_gutter_detour(source, target, obstacles)
        )
    candidates.append(
        _orthogonal_path(
            source,
            target,
            obstacles,
            bus_near=bus_near,
            bus_y=bus_y,
        )
    )
    if abs(source.cx - target.cx) < 0.06:
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
        if _connector_path_has_block_edge_horizontal_jog(
            candidate,
            source=source,
            target=target,
            link_key=link_key,
            graph=graph,
        ):
            continue
        if (
            merge_entry_x is not None
            and link_key is not None
            and link_key in merge_entry_x
            and abs(source.cx - target.cx) >= 0.08
            and _path_crosses_attached_block_edge_band(
                candidate,
                source=source,
                target=target,
            )
        ):
            continue
        if (
            merge_entry_x is not None
            and link_key is not None
            and link_key in merge_entry_x
            and abs(source.cx - target.cx) >= 0.08
            and _spread_merge_horizontal_below_target_corridor(candidate, target)
        ):
            continue
        if (
            merge_entry_x is not None
            and link_key is not None
            and link_key in merge_entry_x
            and not _connector_path_matches_spread_entry_port(
                candidate,
                target=target,
                entry_x=merge_entry_x[link_key],
            )
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
    repaired = _repair_connector_target_top_edge_overlap(
        points,
        source=source,
        target=target,
        entry_x=(
            merge_entry_x.get(link_key, target.cx)
            if merge_entry_x is not None and link_key is not None
            else target.cx
        ),
    )
    if _connector_path_clear_of_blocks(
        repaired,
        source=source,
        target=target,
        obstacles=obstacles,
    ):
        return repaired
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
    merge_entry_x: dict[tuple[int, int], float] | None = None,
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
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
            merge_entry_x=merge_entry_x,
            outgoing=outgoing,
            target_bus=target_bus,
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


def _segment_is_spread_merge_top_entry_approach(
    points: list[tuple[float, float]],
    index: int,
    target: _RenderAnchor,
    *,
    gap: float = 0.04,
    margin: float = CONNECTOR_ATTACHED_BOX_MARGIN,
) -> bool:
    """True when a horizontal leg runs into a vertical rise at the target top entry."""
    if index + 2 >= len(points):
        return False
    x1, y1 = points[index]
    x2, y2 = points[index + 1]
    x3, y3 = points[index + 2]
    if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS:
        return False
    if abs(x2 - x3) > PARALLEL_CONNECTOR_COORD_EPS:
        return False
    entry_y = _connector_target_top_entry_y(target, gap=gap)
    if abs(y3 - entry_y) > PARALLEL_CONNECTOR_COORD_EPS:
        return False
    if y3 < y2 - PARALLEL_CONNECTOR_COORD_EPS:
        return False
    return abs(x2 - target.cx) <= (target.right - target.left) / 2 + margin


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
            index == 0
            and _segment_is_source_departure_channel(source, x1, y1, x2, y2, margin=margin)
        ):
            if index < len(points) - 2 and _segment_penetrates_anchor(
                x1, y1, x2, y2, target, margin=margin
            ):
                if y2 >= target.top - margin:
                    return True
            continue
        if (
            index > 0
            and not _segment_is_source_departure_channel(source, x1, y1, x2, y2, margin=margin)
            and _segment_penetrates_anchor(x1, y1, x2, y2, source, margin=margin)
        ):
            return True
        if index < len(points) - 2 and _segment_penetrates_anchor(
            x1, y1, x2, y2, target, margin=margin
        ):
            if _segment_is_spread_merge_top_entry_approach(points, index, target):
                continue
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
    approach_y = max(y2 + CONNECTOR_EXIT_STUB, _connector_min_bus_y_above_target(target, gap=gap))
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
    side_x = side_candidates[0]
    approach_y = max(y2 + CONNECTOR_EXIT_STUB, _connector_min_bus_y_above_target(target, gap=gap))
    return [
        *_connector_leave_source_to_side(source, target, side_x, gap=gap),
        (side_x, approach_y),
        (x2, approach_y),
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


def _spread_top_entry_bus_y(
    source: _RenderAnchor,
    target: _RenderAnchor,
    *,
    gap: float = 0.04,
) -> float:
    """Shared horizontal channel for spread top-entry ports, above the target top edge."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    entry_y = _connector_target_top_entry_y(target, gap=gap)
    min_bus = _connector_min_bus_y_above_target(target, gap=gap)
    return max(min_bus, min(y1 - CONNECTOR_EXIT_STUB, (y1 + entry_y) / 2))


def _same_column_spread_top_entry_connector_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    entry_x: float,
    *,
    gap: float = 0.04,
) -> list[tuple[float, float]]:
    """Drop vertically, shift on a bus above the target, then enter the spread port."""
    entry_y = _connector_target_top_entry_y(target, gap=gap)
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    if abs(entry_x - target.cx) <= PARALLEL_CONNECTOR_COORD_EPS / 2:
        return [(source.cx, y1), (source.cx, entry_y)]
    y_stub = y1 - CONNECTOR_EXIT_STUB
    y_bus = _spread_top_entry_bus_y(source, target, gap=gap)
    if y_bus <= y_stub + PARALLEL_CONNECTOR_COORD_EPS:
        if y_bus < y_stub - PARALLEL_CONNECTOR_COORD_EPS:
            return [(source.cx, y1), (source.cx, y_bus), (entry_x, y_bus), (entry_x, entry_y)]
        return [(source.cx, y1), (source.cx, y_stub), (entry_x, y_stub), (entry_x, entry_y)]
    return [(source.cx, y1), (source.cx, y_stub), (source.cx, y_bus), (entry_x, y_bus), (entry_x, entry_y)]


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
        left_gutter = _inline_frame_connector_gutter_width(graph, frame, positions, side="left")
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


def _lift_bus_y_above_inline_frame_interiors(
    bus_y: float,
    *,
    graph,
    positions: list,
    x_left: float,
    x_right: float,
    margin: float = CONNECTOR_OBSTACLE_MARGIN,
) -> float:
    """Raise a shared bus that would otherwise run inside a dotted frame envelope."""
    lifted = bus_y
    for frame in graph.inline_frames:
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        if x_right < bounds.left - margin or x_left > bounds.right + margin:
            continue
        # The caption band above the border belongs to the frame as much as its
        # interior does, so a bus crossing the frame has to clear both.
        caption_top = _inline_frame_caption_band_top(frame, bounds)
        if bounds.bottom + margin < lifted < caption_top + margin:
            lifted = max(
                lifted,
                bounds.top + margin + CONNECTOR_EXIT_STUB,
                caption_top + margin,
            )
    return lifted


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
        cleared = min(cleared, _inline_frame_below_exit_y(bounds))
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


_QKV_PROJ_ATTRS = frozenset({"q_proj", "k_proj", "v_proj"})


def _requires_shared_input_source_bus(graph, tgt: int) -> bool:
    """Depthwise-conv q/k/v linears always tee from the shared hidden_states bus."""
    spec = graph.nodes[tgt]
    return spec.block is not None and spec.block.attr_name in _QKV_PROJ_ATTRS


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
    if (
        len(points) >= 4
        and abs(x2 - source.cx) <= PARALLEL_CONNECTOR_COORD_EPS
        and y2 < tee_y - PARALLEL_CONNECTOR_COORD_EPS
    ):
        x3, y3 = points[3]
        if abs(y3 - y2) <= PARALLEL_CONNECTOR_COORD_EPS and abs(x3 - source.cx) > PARALLEL_CONNECTOR_COORD_EPS:
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


def _connector_path_departs_horizontally_from_source(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
) -> bool:
    """True when the first leg leaves the source bottom horizontally without a stub."""
    if len(points) < 2:
        return False
    y_exit = _connector_source_bottom_exit_y(source)
    x0, y0 = points[0]
    x1, y1 = points[1]
    if abs(y0 - y_exit) > CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN:
        return False
    if abs(y0 - y1) > PARALLEL_CONNECTOR_COORD_EPS:
        return False
    return abs(x0 - x1) > PARALLEL_CONNECTOR_COORD_EPS


def _connector_path_has_block_edge_horizontal_jog(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    link_key: tuple[int, int] | None = None,
    graph=None,
) -> bool:
    """True when a connector runs horizontally along a source bottom or target top."""
    if len(points) < 2:
        return False
    return _path_crosses_attached_block_edge_band(
        points,
        source=source,
        target=target,
    )


def _repair_connector_source_departure(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    link_key: tuple[int, int] | None = None,
    graph=None,
) -> list[tuple[float, float]]:
    """Re-insert a vertical departure stub when overlap shifts flatten the source exit."""
    if len(points) >= 3:
        y_exit = _connector_source_bottom_exit_y(source)
        y_stub = y_exit - CONNECTOR_EXIT_STUB
        x0, y0 = points[0]
        x1, y1 = points[1]
        x2, y2 = points[2]
        if (
            abs(x0 - source.cx) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(y0 - y_exit) <= CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN
            and abs(x1 - source.cx) <= PARALLEL_CONNECTOR_COORD_EPS
            and y1 <= y_stub + PARALLEL_CONNECTOR_COORD_EPS
            and abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS
        ):
            return points
    if not _connector_path_has_block_edge_horizontal_jog(
        points,
        source=source,
        target=target,
        link_key=link_key,
        graph=graph,
    ):
        return points
    if len(points) < 2:
        return points
    y_exit = _connector_source_bottom_exit_y(source)
    y_stub = y_exit - CONNECTOR_EXIT_STUB
    _x1, y1 = points[0]
    x2, y2 = points[1]
    if abs(y1 - y_exit) > CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN:
        return points
    route_y = y_stub
    if y2 < y_stub - PARALLEL_CONNECTOR_COORD_EPS:
        route_y = y2
    repaired = [(source.cx, y_exit), (source.cx, route_y), (x2, route_y), *points[2:]]
    repaired = _ensure_orthogonal_connector_path(repaired)
    return _repair_connector_target_top_edge_overlap(
        repaired,
        source=source,
        target=target,
    )


def _connector_source_exit_y(graph, src: int, source: _RenderAnchor) -> float:
    """Y where a detail connector leaves its source block."""
    del graph, src
    return _connector_source_bottom_exit_y(source)


def _connector_turn_before_clearing_source(
    points: list[tuple[float, float]],
    *,
    y_exit: float,
    source_cx: float,
) -> float | None:
    """Y of a horizontal that turns off the source column before the exit stub ends."""
    floor_y = y_exit - CONNECTOR_EXIT_STUB + PARALLEL_CONNECTOR_COORD_EPS
    lowest_source_y = y_exit
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if abs(x1 - source_cx) <= PARALLEL_CONNECTOR_COORD_EPS / 2:
            lowest_source_y = min(lowest_source_y, y1)
        if abs(x2 - source_cx) <= PARALLEL_CONNECTOR_COORD_EPS / 2:
            lowest_source_y = min(lowest_source_y, y2)
        if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS / 2:
            continue
        if abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS / 2:
            continue
        if y1 < floor_y:
            continue
        if min(abs(x1 - source_cx), abs(x2 - source_cx)) > PARALLEL_CONNECTOR_COORD_EPS / 2:
            continue
        if lowest_source_y < floor_y:
            return None
        return y1
    return None


def _assert_connectors_avoid_block_edge_horizontal_jogs(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    stage: str,
) -> None:
    """Runtime check: connectors never jog horizontally along source bottoms or target tops."""
    offenders: list[str] = []
    for link_key, points in link_paths.items():
        if len(points) < 2:
            continue
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue
        early_turn = _connector_turn_before_clearing_source(
            points,
            y_exit=_connector_source_exit_y(graph, src, source),
            source_cx=source.cx,
        )
        if early_turn is not None:
            offenders.append(
                f"{link_key} horizontal before exit stub y={early_turn:.4f}@{stage}"
            )
        y_exit = _connector_source_bottom_exit_y(source)
        y_entry = _connector_target_top_entry_y(target)
        if _connector_path_has_block_edge_horizontal_jog(
            points,
            source=source,
            target=target,
            link_key=link_key,
            graph=graph,
        ):
            for index in range(len(points) - 1):
                x1, y1 = points[index]
                x2, y2 = points[index + 1]
                if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS:
                    continue
                if abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS:
                    continue
                if (
                    index == 0
                    and abs(y1 - y_exit) <= CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN
                ):
                    offenders.append(
                        f"{link_key} horizontal along source bottom y={y1:.4f}@{stage}"
                    )
                if index == len(points) - 2 and abs(y1 - y_entry) <= PARALLEL_CONNECTOR_COORD_EPS / 2:
                    if (
                        abs(y1 - y_entry) <= 1e-9
                        and abs(y2 - y_entry) <= 1e-9
                        and target.left - PARALLEL_CONNECTOR_COORD_EPS
                        <= x2
                        <= target.right + PARALLEL_CONNECTOR_COORD_EPS
                    ):
                        continue
                    offenders.append(
                        f"{link_key} horizontal along target top y={y1:.4f}@{stage}"
                    )
    if offenders:
        raise RuntimeError(
            "connectors must not jog horizontally along block edges: "
            + ", ".join(offenders)
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
            if tee_y + CONNECTOR_EXIT_STUB + PARALLEL_CONNECTOR_COORD_EPS < min_bus:
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
    _assert_connectors_avoid_block_edge_horizontal_jogs(
        link_paths,
        graph=graph,
        anchors=anchors,
        stage=stage,
    )


def _center_spread_top_entry_span(
    link_count: int,
    target_anchor: _RenderAnchor,
) -> float:
    """Horizontal span for multiple top-entry ports clustered near the block center."""
    if link_count <= 1:
        return 0.0
    gap = TOP_ENTRY_PORT_GAP
    desired = (link_count - 1) * gap
    block_width = target_anchor.right - target_anchor.left
    margin = CONNECTOR_ATTACHED_BOX_MARGIN + gap / 2
    max_span = max(0.0, block_width - 2 * margin)
    center_band = min(
        desired,
        max_span,
        block_width * TOP_ENTRY_PORT_MAX_CENTER_BAND_FRACTION,
    )
    return max(center_band, gap)


def _is_same_column_feed(
    positions: list,
    anchors: dict[int, _RenderAnchor],
    src: int,
    tgt: int,
) -> bool:
    return abs(positions[src].cx - anchors[tgt].cx) < 0.08


def _merge_bus_y_clearing_same_column_feeds(
    bus_y: float,
    *,
    tgt: int,
    src: int,
    incoming: dict[int, list[tuple[int, int]]],
    positions: list,
    anchors: dict[int, _RenderAnchor],
) -> float:
    """Raise merge buses above stacked same-column blocks crossed by side feeders."""
    if _is_same_column_feed(positions, anchors, src, tgt):
        return bus_y
    target_cx = anchors[tgt].cx
    siblings = [
        anchors[s]
        for s, _ in incoming.get(tgt, [])
        if s != src and abs(positions[s].cx - target_cx) < 0.08
    ]
    if not siblings:
        return bus_y
    cleared = bus_y
    margin = CONNECTOR_OBSTACLE_MARGIN
    for sibling in siblings:
        if cleared >= sibling.bottom - margin and cleared <= sibling.top + margin:
            cleared = max(cleared, sibling.top + margin)
    return cleared


def _assign_merge_link_bus_for_spread(
    spread_links: list[tuple[int, int]],
    base_bus: float,
    *,
    tgt: int,
    incoming: dict[int, list[tuple[int, int]]],
    positions: list,
    anchors: dict[int, _RenderAnchor],
    merge_link_bus: dict[tuple[int, int], float],
    graph,
) -> None:
    for link in spread_links:
        src = link[0]
        if _is_same_column_feed(positions, anchors, src, tgt):
            continue
        source_block = graph.nodes[src].block if hasattr(graph, "nodes") else None
        target_block = graph.nodes[tgt].block if hasattr(graph, "nodes") else None
        if (
            source_block is not None
            and target_block is not None
            and source_block.attr_name.startswith("@op_")
            and target_block.attr_name.startswith("@op_")
            and len(
                [
                    item
                    for item in spread_links
                    if not _is_same_column_feed(positions, anchors, item[0], tgt)
                ]
            )
            == 1
        ):
            merge_link_bus[link] = _connector_min_bus_y_above_target(anchors[tgt])
            continue
        merge_link_bus[link] = _merge_bus_y_clearing_same_column_feeds(
            base_bus,
            tgt=tgt,
            src=src,
            incoming=incoming,
            positions=positions,
            anchors=anchors,
        )


def _merge_leg_level_fits(
    link: tuple[int, int],
    level: float,
    *,
    entry_x: float,
    target: _RenderAnchor,
    source: _RenderAnchor,
    obstacles: list[_RenderAnchor],
) -> bool:
    """True when a merge leg can run its horizontal at the given level."""
    if level >= _connector_source_bottom_exit_y(source) - CONNECTOR_EXIT_STUB:
        return False
    if level <= _connector_min_bus_y_above_target(target) - PARALLEL_CONNECTOR_COORD_EPS:
        return False
    cleared = _min_bus_y_clearing_horizontal_corridor(
        min(entry_x, source.cx),
        max(entry_x, source.cx),
        obstacles,
        proposed_y=level,
    )
    return cleared <= level + PARALLEL_CONNECTOR_COORD_EPS


def _nest_same_side_merge_bus_levels(
    spread_links: list[tuple[int, int]],
    *,
    tgt: int,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    merge_entry_x: dict[tuple[int, int], float],
    merge_link_bus: dict[tuple[int, int], float],
    obstacles: list[_RenderAnchor],
    incoming: dict[int, list[tuple[int, int]]] | None = None,
    target_bus: dict[int, float] | None = None,
    channel_gap: float = PARALLEL_CONNECTOR_CHANNEL_GAP,
) -> None:
    """Give each merge leg arriving from one side its own corridor into the target.

    Legs sharing a level run along the same line, so several tensors read as one
    trunk that the target taps. Nesting them instead - the outermost source stays
    lowest, nearer ones step up a channel each - keeps every leg visible without
    letting one cross another, since a nearer source never descends past the
    corridor of the source beyond it.
    """
    target = anchors.get(tgt)
    if target is None:
        return
    shared_bus_y = target_bus.get(tgt) if target_bus is not None else None
    incoming = incoming or {}
    for side in (-1.0, 1.0):
        side_links = [
            link
            for link in spread_links
            if link in merge_link_bus
            and link in merge_entry_x
            and link[0] in anchors
            and not _is_same_column_feed(positions, anchors, link[0], tgt)
            and side * (merge_entry_x[link] - target.cx)
            >= TOP_ENTRY_PORT_GAP - PARALLEL_CONNECTOR_COORD_EPS
        ]
        if len(side_links) < 2:
            continue
        side_links.sort(key=lambda link: -abs(anchors[link[0]].cx - target.cx))
        # Routing floors the outermost leg at the entry corridor later on, so step
        # up from that floor or the nesting would collapse to a smaller gap.
        level = max(
            merge_link_bus[side_links[0]],
            _connector_min_bus_y_above_target(target),
        )
        for link in side_links[1:]:
            lifted = max(merge_link_bus[link], level + channel_gap)
            if _merge_leg_level_fits(
                link,
                lifted,
                entry_x=merge_entry_x[link],
                target=target,
                source=anchors[link[0]],
                obstacles=obstacles,
            ):
                merge_link_bus[link] = lifted
            level = merge_link_bus[link]


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


def _spread_link_port_order_key(positions: list):
    """Order top-entry ports by source column, then by closeness to the target.

    Sources stacked in one column feed the same port band, so the lowest source
    takes the innermost port: it drops straight down while the ones above it have
    to pass beside it.
    """

    def key(link: tuple[int, int]) -> tuple[float, float]:
        source = positions[link[0]]
        return (source.cx, source.bottom)

    return key


def _same_column_top_entry_links(
    spread_links: list[tuple[int, int]],
    target_anchor: _RenderAnchor,
    positions: list,
    anchors: dict[int, _RenderAnchor],
) -> list[tuple[int, int]]:
    """Top-entry links whose source sits in the target's own column, lowest first."""
    same_column = [
        link
        for link in spread_links
        if link[0] in anchors
        and abs(positions[link[0]].cx - target_anchor.cx) < TOP_ENTRY_PORT_GAP
    ]
    return sorted(same_column, key=lambda link: anchors[link[0]].bottom)


def _bypass_port_x_beside_blockers(
    target_anchor: _RenderAnchor,
    blockers: list[_RenderAnchor],
    *,
    source_cx: float | None = None,
) -> float | None:
    """Top-entry port on the target edge that clears tiles stacked above it.

    The port has to stay a channel's width inside the target corner, otherwise the
    wire lands on the rounded corner and reads as touching the tile rather than
    entering it.
    """
    if not blockers:
        return None
    left_high = min(blocker.left for blocker in blockers) - CONNECTOR_OBSTACLE_MARGIN
    left_low = target_anchor.left + TOP_ENTRY_PORT_GAP
    right_low = max(blocker.right for blocker in blockers) + CONNECTOR_OBSTACLE_MARGIN
    right_high = target_anchor.right - TOP_ENTRY_PORT_GAP
    left_port = (left_low + left_high) / 2 if left_low <= left_high else None
    right_port = (right_low + right_high) / 2 if right_low <= right_high else None
    if left_port is None:
        return right_port
    if right_port is None:
        return left_port
    if source_cx is not None:
        if source_cx > target_anchor.cx + PARALLEL_CONNECTOR_COORD_EPS:
            return right_port
        if source_cx < target_anchor.cx - PARALLEL_CONNECTOR_COORD_EPS:
            return left_port
        if abs(source_cx - target_anchor.cx) < TOP_ENTRY_PORT_GAP:
            return right_port
    return left_port


def _bypass_gutter_entry_beside_blockers(
    target_anchor: _RenderAnchor,
    blockers: list[_RenderAnchor],
    *,
    source_cx: float | None = None,
) -> tuple[float, float, float] | None:
    """Port beside the target center, reached by a gutter outside the blockers.

    When the passed tiles are nearly as wide as the target there is no port left on
    the target edge beyond them, so the bypass drops outside both tiles instead and
    turns back in along the row gap to enter next to the main feed.
    """
    if not blockers:
        return None
    jog_y = _connector_min_bus_y_above_target(target_anchor)
    lowest_blocker = min(blocker.bottom for blocker in blockers)
    if lowest_blocker - jog_y < CONNECTOR_OBSTACLE_MARGIN:
        return None
    candidates: list[tuple[float, float, float]] = []
    left_port = target_anchor.cx - TOP_ENTRY_PORT_GAP
    if left_port >= target_anchor.left + CONNECTOR_ATTACHED_BOX_MARGIN:
        left_gutter = (
            min(min(blocker.left for blocker in blockers), target_anchor.left)
            - CONNECTOR_OBSTACLE_MARGIN
            - CONNECTOR_ATTACHED_BOX_MARGIN
        )
        candidates.append((left_port, left_gutter, jog_y))
    right_port = target_anchor.cx + TOP_ENTRY_PORT_GAP
    if right_port <= target_anchor.right - CONNECTOR_ATTACHED_BOX_MARGIN:
        right_gutter = (
            max(max(blocker.right for blocker in blockers), target_anchor.right)
            + CONNECTOR_OBSTACLE_MARGIN
            + CONNECTOR_ATTACHED_BOX_MARGIN
        )
        candidates.append((right_port, right_gutter, jog_y))
    if not candidates:
        return None
    if source_cx is not None and abs(source_cx - target_anchor.cx) < TOP_ENTRY_PORT_GAP:
        for port_x, gutter_x, jog in candidates:
            if port_x >= target_anchor.cx:
                return (port_x, gutter_x, jog)
    return candidates[0]


def _same_column_bypass_corridor_y(
    blocker: _RenderAnchor,
    blocker_bus_y: float | None,
) -> float | None:
    """Channel between a passed-by tile's merge bus and its top edge."""
    if blocker_bus_y is None:
        return None
    low = _connector_min_bus_y_above_target(blocker)
    high = blocker_bus_y - PARALLEL_CONNECTOR_CHANNEL_GAP
    if high < low:
        return None
    return (low + high) / 2


@dataclass(frozen=True)
class _SameColumnBypass:
    """Where a stacked feeder enters its target and how it gets past the tiles between."""

    port_x: float
    corridor_y: float
    gutter_x: float | None = None
    jog_y: float | None = None


def _same_column_bypass_assignments(
    links: list[tuple[int, int]],
    target_anchor: _RenderAnchor,
    *,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    target_bus: dict[int, float],
) -> dict[tuple[int, int], _SameColumnBypass]:
    """Bypass port column and tee level for feeders stacked in the target column.

    The lowest feeder drops straight down the column; every feeder above it has
    to pass beside that tile. Teeing below the passed tile's merge bus keeps the
    bypass clear of the wires feeding it, and a port beside the tile keeps the
    entry vertical.
    """
    ordered = _same_column_top_entry_links(links, target_anchor, positions, anchors)
    if len(ordered) < 2:
        return {}
    assignments: dict[tuple[int, int], _SameColumnBypass] = {}
    for index, link in enumerate(ordered[1:], start=1):
        source = anchors[link[0]]
        blocked = [
            lower
            for lower in ordered[:index]
            if anchors[lower[0]].top < source.bottom
            and anchors[lower[0]].bottom > target_anchor.top
        ]
        if not blocked:
            continue
        topmost = max(blocked, key=lambda lower: anchors[lower[0]].top)
        corridor_y = _same_column_bypass_corridor_y(
            anchors[topmost[0]],
            target_bus.get(topmost[0]),
        )
        if corridor_y is None:
            corridor_y = source.bottom - CONNECTOR_EXIT_STUB - CONNECTOR_ATTACHED_BOX_MARGIN
        blockers = [
            anchor
            for node_index, anchor in anchors.items()
            if node_index not in {link[0], link[1]}
            and anchor.top < source.bottom
            and anchor.bottom > target_anchor.top
            and anchor.left < target_anchor.right
            and anchor.right > target_anchor.left
        ]
        if corridor_y >= source.bottom - CONNECTOR_EXIT_STUB:
            needs_exterior_gutter = any(
                blocker.right > target_anchor.cx + CONNECTOR_OBSTACLE_MARGIN
                or blocker.left < target_anchor.cx - CONNECTOR_OBSTACLE_MARGIN
                for blocker in blockers
            )
            if not needs_exterior_gutter:
                continue
            gutter = _bypass_gutter_entry_beside_blockers(
                target_anchor,
                blockers,
                source_cx=source.cx,
            )
            if gutter is None:
                continue
            port_x, gutter_x, jog_y = gutter
            corridor_y = source.bottom - CONNECTOR_EXIT_STUB - CONNECTOR_ATTACHED_BOX_MARGIN
            assignments[link] = _SameColumnBypass(port_x, corridor_y, gutter_x, jog_y)
            continue
        port_x = _bypass_port_x_beside_blockers(target_anchor, blockers, source_cx=source.cx)
        if port_x is not None:
            assignments[link] = _SameColumnBypass(port_x, corridor_y)
            continue
        gutter = _bypass_gutter_entry_beside_blockers(target_anchor, blockers, source_cx=source.cx)
        if gutter is None:
            continue
        port_x, gutter_x, jog_y = gutter
        assignments[link] = _SameColumnBypass(port_x, corridor_y, gutter_x, jog_y)
    return assignments


def _assign_same_column_bypass_entry(
    spread_links: list[tuple[int, int]],
    target_anchor: _RenderAnchor,
    *,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    target_bus: dict[int, float],
    merge_entry_x: dict[tuple[int, int], float],
    merge_link_bus: dict[tuple[int, int], float],
) -> None:
    """Give stacked same-column feeders bypass ports beside the tiles they pass."""
    assignments = _same_column_bypass_assignments(
        spread_links,
        target_anchor,
        positions=positions,
        anchors=anchors,
        target_bus=target_bus,
    )
    ordered = _same_column_top_entry_links(
        spread_links,
        target_anchor,
        positions,
        anchors,
    )
    if len(ordered) < 2:
        return
    merge_entry_x[ordered[0]] = target_anchor.cx
    unassigned = [link for link in ordered[1:] if link not in assignments]
    for offset, link in enumerate(unassigned, start=1):
        merge_entry_x[link] = max(
            target_anchor.left + CONNECTOR_ATTACHED_BOX_MARGIN,
            target_anchor.cx - offset * TOP_ENTRY_PORT_GAP,
        )
    for link, bypass in assignments.items():
        merge_entry_x[link] = bypass.port_x
        merge_link_bus[link] = bypass.corridor_y


def _same_column_bypass_top_entry_route(
    source: _RenderAnchor,
    target: _RenderAnchor,
    bypass: _SameColumnBypass,
    *,
    departure_y: float | None = None,
    gap: float = 0.04,
) -> list[tuple[float, float]]:
    """Tee below a passed tile's merge bus, then drop on the bypass port column."""
    exit_y = _connector_source_bottom_exit_y(source, gap=gap)
    entry_y = _connector_target_top_entry_y(target, gap=gap)
    tee_y = bypass.corridor_y if departure_y is None else departure_y
    if bypass.gutter_x is None or bypass.jog_y is None:
        return _ensure_orthogonal_connector_path(
            [
                (source.cx, exit_y),
                (source.cx, tee_y),
                (bypass.port_x, tee_y),
                (bypass.port_x, entry_y),
            ]
        )
    return _ensure_orthogonal_connector_path(
        [
            (source.cx, exit_y),
            (source.cx, tee_y),
            (bypass.gutter_x, tee_y),
            (bypass.gutter_x, bypass.jog_y),
            (bypass.port_x, bypass.jog_y),
            (bypass.port_x, entry_y),
        ]
    )


def _lower_bypass_corridor_clearing_crossings(
    route: list[tuple[float, float]],
    *,
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    link: tuple[int, int],
    source: _RenderAnchor,
    target: _RenderAnchor,
    anchors: dict[int, _RenderAnchor],
    obstacles: list[_RenderAnchor],
    graph,
    positions: list | None,
) -> list[tuple[float, float]]:
    """Drop a bypass's tee row below the merge rows its gutter would otherwise cut.

    A tile fed by several stacked sources has one row per feed rather than one shared bus,
    so teeing just below the topmost row still leaves the gutter crossing the rest. Teeing
    below the lowest of them is the only row that clears the whole merge.
    """
    if len(route) < 6:
        return route
    corridor_y = route[1][1]
    if abs(corridor_y - route[2][1]) > PARALLEL_CONNECTOR_COORD_EPS:
        return route
    # Only a gutter that leaves the column and comes back owns its tee row; a feed that
    # crosses over once is teeing onto a merge row the whole group agreed on.
    out = route[2][0] - route[1][0]
    back = route[4][0] - route[3][0]
    if out * back >= 0:
        return route
    if not _connector_path_crossing_count({**link_paths, link: route}, link):
        return route
    # The tee row still has to stay above every tile its own leg runs over.
    span_lo, span_hi = sorted((route[1][0], route[2][0]))
    floor = max(
        [
            _connector_min_bus_y_above_target(target),
            *(
                anchor.top + CONNECTOR_OBSTACLE_MARGIN
                for anchor in anchors.values()
                if anchor.left < span_hi
                and anchor.right > span_lo
                and anchor.top < corridor_y
            ),
        ]
    )
    rows = sorted(
        {
            row - PARALLEL_CONNECTOR_CHANNEL_GAP
            for other, points in link_paths.items()
            if other != link
            for orientation, row, _lo, _hi, _index in _connector_axis_segments(points)
            if orientation == "h" and floor <= row - PARALLEL_CONNECTOR_CHANNEL_GAP < corridor_y
        },
        reverse=True,
    )
    for row in rows:
        candidate = _ensure_orthogonal_connector_path(
            [route[0], (route[1][0], row), (route[2][0], row), *route[3:]]
        )
        if not _connector_path_clear_of_blocks(
            candidate,
            source=source,
            target=target,
            obstacles=obstacles,
        ):
            continue
        if positions is not None and (
            _connector_path_violates_inline_frame_bounds(
                candidate,
                graph,
                positions,
                src=link[0],
                tgt=link[1],
            )
            is not None
        ):
            continue
        # Re-seating the row is only worth it when it clears the gutter outright; a row that
        # still cuts something has just traded one merge for another.
        if _connector_path_crossing_count({**link_paths, link: candidate}, link):
            continue
        # Landing on a row another connector already runs along would draw the two as one
        # line, which costs the reader more than the crossing did.
        if _horizontal_run_is_shared({**link_paths, link: candidate}, link, 1):
            continue
        return candidate
    return route


def _lower_bypass_corridors_clearing_crossings(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
) -> None:
    """Re-seat every bypass tee row that later stages left crossing a merge row."""
    for link, route in sorted(link_paths.items()):
        source = anchors.get(link[0])
        target = anchors.get(link[1])
        if source is None or target is None:
            continue
        link_paths[link] = _lower_bypass_corridor_clearing_crossings(
            route,
            link_paths=link_paths,
            link=link,
            source=source,
            target=target,
            anchors=anchors,
            obstacles=_connector_block_obstacles(
                anchors,
                src=link[0],
                tgt=link[1],
                label_obstacles=label_obstacles,
                graph=graph,
                positions=positions,
                link_key=link,
            ),
            graph=graph,
            positions=positions,
        )


def _widen_bypass_gutter_clearing_obstacles(
    bypass: _SameColumnBypass,
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    departure_y: float | None,
) -> _SameColumnBypass:
    """Push a bypass gutter out past every tile its vertical run would graze.

    The gutter is picked from the tiles that share the target's column, but a long bypass
    runs past whole rows on its way down, and a tile further out in one of those rows is
    just as much in the way.
    """
    if bypass.gutter_x is None:
        return bypass
    outward = 1.0 if bypass.gutter_x > source.cx else -1.0
    gutter_x = bypass.gutter_x
    for _ in range(len(obstacles) + 1):
        candidate = _SameColumnBypass(
            bypass.port_x, bypass.corridor_y, gutter_x, bypass.jog_y
        )
        route = _same_column_bypass_top_entry_route(
            source, target, candidate, departure_y=departure_y
        )
        blocking = [
            obstacle
            for obstacle in obstacles
            if _path_hits_obstacles(route, [obstacle], margin=CONNECTOR_OBSTACLE_MARGIN)
        ]
        if not blocking:
            return candidate
        edges = [
            (obstacle.right if outward > 0 else obstacle.left) for obstacle in blocking
        ]
        clear_x = (max(edges) if outward > 0 else min(edges)) + outward * (
            CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN
        )
        if outward * (clear_x - gutter_x) <= PARALLEL_CONNECTOR_COORD_EPS:
            break
        gutter_x = clear_x
    return bypass


def _apply_same_column_bypass_routes(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    incoming: dict[int, list[tuple[int, int]]],
    merge_entry_x: dict[tuple[int, int], float],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
) -> None:
    """Route legs that pass beside a stacked sibling below that sibling's bus."""
    for tgt, link_group in incoming.items():
        target_anchor = anchors.get(tgt)
        if target_anchor is None:
            continue
        assignments = _same_column_bypass_assignments(
            link_group,
            target_anchor,
            positions=positions,
            anchors=anchors,
            target_bus=target_bus,
        )
        for link, bypass in assignments.items():
            source = anchors.get(link[0])
            if source is None or link not in link_paths:
                continue
            if (
                tgt in target_bus
                and abs(merge_entry_x.get(link, target_anchor.cx) - target_anchor.cx)
                < PARALLEL_CONNECTOR_COORD_EPS
            ):
                continue
            merge_entry_x[link] = bypass.port_x
            blockers = [
                anchor
                for node_index, anchor in anchors.items()
                if node_index not in {link[0], tgt}
                and anchor.top < source.bottom
                and anchor.bottom > target_anchor.top
                and anchor.left < target_anchor.right
                and anchor.right > target_anchor.left
            ]
            right_blockers = [
                blocker
                for blocker in blockers
                if blocker.right > source.cx + PARALLEL_CONNECTOR_COORD_EPS
            ]
            side_blockers = [
                anchor
                for node_index, anchor in anchors.items()
                if node_index not in {link[0], tgt}
                and anchor.right > source.cx + PARALLEL_CONNECTOR_COORD_EPS
                and anchor.top < target_anchor.top - PARALLEL_CONNECTOR_COORD_EPS
                and anchor.bottom > source.bottom + CONNECTOR_OBSTACLE_MARGIN
            ]
            if side_blockers:
                right_blockers = sorted(
                    {*right_blockers, *side_blockers},
                    key=lambda blocker: blocker.right,
                )
            bypass_use = bypass
            if (
                bypass.gutter_x is None
                and bypass.port_x > source.cx + PARALLEL_CONNECTOR_COORD_EPS
                and right_blockers
            ):
                gutter_x = (
                    max(blocker.right for blocker in right_blockers)
                    + CONNECTOR_OBSTACLE_MARGIN
                    + CONNECTOR_ATTACHED_BOX_MARGIN
                )
                jog_y = _connector_min_bus_y_above_target(target_anchor)
                bypass_use = _SameColumnBypass(
                    bypass.port_x,
                    bypass.corridor_y,
                    gutter_x,
                    jog_y,
                )
            obstacles = _connector_block_obstacles(
                anchors,
                src=link[0],
                tgt=tgt,
                label_obstacles=label_obstacles,
                graph=graph,
                positions=positions,
                link_key=link,
            )
            bypass_use = _widen_bypass_gutter_clearing_obstacles(
                bypass_use,
                source=source,
                target=target_anchor,
                obstacles=obstacles,
                departure_y=source_bus.get(link[0]),
            )
            route = _same_column_bypass_top_entry_route(
                source,
                target_anchor,
                bypass_use,
                departure_y=source_bus.get(link[0]),
            )
            if not _connector_path_clear_of_blocks(
                route,
                source=source,
                target=target_anchor,
                obstacles=obstacles,
            ) and bypass_use.gutter_x is None:
                continue
            if (
                _connector_path_violates_inline_frame_bounds(
                    route,
                    graph,
                    positions,
                    src=link[0],
                    tgt=tgt,
                )
                is not None
                and bypass_use.gutter_x is None
            ):
                continue
            link_paths[link] = _lower_bypass_corridor_clearing_crossings(
                route,
                link_paths=link_paths,
                link=link,
                source=source,
                target=target_anchor,
                anchors=anchors,
                obstacles=obstacles,
                graph=graph,
                positions=positions,
            )
        for link in link_group:
            if link in assignments:
                continue
            source = anchors.get(link[0])
            if source is None or link not in link_paths:
                continue
            if (
                tgt in target_bus
                and abs(merge_entry_x.get(link, target_anchor.cx) - target_anchor.cx)
                < PARALLEL_CONNECTOR_COORD_EPS
            ):
                continue
            if abs(source.cx - target_anchor.cx) <= TOP_ENTRY_PORT_GAP:
                continue
            sibling_sources = [
                anchors[s]
                for s, t in link_group
                if t == tgt and s != link[0] and s in anchors
            ]
            if not sibling_sources:
                continue
            blockers = sibling_sources + [
                anchor
                for node_index, anchor in anchors.items()
                if node_index not in {link[0], tgt}
                and anchor.top < source.bottom
                and anchor.bottom > target_anchor.top
                and anchor.left < target_anchor.right
                and anchor.right > target_anchor.left
            ]
            if not blockers:
                continue
            sibling_buses = [
                target_bus[src]
                for src, tgt_id in link_group
                if tgt_id == tgt and (src, tgt_id) != link and src in target_bus
            ]
            corridor_y = source.bottom - CONNECTOR_EXIT_STUB - CONNECTOR_ATTACHED_BOX_MARGIN
            if sibling_buses:
                corridor_y = min(corridor_y, min(sibling_buses) - CONNECTOR_OBSTACLE_MARGIN)
            # The corridor rides above the tiles the leg has to pass. Sibling sources are
            # feeds into the same target rather than tiles in the way, and one sharing the
            # source's row would push the corridor back up past the source it just left.
            corridor_y = max(
                corridor_y,
                max(tile.top for tile in blockers) + CONNECTOR_OBSTACLE_MARGIN,
            )
            # The corridor rides above the tiles the leg passes, but a blocker level with the
            # source pushes it above the source's own bottom edge, and a leg that climbs back
            # over the tile it just left is not a bypass.
            if corridor_y > source.bottom - PARALLEL_CONNECTOR_COORD_EPS:
                continue
            gutter_x = (
                max(blocker.right for blocker in blockers)
                + CONNECTOR_OBSTACLE_MARGIN
                + CONNECTOR_ATTACHED_BOX_MARGIN
            )
            port_x = _bypass_port_x_beside_blockers(
                target_anchor,
                blockers,
                source_cx=source.cx,
            )
            if port_x is None:
                port_x = min(
                    target_anchor.right - CONNECTOR_ATTACHED_BOX_MARGIN,
                    target_anchor.cx + TOP_ENTRY_PORT_GAP,
                )
            bypass = _SameColumnBypass(port_x, corridor_y, gutter_x, corridor_y)
            route = _same_column_bypass_top_entry_route(
                source,
                target_anchor,
                bypass,
                departure_y=source_bus.get(link[0]),
            )
            obstacles = _connector_block_obstacles(
                anchors,
                src=link[0],
                tgt=tgt,
                label_obstacles=label_obstacles,
                graph=graph,
                positions=positions,
                link_key=link,
            )
            if not _connector_path_clear_of_blocks(
                route,
                source=source,
                target=target_anchor,
                obstacles=obstacles,
            ):
                continue
            link_paths[link] = route
            merge_entry_x[link] = port_x


def _path_with_horizontal_moved_to_y(
    points: list[tuple[float, float]],
    index: int,
    new_y: float,
) -> list[tuple[float, float]] | None:
    """Return the path with interior horizontal ``index`` moved to ``new_y``.

    None when the move would reverse one of the neighbouring vertical legs, which
    would trade a border overlap for a backtracking connector.
    """
    if not 1 <= index <= len(points) - 3:
        return None
    above = points[index - 1][1]
    below = points[index + 2][1]
    if above < below:
        above, below = below, above
    if not below <= new_y <= above:
        return None
    moved = list(points)
    moved[index] = (points[index][0], new_y)
    moved[index + 1] = (points[index + 1][0], new_y)
    return _ensure_orthogonal_connector_path(moved)


def _lift_connector_horizontals_off_frame_borders(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
) -> None:
    """Move horizontal runs off the dotted borders they would be drawn on top of.

    An approach corridor into a framed tile lands on the border whenever the frame
    pad and the connector stub measure the same, which reads as the connector and
    the border being one line. The run is unambiguous either just outside the frame
    or well inside it, so take whichever side leaves the rest of the path valid.
    """
    if not graph.inline_frames:
        return
    margin = CONNECTOR_OBSTACLE_MARGIN
    for link_key, points in list(link_paths.items()):
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None or len(points) < 4:
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
        current = list(points)
        for frame in graph.inline_frames:
            bounds = _inline_frame_draw_bounds(frame, positions, graph)
            for index in range(1, len(current) - 2):
                x1, y1 = current[index]
                x2, y2 = current[index + 1]
                if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS or abs(x1 - x2) <= 0.06:
                    continue
                if min(x1, x2) >= bounds.right or max(x1, x2) <= bounds.left:
                    continue
                if _horizontal_clears_frame_border_lines(y1, bounds, margin=margin):
                    continue
                edge = (
                    bounds.top
                    if abs(y1 - bounds.top) <= abs(y1 - bounds.bottom)
                    else bounds.bottom
                )
                for candidate_y in (edge + margin, edge - margin):
                    moved = _path_with_horizontal_moved_to_y(current, index, candidate_y)
                    if moved is None:
                        continue
                    if _path_hits_obstacles(moved, obstacles, margin=margin):
                        continue
                    if not _connector_path_clear_of_blocks(
                        moved,
                        source=source,
                        target=target,
                        obstacles=obstacles,
                    ):
                        continue
                    if (
                        _connector_path_violates_inline_frame_bounds(
                            moved,
                            graph,
                            positions,
                            src=src,
                            tgt=tgt,
                        )
                        is not None
                    ):
                        continue
                    current = moved
                    break
                break
        if current != points:
            link_paths[link_key] = current


def _repair_final_connector_tile_collisions(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    source_bus: dict[int, float],
    merge_entry_x: dict[tuple[int, int], float],
    merge_link_bus: dict[tuple[int, int], float],
) -> None:
    """Replace any final path that still penetrates a non-endpoint tile."""
    for link_key, points in list(link_paths.items()):
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
        if not _path_hits_obstacles(
            points,
            obstacles,
            margin=CONNECTOR_OBSTACLE_MARGIN,
        ) and _connector_path_clear_of_blocks(
            points,
            source=source,
            target=target,
            obstacles=obstacles,
        ):
            continue

        candidates: list[list[tuple[float, float]] | None] = []
        tail_frame = _frame_for_tail_node(graph, src)
        if tail_frame is not None and tgt not in tail_frame.node_indices:
            candidates.append(
                _frame_tail_merge_entry_connector_points(
                    source,
                    target,
                    exit_x=source.cx,
                    entry_x=merge_entry_x.get(link_key, target.cx),
                    bus_y=merge_link_bus.get(
                        link_key,
                        _connector_min_bus_y_above_target(target),
                    ),
                    frame_bounds=_inline_frame_draw_bounds(
                        tail_frame,
                        positions,
                        graph,
                    ),
                    obstacles=obstacles,
                )
            )
        candidates.extend(
            [
                _horizontal_departure_side_bypass_route(
                    source,
                    target,
                    obstacles,
                    tee_y=source_bus.get(src),
                ),
                _same_column_side_gutter_detour(source, target, obstacles),
            ]
        )
        for candidate in candidates:
            if candidate is None:
                continue
            if _path_hits_obstacles(
                candidate,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            ):
                continue
            if not _connector_path_clear_of_blocks(
                candidate,
                source=source,
                target=target,
                obstacles=obstacles,
            ):
                continue
            if (
                _connector_path_violates_inline_frame_bounds(
                    candidate,
                    graph,
                    positions,
                    src=src,
                    tgt=tgt,
                )
                is not None
            ):
                continue
            link_paths[link_key] = candidate
            break


def _repair_final_connector_frame_collisions(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    source_bus: dict[int, float],
    merge_entry_x: dict[tuple[int, int], float],
    merge_link_bus: dict[tuple[int, int], float],
) -> None:
    """Reassert frame entry and exit corridors after all path post-processing."""
    for link_key, _frame_id, _reason in _find_connector_inline_frame_overlaps(
        link_paths,
        graph=graph,
        positions=positions,
    ):
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
        tail_frame = _frame_for_tail_node(graph, src)
        candidate = None
        if tail_frame is not None and tgt not in tail_frame.node_indices:
            candidate = _frame_tail_merge_entry_connector_points(
                source,
                target,
                exit_x=source.cx,
                entry_x=merge_entry_x.get(link_key, target.cx),
                bus_y=merge_link_bus.get(
                    link_key,
                    _connector_min_bus_y_above_target(target),
                ),
                frame_bounds=_inline_frame_draw_bounds(
                    tail_frame,
                    positions,
                    graph,
                ),
                obstacles=obstacles,
            )
        else:
            candidate = _outside_to_inline_frame_top_member_route(
                source,
                target,
                obstacles,
                graph,
                positions,
                src=src,
                tgt=tgt,
                entry_x=merge_entry_x.get(link_key, target.cx),
                source_tee_y=source_bus.get(src),
            )
        if candidate is None:
            continue
        if (
            _connector_path_violates_inline_frame_bounds(
                candidate,
                graph,
                positions,
                src=src,
                tgt=tgt,
            )
            is None
            and not _path_hits_obstacles(
                candidate,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            )
        ):
            link_paths[link_key] = candidate


def _apply_directional_fanout_bypass_routes(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    outgoing: dict[int, list[tuple[int, int]]],
    source_bus: dict[int, float],
) -> None:
    """Route a side target on its own side of a straight-down fan-out sibling."""
    for src, link_group in outgoing.items():
        source = anchors.get(src)
        if source is None or len(link_group) < 2:
            continue
        has_straight_sibling = any(
            tgt in anchors
            and abs(anchors[tgt].cx - source.cx) <= TOP_ENTRY_PORT_GAP
            for _, tgt in link_group
        )
        if not has_straight_sibling:
            continue
        for link in link_group:
            _, tgt = link
            target = anchors.get(tgt)
            current = link_paths.get(link)
            if target is None or current is None:
                continue
            delta_x = target.cx - source.cx
            if abs(delta_x) <= TOP_ENTRY_PORT_GAP:
                continue
            blockers = [
                anchor
                for node_index, anchor in anchors.items()
                if node_index not in {src, tgt}
                and anchor.top < source.bottom
                and anchor.bottom > target.top
                and anchor.left < target.right
                and anchor.right > target.left
            ]
            if not blockers:
                continue
            if delta_x < 0:
                gutter_x = (
                    min(target.left, *(blocker.left for blocker in blockers))
                    - CONNECTOR_OBSTACLE_MARGIN
                    - CONNECTOR_ATTACHED_BOX_MARGIN
                )
                port_x = max(
                    target.left + CONNECTOR_ATTACHED_BOX_MARGIN,
                    target.cx - TOP_ENTRY_PORT_GAP,
                )
            else:
                gutter_x = (
                    max(target.right, *(blocker.right for blocker in blockers))
                    + CONNECTOR_OBSTACLE_MARGIN
                    + CONNECTOR_ATTACHED_BOX_MARGIN
                )
                port_x = min(
                    target.right - CONNECTOR_ATTACHED_BOX_MARGIN,
                    target.cx + TOP_ENTRY_PORT_GAP,
                )
            departure_y = source_bus.get(src, source.bottom - CONNECTOR_EXIT_STUB)
            # Stay in the final row corridor until the route has passed every
            # intermediate tile, then enter through the target's top edge.
            jog_y = _connector_min_bus_y_above_target(target)
            candidate = _ensure_orthogonal_connector_path(
                [
                    (source.cx, source.bottom),
                    (source.cx, departure_y),
                    (gutter_x, departure_y),
                    (gutter_x, jog_y),
                    (port_x, jog_y),
                    (port_x, target.top),
                ]
            )
            obstacles = _connector_block_obstacles(
                anchors,
                src=src,
                tgt=tgt,
                label_obstacles=label_obstacles,
                graph=graph,
                positions=positions,
                link_key=link,
            )
            candidate = _clear_connector_path_obstacles(candidate, obstacles)
            if _path_hits_obstacles(candidate, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN):
                continue
            if _connector_path_violates_inline_frame_bounds(
                candidate,
                graph,
                positions,
                src=src,
                tgt=tgt,
            ) is not None:
                continue
            shared_frame = _inline_frame_for_nodes(graph, src, tgt)
            if shared_frame is not None:
                frame_bounds = _inline_frame_routing_bounds(shared_frame, positions, graph)
                if not _path_stays_inside_bounds(
                    candidate,
                    frame_bounds,
                    margin=0.0,
                ):
                    continue
            link_paths[link] = candidate


def _fork_join_side_branch_bypass_route(
    side: _RenderAnchor,
    join: _RenderAnchor,
    main_branch: _RenderAnchor,
    obstacles: list[_RenderAnchor],
    *,
    graph,
    positions: list,
    join_index: int,
) -> list[tuple[float, float]] | None:
    """Route a fork/join side feed above dotted frames and around the main branch."""
    exit_y = _connector_source_bottom_exit_y(side)
    entry_y = _connector_target_top_entry_y(join)
    stub_y = exit_y - CONNECTOR_EXIT_STUB
    target_frame = next(
        (frame for frame in graph.inline_frames if join_index in frame.node_indices),
        None,
    )
    if target_frame is not None:
        route_y = _inline_frame_top_member_route_y(
            side,
            join,
            target_frame,
            positions,
            graph,
        )
    else:
        route_y = exit_y - CONNECTOR_EXIT_STUB
        for obstacle in obstacles:
            if obstacle.bottom <= route_y + PARALLEL_CONNECTOR_COORD_EPS:
                route_y = max(route_y, obstacle.top + CONNECTOR_OBSTACLE_MARGIN)

    if side.cx >= main_branch.cx:
        gutter_x = (
            max(side.right, main_branch.right)
            + CONNECTOR_OBSTACLE_MARGIN
            + CONNECTOR_ATTACHED_BOX_MARGIN
        )
        port_x = join.right - CONNECTOR_ATTACHED_BOX_MARGIN
        for obstacle in obstacles:
            if port_x <= obstacle.right + CONNECTOR_OBSTACLE_MARGIN:
                port_x = max(
                    port_x,
                    obstacle.right
                    + CONNECTOR_OBSTACLE_MARGIN
                    + 2 * PARALLEL_CONNECTOR_COORD_EPS,
                )
        port_x = min(port_x, join.right - CONNECTOR_ATTACHED_BOX_MARGIN)
        if port_x <= join.left + CONNECTOR_ATTACHED_BOX_MARGIN:
            return None
    else:
        gutter_x = (
            min(side.left, main_branch.left)
            - CONNECTOR_OBSTACLE_MARGIN
            - CONNECTOR_ATTACHED_BOX_MARGIN
        )
        port_x = join.left + CONNECTOR_ATTACHED_BOX_MARGIN
        for obstacle in obstacles:
            if port_x >= obstacle.left - CONNECTOR_OBSTACLE_MARGIN:
                port_x = min(
                    port_x,
                    obstacle.left
                    - CONNECTOR_OBSTACLE_MARGIN
                    - 2 * PARALLEL_CONNECTOR_COORD_EPS,
                )
        port_x = max(port_x, join.left + CONNECTOR_ATTACHED_BOX_MARGIN)
        if port_x >= join.right - CONNECTOR_ATTACHED_BOX_MARGIN:
            return None

    candidate = _ensure_orthogonal_connector_path(
        [
            (side.cx, exit_y),
            (side.cx, stub_y),
            (gutter_x, stub_y),
            (gutter_x, route_y),
            (port_x, route_y),
            (port_x, entry_y),
        ]
    )
    if _path_hits_obstacles(candidate, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN):
        return None
    return candidate


def _apply_fork_join_side_branch_bypass_routes(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    merge_entry_x: dict[tuple[int, int], float],
) -> None:
    """Route fork/join side branches around the main-branch tile into the join."""
    from visualizer.computation_graph import _find_fork_join_clusters

    for cluster in _find_fork_join_clusters(graph):
        link = (cluster.side_source, cluster.join)
        side = anchors.get(cluster.side_source)
        join = anchors.get(cluster.join)
        main_branch = anchors.get(cluster.main_branch)
        if side is None or join is None or main_branch is None:
            continue
        current = link_paths.get(link)
        if current is None:
            continue
        frame_violation = _connector_path_violates_inline_frame_bounds(
            current,
            graph,
            positions,
            src=cluster.side_source,
            tgt=cluster.join,
        )
        obstacles = _connector_block_obstacles(
            anchors,
            src=cluster.side_source,
            tgt=cluster.join,
            label_obstacles=label_obstacles,
            graph=graph,
            positions=positions,
            link_key=link,
        )
        crossings = _connector_path_crossing_count(link_paths, link)
        if (
            frame_violation is None
            and not crossings
            and not _path_hits_obstacles(
                current,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            )
        ):
            continue
        candidate = _fork_join_side_branch_bypass_route(
            side,
            join,
            main_branch,
            obstacles,
            graph=graph,
            positions=positions,
            join_index=cluster.join,
        )
        if candidate is None:
            continue
        if _connector_path_violates_inline_frame_bounds(
            candidate,
            graph,
            positions,
            src=cluster.side_source,
            tgt=cluster.join,
        ) is not None:
            continue
        # A row too shallow for the exit stub leaves the detour no choice but to turn on the
        # source edge, which reads as the connector running along the tile.
        if (
            _connector_turn_before_clearing_source(
                candidate,
                y_exit=_connector_source_exit_y(graph, cluster.side_source, side),
                source_cx=side.cx,
            )
            is not None
        ):
            continue
        link_paths[link] = candidate
        merge_entry_x[link] = candidate[-2][0]


def _apply_sideproducer_merge_routes(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    merge_entry_x: dict[tuple[int, int], float],
    merge_link_bus: dict[tuple[int, int], float],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
) -> None:
    """Route sideproducer feeds around intermediate tiles into distant merge targets."""
    for link_key, current in list(link_paths.items()):
        src, tgt = link_key
        if not graph.nodes[src].key.startswith("sideproducer:"):
            continue
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None or len(current) < 2:
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
        hits_obstacles = _path_hits_obstacles(
            current,
            obstacles,
            margin=CONNECTOR_OBSTACLE_MARGIN,
        )
        if (
            abs(target.cx - source.cx) <= TOP_ENTRY_PORT_GAP + 0.15
            and not hits_obstacles
        ):
            continue
        if not hits_obstacles:
            continue
        candidates: list[list[tuple[float, float]] | None] = [
            _horizontal_departure_side_bypass_route(
                source,
                target,
                obstacles,
                tee_y=source_bus.get(src),
            ),
            _same_column_side_gutter_detour(source, target, obstacles),
        ]
        bus_y = merge_link_bus.get(
            link_key,
            target_bus.get(tgt, _connector_min_bus_y_above_target(target)),
        )
        y_exit = _connector_source_bottom_exit_y(source)
        if target.cx >= source.cx:
            gutter_x = (
                min(source.left, target.left)
                - CONNECTOR_OBSTACLE_MARGIN
                - CONNECTOR_ATTACHED_BOX_MARGIN
            )
            port_x = merge_entry_x.get(link_key, target.cx)
            port_x = max(
                target.left + CONNECTOR_ATTACHED_BOX_MARGIN,
                min(port_x, target.cx),
            )
        else:
            gutter_x = (
                max(source.right, target.right)
                + CONNECTOR_OBSTACLE_MARGIN
                + CONNECTOR_ATTACHED_BOX_MARGIN
            )
            port_x = merge_entry_x.get(link_key, target.cx)
            port_x = min(
                target.right - CONNECTOR_ATTACHED_BOX_MARGIN,
                max(port_x, target.cx),
            )
        candidates.append(
            _ensure_orthogonal_connector_path(
                [
                    (source.cx, source.bottom),
                    (source.cx, y_exit),
                    (source.cx, bus_y),
                    (gutter_x, bus_y),
                    (port_x, bus_y),
                    (port_x, target.top),
                ]
            )
        )
        for candidate in candidates:
            if candidate is None:
                continue
            candidate = _clear_connector_path_obstacles(candidate, obstacles)
            if _path_hits_obstacles(
                candidate,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
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
            link_paths[link_key] = candidate
            break


def _find_multiply_stem_side_branches(
    graph,
    incoming: dict[int, list[tuple[int, int]]],
) -> list[tuple[int, int, int]]:
    """Return (branch_src, side_step, join) triples that branch off a shared stem.

    The shape is a fan-out whose two prongs rejoin one step apart: a source feeds the join
    directly and also feeds a single-input step that feeds the same join. The direct prong
    is the stem, so the detour prong should tee off it rather than open its own exit.
    """
    branches: list[tuple[int, int, int]] = []
    predecessors_of = {
        tgt: [src for src, _ in links if (src, tgt) not in graph.dashed_links]
        for tgt, links in incoming.items()
    }
    for join in range(len(graph.nodes)):
        predecessors = predecessors_of.get(join, [])
        if len(predecessors) < 2:
            continue
        for side_step in predecessors:
            sources = predecessors_of.get(side_step, [])
            if len(sources) != 1 or sources[0] not in predecessors:
                continue
            branches.append((sources[0], side_step, join))
    return branches


def _repair_multiply_stem_side_branch_tees(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
) -> None:
    """Tee a side step's feed off the stem it shares with the join, not its own exit."""
    for branch_src, side_step, join_index in _find_multiply_stem_side_branches(graph, incoming):
        link_key = (branch_src, side_step)
        stem_key = (branch_src, join_index)
        branch = anchors.get(branch_src)
        step = anchors.get(side_step)
        join = anchors.get(join_index)
        if branch is None or step is None or join is None:
            continue
        if link_key not in link_paths or stem_key not in link_paths:
            continue
        # The stem carries the shared exit, so its entry port has to stay on the join's top
        # edge; otherwise the tee would leave the connector running along the tile.
        join_entry_y = _connector_target_top_entry_y(join)
        stem_port_x = max(join.cx, step.right + CONNECTOR_OBSTACLE_MARGIN)
        if stem_port_x > join.right + PARALLEL_CONNECTOR_COORD_EPS:
            continue
        y_exit = _connector_source_bottom_exit_y(branch)
        entry_y = _connector_target_top_entry_y(step)
        approach_y = entry_y + CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN
        side_prong = _ensure_orthogonal_connector_path(
            [
                (branch.cx, y_exit),
                (branch.cx, approach_y),
                (step.cx, approach_y),
                (step.cx, entry_y),
            ]
        )
        stem = _ensure_orthogonal_connector_path(
            [
                (branch.cx, y_exit),
                (branch.cx, join_entry_y),
                (stem_port_x, join_entry_y),
            ]
        )
        if not _connector_path_clear_of_blocks(
            stem,
            source=branch,
            target=join,
            obstacles=_connector_block_obstacles(
                anchors,
                src=branch_src,
                tgt=join_index,
                label_obstacles=label_obstacles,
                graph=graph,
                positions=positions,
                link_key=stem_key,
            ),
        ):
            continue
        # The prong that leaves the column has to clear everything it passes.
        if not _connector_path_clear_of_blocks(
            side_prong,
            source=branch,
            target=step,
            obstacles=_connector_block_obstacles(
                anchors,
                src=branch_src,
                tgt=side_step,
                label_obstacles=label_obstacles,
                graph=graph,
                positions=positions,
                link_key=link_key,
            ),
        ):
            continue
        link_paths[link_key] = side_prong
        link_paths[stem_key] = stem


def _apply_stacked_same_side_fanout_routes(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    outgoing: dict[int, list[tuple[int, int]]],
    merge_entry_x: dict[tuple[int, int], float],
) -> None:
    """Keep stacked children on one side attached to one shared fork.

    The top child owns the common approach. Lower children branch at that child's
    top-entry corner and use a side gutter around the intervening stack.
    """
    for src, links in outgoing.items():
        source = anchors.get(src)
        if source is None or len(links) < 2:
            continue
        side_links = [
            link
            for link in links
            if link[1] in anchors
            and abs(anchors[link[1]].cx - source.cx) > TOP_ENTRY_PORT_GAP
        ]
        for link in side_links:
            target = anchors[link[1]]
            group = [
                other
                for other in side_links
                if abs(anchors[other[1]].cx - target.cx) <= TOP_ENTRY_PORT_GAP
                and (anchors[other[1]].cx - source.cx)
                * (target.cx - source.cx)
                > 0
            ]
            if len(group) < 2:
                continue
            ordered = sorted(group, key=lambda item: anchors[item[1]].top, reverse=True)
            top_link = ordered[0]
            top_points = link_paths.get(top_link)
            if top_points is None or len(top_points) < 3:
                continue
            fork_x, fork_y = top_points[-2]
            shared_prefix = list(top_points[:-1])
            for lower_link in ordered[1:]:
                lower = anchors[lower_link[1]]
                approach_y = _connector_min_bus_y_above_target(lower)
                if approach_y >= fork_y - PARALLEL_CONNECTOR_COORD_EPS:
                    continue
                entry_x = merge_entry_x.get(lower_link, lower.cx)
                if not lower.left < entry_x < lower.right:
                    entry_x = lower.cx
                obstacles = _connector_block_obstacles(
                    anchors,
                    src=src,
                    tgt=lower_link[1],
                    label_obstacles=label_obstacles,
                    graph=graph,
                    positions=positions,
                    link_key=lower_link,
                )
                stack_obstacles = [
                    obstacle
                    for obstacle in obstacles
                    if obstacle.top >= approach_y
                    and obstacle.bottom <= fork_y
                    and obstacle.left < max(fork_x, lower.right)
                    and obstacle.right > min(fork_x, lower.left)
                ]
                left_gutter = min(
                    lower.left,
                    *(obstacle.left for obstacle in stack_obstacles),
                ) - CONNECTOR_OBSTACLE_MARGIN - CONNECTOR_ATTACHED_BOX_MARGIN
                right_gutter = max(
                    lower.right,
                    *(obstacle.right for obstacle in stack_obstacles),
                ) + CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN
                gutters = (
                    (left_gutter, right_gutter)
                    if target.cx < source.cx
                    else (right_gutter, left_gutter)
                )
                for gutter_x in gutters:
                    candidate = _ensure_orthogonal_connector_path(
                        [
                            *shared_prefix,
                            (gutter_x, fork_y),
                            (gutter_x, approach_y),
                            (entry_x, approach_y),
                            (entry_x, lower.top),
                        ]
                    )
                    if _path_hits_obstacles(
                        candidate,
                        obstacles,
                        margin=CONNECTOR_OBSTACLE_MARGIN,
                    ):
                        continue
                    if (
                        _connector_path_violates_inline_frame_bounds(
                            candidate,
                            graph,
                            positions,
                            src=src,
                            tgt=lower_link[1],
                        )
                        is not None
                    ):
                        continue
                    link_paths[lower_link] = candidate
                    break


def _target_blocks_same_column_bypass(
    positions: list,
    anchors: dict[int, _RenderAnchor],
    *,
    tgt: int,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
) -> bool:
    """True when a same-column feeder of this tile also feeds a tile below it."""
    target_anchor = anchors.get(tgt)
    if target_anchor is None:
        return False
    for src, _ in incoming.get(tgt, []):
        if src not in anchors:
            continue
        if abs(positions[src].cx - target_anchor.cx) >= TOP_ENTRY_PORT_GAP:
            continue
        for _, below in outgoing.get(src, []):
            if below == tgt or below not in anchors:
                continue
            if anchors[below].top <= target_anchor.bottom + PARALLEL_CONNECTOR_COORD_EPS:
                return True
    return False


def _assign_spread_merge_entry_x(
    spread_links: list[tuple[int, int]],
    target_anchor: _RenderAnchor,
    target_pos,
    positions: list,
    anchors: dict[int, _RenderAnchor],
    merge_entry_x: dict[tuple[int, int], float],
) -> None:
    """Spread top-edge ports near the block center, preserving feeder order."""
    del target_pos, anchors
    if not spread_links:
        return
    if len(spread_links) == 1:
        merge_entry_x[spread_links[0]] = target_anchor.cx
        return

    sorted_links = sorted(spread_links, key=_spread_link_port_order_key(positions))
    span = _center_spread_top_entry_span(len(sorted_links), target_anchor)
    left = target_anchor.cx - span / 2
    source_cxs = [positions[link[0]].cx for link in sorted_links]
    min_src = min(source_cxs)
    max_src = max(source_cxs)
    count = len(sorted_links)
    min_spread = TOP_ENTRY_PORT_GAP
    desired: list[tuple[tuple[int, int], float]] = []

    for index, link in enumerate(sorted_links):
        src_cx = positions[link[0]].cx
        if abs(src_cx - target_anchor.cx) <= 0.08:
            desired.append((link, target_anchor.cx))
            continue
        if abs(max_src - min_src) <= PARALLEL_CONNECTOR_COORD_EPS:
            ratio = index / (count - 1)
        else:
            ratio = (positions[link[0]].cx - min_src) / (max_src - min_src)
        port_x = left + ratio * span
        if abs(port_x - target_anchor.cx) < min_spread - PARALLEL_CONNECTOR_COORD_EPS:
            port_x = (
                target_anchor.cx + min_spread
                if src_cx >= target_anchor.cx
                else target_anchor.cx - min_spread
            )
        desired.append((link, port_x))

    resolved: list[tuple[tuple[int, int], float]] = []
    for link, port_x in desired:
        if not resolved:
            resolved.append((link, port_x))
            continue
        prev_x = resolved[-1][1]
        if port_x - prev_x < min_spread - PARALLEL_CONNECTOR_COORD_EPS:
            port_x = prev_x + min_spread
        resolved.append((link, port_x))

    for link, port_x in resolved:
        merge_entry_x[link] = port_x
    _swap_merge_entry_x_if_crossing(sorted_links, merge_entry_x, positions)


def _fanout_links_excluding_bypasses(
    graph,
    link_group: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """All fan-out links participate in shared top-entry buses."""
    del graph
    return list(link_group)


def _plan_inline_bypass_bus_x(
    graph,
    links: list[tuple[int, int]],
    anchors: dict[int, _RenderAnchor],
    positions: list | None = None,
    *,
    gap: float = 0.04,
    channel_gap: float = PARALLEL_CONNECTOR_CHANNEL_GAP,
) -> dict[tuple[int, int], float]:
    """Assign distinct in-frame bus columns for top-entry skip connectors."""
    drawn = {(src, tgt) for src, tgt in links}
    bus_x_map: dict[tuple[int, int], float] = {}
    if positions is not None:
        for frame in getattr(graph, "inline_frames", None) or []:
            for link_key, corridor_x in _inline_frame_bypass_corridors(
                graph, frame, positions, channel_gap=channel_gap
            ).items():
                if link_key in drawn:
                    bus_x_map[link_key] = corridor_x

    del anchors, gap
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
    coord_tol: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> bool:
    """True when two axis-aligned connector segments occupy the same channel.

    Compare the coordinates directly rather than by bucket: rounding to a grid lets two runs
    a hair apart fall either side of a boundary and pass as separate channels, when on the
    page they are one thick line.
    """
    ori_a, coord_a, lo_a, hi_a, _ = seg_a
    ori_b, coord_b, lo_b, hi_b, _ = seg_b
    if link_a == link_b or ori_a != ori_b:
        return False
    if abs(coord_a - coord_b) > coord_tol:
        return False
    return _ranges_overlap(lo_a, hi_a, lo_b, hi_b)


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
    side_entry_links = _side_entry_links(graph)
    violations: list[tuple[tuple[int, int], str]] = []
    for link_key, points in link_paths.items():
        src, tgt = link_key
        if link_key in side_entry_links:
            continue
        if (
            graph.nodes[src].key.startswith("sideproducer:")
            and graph.nodes[tgt].key.startswith("sidefeed:")
            and _is_multiply_label(graph.nodes[tgt].label)
        ):
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
        if _path_hits_obstacles(points, obstacles, margin=CONNECTOR_OBSTACLE_MARGIN):
            if _is_frame_tail_below_border_path(
                graph,
                positions,
                src=src,
                points=points,
            ) and not _path_penetrates_obstacle_tiles(
                points,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            ):
                continue
            violations.append((link_key, "touches intermediate node"))
    return violations


def _find_connector_entry_approach_violations(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
) -> list[tuple[tuple[int, int], str]]:
    """Return links that sink below their target's top edge and climb back into the port.

    A top entry reads as "this feeds the tile from above". A run that drops past the top
    edge, travels sideways under it and comes back up reads instead as though it left the
    tile, which is the opposite of what the edge means.

    What exempts a link is docking on a side edge, not being eligible to: a link the layout
    marked as a side feed but which ended up landing on the top edge is read as a top entry
    by anyone looking at it, and has to arrive like one.
    """
    side_entry_links = _side_entry_links(graph)
    violations: list[tuple[tuple[int, int], str]] = []
    for link_key, points in link_paths.items():
        target = anchors.get(link_key[1])
        if target is None:
            continue
        if link_key in side_entry_links and _connector_path_docks_on_a_side_edge(
            points, target
        ):
            continue
        if _connector_entry_approach_below_target_corridor(points, target):
            violations.append((link_key, "entry approaches the top port from below"))
    return violations


def _find_connector_entry_port_violations(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
) -> list[tuple[tuple[int, int], str]]:
    """Return feeds into one tile whose entry ports crowd or contradict their approach.

    Two edges into a tile carry two different tensors, so they need ports the reader can
    tell apart, and the ports have to sit in the same left-to-right order as the runs that
    feed them. Ports out of order make the two feeds cross under the tile's own edge.
    """
    side_entry_links = _side_entry_links(graph)
    by_target: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for link_key in link_paths:
        if link_key in side_entry_links:
            continue
        by_target[link_key[1]].append(link_key)
    violations: list[tuple[tuple[int, int], str]] = []
    for tgt, feeds in sorted(by_target.items()):
        if anchors.get(tgt) is None or len(feeds) < 2:
            continue
        ports = {link_key: link_paths[link_key][-1][0] for link_key in feeds}
        approaches = {
            link_key: _connector_entry_approach_x(link_paths[link_key]) for link_key in feeds
        }
        for index, first in enumerate(sorted(feeds)):
            for second in sorted(feeds)[index + 1 :]:
                if first[0] == second[0]:
                    continue
                if abs(ports[first] - ports[second]) < TOP_ENTRY_PORT_GAP - 1e-6:
                    violations.append((first, f"entry port crowds {second}"))
                    continue
                lead = approaches[first] - approaches[second]
                if abs(lead) <= PARALLEL_CONNECTOR_COORD_EPS:
                    continue
                if lead * (ports[first] - ports[second]) < 0:
                    violations.append((first, f"entry port order contradicts {second}"))
    return violations


def _connector_entry_approach_x(points: list[tuple[float, float]]) -> float:
    """X the final drop onto the port is fed from, or the port itself for a straight run."""
    entry_x = points[-1][0]
    for index in range(len(points) - 2, -1, -1):
        if abs(points[index][0] - entry_x) > PARALLEL_CONNECTOR_COORD_EPS:
            return points[index][0]
    return entry_x


def _edge_port_plan(
    approaches: list[tuple[tuple[int, int], float]],
    anchor: _RenderAnchor,
) -> dict[tuple[int, int], float]:
    """Lay one port per link along a tile edge, in the order the links run past it.

    Each link would rather meet the edge straight down the column it arrives on or leaves
    for, so ports start at those columns and are only pushed apart far enough to stay
    legible. Pushing left to right keeps the ports in that same order, which is what stops
    two runs crossing under the tile's own edge to reach ports on the wrong side.
    """
    lo = anchor.left + CONNECTOR_ATTACHED_BOX_MARGIN
    hi = anchor.right - CONNECTOR_ATTACHED_BOX_MARGIN
    if hi <= lo or not approaches:
        return {}
    count = len(approaches)
    step = min(TOP_ENTRY_PORT_GAP, (hi - lo) / (count - 1)) if count > 1 else 0.0
    ordered = sorted(approaches, key=lambda item: (item[1], item[0]))
    ports: list[float] = []
    for _link_key, approach_x in ordered:
        wanted = min(max(approach_x, lo), hi)
        if ports:
            wanted = max(wanted, ports[-1] + step)
        ports.append(wanted)
    overflow = ports[-1] - hi
    if overflow > 0:
        ports = [port - overflow for port in ports]
    if ports[0] < lo:
        ports = [lo + index * step for index in range(count)]
    return {link_key: ports[index] for index, (link_key, _approach) in enumerate(ordered)}


def _reseat_connector_entry_port(
    points: list[tuple[float, float]],
    *,
    target: _RenderAnchor,
    port_x: float,
) -> list[tuple[float, float]] | None:
    """Move a top-edge drop onto a different port, leaving the rest of the route alone."""
    if len(points) < 2:
        return None
    entry_x, entry_y = points[-1]
    if abs(entry_x - port_x) <= PARALLEL_CONNECTOR_COORD_EPS:
        return list(points)
    if abs(entry_y - _connector_target_top_entry_y(target)) > PARALLEL_CONNECTOR_COORD_EPS:
        return None
    path = list(points)
    drop_start = len(path) - 2
    if abs(path[drop_start][0] - entry_x) > PARALLEL_CONNECTOR_COORD_EPS:
        return None
    if drop_start == 0:
        # A run straight down the port's own column has no corner to slide, so it needs a
        # jog, and the jog belongs in the corridor the target keeps above its top edge.
        jog_y = _connector_min_bus_y_above_target(target)
        if jog_y >= path[0][1] - PARALLEL_CONNECTOR_COORD_EPS:
            return None
        path.insert(1, (entry_x, jog_y))
        drop_start = 1
    path[drop_start] = (port_x, path[drop_start][1])
    path[-1] = (port_x, entry_y)
    return _dedupe_polyline_points(_ensure_orthogonal_connector_path(path))


def _connector_exit_departure_x(points: list[tuple[float, float]]) -> float:
    """The column a route heads for once it has cleared its source's bottom edge."""
    if not points:
        return 0.0
    exit_x = points[0][0]
    for x, _y in points[1:]:
        if abs(x - exit_x) > PARALLEL_CONNECTOR_COORD_EPS:
            return x
    return exit_x


def _reseat_connector_exit_port(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    port_x: float,
) -> list[tuple[float, float]] | None:
    """Move a bottom-edge departure onto a different port, leaving the rest of the route alone."""
    if len(points) < 3:
        # A drop straight into the target has no corner to slide: moving its foot would pull
        # the run off the port the target is expecting it at.
        return None
    exit_x, exit_y = points[0]
    if abs(exit_x - port_x) <= PARALLEL_CONNECTOR_COORD_EPS:
        return list(points)
    if abs(exit_y - _connector_source_bottom_exit_y(source)) > PARALLEL_CONNECTOR_COORD_EPS:
        return None
    path = list(points)
    if abs(path[1][0] - exit_x) > PARALLEL_CONNECTOR_COORD_EPS:
        # Already departs sideways rather than dropping, so there is no stem to move.
        return None
    path[0] = (port_x, exit_y)
    path[1] = (port_x, path[1][1])
    return _dedupe_polyline_points(_ensure_orthogonal_connector_path(path))


def _connector_horizontal_run_spans(
    points: list[tuple[float, float]],
) -> list[tuple[int, float, float, float]]:
    """Drawn horizontal runs as (index of the run's first vertex, y, low x, high x)."""
    runs: list[tuple[int, float, float, float]] = []
    for index, ((x1, y1), (x2, y2)) in enumerate(zip(points, points[1:])):
        if (
            abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(x1 - x2) > PARALLEL_CONNECTOR_COORD_EPS
        ):
            runs.append((index, (y1 + y2) / 2, min(x1, x2), max(x1, x2)))
    return runs


def _shift_connector_horizontal_run(
    points: list[tuple[float, float]],
    *,
    index: int,
    row_y: float,
) -> list[tuple[float, float]] | None:
    """Move one horizontal run of a route to another row, keeping both ends' columns."""
    if not 0 <= index < len(points) - 1:
        return None
    path = list(points)
    if index == 0 or index + 1 == len(path) - 1:
        # The run is bounded by an endpoint, which is pinned to a tile edge port, so sliding
        # it would drag the connector off that port.
        return None
    path[index] = (path[index][0], row_y)
    path[index + 1] = (path[index + 1][0], row_y)
    cleaned = _dedupe_polyline_points(_ensure_orthogonal_connector_path(path))
    if _connector_path_reverses_vertical_direction(cleaned):
        return None
    return cleaned


def _connector_vertical_run_spans(
    points: list[tuple[float, float]],
) -> list[tuple[int, float, float, float]]:
    """Drawn vertical runs as (index of the run's first vertex, x, low y, high y)."""
    runs: list[tuple[int, float, float, float]] = []
    for index, ((x1, y1), (x2, y2)) in enumerate(zip(points, points[1:])):
        if (
            abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS
        ):
            runs.append((index, (x1 + x2) / 2, min(y1, y2), max(y1, y2)))
    return runs


def _shift_connector_vertical_run(
    points: list[tuple[float, float]],
    *,
    index: int,
    column_x: float,
    source: _RenderAnchor,
    target: _RenderAnchor,
) -> list[tuple[float, float]] | None:
    """Move one vertical run of a route to another column, keeping both ends' rows.

    The runs at either end carry the ports, so moving one slides the port along the tile
    edge it sits on; that is allowed as far as the edge reaches and no further.
    """
    if not 0 <= index < len(points) - 1:
        return None
    path = list(points)
    if index == 0 and not source.left <= column_x <= source.right:
        return None
    if index + 1 == len(path) - 1 and not target.left <= column_x <= target.right:
        return None
    path[index] = (column_x, path[index][1])
    path[index + 1] = (column_x, path[index + 1][1])
    cleaned = _dedupe_polyline_points(_ensure_orthogonal_connector_path(path))
    if _connector_path_reverses_vertical_direction(cleaned):
        return None
    return cleaned


def _connector_path_is_orthogonal(points: list[tuple[float, float]]) -> bool:
    """True when every drawn run is horizontal or vertical."""
    return all(
        abs(first_x - second_x) <= PARALLEL_CONNECTOR_COORD_EPS
        or abs(first_y - second_y) <= PARALLEL_CONNECTOR_COORD_EPS
        for (first_x, first_y), (second_x, second_y) in zip(points, points[1:])
    )


def _connector_path_reverses_vertical_direction(
    points: list[tuple[float, float]],
) -> bool:
    """True when a route climbs after descending, which reads as a wire doubling back."""
    directions = [
        1 if second_y > first_y else -1
        for (_first_x, first_y), (_second_x, second_y) in zip(points, points[1:])
        if abs(second_y - first_y) > PARALLEL_CONNECTOR_COORD_EPS
    ]
    return any(
        first != second for first, second in zip(directions, directions[1:])
    )


def _connector_path_length(points: list[tuple[float, float]]) -> float:
    return sum(
        abs(second_x - first_x) + abs(second_y - first_y)
        for (first_x, first_y), (second_x, second_y) in zip(points, points[1:])
    )


def _direct_connector_route_candidates(
    points: list[tuple[float, float]],
    *,
    source: _RenderAnchor,
    target: _RenderAnchor,
) -> list[list[tuple[float, float]]]:
    """Plain step-across routes between the ports a link already uses."""
    exit_x, exit_y = points[0]
    entry_x, entry_y = points[-1]
    if abs(exit_x - entry_x) <= PARALLEL_CONNECTOR_COORD_EPS:
        return [[(exit_x, exit_y), (entry_x, entry_y)]]
    floor = _connector_min_bus_y_above_target(target)
    rows = [
        exit_y - CONNECTOR_EXIT_STUB,
        floor,
        *(floor + step * PARALLEL_CONNECTOR_CHANNEL_GAP for step in range(1, 5)),
    ]
    candidates = []
    for row in rows:
        if not entry_y < row < exit_y:
            continue
        candidates.append([(exit_x, exit_y), (exit_x, row), (entry_x, row), (entry_x, entry_y)])
    return candidates


def _shorten_wandering_connector_routes(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
) -> None:
    """Replace a route that travels far further than it needs with one that steps across.

    A connector that leaves its source, runs off past everything on the page and comes back
    reads as a wire that goes nowhere. No fault count objects to it, because such a route
    can cross and overlap nothing at all, so it has to be found by how far it travels
    compared with the shortest way between the two ports it already uses.
    """
    metrics = {
        "graph": graph,
        "anchors": anchors,
        "label_obstacles": label_obstacles,
        "positions": positions,
        "incoming": incoming,
        "outgoing": outgoing,
        "target_bus": target_bus,
        "source_bus": source_bus,
        "merge_link_bus": merge_link_bus,
    }
    for link_key in sorted(link_paths):
        points = link_paths[link_key]
        source = anchors.get(link_key[0])
        target = anchors.get(link_key[1])
        if source is None or target is None or len(points) < 4:
            continue
        reach = abs(points[-1][0] - points[0][0]) + abs(points[-1][1] - points[0][1])
        drawn = _connector_path_length(points)
        if drawn <= 2 * reach + CONNECTOR_EXIT_STUB:
            continue
        obstacles = _connector_block_obstacles(
            anchors,
            src=link_key[0],
            tgt=link_key[1],
            label_obstacles=label_obstacles,
            graph=graph,
            positions=positions,
            link_key=link_key,
        )
        before, _ = _connector_violation_links(link_paths, **metrics)
        for candidate in sorted(
            _direct_connector_route_candidates(points, source=source, target=target),
            key=_connector_path_length,
        ):
            admissible = _admissible_connector_route(
                candidate,
                source=source,
                target=target,
                obstacles=obstacles,
                graph=graph,
                link_key=link_key,
                positions=positions,
            )
            if admissible is None or _connector_path_length(admissible) >= drawn:
                continue
            trial = dict(link_paths)
            trial[link_key] = admissible
            after, _ = _connector_violation_links(trial, **metrics)
            if after > before:
                continue
            link_paths[link_key] = admissible
            break


def _nest_source_fanout_bus_rows(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
) -> None:
    """Give each leg of a fan-out its own row, nested so that no two legs cross.

    A leg that reaches further from the source has to pass over the columns the nearer legs
    turn down, so it takes the row closest to the source and the nearer legs sit beneath it.
    Because the exit ports run in the same order as the columns the legs reach, the outer
    leg's row ends short of every inner leg's port, and its own turn down is outside every
    inner leg's row: neither can meet the other. Legs heading opposite ways never meet at
    all, so each direction is nested on its own.
    """
    metrics = {
        "graph": graph,
        "anchors": anchors,
        "label_obstacles": label_obstacles,
        "positions": positions,
        "incoming": incoming,
        "outgoing": outgoing,
        "target_bus": target_bus,
        "source_bus": source_bus,
        "merge_link_bus": merge_link_bus,
    }
    gap = PARALLEL_CONNECTOR_CHANNEL_GAP
    legs_by_source: dict[int, list[tuple[tuple[int, int], int, float, float, float]]]
    legs_by_source = defaultdict(list)
    for link_key, points in link_paths.items():
        source = anchors.get(link_key[0])
        if source is None or len(points) < 3:
            continue
        if abs(points[0][1] - _connector_source_bottom_exit_y(source)) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        runs = _connector_horizontal_run_spans(points)
        if not runs:
            continue
        index, row_y, lo_x, hi_x = runs[0]
        legs_by_source[link_key[0]].append((link_key, index, row_y, lo_x, hi_x))

    for src, legs in sorted(legs_by_source.items()):
        if len(legs) < 2:
            continue
        for heading in (-1, 1):
            group = [
                leg
                for leg in legs
                if _sign_of(link_paths[leg[0]][leg[1] + 1][0] - link_paths[leg[0]][leg[1]][0])
                == heading
            ]
            if len(group) < 2:
                continue
            base_row = max(row_y for _key, _index, row_y, _lo, _hi in group)
            ordered = sorted(
                group,
                key=lambda leg: (
                    -heading * link_paths[leg[0]][leg[1] + 1][0],
                    leg[0],
                ),
            )
            moved: dict[tuple[int, int], list[tuple[float, float]]] = {}
            for depth, (link_key, index, _row_y, _lo_x, _hi_x) in enumerate(ordered):
                shifted = _shift_connector_horizontal_run(
                    link_paths[link_key],
                    index=index,
                    row_y=base_row - depth * gap,
                )
                if shifted is None or shifted == link_paths[link_key]:
                    continue
                admissible = _admissible_connector_route(
                    shifted,
                    source=anchors[link_key[0]],
                    target=anchors[link_key[1]],
                    obstacles=_connector_block_obstacles(
                        anchors,
                        src=link_key[0],
                        tgt=link_key[1],
                        label_obstacles=label_obstacles,
                        graph=graph,
                        positions=positions,
                        link_key=link_key,
                    ),
                    graph=graph,
                    link_key=link_key,
                    positions=positions,
                )
                if admissible is None:
                    continue
                moved[link_key] = admissible
            if not moved:
                continue
            before, _ = _connector_violation_links(link_paths, **metrics)
            trial = dict(link_paths)
            trial.update(moved)
            after, _ = _connector_violation_links(trial, **metrics)
            if after > before:
                continue
            link_paths.update(moved)


def _nest_source_fanout_legs(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
) -> None:
    """Order a fan-out's ports and rows together so no two of its legs cross.

    Which leg belongs on the row nearest the source, and which port it should leave by, are
    one decision: give a leg the outer row without the outer port and its neighbour's exit
    stem cuts straight through the row it was just given. Moving one at a time therefore
    never looks like progress and never gets taken, which is why this settles the whole
    fan-out at once.

    Nesting also depends on the shape of each leg, not just where it ends: a leg that runs
    out to a gutter, drops, and comes back inwards is not the outer leg it looks like from
    its first turn. Rather than rank the shapes, the orderings are tried and the one that
    crosses least is kept, which is the thing the ranking was a guess at anyway.
    """
    metrics = {
        "graph": graph,
        "anchors": anchors,
        "label_obstacles": label_obstacles,
        "positions": positions,
        "incoming": incoming,
        "outgoing": outgoing,
        "target_bus": target_bus,
        "source_bus": source_bus,
        "merge_link_bus": merge_link_bus,
    }
    for src, legs in sorted(_source_fanout_legs(link_paths, anchors=anchors).items()):
        if len(legs) < 2:
            continue
        source = anchors[src]
        before, _ = _connector_violation_links(link_paths, **metrics)
        best: dict[tuple[int, int], list[tuple[float, float]]] | None = None
        best_score = before
        for arrangement in _source_fanout_arrangements(legs, link_paths, source=source):
            for spacing in _source_fanout_row_spacings():
                trial_paths = _apply_source_fanout_arrangement(
                    arrangement,
                    link_paths,
                    source=source,
                    anchors=anchors,
                    label_obstacles=label_obstacles,
                    positions=positions,
                    graph=graph,
                    spacing=spacing,
                )
                if trial_paths is None:
                    continue
                trial = dict(link_paths)
                trial.update(trial_paths)
                score, _ = _connector_violation_links(trial, **metrics)
                if score < best_score:
                    best_score = score
                    best = trial_paths
                break
        if best is not None:
            link_paths.update(best)


def _source_fanout_row_spacings() -> list[float]:
    """How far apart to set a fan-out's rows, widest first.

    A channel is the spacing the reader wants, but a fan-out often sits in the gap between
    its source and whatever is under it, and that gap is frequently too shallow to hold a
    channel per leg. Rather than give up and leave the legs crossing, the rows close up as
    far as two lines can be told apart.
    """
    gap = PARALLEL_CONNECTOR_CHANNEL_GAP
    floor = PARALLEL_CONNECTOR_COORD_EPS * 1.5
    spacings = [gap, gap * 0.75, gap * 0.5, gap * 0.375, floor]
    return sorted({round(value, 9) for value in spacings if value >= floor}, reverse=True)


def _source_fanout_legs(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    anchors: dict[int, _RenderAnchor],
) -> dict[int, list[tuple[int, int]]]:
    """The links leaving each tile's bottom edge, for tiles more than one leaves."""
    legs: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for link_key, points in sorted(link_paths.items()):
        source = anchors.get(link_key[0])
        if source is None or len(points) < 2:
            continue
        if (
            abs(points[0][1] - _connector_source_bottom_exit_y(source))
            > PARALLEL_CONNECTOR_COORD_EPS
        ):
            continue
        legs[link_key[0]].append(link_key)
    return {src: keys for src, keys in legs.items() if len(keys) > 1}


def _source_fanout_arrangements(
    legs: list[tuple[int, int]],
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    source: _RenderAnchor,
    limit: int = 24,
) -> list[tuple[tuple[int, int], ...]]:
    """Port orders to try for a fan-out, left to right along the source's edge.

    Ordering the ports by where each leg is going is what keeps them from crossing under
    the edge, so that order is the first guess. It is only a guess, though, because a leg
    that runs out to a gutter and comes back is not going where its first turn suggests, so
    for a fan-out small enough to enumerate the other orders are tried behind it.
    """
    outwards = tuple(sorted(legs, key=lambda link_key: (link_paths[link_key][-1][0], link_key)))
    candidates = [outwards]
    if len(legs) <= 4:
        candidates += [
            order for order in itertools.permutations(outwards) if order != outwards
        ]
    else:
        # Too many legs to enumerate, so only neighbours trade places. That is enough for
        # the case the destination order gets wrong: a leg whose target sits between two
        # ports wants the nearer of them, or it reaches over its neighbour's stem.
        for index in range(len(outwards) - 1):
            swapped = list(outwards)
            swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
            candidates.append(tuple(swapped))
    return candidates[:limit]


def _apply_source_fanout_arrangement(
    order: tuple[tuple[int, int], ...],
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    source: _RenderAnchor,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    graph,
    spacing: float,
) -> dict[tuple[int, int], list[tuple[float, float]]] | None:
    """Draw a fan-out with its ports in the given order and a row for each leg that needs one.

    Ports keep the columns the layout already chose and only change hands, so the fan-out
    stays where it was put. Rows are then handed out outermost leg first, and a leg only
    takes a row of its own when it would otherwise be drawn along another leg's: two legs
    reaching past each other need separating, but two reaching opposite ways never meet and
    can share.
    """
    ports = _edge_port_plan(
        [
            (link_key, column)
            for link_key, column in zip(
                order, sorted(link_paths[link_key][0][0] for link_key in order)
            )
        ],
        source,
    )
    if len(ports) != len(order):
        return None

    turns: dict[tuple[int, int], tuple[int, float, float]] = {}
    base_row: float | None = None
    for link_key in order:
        points = link_paths[link_key]
        runs = _connector_horizontal_run_spans(points)
        if not runs:
            continue
        index, row_y, _lo, _hi = runs[0]
        turns[link_key] = (index, row_y, points[index + 1][0])
        base_row = row_y if base_row is None else max(base_row, row_y)
    if base_row is None:
        return None

    depths = _pack_source_fanout_rows(turns, ports)
    if depths is None:
        return None
    drawn: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for link_key in order:
        points = link_paths[link_key]
        moved = points
        if link_key in turns:
            index = turns[link_key][0]
            shifted = _shift_connector_horizontal_run(
                moved, index=index, row_y=base_row - depths[link_key] * spacing
            )
            if shifted is None:
                return None
            moved = shifted
        reseated = _reseat_connector_exit_port(moved, source=source, port_x=ports[link_key])
        if reseated is not None:
            moved = reseated
        elif abs(moved[0][0] - ports[link_key]) > PARALLEL_CONNECTOR_COORD_EPS:
            # A leg dropping straight into its target cannot move its foot, so the port it
            # holds is the one the arrangement has to work around.
            continue
        moved = _straighten_connector_terminal_jog(moved, target=anchors[link_key[1]])
        admissible = _connector_route_no_worse_than(
            moved,
            points,
            source=source,
            target=anchors[link_key[1]],
            obstacles=_connector_block_obstacles(
                anchors,
                src=link_key[0],
                tgt=link_key[1],
                label_obstacles=label_obstacles,
                graph=graph,
                positions=positions,
                link_key=link_key,
            ),
            graph=graph,
            link_key=link_key,
            positions=positions,
        )
        if admissible is None:
            return None
        drawn[link_key] = admissible
    return drawn or None


def _straighten_connector_terminal_jog(
    points: list[tuple[float, float]],
    *,
    target: _RenderAnchor,
) -> list[tuple[float, float]]:
    """Drop straight into the target when the last sidestep is narrower than a channel.

    A step of less than a channel just before the port is a kink the reader sees as a wobble,
    and it puts the entry stem within a hair of whatever else is in that column. The general
    kink straightener cannot help here because moving this run means moving the port, which
    is only allowed while the port stays on the edge it feeds.
    """
    if len(points) < 3:
        return points
    arrival_x, _arrival_y = points[-3]
    entry_x, _entry_y = points[-1]
    if abs(entry_x - arrival_x) > PARALLEL_CONNECTOR_CHANNEL_GAP:
        return points
    if not target.left <= arrival_x <= target.right:
        return points
    straightened = _reseat_connector_entry_port(points, target=target, port_x=arrival_x)
    return points if straightened is None else straightened


def _pack_source_fanout_rows(
    turns: dict[tuple[int, int], tuple[int, float, float]],
    ports: dict[tuple[int, int], float],
) -> dict[tuple[int, int], int] | None:
    """Give each leg the shallowest row it can hold without meeting another leg.

    Legs are placed the furthest-reaching first, because that leg has to pass over where all
    the nearer ones turn down and so belongs closest to the source. A row is free for a leg
    when three things hold: no leg already on it is drawn along the same stretch, no leg
    above it reaches over the port this leg leaves by, and this leg reaches over no port
    below it. The last two are what a shared row costs: every leg's stem has to climb past
    the rows above it to reach the source.

    A leg whose port lies under another leg's stretch cannot be drawn at any depth, because
    dropping it lower only lengthens the stem that already has to cross. That is a fault in
    the order the ports were given, not in the rows, so it is reported as no packing at all
    and the caller is left to try another order.
    """
    stretches = {
        link_key: (
            min(ports[link_key], turn_x),
            max(ports[link_key], turn_x),
        )
        for link_key, (_index, _row, turn_x) in turns.items()
    }
    eps = PARALLEL_CONNECTOR_COORD_EPS
    placed: list[tuple[tuple[int, int], int]] = []
    depths: dict[tuple[int, int], int] = {}
    for link_key in sorted(
        stretches,
        key=lambda key: (-(stretches[key][1] - stretches[key][0]), key),
    ):
        lo, hi = stretches[link_key]
        port = ports[link_key]
        settled: int | None = None
        for depth in range(len(stretches)):
            if any(
                (
                    min(hi, other_hi) - max(lo, other_lo) > eps
                    if other_depth == depth
                    else (
                        other_lo - eps < port < other_hi + eps
                        if other_depth < depth
                        else lo - eps < ports[other] < hi + eps
                    )
                )
                for other, other_depth in placed
                for other_lo, other_hi in (stretches[other],)
            ):
                continue
            settled = depth
            break
        if settled is None:
            return None
        depths[link_key] = settled
        placed.append((link_key, settled))
    return depths


def _sign_of(value: float) -> int:
    if value > PARALLEL_CONNECTOR_COORD_EPS:
        return 1
    if value < -PARALLEL_CONNECTOR_COORD_EPS:
        return -1
    return 0


def _unstack_shared_horizontal_runs(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
) -> None:
    """Move one of any two runs drawn along the same row onto a row of its own.

    Two runs on one row are one line on the page. The shorter of the pair is the cheaper to
    move, so it is the one offered a new row, and rows are tried outwards from the one it
    holds so the route changes as little as the page allows. The ladder of rows offered gets
    finer than a channel because the gap between two tiles is sometimes only wide enough for
    two runs if they are packed against its edges.
    """
    metrics = {
        "graph": graph,
        "anchors": anchors,
        "label_obstacles": label_obstacles,
        "positions": positions,
        "incoming": incoming,
        "outgoing": outgoing,
        "target_bus": target_bus,
        "source_bus": source_bus,
        "merge_link_bus": merge_link_bus,
    }
    gap = PARALLEL_CONNECTOR_CHANNEL_GAP
    for _ in range(6):
        stacked = _stacked_horizontal_run_pairs(link_paths)
        if not stacked:
            return
        moved = False
        for link_key, run_index, row_y in stacked:
            source = anchors.get(link_key[0])
            target = anchors.get(link_key[1])
            if source is None or target is None:
                continue
            obstacles = _connector_block_obstacles(
                anchors,
                src=link_key[0],
                tgt=link_key[1],
                label_obstacles=label_obstacles,
                graph=graph,
                positions=positions,
                link_key=link_key,
            )
            before, _ = _connector_violation_links(link_paths, **metrics)
            before_runs = _stacked_run_pair_count(link_paths)
            span = next(
                (
                    (lo_x, hi_x)
                    for index, _row, lo_x, hi_x in _connector_horizontal_run_spans(
                        link_paths[link_key]
                    )
                    if index == run_index
                ),
                None,
            )
            if span is None:
                continue
            for candidate_y in _connector_row_candidates(
                row_y,
                lo_x=span[0],
                hi_x=span[1],
                obstacles=obstacles,
                gap=gap,
            ):
                shifted = _shift_connector_horizontal_run(
                    link_paths[link_key],
                    index=run_index,
                    row_y=candidate_y,
                )
                if shifted is None or shifted == link_paths[link_key]:
                    continue
                admissible = _connector_route_no_worse_than(
                    shifted,
                    link_paths[link_key],
                    source=source,
                    target=target,
                    obstacles=obstacles,
                    graph=graph,
                    link_key=link_key,
                    positions=positions,
                )
                if admissible is None:
                    continue
                trial = dict(link_paths)
                trial[link_key] = admissible
                after, _ = _connector_violation_links(trial, **metrics)
                if after > before:
                    continue
                if after == before and _stacked_run_pair_count(trial) >= before_runs:
                    continue
                link_paths[link_key] = admissible
                moved = True
                break
            if moved:
                break
        if not moved:
            return


def _unstack_shared_vertical_runs(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
) -> None:
    """Move one of any two runs drawn down the same column onto a column of its own.

    The mirror of unstacking rows, and the one that matters most where legs leave a tile by
    separate ports and then all turn into the same gutter, which undoes the separation the
    ports were there to give.
    """
    metrics = {
        "graph": graph,
        "anchors": anchors,
        "label_obstacles": label_obstacles,
        "positions": positions,
        "incoming": incoming,
        "outgoing": outgoing,
        "target_bus": target_bus,
        "source_bus": source_bus,
        "merge_link_bus": merge_link_bus,
    }
    gap = PARALLEL_CONNECTOR_CHANNEL_GAP
    for _ in range(6):
        stacked = _stacked_vertical_run_pairs(link_paths)
        if not stacked:
            return
        moved = False
        for link_key, run_index, column_x in stacked:
            source = anchors.get(link_key[0])
            target = anchors.get(link_key[1])
            if source is None or target is None:
                continue
            obstacles = _connector_block_obstacles(
                anchors,
                src=link_key[0],
                tgt=link_key[1],
                label_obstacles=label_obstacles,
                graph=graph,
                positions=positions,
                link_key=link_key,
            )
            before, _ = _connector_violation_links(link_paths, **metrics)
            before_runs = _stacked_run_pair_count(link_paths)
            span = next(
                (
                    (lo_y, hi_y)
                    for index, _col, lo_y, hi_y in _connector_vertical_run_spans(
                        link_paths[link_key]
                    )
                    if index == run_index
                ),
                None,
            )
            if span is None:
                continue
            for candidate_x in _connector_column_candidates(
                column_x,
                lo_y=span[0],
                hi_y=span[1],
                obstacles=obstacles,
                gap=gap,
            ):
                shifted = _shift_connector_vertical_run(
                    link_paths[link_key],
                    index=run_index,
                    column_x=candidate_x,
                    source=source,
                    target=target,
                )
                if shifted is None or shifted == link_paths[link_key]:
                    continue
                admissible = _connector_route_no_worse_than(
                    shifted,
                    link_paths[link_key],
                    source=source,
                    target=target,
                    obstacles=obstacles,
                    graph=graph,
                    link_key=link_key,
                    positions=positions,
                )
                if admissible is None:
                    continue
                trial = dict(link_paths)
                trial[link_key] = admissible
                after, _ = _connector_violation_links(trial, **metrics)
                if after > before:
                    continue
                if after == before and _stacked_run_pair_count(trial) >= before_runs:
                    continue
                link_paths[link_key] = admissible
                moved = True
                break
            if moved:
                break
        if not moved:
            return


def _connector_column_candidates(
    column_x: float,
    *,
    lo_y: float,
    hi_y: float,
    obstacles: list[_RenderAnchor],
    gap: float,
) -> list[float]:
    """Columns to try moving a run onto, nearest to the one it holds first."""
    fine = PARALLEL_CONNECTOR_COORD_EPS * 1.5
    columns = {column_x - step * gap for step in range(1, 7)}
    columns.update(column_x + step * gap for step in range(1, 7))
    columns.update(column_x - step * fine for step in range(1, 4))
    columns.update(column_x + step * fine for step in range(1, 4))
    clearance = CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN
    for obstacle in obstacles:
        if min(hi_y, obstacle.top) - max(lo_y, obstacle.bottom) <= 0:
            continue
        columns.add(obstacle.right + clearance)
        columns.add(obstacle.left - clearance)
    return sorted(columns, key=lambda candidate: abs(candidate - column_x))


def _connector_row_candidates(
    row_y: float,
    *,
    lo_x: float,
    hi_x: float,
    obstacles: list[_RenderAnchor],
    gap: float,
) -> list[float]:
    """Rows to try moving a run onto, nearest to the one it holds first.

    A channel apart is what the reader wants, so the ladder starts there. But the gap
    between two tiles is often only wide enough for two runs when both hug its edges, and
    stepping by a fixed amount steps straight over the one row that fits, so the rows just
    clear of every tile the run passes are offered too.
    """
    fine = PARALLEL_CONNECTOR_COORD_EPS * 1.5
    rows = {row_y - step * gap for step in range(1, 7)}
    rows.update(row_y + step * gap for step in range(1, 7))
    rows.update(row_y - step * fine for step in range(1, 4))
    rows.update(row_y + step * fine for step in range(1, 4))
    for obstacle in obstacles:
        if min(hi_x, obstacle.right) - max(lo_x, obstacle.left) <= 0:
            continue
        # Sitting exactly on the margin counts as touching, so clear it by the hair a
        # connector keeps between itself and a tile it attaches to.
        clearance = CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN
        rows.add(obstacle.top + clearance)
        rows.add(obstacle.bottom - clearance)
    return sorted(rows, key=lambda candidate: abs(candidate - row_y))


def _stacked_horizontal_run_pairs(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
) -> list[tuple[tuple[int, int], int, float]]:
    """Runs sharing a row with another link's run, shortest first, as (link, index, y)."""
    return _stacked_parallel_run_pairs(link_paths, spans=_connector_horizontal_run_spans)


def _stacked_vertical_run_pairs(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
) -> list[tuple[tuple[int, int], int, float]]:
    """Runs sharing a column with another link's run, shortest first, as (link, index, x)."""
    return _stacked_parallel_run_pairs(link_paths, spans=_connector_vertical_run_spans)


def _stacked_run_pair_count(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
) -> int:
    """How many drawn runs share their line with another link's run.

    The violation score counts pairs of links, which cannot see progress on a pair that
    doubles up in both axes: freeing the row still leaves the column, so the pair stays
    named and the score stays put. Counting runs shows the step forward that the score
    cannot.
    """
    return len(_stacked_horizontal_run_pairs(link_paths)) + len(
        _stacked_vertical_run_pairs(link_paths)
    )


def _stacked_parallel_run_pairs(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    spans,
) -> list[tuple[tuple[int, int], int, float]]:
    """Runs drawn along the same line as another link's, shortest first."""
    runs = [
        (link_key, index, coord, lo, hi)
        for link_key, points in link_paths.items()
        for index, coord, lo, hi in spans(points)
    ]
    stacked: list[tuple[float, tuple[int, int], int, float]] = []
    for link_a, run_a, coord_a, lo_a, hi_a in runs:
        for link_b, _run_b, coord_b, lo_b, hi_b in runs:
            if link_a == link_b:
                continue
            if abs(coord_a - coord_b) > PARALLEL_CONNECTOR_COORD_EPS:
                continue
            if not _ranges_overlap(lo_a, hi_a, lo_b, hi_b):
                continue
            stacked.append((hi_a - lo_a, link_a, run_a, coord_a))
            break
    stacked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(link_key, run_index, coord) for _length, link_key, run_index, coord in stacked]


def _spread_shared_source_exit_stems(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
) -> None:
    """Give every leg of a fan-out its own port on the source's bottom edge.

    Legs stacked on one column are drawn on top of each other, so the reader sees a single
    thick line that appears to stop where the legs part, and the only way to say the line
    divides rather than ends is a dot claiming a junction. One port per leg removes both
    problems: each leg is its own line from the edge onwards.
    """
    metrics = {
        "graph": graph,
        "anchors": anchors,
        "label_obstacles": label_obstacles,
        "positions": positions,
        "incoming": incoming,
        "outgoing": outgoing,
        "target_bus": target_bus,
        "source_bus": source_bus,
        "merge_link_bus": merge_link_bus,
    }
    by_source: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for link_key, points in link_paths.items():
        source = anchors.get(link_key[0])
        if source is None or len(points) < 2:
            continue
        if abs(points[0][1] - _connector_source_bottom_exit_y(source)) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        by_source[link_key[0]].append(link_key)

    for src, legs in sorted(by_source.items()):
        if len(legs) < 2:
            continue
        source = anchors[src]
        # The leg that carries straight on into its target cannot be moved, so it keeps its
        # column and the rest are laid out around it.
        plan = _edge_port_plan(
            [
                (link_key, _connector_exit_departure_x(link_paths[link_key]))
                for link_key in legs
            ],
            source,
        )
        moved: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for link_key, port_x in plan.items():
            target = anchors.get(link_key[1])
            if target is None:
                continue
            reseated = _reseat_connector_exit_port(
                link_paths[link_key],
                source=source,
                port_x=port_x,
            )
            if reseated is None or reseated == link_paths[link_key]:
                continue
            admissible = _admissible_connector_route(
                reseated,
                source=source,
                target=target,
                obstacles=_connector_block_obstacles(
                    anchors,
                    src=link_key[0],
                    tgt=link_key[1],
                    label_obstacles=label_obstacles,
                    graph=graph,
                    positions=positions,
                    link_key=link_key,
                ),
                graph=graph,
                link_key=link_key,
                positions=positions,
            )
            if admissible is None:
                continue
            moved[link_key] = admissible
        if not moved:
            continue
        before, _ = _connector_violation_links(link_paths, **metrics)
        trial = dict(link_paths)
        trial.update(moved)
        after, _ = _connector_violation_links(trial, **metrics)
        if after > before:
            continue
        link_paths.update(moved)


def _connector_vertical_runs(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    exclude: Collection[tuple[int, int]],
) -> list[tuple[float, float, float, int]]:
    """Drawn vertical runs as (x, low y, high y, source), for lanes already in use."""
    runs: list[tuple[float, float, float, int]] = []
    for link_key, points in link_paths.items():
        if link_key in exclude:
            continue
        for (first_x, first_y), (second_x, second_y) in zip(points, points[1:]):
            if (
                abs(first_x - second_x) <= PARALLEL_CONNECTOR_COORD_EPS
                and abs(first_y - second_y) > PARALLEL_CONNECTOR_COORD_EPS
            ):
                runs.append(
                    (first_x, min(first_y, second_y), max(first_y, second_y), link_key[0])
                )
    return runs


def _lane_is_free_for_run(
    runs: list[tuple[float, float, float, int]],
    *,
    lane_x: float,
    low_y: float,
    high_y: float,
    source: int,
) -> bool:
    """True when nothing else is drawn along the stretch of lane a feed wants.

    Runs from the same source may share, since those legs carry one value and read as a
    split rather than as two lines on top of each other.
    """
    gap = PARALLEL_CONNECTOR_CHANNEL_GAP
    return not any(
        run_source != source
        and abs(run_x - lane_x) < gap - PARALLEL_CONNECTOR_COORD_EPS
        and min(high_y, run_high) - max(low_y, run_low) > PARALLEL_CONNECTOR_COORD_EPS
        for run_x, run_low, run_high, run_source in runs
    )


def _merge_fan_in_routes(
    ordered: list[tuple[int, int]],
    *,
    target: _RenderAnchor,
    anchors: dict[int, _RenderAnchor],
    lane_xs: Sequence[float],
    side: int,
    graph,
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    runs: list[tuple[float, float, float, int]],
) -> dict[tuple[int, int], list[tuple[float, float]]] | None:
    """Nest a tile's feeds so no two are ever drawn along the same run.

    Two orders do the work. Ports go left to right in the order the feeds come across the
    page, so each docks on the side it arrives from and no run has to cross the top edge to
    reach its own port. Approach rows go the other way: the feed with the longest way to
    travel turns lowest, so it passes under the drops of the feeds it overtakes instead of
    through them. A feed with no headroom for its row keeps the route it has rather than
    sinking the whole plan.
    """
    gap = PARALLEL_CONNECTOR_CHANNEL_GAP
    entry_y = _connector_target_top_entry_y(target)
    floor_y = _connector_min_bus_y_above_target(target)
    lo = target.left + CONNECTOR_ATTACHED_BOX_MARGIN
    hi = target.right - CONNECTOR_ATTACHED_BOX_MARGIN
    feeds = [link_key for link_key in ordered if anchors.get(link_key[0]) is not None]
    if len(feeds) < 2:
        return None
    if hi - lo < (len(feeds) - 1) * gap - PARALLEL_CONNECTOR_COORD_EPS:
        return None
    if side < 0:
        lanes = sorted(x for x in lane_xs if x < target.left - gap / 2)
    else:
        lanes = sorted((x for x in lane_xs if x > target.right + gap / 2), reverse=True)

    step = max(gap, (hi - lo) / (len(feeds) - 1))
    ports = {
        link_key: lo + index * step
        for index, link_key in enumerate(
            sorted(feeds, key=lambda key: (anchors[key[0]].cx, key))
        )
    }
    seats = [
        (link_key, floor_y + index * gap, ports[link_key])
        for index, link_key in enumerate(
            sorted(
                feeds,
                key=lambda key: (-abs(anchors[key[0]].cx - ports[key]), key),
            )
        )
    ]
    routes: dict[tuple[int, int], list[tuple[float, float]]] = {}
    available = list(lanes)
    busy = list(runs)
    for link_key, row_y, port_x in seats:
        source = anchors.get(link_key[0])
        if source is None:
            continue
        exit_y = _connector_source_bottom_exit_y(source)
        if exit_y <= row_y + PARALLEL_CONNECTOR_COORD_EPS:
            continue
        stub_y = max(row_y, exit_y - CONNECTOR_EXIT_STUB)
        obstacles = _connector_block_obstacles(
            anchors,
            src=link_key[0],
            tgt=link_key[1],
            label_obstacles=label_obstacles,
            graph=graph,
            positions=positions,
            link_key=link_key,
        )
        # The source's own column comes first: a feed that can drop straight down needs no
        # detour, and taking a lane it does not need would push a feed that does further out.
        for position, lane_x in enumerate((source.cx, *available)):
            if not _lane_is_free_for_run(
                busy,
                lane_x=lane_x,
                low_y=row_y,
                high_y=stub_y,
                source=link_key[0],
            ):
                continue
            if abs(lane_x - source.cx) <= PARALLEL_CONNECTOR_COORD_EPS:
                points = [
                    (source.cx, exit_y),
                    (source.cx, row_y),
                    (port_x, row_y),
                    (port_x, entry_y),
                ]
            else:
                points = [
                    (source.cx, exit_y),
                    (source.cx, stub_y),
                    (lane_x, stub_y),
                    (lane_x, row_y),
                    (port_x, row_y),
                    (port_x, entry_y),
                ]
            candidate = _admissible_connector_route(
                points,
                source=source,
                target=target,
                obstacles=obstacles,
                graph=graph,
                link_key=link_key,
                positions=positions,
            )
            if candidate is None:
                continue
            routes[link_key] = candidate
            busy.extend(_connector_vertical_runs({link_key: candidate}, exclude=()))
            if position:
                del available[:position]
            break
    return routes if len(routes) >= 2 else None


def _reroute_crowded_merge_fan_in(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    merge_entry_x: dict[tuple[int, int], float],
) -> None:
    """Re-plan the feeds of any tile whose runs have ended up on top of each other.

    Several feeds that all have to get past the same stack of tiles will each, on their own,
    pick the one corridor outside it, so they end up doubled or tangled with each other.
    Planning the whole fan-in at once hands out a lane per feed instead of letting them
    compete for the best one, and the plan is only kept when it scores better than what the
    feeds had.
    """
    metrics = {
        "graph": graph,
        "anchors": anchors,
        "label_obstacles": label_obstacles,
        "positions": positions,
        "incoming": incoming,
        "outgoing": outgoing,
        "target_bus": target_bus,
        "source_bus": source_bus,
        "merge_link_bus": merge_link_bus,
    }
    lane_xs, _lane_ys = _connector_lane_coordinates(
        anchors,
        label_obstacles,
        graph=graph,
        positions=positions,
    )
    side_entry_links = _side_entry_links(graph)
    doubled: set[tuple[int, int]] = set()
    for first, second in _find_connector_path_overlaps(
        link_paths,
        incoming=incoming,
        outgoing=outgoing,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
        anchors=anchors,
        graph=graph,
    ):
        doubled.update((first, second))
    tangled = set(doubled)
    for first, second, _point in _find_connector_segment_crossings(link_paths):
        tangled.update((first, second))
    # A feed that dives under its tile or lands on a port it shares is just as much a sign
    # of a fan-in that was routed one leg at a time as two runs drawn on top of each other,
    # and it is the same replanning that clears it.
    for link_key, _reason in (
        *_find_connector_entry_approach_violations(link_paths, graph=graph, anchors=anchors),
        *_find_connector_entry_port_violations(link_paths, graph=graph, anchors=anchors),
    ):
        tangled.add(link_key)
    if not tangled:
        return
    by_target: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for link_key in link_paths:
        if link_key not in side_entry_links:
            by_target[link_key[1]].append(link_key)
    for tgt, feeds in sorted(by_target.items()):
        target = anchors.get(tgt)
        if target is None or len(feeds) < 2 or not tangled & set(feeds):
            continue
        ordered = sorted(
            (key for key in feeds if anchors.get(key[0]) is not None),
            key=lambda key: (anchors[key[0]].top, key),
        )
        if len(ordered) < 2:
            continue
        before, _ = _connector_violation_links(link_paths, **metrics)
        best: dict[tuple[int, int], list[tuple[float, float]]] | None = None
        best_score = before
        # Re-placing every feed is the tidiest arrangement but also the most constrained, so
        # where the whole group cannot be improved the tangled feeds alone are tried too.
        groups = [ordered]
        for selection in (tangled, doubled):
            subset = [link_key for link_key in ordered if link_key in selection]
            if 2 <= len(subset) < len(ordered) and subset not in groups:
                groups.append(subset)
        for group in groups:
            runs = _connector_vertical_runs(link_paths, exclude=group)
            for side in (-1, 1):
                plan = _merge_fan_in_routes(
                    group,
                    target=target,
                    anchors=anchors,
                    lane_xs=lane_xs,
                    side=side,
                    graph=graph,
                    label_obstacles=label_obstacles,
                    positions=positions,
                    runs=runs,
                )
                if plan is None:
                    continue
                after, _ = _connector_violation_links({**link_paths, **plan}, **metrics)
                if after < best_score:
                    best, best_score = plan, after
        if best is None:
            continue
        link_paths.update(best)
        for link_key, path in best.items():
            merge_entry_x[link_key] = path[-1][0]


def _reseat_target_entry_ports(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    label_obstacles: list[_RenderAnchor],
    positions: list | None,
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    merge_entry_x: dict[tuple[int, int], float],
) -> None:
    """Give every feed of a multi-input tile its own top port, in arrival order.

    Two edges into a tile carry two different tensors, so they must not share an entry: a
    shared port draws them on top of each other and puts a junction where the graph has
    none. Ports are only moved when doing so leaves the section no worse off overall.
    """
    from visualizer.computation_graph import _infer_side_entry_links

    side_entry_links = _side_entry_links(graph)
    metrics = {
        "graph": graph,
        "anchors": anchors,
        "label_obstacles": label_obstacles,
        "positions": positions,
        "incoming": incoming,
        "outgoing": outgoing,
        "target_bus": target_bus,
        "source_bus": source_bus,
        "merge_link_bus": merge_link_bus,
    }
    by_target: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for link_key, points in link_paths.items():
        target = anchors.get(link_key[1])
        if target is None or link_key in side_entry_links:
            continue
        if abs(points[-1][1] - _connector_target_top_entry_y(target)) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        by_target[link_key[1]].append(link_key)

    for tgt, feeds in sorted(by_target.items()):
        if len(feeds) < 2:
            continue
        target = anchors[tgt]
        plan = _edge_port_plan(
            [(link_key, _connector_entry_approach_x(link_paths[link_key])) for link_key in feeds],
            target,
        )
        moved: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for link_key, port_x in plan.items():
            source = anchors.get(link_key[0])
            if source is None:
                continue
            reseated = _reseat_connector_entry_port(
                link_paths[link_key],
                target=target,
                port_x=port_x,
            )
            if reseated is None or reseated == link_paths[link_key]:
                continue
            admissible = _admissible_connector_route(
                reseated,
                source=source,
                target=target,
                obstacles=_connector_block_obstacles(
                    anchors,
                    src=link_key[0],
                    tgt=link_key[1],
                    label_obstacles=label_obstacles,
                    graph=graph,
                    positions=positions,
                    link_key=link_key,
                ),
                graph=graph,
                link_key=link_key,
                positions=positions,
            )
            if admissible is None:
                continue
            moved[link_key] = admissible
        if not moved:
            continue
        before, _ = _connector_violation_links(link_paths, **metrics)
        trial = dict(link_paths)
        trial.update(moved)
        after, _ = _connector_violation_links(trial, **metrics)
        if after > before:
            continue
        for link_key, path in moved.items():
            link_paths[link_key] = path
            merge_entry_x[link_key] = path[-1][0]


def _top_entry_leg_geometry(
    points: list[tuple[float, float]],
    target: _RenderAnchor,
) -> tuple[float, float, float] | None:
    """Return (entry_x, jog_y, approach_x) for a jogged drop onto the target top."""
    if len(points) != 4:
        return None
    entry_x, entry_y = points[-1]
    if abs(entry_y - _connector_target_top_entry_y(target)) > PARALLEL_CONNECTOR_COORD_EPS:
        return None
    jog_x, jog_y = points[-2]
    approach_x, approach_y = points[-3]
    if abs(jog_x - entry_x) > PARALLEL_CONNECTOR_COORD_EPS:
        return None
    if abs(jog_y - approach_y) > PARALLEL_CONNECTOR_COORD_EPS:
        return None
    if abs(approach_x - points[0][0]) > PARALLEL_CONNECTOR_COORD_EPS:
        return None
    if abs(approach_x - entry_x) <= PARALLEL_CONNECTOR_COORD_EPS:
        return None
    return entry_x, jog_y, approach_x


def _orthogonal_paths_crossings(
    points_a: list[tuple[float, float]],
    points_b: list[tuple[float, float]],
) -> int:
    """Count places where one path's vertical run cuts through the other's horizontal."""
    crossings = 0
    for first, second in ((points_a, points_b), (points_b, points_a)):
        verticals = [
            segment for segment in _connector_axis_segments(first) if segment[0] == "v"
        ]
        horizontals = [
            segment for segment in _connector_axis_segments(second) if segment[0] == "h"
        ]
        for _, x, y_lo, y_hi, _index in verticals:
            for _, y, x_lo, x_hi, _other in horizontals:
                if (
                    x_lo + PARALLEL_CONNECTOR_COORD_EPS < x < x_hi - PARALLEL_CONNECTOR_COORD_EPS
                    and y_lo + PARALLEL_CONNECTOR_COORD_EPS
                    < y
                    < y_hi - PARALLEL_CONNECTOR_COORD_EPS
                ):
                    crossings += 1
    return crossings


def _path_crossing_total(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    link_key: tuple[int, int],
    points: list[tuple[float, float]],
) -> int:
    """Crossings between one path and every other routed path."""
    return sum(
        _orthogonal_paths_crossings(points, other_points)
        for other_key, other_points in link_paths.items()
        if other_key != link_key
    )


def _uncross_shared_top_entry_ports(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict[int, _RenderAnchor],
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    label_obstacles: list[_RenderAnchor],
    positions: list,
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    merge_entry_x: dict[tuple[int, int], float],
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Move a jogged top-entry port past the lane a higher feed drops through.

    Two feeds share one top edge only when the lower jog stops short of the lane
    the higher one falls in; reaching across it draws a crossing that reads as if
    the tile took a single tangled input.
    """
    updated = dict(link_paths)
    baseline = len(
        _find_connector_path_overlaps(
            updated,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            anchors=anchors,
            graph=graph,
        )
    )
    for tgt, link_group in incoming.items():
        target = anchors.get(tgt)
        if target is None or len(link_group) < 2:
            continue
        if tgt in target_bus:
            continue
        drops: dict[tuple[int, int], float] = {}
        jogged: dict[tuple[int, int], tuple[float, float, float]] = {}
        for link_key in link_group:
            points = updated.get(link_key)
            if points is None or len(points) < 2:
                continue
            entry_x, entry_y = points[-1]
            if abs(entry_y - _connector_target_top_entry_y(target)) > PARALLEL_CONNECTOR_COORD_EPS:
                continue
            drops[link_key] = entry_x
            if link_key in merge_entry_x or link_key in merge_link_bus:
                # Ports planned for a shared merge already carry their own spacing.
                continue
            geometry = _top_entry_leg_geometry(points, target)
            if geometry is not None:
                jogged[link_key] = geometry
        if len(drops) < 2 or not jogged:
            continue
        for link_key, (entry_x, jog_y, approach_x) in jogged.items():
            src, _ = link_key
            source = anchors.get(src)
            if source is None:
                continue
            lo, hi = sorted((entry_x, approach_x))
            blocking = [
                other_x
                for other_key, other_x in drops.items()
                if other_key != link_key
                and lo - PARALLEL_CONNECTOR_COORD_EPS < other_x < hi + PARALLEL_CONNECTOR_COORD_EPS
            ]
            if not blocking:
                continue
            inner = CONNECTOR_ATTACHED_BOX_MARGIN
            if approach_x > entry_x:
                port_x = max(blocking) + TOP_ENTRY_PORT_GAP
                if port_x > min(approach_x, target.right - inner):
                    continue
            else:
                port_x = min(blocking) - TOP_ENTRY_PORT_GAP
                if port_x < max(approach_x, target.left + inner):
                    continue
            route_y = jog_y
            if source.bottom - jog_y <= CONNECTOR_OBSTACLE_MARGIN:
                route_y = max(
                    _connector_min_bus_y_above_target(target),
                    min(
                        source.bottom - CONNECTOR_EXIT_STUB,
                        (source.bottom + target.top) / 2,
                    ),
                )
            candidate = [
                updated[link_key][0],
                (approach_x, route_y),
                (port_x, route_y),
                (port_x, target.top),
            ]
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
                candidate,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
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
            if _path_crossing_total(updated, link_key, candidate) > _path_crossing_total(
                updated,
                link_key,
                updated[link_key],
            ):
                continue
            trial = dict(updated)
            trial[link_key] = candidate
            if len(
                _find_connector_path_overlaps(
                    trial,
                    incoming=incoming,
                    outgoing=outgoing,
                    target_bus=target_bus,
                    source_bus=source_bus,
                    merge_link_bus=merge_link_bus,
                    anchors=anchors,
                    graph=graph,
                )
            ) > baseline:
                continue
            updated = trial
            drops[link_key] = port_x
    return updated


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
    allow_shared_buses: bool = False,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return unordered link pairs whose polylines share a non-bus channel.

    Doubling a run is only readable when both halves carry the same value, so by default a
    shared run between two different sources counts. Callers that must not fail a layout
    the earlier stages cannot yet untangle can set *allow_shared_buses* to fall back to the
    older rule, which treats an agreed merge row as a bus rather than an overlap.
    """
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
            # Legs of one source carry the same value, but stacking them still draws them as
            # a single line that appears to stop where they part, so the run counts. Callers
            # working on a layout the earlier stages cannot yet untangle keep the old rule,
            # which read a shared stem as one bus dividing further down.
            if link_a[0] == link_b[0]:
                if allow_shared_buses:
                    continue
                pair = (link_a, link_b) if link_a <= link_b else (link_b, link_a)
                if pair not in seen:
                    seen.add(pair)
                    overlaps.append(pair)
                continue
            # Otherwise a doubled line is only readable when the two links are in series and
            # share the column of the tile between them. Anything else leaves the reader
            # unable to say which value the run carries.
            if (
                not allow_shared_buses
                and link_b[0] != link_a[1]
                and link_a[0] != link_b[1]
            ):
                pair = (link_a, link_b) if link_a <= link_b else (link_b, link_a)
                if pair not in seen:
                    seen.add(pair)
                    overlaps.append(pair)
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
                if link_a[1] == link_b[1] and len(incoming.get(link_a[1], [])) >= 2:
                    target = anchors.get(link_a[1])
                    if (
                        target is not None
                        and abs(coord_a - seg_b[1]) <= PARALLEL_CONNECTOR_COORD_EPS
                        and seg_a[2] - PARALLEL_CONNECTOR_COORD_EPS
                        <= target.top
                        <= seg_a[3] + PARALLEL_CONNECTOR_COORD_EPS
                        and seg_b[2] - PARALLEL_CONNECTOR_COORD_EPS
                        <= target.top
                        <= seg_b[3] + PARALLEL_CONNECTOR_COORD_EPS
                    ):
                        continue
                if (
                    seg_b[0] == "v"
                    and abs(coord_a - seg_b[1]) <= PARALLEL_CONNECTOR_COORD_EPS
                    and _ranges_overlap(seg_a[2], seg_a[3], seg_b[2], seg_b[3])
                ):
                    source_a = anchors.get(link_a[0])
                    source_b = anchors.get(link_b[0])
                    if (
                        source_a is not None
                        and source_b is not None
                        and abs(source_a.cx - source_b.cx) <= PARALLEL_CONNECTOR_COORD_EPS
                    ):
                        continue
            elif _horizontal_segment_is_shared_bus(
                link_a,
                coord_a,
                target_bus=target_bus,
                source_bus=source_bus,
            ) and _horizontal_segment_is_shared_bus(
                link_b,
                seg_b[1],
                target_bus=target_bus,
                source_bus=source_bus,
            ):
                continue
            elif (
                seg_a[0] == "h"
                and seg_b[0] == "h"
                and abs(coord_a - seg_b[1]) <= PARALLEL_CONNECTOR_COORD_EPS
                and any(
                    abs(coord_a - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS
                    for bus_y in target_bus.values()
                )
            ):
                continue
            elif (
                seg_a[0] == "h"
                and seg_b[0] == "h"
                and abs(coord_a - seg_b[1]) < PARALLEL_CONNECTOR_CHANNEL_GAP
                and seg_a[3] <= anchors.get(link_a[0], anchors[link_a[1]]).cx + 0.05
                and seg_b[3] <= anchors.get(link_b[0], anchors[link_b[1]]).cx + 0.05
            ):
                continue
            elif link_a[1] == link_b[1] and len(incoming.get(link_a[1], [])) >= 2:
                target = anchors.get(link_a[1])
                if (
                    target is not None
                    and target.top - PARALLEL_CONNECTOR_COORD_EPS
                    <= coord_a
                    <= target.top + SAME_COLUMN_BYPASS_CORRIDOR
                    and abs(coord_a - seg_b[1]) <= PARALLEL_CONNECTOR_COORD_EPS
                ):
                    continue
            if link_b[0] == link_a[1] or link_a[0] == link_b[1]:
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
            if (
                link_a[1] == link_b[1]
                and link_a in merge_link_bus
                and link_b in merge_link_bus
                and abs(merge_link_bus[link_a] - merge_link_bus[link_b])
                <= PARALLEL_CONNECTOR_COORD_EPS
                and seg_a[0] == "v"
                and seg_b[0] == "v"
                and _ranges_overlap(seg_a[2], seg_a[3], seg_b[2], seg_b[3])
            ):
                target = anchors.get(link_a[1])
                if target is not None:
                    entry_y = _connector_target_top_entry_y(target)
                    if (
                        abs(seg_a[2] - entry_y) <= PARALLEL_CONNECTOR_COORD_EPS
                        and abs(seg_b[2] - entry_y) <= PARALLEL_CONNECTOR_COORD_EPS
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
    return _ensure_orthogonal_connector_path(adjusted)


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
    graph=None,
    label_obstacles: list[_RenderAnchor] | None = None,
    positions: list | None = None,
    target_bus: dict[int, float] | None = None,
    source_bus: dict[int, float] | None = None,
    merge_link_bus: dict[tuple[int, int], float] | None = None,
    merge_entry_x: dict[tuple[int, int], float] | None = None,
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
) -> list[tuple[float, float]]:
    """Shift horizontal bus levels while keeping the path out of third-party tiles."""
    source = anchors.get(src)
    target = anchors.get(tgt)
    link_key = (src, tgt)
    obstacles = (
        _connector_block_obstacles(
            anchors,
            src=src,
            tgt=tgt,
            label_obstacles=label_obstacles or [],
            graph=graph,
            positions=positions,
            link_key=link_key,
        )
        if graph is not None and source is not None and target is not None
        else [
            anchor for node_index, anchor in anchors.items() if node_index not in {src, tgt}
        ]
    )

    def _candidate_keeps_source_exit_stub(candidate: list[tuple[float, float]]) -> bool:
        """Reject shifts that pull the first horizontal into the source's exit stub."""
        if source is None:
            return True
        return (
            _connector_turn_before_clearing_source(
                candidate,
                y_exit=_connector_source_bottom_exit_y(source),
                source_cx=source.cx,
            )
            is None
        )

    def _candidate_is_valid(candidate: list[tuple[float, float]]) -> bool:
        if not _candidate_keeps_source_exit_stub(candidate):
            return False
        if source is None or target is None:
            return not _path_penetrates_obstacle_tiles(
                candidate,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            )
        return _connector_path_clear_of_blocks(
            candidate,
            source=source,
            target=target,
            obstacles=obstacles,
        )

    for attempt in (delta_y, -delta_y, 2 * delta_y, -2 * delta_y):
        candidate = _shift_path_horizontal_levels(points, delta_y=attempt)
        if _candidate_is_valid(candidate):
            if graph is None or source is None or target is None:
                return candidate
            return _coerce_connector_path_for_link(
                candidate,
                link_key=link_key,
                graph=graph,
                anchors=anchors,
                label_obstacles=label_obstacles or [],
                positions=positions,
                target_bus=target_bus or {},
                source_bus=source_bus or {},
                merge_link_bus=merge_link_bus or {},
                merge_entry_x=merge_entry_x,
                outgoing=outgoing,
            )
    fallback = _shift_path_horizontal_levels(points, delta_y=delta_y)
    if not _candidate_keeps_source_exit_stub(fallback):
        # An unresolved overlap is recoverable by later passes; a connector that
        # turns before it has cleared its own source tile is not.
        return points
    if graph is None or source is None or target is None:
        return fallback
    return _coerce_connector_path_for_link(
        fallback,
        link_key=link_key,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles or [],
        positions=positions,
        target_bus=target_bus or {},
        source_bus=source_bus or {},
        merge_link_bus=merge_link_bus or {},
        merge_entry_x=merge_entry_x,
        outgoing=outgoing,
    )


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
    label_obstacles: list[_RenderAnchor] | None = None,
    positions: list | None = None,
    merge_entry_x: dict[tuple[int, int], float] | None = None,
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Offset channels until no non-bus connector segments overlap."""
    cleared = dict(link_paths)

    def _coerce_shifted(link_key: tuple[int, int], points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if label_obstacles is None:
            return points
        return _coerce_connector_path_for_link(
            points,
            link_key=link_key,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            merge_entry_x=merge_entry_x,
            outgoing=outgoing,
        )

    shift_kwargs = {
        "graph": graph,
        "label_obstacles": label_obstacles,
        "positions": positions,
        "target_bus": target_bus,
        "source_bus": source_bus,
        "merge_link_bus": merge_link_bus,
        "merge_entry_x": merge_entry_x,
        "outgoing": outgoing,
    }
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
        overlap_pair = next(
            (
                (seg_a, seg_b)
                for seg_a in _connector_axis_segments(cleared[link_a])
                for seg_b in _connector_axis_segments(cleared[link_b])
                if _connector_segment_pairs_overlap(
                    seg_a,
                    seg_b,
                    link_a=link_a,
                    link_b=link_b,
                )
            ),
            None,
        )
        if (
            overlap_pair is not None
            and overlap_pair[0][0] == "h"
            and overlap_pair[1][0] == "h"
            and abs(overlap_pair[0][1] - overlap_pair[1][1])
            < PARALLEL_CONNECTOR_CHANNEL_GAP - PARALLEL_CONNECTOR_COORD_EPS
        ):
            shifted = False
            for link_key in (link_b, link_a):
                candidate = _shift_path_resolving_overlap(
                    cleared[link_key],
                    src=link_key[0],
                    tgt=link_key[1],
                    anchors=anchors,
                    delta_y=PARALLEL_CONNECTOR_CHANNEL_GAP,
                    graph=graph,
                    label_obstacles=label_obstacles,
                    positions=positions,
                    target_bus=target_bus,
                    source_bus=source_bus,
                    merge_link_bus=merge_link_bus,
                    merge_entry_x=merge_entry_x,
                    outgoing=outgoing,
                )
                if candidate != cleared[link_key] and not _find_connector_path_overlaps(
                    {**cleared, link_key: candidate},
                    incoming=incoming,
                    outgoing=outgoing,
                    target_bus=target_bus,
                    source_bus=source_bus,
                    merge_link_bus=merge_link_bus,
                    anchors=anchors,
                    graph=graph,
                ):
                    cleared[link_key] = candidate
                    shifted = True
                    break
            if shifted:
                continue
        if link_a[1] == link_b[0]:
            shifted = False
            for delta in (
                PARALLEL_CONNECTOR_CHANNEL_GAP,
                -PARALLEL_CONNECTOR_CHANNEL_GAP,
                2 * PARALLEL_CONNECTOR_CHANNEL_GAP,
                -2 * PARALLEL_CONNECTOR_CHANNEL_GAP,
            ):
                for link_key, candidate_fn in (
                    (link_b, lambda d: _shift_path_vertical_levels(cleared[link_b], delta_x=d)),
                    (link_a, lambda d: _shift_path_horizontal_levels(cleared[link_a], delta_y=-abs(d))),
                ):
                    candidate = candidate_fn(delta)
                    if not _find_connector_path_overlaps(
                        {**cleared, link_key: candidate},
                        incoming=incoming,
                        outgoing=outgoing,
                        target_bus=target_bus,
                        source_bus=source_bus,
                        merge_link_bus=merge_link_bus,
                        anchors=anchors,
                        graph=graph,
                    ):
                        cleared[link_key] = _coerce_shifted(link_key, candidate)
                        shifted = True
                        break
                if shifted:
                    break
            if shifted:
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
        shared_target_merge = (
            link_a[1] == link_b[1]
            and link_a in merge_link_bus
            and link_b in merge_link_bus
            and abs(merge_link_bus[link_a] - merge_link_bus[link_b])
            <= PARALLEL_CONNECTOR_COORD_EPS
        )
        if (
            shared_target_merge
            and seg_a[0] == "v"
            and seg_b[0] == "v"
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
                    cleared[link_b] = _coerce_shifted(link_b, candidate)
                    break
            continue
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
                    cleared[link_b] = _coerce_shifted(link_b, candidate)
                    break
            continue
        cleared[link_b] = _shift_path_resolving_overlap(
            cleared[link_b],
            src=link_b[0],
            tgt=link_b[1],
            anchors=anchors,
            delta_y=PARALLEL_CONNECTOR_CHANNEL_GAP,
            **shift_kwargs,
        )
    if label_obstacles is not None:
        for link_key, points in list(cleared.items()):
            cleared[link_key] = _coerce_shifted(link_key, points)
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
    link_key: tuple[int, int],
    coord: float,
    *,
    target_bus: dict[int, float],
    source_bus: dict[int, float],
) -> bool:
    """True when this link owns an intentional shared merge/source bus.

    A coordinate matching some unrelated bus is not enough. Likewise,
    ``merge_link_bus`` stores per-link approach levels; equal values there do not
    mean the connectors combine into one trunk.
    """
    src, tgt = link_key
    target_y = target_bus.get(tgt)
    if target_y is not None and abs(coord - target_y) <= PARALLEL_CONNECTOR_COORD_EPS:
        return True
    source_y = source_bus.get(src)
    return source_y is not None and abs(coord - source_y) <= PARALLEL_CONNECTOR_COORD_EPS


def _separate_parallel_connector_paths(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph=None,
    positions: list | None = None,
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
            cluster_links = {segment.link_key for segment in cluster}
            if len(cluster_links) < 2:
                continue
            if vertical_axis:
                bus_segments = [
                    segment
                    for segment in cluster
                    if _vertical_segment_is_shared_bus(
                        segment.link_key,
                        segment.coord,
                        incoming=incoming,
                        outgoing=outgoing,
                        target_bus=target_bus,
                        source_bus=source_bus,
                        anchors=anchors,
                    )
                ]
            else:
                bus_segments = [
                    segment
                    for segment in cluster
                    if _horizontal_segment_is_shared_bus(
                        segment.link_key,
                        segment.coord,
                        target_bus=target_bus,
                        source_bus=source_bus,
                    )
                ]
            # Bus members stay on the shared channel so the trunk keeps its stem;
            # only the intruding links move off it.
            pinned_links = {segment.link_key for segment in bus_segments}
            if graph is not None:
                pinned_links |= {
                    link_key
                    for link_key in cluster_links
                    if _is_inline_frame_spine_link(
                        graph,
                        link_key,
                        positions=positions if positions is not None else [],
                        anchors=anchors,
                    )
                }
            movable_links = sorted(cluster_links - pinned_links)
            if not movable_links:
                continue
            base_coord = cluster[0].coord
            first_offset = 1 if pinned_links else 0
            target_map = x_offsets if vertical_axis else y_offsets
            for index, link_key in enumerate(movable_links, start=first_offset):
                target_map.setdefault(link_key, {})[base_coord] = (
                    base_coord - index * channel_gap
                )

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


def _compact_frame_tail_shared_merge_stubs(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    target_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    max_stub: float,
) -> None:
    """Nudge a shared merge bus upward when a frame-tail exit stub is still too long."""
    from visualizer.computation_graph import _inline_frame_tail_indices

    tail_indices = _inline_frame_tail_indices(graph)
    for tgt, bus_y in list(target_bus.items()):
        longest_stub = 0.0
        for (src, link_tgt), points in link_paths.items():
            if link_tgt != tgt or src not in tail_indices or len(points) < 2:
                continue
            if not any(abs(y - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS for _x, y in points):
                continue
            longest_stub = max(longest_stub, points[0][1] - points[1][1])
        if longest_stub <= max_stub + 1e-6:
            continue
        delta = longest_stub - max_stub
        new_y = bus_y + delta
        target_bus[tgt] = new_y
        for link_key, points in list(link_paths.items()):
            if link_key[1] != tgt:
                continue
            adjusted = [
                (x, y + delta if abs(y - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS else y)
                for x, y in points
            ]
            link_paths[link_key] = _ensure_orthogonal_connector_path(adjusted)
        for link_key, level in list(merge_link_bus.items()):
            if link_key[1] == tgt and abs(level - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS:
                merge_link_bus[link_key] = new_y


def _align_merge_feed_bus_horizontals(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    incoming: dict[int, list[tuple[int, int]]],
) -> None:
    """Keep every feed into one summation on the same merge-bus row."""
    for tgt, link_group in incoming.items():
        if not _is_summation_label(graph.nodes[tgt].label):
            continue
        keys = [link for link in link_group if link in link_paths]
        if len(keys) < 2:
            continue
        levels = [
            segment[1]
            for link in keys
            for segment in _connector_axis_segments(link_paths[link])
            if segment[0] == "h"
        ]
        if len(levels) < 2:
            continue
        level = max(levels)
        for link in keys:
            points = link_paths[link]
            adjusted: list[tuple[float, float]] = []
            for index, (x, y) in enumerate(points):
                if index + 1 < len(points):
                    x2, y2 = points[index + 1]
                    if abs(y - y2) <= PARALLEL_CONNECTOR_COORD_EPS and abs(x - x2) > PARALLEL_CONNECTOR_COORD_EPS:
                        adjusted.append((x, level))
                        continue
                if index > 0:
                    x0, y0 = points[index - 1]
                    if abs(y - y0) <= PARALLEL_CONNECTOR_COORD_EPS and abs(x - x0) > PARALLEL_CONNECTOR_COORD_EPS:
                        adjusted.append((x, level))
                        continue
                adjusted.append((x, y))
            link_paths[link] = _ensure_orthogonal_connector_path(adjusted)


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
    preserved_frame_tail_paths: dict[tuple[int, int], list[tuple[float, float]]] = {}
    inline_bypass_bus_x = _plan_inline_bypass_bus_x(graph, links, anchors, positions)
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
            inline_bypass_bus_x=inline_bypass_bus_x,
        )
        if points is not None and len(points) >= 2:
            from visualizer.computation_graph import _inline_frame_tail_indices

            if src in _inline_frame_tail_indices(graph):
                tail_frame = _frame_for_tail_node(graph, src)
                if (
                    tail_frame is not None
                    and tgt not in tail_frame.node_indices
                    and _is_summation_label(graph.nodes[tgt].label)
                ):
                    bounds = _inline_frame_draw_bounds(tail_frame, positions, graph)
                    pipeline_floor = (
                        bounds.bottom
                        - CONNECTOR_OBSTACLE_MARGIN
                        - CONNECTOR_EXIT_STUB
                        - PIPELINE_MERGE_BUS_BELOW_FRAME_GAP
                    )
                    horizontals = [
                        y1
                        for (x1, y1), (x2, y2) in zip(points, points[1:])
                        if abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS
                        and abs(x1 - x2) > 0.06
                    ]
                    if any(
                        y <= pipeline_floor + PARALLEL_CONNECTOR_COORD_EPS for y in horizontals
                    ):
                        preserved_frame_tail_paths[link_key] = list(points)
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
                    merge_entry_x=merge_entry_x,
                )
            link_paths[link_key] = points
    initial_link_paths = {
        link_key: list(points)
        for link_key, points in link_paths.items()
        if points is not None and len(points) >= 2
    }
    separated = _separate_parallel_connector_paths(
        link_paths,
        graph=graph,
        positions=positions,
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
        merge_entry_x=merge_entry_x,
        outgoing=outgoing,
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
        label_obstacles=label_obstacles,
        positions=positions,
        merge_entry_x=merge_entry_x,
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
        merge_entry_x=merge_entry_x,
        outgoing=outgoing,
    )
    _finalize_inline_bypass_spine_tees(
        cleared,
        graph=graph,
        anchors=anchors,
        positions=positions,
        inline_bypass_bus_x=inline_bypass_bus_x,
        merge_entry_x=merge_entry_x,
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
        merge_entry_x=merge_entry_x,
        outgoing=outgoing,
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
                merge_entry_x=merge_entry_x,
            )
        snapped = _snap_connector_path_endpoints(
            routed,
            source=source,
            target=target,
            link_key=link_key,
            graph=graph,
            merge_entry_x=merge_entry_x,
            target_bus=target_bus,
            merge_link_bus=merge_link_bus,
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
    _assert_connector_tees_precede_bus_joins(
        validated,
        graph=graph,
        anchors=anchors,
        outgoing=outgoing,
        source_bus=source_bus,
        target_bus=target_bus,
        stage="final",
    )
    for link_key, points in list(validated.items()):
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue
        # A feed that already docks on the middle of the tile's top edge is
        # correct as drawn; lifting it above the frame would only drop it back
        # across the bus its siblings run along.
        if _path_enters_target_top_center(points, target):
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
        frame_entry = _outside_to_inline_frame_top_member_route(
            source,
            target,
            obstacles,
            graph,
            positions,
            src=src,
            tgt=tgt,
        )
        if frame_entry is not None:
            validated[link_key] = frame_entry
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
    _compact_frame_tail_shared_merge_stubs(
        validated,
        graph=graph,
        target_bus=target_bus,
        merge_link_bus=merge_link_bus,
        max_stub=0.55,
    )
    for link_key, points in list(validated.items()):
        if link_key not in merge_entry_x:
            continue
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None or len(points) < 2:
            continue
        entry_y = _connector_target_top_entry_y(target)
        if abs(source.cx - target.cx) < 0.08:
            if tgt in target_bus:
                continue
            spread_entry_x = merge_entry_x.get(link_key)
            if (
                source.bottom
                >= target.top - (CONNECTOR_OBSTACLE_MARGIN + PARALLEL_CONNECTOR_COORD_EPS)
                and link_key not in merge_link_bus
                and (
                    spread_entry_x is None
                    or abs(spread_entry_x - target.cx) < PARALLEL_CONNECTOR_COORD_EPS
                )
            ):
                continue
            spread_route = _same_column_spread_top_entry_connector_points(
                source,
                target,
                merge_entry_x[link_key],
                gap=0.04,
            )
            obstacles = _connector_block_obstacles(
                anchors,
                src=src,
                tgt=tgt,
                label_obstacles=label_obstacles,
                graph=graph,
                positions=positions,
                link_key=link_key,
            )
            if _connector_path_clear_of_blocks(
                spread_route,
                source=source,
                target=target,
                obstacles=obstacles,
            ) and (
                _connector_path_violates_inline_frame_bounds(
                    spread_route,
                    graph,
                    positions,
                    src=src,
                    tgt=tgt,
                )
                is None
            ):
                validated[link_key] = spread_route
            continue
        if (
            outgoing is not None
            and _source_fanout_splits_before_target_bus(
                graph,
                src,
                outgoing,
                target_bus,
            )
            and abs(points[-1][1] - entry_y) <= PARALLEL_CONNECTOR_COORD_EPS
            and abs(points[-1][0] - merge_entry_x[link_key]) <= PARALLEL_CONNECTOR_COORD_EPS
        ):
            continue
        adjusted = _snap_spread_top_entry_path(
            points,
            entry_x=merge_entry_x[link_key],
            entry_y=entry_y,
            min_bus_y=merge_link_bus.get(link_key),
        )
        leg_bus = merge_link_bus.get(link_key)
        if leg_bus is not None:
            floored = _enforce_merge_link_bus_floor(adjusted, leg_bus)
            obstacles = _connector_block_obstacles(
                anchors,
                src=src,
                tgt=tgt,
                label_obstacles=label_obstacles,
                graph=graph,
                positions=positions,
                link_key=link_key,
            )
            if _connector_path_clear_of_blocks(
                floored,
                source=source,
                target=target,
                obstacles=obstacles,
            ):
                adjusted = floored
        obstacles = _connector_block_obstacles(
            anchors,
            src=src,
            tgt=tgt,
            label_obstacles=label_obstacles,
            graph=graph,
            positions=positions,
            link_key=link_key,
        )
        if _connector_path_clear_of_blocks(
            adjusted,
            source=source,
            target=target,
            obstacles=obstacles,
        ):
            validated[link_key] = _prefer_non_backtracking_connector_path(
                points,
                adjusted,
                source=source,
                target=target,
                obstacles=obstacles,
            )
    if validate_layout:
        validated = _ensure_connector_paths_non_overlapping(
            validated,
            graph=graph,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            merge_entry_x=merge_entry_x,
        )
        for link_key, points in list(validated.items()):
            src, tgt = link_key
            source = anchors.get(src)
            target = anchors.get(tgt)
            if source is None or target is None:
                continue
            validated[link_key] = _repair_connector_source_departure(
                points,
                source=source,
                target=target,
                link_key=link_key,
                graph=graph,
            )
        validated = _reroute_detail_link_paths_clearing_blocks(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            merge_entry_x=merge_entry_x,
            outgoing=outgoing,
        )
    validated = _uncross_shared_top_entry_ports(
        validated,
        graph=graph,
        anchors=anchors,
        incoming=incoming,
        outgoing=outgoing,
        label_obstacles=label_obstacles,
        positions=positions,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
        merge_entry_x=merge_entry_x,
    )
    if validate_layout and _graph_requires_strict_connector_validation(graph):
        for link_key, points in list(validated.items()):
            src, tgt = link_key
            source = anchors.get(src)
            target = anchors.get(tgt)
            if source is None or target is None or len(points) < 2:
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
            entry_y = _connector_target_top_entry_y(target)
            needs_fix = _path_hits_obstacles(
                points,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            )
            if (
                merge_entry_x is not None
                and link_key in merge_entry_x
                and abs(points[-1][1] - entry_y) > PARALLEL_CONNECTOR_COORD_EPS
            ):
                needs_fix = True
            if not needs_fix:
                continue
            bus_y = _link_routing_bus_y(
                link_key,
                target_bus=target_bus,
                source_bus=source_bus,
                merge_link_bus=merge_link_bus,
            )
            rerouted = _reroute_connector_path_clearing_blocks(
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
                merge_entry_x=merge_entry_x,
                outgoing=outgoing,
                target_bus=target_bus,
            )
            rerouted = _repair_connector_source_departure(
                rerouted,
                source=source,
                target=target,
                link_key=link_key,
                graph=graph,
            )
            if (
                merge_entry_x is not None
                and link_key in merge_entry_x
                and abs(rerouted[-1][1] - entry_y) > PARALLEL_CONNECTOR_COORD_EPS
            ):
                rerouted = _snap_spread_top_entry_path(
                    rerouted,
                    entry_x=merge_entry_x[link_key],
                    entry_y=entry_y,
                    min_bus_y=merge_link_bus.get(link_key) if merge_link_bus else None,
                )
            if _path_hits_obstacles(
                rerouted,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            ) and abs(source.cx - target.cx) < 0.08:
                gutter = _same_column_side_gutter_detour(source, target, obstacles)
                if not _path_hits_obstacles(
                    gutter,
                    obstacles,
                    margin=CONNECTOR_OBSTACLE_MARGIN,
                ) and not _connector_path_has_block_edge_horizontal_jog(
                    gutter,
                    source=source,
                    target=target,
                    link_key=link_key,
                    graph=graph,
                ):
                    rerouted = gutter
            if _path_hits_obstacles(
                rerouted,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            ):
                fresh = _connector_points_for_link(
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
                    inline_bypass_bus_x=inline_bypass_bus_x,
                )
                if (
                    fresh is not None
                    and len(fresh) >= 2
                    and not _path_hits_obstacles(
                        fresh,
                        obstacles,
                        margin=CONNECTOR_OBSTACLE_MARGIN,
                    )
                    and not _connector_path_has_block_edge_horizontal_jog(
                        fresh,
                        source=source,
                        target=target,
                        link_key=link_key,
                        graph=graph,
                    )
                ):
                    rerouted = fresh
            if _path_hits_obstacles(
                rerouted,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            ):
                side_bypass = _horizontal_departure_side_bypass_route(
                    source,
                    target,
                    obstacles,
                    tee_y=source_bus.get(src),
                )
                if (
                    side_bypass is not None
                    and not _path_hits_obstacles(
                        side_bypass,
                        obstacles,
                        margin=CONNECTOR_OBSTACLE_MARGIN,
                    )
                    and not _connector_path_has_block_edge_horizontal_jog(
                        side_bypass,
                        source=source,
                        target=target,
                        link_key=link_key,
                        graph=graph,
                    )
                ):
                    rerouted = side_bypass
            if not _path_hits_obstacles(
                rerouted,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            ):
                validated[link_key] = rerouted
    _apply_same_column_bypass_routes(
        validated,
        graph=graph,
        positions=positions,
        anchors=anchors,
        label_obstacles=label_obstacles,
        incoming=incoming,
        merge_entry_x=merge_entry_x,
        target_bus=target_bus,
        source_bus=source_bus,
    )
    # Routing and overlap repair may replace a path with a geometry-specific
    # fallback. Reassert the architectural port contract at the final boundary.
    for link_key, points in list(validated.items()):
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue
        if _is_inline_frame_skip_link(graph, link_key):
            validated[link_key] = points
            continue
        target_frame = next(
            (frame for frame in graph.inline_frames if tgt in frame.node_indices),
            None,
        )
        obstacles = _connector_block_obstacles(
            anchors,
            src=src,
            tgt=tgt,
            label_obstacles=label_obstacles,
            graph=graph,
            positions=positions,
            link_key=link_key,
        )
        endpoint_entries = merge_entry_x
        backward_obstacles = [
            anchor
            for node_index, anchor in anchors.items()
            if node_index not in {src, tgt}
        ] + label_obstacles
        backward_route = None
        if target_frame is None or src in target_frame.node_indices:
            backward_route = _backward_top_entry_gutter_route(
                source,
                target,
                backward_obstacles,
                entry_x=merge_entry_x.get(link_key),
                channel=(src * 37 + tgt) % 17,
            )
        if backward_route is not None:
            points = backward_route
            endpoint_entries = None
        elif (
            target_frame is not None
            and src not in target_frame.node_indices
        ):
            if not _same_column_straight_inline_top_feed(
                points,
                source,
                target,
                graph=graph,
                positions=positions,
                src=src,
                tgt=tgt,
                obstacles=obstacles,
            ):
                frame_entry_x = merge_entry_x.get(link_key, target.cx)
                if (
                    positions[src].spec.synthetic == SYNTHETIC_INPUT
                    and len(incoming.get(tgt, [])) > 1
                ):
                    frame_entry_x = (
                        target.cx - TOP_ENTRY_PORT_GAP
                        if frame_entry_x >= target.cx
                        else target.cx + TOP_ENTRY_PORT_GAP
                    )
                frame_entry = _outside_to_inline_frame_top_member_route(
                    source,
                    target,
                    obstacles,
                    graph,
                    positions,
                    src=src,
                    tgt=tgt,
                    entry_x=frame_entry_x,
                    source_tee_y=source_bus.get(src),
                )
                if frame_entry is None:
                    frame_entry = _outside_to_inline_frame_inner_member_route(
                        source,
                        target,
                        obstacles,
                        graph,
                        positions,
                        src=src,
                        tgt=tgt,
                        entry_x=frame_entry_x,
                    )
                if frame_entry is not None:
                    points = frame_entry
            endpoint_entries = None
        if _connector_path_violates_inline_frame_bounds(
            points,
            graph,
            positions,
            src=src,
            tgt=tgt,
        ) is not None:
            tail_frame = _frame_for_tail_node(graph, src)
            if tail_frame is not None and tgt not in tail_frame.node_indices:
                entry_x = merge_entry_x.get(link_key, target.cx)
                exit_bus_y = merge_link_bus.get(
                    link_key,
                    _connector_min_bus_y_above_target(target),
                )
                points = _frame_tail_merge_entry_connector_points(
                    source,
                    target,
                    exit_x=source.cx,
                    entry_x=entry_x,
                    bus_y=exit_bus_y,
                    frame_bounds=_inline_frame_draw_bounds(
                        tail_frame,
                        positions,
                        graph,
                    ),
                    obstacles=obstacles,
                )
                endpoint_entries = None
            for frame in graph.inline_frames:
                if (
                    _connector_path_violates_inline_frame_bounds(
                        points,
                        graph,
                        positions,
                        src=src,
                        tgt=tgt,
                    )
                    is None
                ):
                    break
                members = set(frame.node_indices)
                if src not in members or tgt in members:
                    continue
                bounds = _inline_frame_draw_bounds(frame, positions, graph)
                if bounds.left < target.cx < bounds.right:
                    continue
                outward_entry = target.cx
                outward_route = _ensure_orthogonal_connector_path(
                    [
                        (source.cx, _connector_source_bottom_exit_y(source)),
                        (
                            source.cx,
                            _connector_source_bottom_exit_y(source)
                            - CONNECTOR_EXIT_STUB,
                        ),
                        (
                            outward_entry,
                            _connector_source_bottom_exit_y(source)
                            - CONNECTOR_EXIT_STUB,
                        ),
                        (outward_entry, _connector_target_top_entry_y(target)),
                    ]
                )
                if _connector_path_violates_inline_frame_bounds(
                    outward_route,
                    graph,
                    positions,
                    src=src,
                    tgt=tgt,
                ) is None:
                    points = outward_route
                    endpoint_entries = None
                    break
        validated[link_key] = _snap_connector_path_endpoints(
            points,
            source=source,
            target=target,
            link_key=link_key,
            graph=graph,
            merge_entry_x=endpoint_entries,
            target_bus=target_bus,
            merge_link_bus=merge_link_bus,
            source_bus=source_bus,
        )
    _apply_same_column_bypass_routes(
        validated,
        graph=graph,
        positions=positions,
        anchors=anchors,
        label_obstacles=label_obstacles,
        incoming=incoming,
        merge_entry_x=merge_entry_x,
        target_bus=target_bus,
        source_bus=source_bus,
    )
    _apply_directional_fanout_bypass_routes(
        validated,
        graph=graph,
        positions=positions,
        anchors=anchors,
        label_obstacles=label_obstacles,
        outgoing=outgoing,
        source_bus=source_bus,
    )
    _apply_stacked_same_side_fanout_routes(
        validated,
        graph=graph,
        positions=positions,
        anchors=anchors,
        label_obstacles=label_obstacles,
        outgoing=outgoing,
        merge_entry_x=merge_entry_x,
    )
    _apply_fork_join_side_branch_bypass_routes(
        validated,
        graph=graph,
        positions=positions,
        anchors=anchors,
        label_obstacles=label_obstacles,
        merge_entry_x=merge_entry_x,
    )
    # Port and frame corrections above can expose a tile that was deliberately
    # absent from an earlier route's obstacle set. Re-run the full-tile repair at
    # the final boundary so no emitted connector can pass through a sibling tile.
    validated = _reroute_detail_link_paths_clearing_blocks(
        validated,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
        merge_entry_x=merge_entry_x,
        outgoing=outgoing,
    )
    _repair_final_connector_tile_collisions(
        validated,
        graph=graph,
        positions=positions,
        anchors=anchors,
        label_obstacles=label_obstacles,
        source_bus=source_bus,
        merge_entry_x=merge_entry_x,
        merge_link_bus=merge_link_bus,
    )
    _repair_final_connector_frame_collisions(
        validated,
        graph=graph,
        positions=positions,
        anchors=anchors,
        label_obstacles=label_obstacles,
        source_bus=source_bus,
        merge_entry_x=merge_entry_x,
        merge_link_bus=merge_link_bus,
    )
    _lift_connector_horizontals_off_frame_borders(
        validated,
        graph=graph,
        positions=positions,
        anchors=anchors,
        label_obstacles=label_obstacles,
    )
    _apply_fork_join_side_branch_bypass_routes(
        validated,
        graph=graph,
        positions=positions,
        anchors=anchors,
        label_obstacles=label_obstacles,
        merge_entry_x=merge_entry_x,
    )
    for link_key, points in list(validated.items()):
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None or len(points) < 2:
            continue
        y_exit = _connector_source_bottom_exit_y(source)
        if abs(points[0][0] - source.cx) > PARALLEL_CONNECTOR_COORD_EPS:
            points = [(source.cx, y_exit), *points[1:]]
        elif abs(points[0][1] - y_exit) > PARALLEL_CONNECTOR_COORD_EPS:
            points = [(source.cx, y_exit), *points[1:]]
        repaired = _ensure_orthogonal_connector_path(points)
        if (
            _is_inline_frame_spine_link(
                graph,
                link_key,
                positions=positions,
                anchors=anchors,
            )
            and len(repaired) == 2
            and abs(repaired[0][0] - repaired[-1][0]) > PARALLEL_CONNECTOR_COORD_EPS
        ):
            target = anchors.get(link_key[1])
            if target is not None:
                repaired = _same_column_straight_connector_points(source, target)
        validated[link_key] = repaired
    _repair_horizontal_backtracking_paths(
        validated,
        graph=graph,
        anchors=anchors,
        incoming=incoming,
        outgoing=outgoing,
        label_obstacles=label_obstacles,
        positions=positions,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_entry_x=merge_entry_x,
        merge_link_bus=merge_link_bus,
        input_index=input_index,
        inline_bypass_bus_x=inline_bypass_bus_x,
        initial_link_paths=initial_link_paths,
    )
    if validate_layout:
        from visualizer.computation_graph import _infer_side_entry_links

        side_entry_links = set(_infer_side_entry_links(graph))
        for _ in range(8):
            validated = _separate_parallel_connector_paths(
                validated,
                graph=graph,
                positions=positions,
                incoming=incoming,
                outgoing=outgoing,
                target_bus=target_bus,
                source_bus=source_bus,
                merge_link_bus=merge_link_bus,
                anchors=anchors,
            )
            validated = _ensure_connector_paths_non_overlapping(
                validated,
                graph=graph,
                incoming=incoming,
                outgoing=outgoing,
                target_bus=target_bus,
                source_bus=source_bus,
                merge_link_bus=merge_link_bus,
                anchors=anchors,
                label_obstacles=label_obstacles,
                positions=positions,
                merge_entry_x=merge_entry_x,
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
            if not overlap_pairs:
                break
    _lower_bypass_corridors_clearing_crossings(
        validated,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
    )
    # Frame cuts made by the repair stages above are the one violation those stages cannot
    # undo themselves, so the generic reroute gets a pass at them first.
    _reseat_target_entry_ports(
        validated,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
        incoming=incoming,
        outgoing=outgoing,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
        merge_entry_x=merge_entry_x,
    )
    _resolve_connector_violations(
        validated,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
        incoming=incoming,
        outgoing=outgoing,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
        merge_entry_x=merge_entry_x,
    )
    for link_key, initial in preserved_frame_tail_paths.items():
        src, tgt = link_key
        target = anchors.get(tgt)
        if target is None:
            continue
        entry_x = merge_entry_x.get(link_key, target.cx)
        validated[link_key] = _complete_frame_tail_exit_path(
            initial,
            target=target,
            entry_x=entry_x,
        )
    for link_key in list(validated):
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        points = validated[link_key]
        has_upward = any(
            y2 > y1 + PARALLEL_CONNECTOR_COORD_EPS
            for (_x1, y1), (_x2, y2) in zip(points, points[1:])
        )
        if has_upward and link_key not in merge_link_bus:
            points = _flatten_upward_connector_steps(points)
        points = _remove_vertical_backtracks(points)
        points = _collapse_collinear_connector_segments(points)
        if target is not None:
            points = _restore_target_top_entry_drop(points, target=target)
        if source is not None and target is not None:
            y_exit = _connector_source_bottom_exit_y(source)
            if _connector_turn_before_clearing_source(
                points,
                y_exit=y_exit,
                source_cx=source.cx,
            ) is not None:
                points = _repair_connector_source_departure(
                    points,
                    source=source,
                    target=target,
                    link_key=link_key,
                    graph=graph,
                )
            elif _connector_path_departs_horizontally_from_source(
                points,
                source=source,
            ):
                points = _repair_connector_source_departure(
                    points,
                    source=source,
                    target=target,
                    link_key=link_key,
                    graph=graph,
                )
        validated[link_key] = points
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
    if overlap_pairs:
        validated = _ensure_connector_paths_non_overlapping(
            validated,
            graph=graph,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            merge_entry_x=merge_entry_x,
        )
    _align_merge_feed_bus_horizontals(
        validated,
        graph=graph,
        incoming=incoming,
    )
    for link_key, points in list(validated.items()):
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None or link_key not in merge_entry_x:
            continue
        validated[link_key] = _snap_connector_path_endpoints(
            points,
            source=source,
            target=target,
            link_key=link_key,
            graph=graph,
            merge_entry_x=merge_entry_x,
            target_bus=target_bus,
            merge_link_bus=merge_link_bus,
            source_bus=source_bus,
        )
    _repair_multiply_stem_side_branch_tees(
        validated,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
        incoming=incoming,
    )
    for link_key, points in list(validated.items()):
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue
        y_exit = _connector_source_bottom_exit_y(source)
        if _connector_turn_before_clearing_source(
            points,
            y_exit=y_exit,
            source_cx=source.cx,
        ) is None:
            continue
        y_stub = y_exit - CONNECTOR_EXIT_STUB
        for index in range(len(points) - 1):
            x1, y1 = points[index]
            x2, y2 = points[index + 1]
            if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS / 2:
                continue
            if abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS / 2:
                continue
            if min(abs(x1 - source.cx), abs(x2 - source.cx)) > PARALLEL_CONNECTOR_COORD_EPS / 2:
                continue
            repaired = [
                (source.cx, y_exit),
                (source.cx, y_stub),
                (x2, y_stub),
                *points[index + 1 :],
            ]
            repaired = _ensure_orthogonal_connector_path(repaired)
            validated[link_key] = _repair_connector_target_top_edge_overlap(
                repaired,
                source=source,
                target=target,
            )
            break
    _repair_horizontal_backtracking_paths(
        validated,
        graph=graph,
        anchors=anchors,
        incoming=incoming,
        outgoing=outgoing,
        label_obstacles=label_obstacles,
        positions=positions,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_entry_x=merge_entry_x,
        merge_link_bus=merge_link_bus,
        input_index=input_index,
        inline_bypass_bus_x=inline_bypass_bus_x,
        initial_link_paths=initial_link_paths,
    )
    if _find_connector_segment_crossings(validated):
        _reroute_gutter_bypass_feeds_clearing_crossings(
            validated,
            graph=graph,
            positions=positions,
            anchors=anchors,
            target_bus=target_bus,
            incoming=incoming,
            merge_entry_x=merge_entry_x,
            source_bus=source_bus,
        )
        _collapse_target_bus_entry_detours_clearing_crossings(
            validated,
            graph=graph,
            anchors=anchors,
            target_bus=target_bus,
            merge_entry_x=merge_entry_x,
        )
        _lift_horizontal_segments_clearing_crossings(
            validated,
            graph=graph,
            anchors=anchors,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
        )
    _lower_bypass_corridors_clearing_crossings(
        validated,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
    )
    # Rerouting and nudging feed each other: a reroute can leave a run grazing a tile the
    # nudge then slides clear, and a nudged run can open a corridor the reroute wanted. Run
    # the pair to a fixpoint so neither is left holding work the other could finish.
    for _ in range(4):
        before = [tuple(points) for points in validated.values()]
        # Stages pick rows and columns independently, so two can land within the tolerance
        # that reads as one column without being equal, which draws as a slant.
        for link_key, points in validated.items():
            validated[link_key] = _ensure_orthogonal_connector_path(points)
        _reroute_crowded_merge_fan_in(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            merge_entry_x=merge_entry_x,
        )
        _reseat_target_entry_ports(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            merge_entry_x=merge_entry_x,
        )
        _spread_shared_source_exit_stems(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
        )
        _shorten_wandering_connector_routes(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
        )
        _nest_source_fanout_bus_rows(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
        )
        _nest_source_fanout_legs(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
        )
        _unstack_shared_horizontal_runs(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
        )
        _unstack_shared_vertical_runs(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
        )
        _resolve_connector_violations(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            merge_entry_x=merge_entry_x,
        )
        _reroute_connector_violation_groups(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
        )
        _polish_connector_violations(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            merge_entry_x=merge_entry_x,
        )
        # Repeated after the single-link stages because they work one connector at a time and
        # can undo a whole fan-in arrangement by improving one of its feeds in isolation.
        _reroute_crowded_merge_fan_in(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            merge_entry_x=merge_entry_x,
        )
        _nudge_connector_runs_clearing_node_margins(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
        )
        if [tuple(points) for points in validated.values()] == before:
            break
    if validate_layout and _graph_requires_strict_connector_validation(graph):
        _assert_connectors_avoid_block_edge_horizontal_jogs(
            validated,
            graph=graph,
            anchors=anchors,
            stage="final",
        )
        # Validate what is drawn: every stage above may still repair the one before it, so
        # judging an intermediate state would reject layouts that come out sound.
        _assert_detail_link_paths_have_no_geometry_violations(
            validated,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
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
    """True when two or more connectors join onto a shared run into one consumer.

    A dot says "the lines meeting here are connected". That is what the reader needs where
    separate wires merge onto a common run feeding one tile. Where a single source's legs
    part company nothing joins: one value simply carries on in more than one direction, and
    a dot there claims a junction the graph does not have. Two runs that touch because they
    happen to pass the same point are not connected either, so the links meeting have to
    agree on the tile they feed.
    """
    if len(link_keys) < 2:
        return False
    if len({link_key[0] for link_key in link_keys}) < 2:
        return False
    if len({link_key[1] for link_key in link_keys}) != 1:
        return False
    del target_bus, source_bus, merge_link_bus, graph, outgoing, anchors
    # Three arms is where runs genuinely meet rather than merely cross or turn.
    return len(_connector_point_arm_directions(x, y, link_keys, link_paths, eps=eps)) >= 3


def _connector_point_arm_directions(
    x: float,
    y: float,
    link_keys: set[tuple[int, int]],
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    eps: float = PARALLEL_CONNECTOR_COORD_EPS,
) -> set[str]:
    """Directions the drawn runs leave a point in.

    Two arms are just a corner or a doubled line, and the reader needs no help reading
    either. Three or more is where a line genuinely divides, which is what a dot marks. A leg
    running through the point without turning contributes both of its directions, since that
    is what the reader sees whether or not the route happens to hold a vertex there.
    """
    directions: set[str] = set()
    for link_key in link_keys:
        path = _dedupe_polyline_points(link_paths[link_key], eps=eps)
        for index, (px, py) in enumerate(path):
            if abs(px - x) > eps or abs(py - y) > eps:
                continue
            for neighbour in (index - 1, index + 1):
                if not 0 <= neighbour < len(path):
                    continue
                nx, ny = path[neighbour]
                if abs(nx - px) > eps:
                    directions.add("right" if nx > px else "left")
                elif abs(ny - py) > eps:
                    directions.add("up" if ny > py else "down")
        for (first_x, first_y), (second_x, second_y) in zip(path, path[1:]):
            if abs(first_x - second_x) <= eps:
                if abs(x - first_x) <= eps and min(first_y, second_y) + eps < y < max(
                    first_y, second_y
                ) - eps:
                    directions.update(("up", "down"))
            elif abs(first_y - second_y) <= eps and abs(y - first_y) <= eps:
                if min(first_x, second_x) + eps < x < max(first_x, second_x) - eps:
                    directions.update(("left", "right"))
    return directions


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
    """Find T joins where a feed drops onto the bus another feed of its target runs along."""
    joins: set[tuple[float, float]] = set()
    outgoing = outgoing or {}
    anchors = anchors or {}
    del outgoing, anchors, graph
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
                # Only feeds of one tile meeting on its approach bus are joined; anything
                # else touching this run is a different value passing by.
                if link_a == link_b or link_a[1] != link_b[1] or link_a[0] == link_b[0]:
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

    # Where a source's legs divide, the leg that turns off has a vertex at the split but the
    # leg carrying straight on just passes through it. Picking the passing leg up from its
    # segments is what lets the split be seen as shared rather than as one link's corner.
    for key, (point_x, point_y) in representatives.items():
        for link_key, points in link_paths.items():
            if link_key not in internal_vertex_links[key] and _connector_path_passes_through(
                points,
                point_x,
                point_y,
                eps=eps,
            ):
                internal_vertex_links[key].add(link_key)

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


def _connector_path_passes_through(
    points: list[tuple[float, float]],
    x: float,
    y: float,
    *,
    eps: float,
) -> bool:
    """True when the drawn path runs through a point without turning there."""
    for (first_x, first_y), (second_x, second_y) in zip(points, points[1:]):
        if abs(first_x - second_x) <= eps:
            if abs(x - first_x) > eps:
                continue
            low, high = sorted((first_y, second_y))
            if low + eps < y < high - eps:
                return True
        elif abs(first_y - second_y) <= eps:
            if abs(y - first_y) > eps:
                continue
            low, high = sorted((first_x, second_x))
            if low + eps < x < high - eps:
                return True
    return False


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
) -> bool:
    for obs in obstacles:
        if (
            x + halo_radius > obs.left
            and x - halo_radius < obs.right
            and y + halo_radius > obs.bottom
            and y - halo_radius < obs.top
        ):
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


def _tee_branch_avoiding_vertical_obstacles(
    source: _RenderAnchor,
    target: _RenderAnchor,
    entry_x: float,
    tee_y: float,
    obstacles: list[_RenderAnchor],
    *,
    gap: float = 0.04,
) -> list[tuple[float, float]]:
    """Branch from a shared tee, detouring around blocks blocking the entry column."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    y2 = _connector_target_top_entry_y(target, gap=gap)
    y_stub = y1 - CONNECTOR_EXIT_STUB
    channel_y = max(tee_y, _connector_min_bus_y_above_target(target, gap=gap))
    if channel_y <= y_stub + PARALLEL_CONNECTOR_COORD_EPS:
        if channel_y < y_stub - PARALLEL_CONNECTOR_COORD_EPS:
            direct = [(source.cx, y1), (source.cx, channel_y), (entry_x, channel_y), (entry_x, y2)]
        else:
            direct = [(source.cx, y1), (source.cx, y_stub), (entry_x, y_stub), (entry_x, y2)]
    else:
        direct = [
            (source.cx, y1),
            (source.cx, y_stub),
            (source.cx, channel_y),
            (entry_x, channel_y),
            (entry_x, y2),
        ]
    if not _path_penetrates_obstacle_tiles(
        direct,
        obstacles,
        margin=CONNECTOR_OBSTACLE_MARGIN,
    ):
        return direct
    margin = CONNECTOR_OBSTACLE_MARGIN
    blockers = [
        obstacle
        for obstacle in obstacles
        if _vertical_segment_crosses_anchor(
            entry_x,
            channel_y,
            y2,
            obstacle,
            margin=margin,
        )
    ]
    if not blockers:
        for index in range(len(direct) - 1):
            x1, y1 = direct[index]
            x2, y2_seg = direct[index + 1]
            if abs(y1 - y2_seg) > PARALLEL_CONNECTOR_COORD_EPS:
                continue
            if abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS:
                continue
            for obstacle in obstacles:
                if _path_horizontal_segments_overlap_bounds(
                    [(x1, y1), (x2, y2_seg)],
                    obstacle,
                    margin=margin,
                ):
                    blockers.append(obstacle)
    if not blockers:
        return direct
    blockers = list({id(obstacle): obstacle for obstacle in blockers}.values())
    if entry_x <= source.cx + margin:
        right_x = max(
            max(obstacle.right for obstacle in blockers),
            target.right,
        ) + margin
        approach_y = max(channel_y, _connector_min_bus_y_above_target(target, gap=gap))
        top_entry_detour = [
            (source.cx, y1),
            (source.cx, channel_y),
            (right_x, channel_y),
            (right_x, approach_y),
            (entry_x, approach_y),
            (entry_x, y2),
        ]
        if not _path_penetrates_obstacle_tiles(
            top_entry_detour,
            obstacles,
            margin=margin,
        ):
            return top_entry_detour
    detour_x = max(obstacle.right for obstacle in blockers) + margin
    if detour_x <= max(source.cx, entry_x) + margin:
        detour_x = max(source.cx, entry_x) + margin
    detour_y = max(obstacle.bottom for obstacle in blockers) + margin
    return [
        (source.cx, y1),
        (source.cx, channel_y),
        (detour_x, channel_y),
        (detour_x, detour_y),
        (entry_x, detour_y),
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

    def _entry_approach_y() -> float:
        return max(bus_y, _connector_min_bus_y_above_target(target, gap=gap))

    def _cross_column_bus_route(approach_y: float) -> list[tuple[float, float]]:
        route_y = max(approach_y, _connector_min_bus_y_above_target(target, gap=gap))
        if route_y <= stub_y + PARALLEL_CONNECTOR_COORD_EPS:
            if route_y < stub_y - PARALLEL_CONNECTOR_COORD_EPS:
                return [
                    (source.cx, y1),
                    (source.cx, route_y),
                    (entry_x, route_y),
                    (entry_x, y2),
                ]
            return [
                (source.cx, y1),
                (source.cx, stub_y),
                (entry_x, stub_y),
                (entry_x, y2),
            ]
        return [
            (source.cx, y1),
            (source.cx, stub_y),
            (source.cx, route_y),
            (entry_x, route_y),
            (entry_x, y2),
        ]

    def _straight_on_bus() -> list[tuple[float, float]]:
        approach_y = _entry_approach_y()
        if abs(source.cx - entry_x) < 0.06:
            if approach_y >= y2 + PARALLEL_CONNECTOR_COORD_EPS:
                if approach_y <= stub_y + PARALLEL_CONNECTOR_COORD_EPS:
                    route = [
                        (source.cx, y1),
                        (source.cx, stub_y),
                        (source.cx, approach_y),
                        (entry_x, y2),
                    ]
                else:
                    route = [
                        (source.cx, y1),
                        (source.cx, stub_y),
                        (source.cx, approach_y),
                        (entry_x, approach_y),
                        (entry_x, y2),
                    ]
                return _prepend_spine(route)
            return [(source.cx, y1), (source.cx, y2)]
        return _prepend_spine(_cross_column_bus_route(approach_y))

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
        branch_y = (
            spine_tee_y
            if prefer_tee and spine_tee_y is not None
            else stub_y
        )
        return _prepend_spine(
            [
                (source.cx, y1),
                (source.cx, branch_y),
                (gutter_x, branch_y),
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
    stub_bus_y = max(
        _spread_top_entry_bus_y(source, target, gap=gap),
        _connector_min_bus_y_above_target(target, gap=gap),
    )
    if (
        stub_bus_y < bus_y - PARALLEL_CONNECTOR_COORD_EPS
        and source.bottom > target.top + 0.5
        and abs(source.cx - entry_x) > 0.06
    ):
        candidates.insert(
            0,
            _prepend_spine(
                [
                    (source.cx, y1),
                    (source.cx, stub_bus_y),
                    (entry_x, stub_bus_y),
                    (entry_x, y2),
                ]
            ),
        )
    if abs(source.cx - entry_x) > 0.06:
        tee_branch = _tee_branch_avoiding_vertical_obstacles(
            source,
            target,
            entry_x,
            bus_y,
            check_obstacles,
            gap=gap,
        )
        candidates.append(_prepend_spine(tee_branch))
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
        if (
            _connector_turn_before_clearing_source(
                candidate,
                y_exit=y1,
                source_cx=source.cx,
            )
            is not None
        ):
            continue
        if _path_crosses_attached_block_edge_band(
            candidate,
            source=source,
            target=target,
        ):
            continue
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


def _frame_tail_below_border_connector_points(
    source: _RenderAnchor,
    *,
    gutter_x: float,
    corridor_y: float,
    entry_x: float,
    entry_y: float,
    bus_y: float | None = None,
    gap: float = 0.04,
    frame_bounds=None,
) -> list[tuple[float, float]]:
    """Drop below the dotted frame border, route in the gutter, then enter the target."""
    y1 = _connector_source_bottom_exit_y(source, gap=gap)
    # A frame tail must leave through the bottom edge before moving sideways.
    # Moving horizontally from the tile edge creates a loop through the frame and
    # can make the connector look detached from the source tile.
    points: list[tuple[float, float]] = [
        (source.cx, y1),
        (source.cx, corridor_y),
        (gutter_x, corridor_y),
    ]
    tap_y = corridor_y
    if bus_y is not None:
        tap_y = min(bus_y, corridor_y)
    if abs(tap_y - points[-1][1]) > PARALLEL_CONNECTOR_COORD_EPS:
        points.append((points[-1][0], tap_y))
    if abs(points[-1][0] - entry_x) > PARALLEL_CONNECTOR_COORD_EPS:
        points.append((entry_x, points[-1][1]))
    points.append((entry_x, entry_y))
    return _ensure_orthogonal_connector_path(points)


def _frame_tail_merge_entry_connector_points(
    source: _RenderAnchor,
    target: _RenderAnchor,
    *,
    exit_x: float,
    entry_x: float,
    bus_y: float,
    frame_bounds,
    gap: float = 0.04,
    obstacles: list[_RenderAnchor] | None = None,
) -> list[tuple[float, float]]:
    """Route an inline-frame tail to a labeled merge port without crossing the frame."""
    corridor_y = _frame_tail_routing_corridor_y(frame_bounds, source, target)
    pipeline_corridor_y = (
        frame_bounds.bottom
        - CONNECTOR_OBSTACLE_MARGIN
        - CONNECTOR_EXIT_STUB
        - PIPELINE_MERGE_BUS_BELOW_FRAME_GAP
    )
    entry_y = _connector_target_top_entry_y(target, gap=gap)
    entry_bus_y = min(
        bus_y,
        _connector_min_bus_y_above_target(target, gap=gap),
    )
    use_below_border = entry_bus_y > pipeline_corridor_y + PARALLEL_CONNECTOR_COORD_EPS
    if not use_below_border:
        direct = _ensure_orthogonal_connector_path(
            [
                (source.cx, _connector_source_bottom_exit_y(source, gap=gap)),
                (source.cx, entry_bus_y),
                (entry_x, entry_bus_y),
                (entry_x, entry_y),
            ]
        )
        if not _path_hits_obstacles(
            direct,
            list(obstacles or []),
            margin=CONNECTOR_OBSTACLE_MARGIN,
        ):
            return direct
    left_gutter = min(
        exit_x,
        frame_bounds.left - CONNECTOR_OBSTACLE_MARGIN - CONNECTOR_EXIT_STUB,
    )
    right_gutter = max(
        exit_x,
        frame_bounds.right + CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_EXIT_STUB,
    )
    gutter_candidates = (
        (left_gutter, right_gutter)
        if entry_x <= source.cx
        else (right_gutter, left_gutter)
    )
    fallback: list[tuple[float, float]] | None = None
    below_corridor_y = min(corridor_y, pipeline_corridor_y)
    for gutter_x in gutter_candidates:
        candidate = _frame_tail_below_border_connector_points(
            source,
            gutter_x=gutter_x,
            corridor_y=below_corridor_y,
            entry_x=entry_x,
            entry_y=entry_y,
            bus_y=entry_bus_y,
            gap=gap,
            frame_bounds=frame_bounds,
        )
        fallback = fallback or candidate
        if not _path_hits_obstacles(
            candidate,
            list(obstacles or []),
            margin=CONNECTOR_OBSTACLE_MARGIN,
        ):
            return candidate
    return fallback or _same_column_straight_connector_points(source, target, gap=gap)


def _complete_frame_tail_exit_path(
    points: list[tuple[float, float]],
    *,
    target: _RenderAnchor,
    entry_x: float,
) -> list[tuple[float, float]]:
    """Finish a below-frame corridor route into the target's top entry port."""
    entry_y = _connector_target_top_entry_y(target)
    completed = list(points)
    if abs(completed[-1][0] - entry_x) > PARALLEL_CONNECTOR_COORD_EPS:
        completed.append((entry_x, completed[-1][1]))
    if abs(completed[-1][1] - entry_y) > PARALLEL_CONNECTOR_COORD_EPS:
        completed.append((entry_x, entry_y))
    return _ensure_orthogonal_connector_path(completed)


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
    graph=None,
    positions: list | None = None,
    link_key: tuple[int, int] | None = None,
) -> list[tuple[float, float]]:
    """Leave an inline-frame column downward before joining a shared merge bus."""
    x2 = entry_x if entry_x is not None else target.cx
    if frame_bounds is not None:
        from visualizer.computation_graph import _graph_has_tensor_ports

        if entry_x is not None and graph is not None and _graph_has_tensor_ports(graph):
            route_obstacles = list(obstacles or [])
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
                graph=graph,
                positions=positions,
                link_key=link_key,
            )
        corridor_y = _frame_tail_routing_corridor_y(frame_bounds, source, target)
        gutter_x = min(
            exit_x,
            frame_bounds.left - CONNECTOR_OBSTACLE_MARGIN - CONNECTOR_EXIT_STUB,
        )
        entry_y = _connector_target_top_entry_y(target, gap=gap)
        below_border = _frame_tail_below_border_connector_points(
            source,
            gutter_x=gutter_x,
            corridor_y=corridor_y,
            entry_x=x2,
            entry_y=entry_y,
            bus_y=bus_y,
            gap=gap,
            frame_bounds=frame_bounds,
        )
        route_obstacles = list(obstacles or [])
        if not _path_hits_obstacles(
            below_border,
            route_obstacles,
            margin=CONNECTOR_OBSTACLE_MARGIN,
        ):
            return below_border
        return below_border
    route_obstacles = list(obstacles or [])
    return _shared_merge_bus_connector_points(
        source,
        target,
        x2,
        bus_y,
        route_obstacles,
        exit_x=exit_x,
        y_stub=None,
        gap=gap,
        graph=graph,
        positions=positions,
        link_key=link_key,
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


def _inline_frame_transitive_links(graph, frame) -> list[tuple[int, int]]:
    """In-frame links that also have a longer route between the same two steps.

    Such a link is a skip: the steps on the longer route sit between its ends, so
    the connector has to pass beside them instead of running down their column.
    """
    members = set(frame.node_indices)
    inside = [(src, tgt) for src, tgt in graph.links if src in members and tgt in members]
    successors: dict[int, list[int]] = {}
    for src, tgt in inside:
        successors.setdefault(src, []).append(tgt)

    def _reaches_via_detour(src: int, tgt: int) -> bool:
        pending = [step for step in successors.get(src, []) if step != tgt]
        seen = set(pending)
        while pending:
            current = pending.pop()
            if current == tgt:
                return True
            for step in successors.get(current, []):
                if step not in seen:
                    seen.add(step)
                    pending.append(step)
        return False

    return [(src, tgt) for src, tgt in inside if _reaches_via_detour(src, tgt)]


def _inline_frame_skipped_steps(
    graph,
    frame,
    link_key: tuple[int, int],
) -> list[int]:
    """Frame steps a skip link has to pass on its way to its target.

    Only steps that share the link's column count: the layout offsets the long arm
    of an outer bypass into a column of its own, so a skip spanning that arm has a
    clear lane and passes nothing.
    """
    from visualizer.computation_graph import (
        _offset_column_nodes,
        _ordered_inline_frame_chain,
    )

    chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
    rank = {index: position for position, index in enumerate(chain)}
    src, tgt = link_key
    if src not in rank or tgt not in rank or rank[tgt] - rank[src] < 2:
        return []
    offset = _offset_column_nodes(graph, frame)
    if (src in offset) != (tgt in offset):
        return []
    return [
        step
        for step in chain[rank[src] + 1 : rank[tgt]]
        if (step in offset) == (src in offset)
    ]


def _is_inline_frame_skip_link(
    graph,
    link_key: tuple[int, int],
) -> bool:
    """True when a link bypasses intermediate rows inside one inline frame."""
    src, tgt = link_key
    frame = _inline_frame_for_nodes(graph, src, tgt)
    if frame is None:
        return False
    from visualizer.computation_graph import _inline_frame_column_skip_links

    return link_key in _inline_frame_column_skip_links(graph, frame)


def _is_inline_frame_spine_link(
    graph,
    link_key: tuple[int, int],
    *,
    positions: list | None = None,
    anchors: dict[int, _RenderAnchor] | None = None,
) -> bool:
    """True for consecutive same-column links inside one inline frame's main chain."""
    from visualizer.computation_graph import (
        _inline_frame_column_skip_links,
        _ordered_inline_frame_chain,
    )

    src, tgt = link_key
    frame = _inline_frame_for_nodes(graph, src, tgt)
    if frame is None:
        return False
    if link_key in _inline_frame_column_skip_links(graph, frame):
        return False
    chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
    if src not in chain or tgt not in chain:
        return False
    if chain.index(tgt) - chain.index(src) != 1:
        return False
    if positions is not None and anchors is not None:
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            return False
        if abs(source.cx - target.cx) > TOP_ENTRY_PORT_GAP:
            return False
    return True


def _inline_frame_bypass_links(graph, frame) -> list[tuple[int, int]]:
    """Top-entry links that skip intermediate rows inside one inline frame."""
    from visualizer.computation_graph import _inline_frame_column_skip_links

    return _inline_frame_column_skip_links(graph, frame)


def _tile_span(position) -> tuple[float, float, float, float]:
    """(left, right, bottom, top) of one laid-out tile."""
    return (
        position.cx - position.width / 2,
        position.cx + position.width / 2,
        position.top_y - position.height,
        position.top_y,
    )


def _inline_frame_tiles_between(
    frame,
    positions: list,
    link: tuple[int, int],
) -> list[int]:
    """Frame tiles a bypass connector has to get past, from the laid-out geometry.

    A tile is in the way when it sits in the rows the connector spans and reaches
    into the horizontal band its two ends occupy.
    """
    src, tgt = link
    src_left, src_right, src_bottom, src_top = _tile_span(positions[src])
    tgt_left, tgt_right, tgt_bottom, tgt_top = _tile_span(positions[tgt])
    row_low = min(src_bottom, tgt_top)
    row_high = max(src_bottom, tgt_top)
    band_left = min(src_left, tgt_left)
    band_right = max(src_right, tgt_right)
    between: list[int] = []
    for step in frame.node_indices:
        if step in link or step >= len(positions):
            continue
        left, right, bottom, top = _tile_span(positions[step])
        if top <= row_low or bottom >= row_high:
            continue
        if right <= band_left or left >= band_right:
            continue
        between.append(step)
    return between


def _inline_frame_bypass_corridors(
    graph,
    frame,
    positions: list,
    *,
    channel_gap: float = PARALLEL_CONNECTOR_CHANNEL_GAP,
) -> dict[tuple[int, int], float]:
    """Corridor column for every bypass connector routed inside one inline frame.

    A corridor hugs the tiles its connector passes rather than the frame border, so
    a skip local to one column stays beside that column instead of crossing the
    frame. Connectors whose vertical spans miss each other share a corridor.
    """
    links = [
        link
        for link in _inline_frame_bypass_links(graph, frame)
        if link[0] < len(positions) and link[1] < len(positions)
    ]
    if not links:
        return {}
    frame_left, frame_right, _, _ = _inline_frame_tile_envelope(frame, positions)

    spans: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for link in links:
        passed = [
            positions[step]
            for step in _inline_frame_tiles_between(frame, positions, link)
        ]
        boxes = [positions[link[0]], positions[link[1]], *passed]
        edges = [_tile_span(position) for position in boxes]
        spans[link] = (
            min(edge[0] for edge in edges),
            max(edge[1] for edge in edges),
            min(edge[2] for edge in edges),
            max(edge[3] for edge in edges),
        )

    corridors: dict[tuple[int, int], float] = {}
    occupied: dict[tuple[str, int], list[tuple[float, float]]] = {}
    def _free_channel(side: str, bottom: float, top: float) -> int:
        channel = 0
        while any(
            other_bottom < top and bottom < other_top
            for other_bottom, other_top in occupied.get((side, channel), [])
        ):
            channel += 1
        return channel

    for link in sorted(links, key=lambda pair: spans[pair][3], reverse=True):
        left, right, bottom, top = spans[link]
        source_x = positions[link[0]].cx
        target_x = positions[link[1]].cx
        if target_x < source_x - PARALLEL_CONNECTOR_COORD_EPS:
            near = "left"
        elif target_x > source_x + PARALLEL_CONNECTOR_COORD_EPS:
            near = "right"
        else:
            near = "left" if left - frame_left <= frame_right - right else "right"
        far = "right" if near == "left" else "left"
        # Overlapping corridors read better facing each other than stacked on one
        # side, so only fall back to a further channel once both sides are taken.
        if abs(target_x - source_x) > PARALLEL_CONNECTOR_COORD_EPS:
            side = near
        else:
            side = min(
                (near, far),
                key=lambda option: (_free_channel(option, bottom, top), option != near),
            )
        channel = _free_channel(side, bottom, top)
        occupied.setdefault((side, channel), []).append((bottom, top))
        # Clear the obstacle margin outright: a corridor sitting exactly on it still
        # counts as touching the tiles it passes.
        offset = (
            CONNECTOR_OBSTACLE_MARGIN
            + CONNECTOR_ATTACHED_BOX_MARGIN
            + channel * channel_gap
        )
        corridors[link] = left - offset if side == "left" else right + offset
    return corridors


def _inline_frame_bypass_link_count(graph, frame) -> int:
    """Count top-entry bypass operand links routed inside one inline frame."""
    return len(_inline_frame_bypass_links(graph, frame))


def _inline_frame_connector_gutter_width(
    graph,
    frame,
    positions: list,
    *,
    side: str = "left",
) -> float:
    """Width the bypass corridors on one side reach past the frame's tiles."""
    corridors = _inline_frame_bypass_corridors(graph, frame, positions)
    if not corridors:
        return 0.0
    frame_left, frame_right, _, _ = _inline_frame_tile_envelope(frame, positions)
    if side == "left":
        overhang = frame_left - min(corridors.values())
    else:
        overhang = max(corridors.values()) - frame_right
    if overhang <= 0:
        return 0.0
    return overhang + CONNECTOR_OBSTACLE_MARGIN


def _estimate_inline_frame_gutter_width(graph, frame) -> float:
    """Upper bound on bypass corridor width, for use before tiles are laid out."""
    count = len(_inline_frame_bypass_links(graph, frame))
    if not count:
        return 0.0
    return 2 * CONNECTOR_OBSTACLE_MARGIN + count * PARALLEL_CONNECTOR_CHANNEL_GAP


def _inline_frame_total_connector_gutter_width(graph, frame, positions: list) -> float:
    """Combined left and right gutter reservation for one inline frame."""
    return _inline_frame_connector_gutter_width(
        graph, frame, positions, side="left"
    ) + _inline_frame_connector_gutter_width(graph, frame, positions, side="right")


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


def _inline_frames_nested_within(graph, frame) -> list:
    """Frames whose members are a strict subset of this frame's members."""
    members = set(frame.node_indices)
    return [
        other
        for other in getattr(graph, "inline_frames", None) or []
        if other.frame_id != frame.frame_id and set(other.node_indices) < members
    ]


def _inline_frame_nesting_depth(graph, frame) -> int:
    """How many dotted borders sit inside this frame at its deepest point."""
    nested = _inline_frames_nested_within(graph, frame)
    if not nested:
        return 0
    return 1 + max(_inline_frame_nesting_depth(graph, other) for other in nested)


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
    left_gutter = _inline_frame_connector_gutter_width(graph, frame, positions, side="left")
    right_gutter = _inline_frame_connector_gutter_width(graph, frame, positions, side="right")
    bounds = ContentBounds(
        left=min_left - pad - left_gutter,
        right=max_right + pad + right_gutter,
        bottom=min_bottom - pad - CONNECTOR_ATTACHED_BOX_MARGIN,
        top=max_top + pad,
    )
    # A frame drawn around another frame's tiles has to clear that frame's border as
    # well, otherwise the two dotted rectangles land on top of each other. Only the
    # sides a nested border reaches need the extra room.
    for nested in _inline_frames_nested_within(graph, frame):
        inner = _inline_frame_draw_bounds(nested, positions, graph, pad=pad)
        bounds = ContentBounds(
            left=min(bounds.left, inner.left - pad),
            right=max(bounds.right, inner.right + pad),
            bottom=min(bounds.bottom, inner.bottom - pad),
            top=max(bounds.top, inner.top + pad),
        )
    return bounds


def _inline_frame_connector_portals(
    points: list[tuple[float, float]],
    bounds,
) -> list[tuple[float, float]]:
    """Return frame-border crossings where a connector needs a clean portal."""
    portals: list[tuple[float, float]] = []
    eps = PARALLEL_CONNECTOR_COORD_EPS
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if abs(x1 - x2) <= eps:
            low, high = sorted((y1, y2))
            if bounds.left - eps <= x1 <= bounds.right + eps:
                for border_y in (bounds.bottom, bounds.top):
                    if low - eps <= border_y <= high + eps:
                        portals.append((x1, border_y))
        elif abs(y1 - y2) <= eps:
            low, high = sorted((x1, x2))
            if bounds.bottom - eps <= y1 <= bounds.top + eps:
                for border_x in (bounds.left, bounds.right):
                    if low - eps <= border_x <= high + eps:
                        portals.append((border_x, y1))
    return list(dict.fromkeys(portals))


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
    inline_bypass_bus_x: dict[tuple[int, int], float] | None = None,
) -> list[tuple[float, float]] | None:
    """Return connector polyline points for one graph link (same routing as draw)."""
    source = anchors.get(src)
    target = anchors.get(tgt)
    if source is None or target is None:
        return None

    route_obstacles = [
        anchor for node_index, anchor in anchors.items() if node_index not in {src, tgt}
    ] + label_obstacles
    gap = 0.04
    spread_entry_x = merge_entry_x.get(link_key)
    backward_route = _backward_top_entry_gutter_route(
        source,
        target,
        _connector_block_obstacles(
            anchors,
            src=src,
            tgt=tgt,
            label_obstacles=label_obstacles,
            graph=graph,
            positions=positions,
            link_key=link_key,
        ),
        entry_x=spread_entry_x,
        channel=(src * 37 + tgt) % 7,
    )
    if backward_route is not None:
        return backward_route
    use_straight_stack = (
        abs(source.cx - target.cx) < 0.08
        and source.bottom
        >= target.top - (CONNECTOR_OBSTACLE_MARGIN + PARALLEL_CONNECTOR_COORD_EPS)
        and tgt not in target_bus
        and link_key not in merge_link_bus
        and (
            spread_entry_x is None
            or abs(spread_entry_x - target.cx) < PARALLEL_CONNECTOR_COORD_EPS
        )
    )
    if use_straight_stack:
        y1 = _connector_source_bottom_exit_y(source, gap=gap)
        y2 = _connector_target_top_entry_y(target, gap=gap)
        if y1 >= y2 - PARALLEL_CONNECTOR_COORD_EPS:
            straight = [(source.cx, y1), (target.cx, y2)]
            blocked = any(
                _vertical_segment_crosses_anchor(
                    source.cx,
                    y2,
                    y1,
                    obstacle,
                    margin=CONNECTOR_OBSTACLE_MARGIN,
                )
                for obstacle in route_obstacles
            )
            if not blocked:
                return straight
            for detour_builder in (
                lambda: _same_column_side_gutter_detour(
                    source, target, route_obstacles
                ),
                lambda: _orthogonal_path(
                    source,
                    target,
                    route_obstacles,
                    gap=gap,
                    graph=graph,
                    positions=positions,
                ),
            ):
                orth = detour_builder()
                if (
                    not _path_hits_obstacles(
                        orth,
                        route_obstacles,
                        margin=CONNECTOR_OBSTACLE_MARGIN,
                    )
                    and not _path_penetrates_attached_boxes(orth, source, target)
                    and not _connector_path_has_block_edge_horizontal_jog(
                        orth,
                        source=source,
                        target=target,
                        link_key=link_key,
                        graph=graph,
                    )
                ):
                    return orth
    if input_index is not None and src == input_index:
        target_frame = next(
            (frame for frame in graph.inline_frames if tgt in frame.node_indices),
            None,
        )
        use_shared_input_fanout = (
            src in source_bus
            and outgoing is not None
            and len(_fanout_links_excluding_bypasses(graph, outgoing.get(src, [])))
            >= SHARED_SOURCE_BUS_MIN_LINKS
        )
        if use_shared_input_fanout and target_frame is not None:
            tee_y = source_bus[src]
            frame_bounds = _inline_frame_draw_bounds(target_frame, positions, graph)
            if (
                not _requires_shared_input_source_bus(graph, tgt)
                and tee_y < frame_bounds.top - PARALLEL_CONNECTOR_COORD_EPS
                and _path_horizontal_segments_overlap_bounds(
                    [
                        (min(source.cx, target.cx), tee_y),
                        (max(source.cx, target.cx), tee_y),
                    ],
                    frame_bounds,
                    margin=CONNECTOR_OBSTACLE_MARGIN,
                )
            ):
                use_shared_input_fanout = False
        if not use_shared_input_fanout:
            if target_frame is not None:
                y1 = _connector_source_bottom_exit_y(source, gap=gap)
                y2 = _connector_target_top_entry_y(target, gap=gap)
                bypass_x = max(source.right, target.right) + CONNECTOR_OBSTACLE_MARGIN
                if target_frame.node_indices and target_frame.node_indices[0] == tgt:
                    route_y = _inline_frame_top_member_route_y(
                        source,
                        target,
                        target_frame,
                        positions,
                        graph,
                        gap=gap,
                    )
                    direct = _ensure_orthogonal_connector_path(
                        [
                            (source.cx, y1),
                            (source.cx, route_y),
                            (target.cx, route_y),
                            (target.cx, y2),
                        ]
                    )
                    if not _path_hits_obstacles(
                        direct,
                        route_obstacles,
                        margin=CONNECTOR_OBSTACLE_MARGIN,
                    ) and not _path_penetrates_attached_boxes(direct, source, target):
                        return direct
                    bypass_x = max(source.right, target.right) + CONNECTOR_OBSTACLE_MARGIN
                    bypass_x = _right_bypass_x_clearing_horizontal_segment(
                        min(source.cx, target.cx),
                        route_y,
                        route_obstacles,
                        initial_bypass_x=bypass_x,
                    )
                    return _ensure_orthogonal_connector_path(
                        [
                            (source.cx, y1),
                            (source.cx, route_y),
                            (bypass_x, route_y),
                            (target.cx, route_y),
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
                y_stub = y1 - CONNECTOR_EXIT_STUB
                return _ensure_orthogonal_connector_path(
                    [
                        (source.cx, y1),
                        (source.cx, y_stub),
                        (bypass_x, y_stub),
                        (bypass_x, gutter_y),
                        (target.cx, gutter_y),
                        (target.cx, y2),
                    ]
                )
        else:
            tee_y = source_bus[src]
            entry_x = merge_entry_x.get(link_key, target.cx)
            y1 = _connector_source_bottom_exit_y(source, gap=gap)
            y2 = _connector_target_top_entry_y(target, gap=gap)
            leg_bus = _fanout_leg_routing_bus_y(
                link_key,
                graph=graph,
                outgoing=outgoing,
                target_bus=target_bus,
                merge_link_bus=merge_link_bus,
                tee_y=tee_y,
                anchors=anchors,
            )
            full_obstacles = _connector_block_obstacles(
                anchors,
                src=src,
                tgt=tgt,
                label_obstacles=label_obstacles,
                graph=graph,
                positions=positions,
                link_key=link_key,
            )

            def _fanout_route_is_valid(points: list[tuple[float, float]]) -> bool:
                return _connector_path_clear_of_blocks(
                    points,
                    source=source,
                    target=target,
                    obstacles=full_obstacles,
                )

            route = _fanout_tee_then_entry_column_points(
                source,
                target,
                entry_x,
                tee_y=tee_y,
                bus_y=leg_bus,
                gap=gap,
            )
            if _fanout_route_is_valid(route):
                return route
            detour = _tee_branch_avoiding_vertical_obstacles(
                source,
                target,
                entry_x,
                tee_y,
                full_obstacles,
                gap=gap,
            )
            if _fanout_route_is_valid(detour):
                return detour
            gutter = _fanout_split_branch_gutter_route(
                source,
                target,
                entry_x,
                tee_y,
                full_obstacles,
                gap=gap,
            )
            if _fanout_route_is_valid(gutter):
                return gutter
            bypass_x = max(source.right, target.right) + CONNECTOR_OBSTACLE_MARGIN
            bypass_x = _right_bypass_x_clearing_vertical_segment(
                tee_y,
                y2,
                route_obstacles,
                initial_bypass_x=bypass_x,
            )
            route_y = y2 + CONNECTOR_EXIT_STUB
            gutter = _ensure_orthogonal_connector_path(
                [
                    (source.cx, y1),
                    (source.cx, tee_y),
                    (bypass_x, tee_y),
                    (bypass_x, route_y),
                    (entry_x, route_y),
                    (entry_x, y2),
                ]
            )
            if _fanout_route_is_valid(gutter):
                return gutter
    target_spec = positions[tgt].spec
    floating_port = target_spec.port_style == "floating"
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
    elif bus_y is None and tgt in target_bus:
        bus_y = target_bus[tgt]
        bus_near = "target"
    elif bus_y is None and src in source_bus:
        bus_y = source_bus[src]
        bus_near = "source"

    shared_frame = _inline_frame_for_nodes(graph, src, tgt)
    frame_bounds = (
        _inline_frame_draw_bounds(shared_frame, positions, graph)
        if shared_frame is not None
        else None
    )
    if (
        shared_frame is not None
        and link_key in _inline_frame_bypass_links(graph, shared_frame)
    ):
        # The frame reserved a gutter and the rows to reach it, so take that route
        # from the start: the steps in between leave no detour to find later.
        skip_route = _inline_skip_top_entry_connector_points(
            source,
            target,
            bus_x=(inline_bypass_bus_x or {}).get(link_key),
            entry_x=merge_entry_x.get(link_key),
            frame_bounds=frame_bounds,
            positions=positions,
            exclude={src, tgt},
        )
        if skip_route is not None:
            return skip_route
    if floating_port:
        return _orthogonal_path(source, target, route_obstacles, bus_near=bus_near, bus_y=bus_y)
    merge_bus_y = merge_link_bus.get(link_key)
    if merge_bus_y is None and tgt in target_bus:
        merge_bus_y = bus_y
    if (
        tgt in target_bus
        and merge_bus_y is not None
    ):
        from visualizer.computation_graph import (
            _graph_has_tensor_ports,
            _inline_frame_tail_indices,
        )

        if link_key in merge_entry_x:
            spread_links_to_target = [
                link for link in merge_entry_x if link[1] == tgt
            ]
            if (
                tgt not in target_bus
                and abs(source.cx - target.cx) < 0.08
                and len(spread_links_to_target) > 1
            ):
                spread_route = _same_column_spread_top_entry_connector_points(
                    source,
                    target,
                    merge_entry_x[link_key],
                    gap=gap,
                )
                if (
                    not _path_hits_obstacles(
                        spread_route,
                        route_obstacles,
                        margin=CONNECTOR_OBSTACLE_MARGIN,
                    )
                    and not _path_penetrates_attached_boxes(spread_route, source, target)
                ):
                    return spread_route

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
        return _tensor_port_connector_points(source, target, route_obstacles, gap=gap)
    if (
        tgt in target_bus
        and bus_y is not None
    ):
        from visualizer.computation_graph import (
            _graph_has_tensor_ports,
            _inline_frame_tail_indices,
        )

        if _graph_has_tensor_ports(graph) and src in _inline_frame_tail_indices(graph):
            # Leaving the frame at its foot only works while the foot is still above the
            # port being fed. Where the target stands higher than the frame's exit, riding
            # that corridor would drop under the tile and climb back up through it, so the
            # link takes the ordinary route instead.
            corridor_bus = merge_link_bus.get(link_key, bus_y)
            if fanout_tee_y is None and corridor_bus > _connector_target_top_entry_y(
                target, gap=gap
            ):
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
                        bus_y=corridor_bus,
                        frame_bounds=draw_bounds,
                        gap=gap,
                        obstacles=route_obstacles,
                        entry_x=merge_entry_x.get(link_key),
                        graph=graph,
                        positions=positions,
                        link_key=link_key,
                    )
    if link_key in merge_entry_x and tgt in target_bus:
        spread_links_to_target = [
            link for link in merge_entry_x if link[1] == tgt
        ]
        if (
            abs(source.cx - target.cx) < 0.08
            and len(spread_links_to_target) > 1
            and tgt not in target_bus
        ):
            spread_route = _same_column_spread_top_entry_connector_points(
                source,
                target,
                merge_entry_x[link_key],
                gap=gap,
            )
            if (
                not _path_hits_obstacles(
                    spread_route,
                    route_obstacles,
                    margin=CONNECTOR_OBSTACLE_MARGIN,
                )
                and not _path_penetrates_attached_boxes(spread_route, source, target)
            ):
                return spread_route
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
    ):
        if abs(source.cx - target.cx) < 0.08:
            entry_x = merge_entry_x[link_key]
            if (
                tgt not in target_bus
                and abs(entry_x - target.cx) <= PARALLEL_CONNECTOR_COORD_EPS / 2
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
            if tgt not in target_bus:
                spread_route = _same_column_spread_top_entry_connector_points(
                    source,
                    target,
                    entry_x,
                    gap=gap,
                )
                if (
                    not _path_hits_obstacles(
                        spread_route,
                        route_obstacles,
                        margin=CONNECTOR_OBSTACLE_MARGIN,
                    )
                    and not _path_penetrates_attached_boxes(spread_route, source, target)
                ):
                    return spread_route
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
            spread_bus_y = _fanout_leg_routing_bus_y(
                link_key,
                graph=graph,
                outgoing=outgoing,
                target_bus=target_bus,
                merge_link_bus=merge_link_bus,
                tee_y=fanout_tee_y,
                anchors=anchors,
            )
        else:
            from visualizer.computation_graph import _inline_frame_tail_indices

            preserve_frame_tail_bus = (
                src in _inline_frame_tail_indices(graph)
                and link_key in merge_link_bus
            )
            if preserve_frame_tail_bus:
                frame = _frame_for_tail_node(graph, src)
                if frame is not None:
                    draw_bounds = _inline_frame_draw_bounds(frame, positions, graph)
                    corridor_y = _frame_tail_routing_corridor_y(
                        draw_bounds,
                        source,
                        target,
                    )
                    spread_bus_y = min(merge_link_bus[link_key], corridor_y)
                else:
                    spread_bus_y = merge_link_bus[link_key]
            else:
                min_target_bus = _connector_min_bus_y_above_target(target, gap=gap)
                corridor_bus = merge_link_bus.get(link_key)
                if (
                    corridor_bus is not None
                    and corridor_bus < min_target_bus - PARALLEL_CONNECTOR_COORD_EPS
                ):
                    spread_bus_y = corridor_bus
                else:
                    spread_bus_y = max(
                        spread_bus_y,
                        min_target_bus,
                        _merge_bus_y_clearing_same_column_feeds(
                            spread_bus_y,
                            tgt=tgt,
                            src=src,
                            incoming=incoming,
                            positions=positions,
                            anchors=anchors,
                        ),
                        _min_bus_y_clearing_horizontal_corridor(
                            source.cx,
                            entry_x,
                            route_obstacles,
                            proposed_y=spread_bus_y,
                        ),
                    )
        tail_frame = _frame_for_tail_node(graph, src)
        if tail_frame is not None and tgt not in tail_frame.node_indices:
            return _frame_tail_merge_entry_connector_points(
                source,
                target,
                exit_x=source.cx,
                entry_x=entry_x,
                bus_y=spread_bus_y,
                frame_bounds=_inline_frame_draw_bounds(tail_frame, positions, graph),
                gap=gap,
                obstacles=route_obstacles,
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
        positions[src].spec.synthetic != SYNTHETIC_TENSOR
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
        src not in source_bus
        and link_key not in merge_entry_x
    ):
        frame_top_route = _outside_to_inline_frame_top_member_route(
            source,
            target,
            route_obstacles,
            graph,
            positions,
            src=src,
            tgt=tgt,
            gap=gap,
        )
        if frame_top_route is not None:
            return frame_top_route
    if (
        src in source_bus
        and link_key in merge_link_bus
        and link_key not in merge_entry_x
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
    from visualizer.computation_graph import _inline_frame_tail_indices

    if src in _inline_frame_tail_indices(graph):
        tail_frame = _frame_for_tail_node(graph, src)
        if tail_frame is not None and tgt not in tail_frame.node_indices:
            return _frame_tail_merge_entry_connector_points(
                source,
                target,
                exit_x=source.cx,
                entry_x=merge_entry_x.get(link_key, target.cx),
                bus_y=merge_link_bus.get(
                    link_key,
                    _connector_min_bus_y_above_target(target, gap=gap),
                ),
                frame_bounds=_inline_frame_draw_bounds(tail_frame, positions, graph),
                gap=gap,
                obstacles=route_obstacles,
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
        main_links = list(link_group)
        if not _should_use_shared_connector_bus(len(main_links)):
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
        if _target_blocks_same_column_bypass(
            positions,
            anchors,
            tgt=tgt,
            incoming=incoming,
            outgoing=outgoing,
        ):
            min_bus_y += SAME_COLUMN_BYPASS_CORRIDOR
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
        main_links = list(link_group)
        top_main_links = list(main_links)
        if len(top_main_links) < SHARED_SOURCE_BUS_MIN_LINKS:
            continue
        target_pos = positions[tgt]
        target_anchor = anchors.get(tgt)
        if target_anchor is None:
            continue
        sorted_links = sorted(top_main_links, key=_spread_link_port_order_key(positions))
        if tgt in target_bus:
            base_bus = target_bus.get(tgt)
            for link in sorted_links:
                merge_entry_x[link] = target_anchor.cx
                if base_bus is not None:
                    merge_link_bus[link] = max(
                        base_bus,
                        _connector_min_bus_y_above_target(target_anchor),
                    )
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
        _assign_same_column_bypass_entry(
            spread_links,
            target_anchor,
            positions=positions,
            anchors=anchors,
            target_bus=target_bus,
            merge_entry_x=merge_entry_x,
            merge_link_bus=merge_link_bus,
        )
        base_bus = target_bus.get(tgt)
        spread_sources = [anchors[src] for src, _ in spread_links if src in anchors]
        involved = {tgt, *(src for src, _ in spread_links)}
        route_obstacles = [
            anchor for node_index, anchor in anchors.items() if node_index not in involved
        ] + label_obstacles + _inline_frame_bounds_obstacles(
            graph,
            positions,
            exclude_nodes=involved,
        )
        if base_bus is None and len(spread_links) >= 2:
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
            _assign_merge_link_bus_for_spread(
                spread_links,
                base_bus,
                tgt=tgt,
                incoming=incoming,
                positions=positions,
                anchors=anchors,
                merge_link_bus=merge_link_bus,
                graph=graph,
            )
            _nest_same_side_merge_bus_levels(
                spread_links,
                tgt=tgt,
                positions=positions,
                anchors=anchors,
                merge_entry_x=merge_entry_x,
                merge_link_bus=merge_link_bus,
                obstacles=route_obstacles,
                incoming=incoming,
                target_bus=target_bus,
            )

    for src, link_group in outgoing.items():
        if src not in source_bus:
            continue
        main_links = _fanout_links_excluding_bypasses(graph, link_group)
        top_main_links = [
            link
            for link in main_links
            if link in merge_entry_x
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
            source_bus[src] = _effective_source_bus_y(
                source_anchor,
                target_anchors,
                source_bus[src],
            )
            source_bus[src] = _lift_bus_y_above_inline_frame_interiors(
                source_bus[src],
                graph=graph,
                positions=positions,
                x_left=min(xs),
                x_right=max(xs),
            )
            _, y_stub = _connector_exit_stub_y(source_anchor.bottom)
            source_bus[src] = min(source_bus[src], y_stub - 0.02)
            continue
        xs = [source_anchor.cx, *(target.cx for target in target_anchors)]
        source_bus[src] = _lift_bus_y_above_inline_frame_interiors(
            source_bus[src],
            graph=graph,
            positions=positions,
            x_left=min(xs),
            x_right=max(xs),
        )

    for src in list(source_bus):
        if _source_fanout_splits_before_target_bus(graph, src, outgoing, target_bus):
            continue
        for _, tgt in outgoing.get(src, []):
            link = (src, tgt)
            if link in merge_link_bus:
                continue
            if tgt in target_bus:
                continue
            target = anchors.get(tgt)
            if target is not None:
                leg_bus = _connector_min_bus_y_above_target(target)
                if leg_bus < source_bus[src] - PARALLEL_CONNECTOR_COORD_EPS:
                    merge_link_bus[link] = leg_bus
                    continue
            merge_link_bus[link] = source_bus[src]

    for link_key in list(merge_link_bus):
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue
        entry_x = merge_entry_x.get(link_key, target.cx)
        x_left = min(source.cx, entry_x, target.cx)
        x_right = max(source.cx, entry_x, target.cx)
        cleared = merge_link_bus[link_key]
        if not _horizontal_bus_y_clears_inline_frames(
            cleared,
            graph,
            positions,
            src=src,
            tgt=tgt,
            x_left=x_left,
            x_right=x_right,
        ):
            cleared = _lift_bus_y_above_inline_frame_interiors(
                cleared,
                graph=graph,
                positions=positions,
                x_left=x_left,
                x_right=x_right,
            )
        cleared = _clamp_bus_y_clearing_inline_frames(
            cleared,
            graph=graph,
            positions=positions,
            x_left=x_left,
            x_right=x_right,
        )
        merge_link_bus[link_key] = max(
            cleared,
            _connector_min_bus_y_above_target(target),
        )

    for (src, tgt), _entry_x in list(merge_entry_x.items()):
        if not graph.nodes[src].key.startswith("sideproducer:"):
            continue
        target = anchors.get(tgt)
        if target is None:
            continue
        merge_entry_x[(src, tgt)] = min(
            target.right - CONNECTOR_ATTACHED_BOX_MARGIN,
            target.cx + TOP_ENTRY_PORT_GAP,
        )

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


def _caption_mask_bbox() -> dict[str, object]:
    """Backing that hides the connector running beneath an inline frame caption.

    A frame caption starts at the frame's left edge and can be wider than the frame
    itself, so it has nowhere to sit clear of the spine entering the frame.
    """
    return {
        "boxstyle": "square,pad=0.15",
        "facecolor": COLORS["detail_fill"],
        "edgecolor": "none",
    }


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
                    bbox=_caption_mask_bbox(),
                    zorder=INLINE_FRAME_CAPTION_ZORDER,
                )
            continue

        from visualizer.render_validate import _caption_top_offset

        label_lines = _inline_frame_label_lines(frame.label, frame_w)
        caption_top = bounds.top + _caption_top_offset(label_lines, frame.sublabel)
        for line_index, line in enumerate(label_lines):
            ax.text(
                frame_left + 0.02,
                caption_top - line_index * INLINE_FRAME_LABEL_LINE_H,
                line,
                ha="left",
                va="bottom",
                fontsize=6.4,
                color=COLORS["muted"],
                bbox=_caption_mask_bbox(),
                zorder=INLINE_FRAME_CAPTION_ZORDER,
            )
        if frame.sublabel:
            sub_lines = [line for line in frame.sublabel.split("\n") if line.strip()]
            for line_index, line in enumerate(sub_lines):
                ax.text(
                    frame_left + 0.02,
                    caption_top - 0.11 - line_index * 0.11,
                    line,
                    ha="left",
                    va="bottom",
                    fontsize=5.6,
                    color=COLORS["muted"],
                    style="italic",
                    bbox=_caption_mask_bbox(),
                    zorder=INLINE_FRAME_CAPTION_ZORDER,
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
                COLORS["embed"],
                text_color=COLORS["text"],
                sublabel=input_sublabel,
                fontsize=7.2,
                pad_x=BLOCK_PAD_X,
                pad_y=BLOCK_PAD_Y,
            )
            plan.node_draws.append((input_leaf, {"edgecolor": _INPUT_NODE_EDGE}))
            continue

        if spec.synthetic == SYNTHETIC_OUTPUT:
            output_leaf = _make_node(
                spec.key,
                pos.cx,
                pos.top_y,
                pos.width,
                pos.height,
                spec.label,
                COLORS["embed"],
                text_color=COLORS["text"],
                fontsize=7.2,
                pad_x=BLOCK_PAD_X,
                pad_y=BLOCK_PAD_Y,
            )
            plan.node_draws.append((output_leaf, {"edgecolor": _INPUT_NODE_EDGE}))
            continue

        if spec.synthetic == SYNTHETIC_TENSOR:
            port_leaf = _make_node(
                spec.key,
                pos.cx,
                pos.top_y,
                pos.width,
                pos.height,
                spec.label,
                COLORS["embed"],
                text_color=COLORS["text"],
                sublabel=spec.sublabel,
                fontsize=7.2,
                pad_x=INPUT_PAD_X,
                pad_y=TENSOR_PORT_PAD_Y,
            )
            plan.node_draws.append((port_leaf, {"edgecolor": _INPUT_NODE_EDGE}))
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
                COLORS["basic_op"],
                text_color=COLORS["text"],
                sublabel=None,
                fontsize=7.4,
            )
            # A method wrapper has no submodule internals, so a dashed border would promise
            # an expansion that does not exist; dashed strokes stay reserved for inline frames.
            plan.node_draws.append(
                (
                    leaf,
                    {"edgecolor": _BASIC_OP_EDGE, "linestyle": "solid"},
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
        if spec.synthetic in {
            SYNTHETIC_INPUT,
            SYNTHETIC_OUTPUT,
            SYNTHETIC_TENSOR,
            SYNTHETIC_HIDDEN,
        }:
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


def _graph_chain_entry_index(positions: list[LayoutPosition]) -> int | None:
    """Index of the node the graph's incoming flow attaches to."""
    for index, pos in enumerate(positions):
        if pos.spec.synthetic == SYNTHETIC_INPUT:
            return index
    for index, pos in enumerate(positions):
        if pos.spec.synthetic in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}:
            continue
        return index
    return None


def _graph_exit_indices(
    graph,
    links: list[tuple[int, int]],
    positions: list[LayoutPosition],
) -> list[int]:
    """Indices of nodes the graph's outgoing flow leaves from, bottom-most last."""
    feeders = {src for src, _tgt in links}
    exits = [
        index
        for index, pos in enumerate(positions)
        if index not in feeders
        and index < len(graph.nodes)
        and pos.spec.synthetic not in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}
    ]
    return sorted(exits, key=lambda index: -positions[index].top_y)


def _shift_graph_chain_onto_axis(
    positions: list[LayoutPosition],
    *,
    axis_x: float,
    entry_index: int,
) -> bool:
    """Slide a laid-out graph sideways so its entry column sits on ``axis_x``."""
    dx = axis_x - positions[entry_index].cx
    if abs(dx) < 1e-6:
        return False
    for pos in positions:
        pos.cx += dx
    return True


def _anchors_from_detail_plan(
    positions: list[LayoutPosition],
    plan: DetailDrawPlan,
    graph=None,
) -> dict[int, _RenderAnchor]:
    """Build connector anchors from finalized layout positions."""
    del plan  # kept for call-site compatibility; anchors follow positions, not draw order
    del graph  # anchors describe tiles; a tile's frame is an obstacle, not a docking surface
    # An anchor is the tile as drawn. Widening it to the enclosing frame would let a port
    # satisfy the anchor while sitting in empty space beside the tile, which draws as a
    # connector that starts or ends attached to nothing.
    return {index: _anchor_from_position(pos) for index, pos in enumerate(positions)}


@dataclass(frozen=True)
class _RenderedGraph:
    """Geometry a caller needs to wire a rendered computation graph into a diagram."""

    bottom: float
    right: float
    entry: _RenderAnchor | None
    exits: tuple[_RenderAnchor, ...] = ()


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
    align_chain_to_cx: bool = False,
    min_left: float | None = None,
    forbidden_regions: list | None = None,
    basic_ops: BasicOpFilter | None = None,
) -> _RenderedGraph:
    """Lay out a computation graph with graph-layout and draw it."""
    from visualizer.render_validate import LAYOUT_MIN_TOP_Y, finalize_detail_layout

    graph = build_computation_graph(
        root,
        prefix_steps=prefix_steps,
        include_input=include_input,
        basic_ops=basic_ops,
    )
    # Standalone body figures get an explicit output. Expanded spine modules already
    # feed the surrounding top-level figure, so a second output tile would be redundant.
    if draw_section_frame:
        add_forward_output(graph)
    if not graph.nodes:
        return _RenderedGraph(bottom=top_y, right=cx + block_w / 2, entry=None)
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
        return _RenderedGraph(bottom=top_y - est_h, right=cx + block_w / 2, entry=None)

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

    from visualizer.computation_graph import (
        _dock_single_consumer_tensor_ports,
        _graph_has_tensor_ports,
    )

    if _graph_has_tensor_ports(graph):
        _dock_single_consumer_tensor_ports(positions, graph)
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)

    entry_index = _graph_chain_entry_index(positions)
    # Overlap resolution keeps content clear of its neighbours but does not preserve the
    # requested center, so a graph drawn on the model spine has to be re-seated on it.
    if align_chain_to_cx and entry_index is not None:
        if _shift_graph_chain_onto_axis(positions, axis_x=cx, entry_index=entry_index):
            plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)

    anchors = _anchors_from_detail_plan(positions, plan, graph)
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

    portal_size = 0.075
    drawn_portals: set[tuple[float, float]] = set()
    for src, tgt in links:
        points = link_paths.get((src, tgt))
        if points is None:
            continue
        for frame in graph.inline_frames:
            members = set(frame.node_indices)
            if (src in members) == (tgt in members):
                continue
            bounds = _inline_frame_draw_bounds(frame, positions, graph)
            for portal_x, portal_y in _inline_frame_connector_portals(points, bounds):
                portal_key = (round(portal_x, 5), round(portal_y, 5))
                if portal_key in drawn_portals:
                    continue
                drawn_portals.add(portal_key)
                ax.add_patch(
                    Rectangle(
                        (portal_x - portal_size / 2, portal_y - portal_size / 2),
                        portal_size,
                        portal_size,
                        linewidth=0,
                        facecolor=COLORS["detail_fill"],
                        zorder=DETAIL_CONNECTOR_ZORDER - 0.1,
                    )
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
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
        incoming=incoming,
        anchors=anchors,
        graph=graph,
        outgoing=outgoing,
    )

    for label, x, y, ha, va in plan.branch_labels:
        _draw_floating_port_label(ax, label, x, y, ha=ha, va=va)

    for label, x, y in merge_link_labels:
        _draw_floating_port_label(ax, label, x, y, ha="center", va="bottom")

    input_anchor = anchors.get(entry_index) if entry_index is not None else None
    exit_anchors = tuple(
        anchors[index]
        for index in _graph_exit_indices(graph, links, positions)
        if index in anchors
    )

    return _RenderedGraph(
        bottom=frame_bottom,
        right=frame_right,
        entry=input_anchor,
        exits=exit_anchors,
    )


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

    rendered = _render_laid_out_computation_graph(
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
    return rendered.bottom, rendered.right


def _detail_section_title(spec: ArchitectureSpec, title: str, tree: BlockNode) -> str:
    """Name a section after the block it expands, as the overview tile labels it."""
    ffn_classes = {
        variant.ffn_class for variant in spec.layer_variants if variant.ffn_class
    }
    spine_moe_class = _spine_moe_class(spec)
    if spine_moe_class:
        ffn_classes.add(spine_moe_class)
    if tree.class_name in ffn_classes:
        return _ffn_class_display_name(tree.class_name)
    if tree.role != "attention":
        return title
    variant_classes = {
        variant.attention_class
        for variant in spec.layer_variants
        if variant.attention_class
    }
    if tree.class_name in variant_classes:
        return tree.class_name
    variant_labels = {
        variant.attention_class or variant.attention_label
        for variant in spec.layer_variants
    }
    # With several attention variants the overview tile lists them all, so there is no
    # single name to share and each section keeps its own class-derived one.
    if len(variant_labels) > 1:
        return title
    return _attention_label_base(spec) or title


def _spine_named_ffn_classes(spec: ArchitectureSpec) -> set[str]:
    """FFN/MoE classes the overview spine names on its tile, one per layer variant."""
    classes = {variant.ffn_class for variant in spec.layer_variants if variant.ffn_class}
    moe_class = _spine_moe_class(spec)
    if moe_class:
        classes.add(moe_class)
    return classes


def _detail_sections_to_render(spec: ArchitectureSpec) -> list[tuple[str, BlockNode, str | None]]:
    """Return titled block trees rendered as internal diagram subsections."""
    sections: list[tuple[str, BlockNode, str | None]] = []
    basic_ops = spec.basic_ops or BasicOpFilter.for_detailed()
    section_trees = prepare_diagram_section_trees(
        architecture_section_trees(spec),
        basic_ops=basic_ops,
    )
    # A spine tile naming an FFN class promises an expansion for it, so those classes
    # earn a section even when their own chain runs straight.
    spine_ffn = _spine_named_ffn_classes(spec)
    for title, tree in section_trees:
        title = _detail_section_title(spec, title, tree)
        # Export trees keep every spine component, including straight-line ones. Those
        # expand inline wherever they are referenced, so a section of their own would
        # repeat the same chain of steps as a submodule.
        if (
            not is_straight_line_module(tree) or tree.class_name in spine_ffn
        ) and subgraph_warrants_export(tree, basic_ops=basic_ops):
            sections.append((title, tree, _format_input_source_sublabel(tree.input_source)))
        for sub_title, sub_tree in collect_nested_diagrams(tree, basic_ops=basic_ops):
            prepared_sub_tree = expand_block_tree_inplace(sub_tree, basic_ops=basic_ops)
            if is_single_function_tree(prepared_sub_tree):
                if _omit_from_detailed_view(prepared_sub_tree):
                    continue
                if not _show_single_function_in_diagram(prepared_sub_tree):
                    continue
            sections.append(
                (
                    sub_title,
                    prepared_sub_tree,
                    _format_input_source_sublabel(prepared_sub_tree.input_source),
                )
            )
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

    if not architecture_section_trees(spec):
        ax.text(
            cx,
            cursor,
            "No modeling source for this checkpoint: internals unavailable, "
            "dashed spine tiles inferred from config",
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
        (
            _box_label_width(
                ax,
                _spine_display_label(comp, spec),
                fontsize=9.0,
                sublabel=_spine_sublabel(comp),
            )
            if _spine_sublabel(comp)
            else _box_label_width(ax, _spine_display_label(comp, spec), fontsize=9.0)
            for comp in stack_tail
        ),
        default=stack_w,
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
        box_style: dict[str, str] | None = None,
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
        _draw_box(ax, node, **(box_style or {}))
        if spine:
            _connect_down(ax, spine[-1], node)
        spine.append(node)
        y = node.bottom - min_vertical_block_gap()
        return node

    place("Tokenized text", COLORS["embed"], node_id="tokens")

    above_block_bottom = spine[-1].bottom if spine else y
    above_block_exits: tuple[_RenderAnchor, ...] = ()
    above_block_right = cx + block_w / 2

    for comp in stack_pre:
        if detailed:
            stack_tree = expand_block_tree_inplace(
                build_stack_component_tree(
                    comp,
                    spec.class_registry,
                    BasicOpFilter.for_detailed(),
                ),
                basic_ops=spec.basic_ops or BasicOpFilter.for_detailed(),
            )
            if is_straight_line_module(stack_tree):
                frame_title = spine_expanded_frame_label(
                    comp,
                    positional_encoding=spec.positional_encoding,
                )
                expanded_top = _spine_expanded_block_top_y(y)
                rendered = _render_laid_out_computation_graph(
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
                    align_chain_to_cx=True,
                    basic_ops=spec.basic_ops or BasicOpFilter.for_detailed(),
                )
                if spine and rendered.entry is not None:
                    _arrow(
                        ax,
                        spine[-1].cx,
                        spine[-1].bottom,
                        rendered.entry.cx,
                        rendered.entry.top,
                    )
                above_block_bottom = rendered.bottom
                above_block_exits = rendered.exits
                above_block_right = rendered.right
                y = rendered.bottom - min_vertical_block_gap()
                continue

        fill, text_color, box_style = _spine_module_style(comp, spec)
        place(
            _spine_display_label(comp, spec),
            fill,
            node_id=comp.attr_name,
            sublabel=_spine_sublabel(comp),
            h=_spine_box_height(comp),
            text_color=text_color,
            box_style=box_style,
        )
        above_block_bottom = spine[-1].bottom
        above_block_exits = ()

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
        flow_source = above_block_exits[-1] if above_block_exits else None
        _connect_spine_block_side_outputs(
            ax,
            above_block_exits,
            cx=cx,
            join_y=above_block_bottom,
            corridor_x=above_block_right + min_vertical_block_gap() / 2,
        )
        _connect_into_block(
            ax,
            None,
            cx=cx,
            frame_top=frame_top,
            frame_left=frame_left,
            repeat_label=repeat_label,
            decoder_label=decoder_label,
            source_x=flow_source.cx if flow_source else cx,
            source_y=flow_source.bottom if flow_source else above_block_bottom,
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
        fill, text_color, box_style = _spine_module_style(comp, spec)
        tail_node = _make_node(
            comp.attr_name,
            cx,
            tail_cursor,
            tail_w,
            _spine_box_height(comp),
            _spine_display_label(comp, spec),
            fill,
            text_color=text_color,
            sublabel=_spine_sublabel(comp),
        )
        layout.add(tail_node)
        _fit_spine_node_to_label(ax, tail_node)
        _draw_box(ax, tail_node, **box_style)
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

    output_top = (
        tail_nodes[-1].bottom - min_vertical_block_gap()
        if tail_nodes
        else frame_bottom - FRAME_PATCH_BOTTOM_OUTSET - min_vertical_block_gap()
    )
    output_node = _make_node(
        "output",
        cx,
        output_top,
        stack_w,
        stack_h,
        "Output",
        COLORS["embed"],
        text_color=COLORS["text"],
    )
    _fit_spine_node_to_label(ax, output_node)
    layout.add(output_node)
    _draw_box(ax, output_node)
    if tail_nodes:
        _connect_down(ax, tail_nodes[-1], output_node)
    else:
        _arrow(
            ax,
            cx,
            frame_bottom - FRAME_PATCH_BOTTOM_OUTSET,
            output_node.cx,
            output_node.top,
        )

    diagram_bottom = output_node.bottom

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
