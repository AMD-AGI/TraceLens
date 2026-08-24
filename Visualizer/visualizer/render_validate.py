"""Measure, validate, and finalize detail diagram layouts before drawing.

Rendering rule for computation-graph diagrams:
1. Build graph structure and initial Sugiyama layout (estimated sizes).
2. Pre-render every tile label with matplotlib to measure true box sizes.
3. Reflow layers and resolve overlaps from measured bounds (repeat until stable).
4. Validate: all tiles visible, no incorrect tile-on-tile overlap.
5. Only then draw connectors, then boxes/text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from visualizer.computation_graph import (
    ComputationGraph,
    LayoutPosition,
    MIN_HORIZONTAL_BLOCK_GAP,
    SYNTHETIC_COMBINE,
    SYNTHETIC_HIDDEN,
    SYNTHETIC_INPUT,
    SYNTHETIC_TENSOR,
    _compact_synthetic_input_spacing,
    _ensure_synthetic_input_clears_consumers,
    _fanout_branch_index,
    _fanout_branch_node_groups,
    _layout_fork_join_branches,
    _node_content_left,
    _node_content_right,
    _resolve_layout_overlaps,
    realign_fanout_branch_columns,
    stack_fanout_branch_columns,
    stack_inline_frame_positions,
    _center_align_vertical_chains,
)
from visualizer.text_measure import ContentBounds, box_bounds_at, box_label_size, floating_port_label_bounds, input_box_label_size, tensor_port_box_label_size
from visualizer.sizing import BLOCK_PAD_Y, box_text_lines

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from visualizer.render import DetailDrawPlan

DEFAULT_MIN_GAP = 0.02
VALIDATE_MIN_GAP = 0.02
LAYOUT_MIN_TOP_Y = 2.5
INLINE_FRAME_SEPARATION_EPS = 1e-3
_TILE_KINDS = frozenset({"box", "combine"})
_OBSTACLE_KINDS = _TILE_KINDS | frozenset({"inline_frame", "frame_label", "frame_sublabel", "floating_label"})


@dataclass
class MeasuredElement:
    """One drawable region used for overlap checks."""

    kind: str
    bounds: ContentBounds
    label: str = ""
    node_index: int | None = None
    frame_node_indices: frozenset[int] = frozenset()
    frame_id: str | None = None


@dataclass
class LayoutValidationReport:
    overlaps: list[str] = field(default_factory=list)
    invisible: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.overlaps and not self.invisible

    def raise_if_invalid(self) -> None:
        if self.ok:
            return
        lines = [*self.invisible, *self.overlaps]
        raise LayoutValidationError("\n".join(lines))


class LayoutValidationError(AssertionError):
    """Raised when measured layout violates render invariants."""


def enforce_text_fit_node_sizes(
    ax: Axes,
    positions: list[LayoutPosition],
    plan: DetailDrawPlan,
) -> None:
    """Grow tiles to fit measured labels; never shrink below text requirements."""
    apply_measured_node_sizes(ax, positions, plan)


def apply_measured_node_sizes(
    ax: Axes,
    positions: list[LayoutPosition],
    plan: DetailDrawPlan,
) -> None:
    """Resize layout tiles from matplotlib-measured label geometry."""
    from visualizer.render import COLORS

    draw_index = 0
    for pos in positions:
        spec = pos.spec
        if _is_combine(spec.synthetic):
            continue
        if spec.synthetic == "@input":
            width, height = input_box_label_size(
                ax,
                spec.label,
                plan.input_sublabel,
                fontsize=7.2,
            )
            pos.width = width
            pos.height = height
            if draw_index < len(plan.node_draws):
                leaf, _ = plan.node_draws[draw_index]
                leaf.w = pos.width
                leaf.h = pos.height
                leaf.sublabel = plan.input_sublabel
                leaf.x = pos.cx - pos.width / 2
                leaf.y = pos.top_y - pos.height
                draw_index += 1
            continue
        if spec.synthetic == "@hidden_states":
            width, height = input_box_label_size(ax, spec.label, None, fontsize=6.5)
            pos.width = width
            pos.height = height
            if draw_index < len(plan.node_draws):
                leaf, _ = plan.node_draws[draw_index]
                leaf.w = pos.width
                leaf.h = pos.height
                leaf.x = pos.cx - pos.width / 2
                leaf.y = pos.top_y - pos.height
                draw_index += 1
            continue
        if spec.synthetic == "@tensor":
            width, height = tensor_port_box_label_size(
                ax,
                spec.label,
                spec.sublabel,
                fontsize=7.0,
            )
            pos.width = width
            pos.height = height
            if draw_index < len(plan.node_draws):
                leaf, _ = plan.node_draws[draw_index]
                leaf.w = pos.width
                leaf.h = pos.height
                leaf.sublabel = spec.sublabel
                leaf.x = pos.cx - pos.width / 2
                leaf.y = pos.top_y - pos.height
                draw_index += 1
            continue
        if draw_index >= len(plan.node_draws):
            continue
        leaf, _draw_kwargs = plan.node_draws[draw_index]
        width, height = box_label_size(
            ax,
            leaf.label,
            leaf.sublabel,
            fontsize=leaf.fontsize,
            white_text_stroke_pad=leaf.facecolor != COLORS["basic_op"],
        )
        pos.width = max(pos.width, width)
        pos.height = max(pos.height, height)
        leaf.w = pos.width
        leaf.h = pos.height
        leaf.x = pos.cx - pos.width / 2
        leaf.y = pos.top_y - pos.height
        draw_index += 1


def collect_measured_elements(
    ax: Axes,
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    plan: DetailDrawPlan,
    *,
    detail_fill: str,
) -> list[MeasuredElement]:
    """Collect bounding boxes for every box, label, and inline frame caption."""
    from visualizer.render import (
        INLINE_FRAME_LABEL_GAP,
        INLINE_FRAME_LABEL_LINE_H,
        INLINE_FRAME_PAD,
        _inline_frame_label_lines,
    )
    from visualizer.text_measure import TILE_ROUNDING_INSET, measure_text_bounds

    tile_visual_inset = TILE_ROUNDING_INSET

    elements: list[MeasuredElement] = []
    frame_members = _frame_member_indices(graph)
    draw_index = 0
    for index, pos in enumerate(positions):
        spec = pos.spec
        if _is_combine(spec.synthetic):
            center_y = pos.top_y - pos.height / 2
            half_w = pos.width / 2
            half_h = pos.height / 2
            elements.append(
                MeasuredElement(
                    kind="combine",
                    bounds=ContentBounds(
                        left=pos.cx - half_w,
                        right=pos.cx + half_w,
                        bottom=center_y - half_h,
                        top=center_y + half_h,
                    ),
                    label=spec.label,
                    node_index=index,
                )
            )
            continue

        if draw_index >= len(plan.node_draws):
            continue
        leaf, _ = plan.node_draws[draw_index]
        tile_bounds = box_bounds_at(pos.cx, pos.top_y, pos.width, pos.height)
        elements.append(
            MeasuredElement(
                kind="box",
                bounds=tile_bounds,
                label=leaf.label,
                node_index=index,
                frame_id=frame_members.get(index),
            )
        )
        pad_y = leaf.pad_y if leaf.pad_y is not None else BLOCK_PAD_Y
        label_bounds = ContentBounds(
            left=tile_bounds.left + tile_visual_inset,
            right=tile_bounds.right - tile_visual_inset,
            bottom=tile_bounds.bottom + tile_visual_inset,
            top=tile_bounds.top - tile_visual_inset,
        )
        for line in box_text_lines(
            pos.top_y,
            pos.height,
            leaf.label,
            leaf.sublabel,
            pad_y=pad_y,
            title_fontsize=leaf.fontsize,
        ):
            line_bounds = measure_text_bounds(
                ax,
                line.text,
                pos.cx,
                line.y,
                fontsize=line.fontsize,
                ha="center",
                va=line.va,
                fontweight=line.fontweight,
            )
            if not label_bounds.contains(line_bounds, min_gap=0.0):
                elements.append(
                    MeasuredElement(
                        kind="text_overflow",
                        bounds=line_bounds,
                        label=f"{'title' if line.fontweight == 'bold' else 'sublabel'} {line.text!r}",
                        node_index=index,
                    )
                )
        draw_index += 1

    for label, x, y, ha, va in plan.branch_labels:
        elements.append(
            MeasuredElement(
                kind="floating_label",
                bounds=floating_port_label_bounds(ax, label, x, y, ha=ha, va=va, detail_fill=detail_fill),
                label=label,
            )
        )

    for frame in graph.inline_frames:
        if not frame.node_indices:
            continue
        frame_positions = [positions[i] for i in frame.node_indices if i < len(positions)]
        if not frame_positions:
            continue
        pad = INLINE_FRAME_PAD
        tile_bounds = _inline_frame_tile_bounds(frame, positions, pad=pad, graph=graph)
        frame_width = tile_bounds.width
        placement = plan.inline_frame_labels.get(frame.frame_id)
        if placement is not None and placement.lines:
            label_bounds = None
            for line in placement.lines:
                line_bounds = measure_text_bounds(
                    ax,
                    line.text,
                    line.x,
                    line.y,
                    fontsize=line.fontsize,
                    ha=line.ha,
                    va=line.va,
                    fontweight=line.fontweight,
                )
                elements.append(
                    MeasuredElement(
                        kind="frame_label" if line.style != "italic" else "frame_sublabel",
                        bounds=line_bounds,
                        label=line.text,
                        frame_id=frame.frame_id,
                    )
                )
                label_bounds = line_bounds if label_bounds is None else label_bounds.union(line_bounds)
            frame_bounds = tile_bounds
        else:
            label_lines = _inline_frame_label_lines(frame.label, frame_width)
            caption_top = tile_bounds.top + INLINE_FRAME_LABEL_GAP
            for line_index, line in enumerate(label_lines):
                caption = measure_text_bounds(
                    ax,
                    line,
                    tile_bounds.left + 0.02,
                    caption_top - line_index * INLINE_FRAME_LABEL_LINE_H,
                    fontsize=6.4,
                    ha="left",
                    va="bottom",
                    fontweight="normal",
                )
                elements.append(
                    MeasuredElement(kind="frame_label", bounds=caption, label=line, frame_id=frame.frame_id)
                )
            if frame.sublabel:
                sub_lines = [line for line in frame.sublabel.split("\n") if line.strip()]
                for line_index, line in enumerate(sub_lines):
                    sub = measure_text_bounds(
                        ax,
                        line,
                        tile_bounds.left + 0.02,
                        caption_top - 0.11 - line_index * 0.11,
                        fontsize=5.6,
                        ha="left",
                        va="bottom",
                        fontweight="normal",
                    )
                    elements.append(
                        MeasuredElement(kind="frame_sublabel", bounds=sub, label=line, frame_id=frame.frame_id)
                    )
            frame_bounds = tile_bounds
        elements.append(
            MeasuredElement(
                kind="inline_frame",
                bounds=frame_bounds,
                label=frame.label,
                frame_node_indices=frozenset(frame.node_indices),
                frame_id=frame.frame_id,
            )
        )

    return elements


def _is_combine(synthetic: str | None) -> bool:
    from visualizer.computation_graph import SYNTHETIC_COMBINE

    return synthetic == SYNTHETIC_COMBINE


def _inline_frame_member_sets(elements: list[MeasuredElement]) -> dict[str, set[int]]:
    return {
        element.frame_id: set(element.frame_node_indices)
        for element in elements
        if element.kind == "inline_frame" and element.frame_id
    }


def _nested_inline_frame_pair(
    left: MeasuredElement,
    right: MeasuredElement,
    *,
    member_sets: dict[str, set[int]],
) -> bool:
    """True when two inline frames are nested (inner nodes subset of outer)."""
    if left.kind != "inline_frame" or right.kind != "inline_frame":
        return False
    left_members = member_sets.get(left.frame_id or "", set(left.frame_node_indices))
    right_members = member_sets.get(right.frame_id or "", set(right.frame_node_indices))
    if not left_members or not right_members:
        return False
    return left_members.issubset(right_members) or right_members.issubset(left_members)


def _frame_caption_belongs_to_nested_frame(
    caption: MeasuredElement,
    frame: MeasuredElement,
    *,
    member_sets: dict[str, set[int]],
) -> bool:
    """True when a frame caption sits inside a containing inline frame."""
    if caption.kind not in {"frame_label", "frame_sublabel"} or frame.kind != "inline_frame":
        return False
    if not caption.frame_id or not frame.frame_id or caption.frame_id == frame.frame_id:
        return False
    inner = member_sets.get(caption.frame_id, set())
    outer = member_sets.get(frame.frame_id, set(frame.frame_node_indices))
    return bool(inner) and inner.issubset(outer)


def validate_render_layout(
    elements: list[MeasuredElement],
    *,
    min_gap: float = DEFAULT_MIN_GAP,
    forbidden_regions: list[ContentBounds] | None = None,
) -> LayoutValidationReport:
    """Check tiles, captions, and text for visibility, overlap, and forbidden regions."""
    report = LayoutValidationReport()
    obstacles = [element for element in elements if element.kind in _OBSTACLE_KINDS]

    for element in elements:
        bounds = element.bounds
        if bounds.width <= 0 or bounds.height <= 0:
            report.invisible.append(f"{element.kind} {element.label!r} has zero-size bounds")
        if element.kind == "text_overflow":
            report.overlaps.append(f"text overflows tile: {element.label}")

    overlap_obstacles = [element for element in obstacles if element.kind != "text_overflow"]
    frame_member_sets = _inline_frame_member_sets(elements)

    for left_index, left in enumerate(overlap_obstacles):
        for right in overlap_obstacles[left_index + 1 :]:
            if left.kind in _TILE_KINDS and right.kind in _TILE_KINDS:
                if left.node_index is not None and left.node_index == right.node_index:
                    continue
            elif left.frame_id and left.frame_id == right.frame_id:
                continue
            elif _nested_inline_frame_pair(left, right, member_sets=frame_member_sets):
                continue
            elif _frame_caption_belongs_to_nested_frame(left, right, member_sets=frame_member_sets):
                continue
            elif _frame_caption_belongs_to_nested_frame(right, left, member_sets=frame_member_sets):
                continue
            elif right.kind == "inline_frame" and left.node_index is not None:
                if left.node_index in right.frame_node_indices:
                    continue
            elif left.kind == "inline_frame" and right.node_index is not None:
                if right.node_index in left.frame_node_indices:
                    continue
            if not left.bounds.overlaps(right.bounds, min_gap=min_gap):
                continue
            report.overlaps.append(
                f"{left.kind} {left.label!r} overlaps {right.kind} {right.label!r} "
                f"(gap<{min_gap:.3f})"
            )

    if forbidden_regions:
        for region in forbidden_regions:
            for element in elements:
                if element.kind not in {"box", "combine", "inline_frame"}:
                    continue
                if element.bounds.overlaps(region, min_gap=min_gap):
                    report.overlaps.append(
                        f"{element.kind} {element.label!r} overlaps forbidden region "
                        f"[{region.left:.2f},{region.right:.2f}]"
                    )

    return report


def assert_valid_render_layout(
    elements: list[MeasuredElement],
    *,
    min_gap: float = DEFAULT_MIN_GAP,
) -> None:
    validate_render_layout(elements, min_gap=min_gap).raise_if_invalid()


def _resolve_same_row_tile_overlaps(
    positions: list[LayoutPosition],
    *,
    min_gap: float,
    graph: ComputationGraph | None = None,
) -> None:
    """Separate tiles on the same row after measured resize widened box widths."""
    if not positions:
        return
    if graph is not None:
        from visualizer.computation_graph import _graph_has_tensor_ports

        if _graph_has_tensor_ports(graph):
            return
    branch_groups = _fanout_branch_node_groups(positions)
    index_to_branch = {
        index: branch
        for branch, indices in branch_groups.items()
        for index in indices
    }

    def row_units(indices: list[int]) -> list[list[int]]:
        ordered = sorted(indices, key=lambda index: positions[index].cx)
        units: list[list[int]] = []
        seen: set[int] = set()
        for index in ordered:
            if index in seen:
                continue
            branch = index_to_branch.get(index)
            unit = list(branch_groups[branch]) if branch is not None else [index]
            for member in unit:
                seen.add(member)
            units.append(unit)
        return units

    for _pass in range(max(1, len(positions))):
        changed = False
        rows: dict[float, list[int]] = {}
        for index, pos in enumerate(positions):
            rows.setdefault(round(pos.top_y, 4), []).append(index)
        for indices in rows.values():
            if len(indices) < 2:
                continue
            units = row_units(indices)
            for offset in range(1, len(units)):
                left_unit = units[offset - 1]
                right_unit = units[offset]
                left_right = max(_node_content_right(positions[index]) for index in left_unit)
                right_left = min(_node_content_left(positions[index]) for index in right_unit)
                overlap = left_right + min_gap - right_left
                if overlap <= 0:
                    continue
                for unit in units[offset:]:
                    for shift_index in unit:
                        positions[shift_index].cx += overlap
                changed = True
        if not changed:
            break
    if branch_groups:
        realign_fanout_branch_columns(positions, graph)


def resolve_measured_overlaps(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    top_y: float | None = None,
    min_gap: float = DEFAULT_MIN_GAP,
) -> None:
    """Push layout tiles apart after measured resize, optionally restacking layers."""
    from visualizer.computation_graph import (
        _align_tensor_port_columns,
        _assign_layered_vertical_positions,
        _graph_has_tensor_ports,
        _topological_layers,
    )

    if top_y is not None and not _graph_has_tensor_ports(graph):
        layers = _topological_layers(graph)
        _assign_layered_vertical_positions(positions, layers, top_y=top_y)
    _resolve_same_row_tile_overlaps(
        positions,
        min_gap=min_gap,
        graph=graph if not _graph_has_tensor_ports(graph) else None,
    )
    _resolve_layout_overlaps(
        positions,
        graph,
        min_horizontal_gap=min_gap,
        min_vertical_gap=min_gap * 2,
    )
    if _graph_has_tensor_ports(graph):
        _align_tensor_port_columns(positions, graph)
    else:
        _resolve_same_row_tile_overlaps(positions, min_gap=min_gap, graph=graph)


def _frame_member_indices(graph: ComputationGraph) -> dict[int, str]:
    """Map node index -> inline frame id."""
    members: dict[int, str] = {}
    for frame in graph.inline_frames:
        for index in frame.node_indices:
            members[index] = frame.frame_id
    return members


def _inline_frame_tile_bounds(
    frame,
    positions: list[LayoutPosition],
    *,
    pad: float,
    graph: ComputationGraph | None = None,
) -> ContentBounds:
    """Bounds of the dotted frame around member tiles (excluding caption)."""
    if graph is not None:
        from visualizer.render import _inline_frame_draw_bounds

        return _inline_frame_draw_bounds(frame, positions, graph, pad=pad)

    frame_positions = [positions[index] for index in frame.node_indices if index < len(positions)]
    if not frame_positions:
        return ContentBounds(left=0.0, right=0.0, bottom=0.0, top=0.0)
    min_left = min(p.cx - p.width / 2 for p in frame_positions)
    max_right = max(p.cx + p.width / 2 for p in frame_positions)
    min_bottom = min(p.top_y - p.height for p in frame_positions)
    max_top = max(p.top_y for p in frame_positions)
    return ContentBounds(
        left=min_left - pad,
        right=max_right + pad,
        bottom=min_bottom - pad,
        top=max_top + pad,
    )


def _stack_inline_frame_label_lines(
    ax: Axes,
    *,
    label_lines: list[str],
    sublabel_lines: list[str],
    anchor_x: float,
    anchor_y: float,
    ha: str,
    anchor_va: str = "bottom",
) -> tuple[list, ContentBounds]:
    """Measure stacked caption lines anchored at the top-left-style origin."""
    from visualizer.render import (
        INLINE_FRAME_LABEL_LINE_H,
        InlineFrameLabelLine,
    )
    from visualizer.text_measure import measure_text_bounds

    rendered: list[InlineFrameLabelLine] = []
    bounds: ContentBounds | None = None
    cursor_y = anchor_y
    for line in label_lines:
        line_bounds = measure_text_bounds(
            ax,
            line,
            anchor_x,
            cursor_y,
            fontsize=6.4,
            ha=ha,
            va="bottom",
            fontweight="normal",
        )
        rendered.append(
            InlineFrameLabelLine(
                text=line,
                x=anchor_x,
                y=cursor_y,
                ha=ha,
                va="bottom",
            )
        )
        bounds = line_bounds if bounds is None else bounds.union(line_bounds)
        cursor_y -= INLINE_FRAME_LABEL_LINE_H

    if sublabel_lines:
        cursor_y -= 0.03
        for line in sublabel_lines:
            line_bounds = measure_text_bounds(
                ax,
                line,
                anchor_x,
                cursor_y,
                fontsize=5.6,
                ha=ha,
                va="bottom",
                fontweight="normal",
            )
            rendered.append(
                InlineFrameLabelLine(
                    text=line,
                    x=anchor_x,
                    y=cursor_y,
                    ha=ha,
                    va="bottom",
                    fontsize=5.6,
                    style="italic",
                )
            )
            bounds = line_bounds if bounds is None else bounds.union(line_bounds)
            cursor_y -= 0.11

    assert bounds is not None
    if anchor_va == "center":
        block_center = (bounds.top + bounds.bottom) / 2
        dy = anchor_y - block_center
        if abs(dy) > 1e-9:
            shifted: list = []
            for line in rendered:
                shifted.append(
                    InlineFrameLabelLine(
                        text=line.text,
                        x=line.x,
                        y=line.y + dy,
                        ha=line.ha,
                        va=line.va,
                        fontsize=line.fontsize,
                        fontweight=line.fontweight,
                        style=line.style,
                    )
                )
            rendered = shifted
            bounds = ContentBounds(
                left=bounds.left,
                right=bounds.right,
                bottom=bounds.bottom + dy,
                top=bounds.top + dy,
            )
    return rendered, bounds


def _inline_frame_has_horizontal_neighbor(
    frame,
    tile_bounds: ContentBounds,
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    *,
    pad: float,
    side: str,
    min_gap: float,
) -> bool:
    """Return True when another inline frame sits immediately to the left or right."""
    for other in graph.inline_frames:
        if other.frame_id == frame.frame_id:
            continue
        other_bounds = _inline_frame_tile_bounds(other, positions, pad=pad, graph=graph)
        vertical = (
            tile_bounds.bottom - min_gap < other_bounds.top
            and other_bounds.bottom - min_gap < tile_bounds.top
        )
        if not vertical:
            continue
        if side == "left" and other_bounds.right <= tile_bounds.left + min_gap * 2:
            return True
        if side == "right" and other_bounds.left >= tile_bounds.right - min_gap * 2:
            return True
    return False


def _label_block_overlaps(
    bounds: ContentBounds,
    obstacles: list[ContentBounds],
    *,
    min_gap: float,
) -> bool:
    return any(bounds.overlaps(obstacle, min_gap=min_gap) for obstacle in obstacles)


def _caption_side_for_bounds(
    caption_bounds: ContentBounds,
    frame_bounds: ContentBounds,
    *,
    gap: float,
) -> str:
    """Infer whether a caption sits above or beside its inline frame."""
    if caption_bounds.right <= frame_bounds.left + gap * 0.5:
        return "left"
    if caption_bounds.left >= frame_bounds.right - gap * 0.5:
        return "right"
    return "top"


def _reserve_frame_caption_space(
    positions: list[LayoutPosition],
    plan: "DetailDrawPlan",
    elements: list[MeasuredElement],
    graph: ComputationGraph,
    *,
    min_gap: float,
    min_left: float | None,
) -> None:
    """Shift inline-frame columns right when left-placed captions clip the left margin."""
    if not plan.inline_frame_labels:
        return

    caption_bounds: dict[str, ContentBounds] = {}
    for element in elements:
        if element.kind not in {"frame_label", "frame_sublabel"} or not element.frame_id:
            continue
        caption_bounds[element.frame_id] = (
            element.bounds
            if element.frame_id not in caption_bounds
            else caption_bounds[element.frame_id].union(element.bounds)
        )

    member_sets = {
        frame.frame_id: set(frame.node_indices) for frame in graph.inline_frames
    }
    for frame_id, placement in plan.inline_frame_labels.items():
        if placement.side != "left":
            continue
        bounds = caption_bounds.get(frame_id)
        if bounds is None or min_left is None:
            continue
        if bounds.left >= min_left:
            continue
        shift = min_left - bounds.left + min_gap
        indices = member_sets.get(frame_id, set())
        if indices:
            _shift_node_indices(positions, indices, shift)


def _layout_inline_frame_labels(
    ax: Axes,
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    plan: DetailDrawPlan,
    elements: list[MeasuredElement],
    *,
    links: list[tuple[int, int]] | None = None,
    min_gap: float,
    min_left: float | None = None,
) -> dict[str, object]:
    """Place inline-frame captions to avoid tiles, other captions, and connector corridors."""
    from visualizer.render import (
        INLINE_FRAME_LABEL_GAP,
        INLINE_FRAME_PAD,
        InlineFrameLabelPlacement,
        _inline_frame_label_lines,
    )

    if not graph.inline_frames:
        return {}

    from visualizer.computation_graph import SYNTHETIC_TENSOR
    from visualizer.text_measure import box_bounds_at

    port_obstacles = [
        box_bounds_at(pos.cx, pos.top_y, pos.width, pos.height)
        for pos in positions
        if pos.spec.synthetic == SYNTHETIC_TENSOR
    ]

    placements: dict[str, InlineFrameLabelPlacement] = {}
    placed_label_bounds: list[tuple[str, ContentBounds]] = []

    ordered_frames = sorted(
        graph.inline_frames,
        key=lambda frame: max(
            (positions[index].top_y for index in frame.node_indices if index < len(positions)),
            default=0.0,
        ),
        reverse=True,
    )

    pad = INLINE_FRAME_PAD
    frame_member_sets = {
        frame.frame_id: set(frame.node_indices) for frame in graph.inline_frames
    }

    def _enclosing_frame_ids(frame_id: str, frame_members: set[int]) -> set[str]:
        if not frame_members:
            return set()
        return {
            other.frame_id
            for other in graph.inline_frames
            if other.frame_id != frame_id
            and frame_members.issubset(frame_member_sets.get(other.frame_id, set()))
        }

    def _external_obstacles(frame_id: str, frame_members: set[int]) -> list[ContentBounds]:
        enclosing = _enclosing_frame_ids(frame_id, frame_members)
        tile_obstacles = [
            element.bounds
            for element in elements
            if element.kind in {"box", "combine", "inline_frame", "floating_label"}
            and not (element.node_index is not None and element.node_index in frame_members)
            and not (element.kind == "inline_frame" and element.frame_id in {frame_id, *enclosing})
            and not (
                element.kind in {"frame_label", "frame_sublabel"}
                and element.frame_id in enclosing
            )
        ]
        return tile_obstacles + port_obstacles + [
            bounds
            for placed_frame_id, bounds in placed_label_bounds
            if placed_frame_id not in enclosing
        ]

    def _side_near_frame(side: str, bounds: ContentBounds, tile_bounds: ContentBounds) -> bool:
        label_center_y = (bounds.top + bounds.bottom) / 2
        frame_center_y = (tile_bounds.top + tile_bounds.bottom) / 2
        if abs(label_center_y - frame_center_y) > max(tile_bounds.height * 0.65, 0.35):
            return False
        if side == "left":
            return bounds.right <= tile_bounds.left + min_gap * 0.25
        return bounds.left >= tile_bounds.right - min_gap * 0.25

    for frame in ordered_frames:
        if not frame.node_indices:
            continue
        tile_bounds = _inline_frame_tile_bounds(frame, positions, pad=pad, graph=graph)
        if tile_bounds.width <= 0:
            continue

        frame_members = frame_member_sets.get(frame.frame_id, set())
        frame_width = tile_bounds.width
        label_lines = _inline_frame_label_lines(frame.label, frame_width)
        sublabel_lines = [line for line in (frame.sublabel or "").split("\n") if line.strip()]
        obstacles = _external_obstacles(frame.frame_id, frame_members)
        caption_obstacles = [
            element.bounds
            for element in elements
            if element.kind == "box"
            and (element.node_index is None or element.node_index not in frame_members)
        ]

        has_left_neighbor = _inline_frame_has_horizontal_neighbor(
            frame,
            tile_bounds,
            graph,
            positions,
            pad=pad,
            side="left",
            min_gap=min_gap,
        )
        has_right_neighbor = _inline_frame_has_horizontal_neighbor(
            frame,
            tile_bounds,
            graph,
            positions,
            pad=pad,
            side="right",
            min_gap=min_gap,
        )

        def _candidate_score(side: str, dx: float, dy: float, overlaps: bool) -> float:
            side_penalty = {"top": 0.0, "left": 25.0, "right": 30.0}[side]
            if side == "left" and has_left_neighbor:
                side_penalty += 40.0
            if side == "right" and has_right_neighbor:
                side_penalty += 40.0
            return side_penalty + abs(dx) + abs(dy) + (1000.0 if overlaps else 0.0)

        best: tuple[float, list, ContentBounds, str] | None = None

        top_anchor_x = tile_bounds.left + 0.02
        top_anchor_y = tile_bounds.top + max(INLINE_FRAME_LABEL_GAP, min_gap)
        for dy in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
            lines, bounds = _stack_inline_frame_label_lines(
                ax,
                label_lines=label_lines,
                sublabel_lines=sublabel_lines,
                anchor_x=top_anchor_x,
                anchor_y=top_anchor_y + dy,
                ha="left",
            )
            overlaps = _label_block_overlaps(bounds, obstacles, min_gap=min_gap)
            near_frame = bounds.bottom >= tile_bounds.top - 0.04 and bounds.bottom <= tile_bounds.top + 0.45
            if near_frame and not overlaps:
                score = _candidate_score("top", 0.0, dy, False)
                if best is None or score < best[0]:
                    best = (score, lines, bounds, "top")

        side_center_y = (tile_bounds.top + tile_bounds.bottom) / 2
        if has_left_neighbor and not has_right_neighbor:
            side_order = [("right", tile_bounds.right + min_gap, "left"), ("left", tile_bounds.left - min_gap, "right")]
        else:
            side_order = [("left", tile_bounds.left - min_gap, "right"), ("right", tile_bounds.right + min_gap, "left")]
        for side, anchor_x, ha in side_order:
            for dy in (0.0, 0.08, -0.08, 0.16, -0.16):
                lines, bounds = _stack_inline_frame_label_lines(
                    ax,
                    label_lines=label_lines,
                    sublabel_lines=sublabel_lines,
                    anchor_x=anchor_x,
                    anchor_y=side_center_y + dy,
                    ha=ha,
                    anchor_va="center",
                )
                if side == "left" and min_left is not None and bounds.left < min_left:
                    continue
                overlaps = _label_block_overlaps(bounds, obstacles, min_gap=min_gap)
                if not _side_near_frame(side, bounds, tile_bounds):
                    continue
                if overlaps:
                    continue
                score = _candidate_score(side, 0.0, dy, False)
                if best is None or score < best[0]:
                    best = (score, lines, bounds, side)

        if best is None:
            fallback_specs = [
                ("right", tile_bounds.right + min_gap, "left"),
                ("left", tile_bounds.left - min_gap, "right"),
            ]
            lines = []
            bounds = ContentBounds(left=0.0, right=0.0, bottom=0.0, top=0.0)
            side = "right"
            for side_name, anchor_x, ha in fallback_specs:
                lines, bounds = _stack_inline_frame_label_lines(
                    ax,
                    label_lines=label_lines,
                    sublabel_lines=sublabel_lines,
                    anchor_x=anchor_x,
                    anchor_y=side_center_y,
                    ha=ha,
                    anchor_va="center",
                )
                side = side_name
                if side_name == "left" and min_left is not None and bounds.left < min_left:
                    continue
                break
        else:
            _, lines, bounds, side = best

        if side == "top":
            anchor_x = lines[0].x if lines else top_anchor_x
            anchor_ha = lines[0].ha if lines else "left"
            anchor_y = lines[0].y if lines else top_anchor_y
            for _ in range(8):
                if not _label_block_overlaps(bounds, caption_obstacles, min_gap=min_gap):
                    break
                anchor_y += 0.04
                lines, bounds = _stack_inline_frame_label_lines(
                    ax,
                    label_lines=label_lines,
                    sublabel_lines=sublabel_lines,
                    anchor_x=anchor_x,
                    anchor_y=anchor_y,
                    ha=anchor_ha,
                )
                if bounds.bottom > tile_bounds.top + 0.5:
                    side_name = "right" if has_left_neighbor else "left"
                    anchor_x_fb = (
                        tile_bounds.right + min_gap
                        if side_name == "right"
                        else tile_bounds.left - min_gap
                    )
                    ha_fb = "left" if side_name == "right" else "right"
                    lines, bounds = _stack_inline_frame_label_lines(
                        ax,
                        label_lines=label_lines,
                        sublabel_lines=sublabel_lines,
                        anchor_x=anchor_x_fb,
                        anchor_y=side_center_y,
                        ha=ha_fb,
                        anchor_va="center",
                    )
                    side = side_name
                    break

        placements[frame.frame_id] = InlineFrameLabelPlacement(
            frame_id=frame.frame_id,
            lines=lines,
            side=side,
        )
        placed_label_bounds.append((frame.frame_id, bounds))

    return placements


def _apply_inline_frame_label_layout(
    ax: Axes,
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    plan: DetailDrawPlan,
    *,
    detail_fill: str,
    min_gap: float,
    min_left: float | None = None,
) -> list[MeasuredElement]:
    """Resolve inline-frame caption positions and refresh measured elements."""
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=detail_fill)
    plan.inline_frame_labels = _layout_inline_frame_labels(
        ax,
        graph,
        positions,
        plan,
        elements,
        links=list(graph.links),
        min_gap=min_gap,
        min_left=min_left,
    )
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=detail_fill)
    _reserve_frame_caption_space(
        positions,
        plan,
        elements,
        graph,
        min_gap=min_gap,
        min_left=min_left,
    )
    plan.inline_frame_labels = _layout_inline_frame_labels(
        ax,
        graph,
        positions,
        plan,
        collect_measured_elements(ax, graph, positions, plan, detail_fill=detail_fill),
        links=list(graph.links),
        min_gap=min_gap,
        min_left=min_left,
    )
    return collect_measured_elements(ax, graph, positions, plan, detail_fill=detail_fill)


def _classify_layout_zones(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> tuple[set[int], set[int], set[int], set[int]]:
    """Partition nodes into fan-out, main-path, side-branch, and input columns."""
    fanout = {
        index
        for index, pos in enumerate(positions)
        if _fanout_branch_index(pos.spec) is not None
    }
    side = {
        index
        for index, pos in enumerate(positions)
        if pos.spec.key.startswith("sideproducer")
    }
    side |= {
        index
        for index, pos in enumerate(positions)
        if "sidefeed" in pos.spec.key and ":gate_act" in pos.spec.key
    }
    side |= {
        index
        for index, pos in enumerate(positions)
        if pos.spec.block is not None
        and pos.spec.block.attr_name in {"up_proj", "w3"}
        and any((index, target) in graph.side_entry_links for target in range(len(graph.nodes)))
    }
    from visualizer.computation_graph import _find_fork_join_clusters

    for cluster in _find_fork_join_clusters(graph):
        side |= {
            cluster.main_source,
            cluster.main_branch,
            cluster.join,
            cluster.tail,
        }
    input_hidden = {
        index
        for index, pos in enumerate(positions)
        if pos.spec.synthetic in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}
    }
    main = set(range(len(positions))) - fanout - side - input_hidden
    return fanout, main, side, input_hidden


def _cluster_horizontal_bounds(
    positions: list[LayoutPosition],
    indices: set[int],
) -> tuple[float, float] | None:
    if not indices:
        return None
    left = min(positions[index].cx - positions[index].width / 2 for index in indices)
    right = max(positions[index].cx + positions[index].width / 2 for index in indices)
    return left, right


def _shift_node_indices(
    positions: list[LayoutPosition],
    indices: set[int],
    delta_x: float,
) -> None:
    if abs(delta_x) <= 1e-9:
        return
    for index in indices:
        positions[index].cx += delta_x


def _ensure_input_above_inline_frames(
    positions: list[LayoutPosition],
    elements: list[MeasuredElement],
    *,
    min_gap: float,
) -> None:
    """Raise input/tensor port tiles so they clear inline-frame caption bands in their column."""
    caption_tops = [
        element.bounds.top
        for element in elements
        if element.kind in {"inline_frame", "frame_label", "frame_sublabel"}
    ]
    if not caption_tops:
        return
    for pos in positions:
        if pos.spec.synthetic not in {SYNTHETIC_INPUT, SYNTHETIC_TENSOR}:
            continue
        port_left = pos.cx - pos.width / 2
        port_right = pos.cx + pos.width / 2
        local_caption_tops = [
            element.bounds.top
            for element in elements
            if element.kind in {"inline_frame", "frame_label", "frame_sublabel"}
            and element.bounds.right + min_gap > port_left
            and element.bounds.left - min_gap < port_right
        ]
        if not local_caption_tops:
            continue
        min_bottom = max(local_caption_tops) + min_gap
        if pos.bottom < min_bottom:
            pos.top_y += min_bottom - pos.bottom


def _align_and_stack_inline_frames(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float | None = None,
) -> None:
    """Keep straight-line inline frames vertically stacked on one column."""
    from visualizer.computation_graph import (
        _align_merge_nodes,
        _graph_has_tensor_ports,
        clear_merge_feeder_columns,
        finalize_tensor_port_pipeline_layout,
        repack_inline_frame_columns,
    )

    stack_inline_frame_positions(positions, graph, min_gap=min_gap)
    if not _graph_has_tensor_ports(graph):
        stack_fanout_branch_columns(positions, graph, min_gap=min_gap)
    _align_merge_nodes(positions, graph)
    if _graph_has_tensor_ports(graph):
        finalize_tensor_port_pipeline_layout(positions, graph)
    elif graph.inline_frames:
        repack_inline_frame_columns(positions, graph)
        stack_inline_frame_positions(positions, graph, min_gap=min_gap)
    if not _graph_has_tensor_ports(graph) and _fanout_branch_node_groups(positions):
        stack_fanout_branch_columns(positions, graph, min_gap=min_gap)
        _align_merge_nodes(positions, graph)
    _layout_fork_join_branches(positions, graph)
    if not _graph_has_tensor_ports(graph):
        realign_fanout_branch_columns(positions, graph)
        _center_align_vertical_chains(positions, graph)
    clear_merge_feeder_columns(positions, graph)


def _layout_zone_gap(min_gap: float) -> float:
    from visualizer.sizing import min_horizontal_block_gap

    return max(min_gap * 2, min_horizontal_block_gap())


def _layout_zones_already_separated(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float,
) -> bool:
    """True when layout zones already honor side-branch separation."""
    fanout, main, side, input_hidden = _classify_layout_zones(positions, graph)
    zone_gap = _layout_zone_gap(min_gap)
    eps = zone_gap / 4

    if not side:
        return True
    core_bounds = _cluster_horizontal_bounds(positions, main | input_hidden)
    side_bounds = _cluster_horizontal_bounds(positions, side)
    if core_bounds is None or side_bounds is None:
        return False
    _core_left, core_right = core_bounds
    side_left, _side_right = side_bounds
    return abs(side_left - (core_right + zone_gap)) <= eps


def _anchor_layout_zone_extents(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    indices: set[int],
    *,
    min_left: float | None,
    max_right: float | None,
) -> None:
    """Shift settled zone columns to honor left/right layout bounds without re-separating."""
    if not indices:
        return
    content_left, content_right = _content_horizontal_extent_from_positions(positions, graph)
    if min_left is not None:
        _shift_node_indices(positions, indices, min_left - content_left)
        content_left, content_right = _content_horizontal_extent_from_positions(positions, graph)
    if max_right is None:
        return
    if content_right <= max_right:
        return
    overflow = content_right - max_right
    _shift_node_indices(positions, indices, -overflow)
    content_left, _content_right = _content_horizontal_extent_from_positions(positions, graph)
    target_min_left = min_left if min_left is not None else content_left
    if content_left < target_min_left:
        _shift_node_indices(positions, indices, target_min_left - content_left)


def _center_input_over_consumers(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    input_hidden: set[int],
) -> None:
    """Place synthetic inputs over the horizontal center of their fan-out consumers."""
    from visualizer.computation_graph import _fanout_branch_index

    for index in input_hidden:
        targets = [target for source, target in graph.links if source == index]
        fanout_targets = [
            target
            for target in targets
            if _fanout_branch_index(graph.nodes[target]) is not None
        ]
        if fanout_targets:
            by_branch: dict[int, int] = {}
            for target in fanout_targets:
                branch = _fanout_branch_index(graph.nodes[target])
                if branch is not None:
                    by_branch.setdefault(branch, target)
            targets = list(by_branch.values())
        if not targets:
            continue
        left = min(positions[target].cx - positions[target].width / 2 for target in targets)
        right = max(positions[target].cx + positions[target].width / 2 for target in targets)
        positions[index].cx = (left + right) / 2


def _place_layout_zones(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    cx: float,
    min_gap: float,
    max_right: float | None = None,
    min_left: float | None = None,
) -> None:
    """Keep side branches outside the core diagram and anchor content to layout bounds."""
    fanout, main, side, input_hidden = _classify_layout_zones(positions, graph)
    all_indices = fanout | main | side | input_hidden
    zone_gap = _layout_zone_gap(min_gap)

    if _layout_zones_already_separated(positions, graph, min_gap=min_gap):
        _center_input_over_consumers(positions, graph, input_hidden)
        _anchor_layout_zone_extents(
            positions,
            graph,
            all_indices,
            min_left=min_left,
            max_right=max_right,
        )
        return

    core_main = main | input_hidden
    if min_left is not None:
        content_left, _content_right = _content_horizontal_extent_from_positions(
            positions,
            graph,
        )
        _shift_node_indices(positions, all_indices, -content_left)
    else:
        core_bounds = _cluster_horizontal_bounds(positions, core_main)
        if core_bounds is not None:
            core_center = (core_bounds[0] + core_bounds[1]) / 2
            _shift_node_indices(positions, all_indices, cx - core_center)

    core_bounds = _cluster_horizontal_bounds(positions, core_main)
    if core_bounds is not None and side:
        _core_left, core_right = core_bounds
        side_bounds = _cluster_horizontal_bounds(positions, side)
        if side_bounds is not None:
            side_left, _side_right = side_bounds
            _shift_node_indices(positions, side, core_right + zone_gap - side_left)

    _center_input_over_consumers(positions, graph, input_hidden)

    _anchor_layout_zone_extents(
        positions,
        graph,
        all_indices,
        min_left=min_left,
        max_right=max_right,
    )


def _content_horizontal_extent_from_positions(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> tuple[float, float]:
    from visualizer.computation_graph import _node_content_left, _node_content_right

    if not positions:
        return 0.0, 0.0
    left = min(_node_content_left(pos) for pos in positions)
    right = max(_node_content_right(pos) for pos in positions)
    return left, right


def _separate_overlapping_inline_frames(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    elements: list[MeasuredElement],
    *,
    min_gap: float,
) -> None:
    """Push apart inline frames that still share horizontal space."""
    frames = sorted(
        (element for element in elements if element.kind == "inline_frame"),
        key=lambda element: element.bounds.left,
    )
    frame_members = {
        frame.frame_id: set(frame.node_indices) for frame in graph.inline_frames
    }
    for left_index in range(len(frames) - 1):
        left = frames[left_index]
        right = frames[left_index + 1]
        left_members = frame_members.get(left.frame_id or "", set())
        right_members = frame_members.get(right.frame_id or "", set())
        if left_members and right_members and (
            left_members.issubset(right_members) or right_members.issubset(left_members)
        ):
            continue
        overlap = left.bounds.right + min_gap - right.bounds.left
        if overlap <= 0:
            continue
        shift = overlap + INLINE_FRAME_SEPARATION_EPS
        members = frame_members.get(right.frame_id or "", set())
        _shift_node_indices(positions, members, shift)
        right.bounds = ContentBounds(
            left=right.bounds.left + shift,
            right=right.bounds.right + shift,
            bottom=right.bounds.bottom,
            top=right.bounds.top,
        )


def _shift_clear_forbidden_regions(
    positions: list[LayoutPosition],
    elements: list[MeasuredElement],
    forbidden_regions: list[ContentBounds],
    *,
    min_gap: float,
) -> None:
    """Move the whole diagram left when tiles intrude on forbidden regions."""
    if not forbidden_regions:
        return
    shift = 0.0
    for region in forbidden_regions:
        for element in elements:
            if element.kind not in {"box", "combine", "inline_frame"}:
                continue
            if not element.bounds.overlaps(region, min_gap=min_gap):
                continue
            needed = element.bounds.right + min_gap - region.left
            shift = min(shift, -needed)
    if shift < 0:
        _shift_positions(positions, shift)


def _nudge_clear_frame_caption_overlaps(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    elements: list[MeasuredElement],
    *,
    min_gap: float,
) -> None:
    """Shift tiles to the right when a frame caption extends past its frame border."""
    frame_members = _frame_member_indices(graph)
    member_sets = {
        frame.frame_id: set(frame.node_indices) for frame in graph.inline_frames
    }
    frame_bounds = {
        element.frame_id: element.bounds
        for element in elements
        if element.kind == "inline_frame" and element.frame_id
    }
    captions = [
        element
        for element in elements
        if element.kind in {"frame_label", "frame_sublabel"} and element.frame_id
    ]
    frame_up_shifts: dict[str, float] = {}
    frame_right_shifts: dict[str, float] = {}
    tile_up_shifts: dict[int, float] = {}
    tile_right_shifts: dict[int, float] = {}

    for caption in captions:
        caption_frame_id = caption.frame_id
        owner = frame_bounds.get(caption_frame_id or "")
        if owner is None:
            continue
        caption_side = _caption_side_for_bounds(caption.bounds, owner, gap=min_gap)
        for element in elements:
            if element.kind not in {"box", "combine", "inline_frame"}:
                continue
            if element.frame_id and element.frame_id == caption_frame_id:
                continue
            if (
                element.node_index is not None
                and frame_members.get(element.node_index) == caption_frame_id
            ):
                if caption.bounds.overlaps(element.bounds, min_gap=min_gap):
                    if caption_side == "left":
                        shift = caption.bounds.right + min_gap - element.bounds.left
                        if shift > 0 and caption_frame_id:
                            frame_right_shifts[caption_frame_id] = max(
                                frame_right_shifts.get(caption_frame_id, 0.0),
                                shift,
                            )
                    elif caption_frame_id:
                        shift = element.bounds.top + min_gap - caption.bounds.bottom
                        if shift > 0:
                            frame_up_shifts[caption_frame_id] = max(
                                frame_up_shifts.get(caption_frame_id, 0.0),
                                shift,
                            )
                continue
            if caption_side == "left":
                if element.frame_id == caption_frame_id:
                    continue
                if not caption.bounds.overlaps(element.bounds, min_gap=min_gap):
                    continue
                if element.kind == "inline_frame" and element.frame_id:
                    shift = caption.bounds.right + min_gap - element.bounds.left
                    if shift > 0 and element.frame_id:
                        frame_right_shifts[element.frame_id] = max(
                            frame_right_shifts.get(element.frame_id, 0.0),
                            shift,
                        )
                    continue
                if element.bounds.left >= owner.left - min_gap:
                    continue
                shift = caption.bounds.right + min_gap - element.bounds.left
                if shift > 0 and element.node_index is not None:
                    frame_id = frame_members.get(element.node_index)
                    if frame_id:
                        frame_right_shifts[frame_id] = max(
                            frame_right_shifts.get(frame_id, 0.0),
                            shift,
                        )
                    else:
                        tile_right_shifts[element.node_index] = max(
                            tile_right_shifts.get(element.node_index, 0.0),
                            shift,
                        )
                continue
            if element.bounds.left < owner.right + min_gap:
                continue
            if not caption.bounds.overlaps(element.bounds, min_gap=min_gap):
                continue
            if element.kind == "inline_frame" and element.frame_id:
                indices = member_sets.get(element.frame_id, set())
            elif element.node_index is not None:
                frame_id = frame_members.get(element.node_index)
                indices = member_sets.get(frame_id, {element.node_index}) if frame_id else {element.node_index}
            else:
                continue
            if not indices:
                continue
            if element.bounds.left >= owner.right - min_gap:
                horizontal = caption.bounds.right + min_gap - element.bounds.left
                if horizontal >= 0:
                    horizontal = max(horizontal, min_gap)
                    if element.kind == "inline_frame" and element.frame_id:
                        frame_right_shifts[element.frame_id] = max(
                            frame_right_shifts.get(element.frame_id, 0.0),
                            horizontal,
                        )
                    else:
                        for index in indices:
                            tile_right_shifts[index] = max(
                                tile_right_shifts.get(index, 0.0),
                                horizontal,
                            )
                vertical = element.bounds.top + min_gap - caption.bounds.bottom
                if vertical > 0:
                    if element.kind == "inline_frame" and element.frame_id:
                        frame_up_shifts[element.frame_id] = max(
                            frame_up_shifts.get(element.frame_id, 0.0),
                            vertical,
                        )
                    else:
                        for index in indices:
                            tile_up_shifts[index] = max(
                                tile_up_shifts.get(index, 0.0),
                                vertical,
                            )

    for frame_id, shift in frame_right_shifts.items():
        indices = member_sets.get(frame_id, set())
        if indices:
            _shift_node_indices(positions, indices, shift)
    for frame_id, shift in frame_up_shifts.items():
        for index in member_sets.get(frame_id, set()):
            positions[index].top_y -= shift
    for index, shift in tile_right_shifts.items():
        positions[index].cx += shift
    for index, shift in tile_up_shifts.items():
        positions[index].top_y -= shift


def _redock_tensor_ports_after_layout(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    elements: list[MeasuredElement],
    *,
    min_gap: float,
) -> None:
    """Re-seat tensor ports on their consumers after vertical layout adjustments."""
    from visualizer.computation_graph import (
        _dock_single_consumer_tensor_ports,
        _graph_has_tensor_ports,
    )

    if not _graph_has_tensor_ports(graph):
        return
    _dock_single_consumer_tensor_ports(positions, graph)
    _ensure_tensor_ports_clear_frame_captions(
        positions,
        elements,
        graph,
        min_gap=min_gap,
    )


def _repack_after_caption_nudges(
    ax: Axes,
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    plan: DetailDrawPlan,
    *,
    detail_fill: str,
    min_gap: float,
) -> None:
    from visualizer.render import _build_detail_draw_plan

    current_plan = plan
    for _ in range(4):
        elements = collect_measured_elements(
            ax,
            graph,
            positions,
            current_plan,
            detail_fill=detail_fill,
        )
        _nudge_clear_frame_caption_overlaps(positions, graph, elements, min_gap=min_gap)
        _align_and_stack_inline_frames(positions, graph)
        current_plan = _build_detail_draw_plan(positions, graph, input_sublabel=current_plan.input_sublabel)


def _resolve_obstacle_layout(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    elements: list[MeasuredElement],
    *,
    cx: float,
    min_gap: float,
    forbidden_regions: list[ContentBounds] | None = None,
    max_right: float | None = None,
    min_left: float | None = None,
) -> None:
    """Repack columns and frames after measured resize."""
    from visualizer.computation_graph import (
        _graph_has_tensor_ports,
        finalize_tensor_port_pipeline_layout,
    )

    _align_and_stack_inline_frames(positions, graph)
    _place_layout_zones(
        positions,
        graph,
        cx=cx,
        min_gap=min_gap,
        max_right=max_right,
        min_left=min_left,
    )
    if _graph_has_tensor_ports(graph):
        finalize_tensor_port_pipeline_layout(positions, graph, min_left=min_left)
        _redock_tensor_ports_after_layout(
            positions,
            graph,
            elements,
            min_gap=min_gap,
        )
        resolve_measured_overlaps(positions, graph, min_gap=min_gap)
        _resolve_same_row_tile_overlaps(positions, min_gap=min_gap, graph=graph)
    _separate_overlapping_inline_frames(positions, graph, elements, min_gap=min_gap)
    if forbidden_regions:
        _shift_clear_forbidden_regions(positions, elements, forbidden_regions, min_gap=min_gap)


def _separate_parallel_tiles_from_inline_frames(
    positions: list[LayoutPosition],
    elements: list[MeasuredElement],
    graph: ComputationGraph,
    *,
    min_gap: float,
) -> None:
    """Shift parallel branch tiles just outside dotted inline-frame borders."""
    fanout, _main, side, _input_hidden = _classify_layout_zones(positions, graph)
    parallel_indices = fanout | side
    frame_members = _frame_member_indices(graph)
    branch_groups = _fanout_branch_node_groups(positions)
    frames = [element for element in elements if element.kind == "inline_frame"]
    boxes = [
        element
        for element in elements
        if element.kind == "box" and element.node_index is not None
    ]
    for box in boxes:
        box_index = box.node_index
        if box_index is None or box_index in frame_members:
            continue
        pos = positions[box_index]
        if pos.spec.synthetic in {SYNTHETIC_TENSOR, SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}:
            continue
        if box_index not in parallel_indices:
            continue
        for frame in frames:
            if box_index in frame.frame_node_indices:
                continue
            if not box.bounds.overlaps(frame.bounds, min_gap=min_gap):
                continue
            if _fanout_branch_index(pos.spec) is not None:
                branch = _fanout_branch_index(pos.spec)
                column = branch_groups.get(branch, [box_index]) if branch is not None else [box_index]
                shift_left = frame.bounds.left - min_gap - box.bounds.right
                shift_right = frame.bounds.right + min_gap - box.bounds.left
                if shift_left < 0 and (shift_right <= 0 or abs(shift_left) <= shift_right):
                    for index in column:
                        positions[index].cx += shift_left
                elif shift_right > 0:
                    for index in column:
                        positions[index].cx += shift_right
            else:
                shift = frame.bounds.right + min_gap - box.bounds.left
                if shift > 0:
                    pos.cx += shift


def _separate_boxes_from_nested_inline_frames(
    positions: list[LayoutPosition],
    elements: list[MeasuredElement],
    graph: ComputationGraph,
    *,
    min_gap: float,
) -> None:
    """Shift sibling tiles right when they overlap a nested inline frame in the same parent."""
    frame_members = _frame_member_indices(graph)
    member_sets = {
        frame.frame_id: set(frame.node_indices) for frame in graph.inline_frames
    }
    frames = [
        element
        for element in elements
        if element.kind == "inline_frame" and element.frame_id
    ]
    boxes = [
        element
        for element in elements
        if element.kind == "box" and element.node_index is not None
    ]
    for box in boxes:
        box_index = box.node_index
        if box_index is None or box_index >= len(positions):
            continue
        parent_frame_id = frame_members.get(box_index)
        if parent_frame_id is None:
            continue
        parent_members = member_sets.get(parent_frame_id, set())
        for frame in frames:
            nested_id = frame.frame_id
            if nested_id is None or nested_id == parent_frame_id:
                continue
            nested_members = member_sets.get(nested_id, set())
            if not nested_members or not nested_members.issubset(parent_members):
                continue
            if box_index in nested_members:
                continue
            if not box.bounds.overlaps(frame.bounds, min_gap=min_gap):
                continue
            shift = frame.bounds.right + min_gap - box.bounds.left
            if shift > 0:
                positions[box_index].cx += shift


def _nudge_apart_remaining_tiles(
    positions: list[LayoutPosition],
    elements: list[MeasuredElement],
    graph: ComputationGraph,
    *,
    min_gap: float,
) -> None:
    """Last-resort nudge for tile pairs that still intersect."""
    frame_members = _frame_member_indices(graph)
    frame_member_sets = {
        frame.frame_id: set(frame.node_indices) for frame in graph.inline_frames
    }
    boxes = [
        element
        for element in elements
        if element.kind == "box" and element.node_index is not None
    ]
    frames = [element for element in elements if element.kind == "inline_frame"]

    for box in boxes:
        box_index = box.node_index
        if box_index is None or box_index >= len(positions):
            continue
        pos = positions[box_index]
        box_frame = frame_members.get(box_index)
        for frame in frames:
            if box_frame and frame.frame_id == box_frame:
                continue
            if box_index in frame.frame_node_indices:
                continue
            if not box.bounds.overlaps(frame.bounds, min_gap=min_gap):
                continue
            if box_frame is not None:
                if pos.spec.synthetic == SYNTHETIC_TENSOR:
                    targets = [target for source, target in graph.links if source == box_index]
                    if len(targets) == 1:
                        if box.bounds.overlaps(frame.bounds, min_gap=min_gap):
                            shift = frame.bounds.top + min_gap - box.bounds.bottom
                            if shift > 0:
                                pos.top_y += shift
                        continue
                    shift = frame.bounds.right + min_gap - box.bounds.left
                    if shift > 0:
                        pos.cx += shift
                    continue
                members = frame_member_sets.get(box_frame, {box_index})
                if box.bounds.left < frame.bounds.left:
                    shift = frame.bounds.left - min_gap - box.bounds.right
                    if shift < 0:
                        _shift_node_indices(positions, members, shift)
                else:
                    shift = frame.bounds.right + min_gap - box.bounds.left
                    if shift > 0:
                        _shift_node_indices(positions, members, shift)
                continue
            if _fanout_branch_index(pos.spec) is not None:
                shift = frame.bounds.left - min_gap - box.bounds.right
                if shift < 0:
                    pos.cx += shift
                continue
            if pos.spec.key.startswith("sideproducer") or (
                "sidefeed" in pos.spec.key and ":gate_act" in pos.spec.key
            ):
                shift = frame.bounds.right + min_gap - box.bounds.left
                if shift > 0:
                    pos.cx += shift
                continue
            if any(
                box_index == source and target < len(graph.nodes)
                for source, target in graph.side_entry_links
            ):
                shift = frame.bounds.right + min_gap - box.bounds.left
                if shift > 0:
                    pos.cx += shift
                continue
            if box.bounds.left < frame.bounds.left:
                shift = frame.bounds.left - min_gap - box.bounds.right
                if shift < 0:
                    pos.cx += shift
            else:
                shift = frame.bounds.right + min_gap - box.bounds.left
                if shift > 0:
                    pos.cx += shift

    tiles = [
        element
        for element in elements
        if element.kind in _TILE_KINDS and element.node_index is not None
    ]
    for left_index, left in enumerate(tiles):
        for right in tiles[left_index + 1 :]:
            left_index_id = left.node_index
            right_index_id = right.node_index
            if left_index_id is None or right_index_id is None:
                continue
            left_frame = frame_members.get(left_index_id)
            right_frame = frame_members.get(right_index_id)
            if left_frame and left_frame == right_frame:
                continue
            if not left.bounds.overlaps(right.bounds, min_gap=min_gap):
                continue
            if right_index_id >= len(positions):
                continue
            overlap = left.bounds.right + min_gap - right.bounds.left
            if overlap > 0:
                shift_indices = {right_index_id}
                if right_frame is not None:
                    shift_indices = frame_member_sets.get(right_frame, shift_indices)
                _shift_node_indices(positions, shift_indices, overlap)


def _content_horizontal_extent(elements: list[MeasuredElement]) -> tuple[float, float]:
    measured = [
        element
        for element in elements
        if element.kind in _OBSTACLE_KINDS | {"text_overflow", "frame_label", "frame_sublabel"}
    ]
    if not measured:
        return 0.0, 0.0
    left = min(element.bounds.left for element in measured)
    right = max(element.bounds.right for element in measured)
    return left, right


def measure_detail_tree_content_width(
    ax: Axes,
    tree,
    *,
    cx: float,
    detail_fill: str,
    min_left: float,
    input_sublabel: str | None = None,
    prefix_steps: list | None = None,
    layout_block_w: float = 18.0,
    basic_ops=None,
) -> float:
    """Measure the finalized horizontal extent of one detail section."""
    from visualizer.block_tree import BlockNode
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.computation_graph import (
        _estimate_graph_height,
        _minimum_graph_layout_width,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import COLORS, MEASURE_CANVAS_WIDTH, _build_detail_draw_plan

    if not isinstance(tree, BlockNode):
        return 1.2

    fill = detail_fill or COLORS["detail_fill"]
    graph = build_computation_graph(
        tree,
        prefix_steps=prefix_steps,
        basic_ops=basic_ops or BasicOpFilter.for_detailed(),
    )
    min_required_width = _minimum_graph_layout_width(graph) + 0.12
    layout_block_w = max(layout_block_w, min_required_width)
    from visualizer.text_measure import ensure_diagram_measure_axes

    ensure_diagram_measure_axes(ax)
    positions = None
    finalized_ok = False
    try:
        measure_graph_node_sizes(ax, graph, input_sublabel=None)
        positions, _ = layout_computation_graph(
            graph,
            cx=cx,
            top_y=10.0,
            block_w=layout_block_w,
            block_h=_estimate_graph_height(graph),
            content_left=min_left,
        )
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=input_sublabel,
            cx=cx,
            top_y=10.0,
            detail_fill=fill,
            min_left=min_left,
            forbidden_regions=None,
        )
        finalized_ok = True
    except LayoutValidationError:
        pass
    finally:
        from visualizer.render import _detail_content_bounds

        fallback_width = max(1.2, min_required_width + 0.05)
        if not finalized_ok or positions is None:
            width = fallback_width
        else:
            frame_left, frame_right, _frame_bottom, _frame_top = _detail_content_bounds(positions)
            width = max(1.2, frame_right - frame_left + 0.05)
            if width > max(25.0, min_required_width * 3):
                width = fallback_width
        return width


def measure_max_detail_content_width(
    ax: Axes,
    block_trees: list[tuple[str, object]],
    *,
    cx: float,
    detail_fill: str,
    min_left: float,
    layout_block_w: float = 18.0,
) -> float:
    """Pre-measure the widest shrink-wrapped detail section to size the canvas."""
    from visualizer.block_tree import BlockNode
    from visualizer.render import _format_input_source_sublabel

    max_width = 1.2
    for _title, tree in block_trees:
        if not isinstance(tree, BlockNode):
            continue
        sub = _format_input_source_sublabel(tree.input_source)
        width = measure_detail_tree_content_width(
            ax,
            tree,
            cx=cx,
            detail_fill=detail_fill,
            min_left=min_left,
            input_sublabel=sub,
            layout_block_w=layout_block_w,
        )
        max_width = max(max_width, width)
    return max_width


def measure_max_detail_section_width(
    ax: Axes,
    spec,
    *,
    cx: float,
    detail_fill: str,
    min_left: float,
    layout_block_w: float = 18.0,
) -> float:
    """Return the widest shrink-wrapped detail subsection (for canvas sizing)."""
    from visualizer.render import COLORS, _detail_sections_to_render

    fill = detail_fill or COLORS["detail_fill"]
    max_width = 1.2
    for _title, tree, input_sublabel in _detail_sections_to_render(spec):
        width = measure_detail_tree_content_width(
            ax,
            tree,
            cx=cx,
            detail_fill=fill,
            min_left=min_left,
            input_sublabel=input_sublabel,
            layout_block_w=layout_block_w,
        )
        max_width = max(max_width, width)
    return max_width


def measure_uniform_detail_section_width(
    ax: Axes,
    spec,
    *,
    cx: float,
    detail_fill: str,
    min_left: float,
    layout_block_w: float = 18.0,
) -> float:
    """Shrink-wrap each rendered subsection and return the widest width."""
    from visualizer.render import COLORS, _detail_sections_to_render

    fill = detail_fill or COLORS["detail_fill"]
    uniform_w = 1.2
    for _title, tree, input_sublabel in _detail_sections_to_render(spec):
        width = measure_detail_tree_content_width(
            ax,
            tree,
            cx=cx,
            detail_fill=fill,
            min_left=min_left,
            input_sublabel=input_sublabel,
            layout_block_w=layout_block_w,
        )
        uniform_w = max(uniform_w, width)
    return uniform_w


def _shift_positions(positions: list[LayoutPosition], delta_x: float) -> None:
    if abs(delta_x) <= 1e-9:
        return
    for pos in positions:
        pos.cx += delta_x


def _pack_tensor_port_pipeline_layout(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_left: float | None,
) -> None:
    """Repack kernel pipeline frames and realign modeling tensor ports after horizontal nudges."""
    from visualizer.computation_graph import (
        _graph_has_tensor_ports,
        finalize_tensor_port_pipeline_layout,
    )

    if not _graph_has_tensor_ports(graph):
        return
    finalize_tensor_port_pipeline_layout(positions, graph, min_left=min_left)


def _clamp_content_horizontal(
    positions: list[LayoutPosition],
    elements: list[MeasuredElement],
    *,
    min_left: float | None,
    max_right: float | None,
) -> None:
    """Shift diagram content horizontally to stay inside allowed bounds when possible.

    Tile widths are never scaled down — text must always fit inside its box.
    When content is wider than the allowed band, leave sizes intact and rely on
    downstream overlap resolution and figure fitting instead of compressing tiles.
    """
    if min_left is None and max_right is None:
        return
    content_left, content_right = _content_horizontal_extent(elements)
    shift = 0.0
    if max_right is not None and content_right > max_right:
        shift -= content_right - max_right
    if min_left is not None:
        shift += min_left - (content_left + shift)
    _shift_positions(positions, shift)


def _forbidden_max_right(
    forbidden_regions: list[ContentBounds] | None,
    *,
    min_gap: float,
) -> float | None:
    if not forbidden_regions:
        return None
    from visualizer.render import INLINE_FRAME_PAD

    return min(region.left for region in forbidden_regions) - min_gap - INLINE_FRAME_PAD


def _anchor_detail_layout_to_top_y(
    positions: list[LayoutPosition],
    *,
    top_y: float,
) -> None:
    """Shift diagram content so its visual top aligns with the section ``top_y`` anchor."""
    from visualizer.render import _detail_content_extents

    if not positions:
        return
    _left, _right, _bottom, max_top = _detail_content_extents(positions)
    shift = top_y - max_top
    if abs(shift) <= 1e-6:
        return
    for pos in positions:
        pos.top_y += shift


def _finalize_spine_aligned_plan(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    input_sublabel: str | None,
) -> DetailDrawPlan:
    """Center-align spine columns and rebuild the draw plan."""
    from visualizer.computation_graph import _align_merge_nodes, _center_align_vertical_chains
    from visualizer.render import _build_detail_draw_plan

    _align_merge_nodes(positions, graph)
    _center_align_vertical_chains(positions, graph)
    from visualizer.computation_graph import _align_tensor_port_merge_nodes, _graph_has_tensor_ports

    if _graph_has_tensor_ports(graph):
        _align_tensor_port_merge_nodes(positions, graph)
    return _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)


def _ensure_tensor_ports_clear_frame_captions(
    positions: list[LayoutPosition],
    elements: list[MeasuredElement],
    graph: ComputationGraph,
    *,
    min_gap: float,
) -> None:
    """Raise tensor ports when they overlap inline-frame caption text."""
    from visualizer.text_measure import box_bounds_at

    port_indices = [
        index for index, spec in enumerate(graph.nodes) if spec.synthetic == SYNTHETIC_TENSOR
    ]
    if not port_indices:
        return
    captions = [element for element in elements if element.kind == "frame_label"]
    if not captions:
        return
    for port_index in port_indices:
        pos = positions[port_index]
        port_bounds = box_bounds_at(pos.cx, pos.top_y, pos.width, pos.height)
        for caption in captions:
            if not port_bounds.overlaps(caption.bounds, min_gap=min_gap):
                continue
            shift = caption.bounds.top + min_gap - port_bounds.bottom
            if shift > 0:
                pos.top_y += shift
                port_bounds = box_bounds_at(pos.cx, pos.top_y, pos.width, pos.height)


def finalize_detail_layout(
    ax: Axes,
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    *,
    input_sublabel: str | None,
    cx: float,
    top_y: float,
    detail_fill: str,
    min_gap: float = DEFAULT_MIN_GAP,
    max_passes: int = 12,
    min_left: float | None = None,
    forbidden_regions: list[ContentBounds] | None = None,
) -> DetailDrawPlan:
    """
    Pre-render all boxes and labels to measure bounds, re-layout, then validate.

    Must run before drawing connectors or tiles.
    """
    from visualizer.render import COLORS, _build_detail_draw_plan, _resize_input_nodes

    fill = detail_fill or COLORS["detail_fill"]
    layout_top_y = max(top_y, LAYOUT_MIN_TOP_Y)
    forbidden = list(forbidden_regions or [])
    max_right = _forbidden_max_right(forbidden, min_gap=min_gap)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)

    for _pass in range(max_passes):
        apply_measured_node_sizes(ax, positions, plan)
        _resize_input_nodes(positions, input_sublabel)
        apply_measured_node_sizes(ax, positions, plan)
        _align_and_stack_inline_frames(positions, graph)
        if positions:
            resolve_measured_overlaps(
                positions,
                graph,
                top_y=layout_top_y,
                min_gap=MIN_HORIZONTAL_BLOCK_GAP,
            )
            _align_and_stack_inline_frames(positions, graph)
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
        elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
        _ensure_input_above_inline_frames(positions, elements, min_gap=VALIDATE_MIN_GAP)
        _resolve_obstacle_layout(
            positions,
            graph,
            elements,
            cx=cx,
            min_gap=VALIDATE_MIN_GAP,
            forbidden_regions=forbidden,
            max_right=max_right,
            min_left=min_left,
        )
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
        elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
        _clamp_content_horizontal(
            positions,
            elements,
            min_left=min_left,
            max_right=max_right,
        )
        _repack_after_caption_nudges(
            ax,
            graph,
            positions,
            plan,
            detail_fill=fill,
            min_gap=VALIDATE_MIN_GAP,
        )
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
        elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
        _separate_parallel_tiles_from_inline_frames(
            positions,
            elements,
            graph,
            min_gap=VALIDATE_MIN_GAP,
        )
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
        elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
        report = validate_render_layout(elements, min_gap=VALIDATE_MIN_GAP, forbidden_regions=forbidden)
        if report.ok:
            break
        _nudge_apart_remaining_tiles(positions, elements, graph, min_gap=VALIDATE_MIN_GAP)
        _separate_parallel_tiles_from_inline_frames(
            positions,
            elements,
            graph,
            min_gap=VALIDATE_MIN_GAP,
        )
        if positions:
            resolve_measured_overlaps(
                positions,
                graph,
                top_y=layout_top_y,
                min_gap=MIN_HORIZONTAL_BLOCK_GAP,
            )
            _align_and_stack_inline_frames(positions, graph)
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)

    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _resolve_obstacle_layout(
        positions,
        graph,
        elements,
        cx=cx,
        min_gap=VALIDATE_MIN_GAP,
        forbidden_regions=forbidden,
        max_right=max_right,
        min_left=min_left,
    )
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _clamp_content_horizontal(
        positions,
        elements,
        min_left=min_left,
        max_right=max_right,
    )
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    if positions:
        resolve_measured_overlaps(
            positions,
            graph,
            top_y=layout_top_y,
            min_gap=MIN_HORIZONTAL_BLOCK_GAP,
        )
        _align_and_stack_inline_frames(positions, graph)
    _repack_after_caption_nudges(
        ax,
        graph,
        positions,
        plan,
        detail_fill=fill,
        min_gap=VALIDATE_MIN_GAP,
    )
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _ensure_input_above_inline_frames(positions, elements, min_gap=VALIDATE_MIN_GAP)
    _separate_parallel_tiles_from_inline_frames(
        positions,
        elements,
        graph,
        min_gap=VALIDATE_MIN_GAP,
    )
    _separate_overlapping_inline_frames(
        positions,
        graph,
        elements,
        min_gap=VALIDATE_MIN_GAP,
    )
    _separate_boxes_from_nested_inline_frames(
        positions,
        elements,
        graph,
        min_gap=VALIDATE_MIN_GAP,
    )
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _layout_fork_join_branches(positions, graph)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _compact_synthetic_input_spacing(
        positions,
        graph,
        min_gap=VALIDATE_MIN_GAP,
        elements=elements,
    )
    _separate_parallel_tiles_from_inline_frames(
        positions,
        elements,
        graph,
        min_gap=VALIDATE_MIN_GAP,
    )
    _layout_fork_join_branches(positions, graph)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _compact_synthetic_input_spacing(
        positions,
        graph,
        min_gap=VALIDATE_MIN_GAP,
        elements=elements,
    )
    _anchor_detail_layout_to_top_y(positions, top_y=top_y)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    _resolve_same_row_tile_overlaps(positions, min_gap=VALIDATE_MIN_GAP, graph=graph)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _separate_parallel_tiles_from_inline_frames(
        positions,
        elements,
        graph,
        min_gap=VALIDATE_MIN_GAP,
    )
    elements = _apply_inline_frame_label_layout(
        ax,
        graph,
        positions,
        plan,
        detail_fill=fill,
        min_gap=VALIDATE_MIN_GAP,
        min_left=min_left,
    )
    _separate_overlapping_inline_frames(
        positions,
        graph,
        elements,
        min_gap=VALIDATE_MIN_GAP,
    )
    _resolve_same_row_tile_overlaps(positions, min_gap=VALIDATE_MIN_GAP, graph=graph)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    if positions:
        _align_and_stack_inline_frames(positions, graph)
        _resolve_same_row_tile_overlaps(positions, min_gap=VALIDATE_MIN_GAP, graph=graph)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _separate_parallel_tiles_from_inline_frames(
        positions,
        elements,
        graph,
        min_gap=VALIDATE_MIN_GAP,
    )
    _nudge_apart_remaining_tiles(positions, elements, graph, min_gap=VALIDATE_MIN_GAP)
    if positions:
        _align_and_stack_inline_frames(positions, graph)
    plan = _finalize_spine_aligned_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    _layout_fork_join_branches(positions, graph)
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _separate_parallel_tiles_from_inline_frames(
        positions,
        elements,
        graph,
        min_gap=VALIDATE_MIN_GAP,
    )
    _layout_fork_join_branches(positions, graph)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _compact_synthetic_input_spacing(
        positions,
        graph,
        min_gap=VALIDATE_MIN_GAP,
        elements=elements,
    )
    _anchor_detail_layout_to_top_y(positions, top_y=top_y)
    _ensure_synthetic_input_clears_consumers(positions, graph)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    _resolve_same_row_tile_overlaps(positions, min_gap=VALIDATE_MIN_GAP, graph=graph)
    elements = _apply_inline_frame_label_layout(
        ax,
        graph,
        positions,
        plan,
        detail_fill=fill,
        min_gap=VALIDATE_MIN_GAP,
        min_left=min_left,
    )
    _separate_overlapping_inline_frames(
        positions,
        graph,
        elements,
        min_gap=VALIDATE_MIN_GAP,
    )
    _pack_tensor_port_pipeline_layout(positions, graph, min_left=min_left)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _clamp_content_horizontal(
        positions,
        elements,
        min_left=min_left,
        max_right=max_right,
    )
    _pack_tensor_port_pipeline_layout(positions, graph, min_left=min_left)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _separate_parallel_tiles_from_inline_frames(
        positions,
        elements,
        graph,
        min_gap=VALIDATE_MIN_GAP,
    )
    _separate_overlapping_inline_frames(
        positions,
        graph,
        elements,
        min_gap=VALIDATE_MIN_GAP,
    )
    _nudge_apart_remaining_tiles(positions, elements, graph, min_gap=VALIDATE_MIN_GAP)
    _resolve_same_row_tile_overlaps(positions, min_gap=VALIDATE_MIN_GAP, graph=graph)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    elements = _apply_inline_frame_label_layout(
        ax,
        graph,
        positions,
        plan,
        detail_fill=fill,
        min_gap=VALIDATE_MIN_GAP,
        min_left=min_left,
    )
    _ensure_tensor_ports_clear_frame_captions(
        positions,
        elements,
        graph,
        min_gap=VALIDATE_MIN_GAP,
    )
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    for _ in range(4):
        elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
        _nudge_clear_frame_caption_overlaps(positions, graph, elements, min_gap=VALIDATE_MIN_GAP)
        _ensure_tensor_ports_clear_frame_captions(
            positions,
            elements,
            graph,
            min_gap=VALIDATE_MIN_GAP,
        )
        _resolve_same_row_tile_overlaps(positions, min_gap=VALIDATE_MIN_GAP, graph=graph)
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
        enforce_text_fit_node_sizes(ax, positions, plan)
        elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
        frame_caption_overlaps = [
            line
            for line in validate_render_layout(elements, min_gap=VALIDATE_MIN_GAP, forbidden_regions=forbidden).overlaps
            if "frame_label" in line or "frame_sublabel" in line
        ]
        if not frame_caption_overlaps:
            break
    elements = _apply_inline_frame_label_layout(
        ax,
        graph,
        positions,
        plan,
        detail_fill=fill,
        min_gap=VALIDATE_MIN_GAP,
        min_left=min_left,
    )
    for _ in range(4):
        elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
        _nudge_clear_frame_caption_overlaps(positions, graph, elements, min_gap=VALIDATE_MIN_GAP)
        elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
        frame_caption_overlaps = [
            line
            for line in validate_render_layout(elements, min_gap=VALIDATE_MIN_GAP, forbidden_regions=forbidden).overlaps
            if "frame_label" in line or "frame_sublabel" in line
        ]
        if not frame_caption_overlaps:
            break
    _layout_fork_join_branches(positions, graph)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _redock_tensor_ports_after_layout(positions, graph, elements, min_gap=VALIDATE_MIN_GAP)
    from visualizer.computation_graph import _graph_has_tensor_ports
    from visualizer.shrinkwrap import shrinkwrap_detail_layout

    shrinkwrap_detail_layout(
        positions,
        graph,
        min_gap=VALIDATE_MIN_GAP,
        min_left=min_left,
    )
    if positions and not _graph_has_tensor_ports(graph):
        _resolve_same_row_tile_overlaps(positions, min_gap=VALIDATE_MIN_GAP, graph=graph)
        _align_and_stack_inline_frames(positions, graph)
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
        enforce_text_fit_node_sizes(ax, positions, plan)
        elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
        _separate_overlapping_inline_frames(
            positions,
            graph,
            elements,
            min_gap=VALIDATE_MIN_GAP,
        )
        _separate_parallel_tiles_from_inline_frames(
            positions,
            elements,
            graph,
            min_gap=VALIDATE_MIN_GAP,
        )
        _resolve_same_row_tile_overlaps(positions, min_gap=VALIDATE_MIN_GAP, graph=graph)
        if min_left is not None:
            from visualizer.computation_graph import _align_positions_left

            _align_positions_left(positions, min_left)
    from visualizer.computation_graph import _center_align_vertical_chains

    _center_align_vertical_chains(positions, graph)
    _layout_fork_join_branches(positions, graph)
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _separate_parallel_tiles_from_inline_frames(
        positions,
        elements,
        graph,
        min_gap=VALIDATE_MIN_GAP,
    )
    _resolve_same_row_tile_overlaps(positions, min_gap=VALIDATE_MIN_GAP, graph=graph)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    elements = _apply_inline_frame_label_layout(
        ax,
        graph,
        positions,
        plan,
        detail_fill=fill,
        min_gap=VALIDATE_MIN_GAP,
        min_left=min_left,
    )
    _compact_synthetic_input_spacing(
        positions,
        graph,
        min_gap=VALIDATE_MIN_GAP,
        elements=elements,
    )
    if positions and not _graph_has_tensor_ports(graph):
        _align_and_stack_inline_frames(positions, graph)
        realign_fanout_branch_columns(positions, graph)
        _center_align_vertical_chains(positions, graph)
    from visualizer.computation_graph import _ensure_top_entry_clearance_below_inline_frames

    _ensure_top_entry_clearance_below_inline_frames(positions, graph)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    enforce_text_fit_node_sizes(ax, positions, plan)
    elements = _apply_inline_frame_label_layout(
        ax,
        graph,
        positions,
        plan,
        detail_fill=fill,
        min_gap=VALIDATE_MIN_GAP,
        min_left=min_left,
    )
    _compact_synthetic_input_spacing(
        positions,
        graph,
        min_gap=VALIDATE_MIN_GAP,
        elements=elements,
    )
    from visualizer.computation_graph import _compact_parallel_feeder_frame_exit_stubs

    _compact_parallel_feeder_frame_exit_stubs(positions, graph)
    from visualizer.computation_graph import _compact_fanout_branch_tail_spacing

    _compact_fanout_branch_tail_spacing(positions, graph)
    from visualizer.computation_graph import (
        _align_k_proj_adjacent_to_chunk_pipeline,
        _ensure_multi_branch_input_fanout_clearance,
    )
    from visualizer.sizing import min_horizontal_block_gap

    zone_gap = max(VALIDATE_MIN_GAP * 2, min_horizontal_block_gap())
    _align_k_proj_adjacent_to_chunk_pipeline(
        positions,
        graph,
        max_cx_gap=zone_gap * 12 - 1e-3,
    )
    _ensure_multi_branch_input_fanout_clearance(
        positions,
        graph,
        min_gap=VALIDATE_MIN_GAP,
    )
    if positions and not _graph_has_tensor_ports(graph):
        from visualizer.computation_graph import (
            _compact_exit_feeder_branch_indices,
            _ensure_exit_feeder_branches_left_of_spine,
            _input_hidden_indices,
            _order_fanout_branch_positions,
        )

        input_indices = _input_hidden_indices(graph)
        if _compact_exit_feeder_branch_indices(
            graph,
            _fanout_branch_node_groups(positions),
            input_indices=input_indices,
        ):
            _order_fanout_branch_positions(positions, graph)
            _ensure_exit_feeder_branches_left_of_spine(
                positions,
                graph,
                min_gap=MIN_HORIZONTAL_BLOCK_GAP,
            )
            realign_fanout_branch_columns(positions, graph)
            _center_align_vertical_chains(positions, graph)
            _ensure_exit_feeder_branches_left_of_spine(
                positions,
                graph,
                min_gap=MIN_HORIZONTAL_BLOCK_GAP,
            )
        _resolve_same_row_tile_overlaps(
            positions,
            min_gap=VALIDATE_MIN_GAP,
            graph=graph,
        )
    if min_left is not None:
        from visualizer.computation_graph import _align_positions_left

        _align_positions_left(positions, min_left)
    saved_inline_frame_labels = plan.inline_frame_labels
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
    plan.inline_frame_labels = saved_inline_frame_labels
    enforce_text_fit_node_sizes(ax, positions, plan)
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    validate_render_layout(elements, min_gap=VALIDATE_MIN_GAP, forbidden_regions=forbidden).raise_if_invalid()
    return plan
