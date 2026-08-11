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
    _compact_synthetic_input_spacing,
    _ensure_synthetic_input_clears_consumers,
    _fanout_branch_index,
    _layout_fork_join_branches,
    _resolve_layout_overlaps,
    stack_inline_frame_positions,
    _center_align_vertical_chains,
)
from visualizer.text_measure import ContentBounds, box_bounds_at, box_label_size, floating_port_label_bounds, input_box_label_size
from visualizer.sizing import BLOCK_PAD_Y, INPUT_PAD_X, INPUT_PAD_Y, box_text_lines

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from visualizer.render import DetailDrawPlan

DEFAULT_MIN_GAP = 0.02
VALIDATE_MIN_GAP = 0.02
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
        if draw_index >= len(plan.node_draws):
            continue
        leaf, _draw_kwargs = plan.node_draws[draw_index]
        width, height = box_label_size(ax, leaf.label, leaf.sublabel, fontsize=leaf.fontsize)
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
    from visualizer.text_measure import measure_text_bounds

    elements: list[MeasuredElement] = []
    frame_members = _frame_member_indices(graph)
    draw_index = 0
    for index, pos in enumerate(positions):
        spec = pos.spec
        if _is_combine(spec.synthetic):
            center_y = pos.top_y - pos.height / 2
            half = pos.width / 2
            elements.append(
                MeasuredElement(
                    kind="combine",
                    bounds=ContentBounds(
                        left=pos.cx - half,
                        right=pos.cx + half,
                        bottom=center_y - half,
                        top=center_y + half,
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
            if not tile_bounds.contains(line_bounds, min_gap=-0.015):
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
        min_left = min(p.cx - p.width / 2 for p in frame_positions)
        max_right = max(p.cx + p.width / 2 for p in frame_positions)
        min_bottom = min(p.top_y - p.height for p in frame_positions)
        max_top = max(p.top_y for p in frame_positions)
        pad = INLINE_FRAME_PAD
        frame_width = max_right - min_left + 2 * pad
        label_lines = _inline_frame_label_lines(frame.label, frame_width)
        caption_top = max_top + pad + INLINE_FRAME_LABEL_GAP
        caption_lines = len(label_lines) + (len(frame.sublabel.split("\n")) if frame.sublabel else 0)
        reserved_top = caption_top + 0.08 + INLINE_FRAME_LABEL_LINE_H * max(0, caption_lines - 1)
        frame_bounds = ContentBounds(
            left=min_left - pad,
            right=max_right + pad,
            bottom=min_bottom - pad,
            top=reserved_top,
        )
        elements.append(
            MeasuredElement(
                kind="inline_frame",
                bounds=frame_bounds,
                label=frame.label,
                frame_node_indices=frozenset(frame.node_indices),
                frame_id=frame.frame_id,
            )
        )
        caption_top = max_top + pad + INLINE_FRAME_LABEL_GAP
        for line_index, line in enumerate(label_lines):
            caption = measure_text_bounds(
                ax,
                line,
                min_left + 0.02,
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
                    min_left + 0.02,
                    caption_top - 0.11 - line_index * 0.11,
                    fontsize=5.6,
                    ha="left",
                    va="bottom",
                    fontweight="normal",
                )
                elements.append(
                    MeasuredElement(kind="frame_sublabel", bounds=sub, label=line, frame_id=frame.frame_id)
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
            report.invisible.append(f"text overflows tile: {element.label}")

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


def resolve_measured_overlaps(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    top_y: float | None = None,
    min_gap: float = DEFAULT_MIN_GAP,
) -> None:
    """Push layout tiles apart after measured resize, optionally restacking layers."""
    from visualizer.computation_graph import (
        _assign_layered_vertical_positions,
        _order_fanout_branch_positions,
        _topological_layers,
    )

    if top_y is not None:
        layers = _topological_layers(graph)
        _assign_layered_vertical_positions(positions, layers, top_y=top_y)
        _order_fanout_branch_positions(positions)
    _resolve_layout_overlaps(
        positions,
        graph,
        min_horizontal_gap=min_gap,
        min_vertical_gap=min_gap * 2,
    )


def _frame_member_indices(graph: ComputationGraph) -> dict[int, str]:
    """Map node index -> inline frame id."""
    members: dict[int, str] = {}
    for frame in graph.inline_frames:
        for index in frame.node_indices:
            members[index] = frame.frame_id
    return members


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
    """Raise the synthetic input tile so it clears inline-frame caption bands in its column."""
    input_pos = next((pos for pos in positions if pos.spec.synthetic == SYNTHETIC_INPUT), None)
    if input_pos is None:
        return
    input_left = input_pos.cx - input_pos.width / 2
    input_right = input_pos.cx + input_pos.width / 2
    caption_tops = [
        element.bounds.top
        for element in elements
        if element.kind in {"inline_frame", "frame_label", "frame_sublabel"}
        and element.bounds.right + min_gap > input_left
        and element.bounds.left - min_gap < input_right
    ]
    if not caption_tops:
        return
    max_caption_top = max(caption_tops)
    min_bottom = max_caption_top + min_gap
    if input_pos.bottom < min_bottom:
        input_pos.top_y += min_bottom - input_pos.bottom


def _align_and_stack_inline_frames(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float | None = None,
) -> None:
    """Keep straight-line inline frames vertically stacked on one column."""
    from visualizer.computation_graph import _align_merge_nodes, repack_inline_frame_columns

    stack_inline_frame_positions(positions, graph, min_gap=min_gap)
    _align_merge_nodes(positions, graph)
    repack_inline_frame_columns(positions, graph)
    _layout_fork_join_branches(positions, graph)


def _place_layout_zones(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    cx: float,
    min_gap: float,
    max_right: float | None = None,
    min_left: float | None = None,
) -> None:
    """Separate fan-out, main path, and side branches into non-overlapping columns."""
    fanout, main, side, input_hidden = _classify_layout_zones(positions, graph)
    from visualizer.sizing import min_horizontal_block_gap

    zone_gap = max(min_gap * 2, min_horizontal_block_gap())
    all_indices = fanout | main | side | input_hidden

    main_bounds = _cluster_horizontal_bounds(positions, main | input_hidden)
    if main_bounds is None:
        return
    main_left, main_right = main_bounds
    if min_left is not None:
        _shift_node_indices(positions, main | input_hidden, -main_left)
    else:
        main_center = (main_left + main_right) / 2
        _shift_node_indices(positions, main | input_hidden, cx - main_center)

    main_bounds = _cluster_horizontal_bounds(positions, main | input_hidden)
    if main_bounds is None:
        return
    main_left, main_right = main_bounds

    fan_bounds = _cluster_horizontal_bounds(positions, fanout)
    if fan_bounds is not None:
        _fan_left, fan_right = fan_bounds
        _shift_node_indices(positions, fanout, main_left - zone_gap - fan_right)

    side_bounds = _cluster_horizontal_bounds(positions, side)
    if side_bounds is not None:
        side_left, side_right = side_bounds
        target_side_left = main_right + zone_gap
        _shift_node_indices(positions, side, target_side_left - side_left)

    content_left, content_right = _content_horizontal_extent_from_positions(positions, graph)
    if min_left is not None:
        _shift_node_indices(positions, all_indices, min_left - content_left)
        content_left, content_right = _content_horizontal_extent_from_positions(positions, graph)

    if max_right is None:
        return
    if content_right <= max_right:
        return
    overflow = content_right - max_right
    _shift_node_indices(positions, all_indices, -overflow)
    content_left, _content_right = _content_horizontal_extent_from_positions(positions, graph)
    target_min_left = min_left if min_left is not None else content_left
    if content_left < target_min_left:
        _shift_node_indices(positions, all_indices, target_min_left - content_left)


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

    for caption in captions:
        caption_frame_id = caption.frame_id
        owner = frame_bounds.get(caption_frame_id or "")
        if owner is None:
            continue
        for element in elements:
            if element.kind not in {"box", "combine", "inline_frame"}:
                continue
            if element.frame_id and element.frame_id == caption_frame_id:
                continue
            if (
                element.node_index is not None
                and frame_members.get(element.node_index) == caption_frame_id
            ):
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
                    _shift_node_indices(positions, indices, horizontal)
                vertical = element.bounds.top + min_gap - caption.bounds.bottom
                if vertical > 0:
                    for index in indices:
                        positions[index].top_y -= vertical


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
    _align_and_stack_inline_frames(positions, graph)
    _place_layout_zones(
        positions,
        graph,
        cx=cx,
        min_gap=min_gap,
        max_right=max_right,
        min_left=min_left,
    )
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
        if box_index not in parallel_indices:
            continue
        pos = positions[box_index]
        if pos.spec.synthetic in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}:
            continue
        for frame in frames:
            if box_index in frame.frame_node_indices:
                continue
            if not box.bounds.overlaps(frame.bounds, min_gap=min_gap):
                continue
            if _fanout_branch_index(pos.spec) is not None:
                shift = frame.bounds.left - min_gap - box.bounds.right
                if shift < 0:
                    pos.cx += shift
            else:
                shift = frame.bounds.right + min_gap - box.bounds.left
                if shift > 0:
                    pos.cx += shift


def _nudge_apart_remaining_tiles(
    positions: list[LayoutPosition],
    elements: list[MeasuredElement],
    graph: ComputationGraph,
    *,
    min_gap: float,
) -> None:
    """Last-resort nudge for tile pairs that still intersect."""
    frame_members = _frame_member_indices(graph)
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
        box_frame = frame_members.get(box_index)
        for frame in frames:
            if box_frame and frame.frame_id == box_frame:
                continue
            if box_index in frame.frame_node_indices:
                continue
            if not box.bounds.overlaps(frame.bounds, min_gap=min_gap):
                continue
            pos = positions[box_index]
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
                positions[right_index_id].cx += overlap


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
    import matplotlib.pyplot as plt

    measure_fig, measure_ax = plt.subplots(figsize=(MEASURE_CANVAS_WIDTH, 13))
    measure_fig.canvas.draw()
    try:
        measure_graph_node_sizes(measure_ax, graph, input_sublabel=input_sublabel)
        positions, _ = layout_computation_graph(
            graph,
            cx=cx,
            top_y=10.0,
            block_w=layout_block_w,
            block_h=_estimate_graph_height(graph),
            content_left=min_left,
        )
        finalize_detail_layout(
            measure_ax,
            graph,
            positions,
            input_sublabel=input_sublabel,
            cx=cx,
            top_y=10.0,
            detail_fill=fill,
            min_left=min_left,
            forbidden_regions=None,
        )
    except LayoutValidationError:
        pass
    finally:
        from visualizer.render import _detail_content_bounds

        frame_left, frame_right, _frame_bottom, _frame_top = _detail_content_bounds(positions)
        width = max(1.2, frame_right - frame_left + 0.05)
        plt.close(measure_fig)
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
    if min_left is not None and content_left + shift < min_left:
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
    return _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)


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
                top_y=top_y,
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
                top_y=top_y,
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
            top_y=top_y,
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
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    _separate_parallel_tiles_from_inline_frames(
        positions,
        elements,
        graph,
        min_gap=VALIDATE_MIN_GAP,
    )
    validate_render_layout(elements, min_gap=VALIDATE_MIN_GAP, forbidden_regions=forbidden).raise_if_invalid()
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
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    validate_render_layout(elements, min_gap=VALIDATE_MIN_GAP, forbidden_regions=forbidden).raise_if_invalid()
    return plan
