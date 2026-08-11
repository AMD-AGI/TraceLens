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
    _center_positions_horizontally,
    _resolve_layout_overlaps,
)
from visualizer.text_measure import ContentBounds, box_bounds_at, box_label_size, floating_port_label_bounds

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from visualizer.render import DetailDrawPlan

DEFAULT_MIN_GAP = 0.02
VALIDATE_MIN_GAP = 0.02
_TILE_KINDS = frozenset({"box", "combine"})


@dataclass
class MeasuredElement:
    """One drawable region used for overlap checks."""

    kind: str
    bounds: ContentBounds
    label: str = ""
    node_index: int | None = None
    frame_node_indices: frozenset[int] = frozenset()


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
            width, height = box_label_size(
                ax,
                spec.label,
                plan.input_sublabel,
                fontsize=7.2,
            )
            pos.width = max(pos.width, width)
            pos.height = max(pos.height, height)
            if draw_index < len(plan.node_draws):
                leaf, _ = plan.node_draws[draw_index]
                leaf.w = pos.width
                leaf.h = pos.height
                leaf.x = pos.cx - pos.width / 2
                leaf.y = pos.top_y - pos.height
                draw_index += 1
            continue
        if spec.synthetic == "@hidden_states":
            width, height = box_label_size(ax, spec.label, None, fontsize=6.5)
            pos.width = max(pos.width, width)
            pos.height = max(pos.height, height)
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
    from visualizer.render import INLINE_FRAME_LABEL_GAP, INLINE_FRAME_PAD
    from visualizer.text_measure import measure_text_bounds

    elements: list[MeasuredElement] = []
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
        elements.append(
            MeasuredElement(
                kind="box",
                bounds=box_bounds_at(pos.cx, pos.top_y, pos.width, pos.height),
                label=leaf.label,
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
        frame_bounds = ContentBounds(
            left=min_left - pad,
            right=max_right + pad,
            bottom=min_bottom - pad,
            top=max_top + pad,
        )
        elements.append(
            MeasuredElement(
                kind="inline_frame",
                bounds=frame_bounds,
                label=frame.label,
                frame_node_indices=frozenset(frame.node_indices),
            )
        )
        caption_top = max_top + pad + INLINE_FRAME_LABEL_GAP
        caption = measure_text_bounds(
            ax,
            frame.label,
            min_left + 0.02,
            caption_top,
            fontsize=6.4,
            ha="left",
            va="bottom",
            fontweight="normal",
        )
        elements.append(MeasuredElement(kind="frame_label", bounds=caption, label=frame.label))
        if frame.sublabel:
            sub = measure_text_bounds(
                ax,
                frame.sublabel,
                min_left + 0.02,
                caption_top - 0.11,
                fontsize=5.6,
                ha="left",
                va="bottom",
                fontweight="normal",
            )
            elements.append(MeasuredElement(kind="frame_sublabel", bounds=sub, label=frame.sublabel))

    return elements


def _is_combine(synthetic: str | None) -> bool:
    from visualizer.computation_graph import SYNTHETIC_COMBINE

    return synthetic == SYNTHETIC_COMBINE


def validate_render_layout(
    elements: list[MeasuredElement],
    *,
    min_gap: float = DEFAULT_MIN_GAP,
) -> LayoutValidationReport:
    """Check that tiles are visible and that solid boxes do not overlap."""
    report = LayoutValidationReport()
    tiles = [element for element in elements if element.kind in _TILE_KINDS]

    for element in elements:
        bounds = element.bounds
        if bounds.width <= 0 or bounds.height <= 0:
            report.invisible.append(f"{element.kind} {element.label!r} has zero-size bounds")

    for left_index, left in enumerate(tiles):
        for right in tiles[left_index + 1 :]:
            if left.node_index is not None and left.node_index == right.node_index:
                continue
            if not left.bounds.overlaps(right.bounds, min_gap=min_gap):
                continue
            report.overlaps.append(
                f"{left.kind} {left.label!r} overlaps {right.kind} {right.label!r} "
                f"(gap<{min_gap:.3f})"
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


def _nudge_apart_remaining_tiles(
    positions: list[LayoutPosition],
    elements: list[MeasuredElement],
    *,
    min_gap: float,
) -> None:
    """Last-resort horizontal nudge for tile pairs that still intersect."""
    tiles = [
        element
        for element in elements
        if element.kind in _TILE_KINDS and element.node_index is not None
    ]
    for left_index, left in enumerate(tiles):
        for right in tiles[left_index + 1 :]:
            if not left.bounds.overlaps(right.bounds, min_gap=min_gap):
                continue
            right_index = right.node_index
            if right_index is None or right_index >= len(positions):
                continue
            overlap = left.bounds.right + min_gap - right.bounds.left
            if overlap > 0:
                positions[right_index].cx += overlap
                delta = overlap
                right.bounds = ContentBounds(
                    left=right.bounds.left + delta,
                    right=right.bounds.right + delta,
                    bottom=right.bounds.bottom,
                    top=right.bounds.top,
                )


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
    max_passes: int = 6,
) -> DetailDrawPlan:
    """
    Pre-render all boxes and labels to measure bounds, re-layout, then validate.

    Must run before drawing connectors or tiles.
    """
    from visualizer.render import COLORS, _build_detail_draw_plan, _resize_input_nodes

    fill = detail_fill or COLORS["detail_fill"]
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)

    for _pass in range(max_passes):
        apply_measured_node_sizes(ax, positions, plan)
        _resize_input_nodes(positions, input_sublabel)
        apply_measured_node_sizes(ax, positions, plan)
        if positions:
            _center_positions_horizontally(positions, cx)
            resolve_measured_overlaps(
                positions,
                graph,
                top_y=top_y,
                min_gap=MIN_HORIZONTAL_BLOCK_GAP,
            )
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
        elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
        report = validate_render_layout(elements, min_gap=VALIDATE_MIN_GAP)
        if report.ok:
            return plan
        _nudge_apart_remaining_tiles(positions, elements, min_gap=VALIDATE_MIN_GAP)
        if positions:
            resolve_measured_overlaps(
                positions,
                graph,
                top_y=top_y,
                min_gap=MIN_HORIZONTAL_BLOCK_GAP,
            )
        plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)

    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=fill)
    validate_render_layout(elements, min_gap=VALIDATE_MIN_GAP).raise_if_invalid()
    return plan
