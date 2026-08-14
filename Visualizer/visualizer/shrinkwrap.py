"""Post-layout shrinkwrap: tighten layout without rerouting connectors.

Shrinkwrap performs in-place edits only. It never calls connector routing
(``_connector_points_for_link``, ``_reroute_connector_path_clearing_blocks``,
or similar). Two passes run at different stages:

1. ``shrinkwrap_detail_layout`` — before connectors exist; moves block tiles in
   ``positions`` (feeder frame columns, horizontal packing).
2. ``shrinkwrap_detail_link_paths`` — after routing finishes; shortens existing
   polylines (bus Y shifts, horizontal bus trimming) and keeps ``positions``,
   anchors, and bus metadata aligned with the tightened corridors.

Path edits may call ``_ensure_orthogonal_connector_path`` to preserve right
angles; that is coordinate surgery, not rerouting.
"""

from __future__ import annotations

import copy

from visualizer.computation_graph import (
    ComputationGraph,
    LayoutPosition,
    SYNTHETIC_TENSOR,
    _compact_parallel_feeder_frame_exit_stubs,
    _graph_has_tensor_ports,
    _inline_frame_for_tail_node,
    _inline_frame_tail_indices,
    _separate_parallel_merge_horiz_corridors,
    _shift_inline_frame_column_and_ports,
    compact_horizontal_shrink_wrap,
)
from visualizer.render import (
    CONNECTOR_EXIT_STUB,
    CONNECTOR_OBSTACLE_MARGIN,
    PARALLEL_CONNECTOR_COORD_EPS,
    _connector_block_obstacles,
    _connector_min_bus_y_above_target,
    _connector_path_clear_of_blocks,
    _connector_path_violates_inline_frame_bounds,
    _connector_source_bottom_exit_y,
    _ensure_orthogonal_connector_path,
    _path_hits_obstacles,
)

SHRINKWRAP_MIN_GAP = CONNECTOR_OBSTACLE_MARGIN


def shrinkwrap_detail_layout(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float = SHRINKWRAP_MIN_GAP,
    min_left: float | None = None,
) -> None:
    """Apply vertical and horizontal shrinkwrap after layout rules have settled."""
    if not positions or not graph.nodes:
        return

    _shrinkwrap_vertical_layout(positions, graph, min_gap=min_gap)
    _shrinkwrap_horizontal_layout(positions, graph, min_left=min_left)


def _shrinkwrap_vertical_layout(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float,
) -> None:
    """Shorten vertical straight stubs by moving feeder columns toward merge buses."""
    del min_gap
    _compact_parallel_feeder_frame_exit_stubs(positions, graph)
    _separate_parallel_merge_horiz_corridors(positions, graph)


def _shrinkwrap_horizontal_layout(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_left: float | None = None,
) -> None:
    """Tighten horizontal packing and re-anchor content to ``min_left`` when set."""
    if _graph_has_tensor_ports(graph):
        if min_left is not None:
            from visualizer.computation_graph import _align_positions_left

            _align_positions_left(positions, min_left)
        return
    compact_horizontal_shrink_wrap(positions, graph, min_left=min_left)


def shrinkwrap_detail_link_paths(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict,
    label_obstacles: list,
    positions: list,
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    incoming: dict[int, list[tuple[int, int]]] | None = None,
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
    min_gap: float = SHRINKWRAP_MIN_GAP,
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Shorten groups of collinear connector segments while preserving clearance.

    Applies in-place bus Y shifts and horizontal trimming only. When a bus moves,
    feeder inline-frame columns in ``positions`` (and matching anchors / path
    segments above the bus) shift with it so layout state stays consistent.
    """
    if not link_paths:
        return link_paths

    shrunk = {key: list(points) for key, points in link_paths.items()}
    for bus_y, targets in _shared_bus_y_groups(
        shrunk,
        target_bus,
        merge_link_bus,
        source_bus=source_bus,
        outgoing=outgoing,
    ).items():
        if _is_source_fanout_bus_y(bus_y, source_bus, merge_link_bus):
            continue
        if _is_frame_tail_corridor_bus_y(
            bus_y,
            graph=graph,
            positions=positions,
            anchors=anchors,
            merge_link_bus=merge_link_bus,
            link_paths=shrunk,
        ):
            continue
        new_y = _max_upward_bus_shift(
            bus_y,
            shrunk,
            anchors=anchors,
            targets=targets,
            min_gap=min_gap,
        )
        if new_y is None or abs(new_y - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS:
            continue
        trial_paths = {key: list(points) for key, points in shrunk.items()}
        _apply_bus_y_shift(trial_paths, bus_y, new_y)
        trial_positions = [copy.copy(pos) for pos in positions]
        trial_anchors = {index: copy.copy(anchor) for index, anchor in anchors.items()}
        trial_source_bus = dict(source_bus)
        _shift_feeder_layout_for_bus_y(
            trial_positions,
            trial_anchors,
            trial_paths,
            graph=graph,
            old_y=bus_y,
            new_y=new_y,
            targets=targets,
            incoming=incoming,
            source_bus=trial_source_bus,
        )
        validated = _validate_shrunk_paths(
            trial_paths,
            graph=graph,
            anchors=trial_anchors,
            label_obstacles=label_obstacles,
            positions=trial_positions,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=trial_source_bus,
            merge_link_bus=merge_link_bus,
        )
        if validated is not None:
            shrunk = validated
            _apply_bus_y_to_maps(
                target_bus,
                merge_link_bus,
                source_bus,
                bus_y,
                new_y,
            )
            _shift_feeder_layout_for_bus_y(
                positions,
                anchors,
                shrunk,
                graph=graph,
                old_y=bus_y,
                new_y=new_y,
                targets=targets,
                incoming=incoming,
                source_bus=source_bus,
            )

    shrunk = _compact_horizontal_bus_spans(
        shrunk,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
        min_gap=min_gap,
        incoming=incoming,
        outgoing=outgoing,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
    )
    return shrunk


def _feeder_frames_for_bus_targets(
    graph: ComputationGraph,
    targets: set[int],
    incoming: dict[int, list[tuple[int, int]]] | None,
) -> list[tuple[object, int]]:
    """Inline frames whose tail nodes feed merge targets on one shared bus."""
    if incoming is None:
        return []
    frame_tails = _inline_frame_tail_indices(graph)
    frames: list[tuple[object, int]] = []
    seen: set[int] = set()
    for target in targets:
        for src, tgt in incoming.get(target, []):
            if tgt != target or src not in frame_tails:
                continue
            frame = _inline_frame_for_tail_node(graph, src)
            if frame is None:
                continue
            frame_key = id(frame)
            if frame_key in seen:
                continue
            seen.add(frame_key)
            frames.append((frame, src))
    return frames


def _shift_anchor_y(anchor, delta_y: float) -> None:
    """Apply the same vertical delta used by ``_shift_inline_frame_column``."""
    anchor.top -= delta_y
    anchor.bottom -= delta_y


def _shift_feeder_layout_for_bus_y(
    positions: list[LayoutPosition],
    anchors: dict,
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph: ComputationGraph,
    old_y: float,
    new_y: float,
    targets: set[int],
    incoming: dict[int, list[tuple[int, int]]] | None,
    source_bus: dict[int, float],
) -> set[int]:
    """Move feeder columns up with a tightened merge bus and sync anchors/paths."""
    delta_y = old_y - new_y
    lift = new_y - old_y
    if abs(delta_y) <= PARALLEL_CONNECTOR_COORD_EPS:
        return set()

    frames = _feeder_frames_for_bus_targets(graph, targets, incoming)
    if not frames:
        return set()

    shifted_sources: set[int] = set()
    shifted_indices: set[int] = set()
    for frame, tail_src in frames:
        _shift_inline_frame_column_and_ports(positions, graph, frame, delta_y)
        shifted_sources.add(tail_src)
        shifted_indices.update(frame.node_indices)
        for source, target in graph.links:
            if (
                target in frame.node_indices
                and graph.nodes[source].synthetic == SYNTHETIC_TENSOR
            ):
                shifted_indices.add(source)

    for index in shifted_indices:
        anchor = anchors.get(index)
        if anchor is not None:
            _shift_anchor_y(anchor, delta_y)

    for src in shifted_sources:
        if src in source_bus:
            source_bus[src] += lift

    _shift_path_y_above_bus_for_sources(
        link_paths,
        shifted_sources,
        new_y=new_y,
        lift=lift,
    )
    return shifted_sources


def _shift_path_y_above_bus_for_sources(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    sources: set[int],
    *,
    new_y: float,
    lift: float,
) -> None:
    """Lift source-side path segments that sit above a raised merge bus."""
    if not sources or abs(lift) <= PARALLEL_CONNECTOR_COORD_EPS:
        return
    for link_key, points in link_paths.items():
        if link_key[0] not in sources:
            continue
        updated = [
            (x, y + lift if y > new_y + PARALLEL_CONNECTOR_COORD_EPS else y)
            for x, y in points
        ]
        link_paths[link_key] = _ensure_orthogonal_connector_path(updated)


def _apply_bus_y_to_maps(
    target_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    source_bus: dict[int, float],
    old_y: float,
    new_y: float,
) -> None:
    for tgt, bus_y in list(target_bus.items()):
        if abs(bus_y - old_y) <= PARALLEL_CONNECTOR_COORD_EPS:
            target_bus[tgt] = new_y
    for link_key, bus_y in list(merge_link_bus.items()):
        if abs(bus_y - old_y) <= PARALLEL_CONNECTOR_COORD_EPS:
            merge_link_bus[link_key] = new_y
    for src, bus_y in list(source_bus.items()):
        if abs(bus_y - old_y) <= PARALLEL_CONNECTOR_COORD_EPS:
            source_bus[src] = new_y


def _shared_bus_y_groups(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    target_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    *,
    source_bus: dict[int, float] | None = None,
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
) -> dict[float, set[int]]:
    """Group targets by the horizontal bus Y their connectors actually use."""
    del source_bus, outgoing
    buses: dict[float, set[int]] = {}
    for tgt, bus_y in target_bus.items():
        buses.setdefault(round(bus_y, 4), set()).add(tgt)
    for (_src, tgt), bus_y in merge_link_bus.items():
        buses.setdefault(round(bus_y, 4), set()).add(tgt)
    return buses


def _is_frame_tail_corridor_bus_y(
    bus_y: float,
    *,
    graph: ComputationGraph,
    positions: list,
    anchors: dict,
    merge_link_bus: dict[tuple[int, int], float],
    link_paths: dict[tuple[int, int], list[tuple[float, float]]] | None = None,
) -> bool:
    """True when a merge bus sits in the below-frame corridor for a frame tail."""
    from visualizer.render import (
        _inline_frame_below_exit_y,
        _inline_frame_draw_bounds,
    )

    for (src, _tgt), link_bus in merge_link_bus.items():
        if src not in _inline_frame_tail_indices(graph):
            continue
        if abs(link_bus - bus_y) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        frame = _inline_frame_for_tail_node(graph, src)
        source = anchors.get(src)
        if frame is None or source is None:
            continue
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        corridor_y = _inline_frame_below_exit_y(
            bounds,
            source_bottom=source.bottom,
        )
        if abs(link_bus - corridor_y) <= PARALLEL_CONNECTOR_COORD_EPS:
            return True
    if link_paths is None:
        return False
    for (src, _tgt), points in link_paths.items():
        if src not in _inline_frame_tail_indices(graph):
            continue
        frame = _inline_frame_for_tail_node(graph, src)
        source = anchors.get(src)
        if frame is None or source is None:
            continue
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        corridor_y = _inline_frame_below_exit_y(
            bounds,
            source_bottom=source.bottom,
        )
        if not any(
            abs(y - corridor_y) <= PARALLEL_CONNECTOR_COORD_EPS for _x, y in points
        ):
            continue
        if any(abs(y - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS for _x, y in points):
            return True
    return False


def _is_source_fanout_bus_y(
    bus_y: float,
    source_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
) -> bool:
    """True for per-leg fan-out buses that should shrink toward targets, not the source."""
    if not source_bus:
        return False
    for (src, _tgt), link_bus in merge_link_bus.items():
        if src in source_bus and abs(link_bus - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS:
            return True
    return False


def _max_upward_bus_shift(
    bus_y: float,
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    anchors: dict,
    targets: set[int],
    min_gap: float,
) -> float | None:
    lower_bound = bus_y
    for tgt in targets:
        target = anchors.get(tgt)
        if target is None:
            continue
        lower_bound = max(
            lower_bound,
            _connector_min_bus_y_above_target(target, gap=min_gap),
        )

    upper_bound = bus_y
    for link_key, points in link_paths.items():
        if not _path_uses_bus_y(points, bus_y):
            continue
        source = anchors.get(link_key[0])
        if source is not None:
            upper_bound = max(
                upper_bound,
                _connector_source_bottom_exit_y(source, gap=min_gap)
                - CONNECTOR_EXIT_STUB,
            )
        for index in range(len(points) - 1):
            x1, y1 = points[index]
            x2, y2 = points[index + 1]
            if abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS and abs(y2 - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS:
                if y1 > bus_y + PARALLEL_CONNECTOR_COORD_EPS:
                    upper_bound = max(
                        upper_bound,
                        y1 - CONNECTOR_EXIT_STUB,
                    )
            if abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS and abs(y1 - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS:
                if index > 0:
                    prev_x, prev_y = points[index - 1]
                    if (
                        abs(prev_x - x1) <= PARALLEL_CONNECTOR_COORD_EPS
                        and prev_y > bus_y + PARALLEL_CONNECTOR_COORD_EPS
                    ):
                        upper_bound = max(
                            upper_bound,
                            prev_y - CONNECTOR_EXIT_STUB,
                        )

    if upper_bound <= bus_y + PARALLEL_CONNECTOR_COORD_EPS:
        return None
    if upper_bound < lower_bound - PARALLEL_CONNECTOR_COORD_EPS:
        return None
    return upper_bound


def _path_uses_bus_y(points: list[tuple[float, float]], bus_y: float) -> bool:
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if abs(x1 - x2) <= 0.06:
            continue
        if abs(y1 - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS:
            return True
    return False


def _apply_bus_y_shift(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    old_y: float,
    new_y: float,
) -> None:
    for link_key, points in link_paths.items():
        link_paths[link_key] = _ensure_orthogonal_connector_path(
            [
                (x, new_y if abs(y - old_y) <= PARALLEL_CONNECTOR_COORD_EPS else y)
                for x, y in points
            ]
        )


def _compact_horizontal_bus_spans(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict,
    label_obstacles: list,
    positions: list,
    min_gap: float,
    incoming: dict[int, list[tuple[int, int]]] | None = None,
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
    target_bus: dict[int, float] | None = None,
    source_bus: dict[int, float] | None = None,
    merge_link_bus: dict[tuple[int, int], float] | None = None,
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Trim shared horizontal bus segments to the span their feeders require."""
    buses: dict[float, list[tuple[float, float]]] = {}
    for points in link_paths.values():
        for index in range(len(points) - 1):
            x1, y1 = points[index]
            x2, y2 = points[index + 1]
            if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS:
                continue
            if abs(x1 - x2) <= 0.06:
                continue
            key = round(y1, 4)
            buses.setdefault(key, []).extend([x1, x2])

    if not buses:
        return link_paths

    trimmed = {key: list(points) for key, points in link_paths.items()}
    for bus_y, xs in buses.items():
        if len(xs) < 4:
            continue
        needed_left = min(xs)
        needed_right = max(xs)
        trial = {key: list(points) for key, points in trimmed.items()}
        changed = False
        for link_key, points in trial.items():
            new_points: list[tuple[float, float]] = []
            for index, (x, y) in enumerate(points):
                if abs(y - bus_y) > PARALLEL_CONNECTOR_COORD_EPS:
                    new_points.append((x, y))
                    continue
                if index > 0 and abs(points[index - 1][1] - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS:
                    prev_x = new_points[-1][0]
                    if abs(prev_x - x) > 0.06 and abs(points[index - 1][1] - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS:
                        clamped_x = max(needed_left, min(needed_right, x))
                        if abs(clamped_x - x) > PARALLEL_CONNECTOR_COORD_EPS:
                            changed = True
                        new_points.append((clamped_x, y))
                        continue
                new_points.append((x, y))
            trial[link_key] = _ensure_orthogonal_connector_path(new_points)
        if not changed:
            continue
        validated = _validate_shrunk_paths(
            trial,
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
        if validated is not None:
            trimmed = validated
    return trimmed


def _validate_shrunk_paths(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    graph,
    anchors: dict,
    label_obstacles: list,
    positions: list,
    incoming: dict[int, list[tuple[int, int]]] | None = None,
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
    target_bus: dict[int, float] | None = None,
    source_bus: dict[int, float] | None = None,
    merge_link_bus: dict[tuple[int, int], float] | None = None,
) -> dict[tuple[int, int], list[tuple[float, float]]] | None:
    for link_key, points in link_paths.items():
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
        if not _connector_path_clear_of_blocks(
            points,
            source=source,
            target=target,
            obstacles=obstacles,
        ):
            return None
        if _path_hits_obstacles(
            points,
            obstacles,
            margin=CONNECTOR_OBSTACLE_MARGIN,
        ):
            return None
        if _connector_path_violates_inline_frame_bounds(
            points,
            graph,
            positions,
            src=src,
            tgt=tgt,
        ):
            return None
    if incoming is not None and outgoing is not None:
        from visualizer.render import _find_connector_path_overlaps

        if _find_connector_path_overlaps(
            link_paths,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus or {},
            source_bus=source_bus or {},
            merge_link_bus=merge_link_bus or {},
            anchors=anchors,
            graph=graph,
        ):
            return None
    return link_paths
