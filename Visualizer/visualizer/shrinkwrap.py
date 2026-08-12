"""Post-layout shrinkwrap: tighten straight connector corridors without breaking bounds."""

from __future__ import annotations

from visualizer.computation_graph import (
    ComputationGraph,
    LayoutPosition,
    _compact_parallel_feeder_frame_exit_stubs,
    _graph_has_tensor_ports,
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
) -> None:
    """Apply vertical and horizontal shrinkwrap after layout rules have settled."""
    if not positions or not graph.nodes:
        return

    _shrinkwrap_vertical_layout(positions, graph, min_gap=min_gap)
    _shrinkwrap_horizontal_layout(positions, graph)


def _shrinkwrap_vertical_layout(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float,
) -> None:
    """Shorten vertical straight stubs by moving feeder columns toward merge buses."""
    del min_gap
    if _graph_has_tensor_ports(graph):
        _compact_parallel_feeder_frame_exit_stubs(positions, graph)


def _shrinkwrap_horizontal_layout(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Tighten horizontal packing for non-tensor detail graphs."""
    if _graph_has_tensor_ports(graph):
        return
    compact_horizontal_shrink_wrap(positions, graph)


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
    min_gap: float = SHRINKWRAP_MIN_GAP,
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Shorten groups of collinear connector segments while preserving clearance."""
    if not link_paths:
        return link_paths

    shrunk = {key: list(points) for key, points in link_paths.items()}
    for bus_y, targets in _shared_bus_y_groups(shrunk, target_bus, merge_link_bus).items():
        new_y = _max_upward_bus_shift(
            bus_y,
            shrunk,
            anchors=anchors,
            targets=targets,
            min_gap=min_gap,
        )
        if new_y is None or abs(new_y - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS:
            continue
        trial = {key: list(points) for key, points in shrunk.items()}
        _apply_bus_y_shift(trial, bus_y, new_y)
        validated = _validate_shrunk_paths(
            trial,
            graph=graph,
            anchors=anchors,
            label_obstacles=label_obstacles,
            positions=positions,
        )
        if validated is not None:
            shrunk = validated
            _apply_bus_y_to_maps(target_bus, merge_link_bus, bus_y, new_y)

    shrunk = _compact_horizontal_bus_spans(
        shrunk,
        graph=graph,
        anchors=anchors,
        label_obstacles=label_obstacles,
        positions=positions,
        min_gap=min_gap,
    )
    return shrunk


def _apply_bus_y_to_maps(
    target_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    old_y: float,
    new_y: float,
) -> None:
    for tgt, bus_y in list(target_bus.items()):
        if abs(bus_y - old_y) <= PARALLEL_CONNECTOR_COORD_EPS:
            target_bus[tgt] = new_y
    for link_key, bus_y in list(merge_link_bus.items()):
        if abs(bus_y - old_y) <= PARALLEL_CONNECTOR_COORD_EPS:
            merge_link_bus[link_key] = new_y


def _shared_bus_y_groups(
    link_paths: dict[tuple[int, int], list[tuple[float, float]]],
    target_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
) -> dict[float, set[int]]:
    buses: dict[float, set[int]] = {}
    for tgt, bus_y in target_bus.items():
        buses.setdefault(round(bus_y, 4), set()).add(tgt)
    for (_src, tgt), bus_y in merge_link_bus.items():
        buses.setdefault(round(bus_y, 4), set()).add(tgt)
    return buses


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
                - CONNECTOR_EXIT_STUB
                - min_gap,
            )
        for index in range(len(points) - 1):
            x1, y1 = points[index]
            x2, y2 = points[index + 1]
            if abs(x1 - x2) <= PARALLEL_CONNECTOR_COORD_EPS and abs(y2 - bus_y) <= PARALLEL_CONNECTOR_COORD_EPS:
                if y1 > bus_y + PARALLEL_CONNECTOR_COORD_EPS:
                    upper_bound = max(
                        upper_bound,
                        y1 - CONNECTOR_EXIT_STUB - min_gap,
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
                            prev_y - CONNECTOR_EXIT_STUB - min_gap,
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
    return link_paths
