"""Per-link connector path routing helpers (extracted from render.py).

Used by ``visualizer.elk_routing`` for fan-out, merge-bus, and frame-entry links.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visualizer.render import _RenderAnchor


def _shared_merge_entry_x(
    target: _RenderAnchor,
    link_key: tuple[int, int],
    merge_entry_x: dict[tuple[int, int], float],
    target_bus: dict[int, float],
    *,
    source: _RenderAnchor | None = None,
    source_bus: dict[int, float] | None = None,
) -> float:
    """Targets with a shared merge bus usually enter at center.

    Fan-out legs that branch sideways onto another target keep their spread port.
    """
    if link_key[1] not in target_bus:
        return merge_entry_x.get(link_key, target.cx)
    if (
        source is not None
        and source_bus is not None
        and link_key[0] in source_bus
        and abs(source.cx - target.cx) >= 0.08
    ):
        return merge_entry_x.get(link_key, target.cx)
    return target.cx

def connector_points_for_link(
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
    from visualizer.render import (
        CONNECTOR_EXIT_STUB,
        CONNECTOR_OBSTACLE_MARGIN,
        PARALLEL_CONNECTOR_COORD_EPS,
        SHARED_SOURCE_BUS_MIN_LINKS,
        SYNTHETIC_TENSOR,
        _backward_top_entry_gutter_route,
        _connector_block_obstacles,
        _connector_min_bus_y_above_target,
        _connector_path_clear_of_blocks,
        _connector_path_has_block_edge_horizontal_jog,
        _connector_source_bottom_exit_y,
        _connector_target_top_entry_y,
        _effective_fanout_spine_tee_y,
        _ensure_orthogonal_connector_path,
        _fanout_leg_routing_bus_y,
        _fanout_links_excluding_bypasses,
        _fanout_source_bus_y,
        _fanout_split_branch_gutter_route,
        _fanout_tee_then_entry_column_points,
        _frame_for_tail_node,
        _frame_tail_merge_entry_connector_points,
        _frame_tail_routing_corridor_y,
        _inline_frame_bypass_links,
        _inline_frame_draw_bounds,
        _inline_frame_for_nodes,
        _inline_frame_top_member_route_y,
        _inline_skip_top_entry_connector_points,
        _merge_bus_y_clearing_same_column_feeds,
        _min_bus_y_clearing_horizontal_corridor,
        _orthogonal_path,
        _outside_to_inline_frame_top_member_route,
        _path_hits_obstacles,
        _path_horizontal_segments_overlap_bounds,
        _path_penetrates_attached_boxes,
        _pipeline_frame_exit_connector_points,
        _pipeline_frame_exit_x,
        _requires_shared_input_source_bus,
        _right_bypass_x_clearing_horizontal_segment,
        _right_bypass_x_clearing_vertical_segment,
        _same_column_side_gutter_detour,
        _same_column_spread_top_entry_connector_points,
        _same_column_straight_connector_points,
        _shared_merge_bus_connector_points,
        _source_fanout_splits_before_target_bus,
        _tee_branch_avoiding_vertical_obstacles,
        _tensor_port_connector_points,
        _vertical_segment_crosses_anchor,
    )

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
    spine_tee_y = _effective_fanout_spine_tee_y(
        link_key,
        fanout_tee_y=fanout_tee_y,
        merge_link_bus=merge_link_bus,
    )
    if spine_tee_y is None:
        prefer_tee_branch = False
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

        if spine_tee_y is not None:
            entry_x = _shared_merge_entry_x(target, link_key, merge_entry_x, target_bus, source=source, source_bus=source_bus)
            return _shared_merge_bus_connector_points(
                source,
                target,
                entry_x,
                target_bus[tgt],
                route_obstacles,
                gap=gap,
                spine_tee_y=spine_tee_y,
                graph=graph,
                positions=positions,
                link_key=link_key,
                prefer_tee_branch=prefer_tee_branch,
            )
        is_frame_tail = (
            _graph_has_tensor_ports(graph) and src in _inline_frame_tail_indices(graph)
        )
        if not is_frame_tail:
            entry_x = _shared_merge_entry_x(target, link_key, merge_entry_x, target_bus, source=source, source_bus=source_bus)
            return _shared_merge_bus_connector_points(
                source,
                target,
                entry_x,
                target_bus[tgt],
                route_obstacles,
                gap=gap,
                spine_tee_y=spine_tee_y,
                graph=graph,
                positions=positions,
                link_key=link_key,
                prefer_tee_branch=prefer_tee_branch,
            )
    if positions[src].spec.synthetic == SYNTHETIC_TENSOR:
        if tgt in target_bus:
            entry_x = _shared_merge_entry_x(target, link_key, merge_entry_x, target_bus, source=source, source_bus=source_bus)
            return _shared_merge_bus_connector_points(
                source,
                target,
                entry_x,
                target_bus[tgt],
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
                    pipeline_bus_y = (
                        target_bus[tgt]
                        if tgt in target_bus
                        else merge_link_bus.get(link_key, bus_y)
                    )
                    return _pipeline_frame_exit_connector_points(
                        source,
                        target,
                        exit_x=exit_x,
                        bus_y=pipeline_bus_y,
                        frame_bounds=draw_bounds,
                        gap=gap,
                        obstacles=route_obstacles,
                        entry_x=_shared_merge_entry_x(
                            target, link_key, merge_entry_x, target_bus,
                            source=source, source_bus=source_bus,
                        ),
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
            _shared_merge_entry_x(target, link_key, merge_entry_x, target_bus, source=source, source_bus=source_bus),
            target_bus[tgt],
            route_obstacles,
            gap=gap,
            spine_tee_y=spine_tee_y,
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
        elif (
            fanout_tee_y is not None
            and tgt not in target_bus
            and link_key not in merge_link_bus
        ):
            spread_bus_y = fanout_tee_y
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
            spine_tee_y=spine_tee_y,
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
            spine_tee_y=spine_tee_y,
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

