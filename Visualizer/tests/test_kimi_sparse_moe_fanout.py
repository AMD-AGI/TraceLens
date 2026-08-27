"""Tests for KimiSparseMoeBlock connector routing and runtime layout checks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from visualizer.ast_analyze import analyze_source
from visualizer.basic_ops import BasicOpFilter
from visualizer.block_tree import build_block_node
from visualizer.computation_graph import (
    SYNTHETIC_INPUT,
    _estimate_graph_height,
    build_computation_graph,
    layout_computation_graph,
    measure_graph_node_sizes,
)
from visualizer.render import (
    COLORS,
    PARALLEL_CONNECTOR_COORD_EPS,
    PIPELINE_MERGE_BUS_BELOW_FRAME_GAP,
    _RenderAnchor,
    _anchors_from_detail_plan,
    _build_detail_draw_plan,
    _collect_detail_link_paths,
    _compute_detail_connector_buses,
    _connector_fanout_branch_tee_y,
    _connector_source_bottom_exit_y,
    _fanout_links_excluding_bypasses,
)
from visualizer.render_validate import finalize_detail_layout

_KIMI_CODE = (
    Path.home()
    / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
)
_KIMI_ROUTER_CONFIG = {
    "hidden_size": 7168,
    "num_experts": 896,
    "num_experts_per_token": 16,
    "num_expert_group": 1,
    "topk_group": 1,
    "moe_router_activation_func": "sigmoid",
    "moe_renormalize": True,
    "routed_scaling_factor": 1.0,
}


def _kimi_sparse_moe_layout(*, cx: float = 2.6, top_y: float = 10.0):
    if not _KIMI_CODE.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")
    analysis = analyze_source(
        _KIMI_CODE.read_text(),
        filename="modeling_kimi_linear.py",
        config=_KIMI_ROUTER_CONFIG,
    )
    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    moe = build_block_node(
        attr_name="block_sparse_moe",
        class_name="KimiSparseMoeBlock",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    graph = build_computation_graph(moe, basic_ops=basic)
    fig, ax = plt.subplots(figsize=(16, 13))
    measure_graph_node_sizes(ax, graph)
    positions, links = layout_computation_graph(
        graph,
        cx=cx,
        top_y=top_y,
        block_w=8.0,
        block_h=_estimate_graph_height(graph),
        content_left=0.6,
    )
    finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=None,
        cx=cx,
        top_y=top_y,
        detail_fill=COLORS["detail_fill"],
        min_left=0.6,
    )
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
    anchors = _anchors_from_detail_plan(positions, plan)
    incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
    outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for src, tgt in links:
        incoming[tgt].append((src, tgt))
        outgoing[src].append((src, tgt))
    input_index = next(i for i, n in enumerate(graph.nodes) if n.synthetic == SYNTHETIC_INPUT)
    buses = _compute_detail_connector_buses(
        graph,
        positions,
        anchors,
        incoming,
        outgoing,
        plan.label_obstacles,
    )
    link_paths = _collect_detail_link_paths(
        graph=graph,
        links=links,
        positions=positions,
        anchors=anchors,
        incoming=incoming,
        label_obstacles=plan.label_obstacles,
        target_bus=buses[0],
        source_bus=buses[1],
        merge_entry_x=buses[2],
        merge_link_bus=buses[3],
        input_index=input_index,
    )
    return fig, graph, positions, anchors, incoming, outgoing, input_index, buses, link_paths


def _link_by_labels(graph, link_paths, src_label: str, tgt_label: str):
    by_label = {spec.label: index for index, spec in enumerate(graph.nodes)}
    link_key = (by_label[src_label], by_label[tgt_label])
    return link_key, link_paths[link_key]


def test_moe_input_fanout_departs_vertically():
    fig, graph, _positions, anchors, _incoming, outgoing, input_index, buses, link_paths = (
        _kimi_sparse_moe_layout()
    )
    try:
        source_bus = buses[1]
        tee_y = source_bus[input_index]
        source = anchors[input_index]
        y_exit = _connector_source_bottom_exit_y(source)
        main_links = _fanout_links_excluding_bypasses(graph, outgoing[input_index])
        for link in main_links:
            points = link_paths[link]
            assert abs(points[0][0] - points[1][0]) < PARALLEL_CONNECTOR_COORD_EPS, link
            assert abs(points[0][1] - y_exit) < PARALLEL_CONNECTOR_COORD_EPS, link
            branch_tee = _connector_fanout_branch_tee_y(points, source=source)
            if branch_tee is not None:
                assert abs(branch_tee - tee_y) <= PARALLEL_CONNECTOR_COORD_EPS + 1e-9, link
            assert abs(points[-1][1] - anchors[link[1]].top) < PARALLEL_CONNECTOR_COORD_EPS
    finally:
        plt.close(fig)


def test_sigmoid_uses_one_exit_then_splits_to_top_entries():
    fig, graph, _positions, anchors, _incoming, outgoing, _input_index, buses, link_paths = (
        _kimi_sparse_moe_layout()
    )
    try:
        sigmoid = next(
            index
            for index, spec in enumerate(graph.nodes)
            if spec.label == "Sigmoid" and spec.block is not None
        )
        links = outgoing[sigmoid]
        add_link = next(link for link in links if graph.nodes[link[1]].label == "Add")
        gather_link = next(link for link in links if graph.nodes[link[1]].label == "Gather")

        add_points = link_paths[add_link]
        gather_points = link_paths[gather_link]
        assert add_points[0] == gather_points[0]
        assert add_points[0] == (anchors[sigmoid].cx, anchors[sigmoid].bottom)
        assert gather_points[0] == (anchors[sigmoid].cx, anchors[sigmoid].bottom)

        from visualizer.render import _connector_target_top_entry_y

        for link, points in ((add_link, add_points), (gather_link, gather_points)):
            target = anchors[link[1]]
            assert abs(points[-1][1] - _connector_target_top_entry_y(target)) < PARALLEL_CONNECTOR_COORD_EPS
    finally:
        plt.close(fig)


def test_moe_shared_experts_link_uses_top_entry():
    fig, graph, _positions, anchors, _incoming, _outgoing, input_index, buses, link_paths = (
        _kimi_sparse_moe_layout()
    )
    try:
        shared_entry = next(
            index
            for frame in graph.inline_frames
            if frame.frame_id == "shared_experts"
            for index in frame.node_indices
            if index == frame.node_indices[0]
        )
        link = (input_index, shared_entry)
        tee_y = buses[1][input_index]
        source = anchors[input_index]
        points = link_paths[link]
        branch_tee = _connector_fanout_branch_tee_y(points, source=source)
        if branch_tee is not None:
            assert abs(branch_tee - tee_y) <= PARALLEL_CONNECTOR_COORD_EPS + 1e-9
        assert abs(points[0][0] - points[1][0]) < PARALLEL_CONNECTOR_COORD_EPS
        assert abs(points[-1][1] - anchors[shared_entry].top) < PARALLEL_CONNECTOR_COORD_EPS
    finally:
        plt.close(fig)


def test_moe_shared_experts_to_plus_enters_target_top():
    fig, graph, positions, anchors, _incoming, _outgoing, _input_index, _buses, link_paths = (
        _kimi_sparse_moe_layout()
    )
    try:
        from visualizer.render import (
            CONNECTOR_EXIT_STUB,
            CONNECTOR_OBSTACLE_MARGIN,
            _inline_frame_draw_bounds,
        )

        shared_members = next(
            frame.node_indices
            for frame in graph.inline_frames
            if frame.frame_id == "shared_experts"
        )
        plus_index = next(
            tgt
            for src, tgt in graph.links
            if src in shared_members and graph.nodes[tgt].label == "Add"
        )
        shared_tail = next(
            index
            for frame in graph.inline_frames
            if frame.frame_id == "shared_experts"
            for index in reversed(frame.node_indices)
            if any(src == index and tgt == plus_index for src, tgt in graph.links)
        )
        link = (shared_tail, plus_index)
        points = link_paths[link]
        assert anchors[plus_index].left <= points[-1][0] <= anchors[plus_index].right
        assert abs(points[-1][1] - anchors[plus_index].top) < PARALLEL_CONNECTOR_COORD_EPS
        frame_bounds = _inline_frame_draw_bounds(
            next(frame for frame in graph.inline_frames if frame.frame_id == "shared_experts"),
            positions,
            graph,
        )
        horizontals = [
            (y1, x1, x2)
            for (x1, y1), (x2, y2) in zip(points, points[1:])
            if abs(y1 - y2) < PARALLEL_CONNECTOR_COORD_EPS and abs(x1 - x2) > 0.06
        ]
        assert horizontals, "shared experts feed should include a below-frame horizontal leg"
        corridor_y = (
            frame_bounds.bottom
            - CONNECTOR_OBSTACLE_MARGIN
            - CONNECTOR_EXIT_STUB
            - PIPELINE_MERGE_BUS_BELOW_FRAME_GAP
        )
        below_frame_horizontals = [
            y for y, _x1, _x2 in horizontals if y <= corridor_y + PARALLEL_CONNECTOR_COORD_EPS
        ]
        assert below_frame_horizontals, (
            "shared experts exit must run below the dashed frame border"
        )
        y_exit = _connector_source_bottom_exit_y(anchors[shared_tail])
        outside_tees = [
            y
            for y, x1, x2 in horizontals
            if abs(y - y_exit) < PARALLEL_CONNECTOR_COORD_EPS and abs(x1 - x2) > 0.06
        ]
        if outside_tees:
            assert len(points) >= 3 and points[2][1] < y_exit - PARALLEL_CONNECTOR_COORD_EPS, (
                "outside-gutter tee must drop below the source before routing to +"
            )
    finally:
        plt.close(fig)


def test_moe_route_scaling_uses_direct_merge_bus_to_aggregation():
    fig, graph, positions, anchors, _incoming, _outgoing, _input_index, buses, link_paths = (
        _kimi_sparse_moe_layout()
    )
    try:
        from visualizer.render import (
            _connector_target_top_entry_y,
            _find_connector_inline_frame_overlaps,
            _path_penetrates_attached_boxes,
        )

        route_index = next(
            index
            for index, spec in enumerate(graph.nodes)
            if spec.label == "Multiply"
            and spec.block is not None
            and spec.block.attr_name.startswith("@op_")
        )
        aggregation_index = next(
            index for index, spec in enumerate(graph.nodes) if spec.label == "MoE aggregation"
        )
        link_key = (route_index, aggregation_index)
        points = link_paths[link_key]
        route_scaling = anchors[link_key[0]]
        aggregation = anchors[link_key[1]]
        y_exit = _connector_source_bottom_exit_y(route_scaling)
        assert abs(points[0][1] - y_exit) < PARALLEL_CONNECTOR_COORD_EPS
        assert abs(points[0][0] - points[1][0]) < PARALLEL_CONNECTOR_COORD_EPS, (
            "Route scaling must drop vertically before joining the merge bus"
        )
        horizontals = [
            (y1, x1, x2)
            for (x1, y1), (x2, y2) in zip(points, points[1:])
            if abs(y1 - y2) < PARALLEL_CONNECTOR_COORD_EPS and abs(x1 - x2) > 0.06
        ]
        assert horizontals, "Final router Multiply must route horizontally to aggregation"
        assert all(x >= route_scaling.cx - PARALLEL_CONNECTOR_COORD_EPS for x, _y in points), (
            "Route scaling must not detour left of its column"
        )
        assert not _path_penetrates_attached_boxes(points, route_scaling, aggregation)
        assert not _find_connector_inline_frame_overlaps(
            link_paths,
            graph=graph,
            positions=positions,
        )
        entry_x = buses[2][link_key]
        assert abs(points[-1][0] - entry_x) < PARALLEL_CONNECTOR_COORD_EPS
        assert abs(points[-1][1] - _connector_target_top_entry_y(aggregation)) < PARALLEL_CONNECTOR_COORD_EPS

        down_index = next(
            index
            for index, spec in enumerate(graph.nodes)
            if spec.block is not None and spec.block.attr_name == "routed_expert_down_proj"
        )
        assert (down_index, aggregation_index) in link_paths
    finally:
        plt.close(fig)


def test_moe_aggregation_is_regular_block_with_dual_top_entry_ports():
    from visualizer.render import (
        CONNECTOR_ATTACHED_BOX_MARGIN,
        CONNECTOR_EXIT_STUB,
        CONNECTOR_OBSTACLE_MARGIN,
        PIPELINE_MERGE_BUS_BELOW_FRAME_GAP,
        TOP_ENTRY_PORT_GAP,
        _connector_target_top_entry_y,
        _inline_frame_draw_bounds,
    )

    fig, graph, positions, anchors, _incoming, _outgoing, _input_index, buses, link_paths = (
        _kimi_sparse_moe_layout()
    )
    try:
        by_label = {spec.label: index for index, spec in enumerate(graph.nodes)}
        agg_index = by_label["MoE aggregation"]
        route_index = next(
            index
            for index, spec in enumerate(graph.nodes)
            if spec.label == "Multiply"
            and spec.block is not None
            and spec.block.attr_name.startswith("@op_")
        )
        down_index = next(
            index
            for index, spec in enumerate(graph.nodes)
            if spec.block and spec.block.attr_name == "routed_expert_down_proj"
        )
        assert graph.nodes[agg_index].synthetic is None
        assert graph.nodes[agg_index].label == "MoE aggregation"

        merge_entry_x = buses[2]
        route_link = (route_index, agg_index)
        down_link = (down_index, agg_index)
        aggregation_entries = {
            link: x for link, x in merge_entry_x.items() if link[1] == agg_index
        }
        assert len(aggregation_entries) == 2
        assert route_link in merge_entry_x and down_link in merge_entry_x
        assert merge_entry_x[route_link] < merge_entry_x[down_link]

        agg_anchor = anchors[agg_index]
        top_y = _connector_target_top_entry_y(agg_anchor)
        for link_key in (route_link, down_link):
            end_x, end_y = link_paths[link_key][-1]
            assert abs(end_y - top_y) < PARALLEL_CONNECTOR_COORD_EPS
            assert abs(end_x - merge_entry_x[link_key]) < PARALLEL_CONNECTOR_COORD_EPS

        assert positions[route_index].cx < positions[down_index].cx
        assert merge_entry_x[route_link] < merge_entry_x[down_link]
        span = merge_entry_x[down_link] - merge_entry_x[route_link]
        assert span >= TOP_ENTRY_PORT_GAP - PARALLEL_CONNECTOR_COORD_EPS
        assert span <= agg_anchor.right - agg_anchor.left

        gate = next(frame for frame in graph.inline_frames if frame.frame_id == "gate")
        gate_bounds = _inline_frame_draw_bounds(gate, positions, graph)
        clearance = (
            CONNECTOR_EXIT_STUB
            + PIPELINE_MERGE_BUS_BELOW_FRAME_GAP
            + CONNECTOR_OBSTACLE_MARGIN
            + CONNECTOR_ATTACHED_BOX_MARGIN
        )
        if agg_anchor.left < gate_bounds.right and agg_anchor.right > gate_bounds.left:
            assert agg_anchor.top <= gate_bounds.bottom - clearance + PARALLEL_CONNECTOR_COORD_EPS
    finally:
        plt.close(fig)


def test_moe_connectors_do_not_run_along_inline_frame_borders():
    """No horizontal run may be drawn on a dotted border, which reads as one line."""
    from visualizer.render import (
        CONNECTOR_OBSTACLE_MARGIN,
        _inline_frame_draw_bounds,
    )

    fig, graph, positions, _anchors, _incoming, _outgoing, _input_index, _buses, link_paths = (
        _kimi_sparse_moe_layout()
    )
    try:
        assert graph.inline_frames
        for frame in graph.inline_frames:
            bounds = _inline_frame_draw_bounds(frame, positions, graph)
            for link_key, points in link_paths.items():
                for (x1, y1), (x2, y2) in zip(points, points[1:]):
                    if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS or abs(x1 - x2) <= 0.06:
                        continue
                    if min(x1, x2) >= bounds.right or max(x1, x2) <= bounds.left:
                        continue
                    for edge_name, edge in (("top", bounds.top), ("bottom", bounds.bottom)):
                        assert abs(y1 - edge) >= CONNECTOR_OBSTACLE_MARGIN - 1e-9, (
                            f"{link_key} runs along the {frame.frame_id!r} {edge_name} "
                            f"border (y={y1:.4f}, border={edge:.4f})"
                        )
    finally:
        plt.close(fig)


def test_moe_up_projection_bypasses_situ_and_aggregation_feeds_do_not_overlap():
    from visualizer.render import (
        CONNECTOR_OBSTACLE_MARGIN,
        _find_connector_path_overlaps,
        _path_hits_obstacles,
    )

    fig, graph, positions, anchors, incoming, outgoing, _input_index, buses, link_paths = (
        _kimi_sparse_moe_layout()
    )
    try:
        situ = next(index for index, node in enumerate(graph.nodes) if node.label == "Situ")
        multiply = next(
            index
            for index, node in enumerate(graph.nodes)
            if node.label == "Multiply"
            and any(
                graph.nodes[source].block
                and graph.nodes[source].block.attr_name == "up_proj"
                for source, target in graph.links
                if target == index
            )
        )
        up_projection = next(
            source
            for source, target in graph.links
            if target == multiply
            and graph.nodes[source].block
            and graph.nodes[source].block.attr_name == "up_proj"
        )
        assert not _path_hits_obstacles(
            link_paths[(up_projection, multiply)],
            [anchors[situ]],
            margin=CONNECTOR_OBSTACLE_MARGIN,
        )

        aggregation = next(
            index for index, node in enumerate(graph.nodes) if node.label == "MoE aggregation"
        )
        aggregation_links = [link for link in link_paths if link[1] == aggregation]
        assert len(aggregation_links) == 2
        overlaps = _find_connector_path_overlaps(
            {link: link_paths[link] for link in aggregation_links},
            incoming=incoming,
            outgoing=outgoing,
            target_bus=buses[0],
            source_bus=buses[1],
            merge_link_bus=buses[3],
            anchors=anchors,
            graph=graph,
        )
        assert not overlaps
        assert not any(
            y2 > y1 + PARALLEL_CONNECTOR_COORD_EPS
            for link in aggregation_links
            for (_x1, y1), (_x2, y2) in zip(link_paths[link], link_paths[link][1:])
        )
    finally:
        plt.close(fig)


def test_moe_collect_detail_link_paths_enforces_runtime_invariants():
    fig, graph, positions, _anchors, _incoming, _outgoing, _input_index, _buses, link_paths = (
        _kimi_sparse_moe_layout()
    )
    try:
        assert link_paths
        from visualizer.render import _find_connector_inline_frame_overlaps

        assert not _find_connector_inline_frame_overlaps(
            link_paths,
            graph=graph,
            positions=positions,
        )
    finally:
        plt.close(fig)
