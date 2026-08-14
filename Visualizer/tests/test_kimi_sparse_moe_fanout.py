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
    _side_entry_combine_entry_x,
)
from visualizer.render_validate import finalize_detail_layout

_KIMI_CODE = (
    Path.home()
    / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
)


def _kimi_sparse_moe_layout(*, cx: float = 2.6, top_y: float = 10.0):
    if not _KIMI_CODE.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")
    analysis = analyze_source(_KIMI_CODE.read_text(), filename="modeling_kimi_linear.py")
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
            assert branch_tee is not None, link
            assert abs(branch_tee - tee_y) < PARALLEL_CONNECTOR_COORD_EPS, link
    finally:
        plt.close(fig)


def test_moe_shared_experts_link_uses_source_bus_tee():
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
        assert branch_tee is not None
        assert abs(branch_tee - tee_y) < PARALLEL_CONNECTOR_COORD_EPS
        assert abs(points[0][0] - points[1][0]) < PARALLEL_CONNECTOR_COORD_EPS
    finally:
        plt.close(fig)


def test_moe_shared_experts_to_plus_enters_from_producer_side():
    fig, graph, positions, anchors, _incoming, _outgoing, _input_index, _buses, link_paths = (
        _kimi_sparse_moe_layout()
    )
    try:
        from visualizer.render import (
            CONNECTOR_EXIT_STUB,
            CONNECTOR_OBSTACLE_MARGIN,
            _inline_frame_draw_bounds,
        )

        plus_index = next(
            i for i, spec in enumerate(graph.nodes) if spec.label == "+"
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
        expected_x = _side_entry_combine_entry_x(anchors[shared_tail], anchors[plus_index])
        assert abs(points[-1][0] - expected_x) < PARALLEL_CONNECTOR_COORD_EPS
        assert expected_x > anchors[plus_index].cx
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


def test_moe_route_scaling_exits_frame_tail_vertically():
    fig, graph, positions, anchors, _incoming, _outgoing, _input_index, buses, link_paths = (
        _kimi_sparse_moe_layout()
    )
    try:
        from visualizer.render import (
            _find_connector_inline_frame_overlaps,
            _inline_frame_draw_bounds,
            _frame_tail_routing_corridor_y,
            CONNECTOR_EXIT_STUB,
            CONNECTOR_OBSTACLE_MARGIN,
        )

        link_key, points = _link_by_labels(
            graph, link_paths, "Route scaling", "MoE aggregation"
        )
        route_scaling = anchors[link_key[0]]
        y_exit = _connector_source_bottom_exit_y(route_scaling)
        assert abs(points[0][1] - y_exit) < PARALLEL_CONNECTOR_COORD_EPS
        assert abs(points[0][0] - points[1][0]) < PARALLEL_CONNECTOR_COORD_EPS, (
            "Route scaling must drop vertically before leaving the gate column"
        )
        assert points[1][1] < points[0][1]
        gate = next(frame for frame in graph.inline_frames if frame.frame_id == "gate")
        gate_bounds = _inline_frame_draw_bounds(gate, positions, graph)
        corridor_y = _frame_tail_routing_corridor_y(
            gate_bounds,
            route_scaling,
            anchors[link_key[1]],
        )
        assert any(
            y <= corridor_y + PARALLEL_CONNECTOR_COORD_EPS for _x, y in points
        ), "Route scaling must route through the below-frame corridor"
        below_frame_y = (
            gate_bounds.bottom - CONNECTOR_OBSTACLE_MARGIN - CONNECTOR_EXIT_STUB
        )
        merge_bus_y = buses[3][link_key]
        if abs(merge_bus_y - corridor_y) <= PARALLEL_CONNECTOR_COORD_EPS:
            assert any(
                abs(y - merge_bus_y) < PARALLEL_CONNECTOR_COORD_EPS for _x, y in points
            )
        assert not _find_connector_inline_frame_overlaps(
            link_paths,
            graph=graph,
            positions=positions,
        )
        horizontals = [
            (y1, x1, x2)
            for (x1, y1), (x2, y2) in zip(points, points[1:])
            if abs(y1 - y2) < PARALLEL_CONNECTOR_COORD_EPS and abs(x1 - x2) > 0.06
        ]
        assert any(
            y <= corridor_y + PARALLEL_CONNECTOR_COORD_EPS for y, _x1, _x2 in horizontals
        ), "Route scaling must turn horizontally in the below-frame corridor"

        down_proj_points = link_paths[(9, 10)]
        down_horiz_y = next(
            y1 for (x1, y1), (x2, y2) in zip(down_proj_points, down_proj_points[1:]) if abs(y1 - y2) < 1e-6
        )
        assert abs(down_horiz_y - buses[3][link_key]) < PARALLEL_CONNECTOR_COORD_EPS
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
