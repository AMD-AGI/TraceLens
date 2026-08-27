"""Tests for KimiMLP input fan-out connector routing and layout shrinkwrap."""

from __future__ import annotations

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from visualizer.computation_graph import (
    DETAIL_LAYER_GAP,
    SYNTHETIC_INPUT,
    GraphNodeSpec,
    LayoutPosition,
    _compact_synthetic_input_spacing,
    _estimate_graph_height,
    build_computation_graph,
    layout_computation_graph,
    measure_graph_node_sizes,
)
from visualizer.extract import load_architecture
from visualizer.render import (
    COLORS,
    CONNECTOR_ATTACHED_BOX_MARGIN,
    CONNECTOR_EXIT_STUB,
    CONNECTOR_OBSTACLE_MARGIN,
    DIAGRAM_LEFT_MARGIN,
    PARALLEL_CONNECTOR_COORD_EPS,
    _RenderAnchor,
    _anchors_from_detail_plan,
    _assert_detail_fanout_connector_invariants,
    _assert_fanout_avoids_input_horizontal_departure,
    _assert_shared_fanout_branch_tees_aligned,
    _build_detail_draw_plan,
    _clamp_bus_y_clearing_inline_frames,
    _collect_detail_link_paths,
    _compute_detail_connector_buses,
    _connector_fanout_branch_tee_y,
    _connector_min_bus_y_above_target,
    _connector_path_violates_inline_frame_bounds,
    _connector_points_for_link,
    _connector_source_bottom_exit_y,
    _detail_sections_to_render,
    _effective_source_bus_y,
    _fanout_tee_then_entry_column_points,
    _plan_inline_bypass_bus_x,
    _reroute_connector_path_clearing_blocks,
)
from visualizer.render_validate import finalize_detail_layout, measure_detail_tree_content_width


def _anchor(pos: LayoutPosition) -> _RenderAnchor:
    left = pos.cx - pos.width / 2
    return _RenderAnchor(
        cx=pos.cx,
        top=pos.top_y,
        bottom=pos.bottom,
        left=left,
        right=left + pos.width,
    )


def _kimi_mlp_layout(*, cx: float = 3.0, top_y: float = 10.0):
    """Build a finalized KimiMLP detail graph matching the full-diagram layout path."""
    spec = load_architecture("moonshotai/Kimi-K3", detailed=True)
    tree = next(t for title, t, _ in _detail_sections_to_render(spec) if title.startswith("KimiMLP"))
    graph = build_computation_graph(tree, include_input=True)
    fig, ax = plt.subplots(figsize=(11, 13))
    input_sublabel = "← RMSNorm"
    measure_graph_node_sizes(ax, graph, input_sublabel=input_sublabel)
    min_left = DIAGRAM_LEFT_MARGIN + 0.05
    section_w = measure_detail_tree_content_width(
        ax,
        tree,
        cx=cx,
        detail_fill=COLORS["detail_fill"],
        min_left=min_left,
        input_sublabel=input_sublabel,
    )
    positions, links = layout_computation_graph(
        graph,
        cx=cx,
        top_y=top_y,
        block_w=section_w,
        block_h=_estimate_graph_height(graph),
    )
    finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=input_sublabel,
        cx=cx,
        top_y=top_y,
        detail_fill=COLORS["detail_fill"],
        min_left=min_left,
    )
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=input_sublabel)
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
    return fig, graph, positions, anchors, plan, incoming, outgoing, input_index, buses, link_paths


def _fanout_linear_links(graph, input_index: int, links) -> list[tuple[int, int]]:
    return [
        (src, tgt)
        for src, tgt in links
        if src == input_index and graph.nodes[tgt].label == "Linear"
    ]


def test_clamp_bus_y_skips_frames_above_bus():
    """Buses above a dotted frame must not be forced below it merely due to X overlap."""
    frame = type("Frame", (), {"node_indices": [3, 4], "label": "SituAndMul"})()
    graph = type("G", (), {"inline_frames": [frame]})()
    positions = [
        LayoutPosition(spec=GraphNodeSpec(key="n", label="n"), cx=1.0, top_y=10.0, width=1.0, height=0.5)
    ]
    bounds = type("B", (), {"left": 0.5, "right": 2.5, "top": 7.5, "bottom": 6.5})()

    from visualizer import render as render_mod

    original = render_mod._inline_frame_draw_bounds
    render_mod._inline_frame_draw_bounds = lambda _frame, _positions, _graph: bounds
    try:
        high_bus = 9.32
        cleared = _clamp_bus_y_clearing_inline_frames(
            high_bus,
            graph=graph,
            positions=positions,
            x_left=1.0,
            x_right=2.5,
        )
        assert abs(cleared - high_bus) < 1e-6

        low_bus = 7.0
        cleared_low = _clamp_bus_y_clearing_inline_frames(
            low_bus,
            graph=graph,
            positions=positions,
            x_left=1.0,
            x_right=2.5,
        )
        assert cleared_low < low_bus
    finally:
        render_mod._inline_frame_draw_bounds = original


def test_effective_source_bus_y_not_below_highest_target_min_bus():
    """Shared tee Y must stay at or above every fan-out target's min merge level."""
    source = _RenderAnchor(cx=1.1, top=9.9, bottom=9.4, left=0.7, right=1.5)
    high = _RenderAnchor(cx=1.6, top=9.2, bottom=8.8, left=1.3, right=1.9)
    low = _RenderAnchor(cx=2.5, top=8.5, bottom=8.1, left=2.2, right=2.8)
    min_high = _connector_min_bus_y_above_target(high)
    min_low = _connector_min_bus_y_above_target(low)
    tee_y = _effective_source_bus_y(source, [high, low], proposed_bus_y=6.7)
    assert tee_y + PARALLEL_CONNECTOR_COORD_EPS >= min_high
    assert tee_y + PARALLEL_CONNECTOR_COORD_EPS >= min_low
    assert tee_y + PARALLEL_CONNECTOR_COORD_EPS >= max(min_high, min_low)


def test_compact_synthetic_input_spacing_uplifts_downstream_targets():
    """When the input sits too high, downstream nodes move up toward it."""
    graph = type(
        "G",
        (),
        {
            "links": [(0, 1), (0, 2)],
            "nodes": [
                GraphNodeSpec(key="@input", synthetic=SYNTHETIC_INPUT, label="@input"),
                GraphNodeSpec(key="l1", label="Linear"),
                GraphNodeSpec(key="l2", label="Linear"),
            ],
            "inline_frames": [],
        },
    )()
    positions = [
        LayoutPosition(spec=graph.nodes[0], cx=1.0, top_y=10.0, width=1.0, height=0.5),
        LayoutPosition(spec=graph.nodes[1], cx=1.0, top_y=8.0, width=1.0, height=0.4),
        LayoutPosition(spec=graph.nodes[2], cx=2.0, top_y=7.0, width=1.0, height=0.4),
    ]
    before_gap = positions[0].bottom - max(positions[1].top_y, positions[2].top_y)
    assert before_gap > DETAIL_LAYER_GAP + 0.05
    tops_before = [positions[1].top_y, positions[2].top_y]
    _compact_synthetic_input_spacing(positions, graph)
    tops_after = [positions[1].top_y, positions[2].top_y]
    assert tops_after[0] > tops_before[0]
    assert tops_after[1] > tops_before[1]
    after_gap = positions[0].bottom - max(positions[1].top_y, positions[2].top_y)
    assert abs(after_gap - DETAIL_LAYER_GAP) < 0.02


def test_fanout_tee_then_entry_column_direct_drop_when_bus_at_tee():
    """When per-leg bus is at or above the tee, drop straight from the branch."""
    source = _RenderAnchor(cx=1.0, top=9.9, bottom=9.4, left=0.7, right=1.3)
    target = _RenderAnchor(cx=1.6, top=9.2, bottom=8.8, left=1.3, right=1.9)
    tee_y = 9.3
    bus_y = 9.35
    points = _fanout_tee_then_entry_column_points(
        source,
        target,
        target.cx,
        tee_y=tee_y,
        bus_y=bus_y,
    )
    assert len(points) == 4
    assert abs(points[1][1] - tee_y) < 1e-6
    assert abs(points[2][1] - tee_y) < 1e-6
    assert abs(points[3][1] - target.top) < 1e-6


def test_connector_fanout_branch_tee_y_detects_shared_branch():
    source = _RenderAnchor(cx=1.0, top=9.9, bottom=9.4, left=0.7, right=1.3)
    tee_y = 9.3
    points = [
        (1.0, 9.4),
        (1.0, tee_y),
        (1.7, tee_y),
        (1.7, 9.2),
    ]
    assert abs(_connector_fanout_branch_tee_y(points, source=source) - tee_y) < 1e-6


def test_kimi_mlp_fanout_tee_levels_aligned():
    fig, graph, _positions, anchors, _plan, _incoming, _outgoing, input_index, buses, link_paths = (
        _kimi_mlp_layout()
    )
    try:
        source_bus = buses[1]
        assert input_index in source_bus
        tee_y = source_bus[input_index]
        source = anchors[input_index]
        for link in _fanout_linear_links(graph, input_index, graph.links):
            points = link_paths[link]
            branch_tee = _connector_fanout_branch_tee_y(points, source=source)
            assert branch_tee is not None, f"missing branch tee on {link}: {points}"
            assert abs(branch_tee - tee_y) < PARALLEL_CONNECTOR_COORD_EPS
            assert abs(points[1][1] - tee_y) < PARALLEL_CONNECTOR_COORD_EPS
            assert abs(points[2][1] - tee_y) < PARALLEL_CONNECTOR_COORD_EPS
    finally:
        plt.close(fig)


def test_kimi_mlp_fanout_leaves_input_vertically():
    fig, graph, _positions, anchors, _plan, _incoming, _outgoing, input_index, _buses, link_paths = (
        _kimi_mlp_layout()
    )
    try:
        source = anchors[input_index]
        y_exit = _connector_source_bottom_exit_y(source)
        for link in _fanout_linear_links(graph, input_index, graph.links):
            points = link_paths[link]
            assert abs(points[0][0] - source.cx) < PARALLEL_CONNECTOR_COORD_EPS
            assert abs(points[0][1] - y_exit) < PARALLEL_CONNECTOR_COORD_EPS
            assert abs(points[1][0] - source.cx) < PARALLEL_CONNECTOR_COORD_EPS
            assert points[1][1] < y_exit - PARALLEL_CONNECTOR_COORD_EPS
            x1, y1 = points[0]
            x2, y2 = points[1]
            if (
                abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS
                and abs(x1 - x2) > PARALLEL_CONNECTOR_COORD_EPS
                and abs(y1 - y_exit) <= CONNECTOR_OBSTACLE_MARGIN + PARALLEL_CONNECTOR_COORD_EPS
            ):
                pytest.fail(f"{link} has horizontal departure at input bottom: {points}")
    finally:
        plt.close(fig)


def test_kimi_mlp_input_to_linear_gap_is_tight():
    fig, graph, positions, _anchors, _plan, _incoming, _outgoing, input_index, _buses, _link_paths = (
        _kimi_mlp_layout()
    )
    try:
        linear_targets = [tgt for src, tgt in graph.links if src == input_index]
        highest_top = max(positions[tgt].top_y for tgt in linear_targets)
        input_bottom = positions[input_index].bottom
        gap = input_bottom - highest_top
        assert gap <= DETAIL_LAYER_GAP + 0.02
    finally:
        plt.close(fig)


def test_kimi_mlp_exit_to_tee_stub_is_short():
    fig, _graph, _positions, anchors, _plan, _incoming, _outgoing, input_index, buses, _link_paths = (
        _kimi_mlp_layout()
    )
    try:
        source = anchors[input_index]
        y_exit = _connector_source_bottom_exit_y(source)
        tee_y = buses[1][input_index]
        assert y_exit - tee_y <= CONNECTOR_EXIT_STUB + 0.03
    finally:
        plt.close(fig)


def test_kimi_mlp_post_split_drops_are_minimal():
    fig, graph, _positions, anchors, _plan, _incoming, _outgoing, input_index, buses, link_paths = (
        _kimi_mlp_layout()
    )
    try:
        merge_link_bus = buses[3]
        max_drop = CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN + 0.01
        for link in _fanout_linear_links(graph, input_index, graph.links):
            points = link_paths[link]
            drop = abs(points[-2][1] - points[-1][1])
            assert drop <= max_drop, f"{link} final drop {drop:.4f} > {max_drop:.4f}"
            target = anchors[link[1]]
            assert abs(points[-1][1] - target.top) < 1e-6
            if merge_link_bus[link] < buses[1][input_index] - PARALLEL_CONNECTOR_COORD_EPS:
                assert len(points) == 5
            else:
                assert len(points) == 4
    finally:
        plt.close(fig)


def test_kimi_mlp_l2_routes_via_tee_not_input_horizontal():
    """Regression: lower Linear leg must not reroute horizontally at the input bottom."""
    fig, graph, positions, anchors, plan, incoming, outgoing, input_index, buses, link_paths = (
        _kimi_mlp_layout()
    )
    try:
        linear_indices = [tgt for src, tgt in graph.links if src == input_index and graph.nodes[tgt].label == "Linear"]
        assert len(linear_indices) == 2
        right_linear = max(linear_indices, key=lambda index: positions[index].cx)
        link = (input_index, right_linear)
        points = link_paths[link]
        source = anchors[input_index]
        tee_y = buses[1][input_index]
        assert abs(points[1][1] - tee_y) < PARALLEL_CONNECTOR_COORD_EPS
        assert abs(points[2][1] - tee_y) < PARALLEL_CONNECTOR_COORD_EPS

        inline_bypass_bus_x = _plan_inline_bypass_bus_x(graph, graph.links, anchors, positions)
        initial = _connector_points_for_link(
            graph=graph,
            positions=positions,
            anchors=anchors,
            src=input_index,
            tgt=right_linear,
            link_key=link,
            incoming=incoming,
            outgoing=outgoing,
            label_obstacles=plan.label_obstacles,
            target_bus=buses[0],
            source_bus=buses[1],
            merge_entry_x=buses[2],
            merge_link_bus=buses[3],
            input_index=input_index,
            inline_bypass_bus_x=inline_bypass_bus_x,
        )
        straight = [
            (source.cx, _connector_source_bottom_exit_y(source)),
            (source.cx, buses[3][link]),
            (anchors[right_linear].cx, buses[3][link]),
            (anchors[right_linear].cx, anchors[right_linear].top),
        ]
        route_obstacles = [
            a for i, a in anchors.items() if i not in {input_index, right_linear}
        ] + plan.label_obstacles
        assert _connector_path_violates_inline_frame_bounds(
            straight,
            graph,
            positions,
            src=input_index,
            tgt=right_linear,
        ) is None
        rerouted = _reroute_connector_path_clearing_blocks(
            initial,
            source=source,
            target=anchors[right_linear],
            obstacles=route_obstacles,
            bus_y=buses[3][link],
            graph=graph,
            positions=positions,
            link_key=link,
            source_bus=buses[1],
            merge_link_bus=buses[3],
        )
        assert abs(rerouted[1][1] - tee_y) < PARALLEL_CONNECTOR_COORD_EPS
        assert abs(rerouted[1][1] - _connector_source_bottom_exit_y(source)) > 0.05
    finally:
        plt.close(fig)


def test_kimi_mlp_collect_detail_link_paths_enforces_runtime_invariants():
    """Full collect pipeline must satisfy fan-out runtime checks without raising."""
    fig, *_rest, link_paths = _kimi_mlp_layout()
    try:
        assert link_paths
    finally:
        plt.close(fig)


def test_runtime_branch_tee_alignment_raises_on_mismatch():
    source = _RenderAnchor(cx=1.0, top=9.9, bottom=9.4, left=0.7, right=1.3)
    anchors = {0: source, 1: source, 2: source}
    outgoing = {0: [(0, 1), (0, 2)]}
    link_paths = {
        (0, 1): [(1.0, 9.4), (1.0, 9.31), (1.5, 9.31), (1.5, 9.0)],
        (0, 2): [(1.0, 9.4), (1.0, 9.27), (2.0, 9.27), (2.0, 8.5)],
    }
    graph = type("G", (), {})()
    with pytest.raises(RuntimeError, match="fan-out branch tees misaligned"):
        _assert_shared_fanout_branch_tees_aligned(
            link_paths,
            graph=graph,
            anchors=anchors,
            outgoing=outgoing,
            source_bus={0: 9.29},
            target_bus={},
            stage="test",
        )


def test_runtime_branch_tee_alignment_skips_side_routed_legs():
    """Legs without a source-column tee (side bypass) must not fail alignment checks."""
    source = _RenderAnchor(cx=3.0, top=9.9, bottom=9.4, left=2.7, right=3.3)
    anchors = {0: source, 1: source, 12: source}
    outgoing = {0: [(0, 1), (0, 12)]}
    link_paths = {
        (0, 1): [(3.0, 9.4), (3.0, 9.3), (2.5, 9.3), (2.5, 9.0)],
        (0, 12): [(3.0, 9.4), (3.5, 9.4), (1.0, 9.4), (1.0, 8.5)],
    }
    graph = type("G", (), {})()
    _assert_shared_fanout_branch_tees_aligned(
        link_paths,
        graph=graph,
        anchors=anchors,
        outgoing=outgoing,
        source_bus={0: 9.3},
        target_bus={},
        stage="test",
    )


def test_runtime_horizontal_departure_passes_for_pure_tee_fanout():
    source = _RenderAnchor(cx=1.0, top=9.9, bottom=9.4, left=0.7, right=1.3)
    anchors = {0: source, 1: source, 2: source}
    outgoing = {0: [(0, 1), (0, 2)]}
    y_exit = _connector_source_bottom_exit_y(source)
    link_paths = {
        (0, 1): [(1.0, y_exit), (1.0, 9.3), (1.5, 9.3), (1.5, 9.0)],
        (0, 2): [(1.0, y_exit), (1.0, 9.3), (2.0, 9.3), (2.0, 8.5)],
    }
    graph = type("G", (), {})()
    _assert_fanout_avoids_input_horizontal_departure(
        link_paths,
        graph=graph,
        anchors=anchors,
        outgoing=outgoing,
        source_bus={0: 9.3},
        stage="test",
    )


def test_runtime_horizontal_departure_skips_mixed_side_routed_fanout():
    """Side-routed legs in a mixed fan-out must not trigger the pure-tee check."""
    source = _RenderAnchor(cx=3.0, top=9.9, bottom=9.4, left=2.7, right=3.3)
    anchors = {0: source, 1: source, 12: source}
    outgoing = {0: [(0, 1), (0, 12)]}
    y_exit = _connector_source_bottom_exit_y(source)
    link_paths = {
        (0, 1): [(3.0, y_exit), (3.0, 9.3), (2.5, 9.3), (2.5, 9.0)],
        (0, 12): [(3.0, y_exit), (3.5, y_exit), (1.0, y_exit), (1.0, 8.5)],
    }
    graph = type("G", (), {})()
    _assert_fanout_avoids_input_horizontal_departure(
        link_paths,
        graph=graph,
        anchors=anchors,
        outgoing=outgoing,
        source_bus={0: 9.3},
        stage="test",
    )


def test_runtime_detail_fanout_invariants_hold_for_kimi_mlp_paths():
    fig, graph, _positions, anchors, _plan, _incoming, outgoing, input_index, buses, link_paths = (
        _kimi_mlp_layout()
    )
    try:
        _assert_detail_fanout_connector_invariants(
            link_paths,
            graph=graph,
            anchors=anchors,
            outgoing=outgoing,
            source_bus=buses[1],
            target_bus=buses[0],
            stage="test",
        )
    finally:
        plt.close(fig)
