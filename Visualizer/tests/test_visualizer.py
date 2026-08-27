"""Tests for TraceLens Visualizer (CPU-only)."""

from pathlib import Path

import pytest

from visualizer.ast_analyze import analyze_source, dump_ast
from visualizer.extract import load_architecture, parse_architecture
from visualizer.render import (
    COLORS,
    MERGE_OUTPUT_GAP,
    RESIDUAL_ADD_ENTRY_BAND,
    RESIDUAL_ADD_HALF_H,
    RESIDUAL_BRANCH_LIFT,
    TOP_ENTRY_PORT_GAP,
    _collect_sublayer_pairs,
    _make_node,
    _merge_edge_above,
    _merge_edge_below,
    _merge_y_for_module,
    _ordered_block_components,
    _residual_branch_y,
    _residual_merge,
    _residual_merge_ports,
    render_diagram,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_merge_node_sits_below_module_box():
    module_bottom = 4.0
    merge_y = _merge_y_for_module(module_bottom)
    merge_top = _merge_edge_above(merge_y)
    assert merge_top < module_bottom
    # Both inputs approach through lanes stacked in the band above the tile.
    assert merge_top + RESIDUAL_ADD_ENTRY_BAND <= module_bottom - MERGE_OUTPUT_GAP + 1e-9
    assert merge_top - _merge_edge_below(merge_y) == pytest.approx(2 * RESIDUAL_ADD_HALF_H)


def test_repeat_label_clears_positional_and_routes_around_bbox():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.render import (
        BLOCK_FRAME_LABEL_PAD_X,
        BLOCK_FRAME_REPEAT_CONNECTOR_MARGIN,
        BLOCK_FRAME_REPEAT_LABEL_GAP,
        BLOCK_FRAME_DECODER_FRAME_GAP,
        BLOCK_FRAME_DECODER_OUTSIDE_GAP,
        BLOCK_FRAME_REPEAT_OUTSIDE_GAP,
        DIAGRAM_LEFT_MARGIN,
        FRAME_PATCH_TOP_OUTSET,
        STACK_BOX_BOTTOM_OUTSET,
        _block_frame_top,
        _block_top_below_repeat_label,
        _block_width_for_repeat_label,
        _decoder_label_bbox,
        _effective_repeat_outside_gap,
        _main_block_width,
        _outside_block_labels_bbox,
        _repeat_label_bbox,
        _text_size_in_axes,
    )
    from visualizer.sizing import FRAME_LABEL_PAD_X, box_width_for_text_width

    fig, ax = plt.subplots(figsize=(11, 13))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13)
    fig.canvas.draw()
    above_bottom = 10.195
    repeat_label = "93 × Transformer block"
    decoder_label = "KimiDecoderLayer"
    block_w = _main_block_width(
        ax,
        repeat_label=repeat_label,
        decoder_label=decoder_label,
        inner_w=3.0,
    )
    repeat_w = _block_width_for_repeat_label(ax, repeat_label)
    assert block_w >= repeat_w
    text_w, _ = _text_size_in_axes(ax, repeat_label, fontsize=10.0, fontweight="bold")
    assert repeat_w == pytest.approx(2 * box_width_for_text_width(text_w, pad_x=FRAME_LABEL_PAD_X))

    cx = DIAGRAM_LEFT_MARGIN + block_w / 2
    block_top = _block_top_below_repeat_label(
        ax,
        cx=cx,
        block_w=block_w,
        above_bottom=above_bottom,
        repeat_label=repeat_label,
        decoder_label=decoder_label,
    )
    frame_top = _block_frame_top(block_top)
    text_x = cx - block_w / 2 + BLOCK_FRAME_LABEL_PAD_X
    outside_gap = _effective_repeat_outside_gap(ax, repeat_label, decoder_label)
    outside_bb = _outside_block_labels_bbox(
        ax,
        text_x,
        frame_top,
        repeat_label,
        decoder_label,
    )
    repeat_bb = _repeat_label_bbox(
        ax,
        text_x,
        frame_top + outside_gap,
        repeat_label,
    )
    decoder_bb = _decoder_label_bbox(
        ax,
        text_x,
        repeat_bb.y0 - BLOCK_FRAME_DECODER_OUTSIDE_GAP,
        decoder_label,
        va="top",
    )

    assert outside_bb.y1 <= above_bottom - STACK_BOX_BOTTOM_OUTSET - BLOCK_FRAME_REPEAT_LABEL_GAP
    assert decoder_bb.y1 <= repeat_bb.y0 - BLOCK_FRAME_DECODER_OUTSIDE_GAP + 1e-6
    assert decoder_bb.y0 >= frame_top + FRAME_PATCH_TOP_OUTSET + BLOCK_FRAME_DECODER_FRAME_GAP - 1e-6
    frame_left = cx - block_w / 2
    assert repeat_bb.x0 >= frame_left + BLOCK_FRAME_LABEL_PAD_X - 1e-6
    label_w = repeat_bb.x1 - repeat_bb.x0
    assert repeat_bb.x1 <= frame_left + BLOCK_FRAME_LABEL_PAD_X + label_w + 1e-6
    assert outside_bb.x1 + BLOCK_FRAME_REPEAT_CONNECTOR_MARGIN <= cx


def _record_residual_merge(*, module_cx: float, spine_x: float, branch_x: float, module_bottom: float):
    """Run _residual_merge with drawing stubbed out; return (segments, add_node)."""
    from unittest.mock import patch

    segments: list[tuple[float, float, float, float]] = []

    def record_line(ax, x1, y1, x2, y2, **kwargs):
        segments.append((x1, y1, x2, y2))

    def record_arrow(ax, x1, y1, x2, y2, **kwargs):
        segments.append((x1, y1, x2, y2))

    def record_path(ax, points, **kwargs):
        for start, end in zip(points, points[1:]):
            segments.append((*start, *end))

    with patch("visualizer.render._line", side_effect=record_line), patch(
        "visualizer.render._arrow", side_effect=record_arrow
    ), patch("visualizer.render._draw_path", side_effect=record_path), patch(
        "visualizer.render._draw_box"
    ):
        merge_y, add_node = _residual_merge(
            None,
            merge_id="ffn_add",
            module_cx=module_cx,
            module_bottom=module_bottom,
            skip_from_y=module_bottom + 1.0,
            spine_x=spine_x,
            branch_x=branch_x,
            width=1.35,
        )

    assert merge_y == pytest.approx(_merge_y_for_module(module_bottom))
    return segments, add_node


@pytest.mark.parametrize("module_cx_offset", [0.0, 0.4])
def test_residual_merge_inputs_enter_distinct_top_ports(module_cx_offset):
    spine_x = 3.0
    module_bottom = 5.0
    segments, add_node = _record_residual_merge(
        module_cx=spine_x + module_cx_offset,
        spine_x=spine_x,
        branch_x=1.0,
        module_bottom=module_bottom,
    )

    module_port_x, residual_port_x = _residual_merge_ports(spine_x, add_node.w)
    assert module_port_x != residual_port_x
    assert residual_port_x < module_port_x
    assert module_port_x == pytest.approx(add_node.cx)
    assert module_port_x - residual_port_x >= TOP_ENTRY_PORT_GAP
    # Both ports stay on the top edge, clear of the tile's rounded corners.
    for port_x in (module_port_x, residual_port_x):
        assert add_node.x + TOP_ENTRY_PORT_GAP <= port_x <= add_node.x + add_node.w - TOP_ENTRY_PORT_GAP

    landings = [(x2, y2) for _, _, x2, y2 in segments if y2 == pytest.approx(add_node.top)]
    assert sorted(x for x, _ in landings) == [
        pytest.approx(residual_port_x),
        pytest.approx(module_port_x),
    ]
    # Every approach is vertical, so nothing arrives along the tile's flanks.
    for x1, y1, x2, y2 in segments:
        if y2 != pytest.approx(add_node.top):
            continue
        assert x1 == pytest.approx(x2)
        assert y1 > y2
    assert not any(
        y1 == pytest.approx(y2)
        and (
            x2 == pytest.approx(add_node.x)
            or x2 == pytest.approx(add_node.x + add_node.w)
        )
        for x1, y1, x2, y2 in segments
    )
    # Nothing crosses the tile body.
    assert all(
        min(y1, y2) >= add_node.top - 1e-9 or max(y1, y2) <= add_node.bottom + 1e-9
        for x1, y1, x2, y2 in segments
    )


def test_residual_merge_output_leaves_the_bottom_edge():
    from unittest.mock import patch

    from visualizer.render import _connect_from_merge

    module_bottom = 5.0
    _, add_node = _record_residual_merge(
        module_cx=3.0, spine_x=3.0, branch_x=1.0, module_bottom=module_bottom
    )
    merge_y = _merge_y_for_module(module_bottom)
    target = _make_node(
        "norm2", 3.0, merge_y - 0.4, 1.35, 0.32, "RMSNorm", COLORS["norm"], text_color=COLORS["text"]
    )

    exits: list[tuple[float, float, float, float]] = []
    with patch(
        "visualizer.render._arrow",
        side_effect=lambda ax, x1, y1, x2, y2, **kwargs: exits.append((x1, y1, x2, y2)),
    ):
        _connect_from_merge(None, 3.0, merge_y, target)

    assert len(exits) == 1
    x1, y1, x2, y2 = exits[0]
    assert (x1, y1) == (pytest.approx(add_node.cx), pytest.approx(add_node.bottom))
    assert (x2, y2) == (pytest.approx(target.cx), pytest.approx(target.top))


def test_residual_branch_routes_above_norm():
    norm = _make_node("norm", 5.0, 10.0, 1.35, 0.32, "RMSNorm", COLORS["norm"], text_color=COLORS["text"])
    branch_y = _residual_branch_y(norm.top)
    assert branch_y > norm.top
    assert branch_y - norm.top >= RESIDUAL_BRANCH_LIFT - 1e-6


def test_collect_connector_join_points_finds_shared_vertices():
    from visualizer.render import _RenderAnchor, _collect_connector_join_points

    link_paths = {
        (0, 2): [(0.0, 2.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
        (1, 2): [(2.0, 2.0), (2.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
    }
    joins = _collect_connector_join_points(
        link_paths,
        target_bus={2: 1.0},
        incoming={2: [(0, 2), (1, 2)]},
        anchors={2: _RenderAnchor(cx=1.0, top=0.1, bottom=0.0, left=0.5, right=1.5)},
    )
    assert (1.0, 1.0) in joins
    assert (1.0, 0.0) not in joins


def test_collect_connector_join_points_ignores_box_endpoints():
    from visualizer.render import _RenderAnchor, _collect_connector_join_points

    link_paths = {
        (0, 2): [(0.0, 2.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
        (1, 2): [(2.0, 2.0), (2.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
    }
    joins = _collect_connector_join_points(
        link_paths,
        target_bus={2: 1.0},
        incoming={2: [(0, 2), (1, 2)]},
        anchors={2: _RenderAnchor(cx=1.0, top=0.1, bottom=0.0, left=0.5, right=1.5)},
    )
    assert all(
        (x, y) != (0.0, 2.0) and (x, y) != (2.0, 2.0) and (x, y) != (1.0, 0.0)
        for x, y in joins
    )


def test_collect_connector_join_points_ignores_single_link_corners():
    from visualizer.render import _collect_connector_join_points

    link_paths = {
        (0, 2): [(0.0, 2.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
    }
    joins = _collect_connector_join_points(link_paths)
    assert joins == []


def test_collect_connector_join_points_ignores_l_bends_on_bus():
    from visualizer.render import _collect_connector_join_points

    link_paths = {
        (0, 2): [(0.0, 2.0), (0.0, 1.0), (1.5, 1.0), (2.0, 1.0), (2.0, 0.0)],
        (1, 2): [(1.5, 2.0), (1.5, 1.0), (1.5, 0.0)],
    }
    joins = _collect_connector_join_points(
        link_paths,
        target_bus={2: 1.0},
    )
    assert (1.5, 1.0) in joins
    single_link = {
        (0, 2): [(0.0, 2.0), (0.0, 1.0), (1.5, 1.0), (1.5, 0.0)],
    }
    assert _collect_connector_join_points(single_link, target_bus={2: 1.0}) == []


def test_collect_connector_join_points_ignores_crossings():
    from visualizer.render import _collect_connector_join_points

    link_paths = {
        (0, 1): [(0.0, 1.0), (2.0, 1.0)],
        (2, 3): [(1.0, 2.0), (1.0, 0.0)],
    }
    joins = _collect_connector_join_points(link_paths)
    assert not any(abs(x - 1.0) < 1e-6 and abs(y - 1.0) < 1e-6 for x, y in joins)


def test_draw_path_uses_flat_line_caps():
    import matplotlib.pyplot as plt

    from visualizer.render import _draw_path

    fig, ax = plt.subplots()
    try:
        _draw_path(ax, [(0.0, 1.0), (0.0, 0.5), (1.0, 0.5)])
        line = ax.lines[-1]
        assert line.get_solid_capstyle() == "butt"
        assert line.get_solid_joinstyle() == "miter"
        assert len(ax.lines) == 1
    finally:
        plt.close(fig)


def test_spread_top_entry_ports_cluster_near_center():
    from visualizer.render import (
        TOP_ENTRY_PORT_GAP,
        _RenderAnchor,
        _assign_spread_merge_entry_x,
    )

    target = _RenderAnchor(cx=5.0, top=6.0, bottom=5.0, left=4.0, right=6.0)
    positions = [
        None,
        type("Pos", (), {"cx": 3.0, "bottom": 7.0})(),
        type("Pos", (), {"cx": 4.5, "bottom": 7.0})(),
        type("Pos", (), {"cx": 6.5, "bottom": 7.0})(),
    ]
    merge_entry_x: dict[tuple[int, int], float] = {}
    links = [(1, 2), (2, 2), (3, 2)]
    _assign_spread_merge_entry_x(links, target, None, positions, {}, merge_entry_x)
    xs = sorted(merge_entry_x[link] for link in links)
    assert max(xs) - min(xs) >= TOP_ENTRY_PORT_GAP
    assert all(abs(x - target.cx) <= TOP_ENTRY_PORT_GAP * 2 for x in xs)


def test_snap_connector_path_endpoints_keeps_paths_orthogonal():
    from visualizer.render import _RenderAnchor, _snap_connector_path_endpoints

    source = _RenderAnchor(cx=1.0, top=2.0, bottom=1.0, left=0.5, right=1.5)
    target = _RenderAnchor(cx=3.0, top=0.5, bottom=0.0, left=2.5, right=3.5)
    graph = type("Graph", (), {})()
    snapped = _snap_connector_path_endpoints(
        [(1.0, 1.0), (2.0, 0.8), (3.0, 0.5)],
        source=source,
        target=target,
        link_key=(0, 1),
        graph=graph,
    )
    for index in range(len(snapped) - 1):
        x1, y1 = snapped[index]
        x2, y2 = snapped[index + 1]
        assert abs(x1 - x2) < 1e-6 or abs(y1 - y2) < 1e-6


def test_junction_dot_fits_rejects_box_overlap():
    from visualizer.render import (
        CONNECTOR_JUNCTION_HALO_RADIUS,
        _RenderAnchor,
        _junction_dot_fits,
    )

    anchor = _RenderAnchor(cx=1.0, top=1.1, bottom=0.9, left=0.8, right=1.2)
    assert not _junction_dot_fits(
        1.0,
        1.0,
        [anchor],
        halo_radius=CONNECTOR_JUNCTION_HALO_RADIUS,
    )
    assert _junction_dot_fits(
        2.0,
        2.0,
        [anchor],
        halo_radius=CONNECTOR_JUNCTION_HALO_RADIUS,
    )


def test_detail_connector_linestyle_is_always_solid():
    from visualizer.computation_graph import (
        ComputationGraph,
        GraphNodeSpec,
        LayoutPosition,
        SYNTHETIC_TENSOR,
    )
    from visualizer.render import _RenderAnchor, _detail_connector_linestyle

    graph = ComputationGraph()
    graph.nodes.append(GraphNodeSpec(key="q", label="q", synthetic=SYNTHETIC_TENSOR))
    graph.nodes.append(GraphNodeSpec(key="sum", label="Sum"))
    positions = [
        LayoutPosition(spec=graph.nodes[0], cx=0.5, top_y=10.0, width=0.5, height=0.4),
        LayoutPosition(spec=graph.nodes[1], cx=2.0, top_y=8.0, width=0.5, height=0.4),
    ]
    source = _RenderAnchor(cx=0.5, top=10.0, bottom=9.6, left=0.25, right=0.75)
    target = _RenderAnchor(cx=2.0, top=8.0, bottom=7.6, left=1.75, right=2.25)
    assert _detail_connector_linestyle(graph, src=0, positions=positions, source=source, target=target) == "solid"

    near_target = _RenderAnchor(cx=2.05, top=10.0, bottom=9.6, left=1.8, right=2.3)
    assert _detail_connector_linestyle(graph, src=0, positions=positions, source=near_target, target=target) == "solid"


def test_converging_connectors_share_target_bus():
    from visualizer.render import (
        _RenderAnchor,
        _compute_shared_target_bus_y,
        _orthogonal_path,
    )

    sources = [
        _RenderAnchor(cx=1.0, top=8.0, bottom=7.5, left=0.5, right=1.5),
        _RenderAnchor(cx=3.0, top=8.0, bottom=7.5, left=2.5, right=3.5),
        _RenderAnchor(cx=5.0, top=8.0, bottom=7.5, left=4.5, right=5.5),
        _RenderAnchor(cx=7.0, top=8.0, bottom=7.5, left=6.5, right=7.5),
    ]
    target = _RenderAnchor(cx=4.0, top=6.0, bottom=5.5, left=3.5, right=4.5)
    bus_y = _compute_shared_target_bus_y(sources, target, obstacles=[])

    for source in sources:
        points = _orthogonal_path(source, target, [], bus_y=bus_y)
        assert points[-2][0] == target.cx
        assert points[-2][1] == bus_y
        assert points[-1] == (target.cx, target.top)


def test_three_input_block_skips_shared_target_bus():
    from visualizer.render import SHARED_CONNECTOR_BUS_MIN_LINKS, _should_use_shared_connector_bus

    assert SHARED_CONNECTOR_BUS_MIN_LINKS == 4
    assert not _should_use_shared_connector_bus(3)
    assert _should_use_shared_connector_bus(4)


def test_fanout_connectors_share_source_bus():
    from visualizer.render import (
        _RenderAnchor,
        _compute_shared_source_bus_y,
        _orthogonal_path,
    )

    source = _RenderAnchor(cx=4.0, top=8.0, bottom=7.5, left=3.5, right=4.5)
    targets = [
        _RenderAnchor(cx=1.0, top=6.0, bottom=5.5, left=0.5, right=1.5),
        _RenderAnchor(cx=3.0, top=6.0, bottom=5.5, left=2.5, right=3.5),
        _RenderAnchor(cx=5.0, top=6.0, bottom=5.5, left=4.5, right=5.5),
        _RenderAnchor(cx=7.0, top=6.0, bottom=5.5, left=6.5, right=7.5),
    ]
    bus_y = _compute_shared_source_bus_y(source, targets, obstacles=[])

    for target in targets:
        points = _orthogonal_path(source, target, [], bus_near="source", bus_y=bus_y)
        assert points[1] == (source.cx, bus_y)
        if abs(source.cx - target.cx) < 0.06:
            assert points[-1][0] == source.cx
        else:
            assert points[2][1] == bus_y
            assert points[-2][0] == target.cx


def test_min_vertical_block_gap_matches_top_text_inset():
    from visualizer.sizing import min_vertical_block_gap, single_line_box_height

    assert min_vertical_block_gap() == single_line_box_height() / 2


def test_input_box_uses_detail_tile_padding():
    import matplotlib.pyplot as plt

    from visualizer.sizing import BLOCK_PAD_Y
    from visualizer.text_measure import (
        box_label_size,
        input_box_label_size,
        measure_stacked_label_bounds,
    )

    fig, ax = plt.subplots(figsize=(13, 13))
    fig.canvas.draw()
    label = "hidden_states"
    sublabel = "← upstream"
    fontsize = 7.2
    input_w, input_h = input_box_label_size(ax, label, sublabel, fontsize=fontsize)
    detail_w, detail_h = box_label_size(
        ax,
        label,
        sublabel,
        fontsize=fontsize,
        white_text_stroke_pad=False,
    )
    assert input_w == pytest.approx(detail_w)
    assert input_h == pytest.approx(detail_h)

    text_bounds = measure_stacked_label_bounds(ax, label, sublabel, fontsize=fontsize)
    _, single_h = input_box_label_size(ax, label, None, fontsize=fontsize)
    assert input_h > single_h
    assert input_h >= (text_bounds.top - text_bounds.bottom) + 2 * BLOCK_PAD_Y - 1e-6


def test_box_label_size_scales_with_diagram_axes():
    """Regression: diagram axes must span figure width so point-sized fonts fit in tiles."""
    import matplotlib.pyplot as plt

    from visualizer.text_measure import box_label_size, ensure_diagram_measure_axes

    fig, ax = plt.subplots(figsize=(13, 13))
    fig.canvas.draw()
    assert ax.get_xlim()[1] - ax.get_xlim()[0] <= 1.5
    ensure_diagram_measure_axes(ax)
    assert ax.get_xlim()[1] - ax.get_xlim()[0] >= 12.0
    width, _height = box_label_size(ax, "Linear", None, fontsize=7.6)
    assert width >= 0.6


def test_basic_op_tile_text_fits_on_render_axes():
    """Gray basic-op tiles must contain labels when measured on the render axis scale."""
    import matplotlib.pyplot as plt

    from visualizer.block_tree import BlockNode
    from visualizer.computation_graph import build_computation_graph, layout_computation_graph, measure_graph_node_sizes, _estimate_graph_height
    from visualizer.render import COLORS, _build_detail_draw_plan
    from visualizer.render_validate import collect_measured_elements, finalize_detail_layout, validate_render_layout, VALIDATE_MIN_GAP
    from visualizer.text_measure import ensure_diagram_measure_axes

    root = BlockNode(
        attr_name="mlp",
        class_name="MLP",
        role="ffn",
        label="MLP",
        children=[
            BlockNode(attr_name="gate_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="up_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="down_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
        ],
    )
    fig, ax = plt.subplots(figsize=(11, 13))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 13)
    fig.canvas.draw()
    ensure_diagram_measure_axes(ax)
    graph = build_computation_graph(root)
    measure_graph_node_sizes(ax, graph)
    positions, _ = layout_computation_graph(graph, cx=5.5, top_y=10.0, block_w=8.0, block_h=_estimate_graph_height(graph))
    plan = finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=None,
        cx=5.5,
        top_y=10.0,
        detail_fill=COLORS["detail_fill"],
    )
    elements = collect_measured_elements(ax, graph, positions, plan, detail_fill=COLORS["detail_fill"])
    overflows = [line for line in validate_render_layout(elements, min_gap=VALIDATE_MIN_GAP).overlaps if line.startswith("text overflows")]
    assert not overflows
    gray_boxes = [
        element
        for element in elements
        if element.kind == "box" and element.label == "Linear"
    ]
    assert gray_boxes
    assert all(draw[0].facecolor == COLORS["basic_op"] for draw in plan.node_draws if draw[0].label == "Linear")
    assert all(draw[0].text_color == COLORS["text"] for draw in plan.node_draws if draw[0].facecolor == COLORS["basic_op"])


def test_basic_op_labels_export_as_dark_text_without_stroke(tmp_path: Path):
    """Gray basic-op tiles must use plain dark text, not white fill + black stroke."""
    import re

    from visualizer.basic_ops import BasicOpFilter
    from visualizer.render import COLORS, render_diagram

    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    out = render_diagram(spec, tmp_path / "basic_op_text.svg", detailed=True)
    svg = out.read_text(encoding="utf-8")
    basic_labels = ("Linear", "Silu", "Softmax", "MatMul")
    for label in basic_labels:
        for match in re.finditer(
            rf'<!-- {re.escape(label)} -->\s*\n\s*<g style="fill: (#[^"]+)" transform="translate\([^"]+\) scale\([^"]+\)">'
            r"((?:(?!</g>).)*?)</g>",
            svg,
            re.DOTALL,
        ):
            fill = match.group(1)
            body = match.group(2)
            if fill.lower() in {"#ffffff", "#fff"}:
                continue
            assert fill == COLORS["text"]
            assert "stroke: #000000" not in body
            assert "paint-order: stroke fill" not in body


def test_attn_norm_and_ffn_norm_classify_as_norm():
    from visualizer.ast_analyze import _classify_role
    from visualizer.blocks import BlockComponent, collect_norm_module_pairs

    assert _classify_role("attn_norm", "RMSNorm") == "norm"
    assert _classify_role("ffn_norm", "RMSNorm") == "norm"
    assert _classify_role("post_attention_layernorm", "RMSNorm") == "norm"
    assert _classify_role("attn", "Attention") == "attention"

    pairs = collect_norm_module_pairs(
        [
            BlockComponent("hc_pre", "hc_pre", "other", "hc pre", forward_order=0),
            BlockComponent("attn_norm", "RMSNorm", "norm", "RMSNorm", forward_order=1),
            BlockComponent("attn", "Attention", "attention", "Attn", forward_order=2),
            BlockComponent("hc_post", "hc_post", "other", "hc post", forward_order=3),
            BlockComponent("hc_pre", "hc_pre", "other", "hc pre", forward_order=4),
            BlockComponent("ffn_norm", "RMSNorm", "norm", "RMSNorm", forward_order=5),
            BlockComponent("ffn", "MoE", "ffn", "MoE", forward_order=6),
            BlockComponent("hc_post", "hc_post", "other", "hc post", forward_order=7),
        ]
    )
    assert [(norm.attr_name, module.attr_name) for norm, module in pairs] == [
        ("attn_norm", "attn"),
        ("ffn_norm", "ffn"),
    ]


def _drawn_block_boxes(spec, tmp_path: Path, name: str):
    """Render a diagram and return the (label, bottom, top) of every box drawn."""
    import visualizer.render as render_module

    boxes: list[tuple[str, float, float]] = []
    original = render_module._draw_box

    def record(ax, node, **kwargs):
        boxes.append((node.label, node.bottom, node.top))
        return original(ax, node, **kwargs)

    render_module._draw_box = record
    try:
        render_module.render_diagram(spec, tmp_path / name)
    finally:
        render_module._draw_box = original
    return boxes


def test_block_interior_falls_back_to_model_components_when_pairing_fails(tmp_path: Path):
    from visualizer.blocks import BlockComponent, collect_norm_module_pairs

    config_path = FIXTURES / "llama_like" / "config.json"
    spec = load_architecture(config_path, name="Test Llama", analyze_code=False)
    # Roles the classifier failed to recognise: nothing pairs, but the modules are known.
    spec.block_components = [
        BlockComponent("hc_pre", "hc_pre", "other", "hc pre", forward_order=0),
        BlockComponent("attn", "Attention", "other", "Attn", forward_order=1),
        BlockComponent("hc_post", "hc_post", "other", "hc post", forward_order=2),
    ]
    assert collect_norm_module_pairs(spec.block_components) == []

    labels = [label for label, _bottom, _top in _drawn_block_boxes(spec, tmp_path, "fallback.svg")]
    for expected in ("hc pre", "Attn", "hc post"):
        assert expected in labels, f"{expected} missing; block drew empty instead of the model"


def test_block_spine_connectors_span_the_frame_boundary(tmp_path: Path):
    import visualizer.render as render_module

    config_path = FIXTURES / "llama_like" / "config.json"
    spec = load_architecture(config_path, name="Test Llama", analyze_code=False)

    boxes: list[tuple[str, float, float, float]] = []
    segments: list[tuple[float, float, float, float]] = []
    original_box = render_module._draw_box
    original_line = render_module._line
    original_path = render_module._draw_path

    def record_box(ax, node, **kwargs):
        boxes.append((node.label, node.cx, node.bottom, node.top))
        return original_box(ax, node, **kwargs)

    def record_line(ax, x1, y1, x2, y2, **kwargs):
        segments.append((x1, y1, x2, y2))
        return original_line(ax, x1, y1, x2, y2, **kwargs)

    def record_path(ax, points, **kwargs):
        for start, end in zip(points, points[1:]):
            segments.append((*start, *end))
        return original_path(ax, points, **kwargs)

    render_module._draw_box = record_box
    render_module._line = record_line
    render_module._draw_path = record_path
    try:
        render_module.render_diagram(spec, tmp_path / "spine.svg")
    finally:
        render_module._draw_box = original_box
        render_module._line = original_line
        render_module._draw_path = original_path

    spine_cx = next(box[1] for box in boxes if box[0].startswith("Tokenized"))
    column = sorted(
        (box for box in boxes if abs(box[1] - spine_cx) < 0.08),
        key=lambda box: -box[3],
    )
    verticals = [
        (min(y1, y2), max(y1, y2))
        for x1, y1, x2, y2 in segments
        if abs(x1 - x2) < 1e-6 and abs(x1 - spine_cx) < 0.08
    ]
    # The lane band above a residual Add tile legitimately breaks the spine: the
    # module output jogs to its own top port there when it is not tile-centered.
    merge_span = render_module.RESIDUAL_ADD_ENTRY_BAND + render_module.RESIDUAL_ADD_EXIT_GAP

    for upper, lower in zip(column, column[1:]):
        gap_top, gap_bottom = upper[2], lower[3]
        if gap_top - gap_bottom <= 1e-9:
            continue
        spans = sorted(seg for seg in verticals if seg[1] > gap_bottom and seg[0] < gap_top)
        cursor = gap_bottom
        holes: list[tuple[float, float]] = []
        for low, high in spans:
            if low > cursor + 1e-6:
                holes.append((cursor, low))
            cursor = max(cursor, high)
        if cursor < gap_top - 1e-6:
            holes.append((cursor, gap_top))
        unexplained = [hole for hole in holes if hole[1] - hole[0] > merge_span + 1e-3]
        assert not unexplained, (
            f"spine breaks between {upper[0]!r} and {lower[0]!r}: {unexplained}"
        )


def test_basic_rmsnorm_has_no_box_sublabel():
    from visualizer.block_tree import BlockNode
    from visualizer.sizing import block_sublabel

    norm = BlockNode(
        attr_name="input_layernorm",
        class_name="KimiRMSNorm",
        role="norm",
        label="RMSNorm",
        is_basic=False,
    )
    assert block_sublabel(norm) is None


def test_parse_llama_like_config():
    config_path = FIXTURES / "llama_like" / "config.json"
    spec = load_architecture(config_path, name="Test Llama", analyze_code=False)
    assert spec.attention_type == "GQA"
    assert spec.decoder_type == "Dense"
    assert spec.ffn_type == "SwiGLU"
    assert spec.num_hidden_layers == 16
    assert spec.kv_cache_per_token_bf16 is not None


def test_parse_moe_config():
    config = {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "vocab_size": 151936,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "hidden_act": "silu",
    }
    spec = parse_architecture(config, "test", name="Qwen3 MoE")
    assert spec.decoder_type == "Sparse MoE"
    assert spec.num_experts == 128
    assert spec.active_params_hint is not None


def test_render_diagram(tmp_path: Path):
    config_path = FIXTURES / "llama_like" / "config.json"
    spec = load_architecture(config_path, name="Test Llama", analyze_code=False)
    out = render_diagram(spec, tmp_path / "diagram.svg")
    assert out.exists()
    assert out.suffix == ".svg"
    assert out.stat().st_size > 1_000


def test_stroke_white_text_in_svg():
    from visualizer.render import (
        WHITE_TEXT_OUTLINE_PX,
        _finalize_svg_styling,
        _stroke_white_text_in_svg,
        white_text_has_black_outline_in_svg,
    )

    svg = (
        '<g style="fill: #ffffff" transform="translate(1 2) scale(0.076 -0.076)">'
        '<use xlink:href="#A"/>'
        '<use xlink:href="#B" transform="translate(10 0)"/>'
        "</g>"
        '<path style="fill: #ffffff; stroke: #d0d0d0"/>'
        '<path style="fill: #bdc3c7; stroke: #bdc3c7; stroke-width: 1.2"/>'
    )
    stroked = _stroke_white_text_in_svg(svg)
    expected_width = f"{WHITE_TEXT_OUTLINE_PX / 0.076:.4f}"
    assert expected_width in stroked
    assert stroked.count('stroke: #000000') == 2
    assert 'style="fill: #ffffff; stroke: #d0d0d0"' in stroked
    assert white_text_has_black_outline_in_svg(stroked)

    finalized = _finalize_svg_styling(svg)
    assert "fill: #bdc3c7; stroke: #000000;" in finalized
    assert "fill: #bdc3c7; stroke: #bdc3c7;" not in finalized


def test_render_diagram_white_text_has_black_outline(tmp_path: Path):
    """Exported SVGs must keep the 2px black outline on white tile labels."""
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.render import white_text_has_black_outline_in_svg

    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    out = render_diagram(spec, tmp_path / "white_text_stroke.svg", detailed=True)
    svg = out.read_text(encoding="utf-8")
    assert white_text_has_black_outline_in_svg(svg)
    assert "paint-order: stroke fill" in svg
    assert "stroke: #000000" in svg


def test_ast_custom_decoder_layer():
    source = (FIXTURES / "custom_model" / "modeling_custom.py").read_text(encoding="utf-8")
    analysis = analyze_source(source, filename="modeling_custom.py")

    assert analysis.decoder_class == "CustomDecoderLayer"
    assert analysis.attention_type == "MLA"
    assert analysis.decoder_type == "Sparse MoE"
    assert analysis.forward_sequence == [
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "block_sparse_moe",
    ]
    assert "CustomSharedExpertMoE" in {comp.class_name for comp in analysis.block_components}


def test_load_architecture_with_local_modeling(tmp_path: Path):
    fixture_dir = FIXTURES / "custom_model"
    spec = load_architecture(fixture_dir, name="Custom MLA MoE")

    assert spec.decoder_class == "CustomDecoderLayer"
    assert spec.attention_type == "MLA"
    assert spec.decoder_type == "Sparse MoE"
    assert len(spec.block_components) >= 4
    assert spec.forward_sequence[1] == "self_attn"

    out = render_diagram(spec, tmp_path / "custom.png")
    assert out.exists()


def test_dump_ast_contains_decoder_class():
    source = (FIXTURES / "custom_model" / "modeling_custom.py").read_text(encoding="utf-8")
    ast_dump = dump_ast(source, filename="modeling_custom.py")
    assert "CustomDecoderLayer" in ast_dump
    assert "CustomLatentAttention" in ast_dump


def _svg_patch_y_range(svg_text: str, style: str) -> tuple[float, float]:
    import re

    pattern = rf'<path d="([^"]+)"[^>]*style="[^"]*{re.escape(style)}'
    match = re.search(pattern, svg_text, flags=re.DOTALL)
    assert match is not None, f"Missing SVG patch with style {style!r}"
    ys = [float(y) for y in re.findall(r"L [\d.]+ ([\d.]+)", match.group(1))]
    assert ys, f"Could not parse SVG path coordinates for style {style!r}"
    return min(ys), max(ys)


def _svg_patch_x_range(svg_text: str, style: str) -> tuple[float, float]:
    import re

    pattern = rf'<path d="([^"]+)"[^>]*style="[^"]*{re.escape(style)}'
    match = re.search(pattern, svg_text, flags=re.DOTALL)
    assert match is not None, f"Missing SVG patch with style {style!r}"
    coords = re.findall(r"[MLQ] ([\d.]+) ([\d.]+)", match.group(1))
    xs = [float(x) for x, _ in coords]
    assert xs, f"Could not parse SVG path x coordinates for style {style!r}"
    return min(xs), max(xs)


def test_fact_sheet_sits_to_the_right_of_transformer_block(tmp_path: Path):
    fixture_dir = FIXTURES / "custom_model"
    spec = load_architecture(fixture_dir, name="Custom MLA MoE")
    pairs = _collect_sublayer_pairs(_ordered_block_components(spec))
    assert len(pairs) >= 2

    out = render_diagram(spec, tmp_path / "custom.svg")
    svg = out.read_text(encoding="utf-8")
    block_left, block_right = _svg_patch_x_range(svg, "fill: #fff5f4; stroke: #c0392b")
    fact_left, _ = _svg_patch_x_range(svg, "fill: #ffffff; stroke: #d0d0d0")

    assert block_left < block_right
    assert fact_left > block_right
