"""Render the chunk_kda KernelPipeline in isolation as SVG."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

KIMI_CODE_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
)
OUTPUT_SVG = Path(__file__).resolve().parent.parent / "chunk_kda_pipeline.svg"


def _load_chunk_kda_pipeline():
    from visualizer.ast_analyze import analyze_source
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node

    if not KIMI_CODE_PATH.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    analysis = analyze_source(KIMI_CODE_PATH.read_text(), filename="modeling_kimi_linear.py")
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    pipeline = next(child for child in attn.children if child.class_name == "KernelPipeline")
    return pipeline, basic


def test_chunk_kda_pipeline_renders_svg():
    """Render only the chunk_kda pipeline contents to a standalone SVG."""
    import matplotlib.pyplot as plt

    from visualizer.render import (
        COLORS,
        DIAGRAM_LEFT_MARGIN,
        DiagramLayout,
        _finalize_svg_styling,
        _fit_figure_to_content,
        _render_laid_out_computation_graph,
    )

    pipeline, basic = _load_chunk_kda_pipeline()
    detail_min_left = DIAGRAM_LEFT_MARGIN + 0.05
    block_w = 18.0
    cx = detail_min_left + block_w / 2
    top_y = 12.0

    fig, ax = plt.subplots(figsize=(16, 13), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor(COLORS["bg"])
    ax.axis("off")
    try:
        layout = DiagramLayout()
        _render_laid_out_computation_graph(
            layout,
            ax,
            pipeline,
            cx=cx,
            top_y=top_y,
            block_w=block_w,
            include_input=False,
            draw_section_frame=True,
            root_frame_label=pipeline.label,
            min_left=detail_min_left,
            basic_ops=basic,
        )
        _fit_figure_to_content(ax, fig, margin=0.35)
        OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            OUTPUT_SVG,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor=COLORS["bg"],
        )
    finally:
        plt.close(fig)

    svg = OUTPUT_SVG.read_text(encoding="utf-8")
    OUTPUT_SVG.write_text(_finalize_svg_styling(svg), encoding="utf-8")
    svg = OUTPUT_SVG.read_text(encoding="utf-8")

    assert OUTPUT_SVG.exists()
    assert OUTPUT_SVG.stat().st_size > 5000
    for label in ("q", "k", "v", "beta"):
        assert f"<!-- {label} -->" in svg, f"missing pipeline block {label!r}"
    assert "<!-- Intra-chunk WY -->" in svg or "<!-- chunk_kda_fwd_intra -->" in svg

    dashed_flow = re.findall(
        rf'style="[^"]*stroke-dasharray[^"]*stroke: {re.escape(COLORS["flow"])}',
        svg,
    )
    assert not dashed_flow, f"found {len(dashed_flow)} dashed flow connectors in SVG"
    dashed_frames = re.findall(
        rf'style="[^"]*stroke-dasharray[^"]*stroke: {re.escape(COLORS["detail_border"])}',
        svg,
    )
    assert dashed_frames, "expected dashed strokes around expanded inline frames"


def test_q_tensor_port_links_to_l2norm_entry():
    """q must feed the l2norm_fwd entry (Sum), not the tail multiply at index 0."""
    from visualizer.computation_graph import build_computation_graph

    pipeline, _basic = _load_chunk_kda_pipeline()
    graph = build_computation_graph(pipeline)
    q_idx = next(i for i, node in enumerate(graph.nodes) if node.label == "q")
    target_idx = next(t for s, t in graph.links if s == q_idx)
    assert graph.nodes[target_idx].label == "Sum"
    assert target_idx == 0


def test_q_tensor_port_sits_above_l2norm_frame():
    """After layout, q sits above the l2norm_fwd frame with a downward connector into Sum."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    from visualizer.computation_graph import (
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
        _estimate_graph_height,
    )
    from visualizer.render import (
        COLORS,
        DIAGRAM_LEFT_MARGIN,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _connector_points_for_link,
    )
    from visualizer.render_validate import finalize_detail_layout

    pipeline, basic = _load_chunk_kda_pipeline()
    graph = build_computation_graph(pipeline)
    fig, ax = plt.subplots(figsize=(16, 13))
    try:
        measure_graph_node_sizes(ax, graph)
        positions, _ = layout_computation_graph(
            graph,
            cx=5.0,
            top_y=12.0,
            block_w=18.0,
            block_h=_estimate_graph_height(graph),
        )
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=5.0,
            top_y=12.0,
            detail_fill=COLORS["detail_fill"],
            min_left=DIAGRAM_LEFT_MARGIN + 0.05,
        )
        q_idx = next(i for i, node in enumerate(graph.nodes) if node.label == "q")
        sum_idx = next(i for i, node in enumerate(graph.nodes) if node.label == "Sum")
        q_pos = positions[q_idx]
        sum_pos = positions[sum_idx]
        assert q_pos.bottom > sum_pos.top_y + 0.05, "q port should sit above l2norm Sum entry"
        assert abs(q_pos.cx - sum_pos.cx) < 0.05, "q port should align with l2norm column"

        plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
        anchors = _anchors_from_detail_plan(positions, plan)
        incoming = defaultdict(list)
        for src, tgt in graph.links:
            incoming[tgt].append((src, tgt))
        points = _connector_points_for_link(
            graph=graph,
            positions=positions,
            anchors=anchors,
            src=q_idx,
            tgt=sum_idx,
            link_key=(q_idx, sum_idx),
            incoming=incoming,
            label_obstacles=[],
            target_bus={},
            source_bus={},
            merge_entry_x={},
            merge_link_bus={},
            input_index=None,
        )
        assert points is not None and len(points) >= 2
        assert points[0][1] > points[-1][1], "q connector should descend into l2norm Sum"
    finally:
        plt.close(fig)


def test_cumsum_fanout_routes_to_intra_and_h_without_crossing():
    """CumSum outputs tee vertically, enter h from the right, and avoid box overlap."""
    import matplotlib.pyplot as plt

    from visualizer.render import (
        _connector_source_bottom_exit_y,
        _segment_orientation,
    )

    fig, graph, anchors, _plan, incoming, outgoing, target_bus, source_bus, _merge_link_bus, link_paths = (
        _chunk_kda_pipeline_link_paths()
    )
    try:
        cumsum_idx = next(i for i, node in enumerate(graph.nodes) if node.label == "CumSum")
        intra_idx = next(
            i for i, node in enumerate(graph.nodes) if node.label == "chunk_kda_fwd_intra"
        )
        h_idx = next(
            i
            for i, node in enumerate(graph.nodes)
            if node.label == "chunk_gated_delta_rule_fwd_h"
        )
        cumsum = anchors[cumsum_idx]
        h = anchors[h_idx]

        assert cumsum_idx in source_bus

        intra_points = link_paths[(cumsum_idx, intra_idx)]
        h_points = link_paths[(cumsum_idx, h_idx)]

        y_exit = _connector_source_bottom_exit_y(cumsum)
        assert abs(intra_points[0][1] - y_exit) < 1e-6
        assert abs(h_points[0][1] - y_exit) < 1e-6
        assert abs(h_points[1][0] - h_points[0][0]) < 1e-6, "CumSum->h must leave vertically, not sideways"
        assert intra_points[-1][0] != h_points[-1][0], "fan-out legs must land on separate top-entry ports"

        from visualizer.render import (
            _connector_path_respects_tee_before_bus_join,
            _source_fanout_splits_before_target_bus,
        )

        assert _source_fanout_splits_before_target_bus(
            graph, cumsum_idx, outgoing, target_bus
        )
        tee_y = source_bus[cumsum_idx]
        merge_y = target_bus[intra_idx]
        y_exit = _connector_source_bottom_exit_y(cumsum)
        expected_tee = y_exit - (y_exit - merge_y) * 0.5
        assert abs(tee_y - expected_tee) < 1e-6, "short fan-out tee should sit midway to merge bus"
        from visualizer.render import CONNECTOR_OBSTACLE_MARGIN

        assert abs(intra_points[1][1] - tee_y) < CONNECTOR_OBSTACLE_MARGIN + 1e-6, (
            "CumSum fan-out must tee before merge bus"
        )
        assert abs(h_points[1][1] - tee_y) < CONNECTOR_OBSTACLE_MARGIN + 1e-6
        assert abs(intra_points[1][0] - intra_points[2][0]) < 1e-6, "merge-bus leg stays vertical through tee"
        assert abs(h_points[2][0] - h_points[1][0]) > 0.06, "branch leg joins tee bus horizontally"
        assert abs(intra_points[2][1] - merge_y) < 1e-6
        assert abs(h_points[1][1] - intra_points[2][1]) > 0.05, "h branch must leave before intra bus"
        path_tee_y = intra_points[1][1]
        path_merge_y = intra_points[2][1]
        assert _connector_path_respects_tee_before_bus_join(
            intra_points,
            source=cumsum,
            tee_y=path_tee_y,
            merge_bus_y=path_merge_y,
        )

        from visualizer.render import _collect_connector_join_points

        join_points = _collect_connector_join_points(
            link_paths,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=_merge_link_bus,
            anchors=anchors,
            graph=graph,
            outgoing=outgoing,
        )
        assert not any(
            abs(jx - cumsum.cx) < 0.03 and abs(jy - path_tee_y) < 0.03
            for jx, jy in join_points
        ), "fan-out split tee must not get a bus junction dot"

        for index in range(len(h_points) - 1):
            orientation = _segment_orientation(
                h_points[index][0],
                h_points[index][1],
                h_points[index + 1][0],
                h_points[index + 1][1],
            )
            assert orientation is not None

        assert abs(h_points[-1][1] - h.top) < 1e-6, "CumSum->h must enter on the top edge"
        assert h_points[-1][0] > h.cx, "CumSum->h must enter to the right of center"

        intra_h = link_paths[(intra_idx, h_idx)]
        assert len(intra_h) == 2
        assert abs(intra_h[0][0] - intra_h[1][0]) < 1e-6, "intra->h must drop straight down"
        assert abs(intra_h[-1][0] - h.cx) < 1e-6
        assert abs(intra_h[-1][1] - h.top) < 1e-6
        assert h_points[-1][0] > intra_h[-1][0]
    finally:
        plt.close(fig)


def test_cumsum_connectors_avoid_intra_chunk_wy_box():
    """CumSum fan-out must not draw through the Intra-chunk WY tile."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    from visualizer.computation_graph import (
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
        _dock_single_consumer_tensor_ports,
        _estimate_graph_height,
        _graph_has_tensor_ports,
    )
    from visualizer.render import (
        COLORS,
        CONNECTOR_OBSTACLE_MARGIN,
        DIAGRAM_LEFT_MARGIN,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _collect_detail_link_paths,
        _compute_detail_connector_buses,
        _path_hits_obstacles,
    )
    from visualizer.render_validate import finalize_detail_layout

    pipeline, _basic = _load_chunk_kda_pipeline()
    graph = build_computation_graph(pipeline)
    fig, ax = plt.subplots(figsize=(16, 13))
    try:
        measure_graph_node_sizes(ax, graph)
        cx = DIAGRAM_LEFT_MARGIN + 0.05 + 9.0
        positions, links = layout_computation_graph(
            graph,
            cx=cx,
            top_y=12.0,
            block_w=18.0,
            block_h=_estimate_graph_height(graph),
        )
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=cx,
            top_y=12.0,
            detail_fill=COLORS["detail_fill"],
            min_left=DIAGRAM_LEFT_MARGIN + 0.05,
        )
        if _graph_has_tensor_ports(graph):
            _dock_single_consumer_tensor_ports(positions, graph)

        plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
        anchors = _anchors_from_detail_plan(positions, plan)
        label_obstacles = plan.label_obstacles
        incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
        outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for src, tgt in links:
            incoming[tgt].append((src, tgt))
            outgoing[src].append((src, tgt))

        target_bus, source_bus, merge_entry_x, merge_link_bus = _compute_detail_connector_buses(
            graph,
            positions,
            anchors,
            incoming,
            outgoing,
            label_obstacles,
        )

        link_paths = _collect_detail_link_paths(
            graph=graph,
            links=links,
            positions=positions,
            anchors=anchors,
            incoming=incoming,
            label_obstacles=label_obstacles,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_entry_x=merge_entry_x,
            merge_link_bus=merge_link_bus,
            input_index=None,
        )

        cumsum_idx = next(i for i, node in enumerate(graph.nodes) if node.label == "CumSum")
        intra_idx = next(
            i for i, node in enumerate(graph.nodes) if node.label == "chunk_kda_fwd_intra"
        )
        h_idx = next(
            i
            for i, node in enumerate(graph.nodes)
            if node.label == "chunk_gated_delta_rule_fwd_h"
        )
        intra_anchor = anchors[intra_idx]

        intra_points = link_paths[(cumsum_idx, intra_idx)]
        assert len(intra_points) >= 2
        assert not _path_hits_obstacles(
            intra_points[:-1],
            [intra_anchor],
            margin=CONNECTOR_OBSTACLE_MARGIN,
        ), "CumSum bus should clear Intra-chunk WY before entry"

        h_points = link_paths[(cumsum_idx, h_idx)]
        assert not _path_hits_obstacles(
            h_points,
            [intra_anchor],
            margin=CONNECTOR_OBSTACLE_MARGIN,
        ), "CumSum fan-out to h must not cut through Intra-chunk WY"
    finally:
        plt.close(fig)


def test_inline_frame_skip_connectors_stay_inside_submodule():
    """Skip/residual operand links must route inside their dotted submodule frame."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    from visualizer.computation_graph import (
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
        _dock_single_consumer_tensor_ports,
        _estimate_graph_height,
        _graph_has_tensor_ports,
    )
    from visualizer.render import (
        COLORS,
        CONNECTOR_OBSTACLE_MARGIN,
        DIAGRAM_LEFT_MARGIN,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _collect_detail_link_paths,
        _compute_detail_connector_buses,
        _inline_frame_draw_bounds,
        _path_stays_inside_bounds,
    )
    from visualizer.render_validate import finalize_detail_layout

    pipeline, _basic = _load_chunk_kda_pipeline()
    graph = build_computation_graph(pipeline)
    fig, ax = plt.subplots(figsize=(16, 13))
    try:
        measure_graph_node_sizes(ax, graph)
        cx = DIAGRAM_LEFT_MARGIN + 0.05 + 9.0
        positions, links = layout_computation_graph(
            graph,
            cx=cx,
            top_y=12.0,
            block_w=18.0,
            block_h=_estimate_graph_height(graph),
        )
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=cx,
            top_y=12.0,
            detail_fill=COLORS["detail_fill"],
            min_left=DIAGRAM_LEFT_MARGIN + 0.05,
        )
        if _graph_has_tensor_ports(graph):
            _dock_single_consumer_tensor_ports(positions, graph)

        plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
        anchors = _anchors_from_detail_plan(positions, plan)
        incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
        outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for src, tgt in links:
            incoming[tgt].append((src, tgt))
            outgoing[src].append((src, tgt))

        target_bus, source_bus, merge_entry_x, merge_link_bus = _compute_detail_connector_buses(
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
            target_bus=target_bus,
            source_bus=source_bus,
            merge_entry_x=merge_entry_x,
            merge_link_bus=merge_link_bus,
            input_index=None,
        )

        offenders = []
        for src, tgt in graph.inline_binary_operand_links:
            frame = next(
                (
                    frame
                    for frame in graph.inline_frames
                    if src in frame.node_indices and tgt in frame.node_indices
                ),
                None,
            )
            if frame is None:
                continue
            bounds = _inline_frame_draw_bounds(frame, positions, graph)
            points = link_paths[(src, tgt)]
            if not _path_stays_inside_bounds(
                points, bounds, margin=CONNECTOR_OBSTACLE_MARGIN
            ):
                offenders.append((graph.nodes[src].label, graph.nodes[tgt].label, frame.frame_id))

        assert not offenders, f"skip connectors escaped submodule frames: {offenders}"

        frames_with_skips = {
            frame.frame_id
            for frame in graph.inline_frames
            if any(
                src in frame.node_indices and tgt in frame.node_indices
                for src, tgt in graph.inline_binary_operand_links
            )
        }
        assert frames_with_skips, "expected at least one inline frame with skip connectors"
    finally:
        plt.close(fig)


def test_chunk_kda_pipeline_connectors_avoid_attached_boxes():
    """Connectors must not pass through their source or target tiles."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    from visualizer.computation_graph import (
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
        _dock_single_consumer_tensor_ports,
        _estimate_graph_height,
        _graph_has_tensor_ports,
    )
    from visualizer.render import (
        COLORS,
        DIAGRAM_LEFT_MARGIN,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _collect_detail_link_paths,
        _compute_detail_connector_buses,
        _path_penetrates_attached_boxes,
    )
    from visualizer.render_validate import finalize_detail_layout

    pipeline, _basic = _load_chunk_kda_pipeline()
    graph = build_computation_graph(pipeline)
    fig, ax = plt.subplots(figsize=(16, 13))
    try:
        measure_graph_node_sizes(ax, graph)
        cx = DIAGRAM_LEFT_MARGIN + 0.05 + 9.0
        positions, links = layout_computation_graph(
            graph,
            cx=cx,
            top_y=12.0,
            block_w=18.0,
            block_h=_estimate_graph_height(graph),
        )
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=cx,
            top_y=12.0,
            detail_fill=COLORS["detail_fill"],
            min_left=DIAGRAM_LEFT_MARGIN + 0.05,
        )
        if _graph_has_tensor_ports(graph):
            _dock_single_consumer_tensor_ports(positions, graph)

        plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
        anchors = _anchors_from_detail_plan(positions, plan)
        incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
        outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for src, tgt in links:
            incoming[tgt].append((src, tgt))
            outgoing[src].append((src, tgt))

        target_bus, source_bus, merge_entry_x, merge_link_bus = _compute_detail_connector_buses(
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
            target_bus=target_bus,
            source_bus=source_bus,
            merge_entry_x=merge_entry_x,
            merge_link_bus=merge_link_bus,
            input_index=None,
        )

        offenders = []
        for (src, tgt), points in link_paths.items():
            if len(points) < 2:
                continue
            source = anchors.get(src)
            target = anchors.get(tgt)
            if source is None or target is None:
                continue
            if _path_penetrates_attached_boxes(points, source, target):
                offenders.append((graph.nodes[src].label, graph.nodes[tgt].label))
        assert not offenders, f"connectors crossed attached tiles: {offenders}"
    finally:
        plt.close(fig)


def _chunk_kda_pipeline_link_paths():
    """Build positioned link paths for the chunk_kda KernelPipeline."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    from visualizer.computation_graph import (
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
        _dock_single_consumer_tensor_ports,
        _estimate_graph_height,
        _graph_has_tensor_ports,
    )
    from visualizer.render import (
        COLORS,
        DIAGRAM_LEFT_MARGIN,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _collect_detail_link_paths,
        _compute_detail_connector_buses,
    )
    from visualizer.render_validate import finalize_detail_layout

    pipeline, _basic = _load_chunk_kda_pipeline()
    graph = build_computation_graph(pipeline)
    fig, ax = plt.subplots(figsize=(16, 13))
    measure_graph_node_sizes(ax, graph)
    cx = DIAGRAM_LEFT_MARGIN + 0.05 + 9.0
    positions, links = layout_computation_graph(
        graph,
        cx=cx,
        top_y=12.0,
        block_w=18.0,
        block_h=_estimate_graph_height(graph),
    )
    finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=None,
        cx=cx,
        top_y=12.0,
        detail_fill=COLORS["detail_fill"],
        min_left=DIAGRAM_LEFT_MARGIN + 0.05,
    )
    if _graph_has_tensor_ports(graph):
        _dock_single_consumer_tensor_ports(positions, graph)
    plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
    anchors = _anchors_from_detail_plan(positions, plan)
    incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
    outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for src, tgt in links:
        incoming[tgt].append((src, tgt))
        outgoing[src].append((src, tgt))

    target_bus, source_bus, merge_entry_x, merge_link_bus = _compute_detail_connector_buses(
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
        target_bus=target_bus,
        source_bus=source_bus,
        merge_entry_x=merge_entry_x,
        merge_link_bus=merge_link_bus,
        input_index=None,
    )
    return (
        fig,
        graph,
        anchors,
        plan,
        incoming,
        outgoing,
        target_bus,
        source_bus,
        merge_link_bus,
        link_paths,
        positions,
        links,
    )


def test_v_tensor_port_connector_to_intra_bus_is_solid():
    """The v feed into the intra-chunk merge bus must use a solid connector."""
    import matplotlib.pyplot as plt

    from visualizer.render import _assert_detail_connector_linestyles_are_solid, _detail_connector_linestyle

    (
        fig,
        graph,
        anchors,
        _plan,
        _incoming,
        _outgoing,
        _target_bus,
        _source_bus,
        _merge_link_bus,
        link_paths,
        positions,
        links,
    ) = _chunk_kda_pipeline_link_paths()
    try:
        _assert_detail_connector_linestyles_are_solid(
            graph,
            links=links,
            positions=positions,
            anchors=anchors,
        )
        v_idx = next(index for index, node in enumerate(graph.nodes) if node.label == "v")
        intra_idx = next(
            index for index, node in enumerate(graph.nodes) if node.label == "chunk_kda_fwd_intra"
        )
        assert (v_idx, intra_idx) in link_paths
        style = _detail_connector_linestyle(
            graph,
            src=v_idx,
            positions=positions,
            source=anchors[v_idx],
            target=anchors[intra_idx],
        )
        assert style == "solid"
    finally:
        plt.close(fig)


def test_chunk_kda_pipeline_connectors_attach_flush_to_box_borders():
    """Connector endpoints should meet tile borders exactly, without gaps or overlap."""
    import matplotlib.pyplot as plt

    from visualizer.render import (
        CONNECTOR_EXIT_STUB,
        _connector_source_bottom_exit_y,
        _connector_target_side_entry_y,
        _connector_target_top_entry_y,
    )

    fig, graph, anchors, _plan, _incoming, _outgoing, _target_bus, _source_bus, _merge_link_bus, link_paths = (
        _chunk_kda_pipeline_link_paths()
    )
    try:
        side_links = set(graph.inline_binary_operand_links)
        for (src, tgt), points in link_paths.items():
            source = anchors[src]
            target = anchors[tgt]
            start_x, start_y = points[0]
            end_x, end_y = points[-1]
            if (src, tgt) in side_links:
                y_stub = _connector_source_bottom_exit_y(source) - CONNECTOR_EXIT_STUB
                assert start_x == source.cx
                assert start_y == y_stub
                assert end_y == _connector_target_side_entry_y(target)
                assert end_x in {target.left, target.right}
            elif end_x in {target.left, target.right} and abs(
                end_y - _connector_target_side_entry_y(target)
            ) < 1e-6:
                assert start_y == _connector_source_bottom_exit_y(source)
                assert end_x in {target.left, target.right}
            else:
                assert start_y == _connector_source_bottom_exit_y(source)
                if len(points) == 2:
                    assert start_x == source.cx
                assert end_y == _connector_target_top_entry_y(target)
    finally:
        plt.close(fig)


def test_chunk_kda_pipeline_connectors_do_not_overlap():
    """Rendered connectors must not share non-bus channels."""
    import matplotlib.pyplot as plt

    from visualizer.render import _find_connector_path_overlaps

    fig, graph, anchors, _plan, incoming, outgoing, target_bus, source_bus, merge_link_bus, link_paths = (
        _chunk_kda_pipeline_link_paths()
    )
    try:
        overlaps = _find_connector_path_overlaps(
            link_paths,
            incoming=incoming,
            outgoing=outgoing,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            anchors=anchors,
            graph=graph,
        )
        assert not overlaps, (
            "connector overlaps: "
            + ", ".join(
                f"{graph.nodes[a[0]].label}->{graph.nodes[a[1]].label}|"
                f"{graph.nodes[b[0]].label}->{graph.nodes[b[1]].label}"
                for a, b in overlaps
            )
        )
    finally:
        plt.close(fig)


def test_small_fan_in_blocks_skip_shared_target_bus():
    """Blocks with three or fewer inputs route directly without a merge bus."""
    import matplotlib.pyplot as plt

    fig, graph, _anchors, _plan, _incoming, _outgoing, target_bus, _source_bus, _merge_link_bus, link_paths = (
        _chunk_kda_pipeline_link_paths()
    )
    try:
        h_idx = next(
            index
            for index, node in enumerate(graph.nodes)
            if node.label == "chunk_gated_delta_rule_fwd_h"
        )
        intra_idx = next(
            index
            for index, node in enumerate(graph.nodes)
            if node.label == "chunk_kda_fwd_intra"
        )
        assert h_idx not in target_bus
        assert intra_idx in target_bus

        entry_x = {link_paths[(src, h_idx)][-1][0] for src, tgt in link_paths if tgt == h_idx}
        assert len(entry_x) == 2
        assert max(entry_x) - min(entry_x) > 0.05, "no-bus fan-in should spread entry points"
    finally:
        plt.close(fig)


def test_l2norm_fwd_bypass_connectors_are_separated():
    """l2norm_fwd frames with two bypasses route on left and right gutters."""
    import matplotlib.pyplot as plt

    from visualizer.computation_graph import _inline_frame_vertical_gap
    from visualizer.render import PARALLEL_CONNECTOR_CHANNEL_GAP
    from visualizer.sizing import min_vertical_block_gap

    fig, graph, anchors, _plan, _incoming, _outgoing, _target_bus, _source_bus, _merge_link_bus, link_paths = (
        _chunk_kda_pipeline_link_paths()
    )
    try:
        for frame in graph.inline_frames:
            if "l2norm_fwd" not in frame.frame_id:
                continue
            bypass_links = [
                (src, tgt)
                for src, tgt in graph.inline_binary_operand_links
                if src in frame.node_indices and tgt in frame.node_indices
            ]
            assert len(bypass_links) >= 2, frame.frame_id
            gap = _inline_frame_vertical_gap(graph, frame)
            assert gap >= min_vertical_block_gap() + 2 * 0.05

            horizontals: list[float] = []
            vertical_bus_x: list[float] = []
            column_cx = sum(anchors[index].cx for index in frame.node_indices) / len(frame.node_indices)
            for src, tgt in bypass_links:
                points = link_paths[(src, tgt)]
                for (x1, y1), (x2, y2) in zip(points, points[1:]):
                    if abs(y1 - y2) < 1e-6 and abs(x1 - x2) > 0.06:
                        if abs(x1 - column_cx) < 0.06 or abs(x2 - column_cx) < 0.06:
                            horizontals.append(y1)
                    if abs(x1 - x2) < 1e-6:
                        vertical_bus_x.append(x1)
            for left, right in zip(sorted(horizontals), sorted(horizontals)[1:]):
                assert abs(left - right) >= PARALLEL_CONNECTOR_CHANNEL_GAP / 2, (
                    f"{frame.frame_id} bypass buses too close: {left:.3f} vs {right:.3f}"
                )
            assert any(x < column_cx for x in vertical_bus_x), (
                f"{frame.frame_id} expected a left-gutter bypass"
            )
            assert any(x > column_cx for x in vertical_bus_x), (
                f"{frame.frame_id} expected a right-gutter bypass"
            )
    finally:
        plt.close(fig)


def test_l2norm_fwd_main_chain_uses_straight_vertical_connectors():
    """Spine links inside l2norm_fwd frames should be simple vertical lines."""
    import matplotlib.pyplot as plt

    fig, graph, _anchors, _plan, _incoming, _outgoing, _target_bus, _source_bus, _merge_link_bus, link_paths = (
        _chunk_kda_pipeline_link_paths()
    )
    try:
        for frame in graph.inline_frames:
            if "l2norm_fwd" not in frame.frame_id:
                continue
            members = set(frame.node_indices)
            for src, tgt in graph.links:
                if src not in members or tgt not in members:
                    continue
                if (src, tgt) in graph.inline_binary_operand_links:
                    continue
                points = link_paths[(src, tgt)]
                assert len(points) == 2, (
                    f"{frame.frame_id} spine {graph.nodes[src].label}->{graph.nodes[tgt].label} "
                    f"should be a straight vertical feed, got {points}"
                )
                assert abs(points[0][0] - points[1][0]) < 0.06
                assert points[0][1] > points[1][1]
    finally:
        plt.close(fig)


def test_fused_beta_sigmoid_frame_has_wider_tile_spacing():
    """Sigmoid and × scale need the same extra spacing as other shortcut frames."""
    import matplotlib.pyplot as plt

    from visualizer.computation_graph import _inline_frame_vertical_gap
    from visualizer.sizing import min_vertical_block_gap

    fig, graph, _anchors, _plan, _incoming, _outgoing, _target_bus, _source_bus, _merge_link_bus, link_paths = (
        _chunk_kda_pipeline_link_paths()
    )
    try:
        frame = next(
            frame for frame in graph.inline_frames if frame.frame_id == "forward_fused_beta_sigmoid_beta"
        )
        gap = _inline_frame_vertical_gap(graph, frame)
        assert gap >= min_vertical_block_gap() + 0.14

        sigmoid_idx, scale_idx = frame.node_indices[:2]
        points = link_paths[(sigmoid_idx, scale_idx)]
        assert len(points) == 2
        assert abs(points[0][0] - points[1][0]) < 0.06
    finally:
        plt.close(fig)


def test_parallel_feeder_frame_exit_stubs_are_shrinkwrapped():
    """l2norm(k) and fused_beta columns compact down to the shared merge-bus corridor."""
    import matplotlib.pyplot as plt

    fig, graph, _anchors, _plan, _incoming, _outgoing, target_bus, _source_bus, _merge_link_bus, link_paths = (
        _chunk_kda_pipeline_link_paths()
    )
    try:
        intra_idx = next(
            index for index, node in enumerate(graph.nodes) if node.label == "chunk_kda_fwd_intra"
        )
        bus_y = target_bus[intra_idx]

        compacted_frames = (
            "forward_l2norm_fwd_q",
            "forward_l2norm_fwd_k",
            "forward_fused_beta_sigmoid_beta",
        )
        for frame_id in compacted_frames:
            frame = next(item for item in graph.inline_frames if item.frame_id == frame_id)
            tail_idx = frame.node_indices[-1]
            points = link_paths[(tail_idx, intra_idx)]
            assert abs(points[0][0] - points[1][0]) < 1e-6 or frame_id == "forward_l2norm_fwd_q"
            if frame_id == "forward_l2norm_fwd_q":
                horizontals = [
                    y1
                    for (x1, y1), (x2, y2) in zip(points, points[1:])
                    if abs(y1 - y2) < 1e-6 and abs(x1 - x2) > 0.06
                ]
                bus_horizontals = [y for y in horizontals if abs(y - bus_y) < 0.03]
                same_column_bus_tee = (
                    len(points) == 3
                    and abs(points[0][0] - points[1][0]) < 1e-6
                    and abs(points[1][1] - bus_y) < 0.03
                )
                assert bus_horizontals or same_column_bus_tee, (
                    f"{frame_id} must tee onto the shared merge bus"
                )
                if same_column_bus_tee:
                    assert points[0][1] - points[1][1] < 0.55, (
                        f"{frame_id} output stub should be short after shrinkwrap"
                    )
                    continue
                gutter_verticals = [
                    abs(y1 - y2)
                    for (x1, y1), (x2, y2) in zip(points, points[1:])
                    if abs(x1 - x2) < 1e-6 and abs(y1 - y2) > 0.06
                ]
                assert max(gutter_verticals) < 1.2, (
                    f"{frame_id} gutter vertical should shrinkwrap toward the merge bus"
                )
                continue
            assert abs(points[1][1] - bus_y) < 0.03, (
                f"{frame_id} should drop straight onto the merge bus"
            )
            assert points[0][1] - points[1][1] < 0.55, (
                f"{frame_id} output stub should be short after shrinkwrap"
            )
    finally:
        plt.close(fig)


def test_frame_exit_connectors_avoid_expanded_box_overlap():
    """Tail exits must route below the dotted frame, not along its bottom edge."""
    import matplotlib.pyplot as plt

    from visualizer.computation_graph import (
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
        _dock_single_consumer_tensor_ports,
        _estimate_graph_height,
        _graph_has_tensor_ports,
    )
    from visualizer.render import (
        COLORS,
        CONNECTOR_EXIT_STUB,
        DIAGRAM_LEFT_MARGIN,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _collect_detail_link_paths,
        _compute_detail_connector_buses,
        _inline_frame_draw_bounds,
        _path_horizontal_segments_overlap_bounds,
    )
    from visualizer.render_validate import finalize_detail_layout

    pipeline, _basic = _load_chunk_kda_pipeline()
    graph = build_computation_graph(pipeline)
    fig, ax = plt.subplots(figsize=(16, 13))
    try:
        measure_graph_node_sizes(ax, graph)
        cx = DIAGRAM_LEFT_MARGIN + 0.05 + 9.0
        positions, links = layout_computation_graph(
            graph,
            cx=cx,
            top_y=12.0,
            block_w=18.0,
            block_h=_estimate_graph_height(graph),
        )
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=cx,
            top_y=12.0,
            detail_fill=COLORS["detail_fill"],
            min_left=DIAGRAM_LEFT_MARGIN + 0.05,
        )
        if _graph_has_tensor_ports(graph):
            _dock_single_consumer_tensor_ports(positions, graph)

        plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
        anchors = _anchors_from_detail_plan(positions, plan)
        incoming = {tgt: [] for tgt in range(len(graph.nodes))}
        outgoing = {src: [] for src in range(len(graph.nodes))}
        for src, tgt in links:
            incoming[tgt].append((src, tgt))
            outgoing[src].append((src, tgt))
        target_bus, source_bus, merge_entry_x, merge_link_bus = _compute_detail_connector_buses(
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
            target_bus=target_bus,
            source_bus=source_bus,
            merge_entry_x=merge_entry_x,
            merge_link_bus=merge_link_bus,
            input_index=None,
        )

        intra_idx = next(
            index for index, node in enumerate(graph.nodes) if node.label == "chunk_kda_fwd_intra"
        )
        offenders = []
        for frame in graph.inline_frames:
            if not frame.node_indices:
                continue
            tail_idx = frame.node_indices[-1]
            link = (tail_idx, intra_idx)
            if link not in link_paths:
                continue
            bounds = _inline_frame_draw_bounds(frame, positions, graph)
            points = link_paths[link]
            if _path_horizontal_segments_overlap_bounds(points, bounds):
                offenders.append(frame.frame_id)
            horizontals = [
                y1
                for (x1, y1), (x2, y2) in zip(points, points[1:])
                if abs(y1 - y2) < 1e-6 and abs(x1 - x2) > 0.06
            ]
            if horizontals:
                first_y = horizontals[0]
                assert first_y <= bounds.bottom - CONNECTOR_EXIT_STUB + 1e-6, (
                    f"{frame.frame_id} exit horizontal too high: {first_y:.3f} vs "
                    f"frame bottom {bounds.bottom:.3f}"
                )
        assert not offenders, f"frame exits overlap dotted boxes: {offenders}"
    finally:
        plt.close(fig)


def test_pipeline_merge_bus_sits_below_frame_exit_corridors():
    """Shared merge buses must clear every feeder frame's expanded bounds."""
    import matplotlib.pyplot as plt

    from visualizer.computation_graph import (
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
        _dock_single_consumer_tensor_ports,
        _estimate_graph_height,
        _graph_has_tensor_ports,
    )
    from visualizer.render import (
        COLORS,
        CONNECTOR_EXIT_STUB,
        DIAGRAM_LEFT_MARGIN,
        PIPELINE_MERGE_BUS_BELOW_FRAME_GAP,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _collect_detail_link_paths,
        _compute_detail_connector_buses,
        _frame_tail_exit_horiz_y,
        _inline_frame_draw_bounds,
    )
    from visualizer.render_validate import finalize_detail_layout

    pipeline, _basic = _load_chunk_kda_pipeline()
    graph = build_computation_graph(pipeline)
    fig, ax = plt.subplots(figsize=(16, 13))
    try:
        measure_graph_node_sizes(ax, graph)
        cx = DIAGRAM_LEFT_MARGIN + 0.05 + 9.0
        positions, links = layout_computation_graph(
            graph,
            cx=cx,
            top_y=12.0,
            block_w=18.0,
            block_h=_estimate_graph_height(graph),
        )
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=cx,
            top_y=12.0,
            detail_fill=COLORS["detail_fill"],
            min_left=DIAGRAM_LEFT_MARGIN + 0.05,
        )
        if _graph_has_tensor_ports(graph):
            _dock_single_consumer_tensor_ports(positions, graph)

        plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
        anchors = _anchors_from_detail_plan(positions, plan)
        incoming = {tgt: [] for tgt in range(len(graph.nodes))}
        outgoing = {src: [] for src in range(len(graph.nodes))}
        for src, tgt in links:
            incoming[tgt].append((src, tgt))
            outgoing[src].append((src, tgt))
        target_bus, source_bus, merge_entry_x, merge_link_bus = _compute_detail_connector_buses(
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
            target_bus=target_bus,
            source_bus=source_bus,
            merge_entry_x=merge_entry_x,
            merge_link_bus=merge_link_bus,
            input_index=None,
        )

        intra_idx = next(
            index for index, node in enumerate(graph.nodes) if node.label == "chunk_kda_fwd_intra"
        )
        bus_y = target_bus[intra_idx]
        frame_tails = {
            frame.frame_id: frame.node_indices[-1]
            for frame in graph.inline_frames
            if frame.node_indices
        }
        feeder_frames = (
            "forward_l2norm_fwd_k",
            "forward_fused_beta_sigmoid_beta",
            "chunk_kda_fwd_kda_gate_chunk_cumsum_g",
        )
        for frame_id in feeder_frames:
            tail_idx = frame_tails[frame_id]
            link = (tail_idx, intra_idx)
            assert link in link_paths, frame_id
            points = link_paths[link]
            frame = next(item for item in graph.inline_frames if item.frame_id == frame_id)
            bounds = _inline_frame_draw_bounds(frame, positions, graph)
            exit_horiz_y = _frame_tail_exit_horiz_y(graph, positions, tail_idx)
            assert exit_horiz_y is not None, frame_id
            assert bus_y <= exit_horiz_y - CONNECTOR_EXIT_STUB - PIPELINE_MERGE_BUS_BELOW_FRAME_GAP + 1e-6, (
                f"{frame_id} merge bus {bus_y:.3f} overlaps exit corridor "
                f"(exit horiz {exit_horiz_y:.3f}, frame bottom {bounds.bottom:.3f})"
            )
            for (x1, y1), (x2, y2) in zip(points, points[1:]):
                if abs(x1 - x2) < 1e-6 and y2 > y1 + 1e-6:
                    raise AssertionError(
                        f"{frame_id} frame-exit path has upward segment: "
                        f"({x1:.3f},{y1:.3f}) -> ({x2:.3f},{y2:.3f})"
                    )
    finally:
        plt.close(fig)


def test_bypass_connectors_branch_from_spine_at_intermediate_op():
    """Bypass paths tee on the frame gutter at the first skipped tile's height."""
    import matplotlib.pyplot as plt

    from visualizer.computation_graph import _ordered_inline_frame_chain
    from visualizer.render import (
        CONNECTOR_EXIT_STUB,
        _connector_source_bottom_exit_y,
        _connector_target_side_entry_y,
    )

    fig, graph, anchors, _plan, _incoming, _outgoing, _target_bus, _source_bus, _merge_link_bus, link_paths = (
        _chunk_kda_pipeline_link_paths()
    )
    try:
        for src, tgt in graph.inline_binary_operand_links:
            points = link_paths[(src, tgt)]
            source = anchors[src]
            frame = next(
                frame
                for frame in graph.inline_frames
                if src in frame.node_indices and tgt in frame.node_indices
            )
            chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
            src_index = chain.index(src)
            tgt_index = chain.index(tgt)
            if src_index >= tgt_index - 1:
                continue
            intermediate = chain[src_index + 1]
            tee_y = _connector_target_side_entry_y(anchors[intermediate])
            y_exit = _connector_source_bottom_exit_y(source)
            y_stub = y_exit - CONNECTOR_EXIT_STUB
            assert abs(points[0][0] - source.cx) < 1e-6
            assert abs(points[0][1] - y_stub) < 1e-6, (
                f"{graph.nodes[src].label}->{graph.nodes[tgt].label} "
                f"should tee from the spine stub (y={y_stub:.3f}), "
                f"got y={points[0][1]:.3f}"
            )
            assert abs(points[0][1] - y_exit) > 1e-6, "bypass must not leave from source exit"
            assert abs(points[1][1] - y_stub) < 1e-6
            assert abs(points[2][0] - points[1][0]) < 1e-6
            assert abs(points[2][1] - tee_y) < 1e-6, (
                f"{graph.nodes[src].label}->{graph.nodes[tgt].label} "
                f"should reach gutter tee at {graph.nodes[intermediate].label} "
                f"(y={tee_y:.3f}), got y={points[2][1]:.3f}"
            )
    finally:
        plt.close(fig)


def test_bypass_connectors_never_pass_through_skipped_blocks():
    """Bypass connectors must route around intermediate ops, not through them."""
    import matplotlib.pyplot as plt

    from visualizer.computation_graph import _ordered_inline_frame_chain
    from visualizer.render import (
        CONNECTOR_OBSTACLE_MARGIN,
        _path_penetrates_obstacle_tiles,
    )

    fig, graph, anchors, _plan, _incoming, _outgoing, _target_bus, _source_bus, _merge_link_bus, link_paths = (
        _chunk_kda_pipeline_link_paths()
    )
    try:
        offenders = []
        for src, tgt in graph.inline_binary_operand_links:
            frame = next(
                frame
                for frame in graph.inline_frames
                if src in frame.node_indices and tgt in frame.node_indices
            )
            chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
            src_index = chain.index(src)
            tgt_index = chain.index(tgt)
            skipped = chain[src_index + 1 : tgt_index]
            obstacles = [anchors[index] for index in skipped]
            points = link_paths[(src, tgt)]
            if _path_penetrates_obstacle_tiles(
                points,
                obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            ):
                offenders.append((graph.nodes[src].label, graph.nodes[tgt].label))
        assert not offenders, f"bypass connectors crossed skipped blocks: {offenders}"
    finally:
        plt.close(fig)


def test_connectors_never_pass_through_unrelated_blocks():
    """Every laid-out connector must clear all intermediate tiles at routing time."""
    import matplotlib.pyplot as plt

    from visualizer.render import (
        _connector_block_obstacles,
        _connector_path_clear_of_blocks,
    )

    fig, graph, anchors, plan, _incoming, _outgoing, _target_bus, _source_bus, _merge_link_bus, link_paths = (
        _chunk_kda_pipeline_link_paths()
    )
    try:
        offenders = []
        for (src, tgt), points in link_paths.items():
            source = anchors[src]
            target = anchors[tgt]
            obstacles = _connector_block_obstacles(
                anchors,
                src=src,
                tgt=tgt,
                label_obstacles=plan.label_obstacles,
                graph=graph,
                link_key=(src, tgt),
            )
            if not _connector_path_clear_of_blocks(
                points,
                source=source,
                target=target,
                obstacles=obstacles,
            ):
                offenders.append((graph.nodes[src].label, graph.nodes[tgt].label))
        assert not offenders, f"connectors crossed blocks: {offenders}"
    finally:
        plt.close(fig)


def test_l2norm_fwd_q_output_avoids_v_tensor_port():
    """The q-column l2norm tail feed to intra must route around the v input tile."""
    import matplotlib.pyplot as plt

    from visualizer.render import CONNECTOR_OBSTACLE_MARGIN, _path_hits_obstacles

    fig, graph, anchors, _plan, _incoming, _outgoing, _target_bus, _source_bus, _merge_link_bus, link_paths = (
        _chunk_kda_pipeline_link_paths()
    )
    try:
        q_frame = next(frame for frame in graph.inline_frames if frame.frame_id == "forward_l2norm_fwd_q")
        tail_idx = q_frame.node_indices[-1]
        intra_idx = next(
            index for index, node in enumerate(graph.nodes) if node.label == "chunk_kda_fwd_intra"
        )
        v_idx = next(index for index, node in enumerate(graph.nodes) if node.label == "v")
        points = link_paths[(tail_idx, intra_idx)]
        assert not _path_hits_obstacles(
            points,
            [anchors[v_idx]],
            margin=CONNECTOR_OBSTACLE_MARGIN,
        ), "q l2norm tail connector must clear the v input tile"
    finally:
        plt.close(fig)


def test_intra_chunk_merge_bus_uses_single_straight_channel():
    """All intra-chunk WY feeders tee onto one horizontal bus and enter at center."""
    import matplotlib.pyplot as plt

    from visualizer.render import (
        CONNECTOR_OBSTACLE_MARGIN,
        _connector_min_bus_y_above_target,
        _path_hits_obstacles,
        _segment_orientation,
    )

    fig, graph, anchors, _plan, _incoming, _outgoing, target_bus, _source_bus, _merge_link_bus, link_paths = (
        _chunk_kda_pipeline_link_paths()
    )
    try:
        intra_idx = next(
            index for index, node in enumerate(graph.nodes) if node.label == "chunk_kda_fwd_intra"
        )
        intra = anchors[intra_idx]
        bus_y = target_bus[intra_idx]
        assert bus_y >= _connector_min_bus_y_above_target(intra) - 1e-6

        intra_links = [(src, tgt) for src, tgt in link_paths if tgt == intra_idx]
        assert len(intra_links) >= 4

        for _src, tgt in intra_links:
            points = link_paths[(_src, tgt)]
            horizontals = [
                (index, y1)
                for index, ((x1, y1), (x2, y2)) in enumerate(zip(points, points[1:]))
                if abs(y1 - y2) < 1e-6 and abs(x1 - x2) > 0.06
            ]
            bus_horizontals = [(index, y) for index, y in horizontals if abs(y - bus_y) < 0.02]
            same_column_bus_tee = (
                len(points) == 3
                and abs(points[0][0] - intra.cx) < 1e-6
                and abs(points[1][0] - intra.cx) < 1e-6
                and abs(points[1][1] - bus_y) < 0.02
            )
            assert bus_horizontals or same_column_bus_tee, (
                f"{graph.nodes[_src].label} must tee onto the shared merge bus"
            )
            assert len(bus_horizontals) <= 1, (
                f"{graph.nodes[_src].label} must not backtrack on the merge bus"
            )
            if bus_horizontals:
                _bus_index, _bus_segment_y = bus_horizontals[0]
                x1, _ = points[_bus_index]
                x2, _ = points[_bus_index + 1]
                assert min(x1, x2) <= intra.cx <= max(x1, x2), (
                    f"{graph.nodes[_src].label} bus segment must span the intra center"
                )
            assert abs(points[-1][0] - intra.cx) < 1e-6
            assert abs(points[-1][1] - intra.top) < 1e-6
            assert not _path_hits_obstacles(
                points[:-1],
                [intra],
                margin=CONNECTOR_OBSTACLE_MARGIN,
            ), f"{graph.nodes[_src].label} bus must clear intra before entry"
            for index in range(len(points) - 1):
                assert _segment_orientation(
                    points[index][0],
                    points[index][1],
                    points[index + 1][0],
                    points[index + 1][1],
                ) is not None
    finally:
        plt.close(fig)


def test_l2norm_fwd_junction_dots_only_on_shared_buses():
    """Junction dots mark shared merge buses, not bypass bend corners."""
    import matplotlib.pyplot as plt

    from visualizer.render import _collect_connector_join_points

    fig, graph, anchors, _plan, incoming, _outgoing, target_bus, source_bus, merge_link_bus, link_paths = (
        _chunk_kda_pipeline_link_paths()
    )
    try:
        endpoint_keys = set()
        for points in link_paths.values():
            endpoint_keys.add((round(points[0][0], 3), round(points[0][1], 3)))
            endpoint_keys.add((round(points[-1][0], 3), round(points[-1][1], 3)))

        join_points = _collect_connector_join_points(
            link_paths,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            incoming=incoming,
            anchors=anchors,
        )
        assert join_points
        assert all(
            (round(x, 3), round(y, 3)) not in endpoint_keys for x, y in join_points
        ), "junction dots must not mark box attachment points"

        q_frame = next(
            frame for frame in graph.inline_frames if frame.frame_id == "forward_l2norm_fwd_q"
        )
        bypass_bus_points: set[tuple[float, float]] = set()
        for src, tgt in graph.inline_binary_operand_links:
            if src not in q_frame.node_indices or tgt not in q_frame.node_indices:
                continue
            for x, y in link_paths[(src, tgt)][1:-1]:
                bypass_bus_points.add((round(x, 3), round(y, 3)))

        assert not any(
            (round(x, 3), round(y, 3)) in bypass_bus_points for x, y in join_points
        ), "bypass corners must not get junction dots"

        intra_idx = next(
            index
            for index, node in enumerate(graph.nodes)
            if node.label == "chunk_kda_fwd_intra"
        )
        bus_y = target_bus[intra_idx]
        for (src, tgt), points in link_paths.items():
            if tgt != intra_idx:
                continue
            path = points
            for index in range(1, len(path) - 1):
                if not _is_bus_bend(path, index):
                    continue
                x, y = path[index]
                if abs(y - bus_y) > 0.03:
                    continue
                others_on_bus = [
                    lk
                    for lk, other_pts in link_paths.items()
                    if lk == (src, tgt)
                    or not any(abs(py - y) < 0.01 for _, py in other_pts)
                ]
                if len(others_on_bus) == 1:
                    assert not any(
                        abs(jx - x) < 0.03 and abs(jy - y) < 0.03 for jx, jy in join_points
                    ), f"single-link L-bend must not get a dot at ({x:.3f}, {y:.3f})"

        bus_junctions = [
            (x, y)
            for x, y in join_points
            if abs(y - bus_y) < 0.03
        ]
        assert bus_junctions, "expected a junction dot on the intra-chunk merge bus"
    finally:
        plt.close(fig)


def _is_bus_bend(points, index):
    from visualizer.render import _is_connector_path_bend

    return _is_connector_path_bend(points, index)


def test_chunk_kda_pipeline_draws_connector_junction_dots():
    """Rendered SVG should include junction dots where connector paths meet."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    from visualizer.computation_graph import (
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
        _dock_single_consumer_tensor_ports,
        _estimate_graph_height,
        _graph_has_tensor_ports,
    )
    from visualizer.render import (
        COLORS,
        CONNECTOR_JUNCTION_DOT_RADIUS,
        CONNECTOR_JUNCTION_HALO_RADIUS,
        DIAGRAM_LEFT_MARGIN,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _collect_connector_join_points,
        _collect_detail_link_paths,
        _compute_detail_connector_buses,
        _draw_connector_junction_dots,
        _junction_dot_fits,
    )
    from visualizer.render_validate import finalize_detail_layout

    pipeline, _basic = _load_chunk_kda_pipeline()
    graph = build_computation_graph(pipeline)
    fig, ax = plt.subplots(figsize=(16, 13))
    try:
        measure_graph_node_sizes(ax, graph)
        cx = DIAGRAM_LEFT_MARGIN + 0.05 + 9.0
        positions, links = layout_computation_graph(
            graph,
            cx=cx,
            top_y=12.0,
            block_w=18.0,
            block_h=_estimate_graph_height(graph),
        )
        finalize_detail_layout(
            ax,
            graph,
            positions,
            input_sublabel=None,
            cx=cx,
            top_y=12.0,
            detail_fill=COLORS["detail_fill"],
            min_left=DIAGRAM_LEFT_MARGIN + 0.05,
        )
        if _graph_has_tensor_ports(graph):
            _dock_single_consumer_tensor_ports(positions, graph)

        plan = _build_detail_draw_plan(positions, graph, input_sublabel=None)
        anchors = _anchors_from_detail_plan(positions, plan)
        incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
        outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for src, tgt in links:
            incoming[tgt].append((src, tgt))
            outgoing[src].append((src, tgt))

        target_bus, source_bus, merge_entry_x, merge_link_bus = _compute_detail_connector_buses(
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
            target_bus=target_bus,
            source_bus=source_bus,
            merge_entry_x=merge_entry_x,
            merge_link_bus=merge_link_bus,
            input_index=None,
        )
        join_points = _collect_connector_join_points(
            link_paths,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            incoming=incoming,
            anchors=anchors,
        )
        assert join_points, "expected connector join points in chunk_kda pipeline"

        combine_ops = [(op_x, op_y) for op_x, op_y, _symbol, _sublabel in plan.combine_ops]
        drawable = [
            point
            for point in join_points
            if _junction_dot_fits(
                point[0],
                point[1],
                list(anchors.values()) + plan.label_obstacles,
                halo_radius=CONNECTOR_JUNCTION_HALO_RADIUS,
                combine_ops=combine_ops,
            )
        ]
        assert drawable, "expected at least one junction dot clear of boxes"

        _draw_connector_junction_dots(
            ax,
            link_paths,
            obstacles=list(anchors.values()) + plan.label_obstacles,
            combine_ops=combine_ops,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_link_bus=merge_link_bus,
            incoming=incoming,
            anchors=anchors,
        )
        from matplotlib.patches import Circle

        junction_patches = [
            patch
            for patch in ax.patches
            if isinstance(patch, Circle)
            and abs(patch.get_radius() - CONNECTOR_JUNCTION_DOT_RADIUS) < 1e-6
        ]
        assert junction_patches, "expected junction dot circles on the axes"
        assert all(
            patch.get_edgecolor() == (0.0, 0.0, 0.0, 0.0) or patch.get_linewidth() == 0.0
            for patch in junction_patches
        ), "junction dots must not have a visible border"
    finally:
        plt.close(fig)
