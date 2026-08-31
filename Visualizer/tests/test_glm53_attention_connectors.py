"""Connector routing regressions for GLM-5.3-Flash attention detail sections."""

from __future__ import annotations

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from visualizer.computation_graph import (
    SYNTHETIC_INPUT,
    _estimate_graph_height,
    build_computation_graph,
    layout_computation_graph,
    measure_graph_node_sizes,
)
from visualizer.loader import load_model_spec
from visualizer.render import (
    COLORS,
    DIAGRAM_LEFT_MARGIN,
    PARALLEL_CONNECTOR_COORD_EPS,
    _anchors_from_detail_plan,
    _build_detail_draw_plan,
    _collect_connector_join_points,
    _collect_detail_link_paths,
    _compute_detail_connector_buses,
    _detail_sections_to_render,
    _path_enters_target_top_center,
)
from visualizer.render_validate import finalize_detail_layout, measure_detail_tree_content_width


def _path_has_horizontal_backtrack(points: list[tuple[float, float]]) -> bool:
    for index in range(len(points) - 2):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        x3, y3 = points[index + 2]
        if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS or abs(y2 - y3) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if abs(x2 - x1) <= PARALLEL_CONNECTOR_COORD_EPS or abs(x3 - x2) <= PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if (x2 - x1) * (x3 - x2) < 0:
            return True
    return False


def _horizontal_bus_levels(points: list[tuple[float, float]]) -> list[float]:
    levels: list[float] = []
    for index in range(1, len(points)):
        x1, y1 = points[index - 1]
        x2, y2 = points[index]
        if abs(y1 - y2) <= PARALLEL_CONNECTOR_COORD_EPS and abs(x1 - x2) > PARALLEL_CONNECTOR_COORD_EPS:
            levels.append(y2)
    return levels


def _layout_section(title: str):
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    tree = next(block for section_title, block, _ in _detail_sections_to_render(spec) if section_title == title)
    graph = build_computation_graph(tree, include_input=True)
    fig, ax = plt.subplots(figsize=(16, 16))
    input_sublabel = "← RMSNorm"
    measure_graph_node_sizes(ax, graph, input_sublabel=input_sublabel)
    cx, top_y = 8.0, 15.0
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
    input_index = next(
        index for index, node in enumerate(graph.nodes) if node.synthetic == SYNTHETIC_INPUT
    )
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
        input_index=input_index,
    )
    return fig, graph, positions, anchors, incoming, outgoing, merge_link_bus, link_paths


@pytest.mark.parametrize(
    "section_title",
    ["Glm5NextTextAttention", "Glm5NextTextLinearAttention"],
)
def test_glm53_attention_connectors_avoid_backtracking(section_title: str):
    pytest.importorskip("huggingface_hub")
    _fig, graph, _positions, _anchors, _incoming, _outgoing, _merge_link_bus, link_paths = _layout_section(
        section_title
    )
    for (src, tgt), points in link_paths.items():
        assert not _path_has_horizontal_backtrack(points), (
            f"{graph.nodes[src].label} -> {graph.nodes[tgt].label} backtracks: {points}"
        )


def test_glm53_sparse_attention_expand_kv_feed_is_straight():
    pytest.importorskip("huggingface_hub")
    _fig, graph, positions, anchors, _incoming, _outgoing, _merge_link_bus, link_paths = _layout_section(
        "Glm5NextTextAttention"
    )
    expand_kv = next(frame for frame in graph.inline_frames if frame.label == "expand_kv")
    top_member = max(expand_kv.node_indices, key=lambda index: positions[index].top_y)
    for src, tgt in link_paths:
        if graph.nodes[src].label != "RMSNorm" or tgt != top_member:
            continue
        points = link_paths[(src, tgt)]
        assert _path_enters_target_top_center(points, anchors[tgt])
        assert len(points) == 2, f"expected straight vertical feed, got {points}"


def test_glm53_sparse_attention_inputs_use_nested_merge_buses():
    pytest.importorskip("huggingface_hub")
    _fig, graph, _positions, _anchors, _incoming, _outgoing, merge_link_bus, link_paths = _layout_section(
        "Glm5NextTextAttention"
    )
    attn_index = next(index for index, node in enumerate(graph.nodes) if node.label == "Attention")
    assigned = {
        graph.nodes[src].label: merge_link_bus[(src, attn_index)]
        for src, tgt in link_paths
        if tgt == attn_index and (src, attn_index) in merge_link_bus
    }
    assert len({round(level, 4) for level in assigned.values()}) >= 2, assigned
    horizontals = {
        graph.nodes[src].label: _horizontal_bus_levels(link_paths[(src, attn_index)])
        for src, tgt in link_paths
        if tgt == attn_index
    }
    assert len({round(levels[0], 4) for levels in horizontals.values() if levels}) >= 2, horizontals
    joins = _collect_connector_join_points(
        link_paths,
        merge_link_bus=merge_link_bus,
        graph=graph,
    )
    shared_bus_joins = [point for point in joins if abs(point[1] - 11.506) < 0.02]
    assert len(shared_bus_joins) <= 1, shared_bus_joins
