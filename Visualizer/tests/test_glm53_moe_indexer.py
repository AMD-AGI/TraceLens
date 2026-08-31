"""GLM-5.3-Flash MoE router and sparse-attention indexer diagram tests."""

from __future__ import annotations

import pytest

from visualizer.block_tree import (
    build_block_node,
    collect_function_steps,
    expand_block_tree_inplace,
    is_inline_expandable_module,
)
from visualizer.computation_graph import build_computation_graph
from visualizer.loader import load_model_spec
from visualizer.render import _detail_sections_to_render, _ffn_label, _spine_display_label


@pytest.mark.parametrize("class_name", ["Glm5NextTextTopkRouter"])
def test_glm53_topk_router_parses_topk_chain(class_name: str):
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    router = spec.class_registry[class_name]
    labels = [
        router.forward_operations[key].label
        for key in router.forward_calls
        if key in router.forward_operations
    ]
    assert "Linear" in labels
    assert "TopK" in labels
    assert labels.count("TopK") >= 2


def test_glm53_moe_gate_keeps_topk_in_computation_graph():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    _title, tree = next(item for item in spec.detailed_block_trees if item[0] == "Glm5NextTextMoE")
    tree = expand_block_tree_inplace(tree, basic_ops=spec.basic_ops)
    graph = build_computation_graph(tree, basic_ops=spec.basic_ops)
    labels = [node.label for node in graph.nodes]
    assert labels.count("TopK") >= 2
    assert "Sigmoid" in labels


def test_glm53_ffn_spine_and_overview_use_moe_and_mlp_classes():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    assert _ffn_label(spec)[0] == "Glm5NextTextMoE / Glm5NextTextMLP"
    mlp = next(comp for comp in spec.block_components if comp.attr_name == "mlp")
    assert _spine_display_label(mlp, spec) == "Glm5NextTextMoE / Glm5NextTextMLP"


def test_glm53_large_indexer_is_separate_detail_section():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    attn_title, attn_tree = next(
        item for item in spec.detailed_block_trees if item[0] == "Glm5NextText Attn"
    )
    indexer = next(child for child in attn_tree.children if child.attr_name == "indexer")
    assert len(collect_function_steps(indexer)) >= 12
    assert not is_inline_expandable_module(indexer)

    attn_graph = build_computation_graph(attn_tree, basic_ops=spec.basic_ops)
    assert any(node.label == "Glm5NextTextIndexer" for node in attn_graph.nodes)

    section_titles = [title for title, _tree, _sub in _detail_sections_to_render(spec)]
    assert "Glm5NextTextIndexer" in section_titles


def test_glm53_indexer_connectors_have_no_overlaps():
    pytest.importorskip("huggingface_hub")
    from collections import defaultdict

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.computation_graph import (
        SYNTHETIC_INPUT,
        _estimate_graph_height,
        add_forward_output,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import (
        COLORS,
        DIAGRAM_LEFT_MARGIN,
        _anchors_from_detail_plan,
        _build_detail_draw_plan,
        _collect_detail_link_paths,
        _compute_detail_connector_buses,
        _detail_sections_to_render,
        _find_connector_path_overlaps,
    )
    from visualizer.render_validate import finalize_detail_layout, measure_detail_tree_content_width

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    tree = next(
        block
        for title, block, _ in _detail_sections_to_render(spec)
        if title == "Glm5NextTextIndexer"
    )
    graph = build_computation_graph(tree, include_input=True)
    add_forward_output(graph)
    fig, ax = plt.subplots(figsize=(16, 16))
    input_sublabel = "← RMSNorm"
    measure_graph_node_sizes(ax, graph, input_sublabel=input_sublabel)
    cx = DIAGRAM_LEFT_MARGIN + 2.0
    top_y = 19.26
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
        block_w=section_w - 0.2,
        block_h=_estimate_graph_height(graph),
    )
    plan = finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=input_sublabel,
        cx=cx,
        top_y=top_y,
        detail_fill=COLORS["detail_fill"],
        min_left=min_left,
    )
    anchors = _anchors_from_detail_plan(positions, plan, graph)
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
        validate_layout=False,
    )
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
    plt.close(fig)
    assert not overlaps, f"indexer connector overlaps: {overlaps[:4]}"


def test_glm53_moe_experts_tiles_stay_inside_inline_frame():
    pytest.importorskip("huggingface_hub")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.block_tree import expand_block_tree_inplace
    from visualizer.computation_graph import (
        _estimate_graph_height,
        add_forward_output,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import (
        COLORS,
        DIAGRAM_LEFT_MARGIN,
        _detail_sections_to_render,
        _inline_frame_draw_bounds,
    )
    from visualizer.render_validate import finalize_detail_layout, measure_detail_tree_content_width

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    tree = next(
        block
        for title, block, _ in _detail_sections_to_render(spec)
        if title == "Glm5NextTextMoE"
    )
    tree = expand_block_tree_inplace(tree, basic_ops=spec.basic_ops)
    graph = build_computation_graph(tree, include_input=True, basic_ops=spec.basic_ops)
    add_forward_output(graph)
    fig, ax = plt.subplots(figsize=(16, 16))
    input_sublabel = "← RMSNorm"
    measure_graph_node_sizes(ax, graph, input_sublabel=input_sublabel)
    cx = DIAGRAM_LEFT_MARGIN + 2.0
    top_y = 10.0
    min_left = DIAGRAM_LEFT_MARGIN + 0.05
    section_w = measure_detail_tree_content_width(
        ax,
        tree,
        cx=cx,
        detail_fill=COLORS["detail_fill"],
        min_left=min_left,
        input_sublabel=input_sublabel,
    )
    positions, _links = layout_computation_graph(
        graph,
        cx=cx,
        top_y=top_y,
        block_w=section_w - 0.2,
        block_h=_estimate_graph_height(graph),
    )
    plan = finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=input_sublabel,
        cx=cx,
        top_y=top_y,
        detail_fill=COLORS["detail_fill"],
        min_left=min_left,
    )
    experts = next(frame for frame in graph.inline_frames if frame.frame_id == "experts")
    bounds = _inline_frame_draw_bounds(experts, positions, graph)
    draw_index = 0
    for index, _pos in enumerate(positions):
        if draw_index >= len(plan.node_draws):
            break
        leaf, _ = plan.node_draws[draw_index]
        pos = positions[index]
        draw_index += 1
        if index not in experts.node_indices:
            continue
        assert abs(leaf.cx - pos.cx) < 1e-6, (
            f"experts tile plan cx drifted from layout for {leaf.label}"
        )
        left = pos.cx - pos.width / 2
        right = pos.cx + pos.width / 2
        assert bounds.left <= left + 1e-6 and right <= bounds.right + 1e-6, (
            f"{leaf.label} rendered outside Glm5NextTextExperts frame"
        )
    plt.close(fig)


def test_glm53_indexer_subtract_to_pad_has_no_vertical_backtrack():
    pytest.importorskip("huggingface_hub")
    from collections import defaultdict

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualizer.computation_graph import (
        _estimate_graph_height,
        add_forward_output,
        build_computation_graph,
        layout_computation_graph,
        measure_graph_node_sizes,
    )
    from visualizer.render import (
        COLORS,
        DIAGRAM_LEFT_MARGIN,
        _anchors_from_detail_plan,
        _collect_detail_link_paths,
        _compute_detail_connector_buses,
        _detail_sections_to_render,
    )
    from visualizer.render_validate import finalize_detail_layout, measure_detail_tree_content_width

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    tree = next(
        block
        for title, block, _ in _detail_sections_to_render(spec)
        if title == "Glm5NextTextIndexer"
    )
    graph = build_computation_graph(tree, include_input=True)
    add_forward_output(graph)
    fig, ax = plt.subplots(figsize=(16, 16))
    input_sublabel = "← RMSNorm"
    measure_graph_node_sizes(ax, graph, input_sublabel=input_sublabel)
    cx = DIAGRAM_LEFT_MARGIN + 2.0
    top_y = 19.26
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
        block_w=section_w - 0.2,
        block_h=_estimate_graph_height(graph),
    )
    plan = finalize_detail_layout(
        ax,
        graph,
        positions,
        input_sublabel=input_sublabel,
        cx=cx,
        top_y=top_y,
        detail_fill=COLORS["detail_fill"],
        min_left=min_left,
    )
    anchors = _anchors_from_detail_plan(positions, plan, graph)
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
    link_paths = dict(
        _collect_detail_link_paths(
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
            input_index=0,
            validate_layout=False,
        )
    )
    subtract_pad = next(
        path
        for (src, tgt), path in link_paths.items()
        if graph.nodes[src].label == "Subtract" and graph.nodes[tgt].label == "Pad"
    )
    for (x1, y1), (x2, y2) in zip(subtract_pad, subtract_pad[1:]):
        if abs(x1 - x2) <= 1e-6 and abs(y1 - y2) > 1e-6:
            continue
        if abs(y1 - y2) <= 1e-6:
            continue
    for index in range(len(subtract_pad) - 2):
        x0, y0 = subtract_pad[index]
        x1, y1 = subtract_pad[index + 1]
        x2, y2 = subtract_pad[index + 2]
        if abs(x0 - x1) <= 1e-6 and abs(x1 - x2) <= 1e-6:
            assert (y1 - y0) * (y2 - y1) >= 0, f"Subtract→Pad backtracks: {subtract_pad}"
    plt.close(fig)
