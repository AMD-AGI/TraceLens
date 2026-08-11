"""Build computation graphs from block trees and lay them out with graph-layout."""

from __future__ import annotations

from dataclasses import dataclass, field

from graph_layout import SugiyamaLayout

from visualizer.block_tree import (
    BlockNode,
    CombineSegment,
    ComputationSegment,
    FanOutSegment,
    PortStyle,
    ResidualAddSegment,
    SeqSegment,
    SideCombineSegment,
    SideFeedSegment,
    collect_computation_segments,
    flatten_computation_segments,
    inline_block_frame_label,
    inline_composite_steps,
    inline_wrapper_step_label,
    is_straight_line_module,
    is_method_wrapper,
    wrapper_bullet_lines,
)
from visualizer.sizing import (
    block_sublabel,
    estimate_block_size_for_node,
    min_vertical_block_gap,
    PIXELS_PER_UNIT,
    to_layout_pixels,
)

# Match render.py MERGE_RADIUS + MERGE_CLEARANCE for combine (×) nodes.
COMBINE_OP_SIZE = 0.32

SYNTHETIC_INPUT = "@input"
SYNTHETIC_HIDDEN = "@hidden_states"  # legacy alias; replaced by SYNTHETIC_INPUT in graphs
SYNTHETIC_COMBINE = "@combine"
SYNTHETIC_MULTIPLY = SYNTHETIC_COMBINE  # backwards-compatible alias


@dataclass
class GraphNodeSpec:
    """One vertex in a computation graph."""

    key: str
    block: BlockNode | None = None
    label: str = ""
    sublabel: str | None = None
    port_label: str | None = None
    port_style: PortStyle | None = None
    synthetic: str | None = None
    width: float = 90.0
    height: float = 42.0


@dataclass
class InlineFrameSpec:
    """Dotted frame around steps expanded inline from a linear composite sub-block."""

    frame_id: str
    label: str
    node_indices: list[int] = field(default_factory=list)


@dataclass
class ComputationGraph:
    """Directed graph ready for Sugiyama layout."""

    nodes: list[GraphNodeSpec] = field(default_factory=list)
    links: list[tuple[int, int]] = field(default_factory=list)
    dashed_links: set[tuple[int, int]] = field(default_factory=set)
    side_entry_links: set[tuple[int, int]] = field(default_factory=set)
    inline_frames: list[InlineFrameSpec] = field(default_factory=list)


@dataclass
class LayoutPosition:
    """Positioned node in diagram coordinates (matplotlib, y-up)."""

    spec: GraphNodeSpec
    cx: float
    top_y: float
    width: float
    height: float

    @property
    def bottom(self) -> float:
        return self.top_y - self.height


def _diagram_size_for_block(block: BlockNode | None, label: str | None = None) -> tuple[float, float]:
    """Return (width, height) in diagram units for graph layout."""
    return estimate_block_size_for_node(block, label)


def _is_combine_synthetic(synthetic: str | None) -> bool:
    return synthetic == SYNTHETIC_COMBINE


def _estimate_node_size(spec: GraphNodeSpec) -> tuple[float, float]:
    if _is_combine_synthetic(spec.synthetic):
        return to_layout_pixels(COMBINE_OP_SIZE, COMBINE_OP_SIZE)
    if spec.synthetic in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}:
        return 48.0, 20.0
    diagram_w, diagram_h = _diagram_size_for_rendered_spec(spec)
    return to_layout_pixels(diagram_w, diagram_h)


def _add_node(
    graph: ComputationGraph,
    *,
    key: str,
    block: BlockNode | None = None,
    label: str | None = None,
    sublabel: str | None = None,
    port_label: str | None = None,
    port_style: PortStyle | None = None,
    synthetic: str | None = None,
) -> int:
    display = label or (block.label if block else key)
    node_sublabel = sublabel if sublabel is not None else block_sublabel(block)
    spec = GraphNodeSpec(
        key=key,
        block=block,
        label=display,
        sublabel=node_sublabel,
        port_label=port_label,
        port_style=port_style,
        synthetic=synthetic,
    )
    spec.width, spec.height = _estimate_node_size(spec)
    graph.nodes.append(spec)
    return len(graph.nodes) - 1


def _add_method_wrapper_node(
    graph: ComputationGraph,
    step: BlockNode,
    *,
    key: str,
) -> int:
    label, _attr = wrapper_bullet_lines(step)
    return _add_node(graph, key=key, block=step, label=label, sublabel=_attr)


def _track_attr_index(attr_last_index: dict[str, int], attr_name: str, index: int) -> None:
    attr_last_index[attr_name] = index


def _add_chain(
    graph: ComputationGraph,
    steps: list[BlockNode],
    *,
    port_label: str | None = None,
    port_style: PortStyle | None = None,
    key_prefix: str,
    attr_last_index: dict[str, int] | None = None,
) -> tuple[int | None, int | None]:
    first_index: int | None = None
    previous: int | None = None
    for index, step in enumerate(steps):
        if is_method_wrapper(step):
            node_index = _add_method_wrapper_node(
                graph,
                step,
                key=f"{key_prefix}:{step.attr_name}:{index}",
            )
        else:
            node_index = _add_node(
                graph,
                key=f"{key_prefix}:{step.attr_name}:{index}",
                block=step,
                port_label=port_label if index == 0 else None,
                port_style=port_style if index == 0 else None,
            )
        if attr_last_index is not None:
            _track_attr_index(attr_last_index, step.attr_name, node_index)
        if first_index is None:
            first_index = node_index
        if previous is not None:
            graph.links.append((previous, node_index))
        previous = node_index
    return first_index, previous


def _input_label_for(root: BlockNode) -> str:
    return root.input_label or "hidden_states"


def _add_forward_input(graph: ComputationGraph, root: BlockNode) -> int:
    return _add_node(
        graph,
        key=SYNTHETIC_INPUT,
        label=_input_label_for(root),
        synthetic=SYNTHETIC_INPUT,
    )


def add_root_pipeline_frame(
    graph: ComputationGraph,
    block: BlockNode,
    *,
    label: str | None = None,
) -> None:
    """Group all non-input steps of a root block tree in one inline frame."""
    if not is_straight_line_module(block):
        return
    indices = [
        index
        for index, spec in enumerate(graph.nodes)
        if spec.synthetic not in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}
        and not _is_combine_synthetic(spec.synthetic)
    ]
    if len(indices) < 2:
        return
    graph.inline_frames.append(
        InlineFrameSpec(
            frame_id=block.attr_name,
            label=label or inline_block_frame_label(block),
            node_indices=indices,
        )
    )


def _link_forward_input(
    graph: ComputationGraph,
    input_index: int,
    target_index: int,
    *,
    dashed: bool,
) -> None:
    graph.links.append((input_index, target_index))
    if dashed:
        graph.dashed_links.add((input_index, target_index))


def _start_inline_frame(graph: ComputationGraph, wrapper: BlockNode) -> InlineFrameSpec:
    frame = InlineFrameSpec(frame_id=wrapper.attr_name, label=inline_block_frame_label(wrapper))
    graph.inline_frames.append(frame)
    return frame


def _append_inline_frame_node(frame: InlineFrameSpec, node_index: int) -> None:
    frame.node_indices.append(node_index)


def _add_linear_pipeline_chain(
    graph: ComputationGraph,
    steps: list[BlockNode],
    *,
    wrapper: BlockNode | None,
    key_prefix: str,
    attr_last_index: dict[str, int] | None = None,
    port_label: str | None = None,
    port_style: PortStyle | None = None,
    input_index: int | None = None,
    last_index: int | None = None,
    fork_from_input: bool = False,
    branch_from_input_dashed: bool = False,
) -> tuple[list[int], int | None]:
    """Add a straight-line chain of nodes, optionally grouped in an inline frame."""
    if not steps:
        return [], last_index

    frame = _start_inline_frame(graph, wrapper) if wrapper is not None else None
    indices: list[int] = []
    chain_last = last_index

    for sub_index, sub_step in enumerate(steps):
        step_index = _add_node(
            graph,
            key=f"{key_prefix}:{sub_step.attr_name}:{sub_index}",
            block=sub_step,
            label=inline_wrapper_step_label(wrapper, sub_step, sub_index),
            port_label=port_label if sub_index == 0 else None,
            port_style=port_style if sub_index == 0 else None,
        )
        if frame is not None:
            _append_inline_frame_node(frame, step_index)
        if attr_last_index is not None:
            _track_attr_index(attr_last_index, sub_step.attr_name, step_index)

        if sub_index == 0:
            if branch_from_input_dashed and input_index is not None:
                _link_forward_input(graph, input_index, step_index, dashed=True)
            else:
                use_fork = fork_from_input and input_index is not None
                _append_step_link(
                    graph,
                    input_index=input_index if input_index is not None else 0,
                    last_index=chain_last,
                    step_index=step_index,
                    fork_from_input=use_fork,
                )
        else:
            graph.links.append((indices[-1], step_index))

        indices.append(step_index)

    return indices, indices[-1]


def _consumer_port_label(sides: list) -> str | None:
    if not sides:
        return None
    labels = [side.port_label for side in sides]
    if len(set(labels)) == 1:
        return labels[0]
    return labels[0]


def _upcoming_side_combine(
    segments: list[ComputationSegment],
    segment_index: int,
) -> SideCombineSegment | None:
    if segment_index + 1 >= len(segments):
        return None
    nxt = segments[segment_index + 1]
    return nxt if isinstance(nxt, SideCombineSegment) else None


def _side_source_tail_index(
    segment: SideCombineSegment,
    attr_last_index: dict[str, int],
) -> int | None:
    for side in segment.sides:
        if side.source_kind != "prior_step" or not side.source_chain:
            continue
        index = attr_last_index.get(side.source_chain[-1])
        if index is not None:
            return index
    return None


def _should_fork_main_path_from_input(
    segments: list[ComputationSegment],
    segment_index: int,
    last_index: int | None,
    attr_last_index: dict[str, int],
) -> bool:
    """True when the next step should branch from input, not the router side-path tail."""
    if last_index is None:
        return False
    side_combine = _upcoming_side_combine(segments, segment_index)
    if side_combine is None:
        return False
    side_tail = _side_source_tail_index(side_combine, attr_last_index)
    return side_tail is not None and side_tail == last_index


def _append_step_link(
    graph: ComputationGraph,
    *,
    input_index: int | None,
    last_index: int | None,
    step_index: int,
    fork_from_input: bool,
) -> None:
    if fork_from_input:
        if input_index is not None:
            graph.links.append((input_index, step_index))
    elif last_index is not None:
        graph.links.append((last_index, step_index))
    elif input_index is not None:
        graph.links.append((input_index, step_index))


def build_computation_graph(
    root: BlockNode,
    *,
    prefix_steps: list[BlockNode] | None = None,
    include_input: bool = True,
) -> ComputationGraph:
    """Convert a block tree into a directed acyclic computation graph."""
    graph = ComputationGraph()

    if root.is_basic or not root.children:
        input_index = _add_forward_input(graph, root) if include_input else None
        node_index = _add_node(graph, key=root.attr_name, block=root)
        if input_index is not None:
            graph.links.append((input_index, node_index))
        return graph

    segments = flatten_computation_segments(root)
    input_index = _add_forward_input(graph, root) if include_input else None

    last_index: int | None = None
    attr_last_index: dict[str, int] = {}
    for prefix_index, step in enumerate(prefix_steps or []):
        if not is_method_wrapper(step):
            continue
        step_index = _add_method_wrapper_node(
            graph,
            step,
            key=f"prefix:{step.attr_name}:{prefix_index}",
        )
        if last_index is not None:
            graph.links.append((last_index, step_index))
        elif input_index is not None:
            graph.links.append((input_index, step_index))
        last_index = step_index
        _track_attr_index(attr_last_index, step.attr_name, step_index)

    for segment_index, segment in enumerate(segments):
        if isinstance(segment, FanOutSegment):
            tails: list[int] = []
            for branch_index, branch in enumerate(segment.branches):
                first_index, tail = _add_chain(
                    graph,
                    branch.steps,
                    port_label=branch.port_label,
                    port_style=branch.port_style,
                    key_prefix=f"fan{segment_index}-{branch_index}",
                    attr_last_index=attr_last_index,
                )
                if input_index is not None and first_index is not None:
                    _link_forward_input(graph, input_index, first_index, dashed=False)
                if tail is not None:
                    tails.append(tail)
            merge_index = _add_node(
                graph,
                key=f"merge:{segment_index}",
                block=segment.merge,
            )
            for tail in tails:
                graph.links.append((tail, merge_index))
            last_index = merge_index
            _track_attr_index(attr_last_index, segment.merge.attr_name, merge_index)
            continue

        if isinstance(segment, SideCombineSegment):
            combine_index = _add_node(
                graph,
                key=f"sidecombine:{segment_index}:{segment.consumer.attr_name}",
                label=segment.op,
                sublabel=segment.op_sublabel,
                synthetic=SYNTHETIC_COMBINE,
            )
            if last_index is not None:
                graph.links.append((last_index, combine_index))
            elif input_index is not None:
                graph.links.append((input_index, combine_index))
            for side in segment.sides:
                if side.source_kind == "forward_input":
                    if input_index is not None:
                        _link_forward_input(graph, input_index, combine_index, dashed=True)
                    continue
                source_attr = side.source_chain[-1] if side.source_chain else None
                if source_attr is None:
                    continue
                source_index = attr_last_index.get(source_attr)
                if source_index is None:
                    continue
                graph.links.append((source_index, combine_index))
                graph.side_entry_links.add((source_index, combine_index))
            last_index = combine_index
            _track_attr_index(attr_last_index, segment.consumer.attr_name, combine_index)
            continue

        if isinstance(segment, ResidualAddSegment):
            module = segment.module
            expanded_steps, wrapper = inline_composite_steps(module)
            if wrapper is not None:
                _branch_indices, module_tail = _add_linear_pipeline_chain(
                    graph,
                    expanded_steps,
                    wrapper=wrapper,
                    key_prefix=f"residual_branch:{segment_index}:{module.attr_name}",
                    attr_last_index=attr_last_index,
                    input_index=input_index,
                    last_index=None,
                    branch_from_input_dashed=True,
                )
                _track_attr_index(attr_last_index, wrapper.attr_name, module_tail)
                _track_attr_index(attr_last_index, module.attr_name, module_tail)
            else:
                module_index = _add_node(
                    graph,
                    key=f"residual_branch:{segment_index}:{module.attr_name}",
                    block=module,
                )
                if input_index is not None:
                    _link_forward_input(graph, input_index, module_index, dashed=True)
                _track_attr_index(attr_last_index, module.attr_name, module_index)
                module_tail = module_index
            combine_index = _add_node(
                graph,
                key=f"residual_add:{segment_index}",
                label="+",
                sublabel=None,
                synthetic=SYNTHETIC_COMBINE,
            )
            if last_index is not None:
                graph.links.append((last_index, combine_index))
            graph.links.append((module_tail, combine_index))
            graph.dashed_links.add((module_tail, combine_index))
            graph.side_entry_links.add((module_tail, combine_index))
            last_index = combine_index
            continue

        if isinstance(segment, SideFeedSegment):
            consumer = segment.consumer
            port_label = _consumer_port_label(segment.sides)
            if is_method_wrapper(consumer):
                consumer_index = _add_method_wrapper_node(
                    graph,
                    consumer,
                    key=f"sidefeed:{segment_index}:{consumer.attr_name}",
                )
                if port_label:
                    graph.nodes[consumer_index].port_label = port_label
                    graph.nodes[consumer_index].port_style = "inline"
                    graph.nodes[consumer_index].width, graph.nodes[consumer_index].height = _estimate_node_size(
                        graph.nodes[consumer_index]
                    )
            else:
                consumer_index = _add_node(
                    graph,
                    key=f"sidefeed:{segment_index}:{consumer.attr_name}",
                    block=consumer,
                    port_label=port_label,
                    port_style="inline" if port_label else None,
                )
            if last_index is not None:
                graph.links.append((last_index, consumer_index))
            elif input_index is not None:
                graph.links.append((input_index, consumer_index))
            for side in segment.sides:
                if side.source_kind == "forward_input":
                    if input_index is not None:
                        _link_forward_input(graph, input_index, consumer_index, dashed=True)
                    continue
                source_attr = side.source_chain[-1] if side.source_chain else None
                if source_attr is None:
                    continue
                source_index = attr_last_index.get(source_attr)
                if source_index is None:
                    continue
                graph.links.append((source_index, consumer_index))
                graph.dashed_links.add((source_index, consumer_index))
            last_index = consumer_index
            _track_attr_index(attr_last_index, consumer.attr_name, consumer_index)
            continue

        if isinstance(segment, CombineSegment):
            after_nodes = list(segment.after)
            if is_method_wrapper(segment.side):
                if last_index is None:
                    continue
                first_after, tail = _add_chain(
                    graph,
                    after_nodes,
                    key_prefix=f"post:{segment_index}",
                    attr_last_index=attr_last_index,
                )
                if first_after is not None:
                    graph.links.append((last_index, first_after))
                last_index = tail
                continue

            side = segment.side
            expanded_side, side_wrapper = inline_composite_steps(side)
            if side_wrapper is not None:
                _side_indices, side_tail = _add_linear_pipeline_chain(
                    graph,
                    expanded_side,
                    wrapper=side_wrapper,
                    key_prefix=f"side:{segment_index}",
                    attr_last_index=attr_last_index,
                    port_label=segment.side_port_label,
                    port_style=segment.side_port_style,
                    input_index=input_index,
                    last_index=None,
                    branch_from_input_dashed=segment.side_source == "forward_input",
                )
                side_index = side_tail
                _track_attr_index(attr_last_index, side_wrapper.attr_name, side_index)
                _track_attr_index(attr_last_index, side.attr_name, side_index)
            else:
                side_index = _add_node(
                    graph,
                    key=f"side:{segment_index}",
                    block=side,
                    port_label=segment.side_port_label,
                    port_style=segment.side_port_style,
                )
                _track_attr_index(attr_last_index, side.attr_name, side_index)
                if input_index is not None and segment.side_source == "forward_input":
                    _link_forward_input(graph, input_index, side_index, dashed=True)
            if last_index is None:
                continue

            mult_index = _add_node(
                graph,
                key=f"combine:{segment_index}",
                label=segment.op,
                sublabel=None,
                synthetic=SYNTHETIC_COMBINE,
            )
            graph.links.append((last_index, mult_index))
            graph.links.append((side_index, mult_index))
            if segment.side_source == "forward_input":
                graph.dashed_links.add((side_index, mult_index))
            graph.side_entry_links.add((side_index, mult_index))
            first_after, tail = _add_chain(
                graph,
                after_nodes,
                key_prefix=f"post:{segment_index}",
                attr_last_index=attr_last_index,
            )
            if first_after is not None:
                graph.links.append((mult_index, first_after))
            last_index = tail
            continue

        if isinstance(segment, SeqSegment):
            step = segment.step
            fork_from_input = _should_fork_main_path_from_input(
                segments,
                segment_index,
                last_index,
                attr_last_index,
            )
            if is_method_wrapper(step):
                step_index = _add_method_wrapper_node(
                    graph,
                    step,
                    key=f"seq:{segment_index}:{step.attr_name}",
                )
                _append_step_link(
                    graph,
                    input_index=input_index,
                    last_index=last_index,
                    step_index=step_index,
                    fork_from_input=fork_from_input,
                )
                last_index = step_index
                _track_attr_index(attr_last_index, step.attr_name, step_index)
                continue
            expanded_steps, wrapper = inline_composite_steps(step)
            if wrapper is not None:
                _step_indices, last_index = _add_linear_pipeline_chain(
                    graph,
                    expanded_steps,
                    wrapper=wrapper,
                    key_prefix=f"seq:{segment_index}:{step.attr_name}",
                    attr_last_index=attr_last_index,
                    input_index=input_index,
                    last_index=last_index,
                    fork_from_input=fork_from_input,
                )
                _track_attr_index(attr_last_index, wrapper.attr_name, last_index)
                continue
            for sub_index, sub_step in enumerate(expanded_steps):
                step_index = _add_node(
                    graph,
                    key=f"seq:{segment_index}:{step.attr_name}:{sub_step.attr_name}:{sub_index}",
                    block=sub_step,
                    label=inline_wrapper_step_label(None, sub_step, sub_index),
                )
                use_fork = fork_from_input and sub_index == 0
                _append_step_link(
                    graph,
                    input_index=input_index,
                    last_index=last_index,
                    step_index=step_index,
                    fork_from_input=use_fork,
                )
                last_index = step_index
                _track_attr_index(attr_last_index, sub_step.attr_name, step_index)
            if expanded_steps:
                _track_attr_index(attr_last_index, step.attr_name, last_index)

    return graph


FLOATING_PORT_LABEL_CLEARANCE = 0.46
DETAIL_LAYER_GAP = 2 * min_vertical_block_gap()
DETAIL_TOP_INSET = 0.04
DETAIL_BOTTOM_INSET = 0.04


def _rendered_label_and_sublabel(spec: GraphNodeSpec) -> tuple[str, str | None]:
    """Label pair used when drawing a graph node (matches render.py inline ports)."""
    label = spec.label
    sublabel = spec.sublabel or block_sublabel(spec.block)
    if spec.port_label and spec.port_style == "inline":
        label = spec.port_label
        if spec.block is not None:
            sublabel = spec.block.attr_name
    return label, sublabel


def _diagram_size_for_rendered_spec(spec: GraphNodeSpec) -> tuple[float, float]:
    from visualizer.sizing import estimate_block_size

    label, sublabel = _rendered_label_and_sublabel(spec)
    return estimate_block_size(label, sublabel)


def _diagram_size_for_spec(spec: GraphNodeSpec) -> tuple[float, float]:
    if _is_combine_synthetic(spec.synthetic):
        return COMBINE_OP_SIZE, COMBINE_OP_SIZE
    if spec.synthetic == SYNTHETIC_INPUT:
        return 0.90, 0.38
    if spec.synthetic == SYNTHETIC_HIDDEN:
        return 0.62, 0.30
    return _diagram_size_for_rendered_spec(spec)


def _topological_layers(graph: ComputationGraph) -> list[list[int]]:
    """Group node indices into layers for tight vertical stacking."""
    node_count = len(graph.nodes)
    if node_count == 0:
        return []

    incoming = [0] * node_count
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for src, tgt in graph.links:
        adjacency[src].append(tgt)
        incoming[tgt] += 1

    layer = [0] * node_count
    queue = [index for index in range(node_count) if incoming[index] == 0]
    while queue:
        node = queue.pop(0)
        for tgt in adjacency[node]:
            layer[tgt] = max(layer[tgt], layer[node] + 1)
            incoming[tgt] -= 1
            if incoming[tgt] == 0:
                queue.append(tgt)

    layers_map: dict[int, list[int]] = {}
    for index, layer_id in enumerate(layer):
        layers_map.setdefault(layer_id, []).append(index)
    return [layers_map[layer_id] for layer_id in sorted(layers_map)]


def _assign_layered_vertical_positions(
    positions: list[LayoutPosition],
    layers: list[list[int]],
    *,
    top_y: float,
    min_gap: float = DETAIL_LAYER_GAP,
) -> None:
    """Stack graph layers downward using only the space each row needs."""
    cursor_top = top_y - DETAIL_TOP_INSET
    for layer_indices in layers:
        layer_positions = [positions[index] for index in layer_indices]
        row_height = max(pos.height for pos in layer_positions)
        for pos in layer_positions:
            pos.top_y = cursor_top
        cursor_top -= row_height + min_gap


def _node_content_left(pos: LayoutPosition) -> float:
    """Left edge of a positioned node including any floating port label."""
    left = pos.cx - pos.width / 2
    if pos.spec.port_style == "floating" and pos.spec.port_label:
        left -= FLOATING_PORT_LABEL_CLEARANCE
    return left


def _node_content_right(pos: LayoutPosition) -> float:
    return pos.cx + pos.width / 2


def _center_positions_horizontally(positions: list[LayoutPosition], cx: float) -> None:
    """Shift nodes so diagram content (boxes + port labels) is centered on cx."""
    if not positions:
        return
    min_left = min(_node_content_left(pos) for pos in positions)
    max_right = max(_node_content_right(pos) for pos in positions)
    shift = cx - (min_left + max_right) / 2
    if abs(shift) < 0.001:
        return
    for pos in positions:
        pos.cx += shift


def layout_computation_graph(
    graph: ComputationGraph,
    *,
    cx: float,
    top_y: float,
    block_w: float,
    block_h: float | None = None,
) -> tuple[list[LayoutPosition], list[tuple[int, int]]]:
    """Run Sugiyama layout and map coordinates into the diagram frame."""
    if not graph.nodes:
        return [], []

    if len(graph.nodes) == 1:
        spec = graph.nodes[0]
        diagram_w, diagram_h = _diagram_size_for_spec(spec)
        width = min(block_w - 0.4, diagram_w)
        height = diagram_h
        return [
            LayoutPosition(
                spec=spec,
                cx=cx,
                top_y=top_y - DETAIL_TOP_INSET,
                width=width,
                height=height,
            )
        ], []

    diagram_sizes = [_diagram_size_for_spec(spec) for spec in graph.nodes]
    gl_nodes: list[dict[str, float]] = []
    for diagram_w, diagram_h in diagram_sizes:
        gl_nodes.append({"width": diagram_w * PIXELS_PER_UNIT, "height": diagram_h * PIXELS_PER_UNIT})

    links = [{"source": src, "target": tgt} for src, tgt in graph.links]
    canvas_w = max(320.0, block_w * 110.0)
    max_layer_h = max(height for _width, height in diagram_sizes)
    layer_separation = max(48.0, max_layer_h * PIXELS_PER_UNIT + 16.0)
    canvas_h = block_h * 100.0 if block_h is not None else max(240.0, len(graph.nodes) * layer_separation)

    layout = SugiyamaLayout(
        nodes=gl_nodes,
        links=links,
        size=(canvas_w, canvas_h),
        layer_separation=layer_separation,
        node_separation=36.0,
        orientation="top-to-bottom",
    )
    layout.run()

    xs = [node.x for node in layout.nodes]
    min_x = min(xs)
    max_x = max(xs)
    x_span = max_x - min_x

    inner_w = block_w - 0.45
    pad_x = 0.20
    usable_w = max(0.5, inner_w - 2 * pad_x)

    positions: list[LayoutPosition] = []
    for index, spec in enumerate(graph.nodes):
        gl_node = layout.nodes[index]
        if x_span < 1.0:
            px = cx
        else:
            px = cx - usable_w / 2 + ((gl_node.x - min_x) / x_span) * usable_w
        width, height = diagram_sizes[index]
        positions.append(
            LayoutPosition(
                spec=spec,
                cx=px,
                top_y=top_y,
                width=width,
                height=height,
            )
        )

    _assign_layered_vertical_positions(positions, _topological_layers(graph), top_y=top_y)
    _center_positions_horizontally(positions, cx)

    return positions, graph.links


def _estimate_graph_height(graph: ComputationGraph) -> float:
    """Estimate diagram height from stacked layers before layout."""
    if not graph.nodes:
        return 2.0
    layers = _topological_layers(graph)
    heights = [_diagram_size_for_spec(spec)[1] for spec in graph.nodes]
    content = sum(max(heights[index] for index in layer) for layer in layers)
    gaps = DETAIL_LAYER_GAP * max(0, len(layers) - 1)
    return content + gaps + DETAIL_TOP_INSET + DETAIL_BOTTOM_INSET


def _resolve_vertical_overlaps(positions: list[LayoutPosition], *, min_gap: float = DETAIL_LAYER_GAP) -> None:
    """Push nodes down so rendered boxes do not overlap vertically."""
    ordered = sorted(positions, key=lambda pos: pos.top_y, reverse=True)
    for index in range(1, len(ordered)):
        above = ordered[index - 1]
        current = ordered[index]
        allowed_top = above.bottom - min_gap
        if current.top_y > allowed_top:
            current.top_y = allowed_top
