"""Build computation graphs from block trees and lay them out with graph-layout."""

from __future__ import annotations

import re
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
    gated_norm_activation,
    inline_block_frame_label,
    inline_block_frame_sublabel,
    inline_composite_steps,
    inline_wrapper_step_label,
    is_gated_norm_module,
    is_straight_line_module,
    is_method_wrapper,
    block_purpose,
    side_producer_has_activation,
    wrapper_bullet_lines,
)
from visualizer.ast_analyze import SYNTHETIC_ATTENTION, SYNTHETIC_GATE_ACTIVATION
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
    diagram_width: float | None = None
    diagram_height: float | None = None


@dataclass
class InlineFrameSpec:
    """Dotted frame around steps expanded inline from a linear composite sub-block."""

    frame_id: str
    label: str
    sublabel: str | None = None
    node_indices: list[int] = field(default_factory=list)


@dataclass
class ComputationGraph:
    """Directed graph ready for Sugiyama layout."""

    nodes: list[GraphNodeSpec] = field(default_factory=list)
    links: list[tuple[int, int]] = field(default_factory=list)
    dashed_links: set[tuple[int, int]] = field(default_factory=set)
    side_entry_links: set[tuple[int, int]] = field(default_factory=set)
    link_port_labels: dict[tuple[int, int], str] = field(default_factory=dict)
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
    """Add a sequential chain, inlining straight-line composite wrappers."""
    first_index: int | None = None
    previous: int | None = None
    for index, step in enumerate(steps):
        step_port = port_label if index == 0 and first_index is None else None
        step_port_style = port_style if index == 0 and first_index is None else None

        if is_method_wrapper(step):
            node_index = _add_method_wrapper_node(
                graph,
                step,
                key=f"{key_prefix}:{step.attr_name}:{index}",
            )
            if attr_last_index is not None:
                _track_attr_index(attr_last_index, step.attr_name, node_index)
            if first_index is None:
                first_index = node_index
            if previous is not None:
                graph.links.append((previous, node_index))
            previous = node_index
            continue

        expanded_steps, wrapper = inline_composite_steps(step)
        if wrapper is not None:
            chain_indices, tail = _add_linear_pipeline_chain(
                graph,
                expanded_steps,
                wrapper=wrapper,
                key_prefix=f"{key_prefix}:{step.attr_name}",
                attr_last_index=attr_last_index,
                port_label=step_port,
                port_style=step_port_style,
                input_index=None,
                last_index=previous,
            )
            if first_index is None and chain_indices:
                first_index = chain_indices[0]
            previous = tail
            if attr_last_index is not None:
                _track_attr_index(attr_last_index, wrapper.attr_name, tail)
                _track_attr_index(attr_last_index, step.attr_name, tail)
            continue

        for sub_index, sub_step in enumerate(expanded_steps):
            node_index = _add_node(
                graph,
                key=f"{key_prefix}:{step.attr_name}:{sub_step.attr_name}:{sub_index}",
                block=sub_step,
                port_label=step_port if sub_index == 0 else None,
                port_style=step_port_style if sub_index == 0 else None,
            )
            if attr_last_index is not None:
                _track_attr_index(attr_last_index, sub_step.attr_name, node_index)
            if first_index is None:
                first_index = node_index
            if previous is not None:
                graph.links.append((previous, node_index))
            previous = node_index
        if expanded_steps and attr_last_index is not None:
            _track_attr_index(attr_last_index, step.attr_name, previous)
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
    frame = InlineFrameSpec(
        frame_id=wrapper.attr_name,
        label=inline_block_frame_label(wrapper),
        sublabel=inline_block_frame_sublabel(wrapper),
    )
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
            sublabel="" if wrapper is not None else None,
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


def _append_side_producer_link(
    graph: ComputationGraph,
    *,
    source_index: int,
    target_index: int,
) -> None:
    graph.links.append((source_index, target_index))
    graph.dashed_links.add((source_index, target_index))
    graph.side_entry_links.add((source_index, target_index))


def _add_side_producer_index(
    graph: ComputationGraph,
    producer: BlockNode,
    *,
    segment_index: int,
    source_attr: str,
    port_label: str | None,
    port_style: PortStyle | None,
    input_index: int | None,
    attr_last_index: dict[str, int],
) -> int | None:
    """Add a side-path producer, inlining straight-line output gates when possible."""
    expanded_steps, wrapper = inline_composite_steps(producer)
    if wrapper is not None:
        chain_indices, tail = _add_linear_pipeline_chain(
            graph,
            expanded_steps,
            wrapper=wrapper,
            key_prefix=f"sideproducer:{segment_index}:{source_attr}",
            attr_last_index=attr_last_index,
            port_label=port_label,
            port_style=port_style or "inline",
            input_index=input_index,
            last_index=None,
            branch_from_input_dashed=True,
        )
        if tail is not None:
            _track_attr_index(attr_last_index, wrapper.attr_name, tail)
            _track_attr_index(attr_last_index, source_attr, tail)
        return tail

    source_index = _add_node(
        graph,
        key=f"sideproducer:{segment_index}:{source_attr}",
        block=producer,
        port_label=port_label,
        port_style=port_style,
    )
    if input_index is not None:
        _link_forward_input(graph, input_index, source_index, dashed=True)
    _track_attr_index(attr_last_index, source_attr, source_index)
    return source_index


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
            branch_tails: list[tuple[int, str]] = []
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
                    branch_tails.append((tail, branch.port_label))
            merge_steps, merge_wrapper = inline_composite_steps(segment.merge)
            if merge_wrapper is not None:
                merge_indices, merge_tail = _add_linear_pipeline_chain(
                    graph,
                    merge_steps,
                    wrapper=merge_wrapper,
                    key_prefix=f"merge:{segment_index}",
                    attr_last_index=attr_last_index,
                )
                merge_first = merge_indices[0] if merge_indices else None
                if merge_first is not None:
                    for tail, port_label in branch_tails:
                        graph.links.append((tail, merge_first))
                        graph.link_port_labels[(tail, merge_first)] = port_label
                last_index = merge_tail
                if merge_tail is not None:
                    _track_attr_index(attr_last_index, merge_wrapper.attr_name, merge_tail)
            else:
                merge_index = _add_node(
                    graph,
                    key=f"merge:{segment_index}",
                    block=segment.merge,
                )
                for tail, port_label in branch_tails:
                    graph.links.append((tail, merge_index))
                    graph.link_port_labels[(tail, merge_index)] = port_label
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

            if is_gated_norm_module(consumer):
                norm_index = _add_node(
                    graph,
                    key=f"sidefeed:{segment_index}:{consumer.attr_name}:norm",
                    block=consumer,
                    label="RMSNorm",
                )
                if last_index is not None:
                    graph.links.append((last_index, norm_index))
                elif input_index is not None:
                    graph.links.append((input_index, norm_index))

                combine_index = _add_node(
                    graph,
                    key=f"sidefeed:{segment_index}:{consumer.attr_name}:mul",
                    label="×",
                    sublabel="× gate",
                    synthetic=SYNTHETIC_COMBINE,
                )
                graph.links.append((norm_index, combine_index))

                activation = gated_norm_activation(consumer)
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
                        producer = segment.side_producer_nodes.get(source_attr)
                        if producer is None:
                            continue
                        source_index = _add_side_producer_index(
                            graph,
                            producer,
                            segment_index=segment_index,
                            source_attr=source_attr,
                            port_label=side.port_label,
                            port_style="inline",
                            input_index=input_index,
                            attr_last_index=attr_last_index,
                        )
                    if source_index is None:
                        continue
                    gate_index = source_index
                    producer = segment.side_producer_nodes.get(source_attr)
                    if (
                        activation
                        and producer is not None
                        and not side_producer_has_activation(producer)
                    ):
                        gate_index = _add_node(
                            graph,
                            key=f"sidefeed:{segment_index}:{consumer.attr_name}:gate_act",
                            label=activation,
                            sublabel=None,
                            synthetic=SYNTHETIC_GATE_ACTIVATION,
                        )
                        _append_side_producer_link(graph, source_index=source_index, target_index=gate_index)
                    _append_side_producer_link(graph, source_index=gate_index, target_index=combine_index)

                last_index = combine_index
                _track_attr_index(attr_last_index, consumer.attr_name, combine_index)
                continue

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
                    producer = segment.side_producer_nodes.get(source_attr)
                    if producer is None:
                        continue
                    source_index = _add_side_producer_index(
                        graph,
                        producer,
                        segment_index=segment_index,
                        source_attr=source_attr,
                        port_label=side.port_label,
                        port_style="inline",
                        input_index=input_index,
                        attr_last_index=attr_last_index,
                    )
                    if source_index is None:
                        continue
                _append_side_producer_link(graph, source_index=source_index, target_index=consumer_index)
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


def _rendered_label_and_sublabel(
    spec: GraphNodeSpec,
    *,
    inline_frame_members: frozenset[int] | None = None,
    node_index: int | None = None,
) -> tuple[str, str | None]:
    """Label pair used when drawing a graph node (matches render.py inline ports)."""
    label = spec.label
    if inline_frame_members is not None and node_index is not None and node_index in inline_frame_members:
        sublabel = None
    elif spec.sublabel is not None:
        sublabel = spec.sublabel or None
    else:
        sublabel = block_sublabel(spec.block)
    if spec.port_label and spec.port_style == "inline":
        label = spec.port_label
        if (
            spec.block is not None
            and not (
                inline_frame_members is not None
                and node_index is not None
                and node_index in inline_frame_members
            )
        ):
            sublabel = spec.block.attr_name
    return label, sublabel


def inline_frame_member_indices(graph: ComputationGraph) -> frozenset[int]:
    """Node indices grouped inside dotted inline frames."""
    return frozenset(index for frame in graph.inline_frames for index in frame.node_indices)


def measure_graph_node_sizes(
    ax,
    graph: ComputationGraph,
    *,
    input_sublabel: str | None = None,
    title_fontsize: float = 7.6,
) -> None:
    """Measure every tile label at draw time and cache diagram-unit sizes before layout."""
    from visualizer.text_measure import box_label_size

    inline_members = inline_frame_member_indices(graph)
    for index, spec in enumerate(graph.nodes):
        if _is_combine_synthetic(spec.synthetic):
            spec.diagram_width, spec.diagram_height = COMBINE_OP_SIZE, COMBINE_OP_SIZE
            continue
        if spec.synthetic == SYNTHETIC_INPUT:
            width, height = box_label_size(ax, spec.label, input_sublabel, fontsize=7.2)
            spec.diagram_width, spec.diagram_height = width, height
            continue
        if spec.synthetic == SYNTHETIC_HIDDEN:
            width, height = box_label_size(ax, spec.label, None, fontsize=6.5)
            spec.diagram_width, spec.diagram_height = width, height
            continue
        label, sublabel = _rendered_label_and_sublabel(
            spec,
            inline_frame_members=inline_members,
            node_index=index,
        )
        width, height = box_label_size(ax, label, sublabel, fontsize=title_fontsize)
        spec.diagram_width, spec.diagram_height = width, height


def _diagram_size_for_rendered_spec(
    spec: GraphNodeSpec,
    *,
    inline_frame_members: frozenset[int] | None = None,
    node_index: int | None = None,
) -> tuple[float, float]:
    from visualizer.sizing import estimate_block_size

    if spec.diagram_width is not None and spec.diagram_height is not None:
        return spec.diagram_width, spec.diagram_height
    label, sublabel = _rendered_label_and_sublabel(
        spec,
        inline_frame_members=inline_frame_members,
        node_index=node_index,
    )
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


MIN_HORIZONTAL_BLOCK_GAP = 0.14


def _resolve_horizontal_overlaps(
    positions: list[LayoutPosition],
    layers: list[list[int]],
    *,
    min_gap: float = MIN_HORIZONTAL_BLOCK_GAP,
) -> None:
    """Separate nodes in the same layer so boxes and port labels do not overlap."""
    for layer_indices in layers:
        if len(layer_indices) < 2:
            continue
        layer_positions = sorted((positions[index] for index in layer_indices), key=lambda pos: pos.cx)
        for index in range(1, len(layer_positions)):
            left = layer_positions[index - 1]
            right = layer_positions[index]
            overlap = _node_content_right(left) + min_gap - _node_content_left(right)
            if overlap > 0:
                right.cx += overlap


def _fanout_branch_index(spec: GraphNodeSpec) -> int | None:
    """Extract fan-out branch index from node keys like fan0-2:q_proj:0."""
    match = re.match(r"fan\d+-(\d+):", spec.key)
    return int(match.group(1)) if match else None


def _order_fanout_branch_positions(positions: list[LayoutPosition]) -> None:
    """Reorder fan-out layer nodes left-to-right by branch index to reduce wire crossings."""
    layer_groups: dict[float, list[LayoutPosition]] = {}
    for pos in positions:
        branch_index = _fanout_branch_index(pos.spec)
        if branch_index is None:
            continue
        layer_groups.setdefault(pos.top_y, []).append(pos)

    for layer_positions in layer_groups.values():
        if len(layer_positions) < 2:
            continue
        ordered = sorted(layer_positions, key=lambda pos: _fanout_branch_index(pos.spec) or 0)
        left = min(pos.cx - pos.width / 2 for pos in ordered)
        right = max(pos.cx + pos.width / 2 for pos in ordered)
        span = max(right - left, sum(pos.width for pos in ordered) + MIN_HORIZONTAL_BLOCK_GAP * (len(ordered) - 1))
        cursor = left + span / 2 - sum(pos.width for pos in ordered) / 2 - MIN_HORIZONTAL_BLOCK_GAP * (len(ordered) - 1) / 2
        for pos in ordered:
            pos.cx = cursor + pos.width / 2
            cursor += pos.width + MIN_HORIZONTAL_BLOCK_GAP


def _resolve_layout_overlaps(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_vertical_gap: float = DETAIL_LAYER_GAP,
    min_horizontal_gap: float = MIN_HORIZONTAL_BLOCK_GAP,
) -> None:
    """Resolve horizontal and vertical box overlaps after Sugiyama placement."""
    if not positions:
        return
    layers = _topological_layers(graph)
    _resolve_horizontal_overlaps(positions, layers, min_gap=min_horizontal_gap)
    _resolve_vertical_overlaps(positions, min_gap=min_vertical_gap)
    _resolve_horizontal_overlaps(positions, layers, min_gap=min_horizontal_gap)


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
        node_separation=48.0,
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

    layers = _topological_layers(graph)
    _assign_layered_vertical_positions(positions, layers, top_y=top_y)
    _order_fanout_branch_positions(positions)
    _resolve_layout_overlaps(positions, graph)
    _center_positions_horizontally(positions, cx)
    _resolve_layout_overlaps(positions, graph)

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


def _boxes_overlap_horizontally(
    left: LayoutPosition,
    right: LayoutPosition,
    *,
    min_gap: float,
) -> bool:
    return (
        _node_content_right(left) + min_gap > _node_content_left(right)
        and _node_content_right(right) + min_gap > _node_content_left(left)
    )


def _boxes_overlap_vertically(
    above: LayoutPosition,
    below: LayoutPosition,
    *,
    min_gap: float,
) -> bool:
    return below.top_y > above.bottom - min_gap


def _resolve_vertical_overlaps(
    positions: list[LayoutPosition],
    *,
    min_gap: float = DETAIL_LAYER_GAP,
    layer_y_epsilon: float = 1e-6,
) -> None:
    """Push lower nodes down when they overlap a higher node horizontally."""
    changed = True
    while changed:
        changed = False
        ordered = sorted(positions, key=lambda pos: pos.top_y, reverse=True)
        for above_index, above in enumerate(ordered):
            for below in ordered[above_index + 1 :]:
                if abs(above.top_y - below.top_y) <= layer_y_epsilon:
                    continue
                if not _boxes_overlap_horizontally(above, below, min_gap=min_gap):
                    continue
                allowed_top = above.bottom - min_gap
                if below.top_y > allowed_top:
                    below.top_y = allowed_top
                    changed = True
