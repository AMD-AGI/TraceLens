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
    output_gate_combine_sublabel,
    side_producer_has_activation,
    tile_display_labels,
    tile_sublabel,
    wrapper_bullet_lines,
)
from visualizer.ast_analyze import SYNTHETIC_ATTENTION, SYNTHETIC_GATE_ACTIVATION
from visualizer.basic_ops import BasicOpFilter, keep_detail_graph_node
from visualizer.sizing import (
    block_sublabel,
    estimate_block_size_for_node,
    min_horizontal_block_gap,
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

    frame = _start_inline_frame(graph, wrapper) if wrapper is not None and len(steps) > 1 else None
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

    block = expanded_steps[0] if len(expanded_steps) == 1 else producer
    source_index = _add_node(
        graph,
        key=f"sideproducer:{segment_index}:{source_attr}",
        block=block,
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


def _filter_graph_basic_only(graph: ComputationGraph) -> ComputationGraph:
    """Drop modeled-op nodes and bridge links across removed vertices."""
    remove_indices = {
        index
        for index, spec in enumerate(graph.nodes)
        if not keep_detail_graph_node(
            block=spec.block,
            synthetic=spec.synthetic,
            label=spec.label,
            basic_only=True,
        )
    }
    if not remove_indices:
        return graph

    preds: dict[int, list[int]] = {index: [] for index in range(len(graph.nodes))}
    succs: dict[int, list[int]] = {index: [] for index in range(len(graph.nodes))}
    for source, target in graph.links:
        preds[target].append(source)
        succs[source].append(target)

    def _expand_preds(index: int) -> list[int]:
        if index not in remove_indices:
            return [index]
        expanded: list[int] = []
        for source in preds[index]:
            expanded.extend(_expand_preds(source))
        return expanded

    def _expand_succs(index: int) -> list[int]:
        if index not in remove_indices:
            return [index]
        expanded: list[int] = []
        for target in succs[index]:
            expanded.extend(_expand_succs(target))
        return expanded

    bridged_links: set[tuple[int, int]] = {
        (source, target)
        for source, target in graph.links
        if source not in remove_indices and target not in remove_indices
    }
    bridged_port_labels: dict[tuple[int, int], str] = {}
    bridged_dashed: set[tuple[int, int]] = set()
    bridged_side: set[tuple[int, int]] = set()

    for removed in remove_indices:
        for source in preds[removed]:
            for target in succs[removed]:
                port_label = graph.link_port_labels.get((source, removed)) or graph.link_port_labels.get(
                    (removed, target)
                )
                dashed = (source, removed) in graph.dashed_links or (removed, target) in graph.dashed_links
                side = (source, removed) in graph.side_entry_links or (removed, target) in graph.side_entry_links
                for kept_source in _expand_preds(source):
                    for kept_target in _expand_succs(target):
                        if kept_source == kept_target:
                            continue
                        bridged_links.add((kept_source, kept_target))
                        if port_label and (kept_source, kept_target) not in bridged_port_labels:
                            bridged_port_labels[(kept_source, kept_target)] = port_label
                        if dashed:
                            bridged_dashed.add((kept_source, kept_target))
                        if side:
                            bridged_side.add((kept_source, kept_target))

    old_to_new: dict[int, int] = {}
    new_nodes: list[GraphNodeSpec] = []
    for index, spec in enumerate(graph.nodes):
        if index in remove_indices:
            continue
        old_to_new[index] = len(new_nodes)
        new_nodes.append(spec)

    def _remap(index: int) -> int:
        return old_to_new[index]

    filtered = ComputationGraph(
        nodes=new_nodes,
        links=[(_remap(source), _remap(target)) for source, target in bridged_links],
        dashed_links={(_remap(source), _remap(target)) for source, target in bridged_dashed},
        side_entry_links={(_remap(source), _remap(target)) for source, target in bridged_side},
        link_port_labels={
            (_remap(source), _remap(target)): label
            for (source, target), label in bridged_port_labels.items()
            if source not in remove_indices and target not in remove_indices
        },
    )

    for frame in graph.inline_frames:
        kept_indices = [_remap(index) for index in frame.node_indices if index not in remove_indices]
        if len(kept_indices) >= 2:
            filtered.inline_frames.append(
                InlineFrameSpec(
                    frame_id=frame.frame_id,
                    label=frame.label,
                    sublabel=frame.sublabel,
                    node_indices=kept_indices,
                )
            )
        elif len(kept_indices) == 1:
            # Unwrap single-step inline frames after filtering modeled steps away.
            pass

    return filtered


def build_computation_graph(
    root: BlockNode,
    *,
    prefix_steps: list[BlockNode] | None = None,
    include_input: bool = True,
    basic_ops: BasicOpFilter | None = None,
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
                    sublabel="",
                )
                if last_index is not None:
                    graph.links.append((last_index, norm_index))
                elif input_index is not None:
                    graph.links.append((input_index, norm_index))

                gate_producer = None
                for side in segment.sides:
                    if side.source_chain:
                        gate_producer = segment.side_producer_nodes.get(side.source_chain[-1])
                        if gate_producer is not None:
                            break

                combine_sublabel = output_gate_combine_sublabel(gate_producer) or "norm × gate"
                combine_index = _add_node(
                    graph,
                    key=f"sidefeed:{segment_index}:{consumer.attr_name}:mul",
                    label="×",
                    sublabel=combine_sublabel,
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
                        and not is_gated_norm_module(consumer)
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
                side_block = expanded_side[0] if len(expanded_side) == 1 else side
                side_index = _add_node(
                    graph,
                    key=f"side:{segment_index}",
                    block=side_block,
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

    if basic_ops is not None and basic_ops.basic_only:
        return _filter_graph_basic_only(graph)
    return graph


FLOATING_PORT_LABEL_CLEARANCE = 0.46
DETAIL_LAYER_GAP = 2 * min_vertical_block_gap()
DETAIL_TOP_INSET = 0.04
DETAIL_BOTTOM_INSET = 0.04
LAYOUT_CROSSING_ITERATIONS = 256
LAYOUT_EXACT_LAYER_ORDER_MAX = 8


def _graph_link_dicts(graph: ComputationGraph) -> list[dict[str, int]]:
    return [{"source": src, "target": tgt} for src, tgt in graph.links]


def _real_sugiyama_layers(layout: SugiyamaLayout, node_count: int) -> list[list[int]]:
    """Extract per-layer node order from Sugiyama, dropping dummy nodes."""
    layers: list[list[int]] = []
    for layer in layout._layers:
        real = [node for node in layer if node < node_count]
        if real:
            layers.append(real)
    return layers


def _count_layout_crossings(layers: list[list[int]], graph: ComputationGraph) -> int:
    from graph_layout.preprocessing import count_crossings

    return count_crossings(layers, _graph_link_dicts(graph))


def _optimize_layer_order(
    layers: list[list[int]],
    graph: ComputationGraph,
    *,
    iterations: int = LAYOUT_CROSSING_ITERATIONS,
    exact_max: int = LAYOUT_EXACT_LAYER_ORDER_MAX,
) -> list[list[int]]:
    """Minimize edge crossings with barycenter sweeps plus local/exhaustive search."""
    import itertools

    from graph_layout.preprocessing import count_crossings, minimize_crossings_barycenter

    if len(layers) < 2:
        return [list(layer) for layer in layers]

    links = _graph_link_dicts(graph)
    best = minimize_crossings_barycenter(layers, links, iterations=iterations)
    best_count = count_crossings(best, links)

    improved = True
    while improved:
        improved = False
        for layer_idx, layer in enumerate(best):
            if len(layer) <= 1:
                continue
            if len(layer) <= exact_max:
                permutations = itertools.permutations(layer)
            else:
                permutations = []
                current = list(layer)
                for swap_index in range(len(current) - 1):
                    swapped = list(current)
                    swapped[swap_index], swapped[swap_index + 1] = (
                        swapped[swap_index + 1],
                        swapped[swap_index],
                    )
                    permutations.append(tuple(swapped))

            for perm in permutations:
                trial = [list(row) for row in best]
                trial[layer_idx] = list(perm)
                count = count_crossings(trial, links)
                if count < best_count:
                    best = trial
                    best_count = count
                    improved = True
                    break

    return best


def _is_layout_chain_node(spec: GraphNodeSpec) -> bool:
    if spec.synthetic in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}:
        return False
    return not _is_combine_synthetic(spec.synthetic)


def _center_align_vertical_chains(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Center-align stacked blocks on straight-line paths to reduce connector jogs."""
    node_count = len(positions)
    if node_count == 0:
        return

    incoming: list[list[int]] = [[] for _ in range(node_count)]
    outgoing: list[list[int]] = [[] for _ in range(node_count)]
    for source, target in graph.links:
        incoming[target].append(source)
        outgoing[source].append(target)

    frame_members = inline_frame_member_indices(graph)

    def chain_anchor(index: int) -> float | None:
        spec = positions[index].spec
        if not _is_layout_chain_node(spec):
            return None
        if index in frame_members:
            return None
        predecessors = incoming[index]
        if len(predecessors) != 1:
            return None
        predecessor = predecessors[0]
        if not _is_layout_chain_node(positions[predecessor].spec):
            return None
        if len(outgoing[predecessor]) != 1:
            return None
        return positions[predecessor].cx

    for _ in range(node_count):
        changed = False
        for index in range(node_count):
            anchor = chain_anchor(index)
            if anchor is None:
                continue
            if abs(positions[index].cx - anchor) > 1e-6:
                positions[index].cx = anchor
                changed = True
        if not changed:
            break

    for frame in graph.inline_frames:
        indices = _ordered_inline_frame_chain(graph, list(frame.node_indices))
        if len(indices) < 2:
            continue
        frame_cx = sum(positions[index].cx for index in indices) / len(indices)
        for index in indices:
            positions[index].cx = frame_cx


def _pack_ordered_layer_row(
    positions: list[LayoutPosition],
    layer_indices: list[int],
    *,
    anchor_x: float,
    align_left: bool,
    min_gap: float,
) -> None:
    """Place one layer left-to-right using a fixed index order."""
    if not layer_indices:
        return
    layer_positions = [positions[index] for index in layer_indices]
    if len(layer_positions) == 1:
        pos = layer_positions[0]
        pos.cx = anchor_x + pos.width / 2 if align_left else anchor_x
        return
    total_w = sum(pos.width for pos in layer_positions) + min_gap * (len(layer_positions) - 1)
    cursor = anchor_x if align_left else anchor_x - total_w / 2
    for pos in layer_positions:
        pos.cx = cursor + pos.width / 2
        cursor += pos.width + min_gap


def _layer_order_from_positions(
    positions: list[LayoutPosition],
    layers: list[list[int]],
) -> list[list[int]]:
    """Rebuild layer node order from the current left-to-right positions."""
    return [sorted(layer_indices, key=lambda index: positions[index].cx) for layer_indices in layers]


def _rendered_label_and_sublabel(
    spec: GraphNodeSpec,
    *,
    inline_frame_members: frozenset[int] | None = None,
    node_index: int | None = None,
) -> tuple[str, str | None]:
    """Label pair used when drawing a graph node (matches render.py inline ports)."""
    if spec.sublabel is not None:
        return spec.label, spec.sublabel or None
    in_inline_frame = (
        inline_frame_members is not None
        and node_index is not None
        and node_index in inline_frame_members
    )
    return tile_display_labels(
        spec.block,
        spec_label=spec.label,
        in_inline_frame=in_inline_frame,
        port_label=spec.port_label,
        port_style=spec.port_style,
    )


def inline_frame_member_indices(graph: ComputationGraph) -> frozenset[int]:
    """Node indices grouped inside dotted inline frames."""
    return frozenset(index for frame in graph.inline_frames for index in frame.node_indices)


def _ordered_inline_frame_chain(
    graph: ComputationGraph,
    frame_indices: list[int],
) -> list[int]:
    """Return frame member indices in forward link order when they form a chain."""
    if len(frame_indices) <= 1:
        return list(frame_indices)

    index_set = set(frame_indices)
    incoming = {index: 0 for index in frame_indices}
    outgoing: dict[int, list[int]] = {index: [] for index in frame_indices}
    for source, target in graph.links:
        if source in index_set and target in index_set:
            outgoing[source].append(target)
            incoming[target] += 1

    starts = [index for index in frame_indices if incoming[index] == 0]
    if len(starts) != 1:
        return sorted(frame_indices)

    ordered: list[int] = []
    cursor = starts[0]
    while cursor is not None:
        ordered.append(cursor)
        next_nodes = outgoing.get(cursor, [])
        cursor = next_nodes[0] if len(next_nodes) == 1 else None

    if len(ordered) == len(frame_indices):
        return ordered
    return sorted(frame_indices, key=lambda index: index)


def stack_inline_frame_positions(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float | None = None,
) -> None:
    """Re-stack each inline frame column using measured tile heights."""
    gap = min_vertical_block_gap() if min_gap is None else min_gap
    for frame in graph.inline_frames:
        indices = _ordered_inline_frame_chain(graph, list(frame.node_indices))
        if len(indices) < 2:
            if indices:
                positions[indices[0]].cx = sum(positions[index].cx for index in indices) / len(indices)
            continue

        frame_cx = sum(positions[index].cx for index in indices) / len(indices)
        cursor_top = max(positions[index].top_y for index in indices)
        for index in indices:
            pos = positions[index]
            pos.cx = frame_cx
            pos.top_y = cursor_top
            cursor_top -= pos.height + gap


def _inline_frame_internal_pairs(graph: ComputationGraph) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for frame in graph.inline_frames:
        chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
        pairs.update(zip(chain, chain[1:]))
    return pairs


def _shift_node_subtree(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    root: int,
    delta_y: float,
) -> None:
    """Shift ``root`` and all downstream nodes by ``delta_y`` (negative moves up)."""
    if abs(delta_y) < 1e-9:
        return
    visited: set[int] = set()
    queue = [root]
    while queue:
        index = queue.pop(0)
        if index in visited:
            continue
        visited.add(index)
        positions[index].top_y -= delta_y
        for source, target in graph.links:
            if source == index and target not in visited:
                queue.append(target)


def _align_merge_nodes(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    merge_gap: float | None = None,
) -> None:
    """Place merge/combine nodes one layer gap below the deepest incoming branch."""
    gap = DETAIL_LAYER_GAP if merge_gap is None else merge_gap
    incoming: dict[int, list[int]] = {index: [] for index in range(len(positions))}
    for source, target in graph.links:
        incoming[target].append(source)

    for target, sources in incoming.items():
        if len(sources) < 2:
            continue
        deepest_bottom = min(positions[source].bottom for source in sources)
        target_top = deepest_bottom - gap
        if abs(positions[target].top_y - target_top) <= 1e-6:
            continue
        _shift_node_subtree(
            positions,
            graph,
            target,
            positions[target].top_y - target_top,
        )


def _column_horizontal_bounds(
    positions: list[LayoutPosition],
    indices: list[int],
) -> tuple[float, float]:
    left = min(_node_content_left(positions[index]) for index in indices)
    right = max(_node_content_right(positions[index]) for index in indices)
    return left, right


def compact_horizontal_shrink_wrap(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_left: float | None = None,
) -> None:
    """Pack inline-frame columns tightly; align loose nodes from graph connectivity."""
    if not positions:
        return

    gap = min_horizontal_block_gap()
    frame_members = inline_frame_member_indices(graph)
    from visualizer.render import INLINE_FRAME_PAD

    frame_pad = INLINE_FRAME_PAD
    frame_columns: list[list[int]] = []
    for frame in graph.inline_frames:
        indices = _ordered_inline_frame_chain(graph, list(frame.node_indices))
        if indices:
            frame_columns.append(indices)

    def frame_column_bounds(indices: list[int]) -> tuple[float, float]:
        left = min(_node_content_left(positions[index]) for index in indices) - frame_pad
        right = max(_node_content_right(positions[index]) for index in indices) + frame_pad
        return left, right

    if frame_columns:
        frame_columns.sort(key=lambda indices: frame_column_bounds(indices)[0])
        cursor_left: float | None = None
        for indices in frame_columns:
            left, right = frame_column_bounds(indices)
            width = right - left
            if cursor_left is None:
                cursor_left = left
            shift = cursor_left - left
            for index in indices:
                positions[index].cx += shift
            cursor_left += width + gap

    free_indices = [
        index
        for index, pos in enumerate(positions)
        if index not in frame_members
        and pos.spec.synthetic not in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}
    ]
    incoming: dict[int, list[int]] = {index: [] for index in range(len(positions))}
    outgoing: dict[int, list[int]] = {index: [] for index in range(len(positions))}
    for source, target in graph.links:
        incoming[target].append(source)
        outgoing[source].append(target)

    for index in free_indices:
        refs = incoming[index] + outgoing[index]
        if not refs:
            continue
        chain_predecessors = incoming[index]
        if (
            len(chain_predecessors) == 1
            and len(outgoing[chain_predecessors[0]]) == 1
            and _is_layout_chain_node(positions[index].spec)
            and _is_layout_chain_node(positions[chain_predecessors[0]].spec)
        ):
            positions[index].cx = positions[chain_predecessors[0]].cx
            continue
        positions[index].cx = sum(positions[ref].cx for ref in refs) / len(refs)

    if free_indices or frame_columns:
        _resolve_horizontal_overlaps(positions, _topological_layers(graph), min_gap=gap)

    packed = [
        pos
        for pos in positions
        if pos.spec.synthetic not in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}
    ]
    if packed:
        content_left = min(_node_content_left(pos) for pos in packed)
        content_right = max(_node_content_right(pos) for pos in packed)
        content_cx = (content_left + content_right) / 2
        for pos in positions:
            if pos.spec.synthetic in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}:
                pos.cx = content_cx

    if min_left is not None:
        _align_positions_left(positions, min_left)


def repack_inline_frame_columns(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Re-pack inline-frame columns after measured resize; leave loose nodes in place."""
    if not graph.inline_frames:
        return

    from visualizer.render import INLINE_FRAME_PAD

    gap = min_horizontal_block_gap()
    pad = INLINE_FRAME_PAD
    frame_columns: list[list[int]] = []
    for frame in graph.inline_frames:
        indices = _ordered_inline_frame_chain(graph, list(frame.node_indices))
        if indices:
            frame_columns.append(indices)

    if not frame_columns:
        return

    def frame_bounds(indices: list[int]) -> tuple[float, float]:
        left = min(_node_content_left(positions[index]) for index in indices) - pad
        right = max(_node_content_right(positions[index]) for index in indices) + pad
        return left, right

    frame_columns.sort(key=lambda indices: frame_bounds(indices)[0])
    cursor_left: float | None = None
    for indices in frame_columns:
        left, right = frame_bounds(indices)
        width = right - left
        if cursor_left is None:
            cursor_left = left
        shift = cursor_left - left
        for index in indices:
            positions[index].cx += shift
        cursor_left += width + gap


def _inline_frame_internal_gap(
    graph: ComputationGraph,
    layers: list[list[int]],
    *,
    upper_layer_index: int,
) -> float:
    """Gap between two layer rows when they advance the same inline frame chain."""
    if upper_layer_index < 0 or upper_layer_index + 1 >= len(layers):
        return DETAIL_LAYER_GAP

    upper = set(layers[upper_layer_index])
    lower = set(layers[upper_layer_index + 1])
    for frame in graph.inline_frames:
        chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
        for source, target in zip(chain, chain[1:]):
            if source in upper and target in lower:
                return min_vertical_block_gap()
    return DETAIL_LAYER_GAP


def measure_graph_node_sizes(
    ax,
    graph: ComputationGraph,
    *,
    input_sublabel: str | None = None,
    title_fontsize: float = 7.6,
) -> None:
    """Measure every tile label at draw time and cache diagram-unit sizes before layout."""
    from visualizer.text_measure import box_label_size, input_box_label_size

    inline_members = inline_frame_member_indices(graph)
    for index, spec in enumerate(graph.nodes):
        if _is_combine_synthetic(spec.synthetic):
            spec.diagram_width, spec.diagram_height = COMBINE_OP_SIZE, COMBINE_OP_SIZE
            continue
        if spec.synthetic == SYNTHETIC_INPUT:
            width, height = input_box_label_size(ax, spec.label, input_sublabel, fontsize=7.2)
            spec.diagram_width, spec.diagram_height = width, height
            continue
        if spec.synthetic == SYNTHETIC_HIDDEN:
            width, height = input_box_label_size(ax, spec.label, None, fontsize=6.5)
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


INPUT_INLINE_FRAME_CAPTION_CLEARANCE = 0.28


def _assign_layered_vertical_positions(
    positions: list[LayoutPosition],
    layers: list[list[int]],
    *,
    top_y: float,
    min_gap: float = DETAIL_LAYER_GAP,
    graph: ComputationGraph | None = None,
) -> None:
    """Stack graph layers downward using only the space each row needs."""
    cursor_top = top_y - DETAIL_TOP_INSET
    for layer_index, layer_indices in enumerate(layers):
        layer_positions = [positions[index] for index in layer_indices]
        row_height = max(pos.height for pos in layer_positions)
        for pos in layer_positions:
            pos.top_y = cursor_top
        cursor_top -= row_height + min_gap
        if (
            layer_index == 0
            and graph is not None
            and graph.inline_frames
            and layer_index + 1 < len(layers)
        ):
            cursor_top -= INPUT_INLINE_FRAME_CAPTION_CLEARANCE


def _node_content_left(pos: LayoutPosition) -> float:
    """Left edge of a positioned node including any floating port label."""
    left = pos.cx - pos.width / 2
    if pos.spec.port_style == "floating" and pos.spec.port_label:
        left -= FLOATING_PORT_LABEL_CLEARANCE
    return left


def _node_content_right(pos: LayoutPosition) -> float:
    return pos.cx + pos.width / 2


MIN_HORIZONTAL_BLOCK_GAP = min_horizontal_block_gap()


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


def _compact_layer_positions(
    positions: list[LayoutPosition],
    layers: list[list[int]],
    anchor_x: float,
    *,
    min_gap: float = MIN_HORIZONTAL_BLOCK_GAP,
    align_left: bool = False,
) -> None:
    """Pack each topological layer to minimum width using the given node order."""
    for layer_indices in layers:
        _pack_ordered_layer_row(
            positions,
            layer_indices,
            anchor_x=anchor_x,
            align_left=align_left,
            min_gap=min_gap,
        )


def _fanout_branch_index(spec: GraphNodeSpec) -> int | None:
    """Extract fan-out branch index from node keys like fan0-2:q_proj:0."""
    match = re.match(r"fan\d+-(\d+):", spec.key)
    return int(match.group(1)) if match else None


def _fanout_branch_node_groups(positions: list[LayoutPosition]) -> dict[int, list[int]]:
    """Group positioned nodes by fan-out branch index."""
    groups: dict[int, list[int]] = {}
    for index, pos in enumerate(positions):
        branch_index = _fanout_branch_index(pos.spec)
        if branch_index is None:
            continue
        groups.setdefault(branch_index, []).append(index)
    return groups


def _order_fanout_branch_positions(positions: list[LayoutPosition]) -> None:
    """Reorder entire fan-out branch columns left-to-right to reduce wire crossings."""
    branch_groups = _fanout_branch_node_groups(positions)
    if len(branch_groups) < 2:
        return

    columns: list[tuple[int, list[int], float, float]] = []
    for branch_index in sorted(branch_groups):
        indices = branch_groups[branch_index]
        left = min(_node_content_left(positions[index]) for index in indices)
        right = max(_node_content_right(positions[index]) for index in indices)
        columns.append((branch_index, indices, left, right))

    cursor_left = min(left for _branch, _indices, left, _right in columns)
    for _branch_index, indices, left, right in columns:
        shift = cursor_left - left
        for index in indices:
            positions[index].cx += shift
        cursor_left += (right - left) + MIN_HORIZONTAL_BLOCK_GAP


def _align_fanout_branch_columns(positions: list[LayoutPosition]) -> None:
    """Give every node in a fan-out branch the same column center."""
    for indices in _fanout_branch_node_groups(positions).values():
        if len(indices) < 2:
            continue
        column_cx = sum(positions[index].cx for index in indices) / len(indices)
        for index in indices:
            positions[index].cx = column_cx


def _resolve_branch_column_overlaps(
    positions: list[LayoutPosition],
    *,
    min_gap: float,
) -> None:
    """Separate fan-out branch columns without breaking intra-column alignment."""
    branch_groups = _fanout_branch_node_groups(positions)
    if len(branch_groups) < 2:
        return

    columns: list[tuple[list[int], float, float]] = []
    for branch_index in sorted(branch_groups):
        indices = branch_groups[branch_index]
        left = min(_node_content_left(positions[index]) for index in indices)
        right = max(_node_content_right(positions[index]) for index in indices)
        columns.append((indices, left, right))

    columns.sort(key=lambda item: item[1])
    for index in range(1, len(columns)):
        prev_indices, prev_left, prev_right = columns[index - 1]
        curr_indices, curr_left, curr_right = columns[index]
        overlap = prev_right + min_gap - curr_left
        if overlap <= 0:
            continue
        for node_index in curr_indices:
            positions[node_index].cx += overlap
        columns[index] = (
            curr_indices,
            curr_left + overlap,
            curr_right + overlap,
        )


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
    branch_groups = _fanout_branch_node_groups(positions)
    if branch_groups:
        _align_fanout_branch_columns(positions)
        _resolve_branch_column_overlaps(positions, min_gap=min_horizontal_gap)
    else:
        _resolve_horizontal_overlaps(positions, layers, min_gap=min_horizontal_gap)
    _resolve_vertical_overlaps(positions, min_gap=min_vertical_gap)
    if branch_groups:
        _align_fanout_branch_columns(positions)
        _resolve_branch_column_overlaps(positions, min_gap=min_horizontal_gap)
    else:
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


def _align_positions_left(positions: list[LayoutPosition], content_left: float) -> None:
    """Shift nodes so the leftmost content edge starts at ``content_left``."""
    if not positions:
        return
    min_left = min(_node_content_left(pos) for pos in positions)
    shift = content_left - min_left
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
    content_left: float | None = None,
) -> tuple[list[LayoutPosition], list[tuple[int, int]]]:
    """Run Sugiyama layout and map coordinates into the diagram frame."""
    if not graph.nodes:
        return [], []

    if len(graph.nodes) == 1:
        spec = graph.nodes[0]
        diagram_w, diagram_h = _diagram_size_for_spec(spec)
        width = min(block_w - 0.4, diagram_w)
        height = diagram_h
        node_cx = (content_left + width / 2) if content_left is not None else cx
        return [
            LayoutPosition(
                spec=spec,
                cx=node_cx,
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
        crossing_iterations=LAYOUT_CROSSING_ITERATIONS,
    )
    layout.run()

    node_count = len(graph.nodes)
    layers = _real_sugiyama_layers(layout, node_count) or _topological_layers(graph)
    layers = _optimize_layer_order(layers, graph)

    align_left = content_left is not None
    anchor_x = content_left if align_left else cx

    positions: list[LayoutPosition] = []
    for index, spec in enumerate(graph.nodes):
        width, height = diagram_sizes[index]
        positions.append(
            LayoutPosition(
                spec=spec,
                cx=anchor_x,
                top_y=top_y,
                width=width,
                height=height,
            )
        )

    _assign_layered_vertical_positions(positions, layers, top_y=top_y, graph=graph)
    _compact_layer_positions(positions, layers, anchor_x, align_left=align_left)
    stack_inline_frame_positions(positions, graph)
    _align_merge_nodes(positions, graph)
    _center_align_vertical_chains(positions, graph)
    layers = _optimize_layer_order(layers, graph)
    _compact_layer_positions(positions, layers, anchor_x, align_left=align_left)
    compact_horizontal_shrink_wrap(positions, graph, min_left=content_left if align_left else None)
    _center_align_vertical_chains(positions, graph)
    _order_fanout_branch_positions(positions)
    _align_fanout_branch_columns(positions)
    layers = _optimize_layer_order(_layer_order_from_positions(positions, layers), graph)
    _compact_layer_positions(positions, layers, anchor_x, align_left=align_left)
    _center_align_vertical_chains(positions, graph)
    _resolve_layout_overlaps(positions, graph)
    if align_left:
        _align_positions_left(positions, content_left)
    else:
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
    gaps = 0.0
    for layer_index in range(max(0, len(layers) - 1)):
        gaps += _inline_frame_internal_gap(graph, layers, upper_layer_index=layer_index)
    if graph.inline_frames and len(layers) > 1:
        gaps += INPUT_INLINE_FRAME_CAPTION_CLEARANCE
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
