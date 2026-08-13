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
    TensorPortsSegment,
    collect_computation_segments,
    flatten_computation_segments,
    gated_norm_activation,
    inline_block_frame_label,
    inline_block_frame_sublabel,
    inline_composite_steps,
    inline_wrapper_step_label,
    is_basic_op_tile,
    is_gated_norm_module,
    is_kernel_pipeline_tree,
    is_situ_gated_mlp,
    is_straight_line_module,
    is_method_wrapper,
    block_purpose,
    output_gate_combine_sublabel,
    gated_norm_combine_sublabel,
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

# Match render.py MERGE_RADIUS + MERGE_CLEARANCE for compact combine (×) nodes.
COMBINE_OP_SIZE = 0.32
LABELED_COMBINE_DEFAULT_WIDTH = 1.35
LABELED_COMBINE_DEFAULT_HEIGHT = 0.34

SYNTHETIC_INPUT = "@input"
SYNTHETIC_HIDDEN = "@hidden_states"  # legacy alias; replaced by SYNTHETIC_INPUT in graphs
SYNTHETIC_TENSOR = "@tensor"
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
    inline_binary_operand_links: set[tuple[int, int]] = field(default_factory=set)
    link_port_labels: dict[tuple[int, int], str] = field(default_factory=dict)
    inline_frames: list[InlineFrameSpec] = field(default_factory=list)


@dataclass(frozen=True)
class ForkJoinCluster:
    """Main spine and side branch meeting at a combine node, then continuing downstream."""

    main_source: int
    side_source: int
    main_branch: int
    join: int
    tail: int


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
        from visualizer.ast_analyze import is_compact_combine_label

        if is_compact_combine_label(spec.label):
            return to_layout_pixels(COMBINE_OP_SIZE, COMBINE_OP_SIZE)
        return to_layout_pixels(LABELED_COMBINE_DEFAULT_WIDTH, LABELED_COMBINE_DEFAULT_HEIGHT)
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
    display = label if label is not None else (block.label if block else key)
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
    return _add_node(graph, key=key, block=step, label=label, sublabel=None)


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
    basic_ops: BasicOpFilter | None = None,
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

        expanded_steps, wrapper = inline_composite_steps(step, basic_ops=basic_ops)
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
) -> None:
    """Link the synthetic forward input to a downstream node (residual/skip feeds stay solid)."""
    graph.links.append((input_index, target_index))


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


def _add_kernel_pipeline_merge_chain(
    graph: ComputationGraph,
    merge_steps: list[BlockNode],
    *,
    key_prefix: str,
    attr_last_index: dict[str, int] | None = None,
) -> tuple[list[int], int | None]:
    """Expand a kernel pipeline in its own sub-frame and append the output kernel step."""
    if len(merge_steps) != 2:
        return _add_linear_pipeline_chain(
            graph,
            merge_steps,
            wrapper=merge_steps[0] if merge_steps else None,
            key_prefix=key_prefix,
            attr_last_index=attr_last_index,
        )

    pipeline_step, output_step = merge_steps
    inner_steps, pipeline_wrapper = inline_composite_steps(pipeline_step)
    pipeline_indices, pipeline_tail = _add_linear_pipeline_chain(
        graph,
        inner_steps,
        wrapper=pipeline_wrapper,
        key_prefix=f"{key_prefix}:pipeline",
        attr_last_index=attr_last_index,
    )
    output_index = _add_node(
        graph,
        key=f"{key_prefix}:output",
        block=output_step,
    )
    linked_output = False
    if attr_last_index is not None:
        for pred_attr in output_step.kernel_predecessors:
            pred_index = attr_last_index.get(pred_attr)
            if pred_index is not None:
                graph.links.append((pred_index, output_index))
                linked_output = True
    if not linked_output and pipeline_tail is not None:
        graph.links.append((pipeline_tail, output_index))
    if attr_last_index is not None:
        _track_attr_index(attr_last_index, output_step.attr_name, output_index)
        if pipeline_wrapper is not None:
            _track_attr_index(attr_last_index, pipeline_wrapper.attr_name, pipeline_tail)
    return list(pipeline_indices) + [output_index], output_index


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
        inner_steps, inner_wrapper = inline_composite_steps(sub_step)
        if inner_wrapper is not None:
            inner_indices, inner_tail = _add_linear_pipeline_chain(
                graph,
                inner_steps,
                wrapper=inner_wrapper,
                key_prefix=f"{key_prefix}:{sub_step.attr_name}",
                attr_last_index=attr_last_index,
                input_index=input_index if sub_index == 0 else None,
                last_index=chain_last if sub_index == 0 else indices[-1],
                fork_from_input=fork_from_input and sub_index == 0,
                branch_from_input_dashed=branch_from_input_dashed and sub_index == 0,
                port_label=port_label if sub_index == 0 else None,
                port_style=port_style if sub_index == 0 else None,
            )
            if frame is not None:
                for inner_index in inner_indices:
                    _append_inline_frame_node(frame, inner_index)
            if attr_last_index is not None:
                _track_attr_index(attr_last_index, sub_step.attr_name, inner_tail)
                _track_attr_index(attr_last_index, inner_wrapper.attr_name, inner_tail)
            indices.extend(inner_indices)
            chain_last = inner_tail
            continue

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
                _link_forward_input(graph, input_index, step_index)
            else:
                use_fork = fork_from_input and input_index is not None
                _append_step_link(
                    graph,
                    input_index=input_index,
                    last_index=chain_last,
                    step_index=step_index,
                    fork_from_input=use_fork,
                )
        else:
            graph.links.append((indices[-1], step_index))

        if _is_binary_kernel_op_label(sub_step.label) and len(indices) >= 2:
            _append_inline_binary_operand_link(
                graph,
                source_index=indices[-2],
                target_index=step_index,
            )

        indices.append(step_index)

    return indices, indices[-1]


def _add_situ_gated_mlp_chain(
    graph: ComputationGraph,
    node: BlockNode,
    *,
    key_prefix: str,
    attr_last_index: dict[str, int] | None = None,
    input_index: int | None = None,
    last_index: int | None = None,
    branch_from_input_dashed: bool = False,
    port_label: str | None = None,
    port_style: PortStyle | None = None,
    create_outer_frame: bool = False,
) -> tuple[list[int], int | None]:
    """Expand gate/up → Situ × up → down with both multiply inputs visible."""
    from visualizer.block_tree import _situ_gated_mlp_parts

    parts = _situ_gated_mlp_parts(node)
    if parts is None:
        return [], last_index
    gate, up, act_fn, situ, down = parts

    outer_frame = _start_inline_frame(graph, node) if create_outer_frame else None
    indices: list[int] = []

    def _track(node_block: BlockNode, index: int | None) -> None:
        if attr_last_index is not None and index is not None:
            _track_attr_index(attr_last_index, node_block.attr_name, index)

    def _append_outer(index: int) -> None:
        if outer_frame is not None:
            _append_inline_frame_node(outer_frame, index)

    gate_index = _add_node(
        graph,
        key=f"{key_prefix}:gate_proj",
        block=gate,
        port_label=port_label,
        port_style=port_style,
    )
    if branch_from_input_dashed and input_index is not None:
        _link_forward_input(graph, input_index, gate_index)
    else:
        _append_step_link(
            graph,
            input_index=input_index,
            last_index=last_index,
            step_index=gate_index,
            fork_from_input=last_index is None and input_index is not None,
        )
    _track(gate, gate_index)
    _append_outer(gate_index)
    indices.append(gate_index)

    up_index = _add_node(
        graph,
        key=f"{key_prefix}:up_proj",
        block=up,
        port_label="up",
        port_style="inline",
    )
    if input_index is not None:
        _link_forward_input(graph, input_index, up_index)
    _track(up, up_index)
    indices.append(up_index)

    act_frame = _start_inline_frame(graph, act_fn)
    situ_index = _add_node(
        graph,
        key=f"{key_prefix}:situ",
        block=situ,
    )
    graph.links.append((gate_index, situ_index))
    _append_inline_frame_node(act_frame, situ_index)
    _track(situ, situ_index)
    _append_outer(situ_index)
    indices.append(situ_index)

    mult_index = _add_node(
        graph,
        key=f"{key_prefix}:mul",
        label="×",
        sublabel=None,
        synthetic=SYNTHETIC_COMBINE,
    )
    graph.links.append((situ_index, mult_index))
    graph.links.append((up_index, mult_index))
    graph.side_entry_links.add((up_index, mult_index))
    _append_inline_frame_node(act_frame, mult_index)
    _track(act_fn, mult_index)
    _append_outer(mult_index)
    indices.append(mult_index)

    down_index = _add_node(
        graph,
        key=f"{key_prefix}:down_proj",
        block=down,
    )
    graph.links.append((mult_index, down_index))
    _track(down, down_index)
    _append_outer(down_index)
    indices.append(down_index)

    return indices, down_index


def _is_binary_kernel_op_label(label: str) -> bool:
    """True for inline kernel sub-ops that combine two prior chain values."""
    return label.strip() in {"×", "÷", "+", "−"}


def _append_inline_binary_operand_link(
    graph: ComputationGraph,
    *,
    source_index: int,
    target_index: int,
) -> None:
    """Wire the left operand of a binary inline op into its target tile."""
    link = (source_index, target_index)
    if link not in graph.links:
        graph.links.append(link)
    graph.side_entry_links.add(link)
    graph.inline_binary_operand_links.add(link)


def _append_side_producer_link(
    graph: ComputationGraph,
    *,
    source_index: int,
    target_index: int,
) -> None:
    graph.links.append((source_index, target_index))
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
    basic_ops: BasicOpFilter | None = None,
) -> int | None:
    """Add a side-path producer, inlining straight-line output gates when possible."""
    expanded_steps, wrapper = inline_composite_steps(producer, basic_ops=basic_ops)
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
        _link_forward_input(graph, input_index, source_index)
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

    def _expand_preds(index: int, visiting: frozenset[int] | None = None) -> list[int]:
        if index not in remove_indices:
            return [index]
        active = visiting or frozenset()
        if index in active:
            return []
        expanded: list[int] = []
        for source in preds[index]:
            expanded.extend(_expand_preds(source, active | {index}))
        return expanded

    def _expand_succs(index: int, visiting: frozenset[int] | None = None) -> list[int]:
        if index not in remove_indices:
            return [index]
        active = visiting or frozenset()
        if index in active:
            return []
        expanded: list[int] = []
        for target in succs[index]:
            expanded.extend(_expand_succs(target, active | {index}))
        return expanded

    bridged_links: set[tuple[int, int]] = set()
    bridged_port_labels: dict[tuple[int, int], str] = {}
    for source, target in graph.links:
        if source in remove_indices or target in remove_indices:
            continue
        bridged_links.add((source, target))
        port_label = graph.link_port_labels.get((source, target))
        if port_label:
            bridged_port_labels[(source, target)] = port_label
    bridged_dashed: set[tuple[int, int]] = set()
    bridged_side: set[tuple[int, int]] = set()
    bridged_inline_binary: set[tuple[int, int]] = set()

    for removed in remove_indices:
        for source in preds[removed]:
            for target in succs[removed]:
                port_label = graph.link_port_labels.get((source, removed)) or graph.link_port_labels.get(
                    (removed, target)
                )
                dashed = (source, removed) in graph.dashed_links or (removed, target) in graph.dashed_links
                side = (source, removed) in graph.side_entry_links or (removed, target) in graph.side_entry_links
                inline_binary = (
                    (source, removed) in graph.inline_binary_operand_links
                    or (removed, target) in graph.inline_binary_operand_links
                )
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
                        if inline_binary:
                            bridged_inline_binary.add((kept_source, kept_target))

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
        inline_binary_operand_links={
            (_remap(source), _remap(target)) for source, target in bridged_inline_binary
        },
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


def _producer_label_from_attr(attr_name: str) -> str:
    """Map a modeling attr name to a short upstream operator label."""
    lowered = attr_name.lower()
    if "proj" in lowered:
        return "Linear"
    if "conv" in lowered:
        return "Conv1d"
    if "norm" in lowered:
        return "RMSNorm"
    return attr_name.replace("_", " ")


def _tensor_port_input_sublabels(attention_inputs: dict[str, list[str]]) -> dict[str, str]:
    """Format per-port upstream hints like ``← Linear`` from provenance chains."""
    labels: dict[str, str] = {}
    for port, chain in attention_inputs.items():
        if not chain:
            continue
        source = _producer_label_from_attr(chain[-1])
        head = source.split(" in ", 1)[0]
        labels[port] = f"← {head}"
    return labels


def _add_tensor_ports_segment(
    graph: ComputationGraph,
    segment: TensorPortsSegment,
    *,
    key_prefix: str,
    port_sublabels: dict[str, str] | None = None,
) -> int | None:
    """Add labeled tensor inputs fanning into the pipeline steps that consume them."""
    if not segment.steps:
        return None

    step_indices: dict[str, int] = {}
    step_entries: dict[str, int] = {}
    for step_index, step in enumerate(segment.steps):
        if step.children and len(step.children) >= 2:
            sub_indices, sub_tail = _add_linear_pipeline_chain(
                graph,
                step.children,
                wrapper=step,
                key_prefix=f"{key_prefix}:pipeline:{step.attr_name}",
            )
            step_indices[step.attr_name] = sub_tail if sub_tail is not None else sub_indices[-1]
            step_entries[step.attr_name] = sub_indices[0]
            for pred_attr in step.kernel_predecessors:
                pred_index = step_indices.get(pred_attr)
                if pred_index is not None:
                    graph.links.append((pred_index, sub_indices[0]))
            continue

        node_index = _add_node(
            graph,
            key=f"{key_prefix}:pipeline:{step.attr_name}:{step_index}",
            block=step,
        )
        step_indices[step.attr_name] = node_index
        step_entries[step.attr_name] = node_index
        for pred_attr in step.kernel_predecessors:
            pred_index = step_indices.get(pred_attr)
            if pred_index is not None:
                graph.links.append((pred_index, node_index))

    default_target = segment.steps[0].attr_name
    for label_index, label in enumerate(segment.labels):
        target_attr = segment.targets.get(label, default_target)
        target_index = step_entries.get(target_attr)
        if target_index is None:
            target_index = step_indices.get(target_attr)
        if target_index is None:
            continue
        port_index = _add_node(
            graph,
            key=f"{key_prefix}:tensor:{label_index}",
            label=label,
            sublabel=(port_sublabels or {}).get(label),
            synthetic=SYNTHETIC_TENSOR,
        )
        graph.links.append((port_index, target_index))

    return step_indices.get(segment.steps[-1].attr_name)


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

    resolved_include_input = include_input and not root.tensor_input_labels
    segments = flatten_computation_segments(root)
    input_index = _add_forward_input(graph, root) if resolved_include_input else None

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

    if is_situ_gated_mlp(root):
        _, last_index = _add_situ_gated_mlp_chain(
            graph,
            root,
            key_prefix=root.attr_name,
            attr_last_index=attr_last_index,
            input_index=input_index,
            last_index=last_index,
            create_outer_frame=False,
        )
        return graph

    for segment_index, segment in enumerate(segments):
        if isinstance(segment, TensorPortsSegment):
            tail = _add_tensor_ports_segment(
                graph,
                segment,
                key_prefix=f"{root.attr_name}:tensor{segment_index}",
                port_sublabels=_tensor_port_input_sublabels(root.attention_inputs),
            )
            if tail is not None:
                last_index = tail
                _track_attr_index(attr_last_index, segment.steps[-1].attr_name, tail)
            continue

        if isinstance(segment, FanOutSegment):
            branch_tails: list[int] = []
            branch_specs: list = []
            for branch_index, branch in enumerate(segment.branches):
                first_index, tail = _add_chain(
                    graph,
                    branch.steps,
                    key_prefix=f"fan{segment_index}-{branch_index}",
                    attr_last_index=attr_last_index,
                    basic_ops=basic_ops,
                    port_label=branch.port_label,
                    port_style=branch.port_style or "floating",
                )
                branch_specs.append(branch)
                if input_index is not None and first_index is not None:
                    _link_forward_input(graph, input_index, first_index)
                if tail is not None:
                    branch_tails.append(tail)
            merge_steps, merge_wrapper = inline_composite_steps(segment.merge, basic_ops=basic_ops)
            if merge_wrapper is not None and merge_wrapper.class_name == "KernelPipeline":
                merge_index = _add_node(
                    graph,
                    key=f"merge:{segment_index}:pipeline",
                    block=merge_wrapper,
                )
                for tail, branch in zip(branch_tails, branch_specs):
                    graph.links.append((tail, merge_index))
                    graph.link_port_labels[(tail, merge_index)] = branch.port_label
                last_index = merge_index
                if merge_wrapper is not None:
                    _track_attr_index(attr_last_index, merge_wrapper.attr_name, merge_index)
            elif merge_wrapper is not None and len(merge_steps) == 2 and merge_steps[1].class_name == "KernelOutput":
                merge_indices, merge_tail = _add_kernel_pipeline_merge_chain(
                    graph,
                    merge_steps,
                    key_prefix=f"merge:{segment_index}",
                    attr_last_index=attr_last_index,
                )
                merge_first = merge_indices[0] if merge_indices else None
                if merge_first is not None:
                    for tail in branch_tails:
                        graph.links.append((tail, merge_first))
                last_index = merge_tail
                if merge_tail is not None:
                    _track_attr_index(attr_last_index, merge_wrapper.attr_name, merge_tail)
            elif merge_wrapper is not None:
                merge_indices, merge_tail = _add_linear_pipeline_chain(
                    graph,
                    merge_steps,
                    wrapper=merge_wrapper,
                    key_prefix=f"merge:{segment_index}",
                    attr_last_index=attr_last_index,
                )
                merge_first = merge_indices[0] if merge_indices else None
                if merge_first is not None:
                    for tail in branch_tails:
                        graph.links.append((tail, merge_first))
                last_index = merge_tail
                if merge_tail is not None:
                    _track_attr_index(attr_last_index, merge_wrapper.attr_name, merge_tail)
            else:
                merge_index = _add_node(
                    graph,
                    key=f"merge:{segment_index}",
                    block=segment.merge,
                )
                for tail in branch_tails:
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
                        _link_forward_input(graph, input_index, combine_index)
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
            if is_situ_gated_mlp(module):
                _branch_indices, module_tail = _add_situ_gated_mlp_chain(
                    graph,
                    module,
                    key_prefix=f"residual_branch:{segment_index}:{module.attr_name}",
                    attr_last_index=attr_last_index,
                    input_index=input_index,
                    last_index=None,
                    branch_from_input_dashed=True,
                    create_outer_frame=True,
                )
                _track_attr_index(attr_last_index, module.attr_name, module_tail)
            else:
                expanded_steps, wrapper = inline_composite_steps(module, basic_ops=basic_ops)
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
                        _link_forward_input(graph, input_index, module_index)
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

                combine_sublabel = gated_norm_combine_sublabel(consumer, gate_producer)
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
                            _link_forward_input(graph, input_index, combine_index)
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
                            basic_ops=basic_ops,
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
                        _link_forward_input(graph, input_index, consumer_index)
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
                        basic_ops=basic_ops,
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
                    basic_ops=basic_ops,
                )
                if first_after is not None:
                    graph.links.append((last_index, first_after))
                last_index = tail
                continue

            side = segment.side
            expanded_side, side_wrapper = inline_composite_steps(side, basic_ops=basic_ops)
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
                    _link_forward_input(graph, input_index, side_index)
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
                graph.side_entry_links.add((side_index, mult_index))
            first_after, tail = _add_chain(
                graph,
                after_nodes,
                key_prefix=f"post:{segment_index}",
                attr_last_index=attr_last_index,
                basic_ops=basic_ops,
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
            expanded_steps, wrapper = inline_composite_steps(step, basic_ops=basic_ops)
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
    return True


def _spine_predecessor(
    graph: ComputationGraph,
    incoming: list[list[int]],
    index: int,
) -> int | None:
    """Return the forward-path predecessor, ignoring dashed side feeds."""
    predecessors = incoming[index]
    if not predecessors:
        return None
    forward = [
        pred
        for pred in predecessors
        if (pred, index) not in graph.dashed_links and (pred, index) not in graph.side_entry_links
    ]
    if len(forward) == 1:
        return forward[0]
    if len(predecessors) == 1:
        return predecessors[0]
    return None


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
        if _graph_has_tensor_ports(graph):
            non_tensor_incoming = [
                source
                for source in incoming[index]
                if graph.nodes[source].synthetic != SYNTHETIC_TENSOR
            ]
            if len(non_tensor_incoming) >= 2:
                return None
        predecessor = _spine_predecessor(graph, incoming, index)
        if predecessor is None:
            return None
        if not _is_layout_chain_node(positions[predecessor].spec):
            return None
        forward_successors = [
            target
            for target in outgoing[predecessor]
            if (predecessor, target) not in graph.dashed_links
            and (predecessor, target) not in graph.side_entry_links
        ]
        if len(forward_successors) != 1 or forward_successors[0] != index:
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
        if _frame_has_fork_join_branching(graph, frame):
            continue
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
    graph: ComputationGraph | None = None,
) -> None:
    """Place one layer left-to-right using a fixed index order."""
    if not layer_indices:
        return
    if graph is not None and graph.inline_frames:
        units = _layer_packing_units(graph, layer_indices)
        unit_positions = [
            (
                unit,
                sum(positions[index].cx for index in unit) / len(unit),
                _packing_unit_width(graph, unit, positions, min_gap=min_gap),
            )
            for unit in units
        ]
        unit_positions.sort(key=lambda item: item[1])
        if len(unit_positions) == 1:
            unit, _center, width = unit_positions[0]
            target_cx = anchor_x + width / 2 if align_left else anchor_x
            shift = target_cx - unit_positions[0][1]
            for index in unit:
                positions[index].cx += shift
            return
        total_w = sum(width for _unit, _center, width in unit_positions) + min_gap * (
            len(unit_positions) - 1
        )
        cursor = anchor_x if align_left else anchor_x - total_w / 2
        for unit, _center, width in unit_positions:
            target_cx = cursor + width / 2
            shift = target_cx - (sum(positions[index].cx for index in unit) / len(unit))
            for index in unit:
                positions[index].cx += shift
            cursor += width + min_gap
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


def _inline_frame_id_for_node(graph: ComputationGraph, node_index: int) -> str | None:
    for frame in graph.inline_frames:
        if node_index in frame.node_indices:
            return frame.frame_id
    return None


def _layer_packing_units(graph: ComputationGraph, layer_indices: list[int]) -> list[list[int]]:
    """Group same-layer nodes that belong to one inline frame into a single packing unit."""
    units: list[list[int]] = []
    seen: set[int] = set()
    for index in layer_indices:
        if index in seen:
            continue
        frame_id = _inline_frame_id_for_node(graph, index)
        if frame_id is None:
            units.append([index])
            seen.add(index)
            continue
        unit = [member for member in layer_indices if _inline_frame_id_for_node(graph, member) == frame_id]
        units.append(unit)
        seen.update(unit)
    return units


def _estimate_inline_frame_column_width(
    graph: ComputationGraph,
    frame_id: str,
    *,
    min_gap: float,
) -> float:
    from visualizer.render import INLINE_FRAME_PAD, _inline_frame_total_connector_gutter_width

    frame = next(frame for frame in graph.inline_frames if frame.frame_id == frame_id)
    widths = [_diagram_size_for_spec(graph.nodes[index])[0] for index in frame.node_indices]
    if not widths:
        return 0.0
    gutter = _inline_frame_total_connector_gutter_width(graph, frame)
    chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
    if len(chain) >= 2:
        return max(widths) + 2 * INLINE_FRAME_PAD + gutter
    return sum(widths) + min_gap * max(0, len(widths) - 1) + 2 * INLINE_FRAME_PAD + gutter


def _inline_frame_for_indices(graph: ComputationGraph, indices: list[int]):
    index_set = set(indices)
    for frame in graph.inline_frames:
        if index_set.issubset(set(frame.node_indices)):
            return frame
    return None


def _inline_frame_column_bounds(
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    indices: list[int],
    *,
    pad: float,
) -> tuple[float, float]:
    from visualizer.render import _inline_frame_connector_gutter_width

    frame = _inline_frame_for_indices(graph, indices)
    left = min(_node_content_left(positions[index]) for index in indices)
    right = max(_node_content_right(positions[index]) for index in indices)
    if frame is not None:
        from visualizer.render import _inline_frame_total_connector_gutter_width

        gutter = _inline_frame_total_connector_gutter_width(graph, frame)
    else:
        gutter = 0.0
    return left - pad - gutter, right + pad


def _inter_inline_frame_gap(
    graph: ComputationGraph,
    left_indices: list[int],
    right_indices: list[int],
    *,
    base_gap: float,
) -> float:
    from visualizer.render import INLINE_FRAME_SIDE_ENTRY_EXTRA_GAP, _inline_frame_side_entry_link_count

    extra = 0.0
    for indices in (left_indices, right_indices):
        frame = _inline_frame_for_indices(graph, indices)
        if frame is not None and _inline_frame_side_entry_link_count(graph, frame) > 0:
            extra = max(extra, INLINE_FRAME_SIDE_ENTRY_EXTRA_GAP)
    return base_gap + extra


def _packing_unit_width(
    graph: ComputationGraph,
    unit: list[int],
    positions: list[LayoutPosition],
    *,
    min_gap: float,
) -> float:
    from visualizer.render import INLINE_FRAME_PAD

    if len(unit) == 1:
        frame_id = _inline_frame_id_for_node(graph, unit[0])
        if frame_id is not None:
            frame = next(frame for frame in graph.inline_frames if frame.frame_id == frame_id)
            if len(frame.node_indices) >= 2:
                return _estimate_inline_frame_column_width(graph, frame_id, min_gap=min_gap)
        return positions[unit[0]].width

    left = min(_node_content_left(positions[index]) for index in unit)
    right = max(_node_content_right(positions[index]) for index in unit)
    frame = _inline_frame_for_indices(graph, unit)
    gutter = 0.0
    if frame is not None:
        from visualizer.render import _inline_frame_total_connector_gutter_width

        gutter = _inline_frame_total_connector_gutter_width(graph, frame)
    return right - left + 2 * INLINE_FRAME_PAD + gutter


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
        spine_successors = [
            target
            for target in outgoing.get(cursor, [])
            if (cursor, target) not in graph.side_entry_links
        ]
        cursor = spine_successors[0] if len(spine_successors) == 1 else None

    if len(ordered) == len(frame_indices):
        return ordered
    return sorted(frame_indices, key=lambda index: index)


def _shared_fork_predecessors(
    incoming: dict[int, list[int]],
    left: int,
    right: int,
) -> set[int]:
    return set(incoming[left]) & set(incoming[right])


def _is_multiply_combine(graph: ComputationGraph, join: int) -> bool:
    label = graph.nodes[join].label.strip()
    return label in {"×", "x", "*", "⨉"}


def _find_fork_join_clusters(graph: ComputationGraph) -> list[ForkJoinCluster]:
    """Return fork/join clusters: parallel branches meeting at ×, then continuing downstream."""
    clusters: list[ForkJoinCluster] = []
    seen: set[tuple[int, int, int, int, int]] = set()

    incoming: dict[int, list[int]] = {index: [] for index in range(len(graph.nodes))}
    outgoing: dict[int, list[int]] = {index: [] for index in range(len(graph.nodes))}
    link_set = set(graph.links)
    for source, target in graph.links:
        incoming[target].append(source)
        outgoing[source].append(target)

    for side_source, join in graph.side_entry_links:
        join_spec = graph.nodes[join]
        if not _is_combine_synthetic(join_spec.synthetic):
            continue
        if not _is_multiply_combine(graph, join):
            continue

        main_branch_candidates = [
            source
            for source in incoming[join]
            if (source, join) not in graph.side_entry_links
        ]
        if len(main_branch_candidates) != 1:
            continue
        main_branch = main_branch_candidates[0]

        main_source_candidates = [
            source
            for source in incoming[main_branch]
            if (source, main_branch) not in graph.dashed_links
            and (source, main_branch) not in graph.side_entry_links
        ]
        if len(main_source_candidates) != 1:
            continue
        main_source = main_source_candidates[0]

        if (main_source, main_branch) not in link_set or (main_branch, join) not in link_set:
            continue
        if not _shared_fork_predecessors(incoming, main_source, side_source):
            continue

        tail_candidates = [
            target
            for target in outgoing[join]
            if (join, target) not in graph.dashed_links
        ]
        if len(tail_candidates) != 1:
            continue
        tail = tail_candidates[0]

        cluster_key = (main_source, side_source, main_branch, join, tail)
        if cluster_key in seen:
            continue
        seen.add(cluster_key)
        clusters.append(
            ForkJoinCluster(
                main_source=main_source,
                side_source=side_source,
                main_branch=main_branch,
                join=join,
                tail=tail,
            )
        )

    return clusters


def _inner_act_frame_indices(
    graph: ComputationGraph,
    cluster: ForkJoinCluster,
) -> list[int]:
    for frame in graph.inline_frames:
        frame_set = set(frame.node_indices)
        if (
            cluster.join in frame_set
            and cluster.main_branch in frame_set
            and cluster.side_source not in frame_set
        ):
            return list(frame.node_indices)
    return []


def _layout_fork_join_branch(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    cluster: ForkJoinCluster,
) -> None:
    """Lay out a fork/join cluster: gate above inner frame, up beside join, down below."""
    from visualizer.render import (
        INLINE_FRAME_LABEL_GAP,
        INLINE_FRAME_LABEL_LINE_H,
        INLINE_FRAME_PAD,
    )

    v_gap = min_vertical_block_gap()
    h_gap = min_horizontal_block_gap()
    main_cx = positions[cluster.main_source].cx
    inner_frame_indices = _inner_act_frame_indices(graph, cluster)
    caption_band = (
        INLINE_FRAME_PAD + INLINE_FRAME_LABEL_GAP + INLINE_FRAME_LABEL_LINE_H
        if inner_frame_indices
        else 0.0
    )

    join_pos = positions[cluster.join]
    main_branch_pos = positions[cluster.main_branch]
    main_source_pos = positions[cluster.main_source]
    tail_pos = positions[cluster.tail]

    join_pos.cx = main_cx
    main_branch_pos.cx = main_cx
    main_branch_pos.top_y = join_pos.top_y + v_gap + main_branch_pos.height

    inner_block_top = main_branch_pos.top_y + caption_band
    main_source_pos.cx = main_cx
    main_source_pos.top_y = inner_block_top + v_gap + main_source_pos.height

    tail_pos.cx = main_cx
    plus_index = next(
        (
            target
            for source, target in graph.links
            if source == cluster.tail and graph.nodes[target].label == "+"
        ),
        None,
    )
    if plus_index is not None:
        merge_siblings = [
            source
            for source, target in graph.links
            if target == plus_index and source != cluster.tail
        ]
        if len(merge_siblings) == 1:
            tail_pos.top_y = positions[merge_siblings[0]].top_y
        elif inner_frame_indices:
            inner_bottom = min(positions[index].bottom for index in inner_frame_indices)
            tail_pos.top_y = inner_bottom - INLINE_FRAME_PAD - v_gap
        else:
            tail_pos.top_y = join_pos.bottom - v_gap
    elif inner_frame_indices:
        inner_bottom = min(positions[index].bottom for index in inner_frame_indices)
        tail_pos.top_y = inner_bottom - INLINE_FRAME_PAD - v_gap
    else:
        tail_pos.top_y = join_pos.bottom - v_gap

    side_pos = positions[cluster.side_source]
    align_pos = main_branch_pos
    side_pos.top_y = align_pos.top_y + (align_pos.height - side_pos.height) / 2
    side_pos.cx = _node_content_right(join_pos) + h_gap + side_pos.width / 2

    if inner_frame_indices:
        frame_right = (
            max(_node_content_right(positions[index]) for index in inner_frame_indices)
            + INLINE_FRAME_PAD
        )
        min_side_cx = frame_right + h_gap + side_pos.width / 2
        if side_pos.cx < min_side_cx:
            side_pos.cx = min_side_cx


def _ensure_synthetic_input_clears_consumers(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float | None = None,
) -> None:
    """Keep the synthetic input at least one detail layer gap above every direct consumer."""
    gap = DETAIL_LAYER_GAP if min_gap is None else max(min_gap, DETAIL_LAYER_GAP)
    input_index = next(
        (index for index, pos in enumerate(positions) if pos.spec.synthetic == SYNTHETIC_INPUT),
        None,
    )
    if input_index is None:
        return
    targets = [target for source, target in graph.links if source == input_index]
    if not targets:
        return
    input_pos = positions[input_index]
    required_bottom = max(positions[target].top_y for target in targets) + gap
    if input_pos.bottom < required_bottom:
        input_pos.top_y += required_bottom - input_pos.bottom


def _compact_synthetic_input_spacing(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float | None = None,
    elements: list[MeasuredElement] | None = None,
) -> None:
    """Place the synthetic input one standard gap above its downstream nodes and frame captions."""
    gap = DETAIL_LAYER_GAP if min_gap is None else max(min_gap, DETAIL_LAYER_GAP)
    input_index = next(
        (index for index, pos in enumerate(positions) if pos.spec.synthetic == SYNTHETIC_INPUT),
        None,
    )
    if input_index is None:
        return
    targets = [target for source, target in graph.links if source == input_index]
    if not targets:
        return
    highest_top = max(positions[target].top_y for target in targets)
    input_pos = positions[input_index]
    desired_bottom = highest_top + gap
    if elements:
        input_left = input_pos.cx - input_pos.width / 2
        input_right = input_pos.cx + input_pos.width / 2
        caption_tops = [
            element.bounds.top
            for element in elements
            if element.kind in {"inline_frame", "frame_label", "frame_sublabel"}
            and element.bounds.right + gap > input_left
            and element.bounds.left - gap < input_right
        ]
        if caption_tops:
            desired_bottom = max(desired_bottom, max(caption_tops) + gap)
    shift = desired_bottom - input_pos.bottom
    if shift > 0:
        input_pos.top_y += shift
    elif shift < -1e-6:
        # Input sits too far above its downstream targets; lift targets toward it.
        for index, pos in enumerate(positions):
            if index == input_index:
                continue
            if pos.top_y > input_pos.top_y + 1e-6:
                continue
            pos.top_y -= shift


def _ensure_input_above_fork_join_clusters(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float,
) -> None:
    """Keep the synthetic input above fork/join main sources after branch layout."""
    input_pos = next((pos for pos in positions if pos.spec.synthetic == SYNTHETIC_INPUT), None)
    if input_pos is None:
        return
    for cluster in _find_fork_join_clusters(graph):
        main_source = positions[cluster.main_source]
        if (
            input_pos.cx + input_pos.width / 2 + min_gap <= main_source.cx - main_source.width / 2
            or main_source.cx + main_source.width / 2 + min_gap <= input_pos.cx - input_pos.width / 2
        ):
            continue
        min_input_bottom = main_source.top_y + min_gap
        if input_pos.bottom < min_input_bottom:
            input_pos.top_y += min_input_bottom - input_pos.bottom


def _router_spine_column_indices(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> set[int]:
    """Node indices in the MoE router spine column (MoE aggregation and downstream main path)."""
    from visualizer.ast_analyze import MOE_AGGREGATION_LABEL

    sigma = next(
        (
            index
            for index, spec in enumerate(graph.nodes)
            if spec.label == MOE_AGGREGATION_LABEL and _is_combine_synthetic(spec.synthetic)
        ),
        None,
    )
    if sigma is None:
        return set()

    fork_join = set()
    for cluster in _find_fork_join_clusters(graph):
        fork_join |= {
            cluster.main_source,
            cluster.side_source,
            cluster.main_branch,
            cluster.join,
            cluster.tail,
        }

    outgoing: dict[int, list[int]] = {index: [] for index in range(len(positions))}
    for source, target in graph.links:
        outgoing[source].append(target)

    spine: set[int] = set()
    queue = [sigma]
    while queue:
        index = queue.pop(0)
        if index in spine or index in fork_join:
            continue
        spine.add(index)
        for target in outgoing[index]:
            if target not in fork_join:
                queue.append(target)

    plus = next(
        (
            index
            for index, spec in enumerate(graph.nodes)
            if spec.label == "+" and _is_combine_synthetic(spec.synthetic)
        ),
        None,
    )
    if plus is not None:
        spine.add(plus)
    return spine


def _align_router_spine_column(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Keep the MoE router spine (MoE aggregation and its non-fork/join chain) in one column."""
    from visualizer.ast_analyze import MOE_AGGREGATION_LABEL

    sigma = next(
        (
            index
            for index, spec in enumerate(graph.nodes)
            if spec.label == MOE_AGGREGATION_LABEL and _is_combine_synthetic(spec.synthetic)
        ),
        None,
    )
    if sigma is None:
        return

    spine_cx = positions[sigma].cx
    for index in _router_spine_column_indices(positions, graph):
        positions[index].cx = spine_cx


def _layout_fork_join_branches(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    for cluster in _find_fork_join_clusters(graph):
        _layout_fork_join_branch(positions, graph, cluster)
    _clear_side_branches_from_gate_frame(positions, graph)
    _align_router_spine_column(positions, graph)
    plus_index = next(
        (
            index
            for index, spec in enumerate(graph.nodes)
            if spec.label == "+" and _is_combine_synthetic(spec.synthetic)
        ),
        None,
    )
    if plus_index is not None:
        _align_merge_nodes(positions, graph, only_targets={plus_index})


def _clear_side_branches_from_gate_frame(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Keep MoE router side paths and fork/join columns outside the gate inline frame."""
    gate_frame = next((frame for frame in graph.inline_frames if frame.frame_id == "gate"), None)
    from visualizer.render import INLINE_FRAME_PAD

    min_left = 0.0
    if gate_frame is not None:
        gate_right = max(_node_content_right(positions[index]) for index in gate_frame.node_indices)
        min_left = gate_right + INLINE_FRAME_PAD + min_horizontal_block_gap()

    fork_join_joins = {cluster.join for cluster in _find_fork_join_clusters(graph)}
    spine_indices = _router_spine_column_indices(positions, graph)
    if spine_indices:
        min_left = max(
            min_left,
            max(_node_content_right(positions[index]) for index in spine_indices)
            + min_horizontal_block_gap()
            + INLINE_FRAME_PAD,
        )
    for index, spec in enumerate(graph.nodes):
        if not _is_combine_synthetic(spec.synthetic) or index in fork_join_joins:
            continue
        if index in spine_indices:
            continue
        combine_right = _node_content_right(positions[index])
        min_left = max(min_left, combine_right + min_horizontal_block_gap() + INLINE_FRAME_PAD)

    for cluster in _find_fork_join_clusters(graph):
        cluster_indices = {
            cluster.main_source,
            cluster.side_source,
            cluster.main_branch,
            cluster.join,
            cluster.tail,
        }
        cluster_min_left = min_left
        side_left = _node_content_left(positions[cluster.side_source])
        for frame in graph.inline_frames:
            members = set(frame.node_indices)
            if cluster.side_source in members:
                continue
            if not members.intersection(
                {cluster.main_source, cluster.main_branch, cluster.join, cluster.tail}
            ):
                continue
            frame_right = max(_node_content_right(positions[index]) for index in frame.node_indices)
            required_left = frame_right + INLINE_FRAME_PAD + min_horizontal_block_gap()
            if frame.frame_id == "shared_experts" or side_left < required_left:
                cluster_min_left = max(cluster_min_left, required_left)
        cluster_left = min(_node_content_left(positions[index]) for index in cluster_indices)
        if cluster_left < cluster_min_left:
            delta = cluster_min_left - cluster_left
            for index in cluster_indices:
                positions[index].cx += delta

    if gate_frame is None:
        return
    gate_right = max(_node_content_right(positions[index]) for index in gate_frame.node_indices)
    min_left = gate_right + INLINE_FRAME_PAD + min_horizontal_block_gap()
    for index, spec in enumerate(graph.nodes):
        if "routed_expert_down" not in spec.key:
            continue
        left = _node_content_left(positions[index])
        if left < min_left:
            positions[index].cx += min_left - left


def _frame_has_fork_join_branching(
    graph: ComputationGraph,
    frame: InlineFrameSpec,
) -> bool:
    """True when an inline frame wraps a fork/join spine (gate, branch, join, tail)."""
    frame_members = set(frame.node_indices)
    for cluster in _find_fork_join_clusters(graph):
        spine = {
            cluster.main_source,
            cluster.main_branch,
            cluster.join,
            cluster.tail,
        }
        if spine.issubset(frame_members):
            return True
    return False


def _align_inline_frame_column_cx(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Keep every straight inline-frame chain on a single vertical column."""
    for frame in graph.inline_frames:
        if _frame_has_fork_join_branching(graph, frame):
            continue
        indices = _ordered_inline_frame_chain(graph, list(frame.node_indices))
        if not indices:
            continue
        frame_cx = sum(positions[index].cx for index in indices) / len(indices)
        for index in indices:
            positions[index].cx = frame_cx


def _inline_frame_has_external_port_feed(graph, frame) -> bool:
    """True when a tensor port outside the frame feeds a member tile."""
    from visualizer.computation_graph import SYNTHETIC_TENSOR

    members = set(frame.node_indices)
    return any(
        tgt in members
        and src not in members
        and graph.nodes[src].synthetic == SYNTHETIC_TENSOR
        for src, tgt in graph.links
    )


def _inline_frame_vertical_gap(graph, frame) -> float:
    """Vertical spacing between tiles in one inline frame."""
    from visualizer.render import (
        INLINE_FRAME_MULTI_BYPASS_EXTRA_GAP,
        INLINE_FRAME_SIDE_ENTRY_EXTRA_GAP,
        _inline_frame_side_entry_link_count,
    )
    from visualizer.sizing import min_vertical_block_gap

    link_count = _inline_frame_side_entry_link_count(graph, frame)
    gap = min_vertical_block_gap()
    if link_count >= 2:
        return gap + INLINE_FRAME_MULTI_BYPASS_EXTRA_GAP * link_count
    if link_count >= 1 or _inline_frame_has_external_port_feed(graph, frame):
        return gap + INLINE_FRAME_SIDE_ENTRY_EXTRA_GAP
    return gap


def stack_inline_frame_positions(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float | None = None,
) -> None:
    """Re-stack each inline frame column using measured tile heights."""
    _layout_fork_join_branches(positions, graph)
    for frame in graph.inline_frames:
        if _frame_has_fork_join_branching(graph, frame):
            continue
        indices = _ordered_inline_frame_chain(graph, list(frame.node_indices))
        if len(indices) < 2:
            if indices:
                positions[indices[0]].cx = sum(positions[index].cx for index in indices) / len(indices)
            continue

        frame_cx = sum(positions[index].cx for index in indices) / len(indices)
        frame_gap = _inline_frame_vertical_gap(graph, frame) if min_gap is None else min_gap
        cursor_top = max(positions[index].top_y for index in indices)
        for index in indices:
            pos = positions[index]
            pos.cx = frame_cx
            pos.top_y = cursor_top
            cursor_top -= pos.height + frame_gap


def finalize_tensor_port_pipeline_layout(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_left: float | None = None,
) -> None:
    """Pack kernel-pipeline inline frames and dock modeling inputs above their consumers."""
    if not _graph_has_tensor_ports(graph):
        return

    stack_inline_frame_positions(positions, graph)
    _align_tensor_port_merge_nodes(positions, graph)
    _align_tensor_port_columns(positions, graph)
    _align_inline_frame_column_cx(positions, graph)
    repack_inline_frame_columns(positions, graph)
    stack_inline_frame_positions(positions, graph)
    _align_tensor_port_pipeline_merge_clearance(positions, graph)

    if min_left is None or not positions:
        return
    content_left = min(_node_content_left(pos) for pos in positions)
    shift = min_left - content_left
    if abs(shift) <= 1e-6:
        return
    for pos in positions:
        pos.cx += shift
    stack_inline_frame_positions(positions, graph)


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
    only_targets: set[int] | None = None,
) -> None:
    """Place merge/combine nodes one layer gap below the deepest incoming branch."""
    gap = DETAIL_LAYER_GAP if merge_gap is None else merge_gap
    incoming = _build_incoming_links(graph, node_count=len(positions))

    for target, sources in incoming.items():
        if only_targets is not None and target not in only_targets:
            continue
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
    skip_frame_repack: bool = False,
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

    if frame_columns and not skip_frame_repack:
        frame_columns.sort(key=lambda indices: _inline_frame_column_bounds(graph, positions, indices, pad=frame_pad)[0])
        cursor_left: float | None = None
        prev_indices: list[int] | None = None
        for indices in frame_columns:
            left, right = _inline_frame_column_bounds(graph, positions, indices, pad=frame_pad)
            width = right - left
            if cursor_left is None:
                cursor_left = left
            else:
                column_gap = _inter_inline_frame_gap(
                    graph,
                    prev_indices or [],
                    indices,
                    base_gap=gap,
                )
                cursor_left += column_gap
            shift = cursor_left - left
            for index in indices:
                positions[index].cx += shift
            cursor_left += width
            prev_indices = indices

    free_indices = [
        index
        for index, pos in enumerate(positions)
        if index not in frame_members
        and pos.spec.synthetic not in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}
    ]
    incoming = _build_incoming_links(graph, node_count=len(positions))
    outgoing: dict[int, list[int]] = {index: [] for index in range(len(positions))}
    for source, target in _layout_graph_links(graph):
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

    if _graph_has_tensor_ports(graph):
        _align_inline_frame_column_cx(positions, graph)

    gap = min_horizontal_block_gap()
    pad = INLINE_FRAME_PAD

    frame_columns: list[list[int]] = []
    for frame in graph.inline_frames:
        indices = _ordered_inline_frame_chain(graph, list(frame.node_indices))
        if indices:
            frame_columns.append(indices)

    if not frame_columns:
        return

    if _graph_has_tensor_ports(graph):
        def frame_vertical_band(indices: list[int]) -> float:
            return max(positions[index].top_y for index in indices)

        bands: dict[float, list[list[int]]] = {}
        for indices in frame_columns:
            bands.setdefault(frame_vertical_band(indices), []).append(indices)
        grouped_columns = list(bands.values())
    else:
        grouped_columns = [frame_columns]

    for columns in grouped_columns:
        columns.sort(
            key=lambda indices: _inline_frame_column_bounds(graph, positions, indices, pad=pad)[0]
        )
        cursor_left: float | None = None
        prev_indices: list[int] | None = None
        for indices in columns:
            left, right = _inline_frame_column_bounds(graph, positions, indices, pad=pad)
            width = right - left
            if cursor_left is None:
                cursor_left = left
            else:
                cursor_left += _inter_inline_frame_gap(
                    graph,
                    prev_indices or [],
                    indices,
                    base_gap=gap,
                )
            shift = cursor_left - left
            for index in indices:
                positions[index].cx += shift
            cursor_left += width
            prev_indices = indices


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
    max_gap = DETAIL_LAYER_GAP
    for frame in graph.inline_frames:
        members = set(frame.node_indices)
        if not (upper & members) or not (lower & members):
            continue
        chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
        for source, target in zip(chain, chain[1:]):
            if source in upper and target in lower:
                max_gap = max(max_gap, _inline_frame_vertical_gap(graph, frame))
    return max_gap


def measure_graph_node_sizes(
    ax,
    graph: ComputationGraph,
    *,
    input_sublabel: str | None = None,
    title_fontsize: float = 7.6,
) -> None:
    """Measure every tile label at draw time and cache diagram-unit sizes before layout."""
    from visualizer.text_measure import box_label_size, input_box_label_size, tensor_port_box_label_size

    inline_members = inline_frame_member_indices(graph)
    for index, spec in enumerate(graph.nodes):
        if _is_combine_synthetic(spec.synthetic):
            from visualizer.ast_analyze import is_compact_combine_label
            from visualizer.text_measure import box_label_size

            if is_compact_combine_label(spec.label):
                spec.diagram_width, spec.diagram_height = COMBINE_OP_SIZE, COMBINE_OP_SIZE
            else:
                width, height = box_label_size(
                    ax,
                    spec.label,
                    spec.sublabel,
                    fontsize=6.8,
                    pad_x=0.08,
                    pad_y=0.06,
                    white_text_stroke_pad=False,
                )
                spec.diagram_width, spec.diagram_height = width, height
            continue
        if spec.synthetic == SYNTHETIC_INPUT:
            width, height = input_box_label_size(ax, spec.label, input_sublabel, fontsize=7.2)
            spec.diagram_width, spec.diagram_height = width, height
            continue
        if spec.synthetic == SYNTHETIC_HIDDEN:
            width, height = input_box_label_size(ax, spec.label, None, fontsize=6.5)
            spec.diagram_width, spec.diagram_height = width, height
            continue
        if spec.synthetic == SYNTHETIC_TENSOR:
            width, height = tensor_port_box_label_size(ax, spec.label, spec.sublabel, fontsize=7.0)
            spec.diagram_width, spec.diagram_height = width, height
            continue
        label, sublabel = _rendered_label_and_sublabel(
            spec,
            inline_frame_members=inline_members,
            node_index=index,
        )
        width, height = box_label_size(
            ax,
            label,
            sublabel,
            fontsize=title_fontsize,
            white_text_stroke_pad=not is_basic_op_tile(spec.block),
        )
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
        from visualizer.ast_analyze import is_compact_combine_label

        if is_compact_combine_label(spec.label):
            return COMBINE_OP_SIZE, COMBINE_OP_SIZE
        if spec.diagram_width is not None and spec.diagram_height is not None:
            return spec.diagram_width, spec.diagram_height
        return LABELED_COMBINE_DEFAULT_WIDTH, LABELED_COMBINE_DEFAULT_HEIGHT
    return _diagram_size_for_rendered_spec(spec)


def _block_layout_width(block: BlockNode | None, *, min_gap: float) -> float:
    """Estimate horizontal space for a block, including expanded kernel sub-op chains."""
    from visualizer.sizing import estimate_block_size

    if block is not None and len(block.children) >= 2:
        widths = [estimate_block_size(child.label, None)[0] for child in block.children]
        frame_pad = 0.24
        return max(widths) + frame_pad
    if block is not None:
        return estimate_block_size(block.label, None)[0]
    return 0.0


def _minimum_graph_layout_width(
    graph: ComputationGraph,
    *,
    min_gap: float | None = None,
) -> float:
    """Minimum horizontal band needed to lay out the widest topological layer."""
    if not graph.nodes:
        return 0.0
    gap = MIN_HORIZONTAL_BLOCK_GAP if min_gap is None else min_gap
    layers = _topological_layers(graph)
    widest = 0.0
    for layer_indices in layers:
        if not layer_indices:
            continue
        widths = [
            _block_layout_width(graph.nodes[index].block, min_gap=gap)
            if graph.nodes[index].block is not None
            else _diagram_size_for_spec(graph.nodes[index])[0]
            for index in layer_indices
        ]
        if graph.inline_frames:
            units = _layer_packing_units(graph, layer_indices)
            grouped_widths: list[float] = []
            index_to_width = dict(zip(layer_indices, widths))
            for unit in units:
                frame_id = _inline_frame_id_for_node(graph, unit[0]) if len(unit) == 1 else None
                if frame_id is not None and len(unit) == 1:
                    grouped_widths.append(
                        _estimate_inline_frame_column_width(graph, frame_id, min_gap=gap)
                    )
                else:
                    grouped_widths.append(
                        sum(index_to_width[index] for index in unit)
                        + gap * max(0, len(unit) - 1)
                    )
            row_width = sum(grouped_widths) + gap * max(0, len(grouped_widths) - 1)
        else:
            row_width = sum(widths) + gap * max(0, len(widths) - 1)
        widest = max(widest, row_width)
    return widest


def _split_independent_kernel_op_layers(
    layers: list[list[int]],
    graph: ComputationGraph,
) -> list[list[int]]:
    """Stack parallel kernel ops vertically when they share a layer but have no step-to-step edges."""
    if _graph_has_tensor_ports(graph):
        # Tensor ports sit in one row above their targets; keep parallel kernel ops horizontal.
        return layers
    split: list[list[int]] = []
    for layer in layers:
        kernel_ops = [
            index
            for index in layer
            if graph.nodes[index].block is not None
            and graph.nodes[index].block.class_name == "KernelOp"
        ]
        if len(kernel_ops) <= 1:
            split.append(layer)
            continue

        kernel_set = set(kernel_ops)
        has_internal_kernel_edge = any(
            src in kernel_set and tgt in kernel_set
            for src, tgt in _layout_graph_links(graph)
        )
        if has_internal_kernel_edge:
            split.append(layer)
            continue

        non_kernel = [index for index in layer if index not in kernel_set]
        if non_kernel:
            split.append(non_kernel)
        split.extend([[index] for index in kernel_ops])
    return split


def _layout_graph_links(graph: ComputationGraph) -> list[tuple[int, int]]:
    """Links that influence Sugiyama/layer placement (omit display-only operand feeds)."""
    return [
        link
        for link in graph.links
        if link not in graph.inline_binary_operand_links
    ]


def _build_incoming_links(
    graph: ComputationGraph,
    *,
    node_count: int | None = None,
) -> dict[int, list[int]]:
    """Map target index -> layout-relevant source indices."""
    count = len(graph.nodes) if node_count is None else node_count
    incoming: dict[int, list[int]] = {index: [] for index in range(count)}
    for source, target in _layout_graph_links(graph):
        if source < count and target < count:
            incoming[target].append(source)
    return incoming


def _topological_layers(graph: ComputationGraph) -> list[list[int]]:
    """Group node indices into layers for tight vertical stacking."""
    node_count = len(graph.nodes)
    if node_count == 0:
        return []

    incoming = [0] * node_count
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for src, tgt in _layout_graph_links(graph):
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
    layers = [layers_map[layer_id] for layer_id in sorted(layers_map)]
    return _split_parallel_inline_frame_layers(
        _split_independent_kernel_op_layers(layers, graph),
        graph,
    )


def _split_parallel_inline_frame_layers(
    layers: list[list[int]],
    graph: ComputationGraph,
) -> list[list[int]]:
    """Stack parallel inline-frame columns vertically in tensor-port pipeline graphs."""
    if not _graph_has_tensor_ports(graph) or not graph.inline_frames:
        return layers

    split: list[list[int]] = []
    for layer in layers:
        if len(layer) <= 1:
            split.append(layer)
            continue
        if all(graph.nodes[index].synthetic == SYNTHETIC_TENSOR for index in layer):
            split.append(layer)
            continue
        units = _layer_packing_units(graph, layer)
        if len(units) <= 1:
            split.append(layer)
            continue
        split.extend(units)
    return split


INPUT_INLINE_FRAME_CAPTION_CLEARANCE = 0.28
TENSOR_PORT_LAYER_EXTRA_GAP = 0.28


def _graph_has_tensor_ports(graph: ComputationGraph) -> bool:
    return any(spec.synthetic == SYNTHETIC_TENSOR for spec in graph.nodes)


def _inline_frame_tail_indices(graph: ComputationGraph) -> set[int]:
    """Tail node of each inline-frame vertical chain."""
    tails: set[int] = set()
    for frame in graph.inline_frames:
        chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
        if chain:
            tails.add(chain[-1])
    return tails


def _shift_inline_frame_column(
    positions: list[LayoutPosition],
    frame,
    delta_y: float,
) -> None:
    """Shift every tile in one inline frame by ``delta_y`` (negative moves up)."""
    for index in frame.node_indices:
        positions[index].top_y -= delta_y


def _inline_frame_for_tail_node(graph: ComputationGraph, src: int):
    for frame in graph.inline_frames:
        if frame.node_indices and frame.node_indices[-1] == src:
            return frame
    return None


def _align_tensor_port_pipeline_merge_clearance(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Make room below frame exits for shared merge buses."""
    from visualizer.render import (
        CONNECTOR_ATTACHED_BOX_MARGIN,
        CONNECTOR_EXIT_STUB,
        CONNECTOR_OBSTACLE_MARGIN,
        PIPELINE_MERGE_BUS_BELOW_FRAME_GAP,
        _frame_tail_exit_horiz_y,
    )

    if not _graph_has_tensor_ports(graph):
        return

    incoming = _build_incoming_links(graph, node_count=len(positions))
    frame_tails = _inline_frame_tail_indices(graph)

    for target, sources in incoming.items():
        tail_sources = [source for source in sources if source in frame_tails]
        if len(tail_sources) < 2:
            continue

        exit_horiz_y = float("inf")
        tightest_source: int | None = None
        for source in tail_sources:
            horiz_y = _frame_tail_exit_horiz_y(graph, positions, source)
            if horiz_y is not None and horiz_y < exit_horiz_y:
                exit_horiz_y = horiz_y
                tightest_source = source
        if exit_horiz_y == float("inf") or tightest_source is None:
            continue

        desired_bus_y = exit_horiz_y - CONNECTOR_EXIT_STUB - PIPELINE_MERGE_BUS_BELOW_FRAME_GAP
        min_bus_y = (
            positions[target].top_y
            + CONNECTOR_OBSTACLE_MARGIN
            + CONNECTOR_ATTACHED_BOX_MARGIN
        )
        if desired_bus_y + 1e-6 >= min_bus_y:
            continue

        shift_amount = min_bus_y - desired_bus_y + 0.04
        _shift_node_subtree(positions, graph, target, shift_amount)


def _shift_inline_frame_column_and_ports(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    frame,
    delta_y: float,
) -> None:
    """Shift one inline-frame column and tensor ports that feed any member tile."""
    if abs(delta_y) < 1e-9:
        return
    members = set(frame.node_indices)
    _shift_inline_frame_column(positions, frame, delta_y)
    for source, target in graph.links:
        if target in members and graph.nodes[source].synthetic == SYNTHETIC_TENSOR:
            positions[source].top_y -= delta_y


def _pipeline_merge_bus_y_for_layout(
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    target: int,
    tail_sources: list[int],
) -> float | None:
    """Shared merge-bus Y for inline-frame tails feeding one pipeline merge target."""
    from visualizer.render import (
        CONNECTOR_ATTACHED_BOX_MARGIN,
        CONNECTOR_EXIT_STUB,
        CONNECTOR_OBSTACLE_MARGIN,
        PIPELINE_MERGE_BUS_BELOW_FRAME_GAP,
        _frame_tail_exit_horiz_y,
    )

    exit_horiz_y = float("inf")
    for source in tail_sources:
        horiz_y = _frame_tail_exit_horiz_y(graph, positions, source)
        if horiz_y is not None:
            exit_horiz_y = min(exit_horiz_y, horiz_y)
    if exit_horiz_y == float("inf"):
        return None

    desired_bus_y = exit_horiz_y - CONNECTOR_EXIT_STUB - PIPELINE_MERGE_BUS_BELOW_FRAME_GAP
    min_bus_y = (
        positions[target].top_y
        + CONNECTOR_OBSTACLE_MARGIN
        + CONNECTOR_ATTACHED_BOX_MARGIN
    )
    return max(desired_bus_y, min_bus_y)


def _horizontal_bounds_overlap(
    left_a: float,
    right_a: float,
    left_b: float,
    right_b: float,
    *,
    gap: float,
) -> bool:
    return left_a <= right_b + gap and left_b <= right_a + gap


def _max_inline_frame_downward_shift(
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    frame,
    *,
    min_gap: float = 0.02,
) -> float:
    """Maximum downward shift before this frame column hits another tile or frame."""
    from visualizer.render import _inline_frame_draw_bounds

    bounds = _inline_frame_draw_bounds(frame, positions, graph)
    members = set(frame.node_indices)
    max_shift = bounds.bottom - min_gap

    for index, pos in enumerate(positions):
        if index in members or pos.spec.synthetic == SYNTHETIC_HIDDEN:
            continue
        left = _node_content_left(pos)
        right = _node_content_right(pos)
        if not _horizontal_bounds_overlap(bounds.left, bounds.right, left, right, gap=min_gap):
            continue
        if pos.top_y >= bounds.bottom - min_gap:
            continue
        allowed = bounds.bottom - (pos.top_y + min_gap)
        max_shift = min(max_shift, allowed)

    for other in graph.inline_frames:
        if other.frame_id == frame.frame_id:
            continue
        other_bounds = _inline_frame_draw_bounds(other, positions, graph)
        if not _horizontal_bounds_overlap(
            bounds.left,
            bounds.right,
            other_bounds.left,
            other_bounds.right,
            gap=min_gap,
        ):
            continue
        if other_bounds.top >= bounds.bottom - min_gap:
            continue
        allowed = bounds.bottom - (other_bounds.top + min_gap)
        max_shift = min(max_shift, allowed)

    return max(0.0, max_shift)


def _max_frame_exit_downward_shift(
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    frame,
    source: int,
    *,
    min_gap: float | None = None,
) -> float:
    """Maximum downward shift before frame-exit corridors or same-column tiles are hit."""
    from visualizer.render import (
        CONNECTOR_EXIT_STUB,
        CONNECTOR_OBSTACLE_MARGIN,
        _frame_exit_horizontal_y,
        _inline_frame_draw_bounds,
    )

    if min_gap is None:
        min_gap = CONNECTOR_OBSTACLE_MARGIN

    max_shift = _max_inline_frame_downward_shift(graph, positions, frame, min_gap=min_gap)
    bounds = _inline_frame_draw_bounds(frame, positions, graph)
    source_pos = positions[source]
    exit_y = _frame_exit_horizontal_y(bounds, source_bottom=source_pos.bottom)

    for index, pos in enumerate(positions):
        if index in frame.node_indices or pos.spec.synthetic == SYNTHETIC_HIDDEN:
            continue
        if abs(pos.cx - source_pos.cx) > 0.06:
            continue
        if pos.top_y >= source_pos.bottom - min_gap:
            continue
        allowed = exit_y - (pos.top_y + min_gap)
        max_shift = min(max_shift, allowed)

    for other in graph.inline_frames:
        if other.frame_id == frame.frame_id:
            continue
        other_bounds = _inline_frame_draw_bounds(other, positions, graph)
        if abs((other_bounds.left + other_bounds.right) / 2 - source_pos.cx) > 0.06:
            continue
        if other_bounds.top >= exit_y - min_gap:
            continue
        allowed = exit_y - (other_bounds.top + min_gap)
        max_shift = min(max_shift, allowed)

    return max(0.0, max_shift)


def _compact_parallel_feeder_frame_exit_stubs(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Shrinkwrap parallel feeder columns down to the shared merge-bus corridor."""
    from visualizer.render import (
        CONNECTOR_EXIT_STUB,
        PIPELINE_MERGE_BUS_BELOW_FRAME_GAP,
        _frame_tail_exit_horiz_y,
    )

    from visualizer.render import CONNECTOR_OBSTACLE_MARGIN

    incoming = _build_incoming_links(graph, node_count=len(positions))
    frame_tails = _inline_frame_tail_indices(graph)
    corridor_eps = CONNECTOR_OBSTACLE_MARGIN / 3
    min_gap = CONNECTOR_OBSTACLE_MARGIN

    for target, sources in incoming.items():
        tail_sources = [source for source in sources if source in frame_tails]
        if len(tail_sources) < 2:
            continue

        bus_y = _pipeline_merge_bus_y_for_layout(graph, positions, target, tail_sources)
        if bus_y is None:
            continue

        target_exit_y = bus_y + CONNECTOR_EXIT_STUB + PIPELINE_MERGE_BUS_BELOW_FRAME_GAP
        for source in tail_sources:
            exit_horiz_y = _frame_tail_exit_horiz_y(graph, positions, source)
            if exit_horiz_y is None:
                continue
            shift = exit_horiz_y - target_exit_y
            if shift <= corridor_eps:
                continue
            frame = _inline_frame_for_tail_node(graph, source)
            if frame is None:
                continue
            shift = min(
                shift,
                _max_frame_exit_downward_shift(
                    graph,
                    positions,
                    frame,
                    source,
                    min_gap=min_gap,
                ),
            )
            if shift <= corridor_eps:
                continue
            _shift_inline_frame_column_and_ports(positions, graph, frame, shift)


def _align_tensor_port_merge_nodes(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Place pipeline merge nodes on the main handoff column, not branch frame tails."""
    if not _graph_has_tensor_ports(graph):
        return

    incoming = _build_incoming_links(graph, node_count=len(positions))
    frame_tails = _inline_frame_tail_indices(graph)
    for target, sources in incoming.items():
        if len(sources) < 2:
            continue
        candidates = [
            source
            for source in sources
            if graph.nodes[source].synthetic != SYNTHETIC_TENSOR
        ]
        if not candidates:
            continue
        primary = [source for source in candidates if source not in frame_tails]
        anchor_source = min(
            primary or candidates,
            key=lambda index: (positions[index].cx, -positions[index].top_y),
        )
        positions[target].cx = positions[anchor_source].cx


def _tensor_port_outgoing(graph: ComputationGraph) -> dict[int, list[int]]:
    outgoing: dict[int, list[int]] = {index: [] for index in range(len(graph.nodes))}
    for source, target in graph.links:
        outgoing[source].append(target)
    return outgoing


def _dock_single_consumer_tensor_ports(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    gap: float | None = None,
) -> None:
    """Place modeling inputs with one consumer beside or above that consumer."""
    if not _graph_has_tensor_ports(graph):
        return

    row_gap = DETAIL_LAYER_GAP if gap is None else gap
    side_gap = MIN_HORIZONTAL_BLOCK_GAP
    outgoing = _tensor_port_outgoing(graph)
    for index, spec in enumerate(graph.nodes):
        if spec.synthetic != SYNTHETIC_TENSOR:
            continue
        targets = outgoing[index]
        if len(targets) != 1:
            continue
        target_index = targets[0]
        target_pos = positions[target_index]
        port_pos = positions[index]

        frame_tails = _inline_frame_tail_indices(graph)
        shared_column_tail = any(
            tgt == target_index
            for src, tgt in graph.links
            if src in frame_tails and abs(positions[src].cx - target_pos.cx) <= 0.06
        )
        if shared_column_tail:
            port_pos.cx = _node_content_left(target_pos) - side_gap - port_pos.width / 2
            port_pos.top_y = target_pos.top_y + row_gap + port_pos.height
            continue

        target_spec = graph.nodes[target_index]
        if target_spec.label == "×":
            port_pos.cx = _node_content_left(target_pos) - side_gap - port_pos.width / 2
            port_pos.top_y = target_pos.top_y + row_gap + port_pos.height
            continue

        frame_id = _inline_frame_id_for_node(graph, target_index)
        if frame_id is not None:
            frame = next(item for item in graph.inline_frames if item.frame_id == frame_id)
            chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
            if chain and chain[-1] == target_index and chain[0] != target_index:
                port_pos.top_y = target_pos.top_y
                port_pos.cx = _node_content_left(target_pos) - side_gap - port_pos.width / 2
                continue
            if chain:
                frame_top = max(positions[node_index].top_y for node_index in chain)
                port_pos.cx = target_pos.cx
                port_pos.top_y = (
                    frame_top + INPUT_INLINE_FRAME_CAPTION_CLEARANCE + row_gap + port_pos.height
                )
                continue

        port_pos.cx = target_pos.cx
        port_pos.top_y = target_pos.top_y + row_gap + port_pos.height


def _align_tensor_port_columns(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Place each modeling tensor port above the kernel step it feeds."""
    for source, target in graph.links:
        if graph.nodes[source].synthetic != SYNTHETIC_TENSOR:
            continue
        if graph.nodes[target].label == "×":
            continue
        positions[source].cx = positions[target].cx

    port_indices = [
        index for index, spec in enumerate(graph.nodes) if spec.synthetic == SYNTHETIC_TENSOR
    ]
    if len(port_indices) < 2:
        return
    port_positions = sorted((positions[index] for index in port_indices), key=lambda pos: pos.cx)
    for index in range(1, len(port_positions)):
        left = port_positions[index - 1]
        right = port_positions[index]
        overlap = _node_content_right(left) + MIN_HORIZONTAL_BLOCK_GAP - _node_content_left(right)
        if overlap > 0:
            right.cx += overlap


def _snap_tensor_ports_to_targets(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Align modeling tensor ports with their feed targets for connector routing."""
    for source, target in graph.links:
        if graph.nodes[source].synthetic != SYNTHETIC_TENSOR:
            continue
        positions[source].cx = positions[target].cx


def _inline_frame_bounds_for_node(
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    node_index: int,
) -> tuple[float, float] | None:
    """Return (left, right) content bounds for the inline frame containing ``node_index``."""
    for frame in graph.inline_frames:
        if node_index not in frame.node_indices:
            continue
        indices = list(frame.node_indices)
        left = min(_node_content_left(positions[index]) for index in indices)
        right = max(_node_content_right(positions[index]) for index in indices)
        return left, right
    return None


def _assign_layered_vertical_positions(
    positions: list[LayoutPosition],
    layers: list[list[int]],
    *,
    top_y: float,
    min_gap: float = DETAIL_LAYER_GAP,
    graph: ComputationGraph | None = None,
) -> None:
    """Stack graph layers downward using only the space each row needs."""
    tensor_ports = _graph_has_tensor_ports(graph) if graph is not None else False
    cursor_top = top_y - DETAIL_TOP_INSET
    for layer_index, layer_indices in enumerate(layers):
        layer_positions = [positions[index] for index in layer_indices]
        row_height = max(pos.height for pos in layer_positions)
        for pos in layer_positions:
            pos.top_y = cursor_top
        row_gap = min_gap
        if tensor_ports and layer_index + 1 < len(layers):
            current_has_tensor = any(
                graph is not None and graph.nodes[index].synthetic == SYNTHETIC_TENSOR
                for index in layer_indices
            )
            next_has_kernel = any(
                graph is not None
                and graph.nodes[index].block is not None
                and graph.nodes[index].block.class_name == "KernelOp"
                for index in layers[layer_index + 1]
            )
            if current_has_tensor and next_has_kernel:
                row_gap += TENSOR_PORT_LAYER_EXTRA_GAP
            if (
                current_has_tensor
                and graph is not None
                and graph.inline_frames
                and any(
                    _inline_frame_id_for_node(graph, index) is not None
                    for index in layers[layer_index + 1]
                )
            ):
                row_gap += INPUT_INLINE_FRAME_CAPTION_CLEARANCE
        cursor_top -= row_height + row_gap
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
    graph: ComputationGraph | None = None,
) -> None:
    """Separate nodes in the same layer so boxes and port labels do not overlap."""
    for layer_indices in layers:
        if len(layer_indices) < 2:
            continue
        if graph is not None and graph.inline_frames:
            units = _layer_packing_units(graph, layer_indices)
            unit_entries = [
                (
                    unit,
                    sum(positions[index].cx for index in unit) / len(unit),
                    _packing_unit_width(graph, unit, positions, min_gap=min_gap),
                )
                for unit in units
            ]
            unit_entries.sort(key=lambda item: item[1])
            for index in range(1, len(unit_entries)):
                left_unit, left_cx, left_width = unit_entries[index - 1]
                right_unit, right_cx, right_width = unit_entries[index]
                left_right = left_cx + left_width / 2
                right_left = right_cx - right_width / 2
                overlap = left_right + min_gap - right_left
                if overlap <= 0:
                    continue
                for node_index in right_unit:
                    positions[node_index].cx += overlap
                unit_entries[index] = (
                    right_unit,
                    right_cx + overlap,
                    right_width,
                )
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
    graph: ComputationGraph | None = None,
) -> None:
    """Pack each topological layer to minimum width using the given node order."""
    for layer_indices in layers:
        _pack_ordered_layer_row(
            positions,
            layer_indices,
            anchor_x=anchor_x,
            align_left=align_left,
            min_gap=min_gap,
            graph=graph,
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
        _resolve_horizontal_overlaps(positions, layers, min_gap=min_horizontal_gap, graph=graph)
    _resolve_vertical_overlaps(positions, min_gap=min_vertical_gap)
    if branch_groups:
        _align_fanout_branch_columns(positions)
        _resolve_branch_column_overlaps(positions, min_gap=min_horizontal_gap)
    else:
        _resolve_horizontal_overlaps(positions, layers, min_gap=min_horizontal_gap, graph=graph)


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
    max_diagram_w = max(width for width, _height in diagram_sizes)
    gl_nodes: list[dict[str, float]] = []
    for diagram_w, diagram_h in diagram_sizes:
        gl_nodes.append({"width": diagram_w * PIXELS_PER_UNIT, "height": diagram_h * PIXELS_PER_UNIT})

    links = [{"source": src, "target": tgt} for src, tgt in _layout_graph_links(graph)]
    canvas_w = max(320.0, block_w * 110.0)
    max_layer_h = max(height for _width, height in diagram_sizes)
    layer_separation = max(48.0, max_layer_h * PIXELS_PER_UNIT + 16.0)
    node_separation = max(48.0, max_diagram_w * PIXELS_PER_UNIT * 0.35)
    canvas_h = block_h * 100.0 if block_h is not None else max(240.0, len(graph.nodes) * layer_separation)

    layout = SugiyamaLayout(
        nodes=gl_nodes,
        links=links,
        size=(canvas_w, canvas_h),
        layer_separation=layer_separation,
        node_separation=node_separation,
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
    _compact_layer_positions(positions, layers, anchor_x, align_left=align_left, graph=graph)
    stack_inline_frame_positions(positions, graph)
    _align_merge_nodes(positions, graph)
    _center_align_vertical_chains(positions, graph)
    layers = _optimize_layer_order(layers, graph)
    _compact_layer_positions(positions, layers, anchor_x, align_left=align_left, graph=graph)
    compact_horizontal_shrink_wrap(positions, graph, min_left=content_left if align_left else None)
    _center_align_vertical_chains(positions, graph)
    _order_fanout_branch_positions(positions)
    _align_fanout_branch_columns(positions)
    layers = _optimize_layer_order(_layer_order_from_positions(positions, layers), graph)
    _compact_layer_positions(positions, layers, anchor_x, align_left=align_left, graph=graph)
    _center_align_vertical_chains(positions, graph)
    _resolve_layout_overlaps(positions, graph)
    if align_left:
        _align_positions_left(positions, content_left)
    else:
        _center_positions_horizontally(positions, cx)
    stack_inline_frame_positions(positions, graph)
    finalize_tensor_port_pipeline_layout(
        positions,
        graph,
        min_left=content_left if align_left else None,
    )
    _resolve_layout_overlaps(positions, graph)
    stack_inline_frame_positions(positions, graph)
    _layout_fork_join_branches(positions, graph)
    from visualizer.render_validate import _resolve_same_row_tile_overlaps

    for _ in range(max(1, len(positions))):
        snapshot = [(pos.cx, pos.width, pos.top_y) for pos in positions]
        _resolve_same_row_tile_overlaps(positions, min_gap=MIN_HORIZONTAL_BLOCK_GAP)
        _resolve_layout_overlaps(positions, graph)
        if _graph_has_tensor_ports(graph):
            _align_inline_frame_column_cx(positions, graph)
        if [(pos.cx, pos.width, pos.top_y) for pos in positions] == snapshot:
            break

    finalize_tensor_port_pipeline_layout(
        positions,
        graph,
        min_left=content_left if align_left else None,
    )

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
