"""Build computation graphs from block trees and lay them out with graph-layout."""

from __future__ import annotations

import re
from collections.abc import Sequence
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
    side_producer_has_activation,
    tile_display_labels,
    tile_sublabel,
    wrapper_bullet_lines,
)
from visualizer.ast_analyze import (
    FORWARD_METHOD_INPUT,
    SYNTHETIC_ATTENTION,
    SYNTHETIC_GATE_ACTIVATION,
    is_forward_operation,
)
from visualizer.basic_ops import BasicOpFilter, keep_detail_graph_node
from visualizer.sizing import (
    INLINE_FRAME_CAPTION_BAND,
    INLINE_FRAME_PAD,
    block_sublabel,
    estimate_block_size_for_node,
    min_horizontal_block_gap,
    min_vertical_block_gap,
    PIXELS_PER_UNIT,
    to_layout_pixels,
)

SYNTHETIC_INPUT = "@input"
SYNTHETIC_OUTPUT = "@output"
SYNTHETIC_HIDDEN = "@hidden_states"  # legacy alias; replaced by SYNTHETIC_INPUT in graphs
SYNTHETIC_TENSOR = "@tensor"


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
    link_port_labels: dict[tuple[int, int], str] = field(default_factory=dict)
    inline_frames: list[InlineFrameSpec] = field(default_factory=list)
    side_effect_frame_ids: set[str] = field(default_factory=set)
    # Filled in by the renderer once tiles are positioned: side feeds that drop
    # straight down their source column instead of entering a target's flank.


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


def _operation_tile_label(label: str) -> str:
    """Use ordinary operation names instead of symbolic combine glyphs."""
    return {
        "+": "Add",
        "×": "Multiply",
        "*": "Multiply",
        "ƒ": "Function",
    }.get(label, label)


def _estimate_node_size(spec: GraphNodeSpec) -> tuple[float, float]:
    if spec.synthetic in {SYNTHETIC_INPUT, SYNTHETIC_OUTPUT, SYNTHETIC_HIDDEN}:
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


def _operation_source_indices(
    step: BlockNode,
    attr_last_index: dict[str, int] | None,
    *,
    chain_input_index: int | None = None,
) -> list[int]:
    """Nodes an operation reads from, when its forward names them outright."""
    if attr_last_index is None:
        return []
    sources: list[int] = []
    for predecessor in step.operation_predecessors:
        if predecessor == FORWARD_METHOD_INPUT:
            source_index = chain_input_index
        else:
            source_index = attr_last_index.get(predecessor)
        if source_index is not None and source_index not in sources:
            sources.append(source_index)
    return sources


def _reads_only_a_side_parameter(step: BlockNode) -> bool:
    """True when an operation's operands are a forward parameter rather than the chain.

    Such an operation has no source among the steps it sits between, so falling back to
    the previous step would draw a dataflow edge the forward never performs.
    """
    return (
        is_forward_operation(step.attr_name)
        and bool(step.param_inputs)
        and not step.operation_predecessors
    )


def _side_feed_param_indices(
    graph: ComputationGraph,
    chain_indices: list[int],
) -> dict[str, int]:
    """Map an expanded consumer's forward parameter names to the step that reads them.

    A side-fed module that expands into visible steps has a real docking point for
    each extra argument, so the feed can land on the step that consumes it instead
    of on the chain head.
    """
    targets: dict[str, int] = {}
    for index in chain_indices:
        block = graph.nodes[index].block
        if block is None:
            continue
        for param in block.param_inputs:
            targets.setdefault(param, index)
    return targets


def _is_local_operation_port(spec: GraphNodeSpec) -> bool:
    """True for an external scalar/config operand docked beside one operation."""
    return spec.synthetic == SYNTHETIC_TENSOR and ":external:" in spec.key


def _condition_detail(block: BlockNode | None) -> str | None:
    if block is None:
        return None
    return next(
        (detail for detail in block.details if detail.startswith("condition: ")),
        None,
    )


def _add_conditional_alternative_links(graph: ComputationGraph) -> None:
    """Join complementary assignment branches while keeping the bypass on the side."""
    for earlier, earlier_spec in enumerate(graph.nodes):
        earlier_block = earlier_spec.block
        condition = _condition_detail(earlier_block)
        if earlier_block is None or condition is None:
            continue
        expression = condition.removeprefix("condition: ")
        complement = (
            expression.removeprefix("not (").removesuffix(")")
            if expression.startswith("not (") and expression.endswith(")")
            else f"not ({expression})"
        )
        for later in range(earlier + 1, len(graph.nodes)):
            later_block = graph.nodes[later].block
            if (
                later_block is None
                or _condition_detail(later_block) != f"condition: {complement}"
                or later_block.operation_predecessors
                != earlier_block.operation_predecessors
            ):
                continue
            if (earlier, later) not in graph.links:
                graph.links.append((earlier, later))
            break


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


def add_forward_output(graph: ComputationGraph, *, label: str = "Output") -> int | None:
    """Append one output node fed by every terminal computation path."""
    if not graph.nodes:
        return None
    source_indices = {src for src, _target in graph.links}
    exits = [
        index
        for index, spec in enumerate(graph.nodes)
        if index not in source_indices
        and spec.synthetic not in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN, SYNTHETIC_TENSOR}
    ]
    if not exits:
        return None
    framed_indices = {
        index for frame in graph.inline_frames for index in frame.node_indices
    }
    unframed_exits = [index for index in exits if index not in framed_indices]
    if unframed_exits:
        exits = unframed_exits
    output_index = _add_node(
        graph,
        key=SYNTHETIC_OUTPUT,
        label=label,
        synthetic=SYNTHETIC_OUTPUT,
    )
    graph.links.extend((index, output_index) for index in exits)
    return output_index


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
        if spec.synthetic not in {SYNTHETIC_INPUT, SYNTHETIC_OUTPUT, SYNTHETIC_HIDDEN}
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
    chain_input_index = last_index if last_index is not None else input_index

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

        explicit_sources = _operation_source_indices(
            sub_step,
            attr_last_index,
            chain_input_index=chain_input_index,
        )
        for source_index in explicit_sources:
            graph.links.append((source_index, step_index))

        if not explicit_sources and not _reads_only_a_side_parameter(sub_step):
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

        _append_kernel_second_operand_link(
            graph,
            sub_step,
            step_index=step_index,
            attr_last_index=attr_last_index,
            chain_input_index=chain_input_index,
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
    )
    graph.links.append((situ_index, mult_index))
    graph.links.append((up_index, mult_index))
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


def _resolve_kernel_second_operand_index(
    step: BlockNode,
    attr_last_index: dict[str, int] | None,
    *,
    chain_input_index: int | None,
) -> int | None:
    """Resolve the optional second operand for an inline kernel sub-op."""
    second = step.kernel_second_operand
    if second is None:
        return None
    if second == "input":
        return chain_input_index
    if attr_last_index is None:
        return None
    return attr_last_index.get(second)


def _append_kernel_second_operand_link(
    graph: ComputationGraph,
    step: BlockNode,
    *,
    step_index: int,
    attr_last_index: dict[str, int] | None,
    chain_input_index: int | None,
) -> None:
    source_index = _resolve_kernel_second_operand_index(
        step,
        attr_last_index,
        chain_input_index=chain_input_index,
    )
    if source_index is None:
        return
    _append_operand_link(
        graph,
        source_index=source_index,
        target_index=step_index,
    )


def _append_operand_link(
    graph: ComputationGraph,
    *,
    source_index: int,
    target_index: int,
) -> None:
    """Wire an explicit operand into its target tile."""
    link = (source_index, target_index)
    if link not in graph.links:
        graph.links.append(link)


def _append_side_producer_link(
    graph: ComputationGraph,
    *,
    source_index: int,
    target_index: int,
) -> None:
    graph.links.append((source_index, target_index))



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

    for removed in remove_indices:
        for source in preds[removed]:
            for target in succs[removed]:
                port_label = graph.link_port_labels.get((source, removed)) or graph.link_port_labels.get(
                    (removed, target)
                )
                dashed = (source, removed) in graph.dashed_links or (removed, target) in graph.dashed_links
                for kept_source in _expand_preds(source):
                    for kept_target in _expand_succs(target):
                        if kept_source == kept_target:
                            continue
                        bridged_links.add((kept_source, kept_target))
                        if port_label and (kept_source, kept_target) not in bridged_port_labels:
                            bridged_port_labels[(kept_source, kept_target)] = port_label
                        if dashed:
                            bridged_dashed.add((kept_source, kept_target))

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
    input_operand_attrs: dict[str, list[str]] = {}
    step_attr_indices: dict[str, dict[str, int]] = {}
    for step_index, step in enumerate(segment.steps):
        if step.children and len(step.children) >= 2:
            attr_last_index: dict[str, int] = {}
            input_operand_attrs[step.attr_name] = [
                child.attr_name
                for child in step.children
                if child.kernel_second_operand == "input"
            ]
            sub_indices, sub_tail = _add_linear_pipeline_chain(
                graph,
                step.children,
                wrapper=step,
                key_prefix=f"{key_prefix}:pipeline:{step.attr_name}",
                attr_last_index=attr_last_index,
            )
            step_attr_indices[step.attr_name] = attr_last_index
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
        for child_attr in input_operand_attrs.get(target_attr, []):
            child_index = step_attr_indices.get(target_attr, {}).get(child_attr)
            if child_index is not None:
                graph.links.append((port_index, child_index))

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
                for branch_offset, tail in enumerate(branch_tails):
                    graph.links.append((tail, merge_index))
                last_index = merge_index
                _track_attr_index(attr_last_index, segment.merge.attr_name, merge_index)
            continue

        if isinstance(segment, SideCombineSegment):
            from visualizer.ast_analyze import MOE_AGGREGATION_LABEL, combine_op_from_step_details

            if combine_op_from_step_details(list(segment.consumer.details or [])) == MOE_AGGREGATION_LABEL:

                agg_index = _add_node(
                    graph,
                    key=f"moe_agg:{segment_index}:{segment.consumer.attr_name}",
                    label=MOE_AGGREGATION_LABEL,
                )
                if last_index is not None:
                    graph.links.append((last_index, agg_index))
                elif input_index is not None:
                    graph.links.append((input_index, agg_index))
                for side in segment.sides:
                    if side.source_kind == "forward_input":
                        if input_index is not None:
                            _link_forward_input(graph, input_index, agg_index)
                        continue
                    source_attr = side.source_chain[-1] if side.source_chain else None
                    if source_attr is None:
                        continue
                    source_index = attr_last_index.get(source_attr)
                    if source_index is None:
                        continue
                    link_key = (source_index, agg_index)
                    graph.links.append(link_key)
                    if side.port_label and side.port_label != "router":
                        graph.link_port_labels[link_key] = side.port_label
                last_index = agg_index
                _track_attr_index(attr_last_index, segment.consumer.attr_name, agg_index)
                continue

            combine_index = _add_node(
                graph,
                key=f"sidecombine:{segment_index}:{segment.consumer.attr_name}",
                label=_operation_tile_label(segment.op),
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
                    if any(side.side_effect_call for side in segment.sides):
                        graph.side_effect_frame_ids.add(wrapper.attr_name)
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
                label="Add",
            )
            if last_index is not None:
                graph.links.append((last_index, combine_index))
            graph.links.append((module_tail, combine_index))
            last_index = combine_index
            continue

        if isinstance(segment, SideFeedSegment):
            consumer = segment.consumer
            port_label = _consumer_port_label(segment.sides)
            # A module invoked on the forward input branches there instead of running in
            # series behind the step before it, even when its extra operands come from
            # elsewhere in the graph.
            forward_input_is_main = consumer.attr_name in root.input_fed_steps or any(
                side.source_kind == "forward_input" for side in segment.sides
            )

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

                combine_index = _add_node(
                    graph,
                    key=f"sidefeed:{segment_index}:{consumer.attr_name}:mul",
                    label="Multiply",
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

            entry_index: int | None = None
            param_entry_indices: dict[str, int] = {}
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
                # A straight-line consumer expands into its own steps here too, so a
                # side-fed module is not left as an opaque tile with nothing behind it.
                expanded_steps, wrapper = inline_composite_steps(
                    consumer,
                    basic_ops=basic_ops,
                )
                if wrapper is not None:
                    chain_indices, chain_tail = _add_linear_pipeline_chain(
                        graph,
                        expanded_steps,
                        wrapper=wrapper,
                        key_prefix=f"sidefeed:{segment_index}:{consumer.attr_name}",
                        attr_last_index=attr_last_index,
                        port_label=port_label,
                        port_style="inline" if port_label else None,
                        input_index=input_index,
                        last_index=input_index if forward_input_is_main else last_index,
                    )
                    entry_index = chain_indices[0] if chain_indices else None
                    consumer_index = chain_tail
                    param_entry_indices = _side_feed_param_indices(graph, chain_indices)
                else:
                    consumer_index = _add_node(
                        graph,
                        key=f"sidefeed:{segment_index}:{consumer.attr_name}",
                        block=consumer,
                        port_label=port_label,
                        port_style="inline" if port_label else None,
                    )
                    entry_index = consumer_index
            if entry_index is None:
                entry_index = consumer_index
                if forward_input_is_main and input_index is not None:
                    _link_forward_input(graph, input_index, consumer_index)
                elif last_index is not None:
                    graph.links.append((last_index, consumer_index))
                elif input_index is not None:
                    graph.links.append((input_index, consumer_index))
            for side in segment.sides:
                if side.source_kind == "forward_input":
                    if input_index is not None and not forward_input_is_main:
                        _link_forward_input(graph, input_index, entry_index)
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
                # A feed from the step right before the consumer is already the spine
                # link; do not duplicate that operand edge.
                if source_index == last_index and not forward_input_is_main:
                    continue
                _append_side_producer_link(
                    graph,
                    source_index=source_index,
                    target_index=param_entry_indices.get(side.arg_name, entry_index),
                )
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
                label=_operation_tile_label(segment.op),
            )
            graph.links.append((last_index, mult_index))
            graph.links.append((side_index, mult_index))
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
            ) or step.attr_name in root.input_fed_steps
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
                # An operation naming the steps it reads keeps those dataflow edges.
                explicit_sources = _operation_source_indices(sub_step, attr_last_index)
                if explicit_sources:
                    for source_index in explicit_sources:
                        graph.links.append((source_index, step_index))
                elif not _reads_only_a_side_parameter(sub_step):
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

    _add_conditional_alternative_links(graph)
    if basic_ops is not None and basic_ops.basic_only:
        return _filter_graph_basic_only(graph)
    return graph


FLOATING_PORT_LABEL_CLEARANCE = 0.46
DETAIL_LAYER_GAP = 2 * min_vertical_block_gap()
# Room a dotted border needs outside itself before it reads as touching the tile it
# leaves out: a layer gap once the frame's own padding is taken. Stacking a frame and
# validating its borders both work from this, so neither can undo the other's spacing.
FRAME_BORDER_CLEARANCE = DETAIL_LAYER_GAP - INLINE_FRAME_PAD
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
        if (pred, index) not in graph.dashed_links
    ]
    if len(forward) == 1:
        return forward[0]
    if len(predecessors) == 1:
        return predecessors[0]
    return None


def _forward_successors(
    graph: ComputationGraph,
    outgoing: list[list[int]],
    index: int,
) -> list[int]:
    """Targets on the forward path, ignoring dashed feeds."""
    return [
        target
        for target in outgoing[index]
        if (index, target) not in graph.dashed_links
    ]


def _chain_feeder_cx(
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    incoming: list[list[int]],
    outgoing: list[list[int]],
    index: int,
    *,
    exclude: set[int] | None = None,
) -> float | None:
    """Column of the step feeding this one, when the chain simply continues into it."""
    predecessor = _spine_predecessor(graph, incoming, index)
    if predecessor is None or (exclude is not None and predecessor in exclude):
        return None
    if not _is_layout_chain_node(positions[predecessor].spec):
        return None
    successors = _forward_successors(graph, outgoing, predecessor)
    if len(successors) != 1 or successors[0] != index:
        return None
    return positions[predecessor].cx


def _align_input_over_single_target(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Sit the input tile on the column of the one step it feeds, leaving no jog."""
    outgoing: list[list[int]] = [[] for _ in range(len(positions))]
    for source, target in graph.links:
        if source < len(positions) and target < len(positions):
            outgoing[source].append(target)
    for index, pos in enumerate(positions):
        if pos.spec.synthetic not in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}:
            continue
        targets = _forward_successors(graph, outgoing, index)
        if len(targets) == 1:
            pos.cx = positions[targets[0]].cx


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
        side_nodes = _operation_dag_side_nodes(graph, frame)
        center_indices = [index for index in indices if index not in side_nodes] or indices
        # A frame the chain simply flows into belongs in the column that feeds it,
        # rather than wherever the layer packing happened to leave its own steps.
        frame_cx = _chain_feeder_cx(
            graph,
            positions,
            incoming,
            outgoing,
            indices[0],
            exclude=set(indices),
        )
        if frame_cx is None:
            frame_cx = sum(positions[index].cx for index in center_indices) / len(center_indices)
        for index in indices:
            positions[index].cx = frame_cx
        _layout_operation_dag_frame(positions, graph, frame)

    # A section that expands a module's own math has no frame around those steps,
    # so its bypass still needs the skipped step moved out of the main column.
    unframed = _unframed_step_indices(graph)
    if len(unframed) > 1:
        _layout_operation_dag_columns(positions, graph, unframed)

    _align_input_over_single_target(positions, graph)


def _pack_ordered_layer_row(
    positions: list[LayoutPosition],
    layer_indices: list[int],
    *,
    anchor_x: float,
    align_left: bool,
    min_gap: float,
    graph: ComputationGraph | None = None,
    column_cx: float | None = None,
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
            if align_left:
                target_cx = column_cx if column_cx is not None else anchor_x + width / 2
            else:
                target_cx = anchor_x
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
        if align_left:
            pos.cx = column_cx if column_cx is not None else anchor_x + pos.width / 2
        else:
            pos.cx = anchor_x
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


def _inline_frame_top_member_index(graph: ComputationGraph, node_index: int) -> bool:
    """True when node_index is the topmost member of an inline frame column."""
    for frame in graph.inline_frames:
        if frame.node_indices and frame.node_indices[0] == node_index:
            return True
    return False


def _input_fanout_target_top_y(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    target_index: int,
) -> float:
    """Top Y used for input fan-out clearance, honoring expanded frame envelopes."""
    for frame in graph.inline_frames:
        if frame.node_indices and frame.node_indices[0] == target_index:
            from visualizer.render import (
                _inline_frame_caption_band_top,
                _inline_frame_draw_bounds,
            )

            bounds = _inline_frame_draw_bounds(frame, positions, graph)
            # A feed into the frame's first tile crosses the border above that tile, so
            # it has to clear the caption sitting in the band there too.
            return _inline_frame_caption_band_top(frame, bounds)
    return positions[target_index].top_y


def _compact_attention_side_feeder_columns(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float | None = None,
) -> None:
    """Shrinkwrap side feeder columns toward Attention-style merge nodes."""
    from visualizer.sizing import min_vertical_block_gap

    gap = min_vertical_block_gap() if min_gap is None else min_gap
    incoming = _build_incoming_links(graph, node_count=len(positions))
    for target, sources in incoming.items():
        if graph.nodes[target].label != "Attention":
            continue
        target_pos = positions[target]
        side_sources = [
            source
            for source in sources
            if abs(positions[source].cx - target_pos.cx) > 0.08
        ]
        if not side_sources:
            continue
        lowest_needed_top = target_pos.top_y + gap
        for source in side_sources:
            branch_indices = next(
                (
                    indices
                    for indices in _fanout_branch_node_groups(positions).values()
                    if source in indices
                ),
                [source],
            )
            branch_top = max(positions[index].top_y for index in branch_indices)
            shift = branch_top - lowest_needed_top
            if shift <= gap:
                continue
            for index in branch_indices:
                positions[index].top_y -= shift


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
    from visualizer.render import (
        INLINE_FRAME_PAD,
        _estimate_inline_frame_gutter_width,
        _inline_frame_nesting_depth,
    )

    frame = next(frame for frame in graph.inline_frames if frame.frame_id == frame_id)
    widths = [_diagram_size_for_spec(graph.nodes[index])[0] for index in frame.node_indices]
    if not widths:
        return 0.0
    gutter = _estimate_inline_frame_gutter_width(graph, frame)
    pad = INLINE_FRAME_PAD * (1 + _inline_frame_nesting_depth(graph, frame))
    chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
    if len(chain) >= 2:
        return max(widths) + 2 * pad + gutter
    return sum(widths) + min_gap * max(0, len(widths) - 1) + 2 * pad + gutter


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
    from visualizer.render import (
        _inline_frame_connector_gutter_width,
        _inline_frame_nesting_depth,
    )

    frame = _inline_frame_for_indices(graph, indices)
    left = min(_node_content_left(positions[index]) for index in indices)
    right = max(_node_content_right(positions[index]) for index in indices)
    if frame is not None:
        left_gutter = _inline_frame_connector_gutter_width(graph, frame, positions, side="left")
        right_gutter = _inline_frame_connector_gutter_width(graph, frame, positions, side="right")
        pad *= 1 + _inline_frame_nesting_depth(graph, frame)
        return left - pad - left_gutter, right + pad + right_gutter
    return left - pad, right + pad


def _inter_inline_frame_gap(
    graph: ComputationGraph,
    left_indices: list[int],
    right_indices: list[int],
    *,
    base_gap: float,
) -> float:
    from visualizer.render import INLINE_FRAME_LABEL_CHAR_W

    extra = 0.0
    for indices in (left_indices, right_indices):
        frame = _inline_frame_for_indices(graph, indices)
        if frame is None:
            continue
        label = frame.label.strip()
        if label:
            label_extent = min(len(label) * INLINE_FRAME_LABEL_CHAR_W, 1.6)
            extra = max(extra, label_extent)
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
    pad = INLINE_FRAME_PAD
    if frame is not None:
        from visualizer.render import (
            _inline_frame_nesting_depth,
            _inline_frame_total_connector_gutter_width,
        )

        gutter = _inline_frame_total_connector_gutter_width(graph, frame, positions)
        pad *= 1 + _inline_frame_nesting_depth(graph, frame)
    return right - left + 2 * pad + gutter


def _ordered_inline_frame_chain(
    graph: ComputationGraph,
    frame_indices: list[int],
) -> list[int]:
    """Return frame member indices in forward link order when they form a chain."""
    frame_indices = [
        index
        for index in frame_indices
        if not _is_local_operation_port(graph.nodes[index])
    ]
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
        spine_successors = list(outgoing.get(cursor, []))
        cursor = spine_successors[0] if len(spine_successors) == 1 else None

    if len(ordered) == len(frame_indices):
        return ordered
    return sorted(frame_indices, key=lambda index: index)


def _unframed_step_indices(graph: ComputationGraph) -> list[int]:
    """Steps of a section that no inline frame groups."""
    framed = {index for frame in graph.inline_frames for index in frame.node_indices}
    return [index for index in range(len(graph.nodes)) if index not in framed]


def _operation_dag_side_nodes(
    graph: ComputationGraph,
    frame: InlineFrameSpec,
) -> set[int]:
    """Nodes on the long arm of an ancestor-bypass join inside an op frame."""
    return _operation_dag_side_nodes_among(graph, frame.node_indices)


def _operation_dag_side_nodes_among(
    graph: ComputationGraph,
    member_indices: Sequence[int],
) -> set[int]:
    """Nodes on the long arm of an ancestor-bypass join among the given steps."""
    if not any(
        graph.nodes[index].block is not None
        and graph.nodes[index].block.operation_predecessors
        for index in member_indices
    ):
        return set()
    members = set(member_indices)
    # Feeds arriving from outside count too, so a frame nested inside another sees the
    # same forks its parent does. Blind to them it would read a fork as a plain chain
    # and collapse the columns the parent deliberately split, and the two passes would
    # then shunt the frame further sideways on every round.
    incoming: dict[int, list[int]] = {index: [] for index in members}
    for source, target in graph.links:
        if target in members:
            incoming[target].append(source)

    def path_to_ancestor(start: int, ancestor: int, visited: set[int]) -> list[int] | None:
        if start == ancestor:
            return []
        if start in visited:
            return None
        visited.add(start)
        for predecessor in incoming.get(start, []):
            path = path_to_ancestor(predecessor, ancestor, visited)
            if path is not None:
                return [start, *path]
        return None

    def longest_path_to_ancestor(
        start: int,
        ancestor: int,
        visited: set[int],
    ) -> list[int] | None:
        if start == ancestor:
            return []
        if start in visited:
            return None
        candidates = [
            [start, *path]
            for predecessor in incoming.get(start, [])
            if (
                path := longest_path_to_ancestor(
                    predecessor,
                    ancestor,
                    visited | {start},
                )
            )
            is not None
        ]
        return max(candidates, key=len) if candidates else None

    def ancestors(start: int) -> set[int]:
        found: set[int] = set()
        pending = list(incoming.get(start, []))
        while pending:
            predecessor = pending.pop()
            if predecessor in found:
                continue
            found.add(predecessor)
            pending.extend(incoming.get(predecessor, []))
        return found

    side_nodes: set[int] = set()
    for join, predecessors in incoming.items():
        if len(predecessors) < 2:
            continue
        found_ancestor_bypass = False
        for direct in predecessors:
            for branch_tail in predecessors:
                if branch_tail == direct:
                    continue
                path = path_to_ancestor(branch_tail, direct, set())
                if path:
                    side_nodes.update(path)
                    found_ancestor_bypass = True
                    break
                branch_block = graph.nodes[branch_tail].block
                if branch_block is not None and branch_block.external_inputs:
                    side_nodes.add(branch_tail)
                    found_ancestor_bypass = True
                    break
        if found_ancestor_bypass:
            continue

        # A true fork can have two sibling arms that meet here. Put the longer
        # arm beside the short one; otherwise a scalar branch such as
        # `(up + 1) * glu` gets interleaved with the gated activation chain.
        if (
            not _is_multiply_label(graph.nodes[join].label)
            or not any(_is_summation_label(graph.nodes[index].label) for index in predecessors)
        ):
            continue
        for index, left in enumerate(predecessors):
            for right in predecessors[index + 1 :]:
                common = ancestors(left) & ancestors(right)
                candidate_paths = [
                    path
                    for ancestor in common
                    for start in (left, right)
                    if (
                        path := longest_path_to_ancestor(start, ancestor, set())
                    )
                ]
                if not candidate_paths:
                    continue
                longest = max(candidate_paths, key=len)
                if len(longest) > 1:
                    side_nodes.update(longest)
    return side_nodes & members


def _dead_end_branch_nodes_among(
    graph: ComputationGraph,
    member_indices: Sequence[int],
) -> set[int]:
    """Members nothing reads, sitting beside a chain that carries on past them.

    A step with no consumer is off the flow, so leaving it in the column puts it
    under the connector that carries the chain past it. A column of its own keeps
    that lane clear.
    """
    members = set(member_indices)
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {index: [] for index in members}
    for source, target in graph.links:
        outgoing.setdefault(source, []).append(target)
        if source in members and target in members:
            incoming[target].append(source)

    side: set[int] = set()
    for index in member_indices:
        if outgoing.get(index):
            continue
        feeders = incoming.get(index, [])
        if len(feeders) != 1:
            continue
        if not any(
            target in members and target != index
            for target in outgoing.get(feeders[0], [])
        ):
            continue
        side.add(index)
    return side


def _offset_column_nodes(
    graph: ComputationGraph,
    frame: InlineFrameSpec,
) -> set[int]:
    """Frame members the layout parks in a column beside the main chain."""
    return _operation_dag_side_nodes(graph, frame) | _dead_end_branch_nodes_among(
        graph,
        frame.node_indices,
    )


def _inline_frame_column_skip_links(
    graph: ComputationGraph,
    frame: InlineFrameSpec,
) -> list[tuple[int, int]]:
    """In-frame links that have to pass steps standing in their own column."""
    members = set(frame.node_indices)
    chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
    rank = {index: position for position, index in enumerate(chain)}
    offset = _offset_column_nodes(graph, frame)
    links = []
    for source, target in graph.links:
        if source not in members or target not in members:
            continue
        if source not in rank or target not in rank:
            continue
        if rank[target] - rank[source] < 2:
            continue
        if (source in offset) != (target in offset):
            continue
        if any(
            (step in offset) == (source in offset)
            for step in chain[rank[source] + 1 : rank[target]]
        ):
            links.append((source, target))
    return links


def _row_gap_rules(graph: ComputationGraph):
    """Row-gap floors between two tiles stacked one above the other."""
    frame_member_sets = [set(frame.node_indices) for frame in graph.inline_frames]
    caption_heads = {
        frame.node_indices[0]: set(frame.node_indices)
        for frame in graph.inline_frames
        if frame.node_indices and frame.label.strip()
    }
    skips = [
        link
        for frame in graph.inline_frames
        for link in _inline_frame_column_skip_links(graph, frame)
    ]
    skip_sources = {source for source, _ in skips}
    skip_targets = {target for _, target in skips}
    offset_nodes: set[int] = set()
    for frame in graph.inline_frames:
        offset_nodes |= _offset_column_nodes(graph, frame)
    band = _skip_crossing_band()
    channel = _approach_channel_gap()
    side_entries = set(_infer_side_entry_links(graph))
    feeds: dict[int, list[tuple[int, int]]] = {index: [] for index in range(len(graph.nodes))}
    for link in graph.links:
        if link not in side_entries:
            feeds[link[1]].append(link)

    def required(upper: int, lower: int) -> float:
        gap = _frame_border_gap(upper, lower, frame_member_sets)
        # Entering a captioned frame costs its caption band as well as its border, and
        # the tile above has to clear both or the caption lands on it.
        members = caption_heads.get(lower)
        if members is not None and upper not in members:
            gap = max(gap, INLINE_FRAME_PAD + INLINE_FRAME_CAPTION_BAND + FRAME_BORDER_CLEARANCE)
        # A skip leaves its column just below its source and rejoins just above its
        # target, so those two rows carry its horizontal runs. A step parked beside
        # the chain is reached the same way, by a run in the row above it.
        crosses_columns = (upper in offset_nodes) != (lower in offset_nodes)
        if crosses_columns or upper in skip_sources or lower in skip_targets:
            gap = max(gap, band)
        # Only the tile directly above can drop straight into its target; every other feed
        # has to turn in the row between the two, and two feeds turning at the same height
        # would be drawn as one line. Deepen the row so each gets an approach of its own.
        turning = sum(1 for link in feeds[lower] if link[0] != upper)
        if turning > 1:
            gap = max(gap, band + (turning - 1) * channel)
        return gap

    return required


def _skip_crossing_band() -> float:
    """Row depth a skip needs to leave its column and reach its gutter.

    The jog wants a full exit stub, the obstacle margin of the step it passes, and
    the hair that keeps the run off that step's edge; with any less the connector
    has to run through the step instead of around it.
    """
    from visualizer.render import (
        CONNECTOR_ATTACHED_BOX_MARGIN,
        CONNECTOR_EXIT_STUB,
        CONNECTOR_OBSTACLE_MARGIN,
    )

    return CONNECTOR_EXIT_STUB + CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN


def _approach_channel_gap() -> float:
    """Height one more approach row costs, matching the router's channel spacing."""
    from visualizer.render import PARALLEL_CONNECTOR_CHANNEL_GAP

    return PARALLEL_CONNECTOR_CHANNEL_GAP


def _layout_operation_dag_frame(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    frame: InlineFrameSpec,
) -> None:
    _layout_operation_dag_columns(positions, graph, frame.node_indices)


def _layout_operation_dag_columns(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    member_indices: Sequence[int],
) -> None:
    """Offset the long arm of a bypass join into a column beside the main chain."""
    side_nodes = _operation_dag_side_nodes_among(graph, member_indices)
    side_nodes |= _dead_end_branch_nodes_among(graph, member_indices)
    if not side_nodes:
        return
    central = [index for index in member_indices if index not in side_nodes]
    if not central:
        return
    frame_cx = _dominant_frame_column(positions, central)
    central_half = max(positions[index].width for index in central) / 2
    side_half = max(positions[index].width for index in side_nodes) / 2
    side_cx = frame_cx - central_half - side_half - min_horizontal_block_gap()
    for index in central:
        positions[index].cx = frame_cx
    for index in side_nodes:
        positions[index].cx = side_cx


def _shared_fork_predecessors(
    incoming: dict[int, list[int]],
    left: int,
    right: int,
) -> set[int]:
    return set(incoming[left]) & set(incoming[right])


def _is_multiply_label(label: str) -> bool:
    """Element-wise product, however the model spells it."""
    return label.strip() in {"×", "x", "*", "⨉", "Multiply"}


def _is_summation_label(label: str) -> bool:
    """Element-wise sum, however the model spells it."""
    return label.strip() in {"Add", "+"}


def _is_multiply_combine(graph: ComputationGraph, join: int) -> bool:
    return _is_multiply_label(graph.nodes[join].label)


def _infer_side_entry_links(graph: ComputationGraph) -> list[tuple[int, int]]:
    """Side feeds that enter a combine from outside its inline activation frame."""
    incoming: dict[int, list[int]] = {index: [] for index in range(len(graph.nodes))}
    for source, target in graph.links:
        incoming[target].append(source)

    side_entries: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for frame in graph.inline_frames:
        frame_members = set(frame.node_indices)
        for join in frame_members:
            if not _is_multiply_combine(graph, join):
                continue
            for source in incoming[join]:
                if source in frame_members:
                    continue
                link = (source, join)
                if link not in seen:
                    seen.add(link)
                    side_entries.append(link)
    return side_entries


def _find_fork_join_clusters(graph: ComputationGraph) -> list[ForkJoinCluster]:
    """Return fork/join clusters: parallel branches meeting at ×, then continuing downstream."""
    clusters: list[ForkJoinCluster] = []
    seen: set[tuple[int, int, int, int, int]] = set()

    incoming: dict[int, list[int]] = {index: [] for index in range(len(graph.nodes))}
    outgoing: dict[int, list[int]] = {index: [] for index in range(len(graph.nodes))}
    link_set = set(graph.links)
    side_entry_links = set(_infer_side_entry_links(graph))
    for source, target in graph.links:
        incoming[target].append(source)
        outgoing[source].append(target)

    for side_source, join in side_entry_links:
        if not _is_multiply_combine(graph, join):
            continue

        main_branch_candidates = [
            source
            for source in incoming[join]
            if (source, join) not in side_entry_links
        ]
        if len(main_branch_candidates) != 1:
            continue
        main_branch = main_branch_candidates[0]

        main_source_candidates = [
            source
            for source in incoming[main_branch]
            if (source, main_branch) not in graph.dashed_links
            and (source, main_branch) not in side_entry_links
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
    # The side source is parked on the main branch's row, so its feed into the join runs
    # through this gap rather than dropping straight in, and needs the corridor band.
    required_row_gap = _row_gap_rules(graph)
    join_gap = max(
        v_gap,
        required_row_gap(cluster.main_branch, cluster.join),
        required_row_gap(cluster.side_source, cluster.join),
    )
    main_branch_pos.top_y = join_pos.top_y + join_gap + main_branch_pos.height

    inner_block_top = main_branch_pos.top_y + caption_band
    main_source_pos.cx = main_cx
    main_source_pos.top_y = inner_block_top + v_gap + main_source_pos.height

    tail_pos.cx = main_cx
    plus_index = next(
        (
            target
            for source, target in graph.links
            if source == cluster.tail and _is_summation_label(graph.nodes[target].label)
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


def _ensure_multi_branch_input_fanout_clearance(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float | None = None,
) -> None:
    """Reserve connector clearance above dense multi-branch fan-outs."""
    if len(_fanout_branch_node_groups(positions)) < 3:
        return
    from visualizer.render import CONNECTOR_ATTACHED_BOX_MARGIN, CONNECTOR_OBSTACLE_MARGIN

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
    min_fanout_clearance = gap + CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN
    input_pos = positions[input_index]
    highest_top = max(
        _input_fanout_target_top_y(positions, graph, target) for target in targets
    )
    required_bottom = highest_top + min_fanout_clearance
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
    highest_top = max(
        _input_fanout_target_top_y(positions, graph, target) for target in targets
    )
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
        from visualizer.render import CONNECTOR_ATTACHED_BOX_MARGIN, CONNECTOR_OBSTACLE_MARGIN

        min_fanout_clearance = gap + CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN
        allowable = min(
            positions[target].top_y + min_fanout_clearance - input_pos.bottom
            for target in targets
        )
        shift = max(shift, allowable)
        if shift >= -1e-6:
            return
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
            if spec.label == MOE_AGGREGATION_LABEL
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

    shared_experts_frame = next(
        (frame for frame in graph.inline_frames if frame.frame_id == "shared_experts"),
        None,
    )
    if shared_experts_frame is not None:
        spine -= set(shared_experts_frame.node_indices)
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
            if spec.label == MOE_AGGREGATION_LABEL
        ),
        None,
    )
    if sigma is None:
        return

    spine_cx = positions[sigma].cx
    spine_indices = _router_spine_column_indices(positions, graph)
    for index in spine_indices:
        positions[index].cx = spine_cx
    _compact_router_spine_vertical_gaps(positions, graph, spine_indices)


def _compact_router_spine_vertical_gaps(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    spine_indices: set[int],
) -> None:
    """Keep the routed-expert residual merge one detail layer below the spine up tile."""
    by_label = {spec.label: index for index, spec in enumerate(graph.nodes)}
    residual_label = "+" if "+" in by_label else "Add"
    if residual_label not in by_label:
        return
    up_index = next(
        (
            index
            for index in spine_indices
            if positions[index].spec.block is not None
            and positions[index].spec.block.attr_name == "routed_expert_up_proj"
        ),
        None,
    )
    add_index = by_label[residual_label]
    if up_index is None or add_index not in spine_indices:
        return
    desired_top = positions[up_index].bottom - DETAIL_LAYER_GAP
    if positions[add_index].top_y < desired_top - 1e-6:
        positions[add_index].top_y = desired_top


def _align_gated_norm_output_spine(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Keep @attn_output → o_norm → gated multiply → o_proj on one column."""
    by_attr = {
        pos.spec.block.attr_name: index
        for index, pos in enumerate(positions)
        if pos.spec.block is not None
    }
    required = ("@attn_output", "o_norm", "o_proj")
    if not all(name in by_attr for name in required):
        return
    combine = next(
        (index for index, pos in enumerate(positions) if _is_multiply_label(pos.spec.label)),
        None,
    )
    if combine is None:
        return
    o_norm = by_attr["o_norm"]
    o_proj = by_attr["o_proj"]
    incoming = _build_incoming_links(graph, node_count=len(positions))
    if o_norm not in incoming.get(combine, []) or combine not in incoming.get(o_proj, []):
        return
    spine_cx = positions[by_attr["@attn_output"]].cx
    for index in (by_attr["@attn_output"], o_norm, combine, o_proj):
        positions[index].cx = spine_cx


def _layout_fork_join_branches(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    for cluster in _find_fork_join_clusters(graph):
        _layout_fork_join_branch(positions, graph, cluster)
    _clear_side_branches_from_gate_frame(positions, graph)
    _align_router_spine_column(positions, graph)
    _align_gated_norm_output_spine(positions, graph)


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

    spine_indices = _router_spine_column_indices(positions, graph)
    if spine_indices and gate_frame is not None:
        spine_left = min(_node_content_left(positions[index]) for index in spine_indices)
        if spine_left < min_left:
            delta = min_left - spine_left
            for index in spine_indices:
                positions[index].cx += delta
    if spine_indices:
        min_left = max(
            min_left,
            max(_node_content_right(positions[index]) for index in spine_indices)
            + min_horizontal_block_gap()
            + INLINE_FRAME_PAD,
        )
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
            # Members that travel with the cluster cannot constrain where the cluster
            # goes: shifting it would carry them along and demand the same shift again.
            anchors = [index for index in frame.node_indices if index not in cluster_indices]
            if not anchors:
                continue
            frame_right = max(_node_content_right(positions[index]) for index in anchors)
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
        side_nodes = _operation_dag_side_nodes(graph, frame)
        center_indices = [index for index in indices if index not in side_nodes] or indices
        frame_cx = sum(positions[index].cx for index in center_indices) / len(center_indices)
        for index in indices:
            positions[index].cx = frame_cx


def _inline_frame_vertical_gap(graph, frame) -> float:
    """Reserve one capped row corridor for any number of top-entry bypasses."""
    from visualizer.render import (
        INLINE_FRAME_BYPASS_ROW_GAP,
        _inline_frame_bypass_link_count,
    )
    from visualizer.sizing import min_vertical_block_gap

    extra = INLINE_FRAME_BYPASS_ROW_GAP if _inline_frame_bypass_link_count(graph, frame) else 0.0
    return min_vertical_block_gap() + extra


def stack_fanout_branch_columns(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float | None = None,
) -> None:
    """Re-stack each fan-out branch as a straight vertical chain on one column."""
    from visualizer.sizing import min_vertical_block_gap

    branch_groups = _fanout_branch_node_groups(positions)
    if not branch_groups:
        return
    gap = min_vertical_block_gap() if min_gap is None else min_gap
    for indices in branch_groups.values():
        chain = _ordered_inline_frame_chain(graph, list(indices))
        stack_chain = [
            index
            for index in chain
            if graph.nodes[index].label in {"Linear", "RMSNorm"}
        ]
        if len(stack_chain) < 2:
            if stack_chain:
                positions[stack_chain[0]].cx = sum(
                    positions[index].cx for index in indices
                ) / len(indices)
            continue
        column_cx = sum(positions[index].cx for index in stack_chain) / len(stack_chain)
        restack = chain[chain.index(stack_chain[0]) :]
        cursor_top = max(positions[index].top_y for index in stack_chain)
        # Re-stacking must not undo the extra room an inline frame asked for to fit
        # the corridors its bypass connectors tee into.
        frame = _inline_frame_for_indices(graph, restack)
        chain_gap = gap if frame is None else max(gap, _inline_frame_vertical_gap(graph, frame))
        required_row_gap = _row_gap_rules(graph)
        for position_in_chain, index in enumerate(restack):
            pos = positions[index]
            pos.cx = column_cx
            pos.top_y = cursor_top
            next_gap = chain_gap
            if position_in_chain + 1 < len(restack):
                next_gap = max(
                    next_gap,
                    required_row_gap(index, restack[position_in_chain + 1]),
                )
            cursor_top -= pos.height + next_gap
        for index in chain:
            if index not in restack:
                positions[index].cx = column_cx


def pack_input_fed_inline_frame_branches(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Dock input-fed inline-frame branches beside the main computation column."""
    from visualizer.render import _inline_frame_draw_bounds

    for frame in graph.inline_frames:
        if frame.frame_id not in graph.side_effect_frame_ids:
            continue
        members = set(frame.node_indices)

        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        right_obstacles = [
            _node_content_left(pos)
            for index, pos in enumerate(positions)
            if index not in members
            and pos.spec.synthetic not in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}
            and pos.bottom < bounds.top
            and pos.top_y > bounds.bottom
            and _node_content_left(pos) > bounds.right
        ]
        if not right_obstacles:
            continue
        desired_right = min(right_obstacles) - MIN_HORIZONTAL_BLOCK_GAP
        shift = desired_right - bounds.right
        if shift <= 0:
            continue
        for index in members:
            positions[index].cx += shift


def realign_fanout_branch_columns(
    positions: list[LayoutPosition],
    graph: ComputationGraph | None = None,
) -> None:
    """Restore fan-out column alignment after row-wise overlap nudges."""
    if not _fanout_branch_node_groups(positions):
        return
    _align_fanout_branch_columns(positions, graph)
    _resolve_branch_column_overlaps(
        positions,
        min_gap=MIN_HORIZONTAL_BLOCK_GAP,
        graph=graph,
    )
    _align_fanout_branch_columns(positions, graph)
    if graph is not None:
        _center_align_vertical_chains(positions, graph)


def _frame_border_gap(
    upper: int,
    lower: int,
    frame_member_sets: list[set[int]],
) -> float:
    """Row gap two tiles need when a frame's border runs between them.

    Members pack tighter than a layer gap, but a border and its padding still have
    to fit in the space between the tile a frame encloses and the tile it leaves
    out, or the dotted rectangle lands on that tile.
    """
    for members in frame_member_sets:
        if (upper in members) != (lower in members):
            return INLINE_FRAME_PAD + FRAME_BORDER_CLEARANCE
    return 0.0


def _dominant_frame_column(
    positions: list[LayoutPosition],
    indices: Sequence[int],
    *,
    tol: float = 0.01,
) -> float:
    """The column most of a frame's tiles already sit on.

    Averaging instead lets a tile that a later pass holds off-column drag the whole
    frame the same step sideways on every pass, so the stacking never settles.
    """
    groups: list[list[int]] = []
    for index in indices:
        cx = positions[index].cx
        for group in groups:
            if abs(positions[group[0]].cx - cx) <= tol:
                group.append(index)
                break
        else:
            groups.append([index])
    widest = max(groups, key=len)
    if sum(1 for group in groups if len(group) == len(widest)) > 1:
        # No column holds a majority, so there is nothing to anchor on but the middle.
        return sum(positions[index].cx for index in indices) / len(indices)
    return sum(positions[index].cx for index in widest) / len(widest)


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

        # An operation-DAG frame keeps a side column offset left of its spine, so averaging
        # it in would drag the whole frame left a little further on every pass.
        side_nodes = _operation_dag_side_nodes(graph, frame)
        column_indices = [index for index in indices if index not in side_nodes]
        # With the whole chain on the offset arm there is no spine here to centre on.
        # Recentring anyway would adopt the offset column as the frame's own, and the
        # offset would then be applied to it again on every pass.
        frame_cx = _dominant_frame_column(positions, column_indices) if column_indices else None
        frame_gap = _inline_frame_vertical_gap(graph, frame)
        if min_gap is not None:
            frame_gap = max(frame_gap, min_gap)
        cursor_top = max(positions[index].top_y for index in indices)
        frame_members = set(frame.node_indices)
        internal_outgoing = {
            index: sum(
                source == index and target in frame_members
                for source, target in graph.links
            )
            for index in indices
        }
        internal_incoming = {
            index: sum(
                target == index and source in frame_members
                for source, target in graph.links
            )
            for index in indices
        }
        required_row_gap = _row_gap_rules(graph)
        for position_in_chain, index in enumerate(indices):
            pos = positions[index]
            if frame_cx is not None:
                pos.cx = frame_cx
            pos.top_y = cursor_top
            feeds_join = any(
                source == index
                and target in frame_members
                and internal_incoming[target] > 1
                for source, target in graph.links
            )
            next_gap = (
                max(frame_gap, DETAIL_LAYER_GAP + 0.04)
                if internal_outgoing[index] > 1 or feeds_join
                else frame_gap
            )
            if position_in_chain + 1 < len(indices):
                next_gap = max(
                    next_gap,
                    required_row_gap(index, indices[position_in_chain + 1]),
                )
            cursor_top -= pos.height + next_gap
        _layout_operation_dag_frame(positions, graph, frame)


def _align_k_proj_adjacent_to_chunk_pipeline(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    max_cx_gap: float | None = None,
) -> None:
    """Keep the chunk_kda pipeline handoff near the k_proj fan-out column."""
    chunk_indices = [
        index
        for index, spec in enumerate(graph.nodes)
        if spec.label == "chunk_kda pipeline"
    ]
    k_indices = [
        index
        for index, spec in enumerate(graph.nodes)
        if spec.block is not None and spec.block.attr_name == "k_proj"
    ]
    if not chunk_indices or not k_indices:
        return
    chunk_index = chunk_indices[0]
    k_index = k_indices[0]
    chunk_pos = positions[chunk_index]
    k_pos = positions[k_index]
    gap_limit = (
        max_cx_gap
        if max_cx_gap is not None
        else MIN_HORIZONTAL_BLOCK_GAP * 12
    )
    cx_gap = abs(chunk_pos.cx - k_pos.cx)
    if cx_gap <= gap_limit + 1e-6:
        return
    if k_pos.cx <= chunk_pos.cx:
        return
    chunk_pos.cx = k_pos.cx - gap_limit


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
    _align_k_proj_adjacent_to_chunk_pipeline(positions, graph)

    if min_left is None or not positions:
        return
    content_left = min(_node_content_left(pos) for pos in positions)
    shift = min_left - content_left
    if abs(shift) <= 1e-6:
        return
    for pos in positions:
        pos.cx += shift
    stack_inline_frame_positions(positions, graph)


def clear_merge_feeder_columns(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Align routed-expert output with its aggregation spine."""
    for source, target in graph.links:
        source_pos = positions[source]
        target_pos = positions[target]
        if (
            "routed_expert" in graph.nodes[source].key
            and graph.nodes[target].label == "MoE aggregation"
        ):
            source_pos.cx = target_pos.cx


def _inline_frame_internal_pairs(graph: ComputationGraph) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for frame in graph.inline_frames:
        chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
        pairs.update(zip(chain, chain[1:]))
    return pairs


def _open_sideways_feed_corridors(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    max_passes: int = 8,
) -> None:
    """Reserve the row gap a feed needs when it has to reach its target sideways.

    A feed whose source column misses its target's top edge cannot drop straight in: it
    leaves its source, crosses the gap between the two rows, and turns down onto the
    target. That run needs the whole row-gap band, but the band only gets reserved along
    the chain, so a step parked beside the chain can end up with nowhere to route.
    """
    from visualizer.text_measure import box_bounds_at

    required_row_gap = _row_gap_rules(graph)
    for _ in range(max_passes):
        opened = False
        for source, target in graph.links:
            upper = box_bounds_at(
                positions[source].cx,
                positions[source].top_y,
                positions[source].width,
                positions[source].height,
            )
            lower = box_bounds_at(
                positions[target].cx,
                positions[target].top_y,
                positions[target].width,
                positions[target].height,
            )
            if lower.top >= upper.bottom:
                continue
            if lower.left <= positions[source].cx <= lower.right:
                continue
            deficit = required_row_gap(source, target) - (upper.bottom - lower.top)
            if deficit <= 1e-9:
                continue
            _shift_node_subtree(positions, graph, target, deficit)
            opened = True
        if not opened:
            return


def _open_fan_in_approach_rows(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    max_passes: int = 8,
) -> None:
    """Reserve one approach row per feed above a tile several connectors converge on.

    When the tile directly above covers a target's whole top edge, every other feed has to
    come down outside the pair and turn in the row between them to reach a port. One row
    only fits one such turn, so a crowded target needs a deeper row above it than the chain
    reserves; without the depth those feeds have no choice but to double up on one run.
    """
    from visualizer.text_measure import box_bounds_at

    required_row_gap = _row_gap_rules(graph)
    for _ in range(max_passes):
        opened = False
        for target in range(len(graph.nodes)):
            lower = box_bounds_at(
                positions[target].cx,
                positions[target].top_y,
                positions[target].width,
                positions[target].height,
            )
            lids = [
                source
                for source, tgt in graph.links
                if tgt == target
                and positions[source].top_y - positions[source].height >= lower.top
                and lower.left <= positions[source].cx <= lower.right
            ]
            if not lids:
                continue
            lid = min(lids, key=lambda index: positions[index].top_y - positions[index].height)
            deficit = required_row_gap(lid, target) - (
                positions[lid].top_y - positions[lid].height - lower.top
            )
            if deficit <= 1e-9:
                continue
            _shift_node_subtree(positions, graph, target, deficit)
            opened = True
        if not opened:
            return


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
    from visualizer.render import (
        CONNECTOR_OBSTACLE_MARGIN,
        PARALLEL_CONNECTOR_CHANNEL_GAP,
        _inline_frame_draw_bounds,
    )

    gap = DETAIL_LAYER_GAP if merge_gap is None else merge_gap
    incoming = _build_incoming_links(graph, node_count=len(positions))
    framed = inline_frame_member_indices(graph)

    for target, sources in incoming.items():
        if only_targets is not None and target not in only_targets:
            continue
        if len(sources) < 2:
            continue
        # Inline-frame stacking already places its merge rows. Shifting a framed
        # target's whole downstream subtree here is undone for frame members on
        # the next stack pass while leaking into nodes below the frame.
        if target in framed:
            continue
        deepest_bottom = min(positions[source].bottom for source in sources)
        target_top = deepest_bottom - gap
        # A multi-input target below an inline frame needs a horizontal merge
        # channel between its top edge and the dotted frame border.
        for frame in graph.inline_frames:
            if not any(source in frame.node_indices for source in sources):
                continue
            bounds = _inline_frame_draw_bounds(frame, positions, graph)
            target_top = min(
                target_top,
                bounds.bottom
                - PARALLEL_CONNECTOR_CHANNEL_GAP
                - CONNECTOR_OBSTACLE_MARGIN,
            )
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
        for band_columns in _group_frame_columns_by_vertical_band(positions, frame_columns):
            _sort_frame_columns_for_packing(graph, positions, band_columns, pad=frame_pad)
            cursor_left: float | None = None
            prev_indices: list[int] | None = None
            for indices in band_columns:
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
        _resolve_horizontal_overlaps(
            positions,
            _topological_layers(graph),
            min_gap=gap,
            graph=graph,
        )

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
        _align_input_over_single_target(positions, graph)

    if min_left is not None:
        _align_positions_left(positions, min_left)


def _outermost_inline_frames(graph: ComputationGraph) -> list:
    """Frames that no other frame encloses; nested frames ride along with their parent."""
    member_sets = [(frame, frozenset(frame.node_indices)) for frame in graph.inline_frames]
    outermost = []
    seen: set[frozenset[int]] = set()
    for frame, members in member_sets:
        if not members or members in seen:
            continue
        if any(members < other_members for _other, other_members in member_sets):
            continue
        seen.add(members)
        outermost.append(frame)
    return outermost


def _inline_frame_column_exit_count(graph: ComputationGraph, indices: list[int]) -> int:
    """How many tiles below a frame column its members feed.

    A frame that both continues the chain and skips ahead to a later tile has to sit
    in the chain's own column: routing that skip from a column off to the side would
    make it cross the wires feeding the tile in between.
    """
    members = set(indices)
    return len({tgt for src, tgt in graph.links if src in members and tgt not in members})


def _sort_frame_columns_for_packing(
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    columns: list[list[int]],
    *,
    pad: float,
) -> None:
    """Order frame columns left to right, letting a frame claim the chain's column."""
    columns.sort(
        key=lambda indices: (
            -_inline_frame_column_exit_count(graph, indices),
            _inline_frame_column_bounds(graph, positions, indices, pad=pad)[0],
        )
    )


def _group_frame_columns_by_vertical_band(
    positions: list[LayoutPosition],
    frame_columns: list[list[int]],
) -> list[list[list[int]]]:
    """Group frame columns that share a vertical band, so only those pack side by side.

    Frames that follow one another down the spine are not competing for horizontal
    space; packing them into neighbouring columns would break the spine into steps.
    """
    groups: list[list] = []
    for indices in sorted(
        frame_columns,
        key=lambda column: -max(positions[index].top_y for index in column),
    ):
        top = max(positions[index].top_y for index in indices)
        bottom = min(positions[index].top_y - positions[index].height for index in indices)
        for group in groups:
            if bottom < group[1] and group[0] < top:
                group[0] = min(group[0], bottom)
                group[1] = max(group[1], top)
                group[2].append(indices)
                break
        else:
            groups.append([bottom, top, [indices]])
    return [group[2] for group in groups]


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
    for frame in _outermost_inline_frames(graph):
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
        grouped_columns = _group_frame_columns_by_vertical_band(positions, frame_columns)

    for columns in grouped_columns:
        _sort_frame_columns_for_packing(graph, positions, columns, pad=pad)
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

    if _graph_has_tensor_ports(graph):
        _separate_overlapping_inline_frame_draw_bounds(positions, graph, min_gap=gap)


def _separate_overlapping_inline_frame_draw_bounds(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float,
) -> None:
    """Push inline-frame columns apart when expanded draw bounds overlap across rows."""
    from visualizer.render import INLINE_FRAME_LABEL_CHAR_W, _inline_frame_draw_bounds

    def caption_reserve(frame) -> float:
        label = frame.label.strip()
        if not label:
            return 0.0
        return min(len(label) * INLINE_FRAME_LABEL_CHAR_W, 0.55)

    frames = sorted(
        graph.inline_frames,
        key=lambda frame: _inline_frame_draw_bounds(frame, positions, graph).left,
    )
    for left_index in range(len(frames) - 1):
        left_frame = frames[left_index]
        left_bounds = _inline_frame_draw_bounds(left_frame, positions, graph)
        right_bounds = _inline_frame_draw_bounds(frames[left_index + 1], positions, graph)
        required_gap = min_gap + caption_reserve(left_frame)
        overlap = left_bounds.right + required_gap - right_bounds.left
        if overlap <= 0:
            continue
        for shift_index in range(left_index + 1, len(frames)):
            for node_index in frames[shift_index].node_indices:
                positions[node_index].cx += overlap
    _align_tensor_port_columns(positions, graph)


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
        if spec.label.strip() in {"×", "x", "*", "⨉"}:
            mult_w, _ = box_label_size(
                ax,
                "Multiply",
                None,
                fontsize=title_fontsize,
                white_text_stroke_pad=not is_basic_op_tile(spec.block),
            )
            width = max(width, mult_w)
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
    """Every data dependency influences placement in the top-entry graph."""
    return list(graph.links)


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


def _delay_pass_through_nodes_to_their_consumer(
    layers: list[list[int]],
    graph: ComputationGraph,
) -> list[list[int]]:
    """Seat early-produced, late-consumed values beside their consumer."""
    if not graph.side_effect_frame_ids:
        return layers

    layer_of = {
        index: layer_index
        for layer_index, members in enumerate(layers)
        for index in members
    }
    predecessors: dict[int, list[int]] = {}
    successors: dict[int, list[int]] = {}
    for source, target in _layout_graph_links(graph):
        predecessors.setdefault(target, []).append(source)
        successors.setdefault(source, []).append(target)

    framed = {index for frame in graph.inline_frames for index in frame.node_indices}
    delayed: dict[int, int] = {}
    for index, spec in enumerate(graph.nodes):
        if index in framed or spec.synthetic is not None:
            continue
        if len(predecessors.get(index, ())) != 1 or len(successors.get(index, ())) != 1:
            continue
        producer_layer = layer_of[predecessors[index][0]]
        current_layer = layer_of[index]
        target_layer = layer_of[successors[index][0]] - 1
        if target_layer > max(producer_layer, current_layer):
            delayed[index] = target_layer

    if not delayed:
        return layers
    rebuilt = [[index for index in members if index not in delayed] for members in layers]
    for index, target_layer in delayed.items():
        rebuilt[target_layer].append(index)
    return [members for members in rebuilt if members]


def _ungrounded_branch_indices(
    graph: ComputationGraph,
    predecessors: list[list[int]],
) -> set[int]:
    """Nodes on a branch computed from a forward parameter instead of from the input.

    Nothing above such a branch anchors it, so ranking it by distance from the top
    would strand it at the start of the section rather than beside its consumer.
    """
    ungrounded = {
        index
        for index, spec in enumerate(graph.nodes)
        if not predecessors[index]
        and spec.block is not None
        and _reads_only_a_side_parameter(spec.block)
    }
    if not ungrounded:
        return ungrounded
    changed = True
    while changed:
        changed = False
        for index, sources in enumerate(predecessors):
            if index in ungrounded or not sources:
                continue
            if all(source in ungrounded for source in sources):
                ungrounded.add(index)
                changed = True
    return ungrounded


def _topological_layers(graph: ComputationGraph) -> list[list[int]]:
    """Group node indices into layers for tight vertical stacking."""
    node_count = len(graph.nodes)
    if node_count == 0:
        return []

    incoming = [0] * node_count
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    predecessors: list[list[int]] = [[] for _ in range(node_count)]
    for src, tgt in _layout_graph_links(graph):
        adjacency[src].append(tgt)
        predecessors[tgt].append(src)
        incoming[tgt] += 1

    layer = [0] * node_count
    order: list[int] = []
    queue = [index for index in range(node_count) if incoming[index] == 0]
    while queue:
        node = queue.pop(0)
        order.append(node)
        for tgt in adjacency[node]:
            layer[tgt] = max(layer[tgt], layer[node] + 1)
            incoming[tgt] -= 1
            if incoming[tgt] == 0:
                queue.append(tgt)

    # An unanchored branch is ranked from its consumer instead, so it sits directly
    # above the step that reads it rather than floating up beside the input.
    ungrounded = _ungrounded_branch_indices(graph, predecessors)
    for node in reversed(order):
        if node not in ungrounded or not adjacency[node]:
            continue
        layer[node] = min(layer[tgt] for tgt in adjacency[node]) - 1

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
        CONNECTOR_EXIT_STUB,
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
        min_bus_y = _pipeline_merge_min_bus_y(graph, positions, target)
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


def _feeder_bypasses_merge_target(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    target: int,
) -> bool:
    """True when a feeder in the target's own column carries on to a tile below it."""
    from visualizer.render import TOP_ENTRY_PORT_GAP

    target_pos = positions[target]
    target_bottom = target_pos.top_y - target_pos.height
    for source, tgt in graph.links:
        if tgt != target or abs(positions[source].cx - target_pos.cx) >= TOP_ENTRY_PORT_GAP:
            continue
        if any(
            below != target and positions[below].top_y <= target_bottom
            for other, below in graph.links
            if other == source
        ):
            return True
    return False


def _pipeline_merge_min_bus_y(
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    target: int,
) -> float:
    """Highest level a pipeline merge bus may sit at above its target."""
    from visualizer.render import (
        CONNECTOR_ATTACHED_BOX_MARGIN,
        CONNECTOR_OBSTACLE_MARGIN,
        SAME_COLUMN_BYPASS_CORRIDOR,
    )

    min_bus_y = (
        positions[target].top_y
        + CONNECTOR_OBSTACLE_MARGIN
        + CONNECTOR_ATTACHED_BOX_MARGIN
    )
    if _feeder_bypasses_merge_target(positions, graph, target):
        # That feeder's bypass tees between the merge bus and the tile below it.
        min_bus_y += SAME_COLUMN_BYPASS_CORRIDOR
    return min_bus_y


def _pipeline_merge_bus_y_for_layout(
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    target: int,
    tail_sources: list[int],
) -> float | None:
    """Shared merge-bus Y for inline-frame tails feeding one pipeline merge target."""
    from visualizer.render import (
        CONNECTOR_EXIT_STUB,
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
    return max(desired_bus_y, _pipeline_merge_min_bus_y(graph, positions, target))


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


def _top_entry_incoming_sources(
    graph: ComputationGraph,
    incoming: dict[int, list[int]],
    target: int,
) -> list[int]:
    """All sources enter a merge target from its top edge."""
    return list(incoming.get(target, []))


def _ensure_independent_merge_leg_corridors(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Reserve one deterministic horizontal channel per same-side merge leg.

    Independent operands may share a target but must not acquire a visual bus
    merely because their computed approach levels coincide. The outermost leg
    stays in the lowest corridor; each nearer leg gets one channel above it.
    """
    from visualizer.render import (
        CONNECTOR_ATTACHED_BOX_MARGIN,
        CONNECTOR_EXIT_STUB,
        CONNECTOR_OBSTACLE_MARGIN,
        PARALLEL_CONNECTOR_CHANNEL_GAP,
        TOP_ENTRY_PORT_GAP,
    )

    incoming = _build_incoming_links(graph, node_count=len(positions))
    target_clearance = CONNECTOR_OBSTACLE_MARGIN + CONNECTOR_ATTACHED_BOX_MARGIN
    for target, sources in incoming.items():
        if len(sources) < 2:
            continue
        target_pos = positions[target]
        max_top_y = target_pos.top_y
        constrained = False
        for side in (-1.0, 1.0):
            side_sources = [
                source
                for source in sources
                if side * (positions[source].cx - target_pos.cx) > TOP_ENTRY_PORT_GAP
            ]
            if len(side_sources) < 2:
                continue
            side_sources.sort(
                key=lambda source: -abs(positions[source].cx - target_pos.cx)
            )
            for channel, source in enumerate(side_sources):
                allowed_top_y = (
                    positions[source].bottom
                    - CONNECTOR_EXIT_STUB
                    - target_clearance
                    # Keep one channel in reserve for the route planner's base
                    # approach, then one more for each nested same-side leg.
                    - (channel + 1) * PARALLEL_CONNECTOR_CHANNEL_GAP
                )
                max_top_y = min(max_top_y, allowed_top_y)
                constrained = True
        if constrained and target_pos.top_y > max_top_y + 1e-6:
            _shift_node_subtree(
                positions,
                graph,
                target,
                target_pos.top_y - max_top_y,
            )


def _ensure_top_entry_clearance_below_inline_frames(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Leave corridor space below dotted inline frames for dual top-entry merge buses."""
    from visualizer.render import (
        CONNECTOR_ATTACHED_BOX_MARGIN,
        CONNECTOR_EXIT_STUB,
        CONNECTOR_OBSTACLE_MARGIN,
        PARALLEL_CONNECTOR_COORD_EPS,
        PIPELINE_MERGE_BUS_BELOW_FRAME_GAP,
        _inline_frame_draw_bounds,
    )

    incoming = _build_incoming_links(graph, node_count=len(positions))
    frame_tails = _inline_frame_tail_indices(graph)
    clearance = (
        CONNECTOR_EXIT_STUB
        + PIPELINE_MERGE_BUS_BELOW_FRAME_GAP
        + CONNECTOR_OBSTACLE_MARGIN
        + CONNECTOR_ATTACHED_BOX_MARGIN
    )

    for target in incoming:
        top_sources = _top_entry_incoming_sources(graph, incoming, target)
        if len(top_sources) < 2:
            continue
        # Only tails offset from the target need a corridor: their connector jogs
        # horizontally below the frame, whereas an aligned tail drops straight through.
        tail_sources = [
            source
            for source in top_sources
            if source in frame_tails
            and abs(positions[source].cx - positions[target].cx) > PARALLEL_CONNECTOR_COORD_EPS
        ]
        if not tail_sources:
            continue
        # A shared target must sit below the lowest of all feeder frames; using
        # the highest bottom leaves no corridor for a taller sibling frame.
        frame_bottom = float("inf")
        for source in tail_sources:
            frame = _inline_frame_for_tail_node(graph, source)
            if frame is None:
                continue
            bounds = _inline_frame_draw_bounds(frame, positions, graph)
            frame_bottom = min(frame_bottom, bounds.bottom)
        if frame_bottom == float("inf"):
            continue
        max_top_y = frame_bottom - clearance
        if positions[target].top_y <= max_top_y + 1e-6:
            continue
        _shift_node_subtree(
            positions,
            graph,
            target,
            positions[target].top_y - max_top_y,
        )


def _compact_parallel_feeder_frame_exit_stubs(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Shrinkwrap parallel feeder columns down to the shared merge-bus corridor."""
    from visualizer.ast_analyze import MOE_AGGREGATION_LABEL
    from visualizer.render import (
        CONNECTOR_ATTACHED_BOX_MARGIN,
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
        top_sources = _top_entry_incoming_sources(graph, incoming, target)
        if len(top_sources) < 2 or not tail_sources:
            continue
        if len(tail_sources) < 2 and graph.nodes[target].label != MOE_AGGREGATION_LABEL:
            continue

        bus_y = _pipeline_merge_bus_y_for_layout(graph, positions, target, tail_sources)
        if bus_y is None:
            continue

        min_bus_y = (
            positions[target].top_y
            + CONNECTOR_OBSTACLE_MARGIN
            + CONNECTOR_ATTACHED_BOX_MARGIN
        )
        min_exit_horiz = (
            min_bus_y + CONNECTOR_EXIT_STUB + PIPELINE_MERGE_BUS_BELOW_FRAME_GAP + corridor_eps
        )
        for _pass in range(8):
            shifted = False
            for source in tail_sources:
                max_tail_stub = (
                    0.55
                    if abs(positions[source].cx - positions[target].cx) > 0.08
                    else 0.45
                )
                exit_horiz_y = _frame_tail_exit_horiz_y(graph, positions, source)
                if exit_horiz_y is None:
                    continue
                shift = positions[source].bottom - (bus_y + max_tail_stub)
                if shift <= corridor_eps:
                    continue
                frame = _inline_frame_for_tail_node(graph, source)
                if frame is None:
                    continue
                max_shift = _max_frame_exit_downward_shift(
                    graph,
                    positions,
                    frame,
                    source,
                    min_gap=min_gap,
                )
                if abs(positions[source].cx - positions[target].cx) > 0.08:
                    post_exit = exit_horiz_y - shift
                    side_min_exit = min_bus_y + CONNECTOR_EXIT_STUB + corridor_eps
                    if post_exit < side_min_exit:
                        shift = min(shift, max(0.0, exit_horiz_y - side_min_exit))
                    elif post_exit < min_exit_horiz:
                        shift = min(shift, max(0.0, exit_horiz_y - min_exit_horiz))
                shift = min(shift, max_shift)
                if shift <= corridor_eps:
                    continue
                _shift_inline_frame_column_and_ports(positions, graph, frame, shift)
                shifted = True
            if not shifted:
                break
            bus_y = _pipeline_merge_bus_y_for_layout(graph, positions, target, tail_sources)
            if bus_y is None:
                break

        max_side_stub = 0.55
        side_sources = [
            source
            for source in tail_sources
            if abs(positions[source].cx - positions[target].cx) > 0.08
        ]
        if not side_sources:
            continue
        for _lift_pass in range(4):
            bus_y = _pipeline_merge_bus_y_for_layout(graph, positions, target, tail_sources)
            if bus_y is None:
                break
            longest_stub = max(
                positions[source].bottom - bus_y for source in side_sources
            )
            if longest_stub <= max_side_stub + corridor_eps:
                break
            lift = longest_stub - max_side_stub
            candidate_bus = bus_y + lift
            corridor_ok = True
            for source in tail_sources:
                exit_horiz_y = _frame_tail_exit_horiz_y(graph, positions, source)
                if exit_horiz_y is None:
                    continue
                max_bus = (
                    exit_horiz_y
                    - CONNECTOR_EXIT_STUB
                    - PIPELINE_MERGE_BUS_BELOW_FRAME_GAP
                )
                if candidate_bus > max_bus + 1e-6:
                    corridor_ok = False
                    break
            if not corridor_ok or lift <= corridor_eps:
                break
            _shift_node_subtree(positions, graph, target, -lift)


def _compact_fanout_branch_tail_spacing(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float | None = None,
    max_compaction: float = 1.0,
) -> None:
    """Shrinkwrap branch tail tiles (e.g. Pad) toward the Linear/RMSNorm stack above."""
    from visualizer.sizing import min_vertical_block_gap

    gap = min_vertical_block_gap() if min_gap is None else min_gap
    stack_labels = {"Linear", "RMSNorm"}

    for indices in _fanout_branch_node_groups(positions).values():
        chain = _ordered_inline_frame_chain(graph, list(indices))
        stack_indices = [index for index in chain if graph.nodes[index].label in stack_labels]
        tail_indices = [index for index in chain if index not in stack_indices]
        if not stack_indices or not tail_indices:
            continue
        if not all(graph.nodes[index].label == "Pad" for index in tail_indices):
            continue
        anchor_index = stack_indices[-1]
        anchor_bottom = positions[anchor_index].bottom
        tail_indices.sort(key=lambda index: positions[index].top_y, reverse=True)
        closest_tail = tail_indices[0]
        empty = anchor_bottom - positions[closest_tail].top_y
        if empty <= gap * 1.5:
            continue
        shift = min(empty * max_compaction, empty - gap)
        if shift <= 1e-4:
            continue
        for index in tail_indices:
            positions[index].top_y += shift


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


def _module_input_port_clearance_above_target(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    port_index: int,
    target_index: int,
    *,
    row_gap: float,
) -> float:
    """Minimum top_y for a modeling port sitting above its consumer column."""
    target_pos = positions[target_index]
    port_height = positions[port_index].height
    frame_id = _inline_frame_id_for_node(graph, target_index)
    if frame_id is not None:
        frame = next(item for item in graph.inline_frames if item.frame_id == frame_id)
        chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
        if chain and chain[-1] == target_index and chain[0] != target_index:
            return target_pos.top_y + row_gap + port_height
        if chain:
            frame_top = max(positions[node_index].top_y for node_index in chain)
            return (
                frame_top
                + INPUT_INLINE_FRAME_CAPTION_CLEARANCE
                + row_gap
                + port_height
            )
    return target_pos.top_y + row_gap + port_height


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
    frame_tails = _inline_frame_tail_indices(graph)
    docked_above: dict[int, list[int]] = {}
    for index, spec in enumerate(graph.nodes):
        if spec.synthetic != SYNTHETIC_TENSOR:
            continue
        targets = outgoing[index]
        if len(targets) != 1:
            continue
        target_index = targets[0]
        target_pos = positions[target_index]
        port_pos = positions[index]

        if _is_local_operation_port(spec):
            port_pos.cx = target_pos.cx
            port_pos.top_y = target_pos.top_y + row_gap + port_pos.height
            continue

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
        if _is_multiply_label(target_spec.label):
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
        docked_above.setdefault(target_index, []).append(index)

    _spread_ports_sharing_a_consumer(positions, docked_above)
    _align_module_input_ports_row(positions, graph, gap=row_gap)


def _spread_ports_sharing_a_consumer(
    positions: list[LayoutPosition],
    docked_above: dict[int, list[int]],
) -> None:
    """Fan a consumer's inputs across its width, since docking alone stacks them."""
    for target_index, port_indices in docked_above.items():
        if len(port_indices) < 2:
            continue
        span = sum(positions[index].width for index in port_indices) + MIN_HORIZONTAL_BLOCK_GAP * (
            len(port_indices) - 1
        )
        cursor = positions[target_index].cx - span / 2
        for port_index in port_indices:
            positions[port_index].cx = cursor + positions[port_index].width / 2
            cursor += positions[port_index].width + MIN_HORIZONTAL_BLOCK_GAP


def _align_module_input_ports_row(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    gap: float | None = None,
) -> None:
    """Keep modeling inputs on one horizontal row when each can stay above its consumer."""
    if not _graph_has_tensor_ports(graph):
        return

    row_gap = DETAIL_LAYER_GAP if gap is None else gap
    port_indices = [
        index
        for index, spec in enumerate(graph.nodes)
        if spec.synthetic == SYNTHETIC_TENSOR and not _is_local_operation_port(spec)
    ]
    if len(port_indices) < 2:
        return

    outgoing = _tensor_port_outgoing(graph)
    min_tops: list[float] = []
    for port_index in port_indices:
        targets = outgoing.get(port_index, [])
        if len(targets) != 1:
            continue
        min_tops.append(
            _module_input_port_clearance_above_target(
                positions,
                graph,
                port_index,
                targets[0],
                row_gap=row_gap,
            )
        )
    if not min_tops:
        return

    row_top = max(min_tops)
    for port_index in port_indices:
        if len(outgoing.get(port_index, [])) != 1:
            continue
        if _is_multiply_label(graph.nodes[outgoing[port_index][0]].label):
            continue
        frame_id = _inline_frame_id_for_node(graph, outgoing[port_index][0])
        if frame_id is not None:
            frame = next(item for item in graph.inline_frames if item.frame_id == frame_id)
            chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
            if chain and chain[-1] == outgoing[port_index][0] and chain[0] != outgoing[port_index][0]:
                continue
        positions[port_index].top_y = row_top


def _align_tensor_port_columns(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Place each modeling tensor port above the kernel step it feeds."""
    for source, target in graph.links:
        if graph.nodes[source].synthetic != SYNTHETIC_TENSOR:
            continue
        if _is_local_operation_port(graph.nodes[source]):
            continue
        if _is_multiply_label(graph.nodes[target].label):
            continue
        positions[source].cx = positions[target].cx

    port_indices = [
        index
        for index, spec in enumerate(graph.nodes)
        if spec.synthetic == SYNTHETIC_TENSOR and not _is_local_operation_port(spec)
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
        if _is_local_operation_port(graph.nodes[source]):
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


def _frame_head_entry_gap(
    graph: ComputationGraph,
    layers: list[list[int]],
    layer_index: int,
    *,
    min_gap: float,
) -> float:
    """Row gap needed where the next layer starts a captioned inline frame.

    A frame's caption occupies the band above its top border, so the row feeding the
    frame's first tile needs room for the caption and for the connector crossing it.
    """
    if layer_index + 1 >= len(layers):
        return 0.0
    entered = set(layers[layer_index + 1])
    current = set(layers[layer_index])
    starts_frame = any(
        frame.node_indices
        and frame.label.strip()
        and frame.node_indices[0] in entered
        and not current & set(frame.node_indices)
        for frame in graph.inline_frames
    )
    return min_gap + INPUT_INLINE_FRAME_CAPTION_CLEARANCE if starts_frame else 0.0


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
        if graph is not None:
            row_gap = max(
                row_gap,
                _frame_head_entry_gap(graph, layers, layer_index, min_gap=min_gap),
            )
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
    column_cx = (
        _left_aligned_column_cx(positions, layers, anchor_x, min_gap=min_gap, graph=graph)
        if align_left
        else None
    )
    for layer_indices in layers:
        _pack_ordered_layer_row(
            positions,
            layer_indices,
            anchor_x=anchor_x,
            align_left=align_left,
            min_gap=min_gap,
            graph=graph,
            column_cx=column_cx,
        )


def _left_aligned_column_cx(
    positions: list[LayoutPosition],
    layers: list[list[int]],
    anchor_x: float,
    *,
    min_gap: float,
    graph: ComputationGraph | None = None,
) -> float | None:
    """Shared centre for an unbranched column, so its spine runs straight.

    Left-aligning each row on its own centres every row at half its own width, slanting
    the connectors of a chain whose tiles differ in width. Only a column that branches
    nowhere can adopt one centre; elsewhere rows must stay free to align with the
    neighbours they feed.
    """
    widths: list[float] = []
    for layer_indices in layers:
        if not layer_indices:
            continue
        if graph is not None and graph.inline_frames:
            units = _layer_packing_units(graph, layer_indices)
            if len(units) != 1:
                return None
            widths.append(_packing_unit_width(graph, units[0], positions, min_gap=min_gap))
        elif len(layer_indices) != 1:
            return None
        else:
            widths.append(positions[layer_indices[0]].width)
    if not widths:
        return None
    return anchor_x + max(widths) / 2


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


def _layout_forward_successors(
    graph: ComputationGraph,
    index: int,
) -> list[int]:
    """Return forward-path successors, ignoring dashed feeds."""
    return [
        target
        for source, target in graph.links
        if source == index
        and (source, target) not in graph.dashed_links
    ]


def _has_straight_line_to_exit(graph: ComputationGraph, start: int) -> bool:
    """True when the forward path from start is a single chain to a sink node."""
    visited: set[int] = set()
    current = start
    while True:
        successors = _layout_forward_successors(graph, current)
        if not successors:
            return True
        if len(successors) != 1:
            return False
        next_index = successors[0]
        if next_index in visited:
            return False
        visited.add(current)
        current = next_index


def _forward_hops_to_exit(graph: ComputationGraph, start: int) -> int:
    """Count forward hops along a straight path; large when the path branches."""
    hops = 0
    visited: set[int] = set()
    current = start
    while True:
        successors = _layout_forward_successors(graph, current)
        if not successors:
            return hops
        if len(successors) != 1:
            return hops + 10_000
        next_index = successors[0]
        if next_index in visited:
            return hops + 10_000
        visited.add(current)
        current = next_index
        hops += 1


def _branch_merge_consumers(
    graph: ComputationGraph,
    branch_indices: list[int],
) -> set[int]:
    branch_set = set(branch_indices)
    consumers: set[int] = set()
    for index in branch_indices:
        for target in _layout_forward_successors(graph, index):
            if target not in branch_set:
                consumers.add(target)
    return consumers


def _consumer_leads_to_exit(graph: ComputationGraph, consumer: int) -> bool:
    """True when a merge consumer has exits or a straight-line path to the output."""
    if _has_straight_line_to_exit(graph, consumer):
        return True
    return bool(_layout_forward_successors(graph, consumer))


def _input_hidden_indices(graph: ComputationGraph) -> set[int]:
    return {
        index
        for index, spec in enumerate(graph.nodes)
        if spec.synthetic in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN}
    }


def _branch_feeds_exit_path(
    graph: ComputationGraph,
    branch_indices: list[int],
    *,
    input_indices: set[int],
) -> bool:
    """True when an input-fed branch reaches a consumer on a path to the output."""
    if not branch_indices or not input_indices:
        return False
    branch_set = set(branch_indices)
    fed_from_input = any(
        source in input_indices
        for index in branch_indices
        for source, target in graph.links
        if target == index and source not in branch_set
    )
    if not fed_from_input:
        return False
    consumers = _branch_merge_consumers(graph, branch_indices)
    return any(_consumer_leads_to_exit(graph, consumer) for consumer in consumers)


def _fanout_branch_outside_sort_key(
    graph: ComputationGraph,
    positions: list[LayoutPosition],
    branch_index: int,
    indices: list[int],
    *,
    input_indices: set[int],
) -> tuple[int, int, int, float, int]:
    """Sort key: exit feeders outside first; shorter / closer-to-output branches further out."""
    chain_len = len(indices)
    consumers = _branch_merge_consumers(graph, indices)
    feeds_exit = _branch_feeds_exit_path(graph, indices, input_indices=input_indices)
    if not feeds_exit or not consumers:
        return (1, chain_len, 0, 0.0, branch_index)
    exit_hops = min(_forward_hops_to_exit(graph, consumer) for consumer in consumers)
    tail_bottom = min(positions[index].bottom for index in indices)
    return (0, chain_len, exit_hops, tail_bottom, branch_index)


def _order_fanout_branch_positions(
    positions: list[LayoutPosition],
    graph: ComputationGraph | None = None,
) -> None:
    """Pack fan-out branch columns left-to-right, pushing exit feeders to the outside."""
    branch_groups = _fanout_branch_node_groups(positions)
    if len(branch_groups) < 2:
        return

    input_indices = _input_hidden_indices(graph) if graph is not None else set()
    use_exit_heuristic = graph is not None and input_indices
    branch_order = sorted(
        branch_groups,
        key=lambda branch_index: (
            _fanout_branch_outside_sort_key(
                graph,
                positions,
                branch_index,
                branch_groups[branch_index],
                input_indices=input_indices,
            )
            if use_exit_heuristic
            else (0, 0, 0, 0.0, branch_index)
        ),
    )

    columns: list[tuple[int, list[int], float, float]] = []
    for branch_index in branch_order:
        indices = branch_groups[branch_index]
        left, right = _branch_column_extent(graph, positions, indices)
        columns.append((branch_index, indices, left, right))

    cursor_left = min(left for _branch, _indices, left, _right in columns)
    for _branch_index, indices, left, right in columns:
        shift = cursor_left - left
        for index in indices:
            positions[index].cx += shift
        cursor_left += (right - left) + MIN_HORIZONTAL_BLOCK_GAP


def _main_output_spine_indices(
    graph: ComputationGraph,
    positions: list[LayoutPosition],
) -> set[int]:
    """Node indices on the straight main-path column from output back toward the merge."""
    branch_members: set[int] = set()
    for indices in _fanout_branch_node_groups(positions).values():
        branch_members.update(indices)

    incoming = _build_incoming_links(graph, node_count=len(positions))
    sinks = [
        index
        for index in range(len(positions))
        if index not in branch_members
        and not _layout_forward_successors(graph, index)
        and positions[index].spec.synthetic
        not in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN, SYNTHETIC_TENSOR}
    ]

    spine: set[int] = set()
    for sink in sinks:
        current = sink
        while True:
            if current in branch_members:
                break
            spec = positions[current].spec
            if spec.synthetic in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN, SYNTHETIC_TENSOR}:
                break
            predecessors = [
                source
                for source in incoming[current]
                if (source, current) not in graph.dashed_links
            ]
            if len(predecessors) >= 2:
                break
            spine.add(current)
            if len(predecessors) != 1:
                break
            predecessor = predecessors[0]
            if predecessor in branch_members:
                break
            current = predecessor
    return spine


def _compact_exit_feeder_branch_indices(
    graph: ComputationGraph,
    branch_groups: dict[int, list[int]],
    *,
    input_indices: set[int],
) -> set[int]:
    """Branches of one or two input-fed tiles that hand off toward the output path."""
    compact: set[int] = set()
    for indices in branch_groups.values():
        if len(indices) > 2:
            continue
        if not _branch_feeds_exit_path(graph, indices, input_indices=input_indices):
            continue
        compact.update(indices)
    return compact


def _ensure_exit_feeder_branches_left_of_spine(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float | None = None,
) -> None:
    """Keep compact input-fed exit feeders left of the main output spine column."""
    gap = MIN_HORIZONTAL_BLOCK_GAP if min_gap is None else min_gap
    branch_groups = _fanout_branch_node_groups(positions)
    if not branch_groups:
        return

    input_indices = _input_hidden_indices(graph)
    exit_feeder_nodes = _compact_exit_feeder_branch_indices(
        graph,
        branch_groups,
        input_indices=input_indices,
    )
    if not exit_feeder_nodes:
        return

    spine = _main_output_spine_indices(graph, positions)
    if not spine:
        return

    feeder_right = max(_node_content_right(positions[index]) for index in exit_feeder_nodes)
    spine_left = min(_node_content_left(positions[index]) for index in spine)
    shift = feeder_right + gap - spine_left
    if shift <= 1e-6:
        return
    for index in spine:
        positions[index].cx += shift


def _separate_parallel_merge_horiz_corridors(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float | None = None,
) -> None:
    """Separate parallel side-feeder horizontal exit segments after vertical shrinkwrap."""
    from visualizer.render import (
        PARALLEL_CONNECTOR_CHANNEL_GAP,
        PARALLEL_CONNECTOR_COORD_EPS,
        _frame_tail_exit_horiz_y,
    )

    channel_gap = PARALLEL_CONNECTOR_CHANNEL_GAP if min_gap is None else max(
        min_gap,
        PARALLEL_CONNECTOR_CHANNEL_GAP,
    )
    incoming = _build_incoming_links(graph, node_count=len(positions))
    frame_tails = _inline_frame_tail_indices(graph)

    for target, sources in incoming.items():
        tail_sources = [source for source in sources if source in frame_tails]
        side_sources = [
            source
            for source in tail_sources
            if abs(positions[source].cx - positions[target].cx) > 0.08
        ]
        if len(side_sources) < 2:
            continue

        levels: list[list[tuple[int, float]]] = []
        for source in side_sources:
            exit_y = _frame_tail_exit_horiz_y(graph, positions, source)
            if exit_y is None:
                continue
            for group in levels:
                if abs(group[0][1] - exit_y) <= PARALLEL_CONNECTOR_COORD_EPS:
                    group.append((source, exit_y))
                    break
            else:
                levels.append([(source, exit_y)])

        if len(levels) < 2:
            continue
        levels.sort(key=lambda group: group[0][1], reverse=True)
        for index in range(1, len(levels)):
            upper_y = levels[index - 1][0][1]
            lower_y = levels[index][0][1]
            if upper_y - lower_y >= channel_gap - PARALLEL_CONNECTOR_COORD_EPS:
                continue
            lift = channel_gap - (upper_y - lower_y)
            for source, exit_y in levels[index]:
                frame = _inline_frame_for_tail_node(graph, source)
                if frame is None:
                    continue
                max_shift = _max_frame_exit_downward_shift(
                    graph,
                    positions,
                    frame,
                    source,
                    min_gap=channel_gap / 2,
                )
                shift = min(lift, max_shift)
                if shift <= PARALLEL_CONNECTOR_COORD_EPS:
                    continue
                _shift_inline_frame_column_and_ports(positions, graph, frame, -shift)
                levels[index] = [(source, exit_y - shift) for source, exit_y in levels[index]]


def _branch_inline_frame_groups(
    graph: ComputationGraph | None,
    indices: list[int],
) -> tuple[list[int], list[tuple[InlineFrameSpec, list[int]]]]:
    """Split a fan-out branch into loose tiles and the inline frames it contains."""
    if graph is None or not graph.inline_frames:
        return list(indices), []
    index_set = set(indices)
    groups: list[tuple[InlineFrameSpec, list[int]]] = []
    grouped: set[int] = set()
    for frame in graph.inline_frames:
        members = [index for index in frame.node_indices if index in index_set]
        if len(members) < 2:
            continue
        if grouped.intersection(members):
            continue
        groups.append((frame, members))
        grouped.update(members)
    return [index for index in indices if index not in grouped], groups


def _branch_frame_spine_indices(
    graph: ComputationGraph,
    frame: InlineFrameSpec,
    members: list[int],
) -> list[int]:
    """Frame members on the main column, ignoring a deliberately offset side arm."""
    side_nodes = _operation_dag_side_nodes(graph, frame)
    spine = [index for index in members if index not in side_nodes]
    return spine or list(members)


def _branch_column_extent(
    graph: ComputationGraph | None,
    positions: list[LayoutPosition],
    indices: list[int],
) -> tuple[float, float]:
    """Horizontal footprint of a fan-out branch, including inline-frame borders."""
    left = min(_node_content_left(positions[index]) for index in indices)
    right = max(_node_content_right(positions[index]) for index in indices)
    _loose, frame_groups = _branch_inline_frame_groups(graph, indices)
    if not frame_groups or graph is None:
        return left, right

    from visualizer.render import _inline_frame_draw_bounds

    for frame, _members in frame_groups:
        bounds = _inline_frame_draw_bounds(frame, positions, graph)
        left = min(left, bounds.left)
        right = max(right, bounds.right)
    return left, right


def _align_fanout_branch_columns(
    positions: list[LayoutPosition],
    graph: ComputationGraph | None = None,
) -> None:
    """Give every node in a fan-out branch the same column center."""
    for indices in _fanout_branch_node_groups(positions).values():
        if len(indices) < 2:
            continue
        loose, frame_groups = _branch_inline_frame_groups(graph, indices)
        if not frame_groups or graph is None:
            column_cx = sum(positions[index].cx for index in indices) / len(indices)
            for index in indices:
                positions[index].cx = column_cx
            continue

        # An inline frame keeps its own internal columns, so collapsing every member
        # onto one centre understates the branch footprint: the frame border springs
        # back out on the next stacking pass, on top of the neighbouring column.
        anchors = loose or [
            index
            for frame, members in frame_groups
            for index in _branch_frame_spine_indices(graph, frame, members)
        ]
        column_cx = sum(positions[index].cx for index in anchors) / len(anchors)
        for index in loose:
            positions[index].cx = column_cx
        for frame, members in frame_groups:
            spine = _branch_frame_spine_indices(graph, frame, members)
            spine_cx = sum(positions[index].cx for index in spine) / len(spine)
            shift = column_cx - spine_cx
            if abs(shift) <= 1e-9:
                continue
            for index in members:
                positions[index].cx += shift


def _resolve_branch_column_overlaps(
    positions: list[LayoutPosition],
    *,
    min_gap: float,
    graph: ComputationGraph | None = None,
) -> None:
    """Separate fan-out branch columns without breaking intra-column alignment."""
    branch_groups = _fanout_branch_node_groups(positions)
    if len(branch_groups) < 2:
        return

    columns: list[tuple[list[int], float, float]] = []
    for branch_index in sorted(branch_groups):
        indices = branch_groups[branch_index]
        left, right = _branch_column_extent(graph, positions, indices)
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


def _enforce_jog_corridor_gaps(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    min_gap: float = DETAIL_LAYER_GAP,
    column_epsilon: float = 0.01,
    max_passes: int = 32,
) -> None:
    """Deepen rows where a side feed has no room to turn towards its merge.

    A feed coming from another column turns sideways in the row between the two tiles, so
    the row has to hold an exit stub plus the clearance the tiles it passes need. Vertical
    overlap resolution reserves that room only for tiles sharing a column, which can leave
    a diagonal neighbour packed tighter than its own connector can be drawn. Tiles inside a
    dotted frame keep their spacing: the frame sizes its own rows.
    """
    from visualizer.render import CONNECTOR_OBSTACLE_MARGIN

    band = _skip_crossing_band()
    framed = {index for frame in graph.inline_frames for index in frame.node_indices}
    fan_in: dict[int, int] = {}
    for _source, target in graph.links:
        fan_in[target] = fan_in.get(target, 0) + 1
    for _ in range(max_passes):
        changed = False
        for source, target in graph.links:
            if source >= len(positions) or target >= len(positions):
                continue
            if source in framed or target in framed:
                continue
            if fan_in.get(target, 0) < 2:
                continue
            upper = positions[source]
            lower = positions[target]
            if abs(upper.cx - lower.cx) <= column_epsilon:
                continue
            if lower.top_y > upper.bottom:
                continue
            if upper.bottom - lower.top_y >= band - 1e-9:
                continue
            if not _jog_row_is_blocked(
                positions,
                source=source,
                target=target,
                margin=CONNECTOR_OBSTACLE_MARGIN,
            ):
                continue
            lower.top_y = upper.bottom - band
            changed = True
        if _deepen_fan_in_approach_rows(positions, graph, framed=framed):
            changed = True
        if not changed:
            return
        _resolve_vertical_overlaps(positions, graph=graph, min_gap=min_gap)


def _deepen_fan_in_approach_rows(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
    *,
    framed: set[int],
) -> bool:
    """Give a crowded target one approach row per feed above it.

    Where the tile directly above covers a target's whole top edge, only that tile can drop
    straight in. Every other feed has to come down outside the pair and turn in the row
    between them, and a row only holds one such turn, so several feeds sharing a shallow row
    have to double up on the same run. Deepening the row is what lets each take its own.
    """
    required_row_gap = _row_gap_rules(graph)
    deepened = False
    for target in range(min(len(positions), len(graph.nodes))):
        if target in framed:
            continue
        lower = positions[target]
        lids = [
            source
            for source, tgt in graph.links
            if tgt == target
            and source < len(positions)
            and source not in framed
            and positions[source].bottom >= lower.top_y
            and lower.cx - lower.width / 2 <= positions[source].cx <= lower.cx + lower.width / 2
        ]
        if not lids:
            continue
        lid = min(lids, key=lambda index: positions[index].bottom)
        required = required_row_gap(lid, target)
        if positions[lid].bottom - lower.top_y >= required - 1e-9:
            continue
        lower.top_y = positions[lid].bottom - required
        deepened = True
    return deepened


def _jog_row_is_blocked(
    positions: list[LayoutPosition],
    *,
    source: int,
    target: int,
    margin: float,
) -> bool:
    """True when a third tile leaves the row between two tiles too narrow to jog through."""
    upper = positions[source]
    lower = positions[target]
    row_top = upper.bottom
    row_bottom = lower.top_y
    left = min(upper.cx, lower.cx)
    right = max(upper.cx, lower.cx)
    for index, pos in enumerate(positions):
        if index in {source, target}:
            continue
        if pos.cx + pos.width / 2 + margin <= left or pos.cx - pos.width / 2 - margin >= right:
            continue
        if pos.top_y + margin <= row_bottom or pos.bottom - margin >= row_top:
            continue
        return True
    return False


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
        _align_fanout_branch_columns(positions, graph)
        _resolve_branch_column_overlaps(
            positions, min_gap=min_horizontal_gap, graph=graph
        )
    else:
        _resolve_horizontal_overlaps(positions, layers, min_gap=min_horizontal_gap, graph=graph)
    _resolve_vertical_overlaps(positions, graph=graph, min_gap=min_vertical_gap)
    if branch_groups:
        _align_fanout_branch_columns(positions, graph)
        _resolve_branch_column_overlaps(
            positions, min_gap=min_horizontal_gap, graph=graph
        )
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
    layers = _delay_pass_through_nodes_to_their_consumer(layers, graph)
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
    _align_fanout_branch_columns(positions, graph)
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
    if not _graph_has_tensor_ports(graph):
        stack_fanout_branch_columns(positions, graph)
    _layout_fork_join_branches(positions, graph)
    from visualizer.render_validate import _resolve_same_row_tile_overlaps

    for _ in range(max(1, len(positions))):
        snapshot = [(pos.cx, pos.width, pos.top_y) for pos in positions]
        _resolve_same_row_tile_overlaps(
            positions,
            min_gap=MIN_HORIZONTAL_BLOCK_GAP,
            graph=graph if not _graph_has_tensor_ports(graph) else None,
        )
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
    clear_merge_feeder_columns(positions, graph)
    pack_input_fed_inline_frame_branches(positions, graph)
    _enforce_top_entry_vertical_order(positions, graph)
    _open_sideways_feed_corridors(positions, graph)
    _open_fan_in_approach_rows(positions, graph)

    return positions, graph.links


def _estimate_graph_height(graph: ComputationGraph) -> float:
    """Height the diagram needs, from stacked layers and from the layout itself.

    Stacking layers only prices the floors that are known before anything is placed. The
    passes that open feed corridors and approach rows deepen individual rows by amounts
    that depend on where the tiles land, so the stacked sum can come out under what the
    layout goes on to use. Laying the graph out reports that exactly, and it is free to
    ask: vertical placement does not depend on the block box it is given.
    """
    if not graph.nodes:
        return 2.0
    stacked = _stacked_layer_height(graph)
    positions, _ = layout_computation_graph(
        graph, cx=0.0, top_y=0.0, block_w=1.0, block_h=stacked
    )
    span = max(pos.top_y for pos in positions) - min(pos.bottom for pos in positions)
    return max(stacked, span + DETAIL_TOP_INSET + DETAIL_BOTTOM_INSET)


def _stacked_layer_height(graph: ComputationGraph) -> float:
    """Diagram height implied by stacking the topological layers with their row floors."""
    layers = _topological_layers(graph)
    heights = [_diagram_size_for_spec(spec)[1] for spec in graph.nodes]
    content = sum(max(heights[index] for index in layer) for layer in layers)
    required_row_gap = _row_gap_rules(graph)
    linked = set(graph.links)
    gaps = 0.0
    for layer_index in range(max(0, len(layers) - 1)):
        # The row floors are per row, not once for the whole graph: layout deepens every
        # row that has to carry an approach, so the estimate has to pay for each of them.
        gaps += max(
            _inline_frame_internal_gap(graph, layers, upper_layer_index=layer_index),
            _frame_head_entry_gap(
                graph,
                layers,
                layer_index,
                min_gap=min_vertical_block_gap(),
            ),
            max(
                (
                    required_row_gap(upper, lower)
                    for upper in layers[layer_index]
                    for lower in layers[layer_index + 1]
                    if (upper, lower) in linked
                ),
                default=0.0,
            ),
        )
    if graph.inline_frames and len(layers) > 1:
        gaps += INPUT_INLINE_FRAME_CAPTION_CLEARANCE
    if _find_fork_join_clusters(graph):
        from visualizer.render import (
            INLINE_FRAME_LABEL_GAP,
            INLINE_FRAME_LABEL_LINE_H,
            INLINE_FRAME_PAD,
        )

        gaps += len(_find_fork_join_clusters(graph)) * (
            INLINE_FRAME_PAD + INLINE_FRAME_LABEL_GAP + INLINE_FRAME_LABEL_LINE_H
        )
    if _graph_has_tensor_ports(graph) and len(layers) > 1:
        gaps += TENSOR_PORT_LAYER_EXTRA_GAP
    for frame in graph.inline_frames:
        chain = _ordered_inline_frame_chain(graph, list(frame.node_indices))
        if len(chain) < 2:
            continue
        frame_gap = _inline_frame_vertical_gap(graph, frame)
        stack_h = sum(heights[index] for index in chain) + frame_gap * (len(chain) - 1)
        frame_layers = [
            layer_index
            for layer_index, layer in enumerate(layers)
            if set(chain) & set(layer)
        ]
        counted = (
            max(max(heights[index] for index in layer) for layer in (layers[li] for li in frame_layers))
            if frame_layers
            else 0.0
        )
        content += max(0.0, stack_h - counted)
    total = content + gaps + DETAIL_TOP_INSET + DETAIL_BOTTOM_INSET
    if graph.inline_frames:
        total += min_vertical_block_gap() * len(graph.inline_frames)
    if len(layers) > 1:
        total += min_vertical_block_gap()
    return total


def _enforce_top_entry_vertical_order(
    positions: list[LayoutPosition],
    graph: ComputationGraph,
) -> None:
    """Place every target fully below all sources so top-entry routing is possible."""
    incoming: dict[int, list[int]] = {}
    for source, target in graph.links:
        if source < len(positions) and target < len(positions):
            incoming.setdefault(target, []).append(source)
    for layer in _topological_layers(graph)[1:]:
        required_shift = 0.0
        for target in layer:
            sources = incoming.get(target, [])
            if not sources:
                continue
            highest_allowed_top = min(
                positions[source].bottom - min_vertical_block_gap() for source in sources
            )
            required_shift = max(
                required_shift,
                positions[target].top_y - highest_allowed_top,
            )
        if required_shift > 0:
            for target in layer:
                positions[target].top_y -= required_shift


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


def _consumer_row_pairs(
    positions: list[LayoutPosition],
    graph: ComputationGraph | None,
) -> list[tuple[LayoutPosition, LayoutPosition]]:
    """Producer/consumer row pairs that a top-entry connector has to span downward."""
    if graph is None:
        return []
    count = len(positions)
    return [
        (positions[source], positions[target])
        for source, target in _layout_graph_links(graph)
        if source < count and target < count
    ]


def _seat_consumers_below_producers(
    pairs: Sequence[tuple[LayoutPosition, LayoutPosition]],
    *,
    min_gap: float,
) -> bool:
    """Drop any consumer that is not clear of its producer's bottom edge."""
    moved = False
    for producer, consumer in pairs:
        allowed_top = producer.bottom - min_gap
        if consumer.top_y > allowed_top:
            consumer.top_y = allowed_top
            moved = True
    return moved


def _resolve_vertical_overlaps(
    positions: list[LayoutPosition],
    *,
    graph: ComputationGraph | None = None,
    min_gap: float = DETAIL_LAYER_GAP,
    layer_y_epsilon: float = 1e-6,
    max_passes: int = 400,
) -> None:
    """Push lower nodes down when they overlap a higher node horizontally.

    Separating a pair can drop a producer past one of its own consumers, and a
    top-entry connector cannot climb back up to reach it, so consumers follow
    their producer down in the same relaxation.
    """
    consumer_pairs = _consumer_row_pairs(positions, graph)
    for _pass in range(max_passes):
        changed = False
        ordered = sorted(positions, key=lambda pos: pos.top_y, reverse=True)
        for above_index, above in enumerate(ordered):
            for below in ordered[above_index + 1 :]:
                if not _boxes_overlap_horizontally(above, below, min_gap=min_gap):
                    continue
                if abs(above.top_y - below.top_y) <= layer_y_epsilon:
                    below.top_y = above.bottom - min_gap
                    changed = True
                    continue
                allowed_top = above.bottom - min_gap
                if below.top_y > allowed_top:
                    below.top_y = allowed_top
                    changed = True
        if _seat_consumers_below_producers(consumer_pairs, min_gap=min_gap):
            changed = True
        if not changed:
            return
