"""Build computation graphs from block trees."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

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


@dataclass
class InlineFrameSpec:
    """Dotted frame around steps expanded inline from a linear composite sub-block."""

    frame_id: str
    label: str
    sublabel: str | None = None
    node_indices: list[int] = field(default_factory=list)


@dataclass
class ComputationGraph:
    """Directed graph built from a block tree."""

    nodes: list[GraphNodeSpec] = field(default_factory=list)
    links: list[tuple[int, int]] = field(default_factory=list)
    link_port_labels: dict[tuple[int, int], str] = field(default_factory=dict)
    inline_frames: list[InlineFrameSpec] = field(default_factory=list)
    side_effect_frame_ids: set[str] = field(default_factory=set)
    excluded_output_indices: set[int] = field(default_factory=set)
    primary_output_index: int | None = None
    dead_node_indices: set[int] = field(default_factory=set)


def _operation_tile_label(label: str) -> str:
    """Use ordinary operation names instead of symbolic combine glyphs."""
    return {
        "+": "Add",
        "×": "Multiply",
        "*": "Multiply",
        "ƒ": "Function",
    }.get(label, label)


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
    spec = GraphNodeSpec(
        key=key,
        block=block,
        label=display,
        sublabel=sublabel,
        port_label=port_label,
        port_style=port_style,
        synthetic=synthetic,
    )
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


def _rebuild_attr_last_index(graph: ComputationGraph) -> dict[str, int]:
    attr_last_index: dict[str, int] = {}
    for index, spec in enumerate(graph.nodes):
        if spec.block is not None:
            _track_attr_index(attr_last_index, spec.block.attr_name, index)
    return attr_last_index


def _wire_operation_predecessor_links(graph: ComputationGraph, root: BlockNode) -> None:
    """Attach operand edges for inline ops once every forward step has been materialized."""
    attr_last_index = _rebuild_attr_last_index(graph)
    last_forward_order = max((child.forward_order or 0 for child in root.children), default=0)
    for child in root.children:
        if not is_forward_operation(child.attr_name):
            continue
        if not child.operation_predecessors:
            continue
        target_index = attr_last_index.get(child.attr_name)
        if target_index is None:
            continue
        module_preds = [
            pred
            for pred in child.operation_predecessors
            if pred != FORWARD_METHOD_INPUT and not is_forward_operation(pred)
        ]
        for pred in child.operation_predecessors:
            if pred == FORWARD_METHOD_INPUT:
                continue
            source_index = attr_last_index.get(pred)
            if source_index is None:
                continue
            link = (source_index, target_index)
            if link not in graph.links:
                graph.links.append(link)
        if len(module_preds) >= 2 and (child.forward_order or 0) < last_forward_order:
            graph.excluded_output_indices.add(target_index)


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
    if graph.primary_output_index is not None:
        exits = [graph.primary_output_index]
    else:
        exits = [index for index in exits if index not in graph.excluded_output_indices]
    if not exits:
        return None
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
    link_input: bool = True,
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
            input_index=input_index if link_input else None,
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
    if link_input and input_index is not None:
        _link_forward_input(graph, input_index, source_index)
    _track_attr_index(attr_last_index, source_attr, source_index)
    return source_index


def _ensure_side_chain_tail_index(
    graph: ComputationGraph,
    segment: SideFeedSegment,
    side,
    *,
    segment_index: int,
    input_index: int | None,
    attr_last_index: dict[str, int],
    root: BlockNode,
    basic_ops: BasicOpFilter | None = None,
) -> int | None:
    """Materialize a prior-step side chain, preserving g_a → g_b style gate pipelines."""
    source_attr = side.source_chain[-1] if side.source_chain else None
    if source_attr is None:
        return None

    cached = attr_last_index.get(source_attr)
    if cached is not None:
        return cached

    chain = segment.side_producer_chains.get(source_attr)
    if not chain:
        producer = segment.side_producer_nodes.get(source_attr)
        if producer is None:
            return None
        return _add_side_producer_index(
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

    tail_index: int | None = None
    for step in chain:
        attr = step.attr_name
        existing = attr_last_index.get(attr)
        if existing is not None:
            tail_index = existing
            continue

        if tail_index is None:
            branch_from_input = attr in root.input_fed_steps or len(chain) == 1
            tail_index = _add_side_producer_index(
                graph,
                step,
                segment_index=segment_index,
                source_attr=attr,
                port_label=side.port_label if attr == source_attr else None,
                port_style="inline",
                input_index=input_index,
                attr_last_index=attr_last_index,
                basic_ops=basic_ops,
                link_input=branch_from_input,
            )
            continue

        step_index = _add_node(
            graph,
            key=f"sideproducer:{segment_index}:{attr}",
            block=step,
        )
        graph.links.append((tail_index, step_index))
        _track_attr_index(attr_last_index, attr, step_index)
        tail_index = step_index

    return tail_index


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


def _dead_node_indices(
    graph: ComputationGraph,
    root: BlockNode,
    *,
    strip_unused_return_branches: bool,
) -> set[int]:
    """Nodes not on any path feeding kept return values."""
    if graph.primary_output_index is None or not root.primary_output_step:
        return set()
    preds: dict[int, list[int]] = {index: [] for index in range(len(graph.nodes))}
    for source, target in graph.links:
        preds[target].append(source)

    keep: set[int] = set()
    seeds: list[int] = [graph.primary_output_index]
    if not strip_unused_return_branches:
        for producer in root.referenced_return_producers:
            for index, spec in enumerate(graph.nodes):
                if spec.block is not None and spec.block.attr_name == producer:
                    seeds.append(index)
                    break

    pending = list(dict.fromkeys(seeds))
    while pending:
        index = pending.pop()
        if index in keep:
            continue
        keep.add(index)
        pending.extend(preds[index])

    dead: set[int] = set()
    for index, spec in enumerate(graph.nodes):
        if index in keep:
            continue
        if spec.synthetic in {SYNTHETIC_INPUT, SYNTHETIC_OUTPUT}:
            continue
        dead.add(index)
    return dead


def _prune_computation_nodes(
    graph: ComputationGraph,
    remove_indices: set[int],
) -> ComputationGraph:
    """Drop selected nodes and bridge links across removed vertices."""
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
    for removed in remove_indices:
        for source in preds[removed]:
            for target in succs[removed]:
                port_label = graph.link_port_labels.get((source, removed)) or graph.link_port_labels.get(
                    (removed, target)
                )
                for kept_source in _expand_preds(source):
                    for kept_target in _expand_succs(target):
                        if kept_source == kept_target:
                            continue
                        bridged_links.add((kept_source, kept_target))
                        if port_label and (kept_source, kept_target) not in bridged_port_labels:
                            bridged_port_labels[(kept_source, kept_target)] = port_label

    old_to_new: dict[int, int] = {}
    new_nodes: list[GraphNodeSpec] = []
    for index, spec in enumerate(graph.nodes):
        if index in remove_indices:
            continue
        old_to_new[index] = len(new_nodes)
        new_nodes.append(spec)

    def _remap(index: int | None) -> int | None:
        if index is None:
            return None
        return old_to_new.get(index)

    filtered = ComputationGraph(
        nodes=new_nodes,
        links=[
            (kept_source, kept_target)
            for source, target in bridged_links
            if (kept_source := _remap(source)) is not None and (kept_target := _remap(target)) is not None
        ],
        link_port_labels={
            (kept_source, kept_target): label
            for (source, target), label in bridged_port_labels.items()
            if (kept_source := _remap(source)) is not None and (kept_target := _remap(target)) is not None
        },
        excluded_output_indices={_remap(index) for index in graph.excluded_output_indices if _remap(index) is not None},
        primary_output_index=_remap(graph.primary_output_index),
        dead_node_indices=set(),
    )

    for frame in graph.inline_frames:
        kept_indices = [_remap(index) for index in frame.node_indices if _remap(index) is not None]
        if len(kept_indices) >= 2:
            filtered.inline_frames.append(
                InlineFrameSpec(
                    frame_id=frame.frame_id,
                    label=frame.label,
                    sublabel=frame.sublabel,
                    node_indices=kept_indices,
                )
            )

    return filtered


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
    return _prune_computation_nodes(graph, remove_indices)


def _strip_dead_nodes(graph: ComputationGraph) -> ComputationGraph:
    """Remove branch tails that do not feed the primary return value."""
    return _prune_computation_nodes(graph, set(graph.dead_node_indices))


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
    strip_unused_return_branches: bool = False,
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
                    label="Gated RMSNorm",
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
                    source_index = _ensure_side_chain_tail_index(
                        graph,
                        segment,
                        side,
                        segment_index=segment_index,
                        input_index=input_index,
                        attr_last_index=attr_last_index,
                        root=root,
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
                source_index = _ensure_side_chain_tail_index(
                    graph,
                    segment,
                    side,
                    segment_index=segment_index,
                    input_index=input_index,
                    attr_last_index=attr_last_index,
                    root=root,
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

    _wire_operation_predecessor_links(graph, root)
    if root.primary_output_step:
        for index, spec in enumerate(graph.nodes):
            if spec.block is not None and spec.block.attr_name == root.primary_output_step:
                graph.primary_output_index = index
                break
        else:
            graph.primary_output_index = last_index
    else:
        graph.primary_output_index = last_index
    graph.dead_node_indices = _dead_node_indices(
        graph,
        root,
        strip_unused_return_branches=strip_unused_return_branches,
    )
    _add_conditional_alternative_links(graph)
    graph = _strip_dead_nodes(graph)
    if basic_ops is not None and basic_ops.basic_only:
        return _filter_graph_basic_only(graph)
    return graph

