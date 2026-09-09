###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build computation graphs from block trees."""

from __future__ import annotations

from dataclasses import dataclass, field

from TraceLens.ModelUtils.block_tree import (
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
    collect_function_steps,
    flatten_computation_segments,
    inline_block_frame_label,
    inline_block_frame_sublabel,
    inline_composite_steps,
    inline_wrapper_step_label,
    is_kernel_pipeline_tree,
    is_situ_gated_mlp,
    is_straight_line_module,
    is_transparent_inline_expansion,
    is_method_wrapper,
    wrapper_bullet_lines,
)
from TraceLens.ModelUtils.ast_analyze import (
    FORWARD_METHOD_INPUT,
    SYNTHETIC_ATTENTION,
    is_forward_operation,
)
from TraceLens.ModelUtils.basic_ops import BasicOpFilter, keep_detail_graph_node

SYNTHETIC_INPUT = "@input"
SYNTHETIC_OUTPUT = "@output"
SYNTHETIC_LOOP_CARRIED = "@loop_carried"
SYNTHETIC_HIDDEN = (
    "@hidden_states"  # legacy alias; replaced by SYNTHETIC_INPUT in graphs
)
SYNTHETIC_TENSOR = "@tensor"


def _maybe_inline(
    step: BlockNode,
    *,
    basic_ops: BasicOpFilter | None = None,
    inline_expansion: bool = True,
) -> tuple[list[BlockNode], BlockNode | None]:
    """Conditionally inline composite steps; returns ``([step], None)`` when disabled."""
    if not inline_expansion:
        return [step], None
    return inline_composite_steps(step, basic_ops=basic_ops)


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
    transparent: bool = False


@dataclass
class ComputationGraph:
    """Directed graph built from a block tree."""

    nodes: list[GraphNodeSpec] = field(default_factory=list)
    links: list[tuple[int, int]] = field(default_factory=list)
    link_port_labels: dict[tuple[int, int], str] = field(default_factory=dict)
    link_output_ports: dict[tuple[int, int], str] = field(default_factory=dict)
    inline_frames: list[InlineFrameSpec] = field(default_factory=list)
    side_effect_frame_ids: set[str] = field(default_factory=set)
    excluded_output_indices: set[int] = field(default_factory=set)
    primary_output_index: int | None = None
    output_node_index: int | None = None
    output_ports: dict[str, int] = field(default_factory=dict)
    primary_output_port: str | None = None
    loop_carried_nodes: dict[str, int] = field(default_factory=dict)
    attr_output_indices: dict[str, int] = field(default_factory=dict)
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


def _track_attr_index(
    attr_last_index: dict[str, int], attr_name: str, index: int
) -> None:
    attr_last_index[attr_name] = index


def _rebuild_attr_last_index(graph: ComputationGraph) -> dict[str, int]:
    attr_last_index: dict[str, int] = dict(graph.attr_output_indices)
    for index, spec in enumerate(graph.nodes):
        if spec.block is not None:
            _track_attr_index(attr_last_index, spec.block.attr_name, index)
    return attr_last_index


def _normalize_param_name(name: str) -> str:
    """Normalize a parameter name for fuzzy matching.

    Handles differences between call-site variable names (e.g. ``topk_indices``)
    and forward-method parameter names (e.g. ``top_k_index``).
    """
    return name.replace("indices", "index").replace("_", "").rstrip("s")


def _build_module_param_entries(
    graph: ComputationGraph,
) -> dict[str, dict[str, int]]:
    """Map each expanded module to ``{param_name: first_graph_index}``.

    Uses inline frames (keyed by module attr) and ``param_inputs`` on
    individual pipeline steps to find arg-specific entry points.
    """
    result: dict[str, dict[str, int]] = {}
    for frame in graph.inline_frames:
        entries: dict[str, int] = {}
        for index in frame.node_indices:
            block = graph.nodes[index].block
            if block is None:
                continue
            for param in block.param_inputs:
                entries.setdefault(param, index)
        if entries:
            result[frame.frame_id] = entries
    return result


def _resolve_primary_input(
    consumer_attr: str,
    root: "BlockNode",
    attr_last_index: dict[str, int],
    input_index: int | None,
    last_index: int | None,
) -> int | None:
    """Find the graph index for the consumer's primary (non-side) input.

    Inspects ``forward_step_predecessor_args`` to find which predecessor
    provides the primary input (the one that maps to @method_input inside
    the expanded pipeline), and resolves it via *attr_last_index*.
    Falls back to *last_index* → *input_index* when data is unavailable.
    """
    arg_map = root.forward_step_predecessor_args.get(consumer_attr)
    if not arg_map:
        return last_index if last_index is not None else input_index

    # Build the set of param names that have dedicated pipeline entry points.
    # The primary input is the arg NOT in this set.
    from TraceLens.ModelUtils.block_tree import BlockNode as _BN

    child = next(
        (c for c in root.children if c.attr_name == consumer_attr), None
    )
    if child is None:
        return last_index if last_index is not None else input_index

    side_params: set[str] = set()
    for gc in child.children:
        side_params.update(gc.param_inputs)

    for arg_name, pred in arg_map.items():
        if _normalize_param_name(arg_name) in {
            _normalize_param_name(p) for p in side_params
        }:
            continue
        # This is the primary (non-side) input.
        if pred == FORWARD_METHOD_INPUT:
            return input_index
        resolved = attr_last_index.get(pred)
        if resolved is not None:
            return resolved

    return last_index if last_index is not None else input_index


def _resolve_return_slot_source(
    producer: "BlockNode",
    arg_name: str,
    attr_last_index: dict[str, int],
    default: int,
) -> int:
    """When *producer* is multi-return, find the graph index of the specific
    return slot that matches *arg_name* (with normalized fallback)."""
    normalized = _normalize_param_name(arg_name)
    for slot_name, producer_attr in producer.forward_return_slots.items():
        if _normalize_param_name(slot_name) == normalized:
            resolved = attr_last_index.get(producer_attr)
            if resolved is not None:
                return resolved
    return default


def _lookup_param_entry(
    param_entries: dict[str, int],
    arg_name: str,
    default: int,
) -> int:
    """Look up an arg-specific pipeline entry, with fuzzy fallback."""
    exact = param_entries.get(arg_name)
    if exact is not None:
        return exact
    normalized = _normalize_param_name(arg_name)
    for param, index in param_entries.items():
        if _normalize_param_name(param) == normalized:
            return index
    return default


_KERNEL_CLASS_NAMES = frozenset(
    {"AttentionOp", "KernelOp", "AttentionMerge"}
)

SYNTHETIC_KERNEL_PORT = "@kernel_port"
SYNTHETIC_KERNEL_PORT_IN = "@kernel_port_in"
SYNTHETIC_KERNEL_PORT_OUT = "@kernel_port_out"


def _kernel_input_names(spec: NodeSpec) -> list[str]:
    """Extract declared input names from a kernel block's ``inputs:`` detail."""
    if spec.block is None:
        return []
    for detail in spec.block.details:
        if detail.startswith("inputs:"):
            raw = detail.split(":", 1)[1].strip()
            return [name.strip() for name in raw.split(",") if name.strip()]
    return []


def _add_kernel_port_nodes(graph: ComputationGraph) -> None:
    """Insert port nodes for every input on kernel tiles.

    For every kernel (attention/GPU kernel) node, create a port node per
    incoming edge and rewire:
    ``source → kernel`` becomes ``source → port_node(label) → kernel``.

    Labeled edges use their explicit label.  Unlabeled edges are matched
    against the kernel's declared ``inputs:`` list: any input name not
    already claimed by a labeled edge is assigned in order.
    """
    kernel_indices = [
        index
        for index, spec in enumerate(graph.nodes)
        if spec.block is not None
        and spec.block.class_name in _KERNEL_CLASS_NAMES
    ]
    for kernel_index in kernel_indices:
        kernel_spec = graph.nodes[kernel_index]
        declared_names = _kernel_input_names(kernel_spec)

        # First pass: collect labeled and unlabeled edges.
        labeled: list[tuple[int, str]] = []
        unlabeled_sources: list[int] = []
        for source, target in graph.links:
            if target != kernel_index:
                continue
            label = graph.link_port_labels.get((source, target))
            if label:
                labeled.append((source, label))
            else:
                unlabeled_sources.append(source)

        # Determine which declared names are already used by labeled edges.
        # Compound labels like "query/key/value" represent a bundled tensor
        # and do NOT claim the individual names (those go to separate inputs).
        used_names: set[str] = set()
        for _, lbl in labeled:
            if "/" not in lbl:
                used_names.add(lbl.strip().lower())

        # Remaining declared names, in declaration order, for unlabeled edges.
        remaining = [
            name for name in declared_names if name.lower() not in used_names
        ]

        all_inputs: list[tuple[int, str]] = []
        seen_labels: dict[str, int] = {}

        for source, label in labeled:
            count = seen_labels.get(label, 0)
            seen_labels[label] = count + 1
            if count > 0:
                label = f"{label}_{count + 1}"
            all_inputs.append((source, label))

        for idx, source in enumerate(unlabeled_sources):
            if idx < len(remaining):
                label = remaining[idx]
            else:
                src_spec = graph.nodes[source]
                label = src_spec.label or f"input_{len(all_inputs)}"
            count = seen_labels.get(label, 0)
            seen_labels[label] = count + 1
            if count > 0:
                label = f"{label}_{count + 1}"
            all_inputs.append((source, label))

        if all_inputs:
            remove_in: set[tuple[int, int]] = {
                (src, kernel_index) for src, _ in all_inputs
            }
            graph.links = [lk for lk in graph.links if lk not in remove_in]

            for source, label in all_inputs:
                graph.link_port_labels.pop((source, kernel_index), None)
                safe_label = label.replace("/", "_")
                port_index = _add_node(
                    graph,
                    key=f"@kernel_in:{kernel_index}:{safe_label}",
                    label=label,
                    synthetic=SYNTHETIC_KERNEL_PORT_IN,
                )
                graph.links.append((source, port_index))
                graph.links.append((port_index, kernel_index))



def _add_kernel_output_port_nodes(graph: ComputationGraph) -> None:
    """Insert port nodes for every output on kernel tiles with ≥2 outgoing edges."""
    kernel_indices = [
        index
        for index, spec in enumerate(graph.nodes)
        if spec.block is not None
        and spec.block.class_name in _KERNEL_CLASS_NAMES
    ]
    for kernel_index in kernel_indices:
        all_outputs: list[tuple[int, str]] = []
        seen_labels: dict[str, int] = {}
        for source, target in graph.links:
            if source != kernel_index:
                continue
            label = graph.link_port_labels.get((source, target))
            if not label:
                tgt_spec = graph.nodes[target]
                label = tgt_spec.label or f"output_{len(all_outputs)}"
            count = seen_labels.get(label, 0)
            seen_labels[label] = count + 1
            if count > 0:
                label = f"{label}_{count + 1}"
            all_outputs.append((target, label))

        if len(all_outputs) >= 2:
            remove_out: set[tuple[int, int]] = {
                (kernel_index, tgt) for tgt, _ in all_outputs
            }
            graph.links = [lk for lk in graph.links if lk not in remove_out]

            for target, label in all_outputs:
                graph.link_port_labels.pop((kernel_index, target), None)
                graph.link_output_ports.pop((kernel_index, target), None)
                safe_label = label.replace("/", "_")
                port_index = _add_node(
                    graph,
                    key=f"@kernel_out:{kernel_index}:{safe_label}",
                    label=label,
                    synthetic=SYNTHETIC_KERNEL_PORT_OUT,
                )
                graph.links.append((kernel_index, port_index))
                graph.links.append((port_index, target))
                for port_name, src in list(graph.output_ports.items()):
                    if src == kernel_index and port_name == label:
                        graph.output_ports[port_name] = port_index


def _wire_all_predecessor_edges(
    graph: ComputationGraph,
    root: BlockNode,
    *,
    input_index: int | None = None,
    skip_forward_links: bool = False,
) -> None:
    """Uniform predecessor-based edge wiring for all node types.

    Consolidates the four former post-hoc wiring passes into one entry point:
      1. Inline-op predecessor edges  (was ``_wire_operation_predecessor_links``)
      2. Attention provenance edges   (was ``_wire_attention_provenance_links``)
      3. Loop frames                  (structural, interleaved for ordering)
      4. Multi-input op forward links (was ``_wire_multi_input_op_forward_links``)
      5. Loop-carried nodes           (structural, interleaved for ordering)
      6. Inline-frame dangling outputs(was ``_wire_inline_frame_dangling_outputs``)
    """
    attr_last_index = _rebuild_attr_last_index(graph)

    # --- 1. Inline-op predecessor edges ---
    last_forward_order = max(
        (child.forward_order or 0 for child in root.children), default=0
    )
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
        multi_input = len(child.operation_predecessors) >= 2
        for pred in child.operation_predecessors:
            if pred == FORWARD_METHOD_INPUT:
                source_index = input_index
            else:
                source_index = attr_last_index.get(pred)
            if source_index is None:
                continue
            link = (source_index, target_index)
            if link not in graph.links:
                graph.links.append(link)
            if multi_input and link not in graph.link_port_labels:
                source_label = (
                    graph.nodes[source_index].label
                    if source_index < len(graph.nodes)
                    else None
                )
                if source_label:
                    graph.link_port_labels[link] = source_label
        if len(module_preds) >= 2 and (child.forward_order or 0) < last_forward_order:
            graph.excluded_output_indices.add(target_index)

    # --- 1b. Module-call predecessor edges from forward_step_predecessors ---
    if root.forward_step_predecessors:
        steps_by_attr = _forward_steps_by_attr(root)
        pred_arg_maps = root.forward_step_predecessor_args

        # Build per-module param entry indices from inline frames so that
        # side-fed arguments land on the correct expanded pipeline node.
        module_param_entries = _build_module_param_entries(graph)

        for step_attr, preds in root.forward_step_predecessors.items():
            step_node = steps_by_attr.get(step_attr)
            if step_node is None:
                continue
            default_target = _first_graph_index_for_module(
                step_node, attr_last_index
            )
            if default_target is None:
                default_target = attr_last_index.get(step_attr)
            if default_target is None:
                continue
            arg_map = pred_arg_maps.get(step_attr, {})
            param_entries = module_param_entries.get(step_attr, {})
            multi = len(preds) >= 2

            # Build (pred, arg_name) pairs.  When an arg_map is available,
            # iterate over its entries so that two args from the same
            # predecessor produce two separate edges (e.g. topk_indices and
            # topk_weights both sourced from gate).
            if arg_map:
                pairs: list[tuple[str, str | None]] = [
                    (src, name) for name, src in arg_map.items()
                ]
            else:
                pairs = [(pred, None) for pred in preds]

            for pred, arg_name in pairs:
                if pred == FORWARD_METHOD_INPUT:
                    source_index = input_index
                else:
                    source_index = attr_last_index.get(pred)
                if source_index is None:
                    continue
                # When the predecessor is a multi-return module, resolve the
                # arg name to the specific return-slot producer so the edge
                # starts from the correct pipeline node.
                if arg_name and source_index is not None:
                    pred_node = steps_by_attr.get(pred)
                    if pred_node is not None and pred_node.forward_return_slots:
                        source_index = _resolve_return_slot_source(
                            pred_node,
                            arg_name,
                            attr_last_index,
                            source_index,
                        )
                # Resolve arg-specific target when the predecessor maps to a
                # named parameter with its own pipeline entry point.
                target_index = (
                    _lookup_param_entry(param_entries, arg_name, default_target)
                    if arg_name
                    else default_target
                )
                link = (source_index, target_index)
                if link not in graph.links:
                    graph.links.append(link)
                if multi and link not in graph.link_port_labels and arg_name:
                    graph.link_port_labels[link] = arg_name

    # --- 2. Attention provenance edges ---
    if root.attention_inputs:
        targets = [
            index
            for index, spec in enumerate(graph.nodes)
            if spec.block is not None
            and spec.block.attr_name == SYNTHETIC_ATTENTION
        ]
        for target_index in targets:
            # If the kernel declares its own ``inputs:`` list, its edges
            # are already wired by normal predecessor tracking; skip all
            # provenance edges for declared ports.
            kernel_declared = {
                n.lower() for n in _kernel_input_names(graph.nodes[target_index])
            }

            ports_by_source: dict[int, list[str]] = {}
            for port, chain in root.attention_inputs.items():
                if kernel_declared and port.lower() in kernel_declared:
                    continue

                # Follow the provenance chain (actual data-flow from AST
                # analysis) to find the last graph node in the chain.
                source_index = next(
                    (
                        attr_last_index[attr]
                        for attr in reversed(chain)
                        if attr in attr_last_index
                    ),
                    None,
                )
                if source_index is None or source_index == target_index:
                    continue
                ports_by_source.setdefault(source_index, []).append(port)
            for source_index, ports in ports_by_source.items():
                link = (source_index, target_index)
                if link not in graph.links:
                    graph.links.append(link)
                graph.link_port_labels[link] = "/".join(ports)

    # --- 2b. Kernel input/output port nodes ---
    _add_kernel_port_nodes(graph)

    # --- 3. Loop frames (structural pass, must precede forward links) ---
    _add_loop_frames(graph)

    # --- 4. Multi-input op forward links ---
    if not skip_forward_links:
        _wire_multi_input_op_forward_links(graph, root, attr_last_index)

    # --- 5. Loop-carried nodes (must precede inline-frame pass) ---
    _add_loop_carried_nodes(graph, root)

    # --- 6. Inline-frame dangling outputs ---
    if not skip_forward_links:
        _wire_inline_frame_dangling_outputs(graph)


def _wire_multi_input_op_forward_links(
    graph: ComputationGraph,
    root: BlockNode,
    attr_last_index: dict[str, int],
) -> None:
    """Connect multi-input ops to the next forward step once all operands are wired."""
    steps_by_attr = _forward_steps_by_attr(root)
    ordered_steps = sorted(
        root.children,
        key=lambda step: (step.forward_order or 0, step.attr_name),
    )

    live_returns = root.referenced_return_producers
    for step in ordered_steps:
        if not step.operation_predecessors:
            continue
        if step.attr_name in live_returns:
            continue
        source_index = attr_last_index.get(step.attr_name)
        if source_index is None or _node_has_outgoing_links(graph, source_index):
            continue

        pred_orders = [
            steps_by_attr[pred].forward_order or 0
            for pred in step.operation_predecessors
            if pred in steps_by_attr
        ]
        if not pred_orders:
            continue

        min_consumer_order = max(pred_orders) + 1
        consumer = next(
            (
                candidate
                for candidate in ordered_steps
                if (candidate.forward_order or 0) >= min_consumer_order
                and candidate.attr_name != step.attr_name
            ),
            None,
        )
        if consumer is None:
            continue
        named = consumer.operation_predecessors
        if named and step.attr_name not in named:
            continue

        target_index = _first_graph_index_for_module(consumer, attr_last_index)
        if target_index is None:
            target_index = attr_last_index.get(consumer.attr_name)
        if target_index is not None and (source_index, target_index) not in graph.links:
            graph.links.append((source_index, target_index))


def _inline_frame_exit_index(
    graph: ComputationGraph, member_indices: set[int]
) -> int | None:
    exit_candidates = [
        source
        for source, target in graph.links
        if source in member_indices and target not in member_indices
    ]
    if exit_candidates:
        return exit_candidates[-1]

    sources_inside = {
        source for source, _target in graph.links if source in member_indices
    }
    dangling = [index for index in member_indices if index not in sources_inside]
    return dangling[-1] if dangling else None


def _wire_inline_frame_dangling_outputs(graph: ComputationGraph) -> None:
    """No-op: previously connected dead-end inline-frame nodes to the frame
    exit, but this fabricated edges not present in the model.  Dead-end nodes
    now remain unconnected, reflecting the actual data flow."""


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
    inline_expansion: bool = True,
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

        expanded_steps, wrapper = _maybe_inline(step, basic_ops=basic_ops, inline_expansion=inline_expansion)
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
                inline_expansion=inline_expansion,
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


def _add_forward_param_inputs(graph: ComputationGraph, root: BlockNode) -> None:
    """Give each extra forward parameter its own boundary input.

    A forward like ``forward(hidden_states, gate)`` reads two tensors, but only the
    primary one arrives on the chain. The steps that open the other parameter's path
    read a name that no node produces, so without an input of its own that path
    starts from nothing and the parameter has nowhere to dock.
    """
    primary = _input_label_for(root)
    incoming_count: dict[int, int] = {}
    for _source, target in graph.links:
        incoming_count[target] = incoming_count.get(target, 0) + 1
    param_index: dict[str, int] = {}
    param_consumers: set[str] = set()
    for index, spec in enumerate(list(graph.nodes)):
        block = spec.block
        if block is None or not is_forward_operation(block.attr_name):
            continue
        for param in block.param_inputs:
            # Nested expressions can repeat a parameter on each extracted operation
            # (one_hot(x).permute(...)). Dock it only at its first visible consumer.
            if (
                param == primary
                or param in param_consumers
                or block.boundary_input_name != param
                or param not in root.forward_param_inputs
                or incoming_count.get(index, 0) > len(block.operation_predecessors)
            ):
                continue
            param_consumers.add(param)
            source = param_index.get(param)
            if source is None:
                source = _add_node(
                    graph,
                    key=f"{SYNTHETIC_INPUT}:{param}",
                    label=param,
                    synthetic=SYNTHETIC_INPUT,
                )
                param_index[param] = source
            if (source, index) not in graph.links:
                graph.links.append((source, index))


def add_forward_output(
    graph: ComputationGraph,
    *,
    root: BlockNode | None = None,
    label: str = "Output",
) -> int | None:
    """Append the graph boundary Output with one named port per return value."""
    if graph.output_node_index is not None:
        return graph.output_node_index
    if not graph.nodes:
        return None
    if not any(spec.synthetic == SYNTHETIC_INPUT for spec in graph.nodes):
        return None
    attr_last_index = _rebuild_attr_last_index(graph)
    input_index = next(
        (
            index
            for index, spec in enumerate(graph.nodes)
            if spec.synthetic == SYNTHETIC_INPUT
        ),
        None,
    )
    ports: dict[str, int] = {}
    if root is not None and root.forward_return_slots:
        for slot in root.forward_return_order:
            producer = root.forward_return_slots.get(slot)
            if not producer:
                continue
            source = graph.loop_carried_nodes.get(producer)
            if source is None:
                source = (
                    input_index
                    if producer == FORWARD_METHOD_INPUT
                    else attr_last_index.get(producer)
                )
            if source is not None:
                ports[slot] = source

    source_indices = {src for src, _target in graph.links}
    exits = [
        index
        for index, spec in enumerate(graph.nodes)
        if index not in source_indices
        and spec.synthetic not in {SYNTHETIC_INPUT, SYNTHETIC_HIDDEN, SYNTHETIC_TENSOR}
    ]
    if not ports:
        framed_indices = {
            index for frame in graph.inline_frames for index in frame.node_indices
        }
        unframed_exits = [index for index in exits if index not in framed_indices]
        if unframed_exits:
            exits = unframed_exits
        if graph.primary_output_index is not None:
            exits = [graph.primary_output_index]
        else:
            exits = [
                index for index in exits if index not in graph.excluded_output_indices
            ]
        ports = {
            ("result" if len(exits) == 1 else f"result_{position + 1}"): index
            for position, index in enumerate(exits)
        }
    if not ports:
        return None
    output_index = _add_node(
        graph,
        key=SYNTHETIC_OUTPUT,
        label=label,
        synthetic=SYNTHETIC_OUTPUT,
    )
    for port, source in ports.items():
        graph.links.append((source, output_index))
        graph.link_port_labels[(source, output_index)] = port
    graph.output_node_index = output_index
    graph.output_ports = ports
    graph.primary_output_port = (
        root.primary_return_slot
        if root is not None and root.primary_return_slot in ports
        else next(iter(ports))
    )
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
        transparent=is_transparent_inline_expansion(wrapper),
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
    inline_expansion: bool = True,
) -> tuple[list[int], int | None]:
    """Expand a kernel pipeline in its own sub-frame and append the output kernel step."""
    if len(merge_steps) != 2:
        return _add_linear_pipeline_chain(
            graph,
            merge_steps,
            wrapper=merge_steps[0] if merge_steps else None,
            key_prefix=key_prefix,
            attr_last_index=attr_last_index,
            inline_expansion=inline_expansion,
        )

    pipeline_step, output_step = merge_steps
    inner_steps, pipeline_wrapper = _maybe_inline(pipeline_step, inline_expansion=inline_expansion)
    pipeline_indices, pipeline_tail = _add_linear_pipeline_chain(
        graph,
        inner_steps,
        wrapper=pipeline_wrapper,
        key_prefix=f"{key_prefix}:pipeline",
        attr_last_index=attr_last_index,
        inline_expansion=inline_expansion,
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
            _track_attr_index(
                attr_last_index, pipeline_wrapper.attr_name, pipeline_tail
            )
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
    inline_expansion: bool = True,
) -> tuple[list[int], int | None]:
    """Add a straight-line chain of nodes, optionally grouped in an inline frame."""
    if not steps:
        return [], last_index

    frame = (
        _start_inline_frame(graph, wrapper)
        if wrapper is not None and len(steps) > 1
        else None
    )
    indices: list[int] = []
    chain_last = last_index
    chain_input_index = last_index if last_index is not None else input_index

    for sub_index, sub_step in enumerate(steps):
        inner_steps, inner_wrapper = _maybe_inline(sub_step, inline_expansion=inline_expansion)
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
                inline_expansion=inline_expansion,
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
    from TraceLens.ModelUtils.block_tree import _situ_gated_mlp_parts

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
    inline_expansion: bool = True,
) -> int | None:
    """Add a side-path producer, inlining straight-line output gates when possible."""
    expanded_steps, wrapper = _maybe_inline(producer, basic_ops=basic_ops, inline_expansion=inline_expansion)
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
            inline_expansion=inline_expansion,
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
    inline_expansion: bool = True,
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
            inline_expansion=inline_expansion,
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
                inline_expansion=inline_expansion,
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


def _node_has_outgoing_links(graph: ComputationGraph, index: int) -> bool:
    return any(source == index for source, _target in graph.links)


def _forward_steps_by_attr(root: BlockNode) -> dict[str, BlockNode]:
    return {step.attr_name: step for step in root.children if step.attr_name}


def _first_graph_index_for_module(
    module: BlockNode,
    attr_last_index: dict[str, int],
) -> int | None:
    steps = collect_function_steps(module)
    if not steps:
        return attr_last_index.get(module.attr_name)
    first = min(steps, key=lambda step: step.forward_order or 0)
    return attr_last_index.get(first.attr_name)


def _add_loop_frames(graph: ComputationGraph) -> None:
    """Group contiguous loop-body operations without introducing graph cycles."""
    active_detail: str | None = None
    active_indices: list[int] = []

    def flush() -> None:
        nonlocal active_detail, active_indices
        if active_detail is not None and len(active_indices) >= 2:
            graph.inline_frames.append(
                InlineFrameSpec(
                    frame_id=f"loop:{graph.nodes[active_indices[0]].key}",
                    label=active_detail.replace("loop:", "Loop ·", 1).strip(),
                    node_indices=list(active_indices),
                )
            )
        active_detail = None
        active_indices = []

    for index, spec in enumerate(graph.nodes):
        loop_detail = next(
            (
                detail
                for detail in (spec.block.details if spec.block is not None else [])
                if detail.startswith("loop:")
            ),
            None,
        )
        if loop_detail != active_detail:
            flush()
            active_detail = loop_detail
        if loop_detail is not None:
            active_indices.append(index)
    flush()


def _collect_loop_carried(root: BlockNode) -> list:
    """Gather loop_carried specs from root and all inlined children."""
    from TraceLens.ModelUtils.ast_analyze import LoopCarriedSpec

    specs: list[LoopCarriedSpec] = list(root.loop_carried)
    for child in root.children:
        specs.extend(child.loop_carried)
    return specs


def _add_loop_carried_nodes(graph: ComputationGraph, root: BlockNode) -> None:
    """Materialize acyclic loop-result boundaries for values updated by a loop."""
    all_carried = _collect_loop_carried(root)
    if not all_carried:
        return
    attr_last_index = _rebuild_attr_last_index(graph)
    input_index = next(
        (
            index
            for index, spec in enumerate(graph.nodes)
            if spec.synthetic == SYNTHETIC_INPUT
        ),
        None,
    )
    for carried in all_carried:
        initial_index = (
            input_index
            if carried.initial_producer == FORWARD_METHOD_INPUT
            else attr_last_index.get(carried.initial_producer)
        )
        updated_index = attr_last_index.get(carried.updated_producer)
        if initial_index is None or updated_index is None:
            continue
        member_indices = {
            index
            for index, spec in enumerate(graph.nodes)
            if spec.block is not None and spec.block.attr_name in carried.operation_ids
        }
        matching_frames = [
            frame
            for frame in graph.inline_frames
            if member_indices and member_indices.issubset(set(frame.node_indices))
        ]
        loop_frame = (
            min(matching_frames, key=lambda f: len(f.node_indices))
            if matching_frames
            else None
        )
        iter_sublabel = (
            f"{carried.variable} · {carried.iteration_count} iterations"
            if carried.iteration_count is not None
            else f"{carried.variable} · repeated"
        )
        in_node_index = _add_node(
            graph,
            key=f"@loop_carried_in:{carried.loop_id}:{carried.variable}",
            label="Loop carried dependencies in",
            sublabel=iter_sublabel,
            synthetic=SYNTHETIC_LOOP_CARRIED,
        )
        out_node_index = _add_node(
            graph,
            key=f"@loop_carried_out:{carried.loop_id}:{carried.variable}",
            label="Loop carried dependencies out",
            sublabel=iter_sublabel,
            synthetic=SYNTHETIC_LOOP_CARRIED,
        )

        rewired: list[tuple[int, int]] = []
        in_feeds_member = False
        for source, target in graph.links:
            if source == updated_index and target not in member_indices:
                # Route outgoing edges from the loop's updated value through
                # the "out" boundary node.
                rewired.append((out_node_index, target))
                port = graph.link_port_labels.pop((source, target), None)
                if port:
                    graph.link_port_labels[(out_node_index, target)] = port
                output_port = graph.link_output_ports.pop((source, target), None)
                if output_port:
                    graph.link_output_ports[(out_node_index, target)] = output_port
            elif source == initial_index and target in member_indices:
                # Route the initial value into the loop body through the "in"
                # boundary node so the dependency is visible.
                rewired.append((in_node_index, target))
                port = graph.link_port_labels.pop((source, target), None)
                if port:
                    graph.link_port_labels[(in_node_index, target)] = port
                in_feeds_member = True
            else:
                rewired.append((source, target))
        graph.links = rewired
        graph.links.append((initial_index, in_node_index))
        graph.link_port_labels[(initial_index, in_node_index)] = "initial"
        graph.links.append((updated_index, out_node_index))
        graph.link_port_labels[(updated_index, out_node_index)] = "updated"
        graph.links.append((out_node_index, in_node_index))
        graph.link_port_labels[(out_node_index, in_node_index)] = "next iteration"
        graph.loop_carried_nodes[carried.updated_producer] = out_node_index
        if loop_frame is not None:
            loop_frame.node_indices.extend([in_node_index, out_node_index])


def _predecessor_map(graph: ComputationGraph) -> dict[int, list[int]]:
    preds: dict[int, list[int]] = {index: [] for index in range(len(graph.nodes))}
    for source, target in graph.links:
        preds[target].append(source)
    return preds


def _live_node_indices_to_fixpoint(
    graph: ComputationGraph,
    seeds: list[int],
) -> set[int]:
    """Backward close seeds under predecessors until the live set reaches a fixed point.

    Long operand chains and cyclic graphs both require repeated predecessor expansion.
    Conditional alternative links are materialized first so branch recurrences are included.
    """
    _add_conditional_alternative_links(graph)
    keep = set(seeds)
    changed = True
    while changed:
        changed = False
        preds = _predecessor_map(graph)
        for index in list(keep):
            for pred in preds[index]:
                if pred not in keep:
                    keep.add(pred)
                    changed = True
    return keep


def _dead_node_indices(
    graph: ComputationGraph,
    root: BlockNode,
    *,
    strip_unused_return_branches: bool,
) -> set[int]:
    """Nodes not on any path feeding kept return values."""
    if graph.primary_output_index is None or not root.primary_output_step:
        return set()
    if not strip_unused_return_branches or not root.multi_return_module:
        return set()

    seed_indices = [graph.primary_output_index]
    referenced_returns = root.referenced_return_producers
    if referenced_returns:
        seed_indices.extend(
            index
            for index, spec in enumerate(graph.nodes)
            if spec.block is not None
            and spec.block.attr_name in referenced_returns
            and index not in seed_indices
        )
        seed_indices.extend(
            index
            for producer, index in graph.loop_carried_nodes.items()
            if producer in referenced_returns and index not in seed_indices
        )
    keep = _live_node_indices_to_fixpoint(graph, seed_indices)

    dead: set[int] = set()
    for index, spec in enumerate(graph.nodes):
        if index in keep:
            continue
        if spec.synthetic in {SYNTHETIC_INPUT, SYNTHETIC_OUTPUT}:
            continue
        dead.add(index)
    return dead


def _apply_dead_code_elimination(
    graph: ComputationGraph,
    root: BlockNode,
    *,
    strip_unused_return_branches: bool,
) -> ComputationGraph:
    """Remove unreachable nodes, iterating until the graph reaches a fixed point."""
    if not strip_unused_return_branches or not root.multi_return_module:
        graph.dead_node_indices = set()
        return graph

    max_passes = max(len(graph.nodes), 1) + 1
    for _ in range(max_passes):
        dead = _dead_node_indices(
            graph,
            root,
            strip_unused_return_branches=True,
        )
        graph.dead_node_indices = dead
        if not dead:
            return graph
        pruned = _strip_dead_nodes(graph)
        if len(pruned.nodes) >= len(graph.nodes):
            return pruned
        graph = pruned

    graph.dead_node_indices = _dead_node_indices(
        graph,
        root,
        strip_unused_return_branches=True,
    )
    return graph


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
    bridged_output_ports: dict[tuple[int, int], str] = {}
    for source, target in graph.links:
        if source in remove_indices or target in remove_indices:
            continue
        bridged_links.add((source, target))
        port_label = graph.link_port_labels.get((source, target))
        if port_label:
            bridged_port_labels[(source, target)] = port_label
        output_port = graph.link_output_ports.get((source, target))
        if output_port:
            bridged_output_ports[(source, target)] = output_port
    for removed in remove_indices:
        for source in preds[removed]:
            for target in succs[removed]:
                port_label = graph.link_port_labels.get(
                    (source, removed)
                ) or graph.link_port_labels.get((removed, target))
                for kept_source in _expand_preds(source):
                    for kept_target in _expand_succs(target):
                        if kept_source == kept_target:
                            continue
                        bridged_links.add((kept_source, kept_target))
                        if (
                            port_label
                            and (kept_source, kept_target) not in bridged_port_labels
                        ):
                            bridged_port_labels[(kept_source, kept_target)] = port_label
                        output_port = graph.link_output_ports.get(
                            (source, removed)
                        ) or graph.link_output_ports.get((removed, target))
                        if (
                            output_port
                            and (kept_source, kept_target) not in bridged_output_ports
                        ):
                            bridged_output_ports[(kept_source, kept_target)] = (
                                output_port
                            )

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
            if (kept_source := _remap(source)) is not None
            and (kept_target := _remap(target)) is not None
        ],
        link_port_labels={
            (kept_source, kept_target): label
            for (source, target), label in bridged_port_labels.items()
            if (kept_source := _remap(source)) is not None
            and (kept_target := _remap(target)) is not None
        },
        link_output_ports={
            (kept_source, kept_target): port
            for (source, target), port in bridged_output_ports.items()
            if (kept_source := _remap(source)) is not None
            and (kept_target := _remap(target)) is not None
        },
        excluded_output_indices={
            _remap(index)
            for index in graph.excluded_output_indices
            if _remap(index) is not None
        },
        primary_output_index=_remap(graph.primary_output_index),
        output_node_index=_remap(graph.output_node_index),
        output_ports={
            port: kept
            for port, index in graph.output_ports.items()
            if (kept := _remap(index)) is not None
        },
        primary_output_port=graph.primary_output_port,
        loop_carried_nodes={
            producer: kept
            for producer, index in graph.loop_carried_nodes.items()
            if (kept := _remap(index)) is not None
        },
        attr_output_indices={
            attr: kept
            for attr, index in graph.attr_output_indices.items()
            if (kept := _remap(index)) is not None
        },
        dead_node_indices=set(),
    )

    for frame in graph.inline_frames:
        kept_indices = [
            _remap(index) for index in frame.node_indices if _remap(index) is not None
        ]
        if len(kept_indices) >= 2:
            filtered.inline_frames.append(
                InlineFrameSpec(
                    frame_id=frame.frame_id,
                    label=frame.label,
                    sublabel=frame.sublabel,
                    node_indices=kept_indices,
                    transparent=frame.transparent,
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


def _strip_dangling_leaves(
    graph: ComputationGraph,
    root: BlockNode | None = None,
) -> ComputationGraph:
    """Remove nodes with no outgoing edges that are not outputs or other sinks."""
    source_indices = {source for source, _target in graph.links}
    kept_sinks = {graph.primary_output_index, graph.output_node_index}
    kept_sinks.update(graph.output_ports.values())
    kept_sinks.update(graph.loop_carried_nodes.values())
    # Keep attr_output_indices only for attrs that are referenced as return producers.
    referenced = (
        root.referenced_return_producers if root is not None else set()
    )
    for attr, index in graph.attr_output_indices.items():
        if attr in referenced:
            kept_sinks.add(index)
    kept_sinks.discard(None)

    sink_synthetics = {SYNTHETIC_OUTPUT, SYNTHETIC_LOOP_CARRIED}
    # Never strip nodes inside inline frames — their internal topology must stay intact.
    framed_indices: set[int] = set()
    for frame in graph.inline_frames:
        framed_indices.update(frame.node_indices)
    # Keep nodes fed by framed nodes — stripping them would orphan the frame.
    fed_by_frame: set[int] = set()
    for src, tgt in graph.links:
        if src in framed_indices and tgt not in framed_indices:
            fed_by_frame.add(tgt)

    dangling: set[int] = set()
    for index, spec in enumerate(graph.nodes):
        if index in source_indices:
            continue
        if index in kept_sinks:
            continue
        if index in framed_indices:
            continue
        if index in fed_by_frame:
            continue
        if spec.synthetic in sink_synthetics:
            continue
        # Only strip non-synthetic leaves (real ops with no consumers).
        if spec.synthetic is not None:
            continue
        dangling.add(index)
    if not dangling:
        return graph
    return _prune_computation_nodes(graph, dangling)


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


def _tensor_port_input_sublabels(
    attention_inputs: dict[str, list[str]],
) -> dict[str, str]:
    """Format per-port upstream hints like ``← Linear`` from provenance chains."""
    labels: dict[str, str] = {}
    for port, chain in attention_inputs.items():
        if not chain:
            continue
        source = _producer_label_from_attr(chain[-1])
        head = source.split(" in ", 1)[0]
        labels[port] = f"← {head}"
    return labels


def _label_multi_input_kernel_edges(
    graph: ComputationGraph,
    step_entries: dict[str, int],
    step_indices: dict[str, int],
    tensor_links: list[tuple[int, int]],
) -> None:
    """Add port labels to incoming edges of kernel steps that have more than one input."""
    all_kernel_indices = set(step_entries.values()) | set(step_indices.values())
    inputs_per_target: dict[int, list[tuple[int, int]]] = {}
    for source, target in graph.links:
        if target in all_kernel_indices:
            inputs_per_target.setdefault(target, []).append((source, target))
    tensor_link_set = set(tensor_links)
    for target, edges in inputs_per_target.items():
        if len(edges) < 2:
            continue
        for source, tgt in edges:
            if (source, tgt) in graph.link_port_labels:
                continue
            source_spec = graph.nodes[source] if source < len(graph.nodes) else None
            if source_spec is None:
                continue
            if (source, tgt) in tensor_link_set:
                label = source_spec.label or ""
            else:
                label = source_spec.label or source_spec.key or ""
            if label:
                graph.link_port_labels[(source, tgt)] = label


def _add_tensor_ports_segment(
    graph: ComputationGraph,
    segment: TensorPortsSegment,
    *,
    key_prefix: str,
    port_sublabels: dict[str, str] | None = None,
    inline_expansion: bool = True,
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
                inline_expansion=inline_expansion,
            )
            step_attr_indices[step.attr_name] = attr_last_index
            step_indices[step.attr_name] = (
                sub_tail if sub_tail is not None else sub_indices[-1]
            )
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
    tensor_links: list[tuple[int, int]] = []
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
        tensor_links.append((port_index, target_index))
        for child_attr in input_operand_attrs.get(target_attr, []):
            child_index = step_attr_indices.get(target_attr, {}).get(child_attr)
            if child_index is not None:
                graph.links.append((port_index, child_index))
                tensor_links.append((port_index, child_index))

    _label_multi_input_kernel_edges(graph, step_entries, step_indices, tensor_links)

    return step_indices.get(segment.steps[-1].attr_name)


def _fanout_merge_key_prefix(merge: BlockNode, segment_index: int) -> str:
    """Stable node-id prefix for fan-out merge steps."""
    if merge.class_name == "KernelPipeline" and merge.attr_name:
        return merge.attr_name.lstrip("@")
    return f"merge:{segment_index}"


def build_computation_graph(
    root: BlockNode,
    *,
    prefix_steps: list[BlockNode] | None = None,
    include_input: bool = True,
    basic_ops: BasicOpFilter | None = None,
    strip_unused_return_branches: bool = False,
    inline_expansion: bool = True,
) -> ComputationGraph:
    """Convert a block tree into a directed acyclic computation graph."""
    graph = ComputationGraph()

    if root.is_basic or not root.children:
        input_index = _add_forward_input(graph, root) if include_input else None
        node_index = _add_node(graph, key=root.attr_name, block=root)
        if input_index is not None:
            graph.links.append((input_index, node_index))
            graph.primary_output_index = node_index
            add_forward_output(graph, root=root)
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
        graph.primary_output_index = last_index
        add_forward_output(graph, root=root)
        return graph

    for segment_index, segment in enumerate(segments):
        if isinstance(segment, TensorPortsSegment):
            tail = _add_tensor_ports_segment(
                graph,
                segment,
                key_prefix=f"{root.attr_name}:tensor{segment_index}",
                port_sublabels=_tensor_port_input_sublabels(root.attention_inputs),
                inline_expansion=inline_expansion,
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
                    inline_expansion=inline_expansion,
                )
                branch_specs.append(branch)
                if input_index is not None and first_index is not None:
                    _link_forward_input(graph, input_index, first_index)
                if tail is not None:
                    branch_tails.append(tail)
            merge_steps, merge_wrapper = _maybe_inline(
                segment.merge, basic_ops=basic_ops, inline_expansion=inline_expansion
            )
            merge_key_prefix = _fanout_merge_key_prefix(segment.merge, segment_index)
            if (
                merge_wrapper is not None
                and is_kernel_pipeline_tree(segment.merge)
                and segment.merge.tensor_input_labels
            ):
                provenance = (
                    segment.merge.attention_inputs or root.attention_inputs or {}
                )
                frame = _start_inline_frame(graph, merge_wrapper)
                start_index = len(graph.nodes)
                pipeline_tail = _add_tensor_ports_segment(
                    graph,
                    TensorPortsSegment(
                        labels=list(segment.merge.tensor_input_labels),
                        targets=dict(segment.merge.tensor_step_targets),
                        steps=list(segment.merge.children),
                    ),
                    key_prefix=merge_key_prefix,
                    port_sublabels=_tensor_port_input_sublabels(provenance),
                    inline_expansion=inline_expansion,
                )
                for index in range(start_index, len(graph.nodes)):
                    _append_inline_frame_node(frame, index)
                port_index_by_label = {
                    spec.label: index
                    for index, spec in enumerate(graph.nodes)
                    if spec.synthetic == SYNTHETIC_TENSOR
                    and spec.key.startswith(f"{merge_key_prefix}:tensor:")
                }
                for tail, branch in zip(branch_tails, branch_specs):
                    port_index = port_index_by_label.get(branch.port_label)
                    if port_index is not None:
                        graph.links.append((tail, port_index))
                        graph.link_port_labels[(tail, port_index)] = branch.port_label
                last_index = pipeline_tail
                if pipeline_tail is not None:
                    _track_attr_index(
                        attr_last_index, merge_wrapper.attr_name, pipeline_tail
                    )
                continue
            if (
                merge_wrapper is not None
                and len(merge_steps) == 2
                and merge_steps[1].class_name == "KernelOutput"
            ):
                merge_indices, merge_tail = _add_kernel_pipeline_merge_chain(
                    graph,
                    merge_steps,
                    key_prefix=merge_key_prefix,
                    attr_last_index=attr_last_index,
                    inline_expansion=inline_expansion,
                )
                merge_first = merge_indices[0] if merge_indices else None
                if merge_first is not None:
                    for tail in branch_tails:
                        graph.links.append((tail, merge_first))
                last_index = merge_tail
                if merge_tail is not None:
                    _track_attr_index(
                        attr_last_index, merge_wrapper.attr_name, merge_tail
                    )
            elif merge_wrapper is not None:
                merge_indices, merge_tail = _add_linear_pipeline_chain(
                    graph,
                    merge_steps,
                    wrapper=merge_wrapper,
                    key_prefix=merge_key_prefix,
                    attr_last_index=attr_last_index,
                    inline_expansion=inline_expansion,
                )
                merge_first = merge_indices[0] if merge_indices else None
                if merge_first is not None:
                    for tail in branch_tails:
                        graph.links.append((tail, merge_first))
                last_index = merge_tail
                if merge_tail is not None:
                    _track_attr_index(
                        attr_last_index, merge_wrapper.attr_name, merge_tail
                    )
            else:
                merge_index = _add_node(
                    graph,
                    key=merge_key_prefix,
                    block=segment.merge,
                )
                for branch_offset, tail in enumerate(branch_tails):
                    graph.links.append((tail, merge_index))
                last_index = merge_index
                _track_attr_index(attr_last_index, segment.merge.attr_name, merge_index)
            continue

        if isinstance(segment, SideCombineSegment):
            from TraceLens.ModelUtils.ast_analyze import (
                MOE_AGGREGATION_LABEL,
                combine_op_from_step_details,
            )

            if (
                combine_op_from_step_details(list(segment.consumer.details or []))
                == MOE_AGGREGATION_LABEL
            ):

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
                _track_attr_index(
                    attr_last_index, segment.consumer.attr_name, agg_index
                )
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
            _track_attr_index(
                attr_last_index, segment.consumer.attr_name, combine_index
            )
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
                expanded_steps, wrapper = _maybe_inline(
                    module, basic_ops=basic_ops, inline_expansion=inline_expansion
                )
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
                        inline_expansion=inline_expansion,
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

            entry_index: int | None = None
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
                expanded_steps, wrapper = _maybe_inline(
                    consumer,
                    basic_ops=basic_ops,
                    inline_expansion=inline_expansion,
                )
                if wrapper is not None:
                    # Resolve the primary input source from
                    # forward_step_predecessor_args so the pipeline's
                    # @method_input ops connect to the correct predecessor
                    # instead of the overall forward method input.
                    primary_input = _resolve_primary_input(
                        consumer.attr_name,
                        root,
                        attr_last_index,
                        input_index,
                        last_index,
                    )
                    chain_indices, chain_tail = _add_linear_pipeline_chain(
                        graph,
                        expanded_steps,
                        wrapper=wrapper,
                        key_prefix=f"sidefeed:{segment_index}:{consumer.attr_name}",
                        attr_last_index=attr_last_index,
                        port_label=port_label,
                        port_style="inline" if port_label else None,
                        input_index=input_index,
                        last_index=primary_input,
                        inline_expansion=inline_expansion,
                    )
                    entry_index = chain_indices[0] if chain_indices else None
                    consumer_index = chain_tail
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
                # Edge wiring deferred to _wire_all_predecessor_edges via
                # forward_step_predecessors; only track the node index here.
            # Materialize side-chain producer nodes so they exist in the
            # graph for the generic predecessor pass to wire up.
            for side in segment.sides:
                if side.source_kind == "forward_input":
                    continue
                if not side.source_chain:
                    continue
                _ensure_side_chain_tail_index(
                    graph,
                    segment,
                    side,
                    segment_index=segment_index,
                    input_index=input_index,
                    attr_last_index=attr_last_index,
                    root=root,
                    basic_ops=basic_ops,
                    inline_expansion=inline_expansion,
                )
            last_index = consumer_index
            _track_attr_index(attr_last_index, consumer.attr_name, consumer_index)
            continue

        if isinstance(segment, CombineSegment):
            after_nodes = list(segment.after)
            side = segment.side
            if is_method_wrapper(side):
                side_index = _add_method_wrapper_node(
                    graph,
                    side,
                    key=f"side:{segment_index}",
                )
                if input_index is not None and segment.side_source == "forward_input":
                    _link_forward_input(graph, input_index, side_index)
                _track_attr_index(attr_last_index, side.attr_name, side_index)
            else:
                expanded_side, side_wrapper = _maybe_inline(
                    side, basic_ops=basic_ops, inline_expansion=inline_expansion
                )
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
                        inline_expansion=inline_expansion,
                    )
                    side_index = side_tail
                    _track_attr_index(
                        attr_last_index, side_wrapper.attr_name, side_index
                    )
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
                    if (
                        input_index is not None
                        and segment.side_source == "forward_input"
                    ):
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
                inline_expansion=inline_expansion,
                basic_ops=basic_ops,
            )
            if first_after is not None:
                graph.links.append((mult_index, first_after))
            last_index = tail
            continue

        if isinstance(segment, SeqSegment):
            step = segment.step
            fork_from_input = (
                _should_fork_main_path_from_input(
                    segments,
                    segment_index,
                    last_index,
                    attr_last_index,
                )
                or step.attr_name in root.input_fed_steps
            )
            if is_method_wrapper(step):
                step_index = _add_method_wrapper_node(
                    graph,
                    step,
                    key=f"seq:{segment_index}:{step.attr_name}",
                )
                # Edge wiring deferred to _wire_all_predecessor_edges via
                # forward_step_predecessors; only track the node index here.
                last_index = step_index
                _track_attr_index(attr_last_index, step.attr_name, step_index)
                continue
            expanded_steps, wrapper = _maybe_inline(step, basic_ops=basic_ops, inline_expansion=inline_expansion)
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
                    inline_expansion=inline_expansion,
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
                explicit_sources = _operation_source_indices(
                    sub_step,
                    attr_last_index,
                    chain_input_index=input_index,
                )
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

    graph.attr_output_indices.update(attr_last_index)
    skip_fwd = strip_unused_return_branches and root.multi_return_module
    _wire_all_predecessor_edges(
        graph, root, input_index=input_index, skip_forward_links=skip_fwd,
    )
    if root.primary_output_step:
        for index, spec in enumerate(graph.nodes):
            if (
                spec.block is not None
                and spec.block.attr_name == root.primary_output_step
            ):
                graph.primary_output_index = graph.loop_carried_nodes.get(
                    root.primary_output_step, index
                )
                break
        else:
            graph.primary_output_index = last_index
    else:
        graph.primary_output_index = last_index
    if resolved_include_input:
        _add_forward_param_inputs(graph, root)
    graph = _apply_dead_code_elimination(
        graph,
        root,
        strip_unused_return_branches=strip_unused_return_branches,
    )
    graph = _strip_dangling_leaves(graph, root=root)
    add_forward_output(graph, root=root)
    _add_kernel_output_port_nodes(graph)
    if basic_ops is not None and basic_ops.basic_only:
        return _filter_graph_basic_only(graph)
    return graph
