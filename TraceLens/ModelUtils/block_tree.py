###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Recursive block trees for detailed architecture diagrams."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from typing import Literal

from TraceLens.ModelUtils.ast_analyze import (
    FORWARD_METHOD_INPUT,
    GATE_ACTIVATION_DETAIL_PREFIX,
    LAYOUT_ONLY_LABELS,
    SYNTHETIC_ATTENTION,
    SYNTHETIC_GATE_ACTIVATION,
    ClassStructure,
    ForwardOperation,
    LoopCarriedSpec,
    SideInputSpec,
    attention_kernel_details,
    attention_kernel_label,
    base_submodule_attr,
    displays_as_linear,
    effective_forward_calls,
    expand_conditional_block_components,
    function_display_label,
    functional_display_label,
    is_function_synthetic,
    is_functional_synthetic,
    is_forward_operation,
    is_positional_synthetic,
    operation_display_label,
    positional_display_label,
    _synthetic_call_function_name,
    is_kernel_pipeline_step,
    kernel_kwarg_ports,
    kernel_name_from_step_details,
    tensor_input_label_order,
    _build_components,
    _classify_role,
    _label_for,
)
from TraceLens.ModelUtils.basic_ops import (
    BasicOpFilter,
    introspect_is_modeling_operation,
    is_fused_silu_mul_class,
    resolve_is_basic,
)
from TraceLens.ModelUtils.blocks import (
    BlockComponent,
    input_sources_from_forward_sequence,
    upstream_input_sources,
)

_log = logging.getLogger(__name__)

_SKIP_INIT_CLASS_NAMES = frozenset({"Parameter", "Buffer", "getattr"})


def is_method_wrapper(node: BlockNode) -> bool:
    """True for forward steps that call a helper method with no submodule internals."""
    if node.children or node.attr_name == SYNTHETIC_ATTENTION:
        return False
    return bool(node.details) and node.details[0].startswith("method `")


def wrapper_bullet_lines(node: BlockNode) -> tuple[str, str]:
    """Return display label and method name for wrapped-module bullets."""
    attr = node.attr_name
    label = node.label.strip()
    if label in {"", attr, node.class_name}:
        label = attr.strip("_").replace("_", " ")
    else:
        label = label.lstrip("_")
    return label, attr


def wrapper_bullet(node: BlockNode) -> str:
    """Human-readable bullet text for a method wrapper."""
    label, attr = wrapper_bullet_lines(node)
    if is_method_wrapper(node):
        return f"{label} ({attr})"
    if label.replace(" ", "_").strip("_") == attr.strip("_"):
        return label
    return f"{label} ({attr})"


_SKIP_WRAPPER_COMMENT_ATTRS = frozenset({"tokenization", "embed_tokens"})

_FUNCTIONAL_CALL_DETAIL_RE = re.compile(
    r"(?i)^(?:F\.|torch\.nn\.functional\.)[\w.]+\(\.\.\.\)$"
)


def wrapper_skips_comment(node: BlockNode) -> bool:
    """True when the wrapped-module panel should omit the descriptive comment line."""
    attr = node.attr_name.lower()
    if attr in _SKIP_WRAPPER_COMMENT_ATTRS:
        return True
    if node.role == "attention" or attr in {
        "self_attn",
        "self_attention",
        "attn",
        "attention",
    }:
        return True
    if "residual" in attr:
        return True
    return any("residual" in detail.lower() for detail in node.details)


def block_purpose(node: BlockNode) -> str | None:
    """Generic one-line description of what a block computes."""
    if node.class_name == "AttentionOp" or node.attr_name == SYNTHETIC_ATTENTION:
        if any("delta rule" in detail.lower() for detail in node.details):
            return node.details[0] if node.details else None
        return None

    if node.class_name == "OutputGate" or (node.role == "gate" and node.details):
        for detail in node.details:
            cleaned = detail.strip()
            if cleaned and cleaned != "Linear":
                return cleaned
        return None

    if node.class_name == "KernelPipeline":
        return node.details[0] if node.details else None

    if node.class_name == "ShortConvolution":
        if node.label == "Depthwise Conv":
            return None
        if node.details and node.details[0] == "depthwise conv":
            return "depthwise conv"
        if _short_conv_activation(node.details) and len(node.details) == 1:
            return "depthwise conv"

    for detail in node.details:
        cleaned = detail.strip()
        if (
            cleaned
            and not cleaned.startswith("method `")
            and not cleaned.startswith("kernel:")
            # ``raw_op: <name>`` is machine metadata for the type-check (lifted to
            # its own node attr at export), never a human-facing description.
            and not cleaned.startswith("raw_op:")
            # ``gate activation: <name>`` is structured metadata resolved at
            # init time and consumed by ``gated_norm_activation`` to key the
            # structural gated-norm purpose below; not a human-facing purpose.
            and not cleaned.startswith(GATE_ACTIVATION_DETAIL_PREFIX)
        ):
            if _FUNCTIONAL_CALL_DETAIL_RE.match(cleaned):
                continue
            return cleaned

    class_name = node.class_name or ""
    role = node.role
    attr = node.attr_name.lower()

    if class_name == "OutputGate" or role == "gate":
        return "Output gate — scales normalized output"
    if class_name == "ShortConvolution":
        activation = node.details[0] if node.details else None
        return f"depthwise conv" + (f" · {activation}" if activation else "")
    if class_name == "AttentionMerge":
        for detail in node.details:
            if detail.startswith("ports:"):
                return detail.replace("ports:", "ports ·").strip()
        return None
    if class_name == "KernelPipeline" and node.details:
        return node.details[0]
    if class_name in {"KernelOp", "KernelOutput"}:
        return None
    if is_fused_silu_mul_class(class_name):
        match = re.match(r"(?i)si[tl]u", class_name)
        stem = class_name[: match.end()] if match else "SiLU"
        return f"{stem}(gate) × up branch"
    if role == "norm" and gated_norm_activation(node):
        return "Normalize, then multiply by gate"
    if class_name == "Split" or node.attr_name == "split_gate_up":
        return "Split fused gate/up projection"
    if class_name in {"ActivationOp", "SituActivation"}:
        if class_name == "ActivationOp" and node.attr_name.endswith("_activation"):
            return None
        return f"Apply {node.label} to gate half"
    if class_name == "Multiply" or node.label in {"×", "Elementwise ×"}:
        return "Multiply gate and up activations"
    if role == "head" or attr == "lm_head":
        return "Project to vocabulary logits"
    if role == "router" or attr in {"gate", "router"}:
        return "Score and route tokens to experts"
    if role == "ffn":
        return "Position-wise feed-forward transform"
    if role == "embedding":
        return "Gather rows by token id"
    return None


def wrapper_module_comment(node: BlockNode) -> str | None:
    """Short description of what a wrapped module does, when applicable."""
    if wrapper_skips_comment(node):
        return None
    return block_purpose(node)


def wrapper_panel_line(node: BlockNode) -> str:
    """Single panel row: bullet label plus optional inline description."""
    line = wrapper_bullet(node)
    comment = wrapper_module_comment(node)
    if comment:
        return f"{line} — {comment}"
    return line


def inline_wrapper_step_label(
    wrapper: BlockNode | None,
    sub_step: BlockNode,
    sub_index: int,
) -> str | None:
    """Label for the first inlined step of a composite wrapper."""
    if sub_index != 0 or wrapper is None:
        return None
    return sub_step.label


def collect_function_steps(node: BlockNode) -> list[BlockNode]:
    """Collect leaf/basic/method-wrapper nodes that represent one forward op."""
    if is_method_wrapper(node) or (node.is_basic and not node.children):
        return [node]
    steps: list[BlockNode] = []
    for child in node.children:
        steps.extend(collect_function_steps(child))
    if not steps and not _is_composite_block(node):
        return [node]
    return steps


MIN_SUBGRAPH_OPERATIONS = 2


def forward_operation_count(
    node: BlockNode,
    *,
    basic_ops: BasicOpFilter | None = None,
) -> int:
    """Return forward ops represented by a block tree's computation graph."""
    from TraceLens.ModelUtils.basic_ops import BasicOpFilter as _BasicOpFilter
    from TraceLens.ModelUtils.computation_graph import SYNTHETIC_INPUT, build_computation_graph

    resolved = basic_ops or _BasicOpFilter.for_detailed()
    graph = build_computation_graph(node, basic_ops=resolved)
    return sum(
        1
        for spec in graph.nodes
        if spec.synthetic not in {SYNTHETIC_INPUT, "@output", "@combine"}
        and spec.label not in {"×", "+", "Elementwise ×"}
    )


def subgraph_expands_on_export(node: BlockNode) -> bool:
    """True when export expands this block inline or as a nested subgraph."""
    if is_inline_expandable_module(node):
        return True
    return _is_composite_block(node)


def subgraph_warrants_export(
    node: BlockNode,
    *,
    basic_ops: BasicOpFilter | None = None,
    min_operations: int = MIN_SUBGRAPH_OPERATIONS,
) -> bool:
    """True when a block tree should be exported as its own subgraph/section."""
    op_count = forward_operation_count(node, basic_ops=basic_ops)
    if op_count >= min_operations:
        return True
    if op_count == 1 and subgraph_expands_on_export(node):
        return True
    return False


def subgraph_warrants_json_export(
    node: BlockNode,
    *,
    basic_ops: BasicOpFilter | None = None,
) -> bool:
    """True when a block tree should appear as its own section in operator JSON export."""
    if _omit_from_detailed_view(node):
        return False
    return forward_operation_count(node, basic_ops=basic_ops) >= 1


def _clone_block_node(node: BlockNode, **changes: object) -> BlockNode:
    return replace(node, **changes)


def _is_substitutable_single_op_subgraph(
    node: BlockNode,
    *,
    basic_ops: BasicOpFilter | None = None,
) -> bool:
    """True when a composite wrapper should be replaced by its single inner op."""
    if is_method_wrapper(node) or node.is_basic:
        return False
    if is_inline_expandable_module(node):
        return False
    if not _is_composite_block(node):
        return False
    return forward_operation_count(node, basic_ops=basic_ops) == 1


def _substitute_single_op_subgraph(
    node: BlockNode,
    *,
    basic_ops: BasicOpFilter | None = None,
) -> BlockNode:
    """Return the single forward op represented by a one-op composite wrapper."""
    steps = collect_function_steps(node)
    if len(steps) == 1:
        substitute = steps[0]
    else:
        inner = straight_line_steps(node)
        substitute = inner[0] if len(inner) == 1 else node
    if substitute is node:
        return node
    input_source = node.input_source or substitute.input_source
    if input_source == substitute.input_source:
        return substitute
    return _clone_block_node(substitute, input_source=input_source)


def expand_block_tree_inplace(
    node: BlockNode,
    *,
    basic_ops: BasicOpFilter | None = None,
) -> BlockNode:
    """Expand straight-line composites and substitute single-op subgraph wrappers."""
    expanded_children = [
        expand_block_tree_inplace(child, basic_ops=basic_ops) for child in node.children
    ]
    node = _clone_block_node(node, children=expanded_children)

    if _is_substitutable_single_op_subgraph(node, basic_ops=basic_ops):
        substitute = _substitute_single_op_subgraph(node, basic_ops=basic_ops)
        return expand_block_tree_inplace(substitute, basic_ops=basic_ops)

    if is_straight_line_module(node):
        inner_steps = [
            expand_block_tree_inplace(step, basic_ops=basic_ops)
            for step in straight_line_steps(node)
        ]
        if len(inner_steps) == 1:
            substitute = inner_steps[0]
            input_source = node.input_source or substitute.input_source
            if input_source != substitute.input_source:
                substitute = _clone_block_node(substitute, input_source=input_source)
            return substitute
        return _clone_block_node(node, children=inner_steps)

    return node


def prepare_diagram_section_trees(
    trees: list[tuple[str, BlockNode]],
    *,
    basic_ops: BasicOpFilter | None = None,
) -> list[tuple[str, BlockNode]]:
    """Prepare parsed section block trees for Model Explorer export."""
    return [
        (title, expand_block_tree_inplace(tree, basic_ops=basic_ops))
        for title, tree in trees
    ]


def subgraph_warrants_diagram(
    node: BlockNode,
    *,
    basic_ops: BasicOpFilter | None = None,
    min_operations: int = MIN_SUBGRAPH_OPERATIONS,
) -> bool:
    """Alias for :func:`subgraph_warrants_export`."""
    return subgraph_warrants_export(
        node,
        basic_ops=basic_ops,
        min_operations=min_operations,
    )


_PIPELINE_WRAPPER_ATTRS = frozenset({"tokenization", "tokenizer"})


def _is_pipeline_wrapper(node: BlockNode) -> bool:
    """True for embed/head/pipeline modules that stay as spine tiles, not inlined subgraphs."""
    if node.attr_name.lower() in _PIPELINE_WRAPPER_ATTRS:
        return True
    return node.role in {"embedding", "head"}


def is_kernel_pipeline_tree(node: BlockNode) -> bool:
    """True for derived kernel pipelines rendered as nested trees."""
    return node.class_name == "KernelPipeline" and bool(node.children)


def _bypass_spans(node: BlockNode) -> list[tuple[int, int]]:
    """Step ranges that a tensor skips over on its way to a later step."""
    step_index = {
        child.attr_name: index
        for index, child in enumerate(node.children)
        if child.attr_name
    }
    spans: list[tuple[int, int]] = []
    for child in node.children:
        target = step_index.get(child.attr_name)
        if target is None:
            continue
        for name in child.operation_predecessors:
            source = step_index.get(name)
            if source is not None and target - source > 1:
                spans.append((source, target))
    return spans


def _has_overlapping_bypass_spans(node: BlockNode) -> bool:
    """True when bypassed step ranges nest or cross each other.

    Inline expansion draws a bypass by offsetting the steps it skips into a single
    side column. Consecutive bypasses reuse that column in turn, but nested or
    crossing ones would need columns of their own, so the block only reads clearly
    as a diagram of its own.
    """
    spans = _bypass_spans(node)
    return any(
        first is not second and first[0] < second[1] and second[0] < first[1]
        for index, first in enumerate(spans)
        for second in spans[index + 1 :]
    )


def is_straight_line_module(node: BlockNode) -> bool:
    """True when a composite block is a simple straight-line pipeline with no branching."""
    if is_kernel_pipeline_tree(node):
        return False
    if not _is_composite_block(node):
        return False
    if _is_pipeline_wrapper(node):
        return False
    segments = collect_computation_segments(node)
    if not segments:
        return False
    return all(isinstance(segment, SeqSegment) for segment in segments)


def is_linear_pipeline_block(node: BlockNode) -> bool:
    """True for straight-line composites and Situ-gated MLPs that expand inline."""
    return is_straight_line_module(node) or is_situ_gated_mlp(node)


def is_inline_expandable_module(node: BlockNode) -> bool:
    """True when a composite should render as expanded inline steps rather than a nested diagram."""
    if re.search(r"Indexer", node.class_name or "", re.I):
        # Large sparse-attention indexers stay separate sections; straight-line ones expand inline.
        return len(collect_function_steps(node)) < 12
    return is_linear_pipeline_block(node)


def _step_forms_own_frame(step: BlockNode) -> bool:
    """True when a step renders as its own namespace frame (submodule / method)."""
    return _is_composite_block(step) or is_method_wrapper(step)


def is_transparent_loop_wrapper(node: BlockNode) -> bool:
    """True when a composite's body is just leaf preamble/postamble around one loop.

    A wrapper whose forward reduces to some setup ops and a single loop (for
    example an expert-dispatch module: ``final = zeros_like(...); ...; for expert
    in hit: ...``) adds no grouping value beyond the loop it contains — the loop
    already renders as its own ``Loop_N_iterations`` frame. Flattening the wrapper
    lets that loop sit directly under the parent module.

    A wrapper is *not* transparent when a non-loop step forms its own frame (a
    nested submodule such as a normalization, or a helper-method call outside the
    loop) or when it contains more than one distinct loop: those group meaningfully.
    No class-name checks — the decision is purely structural.
    """
    if not _is_composite_block(node):
        return False
    segments = collect_computation_segments(node)
    if not segments:
        return False
    loop_details: set[str] = set()
    has_preamble = False
    for segment in segments:
        if not isinstance(segment, SeqSegment):
            # Fan-out / combine / side-feed structure is real grouping.
            return False
        step = segment.step
        loop_detail = next(
            (detail for detail in step.details if detail.startswith("loop:")), None
        )
        if loop_detail is not None:
            # A loop-body step (including helper methods expanded inside the loop).
            loop_details.add(loop_detail)
            continue
        # A step at this wrapper's own scope, outside any loop. If it forms its own
        # frame the wrapper groups more than a loop; otherwise it is loop setup.
        if _step_forms_own_frame(step):
            return False
        if step.label not in LAYOUT_ONLY_LABELS:
            has_preamble = True
    # Requiring a non-loop preamble step is what distinguishes a module that *owns*
    # a loop (setup before its `for`) from a helper/submodule whose every step
    # merely inherits an ancestor loop's detail because it is called inside one —
    # the latter has no step of its own outside the loop and must keep its frame.
    return has_preamble and len(loop_details) == 1


def is_transparent_inline_expansion(node: BlockNode) -> bool:
    """True when an expanded wrapper adds no useful grouping boundary.

    Helper methods that are straight pipelines already live inside their caller,
    while modules with only one value-changing operation are more legible as that
    operation in the surrounding flow. Layout steps remain visible, but do not
    justify a frame of their own.
    """
    if not is_straight_line_module(node):
        return False
    return (
        sum(
            step.label not in LAYOUT_ONLY_LABELS
            for step in collect_function_steps(node)
        )
        <= 1
    )


def should_expand_composite_wrapper(node: BlockNode) -> bool:
    """True when a composite wrapper should expand inline (straight-line modules only)."""
    return is_inline_expandable_module(node)


def straight_line_steps(node: BlockNode) -> list[BlockNode]:
    """Expand a straight-line composite into ordered direct child steps."""
    if not is_straight_line_module(node):
        return [node]
    steps: list[BlockNode] = []
    for segment in collect_computation_segments(node):
        if not isinstance(segment, SeqSegment):
            return [node]
        steps.append(segment.step)
    return steps


def linear_pipeline_steps(node: BlockNode) -> list[BlockNode]:
    """Alias for :func:`straight_line_steps`."""
    return straight_line_steps(node)


def _kernel_inline_frame_label(block: BlockNode) -> str | None:
    """Give per-tensor-port kernel expansions distinct frame namespaces."""
    if block.class_name != "KernelOp" or not block.children:
        return None
    from TraceLens.ModelUtils.kernel_pipeline import tensor_port_kernel_frame_label

    return tensor_port_kernel_frame_label(block.attr_name)


def inline_block_frame_label(block: BlockNode) -> str:
    """Display label for a dotted inline frame around an expanded sub-block."""
    if block.class_name == "KernelPipeline":
        return block.label
    kernel_label = _kernel_inline_frame_label(block)
    if kernel_label is not None:
        return kernel_label
    if block.class_name == "KernelOp" and block.children:
        return block.label
    if is_fused_silu_mul_class(block.class_name):
        return block.class_name
    # A traced free-function frame carries the raw synthetic call attr as both its
    # class_name and attr_name (``@positional_l1615_apply_rotary_pos_emb_vision``).
    # Name the frame after the source function so it reads like any module call
    # (``apply_rotary_pos_emb_vision``) rather than the internal synthetic attr; the
    # raw attr still survives inside each op's id, so this only renames the frame.
    if is_positional_synthetic(block.attr_name) or is_function_synthetic(
        block.attr_name
    ):
        return _synthetic_call_function_name(block.attr_name) or block.attr_name
    # A frame holding a whole computation is identified by the module class that
    # implements it; a frame around a single step is better named by the attribute
    # that step is reached through.
    if len(collect_function_steps(block)) > 1 and block.class_name != block.attr_name:
        return block.class_name
    return block.attr_name


def inline_block_frame_sublabel(block: BlockNode) -> str | None:
    """Expanded composites show step labels only; omit frame purpose lines."""
    return None


def is_single_function_tree(node: BlockNode) -> bool:
    """True when a block tree only contains one forward operation worth demoting to the panel."""
    if should_expand_composite_wrapper(node):
        return False
    return len(collect_function_steps(node)) == 1


def inline_composite_steps(
    step: BlockNode,
    *,
    basic_ops: BasicOpFilter | None = None,
) -> tuple[list[BlockNode], BlockNode | None]:
    """Inline a composite wrapper into its internal forward steps.

    Prefer showing actual computations (linear layers, activations, kernel substeps)
    over a single opaque module tile whenever internal structure is known.
    """
    del basic_ops
    if is_kernel_pipeline_tree(step) and step.children:
        return list(step.children), step

    if _is_output_gate_node(step) and step.children:
        inner_steps = (
            straight_line_steps(step)
            if is_straight_line_module(step)
            else list(step.children)
        )
        if inner_steps:
            return inner_steps, step

    if _is_composite_block(step):
        inner_ops = collect_function_steps(step)
        if len(inner_ops) == 1:
            # A semantic wrapper around one operation adds no information. Replace
            # it directly in its parent's dataflow (for example hc_head -> Mean).
            return inner_ops, None

    if not is_straight_line_module(step):
        return [step], None
    if not is_inline_expandable_module(step):
        return [step], None
    inner_steps = straight_line_steps(step)
    if len(inner_steps) > 1:
        return inner_steps, step
    if len(inner_steps) == 1 and _is_output_gate_node(step):
        return inner_steps, step
    return [step], None


_OMIT_DETAILED_TREES = frozenset({"tokenization", "tokenizer"})


def _omit_from_detailed_view(tree: BlockNode) -> bool:
    return tree.attr_name in _OMIT_DETAILED_TREES


def _show_single_function_in_diagram(tree: BlockNode) -> bool:
    """True when a one-op module should render as a diagram tile instead of the panel."""
    if tree.attr_name == "embed_tokens":
        return True
    return displays_as_linear(tree.attr_name, tree.class_name)


def partition_detail_trees(
    trees: list[tuple[str, BlockNode]],
) -> list[tuple[str, BlockNode]]:
    """Keep block trees that warrant a dedicated internal diagram."""
    kept: list[tuple[str, BlockNode]] = []
    for title, tree in trees:
        if _omit_from_detailed_view(tree):
            continue
        if is_straight_line_module(tree):
            continue
        if is_single_function_tree(tree) and _show_single_function_in_diagram(tree):
            continue
        kept.append((title, tree))
    return kept


def collect_method_wrappers(node: BlockNode) -> list[BlockNode]:
    """Collect method-wrapper leaves under a block tree."""
    if is_method_wrapper(node):
        return [node]
    wrappers: list[BlockNode] = []
    for child in node.children:
        wrappers.extend(collect_method_wrappers(child))
    return wrappers


def _is_output_gate_node(node: BlockNode) -> bool:
    """True for parallel output-gate modules (e.g. g_proj wrapped as OutputGate)."""
    return node.class_name == "OutputGate" or node.role == "gate"


def collect_parallel_gate_wrappers(
    node: BlockNode, *, inside_output_gate: bool = False
) -> list[BlockNode]:
    """Collect parallel gate side branches referenced inline but listed in the panel."""
    wrappers: list[BlockNode] = []
    gate_attrs = set(node.parallel_gates)
    for child in node.children:
        in_gate = inside_output_gate or _is_output_gate_node(child)
        if not in_gate and child.attr_name in gate_attrs:
            if not _is_composite_block(child):
                wrappers.append(child)
        wrappers.extend(
            collect_parallel_gate_wrappers(child, inside_output_gate=in_gate)
        )
    return wrappers


@dataclass
class BlockNode:
    """One node in a recursive block diagram."""

    attr_name: str
    class_name: str
    role: str
    label: str
    forward_order: int | None = None
    details: list[str] = field(default_factory=list)
    children: list[BlockNode] = field(default_factory=list)
    is_basic: bool = False
    norm_before: list[str] = field(default_factory=list)
    attention_inputs: dict[str, list[str]] = field(default_factory=dict)
    parallel_gates: list[str] = field(default_factory=list)
    input_fed_steps: list[str] = field(default_factory=list)
    side_inputs: dict[str, list[SideInputSpec]] = field(default_factory=dict)
    input_label: str | None = None
    input_source: str | None = None
    kernel_tensor_ports: dict[str, str] = field(default_factory=dict)
    tensor_input_labels: list[str] = field(default_factory=list)
    tensor_step_targets: dict[str, str] = field(default_factory=dict)
    kernel_predecessors: list[str] = field(default_factory=list)
    operation_predecessors: list[str] = field(default_factory=list)
    # Ordered output-port names for a multi-output op (split/chunk/unbind that was
    # tuple-unpacked); ``operation_predecessor_ports`` maps a producer attr this
    # node consumes to the output ordinal(s) it reads. Almost always one ordinal;
    # a node that reassembles two different slices of the *same* multi-output
    # producer (``torch.cat((q_pass, q_rot), dim=-1)``) reads two, so the value is
    # an ordered tuple rather than a single int. Both empty for ordinary ops.
    output_names: list[str] = field(default_factory=list)
    operation_predecessor_ports: dict[str, tuple[int, ...]] = field(default_factory=dict)
    kernel_second_operand: str | None = None
    external_inputs: list[str] = field(default_factory=list)
    param_inputs: list[str] = field(default_factory=list)
    boundary_input_name: str | None = None
    # Output ordinal read from a tuple-unpacked boundary input, when this op
    # consumes one slot of it (``cos = position_embeddings[0]`` -> 0). Lets the
    # boundary input node fan out one port per consuming op instead of collapsing
    # every consumer onto slot 0. ``None`` for a whole-tensor boundary read.
    boundary_input_ordinal: int | None = None
    forward_param_inputs: list[str] = field(default_factory=list)
    primary_output_step: str | None = None
    forward_return_slots: dict[str, str] = field(default_factory=dict)
    forward_return_order: list[str] = field(default_factory=list)
    primary_return_slot: str | None = None
    referenced_return_producers: set[str] = field(default_factory=set)
    loop_carried: list[LoopCarriedSpec] = field(default_factory=list)
    multi_return_module: bool = False
    forward_step_predecessors: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    forward_step_predecessor_args: dict[str, dict[str, str]] = field(
        default_factory=dict
    )
    forward_step_predecessor_ordinals: dict[str, dict[str, int]] = field(
        default_factory=dict
    )
    # Inline-expanded tuple-returning free function: call attr -> ordered
    # internal producer attrs (ordinal -> producer), so a consumer reading a
    # specific return slot docks onto the matching internal op, not the frame's
    # last op. Empty for single-return helpers.
    forward_step_return_producers: dict[str, list[str]] = field(
        default_factory=dict
    )
    # Submodule call attr -> secondary forward params it reads directly as bare
    # boundary args (``q = self.wq_b(q_resid)`` -> ``{'wq_b': ('q_resid',)}``).
    # A secondary param handed straight to a plain submodule has no internal
    # producer, so without this the submodule's input would collapse onto the
    # frame's primary ``@input``; the graph pass turns each into a dedicated
    # ``@input:<param>`` boundary port the submodule reads and the caller docks.
    forward_step_boundary_params: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    # A traced free-function call whose body (or a callee) materialises a tensor on
    # the host (``.tolist()``/``.item()``) — rendered with a ``device: cpu`` label.
    runs_on_host: bool = False


PortStyle = Literal["floating", "inline"]
SegmentSource = Literal["forward_input"]


@dataclass
class Branch:
    """One parallel path feeding a fan-in node."""

    label: str
    steps: list[BlockNode]
    port_style: PortStyle | None = None

    @property
    def port_label(self) -> str:
        return self.label


@dataclass
class SeqSegment:
    step: BlockNode


@dataclass
class TensorPortsSegment:
    """Labeled tensor inputs fanning into specific steps of a linear pipeline."""

    labels: list[str]
    targets: dict[str, str]
    steps: list[BlockNode]


@dataclass
class FanOutSegment:
    """Multiple parallel branches converging on one merge node."""

    branches: list[Branch]
    merge: BlockNode
    source: SegmentSource | None = "forward_input"


@dataclass
class CombineSegment:
    """Main path combined with a parallel side branch."""

    side: BlockNode
    after: list[BlockNode] = field(default_factory=list)
    side_port_label: str | None = None
    side_port_style: PortStyle = "inline"
    side_source: SegmentSource | None = "forward_input"
    op: str = "×"


@dataclass
class SideFeedSegment:
    """Sequential consumer with extra side inputs from prior steps or forward input."""

    consumer: BlockNode
    sides: list[SideInputSpec]
    side_producer_nodes: dict[str, BlockNode] = field(default_factory=dict)
    side_producer_chains: dict[str, list[BlockNode]] = field(default_factory=dict)


@dataclass
class SideCombineSegment:
    """Side-fed combine operator replacing a method wrapper (e.g. moe infer Σ)."""

    consumer: BlockNode
    sides: list[SideInputSpec]
    op: str


@dataclass
class ResidualAddSegment:
    """Parallel module on a saved residual, merged with the main path via +."""

    module: BlockNode
    sides: list[SideInputSpec]


ComputationSegment = (
    SeqSegment
    | TensorPortsSegment
    | FanOutSegment
    | CombineSegment
    | SideFeedSegment
    | SideCombineSegment
    | ResidualAddSegment
)


def _method_combine_op(step: BlockNode) -> str:
    """Map a method wrapper to its operation label without operand captions."""
    from TraceLens.ModelUtils.ast_analyze import combine_op_from_step_details

    op = combine_op_from_step_details(step.details)
    if op is not None:
        return op
    return "Function"


def _side_feed_chain_attrs(node: BlockNode) -> set[str]:
    """Attrs that only exist to feed a later side-input consumer."""
    attrs: set[str] = set()
    for specs in node.side_inputs.values():
        for spec in specs:
            if spec.source_kind == "prior_step" and spec.source_chain:
                attrs.update(spec.source_chain)
    return attrs


def _segment_for_step(node: BlockNode, step: BlockNode) -> ComputationSegment:
    side_specs = node.side_inputs.get(step.attr_name, [])
    if not side_specs:
        return SeqSegment(step=step)

    has_prior_side = any(side.source_kind == "prior_step" for side in side_specs)
    has_residual_side = any(side.source_kind == "forward_input" for side in side_specs)
    by_attr = {child.attr_name: child for child in node.children}
    side_producer_nodes = {
        side.source_chain[-1]: by_attr[side.source_chain[-1]]
        for side in side_specs
        if side.source_kind == "prior_step"
        and side.source_chain
        and side.source_chain[-1] in by_attr
    }
    side_producer_chains = {
        side.source_chain[-1]: [
            by_attr[attr] for attr in side.source_chain if attr in by_attr
        ]
        for side in side_specs
        if side.source_kind == "prior_step" and side.source_chain
    }

    if is_method_wrapper(step) and has_prior_side:
        return SideCombineSegment(
            consumer=step,
            sides=list(side_specs),
            op=_method_combine_op(step),
        )

    if has_residual_side and not has_prior_side:
        # Some forwards expose the residual branch call and its explicit Add as
        # separate operations. In that case the call is only a parallel side feed;
        # synthesizing another residual Add leaves a duplicate dead combine tile.
        step_index = next(
            (index for index, child in enumerate(node.children) if child is step),
            -1,
        )
        explicit_add = any(
            index > step_index
            and child.label == "Add"
            and step.attr_name in child.operation_predecessors
            for index, child in enumerate(node.children)
        )
        if not explicit_add:
            return ResidualAddSegment(module=step, sides=list(side_specs))

    if is_method_wrapper(step):
        return SideCombineSegment(
            consumer=step,
            sides=list(side_specs),
            op=_method_combine_op(step),
        )

    return SideFeedSegment(
        consumer=step,
        sides=list(side_specs),
        side_producer_nodes=side_producer_nodes,
        side_producer_chains=side_producer_chains,
    )


def _label_for_call(attr_name: str, class_name: str | None) -> str:
    if attr_name == SYNTHETIC_ATTENTION:
        return "Attention kernel"
    if class_name == "ShortConvolution":
        return "Depthwise Conv"
    if displays_as_linear(attr_name, class_name):
        return "Linear"
    if class_name:
        role = _classify_role(attr_name, class_name)
        return _label_for(role, class_name, attr_name)
    readable = attr_name.replace("_", " ")
    return readable[:24]


def _predecessor_ports_dict(
    pairs: tuple[tuple[str, int], ...],
) -> dict[str, tuple[int, ...]]:
    """Producer attr -> ordinal(s) a step's expression reads from it.

    Usually one ordinal per producer. A step that reads two different output
    slots of the *same* multi-output producer within one expression
    (``torch.cat((q_pass, q_rot), dim=-1)`` reassembling a ``torch.split``)
    needs both preserved, in read order -- a plain ``dict(pairs)`` would keep
    only the last-written ordinal and silently drop the other slice's edge.
    """
    ordinals: dict[str, list[int]] = {}
    for producer, ordinal in pairs:
        ordinals.setdefault(producer, []).append(ordinal)
    return {producer: tuple(values) for producer, values in ordinals.items()}


def _leaf_node(
    *,
    attr_name: str,
    class_name: str,
    forward_order: int | None,
    details: list[str] | None = None,
    basic: bool = True,
    label: str | None = None,
    kernel_predecessors: list[str] | None = None,
    operation_predecessors: list[str] | None = None,
    output_names: list[str] | None = None,
    operation_predecessor_ports: dict[str, tuple[int, ...]] | None = None,
    kernel_second_operand: str | None = None,
    external_inputs: list[str] | None = None,
    param_inputs: list[str] | None = None,
    boundary_input_name: str | None = None,
    boundary_input_ordinal: int | None = None,
    runs_on_host: bool = False,
) -> BlockNode:
    # A traced free function's synthetic attr embeds the callee name
    # (``@fn_l1840_get_vision_attention_seqlens``). Token-classifying that name
    # misreads incidental words: ``get_vision_attention_seqlens`` matches the
    # "attention" token and the node is treated as an attention kernel, which the
    # graph then folds away — so a real, consumed computation vanishes. These
    # synthetics are their own op kind, not module roles, so give them a neutral
    # role and let the dedicated ``@attention`` kernel keep the attention role.
    if is_function_synthetic(attr_name):
        role = "other"
    else:
        role = _classify_role(attr_name, class_name)
    return BlockNode(
        attr_name=attr_name,
        class_name=class_name,
        role=role,
        label=label or _label_for_call(attr_name, class_name),
        forward_order=forward_order,
        details=list(details or []),
        is_basic=basic,
        kernel_predecessors=list(kernel_predecessors or []),
        operation_predecessors=list(operation_predecessors or []),
        output_names=list(output_names or []),
        operation_predecessor_ports=dict(operation_predecessor_ports or {}),
        kernel_second_operand=kernel_second_operand,
        external_inputs=list(external_inputs or []),
        param_inputs=list(param_inputs or []),
        boundary_input_name=boundary_input_name,
        boundary_input_ordinal=boundary_input_ordinal,
        runs_on_host=runs_on_host,
    )


def _expanded_free_function_node(
    call_attr: str,
    cls: ClassStructure,
    *,
    child_order: int | None,
    child_details: list[str],
    label: str,
    fallback_class_name: str,
    fallback_param_inputs: list[str] | None,
) -> BlockNode:
    """Render a traced free-function call, expanded into its body when known.

    A rope helper or other module-level function called from a forward
    (``apply_rotary_pos_emb_vision(q, k, cos, sin)``) shows the computation it
    performs — mirroring the class-method ``multi_op_methods`` expansion — instead
    of a single opaque tile. Falls back to the opaque op leaf when the callee's
    body was not collected (keeping prior behaviour for un-expandable helpers).
    """
    method_ops = cls.multi_op_methods.get(call_attr)
    output_names = cls.forward_step_output_names.get(call_attr)
    runs_on_host = cls.forward_step_runs_on_host.get(call_attr, False)
    # A host-only helper (``get_vision_position_ids``) builds integer index
    # bookkeeping whose per-op shapes are not meaningfully inferable (advanced
    # indexing / host arithmetic), so expanding it only surfaces wrong inner
    # shapes. Keep it a single opaque leaf (still ``device: cpu`` labelled); only
    # device-side helpers (rope math) expand into their visible tensor ops.
    if method_ops and not runs_on_host:
        name = _synthetic_call_function_name(call_attr) or call_attr
        call_context = [
            detail
            for detail in child_details
            if detail.startswith(("loop:", "condition:"))
        ]
        # A callee parameter fed from a boundary forward input at the call site
        # (``cos``/``sin`` <- ``position_embeddings``) has no producer inside the
        # frame. Rewrite those ops so they read the boundary input by name, and
        # tag the tuple-unpack ordinal so it fans out one port per slot.
        boundary_arg_params = cls.forward_step_boundary_arg_params.get(call_attr, {})
        # A callee parameter fed from a real, already-computed producer at the
        # call site (``apply_rotary_pos_emb(query_states, key_states, cos,
        # sin)`` -- ``q``/``k`` each resolve to their own ``.transpose(...)``
        # node, not a boundary alias) has no dedicated port mechanism of its
        # own: unlike the boundary case, nothing renders a synthetic tile for
        # it, so leaving the raw parameter name in ``param_inputs`` strands the
        # reference -- the op that reads it never gets an edge, and (for a
        # multi-return callee like this one) the producer that feeds a
        # non-primary parameter is left with no consumer at all. Route it as a
        # normal operation-to-operation edge instead: the producer attr is
        # already a real node in the caller's graph, so the standard
        # attr-based predecessor wiring picks it up like any other dataflow
        # edge once it is on ``operation_predecessors``.
        producer_arg_params = cls.forward_step_predecessor_args.get(call_attr, {})
        # Two callee parameters can be fed by the *same* real producer at the
        # call site (``apply_rotary_pos_emb(x, cos=rotary_emb, sin=rotary_emb)``
        # -- both trace to one multi-return submodule call). ``extra_predecessors``
        # then names that one producer for both a ``cos``-reading op and a
        # ``sin``-reading op alike, with no way to tell them apart downstream.
        # The call site's own per-arg ordinal (``cos``: 0, ``sin``: 1, from the
        # producer's own return-tuple order) disambiguates them; carry it onto
        # the op's own predecessor ports.
        ordinal_arg_params = cls.forward_step_predecessor_ordinals.get(call_attr, {})
        # A callee whose body reassigns a parameter name (``cos =
        # cos.repeat_interleave(...).unsqueeze(...)``) reads the raw producer only
        # at the *first* op that consumes that slot; every later op bearing the
        # same name reads the rebound local through its own internal predecessor.
        # Attach the producer edge for a given ``(producer, ordinal)`` slot to that
        # first consumer alone -- mirroring how :func:`_build_module_param_ordinal_entries`
        # keeps only the first consumer of each boundary slot -- so the raw producer
        # is not re-wired as a spurious extra tensor operand onto every rebound
        # re-read (an ``unsqueeze``/``multiply`` that already reads the rebinding).
        claimed_producer_slots: set[tuple[str, int]] = set()
        children: list[BlockNode] = []
        for operation_index, operation in enumerate(method_ops):
            translated: list[str] = []
            boundary_name: str | None = None
            boundary_ordinal: int | None = None
            extra_predecessors: list[str] = []
            extra_predecessor_ports: list[tuple[str, int]] = []
            for param in operation.param_inputs:
                mapping = boundary_arg_params.get(param)
                if mapping is None:
                    producer_attr = producer_arg_params.get(param)
                    if producer_attr is not None:
                        ordinal = ordinal_arg_params.get(param)
                        if ordinal is not None:
                            slot = (producer_attr, ordinal)
                            if slot in claimed_producer_slots:
                                # A rebound re-read of this slot; the value already
                                # flows in through this op's internal predecessor.
                                continue
                            claimed_producer_slots.add(slot)
                            extra_predecessor_ports.append((producer_attr, ordinal))
                            # Expose the callee param as this op's own entry point,
                            # exactly as a boundary-fed origin lands in ``translated``
                            # below. Naming the entry point lets the caller resolve
                            # ``cos``/``sin`` to THIS op (op0/op2) and skip the
                            # unrelated side args (``x``, the other slot, ...) via the
                            # ``entry_params`` dump guard -- rather than dumping every
                            # side arg onto the frame's first op. Scoped to the slot's
                            # first consumer so a rebound re-read is not re-exposed.
                            if param not in translated:
                                translated.append(param)
                        if producer_attr not in extra_predecessors:
                            extra_predecessors.append(producer_attr)
                        continue
                    translated.append(param)
                    continue
                origin, ordinal = mapping
                if origin not in translated:
                    translated.append(origin)
                # An op reading a single boundary slot docks that slot; one that
                # reads several (an over-collected ``cos``+``sin`` on a downstream
                # Add) keeps the first — the extra edges are pruned as the slot's
                # real consumer already docked it.
                if boundary_name is None:
                    boundary_name = origin
                    boundary_ordinal = ordinal
            operation_predecessors = list(operation.predecessors)
            for producer_attr in extra_predecessors:
                if producer_attr not in operation_predecessors:
                    operation_predecessors.append(producer_attr)
            children.append(
                _leaf_node(
                    attr_name=operation.attr_name,
                    class_name=operation.class_name,
                    forward_order=operation_index,
                    details=[
                        *operation.details,
                        *(
                            detail
                            for detail in call_context
                            if detail not in operation.details
                        ),
                    ],
                    label=operation.label,
                    basic=True,
                    operation_predecessors=operation_predecessors,
                    output_names=list(operation.output_names),
                    operation_predecessor_ports=_predecessor_ports_dict(
                        tuple(operation.predecessor_ports) + tuple(extra_predecessor_ports)
                    ),
                    external_inputs=list(operation.external_inputs),
                    param_inputs=translated,
                    boundary_input_name=boundary_name,
                    boundary_input_ordinal=boundary_ordinal,
                    # A host helper (``get_vision_position_ids``) expands into its
                    # index-bookkeeping ops; each one still runs on the host, so the
                    # ``device: cpu`` label/style propagates onto every child rather
                    # than being lost when the single opaque tile opens up.
                    runs_on_host=runs_on_host,
                )
            )
        return BlockNode(
            attr_name=call_attr,
            class_name=call_attr,
            role="other",
            label=label,
            forward_order=child_order,
            details=[f"function `{name}()`", *call_context],
            output_names=list(output_names or []),
            children=children,
            runs_on_host=runs_on_host,
        )
    return _leaf_node(
        attr_name=call_attr,
        class_name=fallback_class_name,
        forward_order=child_order,
        details=list(child_details),
        label=label,
        basic=False,
        output_names=output_names,
        param_inputs=fallback_param_inputs,
        runs_on_host=runs_on_host,
    )


def _boundary_input_name(
    operation: ForwardOperation,
    cls: ClassStructure,
) -> str | None:
    """The forward parameter an operation reads straight from the block boundary.

    Only these operations sit on the block's edge, so they are what a boundary input
    has to be named after: everything else reads a tensor produced inside the block.
    """
    if FORWARD_METHOD_INPUT in operation.predecessors:
        return cls.forward_input_name
    # `zeros_like` takes its shape from the tensor it is handed, and the analysis
    # records no producer for that read, so the boundary would otherwise fall back
    # to whichever edge happened to reach it.
    if operation.label == "Zeros like" and not operation.predecessors:
        return cls.forward_input_name
    mutated = next(
        (
            detail.removeprefix("mutates: ")
            for detail in operation.details
            if detail.startswith("mutates: ")
        ),
        None,
    )
    if mutated:
        return mutated
    if operation.param_inputs:
        return operation.param_inputs[0]
    return None


def _boundary_input_ordinal(
    operation: ForwardOperation,
    cls: ClassStructure,
) -> int | None:
    """The unpack ordinal of the boundary parameter named by ``_boundary_input_name``.

    A tuple-unpacked alias of a secondary forward parameter (``cos``, ``sin``
    aliasing ``position_embeddings``) carries its slot's ordinal in
    ``param_input_ordinals`` (keyed by the same origin name ``_boundary_input_name``
    resolved to). Without this, several aliases of the same tensor collapse onto
    one boundary edge instead of each reaching its own slot.
    """
    name = _boundary_input_name(operation, cls)
    if name is None:
        return None
    for param, ordinal in operation.param_input_ordinals:
        if param == name:
            return ordinal
    return None


def _gate_side_consumer(
    side_inputs: dict[str, list[SideInputSpec]],
    gate_attr: str,
) -> tuple[str | None, SideInputSpec | None]:
    for consumer, specs in side_inputs.items():
        for spec in specs:
            if spec.source_chain and spec.source_chain[-1] == gate_attr:
                return consumer, spec
    return None, None


def _output_gate_details(
    gate_attr: str,
    *,
    side_inputs: dict[str, list[SideInputSpec]],
    gate_activations: dict[str, str],
    consumer_class: str | None = None,
    norm_gate_activation: str | None = None,
) -> list[str]:
    """Describe how a parallel output gate is computed and consumed."""
    consumer, spec = _gate_side_consumer(side_inputs, gate_attr)
    inline_activation = gate_activations.get(gate_attr)

    lines: list[str] = ["Linear"]

    # Structural signal, not the consumer's class name: the consumer is a gated
    # norm exactly when the extractor resolved a gate activation for it. An
    # unresolved gate is left unlabelled rather than defaulting to a guess.
    if norm_gate_activation is not None:
        if inline_activation:
            lines.append(f"{inline_activation}(linear out)")
            lines.append(f"norm(attn_out) × gate → {consumer or 'o_norm'}")
        else:
            lines.append(f"{norm_gate_activation} inside {consumer or 'o_norm'}")
            lines.append("norm(attn_out) × gate")
    elif inline_activation:
        lines.append(f"{inline_activation}(linear out)")
        if consumer:
            port = spec.port_label if spec else None
            suffix = f" port {port!r}" if port else ""
            lines.append(f"feeds {consumer}{suffix}")
    elif consumer:
        port = spec.port_label if spec else None
        suffix = f" port {port!r}" if port else ""
        lines.append(f"feeds {consumer}{suffix}")
    else:
        lines.append("output gate for normalized branch")

    return lines


def _kernel_pipeline_block_nodes(
    *,
    forward_order: int | None,
    details: list[str],
    attention_inputs: dict[str, list[str]] | None = None,
    parent_class_name: str | None = None,
) -> tuple[BlockNode, BlockNode | None]:
    """Expand a multi-input kernel attention step into pipeline and output sibling nodes."""
    from TraceLens.ModelUtils.kernel_pipeline import (
        compute_tensor_step_targets,
        introspect_kernel_pipeline,
    )

    kernel = kernel_name_from_step_details(details) or "kernel"
    pipeline_steps, output_steps = introspect_kernel_pipeline(details)
    inputs = dict(attention_inputs or {})
    tensor_ports = kernel_kwarg_ports(details)
    ordered_labels = tensor_input_label_order(details, inputs)
    step_targets = compute_tensor_step_targets(details, pipeline_steps)

    pipeline_children: list[BlockNode] = []
    for index, step in enumerate(pipeline_steps):
        if len(step.children) >= 2:
            sub_children = [
                _leaf_node(
                    attr_name=child.attr_name,
                    class_name=child.class_name,
                    forward_order=sub_index,
                    label=child.label,
                    details=[],
                    basic=False,
                    kernel_second_operand=child.second_operand,
                )
                for sub_index, child in enumerate(step.children)
            ]
            pipeline_children.append(
                BlockNode(
                    attr_name=step.attr_name,
                    class_name="KernelOp",
                    role="other",
                    label=step.call_name,
                    forward_order=index,
                    details=[],
                    is_basic=False,
                    children=sub_children,
                    kernel_predecessors=list(step.predecessors),
                )
            )
        else:
            pipeline_children.append(
                _leaf_node(
                    attr_name=step.attr_name,
                    class_name=step.class_name,
                    forward_order=index,
                    label=step.call_name,
                    details=[],
                    basic=False,
                    kernel_predecessors=list(step.predecessors),
                )
            )

    opaque_kernel = not pipeline_children
    if opaque_kernel:
        pipeline_children.append(
            _leaf_node(
                attr_name="@attn_pipeline_core",
                class_name="KernelOp",
                forward_order=0,
                label=kernel,
                details=[f"kernel: {kernel}"],
                basic=False,
            )
        )

    pipeline_label = (
        pipeline_children[0].label
        if len(pipeline_children) == 1
        else f"{kernel} pipeline"
    )
    pipeline_node = BlockNode(
        attr_name="@attn_pipeline",
        class_name="KernelPipeline",
        role="attention",
        label=pipeline_label,
        forward_order=forward_order,
        details=[f"kernel pipeline · {kernel}"],
        is_basic=False,
        children=pipeline_children,
        attention_inputs=inputs,
        kernel_tensor_ports=tensor_ports,
        tensor_input_labels=ordered_labels,
        tensor_step_targets=step_targets,
    )

    output_node: BlockNode | None
    if output_steps:
        output = output_steps[0]
        output_node = _leaf_node(
            attr_name="@attn_output",
            class_name="KernelOutput",
            forward_order=(forward_order or 0) + 1,
            label=output.call_name,
            details=[],
            basic=False,
            kernel_predecessors=list(output.predecessors),
        )
    elif opaque_kernel:
        # One opaque kernel call with no parsed stages: the pipeline tile is the whole call,
        # so a separate output tile would just repeat the kernel name.
        output_node = None
    else:
        output_node = _leaf_node(
            attr_name="@attn_output",
            class_name="KernelOutput",
            forward_order=(forward_order or 0) + 1,
            label=kernel,
            details=[f"kernel: {kernel}"],
            basic=False,
        )

    return pipeline_node, output_node


def _output_gate_block_node(
    linear_step: BlockNode,
    activation: str | None,
    details: list[str] | None = None,
    *,
    consumer_class: str | None = None,
) -> BlockNode:
    """Expand a parallel output gate into its own nested sub-diagram."""
    gate_details = list(details or [])
    linear_label = (
        "Linear"
        if displays_as_linear(linear_step.attr_name, linear_step.class_name)
        else linear_step.label
    )
    linear_step = _leaf_node(
        attr_name=linear_step.attr_name,
        class_name=linear_step.class_name,
        forward_order=linear_step.forward_order,
        label=linear_label,
        details=[],
    )
    children: list[BlockNode] = [linear_step]
    step_order = (linear_step.forward_order or 0) + 1
    if activation:
        children.append(
            _leaf_node(
                attr_name=SYNTHETIC_GATE_ACTIVATION,
                class_name="ActivationOp",
                forward_order=step_order,
                label=activation,
                details=[f"{activation}(linear out)"],
                basic=False,
            )
        )
    return BlockNode(
        attr_name=linear_step.attr_name,
        class_name="OutputGate",
        role="gate",
        label="Output gate",
        forward_order=linear_step.forward_order,
        details=gate_details,
        is_basic=False,
        children=children,
    )


def _is_simple_parallel_output_gate(child: BlockNode) -> bool:
    """True when a parallel gate is a single linear (+ optional activation), not a composite module."""
    if displays_as_linear(child.attr_name, child.class_name):
        return True
    return len(collect_function_steps(child)) <= 1


_OP_STEP_NAME_RE = re.compile(r"^@op_l\d+_c\d+_(.+)$")


def _next_sibling_is_activation_step(
    child_nodes: list[BlockNode], index: int, activation: str
) -> bool:
    """True when the very next sibling already IS this gate's activation.

    ``g = self.gate_proj(x).sigmoid()`` (assigned to its own local, as opposed
    to being applied fully inline inside another expression) gets its ``.sigmoid()``
    extracted as its own ordinary tensor-op step immediately following the gate's
    call in ``forward_calls`` -- the same statement that made ``_parallel_gate_activation``
    recognize the gate in the first place. Synthesizing a SECOND ``@gate_activation``
    node inside the gate's own frame in that case would duplicate a computation that
    the plain sibling step already performs and wires correctly, producing two
    producers for one value (the downstream consumer then wires onto both).
    """
    if index + 1 >= len(child_nodes):
        return False
    match = _OP_STEP_NAME_RE.match(child_nodes[index + 1].attr_name)
    return bool(match) and match.group(1) == activation.lower()


def _wrap_parallel_gate_children(
    child_nodes: list[BlockNode],
    parallel_gates: list[str],
    gate_activations: dict[str, str],
    side_inputs: dict[str, list[SideInputSpec]] | None = None,
) -> list[BlockNode]:
    if not parallel_gates:
        return child_nodes
    gate_attrs = set(parallel_gates)
    side_inputs = side_inputs or {}
    wrapped: list[BlockNode] = []
    for index, child in enumerate(child_nodes):
        if child.attr_name not in gate_attrs:
            wrapped.append(child)
            continue
        if not _is_simple_parallel_output_gate(child):
            wrapped.append(child)
            continue
        activation = gate_activations.get(child.attr_name)
        if activation is None and displays_as_linear(child.attr_name, child.class_name):
            wrapped.append(child)
            continue
        if activation and _next_sibling_is_activation_step(
            child_nodes, index, activation
        ):
            # Already represented as its own ordinary step right after this gate;
            # let that step (and its real predecessor/consumer wiring) stand alone
            # instead of also nesting a synthetic duplicate inside this frame.
            wrapped.append(child)
            continue
        consumer, _ = _gate_side_consumer(side_inputs, child.attr_name)
        consumer_node = next(
            (node for node in child_nodes if node.attr_name == consumer), None
        )
        details = _output_gate_details(
            child.attr_name,
            side_inputs=side_inputs,
            gate_activations=gate_activations,
            consumer_class=consumer_node.class_name if consumer_node else None,
            norm_gate_activation=(
                gated_norm_activation(consumer_node) if consumer_node else None
            ),
        )
        wrapped.append(
            _output_gate_block_node(
                child,
                gate_activations.get(child.attr_name),
                details=details,
                consumer_class=consumer_node.class_name if consumer_node else None,
            )
        )
    return wrapped


def side_producer_has_activation(producer: BlockNode) -> bool:
    """True when an output-gate side producer already ends with an activation step."""
    if producer.class_name == "OutputGate":
        return any(
            child.attr_name == SYNTHETIC_GATE_ACTIVATION for child in producer.children
        )
    return False


def gated_norm_activation(node: BlockNode) -> str | None:
    """Return the gate activation applied inside a gated norm module, if resolved.

    The activation is read structurally from the tag the extractor writes once it
    has resolved the name generically -- from the module's ``activation=`` kwarg or
    its own ``ACT2FN[self.<x>]`` init/config symbol table (see
    ``ast_analyze._gated_norm_activation_from_forward``). The class name is never
    consulted, and an unresolved gate is left ``None`` rather than defaulting to a
    guessed activation.
    """
    for detail in node.details:
        if detail.startswith(GATE_ACTIVATION_DETAIL_PREFIX):
            resolved = detail[len(GATE_ACTIVATION_DETAIL_PREFIX) :].strip()
            if resolved:
                return resolved
    return None


def _short_conv_activation(details: list[str] | None) -> str | None:
    """Return the init-time activation name for a ShortConvolution, if any."""
    if not details:
        return None
    for detail in details:
        cleaned = detail.strip()
        if (
            not cleaned
            or cleaned.startswith("method `")
            or cleaned.startswith("kernel:")
        ):
            continue
        if "=" in cleaned:
            continue
        return cleaned
    return None


def _short_convolution_block_node(
    *,
    attr_name: str,
    forward_order: int | None,
    activation: str,
    details: list[str] | None = None,
) -> list[BlockNode]:
    """Expand ShortConvolution + activation into separate conv and act steps."""
    base_order = forward_order or 0
    return [
        _leaf_node(
            attr_name=attr_name,
            class_name="ShortConvolution",
            forward_order=base_order,
            label="Depthwise Conv",
            details=[],
            basic=False,
        ),
        _leaf_node(
            attr_name=f"{attr_name}_activation",
            class_name="ActivationOp",
            forward_order=base_order + 1,
            label=activation,
            details=[],
            basic=False,
        ),
    ]


def _gate_up_linear_attr_names() -> frozenset[str]:
    return frozenset({"gate_proj", "up_proj", "w1", "w3"})


def _has_upstream_gate_up_linears(prior_steps: list[BlockNode]) -> bool:
    """True when separate gate/up projection linears already precede act_fn."""
    found = {
        step.attr_name
        for step in prior_steps
        if displays_as_linear(step.attr_name, step.class_name)
        and step.attr_name in _gate_up_linear_attr_names()
    }
    return {"gate_proj", "up_proj"}.issubset(found) or {"w1", "w3"}.issubset(found)


def _situ_and_mul_block_node(
    *,
    attr_name: str,
    forward_order: int | None,
    details: list[str] | None = None,
    role: str = "ffn",
    prior_steps: list[BlockNode] | None = None,
    class_name: str = "SituAndMul",
) -> BlockNode:
    """Expand a fused SiLU/SiTU-and-multiply module into a small internal pipeline."""
    prior_steps = list(prior_steps or [])
    children: list[BlockNode] = []
    step_order = 0
    if not _has_upstream_gate_up_linears(prior_steps):
        children.append(
            _leaf_node(
                attr_name="split_gate_up",
                class_name="Linear",
                forward_order=step_order,
                label="Linear",
                details=[],
            )
        )
        step_order += 1
    match = re.match(r"(?i)si[tl]u", class_name)
    stem = class_name[: match.end()] if match else "SiLU"
    purpose = f"{stem}(gate) × up branch"
    children.extend(
        [
            _leaf_node(
                attr_name="situ_activation",
                class_name=f"{stem}Activation",
                forward_order=step_order,
                label=stem,
                details=["activation on gate half"],
                basic=False,
            ),
            _leaf_node(
                attr_name="elementwise_mul",
                class_name="Multiply",
                forward_order=step_order + 1,
                label="×",
                details=["gate × up"],
                basic=False,
            ),
        ]
    )
    return BlockNode(
        attr_name=attr_name,
        class_name=class_name,
        role=role,
        label="Gated multiply",
        forward_order=forward_order,
        details=list(details or [purpose]),
        is_basic=False,
        input_label="gate_up",
        children=children,
    )


def _nested_input_source(parent: BlockNode, child: BlockNode) -> str:
    """Describe where a nested block's primary input comes from."""
    parent_cls = parent.class_name or parent.label
    if child.role == "ffn" and parent.role == "moe":
        return f"Linear in {parent_cls}"
    if is_fused_silu_mul_class(child.class_name):
        return f"gate_up in {parent_cls}"
    if child.input_label and child.input_label not in {"hidden_states", "x"}:
        return f"{child.input_label} in {parent_cls}"
    return f"{parent_cls}"


def _append_branch_followups(
    steps: list[BlockNode], pre_merge: list[BlockNode]
) -> list[BlockNode]:
    """Append immediate post-step siblings (e.g. conv → activation) to a provenance branch."""
    if not steps:
        return steps
    by_attr = {node.attr_name: node for node in pre_merge}
    order = [node.attr_name for node in pre_merge]
    extended = list(steps)
    last_attr = steps[-1].attr_name
    while True:
        follow_attr = f"{last_attr}_activation"
        if follow_attr not in by_attr:
            break
        if order.index(follow_attr) != order.index(last_attr) + 1:
            break
        extended.append(by_attr[follow_attr])
        last_attr = follow_attr
    return extended


def _branches_from_provenance(
    pre_merge: list[BlockNode],
    provenance: dict[str, list[str]],
) -> list[Branch]:
    """Build parallel branches from named provenance chains captured in the AST."""
    by_attr = {node.attr_name: node for node in pre_merge}
    branches: list[Branch] = []

    for label in provenance:
        chain = provenance.get(label, [])
        if not chain:
            continue
        nodes = [by_attr[attr] for attr in chain if attr in by_attr]
        nodes = _append_branch_followups(nodes, pre_merge)
        if nodes:
            branches.append(Branch(label=label, steps=nodes, port_style="inline"))

    if len(branches) >= 2:
        return _collapse_identical_branches(branches)
    return []


def _collapse_identical_branches(branches: list[Branch]) -> list[Branch]:
    """Merge branches that follow identical module chains."""
    by_signature: dict[tuple[str, ...], list[Branch]] = {}
    for branch in branches:
        signature = tuple(node.attr_name for node in branch.steps)
        by_signature.setdefault(signature, []).append(branch)

    collapsed: list[Branch] = []
    for group in by_signature.values():
        if len(group) == 1:
            collapsed.append(group[0])
            continue
        labels = [branch.label for branch in group]
        merged_label = (
            labels[0]
            if all(label == labels[0] for label in labels)
            else "/".join(labels)
        )
        collapsed.append(
            Branch(
                label=merged_label, steps=group[0].steps, port_style=group[0].port_style
            )
        )
    return collapsed


def _partition_named_branches(pre_merge: list[BlockNode]) -> list[Branch]:
    """Partition steps into labeled branches using attribute-name prefix clustering."""
    buckets: dict[str, list[BlockNode]] = {}
    for node in pre_merge:
        prefix = (
            node.attr_name.split("_", 1)[0] if "_" in node.attr_name else node.attr_name
        )
        buckets.setdefault(prefix, []).append(node)

    branches: list[Branch] = []
    for prefix, nodes in buckets.items():
        if nodes:
            branches.append(Branch(label=prefix, steps=nodes, port_style="inline"))
    return branches


def _parallel_side_port_label(side: BlockNode) -> str:
    """Readable inline port label for a parallel side branch."""
    steps = collect_function_steps(side)
    if steps:
        return steps[0].label
    return side.label


def _side_feed_targets(node: BlockNode) -> dict[str, str]:
    """Map side-producer attr -> consumer attr for prior-step side feeds."""
    targets: dict[str, str] = {}
    for consumer_name, specs in node.side_inputs.items():
        for spec in specs:
            if spec.source_kind != "prior_step" or not spec.source_chain:
                continue
            targets[spec.source_chain[-1]] = consumer_name
    return targets


def _forward_side_combine_producers(node: BlockNode) -> set[str]:
    """Side-chain attrs that still expand as forward segments before a SideCombine consumer."""
    side_chain_attrs = _side_feed_chain_attrs(node)
    if not side_chain_attrs:
        return set()
    targets = _side_feed_targets(node)
    by_attr = {child.attr_name: child for child in node.children}
    producers: set[str] = set()
    for producer_attr in side_chain_attrs:
        consumer_name = targets.get(producer_attr)
        if consumer_name is None:
            continue
        producer = by_attr.get(producer_attr)
        consumer = by_attr.get(consumer_name)
        if producer is None or consumer is None:
            continue
        if not isinstance(_segment_for_step(node, producer), SeqSegment):
            continue
        if isinstance(_segment_for_step(node, consumer), SideCombineSegment):
            producers.add(producer_attr)
    return producers


def _situ_gated_mlp_parts(
    node: BlockNode,
) -> tuple[BlockNode, BlockNode, BlockNode, BlockNode, BlockNode] | None:
    """Return (gate, up, act_fn, situ, down) when ``node`` is a Situ-gated MLP."""
    if not node.children:
        return None
    by_attr = {child.attr_name: child for child in node.children}
    act_fn = by_attr.get("act_fn")
    if act_fn is None or not is_fused_silu_mul_class(act_fn.class_name):
        return None
    gate = by_attr.get("gate_proj") or by_attr.get("w1")
    up = by_attr.get("up_proj") or by_attr.get("w3")
    down = by_attr.get("down_proj") or by_attr.get("w2")
    if gate is None or up is None or down is None:
        return None
    situ = next(
        (
            child
            for child in act_fn.children
            if re.search(r"(?i)si[tl]uactivation", child.class_name)
        ),
        None,
    )
    if situ is None:
        return None
    return gate, up, act_fn, situ, down


def is_situ_gated_mlp(node: BlockNode) -> bool:
    """True when a block is a gate/up projection pair feeding a fused SiLU-and-multiply."""
    return _situ_gated_mlp_parts(node) is not None


def _situ_gated_mlp_segments(node: BlockNode) -> list[ComputationSegment] | None:
    """Model Situ-gated MLPs as gate → Situ combined with a parallel up branch."""
    parts = _situ_gated_mlp_parts(node)
    if parts is None:
        return None
    gate, up, _act_fn, situ, down = parts
    return [
        SeqSegment(step=gate),
        SeqSegment(step=situ),
        CombineSegment(
            side=up,
            after=[down],
            side_port_label="up",
            side_port_style="inline",
            op="×",
        ),
    ]


def _is_attention_merge_node(child: BlockNode) -> bool:
    if child.attr_name == SYNTHETIC_ATTENTION:
        return True
    if child.attr_name == "@attn_pipeline" or child.class_name == "KernelPipeline":
        return True
    return False


def _tensor_ports_segment(node: BlockNode) -> TensorPortsSegment | None:
    """Build a tensor fan-in segment when a block exposes labeled kernel ports."""
    if not node.tensor_input_labels or not node.children:
        return None
    return TensorPortsSegment(
        labels=list(node.tensor_input_labels),
        targets=dict(node.tensor_step_targets),
        steps=list(node.children),
    )


def collect_computation_segments(node: BlockNode) -> list[ComputationSegment]:
    """Build generic render segments from forward-ordered block children."""
    tensor_segment = _tensor_ports_segment(node)
    if tensor_segment is not None:
        return [tensor_segment]

    children = node.children
    if not children:
        return []

    merge_idx = next(
        (i for i, child in enumerate(children) if _is_attention_merge_node(child)),
        None,
    )
    if merge_idx is None:
        situ_segments = _situ_gated_mlp_segments(node)
        if situ_segments is not None:
            return situ_segments
        side_chain_attrs = _side_feed_chain_attrs(node)
        side_combine_producers = _forward_side_combine_producers(node)
        return [
            _segment_for_step(node, child)
            for child in children
            if child.attr_name not in side_chain_attrs
            or child.attr_name in side_combine_producers
        ]

    pre_merge = children[:merge_idx]
    merge_node = children[merge_idx]
    post_merge = children[merge_idx + 1 :]

    provenance = node.attention_inputs
    pipeline_child = next(
        (child for child in node.children if child.class_name == "KernelPipeline"), None
    )
    if pipeline_child and pipeline_child.attention_inputs:
        provenance = pipeline_child.attention_inputs

    branches = _branches_from_provenance(pre_merge, provenance)
    represented = {step.attr_name for branch in branches for step in branch.steps}
    incomplete_provenance = any(step.attr_name not in represented for step in pre_merge)
    if incomplete_provenance:
        # A single fan-out cannot represent a projection fan-in followed by a
        # second q/k/v/g/beta fan-out. Keep every step and let explicit
        # predecessor wiring carry the DAG instead of dropping shared setup.
        branches = []
    if len(branches) < 2 and not incomplete_provenance:
        branches = [
            Branch(label=branch.label, steps=branch.steps, port_style="inline")
            for branch in _partition_named_branches(pre_merge)
            if branch.steps
        ]

    segments: list[ComputationSegment] = []
    if len(branches) >= 2:
        segments.append(FanOutSegment(branches=branches, merge=merge_node))
    else:
        segments.extend(SeqSegment(step=step) for step in pre_merge)
        segments.append(SeqSegment(step=merge_node))

    remaining = list(post_merge)
    side_feed_targets = _side_feed_targets(node)
    side_chain_attrs = _side_feed_chain_attrs(node)
    side_combine_producers = _forward_side_combine_producers(node)
    skip_sequential: set[str] = side_chain_attrs - side_combine_producers
    for index, step in enumerate(remaining):
        consumer = remaining[index + 1] if index + 1 < len(remaining) else None
        if consumer and side_feed_targets.get(step.attr_name) == consumer.attr_name:
            skip_sequential.add(step.attr_name)

    if (
        remaining
        and remaining[0].attr_name in node.parallel_gates
        and len(remaining) >= 2
        and side_feed_targets.get(remaining[0].attr_name) != remaining[1].attr_name
    ):
        side = remaining[0]
        segments.append(
            CombineSegment(
                side=side,
                after=remaining[1:],
                side_port_label=_parallel_side_port_label(side),
                side_port_style="inline",
            )
        )
    else:
        for step in remaining:
            if step.attr_name in skip_sequential:
                continue
            segments.extend([_segment_for_step(node, step)])
    return segments


def _is_composite_block(node: BlockNode | None) -> bool:
    """True when a block has internal structure worth its own diagram."""
    return (
        node is not None
        and not node.is_basic
        and bool(node.children)
        and not is_method_wrapper(node)
    )


def is_basic_op_tile(block: BlockNode | None) -> bool:
    """True when a detail tile uses the gray basic-op styling."""
    if block is None:
        return False
    return block.is_basic or is_simple_modeled_tile(block)


def is_simple_modeled_tile(node: BlockNode) -> bool:
    """Modeled op rendered as a gray tile with operation text (not a role-colored leaf)."""
    if node.is_basic or is_method_wrapper(node):
        return False
    if _is_output_gate_node(node) or is_kernel_pipeline_tree(node):
        return False
    if _is_composite_block(node):
        return False
    return introspect_is_modeling_operation(
        node.class_name, node.attr_name, node.details
    )


def tile_sublabel(
    block: BlockNode | None, *, in_inline_frame: bool = False
) -> str | None:
    """Secondary label for a detail tile; in-box sublabels are disabled."""
    del block, in_inline_frame
    return None


def tile_purpose_annotation(block: BlockNode | None) -> str | None:
    """Short purpose line rendered below a tile (outside the box).

    Disabled: tile labels already name the operation; extra prose was redundant
    or incorrect (e.g. RoPE freq/apply descriptions, router step duplicates).
    """
    return None


def tile_display_labels(
    block: BlockNode | None,
    *,
    spec_label: str | None = None,
    in_inline_frame: bool = False,
    port_label: str | None = None,
    port_style: PortStyle | None = None,
) -> tuple[str, str | None]:
    """Primary and secondary labels for a diagram tile from block-tree metadata."""
    if block is not None and in_inline_frame:
        return block.label, tile_sublabel(block, in_inline_frame=True)

    if port_label and port_style == "inline":
        if block is not None and block.is_basic:
            return block.label, None

    if block is not None and block.is_basic:
        return block.label, None

    if block is not None and block.class_name == "KernelOp":
        from TraceLens.ModelUtils.kernel_pipeline import kernel_op_display_label

        return kernel_op_display_label(block.label), None

    if block is not None and block.class_name in {
        "ActivationOp",
        "AttentionOp",
        "KernelPipeline",
        "KernelOutput",
    }:
        return block.label, None

    if block is None:
        return spec_label or "", None

    sublabel = tile_sublabel(block, in_inline_frame=False)
    return spec_label or block.label, sublabel


def collect_nested_diagrams(
    root: BlockNode,
    *,
    basic_ops: BasicOpFilter | None = None,
) -> list[tuple[str, BlockNode]]:
    """Collect composite blocks referenced in a diagram for separate sub-diagrams."""
    from TraceLens.ModelUtils.computation_graph import build_computation_graph
    from TraceLens.ModelUtils.basic_ops import BasicOpFilter as _BasicOpFilter

    resolved_basic_ops = basic_ops or _BasicOpFilter.for_detailed()

    seen: set[str] = set()
    ordered: list[tuple[str, BlockNode]] = []

    def consider(block: BlockNode | None, parent_block: BlockNode | None) -> None:
        if block is None or not _is_composite_block(block):
            return
        if is_inline_expandable_module(block):
            return
        if not subgraph_warrants_export(block, basic_ops=resolved_basic_ops):
            return
        if block.attr_name in seen:
            return
        seen.add(block.attr_name)
        if block.input_source is None and parent_block is not None:
            block.input_source = _nested_input_source(parent_block, block)
        ordered.append((block.label, block))
        inner = build_computation_graph(block, basic_ops=resolved_basic_ops)
        for spec in inner.nodes:
            consider(spec.block, block)

    graph = build_computation_graph(root, basic_ops=resolved_basic_ops)
    for spec in graph.nodes:
        consider(spec.block, root)

    return ordered


def flatten_computation_segments(node: BlockNode) -> list[ComputationSegment]:
    """Prepare top-level segments for graph export, keeping composite blocks intact."""
    return collect_computation_segments(node)


def _is_expandable_registered_class(
    registry: dict[str, ClassStructure], class_name: str | None
) -> bool:
    """True when a class is registered with a parseable forward we can expand.

    Keys on structural facts -- registry membership plus an extracted forward
    op list / submodule-call list -- never a class-name allowlist. A fused
    activation whose source lives in the modeling file (e.g. Kimi
    ``SituAndMul``) is in the registry with populated ``forward_operations``, so
    it expands into its real primitive ops through the normal recursion path. An
    unresolved *external* fused activation (``SiluAndMul`` imported from a kernel
    package) is absent from the registry, so callers keep its synthetic
    SiLU-and-multiply leaf instead.
    """
    if not class_name:
        return False
    cls = registry.get(class_name)
    return cls is not None and bool(cls.forward_operations or cls.forward_calls)


def build_block_node(
    *,
    attr_name: str,
    class_name: str,
    registry: dict[str, ClassStructure],
    basic_ops: BasicOpFilter,
    visited: frozenset[str] | None = None,
    details: list[str] | None = None,
    forward_order: int | None = None,
    infer_init_steps: bool = False,
) -> BlockNode:
    """Expand one submodule into a recursive block tree using forward-pass order."""
    visited = visited or frozenset()
    role = _classify_role(attr_name, class_name)
    label = _label_for_call(attr_name, class_name)

    # A gated norm whose class lives in the registry (its forward is available)
    # carries a generically-resolved gate activation; surface it as the tagged
    # detail so a leaf-rendered gated norm reports its activation without any
    # class-name matching. Modules constructed with an ``activation=`` kwarg
    # already carry the tag via ``details`` from the parent's init extraction.
    _registered = registry.get(class_name)
    if (
        role == "norm"
        and _registered is not None
        and _registered.gate_activation
        and not any(
            detail.startswith(GATE_ACTIVATION_DETAIL_PREFIX)
            for detail in (details or [])
        )
    ):
        details = [
            *(details or []),
            f"{GATE_ACTIVATION_DETAIL_PREFIX}{_registered.gate_activation}",
        ]

    if attr_name == SYNTHETIC_ATTENTION:
        step_details = list(details or [])
        if is_kernel_pipeline_step(step_details):
            pipeline_node, _output_node = _kernel_pipeline_block_nodes(
                forward_order=forward_order,
                details=step_details,
                parent_class_name=class_name,
            )
            return pipeline_node
        return _leaf_node(
            attr_name=attr_name,
            class_name="AttentionOp",
            forward_order=forward_order,
            label=attention_kernel_label(step_details),
            details=attention_kernel_details(step_details),
            basic=False,
        )

    if resolve_is_basic(
        class_name,
        attr_name,
        basic_ops,
        details=details,
        in_registry=class_name in registry,
    ):
        return _leaf_node(
            attr_name=attr_name,
            class_name=class_name,
            forward_order=forward_order,
            details=details,
        )

    if class_name not in registry:
        return _leaf_node(
            attr_name=attr_name,
            class_name=class_name,
            forward_order=forward_order,
            details=details,
            basic=False,
        )

    if class_name in visited:
        return BlockNode(
            attr_name=attr_name,
            class_name=class_name,
            role=role,
            label=label,
            forward_order=forward_order,
            details=["recursive reference"],
            is_basic=True,
        )

    cls = registry[class_name]
    parsed_steps = [
        step for step in cls.forward_calls if step not in _SKIP_INIT_CLASS_NAMES
    ]
    if infer_init_steps:
        if cls.forward_operations:
            forward_steps = parsed_steps
        elif not parsed_steps:
            forward_steps = effective_forward_calls(cls)
        else:
            uses_init_modules = any(
                base_submodule_attr(step) in cls.init_assignments
                for step in parsed_steps
            )
            forward_steps = (
                parsed_steps if not uses_init_modules else effective_forward_calls(cls)
            )
    else:
        forward_steps = parsed_steps

    if not forward_steps:
        return BlockNode(
            attr_name=attr_name,
            class_name=class_name,
            role=role,
            label=label,
            forward_order=forward_order,
            details=list(details or []),
            is_basic=True,
        )

    order_map = (
        {name: idx for idx, name in enumerate(cls.forward_calls)}
        if cls.forward_calls
        else {name: idx for idx, name in enumerate(forward_steps)}
    )
    child_nodes: list[BlockNode] = []

    for index, call_attr in enumerate(forward_steps):
        child_order = order_map.get(call_attr, index)
        # A repeated submodule/method call carries a call-site ``@l{lineno}`` suffix
        # so its wiring stays distinct (``recomposition_frequencies@l1773`` vs
        # ``@l1774``); the base attr is the join key for the child class / method-op
        # tables, which are keyed once per child.
        base_attr = base_submodule_attr(call_attr)
        child_details = (
            cls.forward_step_details.get(call_attr)
            or cls.forward_step_details.get(base_attr)
            or cls.init_details.get(base_attr, [])
        )

        if call_attr == SYNTHETIC_ATTENTION:
            # A phantom attention hoisted from a looped submodule's own forward
            # (see ``_drop_phantom_attention_steps`` in ast_analyze) survives in
            # ``forward_calls`` but was pruned from ``forward_step_predecessors``
            # and never had ``attention_inputs``. Without a backing step or
            # inputs it is not a real call at this scope; materialising it as a
            # node produces a duplicate attention the exporter later wires into a
            # cycle. The genuine attention still renders inside the submodule.
            if (
                call_attr not in cls.forward_step_predecessors
                and not cls.attention_inputs
            ):
                continue
            if is_kernel_pipeline_step(child_details, cls.attention_inputs):
                pipeline_node, output_node = _kernel_pipeline_block_nodes(
                    forward_order=child_order,
                    details=child_details,
                    attention_inputs=cls.attention_inputs,
                    parent_class_name=class_name,
                )
                child_nodes.append(pipeline_node)
                if output_node is not None:
                    child_nodes.append(output_node)
            else:
                # A boundary attention input (empty provenance chain) is a
                # forward parameter fed straight to the kernel from an enclosing
                # scope — e.g. ``cu_seqlens`` produced before the block loop.
                # Declaring it as a ``param_inputs`` entry gives the kernel a
                # dedicated pipeline entry point so the caller's predecessor edge
                # (``@fn_..._get_vision_attention_seqlens -> @attention``) docks
                # on the kernel instead of being dropped onto the block's head
                # norm (which the guard in section 1b would then skip, leaving
                # the producer dangling and stripped).
                boundary_inputs = [
                    port
                    for port, chain in cls.attention_inputs.items()
                    if not chain
                ]
                child_nodes.append(
                    _leaf_node(
                        attr_name=call_attr,
                        class_name="AttentionOp",
                        forward_order=child_order,
                        label=attention_kernel_label(child_details),
                        details=attention_kernel_details(
                            child_details, cls.attention_inputs
                        ),
                        param_inputs=boundary_inputs,
                        basic=False,
                    )
                )
            continue

        if is_forward_operation(call_attr):
            operation = cls.forward_operations.get(call_attr)
            if operation is None:
                continue
            child_nodes.append(
                _leaf_node(
                    attr_name=operation.attr_name,
                    class_name=operation.class_name,
                    forward_order=child_order,
                    details=list(operation.details),
                    label=operation_display_label(
                        operation.label, class_name=operation.class_name
                    ),
                    basic=True,
                    operation_predecessors=list(operation.predecessors),
                    output_names=list(operation.output_names),
                    operation_predecessor_ports=_predecessor_ports_dict(operation.predecessor_ports),
                    external_inputs=list(operation.external_inputs),
                    param_inputs=list(operation.param_inputs),
                    boundary_input_name=_boundary_input_name(operation, cls),
                    boundary_input_ordinal=_boundary_input_ordinal(operation, cls),
                )
            )
            continue

        if is_functional_synthetic(call_attr):
            op_label = functional_display_label(call_attr)
            child_nodes.append(
                _leaf_node(
                    attr_name=call_attr,
                    class_name=op_label,
                    forward_order=child_order,
                )
            )
            continue

        if is_positional_synthetic(call_attr):
            expanded = _expanded_free_function_node(
                call_attr,
                cls,
                child_order=child_order,
                child_details=child_details,
                label=positional_display_label(call_attr),
                fallback_class_name="PositionalOp",
                fallback_param_inputs=list(
                    cls.forward_step_boundary_params.get(call_attr, ())
                ),
            )
            child_nodes.append(expanded)
            continue

        if is_function_synthetic(call_attr):
            expanded = _expanded_free_function_node(
                call_attr,
                cls,
                child_order=child_order,
                child_details=child_details,
                label=function_display_label(call_attr),
                fallback_class_name="FunctionOp",
                fallback_param_inputs=None,
            )
            child_nodes.append(expanded)
            continue

        child_class = cls.init_assignments.get(base_attr)
        if child_class is None or child_class in _SKIP_INIT_CLASS_NAMES:
            if child_class in _SKIP_INIT_CLASS_NAMES:
                continue
            method_ops = cls.multi_op_methods.get(base_attr)
            if method_ops:
                call_context = [
                    detail
                    for detail in child_details
                    if detail.startswith(("loop:", "condition:"))
                ]
                # A tuple-returning method (``key_states, value_states =
                # self.expand_kv(...)``) carries its return slots so the frame
                # exposes each as its own output port; a consumer then docks the
                # matching slot instead of every slot collapsing onto the tail.
                method_returns = cls.multi_op_method_returns.get(base_attr)
                m_return_slots, m_return_order, m_primary_return = (
                    method_returns if method_returns else ({}, [], None)
                )
                m_primary_output_step = (
                    m_return_slots.get(m_primary_return)
                    if m_primary_return and m_return_slots
                    else None
                )
                # A submodule invoked mid-expression inside this helper method
                # (``self.act_fn(gate)`` in ``return self.act_fn(gate) * up``)
                # is captured by the extractor only as a bare predecessor NAME
                # on whichever op reads its result -- unlike a call that sits
                # directly in the class's own ``forward``, it never goes through
                # the ``child_class = cls.init_assignments.get(base_attr)``
                # resolution above, so it would otherwise never become a node:
                # the consumer's edge dangles on a name with no producer, and
                # anything that fed ONLY that submodule call (the "gate" branch
                # here) looks unconsumed. Resolve every such name against this
                # class's own submodule registry here too and materialise it as
                # a real sibling child (same recursive expansion a top-level
                # forward call gets), so the pre-existing predecessor name
                # resolves to it.
                method_op_attrs = {operation.attr_name for operation in method_ops}
                submodule_children: list[BlockNode] = []
                seen_submodule_attrs: set[str] = set()
                for operation in method_ops:
                    referenced = {
                        *operation.predecessors,
                        *(attr for attr, _ordinal in operation.predecessor_ports),
                    }
                    for name in referenced:
                        if (
                            name == FORWARD_METHOD_INPUT
                            or name in method_op_attrs
                            or name in seen_submodule_attrs
                        ):
                            continue
                        submodule_class = cls.init_assignments.get(name)
                        if (
                            submodule_class is None
                            or submodule_class in _SKIP_INIT_CLASS_NAMES
                        ):
                            continue
                        seen_submodule_attrs.add(name)
                        submodule_children.append(
                            build_block_node(
                                attr_name=name,
                                class_name=submodule_class,
                                registry=registry,
                                basic_ops=basic_ops,
                                visited=visited | {class_name},
                                details=cls.init_details.get(name, []),
                                forward_order=child_order,
                                infer_init_steps=infer_init_steps,
                            )
                        )
                # The section 1b wiring pass (``forward_step_predecessors`` on the
                # ENCLOSING composite node) is what actually docks a submodule
                # child's input edge -- the same mechanism a top-level forward
                # call's submodule uses. Carrying the extractor's full
                # ``step_predecessors`` map here (harmless for the ordinary op
                # steps too: any key that does not match one of this frame's own
                # children is simply skipped) is what lets the newly
                # materialised ``submodule_children`` above dock their input.
                method_step_predecessors = cls.multi_op_method_step_predecessors.get(
                    base_attr, {}
                )
                # This frame's own body is straight-line and inlined at
                # construction time (``_add_linear_pipeline_chain``), not via the
                # section-1b wiring pass -- so a materialised submodule child's
                # input actually resolves through ``_submodule_chain_input``,
                # which reads ``forward_step_predecessor_args`` (arg name ->
                # producer), not ``forward_step_predecessors``. Carry the
                # method's own map here too, the same way a top-level forward's
                # submodule call already relies on it.
                method_step_predecessor_args = (
                    cls.multi_op_method_step_predecessor_args.get(base_attr, {})
                )

                def _method_op_leaf(operation: ForwardOperation, position: int) -> BlockNode:
                    return _leaf_node(
                        attr_name=operation.attr_name,
                        class_name=operation.class_name,
                        forward_order=position,
                        details=[
                            *operation.details,
                            *(
                                detail
                                for detail in call_context
                                if detail not in operation.details
                            ),
                        ],
                        label=operation.label,
                        basic=True,
                        operation_predecessors=list(operation.predecessors),
                        output_names=list(operation.output_names),
                        operation_predecessor_ports=_predecessor_ports_dict(
                            operation.predecessor_ports
                        ),
                        external_inputs=list(operation.external_inputs),
                        param_inputs=list(operation.param_inputs),
                    )

                # A submodule call embedded mid-expression (``act_fn``) must sit
                # in its real evaluation-order position -- after the op that
                # produces its operand, before the op that consumes its result
                # -- not merely first or last, or the frame's own straight-line
                # construction-time wiring (which feeds each step from the
                # previous sibling by default) docks it to the wrong producer.
                # ``multi_op_method_order`` (when the method has such an
                # embedded call) gives that merged true order; fall back to the
                # unordered submodule-children-first layout otherwise.
                method_order = cls.multi_op_method_order.get(base_attr)
                submodule_children_by_attr = {
                    child.attr_name: child for child in submodule_children
                }
                method_ops_by_attr = {
                    operation.attr_name: operation for operation in method_ops
                }
                if method_order:
                    method_children: list[BlockNode] = []
                    for position, name in enumerate(method_order):
                        submodule_child = submodule_children_by_attr.get(name)
                        if submodule_child is not None:
                            # Re-stamp to its merged-order position. Its own
                            # ``forward_order`` still reflects wherever the
                            # extractor built it (the submodule-call's own,
                            # generally much larger, source index), which would
                            # outrank an op step that actually precedes it. A
                            # sibling wiring pass (``_first_graph_index_for_module``)
                            # picks this frame's "first" step by ``forward_order``
                            # to resolve the frame's own entry-point edge, so a
                            # stale value here misdirects a real predecessor
                            # edge onto the wrong internal op (e.g. a caller's
                            # boundary arg landing on the *second* step instead
                            # of the submodule call that is genuinely first).
                            submodule_child.forward_order = position
                            method_children.append(submodule_child)
                            continue
                        operation = method_ops_by_attr.get(name)
                        if operation is not None:
                            method_children.append(_method_op_leaf(operation, position))
                    covered = {node.attr_name for node in method_children}
                    # Safety net: any op the merge failed to place (should not
                    # happen) still gets rendered, appended in its own order.
                    method_children.extend(
                        _method_op_leaf(operation, len(method_children) + index)
                        for index, operation in enumerate(method_ops)
                        if operation.attr_name not in covered
                    )
                else:
                    method_children = [
                        *submodule_children,
                        *(
                            _method_op_leaf(operation, operation_index)
                            for operation_index, operation in enumerate(method_ops)
                        ),
                    ]
                child_nodes.append(
                    BlockNode(
                        attr_name=call_attr,
                        class_name=base_attr,
                        role=_classify_role(base_attr, base_attr),
                        label=base_attr.strip("_").replace("_", " "),
                        forward_order=child_order,
                        details=[f"method `{base_attr}()`", *call_context],
                        input_label=cls.multi_op_method_inputs.get(base_attr),
                        forward_return_slots=dict(m_return_slots),
                        forward_return_order=list(m_return_order),
                        primary_return_slot=m_primary_return,
                        primary_output_step=m_primary_output_step,
                        multi_return_module=len(m_return_order) >= 2,
                        forward_step_predecessors=dict(method_step_predecessors),
                        forward_step_predecessor_args=dict(method_step_predecessor_args),
                        children=method_children,
                    )
                )
                continue
            single_op = cls.single_op_methods.get(base_attr)
            if single_op is not None:
                # Keep the method's attr_name so forward wiring still resolves the step.
                child_nodes.append(
                    _leaf_node(
                        attr_name=call_attr,
                        class_name=single_op.class_name,
                        forward_order=child_order,
                        details=list(single_op.details),
                        label=single_op.label,
                        basic=True,
                        operation_predecessors=list(single_op.predecessors),
                        output_names=list(single_op.output_names),
                        operation_predecessor_ports=_predecessor_ports_dict(single_op.predecessor_ports),
                        external_inputs=list(single_op.external_inputs),
                        param_inputs=list(single_op.param_inputs),
                    )
                )
                continue
            child_nodes.append(
                _leaf_node(
                    attr_name=call_attr,
                    class_name=base_attr,
                    forward_order=child_order,
                    details=child_details or [f"method `{base_attr}()`"],
                )
            )
            continue

        if is_fused_silu_mul_class(child_class) and not _is_expandable_registered_class(
            registry, child_class
        ):
            # Unresolved external fused activation (its source is not in the
            # registry, e.g. ``SiluAndMul`` imported from a kernel package):
            # emit the synthetic SiLU-and-multiply leaf. A registered fused
            # activation (Kimi ``SituAndMul``) falls through to the recursion
            # path below and expands into its real primitive ops.
            child_nodes.append(
                _situ_and_mul_block_node(
                    attr_name=call_attr,
                    forward_order=child_order,
                    details=child_details,
                    role=role,
                    prior_steps=child_nodes,
                    class_name=child_class,
                )
            )
            continue

        if child_class == "ShortConvolution":
            activation = _short_conv_activation(child_details)
            if activation:
                child_nodes.extend(
                    _short_convolution_block_node(
                        attr_name=call_attr,
                        forward_order=child_order,
                        activation=activation,
                        details=child_details,
                    )
                )
                continue

        child_nodes.append(
            build_block_node(
                attr_name=call_attr,
                class_name=child_class,
                registry=registry,
                basic_ops=basic_ops,
                visited=visited | {class_name},
                details=child_details,
                forward_order=child_order,
                infer_init_steps=infer_init_steps,
            )
        )

    child_nodes = _wrap_parallel_gate_children(
        child_nodes,
        cls.parallel_gates,
        cls.gate_activations,
        cls.side_inputs,
    )

    attention_inputs = dict(cls.attention_inputs)
    primary_output_step = None
    if cls.primary_return_slot and cls.forward_return_slots:
        primary_output_step = cls.forward_return_slots.get(cls.primary_return_slot)

    return BlockNode(
        attr_name=attr_name,
        class_name=class_name,
        role=role,
        label=label,
        forward_order=forward_order,
        details=list(details or []),
        children=child_nodes,
        is_basic=False,
        norm_before=list(cls.norm_before),
        attention_inputs=attention_inputs,
        parallel_gates=list(cls.parallel_gates),
        input_fed_steps=list(cls.input_fed_calls),
        side_inputs=dict(cls.side_inputs),
        input_label=cls.forward_input_name,
        primary_output_step=primary_output_step,
        forward_return_slots=dict(cls.forward_return_slots),
        forward_return_order=list(cls.forward_return_order),
        forward_param_inputs=list(cls.forward_param_inputs),
        primary_return_slot=cls.primary_return_slot,
        referenced_return_producers=set(cls.referenced_return_producers),
        loop_carried=list(cls.loop_carried),
        multi_return_module=len(cls.forward_return_order) >= 2,
        forward_step_predecessors=dict(cls.forward_step_predecessors),
        forward_step_predecessor_args=dict(cls.forward_step_predecessor_args),
        forward_step_predecessor_ordinals=dict(
            cls.forward_step_predecessor_ordinals
        ),
        forward_step_return_producers=dict(cls.forward_step_return_producers),
        forward_step_boundary_params=dict(cls.forward_step_boundary_params),
    )


def _input_sources_for_components(
    components: list[BlockComponent],
    *,
    forward_sequence: list[str] | None = None,
) -> dict[str, str]:
    """Map compute module attr names to the upstream operator that feeds them."""
    if forward_sequence:
        return input_sources_from_forward_sequence(components, forward_sequence)
    return upstream_input_sources(components)


def build_stack_component_tree(
    component: BlockComponent,
    registry: dict[str, ClassStructure],
    basic_ops: BasicOpFilter,
) -> BlockNode:
    """Build a block tree for one main-diagram stack component."""
    if component.class_name in registry:
        return build_block_node(
            attr_name=component.attr_name,
            class_name=component.class_name,
            registry=registry,
            basic_ops=basic_ops,
            details=list(component.details),
            forward_order=component.forward_order,
        )
    return BlockNode(
        attr_name=component.attr_name,
        class_name=component.class_name,
        role=component.role,
        label=component.label,
        forward_order=component.forward_order,
        details=list(component.details),
        is_basic=resolve_is_basic(
            component.class_name,
            component.attr_name,
            basic_ops,
            details=list(component.details),
            in_registry=False,
        ),
    )


def spine_expanded_frame_label(
    component: BlockComponent, *, positional_encoding: str
) -> str:
    """Dotted-frame title for a straight-line module expanded on the main spine."""
    if component.role == "positional":
        return f"Positional ({positional_encoding}) ({component.attr_name})"
    return f"{component.label} ({component.attr_name})"


def _build_component_block_trees(
    components: list[BlockComponent],
    registry: dict[str, ClassStructure],
    basic_ops: BasicOpFilter,
    *,
    include_norms: bool = False,
    skip_method_wrappers: bool = True,
    infer_init_steps: bool = False,
) -> list[tuple[str, BlockNode]]:
    """Build block trees for a flat list of stack or decoder components."""
    trees: list[tuple[str, BlockNode]] = []
    seen: set[tuple[str, str]] = set()
    ordered = sorted(
        components,
        key=lambda comp: (
            comp.forward_order is None,
            comp.forward_order if comp.forward_order is not None else 999,
            comp.attr_name,
            comp.class_name,
        ),
    )
    for comp in ordered:
        if comp.role == "norm" and not include_norms:
            continue
        if comp.forward_order is None and comp.role not in {
            "embedding",
            "head",
            "positional",
            "norm",
        }:
            continue
        key = (comp.attr_name, comp.class_name)
        if key in seen:
            continue
        seen.add(key)
        tree = build_block_node(
            attr_name=comp.attr_name,
            class_name=comp.class_name,
            registry=registry,
            basic_ops=basic_ops,
            details=list(comp.details),
            forward_order=comp.forward_order,
            infer_init_steps=infer_init_steps,
        )
        if skip_method_wrappers and is_method_wrapper(tree):
            continue
        cls_info = registry.get(comp.class_name)
        if comp.role in {"embedding", "head", "positional", "norm"}:
            tree.input_label = (
                cls_info.forward_input_name
                if cls_info and cls_info.forward_input_name
                else ("input_ids" if comp.role == "embedding" else "hidden_states")
            )
        trees.append((comp.label, tree))
    return trees


def build_pipeline_block_trees(
    *,
    stack_pre: list[BlockComponent],
    registry: dict[str, ClassStructure],
    basic_ops: BasicOpFilter,
    include_norms: bool = False,
    infer_init_steps: bool = False,
) -> list[tuple[str, BlockNode]]:
    """Build pre-decoder detail sections (embeddings, positional encoding)."""
    return _build_component_block_trees(
        stack_pre,
        registry,
        basic_ops,
        include_norms=include_norms,
        infer_init_steps=infer_init_steps,
    )


def build_head_block_trees(
    *,
    stack_tail: list[BlockComponent],
    registry: dict[str, ClassStructure],
    basic_ops: BasicOpFilter,
    include_norms: bool = False,
    infer_init_steps: bool = False,
) -> list[tuple[str, BlockNode]]:
    """Build post-decoder detail sections (final norm, LM head)."""
    return _build_component_block_trees(
        stack_tail,
        registry,
        basic_ops,
        include_norms=include_norms,
        infer_init_steps=infer_init_steps,
    )


def build_decoder_block_trees(
    components: list[BlockComponent],
    registry: dict[str, ClassStructure],
    basic_ops: BasicOpFilter,
    *,
    decoder_class: str | None = None,
    include_norms: bool = False,
    infer_init_steps: bool = False,
) -> list[tuple[str, BlockNode]]:
    """Build recursive trees for detailed diagrams."""
    trees: list[tuple[str, BlockNode]] = []
    seen: set[tuple[str, str]] = set()
    decoder = registry.get(decoder_class) if decoder_class else None
    detail_components = (
        expand_conditional_block_components(decoder, components)
        if decoder is not None
        else list(components)
    )
    forward_sequence = list(decoder.forward_calls) if decoder is not None else None
    input_sources = _input_sources_for_components(
        detail_components,
        forward_sequence=forward_sequence,
    )

    ordered = sorted(
        detail_components,
        key=lambda comp: (
            comp.forward_order is None,
            comp.forward_order if comp.forward_order is not None else 999,
            comp.attr_name,
            comp.class_name,
        ),
    )

    for comp in ordered:
        if comp.role == "norm" and not include_norms:
            continue
        if comp.forward_order is None:
            continue
        key = (comp.attr_name, comp.class_name)
        if key in seen:
            continue
        seen.add(key)
        title = comp.label
        tree = build_block_node(
            attr_name=comp.attr_name,
            class_name=comp.class_name,
            registry=registry,
            basic_ops=basic_ops,
            details=list(comp.details),
            forward_order=comp.forward_order,
            infer_init_steps=infer_init_steps,
        )
        if is_method_wrapper(tree):
            continue
        cls_info = registry.get(comp.class_name)
        tree.input_label = (
            cls_info.forward_input_name
            if cls_info and cls_info.forward_input_name
            else "hidden_states"
        )
        tree.input_source = input_sources.get(comp.attr_name)
        trees.append((title, tree))

    return trees


def build_full_detailed_block_trees(
    *,
    components: list[BlockComponent],
    registry: dict[str, ClassStructure],
    basic_ops: BasicOpFilter,
    positional_encoding: str,
    norm_type: str,
    decoder_class: str | None = None,
    stack_pre: list[BlockComponent] | None = None,
    stack_tail: list[BlockComponent] | None = None,
    partition: bool = True,
    include_norms: bool = False,
    infer_init_steps: bool = False,
) -> list[tuple[str, BlockNode]]:
    """Build pipeline, decoder-layer, and LM head detail sections in main-diagram order."""
    pipeline = build_pipeline_block_trees(
        stack_pre=stack_pre or [],
        registry=registry,
        basic_ops=basic_ops,
        include_norms=include_norms,
        infer_init_steps=infer_init_steps,
    )
    decoder_trees: list[tuple[str, BlockNode]] = []
    if components:
        decoder_trees = build_decoder_block_trees(
            components,
            registry,
            basic_ops,
            decoder_class=decoder_class,
            include_norms=include_norms,
            infer_init_steps=infer_init_steps,
        )
    head_trees = build_head_block_trees(
        stack_tail=stack_tail or [],
        registry=registry,
        basic_ops=basic_ops,
        include_norms=include_norms,
        infer_init_steps=infer_init_steps,
    )
    trees = pipeline + decoder_trees + head_trees
    if partition:
        return partition_detail_trees(trees)
    return trees


def collect_graph_segments(
    nodes: list[BlockNode],
    norm_before: list[str],
    *,
    use_residual: bool,
) -> list[tuple[str, ...]]:
    """Split ordered nodes into sequential steps and norm→module (+residual) sublayers."""
    segments: list[tuple[str, ...]] = []
    pending_norm: BlockNode | None = None
    norm_pairs = set(norm_before)

    for node in nodes:
        if node.role == "norm":
            pending_norm = node
            continue

        if use_residual and pending_norm is not None and node.attr_name in norm_pairs:
            segments.append(("sublayer", pending_norm, node))
            pending_norm = None
            continue

        if pending_norm is not None:
            segments.append(("seq", pending_norm))
            pending_norm = None
        segments.append(("seq", node))

    if pending_norm is not None:
        segments.append(("seq", pending_norm))
    return segments


def components_from_registry(
    class_name: str,
    registry: dict[str, ClassStructure],
) -> list[BlockComponent]:
    """Rebuild BlockComponent list for a class in the registry."""
    cls = registry.get(class_name)
    if cls is None:
        return []
    return _build_components(cls)
