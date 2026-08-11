"""Recursive block trees for detailed architecture diagrams."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from visualizer.ast_analyze import (
    SYNTHETIC_ATTENTION,
    SYNTHETIC_FUNCTIONAL_LINEAR,
    SYNTHETIC_GATE_ACTIVATION,
    SYNTHETIC_ROUTER_ACTIVATION,
    SYNTHETIC_ROUTER_BIAS,
    SYNTHETIC_ROUTER_GATHER,
    SYNTHETIC_ROUTER_GROUP,
    SYNTHETIC_ROUTER_RENORM,
    SYNTHETIC_ROUTER_SCALE,
    SYNTHETIC_ROUTER_TOPK,
    _ROUTER_SYNTHETICS,
    ClassStructure,
    SideInputSpec,
    attention_kernel_details,
    attention_kernel_label,
    displays_as_linear,
    expand_conditional_block_components,
    _build_components,
    _classify_role,
    _label_for,
)
from visualizer.basic_ops import BasicOpFilter
from visualizer.blocks import BlockComponent, norm_input_sources

_SKIP_INIT_CLASS_NAMES = frozenset({"Parameter", "getattr"})


def is_method_wrapper(node: BlockNode) -> bool:
    """True for forward steps that call a helper method with no submodule internals."""
    if node.children or node.attr_name == SYNTHETIC_ATTENTION:
        return False
    return bool(node.details) and node.details[0].startswith("method `")


def wrapper_bullet_lines(node: BlockNode) -> tuple[str, str]:
    """Return display label and method name for wrapped-module bullets."""
    attr = node.attr_name
    if _is_output_gate_node(node) or _looks_like_output_gate_attr(attr):
        return _parallel_side_port_label(node), attr
    if displays_as_linear(attr, node.class_name):
        return "Linear", attr

    label = node.label.strip()
    if label in {"", attr, node.class_name}:
        label = attr.strip("_").replace("_", " ")
    else:
        label = label.lstrip("_")
    return label, attr


def wrapper_bullet(node: BlockNode) -> str:
    """Human-readable bullet text for a method wrapper."""
    label, attr = wrapper_bullet_lines(node)
    return f"{label} ({attr})"


_SKIP_WRAPPER_COMMENT_ATTRS = frozenset({"tokenization", "embed_tokens"})


def wrapper_skips_comment(node: BlockNode) -> bool:
    """True when the wrapped-module panel should omit the descriptive comment line."""
    attr = node.attr_name.lower()
    if attr in _SKIP_WRAPPER_COMMENT_ATTRS:
        return True
    if node.role == "attention" or attr in {"self_attn", "self_attention", "attn", "attention"}:
        return True
    if "residual" in attr:
        return True
    return any("residual" in detail.lower() for detail in node.details)


def block_purpose(node: BlockNode) -> str | None:
    """Generic one-line description of what a block computes."""
    if node.class_name == "AttentionOp" or node.attr_name == SYNTHETIC_ATTENTION:
        if any("delta rule" in detail.lower() for detail in node.details):
            return "\n".join(detail for detail in node.details if not detail.startswith("kernel:"))

    if node.class_name == "OutputGate" or (node.role == "gate" and node.details):
        if node.details:
            return "\n".join(node.details)

    for detail in node.details:
        cleaned = detail.strip()
        if cleaned and not cleaned.startswith("method `") and not cleaned.startswith("kernel:"):
            return cleaned

    class_name = node.class_name or ""
    role = node.role
    attr = node.attr_name.lower()

    if class_name == "OutputGate" or role == "gate":
        return "Output gate — scales normalized output"
    if class_name == "SituAndMul":
        return "SiLU(gate) × up branch"
    if class_name == "FusedRMSNormGated":
        return "RMSNorm × gated activation"
    if class_name == "Split" or node.attr_name == "split_gate_up":
        return "Split fused gate/up projection"
    if class_name in {"ActivationOp", "SituActivation"}:
        return f"Apply {node.label} to gate half"
    if class_name == "Multiply" or node.label in {"×", "Elementwise ×"}:
        return "Multiply gate and up activations"
    if role == "head" or attr == "lm_head":
        return "Project to vocabulary logits"
    if role == "router" or attr in {"gate", "router"}:
        return "Score and route tokens to experts"
    if role == "moe" or "moe" in attr or "expert" in attr:
        return "Dispatch tokens to experts and combine outputs"
    if role == "ffn":
        return "Position-wise feed-forward transform"
    if role == "norm":
        return "Normalize activations"
    if role == "embedding" or "embedding" in class_name.lower():
        return "Map token IDs to embeddings"
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
    if sub_step.attr_name == SYNTHETIC_FUNCTIONAL_LINEAR or displays_as_linear(
        sub_step.attr_name, sub_step.class_name
    ):
        return "Linear"
    return sub_step.label


def collect_function_steps(node: BlockNode) -> list[BlockNode]:
    """Collect leaf/basic/method-wrapper nodes that represent one forward op."""
    if is_method_wrapper(node) or (node.is_basic and not node.children):
        return [node]
    steps: list[BlockNode] = []
    for child in node.children:
        steps.extend(collect_function_steps(child))
    return steps


_PIPELINE_WRAPPER_ATTRS = frozenset({"tokenization", "tokenizer"})


def _is_pipeline_wrapper(node: BlockNode) -> bool:
    """True for embed/head/pipeline modules that stay as spine tiles, not inlined subgraphs."""
    if node.attr_name.lower() in _PIPELINE_WRAPPER_ATTRS:
        return True
    return node.role in {"embedding", "head"}


def is_straight_line_module(node: BlockNode) -> bool:
    """True when a composite block is a simple straight-line pipeline with no branching."""
    if not _is_composite_block(node):
        return False
    if _is_pipeline_wrapper(node):
        return False
    segments = collect_computation_segments(node)
    if not segments:
        return False
    return all(isinstance(segment, SeqSegment) for segment in segments)


def is_linear_pipeline_block(node: BlockNode) -> bool:
    """Alias for :func:`is_straight_line_module`."""
    return is_straight_line_module(node)


def should_expand_composite_wrapper(node: BlockNode) -> bool:
    """True when a composite wrapper should expand inline (straight-line modules only)."""
    return is_straight_line_module(node)


def straight_line_steps(node: BlockNode) -> list[BlockNode]:
    """Expand a straight-line composite into ordered steps, recursively inlining nested modules."""
    if not is_straight_line_module(node):
        return [node]
    steps: list[BlockNode] = []
    for segment in collect_computation_segments(node):
        if not isinstance(segment, SeqSegment):
            return [node]
        child = segment.step
        if is_straight_line_module(child):
            steps.extend(straight_line_steps(child))
        else:
            steps.append(child)
    return steps


def linear_pipeline_steps(node: BlockNode) -> list[BlockNode]:
    """Alias for :func:`straight_line_steps`."""
    return straight_line_steps(node)


def inline_block_frame_label(block: BlockNode) -> str:
    """Display label for a dotted inline frame around an expanded sub-block."""
    return f"{block.label} ({block.attr_name})"


def inline_block_frame_sublabel(block: BlockNode) -> str | None:
    """Optional purpose line shown under an inline frame label."""
    return block_purpose(block)


def is_single_function_tree(node: BlockNode) -> bool:
    """True when a block tree only contains one forward operation worth demoting to the panel."""
    if should_expand_composite_wrapper(node):
        return False
    return len(collect_function_steps(node)) == 1


def inline_composite_steps(step: BlockNode) -> tuple[list[BlockNode], BlockNode | None]:
    """Inline a straight-line composite wrapper into its internal forward steps."""
    if not is_straight_line_module(step):
        return [step], None
    inner_steps = straight_line_steps(step)
    if inner_steps and (len(inner_steps) > 1 or inner_steps[0] is not step):
        return inner_steps, step
    return [step], None


def _router_synthetic_label(attr_name: str, details: list[str]) -> str:
    if details:
        return details[0]
    defaults = {
        SYNTHETIC_ROUTER_ACTIVATION: "Score activation",
        SYNTHETIC_ROUTER_BIAS: "Expert bias",
        SYNTHETIC_ROUTER_GROUP: "Group routing",
        SYNTHETIC_ROUTER_TOPK: "Top-k experts",
        SYNTHETIC_ROUTER_GATHER: "Gather weights",
        SYNTHETIC_ROUTER_RENORM: "Renormalize",
        SYNTHETIC_ROUTER_SCALE: "Route scaling",
    }
    return defaults.get(attr_name, attr_name.strip("@").replace("_", " "))


_OMIT_DETAILED_TREES = frozenset({"tokenization", "tokenizer"})


def _omit_from_detailed_view(tree: BlockNode) -> bool:
    return tree.attr_name in _OMIT_DETAILED_TREES


def _show_single_function_in_diagram(tree: BlockNode) -> bool:
    """True when a one-op module should render as a diagram tile instead of the panel."""
    return displays_as_linear(tree.attr_name, tree.class_name)


def partition_detail_trees(
    trees: list[tuple[str, BlockNode]],
    wrappers: list[BlockNode],
) -> tuple[list[tuple[str, BlockNode]], list[BlockNode]]:
    """Move single-function block trees into the wrapped-modules list."""
    kept: list[tuple[str, BlockNode]] = []
    wrapped = [node for node in wrappers if not _omit_from_detailed_view(node)]
    seen = {node.attr_name for node in wrapped}
    for title, tree in trees:
        if _omit_from_detailed_view(tree):
            continue
        if is_single_function_tree(tree):
            if _show_single_function_in_diagram(tree):
                kept.append((title, tree))
                continue
            if tree.attr_name not in seen:
                seen.add(tree.attr_name)
                wrapped.append(tree)
            continue
        kept.append((title, tree))
    return kept, wrapped


def collect_method_wrappers(node: BlockNode) -> list[BlockNode]:
    """Collect method-wrapper leaves under a block tree."""
    if is_method_wrapper(node):
        return [node]
    wrappers: list[BlockNode] = []
    for child in node.children:
        wrappers.extend(collect_method_wrappers(child))
    return wrappers


def _looks_like_output_gate_attr(attr_name: str) -> bool:
    """True for common output-gate attribute names (e.g. g_proj, out_gate)."""
    attr = attr_name.strip("_").lower()
    if attr in {"gate", "router"}:
        return False
    return bool(re.search(r"g_proj|out_gate|output_gate|_gate$|^gate_", attr))


def _is_output_gate_node(node: BlockNode) -> bool:
    """True for parallel output-gate modules (e.g. g_proj wrapped as OutputGate)."""
    return node.class_name == "OutputGate" or node.role == "gate"


def collect_parallel_gate_wrappers(node: BlockNode, *, inside_output_gate: bool = False) -> list[BlockNode]:
    """Collect parallel gate side branches referenced inline but listed in the panel."""
    wrappers: list[BlockNode] = []
    gate_attrs = set(node.parallel_gates)
    for child in node.children:
        in_gate = inside_output_gate or _is_output_gate_node(child)
        if not in_gate and child.attr_name in gate_attrs:
            if not _is_composite_block(child):
                wrappers.append(child)
        wrappers.extend(collect_parallel_gate_wrappers(child, inside_output_gate=in_gate))
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
    side_inputs: dict[str, list[SideInputSpec]] = field(default_factory=dict)
    input_label: str | None = None
    input_source: str | None = None


PortStyle = Literal["floating", "inline"]
SegmentSource = Literal["forward_input"]


@dataclass
class Branch:
    """One parallel path feeding a fan-in node."""

    label: str
    steps: list[BlockNode]
    port_style: PortStyle = "floating"

    @property
    def port_label(self) -> str:
        if self.label == "KV":
            return "K/V"
        return self.label


@dataclass
class SeqSegment:
    step: BlockNode


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


@dataclass
class SideCombineSegment:
    """Side-fed combine operator replacing a method wrapper (e.g. moe infer Σ)."""

    consumer: BlockNode
    sides: list[SideInputSpec]
    op: str
    op_sublabel: str | None = None


@dataclass
class ResidualAddSegment:
    """Parallel module on a saved residual, merged with the main path via +."""

    module: BlockNode
    sides: list[SideInputSpec]


ComputationSegment = (
    SeqSegment | FanOutSegment | CombineSegment | SideFeedSegment | SideCombineSegment | ResidualAddSegment
)


def _method_combine_op(step: BlockNode) -> tuple[str, str | None]:
    """Map a method wrapper to an operator symbol and short computation label."""
    if step.attr_name == "moe_infer":
        return "Σ", "∑ w·expert"
    label, _attr = wrapper_bullet_lines(step)
    return "ƒ", label


def _segment_for_step(node: BlockNode, step: BlockNode) -> ComputationSegment:
    side_specs = node.side_inputs.get(step.attr_name, [])
    if not side_specs:
        return SeqSegment(step=step)

    has_prior_side = any(side.source_kind == "prior_step" for side in side_specs)
    has_residual_side = any(side.source_kind == "forward_input" for side in side_specs)
    side_producer_nodes = {
        side.source_chain[-1]: child
        for side in side_specs
        if side.source_kind == "prior_step" and side.source_chain
        for child in node.children
        if child.attr_name == side.source_chain[-1]
    }

    if is_method_wrapper(step) and has_prior_side:
        op, sublabel = _method_combine_op(step)
        return SideCombineSegment(
            consumer=step,
            sides=list(side_specs),
            op=op,
            op_sublabel=sublabel,
        )

    if has_residual_side and not has_prior_side:
        return ResidualAddSegment(module=step, sides=list(side_specs))

    if is_method_wrapper(step):
        op, sublabel = _method_combine_op(step)
        return SideCombineSegment(
            consumer=step,
            sides=list(side_specs),
            op=op,
            op_sublabel=sublabel,
        )

    return SideFeedSegment(
        consumer=step,
        sides=list(side_specs),
        side_producer_nodes=side_producer_nodes,
    )


def _label_for_call(attr_name: str, class_name: str | None) -> str:
    if attr_name == SYNTHETIC_ATTENTION:
        return "Attention kernel"
    if displays_as_linear(attr_name, class_name):
        return "Linear"
    if class_name:
        role = _classify_role(attr_name, class_name)
        return _label_for(role, class_name, attr_name)
    readable = attr_name.replace("_", " ")
    return readable[:24]


def _leaf_node(
    *,
    attr_name: str,
    class_name: str,
    forward_order: int | None,
    details: list[str] | None = None,
    basic: bool = True,
    label: str | None = None,
) -> BlockNode:
    role = _classify_role(attr_name, class_name)
    return BlockNode(
        attr_name=attr_name,
        class_name=class_name,
        role=role,
        label=label or _label_for_call(attr_name, class_name),
        forward_order=forward_order,
        details=list(details or []),
        is_basic=basic,
    )


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

    lines: list[str] = [f"g = {gate_attr}(hidden_states)"]

    if consumer_class == "FusedRMSNormGated" or (consumer and "Gated" in (consumer_class or "")):
        lines.append("reshape → [num_heads, head_dim]")
        if inline_activation:
            lines.append(f"g = {inline_activation}(g)")
            lines.append(f"norm(attn_out) × g → {consumer or 'o_norm'}")
        else:
            activation = norm_gate_activation or "σ"
            lines.append(f"{activation}(g) inside {consumer or 'o_norm'}")
            lines.append("norm(attn_out) × gate")
    elif inline_activation:
        lines.append(f"g = {inline_activation}(g)")
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


def _output_gate_block_node(
    linear_step: BlockNode,
    activation: str | None,
    details: list[str] | None = None,
) -> BlockNode:
    """Expand a parallel output gate into its own nested sub-diagram."""
    gate_details = list(details or [])
    linear_step = _leaf_node(
        attr_name=linear_step.attr_name,
        class_name=linear_step.class_name,
        forward_order=linear_step.forward_order,
        label=linear_step.label,
        details=gate_details[:1] if gate_details else ["Linear projection from hidden_states"],
    )
    children: list[BlockNode] = [linear_step]
    if activation:
        children.append(
            _leaf_node(
                attr_name=SYNTHETIC_GATE_ACTIVATION,
                class_name="ActivationOp",
                forward_order=(linear_step.forward_order or 0) + 1,
                label=activation,
                details=[f"g = {activation}(g)"],
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
    for child in child_nodes:
        if child.attr_name not in gate_attrs:
            wrapped.append(child)
            continue
        consumer, _ = _gate_side_consumer(side_inputs, child.attr_name)
        consumer_node = next((node for node in child_nodes if node.attr_name == consumer), None)
        details = _output_gate_details(
            child.attr_name,
            side_inputs=side_inputs,
            gate_activations=gate_activations,
            consumer_class=consumer_node.class_name if consumer_node else None,
            norm_gate_activation=gated_norm_activation(consumer_node) if consumer_node else None,
        )
        wrapped.append(
            _output_gate_block_node(
                child,
                gate_activations.get(child.attr_name),
                details=details,
            )
        )
    return wrapped


def side_producer_has_activation(producer: BlockNode) -> bool:
    """True when an output-gate side producer already ends with an activation step."""
    if producer.class_name == "OutputGate":
        return any(child.attr_name == SYNTHETIC_GATE_ACTIVATION for child in producer.children)
    return False


def is_gated_norm_module(node: BlockNode) -> bool:
    """True for fused RMS/Layer norms that combine normalization with a gate input."""
    class_name = node.class_name or ""
    if class_name == "FusedRMSNormGated":
        return True
    return node.role == "norm" and bool(re.search(r"Fused.*Gated|Gated.*Norm", class_name, re.I))


def gated_norm_activation(node: BlockNode) -> str | None:
    """Return the gate activation applied inside a gated norm module, if known."""
    known = {"Sigmoid", "SiLU", "GELU", "Tanh", "ReLU", "Silu", "Gelu"}
    for detail in node.details:
        if detail in known:
            return detail
    if node.class_name == "FusedRMSNormGated":
        return "Sigmoid"
    return None


def _situ_and_mul_block_node(
    *,
    attr_name: str,
    forward_order: int | None,
    details: list[str] | None = None,
    role: str = "ffn",
) -> BlockNode:
    """Expand SituAndMul into a small internal pipeline for its own nested diagram."""
    return BlockNode(
        attr_name=attr_name,
        class_name="SituAndMul",
        role=role,
        label="Gated multiply",
        forward_order=forward_order,
        details=list(details or ["SiLU(gate) × up branch"]),
        is_basic=False,
        input_label="gate_up",
        children=[
            _leaf_node(
                attr_name="split_gate_up",
                class_name="Split",
                forward_order=0,
                label="Split gate | up",
                details=["split fused projection"],
            ),
            _leaf_node(
                attr_name="situ_activation",
                class_name="SituActivation",
                forward_order=1,
                label="SiLU",
                details=["activation on gate half"],
            ),
            _leaf_node(
                attr_name="elementwise_mul",
                class_name="Multiply",
                forward_order=2,
                label="×",
                details=["gate × up"],
            ),
        ],
    )


def _nested_input_source(parent: BlockNode, child: BlockNode) -> str:
    """Describe where a nested block's primary input comes from."""
    if child.class_name == "KimiMLP" and parent.class_name == "KimiSparseMoeBlock":
        return "Linear in KimiSparseMoeBlock"
    if child.class_name == "SituAndMul":
        return "gate_up in KimiMLP"
    if child.input_label and child.input_label not in {"hidden_states", "x"}:
        return f"{child.input_label} in {parent.class_name}"
    return f"{parent.class_name}"


def _branches_from_provenance(
    pre_merge: list[BlockNode],
    provenance: dict[str, list[str]],
) -> list[Branch]:
    """Build parallel branches from named provenance chains (e.g. attention Q/K/V inputs)."""
    by_attr = {node.attr_name: node for node in pre_merge}
    branches: list[Branch] = []

    for label in provenance:
        chain = provenance.get(label, [])
        if not chain:
            continue
        nodes = [by_attr[attr] for attr in chain if attr in by_attr]
        if nodes:
            branches.append(Branch(label=label, steps=nodes))

    if len(branches) >= 2:
        return _collapse_identical_kv_branches(branches)
    return []


def _collapse_identical_kv_branches(branches: list[Branch]) -> list[Branch]:
    """Merge identical K/V module chains into one shared branch."""
    by_label = {branch.label: branch for branch in branches}
    if "K" not in by_label or "V" not in by_label:
        return branches
    if [node.attr_name for node in by_label["K"].steps] != [node.attr_name for node in by_label["V"].steps]:
        return branches
    collapsed = [branch for branch in branches if branch.label not in {"K", "V"}]
    collapsed.append(Branch(label="KV", steps=by_label["K"].steps))
    return collapsed


def _partition_named_branches(pre_merge: list[BlockNode], prefix_rules: dict[str, str]) -> list[Branch]:
    """Partition steps into labeled branches using attribute-name prefix rules."""
    buckets: dict[str, list[BlockNode]] = {label: [] for label in prefix_rules.values()}
    for node in pre_merge:
        lower = node.attr_name.lower()
        for prefixes, label in prefix_rules.items():
            if any(lower.startswith(prefix) or lower == prefix for prefix in prefixes.split("|")):
                buckets[label].append(node)
                break

    branches: list[Branch] = []
    if buckets.get("K") and not buckets.get("V") and buckets.get("Q"):
        branches.append(Branch(label="Q", steps=buckets["Q"]))
        branches.append(Branch(label="KV", steps=list(buckets["K"])))
    elif buckets.get("K") and buckets.get("V") and [n.attr_name for n in buckets["K"]] == [n.attr_name for n in buckets["V"]]:
        if buckets.get("Q"):
            branches.append(Branch(label="Q", steps=buckets["Q"]))
        branches.append(Branch(label="KV", steps=list(buckets["K"])))
    else:
        for label, nodes in buckets.items():
            if nodes:
                branches.append(Branch(label=label, steps=nodes))
    return branches


def _name_prefix_branch_rules() -> dict[str, str]:
    return {
        "q_|q_proj|query": "Q",
        "kv_|k_|k_proj|key": "K",
        "v_|v_proj|value": "V",
    }


def _parallel_side_port_label(side: BlockNode) -> str:
    """Readable inline port label for a parallel side branch."""
    if _is_output_gate_node(side) or _looks_like_output_gate_attr(side.attr_name):
        return "gate"
    return _label_for_call(side.attr_name, side.class_name)


def _side_feed_targets(node: BlockNode) -> dict[str, str]:
    """Map side-producer attr -> consumer attr for prior-step side feeds."""
    targets: dict[str, str] = {}
    for consumer_name, specs in node.side_inputs.items():
        for spec in specs:
            if spec.source_kind != "prior_step" or not spec.source_chain:
                continue
            targets[spec.source_chain[-1]] = consumer_name
    return targets


def collect_computation_segments(node: BlockNode) -> list[ComputationSegment]:
    """Build generic render segments from forward-ordered block children."""
    children = node.children
    if not children:
        return []

    merge_idx = next(
        (i for i, child in enumerate(children) if child.attr_name == SYNTHETIC_ATTENTION),
        None,
    )
    if merge_idx is None:
        return [_segment_for_step(node, child) for child in children]

    pre_merge = children[:merge_idx]
    merge_node = children[merge_idx]
    post_merge = children[merge_idx + 1 :]

    branches = _branches_from_provenance(pre_merge, node.attention_inputs)
    if len(branches) < 2:
        branches = [
            Branch(label=branch.label, steps=branch.steps)
            for branch in _partition_named_branches(pre_merge, _name_prefix_branch_rules())
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
    skip_sequential: set[str] = set()
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
    return node is not None and not node.is_basic and bool(node.children) and not is_method_wrapper(node)


def collect_nested_diagrams(root: BlockNode) -> list[tuple[str, BlockNode]]:
    """Collect composite blocks referenced in a diagram for separate sub-diagrams."""
    from visualizer.computation_graph import build_computation_graph

    seen: set[str] = set()
    ordered: list[tuple[str, BlockNode]] = []

    def consider(block: BlockNode | None, parent_block: BlockNode | None) -> None:
        if block is None or not _is_composite_block(block):
            return
        if is_straight_line_module(block):
            return
        if block.attr_name in seen:
            return
        seen.add(block.attr_name)
        if block.input_source is None and parent_block is not None:
            block.input_source = _nested_input_source(parent_block, block)
        ordered.append((f"{block.label} ({block.attr_name})", block))
        inner = build_computation_graph(block)
        for spec in inner.nodes:
            consider(spec.block, block)

    graph = build_computation_graph(root)
    for spec in graph.nodes:
        consider(spec.block, root)

    return ordered


def flatten_computation_segments(node: BlockNode) -> list[ComputationSegment]:
    """Prepare top-level segments for graph layout, keeping composite blocks intact."""
    return collect_computation_segments(node)


def build_block_node(
    *,
    attr_name: str,
    class_name: str,
    registry: dict[str, ClassStructure],
    basic_ops: BasicOpFilter,
    visited: frozenset[str] | None = None,
    details: list[str] | None = None,
    forward_order: int | None = None,
) -> BlockNode:
    """Expand one submodule into a recursive block tree using forward-pass order."""
    visited = visited or frozenset()
    role = _classify_role(attr_name, class_name)
    label = _label_for_call(attr_name, class_name)

    if attr_name == SYNTHETIC_ATTENTION:
        step_details = list(details or [])
        return _leaf_node(
            attr_name=attr_name,
            class_name="AttentionOp",
            forward_order=forward_order,
            label=attention_kernel_label(step_details),
            details=attention_kernel_details(step_details),
        )

    if basic_ops.is_basic(class_name, attr_name):
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
    forward_steps = [step for step in cls.forward_calls if step not in _SKIP_INIT_CLASS_NAMES]

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

    order_map = {name: idx for idx, name in enumerate(cls.forward_calls)}
    child_nodes: list[BlockNode] = []

    for index, call_attr in enumerate(forward_steps):
        child_order = order_map.get(call_attr, index)
        child_details = cls.forward_step_details.get(call_attr) or cls.init_details.get(call_attr, [])

        if call_attr == SYNTHETIC_ATTENTION:
            child_nodes.append(
                _leaf_node(
                    attr_name=call_attr,
                    class_name="AttentionOp",
                    forward_order=child_order,
                    label=attention_kernel_label(child_details),
                    details=attention_kernel_details(child_details, cls.attention_inputs),
                )
            )
            continue

        if call_attr == SYNTHETIC_FUNCTIONAL_LINEAR:
            child_nodes.append(
                _leaf_node(
                    attr_name=call_attr,
                    class_name="Linear",
                    forward_order=child_order,
                    details=["F.linear(...)"],
                )
            )
            continue

        if call_attr in _ROUTER_SYNTHETICS:
            child_nodes.append(
                _leaf_node(
                    attr_name=call_attr,
                    class_name="RouterOp",
                    forward_order=child_order,
                    details=child_details,
                    label=_router_synthetic_label(call_attr, child_details),
                )
            )
            continue

        child_class = cls.init_assignments.get(call_attr)
        if child_class is None or child_class in _SKIP_INIT_CLASS_NAMES:
            if child_class in _SKIP_INIT_CLASS_NAMES:
                continue
            child_nodes.append(
                _leaf_node(
                    attr_name=call_attr,
                    class_name=call_attr,
                    forward_order=child_order,
                    details=[f"method `{call_attr}()`"],
                )
            )
            continue

        if child_class == "SituAndMul":
            child_nodes.append(
                _situ_and_mul_block_node(
                    attr_name=call_attr,
                    forward_order=child_order,
                    details=child_details,
                    role=role,
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
            )
        )

    child_nodes = _wrap_parallel_gate_children(
        child_nodes,
        cls.parallel_gates,
        cls.gate_activations,
        cls.side_inputs,
    )

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
        attention_inputs=dict(cls.attention_inputs),
        parallel_gates=list(cls.parallel_gates),
        side_inputs=dict(cls.side_inputs),
    )


def _input_sources_for_components(components: list[BlockComponent]) -> dict[str, str]:
    """Map compute module attr names to the norm label that feeds them in the decoder layer."""
    return norm_input_sources(components)


def fallback_positional_tree(positional_encoding: str) -> BlockNode:
    """Fallback positional pipeline when AST does not declare a rotary module."""
    return BlockNode(
        attr_name="rotary_emb",
        class_name="RotaryEmbedding",
        role="positional",
        label=positional_encoding,
        children=[
            BlockNode(
                attr_name="freqs",
                class_name="RotaryEmbedding",
                role="other",
                label="Freq computation",
                is_basic=True,
            ),
            BlockNode(
                attr_name="apply_rotary",
                class_name="ApplyRotary",
                role="other",
                label="Apply to Q/K",
                is_basic=True,
            ),
        ],
    )


_synthetic_rope_tree = fallback_positional_tree


def build_positional_block_tree(
    component: BlockComponent,
    registry: dict[str, ClassStructure],
    basic_ops: BasicOpFilter,
) -> BlockNode:
    """Build a positional block tree from AST or config fallback."""
    if component.class_name in registry:
        return build_block_node(
            attr_name=component.attr_name,
            class_name=component.class_name,
            registry=registry,
            basic_ops=basic_ops,
            details=list(component.details),
        )
    return fallback_positional_tree(component.label)


def build_stack_component_tree(
    component: BlockComponent,
    registry: dict[str, ClassStructure],
    basic_ops: BasicOpFilter,
) -> BlockNode:
    """Build a block tree for one main-diagram stack component."""
    if component.role == "positional":
        return build_positional_block_tree(component, registry, basic_ops)
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
        is_basic=True,
    )


def spine_expanded_frame_label(component: BlockComponent, *, positional_encoding: str) -> str:
    """Dotted-frame title for a straight-line module expanded on the main spine."""
    if component.role == "positional":
        return f"Positional ({positional_encoding}) ({component.attr_name})"
    return f"{component.label} ({component.attr_name})"


def build_pipeline_block_trees(
    *,
    positional_encoding: str,
    registry: dict[str, ClassStructure],
    basic_ops: BasicOpFilter,
) -> list[tuple[str, BlockNode]]:
    """Build pre-decoder detail sections matching the main diagram stack."""
    return []


def build_head_block_trees(
    *,
    norm_type: str,
    registry: dict[str, ClassStructure],
    basic_ops: BasicOpFilter,
) -> list[tuple[str, BlockNode]]:
    """Build post-decoder LM head detail section."""
    return []


def build_decoder_block_trees(
    components: list[BlockComponent],
    registry: dict[str, ClassStructure],
    basic_ops: BasicOpFilter,
    *,
    decoder_class: str | None = None,
) -> tuple[list[tuple[str, BlockNode]], list[BlockNode]]:
    """Build recursive trees for detailed diagrams and collect standalone method wrappers."""
    trees: list[tuple[str, BlockNode]] = []
    standalone_wrappers: list[BlockNode] = []
    seen: set[tuple[str, str]] = set()
    decoder = registry.get(decoder_class) if decoder_class else None
    detail_components = (
        expand_conditional_block_components(decoder, components)
        if decoder is not None
        else list(components)
    )
    input_sources = _input_sources_for_components(detail_components)

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
        if comp.role == "norm":
            continue
        if comp.forward_order is None:
            continue
        key = (comp.attr_name, comp.class_name)
        if key in seen:
            continue
        seen.add(key)
        title = f"{comp.label} ({comp.attr_name})"
        tree = build_block_node(
            attr_name=comp.attr_name,
            class_name=comp.class_name,
            registry=registry,
            basic_ops=basic_ops,
            details=list(comp.details),
            forward_order=comp.forward_order,
        )
        if is_method_wrapper(tree):
            standalone_wrappers.append(tree)
            continue
        tree.input_label = "hidden_states"
        tree.input_source = input_sources.get(comp.attr_name)
        trees.append((title, tree))

    return trees, standalone_wrappers


def build_full_detailed_block_trees(
    *,
    components: list[BlockComponent],
    registry: dict[str, ClassStructure],
    basic_ops: BasicOpFilter,
    positional_encoding: str,
    norm_type: str,
    decoder_class: str | None = None,
) -> tuple[list[tuple[str, BlockNode]], list[BlockNode]]:
    """Build pipeline, decoder-layer, and LM head detail sections in main-diagram order."""
    pipeline = build_pipeline_block_trees(
        positional_encoding=positional_encoding,
        registry=registry,
        basic_ops=basic_ops,
    )
    decoder_trees: list[tuple[str, BlockNode]] = []
    wrappers: list[BlockNode] = []
    if components:
        decoder_trees, wrappers = build_decoder_block_trees(
            components,
            registry,
            basic_ops,
            decoder_class=decoder_class,
        )
    head_trees = build_head_block_trees(
        norm_type=norm_type,
        registry=registry,
        basic_ops=basic_ops,
    )
    return partition_detail_trees(pipeline + decoder_trees + head_trees, wrappers)


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
