###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regex filters for treating modeled blocks as leaf/basic operations."""

from __future__ import annotations

import ast
import re

# Default leaf patterns: PyTorch ATen ops (avoid matching "Attention", etc.).
DEFAULT_BASIC_OP_PATTERNS: tuple[str, ...] = (
    r"(?i)aten\.",
    r"(?i)\.aten\.",
    r"(?i)^aten_",
)

# Common PyTorch leaf modules (not enabled by default; add via --basic-op-add).
COMMON_LEAF_PATTERNS: tuple[str, ...] = (
    r"(?i)^Linear$",
    r"(?i)^Embedding$",
    r"(?i)^Conv\d*d$",
    r"(?i)^Dropout$",
    r"(?i)^Identity$",
    r"(?i)^Parameter$",
)

# Module classes with a direct, shape-only TraceLens.PerfModel formula (GEMM, CONV,
# Normalization, Softmax). Used to stop expansion exactly at the granularity a
# roofline projection can cost -- see BasicOpFilter.for_roofline().
# A grouped-expert MoE is the same kind of stop (GroupedGemm costs the whole
# expert loop) but it is not a name match: see forward_is_grouped_expert_kernel.
ROOFLINE_LEAF_PATTERNS: tuple[str, ...] = (
    r"(?i)^Linear$",
    r"(?i)^Embedding$",
    r"(?i)^Dropout$",
    r"(?i)^Identity$",
    r"(?i)^Parameter$",
    r"(?i)^Conv\d*d$",
    r"(?i)^RMSNorm$",
    r"(?i)^LayerNorm$",
    r"(?i)^GroupNorm$",
    r"(?i)^BatchNorm\d*d$",
    r"(?i)^InstanceNorm\d*d$",
    r"(?i)^Softmax$",
)

# CamelCase tokens of a routed-expert module. Case-sensitive so an attribute
# such as ``shared_experts`` is not mistaken for the class.
_GROUPED_MOE_CLASS_RE = re.compile(r"MoE|Moe|Experts")


def forward_is_grouped_expert_kernel(
    class_name: str, class_node: ast.AST | None
) -> bool:
    """True when *class_name*'s own ``forward`` is the grouped expert GEMM.

    That forward is one PerfModel op (:class:`GroupedGemm` over the selected
    experts), so a roofline projection stops on it instead of opening the loop
    into a GEMM per expert. Two near misses stay open:

    - a MoE container whose children are already leaves (a router ``Linear``
      and a shared expert) — those children are the ops with a formula;
    - one expert MLP invoked from an outer ``for`` — the loop multiplies those
      GEMMs, and stopping here would charge the full grouped formula once per
      expert.

    A shared-expert class is the dense MLP, even when its name says expert.
    """
    if not class_name or class_node is None:
        return False
    if "shared" in class_name.lower():
        return False
    if _GROUPED_MOE_CLASS_RE.search(class_name) is None:
        return False
    forward = next(
        (
            item
            for item in getattr(class_node, "body", ())
            if isinstance(item, ast.FunctionDef) and item.name == "forward"
        ),
        None,
    )
    if forward is None:
        return False
    return any(isinstance(node, ast.For) for node in ast.walk(forward))


# Submodule attribute names observed in modeling forwards (conv paths, etc.).
_MODELING_ATTR_PATTERNS: tuple[str, ...] = (
    r"(?i)_conv1d$",
    r"(?i)_conv2d$",
    r"(?i)^conv1d$",
)

# Parallel output-gate attrs detected by AST (`g_proj`, etc.) — not MoE routers.
_OUTPUT_GATE_ATTR_PATTERNS: tuple[str, ...] = (
    r"(?i)g_proj",
    r"(?i)out_gate",
    r"(?i)output_gate",
    r"(?i)_gate$",
)


def _is_functional_synthetic_basic(attr_name: str) -> bool:
    return attr_name.startswith("@functional_")


class BasicOpFilter:
    """Match class or attribute names against basic-operation regex patterns."""

    def __init__(
        self,
        patterns: list[str | re.Pattern[str]],
        *,
        basic_only: bool = False,
        stops_at_grouped_moe: bool = False,
    ) -> None:
        compiled: list[re.Pattern[str]] = []
        for pattern in patterns:
            compiled.append(
                pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)
            )
        self.patterns = compiled
        self.basic_only = basic_only
        # GroupedGemm prices a routed-expert forward as one op. Set on the
        # roofline filter so that forward stays a leaf; detailed diagrams still
        # open it.
        self.stops_at_grouped_moe = stops_at_grouped_moe

    @classmethod
    def from_cli(
        cls,
        *,
        add: list[str] | None = None,
        remove: list[str] | None = None,
    ) -> BasicOpFilter:
        selected = list(DEFAULT_BASIC_OP_PATTERNS)
        for pattern in remove or []:
            selected = [item for item in selected if item != pattern]
        for pattern in add or []:
            if pattern not in selected:
                selected.append(pattern)
        return cls(selected)

    @classmethod
    def for_detailed(cls) -> BasicOpFilter:
        """Leaf patterns for detailed internal diagrams (Linear, norms, etc.)."""
        patterns = list(DEFAULT_BASIC_OP_PATTERNS)
        for pattern in COMMON_LEAF_PATTERNS:
            if pattern == r"(?i)^Embedding$":
                continue
            if pattern not in patterns:
                patterns.append(pattern)
        # Detailed diagrams show only leaf/basic module tiles, not modeled ops.
        norm_patterns = (r"(?i)^RMSNorm$", r"(?i)^LayerNorm$")
        for pattern in norm_patterns:
            if pattern not in patterns:
                patterns.append(pattern)
        return cls(patterns, basic_only=True)

    @classmethod
    def for_roofline(cls) -> BasicOpFilter:
        """Leaf patterns that stop expansion at ops with a TraceLens.PerfModel formula.

        A roofline projection costs a GEMM, a norm, a softmax, ... as one analytical
        op; expanding further (into the mean/rsqrt/mul chain behind an RMSNorm, say)
        only adds primitives the projection has no better formula for anyway.
        A routed-expert forward is the same stop: GroupedGemm costs the loop, so
        the loop is not opened into one GEMM per expert. A kernel call that names
        a perf-model op is that stop too: the collector keeps the call as one row,
        so the call is not opened into the ops inside the kernel.
        """
        patterns = list(DEFAULT_BASIC_OP_PATTERNS)
        for pattern in ROOFLINE_LEAF_PATTERNS:
            if pattern not in patterns:
                patterns.append(pattern)
        return cls(patterns, basic_only=True, stops_at_grouped_moe=True)

    def is_basic(self, *names: str) -> bool:
        for name in names:
            if not name:
                continue
            if any(pattern.search(name) for pattern in self.patterns):
                return True
        return False

    def pattern_strings(self) -> list[str]:
        return [pattern.pattern for pattern in self.patterns]


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    if not name:
        return False
    return any(re.search(pattern, name) for pattern in patterns)


def _detail_implies_modeling(details: list[str] | None) -> bool:
    if not details:
        return False
    for line in details:
        lowered = line.lower()
        if lowered.startswith("kernel:"):
            return True
        if lowered.startswith("ports:"):
            return True
        if line.startswith("method `"):
            return False
    return False


def introspect_is_modeling_operation(
    class_name: str,
    attr_name: str,
    details: list[str] | None = None,
    *,
    parallel_gate: bool = False,
    router_synthetic: bool = False,
) -> bool:
    """Return True when AST/block-tree introspection marks a modeled composite op.

    These render with role colors and expand in diagrams — not as gray basic-op tiles.
    """
    if parallel_gate or router_synthetic:
        return True
    if attr_name.startswith("@"):
        return not _is_functional_synthetic_basic(attr_name)
    if _matches_any(attr_name, _MODELING_ATTR_PATTERNS):
        return True
    if _matches_any(attr_name, _OUTPUT_GATE_ATTR_PATTERNS):
        attr_key = attr_name.strip("_").lower()
        if attr_key not in {"gate", "router"}:
            if class_name and re.match(r"(?i)^Linear$", class_name):
                return False
            return True
    if _detail_implies_modeling(details):
        return True
    # Imported modules referenced in __init__ but not parsed (e.g. fla ShortConvolution).
    if class_name and re.search(
        r"(?i)Conv|Gate|Attention|Merge|Scan|Router|NormGated|Recurrent|Kernel",
        class_name,
    ):
        if not re.match(r"(?i)^Linear$|^Embedding$|^Identity$|^Dropout$", class_name):
            return True
    return False


def resolve_is_basic(
    class_name: str,
    attr_name: str,
    basic_ops: BasicOpFilter,
    *,
    details: list[str] | None = None,
    in_registry: bool = False,
    parallel_gate: bool = False,
    router_synthetic: bool = False,
) -> bool:
    """Decide whether a module renders as a gray basic-op tile."""
    if introspect_is_modeling_operation(
        class_name,
        attr_name,
        details,
        parallel_gate=parallel_gate,
        router_synthetic=router_synthetic,
    ):
        return False
    if basic_ops.is_basic(class_name, attr_name):
        return True
    # Unparsed external modules default to modeled ops, not basic tiles.
    if not in_registry:
        return False
    return False


def show_in_detail_graph(
    block: "BlockNode | None",
    *,
    basic_only: bool,
) -> bool:
    """Return True when a block tree node should appear in a detailed computation graph."""
    from TraceLens.ModelUtils.block_tree import is_method_wrapper

    if not basic_only:
        return True
    if block is None:
        return False
    return block.is_basic and not is_method_wrapper(block)


_BASIC_DETAIL_LABELS = frozenset({"Linear", "RMSNorm", "LayerNorm", "Embedding"})

_DETAIL_OPERATION_LABELS = _BASIC_DETAIL_LABELS | frozenset(
    {
        "Causal depthwise conv",
        "SiLU",
        "Silu",
        "Sigmoid",
        "GELU",
        "Gelu",
        "Tanh",
        "ReLU",
        "Relu",
        "Attention",
        "L2Norm",
        "Output o",
        "Token Embedding",
        "Embedding",
        "×",
        "Multiply",
        "MoE aggregation",
    }
)

_DETAIL_COMBINE_LABELS = frozenset({"×", "+", "Elementwise ×", "Multiply", "Add"})

_DETAIL_OPERATION_CLASSES = frozenset(
    {
        "ActivationOp",
        "AttentionOp",
        "AttentionMerge",
        "KernelPipeline",
        "KernelOp",
        "KernelOutput",
        "Multiply",
    }
)


def keep_detail_graph_node(
    *,
    block: "BlockNode | None" = None,
    synthetic: str | None = None,
    label: str = "",
    basic_only: bool,
) -> bool:
    """Return True when a graph node spec should be kept in basic-only detail mode."""
    if not basic_only:
        return True
    if synthetic in {
        "@input",
        "@output",
        "@loop_carried",
        "@hidden_states",
        "@tensor",
    }:
        return True
    if synthetic is not None:
        return label in _DETAIL_COMBINE_LABELS
    display = label or (block.label if block else "")
    if display in _DETAIL_COMBINE_LABELS or display in _DETAIL_OPERATION_LABELS:
        return True
    if block is not None and block.class_name in _DETAIL_OPERATION_CLASSES:
        return True
    return show_in_detail_graph(block, basic_only=basic_only)
