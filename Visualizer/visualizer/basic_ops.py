"""Regex filters for treating modeled blocks as leaf/basic operations."""

from __future__ import annotations

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

# Class names produced by AST/block-tree expansion for modeled composite ops.
_MODELING_CLASS_PATTERNS: tuple[str, ...] = (
    r"(?i)ShortConv",
    r"(?i)OutputGate",
    r"(?i)KernelPipeline",
    r"(?i)AttentionMerge",
    r"(?i)KernelOp",
    r"(?i)KernelOutput",
    r"(?i)AttentionOp",
    r"(?i)ActivationOp",
    r"(?i)RouterOp",
    r"(?i)SituAndMul",
    r"(?i)SituActivation",
    r"(?i)Split",
    r"(?i)Multiply",
    r"(?i)FusedRMSNormGated",
    r"(?i)Fused.*Norm.*Gated",
    r"(?i)^ApplyRotary$",
    r"(?i)^RotaryEmbedding$",
)

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
    ) -> None:
        compiled: list[re.Pattern[str]] = []
        for pattern in patterns:
            compiled.append(pattern if isinstance(pattern, re.Pattern) else re.compile(pattern))
        self.patterns = compiled
        self.basic_only = basic_only

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
    if _matches_any(class_name, _MODELING_CLASS_PATTERNS):
        return True
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
    from visualizer.block_tree import is_method_wrapper

    if not basic_only:
        return True
    if block is None:
        return False
    return block.is_basic and not is_method_wrapper(block)


_BASIC_DETAIL_LABELS = frozenset({"Linear", "RMSNorm", "LayerNorm", "Embedding"})

_DETAIL_OPERATION_LABELS = _BASIC_DETAIL_LABELS | frozenset(
    {
        "Depthwise Conv",
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
        "Expert bias",
        "Group routing",
        "Top-k experts",
        "Gather weights",
        "Renormalize",
        "Route scaling",
        "Score activation",
    }
)

_DETAIL_COMBINE_LABELS = frozenset({"×", "Σ", "+", "Elementwise ×"})

_DETAIL_OPERATION_CLASSES = frozenset(
    {
        "ActivationOp",
        "AttentionOp",
        "AttentionMerge",
        "KernelPipeline",
        "KernelOp",
        "KernelOutput",
        "ShortConvolution",
        "RotaryEmbedding",
        "ApplyRotary",
        "RouterOp",
        "Multiply",
        "SituActivation",
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
    if synthetic in {"@input", "@hidden_states", "@tensor"}:
        return True
    if synthetic is not None:
        return label in _DETAIL_COMBINE_LABELS
    display = label or (block.label if block else "")
    if display in _DETAIL_OPERATION_LABELS:
        return True
    if block is not None and block.class_name in _DETAIL_OPERATION_CLASSES:
        return True
    return show_in_detail_graph(block, basic_only=basic_only)
