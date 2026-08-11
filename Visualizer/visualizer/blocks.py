"""Datatypes for code-derived architecture blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visualizer.ast_analyze import ClassStructure


@dataclass
class LayerVariant:
    """One distinct decoder-layer template and how many layers use it."""

    label: str
    count: int
    attention_label: str
    attention_class: str | None = None
    ffn_label: str = "FFN"
    ffn_class: str | None = None
    ffn_attr: str | None = None


@dataclass
class BlockComponent:
    """One submodule inside a decoder layer, ordered for diagram rendering."""

    attr_name: str
    class_name: str
    role: str  # attention, moe, ffn, norm, router, embedding, head, positional, other
    label: str
    forward_order: int | None = None
    details: list[str] = field(default_factory=list)


def ordered_components(components: list[BlockComponent]) -> list[BlockComponent]:
    """Sort components by forward order."""
    return sorted(
        components,
        key=lambda comp: (
            comp.forward_order is None,
            comp.forward_order if comp.forward_order is not None else 999,
            comp.attr_name,
        ),
    )


def collect_norm_module_pairs(
    components: list[BlockComponent],
) -> list[tuple[BlockComponent, BlockComponent]]:
    """Pair each norm with every compute module it feeds before the next norm."""
    pairs: list[tuple[BlockComponent, BlockComponent]] = []
    pending_norm: BlockComponent | None = None
    paired_order: int | None = None
    for comp in ordered_components(components):
        if comp.role == "norm":
            pending_norm = comp
            paired_order = None
            continue
        if pending_norm is None:
            continue
        if paired_order is None:
            paired_order = comp.forward_order
        elif comp.forward_order != paired_order:
            pending_norm = None
            paired_order = None
            continue
        pairs.append((pending_norm, comp))
    return pairs


def upstream_input_sources(components: list[BlockComponent]) -> dict[str, str]:
    """Map compute module attr names to the nearest upstream operator in forward order."""
    ordered = ordered_components(components)
    sources: dict[str, str] = {}
    for comp in ordered:
        if comp.role == "norm" or comp.forward_order is None:
            continue
        upstream_candidates = [
            candidate
            for candidate in ordered
            if candidate.forward_order is not None and candidate.forward_order < comp.forward_order
        ]
        if not upstream_candidates:
            continue
        upstream_order = max(candidate.forward_order for candidate in upstream_candidates)
        upstream_at_order = [
            candidate for candidate in upstream_candidates if candidate.forward_order == upstream_order
        ]
        sources[comp.attr_name] = upstream_at_order[-1].attr_name
    return sources


def norm_input_sources(components: list[BlockComponent]) -> dict[str, str]:
    """Backwards-compatible alias for upstream_input_sources."""
    return upstream_input_sources(components)


@dataclass
class CodeAnalysis:
    """Architecture facts extracted from Python AST inspection."""

    decoder_class: str | None = None
    attention_class: str | None = None
    model_class: str | None = None
    stack_model_class: str | None = None
    causal_lm_class: str | None = None
    block_components: list[BlockComponent] = field(default_factory=list)
    stack_pre: list[BlockComponent] = field(default_factory=list)
    stack_tail: list[BlockComponent] = field(default_factory=list)
    forward_sequence: list[str] = field(default_factory=list)
    norm_placement: str | None = None
    norm_type: str | None = None
    attention_type: str | None = None
    decoder_type: str | None = None
    ffn_type: str | None = None
    custom_blocks: list[str] = field(default_factory=list)
    layer_repeat_lines: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    class_registry: dict[str, ClassStructure] = field(default_factory=dict)

    def has_block_graph(self) -> bool:
        return bool(self.block_components or self.forward_sequence or self.stack_pre or self.stack_tail)
