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
    """Pair each norm with the compute module that follows it in forward order."""
    pairs: list[tuple[BlockComponent, BlockComponent]] = []
    pending_norm: BlockComponent | None = None
    for comp in ordered_components(components):
        if comp.role == "norm":
            pending_norm = comp
            continue
        if pending_norm is None:
            continue
        pairs.append((pending_norm, comp))
        pending_norm = None
    return pairs


def norm_input_sources(components: list[BlockComponent]) -> dict[str, str]:
    """Map compute module attr names to the norm label that feeds them."""
    return {
        module.attr_name: norm.label for norm, module in collect_norm_module_pairs(components)
    }


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
