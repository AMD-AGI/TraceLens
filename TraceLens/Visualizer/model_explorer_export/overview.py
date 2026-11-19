###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build a top-level model overview graph for Model Explorer."""

from __future__ import annotations


from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.block_tree import (
    BlockNode,
    _omit_from_detailed_view,
    _show_single_function_in_diagram,
    collect_nested_diagrams,
    expand_block_tree_inplace,
    is_single_function_tree,
    is_straight_line_module,
    prepare_diagram_section_trees,
    subgraph_warrants_export,
)
from TraceLens.ModelUtils.ast_analyze import displays_as_linear
from TraceLens.ModelUtils.blocks import BlockComponent, LayerVariant
from TraceLens.ModelUtils.extract import ArchitectureSpec, architecture_section_trees


def _stack_pre_components(spec: ArchitectureSpec) -> list[BlockComponent]:
    if spec.stack_pre:
        return list(spec.stack_pre)
    if spec.vocab_size is None:
        return []
    return [
        BlockComponent(
            attr_name="embed_tokens",
            class_name="Embedding",
            role="embedding",
            label="Token Embedding",
            forward_order=0,
            inferred_from_config=True,
        )
    ]


def _stack_tail_components(spec: ArchitectureSpec) -> list[BlockComponent]:
    if spec.stack_tail:
        return list(spec.stack_tail)
    components: list[BlockComponent] = []
    if spec.norm_type:
        components.append(
            BlockComponent(
                attr_name="norm",
                class_name=spec.norm_type,
                role="norm",
                label=spec.norm_type,
                forward_order=0,
                inferred_from_config=True,
            )
        )
    if spec.vocab_size is not None:
        components.append(
            BlockComponent(
                attr_name="lm_head",
                class_name="Linear",
                role="head",
                label="Linear",
                forward_order=1,
                inferred_from_config=True,
            )
        )
    return components


def _ordered_decoder_components(spec: ArchitectureSpec) -> list[BlockComponent]:
    if not spec.block_components:
        return []
    return sorted(
        spec.block_components,
        key=lambda comp: (
            comp.forward_order is None,
            comp.forward_order if comp.forward_order is not None else 999,
            comp.attr_name,
        ),
    )


def _display_label(component: BlockComponent, spec: ArchitectureSpec) -> str:
    if component.role == "norm":
        return spec.norm_type or component.label or "RMSNorm"
    if component.role == "head" and displays_as_linear(
        component.attr_name, component.class_name
    ):
        # The lm_head is a real Linear projection (matmul, learned weight hidden),
        # whether it is literally ``nn.Linear`` or an engine's sharded wrapper
        # (``ParallelLMHead``); render it as the ``Linear`` op it is, like every
        # other Linear, rather than as an opaque ``Logits`` box. ``logits`` stays
        # the output tensor's name (see ``shapes.py`` output handling), but the
        # node reads as Linear.
        return "Linear"
    if component.role == "embedding":
        return component.label
    return component.label or component.attr_name


def _forward_step_display_label(
    attr_name: str,
    *,
    spec: ArchitectureSpec,
    components_by_attr: dict[str, BlockComponent],
) -> str:
    """Map one decoder ``forward()`` step to the same label used on graph tiles."""
    component = components_by_attr.get(attr_name)
    if component is not None:
        return _display_label(component, spec)

    from TraceLens.ModelUtils.ast_analyze import (
        classify_matmul_label,
        operation_display_label,
    )

    decoder = spec.class_registry.get(spec.decoder_class or "")
    if decoder is not None:
        operation = decoder.forward_operations.get(attr_name)
        if operation is not None:
            if operation.label in {"MatMul", "MatMult"}:
                return classify_matmul_label(external_inputs=operation.external_inputs)
            return operation_display_label(
                operation.label, class_name=operation.class_name
            )
    return attr_name.replace("_", " ")


def forward_sequence_display_labels(spec: ArchitectureSpec) -> list[str]:
    """Return decoder forward steps using graph tile labels instead of attr names."""
    if not spec.forward_sequence:
        return []
    components_by_attr = {
        component.attr_name: component
        for component in _ordered_decoder_components(spec)
    }
    return [
        _forward_step_display_label(
            attr_name, spec=spec, components_by_attr=components_by_attr
        )
        for attr_name in spec.forward_sequence
    ]


def format_forward_sequence(spec: ArchitectureSpec, *, arrow: str = " → ") -> str:
    """Format the decoder spine order for metadata and the fact sheet."""
    return arrow.join(forward_sequence_display_labels(spec))


_DECODER_NORM_ATTRS = frozenset({"input_layernorm", "post_attention_layernorm"})
_VARIANT_ATTENTION_ATTRS = frozenset(
    {"self_attn", "self_attention", "attn", "attention"}
)


def _component_uses_variant_attention_class(
    component: BlockComponent,
    variant: LayerVariant | None,
) -> bool:
    return (
        variant is not None
        and bool(variant.attention_class)
        and component.attr_name in _VARIANT_ATTENTION_ATTRS
    )


def _component_uses_variant_ffn_class(
    component: BlockComponent,
    variant: LayerVariant | None,
) -> bool:
    if variant is None or not variant.ffn_class:
        return False
    if variant.ffn_attr and component.attr_name == variant.ffn_attr:
        return True
    return component.attr_name == "mlp" and component.class_name == variant.ffn_class


def _shared_decoder_class_attr_names(
    spec: ArchitectureSpec, class_name: str
) -> list[str]:
    return [
        component.attr_name
        for component in spec.block_components
        if component.class_name == class_name
    ]


def _section_namespace_segment(
    spec: ArchitectureSpec,
    component: BlockComponent,
    *,
    variant: LayerVariant | None = None,
) -> str:
    from TraceLens.Visualizer.model_explorer_export.adapter import (
        _sanitize_namespace_segment,
    )

    if component.role == "norm":
        # Every norm shares one class name, so keying the namespace on the class
        # would merge distinct instances into one group and make the spine look
        # like it loops back (attn_hc -> norm -> ... -> ffn_hc -> same norm).
        norm_attrs = {
            other.attr_name
            for other in (spec.block_components or [])
            if other.role == "norm"
        }
        if len(norm_attrs) > 1:
            return _sanitize_namespace_segment(component.attr_name)
        return _sanitize_namespace_segment(spec.norm_type or "RMSNorm")
    if variant is not None:
        if _component_uses_variant_attention_class(component, variant):
            return _sanitize_namespace_segment(variant.attention_class or "")
        if _component_uses_variant_ffn_class(component, variant):
            return _sanitize_namespace_segment(variant.ffn_class or "")
        if variant.ffn_attr and component.attr_name == variant.ffn_attr:
            return _sanitize_namespace_segment(variant.ffn_attr)
    if (
        component.class_name
        and len(_shared_decoder_class_attr_names(spec, component.class_name)) > 1
    ):
        return _sanitize_namespace_segment(component.attr_name)
    if component.class_name and component.role in {"attention", "moe", "ffn"}:
        return _sanitize_namespace_segment(component.class_name)
    return _sanitize_namespace_segment(component.attr_name)


def _decoder_namespace(spec: ArchitectureSpec) -> str:
    count = spec.num_hidden_layers if spec.num_hidden_layers is not None else "?"
    decoder_name = spec.decoder_class or spec.decoder_type or "DecoderLayer"
    return f"{count}x_{decoder_name}"


def _spine_moe_class(spec: ArchitectureSpec) -> str | None:
    """MoE class the spine's FFN tile stands for when every layer builds the same one."""
    if spec.layer_variants:
        return None
    for component in _ordered_decoder_components(spec):
        if component.role == "moe" and component.class_name:
            return component.class_name
    return None


def _spine_named_ffn_classes(spec: ArchitectureSpec) -> set[str]:
    classes = {
        variant.ffn_class for variant in spec.layer_variants if variant.ffn_class
    }
    moe_class = _spine_moe_class(spec)
    if moe_class:
        classes.add(moe_class)
    return classes


def _detail_section_trees(spec: ArchitectureSpec) -> list[BlockNode]:
    """Block trees exported as expandable internal subsections."""
    basic_ops = spec.basic_ops or BasicOpFilter.for_detailed()
    section_trees = prepare_diagram_section_trees(
        architecture_section_trees(spec),
        basic_ops=basic_ops,
    )
    spine_ffn = _spine_named_ffn_classes(spec)
    trees: list[BlockNode] = []
    for _title, tree in section_trees:
        if (
            not is_straight_line_module(tree) or tree.class_name in spine_ffn
        ) and subgraph_warrants_export(tree, basic_ops=basic_ops):
            trees.append(tree)
        for _sub_title, sub_tree in collect_nested_diagrams(tree, basic_ops=basic_ops):
            prepared = expand_block_tree_inplace(sub_tree, basic_ops=basic_ops)
            if is_single_function_tree(prepared):
                if _omit_from_detailed_view(prepared):
                    continue
                if not _show_single_function_in_diagram(prepared):
                    continue
            trees.append(prepared)
    return trees


def component_has_detail_section(
    component: BlockComponent, spec: ArchitectureSpec
) -> bool:
    """True when this module expands into a detail subsection below the overview spine."""
    for tree in _detail_section_trees(spec):
        if tree.attr_name == component.attr_name:
            return True
    basic_ops = spec.basic_ops or BasicOpFilter.for_detailed()
    for _title, tree in architecture_section_trees(spec):
        if tree.attr_name != component.attr_name:
            continue
        prepared = expand_block_tree_inplace(tree, basic_ops=basic_ops)
        if subgraph_warrants_export(prepared, basic_ops=basic_ops):
            return True
    return False


def _flat_spine_namespace(
    component: BlockComponent,
    namespace_prefix: str,
    *,
    variant: LayerVariant | None,
) -> str:
    """Namespace for a single-tile spine module that does not expand into a subgraph."""
    if variant is not None:
        return namespace_prefix
    if component.role in {"embedding", "head", "norm"}:
        return ""
    return namespace_prefix


def _edge(source_id: str, target_id: str) -> dict[str, str]:
    return {
        "sourceNodeId": source_id,
        "sourceNodeOutputId": "0",
        "targetNodeInputId": "0",
    }
