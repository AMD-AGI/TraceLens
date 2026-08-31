"""Build a top-level model overview graph for Model Explorer."""

from __future__ import annotations

import re
from typing import Any

from visualizer.basic_ops import BasicOpFilter
from visualizer.block_tree import (
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
from visualizer.blocks import BlockComponent, LayerVariant
from visualizer.extract import ArchitectureSpec, architecture_section_trees

from model_explorer_export.styles import ROLE_COLORS, ensure_readable_text, spine_tile_style


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
    if component.role == "head" and component.class_name == "Linear":
        return "Logits"
    if component.role == "embedding":
        return component.label
    return component.label or component.attr_name


_DECODER_NORM_ATTRS = frozenset({"input_layernorm", "post_attention_layernorm"})


def _section_namespace_segment(
    spec: ArchitectureSpec,
    component: BlockComponent,
    *,
    variant: LayerVariant | None = None,
) -> str:
    from model_explorer_export.adapter import _sanitize_namespace_segment

    if component.role == "norm":
        return _sanitize_namespace_segment(spec.norm_type or "RMSNorm")
    if variant is not None:
        if component.role == "attention" and variant.attention_class:
            return _sanitize_namespace_segment(variant.attention_class)
        if component.role in {"moe", "ffn"}:
            if variant.ffn_class:
                return _sanitize_namespace_segment(variant.ffn_class)
            if variant.ffn_attr:
                return _sanitize_namespace_segment(variant.ffn_attr)
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
    classes = {variant.ffn_class for variant in spec.layer_variants if variant.ffn_class}
    moe_class = _spine_moe_class(spec)
    if moe_class:
        classes.add(moe_class)
    return classes


def _detail_section_trees(spec: ArchitectureSpec) -> list[BlockNode]:
    """Block trees rendered as expandable internal subsections in the SVG diagram."""
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


def component_has_detail_section(component: BlockComponent, spec: ArchitectureSpec) -> bool:
    """True when the SVG detailed figure expands this module below the overview spine."""
    for tree in _detail_section_trees(spec):
        if tree.attr_name == component.attr_name or tree.class_name == component.class_name:
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


def _style_for_component(component: BlockComponent) -> dict[str, str] | None:
    if component.role in {"attention", "moe", "ffn"} and component.role in ROLE_COLORS:
        return ensure_readable_text(dict(ROLE_COLORS[component.role]))
    return ensure_readable_text(spine_tile_style())


def _edge(source_id: str, target_id: str) -> dict[str, str]:
    return {
        "sourceNodeId": source_id,
        "sourceNodeOutputId": "0",
        "targetNodeInputId": "0",
    }


def _subgraph_ids(attr_name: str, attr_name_to_graph_id: dict[str, str]) -> list[str] | None:
    graph_id = attr_name_to_graph_id.get(attr_name)
    return [graph_id] if graph_id else None


def build_overview_graph(
    spec: ArchitectureSpec,
    *,
    attr_name_to_graph_id: dict[str, str],
    graph_id: str = "model",
) -> dict[str, Any]:
    """Build the main model spine graph shown in the SVG overview diagram."""
    nodes: list[dict[str, Any]] = []
    incoming: dict[str, list[dict[str, str]]] = {}
    previous_id = "@input"

    def append_node(
        node_id: str,
        label: str,
        *,
        namespace: str = "",
        component: BlockComponent | None = None,
    ) -> None:
        nonlocal previous_id
        node: dict[str, Any] = {
            "id": node_id,
            "label": label,
            "namespace": namespace,
        }
        attrs: list[dict[str, str]] = []
        if component is not None:
            attrs.extend(
                [
                    {"key": "attr_name", "value": component.attr_name},
                    {"key": "class_name", "value": component.class_name},
                    {"key": "role", "value": component.role},
                ]
            )
        if attrs:
            node["attrs"] = attrs
        if component is not None:
            style = _style_for_component(component)
            if style:
                node["style"] = style
            linked = _subgraph_ids(component.attr_name, attr_name_to_graph_id)
            if linked:
                node["subgraphIds"] = linked
        incoming.setdefault(node_id, []).append(_edge(previous_id, node_id))
        nodes.append(node)
        previous_id = node_id

    nodes.append(
        {
            "id": "@input",
            "label": "Tokenized text",
            "namespace": "",
            "attrs": [{"key": "synthetic", "value": "@input"}],
            "style": ensure_readable_text({"backgroundColor": "#d9e8f5", "textColor": "#1a1a1a"}),
        }
    )

    for component in _stack_pre_components(spec):
        append_node(
            component.attr_name,
            _display_label(component, spec),
            component=component,
        )

    decoder_namespace = _decoder_namespace(spec)
    for component in _ordered_decoder_components(spec):
        append_node(
            f"decoder/{component.attr_name}",
            _display_label(component, spec),
            namespace=decoder_namespace,
            component=component,
        )

    for component in _stack_tail_components(spec):
        append_node(
            component.attr_name,
            _display_label(component, spec),
            component=component,
        )

    for node in nodes:
        node_id = node["id"]
        if node_id in incoming:
            node["incomingEdges"] = incoming[node_id]

    model_attrs: dict[str, str] = {
        "title": spec.name,
        "model_type": spec.model_type,
        "decoder": spec.decoder_type,
        "layers": str(spec.num_hidden_layers or "?"),
    }
    if spec.forward_sequence:
        model_attrs["forward"] = " → ".join(spec.forward_sequence)
    if spec.decoder_class:
        model_attrs["decoder_class"] = spec.decoder_class

    return {
        "id": graph_id,
        "nodes": nodes,
        "groupNodeAttributes": {
            "": model_attrs,
            decoder_namespace: {
                "repeat": _decoder_namespace(spec),
                "forward": " → ".join(spec.forward_sequence or []),
            },
        },
        "groupNodeConfigs": [
            {
                "namespaceRegex": re.escape(decoder_namespace),
                "backgroundColor": "#fff5f4",
                "borderColor": "#c0392b",
                "textColor": "#1a1a1a",
                "layoutDirection": "TOP_BOTTOM",
            }
        ],
    }
