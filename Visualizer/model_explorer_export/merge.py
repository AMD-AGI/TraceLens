"""Build one merged Model Explorer graph with in-place namespace expansion."""

from __future__ import annotations

import re
from typing import Any

from visualizer.basic_ops import BasicOpFilter
from visualizer.block_tree import BlockNode, collect_nested_diagrams, subgraph_warrants_json_export
from visualizer.blocks import BlockComponent, LayerVariant
from visualizer.computation_graph import ComputationGraph, build_computation_graph
from visualizer.extract import ArchitectureSpec, architecture_section_trees
from visualizer.shape_inference import ShapeInferencer

from model_explorer_export.adapter import (
    _incoming_edges,
    _node_attrs,
    _node_namespaces,
    _node_style,
    _sanitize_namespace_segment,
)
from model_explorer_export.fact_sheet import build_fact_sheet_group_attributes
from model_explorer_export.shapes import annotate_nodes_with_shapes, infer_block_tree_shapes
from model_explorer_export.labels import (
    apply_kernel_frame_labels,
    skip_merged_tensor_port_parent,
    tensor_port_input_label,
)
from model_explorer_export.overview import (
    _DECODER_NORM_ATTRS,
    _decoder_namespace,
    _display_label,
    _flat_spine_namespace,
    _ordered_decoder_components,
    _section_namespace_segment,
    _stack_pre_components,
    _stack_tail_components,
    component_has_detail_section,
)
from model_explorer_export.styles import ROLE_COLORS, _GPU_KERNEL_BORDER, build_group_node_configs, ensure_readable_text, finalize_graph_node_styles, input_port_style, spine_tile_style


def _join_namespace(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}/{suffix}"


def _merge_node_id(prefix: str, local_id: str) -> str:
    return f"{prefix}/{local_id}" if prefix else local_id


def _is_synthetic_input(node: dict[str, Any]) -> bool:
    node_id = node.get("id", "")
    if node_id == "@input" or re.search(r"/@input(?::|$)", node_id):
        return True
    for attr in node.get("attrs", []):
        if attr.get("key") == "synthetic" and attr.get("value") == "@input":
            return True
    return False


def _node_attr(node: dict[str, Any], key: str) -> str | None:
    for attr in node.get("attrs", []):
        if attr.get("key") == key:
            value = attr.get("value")
            if isinstance(value, str):
                return value
    return None


def _group_input_id(prefix: str, port_label: str | None = None) -> str:
    if port_label:
        return f"{prefix}/@input:{port_label}"
    return f"{prefix}/@input"


def _edge_port_label(edge: dict[str, Any]) -> str | None:
    metadata = edge.get("metadata") or {}
    value = metadata.get("port_label")
    return str(value) if value else None


def _label_input_edge(edge: dict[str, Any], label: str) -> dict[str, Any]:
    labeled = dict(edge)
    labeled.setdefault("metadata", {})["port_label"] = label
    return labeled


def _set_input_port_metadata(node: dict[str, Any], input_id: str, label: str) -> None:
    items = list(node.get("inputsMetadata") or [])
    for item in items:
        if item.get("id") != input_id:
            continue
        attrs = [attr for attr in item.get("attrs", []) if attr.get("key") != "port_label"]
        attrs.append({"key": "port_label", "value": label})
        item["attrs"] = attrs
        node["inputsMetadata"] = items
        return
    items.append({"id": input_id, "attrs": [{"key": "port_label", "value": label}]})
    node["inputsMetadata"] = items


def _apply_labeled_external_entry_ports(
    entry_ports: list[tuple[str | None, dict[str, Any], dict[str, Any]]],
    internal_ids: set[str],
) -> bool:
    """Keep real upstream nodes and label their entry edges instead of adding @input ports."""
    if not entry_ports:
        return False

    labels = [label for label, _, _ in entry_ports if label]
    multi_labeled = (
        len(entry_ports) >= 2
        and len(labels) == len(entry_ports)
        and len(set(labels)) == len(labels)
    )
    single_labeled = len(entry_ports) == 1 and entry_ports[0][0] is not None
    if not multi_labeled and not single_labeled:
        return False

    for label, edge, target in entry_ports:
        if not label:
            continue
        input_id = str(edge.get("targetNodeInputId", "0"))
        incoming = list(target.get("incomingEdges", []))
        relabeled: list[dict[str, Any]] = []
        for item in incoming:
            if item["sourceNodeId"] in internal_ids:
                relabeled.append(item)
                continue
            if (
                item["sourceNodeId"] == edge["sourceNodeId"]
                and str(item.get("targetNodeInputId", "0")) == input_id
            ):
                relabeled.append(_label_input_edge(item, label))
            else:
                relabeled.append(item)
        target["incomingEdges"] = relabeled
        _set_input_port_metadata(target, input_id, label)
    return True


def _labeled_tensor_port_label(node: dict[str, Any] | None) -> str | None:
    if node is None or _node_attr(node, "synthetic") != "@tensor":
        return None
    label = node.get("label")
    return label if isinstance(label, str) and label else None


def _infer_entry_port_label(
    edge: dict[str, Any],
    target: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
) -> str | None:
    return (
        _edge_port_label(edge)
        or _labeled_tensor_port_label(target)
        or _node_attr(target, "port_label")
        or _labeled_tensor_port_label(node_by_id.get(edge["sourceNodeId"]))
    )


def _collect_group_entry_ports(
    group_nodes: list[dict[str, Any]],
    internal_ids: set[str],
    node_by_id: dict[str, dict[str, Any]],
) -> list[tuple[str | None, dict[str, Any], dict[str, Any]]]:
    ports: list[tuple[str | None, dict[str, Any], dict[str, Any]]] = []
    seen: set[tuple[str, str, str]] = set()
    for node in group_nodes:
        for edge in node.get("incomingEdges", []):
            source_id = edge["sourceNodeId"]
            if source_id in internal_ids:
                continue
            key = (source_id, node["id"], str(edge.get("targetNodeInputId", "0")))
            if key in seen:
                continue
            seen.add(key)
            ports.append((_infer_entry_port_label(edge, node, node_by_id), edge, node))
    return ports


def _make_group_input_node(
    *,
    input_id: str,
    label: str,
    namespace: str,
    incoming_edges: list[dict[str, Any]] | None = None,
    port_label: str | None = None,
) -> dict[str, Any]:
    attrs: list[dict[str, str]] = [{"key": "synthetic", "value": "@input"}]
    if port_label:
        attrs.append({"key": "port_label", "value": port_label})
    node: dict[str, Any] = {
        "id": input_id,
        "label": label,
        "namespace": namespace,
        "attrs": attrs,
        "style": _input_style(),
    }
    if incoming_edges:
        node["incomingEdges"] = incoming_edges
    return node


def _input_style() -> dict[str, str]:
    return ensure_readable_text(input_port_style())


def _computation_nodes(
    computation: ComputationGraph,
    *,
    id_prefix: str,
    namespace_prefix: str,
    skip_synthetic_input: bool = False,
) -> list[dict[str, Any]]:
    index_to_local = {
        index: spec.key or f"node:{index}"
        for index, spec in enumerate(computation.nodes)
    }
    local_to_prefixed = {
        local_id: _merge_node_id(id_prefix, local_id) for local_id in index_to_local.values()
    }
    relative_namespaces = _node_namespaces(computation)
    index_to_prefixed = {
        index: local_to_prefixed[local_id] for index, local_id in index_to_local.items()
    }
    incoming_local = _incoming_edges(computation, index_to_local)

    nodes: list[dict[str, Any]] = []
    for index, spec in enumerate(computation.nodes):
        local_id = index_to_local[index]
        prefixed_id = index_to_prefixed[index]
        if skip_synthetic_input and (local_id == "@input" or spec.synthetic == "@input"):
            continue

        relative_ns = relative_namespaces.get(index, "")
        namespace = _join_namespace(namespace_prefix, relative_ns)
        node: dict[str, Any] = {
            "id": prefixed_id,
            "label": spec.label or local_id,
            "namespace": namespace,
        }
        attrs = _node_attrs(spec)
        if attrs:
            node["attrs"] = attrs
        if local_id == "@input" or spec.synthetic == "@input":
            node["style"] = _input_style()
        else:
            style = _node_style(spec)
            if style:
                node["style"] = ensure_readable_text(style)

        remapped_incoming: list[dict[str, Any]] = []
        for edge in incoming_local.get(local_id, []):
            source_local = edge["sourceNodeId"]
            if skip_synthetic_input and source_local == "@input":
                continue
            remapped = dict(edge)
            remapped["sourceNodeId"] = local_to_prefixed[source_local]
            remapped_incoming.append(remapped)
        if remapped_incoming:
            node["incomingEdges"] = remapped_incoming
        nodes.append(node)

    return nodes


def _boundary_nodes(nodes: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    node_ids = {node["id"] for node in nodes}
    sources = {
        edge["sourceNodeId"]
        for node in nodes
        for edge in node.get("incomingEdges", [])
    }
    targets = {
        node["id"]
        for node in nodes
        for edge in node.get("incomingEdges", [])
    }
    entries = sorted(node_id for node_id in node_ids if node_id not in targets)
    exits = sorted(node_id for node_id in node_ids if node_id not in sources)
    return entries or sorted(node_ids)[:1], exits or sorted(node_ids)[-1:]


def _is_primary_section_input(node: dict[str, Any]) -> bool:
    """Section spine @input only — not labeled multi-port feeders like @input:q."""
    if not _is_synthetic_input(node):
        return False
    node_id = node.get("id", "")
    return node_id == "@input" or node_id.endswith("/@input")


def _section_input_nodes(
    section_nodes: list[dict[str, Any]],
    namespace_prefix: str,
) -> list[str]:
    return [
        node["id"]
        for node in section_nodes
        if _is_primary_section_input(node) and node.get("namespace", "") == namespace_prefix
    ]


def _connect_external_inputs(
    section_nodes: list[dict[str, Any]],
    *,
    namespace_prefix: str,
    previous_exits: list[str],
) -> None:
    if not previous_exits:
        return
    input_ids = _section_input_nodes(section_nodes, namespace_prefix)
    connect_targets = input_ids or _boundary_nodes(section_nodes)[0]
    node_by_id = {node["id"]: node for node in section_nodes}
    for target_id in connect_targets:
        target = node_by_id.get(target_id)
        if target is None:
            continue
        target["incomingEdges"] = [
            {
                "sourceNodeId": source_id,
                "sourceNodeOutputId": "0",
                "targetNodeInputId": str(index),
            }
            for index, source_id in enumerate(previous_exits)
        ]


def _common_id_prefix(ids: list[str]) -> str:
    if not ids:
        return ""
    prefix = ids[0]
    for node_id in ids[1:]:
        while prefix and not node_id.startswith(prefix):
            prefix = prefix[:-1]
        prefix = prefix.rstrip("/:")
    return prefix


def _group_input_prefix(group_ids: list[str]) -> str:
    """Pick a stable group input id prefix, avoiding partial kernel sub-op stems."""
    prefix = _common_id_prefix(group_ids) or group_ids[0].rsplit("/", 1)[0]
    if ":" in prefix:
        head, tail = prefix.rsplit(":", 1)
        if "_sub_" in tail:
            tail = tail.split("_sub_", 1)[0]
            return f"{head}:{tail}"
    elif "_sub_" in prefix:
        prefix = prefix.split("_sub_", 1)[0]
    return prefix


def _infer_group_input_label(
    group_nodes: list[dict[str, Any]],
    namespace: str,
    *,
    entry_ports: list[tuple[str | None, dict[str, Any], dict[str, Any]]] | None = None,
) -> str:
    if entry_ports:
        labels = [label for label, _, _ in entry_ports if label]
        if len(labels) == 1:
            return labels[0]
    for node in group_nodes:
        for attr in node.get("attrs", []):
            if attr.get("key") == "port_label" and attr.get("value"):
                return str(attr["value"])
    segment = namespace.rsplit("/", 1)[-1]
    port_label = tensor_port_input_label(namespace)
    if port_label is not None:
        return port_label
    if segment in {"KimiMLP", "KimiMoEGate"}:
        return "x" if segment == "KimiMLP" else "hidden_states"
    return "hidden_states"


def _skip_variant_root_input(component: BlockComponent) -> bool:
    """Decoder norms at the variant namespace are single tiles, not expanded input groups."""
    return component.role == "norm" and component.attr_name in _DECODER_NORM_ATTRS


def _namespace_is_descendant(node_namespace: str, group_namespace: str) -> bool:
    if not group_namespace:
        return node_namespace == ""
    return node_namespace == group_namespace or node_namespace.startswith(f"{group_namespace}/")


def _namespace_internal_ids(section_nodes: list[dict[str, Any]], namespace: str) -> set[str]:
    """Include nested namespaces so SituAndMul ops stay inside KimiMLP groups."""
    return {
        node["id"]
        for node in section_nodes
        if _namespace_is_descendant(node.get("namespace", ""), namespace)
    }


_INLINE_FRAME_NAMESPACE_SUFFIXES = frozenset({"SituAndMul", "SiluAndMul"})


def _skip_nested_inline_frame_input(section_nodes: list[dict[str, Any]], namespace: str) -> bool:
    """Inline activation frames inherit the parent KimiMLP input port."""
    segment = namespace.rsplit("/", 1)[-1]
    if segment not in _INLINE_FRAME_NAMESPACE_SUFFIXES or "/" not in namespace:
        return False
    parent = namespace.rsplit("/", 1)[0]
    return any(
        _is_synthetic_input(node) and node.get("namespace", "") == parent
        for node in section_nodes
    )


def _inject_group_inputs(
    section_nodes: list[dict[str, Any]],
    *,
    skip_namespaces: frozenset[str] = frozenset(),
) -> None:
    """Add a visible @input port to expanded namespace groups that lack one."""
    node_by_id = {node["id"]: node for node in section_nodes}
    namespaces = sorted({node.get("namespace", "") for node in section_nodes if node.get("namespace")})

    for namespace in namespaces:
        if namespace in skip_namespaces:
            continue
        group_nodes = [node for node in section_nodes if node.get("namespace", "") == namespace]
        if any(_is_synthetic_input(node) for node in group_nodes):
            continue
        if _skip_nested_inline_frame_input(section_nodes, namespace):
            continue
        if skip_merged_tensor_port_parent(namespace, group_nodes):
            continue

        internal_ids = _namespace_internal_ids(section_nodes, namespace)
        entry_ports = _collect_group_entry_ports(group_nodes, internal_ids, node_by_id)
        entry_nodes: list[dict[str, Any]] = []
        outside_sources: set[str] = set()

        for node in group_nodes:
            incoming = list(node.get("incomingEdges", []))
            external = [edge for edge in incoming if edge["sourceNodeId"] not in internal_ids]
            if external or not incoming:
                entry_nodes.append(node)
                outside_sources.update(edge["sourceNodeId"] for edge in external)

        if not entry_nodes:
            continue

        if _apply_labeled_external_entry_ports(entry_ports, internal_ids):
            continue

        group_ids = [node["id"] for node in group_nodes]
        prefix = _group_input_prefix(group_ids) or group_ids[0].rsplit("/", 1)[0]

        input_id = _group_input_id(prefix)
        if input_id in node_by_id:
            continue

        input_node = _make_group_input_node(
            input_id=input_id,
            label=_infer_group_input_label(group_nodes, namespace, entry_ports=entry_ports),
            namespace=namespace,
        )
        if outside_sources:
            input_node["incomingEdges"] = [
                {
                    "sourceNodeId": source_id,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": str(index),
                }
                for index, source_id in enumerate(sorted(outside_sources))
            ]

        for entry in entry_nodes:
            incoming = list(entry.get("incomingEdges", []))
            internal = [edge for edge in incoming if edge["sourceNodeId"] in internal_ids]
            entry["incomingEdges"] = internal + [
                {
                    "sourceNodeId": input_id,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": str(len(internal)),
                }
            ]

        section_nodes.append(input_node)
        node_by_id[input_id] = input_node


def _nested_namespace_segment(nested_block: BlockNode, nested_label: str) -> str:
    if nested_block.class_name == "KernelPipeline":
        return _sanitize_namespace_segment(nested_label)
    if nested_block.attr_name.startswith("@"):
        if nested_label and nested_label != nested_block.attr_name:
            return _sanitize_namespace_segment(nested_label)
        return _sanitize_namespace_segment(nested_block.attr_name.lstrip("@"))
    return _sanitize_namespace_segment(nested_block.attr_name)


def _is_tensor_port(node: dict[str, Any]) -> bool:
    return _node_attr(node, "synthetic") == "@tensor"


def _integrate_kernel_pipeline_merge(
    section_nodes: list[dict[str, Any]],
    *,
    namespace_prefix: str,
    pipeline_namespace: str,
    pipeline_prefix: str,
    pipeline_label: str,
    group_node_attributes: dict[str, dict[str, str]] | None = None,
    inject_skip: set[str] | None = None,
) -> None:
    """Replace the collapsed KernelPipeline merge tile with an expanded subgraph."""
    merge_nodes = [
        node
        for node in section_nodes
        if _node_attr(node, "attr_name") == "@attn_pipeline"
        and _node_attr(node, "class_name") == "KernelPipeline"
        and node.get("namespace") in {namespace_prefix, pipeline_namespace}
    ]
    if not merge_nodes:
        return

    merge = merge_nodes[0]
    merge_id = merge["id"]
    merge_edges = list(merge.get("incomingEdges", []))

    tensor_by_label = {
        node.get("label", ""): node
        for node in section_nodes
        if node.get("namespace") == pipeline_namespace and _is_tensor_port(node)
    }
    if not tensor_by_label:
        return

    if inject_skip is not None:
        inject_skip.add(pipeline_namespace)

    section_nodes.remove(merge)

    default_labels = ["q", "k", "v", "g", "beta"]
    merge_edge_by_port: dict[int, dict[str, Any]] = {}
    for edge in merge_edges:
        try:
            port_index = int(edge.get("targetNodeInputId", "0"))
        except ValueError:
            continue
        if 0 <= port_index < len(default_labels):
            merge_edge_by_port[port_index] = edge

    for port_index, label in enumerate(default_labels):
        tensor = tensor_by_label.get(label)
        if tensor is None:
            continue
        merge_edge = merge_edge_by_port.get(port_index)
        if merge_edge is not None:
            tensor["incomingEdges"] = [
                _label_input_edge({**merge_edge, "targetNodeInputId": "0"}, label)
            ]
            _set_input_port_metadata(tensor, "0", label)

    pipeline_nodes = [
        node
        for node in section_nodes
        if node.get("namespace", "").startswith(pipeline_namespace)
    ]
    _, pipeline_exits = _boundary_nodes(pipeline_nodes)
    pipeline_exit = next(
        (
            node_id
            for node_id in reversed(pipeline_exits)
            if "chunk_gated_delta_rule_fwd_h" in node_id
        ),
        pipeline_exits[-1] if pipeline_exits else None,
    )

    if pipeline_exit is not None:
        for node in section_nodes:
            if node.get("namespace") != namespace_prefix:
                continue
            if _node_attr(node, "attr_name") != "@attn_output":
                continue
            rewired: list[dict[str, Any]] = []
            for edge in node.get("incomingEdges", []):
                if edge["sourceNodeId"] == merge_id:
                    rewired.append({**edge, "sourceNodeId": pipeline_exit})
                else:
                    rewired.append(edge)
            node["incomingEdges"] = rewired

    if group_node_attributes is not None:
        attrs = {
            "label": pipeline_label,
            "operation": "kernel pipeline",
        }
        details = _node_attr(merge, "details")
        if details:
            attrs["details"] = details
        group_node_attributes[pipeline_namespace] = attrs


def _resolve_section_tree_by_class(
    spec: ArchitectureSpec,
    class_name: str | None,
    *,
    basic_ops: BasicOpFilter,
) -> tuple[str, BlockNode] | None:
    if not class_name:
        return None
    matches = [
        (title, tree)
        for title, tree in architecture_section_trees(spec)
        if tree.class_name == class_name and subgraph_warrants_json_export(tree, basic_ops=basic_ops)
    ]
    if not matches:
        return None
    return matches[0]


def _resolve_section_tree_for_component(
    spec: ArchitectureSpec,
    component: BlockComponent,
    *,
    variant: LayerVariant | None,
    basic_ops: BasicOpFilter,
) -> tuple[str, BlockNode] | None:
    if variant is not None:
        if component.role == "attention" and variant.attention_class:
            resolved = _resolve_section_tree_by_class(
                spec,
                variant.attention_class,
                basic_ops=basic_ops,
            )
            if resolved is not None:
                return resolved
        if component.role in {"moe", "ffn"}:
            if variant.ffn_class:
                resolved = _resolve_section_tree_by_class(
                    spec,
                    variant.ffn_class,
                    basic_ops=basic_ops,
                )
                if resolved is not None:
                    return resolved
            if variant.ffn_attr:
                resolved = _resolve_section_tree(
                    spec,
                    variant.ffn_attr,
                    component_label=variant.ffn_label,
                    basic_ops=basic_ops,
                )
                if resolved is not None:
                    return resolved
    return _resolve_section_tree(
        spec,
        component.attr_name,
        component_label=component.label,
        basic_ops=basic_ops,
    )


def _variant_namespace_slug(variant: LayerVariant) -> str:
    attention = variant.attention_class or variant.attention_label
    ffn = variant.ffn_class or variant.ffn_label
    return _sanitize_namespace_segment(f"{variant.count}x_{attention}_{ffn}")


def _variant_group_label(variant: LayerVariant) -> str:
    attention = variant.attention_class or variant.attention_label
    ffn = variant.ffn_class or variant.ffn_label
    return f"{variant.count}× {attention} + {ffn}"


def _section_namespace_for_component(
    spec: ArchitectureSpec,
    component: BlockComponent,
    *,
    variant: LayerVariant | None,
    namespace_prefix: str,
) -> str:
    if component.role == "norm" and component.attr_name in _DECODER_NORM_ATTRS:
        return namespace_prefix
    segment = _section_namespace_segment(spec, component, variant=variant)
    return _join_namespace(namespace_prefix, segment)


def _group_node_label(spec: ArchitectureSpec, component: BlockComponent) -> str:
    if component.role == "norm":
        return spec.norm_type or "RMSNorm"
    return _display_label(component, spec)


def _resolve_section_tree(
    spec: ArchitectureSpec,
    attr_name: str,
    *,
    component_label: str,
    basic_ops: BasicOpFilter,
) -> tuple[str, BlockNode] | None:
    matches = [
        (title, tree)
        for title, tree in architecture_section_trees(spec)
        if tree.attr_name == attr_name and subgraph_warrants_json_export(tree, basic_ops=basic_ops)
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    for title, tree in matches:
        if component_label and (component_label in title or title in component_label):
            return (title, tree)
    return max(matches, key=lambda item: len(item[1].children or ()))


def _summary_node(
    node_id: str,
    label: str,
    *,
    namespace: str,
    component: BlockComponent,
) -> dict[str, Any]:
    style = spine_tile_style()
    node: dict[str, Any] = {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [
            {"key": "attr_name", "value": component.attr_name},
            {"key": "class_name", "value": component.class_name},
            {"key": "role", "value": component.role},
        ],
    }
    if style:
        node["style"] = ensure_readable_text(style)
    return node


def _append_section(
    merged_nodes: list[dict[str, Any]],
    *,
    spec: ArchitectureSpec,
    component: BlockComponent,
    id_prefix: str,
    namespace_prefix: str,
    basic_ops: BasicOpFilter,
    previous_exits: list[str],
    variant: LayerVariant | None = None,
    group_node_attributes: dict[str, dict[str, str]] | None = None,
    shape_inferencer: ShapeInferencer | None = None,
) -> list[str]:
    if not component_has_detail_section(component, spec):
        summary = _summary_node(
            id_prefix,
            _group_node_label(spec, component)
            if component.role == "norm"
            else _display_label(component, spec),
            namespace=_flat_spine_namespace(
                component,
                namespace_prefix,
                variant=variant,
            ),
            component=component,
        )
        if previous_exits:
            summary["incomingEdges"] = [
                {
                    "sourceNodeId": source_id,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
                for source_id in previous_exits
            ]
        merged_nodes.append(summary)
        return [id_prefix]

    resolved = _resolve_section_tree_for_component(
        spec,
        component,
        variant=variant,
        basic_ops=basic_ops,
    )
    if resolved is None:
        summary = _summary_node(
            id_prefix,
            _display_label(component, spec),
            namespace=_flat_spine_namespace(
                component,
                namespace_prefix,
                variant=variant,
            ),
            component=component,
        )
        if previous_exits:
            summary["incomingEdges"] = [
                {
                    "sourceNodeId": source_id,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
                for source_id in previous_exits
            ]
        merged_nodes.append(summary)
        return [id_prefix]

    _title, block_tree = resolved
    computation = build_computation_graph(block_tree, basic_ops=basic_ops)
    skip_variant_root_input = _skip_variant_root_input(component)
    section_nodes = _computation_nodes(
        computation,
        id_prefix=id_prefix,
        namespace_prefix=namespace_prefix,
        skip_synthetic_input=skip_variant_root_input,
    )
    if shape_inferencer is not None:
        annotate_nodes_with_shapes(
            section_nodes,
            infer_block_tree_shapes(shape_inferencer, block_tree, title=_title),
            id_prefix=id_prefix,
        )

    seen_ids = {node["id"] for node in merged_nodes}
    pipeline_inject_skip: set[str] = set()
    for nested_label, nested_block in collect_nested_diagrams(block_tree, basic_ops=basic_ops):
        nested_prefix = _merge_node_id(id_prefix, nested_block.attr_name)
        if any(node["id"].startswith(f"{nested_prefix}/") for node in section_nodes):
            continue
        nested_namespace = _join_namespace(
            namespace_prefix,
            _nested_namespace_segment(nested_block, nested_label),
        )
        nested_computation = build_computation_graph(nested_block, basic_ops=basic_ops)
        section_nodes.extend(
            _computation_nodes(
                nested_computation,
                id_prefix=nested_prefix,
                namespace_prefix=nested_namespace,
            )
        )
        if shape_inferencer is not None:
            annotate_nodes_with_shapes(
                section_nodes,
                infer_block_tree_shapes(shape_inferencer, nested_block, title=nested_label),
                id_prefix=nested_prefix,
            )
        if nested_block.class_name == "KernelPipeline":
            _integrate_kernel_pipeline_merge(
                section_nodes,
                namespace_prefix=namespace_prefix,
                pipeline_namespace=nested_namespace,
                pipeline_prefix=nested_prefix,
                pipeline_label=nested_label,
                group_node_attributes=group_node_attributes,
                inject_skip=pipeline_inject_skip,
            )

    section_nodes = [node for node in section_nodes if node["id"] not in seen_ids]
    inject_skip = set(pipeline_inject_skip)
    if skip_variant_root_input:
        inject_skip.add(namespace_prefix)
    _inject_group_inputs(section_nodes, skip_namespaces=frozenset(inject_skip))
    _connect_external_inputs(
        section_nodes,
        namespace_prefix=namespace_prefix,
        previous_exits=previous_exits,
    )
    _entries, exits = _boundary_nodes(section_nodes)

    merged_nodes.extend(section_nodes)
    apply_kernel_frame_labels(section_nodes, group_node_attributes)
    if (
        group_node_attributes is not None
        and component.role == "norm"
        and component.attr_name not in _DECODER_NORM_ATTRS
    ):
        group_node_attributes[namespace_prefix] = {
            "label": _group_node_label(spec, component),
            "operation": spec.norm_type or "RMSNorm",
        }
    return exits


def _append_variant_layer(
    merged_nodes: list[dict[str, Any]],
    *,
    spec: ArchitectureSpec,
    variant: LayerVariant,
    id_prefix: str,
    namespace_prefix: str,
    basic_ops: BasicOpFilter,
    previous_exits: list[str],
    group_node_attributes: dict[str, dict[str, str]],
    shape_inferencer: ShapeInferencer | None = None,
) -> list[str]:
    chain_exits = list(previous_exits)
    for component in _ordered_decoder_components(spec):
        section_prefix = _merge_node_id(id_prefix, component.attr_name)
        section_namespace = _section_namespace_for_component(
            spec,
            component,
            variant=variant,
            namespace_prefix=namespace_prefix,
        )
        chain_exits = _append_section(
            merged_nodes,
            spec=spec,
            component=component,
            id_prefix=section_prefix,
            namespace_prefix=section_namespace,
            basic_ops=basic_ops,
            previous_exits=chain_exits,
            variant=variant,
            group_node_attributes=group_node_attributes,
            shape_inferencer=shape_inferencer,
        )
    return chain_exits


def _append_decoder_layers(
    merged_nodes: list[dict[str, Any]],
    *,
    spec: ArchitectureSpec,
    decoder_namespace: str,
    basic_ops: BasicOpFilter,
    previous_exits: list[str],
    group_node_configs: list[dict[str, Any]],
    group_node_attributes: dict[str, dict[str, str]],
    shape_inferencer: ShapeInferencer | None = None,
) -> list[str]:
    if spec.layer_variants:
        variant_exits: list[str] = []
        for variant in spec.layer_variants:
            slug = _variant_namespace_slug(variant)
            variant_prefix = _merge_node_id("decoder", slug)
            variant_namespace = _join_namespace(decoder_namespace, slug)
            group_node_attributes[variant_namespace] = {
                "label": _variant_group_label(variant),
                "count": str(variant.count),
                "attention": variant.attention_class or variant.attention_label,
                "ffn": variant.ffn_class or variant.ffn_label,
            }
            group_node_configs.append(
                {
                    "namespaceRegex": f"^{re.escape(variant_namespace)}$",
                    "backgroundColor": "#fff5f4",
                    "borderColor": "#c0392b",
                    "textColor": "#1a1a1a",
                    "layoutDirection": "TOP_BOTTOM",
                }
            )
            exits = _append_variant_layer(
                merged_nodes,
                spec=spec,
                variant=variant,
                id_prefix=variant_prefix,
                namespace_prefix=variant_namespace,
                basic_ops=basic_ops,
                previous_exits=previous_exits,
                group_node_attributes=group_node_attributes,
                shape_inferencer=shape_inferencer,
            )
            variant_exits.extend(exits)
        return variant_exits or list(previous_exits)

    chain_exits = list(previous_exits)
    for component in _ordered_decoder_components(spec):
        section_prefix = _merge_node_id("decoder", component.attr_name)
        section_namespace = _section_namespace_for_component(
            spec,
            component,
            variant=None,
            namespace_prefix=decoder_namespace,
        )
        group = _group_config_for_role(section_namespace, component.role)
        if group:
            group_node_configs.append(group)
        chain_exits = _append_section(
            merged_nodes,
            spec=spec,
            component=component,
            id_prefix=section_prefix,
            namespace_prefix=section_namespace,
            basic_ops=basic_ops,
            previous_exits=chain_exits,
            group_node_attributes=group_node_attributes,
            shape_inferencer=shape_inferencer,
        )
    return chain_exits


def _group_config_for_role(namespace: str, role: str) -> dict[str, Any] | None:
    style = ROLE_COLORS.get(role)
    if style is None or not namespace:
        return None
    style = ensure_readable_text(style)
    config: dict[str, Any] = {
        "namespaceRegex": f"^{re.escape(namespace)}$",
        "backgroundColor": style["backgroundColor"],
        "textColor": style["textColor"],
        "layoutDirection": "TOP_BOTTOM",
    }
    if role == "moe":
        config["borderColor"] = _GPU_KERNEL_BORDER
    elif role == "ffn":
        config["borderColor"] = "#566573"
    return config


def build_merged_model_graph(
    spec: ArchitectureSpec,
    *,
    basic_ops: BasicOpFilter | None = None,
    graph_id: str = "model",
    shape_inferencer: ShapeInferencer | None = None,
) -> dict[str, Any]:
    """Build a single graph with overview spine and inlined computation subgraphs."""
    resolved_basic_ops = basic_ops or spec.basic_ops
    nodes: list[dict[str, Any]] = [
        {
            "id": "@input",
            "label": "Tokenized text",
            "namespace": "",
            "attrs": [{"key": "synthetic", "value": "@input"}],
            "style": ensure_readable_text(input_port_style()),
        }
    ]
    previous_exits = ["@input"]
    group_node_configs: list[dict[str, Any]] = []
    group_node_attributes: dict[str, dict[str, str]] = {}

    for component in _stack_pre_components(spec):
        expands = component_has_detail_section(component, spec)
        namespace_prefix = (
            _sanitize_namespace_segment(component.attr_name) if expands else ""
        )
        if expands:
            group = _group_config_for_role(namespace_prefix, component.role)
            if group:
                group_node_configs.append(group)
        previous_exits = _append_section(
            nodes,
            spec=spec,
            component=component,
            id_prefix=component.attr_name,
            namespace_prefix=namespace_prefix,
            basic_ops=resolved_basic_ops,
            previous_exits=previous_exits,
            shape_inferencer=shape_inferencer,
        )

    decoder_namespace = _decoder_namespace(spec)
    previous_exits = _append_decoder_layers(
        nodes,
        spec=spec,
        decoder_namespace=decoder_namespace,
        basic_ops=resolved_basic_ops,
        previous_exits=previous_exits,
        group_node_configs=group_node_configs,
        group_node_attributes=group_node_attributes,
        shape_inferencer=shape_inferencer,
    )

    for component in _stack_tail_components(spec):
        expands = component_has_detail_section(component, spec)
        namespace_prefix = (
            _sanitize_namespace_segment(component.attr_name) if expands else ""
        )
        if expands:
            group = _group_config_for_role(namespace_prefix, component.role)
            if group:
                group_node_configs.append(group)
        previous_exits = _append_section(
            nodes,
            spec=spec,
            component=component,
            id_prefix=component.attr_name,
            namespace_prefix=namespace_prefix,
            basic_ops=resolved_basic_ops,
            previous_exits=previous_exits,
            shape_inferencer=shape_inferencer,
        )

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
    if spec.layer_mix:
        model_attrs["layer_mix"] = spec.layer_mix
    model_attrs.update(build_fact_sheet_group_attributes(spec))

    finalize_graph_node_styles(nodes)

    return {
        "id": graph_id,
        "nodes": nodes,
        "groupNodeAttributes": {
            "": model_attrs,
            decoder_namespace: {
                "repeat": decoder_namespace,
                "forward": " → ".join(spec.forward_sequence or []),
                **({"layer_mix": spec.layer_mix} if spec.layer_mix else {}),
            },
            **group_node_attributes,
        },
        "groupNodeConfigs": build_group_node_configs(
            decoder_namespace=decoder_namespace,
            group_node_attributes={
                "": model_attrs,
                decoder_namespace: {
                    "repeat": decoder_namespace,
                },
                **group_node_attributes,
            },
            role_configs=group_node_configs,
        ),
    }
