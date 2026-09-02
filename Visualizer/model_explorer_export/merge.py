###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build one merged Model Explorer graph with in-place namespace expansion."""

from __future__ import annotations

import re
from typing import Any

from visualizer.basic_ops import BasicOpFilter
from visualizer.ast_analyze import (
    _pick_stack_model_class,
    expand_class_forward_dataflow,
    stack_entry_dataflow,
)
from visualizer.block_tree import (
    BlockNode,
    collect_nested_diagrams,
    expand_block_tree_inplace,
    subgraph_warrants_json_export,
)
from visualizer.blocks import BlockComponent, LayerVariant
from visualizer.computation_graph import ComputationGraph, build_computation_graph
from visualizer.extract import ArchitectureSpec, architecture_section_trees
from visualizer.shape_inference import ShapeInferencer, Symbol, TensorSpec

from model_explorer_export.adapter import (
    _incoming_edges,
    _node_attrs,
    _node_namespaces,
    _node_style,
    _output_port_metadata,
    _sanitize_namespace_segment,
)
from model_explorer_export.fact_sheet import build_fact_sheet_group_attributes
from model_explorer_export.shapes import (
    annotate_nodes_with_shapes,
    apply_shape_attrs,
    fill_missing_node_shapes,
    group_boundary_shapes,
    infer_block_tree_shapes,
    node_output_spec,
)
from model_explorer_export.labels import (
    apply_kernel_frame_labels,
    skip_merged_tensor_port_parent,
    tensor_port_input_label,
)
from model_explorer_export.overview import (
    _DECODER_NORM_ATTRS,
    _component_uses_variant_attention_class,
    _component_uses_variant_ffn_class,
    _decoder_namespace,
    _display_label,
    _flat_spine_namespace,
    _ordered_decoder_components,
    _section_namespace_segment,
    _stack_pre_components,
    _stack_tail_components,
    component_has_detail_section,
    format_forward_sequence,
)
from model_explorer_export.styles import (
    ROLE_COLORS,
    _GPU_KERNEL_BORDER,
    build_group_node_configs,
    ensure_readable_text,
    finalize_graph_node_styles,
    input_port_style,
    output_port_style,
    spine_tile_style,
)

_SKIPPED_SECTION_INPUT = "@skipped_section_input"
SourceRef = str | tuple[str, str]


def _config_leaf(config: dict[str, Any], name: str) -> Any:
    if name in config:
        return config[name]
    for value in config.values():
        if isinstance(value, dict):
            resolved = _config_leaf(value, name)
            if resolved is not None:
                return resolved
    return None


def _operation_detail(operation: Any, key: str) -> str | None:
    prefix = f"{key}:"
    return next(
        (
            str(detail)[len(prefix) :].strip()
            for detail in operation.details
            if str(detail).startswith(prefix)
        ),
        None,
    )


def _data_movement_shape(
    operation: Any,
    source: TensorSpec | None,
    *,
    spec: ArchitectureSpec,
) -> TensorSpec | None:
    if source is None:
        return None
    label = str(operation.label)
    if label == "Unsqueeze":
        dim_text = _operation_detail(operation, "dim")
        if dim_text is None:
            return source
        try:
            dim = int(dim_text)
        except ValueError:
            return source
        if dim < 0:
            dim += len(source.shape) + 1
        dim = max(0, min(dim, len(source.shape)))
        shape = list(source.shape)
        shape.insert(dim, 1)
        return TensorSpec(tuple(shape), source.dtype)
    if label in {"Expand", "Reshape", "View"}:
        shape_text = _operation_detail(operation, "shape")
        if not shape_text:
            return source
        resolved: list[Any] = []
        parts = [part.strip() for part in shape_text.split(",")]
        for index, part in enumerate(parts):
            if part == "-1" and index < len(source.shape):
                resolved.append(source.shape[index])
            elif re.fullmatch(r"-?\d+", part):
                resolved.append(int(part))
            elif ".config." in part:
                value = _config_leaf(spec.raw_config, part.rsplit(".", 1)[-1])
                resolved.append(value if value is not None else part)
            else:
                resolved.append(part)
        return TensorSpec(tuple(resolved), source.dtype)
    return source


def _append_stack_entry_dataflow(
    nodes: list[dict[str, Any]],
    *,
    spec: ArchitectureSpec,
    module_sources: dict[str, SourceRef],
    shape_inferencer: ShapeInferencer | None,
) -> list[SourceRef] | None:
    cls = spec.class_registry.get(spec.stack_model_class or "")
    if cls is None:
        cls = _pick_stack_model_class(spec.class_registry, None)
    dataflow = stack_entry_dataflow(cls) if cls is not None else None
    if dataflow is None:
        return None

    refs: dict[str, SourceRef] = dict(module_sources)
    node_by_id = {str(node.get("id", "")): node for node in nodes}
    ref_specs: dict[str, TensorSpec] = {}
    if shape_inferencer is not None:
        context = shape_inferencer.context
        for component in spec.stack_pre:
            if component.role != "embedding" or component.attr_name not in refs:
                continue
            hidden = context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            ref_specs[component.attr_name] = TensorSpec(
                (Symbol.BATCH.value, Symbol.SEQ.value, hidden),
                context.dtype,
            )
    for operation in dataflow.operations:
        source_pairs = [
            (predecessor, refs[predecessor])
            for predecessor in operation.predecessors
            if predecessor in refs
        ]
        sources = [source for _predecessor, source in source_pairs]
        if not sources:
            continue
        node_id = _merge_node_id("@model_forward", operation.attr_name)
        node: dict[str, Any] = {
            "id": node_id,
            "label": operation.label,
            "namespace": "",
            "attrs": [
                {"key": "operation", "value": "source"},
                *[
                    {"key": "detail", "value": str(detail)}
                    for detail in operation.details
                ],
            ],
            "incomingEdges": [
                _source_edge(source, str(index))
                for index, source in enumerate(sources)
            ],
        }
        if shape_inferencer is not None:
            predecessor, source = source_pairs[0]
            source_spec = ref_specs.get(predecessor)
            if source_spec is None:
                source_id, source_port = _source_parts(source)
                source_node = node_by_id.get(source_id)
                source_spec = (
                    node_output_spec(source_node, source_port)
                    if source_node is not None
                    else None
                )
            operation_spec = _data_movement_shape(
                operation, source_spec, spec=spec
            )
            if operation_spec is not None:
                apply_shape_attrs(node, operation_spec)
                ref_specs[operation.attr_name] = operation_spec
        nodes.append(node)
        node_by_id[node_id] = node
        refs[operation.attr_name] = node_id
    output = refs.get(dataflow.output_producer)
    return [output] if output is not None else None


def _source_parts(source: SourceRef) -> tuple[str, str]:
    return source if isinstance(source, tuple) else (source, "0")


def _source_edge(source: SourceRef, target_input_id: str) -> dict[str, str]:
    source_id, output_port = _source_parts(source)
    return {
        "sourceNodeId": source_id,
        "sourceNodeOutputId": output_port,
        "targetNodeInputId": target_input_id,
    }


def _port_output_id(base_id: str, port: str, port_count: int) -> str:
    return base_id if port_count == 1 else f"{base_id}:{port}"


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


def _is_synthetic_output(node: dict[str, Any]) -> bool:
    node_id = node.get("id", "")
    if node_id == "@output" or node_id.endswith("/@output"):
        return True
    return _node_attr(node, "synthetic") == "@output"


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
        attrs = [
            attr for attr in item.get("attrs", []) if attr.get("key") != "port_label"
        ]
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


def _make_group_output_node(
    *,
    output_id: str,
    namespace: str,
    ports: list[tuple[str, str, str]],
) -> dict[str, Any]:
    metadata = [
        {
            "id": port,
            "attrs": [{"key": "port_label", "value": port}],
        }
        for port, _source, _source_port in ports
    ]
    return {
        "id": output_id,
        "label": "Output",
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": "@output"}],
        "style": ensure_readable_text(output_port_style()),
        "incomingEdges": [
            {
                "sourceNodeId": source,
                "sourceNodeOutputId": source_port,
                "targetNodeInputId": port,
                "metadata": {"port_label": port},
            }
            for port, source, source_port in ports
        ],
        "inputsMetadata": [dict(item) for item in metadata],
        "outputsMetadata": metadata,
    }


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
        local_id: _merge_node_id(id_prefix, local_id)
        for local_id in index_to_local.values()
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
        if skip_synthetic_input and (
            local_id == "@input" or spec.synthetic == "@input"
        ):
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
                remapped = dict(edge)
                remapped["sourceNodeId"] = _SKIPPED_SECTION_INPUT
                remapped_incoming.append(remapped)
                continue
            remapped = dict(edge)
            remapped["sourceNodeId"] = local_to_prefixed[source_local]
            remapped_incoming.append(remapped)
        if remapped_incoming:
            node["incomingEdges"] = remapped_incoming
        if index == computation.output_node_index:
            ports = _output_port_metadata(computation)
            if len(ports) > 1:
                for port_metadata in ports:
                    port = str(port_metadata.get("id", "result"))
                    port_node = dict(node)
                    port_node["id"] = _port_output_id(prefixed_id, port, len(ports))
                    port_node["incomingEdges"] = [
                        edge
                        for edge in remapped_incoming
                        if str(edge.get("targetNodeInputId", "")) == port
                    ]
                    port_node["inputsMetadata"] = [dict(port_metadata)]
                    port_node["outputsMetadata"] = [dict(port_metadata)]
                    nodes.append(port_node)
                continue
            node["inputsMetadata"] = [dict(port) for port in ports]
            node["outputsMetadata"] = ports
        nodes.append(node)

    return nodes


def _boundary_nodes(nodes: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    node_ids = {node["id"] for node in nodes}
    sources = {
        edge["sourceNodeId"] for node in nodes for edge in node.get("incomingEdges", [])
    }
    targets = {node["id"] for node in nodes for edge in node.get("incomingEdges", [])}
    entries = sorted(node_id for node_id in node_ids if node_id not in targets)
    exits = sorted(node_id for node_id in node_ids if node_id not in sources)
    return entries or sorted(node_ids)[:1], exits or sorted(node_ids)[-1:]


def _block_tile_ids(computation: ComputationGraph, *, id_prefix: str) -> dict[int, str]:
    """Merged node id of the tile each block renders as, keyed by block identity."""
    tiles: dict[int, str] = {}
    for index, spec in enumerate(computation.nodes):
        if spec.block is None:
            continue
        tiles.setdefault(
            id(spec.block), _merge_node_id(id_prefix, spec.key or f"node:{index}")
        )
    return tiles


def _replace_tile_with_group(
    section_nodes: list[dict[str, Any]],
    nested_nodes: list[dict[str, Any]],
    *,
    tile_id: str,
    exit_ref: SourceRef | None = None,
    exit_id: str | None = None,
) -> None:
    """Put a nested diagram where its collapsed tile sat, instead of beside the section.

    Model Explorer draws an unconnected group as a floating island, so the tile's
    producers feed the group's input port and its consumers read the group's output.
    """
    tile = next((node for node in section_nodes if node["id"] == tile_id), None)
    if tile is None:
        return
    incoming = tile.get("incomingEdges", [])
    entries = [node for node in nested_nodes if _is_primary_section_input(node)]
    if not entries:
        entry_ids, _exits = _boundary_nodes(nested_nodes)
        entries = [node for node in nested_nodes if node["id"] in set(entry_ids)]
    for entry in entries:
        if incoming and not entry.get("incomingEdges"):
            entry["incomingEdges"] = [dict(edge) for edge in incoming]

    section_nodes.remove(tile)
    if exit_ref is None:
        exit_ref = exit_id
    if exit_ref is None:
        return
    exit_id, exit_port = _source_parts(exit_ref)
    for node in section_nodes:
        for edge in node.get("incomingEdges", []):
            if edge.get("sourceNodeId") == tile_id:
                edge["sourceNodeId"] = exit_id
                edge["sourceNodeOutputId"] = exit_port


def _section_exits(
    computation: ComputationGraph,
    section_nodes: list[dict[str, Any]],
    *,
    id_prefix: str,
    replacements: dict[str, SourceRef] | None = None,
) -> list[SourceRef]:
    """Return the section output node(s) that feed the next spine step.

    Branchy inline graphs (for example mHC hyperconnections) can leave several
    terminal ops (`post`, `comb`, `collapsed`). Only the primary output should
    continue on the main forward path.
    """
    node_ids = {node["id"] for node in section_nodes}
    if computation.output_node_index is not None:
        output_local = computation.nodes[computation.output_node_index].key or "@output"
        output_id = _merge_node_id(id_prefix, output_local)
        output_id = _port_output_id(
            output_id,
            computation.primary_output_port or "result",
            len(computation.output_ports),
        )
        if output_id in node_ids:
            return [
                (
                    output_id,
                    computation.primary_output_port or "result",
                )
            ]
    if computation.primary_output_index is not None:
        index_to_local = {
            index: spec.key or f"node:{index}"
            for index, spec in enumerate(computation.nodes)
        }
        primary_local = index_to_local.get(computation.primary_output_index)
        if primary_local is not None:
            primary_id = _merge_node_id(id_prefix, primary_local)
            replacement = (replacements or {}).get(primary_id, primary_id)
            replacement_id, _port = _source_parts(replacement)
            if replacement_id in node_ids:
                return [replacement]
    _entries, exits = _boundary_nodes(section_nodes)
    return exits


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
        if _is_primary_section_input(node)
        and node.get("namespace", "") == namespace_prefix
    ]


def _connect_external_inputs(
    section_nodes: list[dict[str, Any]],
    *,
    namespace_prefix: str,
    previous_exits: list[SourceRef],
) -> None:
    if not previous_exits:
        return
    placeholder_targets = [
        node
        for node in section_nodes
        if any(
            edge.get("sourceNodeId") == _SKIPPED_SECTION_INPUT
            for edge in node.get("incomingEdges", [])
        )
    ]
    if placeholder_targets:
        for node in placeholder_targets:
            preserved = [
                edge
                for edge in node.get("incomingEdges", [])
                if edge.get("sourceNodeId") != _SKIPPED_SECTION_INPUT
            ]
            replacement_edges = [
                _source_edge(source, str(index))
                for index, source in enumerate(previous_exits)
            ]
            node["incomingEdges"] = [*preserved, *replacement_edges]
        return
    input_ids = _section_input_nodes(section_nodes, namespace_prefix)
    connect_targets = input_ids or _boundary_nodes(section_nodes)[0]
    node_by_id = {node["id"]: node for node in section_nodes}
    for target_id in connect_targets:
        target = node_by_id.get(target_id)
        if target is None:
            continue
        target["incomingEdges"] = [
            _source_edge(source, str(index))
            for index, source in enumerate(previous_exits)
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
    """Actual expanded decoder blocks keep their explicit Input boundary."""
    del component
    return False


def _namespace_is_descendant(node_namespace: str, group_namespace: str) -> bool:
    if not group_namespace:
        return node_namespace == ""
    return node_namespace == group_namespace or node_namespace.startswith(
        f"{group_namespace}/"
    )


def _namespace_internal_ids(
    section_nodes: list[dict[str, Any]], namespace: str
) -> set[str]:
    """Include nested namespaces so SituAndMul ops stay inside KimiMLP groups."""
    return {
        node["id"]
        for node in section_nodes
        if _namespace_is_descendant(node.get("namespace", ""), namespace)
    }


_INLINE_FRAME_NAMESPACE_SUFFIXES = frozenset({"SituAndMul", "SiluAndMul"})


def _skip_nested_inline_frame_input(
    section_nodes: list[dict[str, Any]], namespace: str
) -> bool:
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
    namespaces = sorted(
        {node.get("namespace", "") for node in section_nodes if node.get("namespace")}
    )

    for namespace in namespaces:
        if namespace in skip_namespaces:
            continue
        group_nodes = [
            node for node in section_nodes if node.get("namespace", "") == namespace
        ]
        if any(_is_synthetic_input(node) for node in group_nodes):
            continue
        if _skip_nested_inline_frame_input(section_nodes, namespace):
            continue
        if skip_merged_tensor_port_parent(namespace, group_nodes):
            continue

        internal_ids = _namespace_internal_ids(section_nodes, namespace)
        entry_ports = _collect_group_entry_ports(group_nodes, internal_ids, node_by_id)
        entry_nodes: list[dict[str, Any]] = []
        outside_sources: set[tuple[str, str]] = set()

        for node in group_nodes:
            incoming = list(node.get("incomingEdges", []))
            external = [
                edge for edge in incoming if edge["sourceNodeId"] not in internal_ids
            ]
            if external or not incoming:
                entry_nodes.append(node)
                outside_sources.update(
                    (
                        edge["sourceNodeId"],
                        edge.get("sourceNodeOutputId", "0"),
                    )
                    for edge in external
                )

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
            label=_infer_group_input_label(
                group_nodes, namespace, entry_ports=entry_ports
            ),
            namespace=namespace,
        )
        if outside_sources:
            input_node["incomingEdges"] = [
                {
                    "sourceNodeId": source_id,
                    "sourceNodeOutputId": source_port,
                    "targetNodeInputId": str(index),
                }
                for index, (source_id, source_port) in enumerate(
                    sorted(outside_sources)
                )
            ]

        for entry in entry_nodes:
            incoming = list(entry.get("incomingEdges", []))
            internal = [
                edge for edge in incoming if edge["sourceNodeId"] in internal_ids
            ]
            entry["incomingEdges"] = internal + [
                {
                    "sourceNodeId": input_id,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": str(len(internal)),
                }
            ]

        section_nodes.append(input_node)
        node_by_id[input_id] = input_node


def _tile_prefix_attr_name(prefix: str) -> str:
    """Recover the submodule attribute encoded in a tile id.

    Tile ids carry their slot and producer (``seq:7:forget_gate``,
    ``sidefeed:11:o_norm:@op_l35``); only the attribute names the tensor.
    """
    segment = prefix.rsplit("/", 1)[-1]
    parts = segment.split(":")
    if len(parts) >= 3 and parts[0] in {"seq", "sidefeed", "branch"}:
        return parts[2]
    return segment


def _inject_group_outputs(
    section_nodes: list[dict[str, Any]],
    *,
    output_names: dict[str, str] | None = None,
    resolve_name: Any = None,
) -> None:
    """Give every visible Input namespace a matching Output boundary."""
    input_namespaces = {
        node.get("namespace", "") for node in section_nodes if _is_synthetic_input(node)
    }
    for namespace in sorted(
        (item for item in input_namespaces if item),
        key=lambda item: item.count("/"),
        reverse=True,
    ):
        if any(
            _is_synthetic_output(node) and node.get("namespace", "") == namespace
            for node in section_nodes
        ):
            continue
        input_node = next(
            node
            for node in section_nodes
            if _is_synthetic_input(node) and node.get("namespace", "") == namespace
        )
        prefix = input_node["id"].split("/@input", 1)[0]
        output_id = f"{prefix}/@output"
        internal_ids = _namespace_internal_ids(section_nodes, namespace)
        outgoing: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for target in section_nodes:
            if target["id"] in internal_ids:
                continue
            for edge in target.get("incomingEdges", []):
                if edge.get("sourceNodeId") in internal_ids:
                    outgoing.append((target, edge))

        sources: list[tuple[str, str]] = []
        for _target, edge in outgoing:
            source = (
                edge["sourceNodeId"],
                edge.get("sourceNodeOutputId", "0"),
            )
            if source not in sources:
                sources.append(source)
        if not sources:
            graph_sources = {
                edge["sourceNodeId"]
                for node in section_nodes
                if node["id"] in internal_ids
                for edge in node.get("incomingEdges", [])
                if edge.get("sourceNodeId") in internal_ids
            }
            sources = [
                (node["id"], "0")
                for node in section_nodes
                if node["id"] in internal_ids
                and node["id"] not in graph_sources
                and not _is_synthetic_input(node)
                and not _is_synthetic_output(node)
            ]
        if not sources:
            continue

        name = (output_names or {}).get(prefix)
        if not name and resolve_name is not None:
            name = resolve_name(prefix)
        name = name or "result"
        ports = [
            (
                name if len(sources) == 1 else f"{name}_{index + 1}",
                source,
                source_port,
            )
            for index, (source, source_port) in enumerate(sources)
        ]
        port_by_source = {
            (source, source_port): port for port, source, source_port in ports
        }
        for _target, edge in outgoing:
            source = (
                edge["sourceNodeId"],
                edge.get("sourceNodeOutputId", "0"),
            )
            edge["sourceNodeId"] = output_id
            edge["sourceNodeOutputId"] = port_by_source[source]

        section_nodes.append(
            _make_group_output_node(
                output_id=output_id,
                namespace=namespace,
                ports=ports,
            )
        )


def _flatten_transparent_group_inputs(section_nodes: list[dict[str, Any]]) -> None:
    """Remove Input nodes from inline expansions that have no block Output."""
    output_namespaces = {
        str(node.get("namespace", ""))
        for node in section_nodes
        if _is_synthetic_output(node)
    }
    transparent_inputs = [
        node
        for node in section_nodes
        if _is_primary_section_input(node)
        and str(node.get("label", "")).lower() in {"hidden_states", "input"}
        and str(node.get("namespace", "")) not in output_namespaces
    ]
    for input_node in transparent_inputs:
        input_id = str(input_node["id"])
        incoming = list(input_node.get("incomingEdges", []))
        if not incoming:
            continue
        for target in section_nodes:
            replacement: list[dict[str, Any]] = []
            changed = False
            for edge in target.get("incomingEdges", []):
                if edge.get("sourceNodeId") != input_id:
                    replacement.append(edge)
                    continue
                changed = True
                target_port = str(edge.get("targetNodeInputId", "0"))
                for index, source_edge in enumerate(incoming):
                    rewired = dict(source_edge)
                    rewired["targetNodeInputId"] = (
                        target_port if len(incoming) == 1 else f"{target_port}_{index}"
                    )
                    replacement.append(rewired)
            if changed:
                target["incomingEdges"] = replacement
        section_nodes.remove(input_node)


_GENERIC_OUTPUT_PORT_RE = re.compile(r"^result(_\d+)?$")


def _caller_output_name(
    spec: ArchitectureSpec,
    parent_class: str | None,
    attr_name: str,
) -> str | None:
    """Tensor name the calling module binds this submodule's result to."""
    parent = spec.class_registry.get(parent_class or "")
    if parent is None:
        return None
    return parent.forward_call_output_names.get(attr_name)


def _rename_generic_output_ports(
    nodes: list[dict[str, Any]],
    *,
    output_id: str,
    name: str,
) -> dict[str, str]:
    """Replace placeholder ``result`` ports with the caller's tensor name.

    Modules returning a bare expression carry no name of their own, so the
    boundary would otherwise read ``result`` instead of the tensor the caller
    actually threads onward.
    """
    output = next((node for node in nodes if node.get("id") == output_id), None)
    if output is None:
        return {}
    ports = [str(item.get("id")) for item in output.get("outputsMetadata", [])]
    renames = {
        port: (name if len(ports) == 1 else f"{name}_{index + 1}")
        for index, port in enumerate(ports)
        if _GENERIC_OUTPUT_PORT_RE.match(port)
    }
    if not renames:
        return {}
    for key in ("inputsMetadata", "outputsMetadata"):
        for metadata in output.get(key, []):
            renamed = renames.get(str(metadata.get("id")))
            if not renamed:
                continue
            metadata["id"] = renamed
            for attr in metadata.get("attrs", []):
                if attr.get("key") == "port_label":
                    attr["value"] = renamed
    for edge in output.get("incomingEdges", []):
        renamed = renames.get(str(edge.get("targetNodeInputId", "")))
        if not renamed:
            continue
        edge["targetNodeInputId"] = renamed
        metadata = edge.get("metadata")
        if isinstance(metadata, dict) and "port_label" in metadata:
            metadata["port_label"] = renamed
    for node in nodes:
        for edge in node.get("incomingEdges", []):
            if str(edge.get("sourceNodeId")) != output_id:
                continue
            renamed = renames.get(str(edge.get("sourceNodeOutputId", "0")))
            if renamed:
                edge["sourceNodeOutputId"] = renamed
    return renames


def _remove_transparent_root_output(
    section_nodes: list[dict[str, Any]],
    *,
    id_prefix: str,
    primary_port: str | None,
) -> SourceRef | None:
    """Replace an inline-expanded root Output boundary with its real producer."""
    output_id = _merge_node_id(id_prefix, "@output")
    output = next(
        (node for node in section_nodes if node.get("id") == output_id),
        None,
    )
    if output is None:
        return None
    incoming = list(output.get("incomingEdges", []))
    selected = next(
        (
            edge
            for edge in incoming
            if str(edge.get("targetNodeInputId", "")) == str(primary_port or "")
        ),
        incoming[0] if incoming else None,
    )
    section_nodes.remove(output)
    if selected is None:
        return None
    return (
        str(selected["sourceNodeId"]),
        str(selected.get("sourceNodeOutputId", "0")),
    )


def _wrap_actual_group_boundary(
    nodes: list[dict[str, Any]],
    *,
    namespace: str,
    id_prefix: str,
    inputs: list[SourceRef],
    outputs: list[SourceRef],
    first_node_index: int,
) -> list[SourceRef]:
    """Put explicit boundaries around one real, non-transparent source block."""
    if not inputs or not outputs:
        return outputs
    input_id = _merge_node_id(id_prefix, "@input")
    input_node = _make_group_input_node(
        input_id=input_id,
        label="hidden_states",
        namespace=namespace,
        incoming_edges=[
            _source_edge(source, str(index)) for index, source in enumerate(inputs)
        ],
    )
    input_keys = {_source_parts(source) for source in inputs}
    for node in nodes[first_node_index:]:
        if not _namespace_is_descendant(str(node.get("namespace", "")), namespace):
            continue
        for edge in node.get("incomingEdges", []):
            source = (
                str(edge.get("sourceNodeId", "")),
                str(edge.get("sourceNodeOutputId", "0")),
            )
            if source in input_keys:
                edge["sourceNodeId"] = input_id
                edge["sourceNodeOutputId"] = "0"
    nodes.append(input_node)

    output_refs: list[SourceRef] = []
    for index, source in enumerate(outputs):
        port = "result" if len(outputs) == 1 else f"result_{index + 1}"
        output_id = _port_output_id(
            _merge_node_id(id_prefix, "@output"), port, len(outputs)
        )
        nodes.append(
            _make_group_output_node(
                output_id=output_id,
                namespace=namespace,
                ports=[(port, *_source_parts(source))],
            )
        )
        output_refs.append((output_id, port))
    return output_refs


def _label_boundary_outputs_by_port(nodes: list[dict[str, Any]]) -> None:
    """Name a single-port boundary Output after the tensor it carries.

    A multi-return module splits into one Output per slot, so a generic
    ``Output`` label would hide which of ``post``/``comb``/``collapsed`` each
    boundary is. Matching the mirror outside keeps one tensor reading the same
    on both sides of the block.
    """
    for node in nodes:
        if not _is_synthetic_output(node):
            continue
        ports = [str(item.get("id", "")) for item in node.get("outputsMetadata", [])]
        if len(ports) == 1 and ports[0]:
            node["label"] = ports[0]


def _mirror_boundary_outputs(nodes: list[dict[str, Any]]) -> None:
    """Pair each submodule Output port with a same-named node outside the block.

    The Output sits inside the block namespace, so consumers otherwise reach across
    the boundary. A mirror in the parent namespace gives the escaping tensor a
    visible identity on both sides of the block.
    """
    mirrors: list[dict[str, Any]] = []
    for node in list(nodes):
        if not _is_synthetic_output(node):
            continue
        namespace = str(node.get("namespace", ""))
        if not namespace:
            continue
        output_id = str(node.get("id"))
        parent_namespace = namespace.rsplit("/", 1)[0] if "/" in namespace else ""
        internal = _namespace_internal_ids(nodes, namespace)
        for metadata in node.get("outputsMetadata", []):
            port = str(metadata.get("id", ""))
            if not port:
                continue
            consumers = [
                (target, edge)
                for target in nodes
                if target["id"] not in internal
                for edge in target.get("incomingEdges", [])
                if str(edge.get("sourceNodeId")) == output_id
                and str(edge.get("sourceNodeOutputId", "0")) == port
            ]
            if not consumers:
                continue
            mirror_id = f"{output_id}^{port}"
            mirrors.append(
                {
                    "id": mirror_id,
                    "label": port,
                    "namespace": parent_namespace,
                    "attrs": [{"key": "synthetic", "value": "@output_mirror"}],
                    "style": ensure_readable_text(output_port_style()),
                    "incomingEdges": [
                        {
                            "sourceNodeId": output_id,
                            "sourceNodeOutputId": port,
                            "targetNodeInputId": port,
                        }
                    ],
                    "outputsMetadata": [
                        {
                            "id": port,
                            "attrs": [{"key": "port_label", "value": port}],
                        }
                    ],
                }
            )
            for _target, edge in consumers:
                edge["sourceNodeId"] = mirror_id
    nodes.extend(mirrors)


def _prune_unconsumed_outputs(nodes: list[dict[str, Any]]) -> None:
    """Strip unused boundary ports, then remove their dead producer subgraphs."""
    outgoing_ports: dict[str, set[str]] = {}
    for node in nodes:
        for edge in node.get("incomingEdges", []):
            outgoing_ports.setdefault(edge["sourceNodeId"], set()).add(
                str(edge.get("sourceNodeOutputId", "0"))
            )

    dead_candidates: set[str] = set()
    for node in nodes:
        if not _is_synthetic_output(node) or node.get("id") == "@output":
            continue
        used = outgoing_ports.get(str(node.get("id")), set())
        if not used:
            dead_candidates.add(str(node.get("id")))
        dead_candidates.update(
            str(edge["sourceNodeId"])
            for edge in node.get("incomingEdges", [])
            if str(edge.get("targetNodeInputId")) not in used
        )
        node["inputsMetadata"] = [
            metadata
            for metadata in node.get("inputsMetadata", [])
            if str(metadata.get("id")) in used
        ]
        node["outputsMetadata"] = [
            metadata
            for metadata in node.get("outputsMetadata", [])
            if str(metadata.get("id")) in used
        ]
        node["incomingEdges"] = [
            edge
            for edge in node.get("incomingEdges", [])
            if str(edge.get("targetNodeInputId")) in used
        ]

    node_by_id = {str(node.get("id")): node for node in nodes}
    outgoing_count: dict[str, int] = {}
    for node in nodes:
        for edge in node.get("incomingEdges", []):
            source_id = str(edge["sourceNodeId"])
            outgoing_count[source_id] = outgoing_count.get(source_id, 0) + 1
    dead: set[str] = set()
    pending = list(dead_candidates)
    while pending:
        node_id = pending.pop()
        if (
            node_id in dead
            or outgoing_count.get(node_id, 0)
            or node_id not in node_by_id
        ):
            continue
        node = node_by_id[node_id]
        if _is_synthetic_input(node):
            continue
        dead.add(node_id)
        for edge in node.get("incomingEdges", []):
            source_id = str(edge["sourceNodeId"])
            outgoing_count[source_id] = max(outgoing_count.get(source_id, 0) - 1, 0)
            pending.append(source_id)
    if dead:
        nodes[:] = [node for node in nodes if str(node.get("id")) not in dead]


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


def _kernel_pipeline_step(block_tree: BlockNode) -> BlockNode | None:
    """Return the kernel pipeline child of an attention block tree, if present."""
    return next(
        (
            child
            for child in block_tree.children
            if child.class_name == "KernelPipeline"
        ),
        None,
    )


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
    """Ensure kernel pipeline merge tiles expand and keep a stable group label."""
    del pipeline_prefix
    pipeline_nodes = [
        node
        for node in section_nodes
        if node.get("namespace", "").startswith(pipeline_namespace)
    ]
    merge_nodes = [
        node
        for node in section_nodes
        if _node_attr(node, "attr_name") == "@attn_pipeline"
        and _node_attr(node, "class_name") == "KernelPipeline"
        and node.get("namespace") in {namespace_prefix, pipeline_namespace}
    ]
    if not merge_nodes and not pipeline_nodes:
        return

    merge = merge_nodes[0] if merge_nodes else None
    merge_id = merge["id"] if merge is not None else None

    if inject_skip is not None:
        inject_skip.add(pipeline_namespace)

    tensor_by_label = {
        node.get("label", ""): node
        for node in section_nodes
        if node.get("namespace") == pipeline_namespace and _is_tensor_port(node)
    }

    if merge is not None and tensor_by_label:
        merge_edges = list(merge.get("incomingEdges", []))
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

    _, pipeline_exits = _boundary_nodes(pipeline_nodes) if pipeline_nodes else ([], [])
    pipeline_exit = next(
        (
            node_id
            for node_id in reversed(pipeline_exits)
            if "chunk_gated_delta_rule_fwd_h" in node_id
        ),
        pipeline_exits[-1] if pipeline_exits else None,
    )

    if merge_id is not None and pipeline_exit is not None:
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

    if group_node_attributes is not None and pipeline_nodes:
        attrs = {
            "label": pipeline_label,
            "operation": "kernel pipeline",
        }
        if merge is not None:
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
        if tree.class_name == class_name
        and subgraph_warrants_json_export(tree, basic_ops=basic_ops)
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
        if _component_uses_variant_attention_class(component, variant):
            resolved = _resolve_section_tree_by_class(
                spec,
                variant.attention_class,
                basic_ops=basic_ops,
            )
            if resolved is not None:
                return resolved
        if _component_uses_variant_ffn_class(component, variant):
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
        if tree.attr_name == attr_name
        and subgraph_warrants_json_export(tree, basic_ops=basic_ops)
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
    previous_exits: list[SourceRef],
    variant: LayerVariant | None = None,
    spine_namespace_prefix: str | None = None,
    group_node_attributes: dict[str, dict[str, str]] | None = None,
    shape_inferencer: ShapeInferencer | None = None,
    parent_class: str | None = None,
) -> list[SourceRef]:
    if not component_has_detail_section(component, spec):
        summary_namespace = spine_namespace_prefix or _flat_spine_namespace(
            component,
            namespace_prefix,
            variant=variant,
        )
        summary_label = (
            _group_node_label(spec, component)
            if component.role == "norm"
            else _display_label(component, spec)
        )
        single_op = _resolve_section_tree_for_component(
            spec,
            component,
            variant=variant,
            basic_ops=basic_ops,
        )
        if single_op is not None:
            _single_title, single_tree = single_op
            prepared_single = expand_block_tree_inplace(
                single_tree, basic_ops=basic_ops
            )
            if single_tree.children and not prepared_single.children:
                summary_label = prepared_single.label
        summary = _summary_node(
            id_prefix,
            summary_label,
            namespace=summary_namespace,
            component=component,
        )
        if previous_exits:
            summary["incomingEdges"] = [
                _source_edge(source, "0") for source in previous_exits
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
                _source_edge(source, "0") for source in previous_exits
            ]
        merged_nodes.append(summary)
        return [id_prefix]

    _title, block_tree = resolved
    prepared_tree = expand_block_tree_inplace(block_tree, basic_ops=basic_ops)
    if block_tree.children and not prepared_tree.children:
        # A wrapper around one real operation belongs directly on the spine. Keep
        # the component ID for wiring, but show only the operation (hc_head -> Mean).
        summary = _summary_node(
            id_prefix,
            prepared_tree.label,
            namespace=_flat_spine_namespace(
                component,
                namespace_prefix,
                variant=variant,
            ),
            component=component,
        )
        if previous_exits:
            summary["incomingEdges"] = [
                _source_edge(source, "0") for source in previous_exits
            ]
        merged_nodes.append(summary)
        return [id_prefix]
    block_tree = prepared_tree
    computation = build_computation_graph(
        block_tree,
        basic_ops=basic_ops,
        # Return branches may be consumed by later caller-side operations. Keeping
        # them also preserves real setup work such as mHC's Sinkhorn projection.
        strip_unused_return_branches=False,
    )
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
    pipeline_step = _kernel_pipeline_step(block_tree)
    if pipeline_step is not None:
        _integrate_kernel_pipeline_merge(
            section_nodes,
            namespace_prefix=namespace_prefix,
            pipeline_namespace=_join_namespace(
                namespace_prefix,
                _sanitize_namespace_segment(pipeline_step.label),
            ),
            pipeline_prefix=id_prefix,
            pipeline_label=pipeline_step.label,
            group_node_attributes=group_node_attributes,
            inject_skip=pipeline_inject_skip,
        )
    tile_ids = _block_tile_ids(computation, id_prefix=id_prefix)
    tile_replacements: dict[str, SourceRef] = {}
    nested_output_names: dict[str, str] = {}
    for nested_label, nested_block in collect_nested_diagrams(
        block_tree, basic_ops=basic_ops
    ):
        tile_id = tile_ids.get(id(nested_block))
        nested_prefix = tile_id or _merge_node_id(id_prefix, nested_block.attr_name)
        nested_name = _caller_output_name(
            spec, block_tree.class_name, nested_block.attr_name
        )
        if nested_name:
            nested_output_names[nested_prefix] = nested_name
        if any(node["id"].startswith(f"{nested_prefix}/") for node in section_nodes):
            continue
        nested_namespace = _join_namespace(
            namespace_prefix,
            (
                _sanitize_namespace_segment(nested_label)
                if tile_id
                else _nested_namespace_segment(nested_block, nested_label)
            ),
        )
        if nested_block.class_name == "KernelPipeline" and any(
            node.get("namespace", "").startswith(nested_namespace)
            and _is_tensor_port(node)
            for node in section_nodes
        ):
            continue
        nested_computation = build_computation_graph(
            nested_block,
            basic_ops=basic_ops,
            strip_unused_return_branches=False,
        )
        nested_nodes = _computation_nodes(
            nested_computation,
            id_prefix=nested_prefix,
            namespace_prefix=nested_namespace,
        )
        section_nodes.extend(nested_nodes)
        if tile_id is not None:
            nested_exits = _section_exits(
                nested_computation,
                nested_nodes,
                id_prefix=nested_prefix,
            )
            _replace_tile_with_group(
                section_nodes,
                nested_nodes,
                tile_id=tile_id,
                exit_ref=nested_exits[0] if nested_exits else None,
            )
            if nested_exits:
                tile_replacements[tile_id] = nested_exits[0]
        if shape_inferencer is not None:
            annotate_nodes_with_shapes(
                section_nodes,
                infer_block_tree_shapes(
                    shape_inferencer, nested_block, title=nested_label
                ),
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
    _inject_group_outputs(
        section_nodes,
        output_names=nested_output_names,
        resolve_name=lambda prefix: _caller_output_name(
            spec, block_tree.class_name, _tile_prefix_attr_name(prefix)
        ),
    )
    _connect_external_inputs(
        section_nodes,
        namespace_prefix=namespace_prefix,
        previous_exits=previous_exits,
    )
    transparent_exit = None
    _flatten_transparent_group_inputs(section_nodes)
    caller_name = _caller_output_name(spec, parent_class, component.attr_name)
    port_renames = (
        _rename_generic_output_ports(
            section_nodes,
            output_id=_merge_node_id(id_prefix, "@output"),
            name=caller_name,
        )
        if caller_name
        else {}
    )
    exits = (
        [transparent_exit]
        if transparent_exit is not None
        else _section_exits(
            computation,
            section_nodes,
            id_prefix=id_prefix,
            replacements=tile_replacements,
        )
    )
    if port_renames:
        exits = [
            (node_id, port_renames.get(port, port))
            for node_id, port in (_source_parts(exit_ref) for exit_ref in exits)
        ]

    merged_nodes.extend(section_nodes)
    apply_kernel_frame_labels(section_nodes, group_node_attributes)
    if group_node_attributes is not None:
        group_node_attributes[namespace_prefix] = {
            "label": (
                _group_node_label(spec, component)
                if component.role == "norm"
                else _display_label(component, spec)
            ),
            "operation": component.class_name or component.label or component.attr_name,
        }
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
    previous_exits: list[SourceRef],
    group_node_configs: list[dict[str, Any]],
    group_node_attributes: dict[str, dict[str, str]],
    shape_inferencer: ShapeInferencer | None = None,
) -> list[SourceRef]:
    chain_exits = list(previous_exits)
    residual_source = chain_exits[0] if chain_exits else None
    hc_outputs: dict[str, SourceRef] = {}

    def output_refs(section_prefix: str) -> dict[str, SourceRef]:
        output_prefix = _merge_node_id(section_prefix, "@output")
        refs: dict[str, SourceRef] = {}
        for output in merged_nodes:
            output_id = str(output.get("id", ""))
            if output_id != output_prefix and not output_id.startswith(
                f"{output_prefix}:"
            ):
                continue
            for metadata in output.get("outputsMetadata", []):
                port = str(metadata.get("id", ""))
                if port:
                    refs[port] = (output_id, port)
        return refs

    def append_residual_mix(site: str, branch_output: SourceRef) -> SourceRef:
        post = hc_outputs["post"]
        comb = hc_outputs["comb"]
        base = residual_source
        assert base is not None
        residual_namespace = _join_namespace(namespace_prefix, f"{site} residual")
        matmul_id = _merge_node_id(id_prefix, f"@residual:{site}:matmul")
        multiply_id = _merge_node_id(id_prefix, f"@residual:{site}:multiply")
        add_id = _merge_node_id(id_prefix, f"@residual:{site}:add")
        merged_nodes.extend(
            [
                {
                    "id": matmul_id,
                    "label": "MatMul",
                    "namespace": residual_namespace,
                    "incomingEdges": [
                        _source_edge(comb, "comb"),
                        _source_edge(base, "residual"),
                    ],
                },
                {
                    "id": multiply_id,
                    "label": "Multiply",
                    "namespace": residual_namespace,
                    "incomingEdges": [
                        _source_edge(post, "post"),
                        _source_edge(branch_output, "hidden_states"),
                    ],
                },
                {
                    "id": add_id,
                    "label": "Add",
                    "namespace": residual_namespace,
                    "incomingEdges": [
                        {
                            "sourceNodeId": multiply_id,
                            "sourceNodeOutputId": "0",
                            "targetNodeInputId": "post",
                        },
                        {
                            "sourceNodeId": matmul_id,
                            "sourceNodeOutputId": "0",
                            "targetNodeInputId": "comb",
                        },
                    ],
                },
            ]
        )
        return add_id

    for component in _ordered_decoder_components(spec):
        if component.attr_name in {"attn_hc", "ffn_hc"}:
            residual_source = chain_exits[0] if chain_exits else None
        section_prefix = _merge_node_id(id_prefix, component.attr_name)
        section_namespace = _section_namespace_for_component(
            spec,
            component,
            variant=variant,
            namespace_prefix=namespace_prefix,
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
            variant=variant,
            spine_namespace_prefix=namespace_prefix,
            group_node_attributes=group_node_attributes,
            shape_inferencer=shape_inferencer,
            parent_class=spec.decoder_class,
        )
        if component.attr_name in {"attn_hc", "ffn_hc"}:
            hc_outputs = output_refs(section_prefix)
        elif (
            component.attr_name == "self_attn"
            and {
                "post",
                "comb",
            }
            <= hc_outputs.keys()
        ):
            chain_exits = [append_residual_mix("attention", chain_exits[0])]
        elif (
            component.attr_name == "mlp"
            and {
                "post",
                "comb",
            }
            <= hc_outputs.keys()
        ):
            chain_exits = [append_residual_mix("ffn", chain_exits[0])]
    return chain_exits


def _exported_section_outputs(
    nodes: list[dict[str, Any]], section_prefix: str
) -> dict[str, SourceRef]:
    output_prefix = _merge_node_id(section_prefix, "@output")
    refs: dict[str, SourceRef] = {}
    for node in nodes:
        node_id = str(node.get("id", ""))
        if node_id != output_prefix and not node_id.startswith(f"{output_prefix}:"):
            continue
        for metadata in node.get("outputsMetadata", []):
            port = str(metadata.get("id", ""))
            if port:
                refs[port] = (node_id, port)
    return refs


def _append_source_decoder_layer(
    merged_nodes: list[dict[str, Any]],
    *,
    spec: ArchitectureSpec,
    id_prefix: str,
    namespace_prefix: str,
    basic_ops: BasicOpFilter,
    previous_exits: list[SourceRef],
    variant: LayerVariant | None,
    group_node_configs: list[dict[str, Any]],
    group_node_attributes: dict[str, dict[str, str]],
    shape_inferencer: ShapeInferencer | None,
) -> list[SourceRef]:
    """Export one decoder layer from its parsed forward dependency graph."""
    decoder = spec.class_registry.get(spec.decoder_class or "")
    if decoder is None or not decoder.forward_calls:
        return list(previous_exits)
    components = {component.attr_name: component for component in spec.block_components}
    refs: dict[str, SourceRef] = {}
    if previous_exits:
        refs["@method_input"] = previous_exits[0]

    for step in decoder.forward_calls:
        operation = decoder.forward_operations.get(step)
        if operation is not None:
            sources = [
                refs[predecessor]
                for predecessor in operation.predecessors
                if predecessor in refs
            ]
            if not sources:
                continue
            node_id = _merge_node_id(id_prefix, step)
            merged_nodes.append(
                {
                    "id": node_id,
                    "label": operation.label,
                    "namespace": namespace_prefix,
                    "attrs": [
                        {"key": "operation", "value": "source"},
                        *[
                            {"key": "detail", "value": str(detail)}
                            for detail in operation.details
                        ],
                    ],
                    "incomingEdges": [
                        _source_edge(source, str(index))
                        for index, source in enumerate(sources)
                    ],
                }
            )
            refs[step] = node_id
            continue

        component = components.get(step)
        if component is None:
            continue
        predecessors = decoder.forward_step_predecessors.get(step, ())
        inputs = [
            refs[predecessor] for predecessor in predecessors if predecessor in refs
        ]
        if not inputs:
            continue
        section_prefix = _merge_node_id(id_prefix, component.attr_name)
        section_namespace = _section_namespace_for_component(
            spec,
            component,
            variant=variant,
            namespace_prefix=namespace_prefix,
        )
        group = _group_config_for_role(section_namespace, component.role)
        if group:
            group_node_configs.append(group)
        exits = _append_section(
            merged_nodes,
            spec=spec,
            component=component,
            id_prefix=section_prefix,
            namespace_prefix=section_namespace,
            basic_ops=basic_ops,
            previous_exits=inputs,
            variant=variant,
            spine_namespace_prefix=namespace_prefix,
            group_node_attributes=group_node_attributes,
            shape_inferencer=shape_inferencer,
            parent_class=spec.decoder_class,
        )
        if exits:
            refs[step] = exits[0]
        outputs = _exported_section_outputs(merged_nodes, section_prefix)
        class_name = component.class_name
        if variant is not None and _component_uses_variant_attention_class(
            component, variant
        ):
            class_name = variant.attention_class or class_name
        elif variant is not None and _component_uses_variant_ffn_class(
            component, variant
        ):
            class_name = variant.ffn_class or class_name
        callee = spec.class_registry.get(class_name)
        if callee is not None:
            for slot, producer in callee.forward_return_slots.items():
                if slot in outputs:
                    refs[producer] = outputs[slot]

    if decoder.primary_return_slot:
        producer = decoder.forward_return_slots.get(decoder.primary_return_slot)
        if producer in refs:
            return [refs[producer]]
    return [refs[step] for step in reversed(decoder.forward_calls) if step in refs][:1]


def _append_decoder_layers(
    merged_nodes: list[dict[str, Any]],
    *,
    spec: ArchitectureSpec,
    decoder_namespace: str,
    basic_ops: BasicOpFilter,
    previous_exits: list[SourceRef],
    group_node_configs: list[dict[str, Any]],
    group_node_attributes: dict[str, dict[str, str]],
    shape_inferencer: ShapeInferencer | None = None,
) -> list[SourceRef]:
    decoder = spec.class_registry.get(spec.decoder_class or "")
    if decoder is not None:
        expand_class_forward_dataflow(decoder, spec.class_registry)
    if spec.layer_variants:
        variant_exits: list[SourceRef] = []
        for variant in spec.layer_variants:
            first_node_index = len(merged_nodes)
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
            decoder = spec.class_registry.get(spec.decoder_class or "")
            if decoder is not None and (
                decoder.forward_operations
                or decoder.forward_step_predecessors
            ):
                exits = _append_source_decoder_layer(
                    merged_nodes,
                    spec=spec,
                    id_prefix=variant_prefix,
                    namespace_prefix=variant_namespace,
                    basic_ops=basic_ops,
                    previous_exits=previous_exits,
                    variant=variant,
                    group_node_configs=group_node_configs,
                    group_node_attributes=group_node_attributes,
                    shape_inferencer=shape_inferencer,
                )
            else:
                exits = _append_variant_layer(
                    merged_nodes,
                    spec=spec,
                    variant=variant,
                    id_prefix=variant_prefix,
                    namespace_prefix=variant_namespace,
                    basic_ops=basic_ops,
                    previous_exits=previous_exits,
                    group_node_configs=group_node_configs,
                    group_node_attributes=group_node_attributes,
                    shape_inferencer=shape_inferencer,
                )
            variant_exits.extend(
                _wrap_actual_group_boundary(
                    merged_nodes,
                    namespace=variant_namespace,
                    id_prefix=variant_prefix,
                    inputs=previous_exits,
                    outputs=exits,
                    first_node_index=first_node_index,
                )
            )
        result = variant_exits or list(previous_exits)
        if shape_inferencer is not None:
            fill_missing_node_shapes(
                merged_nodes, context=shape_inferencer.context
            )
        return result

    decoder_inputs = list(previous_exits)
    first_node_index = len(merged_nodes)
    for component in _ordered_decoder_components(spec):
        section_namespace = _section_namespace_for_component(
            spec,
            component,
            variant=None,
            namespace_prefix=decoder_namespace,
        )
        group = _group_config_for_role(section_namespace, component.role)
        if group:
            group_node_configs.append(group)
    chain_exits = _append_source_decoder_layer(
        merged_nodes,
        spec=spec,
        id_prefix="decoder",
        namespace_prefix=decoder_namespace,
        basic_ops=basic_ops,
        previous_exits=previous_exits,
        variant=None,
        group_node_configs=group_node_configs,
        group_node_attributes=group_node_attributes,
        shape_inferencer=shape_inferencer,
    )
    result = _wrap_actual_group_boundary(
        merged_nodes,
        namespace=decoder_namespace,
        id_prefix="decoder",
        inputs=decoder_inputs,
        outputs=chain_exits,
        first_node_index=first_node_index,
    )
    if shape_inferencer is not None:
        fill_missing_node_shapes(merged_nodes, context=shape_inferencer.context)
    return result


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
    stack_module_sources: dict[str, SourceRef] = {}
    group_node_configs: list[dict[str, Any]] = []
    group_node_attributes: dict[str, dict[str, str]] = {}
    stack_cls = spec.class_registry.get(spec.stack_model_class or "")
    if stack_cls is None:
        stack_cls = _pick_stack_model_class(spec.class_registry, None)

    for component in _stack_pre_components(spec):
        if (
            component.role == "positional"
            and stack_cls is not None
            and component.attr_name not in stack_cls.forward_calls
        ):
            continue
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
            parent_class=spec.stack_model_class,
        )
        if previous_exits:
            stack_module_sources[component.attr_name] = previous_exits[0]

    source_entry = _append_stack_entry_dataflow(
        nodes,
        spec=spec,
        module_sources=stack_module_sources,
        shape_inferencer=shape_inferencer,
    )
    if source_entry is not None:
        previous_exits = source_entry

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
            parent_class=spec.stack_model_class,
        )

    root_sources = [_source_parts(source) for source in previous_exits]
    root_ports = [
        (
            "result" if len(root_sources) == 1 else f"result_{index + 1}",
            source_id,
            source_port,
        )
        for index, (source_id, source_port) in enumerate(root_sources)
    ]
    if root_ports:
        nodes.append(
            _make_group_output_node(
                output_id="@output",
                namespace="",
                ports=root_ports,
            )
        )
    _prune_unconsumed_outputs(nodes)
    _label_boundary_outputs_by_port(nodes)
    _mirror_boundary_outputs(nodes)

    if shape_inferencer is not None:
        fill_missing_node_shapes(nodes, context=shape_inferencer.context)

    model_attrs: dict[str, str] = {
        "title": spec.name,
        "model_type": spec.model_type,
        "decoder": spec.decoder_type,
        "layers": str(spec.num_hidden_layers or "?"),
    }
    if spec.forward_sequence:
        model_attrs["forward"] = format_forward_sequence(spec)
    if spec.decoder_class:
        model_attrs["decoder_class"] = spec.decoder_class
    if spec.layer_mix:
        model_attrs["layer_mix"] = spec.layer_mix
    model_attrs.update(build_fact_sheet_group_attributes(spec))

    finalize_graph_node_styles(nodes)

    graph_attributes: dict[str, dict[str, str]] = {
        "": model_attrs,
        decoder_namespace: {
            "repeat": decoder_namespace,
            "forward": format_forward_sequence(spec),
            **({"layer_mix": spec.layer_mix} if spec.layer_mix else {}),
        },
        **group_node_attributes,
    }
    if shape_inferencer is not None:
        for namespace, boundary in group_boundary_shapes(nodes).items():
            graph_attributes.setdefault(namespace, {}).update(boundary)

    return {
        "id": graph_id,
        "nodes": nodes,
        "groupNodeAttributes": graph_attributes,
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
