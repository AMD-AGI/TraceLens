###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build one merged Model Explorer graph with in-place namespace expansion."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.ast_analyze import (
    _pick_stack_model_class,
    base_submodule_attr,
    expand_class_forward_dataflow,
    FORWARD_METHOD_INPUT,
    stack_entry_dataflow,
)
from TraceLens.ModelUtils.block_tree import (
    BlockNode,
    collect_nested_diagrams,
    expand_block_tree_inplace,
    is_transparent_inline_expansion,
    subgraph_warrants_json_export,
)
from TraceLens.ModelUtils.blocks import BlockComponent, LayerVariant
from TraceLens.ModelUtils.computation_graph import ComputationGraph, build_computation_graph
from TraceLens.ModelUtils.extract import (
    ArchitectureSpec,
    architecture_section_trees,
    image_placeholder_token_id,
    vision_tower_component,
)
from TraceLens.ModelUtils.shape_inference import (
    _merge_flatten_dim,
    _permute_shape,
    ShapeInferencer,
    Symbol,
    TensorSpec,
)

from TraceLens.Visualizer.model_explorer_export.adapter import (
    _incoming_edges,
    _node_attrs,
    _node_namespaces,
    _node_style,
    _output_port_metadata,
    _sanitize_namespace_segment,
)
from TraceLens.Visualizer.model_explorer_export.fact_sheet import build_fact_sheet_group_attributes
from TraceLens.Visualizer.model_explorer_export.type_check import (
    integrity_check_graph_nodes,
    type_check_graph_nodes,
)
from TraceLens.Visualizer.model_explorer_export.shapes import (
    annotate_nodes_with_shapes,
    apply_shape_attrs,
    fill_missing_node_shapes,
    group_boundary_shapes,
    infer_block_tree_shapes,
    node_output_spec,
    format_shape_dims,
    parse_shape_dims,
    SHAPE_SEPARATOR,
)
from TraceLens.Visualizer.model_explorer_export.labels import (
    apply_kernel_frame_labels,
    skip_merged_tensor_port_parent,
    tensor_port_input_label,
)
from TraceLens.Visualizer.model_explorer_export.overview import (
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
from TraceLens.Visualizer.model_explorer_export.styles import (
    ROLE_COLORS,
    _GPU_KERNEL_BORDER,
    build_group_node_configs,
    detail_tile_style,
    ensure_readable_text,
    finalize_graph_node_styles,
    input_port_style,
    is_layout_only_label,
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
        neg_index: int | None = None
        parts = [part.strip() for part in shape_text.split(",")]
        for index, part in enumerate(parts):
            if part == "-1":
                # Expand keeps the source dim at a ``-1`` slot; Reshape/View flatten
                # it, so defer to a merged-product computation below.
                if label == "Expand" and index < len(source.shape):
                    resolved.append(source.shape[index])
                elif label != "Expand" and neg_index is None:
                    neg_index = len(resolved)
                    resolved.append(part)
                else:
                    resolved.append(part)
            elif re.fullmatch(r"-?\d+", part):
                resolved.append(int(part))
            elif ".config." in part:
                value = _config_leaf(spec.raw_config, part.rsplit(".", 1)[-1])
                resolved.append(value if value is not None else part)
            else:
                resolved.append(part)
        if neg_index is not None:
            explicit = [d for i, d in enumerate(resolved) if i != neg_index]
            merged = _merge_flatten_dim(tuple(source.shape), explicit)
            if merged is not None:
                resolved[neg_index] = merged
            elif neg_index < len(source.shape):
                # Element conservation failed; fall back to the same-index source
                # dim rather than leaking a literal ``-1`` into the display shape.
                resolved[neg_index] = source.shape[neg_index]
        return TensorSpec(tuple(resolved), source.dtype)
    if label in {"Permute", "Transpose"}:
        permuted = _permute_shape(source.shape, _operation_detail(operation, "dims"))
        if permuted is not None:
            return TensorSpec(permuted, source.dtype)
        return source
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
                _source_edge(source, str(index)) for index, source in enumerate(sources)
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
            operation_spec = _data_movement_shape(operation, source_spec, spec=spec)
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


def _set_node_attr(node: dict[str, Any], key: str, value: str) -> None:
    """Set (replace or append) a single ``{key, value}`` attr on a node."""
    attrs = node.setdefault("attrs", [])
    for attr in attrs:
        if attr.get("key") == key:
            attr["value"] = value
            return
    attrs.append({"key": key, "value": value})


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
    node_by_id: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """Label pipeline tensor ports directly; module boundaries get input blocks."""
    if not entry_ports:
        return False
    node_by_id = node_by_id or {}
    tensor_ports_only = all(
        _labeled_tensor_port_label(target)
        or _labeled_tensor_port_label(node_by_id.get(edge["sourceNodeId"]))
        for _label, edge, target in entry_ports
    )
    if not tensor_ports_only:
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


def _source_output_port_label(node: dict[str, Any] | None, port: str) -> str | None:
    """The ``port_label`` recorded for a producer's ``port`` output slot, if any."""
    if node is None:
        return None
    for item in node.get("outputsMetadata", []) or []:
        if str(item.get("id", "")) != str(port):
            continue
        for attr in item.get("attrs", []):
            if attr.get("key") == "port_label" and attr.get("value"):
                return str(attr["value"])
    return None


def _source_slice_label(
    source_id: str,
    source_port: str,
    node_by_id: dict[str, dict[str, Any]],
) -> str | None:
    """Name a boundary after the producer *slice* it reads, when that slice is one
    of several a multi-output producer emits.

    A consumer reading one named slice of a multi-output producer -- ``key_states``
    from a qkv ``unbind``, ``cos``/``sin`` from a rotary block that returns a tuple --
    should show that slice's own name on its boundary tile, not the generic forward
    parameter (``hidden_states`` / ``position_embeddings``). Fires only when the
    producer is genuinely multi-output, so an ordinary single-tensor input keeps its
    parameter name (bounded blast radius).
    """
    source = node_by_id.get(source_id)
    if source is None:
        return None
    port = str(source_port)
    # (a) A multi-output op (unbind/split/chunk) tags its slices with output_names.
    names_attr = _node_attr(source, "output_names")
    if names_attr:
        names = [name for name in names_attr.split(",") if name]
        if len(names) > 1:
            label = _source_output_port_label(source, port)
            if label:
                return label
            if port.isdigit() and int(port) < len(names):
                return names[int(port)]
    # (b) A named slice boundary tile: an @output / @input (or their mirrors)
    #     that is one of >= 2 sibling boundary tiles in its namespace, i.e. a
    #     specific named slice rather than a lone generic boundary. A nested block
    #     reading the split ``cos``/``sin`` tiles a parent already carved out keeps
    #     the slice name instead of collapsing back onto the tuple parameter. A
    #     single-slot boundary (one @output feeding an input) is left untouched.
    syn = _node_attr(source, "synthetic")
    families = ({"@output", "@output_mirror"}, {"@input", "@input_mirror"})
    family = next((fam for fam in families if syn in fam), None)
    if family is not None:
        namespace = source.get("namespace", "")
        slots = sum(
            1
            for other in node_by_id.values()
            if _node_attr(other, "synthetic") in family
            and other.get("namespace", "") == namespace
        )
        if slots >= 2:
            label = _source_output_port_label(source, port)
            if label:
                return label
            source_label = source.get("label")
            if isinstance(source_label, str) and source_label:
                return source_label
    return None


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


# Both image-side model inputs — the pixel/patch tensor and the image-placeholder
# mask — render at top level (no separate box). ``_order_model_inputs`` keeps them
# adjacent in sort order, next to each other rather than the mask floating beside
# its distant ``masked_scatter`` consumer.
_IMAGE_MASK_ID = "@image_mask"


def _ensure_image_mask_node(
    nodes: list[dict[str, Any]],
    *,
    namespace: str,
    token_id: int | None = None,
) -> str:
    """Append the ``@image_mask`` boundary if absent; return its id.

    The mask is the ``input_ids == image_token_id`` selector the vision-language
    ``masked_scatter`` uses. It may be pre-created next to the image-patch input
    (when the model exposes an image-token key); this keeps the combine's
    reference a no-op in that case, and still synthesizes the boundary for VLMs
    with no such key.
    """
    for node in nodes:
        if node.get("id") == _IMAGE_MASK_ID:
            return _IMAGE_MASK_ID
    node = _make_group_input_node(
        input_id=_IMAGE_MASK_ID,
        label="image_mask",
        namespace=namespace,
        port_label="image_mask",
    )
    if token_id is not None:
        node["attrs"].append(
            {"key": "detail", "value": f"input_ids == image_token_id ({token_id})"}
        )
    # ``input_ids == image_token_id`` is a boolean ``[B, S]`` selector. Stamp it
    # explicitly so the boundary shows the true mask shape rather than inheriting
    # the combine's ``[B, S, hidden]`` embedding shape by back-fill.
    apply_shape_attrs(node, TensorSpec((Symbol.BATCH.value, Symbol.SEQ.value), "bool"))
    nodes.append(node)
    return _IMAGE_MASK_ID


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

    # Float each loop-carried-in node to the top of its loop group and each
    # loop-carried-out to the bottom, so ME's dagre layout renders the carried
    # state entering above the body and leaving below it. The body frequently
    # lives in *child* namespaces (LC-in in ``visual/Block``, body ops in
    # ``visual/Block/norm1``), so an exact-namespace bucket sort can't reach it:
    # compare by namespace *subtree* (prefix) instead. Only the boundary nodes
    # move; every other node keeps its relative order, so cross-namespace
    # dataflow edges are untouched (edges bind ids, not positions).
    def _in_subtree(root_ns: str, node_ns: str) -> bool:
        return node_ns == root_ns or node_ns.startswith(root_ns + "/")

    lc_in_nodes = [n for n in nodes if "@loop_carried_in:" in n.get("id", "")]
    lc_out_nodes = [n for n in nodes if "@loop_carried_out:" in n.get("id", "")]
    if lc_in_nodes or lc_out_nodes:
        movers = {id(n) for n in lc_in_nodes} | {id(n) for n in lc_out_nodes}
        result = [n for n in nodes if id(n) not in movers]
        for lc in lc_in_nodes:
            ns = lc.get("namespace", "")
            insert_at = next(
                (
                    index
                    for index, node in enumerate(result)
                    if _in_subtree(ns, node.get("namespace", ""))
                ),
                len(result),
            )
            result.insert(insert_at, lc)
        for lc in lc_out_nodes:
            ns = lc.get("namespace", "")
            last = None
            for index, node in enumerate(result):
                if _in_subtree(ns, node.get("namespace", "")):
                    last = index
            result.insert(last + 1 if last is not None else len(result), lc)
        nodes = result

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
    incoming = list(tile.get("incomingEdges", []))
    entries = [node for node in nested_nodes if _is_synthetic_input(node)]
    if not entries:
        entry_ids, _exits = _boundary_nodes(nested_nodes)
        entries = [node for node in nested_nodes if node["id"] in set(entry_ids)]
    unclaimed = list(incoming)
    empty_entries = [entry for entry in entries if not entry.get("incomingEdges")]
    for entry in empty_entries:
        label = _node_attr(entry, "port_label") or str(entry.get("label", ""))
        matches = [
            edge
            for edge in unclaimed
            if (_edge_port_label(edge) or str(edge.get("targetNodeInputId", "")))
            == label
        ]
        if matches:
            entry["incomingEdges"] = [dict(edge) for edge in matches]
            unclaimed = [edge for edge in unclaimed if edge not in matches]
    unresolved = [entry for entry in empty_entries if not entry.get("incomingEdges")]
    if len(unresolved) == len(unclaimed):
        for entry, edge in zip(unresolved, unclaimed):
            entry["incomingEdges"] = [dict(edge)]
        unclaimed = []
    elif len(unresolved) == 1 and unclaimed:
        unresolved[0]["incomingEdges"] = [dict(edge) for edge in unclaimed]

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
    """Inline-expanded frames keep their ops' direct edges, no synthetic @input.

    An activation frame (``SituAndMul``) inherits the parent MLP's single input and
    must not receive its own group boundary.

    Traced free-function frames (a rope helper, ``get_vision_position_ids``) are
    NOT skipped: the owner rule is that a free-function call renders like any other
    module call, with real ``@input``/``@output`` boundaries. A tuple fan-out on the
    boundary (cos/sin ``position_embeddings[0]``/``[1]``) is preserved by the same
    per-ordinal machinery a multi-return module uses -- ``_group_entry_buckets``
    keys a bucket by the producer node, so both ordinal ports land on one boundary
    tile that re-exposes them.
    """
    if "/" not in namespace:
        return False
    segment = namespace.rsplit("/", 1)[-1]
    if segment not in _INLINE_FRAME_NAMESPACE_SUFFIXES:
        return False
    parent = namespace.rsplit("/", 1)[0]
    return any(
        _is_synthetic_input(node) and node.get("namespace", "") == parent
        for node in section_nodes
    )


def _group_entry_buckets(
    entry_nodes: list[dict[str, Any]],
    internal_ids: set[str],
) -> list[tuple[frozenset[tuple[str, str]], list[dict[str, Any]]]]:
    """Split a group's entry nodes by the outside tensor each one reads.

    ``forward(hidden_states, gate)`` enters the block at two unrelated steps fed by
    two unrelated producers. Collapsing them onto one boundary tile would claim the
    normalization reads the gate, so each distinct producer keeps its own entry.
    """
    # Key each bucket by the producer NODE, not by (node, port): a tuple producer
    # feeding two ports (``position_embeddings`` -> cos at port 0, sin at port 1)
    # is ONE logical input that fans out, so both ports land on one boundary tile
    # that re-exposes them. Distinct producers still key distinct buckets, so
    # ``forward(hidden_states, gate)`` keeps one boundary per unrelated producer.
    buckets: dict[str, tuple[set[tuple[str, str]], list[dict[str, Any]]]] = {}
    order: list[str] = []
    for node in entry_nodes:
        sources = [
            (edge["sourceNodeId"], edge.get("sourceNodeOutputId", "0"))
            for edge in node.get("incomingEdges", [])
            if edge["sourceNodeId"] not in internal_ids
        ]
        keys = {source_id for source_id, _ in sources} or {""}
        for key in keys:
            if key not in buckets:
                buckets[key] = (set(), [])
                order.append(key)
            source_set, members = buckets[key]
            source_set.update(source for source in sources if source[0] == key)
            members.append(node)
    return [(frozenset(buckets[key][0]), buckets[key][1]) for key in order]


def _entry_bucket_label(
    sources: frozenset[tuple[str, str]],
    entries: list[dict[str, Any]],
    internal_ids: set[str],
    node_by_id: dict[str, dict[str, Any]],
    resolve_slot_label: Any = None,
) -> str | None:
    """Name a boundary input after the tensor arriving on it, when it is known."""
    for node in entries:
        for edge in node.get("incomingEdges", []):
            source = (edge["sourceNodeId"], edge.get("sourceNodeOutputId", "0"))
            if source not in sources:
                continue
            # A boundary that reads one named return slot of a multi-return child
            # (``cos``/``sin`` of the rotary block, arriving as the tuple parameter
            # ``position_embeddings``) is named after that slot, so the two slices
            # stay distinct tiles instead of collapsing onto the tuple's name.
            slot_label = (
                resolve_slot_label(
                    edge["sourceNodeId"], edge.get("sourceNodeOutputId", "0")
                )
                if resolve_slot_label is not None
                else None
            )
            label = (
                slot_label
                or _source_slice_label(
                    edge["sourceNodeId"],
                    edge.get("sourceNodeOutputId", "0"),
                    node_by_id,
                )
                or _edge_port_label(edge)
                or _labeled_tensor_port_label(node_by_id.get(edge["sourceNodeId"]))
            )
            if label:
                return label
    for node in entries:
        # The consuming step records which forward parameter it reads, which names
        # the boundary more reliably than anything recoverable from the edge.
        parameter = _node_attr(node, "boundary_input")
        external_count = sum(
            edge["sourceNodeId"] not in internal_ids
            for edge in node.get("incomingEdges", [])
        )
        if parameter and external_count == 1:
            return parameter
    # A kernel input port (``@kernel_port_in``) names the exact tensor it
    # carries into the kernel (``cu_seqlens``). When such a port is the group's
    # entry for an outside producer, that label names the boundary far better
    # than a generic ``hidden_states_2`` fallback.
    for node in entries:
        if _node_attr(node, "synthetic") == "@kernel_port_in" and node.get("label"):
            return node["label"]
    return None


def _next_input_label(label: str, used: set[str]) -> str:
    index = 2
    while f"{label}_{index}" in used:
        index += 1
    return f"{label}_{index}"


def _inject_group_inputs(
    section_nodes: list[dict[str, Any]],
    *,
    skip_namespaces: frozenset[str] = frozenset(),
    resolve_slot_label: Any = None,
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
        scoped_nodes = [
            node
            for node in section_nodes
            if node in group_nodes
            or (
                node["id"] in internal_ids
                and _node_attr(node, "boundary_input") is not None
            )
        ]
        entry_ports = _collect_group_entry_ports(scoped_nodes, internal_ids, node_by_id)
        entry_nodes: list[dict[str, Any]] = []
        outside_sources: set[tuple[str, str]] = set()

        for node in scoped_nodes:
            # A ``constant`` leaf (a materialized buffer/param read such as
            # ``self.attention_scaling``) is an internal value source the render
            # filter drops, not an activation the group reads from outside. It has
            # no incoming edge, so without this guard it would be mistaken for an
            # entry point and spawn a spurious ``@input`` boundary.
            if _node_attr(node, "constant") == "true":
                continue
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

        if _apply_labeled_external_entry_ports(entry_ports, internal_ids, node_by_id):
            continue

        group_ids = [node["id"] for node in group_nodes]
        prefix = _group_input_prefix(group_ids) or group_ids[0].rsplit("/", 1)[0]

        buckets = _group_entry_buckets(entry_nodes, internal_ids)
        bucket_labels = [
            _entry_bucket_label(
                sources, entries, internal_ids, node_by_id, resolve_slot_label
            )
            for sources, entries in buckets
        ]
        # Entry steps that read the same forward parameter share one tile even if
        # their edges were introduced separately. Distinct parameter names remain
        # distinct boundaries (for example hidden_states, top_k_index, and
        # top_k_weights on an Experts block).
        merged_buckets: dict[
            tuple[str, object],
            tuple[set[tuple[str, str]], list[dict[str, Any]], str | None],
        ] = {}
        merged_order: list[tuple[str, object]] = []
        labels_by_source: dict[tuple[str, str], set[str]] = {}
        for (sources, _entries), label in zip(buckets, bucket_labels):
            if label:
                for source in sources:
                    labels_by_source.setdefault(source, set()).add(label)
        for (sources, entries), label in zip(buckets, bucket_labels):
            if label is None:
                inferred = {
                    candidate
                    for source in sources
                    for candidate in labels_by_source.get(source, set())
                }
                if len(inferred) == 1:
                    label = inferred.pop()
            key: tuple[str, object] = (
                ("label", label) if label else ("sources", sources)
            )
            if key not in merged_buckets:
                merged_buckets[key] = (set(), [], label)
                merged_order.append(key)
            merged_sources, merged_entries, _ = merged_buckets[key]
            merged_sources.update(sources)
            merged_entries.extend(entries)
        buckets = [
            (frozenset(merged_buckets[key][0]), merged_buckets[key][1])
            for key in merged_order
        ]
        bucket_labels = [merged_buckets[key][2] for key in merged_order]
        single = len(buckets) == 1

        used_labels: set[str] = set()
        entry_inputs: dict[str, list[str]] = {}
        # A boundary that gathers one producer at several output ports (the cos/sin
        # tuple ``position_embeddings[0]``/``[1]`` both flow from one rope concat)
        # re-exposes those ports so each consumer keeps reading its own slot. Map
        # every gathered ``(source, port)`` to the boundary slot it lands on.
        tile_source_slot: dict[str, dict[tuple[str, str], int]] = {}
        for (sources, entries), label in zip(buckets, bucket_labels):
            if not label:
                label = _infer_group_input_label(group_nodes, namespace)
            while label in used_labels:
                label = _next_input_label(label, used_labels)
            used_labels.add(label)

            input_id = _group_input_id(prefix, None if single else label)
            if input_id in node_by_id:
                continue

            input_node = _make_group_input_node(
                input_id=input_id,
                label=label,
                namespace=namespace,
                port_label=None if single else label,
            )
            bucket_sources = sorted(sources)
            if bucket_sources:
                input_node["incomingEdges"] = [
                    {
                        "sourceNodeId": source_id,
                        "sourceNodeOutputId": source_port,
                        "targetNodeInputId": str(index),
                    }
                    for index, (source_id, source_port) in enumerate(bucket_sources)
                ]
            tile_source_slot[input_id] = {
                source: index for index, source in enumerate(bucket_sources)
            }

            for entry in entries:
                entry_inputs.setdefault(entry["id"], []).append(input_id)

            section_nodes.append(input_node)
            node_by_id[input_id] = input_node

        for entry_id, input_ids in entry_inputs.items():
            entry = node_by_id[entry_id]
            original = list(entry.get("incomingEdges", []))
            internal = [
                edge for edge in original if edge["sourceNodeId"] in internal_ids
            ]
            external = [
                edge for edge in original if edge["sourceNodeId"] not in internal_ids
            ]
            # Redirect each external edge onto the boundary tile its own source
            # landed on, preserving that edge's ORIGINAL targetNodeInputId instead
            # of renumbering sequentially from ``len(internal)``. An order-sensitive
            # op (``cat``/``stack``) can read its external operand before an
            # internal one (operand 0 external, operand 1 internal); renumbering
            # then collides with an internal edge that already holds the target id
            # the formula computes, silently dropping the external operand.
            redocked: list[dict[str, Any]] = []
            for edge in external:
                source = (
                    edge["sourceNodeId"],
                    edge.get("sourceNodeOutputId", "0"),
                )
                input_id = next(
                    (
                        candidate
                        for candidate in input_ids
                        if source in tile_source_slot.get(candidate, {})
                    ),
                    input_ids[0] if input_ids else None,
                )
                if input_id is None:
                    redocked.append(edge)
                    continue
                slot_map = tile_source_slot.get(input_id, {})
                # Preserve the exact tuple slot this consumer read from the shared
                # producer; a single-source boundary keeps port "0" (unchanged).
                port = str(slot_map.get(source, 0))
                rewired = dict(edge)
                rewired["sourceNodeId"] = input_id
                rewired["sourceNodeOutputId"] = port
                redocked.append(rewired)
            entry["incomingEdges"] = internal + redocked


def _tile_prefix_attr_name(prefix: str) -> str:
    """Recover the submodule attribute encoded in a tile id.

    Tile ids carry their slot and producer (``seq:7:forget_gate``,
    ``sidefeed:11:o_norm:@op_l35``); only the attribute names the tensor.
    """
    segment = prefix.rsplit("/", 1)[-1]
    parts = segment.split(":")
    if len(parts) >= 3 and parts[0] in {"seq", "sidefeed", "branch", "sideproducer"}:
        return parts[2]
    return segment


def _inject_group_outputs(
    section_nodes: list[dict[str, Any]],
    *,
    output_names: dict[str, str] | None = None,
    resolve_name: Any = None,
    resolve_slot_names: Any = None,
) -> None:
    """Give every visible Input namespace a matching Output boundary."""
    input_namespaces = {
        node.get("namespace", "") for node in section_nodes if _is_synthetic_input(node)
    }
    for namespace in sorted(
        (item for item in input_namespaces if item),
        # Deepest namespaces first; the namespace string is a secondary key so
        # sibling namespaces at equal depth emit in a deterministic (not
        # hash-seed dependent) order, keeping exports byte-reproducible.
        key=lambda item: (-item.count("/"), item),
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
        # A nested boundary (e.g. a loop body whose @input nodes live at an
        # ancestor's id-prefix) can derive an output id already owned by an
        # ancestor section's real boundary. Two nodes sharing an id are fused by
        # the viewer, merging their edge sets into a false cycle. Disambiguate by
        # namespace when the id is already taken by a node in another namespace.
        if any(
            node["id"] == output_id and node.get("namespace", "") != namespace
            for node in section_nodes
        ):
            output_id = f"{namespace}/@output"
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

        # Try to resolve named return slots for multi-output modules.
        slot_names: dict[str, list[str]] | None = None
        if len(sources) > 1 and resolve_slot_names is not None:
            slot_names = resolve_slot_names(prefix)

        if slot_names and len(sources) > 1:
            ports = []
            # When several return slots trace to the same producer op, the
            # source output ordinal selects which slot each edge carries so the
            # boundary emits distinct per-slot outputs (cos=0, sin=1) instead of
            # collapsing both onto the last-written slot name.
            producer_cursor: dict[str, int] = {}
            for source, source_port in sources:
                slots = next(
                    (
                        slot_list
                        for attr, slot_list in slot_names.items()
                        if attr in source
                    ),
                    None,
                )
                # Tolerate a scalar mapping (attr → single slot) as well as the
                # ordinal-aware list form so callers can pass either.
                if isinstance(slots, str):
                    slots = [slots]
                slot = None
                if slots:
                    if len(slots) == 1:
                        slot = slots[0]
                    else:
                        idx = (
                            int(source_port)
                            if str(source_port).isdigit()
                            else None
                        )
                        if idx is None or idx >= len(slots):
                            idx = producer_cursor.get(source, 0)
                        slot = slots[idx] if idx < len(slots) else None
                        producer_cursor[source] = producer_cursor.get(source, 0) + 1
                port_label = slot or f"{name}_{len(ports) + 1}"
                ports.append((port_label, source, source_port))
        else:
            ports = [
                (
                    name if len(sources) == 1 else f"{name}_{index + 1}",
                    source,
                    source_port,
                )
                for index, (source, source_port) in enumerate(sources)
            ]
        if len(ports) <= 1:
            # Single output → one Output node.
            port_by_source = {
                (source, source_port): port
                for port, source, source_port in ports
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
        else:
            # Multiple outputs → one Output node per source,
            # matching _wrap_actual_group_boundary for consistency.
            source_to_port_id: dict[tuple[str, str], str] = {}
            for port, source, source_port in ports:
                per_port_id = _port_output_id(output_id, port, len(ports))
                source_to_port_id[(source, source_port)] = per_port_id
                section_nodes.append(
                    _make_group_output_node(
                        output_id=per_port_id,
                        namespace=namespace,
                        ports=[(port, source, source_port)],
                    )
                )

            for _target, edge in outgoing:
                source = (
                    edge["sourceNodeId"],
                    edge.get("sourceNodeOutputId", "0"),
                )
                per_port_id = source_to_port_id[source]
                port_label = next(
                    p for p, s, sp in ports if (s, sp) == source
                )
                edge["sourceNodeId"] = per_port_id
                edge["sourceNodeOutputId"] = port_label


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
    # A block whose sole input reads one named slice of a multi-output producer
    # (q_norm reads ``query_states``, k_norm reads ``key_states`` of the qkv unbind)
    # names its boundary after that slice, so two sibling blocks fed by different
    # slices no longer look like they read the same tensor.
    input_label = "hidden_states"
    if len(inputs) == 1:
        source_id, source_port = _source_parts(inputs[0])
        slice_label = _source_slice_label(
            source_id, source_port, {node["id"]: node for node in nodes}
        )
        if slice_label:
            input_label = slice_label
    input_node = _make_group_input_node(
        input_id=input_id,
        label=input_label,
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

    A block returning one tensor needs no mirror: its single Output already carries
    that tensor's name, and repeating it outside only doubles the tile.
    """
    outputs_per_namespace: dict[str, int] = {}
    for node in nodes:
        if _is_synthetic_output(node):
            namespace = str(node.get("namespace", ""))
            outputs_per_namespace[namespace] = (
                outputs_per_namespace.get(namespace, 0) + 1
            )

    mirrors: list[dict[str, Any]] = []
    for node in list(nodes):
        if not _is_synthetic_output(node):
            continue
        namespace = str(node.get("namespace", ""))
        if not namespace:
            continue
        if outputs_per_namespace.get(namespace, 0) < 2:
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


def _mirror_boundary_inputs(nodes: list[dict[str, Any]]) -> None:
    """Pair multi-input submodule ports with same-named nodes in the caller.

    Input blocks live inside the callee namespace. Mirroring each port into the
    parent makes the call site show every operand, matching the two-sided boundary
    already used for modules with multiple outputs.
    """
    inputs_per_namespace: dict[str, int] = {}
    for node in nodes:
        if _is_synthetic_input(node):
            namespace = str(node.get("namespace", ""))
            inputs_per_namespace[namespace] = inputs_per_namespace.get(namespace, 0) + 1

    mirrors: list[dict[str, Any]] = []
    for node in list(nodes):
        if not _is_synthetic_input(node):
            continue
        namespace = str(node.get("namespace", ""))
        if not namespace or inputs_per_namespace.get(namespace, 0) < 2:
            continue
        internal = _namespace_internal_ids(nodes, namespace)
        external = [
            edge
            for edge in node.get("incomingEdges", [])
            if edge.get("sourceNodeId") not in internal
        ]
        if not external:
            continue
        input_id = str(node["id"])
        port = _node_attr(node, "port_label") or str(node.get("label", "input"))
        mirror_id = f"{input_id.replace('/@input', '/@input_mirror', 1)}^{port}"
        parent_namespace = namespace.rsplit("/", 1)[0] if "/" in namespace else ""
        mirrors.append(
            {
                "id": mirror_id,
                "label": port,
                "namespace": parent_namespace,
                "attrs": [
                    {"key": "synthetic", "value": "@input_mirror"},
                    {"key": "port_label", "value": port},
                ],
                "style": _input_style(),
                "incomingEdges": [dict(edge) for edge in external],
                "outputsMetadata": [
                    {
                        "id": port,
                        "attrs": [{"key": "port_label", "value": port}],
                    }
                ],
            }
        )
        node["incomingEdges"] = [
            edge
            for edge in node.get("incomingEdges", [])
            if edge.get("sourceNodeId") in internal
        ] + [
            {
                "sourceNodeId": mirror_id,
                "sourceNodeOutputId": port,
                "targetNodeInputId": port,
                "metadata": {"port_label": port},
            }
        ]
    nodes.extend(mirrors)


def _collapse_mirror_boundary_passthroughs(nodes: list[dict[str, Any]]) -> None:
    """Merge an ``@output_mirror`` and ``@input_mirror`` for one tensor in one scope.

    A tensor produced inside child ``A`` and consumed inside sibling child ``B``,
    both under parent ``P``, otherwise renders as four tiles: ``A``'s ``@output``,
    ``P``'s ``@output_mirror``, ``P``'s ``@input_mirror`` and ``B``'s ``@input``. The
    two mirror tiles both sit in ``P`` and carry the same tensor name (a
    ``topk_weights`` feeding a ``topk_weights``), reading as a dangling duplicate
    pair. When an ``@input_mirror``'s incoming edges all come from a single
    ``@output_mirror`` in the *same* namespace, the two describe the same tensor at
    the same scope: drop the ``@input_mirror`` and repoint its consumers onto the
    ``@output_mirror``. Dataflow is unchanged (the same producer reaches the same
    consumers); one redundant tile per crossing disappears. General — any multi-slot
    boundary handoff between siblings collapses, not just the router's.
    """
    by_id = {str(node["id"]): node for node in nodes}
    consumers: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        for edge in node.get("incomingEdges", []):
            consumers.setdefault(str(edge.get("sourceNodeId")), []).append(node)

    removed: set[str] = set()
    for node in list(nodes):
        if _node_attr(node, "synthetic") != "@input_mirror":
            continue
        incoming = node.get("incomingEdges", [])
        sources = {str(edge.get("sourceNodeId")) for edge in incoming}
        if len(sources) != 1:
            continue
        (source_id,) = tuple(sources)
        producer = by_id.get(source_id)
        if producer is None or _node_attr(producer, "synthetic") != "@output_mirror":
            continue
        if producer.get("namespace") != node.get("namespace"):
            continue
        producer_ports = [
            str(item.get("id", "")) for item in producer.get("outputsMetadata", [])
        ]
        producer_port = producer_ports[0] if producer_ports else ""
        input_mirror_id = str(node["id"])
        for consumer in consumers.get(input_mirror_id, []):
            for edge in consumer.get("incomingEdges", []):
                if str(edge.get("sourceNodeId")) == input_mirror_id:
                    edge["sourceNodeId"] = source_id
                    edge["sourceNodeOutputId"] = producer_port
        removed.add(input_mirror_id)

    if removed:
        nodes[:] = [node for node in nodes if str(node["id"]) not in removed]


def _collapse_kernel_input_passthroughs(nodes: list[dict[str, Any]]) -> None:
    """Drop a module ``@input`` tile that only feeds a same-scope kernel port.

    A kernel input that is never transformed between the module boundary and the
    kernel renders as two consecutive same-named tiles in one box: the module
    ``@input:cu_seqlens`` and, right below it, the kernel's
    ``@kernel_in:…:cu_seqlens`` port. (Contrast a real kernel input like
    ``query_states``, whose port is fed by the rotary op, not by an input tile --
    the ``apply_rotary_pos_emb_vision`` handling -- so it shows one tile.) When a
    ``@kernel_port_in`` port is fed solely by an ``@input`` tile in the *same*
    namespace carrying the *same* name, and that input tile has no other consumer,
    the two describe one untransformed passthrough: drop the ``@input`` tile and
    let the kernel port read the input's own source. Dataflow is unchanged; one
    redundant tile disappears. General -- any same-scope module-input-into-kernel
    passthrough collapses, not just ``cu_seqlens``.
    """
    by_id = {str(node["id"]): node for node in nodes}
    consumers: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        for edge in node.get("incomingEdges", []):
            consumers.setdefault(str(edge.get("sourceNodeId")), []).append(node)

    def _port_name(node: dict[str, Any]) -> str:
        return _node_attr(node, "port_label") or str(node.get("label", ""))

    removed: set[str] = set()
    for port in list(nodes):
        if _node_attr(port, "synthetic") != "@kernel_port_in":
            continue
        incoming = port.get("incomingEdges", [])
        if len(incoming) != 1:
            continue
        source_id = str(incoming[0].get("sourceNodeId"))
        module_input = by_id.get(source_id)
        if module_input is None or source_id in removed:
            continue
        if _node_attr(module_input, "synthetic") != "@input":
            continue
        if module_input.get("namespace") != port.get("namespace"):
            continue
        if _port_name(module_input) != _port_name(port):
            continue
        # The input tile must be a pure passthrough: this kernel port is its only
        # consumer, and it forwards a single upstream source.
        if [c["id"] for c in consumers.get(source_id, [])] != [port["id"]]:
            continue
        upstream = module_input.get("incomingEdges", [])
        if len(upstream) != 1:
            continue
        forwarded = upstream[0]
        incoming[0]["sourceNodeId"] = forwarded.get("sourceNodeId")
        incoming[0]["sourceNodeOutputId"] = forwarded.get("sourceNodeOutputId")
        removed.add(source_id)

    if removed:
        nodes[:] = [node for node in nodes if str(node["id"]) not in removed]


# Boundary tiles a value flows *out of* / *into* at a namespace edge. A same-name
# ``@output``->``@input`` pair describes one untransformed tensor rendered as two
# tiles, so it collapses to one (see below).
_OUTPUT_BOUNDARY_SYNTHETIC = frozenset({"@output", "@output_mirror"})
_INPUT_BOUNDARY_SYNTHETIC = frozenset({"@input", "@input_mirror", "@kernel_port_in"})


def _collapse_same_name_boundary_passthroughs(nodes: list[dict[str, Any]]) -> None:
    """Collapse every same-name ``@output``->``@input`` boundary passthrough to one tile.

    A producer's ``@output`` tile feeding a consumer's ``@input`` (or
    ``@input_mirror``/``@kernel_port_in``) tile that carries the *identical*
    tensor name is one untransformed value rendered as two stacked/edge-linked
    tiles -- e.g. ``expand_kv/@output:key_states`` -> ``@output_mirror`` ->
    the attention ``@kernel_in:key_states`` port, or
    ``input_layernorm/@output:hidden_states`` -> ``self_attn/@input:hidden_states``.
    Fold each such pair to a single tile.

    General: keyed purely on same-name equality (via the producer's specific
    output-port label) and on the producer being an ``@output``-family boundary
    that is a pure passthrough of one upstream value -- no class/config/name
    literals. Boundaries where the name genuinely changes across the edge (a real
    rename such as ``@input:hidden_states <- @output:collapsed``) differ by name
    and are left untouched, as are real-op-fed inputs. Only same-*hierarchy-level*
    pairs collapse: when the producer @output and consumer @input belong to
    different owning modules the crossing is a real module entry/exit
    (``input_layernorm/@output:hidden_states`` -> ``self_attn/@input``) and both
    tiles are kept, so every module keeps its own @input/@output boundary.
    ``@loop_carried_*`` and ``@slice_out`` tiles are outside both synthetic sets,
    so loop boundaries and split tiles -- and the one permitted loop back edge --
    are never disturbed.

    Which tile survives: for a kernel port the port is kept (the kernel needs it)
    and made to read the producer's own upstream; for an ``@input``/``@input_mirror``
    the producer ``@output`` is kept and the dropped tile's consumers are repointed
    onto it. Repointing always targets an *upstream* node, so forward paths only
    shorten -- acyclicity is preserved -- and collapsed tiles are removed from the
    node list (not merely detached), so no dead nodes are introduced. Applied
    iteratively so multi-hop chains (``@output`` -> ``@output_mirror`` ->
    ``@kernel_port_in``) reduce to a single edge.
    """

    def _port_name(node: dict[str, Any]) -> str:
        return _node_attr(node, "port_label") or str(node.get("label", ""))

    def _source_port_name(source: dict[str, Any], output_id: Any) -> str:
        # Match the producer's *specific* output port so a multi-output @output
        # tile compares per port, not by the tile's aggregate label.
        for port in source.get("outputsMetadata", []):
            if str(port.get("id")) == str(output_id):
                for attr in port.get("attrs", []):
                    if attr.get("key") == "port_label":
                        return str(attr.get("value"))
        return _port_name(source)

    def _owner_namespace(node_id: str) -> str:
        # The module namespace that owns a boundary tile -- its id minus the
        # trailing ``/@...`` boundary token. Two boundary tiles sharing an owner
        # are at the same hierarchy level; differing owners means the crossing
        # enters or exits a module.
        idx = node_id.rfind("/@")
        return node_id[:idx] if idx != -1 else ""

    changed = True
    while changed:
        changed = False
        by_id = {str(node["id"]): node for node in nodes}
        consumers: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            for edge in node.get("incomingEdges", []):
                consumers.setdefault(str(edge.get("sourceNodeId")), []).append(node)

        removed: set[str] = set()
        for consumer in list(nodes):
            if str(consumer["id"]) in removed:
                continue
            consumer_syn = _node_attr(consumer, "synthetic")
            if consumer_syn not in _INPUT_BOUNDARY_SYNTHETIC:
                continue
            incoming = consumer.get("incomingEdges", [])
            if len(incoming) != 1:
                continue
            source_id = str(incoming[0].get("sourceNodeId"))
            if source_id in removed:
                continue
            producer = by_id.get(source_id)
            if producer is None:
                continue
            if _node_attr(producer, "synthetic") not in _OUTPUT_BOUNDARY_SYNTHETIC:
                continue
            # The producer must be a pure passthrough: exactly one upstream source.
            producer_upstream = producer.get("incomingEdges", [])
            if len(producer_upstream) != 1:
                continue
            output_id = incoming[0].get("sourceNodeOutputId", "0")
            if _source_port_name(producer, output_id) != _port_name(consumer):
                continue
            # Only collapse a same-name pair at the *same* hierarchy level. When
            # the producer @output and consumer @input belong to different owning
            # modules, the crossing is a real module entry/exit boundary
            # (``input_layernorm/@output:hidden_states`` -> ``self_attn/@input``);
            # keep both tiles so every module keeps its @input/@output boundary.
            if _owner_namespace(str(producer["id"])) != _owner_namespace(
                str(consumer["id"])
            ):
                continue

            if consumer_syn == "@kernel_port_in":
                # Keep the kernel port; make it read the producer's own upstream.
                forwarded = producer_upstream[0]
                incoming[0]["sourceNodeId"] = forwarded.get("sourceNodeId")
                incoming[0]["sourceNodeOutputId"] = forwarded.get("sourceNodeOutputId")
                # Drop the producer only when this port was its sole consumer.
                if [c["id"] for c in consumers.get(source_id, [])] == [consumer["id"]]:
                    removed.add(source_id)
            else:
                # @input / @input_mirror: keep the producer @output, drop this tile,
                # and repoint every consumer of it back onto the producer.
                consumer_id = str(consumer["id"])
                for downstream in consumers.get(consumer_id, []):
                    for edge in downstream.get("incomingEdges", []):
                        if str(edge.get("sourceNodeId")) == consumer_id:
                            edge["sourceNodeId"] = producer["id"]
                            edge["sourceNodeOutputId"] = output_id
                removed.add(consumer_id)

            changed = True

        if removed:
            nodes[:] = [node for node in nodes if str(node["id"]) not in removed]


def _prune_noop_cast_nodes(nodes: list[dict[str, Any]]) -> None:
    """Remove ``Cast`` nodes whose output dtype equals their (single) input's
    dtype — the AST-path counterpart of the torch-trace backend's no-op
    ``.to()``/``.float()``/``.type()`` removal.

    A cast never changes shape, so a ``Cast`` op whose own dtype matches its
    predecessor's dtype does nothing real: e.g. ``Glm5NextTextRMSNorm``'s
    ``hidden_states.to(input_dtype)`` is a genuine float32→bfloat16 downcast
    when called on a bfloat16 tensor directly, but becomes a no-op in
    ``Glm5NextTextHyperConnection`` (``attn_hc``/``ffn_hc``), which pre-casts
    its argument to float32 before calling it — the SAME instance-specific
    pattern the torch-trace fix caught. Keeping a no-op cast visible implies
    a data transformation that never actually happens; downstream consumers
    are rewired straight through to the real predecessor instead.
    """
    def _is_synthetic(candidate: dict[str, Any]) -> bool:
        return any(
            attr.get("key") == "synthetic" for attr in candidate.get("attrs", [])
        )

    node_by_id = {str(node.get("id")): node for node in nodes}
    redirect: dict[str, tuple[str, str]] = {}

    def _real_dtype_source(
        source: dict[str, Any], port: str
    ) -> tuple[dict[str, Any] | None, str]:
        """Follow single-input synthetic passthrough tiles back to the first real
        producer, returning ``(real_node, port)`` or ``(None, "")`` when the chain
        forks/merges or never reaches a real op.

        A non-synthetic source is returned unchanged (the original behaviour). A
        synthetic boundary/mirror tile carries no independent dtype ground truth --
        its dtype can be circularly back-filled from a consumer -- so its true
        upstream dtype is whatever real op ultimately feeds it.
        """
        node = source
        seen: set[str] = set()
        while node is not None and _is_synthetic(node):
            nid = str(node.get("id"))
            if nid in seen:
                return None, ""
            seen.add(nid)
            incoming = [
                edge
                for edge in node.get("incomingEdges", []) or []
                if not _is_loop_back_edge(str(edge.get("sourceNodeId") or ""), nid)
            ]
            if len(incoming) != 1:
                return None, ""
            edge = incoming[0]
            port = str(edge.get("sourceNodeOutputId", "0"))
            node = node_by_id.get(str(edge.get("sourceNodeId") or ""))
        if node is None:
            return None, ""
        return node, port

    for node in nodes:
        if str(node.get("label") or "") != "Cast":
            continue
        if _node_attr(node, "constant") == "true":
            # A constant/buffer cast (e.g. ``self.inv_freq`` read via
            # ``external_inputs``) is a value SOURCE, never a same-dtype
            # passthrough to elide. Eliding it would delete the operand and
            # collapse its consumer's two edges onto the sibling activation.
            continue
        incoming = node.get("incomingEdges", [])
        if len(incoming) != 1:
            continue  # only the simple single-input case is unambiguous
        edge = incoming[0]
        source_id = str(edge.get("sourceNodeId") or "")
        source_port = str(edge.get("sourceNodeOutputId", "0"))
        source_node = node_by_id.get(source_id)
        if not source_id or source_node is None:
            continue
        # Compare the cast's dtype against a REAL producer's dtype. Boundary/mirror
        # ports (`@input`, `@input:NAME`, `@output`, ...) can have their OWN dtype
        # back-filled from whatever consumes them (no independent ground truth for
        # a synthetic port), so comparing directly against a synthetic predecessor
        # risks a circular false match. Instead walk back through pure single-input
        # synthetic passthrough tiles to the first real producer and use ITS dtype:
        # a genuine `gate.to(torch.float32)` whose `@input:gate` traces to a
        # bfloat16 producer correctly stays a real downcast, while an already-
        # float32 value flowing through a boundary into a `.to(torch.float32)`
        # (hc_head's pre-`.float()`-ed input_norm) is exposed as the no-op it is.
        # Consumers still rewire to the immediate source, preserving the boundary.
        cmp_node, cmp_port = _real_dtype_source(source_node, source_port)
        if cmp_node is None:
            continue
        own_spec = node_output_spec(node, "0")
        source_spec = node_output_spec(cmp_node, cmp_port)
        if own_spec is None or source_spec is None or not own_spec.dtype:
            continue
        if not source_spec.dtype or own_spec.dtype != source_spec.dtype:
            continue
        redirect[str(node.get("id"))] = (source_id, source_port)

    if not redirect:
        return

    def _resolve(node_id: str, port: str) -> tuple[str, str]:
        seen: set[str] = set()
        while node_id in redirect and node_id not in seen:
            seen.add(node_id)
            node_id, port = redirect[node_id]
        return node_id, port

    removed_ids = set(redirect)
    for node in nodes:
        if str(node.get("id")) in removed_ids:
            continue
        incoming = node.get("incomingEdges", [])
        if not incoming:
            continue
        changed = False
        new_incoming = []
        for edge in incoming:
            source_id = str(edge.get("sourceNodeId") or "")
            if source_id in redirect:
                real_id, real_port = _resolve(
                    source_id, str(edge.get("sourceNodeOutputId", "0"))
                )
                edge = dict(edge)
                edge["sourceNodeId"] = real_id
                edge["sourceNodeOutputId"] = real_port
                changed = True
            new_incoming.append(edge)
        if changed:
            node["incomingEdges"] = new_incoming

    nodes[:] = [node for node in nodes if str(node.get("id")) not in removed_ids]


def _elide_view_split_onto_producer(nodes: list[dict[str, Any]]) -> None:
    """Fold a multi-output slice op (``Unbind``/``Split``/``Chunk``) into the
    layout-only op that produces its input, exposing the slices as named output
    ports on that producer instead of via a separate tile.

    The ``q, k, v = qkv(h).reshape(...).permute(...).unbind(0)`` idiom (and its
    ``... .transpose(...).split(...)`` cousins) chains pure view ops that
    together do one thing -- carve a fused tensor into named slices. The trailing
    ``unbind``/``split`` computes nothing; when its sole producer is itself a
    layout-only op feeding nobody else, the split tile is a redundant hop -- the
    bare ``unbind`` → ``q_norm``/``k_norm``/``transpose`` pass-through the owner
    flagged. Dropping it and re-homing its per-slice ports (``query_states`` /
    ``key_states`` / ``value_states``, shapes intact) onto the producer keeps
    every slice visible -- no computation is hidden -- with one fewer hop.

    Guarded to stay general and safe:
      * only the pure slice ops (``Unbind``/``Split``/``Chunk``);
      * exactly one incoming edge (a single producer port to re-home onto);
      * the producer is a REAL layout-only op (not synthetic, not a compute op
        like ``Linear`` -- a fused-weight split such as ``pre_w/post_w/comb_w``
        stays its own tile, since its producer genuinely transforms values);
      * that producer port feeds ONLY this split (else re-labeling its port as
        slices would corrupt the other consumer) and the producer exposes a
        single output port to replace.
    """
    node_by_id = {str(node.get("id")): node for node in nodes}

    def _is_synthetic(candidate: dict[str, Any]) -> bool:
        return any(
            attr.get("key") == "synthetic" for attr in candidate.get("attrs", [])
        )

    # How many edges read each (producer_id, port) endpoint.
    consumer_count: dict[tuple[str, str], int] = {}
    for node in nodes:
        for edge in node.get("incomingEdges", []):
            key = (
                str(edge.get("sourceNodeId") or ""),
                str(edge.get("sourceNodeOutputId", "0")),
            )
            consumer_count[key] = consumer_count.get(key, 0) + 1

    redirect: dict[str, str] = {}  # split id -> producer id (ports preserved)
    for node in nodes:
        if str(node.get("label") or "") not in {"Unbind", "Split", "Chunk"}:
            continue
        incoming = node.get("incomingEdges", [])
        if len(incoming) != 1:
            continue
        edge = incoming[0]
        source_id = str(edge.get("sourceNodeId") or "")
        source_port = str(edge.get("sourceNodeOutputId", "0"))
        producer = node_by_id.get(source_id)
        if producer is None or _is_synthetic(producer):
            continue
        if not is_layout_only_label(str(producer.get("label") or "")):
            continue
        if consumer_count.get((source_id, source_port), 0) != 1:
            continue  # producer port fans out beyond this split
        if len(producer.get("outputsMetadata", []) or []) != 1:
            continue  # producer already multi-port -- don't clobber
        # Re-home the split's per-slice ports (labels + shapes) onto the producer.
        producer["outputsMetadata"] = [
            dict(port) for port in node.get("outputsMetadata", [])
        ]
        names = _node_attr(node, "output_names")
        if names is not None:
            attrs = [
                attr
                for attr in producer.get("attrs", [])
                if attr.get("key") != "output_names"
            ]
            attrs.append({"key": "output_names", "value": names})
            producer["attrs"] = attrs
        redirect[str(node.get("id"))] = source_id

    if not redirect:
        return

    removed_ids = set(redirect)
    for node in nodes:
        if str(node.get("id")) in removed_ids:
            continue
        incoming = node.get("incomingEdges", [])
        if not incoming:
            continue
        changed = False
        new_incoming = []
        for edge in incoming:
            source_id = str(edge.get("sourceNodeId") or "")
            if source_id in redirect:
                edge = dict(edge)
                edge["sourceNodeId"] = redirect[source_id]
                changed = True
            new_incoming.append(edge)
        if changed:
            node["incomingEdges"] = new_incoming

    nodes[:] = [node for node in nodes if str(node.get("id")) not in removed_ids]


def _port_meta_attr(port: dict[str, Any], key: str) -> str | None:
    """Read one attr value from a port's ``outputsMetadata`` entry."""
    for attr in port.get("attrs", []):
        if attr.get("key") == key:
            value = attr.get("value")
            if isinstance(value, str):
                return value
    return None


def _add_split_slice_tiles(nodes: list[dict[str, Any]]) -> None:
    """Surface each output slice of a multi-output split (``Unbind`` / ``Split`` /
    ``Chunk``) as its own named, input-styled passthrough tile.

    Model Explorer does not render output-*port* names, so a split's per-slice
    ports (``query_states`` / ``key_states`` / ``value_states``) are invisible on
    the split node itself, and folding the split away would hide the split
    entirely. Mirroring how the rotary block surfaces named ``cos`` / ``sin``
    tiles, synthesize one tile per slice carrying the slot label and that slice's
    shape, and rewire every consumer of the port to read the tile instead. The
    split node stays visible -- no computation is hidden -- and every slice now
    shows a readable name and its own shape.

    General: fires for any ``Unbind`` / ``Split`` / ``Chunk`` node with two or
    more output slices. Per-slice shapes come from ``outputsMetadata`` when a
    ``ShapeInferencer`` drove the build; without one the tiles still appear, named
    from the node's ``output_names`` attr (index-aligned to the ports consumers
    already read) -- the split's topology must not depend on shape inference being
    available. Uses a distinct ``@slice_out`` synthetic tag -- it must not
    reintroduce the retired ``@split_out:`` id prefix.
    """
    tiles: list[dict[str, Any]] = []
    # (split_id, port_id) -> (tile_id, tile_port) for every slice we surface.
    port_redirect: dict[tuple[str, str], tuple[str, str]] = {}
    for node in nodes:
        if str(node.get("label") or "") not in {"Unbind", "Split", "Chunk"}:
            continue
        split_id = str(node.get("id"))
        namespace = str(node.get("namespace", ""))
        # A constant split (e.g. ``pre_b, post_b, comb_b = self.base.split(...)``)
        # produces constant slices; each tile must carry the tag so the render
        # filter drops the whole learned-param closure together instead of
        # orphaning the untagged slice tiles.
        split_is_constant = _node_attr(node, "constant") == "true"
        ports = node.get("outputsMetadata", []) or []
        # (port_id, port_label, shape_value) per slice. Prefer the shape-bearing
        # ``outputsMetadata``; fall back to the ``output_names`` attr so the tiles
        # exist even when no ShapeInferencer stamped per-port shapes.
        slices: list[tuple[str, str, str | None]] = []
        if len(ports) >= 2:
            for port in ports:
                port_id = str(port.get("id", "0"))
                label = _port_meta_attr(port, "port_label") or f"slice_{port_id}"
                slices.append((port_id, label, _port_meta_attr(port, "shape")))
        else:
            names = [
                name.strip()
                for name in (_node_attr(node, "output_names") or "").split(",")
                if name.strip()
            ]
            if len(names) >= 2:
                slices = [(str(i), name, None) for i, name in enumerate(names)]
        if len(slices) < 2:
            continue  # a single-output "split" is not really a split
        for port_id, port_label, shape_value in slices:
            tile_id = f"{split_id}^@slice_out:{port_id}"
            tile = _make_group_input_node(
                input_id=tile_id,
                label=port_label,
                namespace=namespace,
                port_label=port_label,
                incoming_edges=[
                    {
                        "sourceNodeId": split_id,
                        "sourceNodeOutputId": port_id,
                        "targetNodeInputId": "0",
                        "metadata": {"port_label": port_label},
                    }
                ],
            )
            # A slice tile is not a real graph-input boundary; tag it distinctly so
            # boundary passes (mirroring, pruning) leave it alone.
            tile["attrs"] = [
                {"key": "synthetic", "value": "@slice_out"},
                {"key": "port_label", "value": port_label},
            ]
            if split_is_constant:
                tile["attrs"].append({"key": "constant", "value": "true"})
            if shape_value:
                dims, dtype = _split_shape_dtype(shape_value)
                apply_shape_attrs(tile, TensorSpec(tuple(dims), dtype or "float16"))
            tiles.append(tile)
            port_redirect[(split_id, port_id)] = (tile_id, "0")

    if not port_redirect:
        return

    # Repoint each existing consumer of a split port onto its new tile. The tiles
    # are not in ``nodes`` yet, so their own edge back to the split is untouched.
    for node in nodes:
        for edge in node.get("incomingEdges", []) or []:
            key = (
                str(edge.get("sourceNodeId") or ""),
                str(edge.get("sourceNodeOutputId", "0")),
            )
            target = port_redirect.get(key)
            if target is not None:
                edge["sourceNodeId"], edge["sourceNodeOutputId"] = target

    nodes.extend(tiles)


def _node_output_dims(node: dict[str, Any], port: str) -> list[str] | None:
    """Dims recorded for one output port of ``node`` (dtype stripped), if known.

    Reads the per-port ``outputsMetadata`` shape (present in the final payload and
    for multi-port split/unbind tiles), falling back to the node-level
    ``output_shape`` attr (stamped when a ``ShapeInferencer`` drives the build).
    Returns ``None`` when no shape is recorded yet.
    """
    meta = node.get("outputsMetadata", []) or []
    if len(meta) == 1:
        chosen: dict[str, Any] | None = meta[0]
    else:
        chosen = next((m for m in meta if str(m.get("id", "0")) == str(port)), None)
    if chosen is not None:
        for attr in chosen.get("attrs", []):
            if attr.get("key") == "shape":
                dims, _dtype = _split_shape_dtype(str(attr.get("value", "")))
                return dims
    for attr in node.get("attrs", []):
        if attr.get("key") == "output_shape":
            dims, _dtype = _split_shape_dtype(str(attr.get("value", "")))
            return dims
    return None


def _elide_noop_single_input_concat(nodes: list[dict[str, Any]]) -> None:
    """Fold a single-input ``Concat`` whose output shape equals its input shape
    onto its producer.

    ``torch.cat([x], dim=d)`` is ``x`` -- a Concat left with one incoming edge
    concatenates a tensor with nothing and computes nothing. The one that survives
    here is the vision attention sdpa fallback's per-chunk reassembly
    (``torch.cat([interface(q,k,v) for q,k,v in zip(*splits)], dim=1)``): the
    windowed comprehension collapses to a single representative iteration, so the
    reassembly cat reads exactly one kernel output and restores the same
    ``[Pv, 1024]`` it received. The block-diagonal windowing that cat re-stitches
    is already carried by the kernel's ``cu_seqlens`` input, so the cat re-stitches
    nothing the graph does not already show -- eliding it hides no computation and
    enforces the owner invariant that a Concat has more than one input.

    General and guarded: fires only when a node is labeled ``Concat`` and has
    exactly one incoming edge -- ``torch.cat`` over a single tensor is that tensor,
    so it necessarily preserves shape. When both endpoints' dims are recorded and
    *differ*, the node is left untouched: that is a replication (``cat([x, x])``,
    already relabeled ``Tile`` upstream), not an identity.
    """
    node_by_id = {str(node.get("id")): node for node in nodes}

    redirect: dict[str, tuple[str, str]] = {}  # concat id -> (producer id, port)
    for node in nodes:
        if str(node.get("label") or "") != "Concat":
            continue
        incoming = node.get("incomingEdges", [])
        if len(incoming) != 1:
            continue
        edge = incoming[0]
        source_id = str(edge.get("sourceNodeId") or "")
        source_port = str(edge.get("sourceNodeOutputId", "0"))
        producer = node_by_id.get(source_id)
        if producer is None:
            continue
        out_dims = _node_output_dims(node, "0")
        in_dims = _node_output_dims(producer, source_port)
        if out_dims is not None and in_dims is not None and out_dims != in_dims:
            continue  # shape-changing replication -- not a provable identity
        redirect[str(node.get("id"))] = (source_id, source_port)

    if not redirect:
        return

    removed_ids = set(redirect)
    for node in nodes:
        if str(node.get("id")) in removed_ids:
            continue
        incoming = node.get("incomingEdges", [])
        if not incoming:
            continue
        changed = False
        new_incoming = []
        for edge in incoming:
            source_id = str(edge.get("sourceNodeId") or "")
            if source_id in redirect:
                producer_id, producer_port = redirect[source_id]
                edge = dict(edge)
                edge["sourceNodeId"] = producer_id
                edge["sourceNodeOutputId"] = producer_port
                changed = True
            new_incoming.append(edge)
        if changed:
            node["incomingEdges"] = new_incoming

    nodes[:] = [node for node in nodes if str(node.get("id")) not in removed_ids]


def _dim_is_weak(dim: str) -> bool:
    """A shape dim carries no real information when it is ``-1`` or a collapsed
    product (``BxS``) — either hides the batch/seq structure a sibling edge end
    may still spell out concretely."""
    d = dim.strip()
    if d in {"-1", "?", ""}:
        return True
    # A collapsed dim like ``BxS`` / ``B*S`` keeps an ``x``/``*`` inside a single
    # token (the shape separator is `` x `` with spaces, so a real dim never does).
    return ("x" in d.lower() or "*" in d) and not d.lstrip("-").isdigit()


def _port_shape_attrs(metadata: dict[str, Any]) -> dict[str, Any] | None:
    for attr in metadata.get("attrs", []):
        if attr.get("key") == "shape":
            return attr
    return None


def _split_shape_dtype(text: str) -> tuple[list[str], str]:
    """Split a stored ``[B, S, 4096] float16`` value into dims and dtype."""
    stripped = text.strip()
    dtype = ""
    if stripped.endswith("]"):
        shape_text = stripped
    else:
        # Bracketed dims followed by a dtype suffix: ``[B, S, 4096] float16``.
        close = stripped.rfind("]")
        if close != -1:
            dtype = stripped[close + 1 :].strip()
            shape_text = stripped[: close + 1]
        else:
            # Legacy `` x ``-separated form without brackets.
            tail = stripped.rsplit(" ", 1)
            if len(tail) == 2 and SHAPE_SEPARATOR not in tail[1]:
                shape_text, dtype = tail[0], tail[1]
            else:
                shape_text = stripped
    return parse_shape_dims(shape_text), dtype


# Detail keys whose value is a scalar operand argument (not a tensor edge), e.g.
# ``dim: -1`` on an unsqueeze/squeeze/select. These are the "inputs not shown in
# the graph" the profiler records as ``[]``/``Scalar``/concrete-value entries.
_SCALAR_DETAIL_KEYS = frozenset(
    {"dim", "dim0", "dim1", "select_dim", "resize_dim", "start_dim", "end_dim"}
)


def _is_operation_node(node: dict[str, Any]) -> bool:
    """A node representing a real computation (op/kernel/module), not a synthetic
    boundary/mirror/loop tile — the nodes worth describing profiler-style and
    type-checking."""
    if _is_synthetic_input(node) or _is_synthetic_output(node):
        return False
    if _node_attr(node, "synthetic") == "@loop_carried":
        return False
    # A genuine op/kernel/module node carries both a class label and an inferred
    # output shape; synthetic passthrough tiles carry neither together.
    return bool(_node_attr(node, "class_name")) and bool(
        _node_attr(node, "output_shape")
    )


def _producer_output_dims(
    source: dict[str, Any] | None, port: str
) -> tuple[list[str], str] | None:
    """Resolve one producer output port to ``(dims, dtype)``, honouring per-port
    shapes on multi-output (split/unbind) producers, else the node's
    ``output_shape`` attr."""
    if source is None:
        return None
    metadata = _find_port_metadata(source, "outputsMetadata", port)
    if metadata is not None:
        shape_attr = _port_shape_attrs(metadata)
        if shape_attr is not None:
            return _split_shape_dtype(str(shape_attr.get("value", "")))
    text = _node_attr(source, "output_shape")
    if text:
        return _split_shape_dtype(text)
    return None


def _op_scalar_details(node: dict[str, Any]) -> list[str]:
    """Scalar operand arguments declared in a node's ``details`` attr (``dim: -1``
    -> ``"-1"``), in declaration order."""
    raw = _node_attr(node, "details")
    if not raw:
        return []
    scalars: list[str] = []
    for part in raw.split(";"):
        key, sep, value = part.partition(":")
        if sep and key.strip() in _SCALAR_DETAIL_KEYS:
            scalars.append(value.strip())
    return scalars


def _annotate_op_input_signatures(nodes: list[dict[str, Any]]) -> None:
    """Attach PyTorch-profiler-style input descriptions + a machine-readable
    ``op_type`` to every operation node.

    The profiler represents an operator's inputs as three parallel lists: shapes
    (``Input Dims`` — a scalar is ``[]``), types (``Input type`` — a scalar is
    ``"Scalar"``), and concrete scalar values (``Concrete Inputs`` — ``""`` for a
    tensor). We mirror that: tensor operands come from the node's incoming edges
    (resolved to each producer's output-port shape/dtype, in ``targetNodeInputId``
    order), followed by the scalar operand args declared in ``details`` (e.g. an
    ``unsqueeze``'s ``dim``), which are real inputs that are never drawn as graph
    edges. This makes those hidden scalar inputs explicit and gives the type-check
    pass a uniform, profiler-shaped view of every op's operands.
    """
    node_by_id = {str(node.get("id")): node for node in nodes}
    for node in nodes:
        if not _is_operation_node(node):
            continue
        input_shapes: list[list[str]] = []
        input_types: list[str] = []
        concrete_inputs: list[str] = []
        edges = sorted(
            node.get("incomingEdges", []),
            key=lambda edge: str(edge.get("targetNodeInputId", "0")),
        )
        for edge in edges:
            source = node_by_id.get(str(edge.get("sourceNodeId") or ""))
            port = str(edge.get("sourceNodeOutputId", "0"))
            resolved = _producer_output_dims(source, port)
            if resolved is None:
                input_shapes.append([])
                input_types.append("Tensor")
            else:
                dims, dtype = resolved
                input_shapes.append(dims)
                input_types.append(dtype or "Tensor")
            # A constant/learned-weight/buffer operand keeps its shape (so a
            # shape jump like the rotary ``Multiply`` stays explained) but is
            # typed ``Constant`` so the type-check pass counts only real
            # activation operands.
            if source is not None and _node_attr(source, "constant") == "true":
                input_types[-1] = "Constant"
            concrete_inputs.append("")
        for value in _op_scalar_details(node):
            input_shapes.append([])
            input_types.append("Scalar")
            concrete_inputs.append(value)
        op_type = node.get("label") or _node_attr(node, "class_name")
        if op_type:
            _set_node_attr(node, "op_type", str(op_type))
        _set_node_attr(node, "input_shapes", json.dumps(input_shapes))
        _set_node_attr(node, "input_types", json.dumps(input_types))
        _set_node_attr(node, "concrete_inputs", json.dumps(concrete_inputs))


def _stamp_boundary_input_shapes(
    nodes: list[dict[str, Any]], shape_inferencer: ShapeInferencer
) -> None:
    """Restamp leaf ``@input`` boundary tiles from meta-device ground truth.

    A forward-parameter boundary (``@input:<param>``) that is a pure leaf -- no
    incoming producer edge -- is shaped upstream by generic activation
    heuristics, which mis-size a boundary whose name merely resembles an
    activation. The clearest case: the ``attention_mask`` boundary reaching the
    indexer is stamped ``(B, S, hidden)`` by the ``"attention" in id`` heuristic,
    when the real parameter is a ``(B, S)`` bool mask -- inflating the downstream
    ``cat`` operand to rank 4 and tripping the type-check.

    When a meta forward observed the real parameter shape,
    ``boundary_input_spec`` returns it and that is authoritative, so restamp the
    tile. A connected boundary (it already carries a real producer edge, e.g.
    ``position_embeddings`` from a ``rotary_pos_emb``) inherits its producer's
    shape and is left untouched -- meta never overrides a real dataflow edge.

    Because this pass *overrides* an already-shaped tile, it trusts only the
    class-scoped (locally-definite) meta shape, never the global fallback -- a
    parameter observed in just one module family (``position_ids`` as ``(1, S)``
    on the text stack, but ``[Pv, 2]`` on the vision rotary) must not have its
    other family's authoritative shape clobbered.
    """
    for node in nodes:
        if _node_attr(node, "synthetic") not in {"@input", "@input_mirror"}:
            continue
        if node.get("incomingEdges"):
            continue
        param = _node_attr(node, "boundary_input") or str(node.get("label") or "")
        param = param.strip()
        if not param:
            continue
        namespace = str(node.get("namespace") or "")
        spec = shape_inferencer.boundary_input_spec(
            param, namespace, class_scoped_only=True
        )
        if spec is None:
            continue
        apply_shape_attrs(node, spec)


def _reconcile_edge_endpoint_shapes(nodes: list[dict[str, Any]]) -> None:
    """Make both ends of every edge agree on shape where one end is under-specified.

    A boundary/mirror port sometimes falls back to a collapsed ``-1``/``BxS``
    shape even though the op on the other side of the edge still spells out the
    concrete ``B x S x …`` dims. Only reconcile when the two ends have the SAME
    rank — a genuine rank change (a ``view(-1, hidden)`` flatten, a
    ``masked_scatter`` combine) must be preserved, not papered over — and only
    ever replace a weak dim with a concrete one, never the reverse.
    """
    node_by_id = {str(node.get("id")): node for node in nodes}

    def _reconcile(a: dict[str, Any], b: dict[str, Any]) -> None:
        attr_a, attr_b = _port_shape_attrs(a), _port_shape_attrs(b)
        if attr_a is None or attr_b is None:
            return
        dims_a, dt_a = _split_shape_dtype(str(attr_a.get("value", "")))
        dims_b, dt_b = _split_shape_dtype(str(attr_b.get("value", "")))
        if not dims_a or len(dims_a) != len(dims_b):
            return
        changed_a = changed_b = False
        for i, (da, db) in enumerate(zip(dims_a, dims_b)):
            if da == db:
                continue
            if _dim_is_weak(da) and not _dim_is_weak(db):
                dims_a[i] = db
                changed_a = True
            elif _dim_is_weak(db) and not _dim_is_weak(da):
                dims_b[i] = da
                changed_b = True
        if changed_a:
            _write_port_shape(a, attr_a, dims_a, dt_a)
        if changed_b:
            _write_port_shape(b, attr_b, dims_b, dt_b)

    for node in nodes:
        for edge in node.get("incomingEdges", []):
            source = node_by_id.get(str(edge.get("sourceNodeId") or ""))
            if source is None:
                continue
            src_port = str(edge.get("sourceNodeOutputId", "0"))
            tgt_port = str(edge.get("targetNodeInputId", "0"))
            src_md = _find_port_metadata(source, "outputsMetadata", src_port)
            tgt_md = _find_port_metadata(node, "inputsMetadata", tgt_port)
            if src_md is None or tgt_md is None:
                continue
            _reconcile(src_md, tgt_md)


def _assert_edge_endpoint_shapes_agree(nodes: list[dict[str, Any]]) -> None:
    """Fail the export when both ends of a wire spell concrete, disagreeing dims.

    Run AFTER ``_reconcile_edge_endpoint_shapes`` (which fills weak dims from a
    concrete sibling): any remaining position where the source output and the
    target input port both carry a non-weak dim yet disagree is a genuine wiring
    fidelity bug — e.g. a vision boundary handing ``[Pv, 1024]`` into a patch-embed
    that consumes ``[Pv, 1176]``. A real rank change (reshape/flatten) is a
    different rank and is left alone. All violations are collected so one run
    surfaces every bad edge.
    """
    node_by_id = {str(node.get("id")): node for node in nodes}
    violations: list[str] = []
    for node in nodes:
        for edge in node.get("incomingEdges", []):
            source = node_by_id.get(str(edge.get("sourceNodeId") or ""))
            if source is None:
                continue
            src_port = str(edge.get("sourceNodeOutputId", "0"))
            tgt_port = str(edge.get("targetNodeInputId", "0"))
            src_md = _find_port_metadata(source, "outputsMetadata", src_port)
            tgt_md = _find_port_metadata(node, "inputsMetadata", tgt_port)
            if src_md is None or tgt_md is None:
                continue
            attr_s, attr_t = _port_shape_attrs(src_md), _port_shape_attrs(tgt_md)
            if attr_s is None or attr_t is None:
                continue
            dims_s, _ = _split_shape_dtype(str(attr_s.get("value", "")))
            dims_t, _ = _split_shape_dtype(str(attr_t.get("value", "")))
            if not dims_s or not dims_t or len(dims_s) != len(dims_t):
                continue
            for ds, dt in zip(dims_s, dims_t):
                if ds == dt or _dim_is_weak(ds) or _dim_is_weak(dt):
                    continue
                violations.append(
                    f"  {source.get('id')} [{source.get('label')}]"
                    f" out#{src_port}={format_shape_dims(dims_s)}"
                    f"  ->  {node.get('id')} [{node.get('label')}]"
                    f" in#{tgt_port}={format_shape_dims(dims_t)}"
                )
                break
    if violations:
        raise ValueError(
            "Edge endpoint shapes disagree on concrete dims "
            f"({len(violations)} wire(s)):\n" + "\n".join(violations)
        )


def _find_port_metadata(
    node: dict[str, Any], key: str, port: str
) -> dict[str, Any] | None:
    items = node.get(key, [])
    match = next((m for m in items if str(m.get("id", "0")) == port), None)
    if match is not None:
        return match
    return items[0] if len(items) == 1 else None


def _write_port_shape(
    metadata: dict[str, Any], shape_attr: dict[str, Any], dims: list[str], dtype: str
) -> None:
    display = format_shape_dims(dims)
    if dtype:
        display = f"{display} {dtype}"
    shape_attr["value"] = display
    for attr in metadata.get("attrs", []):
        if attr.get("key") == "tensor_shape":
            tensor = format_shape_dims(dims)
            attr["value"] = f"{tensor} {dtype}" if dtype else tensor


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
        if _is_synthetic_input(node) or _node_attr(node, "synthetic") == "@loop_carried":
            continue
        dead.add(node_id)
        for edge in node.get("incomingEdges", []):
            source_id = str(edge["sourceNodeId"])
            outgoing_count[source_id] = max(outgoing_count.get(source_id, 0) - 1, 0)
            pending.append(source_id)
    if dead:
        nodes[:] = [node for node in nodes if str(node.get("id")) not in dead]

    _prune_dangling_op_leaves(nodes)


def _prune_dangling_op_leaves(nodes: list[dict[str, Any]]) -> None:
    """Remove no-consumer op leaves whose inputs are shared with live nodes.

    Beyond dead boundary ports, a plain op whose result feeds only control flow
    the graph never models (an int kernel arg, an index, a mask) surfaces with no
    consumer — e.g. the vision ``max_seqlen`` kernel input or the patch-merger
    ``Clamp``. Those are safe to drop. But some no-consumer leaves cap a *dedicated*
    compute chain the viewer still wants shown (the sparse-attention indexer's
    ``Split``/``Expand``/``TopK`` over the ``kv_a_layernorm`` output). Unravelling
    those transitively deletes legitimate, named layers.

    Prune a leaf only when removing it orphans none of its producers — i.e. every
    producer keeps another consumer, or is a synthetic input allowed to dangle.
    That drops artifacts sitting on shared tensors while leaving dedicated chains
    intact. Iterate to a fixpoint; the orphan guard prevents chain unravelling.
    """
    while True:
        outgoing_count: dict[str, int] = {}
        for node in nodes:
            for edge in node.get("incomingEdges", []):
                source_id = str(edge["sourceNodeId"])
                outgoing_count[source_id] = outgoing_count.get(source_id, 0) + 1
        node_by_id = {str(node.get("id")): node for node in nodes}

        dead: set[str] = set()
        for node in nodes:
            node_id = str(node.get("id"))
            if outgoing_count.get(node_id, 0):
                continue
            if (
                node_id == "@output"
                or _is_synthetic_input(node)
                or _is_synthetic_output(node)
                or _node_attr(node, "synthetic") == "@loop_carried"
            ):
                continue
            # Count edges to each producer so a producer feeding this leaf more
            # than once is judged on its remaining (other) consumers.
            edges_to: dict[str, int] = {}
            for edge in node.get("incomingEdges", []):
                source_id = str(edge["sourceNodeId"])
                edges_to[source_id] = edges_to.get(source_id, 0) + 1
            orphans = False
            for source_id, count in edges_to.items():
                producer = node_by_id.get(source_id)
                if producer is not None and _is_synthetic_input(producer):
                    continue
                if outgoing_count.get(source_id, 0) - count < 1:
                    orphans = True
                    break
            if not orphans:
                dead.add(node_id)

        if not dead:
            return
        nodes[:] = [node for node in nodes if str(node.get("id")) not in dead]


def _nested_namespace_segment(nested_block: BlockNode, nested_label: str) -> str:
    if nested_block.class_name == "KernelPipeline":
        return _sanitize_namespace_segment(nested_label)
    if nested_block.attr_name.startswith("@"):
        if nested_label and nested_label != nested_block.attr_name:
            return _sanitize_namespace_segment(nested_label)
        return _sanitize_namespace_segment(nested_block.attr_name.lstrip("@"))
    return _sanitize_namespace_segment(nested_block.attr_name)


def _nested_group_segment(
    nested_block: BlockNode,
    nested_label: str,
    *,
    has_tile: bool,
    duplicate_labels: set[str],
) -> str:
    """Namespace for a nested submodule, unique when siblings share a class label."""
    if nested_block.class_name == "KernelPipeline":
        return _sanitize_namespace_segment(nested_label)
    if nested_label in duplicate_labels:
        return _sanitize_namespace_segment(
            nested_block.attr_name.lstrip("@") or nested_block.attr_name
        )
    if has_tile:
        return _sanitize_namespace_segment(nested_label)
    return _nested_namespace_segment(nested_block, nested_label)


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
    inline_expansion: bool = True,
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
    # Blocks that are re-exported as their own nested diagram materialize their
    # own loop-carried boundaries there; exclude their specs from this parent
    # scope so an inlined child loop is not duplicated as an orphan loop frame.
    nested_diagrams = collect_nested_diagrams(block_tree, basic_ops=basic_ops)
    exclude_carried_from = frozenset(id(block) for _label, block in nested_diagrams)
    computation = build_computation_graph(
        block_tree,
        basic_ops=basic_ops,
        # Return branches may be consumed by later caller-side operations. Keeping
        # them also preserves real setup work such as mHC's Sinkhorn projection.
        strip_unused_return_branches=False,
        inline_expansion=inline_expansion,
        exclude_carried_from=exclude_carried_from,
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
    nested_label_counts: dict[str, int] = {}
    for nested_label, _nested_block in nested_diagrams:
        nested_label_counts[nested_label] = nested_label_counts.get(nested_label, 0) + 1
    duplicate_nested_labels = {
        label for label, count in nested_label_counts.items() if count > 1
    }
    for nested_label, nested_block in nested_diagrams:
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
            _nested_group_segment(
                nested_block,
                nested_label,
                has_tile=tile_id is not None,
                duplicate_labels=duplicate_nested_labels,
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
            inline_expansion=inline_expansion,
            exclude_carried_from=exclude_carried_from,
        )
        nested_nodes = _computation_nodes(
            nested_computation,
            id_prefix=nested_prefix,
            namespace_prefix=nested_namespace,
        )
        section_nodes.extend(nested_nodes)
        # ``collect_nested_diagrams`` walks arbitrarily deep (a composite nested
        # inside another non-inline composite, e.g. ``scorer`` inside
        # ``indexer``), returning one flat, parent-before-child ordered list.
        # ``tile_ids`` built from the *top* computation only ever covers the
        # first level, so a doubly-nested entry never finds its real tile and
        # falls back to a fresh, unwired sibling namespace/diagram instead of
        # the group its own parent's nested diagram actually produced. Merge
        # each nested diagram's own tile ids in as it is built so any deeper
        # entries processed later resolve against it.
        tile_ids.update(_block_tile_ids(nested_computation, id_prefix=nested_prefix))
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

    def _return_slot_label(source_id: str, source_port: str) -> str | None:
        """Name a boundary input after the child return slot it reads.

        Symmetric to the output side (``_resolve_slot_names_for_prefix``): when a
        boundary reads one slice of a multi-return child (the rotary block returns
        ``(cos, sin)`` as the caller's ``position_embeddings`` tuple), it is named
        after that slot so the slices stay distinct tiles rather than collapsing
        onto the tuple parameter's name. Runs before ``@output`` tiles exist, so it
        matches the child's recorded return producers against the raw source id.
        """
        for child in block_tree.children:
            slots = child.forward_return_slots
            if not slots or len(slots) < 2:
                continue
            order = child.forward_return_order or list(slots.keys())
            producer_slots: dict[str, list[str]] = {}
            for slot in order:
                producer = slots.get(slot)
                if producer is not None:
                    producer_slots.setdefault(producer, []).append(slot)
            for slot, producer in slots.items():
                if producer is not None and slot not in producer_slots.get(
                    producer, []
                ):
                    producer_slots.setdefault(producer, []).append(slot)
            for producer, slot_list in producer_slots.items():
                if not producer or producer not in source_id:
                    continue
                if len(slot_list) == 1:
                    return slot_list[0]
                idx = int(source_port) if str(source_port).isdigit() else None
                if idx is not None and idx < len(slot_list):
                    return slot_list[idx]
        return None

    _inject_group_inputs(
        section_nodes,
        skip_namespaces=frozenset(inject_skip),
        resolve_slot_label=_return_slot_label,
    )
    def _resolve_slot_names_for_prefix(prefix: str) -> dict[str, list[str]] | None:
        """Map source attr_names → return slot names for multi-return children.

        A producer that supplies more than one return slot (e.g. a tuple-
        returning ``cos, sin = self.recomposition_frequencies(...)`` where both
        slots trace to the same op) maps to an *ordered* list of slots; the
        consumer disambiguates by the source output ordinal so the two boundary
        outputs get distinct ids/labels instead of colliding onto one.
        """
        attr = _tile_prefix_attr_name(prefix)
        child = next(
            (c for c in block_tree.children if c.attr_name == attr), None
        )
        if child is None or not child.forward_return_slots:
            return None
        order = child.forward_return_order or list(
            child.forward_return_slots.keys()
        )
        result: dict[str, list[str]] = {}
        for slot in order:
            producer = child.forward_return_slots.get(slot)
            if producer is None:
                continue
            result.setdefault(producer, []).append(slot)
        # Include any slots not covered by the recorded order (defensive).
        for slot, producer in child.forward_return_slots.items():
            if producer is not None and slot not in result.get(producer, []):
                result.setdefault(producer, []).append(slot)
        return result

    _inject_group_outputs(
        section_nodes,
        output_names=nested_output_names,
        resolve_name=lambda prefix: _caller_output_name(
            spec, block_tree.class_name, _tile_prefix_attr_name(prefix)
        ),
        resolve_slot_names=_resolve_slot_names_for_prefix,
    )
    _connect_external_inputs(
        section_nodes,
        namespace_prefix=namespace_prefix,
        previous_exits=previous_exits,
    )
    transparent_exit = None
    if is_transparent_inline_expansion(block_tree) and not namespace_prefix:
        input_id = _merge_node_id(id_prefix, "@input")
        output_id = _merge_node_id(id_prefix, "@output")
        input_node = next(
            (node for node in section_nodes if node["id"] == input_id),
            None,
        )
        output_node = next(
            (node for node in section_nodes if node["id"] == output_id),
            None,
        )
        if input_node is not None:
            incoming = list(input_node.get("incomingEdges", []))
            for node in section_nodes:
                for edge in node.get("incomingEdges", []):
                    if edge.get("sourceNodeId") != input_id or len(incoming) != 1:
                        continue
                    source = incoming[0]
                    edge["sourceNodeId"] = source["sourceNodeId"]
                    edge["sourceNodeOutputId"] = source.get("sourceNodeOutputId", "0")
            section_nodes.remove(input_node)
        if output_node is not None:
            incoming = list(output_node.get("incomingEdges", []))
            if len(incoming) == 1:
                edge = incoming[0]
                transparent_exit = (
                    str(edge["sourceNodeId"]),
                    str(edge.get("sourceNodeOutputId", "0")),
                )
                section_nodes.remove(output_node)
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
    inline_expansion: bool = True,
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
            inline_expansion=inline_expansion,
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
    inline_expansion: bool = True,
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
            inline_expansion=inline_expansion,
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
    inline_expansion: bool = True,
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
                decoder.forward_operations or decoder.forward_step_predecessors
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
                    inline_expansion=inline_expansion,
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
                    inline_expansion=inline_expansion,
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
                merged_nodes,
                context=shape_inferencer.context,
                boundary_spec=shape_inferencer.boundary_input_spec,
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
        inline_expansion=inline_expansion,
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
        fill_missing_node_shapes(
            merged_nodes,
            context=shape_inferencer.context,
            boundary_spec=shape_inferencer.boundary_input_spec,
        )
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


def _append_vision_section(
    nodes: list[dict[str, Any]],
    *,
    spec: ArchitectureSpec,
    basic_ops: BasicOpFilter,
    group_node_configs: list[dict[str, Any]],
    group_node_attributes: dict[str, dict[str, str]],
    shape_inferencer: ShapeInferencer | None,
    inline_expansion: bool,
) -> SourceRef | None:
    """Emit a VLM vision tower as an expandable group; return its output ref.

    Returns ``None`` for a text-only model (nothing is appended) or when the tower
    is too small to warrant its own subgraph, so the text-only path is untouched.
    """
    component = vision_tower_component(spec)
    if component is None or not component_has_detail_section(component, spec):
        return None

    # When the model exposes an image-placeholder token key, the pixel input and
    # the ``input_ids == image_token_id`` mask are both image-side model inputs;
    # group them so the mask renders next to the patches rather than beside its
    # distant ``masked_scatter`` consumer. Keyless VLMs keep the flat top-level
    # input and let the combine synthesize the mask.
    mask_token = image_placeholder_token_id(spec.raw_config)
    vision_input_node = {
        "id": "@vision_input",
        "label": "Image patches",
        "namespace": "",
        "attrs": [{"key": "synthetic", "value": "@input"}],
        "style": ensure_readable_text(input_port_style()),
    }
    # The image-patch boundary is the raw flat patch tensor ``[Pv, C*T*P*P]``,
    # not the tower's hidden width. Stamp it here so the boundary node (which is
    # synthesized in merge, outside the shape inferencer's own graph) carries the
    # true shape and the downstream wire-consistency check sees matching ends.
    patch_flat = getattr(shape_inferencer, "_vision_patch_flat", None)
    if patch_flat is not None:
        apply_shape_attrs(
            vision_input_node,
            TensorSpec(
                (Symbol.VISION_PATCH.value, patch_flat),
                shape_inferencer.context.dtype,
            ),
        )
    nodes.append(vision_input_node)
    if mask_token is not None:
        _ensure_image_mask_node(nodes, namespace="", token_id=mask_token)
    resolved = _resolve_section_tree_for_component(
        spec, component, variant=None, basic_ops=basic_ops
    )
    transparent = resolved is not None and is_transparent_inline_expansion(
        expand_block_tree_inplace(resolved[1], basic_ops=basic_ops)
    )
    namespace_prefix = (
        _sanitize_namespace_segment(component.attr_name) if not transparent else ""
    )
    group = _group_config_for_role(namespace_prefix, component.role)
    if group:
        group_node_configs.append(group)
    exits = _append_section(
        nodes,
        spec=spec,
        component=component,
        id_prefix=component.attr_name,
        namespace_prefix=namespace_prefix,
        basic_ops=basic_ops,
        previous_exits=["@vision_input"],
        group_node_attributes=group_node_attributes,
        shape_inferencer=shape_inferencer,
        parent_class=None,
        inline_expansion=inline_expansion,
    )
    return exits[0] if exits else None


def _attach_vision_language_combine(
    nodes: list[dict[str, Any]],
    *,
    vision_exit: SourceRef,
    text_exits: list[SourceRef],
    shape_inferencer: ShapeInferencer | None,
) -> list[SourceRef]:
    """Merge image-patch embeddings into the text token embeddings explicitly.

    A VLM wrapper doesn't just embed tokens — it ``masked_scatter``s the vision
    tower's patch embeddings into the positions of the image placeholder tokens
    (``inputs_embeds.masked_scatter(image_mask, image_embeds)``). Rendering the
    embedding tile as the merge point hides that computation and makes the
    embedding node misleadingly take two inputs. Instead, emit a dedicated
    combine node fed by (text embeddings, image embeddings); it becomes the new
    entry to the language stack. Returns the exits the decoder should consume.
    """
    if not text_exits:
        return text_exits
    text_ref = text_exits[0]
    # ``image_mask`` is the boolean placeholder-token selector derived from the
    # token ids (``input_ids == image_token_id``). It's a genuine control input
    # to the scatter, not a data tensor produced by either stack, so surface it
    # as a dedicated synthetic boundary. It is normally pre-created next to the
    # image-patch input (grouped there); this is a no-op then and only synthesizes
    # the boundary for a VLM that exposes no image-token key.
    mask_id = _ensure_image_mask_node(nodes, namespace="")
    combine_id = "@vision_language_combine"
    node: dict[str, Any] = {
        "id": combine_id,
        "label": "Masked scatter",
        "namespace": "",
        "attrs": [
            {
                "key": "detail",
                "value": (
                    "inputs_embeds.masked_scatter(image_mask, image_embeds) — "
                    "scatters vision patch embeddings into image placeholder positions"
                ),
            }
        ],
        "incomingEdges": [
            _source_edge(text_ref, "inputs_embeds"),
            _source_edge((mask_id, "0"), "image_mask"),
            _source_edge(vision_exit, "image_embeds"),
        ],
        "inputsMetadata": [
            {"id": "inputs_embeds", "attrs": [{"key": "port_label", "value": "inputs_embeds"}]},
            {"id": "image_mask", "attrs": [{"key": "port_label", "value": "image_mask"}]},
            {"id": "image_embeds", "attrs": [{"key": "port_label", "value": "image_embeds"}]},
        ],
    }
    if shape_inferencer is not None:
        context = shape_inferencer.context
        hidden = context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
        apply_shape_attrs(
            node,
            TensorSpec(
                (Symbol.BATCH.value, Symbol.SEQ.value, hidden), context.dtype
            ),
        )
    nodes.append(node)
    return [(combine_id, "0")]


def _rename_namespace_prefix(
    nodes: list[dict[str, Any]],
    group_node_attributes: dict[str, dict[str, str]],
    group_node_configs: list[dict[str, Any]],
    old: str,
    new: str,
) -> None:
    """Rewrite the namespace segment ``old`` -> ``new`` across nodes, group
    attributes, and config regexes. Only the ``namespace`` field is touched —
    node ids (and therefore edges) are left intact, exactly as the decoder group
    keeps ``decoder/...`` ids under the ``45x_...`` namespace."""
    for node in nodes:
        ns = node.get("namespace", "")
        if ns == old or ns.startswith(old + "/"):
            node["namespace"] = new + ns[len(old):]
    for key in list(group_node_attributes.keys()):
        if key == old or key.startswith(old + "/"):
            group_node_attributes[new + key[len(old):]] = group_node_attributes.pop(key)
    esc_old, esc_new = re.escape(old), re.escape(new)
    for config in group_node_configs:
        regex = config.get("namespaceRegex")
        if isinstance(regex, str) and esc_old in regex:
            config["namespaceRegex"] = regex.replace(esc_old, esc_new)


def _tag_secondary_module_groups(
    nodes: list[dict[str, Any]],
    *,
    spec: ArchitectureSpec,
    group_node_attributes: dict[str, dict[str, str]],
    group_node_configs: list[dict[str, Any]],
) -> None:
    """Tag secondary repeated ModuleLists (e.g. a VLM vision tower's block) as
    their own ``N×`` group, mirroring the decoder banner.

    The primary decoder ModuleList is already grouped via ``_decoder_namespace``.
    Every *other* live ``MetaModuleGroup`` (vision tower blocks, etc.) is rendered
    as a single inline body namespaced by its element class — a repeated block with
    no visible count. Here we rename that body's namespace segment ``Cls`` ->
    ``{length}x_Cls`` so the viewer shows the count exactly like the decoder, and
    add the matching ``repeat`` attribute + red group styling. Purely a
    namespace/attribute rewrite — no new nodes or edges, so it cannot create a
    cycle. Degrades to a no-op when the live tree is unavailable or no rendered
    namespace matches a secondary group's element class."""
    groups = getattr(spec, "meta_module_groups", None)
    if not groups:
        return
    decoder_seg = _sanitize_namespace_segment(spec.decoder_class or "")
    existing = {node.get("namespace", "") for node in nodes}
    for group in groups:
        if group.length < 2:
            continue
        seg = _sanitize_namespace_segment(group.element_class)
        if not seg or seg == decoder_seg:
            continue
        # The block body is the outermost rendered namespace whose final segment
        # is the element class (its children carry their own class segments).
        candidates = [ns for ns in existing if ns and ns.rsplit("/", 1)[-1] == seg]
        if not candidates:
            continue
        old_ns = min(candidates, key=len)
        parent, _, _ = old_ns.rpartition("/")
        new_seg = f"{group.length}x_{seg}"
        new_ns = f"{parent}/{new_seg}" if parent else new_seg
        if new_ns == old_ns or new_ns in existing:
            continue
        _rename_namespace_prefix(
            nodes, group_node_attributes, group_node_configs, old_ns, new_ns
        )
        group_node_attributes.setdefault(new_ns, {})["repeat"] = new_seg
        group_node_configs.append(
            {
                "namespaceRegex": f"^{re.escape(new_ns)}$",
                "backgroundColor": "#fff5f4",
                "borderColor": "#c0392b",
                "textColor": "#1a1a1a",
                "layoutDirection": "TOP_BOTTOM",
            }
        )
        existing = {node.get("namespace", "") for node in nodes}


_REPEAT_SEGMENT_RE = re.compile(r"^(\d+)x_")


def _fill_repeated_loop_counts(nodes: list[dict[str, Any]]) -> None:
    """Show the trip count on loop-carried boundaries of ModuleList loops.

    A ``for blk in self.blocks:`` loop has no static ``range(...)`` bound, so the
    AST analyzer leaves its ``@loop_carried`` boundary labeled ``<var> · repeated``.
    But the block body lives in an ``{N}x_Cls`` namespace whose count is authoritative
    (placed from the live meta tree — the decoder banner and ``_tag_secondary_module_groups``).
    Fill the count from the nearest enclosing ``{N}x_`` namespace segment so the
    boundary reads ``<var> · {N} iterations`` like the config-bounded inner loops.
    Cosmetic (sublabel only); no nodes or edges change, so acyclicity is untouched.
    """
    for node in nodes:
        if _node_attr(node, "synthetic") != "@loop_carried":
            continue
        sublabel_attr = next(
            (a for a in node.get("attrs", []) if a.get("key") == "sublabel"), None
        )
        if sublabel_attr is None or not sublabel_attr.get("value", "").endswith("· repeated"):
            continue
        # Nearest (deepest) enclosing repeat group owns this loop's trip count.
        count = next(
            (
                match.group(1)
                for segment in reversed(node.get("namespace", "").split("/"))
                if (match := _REPEAT_SEGMENT_RE.match(segment))
            ),
            None,
        )
        if count is None:
            continue
        variable = sublabel_attr["value"].rsplit(" · ", 1)[0]
        sublabel_attr["value"] = f"{variable} · {count} iterations"
        # Mirror the count into the "Loop in" label when the boundary was built
        # with a symbolic count (CG loops whose trip count is only known from the
        # enclosing ``{N}x_`` namespace, e.g. the vision block loop).
        if node.get("label") == "Loop in - iterations:N":
            node["label"] = f"Loop in - iterations:{count}"


def _repeat_group_container(namespace: str) -> str | None:
    """Outermost ``{N}x_`` repeat group enclosing ``namespace`` (or ``None``).

    Truncates at (and including) the FIRST ``{N}x_`` segment, so a nested variant
    like ``45x_DecoderLayer/31x_.../attn_hc`` maps to its OUTER loop container
    ``45x_DecoderLayer`` -- the three parallel variant branches belong to one loop.
    """
    segments = namespace.split("/")
    for index, segment in enumerate(segments):
        if _REPEAT_SEGMENT_RE.match(segment):
            return "/".join(segments[: index + 1])
    return None


def _copy_carried_shape(dst: dict[str, Any], src: dict[str, Any] | None) -> None:
    """Mirror a source node's output shape onto a synthesized carried tile.

    The pass runs after shape inference has settled, so a freshly built tile would
    otherwise render shapeless; copy the carried tensor's shape/dtype so the
    rerouted wires keep their shape label like the CG-built vision boundary.
    """
    if src is None:
        return
    shape = _node_attr(src, "output_shape")
    if shape is None:
        return
    dtype = _node_attr(src, "output_dtype")
    dst["attrs"].append({"key": "output_shape", "value": shape})
    port_attrs = [
        {"key": "shape", "value": shape},
        {"key": "tensor_shape", "value": shape},
    ]
    if dtype is not None:
        dst["attrs"].append({"key": "output_dtype", "value": dtype})
        port_attrs.append({"key": "dtype", "value": dtype})
    dst["outputsMetadata"] = [{"id": "0", "attrs": port_attrs}]


def _wrap_container_loop_carried(nodes: list[dict[str, Any]], container: str) -> None:
    """Wrap one ``{N}x_`` repeat group with a ``@loop_carried`` in/out boundary.

    No-op when the group already carries a container-level ``@loop_carried`` tile
    (the vision tower, built by the ComputationGraph) or when its boundary does not
    fit the single-carried-variable shape (multiple external inputs, no exit).
    """
    subtree_ids = {
        node["id"]
        for node in nodes
        if _namespace_is_descendant(node.get("namespace", ""), container)
    }
    if not subtree_ids:
        return
    # Already wrapped by the CG (vision): a loop-carried tile sits at this exact
    # namespace level. Inner-loop tiles live deeper, so they do not count.
    for node in nodes:
        if (
            node.get("namespace") == container
            and _node_attr(node, "synthetic") == "@loop_carried"
            and "@loop_carried_in:" in node.get("id", "")
        ):
            return

    # Entry edges: external source -> a synthetic @input tile inside the group.
    entries: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for node in nodes:
        if node["id"] not in subtree_ids or not _is_synthetic_input(node):
            continue
        for edge in node.get("incomingEdges", []):
            source = edge.get("sourceNodeId")
            if source is not None and source not in subtree_ids:
                entries.setdefault(source, []).append((node, edge))
    # A single external source is the loop-carried variable. Multiple sources mean
    # loop-invariant inputs to hoist as well (as vision's @input:cos/sin) -- no
    # current structural-spine group has those, so stay conservative and skip.
    if len(entries) != 1:
        return
    carried_source, carried_targets = next(iter(entries.items()))
    variable = carried_targets[0][0].get("label") or "hidden_states"

    # Exit edges: an interior node -> an external consumer. The distinct interior
    # sources are the loop's updated value -- one @output for a single-template
    # group, or the parallel variant branches' @outputs (all the pre-collapse
    # carried tensor, before a post-loop head like ``hc_head`` collapses it).
    exit_edges = [
        (node, edge)
        for node in nodes
        if node["id"] not in subtree_ids
        for edge in node.get("incomingEdges", [])
        if edge.get("sourceNodeId") in subtree_ids
    ]
    if not exit_edges:
        return
    exit_sources: list[str] = []
    for _consumer, edge in exit_edges:
        source = edge["sourceNodeId"]
        if source not in exit_sources:
            exit_sources.append(source)

    # Heterogeneous loop: the group instantiates *different* modules by iteration
    # (the decoder's 31/11/3 variant runs), so the collapsed body is really a
    # sequence of distinct variant blocks feeding a post-loop head -- multiple
    # distinct interior exit sources. A single loop-carried abstraction
    # misrepresents that, so synthesize nothing: return before creating or wiring
    # any @loop_carried tile, leaving the body @input edges (already sourced from
    # the external producer) and @output edges (already targeting the external
    # consumers) exactly as the earlier passes built them -- i.e. direct
    # producer->submodule and submodule->consumer wiring, no back edge. Uniform
    # loops (a single exit source) keep their loop-carried boundary.
    if len(exit_sources) > 1:
        return

    node_by_id = {node["id"]: node for node in nodes}
    id_prefix = carried_targets[0][0]["id"].split("/", 1)[0]
    in_id = f"{id_prefix}/@loop_carried_in:{id_prefix}:{variable}"
    out_id = f"{id_prefix}/@loop_carried_out:{id_prefix}:{variable}"
    style = ensure_readable_text(detail_tile_style(None, synthetic="@loop_carried"))

    def _make_tile(tile_id: str, label: str) -> dict[str, Any]:
        return {
            "id": tile_id,
            "label": label,
            "namespace": container,
            "attrs": [
                {"key": "sublabel", "value": f"{variable} · repeated"},
                {"key": "synthetic", "value": "@loop_carried"},
                {"key": "operation", "value": "synthetic"},
            ],
            "style": style,
        }

    # Trip count for the "Loop in" label comes from the enclosing ``{N}x_`` repeat
    # segment; when no static count is present, use a symbolic ``N`` so the loop
    # still reads as bounded by an iteration variable.
    count_token = next(
        (
            match.group(1)
            for segment in container.split("/")
            if (match := _REPEAT_SEGMENT_RE.match(segment))
        ),
        "N",
    )
    in_node = _make_tile(in_id, f"Loop in - iterations:{count_token}")
    out_node = _make_tile(out_id, "Loop out")
    _copy_carried_shape(in_node, node_by_id.get(carried_source))
    _copy_carried_shape(out_node, node_by_id.get(carried_source))

    # Carried-in: initial value from the external source + back edge from carried-out.
    src_port = carried_targets[0][1].get("sourceNodeOutputId", "0")
    in_node["incomingEdges"] = [
        {
            "sourceNodeId": out_id,
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "1",
            "metadata": {"port_label": "next iteration"},
        },
        {
            "sourceNodeId": carried_source,
            "sourceNodeOutputId": src_port,
            "targetNodeInputId": "1",
        },
    ]
    for _tile, edge in carried_targets:
        edge["sourceNodeId"] = in_id
        edge["sourceNodeOutputId"] = "0"

    # Carried-out: fed by the interior exit sources (the loop's updated value);
    # their external consumers now read carried-out so the updated value flows out
    # through the boundary, then on to any post-loop head.
    out_node["incomingEdges"] = [
        {
            "sourceNodeId": source,
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "0",
            "metadata": {"port_label": "updated"},
        }
        for source in exit_sources
    ]
    for _consumer, edge in exit_edges:
        edge["sourceNodeId"] = out_id
        edge["sourceNodeOutputId"] = "0"
    # Several exit sources feeding one consumer (a hyper-head merging the parallel
    # variant branches) now all read carried-out -- the single merged loop value --
    # so collapse them to one edge instead of N identical slots.
    for consumer_id in {consumer["id"] for consumer, _edge in exit_edges}:
        consumer = node_by_id[consumer_id]
        deduped: list[dict[str, Any]] = []
        seen_carried_out = False
        for edge in consumer.get("incomingEdges", []):
            if edge.get("sourceNodeId") == out_id:
                if seen_carried_out:
                    continue
                seen_carried_out = True
            deduped.append(edge)
        consumer["incomingEdges"] = deduped

    nodes.append(in_node)
    nodes.append(out_node)


def _is_loop_back_edge(source_id: str | None, target_id: str) -> bool:
    """The single permitted cycle-creating edge: ``@loop_carried_out -> _in``.

    This is the only edge a topological sort must ignore; every other edge is a
    genuine dataflow dependency. Mirrors the acyclicity check used in the tests.
    """
    return (
        "@loop_carried_in:" in target_id
        and source_id is not None
        and "@loop_carried_out:" in source_id
    )


# Top-level model-input boundaries, in the order they appear in the model's
# forward signature (``input_ids`` precedes ``pixel_values``); the derived
# image-placeholder mask renders immediately after the pixel input so the two
# image-side inputs stay adjacent. Any other top-level ``@input`` keeps its
# relative order after these.
_MODEL_INPUT_ORDER = ("@input", "@vision_input", _IMAGE_MASK_ID)


def _order_model_inputs(nodes: list[dict[str, Any]]) -> None:
    """Move the top-level model-input boundaries to the front in forward order.

    They are graph sources (no incoming edges), so hoisting them ahead of the
    body is topology-safe; making them contiguous keeps the two image-side inputs
    adjacent (the reason they no longer need a shared box). The following stable
    :func:`_topologically_order_nodes` preserves this relative order.
    """
    priority = {nid: i for i, nid in enumerate(_MODEL_INPUT_ORDER)}
    model_inputs = [
        node
        for node in nodes
        if not node.get("namespace") and _is_synthetic_input(node)
    ]
    if len(model_inputs) < 2:
        return
    hoisted = {id(node) for node in model_inputs}
    index_of = {id(node): i for i, node in enumerate(nodes)}
    ordered = sorted(
        model_inputs,
        key=lambda node: (
            priority.get(node["id"], len(priority)),
            index_of[id(node)],
        ),
    )
    rest = [node for node in nodes if id(node) not in hoisted]
    nodes[:] = ordered + rest


def _topologically_order_nodes(nodes: list[dict[str, Any]]) -> None:
    """Reorder ``nodes`` producer-before-consumer with a stable Kahn sort.

    Every dataflow edge (stored as a consumer's ``incomingEdges``) except the one
    permitted ``@loop_carried_out -> @loop_carried_in`` back edge per loop becomes
    an ordering constraint, so each ``@loop_carried_in`` sorts ahead of its loop
    body and each ``@loop_carried_out`` after it. Ties break by original list
    index, preserving within-namespace sibling order and byte-determinism (Model
    Explorer groups by ``namespace``; list order only sets sibling order). Any
    residual left by an unexpected real cycle is appended in original order rather
    than dropped — acyclicity itself is asserted by the graph tests.
    """
    import heapq

    index_of = {node["id"]: i for i, node in enumerate(nodes)}
    count = len(nodes)
    adjacency: list[list[int]] = [[] for _ in range(count)]
    indegree = [0] * count
    for target_i, node in enumerate(nodes):
        target_id = node["id"]
        seen: set[int] = set()
        for edge in node.get("incomingEdges", []):
            source_id = edge.get("sourceNodeId")
            source_i = index_of.get(source_id)
            if source_i is None or source_i in seen:
                continue
            if _is_loop_back_edge(source_id, target_id):
                continue
            seen.add(source_i)
            adjacency[source_i].append(target_i)
            indegree[target_i] += 1

    ready = [i for i in range(count) if indegree[i] == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        current = heapq.heappop(ready)
        order.append(current)
        for nxt in adjacency[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(ready, nxt)

    if len(order) != count:
        placed = set(order)
        order.extend(i for i in range(count) if i not in placed)
    nodes[:] = [nodes[i] for i in order]


def _hoist_loop_carried_in_ahead_of_body(nodes: list[dict[str, Any]]) -> None:
    """Pull each ``@loop_carried_in`` ahead of its loop-body namespace subtree.

    Model Explorer groups by the ``namespace`` field and renders sibling nodes /
    subgroups in list order, so a loop's ``@loop_carried_in`` should be the first
    element under its loop scope. The stable topological sort only guarantees a
    ``@loop_carried_in`` precedes its *direct* dataflow consumers; a loop body's
    graph sources (hidden-constant closures, per-branch inputs) carry no edge from
    the loop input, so the sort can legitimately place them ahead of it -- e.g. a
    repeated decoder block whose per-layer RMSNorm-eps / hyper-connection constants
    sorted to the very front, pushing the whole layer body above the loop-in tile.

    Every loop scope (a ``{N}x_`` repeat group and an inner ``for`` loop alike) is
    captured by a dedicated ``namespace`` subtree, so the body of the loop owning a
    ``@loop_carried_in`` is exactly the nodes whose namespace equals or descends
    from the boundary's namespace. Move the boundary just ahead of the earliest
    such node. This is topology-safe: a ``@loop_carried_in`` is a graph source (its
    only real dependency is the permitted back edge, which is excluded), so nothing
    it must follow gets pushed after it -- but we still clamp past any genuine seed
    producer it *does* depend on (an ``@input:<name>`` initial value living inside
    the loop scope), keeping producer-before-consumer intact.
    """

    def _namespace(node: dict[str, Any]) -> str:
        return node.get("namespace") or ""

    # Outermost loops first (shallowest namespace) so an enclosing boundary lands
    # ahead of a nested one sharing the subtree.
    carried_ins = sorted(
        (node for node in nodes if "@loop_carried_in:" in str(node.get("id"))),
        key=lambda node: _namespace(node).count("/"),
    )
    for boundary in carried_ins:
        cur = nodes.index(boundary)
        ns = _namespace(boundary)
        # Earliest node in the boundary's loop-body namespace subtree.
        earliest = None
        for i, node in enumerate(nodes):
            if node is boundary:
                continue
            node_ns = _namespace(node)
            if node_ns == ns or (ns and node_ns.startswith(ns + "/")):
                earliest = i
                break
        if earliest is None or earliest >= cur:
            continue
        # Never move ahead of a genuine (non-back-edge) producer of the boundary.
        producer_ids = {
            str(edge.get("sourceNodeId"))
            for edge in boundary.get("incomingEdges", []) or []
            if not _is_loop_back_edge(
                str(edge.get("sourceNodeId") or ""), str(boundary.get("id"))
            )
        }
        floor = 0
        for i in range(cur):
            if str(nodes[i].get("id")) in producer_ids:
                floor = i + 1
        target = max(earliest, floor)
        if target < cur:
            nodes.insert(target, nodes.pop(cur))


def _stack_primary_input_name(cls: Any) -> str | None:
    """First non-self ``forward`` parameter of a class (its primary input)."""
    forward = next(
        (
            item
            for item in cls.node.body
            if isinstance(item, ast.FunctionDef) and item.name == "forward"
        ),
        None,
    )
    if forward is None:
        return None
    params = [
        arg.arg
        for arg in forward.args.posonlyargs + forward.args.args
        if arg.arg != "self"
    ]
    return params[0] if params else None


def _resolve_existing_producer_node(
    node_by_id: dict[str, dict[str, Any]], attr: str
) -> str | None:
    """Node id already materialised for a forward-step producer attr, if any."""
    if attr in node_by_id:
        return attr
    candidate = f"@model_forward/{attr}"
    if candidate in node_by_id:
        return candidate
    return None


def _loop_invariant_producer_label(cls: Any, producer_attr: str) -> str:
    """Readable label for a materialised model-scope loop-invariant producer."""
    base = base_submodule_attr(producer_attr)
    class_name = cls.init_assignments.get(base)
    if class_name:
        return class_name
    if base.startswith("@fn_") or base.startswith("@positional_"):
        # ``@fn_l1303_create_sliding_window_causal_mask`` -> the callee name.
        tail = base.split("_", 2)[-1] if base.count("_") >= 2 else base
        return tail
    return base


def _materialize_model_scope_producer(
    nodes: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    *,
    cls: Any,
    producer_attr: str,
) -> str | None:
    """Emit (or reuse) a model-scope source node for a loop-invariant producer.

    The node is wired from whichever of the producer's own stack-entry
    predecessors are already materialised. Returns ``None`` when none are, so the
    caller leaves the boundary unsourced rather than inventing a rootless node.
    """
    existing = _resolve_existing_producer_node(node_by_id, producer_attr)
    if existing is not None:
        return existing
    node_id = f"@model_forward/{producer_attr}"
    incoming: list[dict[str, str]] = []
    for pred in cls.forward_step_predecessors.get(producer_attr, ()):  # type: ignore[attr-defined]
        source = _resolve_existing_producer_node(node_by_id, pred)
        if source is not None:
            incoming.append(
                {
                    "sourceNodeId": source,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": str(len(incoming)),
                }
            )
    if not incoming:
        return None
    node = {
        "id": node_id,
        "label": _loop_invariant_producer_label(cls, producer_attr),
        "namespace": "",
        "attrs": [{"key": "operation", "value": "source"}],
        "incomingEdges": incoming,
    }
    nodes.append(node)
    node_by_id[node_id] = node
    return node_id


def _ensure_top_level_input(
    nodes: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    param: str,
) -> str:
    """Node id of a top-level model-input boundary for ``param`` (create if new)."""
    node_id = f"@input:{param}"
    if node_id not in node_by_id:
        node = {
            "id": node_id,
            "label": param,
            "namespace": "",
            "attrs": [{"key": "synthetic", "value": "@input"}],
            "style": ensure_readable_text(input_port_style()),
        }
        nodes.append(node)
        node_by_id[node_id] = node
    return node_id


def _forward_param_producer_map(cls: Any) -> dict[str, str]:
    """Map a stack-model forward local ``name`` -> submodule attr producing it.

    Introspects assignments of the form ``name = self.<attr>(...)`` where ``attr``
    names a known submodule, e.g. ``position_embeddings = self.rotary_emb(...)``.
    This lets the loop-invariant resolver recover a value's producer structurally
    even when the extractor left ``forward_step_predecessor_args`` empty (a variant
    decoder loop the AST recovery could not map to keyword producers).
    """
    producers: dict[str, str] = {}
    forward = next(
        (
            item
            for item in cls.node.body
            if isinstance(item, ast.FunctionDef) and item.name == "forward"
        ),
        None,
    )
    if forward is None:
        return producers
    for stmt in ast.walk(forward):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        call = stmt.value
        if not isinstance(target, ast.Name) or not isinstance(call, ast.Call):
            continue
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
            and func.attr in cls.init_assignments
        ):
            producers.setdefault(target.id, func.attr)
    return producers


def _resolve_submodule_output_node(
    node_by_id: dict[str, dict[str, Any]], attr: str
) -> str | None:
    """Existing model-scope output node id for a submodule ``attr`` producer."""
    if attr in node_by_id:
        return attr
    prefix = f"{attr}/"
    outputs = sorted(
        nid
        for nid in node_by_id
        if nid.startswith(prefix) and "/@output" in nid
    )
    return outputs[0] if outputs else None


def _resolve_loop_invariant_source(
    nodes: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    *,
    cls: Any,
    param: str,
    primary_param: str | None,
    pred_args: dict[str, str],
    producer_map: dict[str, str],
) -> str | None:
    """Legitimate model-level source id for a loop-invariant decoder input.

    Resolution keys only on structural facts, in priority order:
    1. The extractor recovered the loop call's exact keyword producer
       (``forward_step_predecessor_args``) -- honour it (submodule/free-fn call
       materialised, top-level parameter docked, primary spine input as ``@input``).
    2. Otherwise (variant loops that leave that map empty) resolve by the input's
       own name: the primary spine input, a top-level forward parameter, or a value
       assigned from a ``self.<submodule>(...)`` call in the stack-model forward.
    """
    if pred_args and param in pred_args:
        producer_attr = pred_args[param]
        if producer_attr == FORWARD_METHOD_INPUT or param == primary_param:
            return "@input"
        if param in cls.forward_param_inputs:
            # A top-level model forward parameter (``attention_mask``): dock onto
            # its own model-input boundary. The host-side construction feeding it
            # is a CPU helper the export collapses, so the parameter is the root.
            return _ensure_top_level_input(nodes, node_by_id, param)
        return _materialize_model_scope_producer(
            nodes, node_by_id, cls=cls, producer_attr=producer_attr
        )
    if param == primary_param:
        return "@input"
    if param in cls.forward_param_inputs:
        return _ensure_top_level_input(nodes, node_by_id, param)
    attr = producer_map.get(param)
    if attr is not None:
        return _resolve_submodule_output_node(node_by_id, attr)
    return None


def _thread_loop_invariant_inputs(
    nodes: list[dict[str, Any]],
    *,
    spec: ArchitectureSpec,
) -> None:
    """Source a repeat group's loop-invariant ``@input:<param>`` boundaries.

    A decoder layer reads tensors handed to every iteration by keyword
    (``layer(hidden_states, position_embeddings=..., attention_mask=...)``). The
    collapsed repeat group surfaces each as an ``@input:<param>`` tile deep inside
    the body, but -- unlike the primary spine input, threaded by the loop-carried
    boundary -- nothing sources them, so they float. This pass reconnects each to
    its legitimate model-level producer, read structurally from the stack model's
    decoder-loop call arguments (``forward_step_predecessor_args``): a producer
    that is a submodule / free-function call is materialised (or reused) as a
    model-scope source node; a producer that is a top-level forward parameter
    docks onto that model-input boundary.

    Keyed only on structural facts (a floating namespaced ``@input:<param>`` whose
    name is a decoder-loop argument), never on a specific parameter or class name.
    """
    cls = spec.class_registry.get(spec.stack_model_class or "")
    if cls is None:
        cls = _pick_stack_model_class(spec.class_registry, None)
    if cls is None:
        return
    loop_attr = next(
        (
            attr
            for attr, name in cls.init_assignments.items()
            if name == spec.decoder_class
        ),
        None,
    )
    pred_args = (
        cls.forward_step_predecessor_args.get(loop_attr, {}) if loop_attr else {}
    ) or {}
    primary_param = _stack_primary_input_name(cls)
    producer_map = _forward_param_producer_map(cls)
    node_by_id = {node["id"]: node for node in nodes}

    # (1) Floating (unsourced) namespaced ``@input:<param>`` tiles, grouped by
    # (repeat-group container, id prefix, param). ``container`` is ``None`` for a
    # boundary outside any repeat group (a model-scope submodule's own input).
    pending: dict[tuple[str | None, str, str], list[dict[str, Any]]] = {}
    for node in nodes:
        match = re.search(r"/@input:([^/^]+)$", node["id"])
        if match is None or node.get("incomingEdges"):
            continue
        container = _repeat_group_container(node.get("namespace", ""))
        param = match.group(1)
        prefix = node["id"].split("/", 1)[0]
        pending.setdefault((container, prefix, param), []).append(node)

    for (container, prefix, param), tiles in pending.items():
        source = _resolve_loop_invariant_source(
            nodes,
            node_by_id,
            cls=cls,
            param=param,
            primary_param=primary_param,
            pred_args=pred_args,
            producer_map=producer_map,
        )
        if source is None:
            continue
        if container is None:
            # A submodule's own input boundary (``rotary_emb/@input:position_ids``):
            # wire it straight to the model-level source, no group boundary.
            for tile in tiles:
                tile["incomingEdges"] = [
                    {
                        "sourceNodeId": source,
                        "sourceNodeOutputId": "0",
                        "targetNodeInputId": "0",
                    }
                ]
            continue
        boundary_id = f"{prefix}/@input:{param}"
        boundary = node_by_id.get(boundary_id)
        if boundary is None:
            boundary = {
                "id": boundary_id,
                "label": param,
                "namespace": container,
                "attrs": [{"key": "synthetic", "value": "@input"}],
                "style": ensure_readable_text(input_port_style()),
            }
            nodes.append(boundary)
            node_by_id[boundary_id] = boundary
        boundary["incomingEdges"] = [
            {
                "sourceNodeId": source,
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "0",
            }
        ]
        for tile in tiles:
            if tile["id"] == boundary_id:
                continue
            tile["incomingEdges"] = [
                {
                    "sourceNodeId": boundary_id,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            ]

    # (2) Floating bare ``.../@input`` tiles (the primary spine input mirrored deep
    # into a nested submodule, e.g. an attention indexer): mirror each from the
    # nearest enclosing already-sourced ``@input`` boundary carrying the same name.
    for node in nodes:
        nid = node["id"]
        if node.get("incomingEdges") or not nid.endswith("/@input"):
            continue
        label = node.get("label")
        ancestor = nid[: -len("/@input")]
        while "/" in ancestor:
            ancestor = ancestor.rsplit("/", 1)[0]
            candidate = f"{ancestor}/@input"
            enclosing = node_by_id.get(candidate)
            if (
                enclosing is not None
                and enclosing.get("incomingEdges")
                and enclosing.get("label") == label
            ):
                node["incomingEdges"] = [
                    {
                        "sourceNodeId": candidate,
                        "sourceNodeOutputId": "0",
                        "targetNodeInputId": "0",
                    }
                ]
                break


def _prune_dead_stack_entry_sources(nodes: list[dict[str, Any]]) -> None:
    """Drop model-scope ``@model_forward`` source ops that nothing consumes.

    The stack-entry dataflow materialises the producers feeding the decoder loop
    by keyword (``position_ids``), but a given model may route that value only
    through a submodule the collapsed body never re-exposes (a decoder variant's
    internal rotary). The op then has no consumer. It is a speculative
    intermediate, not a missing-edge dead node, so remove it (and any predecessor
    left consumer-less in turn) rather than surface an I1 warning. Confined to the
    ``@model_forward`` synthetic scope so real module leaves are never touched.
    """
    while True:
        consumed = {
            edge.get("sourceNodeId")
            for node in nodes
            for edge in node.get("incomingEdges", []) or []
        }
        dead = [
            node
            for node in nodes
            if node.get("namespace", "") == ""
            and str(node.get("id", "")).startswith("@model_forward/")
            and node["id"] not in consumed
        ]
        if not dead:
            return
        dead_ids = {node["id"] for node in dead}
        nodes[:] = [node for node in nodes if node["id"] not in dead_ids]


def _synthesize_repeat_loop_boundaries(nodes: list[dict[str, Any]]) -> None:
    """Give every ``{N}x_`` repeat group a loop-carried boundary like the vision tower.

    The vision block loop is computation-graphed, so ``_add_loop_carried_nodes``
    already produced its ``@loop_carried_in/out`` tiles -- this pass finds that
    boundary present and skips it. The decoder's outer ``for layer in self.layers``
    loop is only rendered structurally (``@input``/``@output`` tiles), so this pass
    synthesizes the matching boundary + single back edge for it. One uniform pass
    over both groups: the same code handles the vision tower and the main spine.
    """
    seen: set[str] = set()
    containers: list[str] = []
    for node in nodes:
        container = _repeat_group_container(node.get("namespace", ""))
        if container is not None and container not in seen:
            seen.add(container)
            containers.append(container)
    for container in containers:
        _wrap_container_loop_carried(nodes, container)


def build_merged_model_graph(
    spec: ArchitectureSpec,
    *,
    basic_ops: BasicOpFilter | None = None,
    graph_id: str = "model",
    shape_inferencer: ShapeInferencer | None = None,
    inline_expansion: bool = True,
) -> dict[str, Any]:
    """Build a single graph with overview spine and inlined computation subgraphs."""
    resolved_basic_ops = basic_ops or spec.basic_ops
    nodes: list[dict[str, Any]] = []
    group_node_configs: list[dict[str, Any]] = []
    group_node_attributes: dict[str, dict[str, str]] = {}

    # A VLM wrapper's vision tower renders as its own expandable group before the
    # tokenized-text input; its output later feeds the language embedding tile.
    vision_exit = _append_vision_section(
        nodes,
        spec=spec,
        basic_ops=resolved_basic_ops,
        group_node_configs=group_node_configs,
        group_node_attributes=group_node_attributes,
        shape_inferencer=shape_inferencer,
        inline_expansion=inline_expansion,
    )

    nodes.append(
        {
            "id": "@input",
            "label": "Tokenized text",
            "namespace": "",
            "attrs": [{"key": "synthetic", "value": "@input"}],
            "style": ensure_readable_text(input_port_style()),
        }
    )
    previous_exits = ["@input"]
    stack_module_sources: dict[str, SourceRef] = {}
    embedding_target_attr: str | None = None
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
        resolved = _resolve_section_tree_for_component(
            spec,
            component,
            variant=None,
            basic_ops=resolved_basic_ops,
        )
        transparent = resolved is not None and is_transparent_inline_expansion(
            expand_block_tree_inplace(resolved[1], basic_ops=resolved_basic_ops)
        )
        namespace_prefix = (
            _sanitize_namespace_segment(component.attr_name)
            if expands and not transparent
            else ""
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
            inline_expansion=inline_expansion,
        )
        if embedding_target_attr is None and component.role == "embedding":
            embedding_target_attr = component.attr_name
        if previous_exits:
            stack_module_sources[component.attr_name] = previous_exits[0]

    if vision_exit is not None:
        previous_exits = _attach_vision_language_combine(
            nodes,
            vision_exit=vision_exit,
            text_exits=previous_exits,
            shape_inferencer=shape_inferencer,
        )
        # Route the language stack's embedding consumers through the combine so
        # the merged embeddings — not the raw token embeddings — flow downstream.
        if embedding_target_attr is not None and previous_exits:
            stack_module_sources[embedding_target_attr] = previous_exits[0]

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
        inline_expansion=inline_expansion,
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
            inline_expansion=inline_expansion,
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
    _mirror_boundary_inputs(nodes)
    _mirror_boundary_outputs(nodes)
    _collapse_mirror_boundary_passthroughs(nodes)
    _collapse_kernel_input_passthroughs(nodes)
    _collapse_same_name_boundary_passthroughs(nodes)

    if shape_inferencer is not None:
        fill_missing_node_shapes(
            nodes,
            context=shape_inferencer.context,
            boundary_spec=shape_inferencer.boundary_input_spec,
        )
        _stamp_boundary_input_shapes(nodes, shape_inferencer)
        _reconcile_edge_endpoint_shapes(nodes)
        _assert_edge_endpoint_shapes_agree(nodes)

    # Elide no-op (same-dtype) casts and keep dtype-changing ones. Runs AFTER
    # the shape-settle block so the dynamic ``.to(x.dtype)`` casts have their
    # dtype attrs populated and resolve as bfloat16->bfloat16 no-ops.
    _prune_noop_cast_nodes(nodes)

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

    # Surface each split/unbind slice (qkv unbind, q/k/v split) as its own named,
    # input-styled tile. Model Explorer does not render output-port names, so the
    # split node stays visible and each slice gets a readable name + its own
    # shape. Runs after shapes settle so the tiles carry final per-slice shapes.
    _add_split_slice_tiles(nodes)

    # Drop no-op single-input Concats (in==out): the vision sdpa fallback's
    # per-chunk reassembly cat collapses to one representative kernel output, so it
    # concatenates nothing -- the windowing is already carried by the kernel's
    # cu_seqlens. Runs after shapes settle so the identity check sees final dims.
    _elide_noop_single_input_concat(nodes)

    # Describe every operation node's operands PyTorch-profiler-style (op_type +
    # input_shapes/input_types/concrete_inputs, including scalar args that are not
    # graph edges), then type-check the ops whose operand contract is known.
    # Warnings only -- an offender is a wiring/extraction fidelity bug to fix
    # upstream, not a reason to fail the export. Runs after edges/shapes are final.
    _annotate_op_input_signatures(nodes)
    type_check_graph_nodes(nodes)

    finalize_graph_node_styles(nodes)

    # Tag secondary repeated ModuleLists (e.g. the vision tower block) as their
    # own N× groups from the live meta tree, mirroring the decoder banner. Runs
    # after styles settle and before group attributes/configs are assembled so
    # the renamed namespaces flow into both.
    _tag_secondary_module_groups(
        nodes,
        spec=spec,
        group_node_attributes=group_node_attributes,
        group_node_configs=group_node_configs,
    )

    # Give every ``{N}x_`` repeat group the same loop-carried boundary the vision
    # tower gets from the ComputationGraph. No-op on groups already wrapped (vision);
    # synthesizes the boundary for the structural decoder spine. Runs after the
    # ``{N}x_`` namespaces are final so the no-op guard and container detection see
    # them, and before the count fill below stamps ``· N iterations``.
    _synthesize_repeat_loop_boundaries(nodes)

    # Reconnect the repeat group's loop-invariant ``@input:<param>`` tiles
    # (``position_embeddings``/``attention_mask``, forwarded to every iteration by
    # keyword) to their model-level producers. Runs after the loop-carried spine is
    # synthesized so the extra sourced boundaries do not perturb its single-entry
    # detection, and before integrity checks so the reconnected tiles read as sourced.
    _thread_loop_invariant_inputs(nodes, spec=spec)

    # A loop-invariant producer feeding no re-exposed consumer (a decoder variant's
    # internal-only rotary) leaves its stack-entry prep ops orphaned; drop them.
    _prune_dead_stack_entry_sources(nodes)

    # Fill trip counts on ModuleList loop-carried boundaries (``<var> · repeated``
    # -> ``<var> · N iterations``) from the ``{N}x_`` namespaces just finalized.
    _fill_repeated_loop_counts(nodes)

    # Hoist the model-input boundaries to the front in forward-signature order
    # (keeping the two image-side inputs adjacent), then emit nodes
    # producer-before-consumer so each ``@loop_carried_in`` renders ahead of its
    # loop body and each ``@loop_carried_out`` after it.
    _order_model_inputs(nodes)
    _topologically_order_nodes(nodes)
    _hoist_loop_carried_in_ahead_of_body(nodes)

    # Structural-integrity check on the FINAL built graph (after loop-carried
    # synthesis + ordering): I1 dead-node / I2 no-source / I3 constant soundness.
    # Warnings only -- an offender is a wiring/extraction fidelity bug to fix
    # upstream. The render-filtered graph is checked separately in the viewer.
    integrity_check_graph_nodes(nodes, label="built")

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
