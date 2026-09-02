###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Convert TraceLens ``ComputationGraph`` objects to Model Explorer input graphs."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from visualizer.computation_graph import ComputationGraph
from visualizer.model_graph import classify_operation

from model_explorer_export.styles import (
    detail_tile_style,
    ensure_readable_text,
    finalize_graph_node_styles,
)

_DEFAULT_OUTPUT_ID = "0"


def _sanitize_namespace_segment(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
    return cleaned or "group"


def _duplicate_frame_labels(computation: ComputationGraph) -> set[str]:
    """Labels used by more than one inline frame in this graph.

    Two sibling modules of the same class (``q_a_layernorm`` and
    ``kv_a_layernorm``) both frame as that class. Keying the namespace on the
    class then merges their internals into one group, which reads as a loop.
    """
    counts: dict[str, int] = defaultdict(int)
    for frame in computation.inline_frames:
        counts[frame.label] += 1
    return {label for label, count in counts.items() if count > 1}


def _frame_namespace_segment(frame, *, duplicate_labels: set[str]) -> str:
    if frame.label in duplicate_labels:
        return _sanitize_namespace_segment(frame.frame_id)
    return _sanitize_namespace_segment(frame.label)


def _node_namespaces(computation: ComputationGraph) -> dict[int, str]:
    namespaces: dict[int, str] = {index: "" for index in range(len(computation.nodes))}
    duplicate_labels = _duplicate_frame_labels(computation)
    # Outer frames contain their nested frames' nodes. Apply larger frames first
    # so a helper called inside a loop becomes `Loop/_apply_gate`, not the
    # inverted `_apply_gate/Loop` hierarchy.
    for frame in sorted(
        computation.inline_frames,
        key=lambda item: -len(set(item.node_indices)),
    ):
        if frame.transparent:
            continue
        segment = _frame_namespace_segment(frame, duplicate_labels=duplicate_labels)
        for index in frame.node_indices:
            if index not in namespaces:
                continue
            parent = namespaces[index]
            namespaces[index] = f"{parent}/{segment}" if parent else segment
    return namespaces


def _kv(key: str, value: str) -> dict[str, str]:
    return {"key": key, "value": value}


def _node_attrs(spec) -> list[dict[str, str]]:
    attrs: list[dict[str, str]] = []
    block = spec.block
    if block is not None and block.attr_name:
        attrs.append(_kv("attr_name", block.attr_name))
    if block is not None and block.class_name:
        attrs.append(_kv("class_name", block.class_name))
    if block is not None and block.role:
        attrs.append(_kv("role", block.role))
    if block is not None and block.boundary_input_name:
        attrs.append(_kv("boundary_input", block.boundary_input_name))
    if spec.sublabel:
        attrs.append(_kv("sublabel", spec.sublabel))
    if spec.port_label:
        attrs.append(_kv("port_label", spec.port_label))
    if spec.port_style:
        attrs.append(_kv("port_style", spec.port_style))
    if spec.synthetic:
        attrs.append(_kv("synthetic", spec.synthetic))
    operation = classify_operation(block, synthetic=spec.synthetic, label=spec.label)
    if operation is not None:
        attrs.append(_kv("operation", operation.value))
    if block is not None and block.details:
        attrs.append(_kv("details", "; ".join(block.details)))
    return attrs


def _node_style(spec) -> dict[str, str] | None:
    return ensure_readable_text(
        detail_tile_style(
            spec.block,
            synthetic=spec.synthetic,
            label=spec.label or "",
        )
    )


def _incoming_edges(
    computation: ComputationGraph, index_to_id: dict[int, str]
) -> dict[str, list[dict[str, Any]]]:
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_input_counter: dict[str, int] = defaultdict(int)

    for source_index, target_index in computation.links:
        source_id = index_to_id.get(source_index)
        target_id = index_to_id.get(target_index)
        if source_id is None or target_id is None:
            continue
        if target_index == computation.output_node_index:
            continue

        metadata: dict[str, str] = {}
        port_label = computation.link_port_labels.get((source_index, target_index))
        if port_label:
            metadata["port_label"] = port_label

        if target_index == computation.output_node_index and port_label:
            target_input_id = port_label
        else:
            target_input_id = str(target_input_counter[target_id])
            target_input_counter[target_id] += 1

        edge: dict[str, Any] = {
            "sourceNodeId": source_id,
            "sourceNodeOutputId": computation.link_output_ports.get(
                (source_index, target_index), _DEFAULT_OUTPUT_ID
            ),
            "targetNodeInputId": target_input_id,
        }
        if metadata:
            edge["metadata"] = metadata
        incoming[target_id].append(edge)

    if computation.output_node_index is not None:
        target_id = index_to_id.get(computation.output_node_index)
        if target_id is not None:
            for port, source_index in computation.output_ports.items():
                source_id = index_to_id.get(source_index)
                if source_id is None:
                    continue
                incoming[target_id].append(
                    {
                        "sourceNodeId": source_id,
                        "sourceNodeOutputId": _DEFAULT_OUTPUT_ID,
                        "targetNodeInputId": port,
                        "metadata": {"port_label": port},
                    }
                )

    return incoming


def _output_port_metadata(computation: ComputationGraph) -> list[dict[str, Any]]:
    return [
        {
            "id": port,
            "attrs": [_kv("port_label", port)],
        }
        for port in computation.output_ports
    ]


def computation_graph_to_explorer_graph(
    computation: ComputationGraph,
    *,
    graph_id: str,
    label: str | None = None,
) -> dict[str, Any]:
    """Convert one TraceLens computation graph to a Model Explorer ``Graph`` dict."""
    index_to_id = {
        index: spec.key or f"node:{index}"
        for index, spec in enumerate(computation.nodes)
    }
    namespaces = _node_namespaces(computation)
    incoming = _incoming_edges(computation, index_to_id)

    nodes: list[dict[str, Any]] = []
    for index, spec in enumerate(computation.nodes):
        node_id = index_to_id[index]
        node: dict[str, Any] = {
            "id": node_id,
            "label": spec.label or node_id,
            "namespace": namespaces.get(index, ""),
        }
        attrs = _node_attrs(spec)
        if attrs:
            node["attrs"] = attrs
        style = _node_style(spec)
        if style:
            node["style"] = style
        if node_id in incoming:
            node["incomingEdges"] = incoming[node_id]
        if index == computation.output_node_index:
            ports = _output_port_metadata(computation)
            node["inputsMetadata"] = [dict(port) for port in ports]
            node["outputsMetadata"] = ports
        nodes.append(node)

    finalize_graph_node_styles(nodes)

    graph: dict[str, Any] = {
        "id": graph_id,
        "nodes": nodes,
    }
    if label:
        graph["groupNodeAttributes"] = {
            "": {
                "title": label,
                "source": "TraceLens computation graph",
            }
        }
    return graph


def attach_subgraph_links(
    graphs: list[dict[str, Any]],
    *,
    attr_name_to_graph_id: dict[str, str],
) -> None:
    """Attach ``subgraphIds`` to nodes that reference nested computation graphs."""
    for graph in graphs:
        graph_id = graph["id"]
        for node in graph.get("nodes", []):
            candidate_ids: list[str] = []
            node_id = node.get("id")
            if isinstance(node_id, str) and node_id in attr_name_to_graph_id:
                candidate_ids.append(attr_name_to_graph_id[node_id])
            for attr in node.get("attrs", []):
                if attr.get("key") != "attr_name":
                    continue
                attr_name = attr.get("value")
                if isinstance(attr_name, str) and attr_name in attr_name_to_graph_id:
                    candidate_ids.append(attr_name_to_graph_id[attr_name])

            linked = sorted({sub_id for sub_id in candidate_ids if sub_id != graph_id})
            if linked:
                node["subgraphIds"] = linked
