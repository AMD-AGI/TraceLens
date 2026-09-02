###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Attach symbolic tensor shapes from shape inference to Model Explorer nodes."""

from __future__ import annotations

from typing import Any

from visualizer.block_tree import BlockNode
from visualizer.shape_inference import ShapeContext, ShapeInferencer, Symbol, TensorSpec

SHAPE_SEPARATOR = " x "


def _font_safe(dim: Any) -> str:
    """Model Explorer renders characters outside its font atlas as ``?``."""
    text = str(dim).replace("\u00d7", "x").replace("\u2217", "*")
    return "".join(char if 32 <= ord(char) < 127 else "" for char in text)


def format_shape(spec: TensorSpec) -> str:
    """Human-readable shape for node attrs (Model Explorer font-safe)."""
    return SHAPE_SEPARATOR.join(_font_safe(dim) for dim in spec.shape)


def format_shape_tensor(spec: TensorSpec) -> str:
    """Compact ``BxSxH`` shape for outputsMetadata tensor_shape conversion."""
    return "x".join(_font_safe(dim) for dim in spec.shape)


def format_shape_bracket(spec: TensorSpec) -> str:
    """Legacy bracket form kept for operator JSON consumers."""
    inner = ", ".join(str(dim) for dim in spec.shape)
    return f"[{inner}]"


def _apply_shape_attrs(node: dict[str, Any], spec: TensorSpec) -> None:
    shape_text = format_shape(spec)
    if not shape_text:
        return
    tensor_shape = format_shape_tensor(spec)
    attrs = [
        item
        for item in node.get("attrs", [])
        if item.get("key") not in {"output_shape", "output_dtype"}
    ]
    attrs.append({"key": "output_shape", "value": shape_text})
    attrs.append({"key": "output_dtype", "value": spec.dtype})
    node["attrs"] = attrs
    existing = node.get("outputsMetadata", [])
    port_ids = [item.get("id", "0") for item in existing] or ["0"]
    labels = {
        item.get("id", "0"): [
            attr for attr in item.get("attrs", []) if attr.get("key") == "port_label"
        ]
        for item in existing
    }
    node["outputsMetadata"] = [
        {
            "id": port_id,
            "attrs": [
                *labels.get(port_id, []),
                {"key": "shape", "value": shape_text},
                {"key": "tensor_shape", "value": tensor_shape},
                {"key": "dtype", "value": spec.dtype},
            ],
        }
        for port_id in port_ids
    ]


def annotate_nodes_with_shapes(
    nodes: list[dict[str, Any]],
    shape_specs: dict[str, TensorSpec],
    *,
    id_prefix: str,
) -> None:
    """Match exported nodes to computation-graph keys under ``id_prefix``."""
    if not shape_specs:
        return
    prefix = f"{id_prefix}/" if id_prefix else ""
    for node in nodes:
        node_id = node.get("id", "")
        if prefix:
            if not node_id.startswith(prefix):
                continue
            local_key = node_id[len(prefix) :]
        else:
            local_key = node_id
        spec = shape_specs.get(local_key)
        if spec is None:
            continue
        _apply_shape_attrs(node, spec)
        if any(
            attr.get("key") == "synthetic" and attr.get("value") == "@output"
            for attr in node.get("attrs", [])
        ):
            incoming_by_port = {
                edge.get("targetNodeInputId"): edge.get("sourceNodeId", "")
                for edge in node.get("incomingEdges", [])
            }
            for metadata in node.get("outputsMetadata", []):
                port = metadata.get("id")
                source_id = incoming_by_port.get(port, "")
                source_key = (
                    source_id[len(prefix) :]
                    if prefix and source_id.startswith(prefix)
                    else source_id
                )
                port_spec = shape_specs.get(source_key)
                if port_spec is None:
                    continue
                metadata["attrs"] = [
                    attr
                    for attr in metadata.get("attrs", [])
                    if attr.get("key") not in {"shape", "tensor_shape", "dtype"}
                ] + [
                    {"key": "shape", "value": format_shape(port_spec)},
                    {
                        "key": "tensor_shape",
                        "value": format_shape_tensor(port_spec),
                    },
                    {"key": "dtype", "value": port_spec.dtype},
                ]
            output_by_port = {
                str(metadata.get("id")): metadata
                for metadata in node.get("outputsMetadata", [])
            }
            node["inputsMetadata"] = [
                {
                    "id": metadata.get("id"),
                    "attrs": list(
                        output_by_port.get(str(metadata.get("id")), metadata).get(
                            "attrs", []
                        )
                    ),
                }
                for metadata in node.get("inputsMetadata", [])
            ]


def _node_spec(node: dict[str, Any]) -> TensorSpec | None:
    for metadata in node.get("outputsMetadata", []):
        attrs = {
            item.get("key"): item.get("value") for item in metadata.get("attrs", [])
        }
        shape_text = attrs.get("shape")
        if shape_text:
            return TensorSpec(
                shape=tuple(shape_text.split(SHAPE_SEPARATOR)),
                dtype=str(attrs.get("dtype") or ""),
            )
    return None


def _incoming_source_ids(node: dict[str, Any]) -> list[str]:
    return [
        str(edge.get("sourceNodeId"))
        for edge in node.get("incomingEdges", [])
        if edge.get("sourceNodeId")
    ]


def fill_missing_node_shapes(
    nodes: list[dict[str, Any]], *, context: ShapeContext
) -> None:
    """Give every node a shape so the viewer never falls back to rendering ``?``.

    Spine summaries, group input ports and nested-diagram nodes have no entry in the
    per-section inference results, so they inherit the shape of whatever feeds them and
    otherwise fall back to the model's activation shape.
    """
    hidden = context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
    vocab = context.dims.get(Symbol.VOCAB.value, Symbol.VOCAB.value)
    activation = TensorSpec(
        (Symbol.BATCH.value, Symbol.SEQ.value, hidden), context.dtype
    )
    logits = TensorSpec((Symbol.BATCH.value, Symbol.SEQ.value, vocab), context.dtype)
    tokens = TensorSpec((Symbol.BATCH.value, Symbol.SEQ.value), "int64")

    known: dict[str, TensorSpec] = {}
    pending: list[dict[str, Any]] = []
    for node in nodes:
        spec = _node_spec(node)
        if spec is None:
            pending.append(node)
        else:
            known[node.get("id", "")] = spec

    for node in list(pending):
        label = str(node.get("label") or "").strip().lower()
        node_id = str(node.get("id", ""))
        seeded: TensorSpec | None = None
        if node_id == "@input" or label in {"tokenized text", "input_ids"}:
            seeded = tokens
        elif label == "logits" or node_id.split("/")[-1] in {"lm_head", "output"}:
            seeded = logits
        elif "embedding" in label or "embed_tokens" in node_id:
            # Embeddings widen token ids, so they must not inherit the (B, S) input shape.
            seeded = activation
        elif "norm" in label:
            # Spine norms sit on the residual stream whatever the preceding tile computed.
            seeded = activation
        if seeded is not None:
            known[node.get("id", "")] = seeded
            _apply_shape_attrs(node, seeded)
            pending.remove(node)

    while pending:
        progressed = False
        for node in list(pending):
            sources = [
                known[source]
                for source in _incoming_source_ids(node)
                if source in known
            ]
            if not sources:
                continue
            spec = max(sources, key=lambda item: len(item.shape))
            known[node.get("id", "")] = spec
            _apply_shape_attrs(node, spec)
            pending.remove(node)
            progressed = True
        if not progressed:
            break

    for node in pending:
        _apply_shape_attrs(node, activation)


def _namespace_chain(namespace: str) -> list[str]:
    """Every group a node sits in, outermost first."""
    segments = [segment for segment in str(namespace or "").split("/") if segment]
    return ["/".join(segments[: index + 1]) for index in range(len(segments))]


def _record_shape(store: dict[str, list[str]], group: str, shape_text: str) -> None:
    shapes = store.setdefault(group, [])
    if shape_text not in shapes:
        shapes.append(shape_text)


def group_boundary_shapes(nodes: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Shapes crossing each expandable group's boundary, as layer attributes.

    Model Explorer only labels an edge when both endpoints are op nodes, so an edge that
    ends on a collapsed group renders bare. Recording what enters and leaves the group
    keeps those shapes readable without expanding it.
    """
    namespaces = {
        str(node.get("id", "")): str(node.get("namespace") or "") for node in nodes
    }
    shapes: dict[str, str] = {}
    for node in nodes:
        spec = _node_spec(node)
        if spec is not None:
            shapes[str(node.get("id", ""))] = format_shape(spec)

    inputs: dict[str, list[str]] = {}
    outputs: dict[str, list[str]] = {}
    for node in nodes:
        target_chain = _namespace_chain(namespaces.get(str(node.get("id", "")), ""))
        for source_id in _incoming_source_ids(node):
            shape_text = shapes.get(source_id)
            if not shape_text or source_id not in namespaces:
                continue
            source_chain = _namespace_chain(namespaces[source_id])
            for group in target_chain:
                if group not in source_chain:
                    _record_shape(inputs, group, shape_text)
            for group in source_chain:
                if group not in target_chain:
                    _record_shape(outputs, group, shape_text)

    attributes: dict[str, dict[str, str]] = {}
    for key, store in (("input_shape", inputs), ("output_shape", outputs)):
        for group, group_shapes in store.items():
            attributes.setdefault(group, {})[key] = ", ".join(group_shapes[:3])

    for node in nodes:
        namespace = str(node.get("namespace") or "")
        if not namespace:
            continue
        synthetic = next(
            (
                attr.get("value")
                for attr in node.get("attrs", [])
                if attr.get("key") == "synthetic"
            ),
            None,
        )
        if synthetic not in {"@input", "@output"}:
            continue
        output_metadata = node.get("outputsMetadata", [])
        values: list[str] = []
        for metadata in output_metadata:
            metadata_attrs = {
                attr.get("key"): attr.get("value") for attr in metadata.get("attrs", [])
            }
            shape = metadata_attrs.get("shape")
            if not shape:
                continue
            port = str(metadata.get("id", "0"))
            if len(output_metadata) == 1:
                values.append(str(shape))
            else:
                label = str(node.get("label") or "input") if port == "0" else port
                values.append(f"{label}: {shape}")
        if values:
            key = "input_shape" if synthetic == "@input" else "output_shape"
            attributes.setdefault(namespace, {})[key] = ", ".join(values)
    return attributes


def infer_block_tree_shapes(
    inferencer: ShapeInferencer,
    block_tree: BlockNode,
    *,
    title: str,
) -> dict[str, TensorSpec]:
    return inferencer.infer_block_tree(block_tree, title=title)
