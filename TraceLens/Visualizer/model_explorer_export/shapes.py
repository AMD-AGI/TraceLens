###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Attach symbolic tensor shapes from shape inference to Model Explorer nodes."""

from __future__ import annotations

import re
from typing import Any, Callable

from TraceLens.ModelUtils.block_tree import BlockNode
from TraceLens.ModelUtils.shape_inference import (
    DimExpr,
    PORT_SPEC_SEP,
    ShapeContext,
    ShapeInferencer,
    Symbol,
    TensorSpec,
    _divide_dim,
    _ENGINE_GATED_ACTIVATION_RE,
    _multi_output_slice_shape,
    _permute_shape,
    _reduce_conv_spatial,
    _reduction_axes,
    _replace_last_dim,
    _resolve_cast_dtype,
    _resolve_expand_shape,
    _resolve_view_shape,
    op_cannot_change_shape,
)

# Torch reductions a collapsed spine tile can still be. Keyed on ``raw_op``,
# the name the extractor records, which is the method the model called.
_REDUCTION_OPS = frozenset(
    {
        "all",
        "amax",
        "amin",
        "any",
        "argmax",
        "argmin",
        "logsumexp",
        "max",
        "mean",
        "min",
        "norm",
        "prod",
        "std",
        "sum",
        "var",
    }
)

SHAPE_SEPARATOR = " x "
BRACKET_SEPARATOR = ", "


def _font_safe(dim: Any) -> str:
    """Model Explorer renders characters outside its font atlas as ``?``.

    A multiplication marker is written as ``*`` so a merged reshape dim reads
    ``B*S``.
    """
    text = str(dim).replace("\u00d7", "*").replace("\u2217", "*")
    return "".join(char if 32 <= ord(char) < 127 else "" for char in text)


def format_shape(spec: TensorSpec) -> str:
    """Bracketed human-readable shape for node attrs, e.g. ``[B, S, 4096]``."""
    if not spec.shape:
        return ""
    inner = BRACKET_SEPARATOR.join(_font_safe(dim) for dim in spec.shape)
    return f"[{inner}]"


def format_shape_dims(dims: list[str]) -> str:
    """Bracket a list of already-stringified dims, e.g. ``["B", "S"]`` -> ``[B, S]``."""
    if not dims:
        return ""
    return f"[{BRACKET_SEPARATOR.join(dims)}]"


def parse_shape_dims(text: str) -> list[str]:
    """Inverse of :func:`format_shape` \u2014 recover dims from a display string.

    Accepts the bracket form (``[B, S, 4096]``) and, defensively, the legacy
    `` x ``-separated form so mixed data never crashes the round trip.
    """
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return [part.strip() for part in stripped.split(SHAPE_SEPARATOR) if part.strip()]


def format_shape_with_dtype(spec: TensorSpec) -> str:
    """Shape string with dtype suffix for port metadata, edges, and node attrs."""
    dims = format_shape(spec)
    if not dims:
        return ""
    if not spec.dtype:
        return dims
    return f"{dims} {spec.dtype}"


def _promote_tensor_shape_text(value: str) -> str:
    """Rewrite an adapter ``tensor_shape`` value the way the viewer used to.

    Model Explorer renames ``tensor_shape`` to ``shape`` and, on the way, strips
    a ``tensor<...>`` wrapper, substitutes a font-safe multiply sign, and spaces
    a compact ``1x2`` separator. Promoting with the same rules keeps an imported
    port reading as it did before the alias was dropped.
    """
    text = str(value).replace("tensor<", "").replace(">", "").replace("*", "\u2217")
    return re.sub(r"(\d|\?)x", r"\1 x ", text)


def _canonicalize_metadata(metadata: dict[str, Any]) -> None:
    attrs = metadata.get("attrs")
    if not isinstance(attrs, list):
        return
    shape = next((attr for attr in attrs if attr.get("key") == "shape"), None)
    tensor = next((attr for attr in attrs if attr.get("key") == "tensor_shape"), None)
    if shape is None and tensor is not None:
        attrs.append(
            {
                "key": "shape",
                "value": _promote_tensor_shape_text(str(tensor.get("value") or "")),
            }
        )
    if tensor is not None:
        metadata["attrs"] = [
            attr for attr in attrs if attr.get("key") != "tensor_shape"
        ]


def canonicalize_port_shape_keys(nodes: list[dict[str, Any]]) -> None:
    """Keep one shape key on every port.

    ``shape`` is the key Model Explorer draws on an edge. Built-in adapters
    still emit ``tensor_shape``; the viewer renames that alias onto ``shape``,
    and a later alias overwrites an earlier ``shape``. TraceLens writes only
    ``shape``. A port that arrived with only the alias is promoted. A port that
    already has ``shape`` keeps that text, and the alias is removed.
    """
    for node in nodes:
        for key in ("inputsMetadata", "outputsMetadata"):
            for metadata in node.get(key) or []:
                if isinstance(metadata, dict):
                    _canonicalize_metadata(metadata)


# The viewer draws ``output_shape`` and port ``shape``. This key keeps the whole
# tensor in the JSON, including when those displayed keys are later rewritten
# to one GPU's shard. The name is not drawn: the on-node filter matches the
# displayed keys only.
FULL_SHAPE_KEY = "full_shape"


def _copy_missing_attr(attrs: Any, source: str, dest: str) -> None:
    if not isinstance(attrs, list):
        return
    if any(isinstance(item, dict) and item.get("key") == dest for item in attrs):
        return
    value = next(
        (
            item.get("value")
            for item in attrs
            if isinstance(item, dict)
            and item.get("key") == source
            and isinstance(item.get("value"), str)
            and item.get("value")
        ),
        None,
    )
    if isinstance(value, str):
        attrs.append({"key": dest, "value": value})


def stamp_full_shapes(nodes: list[Any]) -> None:
    """Record the unpartitioned shape next to the one the viewer draws.

    With no device split the two texts match. A later partition pass rewrites
    only ``output_shape`` and port ``shape``.
    """
    for node in nodes:
        if not isinstance(node, dict):
            continue
        _copy_missing_attr(node.get("attrs"), "output_shape", FULL_SHAPE_KEY)
        for metadata_key in ("outputsMetadata", "inputsMetadata"):
            for metadata in node.get(metadata_key) or []:
                if isinstance(metadata, dict):
                    _copy_missing_attr(metadata.get("attrs"), "shape", FULL_SHAPE_KEY)


def refresh_displayed_group_shapes(graph: dict[str, Any]) -> None:
    """Point each group's drawn shape at the tensors currently on its boundary.

    ``full_input_shape`` and ``full_output_shape`` stay the unpartitioned
    values. ``input_shape`` and ``output_shape`` are what the viewer draws, so
    they follow a per-GPU rewrite of the nodes.
    """
    nodes = graph.get("nodes")
    attributes = graph.get("groupNodeAttributes")
    if not isinstance(nodes, list) or not isinstance(attributes, dict):
        return
    for namespace, boundary in group_boundary_shapes(nodes).items():
        bucket = attributes.setdefault(namespace, {})
        if not isinstance(bucket, dict):
            continue
        for key, value in boundary.items():
            full_key = f"full_{key}"
            if full_key not in bucket:
                existing = bucket.get(key)
                bucket[full_key] = (
                    existing if isinstance(existing, str) and existing else value
                )
            bucket[key] = value


def canonicalize_payload_shapes(payload: dict[str, Any]) -> None:
    """Fold ``tensor_shape`` into ``shape`` on every graph in a payload."""
    if not isinstance(payload, dict):
        return
    graphs: list[dict[str, Any]] = []
    collections = payload.get("graphCollections")
    if isinstance(collections, list):
        for collection in collections:
            if not isinstance(collection, dict):
                continue
            for graph in collection.get("graphs") or []:
                if isinstance(graph, dict):
                    graphs.append(graph)
    for graph in payload.get("graphs") or []:
        if isinstance(graph, dict):
            graphs.append(graph)
    nodes = payload.get("nodes")
    if isinstance(nodes, list):
        graphs.append(payload)
    seen: set[int] = set()
    for graph in graphs:
        graph_nodes = graph.get("nodes")
        if not isinstance(graph_nodes, list):
            continue
        marker = id(graph_nodes)
        if marker in seen:
            continue
        seen.add(marker)
        canonicalize_port_shape_keys(graph_nodes)


def _apply_shape_attrs(node: dict[str, Any], spec: TensorSpec) -> None:
    display_text = format_shape_with_dtype(spec)
    if not display_text:
        return
    attrs = [
        item
        for item in node.get("attrs", [])
        if item.get("key") not in {"output_shape", "output_dtype"}
    ]
    attrs.append({"key": "output_shape", "value": display_text})
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
                {"key": "shape", "value": display_text},
                {"key": "dtype", "value": spec.dtype},
            ],
        }
        for port_id in port_ids
    ]


def apply_shape_attrs(node: dict[str, Any], spec: TensorSpec) -> None:
    """Attach one known output shape to an exported node."""
    _apply_shape_attrs(node, spec)


def _output_names_attr(node: dict[str, Any]) -> list[str]:
    for attr in node.get("attrs", []):
        if attr.get("key") == "output_names":
            return [name for name in str(attr.get("value") or "").split(",") if name]
    return []


def _port_specs_for(
    local_key: str, shape_specs: dict[str, TensorSpec]
) -> dict[int, TensorSpec]:
    """Per-ordinal slice specs a tuple-unpacked split published for ``local_key``.

    The inferencer stores one spec per consumed output port under the reserved
    ``f"{node_id}{PORT_SPEC_SEP}{ordinal}"`` key (see ``infer_model_graph``).
    """
    prefix = f"{local_key}{PORT_SPEC_SEP}"
    result: dict[int, TensorSpec] = {}
    for key, spec in shape_specs.items():
        if not key.startswith(prefix):
            continue
        try:
            result[int(key[len(prefix) :])] = spec
        except ValueError:
            continue
    return result


def _apply_multi_port_shape_attrs(
    node: dict[str, Any],
    whole_spec: TensorSpec | None,
    port_specs: dict[int, TensorSpec],
) -> None:
    """Stamp one output port per split/unbind slice, each with its own shape.

    A tuple-unpacked split has real output ports (``pre_w``/``post_w``/``comb_w``)
    instead of synthetic per-slice tiles. Consumers were already wired to the
    right ordinal (``sourceNodeOutputId``); here each port carries its slice shape
    so the edge feeding ``view`` reads ``[B, S, 16]`` rather than the whole tensor.

    A ``Pack`` has no "whole" shape to speak of -- its ports are independently-
    shaped tensors bundled for threading through Python, not slices of one
    tensor -- so it never gets an ``output_shape``/``output_dtype`` of its own;
    only the per-port shapes below are real. A branch ``Merge`` reaches here
    (with per-port specs) only when both of its arms are themselves such a
    bundle, so the same rule applies: there is no single tensor either arm's
    "whole" shape could honestly stand in for.
    """
    output_names = _output_names_attr(node)
    is_bundle = str(node.get("label") or "") in {"Pack", "Mutually exclusive branch"}
    if whole_spec is not None and not is_bundle:
        display_text = format_shape_with_dtype(whole_spec)
        attrs = [
            item
            for item in node.get("attrs", [])
            if item.get("key") not in {"output_shape", "output_dtype"}
        ]
        if display_text:
            attrs.append({"key": "output_shape", "value": display_text})
            attrs.append({"key": "output_dtype", "value": whole_spec.dtype})
        node["attrs"] = attrs
    ports: list[dict[str, Any]] = []
    for ordinal in sorted(port_specs):
        port_spec = port_specs[ordinal]
        label = output_names[ordinal] if ordinal < len(output_names) else str(ordinal)
        ports.append(
            {
                "id": str(ordinal),
                "attrs": [
                    {"key": "port_label", "value": label},
                    {"key": "shape", "value": format_shape_with_dtype(port_spec)},
                    {"key": "dtype", "value": port_spec.dtype},
                ],
            }
        )
    if ports:
        node["outputsMetadata"] = ports


def annotate_nodes_with_shapes(
    nodes: list[dict[str, Any]],
    shape_specs: dict[str, TensorSpec],
    *,
    id_prefix: str,
) -> None:
    """Match exported nodes to computation-graph keys under ``id_prefix``."""
    canonicalize_port_shape_keys(nodes)
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
        port_specs = _port_specs_for(local_key, shape_specs)
        if port_specs:
            _apply_multi_port_shape_attrs(node, shape_specs.get(local_key), port_specs)
            continue
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
                    # Keep the ``shape`` field dtype-qualified, matching
                    # ``_apply_shape_attrs``/``_apply_multi_port_shape_attrs``.
                    # ``group_boundary_shapes`` reads this value for an expandable
                    # module's ``output_shape`` layer attribute, so dropping the
                    # dtype here left collapsed modules showing shape without type.
                    {"key": "shape", "value": format_shape_with_dtype(port_spec)},
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


def _node_spec(
    node: dict[str, Any], output_port: str | None = None
) -> TensorSpec | None:
    metadata_items = node.get("outputsMetadata", [])
    if output_port is not None:
        matching = [
            metadata
            for metadata in metadata_items
            if str(metadata.get("id", "0")) == str(output_port)
        ]
        if matching:
            metadata_items = matching
    for metadata in metadata_items:
        attrs = {
            item.get("key"): item.get("value") for item in metadata.get("attrs", [])
        }
        shape_text = attrs.get("shape")
        if shape_text:
            dtype = str(attrs.get("dtype") or "")
            # Strip trailing dtype suffix from display shape text.
            if dtype and shape_text.endswith(f" {dtype}"):
                shape_text = shape_text[: -len(dtype) - 1]
            return TensorSpec(
                shape=tuple(parse_shape_dims(shape_text)),
                dtype=dtype,
            )
    return None


def node_output_spec(
    node: dict[str, Any], output_port: str | None = None
) -> TensorSpec | None:
    """Read one exported output port's symbolic tensor specification."""
    return _node_spec(node, output_port)


def _incoming_sources(node: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [
        (
            str(edge.get("sourceNodeId")),
            str(edge.get("sourceNodeOutputId", "0")),
            str(edge.get("targetNodeInputId", "0")),
        )
        for edge in node.get("incomingEdges", [])
        if edge.get("sourceNodeId")
    ]


def _node_synthetic(node: dict[str, Any]) -> str:
    for attr in node.get("attrs", []):
        if attr.get("key") == "synthetic":
            return str(attr.get("value") or "")
    return ""


def _incoming_source_ids(node: dict[str, Any]) -> list[str]:
    return [
        source_id for source_id, _source_port, _target_port in _incoming_sources(node)
    ]


# Operations whose result broadcasts across their operands. Named as
# OPERATIONS -- what the model called -- which is the one thing a shape may be
# computed from.
_BROADCASTING_OPS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "floor_divide",
        "pow",
        "maximum",
        "minimum",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
    }
)


def _operation_of(node: dict[str, Any]) -> str:
    """The operation a node performs, as the node itself recorded it.

    Read from both channels it can arrive in. The adapter lifts the extractor's
    ``raw_op:`` detail into an attribute of its own, but a node built on the
    source path keeps it in ``detail``, and reading only the attribute left
    GLM's hyper-connection ``unsqueeze`` with no operation at all -- so it
    returned its input unchanged and the residual it feeds lost the hidden axis.
    """
    detail_text = ""
    for attr in node.get("attrs", []):
        key = attr.get("key")
        if key == "raw_op":
            recorded = str(attr.get("value") or "").strip().lower()
            if recorded:
                return recorded
        elif key in ("detail", "details"):
            detail_text = f"{detail_text}; {attr.get('value')}"
    for entry in detail_text.split(";"):
        name, separator, value = entry.partition(":")
        if separator and name.strip() == "raw_op":
            return value.strip().lower()
    return ""


def _detail_parts(details: list[str]) -> list[str]:
    """Split joined ``details`` attr text into the extractor's own entries."""
    parts: list[str] = []
    for detail in details:
        parts.extend(part.strip() for part in str(detail).split(";") if part.strip())
    return parts


def _reduce_recorded_axes(source: TensorSpec, details: list[str]) -> TensorSpec | None:
    """Drop the axes a reduction's ``dim`` names, or ``None`` when it names none.

    ``keepdim=True`` leaves a size-1 axis. An axis this rank does not carry is
    left alone: the symbolic view sometimes omits a stream the call indexes.
    """
    parts = _detail_parts(details)
    dim = next(
        (part.split(":", 1)[1].strip() for part in parts if part.startswith("dim:")),
        None,
    )
    axes = _reduction_axes(dim)
    if not axes:
        return None
    rank = len(source.shape)
    present = sorted(
        {axis + rank if axis < 0 else axis for axis in axes} & set(range(rank))
    )
    if not present:
        return None
    keepdim = any(part.strip() == "keepdim: True" for part in parts)
    dims = list(source.shape)
    if keepdim:
        for axis in present:
            dims[axis] = 1
    else:
        for axis in reversed(present):
            del dims[axis]
    return TensorSpec(tuple(dims), source.dtype)


def _fallback_node_spec(
    node: dict[str, Any],
    sources: list[tuple[str, TensorSpec]],
    *,
    working_dtype: str = "float16",
    dims: dict[str, DimExpr] | None = None,
    conv_geometry: (
        dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] | None
    ) = None,
    module_spec: Callable[[str, TensorSpec], TensorSpec | None] | None = None,
) -> TensorSpec | None:
    """Infer merge-only synthetic ops that have no block-tree shape record."""
    specs = [spec for _target_port, spec in sources]
    # The operation the node records, never its display label. A label is for
    # reading and for naming a producer; what a node COMPUTES is a separate
    # fact, and the extractor records it (``raw_op``) for every op -- including
    # the ones built from Python operators and the ones merge synthesises.
    operation = _operation_of(node)
    node_id = str(node.get("id") or "")
    attrs = node.get("attrs", [])
    details = [
        str(attr.get("value"))
        for attr in attrs
        if attr.get("key") == "detail" or attr.get("key") == "details"
    ]

    class_name = next(
        (str(a.get("value")) for a in attrs if a.get("key") == "class_name"), ""
    )
    label = str(node.get("label") or "")
    if (
        specs
        and specs[0].shape
        and _ENGINE_GATED_ACTIVATION_RE.search(class_name or label)
    ):
        # vLLM/SGLang's gated-activation family (``SiluAndMul``,
        # ``SiluAndMulWithClamp``, ``GeluAndMul``, SGLang's ``SituAndMul``, ...):
        # every one halves its last axis (``gate, up = x.chunk(2, dim=-1)``).
        # These are bare calls with no block-tree module to size them, so this
        # fallback -- not a constructor-driven rule -- is where the shape has
        # to come from. See ``shape_inference._ENGINE_GATED_ACTIVATION_RE``.
        halved = _divide_dim(specs[0].shape[-1], 2)
        if halved is not None:
            return TensorSpec(
                shape=_replace_last_dim(specs[0].shape, halved), dtype=specs[0].dtype
            )

    attr_name_for_module = next(
        (str(a.get("value")) for a in attrs if a.get("key") == "attr_name"), ""
    )
    if module_spec is not None and specs and attr_name_for_module:
        # A parameterised module tile (an embedding, a head) states its output
        # through its own weight. Asked before anything else so a module is
        # never sized by what merely flows into it.
        from_parameters = module_spec(attr_name_for_module, specs[0])
        if from_parameters is not None:
            return from_parameters

    if specs:
        # Vision patch-merger downsample: a Conv{1,2,3}d is not keyed by block-tree
        # inference, so reduce spatial extents here using geometry captured from the
        # constructor (kernel/stride/padding), mirroring the smart inference path.
        # Having that geometry IS what identifies the op -- it is harvested from
        # convolution constructors and from nothing else -- so no label is read.
        attr_name = next(
            (str(a.get("value")) for a in attrs if a.get("key") == "attr_name"), ""
        )
        geometry = (conv_geometry or {}).get(attr_name)
        if geometry is not None:
            kernel, stride, padding = geometry
            shape = list(specs[0].shape)
            channel_axis = 1 if len(shape) >= 2 else 0
            _reduce_conv_spatial(shape, channel_axis, kernel, stride, padding)
            return TensorSpec(tuple(shape), specs[0].dtype)

    if operation in {"view", "reshape"} and specs:
        # Vision-model-level reshapes (spatial merge, patch pooling) are not keyed
        # by the block-tree inference, so resolve them here from the recorded
        # ``shape:`` detail rather than passing the input through unchanged.
        shape_detail = next(
            (
                detail.split(":", 1)[1].strip()
                for detail in details
                if detail.startswith("shape:")
            ),
            "",
        )
        resolved = _resolve_view_shape(shape_detail, specs[0], dims or {})
        if resolved is not None:
            return TensorSpec(resolved, specs[0].dtype)
        declared = [part.strip() for part in shape_detail.split(",") if part.strip()]
        if len(declared) >= 2 and len(declared) != len(specs[0].shape):
            return None

    if operation in {"permute", "transpose"} and specs:
        dims_detail = next(
            (
                detail.split(":", 1)[1].strip()
                for detail in details
                if detail.startswith("dims:")
            ),
            None,
        )
        permuted = _permute_shape(specs[0].shape, dims_detail)
        if permuted is not None:
            return TensorSpec(permuted, specs[0].dtype)

    # A conversion is identified by the target dtype the extractor recorded for
    # it, which only a dtype-converting call carries -- not by being called
    # "Cast". It keeps the shape it was given either way.
    if specs and any(detail.startswith("dtype") for detail in details):
        source = specs[0]
        dtype_detail = next(
            (
                detail.split(":", 1)[1].strip()
                for detail in details
                if detail.startswith("dtype:")
            ),
            "",
        )
        if dtype_detail:
            return TensorSpec(
                source.shape,
                _resolve_cast_dtype(dtype_detail, source.dtype, working_dtype),
            )
        return source

    if operation == "unsqueeze" and specs:
        source = specs[0]
        dim_detail = next(
            (
                detail.split(":", 1)[1].strip()
                for detail in details
                if detail.startswith("dim:")
            ),
            None,
        )
        if dim_detail is not None:
            try:
                dim = int(dim_detail)
            except ValueError:
                dim = 0
            if dim < 0:
                dim += len(source.shape) + 1
            shape = list(source.shape)
            shape.insert(max(0, min(dim, len(shape))), 1)
            return TensorSpec(tuple(shape), source.dtype)

    if operation == "squeeze" and specs:
        source = specs[0]
        dim_detail = next(
            (
                detail.split(":", 1)[1].strip()
                for detail in details
                if detail.startswith("dim:")
            ),
            None,
        )
        shape = list(source.shape)
        if not shape:
            return source
        if dim_detail is None:
            # Bare squeeze drops every size-1 axis.
            return TensorSpec(tuple(size for size in shape if size != 1), source.dtype)
        try:
            dim = int(dim_detail)
        except ValueError:
            return source
        axis = dim % len(shape)
        if 0 <= axis < len(shape) and shape[axis] == 1:
            del shape[axis]
            return TensorSpec(tuple(shape), source.dtype)
        return source

    if operation == "expand" and specs:
        source = specs[0]
        shape_detail = next(
            (
                detail.split(":", 1)[1].strip()
                for detail in details
                if detail.startswith("shape:")
            ),
            "",
        )
        resolved = _resolve_expand_shape(shape_detail, source, dims or {})
        if resolved is not None:
            return TensorSpec(resolved, source.dtype)
        return source

    if operation in _BROADCASTING_OPS and len(specs) >= 2:
        rank = max(len(spec.shape) for spec in specs)
        padded = [(1,) * (rank - len(spec.shape)) + tuple(spec.shape) for spec in specs]
        shape: list[Any] = []
        for dimensions in zip(*padded):
            non_unit = [dimension for dimension in dimensions if dimension != 1]
            if non_unit and all(
                isinstance(dimension, int)
                or (isinstance(dimension, str) and dimension.isdigit())
                for dimension in non_unit
            ):
                shape.append(max(non_unit, key=lambda dimension: int(dimension)))
            else:
                shape.append(non_unit[-1] if non_unit else 1)
        return TensorSpec(tuple(shape), specs[-1].dtype)

    if operation in {"matmul", "mm", "bmm"} and len(specs) >= 2:
        left, right = specs[:2]
        if left.shape and right.shape:
            return TensorSpec((*left.shape[:-1], right.shape[-1]), right.dtype)

    if "@residual:" in node_id and operation == "mul":
        by_input = {target_port: spec for target_port, spec in sources}
        post = by_input.get("post")
        hidden_states = by_input.get("hidden_states")
        if post is not None and hidden_states is not None and hidden_states.shape:
            return TensorSpec(
                (*post.shape, hidden_states.shape[-1]), hidden_states.dtype
            )

    if "@residual:" in node_id and operation == "matmul":
        by_input = {target_port: spec for target_port, spec in sources}
        comb = by_input.get("comb")
        residual = by_input.get("residual")
        if comb is not None and residual is not None and comb.shape and residual.shape:
            return TensorSpec((*comb.shape[:-1], residual.shape[-1]), residual.dtype)

    # A one-op module collapsed onto the spine (GLM's ``hc_head`` is
    # ``hidden_streams.mean(dim=2)``) has no block-tree shape record. Without
    # this it inherits the tensor it was handed, so the axis the mean drops
    # stays in the shape and the norm that reads it disagrees with its cast.
    if operation in _REDUCTION_OPS and specs and specs[0].shape:
        reduced = _reduce_recorded_axes(specs[0], details)
        if reduced is not None:
            return reduced

    return max(
        specs,
        key=lambda item: (
            len(item.shape),
            item.shape[-1] if item.shape and isinstance(item.shape[-1], int) else 0,
        ),
    )


def fill_missing_node_shapes(
    nodes: list[dict[str, Any]],
    *,
    context: ShapeContext,
    boundary_spec: Callable[[str, str], TensorSpec | None] | None = None,
    module_spec: Callable[[str, TensorSpec], TensorSpec | None] | None = None,
) -> None:
    """Give a node the shape its producer or a meta trace already established.

    Spine summaries, group input ports and nested-diagram nodes have no entry in the
    per-section inference results, so they inherit the shape of whatever feeds them.
    A node with neither stays unshaped.

    ``boundary_spec(label, namespace)`` optionally supplies the meta-device ground
    truth for a leaf ``@input`` boundary parameter (e.g. the sparse-attention
    indexer's ``attention_mask`` is ``[B, S]`` bool, not the generic
    ``(B, S, hidden)``) -- what the model actually passed, observed rather than
    inferred from the parameter's name.
    """
    canonicalize_port_shape_keys(nodes)
    tokens = TensorSpec((Symbol.BATCH.value, Symbol.SEQ.value), "int64")

    node_by_id = {str(node.get("id", "")): node for node in nodes}
    known: dict[tuple[str, str], TensorSpec] = {}
    pending: list[dict[str, Any]] = []
    for node in nodes:
        metadata_items = node.get("outputsMetadata", [])
        found = False
        for metadata in metadata_items:
            port = str(metadata.get("id", "0"))
            spec = _node_spec(node, port)
            if spec is not None:
                known[(str(node.get("id", "")), port)] = spec
                found = True
        if not found:
            pending.append(node)

    for node in list(pending):
        orig_label = str(node.get("label") or "").strip()
        node_id = str(node.get("id", ""))
        namespace = str(node.get("namespace") or "")
        synthetic = _node_synthetic(node)
        # A boundary carrying a real producer inherits that producer's shape.
        # Any boundary, not only an entering one: a frame's ``@output`` has a
        # producer inside the frame and must inherit it too.
        connected_boundary = synthetic is not None and bool(_incoming_sources(node))
        # Meta-device ground truth for a forward-parameter boundary. Only for
        # real ``@input`` boundaries and only when the meta trace observed the
        # parameter -- it is a measurement of what the model was passed.
        meta_boundary: TensorSpec | None = None
        if boundary_spec is not None and synthetic in {"@input", "@input_mirror"}:
            meta_boundary = boundary_spec(orig_label, namespace)

        # Nothing here reads what a node is CALLED. There used to be a ladder
        # of label tests -- ``logits``/``lm_head`` meant (B, S, vocab),
        # ``embedding`` or ``embed`` meant (B, S, hidden), so did ``norm``, so
        # did ``attention`` -- each stamping a canonical shape on any tile whose
        # name contained the word. They were wrong wherever a name recurred:
        # MiniMax's rotary publishes ``q_embed``/``k_embed``, which are
        # [B, heads, S, head_dim], and were stamped (B, S, hidden) for
        # containing ``embed``. A shape is what the operation computes from its
        # operands and its own parameters; a name is not evidence about it.
        #
        # What is left is evidence: the model's own entry point, a producer to
        # inherit from, and what the meta trace observed for a real forward
        # parameter.
        seeded: TensorSpec | None = None
        if node_id == "@input":
            seeded = tokens
        elif connected_boundary:
            # A real incoming dataflow edge beats everything: leave it pending
            # so the edge-propagation pass below inherits the producer's
            # concrete shape. Meta ground-truth only seeds LEAF boundaries,
            # which have no producer to inherit from.
            seeded = None
        elif meta_boundary is not None:
            seeded = meta_boundary
        if seeded is not None:
            known[(node_id, "0")] = seeded
            _apply_shape_attrs(node, seeded)
            pending.remove(node)

    while pending:
        progressed = False
        for node in list(pending):
            sources: list[tuple[str, TensorSpec]] = []
            incoming_sources = _incoming_sources(node)
            for source_id, source_port, target_port in incoming_sources:
                spec = known.get((source_id, source_port))
                if spec is None:
                    source_node = node_by_id.get(source_id)
                    if source_node is not None:
                        spec = _node_spec(source_node, source_port)
                if spec is not None:
                    sources.append((target_port, spec))
            if not sources or len(sources) != len(incoming_sources):
                continue
            spec = _fallback_node_spec(
                node,
                sources,
                working_dtype=context.dtype,
                dims=context.dims,
                conv_geometry=context.conv_geometry,
                module_spec=module_spec,
            )
            if spec is None:
                continue
            node_id = str(node.get("id", ""))
            known[(node_id, "0")] = spec
            _apply_shape_attrs(node, spec)
            pending.remove(node)
            progressed = True
        if progressed:
            continue
        # An operand that never gets a shape must not blank the ones that did.
        # A broadcast op can be read off the operands that resolved; a join or
        # a view cannot, and stays unshaped.
        for node in list(pending):
            if _operation_of(node) not in {
                "add",
                "subtract",
                "multiply",
                "divide",
                "power",
                "maximum",
                "minimum",
            }:
                continue
            sources = []
            incoming_sources = _incoming_sources(node)
            for source_id, source_port, target_port in incoming_sources:
                spec = known.get((source_id, source_port))
                if spec is None:
                    source_node = node_by_id.get(source_id)
                    if source_node is not None:
                        spec = _node_spec(source_node, source_port)
                if spec is not None:
                    sources.append((target_port, spec))
            if not sources or len(sources) == len(incoming_sources):
                continue
            if max(len(spec.shape) for _port, spec in sources) < 2:
                continue
            spec = _fallback_node_spec(
                node,
                sources,
                working_dtype=context.dtype,
                dims=context.dims,
                conv_geometry=context.conv_geometry,
                module_spec=module_spec,
            )
            if spec is None:
                continue
            node_id = str(node.get("id", ""))
            known[(node_id, "0")] = spec
            _apply_shape_attrs(node, spec)
            pending.remove(node)
            progressed = True
            break
        if not progressed:
            break

    # Whatever is still pending has no shape of its own and no shaped producer.
    # Leave it unshaped. Inventing the activation here is what made a padding
    # mask report ``[B, S, H]``.


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
    node_by_id = {str(node.get("id", "")): node for node in nodes}

    inputs: dict[str, list[str]] = {}
    outputs: dict[str, list[str]] = {}
    for node in nodes:
        target_chain = _namespace_chain(namespaces.get(str(node.get("id", "")), ""))
        for source_id, source_port, _target_port in _incoming_sources(node):
            source_node = node_by_id.get(source_id)
            spec = (
                _node_spec(source_node, source_port)
                if source_node is not None
                else None
            )
            if spec is None or source_id not in namespaces:
                continue
            shape_text = format_shape_with_dtype(spec)
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

    # Collect every boundary port (label, shape) per (namespace, key). The name is
    # shown ONLY when a boundary carries more than one output/input — a per-port
    # split leaves each synthetic node with a single port, so the decision must be
    # made from the boundary's total arity across all its sibling @input/@output
    # nodes, not from any one node's port count.
    boundary_entries: dict[tuple[str, str], list[tuple[str, str]]] = {}
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
        key = "input_shape" if synthetic == "@input" else "output_shape"
        default_name = "input" if synthetic == "@input" else "output"
        for metadata in node.get("outputsMetadata", []):
            metadata_attrs = {
                attr.get("key"): attr.get("value") for attr in metadata.get("attrs", [])
            }
            shape = metadata_attrs.get("shape")
            if not shape:
                continue
            port = str(metadata.get("id", "0"))
            port_label = metadata_attrs.get("port_label")
            if port_label:
                label = str(port_label)
            elif port in {"0", "result", "output"}:
                label = str(node.get("label") or default_name)
            else:
                label = port
            boundary_entries.setdefault((namespace, key), []).append(
                (label, str(shape))
            )
    for (namespace, key), entries in boundary_entries.items():
        if len(entries) > 1:
            values = [f"{label}: {shape}" for label, shape in entries]
        else:
            values = [shape for _label, shape in entries]
        attributes.setdefault(namespace, {})[key] = ", ".join(dict.fromkeys(values))
    return attributes


def widen_broadcast_results(nodes: list[dict[str, Any]]) -> None:
    """Replace a broadcast result that is narrower than its operands.

    A multiply inferred before its activation had a spec keeps the scalar that
    already had one. The activation is on the edge; once that edge is drawn,
    the result is the broadcast, and a boundary that only forwards it takes
    the same shape. A conversion that never received a spec copies the operand
    the edge names, so the multiply can see it. A later section then sees the
    activation it is handed.
    """
    node_by_id = {str(node.get("id", "")): node for node in nodes}
    for _ in range(8):
        progressed = False
        for node in nodes:
            incoming = _incoming_sources(node)
            if not incoming:
                continue
            sources: list[tuple[str, TensorSpec]] = []
            complete = True
            for source_id, source_port, target_port in incoming:
                source = node_by_id.get(source_id)
                spec = _node_spec(source, source_port) if source is not None else None
                if spec is None or not spec.shape:
                    complete = False
                    break
                sources.append((target_port, spec))
            if not complete or not sources:
                continue
            synthetic = _node_synthetic(node)
            operation = _operation_of(node)
            current = _node_spec(node, "0")
            shape_preserving = (
                len(sources) == 1
                and bool(operation)
                and op_cannot_change_shape(operation)
            )
            if operation in _BROADCASTING_OPS:
                updated = _fallback_node_spec(node, sources)
            elif shape_preserving:
                source_spec = sources[0][1]
                updated = TensorSpec(
                    source_spec.shape,
                    current.dtype if current is not None else source_spec.dtype,
                )
            elif synthetic in {"@output", "@input"} and len(sources) == 1:
                updated = sources[0][1]
            elif (
                current is None
                and len(sources) == 1
                and synthetic
                not in {
                    "@input",
                    "@output",
                    "@input_mirror",
                    "@output_mirror",
                }
            ):
                # A conversion inferred before its operand had a spec stays
                # blank, and the multiply reading it then keeps the scalar it
                # could already see. The operand is on the edge now.
                updated = sources[0][1]
            else:
                continue
            if updated is None or not updated.shape:
                continue
            if (
                current is not None
                and updated.shape == current.shape
                and updated.dtype == current.dtype
            ):
                continue
            if (
                not shape_preserving
                and current is not None
                and len(updated.shape) <= len(current.shape)
            ):
                continue
            apply_shape_attrs(node, updated)
            progressed = True
        if not progressed:
            return


def refine_multi_output_port_shapes(nodes: list[dict[str, Any]]) -> None:
    """Restamp split/chunk/unbind ports from the final whole-tensor shape.

    A node can be inferred again after its initial per-port specs were published.
    The whole shape is then current while its output metadata still carries the
    old undivided shape. Recomputing from the operation's recorded split contract
    keeps each consumer on the slice it actually receives.
    """
    refined: dict[tuple[str, str], TensorSpec] = {}
    for node in nodes:
        operation = _operation_of(node)
        if operation not in {"split", "chunk", "unbind"}:
            continue
        whole_shape = next(
            (
                str(attr.get("value") or "")
                for attr in node.get("attrs", [])
                if attr.get("key") == "output_shape"
            ),
            "",
        )
        whole_dtype = next(
            (
                str(attr.get("value") or "")
                for attr in node.get("attrs", [])
                if attr.get("key") == "output_dtype"
            ),
            "",
        )
        shape_only = whole_shape
        if whole_dtype and shape_only.endswith(f" {whole_dtype}"):
            shape_only = shape_only[: -len(whole_dtype) - 1]
        whole = (
            TensorSpec(tuple(parse_shape_dims(shape_only)), whole_dtype)
            if shape_only
            else None
        )
        metadata = node.get("outputsMetadata", []) or []
        if whole is None or len(metadata) < 2:
            continue
        whole = TensorSpec(
            tuple(
                (
                    int(dimension)
                    if isinstance(dimension, str) and dimension.isdigit()
                    else dimension
                )
                for dimension in whole.shape
            ),
            whole.dtype,
        )
        details = _detail_parts(
            [
                str(attr.get("value") or "")
                for attr in node.get("attrs", [])
                if attr.get("key") in {"detail", "details"}
            ]
        )
        slices = [
            _multi_output_slice_shape(
                whole,
                details,
                operation,
                ordinal,
                {},
                output_count=len(metadata),
            )
            for ordinal in range(len(metadata))
        ]
        if any(spec is None for spec in slices):
            continue
        for port, spec in zip(metadata, slices):
            assert spec is not None
            label_attrs = [
                attr
                for attr in port.get("attrs", [])
                if attr.get("key") == "port_label"
            ]
            display = format_shape_with_dtype(spec)
            port["attrs"] = [
                *label_attrs,
                {"key": "shape", "value": display},
                {"key": "dtype", "value": spec.dtype},
            ]
            refined[(str(node.get("id") or ""), str(port.get("id", "0")))] = spec

    # Tuple-boundary splitting materialises one ``@slice_out`` tile per output
    # port. Those tiles may already carry the undivided shape copied before this
    # final refinement, so restamp them from the exact port they read.
    for node in nodes:
        if _node_synthetic(node) != "@slice_out":
            continue
        incoming = _incoming_sources(node)
        if len(incoming) != 1:
            continue
        source_id, source_port, _target_port = incoming[0]
        spec = refined.get((source_id, source_port))
        if spec is not None:
            apply_shape_attrs(node, spec)


def infer_block_tree_shapes(
    inferencer: ShapeInferencer,
    block_tree: BlockNode,
    *,
    title: str,
    entry_spec: TensorSpec | None = None,
) -> dict[str, TensorSpec]:
    return inferencer.infer_block_tree(block_tree, title=title, entry_spec=entry_spec)
