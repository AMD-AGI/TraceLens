###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build Model Explorer payloads from TraceLens architecture specs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.extract import ArchitectureSpec
from TraceLens.ModelUtils.shape_inference import (
    ShapeInferencer,
    build_operator_export,
    serialize_dim,
)

from TraceLens.Visualizer.model_explorer_export.computation_roles import (
    stamp_computation_roles,
)
from TraceLens.Visualizer.model_explorer_export.fact_sheet import (
    build_fact_sheet_viewer,
)
from TraceLens.Visualizer.model_explorer_export.merge import build_merged_model_graph


def _export_basic_ops(basic_ops: BasicOpFilter) -> BasicOpFilter:
    """Keep full kernel inline substeps in Model Explorer (not basic-only tails)."""
    return BasicOpFilter(
        basic_ops.patterns,
        basic_only=False,
        stops_at_grouped_moe=basic_ops.stops_at_grouped_moe,
    )


# Exact convolution op types -- not a ``startswith("Conv")`` test, which would also
# catch a model's own ``Conv*``-prefixed module classes (``ConvNextDropPath``).
_CONV_OP_TYPES = frozenset(
    {
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "ConvTranspose3d",
    }
)


# A one-segment match on these leaves collides across a bottleneck
# (``layer.0.convolution`` vs ``layer.1.convolution``). A longer suffix,
# which includes the index, is what tells them apart.
_AMBIGUOUS_META_LEAVES = frozenset(
    {
        "convolution",
        "conv",
        "norm",
        "bn",
        "shortcut",
        "activation",
    }
)


# Ops that forward a tensor unchanged (or scale it). Once a real module
# next to them has a concrete spatial shape, they take that shape instead of
# the text-model ``[B, S, H]`` / ``[1]`` the symbolic pass left behind.
_SHAPE_PRESERVING_LABELS = frozenset(
    {
        "Multiply",
        "Add",
        "GELU",
        "ReLU",
        "SiLU",
        "Dropout",
        "Identity",
        "Mutually exclusive branch",
        "DropPath",
        "ConvNextDropPath",
    }
)


def _attach_conv_meta_shapes(graph: dict[str, Any], checkpoint: str | Path) -> None:
    """Stamp live per-module ``[B, C, H, W]`` onto the nodes a meta forward named.

    Symbolic inference sizes a CNN ``[B, S, H]`` (and a vision stem ``[B, S]
    int64``) because it has no spatial rule. A meta forward fed the model's own
    ``pixel_values`` captures every module's concrete in/out shapes, keyed by
    full module path. A node matches the capture whose dotted suffix appears,
    in order, in its id — ``convolution`` as well as ``dwconv`` — under its
    stamped ``meta_path``. The batch axis is symbolised; channels and spatial
    dims stay concrete. Boundaries of that module take its input and output,
    and a shape-preserving neighbour (the layer-scale multiply, the residual
    add) copies the concrete shape of its input.
    """
    from TraceLens.ModelUtils.meta_trace import trace_meta_module_io

    captures = trace_meta_module_io(checkpoint)
    if not captures:
        return
    by_path = {
        str(item.get("path") or ""): item for item in captures if item.get("path")
    }

    def _attr(attrs: list[dict[str, str]], key: str) -> dict[str, str] | None:
        return next((item for item in attrs if item.get("key") == key), None)

    def _set(attrs: list[dict[str, str]], key: str, value: str) -> None:
        existing = _attr(attrs, key)
        if existing is None:
            attrs.append({"key": key, "value": value})
        else:
            existing["value"] = value

    def _dims(shape: list[int]) -> list[str]:
        # Symbolise the batch axis; keep channels and spatial dims concrete.
        # Aliasing a channel of 128 onto ``S`` is what made a stage look textual.
        return ["B"] + [str(int(extent)) for extent in shape[1:]]

    def _shape_text(shape: list[int], dtype: str) -> str:
        return f"[{', '.join(_dims(shape))}] {dtype or 'float32'}"

    def _suffix_score(path: str, ident: str) -> int:
        segments = [segment for segment in path.split(".") if segment]
        for start in range(len(segments)):
            position = 0
            matched = True
            for segment in segments[start:]:
                found = ident.find(segment.lower(), position)
                if found < 0:
                    matched = False
                    break
                position = found + len(segment)
            if matched:
                return len(segments) - start
        return 0

    def _best(ident: str, prefix: str | None) -> dict[str, Any] | None:
        # The module path itself is reserved for the section's own boundaries.
        # A compute node has to name a descendant (``layer.0.convolution``), and
        # one shared leaf (``convolution``) is not enough to tell the three
        # bottleneck convs apart.
        best: dict[str, Any] | None = None
        best_score = 0
        for path, capture in by_path.items():
            if path == prefix:
                continue
            if prefix is not None and not path.startswith(prefix + "."):
                continue
            score = _suffix_score(path, ident)
            if score <= 0 or score < best_score:
                continue
            leaf = path.rsplit(".", 1)[-1]
            # ``convolution`` alone is shared by every bottleneck conv. The
            # index in front of it is what makes the match.
            if score == 1 and (leaf.isdigit() or leaf in _AMBIGUOUS_META_LEAVES):
                continue
            if score == best_score:
                continue
            best, best_score = capture, score
        return best

    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
    for node in nodes:
        attrs = node.get("attrs")
        if not isinstance(attrs, list):
            continue
        meta = _attr(attrs, "meta_path")
        prefix = meta["value"] if meta is not None else None
        synthetic = _attr(attrs, "synthetic")
        kind = None
        class_name = _attr(attrs, "class_name")
        op_type = _attr(attrs, "op_type")
        if class_name is not None and class_name.get("value") in _CONV_OP_TYPES:
            kind = class_name["value"]
        elif op_type is not None and op_type.get("value") in _CONV_OP_TYPES:
            kind = op_type["value"]
        # A permute's axes are on the node. Stamping a module output onto it
        # would freeze the channels-last and channels-first layouts together.
        if str(node.get("label") or "") == "Permute":
            continue
        if synthetic is not None and prefix in by_path:
            match = by_path[prefix]
            which = (
                "input_shape" if synthetic.get("value") == "@input" else "output_shape"
            )
        else:
            match = _best(str(node.get("id", "")).lower(), prefix)
            # A leading conv's meta path IS the module. No descendant carries
            # the geometry, so the section node takes the module itself.
            if match is None and kind is not None and prefix in by_path:
                match = by_path[prefix]
            which = "output_shape"
        if match is None or not match.get(which):
            continue
        dtype = str(match.get("dtype") or "float32")
        if "int" in dtype:
            dtype = "float32"
        _set(attrs, "output_shape", _shape_text(match[which], dtype))
        if which == "output_shape" and match.get("input_shape"):
            _set(attrs, "input_shapes", json.dumps([_dims(match["input_shape"])]))
        if kind is not None:
            _set(attrs, "op_type", kind)
        if match.get("weight_shape"):
            weight = ", ".join(str(int(dim)) for dim in match["weight_shape"])
            bias = 1 if match.get("bias") else 0
            transposed = 1 if match.get("transposed") else 0
            details = _attr(attrs, "details")
            if details is None or "conv_weight=" not in str(details.get("value")):
                _set(
                    attrs,
                    "details",
                    (
                        f"conv_weight=[{weight}]; conv_bias={bias}; "
                        f"conv_transposed={transposed}"
                    ),
                )

    by_id = {str(node.get("id")): node for node in nodes}

    def _concrete(node: dict[str, Any]) -> str | None:
        attrs = node.get("attrs")
        if not isinstance(attrs, list):
            return None
        current = _attr(attrs, "output_shape")
        if current is None:
            return None
        text = str(current.get("value") or "")
        if " S" in text or text.endswith("int64") or text.startswith("[1]"):
            return None
        if text.count(",") < 2:
            return None
        return text

    def _permute_dims(attrs: list[dict[str, str]]) -> list[int] | None:
        for key in ("raw_op", "details"):
            item = _attr(attrs, key)
            text = str(item.get("value") or "") if item is not None else ""
            marker = "permute dims:" if "permute dims:" in text else "dims:"
            if marker not in text:
                continue
            text = text.split(marker, 1)[1].split(";", 1)[0]
            try:
                return [int(part.strip()) for part in text.split(",") if part.strip()]
            except ValueError:
                return None
        return None

    def _permuted(text: str, dims: list[int]) -> str | None:
        head, sep, tail = text.partition("]")
        if not sep or not head.startswith("["):
            return None
        axes = [part.strip() for part in head[1:].split(",")]
        if len(dims) != len(axes) or any(
            index >= len(axes) or index < 0 for index in dims
        ):
            return None
        return "[" + ", ".join(axes[index] for index in dims) + "]" + tail

    for _pass in range(6):
        changed = False
        for node in nodes:
            label = str(node.get("label") or "")
            attrs = node.get("attrs")
            if not isinstance(attrs, list):
                continue
            synthetic = _attr(attrs, "synthetic")
            mirror = synthetic is not None and str(synthetic.get("value")) in {
                "@output_mirror",
                "@input_mirror",
            }
            if label == "Permute":
                dims = _permute_dims(attrs)
                sources = [
                    _concrete(by_id[str(edge.get("sourceNodeId"))])
                    for edge in node.get("incomingEdges") or []
                    if str(edge.get("sourceNodeId")) in by_id
                ]
                sources = [shape for shape in sources if shape]
                if dims is None or len(sources) != 1:
                    continue
                updated = _permuted(sources[0], dims)
                current = _attr(attrs, "output_shape")
                if updated and (current is None or current.get("value") != updated):
                    _set(attrs, "output_shape", updated)
                    changed = True
                continue
            if label not in _SHAPE_PRESERVING_LABELS and not mirror:
                continue
            incoming = [
                _concrete(by_id[str(edge.get("sourceNodeId"))])
                for edge in node.get("incomingEdges") or []
                if str(edge.get("sourceNodeId")) in by_id
            ]
            incoming = [shape for shape in incoming if shape]
            if not incoming:
                continue
            if any(shape != incoming[0] for shape in incoming):
                # A scale multiply: keep the activation, not the rank-1 factor.
                if label != "Multiply":
                    continue
                ranked = sorted(incoming, key=lambda shape: shape.count(","))
                if ranked[-1].count(",") < 2:
                    continue
                incoming = [ranked[-1]]
            current = _attr(attrs, "output_shape")
            if current is not None and current.get("value") == incoming[0]:
                continue
            _set(attrs, "output_shape", incoming[0])
            changed = True
        if not changed:
            break


def build_model_explorer_payload(
    spec: ArchitectureSpec,
    *,
    basic_ops: BasicOpFilter | None = None,
    collection_label: str | None = None,
    include_shapes: bool = True,
    include_operator_export: bool = False,
    inline_expansion: bool = True,
    meta_shapes_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Build a single merged Model Explorer graph with in-place namespace expansion."""
    resolved_basic_ops = basic_ops or spec.basic_ops
    inferencer = (
        ShapeInferencer(spec) if include_shapes or include_operator_export else None
    )
    if inferencer is not None and meta_shapes_checkpoint is not None:
        inferencer.load_meta_shapes(meta_shapes_checkpoint)
    if meta_shapes_checkpoint is not None:
        # Drive the N× repeated-block grouping (count, banner class, sub-variant
        # counts) from the live meta module tree instead of config ints + role
        # regexes. This runs before the graph/fact-sheet read the spec. It is
        # independent of load_meta_shapes (structure needs only instantiation, never
        # the fragile forward); reconcile is a no-op when instantiation fails.
        from TraceLens.ModelUtils.extract import reconcile_live_module_groups
        from TraceLens.ModelUtils.meta_trace import walk_meta_module_tree

        reconcile_live_module_groups(
            spec, walk_meta_module_tree(meta_shapes_checkpoint)
        )
    graph = build_merged_model_graph(
        spec,
        basic_ops=_export_basic_ops(resolved_basic_ops),
        shape_inferencer=inferencer,
        inline_expansion=inline_expansion,
    )
    # A primitive op's role comes from its own computation, not from the name of
    # the module that holds it. The attention score/softmax/value path is read
    # off the dataflow here, once the edges are final.
    stamp_computation_roles(graph)
    if spec.nested_module_groups and meta_shapes_checkpoint is not None:
        # A non-uniform CNN's convs are emitted per stage, but symbolic [B, S, H]
        # inference cannot size them. Overwrite with the concrete per-instance
        # [B, C, H, W] a meta-device forward captures.
        _attach_conv_meta_shapes(graph, meta_shapes_checkpoint)
    layout = getattr(spec, "parallel_layout", None)
    if layout is not None:
        from TraceLens.ModelUtils.parallel_plan import annotate_parallel_nodes
        from TraceLens.Visualizer.model_explorer_export.shapes import (
            refresh_displayed_group_shapes,
        )

        if layout.degree.requested() and inferencer is not None:
            _merge_local_shapes(
                graph,
                spec,
                layout,
                basic_ops=_export_basic_ops(resolved_basic_ops),
                inline_expansion=inline_expansion,
                meta_shapes_checkpoint=meta_shapes_checkpoint,
            )
        annotate_parallel_nodes(graph, layout)
        # The build already published shapes and checked the graph. These nodes
        # did not exist then, so both have to run again or an edge leaves a
        # collective with no shape and the viewer draws "?".
        from TraceLens.Visualizer.model_explorer_export.merge import (
            publish_stated_shapes_and_check,
        )

        publish_stated_shapes_and_check(graph, label="parallel")
        if layout.degree.requested():
            refresh_displayed_group_shapes(graph)

    viewer_meta: dict[str, Any] = {
        "factSheet": build_fact_sheet_viewer(spec),
    }
    if inferencer is not None:
        viewer_meta["dimensions"] = {
            key: serialize_dim(value) for key, value in inferencer.context.dims.items()
        }
        viewer_meta["dtype"] = inferencer.context.dtype
    if include_operator_export and inferencer is not None:
        viewer_meta["operatorExport"] = inferencer.export_architecture()

    return {
        "name": spec.name,
        "model_type": spec.model_type,
        "source": "tracelens-computation-graph",
        "tracelensViewer": viewer_meta,
        "graphCollections": [
            {
                "label": collection_label or spec.name,
                "graphs": [graph],
            }
        ],
    }


def _merge_local_shapes(
    full_graph: dict[str, Any],
    spec: ArchitectureSpec,
    layout: Any,
    *,
    basic_ops: BasicOpFilter,
    inline_expansion: bool,
    meta_shapes_checkpoint: str | None,
) -> None:
    """Overlay this rank's local shapes onto *full_graph*, by node id.

    *full_graph* already carries ``full_shape == output_shape`` on every node
    (``stamp_full_shapes``, inside ``build_merged_model_graph``). A second
    graph, built the same way from a config with
    :func:`TraceLens.ModelUtils.parallel_plan.local_config_overrides`'s divided
    widths applied, gives the local ``output_shape`` for the same nodes through
    the real shape-inference machinery -- the only way to get those sizes right
    through a reshape/transpose/attention chain without guessing which axis is
    the sharded one (see ``parallel_plan._localize_gpu_shapes``). ``full_shape``
    is left exactly as the full build already stamped it.
    """
    from TraceLens.ModelUtils.extract import rebuild_architecture_with_overrides
    from TraceLens.ModelUtils.parallel_plan import (
        local_config_overrides,
        locally_sharded_node_ids,
    )

    overrides = local_config_overrides(layout, spec)
    if not overrides:
        return
    sharded_ids = locally_sharded_node_ids(full_graph, layout)
    local_spec = rebuild_architecture_with_overrides(spec, overrides)
    local_inferencer = ShapeInferencer(local_spec)
    if meta_shapes_checkpoint is not None:
        local_inferencer.load_meta_shapes(meta_shapes_checkpoint)
    local_graph = build_merged_model_graph(
        local_spec,
        basic_ops=basic_ops,
        shape_inferencer=local_inferencer,
        inline_expansion=inline_expansion,
    )
    local_nodes = {
        node.get("id"): node
        for node in local_graph.get("nodes") or []
        if isinstance(node, dict)
    }
    full_nodes = [
        node for node in full_graph.get("nodes") or [] if isinstance(node, dict)
    ]
    missing = [node["id"] for node in full_nodes if node.get("id") not in local_nodes]
    if missing:
        # A locally-sharded config changing the graph's own shape (reshape target,
        # branch condition, ...) rather than just a tensor's size would make this
        # merge-by-id meaningless; surface that loudly instead of pairing the
        # wrong nodes. Pipeline parallelism, which divides num_hidden_layers,
        # is the one override here that can legitimately do this when
        # ArchitectureSpec.layer_variants is not uniform across layers (a stage
        # with a smaller layer count can see a different variant mix than the
        # full model's own representative merge).
        preview = ", ".join(str(item) for item in missing[:5])
        raise ValueError(
            "local-shape graph diverged from the full graph's structure "
            f"(missing node ids: {preview}"
            f"{', ...' if len(missing) > 5 else ''}); cannot merge per-GPU "
            "shapes onto a different topology"
        )
    for node in full_nodes:
        if node["id"] in sharded_ids:
            _overlay_local_output_shape(node, local_nodes[node["id"]])


def _overlay_local_output_shape(
    node: dict[str, Any], local_node: dict[str, Any]
) -> None:
    local_shape = _attr_value(local_node.get("attrs"), "output_shape")
    if local_shape:
        _set_attr_value(node.setdefault("attrs", []), "output_shape", local_shape)
    # ``input_shapes`` is a per-node snapshot the type-checker reads directly
    # (not re-derived from the producer's current ``output_shape``), so a
    # consumer needs its own copy refreshed too.
    local_input_shapes = _attr_value(local_node.get("attrs"), "input_shapes")
    if local_input_shapes:
        _set_attr_value(
            node.setdefault("attrs", []), "input_shapes", local_input_shapes
        )
    for key in ("outputsMetadata", "inputsMetadata"):
        local_by_id = {
            item.get("id"): item
            for item in local_node.get(key) or []
            if isinstance(item, dict)
        }
        for item in node.get(key) or []:
            if not isinstance(item, dict):
                continue
            local_item = local_by_id.get(item.get("id"))
            if local_item is None:
                continue
            local_port_shape = _attr_value(local_item.get("attrs"), "shape")
            if local_port_shape:
                _set_attr_value(item.setdefault("attrs", []), "shape", local_port_shape)


def _attr_value(attrs: Any, key: str) -> str | None:
    if not isinstance(attrs, list):
        return None
    return next(
        (item.get("value") for item in attrs if item.get("key") == key),
        None,
    )


def _set_attr_value(attrs: list[Any], key: str, value: str) -> None:
    for item in attrs:
        if isinstance(item, dict) and item.get("key") == key:
            item["value"] = value
            return
    attrs.append({"key": key, "value": value})


def build_operator_export_payload(spec: ArchitectureSpec) -> dict[str, Any]:
    """Build the flat operator/shape JSON used by tooling and tests."""
    return build_operator_export(spec)


def save_model_explorer_payload(payload: dict[str, Any], path: Path | str) -> Path:
    """Write a Model Explorer payload to JSON."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
