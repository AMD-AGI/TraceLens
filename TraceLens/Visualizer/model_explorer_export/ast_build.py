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

from TraceLens.Visualizer.model_explorer_export.fact_sheet import (
    build_fact_sheet_viewer,
)
from TraceLens.Visualizer.model_explorer_export.merge import build_merged_model_graph


def _export_basic_ops(basic_ops: BasicOpFilter) -> BasicOpFilter:
    """Keep full kernel inline substeps in Model Explorer (not basic-only tails)."""
    return BasicOpFilter(basic_ops.patterns, basic_only=False)


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


def _attach_conv_meta_shapes(graph: dict[str, Any], checkpoint: str | Path) -> None:
    """Overwrite each conv node's shapes with the live per-instance ``[B, C, H, W]``.

    A non-uniform CNN's convolutions are emitted one representative per stage; the
    symbolic text shape inference sizes them ``[B, S, H]`` because it has no spatial
    rule. A meta-device forward (fed the model's own ``pixel_values``) captures each
    conv's concrete in/out shapes keyed by full module path -- the authoritative
    source. Match a node to its capture by its stamped ``meta_path`` (the stage) plus
    the conv attr in its id (``dwconv``/``patch_embeddings``), symbolise the batch
    axis, and stamp ``output_shape``/``input_shapes`` so the viewer and the graph
    roofline read real spatial dims.
    """
    from TraceLens.ModelUtils.meta_trace import trace_meta_conv_ops

    captures = trace_meta_conv_ops(checkpoint)
    if not captures:
        return

    def _attr(attrs: list[dict[str, str]], key: str) -> dict[str, str] | None:
        return next((item for item in attrs if item.get("key") == key), None)

    def _set(attrs: list[dict[str, str]], key: str, value: str) -> None:
        existing = _attr(attrs, key)
        if existing is None:
            attrs.append({"key": key, "value": value})
        else:
            existing["value"] = value

    def _dims(shape: list[int]) -> list[str]:
        # Symbolise the batch axis; keep channels/spatial concrete.
        return ["B"] + [str(int(extent)) for extent in shape[1:]]

    conv_leaves = ("dwconv", "patch_embeddings", "downsample", "conv")
    for node in graph.get("nodes", []):
        attrs = node.get("attrs")
        if not isinstance(attrs, list):
            continue
        op_type = _attr(attrs, "op_type")
        if op_type is None or op_type.get("value") not in _CONV_OP_TYPES:
            continue
        node_id = str(node.get("id", "")).lower()
        leaf = next((name for name in conv_leaves if name in node_id), None)
        meta_path = _attr(attrs, "meta_path")
        prefix = meta_path["value"] if meta_path is not None else None
        match = next(
            (
                capture
                for capture in captures
                if (prefix is None or capture["path"].startswith(prefix))
                and (leaf is None or capture["path"].rsplit(".", 1)[-1] == leaf)
            ),
            None,
        )
        if match is None:
            continue
        dtype = "float32"
        current = _attr(attrs, "output_shape")
        if current is not None and " " in str(current.get("value", "")):
            # Keep a real float dtype (bf16/fp16); never a token int64 placeholder.
            existing_dtype = str(current["value"]).rsplit(" ", 1)[-1]
            if "float" in existing_dtype or existing_dtype.startswith(("bf", "fp")):
                dtype = existing_dtype
        _set(
            attrs,
            "output_shape",
            f"[{', '.join(_dims(match['output_shape']))}] {dtype}",
        )
        if match.get("input_shape"):
            _set(attrs, "input_shapes", json.dumps([_dims(match["input_shape"])]))


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
