"""Build Model Explorer payloads from TraceLens architecture specs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visualizer.basic_ops import BasicOpFilter
from visualizer.extract import ArchitectureSpec
from visualizer.shape_inference import ShapeInferencer, build_operator_export, serialize_dim

from model_explorer_export.fact_sheet import build_fact_sheet_viewer
from model_explorer_export.merge import build_merged_model_graph


def _export_basic_ops(basic_ops: BasicOpFilter) -> BasicOpFilter:
    """Keep full kernel inline substeps in Model Explorer (not basic-only tails)."""
    return BasicOpFilter(basic_ops.patterns, basic_only=False)


def build_model_explorer_payload(
    spec: ArchitectureSpec,
    *,
    basic_ops: BasicOpFilter | None = None,
    collection_label: str | None = None,
    include_shapes: bool = False,
    include_operator_export: bool = False,
) -> dict[str, Any]:
    """Build a single merged Model Explorer graph with in-place namespace expansion."""
    resolved_basic_ops = basic_ops or spec.basic_ops
    inferencer = ShapeInferencer(spec) if include_shapes or include_operator_export else None
    graph = build_merged_model_graph(
        spec,
        basic_ops=_export_basic_ops(resolved_basic_ops),
        shape_inferencer=inferencer,
    )

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


def build_operator_export_payload(spec: ArchitectureSpec) -> dict[str, Any]:
    """Build the flat operator/shape JSON used by tooling and tests."""
    return build_operator_export(spec)


def save_model_explorer_payload(payload: dict[str, Any], path: Path | str) -> Path:
    """Write a Model Explorer payload to JSON."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
