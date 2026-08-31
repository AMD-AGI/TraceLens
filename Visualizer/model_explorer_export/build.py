"""Build Model Explorer payloads from TraceLens architecture specs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visualizer.basic_ops import BasicOpFilter
from visualizer.extract import ArchitectureSpec

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
) -> dict[str, Any]:
    """Build a single merged Model Explorer graph with in-place namespace expansion."""
    resolved_basic_ops = basic_ops or spec.basic_ops
    graph = build_merged_model_graph(spec, basic_ops=_export_basic_ops(resolved_basic_ops))

    return {
        "name": spec.name,
        "model_type": spec.model_type,
        "source": "tracelens-computation-graph",
        "tracelensViewer": {
            "factSheet": build_fact_sheet_viewer(spec),
        },
        "graphCollections": [
            {
                "label": collection_label or spec.name,
                "graphs": [graph],
            }
        ],
    }


def save_model_explorer_payload(payload: dict[str, Any], path: Path | str) -> Path:
    """Write a Model Explorer payload to JSON."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
