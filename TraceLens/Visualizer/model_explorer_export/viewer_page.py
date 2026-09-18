###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compose TraceLens Model Explorer viewer HTML pages."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from TraceLens.Visualizer.model_explorer_export.fact_sheet import (
    with_generated_timestamp,
)
from TraceLens.Visualizer.model_explorer_export.type_check import (
    integrity_check_graph_nodes,
)

VIEWER_DIR = Path(__file__).resolve().parent / "viewer"
PACKAGE_ROOT = Path(__file__).resolve().parent
VISUALIZER_DIST = (
    PACKAGE_ROOT / "node_modules" / "ai-edge-model-explorer-visualizer" / "dist"
)
APP_JS_PATTERN = re.compile(r'    <script src="\./app\.js\?v=\d+"></script>')


def is_html_output(path: Path | str) -> bool:
    return Path(path).suffix.lower() == ".html"


def render_payload_script(payload: dict[str, Any]) -> str:
    """Embed export JSON safely inside HTML."""
    blob = json.dumps(payload, ensure_ascii=False)
    blob = blob.replace("</", "<\\/")
    return f'    <script id="tracelens-payload" type="application/json">{blob}</script>'


def _worker_js_source() -> Path:
    for candidate in (VISUALIZER_DIST / "worker.js", VIEWER_DIR / "worker.js"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "worker.js not found; install ai-edge-model-explorer-visualizer or bundle viewer/worker.js"
    )


def render_worker_script(worker_js: str) -> str:
    """Embed layout worker source for file:// exports (Workers cannot load local paths)."""
    safe = worker_js.replace("</", "<\\/")
    return f'    <script id="tracelens-worker-source" type="text/plain">{safe}</script>'


def _payload_with_generated_timestamp(
    payload: dict[str, Any], generated_at: datetime
) -> dict[str, Any]:
    """Overlay a 'Generated' timestamp on the fact sheet without mutating ``payload``.

    Only the viewer fact sheet is touched; the embedded graph/payload data is
    left identical so the persisted JSON stays byte-reproducible.
    """
    viewer = payload.get("tracelensViewer")
    if not isinstance(viewer, dict):
        return payload
    fact_sheet = viewer.get("factSheet")
    if not isinstance(fact_sheet, dict):
        return payload
    stamped = with_generated_timestamp(fact_sheet, generated_at)
    new_viewer = dict(viewer, factSheet=stamped)
    return dict(payload, tracelensViewer=new_viewer)


def _node_is_constant(node: dict[str, Any]) -> bool:
    """A node tagged ``constant`` (a learned weight / buffer / constant operand)."""
    for attr in node.get("attrs", []) or []:
        if attr.get("key") == "constant" and attr.get("value") == "true":
            return True
    return False


def _namespace_prefixes(namespace: str) -> set[str]:
    """Every ancestor namespace path of ``namespace`` (inclusive), plus root ``""``."""
    prefixes: set[str] = {""}
    if not namespace:
        return prefixes
    segments = namespace.split("/")
    for depth in range(1, len(segments) + 1):
        prefixes.add("/".join(segments[:depth]))
    return prefixes


def _graph_without_constants(graph: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of one Model Explorer graph with ``constant`` nodes removed.

    Constant nodes and any incoming edge that references them are dropped, and
    ``groupNodeAttributes`` entries for namespaces that no longer contain a node
    are pruned so an empty group frame is not drawn. The input is not mutated.
    """
    dropped_ids = {
        node.get("id")
        for node in graph.get("nodes", [])
        if _node_is_constant(node)
    }
    if not dropped_ids:
        return graph

    kept_nodes: list[dict[str, Any]] = []
    for node in graph.get("nodes", []):
        if _node_is_constant(node):
            continue
        edges = node.get("incomingEdges")
        if edges:
            filtered = [e for e in edges if e.get("sourceNodeId") not in dropped_ids]
            if len(filtered) != len(edges):
                node = dict(node)
                if filtered:
                    node["incomingEdges"] = filtered
                else:
                    node.pop("incomingEdges", None)
        kept_nodes.append(node)

    new_graph = dict(graph, nodes=kept_nodes)

    group_attrs = graph.get("groupNodeAttributes")
    if isinstance(group_attrs, dict):
        live_namespaces: set[str] = set()
        for node in kept_nodes:
            live_namespaces |= _namespace_prefixes(node.get("namespace", ""))
        pruned = {ns: cfg for ns, cfg in group_attrs.items() if ns in live_namespaces}
        if len(pruned) != len(group_attrs):
            new_graph["groupNodeAttributes"] = pruned

    # Re-run the structural-integrity check on the render-filtered graph: dropping
    # constants can orphan a survivor that lost its only constant producer, so this
    # is exactly where I1/I2 catch a mistagged/under-propagated constant closure
    # (the attn_hc slice-tile regression). Warnings only.
    integrity_check_graph_nodes(kept_nodes, label="render-filtered")
    return new_graph


def _payload_without_constants(payload: dict[str, Any]) -> dict[str, Any]:
    """Overlay a constant-free view of every graph without mutating ``payload``.

    Only the graphs are rebuilt; the persisted JSON keeps all constant nodes so
    the artifact stays complete. Used at HTML render time so the drawn picture
    contains no constants (the owner rule "never show constants", applied in
    rendering rather than by deleting data).
    """
    collections = payload.get("graphCollections")
    if not isinstance(collections, list):
        return payload
    new_collections: list[Any] = []
    for collection in collections:
        graphs = collection.get("graphs") if isinstance(collection, dict) else None
        if not isinstance(graphs, list):
            new_collections.append(collection)
            continue
        new_graphs = [
            _graph_without_constants(graph) if isinstance(graph, dict) else graph
            for graph in graphs
        ]
        new_collections.append(dict(collection, graphs=new_graphs))
    return dict(payload, graphCollections=new_collections)


def compose_viewer_html(
    payload: dict[str, Any] | None = None,
    *,
    inline_app: bool = False,
    generated_at: datetime | None = None,
    drop_constants: bool = True,
) -> str:
    """Build viewer HTML, optionally embedding payload and app.js.

    When ``generated_at`` is provided, a 'Generated: <date time>' line is added
    to the viewer fact sheet (HTML only; the payload JSON is unchanged). When
    ``drop_constants`` is true (default), ``constant``-tagged nodes are filtered
    out of the embedded payload so the rendered graph shows no constants; the
    persisted JSON is unaffected.
    """
    shell = (VIEWER_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (VIEWER_DIR / "app.js").read_text(encoding="utf-8")

    replacement_parts: list[str] = []
    if payload is not None:
        if generated_at is not None:
            payload = _payload_with_generated_timestamp(payload, generated_at)
        if drop_constants:
            payload = _payload_without_constants(payload)
        replacement_parts.append(render_payload_script(payload))
    if inline_app:
        replacement_parts.append(
            render_worker_script(_worker_js_source().read_text(encoding="utf-8"))
        )
        replacement_parts.append(f"    <script>\n{app_js}\n    </script>")
    else:
        replacement_parts.append('    <script src="./app.js?v=10"></script>')

    if replacement_parts:
        replacement = "\n".join(replacement_parts)
        shell, count = APP_JS_PATTERN.subn(lambda _match: replacement, shell, count=1)
        if count != 1:
            raise RuntimeError("Viewer shell is missing the app.js script tag.")

    return shell


def save_viewer_html(
    payload: dict[str, Any],
    path: Path | str,
    *,
    generated_at: datetime | None = None,
    drop_constants: bool = True,
) -> Path:
    """Write a self-contained standalone viewer page (payload, worker, and app inline).

    ``generated_at`` stamps the fact sheet with a generation time (HTML only).
    ``drop_constants`` (default true) filters ``constant`` nodes out of the
    embedded payload; the persisted JSON is unaffected.
    """
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        compose_viewer_html(
            payload,
            inline_app=True,
            generated_at=generated_at,
            drop_constants=drop_constants,
        ),
        encoding="utf-8",
    )
    return target
