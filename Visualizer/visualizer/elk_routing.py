"""ELK-based orthogonal connector routing for detail computation graphs.

Migration roadmap (legacy router removal)
-----------------------------------------
Phase 1 (done): Route simple 1:1 links with elkjs; helper fallback on failure.
Phase 2 (done): Spread-merge ports via ``port.anchor``; multi-incoming on ELK.
Phase 2b (done): Inline-frame compound nodes with layered hierarchy.
Phase 3 (done): Unified ``route_detail_link_paths()`` entry point.
Phase 4 (done): Fan-out / merge-bus links via render bus helpers (not ELK).
Phase 5 (done): Routing path no longer calls ``_legacy_link_path``.
Phase 6 (done): ``_collect_detail_link_paths`` uses route + shrinkwrap + validation.
Phase 7 (done): Legacy per-link router moved to ``connector_routing.py``.
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from visualizer.sizing import PIXELS_PER_UNIT

if TYPE_CHECKING:
    from visualizer.computation_graph import ComputationGraph, InlineFrameSpec

MIGRATION_PHASE = 7

ELK_PAD = 12.0
LAYER_Y_EPS = 0.04
_PORT_EPS = 1e-6


@dataclass
class RoutingAudit:
    """Counts how many links ELK routed vs legacy fallback."""

    elk: list[tuple[int, int]] = field(default_factory=list)
    helper: list[tuple[int, int]] = field(default_factory=list)

    @property
    def elk_count(self) -> int:
        return len(self.elk)

    @property
    def helper_count(self) -> int:
        return len(self.helper)

    @property
    def legacy_count(self) -> int:
        """Alias for ``helper_count`` (pre-migration name)."""
        return self.helper_count

    @property
    def helper_fraction(self) -> float:
        total = self.elk_count + self.helper_count
        return self.helper_count / total if total else 0.0

    @property
    def legacy_fraction(self) -> float:
        return self.helper_fraction


def _visualizer_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _elk_script_path() -> Path:
    return Path(__file__).resolve().parent / "elk_route.mjs"


def _node_binary() -> str:
    import shutil

    return shutil.which("node") or "node"


def _run_elk_layout(graph: dict[str, Any]) -> dict[str, Any]:
    script = _elk_script_path()
    root = _visualizer_root()
    node_modules = root / "node_modules" / "elkjs"
    if not node_modules.is_dir():
        raise RuntimeError(
            "elkjs is not installed; run `npm install` in the Visualizer directory"
        )
    proc = subprocess.run(
        [_node_binary(), str(script)],
        input=json.dumps(graph),
        capture_output=True,
        text=True,
        cwd=str(root),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "elkjs layout failed:\n"
            + (proc.stderr or proc.stdout or "unknown error")
        )
    return json.loads(proc.stdout)


def _layer_groups(positions: list) -> list[list[int]]:
    order = sorted(
        range(len(positions)),
        key=lambda index: (-positions[index].top_y, positions[index].cx),
    )
    groups: list[list[int]] = []
    current: list[int] = []
    anchor_y: float | None = None
    for index in order:
        top_y = positions[index].top_y
        if anchor_y is None or abs(top_y - anchor_y) <= LAYER_Y_EPS:
            current.append(index)
            anchor_y = top_y if anchor_y is None else anchor_y
            continue
        groups.append(current)
        current = [index]
        anchor_y = top_y
    if current:
        groups.append(current)
    return groups


def _diagram_y_offset(positions: list) -> float:
    max_top = max(pos.top_y for pos in positions)
    return max_top * PIXELS_PER_UNIT + ELK_PAD * 2


def _diagram_to_elk_x(x: float) -> float:
    return x * PIXELS_PER_UNIT + ELK_PAD


def _diagram_to_elk_y(y_up: float, *, y_offset: float) -> float:
    return y_offset - y_up * PIXELS_PER_UNIT


def _elk_to_diagram_x(elk_x: float) -> float:
    return (elk_x - ELK_PAD) / PIXELS_PER_UNIT


def _elk_to_diagram_y(elk_y: float, *, y_offset: float) -> float:
    return (y_offset - elk_y) / PIXELS_PER_UNIT


def _north_port_anchor(anchor, entry_x: float | None) -> str:
    width = max(anchor.right - anchor.left, _PORT_EPS)
    if entry_x is None:
        rel = 0.5
    else:
        rel = (entry_x - anchor.left) / width
    rel = min(1.0, max(0.0, rel))
    return f"({rel:.6f},0)"


def _member_to_frame(graph: ComputationGraph) -> dict[int, InlineFrameSpec]:
    mapping: dict[int, InlineFrameSpec] = {}
    for frame in graph.inline_frames:
        for index in frame.node_indices:
            mapping[index] = frame
    return mapping


def _elk_node_child(
    *,
    index: int,
    anchor,
    graph: ComputationGraph,
    layer_of: dict[int, int],
    last_layer: int,
    y_offset: float,
    origin_elk_x: float = 0.0,
    origin_elk_y: float = 0.0,
) -> dict[str, Any]:
    layer = layer_of.get(index, 0)
    if layer == 0:
        layer_constraint = "FIRST"
    elif layer == last_layer:
        layer_constraint = "LAST"
    else:
        layer_constraint = "NONE"
    width = max(anchor.right - anchor.left, 0.05) * PIXELS_PER_UNIT
    height = max(anchor.top - anchor.bottom, 0.05) * PIXELS_PER_UNIT
    elk_x = _diagram_to_elk_x(anchor.left) - origin_elk_x
    elk_y = _diagram_to_elk_y(anchor.top, y_offset=y_offset) - origin_elk_y
    label = graph.nodes[index].label if index < len(graph.nodes) else str(index)
    return {
        "id": f"n{index}",
        "width": width,
        "height": height,
        "x": elk_x,
        "y": elk_y,
        "labels": [{"text": label}],
        "layoutOptions": {
            "org.eclipse.elk.layered.layering.layerConstraint": layer_constraint,
            "org.eclipse.elk.position": f"({elk_x:.3f},{elk_y:.3f})",
        },
        "ports": [
            {
                "id": f"n{index}_out",
                "width": 1,
                "height": 1,
                "layoutOptions": {"port.side": "SOUTH"},
            },
            {
                "id": f"n{index}_in",
                "width": 1,
                "height": 1,
                "layoutOptions": {"port.side": "NORTH", "port.index": 0},
            },
        ],
    }


def _innermost_frame_for_node(
    graph: ComputationGraph,
    index: int,
) -> InlineFrameSpec | None:
    containing = [
        frame
        for frame in graph.inline_frames
        if index in frame.node_indices
    ]
    if not containing:
        return None
    return min(containing, key=lambda frame: len(frame.node_indices))


def _outermost_inline_frames(graph: ComputationGraph) -> list[InlineFrameSpec]:
    frames = list(graph.inline_frames)
    nested_ids = {
        nested.frame_id
        for frame in frames
        for nested in frames
        if nested.frame_id != frame.frame_id
        and set(nested.node_indices) < set(frame.node_indices)
    }
    return [frame for frame in frames if frame.frame_id not in nested_ids]


def _nested_inline_frames(graph: ComputationGraph, frame: InlineFrameSpec) -> list[InlineFrameSpec]:
    members = set(frame.node_indices)
    return [
        other
        for other in graph.inline_frames
        if other.frame_id != frame.frame_id and set(other.node_indices) < members
    ]


def _build_frame_elk_compound(
    *,
    frame: InlineFrameSpec,
    graph: ComputationGraph,
    positions: list,
    anchors: dict[int, Any],
    layer_of: dict[int, int],
    last_layer: int,
    y_offset: float,
) -> dict[str, Any]:
    from visualizer.render import _inline_frame_draw_bounds

    bounds = _inline_frame_draw_bounds(frame, positions, graph)
    origin_elk_x = _diagram_to_elk_x(bounds.left)
    origin_elk_y = _diagram_to_elk_y(bounds.top, y_offset=y_offset)
    width = max((bounds.right - bounds.left) * PIXELS_PER_UNIT, 1.0)
    height = max((bounds.top - bounds.bottom) * PIXELS_PER_UNIT, 1.0)

    nested_indices = {
        index
        for nested in _nested_inline_frames(graph, frame)
        for index in nested.node_indices
    }
    children: list[dict[str, Any]] = []
    for nested in _nested_inline_frames(graph, frame):
        children.append(
            _build_frame_elk_compound(
                frame=nested,
                graph=graph,
                positions=positions,
                anchors=anchors,
                layer_of=layer_of,
                last_layer=last_layer,
                y_offset=y_offset,
            )
        )
    for index in frame.node_indices:
        if index not in anchors or index in nested_indices:
            continue
        children.append(
            _elk_node_child(
                index=index,
                anchor=anchors[index],
                graph=graph,
                layer_of=layer_of,
                last_layer=last_layer,
                y_offset=y_offset,
                origin_elk_x=origin_elk_x,
                origin_elk_y=origin_elk_y,
            )
        )
    return {
        "id": f"frame_{frame.frame_id}",
        "width": width,
        "height": height,
        "x": origin_elk_x,
        "y": origin_elk_y,
        "labels": [{"text": frame.label}],
        "layoutOptions": {
            "elk.algorithm": "layered",
            "elk.direction": "DOWN",
            "elk.edgeRouting": "ORTHOGONAL",
            "elk.layered.nodePlacement.strategy": "INTERACTIVE",
            "elk.padding": "[top=8,left=8,bottom=8,right=8]",
        },
        "children": children,
    }


def _build_elk_graph(
    *,
    graph: ComputationGraph,
    positions: list,
    anchors: dict[int, Any],
    links: list[tuple[int, int]],
    merge_entry_x: dict[tuple[int, int], float],
) -> dict[str, Any]:
    """Build an ELK graph with inline frames as layered compound nodes."""
    layer_groups = _layer_groups(positions)
    layer_of = {
        index: layer_index
        for layer_index, group in enumerate(layer_groups)
        for index in group
    }
    last_layer = max(layer_of.values()) if layer_of else 0
    y_offset = _diagram_y_offset(positions)

    framed_indices: set[int] = set()
    children: list[dict[str, Any]] = []
    for frame in _outermost_inline_frames(graph):
        member_indices = [index for index in frame.node_indices if index in anchors]
        if not member_indices:
            continue
        framed_indices.update(member_indices)
        children.append(
            _build_frame_elk_compound(
                frame=frame,
                graph=graph,
                positions=positions,
                anchors=anchors,
                layer_of=layer_of,
                last_layer=last_layer,
                y_offset=y_offset,
            )
        )

    for index, anchor in anchors.items():
        if index in framed_indices:
            continue
        children.append(
            _elk_node_child(
                index=index,
                anchor=anchor,
                graph=graph,
                layer_of=layer_of,
                last_layer=last_layer,
                y_offset=y_offset,
            )
        )

    edges: list[dict[str, Any]] = []
    for edge_index, (src, tgt) in enumerate(links):
        link_key = (src, tgt)
        target = anchors[tgt]
        entry_x = merge_entry_x.get(link_key)
        target_port: dict[str, Any] = {
            "id": f"n{tgt}_in_{edge_index}",
            "width": 1,
            "height": 1,
            "layoutOptions": {
                "port.side": "NORTH",
                "port.index": edge_index,
                "port.anchor": _north_port_anchor(target, entry_x),
            },
        }
        _attach_port(children, f"n{tgt}", target_port)
        edges.append(
            {
                "id": f"e{edge_index}_{src}_{tgt}",
                "sources": [f"n{src}_out"],
                "targets": [target_port["id"]],
            }
        )

    return {
        "id": "root",
        "layoutOptions": {
            "elk.algorithm": "layered",
            "elk.direction": "DOWN",
            "elk.edgeRouting": "ORTHOGONAL",
            "elk.hierarchyHandling": "INCLUDE_CHILDREN",
            "elk.layered.nodePlacement.strategy": "INTERACTIVE",
            "elk.layered.layering.strategy": "LONGEST_PATH",
            "elk.spacing.nodeNode": "24",
            "elk.layered.spacing.nodeNodeBetweenLayers": "28",
            "elk.spacing.edgeNode": "14",
            "elk.spacing.edgeEdge": "12",
            "elk.padding": "[top=12,left=12,bottom=12,right=12]",
        },
        "children": children,
        "edges": edges,
        "_meta": {"y_offset": y_offset},
    }


def _attach_port(children: list[dict[str, Any]], node_id: str, port: dict[str, Any]) -> None:
    for child in children:
        if child.get("id") == node_id:
            child.setdefault("ports", []).append(port)
            return
        nested = child.get("children", ())
        if nested:
            _attach_port(list(nested), node_id, port)


def _sections_to_diagram_path(
    sections: list[dict[str, Any]] | None,
    *,
    source,
    target,
    entry_x: float | None,
    y_offset: float,
) -> list[tuple[float, float]]:
    from visualizer.render import _ensure_orthogonal_connector_path

    if not sections:
        return [
            (source.cx, source.bottom),
            (entry_x or target.cx, target.top),
        ]
    raw: list[tuple[float, float]] = [(source.cx, source.bottom)]
    for section in sections:
        for point in section.get("bendPoints", ()) or ():
            raw.append(
                (
                    _elk_to_diagram_x(float(point["x"])),
                    _elk_to_diagram_y(float(point["y"]), y_offset=y_offset),
                )
            )
        end = section.get("endPoint")
        if end is not None:
            raw.append(
                (
                    _elk_to_diagram_x(float(end["x"])),
                    _elk_to_diagram_y(float(end["y"]), y_offset=y_offset),
                )
            )
    raw.append((entry_x or target.cx, target.top))
    return _ensure_orthogonal_connector_path(raw)


def _remove_vertical_backtracks(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Drop interior vertices that reverse vertical direction on one column."""
    from visualizer.render import PARALLEL_CONNECTOR_COORD_EPS, _ensure_orthogonal_connector_path

    cleaned = list(points)
    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        for index in range(1, len(cleaned) - 1):
            x0, y0 = cleaned[index - 1]
            x1, y1 = cleaned[index]
            x2, y2 = cleaned[index + 1]
            if abs(x0 - x1) > PARALLEL_CONNECTOR_COORD_EPS or abs(x1 - x2) > PARALLEL_CONNECTOR_COORD_EPS:
                continue
            if (y1 - y0) * (y2 - y1) < 0:
                del cleaned[index]
                changed = True
                break
    return _ensure_orthogonal_connector_path(cleaned)


def needs_helper_routing(
    link_key: tuple[int, int],
    *,
    graph: ComputationGraph,
    src: int,
    tgt: int,
    source_bus: dict[int, float],
    target_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    merge_entry_x: dict[tuple[int, int], float],
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
    incoming: dict[int, list[tuple[int, int]]] | None = None,
) -> bool:
    """Return True when the link needs tee/bus geometry from render helpers."""
    del merge_entry_x
    if src in source_bus or tgt in target_bus:
        return True
    if link_key in merge_link_bus:
        return True
    if outgoing is not None and len(outgoing.get(src, ())) > 1:
        return True
    if incoming is not None and len(incoming.get(link_key[1], ())) > 1:
        return True
    src_frame = _innermost_frame_for_node(graph, src)
    tgt_frame = _innermost_frame_for_node(graph, tgt)
    return src_frame != tgt_frame


def prefers_legacy_routing(
    link_key: tuple[int, int],
    *,
    graph: ComputationGraph,
    src: int,
    tgt: int,
    source_bus: dict[int, float],
    target_bus: dict[int, float],
    merge_link_bus: dict[tuple[int, int], float],
    merge_entry_x: dict[tuple[int, int], float],
    outgoing: dict[int, list[tuple[int, int]]] | None = None,
    incoming: dict[int, list[tuple[int, int]]] | None = None,
) -> bool:
    """Backward-compatible alias for ``needs_helper_routing``."""
    return needs_helper_routing(
        link_key,
        graph=graph,
        src=src,
        tgt=tgt,
        source_bus=source_bus,
        target_bus=target_bus,
        merge_link_bus=merge_link_bus,
        merge_entry_x=merge_entry_x,
        outgoing=outgoing,
        incoming=incoming,
    )


def elk_path_is_acceptable(
    points: list[tuple[float, float]],
    *,
    obstacles: list[Any],
    margin: float,
    graph: ComputationGraph | None = None,
    positions: list | None = None,
    src: int | None = None,
    tgt: int | None = None,
) -> bool:
    from visualizer.render import (
        PARALLEL_CONNECTOR_COORD_EPS,
        _connector_path_violates_inline_frame_bounds,
        _path_hits_obstacles,
    )

    if len(points) < 2:
        return False
    if _path_hits_obstacles(points, obstacles, margin=margin):
        return False
    if graph is not None and positions is not None and src is not None and tgt is not None:
        if (
            _connector_path_violates_inline_frame_bounds(
                points,
                graph,
                positions,
                src=src,
                tgt=tgt,
            )
            is not None
        ):
            return False
    for index in range(len(points) - 2):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        x3, y3 = points[index + 2]
        if abs(y1 - y2) > PARALLEL_CONNECTOR_COORD_EPS or abs(y2 - y3) > PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if abs(x2 - x1) <= PARALLEL_CONNECTOR_COORD_EPS or abs(x3 - x2) <= PARALLEL_CONNECTOR_COORD_EPS:
            continue
        if (x2 - x1) * (x3 - x2) < 0:
            return False
    return True


def _finalize_elk_path(
    points: list[tuple[float, float]],
    *,
    link_key: tuple[int, int],
    source,
    target,
    graph: ComputationGraph,
    merge_entry_x: dict[tuple[int, int], float],
    merge_link_bus: dict[tuple[int, int], float],
    target_bus: dict[int, float],
    source_bus: dict[int, float] | None = None,
) -> list[tuple[float, float]]:
    from visualizer.render import (
        _connector_target_top_entry_y,
        _enforce_merge_link_bus_floor,
        _ensure_orthogonal_connector_path,
        _shared_merge_target_uses_center_entry,
        _snap_connector_path_endpoints,
        _snap_spread_top_entry_path,
    )

    points = _snap_connector_path_endpoints(
        points,
        source=source,
        target=target,
        link_key=link_key,
        graph=graph,
        merge_entry_x=merge_entry_x,
        target_bus=target_bus,
        merge_link_bus=merge_link_bus,
        source_bus=source_bus,
    )
    tgt = link_key[1]
    leg_bus = target_bus.get(tgt) if tgt in target_bus else merge_link_bus.get(link_key)
    if leg_bus is not None:
        points = _ensure_orthogonal_connector_path(_enforce_merge_link_bus_floor(points, leg_bus))
    if (
        link_key in merge_entry_x
        and not _shared_merge_target_uses_center_entry(
            link_key,
            source=source,
            target=target,
            target_bus=target_bus,
            source_bus=source_bus,
        )
    ):
        spread = _snap_spread_top_entry_path(
            points,
            entry_x=merge_entry_x[link_key],
            entry_y=_connector_target_top_entry_y(target),
            min_bus_y=leg_bus,
        )
        if spread:
            points = spread
    return _remove_vertical_backtracks(points)


def _route_helper_link(
    *,
    graph: ComputationGraph,
    link_key: tuple[int, int],
    src: int,
    tgt: int,
    positions: list,
    anchors: dict[int, Any],
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    label_obstacles: list[Any],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_entry_x: dict[tuple[int, int], float],
    merge_link_bus: dict[tuple[int, int], float],
    input_index: int | None,
    inline_bypass_bus_x: dict[tuple[int, int], float] | None,
) -> list[tuple[float, float]] | None:
    """Route fan-out / merge-bus links via render connector helpers."""
    from visualizer.connector_routing import connector_points_for_link
    from visualizer.render import (
        _connector_block_obstacles,
        _link_routing_bus_y,
        _reroute_connector_path_clearing_blocks,
    )

    source = anchors.get(src)
    target = anchors.get(tgt)
    if source is None or target is None:
        return None
    points = connector_points_for_link(
        graph=graph,
        positions=positions,
        anchors=anchors,
        src=src,
        tgt=tgt,
        link_key=link_key,
        incoming=incoming,
        outgoing=outgoing,
        label_obstacles=label_obstacles,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_entry_x=merge_entry_x,
        merge_link_bus=merge_link_bus,
        input_index=input_index,
        inline_bypass_bus_x=inline_bypass_bus_x,
    )
    if points is None or len(points) < 2:
        return None
    obstacles = _connector_block_obstacles(
        anchors,
        src=src,
        tgt=tgt,
        label_obstacles=label_obstacles,
        graph=graph,
        positions=positions,
        link_key=link_key,
    )
    bus_y = _link_routing_bus_y(
        link_key,
        target_bus=target_bus,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
    )
    routed = _reroute_connector_path_clearing_blocks(
        points,
        source=source,
        target=target,
        obstacles=obstacles,
        bus_y=bus_y,
        graph=graph,
        positions=positions,
        link_key=link_key,
        source_bus=source_bus,
        merge_link_bus=merge_link_bus,
        merge_entry_x=merge_entry_x,
    )
    return _finalize_elk_path(
        routed,
        link_key=link_key,
        source=source,
        target=target,
        graph=graph,
        merge_entry_x=merge_entry_x,
        merge_link_bus=merge_link_bus,
        target_bus=target_bus,
        source_bus=source_bus,
    )


def route_elk_links(
    *,
    graph: ComputationGraph,
    links: list[tuple[int, int]],
    positions: list,
    anchors: dict[int, Any],
    merge_entry_x: dict[tuple[int, int], float],
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Return raw ELK polylines for the given links."""
    if not links:
        return {}
    elk_graph = _build_elk_graph(
        graph=graph,
        positions=positions,
        anchors=anchors,
        links=links,
        merge_entry_x=merge_entry_x,
    )
    y_offset = float(elk_graph.pop("_meta")["y_offset"])
    laid_out = _run_elk_layout(elk_graph)
    edge_by_id = {edge["id"]: edge for edge in laid_out.get("edges", ())}
    paths: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for edge_index, link_key in enumerate(links):
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue
        edge = edge_by_id.get(f"e{edge_index}_{src}_{tgt}")
        sections = edge.get("sections") if edge is not None else None
        points = _sections_to_diagram_path(
            sections,
            source=source,
            target=target,
            entry_x=merge_entry_x.get(link_key),
            y_offset=y_offset,
        )
        if len(points) >= 2:
            paths[link_key] = points
    return paths


def route_detail_link_paths(
    *,
    graph: ComputationGraph,
    links: list[tuple[int, int]],
    positions: list,
    anchors: dict[int, Any],
    incoming: dict[int, list[tuple[int, int]]],
    outgoing: dict[int, list[tuple[int, int]]],
    label_obstacles: list[Any],
    target_bus: dict[int, float],
    source_bus: dict[int, float],
    merge_entry_x: dict[tuple[int, int], float],
    merge_link_bus: dict[tuple[int, int], float],
    input_index: int | None,
    inline_bypass_bus_x: dict[tuple[int, int], float] | None,
    audit: RoutingAudit | None = None,
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Route every link: ELK for simple links, render helpers for buses/fan-out."""
    from visualizer.render import (
        CONNECTOR_OBSTACLE_MARGIN,
        _connector_block_obstacles,
        _link_routing_bus_y,
        _reroute_connector_path_clearing_blocks,
    )

    audit = audit if audit is not None else RoutingAudit()
    paths: dict[tuple[int, int], list[tuple[float, float]]] = {}

    elk_candidates = [
        (src, tgt)
        for src, tgt in links
        if not needs_helper_routing(
            (src, tgt),
            graph=graph,
            src=src,
            tgt=tgt,
            source_bus=source_bus,
            target_bus=target_bus,
            merge_link_bus=merge_link_bus,
            merge_entry_x=merge_entry_x,
            outgoing=outgoing,
            incoming=incoming,
        )
    ]
    elk_paths: dict[tuple[int, int], list[tuple[float, float]]] = {}
    if elk_candidates:
        try:
            elk_paths = route_elk_links(
                graph=graph,
                links=elk_candidates,
                positions=positions,
                anchors=anchors,
                merge_entry_x=merge_entry_x,
            )
        except RuntimeError:
            elk_paths = {}

    for link_key in links:
        src, tgt = link_key
        source = anchors.get(src)
        target = anchors.get(tgt)
        if source is None or target is None:
            continue

        if needs_helper_routing(
            link_key,
            graph=graph,
            src=src,
            tgt=tgt,
            source_bus=source_bus,
            target_bus=target_bus,
            merge_link_bus=merge_link_bus,
            merge_entry_x=merge_entry_x,
            outgoing=outgoing,
            incoming=incoming,
        ):
            helper = _route_helper_link(
                graph=graph,
                link_key=link_key,
                src=src,
                tgt=tgt,
                positions=positions,
                anchors=anchors,
                incoming=incoming,
                outgoing=outgoing,
                label_obstacles=label_obstacles,
                target_bus=target_bus,
                source_bus=source_bus,
                merge_entry_x=merge_entry_x,
                merge_link_bus=merge_link_bus,
                input_index=input_index,
                inline_bypass_bus_x=inline_bypass_bus_x,
            )
            if helper is not None and len(helper) >= 2:
                paths[link_key] = helper
                audit.helper.append(link_key)
            continue

        obstacles = _connector_block_obstacles(
            anchors,
            src=src,
            tgt=tgt,
            label_obstacles=label_obstacles,
            graph=graph,
            positions=positions,
            link_key=link_key,
        )
        points = elk_paths.get(link_key)
        if points is not None and len(points) >= 2:
            bus_y = _link_routing_bus_y(
                link_key,
                target_bus=target_bus,
                source_bus=source_bus,
                merge_link_bus=merge_link_bus,
            )
            routed = _reroute_connector_path_clearing_blocks(
                points,
                source=source,
                target=target,
                obstacles=obstacles,
                bus_y=bus_y,
                graph=graph,
                positions=positions,
                link_key=link_key,
                source_bus=source_bus,
                merge_link_bus=merge_link_bus,
                merge_entry_x=merge_entry_x,
            )
            finalized = _finalize_elk_path(
                routed,
                link_key=link_key,
                source=source,
                target=target,
                graph=graph,
                merge_entry_x=merge_entry_x,
                merge_link_bus=merge_link_bus,
                target_bus=target_bus,
                source_bus=source_bus,
            )
            if elk_path_is_acceptable(
                finalized,
                obstacles=obstacles,
                margin=CONNECTOR_OBSTACLE_MARGIN,
                graph=graph,
                positions=positions,
                src=src,
                tgt=tgt,
            ):
                paths[link_key] = finalized
                audit.elk.append(link_key)
                continue

        helper = _route_helper_link(
            graph=graph,
            link_key=link_key,
            src=src,
            tgt=tgt,
            positions=positions,
            anchors=anchors,
            incoming=incoming,
            outgoing=outgoing,
            label_obstacles=label_obstacles,
            target_bus=target_bus,
            source_bus=source_bus,
            merge_entry_x=merge_entry_x,
            merge_link_bus=merge_link_bus,
            input_index=input_index,
            inline_bypass_bus_x=inline_bypass_bus_x,
        )
        if helper is not None and len(helper) >= 2:
            paths[link_key] = helper
            audit.helper.append(link_key)

    return paths
