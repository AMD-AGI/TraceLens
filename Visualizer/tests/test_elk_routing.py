"""Tests for ELK connector routing integration and migration tracking."""

from __future__ import annotations

import pytest

from visualizer.computation_graph import GraphNodeSpec, InlineFrameSpec, LayoutPosition
from visualizer.elk_routing import (
    MIGRATION_PHASE,
    RoutingAudit,
    prefers_legacy_routing,
    route_detail_link_paths,
    route_elk_links,
)
from visualizer.render import _RenderAnchor


class _Graph:
    def __init__(self, *, inline_frames=(), nodes=None, links=()):
        self.inline_frames = inline_frames
        self.nodes: list[GraphNodeSpec] = list(nodes or [])
        self.links = list(links)


def test_prefers_legacy_for_fanout_and_buses():
    graph = _Graph()
    assert prefers_legacy_routing(
        (0, 1),
        graph=graph,
        src=0,
        tgt=1,
        source_bus={0: 1.0},
        target_bus={},
        merge_link_bus={},
        merge_entry_x={},
    )
    assert prefers_legacy_routing(
        (0, 1),
        graph=graph,
        src=0,
        tgt=1,
        source_bus={},
        target_bus={},
        merge_link_bus={(0, 1): 0.5},
        merge_entry_x={},
    )
    assert prefers_legacy_routing(
        (0, 1),
        graph=graph,
        src=0,
        tgt=1,
        source_bus={},
        target_bus={},
        merge_link_bus={},
        merge_entry_x={},
        outgoing={0: [(0, 1), (0, 2)]},
    )
    assert not prefers_legacy_routing(
        (1, 2),
        graph=graph,
        src=1,
        tgt=2,
        source_bus={},
        target_bus={},
        merge_link_bus={},
        merge_entry_x={(1, 2): 0.3},
        outgoing={1: [(1, 2)]},
        incoming={2: [(1, 2)]},
    )


def test_route_elk_links_simple_stack():
    pytest.importorskip("subprocess")
    positions = [
        LayoutPosition(
            spec=GraphNodeSpec(key="a", label="A"),
            cx=1.0,
            top_y=3.0,
            width=0.8,
            height=0.4,
        ),
        LayoutPosition(
            spec=GraphNodeSpec(key="b", label="B"),
            cx=1.0,
            top_y=2.0,
            width=0.8,
            height=0.4,
        ),
    ]
    graph = _Graph(nodes=[positions[0].spec, positions[1].spec])
    anchors = {
        0: _RenderAnchor(cx=1.0, top=3.0, bottom=2.6, left=0.6, right=1.4),
        1: _RenderAnchor(cx=1.0, top=2.0, bottom=1.6, left=0.6, right=1.4),
    }
    paths = route_elk_links(
        graph=graph,
        links=[(0, 1)],
        positions=positions,
        anchors=anchors,
        merge_entry_x={},
    )
    assert (0, 1) in paths
    assert len(paths[(0, 1)]) >= 2
    assert paths[(0, 1)][0][0] == pytest.approx(1.0, abs=0.05)


@pytest.mark.skipif(
    MIGRATION_PHASE < 2,
    reason="Compound inline frames require phase 2b",
)
def test_route_elk_links_builds_inline_frame_compound():
    pytest.importorskip("subprocess")
    frame = InlineFrameSpec(frame_id="act", label="Act", node_indices=[1, 2])
    positions = [
        LayoutPosition(spec=GraphNodeSpec(key="in", label="x"), cx=1.0, top_y=4.0, width=0.8, height=0.4),
        LayoutPosition(spec=GraphNodeSpec(key="a", label="A"), cx=1.0, top_y=3.0, width=0.8, height=0.4),
        LayoutPosition(spec=GraphNodeSpec(key="b", label="B"), cx=1.2, top_y=2.2, width=0.8, height=0.4),
    ]
    graph = _Graph(
        inline_frames=[frame],
        nodes=[p.spec for p in positions],
        links=[(0, 1), (1, 2)],
    )
    anchors = {
        0: _RenderAnchor(cx=1.0, top=4.0, bottom=3.6, left=0.6, right=1.4),
        1: _RenderAnchor(cx=1.0, top=3.0, bottom=2.6, left=0.6, right=1.4),
        2: _RenderAnchor(cx=1.2, top=2.2, bottom=1.8, left=0.8, right=1.6),
    }
    from visualizer.elk_routing import _build_elk_graph

    elk = _build_elk_graph(
        graph=graph,
        positions=positions,
        anchors=anchors,
        links=[(0, 1), (1, 2)],
        merge_entry_x={},
    )
    frame_nodes = [child for child in elk["children"] if child["id"].startswith("frame_")]
    assert len(frame_nodes) == 1
    assert len(frame_nodes[0]["children"]) == 2


def test_route_detail_link_paths_records_audit():
    pytest.importorskip("subprocess")
    positions = [
        LayoutPosition(spec=GraphNodeSpec(key="a", label="A"), cx=1.0, top_y=3.0, width=0.8, height=0.4),
        LayoutPosition(spec=GraphNodeSpec(key="b", label="B"), cx=1.0, top_y=2.0, width=0.8, height=0.4),
    ]
    graph = _Graph(nodes=[positions[0].spec, positions[1].spec])
    anchors = {
        0: _RenderAnchor(cx=1.0, top=3.0, bottom=2.6, left=0.6, right=1.4),
        1: _RenderAnchor(cx=1.0, top=2.0, bottom=1.6, left=0.6, right=1.4),
    }
    audit = RoutingAudit()
    paths = route_detail_link_paths(
        graph=graph,
        links=[(0, 1)],
        positions=positions,
        anchors=anchors,
        incoming={1: [(0, 1)]},
        outgoing={0: [(0, 1)]},
        label_obstacles=[],
        target_bus={},
        source_bus={},
        merge_entry_x={},
        merge_link_bus={},
        input_index=0,
        inline_bypass_bus_x={},
        audit=audit,
    )
    assert (0, 1) in paths
    assert audit.elk_count + audit.legacy_count == 1
