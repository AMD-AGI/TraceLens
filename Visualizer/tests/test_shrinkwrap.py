"""Tests for post-layout shrinkwrap passes."""

from visualizer.computation_graph import (
    ComputationGraph,
    GraphNodeSpec,
    InlineFrameSpec,
    LayoutPosition,
    stack_fanout_branch_columns,
)
from visualizer.render import PARALLEL_CONNECTOR_COORD_EPS
from visualizer.shrinkwrap import (
    _apply_bus_y_shift,
    _is_source_fanout_bus_y,
    _max_upward_bus_shift,
    _path_uses_bus_y,
    _shift_feeder_layout_for_bus_y,
    _shrinkwrap_vertical_layout,
    _validate_shrunk_paths,
)
from visualizer.sizing import min_vertical_block_gap


class _Anchor:
    def __init__(self, *, cx: float, top: float, bottom: float, left: float, right: float):
        self.cx = cx
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right


def test_max_upward_bus_shift_finds_vertical_slack():
    bus_y = 6.0
    link_paths = {
        (0, 1): [(5.0, 8.0), (5.0, bus_y), (1.0, bus_y), (1.0, 5.8)],
    }
    anchors = {
        0: _Anchor(cx=5.0, top=8.1, bottom=8.0, left=4.7, right=5.3),
        1: _Anchor(cx=1.0, top=5.8, bottom=5.4, left=0.7, right=1.3),
    }
    new_y = _max_upward_bus_shift(
        bus_y,
        link_paths,
        anchors=anchors,
        targets={1},
        min_gap=0.02,
    )
    assert new_y is not None
    assert new_y > bus_y + PARALLEL_CONNECTOR_COORD_EPS
    assert new_y <= 7.9 + 1e-6


def test_max_upward_bus_shift_stays_below_the_lowest_feeder():
    """A shared merge bus may only rise as far as its deepest feeder allows."""
    bus_y = 4.0
    link_paths = {
        (0, 2): [(1.0, 7.0), (1.0, bus_y), (3.0, bus_y), (3.0, 3.8)],
        (1, 2): [(5.0, 4.6), (5.0, bus_y), (3.0, bus_y), (3.0, 3.8)],
    }
    anchors = {
        0: _Anchor(cx=1.0, top=7.4, bottom=7.0, left=0.7, right=1.3),
        1: _Anchor(cx=5.0, top=5.0, bottom=4.6, left=4.7, right=5.3),
        2: _Anchor(cx=3.0, top=3.8, bottom=3.4, left=2.7, right=3.3),
    }
    new_y = _max_upward_bus_shift(
        bus_y,
        link_paths,
        anchors=anchors,
        targets={2},
        min_gap=0.02,
    )
    assert new_y is not None
    assert new_y < anchors[1].bottom, "the bus rose above a feeder, forcing it to double back"


def test_apply_bus_y_shift_updates_shared_horizontal_legs():
    paths = {(0, 1): [(2.0, 6.0), (2.0, 5.0), (4.0, 5.0), (4.0, 4.0)]}
    _apply_bus_y_shift(paths, 5.0, 5.5)
    assert abs(paths[(0, 1)][1][1] - 5.5) < 1e-6
    assert abs(paths[(0, 1)][2][1] - 5.5) < 1e-6
    assert abs(paths[(0, 1)][0][1] - 6.0) < 1e-6


def test_path_uses_bus_y_detects_horizontal_merge_leg():
    points = [(1.0, 6.5), (1.0, 6.0), (4.0, 6.0), (4.0, 5.5)]
    assert _path_uses_bus_y(points, 6.0)
    assert not _path_uses_bus_y(points, 5.0)


def test_validate_shrunk_paths_rejects_block_crossing():
    points = [(1.0, 8.0), (1.0, 4.0)]
    source = _Anchor(cx=1.0, top=8.1, bottom=8.0, left=0.7, right=1.3)
    target = _Anchor(cx=1.0, top=4.0, bottom=3.6, left=0.7, right=1.3)
    obstacle = _Anchor(cx=1.0, top=6.1, bottom=5.7, left=0.7, right=1.3)
    anchors = {0: source, 1: target, 2: obstacle}
    result = _validate_shrunk_paths(
        {(0, 1): points},
        graph=type("G", (), {"links": [], "inline_frames": []})(),
        anchors=anchors,
        label_obstacles=[],
        positions=[],
    )
    assert result is None


def test_shift_feeder_layout_for_bus_y_moves_frame_column_and_anchors():
    """Raising a merge bus also lifts feeder inline-frame columns in positions."""
    graph = type(
        "G",
        (),
        {
            "links": [(0, 2), (1, 2)],
            "inline_frames": [
                InlineFrameSpec(frame_id="feeder_a", label="feeder_a", node_indices=[0]),
                InlineFrameSpec(frame_id="feeder_b", label="feeder_b", node_indices=[1]),
            ],
            "nodes": [
                GraphNodeSpec(key="a", label="a", synthetic=None),
                GraphNodeSpec(key="b", label="b", synthetic=None),
                GraphNodeSpec(key="merge", label="merge", synthetic=None),
            ],
        },
    )()
    positions = [
        LayoutPosition(spec=graph.nodes[0], cx=2.0, top_y=8.0, width=1.0, height=0.4),
        LayoutPosition(spec=graph.nodes[1], cx=4.0, top_y=8.0, width=1.0, height=0.4),
        LayoutPosition(spec=graph.nodes[2], cx=3.0, top_y=4.0, width=1.0, height=0.4),
    ]
    anchors = {
        0: _Anchor(cx=2.0, top=8.1, bottom=8.0, left=1.7, right=2.3),
        1: _Anchor(cx=4.0, top=8.1, bottom=8.0, left=3.7, right=4.3),
        2: _Anchor(cx=3.0, top=4.0, bottom=3.6, left=2.7, right=3.3),
    }
    incoming = {2: [(0, 2), (1, 2)]}
    link_paths = {
        (0, 2): [(2.0, 8.0), (2.0, 6.0), (3.0, 6.0), (3.0, 4.0)],
        (1, 2): [(4.0, 8.0), (4.0, 6.0), (3.0, 6.0), (3.0, 4.0)],
    }
    source_bus: dict[int, float] = {0: 6.0, 1: 6.0}
    old_y = 6.0
    new_y = 6.4
    _apply_bus_y_shift(link_paths, old_y, new_y)
    _shift_feeder_layout_for_bus_y(
        positions,
        anchors,
        link_paths,
        graph=graph,
        old_y=old_y,
        new_y=new_y,
        targets={2},
        incoming=incoming,
        source_bus=source_bus,
    )
    assert abs(positions[0].top_y - 8.4) < 1e-6
    assert abs(positions[1].top_y - 8.4) < 1e-6
    assert abs(anchors[0].bottom - 8.4) < 1e-6
    assert abs(anchors[1].bottom - 8.4) < 1e-6
    assert abs(link_paths[(0, 2)][0][1] - 8.4) < 1e-6
    assert abs(link_paths[(0, 2)][1][1] - new_y) < 1e-6
    assert abs(source_bus[0] - new_y) < 1e-6


def _pad_tail_branch_layout(*, pad_top_y: float):
    """Fan-out branch column ``Linear -> RMSNorm -> Pad`` with the Pad left behind."""
    graph = ComputationGraph(
        nodes=[
            GraphNodeSpec(key="fan0-1:kv_b_proj:0", label="Linear"),
            GraphNodeSpec(key="fan0-1:k_norm:0", label="RMSNorm"),
            GraphNodeSpec(key="fan0-1:pad:0", label="Pad"),
        ],
        links=[(0, 1), (1, 2)],
    )
    positions = [
        LayoutPosition(spec=graph.nodes[0], cx=5.0, top_y=9.0, width=0.6, height=0.2),
        LayoutPosition(spec=graph.nodes[1], cx=5.2, top_y=8.4, width=0.6, height=0.2),
        LayoutPosition(spec=graph.nodes[2], cx=5.4, top_y=pad_top_y, width=0.6, height=0.2),
    ]
    return graph, positions


def test_stack_fanout_branch_columns_stacks_branch_tails():
    """Tail tiles (e.g. Pad) join the branch column instead of keeping their layer row."""
    graph, positions = _pad_tail_branch_layout(pad_top_y=6.0)
    stack_fanout_branch_columns(positions, graph, min_gap=0.1)
    assert abs(positions[0].bottom - positions[1].top_y - 0.1) < 1e-6
    assert abs(positions[1].bottom - positions[2].top_y - 0.1) < 1e-6
    assert len({round(pos.cx, 6) for pos in positions}) == 1


def test_shrinkwrap_vertical_compacts_fanout_branch_tail_fully():
    """The vertical pass closes branch tail slack down to the column gap."""
    graph, positions = _pad_tail_branch_layout(pad_top_y=6.0)
    _shrinkwrap_vertical_layout(positions, graph, min_gap=0.02)
    slack = positions[1].bottom - positions[2].top_y
    assert abs(slack - min_vertical_block_gap()) < 1e-6


def test_is_source_fanout_bus_y_skips_per_leg_merge_buses():
    """Per-leg fan-out merge buses must not be treated as shared shrinkwrap targets."""
    source_bus = {0: 9.3}
    merge_link_bus = {(0, 1): 8.9, (0, 2): 8.5}
    assert _is_source_fanout_bus_y(8.9, source_bus, merge_link_bus)
    assert _is_source_fanout_bus_y(8.5, source_bus, merge_link_bus)
    assert not _is_source_fanout_bus_y(9.3, source_bus, merge_link_bus)
    assert not _is_source_fanout_bus_y(7.0, source_bus, merge_link_bus)
