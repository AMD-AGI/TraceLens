"""Tests for post-layout shrinkwrap passes."""

from visualizer.render import PARALLEL_CONNECTOR_COORD_EPS
from visualizer.shrinkwrap import (
    _apply_bus_y_shift,
    _max_upward_bus_shift,
    _path_uses_bus_y,
    _validate_shrunk_paths,
)


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
    assert new_y <= 7.88 + 1e-6


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
        graph=type("G", (), {"inline_binary_operand_links": set(), "links": [], "inline_frames": []})(),
        anchors=anchors,
        label_obstacles=[],
        positions=[],
    )
    assert result is None
