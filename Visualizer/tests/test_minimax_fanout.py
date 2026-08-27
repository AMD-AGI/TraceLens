"""Regression coverage for MiniMax same-side stacked fan-out routing."""

from visualizer.computation_graph import (
    ComputationGraph,
    GraphNodeSpec,
    LayoutPosition,
)
from visualizer.render import (
    _RenderAnchor,
    _apply_stacked_same_side_fanout_routes,
)


def _anchor(position: LayoutPosition) -> _RenderAnchor:
    return _RenderAnchor(
        cx=position.cx,
        top=position.top_y,
        bottom=position.bottom,
        left=position.cx - position.width / 2,
        right=position.cx + position.width / 2,
    )


def test_stacked_multiply_children_share_stem_until_top_child():
    """Same-side children fork at the top child instead of at an exterior gutter."""
    nodes = [
        GraphNodeSpec(key="linear", label="Linear"),
        GraphNodeSpec(key="top", label="Multiply"),
        GraphNodeSpec(key="lower", label="Multiply"),
        GraphNodeSpec(key="straight", label="Add"),
    ]
    graph = ComputationGraph(
        nodes=nodes,
        links=[(0, 1), (0, 2), (0, 3)],
    )
    positions = [
        LayoutPosition(spec=nodes[0], cx=2.0, top_y=10.0, width=0.8, height=0.4),
        LayoutPosition(spec=nodes[1], cx=1.0, top_y=9.0, width=0.8, height=0.4),
        LayoutPosition(spec=nodes[2], cx=1.0, top_y=8.0, width=0.8, height=0.4),
        LayoutPosition(spec=nodes[3], cx=2.0, top_y=8.5, width=0.8, height=0.4),
    ]
    anchors = {index: _anchor(position) for index, position in enumerate(positions)}
    top_link = (0, 1)
    lower_link = (0, 2)
    paths = {
        top_link: [(2.0, 9.6), (2.0, 9.45), (1.0, 9.45), (1.0, 9.0)],
        lower_link: [
            (2.0, 9.6),
            (2.0, 9.5),
            (0.4, 9.5),
            (0.4, 8.1),
            (1.0, 8.1),
            (1.0, 8.0),
        ],
        (0, 3): [(2.0, 9.6), (2.0, 8.5)],
    }

    _apply_stacked_same_side_fanout_routes(
        paths,
        graph=graph,
        positions=positions,
        anchors=anchors,
        label_obstacles=[],
        outgoing={0: [top_link, lower_link, (0, 3)]},
        merge_entry_x={lower_link: 1.08},
    )

    fork = paths[top_link][-2]
    fork_index = paths[lower_link].index(fork)
    assert paths[lower_link][: fork_index + 1] == paths[top_link][:-1]
    assert paths[lower_link][fork_index + 1][0] != fork[0]
