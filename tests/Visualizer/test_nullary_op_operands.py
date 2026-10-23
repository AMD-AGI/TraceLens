###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An op that takes no tensor operand reads what SIZES it.

``torch.arange`` builds a tensor out of sizes; its schema has no tensor argument
at all. GLM's ``get_vision_position_ids`` is driven entirely by host data read
off ``grid_thw.tolist()``, so the frame's first op is an ``arange`` with nothing
to read -- and an op with no producer docks on whatever chain runs past it.

That handed the arange the vision activation, and the boundary tiles built to
carry it then asserted, all the way up, that the frame reads ``hidden_states``
at ``[Pv, 1176]``. The call is ``get_vision_position_ids(grid_thw,
self.spatial_merge_size)``: it is never passed that tensor. The analyser knew --
``forward_step_predecessor_args = {'grid_thw': '@method_input:grid_thw'}`` -- and
only the drawing disagreed.

Four earlier approaches failed here, each for a measured reason, because they
treated the LABEL as the defect. The label was honest: a boundary is named for
its producer. The edge was the defect.

Leaving it rootless is not the answer either: a range sized by the model's own
data depends on that data, which is why it is docked at all. The op already
records WHICH parameter it reads from the boundary, so it is docked there -- the
grid, not the activation.

The op's own schema decides whether this applies, so no operation is named here.
Only an edge from a synthetic boundary moves: an edge from a real producer is a
dependency something built on purpose.
"""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.merge import (
    _dock_nullary_ops_on_what_sizes_them,
)


def _op(
    node_id: str, raw_op: str, sources: list[str], sized_by: str | None = "grid_thw"
) -> dict:
    attrs = [{"key": "raw_op", "value": raw_op}]
    if sized_by:
        attrs.append({"key": "boundary_input", "value": sized_by})
    return {
        "id": node_id,
        "label": raw_op.title(),
        "namespace": node_id.rsplit("/", 1)[0],
        "attrs": attrs,
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": "0"} for s in sources
        ],
    }


def _boundary(node_id: str, label: str) -> dict:
    return {
        "id": node_id,
        "label": label,
        "namespace": node_id.rsplit("/", 1)[0],
        "attrs": [{"key": "synthetic", "value": "@input"}],
        "incomingEdges": [],
    }


def _sources(nodes: list[dict], node_id: str) -> list[str]:
    node = next(n for n in nodes if n["id"] == node_id)
    return [str(e["sourceNodeId"]) for e in node.get("incomingEdges") or []]


class TestANullaryOpReadsWhatSizesIt:
    def test_arange_leaves_the_chain_for_the_grid(self) -> None:
        nodes = [
            _boundary("f/@input:hidden_states", "hidden_states"),
            _boundary("f/@input:grid_thw", "grid_thw"),
            _op("f/@op_arange", "arange", ["f/@input:hidden_states"]),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources(nodes, "f/@op_arange") == ["f/@input:grid_thw"]

    def test_it_is_not_left_rootless(self) -> None:
        """A range sized by the model's data DEPENDS on that data."""
        nodes = [
            _boundary("f/@input:hidden_states", "hidden_states"),
            _boundary("f/@input:grid_thw", "grid_thw"),
            _op("f/@op_arange", "arange", ["f/@input:hidden_states"]),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources(nodes, "f/@op_arange")

    def test_a_carrier_one_level_up_is_reached(self) -> None:
        nodes = [
            _boundary("f/@input:grid_thw", "grid_thw"),
            _boundary("f/inner/@input:hidden_states", "hidden_states"),
            _op("f/inner/@op_arange", "arange", ["f/inner/@input:hidden_states"]),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources(nodes, "f/inner/@op_arange") == ["f/@input:grid_thw"]

    def test_an_op_naming_nothing_is_left_alone(self) -> None:
        """Without a recorded parameter there is nothing to move it to."""
        nodes = [
            _boundary("f/@input:x", "x"),
            _op("f/@op_arange", "arange", ["f/@input:x"], sized_by=None),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources(nodes, "f/@op_arange") == ["f/@input:x"]


class TestWhatIsLeftAlone:
    def test_an_op_that_does_take_a_tensor_is_untouched(self) -> None:
        nodes = [
            _boundary("f/@input:x", "x"),
            _boundary("f/@input:grid_thw", "grid_thw"),
            _op("f/@op_add", "add", ["f/@input:x"]),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources(nodes, "f/@op_add") == ["f/@input:x"]

    def test_an_edge_from_a_real_producer_survives(self) -> None:
        """Somebody built that dependency; removing it would hide work."""
        nodes = [
            _boundary("f/@input:grid_thw", "grid_thw"),
            _op("f/@op_mul", "mul", []),
            _op("f/@op_arange", "arange", ["f/@op_mul"]),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources(nodes, "f/@op_arange") == ["f/@op_mul"]

    def test_an_op_with_no_known_schema_is_untouched(self) -> None:
        nodes = [
            _boundary("f/@input:x", "x"),
            _boundary("f/@input:grid_thw", "grid_thw"),
            _op("f/@op_custom", "some_custom_kernel", ["f/@input:x"]),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources(nodes, "f/@op_custom") == ["f/@input:x"]

    def test_a_variadic_op_is_untouched(self) -> None:
        """``cat`` takes a tensor LIST; its ceiling is not a real zero."""
        nodes = [
            _boundary("f/@input:x", "x"),
            _boundary("f/@input:grid_thw", "grid_thw"),
            _op("f/@op_cat", "cat", ["f/@input:x"]),
        ]
        _dock_nullary_ops_on_what_sizes_them(nodes)
        assert _sources(nodes, "f/@op_cat") == ["f/@input:x"]
