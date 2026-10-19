###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An op that takes an INDEX takes an integer one.

``torch`` cannot index with a float tensor at all, so a ``gather``/``scatter``/
``embedding`` whose wired operands are every one floating-point is reporting an
index it could not have run: the dtype was lost somewhere upstream, and with it
the shape rule's only way to tell the index from the table it reads.

The check reads which parameters are indices from the op's own aten signature,
so it follows torch rather than a list of op names kept here.

It is a real find, not a theoretical one: it caught three defects when first
written. Two were fixed at the time (a rooted ``@output`` losing its dtype, and
``torch.where`` reporting its CONDITION's dtype); the third needed GLM's
``pool_indices`` chain to report integer shapes, which is why this landed only
once that chain was right.
"""

from __future__ import annotations

import json

from TraceLens.Visualizer.model_explorer_export.type_check import (
    type_check_graph_nodes,
)


def _gather(input_types: list[str], shapes: list[list[str]]) -> list[dict]:
    """A ``gather`` reading two operands, typed as given."""
    sources = [f"@op_src{index}" for index in range(len(input_types))]
    producers = [
        {
            "id": source,
            "label": "Producer",
            "namespace": "m",
            "attrs": [{"key": "output_shape", "value": "[B, S] " + input_types[index]}],
            "incomingEdges": [],
        }
        for index, source in enumerate(sources)
    ]
    node = {
        "id": "@op_gather",
        "label": "gather",
        "namespace": "m",
        "attrs": [
            {"key": "raw_op", "value": "gather"},
            {"key": "op_type", "value": "gather"},
            {"key": "detail", "value": "dim: -1"},
            {"key": "input_shapes", "value": json.dumps(shapes)},
            {"key": "input_types", "value": json.dumps(input_types)},
            {"key": "output_shape", "value": "[B, S] float32"},
        ],
        "incomingEdges": [
            {"sourceNodeId": source, "sourceNodeOutputId": "0"} for source in sources
        ],
    }
    return [*producers, node]


def _index_warnings(nodes: list[dict]) -> list[str]:
    return [
        warning
        for warning in type_check_graph_nodes(nodes)
        if "is never floating-point" in warning
    ]


class TestAnIndexIsAnInteger:
    def test_all_float_operands_are_reported(self) -> None:
        warnings = _index_warnings(
            _gather(["float32", "float32"], [["B", "S"], ["B", "S"]])
        )
        assert warnings, "a gather with no integer operand should be reported"
        assert "'index'" in warnings[0], warnings[0]

    def test_an_integer_index_is_accepted(self) -> None:
        assert (
            _index_warnings(_gather(["float32", "int64"], [["B", "S"], ["B", "S"]]))
            == []
        )

    def test_the_index_may_come_first(self) -> None:
        assert (
            _index_warnings(_gather(["int64", "float32"], [["B", "S"], ["B", "S"]]))
            == []
        )

    def test_a_single_operand_is_not_judged(self) -> None:
        """One wired operand means the index edge is missing, not mis-typed.

        That is the arity check's business, and reporting it here too would say
        the same thing twice about one node.
        """
        assert _index_warnings(_gather(["float32"], [["B", "S"]])) == []
