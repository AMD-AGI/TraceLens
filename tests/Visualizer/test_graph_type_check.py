###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the graph operation type-check pass (no model load)."""

from __future__ import annotations

import json

from TraceLens.Visualizer.model_explorer_export.type_check import type_check_graph_nodes


def _op_node(node_id, op_type, input_types, input_shapes):
    return {
        "id": node_id,
        "label": op_type,
        "attrs": [
            {"key": "op_type", "value": op_type},
            {"key": "input_types", "value": json.dumps(input_types)},
            {"key": "input_shapes", "value": json.dumps(input_shapes)},
        ],
    }


def test_good_unsqueeze_one_tensor_plus_scalar_is_clean():
    node = _op_node("n:unsqueeze", "Unsqueeze", ["int64", "Scalar"], [["Pv", "2"], []])
    assert type_check_graph_nodes([node]) == []


def test_unsqueeze_with_two_tensor_operands_warns():
    # The classic mis-wiring: a second tensor operand where the scalar dim belongs.
    node = _op_node(
        "n:bad_unsqueeze",
        "Unsqueeze",
        ["int64", "float16"],
        [["Pv", "2"], ["Pv", "1176"]],
    )
    warnings = type_check_graph_nodes([node])
    assert len(warnings) == 1
    assert "n:bad_unsqueeze" in warnings[0]
    assert "tensor operand" in warnings[0]


def test_unsqueeze_with_zero_tensor_operands_is_tolerated():
    # A sole hidden constant/buffer operand (pruned per "never show constants")
    # leaves 0 tensor edges -- ambiguous, so no warning.
    node = _op_node("n:const_unsqueeze", "Unsqueeze", ["Scalar"], [[]])
    assert type_check_graph_nodes([node]) == []


def test_concat_rank_disagreement_warns():
    node = _op_node(
        "n:concat",
        "Concat",
        ["float16", "float16", "float16"],
        [["B", "S", "128"], ["B", "S", "128"], ["B", "S", "4096", "1"]],
    )
    warnings = type_check_graph_nodes([node])
    assert len(warnings) == 1
    assert "rank" in warnings[0]


def test_concat_same_rank_is_clean():
    node = _op_node(
        "n:concat_ok",
        "Concat",
        ["float16", "float16"],
        [["B", "S", "128"], ["B", "S", "128"]],
    )
    assert type_check_graph_nodes([node]) == []


def test_unknown_op_is_skipped():
    # An op with no known operand contract must not produce false positives.
    node = _op_node("n:mystery", "SomeCustomKernel", ["float16", "float16"], [[], []])
    assert type_check_graph_nodes([node]) == []


def test_node_without_signature_attrs_is_skipped():
    node = {"id": "n:plain", "label": "Unsqueeze", "attrs": []}
    assert type_check_graph_nodes([node]) == []
