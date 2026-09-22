###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the graph operation type-check pass (no model load)."""

from __future__ import annotations

import json

from TraceLens.Visualizer.model_explorer_export.type_check import (
    integrity_check_graph_nodes,
    type_check_graph_nodes,
)


def _op_node(node_id, op_type, input_types, input_shapes, raw_op=None):
    # ``raw_op`` is the underlying callable/method name the extractor stamps at
    # emit time (``unsqueeze``, ``cat``, ``transpose``); the type-check resolves
    # each op's operand ceiling dynamically from that name's real parameters
    # (``inspect`` signature / aten schema), never from a static op-name list, so
    # a node the extractor tagged carries it here to exercise that resolution.
    attrs = [
        {"key": "op_type", "value": op_type},
        {"key": "input_types", "value": json.dumps(input_types)},
        {"key": "input_shapes", "value": json.dumps(input_shapes)},
    ]
    if raw_op is not None:
        attrs.append({"key": "raw_op", "value": raw_op})
    return {"id": node_id, "label": op_type, "attrs": attrs}


def test_good_unsqueeze_one_tensor_plus_scalar_is_clean():
    node = _op_node(
        "n:unsqueeze",
        "Unsqueeze",
        ["int64", "Scalar"],
        [["Pv", "2"], []],
        raw_op="unsqueeze",
    )
    assert type_check_graph_nodes([node]) == []


def test_unsqueeze_with_two_tensor_operands_warns():
    # The classic mis-wiring: a second tensor operand where the scalar dim belongs.
    # ``unsqueeze``'s aten schema takes one Tensor + an int, so two tensor operands
    # exceed the dynamically-resolved ceiling of 1.
    node = _op_node(
        "n:bad_unsqueeze",
        "Unsqueeze",
        ["int64", "float16"],
        [["Pv", "2"], ["Pv", "1176"]],
        raw_op="unsqueeze",
    )
    warnings = type_check_graph_nodes([node])
    assert len(warnings) == 1
    assert "n:bad_unsqueeze" in warnings[0]
    assert "tensor operand" in warnings[0]


def test_transpose_with_two_tensor_operands_warns():
    # ``transpose(Tensor, int, int)`` bounds to one tensor operand; two mutually
    # exclusive branches both feeding it (the original linear-attention defect)
    # exceed that ceiling. ``transpose`` was absent from the retired static list --
    # the dynamic aten-schema resolution now covers it with no per-op enumeration.
    node = _op_node(
        "n:bad_transpose",
        "Transpose",
        ["bfloat16", "bfloat16"],
        [["B", "S", "H"], ["B", "S", "H"]],
        raw_op="transpose",
    )
    warnings = type_check_graph_nodes([node])
    assert len(warnings) == 1
    assert "n:bad_transpose" in warnings[0]
    assert "tensor operand" in warnings[0]


def test_unsqueeze_with_zero_tensor_operands_now_warns():
    # Constants/buffers are now first-class ``"Constant"`` operands, so a real
    # activation edge is always expected: an axis op left with 0 activation tensor
    # operands means its sole activation operand went missing -- a wiring bug.
    node = _op_node(
        "n:empty_unsqueeze", "Unsqueeze", ["Scalar"], [[]], raw_op="unsqueeze"
    )
    warnings = type_check_graph_nodes([node])
    assert len(warnings) == 1
    assert "n:empty_unsqueeze" in warnings[0]
    assert "tensor operand" in warnings[0]


def test_unsqueeze_with_one_tensor_plus_constant_warns():
    # A hidden ``Constant`` operand with rank >= 1 (a buffer/param read, or the
    # mis-materialized ``[self.qkv_dim] * 3`` split-size that motivated this check)
    # DOES count as a tensor operand: an axis op bounded to one tensor plus such a
    # constant is over-wired and must warn. (Flips the earlier "constant excluded"
    # behavior per the owner's "second input is a constant tensor" case.)
    node = _op_node(
        "n:const_bad_unsqueeze",
        "Unsqueeze",
        ["int64", "Constant"],
        [["Pv", "2"], ["16"]],
        raw_op="unsqueeze",
    )
    warnings = type_check_graph_nodes([node])
    assert len(warnings) == 1
    assert "n:const_bad_unsqueeze" in warnings[0]
    assert "tensor operand" in warnings[0]


def test_unsqueeze_with_one_tensor_plus_rank0_constant_is_clean():
    # A rank-0 ``Constant`` (a scalar setting captured as a constant) is not an
    # activation tensor, so one activation + one rank-0 constant stays clean.
    node = _op_node(
        "n:const_ok_unsqueeze",
        "Unsqueeze",
        ["int64", "Constant"],
        [["Pv", "2"], []],
        raw_op="unsqueeze",
    )
    assert type_check_graph_nodes([node]) == []


def test_cat_with_many_tensor_operands_does_not_count_warn():
    # ``cat(List[Tensor], dim)`` is variadic: any number of tensor operands is
    # legal, so the operand-count ceiling never applies (only the rank check does).
    node = _op_node(
        "n:cat_ok",
        "Concat",
        ["float16", "float16", "float16", "float16"],
        [["B", "S", "8"]] * 4,
        raw_op="cat",
    )
    assert type_check_graph_nodes([node]) == []


def test_concat_rank_disagreement_warns():
    node = _op_node(
        "n:concat",
        "Concat",
        ["float16", "float16", "float16"],
        [["B", "S", "128"], ["B", "S", "128"], ["B", "S", "4096", "1"]],
        raw_op="cat",
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
        raw_op="cat",
    )
    assert type_check_graph_nodes([node]) == []


def test_unknown_op_is_skipped():
    # An op whose raw name resolves to no signature / aten schema has no known
    # operand contract, so it is skipped -- no false positives on custom kernels.
    node = _op_node(
        "n:mystery",
        "SomeCustomKernel",
        ["float16", "float16"],
        [[], []],
        raw_op="some_custom_kernel",
    )
    assert type_check_graph_nodes([node]) == []


def test_op_without_raw_op_is_skipped():
    # Without a stamped ``raw_op`` there is no name to resolve a contract from, so
    # the count check cannot false-positive on ops the extractor did not tag.
    node = _op_node(
        "n:untagged", "Unsqueeze", ["int64", "float16"], [["Pv", "2"], ["Pv", "3"]]
    )
    assert type_check_graph_nodes([node]) == []


def test_node_without_signature_attrs_is_skipped():
    node = {"id": "n:plain", "label": "Unsqueeze", "attrs": []}
    assert type_check_graph_nodes([node]) == []


def _slice_node(node_id, details, input_shape, output_shape, input_type="float32"):
    # A materialized narrowing ``Slice`` op carrying a structured shape-change
    # detail plus profiler-style input/output shapes, exercising the "declared a
    # shape change but the output shape did not change" check.
    return {
        "id": node_id,
        "label": "Slice",
        "attrs": [
            {"key": "op_type", "value": "Slice"},
            {"key": "details", "value": details},
            {"key": "input_types", "value": json.dumps([input_type])},
            {"key": "input_shapes", "value": json.dumps([input_shape])},
            {"key": "output_shape", "value": output_shape},
        ],
    }


def test_shape_slice_that_narrows_the_axis_is_clean():
    # ``rotate_half``'s ``x[..., : x.shape[-1] // 2]`` halves the last axis: the
    # output shape differs from the input, so no warning.
    node = _slice_node(
        "n:rotate_half_slice",
        "shape_slice: -1=|shape[-1] // 2",
        ["3", "Pv", "16", "64"],
        "[3, Pv, 16, 32] float32",
    )
    assert type_check_graph_nodes([node]) == []


def test_shape_slice_with_unchanged_output_shape_warns():
    # The regression this check guards: the declared narrowing was lost, so the
    # slice's output shape still equals its input shape.
    node = _slice_node(
        "n:noop_slice",
        "shape_slice: -1=|shape[-1] // 2",
        ["3", "Pv", "16", "64"],
        "[3, Pv, 16, 64] float32",
    )
    warnings = type_check_graph_nodes([node])
    assert len(warnings) == 1
    assert "n:noop_slice" in warnings[0]
    assert "narrowing was lost" in warnings[0]


def test_select_dim_with_unchanged_output_shape_warns():
    # ``select_dim`` drops an axis; an output of equal rank/shape means the axis
    # drop was lost.
    node = _slice_node(
        "n:noop_select",
        "select_dim: 1",
        ["Pv", "16"],
        "[Pv, 16] float32",
    )
    warnings = type_check_graph_nodes([node])
    assert len(warnings) == 1
    assert "n:noop_select" in warnings[0]


def test_descriptive_slice_detail_never_warns_on_unchanged_shape():
    # A symbolic, non-foldable range slice (``mixed_qkv[:, :, -seq_len:]``) carries
    # only a descriptive ``slice:`` detail; shape inference legitimately passes the
    # shape through, so an unchanged output must NOT warn.
    node = _slice_node(
        "n:symbolic_slice",
        "slice: (:, :, -seq_len:)",
        ["B", "192", "S"],
        "[B, 192, S] float32",
    )
    assert type_check_graph_nodes([node]) == []


# --------------------------------------------------------------------------- #
# I2 no-source: a namespaced module ``@input`` boundary must be fed by its real
# producer; only a *top-level* model-input boundary is a legitimate sourceless
# graph entry. (Exercises the ``_is_top_level_model_input`` exemption directly.)
# --------------------------------------------------------------------------- #


def _input_boundary(node_id, namespace, label):
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": "@input"}],
    }


def test_i2_flags_floating_namespaced_module_input():
    # A kwargs-forwarded decoder invariant surfaced as a namespaced ``@input:<param>``
    # deep in the body: nothing sources it, so it must be flagged like any orphan.
    node = _input_boundary(
        "decoder/self_attn/@input:position_embeddings",
        "43x_DeepseekV4DecoderLayer/DeepseekV4Attention",
        "position_embeddings",
    )
    warnings = [w for w in integrity_check_graph_nodes([node]) if "I2" in w]
    assert len(warnings) == 1
    assert "decoder/self_attn/@input:position_embeddings" in warnings[0]


def test_i2_exempts_top_level_model_inputs():
    # Root-scope model inputs are legitimate sourceless entry points: the primary
    # tokenized-text ``@input`` and a dedicated ``@input:<param>`` parameter
    # boundary (empty namespace, no ``/`` in the id) must NOT be flagged.
    primary = {
        "id": "@input",
        "label": "Tokenized text",
        "namespace": "",
        "attrs": [{"key": "synthetic", "value": "@input"}],
    }
    param = _input_boundary("@input:attention_mask", "", "attention_mask")
    warnings = [w for w in integrity_check_graph_nodes([primary, param]) if "I2" in w]
    assert warnings == []


def test_i2_sourced_namespaced_input_is_clean():
    # The same namespaced boundary, once wired to its real producer, is clean.
    source = _input_boundary("@input:attention_mask", "", "attention_mask")
    node = _input_boundary(
        "decoder/self_attn/@input:attention_mask",
        "43x_DeepseekV4DecoderLayer/DeepseekV4Attention",
        "attention_mask",
    )
    node["incomingEdges"] = [
        {"sourceNodeId": "@input:attention_mask", "sourceNodeOutputId": "0", "targetNodeInputId": "0"}
    ]
    warnings = [w for w in integrity_check_graph_nodes([source, node]) if "I2" in w]
    assert warnings == []
