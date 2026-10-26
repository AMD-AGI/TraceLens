###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shape and type checks on an exported graph, from declared axes to host readers."""

from __future__ import annotations

# --- test_graph_type_check.py ---

import json

from TraceLens.Visualizer.model_explorer_export import type_check as _tc
from TraceLens.Visualizer.model_explorer_export.type_check import (
    group_cycle_check_graph_nodes,
    integrity_check_graph_nodes,
    type_check_graph_nodes,
    warn_degenerate_mutually_exclusive_branches,
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


def test_broadcasting_builtin_with_opaque_inspect_signature_is_not_a_tensor_list():
    # torch.eq exposes an opaque varargs signature to ``inspect`` in some torch
    # builds. Its aten schema is still the authoritative binary contract, so
    # broadcasting a scalar/rank-1 value against a matrix is legal.
    node = _op_node(
        "n:eq_broadcast",
        "Equal",
        ["float16", "float16"],
        [["1"], ["B", "S", "H"]],
        raw_op="eq",
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
# Reshape no-op check (Task M): a ``view``/``reshape``/``expand`` declaring a
# FULLY CONCRETE target shape whose resolved output still equals its (concrete)
# input shape dropped the declared reshape -- the reshape analogue of the no-op
# slice. Non-concrete targets (symbolic / ``-1`` / starred, incl. merged ``B*S``)
# and contiguous-forcing reshapes (declared == input) must never fire.
# --------------------------------------------------------------------------- #


def _reshape_node(node_id, shape_detail, input_shape, output_shape, op_type="Reshape"):
    return {
        "id": node_id,
        "label": op_type,
        "attrs": [
            {"key": "op_type", "value": op_type},
            {"key": "details", "value": f"shape: {shape_detail}"},
            {"key": "input_types", "value": json.dumps(["float32"])},
            {"key": "input_shapes", "value": json.dumps([input_shape])},
            {"key": "output_shape", "value": output_shape},
        ],
    }


def test_concrete_reshape_that_did_not_change_shape_warns():
    # The regression this guards: a fully concrete ``reshape(2, 16)`` on a [4, 8]
    # operand must yield [2, 16]; an output still equal to the [4, 8] input means the
    # declared reshape was dropped and the source passed through.
    node = _reshape_node(
        "n:noop_reshape",
        "2, 16",
        ["4", "8"],
        "[4, 8] float32",
    )
    warnings = type_check_graph_nodes([node])
    assert len(warnings) == 1
    assert "n:noop_reshape" in warnings[0]
    assert "reshape was lost" in warnings[0]


def test_concrete_reshape_that_changed_shape_is_clean():
    # A genuine concrete reshape whose output reflects the declared target does not
    # fire (output != input).
    node = _reshape_node(
        "n:real_reshape",
        "2, 16",
        ["4", "8"],
        "[2, 16] float32",
    )
    assert type_check_graph_nodes([node]) == []


def test_contiguous_forcing_reshape_to_same_shape_is_clean():
    # ``x.reshape(x.shape)`` declares the SAME concrete shape as its input; an output
    # equal to the input is correct (declared == input), so it must NOT fire.
    node = _reshape_node(
        "n:contig_reshape",
        "4, 8",
        ["4", "8"],
        "[4, 8] float32",
    )
    assert type_check_graph_nodes([node]) == []


def test_symbolic_reshape_target_never_warns_on_unchanged_shape():
    # A symbolic declared target (``bsz * seq_len, -1`` on a [B*S, 896] operand) is
    # non-concrete; shape inference may legitimately resolve it to the operand's own
    # shape, so an unchanged output must NOT fire.
    node = _reshape_node(
        "n:symbolic_reshape",
        "bsz * seq_len, -1",
        ["B*S", "896"],
        "[B*S, 896] float32",
    )
    assert type_check_graph_nodes([node]) == []


def test_reshape_with_negative_one_placeholder_never_warns():
    # A ``-1`` axis is an unfolded placeholder, not a concrete dim; even with an
    # otherwise-concrete input the target is non-concrete and the element-count
    # comparison must NOT fire. The RANK is still a fact -- ``reshape(-1, 1,
    # 288)`` yields three axes whatever ``-1`` becomes -- so the output carries
    # the three it declared; what is unknown is their sizes, not how many.
    node = _reshape_node(
        "n:neg_one_reshape",
        "-1, 1, 288",
        ["8", "1"],
        "[8, 1, 288] float32",
    )
    assert type_check_graph_nodes([node]) == []


def test_merged_product_reshape_is_not_flagged():
    # The deliberately-excluded merged-product case: a starred restore
    # ``*orig_shape`` (or a symbolic merge) is non-concrete -> conservatively skipped.
    node = _reshape_node(
        "n:merged_reshape",
        "*orig_shape",
        ["B*S", "33792"],
        "[B*S, 33792] float32",
    )
    assert type_check_graph_nodes([node]) == []


# --------------------------------------------------------------------------- #
# Output-arity check (Task K): a kernel node must not advertise more tensor
# output ports than its resolved wrapper returns (``outputs: N`` detail, stamped
# from the wrapper's own ``return`` -- sdpa -> 1, eager -> 2).
# --------------------------------------------------------------------------- #


def _kernel_node(node_id, details, output_ports):
    return {
        "id": node_id,
        "label": "sdpa",
        "attrs": [
            {"key": "op_type", "value": "sdpa"},
            {"key": "details", "value": details},
        ],
        "outputsMetadata": [{"id": str(p)} for p in output_ports],
    }


def test_sdpa_one_output_one_port_is_clean():
    node = _kernel_node(
        "k:sdpa", "kernel: sdpa; inputs: q,kv,attention_mask; outputs: 1", ["0"]
    )
    assert type_check_graph_nodes([node]) == []


def test_sdpa_one_output_but_two_ports_warns():
    # The phantom-slice regression: a one-output kernel advertising two output
    # ports (an unpacked None slot fanned out as a second tensor).
    node = _kernel_node("k:sdpa", "kernel: sdpa; outputs: 1", ["0", "1"])
    warnings = type_check_graph_nodes([node])
    assert len(warnings) == 1
    assert "k:sdpa" in warnings[0]
    assert "phantom output slot" in warnings[0]


def test_eager_two_outputs_two_ports_is_clean():
    # eager attention genuinely returns (attn_output, attn_weights): two ports OK.
    node = _kernel_node("k:eager", "kernel: eager; outputs: 2", ["0", "1"])
    assert type_check_graph_nodes([node]) == []


def test_output_arity_counts_distinct_consumer_ordinals():
    # No outputsMetadata, but two consumers read ordinals 0 and 1 off a kernel that
    # declares a single output -> the distinct-ordinal count exceeds the arity.
    kernel = {
        "id": "k:sdpa",
        "label": "sdpa",
        "attrs": [
            {"key": "op_type", "value": "sdpa"},
            {"key": "details", "value": "kernel: sdpa; outputs: 1"},
        ],
    }
    c0 = {
        "id": "c0",
        "incomingEdges": [{"sourceNodeId": "k:sdpa", "sourceNodeOutputId": "0"}],
    }
    c1 = {
        "id": "c1",
        "incomingEdges": [{"sourceNodeId": "k:sdpa", "sourceNodeOutputId": "1"}],
    }
    warnings = type_check_graph_nodes([kernel, c0, c1])
    assert any("k:sdpa" in w and "phantom output slot" in w for w in warnings)


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
        {
            "sourceNodeId": "@input:attention_mask",
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "0",
        }
    ]
    warnings = [w for w in integrity_check_graph_nodes([source, node]) if "I2" in w]
    assert warnings == []


# --------------------------------------------------------------------------- #
# No-op cast check: a Cast whose real input dtype already equals its output
# dtype performs no conversion (a redundant cast the merge elision missed).
# --------------------------------------------------------------------------- #


def _cast_node(
    node_id, out_dtype, input_types, *, label="Cast", op_type="Cast", raw_op="float"
):
    """A conversion carrying profiler-style ``input_types``/``output_dtype`` attrs.

    The no-op-cast check reads the operand dtype from the node's own annotation
    (stable across built and render-filtered graphs), not by walking edges.

    It identifies the conversion by the OPERATION the node records plus the
    target dtype the extractor stamped for that call -- not by the node being
    called "Cast". So the node states both, the way a real one does.
    """
    attrs = [
        {"key": "op_type", "value": op_type},
        {"key": "raw_op", "value": raw_op},
        {"key": "details", "value": f"dtype: {out_dtype}"},
        {"key": "input_types", "value": json.dumps(input_types)},
        {"key": "output_dtype", "value": out_dtype},
        {"key": "output_shape", "value": f"[B, S, 4] {out_dtype}"},
    ]
    return {"id": node_id, "label": label, "attrs": attrs}


def test_noop_cast_flagged_when_input_dtype_equals_output():
    nodes = [_cast_node("c", "float32", ["float32"])]
    warnings = type_check_graph_nodes(nodes)
    assert any("c [cast]" in w and "performs no conversion" in w for w in warnings)


def test_noop_cast_not_flagged_on_real_downcast():
    nodes = [_cast_node("c", "float32", ["bfloat16"])]
    warnings = type_check_graph_nodes(nodes)
    assert not any("c [cast]" in w for w in warnings)


def test_noop_cast_not_flagged_when_input_dtype_unknown():
    # No input_types annotation -> operand dtype unknown -> conservative skip.
    nodes = [
        {
            "id": "c",
            "label": "Cast",
            "attrs": [{"key": "output_dtype", "value": "float32"}],
        }
    ]
    warnings = type_check_graph_nodes(nodes)
    assert not any("c [cast]" in w for w in warnings)


def test_noop_cast_not_flagged_on_constant_operand():
    # A learned-weight/buffer dtype conversion (e.g. ``A_log.float()``) records a
    # ``Constant`` operand carrying no concrete dtype -> never a spoofed no-op.
    nodes = [_cast_node("c", "float32", ["Constant"])]
    warnings = type_check_graph_nodes(nodes)
    assert not any("c [cast]" in w for w in warnings)


def test_noop_cast_not_flagged_when_operand_ambiguous():
    # A cast with a spurious spine operand alongside its real (constant) operand --
    # ``["float32", "Constant"]`` from GLM's ``A_log.float()`` -- is ambiguous about
    # which entry is the tensor being cast, so it must NOT be flagged even though
    # the (spine) float32 entry matches the float32 output (constant filtered out in
    # the render graph would otherwise spoof a no-op).
    nodes = [_cast_node("c", "float32", ["float32", "Constant"])]
    warnings = type_check_graph_nodes(nodes)
    assert not any("c [cast]" in w for w in warnings)


def test_noop_cast_flagged_whatever_the_node_is_called():
    # Detection keys on the recorded OPERATION, so the display label is free to
    # be anything at all -- including nothing like "Cast".
    nodes = [_cast_node("c", "float32", ["float32"], label="float", op_type="Alias")]
    warnings = type_check_graph_nodes(nodes)
    assert any("c [cast]" in w for w in warnings)


def test_a_node_called_cast_that_records_no_operation_is_not_judged():
    """A label is not evidence about what an op does."""
    nodes = [
        {
            "id": "c",
            "label": "Cast",
            "attrs": [
                {"key": "op_type", "value": "Cast"},
                {"key": "input_types", "value": json.dumps(["float32"])},
                {"key": "output_dtype", "value": "float32"},
            ],
        }
    ]
    assert not any("c [cast]" in w for w in type_check_graph_nodes(nodes))


def _shape_node(node_id, output_shape, op_type="View", constant=False):
    # A node carrying a rendered ``output_shape`` attr (the string form the export
    # writes, ``[B, S, 4096] bfloat16``) for the unresolved-shape check.
    attrs = [
        {"key": "op_type", "value": op_type},
        {"key": "output_shape", "value": output_shape},
    ]
    if constant:
        attrs.append({"key": "constant", "value": "true"})
    return {"id": node_id, "label": op_type, "attrs": attrs}


def test_resolved_symbolic_shape_is_clean():
    # Named symbolic dims (S, Pv/4, a collapsed product B*S*8192, index_head_dim)
    # are resolved -- just not numeric -- and must never be flagged.
    for shape in (
        "[B, S, 4096] bfloat16",
        "[Pv/4, 4096] bfloat16",
        "[B*S*8192] float32",
        "[B, index_n_heads, S, index_head_dim] float32",
        "[B, 1, S + S, 4096] bfloat16",
    ):
        assert type_check_graph_nodes([_shape_node("n", shape)]) == [], shape


def test_question_mark_dim_warns():
    warnings = type_check_graph_nodes([_shape_node("n:q", "[B, ?, S] float32")])
    assert len(warnings) == 1
    assert "n:q" in warnings[0]
    assert "unresolved dim" in warnings[0]


def test_unfolded_negative_one_dim_warns():
    # A ``-1`` left in a rendered shape is an unfolded reshape placeholder.
    warnings = type_check_graph_nodes([_shape_node("n:neg", "[B, S, -1] bfloat16")])
    assert len(warnings) == 1
    assert "n:neg" in warnings[0]
    assert "-1" in warnings[0]


def test_empty_axis_token_warns():
    # A dropped axis renders as an empty token between commas (``[B, , S]``).
    warnings = type_check_graph_nodes([_shape_node("n:empty", "[B, , S] float32")])
    assert len(warnings) == 1
    assert "n:empty" in warnings[0]


def test_rank0_scalar_shape_is_clean():
    # An empty bracket is a rank-0 scalar (resolved), not an unresolved shape.
    assert type_check_graph_nodes([_shape_node("n:scalar", "[] float32")]) == []


def test_constant_node_unresolved_shape_not_flagged():
    # A learned weight/buffer is filtered from the drawn graph; its shape is not
    # part of the activation dataflow, so it is skipped even if unresolved.
    assert (
        type_check_graph_nodes([_shape_node("n:w", "[?, 4096]", constant=True)]) == []
    )


def test_node_without_output_shape_is_clean():
    # A boundary/module node with no output_shape attr is not an unresolved op.
    node = {
        "id": "n:mod",
        "label": "Linear",
        "attrs": [{"key": "op_type", "value": "Linear"}],
    }
    assert type_check_graph_nodes([node]) == []


# --- H group-cycle check (collapsed-namespace-GROUP level) -------------------


def _grp_node(node_id, namespace, sources=()):
    return {
        "id": node_id,
        "namespace": namespace,
        "incomingEdges": [{"sourceNodeId": s} for s in sources],
    }


def test_h_group_cycle_fires_on_sibling_box_2cycle():
    # The DeepSeek compressor<->kv_norm shape: two sibling sub-boxes of one
    # attention box each feed the other (the spurious back edge closes a rendered
    # 2-cycle) even though the underlying node graph is perfectly acyclic -- the
    # node-level check would miss it. Model it minimally.
    p = "L/Attn"
    nodes = [
        _grp_node(f"{p}/kv_norm:cast", f"{p}/kv_norm", sources=[f"{p}/kv_proj:0"]),
        _grp_node(f"{p}/kv_proj:0", f"{p}", sources=[]),
        # forward edge: kv_norm output feeds the compressor's rotary input.
        _grp_node(
            f"{p}/Compressor:rotary", f"{p}/Compressor", sources=[f"{p}/kv_norm:cast"]
        ),
        # spurious back edge: the compressor's output feeds kv_norm's cast.
        _grp_node(
            f"{p}/kv_norm:cast2",
            f"{p}/kv_norm",
            sources=[f"{p}/Compressor:out"],
        ),
        _grp_node(
            f"{p}/Compressor:out", f"{p}/Compressor", sources=[f"{p}/Compressor:rotary"]
        ),
    ]
    warnings = group_cycle_check_graph_nodes(nodes)
    assert len(warnings) == 1
    assert "L/Attn" in warnings[0]
    assert "kv_norm" in warnings[0]
    assert "Compressor" in warnings[0]


def test_h_group_cycle_clean_when_boxes_do_not_feed_each_other():
    # Same two sibling boxes, but only kv_norm -> compressor (no back edge): a
    # legitimate straight-line dependency, no rendered cycle.
    p = "L/Attn"
    nodes = [
        _grp_node(f"{p}/kv_norm:cast", f"{p}/kv_norm", sources=[f"{p}/kv_proj:0"]),
        _grp_node(f"{p}/kv_proj:0", f"{p}", sources=[]),
        _grp_node(
            f"{p}/Compressor:rotary", f"{p}/Compressor", sources=[f"{p}/kv_norm:cast"]
        ),
        _grp_node(
            f"{p}/Compressor:out", f"{p}/Compressor", sources=[f"{p}/Compressor:rotary"]
        ),
    ]
    assert group_cycle_check_graph_nodes(nodes) == []


def test_h_group_cycle_excludes_sanctioned_loop_carried_back_edge():
    # A repeat group's loop boundary is intentionally cyclic: the carried value's
    # @loop_carried_out feeds the next iteration's @loop_carried_in. That single
    # sanctioned back edge must never be flagged.
    p = "decoder/43x_Layer"
    nodes = [
        # seed, fed only by the sanctioned back edge below.
        _grp_node(
            f"{p}/@loop_carried_in:hs", f"{p}", sources=[f"{p}/@loop_carried_out:hs"]
        ),
        # body reads the seed and produces the iteration output.
        _grp_node(f"{p}/body:out", f"{p}/body", sources=[f"{p}/@loop_carried_in:hs"]),
        # the iteration output exits as the carried-out tile.
        _grp_node(f"{p}/@loop_carried_out:hs", f"{p}", sources=[f"{p}/body:out"]),
    ]
    # The lone back edge (@loop_carried_out -> @loop_carried_in) is sanctioned, so
    # the group-level check is silent even though the graph is cyclic through it.
    assert group_cycle_check_graph_nodes(nodes) == []


def test_h_group_cycle_ignores_legitimate_parent_child_tile_crossings():
    # A parent box's input tile feeds INTO a child box, and the child box feeds
    # a SEPARATE parent output tile back OUT -- boundary tiles legitimately cross
    # a box boundary in both directions. The naive "contract every node to its
    # full namespace globally" approach would draw parent<->child as a 2-cycle
    # (false positive); the per-box immediate-children contraction keeps the two
    # distinct parent tiles as separate leaf units, so there is no cycle.
    p = "L/Attn"
    nodes = [
        _grp_node(f"{p}/@input:hs", f"{p}", sources=[]),
        _grp_node(f"{p}/Norm:op", f"{p}/Norm", sources=[f"{p}/@input:hs"]),  # down
        _grp_node(f"{p}/@output:out", f"{p}", sources=[f"{p}/Norm:op"]),  # up
    ]
    assert group_cycle_check_graph_nodes(nodes) == []


# --------------------------------------------------------------------------- #
# Kernel operand-arity coverage (Task J): a wrapper-expanded kernel must carry a
# DISTINCT tensor input port per required operand of its resolved primitive
# (``kernel_primitive`` attr). sdpa requires query/key/value (the optional
# attn_mask is excluded); a single combined ``kv`` port feeding both key and value
# covers fewer roles than required -- the combined-input defect this check flags.
# The required count is resolved structurally from the primitive's aten schema,
# never a hardcoded op-name -> count table.
# --------------------------------------------------------------------------- #


def _kernel_port(port_id, label, role=None):
    """A kernel operand port. ``role`` is the operand it was bound to supply.

    The role is an identity stamped when the port is bound, which is what the
    coverage check reads -- the label is a display name and may be qualified.
    """
    attrs = [
        {"key": "synthetic", "value": "@kernel_port_in"},
        {"key": "port_label", "value": label},
    ]
    if role:
        attrs.append({"key": "operand_role", "value": role})
    return {"id": port_id, "label": label, "attrs": attrs}


def _sdpa_core_with_ports(
    core_id, port_labels, primitive="scaled_dot_product_attention", roles=None
):
    """An sdpa core node fed by one ``@kernel_port_in`` per label, plus those ports.

    ``roles`` gives each port the operand it supplies, as the wrapper's binding
    would. Omitted, every port declares the operand its label names -- the
    common case where caller var and parameter coincide.
    """
    declared = list(roles) if roles is not None else list(port_labels)
    ports = [
        _kernel_port(
            f"{core_id}/port:{i}", lbl, declared[i] if i < len(declared) else None
        )
        for i, lbl in enumerate(port_labels)
    ]
    core = {
        "id": core_id,
        "label": "sdpa",
        "attrs": [
            {"key": "op_type", "value": "sdpa"},
            {"key": "kernel_primitive", "value": primitive},
        ],
        "incomingEdges": [{"sourceNodeId": p["id"]} for p in ports],
    }
    return [core, *ports]


def test_sdpa_combined_kv_port_flags_missing_operand_coverage():
    # The DeepSeek pre-fix defect: key and value both read one combined ``kv`` port.
    # sdpa requires three distinct tensor operands (query/key/value); ``kv`` covers
    # neither key nor value, so only the query operand is covered -> a warning.
    # The binding gives the combined port ONE role -- a caller var cannot supply
    # two parameters through a single edge -- so ``value`` goes uncovered.
    nodes = _sdpa_core_with_ports(
        "k:sdpa",
        ["q", "kv", "attention_mask"],
        roles=["query", "key", "attention_mask"],
    )
    warnings = type_check_graph_nodes(nodes)
    assert len(warnings) == 1
    assert "k:sdpa" in warnings[0]
    assert "distinct tensor operand" in warnings[0]
    assert "combined input port" in warnings[0]


def test_sdpa_distinct_key_value_ports_is_clean():
    # The post-fix graph: the combined ``kv`` producer fanned into distinct ``key``
    # and ``value`` role ports. All three required operands are now covered --
    # the caller still calls its query ``q``, and the binding says which
    # parameter that is.
    nodes = _sdpa_core_with_ports(
        "k:sdpa",
        ["q", "key", "value", "attention_mask"],
        roles=["query", "key", "value", "attention_mask"],
    )
    assert type_check_graph_nodes(nodes) == []


def test_sdpa_distinct_caller_var_ports_is_clean():
    # Models whose interface already passes distinct key/value caller vars
    # (``query_states``/``key_states``/``value_states``) bind one parameter each
    # and never needed the split. The caller's name differs from the
    # parameter's, which is exactly why the role is carried rather than guessed.
    nodes = _sdpa_core_with_ports(
        "k:sdpa",
        ["query_states", "key_states", "value_states", "attention_mask"],
        roles=["query", "key", "value", "attention_mask"],
    )
    assert type_check_graph_nodes(nodes) == []


def test_sdpa_forwarded_extra_kwarg_port_does_not_break_coverage():
    # A wrapper that forwards an extra non-operand kwarg tensor (MiniMax's
    # ``block_indices``, which the standard sdpa wrapper never passes to the
    # primitive) still covers all three required operands; the extra port must not
    # make the coverage check fire. (The primitive is stamped as ``kernel_primitive``,
    # NOT ``raw_op``, precisely so the generic over-arity ceiling never miscounts it.)
    nodes = _sdpa_core_with_ports(
        "k:sdpa",
        [
            "query_states",
            "key_states",
            "value_states",
            "attention_mask",
            "block_indices",
        ],
        roles=["query", "key", "value", "attention_mask", None],
    )
    assert type_check_graph_nodes(nodes) == []


def test_kernel_primitive_absent_skips_coverage_check():
    # Without a ``kernel_primitive`` attr there is no resolved contract, so the
    # coverage check cannot false-positive on an ordinary node.
    nodes = _sdpa_core_with_ports("k:sdpa", ["q", "kv"])
    nodes[0]["attrs"] = [a for a in nodes[0]["attrs"] if a["key"] != "kernel_primitive"]
    assert type_check_graph_nodes(nodes) == []


def test_coverage_check_ignores_non_kernel_port_sources():
    # A node carrying a ``kernel_primitive`` but fed by ordinary (non
    # ``@kernel_port_in``) edges is not a materialized kernel with declared ports,
    # so the coverage check skips it (leaving it to the ordinary per-op checks).
    core = {
        "id": "k:sdpa",
        "label": "sdpa",
        "attrs": [
            {"key": "op_type", "value": "sdpa"},
            {"key": "kernel_primitive", "value": "scaled_dot_product_attention"},
        ],
        "incomingEdges": [{"sourceNodeId": "some_op"}],
    }
    plain = {"id": "some_op", "label": "Add", "attrs": []}
    assert type_check_graph_nodes([core, plain]) == []


# --------------------------------------------------------------------------- #
# Attribute readers: malformed input must yield "unknown", never a bad answer
# --------------------------------------------------------------------------- #
def _attr_node(**attrs) -> dict:
    return {
        "id": "n",
        "attrs": [{"key": k, "value": v} for k, v in attrs.items()],
    }


def test_load_list_rejects_a_non_string_value():
    assert _tc._load_list({"attrs": [{"key": "k", "value": 5}]}, "k") is None


def test_load_list_rejects_unparseable_json():
    assert _tc._load_list(_attr_node(k="not json"), "k") is None


def test_load_list_rejects_json_that_is_not_a_list():
    assert _tc._load_list(_attr_node(k='{"a": 1}'), "k") is None


def test_load_list_returns_the_parsed_list():
    assert _tc._load_list(_attr_node(k="[1, 2]"), "k") == [1, 2]


def test_load_list_missing_key_is_none():
    assert _tc._load_list(_attr_node(other="[]"), "k") is None


def test_output_shape_dims_handles_missing_and_malformed_values():
    assert _tc._output_shape_dims({"attrs": []}) is None
    assert (
        _tc._output_shape_dims({"attrs": [{"key": "output_shape", "value": 3}]}) is None
    )
    assert _tc._output_shape_dims(_attr_node(output_shape="no brackets")) is None
    assert _tc._output_shape_dims(_attr_node(output_shape="[] float32")) == []
    assert _tc._output_shape_dims(_attr_node(output_shape="[B, S] float32")) == [
        "B",
        "S",
    ]


def test_concrete_int_dims_rejects_symbolic_or_empty_tokens():
    assert _tc._concrete_int_dims(None) is None
    assert _tc._concrete_int_dims(["4", ""]) is None
    assert _tc._concrete_int_dims(["4", "B"]) is None
    assert _tc._concrete_int_dims(["4", "8"]) == [4, 8]


def test_output_dtype_reads_the_suffix_and_tolerates_junk():
    assert _tc._output_dtype(_attr_node(output_shape="[B, S] float32")) == "float32"
    assert _tc._output_dtype(_attr_node(output_shape="[B, S]")) is None
    assert _tc._output_dtype(_attr_node(output_shape="no bracket")) is None
    assert _tc._output_dtype({"attrs": [{"key": "output_shape", "value": 1}]}) is None
    assert _tc._output_dtype({"attrs": []}) is None


def test_a_renamed_port_still_declares_its_operand():
    """A port qualified for the reader keeps the operand identity it was bound to.

    Matching on the display label is what made this impossible: ``key`` against
    ``Reshape.key_states`` shares no prefix either way, so qualifying a port to
    disambiguate it silently uncovered a required operand and the check fired
    on a correct graph.
    """
    nodes = _sdpa_core_with_ports(
        "k:sdpa",
        ["query_states", "Reshape.key_states", "Reshape.value_states", "mask"],
        roles=["query", "key", "value", "attention_mask"],
    )
    assert type_check_graph_nodes(nodes) == []


def test_ports_that_declare_no_operand_are_not_guessed_at():
    """No role, no claim -- a display label is not evidence about wiring."""
    nodes = _sdpa_core_with_ports(
        "k:sdpa", ["q", "kv", "attention_mask"], roles=[None, None, None]
    )
    assert type_check_graph_nodes(nodes) == []


def test_a_renamed_size_boundary_is_not_a_second_tensor_operand():
    """``host_params`` names the parameter; the tile may have been renamed.

    ``torch.split(t, lengths.tolist(), dim=2)`` reads ``cu_seqlens`` on the
    host. The edge says so, even when the boundary tile is labelled for a
    different tensor. Counting that edge as a tensor operand reports a
    one-operand op as taking two.
    """
    from TraceLens.Visualizer.model_explorer_export.merge import (
        _annotate_op_input_signatures,
    )

    nodes = [
        {
            "id": "attn/q",
            "label": "query",
            "attrs": [
                {"key": "output_dtype", "value": "bfloat16"},
                {"key": "output_shape", "value": "[1, 24, nnz, 64]"},
            ],
            "incomingEdges": [],
        },
        {
            "id": "attn/@input:window_index",
            "label": "window_index",
            "attrs": [
                {"key": "port_label", "value": "window_index"},
                {"key": "output_dtype", "value": "int64"},
                {"key": "output_shape", "value": "[nnz]"},
            ],
            "incomingEdges": [],
        },
        {
            "id": "attn/split",
            "label": "Split",
            "attrs": [
                {"key": "raw_op", "value": "split"},
                {
                    "key": "details",
                    "value": "host_params: cu_seqlens",
                },
            ],
            "incomingEdges": [
                {
                    "sourceNodeId": "attn/q",
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                },
                {
                    "sourceNodeId": "attn/@input:window_index",
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "1",
                },
            ],
        },
    ]
    _annotate_op_input_signatures(nodes)
    warnings = [w for w in type_check_graph_nodes(nodes) if "split" in w]
    assert warnings == []


def _branch(node_id, edges, *, synthetic=False):
    attrs = [{"key": "class_name", "value": "Merge"}]
    if synthetic:
        attrs.append({"key": "synthetic", "value": "true"})
    return {
        "id": node_id,
        "label": "Mutually exclusive branch",
        "attrs": attrs,
        "incomingEdges": edges,
    }


def _edge(source):
    return {
        "sourceNodeId": source,
        "sourceNodeOutputId": "0",
        "targetNodeInputId": "0",
    }


def test_a_branch_with_one_live_input_warns_at_the_end_of_generation():
    nodes = [
        {"id": "live", "label": "Linear", "incomingEdges": []},
        {"id": "missing-only", "label": "Linear", "incomingEdges": []},
        _branch("one", [_edge("live")]),
        _branch("two", [_edge("live"), _edge("missing-only")]),
        _branch("boundary", [_edge("live")], synthetic=True),
    ]
    warnings = warn_degenerate_mutually_exclusive_branches(nodes)
    assert [line.split(":")[0] for line in warnings] == ["one"]


# --- test_declared_axis_exists.py ---

import json

from TraceLens.Visualizer.model_explorer_export.type_check import (
    _declared_axis_warnings,
)


def _node__declared_axis_exists(shapes, details, op_type="Slice", node_id="n1"):
    """A node as the exporter writes one: attrs carry JSON / rendered strings."""
    return {
        "id": node_id,
        "attrs": [
            {"key": "op_type", "value": op_type},
            {"key": "input_shapes", "value": json.dumps(shapes)},
            {"key": "details", "value": "; ".join(details)},
        ],
    }


class TestTheAxisMustExist:
    def test_an_axis_beyond_the_operands_rank(self) -> None:
        """Kimi's case: a rank-4 narrowing handed a rank-3 tensor."""
        warnings = _declared_axis_warnings(
            [_node__declared_axis_exists([["B", "S", "7168"]], ["resize_dim: 3=128"])]
        )
        assert len(warnings) == 1
        assert "rank 3" in warnings[0] and "resize_dim 3" in warnings[0]

    def test_a_select_dim_too(self) -> None:
        warnings = _declared_axis_warnings(
            [_node__declared_axis_exists([["B", "S"]], ["select_dim: 2"])]
        )
        assert len(warnings) == 1

    def test_the_operand_two_ranks_short(self) -> None:
        """DeepSeek's compressor: ``resize_dim: 3`` against ``[B*S, 1024]``."""
        warnings = _declared_axis_warnings(
            [_node__declared_axis_exists([["B*S", "1024"]], ["resize_dim: 3=256"])]
        )
        assert len(warnings) == 1


class TestWhatIsAccepted__declared_axis_exists:
    def test_an_axis_the_operand_has(self) -> None:
        assert (
            _declared_axis_warnings(
                [
                    _node__declared_axis_exists(
                        [["B", 96, "S", 128]], ["resize_dim: 3=128"]
                    )
                ]
            )
            == []
        )

    def test_the_last_axis_exactly(self) -> None:
        assert (
            _declared_axis_warnings(
                [_node__declared_axis_exists([["B", "S"]], ["select_dim: 1"])]
            )
            == []
        )

    def test_a_negative_axis_counts_from_the_end(self) -> None:
        assert (
            _declared_axis_warnings(
                [_node__declared_axis_exists([["B", "S", "H"]], ["resize_dim: -1=64"])]
            )
            == []
        )

    def test_a_negative_axis_still_out_of_range(self) -> None:
        assert (
            len(
                _declared_axis_warnings(
                    [_node__declared_axis_exists([["B"]], ["resize_dim: -3=4"])]
                )
            )
            == 1
        )

    def test_the_first_tensor_operand_is_the_one_addressed(self) -> None:
        """A trailing scalar operand carries no axes and must not be read as one."""
        assert (
            _declared_axis_warnings(
                [
                    _node__declared_axis_exists(
                        [["B", 96, "S", 128], []], ["resize_dim: 3=128"]
                    )
                ]
            )
            == []
        )


class TestWhatItDeclines:
    def test_a_node_with_no_declaration(self) -> None:
        assert (
            _declared_axis_warnings(
                [_node__declared_axis_exists([["B", "S"]], ["raw_op: slice"])]
            )
            == []
        )

    def test_a_node_with_no_operand_shapes(self) -> None:
        assert (
            _declared_axis_warnings(
                [_node__declared_axis_exists([], ["resize_dim: 3=128"])]
            )
            == []
        )

    def test_a_declaration_that_is_not_an_index(self) -> None:
        """``shape_slice`` narrows by arithmetic and names no axis index."""
        assert (
            _declared_axis_warnings(
                [
                    _node__declared_axis_exists(
                        [["B", "S"]], ["shape_slice: x.shape[-1] - 1"]
                    )
                ]
            )
            == []
        )

    def test_an_unparseable_axis(self) -> None:
        assert (
            _declared_axis_warnings(
                [_node__declared_axis_exists([["B"]], ["resize_dim: n=4"])]
            )
            == []
        )

    def test_an_empty_graph(self) -> None:
        assert _declared_axis_warnings([]) == []


# --- test_declared_rank_holds.py ---

import json

from TraceLens.Visualizer.model_explorer_export.type_check import (
    _declared_rank_warnings,
    _spans_several_axes,
)


def _node__declared_rank_holds(details, output_shape, op_type="View", node_id="n1"):
    return {
        "id": node_id,
        "attrs": [
            {"key": "op_type", "value": op_type},
            {"key": "details", "value": "; ".join(details)},
            {"key": "output_shape", "value": output_shape},
        ],
    }


class TestTheRankMustMatch:
    def test_fewer_axes_than_declared(self) -> None:
        """DeepSeek's compressor: four declared, two produced."""
        warnings = _declared_rank_warnings(
            [
                _node__declared_rank_holds(
                    ["shape: hidden_states.shape[0], n_windows, ratio, -1"],
                    "[B*S, 1024] bfloat16",
                )
            ]
        )
        assert len(warnings) == 1
        assert "declares 4 target axes" in warnings[0]
        assert "has 2" in warnings[0]

    def test_more_axes_than_declared(self) -> None:
        """MiniMax's repeat_kv: the collapsing reshape never happened."""
        warnings = _declared_rank_warnings(
            [
                _node__declared_rank_holds(
                    ["shape: a, b, c, d"],
                    "[B, 4, 1, S, 128] bfloat16",
                    op_type="Reshape",
                )
            ]
        )
        assert len(warnings) == 1

    def test_a_following_detail_is_not_read_as_more_axes(self) -> None:
        """Details render ``a; b`` on one line; the shape ends at the ``;``."""
        warnings = _declared_rank_warnings(
            [
                _node__declared_rank_holds(
                    [
                        "shape: hidden_states.shape[0], hidden_states.shape[1], -1, 128",
                        "shape_snapshots: hidden_states.shape[0], hidden_states.shape[1]",
                    ],
                    "[B, S, 64, 128] bfloat16",
                )
            ]
        )
        assert warnings == []


class TestWhatIsAccepted__declared_rank_holds:
    def test_the_rank_it_declared(self) -> None:
        assert (
            _declared_rank_warnings(
                [
                    _node__declared_rank_holds(
                        ["shape: a, b, -1"], "[B, S, 4096] bfloat16"
                    )
                ]
            )
            == []
        )

    def test_a_symbolic_axis_is_nobody_s_business_here(self) -> None:
        """Only the count is asserted -- an unresolved dim is not this check's."""
        assert (
            _declared_rank_warnings(
                [
                    _node__declared_rank_holds(
                        ["shape: batch, n_windows, ratio, -1"], "[B, n_windows, 4, 512]"
                    )
                ]
            )
            == []
        )

    def test_one_argument_standing_for_a_whole_shape(self) -> None:
        """``x.reshape(x.shape)`` says nothing about how many axes come out."""
        assert (
            _declared_rank_warnings(
                [_node__declared_rank_holds(["shape: orig_shape"], "[B, S, H]")]
            )
            == []
        )

    def test_an_unpacked_shape_slice(self) -> None:
        """``*x.shape[:2]`` stands for several axes, so the count is not the rank."""
        assert (
            _declared_rank_warnings(
                [
                    _node__declared_rank_holds(
                        ["shape: *hidden.shape[:2], -1"], "[B, S, 64, 128]"
                    )
                ]
            )
            == []
        )

    def test_a_constant_node_is_skipped(self) -> None:
        node = _node__declared_rank_holds(["shape: a, b, c"], "[B, S]")
        node["attrs"].append({"key": "constant", "value": "true"})
        assert _declared_rank_warnings([node]) == []

    def test_a_node_with_no_declaration(self) -> None:
        assert (
            _declared_rank_warnings(
                [_node__declared_rank_holds(["raw_op: view"], "[B, S]")]
            )
            == []
        )

    def test_a_node_with_no_output_shape(self) -> None:
        node = {
            "id": "n",
            "attrs": [
                {"key": "op_type", "value": "View"},
                {"key": "details", "value": "shape: a, b, c"},
            ],
        }
        assert _declared_rank_warnings([node]) == []

    def test_an_empty_graph(self) -> None:
        assert _declared_rank_warnings([]) == []


class TestWhichArgumentsSpanSeveralAxes:
    def test_an_unpack(self) -> None:
        assert _spans_several_axes("*x.shape[:2]") is True

    def test_a_shape_slice(self) -> None:
        assert _spans_several_axes("x.shape[:-1]") is True

    def test_a_scalar_shape_read_is_one_axis(self) -> None:
        assert _spans_several_axes("hidden_states.shape[0]") is False

    def test_a_plain_name_or_literal(self) -> None:
        assert _spans_several_axes("n_windows") is False
        assert _spans_several_axes("-1") is False


def test_json_fixture_shape_matches_the_exporter() -> None:
    """Guards the fixture itself: attrs are rendered strings, not objects."""
    node = _node__declared_rank_holds(["shape: a, b"], "[B, S]")
    assert all(isinstance(a["value"], str) for a in node["attrs"])
    assert json.loads("[]") == []


# --- test_index_operand_dtype.py ---

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


# --- test_phi_forwards_an_arm.py ---

from TraceLens.ModelUtils.ast_analyze import PHI_CLASS, PHI_LABEL
from TraceLens.Visualizer.model_explorer_export.type_check import (
    _phi_forwards_an_arm_warnings,
)


def _node__phi_forwards_an_arm(
    node_id, shape, *, label="op", phi=False, constant=False, sources=()
):
    attrs = [{"key": "output_shape", "value": shape}]
    if phi:
        attrs.append({"key": "class_name", "value": PHI_CLASS})
    if constant:
        attrs.append({"key": "constant", "value": "true"})
    return {
        "id": node_id,
        "label": PHI_LABEL if phi else label,
        "attrs": attrs,
        "incomingEdges": [{"sourceNodeId": s} for s in sources],
    }


class TestTheShapeMustBeAnArmsShape:
    def test_a_shape_no_arm_has(self) -> None:
        nodes = [
            _node__phi_forwards_an_arm("a", "[B, S, 4096] bfloat16"),
            _node__phi_forwards_an_arm("b", "[B, S, 4096] bfloat16"),
            _node__phi_forwards_an_arm(
                "phi", "[B, S, 1, 4096] bfloat16", phi=True, sources=["a", "b"]
            ),
        ]
        warnings = _phi_forwards_an_arm_warnings(nodes)
        assert len(warnings) == 1
        assert "must be one of theirs" in warnings[0]

    def test_the_shape_of_one_arm(self) -> None:
        nodes = [
            _node__phi_forwards_an_arm("a", "[B, S, 4096] bfloat16"),
            _node__phi_forwards_an_arm("b", "[B, S, 4096] bfloat16"),
            _node__phi_forwards_an_arm(
                "phi", "[B, S, 4096] bfloat16", phi=True, sources=["a", "b"]
            ),
        ]
        assert _phi_forwards_an_arm_warnings(nodes) == []

    def test_arms_that_disagree_are_not_this_checks_business(self) -> None:
        """Kimi's unpad: one branch really does hand back another shape."""
        nodes = [
            _node__phi_forwards_an_arm("padded", "[B, S, 7168] bfloat16"),
            _node__phi_forwards_an_arm("unpadded", "[1, nnz, 7168] bfloat16"),
            _node__phi_forwards_an_arm(
                "phi",
                "[1, nnz, 7168] bfloat16",
                phi=True,
                sources=["padded", "unpadded"],
            ),
        ]
        assert _phi_forwards_an_arm_warnings(nodes) == []

    def test_a_dtype_difference_is_not_a_shape(self) -> None:
        nodes = [
            _node__phi_forwards_an_arm("a", "[B, S, 4096] float32"),
            _node__phi_forwards_an_arm("b", "[B, S, 4096] bfloat16"),
            _node__phi_forwards_an_arm(
                "phi", "[B, S, 4096] bfloat16", phi=True, sources=["a", "b"]
            ),
        ]
        assert _phi_forwards_an_arm_warnings(nodes) == []

    def test_the_phi_is_found_by_its_label_too(self) -> None:
        nodes = [
            _node__phi_forwards_an_arm("a", "[B, S] bfloat16"),
            _node__phi_forwards_an_arm("b", "[B, S] bfloat16"),
            {
                "id": "phi",
                "label": PHI_LABEL,
                "attrs": [{"key": "output_shape", "value": "[B] bfloat16"}],
                "incomingEdges": [{"sourceNodeId": "a"}, {"sourceNodeId": "b"}],
            },
        ]
        assert len(_phi_forwards_an_arm_warnings(nodes)) == 1


class TestWhatItSkips:
    def test_a_constant_arm_is_not_a_branch(self) -> None:
        nodes = [
            _node__phi_forwards_an_arm("w", "[4096] bfloat16", constant=True),
            _node__phi_forwards_an_arm("a", "[B, S, 4096] bfloat16"),
            _node__phi_forwards_an_arm(
                "phi", "[B, S, 4096] bfloat16", phi=True, sources=["w", "a"]
            ),
        ]
        assert _phi_forwards_an_arm_warnings(nodes) == []

    def test_a_boundary_tile_carrying_a_phis_output(self) -> None:
        """A tile takes its name from the tensor; it is not the op."""
        node = _node__phi_forwards_an_arm("t", "[B] bfloat16", phi=True, sources=["a"])
        node["attrs"].append({"key": "synthetic", "value": "@output"})
        assert (
            _phi_forwards_an_arm_warnings(
                [_node__phi_forwards_an_arm("a", "[B, S] bfloat16"), node]
            )
            == []
        )

    def test_an_ordinary_op(self) -> None:
        nodes = [
            _node__phi_forwards_an_arm("a", "[B, S] bfloat16"),
            _node__phi_forwards_an_arm(
                "add", "[B, S, 1] bfloat16", label="Add", sources=["a"]
            ),
        ]
        assert _phi_forwards_an_arm_warnings(nodes) == []

    def test_a_phi_with_no_shape(self) -> None:
        nodes = [
            _node__phi_forwards_an_arm("a", "[B, S] bfloat16"),
            {
                "id": "phi",
                "label": PHI_LABEL,
                "attrs": [{"key": "class_name", "value": PHI_CLASS}],
                "incomingEdges": [{"sourceNodeId": "a"}],
            },
        ]
        assert _phi_forwards_an_arm_warnings(nodes) == []

    def test_a_phi_whose_arms_have_no_shape(self) -> None:
        nodes = [
            {"id": "a", "label": "op", "attrs": []},
            _node__phi_forwards_an_arm(
                "phi", "[B, S] bfloat16", phi=True, sources=["a"]
            ),
        ]
        assert _phi_forwards_an_arm_warnings(nodes) == []

    def test_an_empty_graph(self) -> None:
        assert _phi_forwards_an_arm_warnings([]) == []


# --- test_projection_width_agreement.py ---

from TraceLens.Visualizer.model_explorer_export.type_check import (
    projection_weight_bias_widths,
)


def _projection(weight: str, bias: str, raw_op: str = "addmm") -> list[dict]:
    return [
        {
            "id": "m/proj",
            "label": "Linear",
            "attrs": [{"key": "raw_op", "value": raw_op}],
        },
        {
            "id": "m/proj:const:weight",
            "label": "weight",
            "attrs": [{"key": "output_shape", "value": weight}],
        },
        {
            "id": "m/proj:const:bias",
            "label": "bias",
            "attrs": [{"key": "output_shape", "value": bias}],
        },
    ]


class TestTheDefectItCatches__projection_width_agreement:
    def test_a_weight_from_another_module_is_reported(self) -> None:
        """GPT-2's `c_proj` held `c_attn`'s weight against its own bias."""
        assert projection_weight_bias_widths(
            _projection("[768, 2304] float32", "[768] float32")
        )

    def test_the_message_names_both_widths(self) -> None:
        (message,) = projection_weight_bias_widths(
            _projection("[768, 2304] float32", "[768] float32")
        )
        assert "2304" in message and "768" in message
        assert "DIFFERENT modules" in message


class TestWhatItAccepts__projection_width_agreement:
    def test_every_real_gpt2_projection(self) -> None:
        """The four, once each resolved its own weight."""
        for weight, bias in (
            ("[768, 2304] float32", "[2304] float32"),  # c_attn
            ("[768, 768] float32", "[768] float32"),  # attn c_proj
            ("[768, 3072] float32", "[3072] float32"),  # c_fc
            ("[3072, 768] float32", "[768] float32"),  # mlp c_proj
        ):
            assert projection_weight_bias_widths(_projection(weight, bias)) == [], (
                weight,
                bias,
            )

    def test_the_other_weight_layout(self) -> None:
        """``F.linear`` stores [out, in], so the FIRST axis is the width."""
        assert (
            projection_weight_bias_widths(
                _projection("[768, 3072] float32", "[768] float32", raw_op="linear")
            )
            == []
        )


class TestWhereItSaysNothing__projection_width_agreement:
    def test_an_op_whose_layout_is_unknown(self) -> None:
        assert (
            projection_weight_bias_widths(
                _projection("[768, 2304] float32", "[768] float32", raw_op="mystery")
            )
            == []
        )

    def test_a_missing_bias(self) -> None:
        nodes = _projection("[768, 2304] float32", "[768] float32")
        assert projection_weight_bias_widths(nodes[:2]) == []

    def test_a_symbolic_weight(self) -> None:
        assert (
            projection_weight_bias_widths(
                _projection("[H, I] float32", "[768] float32")
            )
            == []
        )

    def test_a_non_matrix_weight(self) -> None:
        """A norm's weight is one axis -- not a projection's."""
        assert (
            projection_weight_bias_widths(_projection("[768] float32", "[768] float32"))
            == []
        )


# --- test_shape_checks_edge_cases.py ---

import json

from TraceLens.Visualizer.model_explorer_export.type_check import (
    _counts_provably_differ,
    _dim_factors,
    _in_place_write_warnings,
    _reshape_conservation_warnings,
    _shape_preserving_op_warnings,
    projection_weight_bias_widths,
)


def _node__shape_checks_edge_cases(node_id="n", label="Cast", **attrs):
    return {
        "id": node_id,
        "label": label,
        "attrs": [{"key": key, "value": value} for key, value in attrs.items()],
    }


class TestAShapePreservingOpWithAnUnusableRecord:
    def test_input_shapes_that_are_not_json(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="float", input_shapes="{not json", output_shape="[B] f32"
        )
        assert _shape_preserving_op_warnings([node]) == []

    def test_input_shapes_that_are_not_a_list(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="float", input_shapes='"B, S"', output_shape="[B, S] f32"
        )
        assert _shape_preserving_op_warnings([node]) == []

    def test_an_empty_operand(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="float", input_shapes="[[]]", output_shape="[B, S] f32"
        )
        assert _shape_preserving_op_warnings([node]) == []

    def test_no_output_shape_recorded(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="float", input_shapes='[["B", "S"]]'
        )
        assert _shape_preserving_op_warnings([node]) == []

    def test_no_input_shapes_recorded(self) -> None:
        assert (
            _shape_preserving_op_warnings(
                [_node__shape_checks_edge_cases(raw_op="float")]
            )
            == []
        )


class TestReadingAShapeAsAnElementCount:
    def test_a_product_is_the_same_count_written_differently(self) -> None:
        assert _dim_factors(["B", "S*768"]) == _dim_factors(["B", "S", "768"])

    def test_a_size_one_axis_multiplies_nothing(self) -> None:
        assert _dim_factors(["1", "8192", "1"]) == ["8192"]

    def test_a_placeholder_axis_states_no_count(self) -> None:
        """``-1`` stands for whatever makes the count come out right."""
        assert _dim_factors(["B", "-1"]) is None

    def test_an_empty_token_states_no_count(self) -> None:
        assert _dim_factors(["B", ""]) is None

    def test_counts_that_cancel_do_not_differ(self) -> None:
        assert not _counts_provably_differ(["B", "S", "768"], ["B", "S*768"])

    def test_two_symbolic_remainders_prove_nothing(self) -> None:
        """The same count can be written with different symbols."""
        assert not _counts_provably_differ(["Pv", "1176"], ["Img", "3"])

    def test_literal_remainders_that_disagree(self) -> None:
        assert _counts_provably_differ(["B", "S", "768"], ["B", "S", "1024"])

    def test_a_placeholder_on_either_side_proves_nothing(self) -> None:
        assert not _counts_provably_differ(["B", "-1"], ["B", "S", "768"])
        assert not _counts_provably_differ(["B", "S", "768"], ["B", "-1"])


class TestTheReshapeConservationCheck:
    def test_a_view_that_cannot_hold_its_elements(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="view",
            label="View",
            input_shapes='[["B*S", "33792"]]',
            output_shape="[B, S, 7168] bfloat16",
        )
        (message,) = _reshape_conservation_warnings([node])
        assert "same number of elements" in message

    def test_a_view_that_only_moves_the_axes(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="view",
            label="View",
            input_shapes='[["B*S", "4096"]]',
            output_shape="[B, S, 4096] bfloat16",
        )
        assert _reshape_conservation_warnings([node]) == []

    def test_a_reshape_that_converts_a_dtype(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="view",
            label="View",
            details="shape: *orig_shape",
            input_shapes='[["nnz", "1"]]',
            input_types='["int64"]',
            output_shape="[nnz, 1] bfloat16",
            output_dtype="bfloat16",
        )
        (message,) = _reshape_conservation_warnings([node])
        assert "int64" in message and "bfloat16" in message

    def test_malformed_input_shapes(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="view", input_shapes="{{", output_shape="[B] f32"
        )
        assert _reshape_conservation_warnings([node]) == []

    def test_a_constant_is_skipped(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="view",
            constant="true",
            input_shapes='[["B*S", "33792"]]',
            output_shape="[B, S, 7168] bfloat16",
        )
        assert _reshape_conservation_warnings([node]) == []

    def test_an_op_that_may_change_the_count(self) -> None:
        """``expand`` really does make more elements than it was given."""
        node = _node__shape_checks_edge_cases(
            raw_op="expand",
            input_shapes='[["B", "1", "S"]]',
            output_shape="[B, 4, S] bfloat16",
        )
        assert _reshape_conservation_warnings([node]) == []

    def test_a_declared_shape_with_no_recorded_dtypes(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="view",
            details="shape: *orig_shape",
            input_shapes='[["B", "S"]]',
            output_shape="[B, S] bfloat16",
        )
        assert _reshape_conservation_warnings([node]) == []

    def test_a_dtype_that_carries_no_concrete_type(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="view",
            details="shape: *orig_shape",
            input_shapes='[["B", "S"]]',
            input_types='["Constant"]',
            output_shape="[B, S] bfloat16",
            output_dtype="bfloat16",
        )
        assert _reshape_conservation_warnings([node]) == []


class TestTheInPlaceWriteCheck:
    def test_a_write_reporting_its_index_instead_of_its_receiver(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="index_add_",
            label="Index add",
            details="mutates: final; loop: 288 iterations",
            input_shapes='[["nnz", "1"], ["nnz", "1"], ["1"]]',
            output_shape="[nnz, 1] int64",
        )
        # Reads what it writes into and reports the same: no complaint.
        assert _in_place_write_warnings([node]) == []

    def test_a_receiver_and_a_result_that_disagree(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="index_add_",
            label="Index add",
            details="mutates: final",
            input_shapes='[["B*S", "4096"], ["nnz", "1"], ["1"]]',
            output_shape="[nnz, 1] int64",
        )
        (message,) = _in_place_write_warnings([node])
        assert "writes into" in message and "'final'" in message

    def test_a_write_that_changes_the_dtype_it_writes_into(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="index_add_",
            label="Index add",
            details="mutates: final",
            input_shapes='[["B*S", "4096"], ["nnz", "1"]]',
            input_types='["bfloat16", "int64"]',
            output_shape="[B*S, 4096] int64",
            output_dtype="int64",
        )
        messages = _in_place_write_warnings([node])
        assert any("keeps the dtype" in m for m in messages)

    def test_a_node_that_mutates_nothing(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="add",
            input_shapes='[["B", "S"]]',
            output_shape="[B, S, 4] f32",
        )
        assert _in_place_write_warnings([node]) == []

    def test_a_constant_is_skipped(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="copy_",
            constant="true",
            details="mutates: x",
            input_shapes='[["B", "S"]]',
            output_shape="[B] f32",
        )
        assert _in_place_write_warnings([node]) == []

    def test_no_shapes_recorded(self) -> None:
        node = _node__shape_checks_edge_cases(raw_op="copy_", details="mutates: x")
        assert _in_place_write_warnings([node]) == []

    def test_an_empty_receiver_record(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="copy_",
            details="mutates: x",
            input_shapes="[[]]",
            output_shape="[B, S] f32",
        )
        assert _in_place_write_warnings([node]) == []

    def test_a_receiver_dtype_that_carries_no_concrete_type(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="copy_",
            details="mutates: x",
            input_shapes='[["B", "S"]]',
            input_types='["Tensor"]',
            output_shape="[B, S] f32",
            output_dtype="f32",
        )
        assert _in_place_write_warnings([node]) == []


class TestTheProjectionWidthCheck:
    def test_a_weight_whose_shape_carries_no_numbers(self) -> None:
        nodes = [
            {
                "id": "m/@op_x:const:weight",
                "label": "weight",
                "attrs": [{"key": "output_shape", "value": "[in, out] bfloat16"}],
            }
        ]
        assert projection_weight_bias_widths(nodes) == []

    def test_a_constant_with_no_recorded_shape(self) -> None:
        nodes = [{"id": "m/@op_x:const:weight", "label": "weight", "attrs": []}]
        assert projection_weight_bias_widths(nodes) == []

    def test_a_constant_that_is_neither_weight_nor_bias(self) -> None:
        nodes = [
            {
                "id": "m/@op_x:const:eps",
                "label": "eps",
                "attrs": [{"key": "output_shape", "value": "[1] bfloat16"}],
            }
        ]
        assert projection_weight_bias_widths(nodes) == []

    def test_a_node_that_is_not_a_constant(self) -> None:
        nodes = [
            {
                "id": "m/@op_x",
                "label": "weight",
                "attrs": [{"key": "output_shape", "value": "[768, 2304] bfloat16"}],
            }
        ]
        assert projection_weight_bias_widths(nodes) == []


class TestEveryCheckSurvivesAnEmptyGraph:
    def test_nothing_to_look_at(self) -> None:
        for check in (
            _shape_preserving_op_warnings,
            _reshape_conservation_warnings,
            _in_place_write_warnings,
            projection_weight_bias_widths,
        ):
            assert check([]) == [], check.__name__

    def test_a_node_with_no_attrs_at_all(self) -> None:
        bare = [{"id": "n", "label": "x"}]
        for check in (
            _shape_preserving_op_warnings,
            _reshape_conservation_warnings,
            _in_place_write_warnings,
            projection_weight_bias_widths,
        ):
            assert check(bare) == [], check.__name__

    def test_input_shapes_holding_a_non_list_entry(self) -> None:
        node = _node__shape_checks_edge_cases(
            raw_op="float",
            input_shapes=json.dumps(["B, S"]),
            output_shape="[B, S] f32",
        )
        assert _shape_preserving_op_warnings([node]) == []


# --- test_shape_preserving_ops.py ---

import json

from TraceLens.ModelUtils.shape_inference import TensorSpec
from TraceLens.Visualizer.model_explorer_export.shapes import (
    apply_shape_attrs,
    refine_multi_output_port_shapes,
    widen_broadcast_results,
)
from TraceLens.Visualizer.model_explorer_export.type_check import (
    _shape_preserving_op_warnings,
)


def _node__shape_preserving_ops(
    reads: list[str] | None,
    produces: str | None,
    *,
    raw_op: str | None = "float",
    op_type: str | None = "Cast",
    label: str = "Cast",
    **extra,
) -> dict:
    attrs = []
    if op_type is not None:
        attrs.append({"key": "op_type", "value": op_type})
    if raw_op is not None:
        attrs.append({"key": "raw_op", "value": raw_op})
    if reads is not None:
        attrs.append({"key": "input_shapes", "value": json.dumps([reads])})
    if produces is not None:
        attrs.append({"key": "output_shape", "value": produces})
    for key, value in extra.items():
        attrs.append({"key": key, "value": value})
    return {"id": "m/@op_x", "label": label, "attrs": attrs}


class TestTheDefectItCatches__shape_preserving_ops:
    def test_the_cast_that_changed_an_axis(self) -> None:
        assert _shape_preserving_op_warnings(
            [
                _node__shape_preserving_ops(
                    ["B", "S", "64", "128"], "[B, S, 48, 128] float32"
                )
            ]
        )

    def test_the_message_shows_both_shapes(self) -> None:
        (message,) = _shape_preserving_op_warnings(
            [
                _node__shape_preserving_ops(
                    ["B", "S", "64", "128"], "[B, S, 48, 128] float32"
                )
            ]
        )
        assert "'64'" in message and "'48'" in message

    def test_a_dropped_axis_is_caught_too(self) -> None:
        assert _shape_preserving_op_warnings(
            [_node__shape_preserving_ops(["B", "S", "128"], "[B, S] float32")]
        )

    def test_the_final_edge_repairs_a_stale_same_rank_shape(self) -> None:
        source = {
            "id": "source",
            "label": "Mean",
        }
        apply_shape_attrs(source, TensorSpec(("B", "S", 1), "bfloat16"))
        target = _node__shape_preserving_ops(
            ["B", "S", "1"],
            "[1] bfloat16",
            raw_op="rsqrt",
            op_type="Reciprocal sqrt",
            label="Reciprocal sqrt",
        )
        apply_shape_attrs(target, TensorSpec((1,), "bfloat16"))
        target["incomingEdges"] = [
            {
                "sourceNodeId": "source",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "0",
            }
        ]
        widen_broadcast_results([source, target])
        shape = next(
            attr["value"] for attr in target["attrs"] if attr["key"] == "output_shape"
        )
        assert shape == "[B, S, 1] bfloat16"

    def test_the_final_edge_also_repairs_a_stale_wider_shape(self) -> None:
        source = {"id": "source", "label": "Mean"}
        apply_shape_attrs(source, TensorSpec((1,), "bfloat16"))
        target = _node__shape_preserving_ops(
            ["1"],
            "[B, S, 1] bfloat16",
            raw_op="rsqrt",
            op_type="Reciprocal sqrt",
            label="Reciprocal sqrt",
        )
        apply_shape_attrs(target, TensorSpec(("B", "S", 1), "bfloat16"))
        target["incomingEdges"] = [
            {
                "sourceNodeId": "source",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "0",
            }
        ]
        widen_broadcast_results([source, target])
        shape = next(
            attr["value"] for attr in target["attrs"] if attr["key"] == "output_shape"
        )
        assert shape == "[1] bfloat16"


class TestFinalMultiOutputPortShapes:
    def test_chunk_ports_are_slices_of_the_final_whole_shape(self) -> None:
        node = {
            "id": "chunk",
            "label": "Chunk",
            "attrs": [
                {"key": "raw_op", "value": "chunk"},
                {"key": "details", "value": "split_size: 2; dim: -1"},
            ],
            "outputsMetadata": [
                {"id": "0", "attrs": [{"key": "port_label", "value": "left"}]},
                {"id": "1", "attrs": [{"key": "port_label", "value": "right"}]},
            ],
        }
        left = {
            "id": "chunk^@slice_out:0",
            "label": "left",
            "attrs": [{"key": "synthetic", "value": "@slice_out"}],
            "incomingEdges": [
                {
                    "sourceNodeId": "chunk",
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            ],
        }
        apply_shape_attrs(node, TensorSpec(("B", "S", 24, 512), "bfloat16"))
        # ``apply_shape_attrs`` gives every port the whole shape; the final
        # refinement reads the operation's own contract and divides them.
        apply_shape_attrs(left, TensorSpec(("B", "S", 24, 512), "bfloat16"))
        refine_multi_output_port_shapes([node, left])
        refine_multi_output_port_shapes([node, left])
        shapes = [
            next(attr["value"] for attr in port["attrs"] if attr["key"] == "shape")
            for port in node["outputsMetadata"]
        ]
        assert shapes == [
            "[B, S, 24, 256] bfloat16",
            "[B, S, 24, 256] bfloat16",
        ]
        left_shape = next(
            attr["value"] for attr in left["attrs"] if attr["key"] == "output_shape"
        )
        assert left_shape == "[B, S, 24, 256] bfloat16"


class TestTheRestOfTheClass:
    """The same overwrite on an op the Cast-only check could not see."""

    def test_an_activation(self) -> None:
        assert _shape_preserving_op_warnings(
            [
                _node__shape_preserving_ops(
                    ["B", "S", "64", "128"],
                    "[B, S, 48, 128] float32",
                    raw_op="silu",
                    op_type=None,
                    label="SiLU",
                )
            ]
        )

    def test_a_dropout(self) -> None:
        assert _shape_preserving_op_warnings(
            [
                _node__shape_preserving_ops(
                    ["B", "S", "768"],
                    "[B, S, 1024] float32",
                    raw_op="dropout",
                    op_type=None,
                    label="Dropout",
                )
            ]
        )

    def test_a_layout_copy(self) -> None:
        assert _shape_preserving_op_warnings(
            [
                _node__shape_preserving_ops(
                    ["B", "S", "768"],
                    "[B, S, 1024] float32",
                    raw_op="contiguous",
                    op_type=None,
                    label="Contiguous",
                )
            ]
        )


class TestWhatItAccepts__shape_preserving_ops:
    def test_a_cast_that_only_changes_dtype(self) -> None:
        assert (
            _shape_preserving_op_warnings(
                [
                    _node__shape_preserving_ops(
                        ["B", "S", "64", "128"], "[B, S, 64, 128] float32"
                    )
                ]
            )
            == []
        )

    def test_a_symbolic_shape_that_matches(self) -> None:
        assert (
            _shape_preserving_op_warnings(
                [
                    _node__shape_preserving_ops(
                        ["B", "S", "index_n_heads"], "[B, S, index_n_heads] float32"
                    )
                ]
            )
            == []
        )

    def test_an_op_that_really_does_change_the_shape(self) -> None:
        """A transpose moving an axis is the op doing its job, not a defect."""
        assert (
            _shape_preserving_op_warnings(
                [
                    _node__shape_preserving_ops(
                        ["B", "S", "H", "D"],
                        "[B, H, S, D] float32",
                        raw_op="transpose",
                        op_type=None,
                        label="Transpose",
                    )
                ]
            )
            == []
        )

    def test_a_reduction(self) -> None:
        assert (
            _shape_preserving_op_warnings(
                [
                    _node__shape_preserving_ops(
                        ["B", "S", "H"],
                        "[B, S] float32",
                        raw_op="sum",
                        op_type=None,
                        label="Sum",
                    )
                ]
            )
            == []
        )


class TestWhereItSaysNothing__shape_preserving_ops:
    def test_a_node_recording_no_operation(self) -> None:
        """Even labelled ``Cast``: the label is not evidence about the op."""
        node = _node__shape_preserving_ops(
            ["B", "S", "64"], "[B, S, 48] float32", raw_op=None
        )
        assert _shape_preserving_op_warnings([node]) == []

    def test_a_constant_cast(self) -> None:
        """A learned weight's conversion loses its operand at render time."""
        node = _node__shape_preserving_ops(
            ["B", "S", "64"], "[B, S, 48] float32", constant="true"
        )
        assert _shape_preserving_op_warnings([node]) == []

    def test_more_than_one_recorded_operand(self) -> None:
        """Ambiguous which entry is the tensor being cast."""
        node = _node__shape_preserving_ops(["B", "S", "64"], "[B, S, 48] float32")
        node["attrs"] = [
            (
                a
                if a["key"] != "input_shapes"
                else {
                    "key": "input_shapes",
                    "value": json.dumps([["B", "S", "64"], ["1"]]),
                }
            )
            for a in node["attrs"]
        ]
        assert _shape_preserving_op_warnings([node]) == []

    def test_nothing_recorded(self) -> None:
        assert (
            _shape_preserving_op_warnings([_node__shape_preserving_ops(None, None)])
            == []
        )


# --- test_unresolved_config_dims.py ---

from TraceLens.Visualizer.model_explorer_export.type_check import (
    unresolved_config_dims,
)


def _node__unresolved_config_dims(shape: str, node_id: str = "m/@op") -> dict:
    return {"id": node_id, "attrs": [{"key": "output_shape", "value": shape}]}


class TestAConfigConstantMustResolve:
    def test_a_bare_head_count_is_reported(self) -> None:
        messages = unresolved_config_dims(
            [_node__unresolved_config_dims("[B, N, S, D] bfloat16")]
        )
        assert messages, "a bare 'N' is an unresolved head count"

    def test_the_message_says_what_it_is_and_where_to_look(self) -> None:
        """So the reader knows where to fix it, not just that it is broken.

        It no longer quotes a list of key spellings: there is no such list to
        quote, because the names a model reads its constants under come from the
        model's own config class.
        """
        (message,) = [
            m
            for m in unresolved_config_dims(
                [_node__unresolved_config_dims("[B, N, S] f32")]
            )
            if "'N'" in m
        ]
        assert "heads" in message, "which dimension failed"
        assert "attribute_map" in message, "where a rename would be declared"
        assert "m/@op" in message, "and which node"

    def test_every_offending_dim_is_found_not_every_other_one(self) -> None:
        """``[B, N, S, D]`` shares the comma between neighbours.

        A pattern that CONSUMES the delimiters matches every other dim, so half
        the offenders go unreported -- which is how a first draft of this check
        passed a shape carrying two unresolved symbols.
        """
        symbols = {
            m.split("bare symbol ")[1].split(" ")[0].strip("'")
            for m in unresolved_config_dims(
                [_node__unresolved_config_dims("[B, N, S, D] bfloat16")]
            )
        }
        assert symbols == {"N", "D"}, symbols


class TestRuntimeAxesAreNeverFlagged:
    def test_batch_sequence_images_and_patches_are_legitimate(self) -> None:
        assert (
            unresolved_config_dims(
                [_node__unresolved_config_dims("[B, S, Img, Pv] bfloat16")]
            )
            == []
        )

    def test_a_resolved_shape_is_silent(self) -> None:
        assert (
            unresolved_config_dims(
                [_node__unresolved_config_dims("[B, 32, S, 128] bfloat16")]
            )
            == []
        )

    def test_a_node_without_a_shape_is_skipped(self) -> None:
        assert unresolved_config_dims([{"id": "m/@x", "attrs": []}]) == []


class TestThePartitionComesFromTheShapeMachinery:
    def test_runtime_axes_are_batch_sequence_images_and_patches(self) -> None:
        runtime = {"B", "S", "Pv", "Img"}
        for value in runtime:
            assert (
                unresolved_config_dims([_node__unresolved_config_dims(f"[{value}, 4]")])
                == []
            ), value
        assert unresolved_config_dims([_node__unresolved_config_dims("[H, 4]")])


# --- test_host_size_readers.py ---

import pytest

_MODEL__host_size_readers = "zai-org/GLM-5.3-Flash"


def _attr__host_size_readers(node: dict, key: str) -> str:
    for item in node.get("attrs") or []:
        if item.get("key") == key:
            return str(item.get("value"))
    return ""


@pytest.fixture(scope="module")
def glm_nodes__host_size_readers(model_graph_nodes):
    return model_graph_nodes(_MODEL__host_size_readers)


def _vision_splits(nodes: list[dict]) -> list[dict]:
    return [
        node
        for node in nodes
        if (node.get("label") or "") == "Split"
        and "blocks:attn" in str(node["id"])
        and "cu_seqlens" in _attr__host_size_readers(node, "details")
    ]


class TestEverySizedOpShowsItsSize:
    def test_all_three_projections_are_split(
        self, glm_nodes__host_size_readers
    ) -> None:
        """q, k and v -- the fan-out is over a literal tuple of all three."""
        assert len(_vision_splits(glm_nodes__host_size_readers)) == 3, [
            n["id"] for n in _vision_splits(glm_nodes__host_size_readers)
        ]

    def test_each_one_reads_the_lengths_it_is_split_by(
        self, glm_nodes__host_size_readers
    ) -> None:
        by_id = {str(node["id"]): node for node in glm_nodes__host_size_readers}
        for split in _vision_splits(glm_nodes__host_size_readers):
            sources = [
                str(edge.get("sourceNodeId"))
                for edge in split.get("incomingEdges") or []
            ]
            sized_by = [
                src
                for src in sources
                if "cu_seqlens" in str((by_id.get(src) or {}).get("label") or "")
                or "cu_seqlens" in src
            ]
            assert sized_by, (split["id"], sources)

    def test_the_size_is_not_drawn_as_a_tensor_operand(
        self, glm_nodes__host_size_readers
    ) -> None:
        """It is a size, so it types as a Scalar -- a split takes one tensor."""
        for split in _vision_splits(glm_nodes__host_size_readers):
            types = _attr__host_size_readers(split, "input_types")
            assert types.count("bfloat16") == 1, (split["id"], types)
            assert "Scalar" in types, (split["id"], types)


# --- test_method_expansion_entry_spec.py ---

import re

import pytest

_MODEL__method_expansion_entry_spec = "MiniMaxAI/MiniMax-M3"


def _attr__method_expansion_entry_spec(node: dict, key: str) -> str:
    for item in node.get("attrs") or []:
        if item.get("key") == key:
            return str(item.get("value"))
    return ""


@pytest.fixture(scope="module")
def mask_builder(model_graph_nodes):
    """Nodes of one ``build_block_mask`` expansion."""
    nodes = model_graph_nodes(_MODEL__method_expansion_entry_spec)
    ids = [str(n["id"]) for n in nodes if "indexer@l468" in str(n["id"])]
    if not ids:
        pytest.skip("no build_block_mask expansion in this graph")
    prefix = sorted({re.match(r"(.*indexer@l468[^/]*)", i).group(1) for i in ids})[0]
    return [n for n in nodes if str(n["id"]).startswith(prefix)]


def _labelled__method_expansion_entry_spec(nodes: list[dict], label: str) -> list[dict]:
    return [n for n in nodes if (n.get("label") or "") == label]


class TestTheExpansionReadsWhatItWasHanded:
    def test_the_boundary_is_the_block_index_tensor(self, mask_builder) -> None:
        tiles = _labelled__method_expansion_entry_spec(mask_builder, "block_indices")
        assert tiles, "no block_indices boundary"
        for tile in tiles:
            assert (
                _attr__method_expansion_entry_spec(tile, "output_shape")
                == "[B, index_n_heads, S, 4] int64"
            ), (
                tile["id"],
                _attr__method_expansion_entry_spec(tile, "output_shape"),
            )

    def test_the_boundary_and_its_mirror_agree(self, mask_builder) -> None:
        """Two halves of one crossing; they carried different shapes."""
        shapes = {
            _attr__method_expansion_entry_spec(tile, "output_shape")
            for tile in _labelled__method_expansion_entry_spec(
                mask_builder, "block_indices"
            )
        }
        assert len(shapes) == 1, shapes

    def test_nothing_in_the_body_reads_the_decoder_activation(
        self, mask_builder
    ) -> None:
        """``[B, S, 6144]`` is the hidden state, not a block-index tensor."""
        offenders = [
            (str(n["id"]), _attr__method_expansion_entry_spec(n, "output_shape"))
            for n in mask_builder
            if "6144" in _attr__method_expansion_entry_spec(n, "output_shape")
        ]
        assert not offenders, offenders

    def test_the_mask_it_builds_is_four_dimensional(self, mask_builder) -> None:
        """ "We build the full 4D attention mask", per the method's docstring."""
        outputs = [
            n
            for n in mask_builder
            if _attr__method_expansion_entry_spec(n, "synthetic") == "@output"
        ]
        assert outputs
        for node in outputs:
            shape = _attr__method_expansion_entry_spec(node, "output_shape")
            assert shape.count(",") == 3, (node["id"], shape)


# --- test_vision_cu_seqlens_shape.py ---

import pytest

_MODEL__vision_cu_seqlens_shape = "zai-org/GLM-5.3-Flash"


def _attr__vision_cu_seqlens_shape(node: dict, key: str) -> str:
    for item in node.get("attrs") or []:
        if item.get("key") == key:
            return str(item.get("value"))
    return ""


def _shape(node: dict) -> str:
    return _attr__vision_cu_seqlens_shape(node, "output_shape")


@pytest.fixture(scope="module")
def glm_nodes__vision_cu_seqlens_shape(model_graph_nodes):
    return model_graph_nodes(_MODEL__vision_cu_seqlens_shape)


def _labelled__vision_cu_seqlens_shape(nodes: list[dict], label: str) -> list[dict]:
    return [node for node in nodes if (node.get("label") or "") == label]


class TestTheGridIsItsOwnInput:
    def test_the_grid_has_a_model_input_boundary(
        self, glm_nodes__vision_cu_seqlens_shape
    ) -> None:
        boundary = [
            node
            for node in glm_nodes__vision_cu_seqlens_shape
            if str(node["id"]) == "@vision_input:grid_thw"
        ]
        assert boundary, "the grid the tower is handed has no boundary of its own"

    def test_the_grid_does_not_read_the_image_patches(
        self, glm_nodes__vision_cu_seqlens_shape
    ) -> None:
        """The defect itself: the grid docking onto the tower's activation."""
        for node in _labelled__vision_cu_seqlens_shape(
            glm_nodes__vision_cu_seqlens_shape, "grid_thw"
        ):
            sources = {
                str(edge.get("sourceNodeId"))
                for edge in node.get("incomingEdges") or []
            }
            assert (
                "visual/@input" not in sources
            ), f"{node['id']} reads the image patches as its grid"

    def test_the_grid_is_rows_of_coordinates(
        self, glm_nodes__vision_cu_seqlens_shape
    ) -> None:
        """``(num_images_or_videos, 3)`` per the source docstring."""
        for node in _labelled__vision_cu_seqlens_shape(
            glm_nodes__vision_cu_seqlens_shape, "grid_thw"
        ):
            assert _shape(node) == "[Img, 3] int64", (node["id"], _shape(node))


class TestCuSeqlens:
    def test_it_is_one_offset_per_segment_boundary(
        self, glm_nodes__vision_cu_seqlens_shape
    ) -> None:
        """``(num_segments + 1,) int32``, per the helper's own docstring.

        The dtype is the ``cumsum(dim=0, dtype=dtype)`` the source asks for,
        where ``dtype`` resolves through ``torch.jit.is_tracing()`` -- False in
        the forward this export documents.
        """
        tiles = _labelled__vision_cu_seqlens_shape(
            glm_nodes__vision_cu_seqlens_shape, "cu_seqlens"
        )
        assert tiles, "no cu_seqlens tile in the graph"
        for node in tiles:
            assert _shape(node) == "[Img + 1] int32", (node["id"], _shape(node))

    def test_no_cast_reports_a_device_as_its_dtype(
        self, glm_nodes__vision_cu_seqlens_shape
    ) -> None:
        """``x.to(inputs_embeds.device)`` moves a tensor; it casts nothing.

        Recorded as a target dtype it made a pure device move look like a dtype
        change, which then survived cast elision as a Cast that changes nothing.
        """
        offenders = [
            (str(node["id"]), _attr__vision_cu_seqlens_shape(node, "details"))
            for node in glm_nodes__vision_cu_seqlens_shape
            if "dtype: " in _attr__vision_cu_seqlens_shape(node, "details")
            and ".device"
            in _attr__vision_cu_seqlens_shape(node, "details").split("dtype: ", 1)[1]
        ]
        assert not offenders, offenders

    def test_no_node_in_the_chain_counts_patches(
        self, glm_nodes__vision_cu_seqlens_shape
    ) -> None:
        """Pv is the patch axis; the grid chain must never report it."""
        chain = [
            node
            for node in glm_nodes__vision_cu_seqlens_shape
            if "get_vision_cu_seqlens" in str(node["id"])
            and not _attr__vision_cu_seqlens_shape(node, "constant")
        ]
        assert chain, "the cu_seqlens frame is missing"
        offenders = [
            (str(node["id"]), _shape(node)) for node in chain if "Pv" in _shape(node)
        ]
        assert not offenders, offenders
