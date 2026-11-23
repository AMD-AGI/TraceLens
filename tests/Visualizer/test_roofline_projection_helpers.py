###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Branch tests for the helpers ``project_roofline`` is built from.

``project_roofline`` walks a merged graph and hands each compute node to a
chain of small helpers. These tests call the helpers directly with hand-built
nodes, so each branch (a skipped reference body, an eager attention chain, a
quantized GEMM, an Origami timing that cannot be made) is reached without
loading a checkpoint.
"""

from __future__ import annotations

import sys
import types
import warnings

import pytest

from TraceLens.Visualizer.model_explorer_export import roofline_projection as rp
from TraceLens.Visualizer.model_explorer_export.computation_roles import (
    ATTENTION_ROLE,
    ROLE_BASIS_COMPUTATION,
)

_ARCH_FULL = {
    "name": "MI300X",
    "mem_bw_gbps": 5300,
    "num_cus": 304,
    "lds_capacity": 65536,
    "freq_mhz": 2100,
    "gemm_units_per_cu": 4,
}
_BF16 = "bfloat16"


@pytest.fixture(autouse=True)
def _clean_module_state(monkeypatch):
    """Leave the runtime singleton and the Origami globals as they were."""
    rp._RT.reset()
    monkeypatch.setattr(rp, "_origami_enabled", False)
    monkeypatch.setattr(rp, "_origami_arch", None)
    monkeypatch.setattr(rp, "_origami_arch_warned", False)
    monkeypatch.setattr(rp, "_sim_note", None)
    yield
    rp._RT.reset()


def _graph_op(
    node_id: str = "decoder/layers/mlp/act",
    computation: str = "matmul",
    role: str = "",
    attrs: list | None = None,
    class_name: str = "Linear",
    input_texts: list | None = None,
    output_text: str = f"[B, S, 8] {_BF16}",
    namespace: str = "decoder",
) -> rp._GraphOp:
    node = {"id": node_id, "namespace": namespace, "attrs": attrs or []}
    return rp._GraphOp(
        node,
        node_id,
        namespace,
        1,
        "decoder",
        computation,
        class_name,
        "",
        output_text,
        input_texts or [],
        role,
        "",
    )


def _run(nodes=None, **overrides) -> rp._ProjectionRun:
    fields = dict(
        nodes=nodes or [],
        dims={"B": 1, "S": 16},
        gpu_arch=None,
        moe_top_k=None,
        moe_intermediate_size=None,
        opaque_attention=set(),
        replaced_eager_attention=set(),
    )
    fields.update(overrides)
    return rp._ProjectionRun(**fields)


def _modeled_op(category: str) -> rp.RooflineOp:
    op = rp._unmodeled_op("decoder/op", "decoder", "matmul", "", 1)
    op.modeled = True
    op.category = category
    return op


# --- graph nodes and skipped reference bodies ------------------------------


def test_a_node_that_is_not_a_dict_is_not_a_graph_op():
    assert rp._graph_op("not a node") is None
    assert rp._graph_op(None) is None


def test_expanded_conv_body_is_the_causal_conv_call_site():
    assert rp._is_expanded_conv_body("decoder/linear_attn/causal_conv1d_fn:@op_3")
    assert not rp._is_expanded_conv_body("decoder/mlp/down_proj:0")


@pytest.mark.parametrize(
    ("namespace", "node_id", "computation", "expected"),
    [
        ("decoder", "decoder/mlp/act_fn:silu:0", "SiLU", True),
        ("decoder", "decoder/mlp/mul:0", "multiply", True),
        ("decoder", "decoder/self_attn/silu:0", "silu", False),
        ("decoder", "decoder/mlp/down_proj:0", "Linear", False),
    ],
)
def test_mlp_activation_interior_needs_the_mlp_segment_and_an_activation(
    namespace, node_id, computation, expected
):
    assert rp._is_mlp_activation_interior(namespace, node_id, computation) is expected


def test_skipped_bodies_are_the_conv_the_padding_mask_and_the_fp8_swiglu_interior():
    conv = _graph_op("decoder/linear_attn/causal_conv1d_fn:@op_1")
    mask = _graph_op("decoder/@fn_l5_apply_mask_to_padding_states:@op_1")
    act = _graph_op("decoder/mlp/act_fn:silu:0", computation="silu")

    assert not rp._is_skipped_body(conv)
    rp._RT.sconv_kernel = 4
    assert rp._is_skipped_body(conv)
    assert rp._is_skipped_body(mask)

    assert not rp._is_skipped_body(act)
    rp._RT.weight_bpe = 1
    assert rp._is_skipped_body(act)


def test_a_skipped_body_adds_no_rows():
    rp._RT.sconv_kernel = 4
    conv = _graph_op("decoder/linear_attn/causal_conv1d_fn:@op_1")
    assert rp._graph_op_rows(_run(), conv) == []


# --- _graph_op_rows routing -------------------------------------------------


@pytest.fixture
def _stub_project_op(monkeypatch):
    """Replace ``_project_op`` and record how ``_graph_op_rows`` called it."""
    calls: list[tuple[tuple, dict]] = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return rp._unmodeled_op("decoder/op", "decoder", "matmul", "", 1)

    monkeypatch.setattr(rp, "_project_op", fake)
    return calls


def test_a_step_with_a_computation_role_is_costed_as_the_op_it_is(
    monkeypatch, _stub_project_op
):
    # The role was read from the op's own computation, so it travels as
    # ``computation_role`` and the block ``role`` stays empty.
    step = _graph_op(
        "decoder/self_attn/seq:1:@attention:@op_1",
        role=ATTENTION_ROLE,
        attrs=[{"key": "role_basis", "value": ROLE_BASIS_COMPUTATION}],
    )
    monkeypatch.setattr(rp, "_eager_attention_rows", lambda run, graph_op: ["EAGER"])

    rows = rp._graph_op_rows(_run(), step)

    assert rows == ["EAGER"]
    args, kwargs = _stub_project_op[0]
    assert args[12] == ""
    assert kwargs["computation_role"] == ATTENTION_ROLE


def test_a_draft_container_is_not_a_second_copy_of_its_children(
    monkeypatch, _stub_project_op
):
    monkeypatch.setattr(rp, "_node_is_draft_container", lambda node: True)
    assert rp._graph_op_rows(_run(), _graph_op()) == []


# --- eager attention chains -------------------------------------------------


_QK_INPUTS = [["B", "8", "S", "64"], ["B", "8", "S", "64"]]
_STEP_ID = "decoder/layers/0/self_attn/seq:15:@attention:@op_4"


def test_attention_scope_is_the_module_before_the_marker():
    assert rp._attention_scope(_STEP_ID) == "decoder/layers/0/self_attn/seq:15"
    assert rp._attention_scope("decoder/self_attn/q_proj") == (
        "decoder/self_attn/q_proj"
    )


def test_an_eager_chain_is_priced_once_as_one_sdpa_row():
    run = _run()
    step = _graph_op(
        _STEP_ID,
        input_texts=_QK_INPUTS,
        output_text=f"[B, 8, S, S] {_BF16}",
        computation="matmul",
    )

    rows = rp._eager_attention_rows(run, step)

    assert len(rows) == 1
    row = rows[0]
    assert row.modeled and row.category == "Attention"
    assert row.name == "decoder/layers/0/self_attn/seq:15"
    # The next step of the same chain is already paid for.
    assert rp._eager_attention_rows(run, step) == []


def test_an_eager_chain_under_an_opaque_block_is_not_priced_again():
    scope = rp._attention_scope(_STEP_ID)
    run = _run(opaque_attention={scope})
    assert rp._eager_attention_rows(run, _graph_op(_STEP_ID)) == []


def test_an_eager_chain_with_no_batch_dim_has_no_row():
    run = _run(dims={"S": 16})
    step = _graph_op(_STEP_ID, input_texts=_QK_INPUTS)
    assert rp._eager_attention_rows(run, step) == []


def test_eager_qk_heads_reads_query_heads_and_head_dim():
    dims = {"B": 1, "S": 16}
    assert rp._eager_qk_heads("", _QK_INPUTS, dims) == (8, 64)


def test_eager_qk_heads_skips_square_and_unresolved_shapes():
    dims = {"B": 1, "S": 16}
    square = [["B", "8", "S", "S"]]
    assert rp._eager_qk_heads("", square, dims) is None
    assert rp._eager_qk_heads("", [["B", "UNKNOWN", "S", "64"]], dims) is None
    assert rp._eager_qk_heads("", [], dims) is None


def test_softmax_row_without_batch_or_sequence_dims_is_none():
    assert rp._softmax_attention_row("scope", "s", 1, {}, 2.0, _BF16, None) is None


def test_softmax_row_takes_heads_from_the_introspected_config():
    rp._RT.attn_heads = 8
    rp._RT.attn_head_dim = 64
    row = rp._softmax_attention_row(
        "scope", "s", 2, {"B": 1, "S": 16}, 2.0, _BF16, None
    )
    assert row is not None and row.modeled
    assert row.instances == 2


def test_softmax_row_without_any_head_count_falls_back_to_the_boundary(monkeypatch):
    dims = {"B": 1, "S": 16}
    monkeypatch.setattr(rp, "_attention_boundary_cost", lambda dims, bpe: None)
    assert rp._softmax_attention_row("scope", "s", 1, dims, 2.0, _BF16, None) is None

    monkeypatch.setattr(
        rp,
        "_attention_boundary_cost",
        lambda dims, bpe: ("Attention", "matrix", 100.0, 200.0),
    )
    row = rp._softmax_attention_row("scope", "s", 1, dims, 2.0, _BF16, None)
    assert row is not None and row.flops == 100.0


# --- opaque attention, MoE and weight-quant side rows ----------------------


@pytest.mark.parametrize("has_projection_leaves", [True, False])
def test_an_opaque_attention_leaf_adds_its_projections_only_when_not_drawn(
    monkeypatch, has_projection_leaves
):
    monkeypatch.setattr(rp, "_attention_variants_cover", lambda instances: False)
    monkeypatch.setattr(
        rp,
        "_attention_has_projection_leaves",
        lambda nodes, node: has_projection_leaves,
    )
    monkeypatch.setattr(rp, "_attention_extra_ops", lambda *a, **k: ["EXTRA"])
    op = _modeled_op("Attention")

    rows = rp._attention_block_rows(
        _run(), _graph_op(role="attention"), op, "attention", 2.0, _BF16
    )

    assert rows == ([op] if has_projection_leaves else [op, "EXTRA"])


def test_an_opaque_moe_block_adds_its_router_and_shared_expert_rows(monkeypatch):
    monkeypatch.setattr(rp, "_moe_leaves_present", lambda nodes, node: frozenset())
    monkeypatch.setattr(rp, "_moe_extra_ops", lambda *a, **k: ["ROUTER"])
    moe = _graph_op(computation="Qwen3MoeSparseMoeBlock", class_name="MoE")

    rows = rp._moe_block_rows(_run(), moe, _modeled_op("MoE"), "moe", 2.0, _BF16)

    assert rows == ["ROUTER"]


def test_weight_quant_rows_route_a_linear_module_and_a_formula_gemm(monkeypatch):
    monkeypatch.setattr(rp, "_quant_side_ops", lambda *a, **k: ["MODULE"])
    monkeypatch.setattr(rp, "_formula_quant_rows", lambda *a, **k: ["FORMULA"])
    rp._RT.weight_bpe = 1
    gemm = _modeled_op("GEMM")

    linear = _graph_op(computation="Linear")
    assert rp._weight_quant_rows(_run(), linear, gemm, "ffn", 2.0, _BF16) == ["MODULE"]
    matmul = _graph_op(computation="matmul")
    assert rp._weight_quant_rows(_run(), matmul, gemm, "ffn", 2.0, _BF16) == ["FORMULA"]


def test_a_formula_gemm_with_a_dense_ffn_gets_its_activation_quantized():
    rp._RT.weight_bpe = 1
    rp._RT.dense_intermediate = 64
    dims = {"B": 1, "S": 4, rp.Symbol.HIDDEN.value: 16}

    rows = rp._formula_quant_rows(
        "decoder/mlp", "decoder", 1, "GEMM", "ffn", dims, 2.0, _BF16, None
    )

    assert [row.computation for row in rows] == ["act_quant"]


def test_narrow_weights_are_dequantized_before_the_multiply():
    # fp4 weights are expanded to the activation width. fp8 are read directly.
    rp._RT.weight_bpe = 0.5
    rows = rp._quant_matrix_rows(
        "decoder/mlp", "decoder", 1, [(8, 32, 16)], 2.0, _BF16, None
    )
    names = [row.computation for row in rows]
    assert "weight_dequant" in names
    dequant = rows[names.index("weight_dequant")]
    assert dequant.bytes == 32 * 16 * 0.5 + 32 * 16 * 2.0


# --- per-node cost contexts -------------------------------------------------


def _context(dims_only, dims=None, bpe=2.0) -> rp._OpContext:
    base = rp._unmodeled_op("decoder/op", "decoder", "x", "", 1)
    return rp._OpContext(base, dims or {"B": 1, "S": 4}, dims_only, _BF16, bpe, None)


def test_a_drawn_output_with_an_unresolved_dim_has_no_element_count():
    ctx = _context(["UNKNOWN", "8"])
    op = rp._finish_drawn(ctx, ("Attention", "matrix", 10.0, 20.0))
    assert op.output_elements is None
    assert op.flops == 10.0


def test_an_attention_block_with_no_introspected_heads_falls_through():
    assert rp._attention_role_op(_context(["B", "S", "8"]), "sdpa", []) is None


def test_an_ffn_role_with_neither_moe_nor_dense_cost_falls_through():
    assert rp._ffn_role_op(_context(["B", "S", "8"]), "ffn", "", "mlp") is None


def test_a_computation_role_makes_the_matmul_an_attention_op():
    ctx = _context(["B", "S", "16"])
    inputs = [["B", "S", "8"], ["8", "16"]]

    plain = rp._dispatch_leaf_op(ctx, "MatMul", inputs, "", "")
    assert plain.category == "GEMM"

    score = rp._dispatch_leaf_op(
        _context(["B", "S", "16"]), "MatMul", inputs, "", ATTENTION_ROLE
    )
    assert score.category == "Attention"
    assert score.flops == plain.flops


# --- gated delta ------------------------------------------------------------


def test_gated_delta_cost_counts_bytes_when_the_width_is_known():
    inputs = [["B", "S", "16", "128"], ["B", "S", "48", "128"]]
    _cat, _kind, flops, bytes_moved = rp._gated_delta_cost(
        inputs, {"B": 1, "S": 2}, 2.0
    )
    assert flops > 0
    assert bytes_moved > 0


@pytest.mark.parametrize(
    "inputs",
    [
        [["UNKNOWN", "S", "4", "64"]],
        [["B", "S", "8"]],
        [["B", "S", "4", "0"]],
    ],
    ids=["unresolved-dim", "no-rank4-input", "empty-head-dim"],
)
def test_gated_delta_cost_is_none_when_the_shapes_cannot_price_it(inputs):
    assert rp._gated_delta_cost(inputs, {"B": 1, "S": 2}, 2.0) is None


# --- Origami ----------------------------------------------------------------


def _hardware(parallel_mi_cu=4):
    return types.SimpleNamespace(
        N_CU=304,
        lds_capacity=65536,
        compute_clock_ghz=2.1,
        parallel_mi_cu=parallel_mi_cu,
    )


def test_origami_arch_is_none_without_a_gpu_arch():
    assert rp._origami_arch_for(None) is None


def test_origami_arch_with_every_field_needs_no_device_and_is_cached():
    first = rp._origami_arch_for(dict(_ARCH_FULL))
    assert first == _ARCH_FULL
    assert rp._origami_arch_for({"name": "other"}) is first


def test_origami_arch_fills_missing_fields_from_the_installed_device(monkeypatch):
    fake = types.SimpleNamespace(get_hardware_for_device=lambda index: _hardware())
    monkeypatch.setitem(sys.modules, "origami", fake)

    arch = rp._origami_arch_for({"name": "MI300X", "mem_bw_gbps": 5300})

    assert arch["num_cus"] == 304
    assert arch["lds_capacity"] == 65536
    assert arch["freq_mhz"] == 2100
    assert arch["gemm_units_per_cu"] == 4


def test_origami_arch_keeps_what_the_arch_already_names(monkeypatch):
    fake = types.SimpleNamespace(
        get_hardware_for_device=lambda index: _hardware(parallel_mi_cu=None)
    )
    monkeypatch.setitem(sys.modules, "origami", fake)

    arch = rp._origami_arch_for({"name": "MI300X", "num_cus": 128})

    assert arch["num_cus"] == 128
    assert "gemm_units_per_cu" not in arch


def test_origami_arch_warns_once_when_no_device_can_fill_the_gaps(monkeypatch):
    monkeypatch.setitem(sys.modules, "origami", None)  # import raises

    with pytest.warns(RuntimeWarning, match="Origami GEMM and SDPA times need"):
        arch = rp._origami_arch_for({"name": "MI300X"})
    assert arch == {"name": "MI300X"}

    monkeypatch.setattr(rp, "_origami_arch", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rp._origami_arch_for({"name": "MI300X"})


def test_origami_time_is_none_without_a_usable_arch(monkeypatch):
    monkeypatch.setattr(rp, "_origami_enabled", True)
    sim = [("gemm", (8, 8, 8, 1))]
    assert rp._origami_time_us("GEMM", "fp16", _BF16, 1.0, None, sim) is None


def test_origami_time_skips_a_note_it_cannot_time(monkeypatch):
    monkeypatch.setattr(rp, "_origami_enabled", True)
    # A gemm note on an Attention op, and an sdpa note missing its shape.
    sim = [("gemm", (8, 8, 8, 1)), ("sdpa", {})]
    assert rp._origami_time_us("Attention", "fp16", _BF16, 1.0, _ARCH_FULL, sim) is None
