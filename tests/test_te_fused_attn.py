###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from TraceLens.PerfModel.perf_model import te_fused_attn
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)

# ---------------------------------------------------------------------------
# Mapping and categorization
# ---------------------------------------------------------------------------


def test_fused_attn_func_mapped():
    assert op_to_perf_model_class_map["FusedAttnFunc"] is te_fused_attn


def test_fused_attn_func_categorizes_as_sdpa_fwd():
    row = {"name": "FusedAttnFunc", "kernel_details": []}
    assert categorize_torch_op(row) == "SDPA_fwd"


def test_fused_attn_func_backward_categorizes_as_sdpa_bwd():
    row = {"name": "FusedAttnFuncBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "SDPA_bwd"


# ---------------------------------------------------------------------------
# Helpers — synthetic events matching real TE FusedAttn trace format
# ---------------------------------------------------------------------------

# TE sbhd format: [seq_len, batch, heads, head_dim]
_Q_SHAPE = [8192, 8, 32, 64]
_K_SHAPE = [8192, 8, 32, 64]
_V_SHAPE = [8192, 8, 32, 64]


def _fused_attn_event(q=None, k=None, v=None, dropout_p="0.1", is_causal="True"):
    q = q or _Q_SHAPE
    k = k or _K_SHAPE
    v = v or _V_SHAPE
    # TE FusedAttnFunc: scalars at [0..8], then Q/K/V at [9..11]
    dims = [[]] * 9 + [q, k, v] + [[]] * 9
    types = (
        ["Scalar", "Scalar", "Scalar", "int", "int", "", "", "", ""]
        + ["c10::BFloat16"] * 3
        + [""] * 9
    )
    concrete = (
        ["True", str(q[0]), str(k[0]), "", "", "", "", "", ""]
        + ["", "", ""]
        + ["", "0.125", dropout_p, is_causal]
        + ["", "", "False", "False", "False"]
    )
    return {
        "name": "FusedAttnFunc",
        "args": {
            "Input Dims": dims,
            "Input type": types,
            "Concrete Inputs": concrete,
        },
    }


# ---------------------------------------------------------------------------
# te_fused_attn — param extraction and FLOPS
# ---------------------------------------------------------------------------


def test_fused_attn_param_details():
    """Verify B, N_Q, H_Q, N_KV, H_KV, d_h from sbhd Q/K/V shapes."""
    event = _fused_attn_event()
    model = te_fused_attn(event)
    assert model.B == 8  # batch
    assert model.N_Q == 8192  # seq_len_q
    assert model.H_Q == 32  # heads_q
    assert model.N_KV == 8192  # seq_len_kv
    assert model.H_KV == 32  # heads_kv
    assert model.d_h_qk == 64  # head_dim


def test_fused_attn_flops_causal():
    """SDPA FLOPS: 2 * B * H * S * S_kv * d_h (scaled by 0.5 for causal)."""
    event = _fused_attn_event(is_causal="True")
    model = te_fused_attn(event)
    B, H, S, d_h = 8, 32, 8192, 64
    flops_qk = 2 * B * H * S * S * d_h
    flops_pv = 2 * B * H * S * S * d_h
    expected = int((flops_qk + flops_pv) * 0.5)  # causal scaling
    assert model.flops() == expected


def test_fused_attn_flops_non_causal():
    event = _fused_attn_event(is_causal="False")
    model = te_fused_attn(event)
    B, H, S, d_h = 8, 32, 8192, 64
    expected = 2 * (2 * B * H * S * S * d_h)  # no causal scaling, full QK + PV
    assert model.flops() == expected


def test_fused_attn_gqa():
    """GQA: H_Q=32, H_KV=8 (4 groups)."""
    q = [4096, 4, 32, 128]
    k = [4096, 4, 8, 128]
    v = [4096, 4, 8, 128]
    event = _fused_attn_event(q=q, k=k, v=v, is_causal="True")
    model = te_fused_attn(event)
    assert model.H_Q == 32
    assert model.H_KV == 8
    assert model.B == 4
    assert model.N_Q == 4096
    assert model.d_h_qk == 128
