###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the flash_attention_backward perf model (PR #512).

Covers:
- Op mapping: flash_attn::_flash_attn_backward → flash_attention_backward
- Categorization as SDPA_bwd
- Inheritance from SDPA (not flash_attention)
- get_param_details: argument indices (dout at 0, q/k/v at 1/2/3)
- get_param_details: causal extracted from concrete[10], dropout from concrete[8]
- d_h attribute is set to d_h_qk in __init__
- flops() delegates to flops_bwd() and returns a positive value
- flops() returns more than flash_attention forward flops for the same shapes
- bytes() with no argument resolves dtype from param_details
- bytes() with explicit bytes_per_element
- fa flag in get_simulation_time_bwd recognises flash_attention_backward
"""

from TraceLens.PerfModel.perf_model import (
    SDPA,
    flash_attention,
    flash_attention_backward,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# q/k/v shape: (B=2, N=512, H=16, d_h=64) in bnhd order
# bhnd_idx=(0,2,1,3) maps to B=2, H=16, N=512, d=64
_Q_SHAPE = [2, 512, 16, 64]
_K_SHAPE = [2, 512, 16, 64]
_V_SHAPE = [2, 512, 16, 64]
_DOUT_SHAPE = [2, 512, 16, 64]

# Stride for a contiguous [2, 512, 16, 64] tensor
_STRIDE = (512 * 16 * 64, 16 * 64, 64, 1)

# Concrete inputs for flash_attn::_flash_attn_backward:
# Index 8  → dropout_p
# Index 10 → causal
_CONCRETE = ["", "", "", "", "", "", "", "", "0.0", "", "True", "True"]


def _bwd_event(concrete=None, q_shape=None, k_shape=None, v_shape=None, dtype="c10::BFloat16"):
    """Build a minimal flash_attn::_flash_attn_backward profiler event dict."""
    q = q_shape or _Q_SHAPE
    k = k_shape or _K_SHAPE
    v = v_shape or _V_SHAPE
    dout = list(q)  # dout has the same shape as q
    return {
        "name": "flash_attn::_flash_attn_backward",
        "args": {
            "Input Dims": [dout, q, k, v],
            "Input type": [dtype, dtype, dtype, dtype],
            "Input Strides": [_STRIDE, _STRIDE, _STRIDE, _STRIDE],
            "Concrete Inputs": concrete if concrete is not None else _CONCRETE,
        },
    }


def _fwd_event():
    """Build a minimal flash_attn::_flash_attn_forward event dict with the same shapes."""
    return {
        "name": "flash_attn::_flash_attn_forward",
        "args": {
            "Input Dims": [_Q_SHAPE, _K_SHAPE, _V_SHAPE],
            "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
            "Input Strides": [_STRIDE, _STRIDE, _STRIDE],
            # concrete[3]=dropout, concrete[5]=causal
            "Concrete Inputs": ["", "", "", "0.0", "", "True"],
        },
    }


# ---------------------------------------------------------------------------
# Mapping and categorization
# ---------------------------------------------------------------------------


def test_flash_attn_backward_is_mapped():
    assert op_to_perf_model_class_map["flash_attn::_flash_attn_backward"] is flash_attention_backward


def test_flash_attn_backward_categorizes_as_sdpa_bwd():
    row = {"name": "flash_attn::_flash_attn_backward", "kernel_details": []}
    assert categorize_torch_op(row) == "SDPA_bwd"


# ---------------------------------------------------------------------------
# Class hierarchy
# ---------------------------------------------------------------------------


def test_flash_attention_backward_inherits_from_sdpa():
    assert issubclass(flash_attention_backward, SDPA)


def test_flash_attention_backward_does_not_inherit_from_flash_attention():
    assert not issubclass(flash_attention_backward, flash_attention)


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------


def test_get_param_details_shapes():
    """q/k/v must be read from argument indices 1/2/3 (index 0 is dout)."""
    m = flash_attention_backward(_bwd_event())
    assert m.B == 2
    assert m.N_Q == 512
    assert m.H_Q == 16
    assert m.N_KV == 512
    assert m.H_KV == 16
    assert m.d_h_qk == 64
    assert m.d_h_v == 64


def test_get_param_details_causal_true():
    m = flash_attention_backward(_bwd_event())
    assert m.param_details["causal"] is True


def test_get_param_details_causal_false():
    concrete = list(_CONCRETE)
    concrete[10] = "False"
    m = flash_attention_backward(_bwd_event(concrete=concrete))
    assert m.param_details["causal"] is False


def test_get_param_details_dropout():
    concrete = list(_CONCRETE)
    concrete[8] = "0.1"
    m = flash_attention_backward(_bwd_event(concrete=concrete))
    assert m.param_details["dropout"] == 0.1


def test_get_param_details_defaults_when_concrete_absent():
    """When Concrete Inputs is missing entirely, causal defaults to True, dropout to 0.0."""
    event = _bwd_event()
    del event["args"]["Concrete Inputs"]
    m = flash_attention_backward(event)
    assert m.param_details["causal"] is True
    assert m.param_details["dropout"] == 0.0


def test_get_param_details_flash_impl_true():
    m = flash_attention_backward(_bwd_event())
    assert m.param_details["flash_impl"] is True


def test_get_param_details_dtype():
    m = flash_attention_backward(_bwd_event(dtype="c10::Half"))
    assert m.param_details["dtype_A_B"][0] == "c10::Half"


# ---------------------------------------------------------------------------
# d_h attribute
# ---------------------------------------------------------------------------


def test_d_h_set_to_d_h_qk():
    """__init__ must set self.d_h = self.d_h_qk for get_simulation_time_bwd_func."""
    m = flash_attention_backward(_bwd_event())
    assert m.d_h == m.d_h_qk


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def test_flops_positive():
    m = flash_attention_backward(_bwd_event())
    assert m.flops() > 0


def test_flops_matches_flops_bwd():
    """flops() must delegate to flops_bwd()."""
    m = flash_attention_backward(_bwd_event())
    assert m.flops() == m.flops_bwd()


def test_flops_greater_than_forward():
    """Backward FLOPs are greater than forward FLOPs for the same shapes."""
    fwd = flash_attention(_fwd_event())
    bwd = flash_attention_backward(_bwd_event())
    assert bwd.flops() > fwd.flops()


def test_flops_non_causal_greater_than_causal():
    concrete = list(_CONCRETE)
    concrete[10] = "False"
    causal = flash_attention_backward(_bwd_event())
    non_causal = flash_attention_backward(_bwd_event(concrete=concrete))
    assert non_causal.flops() > causal.flops()


# ---------------------------------------------------------------------------
# Bytes
# ---------------------------------------------------------------------------


def test_bytes_no_arg_uses_dtype():
    """bytes() with no argument must resolve bytes-per-element from param_details dtype."""
    m = flash_attention_backward(_bwd_event(dtype="c10::BFloat16"))  # bf16 → 2 bytes
    assert m.bytes() == m.bytes(bytes_per_element=2)


def test_bytes_with_explicit_bpe():
    m = flash_attention_backward(_bwd_event())
    assert m.bytes(bytes_per_element=4) == 2 * m.bytes(bytes_per_element=2)


def test_bytes_matches_bytes_bwd():
    """bytes(bpe) must delegate to bytes_bwd(bpe)."""
    m = flash_attention_backward(_bwd_event())
    assert m.bytes(bytes_per_element=2) == m.bytes_bwd(bytes_per_element=2)


# ---------------------------------------------------------------------------
# fa flag in get_simulation_time_bwd
# ---------------------------------------------------------------------------


def test_fa_flag_is_true_for_flash_attention_backward():
    """get_simulation_time_bwd must treat flash_attention_backward as a flash-attention op."""
    m = flash_attention_backward(_bwd_event())
    fa = type(m).__name__ in ("flash_attention", "flash_attention_backward")
    assert fa is True
