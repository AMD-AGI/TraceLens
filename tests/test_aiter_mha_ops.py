###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from TraceLens.PerfModel.perf_model import (
    aiter__mha_fwd,
    aiter__fmha_v3_fwd,
    aiter__mha_bwd,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)

# ---------------------------------------------------------------------------
# Mapping and categorization
# ---------------------------------------------------------------------------


def test_aiter_mha_ops_are_mapped():
    assert op_to_perf_model_class_map["aiter::mha_fwd"] is aiter__mha_fwd
    assert op_to_perf_model_class_map["aiter::fmha_v3_fwd"] is aiter__fmha_v3_fwd
    assert op_to_perf_model_class_map["aiter::mha_bwd"] is aiter__mha_bwd


def test_aiter_mha_fwd_categorizes_as_sdpa_fwd():
    row = {"name": "aiter::mha_fwd", "kernel_details": []}
    assert categorize_torch_op(row) == "SDPA_fwd"


def test_aiter_fmha_v3_fwd_categorizes_as_sdpa_fwd():
    row = {"name": "aiter::fmha_v3_fwd", "kernel_details": []}
    assert categorize_torch_op(row) == "SDPA_fwd"


def test_aiter_mha_bwd_categorizes_as_sdpa_bwd():
    row = {"name": "aiter::mha_bwd", "kernel_details": []}
    assert categorize_torch_op(row) == "SDPA_bwd"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# q/k/v shape: (B=2, N=512, H=16, d_h=64) in bnhd order; bhnd_idx=(0,2,1,3) extracts B,H,N,d_h
_Q_SHAPE = [2, 512, 16, 64]
_K_SHAPE = [2, 512, 16, 64]
_V_SHAPE = [2, 512, 16, 64]

# mha_fwd / fmha_v3_fwd: q[0], k[1], v[2]; dropout_p[3]; softmax_scale[4]; is_causal[5]
_FWD_CONCRETE = ["", "", "", "0.0", "0.125", "True", "-1", "-1", "0", "True", "False"]

# mha_bwd: dout[0], q[1], k[2], v[3], out[4], lse[5]; dropout_p[6]; scale[7]; is_causal[8]
_BWD_CONCRETE = ["", "", "", "", "", "", "0.0", "0.125", "True", "-1", "-1", "False"]


def _fwd_event(name, concrete=None):
    return {
        "name": name,
        "args": {
            "Input Dims": [_Q_SHAPE, _K_SHAPE, _V_SHAPE],
            "Concrete Inputs": concrete or _FWD_CONCRETE,
        },
    }


def _bwd_event():
    return {
        "name": "aiter::mha_bwd",
        "args": {
            "Input Dims": [_Q_SHAPE, _Q_SHAPE, _K_SHAPE, _V_SHAPE],
            "Concrete Inputs": _BWD_CONCRETE,
        },
    }


# ---------------------------------------------------------------------------
# aiter_mha_fwd
# ---------------------------------------------------------------------------


def test_aiter_mha_fwd_flops():
    model = aiter__mha_fwd(_fwd_event("aiter::mha_fwd"))
    # causal=True: 2 * B * H_Q * N_Q * N_KV/2 * (d_h_qk + d_h_v) = SDPA.flops_func causal
    assert model.flops() > 0


def test_aiter_mha_fwd_causal_flag():
    model = aiter__mha_fwd(_fwd_event("aiter::mha_fwd"))
    assert model.param_details["causal"] is True


def test_aiter_mha_fwd_non_causal():
    concrete = list(_FWD_CONCRETE)
    concrete[5] = "False"
    model = aiter__mha_fwd(_fwd_event("aiter::mha_fwd", concrete))
    assert model.param_details["causal"] is False
    causal_model = aiter__mha_fwd(_fwd_event("aiter::mha_fwd"))
    # non-causal attends to full KV → higher flops than causal
    assert model.flops() > causal_model.flops()


# ---------------------------------------------------------------------------
# aiter_fmha_v3_fwd
# ---------------------------------------------------------------------------


def test_aiter_fmha_v3_fwd_flops_match_mha_fwd():
    # Same argument layout as mha_fwd — flops must be identical for same shapes
    mha = aiter__mha_fwd(_fwd_event("aiter::mha_fwd"))
    v3 = aiter__fmha_v3_fwd(_fwd_event("aiter::fmha_v3_fwd"))
    assert mha.flops() == v3.flops()
    assert mha.bytes() == v3.bytes()


# ---------------------------------------------------------------------------
# aiter__mha_bwd
# ---------------------------------------------------------------------------


def test_aiter_mha_bwd_flops():
    model = aiter__mha_bwd(_bwd_event())
    assert model.flops() > 0


def test_aiter_mha_bwd_flops_greater_than_fwd():
    fwd = aiter__mha_fwd(_fwd_event("aiter::mha_fwd"))
    bwd = aiter__mha_bwd(_bwd_event())
    # Backward FLOPs > forward FLOPs (flash bwd is ~2.5x fwd for causal)
    assert bwd.flops() > fwd.flops()
