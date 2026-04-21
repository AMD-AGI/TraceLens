###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for Mamba SSD perf model (#552) — MambaSplitConv1dScanCombinedFn.

FLOPS formula derived from the 4-step SSD algorithm (Mamba-2, Tri Dao et al.):
  conv1d:         2 × B × conv_channels × T × d_conv
  CB^T (step 1):  2 × B × G × T × C × N
  M@X  (step 1):  2 × B × H × T × C × P
  B^T@X (step 2): 2 × B × H × T × N × P
  C@h   (step 4): 2 × B × H × T × N × P
"""

from TraceLens.PerfModel.perf_model import MambaSSD, mamba_ssd_fwd
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)


def _mamba_event(
    batch=4,
    seqlen=2048,
    nheads=32,
    headdim=64,
    ngroups=8,
    d_state=16,
    d_conv=4,
    chunk_size=128,
):
    """Build a synthetic MambaSplitConv1dScanCombinedFn trace event."""
    d_inner = nheads * headdim
    conv_channels = d_inner + 2 * ngroups * d_state
    combined_dim = 2 * d_inner + 2 * ngroups * d_state + nheads

    return {
        "name": "MambaSplitConv1dScanCombinedFn",
        "args": {
            "Input Dims": [
                [batch, seqlen, combined_dim],
                [conv_channels, d_conv],
                [conv_channels],
                [nheads],
                [nheads],
                [nheads],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "float",
                "float",
                "c10::BFloat16",
                "Scalar",
                "",
                "",
                "",
                "Scalar",
                "",
                "Scalar",
                "",
                "",
                "Scalar",
                "Scalar",
                "Scalar",
            ],
            "Concrete Inputs": [
                "",
                "",
                "",
                "",
                "",
                "",
                str(chunk_size),
                "",
                "",
                "",
                "False",
                "",
                "9.9999999999999995e-07",
                "",
                "",
                str(headdim),
                str(ngroups),
                "False",
            ],
        },
    }


# ---------------------------------------------------------------------------
# Mapping and categorization
# ---------------------------------------------------------------------------


def test_mamba_ssd_mapped():
    assert op_to_perf_model_class_map["MambaSplitConv1dScanCombinedFn"] is mamba_ssd_fwd


def test_mamba_ssd_categorizes_as_ssm_fwd():
    row = {"name": "MambaSplitConv1dScanCombinedFn", "kernel_details": []}
    assert categorize_torch_op(row) == "SSM_fwd"


def test_mamba_bwd_categorizes_as_ssm_bwd():
    row = {"name": "MambaSplitConv1dScanCombinedFnBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "SSM_bwd"


def test_mamba_ssd_inherits():
    assert issubclass(mamba_ssd_fwd, MambaSSD)


# ---------------------------------------------------------------------------
# Parameter derivation
# ---------------------------------------------------------------------------


def test_param_derivation_mamba_370m():
    """Mamba 370M: d_state=16, ngroups=8, headdim=64, nheads=32."""
    event = _mamba_event(
        batch=4, seqlen=2048, nheads=32, headdim=64, ngroups=8, d_state=16
    )
    model = mamba_ssd_fwd(event)
    p = model.param_details
    assert p["B"] == 4
    assert p["T"] == 2048
    assert p["H"] == 32
    assert p["P"] == 64
    assert p["G"] == 8
    assert p["N"] == 16
    assert p["d_inner"] == 2048
    assert p["conv_channels"] == 2048 + 2 * 8 * 16  # 2304


def test_param_derivation_zebra_llama():
    """Zebra Llama 1B: d_state=64, ngroups=8, headdim=64, nheads=32."""
    event = _mamba_event(
        batch=8, seqlen=8192, nheads=32, headdim=64, ngroups=8, d_state=64
    )
    model = mamba_ssd_fwd(event)
    p = model.param_details
    assert p["B"] == 8
    assert p["T"] == 8192
    assert p["N"] == 64
    assert p["d_inner"] == 2048
    assert p["conv_channels"] == 2048 + 2 * 8 * 64  # 3072


# ---------------------------------------------------------------------------
# FLOPS
# ---------------------------------------------------------------------------


def test_flops_mamba_370m():
    """Verify FLOPS formula matches manual computation for Mamba 370M config."""
    B, T, H, P, G, N, C, d_conv = 4, 2048, 32, 64, 8, 16, 128, 4
    conv_channels = H * P + 2 * G * N  # 2304

    expected = (
        2 * B * conv_channels * T * d_conv  # conv1d
        + 2 * B * G * T * C * N  # CB^T
        + 2 * B * H * T * C * P  # M@X
        + 2 * B * H * T * N * P  # B^T@X
        + 2 * B * H * T * N * P  # C@h
    )

    model = mamba_ssd_fwd(
        _mamba_event(
            batch=B,
            seqlen=T,
            nheads=H,
            headdim=P,
            ngroups=G,
            d_state=N,
            d_conv=d_conv,
            chunk_size=C,
        )
    )
    assert model.flops() == expected


def test_flops_zebra_llama():
    """Verify FLOPS for Zebra Llama 1B config (larger d_state=64)."""
    B, T, H, P, G, N, C, d_conv = 8, 8192, 32, 64, 8, 64, 128, 4
    conv_channels = H * P + 2 * G * N  # 3072

    expected = (
        2 * B * conv_channels * T * d_conv
        + 2 * B * G * T * C * N
        + 2 * B * H * T * C * P
        + 2 * B * H * T * N * P
        + 2 * B * H * T * N * P
    )

    model = mamba_ssd_fwd(
        _mamba_event(
            batch=B,
            seqlen=T,
            nheads=H,
            headdim=P,
            ngroups=G,
            d_state=N,
            d_conv=d_conv,
            chunk_size=C,
        )
    )
    assert model.flops() == expected


def test_flops_larger_d_state_increases():
    """Larger d_state → more FLOPS (chunk_state + state→output dominate)."""
    small_n = mamba_ssd_fwd(_mamba_event(d_state=16))
    large_n = mamba_ssd_fwd(_mamba_event(d_state=64))
    assert large_n.flops() > small_n.flops()


def test_flops_conv1d_component():
    """Verify the conv1d portion is included in total FLOPS."""
    model = mamba_ssd_fwd(
        _mamba_event(batch=1, seqlen=1024, nheads=1, headdim=1, ngroups=1, d_state=1)
    )
    assert model.flops() > 0


# ---------------------------------------------------------------------------
# Bytes
# ---------------------------------------------------------------------------


def test_bytes_mamba_370m():
    B, T, H, P, G, N, d_conv = 4, 2048, 32, 64, 8, 16, 4
    d_inner = H * P  # 2048
    conv_channels = d_inner + 2 * G * N  # 2304
    combined_dim = 2 * d_inner + 2 * G * N + H  # 4384
    bpe = 2  # bf16

    bpe_dt_bias = 4  # float32 (slot 3)
    bpe_A = 4  # float32 (slot 4)
    bpe_D = 2  # bf16 (slot 5)

    expected = (
        B * T * combined_dim * bpe  # read zxbcdt
        + conv_channels * d_conv * bpe  # read conv weight
        + conv_channels * bpe  # read conv bias
        + H * bpe_dt_bias
        + H * bpe_A
        + H * bpe_D  # per-slot param bytes
        + B * T * d_inner * bpe  # write output
    )

    model = mamba_ssd_fwd(
        _mamba_event(batch=B, seqlen=T, nheads=H, headdim=P, ngroups=G, d_state=N)
    )
    assert model.bytes() == expected


def test_bytes_increases_with_seqlen():
    short = mamba_ssd_fwd(_mamba_event(seqlen=1024))
    long = mamba_ssd_fwd(_mamba_event(seqlen=8192))
    assert long.bytes() > short.bytes()


# ---------------------------------------------------------------------------
# FLOPS breakdown dominance
# ---------------------------------------------------------------------------


def test_ssd_matmuls_dominate_conv():
    """For large d_state, the SSD matmul terms should dominate conv1d."""
    model = mamba_ssd_fwd(
        _mamba_event(batch=8, seqlen=8192, d_state=64, nheads=32, headdim=64)
    )
    p = model.param_details
    B, T, H, P, G, N, C = p["B"], p["T"], p["H"], p["P"], p["G"], p["N"], p["C"]
    conv_channels, d_conv = p["conv_channels"], p["d_conv"]

    flops_conv = 2 * B * conv_channels * T * d_conv
    flops_total = model.flops()
    assert flops_conv < flops_total * 0.05  # conv < 5% of total


# ---------------------------------------------------------------------------
# Backward methods
# ---------------------------------------------------------------------------


def test_flops_bwd_equals_fwd():
    model = mamba_ssd_fwd(_mamba_event())
    assert model.flops_bwd() == model.flops()


def test_bytes_bwd_equals_fwd():
    model = mamba_ssd_fwd(_mamba_event())
    assert model.bytes_bwd() == model.bytes()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_combined_dim_raises():
    """Tampered combined_dim that doesn't satisfy divisibility should raise."""
    import pytest

    event = _mamba_event()
    event["args"]["Input Dims"][0][2] += 1  # break divisibility
    with pytest.raises(ValueError, match="Cannot derive d_state"):
        mamba_ssd_fwd(event)
