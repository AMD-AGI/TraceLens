###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from TraceLens.PerfModel.perf_model import evoformer_attention, SDPA
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Shapes from a real OpenFold3 trace event:
# Q/K/V: [Batch=1, N_seq=129, N_res=256, Head=4, Dim=16]
# res_mask: [1, 129, 1, 1, 256]
# pair_bias: [1, 1, 4, 256, 256]
_Q_SHAPE = [1, 129, 256, 4, 16]
_RES_MASK_SHAPE = [1, 129, 1, 1, 256]
_PAIR_BIAS_SHAPE = [1, 1, 4, 256, 256]


def _event(q_shape=None, dtype="c10::Half"):
    q = q_shape or _Q_SHAPE
    return {
        "name": "EvoformerAttention",
        "args": {
            "Input Dims": [q, q, q, _RES_MASK_SHAPE, _PAIR_BIAS_SHAPE],
            "Input type": [dtype, dtype, dtype, dtype, dtype],
            "Concrete Inputs": ["", "", "", "", ""],
        },
    }


# ---------------------------------------------------------------------------
# Mapping and categorization
# ---------------------------------------------------------------------------


def test_evoformer_attention_is_mapped():
    assert op_to_perf_model_class_map["EvoformerAttention"] is evoformer_attention


def test_evoformer_attention_categorizes_as_sdpa_fwd():
    row = {"name": "EvoformerAttention", "kernel_details": []}
    assert categorize_torch_op(row) == "SDPA_fwd"


def test_evoformer_attention_inherits_sdpa():
    assert issubclass(evoformer_attention, SDPA)


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------


def test_evoformer_attention_param_details():
    evt = _event()
    m = evoformer_attention(evt)
    # N_seq=129 is folded into batch: B_eff = 1 * 129 = 129
    assert m.B == 129
    assert m.N_Q == 256
    assert m.H_Q == 4
    assert m.N_KV == 256
    assert m.H_KV == 4
    assert m.d_h_qk == 16
    assert m.d_h_v == 16
    assert m.param_details["causal"] is False


def test_evoformer_attention_batch_scaling():
    """Larger batch / N_seq folds correctly into B_eff."""
    evt = _event(q_shape=[2, 64, 128, 8, 32])
    m = evoformer_attention(evt)
    assert m.B == 2 * 64  # B * N_seq
    assert m.N_Q == 128
    assert m.H_Q == 8
    assert m.d_h_qk == 32


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def test_evoformer_attention_flops():
    # B_eff=129, H_Q=4, N_Q=N_KV=256, d_h=16, non-causal
    # flops_qk = 129 * 4 * (2 * 256 * 256 * 16) = 1,082,130,432
    # flops_pv = same → total = 2 * 1,082,130,432 = 2,164,260,864
    expected = 129 * 4 * 2 * 256 * 256 * 16 + 129 * 4 * 2 * 256 * 256 * 16  # QK^T  # AV
    evt = _event()
    m = evoformer_attention(evt)
    assert m.flops() == expected


def test_evoformer_attention_flops_scales_with_n_seq():
    """Doubling N_seq should double FLOPs."""
    evt1 = _event(q_shape=[1, 64, 128, 4, 16])
    evt2 = _event(q_shape=[1, 128, 128, 4, 16])
    assert evoformer_attention(evt2).flops() == 2 * evoformer_attention(evt1).flops()


# ---------------------------------------------------------------------------
# Bytes
# ---------------------------------------------------------------------------


def test_evoformer_attention_bytes():
    # Q+K+V+O = 4 tensors, each B_eff * N_res * Head * Dim elements
    # 4 * 129 * 256 * 4 * 16 * 2 bytes = 16,908,288
    expected = 4 * 129 * 256 * 4 * 16 * 2
    evt = _event()
    m = evoformer_attention(evt)
    assert m.bytes(bytes_per_element=2) == expected


def test_evoformer_attention_bytes_bpe4():
    """Float32 (bpe=4) gives twice the bytes of float16 (bpe=2)."""
    evt = _event()
    m = evoformer_attention(evt)
    assert m.bytes(bytes_per_element=4) == 2 * m.bytes(bytes_per_element=2)
