###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for aiter varlen Flash-Attention + aten flash-attention perf models.

Closes TraceLens #650, #290, #590. Sample event payloads come from a real
Wan 2.2 T2V A14B training trace (Primus, BF16, mbs=1).
"""

from TraceLens.PerfModel.perf_model import (
    aiter__fmha_v3_bwd,
    aiter__fmha_v3_varlen_fwd,
    aiter__fmha_v3_varlen_forward,
    aiter__fmha_v3_varlen_bwd,
    aiter__fmha_v3_varlen_backward,
    aten___flash_attention_forward,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)


def test_new_attention_ops_are_mapped():
    assert (
        op_to_perf_model_class_map["aiter::fmha_v3_varlen_fwd"]
        is aiter__fmha_v3_varlen_fwd
    )
    assert (
        op_to_perf_model_class_map["aiter::fmha_v3_varlen_bwd"]
        is aiter__fmha_v3_varlen_bwd
    )
    assert (
        op_to_perf_model_class_map["aiter::wrapper_fmha_v3_varlen_fwd"]
        is aiter__fmha_v3_varlen_forward
    )
    assert (
        op_to_perf_model_class_map["aiter::wrapper_fmha_v3_varlen_bwd"]
        is aiter__fmha_v3_varlen_backward
    )
    assert op_to_perf_model_class_map["aiter::fmha_v3_bwd"] is aiter__fmha_v3_bwd
    assert (
        op_to_perf_model_class_map["aten::_flash_attention_forward"]
        is aten___flash_attention_forward
    )


def test_varlen_fwd_categorizes_as_sdpa_fwd():
    for op in (
        "aiter::fmha_v3_varlen_fwd",
        "aiter::wrapper_fmha_v3_varlen_fwd",
        "aten::_flash_attention_forward",
    ):
        row = {"name": op, "kernel_details": []}
        assert categorize_torch_op(row) == "SDPA_fwd", op


def test_varlen_bwd_categorizes_as_sdpa_bwd():
    for op in (
        "aiter::fmha_v3_varlen_bwd",
        "aiter::wrapper_fmha_v3_varlen_bwd",
        "aiter::fmha_v3_bwd",
    ):
        row = {"name": op, "kernel_details": []}
        assert categorize_torch_op(row) == "SDPA_bwd", op


def test_extension_no_longer_overrides_core_for_varlen_fwd():
    """``aiter::fmha_v3_varlen_fwd`` must resolve to the core SDPA-derived class
    on training traces (no annotation), not the InferenceAttention extension."""
    cls = op_to_perf_model_class_map["aiter::fmha_v3_varlen_fwd"]
    assert cls.__module__.endswith("PerfModel.perf_model"), cls.__module__
    assert cls is aiter__fmha_v3_varlen_fwd


# Real Wan 2.2 event payloads (Primus BF16 training, mbs=1)

_WAN22_VARLEN_FWD = {
    "name": "aiter::fmha_v3_varlen_fwd",
    "args": {
        "Input Dims": ([[32760, 40, 128]] * 3 + [[2], [2]] + [[]] * 20),
        "Input type": ["c10::BFloat16"] * 3 + ["int"] * 2 + ["Scalar"] * 13 + [""] * 7,
        "Input Strides": [[5120, 128, 1]] * 3 + [[1], [1]] + [[]] * 20,
        "Concrete Inputs": [
            "",
            "",
            "",
            "",
            "",
            "32760",
            "32760",
            "0",
            "0.",
            "0.088388347648318447",
            "0.",
            "False",
            "False",
            "-1",
            "-1",
            "True",
            "False",
            "1",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
    },
}

_WAN22_VARLEN_BWD = {
    "name": "aiter::fmha_v3_varlen_bwd",
    "args": {
        "Input Dims": [
            [32760, 40, 128],
            [32760, 40, 128],
            [512, 40, 128],
            [512, 40, 128],
            [32760, 40, 128],
            [40, 32760],
            [2],
            [2],
            *([[]] * 11),
            [32760, 40, 128],
            [512, 40, 128],
            [512, 40, 128],
            [],
            [2],
            [],
            [],
            [],
        ],
        "Input type": ["c10::BFloat16"] * 5
        + ["float", "int", "int"]
        + ["Scalar"] * 11
        + ["c10::BFloat16"] * 3
        + ["", "long int", "", "", ""],
        "Input Strides": [[5120, 128, 1]] * 5
        + [[32760, 1], [1], [1]]
        + [[]] * 11
        + [[5120, 128, 1]] * 3
        + [[], [1], [], [], []],
        "Concrete Inputs": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "32760",
            "512",
            "0.",
            "0.088388347648318447",
            "False",
            "False",
            "-1",
            "-1",
            "False",
            "True",
            "1",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
    },
}

_WAN22_ATEN_FLASH_FWD = {
    "name": "aten::_flash_attention_forward",
    "args": {
        "Input Dims": [
            [1, 6240, 1, 384],
            [1, 6240, 1, 384],
            [1, 6240, 1, 384],
            [],
            [],
            *([[]] * 10),
        ],
        "Input type": ["c10::BFloat16"] * 3 + [""] * 2 + ["Scalar"] * 6 + [""] * 4,
        "Input Strides": [[7188480, 1152, 7188480, 1]] * 3 + [[]] * 12,
        "Concrete Inputs": [
            "",
            "",
            "",
            "",
            "",
            "6240",
            "6240",
            "0.",
            "False",
            "False",
            "0.051031036307982884",
            "",
            "",
            "",
            "",
        ],
    },
}


def test_varlen_fwd_param_extraction():
    m = aiter__fmha_v3_varlen_fwd(_WAN22_VARLEN_FWD)
    p = m.param_details
    assert p["B"] == 1
    assert p["N_Q"] == 32760
    assert p["N_KV"] == 32760
    assert p["H_Q"] == 40
    assert p["H_KV"] == 40
    assert p["d_h_qk"] == 128
    assert p["d_h_v"] == 128
    assert p["dropout"] == 0.0
    assert p["causal"] is False
    assert p["max_seqlen_q"] == 32760.0
    assert p["max_seqlen_kv"] == 32760.0
    assert p["num_seqs_q"] == 1
    assert p["num_seqs_kv"] == 1


def test_varlen_fwd_flops_matches_sdpa_formula():
    m = aiter__fmha_v3_varlen_fwd(_WAN22_VARLEN_FWD)
    # Non-causal self-attention: 4 * B * H * N^2 * d_h (QK^T + PV, each 2*B*H*N^2*d)
    expected = 4 * 1 * 40 * 32760**2 * 128
    assert m.flops() == expected, (m.flops(), expected)


def test_varlen_bwd_param_extraction_cross_attention():
    m = aiter__fmha_v3_varlen_bwd(_WAN22_VARLEN_BWD)
    p = m.param_details
    assert p["B"] == 1
    assert p["N_Q"] == 32760
    assert p["N_KV"] == 512
    assert p["H_Q"] == 40
    assert p["H_KV"] == 40
    assert p["d_h_qk"] == 128
    assert p["d_h_v"] == 128
    assert p["causal"] is False
    assert p["max_seqlen_q"] == 32760.0
    assert p["max_seqlen_kv"] == 512.0


def test_varlen_fwd_multi_seq_packing_accumulates_flops():
    """Multi-sequence packed varlen: when ``cu_seqlens`` length > 2, the FLOPs
    must scale with ``num_seqs_q`` (mirror of ``flash_attention_varlen_forward``).

    Construct a 2-sequence packed varlen event: T=200 with cu_seqlens=[0,100,200],
    so max_seqlen_q=100 and num_seqs_q=2. Single max-seq FLOPs:
        4 * 1 * 4 * 100^2 * 64 = 10_240_000
    Remainder seq (length (200-100)/(2-1) = 100):
        4 * 1 * 4 * 100^2 * 64 = 10_240_000
    Total = 20_480_000  (~ 2x the single-seq estimate at the same T)."""
    event = {
        "name": "aiter::fmha_v3_varlen_fwd",
        "args": {
            "Input Dims": (
                [[200, 4, 64]] * 3  # Q, K, V — packed 2 sequences of 100
                + [[3], [3]]  # cu_seqlens_q, cu_seqlens_kv: 3 entries -> B=2 seqs
                + [[]] * 20
            ),
            "Input type": ["c10::BFloat16"] * 3
            + ["int"] * 2
            + ["Scalar"] * 13
            + [""] * 7,
            "Input Strides": [[256, 64, 1]] * 3 + [[1], [1]] + [[]] * 20,
            "Concrete Inputs": [
                "",
                "",
                "",
                "",
                "",
                "100",
                "100",
                "0",
                "0.",
                "0.125",
                "0.",
                "False",
                "False",
                "-1",
                "-1",
                "True",
                "False",
                "1",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
        },
    }
    m = aiter__fmha_v3_varlen_fwd(event)
    assert m.param_details["num_seqs_q"] == 2
    assert m.param_details["max_seqlen_q"] == 100.0
    expected = 4 * 1 * 4 * 100**2 * 64 + 1 * 4 * 1 * 4 * 100**2 * 64
    assert m.flops() == expected, (m.flops(), expected)


def test_varlen_bwd_multi_seq_packing_accumulates_flops():
    """Same multi-seq packing scenario for the backward perf model.
    bwd / fwd ratio remains 5/2 for the square self-attention case."""
    bwd_event = {
        "name": "aiter::fmha_v3_varlen_bwd",
        "args": {
            "Input Dims": (
                [[200, 4, 64]] * 5  # dout, q, k, v, out
                + [[4, 200], [3], [3]]  # softmax_lse, cu_seqlens_q, cu_seqlens_kv
                + [[]] * 11
                + [[200, 4, 64]] * 3  # dq, dk, dv
                + [[], [3], [], [], []]
            ),
            "Input type": ["c10::BFloat16"] * 5
            + ["float", "int", "int"]
            + ["Scalar"] * 11
            + ["c10::BFloat16"] * 3
            + ["", "long int", "", "", ""],
            "Input Strides": [[256, 64, 1]] * 5
            + [[200, 1], [1], [1]]
            + [[]] * 11
            + [[256, 64, 1]] * 3
            + [[], [1], [], [], []],
            "Concrete Inputs": [
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "100",
                "100",
                "0.",
                "0.125",
                "False",
                "False",
                "-1",
                "-1",
                "False",
                "True",
                "1",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
        },
    }
    fwd_event = {
        "name": "aiter::fmha_v3_varlen_fwd",
        "args": {
            "Input Dims": ([[200, 4, 64]] * 3 + [[3], [3]] + [[]] * 20),
            "Input type": ["c10::BFloat16"] * 3
            + ["int"] * 2
            + ["Scalar"] * 13
            + [""] * 7,
            "Input Strides": [[256, 64, 1]] * 3 + [[1], [1]] + [[]] * 20,
            "Concrete Inputs": [
                "",
                "",
                "",
                "",
                "",
                "100",
                "100",
                "0",
                "0.",
                "0.125",
                "0.",
                "False",
                "False",
                "-1",
                "-1",
                "True",
                "False",
                "1",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
        },
    }
    m_bwd = aiter__fmha_v3_varlen_bwd(bwd_event)
    m_fwd = aiter__fmha_v3_varlen_fwd(fwd_event)
    assert m_bwd.param_details["num_seqs_q"] == 2
    assert m_bwd.flops() / m_fwd.flops() == 2.5


def test_varlen_bwd_fwd_flops_ratio_is_5_over_2_for_square():
    """Standard FA bwd identity: flops_bwd / flops_fwd == 5/2 for N_Q==N_KV."""
    m_fwd = aiter__fmha_v3_varlen_fwd(_WAN22_VARLEN_FWD)
    sym_bwd = {
        "name": "aiter::fmha_v3_varlen_bwd",
        "args": {
            "Input Dims": (
                [[32760, 40, 128]] * 5
                + [[40, 32760], [2], [2]]
                + [[]] * 11
                + [[32760, 40, 128]] * 3
                + [[], [2], [], [], []]
            ),
            "Input type": ["c10::BFloat16"] * 5
            + ["float", "int", "int"]
            + ["Scalar"] * 11
            + ["c10::BFloat16"] * 3
            + ["", "long int", "", "", ""],
            "Input Strides": [[5120, 128, 1]] * 5
            + [[32760, 1], [1], [1]]
            + [[]] * 11
            + [[5120, 128, 1]] * 3
            + [[], [1], [], [], []],
            "Concrete Inputs": [
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "32760",
                "32760",
                "0.",
                "0.088388347648318447",
                "False",
                "False",
                "-1",
                "-1",
                "False",
                "True",
                "1",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
        },
    }
    m_bwd = aiter__fmha_v3_varlen_bwd(sym_bwd)
    assert m_bwd.flops() / m_fwd.flops() == 2.5


def test_aten_flash_attention_forward_param_extraction():
    m = aten___flash_attention_forward(_WAN22_ATEN_FLASH_FWD)
    p = m.param_details
    assert p["B"] == 1
    assert p["N_Q"] == 6240
    assert p["N_KV"] == 6240
    assert p["H_Q"] == 1
    assert p["H_KV"] == 1
    assert p["d_h_qk"] == 384
    assert p["d_h_v"] == 384
    assert p["dropout"] == 0.0
    assert p["causal"] is False


def test_aten_flash_attention_forward_flops_matches_expected():
    """At the Wan 2.2 shape (B=1, S=6240, H=1, d_h=384, non-causal),
    flops = 4 * B * H * N^2 * d_h = 4 * 6240^2 * 384 = 59808153600 (~59.81 GFLOPS)."""
    m = aten___flash_attention_forward(_WAN22_ATEN_FLASH_FWD)
    expected = 4 * 1 * 1 * 6240**2 * 384
    assert m.flops() == expected


def test_wrapper_varlen_fwd_indices_shift_by_one():
    """The wrapper variant adds a leading ``out`` tensor; arg indices shift by +1."""
    wrapper_event = {
        "name": "aiter::wrapper_fmha_v3_varlen_fwd",
        "args": {
            "Input Dims": (
                [[32760, 40, 128]] + [[32760, 40, 128]] * 3 + [[2], [2]] + [[]] * 19
            ),
            "Input type": ["c10::BFloat16"] * 4
            + ["int"] * 2
            + ["Scalar"] * 13
            + [""] * 6,
            "Input Strides": [[5120, 128, 1]] * 4 + [[1], [1]] + [[]] * 19,
            "Concrete Inputs": (
                ["", "", "", "", "", ""]
                + ["32760", "32760", "0"]
                + ["0.", "0.088388347648318447", "0."]
                + ["False", "False", "-1", "-1", "True", "False", "1"]
                + [""] * 5
            ),
        },
    }
    m = aiter__fmha_v3_varlen_forward(wrapper_event)
    p = m.param_details
    assert p["N_Q"] == 32760 and p["N_KV"] == 32760
    assert p["H_Q"] == 40 and p["d_h_qk"] == 128
    assert p["max_seqlen_q"] == 32760.0
    assert p["causal"] is False


def test_aiter_fmha_v3_bwd_uses_mha_bwd_layout():
    """``aiter::fmha_v3_bwd`` shares the ``aiter::mha_bwd`` argument layout."""
    qkv = [2, 512, 16, 64]
    event = {
        "name": "aiter::fmha_v3_bwd",
        "args": {
            "Input Dims": [qkv, qkv, qkv, qkv],
            "Concrete Inputs": [
                "",
                "",
                "",
                "",
                "",
                "",
                "0.0",
                "0.125",
                "True",
                "-1",
                "-1",
                "False",
            ],
        },
    }
    m = aiter__fmha_v3_bwd(event)
    p = m.param_details
    assert p["B"] == 2 and p["N_Q"] == 512 and p["H_Q"] == 16 and p["d_h_qk"] == 64
    assert p["causal"] is True
    assert m.flops() > 0
