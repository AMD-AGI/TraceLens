###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""SDPA byte counts when Q has a different dtype from K and V.

An fp8 KV cache with a bfloat16 query moves K and V at one byte and Q and the
output at two. The forward formula, the backward formula, the perf report's
``bytes()`` call, and the SDPA tile model all have to count them that way.
"""

import pytest

from TraceLens.PerfModel import sdpa_tile
from TraceLens.PerfModel.perf_model import (
    SDPA,
    aiter__fmha_v3_varlen_backward,
    flash_attention,
    flash_attention_backward,
)

BF16 = "c10::BFloat16"
FP8 = "c10::Float8_e4m3fn"

# (B=2, N=512, H=16, d_h=64) in bnhd order
_SHAPE = [2, 512, 16, 64]
_ELEMS = 2 * 512 * 16 * 64
_STRIDE = (512 * 16 * 64, 16 * 64, 64, 1)


def _fwd_event(q=BF16, k=BF16, v=BF16):
    return {
        "name": "flash_attn::_flash_attn_forward",
        "args": {
            "Input Dims": [_SHAPE, _SHAPE, _SHAPE],
            "Input type": [q, k, v],
            "Input Strides": [_STRIDE] * 3,
            "Concrete Inputs": ["", "", "", "0.0", "", "True"],
        },
    }


def _bwd_event(dout=BF16, q=BF16, k=BF16, v=BF16):
    return {
        "name": "flash_attn::_flash_attn_backward",
        "args": {
            "Input Dims": [_SHAPE] * 4,
            "Input type": [dout, q, k, v],
            "Input Strides": [_STRIDE] * 4,
            "Concrete Inputs": ["", "", "", "", "", "", "", "", "0.0", "", "True"],
        },
    }


class TestForward:
    def test_report_call_counts_the_kv_cache_at_its_own_width(self):
        # The report calls bytes() with no argument.
        mixed = flash_attention(_fwd_event(q=BF16, k=FP8, v=FP8))
        # Q and Out at 2 bytes, K and V at 1.
        assert mixed.bytes() == 2 * _ELEMS * 2 + 2 * _ELEMS * 1

    def test_uniform_dtype_is_unchanged(self):
        bf16 = flash_attention(_fwd_event())
        assert bf16.bytes() == 4 * _ELEMS * 2
        assert bf16.bytes() == bf16.bytes(bytes_per_element=2)

    def test_uniform_fp8_counts_one_byte_everywhere(self):
        fp8 = flash_attention(_fwd_event(q=FP8, k=FP8, v=FP8))
        assert fp8.bytes() == 4 * _ELEMS

    def test_explicit_width_is_uniform_as_before(self):
        mixed = flash_attention(_fwd_event(q=BF16, k=FP8, v=FP8))
        assert mixed.bytes(bytes_per_element=2) == 4 * _ELEMS * 2

    def test_explicit_kv_width_overrides_the_trace(self):
        mixed = flash_attention(_fwd_event(q=BF16, k=BF16, v=BF16))
        assert mixed.bytes(kv_bytes_per_element=1) == 2 * _ELEMS * 2 + 2 * _ELEMS

    def test_k_and_v_with_different_dtypes_use_the_wider(self):
        model = flash_attention(_fwd_event(q=BF16, k=FP8, v=BF16))
        assert model.bytes() == 4 * _ELEMS * 2


class TestBackward:
    def test_formula_splits_q_side_and_kv_side(self):
        full = SDPA.bytes_bwd_func(2, 512, 16, 512, 16, 64, 64, True, 2)
        mixed = SDPA.bytes_bwd_func(
            2, 512, 16, 512, 16, 64, 64, True, 2, kv_bytes_per_element=1
        )
        # Q, dOut and dQ at the activation width. K, V, dK and dV at the cache width.
        assert full == 7 * _ELEMS * 2
        assert mixed == 3 * _ELEMS * 2 + 4 * _ELEMS * 1

    def test_formula_without_a_kv_width_is_uniform(self):
        assert SDPA.bytes_bwd_func(
            2, 512, 16, 512, 16, 64, 64, True, 4
        ) == SDPA.bytes_bwd_func(
            2, 512, 16, 512, 16, 64, 64, True, 4, kv_bytes_per_element=4
        )

    def test_report_call_reads_q_k_v_after_dout(self):
        # dout is input 0. Q, K and V are inputs 1 to 3.
        model = flash_attention_backward(_bwd_event(q=BF16, k=FP8, v=FP8))
        assert model.bytes() == 3 * _ELEMS * 2 + 4 * _ELEMS * 1
        assert model.bytes() == model.bytes_bwd()

    def test_dout_dtype_does_not_set_the_kv_width(self):
        model = flash_attention_backward(_bwd_event(dout=FP8, q=BF16, k=BF16, v=BF16))
        assert model.bytes() == 7 * _ELEMS * 2

    def test_wrapper_varlen_backward_reads_after_the_leading_out_tensor(self):
        assert aiter__fmha_v3_varlen_backward.qkv_input_idx == (2, 3, 4)


class TestTileModel:
    def test_tile_model_prices_the_same_bytes_as_the_report(self):
        captured = {}

        def fake_time(arch, dtype, dtype_A_B, nbytes, *args, **kwargs):
            captured["bytes"] = nbytes
            return 1.0

        model = flash_attention(_fwd_event(q=BF16, k=FP8, v=FP8))
        arch = {"mem_bw_gbps": 8000, "num_cus": 256}
        original = sdpa_tile.sdpa_fwd_time_us
        sdpa_tile.sdpa_fwd_time_us = fake_time
        try:
            sdpa_tile.sdpa_tile_time_us(model, arch, gemm_time=None)
        finally:
            sdpa_tile.sdpa_fwd_time_us = original
        assert captured["bytes"] == model.bytes()


class TestMissingTrace:
    class _NoTrace(SDPA):
        @staticmethod
        def get_param_details(event):
            dims = dict(B=1, N_Q=8, H_Q=2, N_KV=8, H_KV=2, d_h_qk=4, d_h_v=4)
            return {**dims, "causal": False}

    def test_no_dtype_anywhere_falls_back_to_two_bytes(self):
        model = self._NoTrace({})
        assert model.bytes() == model.bytes(bytes_per_element=2)

    @pytest.mark.parametrize("types", [[], [BF16], [BF16, BF16]])
    def test_a_short_input_type_list_does_not_raise(self, types):
        event = _fwd_event()
        event["args"]["Input type"] = types
        assert flash_attention(event).bytes() > 0
