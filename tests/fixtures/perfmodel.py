###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared test helpers migrated from test_perfmodel_coverage.py."""

from __future__ import annotations
import sys
from math import prod
from unittest.mock import MagicMock, patch
import pytest
from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.extensions.attention_perf_model_extensions import (
    InferenceAttention,
    aiter_fmha_v3_varlen_fwd,
    aiter_mha_batch_prefill,
    aiter_paged_attention_ragged,
    mha_varlen_fwd,
    mla_decode_fwd,
    mla_tilelang_sparse_fwd,
    pa_decode_gluon,
    pa_sparse_prefill_opus_fwd,
    pseudo_mla_prefill_fwd,
    pseudo_v4_paged_decode_csa,
    pseudo_v4_paged_decode_hca,
    pseudo_v4_paged_decode_swa,
    vllm_unified_mla_attention_with_output,
)
from TraceLens.PerfModel.extensions.moe_perf_model_extensions import (
    moe_aiter_ck_stage1,
    moe_aiter_ck_stage2,
    moe_aiter_fused_blockscale,
    moe_aiter_unfused_down,
    moe_aiter_unfused_up,
    moe_flydsl_stage1,
    moe_flydsl_stage2,
    moe_gptq_awq_down,
    moe_gptq_awq_up,
    moe_triton_invoke_grouped_gemm,
    moe_triton_unfused_down,
    moe_triton_unfused_up,
    sglang_fused_append_shared_experts,
)
from TraceLens.PerfModel.extensions.perf_model_extensions import (
    aiter_dynamic_per_group_scaled_quant_fp4,
    aiter_fused_dynamic_mxfp4_quant_moe_sort_hip,
    aiter_rope_cached_positions_2c_fwd_impl,
    batched_gemm_a16wfp4,
    batched_gemm_a8w8,
    fused_flatten_mxfp4_quant,
    gemm_a16w16,
    gemm_a16w16_asm,
    gemm_afp4wfp4,
    mhc_fused_post_pre_gemm_sqrsum,
    mhc_post,
    mhc_pre_big_fuse_rmsnorm,
    mhc_pre_gemm_sqrsum,
    mixed_sample_outer_exponential,
    sglang_quant_dynamic_mxfp4_quant,
    sglang_store_cache,
    topk_softplus,
    vllm_rocm_unquantized_gemm,
)
from TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions import (
    aiter_add_rmsnorm,
    aiter_rmsnorm,
    aiter_rmsnorm2d_fwd_with_dynamicquant_ck,
    aiter_rmsnorm_quant,
    vllm_rocm_aiter_rmsnorm_fp8_group_quant,
    vllm_rocm_aiter_rmsnorm_with_add_fp8_group_quant,
    vllm_rocm_aiter_triton_add_rmsnorm_pad,
)
from TraceLens.PerfModel.kernel_name_parser import parse_rocm_gemm
from TraceLens.PerfModel.triton_compiled_perf_model import (
    TritonCompiledPerfModel,
    _lookup,
    _meta_from_trace_args,
    _parse_kernel_name,
    _parse_wrapper,
)
from TraceLens.PerfModel.utils import (
    add_simulation_time_columns,
    name2bpe,
    parse_bool,
    simulation_dtype_map,
    torch_dtype_map,
)

ROCM_GEMM = (
    "Custom_Cijk_Alik_Bljk_BBS_BH_Bias_HAS_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB0"
)
_GDN_ANNOTATION = (
    "execute_64_context_0(sq0sk0sqsq0sqsk0)"
    "_generation_64(sq64sk131072sqsq64sqsk131072)"
)
_VLLM_ATTN_ANNOTATION = "(128_256_512_1024_2048_3072_4096_64)"
_ARCH = {
    "name": "mi300x",
    "freq_mhz": 2200,
    "num_cus": 304,
    "gemm_units_per_cu": 4,
    "mem_bw_gbps": 5300,
    "l1_bw_gbps": 100,
}
def _gemm_event(name, a_shape, b_shape, dtypes=None, strides=None, kernel_names=None):
    dtypes = dtypes or ["c10::BFloat16", "c10::BFloat16"]
    event = {
        "name": name,
        "args": {
            "Input Dims": [list(a_shape), list(b_shape)],
            "Input type": list(dtypes),
        },
    }
    if strides:
        event["args"]["Input Strides"] = strides
    if kernel_names:
        event["kernel_names"] = kernel_names
    return event
def _sdpa_event(cls, q, k, v, concrete, strides=None, bhnd=(0, 2, 1, 3)):
    event = {
        "name": cls.__name__,
        "args": {
            "Input Dims": [q, k, v],
            "Input type": ["c10::BFloat16"] * 3,
            "Concrete Inputs": concrete,
        },
    }
    if strides:
        event["args"]["Input Strides"] = strides
    return event
def _norm_event(op_shape, channels, training=True, affine=True, has_bias=True):
    weight = (channels,) if affine else None
    bias = (channels,) if has_bias and affine else None
    return {
        "args": {
            "Input Dims": [op_shape, (channels,), weight, bias],
            "Input type": ["c10::BFloat16"] * 4,
            "Input Strides": [(channels, 1), (1,), (1,), (1,)],
            "Concrete Inputs": [
                "",
                str(list(op_shape[1:])),
                "",
                "",
                "",
                str(training),
                "0.1",
                "1e-5",
                "True",
            ],
        }
    }
def _moe_unfused_event(gated=True, kernel_name="moe_fp8_gemm_kernel"):
    return {
        "args": {
            "Input Dims": [[32, 4096], [32, 8]],
            "Input type": ["c10::BFloat16", "c10::Float32"],
            "MoE topk": 2,
            "MoE GEMM gated": gated,
        },
        "kernel_details": [{"name": kernel_name}],
    }


_GDN = _GDN_ANNOTATION


def _attn_base():
    return {
        "annotation": _GDN,
        "args": {
            "Input Dims": [[64, 8, 64], [64, 8, 64], [64, 8, 128]],
            "Input type": ["c10::BFloat16"] * 3,
        },
    }


def _conv_fwd_event(input_shape, filter_shape):
    nd = len(input_shape) - 2
    if nd == 1:
        stride, padding, dilation, out_pad = "(1)", "(0)", "(1)", "(0)"
    elif nd == 3:
        stride, padding, dilation, out_pad = "(1,1,1)", "(0,0,0)", "(1,1,1)", "(0,0,0)"
    else:
        stride, padding, dilation, out_pad = "(1,1)", "(0,0)", "(1,1)", "(0,0)"
    return {
        "args": {
            "Input Dims": [list(input_shape), list(filter_shape)],
            "Input type": ["c10::BFloat16", "c10::BFloat16"],
            "Input Strides": [[1] * len(input_shape), [1] * len(filter_shape)],
            "Concrete Inputs": [
                "",
                "",
                "",
                stride,
                padding,
                dilation,
                "False",
                out_pad,
                "1",
            ],
        }
    }


def _conv_bwd_event(input_shape, filter_shape):
    nd = len(input_shape) - 2
    if nd == 1:
        stride, padding, dilation, out_pad = "[1]", "[0]", "[1]", "[0]"
    elif nd == 3:
        stride, padding, dilation, out_pad = (
            "[1, 1, 1]",
            "[0, 0, 0]",
            "[1, 1, 1]",
            "[0, 0, 0]",
        )
    else:
        stride, padding, dilation, out_pad = "[1, 1]", "[0, 0]", "[1, 1]", "[0, 0]"
    grad_out = list(input_shape)
    return {
        "args": {
            "Input Dims": [grad_out, list(input_shape), list(filter_shape)],
            "Input type": ["c10::BFloat16"] * 3,
            "Input Strides": [[1] * len(input_shape)] * 3,
            "Concrete Inputs": [
                "",
                "",
                "",
                "[0]",
                stride,
                padding,
                dilation,
                "False",
                out_pad,
                "1",
                "[True, True, False]",
            ],
        }
    }


class _BadPerfModel:
    def __init__(self, *args, **kwargs):
        pass

    def flops(self):
        raise NotImplementedError("no model")

    def bytes(self):
        return 0


class _ExplodingPerfModel:
    def __init__(self, *args, **kwargs):
        pass

    def flops(self):
        raise RuntimeError("boom")

    def bytes(self):
        return 0
