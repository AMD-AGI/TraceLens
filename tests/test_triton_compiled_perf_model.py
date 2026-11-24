###############################################################################
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for TritonCompiledPerfModel (V2 trace-intrinsic path).

Start with RMSNorm (reduction kernel, clear expected FLOPs/bytes).
More cases (V1 wrapper-parsing, multi-word ops, fallback) in follow-ups.
"""

import pytest

import TraceLens.PerfModel.triton_compiled_perf_model as triton_model
from TraceLens.PerfModel.torch_op_mapping import resolve_perf_model_class
from TraceLens.PerfModel.triton_compiled_perf_model import (
    TritonCompiledPerfModel,
    TritonTemplateGemmPerfModel,
    _meta_from_trace_args,
    _parse_kernel_name,
)

# ---------------------------------------------------------------------------
# RMSNorm kernel from a TransformerBlock trace (bf16, batch=8, seq=4096, dim=2048)
# triton_red_fused_add_mean_mul_pow_rsqrt_0
# ---------------------------------------------------------------------------
_RMSNORM_EVENT = {
    "name": "triton_red_fused_add_mean_mul_pow_rsqrt_0",
    "args": {
        "Concrete Inputs": ["", "", "", "32768", "2048"],
        "Input Dims": [[8, 4096, 2048], [2048], [8, 4096, 2048], [], []],
        "Input type": [
            "c10::BFloat16",
            "c10::BFloat16",
            "c10::BFloat16",
            "Scalar",
            "Scalar",
        ],
    },
}


def test_parse_kernel_name_rmsnorm():
    ops = _parse_kernel_name("triton_red_fused_add_mean_mul_pow_rsqrt_0")
    assert ops == [
        "aten.add",
        "aten.mean",
        "aten.mul",
        "aten.pow",
        "aten.rsqrt",
    ]


def test_parse_kernel_name_no_match():
    assert _parse_kernel_name("some_other_kernel") == []


def test_meta_from_trace_args_rmsnorm():
    meta = _meta_from_trace_args(_RMSNORM_EVENT)
    assert meta is not None
    assert meta["xnumel"] == 32768
    assert meta["rnumel"] == 2048
    assert meta["aten_ops"] == [
        "aten.add",
        "aten.mean",
        "aten.mul",
        "aten.pow",
        "aten.rsqrt",
    ]


def test_meta_from_trace_args_missing_fields():
    event = {"name": "triton_red_fused_add_0", "args": {}}
    assert _meta_from_trace_args(event) is None


def test_flops_rmsnorm():
    model = TritonCompiledPerfModel(_RMSNORM_EVENT)
    # add=1, mean=1, mul=1, pow=1 (the square), rsqrt=4 => 8 flops/elem
    assert model.flops() == 8 * 32768 * 2048


def test_bytes_rmsnorm():
    model = TritonCompiledPerfModel(_RMSNORM_EVENT)
    # Three bf16 tensors: [8,4096,2048]*2 + [2048]*2 + [8,4096,2048]*2
    # = 134,217,728 + 4,096 + 134,217,728 = 268,439,552
    expected = (8 * 4096 * 2048 * 2) + (2048 * 2) + (8 * 4096 * 2048 * 2)
    assert model.bytes() == expected


# ---------------------------------------------------------------------------
# Inductor mm template with a fused div + SiLU epilogue (bf16), from an SGLang
# MoE-gate trace: A[M,10240] @ B[320,10240]^T -> C[M,320].
# ---------------------------------------------------------------------------
def _mm_silu_event(m, concrete_scalars):
    return {
        "name": "triton_tem_fused_div_mm_silu_t_0",
        "args": {
            "Concrete Inputs": ["", "", ""] + concrete_scalars,
            "Input Dims": [[m, 10240], [320, 10240], [m, 320]]
            + [[]] * len(concrete_scalars),
            "Input type": ["c10::BFloat16"] * 3 + ["Scalar"] * len(concrete_scalars),
        },
    }


def test_parse_kernel_name_template():
    ops = _parse_kernel_name("triton_tem_fused_div_mm_silu_t_0")
    assert ops == ["aten.div", "aten.mm", "aten.silu", "aten.t"]


def test_parse_kernel_name_template_without_registry(monkeypatch):
    monkeypatch.setattr(triton_model, "_ALL_KNOWN_OPS", set())
    event = _mm_silu_event(32, ["20"])
    assert _parse_kernel_name(event["name"]) == [
        "aten.div",
        "aten.mm",
        "aten.silu",
        "aten.t",
    ]
    assert TritonTemplateGemmPerfModel.can_model(event)


def test_resolve_template_gemm():
    assert (
        resolve_perf_model_class("triton_tem_fused_div_mm_silu_t_0")
        is TritonTemplateGemmPerfModel
    )


@pytest.mark.parametrize(
    "m, scalars", [(32, ["20", "1", "1"]), (2048, ["2048", "1280", "1", "1"])]
)
def test_template_gemm_mm_silu(m, scalars):
    event = _mm_silu_event(m, scalars)
    assert TritonTemplateGemmPerfModel.can_model(event)
    model = TritonTemplateGemmPerfModel(event)
    assert (model.M, model.N, model.K) == (m, 320, 10240)
    # mm: 2*M*N*K; epilogue: div=4 + silu=14 flops per output element
    assert model.flops() == 2 * m * 320 * 10240 + 18 * m * 320
    assert model.bytes() == (m * 10240 + 320 * 10240 + m * 320) * 2
    assert model.get_maf_type() == "matrix"
    assert model.get_compute_precision() == "bf16"
    assert model.param_details["fused_epilogue"] == "aten.div, aten.silu, aten.t"


def test_template_gemm_without_mm_not_modeled():
    event = _mm_silu_event(32, ["20"])
    event["name"] = "triton_tem_fused_convolution_0"
    assert not TritonTemplateGemmPerfModel.can_model(event)


def test_template_gemm_single_tensor_not_modeled():
    event = _mm_silu_event(32, ["20"])
    event["args"]["Input Dims"] = [[32, 10240]]
    event["args"]["Input type"] = ["c10::BFloat16"]
    assert not TritonTemplateGemmPerfModel.can_model(event)


def test_template_gemm_backward_not_implemented():
    model = TritonTemplateGemmPerfModel(_mm_silu_event(32, ["20", "1", "1"]))
    with pytest.raises(NotImplementedError):
        model.flops_bwd()
    with pytest.raises(NotImplementedError):
        model.bytes_bwd()
