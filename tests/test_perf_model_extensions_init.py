###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""No-trace constructor tests for perf-model extensions touched by #870."""

from TraceLens.PerfModel.extensions.perf_model_extensions import (
    aiter_fused_qk_rope_cat_and_cache_mla,
    per_group_quant,
    sgl_kernel_rotary_embedding,
    vllm_triton_per_token_group_quant_fp8,
)


def test_vllm_group_quant_constructs_without_trace():
    event = {
        "args": {
            "Input Dims": ((4, 256), ()),
            "Input type": ("c10::BFloat16", "Scalar"),
            "Concrete Inputs": ("", "128"),
        }
    }
    model = vllm_triton_per_token_group_quant_fp8(event)
    assert model.param_details["M"] == 4
    assert model.param_details["N"] == 256
    assert model.param_details["group_size"] == 128
    assert model.flops() == 6 * 4 * 256
    assert model.bytes() == 4 * 256 * 2 + 4 * 256 * 1 + 4 * 2 * 4


def test_per_group_quant_constructs_without_trace():
    event = {
        "args": {
            "Input Dims": [(4, 256), (4, 256), (4, 2)],
            "Input type": ["c10::Float8_e4m3fn", "c10::BFloat16", "c10::BFloat16"],
            "Input Strides": [(256, 1), (256, 1), (2, 1)],
        }
    }
    model = per_group_quant(event)
    assert model.param_details["shape_in1"] == (4, 256)
    assert model.param_details["shape_in2"] == (4, 2)
    assert model.param_details["shape_out"] == (4, 256)
    assert model.flops() > 0
    assert model.bytes() is not None


def test_aiter_fused_qk_rope_constructs_without_trace():
    event = {
        "args": {
            "Input Dims": [
                (2, 8, 512),
                (2, 8, 64),
                (2, 1, 512),
                (2, 1, 64),
                (128, 1, 1, 64),
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::Float8_e4m3fn",
            ],
        }
    }
    model = aiter_fused_qk_rope_cat_and_cache_mla(event)
    assert model.param_details["T"] == 2
    assert model.param_details["QH"] == 8
    assert model.param_details["KH"] == 1
    assert model.flops() > 0
    assert model.bytes() is not None


def test_sgl_kernel_rotary_embedding_constructs_without_trace():
    event = {
        "args": {
            "Input Dims": [
                (2,),
                (2, 1024),
                (2, 128),
                (),
                (2048, 128),
                (),
            ],
            "Input type": [
                "Scalar",
                "c10::BFloat16",
                "c10::BFloat16",
                "Scalar",
                "c10::BFloat16",
                "Scalar",
            ],
            "Concrete Inputs": ["", "", "", "128", "", ""],
        }
    }
    model = sgl_kernel_rotary_embedding(event)
    assert model.param_details["num_elements"] == 2 * (8 + 1) * 128
    assert model.flops() > 0
    assert model.bytes() is not None
