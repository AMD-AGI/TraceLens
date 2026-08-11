###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens/PerfModel/extensions."""

import pytest

from TraceLens.PerfModel.extensions import (
    FusedMoE,
    InferenceAttention,
    aiter_fused_allreduce_rmsnorm,
    aiter_rms_norm,
    custom_ar_all_reduce,
    gdn_attention_core,
    get_pseudo_op_category_only_mappings,
    get_pseudo_op_mappings,
    moe_aiter_fused_1stage,
)
from TraceLens.PerfModel.extensions.custom_collectives_perf_model_extensions import (
    CustomCollective,
)
from TraceLens.PerfModel.extensions.moe_perf_model_extensions import (
    BiasedGroupedTopk,
    MoeSortScatterGather,
)
from TraceLens.PerfModel.extensions.perf_model_extensions import (
    aiter_gelu_and_mul,
    aiter_gelu_tanh_and_mul,
    aiter_silu_and_mul,
    gemm_a8w8_blockscale,
)
from TraceLens.PerfModel.extensions.pseudo_ops_perf_utils import (
    get_pseudo_op_category_only_mappings as _category_only_from_utils,
)
from TraceLens.PerfModel.extensions.pseudo_ops_perf_utils import (
    get_pseudo_op_mappings as _mappings_from_utils,
)
from TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions import RMSNorm

_GDN_ANNOTATION = (
    "execute_64_context_0(sq0sk0sqsq0sqsk0)"
    "_generation_64(sq64sk131072sqsq64sqsk131072)"
)

MOE_FUSED_EVENT = {
    "args": {
        "Input Dims": [
            [32, 4096],
            [8, 14336, 4096],
            [8, 4096, 7168],
            [32, 2],
        ],
        "Input type": [
            "c10::BFloat16",
            "c10::Float8_e4m3fn",
            "c10::Float8_e4m3fn",
            "c10::Float32",
        ],
    }
}


class TestPseudoOpsPerfUtils:
    def test_get_pseudo_op_mappings_covers_extension_modules(self):
        mappings = get_pseudo_op_mappings()
        utils_mappings = _mappings_from_utils()
        assert mappings.keys() == utils_mappings.keys()
        assert mappings["pseudo_op::moe_aiter_fused_1stage"] is moe_aiter_fused_1stage
        assert mappings["aiter::rms_norm"] is aiter_rms_norm
        assert mappings["vllm::gdn_attention_core"] is gdn_attention_core
        assert mappings["aiter::gemm_a8w8_blockscale_ck"] is gemm_a8w8_blockscale
        assert mappings["_C_custom_ar::all_reduce"] is custom_ar_all_reduce

    def test_get_pseudo_op_category_only_mappings(self):
        category_only = get_pseudo_op_category_only_mappings()
        utils_category_only = _category_only_from_utils()
        assert category_only == utils_category_only
        assert (
            category_only["aiter::fused_dynamic_mxfp4_quant_moe_sort_hip"] == "MoE_aux"
        )
        assert category_only["aiter::indexer_score_topk"] == "InferenceAttention"


class TestPackageExports:
    def test_extension_base_categories(self):
        assert FusedMoE.category == "MoE_fused"
        assert InferenceAttention.category == "InferenceAttention"
        assert CustomCollective.category == "CustomCollective"
        assert RMSNorm.category == "RMSNorm"


class TestFusedMoEHelpers:
    def test_flops_and_bytes_static_helpers(self):
        flops = FusedMoE.flops_func(
            num_tokens=4,
            hidden_dim=128,
            inter_dim=256,
            topk=2,
            gated=True,
        )
        assert flops > 0
        bytes_moved = FusedMoE.bytes_func(
            num_tokens=4,
            hidden_dim=128,
            inter_dim=256,
            num_experts=8,
            topk=2,
            gated=True,
            input_bpe=2,
            weight_bpe=1,
            output_bpe=2,
        )
        assert bytes_moved > 0


class TestMoeExtensions:
    def test_moe_aiter_fused_1stage_constructs_and_estimates(self):
        model = moe_aiter_fused_1stage(MOE_FUSED_EVENT)
        details = model.param_details
        assert details["num_tokens"] == 32
        assert details["hidden_dim"] == 4096
        assert details["inter_dim"] == 7168
        assert details["topk"] == 2
        assert details["gated"] is True
        assert model.flops() > 0
        assert model.bytes() > 0
        assert model.get_maf_type() == "matrix"

    def test_biased_grouped_topk(self):
        event = {
            "args": {
                "Input Dims": [[64, 256], [256], [64, 8], [64, 8]],
                "Input type": [
                    "c10::BFloat16",
                    "c10::BFloat16",
                    "c10::Float32",
                    "c10::Int",
                ],
            }
        }
        model = BiasedGroupedTopk(event)
        assert model.flops() == 64 * 256
        assert model.bytes() > 0

    def test_moe_sort_scatter_gather(self):
        event = {
            "args": {
                "Input Dims": [[128, 8], [128, 8], [1024], [1024], [64], [2]],
                "Input type": ["c10::Int", "c10::Int", "c10::Int", "c10::Int"],
            }
        }
        model = MoeSortScatterGather(event)
        assert model.flops() == 128 * 8
        assert model.get_maf_type() == "vector"


class TestRmsNormExtensions:
    def test_aiter_rms_norm(self):
        event = {
            "args": {
                "Input Dims": [(4, 512), (512,), (), ()],
                "Input type": [
                    "c10::BFloat16",
                    "c10::BFloat16",
                    "Scalar",
                    "Scalar",
                ],
                "Input Strides": [(512, 1), (1,), (), ()],
            }
        }
        model = aiter_rms_norm(event)
        assert model.param_details["num_channels"] == 512
        assert model.flops() > 0
        assert model.bytes() > 0


class TestCustomCollectivesExtensions:
    def test_custom_ar_all_reduce(self):
        event = {
            "args": {
                "Input Dims": [(), (4, 7168), (4, 7168), (), ()],
                "Input type": [
                    "Scalar",
                    "c10::BFloat16",
                    "c10::BFloat16",
                    "Scalar",
                    "Scalar",
                ],
                "Input Strides": [(), (7168, 1), (7168, 1), (), ()],
            }
        }
        model = custom_ar_all_reduce(event)
        assert model.flops() == 0
        assert model.bytes() == 4 * 7168 * 2 * 2

    def test_aiter_fused_allreduce_rmsnorm(self):
        event = {
            "args": {
                "Input Dims": [
                    (1,),
                    (),
                    (4, 7168),
                    (4, 7168),
                    (4, 7168),
                    (4, 7168),
                    (7168,),
                    (),
                    (),
                    (),
                ],
                "Input type": ["Scalar"] * 10,
                "Input Strides": [
                    (1,),
                    (),
                    (7168, 1),
                    (7168, 1),
                    (7168, 1),
                    (7168, 1),
                    (1,),
                    (),
                    (),
                    (),
                ],
            }
        }
        model = aiter_fused_allreduce_rmsnorm(event)
        assert model.param_details["num_channels"] == 7168
        assert model.flops() > 0
        assert model.bytes() > 0


class TestAttentionExtensions:
    def test_gdn_attention_core(self):
        event = {
            "annotation": _GDN_ANNOTATION,
            "args": {
                "Input Dims": [
                    [64, 1536],
                    [64, 8],
                    [64, 8],
                    [64, 8, 128],
                    [],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::BFloat16",
                    "c10::BFloat16",
                    "c10::BFloat16",
                    "Scalar",
                ],
            },
        }
        model = gdn_attention_core(event)
        assert model.H_V == 8
        assert model.d_k == 64
        assert model.d_v == 128
        expected_flops = 64 * 8 * 7 * 128 * 64
        assert model.flops() == expected_flops
        assert model.bytes() > 0


class TestPerfModelExtensions:
    def test_gemm_a8w8_blockscale(self):
        event = {
            "args": {
                "Input Dims": [[128, 256], [512, 256], [128, 2], [4, 2]],
                "Input type": [
                    "signed char",
                    "signed char",
                    "c10::Float",
                    "c10::Float",
                ],
            }
        }
        model = gemm_a8w8_blockscale(event)
        assert model.param_details["M"] == 128
        assert model.param_details["N"] == 512
        assert model.param_details["K"] == 256
        assert model.flops() == 2 * 128 * 512 * 256
        assert model.bytes() > 0

    @pytest.mark.parametrize(
        "model_cls,flops_per_elem",
        [
            (aiter_silu_and_mul, 5),
            (aiter_gelu_and_mul, 8),
            (aiter_gelu_tanh_and_mul, 10),
        ],
    )
    def test_gated_activation_extensions(self, model_cls, flops_per_elem):
        event = {
            "args": {
                "Input Dims": [(4, 256), (4, 512)],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(256, 1), (512, 1)],
            }
        }
        model = model_cls(event)
        num_elems = 4 * 256
        assert model.flops() == flops_per_elem * num_elems
        assert model.bytes() > 0
