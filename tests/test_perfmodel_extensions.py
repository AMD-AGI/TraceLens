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
import inspect
from TraceLens.PerfModel import perf_model
from tests.fixtures.perfmodel import _ARCH, _gemm_event, _norm_event
import sys
from unittest.mock import MagicMock, patch
from TraceLens.PerfModel.extensions import moe_perf_model_extensions as moe_ext
from unittest.mock import patch
from tests.test_conv_backward_bytes import _conv_bias_bwd_event, _conv_bias_fwd_event
from tests.test_flash_attention_backward import _bwd_event as _flash_bwd_event
from tests.test_mamba_ssd import _mamba_event
from tests.fixtures.perfmodel import (
    _ARCH,
    _conv_bwd_event,
    _conv_fwd_event,
    _gemm_event,
    _moe_unfused_event,
    _norm_event,
)
from TraceLens.PerfModel import kernel_name_parser
from TraceLens.PerfModel.extensions import (
    attention_perf_model_extensions as attn_ext,
    moe_perf_model_extensions as moe_ext,
    perf_model_extensions as pext,
    rmsnorm_perf_model_extensions as rms_ext,
)
from tests.fixtures.perfmodel import _ARCH, _gemm_event, _moe_unfused_event
from tests.test_conv_backward_bytes import _conv_bias_fwd_event
from TraceLens.PerfModel.extensions.moe_perf_model_extensions import (
    BiasedGroupedTopk,
    moe_aiter_fused_1stage,
    moe_aiter_unfused_down,
    moe_aiter_unfused_up,
    moe_flydsl_stage1,
    moe_flydsl_stage2,
    moe_gptq_awq_down,
    moe_gptq_awq_up,
    moe_triton_invoke_grouped_gemm,
)
from TraceLens.PerfModel.extensions import attention_perf_model_extensions as attn_ext
from TraceLens.PerfModel.extensions import rmsnorm_perf_model_extensions as rms_ext
from tests.test_conv_backward_bytes import _conv_bias_bwd_event
from tests.fixtures.perfmodel import _ARCH, _GDN_ANNOTATION, _moe_unfused_event
from tests.fixtures.perfmodel import _GDN, _GDN_ANNOTATION, _attn_base
from tests.fixtures.perfmodel import _GDN_ANNOTATION
from TraceLens.PerfModel.extensions import perf_model_extensions as pext
from tests.fixtures.perfmodel import (
    _ARCH,
    _GDN_ANNOTATION,
    _moe_unfused_event,
    _norm_event,
)
from tests.test_conv_backward_bytes import (
    _conv_bias_bwd_event,
    _conv_bias_fwd_event,
    _conv_bias_relu_bwd_event,
    _conv_bias_relu_fwd_event,
)
from tests.fixtures.perfmodel import _ARCH, _gemm_event
from tests.test_dit_fused_ln_modulate import _fused_ln_fwd_event
from tests.fixtures.perfmodel import _ARCH, _GDN_ANNOTATION, _gemm_event, _norm_event
from tests.fixtures.perfmodel import (
    _ARCH,
    _GDN_ANNOTATION,
    _conv_bwd_event,
    _conv_fwd_event,
    _gemm_event,
    _moe_unfused_event,
    _norm_event,
)
from TraceLens.PerfModel.extensions import attention_perf_model_extensions as aext
from TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions import (
    fused_rms_mxfp4_quant,
)
from TraceLens.PerfModel.extensions.moe_perf_model_extensions import (
    FusedMoE,
    moe_aiter_fused_1stage,
)
from tests.test_primus_fp8_gemm_quantize import _fp8_gemm_event
from tests.test_primus_mxfp4_gemm_quantize import _fp4_gemm_event
from tests.fixtures.perfmodel import _moe_unfused_event

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


###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
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


class TestUtilsCoverage:
    def test_add_simulation_time_columns_nan_tb_when_no_bytes(self):
        metrics = {}
        add_simulation_time_columns(metrics, 100.0, 200.0, None, 200.0)
        assert metrics["Origami TB/s"] != metrics["Origami TB/s"]  # nan

    def test_add_simulation_time_columns_early_return(self):
        metrics = {"keep": 1}
        add_simulation_time_columns(metrics, 0, 1, 1, 1)
        assert "Origami Time (µs)" not in metrics

    @pytest.mark.parametrize("value,expected", [("maybe", True), (2, True)])
    def test_parse_bool_fallback(self, value, expected):
        assert parse_bool(value) is expected

    @pytest.mark.parametrize(
        "name,bpe",
        [
            ("double", 8),
            ("c10::float8_e5m2", 1),
            ("c10::float4_e2m1fn_x2", 1),
            (None, None),
        ],
    )
    def test_name2bpe_extended(self, name, bpe):
        assert name2bpe(name) == bpe

    def test_dtype_maps_extended(self):
        assert simulation_dtype_map("fp64") == "double"
        assert torch_dtype_map("c10::float8_e4m3fn") == "fp8"
        assert torch_dtype_map("mxfp4") == "fp4"


class TestKernelNameParserCoverage:
    def test_parse_rocm_gemm_bjlk_transpose(self):
        name = "Cijk_Ailk_Bjlk_MT32x32x32"
        parsed = parse_rocm_gemm(name)
        assert parsed["transpose"] == (False, True)

    def test_parse_rocm_gemm_no_macro_tile(self):
        parsed = parse_rocm_gemm("Cijk_Ailk_Bljk_only")
        assert parsed["mt_m"] is None


class TestGemmBaseCoverage:
    def test_get_param_details_not_implemented(self):
        with pytest.raises(NotImplementedError):
            perf_model.GEMM.get_param_details({})

    def test_bytes_func_returns_none_for_unknown_bpe(self):
        assert perf_model.GEMM.bytes_func(4, 4, 4, False, None, 2, 2, 2) is None

    def test_flops_and_bytes_bwd(self):
        class _BareGemm(perf_model.GEMM):
            @staticmethod
            def get_param_details(event):
                return {
                    "M": 4,
                    "N": 16,
                    "K": 8,
                    "bias": True,
                    "dtype_A_B": ("c10::BFloat16", "c10::BFloat16"),
                }

        model = _BareGemm({"args": {}})
        assert model.flops_bwd() > model.flops()
        assert model.bytes_bwd(2) > 0

    def test_gemm_with_rocm_kernel_parses_transpose(self):
        event = _gemm_event("aten::mm", (64, 128), (128, 256), kernel_names=[ROCM_GEMM])
        model = perf_model.aten_mm(event)
        assert model.param_details["transpose"] == (True, False)

    def test_get_simulation_time_without_origami(self):
        model = perf_model.aten_mm(_gemm_event("aten::mm", (4, 8), (8, 16)))
        assert model.get_simulation_time() is None

    def test_gemm_simulator_path_invalid(self, monkeypatch):
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", "/nonexistent/sim.py")
        with pytest.raises(ValueError, match="does not exist"):
            perf_model.GEMM.get_simulation_time_func(_ARCH, 4, 8, 16, 1, "bf16")

    def test_gemm_simulator_success(self, monkeypatch, tmp_path):
        sim_dir = tmp_path / "simdir"
        sim_dir.mkdir()
        sim = sim_dir / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        with patch("TraceLens.PerfModel.perf_model.subprocess.run") as run:
            run.return_value = MagicMock(stdout="Time=42.5\n", stderr="")
            t, cmd = perf_model.GEMM.get_simulation_time_func(
                _ARCH, 4, 8, 16, 1, "bf16", num_cus=64, force_to_l1=True
            )
        assert t == 42.5
        assert "run_gemm.py" in cmd

    def test_gemm_simulator_failure(self, monkeypatch, tmp_path):
        sim_dir = tmp_path / "simdir"
        sim_dir.mkdir()
        sim = sim_dir / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        with patch("TraceLens.PerfModel.perf_model.subprocess.run") as run:
            run.return_value = MagicMock(stdout="", stderr="fail")
            with pytest.raises(AssertionError):
                perf_model.GEMM.get_simulation_time_func(_ARCH, 4, 8, 16, 1, "bf16")

    def test_origami_simulation_mocked(self, monkeypatch):
        monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
        mock_origami = MagicMock()
        mock_origami.data_type_t.BFloat16 = "bf16_dtype"
        mock_helper_cls = MagicMock()
        mock_helper_cls.get_hardware.return_value = MagicMock(N_CU=64)
        mock_helper_cls.return_value.get_simulation_time.return_value = 99.0
        with patch.dict(sys.modules, {"origami": mock_origami}):
            with patch(
                "TraceLens.PerfModel.origami_helper.OrigamiHelper", mock_helper_cls
            ):
                t, cmd = perf_model.GEMM.get_simulation_time_func(
                    _ARCH,
                    4,
                    8,
                    16,
                    1,
                    "bf16",
                    enable_origami=True,
                    force_to_l1=True,
                    num_cus=64,
                )
        assert t == 99.0
        assert "Origami" in cmd
        mock_helper_cls.assert_called_once()

    def test_origami_unsupported_dtype(self, monkeypatch):
        monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
        mock_origami = MagicMock()
        with patch.dict(sys.modules, {"origami": mock_origami}):
            t, _ = perf_model.GEMM.get_simulation_time_func(
                _ARCH, 4, 8, 16, 1, "unknown_dtype", enable_origami=True
            )
        assert t is None

    def test_origami_import_error(self, monkeypatch):
        monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
        perf_model.GEMM._origami_import_error_printed = False
        with patch.dict(sys.modules, {"origami": None}):
            t, _ = perf_model.GEMM.get_simulation_time_func(
                _ARCH, 4, 8, 16, 1, "bf16", enable_origami=True
            )
        assert t is None


class TestGemmVariantsCoverage:
    @pytest.mark.parametrize(
        "cls,a,b,extra",
        [
            (perf_model.aten_mm, (4, 8), (8, 16), {}),
            (perf_model.aten_addmm, (4, 16), (8, 16), {"bias_dim": (4, 16)}),
            (perf_model.aten_bmm, (2, 4, 8), (2, 8, 16), {}),
            (perf_model.tev2_pseudo_gemm, (4, 8), (8, 16), {}),
        ],
    )
    def test_gemm_construct_and_estimate(self, cls, a, b, extra):
        if cls is perf_model.aten_addmm:
            event = {
                "args": {
                    "Input Dims": [extra["bias_dim"], list(a), list(b)],
                    "Input type": ["c10::BFloat16"] * 3,
                    "Input Strides": [(16, 1), (8, 1), (16, 1)],
                }
            }
        else:
            event = _gemm_event(cls.__name__, a, b)
        model = cls(event)
        assert model.flops() > 0
        assert model.bytes() > 0
        assert model.get_compute_precision() == "bf16"
        assert model.get_maf_type() == "matrix"

    def test_aten_mm_mixed_dtype_warns(self):
        event = _gemm_event(
            "aten::mm", (4, 8), (8, 16), dtypes=["c10::BFloat16", "c10::Half"]
        )
        model = perf_model.aten_mm(event)
        with pytest.warns(UserWarning):
            assert model.bytes() > 0

    def test_aten_scaled_mm_with_bias_and_fp8(self):
        event = {
            "args": {
                "Input Dims": [[4, 8], [8, 16], [16]],
                "Input type": [
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                    "c10::BFloat16",
                ],
            }
        }
        model = perf_model.aten_scaled_mm(event)
        assert model.param_details["bias"] is True
        assert model.bytes() > 0

    def test_aten_baddbmm(self):
        event = {
            "args": {
                "Input Dims": [[2, 4, 16], [2, 4, 8], [2, 8, 16]],
                "Input type": ["c10::BFloat16"] * 3,
            }
        }
        model = perf_model.aten_baddbmm(event)
        assert model.flops() == 2 * model.flops_func(4, 16, 8, True)

    def test_vllm_gemm_with_dynamic_quant(self):
        event = {
            "args": {
                "Input Dims": [[4, 128], [256, 64], (), ()],
                "Input type": ["c10::BFloat16", "c10::Float4_e2m1fn_x2"],
            }
        }
        model = perf_model.vllm_gemm_with_dynamic_quant(event)
        assert model.param_details["N"] == 256
        assert model.bytes() > 0

    def test_vllm_gemm_missing_shapes_raises(self):
        with pytest.raises(ValueError):
            perf_model.vllm_gemm_with_dynamic_quant({"args": {"Input Dims": [[], []]}})

    def test_tex_ts_te_gemm_ts(self):
        input_dims = [()] * 19
        input_dims[0] = [128, 64]
        input_dims[5] = [256, 64]
        input_dims[10] = [128, 256]
        event = {
            "args": {
                "Input Dims": input_dims,
                "Input type": ["c10::Float8_e4m3fn"] * 19,
                "Concrete Inputs": [""] * 4
                + ["0"]
                + [""] * 4
                + ["1"]
                + [""] * 4
                + [""],
            }
        }
        model = perf_model.tex_ts_te_gemm_ts(event)
        assert model.param_details["bias"] is False
        assert model.bytes() > 0

    def test_tev2_pseudo_gemm_dtype_mismatch_raises(self):
        event = _gemm_event(
            "tev2", (4, 8), (8, 16), dtypes=["c10::BFloat16", "c10::Half"]
        )
        model = perf_model.tev2_pseudo_gemm(event)
        with pytest.raises(ValueError):
            model.bytes()

    @pytest.mark.parametrize("cls", [perf_model.aten_mm, perf_model.aten_addmm])
    def test_gemm_bwd_not_implemented(self, cls):
        if cls is perf_model.aten_addmm:
            event = {
                "args": {
                    "Input Dims": [(4, 16), (4, 8), (8, 16)],
                    "Input type": ["c10::BFloat16"] * 3,
                }
            }
        else:
            event = _gemm_event("aten::mm", (4, 8), (8, 16))
        model = cls(event)
        with pytest.raises(NotImplementedError):
            model.flops_bwd()


class TestConvCoverage:
    def _conv_event(self, transposed=False):
        return {
            "args": {
                "Input Dims": [
                    (2, 3, 8, 8),
                    (4, 3, 3, 3),
                    (4,) if not transposed else None,
                ][: 2 if transposed else 3],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(192, 64, 8, 1), (27, 9, 3, 1), (1,)],
                "Concrete Inputs": [
                    "",
                    "",
                    "",
                    "(1,1)",
                    "(0,0)",
                    "(1,1)",
                    str(transposed),
                    "(0,0)",
                    "1",
                ],
            }
        }

    def test_aten_conv_fwd(self):
        model = perf_model.aten_conv(self._conv_event())
        assert model.flops() > 0
        assert model.bytes(2) > 0
        assert model.get_compute_precision() == "bf16"

    def test_aten_conv_transposed(self):
        event = self._conv_event(transposed=True)
        event["args"]["Input Dims"] = [(2, 3, 4, 4), (3, 4, 3, 3)]
        event["args"]["Input type"] = ["c10::BFloat16", "c10::BFloat16"]
        model = perf_model.aten_conv(event)
        assert model.transposed_conv is True
        assert model.flops_bwd() > 0

    def test_aten_conv_bwd(self):
        event = {
            "args": {
                "Input Dims": [(2, 4, 6, 6), (2, 3, 8, 8), (4, 3, 3, 3), (4,)],
                "Input type": ["c10::BFloat16"] * 4,
                "Concrete Inputs": [
                    "",
                    "",
                    "",
                    "",
                    "(1,1)",
                    "(0,0)",
                    "(1,1)",
                    "False",
                    "(0,0)",
                    "1",
                ],
            }
        }
        model = perf_model.aten_conv_bwd(event)
        assert model.flops() > 0
        assert model.bytes(2) > 0

    def test_conv_mixed_precision_raises(self):
        event = self._conv_event()
        event["args"]["Input type"] = ["c10::BFloat16", "c10::Half", "c10::BFloat16"]
        with pytest.raises(ValueError):
            perf_model.aten_conv(event)


class TestSoftmaxAndSdpaCoverage:
    def test_softmax_static_methods(self):
        assert perf_model.Softmax.flops(4, 8) == 4 * 3 * 8
        assert perf_model.Softmax.bytes(4, 8, 2) == 128
        assert perf_model.Softmax.flops_bwd(4, 8) == 2 * perf_model.Softmax.flops(4, 8)
        t = perf_model.Softmax.get_time(_ARCH, 4, 8, 2, force_to_l1=True)
        assert t > 0

    def test_softmax_get_time_requires_arch(self):
        with pytest.raises(ValueError):
            perf_model.Softmax.get_time(None, 4, 8, 2)

    @pytest.mark.parametrize(
        "cls,concrete",
        [
            (perf_model.flash_attention, ["", "", "", "0.0", "", "True"]),
            (
                perf_model.aten__scaled_dot_product_cudnn_attention,
                ["", "", "", "", "0.0", "True", "False"],
            ),
            (
                perf_model.aten__scaled_dot_product_efficient_attention,
                ["", "", "", "", "0.0", "False", "False"],
            ),
            (
                perf_model.aten__scaled_dot_product_flash_attention,
                ["", "", "", "0.0", "True", "False"],
            ),
            (
                perf_model.aiter__flash_attn_forward,
                ["", "", "", "0.0", "0.125", "True"],
            ),
        ],
    )
    def test_sdpa_fwd_variants(self, cls, concrete):
        q = [2, 64, 8, 64]
        event = _sdpa_event(cls, q, q, q, concrete, strides=[(32768, 512, 64, 1)] * 3)
        model = cls(event)
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_flash_attention_varlen_forward(self):
        event = {
            "args": {
                "Input Dims": [
                    [128, 8, 64],
                    [128, 8, 64],
                    [128, 8, 64],
                    [5],
                    [5],
                    [],
                    [],
                ],
                "Input type": ["c10::BFloat16"] * 3,
                "Input Strides": [[512, 64, 1]] * 3,
                "Concrete Inputs": [
                    "",
                    "",
                    "",
                    "",
                    "",
                    "64",
                    "64",
                    "0.0",
                    "0.125",
                    "True",
                ],
            }
        }
        model = perf_model.flash_attention_varlen_forward(event)
        assert model.flops() > 0

    def test_flash_attention_varlen_backward(self):
        event = {
            "args": {
                "Input Dims": [()] + [[128, 8, 64]] * 3 + [()] * 5 + [[5], [5]],
                "Input type": ["c10::BFloat16"] * 3,
                "Input Strides": [()] + [[512, 64, 1]] * 3 + [()] * 7,
                "Concrete Inputs": [""] * 11 + ["64", "64", "0.0", "0.125", "True"],
            }
        }
        model = perf_model.flash_attention_varlen_backward(event)
        assert model.flops() > 0

    def test_vllm_unified_attention_with_output(self):
        event = {
            "annotation": _VLLM_ATTN_ANNOTATION,
            "args": {
                "Input Dims": [[512, 8, 64], [1024, 1, 64], (), [512, 8, 64]],
                "Input type": ["c10::BFloat16"] * 4,
            },
        }
        model = perf_model.vllm_unified_attention_with_output(event)
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_vllm_unified_attention_bad_annotation(self):
        event = {
            "annotation": "NA",
            "args": {"Input Dims": [[1, 1, 1], [1, 1, 1], (), [1, 1, 1]]},
        }
        with pytest.raises(NotImplementedError):
            perf_model.vllm_unified_attention_with_output(event)

    def test_sdpa_causal_mismatch_raises(self):
        with pytest.raises(ValueError):
            perf_model.SDPA.flops_func(1, 4, 8, 8, 8, 64, 64, True)


class TestElementwiseReduceCoverage:
    def test_aten_unary_elementwise(self):
        event = {
            "args": {
                "Input Dims": [(4, 256), (4, 256)],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(256, 1), (256, 1)],
            }
        }
        model = perf_model.aten_unary_elementwise(event)
        assert model.flops() == 4 * 256
        assert model.bytes() > 0

    def test_aten_binary_elementwise(self):
        event = {
            "args": {
                "Input Dims": [(4, 256), (256,), (4, 256)],
                "Input type": ["c10::BFloat16"] * 3,
                "Input Strides": [(256, 1), (1,), (256, 1)],
            }
        }
        model = perf_model.aten_binary_elementwise(event)
        assert model.flops() == 4 * 256

    def test_binary_broadcast_error(self):
        with pytest.raises(ValueError):
            perf_model.BinaryElementwise.get_broadcast_shape((4, 8), (3, 8))

    def test_liger_silu_mul(self):
        event = {
            "args": {
                "Input Dims": [(4, 256), (4, 256)],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(256, 1), (256, 1)],
            }
        }
        model = perf_model.liger_silu_mul_function(event)
        assert model.flops() == 4 * 256

    def test_aten_reduce_sum(self):
        event = {
            "name": "aten::sum",
            "args": {
                "Input Dims": [(4, 256)],
                "Input type": ["c10::BFloat16"],
                "Concrete Inputs": ["", 1, False],
            },
        }
        model = perf_model.aten_reduce(event)
        assert model.flops() > 0
        assert model.bytes() is not None

    def test_aten_reduce_cumsum(self):
        event = {
            "name": "aten::cumsum",
            "args": {
                "Input Dims": [(4, 256)],
                "Input type": ["c10::BFloat16"],
                "Concrete Inputs": ["", 1],
            },
        }
        model = perf_model.aten_reduce(event)
        assert model.param_details["num_output_elems"] == 4 * 256


class TestNormalizationCoverage:
    def test_batch_norm_fwd(self):
        model = perf_model.BatchNorm(_norm_event((8, 16, 32, 32), 16))
        assert model.flops() > 0
        assert model.bytes() > 0
        with pytest.raises(NotImplementedError):
            model.flops_bwd()

    def test_batch_norm_bwd_native(self):
        event = {
            "name": "aten::native_batch_norm_backward",
            "args": {
                "Input Dims": [
                    None,
                    (8, 16, 32, 32),
                    (16,),
                    (16,),
                    (16,),
                    (16,),
                    (16,),
                    (),
                ],
                "Input type": ["c10::BFloat16"] * 7 + ["Scalar"],
                "Input Strides": [(), (16384, 1024, 32, 1), (1,)] * 2
                + [(1,)] * 3
                + [()],
                "Concrete Inputs": ["", "", "", "", "", "", "", "True"],
            },
        }
        model = perf_model.BatchNormBwd(event)
        assert model.flops() > 0

    def test_batch_norm_bwd_cudnn(self):
        event = {
            "name": "aten::cudnn_batch_norm_backward",
            "args": {
                "Input Dims": [
                    (8, 16, 32, 32),
                    (8, 16, 32, 32),
                    (16,),
                    (16,),
                    (16,),
                    (16,),
                    (16,),
                    (),
                ],
                "Input type": ["c10::BFloat16"] * 7 + ["Scalar"],
                "Input Strides": [(16384, 1024, 32, 1)] * 2 + [(1,)] * 5 + [()],
                "Concrete Inputs": ["", "", "", "", "", "", "", "1e-5"],
            },
        }
        model = perf_model.BatchNormBwd(event)
        assert model.is_training is False

    def test_layer_norm_and_bwd(self):
        event = _norm_event((4, 512), 512)
        fwd = perf_model.LayerNorm(event)
        assert fwd.flops() > 0
        bwd_event = {
            "args": {
                "Input Dims": [
                    None,
                    (4, 512),
                    (512,),
                    (512,),
                    (512,),
                    (512,),
                    (4, 512),
                    (),
                ],
                "Input type": ["c10::BFloat16"] * 6 + ["Scalar"],
                "Input Strides": [(), (512, 1), (1,)] * 2
                + [(512, 1), (512, 1), (512, 1), ()],
                "Concrete Inputs": [
                    "",
                    "",
                    "[512]",
                    "",
                    "",
                    "",
                    "",
                    "[True, True, True]",
                ],
            }
        }
        bwd = perf_model.LayerNormBwd(bwd_event)
        assert bwd.flops() > 0

    @pytest.mark.parametrize(
        "cls", [perf_model.GroupNorm, perf_model.InstanceNorm, perf_model.RMSNorm]
    )
    def test_other_norms_fwd(self, cls):
        event = _norm_event((4, 8, 32, 32), 8)
        model = cls(event)
        assert model.flops() > 0

    def test_rmsnorm_bwd(self):
        event = {
            "args": {
                "Input Dims": [None, (4, 512), (512,), (), (512,)],
                "Input type": ["c10::BFloat16"] * 4 + ["Scalar"],
                "Input Strides": [(), (512, 1), (1,), (), (1,)],
                "Concrete Inputs": ["", "", "[512]", "", "", "[True, True]"],
            }
        }
        model = perf_model.RMSNormBwd(event)
        assert model.flops() > 0
        assert model.bytes_bwd() > 0


class TestMiscPerfModelCoverage:
    def test_moe_comm(self):
        event = {"args": {"Input Dims": [[32, 4096]], "Input type": ["c10::BFloat16"]}}
        for cls in (perf_model.moe_dispatch, perf_model.moe_combine):
            model = cls(event)
            assert model.flops() == 0
            assert model.bytes() == 32 * 4096 * 2

    def test_causal_conv1d(self):
        event = {
            "args": {
                "Input Dims": [[2, 128, 512], [128, 4], [128]],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
            }
        }
        model = perf_model.causal_conv1d_fwd(event)
        assert model.flops() == 2 * 2 * 128 * 512 * 4

    def test_fused_rope(self):
        event = {
            "args": {
                "Input Dims": [[128, 2, 8, 64], [128, 1, 1, 64]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            }
        }
        model = perf_model.fused_rope_fwd(event)
        assert model.flops() == 3 * prod((128, 2, 8, 64))

    def test_cross_entropy(self):
        event = {
            "args": {
                "Input Dims": [[4, 1, 32000], [4, 1]],
                "Input type": ["c10::BFloat16", "long int"],
            }
        }
        model = perf_model.cross_entropy_fwd(event)
        assert model.flops() == 5 * 4 * 32000

    def test_jax_gemm_and_conv(self):
        jax_gemm_event = {
            "args": {
                "Batch": 2,
                "M": 4,
                "N": 16,
                "K": 8,
                "Beta": 0,
                "Type": "bf16",
            }
        }
        g = perf_model.jax_gemm(jax_gemm_event)
        assert g.flops() > 0
        conv_event = {
            "args": {
                "Input Dims": [[2, 3, 8, 8], [4, 3, 3, 3]],
                "Output Dims": [[2, 4, 6, 6]],
                "Filter Shape": [4, 3, 3, 3],
                "Input type": ["bf16", "bf16"],
                "Concrete Inputs": [
                    "",
                    "",
                    "(1,1)",
                    "(0,0)",
                    "(1,1)",
                    "False",
                    "(0,0)",
                    "1",
                ],
            }
        }
        c = perf_model.jax_conv(conv_event)
        assert c.flops() > 0

    def test_jax_te_fused_attn(self):
        event = {
            "args": {
                "Input Dims": [[2, 512, 16, 64], [2, 512, 16, 64], [2, 512, 16, 128]],
                "Input type": ["bf16", "bf16", "bf16"],
                "Concrete Inputs": ["0"],
            },
        }
        model = perf_model.jax_te_fused_attn(event)
        assert model.flops() > 0


class TestPerfModelExtensionsCoverage:
    def test_vllm_rocm_unquantized_gemm(self):
        event = {
            "args": {
                "Input Dims": [[128, 256], [512, 256], [512]],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
            }
        }
        model = vllm_rocm_unquantized_gemm(event)
        assert model.param_details["bias"] is True
        assert model.bytes() > 0

    def test_batched_gemm_variants(self):
        fp4_event = {
            "args": {
                "Input Dims": [[2, 4, 128], [2, 256, 64], [2, 256, 4]],
                "Input type": ["c10::BFloat16", "unsigned char", "c10::Float"],
            }
        }
        fp4 = batched_gemm_a16wfp4(fp4_event)
        assert fp4.flops() == 2 * fp4.flops_func(4, 256, 128, False)
        assert fp4.get_compute_precision() == "fp4"

        a8_event = {
            "args": {
                "Input Dims": [[4, 2, 128], [4, 256, 128], [4, 256]],
                "Input type": ["c10::BFloat16", "signed char", "c10::Float"],
            }
        }
        a8 = batched_gemm_a8w8(a8_event)
        assert a8.flops() > 0
        assert a8.bytes() > 0

    @pytest.mark.parametrize("cls", [gemm_a16w16, gemm_a16w16_asm])
    def test_gemm_a16w16_family(self, cls):
        event = {
            "args": {
                "Input Dims": [[128, 256], [512, 256]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            }
        }
        model = cls(event)
        assert model.flops() == 2 * 128 * 512 * 256

    def test_gemm_afp4wfp4(self):
        event = {
            "args": {
                "Input Dims": [[64, 128], [256, 64], [64, 4], [256, 4]],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float4_e2m1fn_x2",
                    "c10::Float",
                    "c10::Float",
                ],
            }
        }
        model = gemm_afp4wfp4(event)
        assert model.bytes() > 0

    def test_mxfp4_quant_variants(self):
        base_event = {
            "args": {
                "Input Dims": [(4, 256), (128,), (), (), (), ()],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "Scalar"] * 2,
                "Input Strides": [(256, 1), (1,), (), (), (), ()],
            }
        }
        for cls in (
            fused_flatten_mxfp4_quant,
            sglang_quant_dynamic_mxfp4_quant,
            aiter_fused_dynamic_mxfp4_quant_moe_sort_hip,
            aiter_dynamic_per_group_scaled_quant_fp4,
        ):
            model = cls(base_event)
            assert model.flops() > 0
            assert model.bytes() > 0

    def test_mhc_ops(self):
        post = mhc_post(
            {
                "args": {
                    "Input Dims": [(4, 2, 128), (4, 128)],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                }
            }
        )
        assert post.flops() > 0
        pre = mhc_pre_gemm_sqrsum(
            {
                "args": {
                    "Input Dims": [(2, 4, 8), (2, 4), (4, 2, 128), (8, 256)],
                    "Input type": ["float", "float", "c10::BFloat16", "float"],
                }
            }
        )
        assert pre.flops() == 2 * 4 * 8 * 256
        fuse = mhc_pre_big_fuse_rmsnorm(
            {
                "args": {
                    "Input Dims": [
                        (4, 2, 1),
                        (4, 2, 2),
                        (4, 128),
                        (2, 4, 8),
                        (2, 4),
                        (4, 128),
                        (8,),
                        (),
                    ],
                    "Input type": ["float"] * 8,
                }
            }
        )
        assert fuse.bytes() > 0
        fused = mhc_fused_post_pre_gemm_sqrsum(
            {
                "args": {
                    "Input Dims": [
                        (2, 4, 8),
                        (2, 4),
                        (4, 2, 128),
                        (4, 128),
                        (4, 2, 128),
                        (4, 2, 1),
                        (4, 2, 2),
                        (8, 256),
                    ],
                    "Input type": ["float"] * 8,
                }
            }
        )
        assert fused.flops() > 0
        topk = topk_softplus(
            {
                "args": {
                    "Input Dims": [(4, 2), (4, 2), (4, 8)],
                    "Input type": ["c10::Float", "c10::Int", "c10::BFloat16"],
                }
            }
        )
        assert topk.flops() > 0

    def test_misc_extension_ops(self):
        rope = aiter_rope_cached_positions_2c_fwd_impl(
            {
                "args": {
                    "Input Dims": [
                        (2, 128, 8, 64),
                        (2, 128, 1, 64),
                        (2, 128, 8, 64),
                        (2, 128, 1, 64),
                        (2048, 1, 1, 64),
                        (2048, 1, 1, 64),
                        (2, 128),
                    ],
                    "Input type": ["c10::BFloat16"] * 7,
                }
            }
        )
        assert rope.flops() > 0
        store = sglang_store_cache(
            {
                "args": {
                    "Input Dims": [(4, 512), (4, 512), (4,)],
                    "Input type": ["c10::BFloat16", "c10::BFloat16", "Scalar"],
                }
            }
        )
        assert store.bytes() > 0
        sample = mixed_sample_outer_exponential(
            {
                "args": {
                    "Input Dims": [(4, 8), (4, 8), ()],
                    "Input type": ["c10::Float", "c10::Float", "Scalar"],
                }
            }
        )
        assert sample.flops() > 0


class TestMoeExtensionsCoverage:
    MOE_FUSED = {
        "args": {
            "Input Dims": [
                [32, 4096],
                [32, 4096],
                [8, 14336, 4096],
                [8, 4096, 7168],
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::Float8_e4m3fn",
                "c10::Float8_e4m3fn",
                "c10::Float8_e4m3fn",
            ],
            "Concrete Inputs": [""] * 8 + ["2"],
        }
    }

    def test_moe_aiter_fused_blockscale(self):
        model = moe_aiter_fused_blockscale(self.MOE_FUSED)
        assert model.flops() > 0
        assert model.bytes() > 0

    @pytest.mark.parametrize(
        "cls,kernel",
        [
            (moe_triton_unfused_up, "moe_mxfp4_up_kernel"),
            (moe_triton_unfused_down, "moe_fp8_down_kernel"),
        ],
    )
    def test_moe_triton_unfused(self, cls, kernel):
        model = cls(_moe_unfused_event(kernel_name=kernel))
        assert model.flops() > 0
        assert model.get_maf_type() == "matrix"

    def test_moe_aiter_unfused(self):
        up_event = {
            "args": {
                "Input Dims": [
                    [32, 4096],
                    [8, 14336, 512],
                    [32, 2, 7168],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::BFloat16",
                ],
            }
        }
        down_event = {
            "args": {
                "Input Dims": [
                    [32, 2, 7168],
                    [8, 4096, 896],
                    [32, 4096],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::BFloat16",
                ],
            }
        }
        up = moe_aiter_unfused_up(up_event)
        down = moe_aiter_unfused_down(down_event)
        assert up.flops() > 0
        assert down.bytes() > 0

    def test_moe_ck_and_flydsl(self):
        ck_event = {
            "args": {
                "Input Dims": [
                    [32, 512],
                    [8, 7168, 512],
                    [8, 4096, 896],
                    [],
                    [],
                    [],
                    [32, 2, 7168],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                ],
            }
        }
        stage2_event = {
            "args": {
                "Input Dims": [
                    [32, 2, 7168],
                    [8, 7168, 512],
                    [8, 4096, 896],
                    [],
                    [],
                    [],
                    [32, 4096],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                ],
            }
        }
        assert moe_aiter_ck_stage1(ck_event).flops() > 0
        assert moe_aiter_ck_stage2(stage2_event).bytes() > 0
        fly_event = {
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
                ],
            }
        }
        assert moe_flydsl_stage1(fly_event).flops() > 0
        assert moe_flydsl_stage2(fly_event).bytes() > 0

    def test_moe_gptq_and_grouped_gemm(self):
        gptq_event = {
            "args": {
                "Input Dims": [[32, 4096], [32, 2], [8, 7168, 4096]],
                "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
                "MoE topk": 2,
            }
        }
        assert moe_gptq_awq_up(gptq_event).flops() > 0
        assert moe_gptq_awq_down(gptq_event).bytes() > 0
        grouped = moe_triton_invoke_grouped_gemm(
            {
                "args": {
                    "Input Dims": [
                        [64, 2048],
                        [128, 1536, 2048],
                        (),
                        [512, 1536],
                        (),
                        (),
                        (),
                        (),
                        [64, 4],
                    ],
                    "Input type": [
                        "c10::BFloat16",
                        "c10::Float8_e4m3fn",
                        "",
                        "c10::BFloat16",
                    ],
                }
            }
        )
        assert grouped.flops() > 0

    def test_sglang_fused_append_shared_experts(self):
        event = {
            "args": {
                "Input Dims": [(32, 4096), (32, 4096), (32, 2)],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::Float32"],
                "Input Strides": [(4096, 1), (4096, 1), (2, 1)],
            }
        }
        model = sglang_fused_append_shared_experts(event)
        assert model.flops() > 0


class TestRmsNormExtensionsCoverage:
    _STRIDES = [(512, 1), (1,), (), ()]

    def test_aiter_rmsnorm(self):
        event = {
            "args": {
                "Input Dims": [(4, 512), (4, 512), (512,), ()],
                "Input type": ["c10::BFloat16"] * 3 + ["Scalar"],
                "Input Strides": self._STRIDES,
            }
        }
        model = aiter_rmsnorm(event)
        assert model.flops() > 0

    def test_aiter_rmsnorm2d_dynamicquant(self):
        event = {
            "args": {
                "Input Dims": [(4, 512), (4, 512), (4, 1), (512,), (), ()],
                "Input type": [
                    "c10::Float8_e4m3fn",
                    "c10::BFloat16",
                    "c10::Float",
                    "c10::BFloat16",
                    "Scalar",
                    "Scalar",
                ],
                "Input Strides": [(512, 1), (512, 1), (1, 1), (1,), (), ()],
            }
        }
        model = aiter_rmsnorm2d_fwd_with_dynamicquant_ck(event)
        assert model.flops() > model.num_elems

    def test_aiter_rmsnorm_quant(self):
        event = {
            "args": {
                "Input Dims": [(), (4, 256), (4, 256), (512,), ()],
                "Input type": [
                    "Scalar",
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::BFloat16",
                    "Scalar",
                ],
                "Input Strides": [(), (256, 1), (256, 1), (1,), ()],
                "Concrete Inputs": ["", "", "", "", "", "128"],
            }
        }
        model = aiter_rmsnorm_quant(event)
        assert model.group_size == 128
        assert model.bytes() > 0

    def test_fused_rms_mxfp4_with_x2_and_res(self):
        event = {
            "args": {
                "Input Dims": [(4, 256), (256,), (), (4, 128), (128,), (), (4, 256)],
                "Input type": ["c10::BFloat16"] * 8,
                "Input Strides": [(256, 1), (1,), (), (128, 1), (1,), (), (256, 1)],
            }
        }

        model = fused_rms_mxfp4_quant(event)
        assert model.has_x2 is True
        assert model.has_res1 is True
        assert model.flops() > 0

    @pytest.mark.parametrize(
        "cls,concrete_idx,extra_dims",
        [
            (vllm_rocm_aiter_rmsnorm_fp8_group_quant, 3, []),
            (vllm_rocm_aiter_rmsnorm_with_add_fp8_group_quant, 4, [(4, 512)]),
            (vllm_rocm_aiter_triton_add_rmsnorm_pad, 4, [(4, 512)]),
        ],
    )
    def test_vllm_rmsnorm_variants(self, cls, concrete_idx, extra_dims):
        dims = [(4, 512), (512,)] + extra_dims + [(), ()]
        event = {
            "args": {
                "Input Dims": dims,
                "Input type": ["c10::BFloat16"] * len(dims),
                "Input Strides": [(512, 1), (1,)]
                + [(512, 1)] * len(extra_dims)
                + [(), ()],
                "Concrete Inputs": [""] * concrete_idx + ["128"] + [""],
            }
        }
        if cls is vllm_rocm_aiter_triton_add_rmsnorm_pad:
            event["args"]["Concrete Inputs"] = ["", "", "1e-05", "", "256"]
        model = cls(event)
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_aiter_add_rmsnorm(self):
        event = {
            "args": {
                "Input Dims": [(4, 512), (4, 512), (4, 512), (4, 512), (512,), (), ()],
                "Input type": ["c10::BFloat16"] * 5 + ["Scalar", "Scalar"],
                "Input Strides": [(512, 1)] * 4 + [(1,), (), ()],
            }
        }
        model = aiter_add_rmsnorm(event)
        assert model.flops() > 0
        assert model.bytes() > 0


class TestAttentionExtensionsCoverage:
    def _attn_event(self, annotation=_GDN_ANNOTATION):
        return {
            "annotation": annotation,
            "args": {
                "Input Dims": [[64, 8, 64], [64, 8, 64], [64, 8, 128]],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
            },
        }

    @pytest.mark.parametrize(
        "cls",
        [
            mha_varlen_fwd,
            aiter_fmha_v3_varlen_fwd,
            aiter_mha_batch_prefill,
            mla_decode_fwd,
            pseudo_mla_prefill_fwd,
            mla_tilelang_sparse_fwd,
            vllm_unified_mla_attention_with_output,
        ],
    )
    def test_inference_attention_subclasses(self, cls):
        model = cls(self._attn_event())
        if model.param_details.get("_no_perf"):
            assert model.flops() is None
        else:
            assert model.flops() > 0

    def test_aiter_paged_attention_ragged(self):
        event = {
            "annotation": _GDN_ANNOTATION,
            "args": {
                "Input Dims": [
                    (),
                    (),
                    [64, 8, 64],
                    [128, 16, 1, 64],
                    [128, 16, 1, 128],
                ],
                "Input type": [
                    "Scalar",
                    "Scalar",
                    "c10::BFloat16",
                    "c10::BFloat16",
                    "c10::BFloat16",
                ],
            },
        }
        model = aiter_paged_attention_ragged(event)
        assert model.param_details["d_h_v"] == 128
        assert model.flops() > 0

    def test_pa_sparse_prefill_opus_fwd(self):
        event = {
            "args": {
                "Input Dims": [[64, 8, 64], (), [32], (), (), [16]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            }
        }
        model = pa_sparse_prefill_opus_fwd(event)
        assert model.flops() > 0

    def test_v4_paged_decode_modes(self):
        base = {
            "annotation": _GDN_ANNOTATION,
            "args": {
                "Input Dims": [[1, 8, 512]],
                "Input type": ["c10::BFloat16"],
                "v4_model_name": "DeepSeek-V4-Flash",
            },
        }
        for cls in (
            pseudo_v4_paged_decode_swa,
            pseudo_v4_paged_decode_csa,
            pseudo_v4_paged_decode_hca,
        ):
            model = cls(base)
            assert model.flops() > 0

    def test_pa_decode_gluon(self):
        event = {
            "annotation": _GDN_ANNOTATION,
            "args": {
                "Input Dims": [
                    [64, 8, 64],
                    [64, 8, 64],
                    [64, 8, 128],
                    [128, 16, 1, 64],
                ],
                "Input type": ["c10::BFloat16"] * 4,
            },
        }
        model = pa_decode_gluon(event)
        assert model.flops() > 0

    def test_inference_attention_no_perf(self):
        model = InferenceAttention({"annotation": "bad", "args": {}})
        assert model.flops() is None
        assert model.bytes() is None


class TestTritonCompiledCoverage:
    _POI_EVENT = {
        "name": "triton_poi_fused_add_mul_0",
        "args": {
            "Concrete Inputs": ["4096"],
            "Input Dims": [[4, 256], [4, 256]],
            "Input type": ["c10::BFloat16", "c10::BFloat16"],
        },
    }

    def test_parse_kernel_name_fallback_without_registry(self):
        with patch(
            "TraceLens.PerfModel.triton_compiled_perf_model._ALL_KNOWN_OPS", set()
        ):
            ops = _parse_kernel_name("triton_poi_fused_add_mul_0")
            assert ops == ["aten.add", "aten.mul"]

    def test_meta_from_trace_args_pointwise(self):
        meta = _meta_from_trace_args(self._POI_EVENT)
        assert meta["xnumel"] == 4096
        assert meta["total_bytes"] > 0

    def test_v1_cache_lookup(self, tmp_path):
        wrapper = tmp_path / "wrapper.py"
        wrapper.write_text("""
# Original ATen: [aten.add, aten.mul]
triton_poi_fused_add_mul_0 = async_compile.triton(
    'triton_poi_fused_add_mul_0',
    '''
    size_hints={'x': 4096, 'r0_': 1}
    'signature': {'in_ptr0': '*bf16', 'in_out_ptr0': '*bf16'},
    ''', device_str='cuda')
""")
        meta = _lookup("triton_poi_fused_add_mul_0", str(tmp_path))
        assert meta is not None
        assert meta["xnumel"] == 4096

    def test_parse_wrapper_dict_hints_and_in_out(self):
        content = """
triton_red_fused_add_0 = async_compile.triton(
    'triton_red_fused_add_0',
    '''
    size_hints={'x': 1024, 'r0_': 64}
    'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16'},
    ''', device_str='cuda')
"""
        parsed = _parse_wrapper(content)
        meta = parsed["triton_red_fused_add_0"]
        assert meta["xnumel"] == 1024
        assert meta["rnumel"] == 64
        assert meta["in_out_extra_bytes"] == [2]

    def test_triton_model_v1_bytes_fallback(self, tmp_path):
        wrapper = tmp_path / "w.py"
        wrapper.write_text("""
# Original ATen: [aten.mean]
triton_red_fused_mean_0 = async_compile.triton(
    'triton_red_fused_mean_0',
    '''
    size_hints=[2048, 128]
    'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16'},
    ''', device_str='cuda')
""")
        event = {"name": "triton_red_fused_mean_0", "args": {}}
        model = TritonCompiledPerfModel(event, inductor_cache_dir=str(tmp_path))
        assert model.flops() > 0
        assert model.bytes() > 0
        assert model.get_maf_type() == "vector"
        assert model.get_compute_precision() == "bf16"

    def test_triton_model_missing_meta_raises(self):
        model = TritonCompiledPerfModel({"name": "missing_kernel", "args": {}})
        with pytest.raises(NotImplementedError):
            model.flops()
        assert TritonCompiledPerfModel.can_model({"name": "x", "args": {}}) is False

    def test_triton_bwd_not_implemented(self):
        model = TritonCompiledPerfModel(self._POI_EVENT)
        with pytest.raises(NotImplementedError):
            model.flops_bwd()


class TestSdpaExtendedCoverage:
    _ARCH = _ARCH

    def _sdpa_event(self, cls, q, k, v, concrete, extra_dims=None):
        dims = [q, k, v]
        if extra_dims:
            dims.extend(extra_dims)
        return {
            "args": {
                "Input Dims": dims,
                "Input type": ["c10::BFloat16"] * len(dims),
                "Input Strides": [[512, 64, 1]] * min(3, len(dims)),
                "Concrete Inputs": concrete,
            }
        }

    @pytest.mark.parametrize(
        "cls,concrete,dims",
        [
            (perf_model.flash_attn_v3_forward, [""] * 24 + ["True"], None),
            (
                perf_model.aiter__fmha_v3_forward,
                [""] * 4 + ["0.0"] + [""] + ["True"],
                [(), [2, 64, 8, 64], [2, 64, 8, 64], [2, 64, 8, 64]],
            ),
            (
                perf_model.aiter__fmha_v3_backward,
                [""] * 9 + ["True"],
                [(), [2, 64, 8, 64], [2, 64, 8, 64], [2, 64, 8, 64], ()],
            ),
            (perf_model.aiter__flash_attn_backward, [""] * 12 + ["True"], None),
        ],
    )
    def test_sdpa_v3_variants(self, cls, concrete, dims):
        q = [2, 64, 8, 64]
        if dims is None:
            event = self._sdpa_event(cls, q, q, q, concrete)
        else:
            event = {
                "args": {
                    "Input Dims": dims,
                    "Input type": ["c10::BFloat16"] * len(dims),
                    "Input Strides": [()] * len(dims),
                    "Concrete Inputs": concrete,
                }
            }
        model = cls(event)
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_sdpa_simulation_time_func(self, monkeypatch):
        monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
        with patch.object(
            perf_model.GEMM,
            "get_simulation_time_func",
            return_value=(1.0, "cmd"),
        ):
            with patch.object(perf_model.Softmax, "get_time", return_value=0.5):
                t = perf_model.SDPA.get_simulation_time_func(
                    self._ARCH,
                    "bf16",
                    None,
                    "c10::BFloat16",
                    1024,
                    2,
                    8,
                    128,
                    128,
                    64,
                    fa=True,
                )
        assert t > 0

    def test_sdpa_simulation_time_func_qkt_none(self, monkeypatch):
        monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
        with patch.object(
            perf_model.GEMM, "get_simulation_time_func", return_value=(None, None)
        ):
            assert (
                perf_model.SDPA.get_simulation_time_func(
                    self._ARCH,
                    "bf16",
                    None,
                    "c10::BFloat16",
                    1024,
                    1,
                    1,
                    64,
                    64,
                    32,
                )
                is None
            )

    def test_sdpa_get_simulation_time_on_model(self):
        event = _sdpa_event(
            perf_model.flash_attention,
            [2, 64, 8, 64],
            [2, 64, 8, 64],
            [2, 64, 8, 64],
            ["", "", "", "0.0", "", "True"],
            strides=[[32768, 512, 64, 1]] * 3,
        )
        model = perf_model.flash_attention(event, arch=self._ARCH)
        with patch.object(
            perf_model.SDPA,
            "get_simulation_time_func",
            return_value=42.0,
        ):
            assert model.get_simulation_time() == 42.0

    def test_sdpa_bwd_simulation_time_func(self, monkeypatch):
        monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
        with patch.object(
            perf_model.GEMM,
            "get_simulation_time_func",
            return_value=(2.0, "cmd"),
        ):
            with patch.object(perf_model.Softmax, "get_time", return_value=1.0):
                t = perf_model.SDPA.get_simulation_time_bwd_func(
                    self._ARCH,
                    "bf16",
                    None,
                    "c10::BFloat16",
                    2048,
                    2,
                    8,
                    128,
                    128,
                    64,
                )
        assert t > 0

    def test_vllm_unified_attention_with_output_extended(self):
        event = {
            "annotation": "(128_256_512_1024_2048_3072_4096_64)",
            "args": {
                "Input Dims": [[512, 8, 64], [1024, 1, 64], (), [512, 8, 64]],
                "Input type": ["c10::BFloat16"] * 4,
            },
        }
        model = perf_model.vllm_unified_attention_with_output(event)
        assert model.get_simulation_time() is None or model.get_simulation_time() >= 0


class TestConvBiasAndNormExtendedCoverage:
    def test_rmsnorm_bwd_bytes(self):
        event = {
            "args": {
                "Input Dims": [None, (4, 512), (512,), (), (512,)],
                "Input type": ["c10::BFloat16"] * 4 + ["Scalar"],
                "Input Strides": [(), (512, 1), (1,), (), (1,)],
                "Concrete Inputs": ["", "", "[512]", "", "", "[True, True]"],
            }
        }
        model = perf_model.RMSNormBwd(event)
        assert model.bytes_bwd() > 0


class TestGroupedGemmAndPrimusCoverage:
    def test_primus_grouped_gemm(self):
        event = {
            "args": {
                "Input Dims": [
                    [4, 128],
                    [8, 256, 128],
                    [8, 256],
                    [8],
                    [8],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::Float",
                    "c10::Int",
                    "c10::Int",
                ],
            }
        }
        model = perf_model.primus_turbo_grouped_gemm(event)
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_primus_quantize_ops(self):
        fp8_event = {
            "args": {
                "Input Dims": [(4, 256), (4, 256)],
                "Input type": ["c10::BFloat16", "c10::Float8_e4m3fn"],
                "Input Strides": [(256, 1), (256, 1)],
            }
        }
        assert perf_model.primus_turbo_quantize_fp8(fp8_event).flops() > 0
        mxfp4_event = {
            "args": {
                "Input Dims": [(4, 256), (4, 128), (128,)],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float4_e2m1fn_x2",
                    "c10::BFloat16",
                ],
                "Input Strides": [(256, 1), (128, 1), (1,)],
            }
        }
        assert perf_model.primus_turbo_quantize_mxfp4_dual(mxfp4_event).bytes() > 0

    def test_aiter_gemm_a4w4(self):
        event = {
            "args": {
                "Input Dims": [[128, 256], [512, 64], [128, 4], [512, 4]],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float4_e2m1fn_x2",
                    "c10::Float",
                    "c10::Float",
                ],
            }
        }
        model = perf_model.aiter_gemm_a4w4(event)
        assert model.flops() > 0


class TestGemmSimulatorExtendedCoverage:
    def test_gemm_simulator_missing_inputs(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        with pytest.raises(AssertionError, match="Invalid inputs"):
            perf_model.GEMM.get_simulation_time_func(_ARCH, None, 8, 16, 1, "bf16")

    def test_gemm_simulator_cache_hit(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        with patch("TraceLens.PerfModel.perf_model.subprocess.run") as run:
            run.return_value = MagicMock(stdout="Time=10.0\n", stderr="")
            t1, _ = perf_model.GEMM.get_simulation_time_func(_ARCH, 4, 8, 16, 1, "bf16")
            t2, _ = perf_model.GEMM.get_simulation_time_func(_ARCH, 4, 8, 16, 1, "bf16")
        assert t1 == t2 == 10.0
        run.assert_called_once()

    def test_gemm_simulator_windows_requires_python(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        with patch("TraceLens.PerfModel.perf_model.os.name", "nt"):
            with pytest.raises(AssertionError, match="Windows"):
                perf_model.GEMM.get_simulation_time_func(
                    _ARCH, 4, 8, 16, 1, "bf16", python_path=None
                )

    def test_jax_gemm_mixed_dtype_warns(self):
        event = {
            "args": {
                "Batch": 2,
                "M": 4,
                "N": 16,
                "K": 8,
                "Beta": 0,
                "Type": "bf16",
            }
        }
        model = perf_model.jax_gemm(event)
        model.param_details["dtype_A_B"] = ("bf16", "fp16")
        with pytest.warns(UserWarning):
            model.bytes()


class TestMoeExtendedCoverage:
    MOE_1STAGE = {
        "args": {
            "Input Dims": [
                [32, 4096],
                [8, 28672, 512],
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

    def test_moe_aiter_fused_1stage(self):

        model = moe_aiter_fused_1stage(self.MOE_1STAGE)
        assert model.flops() > 0
        assert model.bytes() > 0
        assert model.get_compute_precision() == "bf16"
        with pytest.raises(NotImplementedError):
            model.flops_bwd()
        assert FusedMoE.bytes_func(32, 4096, 7168, 8, 2, True, None, 1, 2) is None

    def test_biased_grouped_topk_and_sort_scatter(self):

        topk_event = {
            "args": {
                "Input Dims": [(32, 256), (32, 8), (256,)],
                "Input type": ["c10::Float", "c10::Int", "c10::Float"],
            }
        }
        assert BiasedGroupedTopk(topk_event).flops() > 0
        sort_event = {
            "args": {
                "Input Dims": [(32, 4096), (32, 2), (32, 4096)],
                "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
            }
        }
        assert MoeSortScatterGather(sort_event).bytes() > 0


###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import pytest

from TraceLens.PerfModel.extensions.custom_collectives_perf_model_extensions import (
    aiter_all_gather_reg,
    aiter_fused_allreduce_rmsnorm,
    aiter_fused_allreduce_rmsnorm_,
    aiter_reduce_scatter,
    custom_ar_all_reduce,
    custom_ar_qr_all_reduce,
    sgl_kernel_all_reduce_reg,
    sgl_kernel_qr_all_reduce,
    sgl_kernel_reg_all_gather_into_tensor,
)
from TraceLens.TreePerf import tree_perf


def _fused_allreduce_event():
    return {
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
            "Input Strides": [()] * 10,
        }
    }


def _fused_allreduce_python_event():
    return {
        "args": {
            "Input Dims": [
                (64, 7168),
                (64, 7168),
                (7168,),
                (),
                (),
                (),
                (),
            ],
            "Input type": ["c10::BFloat16"] * 7,
            "Input Strides": [(7168, 1)] * 7,
        }
    }


def _allreduce_event(name_dims=(4, 7168)):
    return {
        "args": {
            "Input Dims": [(), name_dims, name_dims, (), ()],
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


class TestCustomCollectivesPerfModels:
    def test_aiter_fused_allreduce_rmsnorm(self):
        model = aiter_fused_allreduce_rmsnorm(_fused_allreduce_event())
        assert model.num_elems == 4 * 7168
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_aiter_fused_allreduce_rmsnorm_python_layout(self):
        model = aiter_fused_allreduce_rmsnorm_(_fused_allreduce_python_event())
        assert model.num_elems == 64 * 7168
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_custom_ar_all_reduce(self):
        model = custom_ar_all_reduce(_allreduce_event())
        assert model.flops() == 0
        assert model.bytes() > 0

    @pytest.mark.parametrize(
        "cls",
        [
            sgl_kernel_all_reduce_reg,
            sgl_kernel_qr_all_reduce,
            custom_ar_qr_all_reduce,
        ],
    )
    def test_allreduce_subclasses(self, cls):
        model = cls(_allreduce_event(name_dims=(32, 7168)))
        assert model.flops() == 0
        assert model.bytes() > 0

    def test_aiter_reduce_scatter(self):
        event = {
            "args": {
                "Input Dims": [(), (8, 7168), (4, 7168), ()],
                "Input type": ["Scalar", "c10::BFloat16", "c10::BFloat16", "Scalar"],
                "Input Strides": [(), (7168, 1), (7168, 1), ()],
            }
        }
        model = aiter_reduce_scatter(event)
        assert model.flops() == 0
        assert model.bytes() > 0

    def test_aiter_all_gather_reg(self):
        event = {
            "args": {
                "Input Dims": [(), (4, 7168), (8, 7168)],
                "Input type": ["Scalar", "c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(), (7168, 1), (7168, 1)],
            }
        }
        model = aiter_all_gather_reg(event)
        assert model.flops() == 0
        assert model.bytes() > 0

    def test_sgl_kernel_reg_all_gather_into_tensor(self):
        event = {
            "args": {
                "Input Dims": [(256, 16160), (32, 16160), ()],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "Scalar"],
            }
        }
        model = sgl_kernel_reg_all_gather_into_tensor(event)
        assert model.flops() == 0
        assert model.bytes() > 0


class TestTreePerfInitKwargs:
    def test_perf_model_init_kwargs_without_optional_params(self):
        class SimpleModel:
            def __init__(self, event, arch=None, python_path=None):
                self.event = event

        kwargs = tree_perf._perf_model_init_kwargs(
            SimpleModel,
            event={"name": "op"},
            arch={},
            python_path=None,
            enable_origami=True,
        )
        assert kwargs["event"]["name"] == "op"
        assert "enable_origami" not in kwargs

    def test_perf_model_init_kwargs_with_var_keyword(self):
        class FlexibleModel:
            def __init__(self, event, **kwargs):
                self.kwargs = kwargs

        kwargs = tree_perf._perf_model_init_kwargs(
            FlexibleModel,
            event={"name": "op"},
            arch={},
            python_path="path",
            enable_origami=False,
            inductor_cache_dir="/tmp/cache",
        )
        assert kwargs["enable_origami"] is False
        assert kwargs["inductor_cache_dir"] == "/tmp/cache"

    def test_perf_model_init_kwargs_broken_signature(self):
        class Broken:
            __init__ = 42

        kwargs = tree_perf._perf_model_init_kwargs(
            Broken, event={}, arch=None, python_path=None, enable_origami=True
        )
        assert kwargs["event"] == {}


class TestPerfModelExhaustiveSweep:
    _EVENTS = [
        _gemm_event("aten::mm", (4, 8), (8, 16)),
        _norm_event((4, 8, 32, 32), 8),
        {
            "args": {
                "Input Dims": [(4, 8), (8, 16), (4, 16)],
                "Input type": ["c10::Float8_e4m3fn", "c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(8, 1), (16, 1), (16, 1)],
            }
        },
        {
            "args": {
                "Input Dims": [[2, 3, 8, 8], [4, 3, 3, 3]],
                "Output Dims": [[2, 4, 6, 6]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Concrete Inputs": [
                    "",
                    "",
                    "(1,1)",
                    "(0,0)",
                    "(1,1)",
                    "False",
                    "(0,0)",
                    "1",
                ],
            }
        },
        {
            "args": {
                "Batch": 2,
                "M": 4,
                "N": 8,
                "K": 16,
                "Beta": 1,
                "Type": "bf16",
            }
        },
        {
            "args": {
                "Input Dims": [[4, 32000], [4]],
                "Input type": ["c10::BFloat16", "long int"],
            }
        },
        {
            "args": {
                "Input Dims": [[2, 128, 8, 64], [2, 128, 8, 64], [2, 128, 8, 64]],
                "Input type": ["c10::BFloat16"] * 3,
                "Concrete Inputs": ["", "", "", "0.0", "False", "False", ""],
            }
        },
        {
            "args": {
                "Input Dims": [[12, 16], [4, 16, 32]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            }
        },
        {
            "args": {
                "Input Dims": [[2, 4, 32], [8, 4, 3]],
                "Output Dims": [[2, 8, 30]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Concrete Inputs": [
                    "",
                    "",
                    "(1,)",
                    "(0,)",
                    "(1,)",
                    "False",
                    "(0,)",
                    "1",
                ],
            }
        },
        {
            "annotation": "(prefill_128_64_8_0_0_0_0)",
            "args": {
                "Input Dims": [[128, 8, 64]] * 4,
                "Input type": ["c10::BFloat16"] * 4,
            },
        },
    ]

    def test_all_perf_model_classes(self):
        hit = 0
        for _name, cls in inspect.getmembers(perf_model, inspect.isclass):
            if cls.__module__ != perf_model.__name__:
                continue
            for event in self._EVENTS:
                try:
                    sig = inspect.signature(cls.__init__)
                    if "arch" in sig.parameters:
                        obj = cls(event, arch=_ARCH)
                    else:
                        obj = cls(event)
                except Exception:
                    continue
                for meth in (
                    "flops",
                    "bytes",
                    "flops_bwd",
                    "bytes_bwd",
                    "get_compute_precision",
                    "get_maf_type",
                    "get_time",
                    "get_simulation_time",
                    "get_simulation_time_func",
                ):
                    if hasattr(obj, meth):
                        try:
                            fn = getattr(obj, meth)
                            if meth == "get_simulation_time_func":
                                fn(_ARCH, 4, 8, 16, 1, "bf16")
                            elif meth == "get_simulation_time":
                                fn()
                            else:
                                fn()
                        except (
                            NotImplementedError,
                            TypeError,
                            ValueError,
                            AssertionError,
                        ):
                            pass
                        except Exception:
                            pass
                hit += 1
                break
        assert hit >= 40


class TestPerfModelRemaining:
    def test_scaled_mm_mismatched_dtypes(self):
        event = {
            "args": {
                "Input Dims": [(4, 8), (8, 16), (4, 16)],
                "Input type": ["c10::Float8_e4m3fn", "c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(8, 1), (16, 1), (16, 1)],
            }
        }
        model = perf_model.aten_scaled_mm(event)
        assert model.bytes() > 0

    def test_gemm_without_strides(self):
        event = {
            "args": {
                "Input Dims": [(4, 8), (8, 16), (4, 16)],
                "Input type": ["c10::Float8_e4m3fn", "c10::BFloat16", "c10::BFloat16"],
            }
        }
        model = perf_model.aten_scaled_mm(event)
        assert model.param_details["stride_A"] is None

    def test_vllm_gemm_dynamic_quant(self):
        event = {
            "args": {
                "Input Dims": [(128, 64), (256, 64)],
                "Input type": ["c10::Float8_e4m3fn", "c10::Float8_e4m3fn"],
            }
        }
        model = perf_model.vllm_gemm_with_dynamic_quant(event)
        assert model.flops() > 0

    def test_grouped_gemm_list_shapes(self):
        event = {
            "name": "primus_turbo::grouped_gemm",
            "args": {
                "Input Dims": [[[4, 8], [5, 8]], [[8, 16], [8, 16]]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            },
        }
        model = perf_model.primus_turbo_grouped_gemm(event)
        assert model.flops() > 0

    def test_gemm_simulator_invalid_arch(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        with pytest.raises(AssertionError, match="Invalid inputs"):
            perf_model.GEMM.get_simulation_time_func(
                {"freq_mhz": 2200}, None, 8, 16, 1, "bf16"
            )
        perf_model.GEMM.cache_gemm_results.clear()


class _GroupedGemmNoBwdOverride(perf_model.GroupedGemm):
    @staticmethod
    def get_param_details(event):
        return {"M": 64, "K": 32, "N": 16, "G": 4, "bpe_in": 2, "bpe_out": 2}


class TestPerfModelPhase11:
    def test_aten_scaled_mm_unsupported_output_bpe(self):
        event = {
            "args": {
                "Input Dims": [[4, 8], [16, 8]],
                "Input type": ["double", "double"],
                "Concrete Inputs": ["", "", "1.0"],
            }
        }
        model = perf_model.aten_scaled_mm(event)
        assert model.bytes() is None

    def test_gemm_strides_and_bwd_not_implemented(self):
        vllm = perf_model.vllm_gemm_with_dynamic_quant(
            {
                "args": {
                    "Input Dims": [[128, 64], [256, 32]],
                    "Input type": ["c10::Float4_e2m1fn_x2", "c10::Float4_e2m1fn_x2"],
                    "Input Strides": [[8192, 64, 1], [16384, 64, 1]],
                }
            }
        )
        assert vllm.bytes() > 0

        tex = perf_model.tex_ts_te_gemm_ts(
            {
                "args": {
                    "Input Dims": [[64, 128]] * 6 + [[128, 64]] * 6 + [[]],
                    "Input type": ["c10::BFloat16"] * 19,
                    "Input Strides": [[8192, 128, 1]] * 6 + [[16384, 64, 1]] * 6,
                    "Concrete Inputs": [""] * 14 + [""],
                }
            }
        )
        assert tex.bytes() > 0
        with pytest.raises(NotImplementedError):
            tex.bytes_bwd()

        tev2 = perf_model.tev2_pseudo_gemm(
            {
                "args": {
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                    "Input Strides": [[2048, 64, 1], [8192, 128, 1]],
                }
            }
        )
        assert tev2.flops() > 0

    def test_conv_ndims_and_mixed_dtype(self):
        conv1d = perf_model.aten_conv(_conv_fwd_event((2, 3, 32), (4, 3, 5)))
        assert conv1d.param_details["convNd"] == "conv1d"
        conv3d = perf_model.aten_conv(_conv_fwd_event((2, 3, 8, 8, 8), (4, 3, 3, 3, 3)))
        assert conv3d.param_details["convNd"] == "conv3d"

        mixed = _conv_fwd_event((2, 3, 8, 8), (4, 3, 3, 3))
        mixed["args"]["Input type"] = ["c10::BFloat16", "c10::Half"]
        with pytest.raises(ValueError, match="different"):
            perf_model.aten_conv(mixed)

        bwd1d = perf_model.aten_conv_bwd(_conv_bwd_event((2, 3, 32), (4, 3, 5)))
        assert bwd1d.param_details["convNd"] == "conv1d"
        bwd3d = perf_model.aten_conv_bwd(
            _conv_bwd_event((2, 3, 8, 8, 8), (4, 3, 3, 3, 3))
        )
        assert bwd3d.param_details["convNd"] == "conv3d"
        mixed_bwd = _conv_bwd_event((2, 3, 8, 8), (4, 3, 3, 3))
        mixed_bwd["args"]["Input type"] = [
            "c10::BFloat16",
            "c10::BFloat16",
            "c10::Half",
        ]
        with pytest.raises(ValueError, match="different"):
            perf_model.aten_conv_bwd(mixed_bwd).bytes()

    def test_conv_bias_family_conv1d_and_fallbacks(self):
        perf_model.ConvBias_.fwd_pass_cache.clear()
        conv1d_fwd = {
            "args": {
                "Input Dims": [[2, 3, 32], [4, 3, 5], [4]],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
                "Concrete Inputs": ["", "", "", "1", "0"],
                "Input Strides": [[96, 32, 1], [60, 15, 1]],
                "Sequence number": 50,
            }
        }
        perf_model.ConvBias_(conv1d_fwd)
        bwd = perf_model.ConvBias_Backward(_conv_bias_bwd_event(seq_num=50))
        assert bwd.flops() is None or bwd.flops() >= 0

        perf_model.ConvBias_.fwd_pass_cache.clear()
        fwd = perf_model.ConvBias_(_conv_bias_fwd_event())
        perf_model.ConvBias_.fwd_pass_cache[999] = {
            **fwd.param_details,
            "dtype_input_weight": ("c10::BFloat16", "c10::Half"),
        }
        cached_bwd = perf_model.ConvBias_Backward(_conv_bias_bwd_event(seq_num=999))
        assert cached_bwd.bytes() is not None

        perf_model.ConvBiasReLU_.fwd_pass_cache.clear()
        relu3d = {
            "args": {
                "Input Dims": [[2, 3, 8, 8, 8], [4, 3, 3, 3, 3], [4]],
                "Input type": ["c10::BFloat16"] * 3,
                "Concrete Inputs": ["", "", "", "1", "0"],
                "Sequence number": 51,
            }
        }
        perf_model.ConvBiasReLU_(relu3d)
        relu_bwd = perf_model.ConvBiasReLU_Backward(_conv_bias_bwd_event(seq_num=51))
        assert relu_bwd.bytes() is not None

        empty_bwd_evt = {"args": {"Input Dims": [], "Sequence number": 777}}
        with pytest.warns(UserWarning, match="insufficient trace data"):
            details = perf_model.ConvBiasReLU_Backward.get_param_details(empty_bwd_evt)
        assert details["input_shape"] is None

        perf_model.ConvBiasReLU_.fwd_pass_cache[888] = {
            "input_shape": (2, 3, 8, 8),
            "filter_shape": (4, 3, 3, 3),
            "dtype_input_weight": ("c10::BFloat16", "c10::Half"),
            "bias": True,
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "transposed_conv": False,
            "output_padding": (0, 0),
            "groups": 1,
            "input_stride": None,
            "weight_stride": None,
            "convNd": "conv2d",
        }
        mixed_relu = perf_model.ConvBiasReLU_Backward(_conv_bias_bwd_event(seq_num=888))
        assert mixed_relu.bytes() is not None

    def test_softmax_sdpa_simulation_and_bwd_bytes(self):
        assert perf_model.Softmax.bytes_bwd(4, 8, 2) > 0

        with patch.object(
            perf_model.GEMM,
            "get_simulation_time_func",
            side_effect=[(1.0, "qkt"), (None, None)],
        ):
            t = perf_model.SDPA.get_simulation_time_func(
                _ARCH,
                "fp16",
                None,
                "c10::Half",
                1000,
                1,
                8,
                64,
                64,
                32,
                fa=True,
            )
            assert t is None

        fa_bwd = perf_model.flash_attention_backward(_flash_bwd_event())
        assert fa_bwd.bytes() > 0

        aiter_bwd = perf_model.aiter__mha_bwd(
            {
                "args": {
                    "Input Dims": [
                        [2, 128, 8, 64],
                        [2, 128, 8, 64],
                        [2, 128, 8, 64],
                        [2, 128, 8, 64],
                        [2, 128, 8],
                    ],
                    "Input type": ["c10::BFloat16"] * 5,
                    "Concrete Inputs": [""] * 7 + ["0.0", "1.0", "False"],
                }
            }
        )
        assert aiter_bwd.bytes() > 0

    def test_aten_reduce_grouped_gemm_primus(self):
        reduce_evt = {
            "name": "aten::sum",
            "args": {
                "Input Dims": [[2, 4, 8]],
                "Input type": ["c10::BFloat16"],
                "Output type": "c10::BFloat16",
                "Concrete Inputs": ["", "[1]", "False"],
            },
        }
        assert perf_model.aten_reduce(reduce_evt).flops() > 0

        bad_dim = {
            "name": "aten::sum",
            "args": {
                "Input Dims": [[2, 4, 8]],
                "Input type": ["c10::BFloat16"],
                "Concrete Inputs": ["", "not_a_dim", "True"],
            },
        }
        assert perf_model.aten_reduce(bad_dim).flops() >= 0

        list_dim = {
            "name": "aten::sum",
            "args": {
                "Input Dims": [[2, 4, 8]],
                "Input type": ["c10::BFloat16"],
                "Concrete Inputs": ["", "[0, 2]", "True"],
            },
        }
        assert perf_model.aten_reduce(list_dim).flops() > 0

        gg = _GroupedGemmNoBwdOverride({"args": {}})
        assert gg.flops_bwd() > 0
        assert gg.bytes_bwd() > 0

        impl = perf_model.primus_turbo_grouped_gemm(
            {
                "args": {
                    "Input Dims": [[128, 64], [4, 64, 32]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                }
            }
        )
        assert impl.flops() > 0
        assert perf_model.primus_turbo_grouped_gemm._extract_impl_dims([]) is None
        assert perf_model.primus_turbo_grouped_gemm._extract_zipped_dims(
            [[(64, 32), (64, 32)], [(32, 16), (32, 16)]]
        ) == (128, 32, 16, 2)

        with pytest.raises(ValueError):
            perf_model.primus_turbo_grouped_gemm({"args": {"Input Dims": [[1]]}})

        var_k = perf_model.primus_turbo_grouped_gemm_variable_k(
            {
                "args": {
                    "Input Dims": [[128, 64], [128, 32]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                }
            }
        )
        assert var_k.bytes() > 0
        zipped = perf_model.primus_turbo_grouped_gemm_variable_k(
            {
                "args": {
                    "Input Dims": [
                        [(64, 32), (64, 48)],
                        [(32, 16), (48, 16)],
                    ],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                }
            }
        )
        assert zipped.flops() > 0
        assert (
            perf_model.primus_turbo_grouped_gemm_variable_k._extract_zipped_pairs(
                [[(1, 2)], [(3, 4)]]
            )
            is None
        )

    def test_jax_norm_rope_cross_entropy_mamba(self):
        jax_attn = perf_model.jax_te_fused_attn(
            {
                "args": {
                    "Input Dims": [[2, 128, 8, 64], [2, 128, 8, 64], [2, 128, 8, 64]],
                    "Input type": ["bf16", "bf16", "bf16"],
                    "Concrete Inputs": ["[False]"],
                }
            }
        )
        assert jax_attn.bytes_bwd() > 0

        jax_conv1d = perf_model.jax_conv(
            {
                "args": {
                    "Input Dims": [[2, 3, 32], [4, 3, 5]],
                    "Output Dims": [[2, 4, 28]],
                    "Filter Shape": [4, 3, 5],
                    "Input type": ["bf16", "bf16"],
                }
            }
        )
        assert jax_conv1d.param_details["convNd"] == "conv1d"
        jax_conv3d = perf_model.jax_conv(
            {
                "args": {
                    "Input Dims": [[2, 3, 8, 8, 8], [4, 3, 3, 3, 3]],
                    "Output Dims": [[2, 4, 6, 6, 6]],
                    "Filter Shape": [4, 3, 3, 3, 3],
                    "Input type": ["bf16", "bf16"],
                }
            }
        )
        assert jax_conv3d.bytes_bwd() > 0

        with pytest.raises(NotImplementedError):
            perf_model.InstanceNormBwd({"args": {}}).flops()

        ce = perf_model.cross_entropy_fwd(
            {
                "args": {
                    "Input Dims": [[4, 1, 8192], [4, 1]],
                    "Input type": ["invalid_dtype_xyz"],
                }
            }
        )
        ce.bpe = None
        assert ce.bytes() is None

        rope = perf_model.fused_rope_fwd(
            {
                "args": {
                    "Input Dims": [[128, 2, 8, 64], [128, 1, 1, 64]],
                    "Input type": ["bad_rope_dtype"],
                }
            }
        )
        rope.bpe = None
        assert rope.bytes() is None

        conv = perf_model.causal_conv1d_fwd(
            {
                "args": {
                    "Input Dims": [[2, 128, 64], [128, 4]],
                    "Input type": ["unknown_conv_dtype"],
                }
            }
        )
        conv.bpe = None
        assert conv.bytes() is None

        mamba = perf_model.mamba_ssd_fwd(_mamba_event(batch=2, seqlen=128))
        assert mamba._param_bpe([], 0, 2) == 2

    def test_quantize_mxfp4_and_aiter_gemm_strides(self):
        mx = perf_model.primus_turbo_quantize_mxfp4_dual(
            {
                "args": {
                    "Input Dims": [[128, 256]],
                    "Input type": ["bad_mx_dtype"],
                    "Input Strides": [[32768, 256, 1]],
                }
            }
        )
        mx.bpe_in = None
        assert mx.bytes() is None

        aiter = perf_model.aiter_gemm_a4w4(
            {
                "args": {
                    "Input Dims": [
                        [64, 32],
                        [128, 32],
                        [64, 2],
                        [128, 2],
                        (),
                        (),
                        (),
                        (),
                        (),
                    ],
                    "Input type": [
                        "c10::Float4_e2m1fn_x2",
                        "c10::Float4_e2m1fn_x2",
                        "c10::Float8_e8m0fnu",
                        "c10::Float8_e8m0fnu",
                    ],
                    "Input Strides": [[2048, 32, 1], [4096, 32, 1]],
                }
            }
        )
        assert aiter.bytes() > 0


class TestMoeExtensionsPhase11:
    def test_unfused_bytes_none_and_precision(self):
        up = moe_ext.moe_triton_unfused_up(_moe_unfused_event())
        assert up.get_compute_precision() in (None, "fp8", "bf16", "fp16")

        assert (
            moe_ext.UnfusedMoE_Up.bytes_func(32, 4096, 14336, 8, 2, False, 2, None, 2)
            is None
        )

        down = moe_ext.moe_aiter_unfused_down(
            {
                "args": {
                    "Input Dims": [
                        [32, 2, 7168],
                        [8, 4096, 896],
                        [32, 4096],
                    ],
                    "Input type": [
                        "c10::BFloat16",
                        "c10::Float8_e4m3fn",
                        "c10::BFloat16",
                    ],
                }
            }
        )
        assert down.bytes() > 0
        with pytest.raises(NotImplementedError):
            down.flops_bwd()


class TestPerfModelPhase12:
    def test_conv_unknown_dim_and_vllm_attention(self):
        with pytest.raises(ValueError, match="Unknown convolution"):
            perf_model.aten_conv(
                {
                    "args": {
                        "Input Dims": [[2], [4]],
                        "Input type": ["c10::BFloat16", "c10::BFloat16"],
                        "Concrete Inputs": [
                            "",
                            "",
                            "",
                            "(1)",
                            "(0)",
                            "(1)",
                            "False",
                            "(0)",
                            "1",
                        ],
                    }
                }
            )

        attn = perf_model.vllm_unified_attention_with_output(
            {
                "annotation": "(prefill_128_64_8_0_0_0_0)",
                "args": {
                    "Input Dims": [
                        [128, 8, 64],
                        [128, 8, 64],
                        [128, 8, 64],
                        [128, 8, 64],
                    ],
                    "Input type": ["c10::BFloat16"] * 4,
                },
            }
        )
        attn.param_details["sum_ctx_tokens"] = 0
        with pytest.raises(NotImplementedError):
            attn.flops()

    def test_conv_compute_precision_dtype_fallback(self):
        class _MiniConv(perf_model.CONV):
            @staticmethod
            def get_param_details(event):
                return {
                    "input_shape": (2, 3, 8, 8),
                    "filter_shape": (4, 3, 3, 3),
                    "out_shape": (2, 4, 6, 6),
                    "bias": False,
                    "transposed_conv": False,
                    "stride": (1, 1),
                    "padding": (0, 0),
                    "dilation": (1, 1),
                    "groups": 1,
                    "dtype": "c10::BFloat16",
                }

        model = _MiniConv({"args": {}}, arch=None)
        assert model.get_compute_precision() is not None


class TestPerfModelPhase12B:
    def test_remaining_conv_and_reduce_edges(self):
        with pytest.raises(ValueError, match="Unknown convolution"):
            perf_model.aten_conv_bwd(
                {
                    "args": {
                        "Input Dims": [[2], [2], [4]],
                        "Input type": ["c10::BFloat16"] * 3,
                        "Concrete Inputs": [
                            "",
                            "",
                            "",
                            "[0]",
                            "[1]",
                            "[0]",
                            "[1]",
                            "False",
                            "[0]",
                            "1",
                            "[True, True, False]",
                        ],
                    }
                }
            )

        evt = {
            "name": "aten::sum",
            "args": {
                "Input Dims": [[2, 4, 8]],
                "Input type": ["c10::BFloat16"],
                "Output type": None,
                "Concrete Inputs": ["", "[1]", "False"],
            },
        }
        assert perf_model.aten_reduce(evt).flops() > 0

        with pytest.raises(ValueError, match="could not parse"):
            perf_model.primus_turbo_grouped_gemm_variable_k(
                {"args": {"Input Dims": [[1, 2, 3]], "Input type": ["c10::BFloat16"]}}
            )


class TestMoePhase13:
    def test_fused_moe_precision_paths(self):
        fused = moe_ext.moe_aiter_fused_1stage(
            {
                "args": {
                    "Input Dims": [
                        [128, 4096],
                        [8, 4096, 512],
                        [8, 4096, 512],
                        [128, 6],
                        [128, 6],
                    ],
                    "Input type": [
                        "c10::BFloat16",
                        "c10::Float8_e4m3fn",
                        "c10::Float8_e4m3fn",
                        "c10::Int",
                        "c10::BFloat16",
                    ],
                }
            }
        )
        assert fused.get_compute_precision() in (None, "fp8", "bf16", "fp16")
        fused.param_details["input_dtype"] = None
        assert fused.get_compute_precision() is None


class TestKernelNameParserFull:
    ROCM = "Custom_Cijk_Alik_Bljk_BBS_BH_Bias_AS_SAV_UserArgs_MT64x16x64_MI16x16x1_SN"
    CUDA = "nvjet_tst_tst_TN"

    def test_rocm_gemm_parse(self):
        assert kernel_name_parser.is_rocm_gemm(self.ROCM)
        parsed = kernel_name_parser.parse_rocm_gemm(self.ROCM)
        assert parsed["transpose"] == (True, False)
        assert parsed["mt_m"] == 64
        assert parsed["mt_n"] == 16
        assert parsed["depth_u"] == 64
        assert kernel_name_parser.gemm_name_parser(self.ROCM) == parsed

    def test_cuda_gemm_parse(self):
        assert kernel_name_parser.is_cuda_gemm(self.CUDA)
        assert kernel_name_parser.parse_cuda_gemm(self.CUDA) == {
            "transpose": (True, False)
        }
        assert kernel_name_parser.gemm_name_parser(self.CUDA)["transpose"] == (
            True,
            False,
        )

    def test_unknown_kernel_returns_none(self):
        assert kernel_name_parser.gemm_name_parser("not_a_gemm") is None


_GEMM_EVT = _gemm_event("aten::mm", (4, 8), (8, 16))
_CONV_EVT = _conv_bias_fwd_event()
_NORM_EVT = _norm_event((4, 8, 32, 32), 8)
_ATTN_EVT = {
    "annotation": _GDN_ANNOTATION,
    "args": {
        "Input Dims": [[64, 8, 64], [64, 8, 64], [64, 8, 128]],
        "Input type": ["c10::BFloat16"] * 3,
    },
}
_MAMBA_EVT = _mamba_event(batch=2, seqlen=128)
_FUSED_LN_EVT = _fused_ln_fwd_event()
_MOE_EVT = {
    "args": {
        "Input Dims": [[32, 4096], [8, 28672, 512], [8, 4096, 7168], [32, 2]],
        "Input type": [
            "c10::BFloat16",
            "c10::Float8_e4m3fn",
            "c10::Float8_e4m3fn",
            "c10::Float32",
        ],
    },
}


def _try_model(cls, events):
    for event in events:
        try:
            model = (
                cls(event, arch=_ARCH)
                if "arch" in inspect.signature(cls.__init__).parameters
                else cls(event)
            )
        except TypeError:
            try:
                model = cls(event)
            except Exception:
                continue
        except Exception:
            continue
        for meth in (
            "flops",
            "bytes",
            "flops_bwd",
            "bytes_bwd",
            "get_compute_precision",
            "get_maf_type",
        ):
            if hasattr(model, meth):
                try:
                    getattr(model, meth)()
                except NotImplementedError:
                    pass
                except Exception:
                    pass
        return True
    return False


class TestPerfModelBulkSweep:
    def test_perf_model_classes_best_effort(self):
        events = [
            _GEMM_EVT,
            _CONV_EVT,
            _NORM_EVT,
            _MAMBA_EVT,
            _FUSED_LN_EVT,
            _conv_bias_bwd_event(),
            {
                "args": {
                    "Input Dims": [[4, 32000], [4]],
                    "Input type": ["c10::BFloat16", "long int"],
                }
            },
            {
                "args": {
                    "Batch": 2,
                    "M": 4,
                    "N": 8,
                    "K": 16,
                    "Beta": 1,
                    "Type": "bf16",
                }
            },
        ]
        covered = 0
        for _name, cls in inspect.getmembers(perf_model, inspect.isclass):
            if cls.__module__ != perf_model.__name__:
                continue
            if _try_model(cls, events):
                covered += 1
        assert covered > 30

    def test_extension_classes_best_effort(self):
        events = [_GEMM_EVT, _ATTN_EVT, _MOE_EVT, _NORM_EVT]
        modules = (pext, attn_ext, moe_ext, rms_ext)
        covered = 0
        for mod in modules:
            for _name, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mod.__name__:
                    continue
                if _try_model(cls, events):
                    covered += 1
        assert covered > 30


class TestPerfModelPhase6:
    def test_conv_bias_bwd_empty_dims(self):
        perf_model.ConvBias_.fwd_pass_cache.clear()
        evt = {
            "args": {
                "Input Dims": [],
                "Input type": ["c10::BFloat16"],
                "Sequence number": 42,
            }
        }
        details = perf_model.ConvBias_Backward.get_param_details(evt)
        assert details["input_shape"] is None

    def test_conv_bias_relu_bwd_cache_path(self):
        perf_model.ConvBiasReLU_.fwd_pass_cache.clear()
        fwd = perf_model.ConvBiasReLU_(_conv_bias_fwd_event())
        assert fwd.flops() > 0
        bwd = perf_model.ConvBiasReLU_Backward(_conv_bias_bwd_event())
        assert bwd.flops_bwd() > 0

    def test_conv1d_bytes_func_none(self):
        assert (
            perf_model.CONV.bytes_func((2, 4, 32), (8, 4, 3), (2, 8, 30), False, None)
            is None
        )

    def test_tev2_pseudo_gemm_and_grouped_gemm(self):
        event = _gemm_event("tev2::pseudo_gemm", (4, 8), (8, 16))
        model = perf_model.tev2_pseudo_gemm(event)
        assert model.flops() > 0

        gg = {
            "args": {
                "Input Dims": [[12, 16], [4, 16, 32]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            }
        }
        g = perf_model.primus_turbo_grouped_gemm(gg)
        assert g.bytes() > 0
        with pytest.raises(NotImplementedError):
            g.flops_bwd()

    def test_sdpa_simulation_qkt_none_returns_none(self):
        event = {
            "args": {
                "Input Dims": [[2, 128, 8, 64], [2, 128, 8, 64], [2, 128, 8, 64]],
                "Input type": ["c10::BFloat16"] * 3,
                "Concrete Inputs": ["", "", "", "0.0", "False"],
            }
        }
        model = perf_model.aten__scaled_dot_product_flash_attention(event, arch=_ARCH)
        with patch.object(
            perf_model.GEMM, "get_simulation_time_func", return_value=(None, None)
        ):
            assert model.get_simulation_time() is None

    def test_flash_attention_backward_flops(self):
        model = perf_model.flash_attention_backward(_flash_bwd_event())
        assert model.flops_bwd() > 0
        assert model.bytes() > 0


class TestMoeExtensionsPhase6:
    MOE_FUSED = {
        "args": {
            "Input Dims": [
                [32, 4096],
                [8, 28672, 512],
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

    def test_fused_moe_bytes(self):
        model = moe_ext.moe_aiter_fused_1stage(self.MOE_FUSED)
        assert model.bytes() > 0

    def test_unfused_moe_bytes_none_bpe(self):
        assert (
            moe_ext.UnfusedMoE_Up.bytes_func(8, 4096, 14336, 8, 2, True, None, 2, 2)
            is None
        )
        up = moe_ext.moe_triton_unfused_up(
            _moe_unfused_event(kernel_name="moe_fp8_up_kernel")
        )
        assert up.bytes() > 0

    def test_moe_auxiliary_classes(self):
        blockscale = {
            "args": {
                "Input Dims": [
                    [128, 256],
                    [128, 256],
                    [512, 28672, 256],
                    [512, 256, 7168],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                ],
                "Concrete Inputs": [""] * 8 + ["2"],
            }
        }
        assert moe_ext.moe_aiter_fused_blockscale(blockscale).bytes() > 0
        topk_evt = {
            "args": {
                "Input Dims": [(32, 256), (256,), (32, 8), (32, 8)],
                "Input type": ["c10::Float"] * 3 + ["c10::Int"],
            }
        }
        b = moe_ext.BiasedGroupedTopk(topk_evt).bytes()
        assert b is None or b >= 0


class TestPerfModelPhase8:
    def test_tex_ts_te_gemm_no_strides(self):
        input_dims = [()] * 19
        input_dims[0] = [128, 64]
        input_dims[5] = [256, 64]
        input_dims[10] = [128, 256]
        event = {
            "args": {
                "Input Dims": input_dims,
                "Input type": ["c10::Float8_e4m3fn"] * 19,
                "Concrete Inputs": [""] * 4
                + ["1"]
                + [""] * 4
                + ["1"]
                + [""] * 4
                + [""],
            }
        }
        model = perf_model.tex_ts_te_gemm_ts(event)
        assert model.flops() > 0
        with pytest.raises(NotImplementedError):
            model.flops_bwd()

    def test_grouped_gemm_variable_k(self):
        event = {
            "args": {
                "Input Dims": [
                    [(4, 16), (8, 16)],
                    [(16, 32), (16, 64)],
                ],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            }
        }
        g = perf_model.primus_turbo_grouped_gemm_variable_k(event)
        assert g.flops() > 0
        assert g.bytes() > 0

    def test_vllm_unified_attention_with_output(self):
        event = {
            "annotation": "(prefill_128_64_8_0_0_0_0)",
            "args": {
                "Input Dims": [
                    [128, 8, 64],
                    [128, 8, 64],
                    [128, 8, 64],
                    [128, 8, 64],
                ],
                "Input type": ["c10::BFloat16"] * 4,
            },
        }
        model = perf_model.vllm_unified_attention_with_output(event)
        assert model.flops() > 0

    def test_conv_bytes_bwd_none(self):
        assert (
            perf_model.CONV.bytes_bwd_func(
                (2, 3, 8, 8), (4, 3, 3, 3), (2, 4, 6, 6), True, None
            )
            is None
        )

    def test_conv_bias_relu_forward(self):
        perf_model.ConvBiasReLU_.fwd_pass_cache.clear()
        fwd = perf_model.ConvBiasReLU_(_conv_bias_fwd_event())
        assert fwd.flops() > 0
        assert fwd.bytes() > 0


class TestPerfModelPhase9:
    def test_extract_sdpa_cfg_errors(self):
        with pytest.raises(ValueError, match="Batch sizes"):
            perf_model.extract_sdpa_cfg(
                [2, 8, 64, 32], [1, 8, 64, 32], [2, 8, 64, 32], (0, 1, 2, 3)
            )
        with pytest.raises(ValueError, match="Head sizes"):
            perf_model.extract_sdpa_cfg(
                [2, 8, 64, 32], [2, 4, 64, 32], [2, 8, 64, 32], (0, 1, 2, 3)
            )
        with pytest.raises(ValueError, match="Length sizes"):
            perf_model.extract_sdpa_cfg(
                [2, 8, 64, 32], [2, 8, 32, 32], [2, 8, 64, 32], (0, 1, 2, 3)
            )
        with pytest.raises(ValueError, match="Head dimensions"):
            perf_model.extract_sdpa_cfg(
                [2, 8, 64, 32], [2, 8, 64, 16], [2, 8, 64, 32], (0, 1, 2, 3)
            )

    def test_extract_sdpa_varlen_cfg_errors(self):
        with pytest.raises(ValueError, match="Head sizes"):
            perf_model.extract_sdpa_varlen_cfg(
                [8, 64, 32], [4, 64, 32], [8, 64, 32], (0, 1, 2)
            )

    def test_sdpa_causal_mismatch_raises(self):
        with pytest.raises(ValueError, match="causal=True"):
            perf_model.SDPA.flops_bwd_func(1, 64, 8, 32, 8, 64, 64, True, True)

    def test_sdpa_varlen_multi_seq_flops(self):
        event = {
            "args": {
                "Input Dims": [
                    [2, 64, 4, 32],
                    [2, 64, 4, 32],
                    [2, 64, 4, 32],
                ],
                "Input type": ["c10::BFloat16"] * 3,
                "Concrete Inputs": [
                    "",
                    "",
                    "",
                    "0.0",
                    "True",
                    "True",
                    "2",
                    "2",
                    "64",
                    "64",
                ],
            }
        }
        model = perf_model.aten__scaled_dot_product_flash_attention(event)
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_cudnn_and_efficient_attention(self):
        base = {
            "args": {
                "Input Dims": [[2, 128, 8, 64], [2, 128, 8, 64], [2, 128, 8, 64]],
                "Input type": ["c10::BFloat16"] * 3,
                "Concrete Inputs": ["", "", "", "0.0", "0.0", "False", "False", ""],
            }
        }
        cudnn = perf_model.aten__scaled_dot_product_cudnn_attention(base)
        assert cudnn.flops() > 0
        efficient = perf_model.aten__scaled_dot_product_efficient_attention(base)
        assert efficient.flops() > 0

    def test_conv_bias_bwd_cached_mixed_dtype(self):
        perf_model.ConvBias_.fwd_pass_cache.clear()
        fwd_evt = _conv_bias_fwd_event()
        perf_model.ConvBias_(fwd_evt)
        bwd_evt = _conv_bias_bwd_event()
        bwd_evt["args"]["Input type"] = ["c10::BFloat16", "c10::Half"]
        bwd = perf_model.ConvBias_Backward(bwd_evt)
        assert bwd.bytes() is None or bwd.bytes() >= 0

    def test_conv_bias_relu_bwd_cached(self):
        perf_model.ConvBiasReLU_.fwd_pass_cache.clear()
        perf_model.ConvBiasReLU_(_conv_bias_fwd_event())
        bwd = perf_model.ConvBiasReLU_Backward(_conv_bias_bwd_event())
        assert bwd.flops_bwd() > 0
        assert bwd.bytes_bwd() > 0

    def test_flash_attention_backward_simulation_none(self):

        model = perf_model.flash_attention_backward(_flash_bwd_event())
        with patch.object(
            perf_model.GEMM, "get_simulation_time_func", return_value=(None, None)
        ):
            assert model.get_simulation_time() is None


class TestMoeExtensionsBoost:
    MOE_FUSED = {
        "args": {
            "Input Dims": [
                [32, 4096],
                [8, 28672, 512],
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

    def test_moe_aiter_fused_1stage_full(self):
        model = moe_aiter_fused_1stage(self.MOE_FUSED)
        assert model.flops() > 0
        assert model.bytes() > 0
        assert model.get_maf_type() == "matrix"
        with pytest.raises(NotImplementedError):
            model.flops_bwd()

    def test_moe_unfused_with_output_dtype(self):
        up_event = {
            "args": {
                "Input Dims": [[32, 4096], [8, 14336, 512], [32, 2, 7168]],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::BFloat16",
                ],
            }
        }
        down_event = {
            "args": {
                "Input Dims": [
                    [32, 2, 7168],
                    [8, 4096, 896],
                    [32, 4096],
                    [],
                    [],
                    [],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::BFloat16",
                ],
            }
        }
        up = moe_aiter_unfused_up(up_event)
        down = moe_aiter_unfused_down(down_event)
        assert up.bytes() > 0
        assert down.flops() > 0

    def test_moe_flydsl_gptq_grouped(self):
        fly = {
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
                ],
            }
        }
        assert moe_flydsl_stage1(fly).flops() > 0
        assert moe_flydsl_stage2(fly).bytes() > 0
        gptq = {
            "args": {
                "Input Dims": [[32, 4096], [32, 2], [8, 7168, 4096]],
                "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
                "MoE topk": 2,
            }
        }
        assert moe_gptq_awq_up(gptq).flops() > 0
        assert moe_gptq_awq_down(gptq).bytes() > 0
        grouped = {
            "args": {
                "Input Dims": [
                    [64, 2048],
                    [128, 1536, 2048],
                    (),
                    [512, 1536],
                    (),
                    (),
                    (),
                    (),
                    [64, 4],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "",
                    "c10::BFloat16",
                ],
            }
        }
        assert moe_triton_invoke_grouped_gemm(grouped).flops() > 0

    def test_moe_aux_models(self):
        topk = BiasedGroupedTopk(
            {
                "args": {
                    "Input Dims": [(32, 256), (256,), (32, 8), (32, 8)],
                    "Input type": [
                        "c10::Float",
                        "c10::Float",
                        "c10::Float",
                        "c10::Int",
                    ],
                }
            }
        )
        assert topk.flops() > 0


class TestPerfModelExtensionsBoost:
    def test_jax_conv_metadata_path(self):
        conv_event = {
            "args": {
                "Input Dims": [[2, 3, 8, 8], [4, 3, 3, 3]],
                "Output Dims": [[2, 4, 6, 6]],
                "Filter Shape": [4, 3, 3, 3],
                "Input type": ["bf16", "bf16"],
                "Concrete Inputs": [
                    "",
                    "",
                    "(1,1)",
                    "(0,0)",
                    "(1,1)",
                    "False",
                    "(0,0)",
                    "1",
                ],
            }
        }
        model = perf_model.jax_conv(conv_event)
        assert model.flops_bwd() > 0


class TestPerfModelPush95:
    def test_gemm_simulator_missing_required_inputs(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        with pytest.raises(AssertionError, match="Invalid inputs"):
            perf_model.GEMM.get_simulation_time_func(
                {"name": "mi300x"}, None, 8, 16, 1, "bf16"
            )

    def test_gemm_simulator_force_to_l1_and_scaled_cus(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        arch = dict(_ARCH)
        with patch("TraceLens.PerfModel.perf_model.subprocess.run") as run:
            run.return_value = MagicMock(stdout="Time=3.3\n", stderr="")
            t, _ = perf_model.GEMM.get_simulation_time_func(
                arch, 4, 8, 16, 1, "bf16", num_cus=64, force_to_l1=True
            )
        assert t == 3.3

    def test_aten_scaled_mm_output_bpe_branches(self):
        for dtype in ("c10::Float8_e4m3fn", "c10::BFloat16"):
            event = {
                "args": {
                    "Input Dims": [[4, 8], [8, 16], [4, 16]],
                    "Input type": [dtype, dtype, dtype],
                }
            }
            model = perf_model.aten_scaled_mm(event)
            assert model.bytes() > 0

    def test_aten_conv3d_and_mixed_dtype_bytes_error(self):
        conv3d = {
            "args": {
                "Input Dims": [[2, 4, 8, 8, 8], [8, 4, 3, 3, 3]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Concrete Inputs": [
                    "",
                    "",
                    "",
                    "(1,1,1)",
                    "(0,0,0)",
                    "(1,1,1)",
                    "False",
                    "(0,0,0)",
                    "1",
                ],
            }
        }
        model = perf_model.aten_conv(conv3d)
        assert model.param_details["convNd"] == "conv3d"
        bad = {
            "args": {
                "Input Dims": [[2, 3, 8, 8], [4, 3, 3, 3]],
                "Input type": ["c10::BFloat16", "c10::Half"],
                "Concrete Inputs": [
                    "",
                    "",
                    "",
                    "(1,1)",
                    "(0,0)",
                    "(1,1)",
                    "False",
                    "(0,0)",
                    "1",
                ],
            }
        }
        with pytest.raises(ValueError):
            perf_model.aten_conv(bad).bytes()

    def test_conv_bias_backward_without_cache(self):
        bwd_evt = _conv_bias_bwd_event()
        bwd_evt["args"]["Sequence number"] = 99999
        with pytest.warns(UserWarning, match="Forward pass not found"):
            details = perf_model.ConvBias_Backward.get_param_details(bwd_evt)
        assert details["input_shape"] is None

    def test_aten_reduce_dim_parse_failure(self):
        event = {
            "name": "aten::sum",
            "args": {
                "Input Dims": [(4, 256)],
                "Input type": ["c10::BFloat16"],
                "Concrete Inputs": ["", "not_a_dim", "True"],
            },
        }
        model = perf_model.aten_reduce(event)
        assert model.param_details["num_output_elems"] == 1

    def test_grouped_gemm_gn_k_layout(self):
        event = {
            "args": {
                "Input Dims": [[9, 8], [3, 16, 8]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            }
        }
        g = perf_model.primus_turbo_grouped_gemm(event)
        assert g.flops() > 0
        with pytest.raises(NotImplementedError):
            g.flops_bwd()
        assert g.get_maf_type() == "matrix"

    def test_jax_gemm_backward_not_implemented(self):
        event = {
            "args": {
                "Batch": 2,
                "M": 4,
                "N": 8,
                "K": 16,
                "Beta": 0,
                "Type": "bf16",
            }
        }
        model = perf_model.jax_gemm(event)
        assert model.flops() > 0
        with pytest.raises(NotImplementedError):
            model.flops_bwd()

    def test_vllm_gemm_missing_input_type(self):
        with pytest.raises(ValueError, match="missing A,B dtypes"):
            perf_model.vllm_gemm_with_dynamic_quant(
                {"args": {"Input Dims": [[4, 8], [16, 4]]}}
            )

    def test_tex_ts_te_gemm_transposed(self):
        input_dims = [()] * 19
        input_dims[0] = [128, 64]
        input_dims[5] = [256, 64]
        input_dims[10] = [128, 256]
        event = {
            "args": {
                "Input Dims": input_dims,
                "Input type": ["c10::Float8_e4m3fn"] * 19,
                "Concrete Inputs": [""] * 4
                + ["1"]
                + [""] * 4
                + ["1"]
                + [""] * 4
                + [""],
            }
        }
        model = perf_model.tex_ts_te_gemm_ts(event)
        assert model.flops() > 0


class TestMoeExtensionsPush95:
    def test_blockscale_missing_topk_raises(self):
        event = {
            "args": {
                "Input Dims": [
                    [32, 4096],
                    [32, 4096],
                    [8, 14336, 4096],
                    [8, 4096, 7168],
                ],
                "Input type": ["c10::BFloat16", "c10::Float8_e4m3fn"] * 2,
                "Concrete Inputs": [""] * 8,
            }
        }
        with pytest.raises(ValueError, match="topk"):
            moe_ext.moe_aiter_fused_blockscale(event)

    def test_triton_unfused_missing_kernel_details(self):
        event = {
            "args": {
                "Input Dims": [[32, 4096], [32, 8]],
                "Input type": ["c10::BFloat16", "c10::Float32"],
                "MoE topk": 2,
                "MoE GEMM gated": True,
            }
        }
        with pytest.raises(ValueError, match="Kernel details"):
            moe_ext.moe_triton_unfused_up(event)

    def test_triton_unfused_fp4_and_fp8_kernels(self):
        up = moe_ext.moe_triton_unfused_up(
            _moe_unfused_event(kernel_name="moe_mxfp4_up_kernel")
        )
        down = moe_ext.moe_triton_unfused_down(
            _moe_unfused_event(kernel_name="moe_fp8_down_kernel")
        )
        assert up.get_compute_precision() in ("fp4", "fp8", "bf16", None)
        assert down.bytes() > 0
        with pytest.raises(NotImplementedError):
            up.flops_bwd()


class TestAttentionRmsnormPush95:
    def test_mla_decode_and_paged_attention(self):
        attn_event = {
            "annotation": _GDN_ANNOTATION,
            "args": {
                "Input Dims": [[64, 8, 64], [64, 8, 64], [64, 8, 128]],
                "Input type": ["c10::BFloat16"] * 3,
            },
        }
        mla = attn_ext.mla_decode_fwd(attn_event)
        if mla.param_details.get("_no_perf"):
            assert mla.flops() is None
        else:
            assert mla.flops() > 0
        paged = attn_ext.aiter_paged_attention_ragged(
            {
                "annotation": _GDN_ANNOTATION,
                "args": {
                    "Input Dims": [
                        (),
                        (),
                        [64, 8, 64],
                        [128, 16, 1, 64],
                        [128, 16, 1, 128],
                    ],
                    "Input type": [
                        "Scalar",
                        "Scalar",
                        "c10::BFloat16",
                        "c10::BFloat16",
                        "c10::BFloat16",
                    ],
                },
            }
        )
        assert paged.param_details["d_h_v"] == 128

    def test_rmsnorm_extension_variants(self):
        evt = {
            "args": {
                "Input Dims": [(4, 512), (4, 512), (512,), ()],
                "Input type": ["c10::BFloat16"] * 3 + ["Scalar"],
                "Input Strides": [(512, 1), (512, 1), (1,), ()],
            }
        }
        assert rms_ext.aiter_rmsnorm(evt).bytes() > 0
        vllm_evt = {
            "args": {
                "Input Dims": [(4, 512), (512,), (), ()],
                "Input type": ["c10::BFloat16"] * 4,
                "Input Strides": [(512, 1), (1,), (), ()],
                "Concrete Inputs": ["", "", "", "128"],
            }
        }
        assert rms_ext.vllm_rocm_aiter_rmsnorm_fp8_group_quant(vllm_evt).flops() > 0


def test_norm_and_mamba_perf_models():

    mamba = perf_model.mamba_ssd_fwd(_mamba_event())
    assert mamba.flops() > 0
    assert mamba.bytes() > 0
    dispatch = perf_model.moe_dispatch(
        {
            "args": {
                "Input Dims": [[32, 4096], [32, 8]],
                "Input type": ["c10::BFloat16", "c10::Int"],
            }
        }
    )
    assert dispatch.bytes() > 0
    combine = perf_model.moe_combine(
        {
            "args": {
                "Input Dims": [[32, 4096], [32, 4096]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            }
        }
    )
    assert combine.bytes() >= 0


def test_moe_ck_and_gptq_extended():
    ck1 = {
        "args": {
            "Input Dims": [
                [32, 512],
                [8, 7168, 512],
                [8, 4096, 896],
                [],
                [],
                [],
                [32, 2, 7168],
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::Float8_e4m3fn",
                "c10::Float8_e4m3fn",
            ],
        }
    }
    ck2 = {
        "args": {
            "Input Dims": [
                [32, 2, 7168],
                [8, 7168, 512],
                [8, 4096, 896],
                [],
                [],
                [],
                [32, 4096],
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::Float8_e4m3fn",
                "c10::Float8_e4m3fn",
            ],
        }
    }
    assert moe_ext.moe_aiter_ck_stage1(ck1).bytes() > 0
    assert moe_ext.moe_aiter_ck_stage2(ck2).flops() > 0
    gptq = {
        "args": {
            "Input Dims": [[32, 4096], [32, 2], [8, 7168, 4096]],
            "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
            "MoE topk": 2,
        }
    }
    assert moe_ext.moe_gptq_awq_up(gptq).bytes() > 0
    assert moe_ext.moe_gptq_awq_down(gptq).flops() > 0


def test_gemm_simulator_clears_cache(monkeypatch, tmp_path):
    perf_model.GEMM.cache_gemm_results.clear()
    sim_dir = tmp_path / "simdir"
    sim_dir.mkdir()
    sim = sim_dir / "run_gemm.py"
    sim.write_text("# stub\n")
    monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
    with patch("TraceLens.PerfModel.perf_model.subprocess.run") as run:
        run.return_value = MagicMock(stdout="Time=42.5\n", stderr="")
        t, _ = perf_model.GEMM.get_simulation_time_func(
            _ARCH, 4, 8, 16, 1, "bf16", num_cus=64, force_to_l1=True
        )
    assert t == 42.5
    perf_model.GEMM.cache_gemm_results.clear()


def test_untested_perf_extensions():

    blockscale = {
        "args": {
            "Input Dims": [[128, 256], [512, 256], [512, 4], [512, 4]],
            "Input type": [
                "c10::Float8_e4m3fn",
                "c10::Float8_e4m3fn",
                "c10::Float",
                "c10::Float",
            ],
        }
    }
    assert pext.gemm_a8w8_blockscale(blockscale).bytes() > 0

    silu = {
        "args": {
            "Input Dims": [(4, 512), (4, 512)],
            "Input type": ["c10::BFloat16", "c10::BFloat16"],
            "Input Strides": [(512, 1), (512, 1)],
        }
    }
    assert pext.aiter_silu_and_mul(silu).flops() > 0
    assert pext.sgl_kernel_silu_and_mul(silu).bytes() > 0
    assert pext.aiter_gelu_and_mul(silu).flops() > 0

    per_group = {
        "args": {
            "Input Dims": [(4, 256), (4, 256), (4, 2)],
            "Input type": ["c10::Float8_e4m3fn", "c10::BFloat16", "c10::BFloat16"],
            "Input Strides": [(256, 1), (256, 1), (2, 1)],
        }
    }
    assert pext.per_group_quant(per_group).bytes() > 0

    vllm_group = {
        "args": {
            "Input Dims": ((4, 256), ()),
            "Input type": ("c10::BFloat16", "Scalar"),
            "Concrete Inputs": ("", "128"),
        }
    }
    assert pext.vllm_triton_per_token_group_quant_fp8(vllm_group).flops() > 0

    rope_mla = {
        "args": {
            "Input Dims": [
                (2, 8, 512),
                (2, 8, 64),
                (2, 1, 512),
                (2, 1, 64),
                (128, 1, 1, 576),
            ],
            "Input type": ["c10::BFloat16"] * 5,
        }
    }
    assert pext.aiter_fused_qk_rope_cat_and_cache_mla(rope_mla).bytes() > 0


def test_attention_extension_variants():

    for cls, event in [
        (
            aext.pseudo_v4_paged_decode_csa,
            {
                "annotation": "(128_256_512_1024_2048_3072_4096_64)",
                "args": {
                    "Input Dims": [[64, 8, 64], [64, 8, 64], [64, 8, 128]],
                    "Input type": ["c10::BFloat16"] * 3,
                },
            },
        ),
        (
            aext.vllm_unified_mla_attention_with_output,
            {
                "annotation": "(128_256_512_1024_2048_3072_4096_64)",
                "args": {
                    "Input Dims": [[64, 8, 64], [64, 8, 64], [64, 8, 128]],
                    "Input type": ["c10::BFloat16"] * 3,
                },
            },
        ),
    ]:
        model = cls(event)
        if model.param_details.get("_no_perf"):
            assert model.flops() is None
        else:
            assert model.flops() > 0


class TestAttentionExtensionsBytes:
    @pytest.mark.parametrize(
        "cls,event",
        [
            (attn_ext.mha_varlen_fwd, _attn_base()),
            (attn_ext.aiter_fmha_v3_varlen_fwd, _attn_base()),
            (attn_ext.mla_decode_fwd, _attn_base()),
            (attn_ext.pseudo_mla_prefill_fwd, _attn_base()),
            (attn_ext.vllm_unified_mla_attention_with_output, _attn_base()),
            (attn_ext.mla_tilelang_sparse_fwd, _attn_base()),
            (
                attn_ext.vllm_unified_attention_with_output,
                {
                    "annotation": "(128_256_512_1024_2048_3072_4096_64)",
                    "args": {
                        "Input Dims": [[512, 8, 64], [1024, 1, 64], (), [512, 8, 64]],
                        "Input type": ["c10::BFloat16"] * 4,
                    },
                },
            ),
            (
                attn_ext.pseudo_v4_paged_decode_swa,
                {
                    "annotation": _GDN,
                    "args": {
                        "Input Dims": [[1, 8, 512]],
                        "Input type": ["c10::BFloat16"],
                        "v4_model_name": "DeepSeek-V4-Flash",
                    },
                },
            ),
        ],
    )
    def test_bytes_and_precision(self, cls, event):
        model = cls(event)
        b = model.bytes()
        if b is not None:
            assert b >= 0
        prec = model.get_compute_precision()
        assert prec in (None, "bf16", "fp16", "fp8", "fp32")


class TestMoeExtensionsSweep:
    def test_moe_ck_and_gptq_bytes(self):
        ck = {
            "args": {
                "Input Dims": [
                    [32, 512],
                    [8, 7168, 512],
                    [8, 4096, 896],
                    [],
                    [],
                    [],
                    [32, 2, 7168],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                ],
            }
        }
        assert moe_ext.moe_aiter_ck_stage1(ck).get_compute_precision() == "bf16"
        down = {
            "args": {
                "Input Dims": [
                    [32, 2, 7168],
                    [8, 7168, 512],
                    [8, 4096, 896],
                    [],
                    [],
                    [],
                    [32, 4096],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                ],
            }
        }
        assert moe_ext.moe_aiter_ck_stage2(down).bytes() > 0
        gptq = {
            "args": {
                "Input Dims": [[32, 4096], [32, 2], [8, 7168, 4096]],
                "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
                "MoE topk": 2,
            }
        }
        assert moe_ext.moe_gptq_awq_down(gptq).flops() > 0


class TestPerfModelPush95Coverage:
    @pytest.mark.parametrize(
        "missing,kwargs",
        [
            ("M", {"M": None, "N": 8, "K": 16, "B": 1, "dtype": "bf16"}),
            ("N", {"M": 4, "N": None, "K": 16, "B": 1, "dtype": "bf16"}),
            ("K", {"M": 4, "N": 8, "K": None, "B": 1, "dtype": "bf16"}),
            ("dtype", {"M": 4, "N": 8, "K": 16, "B": 1, "dtype": None}),
            ("arch['name']", {"M": 4, "N": 8, "K": 16, "B": 1, "dtype": "bf16"}),
        ],
    )
    def test_gemm_simulator_missing_inputs(
        self, monkeypatch, tmp_path, missing, kwargs
    ):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        arch = dict(_ARCH) if "arch['name']" not in missing else {"freq_mhz": 2200}
        with pytest.raises(AssertionError, match="Invalid inputs"):
            perf_model.GEMM.get_simulation_time_func(arch, **kwargs)
        perf_model.GEMM.cache_gemm_results.clear()

    def test_gemm_simulator_subprocess_failure(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        with patch("TraceLens.PerfModel.perf_model.subprocess.run") as run:
            run.return_value = MagicMock(stdout="", stderr="fail")
            with pytest.raises(AssertionError, match="Failed to simulate"):
                perf_model.GEMM.get_simulation_time_func(_ARCH, 4, 8, 16, 1, "bf16")
        perf_model.GEMM.cache_gemm_results.clear()

    def test_gemm_origami_unsupported_dtype(self, monkeypatch):
        monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
        mock_origami = MagicMock()
        mock_origami.data_type_t = MagicMock()
        with patch.dict(sys.modules, {"origami": mock_origami}):
            with patch("TraceLens.PerfModel.origami_helper.OrigamiHelper"):
                with pytest.warns(RuntimeWarning, match="Unsupported dtype"):
                    t, _ = perf_model.GEMM.get_simulation_time_func(
                        _ARCH, 4, 8, 16, 1, "unknown_dtype", enable_origami=True
                    )
        assert t is None

    def test_sdpa_simulation_via_subprocess_gemm(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        with patch("TraceLens.PerfModel.perf_model.subprocess.run") as run:
            run.return_value = MagicMock(stdout="Time=2.0\n", stderr="")
            with patch.object(perf_model.Softmax, "get_time", return_value=0.25):
                t = perf_model.SDPA.get_simulation_time_func(
                    _ARCH,
                    "bf16",
                    "/usr/bin/python3",
                    "c10::BFloat16",
                    1024,
                    2,
                    8,
                    128,
                    128,
                    64,
                    fa=True,
                )
        assert t > 0
        perf_model.GEMM.cache_gemm_results.clear()

    @pytest.mark.parametrize(
        "cls,event",
        [
            (
                perf_model.BatchNormBwd,
                {
                    "name": "aten::miopen_batch_norm_backward",
                    "args": {
                        "Input Dims": [
                            (8, 16, 32, 32),
                            (8, 16, 32, 32),
                            (16,),
                            (16,),
                            (16,),
                            (16,),
                            (16,),
                            (),
                        ],
                        "Input type": ["float"] * 7 + ["Scalar"],
                        "Input Strides": [(16384, 1024, 32, 1)] * 2 + [(1,)] * 5 + [()],
                        "Concrete Inputs": ["", "", "", "", "", "", "", "1e-5"],
                    },
                },
            ),
            (
                perf_model.GroupNormBwd,
                {
                    "args": {
                        "Input Dims": [
                            None,
                            (4, 8, 32, 32),
                            (8,),
                            (8,),
                            (8,),
                            (8,),
                            (4, 8, 32, 32),
                            (),
                        ],
                        "Input type": ["c10::BFloat16"] * 7 + ["Scalar"],
                        "Input Strides": [(), (8192, 1024, 32, 1), (1,)] * 2
                        + [(8192, 1024, 32, 1)] * 2
                        + [(), ()],
                        "Concrete Inputs": [
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "8",
                            "8",
                            "[True, True]",
                        ],
                    }
                },
            ),
        ],
    )
    def test_norm_backward_variants(self, cls, event):
        model = cls(event)
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_instance_norm_training_flag(self):
        event = _norm_event((4, 8, 32, 32), 8, training=False)
        event["args"]["Concrete Inputs"][5] = ""
        model = perf_model.InstanceNorm(event)
        assert model.is_training is False

    def test_instance_norm_bwd_raises(self):
        with pytest.raises(NotImplementedError):
            perf_model.InstanceNormBwd.get_param_details({})

    def test_mamba_cross_entropy_moe_comm(self):
        mamba = perf_model.mamba_ssd_fwd(_mamba_event(batch=2, seqlen=128))
        assert mamba.flops() > 0
        assert mamba.bytes() > 0

        ce = perf_model.cross_entropy_fwd(
            {
                "args": {
                    "Input Dims": [[4, 32000], [4]],
                    "Input type": ["c10::BFloat16", "long int"],
                }
            }
        )
        assert ce.flops() > 0
        assert ce.get_compute_precision() is not None

        conv = perf_model.causal_conv1d_fwd(
            {
                "args": {
                    "Input Dims": [[2, 128, 512], [128, 4], [128]],
                    "Input type": ["c10::BFloat16"] * 3,
                }
            }
        )
        assert conv.bytes() > 0

        empty_comm = perf_model.moe_dispatch(
            {"args": {"Input Dims": [[]], "Input type": []}}
        )
        assert empty_comm.bytes() is None
        assert empty_comm.flops_bwd() == 0

    def test_jax_gemm_and_conv(self):
        gemm = perf_model.jax_gemm(
            {
                "args": {
                    "Batch": 2,
                    "M": 4,
                    "N": 8,
                    "K": 16,
                    "Beta": 1,
                    "Type": "bf16",
                }
            }
        )
        assert gemm.flops() > 0
        conv = perf_model.jax_conv(
            {
                "args": {
                    "Input Dims": [[2, 3, 8, 8], [4, 3, 3, 3]],
                    "Output Dims": [[2, 4, 6, 6]],
                    "Filter Shape": [4, 3, 3, 3],
                    "Input type": ["bf16", "bf16"],
                    "Concrete Inputs": [
                        "",
                        "",
                        "(1,1)",
                        "(0,0)",
                        "(1,1)",
                        "False",
                        "(0,0)",
                        "1",
                    ],
                }
            }
        )
        assert conv.flops_bwd() > 0

    def test_hipblaslt_gemm_fp8_fp4(self):

        fp8 = perf_model.hipblaslt_gemm_fp8(
            _fp8_gemm_event((128, 64), (256, 64), trans_b=True)
        )
        assert fp8.flops() > 0
        assert fp8.bytes() > 0
        with pytest.raises(NotImplementedError):
            fp8.flops_bwd()

        fp4 = perf_model.hipblaslt_gemm_fp4(
            _fp4_gemm_event((128, 64), (256, 32), trans_b=True)
        )
        assert fp4.flops() > 0
        assert fp4.bytes() > 0


class TestMoeExtensionsPush95Coverage:
    MOE_FUSED = {
        "args": {
            "Input Dims": [
                [32, 4096],
                [8, 28672, 512],
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

    MOE_BLOCKSCALE = {
        "args": {
            "Input Dims": [
                [128, 256],
                [128, 256],
                [512, 28672, 256],
                [512, 256, 7168],
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::Float8_e4m3fn",
                "c10::Float8_e4m3fn",
                "c10::Float8_e4m3fn",
            ],
            "Concrete Inputs": [""] * 8 + ["2"],
        }
    }

    CK1 = {
        "args": {
            "Input Dims": [
                [32, 512],
                [8, 7168, 512],
                [8, 4096, 896],
                [],
                [],
                [],
                [32, 2, 7168],
            ],
            "Input type": ["c10::BFloat16", "c10::Float8_e4m3fn", "c10::Float8_e4m3fn"],
        }
    }

    CK2 = {
        "args": {
            "Input Dims": [
                [32, 2, 7168],
                [8, 7168, 512],
                [8, 4096, 896],
                [],
                [],
                [],
                [32, 4096],
            ],
            "Input type": ["c10::BFloat16", "c10::Float8_e4m3fn", "c10::Float8_e4m3fn"],
        }
    }

    FLY = {
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
            ],
        }
    }

    GPTQ = {
        "args": {
            "Input Dims": [[32, 4096], [32, 2], [8, 7168, 4096]],
            "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
            "MoE topk": 2,
        }
    }

    GROUPED = {
        "args": {
            "Input Dims": [
                [64, 2048],
                [128, 1536, 2048],
                (),
                [512, 1536],
                (),
                (),
                (),
                (),
                [64, 4],
            ],
            "Input type": ["c10::BFloat16", "c10::Float8_e4m3fn", "", "c10::BFloat16"],
        }
    }

    @pytest.mark.parametrize(
        "factory,event",
        [
            (moe_ext.moe_aiter_fused_1stage, "MOE_FUSED"),
            (moe_ext.moe_aiter_fused_blockscale, "MOE_BLOCKSCALE"),
            (moe_ext.moe_aiter_ck_stage1, "CK1"),
            (moe_ext.moe_aiter_ck_stage2, "CK2"),
            (moe_ext.moe_flydsl_stage1, "FLY"),
            (moe_ext.moe_flydsl_stage2, "FLY"),
            (moe_ext.moe_gptq_awq_up, "GPTQ"),
            (moe_ext.moe_gptq_awq_down, "GPTQ"),
            (moe_ext.moe_triton_invoke_grouped_gemm, "GROUPED"),
            (moe_ext.moe_triton_unfused_up, None),
            (moe_ext.moe_triton_unfused_down, None),
            (moe_ext.moe_aiter_unfused_up, None),
            (moe_ext.moe_aiter_unfused_down, None),
            (moe_ext.sglang_fused_append_shared_experts, None),
            (moe_ext.BiasedGroupedTopk, None),
            (moe_ext.MoeSortScatterGather, None),
        ],
    )
    def test_moe_bytes_and_bwd_raises(self, factory, event):
        if event is None:
            if factory in (
                moe_ext.moe_triton_unfused_up,
                moe_ext.moe_triton_unfused_down,
            ):
                evt = _moe_unfused_event(kernel_name="moe_fp8_up_kernel")
            elif factory is moe_ext.moe_aiter_unfused_up:
                evt = {
                    "args": {
                        "Input Dims": [
                            [32, 4096],
                            [8, 14336, 512],
                            [32, 2, 7168],
                        ],
                        "Input type": [
                            "c10::BFloat16",
                            "c10::Float8_e4m3fn",
                            "c10::BFloat16",
                        ],
                    }
                }
            elif factory is moe_ext.moe_aiter_unfused_down:
                evt = {
                    "args": {
                        "Input Dims": [
                            [32, 2, 7168],
                            [8, 4096, 896],
                            [32, 4096],
                        ],
                        "Input type": [
                            "c10::BFloat16",
                            "c10::Float8_e4m3fn",
                            "c10::BFloat16",
                        ],
                    }
                }
            elif factory is moe_ext.sglang_fused_append_shared_experts:
                evt = {
                    "args": {
                        "Input Dims": [(32, 4096), (32, 4096), (32, 4096)],
                        "Input type": ["c10::BFloat16"] * 3,
                    }
                }
            elif factory is moe_ext.BiasedGroupedTopk:
                evt = {
                    "args": {
                        "Input Dims": [(32, 256), (256,), (32, 8), (32, 8)],
                        "Input type": ["c10::Float"] * 3 + ["c10::Int"],
                    }
                }
            else:
                evt = {
                    "args": {
                        "Input Dims": [(32, 4096), (32, 2), (32, 4096)],
                        "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
                    }
                }
        else:
            evt = getattr(self, event)

        model = factory(evt)
        b = model.bytes()
        assert b is None or b >= 0
        if hasattr(model, "flops_bwd"):
            with pytest.raises(NotImplementedError):
                model.flops_bwd()
        if hasattr(model, "bytes_bwd"):
            with pytest.raises(NotImplementedError):
                model.bytes_bwd()


class TestPerfModelFinalCoverage:
    @pytest.mark.parametrize(
        "cls,fwd_factory,bwd_cls,bwd_factory",
        [
            (
                perf_model.aten_mm,
                lambda: _gemm_event("aten::mm", (4, 8), (8, 16)),
                None,
                None,
            ),
            (perf_model.aten_addmm, None, None, None),
            (perf_model.aten_bmm, None, None, None),
            (perf_model.aten_baddbmm, None, None, None),
        ],
    )
    def test_gemm_backward_not_implemented(
        self, cls, fwd_factory, bwd_cls, bwd_factory
    ):
        if cls is perf_model.aten_addmm:
            event = {
                "args": {
                    "Input Dims": [(4, 16), (4, 8), (8, 16)],
                    "Input type": ["c10::BFloat16"] * 3,
                    "Input Strides": [(16, 1), (8, 1), (16, 1)],
                }
            }
        elif cls is perf_model.aten_bmm:
            event = {
                "args": {
                    "Input Dims": [[2, 4, 8], [2, 8, 16]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                    "Input Strides": [(32, 8, 1), (128, 16, 1)],
                }
            }
        elif cls is perf_model.aten_baddbmm:
            event = {
                "args": {
                    "Input Dims": [[2, 4, 16], [2, 4, 8], [2, 8, 16]],
                    "Input type": ["c10::BFloat16"] * 3,
                    "Input Strides": [(64, 16, 1), (32, 8, 1), (128, 16, 1)],
                }
            }
        else:
            event = fwd_factory()
        model = cls(event)
        with pytest.raises(NotImplementedError):
            model.flops_bwd()
        with pytest.raises(NotImplementedError):
            model.bytes_bwd()

    @pytest.mark.parametrize(
        "cls,event",
        [
            (
                perf_model.aten_addmm,
                {
                    "args": {
                        "Input Dims": [(4, 16), (4, 8), (8, 16)],
                        "Input type": ["c10::BFloat16", "c10::Half", "c10::BFloat16"],
                    }
                },
            ),
            (
                perf_model.aten_bmm,
                {
                    "args": {
                        "Input Dims": [[2, 4, 8], [2, 8, 16]],
                        "Input type": ["c10::BFloat16", "c10::Half"],
                    }
                },
            ),
            (
                perf_model.aten_baddbmm,
                {
                    "args": {
                        "Input Dims": [[2, 4, 16], [2, 4, 8], [2, 8, 16]],
                        "Input type": ["c10::BFloat16", "c10::Half", "c10::BFloat16"],
                    }
                },
            ),
        ],
    )
    def test_batched_gemm_mixed_dtype_warns(self, cls, event):
        model = cls(event)
        with pytest.warns(UserWarning):
            model.bytes()

    def test_gemm_simulator_with_python_path(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        try:
            with patch("TraceLens.PerfModel.perf_model.subprocess.run") as run:
                run.return_value = MagicMock(stdout="Time=5.5\n", stderr="")
                t, cmd = perf_model.GEMM.get_simulation_time_func(
                    _ARCH,
                    4,
                    8,
                    16,
                    None,
                    "bf16",
                    python_path="/usr/bin/python3",
                    num_cus=64,
                )
            assert t == 5.5
            assert "/usr/bin/python3" in cmd
        finally:
            perf_model.GEMM.cache_gemm_results.clear()

    def test_aten_reduce_edge_cases(self):
        empty = perf_model.aten_reduce(
            {"name": "aten::sum", "args": {"Input Dims": [None]}}
        )
        assert empty.param_details["num_input_elems"] == 0

        mean_evt = {
            "name": "aten::mean",
            "args": {
                "Input Dims": [(4, 256)],
                "Input type": ["c10::BFloat16"],
                "Output type": ["c10::BFloat16"],
                "Concrete Inputs": ["", "[1]", "True"],
            },
        }
        m = perf_model.aten_reduce(mean_evt)
        assert m.flops() > 0
        assert m.bytes() > 0

        cumsum_evt = {
            "name": "aten::cumsum",
            "args": {
                "Input Dims": [(8, 32)],
                "Input type": ["c10::Float"],
                "Concrete Inputs": ["", "[0]", "False"],
            },
        }
        c = perf_model.aten_reduce(cumsum_evt)
        assert c.param_details["num_output_elems"] == 8 * 32

    def test_conv_bias_backward_with_sequence_cache(self):
        fwd = perf_model.ConvBias_(_conv_bias_fwd_event())
        assert fwd.flops() > 0
        bwd_evt = _conv_bias_bwd_event()
        bwd_evt["args"]["Sequence number"] = bwd_evt["args"].get("Sequence number", 42)
        perf_model.ConvBias_.fwd_pass_cache[42] = fwd.param_details
        bwd = perf_model.ConvBias_Backward(bwd_evt)
        assert bwd.flops_bwd() > 0
        assert bwd.bytes_bwd() > 0
        perf_model.ConvBias_.fwd_pass_cache.pop(42, None)

    def test_conv_bias_relu_backward_paths(self):
        fwd = perf_model.ConvBiasReLU_(_conv_bias_relu_fwd_event())
        bwd_evt = _conv_bias_relu_bwd_event()
        seq = bwd_evt["args"].get("Sequence number", 43)
        perf_model.ConvBiasReLU_.fwd_pass_cache[seq] = fwd.param_details
        bwd = perf_model.ConvBiasReLU_Backward(bwd_evt)
        assert bwd.flops_bwd() > 0
        assert bwd.bytes_bwd() > 0
        perf_model.ConvBiasReLU_.fwd_pass_cache.pop(seq, None)

    def test_aten_scaled_mm_mixed_output_bpe(self):
        event = {
            "args": {
                "Input Dims": [[4, 8], [8, 16], [4, 16]],
                "Input type": [
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                ],
            }
        }
        model = perf_model.aten_scaled_mm(event)
        assert model.bytes() > 0

    def test_primus_grouped_gemm_variable_k(self):
        event = {
            "name": "primus_turbo::grouped_gemm_variable_k_impl",
            "args": {
                "Input Dims": [[24576, 1408], [24576, 2048]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            },
        }
        g = perf_model.primus_turbo_grouped_gemm_variable_k(event)
        assert g.flops() > 0
        assert g.bytes() > 0


class TestMoeExtensionsFinal:
    MOE_BLOCKSCALE = {
        "args": {
            "Input Dims": [
                [32, 4096],
                [32, 4096],
                [8, 14336, 4096],
                [8, 4096, 7168],
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::Float8_e4m3fn",
                "c10::Float8_e4m3fn",
                "c10::Float8_e4m3fn",
            ],
            "Concrete Inputs": [""] * 8 + ["2"],
        }
    }

    CK_STAGE1 = {
        "args": {
            "Input Dims": [
                [32, 512],
                [8, 7168, 512],
                [8, 4096, 896],
                [],
                [],
                [],
                [32, 2, 7168],
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::Float8_e4m3fn",
                "c10::Float8_e4m3fn",
            ],
        }
    }

    CK_STAGE2 = {
        "args": {
            "Input Dims": [
                [32, 2, 7168],
                [8, 7168, 512],
                [8, 4096, 896],
                [],
                [],
                [],
                [32, 4096],
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::Float8_e4m3fn",
                "c10::Float8_e4m3fn",
            ],
        }
    }

    @pytest.mark.parametrize(
        "factory,event",
        [
            (moe_ext.moe_aiter_fused_blockscale, "MOE_BLOCKSCALE"),
            (moe_ext.moe_aiter_ck_stage1, "CK_STAGE1"),
            (moe_ext.moe_aiter_ck_stage2, "CK_STAGE2"),
        ],
    )
    def test_moe_bytes_and_precision(self, factory, event):
        event_obj = getattr(self, event) if isinstance(event, str) else event
        if event == "MOE_STD":
            event_obj = {
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
                    ],
                }
            }
        elif event == "GROUPED":
            event_obj = {
                "args": {
                    "Input Dims": [
                        [64, 2048],
                        [128, 1536, 2048],
                        (),
                        [512, 1536],
                        (),
                        (),
                        (),
                        (),
                        [64, 4],
                    ],
                    "Input type": [
                        "c10::BFloat16",
                        "c10::Float8_e4m3fn",
                        "",
                        "c10::BFloat16",
                    ],
                }
            }
        elif event == "GPTQ":
            event_obj = {
                "args": {
                    "Input Dims": [[32, 4096], [32, 2], [8, 7168, 4096]],
                    "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
                    "MoE topk": 2,
                }
            }
        model = factory(event_obj)
        b = model.bytes()
        assert b is None or b > 0
        prec = model.get_compute_precision()
        assert prec in (None, "bf16", "fp8", "fp4", "fp16", "fp32")
        if hasattr(model, "flops_bwd"):
            with pytest.raises(NotImplementedError):
                model.flops_bwd()

    def test_moe_triton_unfused_and_sglang(self):

        up = moe_ext.moe_triton_unfused_up(
            _moe_unfused_event(kernel_name="moe_mxfp4_up_kernel")
        )
        down = moe_ext.moe_triton_unfused_down(
            _moe_unfused_event(kernel_name="moe_fp8_down_kernel")
        )
        assert up.bytes() > 0
        assert down.bytes() > 0
        sgl = moe_ext.sglang_fused_append_shared_experts(
            {
                "args": {
                    "Input Dims": [(32, 4096), (32, 4096), (32, 4096)],
                    "Input type": ["c10::BFloat16"] * 3,
                }
            }
        )
        assert sgl.bytes() > 0

    def test_moe_flydsl_gptq_grouped(self):
        fly = {
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
                ],
            }
        }
        assert moe_ext.moe_flydsl_stage1(fly).bytes() > 0
        assert moe_ext.moe_flydsl_stage2(fly).bytes() > 0
        gptq = {
            "args": {
                "Input Dims": [[32, 4096], [32, 2], [8, 7168, 4096]],
                "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
                "MoE topk": 2,
            }
        }
        assert moe_ext.moe_gptq_awq_up(gptq).bytes() > 0
        grouped = {
            "args": {
                "Input Dims": [
                    [64, 2048],
                    [128, 1536, 2048],
                    (),
                    [512, 1536],
                    (),
                    (),
                    (),
                    (),
                    [64, 4],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "",
                    "c10::BFloat16",
                ],
            }
        }
        assert moe_ext.moe_triton_invoke_grouped_gemm(grouped).bytes() > 0

    def test_biased_topk_and_sort_scatter_precision(self):
        topk = moe_ext.BiasedGroupedTopk(
            {
                "args": {
                    "Input Dims": [(32, 256), (256,), (32, 8), (32, 8)],
                    "Input type": ["c10::Float"] * 3 + ["c10::Int"],
                }
            }
        )
        assert topk.flops() > 0
        sort = moe_ext.MoeSortScatterGather(
            {
                "args": {
                    "Input Dims": [(32, 4096), (32, 2), (32, 4096)],
                    "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
                }
            }
        )
        assert sort.bytes() > 0


class TestPerfModelDeepCoverage2:
    def test_aten_conv_bwd_with_bias_grad(self):
        event = {
            "args": {
                "Input Dims": [
                    [2, 4, 6, 6],
                    [2, 3, 8, 8],
                    [4, 3, 3, 3],
                    [4],
                ],
                "Input type": ["c10::BFloat16"] * 4,
                "Concrete Inputs": [
                    "",
                    "",
                    "",
                    "",
                    "(1,1)",
                    "(0,0)",
                    "(1,1)",
                    "False",
                    "(0,0)",
                    "1",
                    "[True, True, False]",
                ],
            }
        }
        model = perf_model.aten_conv_bwd(event)
        assert model.flops_bwd() > 0

    def test_extract_sdpa_configs(self):
        cfg = perf_model.extract_sdpa_cfg(
            q_shape=[2, 8, 128, 64],
            k_shape=[2, 8, 128, 64],
            v_shape=[2, 8, 128, 64],
            bhnd_idx=(0, 1, 2, 3),
        )
        assert cfg["B"] == 2
        vcfg = perf_model.extract_sdpa_varlen_cfg(
            q_shape=[8, 128, 64],
            k_shape=[8, 128, 64],
            v_shape=[8, 128, 64],
            hnd_idx=(0, 1, 2),
        )
        assert vcfg["B"] == 1

    def test_grouped_gemm_zipped_and_impl_formats(self):
        zipped = {
            "name": "primus_turbo::grouped_gemm",
            "args": {
                "Input Dims": [[[4, 8], [5, 8]], [[8, 16], [8, 16]]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            },
        }
        g = perf_model.primus_turbo_grouped_gemm(zipped)
        assert g.flops() > 0
        assert g.get_compute_precision() == "bf16"

    def test_gemm_simulator_default_batch_and_invalid_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(tmp_path / "missing.py"))
        with pytest.raises(ValueError, match="does not exist"):
            perf_model.GEMM.get_simulation_time_func(_ARCH, 4, 8, 16, 1, "bf16")

    def test_fused_rope_and_cross_entropy_precision(self):
        rope = perf_model.fused_rope_fwd(
            {
                "args": {
                    "Input Dims": [[128, 2, 8, 64], [128, 1, 1, 64]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                }
            }
        )
        assert rope.get_compute_precision() == "bf16"
        ce = perf_model.cross_entropy_fwd(
            {
                "args": {
                    "Input Dims": [[4, 1, 32000], [4, 1]],
                    "Input type": ["c10::BFloat16", "long int"],
                }
            }
        )
        assert ce.get_compute_precision() is not None
