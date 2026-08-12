###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU-only coverage tests for TraceLens/PerfModel modules.

Uses synthetic kernel names and trace event shapes. Origami/ROCm paths are
mocked where needed. Excludes benchmarking/* and origami_helper.py (omitted in
coverage config).
"""

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
        mock_helper_cls.get_hardware.return_value = MagicMock(N_CU=304)
        mock_helper_cls.return_value.get_simulation_time.return_value = 99.0
        with patch.dict(sys.modules, {"origami": mock_origami}):
            with patch(
                "TraceLens.PerfModel.origami_helper.OrigamiHelper", mock_helper_cls
            ):
                t, _ = perf_model.GEMM.get_simulation_time_func(
                    _ARCH, 4, 8, 16, 1, "bf16", enable_origami=True, force_to_l1=True
                )
        assert t == 99.0

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
        from TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions import (
            fused_rms_mxfp4_quant,
        )

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
        from TraceLens.PerfModel.extensions.moe_perf_model_extensions import (
            FusedMoE,
            moe_aiter_fused_1stage,
        )

        model = moe_aiter_fused_1stage(self.MOE_1STAGE)
        assert model.flops() > 0
        assert model.bytes() > 0
        assert model.get_compute_precision() == "bf16"
        with pytest.raises(NotImplementedError):
            model.flops_bwd()
        assert FusedMoE.bytes_func(32, 4096, 7168, 8, 2, True, None, 1, 2) is None

    def test_biased_grouped_topk_and_sort_scatter(self):
        from TraceLens.PerfModel.extensions.moe_perf_model_extensions import (
            BiasedGroupedTopk,
            MoeSortScatterGather,
        )

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
