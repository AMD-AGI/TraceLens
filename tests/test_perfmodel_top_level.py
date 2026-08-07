###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for modules directly under TraceLens/PerfModel (no subpackages)."""

import importlib.util

import pytest

from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel._known_aten_ops import KNOWN_OPS
from TraceLens.PerfModel.jax_op_mapping import jax_op_to_perf_model_class_map
from TraceLens.PerfModel.kernel_name_parser import (
    gemm_name_parser,
    is_cuda_gemm,
    is_rocm_gemm,
    parse_cuda_gemm,
    parse_rocm_gemm,
)
from TraceLens.PerfModel.perf_model import (
    _collect_2d_shapes,
    dtype_jax2torch,
    extract_sdpa_cfg,
    extract_sdpa_varlen_cfg,
    jax_dtype2bpe,
    jax_dtype_map,
    parse_list,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    get_perf_model_category,
    sheet_category_from_final_category,
)
from TraceLens.PerfModel.triton_compiled_perf_model import (
    _cache_dirs,
    _parse_wrapper,
)
from TraceLens.PerfModel.utils import (
    add_simulation_time_columns,
    name2bpe,
    optional_float,
    optional_int,
    parse_bool,
    simulation_dtype_map,
    torch_dtype_map,
)

ROCM_GEMM = (
    "Custom_Cijk_Alik_Bljk_BBS_BH_Bias_HAS_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB0"
)


class TestPerfModelUtils:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("42", 42),
            ("bad", None),
            (None, None),
        ],
    )
    def test_optional_int(self, value, expected):
        assert optional_int(value, default=None) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1.5", 1.5),
            ("", 0.0),
            ("None", 0.0),
            ("bad", 0.0),
        ],
    )
    def test_optional_float(self, value, expected):
        assert optional_float(value) == expected

    def test_name2bpe_and_dtype_maps(self):
        assert name2bpe("c10::BFloat16") == 2
        assert name2bpe("unknown") is None
        assert simulation_dtype_map("bf16") == "c10::bfloat16"
        assert torch_dtype_map("c10::bfloat16") == "bf16"

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("0", False),
            (True, True),
            (None, False),
        ],
    )
    def test_parse_bool(self, value, expected):
        assert parse_bool(value) is expected

    def test_add_simulation_time_columns(self):
        metrics = {}
        add_simulation_time_columns(
            metrics,
            simulated_time=100.0,
            gflops=200.0,
            bytes_moved=1e12,
            busy_kernel_time=200.0,
        )
        assert metrics["Origami Time (µs)"] == 100.0
        assert metrics["Origami TFLOPS/s"] == pytest.approx(2000.0)
        assert metrics["Pct Origami"] == 50.0


class TestKernelNameParser:
    def test_rocm_gemm_detection_and_parse(self):
        assert is_rocm_gemm(ROCM_GEMM)
        parsed = parse_rocm_gemm(ROCM_GEMM)
        assert parsed["transpose"] == (True, False)
        assert parsed["mt_m"] == 64
        assert parsed["mt_n"] == 16
        assert parsed["depth_u"] == 64
        assert gemm_name_parser(ROCM_GEMM) == parsed

    def test_cuda_gemm_detection_and_parse(self):
        name = "nvjet_tst_32x64x32_TN"
        assert is_cuda_gemm(name)
        assert parse_cuda_gemm(name) == {"transpose": (True, False)}
        assert gemm_name_parser(name) == {"transpose": (True, False)}

    def test_non_gemm_returns_none(self):
        assert gemm_name_parser("unrelated_kernel") is None


class TestPerfModelHelpers:
    def test_extract_sdpa_cfg(self):
        cfg = extract_sdpa_cfg(
            q_shape=[2, 8, 128, 64],
            k_shape=[2, 8, 128, 64],
            v_shape=[2, 8, 128, 64],
            bhnd_idx=(0, 1, 2, 3),
        )
        assert cfg == {
            "B": 2,
            "N_Q": 128,
            "H_Q": 8,
            "N_KV": 128,
            "H_KV": 8,
            "d_h_qk": 64,
            "d_h_v": 64,
        }

    def test_extract_sdpa_varlen_cfg(self):
        cfg = extract_sdpa_varlen_cfg(
            q_shape=[8, 128, 64],
            k_shape=[8, 128, 64],
            v_shape=[8, 128, 64],
            hnd_idx=(0, 1, 2),
        )
        assert cfg["B"] == 1
        assert cfg["H_Q"] == 8

    def test_jax_dtype_helpers(self):
        assert jax_dtype2bpe("bf16") == 2
        assert jax_dtype_map("f16") == "fp16"
        assert dtype_jax2torch("bf16") == "c10::bfloat16"

    def test_parse_list(self):
        assert parse_list("[1, 2, 3]", int) == [1, 2, 3]

    def test_collect_2d_shapes(self):
        nested = [[(4, 8), (2, 8)], [(8, 16), (8, 16)]]
        assert _collect_2d_shapes(nested) == [(4, 8), (2, 8), (8, 16), (8, 16)]


class TestTorchOpMapping:
    def test_sheet_category_from_final_category(self):
        assert sheet_category_from_final_category("GEMM_fwd") == "GEMM"
        assert sheet_category_from_final_category("elementwise") == "elementwise"

    def test_get_perf_model_category(self):
        assert get_perf_model_category(perf_model.GEMM) == "GEMM"

    def test_categorize_torch_op_known_mapping(self):
        row = {"name": "aten::mm"}
        assert categorize_torch_op(row) == "GEMM"

    def test_categorize_torch_op_triton_pattern(self):
        row = {"name": "triton_poi_fused_add_0"}
        assert categorize_torch_op(row) == "triton"


class TestJaxOpMapping:
    def test_jax_op_map_contains_expected_entries(self):
        assert set(jax_op_to_perf_model_class_map) >= {
            "jax_gemm",
            "jax_conv",
            "jax_te_fused_attn",
        }
        assert jax_op_to_perf_model_class_map["jax_gemm"] is perf_model.jax_gemm


class TestKnownAtenOps:
    def test_known_ops_contains_flash_attention(self):
        assert "_flash_attention_forward" in KNOWN_OPS
        assert "aten::mm" not in KNOWN_OPS


class TestTritonCompiledHelpers:
    def test_cache_dirs_honors_explicit_directory(self, tmp_path):
        assert _cache_dirs(str(tmp_path)) == [str(tmp_path)]
        assert _cache_dirs(str(tmp_path / "missing")) == []

    def test_parse_wrapper_extracts_kernel_metadata(self):
        content = """
# Original ATen: [aten.add, aten.mean]
triton_red_fused_add_mean_0 = async_compile.triton(
    'triton_red_fused_add_mean_0',
    '''
    size_hints=[32768, 2048]
    ''', device_str='cuda')
"""
        parsed = _parse_wrapper(content)
        assert "triton_red_fused_add_mean_0" in parsed
        meta = parsed["triton_red_fused_add_mean_0"]
        assert meta["aten_ops"] == ["aten.add", "aten.mean"]
        assert meta["xnumel"] == 32768
        assert meta["rnumel"] == 2048


HAS_ORIGAMI = importlib.util.find_spec("origami") is not None


@pytest.mark.skipif(not HAS_ORIGAMI, reason="origami not installed")
class TestOrigamiHelper:
    def test_get_hardware_maps_mi300(self):
        from TraceLens.PerfModel.origami_helper import OrigamiHelper

        hardware = OrigamiHelper.get_hardware({"name": "mi300x", "freq_mhz": 2200})
        assert hardware.N_CU == 304
        assert hardware.lds_capacity == 64 * 1024
        assert hardware.L2_capacity == 4 * 1024 * 1024
        assert hardware.compute_clock_ghz == pytest.approx(2.2)

    def test_origami_helper_simulation_time_positive(self):
        import origami

        from TraceLens.PerfModel.origami_helper import OrigamiHelper

        hardware = OrigamiHelper.get_hardware({"name": "mi300x", "freq_mhz": 2200})
        helper = OrigamiHelper(
            m=128,
            n=128,
            k=128,
            b=1,
            a_dtype=origami.data_type_t.BFloat16,
            b_dtype=origami.data_type_t.BFloat16,
            out_dtype=origami.data_type_t.BFloat16,
            hardware=hardware,
        )
        assert helper.get_simulation_time() > 0
