###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase-14: cover remaining sub-95% helper modules to pass --cov-fail-under=95."""

from __future__ import annotations

import json
import sys
from types import ModuleType

import pandas as pd
import pytest

from TraceLens.Agent.Analysis.category_analyses import (
    convolution_analysis,
    elementwise_analysis,
    gemm_analysis,
    moe_analysis,
    norm_analysis,
    reduce_analysis,
    triton_analysis,
)
from TraceLens.Agent.Analysis.utils import arch_utils
from TraceLens.PerfModel import kernel_name_parser
from TraceLens.TraceDiff import util as tracediff_util
from TraceLens.util import TraceEventUtils

_TK = TraceEventUtils.TraceKeys


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


class TestArchUtils:
    def test_list_and_load_platform(self):
        platforms = arch_utils.list_platforms()
        assert platforms
        for name in platforms[:3]:
            arch = arch_utils.load_arch(name)
            assert "name" in arch or "mem_bw_gbps" in arch

    def test_tl_extension_override(self, tmp_path, monkeypatch):
        ext_pkg = tmp_path / "fake_ext_pkg"
        arch_dir = ext_pkg / "Agent" / "Analysis" / "utils" / "arch"
        arch_dir.mkdir(parents=True)
        custom = {"name": "CUSTOM", "mem_bw_gbps": 1000, "memory_gb": 80}
        (arch_dir / "CUSTOM.json").write_text(json.dumps(custom))

        init_py = ext_pkg / "__init__.py"
        init_py.write_text("")

        fake_mod = ModuleType("fake_ext_pkg")
        fake_mod.__file__ = str(init_py)
        monkeypatch.setitem(sys.modules, "fake_ext_pkg", fake_mod)
        monkeypatch.setenv("TL_EXTENSION", "fake_ext_pkg")

        mapping = arch_utils._collect_arch_jsons()
        assert "CUSTOM" in mapping
        assert arch_utils.load_arch("CUSTOM")["mem_bw_gbps"] == 1000


class TestCategoryAnalysisHelpers:
    _META = {"peak_hbm_bw_tbs": 5.3, "peak_maf_tflops": {"matrix_fp16": 654}}

    def test_gemm_classifiers(self):
        assert gemm_analysis.detect_quantized_gemm("aten::w8a8_mm")
        info = gemm_analysis.classify_gemm_operation("aten::mm", None)
        assert info["gemm_type"] == "regular"
        qinfo = gemm_analysis.classify_gemm_operation("aten::fp8_mm", None)
        assert qinfo["is_quantized"] is True
        ops = pd.DataFrame(
            {
                "name": ["aten::mm", "aten::fp8_mm"],
                "TFLOPS/s_mean": [100.0, None],
            }
        )
        extra = gemm_analysis.extract_category_specific(ops, self._META)
        assert extra["quantized_count"] == 1
        assert extra["missing_perf_model_count"] == 1

    def test_elementwise_extract(self):
        out = elementwise_analysis.extract_category_specific(
            pd.DataFrame({"name": ["aten::add"]}), self._META
        )
        assert out["peak_hbm_bw_tbs"] == 5.3

    def test_norm_extract(self):
        out = norm_analysis.extract_category_specific(
            pd.DataFrame({"name": ["aten::layer_norm"]}), self._META
        )
        assert out["peak_hbm_bw_tbs"] == 5.3

    def test_reduce_extract(self):
        out = reduce_analysis.extract_category_specific(
            pd.DataFrame({"name": ["aten::sum"]}), self._META
        )
        assert "peak_hbm_bw_tbs" in out

    def test_convolution_extract_with_transpose(self):
        ops = pd.DataFrame(
            {
                "name": ["aten::conv2d", "aten::conv_transpose2d"],
                "Kernel Time (µs)_sum": [1000.0, 500.0],
            }
        )
        out = convolution_analysis.extract_category_specific(ops, self._META)
        assert out["transpose_count"] == 1
        assert out["transpose_time_ms"] == pytest.approx(0.5)

    def test_triton_classifiers(self):
        assert (
            triton_analysis.classify_triton_operation("triton_poi_add", None)[
                "kernel_type"
            ]
            == "pointwise"
        )
        assert (
            triton_analysis.classify_triton_operation("triton_red_sum", None)[
                "kernel_type"
            ]
            == "reduction"
        )
        assert (
            triton_analysis.classify_triton_operation("triton_per_mm", None)[
                "kernel_type"
            ]
            == "persistent"
        )
        assert (
            triton_analysis.classify_triton_operation("other", None)["kernel_type"]
            == "other"
        )
        out = triton_analysis.extract_category_specific(
            pd.DataFrame(
                {
                    "name": [
                        "triton_poi_a",
                        "triton_red_b",
                        "triton_per_c",
                        "other",
                    ]
                }
            ),
            self._META,
        )
        assert out["pointwise_count"] == 1
        assert out["reduction_count"] == 1

    def test_reduce_softmax_detect(self):
        assert reduce_analysis.detect_softmax("aten::_softmax")
        out = reduce_analysis.extract_category_specific(
            pd.DataFrame({"name": ["aten::_softmax", "aten::sum"]}), self._META
        )
        assert out["softmax_count"] == 1

    def test_moe_extract_and_no_data_check(self, tmp_path):
        out = moe_analysis.extract_category_specific(
            pd.DataFrame({"name": ["moe_dispatch"]}), self._META
        )
        assert "peak_hbm_bw_tbs" in out
        missing = moe_analysis._check_moe_data(str(tmp_path), "moe_fused", "standalone")
        assert missing["status"] == "NO_DATA"


class TestTraceDiffUtil:
    def test_sort_and_name_helpers(self):
        nodes = [
            {_TK.UID: 2, _TK.TimeStamp: 20, _TK.Name: "b"},
            {_TK.UID: 1, _TK.TimeStamp: 10, _TK.Name: "a"},
        ]
        assert tracediff_util._sort_by_ts(nodes) == [1, 2]
        assert tracediff_util._get_name_node(nodes[0]) == "b"
        assert tracediff_util._get_name_node(None) is None
        assert tracediff_util._list_to_tuple([1, [2, 3]]) == (1, (2, 3))
        node = {"args": {"Input Dims": [[1, 2]]}}
        assert tracediff_util._get_node_arg(node, "Input Dims") == ((1, 2),)
        assert tracediff_util._get_node_arg(node, "missing") == ""

    def test_gpu_path_and_kernel(self):
        node = {
            _TK.Name: "op",
            _TK.Category: "cpu_op",
            "non_gpu_path": False,
        }
        assert tracediff_util._is_gpu_path(node) is True
        assert tracediff_util._is_kernel({_TK.Category: "kernel"}) is True
        assert tracediff_util._is_kernel({_TK.Category: "cpu_op"}) is False
        assert tracediff_util._is_gpu_path(None) is False
        assert tracediff_util._is_gpu_path({"non_gpu_path": True}) is False

    def test_normalize_name_for_comparison(self):
        raw = "/path/to/model.py(42): forward 0xabc123"
        norm = tracediff_util._normalize_name_for_comparison(raw)
        assert "0xXXXX" in norm
        assert ".py:" in norm
        launch = tracediff_util._normalize_name_for_comparison("hipModuleLaunchKernel")
        assert launch == "__kernel_launch__"
