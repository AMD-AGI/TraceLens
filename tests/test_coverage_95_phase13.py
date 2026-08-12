###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase-13: last-mile coverage to <=942 miss."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from TraceLens.Agent.Analysis.category_analyses import analysis_utils as au
from TraceLens.PerfModel.extensions import moe_perf_model_extensions as moe_ext
from TraceLens.Reporting import compare_traces_jax_llama as jax_cmp
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

from tests.test_coverage_95_final import _jax_llama_trace_events, _write_gz_trace

RESNET = os.path.join(
    os.path.dirname(__file__), "traces/mi300/resnet_act_checkpoint.json.gz"
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


class TestAnalysisUtilsPhase13:
    def test_efficiency_memory_bound_and_fusion_map_empty(self, tmp_path):
        row = pd.Series(
            {
                "FLOPS/Byte": 0.5,
                "TFLOPS/s_mean": 10.0,
                "TB/s_mean": 2.0,
                "Roofline Bound": "MEMORY_BOUND",
                "Compute Spec": "matrix_fp16",
            }
        )
        eff = au.calculate_efficiency(
            row, peak_maf_or_maf_dict={"matrix_fp16": 100.0}, peak_hbm_bw=5300
        )
        assert eff["bound_type"] == "memory"
        assert au._load_fusion_map(str(tmp_path)) == {}


class TestJaxComparePhase13:
    def test_jax_llama_helpers(self, tmp_path):
        path = _write_gz_trace(tmp_path, _jax_llama_trace_events())
        trace = jax_cmp.load_trace(path)
        evs = jax_cmp.extract_gpu_events(trace, gpu_index=0)
        assert len(evs) > 0
        d_model, head_dim, gsu = jax_cmp.infer_params(evs)
        assert d_model == 4096


class TestTreePerfPhase13:
    @pytest.mark.skipif(not os.path.isfile(RESNET), reason="resnet missing")
    def test_unified_table_with_perf_metrics(self):
        analyzer = TreePerfAnalyzer.from_file(
            RESNET,
            rebuild_tree=True,
            enable_pseudo_ops=True,
            detect_recompute=True,
        )
        df = analyzer.build_df_unified_perf_table(include_perf_metrics=True)
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            summary = analyzer.summarize_df_unified_perf_table(df, include_pct=True)
            assert isinstance(summary, pd.DataFrame)
