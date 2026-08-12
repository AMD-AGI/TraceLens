###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase-7 coverage: trace_diff, analysis_utils, reporting, jax, capture merge."""

from __future__ import annotations

import importlib
import json
import os
import sys

import pandas as pd
import pytest

from TraceLens.Agent.Analysis.category_analyses import analysis_utils as au
from TraceLens.Agent.Analysis.category_analyses import kernel_fusion_analysis as kfa
from TraceLens.Reporting import compare_traces_jax_llama as jax_cmp
from TraceLens.Reporting.compare_perf_reports_pytorch import (
    generate_compare_perf_reports_pytorch,
)
from TraceLens.Reporting.generate_multi_rank_collective_report_pytorch import (
    generate_collective_report,
)
from TraceLens.Reporting.pftrace_hip_activity_analysis import PftraceHipActivityAnalyzer
from TraceLens.Reporting.tracediff_comparison_extension import (
    tracediff_perf_summary_from_diff_stats,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)
from TraceLens.TraceDiff.trace_diff import TraceDiff
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

from tests.test_coverage_95_final import _jax_llama_trace_events, _write_gz_trace
from tests.test_reporting_coverage import (
    _minimal_pftrace_events,
    _mk_event,
    _write_trace,
)

INFERENCE_ROOT = os.path.join(os.path.dirname(__file__), "traces/inference")
COMPARE_DIR = os.path.join(os.path.dirname(__file__), "traces/compare_test_ops")


class TestTraceDiffPhase7:
    @pytest.mark.skipif(
        not os.path.isdir(COMPARE_DIR),
        reason="compare traces missing",
    )
    def test_trace_diff_from_compare_traces(self):
        from TraceLens.Trace2Tree.trace_to_tree import TraceToTree

        t1 = os.path.join(COMPARE_DIR, "256thread", "perf_28ch_rank0.json.gz")
        t2 = os.path.join(COMPARE_DIR, "512thread", "perf_28ch_rank0.json.gz")
        if not (os.path.isfile(t1) and os.path.isfile(t2)):
            pytest.skip("compare gz missing")
        tree1 = TraceToTree.from_file(t1, rebuild_tree=True, enable_pseudo_ops=True)
        tree2 = TraceToTree.from_file(t2, rebuild_tree=True, enable_pseudo_ops=True)
        diff = TraceDiff(tree1, tree2)
        stats = diff.get_diff_stats()
        assert isinstance(stats, pd.DataFrame)


class TestAnalysisUtilsPhase7:
    def test_efficiency_and_fusion_branches(self, tmp_path):
        row = pd.Series(
            {
                "FLOPS/Byte": 2.0,
                "TFLOPS/s_mean": 50.0,
                "TB/s_mean": 0.1,
                "Roofline Bound": "COMPUTE_BOUND",
                "Compute Spec": "matrix_fp16",
            }
        )
        eff = au.calculate_efficiency(
            row, peak_maf_or_maf_dict={"matrix_fp16": 100.0}, peak_hbm_bw=5300
        )
        assert eff["bound_type"] == "compute"

        cat_dir = tmp_path / "category_data"
        cat_dir.mkdir()
        (cat_dir / "kernel_fusion_metrics.json").write_text(
            json.dumps(
                {
                    "impact_estimates": [
                        {
                            "candidate_id": "c1",
                            "impact_score": 5.0,
                            "impact_score_low": 3.0,
                            "impact_score_high": 8.0,
                            "confidence": "high",
                        }
                    ]
                }
            )
        )
        loaded = au._load_fusion_map(str(tmp_path))
        assert isinstance(loaded, dict)

        ops = [
            {
                "kernel_names": ["a", "b"],
                "base_name": "Block",
                "instance_count": 3,
                "kernel_type_signature": ["GEMM", "elementwise"],
            }
        ]
        assert len(kfa._filter_and_dedup(ops)) >= 1


class TestJaxLlamaPhase7:
    def test_jax_llama_helpers_full(self, tmp_path):
        path = _write_gz_trace(tmp_path, _jax_llama_trace_events())
        trace = jax_cmp.load_trace(path)
        evs = jax_cmp.extract_gpu_events(trace, gpu_index=0)
        assert len(evs) > 0
        assert jax_cmp.percentile([1, 2, 3, 4], 50) == 2.5
        assert jax_cmp.mk_stats([10, 20]).total_us == 30
        d_model, head_dim, gsu = jax_cmp.infer_params(evs)
        assert d_model == 4096
        stream = [e for e in evs if e.tid == evs[0].tid]
        starts = jax_cmp.token_start_times(stream, "te_layernorm_forward")
        stage_avg, stage_share, per_layer, per_token, notes = (
            jax_cmp.compute_stage_table(stream, starts, (0, 0), (0, 1))
        )
        assert per_layer > 0
        assert jax_cmp.is_loop_multiply_fusion(
            jax_cmp.Event(
                1, 10, 0, 10, "loop_multiply_fusion", {"hlo_op": "loop_multiply_fusion"}
            )
        )


class TestReportingPhase7:
    def test_compare_perf_reports_all_sheets(self, tmp_path):
        from TraceLens.Reporting.generate_perf_report_pytorch import (
            generate_perf_report_pytorch,
        )

        t1 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)], "t1.json")
        t2 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 120)], "t2.json")
        d1 = tmp_path / "r1"
        d2 = tmp_path / "r2"
        generate_perf_report_pytorch(
            profile_json_path=str(t1),
            output_csvs_dir=str(d1),
            output_xlsx_path=str(tmp_path / "r1.xlsx"),
        )
        generate_perf_report_pytorch(
            profile_json_path=str(t2),
            output_csvs_dir=str(d2),
            output_xlsx_path=str(tmp_path / "r2.xlsx"),
        )
        out = tmp_path / "cmp"
        generate_compare_perf_reports_pytorch(
            reports=[str(d1), str(d2)],
            output=str(tmp_path / "cmp.xlsx"),
            sheets=["gpu_timeline", "ops_summary"],
            output_csvs_dir=str(out),
        )
        assert (out / "gpu_timeline.csv").exists()

    def test_pftrace_analyser_extended(self, tmp_path):
        events = _minimal_pftrace_events()
        analyser = PftraceHipActivityAnalyzer(events)
        assert isinstance(analyser.get_df_category_summary(), pd.DataFrame)
        assert isinstance(analyser.get_df_kernel_summary(), pd.DataFrame)
        assert isinstance(analyser.get_df_hip_summary(), pd.DataFrame)

    def test_tracediff_extension_multi_kernel_row(self):
        diff = pd.DataFrame(
            {
                "source": ["trace1", "trace1"],
                "lowest_common_ancestor_id": [5, 5],
                "lowest_common_ancestor_name": ["aten::mm", "aten::mm"],
                "cpu_op_name": ["aten::mm", "aten::add"],
                "busy_time": [100.0, 50.0],
                "name": ["k1", "k2"],
                "gpu_op_uid": [1, 2],
                "nn_module_stack": ["[]", "[]"],
                "nn_module_parent": ["", ""],
                "Input Dims": ["[[2,3]]", "[[2,3]]"],
                "Input type": ["['fp16']", "['fp16']"],
                "Input Strides": ["[]", "[]"],
                "Concrete Inputs": ["", ""],
            }
        )
        summary = tracediff_perf_summary_from_diff_stats(diff)
        assert " | " in summary.iloc[0]["name"]

    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(INFERENCE_ROOT, "sglang_decode")),
        reason="inference fixture missing",
    )
    def test_capture_merge_inference_fixture(self):
        case = os.path.join(INFERENCE_ROOT, "sglang_decode")
        trace_gz = next(f for f in os.listdir(case) if f.endswith(".json.gz"))
        graph = os.path.join(case, trace_gz)
        capture = os.path.join(case, "capture_traces")
        metadata = os.path.join(capture, "execution_details.json")
        merged = merge_capture_trace_into_graph(capture, metadata, graph)
        analyzer = TreePerfAnalyzer(merged, rebuild_tree=False, enable_pseudo_ops=True)
        unified = analyzer.build_df_unified_perf_table(include_nccl=False)
        assert isinstance(unified, pd.DataFrame)

    def test_collective_report_strict_and_heatmap(self, tmp_path):
        for rank in (0, 1):
            (tmp_path / f"rank{rank}_trace.json").write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "ph": "X",
                                "cat": "kernel",
                                "name": "ncclKernel_AllReduce",
                                "pid": rank,
                                "tid": 3,
                                "ts": 1000 + rank,
                                "dur": 40,
                                "args": {
                                    "External id": 10 + rank,
                                    "Collective name": "allreduce",
                                    "stream": 3,
                                    "collective_id": rank,
                                },
                            }
                        ]
                    }
                )
            )
        dfs = generate_collective_report(
            trace_dir=str(tmp_path),
            world_size=2,
            output_csvs_dir=str(tmp_path / "coll"),
            use_multiprocessing=False,
            strict_world_size_check=False,
            all2allv_heatmap=True,
        )
        assert isinstance(dfs, dict)


class TestReportingCliPhase7:
    def test_pftrace_hip_activity_main(self, tmp_path):
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_pftrace_hip_activity"
        )
        trace_path = tmp_path / "pf.json"
        trace_path.write_text(json.dumps({"traceEvents": _minimal_pftrace_events()}))
        old_argv = sys.argv
        sys.argv = [
            "generate_perf_report_pftrace_hip_activity",
            "--trace_path",
            str(trace_path),
            "--output_csvs_dir",
            str(tmp_path / "csv"),
            "--merge_kernels",
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert (tmp_path / "csv" / "category_summary.csv").exists()

    def test_pftrace_memory_copy_main(self, tmp_path):
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_pftrace_memory_copy"
        )
        events = [
            _mk_event("gpu_memcpy", "MemcpyHtoD", 1000, 20, 0, 1, {"bytes": 4096}),
            _mk_event("gpu_memcpy", "MemcpyDtoH", 1100, 15, 0, 1, {"bytes": 2048}),
        ]
        trace_path = tmp_path / "pf.json"
        trace_path.write_text(json.dumps({"traceEvents": events}))
        old_argv = sys.argv
        sys.argv = [
            "generate_perf_report_pftrace_memory_copy",
            "--trace_path",
            str(trace_path),
            "--output_csvs_dir",
            str(tmp_path / "csv"),
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert (tmp_path / "csv" / "memory_copy_by_copy_bytes.csv").exists()
