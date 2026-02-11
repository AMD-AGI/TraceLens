###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for pftrace hip activity report (category, kernel, HIP API summaries)."""

import json
import os
import pytest
import tempfile
import pandas as pd

from TraceLens.util import PftraceParser
from TraceLens.Reporting.pftrace_hip_activity_analysis import (
    PftraceHipActivityAnalyzer,
    extract_time_ns,
    discover_gpus,
    classify,
    build_event_lists,
    build_hip_api_events,
)
from TraceLens.Reporting.generate_perf_report_pftrace_hip_activity import (
    generate_perf_report_pftrace_hip_activity,
)


def _minimal_trace_events_with_agent():
    """Trace with one GPU (agent), one kernel event, one hip_api event."""
    return [
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "xla_fusion_42",
            "pid": 0,
            "tid": 7,
            "ts": 1000,
            "dur": 50000,
            "args": {"agent": "gpu_0", "begin_ns": 1000000, "delta_ns": 50000000},
        },
        {
            "ph": "X",
            "cat": "hip_api",
            "name": "hipLaunchKernelGGL",
            "pid": 100,
            "tid": 1,
            "ts": 900,
            "dur": 20,
            "args": {"stream_ID": 0},
        },
    ]


class TestPftraceHipActivityAnalysis:
    def test_extract_time_ns_from_args(self):
        e = {"args": {"begin_ns": 1000, "delta_ns": 500}}
        ts, dur = extract_time_ns(e)
        assert ts == 1000
        assert dur == 500

    def test_extract_time_ns_from_ts_dur(self):
        e = {"ts": 1, "dur": 2}
        ts, dur = extract_time_ns(e)
        assert ts == 1000
        assert dur == 2000

    def test_discover_gpus(self):
        events = _minimal_trace_events_with_agent()
        agent_to_idx, agents = discover_gpus(events)
        assert "gpu_0" in agent_to_idx
        assert agents == ["gpu_0"]

    def test_classify(self):
        assert classify("ncclAllReduce") == "rccl"
        assert classify("Cijk_gemm") == "gemm"
        assert classify("xla_fusion_1") == "xla"

    def test_build_event_lists(self):
        events = _minimal_trace_events_with_agent()
        compute, rccl, xla_agg, used_fav3, agents = build_event_lists(
            events, merge_kernels=False, min_tid=-10**9, max_tid=10**9
        )
        assert len(agents) == 1
        assert len(compute[0]) == 1
        assert compute[0][0].name == "xla_fusion_42"
        assert "xla_fusion_42" in xla_agg or "xla_fusion_" in str(xla_agg)

    def test_build_hip_api_events(self):
        events = _minimal_trace_events_with_agent()
        hip = build_hip_api_events(events, min_tid=-10**9, max_tid=10**9)
        assert len(hip) == 1
        assert hip[0].name == "hipLaunchKernelGGL"


class TestPftraceHipActivityAnalyzer:
    def test_analyzer_returns_dataframes(self):
        events = _minimal_trace_events_with_agent()
        analyzer = PftraceHipActivityAnalyzer(events, min_event_ns=0)
        df_cat = analyzer.get_df_category_summary()
        assert isinstance(df_cat, pd.DataFrame)
        assert not df_cat.empty
        assert "GPU ID" in df_cat.columns
        assert "Category" in df_cat.columns
        df_xla = analyzer.get_df_xla_top(top_n=10)
        assert isinstance(df_xla, pd.DataFrame)
        df_kern = analyzer.get_df_kernel_summary()
        assert isinstance(df_kern, pd.DataFrame)
        df_hip = analyzer.get_df_hip_summary()
        assert isinstance(df_hip, pd.DataFrame)
        assert len(df_hip) == 1


class TestGeneratePerfReportPftraceHipActivity:
    def test_generate_excel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "trace.json")
            with open(trace_path, "w") as f:
                json.dump({"traceEvents": _minimal_trace_events_with_agent()}, f)
            out_xlsx = os.path.join(tmpdir, "report.xlsx")
            dfs = generate_perf_report_pftrace_hip_activity(
                trace_path=trace_path,
                output_xlsx_path=out_xlsx,
                min_event_ns=0,
            )
            assert os.path.exists(out_xlsx)
            assert "category_summary" in dfs
            assert "xla_top" in dfs
            assert "kernel_summary" in dfs
            assert "hip_summary" in dfs

    def test_generate_csv_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "trace.json")
            with open(trace_path, "w") as f:
                json.dump({"traceEvents": _minimal_trace_events_with_agent()}, f)
            dfs = generate_perf_report_pftrace_hip_activity(
                trace_path=trace_path,
                output_csvs_dir=tmpdir,
                min_event_ns=0,
            )
            assert "category_summary.csv" in os.listdir(tmpdir)
            assert "hip_summary.csv" in os.listdir(tmpdir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
