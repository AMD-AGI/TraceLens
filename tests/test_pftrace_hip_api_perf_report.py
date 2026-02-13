###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import os
import pytest
import tempfile
import pandas as pd
from pathlib import Path

from TraceLens.util import PftraceParser
from TraceLens.Reporting.pftrace_hip_api_analysis import PftraceHipApiAnalyzer
from TraceLens.Reporting.generate_perf_report_pftrace_hip_api import (
    generate_perf_report_pftrace_hip_api,
)


def _make_minimal_pftrace_events():
    """Minimal traceEvents: one hip_api launch and one kernel_dispatch with same corr_id."""
    corr_id = 42
    # ts/dur in microseconds in Perfetto; script converts to ns (ts*1000, dur*1000)
    api_ts, api_dur = 1000.0, 10.0  # 10 µs API
    kern_ts, kern_dur = 1020.0, 50.0  # kernel starts 10µs after API end, runs 50µs
    return [
        {
            "ph": "X",
            "cat": "hip_api",
            "name": "hipLaunchKernelGGL",
            "pid": 100,
            "tid": 1,
            "ts": api_ts,
            "dur": api_dur,
            "args": {"corr_id": corr_id},
        },
        {
            "ph": "X",
            "cat": "kernel_dispatch",
            "name": "my_kernel",
            "pid": 0,
            "tid": 7,
            "ts": kern_ts,
            "dur": kern_dur,
            "args": {
                "corr_id": corr_id,
                "kernel_name": "my_kernel",
            },
        },
    ]


class TestPftraceParser:
    """Test suite for PftraceParser."""

    def test_load_pftrace_data_valid_json(self):
        """Load valid traceEvents JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": _make_minimal_pftrace_events()}, f)
            path = f.name
        try:
            data = PftraceParser.load_pftrace_data(path)
            assert "traceEvents" in data
            assert len(data["traceEvents"]) == 2
        finally:
            os.unlink(path)

    def test_load_pftrace_data_missing_trace_events(self):
        """Invalid JSON without traceEvents raises."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"other": []}, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="missing 'traceEvents'"):
                PftraceParser.load_pftrace_data(path)
        finally:
            os.unlink(path)

    def test_load_pftrace_data_wrong_extension(self):
        """Non-.json/.json.gz path raises."""
        with pytest.raises(ValueError, match="expects .json or .json.gz"):
            PftraceParser.load_pftrace_data("/tmp/trace.pftrace")

    def test_get_events(self):
        """get_events returns traceEvents list."""
        data = {"traceEvents": [{"name": "a"}, {"name": "b"}]}
        events = PftraceParser.get_events(data)
        assert events == [{"name": "a"}, {"name": "b"}]


class TestPftraceHipApiAnalyzer:
    """Test suite for PftraceHipApiAnalyzer."""

    def test_get_df_api_kernel_summary(self):
        """Analyzer produces api_kernel_summary DataFrame with expected columns."""
        events = _make_minimal_pftrace_events()
        analyzer = PftraceHipApiAnalyzer(events)
        df = analyzer.get_df_api_kernel_summary()

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == 1
        assert "PID" in df.columns
        assert "TID" in df.columns
        assert "DevId" in df.columns
        assert "Count" in df.columns
        assert "QCount" in df.columns
        assert "TAvg_ns" in df.columns
        assert "AAvg_ns" in df.columns
        assert "QAvg_ns" in df.columns
        assert "KAvg_ns" in df.columns
        assert "API Name" in df.columns
        assert "Kernel Name" in df.columns

        row = df.iloc[0]
        assert row["API Name"] == "hipLaunchKernelGGL"
        assert row["Kernel Name"] == "my_kernel"
        assert row["Count"] == 1
        assert row["KAvg_ns"] == 50_000  # 50 µs in ns

    def test_exclude_kernel_regex(self):
        """Excluded kernel names are omitted from summary."""
        import re

        events = _make_minimal_pftrace_events()
        events[1]["args"]["kernel_name"] = "redzone_checker_kernel"
        analyzer = PftraceHipApiAnalyzer(
            events, exclude_kernel_re=re.compile(r"redzone_checker")
        )
        df = analyzer.get_df_api_kernel_summary()
        assert df.empty


class TestGeneratePerfReportPftraceHipApi:
    """Test suite for generate_perf_report_pftrace_hip_api."""

    def test_generate_excel_report(self):
        """Generate Excel report from JSON trace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "trace.json")
            with open(trace_path, "w") as f:
                json.dump({"traceEvents": _make_minimal_pftrace_events()}, f)
            output_path = os.path.join(tmpdir, "report.xlsx")

            dfs = generate_perf_report_pftrace_hip_api(
                trace_path=trace_path,
                output_xlsx_path=output_path,
                exclude_kernel_regex=None,
            )

            assert os.path.exists(output_path)
            assert isinstance(dfs, dict)
            assert "api_kernel_summary" in dfs
            assert len(dfs["api_kernel_summary"]) == 1

    def test_generate_csv_reports(self):
        """Generate CSV reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "trace.json")
            with open(trace_path, "w") as f:
                json.dump({"traceEvents": _make_minimal_pftrace_events()}, f)

            dfs = generate_perf_report_pftrace_hip_api(
                trace_path=trace_path,
                output_csvs_dir=tmpdir,
                exclude_kernel_regex=None,
            )

            assert "api_kernel_summary.csv" in os.listdir(tmpdir)
            assert "api_kernel_summary" in dfs

    def test_library_usage_no_output_path(self):
        """Call without output path still returns DataFrames and writes default xlsx."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "trace.json")
            with open(trace_path, "w") as f:
                json.dump({"traceEvents": _make_minimal_pftrace_events()}, f)

            dfs = generate_perf_report_pftrace_hip_api(trace_path=trace_path)

            default_xlsx = os.path.join(tmpdir, "trace_pftrace_hip_api_report.xlsx")
            assert os.path.exists(default_xlsx)
            assert "api_kernel_summary" in dfs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
