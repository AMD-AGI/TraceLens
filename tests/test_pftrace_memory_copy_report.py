###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import os
import tempfile
import pandas as pd

from TraceLens.Reporting.generate_perf_report_pftrace_memory_copy import (
    extract_memory_copy_rows,
    build_memory_copy_count_df,
    generate_perf_report_pftrace_memory_copy,
    _get_copy_bytes,
    _format_direction,
)


def _make_memory_copy_events():
    """Minimal traceEvents with memory_copy events (h2d, d2h, d2d) and copy_bytes."""
    return [
        {
            "ph": "X",
            "cat": "memory_copy",
            "name": "MEMORY_COPY_HOST_TO_DEVICE",
            "pid": 63721,
            "tid": 63882,
            "args": {
                "copy_bytes": 20138,
                "operation": 2,
                "src_agent": 0,
                "dst_agent": 2,
            },
        },
        {
            "ph": "X",
            "cat": "memory_copy",
            "name": "MEMORY_COPY_HOST_TO_DEVICE",
            "pid": 63721,
            "tid": 63882,
            "args": {"copy_bytes": 20138, "src_agent": 0, "dst_agent": 2},
        },
        {
            "ph": "X",
            "cat": "memory_copy",
            "name": "MEMORY_COPY_DEVICE_TO_HOST",
            "pid": 63721,
            "tid": 63882,
            "args": {"copy_bytes": 4096, "src_agent": 1, "dst_agent": 0},
        },
        {
            "ph": "X",
            "cat": "memory_copy",
            "name": "MEMORY_COPY_DEVICE_TO_DEVICE",
            "pid": 63721,
            "tid": 63882,
            "args": {"copy_bytes": 8192, "src_agent": 0, "dst_agent": 1},
        },
    ]


class TestMemoryCopyHelpers:
    def test_get_copy_bytes(self):
        assert _get_copy_bytes({"args": {"copy_bytes": 20138}}) == 20138
        assert _get_copy_bytes({"args": {"copy_bytes": "4096"}}) == 4096
        assert _get_copy_bytes({"args": {}}) is None
        assert _get_copy_bytes({"args": {"copy_bytes": None}}) is None


class TestFormatDirection:
    def test_h2d_d2h_d2d(self):
        assert (
            _format_direction(
                {"name": "MEMORY_COPY_HOST_TO_DEVICE", "args": {"dst_agent": 2}}
            )
            == "h2d (GPU 2)"
        )
        assert (
            _format_direction(
                {"name": "MEMORY_COPY_DEVICE_TO_HOST", "args": {"src_agent": 1}}
            )
            == "d2h (GPU 1)"
        )
        assert (
            _format_direction(
                {
                    "name": "MEMORY_COPY_DEVICE_TO_DEVICE",
                    "args": {"src_agent": 0, "dst_agent": 1},
                }
            )
            == "d2d (GPU 0 -> GPU 1)"
        )


class TestExtractMemoryCopyRows:
    def test_extract_returns_copy_bytes_and_direction(self):
        events = _make_memory_copy_events()
        rows = extract_memory_copy_rows(events)
        assert len(rows) == 4
        # Two H2D with copy_bytes 20138, dst_agent 2
        assert (20138, "h2d (GPU 2)") in rows
        assert sum(1 for r in rows if r == (20138, "h2d (GPU 2)")) == 2
        assert (4096, "d2h (GPU 1)") in rows
        assert (8192, "d2d (GPU 0 -> GPU 1)") in rows

    def test_extract_ignores_non_memory_copy(self):
        events = [{"cat": "hip_api", "name": "x", "args": {"copy_bytes": 100}}]
        assert extract_memory_copy_rows(events) == []

    def test_extract_ignores_memory_copy_without_copy_bytes(self):
        events = [
            {"cat": "memory_copy", "name": "MEMORY_COPY_HOST_TO_DEVICE", "args": {}}
        ]
        assert extract_memory_copy_rows(events) == []


class TestBuildMemoryCopyCountDf:
    def test_count_per_copy_bytes_and_direction(self):
        events = _make_memory_copy_events()
        df = build_memory_copy_count_df(events)
        assert list(df.columns) == ["copy_bytes", "direction", "count"]
        # 20138 h2d x2, 4096 d2h x1, 8192 d2d x1 -> 3 rows
        assert len(df) == 3
        row_20138 = df[(df["copy_bytes"] == 20138) & (df["direction"] == "h2d (GPU 2)")]
        assert len(row_20138) == 1 and row_20138["count"].iloc[0] == 2
        assert (
            df[(df["copy_bytes"] == 4096) & (df["direction"] == "d2h (GPU 1)")][
                "count"
            ].iloc[0]
            == 1
        )
        assert (
            df[
                (df["copy_bytes"] == 8192) & (df["direction"] == "d2d (GPU 0 -> GPU 1)")
            ]["count"].iloc[0]
            == 1
        )

    def test_empty_events(self):
        df = build_memory_copy_count_df([])
        assert df.empty and list(df.columns) == ["copy_bytes", "direction", "count"]


class TestGeneratePerfReportPftraceMemoryCopy:
    def test_generate_writes_excel_and_returns_dfs(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": _make_memory_copy_events()}, f)
            trace_path = f.name
        out_xlsx = trace_path + "_out.xlsx"
        try:
            dfs = generate_perf_report_pftrace_memory_copy(
                trace_path=trace_path,
                output_xlsx_path=out_xlsx,
            )
            assert "memory_copy_by_copy_bytes" in dfs
            assert len(dfs["memory_copy_by_copy_bytes"]) == 3
            assert list(dfs["memory_copy_by_copy_bytes"].columns) == [
                "copy_bytes",
                "direction",
                "count",
            ]
            assert os.path.isfile(out_xlsx)
        finally:
            os.unlink(trace_path)
            if os.path.isfile(out_xlsx):
                os.unlink(out_xlsx)

    def test_generate_csvs_dir(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": _make_memory_copy_events()}, f)
            trace_path = f.name
        with tempfile.TemporaryDirectory() as out_dir:
            try:
                dfs = generate_perf_report_pftrace_memory_copy(
                    trace_path=trace_path,
                    output_csvs_dir=out_dir,
                )
                csv_path = os.path.join(out_dir, "memory_copy_by_copy_bytes.csv")
                assert os.path.isfile(csv_path)
                df = pd.read_csv(csv_path)
                assert list(df.columns) == ["copy_bytes", "direction", "count"]
            finally:
                os.unlink(trace_path)
