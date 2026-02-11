###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
End-to-end tests for JAX perf report generation.

Verifies that generate_perf_report_jax runs successfully on real JAX traces
and produces non-empty output. This catches breaking changes anywhere in
the JAX reporting pipeline (XPlane parsing, tree building, perf analysis,
report writing).

Usage:
    pytest tests/test_jax_perf_report.py -v
"""

import os
import glob
import tempfile

import pytest
import pandas as pd

from TraceLens.Reporting.generate_perf_report_jax import generate_perf_report_jax

# ---------------------------------------------------------------------------
# Test-trace discovery
# ---------------------------------------------------------------------------

TRACES_ROOT = os.path.join(os.path.dirname(__file__), "traces")


def find_jax_traces(root=TRACES_ROOT):
    """Walk the test traces directory and return all .xplane.pb files."""
    traces = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".xplane.pb"):
                traces.append(os.path.join(dirpath, fname))
    return sorted(traces)


JAX_TRACES = find_jax_traces()

# Guard: skip the entire module if no JAX traces are present
if not JAX_TRACES:
    pytest.skip(
        "No .xplane.pb traces found under tests/traces/", allow_module_level=True
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short_id(path):
    """Return a short test ID from the trace path, e.g. 'jax_conv_minimal'."""
    return os.path.basename(os.path.dirname(path))


# ---------------------------------------------------------------------------
# Tests (parametrized over all discovered JAX traces)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trace_path", JAX_TRACES, ids=[_short_id(t) for t in JAX_TRACES]
)
class TestJaxPerfReportE2E:
    """End-to-end tests: generate_perf_report_jax runs without error."""

    def test_generate_report_xlsx(self, trace_path):
        """XLSX report is created and contains DataFrames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_xlsx = os.path.join(tmpdir, "perf_report.xlsx")
            dict_name2df = generate_perf_report_jax(
                profile_path=trace_path,
                output_xlsx_path=output_xlsx,
            )
            assert os.path.exists(output_xlsx), "XLSX file was not created"
            assert len(dict_name2df) > 0, "No DataFrames returned"

    def test_generate_report_csvs(self, trace_path):
        """CSV reports are created, one per sheet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csvs_dir = os.path.join(tmpdir, "csvs")
            dict_name2df = generate_perf_report_jax(
                profile_path=trace_path,
                output_csvs_dir=csvs_dir,
            )
            csv_files = glob.glob(os.path.join(csvs_dir, "*.csv"))
            assert len(csv_files) > 0, "No CSV files created"
            assert len(csv_files) == len(
                dict_name2df
            ), f"CSV count ({len(csv_files)}) != DataFrame count ({len(dict_name2df)})"

    def test_expected_core_sheets(self, trace_path):
        """Report contains the expected core sheets and they are non-empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_xlsx = os.path.join(tmpdir, "perf_report.xlsx")
            dict_name2df = generate_perf_report_jax(
                profile_path=trace_path,
                output_xlsx_path=output_xlsx,
            )
            expected_sheets = [
                "gpu_timeline",
                "gpu_events_averages",
                "kernel_launchers",
                "kernel_launchers_summary",
                "kernel_launchers_summary_by_category",
            ]
            for sheet in expected_sheets:
                assert sheet in dict_name2df, f"Missing expected sheet: '{sheet}'"
                assert not dict_name2df[sheet].empty, f"Sheet '{sheet}' is empty"

    def test_kernel_launchers_structure(self, trace_path):
        """kernel_launchers sheet has rows and the expected columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_xlsx = os.path.join(tmpdir, "perf_report.xlsx")
            dict_name2df = generate_perf_report_jax(
                profile_path=trace_path,
                output_xlsx_path=output_xlsx,
            )
            df = dict_name2df["kernel_launchers"]
            assert len(df) > 0, "kernel_launchers has no rows"

            expected_cols = [
                "name",
                "op category",
                "total_direct_kernel_time",
                "direct_kernel_count",
            ]
            for col in expected_cols:
                assert col in df.columns, f"Missing column: '{col}'"
