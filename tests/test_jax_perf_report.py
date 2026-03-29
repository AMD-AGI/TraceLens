###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
End-to-end tests for JAX perf report generation.

Verifies that generate_perf_report_jax runs successfully on real JAX traces
and matches checked-in CSV reference directories (one .csv per sheet), when
present alongside the trace tree.

Usage:
    pytest tests/test_jax_perf_report.py -v
"""

import glob
import os
import shutil
import tempfile

import pytest

from TraceLens.Reporting.generate_perf_report_jax import generate_perf_report_jax

from conftest import (
    compare_cols,
    format_diff_details,
    list_perf_report_csv_sheets,
    read_perf_report_csv,
)

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


def jax_ref_perf_report_csv_dir(trace_path):
    """
    Reference CSV directory for a trace under tests/traces/.../folder/foo.xplane.pb:
    tests/traces/.../folder_perf_report_csvs/
    """
    trace_dir = os.path.dirname(os.path.abspath(trace_path))
    parent = os.path.dirname(trace_dir)
    folder = os.path.basename(trace_dir)
    return os.path.join(parent, f"{folder}_perf_report_csvs")


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
# Fixture — run report once per trace_path (CSV output only), cache results
# ---------------------------------------------------------------------------

_report_cache = {}


@pytest.fixture(scope="module", autouse=True)
def _cleanup_report_cache():
    """Remove tmpdirs created by jax_report after all tests in this module."""
    yield
    for entry in _report_cache.values():
        shutil.rmtree(entry["tmpdir"], ignore_errors=True)


@pytest.fixture()
def jax_report(trace_path):
    """Run generate_perf_report_jax once per trace_path and cache the results."""
    if trace_path not in _report_cache:
        tmpdir = tempfile.mkdtemp()
        try:
            dict_name2df = generate_perf_report_jax(
                profile_path=trace_path,
                output_csvs_dir=tmpdir,
            )
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        _report_cache[trace_path] = {
            "dict_name2df": dict_name2df,
            "csv_dir": tmpdir,
            "tmpdir": tmpdir,
        }
    return _report_cache[trace_path]


# ---------------------------------------------------------------------------
# Tests (parametrized over all discovered JAX traces)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trace_path", JAX_TRACES, ids=[_short_id(t) for t in JAX_TRACES]
)
class TestJaxPerfReportE2E:
    """End-to-end tests: generate_perf_report_jax runs without error."""

    def test_generate_report_csvs(self, trace_path, jax_report):
        """CSV reports are created in the output dir, one per sheet."""
        csv_dir = jax_report["csv_dir"]
        dict_name2df = jax_report["dict_name2df"]
        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        assert len(csv_files) > 0, "No CSV files created"
        assert len(csv_files) == len(dict_name2df), (
            f"CSV count ({len(csv_files)}) " f"!= DataFrame count ({len(dict_name2df)})"
        )

    def test_expected_core_sheets(self, trace_path, jax_report):
        """Report contains the expected core sheets and they are non-empty."""
        dict_name2df = jax_report["dict_name2df"]
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

    def test_kernel_launchers_structure(self, trace_path, jax_report):
        """kernel_launchers sheet has rows and the expected columns."""
        df = jax_report["dict_name2df"]["kernel_launchers"]
        assert len(df) > 0, "kernel_launchers has no rows"

        expected_cols = [
            "name",
            "op category",
            "total_direct_kernel_time",
            "direct_kernel_count",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: '{col}'"


@pytest.mark.parametrize(
    "trace_path", JAX_TRACES, ids=[_short_id(t) for t in JAX_TRACES]
)
def test_jax_perf_report_csv_regression(trace_path, tmp_path, tol=1e-6):
    """
    When a sibling ``<trace_folder>_perf_report_csvs/`` directory exists under
    tests/traces, generated CSVs must match it (regression).
    """
    ref_dir = jax_ref_perf_report_csv_dir(trace_path)
    if not os.path.isdir(ref_dir):
        pytest.skip(f"No CSV reference directory: {ref_dir}")
    out_dir = str(tmp_path / "jax_perf_report_csvs")
    generate_perf_report_jax(profile_path=trace_path, output_csvs_dir=out_dir)

    sheets = list_perf_report_csv_sheets(ref_dir)
    assert sheets, f"Reference directory has no CSV files: {ref_dir}"

    for sheet in sheets:
        df_ref = read_perf_report_csv(ref_dir, sheet)
        df_fn = read_perf_report_csv(out_dir, sheet)
        if df_ref.empty:
            continue
        cols = [c for c in df_ref.columns if c in df_fn.columns]
        diff_cols = compare_cols(df_fn, df_ref, cols, tol=tol)
        assert (
            not diff_cols
        ), f"Sheet '{sheet}' has differences for {trace_path}:{format_diff_details(diff_cols)}"
