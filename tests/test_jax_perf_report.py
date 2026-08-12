###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# xprof writes SSTABLE cache files next to each .xplane.pb; keep all JAX
# perf-report tests on one xdist worker so trace loads do not race.
pytestmark = pytest.mark.xdist_group("jax_traces")

from conftest import (
    compare_cols,
    format_diff_details,
    list_perf_report_csv_sheets,
    read_perf_report_csv,
    update_reference_csvs,
)
import importlib
import sys
from unittest.mock import MagicMock, patch
import pandas as pd
from TraceLens.TreePerf.tree_perf import JaxTreePerfAnalyzer
from tests.test_jax_analysis_report import _mock_side_inputs, _sample_averages_df
from tests.fixtures.traces import JAX_PB
from TraceLens.Reporting import compare_traces_jax_llama as jax_cmp
from tests.fixtures.reporting import _jax_llama_trace_events, _write_gz_trace

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
def test_jax_perf_report_csv_regression(
    trace_path, tmp_path, update_references, tol=1e-6
):
    """
    When a sibling ``<trace_folder>_perf_report_csvs/`` directory exists under
    tests/traces, generated CSVs must match it (regression).

    When ``--update-references`` is passed,
    the checked-in reference CSVs are overwritten with the freshly generated
    output and the test is skipped so the suite still returns green.
    """
    ref_dir = jax_ref_perf_report_csv_dir(trace_path)
    out_dir = str(tmp_path / "jax_perf_report_csvs")
    generate_perf_report_jax(profile_path=trace_path, output_csvs_dir=out_dir)

    if update_references:
        update_reference_csvs(out_dir, ref_dir)
        pytest.skip(f"Updated reference: {ref_dir}")
        return

    if not os.path.isdir(ref_dir):
        pytest.skip(f"No CSV reference directory: {ref_dir}")

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


# --- migrated from test_coverage_95_final.py ---


class TestJaxAnalysisMain:
    def test_jax_analysis_main(self, tmp_path):
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_jax_analysis"
        )
        categorized, xla_events = _mock_side_inputs()
        gemms = pd.DataFrame({"time ms": [1.0], "percent": [1.0]}, index=["gemm1"])
        gemms_detailed = pd.DataFrame({"name": ["gemm1"], "tflops": [1.0]})
        with patch.object(
            mod.JaxAnalyses,
            "summarize_gpu_events",
            return_value=(_sample_averages_df(), categorized, xla_events),
        ), patch.object(
            mod.JaxAnalyses,
            "summarize_gpu_gemm_events_from_pb",
            return_value=gemms,
        ), patch.object(
            mod.JaxAnalyses,
            "gemm_performance_from_pb",
            return_value=gemms_detailed,
        ):
            old_argv = sys.argv
            sys.argv = [
                "generate_perf_report_jax_analysis",
                "--profile_xplane_pb_path",
                "/fake/profile.xplane.pb",
                "--output_path",
                str(tmp_path),
                "--output_table_formats",
                ".csv",
            ]
            try:
                mod.main()
            finally:
                sys.argv = old_argv
        assert (tmp_path / "trace_analysis_results_gpu_events_averages.csv").exists()

    def test_jax_analysis_permission_error(self, tmp_path, monkeypatch):
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_jax_analysis"
        )
        bad_path = tmp_path / "nope" / "out"
        monkeypatch.setattr(
            mod.Path,
            "mkdir",
            MagicMock(side_effect=PermissionError("denied")),
        )
        with pytest.raises(SystemExit):
            mod.generate_perf_report_jax_analysis(
                "/fake.pb", str(bad_path), "out", [".csv"]
            )


# --- migrated from test_coverage_95_final.py ---


class TestJaxFromFile:
    def test_jax_analyzer_from_pb(self):
        analyzer = JaxTreePerfAnalyzer.from_file(profile_filepath=JAX_PB)
        assert analyzer.tree is not None
        timeline = analyzer.get_df_gpu_timeline()
        assert isinstance(timeline, pd.DataFrame)


# --- migrated from test_coverage_95_phase13.py ---


class TestJaxComparePhase13:
    def test_jax_llama_helpers(self, tmp_path):
        path = _write_gz_trace(tmp_path, _jax_llama_trace_events())
        trace = jax_cmp.load_trace(path)
        evs = jax_cmp.extract_gpu_events(trace, gpu_index=0)
        assert len(evs) > 0
        d_model, head_dim, gsu = jax_cmp.infer_params(evs)
        assert d_model == 4096


# --- migrated from test_reporting_cli_coverage.py ---


def test_jax_report_main(tmp_path):
    trace = os.path.join(
        os.path.dirname(__file__),
        "traces/mi300/jax_conv_minimal_legacy/chi-mi300x-013.ord.vultr.cpe.ice.amd.com.xplane.pb",
    )
    out = tmp_path / "jax.xlsx"
    import TraceLens.Reporting.generate_perf_report_jax as mod

    old_argv = sys.argv
    sys.argv = [
        "generate_perf_report_jax",
        "--profile_path",
        trace,
        "--output_xlsx_path",
        str(out),
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    assert out.exists()
