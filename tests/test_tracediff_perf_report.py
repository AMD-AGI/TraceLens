###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for TraceLens.Reporting.tracediff_comparison_extension.

Matching between ``diff_stats`` rows and ``unified_perf_summary`` rows is
driven exclusively by ``gpu_op_uid`` (with a canonical per-row uid derived as
the minimum gpu_op_uid in that row's ``kernel_details`` /
``kernel_details_summary``). There is no fallback to CPU-UID tree walks or to
``(name, args)`` string-key matching; tests below exercise that contract.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from TraceLens.Reporting.tracediff_comparison_extension import (
    _build_gpu_uid_to_canonical,
    _build_lca_metadata,
    _build_trace2_time_lookup,
    _enrich_sheet_with_trace2,
    _resolve_diff_row_to_key,
    _row_canonical_gpu_uid,
    _trace2_gpu_op_uid_set_for_lca,
    enrich_perf_report_dict_inplace,
    tracediff_perf_summary_from_diff_stats,
)


def _diff_to_summary_report(df: pd.DataFrame) -> dict:
    return {"tracediff_perf_summary": tracediff_perf_summary_from_diff_stats(df)}


def _enrich_perf_from_diff(diff_df, perf1) -> dict:
    return enrich_perf_report_dict_inplace(
        {k: v.copy() for k, v in perf1.items()},
        diff_df,
    )


# ---------------------------------------------------------------------------
# Helpers for building synthetic diff_stats DataFrames
# ---------------------------------------------------------------------------
def _make_diff_stats_row(
    name,
    cpu_op_name,
    source,
    kernel_time,
    lca_name,
    lca_id,
    input_dims="",
    input_strides="",
    input_type="",
    concrete_inputs="",
    nn_module_stack="root",
    nn_module_parent="root",
    cpu_op_uid=None,
    busy_time=None,
    thread_name="",
    gpu_op_uid=None,
):
    if cpu_op_uid is None:
        cpu_op_uid = (lca_id, name, source)
    row = {
        "name": name,
        "cpu_op_name": cpu_op_name,
        "cpu_op_uid": cpu_op_uid,
        "source": source,
        "thread_name": thread_name,
        "Input Dims": input_dims,
        "Input Strides": input_strides,
        "Input type": input_type,
        "Concrete Inputs": concrete_inputs,
        "kernel_time": kernel_time,
        "busy_time": busy_time,
        "lowest_common_ancestor_name": lca_name,
        "lowest_common_ancestor_id": lca_id,
        "nn_module_stack": nn_module_stack,
        "nn_module_parent": nn_module_parent,
    }
    if gpu_op_uid is not None:
        row["gpu_op_uid"] = gpu_op_uid
    return row


def _kd_list(*uids_and_durs):
    """Shorthand: _kd_list((uid, dur), (uid, dur), ...) → list[dict]."""
    return [
        {"gpu_op_uid": uid, "name": f"k_{uid}", "dur": float(dur)}
        for uid, dur in uids_and_durs
    ]


def _kds_str(*uids_and_durs):
    """Same as _kd_list but stringified (how unified_perf_summary round-trips through Excel)."""
    return str(_kd_list(*uids_and_durs))


# ---------------------------------------------------------------------------
# Tests for tracediff_perf_summary_from_diff_stats (unchanged pre-refactor)
# ---------------------------------------------------------------------------
class TestSingleLCASingleKernel:
    """One LCA ID, one kernel from each trace."""

    @pytest.fixture
    def report(self):
        rows = [
            _make_diff_stats_row(
                name="gemm_kernel_v1",
                cpu_op_name="aten::mm",
                source="trace1",
                kernel_time=100.0,
                lca_name="aten::mm",
                lca_id=10,
                busy_time=100.0,
            ),
            _make_diff_stats_row(
                name="gemm_kernel_v2",
                cpu_op_name="aten::mm",
                source="trace2",
                kernel_time=80.0,
                lca_name="aten::mm",
                lca_id=10,
                busy_time=80.0,
            ),
        ]
        df = pd.DataFrame(rows)
        return _diff_to_summary_report(df)

    def test_single_row_in_summary(self, report):
        summary = report["tracediff_perf_summary"]
        assert len(summary) == 1

    def test_operation_name(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["name"] == "aten::mm"

    def test_kernel_times(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["kernel_time_trace1_us"] == 100.0
        assert summary.iloc[0]["kernel_time_trace2_us"] == 80.0

    def test_speedup(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(80.0 / 100.0)

    def test_delta(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["delta_us (trace2 - trace1)"] == pytest.approx(-20.0)

    def test_kernel_names(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["kernel_names_trace1"] == ["gemm_kernel_v1"]
        assert summary.iloc[0]["kernel_names_trace2"] == ["gemm_kernel_v2"]

    def test_kernel_counts(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["num_kernels_trace1"] == 1
        assert summary.iloc[0]["num_kernels_trace2"] == 1

    def test_percentage_sums_to_100(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary["pct_of_trace1_total (%)"].sum() == pytest.approx(100.0)
        assert summary["pct_of_trace2_total (%)"].sum() == pytest.approx(100.0)


class TestMultiStreamOverlap:
    @pytest.fixture
    def report(self):
        rows = [
            _make_diff_stats_row(
                "nhwcSliceCKernel",
                "aten::convolution_backward",
                "trace1",
                100.0,
                "aten::convolution_backward",
                50,
                busy_time=150.0,
            ),
            _make_diff_stats_row(
                "cutlass_dgrad",
                "aten::convolution_backward",
                "trace1",
                130.0,
                "aten::convolution_backward",
                50,
                busy_time=150.0,
            ),
            _make_diff_stats_row(
                "nhwcSliceCKernel",
                "aten::convolution_backward",
                "trace2",
                100.0,
                "aten::convolution_backward",
                50,
                busy_time=120.0,
            ),
            _make_diff_stats_row(
                "cutlass_dgrad",
                "aten::convolution_backward",
                "trace2",
                130.0,
                "aten::convolution_backward",
                50,
                busy_time=120.0,
            ),
        ]
        df = pd.DataFrame(rows)
        return _diff_to_summary_report(df)

    def test_uses_busy_time_not_kernel_sum(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["kernel_time_trace1_us"] == pytest.approx(150.0)
        assert summary.iloc[0]["kernel_time_trace2_us"] == pytest.approx(120.0)

    def test_speedup_uses_busy_time(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(
            120.0 / 150.0
        )

    def test_delta_uses_busy_time(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["delta_us (trace2 - trace1)"] == pytest.approx(-30.0)


class TestTrace2OnlyLCA:
    @pytest.fixture
    def report(self):
        rows = [
            _make_diff_stats_row(
                "new_kernel",
                "aten::new_op",
                "trace2",
                75.0,
                "aten::new_op",
                200,
                busy_time=75.0,
            ),
        ]
        df = pd.DataFrame(rows)
        return _diff_to_summary_report(df)

    def test_falls_back_to_trace2_name(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["name"] == "aten::new_op"

    def test_trace1_time_is_zero(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["kernel_time_trace1_us"] == 0.0

    def test_trace2_time(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["kernel_time_trace2_us"] == 75.0

    def test_speedup_is_nan(self, report):
        summary = report["tracediff_perf_summary"]
        assert math.isnan(summary.iloc[0]["speedup (trace2/trace1)"])


class TestEmptyInput:
    def test_empty_df_returns_empty_summary(self):
        result = _diff_to_summary_report(pd.DataFrame())
        assert "tracediff_perf_summary" in result
        assert result["tracediff_perf_summary"].empty


class TestNaNLCADropped:
    @pytest.fixture
    def report(self):
        rows = [
            _make_diff_stats_row(
                "valid_k",
                "aten::mm",
                "trace1",
                100.0,
                "aten::mm",
                10,
                busy_time=100.0,
            ),
            _make_diff_stats_row(
                "valid_k2",
                "aten::mm",
                "trace2",
                90.0,
                "aten::mm",
                10,
                busy_time=90.0,
            ),
            {
                "name": "orphan_kernel",
                "cpu_op_name": "aten::orphan",
                "source": "trace1",
                "Input Dims": "",
                "Input Strides": "",
                "Input type": "",
                "Concrete Inputs": "",
                "kernel_time": 50.0,
                "busy_time": 50.0,
                "lowest_common_ancestor_name": None,
                "lowest_common_ancestor_id": None,
                "nn_module_stack": "",
                "nn_module_parent": "",
            },
        ]
        df = pd.DataFrame(rows)
        return _diff_to_summary_report(df)

    def test_only_valid_lca_in_summary(self, report):
        summary = report["tracediff_perf_summary"]
        assert len(summary) == 1
        assert summary.iloc[0]["name"] == "aten::mm"


# ---------------------------------------------------------------------------
# Tests: helpers that compute canonical gpu_op_uid keys
# ---------------------------------------------------------------------------
class TestRowCanonicalGpuUid:
    def test_kernel_details_list_returns_min(self):
        row = pd.Series({"kernel_details": _kd_list((30, 100), (10, 50), (20, 75))})
        assert _row_canonical_gpu_uid(row) == 10

    def test_kernel_details_summary_string_returns_min(self):
        row = pd.Series({"kernel_details_summary": _kds_str((42, 100), (7, 50))})
        assert _row_canonical_gpu_uid(row) == 7

    def test_single_kernel(self):
        row = pd.Series({"kernel_details": _kd_list((99, 10))})
        assert _row_canonical_gpu_uid(row) == 99

    def test_missing_columns_returns_none(self):
        row = pd.Series({"name": "aten::mm"})
        assert _row_canonical_gpu_uid(row) is None

    def test_empty_list_returns_none(self):
        row = pd.Series({"kernel_details": []})
        assert _row_canonical_gpu_uid(row) is None

    def test_null_summary_returns_none(self):
        row = pd.Series({"kernel_details_summary": float("nan")})
        assert _row_canonical_gpu_uid(row) is None

    def test_prefers_kernel_details_summary_when_both_present(self):
        """unified_perf_summary normally has only kernel_details_summary, but if
        both columns exist, the summary form is preferred (iteration order)."""
        row = pd.Series(
            {
                "kernel_details_summary": _kds_str((5, 10)),
                "kernel_details": _kd_list((1, 10)),
            }
        )
        assert _row_canonical_gpu_uid(row) == 5


class TestBuildGpuUidToCanonical:
    def test_empty_frame(self):
        idx = _build_gpu_uid_to_canonical(pd.DataFrame())
        assert idx == {}

    def test_no_kernel_details_column(self):
        df = pd.DataFrame({"name": ["aten::mm"]})
        idx = _build_gpu_uid_to_canonical(df)
        assert idx == {}

    def test_maps_every_uid_to_row_canonical(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "kernel_details_summary": [_kds_str((5001, 100), (5002, 50))],
            }
        )
        idx = _build_gpu_uid_to_canonical(df)
        assert idx == {5001: 5001, 5002: 5001}

    def test_multiple_rows_distinct_canonicals(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm", "aten::relu"],
                "kernel_details_summary": [
                    _kds_str((200, 100), (100, 50)),
                    _kds_str((400, 10), (300, 5)),
                ],
            }
        )
        idx = _build_gpu_uid_to_canonical(df)
        assert idx[100] == 100
        assert idx[200] == 100
        assert idx[300] == 300
        assert idx[400] == 300


class TestResolveDiffRowToKey:
    def test_returns_none_when_index_empty(self):
        row = pd.Series({"gpu_op_uid": 1})
        assert _resolve_diff_row_to_key(row, {}) is None

    def test_returns_none_when_no_gpu_op_uid_col(self):
        row = pd.Series({"name": "kernel"})
        assert _resolve_diff_row_to_key(row, {1: 1}) is None

    def test_returns_none_for_null_uid(self):
        row = pd.Series({"gpu_op_uid": float("nan")})
        assert _resolve_diff_row_to_key(row, {1: 1}) is None

    def test_returns_canonical_on_match(self):
        row = pd.Series({"gpu_op_uid": 5002})
        assert _resolve_diff_row_to_key(row, {5001: 5001, 5002: 5001}) == 5001

    def test_returns_none_on_miss(self):
        row = pd.Series({"gpu_op_uid": 9999})
        assert _resolve_diff_row_to_key(row, {5001: 5001}) is None


class TestTrace2GpuOpUidSetForLca:
    def test_returns_empty_when_no_gpu_op_uid_col(self):
        df = pd.DataFrame(
            [_make_diff_stats_row("k", "op", "trace2", 10.0, "op", 1, busy_time=10.0)]
        )
        assert _trace2_gpu_op_uid_set_for_lca(df, 1) == set()

    def test_distinct_gpu_uids_for_lca(self):
        rows = [
            _make_diff_stats_row(
                "k1", "op", "trace2", 10.0, "op", 1, busy_time=10.0, gpu_op_uid=100
            ),
            _make_diff_stats_row(
                "k2", "op", "trace2", 5.0, "op", 1, busy_time=5.0, gpu_op_uid=200
            ),
            _make_diff_stats_row(
                "k3", "op", "trace2", 3.0, "op", 2, busy_time=3.0, gpu_op_uid=300
            ),
        ]
        df = pd.DataFrame(rows)
        assert _trace2_gpu_op_uid_set_for_lca(df, 1) == {100, 200}
        assert _trace2_gpu_op_uid_set_for_lca(df, 2) == {300}


# ---------------------------------------------------------------------------
# Tests: _build_trace2_time_lookup (gpu_op_uid only)
# ---------------------------------------------------------------------------
class TestBuildTrace2TimeLookup:
    def test_empty_df(self):
        lookup, lca_ids, lc = _build_trace2_time_lookup(pd.DataFrame())
        assert lookup == {} and lca_ids == {} and lc == {}

    def test_empty_canonical_map_returns_empty(self):
        rows = [
            _make_diff_stats_row(
                "k",
                "aten::mm",
                "trace1",
                100.0,
                "mm",
                10,
                busy_time=100.0,
                gpu_op_uid=1,
            ),
            _make_diff_stats_row(
                "k", "aten::mm", "trace2", 80.0, "mm", 10, busy_time=80.0, gpu_op_uid=2
            ),
        ]
        lookup, lca_ids, lc = _build_trace2_time_lookup(pd.DataFrame(rows))
        assert lookup == {}

    def test_single_lca_single_kernel(self):
        rows = [
            _make_diff_stats_row(
                "k1",
                "aten::mm",
                "trace1",
                100.0,
                "mm",
                10,
                busy_time=100.0,
                gpu_op_uid=1,
            ),
            _make_diff_stats_row(
                "k2", "aten::mm", "trace2", 80.0, "mm", 10, busy_time=80.0, gpu_op_uid=2
            ),
        ]
        canonical = {1: 1, 2: 2}
        lookup, _, lc = _build_trace2_time_lookup(
            pd.DataFrame(rows), gpu_uid_to_canonical=canonical
        )
        assert lookup == {1: 80.0}
        assert lc == {1: 1}

    def test_multiple_lcas_same_canonical_aggregate(self):
        rows = [
            _make_diff_stats_row(
                "k1", "aten::mm", "trace1", 50.0, "mm", 10, busy_time=50.0, gpu_op_uid=1
            ),
            _make_diff_stats_row(
                "k2", "aten::mm", "trace2", 40.0, "mm", 10, busy_time=40.0, gpu_op_uid=2
            ),
            _make_diff_stats_row(
                "k3", "aten::mm", "trace1", 60.0, "mm", 20, busy_time=60.0, gpu_op_uid=3
            ),
            _make_diff_stats_row(
                "k4", "aten::mm", "trace2", 45.0, "mm", 20, busy_time=45.0, gpu_op_uid=4
            ),
        ]
        # uids 1 and 3 (trace1) both resolve to canonical 1; t2 kernels distinct.
        canonical = {1: 1, 3: 1, 2: 2, 4: 4}
        lookup, _, lc = _build_trace2_time_lookup(
            pd.DataFrame(rows), gpu_uid_to_canonical=canonical
        )
        assert lookup == {1: 40.0 + 45.0}
        assert lc == {1: 2}

    def test_trace1_only_lca_records_zero_trace2_time(self):
        rows = [
            _make_diff_stats_row(
                "k1",
                "aten::mm",
                "trace1",
                100.0,
                "mm",
                10,
                busy_time=100.0,
                gpu_op_uid=1,
            ),
        ]
        canonical = {1: 1}
        lookup, _, lc = _build_trace2_time_lookup(
            pd.DataFrame(rows), gpu_uid_to_canonical=canonical
        )
        assert lookup == {1: 0.0}
        assert lc == {1: 0}

    def test_unmatched_gpu_uid_skipped(self):
        rows = [
            _make_diff_stats_row(
                "k1",
                "aten::mm",
                "trace1",
                100.0,
                "mm",
                10,
                busy_time=100.0,
                gpu_op_uid=999,
            ),
            _make_diff_stats_row(
                "k2",
                "aten::mm",
                "trace2",
                80.0,
                "mm",
                10,
                busy_time=80.0,
                gpu_op_uid=888,
            ),
        ]
        canonical = {1: 1}  # 999 and 888 not in map
        lookup, _, lc = _build_trace2_time_lookup(
            pd.DataFrame(rows), gpu_uid_to_canonical=canonical
        )
        assert lookup == {}


# ---------------------------------------------------------------------------
# Tests: _enrich_sheet_with_trace2
# ---------------------------------------------------------------------------
class TestEnrichSheetWithTrace2:
    def test_empty_sheet_returned_unchanged(self):
        df = pd.DataFrame()
        meta = {
            1: {
                "lca_ids": [1],
                "lca_names": ["mm"],
                "lca_total_kernel_time_trace1_us": 100.0,
                "lca_total_kernel_time_trace2_us": 80.0,
            }
        }
        result = _enrich_sheet_with_trace2(
            df, "Kernel Time (µs)_sum", lca_metadata=meta
        )
        assert result.empty

    def test_empty_metadata_returned_unchanged(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "kernel_details_summary": [_kds_str((1, 100.0))],
                "Kernel Time (µs)_sum": [100.0],
            }
        )
        result = _enrich_sheet_with_trace2(df, "Kernel Time (µs)_sum", lca_metadata={})
        assert "speedup (trace2/trace1)" not in result.columns

    def test_single_row_enrichment(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "kernel_details_summary": [_kds_str((1, 100.0))],
                "Kernel Time (µs)_sum": [100.0],
            }
        )
        meta = {
            1: {
                "lca_ids": [1],
                "lca_names": ["mm"],
                "lca_total_kernel_time_trace1_us": 100.0,
                "lca_total_kernel_time_trace2_us": 80.0,
            }
        }
        result = _enrich_sheet_with_trace2(
            df, "Kernel Time (µs)_sum", count_lookup={1: 3}, lca_metadata=meta
        )
        assert result.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(80.0 / 100.0)
        assert result.iloc[0]["delta_us (trace2 - trace1)"] == pytest.approx(-20.0)
        assert result.iloc[0]["lca_kernel_count_trace2"] == 3.0
        assert result.iloc[0]["lca_total_kernel_time_trace2_us"] == 80.0

    def test_unmatched_row_gets_nan(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "kernel_details_summary": [_kds_str((1, 100.0))],
                "Kernel Time (µs)_sum": [100.0],
            }
        )
        meta = {
            99: {
                "lca_ids": [99],
                "lca_names": ["other"],
                "lca_total_kernel_time_trace1_us": 10.0,
                "lca_total_kernel_time_trace2_us": 5.0,
            }
        }
        result = _enrich_sheet_with_trace2(
            df, "Kernel Time (µs)_sum", lca_metadata=meta
        )
        assert np.isnan(result.iloc[0]["speedup (trace2/trace1)"])

    def test_row_without_kernel_details_gets_nan(self):
        df = pd.DataFrame({"name": ["aten::mm"], "Kernel Time (µs)_sum": [100.0]})
        meta = {
            1: {
                "lca_ids": [1],
                "lca_names": ["mm"],
                "lca_total_kernel_time_trace1_us": 100.0,
                "lca_total_kernel_time_trace2_us": 80.0,
            }
        }
        result = _enrich_sheet_with_trace2(
            df, "Kernel Time (µs)_sum", lca_metadata=meta
        )
        assert np.isnan(result.iloc[0]["speedup (trace2/trace1)"])

    def test_columns_inserted_after_kernel_time(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "kernel_details_summary": [_kds_str((1, 50.0))],
                "GFLOPS": [100.0],
                "Kernel Time (µs)_sum": [50.0],
                "TFLOPS/s_mean": [10.0],
            }
        )
        meta = {
            1: {
                "lca_ids": [1],
                "lca_names": ["mm"],
                "lca_total_kernel_time_trace1_us": 50.0,
                "lca_total_kernel_time_trace2_us": 40.0,
            }
        }
        result = _enrich_sheet_with_trace2(
            df, "Kernel Time (µs)_sum", lca_metadata=meta
        )
        cols = list(result.columns)
        kt_idx = cols.index("Kernel Time (µs)_sum")
        assert cols[kt_idx + 1] == "speedup (trace2/trace1)"
        assert cols[kt_idx + 2] == "delta_us (trace2 - trace1)"
        assert cols[kt_idx + 3] == "lca_kernel_count_trace2"
        assert cols[kt_idx + 4] == "lca_id"
        assert cols[kt_idx + 5] == "lca_name"
        assert cols[kt_idx + 6] == "lca_total_kernel_time_trace1_us"
        assert cols[kt_idx + 7] == "lca_total_kernel_time_trace2_us"
        assert np.isnan(result.iloc[0]["lca_kernel_count_trace2"])
        assert result.iloc[0]["lca_total_kernel_time_trace1_us"] == pytest.approx(50.0)

    def test_lca_id_contains_semicolon_separated_list(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "kernel_details_summary": [_kds_str((1, 50.0))],
                "Kernel Time (µs)_sum": [50.0],
            }
        )
        meta = {
            1: {
                "lca_ids": [101, 102],
                "lca_names": ["mm_scope_a", "mm_scope_b"],
                "lca_total_kernel_time_trace1_us": 50.0,
                "lca_total_kernel_time_trace2_us": 40.0,
            }
        }
        result = _enrich_sheet_with_trace2(
            df,
            "Kernel Time (µs)_sum",
            count_lookup={1: 2},
            lca_metadata=meta,
        )
        assert "101" in str(result.iloc[0]["lca_id"])
        assert "102" in str(result.iloc[0]["lca_id"])
        assert ";" in str(result.iloc[0]["lca_id"])
        assert "mm_scope_a" in str(result.iloc[0]["lca_name"])
        assert "mm_scope_b" in str(result.iloc[0]["lca_name"])

    def test_multi_op_lca_uses_lca_totals_for_speedup(self):
        df = pd.DataFrame(
            {
                "name": ["aiter::fmha_v3_bwd"],
                "kernel_details_summary": [_kds_str((201, 500.0))],
                "Kernel Time (µs)_sum": [500.0],
            }
        )
        lca_meta = {
            201: {
                "lca_ids": [99],
                "lca_names": ["lca"],
                "lca_total_kernel_time_trace1_us": 525.0,
                "lca_total_kernel_time_trace2_us": 300.0,
            }
        }
        result = _enrich_sheet_with_trace2(
            df, "Kernel Time (µs)_sum", lca_metadata=lca_meta
        )
        assert result.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(300.0 / 525.0)
        assert result.iloc[0]["delta_us (trace2 - trace1)"] == pytest.approx(-200.0)
        assert result.iloc[0]["lca_id"] == "99"
        assert result.iloc[0]["lca_name"] == "lca"
        assert result.iloc[0]["lca_total_kernel_time_trace1_us"] == pytest.approx(525.0)
        assert result.iloc[0]["lca_total_kernel_time_trace2_us"] == pytest.approx(300.0)

    def test_zero_trace2_time_gives_zero_speedup(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "kernel_details_summary": [_kds_str((1, 100.0))],
                "Kernel Time (µs)_sum": [100.0],
            }
        )
        meta = {
            1: {
                "lca_ids": [1],
                "lca_names": ["mm"],
                "lca_total_kernel_time_trace1_us": 100.0,
                "lca_total_kernel_time_trace2_us": 0.0,
            }
        }
        result = _enrich_sheet_with_trace2(
            df, "Kernel Time (µs)_sum", lca_metadata=meta
        )
        assert result.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(0.0)
        assert result.iloc[0]["delta_us (trace2 - trace1)"] == pytest.approx(-100.0)

    def test_original_df_not_modified(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "kernel_details_summary": [_kds_str((1, 100.0))],
                "Kernel Time (µs)_sum": [100.0],
            }
        )
        original_cols = list(df.columns)
        meta = {
            1: {
                "lca_ids": [1],
                "lca_names": ["mm"],
                "lca_total_kernel_time_trace1_us": 100.0,
                "lca_total_kernel_time_trace2_us": 80.0,
            }
        }
        _enrich_sheet_with_trace2(df, "Kernel Time (µs)_sum", lca_metadata=meta)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# Tests: _build_lca_metadata
# ---------------------------------------------------------------------------
class TestBuildLCAMetadata:
    def test_empty_df(self):
        assert _build_lca_metadata(pd.DataFrame(), {1: 1}) == {}

    def test_empty_canonical_map(self):
        rows = [
            _make_diff_stats_row(
                "k", "op_a", "trace1", 10.0, "lca", 1, busy_time=10.0, gpu_op_uid=1
            ),
        ]
        assert _build_lca_metadata(pd.DataFrame(rows), {}) == {}

    def test_single_op_lca(self):
        rows = [
            _make_diff_stats_row(
                "k1",
                "aten::mm",
                "trace1",
                100.0,
                "mm",
                10,
                busy_time=100.0,
                gpu_op_uid=1,
            ),
            _make_diff_stats_row(
                "k2", "aten::mm", "trace2", 80.0, "mm", 10, busy_time=80.0, gpu_op_uid=2
            ),
        ]
        canonical = {1: 1, 2: 2}
        meta = _build_lca_metadata(pd.DataFrame(rows), canonical)
        assert meta[1]["lca_ids"] == [10]
        assert meta[1]["lca_names"] == ["mm"]
        assert meta[1]["lca_total_kernel_time_trace1_us"] == pytest.approx(100.0)
        assert meta[1]["lca_total_kernel_time_trace2_us"] == pytest.approx(80.0)

    def test_multi_op_lca(self):
        rows = [
            _make_diff_stats_row(
                "fill_k",
                "aten::fill_",
                "trace1",
                25.0,
                "lca",
                99,
                busy_time=525.0,
                gpu_op_uid=101,
            ),
            _make_diff_stats_row(
                "fmha_k",
                "aiter::fmha_v3_bwd",
                "trace1",
                500.0,
                "lca",
                99,
                busy_time=525.0,
                gpu_op_uid=201,
            ),
            _make_diff_stats_row(
                "flash_k",
                "flash_attn::bwd",
                "trace2",
                300.0,
                "lca",
                99,
                busy_time=300.0,
                gpu_op_uid=301,
            ),
        ]
        canonical = {101: 101, 201: 201, 301: 301}
        meta = _build_lca_metadata(pd.DataFrame(rows), canonical)
        assert meta[101]["lca_ids"] == [99]
        assert meta[201]["lca_ids"] == [99]
        assert meta[101]["lca_names"] == ["lca"]
        assert meta[101]["lca_total_kernel_time_trace1_us"] == pytest.approx(525.0)
        assert meta[101]["lca_total_kernel_time_trace2_us"] == pytest.approx(300.0)
        assert meta[201]["lca_total_kernel_time_trace1_us"] == pytest.approx(525.0)


# ---------------------------------------------------------------------------
# Tests: enrich_perf_report_dict_inplace end-to-end
# ---------------------------------------------------------------------------
class TestEnrichmentEndToEnd:
    @pytest.fixture
    def enriched_report(self):
        diff_rows = [
            _make_diff_stats_row(
                "gemm_k",
                "aten::mm",
                "trace1",
                200.0,
                "mm",
                10,
                busy_time=200.0,
                gpu_op_uid=1,
            ),
            _make_diff_stats_row(
                "gemm_k2",
                "aten::mm",
                "trace2",
                150.0,
                "mm",
                10,
                busy_time=150.0,
                gpu_op_uid=2,
            ),
        ]
        diff_df = pd.DataFrame(diff_rows)

        perf1 = {
            "gpu_timeline": pd.DataFrame({"type": ["total_time"], "time ms": [5.0]}),
            "GEMM": pd.DataFrame(
                {
                    "name": ["aten::mm"],
                    "kernel_details": [_kd_list((1, 200.0))],
                    "Kernel Time (µs)_sum": [200.0],
                    "TFLOPS/s_mean": [10.0],
                }
            ),
            "unified_perf_summary": pd.DataFrame(
                {
                    "name": ["aten::mm"],
                    "kernel_details_summary": [_kds_str((1, 200.0))],
                    "Kernel Time (µs)_sum": [200.0],
                }
            ),
        }
        return _enrich_perf_from_diff(diff_df, perf1)

    def test_unified_perf_summary_enriched(self, enriched_report):
        ups = enriched_report["unified_perf_summary"]
        assert "speedup (trace2/trace1)" in ups.columns
        assert ups.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(150.0 / 200.0)

    def test_gemm_sheet_not_enriched(self, enriched_report):
        gemm = enriched_report["GEMM"]
        assert "speedup (trace2/trace1)" not in gemm.columns
        assert gemm.iloc[0]["TFLOPS/s_mean"] == pytest.approx(10.0)

    def test_gpu_timeline_passthrough(self, enriched_report):
        tl = enriched_report["gpu_timeline"]
        assert "speedup (trace2/trace1)" not in tl.columns


class TestPassthroughWithoutMatchingInfo:
    """When no matching info is available, enrichment is a passthrough."""

    def test_empty_diff_df(self):
        perf1 = {"unified_perf_summary": pd.DataFrame({"name": ["aten::mm"]})}
        result = _enrich_perf_from_diff(pd.DataFrame(), perf1)
        assert "speedup (trace2/trace1)" not in result["unified_perf_summary"].columns

    def test_no_kernel_details_column(self):
        diff_rows = [
            _make_diff_stats_row(
                "k",
                "aten::mm",
                "trace1",
                100.0,
                "mm",
                10,
                busy_time=100.0,
                gpu_op_uid=1,
            ),
            _make_diff_stats_row(
                "k",
                "aten::mm",
                "trace2",
                80.0,
                "mm",
                10,
                busy_time=80.0,
                gpu_op_uid=2,
            ),
        ]
        perf1 = {
            "unified_perf_summary": pd.DataFrame(
                {"name": ["aten::mm"], "Kernel Time (µs)_sum": [100.0]}
            )
        }
        result = _enrich_perf_from_diff(pd.DataFrame(diff_rows), perf1)
        # Without kernel_details_summary in unified_perf_summary, there's no
        # canonical map → no enrichment happens.
        assert "speedup (trace2/trace1)" not in result["unified_perf_summary"].columns

    def test_no_gpu_op_uid_in_diff_stats(self):
        diff_rows = [
            _make_diff_stats_row(
                "k", "aten::mm", "trace1", 100.0, "mm", 10, busy_time=100.0
            ),
            _make_diff_stats_row(
                "k", "aten::mm", "trace2", 80.0, "mm", 10, busy_time=80.0
            ),
        ]
        perf1 = {
            "unified_perf_summary": pd.DataFrame(
                {
                    "name": ["aten::mm"],
                    "kernel_details_summary": [_kds_str((1, 100.0))],
                    "Kernel Time (µs)_sum": [100.0],
                }
            )
        }
        result = _enrich_perf_from_diff(pd.DataFrame(diff_rows), perf1)
        ups = result["unified_perf_summary"]
        # diff_stats rows have no gpu_op_uid → enrichment columns added but NaN.
        if "speedup (trace2/trace1)" in ups.columns:
            assert pd.isna(ups.iloc[0]["speedup (trace2/trace1)"])


class TestMultiOpLCAEnrichmentEndToEnd:
    """Multi-op LCA: perf sheets unchanged; unified summary uses LCA totals."""

    def test_multi_op_rows_get_lca_totals_and_nan_trace2(self):
        diff_rows = [
            _make_diff_stats_row(
                "fill_k",
                "aten::fill_",
                "trace1",
                25.0,
                "lca",
                99,
                busy_time=525.0,
                gpu_op_uid=101,
            ),
            _make_diff_stats_row(
                "fmha_k",
                "aiter::fmha_v3_bwd",
                "trace1",
                500.0,
                "lca",
                99,
                busy_time=525.0,
                gpu_op_uid=201,
            ),
            _make_diff_stats_row(
                "flash_k",
                "flash_attn::bwd",
                "trace2",
                300.0,
                "lca",
                99,
                busy_time=300.0,
                gpu_op_uid=301,
            ),
            _make_diff_stats_row(
                "gemm_k",
                "aten::mm",
                "trace1",
                200.0,
                "mm",
                10,
                busy_time=200.0,
                gpu_op_uid=1,
            ),
            _make_diff_stats_row(
                "gemm_k2",
                "aten::mm",
                "trace2",
                150.0,
                "mm",
                10,
                busy_time=150.0,
                gpu_op_uid=2,
            ),
        ]
        diff_df = pd.DataFrame(diff_rows)

        perf1 = {
            "ops_unique_args": pd.DataFrame(
                {
                    "name": ["aiter::fmha_v3_bwd", "aten::fill_", "aten::mm"],
                    "kernel_details": [
                        _kd_list((201, 500.0)),
                        _kd_list((101, 100.0)),
                        _kd_list((1, 200.0)),
                    ],
                    "total_direct_kernel_time_sum": [500.0, 100.0, 200.0],
                }
            ),
            "unified_perf_summary": pd.DataFrame(
                {
                    "name": ["aiter::fmha_v3_bwd", "aten::fill_", "aten::mm"],
                    "kernel_details_summary": [
                        _kds_str((201, 500.0)),
                        _kds_str((101, 100.0)),
                        _kds_str((1, 200.0)),
                    ],
                    "Kernel Time (µs)_sum": [500.0, 100.0, 200.0],
                }
            ),
        }
        enriched = _enrich_perf_from_diff(diff_df, perf1)

        ops = enriched["ops_unique_args"]
        assert ops[ops["name"] == "aiter::fmha_v3_bwd"].iloc[0][
            "total_direct_kernel_time_sum"
        ] == pytest.approx(500.0)
        assert ops[ops["name"] == "aten::fill_"].iloc[0][
            "total_direct_kernel_time_sum"
        ] == pytest.approx(100.0)

        ups = enriched["unified_perf_summary"]
        fmha_u = ups[ups["name"] == "aiter::fmha_v3_bwd"].iloc[0]
        fill_u = ups[ups["name"] == "aten::fill_"].iloc[0]
        mm_u = ups[ups["name"] == "aten::mm"].iloc[0]

        assert fmha_u["speedup (trace2/trace1)"] == pytest.approx(300.0 / 525.0)
        assert fill_u["speedup (trace2/trace1)"] == pytest.approx(300.0 / 525.0)
        assert fmha_u["lca_total_kernel_time_trace1_us"] == pytest.approx(525.0)
        assert fmha_u["lca_total_kernel_time_trace2_us"] == pytest.approx(300.0)
        assert fill_u["lca_total_kernel_time_trace1_us"] == pytest.approx(525.0)
        assert fill_u["lca_total_kernel_time_trace2_us"] == pytest.approx(300.0)
        assert fmha_u["lca_id"] == "99"
        assert fill_u["lca_id"] == "99"

        assert mm_u["lca_id"] == "10"
        assert mm_u["lca_name"] == "mm"
        assert mm_u["lca_total_kernel_time_trace1_us"] == pytest.approx(200.0)
        assert mm_u["lca_total_kernel_time_trace2_us"] == pytest.approx(150.0)
        assert mm_u["speedup (trace2/trace1)"] == pytest.approx(150.0 / 200.0)


class TestGpuOpUidWithPseudoOp:
    """Direct gpu_op_uid match works even when CPU op name differs completely."""

    def test_match_across_pseudo_op_name(self):
        diff_rows = [
            _make_diff_stats_row(
                "gemm_kernel",
                "pseudo::synthetic_mm",
                "trace1",
                200.0,
                "lca",
                10,
                busy_time=200.0,
                gpu_op_uid=7001,
            ),
            _make_diff_stats_row(
                "gemm_kernel_v2",
                "pseudo::synthetic_mm",
                "trace2",
                150.0,
                "lca",
                10,
                busy_time=150.0,
                gpu_op_uid=8001,
            ),
        ]
        diff_df = pd.DataFrame(diff_rows)

        perf1 = {
            "unified_perf_summary": pd.DataFrame(
                {
                    "name": ["aten::mm"],
                    "kernel_details_summary": [_kds_str((7001, 200.0))],
                    "Kernel Time (µs)_sum": [200.0],
                }
            ),
        }
        enriched = _enrich_perf_from_diff(diff_df, perf1)
        ups = enriched["unified_perf_summary"]
        assert ups.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(150.0 / 200.0)


# ---------------------------------------------------------------------------
# Integration test: synthetic traces through full pipeline
# ---------------------------------------------------------------------------
def _mk_event(cat, name, ts, dur, pid, tid, args=None):
    return {
        "ph": "X",
        "cat": cat,
        "name": name,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "dur": dur,
        "args": args or {},
    }


def _mk_ac2g(corr_id, pid, tid, ts, phase):
    evt = {
        "ph": phase,
        "id": corr_id,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "cat": "ac2g",
        "name": "ac2g",
    }
    if phase == "f":
        evt["bp"] = "e"
    return evt


def _build_synthetic_trace(kernel_specs):
    events = []
    ts = 1000
    corr_id = 100
    cpu_pid, cpu_tid = 100, 100
    gpu_pid, gpu_tid = 0, 7

    for cpu_op_name, kernel_name, kernel_dur in kernel_specs:
        cpu_op_ts = ts
        cpu_op_dur = 100

        events.append(
            _mk_event(
                "cpu_op",
                cpu_op_name,
                ts=cpu_op_ts,
                dur=cpu_op_dur,
                pid=cpu_pid,
                tid=cpu_tid,
                args={"Input Dims": [[32, 64]], "Input type": ["float"]},
            )
        )

        runtime_ts = cpu_op_ts + 10
        events.append(
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=runtime_ts,
                dur=5,
                pid=cpu_pid,
                tid=cpu_tid,
                args={"correlation": corr_id},
            )
        )

        kernel_ts = cpu_op_ts + 50
        events.append(
            _mk_event(
                "kernel",
                kernel_name,
                ts=kernel_ts,
                dur=kernel_dur,
                pid=gpu_pid,
                tid=gpu_tid,
                args={"correlation": corr_id, "stream": 7},
            )
        )

        events.append(
            _mk_ac2g(corr_id, pid=gpu_pid, tid=gpu_tid, ts=kernel_ts, phase="s")
        )
        events.append(
            _mk_ac2g(corr_id, pid=gpu_pid, tid=gpu_tid, ts=kernel_ts, phase="f")
        )

        ts += cpu_op_dur + 200
        corr_id += 1

    return {"traceEvents": events}


class TestIntegrationSyntheticTraces:
    @pytest.fixture(scope="class")
    def traces_and_report(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("traces")

        trace1_specs = [
            ("aten::mm", "gemm_kernel_rocm", 100),
            ("aten::relu", "relu_kernel_v1", 20),
            ("aten::add", "add_kernel_v1", 15),
        ]
        trace2_specs = [
            ("aten::mm", "gemm_kernel_cuda", 80),
            ("aten::relu", "relu_kernel_v2", 25),
            ("aten::add", "add_kernel_v2", 10),
        ]

        trace1_path = str(tmp / "trace1.json")
        trace2_path = str(tmp / "trace2.json")

        with open(trace1_path, "w") as f:
            json.dump(_build_synthetic_trace(trace1_specs), f)
        with open(trace2_path, "w") as f:
            json.dump(_build_synthetic_trace(trace2_specs), f)

        from TraceLens import TraceDiff, TreePerfAnalyzer
        from TraceLens.Reporting.generate_perf_report_pytorch import (
            generate_perf_report_pytorch,
        )

        perf1 = generate_perf_report_pytorch(trace1_path, collective_analysis=False)
        perf2 = generate_perf_report_pytorch(trace2_path, collective_analysis=False)

        pa1 = TreePerfAnalyzer.from_file(trace1_path, add_python_func=True)
        pa2 = TreePerfAnalyzer.from_file(trace2_path, add_python_func=True)

        td = TraceDiff(pa1.tree, pa2.tree)
        td.generate_tracediff_report()

        enriched = enrich_perf_report_dict_inplace(
            {k: v.copy() for k, v in perf1.items()},
            td.diff_stats_df,
        )
        summary = tracediff_perf_summary_from_diff_stats(td.diff_stats_df)

        return td, perf1, perf2, enriched, summary

    def test_tracediff_produced_diff_stats(self, traces_and_report):
        td, _, _, _, _ = traces_and_report
        assert not td.diff_stats_df.empty

    def test_summary_non_negative_times(self, traces_and_report):
        _, _, _, _, summary = traces_and_report
        if not summary.empty:
            for _, row in summary.iterrows():
                assert row["kernel_time_trace1_us"] >= 0
                assert row["kernel_time_trace2_us"] >= 0
                assert (
                    row["kernel_time_trace1_us"] > 0 or row["kernel_time_trace2_us"] > 0
                )

    def test_gpu_timeline_passthrough(self, traces_and_report):
        _, _, _, enriched, _ = traces_and_report
        assert "gpu_timeline" in enriched
