###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for TraceLens.Reporting.tracediff_comparison_extension.

Tests tracediff_perf_summary_from_diff_stats with synthetic diff_stats DataFrames,
enrichment helpers, LCA consolidation, _find_perf_summary_matching_node, and
integration with synthetic traces.
"""

import json
import math

import pandas as pd
import pytest

from TraceLens.Reporting.tracediff_comparison_extension import (
    _apply_lca_consolidation,
    _build_gpu_op_uid_to_key_index,
    _build_lca_consolidation,
    _build_trace2_time_lookup,
    _enrich_sheet_with_trace2,
    _find_perf_summary_matching_node,
    _make_str_key,
    enrich_perf_report_dict_inplace,
    tracediff_perf_summary_from_diff_stats,
)


def _diff_to_summary_report(df: pd.DataFrame) -> dict:
    return {"tracediff_perf_summary": tracediff_perf_summary_from_diff_stats(df)}


def _enrich_perf_from_diff(diff_df, perf1, trace1_tree=None) -> dict:
    return enrich_perf_report_dict_inplace(
        {k: v.copy() for k, v in perf1.items()},
        diff_df,
        trace1_tree,
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


# ---------------------------------------------------------------------------
# Test: single LCA with one kernel per trace
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


# ---------------------------------------------------------------------------
# Test: multi-stream overlap — busy_time < sum(kernel_time)
# ---------------------------------------------------------------------------
class TestMultiStreamOverlap:
    """When kernels run on multiple streams, busy_time (merged intervals)
    is less than the sum of individual kernel durations."""

    @pytest.fixture
    def report(self):
        rows = [
            _make_diff_stats_row(
                name="nhwcSliceCKernel",
                cpu_op_name="aten::convolution_backward",
                source="trace1",
                kernel_time=100.0,
                lca_name="aten::convolution_backward",
                lca_id=50,
                busy_time=150.0,
            ),
            _make_diff_stats_row(
                name="cutlass_dgrad",
                cpu_op_name="aten::convolution_backward",
                source="trace1",
                kernel_time=130.0,
                lca_name="aten::convolution_backward",
                lca_id=50,
                busy_time=150.0,
            ),
            _make_diff_stats_row(
                name="nhwcSliceCKernel",
                cpu_op_name="aten::convolution_backward",
                source="trace2",
                kernel_time=100.0,
                lca_name="aten::convolution_backward",
                lca_id=50,
                busy_time=120.0,
            ),
            _make_diff_stats_row(
                name="cutlass_dgrad",
                cpu_op_name="aten::convolution_backward",
                source="trace2",
                kernel_time=130.0,
                lca_name="aten::convolution_backward",
                lca_id=50,
                busy_time=120.0,
            ),
        ]
        df = pd.DataFrame(rows)
        return _diff_to_summary_report(df)

    def test_uses_busy_time_not_kernel_sum(self, report):
        """busy_time (150) should be used, not sum(kernel_time) (230)."""
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


# ---------------------------------------------------------------------------
# Test: single LCA with multiple kernels (summing)
# ---------------------------------------------------------------------------
class TestSingleLCAMultipleKernels:
    """One LCA ID, multiple kernels per trace — busy_time used for totals."""

    @pytest.fixture
    def report(self):
        rows = [
            _make_diff_stats_row(
                name="bn_mean_kernel",
                cpu_op_name="aten::batch_norm",
                source="trace1",
                kernel_time=30.0,
                lca_name="aten::batch_norm",
                lca_id=42,
                busy_time=100.0,
            ),
            _make_diff_stats_row(
                name="bn_var_kernel",
                cpu_op_name="aten::batch_norm",
                source="trace1",
                kernel_time=50.0,
                lca_name="aten::batch_norm",
                lca_id=42,
                busy_time=100.0,
            ),
            _make_diff_stats_row(
                name="bn_norm_kernel",
                cpu_op_name="aten::batch_norm",
                source="trace1",
                kernel_time=20.0,
                lca_name="aten::batch_norm",
                lca_id=42,
                busy_time=100.0,
            ),
            _make_diff_stats_row(
                name="fused_bn_kernel",
                cpu_op_name="aten::native_batch_norm",
                source="trace2",
                kernel_time=60.0,
                lca_name="aten::batch_norm",
                lca_id=42,
                busy_time=60.0,
            ),
        ]
        df = pd.DataFrame(rows)
        return _diff_to_summary_report(df)

    def test_single_row_for_lca(self, report):
        summary = report["tracediff_perf_summary"]
        assert len(summary) == 1

    def test_trace1_time_is_sum(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["kernel_time_trace1_us"] == pytest.approx(100.0)

    def test_trace2_time(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["kernel_time_trace2_us"] == pytest.approx(60.0)

    def test_speedup_with_summed_times(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(60.0 / 100.0)

    def test_kernel_count_trace1(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["num_kernels_trace1"] == 3

    def test_kernel_count_trace2(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["num_kernels_trace2"] == 1


# ---------------------------------------------------------------------------
# Test: multiple LCA IDs with different cpu_op_names
# ---------------------------------------------------------------------------
class TestMultipleLCAs:
    """Multiple LCA IDs — each should produce its own row, sorted by trace1 time."""

    @pytest.fixture
    def report(self):
        rows = [
            # LCA 10: matmul (big, should sort first)
            _make_diff_stats_row(
                name="gemm_k",
                cpu_op_name="aten::mm",
                source="trace1",
                kernel_time=500.0,
                lca_name="aten::mm",
                lca_id=10,
                busy_time=500.0,
            ),
            _make_diff_stats_row(
                name="gemm_k_v2",
                cpu_op_name="aten::mm",
                source="trace2",
                kernel_time=400.0,
                lca_name="aten::mm",
                lca_id=10,
                busy_time=400.0,
            ),
            # LCA 20: relu (small)
            _make_diff_stats_row(
                name="relu_k",
                cpu_op_name="aten::relu",
                source="trace1",
                kernel_time=10.0,
                lca_name="aten::relu",
                lca_id=20,
                busy_time=10.0,
            ),
            _make_diff_stats_row(
                name="relu_k_v2",
                cpu_op_name="aten::relu",
                source="trace2",
                kernel_time=8.0,
                lca_name="aten::relu",
                lca_id=20,
                busy_time=8.0,
            ),
            # LCA 30: conv (medium)
            _make_diff_stats_row(
                name="conv_k",
                cpu_op_name="aten::conv2d",
                source="trace1",
                kernel_time=200.0,
                lca_name="aten::conv2d",
                lca_id=30,
                busy_time=200.0,
            ),
            _make_diff_stats_row(
                name="conv_k_v2",
                cpu_op_name="aten::conv2d",
                source="trace2",
                kernel_time=150.0,
                lca_name="aten::conv2d",
                lca_id=30,
                busy_time=150.0,
            ),
        ]
        df = pd.DataFrame(rows)
        return _diff_to_summary_report(df)

    def test_three_rows(self, report):
        summary = report["tracediff_perf_summary"]
        assert len(summary) == 3

    def test_sorted_by_trace1_time_descending(self, report):
        summary = report["tracediff_perf_summary"]
        times = summary["kernel_time_trace1_us"].tolist()
        assert times == sorted(times, reverse=True)

    def test_op_names(self, report):
        summary = report["tracediff_perf_summary"]
        names = summary["name"].tolist()
        assert "aten::mm" in names
        assert "aten::relu" in names
        assert "aten::conv2d" in names

    def test_percentage_columns_sum_to_100(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary["pct_of_trace1_total (%)"].sum() == pytest.approx(100.0)
        assert summary["pct_of_trace2_total (%)"].sum() == pytest.approx(100.0)

    def test_cumulative_pct_ends_at_100(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary["cumulative_pct_trace1 (%)"].iloc[-1] == pytest.approx(100.0)

    def test_individual_percentages(self, report):
        summary = report["tracediff_perf_summary"]
        total_t1 = 500.0 + 10.0 + 200.0
        mm_row = summary[summary["name"] == "aten::mm"].iloc[0]
        assert mm_row["pct_of_trace1_total (%)"] == pytest.approx(
            500.0 / total_t1 * 100
        )


# ---------------------------------------------------------------------------
# Test: multiple cpu_op_names under one LCA (pipe-separated)
# ---------------------------------------------------------------------------
class TestMultipleCpuOpsPerLCA:
    """When trace1 has multiple distinct cpu_op_name values for one LCA,
    they should be joined with ' | '."""

    @pytest.fixture
    def report(self):
        rows = [
            _make_diff_stats_row(
                name="conv_k_miopen",
                cpu_op_name="aten::miopen_convolution",
                source="trace1",
                kernel_time=50.0,
                lca_name="aten::miopen_convolution | aten::cudnn_convolution",
                lca_id=92,
                busy_time=80.0,
            ),
            _make_diff_stats_row(
                name="conv_k_miopen_2",
                cpu_op_name="aten::miopen_convolution",
                source="trace1",
                kernel_time=30.0,
                lca_name="aten::miopen_convolution | aten::cudnn_convolution",
                lca_id=92,
                busy_time=80.0,
            ),
            _make_diff_stats_row(
                name="conv_k_cudnn",
                cpu_op_name="aten::cudnn_convolution",
                source="trace2",
                kernel_time=120.0,
                lca_name="aten::miopen_convolution | aten::cudnn_convolution",
                lca_id=92,
                busy_time=120.0,
            ),
        ]
        df = pd.DataFrame(rows)
        return _diff_to_summary_report(df)

    def test_single_cpu_op_for_trace1(self, report):
        """All trace1 rows have the same cpu_op_name, so no pipe separator."""
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["name"] == "aten::miopen_convolution"

    def test_kernel_time_summed(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["kernel_time_trace1_us"] == pytest.approx(80.0)
        assert summary.iloc[0]["kernel_time_trace2_us"] == pytest.approx(120.0)


class TestMultipleDistinctCpuOpsPerLCA:
    """When trace1 has multiple *distinct* cpu_op_name values for one LCA."""

    @pytest.fixture
    def report(self):
        rows = [
            _make_diff_stats_row(
                name="k1",
                cpu_op_name="aten::add",
                source="trace1",
                kernel_time=25.0,
                lca_name="compound_op",
                lca_id=99,
                busy_time=60.0,
            ),
            _make_diff_stats_row(
                name="k2",
                cpu_op_name="aten::mul",
                source="trace1",
                kernel_time=35.0,
                lca_name="compound_op",
                lca_id=99,
                busy_time=60.0,
            ),
            _make_diff_stats_row(
                name="fused_k",
                cpu_op_name="aten::fused_add_mul",
                source="trace2",
                kernel_time=40.0,
                lca_name="compound_op",
                lca_id=99,
                busy_time=40.0,
            ),
        ]
        df = pd.DataFrame(rows)
        return _diff_to_summary_report(df)

    def test_pipe_separated_name(self, report):
        summary = report["tracediff_perf_summary"]
        name = summary.iloc[0]["name"]
        assert "aten::add" in name
        assert "aten::mul" in name
        assert " | " in name

    def test_sorted_alphabetically(self, report):
        summary = report["tracediff_perf_summary"]
        name = summary.iloc[0]["name"]
        assert name == "aten::add | aten::mul"


# ---------------------------------------------------------------------------
# Test: trace2-only LCA (no trace1 kernels)
# ---------------------------------------------------------------------------
class TestTrace2OnlyLCA:
    """An LCA that only has trace2 kernels (trace1 is empty for that LCA)."""

    @pytest.fixture
    def report(self):
        rows = [
            _make_diff_stats_row(
                name="new_kernel",
                cpu_op_name="aten::new_op",
                source="trace2",
                kernel_time=75.0,
                lca_name="aten::new_op",
                lca_id=200,
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
        """With trace1=0, speedup is t2/t1 = 75/0 = NaN."""
        summary = report["tracediff_perf_summary"]
        assert math.isnan(summary.iloc[0]["speedup (trace2/trace1)"])


# ---------------------------------------------------------------------------
# Test: trace1-only LCA (no trace2 kernels)
# ---------------------------------------------------------------------------
class TestTrace1OnlyLCA:
    """An LCA that only has trace1 kernels."""

    @pytest.fixture
    def report(self):
        rows = [
            _make_diff_stats_row(
                name="old_kernel",
                cpu_op_name="aten::old_op",
                source="trace1",
                kernel_time=50.0,
                lca_name="aten::old_op",
                lca_id=300,
                busy_time=50.0,
            ),
        ]
        df = pd.DataFrame(rows)
        return _diff_to_summary_report(df)

    def test_trace2_time_is_zero(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["kernel_time_trace2_us"] == 0.0

    def test_speedup_is_zero(self, report):
        """With trace2=0, speedup is t2/t1 = 0/50 = 0."""
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(0.0)

    def test_delta(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary.iloc[0]["delta_us (trace2 - trace1)"] == pytest.approx(-50.0)


# ---------------------------------------------------------------------------
# Test: empty DataFrame
# ---------------------------------------------------------------------------
class TestEmptyInput:
    def test_empty_df_returns_empty_summary(self):
        result = _diff_to_summary_report(pd.DataFrame())
        assert "tracediff_perf_summary" in result
        assert result["tracediff_perf_summary"].empty


# ---------------------------------------------------------------------------
# Test: NaN lowest_common_ancestor_id rows are dropped
# ---------------------------------------------------------------------------
class TestNaNLCADropped:
    @pytest.fixture
    def report(self):
        rows = [
            _make_diff_stats_row(
                name="valid_k",
                cpu_op_name="aten::mm",
                source="trace1",
                kernel_time=100.0,
                lca_name="aten::mm",
                lca_id=10,
                busy_time=100.0,
            ),
            _make_diff_stats_row(
                name="valid_k2",
                cpu_op_name="aten::mm",
                source="trace2",
                kernel_time=90.0,
                lca_name="aten::mm",
                lca_id=10,
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
# Test: passthrough of perf report DataFrames
# ---------------------------------------------------------------------------
class TestPassthroughSheets:
    """Extension enrich keeps original sheet keys (no *_trace1 / *_trace2 suffixes)."""

    def test_gpu_timeline_passthrough(self):
        rows = [
            _make_diff_stats_row(
                name="k",
                cpu_op_name="op",
                source="trace1",
                kernel_time=10.0,
                lca_name="op",
                lca_id=1,
                busy_time=10.0,
            ),
        ]
        df = pd.DataFrame(rows)
        perf1 = {"gpu_timeline": pd.DataFrame({"type": ["total_time"], "time ms": [5]})}
        result = _enrich_perf_from_diff(df, perf1, None)
        assert "gpu_timeline" in result
        assert result["gpu_timeline"]["time ms"].iloc[0] == 5

    def test_unified_perf_summary_passthrough(self):
        rows = [
            _make_diff_stats_row(
                name="k",
                cpu_op_name="op",
                source="trace1",
                kernel_time=10.0,
                lca_name="op",
                lca_id=1,
                busy_time=10.0,
            ),
        ]
        df = pd.DataFrame(rows)
        perf1 = {
            "unified_perf_summary": pd.DataFrame(
                {"name": ["aten::mm"], "TFLOPS/s": [50.0]}
            )
        }
        result = _enrich_perf_from_diff(df, perf1, None)
        assert "unified_perf_summary" in result
        assert result["unified_perf_summary"]["TFLOPS/s"].iloc[0] == 50.0

    def test_ops_unique_args_passthrough(self):
        rows = [
            _make_diff_stats_row(
                name="k",
                cpu_op_name="op",
                source="trace1",
                kernel_time=10.0,
                lca_name="op",
                lca_id=1,
                busy_time=10.0,
            ),
        ]
        df = pd.DataFrame(rows)
        perf1 = {"ops_unique_args": pd.DataFrame({"name": ["op"]})}
        result = _enrich_perf_from_diff(df, perf1, None)
        assert "ops_unique_args" in result


# ---------------------------------------------------------------------------
# Test: complex scenario with realistic data patterns
# ---------------------------------------------------------------------------
class TestRealisticScenario:
    """Multi-LCA scenario mimicking real TraceDiff output with varying kernel
    counts, different kernel names between traces, and mixed op structures."""

    @pytest.fixture
    def report(self):
        rows = [
            # LCA 134: batch_norm with 3 kernels in trace1, 4 in trace2
            _make_diff_stats_row(
                name="MIOpenBatchNormFwdTrainSpatialMeanVariance",
                cpu_op_name="aten::miopen_batch_norm",
                source="trace1",
                kernel_time=16.519,
                lca_name="aten::miopen_batch_norm | aten::native_batch_norm",
                lca_id=134,
                nn_module_stack="nn.Module: SimpleStemIN;nn.Module: BatchNorm2d",
                nn_module_parent="nn.Module: BatchNorm2d",
                busy_time=86.838,
            ),
            _make_diff_stats_row(
                name="MIOpenBatchNormFwdTrainSpatialFinalMeanVariance",
                cpu_op_name="aten::miopen_batch_norm",
                source="trace1",
                kernel_time=52.76,
                lca_name="aten::miopen_batch_norm | aten::native_batch_norm",
                lca_id=134,
                busy_time=86.838,
            ),
            _make_diff_stats_row(
                name="MIOpenBatchNormFwdTrainSpatialNorm",
                cpu_op_name="aten::miopen_batch_norm",
                source="trace1",
                kernel_time=17.559,
                lca_name="aten::miopen_batch_norm | aten::native_batch_norm",
                lca_id=134,
                busy_time=86.838,
            ),
            _make_diff_stats_row(
                name="fill_kernel",
                cpu_op_name="aten::fill_",
                source="trace2",
                kernel_time=1.407,
                lca_name="aten::miopen_batch_norm | aten::native_batch_norm",
                lca_id=134,
                busy_time=198.719,
            ),
            _make_diff_stats_row(
                name="batch_norm_collect_statistics",
                cpu_op_name="aten::native_batch_norm",
                source="trace2",
                kernel_time=117.888,
                lca_name="aten::miopen_batch_norm | aten::native_batch_norm",
                lca_id=134,
                busy_time=198.719,
            ),
            _make_diff_stats_row(
                name="batch_norm_update_stats",
                cpu_op_name="aten::native_batch_norm",
                source="trace2",
                kernel_time=2.048,
                lca_name="aten::miopen_batch_norm | aten::native_batch_norm",
                lca_id=134,
                busy_time=198.719,
            ),
            _make_diff_stats_row(
                name="batch_norm_transform_input",
                cpu_op_name="aten::native_batch_norm",
                source="trace2",
                kernel_time=77.376,
                lca_name="aten::miopen_batch_norm | aten::native_batch_norm",
                lca_id=134,
                busy_time=198.719,
            ),
            # LCA 92: convolution
            _make_diff_stats_row(
                name="ck_conv_fwd",
                cpu_op_name="aten::miopen_convolution",
                source="trace1",
                kernel_time=51.239,
                lca_name="aten::miopen_convolution | aten::cudnn_convolution",
                lca_id=92,
                busy_time=51.239,
            ),
            _make_diff_stats_row(
                name="implicit_convolve_sgemm",
                cpu_op_name="aten::cudnn_convolution",
                source="trace2",
                kernel_time=128.096,
                lca_name="aten::miopen_convolution | aten::cudnn_convolution",
                lca_id=92,
                busy_time=128.096,
            ),
        ]
        df = pd.DataFrame(rows)
        return _diff_to_summary_report(df)

    def test_two_rows(self, report):
        summary = report["tracediff_perf_summary"]
        assert len(summary) == 2

    def test_batch_norm_kernel_time_sum_trace1(self, report):
        summary = report["tracediff_perf_summary"]
        bn_row = summary[summary["lowest_common_ancestor_id"] == 134].iloc[0]
        assert bn_row["kernel_time_trace1_us"] == pytest.approx(16.519 + 52.76 + 17.559)

    def test_batch_norm_kernel_time_sum_trace2(self, report):
        summary = report["tracediff_perf_summary"]
        bn_row = summary[summary["lowest_common_ancestor_id"] == 134].iloc[0]
        assert bn_row["kernel_time_trace2_us"] == pytest.approx(
            1.407 + 117.888 + 2.048 + 77.376
        )

    def test_batch_norm_num_kernels(self, report):
        summary = report["tracediff_perf_summary"]
        bn_row = summary[summary["lowest_common_ancestor_id"] == 134].iloc[0]
        assert bn_row["num_kernels_trace1"] == 3
        assert bn_row["num_kernels_trace2"] == 4

    def test_conv_kernel_times(self, report):
        summary = report["tracediff_perf_summary"]
        conv_row = summary[summary["lowest_common_ancestor_id"] == 92].iloc[0]
        assert conv_row["kernel_time_trace1_us"] == pytest.approx(51.239)
        assert conv_row["kernel_time_trace2_us"] == pytest.approx(128.096)

    def test_conv_speedup(self, report):
        summary = report["tracediff_perf_summary"]
        conv_row = summary[summary["lowest_common_ancestor_id"] == 92].iloc[0]
        assert conv_row["speedup (trace2/trace1)"] == pytest.approx(128.096 / 51.239)

    def test_sorted_by_trace1_time(self, report):
        summary = report["tracediff_perf_summary"]
        times = summary["kernel_time_trace1_us"].tolist()
        assert times == sorted(times, reverse=True)

    def test_percentages_add_up(self, report):
        summary = report["tracediff_perf_summary"]
        assert summary["pct_of_trace1_total (%)"].sum() == pytest.approx(100.0)
        assert summary["pct_of_trace2_total (%)"].sum() == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Integration test: synthetic traces through full TraceDiff + perf report
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
    """Build a minimal PyTorch-style trace from a list of (cpu_op_name, kernel_name, kernel_dur) tuples.

    Each tuple produces: cpu_op -> hipLaunchKernel -> kernel, linked by ac2g.
    """
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
    """Integration test: build two synthetic traces, run through full pipeline."""

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
            td.baseline,
        )
        summary = tracediff_perf_summary_from_diff_stats(td.diff_stats_df)

        return td, perf1, perf2, enriched, summary

    def test_tracediff_produced_diff_stats(self, traces_and_report):
        td, _, _, _, _ = traces_and_report
        assert not td.diff_stats_df.empty

    def test_at_least_one_source_present(self, traces_and_report):
        """Minimal synthetic traces may not produce both sources due to
        GPU-event parent-chain limitations; at least one must be present."""
        td, _, _, _, _ = traces_and_report
        if "source" in td.diff_stats_df.columns:
            sources = td.diff_stats_df["source"].unique()
            assert len(sources) >= 1

    def test_report_structure(self, traces_and_report):
        _, _, _, enriched, summary = traces_and_report
        assert isinstance(summary, pd.DataFrame)
        assert "diff_stats" not in enriched

    def test_summary_non_negative_times(self, traces_and_report):
        """If the summary has rows, all kernel times should be non-negative."""
        _, _, _, _, summary = traces_and_report
        if not summary.empty:
            for _, row in summary.iterrows():
                t1 = row["kernel_time_trace1_us"]
                t2 = row["kernel_time_trace2_us"]
                assert t1 >= 0
                assert t2 >= 0
                assert t1 > 0 or t2 > 0

    def test_gpu_timeline_passthrough(self, traces_and_report):
        _, _, _, enriched, _ = traces_and_report
        assert "gpu_timeline" in enriched

    def test_percentages_valid(self, traces_and_report):
        _, _, _, _, summary = traces_and_report
        if not summary.empty and "pct_of_trace1_total (%)" in summary.columns:
            total_pct = summary["pct_of_trace1_total (%)"].sum()
            assert total_pct == pytest.approx(100.0, abs=0.1)


# ---------------------------------------------------------------------------
# Tests: _find_perf_summary_matching_node
# ---------------------------------------------------------------------------
class _FakeTraceTree:
    """Minimal tree for _find_perf_summary_matching_node (events_by_uid + categories)."""

    def __init__(self, events_by_uid):
        self.events_by_uid = events_by_uid

    def event_to_category(self, node):
        return node.get("_category", "cpu_op")


def _empty_arg_row():
    return {
        "Input Dims": "",
        "Input type": "",
        "Input Strides": "",
        "Concrete Inputs": "",
    }


class TestFindPerfSummaryMatchingNode:
    """Ancestor walk, then descendant BFS, for unified_perf_summary key alignment."""

    def test_returns_none_without_tree(self):
        key = _make_str_key("aten::mm", _empty_arg_row())
        assert _find_perf_summary_matching_node(1, None, {key}) is None

    def test_returns_none_without_uid(self):
        tree = _FakeTraceTree({1: {"parent": None, "name": "x", "args": {}}})
        key = _make_str_key("aten::mm", _empty_arg_row())
        assert _find_perf_summary_matching_node(None, tree, {key}) is None

    def test_returns_none_when_perf_keys_empty(self):
        tree = _FakeTraceTree({1: {"parent": None, "name": "aten::mm", "args": {}}})
        assert _find_perf_summary_matching_node(1, tree, set()) is None

    def test_returns_none_when_uid_missing(self):
        tree = _FakeTraceTree({})
        key = _make_str_key("aten::mm", _empty_arg_row())
        assert _find_perf_summary_matching_node(99, tree, {key}) is None

    def test_direct_match_on_starting_cpu_op(self):
        args = _empty_arg_row()
        key = _make_str_key("aten::mm", args)
        mm = {
            "parent": None,
            "name": "aten::mm",
            "args": dict(args),
            "_category": "cpu_op",
        }
        tree = _FakeTraceTree({42: mm})
        assert _find_perf_summary_matching_node(42, tree, {key}) is mm

    def test_ancestor_match_when_starting_node_is_kernel(self):
        args = _empty_arg_row()
        key = _make_str_key("aten::mm", args)
        mm = {
            "parent": None,
            "name": "aten::mm",
            "args": dict(args),
            "_category": "cpu_op",
        }
        kernel = {
            "parent": 20,
            "name": "gemm_kernel",
            "args": {},
            "_category": "kernel",
        }
        tree = _FakeTraceTree({10: kernel, 20: mm})
        assert _find_perf_summary_matching_node(10, tree, {key}) is mm

    def test_descendant_match_when_ancestors_do_not_match(self):
        args_wrong = _empty_arg_row()
        args_mm = _empty_arg_row()
        key_mm = _make_str_key("aten::mm", args_mm)
        outer = {
            "parent": None,
            "name": "aten::outer",
            "args": dict(args_wrong),
            "_category": "cpu_op",
        }
        mm = {
            "parent": 1,
            "name": "aten::mm",
            "args": dict(args_mm),
            "_category": "cpu_op",
        }
        tree = _FakeTraceTree({1: outer, 2: mm})
        assert _find_perf_summary_matching_node(1, tree, {key_mm}) is mm

    def test_returns_none_when_no_cpu_op_matches(self):
        args = _empty_arg_row()
        key = _make_str_key("aten::mm", args)
        other = {
            "parent": None,
            "name": "aten::relu",
            "args": dict(args),
            "_category": "cpu_op",
        }
        tree = _FakeTraceTree({1: other})
        assert _find_perf_summary_matching_node(1, tree, {key}) is None

    def test_ancestor_match_wins_before_descendant_bfs(self):
        """Walk upward fully before any child search; parent match beats child match."""
        args = _empty_arg_row()
        key_mm = _make_str_key("aten::mm", args)
        mm_parent = {
            "parent": None,
            "name": "aten::mm",
            "args": dict(args),
            "_category": "cpu_op",
        }
        outer = {
            "parent": 2,
            "name": "aten::outer",
            "args": dict(args),
            "_category": "cpu_op",
        }
        mm_child = {
            "parent": 1,
            "name": "aten::mm",
            "args": dict(args),
            "_category": "cpu_op",
        }
        tree = _FakeTraceTree({1: outer, 2: mm_parent, 3: mm_child})
        assert _find_perf_summary_matching_node(1, tree, {key_mm}) is mm_parent

    def test_bfs_returns_first_matching_child_in_stable_order(self):
        args_mm = _empty_arg_row()
        key_mm = _make_str_key("aten::mm", args_mm)
        outer = {
            "parent": None,
            "name": "aten::outer",
            "args": dict(args_mm),
            "_category": "cpu_op",
        }
        mm_a = {
            "parent": 1,
            "name": "aten::mm",
            "args": dict(args_mm),
            "_category": "cpu_op",
        }
        mm_b = {
            "parent": 1,
            "name": "aten::mm",
            "args": dict(args_mm),
            "_category": "cpu_op",
        }
        # Dict iteration order defines sibling order in _ensure_parent_to_children.
        tree = _FakeTraceTree({1: outer, 2: mm_a, 3: mm_b})
        found = _find_perf_summary_matching_node(1, tree, {key_mm})
        assert found is mm_a


# ---------------------------------------------------------------------------
# Tests: _build_trace2_time_lookup
# ---------------------------------------------------------------------------
class TestBuildTrace2TimeLookup:
    """Verify the lookup mapping from trace1 (name, args) → trace2 kernel time sum."""

    def test_empty_df_returns_empty_dict(self):
        lookup, lca_ids, lc = _build_trace2_time_lookup(pd.DataFrame())
        assert lookup == {}
        assert lca_ids == {}
        assert lc == {}

    def test_no_trace1_rows_returns_empty(self):
        rows = [
            _make_diff_stats_row(
                "k",
                "op",
                "trace2",
                50.0,
                "op",
                1,
                busy_time=50.0,
            ),
        ]
        lookup, lca_ids, lc = _build_trace2_time_lookup(pd.DataFrame(rows))
        assert lookup == {}
        assert lca_ids == {}
        assert lc == {}

    def test_single_lca_single_kernel(self):
        rows = [
            _make_diff_stats_row(
                "k1",
                "aten::mm",
                "trace1",
                100.0,
                "aten::mm",
                10,
                input_dims="[[32,64]]",
                input_type="float",
                busy_time=100.0,
            ),
            _make_diff_stats_row(
                "k2",
                "aten::mm",
                "trace2",
                80.0,
                "aten::mm",
                10,
                input_dims="[[32,64]]",
                input_type="float",
                busy_time=80.0,
            ),
        ]
        lookup, lca_ids, lc = _build_trace2_time_lookup(pd.DataFrame(rows))
        key = ("aten::mm", "((32,64))", "float", "", "", "")
        assert key in lookup
        assert lookup[key] == pytest.approx(80.0)
        assert lc[key] == 1

    def test_multiple_lcas_same_args_aggregated(self):
        """Two LCAs with the same trace1 (name, args) should have their trace2 times summed."""
        rows = [
            _make_diff_stats_row(
                "k1",
                "aten::mm",
                "trace1",
                50.0,
                "aten::mm",
                10,
                input_dims="[32,64]",
                input_type="float",
                busy_time=50.0,
            ),
            _make_diff_stats_row(
                "k2",
                "aten::mm",
                "trace2",
                40.0,
                "aten::mm",
                10,
                input_dims="[32,64]",
                input_type="float",
                busy_time=40.0,
            ),
            _make_diff_stats_row(
                "k3",
                "aten::mm",
                "trace1",
                60.0,
                "aten::mm",
                20,
                input_dims="[32,64]",
                input_type="float",
                busy_time=60.0,
            ),
            _make_diff_stats_row(
                "k4",
                "aten::mm",
                "trace2",
                45.0,
                "aten::mm",
                20,
                input_dims="[32,64]",
                input_type="float",
                busy_time=45.0,
            ),
        ]
        lookup, lca_ids, lc = _build_trace2_time_lookup(pd.DataFrame(rows))
        key = ("aten::mm", "(32,64)", "float", "", "", "")
        assert lookup[key] == pytest.approx(85.0)  # 40 + 45
        assert lc[key] == 2

    def test_multiple_lcas_same_key_unions_cpu_op_uids(self):
        """Same tuple key from two LCAs: count is unique trace2 cpu_op_uid, not row count."""
        shared_uid = 4242
        rows = [
            _make_diff_stats_row(
                "k1",
                "aten::mm",
                "trace1",
                50.0,
                "aten::mm",
                10,
                input_dims="[32,64]",
                input_type="float",
                busy_time=50.0,
            ),
            _make_diff_stats_row(
                "k2",
                "aten::mm",
                "trace2",
                40.0,
                "aten::mm",
                10,
                input_dims="[32,64]",
                input_type="float",
                cpu_op_uid=shared_uid,
                busy_time=40.0,
            ),
            _make_diff_stats_row(
                "k3",
                "aten::mm",
                "trace1",
                60.0,
                "aten::mm",
                20,
                input_dims="[32,64]",
                input_type="float",
                busy_time=60.0,
            ),
            _make_diff_stats_row(
                "k4",
                "aten::mm",
                "trace2",
                45.0,
                "aten::mm",
                20,
                input_dims="[32,64]",
                input_type="float",
                cpu_op_uid=shared_uid,
                busy_time=45.0,
            ),
        ]
        lookup, lca_ids, lc = _build_trace2_time_lookup(pd.DataFrame(rows))
        key = ("aten::mm", "(32,64)", "float", "", "", "")
        assert lookup[key] == pytest.approx(85.0)
        assert lc[key] == 1

    def test_different_args_produce_different_keys(self):
        rows = [
            _make_diff_stats_row(
                "k1",
                "aten::mm",
                "trace1",
                50.0,
                "mm",
                10,
                input_dims="[32,64]",
                input_type="float",
                busy_time=50.0,
            ),
            _make_diff_stats_row(
                "k2",
                "aten::mm",
                "trace2",
                40.0,
                "mm",
                10,
                input_dims="[32,64]",
                input_type="float",
                busy_time=40.0,
            ),
            _make_diff_stats_row(
                "k3",
                "aten::mm",
                "trace1",
                100.0,
                "mm",
                20,
                input_dims="[64,128]",
                input_type="float",
                busy_time=100.0,
            ),
            _make_diff_stats_row(
                "k4",
                "aten::mm",
                "trace2",
                90.0,
                "mm",
                20,
                input_dims="[64,128]",
                input_type="float",
                busy_time=90.0,
            ),
        ]
        lookup, lca_ids, lc = _build_trace2_time_lookup(pd.DataFrame(rows))
        assert len(lookup) == 2
        assert lookup[("aten::mm", "(32,64)", "float", "", "", "")] == pytest.approx(
            40.0
        )
        assert lookup[("aten::mm", "(64,128)", "float", "", "", "")] == pytest.approx(
            90.0
        )
        assert lc[("aten::mm", "(32,64)", "float", "", "", "")] == 1
        assert lc[("aten::mm", "(64,128)", "float", "", "", "")] == 1

    def test_busy_time_used_over_kernel_time_sum(self):
        """When busy_time is present, lookup uses it instead of summing kernel_time."""
        rows = [
            _make_diff_stats_row(
                "k1",
                "aten::conv",
                "trace1",
                100.0,
                "conv",
                10,
                input_dims="[2,224]",
                input_type="float",
                busy_time=100.0,
            ),
            _make_diff_stats_row(
                "slice_k",
                "aten::conv",
                "trace2",
                80.0,
                "conv",
                10,
                input_dims="[2,224]",
                input_type="float",
                busy_time=50.0,
            ),
            _make_diff_stats_row(
                "dgrad_k",
                "aten::conv",
                "trace2",
                60.0,
                "conv",
                10,
                input_dims="[2,224]",
                input_type="float",
                busy_time=50.0,
            ),
        ]
        lookup, lca_ids, lc = _build_trace2_time_lookup(pd.DataFrame(rows))
        key = ("aten::conv", "(2,224)", "float", "", "", "")
        assert lookup[key] == pytest.approx(50.0)

    def test_trace1_only_lca_returns_zero_trace2_time(self):
        rows = [
            _make_diff_stats_row(
                "k1",
                "aten::mm",
                "trace1",
                100.0,
                "mm",
                10,
                busy_time=100.0,
            ),
        ]
        lookup, lca_ids, lc = _build_trace2_time_lookup(pd.DataFrame(rows))
        key = ("aten::mm", "", "", "", "", "")
        assert key in lookup
        assert lookup[key] == pytest.approx(0.0)
        assert lc[key] == 0


# ---------------------------------------------------------------------------
# Tests: _enrich_sheet_with_trace2
# ---------------------------------------------------------------------------
class TestEnrichSheetWithTrace2:
    """Verify trace1 perf sheets get speedup, delta, trace2 time, and operation_count_trace2."""

    def test_empty_sheet_returned_unchanged(self):
        df = pd.DataFrame()
        lookup = {("aten::mm", "", "", "", "", ""): 100.0}
        result = _enrich_sheet_with_trace2(df, lookup, "Kernel Time (µs)_sum")
        assert result.empty

    def test_empty_lookup_returned_unchanged(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "Kernel Time (µs)_sum": [100.0],
            }
        )
        result = _enrich_sheet_with_trace2(df, {}, "Kernel Time (µs)_sum")
        assert "speedup (trace2/trace1)" not in result.columns

    def test_single_row_enrichment(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "Input Dims": ["[32,64]"],
                "Input type": ["float"],
                "Input Strides": [""],
                "Concrete Inputs": [""],
                "Kernel Time (µs)_sum": [100.0],
            }
        )
        key = ("aten::mm", "(32,64)", "float", "", "", "")
        lookup = {key: 80.0}
        result = _enrich_sheet_with_trace2(
            df, lookup, "Kernel Time (µs)_sum", count_lookup={key: 3}
        )

        assert "Kernel Time (µs)_trace2_sum" in result.columns
        assert "operation_count_trace2" in result.columns
        assert "speedup (trace2/trace1)" in result.columns
        assert "delta_us (trace2 - trace1)" in result.columns
        assert result.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(80.0 / 100.0)
        assert result.iloc[0]["delta_us (trace2 - trace1)"] == pytest.approx(-20.0)
        assert result.iloc[0]["operation_count_trace2"] == 3.0

    def test_unmatched_row_gets_nan(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "Input Dims": ["[32,64]"],
                "Input type": ["float"],
                "Input Strides": [""],
                "Concrete Inputs": [""],
                "Kernel Time (µs)_sum": [100.0],
            }
        )
        lookup = {("aten::relu", "(32)", "float", "", "", ""): 10.0}
        result = _enrich_sheet_with_trace2(df, lookup, "Kernel Time (µs)_sum")

        import numpy as np

        assert np.isnan(result.iloc[0]["speedup (trace2/trace1)"])

    def test_multiple_rows_mixed_match(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm", "aten::relu"],
                "Input Dims": ["[32,64]", "[32]"],
                "Input type": ["float", "float"],
                "Input Strides": ["", ""],
                "Concrete Inputs": ["", ""],
                "Kernel Time (µs)_sum": [200.0, 30.0],
            }
        )
        lookup = {
            ("aten::mm", "(32,64)", "float", "", "", ""): 150.0,
        }
        result = _enrich_sheet_with_trace2(df, lookup, "Kernel Time (µs)_sum")

        assert result.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(150.0 / 200.0)
        assert result.iloc[0]["delta_us (trace2 - trace1)"] == pytest.approx(-50.0)

        import numpy as np

        assert np.isnan(result.iloc[1]["speedup (trace2/trace1)"])

    def test_columns_inserted_after_kernel_time(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "Input Dims": [""],
                "Input type": [""],
                "Input Strides": [""],
                "Concrete Inputs": [""],
                "GFLOPS": [100.0],
                "Kernel Time (µs)_sum": [50.0],
                "TFLOPS/s_mean": [10.0],
            }
        )
        lookup = {("aten::mm", "", "", "", "", ""): 40.0}
        result = _enrich_sheet_with_trace2(df, lookup, "Kernel Time (µs)_sum")

        cols = list(result.columns)
        kt_idx = cols.index("Kernel Time (µs)_sum")
        assert cols[kt_idx + 1] == "speedup (trace2/trace1)"
        assert cols[kt_idx + 2] == "delta_us (trace2 - trace1)"
        assert cols[kt_idx + 3] == "Kernel Time (µs)_trace2_sum"
        assert cols[kt_idx + 4] == "operation_count_trace2"
        assert result.iloc[0]["Kernel Time (µs)_trace2_sum"] == 40.0
        import numpy as np

        assert np.isnan(result.iloc[0]["operation_count_trace2"])
        assert "debug_trace2_time_us" not in cols
        assert "debug_lca_ids" not in cols

    def test_debug_inserts_lca_ids_after_trace2_column(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "Input Dims": [""],
                "Input type": [""],
                "Input Strides": [""],
                "Concrete Inputs": [""],
                "Kernel Time (µs)_sum": [50.0],
            }
        )
        key = ("aten::mm", "", "", "", "", "")
        lookup = {key: 40.0}
        lca_ids = {key: [101, 102]}
        result = _enrich_sheet_with_trace2(
            df,
            lookup,
            "Kernel Time (µs)_sum",
            lca_ids_lookup=lca_ids,
            count_lookup={key: 2},
            debug=True,
        )
        cols = list(result.columns)
        kt_idx = cols.index("Kernel Time (µs)_sum")
        assert cols[kt_idx + 3] == "Kernel Time (µs)_trace2_sum"
        assert cols[kt_idx + 4] == "operation_count_trace2"
        assert cols[kt_idx + 5] == "debug_lca_ids"
        assert result.iloc[0]["Kernel Time (µs)_trace2_sum"] == 40.0
        assert result.iloc[0]["operation_count_trace2"] == 2.0

    def test_works_with_ops_unique_args_column(self):
        """Verify enrichment works with 'total_direct_kernel_time_sum' column."""
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "Input Dims": ["[32,64]"],
                "Input type": ["float"],
                "Input Strides": [""],
                "Concrete Inputs": [""],
                "total_direct_kernel_time_sum": [100.0],
            }
        )
        lookup = {("aten::mm", "(32,64)", "float", "", "", ""): 70.0}
        result = _enrich_sheet_with_trace2(df, lookup, "total_direct_kernel_time_sum")
        assert result.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(70.0 / 100.0)

    def test_zero_trace2_time_gives_zero_speedup(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "Input Dims": [""],
                "Input type": [""],
                "Input Strides": [""],
                "Concrete Inputs": [""],
                "Kernel Time (µs)_sum": [100.0],
            }
        )
        lookup = {("aten::mm", "", "", "", "", ""): 0.0}
        result = _enrich_sheet_with_trace2(df, lookup, "Kernel Time (µs)_sum")

        assert result.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(0.0)
        assert result.iloc[0]["delta_us (trace2 - trace1)"] == pytest.approx(-100.0)

    def test_original_df_not_modified(self):
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "Input Dims": [""],
                "Input type": [""],
                "Input Strides": [""],
                "Concrete Inputs": [""],
                "Kernel Time (µs)_sum": [100.0],
            }
        )
        original_cols = list(df.columns)
        lookup = {("aten::mm", "", "", "", "", ""): 80.0}
        _enrich_sheet_with_trace2(df, lookup, "Kernel Time (µs)_sum")
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# Tests: enrichment via enrich_perf_report_dict_inplace
# ---------------------------------------------------------------------------
class TestEnrichmentViaBuildReport:
    """Only ``unified_perf_summary`` gets speedup/delta; op-category and launcher
    sheets do not."""

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
                input_dims="[32,64]",
                input_type="float",
                busy_time=200.0,
            ),
            _make_diff_stats_row(
                "gemm_k2",
                "aten::mm",
                "trace2",
                150.0,
                "mm",
                10,
                input_dims="[32,64]",
                input_type="float",
                busy_time=150.0,
            ),
            _make_diff_stats_row(
                "relu_k",
                "aten::relu",
                "trace1",
                30.0,
                "relu",
                20,
                input_dims="[32]",
                input_type="float",
                busy_time=30.0,
            ),
            _make_diff_stats_row(
                "relu_k2",
                "aten::relu",
                "trace2",
                25.0,
                "relu",
                20,
                input_dims="[32]",
                input_type="float",
                busy_time=25.0,
            ),
        ]
        diff_df = pd.DataFrame(diff_rows)

        perf1 = {
            "gpu_timeline": pd.DataFrame({"type": ["total_time"], "time ms": [5.0]}),
            "GEMM": pd.DataFrame(
                {
                    "name": ["aten::mm"],
                    "Input Dims": ["[32,64]"],
                    "Input type": ["float"],
                    "Input Strides": [""],
                    "Concrete Inputs": [""],
                    "Kernel Time (µs)_sum": [200.0],
                    "TFLOPS/s_mean": [10.0],
                    "Roofline Time (µs)_first": [180.0],
                    "Pct Roofline_mean": [90.0],
                }
            ),
            "Normalization": pd.DataFrame(
                {
                    "name": ["aten::relu"],
                    "Input Dims": ["[32]"],
                    "Input type": ["float"],
                    "Input Strides": [""],
                    "Concrete Inputs": [""],
                    "Kernel Time (µs)_sum": [30.0],
                }
            ),
            "ops_unique_args": pd.DataFrame(
                {
                    "name": ["aten::mm", "aten::relu"],
                    "Input Dims": ["[32,64]", "[32]"],
                    "Input type": ["float", "float"],
                    "Input Strides": ["", ""],
                    "Concrete Inputs": ["", ""],
                    "total_direct_kernel_time_sum": [200.0, 30.0],
                }
            ),
            "unified_perf_summary": pd.DataFrame(
                {
                    "name": ["aten::mm", "aten::relu"],
                    "Input Dims": ["[32,64]", "[32]"],
                    "Input type": ["float", "float"],
                    "Input Strides": ["", ""],
                    "Concrete Inputs": ["", ""],
                    "Kernel Time (µs)_sum": [200.0, 30.0],
                }
            ),
        }
        return _enrich_perf_from_diff(diff_df, perf1, None)

    def test_gemm_sheet_not_enriched(self, enriched_report):
        gemm = enriched_report["GEMM"]
        assert "Kernel Time trace2 (µs)_sum" not in gemm.columns
        assert "speedup (trace2/trace1)" not in gemm.columns
        assert "delta_us (trace2 - trace1)" not in gemm.columns

    def test_normalization_sheet_not_enriched(self, enriched_report):
        norm = enriched_report["Normalization"]
        assert "speedup (trace2/trace1)" not in norm.columns
        assert "delta_us (trace2 - trace1)" not in norm.columns

    def test_ops_unique_args_not_enriched(self, enriched_report):
        ops = enriched_report["ops_unique_args"]
        assert "Kernel Time trace2 (µs)_sum" not in ops.columns
        assert "speedup (trace2/trace1)" not in ops.columns
        assert "delta_us (trace2 - trace1)" not in ops.columns

    def test_unified_perf_summary_enriched(self, enriched_report):
        ups = enriched_report["unified_perf_summary"]
        assert "Kernel Time trace2 (µs)_sum" not in ups.columns
        assert "speedup (trace2/trace1)" in ups.columns

    def test_gpu_timeline_not_enriched(self, enriched_report):
        timeline = enriched_report["gpu_timeline"]
        assert "Kernel Time trace2 (µs)_sum" not in timeline.columns
        assert "speedup (trace2/trace1)" not in timeline.columns

    def test_existing_columns_preserved(self, enriched_report):
        gemm = enriched_report["GEMM"]
        assert "TFLOPS/s_mean" in gemm.columns
        assert "Roofline Time (µs)_first" in gemm.columns
        assert "Pct Roofline_mean" in gemm.columns
        assert gemm.iloc[0]["TFLOPS/s_mean"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Tests: _build_lca_consolidation
# ---------------------------------------------------------------------------
class TestBuildLCAConsolidation:
    """Verify dominant-op identification and time transfer computation."""

    def test_empty_df(self):
        adds, subs, t2map, lca_ids, t2cnt = _build_lca_consolidation(pd.DataFrame())
        assert adds == {}
        assert subs == {}
        assert t2map == {}
        assert lca_ids == {}
        assert t2cnt == {}

    def test_single_op_lca_produces_no_adjustments(self):
        rows = [
            _make_diff_stats_row(
                "k1", "aten::mm", "trace1", 100.0, "mm", 10, busy_time=100.0
            ),
            _make_diff_stats_row(
                "k2", "aten::mm", "trace2", 80.0, "mm", 10, busy_time=80.0
            ),
        ]
        adds, subs, t2map, lca_ids, t2cnt = _build_lca_consolidation(pd.DataFrame(rows))
        assert adds == {}
        assert subs == {}
        assert t2map == {}
        assert lca_ids == {}
        assert t2cnt == {}

    def test_multi_op_lca_dominant_selection(self):
        """Dominant op is the one with highest total kernel time."""
        rows = [
            _make_diff_stats_row(
                "fill_k",
                "aten::fill_",
                "trace1",
                25.0,
                "lca",
                99,
                busy_time=525.0,
            ),
            _make_diff_stats_row(
                "fmha_k",
                "aiter::fmha_v3_bwd",
                "trace1",
                500.0,
                "lca",
                99,
                busy_time=525.0,
            ),
            _make_diff_stats_row(
                "flash_k",
                "flash_attn::bwd",
                "trace2",
                300.0,
                "lca",
                99,
                busy_time=300.0,
            ),
        ]
        adds, subs, t2map, lca_ids, t2cnt = _build_lca_consolidation(pd.DataFrame(rows))

        dominant_key = ("aiter::fmha_v3_bwd", "", "", "", "", "")
        non_dom_key = ("aten::fill_", "", "", "", "", "")

        assert dominant_key in adds
        assert adds[dominant_key] == pytest.approx(25.0)

        assert non_dom_key in subs
        assert subs[non_dom_key] == pytest.approx(25.0)

        assert dominant_key in t2map
        assert t2map[dominant_key] == pytest.approx(300.0)
        assert t2cnt[dominant_key] == 1

    def test_multi_op_lca_multiple_instances_aggregate(self):
        """Multiple multi-op LCAs with same ops accumulate adjustments."""
        rows = [
            _make_diff_stats_row(
                "fill_k1",
                "aten::fill_",
                "trace1",
                20.0,
                "lca",
                1,
                busy_time=420.0,
            ),
            _make_diff_stats_row(
                "fmha_k1", "aiter::fmha", "trace1", 400.0, "lca", 1, busy_time=420.0
            ),
            _make_diff_stats_row(
                "flash_k1",
                "flash_attn::bwd",
                "trace2",
                250.0,
                "lca",
                1,
                busy_time=250.0,
            ),
            _make_diff_stats_row(
                "fill_k2",
                "aten::fill_",
                "trace1",
                30.0,
                "lca",
                2,
                busy_time=630.0,
            ),
            _make_diff_stats_row(
                "fmha_k2", "aiter::fmha", "trace1", 600.0, "lca", 2, busy_time=630.0
            ),
            _make_diff_stats_row(
                "flash_k2",
                "flash_attn::bwd",
                "trace2",
                350.0,
                "lca",
                2,
                busy_time=350.0,
            ),
        ]
        adds, subs, t2map, lca_ids, t2cnt = _build_lca_consolidation(pd.DataFrame(rows))

        dominant_key = ("aiter::fmha", "", "", "", "", "")
        non_dom_key = ("aten::fill_", "", "", "", "", "")

        assert adds[dominant_key] == pytest.approx(50.0)  # 20 + 30
        assert subs[non_dom_key] == pytest.approx(50.0)
        assert t2map[dominant_key] == pytest.approx(600.0)  # 250 + 350
        assert t2cnt[dominant_key] == 2


# ---------------------------------------------------------------------------
# Tests: _apply_lca_consolidation
# ---------------------------------------------------------------------------
class TestApplyLCAConsolidation:
    """Verify perf sheet adjustments from dominant-op consolidation."""

    def test_no_adjustments_passthrough(self):
        sheets = {
            "ops": pd.DataFrame({"name": ["aten::mm"], "Kernel Time (µs)_sum": [100.0]})
        }
        result = _apply_lca_consolidation(sheets, {}, {})
        assert result["ops"].iloc[0]["Kernel Time (µs)_sum"] == 100.0

    def test_addition_applied(self):
        sheets = {
            "ops": pd.DataFrame(
                {
                    "name": ["aiter::fmha"],
                    "Input Dims": [""],
                    "Input type": [""],
                    "Input Strides": [""],
                    "Concrete Inputs": [""],
                    "Kernel Time (µs)_sum": [500.0],
                }
            )
        }
        adds = {("aiter::fmha", "", "", "", "", ""): 25.0}
        result = _apply_lca_consolidation(sheets, adds, {})
        assert result["ops"].iloc[0]["Kernel Time (µs)_sum"] == pytest.approx(525.0)

    def test_subtraction_applied(self):
        sheets = {
            "ops": pd.DataFrame(
                {
                    "name": ["aten::fill_"],
                    "Input Dims": [""],
                    "Input type": [""],
                    "Input Strides": [""],
                    "Concrete Inputs": [""],
                    "Kernel Time (µs)_sum": [100.0],
                }
            )
        }
        subs = {("aten::fill_", "", "", "", "", ""): 25.0}
        result = _apply_lca_consolidation(sheets, {}, subs)
        assert result["ops"].iloc[0]["Kernel Time (µs)_sum"] == pytest.approx(75.0)

    def test_subtraction_removes_row_when_zero(self):
        sheets = {
            "ops": pd.DataFrame(
                {
                    "name": ["aten::fill_", "aten::mm"],
                    "Input Dims": ["", ""],
                    "Input type": ["", ""],
                    "Input Strides": ["", ""],
                    "Concrete Inputs": ["", ""],
                    "Kernel Time (µs)_sum": [25.0, 200.0],
                }
            )
        }
        subs = {("aten::fill_", "", "", "", "", ""): 25.0}
        result = _apply_lca_consolidation(sheets, {}, subs)
        assert len(result["ops"]) == 1
        assert result["ops"].iloc[0]["name"] == "aten::mm"

    def test_both_add_and_subtract(self):
        sheets = {
            "ops": pd.DataFrame(
                {
                    "name": ["aiter::fmha", "aten::fill_"],
                    "Input Dims": ["", ""],
                    "Input type": ["", ""],
                    "Input Strides": ["", ""],
                    "Concrete Inputs": ["", ""],
                    "Kernel Time (µs)_sum": [500.0, 100.0],
                }
            )
        }
        adds = {("aiter::fmha", "", "", "", "", ""): 25.0}
        subs = {("aten::fill_", "", "", "", "", ""): 25.0}
        result = _apply_lca_consolidation(sheets, adds, subs)

        fmha_row = result["ops"][result["ops"]["name"] == "aiter::fmha"].iloc[0]
        fill_row = result["ops"][result["ops"]["name"] == "aten::fill_"].iloc[0]

        assert fmha_row["Kernel Time (µs)_sum"] == pytest.approx(525.0)
        assert fill_row["Kernel Time (µs)_sum"] == pytest.approx(75.0)

    def test_sheet_without_name_column_unchanged(self):
        sheets = {
            "gpu_timeline": pd.DataFrame({"type": ["total_time"], "time ms": [5.0]})
        }
        adds = {("something", "", "", "", ""): 10.0}
        result = _apply_lca_consolidation(sheets, adds, {})
        assert result["gpu_timeline"].equals(sheets["gpu_timeline"])


# ---------------------------------------------------------------------------
# Tests: consolidation through enrich_perf_report_dict_inplace
# ---------------------------------------------------------------------------
class TestConsolidationViaReport:
    """Multi-op LCA consolidation + speedup/delta on enriched perf dict."""

    @pytest.fixture
    def consolidated_report(self):
        diff_rows = [
            # Multi-op LCA 99: fill_ (25us) + fmha (500us) → flash_bwd (300us)
            _make_diff_stats_row(
                "fill_k",
                "aten::fill_",
                "trace1",
                25.0,
                "lca",
                99,
                busy_time=525.0,
            ),
            _make_diff_stats_row(
                "fmha_k",
                "aiter::fmha_v3_bwd",
                "trace1",
                500.0,
                "lca",
                99,
                busy_time=525.0,
            ),
            _make_diff_stats_row(
                "flash_k",
                "flash_attn::bwd",
                "trace2",
                300.0,
                "lca",
                99,
                busy_time=300.0,
            ),
            # Single-op LCA 10: mm (200us) → mm (150us)
            _make_diff_stats_row(
                "gemm_k",
                "aten::mm",
                "trace1",
                200.0,
                "mm",
                10,
                input_dims="[32,64]",
                input_type="float",
                busy_time=200.0,
            ),
            _make_diff_stats_row(
                "gemm_k2",
                "aten::mm",
                "trace2",
                150.0,
                "mm",
                10,
                input_dims="[32,64]",
                input_type="float",
                busy_time=150.0,
            ),
        ]
        diff_df = pd.DataFrame(diff_rows)

        perf1 = {
            "gpu_timeline": pd.DataFrame({"type": ["total_time"], "time ms": [5.0]}),
            "ops_unique_args": pd.DataFrame(
                {
                    "name": ["aiter::fmha_v3_bwd", "aten::fill_", "aten::mm"],
                    "Input Dims": ["", "", "[32,64]"],
                    "Input type": ["", "", "float"],
                    "Input Strides": ["", "", ""],
                    "Concrete Inputs": ["", "", ""],
                    "total_direct_kernel_time_sum": [500.0, 100.0, 200.0],
                }
            ),
            "unified_perf_summary": pd.DataFrame(
                {
                    "name": ["aiter::fmha_v3_bwd", "aten::fill_", "aten::mm"],
                    "Input Dims": ["", "", "[32,64]"],
                    "Input type": ["", "", "float"],
                    "Input Strides": ["", "", ""],
                    "Concrete Inputs": ["", "", ""],
                    "Kernel Time (µs)_sum": [500.0, 100.0, 200.0],
                }
            ),
        }
        enriched = _enrich_perf_from_diff(diff_df, perf1, None)
        summary = tracediff_perf_summary_from_diff_stats(diff_df)
        return {"enriched": enriched, "summary": summary}

    def test_dominant_op_time_after_consolidation(self, consolidated_report):
        ops = consolidated_report["enriched"]["ops_unique_args"]
        fmha = ops[ops["name"] == "aiter::fmha_v3_bwd"].iloc[0]
        assert fmha["total_direct_kernel_time_sum"] == pytest.approx(525.0)

    def test_non_dominant_op_time_after_consolidation(self, consolidated_report):
        ops = consolidated_report["enriched"]["ops_unique_args"]
        fill_row = ops[ops["name"] == "aten::fill_"].iloc[0]
        assert fill_row["total_direct_kernel_time_sum"] == pytest.approx(75.0)

    def test_single_op_lca_unaffected(self, consolidated_report):
        ops = consolidated_report["enriched"]["ops_unique_args"]
        mm = ops[ops["name"] == "aten::mm"].iloc[0]
        assert mm["total_direct_kernel_time_sum"] == pytest.approx(200.0)
        ups = consolidated_report["enriched"]["unified_perf_summary"]
        mm_u = ups[ups["name"] == "aten::mm"].iloc[0]
        assert mm_u["speedup (trace2/trace1)"] == pytest.approx(150.0 / 200.0)

    def test_dominant_op_speedup(self, consolidated_report):
        ups = consolidated_report["enriched"]["unified_perf_summary"]
        fmha = ups[ups["name"] == "aiter::fmha_v3_bwd"].iloc[0]
        assert fmha["speedup (trace2/trace1)"] == pytest.approx(300.0 / 525.0)

    def test_summary_still_uses_pipe_name(self, consolidated_report):
        summary = consolidated_report["summary"]
        lca99 = summary[summary["lowest_common_ancestor_id"] == 99].iloc[0]
        assert "aiter::fmha_v3_bwd" in lca99["name"]
        assert "aten::fill_" in lca99["name"]
        assert " | " in lca99["name"]


# ---------------------------------------------------------------------------
# Tests: gpu_op_uid direct lookup
# ---------------------------------------------------------------------------
class TestGpuOpUidDirectLookup:
    """Verify gpu_op_uid-based matching between diff_stats and unified_perf_summary."""

    def test_direct_match_via_gpu_op_uid(self):
        """diff_stats row with gpu_op_uid matching a kernel in
        unified_perf_summary.kernel_details_summary resolves correctly,
        even when cpu_op_name is a pseudo-op that wouldn't match by name."""
        diff_rows = [
            _make_diff_stats_row(
                "gemm_kernel",
                "pseudo::synthetic_mm",
                "trace1",
                200.0,
                "lca",
                10,
                input_dims="[32,64]",
                input_type="float",
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
                input_dims="[32,64]",
                input_type="float",
                busy_time=150.0,
                gpu_op_uid=8001,
            ),
        ]
        diff_df = pd.DataFrame(diff_rows)

        perf1 = {
            "unified_perf_summary": pd.DataFrame(
                {
                    "name": ["aten::mm"],
                    "Input Dims": ["[32,64]"],
                    "Input type": ["float"],
                    "Input Strides": [""],
                    "Concrete Inputs": [""],
                    "Kernel Time (µs)_sum": [200.0],
                    "kernel_details_summary": [
                        str([{"gpu_op_uid": 7001, "name": "gemm_kernel", "dur": 200.0}])
                    ],
                }
            ),
        }
        enriched = enrich_perf_report_dict_inplace(
            {k: v.copy() for k, v in perf1.items()},
            diff_df,
            None,
        )
        ups = enriched["unified_perf_summary"]
        assert "speedup (trace2/trace1)" in ups.columns
        assert ups.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(150.0 / 200.0)

    def test_fallback_when_gpu_op_uid_absent(self):
        """When gpu_op_uid is not in diff_stats, falls back to cpu_op_uid/name matching."""
        diff_rows = [
            _make_diff_stats_row(
                "gemm_kernel",
                "aten::mm",
                "trace1",
                200.0,
                "mm",
                10,
                input_dims="[32,64]",
                input_type="float",
                busy_time=200.0,
            ),
            _make_diff_stats_row(
                "gemm_kernel_v2",
                "aten::mm",
                "trace2",
                160.0,
                "mm",
                10,
                input_dims="[32,64]",
                input_type="float",
                busy_time=160.0,
            ),
        ]
        diff_df = pd.DataFrame(diff_rows)

        perf1 = {
            "unified_perf_summary": pd.DataFrame(
                {
                    "name": ["aten::mm"],
                    "Input Dims": ["[32,64]"],
                    "Input type": ["float"],
                    "Input Strides": [""],
                    "Concrete Inputs": [""],
                    "Kernel Time (µs)_sum": [200.0],
                }
            ),
        }
        enriched = enrich_perf_report_dict_inplace(
            {k: v.copy() for k, v in perf1.items()},
            diff_df,
            None,
        )
        ups = enriched["unified_perf_summary"]
        assert "speedup (trace2/trace1)" in ups.columns
        assert ups.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(160.0 / 200.0)

    def test_graceful_degradation_no_kernel_details_column(self):
        """When kernel_details_summary is absent from unified_perf_summary,
        the gpu index is empty and matching falls back to name-based keys."""
        diff_rows = [
            _make_diff_stats_row(
                "gemm_kernel",
                "aten::mm",
                "trace1",
                100.0,
                "mm",
                10,
                input_dims="[32,64]",
                input_type="float",
                busy_time=100.0,
                gpu_op_uid=9001,
            ),
            _make_diff_stats_row(
                "gemm_kernel_v2",
                "aten::mm",
                "trace2",
                90.0,
                "mm",
                10,
                input_dims="[32,64]",
                input_type="float",
                busy_time=90.0,
                gpu_op_uid=9002,
            ),
        ]
        diff_df = pd.DataFrame(diff_rows)

        perf1 = {
            "unified_perf_summary": pd.DataFrame(
                {
                    "name": ["aten::mm"],
                    "Input Dims": ["[32,64]"],
                    "Input type": ["float"],
                    "Input Strides": [""],
                    "Concrete Inputs": [""],
                    "Kernel Time (µs)_sum": [100.0],
                }
            ),
        }
        enriched = enrich_perf_report_dict_inplace(
            {k: v.copy() for k, v in perf1.items()},
            diff_df,
            None,
        )
        ups = enriched["unified_perf_summary"]
        assert "speedup (trace2/trace1)" in ups.columns
        assert ups.iloc[0]["speedup (trace2/trace1)"] == pytest.approx(90.0 / 100.0)

    def test_build_gpu_op_uid_index_empty_when_no_column(self):
        """_build_gpu_op_uid_to_key_index returns empty dict when column is missing."""
        df = pd.DataFrame({"name": ["aten::mm"], "Input Dims": [""]})
        idx = _build_gpu_op_uid_to_key_index(df)
        assert idx == {}

    def test_build_gpu_op_uid_index_parses_kernel_details(self):
        """_build_gpu_op_uid_to_key_index correctly maps UIDs to row keys."""
        df = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "Input Dims": ["[32,64]"],
                "Input type": ["float"],
                "Input Strides": [""],
                "Concrete Inputs": [""],
                "kernel_details_summary": [
                    str(
                        [
                            {"gpu_op_uid": 5001, "name": "k1", "dur": 100.0},
                            {"gpu_op_uid": 5002, "name": "k2", "dur": 50.0},
                        ]
                    )
                ],
            }
        )
        idx = _build_gpu_op_uid_to_key_index(df)
        assert 5001 in idx
        assert 5002 in idx
        expected_key = _make_str_key("aten::mm", df.iloc[0])
        assert idx[5001] == expected_key
        assert idx[5002] == expected_key

    def test_gpu_op_uid_preferred_in_trace2_uid_set(self):
        """_trace2_cpu_op_uid_set_for_lca prefers gpu_op_uid over cpu_op_uid."""
        from TraceLens.Reporting.tracediff_comparison_extension import (
            _trace2_cpu_op_uid_set_for_lca,
        )

        rows = [
            _make_diff_stats_row(
                "k1",
                "aten::mm",
                "trace2",
                80.0,
                "mm",
                10,
                busy_time=80.0,
                gpu_op_uid=3001,
                cpu_op_uid=42,
            ),
            _make_diff_stats_row(
                "k2",
                "aten::mm",
                "trace2",
                60.0,
                "mm",
                10,
                busy_time=60.0,
                gpu_op_uid=3002,
                cpu_op_uid=42,
            ),
        ]
        df = pd.DataFrame(rows)
        uid_set = _trace2_cpu_op_uid_set_for_lca(df, 10)
        assert uid_set == {3001, 3002}
