###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.Reporting.generate_perf_report_jax_analysis."""

import importlib
from unittest import mock

import pandas as pd

_JAX_ANALYSIS_MOD = importlib.import_module(
    "TraceLens.Reporting.generate_perf_report_jax_analysis"
)
calculate_gpu_event_statistics = _JAX_ANALYSIS_MOD.calculate_gpu_event_statistics
generate_perf_report_jax_analysis = _JAX_ANALYSIS_MOD.generate_perf_report_jax_analysis
JaxAnalyses = _JAX_ANALYSIS_MOD.JaxAnalyses


def _sample_averages_df():
    """Nine-row averages frame matching JaxAnalyses.summarize_gpu_events layout."""
    return pd.DataFrame(
        {
            "type": [
                "total_time",
                "computation_time",
                "total_comm_time",
                "exposed_comm_time",
                "memcpy_time",
                "idle_time",
                "other",
                "overlap",
                "misc",
            ],
            "time ms": [100.0, 80.0, 20.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "percent": [100.0] * 9,
        }
    )


def _mock_side_inputs():
    categorized = pd.DataFrame(
        {"time ms": [10.0], "percent": [10.0]},
        index=["gemm"],
    )
    xla_events = pd.DataFrame(
        {"time ms": [5.0], "percent": [5.0]},
        index=["dot_general_123"],
    )
    return categorized, xla_events


def test_calculate_gpu_event_statistics_adds_overlapped_comm():
    categorized, xla_events = _mock_side_inputs()
    with mock.patch.object(
        JaxAnalyses,
        "summarize_gpu_events",
        return_value=(_sample_averages_df(), categorized, xla_events),
    ):
        averages, categorized_out, xla_grouped = calculate_gpu_event_statistics(
            "/fake/profile.xplane.pb"
        )

    assert not categorized_out.empty
    assert "short_name_grouped" in xla_grouped.columns
    assert len(averages) == 9


def test_generate_perf_report_jax_analysis_writes_csvs(tmp_path):
    categorized, xla_events = _mock_side_inputs()
    gemms = pd.DataFrame({"time ms": [1.0], "percent": [1.0]}, index=["gemm1"])
    gemms_detailed = pd.DataFrame({"name": ["gemm1"], "tflops": [1.0]})

    with mock.patch.object(
        JaxAnalyses,
        "summarize_gpu_events",
        return_value=(_sample_averages_df(), categorized, xla_events),
    ), mock.patch.object(
        JaxAnalyses,
        "summarize_gpu_gemm_events_from_pb",
        return_value=gemms,
    ), mock.patch.object(
        JaxAnalyses,
        "gemm_performance_from_pb",
        return_value=gemms_detailed,
    ):
        generate_perf_report_jax_analysis(
            profile_xplane_pb_path="/fake/profile.xplane.pb",
            output_path=str(tmp_path),
            output_filename="jax_analysis",
            output_table_formats=[".csv"],
        )

    expected = [
        "jax_analysis_gpu_events_averages.csv",
        "jax_analysis_gpu_events_categorized_mean.csv",
        "jax_analysis_xla_grouped.csv",
        "jax_analysis_gemms.csv",
        "jax_analysis_gemms_detailed.csv",
    ]
    for name in expected:
        assert (tmp_path / name).exists()
