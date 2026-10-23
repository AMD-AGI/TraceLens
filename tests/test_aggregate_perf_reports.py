###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Aggregation rules on small synthetic perf report CSVs."""

import os

import numpy as np
import pandas as pd
import pytest

from TraceLens.Reporting.aggregate_perf_reports_experimental import (
    AggregationError,
    aggregate_perf_report_csvs,
)

KT = "Kernel Time (µs)"


def _unified(instances):
    """Build a unified_perf_summary the way TreePerf does, from raw instances.

    ``instances`` maps ``(name, dims)`` to a list of kernel times.
    """
    rows = []
    for (name, dims), times in instances.items():
        t = pd.Series(times, dtype=float)
        rows.append(
            {
                "name": name,
                "op category": "GEMM",
                "Input Dims": dims,
                "ex_UID": len(rows),
                "operation_count": len(t),
                "total_duration_us": t.sum(),
                "mean_duration_us": t.mean(),
                "std_duration_us": t.std(),
                f"{KT}_mean": t.mean(),
                f"{KT}_std": t.std(),
                f"{KT}_sum": t.sum(),
                f"{KT}_median": t.median(),
                f"{KT}_min": t.min(),
                f"{KT}_max": t.max(),
            }
        )
    df = pd.DataFrame(rows)
    df["Percentage (%)"] = df[f"{KT}_sum"] / df[f"{KT}_sum"].sum() * 100
    df["Cumulative Percentage (%)"] = df["Percentage (%)"].cumsum()
    return df


def _timeline(busy, idle):
    return pd.DataFrame(
        {
            "type": ["busy_time", "idle_time", "total_time"],
            "time ms": [busy, idle, busy + idle],
            "percent": [0.0, 0.0, 100.0],
        }
    )


def _write(tmp_path, report, **tables):
    path = os.path.join(str(tmp_path), report)
    os.makedirs(path, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(os.path.join(path, f"{name}.csv"), index=False)
    return path


def _row(df, name, dims, phase=None):
    match = df[(df["name"] == name) & (df["Input Dims"] == dims)]
    if phase is not None:
        match = match[match["phase"] == phase]
    assert len(match) == 1
    return match.iloc[0]


def test_same_phase_chunks_reproduce_the_full_report(tmp_path):
    full = {("mm", "(4, 4)"): [1.0, 2.0, 4.0, 8.0, 3.0], ("add", "(4,)"): [5.0, 7.0]}
    first = {("mm", "(4, 4)"): [1.0, 2.0], ("add", "(4,)"): [5.0]}
    second = {("mm", "(4, 4)"): [4.0, 8.0, 3.0], ("add", "(4,)"): [7.0]}
    reports = [
        ("decode", _write(tmp_path, "a", unified_perf_summary=_unified(first))),
        ("decode", _write(tmp_path, "b", unified_perf_summary=_unified(second))),
    ]
    out = aggregate_perf_report_csvs(reports, "combine_only_same_phase")
    got = out["unified_perf_summary"]
    want = _unified(full)
    assert len(got) == len(want)
    for _, expected in want.iterrows():
        row = _row(got, expected["name"], expected["Input Dims"])
        for col in (
            "operation_count",
            "total_duration_us",
            "mean_duration_us",
            "std_duration_us",
            f"{KT}_mean",
            f"{KT}_std",
            f"{KT}_sum",
            f"{KT}_min",
            f"{KT}_max",
            "Percentage (%)",
        ):
            assert row[col] == pytest.approx(expected[col]), col
        assert row["phase"] == "decode"
    assert got["Cumulative Percentage (%)"].iloc[-1] == pytest.approx(100.0)


def test_single_instance_std_pools_to_nan_like_pandas(tmp_path):
    reports = [
        (
            "decode",
            _write(tmp_path, "a", unified_perf_summary=_unified({("mm", "x"): [2.0]})),
        )
    ]
    got = aggregate_perf_report_csvs(reports, "combine")["unified_perf_summary"]
    assert np.isnan(got[f"{KT}_std"].iloc[0])


def test_different_phases_stay_separate_under_same_phase_policy(tmp_path):
    prefill = _unified({("mm", "(4, 4)"): [10.0, 30.0]})
    decode = _unified({("mm", "(4, 4)"): [1.0, 3.0], ("add", "(4,)"): [6.0]})
    reports = [
        ("prefill", _write(tmp_path, "p", unified_perf_summary=prefill)),
        ("decode", _write(tmp_path, "d", unified_perf_summary=decode)),
    ]
    got = aggregate_perf_report_csvs(reports, "combine_only_same_phase")[
        "unified_perf_summary"
    ]
    assert len(got) == 3
    assert _row(got, "mm", "(4, 4)", "prefill")[f"{KT}_sum"] == pytest.approx(40.0)
    assert _row(got, "mm", "(4, 4)", "decode")[f"{KT}_sum"] == pytest.approx(4.0)
    assert _row(got, "mm", "(4, 4)", "prefill")["Percentage (%)"] == pytest.approx(
        40.0 / 50.0 * 100
    )
    assert got["Percentage (%)"].sum() == pytest.approx(100.0)


def test_combine_policy_merges_rows_across_phases(tmp_path):
    prefill = _unified({("mm", "(4, 4)"): [10.0, 30.0]})
    decode = _unified({("mm", "(4, 4)"): [1.0, 3.0]})
    reports = [
        ("prefill", _write(tmp_path, "p", unified_perf_summary=prefill)),
        ("decode", _write(tmp_path, "d", unified_perf_summary=decode)),
    ]
    got = aggregate_perf_report_csvs(reports, "combine")["unified_perf_summary"]
    row = _row(got, "mm", "(4, 4)")
    assert row["phase"] == "decode+prefill"
    assert row["operation_count"] == 4
    assert row[f"{KT}_mean"] == pytest.approx(11.0)
    assert row[f"{KT}_std"] == pytest.approx(pd.Series([10.0, 30.0, 1.0, 3.0]).std())


def test_first_columns_take_the_first_report(tmp_path):
    a = _unified({("mm", "x"): [1.0]})
    b = _unified({("mm", "x"): [5.0, 5.0]})
    a["ex_UID"], b["ex_UID"] = 7, 9
    reports = [
        ("decode", _write(tmp_path, "a", unified_perf_summary=a)),
        ("decode", _write(tmp_path, "b", unified_perf_summary=b)),
    ]
    got = aggregate_perf_report_csvs(reports, "combine")["unified_perf_summary"]
    assert got["ex_UID"].iloc[0] == 7
    # Medians come from the report with the most instances.
    assert got[f"{KT}_median"].iloc[0] == pytest.approx(5.0)


def test_gpu_timeline_is_one_summed_timeline(tmp_path):
    reports = [
        ("prefill", _write(tmp_path, "p", gpu_timeline=_timeline(30.0, 10.0))),
        ("decode", _write(tmp_path, "d", gpu_timeline=_timeline(50.0, 10.0))),
    ]
    got = aggregate_perf_report_csvs(reports, "combine_only_same_phase")[
        "gpu_timeline"
    ].set_index("type")
    assert got.loc["busy_time", "time ms"] == pytest.approx(80.0)
    assert got.loc["total_time", "time ms"] == pytest.approx(100.0)
    assert got.loc["busy_time", "percent"] == pytest.approx(80.0)
    assert got.loc["busy_time", "phase"] == "decode+prefill"


def test_ops_summary_categories_are_unioned(tmp_path):
    def summary(categories, ms):
        return pd.DataFrame(
            {
                "name": ["aten::mm"],
                "Count": [2],
                "Categories": [str(categories)],
                "total_direct_kernel_time_ms": [ms],
                "Percentage (%)": [100.0],
                "Cumulative Percentage (%)": [100.0],
            }
        )

    reports = [
        ("decode", _write(tmp_path, "a", ops_summary=summary(["GEMM"], 1.0))),
        ("decode", _write(tmp_path, "b", ops_summary=summary(["other"], 3.0))),
    ]
    got = aggregate_perf_report_csvs(reports, "combine")["ops_summary"]
    assert got["Categories"].iloc[0] == str(["GEMM", "other"])
    assert got["Count"].iloc[0] == 4
    assert got["total_direct_kernel_time_ms"].iloc[0] == pytest.approx(4.0)


def test_roofline_sheet_is_keyed_by_name_and_params(tmp_path):
    def gemm(m, times):
        t = pd.Series(times)
        return pd.DataFrame(
            {
                "name": ["aten::mm"],
                "param: M": [m],
                "GFLOPS_first": [1.0],
                f"{KT}_mean": [t.mean()],
                f"{KT}_std": [t.std()],
                f"{KT}_sum": [t.sum()],
                "name_count": [len(t)],
                "UID_first": [1],
            }
        )

    reports = [
        ("decode", _write(tmp_path, "a", GEMM=gemm(32, [1.0, 2.0]))),
        (
            "decode",
            _write(tmp_path, "b", GEMM=pd.concat([gemm(32, [3.0]), gemm(64, [9.0])])),
        ),
    ]
    got = aggregate_perf_report_csvs(reports, "combine")["GEMM"]
    assert sorted(got["param: M"]) == ["32", "64"]
    m32 = got[got["param: M"] == "32"].iloc[0]
    assert m32["name_count"] == 3
    assert m32[f"{KT}_mean"] == pytest.approx(2.0)
    assert m32[f"{KT}_std"] == pytest.approx(1.0)


def test_writes_one_csv_per_table(tmp_path):
    reports = [("decode", _write(tmp_path, "a", gpu_timeline=_timeline(1.0, 1.0)))]
    out_dir = tmp_path / "agg"
    aggregate_perf_report_csvs(reports, "combine", str(out_dir))
    assert os.listdir(out_dir) == ["gpu_timeline.csv"]


def test_unknown_column_raises(tmp_path):
    df = _unified({("mm", "x"): [1.0]})
    df["brand_new_metric"] = 1.0
    reports = [("decode", _write(tmp_path, "a", unified_perf_summary=df))]
    with pytest.raises(AggregationError, match="brand_new_metric"):
        aggregate_perf_report_csvs(reports, "combine")


def test_unknown_csv_raises(tmp_path):
    reports = [
        ("decode", _write(tmp_path, "a", kernel_summary=pd.DataFrame({"x": [1]})))
    ]
    with pytest.raises(AggregationError, match="kernel_summary"):
        aggregate_perf_report_csvs(reports, "combine")


def test_unknown_policy_raises(tmp_path):
    reports = [("decode", _write(tmp_path, "a", gpu_timeline=_timeline(1.0, 1.0)))]
    with pytest.raises(AggregationError, match="policy"):
        aggregate_perf_report_csvs(reports, "average")


def test_empty_reports_raise():
    with pytest.raises(AggregationError, match="at least one report"):
        aggregate_perf_report_csvs([], "combine")


def test_missing_directory_raises(tmp_path):
    with pytest.raises(AggregationError, match="not found"):
        aggregate_perf_report_csvs([("decode", str(tmp_path / "missing"))], "combine")


def test_non_csv_files_are_ignored(tmp_path):
    path = _write(tmp_path, "a", gpu_timeline=_timeline(1.0, 1.0))
    with open(os.path.join(path, "notes.txt"), "w", encoding="utf-8") as handle:
        handle.write("ignore")
    got = aggregate_perf_report_csvs([("decode", path)], "combine")
    assert list(got) == ["gpu_timeline"]


def test_means_without_a_count_raise(tmp_path):
    df = pd.DataFrame({"name": ["mm"], "mean_duration_us": [1.0]})
    reports = [("decode", _write(tmp_path, "a", ops_summary=df))]
    with pytest.raises(AggregationError, match="instance count"):
        aggregate_perf_report_csvs(reports, "combine")


def test_std_without_a_mean_raises(tmp_path):
    df = pd.DataFrame({"name": ["mm"], "operation_count": [2], "latency_std": [1.0]})
    reports = [("decode", _write(tmp_path, "a", ops_summary=df))]
    with pytest.raises(AggregationError, match="latency_std"):
        aggregate_perf_report_csvs(reports, "combine")


def test_zero_counts_make_the_mean_nan(tmp_path):
    df = _unified({("mm", "x"): [1.0, 2.0]})
    df["operation_count"] = 0
    reports = [("decode", _write(tmp_path, "a", unified_perf_summary=df))]
    got = aggregate_perf_report_csvs(reports, "combine")["unified_perf_summary"]
    assert np.isnan(got["mean_duration_us"].iloc[0])


def test_table_without_a_time_column_is_unchanged(tmp_path):
    df = pd.DataFrame({"op category": ["GEMM"], "Count": [3]})
    reports = [("decode", _write(tmp_path, "a", ops_summary_by_category=df))]
    got = aggregate_perf_report_csvs(reports, "combine")["ops_summary_by_category"]
    assert got["Count"].iloc[0] == 3
    assert got["phase"].iloc[0] == "decode"
    assert "Percentage (%)" not in got.columns


def test_timeline_without_total_time_has_nan_percent(tmp_path):
    df = pd.DataFrame({"type": ["busy_time"], "time ms": [5.0], "percent": [100.0]})
    reports = [("decode", _write(tmp_path, "a", gpu_timeline=df))]
    got = aggregate_perf_report_csvs(reports, "combine")["gpu_timeline"]
    assert np.isnan(got["percent"].iloc[0])


def test_first_column_all_missing_stays_nan(tmp_path):
    df = _unified({("mm", "x"): [1.0]})
    df["ex_UID"] = np.nan
    reports = [("decode", _write(tmp_path, "a", unified_perf_summary=df))]
    got = aggregate_perf_report_csvs(reports, "combine")["unified_perf_summary"]
    assert np.isnan(got["ex_UID"].iloc[0])


def test_input_phase_column_is_replaced(tmp_path):
    df = _timeline(1.0, 1.0)
    df["phase"] = "stale"
    reports = [("decode", _write(tmp_path, "a", gpu_timeline=df))]
    got = aggregate_perf_report_csvs(reports, "combine")["gpu_timeline"]
    assert set(got["phase"]) == {"decode"}
