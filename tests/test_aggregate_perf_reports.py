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

from TraceLens.Reporting.aggregate_perf_reports import (
    AggregationError,
    ReportInput,
    aggregate_perf_report_csvs,
)

KT = "Kernel Time (µs)"
BY_NODE = "combine_by_node"


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


def _report(tmp_path, report, phase="decode", node="n0", **tables):
    return ReportInput(phase, node, _write(tmp_path, report, **tables))


def _row(df, name, dims, node=None):
    match = df[(df["name"] == name) & (df["Input Dims"] == dims)]
    if node is not None:
        match = match[match["node"] == node]
    assert len(match) == 1
    return match.iloc[0]


def test_same_node_chunks_reproduce_the_full_report(tmp_path):
    full = {("mm", "(4, 4)"): [1.0, 2.0, 4.0, 8.0, 3.0], ("add", "(4,)"): [5.0, 7.0]}
    first = {("mm", "(4, 4)"): [1.0, 2.0], ("add", "(4,)"): [5.0]}
    second = {("mm", "(4, 4)"): [4.0, 8.0, 3.0], ("add", "(4,)"): [7.0]}
    reports = [
        _report(tmp_path, "a", unified_perf_summary=_unified(first)),
        _report(tmp_path, "b", unified_perf_summary=_unified(second)),
    ]
    got = aggregate_perf_report_csvs(reports, BY_NODE)["unified_perf_summary"]
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
        assert row["node"] == "n0"
    assert got["Cumulative Percentage (%)"].iloc[-1] == pytest.approx(100.0)


def test_nodes_stack_as_blocks_with_their_own_percentages(tmp_path):
    prefill = _unified({("mm", "(8, 8)"): [10.0, 4.0], ("copy_", "(32,)"): [6.0]})
    decode = _unified(
        {("moe", "(32, 8)"): [50.0, 30.0], ("copy_", "(32,)"): [15.0, 5.0]}
    )
    reports = [
        _report(tmp_path, "p", "prefill", "w_prefill", unified_perf_summary=prefill),
        _report(tmp_path, "d", "decode", "w_decode", unified_perf_summary=decode),
    ]
    got = aggregate_perf_report_csvs(reports, BY_NODE)["unified_perf_summary"]
    # Blocks follow report order; each block is time-sorted on its own.
    assert list(zip(got["node"], got["name"])) == [
        ("w_prefill", "mm"),
        ("w_prefill", "copy_"),
        ("w_decode", "moe"),
        ("w_decode", "copy_"),
    ]
    # A shape shared by two nodes is not merged.
    assert _row(got, "copy_", "(32,)", "w_prefill")[f"{KT}_sum"] == pytest.approx(6.0)
    assert _row(got, "copy_", "(32,)", "w_decode")[f"{KT}_sum"] == pytest.approx(20.0)
    assert _row(got, "mm", "(8, 8)", "w_prefill")["Percentage (%)"] == pytest.approx(
        70.0
    )
    assert _row(got, "moe", "(32, 8)", "w_decode")["Percentage (%)"] == pytest.approx(
        80.0
    )
    for node in ("w_prefill", "w_decode"):
        block = got[got["node"] == node]
        assert block["Cumulative Percentage (%)"].iloc[-1] == pytest.approx(100.0)
        assert block["phase"].nunique() == 1


def test_node_block_lists_every_phase_it_combined(tmp_path):
    reports = [
        _report(
            tmp_path,
            "a",
            "prefill",
            unified_perf_summary=_unified({("mm", "x"): [1.0]}),
        ),
        _report(
            tmp_path, "b", "decode", unified_perf_summary=_unified({("mm", "x"): [3.0]})
        ),
    ]
    got = aggregate_perf_report_csvs(reports, BY_NODE)["unified_perf_summary"]
    assert len(got) == 1
    assert got["phase"].iloc[0] == "decode+prefill"
    assert got["operation_count"].iloc[0] == 2


def test_gpu_timeline_has_a_block_per_node_and_an_all_block(tmp_path):
    reports = [
        _report(tmp_path, "p1", "prefill", "w_p", gpu_timeline=_timeline(12.0, 28.0)),
        _report(tmp_path, "p2", "prefill", "w_p", gpu_timeline=_timeline(12.0, 28.0)),
        _report(tmp_path, "d", "decode", "w_d", gpu_timeline=_timeline(108.0, 12.0)),
    ]
    got = aggregate_perf_report_csvs(reports, BY_NODE)["gpu_timeline"]
    assert list(pd.unique(got["node"])) == ["w_p", "w_d", "all"]
    blocks = {node: df.set_index("type") for node, df in got.groupby("node")}
    assert blocks["w_p"].loc["busy_time", "time ms"] == pytest.approx(24.0)
    assert blocks["w_p"].loc["busy_time", "percent"] == pytest.approx(30.0)
    assert blocks["w_d"].loc["busy_time", "percent"] == pytest.approx(90.0)
    assert blocks["all"].loc["busy_time", "time ms"] == pytest.approx(132.0)
    assert blocks["all"].loc["total_time", "time ms"] == pytest.approx(200.0)
    assert blocks["all"].loc["busy_time", "percent"] == pytest.approx(66.0)
    assert blocks["all"].loc["busy_time", "phase"] == "decode+prefill"
    # Readers that keep the last row per type see the all block.
    last_total = got[got["type"] == "total_time"].iloc[-1]
    assert last_total["node"] == "all"


@pytest.mark.parametrize("policy", ["combine", "combine_by_phase", "combine_stack"])
def test_reserved_policies_are_not_supported_yet(tmp_path, policy):
    reports = [_report(tmp_path, "a", gpu_timeline=_timeline(1.0, 1.0))]
    with pytest.raises(AggregationError, match="not supported yet"):
        aggregate_perf_report_csvs(reports, policy)


@pytest.mark.parametrize("node", [None, "", "  "])
def test_by_node_requires_a_node(tmp_path, node):
    reports = [_report(tmp_path, "a", node=node, gpu_timeline=_timeline(1.0, 1.0))]
    with pytest.raises(AggregationError, match="needs a node"):
        aggregate_perf_report_csvs(reports, BY_NODE)


def test_all_is_a_reserved_node_name(tmp_path):
    reports = [_report(tmp_path, "a", node="all", gpu_timeline=_timeline(1.0, 1.0))]
    with pytest.raises(AggregationError, match="reserved"):
        aggregate_perf_report_csvs(reports, BY_NODE)


def test_single_instance_std_pools_to_nan_like_pandas(tmp_path):
    reports = [
        _report(tmp_path, "a", unified_perf_summary=_unified({("mm", "x"): [2.0]}))
    ]
    got = aggregate_perf_report_csvs(reports, BY_NODE)["unified_perf_summary"]
    assert np.isnan(got[f"{KT}_std"].iloc[0])


def test_first_columns_take_the_first_report(tmp_path):
    a = _unified({("mm", "x"): [1.0]})
    b = _unified({("mm", "x"): [5.0, 5.0]})
    a["ex_UID"], b["ex_UID"] = 7, 9
    reports = [
        _report(tmp_path, "a", unified_perf_summary=a),
        _report(tmp_path, "b", unified_perf_summary=b),
    ]
    got = aggregate_perf_report_csvs(reports, BY_NODE)["unified_perf_summary"]
    assert got["ex_UID"].iloc[0] == 7
    # Medians come from the report with the most instances.
    assert got[f"{KT}_median"].iloc[0] == pytest.approx(5.0)


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
        _report(tmp_path, "a", ops_summary=summary(["GEMM"], 1.0)),
        _report(tmp_path, "b", ops_summary=summary(["other"], 3.0)),
    ]
    got = aggregate_perf_report_csvs(reports, BY_NODE)["ops_summary"]
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
        _report(tmp_path, "a", GEMM=gemm(32, [1.0, 2.0])),
        _report(tmp_path, "b", GEMM=pd.concat([gemm(32, [3.0]), gemm(64, [9.0])])),
    ]
    got = aggregate_perf_report_csvs(reports, BY_NODE)["GEMM"]
    assert sorted(got["param: M"]) == ["32", "64"]
    m32 = got[got["param: M"] == "32"].iloc[0]
    assert m32["name_count"] == 3
    assert m32[f"{KT}_mean"] == pytest.approx(2.0)
    assert m32[f"{KT}_std"] == pytest.approx(1.0)


def test_writes_one_csv_per_table(tmp_path):
    reports = [_report(tmp_path, "a", gpu_timeline=_timeline(1.0, 1.0))]
    out_dir = tmp_path / "agg"
    aggregate_perf_report_csvs(reports, BY_NODE, str(out_dir))
    assert os.listdir(out_dir) == ["gpu_timeline.csv"]


def test_unknown_column_raises(tmp_path):
    df = _unified({("mm", "x"): [1.0]})
    df["brand_new_metric"] = 1.0
    reports = [_report(tmp_path, "a", unified_perf_summary=df)]
    with pytest.raises(AggregationError, match="brand_new_metric"):
        aggregate_perf_report_csvs(reports, BY_NODE)


def test_unknown_csv_raises(tmp_path):
    reports = [_report(tmp_path, "a", kernel_summary=pd.DataFrame({"x": [1]}))]
    with pytest.raises(AggregationError, match="kernel_summary"):
        aggregate_perf_report_csvs(reports, BY_NODE)


def test_unknown_policy_raises(tmp_path):
    reports = [_report(tmp_path, "a", gpu_timeline=_timeline(1.0, 1.0))]
    with pytest.raises(AggregationError, match="unknown aggregation policy"):
        aggregate_perf_report_csvs(reports, "average")


def test_empty_reports_raise():
    with pytest.raises(AggregationError, match="at least one report"):
        aggregate_perf_report_csvs([], BY_NODE)


def test_missing_directory_raises(tmp_path):
    reports = [ReportInput("decode", "n0", str(tmp_path / "missing"))]
    with pytest.raises(AggregationError, match="not found"):
        aggregate_perf_report_csvs(reports, BY_NODE)


def test_non_csv_files_are_ignored(tmp_path):
    report = _report(tmp_path, "a", gpu_timeline=_timeline(1.0, 1.0))
    with open(
        os.path.join(report.csv_dir, "notes.txt"), "w", encoding="utf-8"
    ) as handle:
        handle.write("ignore")
    got = aggregate_perf_report_csvs([report], BY_NODE)
    assert list(got) == ["gpu_timeline"]


def test_means_without_a_count_raise(tmp_path):
    df = pd.DataFrame({"name": ["mm"], "mean_duration_us": [1.0]})
    reports = [_report(tmp_path, "a", ops_summary=df)]
    with pytest.raises(AggregationError, match="instance count"):
        aggregate_perf_report_csvs(reports, BY_NODE)


def test_std_without_a_mean_raises(tmp_path):
    df = pd.DataFrame({"name": ["mm"], "operation_count": [2], "latency_std": [1.0]})
    reports = [_report(tmp_path, "a", ops_summary=df)]
    with pytest.raises(AggregationError, match="latency_std"):
        aggregate_perf_report_csvs(reports, BY_NODE)


def test_zero_counts_make_the_mean_nan(tmp_path):
    df = _unified({("mm", "x"): [1.0, 2.0]})
    df["operation_count"] = 0
    reports = [_report(tmp_path, "a", unified_perf_summary=df)]
    got = aggregate_perf_report_csvs(reports, BY_NODE)["unified_perf_summary"]
    assert np.isnan(got["mean_duration_us"].iloc[0])


def test_table_without_a_time_column_is_unchanged(tmp_path):
    df = pd.DataFrame({"op category": ["GEMM"], "Count": [3]})
    reports = [_report(tmp_path, "a", ops_summary_by_category=df)]
    got = aggregate_perf_report_csvs(reports, BY_NODE)["ops_summary_by_category"]
    assert got["Count"].iloc[0] == 3
    assert got["phase"].iloc[0] == "decode"
    assert "Percentage (%)" not in got.columns


def test_timeline_without_total_time_has_nan_percent(tmp_path):
    df = pd.DataFrame({"type": ["busy_time"], "time ms": [5.0], "percent": [100.0]})
    reports = [_report(tmp_path, "a", gpu_timeline=df)]
    got = aggregate_perf_report_csvs(reports, BY_NODE)["gpu_timeline"]
    assert got["percent"].isna().all()


def test_first_column_all_missing_stays_nan(tmp_path):
    df = _unified({("mm", "x"): [1.0]})
    df["ex_UID"] = np.nan
    reports = [_report(tmp_path, "a", unified_perf_summary=df)]
    got = aggregate_perf_report_csvs(reports, BY_NODE)["unified_perf_summary"]
    assert np.isnan(got["ex_UID"].iloc[0])


def test_input_node_and_phase_columns_are_replaced(tmp_path):
    df = _timeline(1.0, 1.0)
    df["phase"] = "stale"
    df["node"] = "stale"
    reports = [_report(tmp_path, "a", gpu_timeline=df)]
    got = aggregate_perf_report_csvs(reports, BY_NODE)["gpu_timeline"]
    assert set(got["phase"]) == {"decode"}
    assert set(got["node"]) == {"n0", "all"}
