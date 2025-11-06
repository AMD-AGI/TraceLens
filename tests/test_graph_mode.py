###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# This test constructs a minimal synthetic trace to exercise graph mode (CUDA/HIP GraphLaunch) with two identical replays.
# It verifies that reporting correctly groups these replays and aggregates per-kernel-position stats via ops_unique_args.
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:Input list of events is empty.*:UserWarning",
    "ignore:Input DataFrame is empty.*:UserWarning",
    "ignore:Source column 'kernel_details__summarize_kernel_stats' not found.*:UserWarning",
)
import json
import os
from typing import List, Dict

import pandas as pd

from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)


def _mk_event(
    cat: str, name: str, ts: float, dur: float, pid: int, tid: int, args: Dict
) -> Dict:
    return {
        "ph": "X",
        "cat": cat,
        "name": name,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "dur": dur,
        "args": args,
    }


def _mk_graph_trace(
    second_durations: List[float] = [22.0, 28.0, 41.0]
) -> Dict[str, List[Dict]]:
    """
    Build a minimal trace that exercises graph-mode:
      - 2 cudaGraphLaunch runtime events
      - Each followed by identical kernel sequences sharing the same correlation id
    """
    events: List[Dict] = []

    # Replay 1: correlation 500
    corr1 = 500
    # Enclose the graph launch within a CPU op so kernel details have a parent cpu_op for grouping
    events.append(
        _mk_event(
            cat="cpu_op",
            name="aten::graph_wrapper_fwd",
            ts=999_900.0,
            dur=500.0,
            pid=470,
            tid=711,
            args={
                "Input Dims": [(1, 128)],
                "Input Strides": [(128, 1)],
                "Input type": ["fp32"],
                "Concrete Inputs": ["synthetic"],
            },
        )
    )
    events.append(
        _mk_event(
            cat="cuda_runtime",
            name="cudaGraphLaunch",
            ts=1_000_000.0,
            dur=10.0,
            pid=470,
            tid=711,
            args={"cbid": 311, "correlation": corr1},
        )
    )
    # 3 kernels in replay 1 (same stream/grid/block to form a stable signature)
    kts = 1_000_050.0
    events.append(
        _mk_event(
            cat="kernel",
            name="kernel_A",
            ts=kts,
            dur=20.0,
            pid=0,
            tid=7,
            args={
                "device": 0,
                "context": 1,
                "stream": 7,
                "grid": [64, 1, 1],
                "block": [256, 1, 1],
                "correlation": corr1,
            },
        )
    )
    events.append(
        _mk_event(
            cat="kernel",
            name="kernel_B",
            ts=kts + 30.0,
            dur=30.0,
            pid=0,
            tid=7,
            args={
                "device": 0,
                "context": 1,
                "stream": 7,
                "grid": [64, 1, 1],
                "block": [256, 1, 1],
                "correlation": corr1,
            },
        )
    )
    events.append(
        _mk_event(
            cat="kernel",
            name="kernel_C",
            ts=kts + 70.0,
            dur=40.0,
            pid=0,
            tid=7,
            args={
                "device": 0,
                "context": 1,
                "stream": 7,
                "grid": [64, 1, 1],
                "block": [256, 1, 1],
                "correlation": corr1,
            },
        )
    )
    # Add ac2g markers so TraceToTree can attach at least one GPU event to the runtime launch,
    # which is needed for df_gpu_timeline to find GPU events in synthetic traces.
    events.append(
        {
            "ph": "s",
            "id": corr1,
            "pid": 0,
            "tid": 7,
            "ts": kts,
            "cat": "ac2g",
            "name": "ac2g",
        }
    )
    events.append(
        {
            "ph": "f",
            "id": corr1,
            "pid": 0,
            "tid": 7,
            "ts": kts,
            "cat": "ac2g",
            "name": "ac2g",
            "bp": "e",
        }
    )

    # Replay 2: correlation 501 (identical structure -> same signature)
    corr2 = 501
    # Enclose the second graph launch within a CPU op as well
    events.append(
        _mk_event(
            cat="cpu_op",
            name="aten::graph_wrapper_fwd",
            ts=1_999_900.0,
            dur=500.0,
            pid=470,
            tid=711,
            args={
                "Input Dims": [(1, 128)],
                "Input Strides": [(128, 1)],
                "Input type": ["fp32"],
                "Concrete Inputs": ["synthetic"],
            },
        )
    )
    events.append(
        _mk_event(
            cat="cuda_runtime",
            name="cudaGraphLaunch",
            ts=2_000_000.0,
            dur=12.0,
            pid=470,
            tid=711,
            args={"cbid": 311, "correlation": corr2},
        )
    )
    kts2 = 2_000_060.0
    events.append(
        _mk_event(
            cat="kernel",
            name="kernel_A",
            ts=kts2,
            dur=second_durations[0],
            pid=0,
            tid=7,
            args={
                "device": 0,
                "context": 1,
                "stream": 7,
                "grid": [64, 1, 1],
                "block": [256, 1, 1],
                "correlation": corr2,
            },
        )
    )
    events.append(
        _mk_event(
            cat="kernel",
            name="kernel_B",
            ts=kts2 + 30.0,
            dur=second_durations[1],
            pid=0,
            tid=7,
            args={
                "device": 0,
                "context": 1,
                "stream": 7,
                "grid": [64, 1, 1],
                "block": [256, 1, 1],
                "correlation": corr2,
            },
        )
    )
    events.append(
        _mk_event(
            cat="kernel",
            name="kernel_C",
            ts=kts2 + 65.0,
            dur=second_durations[2],
            pid=0,
            tid=7,
            args={
                "device": 0,
                "context": 1,
                "stream": 7,
                "grid": [64, 1, 1],
                "block": [256, 1, 1],
                "correlation": corr2,
            },
        )
    )
    # ac2g markers for the second replay as well
    events.append(
        {
            "ph": "s",
            "id": corr2,
            "pid": 0,
            "tid": 7,
            "ts": kts2,
            "cat": "ac2g",
            "name": "ac2g",
        }
    )
    events.append(
        {
            "ph": "f",
            "id": corr2,
            "pid": 0,
            "tid": 7,
            "ts": kts2,
            "cat": "ac2g",
            "name": "ac2g",
            "bp": "e",
        }
    )

    return {"traceEvents": events}


@pytest.mark.parametrize(
    "second_durations, expected_total",
    [
        ([22.0, 28.0, 41.0], 181.0),
        ([20.0, 30.0, 40.0], 180.0),
    ],
)
def test_graph_mode(tmp_path, second_durations, expected_total):
    """
    End-to-end graph-mode test:
      - Build a tiny trace with two cudaGraphLaunch replays and correlated kernels
      - Run generate_perf_report_pytorch with CSV output
      - Verify grouping and per-kernel-position aggregation via ops_unique_args
    """
    # Write synthetic trace to disk
    profile_path = tmp_path / "profile.json"
    with open(profile_path, "w") as f:
        json.dump(_mk_graph_trace(second_durations), f)

    # Run reporting (simple CSV output)
    outdir = tmp_path / "csvs"
    outdir.mkdir()
    result = generate_perf_report_pytorch(
        profile_json_path=str(profile_path),
        output_xlsx_path=None,
        output_csvs_dir=str(outdir),
        collective_analysis=False,
        short_kernel_study=False,
        short_kernel_threshold_us=100,  # ensure df_short_kernels grouping doesn't KeyError on empty
    )

    # Validate graph replays via ops_unique_args
    df_ops_unique = result.get("ops_unique_args")
    assert isinstance(df_ops_unique, pd.DataFrame)

    # Find the row for the synthetic cpu_op parent that encloses the graph launches
    df_gl = df_ops_unique[df_ops_unique["name"] == "aten::graph_wrapper_fwd"]
    assert (
        not df_gl.empty
    ), "Expected a row for the cpu_op parent of graph launches in ops_unique_args"

    # With two identical replays, per-index 'count' in kernel_details_summary should be 2
    inferred_counts = []
    for rec in df_gl["kernel_details_summary"]:
        if isinstance(rec, list) and rec:
            inferred_counts.extend(
                [int(d.get("count", 0)) for d in rec if isinstance(d, dict)]
            )
    assert any(
        c == 2 for c in inferred_counts
    ), f"Expected at least one per-index count == 2, got {inferred_counts}"

    # Kernel position stats are in kernel_details_summary; validate 3 positions and total duration
    assert (
        "kernel_details_summary" in df_gl.columns
    ), "ops_unique_args must include kernel_details_summary"
    totals = []
    lengths = []
    for rec in df_gl["kernel_details_summary"]:
        if isinstance(rec, list) and rec:
            lengths.append(len(rec))
            totals.append(sum(float(d.get("total_duration_us", 0.0)) for d in rec))
    assert any(
        L == 3 for L in lengths
    ), f"Expected 3 kernel positions per replay, got lengths={lengths}"
    expected = expected_total
    assert any(
        abs(t - expected) < 1e-6 for t in totals
    ), f"Expected total ~{expected}us across both replays, got {totals}"
    # Validate per-index mean durations match the average of replay1 (20,30,40) and replay2
    means_ok = False
    for rec in df_gl["kernel_details_summary"]:
        if isinstance(rec, list) and rec and len(rec) == 3:
            idx_means = [
                float(rec[i].get("mean_duration_us", float("nan"))) for i in range(3)
            ]
            expected_means = [
                (20.0 + second_durations[0]) / 2.0,
                (30.0 + second_durations[1]) / 2.0,
                (40.0 + second_durations[2]) / 2.0,
            ]
            means_ok = all(abs(m - e) < 1e-6 for m, e in zip(idx_means, expected_means))
            break
    assert means_ok, "Per-index mean durations did not match expected averages"

    # Percentages should be numeric in ops_unique_args
    assert "Percentage (%)" in df_gl.columns
    assert df_gl["Percentage (%)"].notna().all()

    # Also validate aggregated total time for the wrapper in ops_summary
    df_summary = result.get("ops_summary")
    assert isinstance(df_summary, pd.DataFrame)
    df_row = df_summary[df_summary["name"] == "aten::graph_wrapper_fwd"]
    assert not df_row.empty
    total_us = float(df_row["total_direct_kernel_time_sum"].iloc[0])
    assert abs(total_us - expected_total) < 1e-6
