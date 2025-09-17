import json
import os
from typing import List, Dict

import pandas as pd

from TraceLens.Reporting.generate_perf_report_pytorch import generate_perf_report_pytorch


def _mk_event(cat: str, name: str, ts: float, dur: float, pid: int, tid: int, args: Dict) -> Dict:
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


def _mk_graph_trace(two_replays: bool = True) -> Dict[str, List[Dict]]:
    """
    Build a minimal trace that exercises graph-mode:
      - 1 or 2 cudaGraphLaunch runtime events
      - Each followed by identical kernel sequences sharing the same correlation id
    """
    events: List[Dict] = []

    # Replay 1: correlation 500
    corr1 = 500
    events.append(_mk_event(
        cat="cuda_runtime",
        name="cudaGraphLaunch",
        ts=1_000_000.0,
        dur=10.0,
        pid=470,
        tid=711,
        args={"cbid": 311, "correlation": corr1},
    ))
    # 3 kernels in replay 1 (same stream/grid/block to form a stable signature)
    kts = 1_000_050.0
    events.append(_mk_event(
        cat="kernel",
        name="kernel_A",
        ts=kts,
        dur=20.0,
        pid=0,
        tid=7,
        args={"device": 0, "context": 1, "stream": 7, "grid": [64, 1, 1], "block": [256, 1, 1], "correlation": corr1},
    ))
    events.append(_mk_event(
        cat="kernel",
        name="kernel_B",
        ts=kts + 30.0,
        dur=30.0,
        pid=0,
        tid=7,
        args={"device": 0, "context": 1, "stream": 7, "grid": [64, 1, 1], "block": [256, 1, 1], "correlation": corr1},
    ))
    events.append(_mk_event(
        cat="kernel",
        name="kernel_C",
        ts=kts + 70.0,
        dur=40.0,
        pid=0,
        tid=7,
        args={"device": 0, "context": 1, "stream": 7, "grid": [64, 1, 1], "block": [256, 1, 1], "correlation": corr1},
    ))
    # Add ac2g markers so TraceToTree can attach at least one GPU event to the runtime launch,
    # which is needed for df_gpu_timeline to find GPU events in synthetic traces.
    events.append({
        "ph": "s", "id": corr1, "pid": 0, "tid": 7, "ts": kts,
        "cat": "ac2g", "name": "ac2g"
    })
    events.append({
        "ph": "f", "id": corr1, "pid": 0, "tid": 7, "ts": kts,
        "cat": "ac2g", "name": "ac2g", "bp": "e"
    })

    # Replay 2: correlation 501 (identical structure -> same signature)
    if two_replays:
        corr2 = 501
        events.append(_mk_event(
            cat="cuda_runtime",
            name="cudaGraphLaunch",
            ts=2_000_000.0,
            dur=12.0,
            pid=470,
            tid=711,
            args={"cbid": 311, "correlation": corr2},
        ))
        kts2 = 2_000_060.0
        events.append(_mk_event(
            cat="kernel",
            name="kernel_A",
            ts=kts2,
            dur=22.0,
            pid=0,
            tid=7,
            args={"device": 0, "context": 1, "stream": 7, "grid": [64, 1, 1], "block": [256, 1, 1], "correlation": corr2},
        ))
        events.append(_mk_event(
            cat="kernel",
            name="kernel_B",
            ts=kts2 + 30.0,
            dur=28.0,
            pid=0,
            tid=7,
            args={"device": 0, "context": 1, "stream": 7, "grid": [64, 1, 1], "block": [256, 1, 1], "correlation": corr2},
        ))
        events.append(_mk_event(
            cat="kernel",
            name="kernel_C",
            ts=kts2 + 65.0,
            dur=41.0,
            pid=0,
            tid=7,
            args={"device": 0, "context": 1, "stream": 7, "grid": [64, 1, 1], "block": [256, 1, 1], "correlation": corr2},
        ))
        # ac2g markers for the second replay as well
        events.append({
            "ph": "s", "id": corr2, "pid": 0, "tid": 7, "ts": kts2,
            "cat": "ac2g", "name": "ac2g"
        })
        events.append({
            "ph": "f", "id": corr2, "pid": 0, "tid": 7, "ts": kts2,
            "cat": "ac2g", "name": "ac2g", "bp": "e"
        })

    return {"traceEvents": events}


def test_graph_mode_reporting(tmp_path):
    """
    End-to-end test:
      - Build a tiny trace with cudaGraphLaunch and correlated kernels
      - Run generate_perf_report_pytorch with CSV output (no Excel dependency)
      - Verify graph_ops_summary groups identical replays and aggregates correctly
    """
    # 1) Write synthetic trace to disk
    profile_path = tmp_path / "profile.json"
    with open(profile_path, "w") as f:
        json.dump(_mk_graph_trace(two_replays=True), f)

    # 2) Run reporting (use CSV output to avoid openpyxl)
    outdir = tmp_path / "csvs"
    outdir.mkdir()
    result = generate_perf_report_pytorch(
        profile_json_path=str(profile_path),
        output_xlsx_path=None,
        output_csvs_dir=str(outdir),
        collective_analysis=False,
        short_kernel_study=False,
    )

    # 3) Extract the graph summary DF
    df_graph = result.get("graph_ops_summary")
    assert isinstance(df_graph, pd.DataFrame)

    # There should be at least one level==0 row summarizing the group
    df_level0 = df_graph[df_graph.get("level") == 0]
    assert not df_level0.empty, "Expected at least one group summary row for graph ops"

    # With two identical replays, we expect one group summary row with launch_count == 2
    # and total kernel time equal to the sum across both replays
    # Replay1 kernels: 20 + 30 + 40 = 90
    # Replay2 kernels: 22 + 28 + 41 = 91
    # Total across replays: 181
    # Note: Busy time computation equals sum for non-overlapping kernels in this synthetic trace
    launch_counts = df_level0["launch_count"].tolist()
    assert 2 in launch_counts, f"Expected launch_count to include 2, got {launch_counts}"

    totals = df_level0["group_total_kernel_time_us"].tolist()
    assert any(abs(t - 181.0) < 1e-6 for t in totals), f"Expected total ~181us in group summary, got {totals}"

    # Also ensure level==1 rows exist for per-kernel index stats and contain 3 entries per group
    df_level1 = df_graph[df_graph.get("level") == 1]
    assert not df_level1.empty, "Expected per-kernel index rows (level==1)"
    # Expect indices 0,1,2 to be present
    idxs = sorted(df_level1["kernel_index"].unique().tolist())
    assert idxs == [0, 1, 2], f"Expected kernel indices [0,1,2], got {idxs}"

    # Percentages should be numeric for the group summary row(s)
    assert "Percentage (%)" in df_level0.columns
    assert df_level0["Percentage (%)"].notna().all()
