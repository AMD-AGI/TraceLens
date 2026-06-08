###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for perf metrics and duration on off-tree graph-replay synthetic ops."""

from copy import deepcopy

import pytest

from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

GEMM_KERNEL = "Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT16x16x64"
OTHER_KERNEL = "Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128"

TEST_ARCH = {
    "name": "test-gpu",
    "mem_bw_gbps": 6000,
    "max_achievable_tflops": {
        "matrix_fp32": 800,
        "matrix_fp16": 800,
        "matrix_bf16": 800,
    },
}


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


def _make_graph_launch_trace():
    """Minimal trace with one hipGraphLaunch and one linked kernel."""
    return [
        _mk_event("cpu_op", "hipGraphLaunch", ts=1000, dur=700, pid=100, tid=100),
        _mk_event(
            "cuda_runtime",
            "hipLaunchKernel",
            ts=1010,
            dur=5,
            pid=100,
            tid=100,
            args={"correlation": 301},
        ),
        _mk_event(
            "kernel",
            GEMM_KERNEL,
            ts=1050,
            dur=20,
            pid=0,
            tid=7,
            args={"correlation": 301, "stream": 7},
        ),
        _mk_ac2g(301, pid=0, tid=7, ts=1050, phase="s"),
        _mk_ac2g(301, pid=0, tid=7, ts=1070, phase="f"),
    ]


def _build_synthetic(analyzer, kernel_name, kernel_dur, parent_dur=700, next_uid=9000):
    """Build an off-tree synthetic the same way collect_unified_perf_events does."""
    parent = {
        "name": "hipGraphLaunch",
        "dur": parent_dur,
        "pid": 100,
        "tid": 100,
        "args": {},
        "children": [],
        "gpu_events": [],
    }
    kernel = {
        "UID": 2,
        "name": kernel_name,
        "dur": kernel_dur,
        "ts": 1050,
        "cat": "kernel",
        "args": {"stream": 7},
    }
    synthetic = dict(parent)
    synthetic["args"] = {}
    synthetic["UID"] = next_uid
    synthetic["name"] = f"hipGraphLaunch->{kernel_name} (Synthetic Op)"
    analyzer._finalize_synthetic_event(synthetic, kernel)
    return synthetic


class TestSyntheticOpPerfMetrics:
    def test_finalize_synthetic_event_uses_kernel_duration(self):
        tree = TraceToTree(deepcopy(_make_graph_launch_trace()))
        tree.build_tree(add_python_func=False)
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)
        synthetic = _build_synthetic(analyzer, GEMM_KERNEL, kernel_dur=20, parent_dur=700)
        assert synthetic["dur"] == 20
        assert synthetic["children"] == []
        assert synthetic["gpu_events"] == [2]

    def test_different_synthetics_get_different_durations(self):
        tree = TraceToTree(deepcopy(_make_graph_launch_trace()))
        tree.build_tree(add_python_func=False)
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)
        syn_a = _build_synthetic(
            analyzer, GEMM_KERNEL, kernel_dur=20, parent_dur=700, next_uid=9001
        )
        syn_b = _build_synthetic(
            analyzer, OTHER_KERNEL, kernel_dur=50, parent_dur=700, next_uid=9002
        )
        assert syn_a["dur"] != syn_b["dur"]
        assert {syn_a["dur"], syn_b["dur"]} == {20, 50}

    def test_compute_perf_metrics_for_off_tree_synthetic(self):
        tree = TraceToTree(deepcopy(_make_graph_launch_trace()))
        tree.build_tree(add_python_func=False)
        analyzer = TreePerfAnalyzer(tree, add_python_func=False, arch=TEST_ARCH)
        synthetic = _build_synthetic(analyzer, GEMM_KERNEL, kernel_dur=20)
        synthetic["args"] = {
            "Input Dims": [[4, 3072], [3072, 256]],
            "Input type": ["float", "float"],
            "Input Strides": [[3072, 1], [1, 3072]],
        }
        metrics = analyzer.compute_perf_metrics(synthetic)
        assert metrics["GFLOPS"] == pytest.approx(2 * 4 * 3072 * 256 / 1e9)
        assert metrics["Kernel Time (µs)"] == 20
        assert metrics["Compute Spec"] == "matrix_fp32"
        assert metrics["Roofline Bound"] in {"COMPUTE_BOUND", "MEMORY_BOUND"}
        assert metrics["Pct Roofline"] > 0

    def test_unified_perf_table_has_roofline_and_per_kernel_duration(self):
        tree = TraceToTree(deepcopy(_make_graph_launch_trace()))
        tree.build_tree(add_python_func=False)
        analyzer = TreePerfAnalyzer(tree, add_python_func=False, arch=TEST_ARCH)
        synthetics = [
            _build_synthetic(
                analyzer, GEMM_KERNEL, kernel_dur=20, next_uid=9001
            ),
            _build_synthetic(
                analyzer, OTHER_KERNEL, kernel_dur=50, next_uid=9002
            ),
        ]
        synthetics[0]["args"] = {
            "Input Dims": [[4, 3072], [3072, 256]],
            "Input type": ["float", "float"],
        }
        synthetics[1]["args"] = {
            "Input Dims": [[4, 3072], [3072, 50016]],
            "Input type": ["c10::Half", "c10::Half"],
        }
        df = analyzer.build_df_unified_perf_table(events=synthetics)
        gemm_row = df[df["name"].str.contains(GEMM_KERNEL, regex=False)].iloc[0]
        other_row = df[df["name"].str.contains(OTHER_KERNEL, regex=False)].iloc[0]
        assert gemm_row["duration_us"] == 20
        assert other_row["duration_us"] == 50
        assert gemm_row["GFLOPS"] > 0
        assert other_row["GFLOPS"] > 0
        assert gemm_row["Pct Roofline"] > 0
        assert other_row["Pct Roofline"] > 0
