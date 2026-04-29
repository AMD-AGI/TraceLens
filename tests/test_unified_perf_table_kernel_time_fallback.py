###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Regression tests for kernel-time reporting when the perf model raises at runtime.

When the analytic perf model fails (missing ``Input Dims``, ``NotImplementedError``
from ``flops`` / ``flops_bwd``, etc.), reports must still show GPU kernel busy time
from trace-derived aggregation where kernels exist.

Covers ``build_df_unified_perf_table`` and ``build_df_perf_metrics``.
"""

from copy import deepcopy
from unittest.mock import patch

import pandas as pd

from TraceLens.PerfModel import perf_model
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer


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


def _make_aten_mm_trace(include_input_dims):
    """One ``aten::mm`` cpu_op -> runtime -> kernel.

    When ``include_input_dims`` is False the args lack ``Input Dims`` so the
    GEMM perf model raises during construction.
    """
    corr = 100
    args = {}
    if include_input_dims:
        args = {
            "Input Dims": [[64, 32], [32, 16], []],
            "Input type": ["float", "float", ""],
            "Input Strides": [[32, 1], [16, 1], []],
        }
    return [
        _mk_event("cpu_op", "aten::mm", ts=1000, dur=100, pid=100, tid=100, args=args),
        _mk_event(
            "cuda_runtime",
            "hipLaunchKernel",
            ts=1010,
            dur=5,
            pid=100,
            tid=100,
            args={"correlation": corr},
        ),
        _mk_event(
            "kernel",
            "Cijk_Ailk_Bljk_SS",
            ts=1050,
            dur=50,
            pid=0,
            tid=7,
            args={"correlation": corr, "stream": 7},
        ),
        _mk_ac2g(corr, pid=0, tid=7, ts=1050, phase="s"),
        _mk_ac2g(corr, pid=0, tid=7, ts=1050, phase="f"),
    ]


def _build_unified_df(events):
    tree = TraceToTree(deepcopy(events))
    tree.build_tree()
    analyzer = TreePerfAnalyzer(tree, add_python_func=False)
    return analyzer.build_df_unified_perf_table(
        include_args=True,
        include_perf_metrics=True,
        include_kernel_details=True,
    )


def test_kernel_time_populated_when_perf_model_succeeds():
    """Sanity check: with valid Input Dims, perf model succeeds and Kernel Time is set."""
    df = _build_unified_df(_make_aten_mm_trace(include_input_dims=True))
    row = df[df["name"] == "aten::mm"].iloc[0]
    assert row["has_perf_model"]
    assert row["Kernel Time (µs)"] == 50.0
    assert row["GFLOPS"] is not None


def test_kernel_time_populated_when_perf_model_fails():
    """
    When the perf model crashes (missing Input Dims), the row should still
    exist and ``Kernel Time (µs)`` should reflect the GPU busy time computed
    from gpu_events — not NaN/0.
    """
    df = _build_unified_df(_make_aten_mm_trace(include_input_dims=False))
    matches = df[df["name"] == "aten::mm"]
    assert not matches.empty, "aten::mm row missing from unified perf table"
    row = matches.iloc[0]

    assert row["has_perf_model"], (
        "aten::mm should still be flagged as having a perf model even though "
        "its constructor failed at runtime"
    )
    assert row["Kernel Time (µs)"] == 50.0, (
        "Kernel Time must fall back to gpu_events busy time when the perf "
        "model fails to instantiate"
    )
    assert row["kernel_details"], "kernel_details column should still be populated"
    assert all(kd["name"] == "Cijk_Ailk_Bljk_SS" for kd in row["kernel_details"])


def test_build_df_perf_metrics_keeps_row_when_perf_model_fails():
    """``build_df_perf_metrics`` must not drop the row when the perf model crashes."""
    tree = TraceToTree(deepcopy(_make_aten_mm_trace(include_input_dims=False)))
    tree.build_tree()
    analyzer = TreePerfAnalyzer(tree, add_python_func=False)
    cpu_ops = [e for e in tree.events if e.get("cat") == "cpu_op"]
    df = analyzer.build_df_perf_metrics(cpu_ops, include_kernel_details=True)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["Kernel Time (µs)"] > 0
    # Partial perf-metrics dict has no GFLOPS when the perf model never completes.
    if "GFLOPS" in df.columns:
        assert pd.isna(r["GFLOPS"])


def test_kernel_time_populated_when_flops_raises_not_implemented_error():
    """
    ``NotImplementedError`` from e.g. ``flops()`` must still yield kernel time via
    partial ``compute_perf_metrics`` return (same as other perf-model failures).
    """
    events = _make_aten_mm_trace(include_input_dims=True)
    tree = TraceToTree(deepcopy(events))
    tree.build_tree()
    analyzer = TreePerfAnalyzer(tree, add_python_func=False)
    with patch.object(
        perf_model.aten_mm,
        "flops",
        side_effect=NotImplementedError("test: forward flops not implemented"),
    ):
        df = analyzer.build_df_unified_perf_table(
            include_args=True,
            include_perf_metrics=True,
            include_kernel_details=True,
        )
    row = df[df["name"] == "aten::mm"].iloc[0]
    assert row["Kernel Time (µs)"] > 0
    assert row["has_perf_model"]
    if "GFLOPS" in df.columns:
        assert pd.isna(row["GFLOPS"]) or row["GFLOPS"] is None
