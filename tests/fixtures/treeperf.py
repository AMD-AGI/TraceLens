###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared test helpers migrated from test_treeperf_coverage.py."""

from __future__ import annotations
import gzip
import json
from copy import deepcopy
import pandas as pd
import pytest
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

def _make_gpu_event(
    uid, ts, dur, cat="kernel", name="kernel", pid=100, tid=100, args=None
):
    event = {
        "ph": "X",
        "UID": uid,
        "ts": ts,
        "dur": dur,
        "cat": cat,
        "name": name or "kernel",
        "pid": pid,
        "tid": tid,
    }
    if args is not None:
        event["args"] = args
    return event
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
def _build_analyzer(events, add_python_func=False, **kwargs):
    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree(add_python_func=add_python_func)
    return TreePerfAnalyzer(
        tree, add_python_func=add_python_func, rebuild_tree=False, **kwargs
    )

def _mk_pytorch_trace():
    corr = 100
    return [
        _make_gpu_event(
            "cpu", 1000, 100, "cpu_op", "aten::mm", pid=100,
            args={"Input Dims": [[32, 64], [64, 128]], "Input type": ["fp16", "fp16"]},
        ),
        _make_gpu_event(
            "rt", 1010, 5, "cuda_runtime", "hipLaunchKernel", pid=100,
            args={"correlation": corr},
        ),
        _make_gpu_event(
            "kern", 1050, 50, "kernel", "gemm_kernel", pid=0, tid=7,
            args={"correlation": corr, "stream": 7},
        ),
        _mk_ac2g(corr, 0, 7, 1050, "s"),
        _mk_ac2g(corr, 0, 7, 1100, "f"),
    ]


def _sweep_treeperf_analyzer(analyzer):
    assert analyzer.tree is not None
    analyzer.check_gpu_only()
    timeline = analyzer.get_df_gpu_timeline(micro_idle_thresh_us=0)
    assert isinstance(timeline, pd.DataFrame)
    launchers = analyzer.get_df_kernel_launchers(
        include_args=True,
        include_kernel_details=True,
        include_call_stack=analyzer.add_python_func,
        id_cols=True,
    )
    assert isinstance(launchers, pd.DataFrame)
    if not launchers.empty:
        TreePerfAnalyzer.get_df_kernel_launchers_summary(launchers)
        TreePerfAnalyzer.get_df_kernel_launchers_summary_by_category(launchers)
        TreePerfAnalyzer.get_df_kernel_launchers_unique_args(launchers, include_pct=True)
    unified = analyzer.build_df_unified_perf_table(include_nccl=True)
    if not unified.empty:
        try:
            TreePerfAnalyzer.summarize_df_unified_perf_table(
                unified, include_pct=True, tree=analyzer.tree,
                agg_metrics=["mean", "median", "max", "min", "std", "sum", "count"],
            )
        except (ValueError, KeyError):
            TreePerfAnalyzer.summarize_df_unified_perf_table(
                unified, include_pct=True, tree=analyzer.tree
            )
