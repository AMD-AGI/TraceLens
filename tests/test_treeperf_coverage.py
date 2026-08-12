###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Additional CPU-only coverage for TreePerfAnalyzer uncovered paths."""

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


class TestTreePerfCoverageExtended:
    def test_from_file_json_gz(self, tmp_path):
        corr = 500
        events = [
            _make_gpu_event(
                "cpu",
                1000,
                100,
                "cpu_op",
                "aten::mm",
                args={
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["fp16", "fp16"],
                },
            ),
            _make_gpu_event(
                "rt",
                1010,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                args={"correlation": corr},
            ),
            _make_gpu_event(
                "kern",
                1050,
                50,
                "kernel",
                "gemm_kernel",
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, 0, 7, 1050, "s"),
            _mk_ac2g(corr, 0, 7, 1100, "f"),
        ]
        trace_path = tmp_path / "trace.json.gz"
        with gzip.open(trace_path, "wt", encoding="utf-8") as f:
            json.dump({"traceEvents": events}, f)

        analyzer = TreePerfAnalyzer.from_file(str(trace_path))
        assert analyzer.get_kernel_launchers()

    def test_get_df_kernels_with_nn_module_detail(self):
        corr = 600
        events = [
            _make_gpu_event(
                "nn",
                900,
                200,
                "python_function",
                "nn.Module: Linear_0",
                args={"Python id": 1},
            ),
            _make_gpu_event(
                "cpu",
                1000,
                100,
                "cpu_op",
                "aten::linear",
                args={
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["fp16", "fp16"],
                },
            ),
            _make_gpu_event(
                "rt",
                1010,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                args={"correlation": corr},
            ),
            _make_gpu_event(
                "kern",
                1050,
                50,
                "kernel",
                "gemm_kernel",
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, 0, 7, 1050, "s"),
            _mk_ac2g(corr, 0, 7, 1100, "f"),
        ]
        analyzer = _build_analyzer(events, add_python_func=True)
        nn_evt = next(
            e for e in analyzer.tree.events if e["name"].startswith("nn.Module")
        )
        cpu_evt = next(e for e in analyzer.tree.events if e["name"] == "aten::linear")
        kernel_evt = next(e for e in analyzer.tree.events if e["cat"] == "kernel")

        nn_evt.setdefault("children", []).append(cpu_evt["UID"])
        cpu_evt["parent"] = nn_evt["UID"]
        nn_evt["gpu_events"] = [kernel_evt["UID"]]

        df = analyzer.get_df_kernels(
            nn_module_detail=True, cpu_op_detail=True, launcher_detail=True
        )
        assert not df.empty
        assert "Parent nn.Module" in df.columns
        assert df.iloc[0]["Parent nn.Module"].startswith("nn.Module")

    def test_get_df_kernels_rejects_python_stack(self):
        analyzer = _build_analyzer(_mk_pytorch_trace())
        analyzer.with_python_stack = True
        with pytest.raises(ValueError, match="Python stack"):
            analyzer.get_df_kernels()

    def test_build_df_from_events_bwd_linked(self):
        corr_fwd = 700
        corr_bwd = 701
        events = [
            _make_gpu_event(
                "fwd",
                1000,
                100,
                "cpu_op",
                "aten::mm",
                args={
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["fp16", "fp16"],
                },
            ),
            _make_gpu_event(
                "rt1",
                1010,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                args={"correlation": corr_fwd},
            ),
            _make_gpu_event(
                "k1",
                1050,
                50,
                "kernel",
                "gemm_fwd",
                pid=0,
                tid=7,
                args={"correlation": corr_fwd, "stream": 7},
            ),
            _mk_ac2g(corr_fwd, 0, 7, 1050, "s"),
            _mk_ac2g(corr_fwd, 0, 7, 1100, "f"),
            _make_gpu_event(
                "bwd",
                2000,
                100,
                "cpu_op",
                "aten::mm_backward",
                args={
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["fp16", "fp16"],
                },
            ),
            _make_gpu_event(
                "rt2",
                2010,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                args={"correlation": corr_bwd},
            ),
            _make_gpu_event(
                "k2",
                2050,
                60,
                "kernel",
                "gemm_bwd",
                pid=0,
                tid=7,
                args={"correlation": corr_bwd, "stream": 7},
            ),
            _mk_ac2g(corr_bwd, 0, 7, 2050, "s"),
            _mk_ac2g(corr_bwd, 0, 7, 2110, "f"),
        ]
        analyzer = _build_analyzer(events)
        fwd_evt = next(e for e in analyzer.tree.events if e["name"] == "aten::mm")
        bwd_evt = next(
            e for e in analyzer.tree.events if e["name"] == "aten::mm_backward"
        )
        fwd_evt["bwd_events"] = [bwd_evt["UID"]]
        bwd_evt["fwd_event"] = fwd_evt["UID"]

        df = analyzer.build_df_unified_perf_table(
            events=[fwd_evt, bwd_evt], include_perf_metrics=True
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1


def _mk_pytorch_trace():
    corr = 100
    return [
        _make_gpu_event(
            "cpu",
            1000,
            100,
            "cpu_op",
            "aten::mm",
            pid=100,
            args={"Input Dims": [[32, 64], [64, 128]], "Input type": ["fp16", "fp16"]},
        ),
        _make_gpu_event(
            "rt",
            1010,
            5,
            "cuda_runtime",
            "hipLaunchKernel",
            pid=100,
            args={"correlation": corr},
        ),
        _make_gpu_event(
            "kern",
            1050,
            50,
            "kernel",
            "gemm_kernel",
            pid=0,
            tid=7,
            args={"correlation": corr, "stream": 7},
        ),
        _mk_ac2g(corr, 0, 7, 1050, "s"),
        _mk_ac2g(corr, 0, 7, 1100, "f"),
    ]
