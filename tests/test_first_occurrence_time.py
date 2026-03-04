###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the first_occurrence_time column in ops_unique_args."""

import pandas as pd
import pytest
from copy import deepcopy

from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer


def _make_df_kernel_launchers(rows):
    """Build a minimal df_kernel_launchers from a list of row dicts."""
    base = {
        "op category": "GEMM",
        "process_name": "python",
        "process_label": "0",
        "thread_name": "main",
        "Input Dims": None,
        "Input type": None,
        "Input Strides": None,
        "Concrete Inputs": None,
        "total_direct_kernel_time": 1.0,
        "total_subtree_kernel_time": 1.0,
        "direct_kernel_count": 1,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


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


class TestFirstOccurrenceTime:
    """Tests for the first_occurrence_time column in ops_unique_args."""

    def test_column_exists(self):
        df = _make_df_kernel_launchers(
            [
                {"name": "aten::mm", "UID": 1, "ts": 1000},
            ]
        )
        result = TreePerfAnalyzer.get_df_kernel_launchers_unique_args(df)
        assert "first_occurrence_time" in result.columns

    def test_single_op_normalized_to_zero(self):
        df = _make_df_kernel_launchers(
            [
                {"name": "aten::mm", "UID": 1, "ts": 5000},
            ]
        )
        result = TreePerfAnalyzer.get_df_kernel_launchers_unique_args(df)
        assert result["first_occurrence_time"].iloc[0] == 0

    def test_two_different_ops_normalized(self):
        """The earlier op should be 0, the later should be the delta."""
        df = _make_df_kernel_launchers(
            [
                {"name": "aten::mm", "UID": 1, "ts": 3000},
                {"name": "aten::addmm", "UID": 2, "ts": 5000},
            ]
        )
        result = TreePerfAnalyzer.get_df_kernel_launchers_unique_args(df)
        ts_map = dict(zip(result["name"], result["first_occurrence_time"]))
        assert ts_map["aten::mm"] == 0
        assert ts_map["aten::addmm"] == 2000

    def test_grouped_ops_uses_min_ts(self):
        """When multiple instances of the same op exist, use the earliest timestamp."""
        df = _make_df_kernel_launchers(
            [
                {"name": "aten::mm", "UID": 1, "ts": 8000},
                {"name": "aten::mm", "UID": 2, "ts": 3000},
                {"name": "aten::mm", "UID": 3, "ts": 6000},
            ]
        )
        result = TreePerfAnalyzer.get_df_kernel_launchers_unique_args(df)
        assert len(result) == 1
        assert result["first_occurrence_time"].iloc[0] == 0

    def test_multiple_groups_min_and_normalized(self):
        """Each group picks its min ts, then all are normalized to the global min."""
        df = _make_df_kernel_launchers(
            [
                {"name": "aten::mm", "UID": 1, "ts": 5000},
                {"name": "aten::mm", "UID": 2, "ts": 2000},
                {"name": "aten::addmm", "UID": 3, "ts": 9000},
                {"name": "aten::addmm", "UID": 4, "ts": 7000},
                {"name": "aten::copy_", "UID": 5, "ts": 4000},
            ]
        )
        result = TreePerfAnalyzer.get_df_kernel_launchers_unique_args(df)
        ts_map = dict(zip(result["name"], result["first_occurrence_time"]))
        assert ts_map["aten::mm"] == 0
        assert ts_map["aten::copy_"] == 2000
        assert ts_map["aten::addmm"] == 5000

    def test_same_name_different_args_separate_groups(self):
        """Same op name but different args should be separate rows with independent timestamps."""
        df = _make_df_kernel_launchers(
            [
                {
                    "name": "aten::mm",
                    "UID": 1,
                    "ts": 1000,
                    "Input Dims": ((2, 3), (3, 4)),
                },
                {
                    "name": "aten::mm",
                    "UID": 2,
                    "ts": 5000,
                    "Input Dims": ((8, 8), (8, 8)),
                },
            ]
        )
        result = TreePerfAnalyzer.get_df_kernel_launchers_unique_args(df)
        assert len(result) == 2
        assert set(result["first_occurrence_time"]) == {0, 4000}
