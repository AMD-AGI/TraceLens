###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for semantic_analyses/generate_semantic_diff.py builders."""

import os, sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEM = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis", "semantic_analyses")
sys.path.insert(0, SEM)
import generate_semantic_diff


def _labeled():
    labeled_a = [
        {
            "name": "kA1",
            "semantic_block": "QKV",
            "nn_module": "attn",
            "cpu_op": "aten::mm",
            "dur": 10.0,
            "input_dims": [[1, 2], (3, 4), 5],
            "gpu_op_uid": 100,
        },
        {
            "name": "kA2",
            "semantic_block": "MLP",
            "perf_category": "Compute",
            "dur": 5.0,
            "input_dims": [],
            "gpu_op_uid": 101,
        },
        {
            "name": "kA3",
            "semantic_block": "OnlyA",
            "dur": 2.0,
            "gpu_op_uid": 102,
        },
    ]
    labeled_b = [
        {
            "name": "kB1",
            "semantic_block": "QKV",
            "nn_module": "attn",
            "cpu_op": "aten::mm",
            "dur": 12.0,
            "input_dims": [[1, 2]],
            "gpu_op_uid": 200,
        },
        {
            "name": "kB2",
            "semantic_block": "MLP",
            "perf_category": "Compute",
            "dur": 6.0,
            "gpu_op_uid": 201,
        },
        {
            "name": "kB3",
            "semantic_block": "OnlyB",
            "dur": 3.0,
            "gpu_op_uid": 202,
        },
    ]
    return labeled_a, labeled_b


def _diff_df():
    labeled_a, labeled_b = _labeled()
    rows, _ = generate_semantic_diff.build_diff_stats(labeled_a, labeled_b)
    df = pd.DataFrame(rows)
    df["busy_time"] = (
        df.groupby(["source", "lowest_common_ancestor_id"])["kernel_time"]
        .transform("sum")
        .round(3)
    )
    return df


def test_build_diff_stats():
    labeled_a, labeled_b = _labeled()
    rows, block_id_map = generate_semantic_diff.build_diff_stats(labeled_a, labeled_b)

    assert len(rows) == 6
    assert block_id_map == {"QKV": 0, "MLP": 1, "OnlyA": 2, "OnlyB": 3}

    by_name = {r["name"]: r for r in rows}
    # nn_module present.
    assert by_name["kA1"]["nn_module_stack"] == "attn"
    # falls back to perf_category.
    assert by_name["kA2"]["nn_module_stack"] == "Compute"
    # falls back to "Others".
    assert by_name["kA3"]["nn_module_stack"] == "Others"

    # cpu_op present vs. falling back to the block name.
    assert by_name["kA1"]["cpu_op_name"] == "aten::mm"
    assert by_name["kA3"]["cpu_op_name"] == "OnlyA"

    # _format_dims: list, tuple, and scalar entries.
    assert by_name["kA1"]["Input Dims"] == "(1, 2), (3, 4), 5"
    # empty dims -> empty string.
    assert by_name["kA2"]["Input Dims"] == ""

    assert by_name["kA1"]["source"] == "trace1"
    assert by_name["kB1"]["source"] == "trace2"
    assert by_name["kA1"]["kernel_time"] == 10.0


def test_build_unique_args_summary():
    df = _diff_df()
    summary = generate_semantic_diff.build_unique_args_summary(df)

    assert "kernel_time_sum" in summary.columns
    assert "operation_count" in summary.columns
    # 6 unique kernels -> 6 aggregated rows.
    assert len(summary) == 6
    # sorted descending by kernel_time_sum.
    vals = list(summary["kernel_time_sum"])
    assert vals == sorted(vals, reverse=True)


def test_build_unique_args_summary_unhashable_fallback():
    # A list-valued grouping column forces the TypeError -> str-repr branch.
    df = pd.DataFrame(
        [
            {
                "name": "x",
                "cpu_op_name": [1, 2],
                "source": "trace1",
                "kernel_time": 1.0,
            },
            {
                "name": "y",
                "cpu_op_name": [3, 4],
                "source": "trace1",
                "kernel_time": 2.0,
            },
            {
                "name": "x",
                "cpu_op_name": [1, 2],
                "source": "trace1",
                "kernel_time": 4.0,
            },
        ]
    )
    summary = generate_semantic_diff.build_unique_args_summary(df)
    assert isinstance(summary, pd.DataFrame)
    assert "kernel_time_sum" in summary.columns
    assert "operation_count" in summary.columns


def test_build_cpu_op_maps():
    df = _diff_df()
    cpu_op_map, t1, t2 = generate_semantic_diff.build_cpu_op_maps(df)

    assert "aten::mm" in cpu_op_map
    assert set(cpu_op_map["aten::mm"].keys()) == {"trace1", "trace2"}
    assert cpu_op_map["aten::mm"]["trace1"]["kernels"] == ["kA1"]
    assert cpu_op_map["aten::mm"]["trace2"]["kernels"] == ["kB1"]

    # Per-source grouped frames.
    t1_map = t1.to_dict()["name"]
    t2_map = t2.to_dict()["name"]
    assert t1_map["aten::mm"] == ["kA1"]
    assert t2_map["OnlyB"] == ["kB3"]
    assert "OnlyA" not in t2_map


def test_build_merged_tree_text():
    labeled_a, labeled_b = _labeled()
    _, block_id_map = generate_semantic_diff.build_diff_stats(labeled_a, labeled_b)
    text = generate_semantic_diff.build_merged_tree_text(
        block_id_map, labeled_a, labeled_b, "MI355", "B200"
    )

    assert text.startswith("└── Root (MI355 vs B200)")
    # Combined block (present in both) shows the plain label.
    assert "QKV" in text
    # trace1-only and trace2-only markers.
    assert ">> trace1: OnlyA" in text
    assert "<< trace2: OnlyB" in text
    # Kernel of a combined block rendered plainly.
    assert "kA1" in text
    # Kernel of a trace1-only block carries the >> marker.
    assert ">> trace1: kA3" in text
    assert "<< trace2: kB3" in text
    # Grouping by nn_module / perf_category.
    assert "attn" in text
    assert "Compute" in text
    assert "Others" in text
