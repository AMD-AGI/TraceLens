###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import pandas as pd
from TraceLens import TreePerfAnalyzer, TraceDiff

FILE1 = "tests/traces/mi355/eager_trace.json.gz"
FILE2 = "tests/traces/b200/eager_trace.json.gz"


@pytest.fixture(scope="module")
def tracediff_report():
    perf_analyzer1 = TreePerfAnalyzer.from_file(FILE1, add_python_func=True)
    perf_analyzer2 = TreePerfAnalyzer.from_file(FILE2, add_python_func=True)

    td = TraceDiff(perf_analyzer1.tree, perf_analyzer2.tree)
    td.generate_tracediff_report()

    df = td.diff_stats_df
    return df


@pytest.fixture(scope="module")
def df_trace1(tracediff_report):
    return tracediff_report[tracediff_report["source"] == "trace1"]


@pytest.fixture(scope="module")
def df_trace2(tracediff_report):
    return tracediff_report[tracediff_report["source"] == "trace2"]


def test_tracediff_report_lengths(df_trace1, df_trace2):
    """There should be a row for each kernel in the trace."""
    assert len(df_trace1) == 848
    assert len(df_trace2) == 927


def test_gemms_share_lca_and_are_exclusive(tracediff_report, df_trace1, df_trace2):
    """GEMMs from both traces should share an LCA, and no other kernels should have the same LCA."""
    gemm_trace1 = df_trace1[
        df_trace1["cpu_op_name"] == "vllm::rocm_unquantized_gemm"
    ].reset_index(drop=True)
    gemm_trace2 = df_trace2[df_trace2["cpu_op_name"] == "aten::addmm"].reset_index(
        drop=True
    )

    assert (
        gemm_trace1["lowest_common_ancestor_id"]
        - gemm_trace2["lowest_common_ancestor_id"]
    ).sum() == 0

    gemm_lca_ids = set(gemm_trace1["lowest_common_ancestor_id"]).union(
        set(gemm_trace2["lowest_common_ancestor_id"])
    )
    other_lca_ids = tracediff_report[
        ~tracediff_report["cpu_op_name"].isin(
            ["vllm::rocm_unquantized_gemm", "aten::addmm"]
        )
    ]["lowest_common_ancestor_id"].unique()

    assert gemm_lca_ids.isdisjoint(other_lca_ids)


def test_non_gpu_paths_traversed_for_gpu_path_in_other_trace(df_trace2):
    """Non-GPU paths should be traversed if the path is a GPU path in the other trace."""
    copy_trace2 = df_trace2[df_trace2["cpu_op_name"] == "aten::copy_"].reset_index(
        drop=True
    )
    matches = copy_trace2[
        copy_trace2["lowest_common_ancestor_name"].str.contains(
            "build_attention_metadata"
        )
    ]
    assert matches.shape[0] == 6


def test_position_based_matching_all_reduce_clusters(tracediff_report, df_trace1):
    """Position-based matching should produce small all-reduce clusters of 2-5 kernels each."""
    all_reduce_trace1 = df_trace1[
        df_trace1["cpu_op_name"] == "_C_custom_ar::all_reduce"
    ]
    for lca_id in all_reduce_trace1["lowest_common_ancestor_id"]:
        count = tracediff_report[
            tracediff_report["lowest_common_ancestor_id"] == lca_id
        ].shape[0]
        assert 2 <= count <= 5, f"Expected 2-5 rows for LCA ID {lca_id}, found {count}"
