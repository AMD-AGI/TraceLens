###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The shared unique-args columns are what the three summaries group by."""

import pandas as pd

from TraceLens.TreePerf.tree_perf import (
    PERF_PARAM_PREFIX,
    UNIQUE_ARGS_GROUPING_COLS,
    TreePerfAnalyzer,
)


def _kernel(stack):
    details = {"name": "index_elementwise_kernel", "dur": 1.0}
    if stack is not None:
        details["call_stack"] = stack
    return [details]


def _rows():
    return pd.DataFrame(
        {
            "name": ["aten::mm", "aten::mm"],
            "op category": ["GEMM", "GEMM"],
            "UID": [1, 2],
            "Input Dims": ["(4, 4)", "(4, 4)"],
            "duration_us": [1.0, 3.0],
            "Kernel Time (µs)": [1.0, 3.0],
        }
    )


def test_shared_grouping_columns_collapse_matching_rows():
    assert UNIQUE_ARGS_GROUPING_COLS[0] == "name"
    assert PERF_PARAM_PREFIX == "param: "

    unique = TreePerfAnalyzer.get_df_kernel_launchers_unique_args(_rows())
    assert list(unique["name"]) == ["aten::mm"]
    assert unique["operation_count"].iloc[0] == 2

    summary = TreePerfAnalyzer.summarize_df_unified_perf_table(
        _rows(), agg_metrics=["mean"], include_pct=False
    )
    assert list(summary["name"]) == ["aten::mm"]
    assert summary["total_duration_us"].iloc[0] == 4.0

    perf = pd.DataFrame(
        {
            "name": ["aten::mm"],
            "GFLOPS": [1.0],
            "Data Moved (MB)": [1.0],
            "FLOPS/Byte": [1.0],
            "TB/s": [1.0],
            "TFLOPS/s": [1.0],
            "Kernel Time (µs)": [2.0],
            "UID": [1],
            f"{PERF_PARAM_PREFIX}M": [32],
        }
    )
    out = TreePerfAnalyzer.summarize_df_perf_metrics(perf, agg_metrics=["mean"])
    assert out[f"{PERF_PARAM_PREFIX}M"].iloc[0] == "32"


def test_summary_keeps_launches_from_different_runners_apart():
    # A decode replay of a captured op shares the prefill launch's arguments.
    # The row used to keep the first stack, so the decode launches were
    # labeled prefill_cuda_graph_runner.
    df = _rows()
    df["kernel_details"] = [
        _kernel(["prefill_cuda_graph_runner", "aten::index"]),
        _kernel(["decode_cuda_graph_runner", "aten::index"]),
    ]
    summary = TreePerfAnalyzer.summarize_df_unified_perf_table(
        df, agg_metrics=["mean"], include_pct=False, include_call_stack=True
    )
    assert len(summary) == 2
    assert summary["operation_count"].sum() == 2
    assert summary["total_duration_us"].sum() == 4.0
    stacks = " ".join(summary["call_stack_full"].astype(str))
    assert "prefill_cuda_graph_runner" in stacks
    assert "decode_cuda_graph_runner" in stacks


def test_summary_merges_launches_that_share_a_call_stack():
    df = _rows()
    stack = ["decode_cuda_graph_runner", "aten::index"]
    df["kernel_details"] = [_kernel(stack), _kernel(stack)]
    summary = TreePerfAnalyzer.summarize_df_unified_perf_table(
        df, agg_metrics=["mean"], include_pct=False, include_call_stack=True
    )
    assert len(summary) == 1
    assert summary["operation_count"].iloc[0] == 2
    assert "decode_cuda_graph_runner" in summary["call_stack_full"].iloc[0]


def test_summary_merges_launches_with_no_call_stack():
    df = _rows()
    df["kernel_details"] = [_kernel(None), _kernel(None)]
    summary = TreePerfAnalyzer.summarize_df_unified_perf_table(
        df, agg_metrics=["mean"], include_pct=False
    )
    assert len(summary) == 1
    assert summary["operation_count"].iloc[0] == 2


def test_unique_args_keeps_kernel_call_stacks_in_the_group_key():
    df = _rows()
    df["kernel_details"] = [
        _kernel(["prefill_cuda_graph_runner"]),
        _kernel(["decode_cuda_graph_runner"]),
    ]
    df["total_direct_kernel_time"] = [1.0, 2.0]
    unique = TreePerfAnalyzer.get_df_kernel_launchers_unique_args(
        df, agg_metrics=["mean"]
    )
    assert len(unique) == 2
    assert set(unique["operation_count"]) == {1}
