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
