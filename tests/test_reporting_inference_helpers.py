###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for helper functions in generate_perf_report_pytorch_inference."""

import os, pandas as pd, pytest
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    add_truncated_kernel_details,
    get_dfs_short_kernels,
    trunc_kernel_details,
)
from TraceLens import TreePerfAnalyzer

torch = pytest.importorskip("torch")


def test_trunc_kernel_details():
    row = {
        "kernel_details": [
            {"name": "a" * 100, "count": 1},
            {"name": "short", "count": 2},
        ]
    }
    out = trunc_kernel_details(row, "kernel_details", trunc_length=20)
    assert len(out[0]["name"]) == 20
    assert out[1]["name"] == "short"


def test_add_truncated_kernel_details():
    df = pd.DataFrame(
        {
            "kernel_details": [
                [{"name": "x" * 80, "count": 1}],
            ]
        }
    )
    out = add_truncated_kernel_details(df, trunc_length=10)
    assert "trunc_kernel_details" in out.columns
    truncated = out.iloc[0]["trunc_kernel_details"]
    assert len(truncated[0]["name"]) == 10


@pytest.mark.gpu
def test_get_dfs_short_kernels():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA/HIP with at least one visible GPU")

    trace = os.path.join(
        "tests",
        "traces",
        "mi300",
        "Falconsai_nsfw_image_detection__1016002.json.gz",
    )
    if not os.path.isfile(trace):
        pytest.skip(f"Trace not found: {trace}")

    analyzer = TreePerfAnalyzer.from_file(trace)
    df_hist, df_top = get_dfs_short_kernels(analyzer, short_kernel_threshold_us=1000)
    assert isinstance(df_hist, pd.DataFrame)
    assert isinstance(df_top, pd.DataFrame)
