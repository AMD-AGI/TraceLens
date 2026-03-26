###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Test that the Roofline Bound column appears in the unified_perf_summary
and per-category sheets when a GPU arch config is provided.
"""

import json
import os
import tempfile

import pandas as pd
import pytest

from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)

from conftest import list_perf_report_csv_sheets

MI300X_ARCH = {
    "name": "MI300X",
    "mem_bw_gbps": 5300,
    "max_achievable_tflops": {
        "matrix_fp16": 654,
        "matrix_bf16": 708,
        "matrix_fp32": 163,
        "matrix_fp64": 81,
        "matrix_fp8": 1273,
        "matrix_int8": 2600,
        "vector_fp16": 163,
        "vector_bf16": 163,
        "vector_fp32": 81,
        "vector_fp64": 40,
    },
    "_reference": "https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html#amd-maf-results",
}

TRACE_PATH = os.path.join(
    "tests",
    "traces",
    "mi300",
    "Falconsai_nsfw_image_detection__1016002.json.gz",
)

VALID_BOUND_VALUES = {"COMPUTE_BOUND", "MEMORY_BOUND"}


@pytest.fixture(scope="module")
def perf_report(tmp_path_factory):
    """Generate a perf report with GPU arch config once for the module (CSV sheets)."""
    output_dir = tmp_path_factory.mktemp("roofline_bound")
    arch_path = str(output_dir / "mi300x.json")
    csv_dir = str(output_dir / "perf_report_csvs")

    with open(arch_path, "w") as f:
        json.dump(MI300X_ARCH, f)

    generate_perf_report_pytorch(
        profile_json_path=TRACE_PATH,
        output_xlsx_path=None,
        output_csvs_dir=csv_dir,
        gpu_arch_json_path=arch_path,
    )

    return csv_dir


def _find_roofline_bound_col(df):
    """Find the Roofline Bound column, which may have an aggregation suffix."""
    for col in df.columns:
        if col == "Roofline Bound" or col.startswith("Roofline Bound"):
            return col
    return None


def test_roofline_bound_in_unified_perf_summary(perf_report):
    """Roofline Bound column must appear in unified_perf_summary."""
    df = pd.read_csv(os.path.join(perf_report, "unified_perf_summary.csv"))
    bound_col = _find_roofline_bound_col(df)
    assert bound_col is not None, (
        f"'Roofline Bound' missing from unified_perf_summary. "
        f"Columns: {list(df.columns)}"
    )
    bound_vals = set(df[bound_col].dropna().unique())
    assert bound_vals <= VALID_BOUND_VALUES, f"Unexpected values: {bound_vals}"


def test_roofline_bound_in_category_sheets(perf_report):
    """Roofline Bound column must appear in per-category sheets that have roofline data."""
    sheets_with_roofline = []
    for sheet in list_perf_report_csv_sheets(perf_report):
        df = pd.read_csv(os.path.join(perf_report, f"{sheet}.csv"))
        roofline_time_cols = [c for c in df.columns if c.startswith("Roofline Time")]
        if roofline_time_cols:
            sheets_with_roofline.append(sheet)
            bound_col = _find_roofline_bound_col(df)
            assert bound_col is not None, (
                f"Sheet '{sheet}' has roofline time but missing "
                f"'Roofline Bound'. Columns: {list(df.columns)}"
            )
            bound_vals = set(df[bound_col].dropna().unique())
            assert (
                bound_vals <= VALID_BOUND_VALUES
            ), f"Sheet '{sheet}': unexpected Roofline Bound values: {bound_vals}"

    assert (
        sheets_with_roofline
    ), "No sheets contain Roofline Time — arch config may not have been applied"
