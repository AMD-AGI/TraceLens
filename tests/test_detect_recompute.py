###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
End-to-end test for --detect_recompute feature.

Uses a ResNet activation-checkpoint trace to verify that:
1. Generated report with --detect_recompute matches a checked-in reference xlsx
2. is_recompute column appears in the expected sheets
3. Feature has zero impact when disabled (default)
"""

import os

import pandas as pd
import pytest

from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)

from conftest import compare_cols, format_diff_details

TRACE_DIR = os.path.join(os.path.dirname(__file__), "traces", "mi300")
TRACE_PATH = os.path.join(TRACE_DIR, "resnet_act_checkpoint.json.gz")
REF_XLSX = os.path.join(TRACE_DIR, "resnet_act_checkpoint_recompute_perf_report.xlsx")

SHEETS_WITH_RECOMPUTE = ["ops_summary", "ops_unique_args", "unified_perf_summary"]
PERF_METRICS_SHEETS = [
    "GEMM",
    "SDPA_fwd",
    "SDPA_bwd",
    "CONV_fwd",
    "CONV_bwd",
    "UnaryElementwise",
    "BinaryElementwise",
    "Normalization",
]

COLS_IGNORE = [
    "kernel_details",
    "kernel_details_summary",
    "trunc_kernel_details",
    "kernel_details__summarize_kernel_stats",
    "perf_params",
]


def test_detect_recompute_e2e(tmp_path):
    """Generate perf report with detect_recompute=True and compare to reference xlsx."""
    if not os.path.exists(TRACE_PATH):
        pytest.skip(f"Test trace not found: {TRACE_PATH}")
    if not os.path.exists(REF_XLSX):
        pytest.skip(f"Reference xlsx not found: {REF_XLSX}")

    fn_report_path = str(tmp_path / "test_recompute_perf_report.xlsx")

    generate_perf_report_pytorch(
        profile_json_path=TRACE_PATH,
        output_xlsx_path=fn_report_path,
        detect_recompute=True,
    )

    assert os.path.exists(fn_report_path), "Generated report not created"

    sheets = pd.ExcelFile(REF_XLSX).sheet_names
    for sheet in sheets:
        df_ref = pd.read_excel(REF_XLSX, sheet_name=sheet)
        df_test = pd.read_excel(fn_report_path, sheet_name=sheet)
        if df_ref.empty:
            continue

        cols = [
            col
            for col in df_ref.columns
            if col not in COLS_IGNORE and col in df_test.columns
        ]
        diff_cols = compare_cols(df_test, df_ref, cols, tol=1e-6)
        assert (
            not diff_cols
        ), f"Sheet '{sheet}' has differences: {format_diff_details(diff_cols)}"

    # Verify is_recompute column exists in the expected sheets
    for sheet_name in SHEETS_WITH_RECOMPUTE:
        df = pd.read_excel(fn_report_path, sheet_name=sheet_name)
        assert (
            "is_recompute" in df.columns
        ), f"is_recompute column missing from {sheet_name}"
        assert df["is_recompute"].any(), f"No recompute rows found in {sheet_name}"
        assert not df[
            "is_recompute"
        ].all(), f"All rows marked as recompute in {sheet_name} — expected a mix"

    # Verify is_recompute propagates to perf metrics sheets (build_df_perf_metrics)
    all_sheets = pd.ExcelFile(fn_report_path).sheet_names
    for sheet_name in PERF_METRICS_SHEETS:
        if sheet_name in all_sheets:
            df = pd.read_excel(fn_report_path, sheet_name=sheet_name)
            assert (
                "is_recompute" in df.columns
            ), f"is_recompute column missing from perf metrics sheet {sheet_name}"


def test_detect_recompute_disabled_no_impact(tmp_path):
    """When detect_recompute=False (default), is_recompute should not appear."""
    if not os.path.exists(TRACE_PATH):
        pytest.skip(f"Test trace not found: {TRACE_PATH}")

    dict_name2df = generate_perf_report_pytorch(
        profile_json_path=TRACE_PATH,
        output_xlsx_path=str(tmp_path / "no_recompute.xlsx"),
        detect_recompute=False,
    )

    for sheet_name in SHEETS_WITH_RECOMPUTE + PERF_METRICS_SHEETS:
        if sheet_name in dict_name2df:
            df = dict_name2df[sheet_name]
            assert (
                "is_recompute" not in df.columns
            ), f"is_recompute should NOT appear in {sheet_name} when disabled"
