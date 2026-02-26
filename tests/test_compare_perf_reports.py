###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for the compare_perf_reports_pytorch tool.

For each test case, compares two performance reports and validates the generated
comparison against a reference output. Sheets to compare are driven by whatever
sheets exist in the reference xlsx.
"""

import os
import shutil

import pandas as pd
import pytest

from TraceLens.Reporting.compare_perf_reports_pytorch import (
    generate_compare_perf_reports_pytorch,
)

from conftest import compare_cols, format_diff_details


def find_compare_test_cases(root="tests/traces"):
    """
    Find compare test case directories. Each must contain:
      - Two input reports
      - A reference/ directory with the expected comparison xlsx
    Returns list of (test_dir, report1, report2, ref_xlsx, names, input_sheets) tuples.
    input_sheets are passed to generate_compare_perf_reports_pytorch; the output
    sheet names (read from the reference xlsx) are used for comparison.
    """
    test_cases = []

    # compare_test: 56ch_rank7 with kernel_summary
    test_dir = os.path.join(root, "compare_test")
    r1 = os.path.join(test_dir, "256thread", "perf_56ch_rank7.xlsx")
    r2 = os.path.join(test_dir, "512thread", "perf_56ch_rank7.xlsx")
    ref = os.path.join(test_dir, "reference", "compare_56ch_rank7_across_threads.xlsx")
    if all(os.path.exists(p) for p in [r1, r2, ref]):
        test_cases.append(
            (
                test_dir,
                r1,
                r2,
                ref,
                ["256thread", "512thread"],
                ["gpu_timeline", "kernel_summary"],
            )
        )

    # compare_test_ops: 28ch_rank0 with ops_summary
    test_dir = os.path.join(root, "compare_test_ops")
    r1 = os.path.join(test_dir, "256thread", "perf_28ch_rank0.xlsx")
    r2 = os.path.join(test_dir, "512thread", "perf_28ch_rank0.xlsx")
    ref = os.path.join(test_dir, "reference", "compare_28ch_rank0_across_threads.xlsx")
    if all(os.path.exists(p) for p in [r1, r2, ref]):
        test_cases.append(
            (
                test_dir,
                r1,
                r2,
                ref,
                ["256thread", "512thread"],
                ["gpu_timeline", "ops_summary"],
            )
        )

    return test_cases


@pytest.mark.parametrize(
    "test_dir,report1,report2,ref_xlsx,names,input_sheets",
    find_compare_test_cases(),
)
def test_compare_perf_reports(
    test_dir, report1, report2, ref_xlsx, names, input_sheets, tol=1e-6
):
    """
    Generate a comparison report and validate against reference.
    input_sheets are passed to the compare tool; output sheets from the
    reference xlsx are used for column-by-column comparison.
    """
    ref_output_sheets = pd.ExcelFile(ref_xlsx).sheet_names

    fn_root = os.path.join(test_dir, "pytest_output")
    os.makedirs(fn_root, exist_ok=True)

    try:
        fn_comparison_path = os.path.join(fn_root, "comparison_output.xlsx")
        generate_compare_perf_reports_pytorch(
            reports=[report1, report2],
            output=fn_comparison_path,
            names=names,
            sheets=input_sheets,
        )

        assert os.path.exists(
            fn_comparison_path
        ), f"Comparison output not created: {fn_comparison_path}"

        for sheet in ref_output_sheets:
            try:
                df_ref = pd.read_excel(ref_xlsx, sheet_name=sheet)
                df_fn = pd.read_excel(fn_comparison_path, sheet_name=sheet)
            except ValueError as e:
                pytest.fail(f"Sheet '{sheet}' not found: {e}")

            if df_ref.empty:
                assert (
                    df_fn.empty
                ), f"Reference sheet '{sheet}' is empty but generated has {len(df_fn)} rows"
                continue

            assert set(df_ref.columns) == set(
                df_fn.columns
            ), f"Sheet '{sheet}' has different columns"

            cols = df_ref.columns.tolist()
            diff_cols = compare_cols(df_fn, df_ref, cols, tol=tol)
            assert (
                not diff_cols
            ), f"Sheet '{sheet}' has differences: {format_diff_details(diff_cols)}"

    finally:
        if os.path.exists(fn_root):
            shutil.rmtree(fn_root)
