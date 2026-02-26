###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Regression tests for perf report generation.

For each .json.gz / .xlsx pair in tests/traces, generates a fresh perf report
and compares every sheet against the checked-in reference xlsx.
"""

import os

import pandas as pd
import pytest

from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)

from conftest import compare_cols, format_diff_details


def find_test_cases(ref_root):
    """
    Recursively find all .json.gz/.xlsx pairs in ref_root. The .xlsx file must
    have the same basename as the .json.gz, but with _perf_report.xlsx.
    Returns a list of tuples: (dirpath, gz_filename, xlsx_filename)
    """
    test_cases = []
    for dirpath, _, filenames in os.walk(ref_root):
        gz_files = [f for f in filenames if f.endswith(".json.gz")]
        for gz in gz_files:
            base = gz[:-8]  # remove .json.gz
            xlsx = base + "_perf_report.xlsx"
            xlsx_path = os.path.join(dirpath, xlsx)
            if os.path.exists(xlsx_path):
                test_cases.append((dirpath, gz, xlsx))
    return test_cases


COLS_IGNORE = [
    "Non-Data-Mov TFLOPS/s_mean",
    "Non-Data-Mov Kernel Time (µs)_sum",
    "Non-Data-Mov Kernel Time (µs)_mean",
]


@pytest.mark.parametrize("dirpath,gz,report_filename", find_test_cases("tests/traces"))
def test_perf_report_regression(dirpath, gz, report_filename, tmp_path, tol=1e-6):
    """
    For each .gz/.xlsx pair, generate a report and compare to the reference .xlsx.
    """
    profile_path = os.path.join(dirpath, gz)
    ref_report_path = os.path.join(dirpath, report_filename)
    fn_report_path = str(tmp_path / report_filename)

    generate_perf_report_pytorch(
        profile_json_path=profile_path,
        output_xlsx_path=fn_report_path,
        kernel_summary=True,
        short_kernel_study=True,
    )

    sheets = pd.ExcelFile(ref_report_path).sheet_names
    for sheet in sheets:
        df_ref = pd.read_excel(ref_report_path, sheet_name=sheet)
        df_fn = pd.read_excel(fn_report_path, sheet_name=sheet)
        if df_ref.empty:
            continue
        cols = [
            col
            for col in df_ref.columns
            if col not in COLS_IGNORE and col in df_fn.columns
        ]
        diff_cols = compare_cols(df_fn, df_ref, cols, tol=tol)
        assert (
            not diff_cols
        ), f"Sheet '{sheet}' has differences in {profile_path}:{format_diff_details(diff_cols)}"
