###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Regression tests for perf report generation.

For each .json.gz / *_perf_report_csvs/ pair in tests/traces, generates a fresh
perf report as CSVs and compares every sheet against the checked-in reference
directory (one .csv per sheet).
"""

import os

import pytest

from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)

from conftest import (
    compare_cols,
    format_diff_details,
    list_perf_report_csv_sheets,
    perf_report_csv_dirname,
    read_perf_report_csv,
    trace_is_mi300,
    arch_mi300_json_path,
)


def find_test_cases(ref_root):
    """
    Recursively find all .json.gz / *_perf_report_csvs/ pairs in ref_root.
    The directory must be named trace_basename + '_perf_report_csvs' and sit
    alongside the .json.gz.
    Returns a list of tuples: (dirpath, gz_filename, report_csv_dirname)
    """
    test_cases = []
    for dirpath, _, filenames in os.walk(ref_root):
        gz_files = [f for f in filenames if f.endswith(".json.gz")]
        for gz in gz_files:
            base = gz[:-8]  # remove .json.gz
            csv_dir_name = perf_report_csv_dirname(base)
            csv_dir_path = os.path.join(dirpath, csv_dir_name)
            if os.path.isdir(csv_dir_path):
                test_cases.append((dirpath, gz, csv_dir_name))
    return test_cases


COLS_IGNORE = [
    "Non-Data-Mov TFLOPS/s_mean",
    "Non-Data-Mov Kernel Time (µs)_sum",
    "Non-Data-Mov Kernel Time (µs)_mean",
]


@pytest.mark.parametrize(
    "dirpath,gz,report_csv_dirname", find_test_cases("tests/traces")
)
def test_perf_report_regression(dirpath, gz, report_csv_dirname, tmp_path, tol=1e-6):
    """
    For each .gz / *_perf_report_csvs/ pair, generate a report and compare to reference CSVs.
    """
    profile_path = os.path.join(dirpath, gz)
    ref_csv_dir = os.path.join(dirpath, report_csv_dirname)
    fn_csv_dir = str(tmp_path / report_csv_dirname)
    if trace_is_mi300(profile_path):
        gpu_arch_json_path = arch_mi300_json_path()
    else:
        gpu_arch_json_path = None
    generate_perf_report_pytorch(
        profile_json_path=profile_path,
        output_xlsx_path=None,
        output_csvs_dir=fn_csv_dir,
        kernel_summary=True,
        short_kernel_study=True,
        gpu_arch_json_path=gpu_arch_json_path,
        enable_origami=gpu_arch_json_path is not None,
    )

    sheets = list_perf_report_csv_sheets(ref_csv_dir)
    for sheet in sheets:
        df_ref = read_perf_report_csv(ref_csv_dir, sheet)
        df_fn = read_perf_report_csv(fn_csv_dir, sheet)
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
