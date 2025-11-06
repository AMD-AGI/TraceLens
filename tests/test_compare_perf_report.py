###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import subprocess

import pandas as pd
import pytest
from pandas.api.types import is_float_dtype
import numpy as np
import ast
import re


def compare_cols(df_test, df_ref, cols, tol=1e-6):
    """Compare columns in two dataframes, skipping rows where ref is None/NaN."""
    diff_details = {}
    for col in cols:
        valid_mask = df_ref[col].notna()
        if not valid_mask.any():
            continue
        ref_col = df_ref.loc[valid_mask, col]
        test_col = df_test.loc[df_test.index.intersection(ref_col.index), col]
        test_col, ref_col = test_col.align(ref_col, join="right")

        # Normalize numpy types to Python native types
        test_col = test_col.apply(normalize_value)
        ref_col = ref_col.apply(normalize_value)

        if is_float_dtype(df_test[col]):
            diff = test_col - ref_col
            max_diff = diff.abs().max()
            if not max_diff < tol:
                diff_indices = diff[diff.abs() >= tol].index.tolist()
                diff_details[col] = {
                    "max_diff": max_diff,
                    "num_diffs": len(diff_indices),
                    "sample_diffs": [
                        (idx, test_col[idx], ref_col[idx], diff[idx])
                        for idx in diff_indices[:5]
                    ],  # Show first 5
                }
        else:
            mismatch = test_col != ref_col
            if mismatch.any():
                diff_indices = mismatch[mismatch].index.tolist()
                diff_details[col] = {
                    "num_diffs": len(diff_indices),
                    "sample_diffs": [
                        (idx, test_col[idx], ref_col[idx]) for idx in diff_indices[:5]
                    ],  # Show first 5
                }
    return diff_details


def format_diff_details(diff_details):
    """Format difference details for readable assertion messages."""
    lines = []
    for col, details in diff_details.items():
        lines.append(f"\n  Column: '{col}'")
        lines.append(f"    Total differences: {details['num_diffs']}")

        if "max_diff" in details:
            lines.append(f"    Max difference: {details['max_diff']:.6e}")
            lines.append("    Sample differences:")
            lines.append(
                f"      {'Index':<8} {'Test Value':<20} {'Ref Value':<20} {'Difference':<15}"
            )
            lines.append(f"      {'-'*8} {'-'*20} {'-'*20} {'-'*15}")
            for idx, test_val, ref_val, diff in details["sample_diffs"]:
                lines.append(
                    f"      {idx:<8} {test_val:<20.6e} {ref_val:<20.6e} {diff:<15.6e}"
                )
        else:
            lines.append("    Sample differences:")
            lines.append(f"      {'Index':<8} {'Test Value':<30} {'Ref Value':<30}")
            lines.append(f"      {'-'*8} {'-'*30} {'-'*30}")
            for idx, test_val, ref_val in details["sample_diffs"]:
                lines.append(f"      {idx:<8} {str(test_val):<30} {str(ref_val):<30}")

    return "\n".join(lines)


def normalize_value(val):
    """Convert numpy scalars and their string representations to Python native types."""
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    elif isinstance(val, list):
        return [normalize_value(v) for v in val]
    elif isinstance(val, dict):
        return {k: normalize_value(v) for k, v in val.items()}
    elif isinstance(val, str):
        # Clean string representations of numpy types before parsing
        # e.g., "np.float64(1.23)" -> "1.23"
        cleaned_val = re.sub(r"np\.(?:float|int)\d*\((.*?)\)", r"\1", val)
        try:
            # Use the cleaned string for parsing
            parsed = ast.literal_eval(cleaned_val)
            return normalize_value(parsed)
        except (ValueError, SyntaxError):
            # Return original value if it's not a parsable literal
            return val
    return val


def generate_perf_report(profile_path, report_path):
    script_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "TraceLens",
            "Reporting",
            "generate_perf_report_pytorch.py",
        )
    )
    cmd = [
        "python3",
        script_path,
        "--profile_json_path",
        profile_path,
        "--output_xlsx_path",
        report_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running command: {result.stderr}")
    return True


def find_test_cases(ref_root):
    """
    Recursively find all .json.gz/.xlsx pairs in ref_root. The .xlsx file must have the same basename as the .json.gz, but with _perf_report.xlsx.
    Returns a list of tuples: (profile_path, report_filename)
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


@pytest.mark.parametrize("dirpath,gz,report_filename", find_test_cases("tests/traces"))
def test_compare_perf_report(dirpath, gz, report_filename, tol=1e-6):
    """
    For each .gz/.xlsx pair, decompress the .gz, generate a report, and compare to the reference .xlsx.
    """
    import shutil

    ref_root = dirpath
    profile_path = os.path.join(dirpath, gz)
    ref_report_path = os.path.join(dirpath, report_filename)
    # Generate a temp output directory for this test
    fn_root = os.path.join(dirpath, "pytest_reports")
    os.makedirs(fn_root, exist_ok=True)
    # Decompress .gz to .json (without deleting .gz)
    jsonfile = profile_path[:-3]
    try:
        subprocess.run(["gunzip", "-kf", profile_path], check=True)
        # Generate report
        fn_report_path = os.path.join(fn_root, report_filename)
        generate_perf_report(jsonfile, fn_report_path)
        # Compare sheets
        sheets = pd.ExcelFile(ref_report_path).sheet_names
        cols_ignore = [
            "Non-Data-Mov TFLOPS/s_mean",
            "Non-Data-Mov Kernel Time (µs)_sum",
            "Non-Data-Mov Kernel Time (µs)_mean",
        ]
        for sheet in sheets:
            df_ref = pd.read_excel(ref_report_path, sheet_name=sheet)
            df_fn = pd.read_excel(fn_report_path, sheet_name=sheet)
            if df_ref.empty:
                continue
            cols = [col for col in df_ref.columns if col not in cols_ignore]
            diff_cols = compare_cols(df_fn, df_ref, cols, tol=tol)
            assert (
                not diff_cols
            ), f"Sheet '{sheet}' has differences in {profile_path}:{format_diff_details(diff_cols)}"
    finally:
        # Cleanup: remove generated .json and report
        if os.path.exists(jsonfile):
            os.remove(jsonfile)
        if os.path.exists(fn_root):
            shutil.rmtree(fn_root)
