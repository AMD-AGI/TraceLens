###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Shared test utilities for comparing DataFrames against reference reports.

These helpers are used by test_perf_report_regression, test_compare_perf_reports,
and test_detect_recompute.
"""

import ast
import os
import re
import shutil

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_float_dtype


def pytest_addoption(parser):
    parser.addoption(
        "--update-references",
        action="store_true",
        default=False,
        help="Update reference CSV directories with freshly generated outputs "
        "instead of comparing against them.",
    )


@pytest.fixture
def update_references(request):
    """Return True when reference traces should be overwritten with new outputs.

    Enabled by the ``--update-references`` CLI flag.
    """
    return request.config.getoption("--update-references", default=False)


def update_reference_csvs(generated_dir, reference_dir):
    """Replace *reference_dir* contents with the CSVs from *generated_dir*.

    The reference directory is removed and recreated so that stale sheets
    that no longer exist in the generated output are cleaned up.
    """
    if os.path.isdir(reference_dir):
        shutil.rmtree(reference_dir)
    shutil.copytree(generated_dir, reference_dir)


def perf_report_csv_dirname(trace_base: str) -> str:
    """Directory name for CSV sheets next to trace_base.json.gz (e.g. foo_perf_report_csvs)."""
    return trace_base + "_perf_report_csvs"


def list_perf_report_csv_sheets(csv_dir: str):
    """Return sorted sheet names (CSV stems) in a perf-report CSV directory."""
    return sorted(f[:-4] for f in os.listdir(csv_dir) if f.endswith(".csv"))


def read_perf_report_csv(csv_dir: str, sheet: str):
    """Load one sheet from a directory of per-sheet CSV files."""
    path = os.path.join(csv_dir, f"{sheet}.csv")
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_value(val):
    """Convert numpy scalars and their string representations to Python native types."""
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    elif isinstance(val, list):
        return [normalize_value(v) for v in val]
    elif isinstance(val, dict):
        return {k: normalize_value(v) for k, v in val.items()}
    elif isinstance(val, str):
        cleaned_val = re.sub(r"np\.(?:float|int)\d*\((.*?)\)", r"\1", val)
        try:
            parsed = ast.literal_eval(cleaned_val)
            return normalize_value(parsed)
        except (ValueError, SyntaxError):
            return val
    return val


def compare_cols(df_test, df_ref, cols, tol=1e-6):
    """Compare columns in two DataFrames, skipping rows where ref is None/NaN."""
    diff_details = {}
    for col in cols:
        valid_mask = df_ref[col].notna()
        if not valid_mask.any():
            continue
        ref_col = df_ref.loc[valid_mask, col]
        test_col = df_test.loc[df_test.index.intersection(ref_col.index), col]
        test_col, ref_col = test_col.align(ref_col, join="right")

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
                    ],
                }
        else:
            mismatch = test_col != ref_col
            if mismatch.any():
                diff_indices = mismatch[mismatch].index.tolist()
                diff_details[col] = {
                    "num_diffs": len(diff_indices),
                    "sample_diffs": [
                        (idx, test_col[idx], ref_col[idx]) for idx in diff_indices[:5]
                    ],
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
