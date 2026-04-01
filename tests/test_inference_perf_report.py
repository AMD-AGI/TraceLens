###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Regression tests for generate_perf_report_pytorch_inference.
# Each test case is a subdirectory under tests/traces/inference/ containing:
#   - A .json.gz trace file
#   - A perf_report.xlsx reference report
#   - Optionally capture_traces/ (graph capture mode)
#   - Optionally gpu_arch.json

import os

import numpy as np
import pandas as pd
import pytest
import ast
import re
from pandas.api.types import is_float_dtype

from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    generate_perf_report_pytorch,
    classify_graph_capture_trace,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Input list of events is empty.*:UserWarning",
    "ignore:Input DataFrame is empty.*:UserWarning",
    "ignore:Source column 'kernel_details__summarize_kernel_stats' not found.*:UserWarning",
    "ignore:Found .* events with failed performance metric computation.*:UserWarning",
    "ignore:Inconsistent kernel list length found.*:UserWarning",
    "ignore:There are hipgraph launches.*:UserWarning",
)

INFERENCE_TRACES_ROOT = "tests/traces/inference"


# ---------------------------------------------------------------------------
# Helpers (shared with test_compare_perf_report.py — kept self-contained here
# so this file can run independently)
# ---------------------------------------------------------------------------


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


UID_DIFF_TOLERANCE = 10


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

        test_col = test_col.apply(normalize_value)
        ref_col = ref_col.apply(normalize_value)
        print(f"Column: {col}")
        if "UID" in col:
            # Allow UID values to differ by up to UID_DIFF_TOLERANCE (synthetic
            # ops now receive fresh UIDs that may shift relative to the reference).
            try:
                diff = test_col.astype(float) - ref_col.astype(float)
                mismatch = diff.abs() >= UID_DIFF_TOLERANCE
                if mismatch.any():
                    diff_indices = mismatch[mismatch].index.tolist()
                    diff_details[col] = {
                        "num_diffs": len(diff_indices),
                        "sample_diffs": [
                            (idx, test_col[idx], ref_col[idx])
                            for idx in diff_indices[:5]
                        ],
                    }
            except (TypeError, ValueError):
                # Fall back to exact comparison if values aren't numeric
                mismatch = test_col != ref_col
                if mismatch.any():
                    diff_indices = mismatch[mismatch].index.tolist()
                    diff_details[col] = {
                        "num_diffs": len(diff_indices),
                        "sample_diffs": [
                            (idx, test_col[idx], ref_col[idx])
                            for idx in diff_indices[:5]
                        ],
                    }
        elif is_float_dtype(df_test[col]):
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


# ---------------------------------------------------------------------------
# Test case discovery
# ---------------------------------------------------------------------------

COLS_IGNORE = [
    "Non-Data-Mov TFLOPS/s_mean",
    "Non-Data-Mov Kernel Time (µs)_sum",
    "Non-Data-Mov Kernel Time (µs)_mean",
]


def find_inference_test_cases():
    """
    Discover test cases under INFERENCE_TRACES_ROOT.
    Each subdirectory with a .json.gz and perf_report.xlsx becomes a test case.
    """
    test_cases = []
    if not os.path.isdir(INFERENCE_TRACES_ROOT):
        return test_cases
    for entry in sorted(os.listdir(INFERENCE_TRACES_ROOT)):
        dirpath = os.path.join(INFERENCE_TRACES_ROOT, entry)
        if not os.path.isdir(dirpath):
            continue
        gz_files = [f for f in os.listdir(dirpath) if f.endswith(".json.gz")]
        ref_xlsx = os.path.join(dirpath, "perf_report.xlsx")
        if not gz_files or not os.path.exists(ref_xlsx):
            continue
        trace_gz = gz_files[0]
        capture_folder = os.path.join(dirpath, "capture_traces")
        if not os.path.isdir(capture_folder):
            capture_folder = None
        gpu_arch = os.path.join(dirpath, "gpu_arch.json")
        if not os.path.isfile(gpu_arch):
            gpu_arch = None
        test_cases.append(
            pytest.param(dirpath, trace_gz, capture_folder, gpu_arch, id=entry)
        )
    return test_cases


# ---------------------------------------------------------------------------
# Regression test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dirpath,trace_gz,capture_folder,gpu_arch_path",
    find_inference_test_cases(),
)
def test_inference_perf_report(
    dirpath, trace_gz, capture_folder, gpu_arch_path, tmp_path, tol=1e-6
):
    """
    Directly call generate_perf_report_pytorch (from the inference module)
    and compare every returned DataFrame against the reference perf_report.xlsx.
    """
    profile_path = os.path.join(dirpath, trace_gz)
    ref_report_path = os.path.join(dirpath, "perf_report.xlsx")

    # Build the augmented graph tree when capture traces are present
    graph_tree = None
    if capture_folder:
        classify_graph_capture_trace(capture_folder)
        metadata_json_path = os.path.join(capture_folder, "execution_details.json")
        graph_tree = merge_capture_trace_into_graph(
            capture_folder, metadata_json_path, profile_path
        )

    # Call the function under test
    output_csvs_dir = str(tmp_path / "csvs")
    os.makedirs(output_csvs_dir, exist_ok=True)
    result = generate_perf_report_pytorch(
        profile_json_path=profile_path,
        augmented_tree=graph_tree,
        output_xlsx_path=None,
        output_csvs_dir=output_csvs_dir,
        enable_pseudo_ops=True,
        group_by_parent_module=True,
        group_by_num_kernels=True,
        collective_analysis=False,
    )

    # result is a dict[str, pd.DataFrame]
    assert isinstance(result, dict), "generate_perf_report_pytorch must return a dict"
    assert len(result) > 0, "Result dict must not be empty"
    assert "gpu_timeline" in result, "gpu_timeline sheet must always be present"

    # Validate returned DataFrames are well-formed
    for sheet_name, df in result.items():
        assert isinstance(df, pd.DataFrame), f"Sheet '{sheet_name}' is not a DataFrame"
        assert not df.empty, f"Sheet '{sheet_name}' is unexpectedly empty"

    # Compare against reference xlsx
    ref_sheets = pd.ExcelFile(ref_report_path).sheet_names
    for sheet in ref_sheets:
        df_ref = pd.read_excel(ref_report_path, sheet_name=sheet)
        if df_ref.empty:
            continue
        assert sheet in result, (
            f"Sheet '{sheet}' exists in reference but was not returned by "
            f"generate_perf_report_pytorch"
        )
        df_test = result[sheet]
        cols = [
            col
            for col in df_ref.columns
            if col not in COLS_IGNORE and col in df_test.columns
        ]
        diff_cols = compare_cols(df_test, df_ref, cols, tol=tol)
        assert not diff_cols, (
            f"Sheet '{sheet}' has differences in {profile_path}:"
            f"{format_diff_details(diff_cols)}"
        )

    # Verify CSV output was written for every sheet
    for sheet_name in result:
        csv_path = os.path.join(output_csvs_dir, f"{sheet_name}.csv")
        assert os.path.exists(
            csv_path
        ), f"CSV output for sheet '{sheet_name}' was not written to {csv_path}"
