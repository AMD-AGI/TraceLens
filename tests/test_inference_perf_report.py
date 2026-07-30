###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Regression tests for generate_perf_report_pytorch_inference.
# Each test case is a subdirectory under tests/traces/inference/ containing:
#   - A .json.gz trace file
#   - A perf_csvs/ folder with reference CSV files (one per output sheet)
#   - Optionally capture_traces/ (graph capture mode)
#   - Optionally gpu_arch.json

import glob
import os

import numpy as np
import pandas as pd
import pytest
import ast
import re
from pandas.api.types import is_float_dtype

import gzip
import json

from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    generate_perf_report_pytorch,
    classify_graph_capture_trace,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
import TraceLens.Trace2Tree.trace_capture_merge_experimental as _merge_mod
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
    get_subtree_events,
    build_execution_graph_root_map,
    _get_cached_capture_tree,
    _capture_kernel_name,
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
    Each subdirectory with a .json.gz and a perf_csvs/ folder becomes a test case.
    """
    test_cases = []
    if not os.path.isdir(INFERENCE_TRACES_ROOT):
        return test_cases
    for entry in sorted(os.listdir(INFERENCE_TRACES_ROOT)):
        dirpath = os.path.join(INFERENCE_TRACES_ROOT, entry)
        if not os.path.isdir(dirpath):
            continue
        gz_files = [f for f in os.listdir(dirpath) if f.endswith(".json.gz")]
        perf_csvs_dir = os.path.join(dirpath, "perf_csvs")
        if not gz_files or not os.path.isdir(perf_csvs_dir):
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


@pytest.fixture(autouse=True)
def _clear_capture_tree_cache():
    # Reproduce per-process isolation: the capture-tree cache is keyed by
    # {batch_size}_{mode}, which collides across fixtures sharing a key.
    _merge_mod._capture_tree_cache.clear()
    yield
    _merge_mod._capture_tree_cache.clear()


@pytest.mark.parametrize(
    "dirpath,trace_gz,capture_folder,gpu_arch_path",
    find_inference_test_cases(),
)
def test_inference_perf_report(
    dirpath, trace_gz, capture_folder, gpu_arch_path, tmp_path, tol=1e-6
):
    """
    Directly call generate_perf_report_pytorch (from the inference module)
    and compare every output CSV against the reference CSVs in perf_csvs/.
    """
    profile_path = os.path.join(dirpath, trace_gz)
    ref_csvs_dir = os.path.join(dirpath, "perf_csvs")

    # Build the augmented graph tree when capture traces are present
    graph_tree = None
    if capture_folder:
        metadata_json_path = os.path.join(capture_folder, "execution_details.json")
        capture_files = glob.glob(os.path.join(capture_folder, "*.json.gz"))
        single_capture_trace = (
            not os.path.exists(metadata_json_path) and len(capture_files) == 1
        )
        if not single_capture_trace:
            classify_graph_capture_trace(capture_folder)
        graph_tree = merge_capture_trace_into_graph(
            capture_folder,
            metadata_json_path,
            profile_path,
            single_capture_trace=single_capture_trace,
        )

    # Call the function under test — write CSVs to a temp directory
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
        include_call_stack=True,
    )

    # result is a dict[str, pd.DataFrame]
    assert isinstance(result, dict), "generate_perf_report_pytorch must return a dict"
    assert len(result) > 0, "Result dict must not be empty"
    assert "gpu_timeline" in result, "gpu_timeline sheet must always be present"

    # Validate returned DataFrames are well-formed
    for sheet_name, df in result.items():
        assert isinstance(df, pd.DataFrame), f"Sheet '{sheet_name}' is not a DataFrame"
        assert not df.empty, f"Sheet '{sheet_name}' is unexpectedly empty"

    # Verify CSV output was written for every sheet
    for sheet_name in result:
        csv_path = os.path.join(output_csvs_dir, f"{sheet_name}.csv")
        assert os.path.exists(
            csv_path
        ), f"CSV output for sheet '{sheet_name}' was not written to {csv_path}"

    # Compare each generated CSV against the reference CSV in perf_csvs/
    ref_csv_files = [f for f in os.listdir(ref_csvs_dir) if f.endswith(".csv")]
    for csv_file in ref_csv_files:
        sheet_name = csv_file[: -len(".csv")]
        ref_csv_path = os.path.join(ref_csvs_dir, csv_file)
        df_ref = pd.read_csv(ref_csv_path)
        if df_ref.empty:
            continue
        assert sheet_name in result, (
            f"'{sheet_name}' exists in reference perf_csvs/ but was not returned by "
            f"generate_perf_report_pytorch"
        )
        df_test = result[sheet_name]
        cols = [
            col
            for col in df_ref.columns
            if col not in COLS_IGNORE and col in df_test.columns
        ]
        diff_cols = compare_cols(df_test, df_ref, cols, tol=tol)
        assert not diff_cols, (
            f"'{sheet_name}' has differences in {profile_path}:"
            f"{format_diff_details(diff_cols)}"
        )


# ---------------------------------------------------------------------------
# Capture merge validation: verifies that kernel timing from the replay trace
# and input args/call stacks from the capture trace are correctly reflected
# in the report for a specific GEMM shape.
# ---------------------------------------------------------------------------

XDIT_TRACE_DIR = os.path.join(INFERENCE_TRACES_ROOT, "xdit_flux_aiter")
TARGET_GEMM_DIMS = ((4608, 15360), (15360, 3072), (4608, 3072))


def test_xdit_capture_merge():
    """Capture merge validation for a specific GEMM in the xDiT trace.

    Selects every aten::mm with dims ((4608, 15360), (15360, 3072), (4608, 3072))
    from the capture trace, finds the corresponding kernels in the replay trace
    via positional index, and checks the report's timing and call stacks match.

    Uses the reference CSVs in perf_csvs/ (validated by test_inference_perf_report).
    """

    _merge_mod._capture_tree_cache.clear()

    profile_path = glob.glob(os.path.join(XDIT_TRACE_DIR, "*.json.gz"))[0]
    capture_folder = os.path.join(XDIT_TRACE_DIR, "capture_traces")
    capture_path = glob.glob(os.path.join(capture_folder, "*.json.gz"))[0]
    ref_csvs_dir = os.path.join(XDIT_TRACE_DIR, "perf_csvs")

    # --- 1. Load capture trace: find target GEMMs ---
    key = ("single", os.path.abspath(capture_path))
    cap_tree, cap_roots, cap_root_data = _get_cached_capture_tree(
        key,
        capture_path,
        TreePerfAnalyzer,
    )
    cached_events, filtered_uids = cap_root_data[0]
    UID = _merge_mod.UID
    cap_filtered = [e for e in cached_events if e[UID] in filtered_uids]

    target_dims = [list(d) for d in TARGET_GEMM_DIMS]
    cap_mm_ops = [
        e
        for e in cap_tree.events
        if e.get("name") == "aten::mm"
        and e.get("cat") == "cpu_op"
        and e.get("args", {}).get("Input Dims") == target_dims
    ]

    # --- 2. Load reference CSVs (generated by test_inference_perf_report) ---
    M, K, N = TARGET_GEMM_DIMS[0][0], TARGET_GEMM_DIMS[0][1], TARGET_GEMM_DIMS[1][1]
    df_gemm = pd.read_csv(os.path.join(ref_csvs_dir, "GEMM.csv"))
    match = df_gemm[
        (df_gemm["param: M"].astype(int) == M)
        & (df_gemm["param: K"].astype(int) == K)
        & (df_gemm["param: N"].astype(int) == N)
    ]
    assert (
        len(match) == 1
    ), f"Expected 1 GEMM row for M={M} K={K} N={N}, got {len(match)}"
    row = match.iloc[0]
    report_time_us = float(row["Kernel Time (µs)_sum"])
    report_count = int(row["name_count"])

    # --- 4. Find exact kernels in replay trace via positional index ---
    # The capture root's filtered events align 1:1 with each graph launch's
    # kernel events. We find the indices of our target GEMM's child kernels
    # in the capture root, then use those same indices into each graph
    # launch's replay kernels to sum the exact runtime.
    cap_uid_to_idx = {e[UID]: i for i, e in enumerate(cap_filtered)}

    target_kernel_indices = []
    for mm_op in cap_mm_ops:
        for child_uid in mm_op.get("children", []):
            if child_uid in cap_uid_to_idx:
                target_kernel_indices.append(cap_uid_to_idx[child_uid])

    graph_perf = TreePerfAnalyzer.from_file(profile_path, add_python_func=True)
    exec_map = build_execution_graph_root_map(
        graph_perf.tree, single_capture_trace=True
    )
    _, graph_roots = exec_map[0]

    replay_time_us = 0
    replay_count = 0
    for gl_root in graph_roots:
        _, gf = get_subtree_events(
            graph_perf.tree,
            gl_root,
            cat_filter=["kernel", "gpu_memset", "gpu_memcpy"],
        )
        for idx in target_kernel_indices:
            cap_name = _capture_kernel_name(cap_filtered[idx])
            replay_name = gf[idx].get("name", "")
            assert cap_name == replay_name, (
                f"Kernel name mismatch at index {idx}: "
                f"capture='{cap_name}' vs replay='{replay_name}'"
            )
            replay_time_us += gf[idx].get("dur", 0)
            replay_count += 1

    assert (
        replay_count == report_count
    ), f"Kernel count mismatch: replay={replay_count} vs report={report_count}"
    assert abs(report_time_us - replay_time_us) / replay_time_us < 0.001, (
        f"Kernel time mismatch: report={report_time_us:.1f}µs "
        f"vs replay={replay_time_us:.1f}µs"
    )

    # --- 5. Verify call stacks contain _run_timed_pipe and aten::mm ---
    df_unified = pd.read_csv(os.path.join(ref_csvs_dir, "unified_perf_summary.csv"))
    mm_rows = df_unified[df_unified["name"] == "aten::mm"]
    has_expected_stack = mm_rows["call_stack_full"].apply(
        lambda x: isinstance(x, str) and "_run_timed_pipe" in x and "aten::mm" in x
    )
    assert has_expected_stack.all(), (
        f"{has_expected_stack.sum()}/{len(mm_rows)} aten::mm rows "
        f"have _run_timed_pipe and aten::mm in call stack"
    )

    _merge_mod._capture_tree_cache.clear()
