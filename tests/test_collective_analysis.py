###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import glob
import os
import shutil
import subprocess

import pandas as pd
import pytest

from conftest import compare_cols, format_diff_details, update_reference_csvs


def generate_nccl_report(trace_pattern, world_size, report_csv_dir):
    """Generate NCCL collective analysis report as per-sheet CSVs."""
    cmd = [
        "python3",
        "-m",
        "TraceLens.Reporting.generate_multi_rank_collective_report_pytorch",
        "--trace_pattern",
        trace_pattern,
        "--world_size",
        str(world_size),
        "--output_csvs_dir",
        report_csv_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running command: {result.stderr}")
    return True


def test_collective_analysis(update_references, tol=1e-6):
    """
    Test NCCL collective analysis report generation.
    Compares generated report against reference for nccl_summary_implicit_sync sheet.

    When ``--update-references`` is passed,
    the checked-in reference CSVs are overwritten with the freshly generated
    output and the test is skipped so the suite still returns green.
    """
    test_dir = "tests/traces/mi300/llama_70b_fsdp"
    trace_pattern = "rank*_trace_no_pyfn.json.gz"

    ref_report_dir = os.path.join(test_dir, "nccl_analysis_report_csvs")
    trace_pattern_full = os.path.join(test_dir, trace_pattern)

    world_size = len(glob.glob(trace_pattern_full))

    fn_root = os.path.join(test_dir, "pytest_reports")
    os.makedirs(fn_root, exist_ok=True)

    try:
        fn_csv_dir = os.path.join(fn_root, "nccl_csvs")
        generate_nccl_report(trace_pattern_full, world_size, fn_csv_dir)

        if update_references:
            update_reference_csvs(fn_csv_dir, ref_report_dir)
            pytest.skip(f"Updated reference: {ref_report_dir}")
            return

        if not os.path.isdir(ref_report_dir):
            pytest.skip(f"Reference CSV directory not found: {ref_report_dir}")

        sheet = "nccl_summary_implicit_sync"
        df_ref = pd.read_csv(os.path.join(ref_report_dir, f"{sheet}.csv"))
        df_fn = pd.read_csv(os.path.join(fn_csv_dir, f"{sheet}.csv"))

        if df_ref.empty:
            assert (
                df_fn.empty
            ), f"Reference is empty but generated report has {len(df_fn)} rows"
            return

        cols = df_ref.columns.tolist()
        diff_cols = compare_cols(df_fn, df_ref, cols, tol=tol)
        assert (
            not diff_cols
        ), f"Sheet '{sheet}' has differences in {test_dir}:{format_diff_details(diff_cols)}"

    finally:
        if os.path.exists(fn_root):
            shutil.rmtree(fn_root)
