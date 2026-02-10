###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import subprocess
import pandas as pd
from test_compare_perf_report import compare_cols, format_diff_details


def generate_nccl_report(trace_pattern, world_size, report_path):
    """Generate NCCL collective analysis report."""
    cmd = [
        "python3",
        "-m",
        "TraceLens.Reporting.generate_multi_rank_collective_report_pytorch",
        "--trace_pattern",
        trace_pattern,
        "--world_size",
        str(world_size),
        "--output_xlsx_path",
        report_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running command: {result.stderr}")
    return True


def test_collective_analysis(tol=1e-6):
    """
    Test NCCL collective analysis report generation.
    Compares generated report against reference for nccl_summary_implicit_sync sheet.
    """
    import shutil
    import glob

    test_dir = "tests/traces/mi300/llama_70b_fsdp"
    trace_pattern = "rank*_trace_no_pyfn.json.gz"

    ref_report_path = os.path.join(test_dir, "nccl_analysis_report.xlsx")
    trace_pattern_full = os.path.join(test_dir, trace_pattern)

    # Count world size from actual files
    world_size = len(glob.glob(trace_pattern_full))

    # Generate a temp output directory for this test
    fn_root = os.path.join(test_dir, "pytest_reports")
    os.makedirs(fn_root, exist_ok=True)

    try:
        # Generate report
        fn_report_path = os.path.join(fn_root, "nccl_analysis_report.xlsx")
        generate_nccl_report(trace_pattern_full, world_size, fn_report_path)

        # Compare the nccl_summary_implicit_sync sheet
        sheet = "nccl_summary_implicit_sync"
        df_ref = pd.read_excel(ref_report_path, sheet_name=sheet)
        df_fn = pd.read_excel(fn_report_path, sheet_name=sheet)

        if df_ref.empty:
            assert (
                df_fn.empty
            ), f"Reference is empty but generated report has {len(df_fn)} rows"
            return

        # Compare all columns
        cols = df_ref.columns.tolist()
        diff_cols = compare_cols(df_fn, df_ref, cols, tol=tol)
        assert (
            not diff_cols
        ), f"Sheet '{sheet}' has differences in {test_dir}:{format_diff_details(diff_cols)}"

    finally:
        # Cleanup: remove generated report
        if os.path.exists(fn_root):
            shutil.rmtree(fn_root)
