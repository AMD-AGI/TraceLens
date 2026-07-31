###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Regression test for TraceDiff output.

Generates TraceDiff reports for a pair of traces and compares CSVs against
checked-in references in tests/traces/tracediff_test/.
"""

import difflib
import os

import pandas as pd
import pytest

from TraceLens import TraceDiff
from TraceLens.TreePerf import TreePerfAnalyzer

from conftest import (
    compare_cols,
    format_diff_details,
    update_reference_csvs,
)

_TRACES_DIR = os.path.join(os.path.dirname(__file__), "traces")
_REF_DIR = os.path.join(_TRACES_DIR, "tracediff_test")
_TRACE1 = os.path.join(
    _TRACES_DIR, "mi300", "facebook_timesformer-base-finetuned-k400__1016002.json.gz"
)
_TRACE2 = os.path.join(
    _TRACES_DIR, "h100", "facebook_timesformer-base-finetuned-k400__1016002.json.gz"
)


def _compare_csv(ref_path, test_path, tol=1e-3):
    """Load two CSVs and compare all shared columns."""
    df_ref = pd.read_csv(ref_path)
    df_test = pd.read_csv(test_path)

    assert len(df_test) == len(df_ref), (
        f"Row count mismatch: reference has {len(df_ref)} rows, "
        f"test has {len(df_test)} rows"
    )

    cols = [c for c in df_ref.columns if c in df_test.columns]
    return compare_cols(df_test, df_ref, cols, tol=tol)


def test_tracediff_regression(tmp_path, update_references):
    """Generate TraceDiff output and compare against checked-in references."""
    output_dir = str(tmp_path / "tracediff_output")

    pa1 = TreePerfAnalyzer.from_file(_TRACE1)
    pa2 = TreePerfAnalyzer.from_file(_TRACE2)
    td = TraceDiff(pa1.tree, pa2.tree)
    td.generate_tracediff_report()
    td.print_tracediff_report_files(output_folder=output_dir)

    if update_references:
        update_reference_csvs(output_dir, _REF_DIR)
        pytest.skip(f"Updated reference: {_REF_DIR}")
        return

    errors = []

    for csv_file in ["diff_stats.csv", "diff_stats_unique_args_summary.csv"]:
        diff = _compare_csv(
            os.path.join(_REF_DIR, csv_file),
            os.path.join(output_dir, csv_file),
        )
        if diff:
            errors.append(f"{csv_file}:{format_diff_details(diff)}")

    for filename in [
        "cpu_op_map.json",
        "cpu_op_map_trace1.json",
        "cpu_op_map_trace2.json",
        "merged_tree_output.txt",
    ]:
        ref_file = os.path.join(_REF_DIR, filename)
        test_file = os.path.join(output_dir, filename)
        if not os.path.exists(ref_file):
            continue
        with open(ref_file) as f:
            ref_lines = f.readlines()
        with open(test_file) as f:
            test_lines = f.readlines()
        if ref_lines != test_lines:
            diff = list(
                difflib.unified_diff(
                    ref_lines,
                    test_lines,
                    fromfile=f"reference/{filename}",
                    tofile=f"test/{filename}",
                    n=3,
                )
            )
            # Truncate to first 30 diff lines to keep output readable
            preview = "".join(diff[:30])
            if len(diff) > 30:
                preview += f"\n... ({len(diff) - 30} more diff lines)\n"
            errors.append(f"{filename}:\n{preview}")

    assert not errors, "\n".join(errors)
