#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Generate a multi-sheet comparison report from two semantic breakdowns.

Combines TraceDiff outputs, comparison CSV, and a human-readable kernel mapping
into a single Excel workbook with per-sheet CSV exports.

Input:
  - Two semantic_labels.json files (one per trace)
  - comparison.csv (from match_and_compare.py)
  - TraceDiff output directory (from generate_semantic_diff.py)

Output:
  - comparison_report.xlsx (multi-sheet Excel workbook)
  - comparison_report_csvs/ (one CSV per sheet)

Sheets:
  1. kernel_mapping         - human-readable kernel-to-kernel mapping
  2. comparison             - block-level timing + roofline
  3. diff_stats             - TraceDiff-compatible per-kernel diff
  4. diff_stats_summary     - aggregated diff stats

Usage:
    python generate_comparison_report.py \\
        trace_a/semantic_labels.json trace_b/semantic_labels.json \\
        --comparison comparison.csv \\
        --diff-dir semantic_diff_output/ \\
        --name-a MI355 --name-b B200 \\
        [-o comparison_report.xlsx] \\
        [--output-csvs-dir comparison_report_csvs/]
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from itertools import zip_longest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trace_breakdown"))
from category_mappings import get_group, get_perf_category


def load_labels(path):
    with open(path) as f:
        return json.load(f)


def build_kernel_mapping(labeled_a, labeled_b, name_a, name_b):
    """Build a human-readable kernel mapping DataFrame.

    Layout:
      semantic_block | {name_a} Kernels | {name_b} Kernels | perf_category
      QKV Projection | gemm_kernel_1    | cublas_gemm_1    | GEMM
                     | gemm_kernel_2    |                  |

    The semantic_block and perf_category are filled on the first row of each
    block and left blank on continuation rows. Kernel names are listed one per
    row (sorted). If one trace has more kernels in a block, the shorter side
    gets blank cells.
    """
    blocks_a = OrderedDict()
    for k in labeled_a:
        b = k["semantic_block"]
        blocks_a.setdefault(b, []).append(k["name"])

    blocks_b = OrderedDict()
    for k in labeled_b:
        b = k["semantic_block"]
        blocks_b.setdefault(b, []).append(k["name"])

    all_blocks = list(
        OrderedDict.fromkeys(list(blocks_a.keys()) + list(blocks_b.keys()))
    )

    col_block = "semantic_block"
    col_a = f"{name_a} Kernels"
    col_b = f"{name_b} Kernels"
    col_cat = "perf_category"

    rows = []
    for block in all_blocks:
        kernels_a = sorted(set(blocks_a.get(block, [])))
        kernels_b = sorted(set(blocks_b.get(block, [])))

        paired = list(zip_longest(kernels_a, kernels_b, fillvalue=""))

        if not paired:
            paired = [("", "")]

        for i, (ka, kb) in enumerate(paired):
            is_first = i == 0
            rows.append(
                {
                    col_block: block if is_first else "",
                    col_a: ka,
                    col_b: kb,
                    col_cat: get_perf_category(block) if is_first else "",
                }
            )

    return pd.DataFrame(rows)


def write_report(dict_name2df, output_xlsx_path=None, output_csvs_dir=None):
    """Write sheets as CSV files and/or a single Excel workbook."""
    if output_csvs_dir:
        os.makedirs(output_csvs_dir, exist_ok=True)
        for sheet_name, df in dict_name2df.items():
            csv_path = os.path.join(output_csvs_dir, f"{sheet_name}.csv")
            df.to_csv(csv_path, index=False)
        print(f"Wrote CSVs to {output_csvs_dir}/", file=sys.stderr)

    if output_xlsx_path:
        with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
            for sheet_name, df in dict_name2df.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Wrote {output_xlsx_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-sheet comparison report"
    )
    parser.add_argument("trace_a", help="Path to trace A semantic_labels.json")
    parser.add_argument("trace_b", help="Path to trace B semantic_labels.json")
    parser.add_argument(
        "--comparison",
        required=True,
        help="Path to comparison.csv (from match_and_compare.py)",
    )
    parser.add_argument(
        "--diff-dir",
        required=True,
        help="TraceDiff output directory (from generate_semantic_diff.py)",
    )
    parser.add_argument("--name-a", default="trace_a", help="Short name for trace A")
    parser.add_argument("--name-b", default="trace_b", help="Short name for trace B")
    parser.add_argument(
        "-o",
        "--output",
        default="comparison_report.xlsx",
        help="Output Excel path (default: comparison_report.xlsx)",
    )
    parser.add_argument(
        "--output-csvs-dir", default=None, help="Directory to write per-sheet CSV files"
    )
    args = parser.parse_args()

    data_a = load_labels(args.trace_a)
    data_b = load_labels(args.trace_b)
    labeled_a = data_a["labeled_kernels"]
    labeled_b = data_b["labeled_kernels"]

    print("Building kernel mapping...", file=sys.stderr)
    kernel_mapping_df = build_kernel_mapping(
        labeled_a, labeled_b, args.name_a, args.name_b
    )

    print("Loading comparison and diff data...", file=sys.stderr)
    comparison_df = pd.read_csv(args.comparison)

    diff_stats_path = os.path.join(args.diff_dir, "diff_stats.csv")
    diff_stats_df = pd.read_csv(diff_stats_path)

    summary_path = os.path.join(args.diff_dir, "diff_stats_unique_args_summary.csv")
    diff_summary_df = pd.read_csv(summary_path)

    dict_name2df = OrderedDict()
    dict_name2df["kernel_mapping"] = kernel_mapping_df
    dict_name2df["comparison"] = comparison_df
    dict_name2df["diff_stats"] = diff_stats_df
    dict_name2df["diff_stats_summary"] = diff_summary_df

    output_xlsx = args.output
    output_csvs = args.output_csvs_dir

    if output_xlsx is None and output_csvs is None:
        output_xlsx = "comparison_report.xlsx"

    write_report(
        dict_name2df, output_xlsx_path=output_xlsx, output_csvs_dir=output_csvs
    )

    for name, df in dict_name2df.items():
        print(f"  Sheet '{name}': {len(df)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
