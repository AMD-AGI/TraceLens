###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Generate a performance report with activation recompute detection and compute
the percentage of GPU time spent on recomputation.

Usage:
    python recompute_analysis.py --profile_json_path trace.json
    python recompute_analysis.py --profile_json_path trace.json --output_xlsx_path report.xlsx
    python recompute_analysis.py --profile_json_path trace.json --output_csvs_dir csvs/
"""

import argparse
import pandas as pd
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)


def compute_recompute_pct(df: pd.DataFrame, time_col: str) -> dict:
    """Compute recompute vs non-recompute time breakdown from a DataFrame
    that has an is_recompute column and a time column."""
    if "is_recompute" not in df.columns or time_col not in df.columns:
        return None

    total = df[time_col].sum()
    recompute = df.loc[df["is_recompute"] == True, time_col].sum()
    non_recompute = total - recompute
    pct = (recompute / total * 100) if total > 0 else 0.0

    return {
        "total_time": total,
        "recompute_time": recompute,
        "non_recompute_time": non_recompute,
        "recompute_pct": pct,
    }


def print_recompute_summary(sheets: dict[str, pd.DataFrame]):
    """Print a recompute summary from the generated report sheets."""
    print("\n" + "=" * 70)
    print("ACTIVATION RECOMPUTE ANALYSIS")
    print("=" * 70)

    # ops_summary: total kernel time by op name, split by is_recompute
    if "ops_summary" in sheets:
        result = compute_recompute_pct(
            sheets["ops_summary"], "total_direct_kernel_time_ms"
        )
        if result:
            print(f"\n--- ops_summary (by operation name) ---")
            print(f"  Total GPU time:        {result['total_time']:.2f} ms")
            print(f"  Recompute GPU time:    {result['recompute_time']:.2f} ms")
            print(f"  Non-recompute time:    {result['non_recompute_time']:.2f} ms")
            print(f"  Recompute %:           {result['recompute_pct']:.2f}%")

    # ops_unique_args: total kernel time by unique (op, args), split by is_recompute
    if "ops_unique_args" in sheets:
        result = compute_recompute_pct(
            sheets["ops_unique_args"], "total_direct_kernel_time_sum"
        )
        if result:
            print(f"\n--- ops_unique_args (by unique op + args) ---")
            print(f"  Total GPU time:        {result['total_time']:.1f} us")
            print(f"  Recompute GPU time:    {result['recompute_time']:.1f} us")
            print(f"  Non-recompute time:    {result['non_recompute_time']:.1f} us")
            print(f"  Recompute %:           {result['recompute_pct']:.2f}%")

    # Per-op breakdown from ops_summary
    if "ops_summary" in sheets:
        df = sheets["ops_summary"]
        if "is_recompute" in df.columns:
            df_recompute = df[df["is_recompute"] == True].sort_values(
                "total_direct_kernel_time_ms", ascending=False
            )
            if not df_recompute.empty:
                print(f"\n--- Top recomputed operations ---")
                for _, row in df_recompute.head(10).iterrows():
                    print(
                        f"  {row['name']:<45s} "
                        f"{row['total_direct_kernel_time_ms']:>8.2f} ms  "
                        f"({row.get('Percentage (%)', 0):>5.2f}%)"
                    )

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a perf report with activation recompute detection "
        "and compute recomputation percentage."
    )
    parser.add_argument(
        "--profile_json_path",
        type=str,
        required=True,
        help="Path to the PyTorch profiler trace JSON (supports .json, .json.gz, .zip)",
    )
    parser.add_argument(
        "--output_xlsx_path",
        type=str,
        default=None,
        help="Path to save the Excel report. Auto-inferred from trace name if not set.",
    )
    parser.add_argument(
        "--output_csvs_dir",
        type=str,
        default=None,
        help="If set, save each sheet as a CSV in this directory instead of Excel.",
    )
    parser.add_argument(
        "--gpu_arch_json_path",
        type=str,
        default=None,
        help="Path to GPU architecture JSON for roofline analysis.",
    )
    parser.add_argument(
        "--topk_ops",
        type=int,
        default=None,
        help="Limit number of rows in ops_unique_args.",
    )
    args = parser.parse_args()

    print(f"Loading trace: {args.profile_json_path}")
    print("Running perf report with detect_recompute=True ...\n")

    sheets = generate_perf_report_pytorch(
        profile_json_path=args.profile_json_path,
        output_xlsx_path=args.output_xlsx_path,
        output_csvs_dir=args.output_csvs_dir,
        gpu_arch_json_path=args.gpu_arch_json_path,
        topk_ops=args.topk_ops,
        detect_recompute=True,
    )

    print_recompute_summary(sheets)


if __name__ == "__main__":
    main()
