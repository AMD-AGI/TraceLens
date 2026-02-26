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


def _recompute_split(df: pd.DataFrame, value_col: str) -> tuple:
    """Return (total, recompute, pct) for a value column split by is_recompute."""
    if "is_recompute" not in df.columns or value_col not in df.columns:
        return None, None, None
    total = df[value_col].sum()
    recompute = df.loc[df["is_recompute"] == True, value_col].sum()
    pct = (recompute / total * 100) if total > 0 else 0.0
    return total, recompute, pct


def print_recompute_summary(sheets: dict[str, pd.DataFrame]):
    """Print a recompute summary from the generated report sheets."""
    print("\n" + "=" * 70)
    print("ACTIVATION RECOMPUTE ANALYSIS")
    print("=" * 70)

    # GPU time breakdown from ops_summary
    if "ops_summary" in sheets:
        df = sheets["ops_summary"]
        total_ms, recompute_ms, time_pct = _recompute_split(
            df, "total_direct_kernel_time_ms"
        )
        if total_ms is not None:
            print(f"\n  Total GPU time:        {total_ms:>10.2f} ms")
            print(f"  Recompute GPU time:    {recompute_ms:>10.2f} ms")
            print(f"  Non-recompute time:    {total_ms - recompute_ms:>10.2f} ms")
            print(f"  Recompute time %:      {time_pct:>10.2f}%")

    # FLOPS breakdown from unified_perf_summary
    if "unified_perf_summary" in sheets:
        df = sheets["unified_perf_summary"]
        if "is_recompute" in df.columns and "GFLOPS" in df.columns:
            df = df.copy()
            df["total_GFLOPS"] = df["GFLOPS"] * df["operation_count"]
            total_gf, recompute_gf, flops_pct = _recompute_split(df, "total_GFLOPS")
            if total_gf is not None:
                print(f"\n  Total GFLOPS:          {total_gf:>10.2f}")
                print(f"  Recompute GFLOPS:      {recompute_gf:>10.2f}")
                print(f"  Recompute FLOPS %:     {flops_pct:>10.2f}%")

    # Per-op breakdown from ops_summary
    if "ops_summary" in sheets:
        df = sheets["ops_summary"]
        if "is_recompute" in df.columns:
            df_recompute = df[df["is_recompute"] == True].sort_values(
                "total_direct_kernel_time_ms", ascending=False
            )
            if not df_recompute.empty:
                print(f"\n  Top recomputed operations:")
                for _, row in df_recompute.head(10).iterrows():
                    print(
                        f"    {row['name']:<45s} "
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
