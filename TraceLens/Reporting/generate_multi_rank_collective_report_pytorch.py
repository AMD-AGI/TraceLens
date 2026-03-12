###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import re
import argparse
import pandas as pd
import glob
import sys
from typing import List, Dict, Optional, Union
import subprocess
import warnings
from TraceLens import NcclAnalyser
from TraceLens.Reporting.reporting_utils import request_install

DEFAULT_RANK_REGEX = r"rank[\[\-_/]?(?P<rank>\d+)"


def find_trace_files(input_dir: str, pattern: str = "rank*_trace.json") -> List[str]:
    """Find all trace files matching the pattern in the input directory."""
    search_pattern = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(search_pattern))

    if not files:
        print(f"No trace files found matching pattern '{pattern}' in {input_dir}")
    else:
        print(f"Found {len(files)} trace files in {input_dir}")

    return files


def infer_world_size(trace_files: List[str]) -> int:
    """Infer world size from the number of trace files."""
    return len(trace_files)


def _resolve_trace_files_glob(
    trace_glob: str,
    world_size: int,
    rank_regex: str = DEFAULT_RANK_REGEX,
    strict: bool = True,
) -> List[str]:
    """Resolve trace file paths from a recursive glob pattern.

    Files are matched against *rank_regex* to extract the rank id.  When
    multiple files match the same rank, the lexicographically first path is
    kept (stable, deterministic selection).

    Returns an ordered list of length *world_size* with one path per rank.
    """
    rx = re.compile(rank_regex)
    matches = sorted(glob.glob(trace_glob, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No files matched --trace_glob: {trace_glob}")

    by_rank: Dict[int, str] = {}
    for path in matches:
        m = rx.search(os.path.basename(path)) or rx.search(path)
        if m is None:
            continue
        rank_str = m.groupdict().get("rank") or m.group(1)
        try:
            rank = int(rank_str)
        except (ValueError, IndexError):
            continue
        by_rank.setdefault(rank, path)

    if not by_rank:
        raise ValueError(
            f"--trace_glob matched {len(matches)} file(s), but none matched "
            f"--rank_regex '{rank_regex}'"
        )

    missing = [r for r in range(world_size) if r not in by_rank]
    if missing:
        msg = f"Missing ranks after glob resolution: {missing}"
        if strict:
            raise FileNotFoundError(msg)
        warnings.warn(msg)

    return [by_rank[r] for r in range(world_size) if r in by_rank]


def generate_collective_report(
    trace_dir: Optional[str] = None,
    trace_pattern: Optional[str] = None,
    trace_glob: Optional[str] = None,
    world_size: Optional[int] = None,
    output_xlsx_path: Optional[str] = None,
    output_csvs_dir: Optional[str] = None,
    detailed_analysis: bool = False,
    agg_metrics: List[str] = ["mean", "median", "min", "max"],
    strict_world_size_check: bool = True,
    use_multiprocessing: bool = False,
    max_workers: Optional[int] = None,
    rank_regex: str = DEFAULT_RANK_REGEX,
) -> Dict[str, pd.DataFrame]:
    """
    Generate comprehensive NCCL communication analysis reports.

    Args:
        trace_dir: Directory containing trace files. The files must match the pattern "rank*_trace.json".
        trace_pattern: Template path with a single `*` placeholder for rank.
                        Example: /path/to/trace_rank_*_step_3.json
                        The `*` will be replaced with 0..world_size-1.
        trace_glob: Glob pattern (supports ``**``) for trace files with arbitrary
                    names.  Requires *world_size* and uses *rank_regex* to extract
                    the rank id from each matched path.
        world_size: Number of ranks
        output_xlsx_path: Path to output Excel file
        output_csvs_dir: Directory to save CSV files
        detailed_analysis: Whether to include detailed per-rank information
        agg_metrics: Aggregation metrics to include in summary
        use_multiprocessing: Whether to use multiprocessing for parallel trace loading (default: False).
                            When enabled, can provide significant speedup (system-dependent). When False, uses sequential loading.
        max_workers: Maximum number of worker processes for parallel loading (only used if use_multiprocessing=True).
                    Default: os.cpu_count(). Override to limit resource usage if needed.
        rank_regex: Regex with a named group ``rank`` (or a single capture group)
                    used to extract the rank id from filenames matched by *trace_glob*.

    Returns:
        Dictionary mapping sheet names to DataFrames
    """
    if world_size is None:
        raise ValueError("world_size must be provided")

    if sum(bool(x) for x in [trace_dir, trace_pattern, trace_glob]) != 1:
        raise ValueError(
            "Provide exactly one of trace_dir, trace_pattern, or trace_glob"
        )

    list_trace_filepaths: List[str] = []

    if trace_glob:
        list_trace_filepaths = _resolve_trace_files_glob(
            trace_glob,
            world_size,
            rank_regex=rank_regex,
            strict=strict_world_size_check,
        )
    elif trace_pattern:
        if trace_pattern.count("*") != 1:
            raise ValueError(
                "trace_pattern must contain exactly one '*' character as a placeholder for rank id"
            )
        for i in range(world_size):
            expected_file = trace_pattern.replace("*", str(i))
            if not os.path.isfile(expected_file):
                if strict_world_size_check:
                    raise FileNotFoundError(
                        f"Expected trace file not found: {expected_file}"
                    )
                else:
                    warnings.warn(
                        f"Expected trace file not found: {expected_file}. Skipping."
                    )
                    continue
            list_trace_filepaths.append(expected_file)
    else:
        for i in range(world_size):
            expected_file = os.path.join(trace_dir, f"rank{i}_trace.json")
            if not os.path.isfile(expected_file):
                if strict_world_size_check:
                    raise FileNotFoundError(
                        f"Expected trace file not found: {expected_file}"
                    )
                else:
                    warnings.warn(
                        f"Expected trace file not found: {expected_file}. Skipping."
                    )
                    continue
            list_trace_filepaths.append(expected_file)

    # Initialize NCCL analyzer
    nccl_analyser = NcclAnalyser(
        list_trace_filepaths,
        world_size,
        use_multiprocessing=use_multiprocessing,
        max_workers=max_workers,
    )

    # Generate DataFrames
    report_dfs = {}

    # Add summary dataframes
    print("Generating summary reports...")
    report_dfs["nccl_summary_implicit_sync"] = (
        nccl_analyser.build_df_summary_nccl_implicit_sync_cat(agg_metrics=agg_metrics)
    )
    report_dfs["nccl_summary_long"] = nccl_analyser.build_df_summary_long()

    # Add detailed per-rank information if requested
    if detailed_analysis:
        print("Generating detailed per-rank analysis...")
        report_dfs["nccl_long"] = nccl_analyser.build_df_long()
        report_dfs["nccl_implicit_sync"] = (
            nccl_analyser.build_df_nccl_implicit_sync_cat(detailed=True)
        )
        df_all2allv = nccl_analyser.build_df_nccl_all2allv(detailed=True)
        if df_all2allv is not None and not df_all2allv.empty:
            report_dfs["nccl_all2allv"] = df_all2allv

    # Export DataFrames
    if output_csvs_dir:
        os.makedirs(output_csvs_dir, exist_ok=True)
        for sheet_name, df in report_dfs.items():
            csv_path = os.path.join(output_csvs_dir, f"{sheet_name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"DataFrame '{sheet_name}' written to {csv_path}")

    if output_xlsx_path:
        try:
            import openpyxl
        except (ImportError, ModuleNotFoundError):
            print("Error importing openpyxl")
            request_install("openpyxl")

        print(f"Writing Excel report to {output_xlsx_path}...")
        with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
            for sheet_name, df in report_dfs.items():
                df.to_excel(
                    writer, sheet_name=sheet_name[:31], index=False
                )  # Excel limits sheet names to 31 chars
        print(f"Excel report successfully written to {output_xlsx_path}")

    return report_dfs


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive NCCL communication analysis reports"
    )

    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--trace_dir", type=str, help="Directory containing trace files"
    )
    input_group.add_argument(
        "--trace_pattern",
        type=str,
        help="Template path with a single * placeholder for rank. Example: /path/to/trace_rank_*_step_3.json",
    )
    input_group.add_argument(
        "--trace_glob",
        type=str,
        help=(
            "Glob for trace files with arbitrary names (supports **). "
            "Requires --world_size and uses --rank_regex to map files to ranks."
        ),
    )

    parser.add_argument(
        "--world_size", type=int, default=None, help="Number of ranks (required)"
    )
    parser.add_argument(
        "--rank_regex",
        type=str,
        default=DEFAULT_RANK_REGEX,
        help=(
            "Regex used with --trace_glob to extract rank id from filenames. "
            "Must contain a named group 'rank' or a single capture group. "
            f"Default: {DEFAULT_RANK_REGEX!r}"
        ),
    )

    # Output arguments
    parser.add_argument(
        "--output_xlsx_path", type=str, default=None, help="Path to output Excel file"
    )
    parser.add_argument(
        "--output_csvs_dir", type=str, default=None, help="Directory to save CSV files"
    )

    # Analysis options
    parser.add_argument(
        "--detailed_analysis", action="store_true", help="Include detailed information"
    )
    parser.add_argument(
        "--agg_metrics",
        type=str,
        nargs="+",
        default=["mean", "median", "min", "max"],
        help="Aggregation metrics to include in summary",
    )
    parser.add_argument(
        "--use_multiprocessing",
        action="store_true",
        help="Enable parallel trace loading using multiprocessing (can provide significant speedup, default: disabled)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of worker processes for parallel loading (requires --use_multiprocessing, default: os.cpu_count())",
    )

    args = parser.parse_args()

    # If no output specified, create default Excel output
    if args.output_xlsx_path is None and args.output_csvs_dir is None:
        if args.trace_dir:
            output_dir = args.trace_dir
        elif args.trace_pattern:
            p0 = args.trace_pattern.replace("*", "0")
            p1 = args.trace_pattern.replace("*", "1")
            d0, d1 = os.path.dirname(p0), os.path.dirname(p1)
            while d0 != d1:
                d0, d1 = os.path.dirname(d0), os.path.dirname(d1)
            output_dir = d0
        else:
            trace_files = _resolve_trace_files_glob(
                args.trace_glob,
                args.world_size,
                rank_regex=args.rank_regex,
                strict=True,
            )
            common = os.path.commonpath([os.path.abspath(p) for p in trace_files])
            output_dir = (
                os.path.dirname(common) if os.path.isfile(common) else common
            )

        default_output = os.path.join(output_dir, "nccl_analysis_report.xlsx")
        print(f"No output specified. Using default: {default_output}")
        args.output_xlsx_path = default_output

    # Generate report
    generate_collective_report(
        trace_dir=args.trace_dir,
        trace_pattern=args.trace_pattern,
        trace_glob=args.trace_glob,
        world_size=args.world_size,
        output_xlsx_path=args.output_xlsx_path,
        output_csvs_dir=args.output_csvs_dir,
        detailed_analysis=args.detailed_analysis,
        agg_metrics=args.agg_metrics,
        use_multiprocessing=args.use_multiprocessing,
        max_workers=args.max_workers,
        rank_regex=args.rank_regex,
    )


if __name__ == "__main__":
    main()
