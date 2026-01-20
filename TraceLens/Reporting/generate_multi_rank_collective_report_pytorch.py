###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import argparse
import pandas as pd
import glob
import sys
from typing import List, Dict, Optional, Union
import subprocess
import warnings
import re
from dataclasses import dataclass
import ast
import time
from TraceLens import NcclAnalyser
from TraceLens.Reporting.reporting_utils import request_install


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


@dataclass(frozen=True)
class _RankedTraceFile:
    rank: int
    path: str


def _fmt_seconds(s: float) -> str:
    if s < 1:
        return f"{s:.2f}s"
    if s < 60:
        return f"{s:.1f}s"
    return f"{s/60:.1f}m"


def _print_stage_start(msg: str) -> float:
    print(f"[TraceLens] {msg}...", flush=True)
    return time.perf_counter()


def _print_stage_done(msg: str, t0: float) -> None:
    dt = time.perf_counter() - t0
    print(f"[TraceLens] {msg} done ({_fmt_seconds(dt)})", flush=True)


def _try_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _select_trace_files(
    *,
    trace_dir: Optional[str],
    trace_pattern: Optional[str],
    trace_glob: Optional[str],
    world_size: int,
    strict_world_size_check: bool,
    rank_regex: str,
) -> List[str]:
    """
    Resolve trace file paths in one of three modes:
      - trace_pattern: exactly one '*' placeholder replaced with 0..world_size-1
      - trace_dir: expects rank{i}_trace.json
      - trace_glob: glob for arbitrary trace names; ranks extracted via regex
    """
    if sum(bool(x) for x in [trace_dir, trace_pattern, trace_glob]) != 1:
        raise ValueError(
            "Provide exactly one of trace_dir, trace_pattern, or trace_glob"
        )

    if trace_glob:
        rx = re.compile(rank_regex)
        matches = glob.glob(trace_glob, recursive=True)
        if not matches:
            raise FileNotFoundError(
                f"No trace files matched --trace_glob: {trace_glob}"
            )

        ranked: List[_RankedTraceFile] = []
        for p in matches:
            m = rx.search(os.path.basename(p)) or rx.search(p)
            if not m:
                continue
            rank_str = m.groupdict().get("rank") or m.group(1)
            try:
                rank = int(rank_str)
            except ValueError:
                continue
            ranked.append(_RankedTraceFile(rank=rank, path=p))

        if not ranked:
            raise ValueError(
                f"--trace_glob matched {len(matches)} files, but none matched --rank_regex={rank_regex}"
            )

        # Keep lowest path for any duplicate rank
        by_rank: Dict[int, str] = {}
        for rt in sorted(ranked, key=lambda x: (x.rank, x.path)):
            by_rank.setdefault(rt.rank, rt.path)

        expected = list(range(world_size))
        missing = [r for r in expected if r not in by_rank]
        if missing:
            msg = (
                f"Missing ranks in trace_glob selection: {missing}. "
                "TraceLens NcclAnalyser assumes rank id == index in the provided trace-file list, "
                "so partial rank sets will produce incorrect results."
            )
            if strict_world_size_check:
                raise FileNotFoundError(msg)
            raise ValueError(msg)

        return [by_rank[r] for r in expected if r in by_rank]

    if trace_pattern:
        if trace_pattern.count("*") != 1:
            raise ValueError(
                "trace_pattern must contain exactly one '*' character as a placeholder for rank id"
            )
        list_trace_filepaths: List[str] = []
        for i in range(world_size):
            expected_file = trace_pattern.replace("*", str(i))
            if not os.path.isfile(expected_file):
                msg = (
                    f"Expected trace file not found: {expected_file}. "
                    "TraceLens NcclAnalyser requires a complete rank set for correct results."
                )
                if strict_world_size_check:
                    raise FileNotFoundError(msg)
                raise ValueError(msg)
            list_trace_filepaths.append(expected_file)
        return list_trace_filepaths

    # trace_dir
    assert trace_dir is not None
    list_trace_filepaths = []
    for i in range(world_size):
        expected_file = os.path.join(trace_dir, f"rank{i}_trace.json")
        if not os.path.isfile(expected_file):
            msg = (
                f"Expected trace file not found: {expected_file}. "
                "TraceLens NcclAnalyser requires a complete rank set for correct results."
            )
            if strict_world_size_check:
                raise FileNotFoundError(msg)
            raise ValueError(msg)
        list_trace_filepaths.append(expected_file)
    return list_trace_filepaths


def _parse_pg_ranks(v: Union[str, List[int], tuple]) -> List[int]:
    if isinstance(v, list):
        return [int(x) for x in v]
    if isinstance(v, tuple):
        return [int(x) for x in v]
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, (list, tuple)):
                return [int(x) for x in parsed]
        except Exception:
            pass
        # Fall back to extracting all integers from the string (common for "0,1,2,3" variants)
        ranks = []
        for s in re.findall(r"\d+", v):
            r = _try_int(s)
            if r is not None:
                ranks.append(r)
        return ranks
    return []


def _add_node_span_columns(df: pd.DataFrame, gpus_per_node: int) -> pd.DataFrame:
    """
    Adds:
      - rank_node: inferred node id for the row's rank (rank // gpus_per_node)
      - nodes_involved: number of nodes participating in the row's process group
      - node_span: intra_node vs inter_node based on Process Group Ranks membership
    """
    if df is None or df.empty:
        return df

    rank_to_node: Dict[int, int] = {}
    for r in df["rank"].unique():
        if pd.isna(r):
            continue
        rr = _try_int(r)
        if rr is None:
            continue
        rank_to_node[rr] = rr // int(gpus_per_node)
    df = df.copy()
    df["rank_node"] = df["rank"].map(rank_to_node)

    def _nodes_set(pg_ranks_val) -> set:
        ranks = _parse_pg_ranks(pg_ranks_val)
        if not ranks:
            return set()
        return {rank_to_node.get(int(r), int(r) // int(gpus_per_node)) for r in ranks}

    def _nodes_involved(pg_ranks_val) -> int:
        nodes = _nodes_set(pg_ranks_val)
        return int(len(nodes)) if nodes else 0

    def _node_span(pg_ranks_val) -> str:
        nodes = _nodes_set(pg_ranks_val)
        if not nodes:
            return "unknown"
        return "intra_node" if len(nodes) <= 1 else "inter_node"

    if "Process Group Ranks" in df.columns:
        df["nodes_involved"] = df["Process Group Ranks"].apply(_nodes_involved)
        df["node_span"] = df["Process Group Ranks"].apply(_node_span)
    else:
        df["nodes_involved"] = 0
        df["node_span"] = "unknown"
    return df


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
    rank_regex: str = r"rank[\[\-_/]?(?P<rank>\d+)",
    gpus_per_node: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate comprehensive NCCL communication analysis reports.

    Args:
        trace_dir: Directory containing trace files. The files must match the pattern "rank*_trace.json".
        trace_pattern: Template path with a single `*` placeholder for rank.
                        Example: /path/to/trace_rank_*_step_3.json
                        The `*` will be replaced with 0..world_size-1.
        world_size: Number of ranks
        output_xlsx_path: Path to output Excel file
        output_csvs_dir: Directory to save CSV files
        detailed_analysis: Whether to include detailed per-rank information
        agg_metrics: Aggregation metrics to include in summary
        use_multiprocessing: Whether to use multiprocessing for parallel trace loading (default: False).
                            When enabled, can provide significant speedup (system-dependent). When False, uses sequential loading.
        max_workers: Maximum number of worker processes for parallel loading (only used if use_multiprocessing=True).
                    Default: os.cpu_count(). Override to limit resource usage if needed.

    Returns:
        Dictionary mapping sheet names to DataFrames
    """
    if world_size is None:
        raise ValueError("world_size must be provided")

    if not trace_dir and not trace_pattern and not trace_glob:
        raise ValueError(
            "One of trace_dir, trace_pattern, or trace_glob must be provided"
        )

    t0 = _print_stage_start("Resolving trace file paths")
    list_trace_filepaths = _select_trace_files(
        trace_dir=trace_dir,
        trace_pattern=trace_pattern,
        trace_glob=trace_glob,
        world_size=world_size,
        strict_world_size_check=strict_world_size_check,
        rank_regex=rank_regex,
    )
    _print_stage_done(f"Resolved {len(list_trace_filepaths)} trace files", t0)

    # Minimal visibility into rank->file mapping without spamming for large world sizes.
    if len(list_trace_filepaths) <= 16:
        for r, p in enumerate(list_trace_filepaths):
            print(f"[TraceLens] rank {r}: {p}", flush=True)
    else:
        for r, p in list(enumerate(list_trace_filepaths))[:8]:
            print(f"[TraceLens] rank {r}: {p}", flush=True)
        print(f"[TraceLens] ... ({len(list_trace_filepaths) - 16} ranks omitted) ...")
        for r, p in list(enumerate(list_trace_filepaths))[-8:]:
            print(f"[TraceLens] rank {r}: {p}", flush=True)

    # Initialize NCCL analyzer
    t0 = _print_stage_start("Loading traces (NcclAnalyser)")
    nccl_analyser = NcclAnalyser(
        list_trace_filepaths,
        world_size,
        use_multiprocessing=use_multiprocessing,
        max_workers=max_workers,
    )
    _print_stage_done("Loaded traces", t0)

    # Generate DataFrames
    report_dfs = {}

    # Add summary dataframes
    print("Generating summary reports...", flush=True)
    t0 = _print_stage_start("Building nccl_summary_implicit_sync")
    report_dfs["nccl_summary_implicit_sync"] = (
        nccl_analyser.build_df_summary_nccl_implicit_sync_cat(agg_metrics=agg_metrics)
    )
    _print_stage_done("Built nccl_summary_implicit_sync", t0)

    t0 = _print_stage_start("Building nccl_summary_long")
    report_dfs["nccl_summary_long"] = nccl_analyser.build_df_summary_long()
    _print_stage_done("Built nccl_summary_long", t0)

    # Optional: also produce summaries split by intra-node vs inter-node (inferred from rank id mapping)
    if gpus_per_node is not None:
        t0 = _print_stage_start(
            "Building node-span summaries (intra_node vs inter_node)"
        )
        df_long = nccl_analyser.build_df_long()
        df_long = _add_node_span_columns(df_long, gpus_per_node=int(gpus_per_node))

        # Use NcclAnalyser's existing summary builder to avoid drift, but with extra grouping keys.
        original_df_long = getattr(nccl_analyser, "df_per_rank_coll", None)
        try:
            nccl_analyser.df_per_rank_coll = df_long
            group_by_cols = [
                "rank",
                "rank_node",
                "nodes_involved",
                "node_span",
                "Process Group Name",
                "Process Group Ranks",
                "Collective name",
                "Group size",
                "dtype",
                "In msg nelems",
                "Out msg nelems",
                "In split size",
                "Out split size",
                "stream",
            ]
            report_dfs["nccl_summary_long_node_span"] = (
                nccl_analyser.build_df_summary_long(group_by_cols=group_by_cols)
            )
        finally:
            if original_df_long is not None:
                nccl_analyser.df_per_rank_coll = original_df_long

        # Implicit sync summary split by node_span (grouping matches NcclAnalyser with an extra node_span)
        df_implicit = nccl_analyser.build_df_nccl_implicit_sync_cat(detailed=False)
        if df_implicit is not None and not df_implicit.empty:
            df_implicit = df_implicit.copy()
            # node_span from Process Group Ranks; rank_node doesn't exist in this wide df
            rank_to_node = {r: int(r) // int(gpus_per_node) for r in range(world_size)}

            def _nodes_set_implicit(pg_ranks_val) -> set:
                ranks = _parse_pg_ranks(pg_ranks_val)
                if not ranks:
                    return set()
                return {
                    rank_to_node.get(int(r), int(r) // int(gpus_per_node))
                    for r in ranks
                }

            def _nodes_involved_implicit(pg_ranks_val) -> int:
                nodes = _nodes_set_implicit(pg_ranks_val)
                return int(len(nodes)) if nodes else 0

            def _node_span_implicit(pg_ranks_val) -> str:
                nodes = _nodes_set_implicit(pg_ranks_val)
                if not nodes:
                    return "unknown"
                return "intra_node" if len(nodes) <= 1 else "inter_node"

            df_implicit["nodes_involved"] = df_implicit["Process Group Ranks"].apply(
                _nodes_involved_implicit
            )
            df_implicit["node_span"] = df_implicit["Process Group Ranks"].apply(
                _node_span_implicit
            )

            metadata_fields = ["Process Group Name", "Group size", "Full msg size (MB)"]
            agg_logic = {
                "comm_latency": agg_metrics + ["size", lambda x: x.sum() / 1000],
                "skew in start time": agg_metrics,
                "skew in end time": agg_metrics,
                "algo bw (GB/s)": agg_metrics,
                "bus bw (GB/s)": agg_metrics,
            }
            metric_fields = list(agg_logic.keys()).copy()
            for col in metadata_fields:
                agg_logic[col] = "first"

            groupby_cols = [
                "nodes_involved",
                "node_span",
                "Collective name",
                "dtype",
                "In msg nelems",
            ]
            agg_result = df_implicit.groupby(groupby_cols).agg(agg_logic)
            agg_result.columns = [
                f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
                for col in agg_result.columns
            ]
            column_renames = {
                "comm_latency_<lambda_0>": "Total comm latency (ms)",
                "comm_latency_size": "count",
            }
            for col in metadata_fields:
                column_renames[col + "_first"] = col
            agg_result.rename(columns=column_renames, inplace=True)
            summary_df = agg_result.reset_index().sort_values(
                by="Total comm latency (ms)", ascending=False
            )
            columns_order = groupby_cols + metadata_fields
            for group in metric_fields:
                for agg in agg_metrics:
                    columns_order.append(f"{group}_{agg}")
            columns_order.extend(["count", "Total comm latency (ms)"])
            report_dfs["nccl_summary_implicit_node_span"] = summary_df[columns_order]
        _print_stage_done("Built node-span summaries", t0)

    # Add detailed per-rank information if requested
    if detailed_analysis:
        print("Generating detailed per-rank analysis...")
        t0 = _print_stage_start(
            "Building detailed sheets (nccl_long / nccl_implicit_sync / nccl_all2allv)"
        )
        report_dfs["nccl_long"] = nccl_analyser.build_df_long()
        report_dfs["nccl_implicit_sync"] = (
            nccl_analyser.build_df_nccl_implicit_sync_cat(detailed=True)
        )
        df_all2allv = nccl_analyser.build_df_nccl_all2allv(detailed=True)
        if df_all2allv is not None and not df_all2allv.empty:
            report_dfs["nccl_all2allv"] = df_all2allv
        _print_stage_done("Built detailed sheets", t0)

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
        help="Glob for trace files with arbitrary names (supports **). Requires --world_size and uses --rank_regex to map files to ranks.",
    )

    parser.add_argument(
        "--world_size", type=int, default=None, help="Number of ranks (required)"
    )
    parser.add_argument(
        "--rank_regex",
        type=str,
        default=r"rank[\[\-_/]?(?P<rank>\d+)",
        help="Regex used with --trace_glob to extract rank id. Must contain a named group 'rank' or a single capture group.",
    )
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=None,
        help="If provided, adds node-aware intra_node vs inter_node breakdown sheets by inferring node_id = rank // gpus_per_node.",
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
        else:
            trace_files = _select_trace_files(
                trace_dir=None,
                trace_pattern=args.trace_pattern,
                trace_glob=args.trace_glob,
                world_size=args.world_size,
                strict_world_size_check=True,
                rank_regex=args.rank_regex,
            )
            common = os.path.commonpath([os.path.abspath(p) for p in trace_files])
            output_dir = os.path.dirname(common) if os.path.isfile(common) else common

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
        gpus_per_node=args.gpus_per_node,
    )


if __name__ == "__main__":
    main()
