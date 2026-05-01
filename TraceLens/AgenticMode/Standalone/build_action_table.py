#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Standalone script to build the Kernel Tuning Action Table.

Loads a trace, extracts call stacks for the top GPU kernels, resolves each
frame to a container path and GitHub URL, and writes
kernel_tuning_action_table.json.

Run inside the container where the trace was collected so that pip packages
and editable installs are visible.

Usage:
    python build_action_table.py --trace-path /path/to/profile_rank_0.json \\
                                 --csv-dir /path/to/perf_report_csvs \\
                                 --output kernel_tuning_action_table.json \\
                                 --platform MI300X \\
                                 --top-n 30 \\
                                 --verbose
"""

import argparse
import json
import os
import sys
import traceback

import pandas as pd
from TraceLens.TreePerf import TreePerfAnalyzer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.kernel_agent_interface_utils import (
    SourceResolver,
    extract_gpu_kernel_names,
    parse_call_stack_frames,
    strip_template_args,
)
from utils.arch_utils import list_platforms, load_arch


def load_tree(trace_path: str, platform: str, enable_pseudo_ops: bool = True):
    """Load the trace and return the tree object."""
    specs = load_arch(platform)

    print(f"Loading trace: {trace_path}")
    print(f"  Platform: {specs['name']} (mem_bw={specs['mem_bw_gbps']} GB/s)")
    print(f"  Pseudo ops: {'enabled' if enable_pseudo_ops else 'disabled'}")

    analyzer = TreePerfAnalyzer.from_file(
        trace_path,
        add_python_func=True,
        enable_pseudo_ops=enable_pseudo_ops,
        arch=specs,
    )
    tree = analyzer.tree
    print(f"  Tree loaded: {len(tree.events)} events")
    return tree


def collect_bottleneck_rows(unified_df: pd.DataFrame, top_n: int):
    """Return the top-N ops by kernel time, deduped by UID."""
    if "Kernel Time (µs)_sum" not in unified_df.columns:
        print("  WARNING: 'Kernel Time (µs)_sum' column missing, using all rows")
        return list(unified_df.head(top_n).itertuples(index=False))

    sorted_df = unified_df.sort_values("Kernel Time (µs)_sum", ascending=False)
    rows = []
    seen_uids: set[int] = set()
    for _, row in sorted_df.iterrows():
        uid = row.get("ex_UID", row.get("UID", None))
        if pd.isna(uid):
            continue
        uid_int = int(uid)
        if uid_int in seen_uids:
            continue
        seen_uids.add(uid_int)
        rows.append(row)
        if len(rows) >= top_n:
            break
    return rows


def build_action_table(
    tree,
    unified_df: pd.DataFrame,
    top_n: int = 30,
    verbose: bool = False,
):
    """Build the action table entries and return them as a list of dicts."""

    rows = collect_bottleneck_rows(unified_df, top_n)
    print(f"\nCollected {len(rows)} bottleneck operations")

    # --- Phase 1: extract call stacks from tree ---
    print("\nExtracting call stacks...")
    raw_entries = []
    for row in rows:
        uid = int(row.get("ex_UID", row.get("UID", 0)))
        call_stack = ""
        nn_module_stack = ""
        try:
            event = tree.get_UID2event(uid)
            call_stack = tree.traverse_parents_and_get_callstack(event, filter=None)
            nn_module_stack = tree.traverse_parents_and_get_callstack(
                event, filter=("nn.Module",)
            )
            breakpoint()
        except (KeyError, Exception) as e:
            if verbose:
                print(f"  UID {uid}: call stack extraction failed: {e}")

        raw_entries.append((row, call_stack, nn_module_stack))

    # --- Phase 2: build resolver from call stacks ---
    print("Building source resolver...")
    call_stacks = [cs for _, cs, _ in raw_entries]
    resolver = SourceResolver.from_call_stacks(call_stacks)

    print(f"  Indexed {len(resolver.packages)} packages:")
    for pkg in resolver.packages:
        loc = pkg.editable_location or pkg.location
        repo = pkg.github_repo or "(no repo)"
        print(f"    {pkg.name}: {loc}  ->  {repo}  @ {pkg.git_hash or '?'}")

    # --- Phase 3: resolve sources and build entries ---
    print("\nResolving source locations...")
    entries = []
    for row, call_stack, nn_module_stack in raw_entries:
        op_name = row.get("name", "Unknown")
        input_dims = str(row.get("Input Dims", ""))
        category = row.get("op category", "")
        kernel_time_us = row.get("Kernel Time (µs)_sum", 0)
        efficiency = row.get("Pct Roofline", None)

        trunc_details = str(row.get("trunc_kernel_details", ""))
        gpu_kernel_names = extract_gpu_kernel_names(trunc_details)
        primary_kernel = gpu_kernel_names[0] if gpu_kernel_names else None

        frames = parse_call_stack_frames(call_stack)
        sources = resolver.resolve_frames(frames)

        has_perf_model = (
            bool(row.get("has_perf_model", False))
            if "has_perf_model" in row.index
            else False
        )

        entry = {
            "operation": op_name,
            "input_dims": input_dims,
            "category": category,
            "call_stack": call_stack,
            "nn_module_stack": nn_module_stack,
            "gpu_kernel_name": (
                strip_template_args(primary_kernel) if primary_kernel else None
            ),
            "gpu_kernel_name_full": primary_kernel,
            "container_path": sources["container_path"],
            "github_url": sources["github_url"],
            "launcher_display": sources["launcher_display"],
            "time_ms": (round(kernel_time_us / 1000, 3) if kernel_time_us else None),
            "efficiency_pct": (
                round(float(efficiency), 2)
                if efficiency and not pd.isna(efficiency)
                else None
            ),
            "has_perf_model": has_perf_model,
        }
        entries.append(entry)

    return entries


def print_table(entries, verbose: bool = False):
    """Pretty-print the action table to stdout."""
    resolved = sum(1 for e in entries if e["github_url"])
    container_only = sum(
        1 for e in entries if e["container_path"] and not e["github_url"]
    )
    unresolved = len(entries) - resolved - container_only

    print(f"\n{'='*100}")
    print(f"KERNEL TUNING ACTION TABLE  ({len(entries)} kernels)")
    print(f"  Resolved to GitHub URL: {resolved}")
    print(f"  Container path only:    {container_only}")
    print(f"  Unresolved:             {unresolved}")
    print(f"{'='*100}\n")

    for i, e in enumerate(entries, 1):
        time_str = f"{e['time_ms']:.2f} ms" if e["time_ms"] else "?"
        eff_str = f"{e['efficiency_pct']:.0f}%" if e["efficiency_pct"] else "?"
        kernel = e["gpu_kernel_name"] or "(no kernel)"

        print(f"[{i:2d}] {e['operation']}")
        print(f"     Category:   {e['category']}")
        print(f"     GPU kernel: {kernel}")
        print(f"     Time:       {time_str}  |  Efficiency: {eff_str}")

        if e["launcher_display"]:
            print(f"     Launcher:   {e['launcher_display']}")
        if e["container_path"]:
            print(f"     Container:  {e['container_path']}")
        if e["github_url"]:
            print(f"     GitHub:     {e['github_url']}")

        if verbose:
            if e["nn_module_stack"]:
                print(f"     Modules:    {e['nn_module_stack']}")
            if e["call_stack"]:
                cs = e["call_stack"]
                if len(cs) > 200:
                    cs = cs[:200] + "..."
                print(f"     Call stack:  {cs}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Build Kernel Tuning Action Table from a trace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (reads CSV from the same output dir as orchestrator_prepare)
  python build_action_table.py \\
      --trace-path /data/traces/profile_rank_0.json \\
      --csv-dir /data/output/perf_report_csvs \\
      --platform MI300X

  # Save JSON and show verbose output
  python build_action_table.py \\
      --trace-path /data/traces/profile_rank_0.json \\
      --csv-dir /data/output/perf_report_csvs \\
      --platform MI300X \\
      --output action_table.json \\
      --top-n 20 \\
      --verbose
""",
    )
    parser.add_argument(
        "--trace-path",
        required=True,
        help="Path to the trace JSON file (e.g., profile_rank_0.json)",
    )
    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Directory containing unified_perf_summary.csv (the perf_report_csvs/ dir)",
    )
    parser.add_argument(
        "--platform",
        required=True,
        choices=list_platforms(),
        help="AMD platform",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON file path (default: print to stdout only)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top operations to include (default: 30)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show call stacks and detailed debug output",
    )
    parser.add_argument(
        "--disable-pseudo-ops",
        action="store_true",
        help="Disable pseudo-op augmentation when loading the trace",
    )

    args = parser.parse_args()

    csv_path = os.path.join(args.csv_dir, "unified_perf_summary.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.trace_path):
        print(f"ERROR: {args.trace_path} not found", file=sys.stderr)
        sys.exit(1)

    try:
        tree = load_tree(
            args.trace_path,
            args.platform,
            enable_pseudo_ops=not args.disable_pseudo_ops,
        )
    except Exception as e:
        print(f"ERROR loading trace: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    unified_df = pd.read_csv(csv_path)
    print(f"Loaded CSV: {len(unified_df)} rows from {csv_path}")

    entries = build_action_table(
        tree,
        unified_df,
        top_n=args.top_n,
        verbose=args.verbose,
    )

    print_table(entries, verbose=args.verbose)

    if args.output:
        action_table = {"kernels": entries}
        with open(args.output, "w") as f:
            json.dump(action_table, f, indent=2)
        print(f"Wrote {args.output} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
