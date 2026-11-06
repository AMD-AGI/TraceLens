###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Standalone CLI for GPU idle time classification of PyTorch profiler traces.

Splits each GPU idle gap into measured parts, labels it by the largest part,
writes an Excel report (idle_overview, idle_groups, idle_intervals), and emits
an augmented trace with annotation tracks for Perfetto.

Usage:
    python -m TraceLens.Reporting.classify_idle_time <trace.json[.gz]> [--micro-thresh 5.0]
        [--launch-latency 15.0] [--include-call-stack] [-o output.json.gz]

To add the same sheets to the main perf report, use
``TraceLens_generate_perf_report_pytorch --enable_idle_analysis`` instead.
"""

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from TraceLens import DataLoader, TreePerfAnalyzer
from TraceLens.IdleTimeAnalyser import IdleThresholds, IdleTimeAnalyser
from TraceLens.IdleTimeAnalyser.classify import find_gpu_pid
from TraceLens.IdleTimeAnalyser.report import build_idle_dataframes, write_idle_excel


def generate_excel_report(
    classified, output_path, gpu_busy_time_us=None, thresholds=None
):
    """Write idle_overview, idle_groups, and idle_intervals sheets to Excel."""
    write_idle_excel(
        build_idle_dataframes(classified, gpu_busy_time_us, thresholds), output_path
    )


def _default_output_path(trace_path):
    p = Path(trace_path)
    # Strip all suffixes like .pt.trace.json.gz
    stem = p.name
    for _ in range(4):
        if "." in stem:
            stem = stem.rsplit(".", 1)[0]
    return str(p.parent / f"{stem}_idle_classified.json.gz")


def _print_summary(df_overview):
    print(f"\n{'=' * 72}")
    print("GPU Idle Overview")
    print(f"{'=' * 72}")
    indent = {"total": "", "label": "  ", "sub_label": "    ", "drain": "  "}
    prev_section = "total"
    for r in df_overview.to_dict("records"):
        section = r["section"]
        if section == "threshold":
            continue
        if r["row"] == "gpu_utilization_pct":
            print(f"GPU utilization: {r['pct_of_trace']:.1f}%")
            continue
        if (section == "label" and prev_section == "total") or (
            section == "drain" and prev_section != "drain"
        ):
            print()
        prev_section = section
        gaps = "" if pd.isna(r["gaps"]) else f"{int(r['gaps'])} gaps"
        print(
            f"{indent[section]}{r['row']:<40} {gaps:>12} "
            f"{r['total_ms']:>10.2f} ms {r['pct_of_idle']:>6.1f}%"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Classify GPU idle time intervals in a PyTorch trace"
    )
    parser.add_argument("trace", help="Path to PyTorch trace JSON (or .json.gz)")
    parser.add_argument(
        "--micro-thresh",
        type=float,
        default=IdleThresholds.noise_us,
        help="Gaps shorter than this (us) count as noise (default: 5.0)",
    )
    parser.add_argument(
        "--launch-latency",
        type=float,
        default=IdleThresholds.launch_latency_us,
        help="Normal launch return -> GPU op start in us (default: 15.0)",
    )
    parser.add_argument(
        "--include-call-stack",
        "--include_call_stack",
        dest="include_call_stack",
        action="store_true",
        help="Add sync_entry_point, sync_call_stack_full and covering_call_stack_full "
        "to idle_intervals (builds the tree with Python functions, like the perf "
        "report flag)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path for augmented trace (default: <input>_idle_classified.json.gz)",
    )
    args = parser.parse_args(argv)

    trace_path = args.trace
    output_path = args.output or _default_output_path(trace_path)

    print(f"Loading trace: {trace_path}")
    tree = TreePerfAnalyzer.from_file(
        trace_path, add_python_func=args.include_call_stack
    ).tree

    print("Classifying idle intervals...")
    thresholds = IdleThresholds(
        noise_us=args.micro_thresh, launch_latency_us=args.launch_latency
    )
    analyser = IdleTimeAnalyser(
        tree, thresholds, include_call_stack=args.include_call_stack
    )
    dfs = analyser.get_dataframes()
    _print_summary(dfs["idle_overview"])

    excel_path = output_path.replace(".json.gz", ".xlsx").replace(".json", ".xlsx")
    if excel_path == output_path:
        excel_path = output_path + ".xlsx"
    write_idle_excel(dfs, excel_path)
    print(f"\nWrote Excel report: {excel_path}")

    print("Loading raw trace for augmentation...")
    raw_data = DataLoader.load_data(trace_path)
    gpu_pid = find_gpu_pid(raw_data["traceEvents"])
    print(f"GPU pid: {gpu_pid}")

    annotation_events = analyser.get_augmented_events(gpu_pid)
    raw_data["traceEvents"].extend(annotation_events)

    print(f"Added {len(annotation_events)} annotation events to trace.")
    print(f"Writing augmented trace to: {output_path}")
    with gzip.open(output_path, "wt") as f:
        json.dump(raw_data, f)

    print("Done.")


if __name__ == "__main__":
    main()
