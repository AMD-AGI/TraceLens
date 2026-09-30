###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Standalone CLI for GPU idle time classification of PyTorch profiler traces.

Classifies each GPU idle interval (noise/macro, drain_type, cpu_during_gap),
writes an Excel report, and emits an augmented trace with annotation tracks
for Perfetto.

Usage:
    python -m TraceLens.Reporting.classify_idle_time <trace.json[.gz]> [--micro-thresh 5.0] [-o output.json.gz]

To add the same sheets to the main perf report, use
``TraceLens_generate_perf_report_pytorch --enable_idle_analysis`` instead.
"""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from TraceLens import DataLoader, TreePerfAnalyzer
from TraceLens.IdleTimeAnalyser.classify import (
    assign_idle_ids,
    classify_idle_intervals,
    find_gpu_pid,
    make_annotation_events,
)
from TraceLens.IdleTimeAnalyser.report import build_idle_dataframes, write_idle_excel


def generate_excel_report(classified, output_path):
    """Write idle_overview, idle_summary, and idle_intervals sheets to Excel."""
    write_idle_excel(build_idle_dataframes(classified), output_path)


def _default_output_path(trace_path):
    p = Path(trace_path)
    # Strip all suffixes like .pt.trace.json.gz
    stem = p.name
    for _ in range(4):
        if "." in stem:
            stem = stem.rsplit(".", 1)[0]
    return str(p.parent / f"{stem}_idle_classified.json.gz")


def _print_summary(classified, micro_thresh):
    noise_count = sum(1 for r in classified if r["label_noise"])
    macro = [r for r in classified if not r["label_noise"]]
    macro_total_us = sum(r["duration"] for r in macro)

    print(f"\n{'=' * 60}")
    print("Idle Interval Summary")
    print(f"{'=' * 60}")
    print(f"Total intervals:    {len(classified)}")
    print(f"  Noise (<{micro_thresh}µs):  {noise_count}")
    print(
        f"  Macro (≥{micro_thresh}µs):  {len(macro)}  ({macro_total_us / 1e3:.2f} ms total)"
    )

    if not macro:
        return

    sync_drain = [r for r in macro if r["drain_type"] == "sync_drain"]
    starved = [r for r in macro if r["drain_type"] == "starved"]
    print("\nDrain Type:")
    print(
        f"  sync_drain:       {len(sync_drain)}  ({sum(r['duration'] for r in sync_drain) / 1e3:.2f} ms)"
    )
    print(
        f"  starved:          {len(starved)}  ({sum(r['duration'] for r in starved) / 1e3:.2f} ms)"
    )
    for st, cnt in Counter(r["sync_type"] for r in sync_drain).most_common():
        st_time = sum(r["duration"] for r in sync_drain if r["sync_type"] == st)
        print(f"    {st}: {cnt} intervals, {st_time / 1e3:.2f} ms")

    print("\nCPU During Gap:")
    for cpu_gap in [
        "LAUNCH_ANOMALY",
        "LAUNCH_OVERHEAD_ONLY",
        "RUNTIME_DOMINATED",
        "CPU_DOMINATED",
        "CPU_UNTRACED",
    ]:
        items = [r for r in macro if r["cpu_during_gap"] == cpu_gap]
        if not items:
            continue
        total_t = sum(r["duration"] for r in items)
        print(f"  {cpu_gap}: {len(items)} intervals, {total_t / 1e3:.2f} ms")
        if cpu_gap == "RUNTIME_DOMINATED":
            for detail, cnt in Counter(
                r["cpu_during_gap_detail"] for r in items
            ).most_common():
                dt = sum(
                    r["duration"] for r in items if r["cpu_during_gap_detail"] == detail
                )
                print(f"    {detail}: {cnt} intervals, {dt / 1e3:.2f} ms")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Classify GPU idle time intervals in a PyTorch trace"
    )
    parser.add_argument("trace", help="Path to PyTorch trace JSON (or .json.gz)")
    parser.add_argument(
        "--micro-thresh",
        type=float,
        default=5.0,
        help="Micro idle threshold in µs (default: 5.0)",
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
    tree = TreePerfAnalyzer.from_file(trace_path).tree

    print("Classifying idle intervals...")
    classified = classify_idle_intervals(tree, micro_thresh_us=args.micro_thresh)
    _print_summary(classified, args.micro_thresh)

    assign_idle_ids(classified)

    excel_path = output_path.replace(".json.gz", ".xlsx").replace(".json", ".xlsx")
    if excel_path == output_path:
        excel_path = output_path + ".xlsx"
    generate_excel_report(classified, excel_path)

    print("\nLoading raw trace for augmentation...")
    raw_data = DataLoader.load_data(trace_path)
    gpu_pid = find_gpu_pid(raw_data["traceEvents"])
    print(f"GPU pid: {gpu_pid}")

    annotation_events = make_annotation_events(classified, gpu_pid)
    raw_data["traceEvents"].extend(annotation_events)

    print(f"Added {len(annotation_events)} annotation events to trace.")
    print(f"Writing augmented trace to: {output_path}")
    with gzip.open(output_path, "wt") as f:
        json.dump(raw_data, f)

    print("Done.")


if __name__ == "__main__":
    main()
