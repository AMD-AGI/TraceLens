#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
CPU/Idle Time Analysis Script

Reports GPU idle time percentage and utilization breakdown.
Flags idle time above IDLE_PCT_THRESHOLD for the agent to investigate.

Outputs cpu_idle_metrics.json with analysis results.
"""

import argparse
import os
from typing import Any

import pandas as pd

from TraceLens.Agent.Analysis.category_analyses.analysis_utils import (
    IDLE_PCT_THRESHOLD,
    SystemFindings,
    load_manifest,
    perf_report_csv_dir,
    write_metrics_json,
)
from TraceLens.Reporting.aggregate_perf_reports import NODE_COL

# Constants
SHORT_KERNEL_US = 10


def load_gpu_timeline(output_dir: str) -> dict[str, dict[str, float]]:
    """Load GPU timeline data from CSV."""
    csv_path = os.path.join(perf_report_csv_dir(output_dir), "gpu_timeline.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"GPU timeline not found: {csv_path}")

    df = pd.read_csv(csv_path)

    timeline = {}
    for _, row in df.iterrows():
        timeline[row["type"]] = {"time_ms": row["time ms"], "percent": row["percent"]}

    return timeline


def load_ops_summary(output_dir: str) -> pd.DataFrame | None:
    """Load operations summary for kernel analysis."""
    csv_path = os.path.join(perf_report_csv_dir(output_dir), "ops_summary.csv")

    if not os.path.exists(csv_path):
        return None

    return pd.read_csv(csv_path)


def analyze_kernel_patterns(ops_df: pd.DataFrame | None) -> dict[str, Any]:
    """Analyze kernel patterns to identify overhead sources."""
    patterns = {
        "short_kernel_count": 0,
        "total_kernel_count": 0,
        "avg_kernel_time_us": 0,
        "kernel_count_by_category": {},
    }

    if ops_df is None or ops_df.empty:
        return patterns

    if "Kernel Time (µs)_sum" in ops_df.columns and "Count" in ops_df.columns:
        patterns["total_kernel_count"] = int(ops_df["Count"].sum())

        total_time = ops_df["Kernel Time (µs)_sum"].sum()
        if patterns["total_kernel_count"] > 0:
            patterns["avg_kernel_time_us"] = total_time / patterns["total_kernel_count"]

        if "Kernel Time (µs)_mean" in ops_df.columns:
            short_ops = ops_df[ops_df["Kernel Time (µs)_mean"] < SHORT_KERNEL_US]
            patterns["short_kernel_count"] = (
                int(short_ops["Count"].sum()) if not short_ops.empty else 0
            )

    if "op category" in ops_df.columns and "Count" in ops_df.columns:
        category_counts = ops_df.groupby("op category")["Count"].sum().to_dict()
        patterns["kernel_count_by_category"] = {
            k: int(v) for k, v in category_counts.items()
        }

    return patterns


def main():
    parser = argparse.ArgumentParser(description="CPU/Idle Time Analysis")
    parser.add_argument("--output-dir", required=True, help="Analysis output directory")
    args = parser.parse_args()

    output_dir = args.output_dir

    print("=" * 80)
    print("CPU/IDLE TIME ANALYSIS")
    print("=" * 80)

    try:
        gpu_timeline = load_gpu_timeline(output_dir)
        ops_df = load_ops_summary(output_dir)

        idle_time = gpu_timeline.get("idle_time", {})
        idle_percent = idle_time.get("percent", 0)
        idle_ms = idle_time.get("time_ms", 0)
        total_time_ms = gpu_timeline.get("total_time", {}).get("time_ms", 0)

        computation_percent = gpu_timeline.get("computation_time", {}).get("percent", 0)
        comm_percent = gpu_timeline.get("exposed_comm_time", {}).get("percent", 0)
        memcpy_percent = gpu_timeline.get("exposed_memcpy_time", {}).get("percent", 0)

        print("\nGPU Utilization Breakdown:")
        print(f"  Total Time: {total_time_ms:.2f} ms")
        print(f"  Computation: {computation_percent:.2f}%")
        print(f"  Communication: {comm_percent:.2f}%")
        print(f"  MemCpy: {memcpy_percent:.2f}%")
        print(f"  Idle: {idle_percent:.2f}%")

        kernel_patterns = analyze_kernel_patterns(ops_df)
        print("\nKernel Analysis:")
        print(f"  Total Kernels: {kernel_patterns['total_kernel_count']}")
        print(
            f"  Short Kernels (<{SHORT_KERNEL_US}µs): {kernel_patterns['short_kernel_count']}"
        )
        print(f"  Avg Kernel Time: {kernel_patterns['avg_kernel_time_us']:.1f} µs")

        gpu_utilization = {
            "total_time_ms": round(total_time_ms, 2),
            "idle_time_ms": round(idle_ms, 2),
            "idle_time_percent": round(idle_percent, 2),
            "computation_percent": round(computation_percent, 2),
            "communication_percent": round(comm_percent, 2),
            "memcpy_percent": round(memcpy_percent, 2),
        }
        system_findings = SystemFindings.rank(
            [
                SystemFindings.make(
                    "High GPU Idle Time",
                    gpu_utilization["idle_time_ms"],
                    gpu_utilization["idle_time_percent"],
                )
            ]
            if idle_percent > IDLE_PCT_THRESHOLD
            else []
        )
        print(f"\n  System findings: {len(system_findings)} (idle {idle_percent:.1f}%)")

        metrics = {
            "status": "OK",
            NODE_COL: load_manifest(output_dir).get("focus_job"),
            "system_findings": system_findings,
            "gpu_utilization": gpu_utilization,
            "kernel_analysis": kernel_patterns,
            "impact_estimates": [],
        }

        metrics_path = write_metrics_json(metrics, output_dir, "cpu_idle")

        print(f"\n✓ Metrics saved: {metrics_path}")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error: {e}")

        error_metrics = {"status": "ERROR", "error": str(e)}
        write_metrics_json(error_metrics, output_dir, "cpu_idle")

        raise


if __name__ == "__main__":
    main()
