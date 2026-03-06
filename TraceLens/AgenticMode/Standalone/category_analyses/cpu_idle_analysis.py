#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
CPU/Idle Time Analysis Script

Reports GPU idle time percentage and utilization breakdown.
Flags idle time exceeding 15% for the agent to investigate.

Outputs cpu_idle_metrics.json with analysis results.
"""

import argparse
import json
import os
import pandas as pd
from typing import Dict, Any, Optional


def load_gpu_timeline(output_dir: str) -> Dict[str, float]:
    """Load GPU timeline data from CSV."""
    csv_path = f"{output_dir}/perf_report_csvs/gpu_timeline.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"GPU timeline not found: {csv_path}")

    df = pd.read_csv(csv_path)

    timeline = {}
    for _, row in df.iterrows():
        timeline[row["type"]] = {"time_ms": row["time ms"], "percent": row["percent"]}

    return timeline


def load_ops_summary(output_dir: str) -> Optional[pd.DataFrame]:
    """Load operations summary for kernel analysis."""
    csv_path = f"{output_dir}/perf_report_csvs/ops_summary.csv"

    if not os.path.exists(csv_path):
        return None

    return pd.read_csv(csv_path)


def load_manifest(output_dir: str) -> Dict:
    """Load category manifest for metadata."""
    manifest_path = f"{output_dir}/category_data/category_manifest.json"

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)

    return {}


def analyze_kernel_patterns(ops_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Analyze kernel patterns to identify overhead sources."""
    patterns = {
        "short_kernel_count": 0,
        "total_kernel_count": 0,
        "avg_kernel_time_us": 0,
        "kernel_count_by_category": {},
    }

    if ops_df is None or ops_df.empty:
        return patterns

    # Count kernels and analyze times
    if "Kernel Time (µs)_sum" in ops_df.columns and "Count" in ops_df.columns:
        patterns["total_kernel_count"] = int(ops_df["Count"].sum())

        total_time = ops_df["Kernel Time (µs)_sum"].sum()
        if patterns["total_kernel_count"] > 0:
            patterns["avg_kernel_time_us"] = total_time / patterns["total_kernel_count"]

        # Count short kernels (< 10µs average)
        if "Kernel Time (µs)_mean" in ops_df.columns:
            short_ops = ops_df[ops_df["Kernel Time (µs)_mean"] < 10]
            patterns["short_kernel_count"] = (
                int(short_ops["Count"].sum()) if not short_ops.empty else 0
            )

    # Analyze by category
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
        # Load data
        gpu_timeline = load_gpu_timeline(output_dir)
        ops_df = load_ops_summary(output_dir)

        # Extract key metrics
        idle_time = gpu_timeline.get("idle_time", {})
        idle_percent = idle_time.get("percent", 0)
        idle_ms = idle_time.get("time_ms", 0)
        total_time_ms = gpu_timeline.get("total_time", {}).get("time_ms", 0)

        computation_percent = gpu_timeline.get("computation_time", {}).get("percent", 0)
        comm_percent = gpu_timeline.get("exposed_comm_time", {}).get("percent", 0)
        memcpy_percent = gpu_timeline.get("exposed_memcpy_time", {}).get("percent", 0)

        print(f"\nGPU Utilization Breakdown:")
        print(f"  Total Time: {total_time_ms:.2f} ms")
        print(f"  Computation: {computation_percent:.2f}%")
        print(f"  Communication: {comm_percent:.2f}%")
        print(f"  MemCpy: {memcpy_percent:.2f}%")
        print(f"  Idle: {idle_percent:.2f}%")

        # Analyze kernel patterns
        kernel_patterns = analyze_kernel_patterns(ops_df)
        print(f"\nKernel Analysis:")
        print(f"  Total Kernels: {kernel_patterns['total_kernel_count']}")
        print(f"  Short Kernels (<10µs): {kernel_patterns['short_kernel_count']}")
        print(f"  Avg Kernel Time: {kernel_patterns['avg_kernel_time_us']:.1f} µs")

        idle_flagged = idle_percent > 15
        print(f"\n  Idle Flagged: {idle_flagged} ({idle_percent:.1f}%)")

        metrics = {
            "status": "OK",
            "idle_flagged": idle_flagged,
            "gpu_utilization": {
                "total_time_ms": round(total_time_ms, 3),
                "idle_time_ms": round(idle_ms, 3),
                "idle_time_percent": round(idle_percent, 2),
                "computation_percent": round(computation_percent, 2),
                "communication_percent": round(comm_percent, 2),
                "memcpy_percent": round(memcpy_percent, 2),
            },
            "kernel_analysis": kernel_patterns,
            "impact_estimates": [],
        }

        # Write metrics JSON
        os.makedirs(f"{output_dir}/category_data", exist_ok=True)
        metrics_path = f"{output_dir}/category_data/cpu_idle_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✓ Metrics saved: {metrics_path}")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")

        # Write error metrics
        error_metrics = {"status": "ERROR", "error": str(e)}

        os.makedirs(f"{output_dir}/category_data", exist_ok=True)
        with open(f"{output_dir}/category_data/cpu_idle_metrics.json", "w") as f:
            json.dump(error_metrics, f, indent=2)

        raise


if __name__ == "__main__":
    main()
