#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Elementwise Operations Analysis

Computes metrics for elementwise operations and outputs JSON for subagent processing.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import (
    load_category_data,
    calculate_time_metrics,
    build_operation_metrics,
    calculate_average_efficiency,
    compute_impact_estimates,
    write_metrics_json,
)


def get_elementwise_config():
    """Return elementwise-specific configuration."""
    return {
        "efficiency_method": "memory_bound",  # Elementwise ops are memory-bound
        "extra_fields": ["FLOPS/Byte"],
    }


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract elementwise-specific aggregate metrics."""
    return {"peak_hbm_bw_tbs": metadata.get("peak_hbm_bw_tbs")}


def main():
    parser = argparse.ArgumentParser(description="Analyze elementwise operations")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    try:
        ops_df, metadata = load_category_data(args.output_dir, "elementwise")
    except FileNotFoundError as e:
        error_metrics = {"category": "elementwise", "status": "ERROR", "error": str(e)}
        write_metrics_json(error_metrics, args.output_dir, "elementwise")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    config = get_elementwise_config()
    peak_hbm_bw = metadata.get("peak_hbm_bw_tbs", 1)
    maf = metadata.get("max_achievable_tflops", metadata.get("peak_bf16_maf_tflops", 1))

    time_metrics = calculate_time_metrics(ops_df, metadata)
    avg_efficiency = calculate_average_efficiency(
        ops_df, peak_hbm_bw, maf, "memory_bound"
    )
    operations = build_operation_metrics(ops_df, metadata, config)
    category_specific = extract_category_specific(ops_df, metadata)

    baseline_ms = metadata.get("gpu_utilization", {}).get("total_time_ms", 0)
    impact_estimates = compute_impact_estimates(operations, "elementwise", baseline_ms=baseline_ms)

    metrics = {
        "category": "elementwise",
        "status": "OK",
        **time_metrics,
        "average_efficiency_percent": avg_efficiency,
        "operations": operations,
        "category_specific": category_specific,
        "impact_estimates": impact_estimates,
    }

    output_path = write_metrics_json(metrics, args.output_dir, "elementwise")
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
