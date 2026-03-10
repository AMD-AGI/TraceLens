#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Generic/Other Operations Analysis

Computes metrics for other/generic operations and outputs JSON for subagent processing.
This script handles Communication, Graph, and miscellaneous operations.
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
    classify_other_operation,
)


def get_other_config():
    """Return other-specific configuration."""
    return {
        "efficiency_method": "prefer_memory",  # Most "other" ops are memory-bound
        "extra_fields": [],
        "operation_classifier": classify_other_op,
    }


def classify_other_op(op_name: str, row) -> dict:
    """Classify other operation type."""
    category = classify_other_operation(op_name)

    return {"sub_category": category}


def extract_category_specific(ops_df, metadata, skipped_comm_ops=None) -> dict:
    """Extract other-specific aggregate metrics."""
    graph_count = 0
    misc_count = 0

    for name in ops_df["name"]:
        category = classify_other_operation(str(name))
        if category == "graph":
            graph_count += 1
        else:
            misc_count += 1

    result = {
        "communication_count": 0,
        "graph_count": graph_count,
        "miscellaneous_count": misc_count,
        "peak_maf_tflops": (
            metadata.get("max_achievable_tflops", {}).get("matrix_bf16")
            if isinstance(metadata.get("max_achievable_tflops"), dict)
            else metadata.get("peak_bf16_maf_tflops")
        ),
        "peak_hbm_bw_tbs": metadata.get("peak_hbm_bw_tbs"),
    }

    if skipped_comm_ops and skipped_comm_ops["count"] > 0:
        result["communication_ops_skipped"] = skipped_comm_ops

    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze other/generic operations")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--category", default="other", help="Category to analyze (default: other)"
    )
    args = parser.parse_args()

    category = args.category

    try:
        ops_df, metadata = load_category_data(args.output_dir, category)
    except FileNotFoundError as e:
        error_metrics = {"category": category, "status": "ERROR", "error": str(e)}
        write_metrics_json(error_metrics, args.output_dir, category)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    comm_mask = ops_df["name"].apply(
        lambda n: classify_other_operation(str(n)) == "communication"
    )
    comm_ops_df = ops_df[comm_mask]
    skipped_comm_ops = {
        "count": len(comm_ops_df),
        "op_names": comm_ops_df["name"].tolist(),
        "message": (
            "Communication kernels detected. Use TraceLens's NCCL Analyzer "
            "for detailed collective communication analysis (see TraceLens/NcclAnalyser/)."
        ),
    }
    if skipped_comm_ops["count"] > 0:
        print(
            f"Skipping {skipped_comm_ops['count']} communication kernel(s) — "
            f"use TraceLens's NCCL Analyzer instead."
        )
    ops_df = ops_df[~comm_mask]

    config = get_other_config()
    peak_hbm_bw = metadata.get("peak_hbm_bw_tbs", 1)
    maf = metadata.get("max_achievable_tflops", metadata.get("peak_bf16_maf_tflops", 1))

    time_metrics = calculate_time_metrics(ops_df, metadata)
    avg_efficiency = calculate_average_efficiency(
        ops_df, peak_hbm_bw, maf, "prefer_memory"
    )
    operations = build_operation_metrics(ops_df, metadata, config)
    category_specific = extract_category_specific(ops_df, metadata, skipped_comm_ops)

    baseline_ms = metadata.get("gpu_utilization", {}).get("total_time_ms", 0)
    impact_estimates = compute_impact_estimates(operations, category, baseline_ms=baseline_ms)

    metrics = {
        "category": category,
        "status": "OK",
        **time_metrics,
        "average_efficiency_percent": avg_efficiency,
        "operations": operations,
        "category_specific": category_specific,
        "impact_estimates": impact_estimates,
    }

    output_path = write_metrics_json(metrics, args.output_dir, category)
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
