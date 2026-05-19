#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import (
    get_peak_specs,
    load_category_data,
    calculate_time_metrics,
    build_operation_metrics,
    build_category_findings,
    compute_impact_estimates,
    perf_report_csv_dir,
    write_metrics_json,
)


def classify_other_operation(op_name: str) -> str:
    """Classify 'other' category operations."""
    op_lower = op_name.lower()

    if any(
        x in op_lower
        for x in [
            "all_reduce",
            "collective",
            "ncclkernel",
            "rccl",
            "broadcast",
            "allgather",
        ]
    ):
        return "communication"

    if any(x in op_lower for x in ["graph", "hipgraph", "cudagraph"]):
        return "graph"

    return "miscellaneous"


def _classify_other_op(op_name: str, row) -> dict:
    """Operation classifier callback for build_operation_metrics."""
    return {"sub_category": classify_other_operation(op_name)}


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
        **get_peak_specs(metadata),
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

    parser.add_argument(
        "--comparison_scope",
        choices=("standalone", "comparative"),
        default="standalone",
        help=(
            "standalone: roofline efficiency in operations[].efficiency; "
            "comparative: 100*t2/t1 (needs TraceDiff CSV columns)"
        ),
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

    # Strip communication kernels from the "other" bucket only; other categories
    # (e.g. customcollective) reuse this script and must keep those ops.
    skipped_comm_ops = None
    if category == "other":
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

    config = {
        "extra_fields": ["Input Dims", "Input type"],
        "operation_classifier": _classify_other_op,
    }

    time_metrics = calculate_time_metrics(ops_df, metadata)
    
    callstacks_df = None
    cs_path = os.path.join(
        perf_report_csv_dir(args.output_dir), "unified_perf_callstacks.csv"
    )
    if os.path.exists(cs_path):
        callstacks_df = pd.read_csv(cs_path)

    operations = build_operation_metrics(
        ops_df,
        metadata,
        config,
        callstacks_df=callstacks_df,
        comparison_scope=args.comparison_scope,
    )
    category_specific = extract_category_specific(ops_df, metadata, skipped_comm_ops)

    baseline_ms = metadata.get("gpu_utilization", {}).get("total_time_ms", 0)
    impact_estimates = compute_impact_estimates(
        operations,
        category,
        baseline_ms=baseline_ms,
    )
    category_findings = build_category_findings(impact_estimates)

    metrics = {
        "category": category,
        "status": "OK",
        "comparison_scope": args.comparison_scope,
        **time_metrics,
        "operations": operations,
        "category_specific": category_specific,
        "impact_estimates": impact_estimates,
        "category_findings": category_findings,
    }

    output_path = write_metrics_json(metrics, args.output_dir, category)
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
