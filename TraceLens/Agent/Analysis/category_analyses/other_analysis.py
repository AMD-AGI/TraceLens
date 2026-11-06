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
from collections import Counter

from TraceLens.Agent.Analysis.category_analyses.analysis_utils import (
    get_peak_specs,
    run_category_analysis,
)

# Constants
COMMUNICATION_KEYWORDS = (
    "all_reduce",
    "collective",
    "ncclkernel",
    "rccl",
    "broadcast",
    "allgather",
)
GRAPH_KEYWORDS = ("graph", "hipgraph", "cudagraph")


def classify_other_operation(op_name: str) -> str:
    """Classify 'other' category operations."""
    op_lower = op_name.lower()

    if any(x in op_lower for x in COMMUNICATION_KEYWORDS):
        return "communication"

    if any(x in op_lower for x in GRAPH_KEYWORDS):
        return "graph"

    return "miscellaneous"


def _classify_other_op(op_name: str, _row) -> dict:
    """Operation classifier callback for build_operation_metrics."""
    return {"sub_category": classify_other_operation(op_name)}


def extract_category_specific(ops_df, metadata, skipped_comm_ops=None) -> dict:
    """Extract other-specific aggregate metrics."""
    counts = Counter(classify_other_operation(str(name)) for name in ops_df["name"])

    result = {
        "communication_count": counts["communication"],
        "graph_count": counts["graph"],
        "miscellaneous_count": counts["miscellaneous"],
        **get_peak_specs(metadata),
    }

    if skipped_comm_ops and skipped_comm_ops["count"] > 0:
        result["communication_ops_skipped"] = skipped_comm_ops

    return result


def _split_communication_ops(ops_df):
    """Return ``ops_df`` without communication kernels, and a record of the dropped ones."""
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
    return ops_df[~comm_mask], skipped_comm_ops


def main():
    parser = argparse.ArgumentParser(description="Analyze other/generic operations")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--category",
        required=True,
        help="Manifest category name; selects <category>_ops.csv",
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

    skipped_comm_ops = None

    def strip_communication_ops(ops_df, _metadata):
        nonlocal skipped_comm_ops
        ops_df, skipped_comm_ops = _split_communication_ops(ops_df)
        return ops_df, {}

    run_category_analysis(
        category=args.category,
        output_dir=args.output_dir,
        config={
            "extra_fields": ["Input Dims", "Input type"],
            "operation_classifier": _classify_other_op,
        },
        extract_fn=lambda ops_df, metadata: extract_category_specific(
            ops_df, metadata, skipped_comm_ops
        ),
        # Strip communication kernels from the "other" bucket only; other categories
        # (e.g. customcollective) reuse this script and must keep those ops.
        pre_process_fn=strip_communication_ops if args.category == "other" else None,
        comparison_scope=args.comparison_scope,
    )


if __name__ == "__main__":
    main()
