#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Reduce Operations Analysis

Computes metrics for reduce operations and outputs JSON for subagent processing.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import run_category_analysis


def detect_softmax(op_name: str) -> bool:
    """Check if operation is a softmax."""
    return "softmax" in op_name.lower()


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract reduce-specific aggregate metrics."""
    softmax_count = sum(1 for name in ops_df["name"] if detect_softmax(str(name)))

    return {
        "softmax_count": int(softmax_count),
        "peak_hbm_bw_tbs": metadata.get("peak_hbm_bw_tbs"),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze reduce operations")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    run_category_analysis(
        category="reduce",
        output_dir=args.output_dir,
        config={"extra_fields": []},
        extract_fn=extract_category_specific,
    )


if __name__ == "__main__":
    main()
