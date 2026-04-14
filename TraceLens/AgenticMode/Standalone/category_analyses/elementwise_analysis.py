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

from analysis_utils import run_category_analysis


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract elementwise-specific aggregate metrics."""
    return {"peak_hbm_bw_tbs": metadata.get("peak_hbm_bw_tbs")}


def main():
    parser = argparse.ArgumentParser(description="Analyze elementwise operations")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    run_category_analysis(
        category="elementwise",
        output_dir=args.output_dir,
        config={"extra_fields": ["FLOPS/Byte"]},
        extract_fn=extract_category_specific,
    )


if __name__ == "__main__":
    main()
