#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton Kernel Analysis

Computes metrics for Triton kernels and outputs JSON for subagent processing.
"""

import argparse
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import (
    get_peak_specs,
    run_category_analysis,
)


def classify_triton_operation(op_name: str, row) -> dict:
    """Classify Triton kernel bound type."""
    flops_byte = row.get("FLOPS/Byte", 0) if not pd.isna(row.get("FLOPS/Byte")) else 0

    if flops_byte > 100:
        bound_type = "compute"
    elif flops_byte < 50:
        bound_type = "memory"
    else:
        bound_type = "mixed"

    return {
        "bound_type": bound_type,
        "flops_per_byte": round(flops_byte, 2) if flops_byte else None,
    }


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract Triton-specific aggregate metrics."""
    compute_bound = 0
    memory_bound = 0

    if "FLOPS/Byte" in ops_df.columns:
        for _, row in ops_df.iterrows():
            fb = row.get("FLOPS/Byte", 0) if not pd.isna(row.get("FLOPS/Byte")) else 0
            if fb > 100:
                compute_bound += 1
            elif fb < 50:
                memory_bound += 1

    return {
        "compute_bound_count": compute_bound,
        "memory_bound_count": memory_bound,
        **get_peak_specs(metadata),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Triton kernels")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    run_category_analysis(
        category="triton",
        output_dir=args.output_dir,
        config={
            "extra_fields": ["FLOPS/Byte"],
            "operation_classifier": classify_triton_operation,
        },
        extract_fn=extract_category_specific,
        compute_impact=False,
    )


if __name__ == "__main__":
    main()
