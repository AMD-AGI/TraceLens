#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Compute optimization priority ranking from a comparison CSV.

Priority score = pct_of_total * (ratio - 1)
Only considers blocks where trace A is slower than trace B (ratio > 1).

Input: comparison.csv (from match_and_compare.py)
Output: priority.json with ranked blocks

Usage:
    python compute_priority.py <comparison.csv> --name-a MI355 --name-b B200 \
        [-o priority.json] [--top 5]
"""

import argparse
import csv
import json
import sys


def compute(csv_path, name_a, name_b, top_n=5):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    scored = []
    for r in rows:
        pct = float(r[f"{name_a}_pct"])
        ratio_str = r[f"{name_a}_vs_{name_b}_ratio"]
        if ratio_str == "inf":
            ratio = float("inf")
        else:
            ratio = float(ratio_str)

        if ratio <= 1.0:
            continue

        gap_us = float(r[f"{name_a}_minus_{name_b}_us"])
        if ratio == float("inf"):
            score = pct * 10
        else:
            score = pct * (ratio - 1)

        entry = {
            "semantic_block": r["semantic_block"],
            "semantic_group": r.get("semantic_group", ""),
            "perf_category": r.get("perf_category", ""),
            "priority_score": round(score, 2),
            f"{name_a}_pct": pct,
            "ratio": round(ratio, 2) if ratio != float("inf") else "inf",
            "gap_us": round(gap_us, 1),
            f"{name_a}_total_us": float(r[f"{name_a}_total_us"]),
            f"{name_b}_total_us": float(r[f"{name_b}_total_us"]),
        }

        if f"{name_a}_TFLOPS_s" in r:
            entry[f"{name_a}_TFLOPS_s"] = float(r[f"{name_a}_TFLOPS_s"])
            entry[f"{name_b}_TFLOPS_s"] = float(r[f"{name_b}_TFLOPS_s"])
            entry["FLOPS_per_Byte"] = float(r.get("FLOPS_per_Byte", 0))

        scored.append(entry)

    scored.sort(key=lambda x: -x["priority_score"])
    return scored[:top_n]


def main():
    parser = argparse.ArgumentParser(
        description="Compute optimization priority ranking"
    )
    parser.add_argument("comparison_csv", help="Path to comparison CSV")
    parser.add_argument("--name-a", default="trace_a", help="Name of trace A")
    parser.add_argument("--name-b", default="trace_b", help="Name of trace B")
    parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    parser.add_argument("--top", type=int, default=5, help="Number of top priorities")
    args = parser.parse_args()

    priorities = compute(args.comparison_csv, args.name_a, args.name_b, args.top)

    result = {
        "trace_a": args.name_a,
        "trace_b": args.name_b,
        "scoring_formula": f"{args.name_a}_runtime_pct * ({args.name_a}/{args.name_b}_ratio - 1)",
        "top_priorities": priorities,
    }

    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(
            f"Top {len(priorities)} priorities written to {args.output}",
            file=sys.stderr,
        )
    else:
        print(output)


if __name__ == "__main__":
    main()
