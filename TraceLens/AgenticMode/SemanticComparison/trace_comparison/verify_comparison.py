#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Verification script for the cross-trace comparison pipeline.

Runs assertions on the comparison CSV and both semantic_labels.json files.
Returns exit code 0 if all pass, 1 if any fail.

Usage:
    python verify_comparison.py <trace_a_labels.json> <trace_b_labels.json> \
        <comparison.csv> --name-a MI355 --name-b B200
"""

import argparse
import csv
import json
import os
import sys


def verify(a_path, b_path, csv_path, name_a, name_b):
    errors = []
    warnings = []

    with open(a_path) as f:
        data_a = json.load(f)
    with open(b_path) as f:
        data_b = json.load(f)
    with open(csv_path) as f:
        csv_rows = list(csv.DictReader(f))

    labeled_a = data_a["labeled_kernels"]
    labeled_b = data_b["labeled_kernels"]
    total_a = data_a.get("total_kernel_time_us", sum(k["dur"] for k in labeled_a))
    total_b = data_b.get("total_kernel_time_us", sum(k["dur"] for k in labeled_b))

    # --- A6.1: Semantic vocabulary overlap ---
    blocks_a = set(k["semantic_block"] for k in labeled_a)
    blocks_b = set(k["semantic_block"] for k in labeled_b)
    only_a = blocks_a - blocks_b
    only_b = blocks_b - blocks_a
    overlap = blocks_a & blocks_b

    if only_a:
        warnings.append(
            f"A6.1 WARNING: {len(only_a)} blocks only in {name_a}: {sorted(only_a)[:5]}"
        )
    if only_b:
        warnings.append(
            f"A6.1 WARNING: {len(only_b)} blocks only in {name_b}: {sorted(only_b)[:5]}"
        )
    if not overlap:
        errors.append("A6.1 FAIL: No overlapping semantic blocks between traces")

    csv_blocks = set(r["semantic_block"] for r in csv_rows)
    missing_from_csv = (blocks_a | blocks_b) - csv_blocks
    if missing_from_csv:
        errors.append(f"A6.2 FAIL: Blocks missing from CSV: {sorted(missing_from_csv)}")

    # --- A6.3: semantic_group / perf_category presence ---
    for r in csv_rows:
        if not r.get("perf_category"):
            errors.append(f"A6.3 FAIL: {r['semantic_block']}: perf_category is empty")
        if "semantic_group" in r and not r["semantic_group"]:
            warnings.append(
                f"A6.3 WARNING: {r['semantic_block']}: semantic_group is empty"
            )

    # --- A7.3: Total kernel count ---
    csv_a_count = sum(int(r[f"{name_a}_kernel_count"]) for r in csv_rows)
    csv_b_count = sum(int(r[f"{name_b}_kernel_count"]) for r in csv_rows)
    if csv_a_count != len(labeled_a):
        errors.append(
            f"A7.3 FAIL: {name_a} kernel count CSV={csv_a_count} vs labels={len(labeled_a)}"
        )
    if csv_b_count != len(labeled_b):
        errors.append(
            f"A7.3 FAIL: {name_b} kernel count CSV={csv_b_count} vs labels={len(labeled_b)}"
        )

    # --- A7.2: Percentages ---
    pct_a = sum(float(r[f"{name_a}_pct"]) for r in csv_rows)
    pct_b = sum(float(r[f"{name_b}_pct"]) for r in csv_rows)
    if abs(pct_a - 100.0) > 2.0:
        errors.append(f"A7.2 FAIL: {name_a} percentages sum to {pct_a:.1f}%")
    if abs(pct_b - 100.0) > 2.0:
        errors.append(f"A7.2 FAIL: {name_b} percentages sum to {pct_b:.1f}%")

    # --- A7.4: Total time ---
    time_a = sum(float(r[f"{name_a}_total_us"]) for r in csv_rows)
    time_b = sum(float(r[f"{name_b}_total_us"]) for r in csv_rows)
    if abs(time_a - total_a) > 1.0:
        errors.append(f"A7.4 FAIL: {name_a} time {time_a:.1f} vs {total_a:.1f}")
    if abs(time_b - total_b) > 1.0:
        errors.append(f"A7.4 FAIL: {name_b} time {time_b:.1f} vs {total_b:.1f}")

    # --- A7.5: Ratio consistency ---
    for r in csv_rows:
        a_t = float(r[f"{name_a}_total_us"])
        b_t = float(r[f"{name_b}_total_us"])
        ratio_str = r[f"{name_a}_vs_{name_b}_ratio"]
        if ratio_str != "inf" and b_t > 0:
            expected = round(a_t / b_t, 3)
            actual = float(ratio_str)
            if abs(expected - actual) > 0.1:
                errors.append(
                    f"A7.5 FAIL: {r['semantic_block']}: "
                    f"ratio {actual} != {expected} ({a_t}/{b_t})"
                )

    # --- A7.6: Roofline consistency (if present) ---
    if f"{name_a}_TFLOPS_s" in csv_rows[0]:
        for r in csv_rows:
            a_t = float(r[f"{name_a}_total_us"])
            gflops = float(r["theoretical_GFLOPS"])
            actual_tflops = float(r[f"{name_a}_TFLOPS_s"])
            if a_t > 0 and gflops > 0.01 and actual_tflops > 0.01:
                expected_tflops = gflops / (a_t / 1e6) / 1e3
                if (
                    abs(expected_tflops - actual_tflops) / max(expected_tflops, 1e-9)
                    > 0.05
                ):
                    errors.append(
                        f"A7.6 FAIL: {r['semantic_block']}: {name_a} TFLOPS/s "
                        f"{actual_tflops} != expected {expected_tflops:.4f}"
                    )

    for w in warnings:
        print(f"  {w}", file=sys.stderr)
    for e in errors:
        print(f"  {e}", file=sys.stderr)

    print(
        f"\nVerification: {len(errors)} errors, {len(warnings)} warnings",
        file=sys.stderr,
    )
    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify cross-trace comparison outputs"
    )
    parser.add_argument("trace_a_labels", help="Path to trace A semantic_labels.json")
    parser.add_argument("trace_b_labels", help="Path to trace B semantic_labels.json")
    parser.add_argument("comparison_csv", help="Path to comparison CSV")
    parser.add_argument("--name-a", default="trace_a", help="Name of trace A")
    parser.add_argument("--name-b", default="trace_b", help="Name of trace B")
    args = parser.parse_args()

    ok = verify(
        args.trace_a_labels,
        args.trace_b_labels,
        args.comparison_csv,
        args.name_a,
        args.name_b,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
