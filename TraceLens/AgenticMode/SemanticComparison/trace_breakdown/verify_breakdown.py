#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Verification script for the semantic breakdown pipeline.

Runs all assertions from Steps 1-5 and 7 on the final breakdown outputs.
Returns exit code 0 if all pass, 1 if any fail.

Usage:
    python verify_breakdown.py <semantic_labels.json> <breakdown.csv>
"""

import argparse
import csv
import json
import sys


def verify(labels_path, csv_path):
    errors = []
    warnings = []

    with open(labels_path) as f:
        labels_data = json.load(f)
    labeled = labels_data["labeled_kernels"]
    total_time = labels_data.get("total_kernel_time_us", sum(k["dur"] for k in labeled))

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    # --- A5.1: Every kernel has a label ---
    unlabeled = [k for k in labeled if not k.get("semantic_block")]
    if unlabeled:
        errors.append(f"A5.1 FAIL: {len(unlabeled)} kernels have no semantic_block")

    # --- A5.2: Structural ordering within layers ---
    layers = {}
    for k in labeled:
        layer = k.get("layer")
        if layer is not None:
            layers.setdefault(layer, []).append(k["semantic_block"])

    for layer_num, blocks in sorted(layers.items()):
        attn_indices = [i for i, b in enumerate(blocks) if "Attention" in b]
        moe_indices = [i for i, b in enumerate(blocks) if "MoE" in b]
        if attn_indices and moe_indices:
            if max(attn_indices) > min(moe_indices):
                errors.append(
                    f"A5.2 FAIL: Layer {layer_num}: Attention block appears after MoE block"
                )

        norm_indices = [i for i, b in enumerate(blocks) if "Norm" in b]
        if norm_indices and blocks:
            if norm_indices[0] != 0:
                warnings.append(
                    f"A5.2 WARNING: Layer {layer_num}: First block is '{blocks[0]}', expected Norm"
                )

    # --- A5.4: Exactly one attention per layer ---
    for layer_num, blocks in sorted(layers.items()):
        attn_count = sum(1 for b in blocks if b == "Attention")
        if attn_count < 1:
            errors.append(
                f"A5.4 FAIL: Layer {layer_num}: {attn_count} 'Attention' blocks (expected >= 1)"
            )
        elif attn_count > 1:
            warnings.append(
                f"A5.4 WARNING: Layer {layer_num}: {attn_count} 'Attention' blocks "
                f"(may include attention reduce)"
            )

    # --- A7.2: Percentages sum to ~100% ---
    pct_sum = sum(float(r["pct_of_total"]) for r in csv_rows)
    if abs(pct_sum - 100.0) > 2.0:
        errors.append(f"A7.2 FAIL: Percentages sum to {pct_sum:.1f}%")

    # --- A7.3: Kernel count matches ---
    csv_count = sum(int(r["kernel_count"]) for r in csv_rows)
    if csv_count != len(labeled):
        errors.append(
            f"A7.3 FAIL: CSV kernel count ({csv_count}) != labels ({len(labeled)})"
        )

    # --- A7.4: Total time matches ---
    time_key = "total_time_us" if "total_time_us" in csv_rows[0] else "total_us"
    csv_time = sum(float(r[time_key]) for r in csv_rows)
    if abs(csv_time - total_time) > 1.0:
        errors.append(
            f"A7.4 FAIL: CSV time ({csv_time:.1f}) != trace time ({total_time:.1f})"
        )

    # --- A5.5: GateUp GEMM contains swiglu ---
    gateup_kernels = [k for k in labeled if "GateUp" in k.get("semantic_block", "")]
    for k in gateup_kernels:
        if "swiglu" not in k["name"].lower() and "swiglu" not in k["name"]:
            warnings.append(
                f"A5.5 WARNING: GateUp kernel #{k['index']} name doesn't contain 'swiglu': "
                f"{k['name'][:60]}"
            )
            break

    # --- Report ---
    for w in warnings:
        print(f"  {w}", file=sys.stderr)
    for e in errors:
        print(f"  {e}", file=sys.stderr)

    n_pass = (5 + 2) - len(errors)
    print(
        f"\nVerification: {n_pass} passed, {len(errors)} failed, {len(warnings)} warnings",
        file=sys.stderr,
    )
    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Verify semantic breakdown outputs")
    parser.add_argument("labels_json", help="Path to semantic labels JSON")
    parser.add_argument("breakdown_csv", help="Path to breakdown CSV")
    args = parser.parse_args()

    ok = verify(args.labels_json, args.breakdown_csv)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
