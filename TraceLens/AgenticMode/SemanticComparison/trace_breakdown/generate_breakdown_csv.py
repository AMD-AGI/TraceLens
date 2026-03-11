#!/usr/bin/env python3
"""
Step 7: Generate the breakdown CSV from a semantic labeling JSON.

The semantic labeling JSON is produced by the LLM (Step 5) and contains
per-kernel semantic roles. This script aggregates them into a CSV.

Input: semantic_labels.json (produced by LLM via the skill)
Output: breakdown.csv

The semantic_labels.json format:
{
  "source_file": "...",
  "total_kernel_time_us": 5351.8,
  "labeled_kernels": [
    {"index": 0, "name": "...", "dur": 3.4, "semantic_block": "Preamble: Position Offset", "layer": null},
    {"index": 1, "name": "...", "dur": 4.6, "semantic_block": "Embedding Lookup", "layer": 0},
    ...
  ]
}

Usage:
    python generate_breakdown_csv.py <semantic_labels.json> [-o breakdown.csv]
"""
import argparse
import csv
import json
import sys
from collections import OrderedDict


def aggregate_by_block(labeled_kernels, total_time):
    """Aggregate labeled kernels into per-semantic-block statistics."""
    blocks = OrderedDict()
    for k in labeled_kernels:
        block = k["semantic_block"]
        if block not in blocks:
            blocks[block] = {
                "kernel_names": set(),
                "durations": [],
                "kernel_count": 0,
                "layers": set(),
                "first_index": k["index"],
            }
        b = blocks[block]
        b["kernel_names"].add(k["name"])
        b["durations"].append(k["dur"])
        b["kernel_count"] += 1
        if k.get("layer") is not None:
            b["layers"].add(k["layer"])

    rows = []
    order = 1
    for block, b in blocks.items():
        total = sum(b["durations"])
        avg = total / b["kernel_count"] if b["kernel_count"] else 0
        pct = 100 * total / total_time if total_time > 0 else 0
        rows.append({
            "semantic_block": block,
            "algorithm_order": order,
            "kernel_names": " | ".join(sorted(b["kernel_names"])),
            "kernel_count": b["kernel_count"],
            "total_us": round(total, 1),
            "avg_us": round(avg, 1),
            "pct_of_total": round(pct, 1),
            "layer_count": len(b["layers"]),
            "first_kernel_index": b["first_index"],
        })
        order += 1
    return rows


def run_assertions(rows, labeled_kernels, total_time):
    errors = []

    csv_total_kernels = sum(r["kernel_count"] for r in rows)
    if csv_total_kernels != len(labeled_kernels):
        errors.append(
            f"A7.3 FAIL: CSV kernel count ({csv_total_kernels}) != "
            f"labeled kernel count ({len(labeled_kernels)})"
        )

    csv_total_time = sum(r["total_us"] for r in rows)
    if abs(csv_total_time - total_time) > 1.0:
        errors.append(
            f"A7.4 FAIL: CSV total time ({csv_total_time:.1f}) != "
            f"trace total time ({total_time:.1f})"
        )

    pct_sum = sum(r["pct_of_total"] for r in rows)
    if abs(pct_sum - 100.0) > 2.0:
        errors.append(
            f"A7.2 FAIL: Percentages sum to {pct_sum:.1f}% (expected ~100%)"
        )

    unlabeled = [k for k in labeled_kernels if not k.get("semantic_block")]
    if unlabeled:
        errors.append(
            f"A5.1 FAIL: {len(unlabeled)} kernels have no semantic_block label"
        )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Generate breakdown CSV from semantic labels")
    parser.add_argument("labels_json", help="Path to semantic labels JSON")
    parser.add_argument("-o", "--output", default="breakdown.csv", help="Output CSV path")
    args = parser.parse_args()

    with open(args.labels_json) as f:
        data = json.load(f)

    labeled = data["labeled_kernels"]
    total_time = data.get("total_kernel_time_us", sum(k["dur"] for k in labeled))

    rows = aggregate_by_block(labeled, total_time)

    errors = run_assertions(rows, labeled, total_time)
    for e in errors:
        print(e, file=sys.stderr)
    if any("FAIL" in e for e in errors):
        sys.exit(1)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "semantic_block", "algorithm_order", "kernel_names",
            "kernel_count", "total_us", "avg_us", "pct_of_total",
            "layer_count", "first_kernel_index",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output} ({len(rows)} semantic blocks, {len(labeled)} kernels)", file=sys.stderr)


if __name__ == "__main__":
    main()
