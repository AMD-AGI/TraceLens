#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Match two semantic breakdowns and compute comparison stats.

Takes two semantic_labels.json files (one per trace) that have already been
individually broken down with matching semantic_block vocabularies.

Input: two semantic_labels.json files
Output: comparison.csv

Usage:
    python match_and_compare.py <trace_a_labels.json> <trace_b_labels.json> \
        --name-a MI355 --name-b B200 \
        [-o comparison.csv]

    For multi-region (per steady-state), use --regions-dir-a and --regions-dir-b:
    python match_and_compare.py --regions-dir-a output/MI355 --regions-dir-b output/B200 \
        --name-a MI355 --name-b B200 -o comparison.csv

    This matches regions by subdir name (e.g. prefill_only_3072) and compares
    only corresponding regions (apples-to-apples).
"""

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import load_labels



def aggregate(labeled_kernels):
    """Aggregate labeled kernels by semantic_block.

    Carries forward perf_category and nn_module from kernel data.
    """
    blocks = OrderedDict()
    for k in labeled_kernels:
        block = k["semantic_block"]
        if block not in blocks:
            blocks[block] = {
                "names": set(),
                "durs": [],
                "count": 0,
                "perf_category": k.get("perf_category"),
                "nn_module": k.get("nn_module"),
            }
        b = blocks[block]
        b["names"].add(k["name"])
        b["durs"].append(k["dur"])
        b["count"] += 1
    return blocks


def load_metadata(path):
    """Load metadata.json and return gpu_timeline dict if present."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            meta = json.load(f)
        return meta.get("gpu_timeline")
    except (json.JSONDecodeError, OSError):
        return None


def build_comparison(
    agg_a,
    agg_b,
    total_a,
    total_b,
    name_a,
    name_b,
    region=None,
    gpu_timeline_a=None,
    gpu_timeline_b=None,
):
    """Build comparison rows for all semantic blocks present in either trace."""
    all_blocks = list(OrderedDict.fromkeys(list(agg_a.keys()) + list(agg_b.keys())))
    has_gpu_timeline = gpu_timeline_a is not None and gpu_timeline_b is not None

    rows = []
    for i, block in enumerate(all_blocks):
        a = agg_a.get(block, {"names": set(), "durs": [], "count": 0})
        b = agg_b.get(block, {"names": set(), "durs": [], "count": 0})

        a_total = sum(a["durs"])
        b_total = sum(b["durs"])
        a_avg = a_total / a["count"] if a["count"] else 0
        b_avg = b_total / b["count"] if b["count"] else 0
        a_pct = 100 * a_total / total_a if total_a > 0 else 0
        b_pct = 100 * b_total / total_b if total_b > 0 else 0
        ratio = a_total / b_total if b_total > 0 else float("inf")
        gap = a_total - b_total

        row = OrderedDict()
        if region is not None:
            row["region"] = region
        row["semantic_block"] = block
        kernel_names_a_str = " | ".join(sorted(a["names"]))
        kernel_names_b_str = " | ".join(sorted(b["names"]))
        row["perf_category"] = (
            a.get("perf_category") or b.get("perf_category") or "Others"
        )
        row["nn_module"] = a.get("nn_module") or b.get("nn_module") or ""
        row["algorithm_order"] = i + 1
        row[f"{name_a}_kernel_names"] = kernel_names_a_str
        row[f"{name_a}_kernel_count"] = a["count"]
        row[f"{name_a}_total_us"] = round(a_total, 2)
        row[f"{name_a}_avg_us"] = round(a_avg, 2)
        row[f"{name_a}_pct"] = round(a_pct, 1)
        row[f"{name_b}_kernel_names"] = kernel_names_b_str
        row[f"{name_b}_kernel_count"] = b["count"]
        row[f"{name_b}_total_us"] = round(b_total, 2)
        row[f"{name_b}_avg_us"] = round(b_avg, 2)
        row[f"{name_b}_pct"] = round(b_pct, 1)
        row[f"{name_a}_vs_{name_b}_ratio"] = (
            round(ratio, 3) if ratio != float("inf") else "inf"
        )
        row[f"{name_a}_minus_{name_b}_us"] = round(gap, 2)

        if has_gpu_timeline:
            row[f"{name_a}_busy_ms"] = round(
                gpu_timeline_a.get("busy_time_us", 0) / 1000, 2
            )
            row[f"{name_a}_idle_pct"] = gpu_timeline_a.get("idle_pct")
            row[f"{name_b}_busy_ms"] = round(
                gpu_timeline_b.get("busy_time_us", 0) / 1000, 2
            )
            row[f"{name_b}_idle_pct"] = gpu_timeline_b.get("idle_pct")

        rows.append(row)
    return rows


def run_assertions(rows, labeled_a, labeled_b, total_a, total_b, name_a, name_b):
    errors = []

    a_count = sum(r[f"{name_a}_kernel_count"] for r in rows)
    if a_count != len(labeled_a):
        errors.append(
            f"A6.1 FAIL: {name_a} kernel count mismatch: {a_count} matched vs {len(labeled_a)} total"
        )

    b_count = sum(r[f"{name_b}_kernel_count"] for r in rows)
    if b_count != len(labeled_b):
        errors.append(
            f"A6.2 FAIL: {name_b} kernel count mismatch: {b_count} matched vs {len(labeled_b)} total"
        )

    a_time = sum(r[f"{name_a}_total_us"] for r in rows)
    if abs(a_time - total_a) > 1.0:
        errors.append(
            f"A6.3 FAIL: {name_a} time mismatch: {a_time:.1f} vs {total_a:.1f}"
        )
    b_time = sum(r[f"{name_b}_total_us"] for r in rows)
    if abs(b_time - total_b) > 1.0:
        errors.append(
            f"A6.3 FAIL: {name_b} time mismatch: {b_time:.1f} vs {total_b:.1f}"
        )

    a_pct = sum(r[f"{name_a}_pct"] for r in rows)
    if abs(a_pct - 100.0) > 2.0:
        errors.append(f"A7.2 FAIL: {name_a} percentages sum to {a_pct:.1f}%")
    b_pct = sum(r[f"{name_b}_pct"] for r in rows)
    if abs(b_pct - 100.0) > 2.0:
        errors.append(f"A7.2 FAIL: {name_b} percentages sum to {b_pct:.1f}%")

    for r in rows:
        a_t = r[f"{name_a}_total_us"]
        b_t = r[f"{name_b}_total_us"]
        if b_t > 0:
            expected_ratio = round(a_t / b_t, 3)
            actual_ratio = r[f"{name_a}_vs_{name_b}_ratio"]
            if actual_ratio != "inf" and abs(expected_ratio - actual_ratio) > 0.1:
                errors.append(
                    f"A7.5 FAIL: {r['semantic_block']}: ratio mismatch "
                    f"{expected_ratio} vs {actual_ratio}"
                )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Compare two semantic breakdowns")
    parser.add_argument(
        "trace_a", nargs="?", help="Path to trace A semantic_labels.json"
    )
    parser.add_argument(
        "trace_b", nargs="?", help="Path to trace B semantic_labels.json"
    )
    parser.add_argument(
        "--regions-dir-a",
        help="Dir with per-region subdirs (e.g. prefill_only_3072/semantic_labels.json)",
    )
    parser.add_argument(
        "--regions-dir-b", help="Dir with per-region subdirs for trace B"
    )
    parser.add_argument("--name-a", default="trace_a", help="Short name for trace A")
    parser.add_argument("--name-b", default="trace_b", help="Short name for trace B")
    parser.add_argument(
        "--region",
        help="Region descriptor (e.g. prefill_only_3072) for multi-section reports",
    )
    parser.add_argument(
        "-o", "--output", default="comparison.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    all_rows = []
    fieldnames = None

    if args.regions_dir_a and args.regions_dir_b:
        regions_a = {
            d
            for d in os.listdir(args.regions_dir_a)
            if os.path.isdir(os.path.join(args.regions_dir_a, d))
        }
        regions_b = {
            d
            for d in os.listdir(args.regions_dir_b)
            if os.path.isdir(os.path.join(args.regions_dir_b, d))
        }
        common_regions = sorted(regions_a & regions_b)
        for region in common_regions:
            labels_a = os.path.join(args.regions_dir_a, region, "semantic_labels.json")
            labels_b = os.path.join(args.regions_dir_b, region, "semantic_labels.json")
            metadata_a_path = os.path.join(args.regions_dir_a, region, "metadata.json")
            metadata_b_path = os.path.join(args.regions_dir_b, region, "metadata.json")
            if not os.path.exists(labels_a) or not os.path.exists(labels_b):
                continue
            data_a = load_labels(labels_a)
            data_b = load_labels(labels_b)
            labeled_a = data_a["labeled_kernels"]
            labeled_b = data_b["labeled_kernels"]
            total_a = data_a.get(
                "total_kernel_time_us", sum(k["dur"] for k in labeled_a)
            )
            total_b = data_b.get(
                "total_kernel_time_us", sum(k["dur"] for k in labeled_b)
            )
            gpu_timeline_a = load_metadata(metadata_a_path)
            gpu_timeline_b = load_metadata(metadata_b_path)
            agg_a = aggregate(labeled_a)
            agg_b = aggregate(labeled_b)
            rows = build_comparison(
                agg_a,
                agg_b,
                total_a,
                total_b,
                args.name_a,
                args.name_b,
                region=region,
                gpu_timeline_a=gpu_timeline_a,
                gpu_timeline_b=gpu_timeline_b,
            )
            all_rows.extend(rows)
            if fieldnames is None and rows:
                fieldnames = list(rows[0].keys())
        if not all_rows:
            print("No matching regions found", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.trace_a or not args.trace_b:
            parser.error(
                "Provide trace_a and trace_b, or --regions-dir-a and --regions-dir-b"
            )
        data_a = load_labels(args.trace_a)
        data_b = load_labels(args.trace_b)
        labeled_a = data_a["labeled_kernels"]
        labeled_b = data_b["labeled_kernels"]
        total_a = data_a.get("total_kernel_time_us", sum(k["dur"] for k in labeled_a))
        total_b = data_b.get("total_kernel_time_us", sum(k["dur"] for k in labeled_b))
        agg_a = aggregate(labeled_a)
        agg_b = aggregate(labeled_b)
        all_rows = build_comparison(
            agg_a,
            agg_b,
            total_a,
            total_b,
            args.name_a,
            args.name_b,
            region=args.region,
        )
        fieldnames = list(all_rows[0].keys()) if all_rows else []
        if all_rows:
            errors = run_assertions(
                all_rows,
                labeled_a,
                labeled_b,
                total_a,
                total_b,
                args.name_a,
                args.name_b,
            )
            for e in errors:
                print(e, file=sys.stderr)
            if any("FAIL" in e for e in errors):
                sys.exit(1)

    if fieldnames and all_rows:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote {args.output} ({len(all_rows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
