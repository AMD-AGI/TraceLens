#!/usr/bin/env python3
"""
Match two semantic breakdowns and compute comparison stats with optional roofline.

Takes two semantic_labels.json files (one per trace) that have already been
individually broken down with matching semantic_block vocabularies.
Optionally takes derived_shapes.json files to enrich the comparison with
theoretical FLOPS/bytes and per-trace achieved performance metrics.

Input: two semantic_labels.json files, optionally two derived_shapes.json files
Output: comparison.csv

Usage:
    python match_and_compare.py <trace_a_labels.json> <trace_b_labels.json> \
        --name-a MI355 --name-b B200 \
        [--shapes-a derived_a.json --shapes-b derived_b.json] \
        [-o comparison.csv]
"""
import argparse
import csv
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trace_breakdown"))
from category_mappings import get_group, get_perf_category


def load_labels(path):
    with open(path) as f:
        return json.load(f)


def load_shapes(path):
    """Load derived_shapes.json and index blocks by semantic_block name."""
    with open(path) as f:
        data = json.load(f)
    return {b["semantic_block"]: b for b in data["blocks"]}


def aggregate(labeled_kernels):
    """Aggregate labeled kernels by semantic_block."""
    blocks = OrderedDict()
    for k in labeled_kernels:
        block = k["semantic_block"]
        if block not in blocks:
            blocks[block] = {"names": set(), "durs": [], "count": 0}
        b = blocks[block]
        b["names"].add(k["name"])
        b["durs"].append(k["dur"])
        b["count"] += 1
    return blocks


def build_comparison(agg_a, agg_b, total_a, total_b, name_a, name_b,
                     shapes_a=None, shapes_b=None):
    """Build comparison rows for all semantic blocks present in either trace."""
    all_blocks = list(OrderedDict.fromkeys(list(agg_a.keys()) + list(agg_b.keys())))
    has_shapes = shapes_a is not None or shapes_b is not None

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
        row["semantic_block"] = block
        row["semantic_group"] = get_group(block)
        row["perf_category"] = get_perf_category(block)
        row["algorithm_order"] = i + 1
        row[f"{name_a}_kernel_names"] = " | ".join(sorted(a["names"]))
        row[f"{name_a}_kernel_count"] = a["count"]
        row[f"{name_a}_total_us"] = round(a_total, 2)
        row[f"{name_a}_avg_us"] = round(a_avg, 2)
        row[f"{name_a}_pct"] = round(a_pct, 1)
        row[f"{name_b}_kernel_names"] = " | ".join(sorted(b["names"]))
        row[f"{name_b}_kernel_count"] = b["count"]
        row[f"{name_b}_total_us"] = round(b_total, 2)
        row[f"{name_b}_avg_us"] = round(b_avg, 2)
        row[f"{name_b}_pct"] = round(b_pct, 1)
        row[f"{name_a}_vs_{name_b}_ratio"] = (
            round(ratio, 3) if ratio != float("inf") else "inf"
        )
        row[f"{name_a}_minus_{name_b}_us"] = round(gap, 2)

        if has_shapes:
            sa = shapes_a.get(block, {}) if shapes_a else {}
            sb = shapes_b.get(block, {}) if shapes_b else {}
            shape_ref = sa or sb

            total_flops = shape_ref.get("total_flops", 0)
            total_bytes = shape_ref.get("total_bytes", 0)
            gflops = total_flops / 1e9
            data_mb = total_bytes / 1e6
            arith_intensity = total_flops / total_bytes if total_bytes > 0 else 0

            row["theoretical_GFLOPS"] = round(gflops, 4)
            row["theoretical_data_MB"] = round(data_mb, 4)
            row["FLOPS_per_Byte"] = round(arith_intensity, 2)

            a_time_s = a_total / 1e6
            b_time_s = b_total / 1e6
            row[f"{name_a}_TFLOPS_s"] = (
                round(total_flops / a_time_s / 1e12, 4) if a_time_s > 0 else 0
            )
            row[f"{name_a}_TB_s"] = (
                round(total_bytes / a_time_s / 1e12, 4) if a_time_s > 0 else 0
            )
            row[f"{name_b}_TFLOPS_s"] = (
                round(total_flops / b_time_s / 1e12, 4) if b_time_s > 0 else 0
            )
            row[f"{name_b}_TB_s"] = (
                round(total_bytes / b_time_s / 1e12, 4) if b_time_s > 0 else 0
            )

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
    parser.add_argument("trace_a", help="Path to trace A semantic_labels.json")
    parser.add_argument("trace_b", help="Path to trace B semantic_labels.json")
    parser.add_argument("--name-a", default="trace_a", help="Short name for trace A")
    parser.add_argument("--name-b", default="trace_b", help="Short name for trace B")
    parser.add_argument("--shapes-a", help="Path to trace A derived_shapes.json")
    parser.add_argument("--shapes-b", help="Path to trace B derived_shapes.json")
    parser.add_argument("-o", "--output", default="comparison.csv", help="Output CSV path")
    args = parser.parse_args()

    data_a = load_labels(args.trace_a)
    data_b = load_labels(args.trace_b)

    labeled_a = data_a["labeled_kernels"]
    labeled_b = data_b["labeled_kernels"]
    total_a = data_a.get("total_kernel_time_us", sum(k["dur"] for k in labeled_a))
    total_b = data_b.get("total_kernel_time_us", sum(k["dur"] for k in labeled_b))

    shapes_a = load_shapes(args.shapes_a) if args.shapes_a else None
    shapes_b = load_shapes(args.shapes_b) if args.shapes_b else None

    agg_a = aggregate(labeled_a)
    agg_b = aggregate(labeled_b)

    rows = build_comparison(
        agg_a, agg_b, total_a, total_b,
        args.name_a, args.name_b,
        shapes_a, shapes_b,
    )

    errors = run_assertions(rows, labeled_a, labeled_b, total_a, total_b, args.name_a, args.name_b)
    for e in errors:
        print(e, file=sys.stderr)
    if any("FAIL" in e for e in errors):
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {args.output} ({len(rows)} blocks, "
        f"{args.name_a}={len(labeled_a)} kernels/{total_a:.0f}us, "
        f"{args.name_b}={len(labeled_b)} kernels/{total_b:.0f}us)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
