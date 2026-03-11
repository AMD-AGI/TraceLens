#!/usr/bin/env python3
"""
Generate a multi-sheet performance report from semantic labels + derived shapes.

Produces a report with column names compatible with the Standalone analysis
pipeline (orchestrator_prepare.py) so that graph-mode traces can be analyzed
using the same sub-agents as eager-mode traces.

Input:
  - semantic_labels.json   (from LLM labeling step)
  - derived_shapes.json    (from derive_shapes.py)
  - gpu_arch.json          (optional, for roofline time / Pct Roofline)

Output:
  - report.xlsx            (multi-sheet Excel workbook)
  - <output_csvs_dir>/     (one CSV per sheet, optional)

Sheets:
  1. category_breakdown     - per perf-category time + GFLOPS breakdown
  2. semantic_group_summary - high-level functional groups
  3. unified_perf_summary   - one row per semantic block (standalone-compatible)
  4. ops_summary            - per unique GPU kernel (standalone-compatible)
  5. gpu_timeline           - GPU utilization breakdown (standalone-compatible)

Usage:
    python generate_semantic_report.py <semantic_labels.json> <derived_shapes.json> \\
        [-o report.xlsx] [--gpu_arch gpu_arch.json] [--output_csvs_dir ./csvs]
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

import pandas as pd

from category_mappings import (
    get_group, get_perf_category, get_timeline_category,
    get_op_category, format_input_dims,
)


# ---------------------------------------------------------------------------
# Roofline helpers
# ---------------------------------------------------------------------------

def compute_roofline(flops, data_bytes, kernel_time_us, gpu_arch):
    """Compute roofline metrics given theoretical FLOPS/bytes and actual time."""
    if flops is None or data_bytes is None or kernel_time_us <= 0:
        return {}

    gflops = flops / 1e9
    data_mb = data_bytes / (1024 * 1024)
    tflops_s = gflops / (kernel_time_us / 1e6) / 1e3 if kernel_time_us > 0 else 0
    tb_s = data_bytes / 1e12 / (kernel_time_us / 1e6) if kernel_time_us > 0 else 0
    ai = flops / data_bytes if data_bytes > 0 else float("inf")

    metrics = {
        "GFLOPS": round(gflops, 4),
        "Data Moved (MB)": round(data_mb, 4),
        "FLOPS/Byte": round(ai, 2),
        "TFLOPS/s_mean": round(tflops_s, 4),
        "TB/s_mean": round(tb_s, 4),
    }

    if gpu_arch:
        mem_bw = gpu_arch.get("mem_bw_gbps", 0)
        peak_tflops_map = gpu_arch.get("max_achievable_tflops", {})
        peak_tflops = peak_tflops_map.get("matrix_bf16",
                      peak_tflops_map.get("matrix_fp16", 0))

        compute_time_us = 0
        if peak_tflops > 0:
            compute_time_us = (gflops / (peak_tflops * 1e3)) * 1e6

        memory_time_us = 0
        if mem_bw > 0:
            memory_time_us = (data_bytes / (mem_bw * 1e9)) * 1e6

        roofline_time = max(compute_time_us, memory_time_us)
        pct_roofline = (roofline_time / kernel_time_us) * 100 if kernel_time_us > 0 else 0

        metrics["Roofline_Time_us"] = round(roofline_time, 2)
        metrics["Pct Roofline"] = round(pct_roofline, 1)

    return metrics


# ---------------------------------------------------------------------------
# Sheet builders
# ---------------------------------------------------------------------------

def build_category_breakdown(labels_data, shapes_data):
    """Sheet 1: category_breakdown -- per perf-category time + GFLOPS.

    Uses TraceLens rocprof-style categories (GEMM, Attention, Normalization,
    Elementwise) sorted descending by time with a total row. Includes total
    theoretical GFLOPS per category from derived_shapes.
    """
    cats = OrderedDict()
    for k in labels_data["labeled_kernels"]:
        cat = get_timeline_category(k["semantic_block"])
        if cat not in cats:
            cats[cat] = {"dur": 0.0, "count": 0}
        cats[cat]["dur"] += k["dur"]
        cats[cat]["count"] += 1

    flops_by_cat = {}
    bytes_by_cat = {}
    for block_info in shapes_data.get("blocks", []):
        cat = get_timeline_category(block_info["semantic_block"])
        if cat not in flops_by_cat:
            flops_by_cat[cat] = 0
            bytes_by_cat[cat] = 0
        if block_info.get("total_flops"):
            flops_by_cat[cat] += block_info["total_flops"]
        if block_info.get("total_bytes"):
            bytes_by_cat[cat] += block_info["total_bytes"]

    total_time_us = sum(c["dur"] for c in cats.values())
    sorted_cats = sorted(cats.items(), key=lambda x: -x[1]["dur"])

    rows = []
    cumul = 0.0
    grand_gflops = 0.0
    for cat_name, info in sorted_cats:
        pct = 100 * info["dur"] / total_time_us if total_time_us > 0 else 0
        cumul += pct
        cat_gflops = flops_by_cat.get(cat_name, 0) / 1e9
        grand_gflops += cat_gflops
        rows.append({
            "category": cat_name,
            "kernel_count": info["count"],
            "total_time_ms": round(info["dur"] / 1000, 4),
            "total_GFLOPS": round(cat_gflops, 4),
            "Percentage (%)": round(pct, 1),
            "Cumulative Percentage (%)": round(cumul, 1),
        })

    rows.append({
        "category": "Total",
        "kernel_count": sum(c["count"] for c in cats.values()),
        "total_time_ms": round(total_time_us / 1000, 4),
        "total_GFLOPS": round(grand_gflops, 4),
        "Percentage (%)": 100.0,
        "Cumulative Percentage (%)": 100.0,
    })
    return pd.DataFrame(rows)


def build_semantic_group_summary(labels_data, shapes_data, gpu_arch):
    """Sheet 2: semantic_group_summary."""
    groups = OrderedDict()
    for k in labels_data["labeled_kernels"]:
        g = get_group(k["semantic_block"])
        if g not in groups:
            groups[g] = {"dur": [], "layers": set(), "kernel_count": 0}
        groups[g]["dur"].append(k["dur"])
        groups[g]["kernel_count"] += 1
        if k.get("layer") is not None:
            groups[g]["layers"].add(k["layer"])

    shapes_by_group = {}
    for block_info in shapes_data.get("blocks", []):
        g = block_info["semantic_group"]
        if g not in shapes_by_group:
            shapes_by_group[g] = {"total_flops": 0, "total_bytes": 0}
        if block_info["total_flops"]:
            shapes_by_group[g]["total_flops"] += block_info["total_flops"]
        if block_info["total_bytes"]:
            shapes_by_group[g]["total_bytes"] += block_info["total_bytes"]

    total_time = sum(sum(g["dur"]) for g in groups.values())
    rows = []
    cumul = 0
    for g_name, g_info in groups.items():
        t = sum(g_info["dur"])
        pct = 100 * t / total_time if total_time > 0 else 0
        cumul += pct
        row = {
            "semantic_group": g_name,
            "kernel_count": g_info["kernel_count"],
            "total_time_us": round(t, 2),
            "total_time_ms": round(t / 1000, 4),
            "pct_of_total": round(pct, 1),
            "cumul_pct": round(cumul, 1),
            "layer_count": len(g_info["layers"]) if g_info["layers"] else 0,
        }
        sg = shapes_by_group.get(g_name, {})
        if sg.get("total_flops"):
            row["total_GFLOPS"] = round(sg["total_flops"] / 1e9, 4)
        if sg.get("total_bytes"):
            row["total_Data_Moved_MB"] = round(sg["total_bytes"] / (1024 * 1024), 4)
        rows.append(row)

    return pd.DataFrame(rows)


PERF_PARAM_COLUMNS = [
    "M", "N", "K",
    "B", "N_Q", "H_Q", "N_KV", "H_KV", "d_h_qk", "d_h_v", "causal",
    "num_elems", "num_channels",
]


def _params_key(params):
    """Produce a hashable key from a perf_params dict."""
    if not params:
        return ()
    return tuple(sorted(params.items()))


def build_unified_perf_summary(labels_data, shapes_data, gpu_arch):
    """Sheet 3: unified_perf_summary -- standalone-compatible format.

    One row per unique (semantic_block, perf_params) combination. Uses column
    names compatible with orchestrator_prepare.py (``name``, ``op category``,
    ``Kernel Time (µs)_sum``, etc.) while keeping semantic metadata as extra
    columns for context.
    """
    shapes_lookup = {s["semantic_block"]: s for s in shapes_data.get("blocks", [])}

    blocks = OrderedDict()
    for k in labels_data["labeled_kernels"]:
        b = k["semantic_block"]
        shape = shapes_lookup.get(b, {})
        pk = _params_key(shape.get("perf_params"))
        key = (b, pk)
        if key not in blocks:
            blocks[key] = {
                "dur": [], "layers": set(), "kernel_count": 0,
                "perf_params": shape.get("perf_params", {}),
                "shape_info": shape,
            }
        blocks[key]["dur"].append(k["dur"])
        blocks[key]["kernel_count"] += 1
        if k.get("layer") is not None:
            blocks[key]["layers"].add(k["layer"])

    total_time = sum(sum(b["dur"]) for b in blocks.values())

    rows = []
    cumul = 0
    uid = 1
    for (block_name, _pk), b_info in blocks.items():
        t = sum(b_info["dur"])
        cnt = b_info["kernel_count"]
        avg = t / cnt if cnt > 0 else 0
        pct = 100 * t / total_time if total_time > 0 else 0
        cumul += pct
        perf_cat = get_perf_category(block_name)

        row = OrderedDict()
        # Standalone-compatible columns (primary)
        row["op category"] = get_op_category(block_name)
        row["name"] = block_name
        row["Kernel Time (µs)_sum"] = round(t, 2)
        row["count"] = cnt
        row["total_duration_us"] = round(t, 2)
        row["ex_UID"] = uid
        uid += 1

        pp = b_info["perf_params"]
        row["Input Dims"] = format_input_dims(pp, perf_cat)
        row["has_perf_model"] = True
        row["Compute Spec"] = "matrix_bf16"

        # Semantic context columns (extras)
        row["semantic_block"] = block_name
        row["semantic_group"] = get_group(block_name)
        row["perf_category"] = perf_cat
        row["avg_us"] = round(avg, 2)
        row["pct_of_total"] = round(pct, 1)
        row["cumul_pct"] = round(cumul, 1)
        row["layer_count"] = len(b_info["layers"]) if b_info["layers"] else 0

        for col in PERF_PARAM_COLUMNS:
            row[col] = pp.get(col, "")

        shape = b_info["shape_info"]
        if shape.get("per_invocation_flops") is not None:
            row["per_invocation_GFLOPS"] = round(shape["per_invocation_flops"] / 1e9, 6)
            row["per_invocation_data_MB"] = round(shape["per_invocation_bytes"] / 1e6, 6)

        if shape.get("total_flops") is not None:
            rf = compute_roofline(shape["total_flops"], shape["total_bytes"], t, gpu_arch)
            row.update(rf)

        rows.append(row)

    return pd.DataFrame(rows)


def build_ops_summary(labels_data):
    """Sheet 4: ops_summary -- per unique kernel, standalone-compatible columns."""
    kernel_map = OrderedDict()
    for k in labels_data["labeled_kernels"]:
        key = (k["semantic_block"], k["name"])
        if key not in kernel_map:
            kernel_map[key] = {"dur": [], "count": 0}
        kernel_map[key]["dur"].append(k["dur"])
        kernel_map[key]["count"] += 1

    total_time = sum(sum(v["dur"]) for v in kernel_map.values())

    rows = []
    for (block, kname), info in kernel_map.items():
        durs = info["dur"]
        t = sum(durs)
        pct = 100 * t / total_time if total_time > 0 else 0
        rows.append({
            "name": kname,
            "op category": get_op_category(block),
            "Kernel Time (µs)_sum": round(t, 2),
            "count": info["count"],
            "semantic_group": get_group(block),
            "semantic_block": block,
            "perf_category": get_perf_category(block),
            "mean_us": round(t / info["count"], 2) if info["count"] else 0,
            "min_us": round(min(durs), 2),
            "max_us": round(max(durs), 2),
            "pct_of_total": round(pct, 1),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("Kernel Time (µs)_sum", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def build_gpu_timeline(labels_data):
    """Sheet 5: gpu_timeline -- GPU utilization breakdown.

    For graph-mode traces, all kernel time is computation (no idle/comm/memcpy).
    """
    total_us = sum(k["dur"] for k in labels_data["labeled_kernels"])
    total_ms = total_us / 1000

    rows = [
        {"type": "total_time", "time ms": round(total_ms, 4), "percent": 100.0},
        {"type": "computation_time", "time ms": round(total_ms, 4), "percent": 100.0},
        {"type": "exposed_comm_time", "time ms": 0.0, "percent": 0.0},
        {"type": "exposed_memcpy_time", "time ms": 0.0, "percent": 0.0},
        {"type": "idle_time", "time ms": 0.0, "percent": 0.0},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_report(dict_name2df, output_xlsx_path=None, output_csvs_dir=None):
    """Write DataFrames as Excel sheets and/or CSV files."""
    if output_csvs_dir:
        os.makedirs(output_csvs_dir, exist_ok=True)
        for sheet_name, df in dict_name2df.items():
            csv_path = os.path.join(output_csvs_dir, f"{sheet_name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"  {sheet_name}.csv ({len(df)} rows)", file=sys.stderr)

    if output_xlsx_path:
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            print("WARNING: openpyxl not installed, skipping xlsx output. "
                  "Install with: pip install openpyxl", file=sys.stderr)
            return
        with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
            for sheet_name, df in dict_name2df.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Wrote {output_xlsx_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-sheet semantic performance report"
    )
    parser.add_argument("labels_json", help="Path to semantic_labels.json")
    parser.add_argument("shapes_json", help="Path to derived_shapes.json")
    parser.add_argument("-o", "--output", default=None,
                        help="Output Excel path (default: <labels_base>_semantic_report.xlsx)")
    parser.add_argument("--gpu_arch", default=None,
                        help="Path to GPU architecture JSON (for roofline)")
    parser.add_argument("--output_csvs_dir", default=None,
                        help="Directory to write per-sheet CSV files")
    args = parser.parse_args()

    with open(args.labels_json) as f:
        labels_data = json.load(f)
    with open(args.shapes_json) as f:
        shapes_data = json.load(f)

    gpu_arch = None
    if args.gpu_arch:
        with open(args.gpu_arch) as f:
            gpu_arch = json.load(f)

    print("Building report sheets...", file=sys.stderr)
    dict_name2df = OrderedDict()
    dict_name2df["category_breakdown"] = build_category_breakdown(labels_data, shapes_data)
    dict_name2df["semantic_group_summary"] = build_semantic_group_summary(
        labels_data, shapes_data, gpu_arch
    )
    dict_name2df["unified_perf_summary"] = build_unified_perf_summary(
        labels_data, shapes_data, gpu_arch
    )
    dict_name2df["ops_summary"] = build_ops_summary(labels_data)
    dict_name2df["gpu_timeline"] = build_gpu_timeline(labels_data)

    output_xlsx = args.output
    if output_xlsx is None and args.output_csvs_dir is None:
        base = args.labels_json.rsplit(".json", 1)[0]
        output_xlsx = base + "_semantic_report.xlsx"

    write_report(dict_name2df, output_xlsx_path=output_xlsx,
                 output_csvs_dir=args.output_csvs_dir)

    for name, df in dict_name2df.items():
        print(f"  Sheet '{name}': {len(df)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
