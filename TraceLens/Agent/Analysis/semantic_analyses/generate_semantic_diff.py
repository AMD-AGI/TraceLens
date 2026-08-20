#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Generate TraceDiff-compatible output from two semantic breakdowns.

Takes two semantic_labels.json files (from the semantic breakdown pipeline)
and produces output files identical in schema to TraceDiff's
print_tracediff_report_files(), enabling the existing comparative mode
(tracelens_diff_analyzer.py) to consume graph-mode trace comparisons.

Instead of the Wagner-Fischer DP tree-merge algorithm, kernels are matched
across traces by their semantic_block label.

Input:
  - Two semantic_labels.json files (one per trace)

Output (in output directory):
  - diff_stats.csv (includes per-kernel ``gpu_op_uid`` -- the TraceTree UID --
    and per-LCA ``busy_time``, so the output is consumable by the perf-report
    comparison enrichment in tracediff_comparison_extension.py)
  - diff_stats_unique_args_summary.csv
  - cpu_op_map.json
  - cpu_op_map_trace1.json
  - cpu_op_map_trace2.json
  - merged_tree_output.txt

Usage:
    python generate_semantic_diff.py \\
        trace_a/semantic_labels.json trace_b/semantic_labels.json \\
        --name-a MI355 --name-b B200 \\
        -o output_dir/
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import load_labels


# ---------------------------------------------------------------------------
# diff_stats.csv generation
# ---------------------------------------------------------------------------


def build_diff_stats(labeled_a, labeled_b):
    """Build a list of row dicts matching TraceDiff's diff_stats.csv schema.

    Each kernel becomes one row.  Kernels are grouped by semantic_block,
    which is the functional label (e.g. "QKV Projection") when available.

    The kernel's ``nn_module`` field is used for the nn_module columns.
    Falls back to perf_category.

    Returns (rows, block_id_map) where block_id_map is {semantic_block: int}.
    """
    all_blocks_ordered = list(
        OrderedDict.fromkeys(
            [k["semantic_block"] for k in labeled_a]
            + [k["semantic_block"] for k in labeled_b]
        )
    )
    block_id_map = {block: idx for idx, block in enumerate(all_blocks_ordered)}

    def _nn_module(k):
        if k.get("nn_module"):
            return k["nn_module"]
        return k.get("perf_category", "Others")

    def _format_dims(dims_list):
        if not dims_list:
            return ""
        parts = []
        for d in dims_list:
            if isinstance(d, (list, tuple)):
                parts.append(str(tuple(d)))
            else:
                parts.append(str(d))
        return ", ".join(parts)

    rows = []
    for source_tag, kernels in [
        ("trace1", labeled_a),
        ("trace2", labeled_b),
    ]:
        for k in kernels:
            block = k["semantic_block"]
            nn_mod = _nn_module(k)
            cpu_op = k.get("cpu_op", "") or block
            rows.append(
                {
                    "name": k["name"],
                    "cpu_op_name": cpu_op,
                    "source": source_tag,
                    "Input Dims": _format_dims(k.get("input_dims", [])),
                    "Input Strides": "",
                    "Input type": "",
                    "Concrete Inputs": "",
                    "kernel_time": k["dur"],
                    "lowest_common_ancestor_name": block,
                    "lowest_common_ancestor_id": block_id_map[block],
                    "nn_module_stack": nn_mod,
                    "nn_module_parent": nn_mod,
                    "gpu_op_uid": k.get("gpu_op_uid"),
                }
            )

    return rows, block_id_map


# ---------------------------------------------------------------------------
# diff_stats_unique_args_summary.csv generation
# ---------------------------------------------------------------------------


def build_unique_args_summary(diff_stats_df):
    """Aggregate diff_stats rows by all non-metric columns.

    Mirrors TraceDiff.get_df_diff_stats_unique_args() with sum aggregation
    on kernel_time.
    """
    metric_columns = ["kernel_time"]
    # gpu_op_uid is unique per kernel (would explode groups) and busy_time is
    # a derived per-LCA aggregate, so neither belongs in the grouping key.
    excluded_cols = {"lowest_common_ancestor_id", "gpu_op_uid", "busy_time"}
    grouping_cols = [
        c
        for c in diff_stats_df.columns
        if c not in metric_columns and c not in excluded_cols
    ]

    agg_dict = {mcol: ["sum", "mean"] for mcol in metric_columns}
    for col in grouping_cols:
        agg_dict[col] = "first"

    try:
        df_agg = diff_stats_df.groupby(grouping_cols, dropna=False).agg(agg_dict)
        df_agg["operation_count"] = diff_stats_df.groupby(
            grouping_cols, dropna=False
        ).size()
    except TypeError:
        str_cols = [f"{col}_str_repr" for col in grouping_cols]
        df_temp = diff_stats_df.copy()
        for col, str_col in zip(grouping_cols, str_cols):
            df_temp[str_col] = df_temp[col].astype(str)
        df_agg = df_temp.groupby(str_cols, dropna=False).agg(agg_dict)
        df_agg["operation_count"] = df_temp.groupby(str_cols, dropna=False).size()

    df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]
    df_agg = df_agg.reset_index(drop=True)

    rename_map = {}
    for col in grouping_cols:
        col_first = f"{col}_first"
        if col_first in df_agg.columns:
            rename_map[col_first] = col
    df_agg = df_agg.rename(columns=rename_map)

    primary_cols = grouping_cols
    metric_cols = []
    for metric in metric_columns:
        for agg in ["sum", "mean"]:
            col_name = f"{metric}_{agg}"
            if col_name in df_agg.columns:
                metric_cols.append(col_name)
    metric_cols = list(dict.fromkeys(metric_cols))
    other_cols = [
        col for col in df_agg.columns if col not in primary_cols + metric_cols
    ]
    df_agg = df_agg[primary_cols + metric_cols + other_cols]

    df_agg = df_agg.rename(columns={"operation_count_": "operation_count"})
    if "operation_count" in df_agg.columns:
        cols = list(df_agg.columns)
        cols.remove("operation_count")
        cols.insert(1, "operation_count")
        df_agg = df_agg[cols]

    sort_col = "kernel_time_sum"
    if sort_col in df_agg.columns:
        df_agg = df_agg.sort_values(by=sort_col, ascending=False, ignore_index=True)

    return df_agg


# ---------------------------------------------------------------------------
# cpu_op_map JSON generation
# ---------------------------------------------------------------------------


def build_cpu_op_maps(diff_stats_df):
    """Build cpu_op_map dicts analogous to TraceDiff.get_cpu_op_to_kernels_json().

    Returns (cpu_op_map, cpu_op_map_trace1_df, cpu_op_map_trace2_df).
    """
    cpu_op_map = {}
    for cpu_op in diff_stats_df["cpu_op_name"].unique():
        cpu_op_map[cpu_op] = {}
        sub = diff_stats_df[diff_stats_df["cpu_op_name"] == cpu_op]
        for source, group in sub.groupby("source"):
            cpu_op_map[cpu_op][source] = {
                "kernels": sorted(list(group["name"].unique())),
                "nn_module_parents": sorted(list(group["nn_module_parent"].unique())),
            }

    cpu_op_map_trace1 = (
        diff_stats_df[diff_stats_df["source"] == "trace1"]
        .groupby("cpu_op_name")
        .agg({"name": lambda x: sorted(set(x))})
        .sort_index()
    )
    cpu_op_map_trace2 = (
        diff_stats_df[diff_stats_df["source"] == "trace2"]
        .groupby("cpu_op_name")
        .agg({"name": lambda x: sorted(set(x))})
        .sort_index()
    )

    return cpu_op_map, cpu_op_map_trace1, cpu_op_map_trace2


# ---------------------------------------------------------------------------
# merged_tree_output.txt generation
# ---------------------------------------------------------------------------


def build_merged_tree_text(block_id_map, labeled_a, labeled_b, name_a, name_b):
    """Build a text tree representation mimicking TraceDiff's merged tree.

    Structure:
      Root
        └── <nn_module or perf_category>
            ├── <semantic_block> (combined / trace1-only / trace2-only)
            │   ├── kernel_name_a  [trace1]
            │   └── kernel_name_b  [trace2]
    """
    blocks_a = OrderedDict()
    for k in labeled_a:
        b = k["semantic_block"]
        blocks_a.setdefault(b, []).append(k["name"])
    blocks_b = OrderedDict()
    for k in labeled_b:
        b = k["semantic_block"]
        blocks_b.setdefault(b, []).append(k["name"])

    block_nn_module = {}
    for k in labeled_a + labeled_b:
        b = k["semantic_block"]
        if b not in block_nn_module:
            if k.get("nn_module"):
                block_nn_module[b] = k["nn_module"]
            else:
                block_nn_module[b] = k.get("perf_category", "Others")

    all_blocks = list(block_id_map.keys())

    groups = OrderedDict()
    for block in all_blocks:
        g = block_nn_module.get(block, "Others")
        groups.setdefault(g, []).append(block)

    lines = []
    lines.append(f"└── Root ({name_a} vs {name_b})")

    group_list = list(groups.items())
    for gi, (group_name, group_blocks) in enumerate(group_list):
        is_last_group = gi == len(group_list) - 1
        g_connector = "└── " if is_last_group else "├── "
        g_prefix = "    " if is_last_group else "│   "
        lines.append(f"    {g_connector}{group_name}")

        for bi, block in enumerate(group_blocks):
            is_last_block = bi == len(group_blocks) - 1
            b_connector = "└── " if is_last_block else "├── "
            b_prefix = "    " if is_last_block else "│   "

            in_a = block in blocks_a
            in_b = block in blocks_b

            if in_a and in_b:
                label = block
            elif in_a:
                label = f">> trace1: {block}"
            else:
                label = f"<< trace2: {block}"

            lines.append(f"    {g_prefix}{b_connector}{label}")

            kernels_a = blocks_a.get(block, [])
            kernels_b = blocks_b.get(block, [])
            kernel_names_a = sorted(set(kernels_a))
            kernel_names_b = sorted(set(kernels_b))

            all_kernel_entries = [(kn, "trace1") for kn in kernel_names_a] + [
                (kn, "trace2") for kn in kernel_names_b
            ]

            for ki, (kn, src) in enumerate(all_kernel_entries):
                is_last_kernel = ki == len(all_kernel_entries) - 1
                k_connector = "└── " if is_last_kernel else "├── "
                if in_a and in_b:
                    k_line = kn
                elif src == "trace1":
                    k_line = f">> {src}: {kn}"
                else:
                    k_line = f"<< {src}: {kn}"
                lines.append(f"    {g_prefix}{b_prefix}{k_connector}{k_line}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def write_outputs(
    output_dir,
    diff_stats_df,
    summary_df,
    cpu_op_map,
    cpu_op_map_trace1,
    cpu_op_map_trace2,
    merged_tree_text,
):
    """Write all output files to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    diff_stats_df.to_csv(os.path.join(output_dir, "diff_stats.csv"), index=False)

    summary_df.to_csv(
        os.path.join(output_dir, "diff_stats_unique_args_summary.csv"), index=False
    )

    with open(os.path.join(output_dir, "cpu_op_map.json"), "w") as f:
        json.dump(cpu_op_map, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "cpu_op_map_trace1.json"), "w") as f:
        json.dump(cpu_op_map_trace1.to_dict()["name"], f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "cpu_op_map_trace2.json"), "w") as f:
        json.dump(cpu_op_map_trace2.to_dict()["name"], f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "merged_tree_output.txt"), "w") as f:
        f.write(merged_tree_text + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate TraceDiff-compatible output from two semantic breakdowns"
    )
    parser.add_argument("trace_a", help="Path to trace A semantic_labels.json")
    parser.add_argument("trace_b", help="Path to trace B semantic_labels.json")
    parser.add_argument("--name-a", default="trace_a", help="Short name for trace A")
    parser.add_argument("--name-b", default="trace_b", help="Short name for trace B")
    parser.add_argument(
        "-o",
        "--output",
        default="semantic_diff_output",
        help="Output directory (default: semantic_diff_output)",
    )
    args = parser.parse_args()

    data_a = load_labels(args.trace_a)
    data_b = load_labels(args.trace_b)

    labeled_a = data_a["labeled_kernels"]
    labeled_b = data_b["labeled_kernels"]

    print(f"Trace A ({args.name_a}): {len(labeled_a)} kernels", file=sys.stderr)
    print(f"Trace B ({args.name_b}): {len(labeled_b)} kernels", file=sys.stderr)

    rows, block_id_map = build_diff_stats(labeled_a, labeled_b)
    diff_stats_df = pd.DataFrame(rows)

    # Per-LCA block total kernel time, broadcast to each row (mirrors
    # TraceDiff's busy_time, which is identical for every row in an LCA
    # group). semantic_labels.json has no timestamps, so sum-of-durations
    # is used as the busy-time proxy (exact for non-overlapping kernels).
    diff_stats_df["busy_time"] = (
        diff_stats_df.groupby(["source", "lowest_common_ancestor_id"])["kernel_time"]
        .transform("sum")
        .round(3)
    )

    summary_df = build_unique_args_summary(diff_stats_df)

    cpu_op_map, cpu_op_map_trace1, cpu_op_map_trace2 = build_cpu_op_maps(diff_stats_df)

    merged_tree_text = build_merged_tree_text(
        block_id_map, labeled_a, labeled_b, args.name_a, args.name_b
    )

    write_outputs(
        args.output,
        diff_stats_df,
        summary_df,
        cpu_op_map,
        cpu_op_map_trace1,
        cpu_op_map_trace2,
        merged_tree_text,
    )

    blocks_a = set(k["semantic_block"] for k in labeled_a)
    blocks_b = set(k["semantic_block"] for k in labeled_b)
    matched = blocks_a & blocks_b
    only_a = blocks_a - blocks_b
    only_b = blocks_b - blocks_a

    print(f"\nSemantic blocks: {len(block_id_map)} total", file=sys.stderr)
    print(f"  Matched (in both traces): {len(matched)}", file=sys.stderr)
    if only_a:
        print(f"  Only in {args.name_a}: {sorted(only_a)}", file=sys.stderr)
    if only_b:
        print(f"  Only in {args.name_b}: {sorted(only_b)}", file=sys.stderr)

    print(f"\nOutput written to: {args.output}/", file=sys.stderr)
    print(f"  diff_stats.csv ({len(diff_stats_df)} rows)", file=sys.stderr)
    print(
        f"  diff_stats_unique_args_summary.csv ({len(summary_df)} rows)",
        file=sys.stderr,
    )
    print(
        f"  cpu_op_map.json, cpu_op_map_trace1.json, cpu_op_map_trace2.json",
        file=sys.stderr,
    )
    print(f"  merged_tree_output.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
