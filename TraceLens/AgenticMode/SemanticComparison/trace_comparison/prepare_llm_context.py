#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Prepare a compact review context for LLM alignment refinement.

Reads the alignment debug output and both traces' semantic_labels.json files,
then produces a structured JSON packet that an LLM can review to identify
category corrections based on model architecture knowledge.

The output contains one entry per aligned semantic block (one layer cycle),
with kernel names, types, categories, and cross-trace matching info.

Usage:
    python prepare_llm_context.py \
        --labels-a <dir_a>/semantic_labels.json \
        --labels-b <dir_b>/semantic_labels.json \
        --debug <dir_a>/alignment_debug.txt \
        --name-a MI355 --name-b B200 \
        -o llm_review_context.json
"""
import argparse
import json
import os
import re
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(__file__))
from functional_label_catalog import FUNCTIONAL_LABEL_CATALOG


def parse_alignment_table(debug_path, name_a, name_b):
    """Parse the Alignment section from alignment_debug.txt."""
    with open(debug_path) as f:
        lines = f.readlines()

    in_alignment = False
    rows = []
    header_re = re.compile(
        r"^(?P<label>\S+)\s+"
        r"(?P<col_a>(?:---|\S+\s*\(\s*\d+\s*\)))\s+"
        r"(?P<col_b>(?:---|\S+\s*\(\s*\d+\s*\)))\s+"
        r"(?P<anchor>\S+)\s*$"
    )
    cell_re = re.compile(r"^(\S+)\s*\(\s*(\d+)\s*\)$")

    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith("Alignment  ("):
            in_alignment = True
            continue
        if in_alignment and stripped.startswith("==="):
            continue
        if in_alignment and stripped.startswith("label"):
            continue
        if in_alignment and stripped.startswith("----"):
            continue
        if in_alignment and stripped == "":
            if rows:
                break
            continue

        if not in_alignment:
            continue

        parts = stripped.split()
        if len(parts) < 4:
            continue

        label = parts[0]

        anchor_val = parts[-1]

        rest = stripped[len(label) :].strip()
        rest = (
            rest[: rest.rfind(anchor_val)].strip()
            if anchor_val != "-"
            else rest[: rest.rfind("-")].strip()
        )

        col_a_str, col_b_str = _split_two_columns(rest)

        def parse_cell(cell_str):
            cell_str = cell_str.strip()
            if cell_str == "---":
                return None, 0
            m = cell_re.match(cell_str)
            if m:
                return m.group(1), int(m.group(2))
            return cell_str, 0

        cat_a, count_a = parse_cell(col_a_str)
        cat_b, count_b = parse_cell(col_b_str)

        rows.append(
            {
                "label": label,
                "category_a": cat_a,
                "kernel_count_a": count_a,
                "category_b": cat_b,
                "kernel_count_b": count_b,
                "anchor": anchor_val if anchor_val != "-" else None,
            }
        )

    return rows


def _split_two_columns(text):
    """Split the two category columns from the alignment table row.

    Each column is either '---' or 'CategoryName (N)'.  We find the boundary
    by looking for ')' followed by whitespace and then '---' or another word.
    """
    text = text.strip()
    paren_pattern = re.compile(r"\)\s+")
    m = paren_pattern.search(text)
    if m:
        return text[: m.start() + 1].strip(), text[m.end() :].strip()

    dash_pattern = re.compile(r"---\s+")
    m = dash_pattern.search(text)
    if m:
        return text[: m.start() + 3].strip(), text[m.end() :].strip()

    mid = len(text) // 2
    return text[:mid].strip(), text[mid:].strip()


def collect_kernel_names(labels_data):
    """Build a mapping: semantic_block -> list of unique kernel names (truncated)."""
    block_names = {}
    for k in labels_data["labeled_kernels"]:
        block = k["semantic_block"]
        if block not in block_names:
            block_names[block] = set()
        name = k["name"]
        if len(name) > 120:
            name = name[:120] + "..."
        block_names[block].add(name)
    return {b: sorted(names) for b, names in block_names.items()}


def build_context(alignment_rows, labels_a, labels_b, name_a, name_b):
    """Build the full LLM review context."""
    names_a = collect_kernel_names(labels_a)
    names_b = collect_kernel_names(labels_b)

    blocks = []
    for i, row in enumerate(alignment_rows):
        label = row["label"]
        prev_label = alignment_rows[i - 1]["label"] if i > 0 else None
        next_label = (
            alignment_rows[i + 1]["label"] if i < len(alignment_rows) - 1 else None
        )

        if row["category_a"] and row["category_b"]:
            match_status = "matched"
        elif row["category_a"]:
            match_status = f"only_in_{name_a}"
        else:
            match_status = f"only_in_{name_b}"

        block = OrderedDict()
        block["label"] = label
        block["perf_category_a"] = row["category_a"]
        block["perf_category_b"] = row["category_b"]
        block["kernel_count_a"] = row["kernel_count_a"]
        block["kernel_count_b"] = row["kernel_count_b"]
        block["kernel_names_a"] = names_a.get(label, [])
        block["kernel_names_b"] = names_b.get(label, [])
        block["anchor"] = row["anchor"]
        block["match_status"] = match_status
        block["prev_block"] = prev_label
        block["next_block"] = next_label
        blocks.append(block)

    model_info = labels_a.get("model_info", {})

    cat_counts_a = {}
    cat_counts_b = {}
    for row in alignment_rows:
        if row["category_a"]:
            cat_counts_a[row["category_a"]] = (
                cat_counts_a.get(row["category_a"], 0) + row["kernel_count_a"]
            )
        if row["category_b"]:
            cat_counts_b[row["category_b"]] = (
                cat_counts_b.get(row["category_b"], 0) + row["kernel_count_b"]
            )

    context = OrderedDict()
    context["name_a"] = name_a
    context["name_b"] = name_b
    context["model_info"] = model_info
    context["summary"] = {
        "total_blocks": len(blocks),
        "matched": sum(1 for b in blocks if b["match_status"] == "matched"),
        f"only_in_{name_a}": sum(
            1 for b in blocks if b["match_status"] == f"only_in_{name_a}"
        ),
        f"only_in_{name_b}": sum(
            1 for b in blocks if b["match_status"] == f"only_in_{name_b}"
        ),
        f"category_counts_{name_a}": cat_counts_a,
        f"category_counts_{name_b}": cat_counts_b,
    }
    context["label_catalog"] = FUNCTIONAL_LABEL_CATALOG
    context["alignment_table"] = blocks
    return context


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LLM review context from alignment outputs"
    )
    parser.add_argument(
        "--labels-a", required=True, help="Path to trace A semantic_labels.json"
    )
    parser.add_argument(
        "--labels-b", required=True, help="Path to trace B semantic_labels.json"
    )
    parser.add_argument("--debug", required=True, help="Path to alignment_debug.txt")
    parser.add_argument("--name-a", default="trace_a", help="Short name for trace A")
    parser.add_argument("--name-b", default="trace_b", help="Short name for trace B")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.labels_a) as f:
        labels_a = json.load(f)
    with open(args.labels_b) as f:
        labels_b = json.load(f)

    alignment_rows = parse_alignment_table(args.debug, args.name_a, args.name_b)
    if not alignment_rows:
        print("ERROR: Could not parse alignment table from debug file", file=sys.stderr)
        sys.exit(1)

    context = build_context(
        alignment_rows, labels_a, labels_b, args.name_a, args.name_b
    )

    with open(args.output, "w") as f:
        json.dump(context, f, indent=2)

    print(
        f"Wrote {args.output} ({len(alignment_rows)} blocks, "
        f"{context['summary']['matched']} matched)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
