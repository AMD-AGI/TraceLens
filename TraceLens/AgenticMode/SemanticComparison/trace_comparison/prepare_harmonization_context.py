#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Prepare a compact review context for LLM semantic harmonization.

Takes the alignment.json (from align_semantic_blocks.py) and both traces'
semantic_labels.json files, then produces a structured JSON packet that
the harmonization agent reads to unify labels across traces.

Usage:
    python prepare_harmonization_context.py \
        --alignment alignment.json \
        --labels-a <dir_a>/semantic_labels.json \
        --labels-b <dir_b>/semantic_labels.json \
        --name-a MI355 --name-b B200 \
        -o harmonization_context.json
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(__file__))
from functional_label_catalog import FUNCTIONAL_LABEL_CATALOG


def _collect_block_details(labels_data, name):
    """Build per-block summary from semantic_labels.json."""
    block_map = OrderedDict()
    for k in labels_data["labeled_kernels"]:
        block = k["semantic_block"]
        if block not in block_map:
            block_map[block] = {
                "kernel_count": 0,
                "total_dur_us": 0.0,
                "perf_category": k.get("perf_category", "Others"),
                "nn_module": k.get("nn_module", ""),
                "kernel_names": set(),
                "kernel_types": set(),
            }
        entry = block_map[block]
        entry["kernel_count"] += 1
        entry["total_dur_us"] += k["dur"]
        kname = k["name"]
        if len(kname) > 120:
            kname = kname[:120] + "..."
        entry["kernel_names"].add(kname)
        entry["kernel_types"].add(k.get("kernel_type", "Unknown"))

    for entry in block_map.values():
        entry["kernel_names"] = sorted(entry["kernel_names"])[:10]
        entry["kernel_types"] = sorted(entry["kernel_types"])
        entry["total_dur_us"] = round(entry["total_dur_us"], 2)

    return block_map


def build_harmonization_context(alignment_data, labels_a, labels_b, name_a, name_b):
    """Build the full harmonization context for LLM review."""
    details_a = _collect_block_details(labels_a, name_a)
    details_b = _collect_block_details(labels_b, name_b)

    enriched_rows = []
    for row in alignment_data["alignment_table"]:
        enriched = OrderedDict(row)

        block_a = row.get(f"semantic_block_{name_a}")
        block_b = row.get(f"semantic_block_{name_b}")

        if block_a and block_a in details_a:
            da = details_a[block_a]
            enriched[f"kernel_types_{name_a}"] = da["kernel_types"]
            enriched[f"kernel_names_detail_{name_a}"] = da["kernel_names"]
        if block_b and block_b in details_b:
            db = details_b[block_b]
            enriched[f"kernel_types_{name_b}"] = db["kernel_types"]
            enriched[f"kernel_names_detail_{name_b}"] = db["kernel_names"]

        label_match = "same"
        if block_a and block_b:
            if block_a == block_b:
                label_match = "same"
            elif row.get(f"perf_category_{name_a}") == row.get(
                f"perf_category_{name_b}"
            ):
                label_match = "different_name_same_category"
            else:
                label_match = "different"
        elif block_a:
            label_match = f"only_in_{name_a}"
        else:
            label_match = f"only_in_{name_b}"

        enriched["label_match"] = label_match
        enriched_rows.append(enriched)

    total_a = labels_a.get("total_kernel_time_us", 0)
    total_b = labels_b.get("total_kernel_time_us", 0)

    context = OrderedDict()
    context["name_a"] = name_a
    context["name_b"] = name_b
    context["model_info_a"] = labels_a.get("model_info", {})
    context["model_info_b"] = labels_b.get("model_info", {})
    context["trace_summary"] = {
        f"total_kernel_time_us_{name_a}": total_a,
        f"total_kernel_time_us_{name_b}": total_b,
        f"total_blocks_{name_a}": len(details_a),
        f"total_blocks_{name_b}": len(details_b),
    }
    context["alignment_summary"] = alignment_data.get("summary", {})

    needs_harmonization = sum(
        1 for r in enriched_rows if r["label_match"] not in ("same",)
    )
    context["harmonization_needed"] = needs_harmonization
    context["alignment_table"] = enriched_rows
    context["label_catalog"] = FUNCTIONAL_LABEL_CATALOG

    return context


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LLM context for cross-trace semantic harmonization"
    )
    parser.add_argument(
        "--alignment",
        required=True,
        help="Path to alignment.json from align_semantic_blocks.py",
    )
    parser.add_argument(
        "--labels-a", required=True, help="Path to trace A semantic_labels.json"
    )
    parser.add_argument(
        "--labels-b", required=True, help="Path to trace B semantic_labels.json"
    )
    parser.add_argument("--name-a", default="trace_a", help="Short name for trace A")
    parser.add_argument("--name-b", default="trace_b", help="Short name for trace B")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.alignment) as f:
        alignment_data = json.load(f)
    with open(args.labels_a) as f:
        labels_a = json.load(f)
    with open(args.labels_b) as f:
        labels_b = json.load(f)

    context = build_harmonization_context(
        alignment_data, labels_a, labels_b, args.name_a, args.name_b
    )

    with open(args.output, "w") as f:
        json.dump(context, f, indent=2)

    print(
        f"Wrote {args.output} ({len(context['alignment_table'])} rows, "
        f"{context['harmonization_needed']} needing harmonization)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
