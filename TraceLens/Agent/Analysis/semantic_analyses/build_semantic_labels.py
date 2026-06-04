#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Build semantic_labels.json deterministically from breakdown artifacts.

Combines extracted.json + classified.json + pattern.json + tree_context.json
into the labeled kernel format consumed by the comparison pipeline.

Each kernel gets a positionally-indexed ``semantic_block`` name (e.g.
``GEMM_0``, ``GEMM_1``, ``Normalization_0``) derived from its position
within the repeating layer cycle.  A global counter ensures block names
are unique across regions (pre-layer, body, post-layer, secondary).
The harmonization agent later renames these to descriptive labels
(e.g. ``QKV Projection``).

Per-kernel output fields:
    index, name, dur, perf_category, semantic_block, region, nn_module,
    cpu_op, input_dims, layer.

Usage:
    python build_semantic_labels.py extracted.json classified.json pattern.json \
        --tree-context tree_context.json [-o semantic_labels.json]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import build_rle, detect_period


def _deepest_module(nn_module_stack):
    """Extract the deepest nn.Module name from the stack list.

    Input like: ["nn.Module: WanTransformer3DModel > nn.Module: WanTransformerBlock"]
    Returns: "WanTransformerBlock"
    """
    if not nn_module_stack:
        return ""
    last_entry = nn_module_stack[-1]
    parts = last_entry.split(" > ")
    deepest = parts[-1]
    if deepest.startswith("nn.Module: "):
        deepest = deepest[len("nn.Module: "):]
    return deepest


def _build_cycle_names(body_rle, period, cat_counter):
    """Build positionally-indexed block names for one layer cycle.

    Uses *cat_counter* (mutated in-place) so indices are globally unique
    across regions.  Returns a list of length ``period`` like
    ``["GEMM_2", "Normalization_1", "GEMM_3", "SDPA_0", ...]``.
    """
    if period <= 0:
        return []
    cycle_names = []
    for cat, _count, _indices, _types in body_rle[:period]:
        idx = cat_counter.get(cat, 0)
        cat_counter[cat] = idx + 1
        cycle_names.append(f"{cat}_{idx}")
    return cycle_names


def _build_region_block_names(index_set, cls_by_idx, cat_counter):
    """Build indexed block names for a region of kernels.

    Groups consecutive kernels by perf_category and assigns globally
    unique ``Category_N`` names using *cat_counter* (mutated in-place).
    Returns a mapping ``{kernel_index: "Category_N"}``.
    """
    if not index_set:
        return {}
    sorted_indices = sorted(index_set)
    groups = []
    cur_cat = cls_by_idx.get(sorted_indices[0], {}).get("perf_category", "Others")
    cur_group = [sorted_indices[0]]
    for idx in sorted_indices[1:]:
        cat = cls_by_idx.get(idx, {}).get("perf_category", "Others")
        if cat == cur_cat:
            cur_group.append(idx)
        else:
            groups.append((cur_cat, cur_group))
            cur_cat = cat
            cur_group = [idx]
    groups.append((cur_cat, cur_group))

    result = {}
    for cat, indices in groups:
        n = cat_counter.get(cat, 0)
        cat_counter[cat] = n + 1
        block_name = f"{cat}_{n}"
        for idx in indices:
            result[idx] = block_name
    return result


def build_labels(extracted, classified, pattern, tree_context=None):
    """Build semantic_labels.json from breakdown artifacts."""
    kernels = extracted["kernels"]
    total_kernels = len(kernels)

    cls_by_idx = {c["index"]: c for c in classified["classified_kernels"]}
    tree_by_idx = {}
    if tree_context:
        tree_by_idx = {k["index"]: k for k in tree_context["kernels"]}

    preamble_set = set(pattern.get("preamble_indices", []))
    epilogue_set = set(pattern.get("epilogue_indices", []))
    secondary_set = set(pattern.get("secondary_stream_indices", []))
    body_indices = [
        i
        for i in range(total_kernels)
        if i not in preamble_set and i not in epilogue_set and i not in secondary_set
    ]

    body_rle = build_rle(body_indices, cls_by_idx)
    period = detect_period(body_rle)
    num_layers = len(body_rle) // period if period > 0 else 0

    cat_counter = {}

    preamble_blocks = _build_region_block_names(
        preamble_set, cls_by_idx, cat_counter
    )
    cycle_names = _build_cycle_names(body_rle, period, cat_counter)
    epilogue_blocks = _build_region_block_names(
        epilogue_set, cls_by_idx, cat_counter
    )
    secondary_blocks = _build_region_block_names(
        secondary_set, cls_by_idx, cat_counter
    )

    body_index_to_rle_group = {}
    for g_idx, (_cat, _count, indices, _types) in enumerate(body_rle):
        for idx in indices:
            body_index_to_rle_group[idx] = g_idx

    labeled_kernels = []
    for i in range(total_kernels):
        k = kernels[i]
        c = cls_by_idx.get(i, {})
        tc = tree_by_idx.get(i, {})

        entry = {
            "index": i,
            "name": k["name"],
            "dur": k["dur"],
            "perf_category": c.get("perf_category", "Others"),
            "nn_module": _deepest_module(tc.get("nn_module_stack", [])),
            "cpu_op": tc.get("cpu_op_name", ""),
            "input_dims": tc.get("input_dims", []),
        }

        if i in body_index_to_rle_group:
            g_idx = body_index_to_rle_group[i]
            entry["region"] = "body"
            entry["layer"] = g_idx // period if period > 0 else 0
            entry["semantic_block"] = cycle_names[g_idx % period]
        elif i in preamble_blocks:
            entry["region"] = "pre"
            entry["layer"] = None
            entry["semantic_block"] = preamble_blocks[i]
        elif i in epilogue_blocks:
            entry["region"] = "post"
            entry["layer"] = None
            entry["semantic_block"] = epilogue_blocks[i]
        elif i in secondary_blocks:
            entry["region"] = "secondary"
            entry["layer"] = None
            entry["semantic_block"] = secondary_blocks[i]
        else:
            entry["region"] = "body"
            entry["layer"] = None
            entry["semantic_block"] = c.get("perf_category", "Others") + "_0"

        labeled_kernels.append(entry)

    result = {
        "source_file": extracted.get("source_file", ""),
        "total_kernel_time_us": round(
            extracted.get("metadata", {}).get("total_kernel_time_us", 0), 2
        ),
        "model_info": {
            "num_layers": num_layers,
            "period": period,
            "graph_mode": extracted.get("metadata", {}).get("is_graph_mode", False),
        },
        "labeled_kernels": labeled_kernels,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build semantic_labels.json deterministically from breakdown artifacts"
    )
    parser.add_argument("extracted_json", help="Path to extracted.json")
    parser.add_argument("classified_json", help="Path to classified.json")
    parser.add_argument("pattern_json", help="Path to pattern.json")
    parser.add_argument(
        "--tree-context", help="Path to tree_context.json (optional, enriches nn_module and cpu_op)"
    )
    parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    with open(args.extracted_json) as f:
        extracted = json.load(f)
    with open(args.classified_json) as f:
        classified = json.load(f)
    with open(args.pattern_json) as f:
        pattern = json.load(f)

    tree_context = None
    if args.tree_context:
        with open(args.tree_context) as f:
            tree_context = json.load(f)

    result = build_labels(extracted, classified, pattern, tree_context)

    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        n = len(result["labeled_kernels"])
        info = result["model_info"]
        print(
            f"Wrote {args.output} ({n} kernels, {info['num_layers']} layers, "
            f"period {info['period']})",
            file=sys.stderr,
        )
    else:
        print(output)


if __name__ == "__main__":
    main()
