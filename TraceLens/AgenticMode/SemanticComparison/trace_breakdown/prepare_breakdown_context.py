#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Prepare a compact LLM context for single-trace semantic labeling.

Combines extracted.json + tree_context.json + classified.json + pattern.json
into a structured JSON packet that the breakdown agent's LLM reads to assign
functional semantic labels.

Usage:
    python prepare_breakdown_context.py \
        --extracted <dir>/extracted.json \
        --tree-context <dir>/tree_context.json \
        --classified <dir>/classified.json \
        --pattern <dir>/pattern.json \
        -o <dir>/breakdown_context.json
"""
import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict

_comparison_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "trace_comparison"
)
sys.path.insert(0, _comparison_dir)

from functional_label_catalog import FUNCTIONAL_LABEL_CATALOG


def _build_rle(kernel_indices, cls_by_idx):
    """Run-length encode kernel indices by perf_category.

    Returns list of (perf_category, [kernel_indices]).
    """
    if not kernel_indices:
        return []

    groups = []
    cur_cat = cls_by_idx.get(kernel_indices[0], {}).get("perf_category", "Others")
    cur_indices = [kernel_indices[0]]

    for idx in kernel_indices[1:]:
        cat = cls_by_idx.get(idx, {}).get("perf_category", "Others")
        if cat == cur_cat:
            cur_indices.append(idx)
        else:
            groups.append((cur_cat, cur_indices[:]))
            cur_cat = cat
            cur_indices = [idx]

    groups.append((cur_cat, cur_indices[:]))
    return groups


def _detect_period(rle_groups):
    """Find the shortest repeating period in RLE groups (by perf_category)."""
    cats = [g[0] for g in rle_groups]
    n = len(cats)
    if n < 6:
        return n

    for p in range(3, n // 2 + 1):
        prefix = cats[:p]
        matches = sum(1 for i in range(p, n) if cats[i] == prefix[i % p])
        total = n - p
        if total > 0 and matches / total > 0.85 and total >= 2 * p:
            return p

    return n


def _kernel_summary(idx, kernels, cls_by_idx, tree_ctx_by_idx):
    """Build a compact summary for one kernel."""
    k = kernels[idx]
    c = cls_by_idx.get(idx, {})
    tc = tree_ctx_by_idx.get(idx, {})
    name = k["name"]
    if len(name) > 120:
        name = name[:120] + "..."

    entry = OrderedDict()
    entry["index"] = idx
    entry["name"] = name
    entry["dur_us"] = k["dur"]
    entry["kernel_type"] = c.get("kernel_type", "Unknown")
    entry["perf_category"] = c.get("perf_category", "Others")

    cpu_op = tc.get("cpu_op_name", "")
    nn_module = tc.get("nn_module_stack", [])
    if cpu_op:
        entry["cpu_op"] = cpu_op
    if nn_module:
        entry["nn_module_stack"] = nn_module

    return entry


def _block_summary(
    group_idx, perf_category, kernel_indices, kernels, cls_by_idx, tree_ctx_by_idx
):
    """Build a compact summary for one RLE block (group of same-category kernels)."""
    block = OrderedDict()
    block["block_index"] = group_idx
    block["perf_category"] = perf_category
    block["kernel_count"] = len(kernel_indices)
    block["total_dur_us"] = round(sum(kernels[i]["dur"] for i in kernel_indices), 2)

    unique_types = sorted(
        set(cls_by_idx.get(i, {}).get("kernel_type", "Unknown") for i in kernel_indices)
    )
    block["kernel_types"] = unique_types

    unique_names = []
    seen = set()
    for i in kernel_indices:
        name = kernels[i]["name"]
        if len(name) > 120:
            name = name[:120] + "..."
        if name not in seen:
            unique_names.append(name)
            seen.add(name)
    block["unique_kernel_names"] = unique_names

    cpu_ops = set()
    nn_modules = set()
    for i in kernel_indices:
        tc = tree_ctx_by_idx.get(i, {})
        op = tc.get("cpu_op_name", "")
        nn = tc.get("nn_module_stack", [])
        if op:
            cpu_ops.add(op)
        if nn:
            nn_modules.add(" > ".join(nn))

    if cpu_ops:
        block["cpu_ops"] = sorted(cpu_ops)
    if nn_modules:
        block["nn_module_stacks"] = sorted(nn_modules)

    tree_count = sum(
        1 for i in kernel_indices if tree_ctx_by_idx.get(i, {}).get("cpu_op_name", "")
    )
    block["tree_context_count"] = tree_count

    return block


def _compute_fingerprint(layer_cycle, preamble_kernels, epilogue_kernels):
    """Compute a stable hash of the layer cycle structure.

    Two regions with the same fingerprint have identical model structure
    (same kernel names and categories in the same order), so the LLM
    would produce identical labels.  The breakdown agent uses this to
    skip redundant LLM calls for duplicate regions.
    """
    parts = []
    for block in layer_cycle:
        parts.append(block["perf_category"])
        parts.extend(block.get("unique_kernel_names", []))
        parts.append("|")
    parts.append("PRE")
    for k in preamble_kernels:
        parts.append(k.get("perf_category", ""))
        parts.append(k.get("name", ""))
    parts.append("EPI")
    for k in epilogue_kernels:
        parts.append(k.get("perf_category", ""))
        parts.append(k.get("name", ""))
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_breakdown_context(extracted, tree_context, classified, pattern):
    """Build the full breakdown context for LLM labeling."""
    kernels = extracted["kernels"]
    total_kernels = len(kernels)

    cls_by_idx = {c["index"]: c for c in classified["classified_kernels"]}
    tree_ctx_by_idx = {k["index"]: k for k in tree_context["kernels"]}

    preamble_set = set(pattern.get("preamble_indices", []))
    epilogue_set = set(pattern.get("epilogue_indices", []))
    secondary_set = set(pattern.get("secondary_stream_indices", []))
    body_indices = [
        i
        for i in range(total_kernels)
        if i not in preamble_set and i not in epilogue_set and i not in secondary_set
    ]

    body_rle = _build_rle(body_indices, cls_by_idx)
    period = _detect_period(body_rle)
    num_layers = len(body_rle) // period if period > 0 else 0

    layer_cycle = []
    for g_idx, (cat, indices) in enumerate(body_rle[:period]):
        layer_cycle.append(
            _block_summary(g_idx, cat, indices, kernels, cls_by_idx, tree_ctx_by_idx)
        )

    preamble_kernels = [
        _kernel_summary(i, kernels, cls_by_idx, tree_ctx_by_idx)
        for i in sorted(preamble_set)
    ]
    epilogue_kernels = [
        _kernel_summary(i, kernels, cls_by_idx, tree_ctx_by_idx)
        for i in sorted(epilogue_set)
    ]
    secondary_kernels_summary = {
        "count": len(secondary_set),
        "categories": {},
    }
    for i in secondary_set:
        cat = cls_by_idx.get(i, {}).get("perf_category", "Others")
        secondary_kernels_summary["categories"][cat] = (
            secondary_kernels_summary["categories"].get(cat, 0) + 1
        )

    is_graph_mode = extracted["metadata"].get("is_graph_mode", False)

    context = OrderedDict()
    context["model_info"] = {
        "graph_mode": is_graph_mode,
        "graph_launch_count": extracted["metadata"].get("graph_launch_count", 0),
        "has_python_stack": extracted["metadata"].get("has_python_stack", False),
    }
    context["trace_info"] = {
        "total_kernels": total_kernels,
        "total_kernel_time_us": extracted["metadata"]["total_kernel_time_us"],
        "source_file": extracted.get("source_file", ""),
    }
    context["tree_coverage"] = {
        "coverage": tree_context["coverage"],
        "labeled_count": tree_context["labeled_count"],
        "unlabeled_count": tree_context["unlabeled_count"],
    }
    context["pattern_info"] = {
        "period": period,
        "num_layers": num_layers,
        "total_body_rle_groups": len(body_rle),
        "preamble_count": len(preamble_set),
        "epilogue_count": len(epilogue_set),
        "secondary_stream": secondary_kernels_summary,
    }
    context["layer_cycle"] = layer_cycle
    context["preamble_kernels"] = preamble_kernels
    context["epilogue_kernels"] = epilogue_kernels
    context["fingerprint"] = _compute_fingerprint(
        layer_cycle, preamble_kernels, epilogue_kernels
    )
    context["label_catalog"] = FUNCTIONAL_LABEL_CATALOG

    return context


def main():
    parser = argparse.ArgumentParser(
        description="Prepare compact LLM context for single-trace semantic labeling"
    )
    parser.add_argument("--extracted", required=True, help="Path to extracted.json")
    parser.add_argument(
        "--tree-context", required=True, help="Path to tree_context.json"
    )
    parser.add_argument("--classified", required=True, help="Path to classified.json")
    parser.add_argument("--pattern", required=True, help="Path to pattern.json")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.extracted) as f:
        extracted = json.load(f)
    with open(args.tree_context) as f:
        tree_context = json.load(f)
    with open(args.classified) as f:
        classified = json.load(f)
    with open(args.pattern) as f:
        pattern = json.load(f)

    context = build_breakdown_context(extracted, tree_context, classified, pattern)

    with open(args.output, "w") as f:
        json.dump(context, f, indent=2)

    n_blocks = len(context["layer_cycle"])
    print(
        f"Wrote {args.output} ({n_blocks} blocks in layer cycle, "
        f"tree coverage {context['tree_coverage']['coverage']:.1%}, "
        f"fingerprint {context['fingerprint']})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
