#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Extract trace tree context (cpu_op ancestors, nn_module_stack) for each kernel.

Builds a TraceToTree from the raw trace and matches each kernel from
extracted.json to its tree event, walking the parent chain to find
cpu_op ancestors and nn_module_stack information.

The output is raw data with no semantic interpretation -- downstream
LLM steps decide how to use cpu_op names and nn_module_stack for labeling.

Usage (single region):
    python extract_tree_context.py <trace.json> <extracted.json> [-o tree_context.json]

Usage (batch -- all regions share one tree build):
    python extract_tree_context.py <trace.json> --regions-dir <output_dir>/
"""
import argparse
import json
import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_this_dir, "..", "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from TraceLens.util import DataLoader, TraceEventUtils
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree


def load_and_build_tree(trace_path):
    """Load a trace file and build a TraceToTree."""
    data = DataLoader.load_data(trace_path)
    events = data.get("traceEvents", [])
    _, non_meta_events = TraceEventUtils.split_event_list(events)
    tree = TraceToTree(non_meta_events, prune_nongpu_paths=True)
    tree.build_tree(add_python_func=True, link_fwd_bwd=False)
    return tree


def _build_ts_index(tree):
    """Build (name, ts) -> [event] index for fast matching."""
    index = {}
    for event in tree.events:
        key = (event.get("name"), event.get("ts"))
        index.setdefault(key, []).append(event)
    return index


def _find_tree_event(ts_index, name, ts, dur):
    """Find a tree event matching by name, timestamp, and duration."""
    candidates = ts_index.get((name, ts), [])
    if len(candidates) == 1:
        return candidates[0]
    for c in candidates:
        if c.get("dur") == dur:
            return c
    return candidates[0] if candidates else None


def _find_cpu_op_ancestor(tree, event):
    """Walk up parent chain to find nearest cpu_op ancestor.

    Returns (cpu_op_name, callstack_str, nn_module_stack_list).
    """
    current = event
    nn_module = event.get("nn_module_stack", [])

    while True:
        parent = tree.get_parent_event(current)
        if parent is None:
            return "", "", nn_module or []
        if tree.event_to_category(parent) == "cpu_op":
            callstack = tree.traverse_parents_and_get_callstack(parent)
            if not nn_module:
                nn_module = parent.get("nn_module_stack", [])
            return parent.get("name", ""), callstack, nn_module or []
        if not nn_module:
            nn_module = parent.get("nn_module_stack", [])
        current = parent


def extract_tree_context(tree, extracted_data, ts_index=None):
    """Match extracted kernels to tree events and extract context.

    Args:
        tree: TraceToTree instance.
        extracted_data: dict from extract_trace_data.py with "kernels" list.
        ts_index: optional pre-built (name, ts) index from _build_ts_index().
            Pass this when processing multiple regions against the same tree.

    Returns dict with per-kernel tree context, labeled/unlabeled indices,
    and coverage stats.
    """
    kernels = extracted_data["kernels"]
    if ts_index is None:
        ts_index = _build_ts_index(tree)

    labeled_indices = []
    unlabeled_indices = []
    kernel_contexts = []

    for i, kernel in enumerate(kernels):
        tree_event = _find_tree_event(
            ts_index, kernel["name"], kernel["ts"], kernel["dur"]
        )

        if tree_event is None:
            kernel_contexts.append(
                {
                    "index": i,
                    "cpu_op_name": "",
                    "cpu_op_callstack": "",
                    "nn_module_stack": [],
                }
            )
            unlabeled_indices.append(i)
            continue

        cpu_op_name, callstack, nn_module = _find_cpu_op_ancestor(tree, tree_event)

        kernel_contexts.append(
            {
                "index": i,
                "cpu_op_name": cpu_op_name,
                "cpu_op_callstack": callstack,
                "nn_module_stack": nn_module,
            }
        )

        if cpu_op_name:
            labeled_indices.append(i)
        else:
            unlabeled_indices.append(i)

    total = len(kernels)
    coverage = len(labeled_indices) / total if total > 0 else 0.0

    return {
        "source_file": extracted_data.get("source_file", ""),
        "total_kernels": total,
        "coverage": round(coverage, 4),
        "labeled_count": len(labeled_indices),
        "unlabeled_count": len(unlabeled_indices),
        "labeled_indices": labeled_indices,
        "unlabeled_indices": unlabeled_indices,
        "kernels": kernel_contexts,
    }


def _find_region_subdirs(regions_dir):
    """Find subdirectories containing extracted.json."""
    regions = []
    for name in sorted(os.listdir(regions_dir)):
        subdir = os.path.join(regions_dir, name)
        if os.path.isdir(subdir) and os.path.isfile(
            os.path.join(subdir, "extracted.json")
        ):
            regions.append((name, subdir))
    return regions


def _process_single(tree, ts_index, extracted_path, output_path):
    """Process one extracted.json against a pre-built tree and ts_index."""
    with open(extracted_path) as f:
        extracted = json.load(f)

    n_kernels = len(extracted["kernels"])
    print(f"  Matching {n_kernels} kernels...", file=sys.stderr, end=" ")
    result = extract_tree_context(tree, extracted, ts_index=ts_index)
    print(f"{result['coverage']:.1%} coverage", file=sys.stderr)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract trace tree context (cpu_op, nn_module) for each kernel"
    )
    parser.add_argument("trace", help="Path to raw trace file (.json or .json.gz)")
    parser.add_argument(
        "extracted_json",
        nargs="?",
        default=None,
        help="Path to extracted.json (single-region mode)",
    )
    parser.add_argument(
        "-o", "--output", help="Output JSON path (single-region: default stdout)"
    )
    parser.add_argument(
        "--regions-dir",
        help="Batch mode: directory containing region subdirs "
        "with extracted.json. Builds tree once, writes "
        "tree_context.json into each region subdir.",
    )
    args = parser.parse_args()

    if not args.extracted_json and not args.regions_dir:
        parser.error("provide either extracted_json or --regions-dir")

    print(f"Building trace tree from {args.trace}...", file=sys.stderr)
    tree = load_and_build_tree(args.trace)
    ts_index = _build_ts_index(tree)

    if args.regions_dir:
        regions = _find_region_subdirs(args.regions_dir)
        if not regions:
            print(
                f"No region subdirs with extracted.json in {args.regions_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Batch mode: {len(regions)} regions found", file=sys.stderr)
        for name, subdir in regions:
            print(f"Region '{name}':", file=sys.stderr)
            extracted_path = os.path.join(subdir, "extracted.json")
            output_path = os.path.join(subdir, "tree_context.json")
            _process_single(tree, ts_index, extracted_path, output_path)
        print(
            f"Wrote tree_context.json to {len(regions)} region subdirs", file=sys.stderr
        )
    else:
        with open(args.extracted_json) as f:
            extracted = json.load(f)

        print(
            f"Matching {len(extracted['kernels'])} kernels to tree events...",
            file=sys.stderr,
        )
        result = extract_tree_context(tree, extracted, ts_index=ts_index)

        print(
            f"Tree context: {result['labeled_count']}/{result['total_kernels']} "
            f"kernels have cpu_op ancestors ({result['coverage']:.1%})",
            file=sys.stderr,
        )

        output = json.dumps(result, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Wrote {args.output}", file=sys.stderr)
        else:
            print(output)


if __name__ == "__main__":
    main()
