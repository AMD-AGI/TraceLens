#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Adapter around tigertron's find_kernel_loops for discovering repeating
kernel patterns in GPU traces.

Handles multi-stream traces by detecting the primary GPU stream (most
kernels) and running pattern discovery on that stream only.  All output
indices refer to the **original** kernel list so downstream tools do not
need to know about the stream split.

Usage:
    python pattern_finder.py <extracted.json> [-o pattern.json]
"""
import argparse
import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from kernel_loop_detection import find_kernel_loops_single_stream, GPUOperation


def _kernels_to_gpu_operations(kernels):
    """Convert extracted kernel dicts to GPUOperation objects."""
    return [
        GPUOperation(name=k["name"], ts=k.get("ts", 0.0), dur=k.get("dur", 0.0))
        for k in kernels
    ]


def _detect_primary_stream(kernels):
    """Identify the primary stream when multiple streams are present.

    Returns (primary_stream_id, idx_map, secondary_indices) where
      - primary_stream_id is the stream with the most kernels (None if single-stream)
      - idx_map maps dense primary-only indices back to original indices
      - secondary_indices is a sorted list of original indices on non-primary streams
    If only one stream exists, returns (None, None, []).
    """
    stream_counts = Counter(k.get("stream_id") for k in kernels)
    non_null = {s: c for s, c in stream_counts.items() if s is not None}

    if len(non_null) <= 1:
        return None, None, []

    primary = max(non_null, key=non_null.get)
    idx_map = []
    secondary = []
    for i, k in enumerate(kernels):
        sid = k.get("stream_id")
        if sid == primary or sid is None:
            idx_map.append(i)
        else:
            secondary.append(i)

    return primary, idx_map, secondary


def find_repeating_pattern(
    extracted_data, min_loop_count=10, min_pattern_length=3, max_pattern_length=100
):
    """Discover repeating kernel patterns in a trace.

    Automatically detects multi-stream traces and runs pattern discovery
    on the primary stream only, remapping indices to the original list.

    Args:
        extracted_data: dict from extract_trace_data.py with "kernels" list
        min_loop_count: minimum number of pattern repetitions required
        min_pattern_length: minimum ops in a pattern to keep
        max_pattern_length: maximum ops before splitting into sub-patterns

    Returns:
        dict with:
          - patterns: list of discovered patterns, each a list of kernel names
          - sequences: for each pattern, list of (start_idx, end_idx) tuples
              (indices into the original kernel list)
          - pattern_labels: letter labels for each pattern (A, B, C, ...)
          - coverage: fraction of primary-stream kernels covered by patterns
          - total_kernels: total number of kernels in the trace
          - preamble_indices: kernel indices before the first pattern occurrence
              that are not covered by any pattern (original indices)
          - epilogue_indices: kernel indices after the last pattern occurrence
              that are not covered by any pattern (original indices)
          - primary_stream_id: stream used for pattern discovery (None if single-stream)
          - secondary_stream_indices: kernel indices on non-primary streams
    """
    kernels = extracted_data["kernels"]
    n = len(kernels)

    primary_sid, idx_map, secondary_indices = _detect_primary_stream(kernels)

    if idx_map is not None:
        primary_kernels = [kernels[i] for i in idx_map]
    else:
        primary_kernels = kernels
        idx_map = list(range(n))

    ops = _kernels_to_gpu_operations(primary_kernels)

    loop_structures = find_kernel_loops_single_stream(
        ops,
        min_loop_count=min_loop_count,
        min_pattern_length=min_pattern_length,
        split_pattern_length_limit=max_pattern_length,
    )

    patterns = []
    sequences = []
    pattern_labels = []
    for pidx, pat_indices in enumerate(loop_structures.patterns):
        pat_names = [loop_structures.pattern_operation_names[i] for i in pat_indices]
        patterns.append(pat_names)
        pattern_labels.append(loop_structures.get_pattern_name(pidx))

        seqs = []
        for seq in loop_structures.sequences[pidx]:
            orig_start = idx_map[seq[0]]
            orig_end = idx_map[seq[-1]]
            seqs.append((orig_start, orig_end))
        sequences.append(seqs)

    coverage = loop_structures.get_pattern_coverage()

    covered_dense = loop_structures.get_covered_indices()
    covered_orig = set(idx_map[i] for i in covered_dense)

    primary_first = n
    primary_last = -1
    if sequences:
        for start, end in sequences[0]:
            primary_first = min(primary_first, start)
            primary_last = max(primary_last, end)

    secondary_set = set(secondary_indices)
    preamble_indices = [
        i
        for i in range(primary_first)
        if i not in covered_orig and i not in secondary_set
    ]
    epilogue_indices = [
        i
        for i in range(primary_last + 1, n)
        if i not in covered_orig and i not in secondary_set
    ]

    return {
        "patterns": patterns,
        "sequences": sequences,
        "pattern_labels": pattern_labels,
        "coverage": coverage,
        "total_kernels": n,
        "preamble_indices": preamble_indices,
        "epilogue_indices": epilogue_indices,
        "primary_stream_id": primary_sid,
        "secondary_stream_indices": secondary_indices,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Discover repeating kernel patterns in a GPU trace"
    )
    parser.add_argument("extracted_json", help="Path to extracted trace data JSON")
    parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    parser.add_argument(
        "--min-loop-count",
        type=int,
        default=10,
        help="Minimum pattern repetitions (default: 10)",
    )
    parser.add_argument(
        "--min-pattern-length",
        type=int,
        default=3,
        help="Minimum ops per pattern (default: 3)",
    )
    parser.add_argument(
        "--max-pattern-length",
        type=int,
        default=100,
        help="Maximum ops before splitting (default: 100)",
    )
    args = parser.parse_args()

    with open(args.extracted_json) as f:
        extracted = json.load(f)

    result = find_repeating_pattern(
        extracted,
        min_loop_count=args.min_loop_count,
        min_pattern_length=args.min_pattern_length,
        max_pattern_length=args.max_pattern_length,
    )

    if result["primary_stream_id"] is not None:
        print(
            f"Multi-stream: primary={result['primary_stream_id']}, "
            f"{len(result['secondary_stream_indices'])} secondary kernels",
            file=sys.stderr,
        )

    for i, (pat, label) in enumerate(zip(result["patterns"], result["pattern_labels"])):
        n_seqs = len(result["sequences"][i])
        print(
            f"Pattern {label}: {len(pat)} kernels, {n_seqs} occurrences",
            file=sys.stderr,
        )

    print(
        f"Coverage: {result['coverage']:.1%} of {result['total_kernels']} kernels",
        file=sys.stderr,
    )
    print(
        f"Preamble: {len(result['preamble_indices'])} kernels, "
        f"Epilogue: {len(result['epilogue_indices'])} kernels",
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
