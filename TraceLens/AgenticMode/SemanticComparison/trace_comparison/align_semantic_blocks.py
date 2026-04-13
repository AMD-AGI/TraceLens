#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Cross-trace semantic block alignment.

Takes two independently-labeled semantic_labels.json files and aligns
their semantic blocks using Needleman-Wunsch DP. Unlike align_and_label.py
(which operates on raw perf_category RLE), this script aligns blocks that
already have functional semantic labels.

The alignment is used by the harmonization agent to unify labels across
traces and ensure cross-trace consistency.

Usage:
    python align_semantic_blocks.py \
        --labels-a <dir_a>/semantic_labels.json \
        --labels-b <dir_b>/semantic_labels.json \
        --name-a MI355 --name-b B200 \
        -o alignment.json
"""
import argparse
import json
import os
import sys
from collections import OrderedDict


def _build_block_sequence(labels_data):
    """Build an ordered sequence of semantic blocks from labeled kernels.

    Groups consecutive kernels with the same semantic_block into blocks.
    Returns list of dicts with block info.
    """
    kernels = labels_data["labeled_kernels"]
    if not kernels:
        return []

    blocks = []
    cur_block = kernels[0]["semantic_block"]
    cur_cat = kernels[0].get("perf_category", "Others")
    cur_module = kernels[0].get("nn_module", "")
    cur_indices = [0]
    cur_dur = kernels[0]["dur"]
    cur_names = {kernels[0]["name"]}

    for i, k in enumerate(kernels[1:], 1):
        block = k["semantic_block"]
        if block == cur_block:
            cur_indices.append(i)
            cur_dur += k["dur"]
            cur_names.add(k["name"])
        else:
            blocks.append(
                {
                    "semantic_block": cur_block,
                    "perf_category": cur_cat,
                    "nn_module": cur_module,
                    "kernel_count": len(cur_indices),
                    "total_dur_us": round(cur_dur, 2),
                    "kernel_indices": cur_indices,
                    "unique_kernel_names": sorted(cur_names),
                    "layer": kernels[cur_indices[0]].get("layer"),
                }
            )
            cur_block = block
            cur_cat = k.get("perf_category", "Others")
            cur_module = k.get("nn_module", "")
            cur_indices = [i]
            cur_dur = k["dur"]
            cur_names = {k["name"]}

    blocks.append(
        {
            "semantic_block": cur_block,
            "perf_category": cur_cat,
            "nn_module": cur_module,
            "kernel_count": len(cur_indices),
            "total_dur_us": round(cur_dur, 2),
            "kernel_indices": cur_indices,
            "unique_kernel_names": sorted(cur_names),
            "layer": kernels[cur_indices[0]].get("layer"),
        }
    )
    return blocks


def _detect_layer_cycle(blocks):
    """Extract one layer cycle from the block sequence.

    Finds the repeating portion by looking for the first block with
    layer=0 and then finding where layer=1 starts.
    Returns (cycle_blocks, cycle_start_idx, cycle_end_idx).
    """
    first_layer0 = None
    first_layer1 = None

    for i, b in enumerate(blocks):
        layer = b.get("layer")
        if layer == 0 and first_layer0 is None:
            first_layer0 = i
        elif layer == 1 and first_layer0 is not None and first_layer1 is None:
            first_layer1 = i
            break

    if first_layer0 is not None and first_layer1 is not None:
        return blocks[first_layer0:first_layer1], first_layer0, first_layer1
    if first_layer0 is not None:
        return blocks[first_layer0:], first_layer0, len(blocks)
    return blocks, 0, len(blocks)


def _block_similarity(ba, bb):
    """Compute similarity score between two blocks for DP alignment.

    Scoring:
    - Same semantic_block name: +3 (strong match)
    - Same perf_category: +2
    - Same nn_module: +1
    - Overlapping kernel names: +0.5
    - Different perf_category: -1
    """
    score = 0.0

    if ba["semantic_block"] == bb["semantic_block"]:
        score += 3.0
    elif ba["perf_category"] == bb["perf_category"]:
        score += 2.0
    else:
        base_a = ba["perf_category"].split("-")[0]
        base_b = bb["perf_category"].split("-")[0]
        if base_a == base_b:
            score += 1.5
        else:
            score -= 1.0

    if ba.get("nn_module") and ba["nn_module"] == bb.get("nn_module"):
        score += 1.0

    names_a = set(ba.get("unique_kernel_names", []))
    names_b = set(bb.get("unique_kernel_names", []))
    if names_a & names_b:
        score += 0.5

    return score


def needleman_wunsch(blocks_a, blocks_b, gap_penalty=-0.5):
    """Align two block sequences using Needleman-Wunsch DP.

    Returns list of (block_a_or_None, block_b_or_None, idx_a, idx_b) tuples.
    """
    na = len(blocks_a)
    nb = len(blocks_b)

    score = [[0.0] * (nb + 1) for _ in range(na + 1)]
    trace = [[0] * (nb + 1) for _ in range(na + 1)]

    for i in range(1, na + 1):
        score[i][0] = score[i - 1][0] + gap_penalty
        trace[i][0] = 1
    for j in range(1, nb + 1):
        score[0][j] = score[0][j - 1] + gap_penalty
        trace[0][j] = 2

    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            s = _block_similarity(blocks_a[i - 1], blocks_b[j - 1])
            diag = score[i - 1][j - 1] + s
            up = score[i - 1][j] + gap_penalty
            left = score[i][j - 1] + gap_penalty

            best = max(diag, up, left)
            score[i][j] = best
            if best == diag:
                trace[i][j] = 0
            elif best == up:
                trace[i][j] = 1
            else:
                trace[i][j] = 2

    alignment = []
    i, j = na, nb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == 0:
            alignment.append((blocks_a[i - 1], blocks_b[j - 1], i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or trace[i][j] == 1):
            alignment.append((blocks_a[i - 1], None, i - 1, None))
            i -= 1
        else:
            alignment.append((None, blocks_b[j - 1], None, j - 1))
            j -= 1

    alignment.reverse()
    return alignment


def build_alignment_output(alignment, name_a, name_b, labels_a, labels_b):
    """Build the alignment.json output."""
    rows = []
    for ba, bb, idx_a, idx_b in alignment:
        row = OrderedDict()

        if ba and bb:
            row["match_status"] = "matched"
        elif ba:
            row["match_status"] = f"only_in_{name_a}"
        else:
            row["match_status"] = f"only_in_{name_b}"

        if ba:
            row[f"semantic_block_{name_a}"] = ba["semantic_block"]
            row[f"perf_category_{name_a}"] = ba["perf_category"]
            row[f"nn_module_{name_a}"] = ba.get("nn_module", "")
            row[f"kernel_count_{name_a}"] = ba["kernel_count"]
            row[f"total_dur_us_{name_a}"] = ba["total_dur_us"]
            names = ba.get("unique_kernel_names", [])
            row[f"kernel_names_{name_a}"] = names[:5]
        else:
            row[f"semantic_block_{name_a}"] = None
            row[f"perf_category_{name_a}"] = None
            row[f"nn_module_{name_a}"] = None
            row[f"kernel_count_{name_a}"] = 0
            row[f"total_dur_us_{name_a}"] = 0
            row[f"kernel_names_{name_a}"] = []

        if bb:
            row[f"semantic_block_{name_b}"] = bb["semantic_block"]
            row[f"perf_category_{name_b}"] = bb["perf_category"]
            row[f"nn_module_{name_b}"] = bb.get("nn_module", "")
            row[f"kernel_count_{name_b}"] = bb["kernel_count"]
            row[f"total_dur_us_{name_b}"] = bb["total_dur_us"]
            names = bb.get("unique_kernel_names", [])
            row[f"kernel_names_{name_b}"] = names[:5]
        else:
            row[f"semantic_block_{name_b}"] = None
            row[f"perf_category_{name_b}"] = None
            row[f"nn_module_{name_b}"] = None
            row[f"kernel_count_{name_b}"] = 0
            row[f"total_dur_us_{name_b}"] = 0
            row[f"kernel_names_{name_b}"] = []

        rows.append(row)

    matched = sum(1 for r in rows if r["match_status"] == "matched")
    only_a = sum(1 for r in rows if r["match_status"] == f"only_in_{name_a}")
    only_b = sum(1 for r in rows if r["match_status"] == f"only_in_{name_b}")

    output = OrderedDict()
    output["name_a"] = name_a
    output["name_b"] = name_b
    output["summary"] = {
        "total_aligned_rows": len(rows),
        "matched": matched,
        f"only_in_{name_a}": only_a,
        f"only_in_{name_b}": only_b,
    }
    output["model_info_a"] = labels_a.get("model_info", {})
    output["model_info_b"] = labels_b.get("model_info", {})
    output["alignment_table"] = rows
    return output


def run_alignment(labels_a, labels_b, name_a="trace_a", name_b="trace_b"):
    """Run the full cross-trace block alignment pipeline."""
    blocks_a = _build_block_sequence(labels_a)
    blocks_b = _build_block_sequence(labels_b)

    cycle_a, start_a, end_a = _detect_layer_cycle(blocks_a)
    cycle_b, start_b, end_b = _detect_layer_cycle(blocks_b)

    print(
        f"{name_a}: {len(blocks_a)} total blocks, " f"cycle has {len(cycle_a)} blocks",
        file=sys.stderr,
    )
    print(
        f"{name_b}: {len(blocks_b)} total blocks, " f"cycle has {len(cycle_b)} blocks",
        file=sys.stderr,
    )

    alignment = needleman_wunsch(cycle_a, cycle_b)

    matched = sum(1 for ba, bb, _, _ in alignment if ba and bb)
    print(
        f"Alignment: {matched} matched, "
        f"{sum(1 for ba, bb, _, _ in alignment if ba and not bb)} {name_a}-only, "
        f"{sum(1 for ba, bb, _, _ in alignment if not ba and bb)} {name_b}-only",
        file=sys.stderr,
    )

    return build_alignment_output(alignment, name_a, name_b, labels_a, labels_b)


def main():
    parser = argparse.ArgumentParser(
        description="Align semantic blocks across two independently-labeled traces"
    )
    parser.add_argument(
        "--labels-a", required=True, help="Path to trace A semantic_labels.json"
    )
    parser.add_argument(
        "--labels-b", required=True, help="Path to trace B semantic_labels.json"
    )
    parser.add_argument("--name-a", default="trace_a", help="Short name for trace A")
    parser.add_argument("--name-b", default="trace_b", help="Short name for trace B")
    parser.add_argument(
        "-o", "--output", required=True, help="Output alignment JSON path"
    )
    args = parser.parse_args()

    with open(args.labels_a) as f:
        labels_a = json.load(f)
    with open(args.labels_b) as f:
        labels_b = json.load(f)

    result = run_alignment(labels_a, labels_b, args.name_a, args.name_b)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"Wrote {args.output} ({result['summary']['total_aligned_rows']} rows, "
        f"{result['summary']['matched']} matched)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
