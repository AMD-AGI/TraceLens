#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Cross-trace semantic harmonization (single LLM-assisted step).

This module consolidates the harmonization cluster into one file with three
subcommands:

  align              Align two independently-labeled semantic_labels.json files
                     using Needleman-Wunsch DP and emit alignment.json.
  prepare-context    Build a compact review context (harmonization_context.json)
                     from the alignment + both traces for the LLM agent.
  apply-corrections  Apply the agent's harmonization_corrections.json back onto
                     the two semantic_labels.json files in place.

The reference label catalog (formerly semantic_label_catalog.py) is included
in-file as SEMANTIC_LABEL_CATALOG.

Usage:
    python harmonization.py align \
        --labels-a <dir_a>/semantic_labels.json \
        --labels-b <dir_b>/semantic_labels.json \
        --name-a MI355 --name-b B200 \
        -o alignment.json

    python harmonization.py prepare-context \
        --alignment alignment.json \
        --labels-a <dir_a>/semantic_labels.json \
        --labels-b <dir_b>/semantic_labels.json \
        --name-a MI355 --name-b B200 \
        -o harmonization_context.json

    python harmonization.py apply-corrections \
        --corrections harmonization_corrections.json \
        --labels-a <dir_a>/semantic_labels.json \
        --labels-b <dir_b>/semantic_labels.json \
        --name-a MI355 --name-b B200
"""

import argparse
import json
import re
import sys
from collections import OrderedDict


# ===========================================================================
# Reference label catalog (formerly semantic_label_catalog.py)
# ===========================================================================
#
# The harmonization agent uses this catalog to pick consistent, broad labels
# for semantic blocks.  Labels from the catalog are preferred; the agent may
# invent a custom label only when no catalog entry fits.
#
# To extend: add new entries to the appropriate perf_category list.

SEMANTIC_LABEL_CATALOG = {
    "GEMM": [
        "QKV Projection",
        "Output Projection",
        "FFN Gate+Up",
        "FFN Down",
        "Embedding Projection",
        "Output Head",
    ],
    "GEMM-MoE": [
        "MoE Expert GEMM",
        "MoE Shared Expert GEMM",
    ],
    "SDPA": [
        "Attention",
    ],
    "Normalization": [
        "Pre-Attn Norm",
        "Post-Attn Norm",
        "FFN Norm",
        "Final Norm",
    ],
    "Elementwise": [
        "Residual Add",
        "Activation",
        "Rotary Embedding",
        "Attention Output Gate",
        "KV Cache Store",
        "Embedding",
    ],
    "Elementwise-MoE": [
        "MoE Routing",
        "MoE Finalize",
    ],
    "Communication": [
        "AllReduce",
        "AllGather",
        "ReduceScatter",
        "AllToAll",
    ],
    "Quantization": [
        "Quantize",
        "Dequantize",
    ],
    "MemCpy": [
        "MemCpy",
    ],
}


# ===========================================================================
# align: cross-trace semantic block alignment (Needleman-Wunsch)
# ===========================================================================


def _block_key(kernel):
    """Get the grouping key for a kernel: semantic_block if present, else perf_category."""
    return kernel.get("semantic_block", kernel.get("perf_category", "Others"))


def _build_block_sequence(labels_data):
    """Build an ordered sequence of semantic blocks from labeled kernels.

    Groups consecutive kernels with the same grouping key into blocks.
    The key is semantic_block when present, otherwise perf_category.
    Returns list of dicts with block info.
    """
    kernels = labels_data["labeled_kernels"]
    if not kernels:
        return []

    def _finish_block(block_key, cat, module, indices, dur, names, cpu_ops):
        return {
            "semantic_block": block_key,
            "perf_category": cat,
            "nn_module": module,
            "kernel_count": len(indices),
            "total_dur_us": round(dur, 2),
            "kernel_indices": indices,
            "unique_kernel_names": sorted(names),
            "cpu_ops": sorted(cpu_ops - {""}),
            "layer": kernels[indices[0]].get("layer"),
        }

    blocks = []
    cur_block = _block_key(kernels[0])
    cur_cat = kernels[0].get("perf_category", "Others")
    cur_module = kernels[0].get("nn_module", "")
    cur_indices = [0]
    cur_dur = kernels[0]["dur"]
    cur_names = {kernels[0]["name"]}
    cur_cpu_ops = {kernels[0].get("cpu_op", "")}

    for i, k in enumerate(kernels[1:], 1):
        block = _block_key(k)
        if block == cur_block:
            cur_indices.append(i)
            cur_dur += k["dur"]
            cur_names.add(k["name"])
            cur_cpu_ops.add(k.get("cpu_op", ""))
        else:
            blocks.append(
                _finish_block(cur_block, cur_cat, cur_module, cur_indices, cur_dur, cur_names, cur_cpu_ops)
            )
            cur_block = block
            cur_cat = k.get("perf_category", "Others")
            cur_module = k.get("nn_module", "")
            cur_indices = [i]
            cur_dur = k["dur"]
            cur_names = {k["name"]}
            cur_cpu_ops = {k.get("cpu_op", "")}

    blocks.append(
        _finish_block(cur_block, cur_cat, cur_module, cur_indices, cur_dur, cur_names, cur_cpu_ops)
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


def _has_descriptive_semantic_block(block):
    """True if the block has a semantic_block that isn't just a perf_category fallback."""
    sb = block.get("semantic_block", "")
    return sb and sb != block.get("perf_category", "Others")


def _block_similarity(ba, bb):
    """Compute similarity score between two blocks for DP alignment.

    Scoring:
    - Same descriptive semantic_block name: +3 (strong match)
    - Same perf_category: +2
    - Same nn_module: +1
    - Overlapping kernel names: +0.5
    - Overlapping cpu_ops: +0.5
    - Different perf_category: -1
    """
    score = 0.0

    has_desc_a = _has_descriptive_semantic_block(ba)
    has_desc_b = _has_descriptive_semantic_block(bb)

    if has_desc_a and has_desc_b and ba["semantic_block"] == bb["semantic_block"]:
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

    ops_a = set(ba.get("cpu_ops", []))
    ops_b = set(bb.get("cpu_ops", []))
    if ops_a & ops_b:
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
            row[f"cpu_ops_{name_a}"] = ba.get("cpu_ops", [])
        else:
            row[f"semantic_block_{name_a}"] = None
            row[f"perf_category_{name_a}"] = None
            row[f"nn_module_{name_a}"] = None
            row[f"kernel_count_{name_a}"] = 0
            row[f"total_dur_us_{name_a}"] = 0
            row[f"kernel_names_{name_a}"] = []
            row[f"cpu_ops_{name_a}"] = []

        if bb:
            row[f"semantic_block_{name_b}"] = bb["semantic_block"]
            row[f"perf_category_{name_b}"] = bb["perf_category"]
            row[f"nn_module_{name_b}"] = bb.get("nn_module", "")
            row[f"kernel_count_{name_b}"] = bb["kernel_count"]
            row[f"total_dur_us_{name_b}"] = bb["total_dur_us"]
            names = bb.get("unique_kernel_names", [])
            row[f"kernel_names_{name_b}"] = names[:5]
            row[f"cpu_ops_{name_b}"] = bb.get("cpu_ops", [])
        else:
            row[f"semantic_block_{name_b}"] = None
            row[f"perf_category_{name_b}"] = None
            row[f"nn_module_{name_b}"] = None
            row[f"kernel_count_{name_b}"] = 0
            row[f"total_dur_us_{name_b}"] = 0
            row[f"kernel_names_{name_b}"] = []
            row[f"cpu_ops_{name_b}"] = []

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


def _build_region_blocks(labels_data, region):
    """Build a deduplicated block sequence for a region (pre/post).

    Unlike _build_block_sequence which groups by consecutive position,
    this merges all kernels sharing the same semantic_block name into one
    block.  This handles interleaving with other kernel types.
    Returns blocks in order of first kernel appearance.
    """
    kernels = labels_data["labeled_kernels"]
    block_map = OrderedDict()
    for i, k in enumerate(kernels):
        if k.get("region") != region:
            continue
        sb = k.get("semantic_block", "")
        if sb not in block_map:
            block_map[sb] = {
                "semantic_block": sb,
                "perf_category": k.get("perf_category", "Others"),
                "nn_module": k.get("nn_module", ""),
                "kernel_count": 0,
                "total_dur_us": 0.0,
                "kernel_indices": [],
                "unique_kernel_names": set(),
                "cpu_ops": set(),
                "layer": k.get("layer"),
            }
        entry = block_map[sb]
        entry["kernel_count"] += 1
        entry["total_dur_us"] += k["dur"]
        entry["kernel_indices"].append(i)
        entry["unique_kernel_names"].add(k["name"])
        entry["cpu_ops"].add(k.get("cpu_op", ""))

    blocks = []
    for entry in block_map.values():
        entry["total_dur_us"] = round(entry["total_dur_us"], 2)
        entry["unique_kernel_names"] = sorted(entry["unique_kernel_names"])
        entry["cpu_ops"] = sorted(entry["cpu_ops"] - {""})
        blocks.append(entry)
    return blocks


def _filter_by_region(labels, region):
    """Return a copy of labels_data containing only kernels in *region*."""
    filtered = dict(labels)
    filtered["labeled_kernels"] = [
        k for k in labels["labeled_kernels"]
        if k.get("region") == region
    ]
    return filtered


def run_alignment(labels_a, labels_b, name_a="trace_a", name_b="trace_b"):
    """Run the full cross-trace block alignment pipeline.

    Aligns pre-layer, cycle, and post-layer regions separately, then
    concatenates them into a single alignment table.  Secondary stream
    blocks are excluded (different GPU stream, no cross-trace match).
    """
    body_a = _filter_by_region(labels_a, "body")
    body_b = _filter_by_region(labels_b, "body")

    body_blocks_a = _build_block_sequence(body_a)
    body_blocks_b = _build_block_sequence(body_b)

    cycle_a, _, _ = _detect_layer_cycle(body_blocks_a)
    cycle_b, _, _ = _detect_layer_cycle(body_blocks_b)

    pre_a = _build_region_blocks(labels_a, "pre")
    pre_b = _build_region_blocks(labels_b, "pre")
    epi_a = _build_region_blocks(labels_a, "post")
    epi_b = _build_region_blocks(labels_b, "post")

    print(
        f"{name_a}: pre {len(pre_a)}, cycle {len(cycle_a)}, post {len(epi_a)}",
        file=sys.stderr,
    )
    print(
        f"{name_b}: pre {len(pre_b)}, cycle {len(cycle_b)}, post {len(epi_b)}",
        file=sys.stderr,
    )

    alignment = []
    if pre_a or pre_b:
        alignment.extend(needleman_wunsch(pre_a, pre_b))
    alignment.extend(needleman_wunsch(cycle_a, cycle_b))
    if epi_a or epi_b:
        alignment.extend(needleman_wunsch(epi_a, epi_b))

    matched = sum(1 for ba, bb, _, _ in alignment if ba and bb)
    print(
        f"Alignment: {matched} matched, "
        f"{sum(1 for ba, bb, _, _ in alignment if ba and not bb)} {name_a}-only, "
        f"{sum(1 for ba, bb, _, _ in alignment if not ba and bb)} {name_b}-only",
        file=sys.stderr,
    )

    return build_alignment_output(alignment, name_a, name_b, labels_a, labels_b)


# ===========================================================================
# prepare-context: compact LLM review context
# ===========================================================================

_INDEXED_RE = re.compile(r"^[A-Z][\w-]*_\d+$")


def _collect_block_details(labels_data):
    """Build per-block summary from semantic_labels.json."""
    block_map = OrderedDict()
    for k in labels_data["labeled_kernels"]:
        block = k.get("semantic_block", k.get("perf_category", "Others"))
        if block not in block_map:
            block_map[block] = {
                "kernel_count": 0,
                "total_dur_us": 0.0,
                "perf_category": k.get("perf_category", "Others"),
                "nn_module": k.get("nn_module", ""),
                "kernel_names": set(),
                "cpu_ops": set(),
                "input_dims": None,
            }
        entry = block_map[block]
        entry["kernel_count"] += 1
        entry["total_dur_us"] += k["dur"]
        kname = k["name"]
        if len(kname) > 120:
            kname = kname[:120] + "..."
        entry["kernel_names"].add(kname)
        cpu_op = k.get("cpu_op", "")
        if cpu_op:
            entry["cpu_ops"].add(cpu_op)
        if entry["input_dims"] is None:
            dims = k.get("input_dims", [])
            if dims:
                entry["input_dims"] = dims

    for entry in block_map.values():
        entry["kernel_names"] = sorted(entry["kernel_names"])[:10]
        entry["cpu_ops"] = sorted(entry["cpu_ops"])
        entry["total_dur_us"] = round(entry["total_dur_us"], 2)
        if entry["input_dims"] is None:
            entry["input_dims"] = []

    return block_map


def build_harmonization_context(alignment_data, labels_a, labels_b, name_a, name_b):
    """Build the full harmonization context for LLM review."""
    details_a = _collect_block_details(labels_a)
    details_b = _collect_block_details(labels_b)

    enriched_rows = []
    for row in alignment_data["alignment_table"]:
        enriched = OrderedDict(row)

        block_a = row.get(f"semantic_block_{name_a}")
        block_b = row.get(f"semantic_block_{name_b}")

        if block_a and block_a in details_a:
            da = details_a[block_a]
            enriched[f"kernel_names_detail_{name_a}"] = da["kernel_names"]
            enriched[f"cpu_ops_detail_{name_a}"] = da["cpu_ops"]
            if da["input_dims"]:
                enriched[f"input_dims_{name_a}"] = da["input_dims"]
        if block_b and block_b in details_b:
            db = details_b[block_b]
            enriched[f"kernel_names_detail_{name_b}"] = db["kernel_names"]
            enriched[f"cpu_ops_detail_{name_b}"] = db["cpu_ops"]
            if db["input_dims"]:
                enriched[f"input_dims_{name_b}"] = db["input_dims"]

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
    context["label_catalog"] = SEMANTIC_LABEL_CATALOG

    already_matched = 0
    rows_needing_work = []
    for r in enriched_rows:
        if r["label_match"] != "same":
            rows_needing_work.append(r)
        else:
            block = (
                r.get(f"semantic_block_{name_a}")
                or r.get(f"semantic_block_{name_b}")
                or ""
            )
            if _INDEXED_RE.match(block):
                r["label_match"] = "needs_rename"
                rows_needing_work.append(r)
            else:
                already_matched += 1

    context["already_matched"] = already_matched
    context["harmonization_needed"] = len(rows_needing_work)
    context["alignment_table"] = rows_needing_work

    return context


# ===========================================================================
# apply-corrections: write the agent's corrections back to the labels files
# ===========================================================================


def _apply_renames(kernels, renames, trace_name):
    """Rename existing semantic_block values."""
    count = 0
    rename_map = {}
    for r in renames:
        if r.get("trace") and r["trace"] != trace_name:
            continue
        rename_map[r["old_semantic_block"]] = r

    for kernel in kernels:
        old = kernel.get("semantic_block", "")
        if old in rename_map:
            r = rename_map[old]
            kernel["semantic_block"] = r["new_semantic_block"]
            if "new_perf_category" in r and r["new_perf_category"]:
                kernel["perf_category"] = r["new_perf_category"]
            if "new_nn_module" in r and r["new_nn_module"]:
                kernel["nn_module"] = r["new_nn_module"]
            count += 1

    return count


def _apply_category_corrections(kernels, corrections, trace_name):
    """Apply perf_category corrections."""
    count = 0
    corr_map = {}
    for c in corrections:
        if c.get("trace") and c["trace"] != trace_name:
            continue
        corr_map[c["semantic_block"]] = c["new_perf_category"]

    for kernel in kernels:
        block = kernel.get("semantic_block", "")
        if block in corr_map:
            kernel["perf_category"] = corr_map[block]
            count += 1

    return count


def _apply_kernel_reassignments(kernels, reassignments, trace_name):
    """Move individual kernels between semantic blocks."""
    count = 0
    for ra in reassignments:
        if ra.get("trace") and ra["trace"] != trace_name:
            continue
        indices = set(ra["kernel_indices"])
        new_block = ra["to_semantic_block"]
        for kernel in kernels:
            if kernel["index"] in indices:
                kernel["semantic_block"] = new_block
                count += 1

    return count


def apply_corrections(labels_path, corrections_data, trace_name):
    """Apply all correction types to a semantic_labels.json file."""
    with open(labels_path) as f:
        data = json.load(f)

    kernels = data["labeled_kernels"]
    stats = {}

    renames = corrections_data.get("label_renames", [])
    if renames:
        stats["renames"] = _apply_renames(kernels, renames, trace_name)

    cat_corrections = corrections_data.get("category_corrections", [])
    if cat_corrections:
        stats["category_corrections"] = _apply_category_corrections(
            kernels, cat_corrections, trace_name
        )

    reassignments = corrections_data.get("kernel_reassignments", [])
    if reassignments:
        stats["kernel_reassignments"] = _apply_kernel_reassignments(
            kernels, reassignments, trace_name
        )

    with open(labels_path, "w") as f:
        json.dump(data, f, indent=2)

    return stats


# ===========================================================================
# CLI
# ===========================================================================


def cmd_align(args):
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


def cmd_prepare_context(args):
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


def cmd_apply_corrections(args):
    with open(args.corrections) as f:
        corrections_data = json.load(f)

    stats_a = apply_corrections(args.labels_a, corrections_data, args.name_a)
    stats_b = apply_corrections(args.labels_b, corrections_data, args.name_b)

    total_a = sum(stats_a.values())
    total_b = sum(stats_b.values())

    if total_a == 0 and total_b == 0:
        print("No corrections applied.", file=sys.stderr)
    else:
        print(
            f"Applied to {args.name_a}: {stats_a}",
            file=sys.stderr,
        )
        print(
            f"Applied to {args.name_b}: {stats_b}",
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Cross-trace semantic harmonization (align / prepare-context / apply-corrections)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_align = sub.add_parser(
        "align",
        help="Align semantic blocks across two independently-labeled traces",
    )
    p_align.add_argument(
        "--labels-a", required=True, help="Path to trace A semantic_labels.json"
    )
    p_align.add_argument(
        "--labels-b", required=True, help="Path to trace B semantic_labels.json"
    )
    p_align.add_argument("--name-a", default="trace_a", help="Short name for trace A")
    p_align.add_argument("--name-b", default="trace_b", help="Short name for trace B")
    p_align.add_argument(
        "-o", "--output", required=True, help="Output alignment JSON path"
    )
    p_align.set_defaults(func=cmd_align)

    p_prep = sub.add_parser(
        "prepare-context",
        help="Prepare LLM context for cross-trace semantic harmonization",
    )
    p_prep.add_argument(
        "--alignment", required=True, help="Path to alignment.json from the align step"
    )
    p_prep.add_argument(
        "--labels-a", required=True, help="Path to trace A semantic_labels.json"
    )
    p_prep.add_argument(
        "--labels-b", required=True, help="Path to trace B semantic_labels.json"
    )
    p_prep.add_argument("--name-a", default="trace_a", help="Short name for trace A")
    p_prep.add_argument("--name-b", default="trace_b", help="Short name for trace B")
    p_prep.add_argument("-o", "--output", required=True, help="Output JSON path")
    p_prep.set_defaults(func=cmd_prepare_context)

    p_apply = sub.add_parser(
        "apply-corrections",
        help="Apply harmonization corrections to semantic_labels.json files",
    )
    p_apply.add_argument(
        "--corrections", required=True, help="Path to harmonization_corrections.json"
    )
    p_apply.add_argument(
        "--labels-a", required=True, help="Path to trace A semantic_labels.json"
    )
    p_apply.add_argument(
        "--labels-b", required=True, help="Path to trace B semantic_labels.json"
    )
    p_apply.add_argument("--name-a", default="trace_a", help="Short name for trace A")
    p_apply.add_argument("--name-b", default="trace_b", help="Short name for trace B")
    p_apply.set_defaults(func=cmd_apply_corrections)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
