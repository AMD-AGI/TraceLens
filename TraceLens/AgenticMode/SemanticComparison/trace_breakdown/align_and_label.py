#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Cross-trace alignment and labeling.

Aligns kernel sequences from two GPU traces using dynamic programming on
perf_category run-length encoded groups, producing consistent semantic_block
labels for both traces simultaneously.

Pipeline:
  1. Reconstruct one "super-cycle" per trace from the body perf_category RLE
  2. Needleman-Wunsch DP alignment of the two super-cycle RLE sequences
  3. Assign labels: anchor kernel types get named labels, others get indexed
  4. Expand labels to full trace via greedy perf_category matching
  5. Label secondary-stream kernels as {category}_secondary
  6. Label preamble / epilogue

Usage:
    python align_and_label.py --dir-a output/MI355/decode_only_3 \
                              --dir-b output/B200/decode_only_3
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_kernels import classify_kernel
from category_mappings import get_perf_category as _label_perf_category

# ---------------------------------------------------------------------------
# Anchor kernel_type -> semantic block label.
# Kernel types that unambiguously identify a specific operation within a
# layer.  Based purely on kernel name regex — no model-type knowledge.
# ---------------------------------------------------------------------------

ANCHOR_MAP = {
    "Attention": "Attention",
    "Linear Attention": "Attention",
    "GDN Gating": "Attention",
    "KV Cache Store": "KV_Cache_Store",
    "Rotary Embedding": "Rotary_Embedding",
    "MoE Routing": "MoE_Routing",
    "MoE Finalize": "MoE_Finalize",
    "MoE Quantize": "MoE_Quantize",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_trace_dir(dir_path):
    """Load extracted, pattern, and classified data from a trace directory."""
    with open(os.path.join(dir_path, "extracted.json")) as f:
        extracted = json.load(f)
    with open(os.path.join(dir_path, "pattern.json")) as f:
        pattern = json.load(f)
    with open(os.path.join(dir_path, "classified.json")) as f:
        classified = json.load(f)

    cls_by_idx = {c["index"]: c for c in classified["classified_kernels"]}
    return extracted, pattern, classified, cls_by_idx


# ---------------------------------------------------------------------------
# RLE construction
# ---------------------------------------------------------------------------


def build_rle(kernel_indices, cls_by_idx):
    """Run-length encode kernel indices by perf_category.

    Returns list of (perf_category, count, [kernel_indices], [kernel_types]).
    """
    if not kernel_indices:
        return []

    def _cls(idx):
        c = cls_by_idx.get(idx, {})
        return c.get("perf_category", "Others"), c.get("kernel_type", "Unknown")

    first_cat, first_kt = _cls(kernel_indices[0])
    groups = []
    cur_cat = first_cat
    cur_indices = [kernel_indices[0]]
    cur_types = [first_kt]

    for idx in kernel_indices[1:]:
        cat, kt = _cls(idx)
        if cat == cur_cat:
            cur_indices.append(idx)
            cur_types.append(kt)
        else:
            groups.append((cur_cat, len(cur_indices), cur_indices[:], cur_types[:]))
            cur_cat = cat
            cur_indices = [idx]
            cur_types = [kt]

    groups.append((cur_cat, len(cur_indices), cur_indices[:], cur_types[:]))
    return groups


# ---------------------------------------------------------------------------
# Period detection
# ---------------------------------------------------------------------------


def detect_period(rle_groups):
    """Find the shortest repeating period in an RLE group sequence.

    Returns the period length (number of RLE groups per super-cycle).
    """
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


# ---------------------------------------------------------------------------
# Anchor helpers
# ---------------------------------------------------------------------------


def get_group_anchor(group):
    """Return anchor label if any kernel in the group has an anchor type."""
    for kt in group[3]:
        if kt in ANCHOR_MAP:
            return ANCHOR_MAP[kt]
    return None


# ---------------------------------------------------------------------------
# Needleman-Wunsch alignment
# ---------------------------------------------------------------------------


def needleman_wunsch(
    groups_a, groups_b, match_score=2, mismatch=-1, gap=-0.5, anchor_bonus=1
):
    """Align two RLE group sequences using Needleman-Wunsch DP.

    Returns list of (group_a_or_None, group_b_or_None) pairs.
    """
    na = len(groups_a)
    nb = len(groups_b)

    score = [[0.0] * (nb + 1) for _ in range(na + 1)]
    trace = [[0] * (nb + 1) for _ in range(na + 1)]

    for i in range(1, na + 1):
        score[i][0] = score[i - 1][0] + gap
        trace[i][0] = 1
    for j in range(1, nb + 1):
        score[0][j] = score[0][j - 1] + gap
        trace[0][j] = 2

    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            ga, gb = groups_a[i - 1], groups_b[j - 1]
            if ga[0] == gb[0]:
                s = match_score
                aa = get_group_anchor(ga)
                ab = get_group_anchor(gb)
                if aa and ab and aa == ab:
                    s += anchor_bonus
            else:
                s = mismatch

            diag = score[i - 1][j - 1] + s
            up = score[i - 1][j] + gap
            left = score[i][j - 1] + gap

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
            alignment.append((groups_a[i - 1], groups_b[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or trace[i][j] == 1):
            alignment.append((groups_a[i - 1], None))
            i -= 1
        else:
            alignment.append((None, groups_b[j - 1]))
            j -= 1

    alignment.reverse()
    return alignment


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------


def assign_labels(alignment):
    """Assign semantic_block labels to aligned blocks.

    Anchor groups get named labels, others get indexed labels per category.
    """
    labels = []
    cat_counters = {}
    anchor_counters = {}

    for ga, gb in alignment:
        cat = (ga or gb)[0]

        anchor = None
        if ga is not None:
            anchor = get_group_anchor(ga)
        if anchor is None and gb is not None:
            anchor = get_group_anchor(gb)

        if anchor:
            idx = anchor_counters.get(anchor, 0)
            label = anchor if idx == 0 else f"{anchor}_{idx}"
            anchor_counters[anchor] = idx + 1
        else:
            idx = cat_counters.get(cat, 0)
            label = f"{cat}_{idx}"
            cat_counters[cat] = idx + 1

        labels.append(label)

    return labels


def build_reference_sequence(alignment, labels, side):
    """Extract the reference perf_category sequence and labels for one trace.

    Only includes entries where this trace has a group (non-gap).
    """
    ref_cats = []
    ref_labels = []
    for (ga, gb), label in zip(alignment, labels):
        group = ga if side == "a" else gb
        if group is not None:
            ref_cats.append(group[0])
            ref_labels.append(label)
    return ref_cats, ref_labels


# ---------------------------------------------------------------------------
# Label expansion to full trace
# ---------------------------------------------------------------------------


def expand_labels_to_trace(ref_cats, ref_labels, body_indices, cls_by_idx):
    """Apply super-cycle labels to all body kernels via greedy matching.

    Walks through the body RLE groups, matching each group's perf_category
    against the reference sequence.  The reference wraps around at each
    layer boundary, incrementing the layer counter.

    Groups that do not match any reference position are labeled
    ``{category}_extra``.
    """
    kernel_labels = {}
    kernel_layers = {}

    if not ref_cats or not body_indices:
        return kernel_labels, kernel_layers

    ref_len = len(ref_cats)
    pos = 0
    layer = 0

    body_rle = build_rle(body_indices, cls_by_idx)

    for group in body_rle:
        cat = group[0]

        saved_pos = pos
        saved_layer = layer
        search_count = 0
        while ref_cats[pos] != cat and search_count < ref_len:
            pos = (pos + 1) % ref_len
            if pos == 0:
                layer += 1
            search_count += 1

        if search_count >= ref_len:
            pos = saved_pos
            layer = saved_layer
            label = f"{cat}_extra"
        else:
            label = ref_labels[pos]

        for kernel_idx in group[2]:
            kernel_labels[kernel_idx] = label
            kernel_layers[kernel_idx] = layer

        if search_count < ref_len:
            pos = (pos + 1) % ref_len
            if pos == 0:
                layer += 1

    return kernel_labels, kernel_layers


# ---------------------------------------------------------------------------
# Preamble / epilogue / secondary-stream labeling
# ---------------------------------------------------------------------------


def label_preamble_epilogue(preamble_indices, epilogue_indices, cls_by_idx):
    """Label preamble/epilogue kernels by their classified perf_category."""
    labels = {}
    layers = {}
    for idx in preamble_indices:
        cat = cls_by_idx.get(idx, {}).get("perf_category", "Others")
        labels[idx] = f"{cat}_preamble"
        layers[idx] = None
    for idx in epilogue_indices:
        cat = cls_by_idx.get(idx, {}).get("perf_category", "Others")
        labels[idx] = f"{cat}_epilogue"
        layers[idx] = None
    return labels, layers


def label_secondary_stream(secondary_indices, cls_by_idx):
    """Label secondary-stream kernels by their perf_category."""
    labels = {}
    layers = {}
    for idx in secondary_indices:
        c = cls_by_idx.get(idx, {})
        cat = c.get("perf_category", "Others")
        labels[idx] = f"{cat}_secondary"
        layers[idx] = None
    return labels, layers


# ---------------------------------------------------------------------------
# Type-category validation
# ---------------------------------------------------------------------------


def _base_category(cat):
    """Strip sub-category suffix to get the base category.

    "GEMM-MoE" -> "GEMM", "Elementwise-GDN" -> "Elementwise", "SDPA" -> "SDPA".
    """
    return cat.split("-")[0] if "-" in cat else cat


def validate_type_category(kernel_labels, kernel_layers, cls_by_idx):
    """Validate that each kernel's classified type matches its assigned category.

    Compares the kernel's classified perf_category against the expected
    perf_category of its assigned semantic block label.  Compatibility uses
    base categories: "GEMM-MoE" is compatible with "GEMM".

    Mismatched body kernels are relabeled to "Others" with layer=None.
    Positional labels (Preamble, Epilogue, *_secondary) are skipped.

    Returns the number of kernels relabeled.
    """
    relabeled = 0
    for idx in list(kernel_labels.keys()):
        label = kernel_labels[idx]
        if (
            label in ("Preamble", "Epilogue")
            or label.endswith("_secondary")
            or label.endswith("_preamble")
            or label.endswith("_epilogue")
        ):
            continue

        c = cls_by_idx.get(idx, {})
        kernel_cat = c.get("perf_category", "Others")
        label_cat = _label_perf_category(label)

        if _base_category(kernel_cat) != _base_category(label_cat):
            kernel_labels[idx] = f"{kernel_cat}_uncovered"
            kernel_layers[idx] = None
            relabeled += 1
    return relabeled


# ---------------------------------------------------------------------------
# Model info inference
# ---------------------------------------------------------------------------


def infer_model_info(cls_by_idx, num_layers, config=None, is_graph_mode=False):
    """Build model_info dict from data, optionally enriched by HF config."""
    architecture = "unknown"
    ffn_type = "unknown"
    layer_types = None

    if config:
        tc = config.get("text_config", config)
        architecture = tc.get("model_type", config.get("model_type", "unknown"))
        num_experts = tc.get("num_experts", tc.get("num_local_experts", 0))
        ffn_type = "moe" if num_experts > 0 else "dense"
        layer_types = tc.get("layer_types")

    if architecture == "unknown":
        moe_indicators = {
            "MoE GEMM",
            "MoE Finalize",
            "MoE Routing",
            "MoE Quantize",
        }
        ktypes = {c.get("kernel_type", "") for c in cls_by_idx.values()}
        ffn_type = "moe" if ktypes & moe_indicators else "dense"

    info = {
        "architecture": architecture,
        "num_layers": num_layers,
        "ffn_type": ffn_type,
        "graph_mode": is_graph_mode,
    }
    if layer_types:
        info["layer_types"] = layer_types
    return info


# ---------------------------------------------------------------------------
# Output builder
# ---------------------------------------------------------------------------


def build_output(
    kernels,
    kernel_labels,
    kernel_layers,
    cls_by_idx,
    total_time,
    source_file,
    model_info,
):
    """Build semantic_labels.json output dict."""
    labeled = []
    for i, k in enumerate(kernels):
        c = cls_by_idx.get(i, {})
        block = kernel_labels.get(i, "Uncovered")
        labeled.append(
            {
                "index": i,
                "name": k["name"],
                "dur": k["dur"],
                "kernel_type": c.get("kernel_type", "Unknown"),
                "semantic_block": block,
                "perf_category": _label_perf_category(block),
                "layer": kernel_layers.get(i),
            }
        )

    return {
        "source_file": source_file,
        "total_kernel_time_us": total_time,
        "model_info": model_info,
        "labeled_kernels": labeled,
    }


# ---------------------------------------------------------------------------
# Debug dump
# ---------------------------------------------------------------------------


def _fmt_group(idx, group):
    """Format one RLE group for debug output."""
    cat, count, _indices, ktypes = group
    anchor = get_group_anchor(group) or "-"
    unique_kt = sorted(set(ktypes))
    kt_str = ", ".join(unique_kt)
    return f"[{idx:3d}] {cat:20s} ({count:3d} kernels)  anchor={anchor:20s}  types=[{kt_str}]"


def dump_alignment_debug(
    path,
    name_a,
    name_b,
    rle_a,
    rle_b,
    period_a,
    period_b,
    sc_a,
    sc_b,
    alignment,
    labels,
):
    """Write a human-readable debug file showing RLE, period, and alignment."""
    lines = []

    # -- Section 1: Full RLE for each trace --
    for name, rle, period in [(name_a, rle_a, period_a), (name_b, rle_b, period_b)]:
        lines.append(f"{'=' * 80}")
        lines.append(f"RLE groups for {name}  ({len(rle)} groups, period={period})")
        lines.append(f"{'=' * 80}")
        for i, g in enumerate(rle):
            marker = " <-- period boundary" if i > 0 and i % period == 0 else ""
            lines.append(_fmt_group(i, g) + marker)
        lines.append("")

    # -- Section 2: Super-cycles --
    for name, sc, period in [(name_a, sc_a, period_a), (name_b, sc_b, period_b)]:
        lines.append(f"{'=' * 80}")
        lines.append(f"Super-cycle for {name}  ({period} groups)")
        lines.append(f"{'=' * 80}")
        for i, g in enumerate(sc):
            lines.append(_fmt_group(i, g))
        lines.append("")

    # -- Section 3: Cross-trace alignment --
    lines.append(f"{'=' * 80}")
    lines.append(f"Alignment  ({name_a} vs {name_b})")
    lines.append(f"{'=' * 80}")
    header = (
        f"{'label':30s}  {'<' + name_a + '>':25s}  {'<' + name_b + '>':25s}  anchor"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for (ga, gb), label in zip(alignment, labels):
        a_str = f"{ga[0]} ({ga[1]})" if ga else "---"
        b_str = f"{gb[0]} ({gb[1]})" if gb else "---"
        anchor_a = get_group_anchor(ga) if ga else None
        anchor_b = get_group_anchor(gb) if gb else None
        anchor = anchor_a or anchor_b or "-"
        lines.append(f"{label:30s}  {a_str:25s}  {b_str:25s}  {anchor}")

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote alignment debug to {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_alignment(
    dir_a,
    dir_b,
    name_a="trace_a",
    name_b="trace_b",
    output_a=None,
    output_b=None,
    config_path=None,
):
    """Run the full cross-trace alignment pipeline.

    Returns (output_dict_a, output_dict_b).
    """
    ext_a, pat_a, _, cls_a = load_trace_dir(dir_a)
    ext_b, pat_b, _, cls_b = load_trace_dir(dir_b)

    kernels_a = ext_a["kernels"]
    kernels_b = ext_b["kernels"]
    n_a = len(kernels_a)
    n_b = len(kernels_b)

    config = None
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    # ---- partition kernel indices ----
    preamble_a = set(pat_a.get("preamble_indices", []))
    epilogue_a = set(pat_a.get("epilogue_indices", []))
    secondary_a = set(pat_a.get("secondary_stream_indices", []))
    body_a = [
        i
        for i in range(n_a)
        if i not in preamble_a and i not in epilogue_a and i not in secondary_a
    ]

    preamble_b = set(pat_b.get("preamble_indices", []))
    epilogue_b = set(pat_b.get("epilogue_indices", []))
    secondary_b = set(pat_b.get("secondary_stream_indices", []))
    body_b = [
        i
        for i in range(n_b)
        if i not in preamble_b and i not in epilogue_b and i not in secondary_b
    ]

    # ---- build body RLE ----
    rle_a = build_rle(body_a, cls_a)
    rle_b = build_rle(body_b, cls_b)

    # ---- detect super-cycle period ----
    period_a = detect_period(rle_a)
    period_b = detect_period(rle_b)
    print(
        f"{name_a}: {len(body_a)} body kernels, {len(rle_a)} RLE groups, "
        f"period {period_a}",
        file=sys.stderr,
    )
    print(
        f"{name_b}: {len(body_b)} body kernels, {len(rle_b)} RLE groups, "
        f"period {period_b}",
        file=sys.stderr,
    )

    # Use one super-cycle (first period) from each trace for alignment
    sc_a = rle_a[:period_a]
    sc_b = rle_b[:period_b]

    # ---- DP alignment of super-cycles ----
    alignment = needleman_wunsch(sc_a, sc_b)

    matched = sum(1 for ga, gb in alignment if ga is not None and gb is not None)
    a_only = sum(1 for ga, gb in alignment if ga is not None and gb is None)
    b_only = sum(1 for ga, gb in alignment if ga is None and gb is not None)
    print(
        f"Alignment: {matched} matched, {a_only} {name_a}-only, "
        f"{b_only} {name_b}-only",
        file=sys.stderr,
    )

    # ---- assign labels ----
    labels = assign_labels(alignment)

    # ---- debug dump ----
    debug_path = os.path.join(dir_a, "alignment_debug.txt")
    dump_alignment_debug(
        debug_path,
        name_a,
        name_b,
        rle_a,
        rle_b,
        period_a,
        period_b,
        sc_a,
        sc_b,
        alignment,
        labels,
    )

    ref_cats_a, ref_labels_a = build_reference_sequence(alignment, labels, "a")
    ref_cats_b, ref_labels_b = build_reference_sequence(alignment, labels, "b")

    # ---- expand to full traces ----
    kl_a, ky_a = expand_labels_to_trace(ref_cats_a, ref_labels_a, body_a, cls_a)
    kl_b, ky_b = expand_labels_to_trace(ref_cats_b, ref_labels_b, body_b, cls_b)

    # ---- preamble / epilogue ----
    pl_a, py_a = label_preamble_epilogue(
        pat_a.get("preamble_indices", []), pat_a.get("epilogue_indices", []), cls_a
    )
    pl_b, py_b = label_preamble_epilogue(
        pat_b.get("preamble_indices", []), pat_b.get("epilogue_indices", []), cls_b
    )
    kl_a.update(pl_a)
    ky_a.update(py_a)
    kl_b.update(pl_b)
    ky_b.update(py_b)

    # ---- secondary stream ----
    sl_a, sy_a = label_secondary_stream(
        pat_a.get("secondary_stream_indices", []), cls_a
    )
    sl_b, sy_b = label_secondary_stream(
        pat_b.get("secondary_stream_indices", []), cls_b
    )
    kl_a.update(sl_a)
    ky_a.update(sy_a)
    kl_b.update(sl_b)
    ky_b.update(sy_b)

    # ---- type-category validation ----
    reval_a = validate_type_category(kl_a, ky_a, cls_a)
    reval_b = validate_type_category(kl_b, ky_b, cls_b)
    if reval_a or reval_b:
        print(
            f"Type validation: relabeled {reval_a} {name_a} + "
            f"{reval_b} {name_b} kernels to classified category",
            file=sys.stderr,
        )

    # ---- coverage stats ----
    labeled_a = sum(1 for i in range(n_a) if i in kl_a)
    labeled_b = sum(1 for i in range(n_b) if i in kl_b)
    print(
        f"{name_a}: {labeled_a}/{n_a} kernels labeled "
        f"({100 * labeled_a / n_a:.1f}%)",
        file=sys.stderr,
    )
    print(
        f"{name_b}: {labeled_b}/{n_b} kernels labeled "
        f"({100 * labeled_b / n_b:.1f}%)",
        file=sys.stderr,
    )

    # ---- build outputs ----
    total_a = ext_a["metadata"]["total_kernel_time_us"]
    total_b = ext_b["metadata"]["total_kernel_time_us"]
    src_a = ext_a.get("source_file", "")
    src_b = ext_b.get("source_file", "")
    graph_a = ext_a["metadata"].get("is_graph_mode", False)
    graph_b = ext_b["metadata"].get("is_graph_mode", False)

    layer_vals_a = [v for v in ky_a.values() if v is not None]
    layer_vals_b = [v for v in ky_b.values() if v is not None]
    num_layers_a = (max(layer_vals_a) + 1) if layer_vals_a else 0
    num_layers_b = (max(layer_vals_b) + 1) if layer_vals_b else 0

    mi_a = infer_model_info(cls_a, num_layers_a, config, graph_a)
    mi_b = infer_model_info(cls_b, num_layers_b, config, graph_b)

    out_a = build_output(kernels_a, kl_a, ky_a, cls_a, total_a, src_a, mi_a)
    out_b = build_output(kernels_b, kl_b, ky_b, cls_b, total_b, src_b, mi_b)

    # ---- write ----
    path_a = output_a or os.path.join(dir_a, "semantic_labels.json")
    path_b = output_b or os.path.join(dir_b, "semantic_labels.json")

    with open(path_a, "w") as f:
        json.dump(out_a, f, indent=2)
    print(f"Wrote {path_a}", file=sys.stderr)

    with open(path_b, "w") as f:
        json.dump(out_b, f, indent=2)
    print(f"Wrote {path_b}", file=sys.stderr)

    # ---- label summary ----
    blocks_a = Counter(kl_a.values())
    blocks_b = Counter(kl_b.values())
    all_labels = sorted(set(list(blocks_a.keys()) + list(blocks_b.keys())))

    print(f"\nLabel summary ({len(all_labels)} unique labels):", file=sys.stderr)
    for label in all_labels:
        ca = blocks_a.get(label, 0)
        cb = blocks_b.get(label, 0)
        print(f"  {label:30s}  {name_a}={ca:4d}  {name_b}={cb:4d}", file=sys.stderr)

    return out_a, out_b


def main():
    parser = argparse.ArgumentParser(
        description="Cross-trace alignment and labeling",
    )
    parser.add_argument(
        "--dir-a",
        required=True,
        help="Directory with trace A's extracted/pattern/classified JSON",
    )
    parser.add_argument(
        "--dir-b",
        required=True,
        help="Directory with trace B's extracted/pattern/classified JSON",
    )
    parser.add_argument(
        "--output-a", help="Output path for trace A semantic_labels.json"
    )
    parser.add_argument(
        "--output-b", help="Output path for trace B semantic_labels.json"
    )
    parser.add_argument("--name-a", default="trace_a", help="Display name for trace A")
    parser.add_argument("--name-b", default="trace_b", help="Display name for trace B")
    parser.add_argument("--config", help="Path to HuggingFace config.json (optional)")
    args = parser.parse_args()

    run_alignment(
        args.dir_a,
        args.dir_b,
        name_a=args.name_a,
        name_b=args.name_b,
        output_a=args.output_a,
        output_b=args.output_b,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
