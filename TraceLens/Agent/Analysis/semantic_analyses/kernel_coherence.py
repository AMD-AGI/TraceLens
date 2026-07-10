#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Second-pass kernel-name coherence (LLM-assisted).

After the first-pass name-first unification (``kernel_unification.py``), the
comparison still has **one-sided** buckets: kernels whose unified name appears
in only one trace (e.g. vendor GEMM families -- MI300 ``Cijk_*`` vs B300
``nvjet_*`` -- that could not be paired by name alone).

This pass uses the first-pass **shared** buckets as cross-trace-stable
positional anchors. For each one-sided bucket it derives the *neighbor context*
(nearest shared symbols to its left and right in the run-length-collapsed kernel
sequence) and lets an LLM:

  * pair a one-sided bucket in trace A with a one-sided bucket in trace B that
    occupies the **same** neighbor context (e.g. the GEMM between ``add_rmsnorm``
    and ``rotary_embedding`` is the QKV projection on both traces), assigning
    both a new shared name; and
  * split a single name that occurs in **different** contexts into different
    buckets (context-dependent granularity).

Two subcommands:

  prepare-context  Build the LLM packet: condensed sequences, shared vs
                   one-sided symbol sets, and per one-sided symbol the unique
                   neighbor contexts with evidence (perf_category, kernels in
                   the run, duration, sample input_dims, raw name). Emits a flat
                   ``context_catalog`` with stable ids.

  apply            Apply the LLM decisions (``context_renames`` +
                   ``fallback_remap_a`` / ``fallback_remap_b``) to recompute each
                   kernel's ``semantic_block`` in place, emit an audit CSV, and
                   warn about any residual one-sided condensed symbols.

Usage:
    python kernel_coherence.py prepare-context \
        --labels-a <dir_a>/semantic_labels.json \
        --labels-b <dir_b>/semantic_labels.json \
        --name-a MI300 --name-b B300 \
        --neighbor-radius 1 \
        -o kernel_coherence_context.json

    python kernel_coherence.py apply \
        --context kernel_coherence_context.json \
        --decisions kernel_coherence_decisions.json \
        [--audit-csv-a per_kernel_final_MI300.csv] \
        [--audit-csv-b per_kernel_final_B300.csv]
"""

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_runlength import (
    collapse_consecutive,
    load_sequence,
    run_index_per_kernel,
    shared_neighbor_windows_skip_non_shared,
)


DEFAULT_RADIUS = 1
DEFAULT_TOP_KERNELS = 5


def _load(path):
    with open(path) as f:
        return json.load(f)


def _dims_repr(dims, limit=160):
    if not dims:
        return ""
    s = json.dumps(dims, separators=(",", ":"))
    return s[:limit] + "..." if len(s) > limit else s


# ===========================================================================
# prepare-context
# ===========================================================================


def _run_to_kernel_indices(run_per_kernel):
    """Map run index -> list of kernel positions in that run."""
    run_to_k = defaultdict(list)
    for ki, rj in enumerate(run_per_kernel):
        run_to_k[rj].append(ki)
    return run_to_k


def _symbol_evidence(kernels, indices, top_kernels):
    """Aggregate evidence for the kernels of one run (or symbol)."""
    by_name_dur = defaultdict(float)
    by_name_cnt = defaultdict(int)
    cats = set()
    sample_dims = ""
    for ki in indices:
        k = kernels[ki]
        nm = k.get("name", "")
        by_name_dur[nm] += k.get("dur", 0.0) or 0.0
        by_name_cnt[nm] += 1
        pc = k.get("perf_category")
        if pc:
            cats.add(pc)
        if not sample_dims:
            sample_dims = _dims_repr(k.get("input_dims"))
    ranked = sorted(by_name_dur.items(), key=lambda x: -x[1])[:top_kernels]
    top = [
        {"kernel_name": n, "total_us": round(d, 3), "kernel_count": by_name_cnt[n]}
        for n, d in ranked
    ]
    return sorted(cats), sample_dims, top


def _collect_contexts(name, kernels, condensed, run_per_kernel, shared, problematic,
                      radius, top_kernels):
    """Per one-sided symbol, the unique (left,right) shared-neighbor contexts."""
    run_to_k = _run_to_kernel_indices(run_per_kernel)
    out = OrderedDict()
    ctr = 0
    for sym in sorted(problematic):
        seen = set()
        contexts = []
        for j, c in enumerate(condensed):
            if c != sym:
                continue
            left, right = shared_neighbor_windows_skip_non_shared(
                condensed, j, shared, radius
            )
            key = (tuple(left), tuple(right))
            if key in seen:
                continue
            seen.add(key)
            indices = run_to_k.get(j, [])
            cats, dims, top = _symbol_evidence(kernels, indices, top_kernels)
            contexts.append(
                OrderedDict(
                    id=f"{name}:{ctr}",
                    first_pass_block=sym,
                    left_window=list(left),
                    right_window=list(right),
                    kernels_in_run=len(indices),
                    perf_categories=cats,
                    sample_input_dims=dims,
                    top_kernel_names_by_dur=top,
                )
            )
            ctr += 1
        out[sym] = {"contexts": contexts, "context_count": len(contexts)}
    return out


def cmd_prepare_context(args):
    labels_a = _load(args.labels_a)
    labels_b = _load(args.labels_b)
    kernels_a = labels_a["labeled_kernels"]
    kernels_b = labels_b["labeled_kernels"]

    seq_a = [k.get("semantic_block", "") for k in kernels_a]
    seq_b = [k.get("semantic_block", "") for k in kernels_b]
    cond_a = collapse_consecutive(seq_a)
    cond_b = collapse_consecutive(seq_b)
    set_a, set_b = set(cond_a), set(cond_b)
    shared = set_a & set_b
    prob_a = set_a - set_b
    prob_b = set_b - set_a

    run_a = run_index_per_kernel(seq_a)
    run_b = run_index_per_kernel(seq_b)

    detail_a = _collect_contexts(
        args.name_a, kernels_a, cond_a, run_a, shared, prob_a,
        args.neighbor_radius, args.top_kernels,
    )
    detail_b = _collect_contexts(
        args.name_b, kernels_b, cond_b, run_b, shared, prob_b,
        args.neighbor_radius, args.top_kernels,
    )

    catalog = []
    for detail, wl in ((detail_a, args.name_a), (detail_b, args.name_b)):
        for sym, pack in detail.items():
            for c in pack["contexts"]:
                catalog.append(
                    OrderedDict(
                        id=c["id"],
                        workload=wl,
                        first_pass_block=sym,
                        left_window=c["left_window"],
                        right_window=c["right_window"],
                    )
                )

    out = OrderedDict()
    out["name_a"] = args.name_a
    out["name_b"] = args.name_b
    out["definitions"] = {
        "condensed_sequence": "Kernel-order semantic_block values with consecutive "
        "duplicates removed (one symbol per contiguous run).",
        "shared_block": "semantic_block present in the condensed sequence of both traces.",
        "one_sided_block": "semantic_block present in only one trace's condensed sequence.",
        "neighbor_context": "left_window/right_window: nearest `neighbor_radius` shared "
        "symbols on each side of the run, skipping non-shared symbols.",
    }
    out["hyperparameters"] = {
        "neighbor_radius": args.neighbor_radius,
        "top_kernels": args.top_kernels,
    }
    out["inputs"] = {"labels_a": args.labels_a, "labels_b": args.labels_b}
    out["condensed_sequence_a"] = cond_a
    out["condensed_sequence_b"] = cond_b
    out["shared_blocks"] = sorted(shared)
    out["one_sided_in_a"] = sorted(prob_a)
    out["one_sided_in_b"] = sorted(prob_b)
    out["one_sided_details_a"] = detail_a
    out["one_sided_details_b"] = detail_b
    out["llm_task"] = (
        "Re-label one-sided blocks so the comparison has no one-sided condensed "
        "symbols. Pair a one-sided block in A with a one-sided block in B that has "
        "the SAME (left_window,right_window) and perf_category by giving both the "
        "same new shared name. Split a block that appears in different contexts via "
        "distinct context ids. See the kernel-coherence agent for the output schema."
    )
    out["llm_output_schema"] = {
        "context_renames": "{context_id -> final_block}",
        "fallback_remap_a": "{first_pass_block -> final_block} for trace A",
        "fallback_remap_b": "{first_pass_block -> final_block} for trace B",
        "notes": "optional string",
    }
    out["context_catalog"] = catalog

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    nca = sum(v["context_count"] for v in detail_a.values())
    ncb = sum(v["context_count"] for v in detail_b.values())
    print(
        f"Wrote {args.output}: shared={len(shared)} "
        f"one_sided_a={len(prob_a)} one_sided_b={len(prob_b)} "
        f"contexts_a={nca} contexts_b={ncb}",
        file=sys.stderr,
    )


# ===========================================================================
# apply
# ===========================================================================


def _context_lookup(catalog):
    """(workload, symbol, left, right) -> context_id."""
    out = {}
    for row in catalog:
        out[
            (
                row["workload"],
                row["first_pass_block"],
                tuple(row.get("left_window") or []),
                tuple(row.get("right_window") or []),
            )
        ] = row["id"]
    return out


def _final_blocks(workload, kernels, shared, problematic, radius,
                  lookup, context_renames, fallback):
    """Compute the final semantic_block for each kernel; return (finals, audit)."""
    seq = [k.get("semantic_block", "") for k in kernels]
    cond = collapse_consecutive(seq)
    run_per_kernel = run_index_per_kernel(seq)

    finals = []
    audit = []
    for i, k in enumerate(kernels):
        fp = seq[i]
        j = run_per_kernel[i]
        sym = cond[j]
        cid = ""
        if sym not in problematic:
            final = fp
        else:
            left, right = shared_neighbor_windows_skip_non_shared(
                cond, j, shared, radius
            )
            cid = lookup.get((workload, sym, tuple(left), tuple(right)), "")
            if cid and cid in context_renames:
                final = context_renames[cid]
            elif sym in fallback:
                final = fallback[sym]
            else:
                final = fp
        finals.append(final)
        audit.append(
            {
                "kernel_index": k.get("index", i),
                "name": k.get("name", ""),
                "first_pass_block": fp,
                "final_block": final,
                "context_id": cid,
            }
        )
    return finals, audit


def _write_audit(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["kernel_index", "name", "first_pass_block",
                        "final_block", "context_id"],
        )
        w.writeheader()
        w.writerows(rows)


def _residual_one_sided(labels_a, labels_b):
    """Return one-sided condensed symbols remaining after apply."""
    ca = set(collapse_consecutive([k.get("semantic_block", "") for k in labels_a["labeled_kernels"]]))
    cb = set(collapse_consecutive([k.get("semantic_block", "") for k in labels_b["labeled_kernels"]]))
    return sorted(ca - cb), sorted(cb - ca)


def cmd_apply(args):
    ctx = _load(args.context)
    dec = _load(args.decisions)
    name_a = ctx["name_a"]
    name_b = ctx["name_b"]
    radius = int(ctx.get("hyperparameters", {}).get("neighbor_radius", DEFAULT_RADIUS))
    labels_a_path = ctx["inputs"]["labels_a"]
    labels_b_path = ctx["inputs"]["labels_b"]

    context_renames = {str(k): str(v) for k, v in (dec.get("context_renames") or {}).items()}
    fb_a = {str(k): str(v) for k, v in (dec.get("fallback_remap_a") or {}).items()}
    fb_b = {str(k): str(v) for k, v in (dec.get("fallback_remap_b") or {}).items()}

    labels_a = _load(labels_a_path)
    labels_b = _load(labels_b_path)
    ka = labels_a["labeled_kernels"]
    kb = labels_b["labeled_kernels"]

    cond_a = collapse_consecutive([k.get("semantic_block", "") for k in ka])
    cond_b = collapse_consecutive([k.get("semantic_block", "") for k in kb])
    shared = set(cond_a) & set(cond_b)
    prob_a = set(cond_a) - set(cond_b)
    prob_b = set(cond_b) - set(cond_a)
    lookup = _context_lookup(ctx.get("context_catalog", []))

    finals_a, audit_a = _final_blocks(
        name_a, ka, shared, prob_a, radius, lookup, context_renames, fb_a
    )
    finals_b, audit_b = _final_blocks(
        name_b, kb, shared, prob_b, radius, lookup, context_renames, fb_b
    )

    changed_a = 0
    for k, fin in zip(ka, finals_a):
        if k.get("semantic_block") != fin:
            changed_a += 1
        k["semantic_block"] = fin
    changed_b = 0
    for k, fin in zip(kb, finals_b):
        if k.get("semantic_block") != fin:
            changed_b += 1
        k["semantic_block"] = fin

    with open(labels_a_path, "w") as f:
        json.dump(labels_a, f, indent=2)
    with open(labels_b_path, "w") as f:
        json.dump(labels_b, f, indent=2)

    if args.audit_csv_a:
        _write_audit(args.audit_csv_a, audit_a)
    if args.audit_csv_b:
        _write_audit(args.audit_csv_b, audit_b)

    blocks_a = set(k["semantic_block"] for k in ka)
    blocks_b = set(k["semantic_block"] for k in kb)
    res_a, res_b = _residual_one_sided(labels_a, labels_b)

    print(
        f"Applied: {name_a} {changed_a} kernels relabeled, "
        f"{name_b} {changed_b} kernels relabeled.",
        file=sys.stderr,
    )
    print(
        f"Shared blocks now: {len(blocks_a & blocks_b)} "
        f"({name_a}-only {len(blocks_a - blocks_b)}, "
        f"{name_b}-only {len(blocks_b - blocks_a)}).",
        file=sys.stderr,
    )
    if res_a or res_b:
        print(
            f"WARNING: condensed one-sided symbols remain -- "
            f"{name_a}: {res_a}  {name_b}: {res_b}",
            file=sys.stderr,
        )
    else:
        print("No one-sided condensed symbols remain.", file=sys.stderr)


# ===========================================================================
# CLI
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Second-pass kernel-name coherence (prepare-context / apply)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare-context", help="Build the coherence LLM context")
    p_prep.add_argument("--labels-a", required=True)
    p_prep.add_argument("--labels-b", required=True)
    p_prep.add_argument("--name-a", default="trace_a")
    p_prep.add_argument("--name-b", default="trace_b")
    p_prep.add_argument(
        "--neighbor-radius", type=int, default=DEFAULT_RADIUS,
        help=f"Shared symbols to collect on each side (default {DEFAULT_RADIUS})",
    )
    p_prep.add_argument(
        "--top-kernels", type=int, default=DEFAULT_TOP_KERNELS,
        help=f"Top kernel names by duration per context (default {DEFAULT_TOP_KERNELS})",
    )
    p_prep.add_argument("-o", "--output", required=True)
    p_prep.set_defaults(func=cmd_prepare_context)

    p_apply = sub.add_parser("apply", help="Apply coherence decisions to labels in place")
    p_apply.add_argument("--context", required=True, help="kernel_coherence_context.json")
    p_apply.add_argument("--decisions", required=True, help="LLM decisions JSON")
    p_apply.add_argument("--audit-csv-a", help="Per-kernel audit CSV for trace A")
    p_apply.add_argument("--audit-csv-b", help="Per-kernel audit CSV for trace B")
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
