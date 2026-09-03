#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Name-first cross-trace kernel-name unification (LLM-assisted).

Graph-mode traces collapse the CPU->GPU call stack under ``hip/cudaGraphLaunch``,
so ``nn_module`` / ``cpu_op`` context is unavailable and the block-alignment
harmonization cannot work.  The only reliable cross-trace signal is the raw GPU
**kernel name**.  This module unifies kernel names across two traces so that the
downstream comparison (which matches on the per-kernel ``semantic_block`` field)
can pair equivalent kernels.

The approach is *name-first*: an LLM inspects the unique kernel names from both
traces and writes a conservative map of names it is **certain** are equivalent
(e.g. ``moe_attn_vllm`` and ``sglang_moe_attention`` -> ``moe_attn``).  Names
that are already identical need no entry -- they unify by default.  The goal is
to establish clear anchors, not to resolve every ambiguity.

Three subcommands:

  prepare-context    Aggregate the unique kernel names from both traces (with
                     per-name stats) into a compact LLM packet.  When the
                     combined unique-name count exceeds ``--threshold`` it emits
                     a representative *sample* plus ``needs_stem_preprocessing:
                     true`` instead of the full lists (see apply-stem-rules).

  apply-stem-rules   Apply an LLM-authored ``stem_rules.json`` (custom regexes +
                     collapse/preserve/drop actions) to the full unique-name set,
                     emitting a ``raw_to_stem`` map and a reduced,
                     stem-level context.  Prints resulting cardinality so the
                     model can iterate.

  apply-map          Apply the LLM's ``kernel_unification_map.json`` back onto
                     both ``semantic_labels.json`` files, writing the unified
                     name into ``semantic_block`` (default = raw name / stem when
                     the map has no entry).

Usage:
    python kernel_unification.py prepare-context \
        --labels-a <dir_a>/semantic_labels.json \
        --labels-b <dir_b>/semantic_labels.json \
        --name-a MI300 --name-b B300 \
        -o kernel_unification_context.json

    python kernel_unification.py apply-stem-rules \
        --labels-a <dir_a>/semantic_labels.json \
        --labels-b <dir_b>/semantic_labels.json \
        --name-a MI300 --name-b B300 \
        --rules stem_rules.json \
        --raw-to-stem raw_to_stem.json \
        -o kernel_unification_context.json

    python kernel_unification.py apply-map \
        --labels-a <dir_a>/semantic_labels.json \
        --labels-b <dir_b>/semantic_labels.json \
        --name-a MI300 --name-b B300 \
        --map kernel_unification_map.json \
        [--raw-to-stem raw_to_stem.json]
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import load_json


DEFAULT_THRESHOLD = 5000
DEFAULT_SAMPLE_SIZE = 300


# ===========================================================================
# shared helpers
# ===========================================================================


def _load(path):
    return load_json(path)


def _dims_repr(dims, limit=160):
    """Compact, length-capped string form of an input_dims value."""
    if not dims:
        return ""
    s = json.dumps(dims, separators=(",", ":"))
    if len(s) > limit:
        s = s[:limit] + "..."
    return s


def aggregate_names(labels, key_fn=None):
    """Aggregate a labels file's kernels by name (or by ``key_fn(name)``).

    Returns an OrderedDict ``{key: entry}`` where entry carries count,
    total duration, the set of perf_categories seen, and one sample
    input_dims.  Order is by descending total duration then key, so the
    most impactful kernels come first for the LLM.
    """
    acc = {}
    for k in labels.get("labeled_kernels", []):
        name = k.get("name", "")
        key = key_fn(name) if key_fn else name
        if key is None:
            continue  # dropped by stem rules -> excluded from the unification set
        entry = acc.get(key)
        if entry is None:
            entry = acc[key] = {
                "name": key,
                "kernel_count": 0,
                "total_dur_us": 0.0,
                "perf_categories": set(),
                "sample_input_dims": "",
                "sample_raw_names": set(),
            }
        entry["kernel_count"] += 1
        entry["total_dur_us"] += k.get("dur", 0.0) or 0.0
        pc = k.get("perf_category")
        if pc:
            entry["perf_categories"].add(pc)
        if not entry["sample_input_dims"]:
            entry["sample_input_dims"] = _dims_repr(k.get("input_dims"))
        if key_fn and len(entry["sample_raw_names"]) < 5 and name != key:
            entry["sample_raw_names"].add(name)

    ordered = sorted(acc.values(), key=lambda e: (-e["total_dur_us"], e["name"]))
    out = OrderedDict()
    for e in ordered:
        e["total_dur_us"] = round(e["total_dur_us"], 3)
        e["perf_categories"] = sorted(e["perf_categories"])
        e["sample_raw_names"] = sorted(e["sample_raw_names"])
        if not e["sample_raw_names"]:
            del e["sample_raw_names"]
        out[e["name"]] = e
    return out


def _entry_list(agg, keys):
    """Materialize entries for *keys* preserving *agg* ordering."""
    keyset = set(keys)
    return [e for name, e in agg.items() if name in keyset]


def _build_context(agg_a, agg_b, name_a, name_b, level, extra=None):
    """Assemble the unification-context dict from two aggregations."""
    set_a = set(agg_a)
    set_b = set(agg_b)
    only_a = [n for n in agg_a if n not in set_b]
    only_b = [n for n in agg_b if n not in set_a]
    in_both = sorted(set_a & set_b)

    ctx = OrderedDict()
    ctx["name_a"] = name_a
    ctx["name_b"] = name_b
    ctx["key_level"] = level  # "raw_name" or "stem"
    ctx["summary"] = {
        f"unique_{name_a}": len(set_a),
        f"unique_{name_b}": len(set_b),
        "combined_unique": len(set_a | set_b),
        "in_both": len(in_both),
        f"only_in_{name_a}": len(only_a),
        f"only_in_{name_b}": len(only_b),
    }
    ctx[f"only_in_{name_a}"] = _entry_list(agg_a, only_a)
    ctx[f"only_in_{name_b}"] = _entry_list(agg_b, only_b)
    ctx["in_both"] = in_both
    if extra:
        ctx.update(extra)
    return ctx


# ===========================================================================
# prepare-context
# ===========================================================================


def _sample_names(agg_a, agg_b, name_a, name_b, sample_size):
    """Deterministic representative sample across both traces.

    Interleaves the two per-trace aggregations (already ordered by impact)
    and takes evenly-spaced picks so the sample spans high- and low-impact
    kernels rather than only the top-N.
    """

    def _spaced(agg, budget):
        items = list(agg.values())
        if len(items) <= budget:
            return items
        step = len(items) / float(budget)
        return [items[int(i * step)] for i in range(budget)]

    half = max(1, sample_size // 2)
    sample = []
    for e in _spaced(agg_a, half):
        row = dict(e)
        row["trace"] = name_a
        sample.append(row)
    for e in _spaced(agg_b, sample_size - half):
        row = dict(e)
        row["trace"] = name_b
        sample.append(row)
    return sample


def cmd_prepare_context(args):  # pragma: no cover
    labels_a = _load(args.labels_a)
    labels_b = _load(args.labels_b)
    agg_a = aggregate_names(labels_a)
    agg_b = aggregate_names(labels_b)

    combined = len(set(agg_a) | set(agg_b))
    needs_stem = combined > args.threshold

    if needs_stem:
        ctx = OrderedDict()
        ctx["name_a"] = args.name_a
        ctx["name_b"] = args.name_b
        ctx["key_level"] = "raw_name"
        ctx["summary"] = {
            f"unique_{args.name_a}": len(agg_a),
            f"unique_{args.name_b}": len(agg_b),
            "combined_unique": combined,
        }
        ctx["needs_stem_preprocessing"] = True
        ctx["threshold"] = args.threshold
        ctx["instructions"] = (
            "Combined unique kernel-name count exceeds the threshold and will "
            "likely overwhelm the context. Inspect the sample below, author a "
            "stem_rules.json (see the kernel-stem-preprocessing agent), and run "
            "'kernel_unification.py apply-stem-rules' to reduce the name set "
            "before unification."
        )
        ctx["sample"] = _sample_names(
            agg_a, agg_b, args.name_a, args.name_b, args.sample_size
        )
    else:
        ctx = _build_context(
            agg_a,
            agg_b,
            args.name_a,
            args.name_b,
            "raw_name",
            extra={"needs_stem_preprocessing": False},
        )

    with open(args.output, "w") as f:
        json.dump(ctx, f, indent=2)

    if needs_stem:
        print(
            f"Wrote {args.output}: {combined} combined unique names > "
            f"threshold {args.threshold} -> STEM PREPROCESSING NEEDED "
            f"({len(ctx['sample'])} sampled names emitted).",
            file=sys.stderr,
        )
    else:
        print(
            f"Wrote {args.output}: {combined} combined unique names "
            f"({ctx['summary']['in_both']} shared, "
            f"{ctx['summary'][f'only_in_{args.name_a}']} {args.name_a}-only, "
            f"{ctx['summary'][f'only_in_{args.name_b}']} {args.name_b}-only).",
            file=sys.stderr,
        )


# ===========================================================================
# apply-stem-rules
# ===========================================================================


def _compile_rules(rules):
    """Compile stem_rules entries; validate shape and actions."""
    compiled = []
    for i, r in enumerate(rules):
        action = r.get("action", "collapse")
        if action not in ("collapse", "preserve", "drop"):
            raise SystemExit(
                f"rule {i}: invalid action {action!r} "
                f"(expected collapse/preserve/drop)"
            )
        pattern = r.get("pattern", "")
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise SystemExit(f"rule {i}: bad regex {pattern!r}: {e}")
        compiled.append(
            {
                "regex": rx,
                "action": action,
                "replacement": r.get("replacement", ""),
                "note": r.get("note", ""),
            }
        )
    return compiled


def stem_for(name, compiled):
    """Return (stem, action) for *name* using the first matching rule.

    Unmatched names default to (name, "preserve").
    """
    for r in compiled:
        if r["regex"].search(name):
            if r["action"] == "collapse":
                try:
                    return r["regex"].sub(r["replacement"], name), "collapse"
                except re.error as e:
                    # A malformed replacement template (e.g. a backreference with
                    # no matching capture group) is only detected by re at
                    # substitution time. Don't let one bad rule crash the whole
                    # run: warn once and fall back to preserving the name.
                    if not r.get("_warned"):
                        print(
                            f"[kernel_unification] ignoring stem rule with bad "
                            f"replacement {r['replacement']!r} for pattern "
                            f"{r['regex'].pattern!r}: {e}",
                            file=sys.stderr,
                        )
                        r["_warned"] = True
                    return name, "preserve"
            if r["action"] == "preserve":
                return name, "preserve"
            return name, "drop"
    return name, "preserve"


def cmd_apply_stem_rules(args):  # pragma: no cover
    labels_a = _load(args.labels_a)
    labels_b = _load(args.labels_b)
    rules_doc = _load(args.rules)
    rules = rules_doc["rules"] if isinstance(rules_doc, dict) else rules_doc
    compiled = _compile_rules(rules)

    agg_a_raw = aggregate_names(labels_a)
    agg_b_raw = aggregate_names(labels_b)
    all_names = set(agg_a_raw) | set(agg_b_raw)

    raw_to_stem = {}
    dropped = set()
    action_counts = {"collapse": 0, "preserve": 0, "drop": 0}
    for name in all_names:
        stem, action = stem_for(name, compiled)
        action_counts[action] += 1
        if action == "drop":
            dropped.add(name)
            raw_to_stem[name] = name  # falls back to raw; excluded from context
        else:
            raw_to_stem[name] = stem

    if args.raw_to_stem:
        with open(args.raw_to_stem, "w") as f:
            json.dump(raw_to_stem, f, indent=2)

    def _stem_key(name):
        if name in dropped:
            return None  # exclude dropped kernels from the stem aggregation
        return raw_to_stem.get(name, name)

    agg_a = aggregate_names(labels_a, key_fn=lambda n: _stem_key(n))
    agg_b = aggregate_names(labels_b, key_fn=lambda n: _stem_key(n))
    agg_a.pop(None, None)
    agg_b.pop(None, None)

    combined_stems = len(set(agg_a) | set(agg_b))
    ctx = _build_context(
        agg_a,
        agg_b,
        args.name_a,
        args.name_b,
        "stem",
        extra={
            "needs_stem_preprocessing": False,
            "stem_preprocessing_applied": True,
            "raw_unique_before": len(all_names),
            "stem_unique_after": combined_stems,
            "action_counts": action_counts,
            "dropped_count": len(dropped),
        },
    )

    with open(args.output, "w") as f:
        json.dump(ctx, f, indent=2)

    status = "OK" if combined_stems <= args.threshold else "STILL ABOVE THRESHOLD"
    print(
        f"Stem rules: {len(all_names)} raw -> {combined_stems} stems "
        f"(collapse {action_counts['collapse']}, preserve "
        f"{action_counts['preserve']}, drop {action_counts['drop']}). "
        f"threshold {args.threshold}: {status}.",
        file=sys.stderr,
    )
    if combined_stems > args.threshold:
        print(
            "  Reduce further: broaden collapse rules or drop more low-value "
            "families, then re-run apply-stem-rules.",
            file=sys.stderr,
        )


# ===========================================================================
# apply-map
# ===========================================================================


def _load_map_side(map_doc, side, name):
    """Extract one trace's {key: unified} map from the LLM map document.

    Accepts either ``map_<side>`` / ``map_<name>`` keys or a nested
    ``{trace_a: {map: {...}}}`` shape.
    """
    for cand in (f"map_{side}", f"map_{name}", side, name):
        if cand in map_doc:
            val = map_doc[cand]
            if isinstance(val, dict) and "map" in val:
                return val["map"]
            if isinstance(val, dict):
                return val
    return {}


def _apply_side(labels, unified_map, raw_to_stem):
    """Write semantic_block = unified name for every kernel; return stats."""
    n_mapped = 0
    n_stemmed = 0
    for k in labels.get("labeled_kernels", []):
        raw = k.get("name", "")
        base = raw
        if raw_to_stem is not None:
            base = raw_to_stem.get(raw, raw)
            if base != raw:
                n_stemmed += 1
        unified = unified_map.get(base, base)
        if base in unified_map:
            n_mapped += 1
        k["semantic_block"] = unified
    return {
        "kernels": len(labels.get("labeled_kernels", [])),
        "mapped": n_mapped,
        "stemmed": n_stemmed,
    }


def cmd_apply_map(args):  # pragma: no cover
    labels_a = _load(args.labels_a)
    labels_b = _load(args.labels_b)
    map_doc = _load(args.map)
    raw_to_stem = _load(args.raw_to_stem) if args.raw_to_stem else None

    map_a = _load_map_side(map_doc, "a", args.name_a)
    map_b = _load_map_side(map_doc, "b", args.name_b)

    stats_a = _apply_side(labels_a, map_a, raw_to_stem)
    stats_b = _apply_side(labels_b, map_b, raw_to_stem)

    with open(args.labels_a, "w") as f:
        json.dump(labels_a, f, indent=2)
    with open(args.labels_b, "w") as f:
        json.dump(labels_b, f, indent=2)

    blocks_a = set(k["semantic_block"] for k in labels_a["labeled_kernels"])
    blocks_b = set(k["semantic_block"] for k in labels_b["labeled_kernels"])
    shared = blocks_a & blocks_b

    print(f"Applied to {args.name_a}: {stats_a}", file=sys.stderr)
    print(f"Applied to {args.name_b}: {stats_b}", file=sys.stderr)
    print(
        f"Unified vocabulary: {len(shared)} shared blocks, "
        f"{len(blocks_a - shared)} {args.name_a}-only, "
        f"{len(blocks_b - shared)} {args.name_b}-only.",
        file=sys.stderr,
    )


# ===========================================================================
# CLI
# ===========================================================================


def _add_common(p):  # pragma: no cover
    p.add_argument("--labels-a", required=True, help="Trace A semantic_labels.json")
    p.add_argument("--labels-b", required=True, help="Trace B semantic_labels.json")
    p.add_argument("--name-a", default="trace_a", help="Short name for trace A")
    p.add_argument("--name-b", default="trace_b", help="Short name for trace B")


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Name-first cross-trace kernel-name unification "
        "(prepare-context / apply-stem-rules / apply-map)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare-context", help="Build the LLM unification context")
    _add_common(p_prep)
    p_prep.add_argument("-o", "--output", required=True, help="Output context JSON")
    p_prep.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"Combined-unique count above which stem preprocessing is flagged "
        f"(default {DEFAULT_THRESHOLD})",
    )
    p_prep.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Names to sample when over threshold (default {DEFAULT_SAMPLE_SIZE})",
    )
    p_prep.set_defaults(func=cmd_prepare_context)

    p_stem = sub.add_parser(
        "apply-stem-rules",
        help="Apply LLM-authored stem_rules.json and emit a reduced context",
    )
    _add_common(p_stem)
    p_stem.add_argument("--rules", required=True, help="stem_rules.json from the LLM")
    p_stem.add_argument(
        "--raw-to-stem", help="Output path for the raw-name -> stem map"
    )
    p_stem.add_argument("-o", "--output", required=True, help="Reduced context JSON")
    p_stem.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"Target combined-stem count (default {DEFAULT_THRESHOLD})",
    )
    p_stem.set_defaults(func=cmd_apply_stem_rules)

    p_apply = sub.add_parser(
        "apply-map",
        help="Apply kernel_unification_map.json onto both label files",
    )
    _add_common(p_apply)
    p_apply.add_argument("--map", required=True, help="kernel_unification_map.json")
    p_apply.add_argument(
        "--raw-to-stem", help="raw-name -> stem map (if stem preprocessing was used)"
    )
    p_apply.set_defaults(func=cmd_apply_map)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
