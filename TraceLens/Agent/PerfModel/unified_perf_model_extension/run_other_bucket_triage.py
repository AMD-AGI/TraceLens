#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Triage for unified-perf-report-postprocess: rank ops in a
unified_perf_summary.csv that are worth a perf model.

Modes
-----
other-bucket (default):
  Rows whose op category is "other", ranked by runtime, up to the cumulative
  --fraction of "other" time (Definition A). --also-global-pareto also prints
  the rows needed to reach --fraction of the global total (Definition B).

top-ops:
  Groups ALL rows by `name`, sums runtime across shapes, and lists every op
  whose summed runtime is at least --threshold (default 4%) of the global
  total and has has_perf_model == False on some row.

Usage (from repo root):
  python3 <skill-dir>/run_other_bucket_triage.py unified_perf_summary.csv
  python3 <skill-dir>/run_other_bucket_triage.py unified_perf_summary.csv \\
    --mode top-ops --threshold 0.04

  # Write a starter extension next to the CSV (<stem>_triage_extension.py):
  python3 <skill-dir>/run_other_bucket_triage.py unified_perf_summary.csv \\
    --emit-extension [--extension-out PATH]

No TraceLens import is needed, except for --check-mapping, which needs
TraceLens importable (`pip install -e .`).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

RUNTIME_COLUMNS = ("Kernel Time (µs)_sum", "total_duration_us")
# Newer TraceLens embeds the call stack in the summary CSV; prefer the full one.
CALL_STACK_COLUMNS = ("call_stack_full", "call_stack", "trunc_call_stack")
REPO_HINTS = (
    (r"aiter/", "AITER (aiter/…)"),
    (r"vllm/", "vLLM (vllm/…)"),
    (r"sglang/", "SGLang (sglang/…)"),
    (r"sgl_kernel/", "SGLang / sgl_kernel"),
    (r"\batom/", "ATOM (atom/…)"),
    (r"torch/_ops\.py|aten::", "PyTorch (torch/aten)"),
    (r"flash_attn", "FlashAttention"),
    (r"triton", "Triton"),
    (r"/tmp/torchinductor", "Inductor cache (/tmp/torchinductor_…)"),
)

Row = Dict[str, str]


def _float(s: Optional[str]) -> float:
    try:
        return float((s or "").strip() or 0.0)
    except ValueError:
        return 0.0


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def _pct(x: float, total: float) -> float:
    return 100.0 * x / total if total > 0 else 0.0


def load_rows(path: Path) -> Tuple[List[Row], str]:
    """Load the summary CSV. Each row gets `call_stack` and its runtime as `_us`."""
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    runtime_col = next((c for c in RUNTIME_COLUMNS if c in fields), None)
    if runtime_col is None:
        raise SystemExit(
            f"{path}: no runtime column; expected one of {RUNTIME_COLUMNS}, got {fields}"
        )

    # Older pipelines kept the call stack in a companion file keyed by row index.
    legacy = path.parent / "unified_perf_callstacks.csv"
    if not {"call_stack_full", "call_stack"} & set(fields) and legacy.is_file():
        with legacy.open(newline="", encoding="utf-8", errors="replace") as f:
            by_id = {
                r.get("row_id", ""): r.get("call_stack", "") for r in csv.DictReader(f)
            }
        for i, r in enumerate(rows):
            r["call_stack"] = by_id.get(str(i), "")

    for r in rows:
        stacks = ((r.get(c) or "").strip() for c in CALL_STACK_COLUMNS)
        r["call_stack"] = next((s for s in stacks if s), "")
        r["_us"] = _float(r.get(runtime_col))
    return rows, runtime_col


def infer_repo_hints(call_stack: str) -> List[str]:
    return [label for pat, label in REPO_HINTS if re.search(pat, call_stack, re.I)]


def greedy_prefix(sorted_rows: Sequence[Row], target: float) -> Tuple[List[Row], float]:
    """Shortest prefix of sorted_rows whose runtime reaches target."""
    picked: List[Row] = []
    acc = 0.0
    if target <= 0:
        return picked, acc
    for r in sorted_rows:
        picked.append(r)
        acc += r["_us"]
        if acc >= target:
            break
    return picked, acc


def unique_names(rows: Sequence[Row]) -> List[str]:
    return list(
        dict.fromkeys(n for n in ((r.get("name") or "").strip() for r in rows) if n)
    )


def report_other_bucket(
    path: Path, rows: List[Row], runtime_col: str, args
) -> List[str]:
    """Print the other-bucket triage; return the Definition A op names."""
    other = [
        r for r in rows if (r.get(args.category_col) or "").strip() == args.other_value
    ]
    other.sort(key=lambda r: r["_us"], reverse=True)
    total_all = sum(r["_us"] for r in rows)
    total_other = sum(r["_us"] for r in other)
    picked, acc = greedy_prefix(other, args.fraction * total_other)

    print("=" * 72)
    print("unified-perf-report-postprocess — other bucket triage")
    print("=" * 72)
    print(f"CSV: {path}")
    print(f"Runtime column: {runtime_col}")
    print(f"Category column: {args.category_col!r} == {args.other_value!r}\n")
    print(f"Rows (all): {len(rows)}")
    print(f"Rows (other): {len(other)}")
    print(f"Total runtime (all rows): {total_all:,.3f} µs")
    print(
        f"Total runtime (other only): {total_other:,.3f} µs  "
        f"({_pct(total_other, total_all):.2f}% of all)\n"
    )
    if total_other > 0:
        print(
            f"Definition A — cumulative {args.fraction:.0%} of time within other only: "
            f"{len(picked)} row(s), covering {acc:,.3f} µs ({_pct(acc, total_other):.2f}% of other)"
        )
    else:
        print("Definition A: N/A (no other time)")
    if args.also_global_pareto:
        picked_g, acc_g = greedy_prefix(other, args.fraction * total_all)
        print(
            f"Definition B — other rows until sum reaches {args.fraction:.0%} of global total: "
            f"{len(picked_g)} row(s), sum {acc_g:,.3f} µs"
        )

    print(
        f"\nTop contributors toward Definition A (name truncated to {args.name_max} chars):"
    )
    print("-" * 72)
    cum = 0.0
    for r in picked:
        cum += r["_us"]
        print(
            f"  {r['_us']:>14,.3f} µs  {_pct(r['_us'], total_other):5.2f}% of other  "
            f"cum {_pct(cum, total_other):5.2f}% of other"
        )
        print(f"    name: {_trunc((r.get('name') or '').strip(), args.name_max)}")
        kernel = (
            r.get("kernel_details_summary") or r.get("trunc_kernel_details") or ""
        ).strip()
        if kernel:
            print(f"    kernel hint: {_trunc(kernel, 120)}")
        print()

    names = unique_names(picked)
    print(f"Unique `name` values in Definition A set: {len(names)}")
    for n in names:
        print(f"  - {_trunc(n, 120)}")

    hints = Counter(h for r in picked for h in infer_repo_hints(r["call_stack"]))
    if hints:
        print("\nInferred source hints (from call_stack on Definition A rows):")
        for h, c in sorted(hints.items(), key=lambda x: (-x[1], x[0])):
            print(f"  [{c} rows] {h}")
    return names


def compute_top_ops(
    rows: List[Row], threshold: float, include_covered: bool = False
) -> List[Dict]:
    """Group rows by `name`; return ops with >= threshold of total runtime, largest first."""
    total = sum(r["_us"] for r in rows)
    by_name: Dict[str, Dict] = {}
    for r in rows:
        name = (r.get("name") or "").strip()
        if not name:
            continue
        e = by_name.setdefault(
            name,
            {
                "name": name,
                "total_us": 0.0,
                "covered": True,
                "op_category": (
                    r.get("op category") or r.get("op_category") or ""
                ).strip(),
                "call_stack": "",
            },
        )
        e["total_us"] += r["_us"]
        e["covered"] &= (r.get("has_perf_model") or "").strip().lower() in (
            "true",
            "1",
            "yes",
        )
        if len(r["call_stack"]) > len(e["call_stack"]):
            e["call_stack"] = r["call_stack"]

    results = []
    for e in by_name.values():
        e["pct_global"] = _pct(e["total_us"], total)
        if (
            total > 0
            and e["pct_global"] >= threshold * 100.0
            and (include_covered or not e["covered"])
        ):
            e["repo_hints"] = infer_repo_hints(e["call_stack"])
            results.append(e)
    return sorted(results, key=lambda e: e["total_us"], reverse=True)


def report_top_ops(path: Path, rows: List[Row], args) -> List[str]:
    """Print the top-ops candidate table; return the op names."""
    results = compute_top_ops(rows, args.threshold, args.include_covered)
    print("=" * 72)
    print("unified-perf-report-postprocess — top-ops mode (EP1)")
    print("=" * 72)
    print(f"CSV: {path}")
    print(
        f"Threshold: >={args.threshold * 100:.1f}% of global total  |  has_perf_model == False"
    )
    print(f"Global total runtime: {sum(r['_us'] for r in rows):,.0f} µs\n")
    if not results:
        print("No ops found above threshold with missing perf model.")
        return []

    print(
        f"{'Op name':<50}  {'Sum µs':>12}  {'% global':>8}  {'Category':<20}  Repo hints"
    )
    print("-" * 110)
    for e in results:
        print(
            f"  {_trunc(e['name'], args.name_max):<48}  {e['total_us']:>12,.0f}  "
            f"{e['pct_global']:>7.2f}%  {e['op_category'][:18]:<20}  {', '.join(e['repo_hints'][:2])}"
        )
        if e["call_stack"]:
            print(f"    stack: {_trunc(e['call_stack'], 123)}")
        print()
    print(f"Total candidate ops: {len(results)}")
    return [e["name"] for e in results]


EXTENSION_TEMPLATE = '''\
"""
Auto-generated by run_other_bucket_triage.py --emit-extension.

Source CSV: {csv_path}
Selection: {selection}

Next steps:
  - Map stable op names to perf model classes in perf_model_extension.
  - Map category-only op names (no perf model) in op_category_extension.
  - Do not give (Synthetic Op) names a perf model; their names are unstable.

Pass to the report generator:
  --extension_file {out_path}
"""

# Candidate profiler `name` strings from triage, in ranked order.
TRIAGE_OP_NAMES = (
{names}
)

# name -> perf model class
perf_model_extension = {{}}

# name -> category, for ops without a perf model
op_category_extension = {{}}
'''


def write_extension(
    out_path: Path, csv_path: Path, op_names: Sequence[str], selection: str
) -> None:
    """Write a starter --extension_file module for the triaged op names."""
    names = (
        "\n".join(f"    {n!r}," for n in op_names) or "    # (no candidate op names)"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        EXTENSION_TEMPLATE.format(
            csv_path=csv_path, selection=selection, out_path=out_path, names=names
        ),
        encoding="utf-8",
    )


def builtin_perf_model_map() -> Optional[Dict[str, type]]:
    """TraceLens' built-in name -> perf model map (incl. pseudo ops), or None."""
    try:
        from TraceLens.PerfModel.torch_op_mapping import op_to_perf_model_class_map
    except ImportError as e:
        print(
            f"\nMapping check skipped, TraceLens is not importable ({e}). "
            "Run `pip install -e .` in the TraceLens repo.",
            file=sys.stderr,
        )
        return None
    return op_to_perf_model_class_map


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("csv_path", type=Path, help="Path to unified_perf_summary.csv")
    p.add_argument(
        "--mode", choices=["other-bucket", "top-ops"], default="other-bucket"
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.04,
        metavar="FRAC",
        help="top-ops: fraction of global runtime (default 0.04 = 4%%).",
    )
    p.add_argument(
        "--include-covered",
        action="store_true",
        help="top-ops: also show ops that already have a perf model.",
    )
    p.add_argument(
        "--category-col", default="op category", help="Category column header"
    )
    p.add_argument("--other-value", default="other", help='Value meaning "other"')
    p.add_argument(
        "--fraction",
        type=float,
        default=0.95,
        help="other-bucket: cumulative fraction for Definitions A and B",
    )
    p.add_argument(
        "--name-max", type=int, default=100, help="Truncate printed op names"
    )
    p.add_argument(
        "--also-global-pareto",
        action="store_true",
        help="other-bucket: also print Definition B (fraction of global total)",
    )
    p.add_argument(
        "--check-mapping",
        action="store_true",
        help="Print whether each candidate is in TraceLens' built-in perf model map",
    )
    p.add_argument(
        "--emit-extension",
        action="store_true",
        help="Write a starter --extension_file module",
    )
    p.add_argument(
        "--extension-out",
        type=Path,
        metavar="PATH",
        help="Output for --emit-extension (default: <csv_stem>_triage_extension.py next to the CSV)",
    )
    args = p.parse_args(argv)

    path = args.csv_path
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 1
    rows, runtime_col = load_rows(path)

    if args.mode == "top-ops":
        names = report_top_ops(path, rows, args)
        selection = f"top-ops, >= {args.threshold:g} of global runtime"
    else:
        names = report_other_bucket(path, rows, runtime_col, args)
        selection = f"other-bucket, cumulative {args.fraction:g} of other runtime"

    if args.emit_extension:
        out = args.extension_out or path.with_name(path.stem + "_triage_extension.py")
        write_extension(out, path, names, selection)
        print(f"\nWrote generated extension module: {out.resolve()}")

    if args.check_mapping:
        mapping = builtin_perf_model_map()
        if mapping is not None:
            print("\nBuilt-in perf model map membership (exact `name` key):")
            for n in names:
                print(f"  {n in mapping!s:<5}  {_trunc(n, 100)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
