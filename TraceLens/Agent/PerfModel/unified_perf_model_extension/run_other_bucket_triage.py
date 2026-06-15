#!/usr/bin/env python3
"""
Step 1 triage for unified-perf-report-postprocess: load unified_perf_summary.csv,
summarize "other" bucket runtime, cumulative 95% (within-other and optional global),
and emit per-row fields for downstream triage.

No TraceLens import required. Optional --check-mapping uses the repo checkout.

Modes
-----
other-bucket (default):
  Filters rows where op category == "other", ranks by runtime, and shows the
  cumulative-95% set (Definition A within-other, optional Definition B global).

top-ops:
  Groups ALL rows by `name`, sums runtime across shapes, and lists every op
  whose summed runtime exceeds --threshold (default 4%) of the global total
  AND has has_perf_model == False.  Use this to find high-impact ops missing
  perf models regardless of their category.

Usage (from repo root):
  python3 <skill-dir>/run_other_bucket_triage.py \\
    perf_skill_debug/unified_perf_summary.csv

  # top-ops mode (EP1 entry point):
  python3 <skill-dir>/run_other_bucket_triage.py \\
    perf_skill_debug/unified_perf_summary.csv --mode top-ops --threshold 0.04

  # Write a starter extension next to the CSV (<stem>_triage_extension.py):
  python3 <skill-dir>/run_other_bucket_triage.py \\
    perf_skill_debug/unified_perf_summary.csv --emit-extension

  # Explicit output path:
  python3 <skill-dir>/run_other_bucket_triage.py \\
    perf_skill_debug/unified_perf_summary.csv \\
    --emit-extension --extension-out /path/to/my_triage_extension.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Column names as emitted by generate_perf_report_pytorch_inference-style pipelines
RUNTIME_COLUMNS_PREFERENCE: Sequence[str] = (
    "Kernel Time (µs)_sum",
    "total_duration_us",
)
CATEGORY_COLUMN_DEFAULT = "op category"
OTHER_VALUE_DEFAULT = "other"


def _parse_float(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def pick_runtime_column(fieldnames: Sequence[str]) -> str:
    for c in RUNTIME_COLUMNS_PREFERENCE:
        if c in fieldnames:
            return c
    raise SystemExit(
        "No known runtime column found. Expected one of: "
        + ", ".join(RUNTIME_COLUMNS_PREFERENCE)
        + f". Got: {list(fieldnames)}"
    )


def load_rows(path: Path) -> Tuple[List[Dict[str, str]], str]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"Empty or headerless CSV: {path}")
        runtime_col = pick_runtime_column(reader.fieldnames)
        rows = list(reader)

    # Merge full call_stack from companion file if available
    callstacks_path = path.parent / "unified_perf_callstacks.csv"
    if callstacks_path.is_file() and "call_stack" not in (rows[0] if rows else {}):
        with callstacks_path.open(newline="", encoding="utf-8", errors="replace") as f:
            cs_reader = csv.DictReader(f)
            cs_rows = list(cs_reader)
        cs_by_id = {r.get("row_id", ""): r.get("call_stack", "") for r in cs_rows}
        for i, r in enumerate(rows):
            r["call_stack"] = cs_by_id.get(str(i), "")

    return rows, runtime_col


def filter_other(
    rows: List[Dict[str, str]],
    category_col: str,
    other_value: str,
) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        if (r.get(category_col) or "").strip() == other_value:
            out.append(r)
    return out


def row_runtime(r: Dict[str, str], runtime_col: str) -> float:
    return _parse_float(r.get(runtime_col, ""))


def cumulative_until_fraction(
    sorted_rows: List[Dict[str, str]],
    runtime_col: str,
    total: float,
    fraction: float,
) -> Tuple[List[Dict[str, str]], float]:
    """Greedy prefix of sorted_rows until cumulative runtime >= fraction * total."""
    if total <= 0:
        return [], 0.0
    target = fraction * total
    acc = 0.0
    picked: List[Dict[str, str]] = []
    for r in sorted_rows:
        t = row_runtime(r, runtime_col)
        picked.append(r)
        acc += t
        if acc >= target:
            break
    return picked, acc


def first_kernel_hint(row: Dict[str, str]) -> str:
    raw = row.get("kernel_details_summary") or row.get("trunc_kernel_details") or ""
    raw = raw.strip()
    if len(raw) > 120:
        return raw[:117] + "..."
    return raw


def infer_repo_hints(call_stack: str) -> List[str]:
    """Lightweight path/prefix hints for Step 1b (deduplicated order)."""
    if not call_stack:
        return []
    hints: List[str] = []
    patterns = [
        (r"aiter/", "AITER (aiter/…)"),
        (r"vllm/", "vLLM (vllm/…)"),
        (r"sglang/", "SGLang (sglang/…)"),
        (r"sgl_kernel/", "SGLang / sgl_kernel"),
        (r"torch/_ops\.py|aten::", "PyTorch (torch/aten)"),
        (r"flash_attn|FlashAttn", "FlashAttention"),
        (r"triton", "Triton"),
        (r"/tmp/torchinductor", "Inductor cache (/tmp/torchinductor_…)"),
    ]
    seen = set()
    for pat, label in patterns:
        if re.search(pat, call_stack, re.I) and label not in seen:
            seen.add(label)
            hints.append(label)
    return hints


def compute_definition_a(
    other_rows: List[Dict[str, str]],
    runtime_col: str,
    fraction: float,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], float, float]:
    """Sort `other` rows by runtime, return (sorted_other, picked_a, acc_a, total_other)."""
    total_other = sum(row_runtime(r, runtime_col) for r in other_rows)
    sorted_other = sorted(other_rows, key=lambda r: row_runtime(r, runtime_col), reverse=True)
    picked_a, acc_a = cumulative_until_fraction(sorted_other, runtime_col, total_other, fraction)
    return sorted_other, picked_a, acc_a, total_other


def unique_op_names_in_order(picked_rows: Sequence[Dict[str, str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for r in picked_rows:
        n = (r.get("name") or "").strip()
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def write_generated_extension(
    out_path: Path,
    csv_path: Path,
    op_names: Sequence[str],
    fraction: float,
) -> None:
    """Emit a loadable TraceLens --extension_file module (stdlib + triage metadata only)."""
    lines_body = ["    " + repr(n) + "," for n in op_names]
    names_block = "\n".join(lines_body) if lines_body else "    # (no op names in Definition A set)"

    content = f'''###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Auto-generated by run_other_bucket_triage.py (--emit-extension).

Source CSV: {csv_path}
Definition A (within-other cumulative fraction): {fraction:g}

Next steps:
  - Add subclasses to perf_model_extension for stable, non-synthetic op names.
  - Extend dict_cat2names_extension so each mapped name appears under a categorize_torch_op label.
  - Keep categorize_extension for (Synthetic Op) and category-only rows (no perf model).

Regenerate after the CSV changes:
  python3 .../run_other_bucket_triage.py <csv> --emit-extension [--extension-out PATH]

Pass to the report generator:
  --extension_file {out_path}
"""

from __future__ import annotations

# Full profiler `name` strings from the triage Definition A set (deduplicated, stable order).
_DEFINITION_A_OP_NAMES = (
{names_block}
)


def categorize_extension(row, plugin):
    """Category-only path for synthetic / uncertain ops; return None to use default categorizer."""
    name = row.get("name")
    if not name:
        return None
    if "(Synthetic Op)" in name:
        if "batched_gemm_a8w8" in name:
            return "GEMM"
        if "unified_mla_attention_with_output" in name:
            return "InferenceAttention"
        if "fused_moe_" in name or name.startswith("aiter::fused_moe_"):
            return "MoE_aux"
        return None
    return None


# TODO: name -> perf model class (do not register synthetic / (Synthetic Op) names).
perf_model_extension = {{
}}

# TODO: category string -> list of profiler names (must match categorize_torch_op vocabulary).
dict_cat2names_extension = {{
}}
'''
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")


def print_report(
    path: Path,
    rows: List[Dict[str, str]],
    other_rows: List[Dict[str, str]],
    runtime_col: str,
    category_col: str,
    other_value: str,
    fraction: float,
    name_max: int,
    print_global_pareto: bool,
    sorted_other: List[Dict[str, str]],
    picked_a: List[Dict[str, str]],
    acc_a: float,
    total_other: float,
) -> None:
    total_all = sum(row_runtime(r, runtime_col) for r in rows)
    share_other_pct = (100.0 * total_other / total_all) if total_all > 0 else 0.0

    print("=" * 72)
    print("unified-perf-report-postprocess — Step 1 (other bucket triage)")
    print("=" * 72)
    print(f"CSV: {path}")
    print(f"Runtime column: {runtime_col}")
    print(f"Category column: {category_col!r} == {other_value!r}")
    print()
    print(f"Rows (all): {len(rows)}")
    print(f"Rows (other): {len(other_rows)}")
    print(f"Total runtime (all rows): {total_all:,.3f} µs")
    print(f"Total runtime (other only): {total_other:,.3f} µs  ({share_other_pct:.2f}% of all)")
    print()
    print(
        f"Definition A — cumulative {fraction:.0%} of time **within other only**: "
        f"{len(picked_a)} row(s), covering {acc_a:,.3f} µs "
        f"({100.0 * acc_a / total_other:.2f}% of other)" if total_other > 0 else "Definition A: N/A (no other time)"
    )

    if print_global_pareto and total_all > 0:
        target_g = fraction * total_all
        acc_g = 0.0
        picked_g: List[Dict[str, str]] = []
        for r in sorted_other:
            acc_g += row_runtime(r, runtime_col)
            picked_g.append(r)
            if acc_g >= target_g:
                break
        print(
            f"Definition B — greedy other rows until sum reaches {fraction:.0%} of **global** total: "
            f"{len(picked_g)} row(s), sum {acc_g:,.3f} µs"
        )

    print()
    print(f"Top contributors toward Definition A (name truncated to {name_max} chars):")
    print("-" * 72)
    cum = 0.0
    for r in picked_a:
        t = row_runtime(r, runtime_col)
        cum += t
        name = (r.get("name") or "").strip()
        disp = name if len(name) <= name_max else name[: name_max - 3] + "..."
        pct_o = (100.0 * t / total_other) if total_other > 0 else 0.0
        cum_pct_o = (100.0 * cum / total_other) if total_other > 0 else 0.0
        print(f"  {t:>14,.3f} µs  {pct_o:5.2f}% of other  cum {cum_pct_o:5.2f}% of other")
        print(f"    name: {disp}")
        kh = first_kernel_hint(r)
        if kh:
            print(f"    kernel hint: {kh[:200]}{'...' if len(kh) > 200 else ''}")
        print()

    uniq = unique_op_names_in_order(picked_a)
    print(f"Unique `name` values in Definition A set: {len(uniq)}")
    for n in uniq:
        print(f"  - {n if len(n) <= 120 else n[:117] + '...'}")

    # Step 1b: aggregate repo hints from call stacks of picked rows
    hint_counts: Dict[str, int] = defaultdict(int)
    for r in picked_a:
        cs = r.get("call_stack") or r.get("trunc_call_stack") or ""
        for h in infer_repo_hints(cs):
            hint_counts[h] += 1
    if hint_counts:
        print()
        print("Step 1b — inferred source hints (from call_stack on Definition A rows):")
        for h, c in sorted(hint_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  [{c} rows] {h}")


def try_mapping_check(names: Sequence[str], repo_root: Path) -> None:
    """Optional: verify op names against torch_op_mapping (needs importable TraceLens)."""
    root = repo_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from TraceLens.PerfModel import torch_op_mapping as tom  # type: ignore
    except Exception as e:
        print()
        print(f"Optional mapping check skipped (import failed): {e}")
        return

    print()
    print("Optional — torch_op_mapping membership (exact `name` key):")
    for name in names:
        in_map = name in tom.op_to_perf_model_class_map
        print(f"  in op_to_perf_model_class_map: {in_map}  {name[:100]}{'...' if len(name) > 100 else ''}")


def _parse_bool(s: str) -> bool:
    return (s or "").strip().lower() in ("true", "1", "yes")


def compute_top_ops(
    rows: List[Dict[str, str]],
    runtime_col: str,
    threshold: float,
    require_missing_perf_model: bool = True,
) -> List[Dict[str, object]]:
    """Group rows by `name`, sum runtime, return ops above threshold.

    Returns list of dicts with keys:
      name, total_us, pct_global, has_perf_model, op_category,
      rep_input_dims, rep_input_type, rep_perf_params, repo_hints, call_stack
    sorted descending by total_us.
    """
    total_all = sum(row_runtime(r, runtime_col) for r in rows)
    if total_all <= 0:
        return []

    by_name: Dict[str, Dict] = {}
    for r in rows:
        name = (r.get("name") or "").strip()
        if not name:
            continue
        if name not in by_name:
            by_name[name] = {
                "name": name,
                "total_us": 0.0,
                "has_perf_model_any": False,
                "has_perf_model_all": True,
                "op_category": (r.get("op category") or r.get("op_category") or "").strip(),
                "rep_input_dims": r.get("Input Dims", ""),
                "rep_input_type": r.get("Input type", ""),
                "rep_perf_params": r.get("perf_params", ""),
                "call_stack": r.get("call_stack") or r.get("trunc_call_stack") or "",
                "row_count": 0,
            }
        entry = by_name[name]
        entry["total_us"] += row_runtime(r, runtime_col)
        entry["row_count"] += 1
        hpm = _parse_bool(r.get("has_perf_model", "False"))
        if hpm:
            entry["has_perf_model_any"] = True
        else:
            entry["has_perf_model_all"] = False
        # prefer longer call stack for display
        cs = r.get("call_stack") or r.get("trunc_call_stack") or ""
        if len(cs) > len(entry["call_stack"]):
            entry["call_stack"] = cs

    results = []
    for entry in by_name.values():
        pct = 100.0 * entry["total_us"] / total_all
        if pct < threshold * 100.0:
            continue
        # Filter: keep only ops missing a perf model (default) or all ops
        if require_missing_perf_model and entry["has_perf_model_all"]:
            continue
        entry["pct_global"] = pct
        entry["repo_hints"] = infer_repo_hints(entry["call_stack"])
        results.append(entry)

    results.sort(key=lambda e: e["total_us"], reverse=True)
    return results


def print_top_ops_report(
    path: Path,
    results: List[Dict[str, object]],
    total_all_us: float,
    threshold: float,
    name_max: int,
) -> List[str]:
    """Print the top-ops candidate table; return list of op names."""
    print("=" * 72)
    print("unified-perf-report-postprocess — top-ops mode (EP1)")
    print("=" * 72)
    print(f"CSV: {path}")
    print(f"Threshold: >{threshold * 100:.1f}% of global total  |  has_perf_model == False")
    print(f"Global total runtime: {total_all_us:,.0f} µs")
    print()

    if not results:
        print("No ops found above threshold with missing perf model.")
        return []

    print(f"{'Op name':<50}  {'Sum µs':>12}  {'% global':>8}  {'Category':<20}  {'Repo hints'}")
    print("-" * 110)
    op_names = []
    for e in results:
        name = e["name"]
        disp = name if len(name) <= name_max else name[: name_max - 3] + "..."
        cat = (e["op_category"] or "")[:18]
        hints = ", ".join(e["repo_hints"][:2]) if e["repo_hints"] else ""
        print(f"  {disp:<48}  {e['total_us']:>12,.0f}  {e['pct_global']:>7.2f}%  {cat:<20}  {hints}")
        if e["call_stack"]:
            cs_short = e["call_stack"][:120]
            print(f"    stack: {cs_short}{'...' if len(e['call_stack']) > 120 else ''}")
        print()
        op_names.append(name)

    print(f"Total candidate ops: {len(results)}")
    return op_names


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "csv_path",
        type=Path,
        help="Path to unified_perf_summary.csv (or unified_perf_report.csv)",
    )
    p.add_argument(
        "--mode",
        choices=["other-bucket", "top-ops"],
        default="other-bucket",
        help=(
            "other-bucket (default): triage rows in the 'other' category. "
            "top-ops: group ALL rows by name, list ops >--threshold of global total with missing perf model."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.04,
        metavar="FRAC",
        help="Fraction of global total runtime (top-ops mode only). Default 0.04 = 4%%.",
    )
    p.add_argument(
        "--include-covered",
        action="store_true",
        help="top-ops mode: also show ops that already have has_perf_model == True.",
    )
    p.add_argument("--category-col", default=CATEGORY_COLUMN_DEFAULT, help="Category column header")
    p.add_argument("--other-value", default=OTHER_VALUE_DEFAULT, help='Value meaning "other"')
    p.add_argument(
        "--fraction",
        type=float,
        default=0.95,
        help="Cumulative fraction for Definition A (within-other) and B (if enabled)",
    )
    p.add_argument("--name-max", type=int, default=100, help="Truncate printed op names")
    p.add_argument(
        "--also-global-pareto",
        action="store_true",
        help="Also print Definition B (greedy other rows until fraction of global total)",
    )
    p.add_argument(
        "--check-mapping",
        action="store_true",
        help="After triage, try importing TraceLens and print op_to_perf_model_class_map hits for A-set names",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root on sys.path for --check-mapping (default: parent of .cursor/skills/...)",
    )
    p.add_argument(
        "--emit-extension",
        action="store_true",
        help="Write generated --extension_file module (see --extension-out for path).",
    )
    p.add_argument(
        "--extension-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output .py for --emit-extension (default: <csv_stem>_triage_extension.py next to the CSV).",
    )
    args = p.parse_args(argv)

    path = args.csv_path
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 1

    rows, runtime_col = load_rows(path)

    if args.mode == "top-ops":
        total_all = sum(row_runtime(r, runtime_col) for r in rows)
        results = compute_top_ops(
            rows,
            runtime_col,
            args.threshold,
            require_missing_perf_model=not args.include_covered,
        )
        a_names = print_top_ops_report(path, results, total_all, args.threshold, args.name_max)
    else:
        other_rows = filter_other(rows, args.category_col, args.other_value)
        sorted_other, picked_a, acc_a, total_other = compute_definition_a(
            other_rows, runtime_col, args.fraction
        )
        print_report(
            path,
            rows,
            other_rows,
            runtime_col,
            args.category_col,
            args.other_value,
            args.fraction,
            args.name_max,
            args.also_global_pareto,
            sorted_other,
            picked_a,
            acc_a,
            total_other,
        )
        a_names = unique_op_names_in_order(picked_a)

    if args.emit_extension:
        ext_path = args.extension_out
        if ext_path is None:
            ext_path = path.with_name(path.stem + "_triage_extension.py")
        write_generated_extension(ext_path, path, a_names, getattr(args, "fraction", 0.95))
        print()
        print(f"Wrote generated extension module: {ext_path.resolve()}")

    if args.check_mapping:
        repo_root = args.repo_root
        if repo_root is None:
            here = Path(__file__).resolve()
            repo_root = here.parents[3] if len(here.parents) >= 4 else Path.cwd()
        try_mapping_check(a_names, repo_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
