#!/usr/bin/env python3
"""
EP3 — Framework source scanner for unified-perf-model-extension.

Statically scans one or more framework repos (aiter, vLLM, sglang / sgl_kernel)
to enumerate CPU-registered GPU ops that TraceLens can model.  Optionally
cross-checks against a unified_perf_callstacks.csv or unified_perf_summary.csv
trace file to annotate which ops appeared in real traces.

Usage
-----
# Scan all frameworks under a common parent:
  python3 scan_framework_ops.py --repo-root /work

# Scan a specific framework dir with optional filter:
  python3 scan_framework_ops.py --aiter /work/aiter --filter attention

# Cross-check against trace callstacks:
  python3 scan_framework_ops.py --aiter /work/aiter \\
      --trace unified_perf_callstacks.csv

# Write the result table to JSON for downstream use (e.g. emit_perf_model.py):
  python3 scan_framework_ops.py --repo-root /work --filter attention \\
      --output-json /tmp/attention_ops.json

Repo registration patterns
--------------------------
aiter     — @compile_ops decorator or direct_register_custom_op call; namespace "aiter::"
vLLM      — torch.library.define / _custom_op / ops.register_fake; namespace "vllm::"
sglang    — sglang_profiler namespace; sgl_kernel uses torch._custom_op; "sgl_kernel::"
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

FRAMEWORKS = ("aiter", "vllm", "sglang")

# Regex patterns per framework for finding CPU-side op registrations.
# Each entry: (label, compiled_regex_for_op_name_extraction)
_AITER_PATTERNS = [
    # @compile_ops("aiter::my_op")
    re.compile(r'@\s*compile_ops\s*\(\s*["\']aiter::([^"\']+)["\']'),
    # direct_register_custom_op("aiter::my_op", ...)
    re.compile(r'direct_register_custom_op\s*\(\s*["\']aiter::([^"\']+)["\']'),
    # torch.library.define("aiter::my_op", ...)
    re.compile(r'torch\.library\.define\s*\(\s*["\']aiter::([^"\']+)["\']'),
]

_VLLM_PATTERNS = [
    # @torch.library.custom_op("vllm::my_op", ...)
    re.compile(r'(?:custom_op|define)\s*\(\s*["\']vllm::([^"\']+)["\']'),
    # direct_register_custom_op("vllm::my_op", ...)
    re.compile(r'direct_register_custom_op\s*\(\s*["\']vllm::([^"\']+)["\']'),
    # ops.register_custom_op("my_op") in vllm._custom_ops — plain name, no namespace
    re.compile(r'register_custom_op\s*\(\s*["\']([^"\']+)["\']'),
]

_SGLANG_PATTERNS = [
    re.compile(r'(?:custom_op|define)\s*\(\s*["\']sgl_kernel::([^"\']+)["\']'),
    re.compile(r'(?:custom_op|define)\s*\(\s*["\']sglang_profiler::([^"\']+)["\']'),
    re.compile(r'direct_register_custom_op\s*\(\s*["\']sgl_kernel::([^"\']+)["\']'),
    re.compile(r'direct_register_custom_op\s*\(\s*["\']sglang_profiler::([^"\']+)["\']'),
]

# Sibling benchmark / test files we surface for theoretical roofline reference.
_BENCH_PATTERNS = re.compile(r"^(?:test_|benchmark_|bench_)", re.IGNORECASE)

# Skip venv / build dirs.
_SKIP_DIRS = {".git", "__pycache__", "build", "dist", ".tox", "node_modules", ".venv", "venv"}


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------


def iter_py_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def _read_safe(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _find_sibling_test_benchmarks(py_file: Path) -> List[str]:
    """Return names of test_*/benchmark_* siblings in the same directory."""
    try:
        siblings = [
            f.name
            for f in py_file.parent.iterdir()
            if f.is_file() and _BENCH_PATTERNS.match(f.name)
        ]
        return sorted(siblings)
    except OSError:
        return []


def _extract_kernel_calls(source: str, op_name: str) -> List[str]:
    """Heuristic: find C++ / HIP / Triton kernel calls near the op registration.

    Looks within 60 lines after the registration line for:
      - calls that look like kernel invocations (identifiers ending in _kernel, _fwd, _hip, _triton)
      - torch.ops.xxx calls
    Returns a deduplicated list.
    """
    lines = source.splitlines()
    hits: List[str] = []
    kernel_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:_kernel|_fwd|_bwd|_hip|_triton|_cuda|_rocm|_gemm|_attn))\s*\(")
    torch_ops_re = re.compile(r"torch\.ops\.([\w.]+)\s*\(")
    for i, line in enumerate(lines):
        if op_name in line:
            window = lines[i : i + 60]
            for wl in window:
                for m in kernel_re.finditer(wl):
                    hits.append(m.group(1))
                for m in torch_ops_re.finditer(wl):
                    hits.append("torch.ops." + m.group(1))
    # Deduplicate preserving first-occurrence order.
    seen: Set[str] = set()
    result: List[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            result.append(h)
    return result[:8]


# ---------------------------------------------------------------------------
# Per-framework scanners
# ---------------------------------------------------------------------------


def scan_aiter(repo_root: Path, filter_kw: Optional[str] = None) -> List[Dict]:
    results: List[Dict] = []
    for py_file in iter_py_files(repo_root):
        source = _read_safe(py_file)
        for pat in _AITER_PATTERNS:
            for m in pat.finditer(source):
                op_base = m.group(1).strip()
                full_name = f"aiter::{op_base}"
                if filter_kw and filter_kw.lower() not in full_name.lower() and filter_kw.lower() not in source.lower():
                    continue
                results.append({
                    "framework": "aiter",
                    "op_name": full_name,
                    "file": str(py_file),
                    "kernel_hints": _extract_kernel_calls(source, op_base),
                    "sibling_benchmarks": _find_sibling_test_benchmarks(py_file),
                    "in_trace": False,
                    "trace_pct": None,
                })
    return _dedup_by_op(results)


def scan_vllm(repo_root: Path, filter_kw: Optional[str] = None) -> List[Dict]:
    results: List[Dict] = []
    for py_file in iter_py_files(repo_root):
        source = _read_safe(py_file)
        for pat in _VLLM_PATTERNS:
            for m in pat.finditer(source):
                op_base = m.group(1).strip()
                # Skip vague matches like "custom_op" pattern in the ops.register_custom_op case
                # that picks up the helper itself.
                if op_base in ("custom_op", "define", "register_custom_op"):
                    continue
                full_name = f"vllm::{op_base}" if "::" not in op_base else op_base
                if filter_kw and filter_kw.lower() not in full_name.lower() and filter_kw.lower() not in source.lower():
                    continue
                results.append({
                    "framework": "vllm",
                    "op_name": full_name,
                    "file": str(py_file),
                    "kernel_hints": _extract_kernel_calls(source, op_base),
                    "sibling_benchmarks": _find_sibling_test_benchmarks(py_file),
                    "in_trace": False,
                    "trace_pct": None,
                })
    return _dedup_by_op(results)


def scan_sglang(repo_root: Path, filter_kw: Optional[str] = None) -> List[Dict]:
    results: List[Dict] = []
    for py_file in iter_py_files(repo_root):
        source = _read_safe(py_file)
        for pat in _SGLANG_PATTERNS:
            for m in pat.finditer(source):
                op_base = m.group(1).strip()
                if "::" in op_base:
                    full_name = op_base
                else:
                    full_name = f"sgl_kernel::{op_base}"
                if filter_kw and filter_kw.lower() not in full_name.lower() and filter_kw.lower() not in source.lower():
                    continue
                results.append({
                    "framework": "sglang",
                    "op_name": full_name,
                    "file": str(py_file),
                    "kernel_hints": _extract_kernel_calls(source, op_base),
                    "sibling_benchmarks": _find_sibling_test_benchmarks(py_file),
                    "in_trace": False,
                    "trace_pct": None,
                })
    return _dedup_by_op(results)


def _dedup_by_op(rows: List[Dict]) -> List[Dict]:
    """Keep first occurrence per op_name; merge kernel_hints across occurrences."""
    seen: Dict[str, Dict] = {}
    for r in rows:
        key = r["op_name"]
        if key not in seen:
            seen[key] = r
        else:
            existing = seen[key]
            for h in r["kernel_hints"]:
                if h not in existing["kernel_hints"]:
                    existing["kernel_hints"].append(h)
            for b in r["sibling_benchmarks"]:
                if b not in existing["sibling_benchmarks"]:
                    existing["sibling_benchmarks"].append(b)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Trace cross-check
# ---------------------------------------------------------------------------


def load_trace_ops(trace_csv: Path) -> Dict[str, float]:
    """Load a unified_perf_callstacks.csv or unified_perf_summary.csv.

    Returns {op_name: total_us} for ops present in the trace.
    """
    op_times: Dict[str, float] = {}
    try:
        with trace_csv.open(newline="", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                for col in ("Duration (us)", "duration_us", "mean_us", "total_us", "time_us"):
                    val = row.get(col, "")
                    if val:
                        try:
                            op_times[name] = op_times.get(name, 0.0) + float(val)
                            break
                        except ValueError:
                            pass
    except OSError as e:
        print(f"Warning: could not read trace CSV {trace_csv}: {e}", file=sys.stderr)
    return op_times


def cross_check_trace(ops: List[Dict], trace_ops: Dict[str, float]) -> None:
    """Annotate each op dict with in_trace and trace_pct fields in-place."""
    total = sum(trace_ops.values())
    for op in ops:
        name = op["op_name"]
        # Try exact match and bare-name match (without namespace).
        bare = name.split("::", 1)[-1]
        matched_val = trace_ops.get(name) or trace_ops.get(bare)
        if matched_val is not None:
            op["in_trace"] = True
            op["trace_pct"] = (100.0 * matched_val / total) if total > 0 else 0.0
        else:
            op["in_trace"] = False
            op["trace_pct"] = 0.0


# ---------------------------------------------------------------------------
# TraceLens mapping check
# ---------------------------------------------------------------------------


def _try_import_tracerlens_map(repo_root: Path) -> Optional[Dict[str, str]]:
    sys.path.insert(0, str(repo_root))
    try:
        from TraceLens.PerfModel import torch_op_mapping as tom  # type: ignore
        return getattr(tom, "op_to_perf_model_class_map", None)
    except Exception:
        try:
            import torch_op_mapping as tom  # type: ignore
            return getattr(tom, "op_to_perf_model_class_map", None)
        except Exception:
            return None
    finally:
        if str(repo_root) in sys.path:
            sys.path.remove(str(repo_root))


def annotate_perf_model_coverage(ops: List[Dict], repo_root: Optional[Path]) -> None:
    """Add has_perf_model field using TraceLens mapping if available."""
    mapping = None
    if repo_root:
        mapping = _try_import_tracerlens_map(repo_root)
    for op in ops:
        name = op["op_name"]
        bare = name.split("::", 1)[-1]
        if mapping is not None:
            op["has_perf_model"] = (name in mapping) or (bare in mapping)
        else:
            op["has_perf_model"] = None  # unknown


# ---------------------------------------------------------------------------
# Output / display
# ---------------------------------------------------------------------------


def print_results(ops: List[Dict], name_max: int = 60) -> None:
    if not ops:
        print("No ops found.")
        return

    # Group by framework
    by_fw: Dict[str, List[Dict]] = {}
    for op in ops:
        by_fw.setdefault(op["framework"], []).append(op)

    for fw, fw_ops in sorted(by_fw.items()):
        print(f"\n{'=' * 70}")
        print(f"  Framework: {fw}  ({len(fw_ops)} ops)")
        print(f"{'=' * 70}")
        for op in fw_ops:
            name = op["op_name"]
            disp = name if len(name) <= name_max else name[: name_max - 3] + "..."
            in_trace = op.get("in_trace", False)
            pct = op.get("trace_pct")
            hpm = op.get("has_perf_model")
            model_tag = "" if hpm is None else (" [HAS MODEL]" if hpm else " [NO MODEL]")
            trace_tag = f"  trace:{pct:.1f}%" if in_trace and pct is not None else ("  in-trace" if in_trace else "")
            print(f"  {disp}{model_tag}{trace_tag}")
            rel_file = op.get("file", "")
            print(f"    file: {rel_file}")
            if op.get("kernel_hints"):
                print(f"    kernels: {', '.join(op['kernel_hints'][:5])}")
            if op.get("sibling_benchmarks"):
                print(f"    roofline refs: {', '.join(op['sibling_benchmarks'][:4])}")
    print()
    print(f"Total ops found: {len(ops)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_framework_root(base: Optional[Path], name: str) -> Optional[Path]:
    """Try <base>/<name> if base is set, else None."""
    if base is None:
        return None
    candidate = base / name
    return candidate if candidate.is_dir() else None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    root_group = p.add_argument_group("repo locations")
    root_group.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Common parent directory containing aiter/, vllm/, and sglang/ subdirs. "
            "Used as fallback when --aiter/--vllm/--sglang are not given."
        ),
    )
    root_group.add_argument("--aiter", type=Path, default=None, metavar="DIR", help="Path to aiter repo root")
    root_group.add_argument("--vllm", type=Path, default=None, metavar="DIR", help="Path to vllm repo root")
    root_group.add_argument(
        "--sglang",
        type=Path,
        default=None,
        metavar="DIR",
        help="Path to sglang or sgl_kernel repo root",
    )

    filter_group = p.add_argument_group("filtering")
    filter_group.add_argument(
        "--filter",
        metavar="KEYWORD",
        default=None,
        help=(
            "Case-insensitive keyword to filter op names and source files "
            "(e.g. 'attention', 'gemm', 'decode')."
        ),
    )
    filter_group.add_argument(
        "--frameworks",
        nargs="+",
        choices=FRAMEWORKS,
        default=list(FRAMEWORKS),
        help="Which frameworks to scan (default: all three).",
    )

    trace_group = p.add_argument_group("trace cross-check")
    trace_group.add_argument(
        "--trace",
        type=Path,
        default=None,
        metavar="CSV",
        help="unified_perf_callstacks.csv or unified_perf_summary.csv to cross-check ops against.",
    )

    out_group = p.add_argument_group("output")
    out_group.add_argument(
        "--output-json",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write the result list as JSON (for emit_perf_model.py input).",
    )
    out_group.add_argument("--name-max", type=int, default=60, help="Truncate op names in table display.")
    out_group.add_argument(
        "--tracerlens-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="TraceLens repo root for has_perf_model annotation (tries sys.path import).",
    )
    out_group.add_argument(
        "--no-model-only",
        action="store_true",
        help="Only show ops where has_perf_model == False (requires --tracerlens-root).",
    )

    args = p.parse_args(argv)

    all_ops: List[Dict] = []

    for fw in args.frameworks:
        fw_dir = getattr(args, fw, None) or _resolve_framework_root(args.repo_root, fw)
        if fw_dir is None or not fw_dir.is_dir():
            if getattr(args, fw, None):
                print(f"Warning: {fw} dir not found: {fw_dir}", file=sys.stderr)
            continue
        print(f"Scanning {fw} at {fw_dir} ...", file=sys.stderr)
        scanner = {"aiter": scan_aiter, "vllm": scan_vllm, "sglang": scan_sglang}[fw]
        found = scanner(fw_dir, filter_kw=args.filter)
        print(f"  -> {len(found)} ops", file=sys.stderr)
        all_ops.extend(found)

    if not all_ops:
        print("No ops found. Check --repo-root / --aiter / --vllm / --sglang paths.", file=sys.stderr)
        return 0

    # Trace cross-check
    if args.trace and args.trace.is_file():
        print(f"Cross-checking against trace: {args.trace}", file=sys.stderr)
        trace_ops = load_trace_ops(args.trace)
        cross_check_trace(all_ops, trace_ops)

    # TraceLens coverage annotation
    annotate_perf_model_coverage(all_ops, args.tracerlens_root)

    # Filter to no-model-only if requested
    if args.no_model_only:
        before = len(all_ops)
        all_ops = [op for op in all_ops if op.get("has_perf_model") is False]
        print(f"Filtered to no-model ops: {len(all_ops)} (was {before})", file=sys.stderr)

    # Display
    print_results(all_ops, args.name_max)

    # JSON output
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as fh:
            json.dump(all_ops, fh, indent=2)
        print(f"\nWrote {len(all_ops)} ops to {args.output_json.resolve()}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
