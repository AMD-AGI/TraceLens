#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
EP3 — Framework source scanner for unified-perf-model-extension.

Statically scans framework repos (aiter, vLLM, SGLang / sgl_kernel) for
Python-side torch op registrations that TraceLens can model. Optionally
cross-checks against a unified_perf_summary.csv to mark which ops appeared in
a real trace.

Usage
-----
# Scan all frameworks under a common parent (<DIR>/aiter, <DIR>/vllm, <DIR>/sglang):
  python3 scan_framework_ops.py --repo-root /work

# Scan one framework with a filter, and cross-check against a trace:
  python3 scan_framework_ops.py --aiter /work/aiter --filter attention \\
      --trace unified_perf_summary.csv

# Write the result table to JSON:
  python3 scan_framework_ops.py --repo-root /work --output-json /tmp/ops.json

Registrations found
-------------------
  @compile_ops("ns::op")
  torch.library.custom_op / define, direct_register_custom_op, register_custom_op
      ("ns::op" or "op", positional or op_name=)
Op names without a namespace get the framework's default namespace. Only the
framework's own namespaces are kept.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Same directory; loads the summary CSV the same way as the triage script.
from run_other_bucket_triage import builtin_perf_model_map, load_rows

# framework -> (namespace for bare op names, namespaces to keep)
FRAMEWORKS = {
    "aiter": ("aiter", {"aiter"}),
    "vllm": ("vllm", {"vllm"}),
    "sglang": ("sgl_kernel", {"sgl_kernel", "sglang_profiler"}),
}
_REGISTRATION_PATTERNS = (
    # @compile_ops's first argument is usually a module name, so require "ns::".
    re.compile(r"@\s*compile_ops\s*\(\s*[\"'](\w+)::(\w+)"),
    re.compile(
        r"(?:custom_op|define)\s*\(\s*(?:op_name\s*=\s*)?[\"'](?:(\w+)::)?(\w+)"
    ),
)
# Kernel-looking calls near a registration, and torch.ops calls.
_KERNEL_CALL = re.compile(
    r"\b(\w+(?:_kernel|_fwd|_bwd|_hip|_triton|_cuda|_rocm|_gemm|_attn)|torch\.ops\.[\w.]+)\s*\("
)
# Sibling test / benchmark files, surfaced as roofline references.
_BENCH_FILE = re.compile(r"^(?:test_|benchmark_|bench_)", re.IGNORECASE)
_SKIP_DIRS = {
    ".git",
    "__pycache__",
    "build",
    "dist",
    ".tox",
    "node_modules",
    ".venv",
    "venv",
}


def iter_py_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        yield from (Path(dirpath) / f for f in filenames if f.endswith(".py"))


def _extend_unique(dst: List[str], items: Iterable[str]) -> None:
    dst.extend(i for i in dict.fromkeys(items) if i not in dst)


def _kernel_calls(lines: List[str], op_base: str) -> List[str]:
    """Kernel-looking calls within 60 lines after each line mentioning op_base."""
    hits = (
        m.group(1)
        for i, line in enumerate(lines)
        if op_base in line
        for window_line in lines[i : i + 60]
        for m in _KERNEL_CALL.finditer(window_line)
    )
    return list(dict.fromkeys(hits))[:8]


def scan(framework: str, root: Path, filter_kw: Optional[str] = None) -> List[Dict]:
    """Op registrations under root; one entry per op name."""
    default_ns, namespaces = FRAMEWORKS[framework]
    kw = (filter_kw or "").lower()
    ops: Dict[str, Dict] = {}
    for py_file in iter_py_files(root):
        try:
            source = py_file.read_text(errors="replace")
        except OSError:
            continue
        in_source = kw in source.lower()
        lines = source.splitlines()
        benches = None
        for pat in _REGISTRATION_PATTERNS:
            for m in pat.finditer(source):
                ns, base = m.group(1) or default_ns, m.group(2)
                name = f"{ns}::{base}"
                if ns not in namespaces or (
                    kw and kw not in name.lower() and not in_source
                ):
                    continue
                if benches is None:
                    benches = sorted(
                        f.name
                        for f in py_file.parent.iterdir()
                        if f.is_file() and _BENCH_FILE.match(f.name)
                    )
                op = ops.setdefault(
                    name,
                    {
                        "framework": framework,
                        "op_name": name,
                        "file": str(py_file),
                        "kernel_hints": [],
                        "sibling_benchmarks": [],
                        "in_trace": False,
                        "trace_pct": None,
                    },
                )
                _extend_unique(op["kernel_hints"], _kernel_calls(lines, base))
                _extend_unique(op["sibling_benchmarks"], benches)
    return list(ops.values())


def cross_check_trace(ops: List[Dict], trace_csv: Path) -> None:
    """Set in_trace / trace_pct on each op from the summary CSV's runtime per name."""
    rows, _ = load_rows(trace_csv)
    trace_us: Dict[str, float] = {}
    for r in rows:
        name = (r.get("name") or "").strip()
        trace_us[name] = trace_us.get(name, 0.0) + r["_us"]
    total = sum(trace_us.values())
    for op in ops:
        us = trace_us.get(op["op_name"], trace_us.get(op["op_name"].split("::", 1)[-1]))
        op["in_trace"] = us is not None
        op["trace_pct"] = 100.0 * us / total if us is not None and total > 0 else 0.0


def print_results(ops: List[Dict], name_max: int = 60) -> None:
    by_fw: Dict[str, List[Dict]] = {}
    for op in ops:
        by_fw.setdefault(op["framework"], []).append(op)
    for fw, fw_ops in sorted(by_fw.items()):
        print(f"\n{'=' * 70}\n  Framework: {fw}  ({len(fw_ops)} ops)\n{'=' * 70}")
        for op in fw_ops:
            name = op["op_name"]
            hpm = op.get("has_perf_model")
            model_tag = (
                "" if hpm is None else (" [HAS MODEL]" if hpm else " [NO MODEL]")
            )
            trace_tag = f"  trace:{op['trace_pct']:.1f}%" if op["in_trace"] else ""
            print(
                f"  {name if len(name) <= name_max else name[: name_max - 3] + '...'}{model_tag}{trace_tag}"
            )
            print(f"    file: {op['file']}")
            if op["kernel_hints"]:
                print(f"    kernels: {', '.join(op['kernel_hints'][:5])}")
            if op["sibling_benchmarks"]:
                print(f"    roofline refs: {', '.join(op['sibling_benchmarks'][:4])}")
    print(f"\nTotal ops found: {len(ops)}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        metavar="DIR",
        help="Parent of aiter/, vllm/, sglang/; used for frameworks without their own flag.",
    )
    for fw in FRAMEWORKS:
        p.add_argument(
            f"--{fw}", type=Path, metavar="DIR", help=f"Path to the {fw} repo root"
        )
    p.add_argument(
        "--filter",
        metavar="KEYWORD",
        help="Case-insensitive keyword matched against op names and source files (e.g. attention).",
    )
    p.add_argument(
        "--frameworks", nargs="+", choices=list(FRAMEWORKS), default=list(FRAMEWORKS)
    )
    p.add_argument(
        "--trace",
        type=Path,
        metavar="CSV",
        help="unified_perf_summary.csv to cross-check against.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        metavar="FILE",
        help="Write the result list as JSON.",
    )
    p.add_argument(
        "--name-max", type=int, default=60, help="Truncate op names in the table."
    )
    p.add_argument(
        "--check-mapping",
        action="store_true",
        help="Mark ops with or without a built-in TraceLens perf model (needs TraceLens importable).",
    )
    p.add_argument(
        "--no-model-only",
        action="store_true",
        help="Only show ops without a built-in perf model (implies --check-mapping).",
    )
    args = p.parse_args(argv)

    ops: List[Dict] = []
    for fw in args.frameworks:
        fw_dir = getattr(args, fw) or (args.repo_root / fw if args.repo_root else None)
        if fw_dir is None or not fw_dir.is_dir():
            if getattr(args, fw):
                print(f"Warning: {fw} dir not found: {fw_dir}", file=sys.stderr)
            continue
        print(f"Scanning {fw} at {fw_dir} ...", file=sys.stderr)
        found = scan(fw, fw_dir, args.filter)
        print(f"  -> {len(found)} ops", file=sys.stderr)
        ops.extend(found)
    if not ops:
        print(
            "No ops found. Check --repo-root / --aiter / --vllm / --sglang paths.",
            file=sys.stderr,
        )
        return 0

    if args.trace and args.trace.is_file():
        print(f"Cross-checking against trace: {args.trace}", file=sys.stderr)
        cross_check_trace(ops, args.trace)

    mapping = (
        builtin_perf_model_map() if args.check_mapping or args.no_model_only else None
    )
    for op in ops:
        bare = op["op_name"].split("::", 1)[-1]
        op["has_perf_model"] = (
            None if mapping is None else (op["op_name"] in mapping or bare in mapping)
        )
    if args.no_model_only:
        ops = [op for op in ops if op["has_perf_model"] is False]

    print_results(ops, args.name_max)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(ops, indent=2))
        print(
            f"\nWrote {len(ops)} ops to {args.output_json.resolve()}", file=sys.stderr
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
