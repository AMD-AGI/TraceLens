###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Validate the kernel-source resolver against a real vLLM/SGLang install.

Run this inside a framework container (where vllm / aiter / sglang are
installed). It exercises the resolver two ways:

1. **Candidates check** -- reads a ``kernel_candidates.json`` and, for each hot
   kernel, runs our resolver and compares the verdict to the ground truth the
   file already carries (``source_file`` and the old ``op_to_source_*`` fields).

2. **Trace scan** (optional) -- pulls the unique GPU kernel names out of a
   ``.pt.trace.json[.gz]`` and resolves each, reporting how many hit / were
   gated as non-patchable / missed.

Nothing here is GPU-dependent: it only reads installed source files.

Examples:
    python tools/validate_kernel_source.py --candidates kernel_candidates.json
    python tools/validate_kernel_source.py --candidates kc.json --trace rank0.pt.trace.json.gz
    python tools/validate_kernel_source.py --candidates kc.json \\
        --search-path /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc
"""

from __future__ import annotations

import argparse
import gzip
import json

from TraceLens.TraceUtils.kernel_source import (
    discover_library_paths,
    resolve,
    resolve_triton_source,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_json(path: str) -> dict:
    """Load a JSON file that may be gzip-compressed."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _resolved_search_paths(cli_paths: list[str]) -> list[str]:
    """Use the paths the caller gave, else auto-discover them."""
    if cli_paths:
        return cli_paths
    discovered = [str(p) for p in discover_library_paths()]
    return discovered


def _short(text: str, n: int = 60) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 1] + "\u2026"


# ---------------------------------------------------------------------------
# Part 1: candidates check
# ---------------------------------------------------------------------------
def check_candidates(candidates_path: str, search_paths: list[str]) -> None:
    """Resolve every hot kernel and compare to the file's own ground truth."""
    doc = _load_json(candidates_path)
    kernels = doc.get("hot_kernels") or doc.get("kernels") or []
    print(f"\n=== Candidates check: {len(kernels)} hot kernels ===\n")

    counts = {
        "native_hit": 0,
        "triton_hit": 0,
        "gated": 0,
        "unresolved": 0,
        "skipped": 0,
    }
    disagreements: list[str] = []

    for k in kernels:
        kid = k.get("kernel_id", "?")
        op_name = k.get("name", "")
        dev = k.get("device_kernel_name") or ""
        launcher = k.get("launcher_source_file") or ""
        gt_source = k.get("source_file") or ""
        gt_patchable = k.get("op_to_source_patchable")

        # Pick the path: native symbol first, else a Triton .py launcher.
        if dev:
            res = resolve(dev, search_paths, op_name=op_name)
        elif launcher.endswith(".py"):
            res = resolve_triton_source(launcher, symbol=op_name)
        else:
            counts["skipped"] += 1
            print(f"[{kid}] SKIP  {op_name}  (no device name / launcher)")
            continue

        # Tally.
        if res.patchable and res.method in ("triton_ast", "trace_kernel_file"):
            counts["triton_hit"] += 1
        elif res.patchable:
            counts["native_hit"] += 1
        elif res.method == "gate_non_patchable":
            counts["gated"] += 1
        else:
            counts["unresolved"] += 1

        verdict = f"patchable={res.patchable} method={res.method}" + (
            f" kind={res.kind}" if res.kind else ""
        )
        print(f"[{kid}] {op_name}")
        print(f"      dev   : {_short(dev, 70)}")
        print(f"      ours  : {verdict}")
        print(f"      ->file: {_short(res.source_file, 80)}")
        print(f"      truth : patchable={gt_patchable} file={_short(gt_source, 80)}")

        # Flag a clear disagreement: we say patchable but the ground truth says
        # not (or vice versa). ``None`` ground truth = old map had no opinion.
        if gt_patchable is not None and bool(res.patchable) != bool(gt_patchable):
            disagreements.append(
                f"{kid} {op_name}: ours={res.patchable} truth={gt_patchable}"
            )

    print("\n--- summary ---")
    for key, val in counts.items():
        print(f"  {key:12}: {val}")
    if disagreements:
        print(f"\n  DISAGREEMENTS vs old map ({len(disagreements)}):")
        for d in disagreements:
            print(f"    - {d}")
    else:
        print("\n  No patchable/non-patchable disagreements vs the old map.")


# ---------------------------------------------------------------------------
# Part 2: trace scan
# ---------------------------------------------------------------------------
def _kernel_names_from_trace(trace_path: str) -> list[str]:
    """Collect the unique GPU kernel names from a Kineto trace."""
    doc = _load_json(trace_path)
    events = doc.get("traceEvents", [])
    names: dict[str, None] = {}
    for ev in events:
        if str(ev.get("cat", "")).lower() == "kernel":
            name = ev.get("name")
            if name:
                names.setdefault(name, None)
    return list(names)


def scan_trace(trace_path: str, search_paths: list[str], limit: int) -> None:
    """Resolve every unique kernel name in a trace; report hit/gated/miss."""
    names = _kernel_names_from_trace(trace_path)
    if limit:
        names = names[:limit]
    print(f"\n=== Trace scan: {len(names)} unique GPU kernel names ===\n")

    counts = {"hit": 0, "gated": 0, "miss": 0}
    for name in names:
        res = resolve(name, search_paths)
        if res.patchable:
            counts["hit"] += 1
            print(f"  HIT   {_short(name, 55):55}  -> {_short(res.source_file, 70)}")
        elif res.method == "gate_non_patchable":
            counts["gated"] += 1
            print(f"  GATE  {_short(name, 55):55}  ({res.kind})")
        else:
            counts["miss"] += 1
            print(f"  MISS  {_short(name, 55):55}")

    print("\n--- summary ---")
    for key, val in counts.items():
        print(f"  {key:6}: {val}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the kernel-source resolver on a real install."
    )
    parser.add_argument(
        "--candidates", default="", help="Path to kernel_candidates.json."
    )
    parser.add_argument(
        "--trace", default="", help="Path to a .pt.trace.json[.gz] to scan (optional)."
    )
    parser.add_argument(
        "--search-path",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory to search (repeatable). Auto-discovered when omitted.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Cap trace kernel names scanned (0 = all)."
    )
    args = parser.parse_args(argv)

    if not args.candidates and not args.trace:
        parser.error("give at least one of --candidates or --trace")

    search_paths = _resolved_search_paths(args.search_path)
    print("Search paths:")
    for p in search_paths:
        print(f"  - {p}")
    if not search_paths:
        print("  (none found -- native resolution will miss everything)")

    if args.candidates:
        check_candidates(args.candidates, search_paths)
    if args.trace:
        scan_trace(args.trace, search_paths, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
