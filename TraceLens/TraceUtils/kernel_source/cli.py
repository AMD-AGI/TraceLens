###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Command-line entry point for the kernel-source resolver.

Resolves a single kernel and prints the outcome as JSON, so the package can be
exercised without writing Python. Examples::

    # Native kernel, explicit search paths:
    TraceLens_resolve_kernel_source --kernel _Z12my_kernelPf \\
        --search-path /opt/vllm/csrc --search-path /opt/aiter/csrc

    # Just the cheap gate verdict:
    TraceLens_resolve_kernel_source --kernel Cijk_Ailk_Bljk --gate-only

    # Triton kernel from a trace kernel_file:
    TraceLens_resolve_kernel_source --triton-kernel-file "/repo/moe.py:120:kernel"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .datatypes import ResolveResult
from .patchability import classify_patchability
from .resolver import resolve
from .triton_pin import resolve_triton_source


def _result_to_dict(result: ResolveResult) -> dict:
    """Flatten a :class:`ResolveResult` (with its nested location) to JSON."""
    return {
        "source_file": result.source_file,
        "line": result.line,
        "framework": result.location.framework if result.location else "",
        "patchable": result.patchable,
        "kind": result.kind,
        "reason": result.reason,
        "method": result.method,
    }


def _read_call_stack(path: str | None) -> list[str]:
    """Read a call-stack file (one frame per line), or return ``[]``."""
    if not path:
        return []
    try:
        return [
            ln.strip()
            for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except OSError as exc:
        print(
            f"warning: could not read call-stack file {path!r}: {exc}", file=sys.stderr
        )
        return []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="TraceLens_resolve_kernel_source",
        description="Resolve a GPU kernel to its editable source (or explain why it has none).",
    )
    parser.add_argument(
        "--kernel", default="", help="Device kernel name/symbol (native or plain)."
    )
    parser.add_argument(
        "--search-path",
        action="append",
        default=[],
        metavar="DIR",
        help="A directory to search for native sources (repeatable). "
        "When omitted, defaults are auto-discovered.",
    )
    parser.add_argument(
        "--op-name",
        default="",
        help="Launching op name (used by the gate, e.g. MIOpen).",
    )
    parser.add_argument(
        "--call-stack-file", default="", help="File with one call-stack frame per line."
    )
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="Print only the patchability gate verdict.",
    )
    parser.add_argument(
        "--triton-kernel-file",
        default="",
        help="Resolve a Triton .py kernel from this trace kernel_file instead of a native symbol.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Resolve one kernel per the CLI args and print the JSON result."""
    args = _build_parser().parse_args(argv)

    if args.triton_kernel_file:
        result = resolve_triton_source(args.triton_kernel_file, symbol=args.kernel)
        print(json.dumps(_result_to_dict(result), indent=2))
        return 0

    if not args.kernel:
        print("error: --kernel (or --triton-kernel-file) is required", file=sys.stderr)
        return 2

    call_stack = _read_call_stack(args.call_stack_file)

    if args.gate_only:
        verdict = classify_patchability(
            args.kernel, op_name=args.op_name, call_stack=call_stack
        )
        print(json.dumps(dataclasses.asdict(verdict), indent=2))
        return 0

    result = resolve(
        args.kernel,
        args.search_path or None,
        op_name=args.op_name,
        call_stack=call_stack,
    )
    print(json.dumps(_result_to_dict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
