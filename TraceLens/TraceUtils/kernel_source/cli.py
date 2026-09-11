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

    # Triton kernel from a trace kernel_file:
    TraceLens_resolve_kernel_source --triton-kernel-file "/repo/moe.py:120:kernel"
"""

from __future__ import annotations

import argparse
import json
import sys

from .datatypes import ResolveResult
from .patchability import classify_patchability
from .resolver import resolve_source_path
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

    gate = classify_patchability(args.kernel, op_name=args.op_name)
    if gate.patchable is False:
        result = ResolveResult(
            location=None,
            patchable=False,
            kind=gate.kind,
            reason=gate.reason,
            method="gate_non_patchable",
        )
        print(json.dumps(_result_to_dict(result), indent=2))
        return 0

    location = resolve_source_path(args.kernel, args.search_path or None)
    if location is not None:
        result = ResolveResult(location=location, patchable=True, method="symbol_index")
    else:
        result = ResolveResult(
            location=None,
            patchable=False,
            method="unresolved",
            reason="no live match",
        )
    print(json.dumps(_result_to_dict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
