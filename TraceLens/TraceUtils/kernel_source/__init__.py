###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Map a GPU kernel to its editable source (or explain why it has none) via gate -> native/Triton resolve -> discovery."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .editable import is_editable_source
from .index import (
    FrameworkRoot,
    SourceIndex,
    build_index,
    discover_frameworks,
    discover_library_paths,
    load_or_build,
)
from .datatypes import Patchability, ResolveResult, SourceLocation
from .patchability import classify_patchability
from .resolver import resolve_kernel, resolve_source_path
from .triton_pin import resolve_triton_source, triton_def_line

__all__ = [
    # Data types
    "SourceLocation",
    "Patchability",
    "ResolveResult",
    # Gate
    "classify_patchability",
    # Native resolution
    "resolve_kernel",
    "resolve_source_path",
    # Triton resolution
    "resolve_triton_source",
    "triton_def_line",
    # One call, either kind
    "resolve_kernel_source",
    # Discovery + index
    "discover_library_paths",
    "discover_frameworks",
    "FrameworkRoot",
    "SourceIndex",
    "build_index",
    "load_or_build",
    # Editability
    "is_editable_source",
]


def resolve_kernel_source(
    kernel_name: str = "",
    *,
    kernel_file: str = "",
    is_triton: bool = False,
    op_name: str = "",
    search_paths: Sequence[str | Path] | None = None,
) -> ResolveResult:
    """Resolve one device kernel to its source, native or Triton, in one call.

    Routes on ``kernel_file`` (a trace only records one for Triton). Set
    ``is_triton`` to force that route on a trace that didn't capture one.
    With neither, tries native first; a plain miss there (not a gate
    rejection -- that's a real verdict on a real native kernel) falls back to
    a Triton symbol search, since the caller may simply not know the kind.

    Args:
        kernel_name (Essential): Device kernel symbol.
        kernel_file (Optional): The trace's Triton launcher string, when known.
        is_triton (Optional): Force the Triton path when ``kernel_file`` is empty.
        op_name (Optional): Launching op name, for the native gate (e.g. MIOpen).
        search_paths (Optional): Optional search roots; defaults to auto-discovery.

    Returns:
        A :class:`~.datatypes.ResolveResult`.
    """
    result = None
    if not (kernel_file or is_triton):
        result = resolve_kernel(kernel_name, op_name=op_name, search_paths=search_paths)
        if result.method != "unresolved":
            return result

    triton_result = resolve_triton_source(
        kernel_file, symbol=kernel_name, search_paths=search_paths
    )
    if kernel_file or is_triton or triton_result.patchable:
        return triton_result
    return result
