###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Resolve a native GPU kernel to its editable source (the "active finder").

This is the core layer. Instead of trusting an absolute path captured once at
build time, it locates a kernel's source in the *currently installed* framework
tree by the kernel's stable identity:

1. demangle the device symbol to its base name (:mod:`.demangle`);
2. look the base name up in the live ``__global__`` index over the caller's
   search paths (:mod:`.index`);
3. rank the candidate definitions, verify the symbol really appears in the file,
   and return the first editable, verified location.

Two public entry points:

* :func:`resolve_source_path` -- the simple "name + paths -> location" answer.
* :func:`resolve` -- runs the cheap :mod:`.patchability` gate first, then
  resolves, returning a richer :class:`~.datatypes.ResolveResult`.

If ``search_paths`` is omitted/empty, both fall back to
:func:`discovery.discover_library_paths` so a caller can use the resolver with
zero configuration.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

from . import discovery, index
from .demangle import base_symbol
from .editable import is_editable_source
from .datatypes import ResolveResult, SourceLocation
from .patchability import classify_patchability

log = logging.getLogger(__name__)

__all__ = ["resolve_source_path", "resolve"]

# Framework labels inferred from a resolved path, for SourceLocation.framework.
_FRAMEWORK_HINTS = ("aiter", "sglang", "vllm")


def _framework_of(path: str) -> str:
    """Best-effort framework label for a resolved path (``""`` if unknown)."""
    low = path.lower()
    for name in _FRAMEWORK_HINTS:
        if name in low:
            return name
    return ""


def _rank_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Rank candidate definition records: arch-tag match, then shorter path.

    When a base name maps to more than one definition, prefer a file whose path
    matches ``$TRACELENS_TARGET_ARCH`` (e.g. ``gfx942``) and then the shorter
    path (the canonical location over a vendored copy).
    """
    arch = os.environ.get("TRACELENS_TARGET_ARCH", "").strip().lower()

    def score(rec: dict[str, object]) -> tuple[int, int]:
        path = str(rec.get("file", "")).lower()
        arch_match = 1 if arch and arch in path else 0
        return (arch_match, -len(path))

    return sorted(records, key=score, reverse=True)


def _verify_symbol(path: str, base: str) -> bool:
    """Confirm ``base`` actually appears in ``path`` (guards a stale index)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        log.debug("active-finder: cannot read %r for symbol verify: %s", path, exc)
        return False
    return base in text


def _resolved_paths(search_paths: Sequence[str | Path] | None) -> list[Path]:
    """Return the caller's search paths, or discover defaults when omitted."""
    if search_paths:
        return [Path(p) for p in search_paths]
    return discovery.discover_library_paths()


def resolve_source_path(
    kernel_name: str,
    search_paths: Sequence[str | Path] | None = None,
    *,
    index_obj: index.SourceIndex | None = None,
) -> SourceLocation | None:
    """Resolve a native kernel name to its editable source location.

    Args:
        kernel_name: Device kernel symbol from the trace (mangled or plain).
        search_paths: Directories to search. When omitted/empty, defaults are
            discovered via :func:`discovery.discover_library_paths`.
        index_obj: Optional prebuilt index; built/cached from ``search_paths``
            when omitted.

    Returns:
        A :class:`~.datatypes.SourceLocation` for the first editable, verified
        definition, or ``None`` when nothing matches.
    """
    base = base_symbol(kernel_name)
    if not base:
        return None

    idx = (
        index_obj
        if index_obj is not None
        else index.load_or_build(_resolved_paths(search_paths))
    )
    records = _rank_records(idx.lookup(base))
    if len(records) > 1:
        # A bare base name mapping to >1 definition is disambiguated only by the
        # ranking heuristic, so leave a breadcrumb for anyone auditing a rewrite.
        log.debug(
            "active-finder: %d candidate records for base %r; ranking by heuristic",
            len(records),
            base,
        )

    for rec in records:
        path = str(rec.get("file", ""))
        if not is_editable_source(path):
            continue
        if not _verify_symbol(path, base):
            continue
        line_val = rec.get("line")
        line_no = line_val if isinstance(line_val, int) and line_val > 0 else None
        return SourceLocation(
            source_file=path, line=line_no, framework=_framework_of(path)
        )

    log.debug("active-finder: no editable/verified source for base %r", base)
    return None


def resolve(
    kernel_name: str,
    search_paths: Sequence[str | Path] | None = None,
    *,
    op_name: str = "",
    call_stack: Sequence[str] = (),
    run_gate: bool = True,
    index_obj: index.SourceIndex | None = None,
) -> ResolveResult:
    """Gate, then resolve a native kernel to its editable source.

    Runs the cheap :func:`~.patchability.classify_patchability` gate first and
    short-circuits non-patchable kernels (no filesystem work), then resolves
    survivors via :func:`resolve_source_path`.

    Args:
        kernel_name: Device kernel symbol from the trace.
        search_paths: Directories to search (discovered defaults when omitted).
        op_name: Launching op name, used by the gate (e.g. MIOpen detection).
        call_stack: Optional call-stack frames, used by the gate (inductor Triton).
        run_gate: Set ``False`` to skip the gate and resolve directly.
        index_obj: Optional prebuilt index.

    Returns:
        A :class:`~.datatypes.ResolveResult` with the outcome and method.
    """
    if run_gate:
        gate = classify_patchability(
            kernel_name, op_name=op_name, call_stack=call_stack
        )
        if gate.patchable is False:
            return ResolveResult(
                location=None,
                patchable=False,
                kind=gate.kind,
                reason=gate.reason,
                method="gate_non_patchable",
            )

    location = resolve_source_path(kernel_name, search_paths, index_obj=index_obj)
    if location is not None:
        return ResolveResult(location=location, patchable=True, method="symbol_index")

    return ResolveResult(
        location=None, patchable=False, method="unresolved", reason="no live match"
    )
