###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Resolve a native GPU kernel to its editable source by demangling -> index lookup -> rank/verify (the "active finder").

Callers run :func:`~.patchability.classify_patchability` first and skip the
lookup when the verdict is non-patchable; see ``cli.main`` for the reference
sequence.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

from . import index
from .demangle import base_symbol
from .editable import is_editable_source
from .datatypes import SourceLocation

log = logging.getLogger(__name__)

__all__ = ["resolve_source_path"]

# Framework labels inferred from a resolved path, for SourceLocation.framework.
# Kept in sync with index._KNOWN (the frameworks we locate by name).
_FRAMEWORK_HINTS = ("aiter", "sglang", "vllm", "atom")


def _framework_of(path: str) -> str:
    """Best-effort framework label for a resolved path (``""`` if unknown)."""
    low = path.lower()
    for name in _FRAMEWORK_HINTS:
        if name in low:
            return name
    return ""


def _rank_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Rank candidate definition records: ``$TRACELENS_TARGET_ARCH`` match first, then shorter path."""
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
    return index.discover_library_paths()


def resolve_source_path(
    kernel_name: str,
    search_paths: Sequence[str | Path] | None = None,
    *,
    index_obj: index.SourceIndex | None = None,
) -> SourceLocation | None:
    """Resolve a native kernel name to its editable source location, or ``None`` if unmatched."""
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
        # Multiple defs are disambiguated only by the ranking heuristic; leave a breadcrumb.
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


