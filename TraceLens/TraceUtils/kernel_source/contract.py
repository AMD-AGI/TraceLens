###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""On-disk audit view of kernel source resolution.

This is an optional reporting layer. It defines a small JSON artifact that
records, per hot kernel, "which file did we decide this kernel lives in, by
what method, and how sure are we". It is deliberately an *audit view*, not a
pipeline contract: consumers that need to act on resolution should use the
:class:`~.datatypes.ResolveResult` returned by the resolver directly.

The helpers here build entries (:func:`make_entry`), wrap them in a document
that carries the producing TraceLens version (:func:`make_document`), and read
one back from disk (:func:`read_document`). Every entry is built with all keys
present (blank when unknown), so no separate runtime validation step is needed;
a light schema check lives in the tests to guard against accidental drift.
"""

from __future__ import annotations

import os
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _tracelens_version() -> str:
    """Return the installed TraceLens version (which embeds the git SHA), or ``""``.

    This is the single version source for the repo: the wheel version built in
    ``setup.py`` looks like ``0.1.0.dev<date>+g<shortsha>``, so stamping it on
    the document records exactly which TraceLens produced the artifact.
    """
    try:
        return version("TraceLens")
    except PackageNotFoundError:  # not installed as a distribution (e.g. source tree)
        return ""


#: A call site is often reported as ``path.py(247): fn_name``; the line and
#: function ride along in the same string.
_LINE_SUFFIX_RE = re.compile(
    r"^(?P<path>.+?)\((?P<line>\d+)\)\s*(?::\s*(?P<function>.*))?$"
)

#: Canonical artifact name, relative to the analysis run directory.
SOURCE_RESOLUTION_FILENAME = "kernel_source_resolution.json"

#: How a location was decided. ``symbol_index`` is the native active-finder hit;
#: ``triton_ast`` / ``trace_kernel_file`` come from the Triton resolver; the
#: remaining methods describe downstream/fallback tiers a caller may layer on.
METHOD_SYMBOL_INDEX = "symbol_index"
METHOD_TRITON_AST = "triton_ast"
METHOD_TRACE_KERNEL_FILE = "trace_kernel_file"
METHOD_GATE_NON_PATCHABLE = "gate_non_patchable"
METHOD_TRACE = "trace_python_stack"
METHOD_GREP = "name_grep"
METHOD_LLM_FALLBACK = "llm_fallback"
METHOD_LLM = "llm_review"
METHOD_UNRESOLVED = "unresolved"


def make_entry(
    *,
    kernel_id: str,
    name: str,
    gpu_pct: float,
    source_file: str = "",
    source_line: int | None = None,
    source_function: str = "",
    method: str = METHOD_UNRESOLVED,
    confidence: float | None = None,
    reason: str = "",
    previous_source_file: str = "",
    previous_method: str = "",
) -> dict[str, Any]:
    """Build one resolution entry with every required key present.

    Args:
        kernel_id: Stable-within-run kernel id (e.g. ``k001``).
        name: Kernel symbol as the profiler reported it.
        gpu_pct: Share of GPU time, used to rank what is worth resolving.
        source_file: Resolved path, or ``""`` when unresolved/non-patchable.
        source_line: 1-based line when the method produced one.
        source_function: Enclosing function when the method produced one.
        method: One of the ``METHOD_*`` labels defined in this module.
        confidence: 0..1 when a method reports one; ``None`` for deterministic
            methods, which are either right or silent.
        reason: Human-readable note -- why this path, or why none.
        previous_source_file: Location replaced by a later review, if any.
        previous_method: Method replaced by a later review, if any.

    Returns:
        The entry dict.
    """
    entry = {
        "kernel_id": str(kernel_id or ""),
        "name": str(name or ""),
        "gpu_pct": float(gpu_pct or 0.0),
        "source_file": str(source_file or ""),
        "source_line": source_line,
        "source_function": str(source_function or ""),
        "method": str(method or METHOD_UNRESOLVED),
        "confidence": confidence,
        "reason": str(reason or ""),
    }
    if previous_source_file:
        entry["previous_source_file"] = str(previous_source_file)
        entry["previous_method"] = str(previous_method or "")
    return entry


def make_document(
    entries: list[dict[str, Any]],
    *,
    generated_by: str,
    model_name: str = "",
    framework: str = "",
) -> dict[str, Any]:
    """Wrap ``entries`` in a document stamped with the producing TraceLens version."""
    return {
        "tracelens_version": _tracelens_version(),
        "generated_by": str(generated_by or ""),
        "model_name": str(model_name or ""),
        "framework": str(framework or ""),
        "entries": list(entries),
    }


def split_line_suffix(path: str) -> tuple[str, int | None, str]:
    """Split a call-site string into ``(path, line, function)``."""
    text = (path or "").strip()
    if not text:
        return "", None, ""
    match = _LINE_SUFFIX_RE.match(text)
    if match is None:
        return text, None, ""
    return (
        match.group("path").strip(),
        int(match.group("line")),
        str(match.group("function") or "").strip(),
    )


def canonical_source_path(path: str, roots: tuple[str, ...]) -> str:
    """Return the validated canonical target for ``path``, or ``""``.

    The file must exist on this host **and** sit under one of ``roots``.
    Requiring existence guards against a fabricated but plausible-looking path
    passing a mere prefix check.
    """
    text = split_line_suffix(path)[0]
    if not text or not os.path.isfile(text):
        return ""
    real = os.path.realpath(text)
    for root in roots:
        if not root:
            continue
        resolved_root = os.path.realpath(str(root)).rstrip(os.sep)
        if real == resolved_root or real.startswith(resolved_root + os.sep):
            return real
    return ""


def path_is_acceptable(path: str, roots: tuple[str, ...]) -> bool:
    """Whether a rewriting tier may write ``path`` as a resolved location."""
    return bool(canonical_source_path(path, roots))


def read_document(path: Path | str) -> dict[str, Any] | None:
    """Load the artifact, or ``None`` when it is absent or unreadable.

    Uses TraceLens's shared :class:`~TraceLens.util.DataLoader` so this artifact
    is read the same way (and with the same fast JSON parser) as every other
    TraceLens file, rather than a separate one-off reader here.
    """
    from TraceLens.util import DataLoader

    try:
        data = DataLoader.load_data(str(path))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None
