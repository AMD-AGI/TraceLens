###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Versioned, on-disk audit view of kernel source resolution.

This is an optional reporting layer. It defines a small, versioned JSON artifact
that records, per hot kernel, "which file did we decide this kernel lives in, by
what method, and how sure are we". It is deliberately an *audit view*, not a
pipeline contract: consumers that need to act on resolution should use the
:class:`~.datatypes.ResolveResult` returned by the resolver directly.

The helpers here build entries (:func:`make_entry`), wrap them in a versioned
envelope (:func:`make_document`), validate a document against the schema
(:func:`validate_document`), and round-trip it to disk (:func:`read_document`).
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

#: Bump the major on any field removal or meaning change; consumers gate on it.
SOURCE_RESOLUTION_SCHEMA_VERSION = "1.0.0"

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

KNOWN_METHODS = frozenset(
    {
        METHOD_SYMBOL_INDEX,
        METHOD_TRITON_AST,
        METHOD_TRACE_KERNEL_FILE,
        METHOD_GATE_NON_PATCHABLE,
        METHOD_TRACE,
        METHOD_GREP,
        METHOD_LLM_FALLBACK,
        METHOD_LLM,
        METHOD_UNRESOLVED,
    }
)

# Methods whose entries legitimately carry no source_file.
_NO_SOURCE_METHODS = frozenset({METHOD_UNRESOLVED, METHOD_GATE_NON_PATCHABLE})

#: Every entry carries these keys, so a consumer can rely on presence without
#: defaulting. Values may be empty; the keys may not be absent.
REQUIRED_ENTRY_KEYS = (
    "kernel_id",
    "name",
    "gpu_pct",
    "source_file",
    "method",
    "reason",
)

REQUIRED_DOCUMENT_KEYS = ("schema_version", "generated_by", "entries")


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
        method: One of :data:`KNOWN_METHODS`.
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
    """Wrap ``entries`` in the versioned envelope."""
    return {
        "schema_version": SOURCE_RESOLUTION_SCHEMA_VERSION,
        "generated_by": str(generated_by or ""),
        "model_name": str(model_name or ""),
        "framework": str(framework or ""),
        "entries": list(entries),
    }


def validate_document(doc: Any) -> list[str]:
    """Return a list of contract violations; empty means the document is valid.

    Reports every problem rather than raising on the first, so a producer test
    failure names all of them at once.
    """
    problems: list[str] = []
    if not isinstance(doc, dict):
        return [f"document is {type(doc).__name__}, expected dict"]
    for key in REQUIRED_DOCUMENT_KEYS:
        if key not in doc:
            problems.append(f"document missing required key {key!r}")
    version = str(doc.get("schema_version") or "")
    if (
        version
        and version.split(".")[0] != SOURCE_RESOLUTION_SCHEMA_VERSION.split(".")[0]
    ):
        problems.append(
            f"schema_version {version!r} has a different major than "
            f"{SOURCE_RESOLUTION_SCHEMA_VERSION!r}"
        )
    entries = doc.get("entries")
    if not isinstance(entries, list):
        problems.append(f"entries is {type(entries).__name__}, expected list")
        return problems
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"entries[{i}] is {type(entry).__name__}, expected dict")
            continue
        for key in REQUIRED_ENTRY_KEYS:
            if key not in entry:
                problems.append(f"entries[{i}] missing required key {key!r}")
        method = str(entry.get("method") or "")
        if method and method not in KNOWN_METHODS:
            problems.append(f"entries[{i}] has unknown method {method!r}")
        src = str(entry.get("source_file") or "")
        if src and method in _NO_SOURCE_METHODS:
            problems.append(f"entries[{i}] has a source_file but method is {method}")
        if not src and method and method not in _NO_SOURCE_METHODS:
            problems.append(f"entries[{i}] has method {method!r} but no source_file")
        confidence = entry.get("confidence")
        if confidence is not None:
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                problems.append(
                    f"entries[{i}] has invalid confidence {confidence!r}; "
                    "expected a finite number in [0, 1]"
                )
    return problems


#: A call site is often reported as ``path.py(247): fn_name``; the line and
#: function ride along in the same string.
_LINE_SUFFIX_RE = re.compile(
    r"^(?P<path>.+?)\((?P<line>\d+)\)\s*(?::\s*(?P<function>.*))?$"
)


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


def strip_line_suffix(path: str) -> str:
    """Return the bare file path from a possibly line-annotated one.

    ``/repo/moe.py(247): _grouped_gemm`` -> ``/repo/moe.py``.
    """
    return split_line_suffix(path)[0]


def canonical_source_path(path: str, roots: tuple[str, ...]) -> str:
    """Return the validated canonical target for ``path``, or ``""``.

    The file must exist on this host **and** sit under one of ``roots``.
    Requiring existence guards against a fabricated but plausible-looking path
    passing a mere prefix check.
    """
    text = strip_line_suffix(path)
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
    """Load the artifact, or ``None`` when it is absent or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
