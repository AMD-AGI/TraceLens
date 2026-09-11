###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Test-only schema check for the kernel-source audit document.

The production module (``contract.py``) no longer validates at runtime: every
entry is built complete-by-construction (all keys present, blank when unknown),
so there is nothing to check while producing. This trimmed checker lives in the
tests only, to guard against accidental schema drift if someone later changes
the entry/document shape.
"""

from __future__ import annotations

import math
from typing import Any

from TraceLens.TraceUtils.kernel_source import contract

#: Keys every entry must carry (values may be blank; the keys may not be absent).
REQUIRED_ENTRY_KEYS = (
    "kernel_id",
    "name",
    "gpu_pct",
    "source_file",
    "method",
    "reason",
)

#: Keys every document must carry.
REQUIRED_DOCUMENT_KEYS = ("generated_by", "entries")

#: Valid ``method`` labels, taken straight from the module's ``METHOD_*`` set.
KNOWN_METHODS = frozenset(
    {
        contract.METHOD_SYMBOL_INDEX,
        contract.METHOD_TRITON_AST,
        contract.METHOD_TRACE_KERNEL_FILE,
        contract.METHOD_GATE_NON_PATCHABLE,
        contract.METHOD_TRACE,
        contract.METHOD_GREP,
        contract.METHOD_LLM_FALLBACK,
        contract.METHOD_LLM,
        contract.METHOD_UNRESOLVED,
    }
)

# Methods whose entries legitimately carry no source_file.
_NO_SOURCE_METHODS = frozenset(
    {contract.METHOD_UNRESOLVED, contract.METHOD_GATE_NON_PATCHABLE}
)


def validate_document(doc: Any) -> list[str]:
    """Return a list of schema violations; empty means the document looks right.

    Reports every problem rather than raising on the first, so a failure names
    all of them at once.
    """
    problems: list[str] = []
    if not isinstance(doc, dict):
        return [f"document is {type(doc).__name__}, expected dict"]
    for key in REQUIRED_DOCUMENT_KEYS:
        if key not in doc:
            problems.append(f"document missing required key {key!r}")
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
