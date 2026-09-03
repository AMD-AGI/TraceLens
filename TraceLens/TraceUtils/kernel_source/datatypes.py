###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The small result objects the rest of the package passes around.

There are three:

* :class:`SourceLocation` -- where a kernel's source is (file and line).
* :class:`Patchability`   -- can this kernel be edited at all? (a yes/no/maybe).
* :class:`ResolveResult`  -- the final answer: a :class:`Patchability` verdict
  plus, when found, a :class:`SourceLocation`.

They live in their own file (instead of in ``resolver``) so other modules can
build results without importing the resolver -- which keeps the imports simple.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceLocation:
    """A resolved source location: the file, an optional line, and its origin.

    Attributes:
        source_file: Absolute path to the source file that defines the kernel.
        line: 1-based definition line when known, else ``None``.
        framework: Best-effort label of the search path/library the file came
            from (e.g. ``"aiter"``); ``""`` when not attributed.
    """

    source_file: str
    line: int | None = None
    framework: str = ""


@dataclass(frozen=True)
class Patchability:
    """Can this kernel be edited? A quick verdict made before searching for source.

    Attributes:
        patchable: ``False`` = definitely no editable source (precompiled or
            generated); ``None`` = "not sure, go look". It is never ``True`` --
            proving a source exists is the resolver's job, not the gate's.
        kind: Why it's not patchable, when ``patchable`` is ``False`` (e.g.
            ``"tensile_precompiled"``, ``"aiter_ck"``); ``""`` otherwise.
        reason: A short human-readable version of ``kind``.
    """

    patchable: bool | None
    kind: str = ""
    reason: str = ""


@dataclass
class ResolveResult:
    """Outcome of :func:`resolver.resolve` -- a gate verdict plus a location.

    Attributes:
        location: The resolved :class:`SourceLocation`, or ``None`` when the
            kernel is non-patchable or no source was found.
        patchable: Whether an editable source exists for this kernel.
        kind: Non-patchable category when applicable (see :class:`Patchability`).
        reason: Human-readable explanation of the outcome.
        method: How the outcome was reached -- ``"gate_non_patchable"``,
            ``"symbol_index"``, ``"triton_ast"``, ``"trace_kernel_file"``,
            or ``"unresolved"``.
    """

    location: SourceLocation | None
    patchable: bool
    kind: str = ""
    reason: str = ""
    method: str = "unresolved"

    @property
    def source_file(self) -> str:
        """Resolved file path, or ``""`` when there is no location."""
        return self.location.source_file if self.location else ""

    @property
    def line(self) -> int | None:
        """Resolved 1-based line, or ``None`` when there is no location."""
        return self.location.line if self.location else None
