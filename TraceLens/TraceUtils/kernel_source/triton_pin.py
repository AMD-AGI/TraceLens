###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Resolve a Triton ``.py`` kernel from the trace's ``kernel_file``.

Native ``.cu``/``.hip`` kernels are found by symbol lookup (:mod:`.resolver`);
Triton kernels come straight from the trace, which records the launcher's
``kernel_file``. This module turns that into an editable source location:

1. parse the launcher form (``a.py:12:foo`` / ``a.py(12): foo`` / ``a.py#L12``)
   down to a bare ``.py`` path;
2. reject generated Triton (inductor cache / ``/tmp``) as non-patchable;
3. pin the exact ``@triton.jit`` def line via AST (no import of the kernel).

The AST step is a pure refinement: a resolved file is still returned when the
def line cannot be pinned.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

from .editable import is_editable_source
from .datatypes import ResolveResult, SourceLocation

__all__ = ["resolve_triton_source", "triton_def_line", "editable_trace_source"]

# Triton decorators marking a device-kernel def (``@triton.jit`` / ``@jit`` and
# the autotune/heuristics wrappers that sit on top of a jit'd kernel).
_TRITON_DECORATORS = frozenset({"jit", "autotune", "heuristics"})

# Launcher-path forms a trace ``kernel_file`` may carry instead of a bare path:
# ``<path>(<line>): <func>``, ``<path>:<line>:<func>``, or ``<path>#L<line>``.
_LAUNCHER_PATH_RE = re.compile(
    r"^(?P<path>.+?\.py)"
    r"(?:\((?P<pline>\d+)\)|[:#]L?(?P<cline>\d+))"
    r"(?::?\s*(?P<func>[A-Za-z_]\w*))?\s*$"
)


def _parse_launcher_form(raw: str) -> tuple[str, int | None, str]:
    """Split a trace ``kernel_file`` into ``(py_path, line, func)``.

    Handles the plain-path case (no line/func) and the launcher forms
    ``a.py(12): foo`` / ``a.py:12:foo`` / ``a.py#L12``. Non-``.py`` inputs are
    returned unchanged with no line/func.
    """
    text = str(raw or "").strip()
    if not text:
        return "", None, ""
    match = _LAUNCHER_PATH_RE.match(text)
    if match:
        line_str = match.group("pline") or match.group("cline")
        return (
            match.group("path").strip(),
            int(line_str) if line_str else None,
            match.group("func") or "",
        )
    return text, None, ""


def editable_trace_source(kernel_file: str, kind: str = "") -> str:
    """Return a trace ``kernel_file`` iff it is an editable source, else ``""``.

    A repo-resident ``.py`` is directly editable; inductor-generated / ``/tmp``
    Triton is not.
    """
    kf = str(kernel_file or "").strip()
    if not kf:
        return ""
    return kf if is_editable_source(kf, kind or None) else ""


def _is_triton_kernel_def(node: ast.AST) -> bool:
    """Return whether an AST function node carries a Triton kernel decorator."""
    for dec in getattr(node, "decorator_list", []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = getattr(target, "attr", None) or getattr(target, "id", None)
        if name in _TRITON_DECORATORS:
            return True
    return False


def _normalize_symbol(symbol: str) -> str:
    """Reduce a device kernel symbol to a bare identifier core for matching.

    Triton device symbols often wrap the ``@triton.jit`` function name with a
    leading ``triton_``/``_`` prefix and a trailing autotune/hash suffix
    (e.g. ``_fwd_kernel_0d1d2``). Strip those so a fuzzy match against the def
    name has a chance.
    """
    core = re.sub(r"[^0-9A-Za-z_].*$", "", str(symbol or "").strip())
    core = re.sub(r"_+\d[\dA-Za-z]*$", "", core)  # drop trailing autotune/hash suffix
    return core.strip("_").lower()


def triton_def_line(py_path: str, *, func: str = "", symbol: str = "") -> int | None:
    """Find a Triton kernel's ``def`` line in a ``.py`` via AST (no import).

    Matching precedence: (1) exact ``func`` name; (2) a ``@triton.jit`` def whose
    name matches the normalized device ``symbol`` (exact then substring); (3) the
    sole ``@triton.jit`` def in the file when unambiguous. Returns ``None`` when
    the file is unreadable/unparseable or no confident match is found.
    """
    try:
        tree = ast.parse(Path(py_path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return None

    jit_defs: dict[str, int] = {}
    all_defs: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_defs.setdefault(node.name, node.lineno)
            if _is_triton_kernel_def(node):
                jit_defs.setdefault(node.name, node.lineno)

    if func and func in all_defs:
        return all_defs[func]

    core = _normalize_symbol(symbol)
    if core:
        for name, line in jit_defs.items():
            if name.lower() == core:
                return line
        for name, line in jit_defs.items():
            low = name.lower()
            if core in low or low in core:
                return line

    if len(jit_defs) == 1:
        return next(iter(jit_defs.values()))
    return None


def resolve_triton_source(
    kernel_file: str,
    *,
    kind: str = "",
    symbol: str = "",
) -> ResolveResult:
    """Resolve a trace ``kernel_file`` to an editable Triton ``.py`` + def line.

    Args:
        kernel_file: The ``kernel_file`` string from the trace (may be a bare
            path or a launcher form like ``a.py:12:foo``).
        kind: Optional kernel-kind hint; ``"triton_inductor_generated"`` is
            treated as non-patchable.
        symbol: Optional device kernel symbol, used to pin the exact def line.

    Returns:
        A :class:`~.datatypes.ResolveResult`. ``method`` is ``"triton_ast"`` (path +
        pinned line), ``"trace_kernel_file"`` (path only), ``"gate_non_patchable"``
        (generated Triton), or ``"unresolved"`` (empty/unusable input).
    """
    path, line, func = _parse_launcher_form(kernel_file)
    if not path:
        return ResolveResult(
            None, patchable=False, method="unresolved", reason="empty kernel_file"
        )

    source = editable_trace_source(path, kind)
    if not source:
        return ResolveResult(
            None,
            patchable=False,
            kind="triton_inductor_generated",
            reason="generated/non-editable Triton source",
            method="gate_non_patchable",
        )

    ast_line: int | None = None
    if source.lower().endswith(".py") and os.path.isfile(source):
        ast_line = triton_def_line(source, func=func, symbol=symbol)
    def_line = ast_line if ast_line is not None else line
    method = "triton_ast" if ast_line is not None else "trace_kernel_file"
    return ResolveResult(
        location=SourceLocation(source_file=source, line=def_line),
        patchable=True,
        method=method,
    )
