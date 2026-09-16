###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Turn a raw GPU kernel name from a trace into its bare name.

A profiler may report a kernel three ways, all handled by :func:`base_symbol`:

* plain name:      ``my_fused_kernel``
* C++ signature:   ``void ns::my_fused_kernel<float>(float*, int)``
* mangled ``_Z``:  ``_ZN2ns15my_fused_kernelEPfi``

The mangled form is decoded by :func:`demangle`, which uses the
``itanium-demangler`` package when it is installed and otherwise falls back to a
tiny built-in parser. Anything that can't be decoded returns ``""``.
"""

from __future__ import annotations

import functools
import logging
import re

log = logging.getLogger(__name__)

try:
    from itanium_demangler import parse as _itanium_parse
except ImportError:  # pragma: no cover - warning path when extra is absent.
    _itanium_parse = None
    log.warning(
        "itanium-demangler is not installed. Kernel classification may be degraded. "
        "Install it with: pip install itanium-demangler (or the TraceLens "
        "[kernel_source] extra)."
    )

__all__ = ["base_symbol", "demangle"]

# A plain C/C++ identifier (used by the length-prefix fallback parser).
_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")

# The itanium-demangler package prints constructors/destructors as placeholder
# tokens (``{ctor}``, ``{base ctor}``, ``{dtor}``, ``{deleting dtor}``, ...) in
# place of the real ``ClassName`` / ``~ClassName``. We recover the real name from
# the qualifier in front of the token (see ``_base_from_demangled``). (Note: a
# lambda's ``{lambda(...)#1}`` is *not* matched here -- it has no ctor/dtor word.)
_ITANIUM_CTOR_DTOR_RE = re.compile(r"\{[^{}]*\b(?:c|d)tor\}")


@functools.lru_cache(maxsize=8192)
def demangle(mangled: str) -> str:
    """Decode a mangled ``_Z...`` name into a readable C++ signature.

    Uses the ``itanium-demangler`` package when installed; returns ``""`` if it
    is unavailable or can't decode the name (callers then fall back to the
    built-in length-prefix parser).

    Example:
        ``_ZN2ns6kernelEPf`` -> ``ns::kernel(float*)`` (``""`` if it can't decode)
    """
    if _itanium_parse is None:
        return ""
    # Only runs where the itanium-demangler extra is installed (it is, in CI and
    # via the [kernel_source] extra); excluded from the line gate for bare envs.
    try:  # pragma: no cover
        node = _itanium_parse(mangled)
        if node is not None:
            return str(node)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        log.debug("itanium demangle failed for %r: %s", mangled, exc)
    return ""  # pragma: no cover


def _strip_trailing_qualifiers(s: str) -> str:
    """Strip trailing cv/ref qualifiers (``const``/``volatile``/``&``/``&&``) after the arg list.

    ``&``/``&&`` are stripped even with no leading space (some demanglers glue them to the
    name, e.g. ``run_rvalue&&``) since those characters can never be part of a real identifier.
    ``const``/``volatile`` require a leading space since they could otherwise (in principle)
    be a substring of a legitimate name.
    """
    while True:
        for suf in (" const", " volatile", " &&", "&&", " &", "&"):
            if s.endswith(suf):
                s = s[: -len(suf)]
                break
        else:
            return s


def _rstrip_balanced(s: str, open_ch: str, close_ch: str) -> str:
    """Drop a trailing ``open_ch...close_ch`` group matched from the end (unchanged if unbalanced)."""
    if not s.endswith(close_ch):
        return s
    depth = 0
    for i in range(len(s) - 1, -1, -1):
        if s[i] == close_ch:
            depth += 1
        elif s[i] == open_ch:
            depth -= 1
            if depth == 0:
                return s[:i]
    return s


def _base_from_demangled(name: str) -> str:
    """Pull the bare kernel name out of a readable C++ signature (e.g. ``void ns::my_kernel<float>(float*, int)`` -> ``my_kernel``)."""
    n = (name or "").strip()
    if not n:
        return ""
    if n.startswith("void "):
        n = n[len("void ") :].strip()
    n = n.replace("(anonymous namespace)::", "")
    # Trailing groups are stripped by matching from the *end* (not the first "(" / "<" found),
    # so an operator's own "()", a lambda's "{lambda(...)...}" scope, or a class template's
    # "<...>" earlier in the qualified name isn't mistaken for the true trailing arg list.
    n = _strip_trailing_qualifiers(n)
    n = _rstrip_balanced(n, "(", ")")
    n = _strip_trailing_qualifiers(n)
    n = _rstrip_balanced(n, "<", ">")
    n = n.strip()
    # itanium-demangler leaves a ctor/dtor as a ``{ctor}``/``{dtor}`` placeholder;
    # the real name is the class, which is the qualifier just before it. Recover
    # it as ``ClassName`` (ctor) or ``~ClassName`` (dtor) rather than emit "{ctor}".
    if _ITANIUM_CTOR_DTOR_RE.search(n):
        parts = n.split("::")
        placeholder = parts[-1]
        cls = parts[-2].split("<", 1)[0] if len(parts) >= 2 else ""
        if not cls:
            return ""
        return ("~" + cls) if "dtor" in placeholder else cls
    if "::" in n:
        n = n.rsplit("::", 1)[-1]
    return n


def _base_from_mangled(mangled: str) -> str:
    """Last-resort decoder: read the name straight out of the mangled string.

    Used only when no real demangler is available. Mangled names write each word
    as a count then that many characters (``2ns`` = ``ns``, ``6kernel`` =
    ``kernel``); this collects those words and returns the first containing
    ``"kernel"``, else the last (or ``""`` if none).

    Example:
        ``_ZN2ns6kernelEPf`` -> ``kernel``
    """
    names: list[str] = []
    i, n = 0, len(mangled)
    while i < n:
        if mangled[i].isdigit():
            j = i
            while j < n and mangled[j].isdigit():
                j += 1
            length = int(mangled[i:j])
            ident = mangled[j : j + length]
            if _IDENT_RE.match(ident):
                names.append(ident)
            i = j + length
        else:
            i += 1
    if not names:
        return ""
    for nm in reversed(names):
        if "kernel" in nm.lower():
            return nm
    return names[-1]


@functools.lru_cache(maxsize=8192)
def base_symbol(device_kernel_name: str) -> str:
    """Turn any kernel name from a trace into its bare name. Main entry point.

    Handles all three forms: a plain name, a C++ signature, or a mangled
    ``_Z...`` name (decoded first). Returns the bare name (e.g. ``my_fused_kernel``),
    or ``""`` if the input is empty.
    """
    raw = (device_kernel_name or "").strip()
    if not raw:
        return ""
    if raw.startswith("_Z"):
        demangled = demangle(raw)
        if demangled and demangled != raw:
            return _base_from_demangled(demangled)
        return _base_from_mangled(raw)
    return _base_from_demangled(raw)
