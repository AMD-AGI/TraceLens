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

The mangled form is decoded by :func:`demangle`, which uses whichever of these is
available: the ``itanium-demangler`` package, the ``c++filt`` program, or a tiny
built-in parser. Anything that can't be decoded returns ``""``.
"""

from __future__ import annotations

import functools
import logging
import re
import shutil
import subprocess  # nosec B404 - invokes c++filt with a fixed, non-shell argv.

log = logging.getLogger(__name__)

try:
    from itanium_demangler import parse as _itanium_parse
except ImportError:
    _itanium_parse = None
    log.warning(
        "itanium-demangler is not installed. Kernel classification may be degraded. "
        "Install it with: pip install itanium-demangler (or the TraceLens "
        "[kernel_source] extra)."
    )

__all__ = ["base_symbol", "demangle"]

# A plain C/C++ identifier (used by the length-prefix fallback parser).
_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


@functools.lru_cache(maxsize=8192)
def _cxxfilt_base(mangled: str) -> str:
    """Decode a mangled name by shelling out to the ``c++filt`` program.

    The second-choice decoder, used when ``itanium-demangler`` isn't available.
    Cached, so each distinct name only launches ``c++filt`` once. Returns ``""``
    if ``c++filt`` is missing or can't decode the name.

    Example:
        ``_Z6kernelv`` -> ``kernel()``
    """
    if not shutil.which("c++filt"):
        return ""
    try:
        proc = subprocess.run(  # nosec B603 B607 - fixed argv, no shell.
            ["c++filt", mangled],
            capture_output=True,
            text=True,
            timeout=5,
        )
        result = proc.stdout.strip()
        # c++filt echoes the input unchanged on failure -- treat that as no result.
        return result if result != mangled else ""
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("c++filt demangle failed for %r: %s", mangled, exc)
        return ""


@functools.lru_cache(maxsize=8192)
def demangle(mangled: str) -> str:
    """Decode a mangled ``_Z...`` name into a readable C++ signature.

    Uses the ``itanium-demangler`` package if available, otherwise ``c++filt``.

    Example:
        ``_ZN2ns6kernelEPf`` -> ``ns::kernel(float*)`` (``""`` if neither works)
    """
    if _itanium_parse is None:
        return _cxxfilt_base(mangled)
    try:
        node = _itanium_parse(mangled)
        if node is not None:
            return str(node)
    except Exception as exc:  # noqa: BLE001 - malformed symbols must not propagate.
        log.debug("itanium demangle failed for %r: %s", mangled, exc)
    # itanium returned nothing / raised: try c++filt before giving up.
    return _cxxfilt_base(mangled)


def _base_from_demangled(name: str) -> str:
    """Pull the bare kernel name out of a readable C++ signature.

    Strips the return type, namespaces, template ``<...>``, and arguments ``(...)``.
    ``(anonymous namespace)::`` is removed first so such names don't wrongly
    collapse to ``void``.

    Example:
        ``void ns::sub::my_kernel<float>(float*, int)`` -> ``my_kernel``
    """
    n = (name or "").strip()
    if not n:
        return ""
    if n.startswith("void "):
        n = n[len("void ") :].strip()
    n = n.replace("(anonymous namespace)::", "")
    n = re.sub(r"<.*$", "", n)
    n = re.sub(r"\(.*$", "", n)
    n = n.strip()
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
