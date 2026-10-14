###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Point a precompiled Tensile kernel at the library file it ships inside.

Tensile GEMM kernels (``Cijk_*``) have no editable ``.cu`` source: they ship as
compiled code in the rocBLAS/hipBLASLt ``library`` directory. We can't hand back
a rewritable file, but we can point at the Tensile *logic* file (``.dat``) whose
solution set includes this kernel -- an audit breadcrumb, not a rewrite target.

The match is by content: a kernel's macro-tile token (``MT<m>x<n>x<k>``) is stored
verbatim inside its logic file, so we scan the target arch's logic files for that
token, after narrowing by operand layout and data type from the kernel name. A
miss returns ``None`` and the caller keeps the empty location it already had.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .datatypes import SourceLocation

__all__ = ["resolve_library_artifact", "discover_library_dirs"]

# csv / os.pathsep list of rocBLAS/hipBLASLt "library" dirs; overrides discovery.
_ENV_LIBRARY_DIRS = "TRACELENS_ROCM_LIBRARY_DIRS"
# Target GPU arch (e.g. "gfx942"); shared with the native resolver's ranking.
_ENV_TARGET_ARCH = "TRACELENS_TARGET_ARCH"

# The macro-tile token a Tensile solution name and its logic file both carry.
_TILE_RE = re.compile(rb"MT\d+x\d+x\d+")
# Operand layout + data-type tokens right after "Cijk_" (e.g. "Alik", "Bljk",
# "BBS"); the layout and the type prefix also appear in the logic file's name.
_SOLUTION_RE = re.compile(r"^Cijk_(A[a-z]+)_(B[a-z]+)_([A-Za-z0-9]+)")

# rocBLAS/hipBLASLt ship their solution libraries under these subpaths of a ROCm
# install; each holds the per-arch ``.dat`` logic files (and ``.co``/.hsaco code).
_LIBRARY_SUBDIRS = ("lib/rocblas/library", "lib/hipblaslt/library")
_LIBRARY_NAMES = ("hipblaslt", "rocblas")


def _rocm_bases() -> list[Path]:
    """Candidate ROCm install roots: ``$ROCM_PATH``/``$ROCM_HOME``, then ``/opt/rocm*``."""
    bases: list[Path] = []
    for var in ("ROCM_PATH", "ROCM_HOME"):
        val = os.environ.get(var, "").strip()
        if val:
            bases.append(Path(val))
    bases.extend(sorted(Path("/opt").glob("rocm*")))
    return bases


def discover_library_dirs() -> list[Path]:
    """rocBLAS/hipBLASLt ``library`` dirs, from ``$TRACELENS_ROCM_LIBRARY_DIRS`` or a ROCm install."""
    raw = os.environ.get(_ENV_LIBRARY_DIRS, "").strip()
    if raw:
        candidates = [
            Path(p) for p in re.split(rf"[,{re.escape(os.pathsep)}]", raw) if p.strip()
        ]
    else:
        candidates = [base / sub for base in _rocm_bases() for sub in _LIBRARY_SUBDIRS]

    dirs: list[Path] = []
    seen: set[str] = set()
    for d in candidates:
        real = os.path.realpath(d)  # /opt/rocm is usually a symlink to /opt/rocm-<ver>.
        if real not in seen and os.path.isdir(real):
            seen.add(real)
            dirs.append(Path(real))
    return dirs


def _library_label(path: Path) -> str:
    """``"rocblas"``/``"hipblaslt"`` inferred from the artifact path (``""`` if neither)."""
    low = str(path).lower()
    for name in _LIBRARY_NAMES:
        if name in low:
            return name
    return ""


def resolve_library_artifact(
    kernel_name: str,
    *,
    dirs: list[Path] | None = None,
    arch: str | None = None,
) -> SourceLocation | None:
    """Point a precompiled Tensile kernel at its ``.dat`` logic file, or ``None``.

    Args:
        kernel_name: The ``Cijk_*`` solution name from the trace.
        dirs: Library dirs to scan; defaults to :func:`discover_library_dirs`.
        arch: Target GPU arch (e.g. ``"gfx942"``); defaults to
            ``$TRACELENS_TARGET_ARCH``. Other arches' logic files are skipped.

    Returns:
        A non-editable :class:`~.datatypes.SourceLocation` at the ``.dat`` logic
        file, or ``None`` when the name has no tile token or nothing matches.
    """
    token_match = _TILE_RE.search((kernel_name or "").encode("ascii", "ignore"))
    if not token_match:
        return None
    tile = token_match.group(0)

    arch = (
        (arch if arch is not None else os.environ.get(_ENV_TARGET_ARCH, ""))
        .strip()
        .lower()
    )
    dirs = dirs if dirs is not None else discover_library_dirs()

    sol = _SOLUTION_RE.match(kernel_name or "")
    layout = [g.lower() for g in sol.groups()[:2]] if sol else []
    # The logic-file name tags input+output dtype as e.g. "Type_BB"; the solution
    # name's third token starts with those same two letters (BBS -> BB).
    dtype_tag = ("type_" + sol.group(3)[:2].lower()) if sol else ""

    candidates: list[Path] = []
    for d in dirs:
        for dat in d.glob("*.dat"):
            low = dat.name.lower()
            # A "fallback" logic file serves every arch; others must match ours.
            if arch and arch not in low and "fallback" not in low:
                continue
            if layout and not all(tok in low for tok in layout):
                continue
            candidates.append(dat)

    def rank(p: Path) -> tuple[int, int]:
        low = p.name.lower()
        return (1 if dtype_tag and dtype_tag in low else 0, -len(str(p)))

    for dat in sorted(candidates, key=rank, reverse=True):
        try:
            if tile in dat.read_bytes():
                return SourceLocation(
                    source_file=str(dat), framework=_library_label(dat)
                )
        except OSError:
            continue
    return None
