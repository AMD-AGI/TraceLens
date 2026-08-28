###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Optional helper: the resolver normally takes explicit folders to search; this
supplies sensible defaults for the currently installed frameworks when a caller
doesn't want to manage them.

Find where framework kernel source lives, so callers get default search paths.

It finds source two ways:

1. By name, for the known frameworks (``vllm`` / ``sglang`` / ``aiter``).
2. Automatically, for any other installed package that ships kernel source
   (a ``csrc``/``kernels`` folder with ``.cu``/``.hip``/... files).

Optional environment overrides:

* ``TRACELENS_DISCOVER_ONLY`` -- only look at these packages (comma-separated).
* ``TRACELENS_FRAMEWORK_SOURCE_ROOTS`` -- pin a package to a folder
  (``name=path``, comma-separated).
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["FrameworkRoot", "discover_frameworks", "discover_library_paths"]

# Serving frameworks always located by name (for version reporting), even when
# they ship no native source.
_KNOWN = ("vllm", "sglang", "aiter")

# Subdirectories (relative to a package dir) that hold GPU kernel source.
_CSRC_DIRS = ("csrc", "sgl-kernel/csrc", "kernels")
_NATIVE_EXTS = (".cu", ".cuh", ".hip", ".h", ".hpp")

_VERSION_RE = re.compile(r"(?:__version__|version)\s*=\s*['\"]([^'\"]+)['\"]")


@dataclass(frozen=True)
class FrameworkRoot:
    """One discovered library: its package dir, version, and native roots."""

    name: str
    root: Path
    version: str
    csrc_roots: tuple[Path, ...] = field(default_factory=tuple)


# ----------------------------------------------------------------------------
# Locating packages
# ----------------------------------------------------------------------------
def _env_source_roots() -> dict[str, Path]:
    """Parse ``$TRACELENS_FRAMEWORK_SOURCE_ROOTS`` (``name=path`` csv)."""
    out: dict[str, Path] = {}
    for item in os.environ.get("TRACELENS_FRAMEWORK_SOURCE_ROOTS", "").split(","):
        name, _, path = item.strip().partition("=")
        p = Path(path.strip())
        if name.strip() and path.strip() and p.is_dir():
            out[name.strip().lower()] = p
    return out


def _discover_only() -> set[str]:
    """Parse ``$TRACELENS_DISCOVER_ONLY`` into a lowercase allowlist (may be empty)."""
    raw = os.environ.get("TRACELENS_DISCOVER_ONLY", "")
    return {n.strip().lower() for n in raw.split(",") if n.strip()}


def _spec_root(name: str) -> Path | None:
    """Best-effort package directory via ``find_spec`` (no import)."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, ModuleNotFoundError):
        return None
    if spec is None:
        return None
    for loc in list(getattr(spec, "submodule_search_locations", None) or []):
        if Path(loc).is_dir():
            return Path(loc)
    origin = getattr(spec, "origin", None)
    if (
        origin
        and origin not in ("built-in", "namespace")
        and Path(origin).parent.is_dir()
    ):
        return Path(origin).parent
    return None


def _locate(name: str, env_roots: dict[str, Path]) -> Path | None:
    """Locate a known framework: env override, then ``find_spec``."""
    return env_roots.get(name) or _spec_root(name)


def _package_dirs(env_roots: dict[str, Path]) -> dict[str, Path]:
    """All candidate top-level package dirs, keyed by dir name (first wins).

    Sources: every child of a ``site-packages``/``dist-packages`` dir on
    ``sys.path``, plus explicit override paths.
    """
    dirs: dict[str, Path] = {}
    for entry in sys.path:
        base = Path(entry)
        if base.name in ("site-packages", "dist-packages") and base.is_dir():
            for child in base.iterdir():
                if child.is_dir():
                    dirs.setdefault(child.name, child)
    for name, path in env_roots.items():
        dirs.setdefault(name, path)
    return dirs


# ----------------------------------------------------------------------------
# Version + native source
# ----------------------------------------------------------------------------
def _version(name: str, root: Path | None) -> str:
    """Installed version: dist metadata first, then a ``_version.py`` literal.

    The source fallback covers packages that ship no dist metadata but record a
    version file; it avoids importing the package (which can require a GPU at
    import time).
    """
    try:
        return importlib_metadata.version(name)
    except (importlib_metadata.PackageNotFoundError, ValueError, OSError):
        pass
    if root is not None:
        for cand in ("_version.py", "version.py"):
            try:
                m = _VERSION_RE.search(
                    (root / cand).read_text(encoding="utf-8", errors="ignore")
                )
            except OSError:
                continue
            if m:
                return m.group(1)
    return ""


def _has_native(directory: Path) -> bool:
    """True if ``directory`` contains any native kernel-source file."""
    for _dirpath, _dirs, names in os.walk(directory):
        if any(nm.lower().endswith(_NATIVE_EXTS) for nm in names):
            return True
    return False


def _find_csrc(pkg_dir: Path) -> tuple[Path, ...]:
    """Native-source dirs under ``pkg_dir`` (and its parent) that hold kernels."""
    roots: list[Path] = []
    for base in (pkg_dir, pkg_dir.parent):
        for sub in _CSRC_DIRS:
            cand = base / sub
            if cand.is_dir() and cand not in roots and _has_native(cand):
                roots.append(cand)
    return tuple(roots)


def _canonical(pkg_name: str) -> str:
    """Map a source-package name to its framework name (``aiter_meta`` -> ``aiter``)."""
    return pkg_name[: -len("_meta")] if pkg_name.endswith("_meta") else pkg_name


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------
def discover_frameworks() -> dict[str, FrameworkRoot]:
    """Discover installed kernel-source libraries + versions.

    Known serving frameworks are located by name (so their versions are always
    reported); any other package shipping native kernel source is auto-detected.
    ``$TRACELENS_DISCOVER_ONLY`` restricts the result to named packages.

    Returns:
        Mapping of framework name to :class:`FrameworkRoot` for each one found.
    """
    env_roots = _env_source_roots()
    only = _discover_only()
    out: dict[str, FrameworkRoot] = {}

    # 1) Known serving frameworks (by name) -- kept even without native source.
    for name in _KNOWN:
        if only and name not in only:
            continue
        root = _locate(name, env_roots)
        if root is not None:
            out[name] = FrameworkRoot(
                name, root, _version(name, root), _find_csrc(root)
            )

    # 2) Auto-enumerate any other package that ships kernel source.
    for pkg_name, pkg_dir in _package_dirs(env_roots).items():
        name = _canonical(pkg_name)
        if only and name not in only:
            continue
        csrc = _find_csrc(pkg_dir)
        if not csrc:
            continue
        if name in out:
            # Merge (e.g. aiter_meta/csrc into the aiter located by name).
            existing = out[name]
            merged = tuple(dict.fromkeys(existing.csrc_roots + csrc))
            version = existing.version or _version(name, pkg_dir)
            out[name] = FrameworkRoot(name, existing.root, version, merged)
        else:
            out[name] = FrameworkRoot(name, pkg_dir, _version(name, pkg_dir), csrc)
    return out


def discover_library_paths(names: tuple[str, ...] = _KNOWN) -> list[Path]:
    """Return native-source search paths for the installed frameworks.

    Convenience wrapper over :func:`discover_frameworks` that flattens every
    discovered ``csrc`` root into a de-duplicated list a caller can pass straight
    to :func:`index.build_index` / :func:`resolver.resolve_source_path`.

    Args:
        names: Framework names to keep. Defaults to the known serving frameworks;
            pass a wider/narrower tuple to include auto-enumerated packages or
            restrict the result. Auto-enumerated packages are included only when
            their canonical name is in ``names``.

    Returns:
        De-duplicated list of existing ``csrc`` directories, order-preserving.
    """
    wanted = {n.strip().lower() for n in names if n.strip()}
    paths: list[Path] = []
    seen: set[Path] = set()
    for fw_name, fr in discover_frameworks().items():
        if wanted and fw_name.lower() not in wanted:
            continue
        for root in fr.csrc_roots:
            if root not in seen and root.is_dir():
                seen.add(root)
                paths.append(root)
    return paths
