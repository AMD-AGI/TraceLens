###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build/cache a kernel-name -> source-file index over search paths, and discover those paths."""

from __future__ import annotations

import getpass
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = [
    "SourceIndex",
    "build_index",
    "load_or_build",
    "fingerprint",
    "reset_index_cache",
    "FrameworkRoot",
    "discover_frameworks",
    "discover_library_paths",
]

# Environment variables this module reads (collected here so every knob is
# discoverable in one place):
#   * _ENV_CACHE_DIR      -- override dir for the on-disk index cache.
#   * _ENV_SOURCE_ROOTS   -- explicit ``name=path`` framework roots (csv).
#   * _ENV_DISCOVER_ONLY  -- restrict discovery to this comma-separated allowlist.
_ENV_CACHE_DIR = "TRACELENS_KSI_CACHE_DIR"
_ENV_SOURCE_ROOTS = "TRACELENS_FRAMEWORK_SOURCE_ROOTS"
_ENV_DISCOVER_ONLY = "TRACELENS_DISCOVER_ONLY"

# Native source extensions to scan (kept in sync with .editable's editability filter).
_NATIVE_EXTS = (".cu", ".cuh", ".hip", ".h", ".hpp")

# Serving frameworks always located by name (for version reporting), even without native source.
_KNOWN = ("vllm", "sglang", "aiter", "atom")

# Fixed subpaths checked (no traversal) against a package dir's *parent* (e.g. site-packages).
_CSRC_DIRS = ("csrc", "sgl-kernel/csrc", "kernels")

# Dir names that mark a kernel-source root anywhere inside a package's own tree.
_CSRC_LEAF_NAMES = frozenset({"csrc", "kernels"})

# Dir names pruned while walking a package tree (tests/docs/vendored builds etc).
_CSRC_PRUNE_NAMES = frozenset(
    {
        "test",
        "tests",
        "testing",
        "benchmark",
        "benchmarks",
        "doc",
        "docs",
        "example",
        "examples",
        "third_party",
        "thirdparty",
        "build",
        "dist",
        "__pycache__",
        "node_modules",
    }
)

# Max depth (relative to a package's own dir) the bounded search descends.
_CSRC_MAX_DEPTH = 4

_VERSION_RE = re.compile(r"(?:__version__|version)\s*=\s*['\"]([^'\"]+)['\"]")

# --- kernel-definition scanning ---------------------------------------------
# Definition head = __global__ <attrs/return type> NAME ( params ) { ; scanned token by token.
_GLOBAL_TOKEN_RE = re.compile(r"\b__global__\b")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_ATTR_KEYWORDS = frozenset(
    {
        "__launch_bounds__",
        "launch_bounds",
        "__attribute__",
        "__maxnreg__",
        "__cluster_dims__",
        "__grid_constant__",
    }
)


def _skip_balanced_parens(text: str, open_pos: int) -> int:
    """Return the index just past the ``)`` matching the ``(`` at ``open_pos``."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def _strip_comments(text: str) -> str:
    """Blank out ``//``/``/* */`` comments (keeping length/newlines) so a commented-out ``__global__`` isn't indexed."""
    out = list(text)
    i, n = 0, len(text)
    state = 0  # 0=code, 1=line comment (//), 2=block comment (/* */)
    while i < n:
        c = text[i]
        if state == 0:
            if c == "/" and i + 1 < n and text[i + 1] == "/":
                out[i] = out[i + 1] = " "
                i += 2
                state = 1
                continue
            if c == "/" and i + 1 < n and text[i + 1] == "*":
                out[i] = out[i + 1] = " "
                i += 2
                state = 2
                continue
            i += 1
        elif state == 1:
            if c == "\n":
                state = 0
            else:
                out[i] = " "
            i += 1
        else:  # state == 2
            if c == "*" and i + 1 < n and text[i + 1] == "/":
                out[i] = out[i + 1] = " "
                i += 2
                state = 0
                continue
            if c != "\n":
                out[i] = " "
            i += 1
    return "".join(out)


def _iter_global_defs(text: str):
    """Yield ``(name, name_pos)`` for each ``__global__`` kernel definition, skipping forward declarations."""
    n = len(text)
    for gm in _GLOBAL_TOKEN_RE.finditer(text):
        pos = gm.end()
        while pos < n:
            if text[pos].isspace():
                pos += 1
                continue
            if text[pos] in ";{}":  # not a definition head we understand
                break
            m = _IDENT_RE.match(text, pos)
            if not m:  # punctuation (``*``, ``&``, ``<``, ``::`` ...)
                pos += 1
                continue
            ident, pos = m.group(0), m.end()
            after = pos
            while after < n and text[after].isspace():
                after += 1
            if after < n and text[after] == "(":
                if ident in _ATTR_KEYWORDS:
                    pos = _skip_balanced_parens(text, after)
                    continue
                # A definition needs a ``{`` body right after ``)``; a ``;`` (fwd decl) or anything else skips it.
                cursor = _skip_balanced_parens(text, after)
                while cursor < n and text[cursor].isspace():
                    cursor += 1
                if cursor < n and text[cursor] == "{":
                    yield ident, m.start()
                break
            # else: a qualifier / return-type token -- keep scanning.


def _scan_file(path: Path) -> list[tuple[str, int]]:
    """Return ``(base_name, def_line)`` for each kernel defined in ``path``."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        log.debug("kernel index: cannot read %s: %s", path, exc)
        return []
    if "__global__" not in text:
        return []
    # Blank out comments first so commented-out kernels aren't indexed (offsets stay valid).
    scan_text = _strip_comments(text)
    return [
        (name, scan_text.count("\n", 0, pos) + 1)
        for name, pos in _iter_global_defs(scan_text)
    ]


def _native_files(root: Path):
    """Yield every native source file under ``root``."""
    if not root.is_dir():
        return
    for dirpath, _dirs, names in os.walk(root):
        for nm in names:
            if nm.lower().endswith(_NATIVE_EXTS):
                yield Path(dirpath) / nm


# --- search-path normalization + fingerprint --------------------------------
def _normalize_paths(search_paths: Sequence[str | Path]) -> list[Path]:
    """De-duplicate and keep only existing directories, order-preserving."""
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in search_paths:
        p = Path(raw)
        if p not in seen and p.is_dir():
            seen.add(p)
            out.append(p)
    return out


def _dir_signature(path: Path) -> str:
    """Recursive change signature for a source dir: newest native-file mtime + file count."""
    try:
        count = 0
        max_mtime_ns = 0
        for dirpath, _dirs, names in os.walk(path):
            for nm in names:
                if not nm.lower().endswith(_NATIVE_EXTS):
                    continue
                try:
                    st = os.stat(os.path.join(dirpath, nm))
                except OSError:
                    continue
                count += 1
                if st.st_mtime_ns > max_mtime_ns:
                    max_mtime_ns = st.st_mtime_ns
        return f"{path}:{max_mtime_ns}:{count}"
    except OSError:
        return f"{path}:missing"


def fingerprint(search_paths: Sequence[str | Path]) -> str:
    """Stable cache key over the search paths' recursive signatures; changes on any add/remove/edit."""
    parts = [_dir_signature(p) for p in sorted(_normalize_paths(search_paths))]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[
        :16
    ]  # nosec B324 - cache key, not security.


# --- index ------------------------------------------------------------------
@dataclass
class SourceIndex:
    """Cached kernel-name -> source records index, plus build metadata."""

    fingerprint: str
    symbol_index: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    build_ms: float = 0.0
    file_count: int = 0
    symbol_count: int = 0

    def lookup(self, base_name: str) -> list[dict[str, object]]:
        """Return all definition records for a base kernel name (``[]`` if none)."""
        return self.symbol_index.get(base_name, [])


def build_index(search_paths: Sequence[str | Path]) -> SourceIndex:
    """Scan the given search paths and build the kernel index (timed)."""
    started = time.perf_counter()
    paths = _normalize_paths(search_paths)
    symbol_index: dict[str, list[dict[str, object]]] = {}
    file_count = 0
    for root in paths:
        for path in _native_files(root):
            defs = _scan_file(path)
            if defs:
                file_count += 1
            for base, line_no in defs:
                symbol_index.setdefault(base, []).append(
                    {"file": str(path), "line": line_no}
                )
    return SourceIndex(
        fingerprint=fingerprint(paths),
        symbol_index=symbol_index,
        build_ms=round((time.perf_counter() - started) * 1000.0, 2),
        file_count=file_count,
        symbol_count=len(symbol_index),
    )


# --- cache ------------------------------------------------------------------
def _cache_path(fp: str) -> Path:
    """Cache file path: ``$TRACELENS_KSI_CACHE_DIR`` if set, else a user-scoped (0o700) temp subdir."""
    raw = os.environ.get(_ENV_CACHE_DIR, "").strip()
    if raw:
        d = Path(raw)
    else:
        # Per-user temp subdir so users on a shared box don't share a cache.
        # POSIX uid (stable, no PII) when available, else the login name.
        uid = str(os.getuid()) if hasattr(os, "getuid") else (getpass.getuser() or "shared")
        d = Path(tempfile.gettempdir()) / f"tracelens_ksi_{uid}"
    try:
        d.mkdir(parents=True, exist_ok=True)
        if not raw:
            os.chmod(d, 0o700)  # best-effort: restrict the user-scoped temp cache.
    except OSError:
        pass  # optimization only; _save_cache no-ops and the index rebuilds
    return d / f"ksi_{fp}.json"


def _load_cache(fp: str) -> SourceIndex | None:
    try:
        with open(_cache_path(fp), encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and data.get("fingerprint") == fp:
            return SourceIndex(**data)
    except (OSError, ValueError, TypeError) as exc:
        # Any read/parse/shape error is a cache miss -> rebuild upstream.
        log.debug("kernel index: cache read miss (%s): %s", fp, exc)
        return None
    return None


def _save_cache(index: SourceIndex) -> None:
    path = _cache_path(index.fingerprint)
    try:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(asdict(index), fh)
        tmp.replace(path)
    except OSError as exc:
        # Best-effort cache write; a failure here must not break index build/use.
        log.debug("kernel index: cache write failed (%s): %s", index.fingerprint, exc)


# Per-fingerprint in-process singletons, so repeated resolves in one run skip the on-disk cache.
_PROCESS_INDEX: dict[str, SourceIndex] = {}


def reset_index_cache() -> None:
    """Drop the in-process index singletons (for tests / a forced rebuild)."""
    _PROCESS_INDEX.clear()


def load_or_build(search_paths: Sequence[str | Path]) -> SourceIndex:
    """Return a cached index (in-process -> on-disk -> fresh build); ``build_ms`` is ``0.0`` on a hit."""
    paths = _normalize_paths(search_paths)
    if not paths:
        log.warning(
            "kernel index: no search paths given/found; native symbol resolution "
            "is disabled for this call"
        )
    fp = fingerprint(paths)

    cached_singleton = _PROCESS_INDEX.get(fp)
    if cached_singleton is not None:
        return cached_singleton

    on_disk = _load_cache(fp)
    if on_disk is not None:
        on_disk.build_ms = 0.0
        _PROCESS_INDEX[fp] = on_disk
        return on_disk

    index = build_index(paths)
    log.info(
        "kernel index: built %d symbols across %d files (fingerprint=%s, build_ms=%.1f)",
        index.symbol_count,
        index.file_count,
        index.fingerprint,
        index.build_ms,
    )
    _save_cache(index)
    _PROCESS_INDEX[fp] = index
    return index


# --- discovery ----------------------------------------------------------------
@dataclass(frozen=True)
class FrameworkRoot:
    """One discovered library: its package dir, version, and native roots."""

    name: str
    root: Path
    version: str
    csrc_roots: tuple[Path, ...] = field(default_factory=tuple)


# --- locating packages -------------------------------------------------------
def _env_source_roots() -> dict[str, Path]:
    """Parse ``$TRACELENS_FRAMEWORK_SOURCE_ROOTS`` (``name=path`` csv)."""
    out: dict[str, Path] = {}
    for item in os.environ.get(_ENV_SOURCE_ROOTS, "").split(","):
        name, _, path = item.strip().partition("=")
        p = Path(path.strip())
        if name.strip() and path.strip() and p.is_dir():
            out[name.strip().lower()] = p
    return out


def _discover_only() -> set[str]:
    """Parse ``$TRACELENS_DISCOVER_ONLY`` into a lowercase allowlist (may be empty)."""
    raw = os.environ.get(_ENV_DISCOVER_ONLY, "")
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
    """All candidate top-level package dirs: site-packages/dist-packages children plus explicit overrides."""
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


# --- version + native source --------------------------------------------------
def _version(name: str, root: Path | None) -> str:
    """Installed version: dist metadata first, else a ``_version.py``/``version.py`` literal."""
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


def _iter_leaf_dirs(root: Path, max_depth: int) -> Iterator[Path]:
    """Yield subdirs of ``root`` (depth <= ``max_depth``) named like a kernel-source dir."""
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir() or child.name.startswith("."):
                continue
            lname = child.name.lower()
            if lname in _CSRC_PRUNE_NAMES:
                continue
            if lname in _CSRC_LEAF_NAMES:
                yield child
                continue  # a kernel-source dir's own contents aren't searched further
            if depth + 1 < max_depth:
                stack.append((child, depth + 1))


def _find_csrc(pkg_dir: Path) -> tuple[Path, ...]:
    """Native-source dirs: ``pkg_dir`` searched recursively, its parent checked exactly."""
    roots: list[Path] = []
    seen: set[Path] = set()

    for cand in _iter_leaf_dirs(pkg_dir, _CSRC_MAX_DEPTH):
        if cand not in seen and _has_native(cand):
            seen.add(cand)
            roots.append(cand)

    for sub in _CSRC_DIRS:
        cand = pkg_dir.parent / sub
        if cand.is_dir() and cand not in seen and _has_native(cand):
            seen.add(cand)
            roots.append(cand)

    return tuple(roots)


def _canonical(pkg_name: str) -> str:
    """Map a source-package name to its framework name (``aiter_meta`` -> ``aiter``)."""
    return pkg_name[: -len("_meta")] if pkg_name.endswith("_meta") else pkg_name


# --- framework enumeration -----------------------------------------------------
def discover_frameworks() -> dict[str, FrameworkRoot]:
    """Discover installed kernel-source libraries + versions: known frameworks by name, others auto-detected."""
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


def discover_library_paths(names: tuple[str, ...] = ()) -> list[Path]:
    """Flatten every discovered framework's ``csrc`` roots into a de-duplicated search-path list."""
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
