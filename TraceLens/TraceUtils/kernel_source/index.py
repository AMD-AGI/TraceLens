###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build (and cache) a kernel-name -> source-file index over the search paths.

The index answers: "which source file and line defines kernel ``<name>``?"
It scans the search folders once for ``__global__`` kernel definitions and saves
the result to a disk cache, so later lookups are almost free.

The cache is keyed to the folders' contents, so editing any ``.cu`` under them
rebuilds the index automatically.

Triton ``.py`` kernels are not indexed here -- there are thousands of them, so
they're resolved on demand from the trace instead (see :mod:`.triton_pin`).
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = [
    "SourceIndex",
    "build_index",
    "load_or_build",
    "fingerprint",
    "reset_index_cache",
]

# Native source extensions to scan. Kept in sync with the editability filter in
# :mod:`.editable`: a ``__global__`` def indexed from an extension the resolver
# would later reject as non-editable is dead weight, so both lists must agree.
_NATIVE_EXTS = (".cu", ".cuh", ".hip", ".h", ".hpp")

# --- kernel-definition scanning ---------------------------------------------
# A definition head is ``__global__`` <attrs / return type> NAME ( params ) { .
# The tricky part is attributes that carry their own parentheses -- notably
# ``__launch_bounds__(NUM_THREADS)`` and ``__attribute__((...))``. A naive
# ``__global__[^()]*?NAME(`` regex stops at the attribute's ``(`` and captures the
# *attribute* as the kernel name. So we scan token by token from ``__global__``,
# skip any attribute call (balanced parens), and take the first remaining
# identifier that is directly followed by ``(`` and a ``{`` body.
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
    """Blank out ``//`` and ``/* */`` comments, preserving length and newlines.

    Comment characters are replaced with spaces (newlines kept) so that a
    ``__global__`` sitting inside commented-out code is not scanned as a
    definition, while every byte offset and line number still lines up with the
    original text. String literals are not tracked -- a ``__global__`` inside a
    string is vanishingly unlikely and not worth the added complexity.
    """
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
    """Yield ``(name, name_pos)`` for each ``__global__`` kernel *definition*.

    Only definitions (a parameter list immediately followed by a ``{`` body) are
    yielded. Forward declarations (``... );``), and ``__global__`` text living
    inside comments, are rejected so the index never points a rewrite at a header
    declaration or dead code.
    """
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
                # Definition, not a declaration: the first non-space character
                # after the matching ``)`` must open a body ``{``. A ``;`` (fwd
                # decl) or anything else (a match inside a comment/string) is
                # skipped -- this ``__global__`` yields no name.
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
    # Blank out comments first so commented-out kernels are not indexed. The
    # stripper preserves length + newlines, so offsets/line numbers still match.
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
    """Recursive change signature for a source dir.

    Folds in the newest native-file ``mtime_ns`` and the native-file count, so an
    edit to a file in a nested subdirectory changes the signature and invalidates
    a stale cached index.
    """
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
    """Stable cache key over the search paths and their recursive signatures.

    Changes when a search path is added/removed or any native source file under
    one is added, removed, or modified, so a cached index is reused iff still
    valid.
    """
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
    """Cache file path (dir from ``$TRACELENS_KSI_CACHE_DIR`` or a temp subdir).

    When falling back to the system temp root (typically a shared ``/tmp`` on a
    multi-user host), the subdir is scoped to the current user and created
    owner-only (0o700) so users cannot collide on each other's cache. An explicit
    ``$TRACELENS_KSI_CACHE_DIR`` is used verbatim.
    """
    raw = os.environ.get("TRACELENS_KSI_CACHE_DIR", "").strip()
    if raw:
        d = Path(raw)
    else:
        try:
            uid = str(os.getuid())  # POSIX: stable per-user, no PII.
        except AttributeError:  # non-POSIX platforms
            uid = getpass.getuser() or "shared"
        d = Path(tempfile.gettempdir()) / f"tracelens_ksi_{uid}"
    try:
        d.mkdir(parents=True, exist_ok=True)
        if not raw:
            os.chmod(d, 0o700)  # best-effort: restrict the user-scoped temp cache.
    except OSError:
        # The on-disk cache is an optimization; if the dir cannot be created,
        # _save_cache no-ops and the index simply rebuilds.
        pass
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


# Per-fingerprint in-process singletons, so repeated resolves for the same search
# paths within one run do not re-read the on-disk cache.
_PROCESS_INDEX: dict[str, SourceIndex] = {}


def reset_index_cache() -> None:
    """Drop the in-process index singletons (for tests / a forced rebuild)."""
    _PROCESS_INDEX.clear()


def load_or_build(search_paths: Sequence[str | Path]) -> SourceIndex:
    """Return a cached index for the given search paths, or build + cache one.

    Resolution order: in-process singleton -> on-disk cache -> fresh build.
    ``build_ms`` is ``0.0`` on a cache hit and the real build time on a miss.
    """
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
