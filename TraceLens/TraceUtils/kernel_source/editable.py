###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Decide whether a source path is one a kernel rewrite can target.

"Editable" means native device code (``.cu``/``.cuh``/``.hip``/``.h``) or a
repo-resident Triton/TileLang ``.py``. Compiler-generated Triton is explicitly
excluded -- a file in an inductor/compile cache (``torchinductor``,
``inductor_cache``, vLLM's ``torch_compile_cache``) or anything under ``/tmp/``
is produced at compile time and has no durable source to rewrite.

This is a pure path-shape check (no filesystem access), reused by both the
native resolver and the Triton resolver.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["is_editable_source"]

# Native device-code extensions that are always editable.
_NATIVE_SOURCE_EXTS = (".cu", ".cuh", ".hip", ".h", ".hpp")

# Path markers for compiler-generated Triton (torch.compile / inductor). These
# files are produced at compile time and have no durable source to rewrite.
# Covers both the classic inductor temp dir and vLLM's on-disk compile cache
# (e.g. ``~/.cache/vllm/torch_compile_cache/.../inductor_cache/....py``).
_GENERATED_MARKERS = ("torchinductor", "inductor_cache", "torch_compile_cache")


def is_editable_source(
    path: str | None,
    kind: str | None = None,
    *,
    extra_exts: Iterable[str] | None = None,
) -> bool:
    """Return whether ``path`` is a source a kernel rewrite can target.

    Args:
        path: Candidate source path (from a trace ``kernel_file`` or the index).
        kind: Optional kernel-kind hint; ``"triton_inductor_generated"`` is
            always rejected.
        extra_exts: Optional extra file extensions to treat as editable native
            source, in addition to the built-in ones. Lets callers extend the
            set over time (e.g. ``(".cc", ".cxx")``) without editing this module.
            Case-insensitive; a leading dot is optional.

    Returns:
        ``True`` for native device code or a repo-resident Triton ``.py``;
        ``False`` for empty paths, generated Triton, and non-source files.
    """
    if not path:
        return False
    native_exts = _NATIVE_SOURCE_EXTS
    if extra_exts:
        # Normalize each extra ext to lowercase with a leading dot before adding.
        extra = tuple(
            e.lower() if e.startswith(".") else "." + e.lower() for e in extra_exts
        )
        native_exts = native_exts + extra
    low = path.lower()
    if low.endswith(native_exts):
        return True
    if low.endswith(".py"):
        if kind == "triton_inductor_generated":
            return False
        # Generated Triton lives in an inductor/compile cache or a temp dir; it
        # has no durable source to rewrite.
        if any(m in low for m in _GENERATED_MARKERS) or path.startswith(
            "/tmp/"
        ):  # nosec B108 - marker for generated artifacts.
            return False
        return True
    return False
