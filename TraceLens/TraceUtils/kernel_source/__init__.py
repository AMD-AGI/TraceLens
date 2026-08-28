###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Map a GPU kernel to its editable source, or explain why it has none.

Given a kernel name (and, for Triton, the trace's ``kernel_file``), this package
answers "where is the source I could rewrite?" against the *currently installed*
framework trees -- self-healing across file moves, renames, and version drift.

Layers, cheapest first:

* **Gate** (:func:`classify_patchability`) -- filesystem-free classification that
  short-circuits precompiled/generated kernels (Tensile, MIOpen, CK, inductor
  Triton).
* **Native resolve** (:func:`resolve_source_path`, :func:`resolve`) -- demangle
  the device symbol, look it up in a live ``__global__`` index over the caller's
  search paths, and return a verified :class:`SourceLocation`.
* **Triton resolve** (:func:`resolve_triton_source`) -- pin the ``@triton.jit``
  def line in a trace-provided ``.py`` via AST.
* **Discovery** (:func:`discover_library_paths`) -- optional helper to build the
  search paths for the installed frameworks when a caller does not supply them.

Typical use::

    from TraceLens.TraceUtils.kernel_source import resolve

    result = resolve(device_kernel_name, search_paths, op_name=op)
    if result.patchable:
        print(result.source_file, result.line)
    else:
        print("non-patchable:", result.kind, result.reason)
"""

from __future__ import annotations

from .discovery import FrameworkRoot, discover_frameworks, discover_library_paths
from .editable import is_editable_source
from .index import SourceIndex, build_index, load_or_build
from .datatypes import Patchability, ResolveResult, SourceLocation
from .patchability import classify_patchability
from .resolver import resolve, resolve_source_path
from .triton_pin import resolve_triton_source, triton_def_line

__all__ = [
    # Data types
    "SourceLocation",
    "Patchability",
    "ResolveResult",
    # Gate
    "classify_patchability",
    # Native resolution
    "resolve",
    "resolve_source_path",
    # Triton resolution
    "resolve_triton_source",
    "triton_def_line",
    # Discovery + index
    "discover_library_paths",
    "discover_frameworks",
    "FrameworkRoot",
    "SourceIndex",
    "build_index",
    "load_or_build",
    # Editability
    "is_editable_source",
]
