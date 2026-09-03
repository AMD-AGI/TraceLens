###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Map a GPU kernel to its editable source (or explain why it has none) via gate -> native/Triton resolve -> discovery."""

from __future__ import annotations

from .editable import is_editable_source
from .index import (
    FrameworkRoot,
    SourceIndex,
    build_index,
    discover_frameworks,
    discover_library_paths,
    load_or_build,
)
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
