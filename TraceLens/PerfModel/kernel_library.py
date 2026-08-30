###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared kernel-library classification for perf reports, agents, and TraceIndex."""

from __future__ import annotations

from typing import Any, Optional

from TraceLens.PerfModel import kernel_name_parser as knp

_OP_NAME_LIBRARY_RULES = [
    ("aiter::", "AITER"),
    ("rocm_aiter", "AITER"),
    ("fbgemm", "FBGEMM"),
    ("miopen", "MIOpen"),
    ("triton", "Triton"),
]

_KERNEL_NAME_LIBRARY_RULES = [
    ("aiter", "AITER"),
    ("ck_tile::", "CK"),
    ("ck_tile6kentry", "CK"),
    ("FmhaFwd", "CK"),
    ("FmhaBwd", "CK"),
    ("Cijk_", "Tensile"),
    ("wvSplitK", "rocBLAS"),
    ("splitKreduce", "rocBLAS"),
    ("rocprim::", "rocPRIM"),
    ("triton_", "Triton"),
    ("void at::native::", "PyTorch Native"),
    ("nccl", "RCCL/NCCL"),
    ("rccl", "RCCL/NCCL"),
    ("composable", "CK"),
]

# Ordered GEMM detectors; extend when kernel_name_parser gains more libraries (#805).
_GEMM_LIBRARY_CHECKS = (
    (knp.is_rocm_gemm, "Tensile"),
    (knp.is_cuda_gemm, "nvjet"),
)


def _coerce_kernel_details(kernel_details: Any) -> str:
    if kernel_details in (None, ""):
        return ""
    if isinstance(kernel_details, float) and kernel_details != kernel_details:
        return ""
    return str(kernel_details)


def _library_from_gemm_kernel_name(kernel_name: str) -> Optional[str]:
    for detector, library in _GEMM_LIBRARY_CHECKS:
        if detector(kernel_name):
            return library
    return None


def classify_kernel_library(op_name: str, kernel_details: Any = "") -> Optional[str]:
    """Identify the GPU library backing an operation from its name or kernel strings."""
    op_lower = (op_name or "").lower()
    for marker, library in _OP_NAME_LIBRARY_RULES:
        if marker in op_lower:
            return library

    kd = _coerce_kernel_details(kernel_details)
    if kd:
        gemm_library = _library_from_gemm_kernel_name(kd)
        if gemm_library is not None:
            return gemm_library

        kd_lower = kd.lower()
        for marker, library in _KERNEL_NAME_LIBRARY_RULES:
            if marker.lower() in kd_lower:
                return library
    return None
