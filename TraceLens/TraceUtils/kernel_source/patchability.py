###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Cheap gate: rule out kernels that can't have editable source.

Given just a kernel name (plus optional op name and call stack), it flags
kernels that are precompiled or compiler-generated, so we don't waste time
searching for source that doesn't exist. It does no filesystem I/O.

Rejected categories (all non-patchable):

* **Tensile** (``Cijk_*``): precompiled GEMM assembly, no ``.cu`` source.
* **MIOpen** (op name contains ``miopen``): precompiled convolution kernels.
* **Inductor Triton** (``torch.compile`` output): generated at compile time.
* **CK** (``ck::`` / ``ck_tile::``): template instantiations, no single source.

The gate answers ``False`` (not patchable) or ``None`` ("don't know, go look") --
never ``True``. Confirming a source exists is the resolver's job.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .demangle import demangle
from .datatypes import Patchability

__all__ = ["classify_patchability"]

# CK (Composable Kernel) namespace in a demangled signature. Boundary-anchored to
# the ``ck`` / ``ck_tile`` namespace so unrelated names that merely end in "ck"
# (``block::``, ``unpack::``, ``flashck::``) are NOT misclassified.
_CK_DEMANGLED_RE = re.compile(r"(?:^|[^A-Za-z0-9_])ck(?:_tile)?::")
# Mangled (Itanium) fallback for when demangling is unavailable: the ``ck`` /
# ``ck_tile`` namespace is length-prefixed (e.g. ``...2ck15kernel...``).
_CK_MANGLED_RE = re.compile(r"\d(?:ck_tile|ck)(?=[0-9IE])")

# Inductor-generated Triton kernel names (``torch.compile`` output). These
# prefixes are inductor-specific; the durable source does not exist on disk.
_INDUCTOR_NAME_RE = re.compile(
    r"^triton_(?:poi|red|tem|for|per|mm|bmm|unk|nop)_", re.IGNORECASE
)


def _is_inductor_stack(call_stack: Sequence[str]) -> bool:
    """True if any call-stack frame points into the torch inductor cache."""
    return any("torchinductor" in str(frame).lower() for frame in call_stack)


def classify_patchability(
    kernel_name: str,
    *,
    op_name: str = "",
    call_stack: Sequence[str] = (),
) -> Patchability:
    """Classify a kernel's patchability from cheap, filesystem-free signals.

    Args:
        kernel_name: Device kernel symbol from the trace (mangled or plain).
        op_name: Launching op name (e.g. ``aten::miopen_convolution``).
        call_stack: Optional call-stack frames from the perf report; used to
            detect inductor-generated Triton kernels.

    Returns:
        A :class:`Patchability`. ``patchable is False`` with a ``kind``/``reason``
        for a known precompiled/generated kernel; ``patchable is None`` when the
        gate cannot decide and the caller should proceed to resolution.
    """
    raw = (kernel_name or "").strip()
    op = (op_name or "").lower()

    # Tensile GEMM kernels: Cijk_<A_layout>_<B_layout>_<config...> -- .co assembly.
    if raw.startswith("Cijk_"):
        return Patchability(
            False, "tensile_precompiled", "Tensile precompiled GEMM (.co assembly)"
        )

    # MIOpen convolutions: identified by op name (device kernel names vary).
    if "miopen" in op:
        return Patchability(
            False, "miopen_precompiled", "MIOpen precompiled convolution kernel"
        )

    # Inductor-generated Triton: no durable source to rewrite.
    if _is_inductor_stack(call_stack) or _INDUCTOR_NAME_RE.match(raw):
        return Patchability(
            False,
            "triton_inductor_generated",
            "torch.compile inductor-generated Triton kernel",
        )

    # CK template instantiations: no single editable __global__ source.
    if _is_ck(raw):
        return Patchability(
            False, "aiter_ck", "Composable Kernel template instantiation"
        )

    return Patchability(None)


def _is_ck(raw: str) -> bool:
    """Whether ``raw`` names a CK (Composable Kernel) instantiation."""
    if not raw:
        return False
    if raw.startswith("_Z"):
        demangled = demangle(raw)
        if demangled:
            return bool(_CK_DEMANGLED_RE.search(demangled.lower()))
        # Demangling failed: classify from the mangled namespace prefix instead.
        return bool(_CK_MANGLED_RE.search(raw))
    return bool(_CK_DEMANGLED_RE.search(raw.lower()))
