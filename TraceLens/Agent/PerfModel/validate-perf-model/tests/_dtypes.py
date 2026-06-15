###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared dtype / activation / quant-type resolvers for the test harnesses.

The ``tests/<category>.py`` modules accept user-friendly dtype strings
(``"bf16"``, ``"fp8"``, ``"fp4x2"``, ...) and need to resolve them to
``torch.dtype`` values before allocating tensors. Centralizing the resolvers
avoids divergence between categories.

All helpers defer the ``torch`` import so that the parent process (which
imports the ``tests`` package only to discover function objects and to compute
varlen-attention statistics) does not need ``torch`` installed.
"""

# Strings accepted from CSV / CLI map onto torch dtypes the kernels recognize.
# These are the names ``aiter.utility.dtypes`` exposes plus a couple of common
# torch aliases. ``fp8`` resolves dynamically based on the current GPU arch
# (see ``aiter/utility/dtypes.py:get_dtype_fp8``).
_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
    "f32":  "float32",
    "i8":   "int8",
    "i32":  "int32",
    "u8":   "uint8",
    "u32":  "uint32",
}


def resolve_dtype(name):
    """Resolve a dtype string (``"bf16"``, ``"fp8"``, ``"fp4x2"``, ...) to a
    real ``torch.dtype``.

    Falls back to ``aiter.dtypes`` for arch-specific dtypes (``fp8`` is
    ``float8_e4m3fnuz`` on gfx942 and ``float8_e4m3fn`` on gfx950). Returns
    ``None`` when ``name`` is ``None`` so callers can short-circuit on
    "default" sentinel.
    """
    import torch
    try:
        from aiter import dtypes as _aiter_dtypes
    except Exception:
        _aiter_dtypes = None

    if name is None:
        return None
    if isinstance(name, torch.dtype):
        return name
    key = str(name).lower()

    if _aiter_dtypes is not None and hasattr(_aiter_dtypes, key):
        d = getattr(_aiter_dtypes, key)
        if isinstance(d, torch.dtype):
            return d

    alias = _DTYPE_ALIASES.get(key, key)
    if hasattr(torch, alias):
        d = getattr(torch, alias)
        if isinstance(d, torch.dtype):
            return d
    raise ValueError(f"Unknown dtype string: {name!r}")


def resolve_activation(name):
    """Resolve ``"silu"`` / ``"gelu"`` / ``"swiglu"`` / ``"no"`` to ``aiter.ActivationType``."""
    from aiter import ActivationType

    if name is None:
        return ActivationType.Silu
    if isinstance(name, ActivationType):
        return name
    key = str(name).strip().lower()
    table = {
        "silu":   ActivationType.Silu,
        "gelu":   ActivationType.Gelu,
        "swiglu": ActivationType.Swiglu,
        "no":     ActivationType.No,
        "none":   ActivationType.No,
    }
    if key not in table:
        raise ValueError(f"Unknown activation: {name!r}")
    return table[key]


def resolve_quant_type(name):
    """Resolve ``"no"`` / ``"per_token"`` / ``"per_1x128"`` / ... to ``aiter.QuantType``."""
    from aiter import QuantType

    if name is None:
        return QuantType.No
    if isinstance(name, QuantType):
        return name
    key = str(name).strip().lower()
    table = {
        "no":           QuantType.No,
        "per_tensor":   QuantType.per_Tensor,
        "per_token":    QuantType.per_Token,
        "per_1x32":     QuantType.per_1x32,
        "per_1x128":    QuantType.per_1x128,
        "per_128x128":  QuantType.per_128x128,
        "per_256x128":  QuantType.per_256x128,
        "per_1024x128": QuantType.per_1024x128,
    }
    if key not in table:
        raise ValueError(f"Unknown quant_type: {name!r}")
    return table[key]
