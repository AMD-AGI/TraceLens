###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Which GPU architectures the Origami GEMM model covers.

Kept free of the ``origami`` import so callers can refuse an unsupported
architecture before loading the package.
"""

import warnings

# The Origami Python bindings model the MI300 and MI350 series only.
ORIGAMI_GFX_ARCHS = frozenset({"gfx942", "gfx950"})

_NAME_TO_GFX = {
    "mi300": "gfx942",
    "mi325": "gfx942",
    "mi350": "gfx950",
    "mi355": "gfx950",
}


class OrigamiUnavailableError(ValueError):
    """The architecture spec names a GPU that Origami does not model."""


def origami_gfx_arch(arch: dict) -> str:
    """
    The gfx target Origami should model for *arch*. The spec's ``gfx_arch``
    wins over its ``name``. Raises :class:`OrigamiUnavailableError` when
    Origami doesn't model that target.
    """
    gfx = arch.get("gfx_arch")
    if gfx is None:
        name = arch.get("name", None)
        if name is None:
            warnings.warn("No architecture name provided; Assuming MI300x")
            name = "mi300x"
        name = name.lower()
        if name.endswith("x"):
            name = name[:-1]
        gfx = _NAME_TO_GFX.get(name, name)
    if gfx not in ORIGAMI_GFX_ARCHS:
        raise OrigamiUnavailableError(
            f"Origami is unavailable on {gfx}; it models "
            f"{', '.join(sorted(ORIGAMI_GFX_ARCHS))} only"
        )
    return gfx
