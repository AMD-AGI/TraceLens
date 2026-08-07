###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Emit highly visible warnings when perf models substitute trace defaults."""

from __future__ import annotations

import warnings

_BAR = "=" * 72


def warn_perf_model_fallback(model_name: str, detail: str) -> None:
    """
    Warn once per callsite when a perf model uses an assumed default instead of
    values from the trace (graph replay often omits fields).
    """
    msg = (
        f"\n{_BAR}\n"
        f"WARNING: TraceLens perf model {model_name!r} used a FALLBACK DEFAULT.\n"
        f"Reported roofline FLOPs / bytes may be wrong.\n"
        f"\n{detail.rstrip()}\n"
        f"{_BAR}\n"
    )
    warnings.warn(msg, UserWarning, stacklevel=2)
