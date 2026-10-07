###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Models that predict an op's time from the parameters TraceLens extracted.

A time model is a function::

    def model(category, params, arch):
        if category != "GEMM":
            return None
        return predicted_time_us

``category`` is the perf-model class's ``category`` (``"GEMM"``,
``"SDPA_fwd"``, ...), ``params`` is a copy of its ``param_details`` (for a
GEMM: M, N, K, B, dtype_A_B, ...), and ``arch`` is the GPU arch dict or
``None``. Return ``None`` for ops the model does not handle.

:class:`TreePerfAnalyzer` reports each registered model as
``<label> Time (µs)``. Origami is the built-in model under ``"Origami"``. An
``--extension_file`` can register one under ``"Specialized"``; a proprietary
library call belongs in that file, not in this repository.
"""

import os
from functools import partial

TIME_MODEL_LABELS = ("Origami", "Specialized")


def time_model_columns(label):
    """``(per-shape, per-instance)`` column names for a time-model label.

    Per-shape columns depend only on the op's params; per-instance columns
    depend on the measured kernel time.
    """
    return (
        [f"{label} Time (µs)", f"{label} TFLOPS/s"],
        [f"{label} TB/s", f"Pct {label}"],
    )


def predict_time(model, perf_model, arch):
    """Call ``model`` for a constructed perf model. Returns µs or None."""
    category = getattr(perf_model, "category", None)
    params = getattr(perf_model, "param_details", None)
    if category is None or params is None:
        return None
    time_us = model(category, dict(params), arch)
    return None if time_us is None else float(time_us)


def origami_perf_model(category, params, arch, python_path=None):
    """GEMM time from Origami, or from ``GEMM_SIMULATOR_PATH`` when it is set."""
    if category != "GEMM" or arch is None:
        return None
    from .perf_model import GEMM
    from .utils import torch_dtype_map

    dtype = params.get("simulation_dtype")
    if dtype is None:
        dtype = torch_dtype_map(params["dtype_A_B"][0])
    time_us, _ = GEMM.get_simulation_time_func(
        arch,
        params["M"],
        params["N"],
        params["K"],
        params.get("B", 1),
        dtype,
        python_path,
        enable_origami=True,
    )
    return time_us


def builtin_origami_model(enable_origami, python_path=None):
    """Origami model to register, or None when neither Origami nor
    ``GEMM_SIMULATOR_PATH`` is enabled."""
    if not enable_origami and "GEMM_SIMULATOR_PATH" not in os.environ:
        return None
    return partial(origami_perf_model, python_path=python_path)
