###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ideal-time estimators for perf reports.

:class:`TreePerfAnalyzer` measures an op, describes its work as an
:class:`OpWork`, and runs every registered estimator on that work. An
estimator is a function::

    def estimator(work, arch):
        return TimeEstimate(time_us, extra_columns) or time_us or None

Each estimator writes one column group, ``<label> Time (µs)``,
``<label> TFLOPS/s``, ``<label> TB/s``, any extra columns (named
``<label> ...``), and ``Pct <label>``. Roofline and Origami are built in.

External time models use a simpler signature and are adapted with
:func:`external_time_model`::

    def model(category, params, arch):
        if category != "GEMM":
            return None
        return predicted_time_us

``category`` is the perf-model class's ``category`` (``"GEMM"``,
``"SDPA_fwd"``, ...), ``params`` is a copy of its ``param_details`` (for a
GEMM: M, N, K, B, dtype_A_B, ...), and ``arch`` is the GPU arch dict or
``None``. Return ``None`` for ops the model does not handle. A proprietary
library call belongs in an ``--extension_file``, not in this repository.
"""

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

from .utils import add_duration_rate_columns

# Built-in labels come first in summaries, in this order; others follow in
# column order.
BUILTIN_LABEL_ORDER = ("Roofline", "Origami", "Specialized")


@dataclass(frozen=True)
class OpWork:
    """What an op does, independent of how long it took."""

    category: Optional[str]
    params: Optional[dict]
    gflops: float
    bytes_moved: Optional[float]
    compute_spec: Optional[str]
    bwd: bool = False
    perf_model: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class TimeEstimate:
    """An ideal time plus extra report columns, named ``<label> ...``."""

    time_us: float
    extra_columns: dict = field(default_factory=dict)


def time_model_columns(label):
    """``(per-shape, per-instance)`` column names for a time-model label.

    Per-shape columns depend only on the op's params; per-instance columns
    depend on the measured kernel time.
    """
    return (
        [f"{label} Time (µs)", f"{label} TFLOPS/s", f"{label} TB/s"],
        [f"Pct {label}"],
    )


def time_estimate_labels(columns):
    """Labels that have estimate columns in ``columns``, built-ins first."""
    columns = list(columns)
    present = set(columns)
    labels = [
        col[len("Pct ") :]
        for col in columns
        if col.startswith("Pct ") and f"{col[len('Pct '):]} Time (µs)" in present
    ]
    rank = {label: i for i, label in enumerate(BUILTIN_LABEL_ORDER)}
    return sorted(labels, key=lambda label: rank.get(label, len(rank)))


def time_estimate_group_columns(columns, label, labels):
    """``label``'s columns, in column order, with ``Pct <label>`` included.

    A column under a longer label that extends ``label`` (``"GEMM Sim"`` vs
    ``"GEMM"``) belongs to that longer label.
    """
    longer = [other for other in labels if other.startswith(f"{label} ")]
    return [
        col
        for col in columns
        if col == f"Pct {label}"
        or (
            col.startswith(f"{label} ")
            and not any(col.startswith(f"{other} ") for other in longer)
        )
    ]


def add_time_estimate_columns(
    dict_metrics, label, estimate, gflops, bytes_moved, busy_kernel_time
):
    """Write ``label``'s column group for ``estimate``; skip a missing estimate."""
    if isinstance(estimate, TimeEstimate):
        time_us, extra_columns = estimate.time_us, estimate.extra_columns
    else:
        time_us, extra_columns = estimate, {}
    if not time_us:
        return
    dict_metrics[f"{label} Time (µs)"] = time_us
    add_duration_rate_columns(dict_metrics, gflops, bytes_moved, time_us, prefix=label)
    dict_metrics.update(extra_columns)
    dict_metrics[f"Pct {label}"] = (
        (time_us / busy_kernel_time) * 100 if busy_kernel_time > 0 else float("nan")
    )


def roofline_estimator(work, arch):
    """Roofline time: the larger of peak-compute time and peak-bandwidth time."""
    if arch is None or work.compute_spec is None:
        return None
    maf_specs = arch.get("max_achievable_tflops")
    peak_tflops = maf_specs.get(work.compute_spec) if maf_specs is not None else None
    mem_bw_gbps = arch.get("mem_bw_gbps")
    if (
        peak_tflops is None
        or mem_bw_gbps is None
        or work.bytes_moved is None
        or not work.gflops > 0
    ):
        return None
    # flops / (peak_tflops * 1e12) gives seconds, convert to µs
    compute_time_us = (work.gflops * 1e9 / (peak_tflops * 1e12)) * 1e6
    # bytes / (bandwidth_gbps * 1e9) gives seconds, convert to µs
    memory_time_us = (work.bytes_moved / (mem_bw_gbps * 1e9)) * 1e6
    bound = "COMPUTE_BOUND" if compute_time_us >= memory_time_us else "MEMORY_BOUND"
    return TimeEstimate(max(compute_time_us, memory_time_us), {"Roofline Bound": bound})


def predict_time(model, perf_model, arch):
    """Call ``model`` for a constructed perf model. Returns µs or None."""
    category = getattr(perf_model, "category", None)
    params = getattr(perf_model, "param_details", None)
    if category is None or params is None:
        return None
    time_us = model(category, dict(params), arch)
    return None if time_us is None else float(time_us)


def external_time_model(model):
    """Adapt ``model(category, params, arch)`` to the estimator interface."""

    def estimate(work, arch):
        if work.bwd or work.category is None or work.params is None:
            return None
        time_us = model(work.category, dict(work.params), arch)
        return None if time_us is None else float(time_us)

    estimate.time_model = model
    return estimate


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


def origami_estimator(enable_origami=False, python_path=None):
    """Origami time: the perf-model class's own simulation (attention's tile
    model), and for GEMMs Origami on the GEMM's params when enabled."""
    gemm_model = builtin_origami_model(enable_origami, python_path)

    def estimate(work, arch):
        simulate = getattr(
            work.perf_model,
            "get_simulation_time_bwd" if work.bwd else "get_simulation_time",
            None,
        )
        time_us = simulate() if simulate is not None else None
        if gemm_model is not None and not work.bwd:
            gemm_time_us = predict_time(gemm_model, work.perf_model, arch)
            if gemm_time_us:
                time_us = gemm_time_us
        return time_us

    return estimate


def default_time_estimators(enable_origami=False, python_path=None):
    """Built-in estimators, in report order."""
    return {
        "Roofline": roofline_estimator,
        "Origami": origami_estimator(enable_origami, python_path),
    }
