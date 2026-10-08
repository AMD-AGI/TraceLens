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
``<label> TFLOPS/s``, ``<label> TB/s``, any extra columns (``{"Bound": ...}``
becomes ``<label> Bound``), and ``Pct <label>``. An estimator that raises
gets no columns for that op, with a one-time warning. Roofline, Origami,
GEMM Simulator (when ``GEMM_SIMULATOR_PATH`` is set), and the SDPA tile model
are built in.

External time models use a simpler signature and are adapted with
:func:`external_time_model`::

    def model(category, params, arch):
        if category != "GEMM":
            return None
        return predicted_time_us

``category`` is the perf-model class's ``category`` (``"GEMM"``,
``"SDPA_fwd"``, ...), or its ``bwd_category`` (``"SDPA_bwd"``, ...) for a
backward op, ``params`` is a copy of its ``param_details`` (for a
GEMM: M, N, K, B, dtype_A_B, ...), and ``arch`` is the GPU arch dict or
``None``. Return ``None`` for ops the model does not handle, or a
``TimeEstimate`` to add extra columns. A proprietary
library call belongs in an ``--extension_file``, not in this repository.
"""

import inspect
import os
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

from .utils import add_duration_rate_columns

# Built-in labels come first in summaries, in this order; others follow in
# column order.
BUILTIN_LABEL_ORDER = (
    "Roofline",
    "Origami",
    "GEMM Simulator",
    "SDPA Tile (Origami)",
    "SDPA Tile (GEMM Simulator)",
    "External",
)


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


def get_compute_spec(perf_model):
    """Compute spec (maf type + precision) of a perf model, such as
    ``"matrix_fp16"`` or ``"vector_bf16"``, or None if not available."""
    maf_type = (
        perf_model.get_maf_type() if hasattr(perf_model, "get_maf_type") else None
    )
    precision = (
        perf_model.get_compute_precision()
        if hasattr(perf_model, "get_compute_precision")
        else None
    )
    if maf_type is None or precision is None:
        return None
    return f"{maf_type}_{precision}"


def op_work(perf_model, bwd=False):
    """The :class:`OpWork` of a constructed perf model, forward or backward."""
    gflops = (perf_model.flops() if not bwd else perf_model.flops_bwd()) / 1e9
    bytes_moved = perf_model.bytes() if not bwd else perf_model.bytes_bwd()
    return OpWork(
        category=getattr(perf_model, "bwd_category" if bwd else "category", None),
        params=getattr(perf_model, "param_details", None),
        gflops=gflops,
        bytes_moved=bytes_moved,
        compute_spec=get_compute_spec(perf_model),
        bwd=bwd,
        perf_model=perf_model,
    )


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
    """Write ``label``'s column group for ``estimate``; skip a missing estimate.

    An extra column ``name`` is written as ``<label> <name>``."""
    if isinstance(estimate, TimeEstimate):
        time_us, extra_columns = estimate.time_us, estimate.extra_columns
    else:
        time_us, extra_columns = estimate, {}
    if not time_us:
        return
    dict_metrics[f"{label} Time (µs)"] = time_us
    add_duration_rate_columns(dict_metrics, gflops, bytes_moved, time_us, prefix=label)
    for name, value in extra_columns.items():
        dict_metrics[f"{label} {name}"] = value
    dict_metrics[f"Pct {label}"] = (
        (time_us / busy_kernel_time) * 100 if busy_kernel_time > 0 else float("nan")
    )


_estimator_warnings = set()


def _warn_estimator_failed(label, work, error):
    key = (label, work.category, type(error).__name__)
    if key in _estimator_warnings:
        return
    _estimator_warnings.add(key)
    warnings.warn(
        f"Time model {label!r} failed on a {work.category} op "
        f"({type(error).__name__}: {error}); its columns are left empty for that "
        "op. Shown once per model, category and error type.",
        RuntimeWarning,
    )


def add_time_estimates(dict_metrics, estimators, work, arch, busy_kernel_time):
    """Run each ``{label: estimator}`` on ``work`` and write its column group.

    An estimator that raises gets no columns for this op; the others and the
    measured metrics are unaffected."""
    for label, estimator in estimators.items():
        try:
            estimate = estimator(work, arch)
        except Exception as error:
            _warn_estimator_failed(label, work, error)
            continue
        add_time_estimate_columns(
            dict_metrics,
            label,
            estimate,
            work.gflops,
            work.bytes_moved,
            busy_kernel_time,
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
    return TimeEstimate(max(compute_time_us, memory_time_us), {"Bound": bound})


def external_time_model(model):
    """Adapt ``model(category, params, arch)`` to the estimator interface.

    The model returns a time in µs, a :class:`TimeEstimate` to add extra
    columns, or None. A backward op is passed its perf model's
    ``bwd_category`` (``"SDPA_bwd"``, ...) with the forward params; ops
    without one (GEMM) are skipped."""

    def estimate(work, arch):
        if work.category is None or work.params is None:
            return None
        result = model(work.category, dict(work.params), arch)
        if result is None or isinstance(result, TimeEstimate):
            return result
        return float(result)

    estimate.time_model = model
    return estimate


def gemm_time(category, params, arch, python_path=None, backend=None):
    """GEMM time from ``backend`` (see ``GEMM.get_simulation_time_func``)."""
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
        backend=backend,
    )
    return time_us


def origami_perf_model(category, params, arch, python_path=None):
    """GEMM time from Origami."""
    return gemm_time(category, params, arch, python_path, backend="origami")


def gemm_simulator_model(category, params, arch, python_path=None):
    """GEMM time from the simulator script at ``GEMM_SIMULATOR_PATH``."""
    return gemm_time(category, params, arch, python_path, backend="simulator")


def _accepts_backend(fn):
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.name == "backend" or p.kind is inspect.Parameter.VAR_KEYWORD
        for p in parameters
    )


def origami_estimator(python_path=None):
    """Origami time for forward GEMMs."""
    return external_time_model(partial(origami_perf_model, python_path=python_path))


def gemm_simulator_estimator(python_path=None):
    """GEMM simulator time for forward GEMMs."""
    return external_time_model(partial(gemm_simulator_model, python_path=python_path))


SDPA_TILE_BACKENDS = {"origami": "Origami", "simulator": "GEMM Simulator"}


def sdpa_tile_label(backend):
    """Report label of the SDPA tile model on ``backend``."""
    return f"SDPA Tile ({SDPA_TILE_BACKENDS[backend]})"


def class_simulation_time(perf_model, bwd, backend):
    """Time from the perf-model class's own simulation on ``backend``. A class
    whose simulation takes no ``backend`` counts as Origami."""
    simulate = getattr(
        perf_model, "get_simulation_time_bwd" if bwd else "get_simulation_time", None
    )
    if simulate is None:
        return None
    if _accepts_backend(simulate):
        return simulate(backend=backend)
    return simulate() if backend == "origami" else None


def sdpa_tile_estimator(backend):
    """TraceLens's attention tile model: one Q·Kᵀ and one P·V tile GEMM timed
    on one CU with ``backend``, scaled by the number of waves, plus softmax and
    memory terms. Origami and the GEMM simulator only model GEMMs; the tiling
    is TraceLens's own."""
    if backend not in SDPA_TILE_BACKENDS:
        raise ValueError(
            f"Unknown SDPA tile model backend {backend!r}; "
            f"expected one of {sorted(SDPA_TILE_BACKENDS)}"
        )

    def estimate(work, arch):
        return class_simulation_time(work.perf_model, work.bwd, backend)

    return estimate


def default_time_estimators(
    enable_origami=False, python_path=None, sdpa_tile_model=None
):
    """Built-in estimators, in report order: Roofline; Origami with
    ``enable_origami``; the GEMM simulator when ``GEMM_SIMULATOR_PATH`` is set;
    the SDPA tile model on ``sdpa_tile_model`` (``"origami"`` or
    ``"simulator"``) when given."""
    estimators = {"Roofline": roofline_estimator}
    if enable_origami:
        estimators["Origami"] = origami_estimator(python_path)
    if "GEMM_SIMULATOR_PATH" in os.environ:
        estimators["GEMM Simulator"] = gemm_simulator_estimator(python_path)
    if sdpa_tile_model is not None:
        if sdpa_tile_model == "simulator" and "GEMM_SIMULATOR_PATH" not in os.environ:
            raise ValueError("sdpa_tile_model='simulator' needs GEMM_SIMULATOR_PATH")
        estimators[sdpa_tile_label(sdpa_tile_model)] = sdpa_tile_estimator(
            sdpa_tile_model
        )
    return estimators
