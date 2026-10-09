###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Op models: per-op values computed from an op's work.

A *perf model* (:mod:`TraceLens.PerfModel.perf_model`) says what work an op
does: its shapes, FLOPs and bytes. An *op model* takes that work and returns
anything to report for the op, usually a predicted time but also values such
as a kernel configuration. :class:`TreePerfAnalyzer` measures an op,
describes its work as an :class:`OpWork`, and runs every registered op model
on it::

    def op_model(work, arch):
        return None or time_us or {"time_us": time_us, "Tile": "256x128x64"}

The output is ``None`` (the model does not handle this op), a time in µs, or
a dict. A dict's ``time_us`` is optional; every other key ``name`` becomes a
``<label> <name>`` column, and only scalar values (numbers, strings, bools)
are written. With a time, the label also gets ``<label> Time (µs)``,
``<label> TFLOPS/s``, ``<label> TB/s`` and ``Pct <label>`` (time as a
percentage of the measured kernel time). Without one, ``<label> Time (µs)``
and ``Pct <label>`` are left empty. A model that raises gets no columns for
that op, with a one-time warning. Op models are called once per op launch,
so a slow model should cache its results.

Roofline is always on; Origami for GEMMs and the SDPA tile model with Origami
are built in behind flags.

External op models use a simpler signature and are adapted with
:func:`external_op_model`::

    def model(category, params, arch):
        if category != "GEMM":
            return None
        return predicted_time_us

``category`` is the perf-model class's ``category`` (``"GEMM"``,
``"SDPA_fwd"``, ...), or its ``bwd_category`` (``"SDPA_bwd"``, ...) for a
backward op. ``params`` is a copy of the perf model's ``param_details``, the
same values as the report's ``param:`` columns; any key may be missing (for
example ``transpose``, which is only set when the kernel name parses).
``arch`` is the GPU arch dict of the report, or ``None``. A proprietary
library call belongs in an ``--extension_file``, not in this repository.
"""

import numbers
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from . import origami_helper
from .utils import add_duration_rate_columns, torch_dtype_map

# Built-in labels come first in summaries, in this order; others follow in
# column order.
BUILTIN_LABEL_ORDER = ("Roofline", "Origami", "SDPA Tile Origami", "External")

TIME_KEY = "time_us"


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


def op_model_columns(label):
    """``(per-shape, per-instance)`` column names for an op-model label.

    Per-shape columns depend only on the op's params; per-instance columns
    depend on the measured kernel time.
    """
    return (
        [f"{label} Time (µs)", f"{label} TFLOPS/s", f"{label} TB/s"],
        [f"Pct {label}"],
    )


def op_model_labels(columns):
    """Labels that have op-model columns in ``columns``, built-ins first."""
    columns = list(columns)
    present = set(columns)
    labels = [
        col[len("Pct ") :]
        for col in columns
        if col.startswith("Pct ") and f"{col[len('Pct '):]} Time (µs)" in present
    ]
    rank = {label: i for i, label in enumerate(BUILTIN_LABEL_ORDER)}
    return sorted(labels, key=lambda label: rank.get(label, len(rank)))


def op_model_group_columns(columns, label, labels):
    """``label``'s columns, in column order, with ``Pct <label>`` included.

    A column under a longer label that extends ``label`` (``"Origami X"`` vs
    ``"Origami"``) belongs to that longer label.
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


def _is_scalar(value):
    return isinstance(value, (str, bool, numbers.Number))


def add_op_model_columns(
    dict_metrics, label, output, gflops, bytes_moved, busy_kernel_time
):
    """Write ``label``'s columns for an op model's ``output``: None, a time in
    µs, or a dict with an optional ``time_us`` and extra scalar columns."""
    if isinstance(output, dict):
        time_us = output.get(TIME_KEY)
        extra_columns = {
            name: value
            for name, value in output.items()
            if name != TIME_KEY and _is_scalar(value)
        }
    else:
        time_us, extra_columns = output, {}
    if not time_us and not extra_columns:
        return
    if time_us:
        dict_metrics[f"{label} Time (µs)"] = time_us
        add_duration_rate_columns(
            dict_metrics, gflops, bytes_moved, time_us, prefix=label
        )
    else:
        dict_metrics[f"{label} Time (µs)"] = float("nan")
    for name, value in extra_columns.items():
        dict_metrics[f"{label} {name}"] = value
    dict_metrics[f"Pct {label}"] = (
        (time_us / busy_kernel_time) * 100
        if time_us and busy_kernel_time > 0
        else float("nan")
    )


_op_model_warnings = set()


def _warn_op_model_failed(label, work, error):
    key = (label, work.category, type(error).__name__)
    if key in _op_model_warnings:
        return
    _op_model_warnings.add(key)
    warnings.warn(
        f"Op model {label!r} failed on a {work.category} op "
        f"({type(error).__name__}: {error}); its columns are left empty for that "
        "op. Shown once per model, category and error type.",
        RuntimeWarning,
    )


def add_op_model_outputs(dict_metrics, op_models, work, arch, busy_kernel_time):
    """Run each ``{label: op_model}`` on ``work`` and write its columns.

    A model that raises gets no columns for this op; the others and the
    measured metrics are unaffected."""
    for label, op_model in op_models.items():
        try:
            output = op_model(work, arch)
        except Exception as error:
            _warn_op_model_failed(label, work, error)
            continue
        add_op_model_columns(
            dict_metrics,
            label,
            output,
            work.gflops,
            work.bytes_moved,
            busy_kernel_time,
        )


def roofline_op_model(work, arch):
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
    return {TIME_KEY: max(compute_time_us, memory_time_us), "Bound": bound}


def external_op_model(model):
    """Adapt ``model(category, params, arch)`` to the op-model interface.

    The model returns None, a time in µs, or a dict (see the module doc). A
    backward op is passed its perf model's ``bwd_category`` (``"SDPA_bwd"``,
    ...) with the forward params; ops without one (GEMM) are skipped."""

    def op_model(work, arch):
        if work.category is None or work.params is None:
            return None
        output = model(work.category, dict(work.params), arch)
        if output is None or isinstance(output, dict):
            return output
        return float(output)

    op_model.external_model = model
    return op_model


def origami_gemm_model(category, params, arch):
    """GEMM time from Origami, as an external-style model."""
    if category != "GEMM" or arch is None:
        return None
    dtype = params.get("simulation_dtype")
    if dtype is None:
        dtype = torch_dtype_map(params["dtype_A_B"][0])
    return origami_helper.gemm_time_us(
        arch, params["M"], params["N"], params["K"], params.get("B", 1), dtype
    )


def sdpa_tile_origami_op_model(work, arch):
    """TraceLens's attention tile model, with Origami timing each tile GEMM
    (see ``SDPA.get_simulation_time_func``). Origami only models GEMMs; the
    tiling is TraceLens's own."""
    simulate = getattr(
        work.perf_model,
        "get_simulation_time_bwd" if work.bwd else "get_simulation_time",
        None,
    )
    if simulate is None:
        return None
    return simulate(gemm_time=origami_helper.gemm_time_us)


def default_op_models(enable_origami_gemm=False, enable_origami_sdpa_tile=False):
    """Built-in op models, in report order: Roofline; Origami for GEMMs with
    ``enable_origami_gemm``; the SDPA tile model with Origami with
    ``enable_origami_sdpa_tile``."""
    op_models = {"Roofline": roofline_op_model}
    if enable_origami_gemm:
        op_models["Origami"] = external_op_model(origami_gemm_model)
    if enable_origami_sdpa_tile:
        op_models["SDPA Tile Origami"] = sdpa_tile_origami_op_model
    return op_models
