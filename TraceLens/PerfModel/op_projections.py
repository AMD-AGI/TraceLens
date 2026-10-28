###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Op projections: per-op predictions computed from an op's work.

A *perf model* (:mod:`TraceLens.PerfModel.perf_model`) says what work an op
does: its shapes, FLOPs and bytes. An *op projection* takes that work and
returns what to report for the op, usually a predicted time but also values
such as a kernel configuration. :class:`TreePerfAnalyzer` measures an op,
describes its work as an :class:`OpWork`, and runs every registered op
projection on it::

    def op_projection(work, arch):
        if work.category != "GEMM":
            return None
        return {"time_us": time_us, "Tile": "256x128x64"}

``work.category`` is the perf-model class's ``category`` (``"GEMM"``,
``"SDPA_fwd"``, ...), or its ``bwd_category`` (``"SDPA_bwd"``, ...) for a
backward op, which is ``None`` for ops without one (GEMM). ``work.params`` is
a copy of the perf model's ``param_details``, the same values as the
report's ``param:`` columns; any key may be missing (for example
``transpose``, which is only set when the kernel name parses). ``arch`` is
the GPU arch dict of the report, or ``None``.

The output is ``None`` (no projection for this op) or a dict. ``time_us`` is
optional; every other key ``name`` becomes a ``<label> <name>`` column, and
only scalar values (numbers, strings, bools) are written. With a time, the
label also gets ``<label> Time (µs)``, ``<label> TFLOPS/s``, ``<label> TB/s``
and ``Pct <label>`` (time as a percentage of the measured kernel time).
Without one, ``<label> Time (µs)`` and ``Pct <label>`` are left empty. An op
projection that raises or returns anything else gets no columns for that
op, with a one-time warning. Op projections are called once per op launch,
so a slow one should cache its results.

Roofline is always on; Origami for GEMMs and the SDPA tile model with Origami
are built in behind flags. A proprietary library call belongs in an
``--extension_file``, not in this repository.
"""

import numbers
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from . import origami_helper, sdpa_tile
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
    params = getattr(perf_model, "param_details", None)
    return OpWork(
        category=getattr(perf_model, "bwd_category" if bwd else "category", None),
        params=dict(params) if params is not None else None,
        gflops=gflops,
        bytes_moved=bytes_moved,
        compute_spec=get_compute_spec(perf_model),
        bwd=bwd,
        perf_model=perf_model,
    )


def op_projection_columns(label):
    """``(per-shape, per-instance)`` column names for an op-projection label.

    Per-shape columns depend only on the op's params; per-instance columns
    depend on the measured kernel time.
    """
    return (
        [f"{label} Time (µs)", f"{label} TFLOPS/s", f"{label} TB/s"],
        [f"Pct {label}"],
    )


def op_projection_labels(columns):
    """Labels that have op-projection columns in ``columns``, built-ins first."""
    columns = list(columns)
    present = set(columns)
    labels = [
        col[len("Pct ") :]
        for col in columns
        if col.startswith("Pct ") and f"{col[len('Pct '):]} Time (µs)" in present
    ]
    rank = {label: i for i, label in enumerate(BUILTIN_LABEL_ORDER)}
    return sorted(labels, key=lambda label: rank.get(label, len(rank)))


def op_projection_group_columns(columns, label, labels):
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


def add_op_projection_columns(
    dict_metrics, label, output, gflops, bytes_moved, busy_kernel_time
):
    """Write ``label``'s columns for an op projection's ``output``: a dict with
    an optional ``time_us`` and extra scalar columns."""
    time_us = output.get(TIME_KEY)
    extra_columns = {
        name: value
        for name, value in output.items()
        if name != TIME_KEY and _is_scalar(value)
    }
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


_op_projection_warnings = set()


def _warn_op_projection_failed(label, work, error):
    key = (label, work.category, type(error).__name__)
    if key in _op_projection_warnings:
        return
    _op_projection_warnings.add(key)
    warnings.warn(
        f"Op projection {label!r} failed on a {work.category} op "
        f"({type(error).__name__}: {error}); its columns are left empty for that "
        "op. Shown once per op projection, category and error type.",
        RuntimeWarning,
    )


def add_op_projection_outputs(
    dict_metrics, op_projections, work, arch, busy_kernel_time
):
    """Run each ``{label: op_projection}`` on ``work`` and write its columns.

    An op projection that raises or returns something other than a dict or
    None gets no columns for this op; the others and the measured metrics are
    unaffected."""
    for label, op_projection in op_projections.items():
        try:
            output = op_projection(work, arch)
            if output is not None and not isinstance(output, dict):
                raise TypeError(
                    f"returned {type(output).__name__}, expected a dict or None"
                )
        except Exception as error:
            _warn_op_projection_failed(label, work, error)
            continue
        if output is None:
            continue
        add_op_projection_columns(
            dict_metrics,
            label,
            output,
            work.gflops,
            work.bytes_moved,
            busy_kernel_time,
        )


def roofline_op_projection(work, arch):
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


def _time_output(time_us):
    return None if time_us is None else {TIME_KEY: time_us}


def origami_gemm_op_projection(work, arch):
    """GEMM time from Origami."""
    if work.category != "GEMM" or arch is None:
        return None
    params = work.params
    dtype = params.get("simulation_dtype")
    if dtype is None:
        dtype = torch_dtype_map(params["dtype_A_B"][0])
    return _time_output(
        origami_helper.gemm_time_us(
            arch, params["M"], params["N"], params["K"], params.get("B", 1), dtype
        )
    )


def sdpa_tile_origami_op_projection(work, arch):
    """TraceLens's attention tile model (``sdpa_tile.py``), with Origami
    timing each tile GEMM. Origami only models GEMMs; the tiling is
    TraceLens's own."""
    return _time_output(
        sdpa_tile.sdpa_tile_time_us(
            work.perf_model, arch, origami_helper.gemm_time_us, bwd=work.bwd
        )
    )


def default_op_projections(enable_origami_gemm=False, enable_origami_sdpa_tile=False):
    """Built-in op projections, in report order: Roofline; Origami for GEMMs with
    ``enable_origami_gemm``; the SDPA tile model with Origami with
    ``enable_origami_sdpa_tile``."""
    op_projections = {"Roofline": roofline_op_projection}
    if enable_origami_gemm:
        op_projections["Origami"] = origami_gemm_op_projection
    if enable_origami_sdpa_tile:
        op_projections["SDPA Tile Origami"] = sdpa_tile_origami_op_projection
    return op_projections
