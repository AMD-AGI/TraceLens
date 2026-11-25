###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""End-to-end roofline plot from a profile and, where the profile has no cost, a projection.

FLOPs and bytes come from the perf report's ``GFLOPS`` and ``FLOPS/Byte``
columns: the PerfModel applied to the traced shapes. The marker divides that
work by the measured kernel time. ``--model-only`` reads no profile and
divides the projection's FLOPs by its roofline time. Hardware roofs come from
the GPU arch spec (``load_arch``, including ``$TL_EXTENSION``).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator

from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms, load_arch
from TraceLens.PerfModel.torch_op_mapping import (
    OP_CATEGORY_PATTERNS,
    OP_CATEGORY_REGISTRY,
)

from TraceLens.Visualizer.model_explorer_export.roofline_augment import (
    categorize_operation,
)

# Constants

# A kernel name that carries one of these belongs to that phase. A name that
# carries neither is unscoped: a mixed capture does not say which phase it ran.
_PREFILL_MARKERS = ("prefill",)
_DECODE_MARKERS = ("qseqlen1", "decode")
_COMM_MARKERS = (
    "allreduce",
    "all_reduce",
    "reduce_scatter",
    "all_gather",
    "alltoall",
    "all_to_all",
    "cross_device_reduce",
    "nccl",
    "rccl",
)
# Compute roofs the plot draws when the arch lists them, lowest peak first.
# The linestyle is also used for the part of the bandwidth diagonal that only
# that peak still governs.
_HARDWARE_ROOFS = (
    ("matrix_bf16", "bfloat16", "-", "0.25"),
    ("matrix_fp8", "fp8", "-.", "0.35"),
    ("matrix_fp4", "fp4", "--", "0.45"),
)
# Each compute roof is also drawn at these fractions: the bandwidth diagonal
# and the horizontal, both scaled, so the ridge stays at the same intensity.
# Each fraction has a gray between the full roofs (darkest, 0.45 and below)
# and the paler gridlines, so each lower fraction is paler than the one above.
_PEAK_FRACTIONS = ((0.75, "0.62"), (0.5, "0.86"), (0.25, "0.92"), (0.1, "0.96"))
_CONCURRENCY_COLORS = ("#4C78A8", "#F58518", "#111111", "#54A24B", "#E45756")
# The note under the legend names unmodeled kernels only when they are more
# than this share of kernel time, both together and one by one.
_UNMODELED_LIST_SHARE = 0.01
_UNMODELED_LIST_LIMIT = 10
_PROFILE_YLABEL = "TFLOP/s per GPU while the operation runs"
_PHASE_PREFIX = re.compile(r"^(?:prefill|decode)/", re.IGNORECASE)
_KERNEL_SUFFIX = re.compile(r"\.kd$", re.IGNORECASE)


@dataclass(frozen=True)
class Series:
    """One marker family. ``categories`` None is the whole modeled forward.

    ``kernel_name`` is one profile kernel. The marker is that kernel's own
    FLOPs and time, not the category it belongs to. ``kernel_shape`` is the
    report's ``Input Dims``; a top-kernel marker is one name at one shape.
    """

    label: str
    marker: str | tuple[int, int, int]
    phase: str | None
    categories: tuple[str, ...] | None
    open_marker: bool = False
    kernel_name: str | None = None
    kernel_shape: str | None = None


# The perf report's catch-all. It is not a series: a kernel that landed here
# has no category to draw.
_UNCLASSIFIED = frozenset({"unknown", "other", "others"})


def _extra_markers() -> tuple[tuple[int, int, int], ...]:
    """Stars Matplotlib does not already name.

    A ``(sides, style, angle)`` marker is filled, so a hollow prefill point can
    still be drawn. Point counts and rotations are far enough apart that two
    stars do not read as the same glyph. Three-point stars are omitted: they
    read as triangles.
    """
    return tuple((sides, 1, angle) for sides in (5, 7, 9, 11) for angle in (45, 60))


# One shape per report category, in ``GEMM`` then alphabetical order. The
# circle, square, plus, and diamond are reserved for attention, routed MoE,
# whole model, and end to end. The categories this profile draws land on
# ``^``, ``<``, ``X``, ``*``, and ``v``, which stay distinct at legend size.
_CATEGORY_MARKERS = (
    "^",  # GEMM
    (4, 1, 0),
    "<",  # CONV_fwd. A six-point star reads as the RMSNorm star.
    (8, 1, 0),
    (10, 1, 0),
    (4, 1, 15),
    (6, 1, 15),
    (8, 1, 15),
    "X",  # GroupQuant
    (10, 1, 15),
    (4, 1, 30),
    (6, 1, 30),
    (8, 1, 30),
    (10, 1, 30),
    (6, 1, 0),
    ">",
    (6, 1, 8),
    (8, 1, 8),
    (10, 1, 8),
    "*",  # RMSNorm
    (4, 1, 8),
    (6, 1, 22),
    (8, 1, 22),
    (10, 1, 22),
    (4, 1, 22),
    (6, 1, 38),
    "v",  # elementwise
    *_extra_markers(),
)


def report_categories() -> tuple[str, ...]:
    """Perf-report op categories, excluding the unclassified bucket.

    The names are the ones ``categorize_torch_op`` writes into a perf report.
    ``unknown`` / ``other`` is the bucket a kernel falls into when no rule
    matches, and it is not a series.
    """
    names = set(OP_CATEGORY_REGISTRY.values())
    names.update(category for _pattern, category in OP_CATEGORY_PATTERNS)
    names.update({"elementwise", "reduce", "multi_tensor_apply"})
    return tuple(sorted(name for name in names if name.lower() not in _UNCLASSIFIED))


def _category_series(names: tuple[str, ...]) -> tuple[Series, ...]:
    """One series per perf-report category. GEMM leads; the rest follow."""
    ordered = [name for name in names if name == "GEMM"]
    ordered.extend(name for name in names if name != "GEMM")
    if len(ordered) > len(_CATEGORY_MARKERS):
        raise RuntimeError(
            f"{len(ordered)} report categories and "
            f"{len(_CATEGORY_MARKERS)} distinct markers"
        )
    return tuple(
        Series(name, _CATEGORY_MARKERS[index], None, (name,))
        for index, name in enumerate(ordered)
    )


DEFAULT_SERIES: tuple[Series, ...] = (
    Series("Prefill attention", "o", "prefill", ("Attention",), True),
    Series("Decode attention", "o", "decode", ("Attention",), False),
    Series("Routed MoE", "s", None, ("MoE",), False),
    *_category_series(report_categories()),
    Series("Whole model", "P", None, None, False),
    Series("End to end", "D", None, None, False),
)

# ``--top-kernels`` with no count draws this many kernels, the ones with the
# most kernel time, plus the end-to-end forward. The category series stay the
# default. ``--model-only`` has no kernel times.
_DEFAULT_TOP_KERNELS = 10
# One glyph per kernel. Closed shapes a hollow marker can still show, in an
# order where neighbors do not read as the same symbol. The end-to-end diamond
# is not in this list.
_KERNEL_MARKERS = (
    "o",
    "s",
    "^",
    "v",
    "<",
    ">",
    "p",
    "h",
    "8",
    "*",
    "X",
    "P",
    (7, 0, 0),
    (9, 0, 0),
    (11, 0, 0),
    "1",
    "2",
    "3",
    "4",
    (7, 1, 0),
    (9, 1, 0),
    (11, 1, 0),
)


def _shape_tensors(shape: str) -> list[str]:
    """Each tensor in a report ``Input Dims`` cell, as ``M×N×K``."""
    tensors = []
    for group in re.findall(r"\(([^()]*)\)", shape):
        dims = [part.strip() for part in group.split(",") if part.strip()]
        if dims and all(part.isdigit() for part in dims):
            tensors.append("×".join(dims))
    return tensors


def _top_kernel_label(
    name: str, short: str, shape: str, peers: list[tuple[str, str]]
) -> str:
    """Device-kernel name, with enough of the shape to tell siblings apart."""
    same = [peer_shape for peer_short, peer_shape in peers if peer_short == short]
    if len(same) < 2:
        return short
    tensors = _shape_tensors(shape)
    if not tensors:
        return f"{short} {shape}" if shape else name
    for count in range(1, len(tensors) + 1):
        bit = ", ".join(tensors[:count])
        peer_bits = [", ".join(_shape_tensors(peer)[:count]) for peer in same]
        if peer_bits.count(bit) == 1:
            return f"{short} {bit}"
    return f"{short} {shape}" if shape else name


def top_kernel_series(
    kernels: list[dict[str, Any]], n: int = _DEFAULT_TOP_KERNELS
) -> tuple[Series, ...]:
    """The ``n`` kernels with the most runtime, then the end-to-end forward.

    Rank is the summed ``time_us`` of each kernel name at one ``Input Dims``.
    The same name at another shape is a separate marker. Communication kernels
    are left out, and so is a shape with no profile FLOPs: ``n`` is a count of
    markers that can be drawn. The label is the device kernel, and a name that
    peels to the same text as another keeps the full profile name, plus the
    shape when the name itself is shared.
    """
    if n < 1:
        raise ValueError("top kernel count must be at least 1")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for kernel in kernels:
        name = str(kernel["name"])
        if is_communication(name):
            continue
        key = (name, str(kernel.get("shape") or ""))
        groups.setdefault(key, []).append(kernel)
    totals: dict[tuple[str, str], float] = {}
    for key, rows in groups.items():
        if not rows or not all(_has_profile_cost(row) for row in rows):
            continue
        time_us = sum(float(row["time_us"]) for row in rows)
        if time_us > 0:
            totals[key] = time_us
    ranked = sorted(
        totals,
        key=lambda key: (-totals[key], key[0], key[1]),
    )
    chosen = ranked[:n]
    if len(chosen) > len(_KERNEL_MARKERS):
        raise ValueError(
            f"{len(chosen)} kernels and {len(_KERNEL_MARKERS)} distinct markers"
        )
    peers = [(_kernel_computation(name), shape) for name, shape in chosen]
    specs = []
    for index, (name, shape) in enumerate(chosen):
        short = peers[index][0]
        specs.append(
            Series(
                _top_kernel_label(name, short, shape, peers),
                _KERNEL_MARKERS[index],
                None,
                None,
                kernel_name=name,
                kernel_shape=shape,
            )
        )
    specs.append(Series("End to end", "D", None, None))
    return tuple(specs)


@dataclass(frozen=True)
class Regime:
    """One projected regime. ``ops`` are projection rows or category rollups."""

    phase: str
    concurrency: int
    ops: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RooflinePoint:
    series: str
    phase: str
    concurrency: int
    arithmetic_intensity: float
    achieved_tflops: float
    flops: float
    bytes: float
    time_us: float
    # ``profile`` when every kernel of the series carried GFLOPS and FLOPS/Byte,
    # ``export`` when none did, ``mixed`` when the series spans both, ``model``
    # when the time is the projection's roofline time and no profile was read.
    cost_source: str = "export"
    # Set when the point sits above the roof it can reach. No kernel can run
    # faster than the roof, so the projection's FLOPs or bytes, or the kernels
    # matched to it, do not describe what ran.
    above_roof: bool = False


def bandwidth_tb_s(arch: dict[str, Any]) -> float:
    """Arch memory bandwidth in TB/s. ``mem_bw_gbps`` is GB/s."""
    return float(arch["mem_bw_gbps"]) / 1000.0


def ridge_flops_per_byte(peak_tflops: float, bandwidth: float) -> float:
    """Arithmetic intensity where the bandwidth roof meets ``peak_tflops``."""
    return peak_tflops / bandwidth


def blend_peak_tflops(
    flops_by_spec: dict[str, float], peaks: dict[str, float]
) -> tuple[float, dict[str, float]] | None:
    """FLOP-weighted harmonic mean of the peaks that the mix actually uses.

    Returns ``(peak_tflops, shares)``. Specs with no published peak are left
    out of the mix. ``None`` when nothing in the mix has a peak.
    """
    used = {
        spec: flops
        for spec, flops in flops_by_spec.items()
        if flops > 0 and peaks.get(spec)
    }
    total = sum(used.values())
    if total <= 0:
        return None
    shares = {spec: flops / total for spec, flops in used.items()}
    peak = 1.0 / sum(share / peaks[spec] for spec, share in shares.items())
    return peak, shares


def kernel_phase(name: str) -> str | None:
    """``prefill`` or ``decode`` when the kernel name says so, else None."""
    lowered = name.lower()
    prefill = any(token in lowered for token in _PREFILL_MARKERS)
    decode = any(token in lowered for token in _DECODE_MARKERS)
    if prefill and not decode:
        return "prefill"
    if decode and not prefill:
        return "decode"
    return None


def is_communication(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _COMM_MARKERS)


def _time_column(fieldnames: Iterable[str] | None) -> str | None:
    for name in fieldnames or ():
        if name.startswith("Kernel Time") and name.endswith("_sum"):
            return name
    return None


def _profile_cost(
    gflops: Any, intensity: Any, count: int
) -> tuple[float, float] | None:
    """Total FLOPs and bytes from one perf-report row.

    ``GFLOPS`` is one launch and ``FLOPS/Byte`` is that launch's arithmetic
    intensity, so the row's totals are ``GFLOPS * 1e9 * count`` and that
    divided by the intensity. Empty or non-positive cells mean the trace had
    no shape for this kernel.
    """
    try:
        gflops_f = float(gflops)
        intensity_f = float(intensity)
    except (TypeError, ValueError):
        return None
    if gflops_f <= 0 or intensity_f <= 0 or count <= 0:
        return None
    flops = gflops_f * 1e9 * count
    return flops, flops / intensity_f


def load_profile_kernels(directory: Path) -> list[dict[str, Any]]:
    """Kernel name, summed GPU time, launch count, and profile FLOPs and bytes.

    FLOPs and bytes are set only when the row's ``GFLOPS`` and ``FLOPS/Byte``
    are filled. A GPU-graph launch leaves them empty.
    """
    kernels: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*_ops.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            column = _time_column(reader.fieldnames)
            if column is None or "name" not in (reader.fieldnames or []):
                continue
            for row in reader:
                raw = row.get(column) or ""
                if not raw:
                    continue
                count_raw = row.get("operation_count") or ""
                count = int(float(count_raw)) if count_raw else 0
                cost = _profile_cost(row.get("GFLOPS"), row.get("FLOPS/Byte"), count)
                kernels.append(
                    {
                        "name": row["name"],
                        "time_us": float(raw),
                        "count": count,
                        "flops": None if cost is None else cost[0],
                        "bytes": None if cost is None else cost[1],
                        "shape": row.get("Input Dims") or "",
                    }
                )
    return kernels


def load_profile_json(path: Path) -> list[dict[str, Any]]:
    """``[{name, time_us, count, flops, bytes}]`` from a JSON file.

    ``flops`` and ``bytes`` are totals over ``count`` launches. Omit them when
    the trace had no shape; the projection then supplies the cost.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be a list of kernel records")
    kernels = []
    for item in payload:
        flops = item.get("flops")
        nbytes = item.get("bytes")
        kernels.append(
            {
                "name": str(item["name"]),
                "time_us": float(item["time_us"]),
                "count": int(item.get("count") or 0),
                "flops": None if not flops else float(flops),
                "bytes": None if not nbytes else float(nbytes),
                "shape": str(item.get("shape") or ""),
            }
        )
    return kernels


def _projection_roofline_time_us(row: dict[str, Any]) -> float:
    """One forward's roofline time for a projection row.

    ``total_roofline_time_us`` is already scaled by ``instances``. A row that
    only has the per-launch time is scaled the same way. Zero means the
    projection did not cost this row.
    """
    total = row.get("total_roofline_time_us")
    if total:
        return float(total)
    per_launch = row.get("roofline_time_us")
    if not per_launch:
        return 0.0
    instances = int(row.get("instances") or 0)
    return float(per_launch) * instances if instances else float(per_launch)


def regimes_from_augmentation(payload: dict[str, Any]) -> list[Regime]:
    """Category rollups from ``roofline_augmentation.json``.

    Each category keeps its launch count (``invocations``) so a multi-step
    capture can scale one forward by the launches it ran. A file from before
    that field keeps ``flops_per_launch`` as the fallback.
    """
    regimes = payload.get("regimes")
    if not regimes and payload.get("regime"):
        regimes = [payload["regime"]]
    summaries = payload.get("roofline_by_category")
    if isinstance(summaries, dict):
        summaries = [summaries]
    if not regimes or not summaries or len(regimes) != len(summaries):
        raise ValueError("augmentation needs one category summary per regime")
    out = []
    for regime, summary in zip(regimes, summaries):
        ops = []
        for category, facts in summary.items():
            ops.append(
                {
                    "category": category,
                    "total_flops": facts.get("flops") or 0.0,
                    "total_bytes": facts.get("bytes") or 0.0,
                    "compute_spec": facts.get("compute_spec"),
                    "flops_per_launch": facts.get("flops_per_invocation") or None,
                    "instances": facts.get("invocations") or 0,
                    # The augmentation already sums one forward. Do not scale
                    # it by invocations again.
                    "roofline_time_us": facts.get("roofline_time_us") or 0.0,
                }
            )
        out.append(
            Regime(
                phase=str(regime.get("phase") or ""),
                concurrency=int(regime["batch_size"]),
                ops=tuple(ops),
            )
        )
    return out


def regime_from_projection(path: Path, phase: str, concurrency: int) -> Regime:
    """One ``--output-json`` projection. Per-instance ``flops`` is one launch."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["ops"] if isinstance(payload, dict) else payload
    ops = []
    for row in rows:
        if not row.get("category"):
            continue
        ops.append(
            {
                "category": _projection_series_category(row),
                "total_flops": row.get("total_flops") or 0.0,
                "total_bytes": row.get("total_bytes") or 0.0,
                "compute_spec": row.get("compute_spec"),
                "flops_per_launch": row.get("flops") or None,
                "instances": row.get("instances") or 0,
                "roofline_time_us": _projection_roofline_time_us(row),
            }
        )
    return Regime(phase=phase, concurrency=concurrency, ops=tuple(ops))


def _bucket_ops(ops: tuple[dict[str, Any], ...], categories: tuple[str, ...] | None):
    if categories is None:
        return [op for op in ops if op.get("category")]
    wanted = set(categories)
    return [op for op in ops if op.get("category") in wanted]


def _bucket_cost(
    ops: tuple[dict[str, Any], ...], categories: tuple[str, ...] | None
) -> tuple[float, float, int, float | None]:
    """One forward's FLOPs and bytes, how many launches it is made of, and a
    per-launch cost for rollups that predate the launch count."""
    chosen = _bucket_ops(ops, categories)
    total_flops = sum(float(op.get("total_flops") or 0.0) for op in chosen)
    total_bytes = sum(float(op.get("total_bytes") or 0.0) for op in chosen)
    # Each projection row is ``instances`` launches of one op, so a bucket is
    # the sum of its rows' instances. Rows can differ in cost (a layer type per
    # compression ratio), so the bucket has a mean launch, not a single cost.
    # A bucket with an unknown instance count has no launch scaling.
    counts = [int(op.get("instances") or 0) for op in chosen]
    instances = sum(counts) if counts and all(counts) else 0
    # A rollup written before the launch count was recorded is one row with the
    # cost of its most expensive launch. It is only used when nothing better
    # exists.
    per_launch = None
    if len(chosen) == 1 and chosen[0].get("flops_per_launch"):
        per_launch = float(chosen[0]["flops_per_launch"])
    return total_flops, total_bytes, instances, per_launch


def flop_shares(ops: tuple[dict[str, Any], ...]) -> dict[str, float]:
    """Modeled FLOPs by ``compute_spec`` for the end-to-end blend."""
    shares: dict[str, float] = {}
    for op in ops:
        spec = op.get("compute_spec")
        flops = float(op.get("total_flops") or 0.0)
        if spec and flops > 0:
            shares[spec] = shares.get(spec, 0.0) + flops
    return shares


_SYNTHETIC_OP_SUFFIX = " (Synthetic Op)"


def _kernel_computation(name: str) -> str:
    """The device kernel. Text before ``->`` is a launcher or a display name."""
    body = (
        name[: -len(_SYNTHETIC_OP_SUFFIX)]
        if name.endswith(_SYNTHETIC_OP_SUFFIX)
        else name
    )
    arrow = body.rfind("->")
    if arrow >= 0:
        tail = body[arrow + 2 :].strip()
        if tail:
            return tail
    return body


def report_kernel_category(name: str) -> str | None:
    """Category of the computation the kernel ran.

    A leading ``prefill/`` or ``decode/`` is this plot's phase tag. The rest of
    the name is the op the perf report categorizes, and the text after ``->``
    is the GPU kernel that categorizer and the trace's kernel-name categories
    both see.
    """
    op_name = _PHASE_PREFIX.sub("", name)
    kernel = _PHASE_PREFIX.sub("", _kernel_computation(op_name))
    return categorize_operation(op_name, kernel)


def _projection_series_category(row: dict[str, Any]) -> str:
    """Map a projection row onto the perf-report category that series use.

    Attention and MoE keep the names the existing series match. A norm, a
    quant, and the remaining elementwise ops take the report's labels.
    """
    category = str(row.get("category") or "")
    computation = str(row.get("computation") or "").lower()
    if category in {"Attention", "MoE", "GEMM"}:
        return category
    if category.startswith("Normalization"):
        return "RMSNorm"
    if category == "Reduce":
        return "reduce"
    if category == "Conv1d":
        return "CONV_fwd"
    if category == "IndexedCopy":
        return "IndexedCopy"
    if category in {"UnaryElementwise", "BinaryElementwise"}:
        if "quant" in computation:
            return "GroupQuant"
        return "elementwise"
    return category


def _matches(
    kernel: dict[str, Any],
    *,
    phase: str,
    categories: tuple[str, ...] | None,
    includes: tuple[str, ...],
    multi_phase: bool,
    kernel_name: str | None = None,
    kernel_shape: str | None = None,
) -> bool:
    name = kernel["name"]
    if is_communication(name):
        return False
    if kernel_name is not None:
        if name != kernel_name:
            return False
        if kernel_shape is not None and str(kernel.get("shape") or "") != kernel_shape:
            return False
    elif categories is not None and report_kernel_category(name) not in categories:
        return False
    if includes and not any(token in name for token in includes):
        return False
    tagged = kernel_phase(name)
    if tagged is not None and tagged != phase:
        return False
    # An unscoped kernel in a mixed capture is not split across phases.
    # An explicit include is the caller saying this kernel belongs here.
    if tagged is None and multi_phase and not includes:
        return False
    return True


def _matched_kernels(
    kernels: list[dict[str, Any]],
    *,
    phase: str,
    categories: tuple[str, ...] | None,
    includes: tuple[str, ...],
    multi_phase: bool,
    kernel_name: str | None = None,
    kernel_shape: str | None = None,
) -> list[dict[str, Any]]:
    return [
        kernel
        for kernel in kernels
        if _matches(
            kernel,
            phase=phase,
            categories=categories,
            includes=includes,
            multi_phase=multi_phase,
            kernel_name=kernel_name,
            kernel_shape=kernel_shape,
        )
    ]


def _measured_launches(matched: list[dict[str, Any]]) -> tuple[float, int]:
    """Summed kernel time, and how many projected launches that time covers.

    Kernels with the same launch count are parts of one launch (a gate-up and
    its down-projection). Kernels with different counts are separate launches,
    so those counts add.
    """
    time_us = sum(kernel["time_us"] for kernel in matched)
    counts = {kernel["count"] for kernel in matched if kernel["count"]}
    return time_us, sum(counts)


def achieved_tflops(
    total_flops: float,
    instances: int,
    launch_count: int,
    time_us: float,
    flops_per_launch: float | None = None,
) -> float:
    """Achieved TFLOP/s while the measured launches ran.

    ``total_flops`` is one forward over ``instances`` launches, so a capture
    with ``launch_count`` launches ran ``launch_count / instances`` forwards.
    Scaling the forward, not one row's launch, keeps a bucket right when its
    rows differ in cost. A rollup with no launch count falls back to
    ``flops_per_launch * launch_count``. With neither, the profile window is
    taken as one forward and ``total_flops`` is the numerator.
    """
    if time_us <= 0:
        raise ValueError("measured time must be positive")
    if instances and launch_count > 0:
        flops = total_flops * launch_count / instances
    elif flops_per_launch and launch_count > 0:
        flops = flops_per_launch * launch_count
    else:
        flops = total_flops
    return flops / (time_us * 1e-6) / 1e12


def roof_tflops(arithmetic_intensity: float, arch: dict[str, Any]) -> float | None:
    """The most a kernel at this intensity can reach on ``arch``.

    The bandwidth line, capped by the highest matrix peak the plot draws, so a
    kernel on a faster datatype than bf16 is not flagged. ``None`` when the
    arch lists no peak.
    """
    peaks = arch.get("max_achievable_tflops") or {}
    values = [float(peaks[key]) for key, *_rest in _HARDWARE_ROOFS if peaks.get(key)]
    if not values:
        return None
    return min(bandwidth_tb_s(arch) * arithmetic_intensity, max(values))


def flag_points_above_roof(
    points: list[RooflinePoint],
    arch: dict[str, Any],
    *,
    tolerance: float = 0.05,
) -> tuple[list[RooflinePoint], list[str]]:
    """Mark each point that sits above the roof it can reach.

    Nothing runs faster than the roof, so a point above it (past ``tolerance``)
    has FLOPs, bytes, or matched kernels that do not describe the work the
    profile timed. The warning gives the memory rate the point implies.
    """
    flagged: list[RooflinePoint] = []
    warnings: list[str] = []
    bandwidth = bandwidth_tb_s(arch)
    for point in points:
        roof = roof_tflops(point.arithmetic_intensity, arch)
        if roof is None or point.achieved_tflops <= roof * (1.0 + tolerance):
            flagged.append(point)
            continue
        flagged.append(replace(point, above_roof=True))
        implied = point.achieved_tflops / point.arithmetic_intensity
        warnings.append(
            f"{point.series} {point.phase} concurrency {point.concurrency}: "
            f"{point.achieved_tflops:,.1f} TFLOP/s is above the "
            f"{roof:,.1f} TFLOP/s roof at {point.arithmetic_intensity:.2f} "
            f"FLOP/byte (implies {implied:,.1f} TB/s against {bandwidth:,.1f} "
            "TB/s). The projection's FLOPs or bytes, or the kernels matched to "
            "this series, do not describe what ran"
        )
    return flagged, warnings


def _has_profile_cost(kernel: dict[str, Any]) -> bool:
    return bool(kernel.get("flops")) and bool(kernel.get("bytes"))


def _category_work(
    kernels: list[dict[str, Any]],
    instances: int,
    *,
    one_forward: bool,
) -> tuple[float, float, str]:
    """FLOPs and bytes from the PerfModel on the traced shapes.

    A kernel with no ``GFLOPS`` is left out. It is not filled from the static
    projection, whose shapes are not this trace.
    """
    costed = [kernel for kernel in kernels if _has_profile_cost(kernel)]
    if not costed:
        # A traced category with no perf-model shape, or a category the
        # capture did not run. The static projection is a different graph.
        return 0.0, 0.0, "none"
    total_flops = sum(float(kernel["flops"]) for kernel in costed)
    total_bytes = sum(float(kernel["bytes"]) for kernel in costed)
    if total_flops <= 0 or total_bytes <= 0:
        return 0.0, 0.0, "none"
    if not one_forward:
        return total_flops, total_bytes, "profile"
    _, launches = _measured_launches(costed)
    if instances and launches:
        scale = instances / launches
        return total_flops * scale, total_bytes * scale, "profile"
    return total_flops, total_bytes, "profile"


def _series_work(
    matched: list[dict[str, Any]],
    ops: tuple[dict[str, Any], ...],
    categories: tuple[str, ...] | None,
    *,
    one_forward: bool,
) -> tuple[float, float, str]:
    """FLOPs and bytes for a series, from the kernels the PerfModel costed."""
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for op in _bucket_ops(ops, categories):
        by_cat.setdefault(str(op.get("category")), []).append(op)
    grouped: dict[str, list[dict[str, Any]]] = {cat: [] for cat in by_cat}
    extra: list[dict[str, Any]] = []
    for kernel in matched:
        cat = report_kernel_category(kernel["name"])
        if cat in grouped:
            grouped[cat].append(kernel)
        elif _has_profile_cost(kernel):
            extra.append(kernel)
    flops = 0.0
    nbytes = 0.0
    sources: set[str] = set()
    for cat, rows in by_cat.items():
        _flops, _bytes, instances, _per_launch = _bucket_cost(tuple(rows), None)
        part_flops, part_bytes, source = _category_work(
            grouped[cat],
            instances,
            one_forward=one_forward,
        )
        flops += part_flops
        nbytes += part_bytes
        if source != "none":
            sources.add(source)
    if not one_forward:
        for kernel in extra:
            flops += float(kernel["flops"])
            nbytes += float(kernel["bytes"])
            sources.add("profile")
    if not sources:
        return 0.0, 0.0, "export"
    source = "mixed" if len(sources) > 1 else sources.pop()
    return flops, nbytes, source


def _accounted_kernels(
    matched: list[dict[str, Any]],
    ops: tuple[dict[str, Any], ...],
    categories: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    """Kernels whose time belongs with the work ``_series_work`` counted.

    Only a kernel the PerfModel costed is included. A kernel with no
    ``GFLOPS`` has no traced shape to plot, and a static projection is not
    a substitute, so its time is not added to a point built from other shapes.
    """
    known = {str(op.get("category")) for op in _bucket_ops(ops, categories)}
    kept = []
    for kernel in matched:
        if not _has_profile_cost(kernel):
            continue
        cat = report_kernel_category(kernel["name"])
        if cat is None or cat in known or categories is None:
            kept.append(kernel)
    return kept


@dataclass(frozen=True)
class _JoinContext:
    """The inputs every series join in ``build_points`` reads, and its warnings."""

    kernels: list[dict[str, Any]]
    includes: dict[tuple[str, str], tuple[str, ...]]
    time_overrides_us: dict[tuple[str, str], float]
    multi_phase: bool
    warnings: list[str]


def build_points(
    regimes: list[Regime],
    kernels: list[dict[str, Any]],
    *,
    series: tuple[Series, ...] = DEFAULT_SERIES,
    includes: dict[tuple[str, str], tuple[str, ...]] | None = None,
    time_overrides_us: dict[tuple[str, str], float] | None = None,
) -> tuple[list[RooflinePoint], list[str]]:
    """Join each series at each regime to the kernels that implement it.

    ``includes`` and ``time_overrides_us`` are keyed by ``(phase, series label)``.
    An include restricts which kernel names count. A time override replaces the
    summed kernel time and, with it, the launch-scaled numerator: the override
    is the time of one forward.
    """
    phases = {regime.phase for regime in regimes}
    warnings: list[str] = []
    context = _JoinContext(
        kernels=kernels,
        includes=includes or {},
        time_overrides_us=time_overrides_us or {},
        multi_phase=len(phases) > 1,
        warnings=warnings,
    )
    points: list[RooflinePoint] = []
    for regime in regimes:
        for spec in series:
            if spec.phase is not None and spec.phase != regime.phase:
                continue
            point = _series_point(context, regime, spec)
            if point is not None:
                points.append(point)
    points = _drop_whole_model_matching_end_to_end(points, warnings)
    return points, warnings


def _roofline_point(
    label: str,
    regime: Regime,
    flops: float,
    nbytes: float,
    time_us: float,
    source: str,
) -> RooflinePoint:
    """The point for ``flops`` and ``nbytes`` of work that ran in ``time_us``."""
    return RooflinePoint(
        series=label,
        phase=regime.phase,
        concurrency=regime.concurrency,
        arithmetic_intensity=flops / nbytes,
        achieved_tflops=flops / (time_us * 1e-6) / 1e12,
        flops=flops,
        bytes=nbytes,
        time_us=time_us,
        cost_source=source,
    )


def _series_point(
    context: _JoinContext, regime: Regime, spec: Series
) -> RooflinePoint | None:
    """One series at one regime, or None when it has nothing to draw."""
    key = (regime.phase, spec.label)
    override = context.time_overrides_us.get(key)
    # End to end in a mixed capture only has phase-tagged kernels, which
    # are not the whole forward. Require an explicit one-forward time.
    # Whole model is the attributed kernels, so it is plotted without one.
    if spec.label == "End to end" and context.multi_phase and override is None:
        context.warnings.append(
            f"{spec.label} {regime.phase}: mixed capture has no "
            "whole-forward time (pass --time-us)"
        )
        return None
    include = context.includes.get(key, ())
    matched = _series_kernels(context, regime.phase, spec, include)
    if spec.kernel_name is not None:
        return _top_kernel_point(context, regime, spec, matched, override)
    return _category_point(context, regime, spec, matched, override, include)


def _series_kernels(
    context: _JoinContext, phase: str, spec: Series, include: tuple[str, ...]
) -> list[dict[str, Any]]:
    """The profile kernels that implement ``spec`` in ``phase``."""
    matched = _matched_kernels(
        context.kernels,
        phase=phase,
        categories=spec.categories,
        includes=() if spec.label == "Whole model" else include,
        multi_phase=context.multi_phase,
        kernel_name=spec.kernel_name,
        kernel_shape=spec.kernel_shape,
    )
    # An include on Whole model adds unscoped kernels to the phase-tagged
    # ones. On any other series the include is the whole match.
    if spec.label == "Whole model" and include:
        seen = {kernel["name"] for kernel in matched}
        for kernel in _matched_kernels(
            context.kernels,
            phase=phase,
            categories=spec.categories,
            includes=include,
            multi_phase=context.multi_phase,
        ):
            if kernel["name"] not in seen:
                matched.append(kernel)
                seen.add(kernel["name"])
    return matched


def _top_kernel_point(
    context: _JoinContext,
    regime: Regime,
    spec: Series,
    matched: list[dict[str, Any]],
    override: float | None,
) -> RooflinePoint | None:
    """A top-kernel marker: the kernel's own profile FLOPs over its time."""
    if not matched:
        return None
    if not all(_has_profile_cost(kernel) for kernel in matched):
        context.warnings.append(f"{spec.label} {regime.phase}: no profile FLOPs")
        return None
    time_us = (
        override
        if override is not None
        else sum(kernel["time_us"] for kernel in matched)
    )
    total_flops = sum(float(kernel["flops"]) for kernel in matched)
    total_bytes = sum(float(kernel["bytes"]) for kernel in matched)
    if time_us <= 0 or total_flops <= 0 or total_bytes <= 0:
        return None
    return _roofline_point(
        spec.label, regime, total_flops, total_bytes, time_us, "profile"
    )


def _leaves_out_unscoped(
    context: _JoinContext, spec: Series, include: tuple[str, ...]
) -> bool:
    """True when a mixed capture's Whole model drops kernels with no phase tag."""
    return (
        spec.label == "Whole model"
        and context.multi_phase
        and any(
            kernel_phase(kernel["name"]) is None
            and not is_communication(kernel["name"])
            for kernel in context.kernels
        )
        and not include
    )


def _category_point(
    context: _JoinContext,
    regime: Regime,
    spec: Series,
    matched: list[dict[str, Any]],
    override: float | None,
    include: tuple[str, ...],
) -> RooflinePoint | None:
    """A category, Whole model, or End to end marker from the costed kernels."""
    one_forward = override is not None
    if one_forward:
        time_us = override
    else:
        accounted = _accounted_kernels(matched, regime.ops, spec.categories)
        time_us = sum(kernel["time_us"] for kernel in accounted)
        if _leaves_out_unscoped(context, spec, include):
            context.warnings.append(
                f"Whole model {regime.phase}: unscoped kernels are left "
                "out of a mixed capture (pass --kernels)"
            )
    if time_us <= 0:
        # No costed kernel. Kernels that ran with no perf-model shape
        # are left off the figure rather than priced from the static
        # graph. A series that matched no kernel at all is missing its
        # profile time.
        ran = any(float(kernel["time_us"]) > 0 for kernel in matched)
        bucket_flops, bucket_bytes, _, _ = _bucket_cost(regime.ops, spec.categories)
        if not ran and bucket_flops > 0 and bucket_bytes > 0:
            context.warnings.append(f"{spec.label} {regime.phase}: no profile time")
        return None
    total_flops, total_bytes, source = _series_work(
        matched,
        regime.ops,
        spec.categories,
        one_forward=one_forward,
    )
    if total_flops <= 0 or total_bytes <= 0:
        return None
    return _roofline_point(
        spec.label, regime, total_flops, total_bytes, time_us, source
    )


def _same_measurement(left: RooflinePoint, right: RooflinePoint) -> bool:
    """True when two points are the same forward: same place, same work, same time."""

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1.0)

    return (
        left.phase == right.phase
        and left.concurrency == right.concurrency
        and close(left.flops, right.flops)
        and close(left.bytes, right.bytes)
        and close(left.time_us, right.time_us)
    )


def _drop_whole_model_matching_end_to_end(
    points: list[RooflinePoint], warnings: list[str]
) -> list[RooflinePoint]:
    """Omit Whole model where it is the same forward as End to end.

    Whole model sums the kernels attributed to the phase. End to end is one
    forward. When every kernel is attributed, those are one measurement, and
    drawing both puts two markers on one point.
    """
    ends = [point for point in points if point.series == "End to end"]
    if not ends:
        return points
    kept: list[RooflinePoint] = []
    for point in points:
        if point.series == "Whole model" and any(
            _same_measurement(point, end) for end in ends
        ):
            warnings.append(
                f"Whole model {point.phase} matches End to end, "
                "so only End to end is drawn"
            )
            continue
        kept.append(point)
    return kept


def _fmt_number(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return f"{value:,.0f}"
    return f"{value:,.1f}"


def _model_forward_time_us(ops: list[dict[str, Any]]) -> float | None:
    """Roofline time of one forward. None when a costed op has no time."""
    time_us = 0.0
    for op in ops:
        if float(op.get("total_flops") or 0.0) <= 0:
            continue
        part = op.get("roofline_time_us")
        if not part:
            return None
        time_us += float(part)
    return time_us


def points_from_model(
    regimes: list[Regime],
    *,
    series: tuple[Series, ...] = DEFAULT_SERIES,
) -> tuple[list[RooflinePoint], list[str]]:
    """Points from a projection alone.

    X is the projection's FLOPs over its bytes. Y divides those FLOPs by the
    projection's roofline time, so a single-spec series sits on the roof those
    shapes imply. No profile is read.
    """
    points: list[RooflinePoint] = []
    warnings: list[str] = []
    for regime in regimes:
        for spec in series:
            if spec.phase is not None and spec.phase != regime.phase:
                continue
            chosen = _bucket_ops(regime.ops, spec.categories)
            flops = sum(float(op.get("total_flops") or 0.0) for op in chosen)
            nbytes = sum(float(op.get("total_bytes") or 0.0) for op in chosen)
            if flops <= 0 or nbytes <= 0:
                continue
            time_us = _model_forward_time_us(chosen)
            if time_us is None or time_us <= 0:
                warnings.append(
                    f"{spec.label} {regime.phase}: projection has no roofline time"
                )
                continue
            points.append(
                RooflinePoint(
                    series=spec.label,
                    phase=regime.phase,
                    concurrency=regime.concurrency,
                    arithmetic_intensity=flops / nbytes,
                    achieved_tflops=flops / (time_us * 1e-6) / 1e12,
                    flops=flops,
                    bytes=nbytes,
                    time_us=time_us,
                    cost_source="model",
                )
            )
    return _drop_whole_model_matching_end_to_end(points, warnings), warnings


@dataclass(frozen=True)
class ModelCoverage:
    """Kernel time a performance model prices, and the kernels it does not."""

    covered_share: float
    unmodeled_count: int
    unmodeled_share: float
    # ``(display name, share of kernel time)``, largest first. Only kernels
    # over ``_UNMODELED_LIST_SHARE``, and only when the unmodeled kernels
    # together are over that share. At most ``_UNMODELED_LIST_LIMIT``.
    listed: tuple[tuple[str, float], ...]


def _coverage_kernel_name(name: str) -> str:
    """The device kernel, without the phase tag or a ``.kd`` suffix."""
    computation = _PHASE_PREFIX.sub("", _kernel_computation(name))
    return _KERNEL_SUFFIX.sub("", computation)


def _modeled_categories(regimes: list[Regime]) -> dict[str, set[str]]:
    """Categories a regime's projection gave a non-zero cost."""
    categories: dict[str, set[str]] = {}
    for regime in regimes:
        priced: dict[str, list[float]] = {}
        for op in regime.ops:
            category = op.get("category")
            if not category:
                continue
            slot = priced.setdefault(str(category), [0.0, 0.0])
            slot[0] += float(op.get("total_flops") or 0.0)
            slot[1] += float(op.get("total_bytes") or 0.0)
        categories[regime.phase] = {
            category
            for category, (flops, nbytes) in priced.items()
            if flops > 0 and nbytes > 0
        }
    return categories


def model_coverage(
    regimes: list[Regime], kernels: list[dict[str, Any]]
) -> ModelCoverage | None:
    """Share of kernel time whose computation a performance model prices.

    A kernel is covered when its device computation maps to a category the
    projection costed for that kernel's phase, or when the profile itself
    recorded its FLOPs and bytes. Communication is outside the roofline and
    is left out of both sides. The same kernel in two phases is one name,
    and its time is the sum.
    """
    if not kernels:
        return None
    known = _modeled_categories(regimes)
    any_known = set().union(*known.values()) if known else set()
    total = 0.0
    covered = 0.0
    unmodeled: dict[str, float] = {}
    for kernel in kernels:
        name = str(kernel["name"])
        if is_communication(name):
            continue
        time_us = float(kernel.get("time_us") or 0.0)
        if time_us <= 0:
            continue
        total += time_us
        phase = kernel_phase(name)
        priced = known.get(phase, any_known) if phase else any_known
        category = report_kernel_category(name)
        if _has_profile_cost(kernel) or (category and category in priced):
            covered += time_us
            continue
        label = _coverage_kernel_name(name)
        unmodeled[label] = unmodeled.get(label, 0.0) + time_us
    if total <= 0:
        return None
    unmodeled_time = total - covered
    listed: tuple[tuple[str, float], ...] = ()
    if unmodeled_time / total > _UNMODELED_LIST_SHARE:
        ranked = sorted(unmodeled.items(), key=lambda item: item[1], reverse=True)
        listed = tuple(
            (label, time_us / total)
            for label, time_us in ranked
            if time_us / total > _UNMODELED_LIST_SHARE
        )[:_UNMODELED_LIST_LIMIT]
    return ModelCoverage(
        covered_share=covered / total,
        unmodeled_count=len(unmodeled),
        unmodeled_share=unmodeled_time / total,
        listed=listed,
    )


def figure_kernel_time_share(
    points: list[RooflinePoint],
    series: tuple[Series, ...],
    kernels: list[dict[str, Any]],
) -> float | None:
    """Share of kernel time drawn as its own marker.

    ``None`` on a category plot. End to end is the whole forward, so it is not
    part of the share. A top kernel the plot skipped has no marker and is not
    part of it either.
    """
    shown = {
        (spec.kernel_name, spec.kernel_shape)
        for spec in series
        if spec.kernel_name is not None
        and any(point.series == spec.label for point in points)
    }
    if not any(spec.kernel_name is not None for spec in series):
        return None
    total = 0.0
    drawn = 0.0
    unscoped = {label for label, shape in shown if shape is None}
    shaped = {(label, shape) for label, shape in shown if shape is not None}
    for kernel in kernels:
        name = str(kernel["name"])
        if is_communication(name):
            continue
        time_us = float(kernel.get("time_us") or 0.0)
        if time_us <= 0:
            continue
        total += time_us
        if name in unscoped or (name, str(kernel.get("shape") or "")) in shaped:
            drawn += time_us
    if total <= 0:
        return None
    return drawn / total


def figure_kernel_note(share: float) -> str:
    """The top-kernel caption: how much of the run the drawn kernels are."""
    return f"Figure shows kernels taking {share * 100:.1f}% of the run time."


def model_coverage_note(coverage: ModelCoverage) -> str:
    """The legend note: coverage, then the unmodeled kernels worth naming."""
    lines = [
        "Performance models cover "
        f"{coverage.covered_share * 100:.1f}% of kernel time."
    ]
    if coverage.unmodeled_share > _UNMODELED_LIST_SHARE:
        lines.append(
            f"{coverage.unmodeled_count} kernels are unmodeled "
            f"({coverage.unmodeled_share * 100:.1f}% of kernel time)."
        )
        if coverage.listed:
            shown = ", ".join(
                f"{name} ({share * 100:.1f}%)" for name, share in coverage.listed
            )
            lines.append(f"Top unmodeled kernels: {shown}.")
    wrapped = [textwrap.fill(line, width=118, break_long_words=False) for line in lines]
    return "\n".join(wrapped)


@dataclass(frozen=True)
class _RoofFrame:
    """The roofs the arch lists and the axis limits the plot draws them in.

    ``hardware`` holds ``(key, label, peak, linestyle, color)`` per roof, in
    ``_HARDWARE_ROOFS`` order.
    """

    bandwidth: float
    hardware: list[tuple[str, str, float, str, str]]
    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float

    @property
    def ordered(self) -> list[tuple[str, str, float, str, str]]:
        """The roofs, lowest peak first."""
        return sorted(self.hardware, key=lambda item: item[2])

    @property
    def lowest_ridge(self) -> float:
        return self.ordered[0][2] / self.bandwidth


def _roof_frame(points: list[RooflinePoint], arch: dict[str, Any]) -> _RoofFrame:
    """The arch's matrix roofs, and limits that show them and every point."""
    bandwidth = bandwidth_tb_s(arch)
    peaks = arch.get("max_achievable_tflops") or {}
    hardware = [
        (key, label, float(peaks[key]), linestyle, color)
        for key, label, linestyle, color in _HARDWARE_ROOFS
        if peaks.get(key)
    ]
    if not hardware:
        raise ValueError("arch spec has no bfloat16, fp8, or fp4 matrix peak")

    intensities = [point.arithmetic_intensity for point in points]
    rates = [point.achieved_tflops for point in points]
    if not intensities:
        intensities = [1.0]
        rates = [1.0]
    peak_values = [peak for _key, _label, peak, _style, _color in hardware]
    x_lo = min(intensities) / 5.0
    x_hi = max(
        max(intensities) * 5.0, max(peak / bandwidth for peak in peak_values) * 5.0
    )
    return _RoofFrame(
        bandwidth=bandwidth,
        hardware=hardware,
        x_lo=x_lo,
        x_hi=x_hi,
        y_lo=min(min(rates) / 5.0, bandwidth * x_lo),
        y_hi=max(max(rates) * 5.0, max(peak_values) * 1.4),
    )


def _diagonal_steps(frame: _RoofFrame) -> Iterator[tuple[float, float, str, str]]:
    """``(start, ridge, linestyle, color)`` of each faster peak's diagonal.

    Each peak above the lowest owns the bandwidth diagonal from the previous
    ridge up to its own. A peak whose ridge is not past the previous owns none.
    """
    previous = frame.lowest_ridge
    for _key, _label, peak, linestyle, color in frame.ordered[1:]:
        ridge = peak / frame.bandwidth
        if ridge <= previous:
            continue
        yield previous, ridge, linestyle, color
        previous = ridge


def _draw_hardware_roofs(ax: Any, frame: _RoofFrame) -> None:
    """The bandwidth diagonal and each compute roof at its full peak."""
    bandwidth = frame.bandwidth
    lowest_ridge = frame.lowest_ridge
    # Below the lowest ridge every peak shares the bandwidth diagonal, drawn
    # solid. Each faster peak then owns the diagonal up to its own ridge, in
    # that peak's linestyle: dash-dot for fp8, dashed for fp4.
    ax.plot(
        [frame.x_lo, lowest_ridge],
        [bandwidth * frame.x_lo, bandwidth * lowest_ridge],
        color="0.45",
        linewidth=1.6,
        label=f"Bandwidth roof, {_fmt_number(bandwidth)} TB/s",
    )
    for start, ridge, linestyle, color in _diagonal_steps(frame):
        ax.plot(
            [start, ridge],
            [bandwidth * start, bandwidth * ridge],
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
        )
    for _key, label, peak, linestyle, color in frame.hardware:
        ridge = ridge_flops_per_byte(peak, bandwidth)
        ax.plot(
            [ridge, frame.x_hi],
            [peak, peak],
            color=color,
            linestyle=linestyle,
            linewidth=1.3,
            label=(
                f"{label} roof, {_fmt_number(peak)} TFLOP/s, "
                f"ridge {_fmt_number(ridge)} FLOP/byte"
            ),
        )


def _draw_peak_fraction(ax: Any, frame: _RoofFrame, fraction: float, gray: str) -> None:
    """Every roof scaled to ``fraction`` of its peak, in ``gray``.

    A lower peak with the same bandwidth would only move the knee left along
    the roof already drawn. Scale the diagonal with the peak so each fraction
    has its own slope, in the same linestyle as the full roof.
    """
    scaled = fraction * frame.bandwidth
    x_lo, x_hi = frame.x_lo, frame.x_hi
    lowest_ridge = frame.lowest_ridge
    if lowest_ridge > x_lo:
        ax.plot(
            [x_lo, lowest_ridge],
            [scaled * x_lo, scaled * lowest_ridge],
            color=gray,
            linewidth=0.7,
            zorder=1,
            label=f"{fraction * 100:.0f}% of peak",
        )
    for start, ridge, linestyle, _color in _diagonal_steps(frame):
        ax.plot(
            [start, ridge],
            [scaled * start, scaled * ridge],
            color=gray,
            linestyle=linestyle,
            linewidth=0.7,
            zorder=1,
        )
    for _key, _label, peak, linestyle, _color in frame.hardware:
        ridge = ridge_flops_per_byte(peak, frame.bandwidth)
        ax.plot(
            [max(ridge, x_lo), x_hi],
            [fraction * peak, fraction * peak],
            color=gray,
            linestyle=linestyle,
            linewidth=0.7,
            zorder=1,
        )


def _draw_blend_roofs(
    ax: Any, frame: _RoofFrame, regimes: list[Regime], peaks: dict[str, float]
) -> None:
    """One blended end-to-end roof per concurrency whose mix spans specs."""
    seen_blend: set[int] = set()
    for regime in regimes:
        if regime.concurrency in seen_blend:
            continue
        blended = blend_peak_tflops(flop_shares(regime.ops), peaks)
        if blended is None:
            continue
        peak, shares = blended
        # A one-spec mix sits on that spec's roof already.
        if len(shares) < 2:
            continue
        seen_blend.add(regime.concurrency)
        ridge = ridge_flops_per_byte(peak, frame.bandwidth)
        other = max(
            (share, spec) for spec, share in shares.items() if spec != "matrix_bf16"
        )
        share, spec = other
        short = spec.removeprefix("matrix_")
        ax.plot(
            [max(ridge, frame.x_lo), frame.x_hi],
            [peak, peak],
            linestyle="-.",
            linewidth=1.0,
            label=(
                f"End-to-end blend at concurrency {regime.concurrency}, "
                f"{share * 100:.0f}% {short} by FLOP, "
                f"{_fmt_number(peak)} TFLOP/s, ridge {_fmt_number(ridge)} FLOP/byte"
            ),
        )


def _concurrency_colors(points: list[RooflinePoint]) -> dict[int, str]:
    """A color per concurrency on the plot, keyed in ascending order."""
    concurrencies = sorted({point.concurrency for point in points})
    return {
        concurrency: _CONCURRENCY_COLORS[index % len(_CONCURRENCY_COLORS)]
        for index, concurrency in enumerate(concurrencies)
    }


def _draw_points(
    ax: Any,
    points: list[RooflinePoint],
    series: tuple[Series, ...],
    colors: dict[int, str],
) -> dict[str, list[RooflinePoint]]:
    """Scatter every point in its series marker. Returns the points by series."""
    by_series: dict[str, list[RooflinePoint]] = {}
    for point in points:
        by_series.setdefault(point.series, []).append(point)
    series_by_label = {spec.label: spec for spec in series}
    for label, group in by_series.items():
        spec = series_by_label.get(label)
        marker = spec.marker if spec else "o"
        for point in group:
            color = colors[point.concurrency]
            # Hollow is the prefill half of {prefill, decode} x {series}.
            hollow = point.phase == "prefill"
            ax.scatter(
                [point.arithmetic_intensity],
                [point.achieved_tflops],
                marker=marker,
                s=60,
                # White, not transparent: a hollow point covers any filled
                # point drawn underneath it at the same place, such as Whole
                # model's plus, which otherwise shows through its center.
                facecolors="white" if hollow else color,
                edgecolors=color,
                linewidths=1.6,
                zorder=3,
            )
    return by_series


def _series_family(spec: Series) -> str:
    """The legend family: both attention phases are one, others are their label."""
    return "Attention" if spec.categories == ("Attention",) else spec.label


def _legend_families(
    series: tuple[Series, ...], by_series: dict[str, list[RooflinePoint]]
) -> list[tuple[str, str | tuple[int, int, int]]]:
    """``(family, marker)`` per family with a point, End to end first."""
    families: list[tuple[str, str | tuple[int, int, int]]] = []
    seen_families: set[str] = set()
    for spec in series:
        if spec.label not in by_series:
            continue
        family = _series_family(spec)
        if family in seen_families:
            continue
        seen_families.add(family)
        families.append((family, spec.marker))
    # End to end is the whole forward, so its key entry leads the markers,
    # ahead of the per-kernel and per-category series. Stable otherwise.
    families.sort(key=lambda item: item[0] != "End to end")
    return families


def _add_family_keys(
    handles: list[Any],
    labels: list[str],
    series: tuple[Series, ...],
    by_series: dict[str, list[RooflinePoint]],
    points: list[RooflinePoint],
    colors: dict[int, str],
) -> None:
    """Append a marker swatch per series family and phase that has a point.

    The points are {prefill, decode} x {attention, whole model, end to end}.
    Fill is the phase. The marker is the series.
    """
    from matplotlib.lines import Line2D

    series_by_label = {spec.label: spec for spec in series}
    point_family = {
        label: _series_family(series_by_label[label])
        for label in by_series
        if label in series_by_label
    }
    concurrencies = list(colors)
    for family, marker in _legend_families(series, by_series):
        for phase in ("prefill", "decode"):
            if not any(
                point.phase == phase and point_family.get(point.series) == family
                for point in points
            ):
                continue
            # One concurrency has one color, so the swatches use it. Several
            # concurrencies keep gray swatches and a separate color key.
            ink = colors[concurrencies[0]] if len(concurrencies) == 1 else "0.2"
            handles.append(
                Line2D(
                    [],
                    [],
                    marker=marker,
                    linestyle="None",
                    markerfacecolor="none" if phase == "prefill" else ink,
                    markeredgecolor=ink,
                    markersize=8,
                )
            )
            labels.append(f"{family}, {phase}")


def _add_concurrency_keys(
    handles: list[Any], labels: list[str], colors: dict[int, str]
) -> None:
    """Append a color swatch per concurrency when the plot has more than one."""
    from matplotlib.lines import Line2D

    # A single concurrency is the only color on the plot, so naming it adds nothing.
    if len(colors) < 2:
        return
    for concurrency, color in colors.items():
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=8,
            )
        )
        labels.append(f"Concurrency {concurrency}")


def _figure_note(
    points: list[RooflinePoint],
    series: tuple[Series, ...],
    regimes: list[Regime],
    kernels: list[dict[str, Any]] | None,
) -> str | None:
    """The caption under the legend: drawn kernel share, else model coverage."""
    share = (
        figure_kernel_time_share(points, series, kernels)
        if kernels is not None
        else None
    )
    if share is not None:
        return figure_kernel_note(share)
    coverage = model_coverage(regimes, kernels) if kernels else None
    return model_coverage_note(coverage) if coverage is not None else None


def _place_note_under_legend(fig: Any, legend: Any, note: str) -> None:
    """Write ``note`` just below the legend, which sits below the axes."""
    fig.canvas.draw()
    below = legend.get_window_extent(fig.canvas.get_renderer())
    below = below.transformed(fig.transFigure.inverted())
    fig.text(
        0.5,
        below.y0 - 0.012,
        note,
        ha="center",
        va="top",
        fontsize=8,
        color="0.2",
        linespacing=1.4,
    )


def plot_e2e_roofline(
    points: list[RooflinePoint],
    arch: dict[str, Any],
    regimes: list[Regime],
    *,
    title: str,
    output_path: Path,
    series: tuple[Series, ...] = DEFAULT_SERIES,
    kernels: list[dict[str, Any]] | None = None,
    ylabel: str = _PROFILE_YLABEL,
):
    """Draw the log-log roofline and write it to ``output_path``."""
    import matplotlib.pyplot as plt

    frame = _roof_frame(points, arch)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(frame.x_lo, frame.x_hi)
    ax.set_ylim(frame.y_lo, frame.y_hi)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, color="0.85")

    _draw_hardware_roofs(ax, frame)
    for fraction, gray in _PEAK_FRACTIONS:
        _draw_peak_fraction(ax, frame, fraction, gray)
    _draw_blend_roofs(ax, frame, regimes, arch.get("max_achievable_tflops") or {})

    colors = _concurrency_colors(points)
    by_series = _draw_points(ax, points, series, colors)

    handles, labels = ax.get_legend_handles_labels()
    _add_family_keys(handles, labels, series, by_series, points, colors)
    _add_concurrency_keys(handles, labels, colors)
    legend = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    ax.set_xlabel("Arithmetic intensity, FLOP per byte of compulsory HBM traffic")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    note = _figure_note(points, series, regimes, kernels)
    if note is not None and legend is not None:
        _place_note_under_legend(fig, legend, note)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_keyed_flag(entries: list[str], value_is_float: bool):
    """``phase|series|value`` flags. Series labels contain spaces."""
    parsed: dict[tuple[str, str], Any] = {}
    for entry in entries:
        phase, sep, rest = entry.partition("|")
        series, sep2, value = rest.partition("|")
        if not sep or not sep2 or not phase or not series or not value:
            raise ValueError(f"expected phase|series|value, got {entry!r}")
        key = (phase, series)
        if value_is_float:
            parsed[key] = float(value)
        else:
            parsed.setdefault(key, [])
            parsed[key].append(value)
    if value_is_float:
        return parsed
    return {key: tuple(values) for key, values in parsed.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="TraceLens_e2e_roofline_plot",
        description=(
            "Plot an end-to-end roofline. FLOPs and bytes come from the profile "
            "when it has them, and from a roofline projection otherwise. Y is "
            "those FLOPs divided by kernel time from the same profile. Markers are "
            "one per category. --top-kernels replaces them with the "
            "longest-running kernels and end to end. "
            "--model-only uses the projection's roofline time instead, and "
            "reads no profile."
        ),
    )
    _add_output_arguments(parser)
    _add_arch_arguments(parser)
    _add_regime_arguments(parser)
    _add_profile_arguments(parser)
    _add_series_arguments(parser)
    return parser


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """The figure title and the PNG to write."""
    parser.add_argument("--title", required=True, help="Figure title")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="PNG path to write",
    )


def _add_arch_arguments(parser: argparse.ArgumentParser) -> None:
    """The GPU arch spec that sets the roofs."""
    parser.add_argument(
        "--gpu-arch-name",
        help=(
            "GPU arch spec name. Resolved by load_arch, so a name that is not "
            f"bundled ({', '.join(list_platforms())}) comes from $TL_EXTENSION."
        ),
    )
    parser.add_argument(
        "--gpu-arch-json-path",
        type=Path,
        help="GPU arch JSON. Takes precedence over --gpu-arch-name.",
    )


def _add_regime_arguments(parser: argparse.ArgumentParser) -> None:
    """The projected regimes: an augmentation file or projection JSONs."""
    parser.add_argument(
        "--augmentation",
        type=Path,
        help="roofline_augmentation.json written by the analysis orchestrator",
    )
    parser.add_argument(
        "--projection",
        action="append",
        default=[],
        metavar="PHASE:CONCURRENCY:PATH",
        help=(
            "Projection JSON from TraceLens_model_roofline_projection "
            "--output-json. Repeat for each regime."
        ),
    )


def _add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    """The profile to read, or ``--model-only`` to read none."""
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Directory of perf-report *_ops.csv files",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help=(
            "Plot the projection with no profile. Y is one forward's FLOPs "
            "divided by the projection's roofline time, so each point sits on "
            "the roof its shapes imply."
        ),
    )
    parser.add_argument(
        "--profile-json",
        type=Path,
        help="JSON list of {name, time_us, count} kernel records",
    )


def _add_series_arguments(parser: argparse.ArgumentParser) -> None:
    """Which markers to draw and how each series matches the profile."""
    parser.add_argument(
        "--top-kernels",
        nargs="?",
        type=int,
        const=_DEFAULT_TOP_KERNELS,
        default=None,
        metavar="N",
        help=(
            "Plot the N kernels with the most kernel time and profile FLOPs, "
            "plus end to end, instead of one marker per category. "
            "A kernel with no FLOPs does not use one of the N slots. "
            f"{_DEFAULT_TOP_KERNELS} when N is omitted. Requires a profile."
        ),
    )
    parser.add_argument(
        "--kernels",
        action="append",
        default=[],
        metavar="PHASE|SERIES|SUBSTRING",
        help=(
            "Restrict a series in one phase to kernel names containing "
            "SUBSTRING. Repeat to add alternatives. Needed when one capture "
            "mixes phases and the kernel name does not say which."
        ),
    )
    parser.add_argument(
        "--time-us",
        action="append",
        default=[],
        metavar="PHASE|SERIES|MICROSECONDS",
        help=(
            "Use this one-forward time instead of the summed kernel time. "
            "The numerator is then the projection's one-forward FLOPs."
        ),
    )


def _projection_regime(entry: str) -> Regime:
    phase, sep, rest = entry.partition(":")
    concurrency, sep2, path = rest.partition(":")
    if not sep or not sep2:
        raise ValueError(f"--projection must be PHASE:CONCURRENCY:PATH, got {entry!r}")
    try:
        concurrency_i = int(concurrency)
    except ValueError as exc:
        raise ValueError(
            f"--projection concurrency must be an int, got {entry!r}"
        ) from exc
    return regime_from_projection(Path(path), phase, concurrency_i)


def _load_cli_arch(args: argparse.Namespace) -> dict[str, Any]:
    """The arch JSON when given, else the named arch spec."""
    if args.gpu_arch_json_path is not None:
        return json.loads(args.gpu_arch_json_path.read_text(encoding="utf-8"))
    if args.gpu_arch_name:
        return load_arch(args.gpu_arch_name)
    raise ValueError("pass --gpu-arch-name or --gpu-arch-json-path")


def _load_cli_regimes(args: argparse.Namespace) -> list[Regime]:
    """Augmentation regimes, then one per ``--projection``."""
    regimes: list[Regime] = []
    if args.augmentation is not None:
        payload = json.loads(args.augmentation.read_text(encoding="utf-8"))
        regimes.extend(regimes_from_augmentation(payload))
    regimes.extend(_projection_regime(entry) for entry in args.projection)
    if not regimes:
        raise ValueError("pass --augmentation or at least one --projection")
    return regimes


def _load_cli_kernels(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Profile kernels from ``--profile-json``, else ``--profile-dir``."""
    if args.profile_json is not None:
        return load_profile_json(args.profile_json)
    if args.profile_dir is not None:
        return load_profile_kernels(args.profile_dir)
    raise ValueError("pass --profile-dir or --profile-json")


@dataclass(frozen=True)
class _CliPlot:
    """What ``main`` plots: the points, their warnings, and how to draw them."""

    points: list[RooflinePoint]
    warnings: list[str]
    series: tuple[Series, ...]
    kernels: list[dict[str, Any]] | None
    ylabel: str


def _cli_plot(args: argparse.Namespace, regimes: list[Regime]) -> _CliPlot:
    """Points from the projection alone, or joined to the profile's kernels."""
    if args.model_only:
        if args.top_kernels is not None:
            raise ValueError("--top-kernels needs a profile")
        points, warnings = points_from_model(regimes, series=DEFAULT_SERIES)
        return _CliPlot(
            points, warnings, DEFAULT_SERIES, None, "TFLOP/s per GPU at the roofline"
        )
    kernels = _load_cli_kernels(args)
    series = DEFAULT_SERIES
    if args.top_kernels is not None:
        series = top_kernel_series(kernels, args.top_kernels)
    includes = _parse_keyed_flag(args.kernels, value_is_float=False)
    overrides = _parse_keyed_flag(args.time_us, value_is_float=True)
    points, warnings = build_points(
        regimes,
        kernels,
        series=series,
        includes=includes,
        time_overrides_us=overrides,
    )
    return _CliPlot(points, warnings, series, kernels, _PROFILE_YLABEL)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        arch = _load_cli_arch(args)
        if args.model_only and (
            args.profile_dir is not None or args.profile_json is not None
        ):
            raise ValueError("--model-only does not read a profile")
        regimes = _load_cli_regimes(args)
        plan = _cli_plot(args, regimes)
        points, roof_warnings = flag_points_above_roof(plan.points, arch)
        warnings = plan.warnings + roof_warnings
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"e2e_roofline: {exc}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"e2e_roofline: {warning}", file=sys.stderr)
    if not points:
        print("e2e_roofline: no points to plot", file=sys.stderr)
        return 1
    plot_e2e_roofline(
        points,
        arch,
        regimes,
        title=args.title,
        output_path=args.output,
        series=plan.series,
        kernels=plan.kernels,
        ylabel=plan.ylabel,
    )
    for point in points:
        print(
            f"{point.series} phase={point.phase} concurrency={point.concurrency} "
            f"AI={point.arithmetic_intensity:.2f} FLOP/byte "
            f"{point.achieved_tflops:.2f} TFLOP/s source={point.cost_source}"
        )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
