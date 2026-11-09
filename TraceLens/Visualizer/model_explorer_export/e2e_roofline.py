###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""End-to-end roofline plot from a profile and, where the profile has no cost, a projection.

FLOPs and bytes come from the profile when its ``GFLOPS`` and ``FLOPS/Byte``
columns are filled. A kernel the trace could not model (a GPU-graph launch with
no ``Input Dims``) falls back to the roofline projection. The profile always
supplies the time. Hardware roofs come from the GPU arch spec (``load_arch``,
including ``$TL_EXTENSION``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms, load_arch
from TraceLens.Visualizer.model_explorer_export.roofline_augment import classify_kernel

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
# The two compute roofs the plot always draws when the arch lists them.
_HARDWARE_ROOFS = (("matrix_bf16", "bfloat16"), ("matrix_fp4", "fp4"))
_CONCURRENCY_COLORS = ("#4C78A8", "#F58518", "#111111", "#54A24B", "#E45756")
_ABOVE_ROOF_COLOR = "#D62728"


@dataclass(frozen=True)
class Series:
    """One marker family. ``categories`` None is the whole modeled forward."""

    label: str
    marker: str
    phase: str | None
    categories: tuple[str, ...] | None
    open_marker: bool = False


DEFAULT_SERIES: tuple[Series, ...] = (
    Series("Prefill attention", "o", "prefill", ("Attention",), True),
    Series("Verify attention", "o", "decode", ("Attention",), False),
    Series("Routed MoE", "s", None, ("MoE",), False),
    Series("Whole model", "P", None, None, False),
    Series("End to end", "D", None, None, False),
)


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
    # ``export`` when none did, ``mixed`` when the series spans both.
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
            }
        )
    return kernels


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
                "category": row.get("category"),
                "total_flops": row.get("total_flops") or 0.0,
                "total_bytes": row.get("total_bytes") or 0.0,
                "compute_spec": row.get("compute_spec"),
                "flops_per_launch": row.get("flops") or None,
                "instances": row.get("instances") or 0,
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


def _matches(
    kernel: dict[str, Any],
    *,
    phase: str,
    categories: tuple[str, ...] | None,
    includes: tuple[str, ...],
    multi_phase: bool,
) -> bool:
    name = kernel["name"]
    if is_communication(name):
        return False
    if categories is not None and classify_kernel(name) not in categories:
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
    values = [float(peaks[key]) for key, _label in _HARDWARE_ROOFS if peaks.get(key)]
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


def _scaled_export(
    flops: float,
    nbytes: float,
    instances: int,
    launch_count: int,
    per_launch: float | None,
    *,
    one_forward: bool,
) -> tuple[float, float]:
    """Export FLOPs and bytes for ``launch_count`` launches, or one forward.

    The same scale applies to both, so arithmetic intensity stays the
    projection's. No launches and not one forward is no work: a category whose
    kernels were not in this phase does not contribute.
    """
    if flops <= 0 or nbytes <= 0:
        return 0.0, 0.0
    if one_forward:
        return flops, nbytes
    if launch_count <= 0:
        return 0.0, 0.0
    if instances:
        scale = launch_count / instances
    elif per_launch:
        scale = per_launch * launch_count / flops
    else:
        scale = 1.0
    return flops * scale, nbytes * scale


def _category_work(
    kernels: list[dict[str, Any]],
    flops: float,
    nbytes: float,
    instances: int,
    per_launch: float | None,
    *,
    one_forward: bool,
) -> tuple[float, float, str]:
    """FLOPs, bytes, and which source produced them for one category.

    The profile cost is used only when every matched kernel has it. One kernel
    without ``GFLOPS`` means the profile does not describe the category, so the
    whole category falls back to the export.
    """
    if kernels and all(_has_profile_cost(kernel) for kernel in kernels):
        total_flops = sum(float(kernel["flops"]) for kernel in kernels)
        total_bytes = sum(float(kernel["bytes"]) for kernel in kernels)
        if not one_forward:
            return total_flops, total_bytes, "profile"
        _, launches = _measured_launches(kernels)
        if instances and launches:
            scale = instances / launches
            return total_flops * scale, total_bytes * scale, "profile"
    launch_count = 0
    if kernels and not one_forward:
        _, launch_count = _measured_launches(kernels)
    exported = _scaled_export(
        flops, nbytes, instances, launch_count, per_launch, one_forward=one_forward
    )
    if exported[0] <= 0:
        return 0.0, 0.0, "none"
    return exported[0], exported[1], "export"


def _series_work(
    matched: list[dict[str, Any]],
    ops: tuple[dict[str, Any], ...],
    categories: tuple[str, ...] | None,
    *,
    one_forward: bool,
) -> tuple[float, float, str]:
    """FLOPs and bytes for a series, profile first and export as the fallback."""
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for op in _bucket_ops(ops, categories):
        by_cat.setdefault(str(op.get("category")), []).append(op)
    grouped: dict[str, list[dict[str, Any]]] = {cat: [] for cat in by_cat}
    extra: list[dict[str, Any]] = []
    for kernel in matched:
        cat = classify_kernel(kernel["name"])
        if cat in grouped:
            grouped[cat].append(kernel)
        elif _has_profile_cost(kernel):
            extra.append(kernel)
    flops = 0.0
    nbytes = 0.0
    sources: set[str] = set()
    for cat, rows in by_cat.items():
        total_flops, total_bytes, instances, per_launch = _bucket_cost(
            tuple(rows), None
        )
        part_flops, part_bytes, source = _category_work(
            grouped[cat],
            total_flops,
            total_bytes,
            instances,
            per_launch,
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

    A kernel with no category and no profile cost has nothing to put on either
    axis, so its time is left out rather than diluting the point. The same
    applies to a category the projection does not contain.
    """
    known = {str(op.get("category")) for op in _bucket_ops(ops, categories)}
    kept = []
    for kernel in matched:
        cat = classify_kernel(kernel["name"])
        if _has_profile_cost(kernel) and (
            cat is None or cat in known or categories is None
        ):
            if categories is not None and cat not in categories:
                continue
            kept.append(kernel)
            continue
        if cat in known:
            kept.append(kernel)
    return kept


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
    includes = includes or {}
    time_overrides_us = time_overrides_us or {}
    phases = {regime.phase for regime in regimes}
    multi_phase = len(phases) > 1
    points: list[RooflinePoint] = []
    warnings: list[str] = []
    for regime in regimes:
        for spec in series:
            if spec.phase is not None and spec.phase != regime.phase:
                continue
            key = (regime.phase, spec.label)
            override = time_overrides_us.get(key)
            # End to end in a mixed capture only has phase-tagged kernels, which
            # are not the whole forward. Require an explicit one-forward time.
            # Whole model is the attributed kernels, so it is plotted without one.
            if spec.label == "End to end" and multi_phase and override is None:
                warnings.append(
                    f"{spec.label} {regime.phase}: mixed capture has no "
                    "whole-forward time (pass --time-us)"
                )
                continue
            include = includes.get(key, ())
            matched = _matched_kernels(
                kernels,
                phase=regime.phase,
                categories=spec.categories,
                includes=() if spec.label == "Whole model" else include,
                multi_phase=multi_phase,
            )
            # An include on Whole model adds unscoped kernels to the phase-tagged
            # ones. On any other series the include is the whole match.
            if spec.label == "Whole model" and include:
                seen = {kernel["name"] for kernel in matched}
                for kernel in _matched_kernels(
                    kernels,
                    phase=regime.phase,
                    categories=spec.categories,
                    includes=include,
                    multi_phase=multi_phase,
                ):
                    if kernel["name"] not in seen:
                        matched.append(kernel)
                        seen.add(kernel["name"])
            one_forward = override is not None
            if one_forward:
                time_us = override
            else:
                accounted = _accounted_kernels(matched, regime.ops, spec.categories)
                time_us = sum(kernel["time_us"] for kernel in accounted)
                if (
                    spec.label == "Whole model"
                    and multi_phase
                    and any(
                        kernel_phase(kernel["name"]) is None
                        and not is_communication(kernel["name"])
                        for kernel in kernels
                    )
                    and not include
                ):
                    warnings.append(
                        f"Whole model {regime.phase}: unscoped kernels are left "
                        "out of a mixed capture (pass --kernels)"
                    )
            if time_us <= 0:
                # A series the projection does not contain, and that matched no
                # kernel, is absent rather than missing its time.
                bucket_flops, bucket_bytes, _, _ = _bucket_cost(
                    regime.ops, spec.categories
                )
                if bucket_flops > 0 and bucket_bytes > 0:
                    warnings.append(f"{spec.label} {regime.phase}: no profile time")
                continue
            total_flops, total_bytes, source = _series_work(
                matched,
                regime.ops,
                spec.categories,
                one_forward=one_forward,
            )
            if total_flops <= 0 or total_bytes <= 0:
                continue
            points.append(
                RooflinePoint(
                    series=spec.label,
                    phase=regime.phase,
                    concurrency=regime.concurrency,
                    arithmetic_intensity=total_flops / total_bytes,
                    achieved_tflops=total_flops / (time_us * 1e-6) / 1e12,
                    flops=total_flops,
                    bytes=total_bytes,
                    time_us=time_us,
                    cost_source=source,
                )
            )
    return points, warnings


def _fmt_number(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return f"{value:,.0f}"
    return f"{value:,.1f}"


def plot_e2e_roofline(
    points: list[RooflinePoint],
    arch: dict[str, Any],
    regimes: list[Regime],
    *,
    title: str,
    output_path: Path,
):
    """Draw the log-log roofline and write it to ``output_path``."""
    import matplotlib.pyplot as plt

    bandwidth = bandwidth_tb_s(arch)
    peaks = arch.get("max_achievable_tflops") or {}
    hardware = [
        (key, label, float(peaks[key]))
        for key, label in _HARDWARE_ROOFS
        if peaks.get(key)
    ]
    if not hardware:
        raise ValueError("arch spec has no bfloat16 or fp4 matrix peak")

    intensities = [point.arithmetic_intensity for point in points] or [1.0]
    rates = [point.achieved_tflops for point in points] or [1.0]
    peak_values = [peak for _key, _label, peak in hardware]
    x_lo = min(intensities) / 5.0
    x_hi = max(
        max(intensities) * 5.0, max(peak / bandwidth for peak in peak_values) * 5.0
    )
    y_lo = min(min(rates) / 5.0, bandwidth * x_lo)
    y_hi = max(max(rates) * 5.0, max(peak_values) * 1.4)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, color="0.85")

    xs_line = [x_lo, max(peak_values) / bandwidth]
    ax.plot(
        xs_line,
        [bandwidth * x for x in xs_line],
        color="0.45",
        linewidth=1.6,
        label=f"Bandwidth roof, {_fmt_number(bandwidth)} TB/s",
    )
    for key, label, peak in hardware:
        ridge = ridge_flops_per_byte(peak, bandwidth)
        ax.plot(
            [ridge, x_hi],
            [peak, peak],
            color="0.25" if key == "matrix_bf16" else "0.45",
            linestyle="-" if key == "matrix_bf16" else "--",
            linewidth=1.3,
            label=(
                f"{label} roof, {_fmt_number(peak)} TFLOP/s, "
                f"ridge {_fmt_number(ridge)} FLOP/byte"
            ),
        )

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
        ridge = ridge_flops_per_byte(peak, bandwidth)
        other = max(
            (share, spec) for spec, share in shares.items() if spec != "matrix_bf16"
        )
        share, spec = other
        short = spec.removeprefix("matrix_")
        ax.plot(
            [max(ridge, x_lo), x_hi],
            [peak, peak],
            linestyle="-.",
            linewidth=1.0,
            label=(
                f"End-to-end blend at concurrency {regime.concurrency}, "
                f"{share * 100:.0f}% {short} by FLOP, "
                f"{_fmt_number(peak)} TFLOP/s, ridge {_fmt_number(ridge)} FLOP/byte"
            ),
        )

    concurrencies = sorted({point.concurrency for point in points})
    colors = {
        concurrency: _CONCURRENCY_COLORS[index % len(_CONCURRENCY_COLORS)]
        for index, concurrency in enumerate(concurrencies)
    }
    by_series: dict[str, list[RooflinePoint]] = {}
    for point in points:
        by_series.setdefault(point.series, []).append(point)
    series_by_label = {spec.label: spec for spec in DEFAULT_SERIES}
    for label, group in by_series.items():
        spec = series_by_label.get(label)
        marker = spec.marker if spec else "o"
        open_marker = spec.open_marker if spec else False
        for point in group:
            color = colors[point.concurrency]
            ax.scatter(
                [point.arithmetic_intensity],
                [point.achieved_tflops],
                marker=marker,
                s=60,
                facecolors="none" if open_marker else color,
                # A point above the roof is a modeling or matching error, not a
                # measurement, so it is outlined in red instead of drawn plain.
                edgecolors=_ABOVE_ROOF_COLOR if point.above_roof else color,
                linewidths=2.6 if point.above_roof else 1.6,
                zorder=4 if point.above_roof else 3,
            )
        # Arrows follow concurrency inside one phase. A prefill point and a
        # decode point are different work, so they are not connected.
        by_phase: dict[str, list[RooflinePoint]] = {}
        for point in group:
            by_phase.setdefault(point.phase, []).append(point)
        for phase_points in by_phase.values():
            ordered = sorted(phase_points, key=lambda item: item.concurrency)
            for start, end in zip(ordered, ordered[1:]):
                ax.annotate(
                    "",
                    xy=(end.arithmetic_intensity, end.achieved_tflops),
                    xytext=(start.arithmetic_intensity, start.achieved_tflops),
                    arrowprops={"arrowstyle": "->", "color": "0.65", "lw": 1.0},
                    zorder=2,
                )

    handles, labels = ax.get_legend_handles_labels()
    for spec in DEFAULT_SERIES:
        if spec.label not in by_series:
            continue
        handles.append(
            plt.Line2D(
                [],
                [],
                marker=spec.marker,
                linestyle="None",
                markerfacecolor="none" if spec.open_marker else "0.2",
                markeredgecolor="0.2",
                markersize=8,
            )
        )
        labels.append(spec.label)
    if any(point.above_roof for point in points):
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=_ABOVE_ROOF_COLOR,
                markeredgewidth=2.2,
                markersize=8,
            )
        )
        labels.append("Above the roof: FLOPs, bytes, or kernels do not match")
    for concurrency in concurrencies:
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markerfacecolor=colors[concurrency],
                markeredgecolor=colors[concurrency],
                markersize=8,
            )
        )
        labels.append(f"Concurrency {concurrency}")
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    ax.set_xlabel("Arithmetic intensity, FLOP per byte of compulsory HBM traffic")
    ax.set_ylabel("TFLOP/s per GPU while the operation runs")
    ax.set_title(title)
    fig.tight_layout()
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
            "those FLOPs divided by kernel time from the same profile."
        ),
    )
    parser.add_argument("--title", required=True, help="Figure title")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="PNG path to write",
    )
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
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Directory of perf-report *_ops.csv files",
    )
    parser.add_argument(
        "--profile-json",
        type=Path,
        help="JSON list of {name, time_us, count} kernel records",
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
    return parser


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.gpu_arch_json_path is not None:
            arch = json.loads(args.gpu_arch_json_path.read_text(encoding="utf-8"))
        elif args.gpu_arch_name:
            arch = load_arch(args.gpu_arch_name)
        else:
            raise ValueError("pass --gpu-arch-name or --gpu-arch-json-path")
        regimes: list[Regime] = []
        if args.augmentation is not None:
            payload = json.loads(args.augmentation.read_text(encoding="utf-8"))
            regimes.extend(regimes_from_augmentation(payload))
        regimes.extend(_projection_regime(entry) for entry in args.projection)
        if not regimes:
            raise ValueError("pass --augmentation or at least one --projection")
        if args.profile_json is not None:
            kernels = load_profile_json(args.profile_json)
        elif args.profile_dir is not None:
            kernels = load_profile_kernels(args.profile_dir)
        else:
            raise ValueError("pass --profile-dir or --profile-json")
        includes = _parse_keyed_flag(args.kernels, value_is_float=False)
        overrides = _parse_keyed_flag(args.time_us, value_is_float=True)
        points, warnings = build_points(
            regimes, kernels, includes=includes, time_overrides_us=overrides
        )
        points, roof_warnings = flag_points_above_roof(points, arch)
        warnings.extend(roof_warnings)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"e2e_roofline: {exc}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"e2e_roofline: {warning}", file=sys.stderr)
    if not points:
        print("e2e_roofline: no points to plot", file=sys.stderr)
        return 1
    plot_e2e_roofline(points, arch, regimes, title=args.title, output_path=args.output)
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
