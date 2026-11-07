###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""End-to-end roofline plot from a projection and a profile.

The projection supplies compulsory FLOPs and bytes, so the x-axis is arithmetic
intensity of one forward. The profile supplies the time those kernels actually
ran, so the y-axis is achieved TFLOP/s. Hardware roofs come from the GPU arch
spec (``load_arch``, including ``$TL_EXTENSION``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
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


def load_profile_kernels(directory: Path) -> list[dict[str, Any]]:
    """Kernel name, summed GPU time, and launch count from ``*_ops.csv``."""
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
                kernels.append(
                    {
                        "name": row["name"],
                        "time_us": float(raw),
                        "count": int(float(count_raw)) if count_raw else 0,
                    }
                )
    return kernels


def load_profile_json(path: Path) -> list[dict[str, Any]]:
    """``[{name, time_us, count}]`` from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be a list of kernel records")
    kernels = []
    for item in payload:
        kernels.append(
            {
                "name": str(item["name"]),
                "time_us": float(item["time_us"]),
                "count": int(item.get("count") or 0),
            }
        )
    return kernels


def regimes_from_augmentation(payload: dict[str, Any]) -> list[Regime]:
    """Category rollups from ``roofline_augmentation.json``.

    Each category keeps ``flops_per_launch`` so a multi-step capture can scale
    one forward's shape by the launch count.
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
) -> tuple[float, float, float | None]:
    chosen = _bucket_ops(ops, categories)
    total_flops = sum(float(op.get("total_flops") or 0.0) for op in chosen)
    total_bytes = sum(float(op.get("total_bytes") or 0.0) for op in chosen)
    # A category rollup is one row with a per-launch cost. A bucket of many
    # projection rows has no single launch cost; the caller then passes a
    # one-forward profile, or --time-us.
    per_launch = None
    if len(chosen) == 1 and chosen[0].get("flops_per_launch"):
        per_launch = float(chosen[0]["flops_per_launch"])
    return total_flops, total_bytes, per_launch


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
    flops_per_launch: float | None,
    launch_count: int,
    time_us: float,
) -> float:
    """Achieved TFLOP/s while the measured launches ran.

    ``flops_per_launch * launch_count`` is the work in the capture when the
    projection gives one launch's FLOPs. Otherwise the profile window is taken
    as one forward and ``total_flops`` is the numerator.
    """
    if time_us <= 0:
        raise ValueError("measured time must be positive")
    if flops_per_launch and launch_count > 0:
        flops = flops_per_launch * launch_count
    else:
        flops = total_flops
    return flops / (time_us * 1e-6) / 1e12


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
            if spec.categories is None and multi_phase and override is None:
                warnings.append(
                    f"{spec.label} {regime.phase}: mixed capture has no "
                    "whole-forward time (pass --time-us)"
                )
                continue
            total_flops, total_bytes, per_launch = _bucket_cost(
                regime.ops, spec.categories
            )
            if total_flops <= 0 or total_bytes <= 0:
                continue
            matched = _matched_kernels(
                kernels,
                phase=regime.phase,
                categories=spec.categories,
                includes=includes.get(key, ()),
                multi_phase=multi_phase,
            )
            if override is not None:
                time_us = override
                # The override is one forward, so the numerator is the forward.
                per_launch = None
                launch_count = 0
            elif not matched:
                warnings.append(f"{spec.label} {regime.phase}: no profile time")
                continue
            else:
                time_us, launch_count = _measured_launches(matched)
            if time_us <= 0:
                warnings.append(f"{spec.label} {regime.phase}: no profile time")
                continue
            points.append(
                RooflinePoint(
                    series=spec.label,
                    phase=regime.phase,
                    concurrency=regime.concurrency,
                    arithmetic_intensity=total_flops / total_bytes,
                    achieved_tflops=achieved_tflops(
                        total_flops, per_launch, launch_count, time_us
                    ),
                    flops=total_flops,
                    bytes=total_bytes,
                    time_us=time_us,
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
                edgecolors=color,
                linewidths=1.6,
                zorder=3,
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
            "Plot an end-to-end roofline. X and the roofs come from a roofline "
            "projection (or its category rollup). Y is that projection's FLOPs "
            "divided by kernel time from a profile of the same regime."
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
            f"{point.achieved_tflops:.2f} TFLOP/s"
        )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
