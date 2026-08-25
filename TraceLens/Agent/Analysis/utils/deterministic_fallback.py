###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Deterministic (no-LLM) bad-trace fallback: detector + `analysis.md` writer.

A graph-under-recorded trace hides its whole workload behind a single
graph-replay launch (`hipGraphLaunch` / `cudaGraphLaunch`) or a torch-compile
region, so per-op decomposition fails and the normal LLM analysis path cannot
run. `check_graph_replay_coverage` reads the already-written `perf_report_csvs/`
and returns a structured verdict gating the fallback (it excludes benign
eager-launch/memcpy plumbing wrappers, which appear in healthy traces too).

On a bad verdict, `render_fallback_report` emits a minimal but
downstream-parser-compatible report recovering only what a graph-collapsed trace
still carries: device-kernel name, time, and %E2E. Every other cell is an em
dash (intentionally unrecoverable, not missing). The output matches the
default-agent `analysis.md` contract that `parse_analysis_md` reads:
`#### P{rank}:` headings, a `reasoning-candidate` marker, and an
`impact-begin kind=p_item category=unknown` marker per P-item.
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from TraceLens.Agent.Analysis.category_analyses.analysis_utils import (
    HEURISTIC_FRACTION_HIGH,
    HEURISTIC_FRACTION_LOW,
    HEURISTIC_FRACTION_MID,
)

# Giant cells in real traces exceed csv's 131072-char default and crash the
# trace-quality gate; raise the limit to the largest value this platform accepts.
_field_size_limit = 2**31 - 1
while True:
    try:
        csv.field_size_limit(_field_size_limit)
        break
    except OverflowError:
        _field_size_limit //= 2

# Constants

GRAPH_REPLAY_FRACTION_MAX = 0.10

MAX_KERNEL_NAME_LEN = 75  # display truncation length (75 + marker = 78-char cell)
MIN_PITEM_PERCENT_E2E = 0.5  # drop P-items below this %E2E share
MAX_PITEM_COUNT = 40  # defensive hard cap on emitted P-items

_TRUNCATION_MARKER = "..."  # ASCII marker appended to a truncated display name

_WEIGHT_COLUMN = "Kernel Time (µs)_sum"
_PERCENT_COLUMN = "Percentage (%)"
_SYNTHETIC_OP_SUFFIX = " (Synthetic Op)"
_EM_DASH = "—"

# Wrapper prefixes (matched on the left of the first `->`) that hide the whole
# workload behind one replay launch: only these count as the pathology.
_GRAPH_REPLAY_WRAPPER_PREFIXES = ("hipGraphLaunch", "cudaGraphLaunch")
_TORCH_COMPILED_PREFIX = "Torch-Compiled Region"

# Generic launch/memcpy shims that are Synthetic Ops but benign runtime plumbing
# (~0.5% in healthy traces). Excluded from both the graph-replay signal and the
# fallback candidate list.
_PLUMBING_WRAPPER_PREFIXES = frozenset(
    {"hipLaunchKernel", "hipModuleLaunchKernel", "hipMemcpyAsync"}
)
_PLUMBING_WRAPPER_STARTSWITH = ("Memcpy", "__amd_rocclr_")

_TABLE_HEADER = (
    "| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | "
    "Count | FLOPS/Byte | Efficiency | Bound |"
)
_TABLE_DIVIDER = "|" + "---|" * 10

# {pct}/{max_pct} are pre-formatted percents; {drop_note} is the conditional
# drop-count sentence (empty when nothing was dropped).
_DEGRADED_BANNER = """> **⚠ Degraded (deterministic fallback) report.** This trace was
> graph-under-recorded (graph-replay fraction {pct} > {max_pct});
> per-op decomposition failed, so the LLM analysis path could not run.
> Only **kernel name, time, and %E2E** are recoverable from a graph-collapsed
> trace. Shapes, launcher path, quant operands, category, and efficiency were
> **never captured** in this trace. Impact ranges are heuristic.{drop_note}"""


@dataclass
class GraphReplayCoverage:
    graph_under_recorded: bool
    reason: str
    graph_replay_fraction: float
    other_category_fraction: float


def check_graph_replay_coverage(perf_csvs_dir: Path) -> GraphReplayCoverage:
    perf_csvs_dir = Path(perf_csvs_dir)

    total_time = 0.0
    graph_replay_time = 0.0
    with open(
        perf_csvs_dir / "unified_perf_summary.csv", newline="", encoding="utf-8"
    ) as fh:
        for row in csv.DictReader(fh):
            weight = float(row[_WEIGHT_COLUMN])
            total_time += weight
            if _is_graph_replay_wrapper(row["name"]):
                graph_replay_time += weight

    graph_replay_fraction = graph_replay_time / total_time if total_time > 0 else 0.0
    graph_under_recorded = graph_replay_fraction > GRAPH_REPLAY_FRACTION_MAX
    other_category_fraction = _other_category_fraction(perf_csvs_dir)

    if graph_under_recorded:
        reason = (
            f"graph-replay fraction {graph_replay_fraction:.1%} > "
            f"{GRAPH_REPLAY_FRACTION_MAX:.0%} "
            f"(other-category {other_category_fraction:.0%})"
        )
    else:
        reason = "graph-replay fraction within bounds"

    return GraphReplayCoverage(
        graph_under_recorded=graph_under_recorded,
        reason=reason,
        graph_replay_fraction=graph_replay_fraction,
        other_category_fraction=other_category_fraction,
    )


def render_fallback_report(perf_csvs_dir: Path, graph_replay_fraction: float) -> str:
    rows = _load_surviving_rows(Path(perf_csvs_dir))
    rows.sort(key=lambda r: r["weight"], reverse=True)

    # Denominator is built over all surviving rows; the display filter below must
    # not feed back into it, or dropping the tail would inflate the survivors.
    group_time = defaultdict(float)
    for row in rows:
        group_time[row["kernel"]] += row["weight"]
    total_time = sum(group_time.values())

    filtered = [row for row in rows if row["percent"] >= MIN_PITEM_PERCENT_E2E]
    filtered = filtered[:MAX_PITEM_COUNT]
    dropped = len(rows) - len(filtered)

    drop_note = ""
    if dropped > 0:
        drop_note = (
            f"\n> Dropped {dropped} P-items below {MIN_PITEM_PERCENT_E2E}% E2E "
            f"(or beyond top-{MAX_PITEM_COUNT})."
        )
    banner = _DEGRADED_BANNER.format(
        pct=f"{graph_replay_fraction:.0%}",
        max_pct=f"{GRAPH_REPLAY_FRACTION_MAX:.0%}",
        drop_note=drop_note,
    )
    lines = ["# Deterministic Fallback Analysis", "", banner, ""]

    for rank, row in enumerate(filtered, start=1):
        kernel = row["kernel"]
        display = _normalize_kernel_name(kernel)
        group_pct = (group_time[kernel] / total_time * 100.0) if total_time > 0 else 0.0
        low = round(group_pct * HEURISTIC_FRACTION_LOW, 2)
        mid = round(group_pct * HEURISTIC_FRACTION_MID, 2)
        high = round(group_pct * HEURISTIC_FRACTION_HIGH, 2)

        lines.append(f"<!-- reasoning-candidate tier=compute rank={rank} -->")
        lines.append(f"#### P{rank}: {display}")
        lines.append(
            "<!-- impact-begin kind=p_item category=unknown "
            f"low={low} mid={mid} high={high} -->"
        )
        lines.append("")
        lines.append("**Data:**")
        lines.append("")
        lines.append(_TABLE_HEADER)
        lines.append(_TABLE_DIVIDER)
        time_ms = row["weight"] / 1000.0
        lines.append(
            f"| {_EM_DASH} | {_EM_DASH} | {_EM_DASH} | {kernel} | {time_ms:.3f} | "
            f"{row['percent']:.2f} | {_EM_DASH} | {_EM_DASH} | {_EM_DASH} | "
            f"{_EM_DASH} |"
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def _is_graph_replay_wrapper(name: str) -> bool:
    if "->" not in name:
        return False
    prefix = name.split("->", 1)[0]
    return prefix in _GRAPH_REPLAY_WRAPPER_PREFIXES or prefix.startswith(
        _TORCH_COMPILED_PREFIX
    )


def _other_category_fraction(perf_csvs_dir: Path) -> float:
    # Optional corroborating cross-check; best-effort, never fatal.
    path = perf_csvs_dir / "ops_summary_by_category.csv"
    if not path.exists():
        return 0.0
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("op category") == "other":
                    return float(row["Percentage (%)"]) / 100.0
    except (OSError, csv.Error, KeyError, ValueError):
        return 0.0
    return 0.0


def _load_surviving_rows(perf_csvs_dir: Path):
    rows = []
    with open(
        perf_csvs_dir / "unified_perf_summary.csv", newline="", encoding="utf-8"
    ) as fh:
        for row in csv.DictReader(fh):
            name = row.get("name", "")
            if "->" in name:
                prefix, tail = name.split("->", 1)
                if _is_plumbing_wrapper(prefix):
                    continue
                kernel = _strip_synthetic_suffix(tail)
            else:
                kernel = _strip_synthetic_suffix(name)
            if _is_plumbing_wrapper(kernel):
                continue
            rows.append(
                {
                    "kernel": kernel,
                    "weight": float(row[_WEIGHT_COLUMN]),
                    "percent": float(row[_PERCENT_COLUMN]),
                }
            )
    return rows


def _is_plumbing_wrapper(prefix: str) -> bool:
    return prefix in _PLUMBING_WRAPPER_PREFIXES or prefix.startswith(
        _PLUMBING_WRAPPER_STARTSWITH
    )


def _strip_synthetic_suffix(name: str) -> str:
    if name.endswith(_SYNTHETIC_OP_SUFFIX):
        name = name[: -len(_SYNTHETIC_OP_SUFFIX)]
    return name.strip()


def _normalize_kernel_name(raw: str) -> str:
    # Display-only shortening for the P-item heading; truncate long device symbols
    # but never demangle (the matched symbol stays raw) and never return empty (an
    # empty heading title fails the downstream heading match and drops the block).
    s = raw.strip()
    if len(s) > MAX_KERNEL_NAME_LEN:
        s = s[:MAX_KERNEL_NAME_LEN] + _TRUNCATION_MARKER
    if not s:
        return raw
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf-csvs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-replay-fraction", type=float, default=None)
    args = parser.parse_args()

    frac = args.graph_replay_fraction
    if frac is None:
        frac = check_graph_replay_coverage(args.perf_csvs_dir).graph_replay_fraction

    report = render_fallback_report(args.perf_csvs_dir, frac)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
