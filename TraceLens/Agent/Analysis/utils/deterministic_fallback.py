###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Deterministic (no-LLM) bad-trace fallback: detector + `analysis.md` writer.

A graph-under-recorded trace hides its whole workload behind a single
graph-replay launch (`hipGraphLaunch` / `cudaGraphLaunch`) or a torch-compile
region, so per-op decomposition fails and the normal LLM analysis path cannot
run. `check_graph_replay_coverage` reads the already-written
`unified_perf_summary.csv` and returns a structured verdict gating the fallback
(it excludes benign
eager-launch/memcpy plumbing wrappers, which appear in healthy traces too).

On a bad verdict, `render_fallback_report` emits a minimal but
downstream-parser-compatible report recovering only what a graph-collapsed trace
still carries: device-kernel name, time, and %E2E. Every other cell is an em
dash (intentionally unrecoverable, not missing). The output matches the
default-agent `analysis.md` contract that `parse_analysis_md` reads:
top-of-report `report-begin kind=report_mode mode=deterministic-fallback` and
`kind=warning` markers, then per P-item a `#### P{rank}:` heading, a
`reasoning-candidate` marker, an `impact-begin kind=p_item category=unknown`
marker, and a one-value `impact-begin kind=op_row` marker. On multi_trace runs
(a `node` CSV column) each P-item is per node: a `**Node**` line and a `node=`
attr on the p_item marker, under a minimal `## Executive Summary` of per-node
total time.
"""

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from TraceLens.Agent.Analysis.category_analyses.analysis_utils import (
    HEURISTIC_FRACTION_HIGH,
    HEURISTIC_FRACTION_LOW,
    HEURISTIC_FRACTION_MID,
    _KERNEL_NAME_TRUNC_LEN,
    gpu_utilization_from_timeline,
)
from TraceLens.Agent.Analysis.utils.validation_utils import (
    EXEC_TOTAL_COL,
    PCT_OF_NODE_COL,
)
from TraceLens.Reporting.aggregate_perf_reports import NODE_COL

# Real traces have CSV cells far larger than csv's 131072-char default, which
# would crash the trace-quality gate; lift the limit to the platform maximum.
csv.field_size_limit(sys.maxsize)

# Constants

GRAPH_REPLAY_FRACTION_MAX = 0.10
MIN_PITEM_PERCENT_E2E = 0.5  # drop P-items below this %E2E share
MAX_PITEM_COUNT = 40  # defensive hard cap on emitted P-items

# Wrapper prefixes (matched on the left of the first `->`) that hide the whole
# workload behind one replay launch: only these count as the pathology.
_GRAPH_REPLAY_WRAPPER_PREFIXES = ("hipGraphLaunch", "cudaGraphLaunch")
_TORCH_COMPILED_PREFIX = "Torch-Compiled Region"

# Generic launch/memcpy shims that are Synthetic Ops but benign runtime plumbing.
# Excluded from both the graph-replay signal and the
# fallback candidate list.
_PLUMBING_WRAPPER_PREFIXES = frozenset(
    {
        "hipLaunchKernel",
        "hipModuleLaunchKernel",
        "hipMemcpyAsync",
        "cudaLaunchKernel",
        "cudaLaunchKernelExC",
        "cudaMemcpyAsync",
    }
)
_PLUMBING_WRAPPER_STARTSWITH = ("Memcpy", "__amd_rocclr_")

# (data-table percent column, what the P-item gate is a share of)
_PERCENT_LABELS = ("%E2E", "E2E")
_PERCENT_LABELS_BY_NODE = (PCT_OF_NODE_COL, "of all nodes")

_TOTAL_TIME_NOTE = "Total is the sum of node times, not wall time."

_TABLE_HEADER = (
    "| Operation | Args | Kernel Path | Kernel Name | Time (ms) | {pct} | "
    "Count | FLOPS/Byte | Efficiency | Bound |"
)
_TABLE_DIVIDER = "|" + "---|" * 10

# {pct}/{max_pct} are pre-formatted percents; {pct_label} names the recovered
# percent column; {drop_note} is the conditional drop-count sentence (empty when
# nothing was dropped).
_DEGRADED_BANNER = """> **⚠ Degraded (deterministic fallback) report.** This trace was
> graph-under-recorded (graph-replay fraction {pct};
> per-op decomposition failed, so the agentic analysis path did not execute.
> Only **kernel name, time, and {pct_label}** are recoverable from a graph-collapsed
> trace. Shapes, launcher path, quant operands, category, and efficiency are
> not available in this trace. Impact estimates are heuristic.{drop_note}"""


@dataclass
class GraphReplayCoverage:
    graph_under_recorded: bool
    reason: str
    graph_replay_fraction: float


def check_graph_replay_coverage(unified_perf_csv: Path) -> GraphReplayCoverage:
    # Keyed by node (None outside multi_trace) so one node cannot hide in another.
    total_time = defaultdict(float)
    graph_replay_time = defaultdict(float)
    with open(unified_perf_csv, newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            node = row.get(NODE_COL)
            weight = float(row["Kernel Time (µs)_sum"])
            total_time[node] += weight
            if _is_graph_replay_wrapper(row["name"]):
                graph_replay_time[node] += weight

    fractions = {n: graph_replay_time[n] / t for n, t in total_time.items() if t > 0}
    worst = max(fractions, key=fractions.get, default=None)
    graph_replay_fraction = fractions.get(worst, 0.0)
    graph_under_recorded = graph_replay_fraction > GRAPH_REPLAY_FRACTION_MAX

    if graph_under_recorded:
        reason = (
            f"graph-replay fraction {graph_replay_fraction:.1%} > "
            f"{GRAPH_REPLAY_FRACTION_MAX:.0%}"
        )
        if worst is not None:
            reason += f" in node {worst}"
    else:
        reason = "graph-replay fraction within bounds"

    return GraphReplayCoverage(
        graph_under_recorded=graph_under_recorded,
        reason=reason,
        graph_replay_fraction=graph_replay_fraction,
    )


def render_fallback_report(unified_perf_csv: Path, graph_replay_fraction: float) -> str:
    with open(unified_perf_csv, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = _surviving_rows(reader)
    by_node = NODE_COL in reader.fieldnames

    # Denominator is built over all surviving rows; the display filter below must
    # not feed back into it, or dropping the tail would inflate the survivors.
    group_time = defaultdict(float)
    group_percent = defaultdict(float)
    group_count = defaultdict(int)
    for row in rows:
        key = row["node"], row["op"]
        group_time[key] += row["weight"]
        group_percent[key] += row["percent"]
        group_count[key] += row["count"]
    total_time = sum(group_time.values())
    gate_percent = group_percent
    if by_node:
        # T_all gates and ranks; each node's own T is the displayed share.
        utilization = gpu_utilization_from_timeline(
            pd.read_csv(unified_perf_csv.with_name("gpu_timeline.csv"))
        )
        total_time = utilization["total_time_ms"] * 1000.0
        gate_percent = {g: t / total_time * 100.0 for g, t in group_time.items()}
        group_percent = {
            g: t / (utilization["nodes"][g[0]]["total_time_ms"] * 1000.0) * 100.0
            for g, t in group_time.items()
        }
    percent_label, gate_label = _PERCENT_LABELS_BY_NODE if by_node else _PERCENT_LABELS

    groups = sorted(group_time, key=group_time.get, reverse=True)
    filtered = [g for g in groups if gate_percent[g] >= MIN_PITEM_PERCENT_E2E]
    filtered = filtered[:MAX_PITEM_COUNT]
    dropped = len(groups) - len(filtered)

    drop_note = ""
    if dropped > 0:
        drop_note = (
            f"\n> Dropped {dropped} P-items below {MIN_PITEM_PERCENT_E2E}% {gate_label} "
            f"(or beyond top-{MAX_PITEM_COUNT})."
        )
    banner = _DEGRADED_BANNER.format(
        pct=f"{graph_replay_fraction:.0%}",
        max_pct=f"{GRAPH_REPLAY_FRACTION_MAX:.0%}",
        pct_label=percent_label,
        drop_note=drop_note,
    )
    lines = [
        "# Deterministic Fallback Analysis",
        "",
        "<!-- report-begin kind=report_mode mode=deterministic-fallback -->",
        "<!-- report-end -->",
        "",
        "<!-- report-begin kind=warning -->",
        banner,
        "<!-- report-end -->",
        "",
    ]
    if by_node:
        columns = [utilization, *utilization["nodes"].values()]
        lines += [
            "## Executive Summary",
            "",
            f"| Metric | {EXEC_TOTAL_COL} | " + " | ".join(utilization["nodes"]) + " |",
            "|---" * (len(columns) + 1) + "|",
            "| Total Time | "
            + " | ".join(f"{c['total_time_ms']:.2f} ms" for c in columns)
            + " |",
            "",
            _TOTAL_TIME_NOTE,
            "",
            "## Compute Kernel Optimizations",
            "",
        ]

    for rank, (node, op) in enumerate(filtered, start=1):
        display = _normalize_kernel_name(op)
        group_pct = (
            (group_time[node, op] / total_time * 100.0) if total_time > 0 else 0.0
        )
        low = round(group_pct * HEURISTIC_FRACTION_LOW, 2)
        mid = round(group_pct * HEURISTIC_FRACTION_MID, 2)
        high = round(group_pct * HEURISTIC_FRACTION_HIGH, 2)

        lines.append(f"<!-- reasoning-candidate tier=compute rank={rank} -->")
        lines.append(f"#### P{rank}: {display}")
        node_attr = ""
        if node is not None:
            lines.append(f"**Node**: {node}")
            node_attr = f" node={node}"
        lines.append(
            "<!-- impact-begin kind=p_item category=unknown "
            f"low={low} mid={mid} high={high}{node_attr} -->"
        )
        lines.append("<!-- impact-end -->")
        lines.append("")
        lines.append("**Data:**")
        lines.append("")
        lines.append(_TABLE_HEADER.format(pct=percent_label))
        lines.append(_TABLE_DIVIDER)
        time_ms = group_time[node, op] / 1000.0
        lines.append(
            f"| — | — | — | {op} | {time_ms:.3f} | "
            f"{group_percent[node, op]:.2f} | {group_count[node, op]} | — | — | "
            f"— |"
        )
        lines.append(f"<!-- impact-begin kind=op_row rank={rank} impacts={mid} -->")
        lines.append("<!-- impact-end -->")
        lines.append("")

    return "\n".join(lines) + "\n"


def _is_graph_replay_wrapper(name: str) -> bool:
    if "->" not in name:
        return False
    prefix = name.split("->", 1)[0]
    return prefix in _GRAPH_REPLAY_WRAPPER_PREFIXES or prefix.startswith(
        _TORCH_COMPILED_PREFIX
    )


def _surviving_rows(csv_rows):
    rows = []
    for row in csv_rows:
        name = row.get("name", "")
        if "->" in name:
            prefix, tail = name.split("->", 1)
            if _is_plumbing_wrapper(prefix):
                continue
            op = _strip_synthetic_suffix(tail)
        else:
            op = _strip_synthetic_suffix(name)
        if _is_plumbing_wrapper(op):
            continue
        rows.append(
            {
                "op": op,
                "node": row.get(NODE_COL),
                "weight": float(row["Kernel Time (µs)_sum"]),
                "percent": float(row["Percentage (%)"]),
                "count": int(float(row.get("operation_count", 0) or 0)),
            }
        )
    return rows


def _is_plumbing_wrapper(prefix: str) -> bool:
    return prefix in _PLUMBING_WRAPPER_PREFIXES or prefix.startswith(
        _PLUMBING_WRAPPER_STARTSWITH
    )


def _strip_synthetic_suffix(name: str) -> str:
    suffix = " (Synthetic Op)"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name.strip()


def _normalize_kernel_name(raw: str) -> str:
    # Display-only shortening for the P-item heading; truncate long device symbols
    # but never demangle (the matched symbol stays raw) and never return empty (an
    # empty heading title fails the downstream heading match and drops the block).
    s = raw.strip()
    if len(s) > _KERNEL_NAME_TRUNC_LEN:
        s = s[:_KERNEL_NAME_TRUNC_LEN] + "..."
    if not s:
        return raw
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-perf-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-replay-fraction", type=float, default=None)
    args = parser.parse_args()

    # Fallback handling for displaying fraction when script invoked directly
    frac = args.graph_replay_fraction
    if frac is None:
        frac = check_graph_replay_coverage(args.unified_perf_csv).graph_replay_fraction

    report = render_fallback_report(args.unified_perf_csv, frac)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
