#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Verify ``kernel_candidates.json`` against a mixed steady-state trace and perf CSVs.

The trace is the ``mixed_steady_state`` split the analysis ran on. Events come
from ``DataLoader.load_data``. Kernel categories and the communication filter
are the same ones ``perf_report_sanity_check`` uses. Summed kernel time counts
overlapped launches twice; TraceLens percentages use exposed GPU time, so those
shares are reported side by side.

Usage::

    python -m TraceLens.Agent.Analysis.triage.verify_kernel_candidates \\
        --analysis-dir Magpie/results/.../analysis_output
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from TraceLens.util import GPU_KERNEL_CATEGORIES, TraceEventUtils

from .checks import _events_of, _load_trace_json, resolve_trace_path

# Constants
TOP_RUNTIME_FRACTION = 0.95
EXPOSED_PCT_ABS_TOLERANCE = 2.0
EXPOSED_PCT_REL_TOLERANCE = 0.5
DISPLAY_TOP_KERNELS = 20
MISMATCH_DISPLAY_LIMIT = 15
_MIXED_STEADY_STATE = "mixed_steady_state"
_EMPTY_SHAPE_TOKENS = frozenset(
    {"", "nan", "none", "null", "()", "[]", "-", "—", "–", "n/a", "unknown"}
)
_NP_NUMBER_RE = re.compile(r"np\.(?:float|int)\d*\(([^)]+)\)")
_SHAPE_RE = re.compile(r"\([^()]*\)")
_PROFILE_PATH_RE = re.compile(r"--profile_json_path\s+(\S+)")
_GPU_EVENT_CATEGORIES = frozenset(GPU_KERNEL_CATEGORIES)


def scan_trace_kernels(events: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Sum duration and count launches for each GPU kernel, memcpy, and memset."""
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "dur_us": 0.0}
    )
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in _GPU_EVENT_CATEGORIES:
            continue
        name = str(event.get("name") or "")
        if not name:
            continue
        dur = event.get("dur")
        totals[name]["count"] += 1
        if isinstance(dur, (int, float)):
            totals[name]["dur_us"] += float(dur)
    return dict(totals)


def _column(fieldnames: list[str] | None, marker: str, suffix: str = "") -> str:
    for name in fieldnames or []:
        if marker in name and name.endswith(suffix):
            return name
    raise ValueError(f"CSV is missing a {marker!r}{suffix} column")


def load_gpu_timeline(path: Path) -> dict[str, float]:
    """Return ``gpu_timeline.csv`` rows as milliseconds keyed by ``type``."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    timeline: dict[str, float] = {}
    for row in rows:
        kind = str(row.get("type") or "").strip()
        if kind:
            timeline[kind] = float(row.get("time ms") or 0.0)
    if "busy_time" not in timeline:
        raise ValueError(f"gpu_timeline.csv has no busy_time row: {path}")
    return timeline


def parse_kernel_details(cell: str) -> list[dict[str, Any]]:
    """Parse a ``kernel_details_summary`` cell into name/count/duration dicts."""
    text = str(cell or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    text = _NP_NUMBER_RE.sub(r"\1", text)
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict) and item.get("name")]


def load_unified_rows(path: Path) -> tuple[list[dict[str, str]], str]:
    """Load ``unified_perf_summary.csv`` and the kernel-time sum column name."""
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        time_col = _column(reader.fieldnames, "Kernel Time", "_sum")
        rows = list(reader)
    return rows, time_col


def kernel_detail_totals(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Sum device-kernel count and duration across unified-perf rows."""
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "dur_us": 0.0}
    )
    for row in rows:
        for item in parse_kernel_details(row.get("kernel_details_summary", "")):
            name = str(item["name"])
            totals[name]["count"] += float(item.get("count") or 0)
            totals[name]["dur_us"] += float(item.get("total_duration_us") or 0.0)
    return dict(totals)


def _float_cell(row: dict[str, str], column: str) -> float:
    try:
        return float(row.get(column) or 0.0)
    except ValueError:
        return 0.0


def shapes_missing(input_dims: str) -> bool:
    """Return True when ``Input Dims`` has no numeric tensor shape."""
    text = str(input_dims or "").strip()
    if text.lower() in _EMPTY_SHAPE_TOKENS:
        return True
    return not any(
        re.search(r"\d", match.group(0)) for match in _SHAPE_RE.finditer(text)
    )


def shape_tokens(input_dims: str) -> list[str]:
    """Numeric tensor shapes from an ``Input Dims`` cell, whitespace removed."""
    tokens: list[str] = []
    for match in _SHAPE_RE.finditer(str(input_dims or "")):
        body = match.group(0)
        if re.search(r"\d", body):
            tokens.append(re.sub(r"\s+", "", body))
    return tokens


def _similar(left: float, right: float, abs_tol: float, rel_tol: float) -> bool:
    return abs(left - right) <= max(abs_tol, rel_tol * max(abs(left), abs(right)))


def _duration_similar(trace_us: float, csv_us: float, busy_us: float) -> bool:
    """True when summed durations agree, allowing a small accounting gap."""
    diff = abs(trace_us - csv_us)
    scale = max(trace_us, csv_us, 1.0)
    return diff <= max(0.05 * scale, 0.002 * busy_us)


def _pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def load_kernel_candidates(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"kernel_candidates.json is not an object: {path}")
    return payload


def index_candidates(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Index hot kernels by op name and by device-kernel name."""
    by_op: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_device: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not payload:
        return by_op, by_device
    for item in payload.get("hot_kernels") or []:
        if not isinstance(item, dict):
            continue
        op_name = str(item.get("name") or "")
        if op_name:
            by_op[op_name].append(item)
            by_device[op_name].append(item)
        device_names = list(item.get("device_kernel_names") or [])
        one = str(item.get("device_kernel_name") or "")
        if one:
            device_names.append(one)
        for device_name in device_names:
            if device_name:
                by_device[device_name].append(item)
    return by_op, by_device


def candidate_shape_blob(items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in items:
        for key in ("shapes", "input_shapes"):
            raw = item.get(key)
            if not isinstance(raw, list):
                continue
            for entry in raw:
                if isinstance(entry, dict):
                    parts.append(str(entry.get("shape") or ""))
                else:
                    parts.append(str(entry))
    return re.sub(r"\s+", "", " ".join(parts))


def candidate_gpu_pct(items: list[dict[str, Any]]) -> float:
    total = 0.0
    for item in items:
        value = item.get("gpu_pct")
        if value is None:
            value = item.get("percent_of_total")
        try:
            total += float(value or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def check_gpu_time(
    rows: list[dict[str, str]],
    time_col: str,
    timeline: dict[str, float],
) -> dict[str, Any]:
    """Summed kernel time must cover exposed GPU busy time."""
    kernel_us = sum(_float_cell(row, time_col) for row in rows)
    busy_us = timeline["busy_time"] * 1e3
    computation_us = timeline.get("computation_time", 0.0) * 1e3
    return {
        "name": "gpu_time",
        "status": "pass" if kernel_us + 1.0 >= busy_us else "fail",
        "kernel_time_us": kernel_us,
        "busy_time_us": busy_us,
        "computation_time_us": computation_us,
        "kernel_over_busy": kernel_us / busy_us if busy_us else None,
    }


def check_kernel_counts(
    trace_kernels: dict[str, dict[str, float]],
    csv_kernels: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Compare per-kernel launch counts in the trace and the unified CSV."""
    trace_total = int(sum(item["count"] for item in trace_kernels.values()))
    csv_total = int(sum(item["count"] for item in csv_kernels.values()))
    names = set(trace_kernels) | set(csv_kernels)
    mismatches = []
    for name in names:
        trace_count = int(trace_kernels.get(name, {}).get("count", 0))
        csv_count = int(csv_kernels.get(name, {}).get("count", 0))
        if trace_count != csv_count:
            mismatches.append(
                {
                    "name": name,
                    "trace_count": trace_count,
                    "csv_count": csv_count,
                    "diff": trace_count - csv_count,
                    "communication": TraceEventUtils.is_communication_string(name),
                }
            )
    mismatches.sort(key=lambda item: abs(item["diff"]), reverse=True)
    other = [item for item in mismatches if not item["communication"]]
    return {
        "name": "kernel_counts",
        "status": "pass" if not other else "fail",
        "trace_kernel_events": trace_total,
        "csv_kernel_launches": csv_total,
        "unique_trace_names": len(trace_kernels),
        "unique_csv_names": len(csv_kernels),
        "mismatch_count": len(mismatches),
        "communication_mismatch_count": len(mismatches) - len(other),
        "mismatches": mismatches[:MISMATCH_DISPLAY_LIMIT],
    }


def _ranked_kernels(
    trace_kernels: dict[str, dict[str, float]],
) -> list[tuple[str, dict[str, float]]]:
    return sorted(
        trace_kernels.items(), key=lambda item: item[1]["dur_us"], reverse=True
    )


def check_ranked_kernels(
    trace_kernels: dict[str, dict[str, float]],
    csv_kernels: dict[str, dict[str, float]],
    by_device: dict[str, list[dict[str, Any]]],
    *,
    busy_us: float,
    candidates_present: bool,
) -> dict[str, Any]:
    """Rank trace kernels by summed duration and compare shares."""
    ranked = _ranked_kernels(trace_kernels)
    trace_total = sum(item["dur_us"] for item in trace_kernels.values())
    csv_total = sum(item["dur_us"] for item in csv_kernels.values())
    covered = 0.0
    rows: list[dict[str, Any]] = []
    missing_csv = 0
    missing_candidates = 0
    sum_mismatch = 0
    exposed_warnings = 0
    for name, stats in ranked:
        covered += stats["dur_us"]
        in_top = (
            _pct(covered - stats["dur_us"], trace_total) < TOP_RUNTIME_FRACTION * 100.0
        )
        csv_stats = csv_kernels.get(name)
        candidates = by_device.get(name) or []
        trace_sum_pct = _pct(stats["dur_us"], trace_total)
        trace_busy_pct = _pct(stats["dur_us"], busy_us)
        csv_us = None if csv_stats is None else csv_stats["dur_us"]
        csv_sum_pct = _pct(csv_us, csv_total) if csv_us is not None else None
        csv_busy_pct = _pct(csv_us, busy_us) if csv_us is not None else None
        gpu_pct = candidate_gpu_pct(candidates) if candidates else None
        sum_ok = csv_us is None or _duration_similar(stats["dur_us"], csv_us, busy_us)
        exposed_ok = gpu_pct is None or _similar(
            trace_busy_pct,
            gpu_pct,
            EXPOSED_PCT_ABS_TOLERANCE,
            EXPOSED_PCT_REL_TOLERANCE,
        )
        if in_top and csv_stats is None:
            missing_csv += 1
        if in_top and candidates_present and not candidates:
            missing_candidates += 1
        if in_top and csv_stats is not None and not sum_ok:
            sum_mismatch += 1
        if in_top and gpu_pct is not None and not exposed_ok:
            exposed_warnings += 1
        if in_top:
            rows.append(
                {
                    "name": name,
                    "trace_us": stats["dur_us"],
                    "trace_count": int(stats["count"]),
                    "trace_pct_of_sum": trace_sum_pct,
                    "trace_pct_of_busy": trace_busy_pct,
                    "in_csv": csv_stats is not None,
                    "csv_us": csv_us,
                    "csv_pct_of_sum": csv_sum_pct,
                    "csv_pct_of_busy": csv_busy_pct,
                    "in_candidates": bool(candidates),
                    "candidate_gpu_pct": gpu_pct,
                    "sum_pct_similar": sum_ok,
                    "exposed_pct_similar": exposed_ok,
                }
            )
        if _pct(covered, trace_total) >= TOP_RUNTIME_FRACTION * 100.0 and not in_top:
            break
    status = "pass"
    if missing_csv or sum_mismatch or (candidates_present and missing_candidates):
        status = "fail"
    elif not candidates_present:
        status = "fail"
    return {
        "name": "ranked_kernels",
        "status": status,
        "candidates_present": candidates_present,
        "top_fraction": TOP_RUNTIME_FRACTION,
        "top_kernel_count": len(rows),
        "missing_from_csv": missing_csv,
        "missing_from_candidates": missing_candidates,
        "sum_pct_mismatches": sum_mismatch,
        "exposed_pct_warnings": exposed_warnings,
        "kernels": rows,
    }


def check_top_rows(
    rows: list[dict[str, str]],
    time_col: str,
    by_op: dict[str, list[dict[str, Any]]],
    *,
    candidates_present: bool,
) -> dict[str, Any]:
    """Top unified rows covering 95% of kernel time must be in the candidates."""
    total = sum(_float_cell(row, time_col) for row in rows)
    ordered = sorted(rows, key=lambda row: _float_cell(row, time_col), reverse=True)
    covered = 0.0
    selected: list[dict[str, Any]] = []
    for row in ordered:
        row_us = _float_cell(row, time_col)
        if total and _pct(covered, total) >= TOP_RUNTIME_FRACTION * 100.0:
            break
        covered += row_us
        op_name = str(row.get("name") or "")
        matches = by_op.get(op_name) or []
        tokens = shape_tokens(row.get("Input Dims", ""))
        blob = candidate_shape_blob(matches)
        missing_shapes = [token for token in tokens if token not in blob]
        selected.append(
            {
                "name": op_name,
                "kernel_time_us": row_us,
                "pct_of_kernel_time": _pct(row_us, total),
                "in_candidates": bool(matches),
                "input_dims_missing": shapes_missing(row.get("Input Dims", "")),
                "missing_shapes": missing_shapes,
            }
        )
    missing_ops = sum(1 for item in selected if not item["in_candidates"])
    shape_gaps = sum(
        1
        for item in selected
        if item["in_candidates"]
        and (item["input_dims_missing"] or item["missing_shapes"])
    )
    if not candidates_present:
        status = "fail"
    elif missing_ops or shape_gaps:
        status = "fail"
    else:
        status = "pass"
    return {
        "name": "top_rows",
        "status": status,
        "candidates_present": candidates_present,
        "top_fraction": TOP_RUNTIME_FRACTION,
        "row_count": len(selected),
        "kernel_time_us": total,
        "covered_us": covered,
        "missing_ops": missing_ops,
        "rows_missing_shapes": shape_gaps,
        "rows": selected,
    }


def check_csv_quality(rows: list[dict[str, str]], time_col: str) -> dict[str, Any]:
    """Measure unified rows that lack shapes or a perf model."""
    total = sum(_float_cell(row, time_col) for row in rows)
    missing_shape_rows = 0
    missing_shape_us = 0.0
    missing_model_rows = 0
    missing_model_us = 0.0
    for row in rows:
        row_us = _float_cell(row, time_col)
        if shapes_missing(row.get("Input Dims", "")):
            missing_shape_rows += 1
            missing_shape_us += row_us
        model = str(row.get("has_perf_model") or "").strip().lower()
        if model not in {"true", "1", "yes"}:
            missing_model_rows += 1
            missing_model_us += row_us
    return {
        "name": "csv_quality",
        "status": "info",
        "row_count": len(rows),
        "kernel_time_us": total,
        "missing_shapes": {
            "rows": missing_shape_rows,
            "kernel_time_us": missing_shape_us,
            "fraction_of_kernel_time": missing_shape_us / total if total else 0.0,
        },
        "missing_perf_model": {
            "rows": missing_model_rows,
            "kernel_time_us": missing_model_us,
            "fraction_of_kernel_time": missing_model_us / total if total else 0.0,
        },
    }


def _recorded_trace(analysis_dir: Path) -> Path | None:
    """Trace path the analysis recorded, via the triage resolver or the perf log."""
    recorded = resolve_trace_path(str(analysis_dir))
    if recorded and os.path.isfile(recorded):
        return Path(recorded)
    log_path = analysis_dir / "perf_report.log"
    if not log_path.is_file():
        return None
    match = _PROFILE_PATH_RE.search(
        log_path.read_text(encoding="utf-8", errors="replace")
    )
    if not match:
        return None
    candidate = Path(match.group(1))
    return candidate if candidate.is_file() else None


def _mixed_candidates(directories: list[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(f"{_MIXED_STEADY_STATE}*.json.gz")):
            if path not in seen:
                seen.add(path)
                found.append(path)
    return found


def resolve_mixed_steady_state_trace(
    analysis_dir: Path,
) -> tuple[Path | None, Path | None]:
    """Return the mixed steady-state trace and the path the analysis recorded.

    When the analysis recorded ``mixed_steady_state`` directly, that file is
    the trace. Otherwise the mixed split sitting next to the recorded trace is
    used. A raw profile is not a substitute.
    """
    recorded = _recorded_trace(analysis_dir)
    if recorded is not None and _MIXED_STEADY_STATE in recorded.name:
        return recorded, recorded
    directories: list[Path] = []
    if recorded is not None:
        directories.append(recorded.parent)
    for root in (analysis_dir, analysis_dir.parent):
        for pattern in ("trace_split", "split*", "splits*"):
            directories.extend(Path(path) for path in glob.glob(str(root / pattern)))
    matches = _mixed_candidates(directories)
    if recorded is not None:
        same_dir = [path for path in matches if path.parent == recorded.parent]
        if len(same_dir) == 1:
            return same_dir[0], recorded
    if len(matches) == 1:
        return matches[0], recorded
    return None, recorded


def discover_inputs(
    analysis_dir: Path,
    *,
    trace: Path | None = None,
    csv_dir: Path | None = None,
    kernel_candidates: Path | None = None,
) -> dict[str, Path | None]:
    """Resolve the mixed steady-state trace, CSVs, and optional candidate file."""
    csv_root = csv_dir or (analysis_dir / "perf_report_csvs")
    unified = csv_root / "unified_perf_summary.csv"
    timeline = csv_root / "gpu_timeline.csv"
    if not unified.is_file():
        raise FileNotFoundError(f"unified_perf_summary.csv not found: {unified}")
    if not timeline.is_file():
        raise FileNotFoundError(f"gpu_timeline.csv not found: {timeline}")
    candidates = kernel_candidates
    if candidates is None:
        for relative in (
            "kernel_candidates.json",
            "hyperloom_kernel_candidates/kernel_candidates.json",
        ):
            candidate = analysis_dir / relative
            if candidate.is_file():
                candidates = candidate
                break
    recorded: Path | None
    if trace is not None:
        if not trace.is_file():
            raise FileNotFoundError(f"trace not found: {trace}")
        resolved_trace = trace
        recorded = trace
    else:
        resolved_trace, recorded = resolve_mixed_steady_state_trace(analysis_dir)
    if resolved_trace is None:
        recorded_note = f" Recorded trace: {recorded}." if recorded else ""
        raise FileNotFoundError(
            f"no {_MIXED_STEADY_STATE} trace for {analysis_dir}.{recorded_note} "
            "Pass --trace to name one."
        )
    return {
        "analysis_dir": analysis_dir,
        "trace": resolved_trace,
        "recorded_trace": recorded,
        "unified": unified,
        "timeline": timeline,
        "kernel_candidates": (
            candidates if candidates and candidates.is_file() else None
        ),
    }


def verify(
    *,
    trace: Path,
    unified: Path,
    timeline_path: Path,
    kernel_candidates: Path | None,
    recorded_trace: Path | None = None,
) -> dict[str, Any]:
    """Run the five checks and return a JSON-serializable report."""
    rows, time_col = load_unified_rows(unified)
    timeline = load_gpu_timeline(timeline_path)
    events = _events_of(_load_trace_json(str(trace)))
    trace_kernels = scan_trace_kernels(events)
    csv_kernels = kernel_detail_totals(rows)
    payload = load_kernel_candidates(kernel_candidates)
    by_op, by_device = index_candidates(payload)
    busy_us = timeline["busy_time"] * 1e3
    checks = [
        check_gpu_time(rows, time_col, timeline),
        check_kernel_counts(trace_kernels, csv_kernels),
        check_ranked_kernels(
            trace_kernels,
            csv_kernels,
            by_device,
            busy_us=busy_us,
            candidates_present=payload is not None,
        ),
        check_top_rows(rows, time_col, by_op, candidates_present=payload is not None),
        check_csv_quality(rows, time_col),
    ]
    failing = [item["name"] for item in checks if item["status"] == "fail"]
    return {
        "trace": str(trace),
        "recorded_trace": None if recorded_trace is None else str(recorded_trace),
        "unified_perf_summary": str(unified),
        "gpu_timeline": str(timeline_path),
        "kernel_candidates": (
            None if kernel_candidates is None else str(kernel_candidates)
        ),
        "status": "fail" if failing else "pass",
        "failing_checks": failing,
        "checks": checks,
    }


def _short_name(name: str, limit: int = 88) -> str:
    return name if len(name) <= limit else name[: limit - 3] + "..."


def format_report(report: dict[str, Any]) -> str:
    """Render a text summary of one verification report."""
    lines = [
        f"status: {report['status']}",
        f"trace: {report['trace']}",
        f"recorded_trace: {report.get('recorded_trace') or report['trace']}",
        f"unified: {report['unified_perf_summary']}",
        f"kernel_candidates: {report['kernel_candidates'] or '(missing)'}",
        "",
    ]
    for check in report["checks"]:
        name = check["name"]
        lines.append(f"[{check['status']}] {name}")
        if name == "gpu_time":
            lines.append(
                f"  kernel time {check['kernel_time_us']:.1f} us, "
                f"busy {check['busy_time_us']:.1f} us, "
                f"computation {check['computation_time_us']:.1f} us, "
                f"kernel/busy {check['kernel_over_busy']:.3f}"
            )
        elif name == "kernel_counts":
            lines.append(
                f"  trace events {check['trace_kernel_events']}, "
                f"csv launches {check['csv_kernel_launches']}, "
                f"mismatched names {check['mismatch_count']} "
                f"({check['communication_mismatch_count']} communication)"
            )
            for item in check["mismatches"]:
                kind = "comm" if item["communication"] else "kernel"
                lines.append(
                    f"    [{kind}] {_short_name(item['name'])}: "
                    f"trace {item['trace_count']} csv {item['csv_count']}"
                )
        elif name == "ranked_kernels":
            candidate_note = (
                ""
                if check["candidates_present"]
                else ", kernel_candidates.json missing"
            )
            lines.append(
                f"  top {check['top_fraction']:.0%} kernels {check['top_kernel_count']}, "
                f"missing csv {check['missing_from_csv']}, "
                f"missing candidates {check['missing_from_candidates']}, "
                f"duration mismatches {check['sum_pct_mismatches']}, "
                f"exposed-% warnings {check['exposed_pct_warnings']}"
                f"{candidate_note}"
            )
            for item in check["kernels"][:DISPLAY_TOP_KERNELS]:
                csv_us = "n/a" if item["csv_us"] is None else f"{item['csv_us']:.1f}"
                csv_busy = (
                    "n/a"
                    if item["csv_pct_of_busy"] is None
                    else f"{item['csv_pct_of_busy']:.2f}"
                )
                cand_pct = (
                    "n/a"
                    if item["candidate_gpu_pct"] is None
                    else f"{item['candidate_gpu_pct']:.2f}"
                )
                lines.append(
                    f"    {_short_name(item['name'])}: "
                    f"us {item['trace_us']:.1f} vs csv {csv_us}, "
                    f"busy% {item['trace_pct_of_busy']:.2f} vs csv {csv_busy} "
                    f"vs candidate {cand_pct}"
                )
        elif name == "top_rows":
            lines.append(
                f"  rows {check['row_count']} covering {check['top_fraction']:.0%} of "
                f"{check['kernel_time_us']:.1f} us, "
                f"missing ops {check['missing_ops']}, "
                f"shape gaps {check['rows_missing_shapes']}"
            )
            gaps = [
                item
                for item in check["rows"]
                if not item["in_candidates"]
                or item["missing_shapes"]
                or item["input_dims_missing"]
            ]
            for item in gaps[:DISPLAY_TOP_KERNELS]:
                lines.append(
                    f"    {_short_name(item['name'])}: "
                    f"in_candidates={item['in_candidates']} "
                    f"missing_shapes={len(item['missing_shapes'])} "
                    f"input_dims_missing={item['input_dims_missing']}"
                )
        elif name == "csv_quality":
            shapes = check["missing_shapes"]
            models = check["missing_perf_model"]
            lines.append(
                f"  missing shapes {shapes['rows']}/{check['row_count']} rows, "
                f"{shapes['fraction_of_kernel_time']:.1%} of kernel time"
            )
            lines.append(
                f"  missing perf model {models['rows']}/{check['row_count']} rows, "
                f"{models['fraction_of_kernel_time']:.1%} of kernel time"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify kernel_candidates.json against a trace and perf-report CSVs."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        action="append",
        default=[],
        help="Analysis directory containing perf_report_csvs. Repeat for each run.",
    )
    parser.add_argument(
        "--trace", type=Path, default=None, help="Trace JSON or JSON.GZ"
    )
    parser.add_argument(
        "--csv-dir", type=Path, default=None, help="Directory with the perf CSVs"
    )
    parser.add_argument("--kernel-candidates", type=Path, default=None)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the machine-readable report. With multiple analyses, this is a directory.",
    )
    return parser.parse_args(argv)


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.analysis_dir and args.trace is None:
        print("error: pass --analysis-dir or --trace with --csv-dir", file=sys.stderr)
        return 2
    targets: list[dict[str, Path | None]]
    if args.analysis_dir:
        if args.trace is not None and len(args.analysis_dir) != 1:
            print("error: --trace applies to a single --analysis-dir", file=sys.stderr)
            return 2
        targets = [
            discover_inputs(
                analysis_dir,
                trace=args.trace,
                csv_dir=args.csv_dir,
                kernel_candidates=args.kernel_candidates,
            )
            for analysis_dir in args.analysis_dir
        ]
    else:
        if args.csv_dir is None:
            print("error: --csv-dir is required with --trace", file=sys.stderr)
            return 2
        targets = [
            discover_inputs(
                args.csv_dir.parent,
                trace=args.trace,
                csv_dir=args.csv_dir,
                kernel_candidates=args.kernel_candidates,
            )
        ]
    exit_code = 0
    reports: list[dict[str, Any]] = []
    for target in targets:
        report = verify(
            trace=target["trace"],
            unified=target["unified"],
            timeline_path=target["timeline"],
            kernel_candidates=target["kernel_candidates"],
            recorded_trace=target["recorded_trace"],
        )
        reports.append(report)
        print(format_report(report))
        if report["status"] != "pass":
            exit_code = 1
    if args.json_out is not None:
        if len(reports) == 1:
            _write_json(args.json_out, reports[0])
        else:
            args.json_out.mkdir(parents=True, exist_ok=True)
            for report in reports:
                unified_path = Path(str(report["unified_perf_summary"]))
                stem = f"{unified_path.parents[2].name}__{unified_path.parents[1].name}"
                _write_json(args.json_out / f"{stem}.json", report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
