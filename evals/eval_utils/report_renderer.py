#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Render PR and fix-ticket markdown reports from report summary JSON."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


BASE_SECTION_ORDER = ["Reasoning", "Kernel Fusion", "Others"]


def _table(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def _read_trace_index(csv_path: Path, suite_name: str) -> Dict[str, Dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    index = {}
    for row in rows:
        trace_id = row.get("id", "").strip()
        if not trace_id:
            continue
        index[trace_id] = {
            "suite": suite_name,
            "trace_path": row.get("trace_path", ""),
            "reference_dir": row.get("reference_dir", ""),
            "platform": row.get("platform", ""),
            "sub_category": row.get("sub_category", ""),
            "test_traces_csv": str(csv_path),
        }
    return index


def _merge_counter_maps(items: Iterable[Dict[str, int]]) -> Counter:
    c = Counter()
    for item in items:
        c.update(item)
    return c


def _render_section_counts(base_counts: Dict[str, int]) -> str:
    ordered_keys = [k for k in BASE_SECTION_ORDER if k in base_counts]
    ordered_keys.extend(sorted(k for k in base_counts.keys() if k not in ordered_keys))
    rows = [[section, str(base_counts.get(section, 0))] for section in ordered_keys]
    return _table(["Section", "Failures"], rows or [["(none)", "0"]])


def _top_n_counter(counter: Counter, n: int) -> List[Tuple[str, int]]:
    return sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:n]


def _short(text: str, max_len: int = 120) -> str:
    value = (text or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 1].rstrip() + "…"


def _render_pr_report(summary: Dict) -> str:
    global_overall = summary["global"]["overall"]
    global_base = summary["global"].get("base_section_counts", {})
    global_standalone = summary["global"].get("standalone_section_counts", {})
    suites = summary.get("suites", {})

    suite_rows = []
    for suite_name in sorted(suites.keys()):
        suite_data = suites[suite_name]
        overall = suite_data.get("overall", {})
        suite_rows.append(
            [
                suite_name,
                str(overall.get("pass", 0)),
                str(overall.get("fail", 0)),
                str(overall.get("missing", 0)),
                _fmt_pct(float(overall.get("overall_pass_rate", 0.0))),
            ]
        )

    standalone_top = _top_n_counter(Counter(global_standalone), 8)
    standalone_rows = [[name, str(count)] for name, count in standalone_top]

    # Aggregate top issues globally.
    issue_counter = Counter()
    for suite_data in suites.values():
        for entry in suite_data.get("top_issue_counts", []):
            issue_counter[entry.get("issue_summary", "Unknown")] += int(entry.get("count", 0))
    top_issues = _top_n_counter(issue_counter, 10)
    top_issue_rows = [[issue, str(count)] for issue, count in top_issues]

    parts = [
        "# Automated Eval Report (PR)",
        "",
        f"Generated at: `{summary.get('generated_at', '')}`",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Overall PASS", str(global_overall.get("pass", 0))],
                ["Overall FAIL", str(global_overall.get("fail", 0))],
                ["Overall MISSING", str(global_overall.get("missing", 0))],
                ["Overall pass rate", _fmt_pct(float(global_overall.get("overall_pass_rate", 0.0)))],
            ],
        ),
        "",
        "## Suite Summary",
        "",
        _table(["Suite", "PASS", "FAIL", "MISSING", "Pass rate"], suite_rows),
        "",
        "## Failure Sections",
        "",
        _render_section_counts(global_base),
        "",
        "## Sections",
        "",
        _table(["Section", "Failures"], standalone_rows or [["(none)", "0"]]),
        "",
        "## Top Failure Issues",
        "",
        _table(["Issue", "Count"], top_issue_rows or [["(none)", "0"]]),
    ]
    return "\n".join(parts) + "\n"


def _render_ticket_report(summary: Dict, trace_index: Dict[str, Dict[str, str]]) -> str:
    suites = summary.get("suites", {})
    global_overall = summary["global"]["overall"]

    parts = [
        "# Automated Eval Report (Fix Ticket)",
        "",
        f"Generated at: `{summary.get('generated_at', '')}`",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Overall PASS", str(global_overall.get("pass", 0))],
                ["Overall FAIL", str(global_overall.get("fail", 0))],
                ["Overall MISSING", str(global_overall.get("missing", 0))],
                ["Overall pass rate", _fmt_pct(float(global_overall.get("overall_pass_rate", 0.0)))],
            ],
        ),
        "",
    ]

    for suite_name in sorted(suites.keys()):
        suite_data = suites[suite_name]
        standalone_counts = suite_data.get("standalone_section_counts", {})
        top_issues = suite_data.get("top_issue_counts", [])
        failure_modes = suite_data.get("failure_modes", [])

        parts.extend(
            [
                f"## Suite: {suite_name}",
                "",
                "### Sections",
                "",
                _table(
                    ["Section", "Failures"],
                    [[name, str(count)] for name, count in _top_n_counter(Counter(standalone_counts), 20)]
                    or [["(none)", "0"]],
                ),
                "",
                "### Top Failure Issues",
                "",
                _table(
                    ["Issue", "Count"],
                    [
                        [entry.get("issue_summary", "Unknown"), str(entry.get("count", 0))]
                        for entry in top_issues[:15]
                    ]
                    or [["(none)", "0"]],
                ),
                "",
                "### Failure Modes (Concise)",
                "",
            ]
        )

        if not failure_modes:
            parts.append("- No failure modes recorded for this suite.")
            parts.append("")
            continue

        mode_rows = []
        for mode in failure_modes[:8]:
            issue = mode.get("issue_summary", "Unknown issue")
            count = mode.get("count", 0)
            likely_cause = _short(mode.get("likely_cause", ""), 110)
            suggested_fix = _short(mode.get("suggested_fix", ""), 110)
            mode_rows.append([issue, str(count), likely_cause, suggested_fix])

        parts.extend(
            [
                _table(
                    ["Issue", "Count", "Likely cause", "Suggested fix"],
                    mode_rows or [["(none)", "0", "", ""]],
                ),
                "",
                "### Top Reproducers",
                "",
            ]
        )

        top_repro_rows = []
        for entry in suite_data.get("top_trace_failures", [])[:5]:
            trace_id = entry.get("trace_id", "")
            trace_fail_count = entry.get("failures", 0)
            trace_meta = trace_index.get(trace_id, {})
            traces_csv = trace_meta.get("test_traces_csv", "<test_traces_csv>")
            platform = trace_meta.get("platform", "unknown")
            cmd = (
                "CONTAINER=<container> NUM_REPEATS=1 MAX_PARALLEL=1 "
                f"TEST_IDS=\"{trace_id}\" TEST_TRACES_CSV=\"{traces_csv}\" "
                "bash evals/eval_scripts/run_repeatability_parallel.sh"
            )
            top_repro_rows.append(
                [
                    trace_id,
                    str(trace_fail_count),
                    platform,
                    f"`{cmd}`",
                ]
            )
        parts.append(
            _table(
                ["Trace/Case", "Failures", "Platform", "Reproducer command"],
                top_repro_rows or [["(none)", "0", "-", "-"]],
            )
        )
        parts.append("")

    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render markdown reports from summary JSON.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--unit-traces-csv", required=True)
    parser.add_argument("--e2e-traces-csv", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text())

    trace_index = {}
    trace_index.update(_read_trace_index(Path(args.unit_traces_csv).resolve(), "unit"))
    trace_index.update(_read_trace_index(Path(args.e2e_traces_csv).resolve(), "e2e"))

    pr_report = _render_pr_report(summary)
    ticket_report = _render_ticket_report(summary, trace_index)

    pr_path = output_dir / "pr_report.md"
    ticket_path = output_dir / "fix_ticket_report.md"
    pr_path.write_text(pr_report)
    ticket_path.write_text(ticket_report)

    print(f"Wrote PR report: {pr_path}")
    print(f"Wrote fix-ticket report: {ticket_path}")


if __name__ == "__main__":
    main()
