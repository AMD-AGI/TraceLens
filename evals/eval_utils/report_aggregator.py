#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Aggregate eval outputs into section-level summary JSON."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _load_rules(path: Path) -> Dict:
    raw = path.read_text()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw)
    except Exception:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Rule file must decode to an object: {path}")
    return data


def _matches(rule: Dict, row: Dict[str, str], text: str) -> bool:
    def _check(field: str, value: str) -> bool:
        regex = rule.get(field)
        if not regex:
            return True
        return re.search(regex, value or "") is not None

    return (
        _check("eval_index_regex", row.get("eval_index", ""))
        and _check("eval_category_regex", row.get("eval_category", ""))
        and _check("issue_summary_regex", row.get("issue_summary", ""))
        and _check("details_regex", row.get("details", ""))
        and _check("text_regex", text)
    )


def _classify_section(
    rules: List[Dict], row: Dict[str, str], text: str, default_value: str
) -> str:
    for rule in rules:
        if _matches(rule, row, text):
            section = rule.get("section")
            if section:
                return str(section)
    return default_value


def _match_failure_mode(mode_rules: List[Dict], text: str, defaults: Dict) -> Tuple[str, str]:
    for rule in mode_rules:
        regex = rule.get("match_regex")
        if regex and re.search(regex, text):
            return (
                str(rule.get("likely_cause", defaults.get("likely_cause", ""))),
                str(rule.get("suggested_fix", defaults.get("suggested_fix", ""))),
            )
    return (
        str(defaults.get("likely_cause", "Unknown likely cause.")),
        str(defaults.get("suggested_fix", "No suggested fix configured.")),
    )


def _parse_pass_rate(summary_value: str) -> float:
    m = re.search(r"\((\d+)%\)", summary_value or "")
    return float(m.group(1)) if m else 0.0


def _suite_metrics(
    suite: str,
    aggregate_root: Path,
    rules: Dict,
) -> Dict:
    aggregated_rows = _read_csv(aggregate_root / "aggregated_results.csv")
    pass_rate_rows = _read_csv(aggregate_root / "pass_rate_summary.csv")
    stream_rows = _read_csv(aggregate_root / "stream_diagnostics.csv")

    defaults = rules.get("defaults", {})
    base_rules = rules.get("base_section_rules", [])
    standalone_rules = rules.get("standalone_section_rules", [])
    mode_rules = rules.get("failure_mode_rules", [])

    pass_count = sum(1 for r in aggregated_rows if r.get("result") == "PASS")
    fail_count = sum(1 for r in aggregated_rows if r.get("result") == "FAIL")
    missing_count = sum(1 for r in aggregated_rows if r.get("result") == "MISSING")
    eval_total = pass_count + fail_count
    overall_pass_rate = (100.0 * pass_count / eval_total) if eval_total else 0.0

    failure_rows = [r for r in aggregated_rows if r.get("result") == "FAIL"]
    base_sections = Counter()
    standalone_sections = Counter()
    issue_counter = Counter()
    trace_counter = Counter()
    failure_modes = defaultdict(
        lambda: {
            "count": 0,
            "base_sections": Counter(),
            "standalone_sections": Counter(),
            "traces": Counter(),
            "eval_indices": Counter(),
            "likely_cause": "",
            "suggested_fix": "",
        }
    )

    for row in failure_rows:
        text = " | ".join(
            [
                row.get("eval_index", ""),
                row.get("issue_summary", ""),
                row.get("details", ""),
            ]
        )
        base_section = _classify_section(
            base_rules,
            row,
            text,
            str(defaults.get("base_section", "Others")),
        )
        standalone_section = _classify_section(
            standalone_rules,
            row,
            text,
            str(defaults.get("standalone_section", "Uncategorized")),
        )
        likely_cause, suggested_fix = _match_failure_mode(mode_rules, text, defaults)

        issue = row.get("issue_summary") or row.get("eval_index") or "Unknown issue"
        trace_id = row.get("trace_id", "unknown_trace")
        eval_index = row.get("eval_index", "")

        base_sections[base_section] += 1
        standalone_sections[standalone_section] += 1
        issue_counter[issue] += 1
        trace_counter[trace_id] += 1

        mode = failure_modes[issue]
        mode["count"] += 1
        mode["base_sections"][base_section] += 1
        mode["standalone_sections"][standalone_section] += 1
        mode["traces"][trace_id] += 1
        mode["eval_indices"][eval_index] += 1
        mode["likely_cause"] = likely_cause
        mode["suggested_fix"] = suggested_fix

    per_trace_pass_rates = []
    for row in pass_rate_rows:
        trace_id = row.get("trace_id", "")
        overall = row.get("overall_pass_rate", "")
        per_trace_pass_rates.append(
            {
                "trace_id": trace_id,
                "overall_pass_rate": overall,
                "overall_pass_percent": _parse_pass_rate(overall),
            }
        )

    stream_outcomes = Counter(row.get("outcome", "unknown") for row in stream_rows)
    last_steps = Counter(row.get("last_step_reached", "none") for row in stream_rows)
    report_written = sum(
        1
        for row in stream_rows
        if str(row.get("report_written", "")).strip().lower() in ("true", "1", "yes")
    )

    mode_entries = []
    for issue, data in failure_modes.items():
        mode_entries.append(
            {
                "issue_summary": issue,
                "count": data["count"],
                "base_sections": dict(data["base_sections"]),
                "standalone_sections": dict(data["standalone_sections"]),
                "traces": dict(data["traces"]),
                "eval_indices": dict(data["eval_indices"]),
                "likely_cause": data["likely_cause"],
                "suggested_fix": data["suggested_fix"],
            }
        )
    mode_entries.sort(key=lambda x: (-x["count"], x["issue_summary"]))

    per_trace_pass_rates.sort(key=lambda x: x["trace_id"])
    top_issue_counts = [
        {"issue_summary": issue, "count": count}
        for issue, count in issue_counter.most_common(25)
    ]
    top_trace_failures = [
        {"trace_id": trace_id, "failures": count}
        for trace_id, count in trace_counter.most_common(25)
    ]

    return {
        "suite": suite,
        "paths": {
            "aggregate_root": str(aggregate_root),
            "aggregated_results_csv": str(aggregate_root / "aggregated_results.csv"),
            "pass_rate_summary_csv": str(aggregate_root / "pass_rate_summary.csv"),
            "stream_diagnostics_csv": str(aggregate_root / "stream_diagnostics.csv"),
        },
        "overall": {
            "pass": pass_count,
            "fail": fail_count,
            "missing": missing_count,
            "rows_total": len(aggregated_rows),
            "eval_total": eval_total,
            "overall_pass_rate": round(overall_pass_rate, 2),
        },
        "stream": {
            "runs_total": len(stream_rows),
            "outcomes": dict(stream_outcomes),
            "last_step_reached": dict(last_steps),
            "report_written": report_written,
        },
        "base_section_counts": dict(base_sections),
        "standalone_section_counts": dict(standalone_sections),
        "top_issue_counts": top_issue_counts,
        "top_trace_failures": top_trace_failures,
        "per_trace_pass_rates": per_trace_pass_rates,
        "failure_modes": mode_entries,
    }


def _parse_suite_arg(values: Iterable[str]) -> Dict[str, Path]:
    result = {}
    for item in values:
        if "=" not in item:
            raise ValueError(
                f"Invalid --suite-aggregate '{item}'. Expected format: <suite>=<path>"
            )
        suite, raw_path = item.split("=", 1)
        result[suite.strip()] = Path(raw_path).resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate eval outputs into report JSON.")
    parser.add_argument(
        "--suite-aggregate",
        action="append",
        default=[],
        help="Suite aggregate mapping in form <suite>=<aggregate_dir>. Can be repeated.",
    )
    parser.add_argument(
        "--rules-file",
        default=str(Path(__file__).resolve().parent / "report_section_rules.yaml"),
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    suite_paths = _parse_suite_arg(args.suite_aggregate)
    if not suite_paths:
        raise ValueError("At least one --suite-aggregate mapping is required.")

    rules = _load_rules(Path(args.rules_file).resolve())

    suite_reports = {}
    global_base = Counter()
    global_standalone = Counter()
    total_pass = total_fail = total_missing = total_rows = 0

    for suite, agg_path in sorted(suite_paths.items()):
        suite_data = _suite_metrics(suite, agg_path, rules)
        suite_reports[suite] = suite_data
        global_base.update(suite_data.get("base_section_counts", {}))
        global_standalone.update(suite_data.get("standalone_section_counts", {}))

        overall = suite_data.get("overall", {})
        total_pass += int(overall.get("pass", 0))
        total_fail += int(overall.get("fail", 0))
        total_missing += int(overall.get("missing", 0))
        total_rows += int(overall.get("rows_total", 0))

    eval_total = total_pass + total_fail
    overall_pass_rate = (100.0 * total_pass / eval_total) if eval_total else 0.0
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules_file": str(Path(args.rules_file).resolve()),
        "global": {
            "overall": {
                "pass": total_pass,
                "fail": total_fail,
                "missing": total_missing,
                "rows_total": total_rows,
                "eval_total": eval_total,
                "overall_pass_rate": round(overall_pass_rate, 2),
            },
            "base_section_counts": dict(global_base),
            "standalone_section_counts": dict(global_standalone),
        },
        "suites": suite_reports,
    }

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"Wrote summary JSON: {output_path}")


if __name__ == "__main__":
    main()
