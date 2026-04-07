#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Acceptance checks for automated eval report outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def _require(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated eval report outputs.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--pr-report", required=True)
    parser.add_argument("--ticket-report", required=True)
    parser.add_argument(
        "--expected-suites",
        default="",
        help="Comma-separated suite names expected in summary JSON (e.g. unit,e2e).",
    )
    args = parser.parse_args()

    errors: List[str] = []
    summary_path = Path(args.summary_json).resolve()
    pr_path = Path(args.pr_report).resolve()
    ticket_path = Path(args.ticket_report).resolve()

    _require(summary_path.exists(), f"Missing summary JSON: {summary_path}", errors)
    _require(pr_path.exists(), f"Missing PR report: {pr_path}", errors)
    _require(ticket_path.exists(), f"Missing ticket report: {ticket_path}", errors)

    if errors:
        raise SystemExit("\n".join(errors))

    summary = json.loads(summary_path.read_text())
    pr_text = pr_path.read_text()
    ticket_text = ticket_path.read_text()

    _require("global" in summary, "summary JSON missing 'global'", errors)
    _require("suites" in summary, "summary JSON missing 'suites'", errors)

    global_data = summary.get("global", {})
    _require("overall" in global_data, "summary JSON missing global.overall", errors)
    _require(
        "base_section_counts" in global_data,
        "summary JSON missing global.base_section_counts",
        errors,
    )
    _require(
        "standalone_section_counts" in global_data,
        "summary JSON missing global.standalone_section_counts",
        errors,
    )

    suites = summary.get("suites", {})
    expected = [s.strip() for s in args.expected_suites.split(",") if s.strip()]
    for suite in expected:
        _require(suite in suites, f"Expected suite '{suite}' missing in summary JSON", errors)

    for suite_name, suite_data in suites.items():
        _require(
            "base_section_counts" in suite_data,
            f"Suite '{suite_name}' missing base_section_counts",
            errors,
        )
        _require(
            "standalone_section_counts" in suite_data,
            f"Suite '{suite_name}' missing standalone_section_counts",
            errors,
        )
        _require(
            "failure_modes" in suite_data,
            f"Suite '{suite_name}' missing failure_modes",
            errors,
        )

    _require(
        "## Failure Sections" in pr_text,
        "PR report missing '## Failure Sections' heading",
        errors,
    )
    _require(
        "## Suite:" in ticket_text,
        "Ticket report missing suite sections",
        errors,
    )
    _require(
        "run_repeatability_parallel.sh" in ticket_text,
        "Ticket report missing reproducer commands",
        errors,
    )

    if errors:
        raise SystemExit("\n".join(errors))

    print("Validation succeeded.")


if __name__ == "__main__":
    main()
