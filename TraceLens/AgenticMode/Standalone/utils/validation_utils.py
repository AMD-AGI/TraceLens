###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Validation utilities for TraceLens AgenticMode.

Three validation levels, each at the boundary where issues are still fixable:

Level 1 — validate_findings_file (Steps 6-7, within each sub-agent)
    Structural check on a single findings file. Sub-agent retries on failure.

Level 2 — validate_subagent_outputs (Step 8, batch)
    Cross-cutting checks that need all files: time sanity, coverage, priority.

Level 3 — validate_report (Step 11.1, after report assembly)
    Final standalone_analysis.md structure: headers, metrics table, placeholders.
"""

import json
import json
import os
import re

from .report_utils import load_manifest, _scan_findings_dir

# ---------------------------------------------------------------------------
# Constants — all validation thresholds and patterns in one place
# ---------------------------------------------------------------------------

# Level 1: findings file structure
_REQUIRED_FINDINGS_HEADERS = ["## Recommendations", "## Detailed Analysis"]
_COMPUTE_P_ITEM_LABELS = ["**Insight**", "**Action**", "**Impact**"]
_SYSTEM_P_ITEM_LABELS = ["**Insight**", "**Action**"]
_P_ITEM_RE = re.compile(r"^### P(\d+):", re.MULTILINE)
_CANDIDATE_RE = re.compile(r"<!-- reasoning-candidate\s+tier=\w+\s+rank=(\d+)\s*-->")

# Markdown table header containing an "Args" column. Match line that starts
# with `|` and has `Args` as one of the column names.
_TABLE_HEADER_RE = re.compile(r"^\|.*\|\s*Args\s*\|.*\|\s*$", re.MULTILINE)

# Level 2: cross-cutting batch checks
_TIME_DISCREPANCY_THRESHOLD = 15  # percent

# Level 3: report structure
_REQUIRED_REPORT_HEADERS = [
    "Executive Summary",
    "Compute Kernel Optimizations",
    "Kernel Fusion Opportunities (Experimental)",
    "System-Level Optimizations",
    "Detailed Analysis",
    "Appendix",
]
_REQUIRED_METRICS_ROWS = [
    "Total Time",
    "Compute %",
    "Idle %",
    "Exposed Communication %",
    "Top Bottleneck Category",
]
_REPORT_PLACEHOLDER_RE = re.compile(r"<[A-Z][a-z_ ]+>")
_KNOWN_REPORT_PLACEHOLDERS = {
    "<Brief Title>",
    "<Library>",
    "<platform>",
    "<model>",
    "<architecture>",
    "<scale>",
    "<precision>",
}


# ---------------------------------------------------------------------------
# Level 1: per-file findings validation (called within each sub-agent)
# ---------------------------------------------------------------------------


def validate_findings_file(filepath, tier):
    """Validate a single findings file against the sub-agent spec contract.

    Checks:
    - Required ## headers present and in correct order
    - P-item labels match tier (compute: Insight/Action/Impact; system: Insight/Action)
    - At least one reasoning-candidate block in Detailed Analysis
    - P-item count matches reasoning-candidate count

    Args:
        filepath: Path to the *_findings.md file
        tier: "compute" or "system"

    Returns:
        Tuple of (passed: bool, errors: list of error strings)
    """
    if not os.path.exists(filepath):
        return False, [f"File not found: {filepath}"]

    with open(filepath, "r") as f:
        content = f.read()

    errors = []

    header_positions = []
    for h in _REQUIRED_FINDINGS_HEADERS:
        pos = content.find(h)
        if pos < 0:
            errors.append(f"Missing required section: {h}")
        header_positions.append(pos)

    if all(p >= 0 for p in header_positions):
        if header_positions[0] > header_positions[1]:
            errors.append("## Recommendations must appear before ## Detailed Analysis")

    rec_start = content.find("## Recommendations")
    da_start = content.find("## Detailed Analysis")

    p_items = []
    if rec_start >= 0:
        rec_end = da_start if da_start > rec_start else len(content)
        rec_section = content[rec_start:rec_end]

        p_items = _P_ITEM_RE.findall(rec_section)
        if not p_items:
            errors.append("No ### P<N>: items found under ## Recommendations")

        expected_labels = (
            _COMPUTE_P_ITEM_LABELS if tier == "compute" else _SYSTEM_P_ITEM_LABELS
        )
        for label in expected_labels:
            if label not in rec_section:
                errors.append(
                    f"Missing label {label} in ## Recommendations "
                    f"(required for {tier} tier)"
                )

        if tier == "system" and "**Impact**" in rec_section:
            errors.append(
                "System-tier ## Recommendations must not contain **Impact** labels"
            )

    candidates = []
    if da_start >= 0:
        da_section = content[da_start:]

        candidates = _CANDIDATE_RE.findall(da_section)
        if not candidates:
            errors.append(
                "No <!-- reasoning-candidate --> blocks in ## Detailed Analysis"
            )

    if p_items and candidates and len(p_items) != len(candidates):
        errors.append(
            f"P-item count ({len(p_items)}) does not match "
            f"reasoning-candidate count ({len(candidates)})"
        )

    # Compute tier only: Args column cells must match operations[].args verbatim
    if tier == "compute":
        errors.extend(_validate_args_column(content, filepath))

    passed = len(errors) == 0
    if passed:
        print("PASS: Findings file is valid")
    else:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")

    return passed, errors


def _scan_args_cells(content):
    """Yield (line_no, cell) for every non-empty Args cell in markdown tables."""
    lines = content.splitlines()
    for header_idx, header_line in enumerate(lines):
        if not _TABLE_HEADER_RE.match(header_line):
            continue
        cols = [c.strip() for c in header_line.strip("|").split("|")]
        try:
            args_col = cols.index("Args")
        except ValueError:
            continue
        # Skip the separator row (|---|...|).
        for row_idx in range(header_idx + 2, len(lines)):
            row = lines[row_idx]
            if not row.strip().startswith("|"):
                break
            cells = [c.strip() for c in row.strip("|").split("|")]
            if args_col < len(cells) and cells[args_col]:
                yield row_idx + 1, cells[args_col]


def _load_valid_args(*metrics_paths):
    """Build set of operations[].args strings from one or more metrics JSONs."""
    valid = set()
    for p in metrics_paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        valid.update(
            op["args"]
            for op in d.get("operations", [])
            if isinstance(op.get("args"), str)
        )
    return valid


def _validate_args_column(content, findings_path):
    """Level-1 check: every Args cell in a compute findings file must match
    operations[].args verbatim in the matching `<cat>_metrics.json`.

    Catches reformat (e.g. `<br>` -> `,` or `x`), summarization, or content
    drift. Skips silently if the metrics JSON is absent or has no `args` field.
    """
    output_dir = os.path.dirname(os.path.dirname(findings_path))
    cat = os.path.basename(findings_path).replace("_findings.md", "")
    metrics_path = os.path.join(output_dir, "category_data", f"{cat}_metrics.json")
    valid_args = _load_valid_args(metrics_path)
    if not valid_args:
        return []
    return [
        f"Args cell on line {ln} does not match operations[].args in "
        f"{cat}_metrics.json (paste verbatim, keep <br>): {cell}"
        for ln, cell in _scan_args_cells(content)
        if cell not in valid_args
    ]


# ---------------------------------------------------------------------------
# Level 2: cross-cutting batch checks (called at Step 8)
# ---------------------------------------------------------------------------


def validate_subagent_outputs(output_dir):
    """Run cross-cutting validation checks on all subagent outputs.

    Returns a dict with check results and prints a summary.
    """
    manifest = load_manifest(output_dir)

    results = {
        "time_check": _check_time_sanity(manifest),
        "coverage_check": _check_coverage(output_dir, manifest),
        "priority_check": _check_priority_consistency(manifest),
    }

    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for check_name, check_result in results.items():
        status = check_result["status"]
        print(f"  {check_name}: {status}")
        for msg in check_result.get("messages", []):
            print(f"    {msg}")
    print("=" * 60)

    return results


def _check_time_sanity(manifest):
    """Verify category GPU kernel time sum approximates computation time."""
    categories = manifest.get("categories", [])
    total_category_time = sum(
        cat.get("gpu_kernel_time_ms", 0)
        for cat in categories
        if cat.get("tier") == "compute_kernel"
    )

    gpu_util = manifest.get("gpu_utilization", {})
    total_time = gpu_util.get("total_time_ms", 0)
    comp_pct = gpu_util.get("computation_time_percent", 0)
    computation_time = total_time * comp_pct / 100

    if computation_time <= 0:
        return {"status": "WARN", "messages": ["No computation time in manifest"]}

    discrepancy = abs(total_category_time - computation_time) / computation_time * 100

    if discrepancy > _TIME_DISCREPANCY_THRESHOLD:
        return {
            "status": "WARN",
            "messages": [
                f"Category sum ({total_category_time:.1f}ms) vs "
                f"Computation time ({computation_time:.1f}ms) = {discrepancy:.1f}% difference"
            ],
        }

    return {"status": "PASS", "messages": []}


def _check_coverage(output_dir, manifest):
    """Verify all expected categories have findings."""
    categories = manifest.get("categories", [])
    messages = []

    found_system = set(_scan_findings_dir(output_dir, "system_findings").keys())
    expected_system = {c["name"] for c in categories if c.get("tier") == "system"}
    missing_system = expected_system - found_system
    if missing_system:
        messages.append(f"Missing system findings: {', '.join(missing_system)}")

    found_compute = set(_scan_findings_dir(output_dir, "category_findings").keys())
    expected_compute = {
        c["name"] for c in categories if c.get("tier") == "compute_kernel"
    }
    missing_compute = expected_compute - found_compute
    if missing_compute:
        messages.append(f"Missing compute findings: {', '.join(missing_compute)}")

    status = "WARN" if messages else "PASS"
    return {"status": status, "messages": messages}


def _check_priority_consistency(manifest):
    """Identify top categories by GPU time for priority verification."""
    categories = manifest.get("categories", [])
    sorted_cats = sorted(
        [c for c in categories if c.get("tier") == "compute_kernel"],
        key=lambda x: x.get("gpu_kernel_time_ms", 0),
        reverse=True,
    )

    top_names = [c["name"] for c in sorted_cats[:3]]
    return {
        "status": "INFO",
        "messages": [
            f"Top 3 by GPU time: {top_names}",
            "Verify these receive P1-P3 in compute kernel recommendations",
        ],
    }


# ---------------------------------------------------------------------------
# Level 3: final report validation (called at Step 11.1)
# ---------------------------------------------------------------------------


def validate_report(output_dir):
    """Validate standalone_analysis.md for structural issues.

    Checks:
    - Required ## section headers in standalone_analysis.md
    - Metrics table under Executive Summary (5 rows, no placeholders)
    - Unfilled template placeholders

    Findings files are validated separately by validate_findings_file
    (Level 1, called within each sub-agent at Steps 6-7).

    Args:
        output_dir: Base output directory containing standalone_analysis.md

    Returns:
        Tuple of (passed: bool, missing: list of error/missing-section strings)
    """
    report_path = os.path.join(output_dir, "standalone_analysis.md")
    if not os.path.exists(report_path):
        return False, ["<file not found>"]

    with open(report_path, "r") as f:
        content = f.read()

    if len(content.strip()) < 100:
        return False, ["<report is empty or too short>"]

    missing = []

    missing.extend(
        f"Missing section: {h}"
        for h in _REQUIRED_REPORT_HEADERS
        if f"## {h}" not in content
    )

    exec_start = content.find("## Executive Summary")
    if exec_start >= 0:
        next_section = content.find("\n## ", exec_start + 1)
        exec_block = (
            content[exec_start:next_section]
            if next_section > 0
            else content[exec_start:]
        )
        if "|" not in exec_block:
            missing.append("No metrics table found under ## Executive Summary")
        else:
            for row_name in _REQUIRED_METRICS_ROWS:
                if row_name not in exec_block:
                    missing.append(f"Missing metrics row: {row_name}")
            for placeholder in ("X ms", "Y%", "Z%", "W%"):
                if (
                    f"| {placeholder}" in exec_block
                    or f"| {placeholder} " in exec_block
                ):
                    missing.append(
                        f"Placeholder value '{placeholder}' in Executive Summary table"
                    )

    found = [
        p
        for p in _REPORT_PLACEHOLDER_RE.findall(content)
        if p in _KNOWN_REPORT_PLACEHOLDERS
    ]
    if found:
        missing.append(f"Unfilled placeholders: {', '.join(sorted(set(found)))}")

    missing.extend(_validate_report_args_column(content, output_dir))

    return len(missing) == 0, missing


def _validate_report_args_column(content, output_dir):
    """Level-3 check: every Args cell in standalone_analysis.md must match some
    operations[].args verbatim across all category metrics JSONs.

    Catches LLM reformatting introduced by the Step 11 orchestrator when it
    pastes per-category Detailed Analysis tables into the final report.
    """
    cat_data_dir = os.path.join(output_dir, "category_data")
    if not os.path.isdir(cat_data_dir):
        return []
    metrics_paths = [
        os.path.join(cat_data_dir, f)
        for f in os.listdir(cat_data_dir)
        if f.endswith("_metrics.json")
    ]
    valid_args = _load_valid_args(*metrics_paths)
    if not valid_args:
        return []
    return [
        f"standalone_analysis.md line {ln}: Args cell does not match any "
        f"operations[].args in category_data/*_metrics.json (paste verbatim, "
        f"keep <br>): {cell}"
        for ln, cell in _scan_args_cells(content)
        if cell not in valid_args
    ]
