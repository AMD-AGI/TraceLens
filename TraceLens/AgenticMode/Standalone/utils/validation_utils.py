###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Validation utilities for TraceLens AgenticMode.

Three validation levels, each at the boundary where issues are still fixable:

Level 1 — validate_findings_file (Steps 6-7, within each sub-agent)
    Structural check on a single findings file, including marker structure
    via MarkerValidator (pairing, kind attributes, per-kind required attrs,
    mandatory p_item). Sub-agent retries on failure.

Level 2 — validate_subagent_outputs (Step 8, batch)
    Cross-cutting checks that need all files: time sanity, coverage, priority.

Level 3 — validate_report (Step 11.1, after report assembly)
    Final standalone_analysis.md structure: headers, metrics table, placeholders,
    and report-level marker structure via MarkerValidator (pairing, kind
    attributes, mandatory top_ops).
"""

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
# Match optional severity emoji prefix (same convention as report Template sections).
_P_ITEM_RE = re.compile(r"^### (?:\S+\s+)?P(\d+):", re.MULTILINE)
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

# Marker enforcement: shared scoping helpers used by MarkerValidator at both
# Level 1 (findings files) and Level 3 (assembled report).
_NOT_QUANTIFIABLE_SENTINEL = re.compile(
    r"not quantifiable from trace data", re.IGNORECASE
)
# Matches both the findings convention `### P<N>:` and the report convention
# `### 🔴 P<N>:` (template uses an emoji severity prefix in the assembled
# report). The optional `\S+\s+` token covers the emoji + space.
_P_ITEM_HEADING_RE = re.compile(r"^### (?:\S+\s+)?P\d+:.*$", re.MULTILINE)
# Report sections that get strict 1:1 ### P<N>: ↔ kind=p_item enforcement.
# System-Level is excluded — system P-items have no per-card **Impact** field
# by template (see standalone_analysis_template.md).
_REPORT_STRICT_P_ITEM_HEADERS = (
    "## Compute Kernel Optimizations",
    "## Kernel Fusion Opportunities (Experimental)",
)
# Detailed Analysis subsections whose `#### ` blocks must each carry either a
# kind=detail_estimate marker or the not-quantifiable sentinel.
_REPORT_DETAIL_ESTIMATE_SUBSECTIONS = (
    "### Compute Kernel Insights",
    "### Kernel Fusion Insights",
)

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
    - Compute tier only: Args column cells match operations[].args verbatim
    - Marker structure: pairing, kind= attribute, per-kind required attrs,
      no mixed null/numeric values, mandatory kind=p_item (except for
      triton_findings.md)
    - Per-card kind=p_item 1:1 for compute findings and
      kernel_fusion_findings.md (every ### P<N>: card carries exactly
      one marker)
    - Per-reasoning-candidate kind=detail_estimate marker OR explicit
      "not quantifiable from trace data" sentinel string

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

        rel_findings = os.path.basename(filepath)
        # kernel_fusion_findings.md uses compute-style P-items (Insight/Action/Impact + markers).
        fusion_strict = rel_findings == "kernel_fusion_findings.md"
        expected_labels = (
            _COMPUTE_P_ITEM_LABELS
            if tier == "compute" or fusion_strict
            else _SYSTEM_P_ITEM_LABELS
        )
        for label in expected_labels:
            if label not in rec_section:
                errors.append(
                    f"Missing label {label} in ## Recommendations "
                    f"(required for {tier} tier)"
                )

        if tier == "system" and "**Impact**" in rec_section and not fusion_strict:
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

    # Marker structure (folded in from the former Level-4 validate_markers).
    file_class = "category_findings" if tier == "compute" else "system_findings"
    errors.extend(MarkerValidator.check_findings_file(filepath, file_class))

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
    - Args column cells match operations[].args verbatim
    - Report-level marker structure: pairing, kind= attribute, per-kind
      required attrs, mandatory kind=top_ops
    - Per-section kind=p_item 1:1 in Compute Kernel Optimizations and
      Kernel Fusion Opportunities (System-Level intentionally excluded)
    - Per-#### block kind=detail_estimate marker OR not-quantifiable
      sentinel under Compute Kernel Insights and Kernel Fusion Insights
      subsections of Detailed Analysis

    Note: the marker checks are intentionally pre-extension. Step 11.2's
    agent_extension.py consumes these markers when it rewrites the report
    into ms-savings form, so re-running validate_report after Step 11.2
    would always fail by design.

    Findings files are validated separately by validate_findings_file
    (Level 1, called within each sub-agent at Steps 6-7), which also
    enforces per-file marker structure.

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

    # Report-level marker structure
    missing.extend(MarkerValidator.check_report(report_path))

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


# ---------------------------------------------------------------------------
# MarkerValidator
#
# Owns all `<!-- impact-begin ... -->` / `<!-- impact-end -->` marker config
# and checks. Used by validate_findings_file (per-file findings markers) and
# validate_report (report markers). .
# ---------------------------------------------------------------------------


class MarkerValidator:
    """All marker (`<!-- impact-begin ... -->` / `<!-- impact-end -->`) checks.

    Stateless: regexes and kind tables are class-level constants and the
    entry points are classmethods.

    Two layers of enforcement:

    - `scan` does whole-file marker hygiene (pairing, kind=, required
      attrs, no mixed null/numeric).
    - Per-card / per-block / per-candidate scoping enforces a 1:1
      contract between visible cards (### P<N>:, #### blocks under
      Detailed Analysis, <!-- reasoning-candidate --> blocks) and their
      paired markers, so a missing marker on any single card or block
      raises an error rather than being silently absorbed by the
      "at least one marker" check.
    """

    BEGIN_RE = re.compile(r"<!--\s*impact-begin\s+([^>]*?)-->", re.DOTALL)
    END_RE = re.compile(r"<!--\s*impact-end\s*-->")
    KIND_ATTR_RE = re.compile(r"\bkind=(\w+)\b")
    ATTR_RE = re.compile(r"\b(\w+)=([^\s]+)")

    KNOWN_KINDS = {"p_item", "detail_estimate", "top_ops"}
    REQUIRED_ATTRS_BY_KIND = {
        "p_item": ("low", "mid", "high"),
        "detail_estimate": ("low", "high"),
    }
    # Compute findings files that do NOT need a p_item marker.
    COMPUTE_NO_P_ITEM = {"triton_findings.md"}
    # System findings files that get strict 1:1 ### P<N>: ↔ kind=p_item
    # enforcement (same contract as compute findings). Other system findings
    # keep the looser "at least one kind=p_item" rule because their P-items
    # are intentionally markerless per template.
    STRICT_P_ITEM_SYSTEM_FILES = {"kernel_fusion_findings.md"}

    @classmethod
    def scan(cls, text, rel):
        """Common per-file marker scan.

        Checks marker pairing, presence of kind= attribute, kind ∈
        KNOWN_KINDS, per-kind required attrs, and no mixed null/numeric
        values in low/mid/high. Returns (errors, seen_kinds).
        """
        errors = []
        begins = cls.BEGIN_RE.findall(text)
        n_end = len(cls.END_RE.findall(text))
        if len(begins) != n_end:
            errors.append(
                f"{rel}: marker pairing mismatch ({len(begins)} begin vs {n_end} end)"
            )

        seen_kinds = set()
        for inner in begins:
            kind_m = cls.KIND_ATTR_RE.search(inner)
            if not kind_m:
                errors.append(
                    f"{rel}: impact-begin missing kind= attribute: {inner.strip()}"
                )
                continue
            kind = kind_m.group(1)
            seen_kinds.add(kind)
            if kind not in cls.KNOWN_KINDS:
                errors.append(f"{rel}: unknown kind={kind}")
                continue
            attrs = dict(cls.ATTR_RE.findall(inner))
            for required in cls.REQUIRED_ATTRS_BY_KIND.get(kind, ()):
                if required not in attrs:
                    errors.append(
                        f"{rel}: kind={kind} missing required attr {required}"
                    )
            nums = [attrs.get(a) for a in ("low", "mid", "high") if a in attrs]
            if nums:
                null_ct = sum(1 for v in nums if v == "null")
                if 0 < null_ct < len(nums):
                    errors.append(
                        f"{rel}: kind={kind} mixes null and numeric values "
                        f"in low/mid/high"
                    )

        return errors, seen_kinds

    @classmethod
    def check_findings_file(cls, path, file_class):
        """Marker checks for a sub-agent findings file.

        file_class must be "category_findings" or "system_findings". On top
        of `scan`, runs:

        - Compute findings (excluding triton): strict 1:1 ### P<N>: ↔
          kind=p_item per card.
        - System findings in STRICT_P_ITEM_SYSTEM_FILES (kernel_fusion):
          same strict 1:1 enforcement.
        - Other system findings: keep the looser "at least one kind=p_item"
          rule (their P-items are markerless by template).
        - All tiers: per-reasoning-candidate detail_estimate-or-sentinel.
        """
        rel = os.path.basename(path)
        with open(path) as f:
            text = f.read()
        errors, seen_kinds = cls.scan(text, rel)

        is_compute_strict = (
            file_class == "category_findings" and rel not in cls.COMPUTE_NO_P_ITEM
        )
        is_system_strict = (
            file_class == "system_findings" and rel in cls.STRICT_P_ITEM_SYSTEM_FILES
        )

        if is_compute_strict or is_system_strict:
            errors.extend(cls._check_p_item_per_card(text, rel))
        elif file_class == "system_findings":
            if "p_item" not in seen_kinds:
                errors.append(f"{rel}: missing required kind=p_item")

        errors.extend(cls._check_detail_estimate_per_candidate(text, rel))
        return errors

    @classmethod
    def check_report(cls, path):
        """Marker checks for the assembled standalone_analysis.md.

        On top of `scan`, runs:

        - Mandatory kind=top_ops marker.
        - Strict 1:1 ### P<N>: ↔ kind=p_item per section in
          _REPORT_STRICT_P_ITEM_HEADERS (Compute Kernel + Kernel Fusion).
        - Per-#### block detail_estimate-or-sentinel under each subsection
          in _REPORT_DETAIL_ESTIMATE_SUBSECTIONS (Compute Kernel Insights +
          Kernel Fusion Insights).

        These are intentionally pre-extension checks: agent_extension.py
        consumes the markers when it rewrites the report into ms-savings
        form, so re-running validate_report after Step 11.2 would always
        fail by design.
        """
        rel = os.path.basename(path)
        with open(path) as f:
            text = f.read()
        errors, seen_kinds = cls.scan(text, rel)
        if "top_ops" not in seen_kinds:
            errors.append(f"{rel}: missing required kind=top_ops")
        errors.extend(cls._check_report_p_item_per_section(text, rel))
        errors.extend(cls._check_report_detail_estimate_per_block(text, rel))
        return errors

    # ---- Per-card / per-block / per-candidate scoping ---------------------

    @classmethod
    def _slice_section(cls, text, start_header, end_headers=("\n## ",)):
        """Return the substring from `start_header` to the first match of any
        end_headers after it, or to EOF. Returns "" if start_header absent.
        """
        start = text.find(start_header)
        if start < 0:
            return ""
        end = -1
        for eh in end_headers:
            i = text.find(eh, start + 1)
            if i >= 0 and (end < 0 or i < end):
                end = i
        return text[start:] if end < 0 else text[start:end]

    @classmethod
    def _count_p_item_markers(cls, text):
        """Count `<!-- impact-begin ... kind=p_item ... -->` markers in text."""
        return sum(
            1
            for m in cls.BEGIN_RE.finditer(text)
            if cls.KIND_ATTR_RE.search(m.group(1)) is not None
            and cls.KIND_ATTR_RE.search(m.group(1)).group(1) == "p_item"
        )

    @classmethod
    def _check_p_item_per_card(cls, text, rel):
        """Strict 1:1: each ### P<N>: card under ## Recommendations must
        contain exactly one kind=p_item marker.
        """
        errors = []
        rec_section = cls._slice_section(
            text, "## Recommendations", end_headers=("\n## ",)
        )
        if not rec_section:
            return errors

        headings = list(_P_ITEM_HEADING_RE.finditer(rec_section))
        if not headings:
            return errors  # already flagged by Level 1 "no P-items" check

        for i, h in enumerate(headings):
            start = h.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(rec_section)
            card = rec_section[start:end]
            n = cls._count_p_item_markers(card)
            heading_text = h.group(0).strip()
            if n == 0:
                errors.append(f"{rel}: {heading_text} card missing kind=p_item marker")
            elif n > 1:
                errors.append(
                    f"{rel}: {heading_text} card has {n} kind=p_item markers (expected 1)"
                )
        return errors

    @classmethod
    def _check_detail_estimate_per_candidate(cls, text, rel):
        """Each <!-- reasoning-candidate ... --> block must contain either a
        kind=detail_estimate marker or the not-quantifiable sentinel.
        """
        errors = []
        candidates = list(_CANDIDATE_RE.finditer(text))
        if not candidates:
            return errors

        for i, c in enumerate(candidates):
            scope_start = c.end()
            scope_end = (
                candidates[i + 1].start() if i + 1 < len(candidates) else len(text)
            )
            scope = text[scope_start:scope_end]
            has_marker = any(
                (km := cls.KIND_ATTR_RE.search(m.group(1))) is not None
                and km.group(1) == "detail_estimate"
                for m in cls.BEGIN_RE.finditer(scope)
            )
            has_sentinel = bool(_NOT_QUANTIFIABLE_SENTINEL.search(scope))
            if not (has_marker or has_sentinel):
                rank = c.group(1) if c.groups() else "?"
                errors.append(
                    f"{rel}: reasoning-candidate rank={rank} missing "
                    f"kind=detail_estimate marker or 'not quantifiable from "
                    f"trace data' sentinel"
                )
        return errors

    @classmethod
    def _check_report_p_item_per_section(cls, text, rel):
        """For each strict report section, count of ### P<N>: headings must
        equal count of kind=p_item markers in that section.
        """
        errors = []
        for header in _REPORT_STRICT_P_ITEM_HEADERS:
            section = cls._slice_section(text, header, end_headers=("\n## ",))
            if not section:
                continue
            n_headings = len(_P_ITEM_HEADING_RE.findall(section))
            n_markers = cls._count_p_item_markers(section)
            if n_headings != n_markers:
                section_name = header.lstrip("# ").strip()
                errors.append(
                    f"{rel}: {section_name} P-item count mismatch "
                    f"({n_headings} ### P<N>: headings vs {n_markers} kind=p_item markers)"
                )
        return errors

    @classmethod
    def _check_report_detail_estimate_per_block(cls, text, rel):
        """In ## Detailed Analysis, each #### block under the strict
        subsections must contain a kind=detail_estimate marker or the
        not-quantifiable sentinel.
        """
        errors = []
        da_section = cls._slice_section(
            text, "## Detailed Analysis", end_headers=("\n## ",)
        )
        if not da_section:
            return errors

        for sub_header in _REPORT_DETAIL_ESTIMATE_SUBSECTIONS:
            sub = cls._slice_section(
                da_section, sub_header, end_headers=("\n### ", "\n## ")
            )
            if not sub:
                continue
            blocks = list(re.finditer(r"^#### .*$", sub, re.MULTILINE))
            if not blocks:
                continue
            sub_name = sub_header.lstrip("# ").strip()
            for i, b in enumerate(blocks):
                start = b.start()
                end = blocks[i + 1].start() if i + 1 < len(blocks) else len(sub)
                block = sub[start:end]
                has_marker = any(
                    (km := cls.KIND_ATTR_RE.search(m.group(1))) is not None
                    and km.group(1) == "detail_estimate"
                    for m in cls.BEGIN_RE.finditer(block)
                )
                has_sentinel = bool(_NOT_QUANTIFIABLE_SENTINEL.search(block))
                if not (has_marker or has_sentinel):
                    first_line = b.group(0).strip()
                    errors.append(
                        f"{rel}: {sub_name} block '{first_line}' missing "
                        f"kind=detail_estimate marker or 'not quantifiable "
                        f"from trace data' sentinel"
                    )
        return errors
