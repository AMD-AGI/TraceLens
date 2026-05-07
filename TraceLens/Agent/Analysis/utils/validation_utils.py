###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Validation utilities for TraceLens Agent.

Three validation levels, each at the boundary where issues are still fixable:

Level 1 — validate_findings_file (Steps 6-7, within each sub-agent)
    Structural check on a single findings file, including marker structure
    via MarkerValidator (pairing, kind attributes, per-kind required attrs,
    mandatory p_item). Sub-agent retries on failure.

Level 2 — validate_subagent_outputs (Step 8, batch)
    Cross-cutting checks that need all files: time sanity, coverage, priority.

Level 3 — validate_report (Step 11.1, after report assembly)
    Final analysis.md structure: headers, metrics table, placeholders,
    and report-level marker structure via MarkerValidator (pairing, kind
    attributes, mandatory top_ops).
"""

import json
import os
import re

from .report_utils import load_manifest, _scan_findings_dir

# Constants

_REQUIRED_FINDINGS_HEADERS = ["## Recommendations", "## Detailed Analysis"]
_COMPUTE_P_ITEM_LABELS = ["**Insight**", "**Action**", "**Impact**"]
_SYSTEM_P_ITEM_LABELS = ["**Insight**", "**Action**"]
_KERNEL_FUSION_FINDINGS = "kernel_fusion_findings.md"
# Optional icon / prefix before P<N> (e.g. kernel fusion `### 🟢 P1:`).
_P_ITEM_RE = re.compile(r"^### .*?P(\d+)\s*:", re.MULTILINE)
_CANDIDATE_RE = re.compile(r"<!-- reasoning-candidate\s+tier=\w+\s+rank=(\d+)\s*-->")
_NOT_QUANTIFIABLE_SENTINEL = re.compile(
    r"not quantifiable from trace data", re.IGNORECASE
)
# Header matcher for the report-wide Args verbatim check (Level 3).
_TABLE_HEADER_RE = re.compile(r"^\|.*\|\s*Args\s*\|.*\|\s*$", re.MULTILINE)
# Mandatory columns of the compute-tier **Data:** Operations Table, in spec
# order (sub_agent_spec.md § Operations Table Schema). Agents may append
# extra columns at the end but must not drop or reorder these.
_COMPUTE_DATA_REQUIRED_COLS = (
    "Operation",
    "Args",
    "Kernel Path",
    "Time (ms)",
    "%E2E",
    "Count",
    "FLOPS/Byte",
    "Efficiency",
    "Bound",
)
_TIME_DISCREPANCY_THRESHOLD = 10  # percent
_ROLLUP_IMPACT_TOL = 0.02  # ms; matches 2-decimal rounding in generate_priority_data
_MARKER_NUMERIC_TOL = 0.005  # ms; half a ULP at 2-decimal marker rendering
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


# Level 1: per-file findings validation (called within each sub-agent)


def _metrics_json_for_findings(filepath):
    """Path to category_data/<stem>_metrics.json for a *_findings.md under category_findings/."""
    output_dir = os.path.dirname(os.path.dirname(filepath))
    stem = os.path.basename(filepath).replace("_findings.md", "")
    return os.path.join(output_dir, "category_data", f"{stem}_metrics.json")


def _category_findings_empty(filepath):
    """True when metrics JSON exists and category_findings is an empty list (sub_agent_spec § empty)."""
    mp = _metrics_json_for_findings(filepath)
    try:
        with open(mp) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    cf = data.get("category_findings")
    return isinstance(cf, list) and len(cf) == 0


def validate_findings_file(filepath, tier):
    """Validate a single findings file against the sub-agent spec contract.

    Checks:
    - Required ## headers present and in correct order
    - P-item labels match tier (compute: Insight/Action/Impact; system: Insight/Action)
    - At least one reasoning-candidate block in Detailed Analysis
    - P-item count matches reasoning-candidate count
    - Compute tier only: per-block Data table shape + Args/Kernel Path cells
      verbatim vs <cat>_metrics.json (see _validate_compute_data_tables)
    - Marker structure: pairing, kind= attribute, per-kind required attrs,
      no mixed null/numeric values, mandatory kind=p_item (except for
      triton_findings.md)

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

    relaxed_empty = tier == "compute" and _category_findings_empty(filepath)

    p_items = []
    if rec_start >= 0:
        rec_end = da_start if da_start > rec_start else len(content)
        rec_section = content[rec_start:rec_end]

        p_items = _P_ITEM_RE.findall(rec_section)
        if not relaxed_empty:
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

        # Kernel fusion (system_findings) uses roofline-backed **Impact** on P-items.
        if (
            tier == "system"
            and "**Impact**" in rec_section
            and os.path.basename(filepath) != _KERNEL_FUSION_FINDINGS
        ):
            errors.append(
                "System-tier ## Recommendations must not contain **Impact** labels"
            )

    candidates = []
    if da_start >= 0:
        da_section = content[da_start:]

        candidates = _CANDIDATE_RE.findall(da_section)
        if not relaxed_empty:
            if not candidates:
                errors.append(
                    "No <!-- reasoning-candidate --> blocks in ## Detailed Analysis"
                )

    if not relaxed_empty and p_items and candidates and len(p_items) != len(candidates):
        errors.append(
            f"P-item count ({len(p_items)}) does not match "
            f"reasoning-candidate count ({len(candidates)})"
        )

    # Compute tier only: shape + Args verbatim + Kernel Path verbatim, all
    # scoped to <!-- reasoning-candidate tier=compute --> blocks.
    if tier == "compute":
        errors.extend(_validate_compute_data_tables(content, filepath))

    # Marker structure (folded in from the former Level-4 validate_markers).
    file_class = "category_findings" if tier == "compute" else "system_findings"
    errors.extend(
        MarkerValidator.check_findings_file(
            filepath, file_class, skip_p_item_required=relaxed_empty
        )
    )

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


def _load_compute_data_metrics(metrics_path):
    """Return (args_set, launcher_paths_set) from one metrics JSON; empty on read failure."""
    try:
        with open(metrics_path) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set(), set()
    ops = d.get("operations", [])
    args = {op["args"] for op in ops if isinstance(op.get("args"), str)}
    paths = {
        op["launcher_path"]
        for op in ops
        if isinstance(op.get("launcher_path"), str) and op["launcher_path"]
    }
    return args, paths


def _iter_compute_candidate_blocks(content):
    """Yield (start, end) line-index range for each tier=compute candidate block."""
    lines = content.splitlines()
    starts = [
        (idx, _CANDIDATE_RE.search(line))
        for idx, line in enumerate(lines)
        if _CANDIDATE_RE.search(line)
    ]
    for i, (idx, m) in enumerate(starts):
        if "tier=compute" not in m.group(0):
            continue
        end = starts[i + 1][0] if i + 1 < len(starts) else len(lines)
        yield idx, end


def _find_data_table(lines, start, end):
    """Locate the first markdown table after **Data:** in lines[start:end].

    Returns (header_line_1based, header_cols, row_iter) or None. row_iter
    yields (line_no_1based, cells) per body row.
    """
    data_idx = next(
        (i for i in range(start, end) if lines[i].strip() == "**Data:**"),
        None,
    )
    if data_idx is None:
        return None
    header_idx = next(
        (
            i
            for i in range(data_idx + 1, end)
            if lines[i].lstrip().startswith("|") and lines[i].rstrip().endswith("|")
        ),
        None,
    )
    if header_idx is None or header_idx + 1 >= end:
        return None
    header_cols = [c.strip() for c in lines[header_idx].strip().strip("|").split("|")]

    def rows():
        for row_idx in range(header_idx + 2, end):
            row = lines[row_idx]
            if not row.strip().startswith("|"):
                break
            yield row_idx + 1, [c.strip() for c in row.strip().strip("|").split("|")]

    return header_idx + 1, header_cols, rows()


def _validate_compute_data_tables(content, findings_path):
    """For each <!-- reasoning-candidate tier=compute --> block: shape check
    (Args column required), Args cells verbatim vs operations[].args, and
    Kernel Path cells verbatim vs operations[].launcher_path when present.
    Skips silently when the metrics JSON is absent.
    """
    metrics_path = _metrics_json_for_findings(findings_path)
    cat_metrics_basename = os.path.basename(metrics_path)
    valid_args, valid_paths = _load_compute_data_metrics(metrics_path)
    lines = content.splitlines()
    errors = []
    for start, end in _iter_compute_candidate_blocks(content):
        table = _find_data_table(lines, start, end)
        if table is None:
            errors.append(
                f"compute-tier reasoning-candidate block at line {start + 1}: "
                f"no **Data:** table found"
            )
            continue
        header_line, header_cols, row_iter = table
        if (
            tuple(header_cols[: len(_COMPUTE_DATA_REQUIRED_COLS)])
            != _COMPUTE_DATA_REQUIRED_COLS
        ):
            errors.append(
                f"compute Data table at line {header_line}: header must start "
                f"with the {len(_COMPUTE_DATA_REQUIRED_COLS)} canonical columns "
                f"in order {list(_COMPUTE_DATA_REQUIRED_COLS)}; got {header_cols} "
                f"(sub_agent_spec.md § Operations Table Schema)"
            )
            continue
        args_idx = _COMPUTE_DATA_REQUIRED_COLS.index("Args")
        kp_idx = _COMPUTE_DATA_REQUIRED_COLS.index("Kernel Path")
        for row_line, cells in row_iter:
            if valid_args and args_idx < len(cells) and cells[args_idx]:
                if cells[args_idx] not in valid_args:
                    errors.append(
                        f"Args cell on line {row_line} does not match "
                        f"operations[].args in {cat_metrics_basename} "
                        f"(paste verbatim, keep <br>): {cells[args_idx]}"
                    )
            if valid_paths and kp_idx < len(cells) and cells[kp_idx]:
                if cells[kp_idx] not in valid_paths:
                    errors.append(
                        f"Kernel Path cell on line {row_line} does not match "
                        f"operations[].launcher_path in {cat_metrics_basename} "
                        f"(paste verbatim): {cells[kp_idx]}"
                    )
    return errors


# Level 2: cross-cutting batch checks (called at Step 8)


def validate_subagent_outputs(output_dir):
    """Run cross-cutting validation checks on all subagent outputs.

    Returns a dict with check results and prints a summary.
    """
    manifest = load_manifest(output_dir)

    results = {
        "time_check": _check_time_sanity(manifest),
        "coverage_check": _check_coverage(output_dir, manifest),
        "priority_check": _check_priority_consistency(output_dir, manifest),
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


def _check_priority_consistency(output_dir, manifest):
    """Verify priority_data.json invariants: findings sort, rank contiguity,
    and per-category rollup of priorities[].impact_score vs findings[] sum.

    Non-blocking: returns WARN on any violation (preserves Step 8 semantics
    matching _check_time_sanity / _check_coverage). manifest is accepted for
    call-site symmetry with the other Step 8 checks.
    """
    del manifest  # unused; kept for signature symmetry with sibling checks
    pd_path = os.path.join(output_dir, "priority_data.json")
    if not os.path.exists(pd_path):
        return {"status": "WARN", "messages": ["priority_data.json not found"]}
    try:
        with open(pd_path) as f:
            pd = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return {"status": "WARN", "messages": [f"priority_data.json unreadable: {e}"]}

    findings = pd.get("findings", []) or []
    priorities = pd.get("priorities", []) or []
    messages = []

    scored = [
        (i, f.get("impact_score"))
        for i, f in enumerate(findings)
        if f.get("impact_score") is not None
    ]
    for (i_a, s_a), (i_b, s_b) in zip(scored, scored[1:]):
        if s_a < s_b:
            messages.append(
                f"INV1: findings[] not sorted desc by impact_score at index "
                f"{i_a}->{i_b} ({s_a} < {s_b})"
            )
            break

    for i, f in enumerate(findings):
        if f.get("global_rank") != i + 1:
            messages.append(
                f"INV2: findings[{i}].global_rank={f.get('global_rank')} "
                f"expected {i + 1}"
            )
            break
    for i, p in enumerate(priorities):
        if p.get("rank") != i + 1:
            messages.append(
                f"INV3: priorities[{i}].rank={p.get('rank')} expected {i + 1}"
            )
            break

    for p in priorities:
        if p.get("source") != "findings_rollup":
            continue
        cat = p.get("category")
        expected = sum(
            f.get("impact_score", 0) for f in findings if f.get("category") == cat
        )
        actual = p.get("impact_score", 0) or 0
        if abs(actual - expected) > _ROLLUP_IMPACT_TOL:
            messages.append(
                f"INV7': priorities[category={cat}].impact_score={actual:.4f} "
                f"!= sum(findings[].impact_score)={expected:.4f}"
            )

    status = "WARN" if messages else "PASS"
    return {"status": status, "messages": messages}


# Level 3: final report validation (called at Step 11.1)


def validate_report(output_dir):
    """Validate analysis.md for structural issues.

    Checks:
    - Required ## section headers in analysis.md
    - Metrics table under Executive Summary (5 rows, no placeholders)
    - Unfilled template placeholders
    - Args column cells match operations[].args verbatim
    - Report-level marker structure: pairing, kind= attribute, per-kind
      required attrs, mandatory kind=top_ops

    Findings files are validated separately by validate_findings_file
    (Level 1, called within each sub-agent at Steps 6-7), which also
    enforces per-file marker structure.

    Args:
        output_dir: Base output directory containing analysis.md

    Returns:
        Tuple of (passed: bool, missing: list of error/missing-section strings)
    """
    report_path = os.path.join(output_dir, "analysis.md")
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

    missing.extend(_validate_report_priority_consistency(content, output_dir))

    # Report-level marker structure
    missing.extend(MarkerValidator.check_report(report_path))

    return len(missing) == 0, missing


def _validate_report_args_column(content, output_dir):
    """Level-3 check: every Args cell in analysis.md must match some
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
        f"analysis.md line {ln}: Args cell does not match any "
        f"operations[].args in category_data/*_metrics.json (paste verbatim, "
        f"keep <br>): {cell}"
        for ln, cell in _scan_args_cells(content)
        if cell not in valid_args
    ]


def _validate_report_priority_consistency(content, output_dir):
    """Cross-check analysis.md against priority_data.json.

    R1: Compute Kernel Optimizations P-item heading count == len(quantified findings).
    R2: Each kind=p_item marker's category attr (in doc order) == findings[N-1].category.
    R3: Each marker's low/mid/high attrs match findings[N-1] impact_score_low / impact_score / impact_score_high.
    R4: Top Operations marker rows == len(priorities).

    Silently skips when priority_data.json is absent (Step 8 already warns).
    Numeric attrs are compared as 2-decimal strings to match the writer's
    rounding in generate_priority_data.
    """
    pd_path = os.path.join(output_dir, "priority_data.json")
    if not os.path.exists(pd_path):
        return []
    try:
        with open(pd_path) as f:
            pd = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    findings = [
        f for f in (pd.get("findings", []) or []) if f.get("impact_score") is not None
    ]
    priorities = pd.get("priorities", []) or []
    errors = []

    sec_start = content.find("## Compute Kernel Optimizations")
    if sec_start < 0:
        return errors
    sec_end = content.find("\n## ", sec_start + 1)
    section = content[sec_start:] if sec_end < 0 else content[sec_start:sec_end]

    n_p = len(_P_ITEM_RE.findall(section))
    if n_p != len(findings):
        errors.append(
            f"R1: Compute Kernel Optimizations has {n_p} P-item headings but "
            f"priority_data.json has {len(findings)} quantified findings"
        )

    p_markers = []
    for m in MarkerValidator.BEGIN_RE.finditer(section):
        inner = m.group(1)
        km = MarkerValidator.KIND_ATTR_RE.search(inner)
        if not km or km.group(1) != "p_item":
            continue
        p_markers.append(dict(MarkerValidator.ATTR_RE.findall(inner)))

    for idx, attrs in enumerate(p_markers):
        if idx >= len(findings):
            errors.append(
                f"R2: extra kind=p_item marker #{idx + 1} in Compute Kernel "
                f"Optimizations beyond {len(findings)} quantified findings"
            )
            break
        f = findings[idx]
        cat_attr = attrs.get("category")
        if cat_attr != f.get("category"):
            errors.append(
                f"R2: P-item marker #{idx + 1} category={cat_attr!r} != "
                f"findings[{idx}].category={f.get('category')!r}"
            )
        for attr_name, json_key in (
            ("low", "impact_score_low"),
            ("mid", "impact_score"),
            ("high", "impact_score_high"),
        ):
            got = attrs.get(attr_name)
            if got is None:
                continue
            raw = f.get(json_key)
            try:
                if abs(float(got) - float(raw)) > _MARKER_NUMERIC_TOL:
                    errors.append(
                        f"R3: P-item marker #{idx + 1} {attr_name}={got} != "
                        f"findings[{idx}].{json_key}={raw}"
                    )
            except (TypeError, ValueError):
                if got != ("null" if raw is None else str(raw)):
                    errors.append(
                        f"R3: P-item marker #{idx + 1} {attr_name}={got} != "
                        f"findings[{idx}].{json_key}={raw}"
                    )

    top_match = re.search(
        r"<!--\s*impact-begin[^>]*?\bkind=top_ops\b[^>]*?-->(.*?)<!--\s*impact-end\s*-->",
        content,
        re.DOTALL,
    )
    if top_match:
        body = top_match.group(1)
        n_rows = sum(
            1
            for ln in body.splitlines()
            if ln.lstrip().startswith("|")
            and not re.match(r"^\s*\|[\s\-:|]+\|\s*$", ln)
        )
        n_data_rows = max(0, n_rows - 1)
        if n_data_rows != len(priorities):
            errors.append(
                f"R4: Top Operations table has {n_data_rows} data rows but "
                f"priority_data.json::priorities has {len(priorities)} entries"
            )

    return errors


class MarkerValidator:
    """All marker (`<!-- impact-begin ... -->` / `<!-- impact-end -->`) checks.

    Stateless: regexes and kind tables are class-level constants and the
    entry points are classmethods.
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
    def check_findings_file(cls, path, file_class, *, skip_p_item_required=False):
        """Marker checks for a sub-agent findings file.

        file_class must be "category_findings" or "system_findings". Adds
        the per-tier "p_item required" check on top of `scan`, with the
        triton exemption for category findings.
        skip_p_item_required: when True (empty category_findings[] per metrics), omit kind=p_item requirement.
        """
        rel = os.path.basename(path)
        with open(path) as f:
            text = f.read()
        errors, seen_kinds = cls.scan(text, rel)
        if file_class == "category_findings" and rel not in cls.COMPUTE_NO_P_ITEM:
            if not skip_p_item_required and "p_item" not in seen_kinds:
                errors.append(f"{rel}: missing required kind=p_item")
        if file_class == "system_findings" and "p_item" not in seen_kinds:
            errors.append(f"{rel}: missing required kind=p_item")
        n_headings = len(_P_ITEM_RE.findall(text))
        n_markers = sum(
            1
            for m in cls.BEGIN_RE.finditer(text)
            if (km := cls.KIND_ATTR_RE.search(m.group(1))) and km.group(1) == "p_item"
        )
        if n_headings and n_markers and n_headings != n_markers:
            errors.append(
                f"{rel}: {n_headings} ### P<N>: headings but {n_markers} kind=p_item markers"
            )
        errors.extend(cls._check_detail_estimate_per_candidate(text, rel))
        return errors

    @classmethod
    def _check_detail_estimate_per_candidate(cls, text, rel):
        """Each <!-- reasoning-candidate --> block must contain either a
        kind=detail_estimate marker or the not-quantifiable sentinel.
        """
        candidates = list(_CANDIDATE_RE.finditer(text))
        if not candidates:
            return []
        errors = []
        for i, c in enumerate(candidates):
            scope = text[
                c.end() : (
                    candidates[i + 1].start() if i + 1 < len(candidates) else len(text)
                )
            ]
            has_marker = any(
                (km := cls.KIND_ATTR_RE.search(m.group(1))) is not None
                and km.group(1) == "detail_estimate"
                for m in cls.BEGIN_RE.finditer(scope)
            )
            if not (has_marker or bool(_NOT_QUANTIFIABLE_SENTINEL.search(scope))):
                errors.append(
                    f"{rel}: reasoning-candidate rank={c.group(1)} missing "
                    f"kind=detail_estimate marker or 'not quantifiable from trace data' sentinel"
                )
        return errors

    @classmethod
    def check_report(cls, path):
        """Marker checks for the assembled analysis.md.

        Adds the report-only "top_ops required" check on top of `scan`.
        """
        rel = os.path.basename(path)
        with open(path) as f:
            text = f.read()
        errors, seen_kinds = cls.scan(text, rel)
        if "top_ops" not in seen_kinds:
            errors.append(f"{rel}: missing required kind=top_ops")
        return errors
