###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pythonic workflow evals.

Checks directory structure, file existence, and output completeness
against the category_manifest.json contract.
"""

import argparse
import csv
import json
import os
import re
import sys

CSV_COLUMNS = [
    "index",
    "category",
    "issue_summary",
    "result",
    "details",
    "root_cause",
    "recommended_fix",
]
_REQUIRED_MODEL_KEYS = {"model", "architecture", "scale", "precision"}

_MIN_REPORT_BYTES = 100
_GARBLED_THRESHOLD = 0.50


def _pre_check_gates(output_dir: str) -> str | None:
    """Return a failure reason if a hard pre-check gate trips, else None.

    Gates (applied in order, first failure wins):
      1. output_dir does not exist
      2. analysis.md missing or too small (< 100 bytes)
      3. analysis.md is garbled (> 50% non-ASCII or looks like JSON)
    """
    if not os.path.isdir(output_dir):
        return "output directory does not exist"

    report = os.path.join(output_dir, "analysis.md")
    if not os.path.isfile(report):
        return "analysis.md not found"
    size = os.path.getsize(report)
    if size < _MIN_REPORT_BYTES:
        return f"analysis.md too small ({size} bytes)"

    with open(report, encoding="utf-8", errors="replace") as f:
        content = f.read()
    if not content.strip():
        return "analysis.md is empty"

    non_ascii = sum(1 for c in content if ord(c) > 127)
    if len(content) > 0 and non_ascii / len(content) > _GARBLED_THRESHOLD:
        return f"analysis.md appears garbled ({non_ascii}/{len(content)} non-ASCII chars)"

    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return "analysis.md contains raw JSON instead of markdown"

    return None


def _load_manifest(output_dir: str) -> dict | None:
    path = os.path.join(output_dir, "category_data", "category_manifest.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _rebase_path(manifest_path: str, output_dir: str) -> str:
    """Rebase an absolute manifest path to the actual output_dir.

    The manifest stores absolute paths using the original output_dir.
    When evaluating against a different directory (e.g. analysis_output_ref),
    we extract the relative suffix and resolve it against the actual dir.
    """
    for marker in (
        "category_data/",
        "metadata/",
        "system_findings/",
        "category_findings/",
    ):
        idx = manifest_path.find(marker)
        if idx != -1:
            return os.path.join(output_dir, manifest_path[idx:])
    return manifest_path


def _check_directories(output_dir: str) -> tuple[str, str]:
    required = ["metadata", "category_data", "system_findings", "category_findings"]
    missing = [d for d in required if not os.path.isdir(os.path.join(output_dir, d))]
    if missing:
        return "FAIL", f"Missing directories: {', '.join(missing)}"
    return "PASS", ""


def _check_metadata_files(output_dir: str) -> tuple[str, str]:
    manifest = _load_manifest(output_dir)
    if manifest is None:
        return "FAIL", "category_manifest.json not found"
    missing = []
    for cat in manifest.get("categories", []):
        mf = cat.get("metadata_file", "")
        if mf and not os.path.isfile(_rebase_path(mf, output_dir)):
            missing.append(os.path.basename(mf))
    if missing:
        return "FAIL", f"Missing metadata files: {', '.join(missing)}"
    return "PASS", ""


def _check_model_info(output_dir: str) -> tuple[str, str]:
    path = os.path.join(output_dir, "metadata", "model_info.json")
    if not os.path.isfile(path):
        return "FAIL", "metadata/model_info.json not found"
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return "FAIL", f"Invalid JSON: {e}"
    missing = _REQUIRED_MODEL_KEYS - data.keys()
    if missing:
        return "FAIL", f"Missing keys: {', '.join(missing)}"
    empty = [k for k in _REQUIRED_MODEL_KEYS if not str(data[k]).strip()]
    if empty:
        return "FAIL", f"Empty values: {', '.join(empty)}"
    return "PASS", ""


def _check_unified_perf_report(output_dir: str) -> tuple[str, str]:
    path = os.path.join(output_dir, "perf_report_csvs", "unified_perf_summary.csv")
    if not os.path.isfile(path):
        return "FAIL", "perf_report_csvs/unified_perf_summary.csv not found"
    return "PASS", ""


def _check_tree_data_files(output_dir: str) -> tuple[str, str]:
    manifest = _load_manifest(output_dir)
    if manifest is None:
        return "FAIL", "category_manifest.json not found"
    missing = []
    for cat in manifest.get("categories", []):
        tf = cat.get("tree_data_file")
        if tf and not os.path.isfile(_rebase_path(tf, output_dir)):
            missing.append(os.path.basename(tf))
    if missing:
        return "FAIL", f"Missing tree data files: {', '.join(missing)}"
    return "PASS", ""


def _check_findings_exist(output_dir: str) -> tuple[str, str]:
    manifest = _load_manifest(output_dir)
    if manifest is None:
        return "FAIL", "category_manifest.json not found"
    gpu_util = manifest.get("gpu_utilization", {})
    idle_pct = gpu_util.get("idle_time_percent", 0)
    missing = []
    for cat in manifest.get("categories", []):
        name = cat["name"]
        if name == "cpu_idle" and idle_pct <= 15:
            continue
        tier = cat.get("tier", "compute_kernel")
        subdir = "system_findings" if tier == "system" else "category_findings"
        path = os.path.join(output_dir, subdir, f"{name}_findings.md")
        if not os.path.isfile(path):
            missing.append(f"{subdir}/{name}_findings.md")
    if missing:
        return "FAIL", f"Parallel subagent failed to invoke for: {', '.join(missing)}"
    return "PASS", ""


def _check_findings_placement(output_dir: str) -> tuple[str, str]:
    manifest = _load_manifest(output_dir)
    if manifest is None:
        return "FAIL", "category_manifest.json not found"
    gpu_util = manifest.get("gpu_utilization", {})
    idle_pct = gpu_util.get("idle_time_percent", 0)
    misplaced = []
    for cat in manifest.get("categories", []):
        name = cat["name"]
        if name == "cpu_idle" and idle_pct <= 15:
            continue
        tier = cat.get("tier", "compute_kernel")
        correct_dir = "system_findings" if tier == "system" else "category_findings"
        wrong_dir = "category_findings" if tier == "system" else "system_findings"
        correct = os.path.join(output_dir, correct_dir, f"{name}_findings.md")
        wrong = os.path.join(output_dir, wrong_dir, f"{name}_findings.md")
        if not os.path.isfile(correct) and os.path.isfile(wrong):
            misplaced.append(f"{name} in {wrong_dir}/ instead of {correct_dir}/")
    if misplaced:
        return "FAIL", f"Misplaced findings: {'; '.join(misplaced)}"
    return "PASS", ""


def _check_plot(output_dir: str) -> tuple[str, str]:
    plot_path = os.path.join(output_dir, "perf_improvement.png")
    plot_data_path = os.path.join(output_dir, "priority_data.json")

    if os.path.isfile(plot_path):
        return "PASS", ""

    if os.path.isfile(plot_data_path):
        with open(plot_data_path) as f:
            plot_data = json.load(f)
        recs = plot_data.get("recommendations", [])
        if not recs:
            return "PASS", "No kernel tuning recommendations — plot correctly skipped"

    cat_data_dir = os.path.join(output_dir, "category_data")
    if os.path.isdir(cat_data_dir):
        total_estimates = 0
        for f in os.listdir(cat_data_dir):
            if f.endswith("_metrics.json"):
                with open(os.path.join(cat_data_dir, f)) as fh:
                    metrics = json.load(fh)
                total_estimates += len(metrics.get("impact_estimates", []))
        if total_estimates == 0:
            return (
                "PASS",
                "No kernel tuning recommendations in any metrics — plot correctly skipped",
            )

    return "FAIL", "perf_improvement.png not found"


EVAL_REGISTRY = [
    (
        "Directory structure created",
        _check_directories,
        "pipeline",
        "Re-run analysis pipeline from Step 1",
    ),
    (
        "Metadata files exist on disk",
        _check_metadata_files,
        "pipeline",
        "Re-run orchestrator_prepare.py to regenerate metadata",
    ),
    (
        "Model info JSON exists and valid",
        _check_model_info,
        "pipeline",
        "Check model-identification subagent; add fallback skeleton in prepare step",
    ),
    (
        "Unified perf. report exists",
        _check_unified_perf_report,
        "pipeline",
        "Re-run TraceLens_generate_perf_report_pytorch (Step 1)",
    ),
    (
        "Tree data files exist on disk",
        _check_tree_data_files,
        "pipeline",
        "Re-run orchestrator_prepare.py to regenerate tree data",
    ),
    (
        "Categorical findings .md files exist",
        _check_findings_exist,
        "pipeline",
        "Retry failed subagent(s) for missing findings files",
    ),
    (
        "All findings exist",
        _check_findings_placement,
        "pipeline",
        "Fix tier assignment in orchestrator_prepare.py",
    ),
    (
        "Plot generated on disk",
        _check_plot,
        "pipeline",
        "Check priority_data.json; add fallback empty plot_data.json",
    ),
]


# ---------------------------------------------------------------------------
# Helpers for per-item deterministic checks (evals 9-14, excl. 12 = LLM)
# ---------------------------------------------------------------------------


def _make_row(index, summary, result, details, root_cause, fix):
    return {
        "index": index,
        "category": "Workflow",
        "issue_summary": summary,
        "result": result,
        "details": details,
        "root_cause": root_cause,
        "recommended_fix": fix,
    }


def _read_report(output_dir):
    path = os.path.join(output_dir, "analysis.md")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_section(content, header):
    """Extract text between a ## header and the next ## header or end of file."""
    escaped = re.escape(header)
    m = re.search(
        rf"^{escaped}[^\n]*\n(.*?)(?=^## |\Z)",
        content,
        re.MULTILINE | re.DOTALL,
    )
    return m.group(1).strip() if m else None


def _find_table_row(section_text, label_synonyms):
    """Find a markdown table row matching any synonym. Returns value cell or None."""
    for line in section_text.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if all(set(c) <= {"-", ":", " "} for c in cells):
            continue
        label = cells[0]
        for synonym in label_synonyms:
            if synonym.lower() in label.lower():
                return cells[1] if len(cells) > 1 else ""
    return None


def _extract_percent(value_str):
    """Extract a percentage value from a string like '82.2%'."""
    m = re.search(r"([\d.]+)\s*%", value_str)
    return float(m.group(1)) if m else None


def _load_gpu_timeline(output_dir):
    """Load gpu_timeline.csv as {type_name: percent}."""
    path = os.path.join(output_dir, "perf_report_csvs", "gpu_timeline.csv")
    if not os.path.isfile(path):
        return None
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data[row["type"]] = float(row["percent"])
            except (KeyError, ValueError):
                continue
    return data


def _extract_p_items(section_text):
    """Extract P-items from a section. Returns [(p_number, text_block), ...]."""
    p_pattern = re.compile(r"^### .+P(\d+):", re.MULTILINE)
    matches = list(p_pattern.finditer(section_text))
    items = []
    for i, m in enumerate(matches):
        pnum = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
        items.append((pnum, section_text[start:end]))
    return items


def _check_p_item_fields(p_text, is_compute):
    """Check P-item for required bold fields. Returns list of missing field names."""
    missing = []
    if not re.search(r"\*\*Insight\*\*", p_text) and not re.search(
        r"\*\*Issue\*\*", p_text
    ):
        missing.append("**Insight** or **Issue**")
    if not re.search(r"\*\*Action\*\*", p_text):
        missing.append("**Action**")
    if is_compute and not re.search(r"\*\*Impact\*\*", p_text):
        missing.append("**Impact**")
    return missing


# ---------------------------------------------------------------------------
# Per-item check functions (evals 9, 10, 11, 14)
# ---------------------------------------------------------------------------

_REPORT_HEADERS = {
    "executive_summary": "Executive Summary",
    "compute": "Compute Kernel Optimizations",
    "system": "System-Level Optimizations",
    "detailed": "Detailed Analysis",
    "appendix": "Appendix",
}


def _check_report_template(output_dir):
    """Eval 9 — per-header check for required section headers + table presence."""
    content = _read_report(output_dir)
    if content is None:
        return [
            _make_row(
                "workflow_eval_9",
                "Report Template Rendering",
                "FAIL",
                "analysis.md not found",
                "pipeline",
                "Re-run report generation",
            )
        ]

    rows = []
    for key, header_text in _REPORT_HEADERS.items():
        found = bool(re.search(rf"^## {re.escape(header_text)}", content, re.MULTILINE))
        rows.append(
            _make_row(
                f"workflow_eval_9_{key}",
                f"Report header: {header_text}",
                "PASS" if found else "FAIL",
                "" if found else f"Missing '## {header_text}' header",
                "template" if not found else "",
                "Add missing section header to report" if not found else "",
            )
        )

    exec_section = _extract_section(content, "## Executive Summary")
    has_table = bool(re.search(r"\|.*\|", exec_section)) if exec_section else False
    rows.append(
        _make_row(
            "workflow_eval_9_metrics_table",
            "Metrics table in Executive Summary",
            "PASS" if has_table else "FAIL",
            "" if has_table else "No markdown table in Executive Summary",
            "template" if not has_table else "",
            "Add metrics table to Executive Summary" if not has_table else "",
        )
    )
    return rows


_EXEC_SUMMARY_CHECKS = [
    ("total_time", ["Total Compute Time", "Total Time"], None, None),
    ("compute_pct", ["Computation", "Compute %", "Compute"], "computation_time", 1.0),
    ("idle_pct", ["Idle Time", "Idle %", "Idle"], "idle_time", 1.0),
    ("comm_pct", ["Exposed Communication", "Exposed Communication %"], None, None),
    ("bottleneck", ["Top Bottleneck Category", "Top Bottleneck"], None, None),
]


def _check_exec_summary(output_dir):
    """Eval 10 — per-row check for exec summary metrics + CSV cross-validation."""
    content = _read_report(output_dir)
    if content is None:
        return [
            _make_row(
                "workflow_eval_10",
                "Executive Summary has metrics table",
                "FAIL",
                "analysis.md not found",
                "pipeline",
                "Re-run report generation",
            )
        ]

    exec_section = _extract_section(content, "## Executive Summary")
    if not exec_section:
        return [
            _make_row(
                "workflow_eval_10",
                "Executive Summary has metrics table",
                "FAIL",
                "Executive Summary section not found",
                "template",
                "Add Executive Summary section",
            )
        ]

    csv_data = _load_gpu_timeline(output_dir)

    rows = []
    for key, labels, csv_type, tolerance in _EXEC_SUMMARY_CHECKS:
        value_str = _find_table_row(exec_section, labels)
        if value_str is None:
            rows.append(
                _make_row(
                    f"workflow_eval_10_{key}",
                    f"Exec Summary row: {labels[0]}",
                    "FAIL",
                    f"Row not found (looked for: {', '.join(labels)})",
                    "template",
                    "Add missing row to Executive Summary table",
                )
            )
            continue

        result, detail = "PASS", ""
        if csv_type and csv_data and tolerance:
            csv_val = csv_data.get(csv_type)
            if csv_val is not None:
                report_val = _extract_percent(value_str)
                if report_val is not None:
                    diff = abs(report_val - csv_val)
                    if diff > tolerance:
                        result = "FAIL"
                        detail = (
                            f"Value mismatch: report={report_val:.1f}%, "
                            f"csv={csv_val:.2f}%, diff={diff:.2f}% "
                            f"(tolerance={tolerance}%)"
                        )
                    else:
                        detail = (
                            f"Matches CSV (report={report_val:.1f}%, "
                            f"csv={csv_val:.2f}%)"
                        )

        rows.append(
            _make_row(
                f"workflow_eval_10_{key}",
                f"Exec Summary row: {labels[0]}",
                result,
                detail,
                "template" if result == "FAIL" else "",
                "Fix value in Executive Summary table" if result == "FAIL" else "",
            )
        )
    return rows


def _check_issue_template(output_dir):
    """Eval 11 — per-P-item check for correct bold fields in each priority item."""
    content = _read_report(output_dir)
    if content is None:
        return [
            _make_row(
                "workflow_eval_11",
                "Issue Template rendering",
                "FAIL",
                "analysis.md not found",
                "pipeline",
                "Re-run report generation",
            )
        ]

    compute_section = _extract_section(content, "## Compute Kernel Optimizations")
    system_section = _extract_section(content, "## System-Level Optimizations")

    rows = []
    if compute_section:
        for pnum, p_text in _extract_p_items(compute_section):
            missing = _check_p_item_fields(p_text, is_compute=True)
            result = "PASS" if not missing else "FAIL"
            rows.append(
                _make_row(
                    f"workflow_eval_11_compute_P{pnum}",
                    f"Compute P{pnum} template",
                    result,
                    f"Missing: {', '.join(missing)}" if missing else "",
                    "template" if result == "FAIL" else "",
                    "Add missing fields to P-item" if result == "FAIL" else "",
                )
            )

    if system_section:
        for pnum, p_text in _extract_p_items(system_section):
            missing = _check_p_item_fields(p_text, is_compute=False)
            result = "PASS" if not missing else "FAIL"
            rows.append(
                _make_row(
                    f"workflow_eval_11_system_P{pnum}",
                    f"System P{pnum} template",
                    result,
                    f"Missing: {', '.join(missing)}" if missing else "",
                    "template" if result == "FAIL" else "",
                    "Add missing fields to P-item" if result == "FAIL" else "",
                )
            )

    if not rows:
        rows.append(
            _make_row(
                "workflow_eval_11",
                "Issue Template rendering",
                "FAIL",
                "No P-items found in report",
                "template",
                "Ensure report contains priority items",
            )
        )
    return rows


_MODEL_INFO_FIELDS = ["model", "architecture", "scale", "precision"]


def _check_model_id(output_dir):
    """Eval 13 — per-field check for model_info.json values in Appendix."""
    model_info_path = os.path.join(output_dir, "metadata", "model_info.json")
    report_path = os.path.join(output_dir, "analysis.md")

    if not os.path.isfile(model_info_path):
        return [
            _make_row(
                "workflow_eval_13",
                "Model identification in report",
                "FAIL",
                "metadata/model_info.json not found",
                "pipeline",
                "Run model identification subagent",
            )
        ]
    if not os.path.isfile(report_path):
        return [
            _make_row(
                "workflow_eval_13",
                "Model identification in report",
                "FAIL",
                "analysis.md not found",
                "pipeline",
                "Re-run report generation",
            )
        ]

    try:
        with open(model_info_path) as f:
            model_info = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [
            _make_row(
                "workflow_eval_13",
                "Model identification in report",
                "FAIL",
                "Invalid model_info.json",
                "pipeline",
                "Fix model identification subagent",
            )
        ]

    with open(report_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    appendix = _extract_section(content, "## Appendix")
    if not appendix:
        return [
            _make_row(
                "workflow_eval_13",
                "Model identification in report",
                "FAIL",
                "Appendix section not found in report",
                "template",
                "Add Appendix section to report",
            )
        ]

    appendix_lower = appendix.lower()
    rows = []
    for field in _MODEL_INFO_FIELDS:
        value = str(model_info.get(field, "")).strip()
        if not value or value.lower() == "cannot be inferred from trace":
            rows.append(
                _make_row(
                    f"workflow_eval_13_{field}",
                    f"Model ID: {field}",
                    "PASS",
                    f"Field not determined ('{value}') — skipped",
                    "",
                    "",
                )
            )
            continue

        found = value.lower() in appendix_lower
        rows.append(
            _make_row(
                f"workflow_eval_13_{field}",
                f"Model ID: {field}",
                "PASS" if found else "FAIL",
                "" if found else f"'{value}' not found in Appendix",
                "template" if not found else "",
                "Add model info field to Appendix" if not found else "",
            )
        )
    return rows


_MULTI_EVAL_CHECKS = [
    _check_report_template,
    _check_exec_summary,
    _check_issue_template,
    _check_model_id,
]

_GATE_FAIL_NEW_EVALS = [
    ("workflow_eval_9", "Report Template Rendering"),
    ("workflow_eval_10", "Executive Summary has metrics table"),
    ("workflow_eval_11", "Issue Template rendering"),
    ("workflow_eval_13", "Model identification in report"),
]


def run(output_dir: str, results_path: str) -> list[dict]:
    rows = []

    gate_fail = _pre_check_gates(output_dir)
    if gate_fail is not None:
        for i, (summary, _func, rc, fix) in enumerate(EVAL_REGISTRY, start=1):
            rows.append(
                {
                    "index": f"workflow_eval_{i}",
                    "category": "Workflow",
                    "issue_summary": summary,
                    "result": "FAIL",
                    "details": f"Pre-check gate: {gate_fail}",
                    "root_cause": "pipeline",
                    "recommended_fix": "Fix pre-check gate failure before running evals",
                }
            )
        for idx, summary in _GATE_FAIL_NEW_EVALS:
            rows.append(
                {
                    "index": idx,
                    "category": "Workflow",
                    "issue_summary": summary,
                    "result": "FAIL",
                    "details": f"Pre-check gate: {gate_fail}",
                    "root_cause": "pipeline",
                    "recommended_fix": "Fix pre-check gate failure before running evals",
                }
            )
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return rows

    for i, (summary, func, rc, fix) in enumerate(EVAL_REGISTRY, start=1):
        result, details = func(output_dir)
        rows.append(
            {
                "index": f"workflow_eval_{i}",
                "category": "Workflow",
                "issue_summary": summary,
                "result": result,
                "details": details,
                "root_cause": rc if result == "FAIL" else "",
                "recommended_fix": fix if result == "FAIL" else "",
            }
        )

    for check_fn in _MULTI_EVAL_CHECKS:
        rows.extend(check_fn(output_dir))

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Pythonic workflow evals")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    rows = run(args.output_dir, args.results)
    passed = sum(1 for r in rows if r["result"] == "PASS")
    sys.exit(0 if passed == len(rows) else 1)


if __name__ == "__main__":
    main()
