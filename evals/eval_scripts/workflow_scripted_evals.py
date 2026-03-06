"""Pythonic workflow evals.

Checks directory structure, file existence, and output completeness
against the category_manifest.json contract.
"""

import argparse
import csv
import json
import os
import sys

CSV_COLUMNS = ["index", "category", "issue_summary", "result", "details"]


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
    for marker in ("category_data/", "metadata/", "system_findings/", "category_findings/"):
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
    missing = []
    for cat in manifest.get("categories", []):
        name = cat["name"]
        tier = cat.get("tier", "compute_kernel")
        subdir = "system_findings" if tier == "system" else "category_findings"
        path = os.path.join(output_dir, subdir, f"{name}_findings.md")
        if not os.path.isfile(path):
            missing.append(f"{subdir}/{name}_findings.md")
    if missing:
        return "FAIL", f"Missing findings: {', '.join(missing)}"
    return "PASS", ""


def _check_findings_placement(output_dir: str) -> tuple[str, str]:
    manifest = _load_manifest(output_dir)
    if manifest is None:
        return "FAIL", "category_manifest.json not found"
    misplaced = []
    for cat in manifest.get("categories", []):
        name = cat["name"]
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
    plot_data_path = os.path.join(output_dir, "plot_data.json")

    if os.path.isfile(plot_path):
        return "PASS", ""

    if os.path.isfile(plot_data_path):
        with open(plot_data_path) as f:
            plot_data = json.load(f)
        recs = plot_data.get("recommendations", [])
        if not recs:
            return "PASS", "No kernel tuning recommendations — plot correctly skipped"

    return "FAIL", "perf_improvement.png not found"


EVAL_REGISTRY = [
    ("Directory structure created", _check_directories),
    ("Metadata files exist on disk", _check_metadata_files),
    ("Unified perf. report exists", _check_unified_perf_report),
    ("Tree data files exist on disk", _check_tree_data_files),
    ("Categorical findings .md files exist", _check_findings_exist),
    ("All findings exist", _check_findings_placement),
    ("Plot generated on disk", _check_plot),
]


def run(output_dir: str, results_path: str) -> list[dict]:
    rows = []
    for i, (summary, func) in enumerate(EVAL_REGISTRY, start=1):
        result, details = func(output_dir)
        rows.append(
            {
                "index": f"workflow_eval_{i}",
                "category": "Workflow",
                "issue_summary": summary,
                "result": result,
                "details": details,
            }
        )
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
