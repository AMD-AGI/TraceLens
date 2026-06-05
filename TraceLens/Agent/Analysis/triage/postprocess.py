###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triage post-processor — discovers triage CSVs under a traces root folder,
aggregates findings, writes a summary report with action items, and builds
reproducer tar.gz packages.

Usage:
    python -m TraceLens.Agent.Analysis.triage.postprocess \
        --traces-root /path/to/traces \
        --report-dir ./triage_report \
        [--top-reproducers 3] \
        [--max-pkg-size-mb 200]

    # Or with a pre-built mapping file:
    python -m TraceLens.Agent.Analysis.triage.postprocess \
        --mapping tracelens_folders_with_triage.txt \
        --report-dir ./triage_report
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TriageFinding:
    __slots__ = ("run_dir", "model_name", "diag_tag", "category",
                 "failure_mode", "evidence", "remedy", "implied_by")

    def __init__(self, run_dir: str, model_name: str, diag_tag: str,
                 category: str, failure_mode: str, evidence: str,
                 remedy: str, implied_by: str):
        self.run_dir = run_dir
        self.model_name = model_name
        self.diag_tag = diag_tag
        self.category = category
        self.failure_mode = failure_mode
        self.evidence = evidence
        self.remedy = remedy
        self.implied_by = implied_by


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_MODEL_RE = re.compile(
    r"/([^/]+)/\d{8}T\d{6}Z/kernel-agent/runs/"
)

_EG_RE = re.compile(r"\(e\.g\.\s+(.+?)\)$")


def extract_model_name(run_dir: str) -> str:
    m = _MODEL_RE.search(run_dir)
    return m.group(1) if m else Path(run_dir).name


def sanitize_filename(name: str, max_len: int = 80) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", name.lower().strip())
    return s[:max_len].rstrip("_")


def extract_action_keys(evidence: str) -> List[str]:
    """Extract op names from evidence strings like '... (e.g. op1, op2, op3)'."""
    m = _EG_RE.search(evidence)
    if not m:
        return []
    raw = m.group(1)
    return [op.strip() for op in raw.split(",") if op.strip()]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_triage_csvs(traces_root: str) -> List[str]:
    """Walk traces_root to find all triage_details.csv files. Returns run_dir paths."""
    run_dirs: List[str] = []
    for dirpath, dirnames, filenames in os.walk(traces_root, followlinks=False):
        if "triage_details.csv" in filenames:
            run_dirs.append(dirpath)
        if len(run_dirs) % 500 == 0 and run_dirs:
            print(f"  ... discovered {len(run_dirs)} so far", file=sys.stderr)
    return sorted(run_dirs)


def load_from_mapping(mapping_path: str) -> List[str]:
    """Read run_dir paths from a tab-separated mapping file."""
    run_dirs: List[str] = []
    with open(mapping_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0].strip():
                run_dirs.append(parts[0].strip())
    return run_dirs


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_findings(run_dirs: List[str]) -> Tuple[List[TriageFinding], int]:
    """Parse triage_details.csv from each run_dir. Returns (findings, total_runs)."""
    findings: List[TriageFinding] = []
    total_runs = len(run_dirs)

    for run_dir in run_dirs:
        csv_path = os.path.join(run_dir, "triage_details.csv")
        if not os.path.isfile(csv_path):
            continue
        model_name = extract_model_name(run_dir)
        try:
            with open(csv_path, newline="") as cf:
                reader = csv.DictReader(cf)
                for row in reader:
                    findings.append(TriageFinding(
                        run_dir=run_dir,
                        model_name=model_name,
                        diag_tag=row.get("DIAG Tag", ""),
                        category=row.get("Category", ""),
                        failure_mode=row.get("Failure Mode", ""),
                        evidence=row.get("Evidence", ""),
                        remedy=row.get("Remedy", ""),
                        implied_by=row.get("Implied By", ""),
                    ))
        except (OSError, csv.Error):
            continue

    return findings, total_runs


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(findings: List[TriageFinding]) -> Dict:
    by_category: Dict[str, Counter] = defaultdict(Counter)
    by_failure_mode: Counter = Counter()
    by_diag_tag: Counter = Counter()
    by_model: Dict[str, List[TriageFinding]] = defaultdict(list)
    models_per_issue: Dict[str, Set[str]] = defaultdict(set)
    action_keys_per_issue: Dict[str, Counter] = defaultdict(Counter)

    for f in findings:
        by_category[f.category][f.failure_mode] += 1
        by_failure_mode[f.failure_mode] += 1
        by_diag_tag[f.diag_tag] += 1
        by_model[f.model_name].append(f)
        models_per_issue[f.failure_mode].add(f.model_name)

        if f.category == "perf_model":
            for key in extract_action_keys(f.evidence):
                action_keys_per_issue[f.failure_mode][key] += 1

    return {
        "by_category": by_category,
        "by_failure_mode": by_failure_mode,
        "by_diag_tag": by_diag_tag,
        "by_model": by_model,
        "models_per_issue": models_per_issue,
        "action_keys_per_issue": action_keys_per_issue,
    }


# ---------------------------------------------------------------------------
# Reproducer selection
# ---------------------------------------------------------------------------

def pick_reproducers(
    findings: List[TriageFinding],
    agg: Dict,
    top_n: int = 3,
) -> List[Tuple[str, str, List[TriageFinding]]]:
    """Select top_n representative runs for reproducers.

    Strategy: score by coverage of top-10 failure modes + diversity of DIAG tags,
    preferring distinct models.
    """
    runs: Dict[str, List[TriageFinding]] = defaultdict(list)
    for f in findings:
        runs[f.run_dir].append(f)

    top_modes = set(m for m, _ in agg["by_failure_mode"].most_common(10))

    scored: List[Tuple[float, str, List[TriageFinding]]] = []
    for run_dir, run_findings in runs.items():
        tags = set(f.diag_tag for f in run_findings)
        modes_hit = set(f.failure_mode for f in run_findings)
        coverage = len(modes_hit & top_modes)
        diversity = len(tags)
        score = coverage * 10 + diversity
        scored.append((score, run_dir, run_findings))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[Tuple[str, str, List[TriageFinding]]] = []
    seen_models: Set[str] = set()
    for _score, run_dir, run_findings in scored:
        model = extract_model_name(run_dir)
        if model in seen_models:
            continue
        seen_models.add(model)
        selected.append((run_dir, model, run_findings))
        if len(selected) >= top_n:
            break

    return selected


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_aggregated_csv(findings: List[TriageFinding], report_dir: str) -> str:
    path = os.path.join(report_dir, "aggregated_triage.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Run Dir", "DIAG Tag", "Category",
            "Failure Mode", "Evidence", "Remedy", "Implied By",
        ])
        for finding in findings:
            writer.writerow([
                finding.model_name, finding.run_dir, finding.diag_tag,
                finding.category, finding.failure_mode, finding.evidence,
                finding.remedy, finding.implied_by,
            ])
    return path


def write_summary_report(
    findings: List[TriageFinding],
    agg: Dict,
    total_runs: int,
    reproducers: List[Tuple[str, str, List[TriageFinding]]],
    report_dir: str,
) -> str:
    path = os.path.join(report_dir, "summary_report.md")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    runs_with_findings = len(set(f.run_dir for f in findings))
    clean_runs = total_runs - runs_with_findings

    lines = [
        "# Triage Summary Report",
        "",
        f"Generated at: `{now}`",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total runs analyzed | {total_runs} |",
        f"| Runs with findings | {runs_with_findings} ({_pct(runs_with_findings, total_runs)}%) |",
        f"| Clean runs (no findings) | {clean_runs} ({_pct(clean_runs, total_runs)}%) |",
        f"| Total findings | {len(findings)} |",
        f"| Unique failure modes | {len(agg['by_failure_mode'])} |",
        "",
        "---",
        "",
        "## Issue Breakdown by Category",
        "",
        "| Category | Total Findings | Unique Failure Modes | Affected Models |",
        "|---|---|---|---|",
    ]

    for cat, mode_counts in sorted(
        agg["by_category"].items(),
        key=lambda x: sum(x[1].values()),
        reverse=True,
    ):
        total = sum(mode_counts.values())
        unique = len(mode_counts)
        affected = len(set(
            f.model_name for f in findings if f.category == cat
        ))
        lines.append(f"| {cat} | {total} | {unique} | {affected} |")

    # Top failure modes
    remedies: Dict[str, str] = {}
    categories: Dict[str, str] = {}
    diag_tags_per_mode: Dict[str, Set[str]] = defaultdict(set)
    for f in findings:
        if f.failure_mode:
            if f.failure_mode not in remedies:
                remedies[f.failure_mode] = f.remedy
                categories[f.failure_mode] = f.category
            if f.diag_tag:
                diag_tags_per_mode[f.failure_mode].add(f.diag_tag)

    lines += [
        "",
        "---",
        "",
        "## Top Failure Modes",
        "",
        "| Diag Tag | Failure Mode | Count | Affected Models | Remedy |",
        "|---|---|---|---|---|",
    ]

    for mode, count in agg["by_failure_mode"].most_common(20):
        n_models = len(agg["models_per_issue"].get(mode, set()))
        remedy = remedies.get(mode, "")
        tags = sorted(diag_tags_per_mode.get(mode, set()))
        tag = tags[0] if tags else ""
        lines.append(f"| {tag} | {mode} | {count} | {n_models} | {remedy} |")

    # Action keys for perf_model issues (only unclassified/synthetic op modes)
    _OP_ATTENTION_MODES = {
        "Unclassified op among significant ops",
        "Synthetic op appears among significant ops",
    }
    _LOOKS_LIKE_OP = re.compile(
        r"^(sglang_profiler::\S+|aiter::\S+|aten::\S+|"
        r"hipGraphLaunch->\S+|hipLaunchKernel->\S+|hipModuleLaunchKernel->\S+)"
    )
    filtered_ops: List[Tuple[str, int, str]] = []
    for mode, op_counts in agg["action_keys_per_issue"].items():
        if mode not in _OP_ATTENTION_MODES:
            continue
        for op, count in op_counts.items():
            if not _LOOKS_LIKE_OP.match(op.strip()):
                continue
            filtered_ops.append((op, count, mode))

    if filtered_ops:
        filtered_ops.sort(key=lambda x: x[1], reverse=True)
        lines += [
            "",
            "---",
            "",
            "## Perf Model — Specific Ops Needing Attention",
            "",
            "Op names extracted from evidence across all perf_model findings:",
            "",
            "| Op Name | Occurrences | Associated Failure Mode |",
            "|---|---|---|",
        ]
        for op, count, mode in filtered_ops[:30]:
            op_display = op[:100]
            lines.append(f"| `{op_display}` | {count} | {mode} |")

    # Action items
    lines += [
        "",
        "---",
        "",
        "## Action Items",
        "",
        "Priority-ordered list of fixes based on frequency and blast radius:",
        "",
    ]

    action_groups: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
    for mode, count in agg["by_failure_mode"].most_common():
        cat = categories.get(mode, "unknown")
        n_models = len(agg["models_per_issue"].get(mode, set()))
        action_groups[cat].append((mode, count, n_models))

    priority = 1
    for cat in sorted(
        action_groups.keys(),
        key=lambda c: sum(cnt for _, cnt, _ in action_groups[c]),
        reverse=True,
    ):
        items = action_groups[cat]
        lines.append(f"### {priority}. Category: `{cat}`")
        lines.append("")
        for mode, count, n_models in items:
            remedy = remedies.get(mode, "N/A")
            lines.append(f"- **{mode}** ({count} occurrences across {n_models} models)")
            lines.append(f"  - Remedy: {remedy}")
        lines.append("")
        priority += 1

    # Reproducers
    lines += [
        "---",
        "",
        "## Reproducers",
        "",
        f"Selected {len(reproducers)} representative runs covering the most "
        "common and diverse failure modes:",
        "",
    ]

    for i, (run_dir, model, run_findings) in enumerate(reproducers, 1):
        pkg_name = sanitize_filename(model)
        lines.append(f"### Reproducer {i}: `{model}`")
        lines.append("")
        lines.append(f"**Run dir:** `{run_dir}`")
        lines.append("")
        lines.append(f"**Findings ({len(run_findings)}):**")
        lines.append("")
        lines.append("| DIAG Tag | Failure Mode | Evidence (truncated) |")
        lines.append("|---|---|---|")
        for f in run_findings:
            ev = f.evidence[:120].replace("|", "\\|")
            lines.append(f"| {f.diag_tag} | {f.failure_mode} | {ev} |")
        lines.append("")
        lines.append(f"**Package:** `reproducers/{pkg_name}.tar.gz`")
        lines.append("")
        lines.append("**Re-run command:**")
        lines.append("```bash")
        lines.append(
            f"python -m TraceLens.Agent.Analysis.triage.runner "
            f"--run-dir '{run_dir}' --detailed"
        )
        lines.append("```")
        lines.append("")

    # DIAG tag frequency
    lines += [
        "---",
        "",
        "## DIAG Tag Frequency",
        "",
        "| DIAG Tag | Count |",
        "|---|---|",
    ]
    for tag, count in agg["by_diag_tag"].most_common():
        lines.append(f"| `{tag}` | {count} |")

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _pct(part: int, total: int) -> int:
    if total == 0:
        return 0
    return part * 100 // total


# ---------------------------------------------------------------------------
# Reproducer packaging
# ---------------------------------------------------------------------------

def _collectible_paths(run_dir: str, max_bytes: int) -> List[Tuple[str, str]]:
    """Identify artifacts to include in a reproducer tarball.

    Returns list of (absolute_path, arcname) pairs, respecting max_bytes cap.
    """
    items: List[Tuple[str, str]] = []
    total_size = 0

    def _add(abs_path: str, arc: str) -> bool:
        nonlocal total_size
        if not os.path.isfile(abs_path):
            return False
        size = os.path.getsize(abs_path)
        if total_size + size > max_bytes:
            return False
        items.append((abs_path, arc))
        total_size += size
        return True

    def _add_dir_gz(directory: str, arc_prefix: str) -> None:
        """Add all .json.gz files from a directory."""
        if not os.path.isdir(directory):
            return
        for fname in sorted(os.listdir(directory)):
            if fname.endswith(".json.gz"):
                _add(os.path.join(directory, fname), f"{arc_prefix}/{fname}")

    # Core metadata (small files first)
    _add(os.path.join(run_dir, "triage_details.csv"), "triage_details.csv")
    _add(os.path.join(run_dir, "triage_diags.txt"), "triage_diags.txt")
    _add(os.path.join(run_dir, "cache", "cmd_prefix.txt"), "cache/cmd_prefix.txt")

    # Manifests
    _add(
        os.path.join(run_dir, "category_data", "category_manifest.json"),
        "category_data/category_manifest.json",
    )
    ti_manifest = os.path.join(
        os.path.dirname(os.path.abspath(run_dir)), "trace_input_manifest.json"
    )
    _add(ti_manifest, "trace_input_manifest.json")

    # Perf report CSVs
    _add(
        os.path.join(run_dir, "perf_report_csvs", "gpu_timeline.csv"),
        "perf_report_csvs/gpu_timeline.csv",
    )
    _add(
        os.path.join(run_dir, "perf_report_csvs", "unified_perf_summary.csv"),
        "perf_report_csvs/unified_perf_summary.csv",
    )

    # Analysis report
    _add(os.path.join(run_dir, "analysis.md"), "analysis.md")

    # Trace split metadata
    split_dir = os.path.join(run_dir, "trace_split")
    _add(os.path.join(split_dir, "execution_details.csv"),
         "trace_split/execution_details.csv")
    _add(os.path.join(split_dir, "execution_details.json"),
         "trace_split/execution_details.json")

    # Trace split .json.gz files (the actual traces)
    _add_dir_gz(split_dir, "trace_split")

    # Graph capture traces
    capture_dir = os.path.join(
        os.path.dirname(os.path.abspath(run_dir)), "capture_traces"
    )
    if not os.path.isdir(capture_dir):
        manifest_path = os.path.join(
            run_dir, "category_data", "category_manifest.json"
        )
        if os.path.isfile(manifest_path):
            try:
                import json
                with open(manifest_path) as mf:
                    mdata = json.load(mf)
                cap_path = mdata.get("capture_folder_path")
                if cap_path and os.path.isdir(cap_path):
                    capture_dir = cap_path
            except (OSError, ValueError):
                pass

    _add_dir_gz(capture_dir, "capture_traces")

    return items


def build_reproducer_packages(
    reproducers: List[Tuple[str, str, List[TriageFinding]]],
    report_dir: str,
    max_pkg_size_mb: int = 200,
) -> int:
    """Create tar.gz reproducer packages. Returns count of packages built."""
    repro_dir = os.path.join(report_dir, "reproducers")
    os.makedirs(repro_dir, exist_ok=True)
    max_bytes = max_pkg_size_mb * 1024 * 1024

    count = 0
    for run_dir, model, run_findings in reproducers:
        pkg_name = sanitize_filename(model)
        pkg_folder = os.path.join(repro_dir, pkg_name)
        os.makedirs(pkg_folder, exist_ok=True)

        # Write README
        readme_lines = [
            f"# Reproducer: {model}",
            "",
            f"**Run dir:** `{run_dir}`",
            "",
            f"## Findings ({len(run_findings)})",
            "",
            "| DIAG Tag | Failure Mode | Evidence |",
            "|---|---|---|",
        ]
        for f in run_findings:
            ev = f.evidence[:200].replace("|", "\\|")
            readme_lines.append(f"| {f.diag_tag} | {f.failure_mode} | {ev} |")
        readme_lines += [
            "",
            "## Re-run triage",
            "",
            "```bash",
            f"python -m TraceLens.Agent.Analysis.triage.runner "
            f"--run-dir '{run_dir}' --detailed",
            "```",
            "",
        ]
        with open(os.path.join(pkg_folder, "README.md"), "w") as f:
            f.write("\n".join(readme_lines))

        # Collect artifacts
        for src, arcname in _collectible_paths(run_dir, max_bytes):
            dest = os.path.join(pkg_folder, arcname)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                shutil.copy2(src, dest)
            except OSError:
                pass

        # Create tarball
        tar_path = os.path.join(repro_dir, f"{pkg_name}.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(pkg_folder, arcname=pkg_name)

        # Clean up uncompressed folder
        shutil.rmtree(pkg_folder, ignore_errors=True)
        count += 1

    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate triage results and produce summary report + reproducers"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--traces-root",
        help="Root folder to discover triage_details.csv files recursively",
    )
    input_group.add_argument(
        "--mapping",
        help="Tab-separated mapping file (run_dir<TAB>triage_diags.txt)",
    )
    parser.add_argument(
        "--report-dir", required=True,
        help="Output directory for reports and reproducer packages",
    )
    parser.add_argument(
        "--top-reproducers", type=int, default=3,
        help="Number of reproducer packages to build (default: 3)",
    )
    parser.add_argument(
        "--max-pkg-size-mb", type=int, default=200,
        help="Max size per reproducer tarball in MB (default: 200)",
    )
    args = parser.parse_args()

    report_dir = args.report_dir
    os.makedirs(report_dir, exist_ok=True)

    # Discover or load run directories
    if args.traces_root:
        print(f"Discovering triage CSVs under: {args.traces_root}")
        run_dirs = discover_triage_csvs(args.traces_root)
    else:
        print(f"Loading run dirs from mapping: {args.mapping}")
        run_dirs = load_from_mapping(args.mapping)

    print(f"  Found {len(run_dirs)} run directories")

    if not run_dirs:
        print("No run directories found. Nothing to process.")
        return 1

    # Collect findings
    print("Collecting findings from triage CSVs...")
    findings, total_runs = collect_findings(run_dirs)
    print(f"  Total runs: {total_runs}")
    print(f"  Total findings: {len(findings)}")

    if not findings:
        print("No findings found. All runs are clean.")
        with open(os.path.join(report_dir, "summary_report.md"), "w") as f:
            f.write("# Triage Summary Report\n\nAll runs passed — no findings.\n")
        return 0

    # Aggregate
    agg = aggregate(findings)
    print(f"  Unique failure modes: {len(agg['by_failure_mode'])}")
    print(f"  Categories: {', '.join(sorted(agg['by_category'].keys()))}")

    # Write aggregated CSV
    csv_path = write_aggregated_csv(findings, report_dir)
    print(f"\nAggregated CSV: {csv_path}")

    # Select reproducers
    reproducers = pick_reproducers(findings, agg, top_n=args.top_reproducers)
    print(f"Selected {len(reproducers)} reproducer(s):")
    for run_dir, model, rf in reproducers:
        print(f"  - {model} ({len(rf)} findings)")

    # Write summary report
    report_path = write_summary_report(
        findings, agg, total_runs, reproducers, report_dir
    )
    print(f"\nSummary report: {report_path}")

    # Build reproducer packages
    n_pkgs = build_reproducer_packages(
        reproducers, report_dir, max_pkg_size_mb=args.max_pkg_size_mb
    )
    print(f"Built {n_pkgs} reproducer package(s) in {report_dir}/reproducers/")

    # Final summary
    print("\n" + "=" * 60)
    print("POST-PROCESSING COMPLETE")
    print("=" * 60)
    runs_with = len(set(f.run_dir for f in findings))
    print(f"  Total runs:        {total_runs}")
    print(f"  Runs with issues:  {runs_with} ({_pct(runs_with, total_runs)}%)")
    print(f"  Total findings:    {len(findings)}")
    print()
    print("  Top 5 failure modes:")
    for mode, count in agg["by_failure_mode"].most_common(5):
        n_models = len(agg["models_per_issue"].get(mode, set()))
        print(f"    {count:4d}x  {mode}  ({n_models} models)")
    print()
    print(f"  Summary report:      {report_path}")
    print(f"  Aggregated CSV:      {csv_path}")
    print(f"  Reproducer packages: {report_dir}/reproducers/ ({n_pkgs} packages)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
