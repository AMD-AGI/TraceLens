###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triage runner — orchestrates checks, writes output files, provides CLI.

Usage:
    python -m TraceLens.Agent.Analysis.triage.runner \
        --run-dir <analysis_output/> [--stream-file <stream.ndjson>]
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

from .checks import ALL_CHECKS, Finding


def _auto_detect_stream(run_dir):
    """Look for an agent stream file in the parent of the run directory."""
    parent = str(Path(run_dir).parent)
    for pattern in ["analysis_stream.ndjson", "*.ndjson", "*.streamJSON"]:
        candidates = sorted(glob.glob(os.path.join(parent, pattern)))
        if candidates:
            return candidates[-1]
    return None


def run_triage(run_dir, stream_file=None):
    """Run all checks and return a list of Findings."""
    if not os.path.isdir(run_dir):
        print(f"[DIAG:pipeline:OUTPUT_INCOMPLETE] Run directory does not exist: {run_dir}")
        sys.exit(1)

    if stream_file and not os.path.isfile(stream_file):
        print(f"Warning: stream file not found: {stream_file} — stream checks will be skipped")
        stream_file = None

    if not stream_file:
        stream_file = _auto_detect_stream(run_dir)
        if stream_file:
            print(f"Auto-detected stream file: {stream_file}")

    findings = []
    findings_by_sublabel = {}
    for spec in ALL_CHECKS:
        try:
            draft = spec.fn(run_dir, stream_file)
        except Exception as e:
            print(f"Warning: check {spec.fn.__name__} raised {e}")
            continue
        if not draft:
            continue
        finding = Finding(
            tag=spec.build_tag(),
            sublabel=spec.sublabel,
            category=spec.category,
            failure_mode=draft.failure_mode,
            evidence=draft.evidence,
            remedy=draft.remedy,
        )
        findings.append(finding)
        if spec.sublabel:
            findings_by_sublabel[spec.sublabel] = finding

    # Second pass: annotate downstream findings with the sublabels of the
    # upstream checks that listed them in `implies_failures` and also failed.
    for spec in ALL_CHECKS:
        if not spec.implies_failures or spec.sublabel not in findings_by_sublabel:
            continue
        for downstream in spec.implies_failures:
            target = findings_by_sublabel.get(downstream)
            if target is not None:
                target.implied_by.append(spec.sublabel)

    for finding in findings:
        print(finding.diag_line())

    return findings


def write_diag_txt(findings, run_dir):
    path = os.path.join(run_dir, "triage_diags.txt")
    with open(path, "w") as f:
        if not findings:
            f.write("No failures detected. Analysis run appears healthy.\n")
        else:
            for finding in findings:
                f.write(finding.diag_line() + "\n")


def write_detail_csv(findings, run_dir):
    path = os.path.join(run_dir, "triage_details.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["DIAG Tag", "Category", "Failure Mode", "Evidence", "Remedy", "Implied By"])
        for finding in findings:
            writer.writerow([
                finding.tag,
                finding.category,
                finding.failure_mode,
                finding.evidence,
                finding.remedy,
                ",".join(finding.implied_by),
            ])


def main():
    parser = argparse.ArgumentParser(description="TraceLens Analysis Triage Checker")
    parser.add_argument("--run-dir", required=True, help="Path to analysis_output/ directory")
    parser.add_argument("--stream-file", default=None, help="Path to agent stream (.ndjson or .streamJSON)")
    args = parser.parse_args()

    findings = run_triage(args.run_dir, args.stream_file)
    write_detail_csv(findings, args.run_dir)
    write_diag_txt(findings, args.run_dir)

    print("\n" + "=" * 60)
    print("TRIAGE RESULTS")
    print("=" * 60)
    if findings:
        for f in findings:
            print(f.diag_line())
        print(f"\n{len(findings)} failure(s) detected.")
        print(f"Details: {os.path.join(args.run_dir, 'triage_details.csv')}")
        sys.exit(1)
    else:
        print("No failures detected. Analysis run appears healthy.")
        print("=" * 60)


if __name__ == "__main__":
    main()
