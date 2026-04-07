#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unified eval reporting pipeline for unit/e2e repeatability suites."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS_DIR = REPO_ROOT / "evals"
UTILS_DIR = EVALS_DIR / "eval_utils"
SCRIPTS_DIR = EVALS_DIR / "eval_scripts"

SUITE_TO_CSV = {
    "unit": EVALS_DIR / "unit_test_traces.csv",
    "e2e": EVALS_DIR / "e2e_test_traces.csv",
}


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _run_cmd(cmd: List[str], env: Dict[str, str] | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def _run_cmd_capture(cmd: List[str], env: Dict[str, str] | None = None) -> str:
    print(f"[cmd] {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail_parts = []
        if stdout:
            detail_parts.append(f"stdout:\n{stdout}")
        if stderr:
            detail_parts.append(f"stderr:\n{stderr}")
        detail = "\n\n".join(detail_parts) if detail_parts else "No captured output."
        raise RuntimeError(
            f"Command failed ({exc.returncode}): {' '.join(cmd)}\n{detail}"
        ) from exc
    return (proc.stdout or "").strip()


def _copy_latest(run_dir: Path, latest_dir: Path) -> None:
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)


def _suite_list(suite_arg: str) -> Iterable[str]:
    if suite_arg == "both":
        return ("unit", "e2e")
    return (suite_arg,)


def _run_suite(
    suite: str,
    run_root: Path,
    container: str,
    num_repeats: int,
    max_parallel: int,
    sleep_between: int,
) -> Dict[str, str]:
    suite_results = run_root / "results" / f"{suite}_repeatability_results"
    suite_agg = run_root / "aggregates" / suite
    suite_results.mkdir(parents=True, exist_ok=True)
    suite_agg.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CONTAINER"] = container
    env["NUM_REPEATS"] = str(num_repeats)
    env["MAX_PARALLEL"] = str(max_parallel)
    env["SLEEP_BETWEEN"] = str(sleep_between)
    env["TEST_TRACES_CSV"] = str(SUITE_TO_CSV[suite])
    env["RESULTS_ROOT"] = str(suite_results)

    _run_cmd(["bash", str(SCRIPTS_DIR / "run_repeatability_parallel.sh")], env=env)

    env["OUTPUT_DIR"] = str(suite_agg)
    _run_cmd([sys.executable, str(UTILS_DIR / "aggregate_repeatability.py")], env=env)

    return {
        "suite": suite,
        "results_root": str(suite_results),
        "aggregate_root": str(suite_agg),
    }


def _create_issue_from_report(
    report_path: Path,
    run_id: str,
    issue_title: str | None,
    issue_labels: List[str],
    issue_assignee: str | None,
) -> int:
    title = issue_title or f"Eval Loop: Follow-up fixes ({run_id})"
    base_cmd = [
        "gh",
        "issue",
        "create",
        "--title",
        title,
        "--body-file",
        str(report_path),
    ]
    cmd = list(base_cmd)
    for label in issue_labels:
        cmd.extend(["--label", label])
    if issue_assignee:
        cmd.extend(["--assignee", issue_assignee])

    try:
        output = _run_cmd_capture(cmd)
    except RuntimeError as exc:
        # Common failure: requested labels do not exist in repo.
        if issue_labels:
            print(
                "[warn] Failed to create issue with provided labels. "
                "Retrying without labels."
            )
            print(f"[warn] Original error: {exc}")
            fallback_cmd = list(base_cmd)
            if issue_assignee:
                fallback_cmd.extend(["--assignee", issue_assignee])
            output = _run_cmd_capture(fallback_cmd)
        else:
            raise

    # gh issue create returns issue URL; parse number from URL.
    m = re.search(r"/issues/(\d+)", output)
    if not m:
        raise RuntimeError(f"Could not parse created issue number from output: {output}")
    issue_number = int(m.group(1))
    print(f"[done] Auto-created issue #{issue_number}: {output}")
    return issue_number


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run unit/e2e repeatability evals and generate reports."
    )
    parser.add_argument(
        "--suite", choices=("unit", "e2e", "both"), default="both", help="Suite to run."
    )
    parser.add_argument(
        "--container",
        default=os.environ.get("CONTAINER"),
        help="Docker container name used by eval scripts.",
    )
    parser.add_argument("--num-repeats", type=int, default=5)
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--sleep-between", type=int, default=30)
    parser.add_argument(
        "--run-id",
        default=f"eval_report_{_timestamp()}",
        help="Run identifier used under evals/eval_reports/<run_id>.",
    )
    parser.add_argument(
        "--reports-root",
        default=str(EVALS_DIR / "eval_reports"),
        help="Root directory for generated run artifacts.",
    )
    parser.add_argument(
        "--rules-file",
        default=str(UTILS_DIR / "report_section_rules.yaml"),
        help="Classifier rule file.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish generated PR/ticket reports to GitHub comments.",
    )
    parser.add_argument("--pr-number", type=int, default=None)
    parser.add_argument("--issue-number", type=int, default=None)
    parser.add_argument(
        "--auto-create-issue",
        action="store_true",
        help="Auto-create an issue from fix_ticket_report.md when --publish is set and --issue-number is missing.",
    )
    parser.add_argument(
        "--issue-title",
        default=None,
        help="Optional title for auto-created issue (used with --auto-create-issue).",
    )
    parser.add_argument(
        "--issue-label",
        action="append",
        default=[],
        help="Label(s) to apply to auto-created issue. Can be repeated.",
    )
    parser.add_argument(
        "--issue-assignee",
        default=None,
        help="Optional assignee for auto-created issue.",
    )
    parser.add_argument(
        "--comment-on-created-issue",
        action="store_true",
        help="Also post fix-ticket report as comment when issue is auto-created (default: disabled to avoid duplicate body/comment).",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip suite execution and only aggregate/render from existing outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render/validate from existing aggregate outputs and never publish comments.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip post-generation validation checks.",
    )
    args = parser.parse_args()

    if args.dry_run:
        args.skip_run = True
        args.publish = False
        args.auto_create_issue = False

    if not args.container and not args.skip_run:
        parser.error("--container (or CONTAINER env var) is required unless --skip-run is used")

    report_root = Path(args.reports_root)
    run_root = report_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "results").mkdir(parents=True, exist_ok=True)
    (run_root / "aggregates").mkdir(parents=True, exist_ok=True)

    suite_outputs = []
    for suite in _suite_list(args.suite):
        if args.skip_run:
            suite_agg = run_root / "aggregates" / suite
            if not suite_agg.exists():
                raise FileNotFoundError(
                    f"Missing aggregate directory for suite '{suite}': {suite_agg}"
                )
            suite_outputs.append(
                {
                    "suite": suite,
                    "results_root": str(run_root / "results" / f"{suite}_repeatability_results"),
                    "aggregate_root": str(suite_agg),
                }
            )
            continue

        suite_outputs.append(
            _run_suite(
                suite=suite,
                run_root=run_root,
                container=args.container,
                num_repeats=args.num_repeats,
                max_parallel=args.max_parallel,
                sleep_between=args.sleep_between,
            )
        )

    summary_json = run_root / "report_summary.json"
    agg_cmd = [
        sys.executable,
        str(UTILS_DIR / "report_aggregator.py"),
        "--rules-file",
        str(args.rules_file),
        "--output-json",
        str(summary_json),
    ]
    for suite_info in suite_outputs:
        agg_cmd.extend(
            [
                "--suite-aggregate",
                f"{suite_info['suite']}={suite_info['aggregate_root']}",
            ]
        )
    _run_cmd(agg_cmd)

    render_cmd = [
        sys.executable,
        str(UTILS_DIR / "report_renderer.py"),
        "--summary-json",
        str(summary_json),
        "--output-dir",
        str(run_root),
        "--unit-traces-csv",
        str(EVALS_DIR / "unit_test_traces.csv"),
        "--e2e-traces-csv",
        str(EVALS_DIR / "e2e_test_traces.csv"),
    ]
    _run_cmd(render_cmd)

    if not args.no_validate:
        validate_cmd = [
            sys.executable,
            str(UTILS_DIR / "validate_report_outputs.py"),
            "--summary-json",
            str(summary_json),
            "--pr-report",
            str(run_root / "pr_report.md"),
            "--ticket-report",
            str(run_root / "fix_ticket_report.md"),
            "--expected-suites",
            ",".join([s["suite"] for s in suite_outputs]),
        ]
        _run_cmd(validate_cmd)

    latest_dir = report_root / "latest"
    _copy_latest(run_root, latest_dir)

    if args.publish:
        issue_number = args.issue_number
        issue_was_auto_created = False
        if issue_number is None and args.auto_create_issue:
            issue_number = _create_issue_from_report(
                report_path=run_root / "fix_ticket_report.md",
                run_id=args.run_id,
                issue_title=args.issue_title,
                issue_labels=args.issue_label,
                issue_assignee=args.issue_assignee,
            )
            issue_was_auto_created = True

        publish_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "publish_eval_reports.py"),
            "--pr-report-path",
            str(run_root / "pr_report.md"),
            "--ticket-report-path",
            str(run_root / "fix_ticket_report.md"),
        ]
        if args.pr_number is not None:
            publish_cmd.extend(["--pr-number", str(args.pr_number)])
            if issue_number is not None:
                publish_cmd.extend(["--related-issue-number", str(issue_number)])
        if issue_number is not None:
            publish_cmd.extend(["--issue-number", str(issue_number)])
        if issue_was_auto_created and not args.comment_on_created_issue:
            publish_cmd.append("--skip-issue-comment")
        _run_cmd(publish_cmd)

    print(f"[done] Run directory: {run_root}")
    print(f"[done] Latest directory: {latest_dir}")
    print(f"[done] Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
