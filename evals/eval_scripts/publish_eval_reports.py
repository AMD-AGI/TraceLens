#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Publish eval markdown reports to PR/Issue comments with marker-based upsert."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


PR_MARKER = "<!-- eval-report-pr -->"
TICKET_MARKER = "<!-- eval-report-ticket -->"


def _run(cmd: List[str], input_text: str | None = None) -> str:
    proc = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def _repo_name() -> str:
    return _run(["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"])


def _list_issue_comments(repo: str, number: int) -> List[Dict]:
    output = _run(
        [
            "gh",
            "api",
            f"repos/{repo}/issues/{number}/comments",
            "--paginate",
            "--jq",
            ".[]",
        ]
    )
    if not output:
        return []
    comments = []
    for line in output.splitlines():
        try:
            comments.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return comments


def _upsert_issue_comment(repo: str, number: int, marker: str, content: str) -> None:
    body = f"{marker}\n\n{content}".strip()
    comments = _list_issue_comments(repo, number)
    existing = next((c for c in comments if marker in (c.get("body") or "")), None)

    if existing:
        comment_id = existing["id"]
        _run(
            [
                "gh",
                "api",
                f"repos/{repo}/issues/comments/{comment_id}",
                "--method",
                "PATCH",
                "--input",
                "-",
            ],
            input_text=json.dumps({"body": body}),
        )
        print(f"Updated comment {comment_id} on issue/PR #{number}")
        return

    _run(
        [
            "gh",
            "api",
            f"repos/{repo}/issues/{number}/comments",
            "--method",
            "POST",
            "--input",
            "-",
        ],
        input_text=json.dumps({"body": body}),
    )
    print(f"Created new comment on issue/PR #{number}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish eval reports to PR/issue comments.")
    parser.add_argument("--pr-number", type=int, default=None)
    parser.add_argument("--issue-number", type=int, default=None)
    parser.add_argument(
        "--related-issue-number",
        type=int,
        default=None,
        help="Optional fix-ticket issue number to reference in PR comment body.",
    )
    parser.add_argument(
        "--skip-issue-comment",
        action="store_true",
        help="Skip publishing fix-ticket report as issue comment.",
    )
    parser.add_argument("--pr-report-path", required=True)
    parser.add_argument("--ticket-report-path", required=True)
    args = parser.parse_args()

    if args.pr_number is None and args.issue_number is None:
        print("No --pr-number or --issue-number provided; nothing to publish.")
        return

    repo = _repo_name()
    pr_report = Path(args.pr_report_path).resolve().read_text()
    ticket_report = Path(args.ticket_report_path).resolve().read_text()

    if args.pr_number is not None:
        pr_body = pr_report
        if args.related_issue_number is not None:
            related_url = (
                f"https://github.com/{repo}/issues/{args.related_issue_number}"
            )
            pr_body = (
                f"{pr_report.rstrip()}\n\n"
                f"---\n"
                f"**Fix ticket:** [#{args.related_issue_number}]({related_url})\n"
            )
        _upsert_issue_comment(repo, args.pr_number, PR_MARKER, pr_body)
    if args.issue_number is not None and not args.skip_issue_comment:
        _upsert_issue_comment(repo, args.issue_number, TICKET_MARKER, ticket_report)


if __name__ == "__main__":
    main()
