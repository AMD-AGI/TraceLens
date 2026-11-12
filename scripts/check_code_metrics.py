#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fail a change that adds much complexity to the production modules it touches.

Every function in a changed module under ``TraceLens/`` or ``scripts/`` is measured
at the head and at the merge base: cyclomatic complexity (radon), cognitive complexity
(complexipy), nested block depth (ruff PLR1702) and lines of code in its body.
A function over a limit fails when the base had it under, or when it grows by more
than 10% (rounded up) of its base value. A touched function over a limit within that
growth, and a changed module over its limit, are listed for a separate cleanup;
untouched functions are ignored. A broad ``except`` (ruff BLE001) fails on the lines
the change adds.

Usage:
    pip install radon==6.0.1 complexipy==8.0.1 ruff==0.16.1
    python scripts/check_code_metrics.py --base origin/staging_agent [--head HEAD]
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import re
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path
from typing import Iterator, NamedTuple

LIMITS = {
    "cyclomatic complexity": 20,
    "cognitive complexity": 30,
    "nested blocks": 4,
    "function lines": 60,
    "module lines": 1200,
}
GROWTH = 0.1  # share of its base value a function already over a limit may add
OPEN_ROWS = 10  # a longer cleanup list is collapsed in the report
SOURCE_ROOTS = ("TraceLens", "scripts")
MODULE = "<module>"
ADDED_LINE_RULE = "BLE001"
WAIVER_LABEL = "waive-code-complexity"
_HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@")
_RUFF_VALUE = re.compile(r"\((\d+) > \d+\)")
_NOT_CODE = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
    }
)
_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


class Def(NamedTuple):
    name: str
    start: int
    end: int
    body: int


class Finding(NamedTuple):
    metric: str
    path: str
    unit: str
    base: int | None
    head: int


def run(argv: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        argv, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def changed_modules(since: str, head: str) -> dict[str, str | None]:
    """Changed production modules at ``head``, each mapped to its path at ``since``."""
    modules = {}
    diff = ["git", "diff", "--name-status", "-M", "--diff-filter=AMR", since, head]
    for line in run([*diff, "--", *SOURCE_ROOTS]).splitlines():
        status, *paths = line.split("\t")
        path = Path(paths[-1])
        if path.suffix == ".py" and "tests" not in path.parts:
            if not path.name.startswith("test_"):
                modules[paths[-1]] = None if status == "A" else paths[0]
    return modules


def changed_lines(
    since: str, head: str
) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Lines at ``head`` the change adds, and those plus the line before each deletion."""
    added: dict[str, set[int]] = {}
    touched: dict[str, set[int]] = {}
    path = ""
    diff = ["git", "diff", "-U0", "-M", "--diff-filter=AMR", since, head]
    for line in run([*diff, "--", *SOURCE_ROOTS]).split("\n"):
        if line.startswith("+++ b/"):
            path = line[len("+++ b/") :]
        elif match := _HUNK.match(line):
            start = int(match[1])
            count = 1 if match[2] is None else int(match[2])
            added.setdefault(path, set()).update(range(start, start + count))
            touched.setdefault(path, set()).update(range(start, start + max(count, 1)))
    return added, touched


def export(ref: str, paths: list[str], root: Path) -> None:
    for path in paths:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(run(["git", "show", f"{ref}:{path}"]), encoding="utf-8")


def walk_defs(node: ast.AST, prefix: str = "") -> Iterator[Def]:
    """Functions under ``node`` with their qualified names; classes only name them."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _DOCSTRING_OWNERS):
            name = prefix + child.name
            if not isinstance(child, ast.ClassDef):
                body = child.body[0].lineno
                yield Def(name, child.lineno, child.end_lineno, body)
            yield from walk_defs(child, name + ".")
        else:
            yield from walk_defs(child, prefix)


def code_lines(source: str, tree: ast.Module) -> set[int]:
    """Lines holding code: not blank, not only a comment, not part of a docstring."""
    lines = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in _NOT_CODE:
            lines.update(range(token.start[0], token.end[0] + 1))
    lines -= {
        row for row, text in enumerate(source.splitlines(), 1) if not text.strip()
    }
    for node in ast.walk(tree):
        if isinstance(node, _DOCSTRING_OWNERS) and ast.get_docstring(node) is not None:
            lines -= set(range(node.body[0].lineno, node.body[0].end_lineno + 1))
    return lines


def function_lines(fn: Def, defs: list[Def], code: set[int]) -> int:
    """Code lines of the body of ``fn``, leaving out the functions nested in it."""
    nested = {
        row
        for d in defs
        if d.name.startswith(fn.name + ".") and fn.start < d.start <= fn.end
        for row in range(d.start, d.end + 1)
    }
    return len((set(range(fn.body, fn.end + 1)) - nested) & code)


def owner(defs: list[Def], line: int) -> str:
    """The function starting on ``line``, else the innermost one enclosing it."""
    enclosing = [(d.start, d.name) for d in defs if d.start <= line <= d.end]
    return max(enclosing)[1] if enclosing else MODULE


def touched_units(
    root: Path, paths: list[str], touched: dict[str, set[int]]
) -> set[tuple[str, str]]:
    """(path, function) for each function in ``paths`` holding a touched line."""
    return {
        (path, fn.name)
        for path in paths
        for fn in walk_defs(ast.parse((root / path).read_text(encoding="utf-8")))
        if any(fn.start <= row <= fn.end for row in touched.get(path, ()))
    }


def relative(root: Path, filename: str) -> str:
    return Path(filename).resolve().relative_to(root.resolve()).as_posix()


def ruff(root: Path, paths: list[str], rules: str, *options: str) -> list[dict]:
    argv = ["ruff", "check", "--isolated", "--no-cache", "--exit-zero", "--preview"]
    argv += ["--config", "lint.explicit-preview-rules=true", "--select", rules]
    argv += ["--output-format", "json", *options, *paths]
    return json.loads(run(argv, cwd=root))


def measure(root: Path, paths: list[str]) -> dict[tuple[str, str, str], int]:
    """Every value over its limit in ``paths``, keyed by (metric, path, unit)."""
    if not paths:
        return {}
    values, defs = {}, {}
    for path in paths:
        source = (root / path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        defs[path], code = list(walk_defs(tree)), code_lines(source, tree)
        values["module lines", path, MODULE] = len(source.splitlines())
        for fn in defs[path]:
            values["function lines", path, fn.name] = function_lines(
                fn, defs[path], code
            )
    for path, blocks in json.loads(
        run(["radon", "cc", "--json", *paths], root)
    ).items():
        while blocks:
            block = blocks.pop()
            blocks += block.get("methods", []) + block.get("closures", [])
            if block["type"] != "class":
                unit = owner(defs[path], block["lineno"])
                values["cyclomatic complexity", path, unit] = block["complexity"]
    report = root / "complexipy.json"
    argv = ["complexipy", *paths, "--ignore-complexity", "--no-ignore", "--output"]
    run([*argv, str(report), "--output-format", "json", "--cache-dir", str(root)], root)
    for row in json.loads(report.read_text(encoding="utf-8")):
        unit = row["function_name"].replace("::", ".")
        values["cognitive complexity", row["path"], unit] = int(row["complexity"])
    nesting = f"lint.pylint.max-nested-blocks={LIMITS['nested blocks']}"
    for row in ruff(root, paths, "PLR1702", "--ignore-noqa", "--config", nesting):
        path = relative(root, row["filename"])
        key = ("nested blocks", path, owner(defs[path], row["location"]["row"]))
        depth = int(_RUFF_VALUE.search(row["message"])[1])
        values[key] = max(values.get(key, 0), depth)
    return {key: v for key, v in values.items() if v > LIMITS[key[0]]}


def compare(
    head: dict[tuple[str, str, str], int],
    base: dict[tuple[str, str, str], int],
    touched: set[tuple[str, str]],
) -> tuple[list[Finding], list[Finding]]:
    """Units over a limit: (failing, listed for cleanup)."""
    failing, cleanup = [], []
    for (metric, path, unit), value in sorted(head.items()):
        before = base.get((metric, path, unit))
        finding = Finding(metric, path, unit, before, value)
        grown = before is None or value > before + math.ceil(before * GROWTH)
        if metric != "module lines" and grown:
            failing.append(finding)
        elif metric == "module lines" or (path, unit) in touched:
            cleanup.append(finding)
    return failing, cleanup


def added_line_findings(
    root: Path, paths: list[str], added: dict[str, set[int]]
) -> list[str]:
    """Report rows for the ``ADDED_LINE_RULE`` diagnostics on added lines."""
    flagged = []
    for row in ruff(root, paths, ADDED_LINE_RULE) if paths else []:
        path, line = relative(root, row["filename"]), row["location"]["row"]
        if line in added.get(path, ()):
            flagged.append(f"| `{path}:{line}` | {row['code']} {row['message']} |")
    return flagged


def table(findings: list[Finding]) -> str:
    return "\n".join(
        [
            "| Unit | Metric | Base | Now | Limit |",
            "|---|---|--:|--:|--:|",
            *(
                f"| `{f.path}` `{f.unit}` | {f.metric} | {f.base or '-'} | {f.head} "
                f"| {LIMITS[f.metric]} |"
                for f in findings
            ),
        ]
    )


def render(failing: list[Finding], cleanup: list[Finding], flagged: list[str]) -> str:
    """The markdown report, one blank line between its blocks."""
    verdict = "FAILED" if failing or flagged else "PASSED"
    out = [f"## Code metrics: {verdict}"]
    if failing:
        out += ["### Failing", table(failing)]
    if flagged:
        out += [
            "### Added lines",
            "\n".join(["| Where | Rule |", "|---|---|", *flagged]),
        ]
    if verdict == "FAILED":
        out.append(
            "Split each failing unit until it is under the limit or within"
            f" {GROWTH:.0%} of the base;"
            f" the `{WAIVER_LABEL}` label waives this, with a reason in the PR."
        )
    if cleanup:
        rows = table(cleanup)
        if len(cleanup) > OPEN_ROWS:
            rows = f"<details><summary>{len(cleanup)} units</summary>\n\n{rows}\n\n</details>"
        out += [
            "### Pre-Existing Failures on Files that Require Cleanup",
            f"Already over a limit and grown by no more than {GROWTH:.0%}, or modules"
            f" over {LIMITS['module lines']} lines. They do not fail this change;"
            " please split them when you can, in a follow-up.",
            rows,
        ]
    if any(f.base is None for f in failing + cleanup):
        out.append("Base `-`: new, or under the limit before this change.")
    return "\n\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="ref the change merges into")
    parser.add_argument("--head", default="HEAD", help="ref of the change")
    args = parser.parse_args(argv)
    since = run(["git", "merge-base", args.base, args.head]).strip()
    modules = changed_modules(since, args.head)
    to_head = {old: new for new, old in modules.items() if old}
    with tempfile.TemporaryDirectory() as tmp:
        head_root, base_root = Path(tmp, "head"), Path(tmp, "base")
        export(args.head, list(modules), head_root)
        export(since, list(to_head), base_root)
        head = measure(head_root, list(modules))
        base = {
            (metric, to_head[path], unit): value
            for (metric, path, unit), value in measure(base_root, list(to_head)).items()
        }
        added, touched = changed_lines(since, args.head)
        flagged = added_line_findings(head_root, list(modules), added)
        units = touched_units(head_root, list(modules), touched)
    failing, cleanup = compare(head, base, units)
    print(render(failing, cleanup, flagged))
    return 1 if failing or flagged else 0


if __name__ == "__main__":
    sys.exit(main())
