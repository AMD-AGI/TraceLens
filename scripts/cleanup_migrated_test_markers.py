#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Remove migration marker comments and hoist duplicate nested imports in tests."""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIGRATION_MARKER = re.compile(r"^\s*#\s*---\s*migrated from\b")

# Keep imports inside these function names (registry isolation, import hooks, etc.).
KEEP_NESTED_IN_FUNCTIONS = frozenset(
    {
        "test_categorize_torch_op_after_registry_reset",
        "test_register_perf_model_categories_idempotent",
        "test_gemm_origami_import_error_path",
        "test_tl_extension_arch_override",
        "test_reload_module_after_monkeypatch",
        "test_caplog_nccl_warning",
    }
)

# Skip hoisting when the enclosing function body mentions these patterns.
KEEP_NESTED_BODY_PATTERNS = (
    "sys.modules",
    "monkeypatch.setitem(sys.modules",
    "registry.clear",
    "OP_CATEGORY_REGISTRY.clear",
    "importlib.reload",
    "importlib.import_module",
    "builtins.__import__",
    "__import__ =",
    "caplog.at_level",
    "_require_cuda_torch",
    "_require_torch",
    "pytest.importorskip",
    " as mod",
    "tests.test_",
    "TraceLens.PerfModel.benchmarking",
    "TraceLens.EventReplay",
    "import origami",
)


def _is_migration_marker(line: str) -> bool:
    return bool(MIGRATION_MARKER.match(line))


def remove_migration_markers(source: str) -> tuple[str, int]:
    lines = source.splitlines(keepends=True)
    kept: list[str] = []
    removed = 0
    for line in lines:
        if _is_migration_marker(line):
            removed += 1
            continue
        kept.append(line)
    text = "".join(kept)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text, removed


def _import_key(node: ast.stmt) -> str | None:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return ast.unparse(node)
    return None


def _function_body_text(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    parts: list[str] = []
    for node in func.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            parts.append(ast.unparse(node))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                parts.append(node.value.value)
    return "\n".join(parts)


def _should_keep_nested(
    func: ast.FunctionDef | ast.AsyncFunctionDef, node: ast.stmt
) -> bool:
    if func.name in KEEP_NESTED_IN_FUNCTIONS:
        return True
    body = _function_body_text(func)
    return any(pattern in body for pattern in KEEP_NESTED_BODY_PATTERNS)


def _find_enclosing_function(
    tree: ast.Module, lineno: int
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    best: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno or 0
            end = node.end_lineno or start
            if start <= lineno <= end:
                if best is None or start >= (best.lineno or 0):
                    best = node
    return best


def hoist_duplicate_nested_imports(source: str) -> tuple[str, int]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    header_keys: set[str] = set()
    header_end = 0
    for node in tree.body:
        key = _import_key(node)
        if key:
            header_keys.add(key)
            header_end = max(header_end, node.end_lineno or 0)
        elif isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            continue
        else:
            break

    extra_nodes: list[ast.stmt] = []
    remove_ranges: list[tuple[int, int]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if (node.lineno or 0) <= header_end:
            continue
        func = _find_enclosing_function(tree, node.lineno or 0)
        if func is None:
            continue
        if _should_keep_nested(func, node):
            continue
        key = _import_key(node)
        if not key:
            continue
        if key not in header_keys:
            extra_nodes.append(node)
            header_keys.add(key)
        remove_ranges.append((node.lineno or 0, node.end_lineno or node.lineno or 0))

    if not extra_nodes and not remove_ranges:
        return source, 0

    insert_at = header_end
    new_import_lines = [ast.unparse(n) + "\n" for n in extra_nodes]
    if new_import_lines:
        lines[insert_at:insert_at] = new_import_lines

    offset = len(new_import_lines)
    for start, end in sorted(remove_ranges, reverse=True):
        start_idx = start - 1 + offset
        end_idx = end + offset
        del lines[start_idx:end_idx]
        if end_idx < len(lines) and lines[end_idx - 1].strip() == "":
            if start_idx > 0 and lines[start_idx - 1].strip() == "":
                del lines[start_idx - 1]
                offset -= 1

    return "".join(lines), len(remove_ranges)


def cleanup_file(path: str, dry_run: bool = False) -> tuple[int, int]:
    with open(path, encoding="utf-8") as handle:
        source = handle.read()

    source, markers_removed = remove_migration_markers(source)
    source, imports_hoisted = hoist_duplicate_nested_imports(source)

    if source.endswith("\n"):
        pass
    elif source:
        source += "\n"

    if (markers_removed or imports_hoisted) and not dry_run:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)

    return markers_removed, imports_hoisted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=[os.path.join(REPO_ROOT, "tests")])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_markers = 0
    total_hoists = 0
    changed_files = 0

    for base in args.paths:
        if os.path.isfile(base):
            files = [base]
        else:
            files = []
            for root, _dirs, names in os.walk(base):
                for name in names:
                    if name.startswith("test_") and name.endswith(".py"):
                        files.append(os.path.join(root, name))
        for path in sorted(files):
            markers, hoists = cleanup_file(path, dry_run=args.dry_run)
            if markers or hoists:
                changed_files += 1
                total_markers += markers
                total_hoists += hoists
                print(f"{path}: removed {markers} markers, hoisted {hoists} imports")

    if not args.dry_run and changed_files:
        subprocess.run(
            [sys.executable, "-m", "black", *args.paths],
            cwd=REPO_ROOT,
            check=False,
        )

    print(
        f"Done: {changed_files} files, {total_markers} markers removed, "
        f"{total_hoists} nested imports hoisted"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
