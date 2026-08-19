#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Remove mid-file copyright/import blocks and dedupe header imports in tests."""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MIDFILE_COPYRIGHT = re.compile(
    r"\n###############################################################################\n"
    r"# Copyright \(c\) [^\n]+\n#\n"
    r"# See LICENSE for license information\.\n"
    r"###############################################################################\n"
)

DUPLICATE_SETUP = re.compile(
    r"^(?:" r"REPO_ROOT = .+|" r"ANALYSIS_DIR = .+|" r"sys\.path\.insert\(.+\)" r")$"
)


def _consume_duplicate_setup_lines(lines: list[str], start: int) -> int:
    """Return index after consecutive duplicate import/path-setup lines."""
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if DUPLICATE_SETUP.match(stripped):
            i += 1
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            if stripped.endswith("("):
                i += 1
                while i < len(lines):
                    if lines[i].strip().endswith(")"):
                        i += 1
                        break
                    i += 1
                continue
            i += 1
            continue
        break
    return i


def _leading_copyright_end(lines: list[str]) -> int:
    if not lines or not lines[0].startswith(
        "###############################################################################"
    ):
        return 0
    for i in range(1, min(8, len(lines))):
        if lines[i].startswith(
            "###############################################################################"
        ):
            return i + 1
    return 0


def _leading_prologue_end(lines: list[str]) -> int:
    i = _leading_copyright_end(lines)
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        break
    return i


def _parse_import_nodes(text: str) -> list[ast.stmt]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    return [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def _extract_setup_section(lines: list[str], start: int) -> tuple[int, list[ast.stmt]]:
    end = _consume_duplicate_setup_lines(lines, start)
    chunk = "".join(lines[start:end])
    return end, _parse_import_nodes(chunk)


def remove_midfile_blocks(source: str) -> tuple[str, int, list[ast.stmt]]:
    removed = 0
    extracted: list[ast.stmt] = []
    lines = source.splitlines(keepends=True)
    while True:
        block_start = None
        for i in range(10, len(lines) - 4):
            if (
                lines[i].startswith(
                    "###############################################################################"
                )
                and "Copyright" in lines[i + 1]
            ):
                block_start = i
                break
        if block_start is None:
            break
        setup_end, block_imports = _extract_setup_section(lines, block_start + 5)
        extracted.extend(block_imports)
        if block_imports or setup_end > block_start + 5:
            lines = lines[:block_start] + lines[setup_end:]
        else:
            lines = lines[:block_start] + lines[block_start + 5 :]
        removed += 1
    source = "".join(lines)
    source = re.sub(r"\n{4,}", "\n\n\n", source)
    return source, removed, extracted


def _header_boundary(tree: ast.Module) -> int:
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                continue
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.lineno or 0
    return 10**9


def _is_type_checking_import(node: ast.stmt, tree: ast.Module) -> bool:
    for parent in tree.body:
        if isinstance(parent, ast.If):
            test = parent.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                if node in parent.body:
                    return True
    return False


def _is_path_setup(node: ast.stmt) -> bool:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {
                "REPO_ROOT",
                "ANALYSIS_DIR",
            }:
                return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call = node.value
        if isinstance(call.func, ast.Attribute) and call.func.attr == "insert":
            value = call.func.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "path"
                and isinstance(value.value, ast.Name)
                and value.value.id == "sys"
            ):
                return True
    return False


def _merge_from_imports(nodes: list[ast.ImportFrom]) -> list[ast.ImportFrom]:
    grouped: dict[tuple[str | None, int], list[ast.alias]] = defaultdict(list)
    order: list[tuple[str | None, int]] = []
    for node in nodes:
        key = (node.module, node.level)
        if key not in order:
            order.append(key)
        seen = {(alias.name, alias.asname) for alias in grouped[key]}
        for alias in node.names:
            pair = (alias.name, alias.asname)
            if pair not in seen:
                grouped[key].append(alias)
                seen.add(pair)
    merged: list[ast.ImportFrom] = []
    for key in order:
        module, level = key
        names = sorted(grouped[key], key=lambda alias: (alias.name, alias.asname or ""))
        merged.append(ast.ImportFrom(module=module, names=names, level=level))
    return merged


def _merge_import_nodes(nodes: list[ast.stmt]) -> list[ast.stmt]:
    plain: dict[str, str | None] = {}
    plain_order: list[str] = []
    from_nodes: list[ast.ImportFrom] = []
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in plain:
                    plain_order.append(alias.name)
                plain[alias.name] = alias.asname
        elif isinstance(node, ast.ImportFrom):
            from_nodes.append(node)
    merged: list[ast.stmt] = []
    if plain_order:
        merged.append(
            ast.Import(
                names=[ast.alias(name=name, asname=plain[name]) for name in plain_order]
            )
        )
    merged.extend(_merge_from_imports(from_nodes))
    return merged


def fix_future_imports(source: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    future_ranges: list[tuple[int, int]] = []
    insert_at = 0
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                insert_at = max(insert_at, node.end_lineno or 0)
                continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            future_ranges.append(
                (node.lineno or 0, node.end_lineno or node.lineno or 0)
            )
        elif future_ranges:
            break

    if not future_ranges:
        return source

    future_lines: list[str] = []
    for start, end in sorted(future_ranges, reverse=True):
        future_lines[:0] = lines[start - 1 : end]
        del lines[start - 1 : end]
        if start <= insert_at:
            insert_at -= end - start + 1

    lines[insert_at:insert_at] = future_lines
    return "".join(lines)


def fix_header_order(source: str) -> str:
    """Place REPO_ROOT/sys.path setup after plain imports, before from-imports."""
    tree = ast.parse(source)
    boundary = _header_boundary(tree)
    lines = source.splitlines(keepends=True)

    path_ranges: list[tuple[int, int]] = []
    plain_import_end = 0
    first_from_import = 0
    for node in tree.body:
        if (node.lineno or 0) >= boundary:
            break
        if _is_path_setup(node):
            path_ranges.append((node.lineno or 0, node.end_lineno or node.lineno or 0))
        elif isinstance(node, ast.Import):
            plain_import_end = max(plain_import_end, node.end_lineno or 0)
        elif isinstance(node, ast.ImportFrom) and not _is_type_checking_import(
            node, tree
        ):
            if not first_from_import:
                first_from_import = node.lineno or 0

    if not path_ranges:
        return source

    target = plain_import_end or first_from_import - 1
    if first_from_import and path_ranges:
        current_start = path_ranges[0][0]
        if current_start <= first_from_import:
            return source
        target = plain_import_end or max(0, first_from_import - 1)

    path_lines: list[str] = []
    for start, end in sorted(path_ranges, reverse=True):
        path_lines[:0] = lines[start - 1 : end]
        del lines[start - 1 : end]
        if start <= target:
            target -= end - start + 1

    lines[target:target] = path_lines
    return "".join(lines)


def dedupe_header_imports(
    source: str, extra_imports: list[ast.stmt] | None = None
) -> tuple[str, int]:
    tree = ast.parse(source)
    boundary = _header_boundary(tree)
    lines = source.splitlines(keepends=True)

    header_imports: list[ast.stmt] = []
    remove_ranges: list[tuple[int, int]] = []
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if _is_type_checking_import(node, tree):
            continue
        if (node.lineno or 0) >= boundary:
            header_imports.append(node)
            remove_ranges.append(
                (node.lineno or 0, node.end_lineno or node.lineno or 0)
            )
            continue
        header_imports.append(node)
        remove_ranges.append((node.lineno or 0, node.end_lineno or node.lineno or 0))

    if extra_imports:
        header_imports.extend(extra_imports)

    if not header_imports:
        return source, 0

    merged = _merge_import_nodes(header_imports)
    if len(merged) >= len(header_imports) and not extra_imports:
        return source, 0

    insert_at = _leading_prologue_end(lines)
    for node in tree.body:
        if (node.lineno or 0) >= boundary:
            break
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                insert_at = max(insert_at, node.end_lineno or 0)
                continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not _is_type_checking_import(node, tree):
                break
        if _is_path_setup(node):
            break

    insert_lines = [ast.unparse(node) + "\n" for node in merged]
    for start, end in sorted(remove_ranges, reverse=True):
        length = end - start + 1
        if start <= insert_at:
            insert_at -= length
        del lines[start - 1 : end]

    lines[insert_at:insert_at] = insert_lines
    text = "".join(lines)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = fix_header_order(text)
    return text, len(header_imports) - len(merged)


def consolidate_file(path: str, dry_run: bool = False) -> tuple[int, int]:
    with open(path, encoding="utf-8") as handle:
        original = handle.read()

    source = original
    source, blocks_removed, extracted = remove_midfile_blocks(source)
    source, imports_merged = dedupe_header_imports(source, extracted)
    source = fix_header_order(source)
    source = fix_future_imports(source)

    if not source.endswith("\n"):
        source += "\n"

    if source != original and not dry_run:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)

    return blocks_removed, imports_merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=[os.path.join(REPO_ROOT, "tests")])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    changed = 0
    total_blocks = 0
    total_imports = 0
    py_files: list[str] = []

    for base in args.paths:
        if os.path.isfile(base):
            py_files.append(base)
        elif os.path.isdir(base):
            for root, _dirs, names in os.walk(base):
                for name in names:
                    if name.startswith("test_") and name.endswith(".py"):
                        py_files.append(os.path.join(root, name))

    for path in sorted(set(py_files)):
        try:
            blocks, imports = consolidate_file(path, dry_run=args.dry_run)
        except SyntaxError as exc:
            print(f"{path}: SKIPPED syntax error: {exc}")
            continue
        if blocks or imports:
            changed += 1
            total_blocks += blocks
            total_imports += imports
            print(
                f"{path}: removed {blocks} mid-file blocks, "
                f"deduped {imports} header imports"
            )

    if not args.dry_run and changed:
        subprocess.run(
            [sys.executable, "-m", "black", *sorted(set(py_files))],
            cwd=REPO_ROOT,
            check=False,
        )

    print(
        f"Done: {changed} files, {total_blocks} blocks removed, "
        f"{total_imports} header imports deduped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
