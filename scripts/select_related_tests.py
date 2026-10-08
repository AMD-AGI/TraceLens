#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Choose pytest files for a pull request from the paths it changes.

A test is related when it imports a changed module, directly or through another
module, when loading it runs a package ``__init__`` that imports the changed
module, or when its source names a changed path, including pieces passed to
``os.path.join``. Changes that can break collection for the rest of the suite
run every test: package metadata, ``tests/conftest.py``, and modules imported
while loading ``TraceLens``.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Constants

PACKAGE_ROOT = "TraceLens"
COPYRIGHT_TEST = "tests/test_copyright_headers.py"
RUN_ALL_PATHS = frozenset(
    {
        "setup.py",
        "MANIFEST.in",
        "setup.cfg",
        "pyproject.toml",
        ".coveragerc",
        "codecov.yml",
        ".test_durations",
        ".github/workflows/unit-tests.yml",
        "scripts/select_related_tests.py",
        "tests/conftest.py",
    }
)
COPYRIGHT_SUFFIXES = frozenset({".py", ".md", ".yml", ".yaml", ".ipynb"})
COPYRIGHT_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ipynb_checkpoints",
        "node_modules",
        "venv",
        "env",
        ".venv",
    }
)
_SKIP_WALK_DIRS = COPYRIGHT_SKIP_DIR_NAMES | {"TraceLens.egg-info"}
_GITHUB_OUTPUT_DELIMITER = "__SELECT_RELATED_TESTS__"


class Selection:
    """Pytest paths to run, or the full suite when a subset is not safe."""

    def __init__(self, mode, test_files):
        self.mode = mode
        self.test_files = tuple(test_files)

    def __eq__(self, other):
        return (
            isinstance(other, Selection)
            and self.mode == other.mode
            and self.test_files == other.test_files
        )

    def __repr__(self):
        return f"Selection(mode={self.mode!r}, test_files={self.test_files!r})"


class _ImportCollector(ast.NodeVisitor):
    """Static imports and path-like string literals from one module."""

    def __init__(self, package):
        self.package = package
        self.imports = set()
        self.paths = set()

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        base = _relative_module(self.package, node.level, node.module)
        if base:
            self.imports.add(base)
            for alias in node.names:
                if alias.name != "*":
                    self.imports.add(f"{base}.{alias.name}")
        self.generic_visit(node)

    def visit_If(self, node):
        # Typing-only imports are not executed at collection time.
        if _is_type_checking(node.test):
            return
        self.generic_visit(node)

    def visit_Call(self, node):
        if _call_name(node) == "import_module" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                self.imports.add(arg.value)
        suffix = _path_join_suffix(node)
        if suffix:
            self.paths.add(suffix)
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and "/" in node.value and len(node.value) <= 500:
            self.paths.add(node.value)
        self.generic_visit(node)


def select_related_tests(repo_root, changed_paths):
    """Return the PR test selection for *changed_paths* under *repo_root*."""
    root = Path(repo_root).resolve()
    changed = tuple(_dedupe(_normalize(path) for path in changed_paths))
    if any(_forces_full_suite(path) for path in changed):
        return Selection("all", ())

    py_files = _python_files(root)
    modules, aliases = _index_modules(py_files)
    for path in changed:
        for name in _module_names(path):
            modules.setdefault(name, path)
            aliases.setdefault(name, name)

    imports, path_literals = _parse_imports(root, py_files, modules)
    resolved = {
        module: _resolve_imports(names, aliases) for module, names in imports.items()
    }
    if _touches_package_import_cone(changed, modules, resolved):
        return Selection("all", ())

    dependents = _dependents(resolved, modules)
    seeds = set()
    for path in changed:
        seeds.update(
            _seeds_for_path(path, modules, aliases, resolved, path_literals, root)
        )

    affected = _walk(seeds, dependents)
    selected = set()
    for module in affected:
        path = modules.get(aliases.get(module, module))
        if path and _is_test_path(path):
            selected.add(path)
    for path in changed:
        if _is_test_path(path) and (root / path).is_file():
            selected.add(path)
    if _needs_copyright_test(changed) and (root / COPYRIGHT_TEST).is_file():
        selected.add(COPYRIGHT_TEST)

    existing = tuple(sorted(path for path in selected if (root / path).is_file()))
    if not existing:
        return Selection("none", ())
    return Selection("selected", existing)


def git_changed_paths(repo_root, base, head="HEAD"):
    """Paths changed between the merge-base of *base*/*head* and *head*."""
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.quotepath=false",
            "diff",
            "--name-status",
            "--find-renames",
            f"{base}...{head}",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_name_status(result.stdout)


def parse_name_status(text):
    """Return changed paths from ``git diff --name-status`` output."""
    paths = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status[:1] in {"R", "C"} and len(parts) >= 3:
            paths.extend(parts[1:])
        elif len(parts) >= 2:
            paths.append(parts[-1])
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--base", help="Diff against this rev (three-dot) to find changed paths"
    )
    parser.add_argument("--head", default="HEAD")
    parser.add_argument(
        "--changed-file", action="append", default=[], help="Changed path, repeatable"
    )
    parser.add_argument(
        "--changed-from", type=Path, help="File of changed paths, one per line"
    )
    parser.add_argument("--all", action="store_true", help="Select the full suite")
    parser.add_argument(
        "--github-output", type=Path, help="Append mode and test_files for Actions"
    )
    parser.add_argument(
        "--write-list", type=Path, help="Write selected test paths, one per line"
    )
    args = parser.parse_args(argv)

    if args.all:
        selection = Selection("all", ())
        changed = []
    else:
        changed = list(args.changed_file)
        if args.changed_from:
            changed.extend(
                line.strip()
                for line in args.changed_from.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if args.base:
            changed.extend(git_changed_paths(args.repo_root, args.base, args.head))
        if not changed and not args.base and args.changed_from is None:
            parser.error("pass --all, --base, --changed-file, or --changed-from")
        selection = select_related_tests(args.repo_root, changed)

    print(f"mode={selection.mode}")
    for path in selection.test_files:
        print(path)
    if changed:
        print(f"# {len(changed)} changed path(s)", file=sys.stderr)
    print(
        f"# {selection.mode}: {len(selection.test_files)} test file(s)",
        file=sys.stderr,
    )
    if args.github_output:
        _write_github_output(args.github_output, selection)
    if args.write_list:
        text = "\n".join(selection.test_files)
        args.write_list.write_text(text + "\n", encoding="utf-8")
    return 0


def _write_github_output(path, selection):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"mode={selection.mode}\n")
        handle.write(f"test_files<<{_GITHUB_OUTPUT_DELIMITER}\n")
        for test_path in selection.test_files:
            handle.write(f"{test_path}\n")
        handle.write(f"{_GITHUB_OUTPUT_DELIMITER}\n")


def _forces_full_suite(path):
    return (
        path in RUN_ALL_PATHS
        or path.endswith(".egg-info")
        or "/.egg-info/" in f"/{path}"
    )


def _touches_package_import_cone(changed, modules, resolved):
    """Modules loaded by ``import TraceLens`` can fail every test at collection."""
    if PACKAGE_ROOT not in modules:
        return False
    cone = _walk({PACKAGE_ROOT}, resolved)
    for path in changed:
        for name in _module_names(path):
            canonical = _canonical(name, _alias_map(modules))
            if canonical in cone:
                return True
    return False


def _seeds_for_path(path, modules, aliases, resolved, path_literals, root):
    seeds = set()
    for name in _module_names(path):
        canonical = _canonical(name, aliases)
        if canonical:
            seeds.add(canonical)
        if Path(path).name == "__init__.py" and canonical:
            prefix = canonical + "."
            for module, deps in resolved.items():
                if any(dep == canonical or dep.startswith(prefix) for dep in deps):
                    seeds.add(module)
    for module, literals in path_literals.items():
        if any(_literal_matches_path(literal, path) for literal in literals):
            seeds.add(module)
    for neighbor in _python_neighbors(root, path):
        for name in _module_names(neighbor):
            canonical = _canonical(name, aliases)
            if canonical:
                seeds.add(canonical)
    return seeds


def _python_neighbors(root, path):
    """Library modules that load a changed data file from a nearby directory.

    Python files are seeds on their own. Data files usually have no import edge,
    so the modules beside them (or one directory up) stand in.
    """
    if path.endswith(".py") or not path.startswith(PACKAGE_ROOT + "/"):
        return []
    directory = Path(path).parent
    neighbors = _py_in_dir(root, directory)
    if not neighbors and str(directory) not in {"", "."}:
        neighbors = _py_in_dir(root, directory.parent)
    return [item for item in neighbors if item != path]


def _py_in_dir(root, directory):
    folder = root / directory
    if not folder.is_dir():
        return []
    return [
        _normalize(str(path.relative_to(root))) for path in sorted(folder.glob("*.py"))
    ]


def _needs_copyright_test(changed):
    for path in changed:
        suffix = Path(path).suffix.lower()
        if suffix not in COPYRIGHT_SUFFIXES:
            continue
        if Path(path).name in {"__init__.py", "LICENSE"}:
            continue
        if any(part in COPYRIGHT_SKIP_DIR_NAMES for part in Path(path).parts):
            continue
        if ".egg-info" in Path(path).parts:
            continue
        return True
    return False


def _parse_imports(root, py_files, modules):
    imports = {}
    path_literals = {}
    alias_of = _alias_map(modules)
    for rel in py_files:
        source_path = root / rel
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=rel)
        except (OSError, SyntaxError, UnicodeError) as exc:
            print(f"# skip {rel}: {exc}", file=sys.stderr)
            continue
        names = _module_names(rel)
        canonical = _canonical(names[0], alias_of) if names else None
        if not canonical:
            continue
        is_package = Path(rel).name == "__init__.py"
        collector = _ImportCollector(
            canonical if is_package else canonical.rpartition(".")[0]
        )
        collector.visit(tree)
        imports[canonical] = collector.imports
        path_literals[canonical] = collector.paths
    return imports, path_literals


def _resolve_imports(names, aliases):
    resolved = set()
    for name in names:
        canonical = _canonical(name, aliases)
        if canonical is None:
            canonical = _longest_known_prefix(name, aliases)
        if canonical:
            resolved.add(canonical)
    return resolved


def _longest_known_prefix(name, aliases):
    parts = name.split(".")
    while len(parts) > 1:
        parts.pop()
        canonical = _canonical(".".join(parts), aliases)
        if canonical:
            return canonical
    return None


def _dependents(resolved, modules):
    """Reverse import edges, plus the package ``__init__`` that runs on import.

    Importing ``pkg.sub`` executes ``pkg/__init__.py``. A test of ``pkg.sub``
    therefore runs every module that ``pkg/__init__.py`` imports, even when the
    test never names those modules.
    """
    packages = {
        name for name, rel in modules.items() if Path(rel).name == "__init__.py"
    }
    dependents = defaultdict(set)
    for module, deps in resolved.items():
        for dep in deps:
            if dep != module:
                dependents[dep].add(module)
            parent = dep.rpartition(".")[0]
            if parent in packages and parent != module:
                dependents[parent].add(module)
    return dependents


def _walk(seeds, edges):
    seen = set()
    stack = list(seeds)
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        seen.add(module)
        stack.extend(edges.get(module, ()))
    return seen


def _index_modules(py_files):
    """Map import names to repo-relative paths. Top-level tests alias bare stems."""
    modules = {}
    aliases = {}
    for rel in py_files:
        names = _module_names(rel)
        if not names:
            continue
        canonical = names[0]
        modules[canonical] = rel
        aliases[canonical] = canonical
        path = Path(rel)
        if path.parts[0] == "tests" and len(path.parts) == 2:
            aliases[path.stem] = canonical
    return modules, aliases


def _alias_map(modules):
    aliases = {name: name for name in modules}
    for name, rel in modules.items():
        path = Path(rel)
        if path.parts and path.parts[0] == "tests" and len(path.parts) == 2:
            aliases[path.stem] = name
    return aliases


def _canonical(name, aliases):
    if name in aliases:
        return aliases[name]
    return None


def _module_names(rel):
    path = Path(rel)
    if path.suffix != ".py" or not path.parts:
        return []
    if path.parts[0] not in {PACKAGE_ROOT, "tests"}:
        return []
    if path.name == "__init__.py":
        parts = path.parts[:-1]
    else:
        parts = path.parts[:-1] + (path.stem,)
    if not parts:
        return []
    return [".".join(parts)]


def _python_files(root):
    found = []
    for top in (PACKAGE_ROOT, "tests"):
        base = root / top
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in _SKIP_WALK_DIRS and not name.endswith(".egg-info")
            ]
            for filename in filenames:
                if filename.endswith(".py"):
                    found.append(
                        _normalize(str(Path(dirpath, filename).relative_to(root)))
                    )
    return found


def _is_test_path(path):
    name = Path(path).name
    return (
        path.startswith("tests/") and name.startswith("test_") and name.endswith(".py")
    )


def _literal_matches_path(literal, changed):
    lit = literal.replace("\\", "/").strip()
    while lit.startswith("./"):
        lit = lit[2:]
    if "/" not in lit or len(lit) < 4 or lit.startswith(("/", "http://", "https://")):
        return False
    lit = lit.strip("/")
    changed = changed.strip("/")
    if lit == changed or changed.startswith(lit + "/") or lit.startswith(changed + "/"):
        return True
    # A relative tail such as eval_utils/workflow_scripted_evals.py.
    filename = lit.rsplit("/", 1)[-1]
    return "." in filename and (changed == lit or changed.endswith("/" + lit))


def _path_join_suffix(node):
    """Constant path tail of an ``os.path.join`` or ``Path.joinpath`` call.

    Dynamic arguments reset the tail, so ``join(root, "eval_utils", "file.py")``
    still yields ``eval_utils/file.py``.
    """
    if not isinstance(node, ast.Call) or not _is_path_join(node):
        return None
    parts = []
    for arg in node.args:
        piece = _constant_path_piece(arg)
        if piece is None:
            parts = []
            continue
        parts.extend(part for part in piece.strip("/").split("/") if part)
    # Directory-only tails such as ``TraceLens/TraceUtils`` are import roots,
    # not a claim that every file under them is covered by this test.
    if len(parts) < 2 or "." not in parts[-1]:
        return None
    suffix = "/".join(parts)
    if len(suffix) < 8:
        return None
    return suffix


def _constant_path_piece(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.replace("\\", "/")
    if isinstance(node, ast.Call):
        return _path_join_suffix(node)
    return None


def _is_path_join(node):
    func = node.func
    if isinstance(func, ast.Name) and func.id in {"join", "joinpath", "Path"}:
        return True
    if isinstance(func, ast.Attribute) and func.attr in {"join", "joinpath"}:
        # ``" ".join(...)`` is string joining, not a filesystem path.
        return not isinstance(func.value, ast.Constant)
    return False


def _relative_module(package, level, module):
    if level == 0:
        return module
    parts = package.split(".") if package else []
    if level > len(parts):
        return None
    prefix = parts[: len(parts) - level + 1]
    if module:
        prefix.extend(module.split("."))
    if not prefix:
        return None
    return ".".join(prefix)


def _is_type_checking(node):
    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    if isinstance(node, ast.Attribute):
        return node.attr == "TYPE_CHECKING"
    return False


def _call_name(node):
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _normalize(path):
    text = path.replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


def _dedupe(paths):
    seen = set()
    for path in paths:
        if path and path not in seen:
            seen.add(path)
            yield path


if __name__ == "__main__":
    sys.exit(main())
