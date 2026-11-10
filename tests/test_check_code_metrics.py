###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the code metrics gate."""

import ast
import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "check_code_metrics",
    ROOT / "scripts" / "check_code_metrics.py",
)
metrics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(metrics)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo, message):
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", *identity, "commit", "-qm", message], cwd=repo, check=True)
    return metrics.run(["git", "rev-parse", "HEAD"], repo).strip()


def test_compare_fails_new_and_worse_units_and_lists_module_growth():
    cyclomatic, module = "cyclomatic complexity", "module lines"
    head = {
        (cyclomatic, "m.py", "added"): 25,
        (cyclomatic, "m.py", "grown"): 30,
        (cyclomatic, "m.py", "kept"): 22,
        (module, "m.py", metrics.MODULE): 1300,
    }
    base = {
        (cyclomatic, "m.py", "grown"): 28,
        (cyclomatic, "m.py", "kept"): 22,
        (module, "m.py", metrics.MODULE): 1250,
    }
    failing, listed = metrics.compare(head, base)
    assert failing == [
        metrics.Finding(cyclomatic, "m.py", "added", None, 25),
        metrics.Finding(cyclomatic, "m.py", "grown", 28, 30),
    ]
    assert listed == [metrics.Finding(module, "m.py", metrics.MODULE, 1250, 1300)]


def test_function_lines_count_body_code_only():
    source = '''class Worker:
    def run(self, a):
        """Docstring
        on two lines."""
        # comment

        x = a

        def inner():
            return 1

        text = """

        """
        return x, text
'''
    tree = ast.parse(source)
    defs = list(metrics.walk_defs(tree))
    code = metrics.code_lines(source, tree)
    sizes = {d.name: metrics.function_lines(d, defs, code) for d in defs}
    assert sizes == {"Worker.run": 4, "Worker.run.inner": 1}


def test_changed_modules_follow_renames_and_skip_tests(tmp_path, monkeypatch):
    body = "".join(f"value_{i} = {i}\n" for i in range(10))
    _write(tmp_path / "TraceLens" / "kept.py", "a = 1\nb = 2\n")
    _write(tmp_path / "TraceLens" / "old.py", body)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    base = _commit(tmp_path, "base")
    _write(tmp_path / "TraceLens" / "kept.py", "a = 1\nb = 2\nc = 3\n")
    (tmp_path / "TraceLens" / "old.py").rename(tmp_path / "TraceLens" / "new.py")
    _write(tmp_path / "TraceLens" / "added.py", "d = 4\n")
    _write(tmp_path / "TraceLens" / "tests" / "test_kept.py", "e = 5\n")
    head = _commit(tmp_path, "head")
    monkeypatch.chdir(tmp_path)
    assert metrics.changed_modules(base, head) == {
        "TraceLens/added.py": None,
        "TraceLens/kept.py": "TraceLens/kept.py",
        "TraceLens/new.py": "TraceLens/old.py",
    }
    assert metrics.added_lines(base, head)["TraceLens/kept.py"] == {3}
