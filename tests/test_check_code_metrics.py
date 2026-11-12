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


def test_compare_fails_new_or_grown_units_and_lists_the_rest_for_cleanup():
    cyclomatic, module = "cyclomatic complexity", "module lines"
    head = {
        (cyclomatic, "m.py", "added"): 25,
        (cyclomatic, "m.py", "grown"): 34,
        (cyclomatic, "m.py", "nudged"): 33,
        (cyclomatic, "m.py", "untouched"): 30,
        (module, "m.py", metrics.MODULE): 1300,
        (module, "n.py", metrics.MODULE): 1201,
    }
    base = {
        (cyclomatic, "m.py", "grown"): 30,
        (cyclomatic, "m.py", "nudged"): 30,
        (cyclomatic, "m.py", "untouched"): 30,
        (module, "m.py", metrics.MODULE): 1250,
    }
    touched = {("m.py", "added"), ("m.py", "grown"), ("m.py", "nudged")}
    failing, cleanup = metrics.compare(head, base, touched)
    assert failing == [
        metrics.Finding(cyclomatic, "m.py", "added", None, 25),
        metrics.Finding(cyclomatic, "m.py", "grown", 30, 34),
    ]
    assert cleanup == [
        metrics.Finding(cyclomatic, "m.py", "nudged", 30, 33),
        metrics.Finding(module, "m.py", metrics.MODULE, 1250, 1300),
        metrics.Finding(module, "n.py", metrics.MODULE, None, 1201),
    ]


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


def test_changed_modules_lines_and_units(tmp_path, monkeypatch):
    body = "".join(f"value_{i} = {i}\n" for i in range(10))
    kept = "def f():\n    a = 1\n    b = 2\n\n\ndef g():\n    return 1\n\n\ndef h():\n    return 2\n"
    _write(tmp_path / "TraceLens" / "kept.py", kept)
    _write(tmp_path / "TraceLens" / "old.py", body)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    base = _commit(tmp_path, "base")
    kept = kept.replace("    b = 2\n", "").replace(
        "    return 1", "    c = 3\n    return c"
    )
    _write(tmp_path / "TraceLens" / "kept.py", kept)
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
    added, touched = metrics.changed_lines(base, head)
    assert (added["TraceLens/kept.py"], touched["TraceLens/kept.py"]) == (
        {6, 7},
        {2, 6, 7},
    )
    assert metrics.touched_units(tmp_path, ["TraceLens/kept.py"], touched) == {
        ("TraceLens/kept.py", "f"),
        ("TraceLens/kept.py", "g"),
    }
