###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for pull-request test selection."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "select_related_tests",
    ROOT / "scripts" / "select_related_tests.py",
)
selector = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(selector)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_parse_name_status_includes_rename_endpoints():
    text = "\n".join(
        [
            "M\tTraceLens/util.py",
            "R100\tTraceLens/old.py\tTraceLens/new.py",
            "A\ttests/test_new.py",
        ]
    )
    assert selector.parse_name_status(text) == [
        "TraceLens/util.py",
        "TraceLens/old.py",
        "TraceLens/new.py",
        "tests/test_new.py",
    ]


def test_leaf_change_does_not_select_unrelated_tests(tmp_path):
    _write(tmp_path / "TraceLens" / "__init__.py", '"""package"""\n')
    _write(tmp_path / "TraceLens" / "leaf.py", "def leaf_fn():\n    return 1\n")
    _write(tmp_path / "TraceLens" / "other.py", "def other_fn():\n    return 2\n")
    _write(
        tmp_path / "tests" / "test_leaf.py",
        "from TraceLens.leaf import leaf_fn\n\ndef test_leaf():\n    assert leaf_fn() == 1\n",
    )
    _write(
        tmp_path / "tests" / "test_other.py",
        "from TraceLens.other import other_fn\n\ndef test_other():\n    assert other_fn() == 2\n",
    )

    selection = selector.select_related_tests(tmp_path, ["TraceLens/leaf.py"])

    assert selection.mode == "selected"
    assert selection.test_files == ("tests/test_leaf.py",)


def test_package_init_change_selects_submodule_tests(tmp_path):
    _write(tmp_path / "TraceLens" / "sub" / "__init__.py", "from .worker import run\n")
    _write(tmp_path / "TraceLens" / "sub" / "worker.py", "def run():\n    return 1\n")
    _write(
        tmp_path / "tests" / "test_worker.py",
        "from TraceLens.sub.worker import run\n\ndef test_run():\n    assert run() == 1\n",
    )

    selection = selector.select_related_tests(tmp_path, ["TraceLens/sub/__init__.py"])

    assert selection.mode == "selected"
    assert selection.test_files == ("tests/test_worker.py",)


def test_module_imported_by_package_root_runs_full_suite(tmp_path):
    _write(tmp_path / "TraceLens" / "__init__.py", "from .util import helper\n")
    _write(tmp_path / "TraceLens" / "util.py", "def helper():\n    return 1\n")
    _write(tmp_path / "TraceLens" / "other.py", "def other():\n    return 2\n")
    _write(
        tmp_path / "tests" / "test_util.py",
        "from TraceLens.util import helper\n\ndef test_helper():\n    assert helper() == 1\n",
    )
    _write(
        tmp_path / "tests" / "test_other.py",
        "from TraceLens.other import other\n\ndef test_other():\n    assert other() == 2\n",
    )

    selection = selector.select_related_tests(tmp_path, ["TraceLens/util.py"])

    assert selection.mode == "all"
    assert selection.test_files == ()


def test_data_path_string_selects_the_test_that_names_it(tmp_path):
    _write(tmp_path / "TraceLens" / "__init__.py", '"""package"""\n')
    _write(
        tmp_path / "tests" / "test_reads_trace.py",
        'ROOT = "tests/traces/inference"\n\ndef test_reads():\n    assert ROOT\n',
    )
    _write(tmp_path / "tests" / "traces" / "inference" / "case.json", "{}\n")

    selection = selector.select_related_tests(
        tmp_path, ["tests/traces/inference/case.json"]
    )

    assert selection.mode == "selected"
    assert selection.test_files == ("tests/test_reads_trace.py",)


def test_conftest_and_metadata_run_full_suite(tmp_path):
    assert selector.select_related_tests(tmp_path, ["tests/conftest.py"]).mode == "all"
    assert selector.select_related_tests(tmp_path, ["setup.py"]).mode == "all"


def test_kernel_source_cli_change_stays_narrow():
    selection = selector.select_related_tests(
        ROOT, ["TraceLens/TraceUtils/kernel_source/cli.py"]
    )
    assert selection.mode == "selected"
    assert "tests/test_kernel_source_cli.py" in selection.test_files
    assert "tests/test_copyright_headers.py" in selection.test_files
    assert "tests/test_jax_perf_report.py" not in selection.test_files


def test_inference_trace_selects_inference_tests():
    selection = selector.select_related_tests(
        ROOT,
        ["tests/traces/inference/sglang_decode/capture_traces/execution_details.json"],
    )
    assert selection.mode == "selected"
    assert "tests/test_inference_perf_report.py" in selection.test_files
    assert "tests/test_kernel_source_cli.py" not in selection.test_files


def test_main_prints_mode(capsys, tmp_path):
    listing = tmp_path / "selected.txt"
    code = selector.main(
        [
            "--repo-root",
            str(ROOT),
            "--changed-file",
            "tests/test_kernel_source_cli.py",
            "--write-list",
            str(listing),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.splitlines()[0] == "mode=selected"
    assert "tests/test_kernel_source_cli.py" in captured.out
    assert "tests/test_kernel_source_cli.py" in listing.read_text(encoding="utf-8")
