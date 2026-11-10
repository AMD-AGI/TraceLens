###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The requested host toolcache Python is removed before setup-python runs."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reset_incompatible_toolcache_python.sh"


def test_requested_toolcache_python_is_removed(tmp_path):
    cache = tmp_path / "Python"
    requested = cache / "3.10.22" / "x64" / "bin"
    retained = cache / "3.12.3" / "x64" / "bin"
    requested.mkdir(parents=True)
    retained.mkdir(parents=True)

    result = subprocess.run(
        ["sh", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "TOOLCACHE_PYTHON_ROOT": str(cache)},
    )

    assert result.returncode == 0, result.stderr
    assert not requested.parent.parent.exists()
    assert retained.is_dir()
    assert str(cache / "3.10.22") in result.stdout


def test_requested_version_can_be_configured(tmp_path):
    cache = tmp_path / "Python"
    requested = cache / "3.12.3" / "x64"
    retained = cache / "3.10.22" / "x64"
    requested.mkdir(parents=True)
    retained.mkdir(parents=True)

    result = subprocess.run(
        ["sh", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TOOLCACHE_PYTHON_ROOT": str(cache),
            "TOOLCACHE_PYTHON_VERSION": "3.12",
        },
    )

    assert result.returncode == 0, result.stderr
    assert not requested.exists()
    assert retained.is_dir()


def test_missing_toolcache_is_a_no_op(tmp_path):
    result = subprocess.run(
        ["sh", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TOOLCACHE_PYTHON_ROOT": str(tmp_path / "absent"),
            "RUNNER_TOOL_CACHE": str(tmp_path / "also-absent"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
