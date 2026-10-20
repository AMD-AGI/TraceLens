###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Host toolcache Pythons that cannot import ssl are removed."""

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reset_incompatible_toolcache_python.sh"


def _python(path, accepts_ssl):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\nexit {code}\n".format(code=0 if accepts_ssl else 1),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_broken_toolcache_python_is_removed_and_working_python_stays(tmp_path):
    cache = tmp_path / "Python"
    broken = cache / "3.10.22" / "x64" / "bin" / "python"
    working = cache / "3.12.3" / "x64" / "bin" / "python3"
    _python(broken, accepts_ssl=False)
    _python(working, accepts_ssl=True)

    result = subprocess.run(
        ["sh", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "TOOLCACHE_PYTHON_ROOT": str(cache)},
    )

    assert result.returncode == 0, result.stderr
    assert not broken.parent.parent.parent.exists()
    assert working.is_file()
    assert str(cache / "3.10.22") in result.stdout


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
