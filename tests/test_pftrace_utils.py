###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.Reporting.pftrace_utils."""

import json
from pathlib import Path
from unittest import mock

import pytest

from TraceLens.Reporting.pftrace_utils import (
    acquire_traceconv,
    ensure_trace_json,
    run,
)


def test_run_success():
    run(["python3", "-c", "print('ok')"])


def test_run_failure_raises():
    with pytest.raises(RuntimeError, match="Command failed"):
        run(["python3", "-c", "import sys; sys.exit(1)"])


def test_ensure_trace_json_returns_json_path(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": []}))
    assert ensure_trace_json(str(trace)) == str(trace.resolve())


def test_ensure_trace_json_returns_json_gz_path(tmp_path):
    trace = tmp_path / "trace.json.gz"
    trace.write_bytes(b"")
    result = ensure_trace_json(str(trace))
    assert result.endswith(".json.gz")


def test_ensure_trace_json_unsupported_format(tmp_path):
    bad = tmp_path / "trace.bin"
    bad.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="Unsupported trace format"):
        ensure_trace_json(str(bad))


def test_acquire_traceconv_uses_preferred_path(tmp_path):
    preferred = tmp_path / "traceconv"
    preferred.write_text("#!/bin/sh\n")
    preferred.chmod(0o755)
    result = acquire_traceconv(preferred, tmp_path / "out")
    assert result == preferred.resolve()


def test_acquire_traceconv_uses_path(tmp_path):
    with mock.patch("shutil.which", return_value="/usr/bin/traceconv"):
        result = acquire_traceconv(None, tmp_path)
    assert result == Path("/usr/bin/traceconv")


def test_ensure_trace_json_pftrace_converts(tmp_path):
    pftrace = tmp_path / "capture.pftrace"
    pftrace.write_bytes(b"PFTRACE")
    fake_traceconv = tmp_path / "traceconv"
    fake_traceconv.write_text("#!/bin/sh\n")
    fake_traceconv.chmod(0o755)

    def fake_run(cmd, cwd=None):
        out_json = Path(cmd[3])
        out_json.write_text(json.dumps({"traceEvents": []}))

    with mock.patch("TraceLens.Reporting.pftrace_utils.run", side_effect=fake_run):
        result = ensure_trace_json(str(pftrace), traceconv_path=str(fake_traceconv))

    assert result == str(pftrace.with_suffix(".json"))
    assert Path(result).exists()


def test_ensure_trace_json_missing_traceconv_raises(tmp_path):
    pftrace = tmp_path / "capture.pftrace"
    pftrace.write_bytes(b"PFTRACE")
    with pytest.raises(FileNotFoundError, match="traceconv not found"):
        ensure_trace_json(str(pftrace), traceconv_path=str(tmp_path / "missing"))
