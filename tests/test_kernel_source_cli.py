###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the ``TraceLens_resolve_kernel_source`` CLI entry point.

These drive ``cli.main(argv)`` directly (no subprocess) and parse the JSON it
prints, so every CLI branch -- gate (non-patchable), Triton, native resolve, and
the missing ``--kernel`` error -- is exercised without a GPU.
"""

from __future__ import annotations

import json

from TraceLens.TraceUtils.kernel_source import cli


def _run(capsys, argv):
    """Run the CLI with ``argv``; return ``(exit_code, parsed_stdout_or_None)``."""
    code = cli.main(argv)
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out) if out else None
    return code, parsed


def test_cli_missing_kernel_is_usage_error(capsys):
    # Neither --kernel nor --triton-kernel-file given -> exit code 2, no JSON.
    code = cli.main([])
    err = capsys.readouterr().err
    assert code == 2
    assert "--kernel" in err


def test_cli_gate_flags_precompiled(capsys):
    # MIOpen op name -> classify_patchability in main() marks it non-patchable.
    code, data = _run(
        capsys,
        [
            "--kernel",
            "some_conv_kernel",
            "--op-name",
            "aten::miopen_convolution",
        ],
    )
    assert code == 0
    assert data["patchable"] is False
    assert data["kind"] == "miopen_precompiled"
    assert data["method"] == "gate_non_patchable"


def test_cli_triton_generated_is_non_patchable(capsys):
    # A generated-Triton path resolves to a non-patchable verdict.
    code, data = _run(
        capsys,
        ["--triton-kernel-file", "/tmp/torchinductor_u/abc.py:10:triton_kernel"],
    )
    assert code == 0
    assert data["patchable"] is False
    assert data["source_file"] in ("", "/tmp/torchinductor_u/abc.py")


def test_cli_native_resolve_miss(capsys, tmp_path):
    # An unknown symbol against an empty search path -> a clean, non-crashing miss.
    code, data = _run(
        capsys,
        ["--kernel", "definitely_no_such_kernel_xyz", "--search-path", str(tmp_path)],
    )
    assert code == 0
    assert data["patchable"] is False
    assert "kind" in data and "method" in data
