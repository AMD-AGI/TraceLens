###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the optional framework-discovery helper.

Uses the ``TRACELENS_FRAMEWORK_SOURCE_ROOTS`` / ``TRACELENS_DISCOVER_ONLY``
environment overrides to pin discovery at a temp tree, so the test is
deterministic regardless of what is installed in the environment.
"""

from TraceLens.TraceUtils.kernel_source import (
    discover_frameworks,
    discover_library_paths,
)


def _make_fake_framework(tmp_path, name="aiter"):
    """Create ``<tmp>/<name>`` with a ``csrc`` dir holding one native file."""
    pkg = tmp_path / name
    csrc = pkg / "csrc"
    csrc.mkdir(parents=True)
    (csrc / "kernel.cu").write_text("__global__ void k(){}", encoding="utf-8")
    return pkg, csrc


def test_discover_frameworks_via_env_override(tmp_path, monkeypatch):
    pkg, csrc = _make_fake_framework(tmp_path, "aiter")
    monkeypatch.setenv("TRACELENS_FRAMEWORK_SOURCE_ROOTS", f"aiter={pkg}")
    monkeypatch.setenv("TRACELENS_DISCOVER_ONLY", "aiter")

    frameworks = discover_frameworks()
    assert "aiter" in frameworks
    assert csrc in frameworks["aiter"].csrc_roots


def test_discover_library_paths_flattens_csrc(tmp_path, monkeypatch):
    pkg, csrc = _make_fake_framework(tmp_path, "aiter")
    monkeypatch.setenv("TRACELENS_FRAMEWORK_SOURCE_ROOTS", f"aiter={pkg}")
    monkeypatch.setenv("TRACELENS_DISCOVER_ONLY", "aiter")

    paths = discover_library_paths(("aiter",))
    assert csrc in paths


def test_discover_library_paths_respects_name_filter(tmp_path, monkeypatch):
    pkg, _csrc = _make_fake_framework(tmp_path, "aiter")
    monkeypatch.setenv("TRACELENS_FRAMEWORK_SOURCE_ROOTS", f"aiter={pkg}")
    monkeypatch.setenv("TRACELENS_DISCOVER_ONLY", "aiter")

    # Asking only for sglang should not return the aiter tree.
    assert discover_library_paths(("sglang",)) == []
