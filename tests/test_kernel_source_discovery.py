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


def test_discover_library_paths_default_has_no_filter(tmp_path, monkeypatch):
    """A framework outside ``_KNOWN`` (e.g. a future TRT-LLM) is still returned
    by the no-args default, since the default filter was removed."""
    pkg, csrc = _make_fake_framework(tmp_path, "trtllm")
    monkeypatch.setenv("TRACELENS_FRAMEWORK_SOURCE_ROOTS", f"trtllm={pkg}")
    monkeypatch.setenv("TRACELENS_DISCOVER_ONLY", "trtllm")

    assert csrc in discover_library_paths()


def test_discover_known_framework_atom(tmp_path, monkeypatch):
    """ATOM is a known serving framework, so it is located by name (and its
    native source, when present, is picked up like the other known ones)."""
    pkg, csrc = _make_fake_framework(tmp_path, "atom")
    monkeypatch.setenv("TRACELENS_FRAMEWORK_SOURCE_ROOTS", f"atom={pkg}")
    monkeypatch.setenv("TRACELENS_DISCOVER_ONLY", "atom")

    frameworks = discover_frameworks()
    assert "atom" in frameworks
    assert csrc in frameworks["atom"].csrc_roots


def test_find_csrc_locates_nested_kernel_dir(tmp_path, monkeypatch):
    """A kernel-source dir nested a few levels deep (like SGLang's ``jit_kernel/csrc``)
    is found by the bounded recursive search, not just ones at the package root."""
    pkg = tmp_path / "sglang"
    nested = pkg / "jit_kernel" / "csrc"
    nested.mkdir(parents=True)
    (nested / "kernel.cuh").write_text("__global__ void k(){}", encoding="utf-8")
    monkeypatch.setenv("TRACELENS_FRAMEWORK_SOURCE_ROOTS", f"sglang={pkg}")
    monkeypatch.setenv("TRACELENS_DISCOVER_ONLY", "sglang")

    frameworks = discover_frameworks()
    assert nested in frameworks["sglang"].csrc_roots


def test_find_csrc_prunes_test_dirs(tmp_path, monkeypatch):
    """A ``kernels``/``csrc`` dir living under a pruned dir name (``tests``, etc.) is
    ignored, so fixture/benchmark trees aren't mistaken for real kernel source."""
    pkg = tmp_path / "vllm"
    pruned = pkg / "tests" / "kernels"
    pruned.mkdir(parents=True)
    (pruned / "kernel.cu").write_text("__global__ void k(){}", encoding="utf-8")
    monkeypatch.setenv("TRACELENS_FRAMEWORK_SOURCE_ROOTS", f"vllm={pkg}")
    monkeypatch.setenv("TRACELENS_DISCOVER_ONLY", "vllm")

    frameworks = discover_frameworks()
    assert frameworks["vllm"].csrc_roots == ()


def test_find_csrc_respects_max_depth(tmp_path, monkeypatch):
    """A kernel-source dir deeper than ``_CSRC_MAX_DEPTH`` is not found."""
    pkg = tmp_path / "vllm"
    too_deep = pkg / "a" / "b" / "c" / "d" / "csrc"
    too_deep.mkdir(parents=True)
    (too_deep / "kernel.cu").write_text("__global__ void k(){}", encoding="utf-8")
    monkeypatch.setenv("TRACELENS_FRAMEWORK_SOURCE_ROOTS", f"vllm={pkg}")
    monkeypatch.setenv("TRACELENS_DISCOVER_ONLY", "vllm")

    frameworks = discover_frameworks()
    assert frameworks["vllm"].csrc_roots == ()


def test_find_csrc_parent_checked_exactly_not_recursively(tmp_path, monkeypatch):
    """``pkg_dir.parent`` is only checked for the exact conventional names -- a
    nested match under the parent (rather than the package itself) is not found."""
    repo_root = tmp_path / "repo"
    pkg = repo_root / "vllm"
    pkg.mkdir(parents=True)
    nested_under_parent = repo_root / "some_other_dir" / "csrc"
    nested_under_parent.mkdir(parents=True)
    (nested_under_parent / "kernel.cu").write_text(
        "__global__ void k(){}", encoding="utf-8"
    )
    monkeypatch.setenv("TRACELENS_FRAMEWORK_SOURCE_ROOTS", f"vllm={pkg}")
    monkeypatch.setenv("TRACELENS_DISCOVER_ONLY", "vllm")

    frameworks = discover_frameworks()
    assert frameworks["vllm"].csrc_roots == ()
