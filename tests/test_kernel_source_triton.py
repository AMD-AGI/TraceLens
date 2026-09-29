###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for Triton ``.py`` resolution and editability.

Covers launcher-form parsing, AST-based def-line pinning, and the rejection of
generated (inductor / ``/tmp``) Triton as non-patchable.

Note: pytest's ``tmp_path`` lives under ``/tmp``, which the editability filter
treats as generated. So AST pinning is tested directly via
:func:`triton_def_line` (which does not apply the filter), while the editable
vs. generated behaviour of :func:`resolve_triton_source` is tested with crafted
paths.
"""

from TraceLens.TraceUtils.kernel_source import is_editable_source, resolve_triton_source
from TraceLens.TraceUtils.kernel_source.triton_pin import triton_def_line

_TRITON_PY = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tl.store(out_ptr + pid, tl.load(x_ptr + pid) + tl.load(y_ptr + pid))
"""


# --- is_editable_source -----------------------------------------------------
def test_editable_native_extensions():
    for path in ("/pkg/csrc/a.cu", "/pkg/b.cuh", "/pkg/c.hip", "/pkg/d.h"):
        assert is_editable_source(path) is True


def test_editable_repo_python():
    assert is_editable_source("/workspace/repo/moe.py") is True


def test_editable_extra_exts_extends_native_set():
    # A ".cc" file isn't editable by default, but callers can opt it in via
    # extra_exts (with or without a leading dot, any case).
    assert is_editable_source("/pkg/kernel.cc") is False
    assert is_editable_source("/pkg/kernel.cc", extra_exts=(".cc",)) is True
    assert is_editable_source("/pkg/kernel.CXX", extra_exts=("cxx",)) is True


def test_not_editable_generated_python():
    assert is_editable_source("/tmp/torchinductor_u/xx.py") is False
    assert is_editable_source("/root/.cache/torchinductor/abc.py") is False


def test_not_editable_vllm_compile_cache():
    # vLLM's on-disk torch.compile cache: generated Triton, not editable.
    # Marker is ``inductor_cache`` / ``torch_compile_cache`` (not ``torchinductor``).
    path = (
        "/root/.cache/vllm/torch_compile_cache/torch_aot_compile/"
        "63fd855e/inductor_cache/uq/cuqmdddxbhw4otwohvvk2viqs4kqk.py"
    )
    assert is_editable_source(path) is False
    result = resolve_triton_source(path, symbol="triton_poi_fused_1")
    assert result.patchable is False
    assert result.kind == "triton_inductor_generated"
    assert result.method == "gate_non_patchable"


def test_not_editable_non_source():
    assert is_editable_source("/repo/readme.md") is False
    assert is_editable_source("") is False
    assert is_editable_source(None) is False


# --- triton_def_line (AST) --------------------------------------------------
def test_triton_def_line_single_jit_def(tmp_path):
    py = tmp_path / "kern.py"
    py.write_text(_TRITON_PY, encoding="utf-8")
    line = triton_def_line(str(py))
    assert line is not None
    assert _TRITON_PY.splitlines()[line - 1].strip().startswith("def add_kernel")


def test_triton_def_line_matches_symbol(tmp_path):
    py = tmp_path / "kern.py"
    py.write_text(_TRITON_PY, encoding="utf-8")
    # A decorated device symbol should still normalize back to the def name.
    line = triton_def_line(str(py), symbol="add_kernel_0d1d2d3de")
    assert line is not None


# --- resolve_triton_source --------------------------------------------------
def test_resolve_triton_launcher_form_path_and_line():
    # Non-/tmp, non-existent .py: parsed to path + line, no AST refinement.
    result = resolve_triton_source("/workspace/repo/moe.py:120:grouped_gemm")
    assert result.patchable is True
    assert result.source_file == "/workspace/repo/moe.py"
    assert result.line == 120
    assert result.method == "trace_kernel_file"


def test_resolve_triton_parenthesized_launcher_form():
    result = resolve_triton_source("/workspace/repo/moe.py(88): grouped_gemm")
    assert result.source_file == "/workspace/repo/moe.py"
    assert result.line == 88


def test_resolve_triton_inductor_is_non_patchable():
    result = resolve_triton_source(
        "/tmp/torchinductor_u/abc/xyz.py:10:triton_poi_fused"
    )
    assert result.patchable is False
    assert result.kind == "triton_inductor_generated"
    assert result.method == "gate_non_patchable"


def test_resolve_triton_empty_input():
    result = resolve_triton_source("")
    assert result.patchable is False
    assert result.method == "unresolved"


# --- symbol-index fallback (exact-name Triton rescue) ------------------------
# A trace that recorded only the bare device symbol (no kernel_file, is_triton
# unset) must still recover a genuine ``@triton.jit`` / ``@gluon.jit`` def via an
# EXACT normalized-name lookup, without ever guessing a wrong file by substring.
_FALLBACK_PY = """
import triton
import triton.language as tl

@triton.jit
def _fused_real_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tl.store(x_ptr + pid, pid)

def moe_dispatch(a, b):
    # plain Python dispatch wrapper, not a @triton.jit kernel
    return _fused_real_kernel[(1,)](a, b, 128)
"""


def _fallback_fixture(tmp_path, monkeypatch):
    """Write a crafted kernels file and force it editable (tmp_path is /tmp)."""
    from TraceLens.TraceUtils.kernel_source import triton_pin
    from TraceLens.TraceUtils.kernel_source.index import reset_index_cache

    monkeypatch.setattr(triton_pin, "is_editable_source", lambda *a, **k: True)
    reset_index_cache()
    py = tmp_path / "real_kernels.py"
    py.write_text(_FALLBACK_PY, encoding="utf-8")
    return str(tmp_path), py


def test_symbol_fallback_exact_resolves_real_jit_def(tmp_path, monkeypatch):
    root, py = _fallback_fixture(tmp_path, monkeypatch)
    # Leading underscore + autotune suffix that the normalizer strips off.
    result = resolve_triton_source(
        "", symbol="_fused_real_kernel_0d1d2d", search_paths=[root], exact=True
    )
    assert result.patchable is True
    assert result.method == "triton_symbol_index"
    assert result.source_file == str(py)
    assert (
        py.read_text()
        .splitlines()[result.line - 1]
        .strip()
        .startswith("def _fused_real_kernel")
    )


def test_non_py_sentinel_launcher_falls_back_to_symbol_index(tmp_path, monkeypatch):
    # A trace may forward a non-path launcher sentinel (e.g. a vendor-fused
    # "AITER (vendor)") as kernel_file. That is not a generated ``.py`` and must
    # not be gated as inductor-generated: it drives the symbol-name fallback,
    # which recovers the genuine ``@triton.jit`` def (the qk-rope regression).
    from TraceLens.TraceUtils.kernel_source import triton_pin
    from TraceLens.TraceUtils.kernel_source.index import reset_index_cache

    # Model reality: the ".py" kernel source is editable, the sentinel is not.
    monkeypatch.setattr(
        triton_pin,
        "is_editable_source",
        lambda p, *a, **k: str(p).lower().endswith(".py"),
    )
    reset_index_cache()
    qk_rope = """
import triton
import triton.language as tl

@triton.jit
def _fused_qk_rope_reshape_and_cache_kernel(q_ptr, k_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tl.store(q_ptr + pid, tl.load(k_ptr + pid))
"""
    py = tmp_path / "fused_kv_cache.py"
    py.write_text(qk_rope, encoding="utf-8")
    result = resolve_triton_source(
        kernel_file="AITER (vendor)",
        symbol="_fused_qk_rope_reshape_and_cache_kernel",
        search_paths=[str(tmp_path)],
    )
    assert result.patchable is True
    assert result.method == "triton_symbol_index"
    assert result.kind != "triton_inductor_generated"
    assert result.source_file == str(py)


def test_symbol_fallback_exact_rejects_substring_only(tmp_path, monkeypatch):
    root, _py = _fallback_fixture(tmp_path, monkeypatch)
    # "fused_real" is a substring of the def name but not an exact normalized
    # match: exact mode must NOT guess a file (the anti-wrong-file guard).
    exact = resolve_triton_source(
        "", symbol="fused_real", search_paths=[root], exact=True
    )
    assert exact.method == "unresolved"
    # Sanity: the substring tier (exact=False) still finds it, proving the symbol
    # is a genuine substring and only the exact gate suppressed the guess.
    loose = resolve_triton_source(
        "", symbol="fused_real", search_paths=[root], exact=False
    )
    assert loose.method == "triton_symbol_index"


def test_symbol_fallback_ignores_non_jit_wrapper(tmp_path, monkeypatch):
    root, _py = _fallback_fixture(tmp_path, monkeypatch)
    # A plain (non-@triton.jit) dispatch wrapper is not in the Triton index, so
    # even an exact name match returns nothing.
    result = resolve_triton_source(
        "", symbol="moe_dispatch", search_paths=[root], exact=True
    )
    assert result.method == "unresolved"


def test_resolve_kernel_source_falls_back_to_exact_triton(tmp_path, monkeypatch):
    from TraceLens.TraceUtils.kernel_source import resolve_kernel_source

    root, py = _fallback_fixture(tmp_path, monkeypatch)
    # No kernel_file, is_triton=False: native resolve misses (no native index),
    # then the exact-name Triton fallback rescues the genuine jit kernel.
    result = resolve_kernel_source("_fused_real_kernel", search_paths=[root])
    assert result.method == "triton_symbol_index"
    assert result.source_file == str(py)


# --- launcher .py jit-def gate (native dispatcher vs. real editable Triton) --
# A native/precompiled kernel can dispatch through a ``triton``-named ``.py``
# wrapper. Such a launcher has NO ``@triton.jit`` def, so it must not be claimed
# patchable: resolution has to fall through to native classification.
_NATIVE_DISPATCHER_PY = """
import torch

def fused_moe(a, b, c):
    # native/precompiled GEMM dispatched through a triton-named wrapper
    return torch.ops.mylib.fused_moe(a, b, c)
"""

_GLUON_PY = """
import gluon
from gluon import language as gl

@gluon.jit
def gluon_add_kernel(x_ptr, n):
    pid = gl.program_id(0)
    gl.store(x_ptr + pid, pid)
"""

# A ``@gluon.jit`` file that never mentions "triton": the name-only index path
# would skip it before the pre-filter was relaxed to also admit "gluon".
_GLUON_NO_TRITON_PY = """
import gluon
from gluon import language as gl

@gluon.jit
def _gluon_only_kernel(x_ptr, n):
    pid = gl.program_id(0)
    gl.store(x_ptr + pid, pid)
"""


def _editable(monkeypatch):
    from TraceLens.TraceUtils.kernel_source import triton_pin

    monkeypatch.setattr(triton_pin, "is_editable_source", lambda *a, **k: True)


def test_native_dispatcher_launcher_is_not_patchable(tmp_path, monkeypatch):
    # resolve_triton_source directly: an editable launcher .py with no jit def
    # must report unresolved, not a bogus patchable Triton location.
    _editable(monkeypatch)
    py = tmp_path / "triton_fused_moe.py"
    py.write_text(_NATIVE_DISPATCHER_PY, encoding="utf-8")
    result = resolve_triton_source(f"{py}:5:fused_moe", symbol="fused_moe_kernel")
    assert result.patchable is False
    assert result.method == "unresolved"


def test_native_dispatcher_launcher_falls_back_to_native(tmp_path, monkeypatch):
    # End-to-end via resolve_kernel_source: the launcher yields no patchable
    # Triton def, native resolve finds no editable source either, so the outcome
    # stays non-patchable/unresolved (the regression this fix targets).
    from TraceLens.TraceUtils.kernel_source import resolve_kernel_source
    from TraceLens.TraceUtils.kernel_source.index import reset_index_cache

    _editable(monkeypatch)
    reset_index_cache()
    py = tmp_path / "triton_fused_moe.py"
    py.write_text(_NATIVE_DISPATCHER_PY, encoding="utf-8")
    result = resolve_kernel_source(
        "fused_moe_kernel",
        kernel_file=f"{py}:5:fused_moe",
        search_paths=[str(tmp_path)],
    )
    assert result.patchable is False
    assert result.method == "unresolved"


def test_triton_jit_launcher_stays_patchable(tmp_path, monkeypatch):
    # A real @triton.jit kernel in an editable launcher .py is still patchable,
    # with the def line pinned by AST.
    _editable(monkeypatch)
    py = tmp_path / "moe_kernels.py"
    py.write_text(_TRITON_PY, encoding="utf-8")
    result = resolve_triton_source(f"{py}:2:add_kernel", symbol="add_kernel")
    assert result.patchable is True
    assert result.source_file == str(py)
    assert result.method == "triton_ast"
    assert (
        py.read_text()
        .splitlines()[result.line - 1]
        .strip()
        .startswith("def add_kernel")
    )


def test_gluon_jit_launcher_stays_patchable(tmp_path, monkeypatch):
    # @gluon.jit matches the same jit-decorator detector, so a gluon kernel in an
    # editable launcher .py stays patchable and line-pinned.
    _editable(monkeypatch)
    py = tmp_path / "gluon_kernels.py"
    py.write_text(_GLUON_PY, encoding="utf-8")
    result = resolve_triton_source(
        f"{py}:5:gluon_add_kernel", symbol="gluon_add_kernel"
    )
    assert result.patchable is True
    assert result.method == "triton_ast"
    assert (
        py.read_text()
        .splitlines()[result.line - 1]
        .strip()
        .startswith("def gluon_add_kernel")
    )


def test_name_only_gluon_fallback_resolves(tmp_path, monkeypatch):
    # Name-only (empty launcher) exact fallback: a @gluon.jit file that never
    # mentions "triton" is now indexed thanks to the relaxed "gluon" pre-filter.
    from TraceLens.TraceUtils.kernel_source.index import reset_index_cache

    _editable(monkeypatch)
    reset_index_cache()
    py = tmp_path / "gluon_only.py"
    py.write_text(_GLUON_NO_TRITON_PY, encoding="utf-8")
    result = resolve_triton_source(
        "", symbol="_gluon_only_kernel", search_paths=[str(tmp_path)], exact=True
    )
    assert result.method == "triton_symbol_index"
    assert result.source_file == str(py)
