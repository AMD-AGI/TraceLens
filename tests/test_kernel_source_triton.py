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


def test_not_editable_generated_python():
    assert is_editable_source("/tmp/torchinductor_u/xx.py") is False
    assert is_editable_source("/root/.cache/torchinductor/abc.py") is False
    assert is_editable_source("/repo/a.py", "triton_inductor_generated") is False


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
