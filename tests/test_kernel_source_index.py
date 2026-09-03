###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for native kernel indexing and resolution.

Builds a tiny ``csrc`` tree on disk and checks that the ``__global__`` scanner
finds real definitions (skipping declarations, attributes, and comments), that
the index caches by content fingerprint, and that the resolver maps a device
symbol to a verified, editable source location.
"""

import pytest

from TraceLens.TraceUtils.kernel_source import build_index, resolve, resolve_source_path
from TraceLens.TraceUtils.kernel_source.index import (
    fingerprint,
    load_or_build,
    reset_index_cache,
)

_CU_SOURCE = """
#include <cstdint>

// A forward declaration (ends in ';', no body) must NOT be indexed.
__global__ void forward_declared_kernel(float* x);

__global__ void __launch_bounds__(256) add_kernel(float* a, float* b, int n) {
    int i = threadIdx.x;
    if (i < n) a[i] += b[i];
}

namespace impl {
__global__ void scale_kernel(float* a, float s, int n) {
    int i = threadIdx.x;
    if (i < n) a[i] *= s;
}
}  // namespace impl
"""


@pytest.fixture()
def csrc_tree(tmp_path):
    """A temp csrc dir with one .cu file, with the index cache reset."""
    reset_index_cache()
    root = tmp_path / "pkg" / "csrc"
    root.mkdir(parents=True)
    (root / "ops.cu").write_text(_CU_SOURCE, encoding="utf-8")
    yield root
    reset_index_cache()


def test_index_finds_definitions_not_declarations(csrc_tree):
    index = build_index([csrc_tree])
    # add_kernel (past __launch_bounds__) and scale_kernel are definitions.
    assert "add_kernel" in index.symbol_index
    assert "scale_kernel" in index.symbol_index
    # The forward declaration (no '{' body) is not a definition.
    assert "forward_declared_kernel" not in index.symbol_index


def test_index_records_correct_line(csrc_tree):
    index = build_index([csrc_tree])
    (rec,) = index.symbol_index["add_kernel"]
    line = rec["line"]
    text = (csrc_tree / "ops.cu").read_text().splitlines()
    assert "add_kernel" in text[line - 1]


def test_resolve_source_path_maps_symbol_to_file(csrc_tree):
    loc = resolve_source_path("add_kernel", [csrc_tree])
    assert loc is not None
    assert loc.source_file.endswith("ops.cu")
    assert loc.line and loc.line > 0


def test_resolve_source_path_unknown_symbol(csrc_tree):
    assert resolve_source_path("does_not_exist_kernel", [csrc_tree]) is None


def test_resolve_runs_gate_before_lookup(csrc_tree):
    # A gated (non-patchable) kernel short-circuits without a source lookup.
    result = resolve("Cijk_Ailk_Bljk", [csrc_tree])
    assert result.patchable is False
    assert result.method == "gate_non_patchable"
    assert result.kind == "tensile_precompiled"


def test_resolve_hit_reports_symbol_index_method(csrc_tree):
    result = resolve("scale_kernel", [csrc_tree])
    assert result.patchable is True
    assert result.method == "symbol_index"
    assert result.source_file.endswith("ops.cu")


def test_resolve_miss_is_unresolved(csrc_tree):
    result = resolve("totally_absent_kernel", [csrc_tree])
    assert result.patchable is False
    assert result.method == "unresolved"


def test_fingerprint_changes_when_source_changes(csrc_tree):
    fp1 = fingerprint([csrc_tree])
    (csrc_tree / "more.cu").write_text(
        "__global__ void relu_kernel(float* a, int n) { int i = threadIdx.x; if (i<n) a[i]=a[i]>0?a[i]:0; }",
        encoding="utf-8",
    )
    fp2 = fingerprint([csrc_tree])
    assert fp1 != fp2


def test_load_or_build_uses_cache_on_second_call(csrc_tree):
    first = load_or_build([csrc_tree])
    assert first.build_ms >= 0.0

    # Second call is served from the in-process singleton: the same object.
    second = load_or_build([csrc_tree])
    assert second is first

    # After dropping the in-process singleton, the on-disk cache serves a fresh
    # object with build_ms reset to 0.0 (a cache hit, not a rebuild).
    reset_index_cache()
    third = load_or_build([csrc_tree])
    assert third is not first
    assert third.build_ms == 0.0
    assert third.symbol_index == first.symbol_index
