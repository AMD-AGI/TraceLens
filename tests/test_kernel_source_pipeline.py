###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""End-to-end test suite for the kernel-source mapping pipeline.

This is a single, self-contained file that exercises **every stage** of the
``TraceLens.TraceUtils.kernel_source`` package so anyone can validate the whole
approach in one run:

* Stage 1 -- symbol demangling (:mod:`.demangle`)
* Stage 2 -- the patchability gate (:mod:`.patchability`)
* Stage 3 -- the editability filter (:mod:`.editable`)
* Stage 4 -- the ``__global__`` source index: scan + cache (:mod:`.index`)
* Stage 5 -- native resolution (:mod:`.resolver`)
* Stage 6 -- Triton ``.py`` resolution (:mod:`.triton_pin`)
* Stage 7 -- discovery of installed framework trees (:mod:`.discovery`)
* Stage 8 -- the on-disk audit contract (:mod:`.contract`)
* Stage 9 -- a full trace-to-source walk that ties the stages together

The tests build a throwaway fake framework tree on ``tmp_path`` and never touch
a real install, so they run anywhere (no GPU, no vLLM/aiter/SGLang needed). The
one exception is the demangler, which opportunistically uses ``c++filt`` /
``itanium-demangler`` when present but always has a pure-Python fallback.

Run just this file::

    pytest tests/test_kernel_source_pipeline.py -v
"""

from __future__ import annotations

import importlib
import json
import os
import sys

import pytest


def _load_kernel_source():
    """Import the ``kernel_source`` package, robust to a heavy parent package.

    Prefer the public path ``TraceLens.TraceUtils.kernel_source``. That import
    runs the top-level ``TraceLens/__init__``, which pulls heavy deps
    (pandas/tqdm/torch) the package under test does not need. So when it is
    unavailable, fall back to importing the self-contained ``kernel_source``
    package standalone (it uses only relative imports + the stdlib), which keeps
    this suite runnable in a bare ``python + pytest`` container.
    """
    try:
        return importlib.import_module("TraceLens.TraceUtils.kernel_source")
    except Exception:  # noqa: BLE001 - any failure -> try the standalone import.
        here = os.path.dirname(os.path.abspath(__file__))  # .../TraceLens/tests
        traceutils = os.path.join(os.path.dirname(here), "TraceLens", "TraceUtils")
        if traceutils not in sys.path:
            sys.path.insert(0, traceutils)
        try:
            return importlib.import_module("kernel_source")
        except Exception:  # noqa: BLE001 - genuinely unavailable -> skip module.
            return None


_ks = _load_kernel_source()
if _ks is None:
    pytest.skip(
        "kernel_source package is not importable in this environment",
        allow_module_level=True,
    )

# Bind the public API + submodules from whichever package object we loaded, so
# both import styles above expose the same names to the tests below.
classify_patchability = _ks.classify_patchability
is_editable_source = _ks.is_editable_source
resolve = _ks.resolve
resolve_source_path = _ks.resolve_source_path
resolve_triton_source = _ks.resolve_triton_source
triton_def_line = _ks.triton_def_line
contract = importlib.import_module(_ks.__name__ + ".contract")
discovery = importlib.import_module(_ks.__name__ + ".discovery")
index_mod = importlib.import_module(_ks.__name__ + ".index")
base_symbol = importlib.import_module(_ks.__name__ + ".demangle").base_symbol


# ---------------------------------------------------------------------------
# Fixtures: a throwaway "installed framework" tree + isolated index cache.
# ---------------------------------------------------------------------------
# A native translation unit with a mix the scanner must get right: a real
# definition, a forward declaration (must be ignored), a commented-out kernel
# (must be ignored), and a def wrapped in a __launch_bounds__ attribute (the
# attribute must not be mistaken for the kernel name).
_KERNELS_CU = """\
#include <hip/hip_runtime.h>

// Forward declaration only -- must NOT be indexed as a definition.
__global__ void forward_only_kernel(float* a, int n);

/*
 * A commented-out kernel -- must NOT be indexed.
 * __global__ void commented_out_kernel(int x) { }
 */

__global__ void paged_attention_kernel(const float* q, const float* k, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) { /* ... */ }
}

__global__ void __launch_bounds__(256, 4)
reshape_and_cache_kernel(float* out, const float* in) {
    int i = threadIdx.x;
    out[i] = in[i];
}
"""


def _line_of(text: str, needle: str) -> int:
    """1-based line number of the first line containing ``needle``."""
    for i, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return i
    raise AssertionError(f"{needle!r} not found in text")


@pytest.fixture
def framework_tree(tmp_path):
    """A fake installed tree with one native TU; returns (root, expected lines)."""
    root = tmp_path / "vllm" / "csrc"
    root.mkdir(parents=True)
    cu = root / "attention.cu"
    cu.write_text(_KERNELS_CU, encoding="utf-8")
    return {
        "root": root,
        "cu": cu,
        "paged_line": _line_of(_KERNELS_CU, "paged_attention_kernel(const float*"),
        "reshape_line": _line_of(_KERNELS_CU, "reshape_and_cache_kernel(float*"),
    }


@pytest.fixture(autouse=True)
def _isolated_index_cache(tmp_path, monkeypatch):
    """Point the on-disk index cache at tmp and clear in-process singletons.

    Autouse so no test can read or write a real user cache, and every test sees
    a clean index state.
    """
    monkeypatch.setenv("TRACELENS_KSI_CACHE_DIR", str(tmp_path / "ksi_cache"))
    monkeypatch.delenv("TRACELENS_TARGET_ARCH", raising=False)
    index_mod.reset_index_cache()
    yield
    index_mod.reset_index_cache()


# ===========================================================================
# Stage 1 -- demangling
# ===========================================================================
class TestDemangle:
    def test_plain_name_passthrough(self):
        assert base_symbol("paged_attention_kernel") == "paged_attention_kernel"

    def test_empty_is_empty(self):
        assert base_symbol("") == ""
        assert base_symbol("   ") == ""

    def test_cpp_signature_strips_return_ns_template_args(self):
        sig = "void ns::sub::my_kernel<float>(float*, int)"
        assert base_symbol(sig) == "my_kernel"

    def test_anonymous_namespace_does_not_collapse_to_void(self):
        assert base_symbol("(anonymous namespace)::softmax_kernel(int)") == (
            "softmax_kernel"
        )

    def test_itanium_mangled_name(self):
        # Decodes via itanium-demangler / c++filt when present, and via the
        # pure-Python length-prefix fallback otherwise -- all paths agree here.
        assert base_symbol("_ZN2ns6kernelEPf") == "kernel"

    def test_mangled_prefers_kernel_token(self):
        assert base_symbol("_Z18my_fused_op_kernelPfS_i") == "my_fused_op_kernel"


# ===========================================================================
# Stage 2 -- patchability gate (filesystem-free)
# ===========================================================================
class TestPatchabilityGate:
    def test_tensile_precompiled(self):
        v = classify_patchability("Cijk_Alik_Bljk_HHS_BH_MT128x128")
        assert v.patchable is False
        assert v.kind == "tensile_precompiled"

    def test_miopen_by_op_name(self):
        v = classify_patchability("some_conv_kernel", op_name="aten::miopen_convolution")
        assert v.patchable is False
        assert v.kind == "miopen_precompiled"

    def test_inductor_by_name(self):
        v = classify_patchability("triton_poi_fused_add_0")
        assert v.patchable is False
        assert v.kind == "triton_inductor_generated"

    def test_inductor_by_call_stack(self):
        v = classify_patchability(
            "some_kernel",
            call_stack=["/root/.cache/torchinductor_x/abc.py:10"],
        )
        assert v.patchable is False
        assert v.kind == "triton_inductor_generated"

    def test_ck_demangled_namespace(self):
        v = classify_patchability("ck_tile::fmha_fwd_kernel<...>")
        assert v.patchable is False
        assert v.kind == "aiter_ck"

    def test_ck_mangled_namespace(self):
        # No c++filt needed: the mangled ck namespace prefix is detected directly.
        v = classify_patchability("_ZN2ck15some_gemm_kernelIEEvPf")
        assert v.patchable is False
        assert v.kind == "aiter_ck"

    def test_unknown_kernel_defers_to_resolver(self):
        v = classify_patchability("paged_attention_kernel")
        # None == "don't know, go look"; never a positive True from the gate.
        assert v.patchable is None

    def test_lookalike_not_misclassified_as_ck(self):
        # A name that merely ends in "ck" (e.g. flashck::) must not be CK.
        v = classify_patchability("flashck::attention_kernel")
        assert v.patchable is None


# ===========================================================================
# Stage 3 -- editability filter (pure path shape)
# ===========================================================================
class TestEditable:
    @pytest.mark.parametrize(
        "path",
        ["/pkg/csrc/a.cu", "/pkg/b.cuh", "/pkg/c.hip", "/pkg/d.h", "/pkg/e.hpp"],
    )
    def test_native_extensions_editable(self, path):
        assert is_editable_source(path) is True

    def test_repo_python_editable(self):
        assert is_editable_source("/workspace/vllm/moe.py") is True

    @pytest.mark.parametrize(
        "path",
        [
            "/tmp/torchinductor_u/xx.py",
            "/root/.cache/torchinductor/abc.py",
            "/root/.cache/vllm/torch_compile_cache/x/inductor_cache/y/z.py",
        ],
    )
    def test_generated_python_not_editable(self, path):
        assert is_editable_source(path) is False

    def test_kind_hint_rejects_generated(self):
        assert is_editable_source("/repo/a.py", "triton_inductor_generated") is False

    def test_non_source_and_empty(self):
        assert is_editable_source("/repo/readme.md") is False
        assert is_editable_source("") is False
        assert is_editable_source(None) is False


# ===========================================================================
# Stage 4 -- __global__ index: scanning + caching
# ===========================================================================
class TestIndexScanning:
    def test_indexes_real_definitions_only(self, framework_tree):
        idx = index_mod.build_index([framework_tree["root"]])
        assert "paged_attention_kernel" in idx.symbol_index
        assert "reshape_and_cache_kernel" in idx.symbol_index
        # Declarations, commented code, and attribute tokens are excluded.
        assert "forward_only_kernel" not in idx.symbol_index
        assert "commented_out_kernel" not in idx.symbol_index
        assert "__launch_bounds__" not in idx.symbol_index
        assert "launch_bounds" not in idx.symbol_index

    def test_line_numbers_are_accurate(self, framework_tree):
        idx = index_mod.build_index([framework_tree["root"]])
        rec = idx.lookup("paged_attention_kernel")[0]
        assert rec["line"] == framework_tree["paged_line"]
        assert rec["file"].endswith("attention.cu")

    def test_launch_bounds_kernel_named_correctly(self, framework_tree):
        idx = index_mod.build_index([framework_tree["root"]])
        rec = idx.lookup("reshape_and_cache_kernel")[0]
        assert rec["line"] == framework_tree["reshape_line"]


class TestIndexCaching:
    def test_in_process_singleton_reused(self, framework_tree):
        first = index_mod.load_or_build([framework_tree["root"]])
        second = index_mod.load_or_build([framework_tree["root"]])
        assert first is second  # same object -> no rebuild
        assert first.symbol_count >= 2
        assert first.file_count >= 1

    def test_fingerprint_is_stable(self, framework_tree):
        idx = index_mod.load_or_build([framework_tree["root"]])
        assert index_mod.fingerprint([framework_tree["root"]]) == idx.fingerprint

    def test_cache_invalidates_when_a_file_is_added(self, framework_tree):
        before = index_mod.load_or_build([framework_tree["root"]])
        # Add a new native file with a new kernel -> the dir signature changes.
        extra = framework_tree["root"] / "extra.cu"
        extra.write_text(
            "__global__ void newly_added_kernel(int* p) { p[0] = 0; }\n",
            encoding="utf-8",
        )
        after = index_mod.load_or_build([framework_tree["root"]])
        assert after.fingerprint != before.fingerprint
        assert "newly_added_kernel" in after.symbol_index


# ===========================================================================
# Stage 5 -- native resolution
# ===========================================================================
class TestNativeResolve:
    def test_resolve_source_path_hit(self, framework_tree):
        loc = resolve_source_path("paged_attention_kernel", [framework_tree["root"]])
        assert loc is not None
        assert loc.source_file.endswith("attention.cu")
        assert loc.line == framework_tree["paged_line"]
        assert loc.framework == "vllm"

    def test_resolve_source_path_miss(self, framework_tree):
        assert resolve_source_path("no_such_kernel_xyz", [framework_tree["root"]]) is None

    def test_resolve_hit_returns_symbol_index(self, framework_tree):
        res = resolve("paged_attention_kernel", [framework_tree["root"]])
        assert res.patchable is True
        assert res.method == "symbol_index"
        assert res.source_file.endswith("attention.cu")
        assert res.line == framework_tree["paged_line"]

    def test_resolve_gate_short_circuits_tensile(self, framework_tree):
        res = resolve("Cijk_Alik_Bljk_HHS", [framework_tree["root"]])
        assert res.patchable is False
        assert res.method == "gate_non_patchable"
        assert res.kind == "tensile_precompiled"

    def test_resolve_gate_short_circuits_ck(self, framework_tree):
        res = resolve("ck_tile::gemm_kernel<float>", [framework_tree["root"]])
        assert res.patchable is False
        assert res.kind == "aiter_ck"

    def test_resolve_miss_is_unresolved(self, framework_tree):
        res = resolve("no_such_kernel_xyz", [framework_tree["root"]])
        assert res.patchable is False
        assert res.method == "unresolved"

    def test_run_gate_false_skips_classification(self, framework_tree):
        # A CK name with the gate off is not short-circuited; it just misses the
        # index (no CK source there) and reports unresolved rather than gated.
        res = resolve(
            "ck_tile::gemm_kernel<float>", [framework_tree["root"]], run_gate=False
        )
        assert res.method == "unresolved"

    def test_prebuilt_index_object_is_used(self, framework_tree):
        idx = index_mod.build_index([framework_tree["root"]])
        loc = resolve_source_path("reshape_and_cache_kernel", [], index_obj=idx)
        assert loc is not None
        assert loc.line == framework_tree["reshape_line"]

    def test_arch_ranking_prefers_target_arch(self, tmp_path, monkeypatch):
        # Same kernel in two trees; the one whose path carries the target arch
        # tag should win the ranking.
        body = "__global__ void ranked_kernel(int* p) { p[0] = 1; }\n"
        generic = tmp_path / "generic" / "csrc"
        arch = tmp_path / "builds" / "gfx942" / "csrc"
        generic.mkdir(parents=True)
        arch.mkdir(parents=True)
        (generic / "k.cu").write_text(body, encoding="utf-8")
        (arch / "k.cu").write_text(body, encoding="utf-8")
        monkeypatch.setenv("TRACELENS_TARGET_ARCH", "gfx942")
        index_mod.reset_index_cache()
        loc = resolve_source_path("ranked_kernel", [generic, arch])
        assert loc is not None
        assert "gfx942" in loc.source_file

    def test_stale_index_symbol_verification(self, tmp_path):
        # A record whose file no longer contains the symbol must be skipped
        # (the resolver reads the file to confirm before trusting the index).
        root = tmp_path / "csrc"
        root.mkdir()
        f = root / "k.cu"
        f.write_text("__global__ void real_kernel(int* p) { p[0]=0; }\n", encoding="utf-8")
        idx = index_mod.build_index([root])
        # Rewrite the file so the indexed symbol is gone -> verification fails.
        f.write_text("// symbol removed\n", encoding="utf-8")
        assert resolve_source_path("real_kernel", [root], index_obj=idx) is None


# ===========================================================================
# Stage 6 -- Triton .py resolution
# ===========================================================================
_TRITON_PY = """\
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tl.store(out_ptr + pid, tl.load(x_ptr + pid) + tl.load(y_ptr + pid))
"""


class TestTritonResolve:
    def test_def_line_single_jit(self, tmp_path):
        py = tmp_path / "kern.py"
        py.write_text(_TRITON_PY, encoding="utf-8")
        line = triton_def_line(str(py))
        assert line == _line_of(_TRITON_PY, "def add_kernel")

    def test_def_line_by_symbol_normalization(self, tmp_path):
        py = tmp_path / "kern.py"
        py.write_text(_TRITON_PY, encoding="utf-8")
        # A decorated device symbol normalizes back to the def name.
        line = triton_def_line(str(py), symbol="add_kernel_0d1d2d3de")
        assert line == _line_of(_TRITON_PY, "def add_kernel")

    def test_def_line_exact_func_wins(self, tmp_path):
        py = tmp_path / "kern.py"
        py.write_text(_TRITON_PY, encoding="utf-8")
        assert triton_def_line(str(py), func="add_kernel") == _line_of(
            _TRITON_PY, "def add_kernel"
        )

    # Note: the AST def-line pinning of resolve_triton_source is exercised via
    # the triton_def_line tests above. It cannot be driven end-to-end here
    # because pytest's tmp dir lives under /tmp, which the editability filter
    # treats as generated (see test_resolve_triton_generated_is_non_patchable).

    def test_resolve_triton_launcher_form(self):
        res = resolve_triton_source("/workspace/repo/moe.py:120:grouped_gemm")
        assert res.patchable is True
        assert res.source_file == "/workspace/repo/moe.py"
        assert res.line == 120
        assert res.method == "trace_kernel_file"

    def test_resolve_triton_parenthesized_form(self):
        res = resolve_triton_source("/workspace/repo/moe.py(88): grouped_gemm")
        assert res.source_file == "/workspace/repo/moe.py"
        assert res.line == 88

    @pytest.mark.parametrize(
        "kf",
        [
            "/tmp/torchinductor_u/abc/xyz.py:10:triton_poi_fused",
            "/root/.cache/vllm/torch_compile_cache/h/inductor_cache/uq/c.py",
        ],
    )
    def test_resolve_triton_generated_is_non_patchable(self, kf):
        res = resolve_triton_source(kf, symbol="triton_poi_fused_1")
        assert res.patchable is False
        assert res.kind == "triton_inductor_generated"
        assert res.method == "gate_non_patchable"

    def test_resolve_triton_empty(self):
        res = resolve_triton_source("")
        assert res.patchable is False
        assert res.method == "unresolved"


# ===========================================================================
# Stage 7 -- discovery of installed framework trees
# ===========================================================================
class TestDiscovery:
    def test_discovers_pinned_framework_root(self, tmp_path, monkeypatch):
        pkg = tmp_path / "vllm"
        csrc = pkg / "csrc"
        csrc.mkdir(parents=True)
        (csrc / "op.cu").write_text(
            "__global__ void k(int* p){p[0]=0;}\n", encoding="utf-8"
        )
        monkeypatch.setenv("TRACELENS_FRAMEWORK_SOURCE_ROOTS", f"vllm={pkg}")
        monkeypatch.setenv("TRACELENS_DISCOVER_ONLY", "vllm")
        paths = discovery.discover_library_paths(("vllm",))
        assert any(str(csrc) == str(p) for p in paths)

    def test_discover_only_allowlist_excludes_others(self, tmp_path, monkeypatch):
        pkg = tmp_path / "vllm"
        csrc = pkg / "csrc"
        csrc.mkdir(parents=True)
        (csrc / "op.cu").write_text(
            "__global__ void k(int* p){p[0]=0;}\n", encoding="utf-8"
        )
        monkeypatch.setenv("TRACELENS_FRAMEWORK_SOURCE_ROOTS", f"vllm={pkg}")
        monkeypatch.setenv("TRACELENS_DISCOVER_ONLY", "sglang")
        # Allowlist names only sglang, so the vllm root must not appear.
        paths = discovery.discover_library_paths(("vllm", "sglang"))
        assert all("vllm" not in str(p) for p in paths)


# ===========================================================================
# Stage 8 -- on-disk audit contract
# ===========================================================================
class TestContract:
    def test_valid_document_round_trips(self, tmp_path):
        entries = [
            contract.make_entry(
                kernel_id="k001",
                name="paged_attention_kernel",
                gpu_pct=12.5,
                source_file="/pkg/csrc/attention.cu",
                source_line=42,
                method=contract.METHOD_SYMBOL_INDEX,
                confidence=1.0,
            ),
            contract.make_entry(
                kernel_id="k002",
                name="Cijk_gemm",
                gpu_pct=8.0,
                method=contract.METHOD_GATE_NON_PATCHABLE,
                reason="Tensile precompiled",
            ),
        ]
        doc = contract.make_document(entries, generated_by="pytest", framework="vllm")
        assert contract.validate_document(doc) == []

        out = tmp_path / contract.SOURCE_RESOLUTION_FILENAME
        out.write_text(json.dumps(doc), encoding="utf-8")
        loaded = contract.read_document(out)
        assert loaded is not None
        assert contract.validate_document(loaded) == []
        assert len(loaded["entries"]) == 2

    def test_unknown_method_flagged(self):
        entry = contract.make_entry(
            kernel_id="k1", name="x", gpu_pct=1.0, source_file="/a.cu", method="bogus"
        )
        doc = contract.make_document([entry], generated_by="pytest")
        problems = contract.validate_document(doc)
        assert any("unknown method" in p for p in problems)

    def test_source_with_non_patchable_method_flagged(self):
        entry = contract.make_entry(
            kernel_id="k1",
            name="x",
            gpu_pct=1.0,
            source_file="/should/not/be/here.cu",
            method=contract.METHOD_GATE_NON_PATCHABLE,
        )
        doc = contract.make_document([entry], generated_by="pytest")
        problems = contract.validate_document(doc)
        assert any("source_file but method is" in p for p in problems)

    def test_out_of_range_confidence_flagged(self):
        entry = contract.make_entry(
            kernel_id="k1",
            name="x",
            gpu_pct=1.0,
            source_file="/a.cu",
            method=contract.METHOD_SYMBOL_INDEX,
            confidence=1.5,
        )
        doc = contract.make_document([entry], generated_by="pytest")
        problems = contract.validate_document(doc)
        assert any("invalid confidence" in p for p in problems)

    def test_split_line_suffix(self):
        path, line, func = contract.split_line_suffix("/repo/moe.py(247): _grouped_gemm")
        assert path == "/repo/moe.py"
        assert line == 247
        assert func == "_grouped_gemm"

    def test_canonical_source_path_requires_existence_and_root(self, tmp_path):
        real = tmp_path / "csrc" / "a.cu"
        real.parent.mkdir(parents=True)
        real.write_text("__global__ void k(){}\n", encoding="utf-8")
        # Exists and under root -> canonicalized.
        assert contract.canonical_source_path(str(real), (str(tmp_path),)) != ""
        # Under root but does not exist -> rejected.
        assert contract.canonical_source_path(str(tmp_path / "csrc" / "nope.cu"), (str(tmp_path),)) == ""
        # Exists but outside the roots -> rejected.
        assert contract.canonical_source_path(str(real), ("/some/other/root",)) == ""


# ===========================================================================
# Stage 9 -- full trace-to-source walk (ties the stages together)
# ===========================================================================
class TestEndToEnd:
    def test_mixed_batch_routes_correctly(self, framework_tree):
        """One resolve() call per kernel kind, as the pipeline would issue them."""
        root = framework_tree["root"]

        # 1) A native kernel referenced by its (already-demangled) trace name.
        native = resolve("paged_attention_kernel", [root])
        assert native.patchable is True
        assert native.source_file.endswith("attention.cu")

        # 2) A native kernel referenced by a mangled symbol -> demangle -> hit.
        mangled = resolve("_Z24reshape_and_cache_kernelPfPKf", [root])
        assert mangled.patchable is True
        assert mangled.source_file.endswith("attention.cu")
        assert mangled.line == framework_tree["reshape_line"]

        # 3) A precompiled GEMM -> gated, no filesystem work.
        gemm = resolve("Cijk_Alik_Bljk_HHS_BH", [root])
        assert gemm.patchable is False
        assert gemm.method == "gate_non_patchable"

        # 4) An inductor-generated Triton kernel from a compile cache -> gated.
        triton_gen = resolve_triton_source(
            "/root/.cache/vllm/torch_compile_cache/x/inductor_cache/a/b.py",
            symbol="triton_poi_fused_1",
        )
        assert triton_gen.patchable is False

        # 5) An unknown kernel with no source in the tree -> unresolved miss.
        miss = resolve("mystery_kernel_zzz", [root])
        assert miss.patchable is False
        assert miss.method == "unresolved"
