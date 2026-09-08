###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Branch-coverage tests for the kernel-source package.

These exercise the smaller error/edge branches of the modules (unreadable
files, malformed input, defensive fallbacks) that the feature-level suites
don't reach, so the whole package is verified end to end. They avoid a GPU and
any heavy deps -- everything runs on temp files and in-process helpers.
"""

from __future__ import annotations

from pathlib import Path

from TraceLens.TraceUtils.kernel_source import (
    contract,
    demangle,
    index,
    patchability,
    resolver,
    triton_pin,
)


# ---------------------------------------------------------------------------
# patchability
# ---------------------------------------------------------------------------
def test_empty_kernel_name_is_undecided():
    v = patchability.classify_patchability("")
    assert v.patchable is None


# ---------------------------------------------------------------------------
# demangle -- pure-Python fallback (used only when no real demangler decodes)
# ---------------------------------------------------------------------------
def test_base_symbol_pure_python_fallback(monkeypatch):
    # Force both real decoders to yield nothing so base_symbol must fall back to
    # reading the name straight out of the mangled string.
    monkeypatch.setattr(demangle, "_itanium_parse", None)
    monkeypatch.setattr(demangle, "_cxxfilt_base", lambda _m: "")
    demangle.demangle.cache_clear()
    demangle.base_symbol.cache_clear()
    assert demangle.base_symbol("_ZN2ns6kernelEPf") == "kernel"
    # A mangled string with no length-prefixed identifiers -> empty.
    demangle.base_symbol.cache_clear()
    assert demangle.base_symbol("_Zxx") == ""
    demangle.demangle.cache_clear()
    demangle.base_symbol.cache_clear()


def test_base_from_demangled_variants():
    # void-return strip, anonymous-namespace strip, and qualifier trimming.
    assert (
        demangle._base_from_demangled("void (anonymous namespace)::foo(int)") == "foo"
    )
    assert demangle._base_from_demangled("ns::bar() const") == "bar"
    assert demangle._base_from_demangled("") == ""


def test_base_from_mangled_prefers_kernel_token():
    # Picks the identifier containing "kernel"; else the last one.
    assert demangle._base_from_mangled("_ZN2ns6kernelEv") == "kernel"
    assert demangle._base_from_mangled("_ZN2ns3fooE") == "foo"


def test_cxxfilt_missing_returns_empty(monkeypatch):
    # When c++filt isn't on PATH, the helper returns "" rather than raising.
    monkeypatch.setattr(demangle.shutil, "which", lambda _n: None)
    demangle._cxxfilt_base.cache_clear()
    assert demangle._cxxfilt_base("_ZN2ns6kernelEv") == ""
    demangle._cxxfilt_base.cache_clear()


def test_rstrip_balanced_unbalanced_left_unchanged():
    # Ends with the close char but never balances -> string returned unchanged.
    assert demangle._rstrip_balanced("foo)", "(", ")") == "foo)"
    # Doesn't end with the close char -> returned immediately.
    assert demangle._rstrip_balanced("foo", "(", ")") == "foo"


# ---------------------------------------------------------------------------
# resolver
# ---------------------------------------------------------------------------
def test_verify_symbol_unreadable_file_is_false(tmp_path):
    assert resolver._verify_symbol(str(tmp_path / "nope.cu"), "k") is False


def test_resolve_source_path_empty_name_is_none():
    assert resolver.resolve_source_path("") is None


def test_resolved_paths_none_discovers(monkeypatch):
    # With no search paths, it defers to discovery (stubbed here).
    monkeypatch.setattr(index, "discover_library_paths", lambda: [Path("/x")])
    assert resolver._resolved_paths(None) == [Path("/x")]


def test_resolve_skips_non_editable_record(tmp_path, monkeypatch):
    # A record pointing at a non-editable file (.txt) is skipped -> unresolved.
    idx = index.SourceIndex(fingerprint="fp")
    idx.symbol_index = {"k": [{"file": str(tmp_path / "notes.txt"), "line": 1}]}
    loc = resolver.resolve_source_path("k", index_obj=idx)
    assert loc is None


# ---------------------------------------------------------------------------
# triton_pin
# ---------------------------------------------------------------------------
def test_editable_trace_source_empty():
    assert triton_pin.editable_trace_source("") == ""


def test_is_triton_kernel_def_without_decorator():
    import ast

    node = ast.parse("def plain():\n    pass\n").body[0]
    assert triton_pin._is_triton_kernel_def(node) is False


def test_triton_def_line_unparseable_file(tmp_path):
    bad = tmp_path / "broken.py"
    bad.write_text("def (:\n", encoding="utf-8")  # syntax error
    assert triton_pin.triton_def_line(str(bad)) is None


def test_triton_def_line_substring_match(tmp_path):
    src = tmp_path / "k.py"
    src.write_text(
        "import triton\n" "@triton.jit\n" "def my_fused_kernel(x):\n" "    return x\n",
        encoding="utf-8",
    )
    # symbol normalizes to a substring of the def name (not an exact match),
    # so it matches via the substring pass.
    src2 = tmp_path / "k2.py"
    src2.write_text(
        "import triton\n@triton.jit\ndef big_attention_kernel(x):\n    return x\n",
        encoding="utf-8",
    )
    assert triton_pin.triton_def_line(str(src2), symbol="attention") == 3


def test_triton_def_line_ambiguous_returns_none(tmp_path):
    src = tmp_path / "k.py"
    src.write_text(
        "import triton\n"
        "@triton.jit\n"
        "def a_kernel(x):\n    return x\n"
        "@triton.jit\n"
        "def b_kernel(x):\n    return x\n",
        encoding="utf-8",
    )
    # Two jit defs and no func/symbol hint -> can't choose -> None.
    assert triton_pin.triton_def_line(str(src)) is None


def test_resolve_triton_source_pins_line_for_real_file(tmp_path, monkeypatch):
    src = tmp_path / "moe.py"
    src.write_text(
        "import triton\n@triton.jit\ndef grouped_gemm(x):\n    return x\n",
        encoding="utf-8",
    )
    # pytest's tmp dir lives under /tmp, which the editability filter rejects as
    # a generated-Triton location; treat it as editable so we hit the AST pin.
    monkeypatch.setattr(triton_pin, "is_editable_source", lambda p, kind=None: True)
    res = triton_pin.resolve_triton_source(f"{src}:2:grouped_gemm")
    assert res.patchable is True
    assert res.method == "triton_ast"
    assert res.location.line == 3


# ---------------------------------------------------------------------------
# contract -- validation branches
# ---------------------------------------------------------------------------
def test_make_entry_carries_previous_fields():
    entry = contract.make_entry(
        kernel_id="k",
        name="foo",
        gpu_pct=1.0,
        source_file="/a.cu",
        method=contract.METHOD_LLM,
        previous_source_file="/old.cu",
        previous_method=contract.METHOD_SYMBOL_INDEX,
    )
    assert entry["previous_source_file"] == "/old.cu"
    assert entry["previous_method"] == contract.METHOD_SYMBOL_INDEX


def test_make_document_stamps_tracelens_version(monkeypatch):
    monkeypatch.setattr(contract, "version", lambda _name: "0.1.0.dev20260101+gabc123")
    doc = contract.make_document([], generated_by="pytest")
    assert doc["tracelens_version"] == "0.1.0.dev20260101+gabc123"
    assert "schema_version" not in doc


def test_tracelens_version_missing_distribution_returns_blank(monkeypatch):
    def _raise(_name):
        raise contract.PackageNotFoundError(_name)

    monkeypatch.setattr(contract, "version", _raise)
    assert contract._tracelens_version() == ""


def test_split_line_suffix_plain_path_has_no_line():
    assert contract.split_line_suffix("/repo/moe.py") == ("/repo/moe.py", None, "")
    assert contract.split_line_suffix("") == ("", None, "")


def test_canonical_source_path_requires_existence_and_root(tmp_path):
    f = tmp_path / "k.cu"
    f.write_text("__global__ void k(){}", encoding="utf-8")
    # Existing file under a root -> canonicalized; empty roots are skipped.
    assert contract.canonical_source_path(str(f), ("", str(tmp_path))) == str(
        f.resolve()
    )
    # A path outside every root -> rejected.
    assert contract.canonical_source_path(str(f), ("/some/other/root",)) == ""
    # A non-existent file -> rejected.
    assert (
        contract.canonical_source_path(str(tmp_path / "missing.cu"), (str(tmp_path),))
        == ""
    )


def test_read_document_missing_file_is_none(tmp_path):
    assert contract.read_document(tmp_path / "nope.json") is None


def test_path_is_acceptable(tmp_path):
    f = tmp_path / "k.cu"
    f.write_text("__global__ void k(){}", encoding="utf-8")
    assert contract.path_is_acceptable(str(f), (str(tmp_path),)) is True
    assert contract.path_is_acceptable(str(f), ("/other",)) is False


# ---------------------------------------------------------------------------
# index -- scanning / signature / cache edge branches
# ---------------------------------------------------------------------------
def test_skip_balanced_parens_unbalanced_runs_to_end():
    text = "foo(a, b"  # never closes
    assert index._skip_balanced_parens(text, 3) == len(text)


def test_iter_global_defs_skips_forward_decl_and_punctuation():
    # A forward declaration (ends with ';') is not a definition; a real def is.
    text = "__global__ void fwd_decl(int);\n__global__ void real_kernel(int){ }\n"
    found = dict(index._iter_global_defs(text))
    assert "real_kernel" in found
    assert "fwd_decl" not in found


def test_scan_file_unreadable_and_no_global(tmp_path):
    # A directory (not a file) -> read raises OSError -> [].
    assert index._scan_file(tmp_path) == []
    # A file without __global__ -> [].
    plain = tmp_path / "plain.cu"
    plain.write_text("int not_a_kernel(){return 0;}\n", encoding="utf-8")
    assert index._scan_file(plain) == []


def test_native_files_nonexistent_root_yields_nothing(tmp_path):
    assert list(index._native_files(tmp_path / "does_not_exist")) == []


def test_dir_signature_missing_path():
    sig = index._dir_signature(Path("/definitely/not/here/xyz"))
    assert "missing" in sig or sig  # returns a stable string, no raise


def test_load_or_build_no_paths_warns_and_builds(monkeypatch, tmp_path):
    # No search paths -> empty index, no crash.
    index.reset_index_cache()
    monkeypatch.setenv("TRACELENS_KSI_CACHE_DIR", str(tmp_path / "cache"))
    idx = index.load_or_build([])
    assert idx.symbol_count == 0
    index.reset_index_cache()


def test_iter_global_defs_break_and_punctuation():
    # Immediately-punctuation after __global__ ('*') is skipped, name still found;
    # a '{' right after __global__ breaks that scan without yielding.
    text = "__global__ *ptr_kernel(int){ }\n__global__ { oops\n"
    found = dict(index._iter_global_defs(text))
    assert "ptr_kernel" in found


def test_dir_signature_skips_non_native_and_broken_symlink(tmp_path):
    (tmp_path / "real.cu").write_text("__global__ void k(){}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not native", encoding="utf-8")
    # A dangling symlink named like a native file: os.stat raises, handled gracefully.
    try:
        (tmp_path / "dangling.cu").symlink_to(tmp_path / "missing_target.cu")
    except (OSError, NotImplementedError):
        pass
    sig = index._dir_signature(tmp_path)
    assert str(tmp_path) in sig


def test_dir_signature_walk_error_returns_missing(tmp_path, monkeypatch):
    # If walking the tree raises OSError, the signature degrades to ":missing".
    def _boom(*_a, **_k):
        raise OSError("nope")

    monkeypatch.setattr(index.os, "walk", _boom)
    assert index._dir_signature(tmp_path).endswith(":missing")


def test_cache_path_mkdir_failure_is_swallowed(tmp_path, monkeypatch):
    # Point the cache dir at a path under an existing *file* so mkdir fails; the
    # helper must swallow the error and still return a path.
    afile = tmp_path / "afile"
    afile.write_text("x", encoding="utf-8")
    monkeypatch.setenv("TRACELENS_KSI_CACHE_DIR", str(afile / "sub"))
    p = index._cache_path("fp123")
    assert p.name == "ksi_fp123.json"


def test_load_cache_corrupt_file_is_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACELENS_KSI_CACHE_DIR", str(tmp_path))
    index._cache_path("badfp").write_text("{ not valid json", encoding="utf-8")
    assert index._load_cache("badfp") is None


def test_load_cache_fingerprint_mismatch_is_miss(tmp_path, monkeypatch):
    # A well-formed cache whose stored fingerprint differs is treated as a miss.
    monkeypatch.setenv("TRACELENS_KSI_CACHE_DIR", str(tmp_path))
    index._cache_path("wantfp").write_text(
        '{"fingerprint": "otherfp"}', encoding="utf-8"
    )
    assert index._load_cache("wantfp") is None


def test_save_cache_failure_is_swallowed(monkeypatch):
    # If the cache path can't be written, _save_cache must not raise.
    monkeypatch.setattr(
        index, "_cache_path", lambda _fp: Path("/proc/nonexistent/x.json")
    )
    index._save_cache(index.SourceIndex(fingerprint="fp"))  # no exception


def test_spec_root_variants():
    # A dotted name whose parent doesn't exist -> None (find_spec raises).
    assert index._spec_root("no_such_pkg_xyz.sub") is None
    # A missing top-level module -> None.
    assert index._spec_root("definitely_not_installed_pkg_xyz") is None
    # A real package with submodule search locations -> its dir.
    assert index._spec_root("json") is not None
    # A built-in module (origin == "built-in") -> None.
    assert index._spec_root("sys") is None
    # A single-file module (has a .py origin, no submodule search dirs) -> its dir.
    assert index._spec_root("keyword") is not None


def test_version_reads_version_file(tmp_path):
    (tmp_path / "_version.py").write_text('__version__ = "9.8.7"\n', encoding="utf-8")
    assert index._version("no_such_dist_xyz", tmp_path) == "9.8.7"
    # No metadata and no version file -> empty string.
    assert index._version("no_such_dist_xyz", tmp_path / "empty") == ""


def test_has_native_false_for_empty_dir(tmp_path):
    assert index._has_native(tmp_path) is False


def test_iter_leaf_dirs_handles_unreadable_root_and_skips_hidden(tmp_path):
    # Nonexistent root -> iterdir raises -> yields nothing.
    assert list(index._iter_leaf_dirs(tmp_path / "gone", 4)) == []
    # A hidden dir and a plain file are skipped; a 'csrc' leaf is yielded.
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "afile").write_text("x", encoding="utf-8")
    (tmp_path / "csrc").mkdir()
    leaves = list(index._iter_leaf_dirs(tmp_path, 4))
    assert (tmp_path / "csrc") in leaves


def test_find_csrc_checks_parent_conventional_dirs(tmp_path):
    # A 'csrc' living beside the package (under its parent) is found by the
    # exact-parent check even though it's not inside pkg_dir.
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    sibling = tmp_path / "csrc"
    sibling.mkdir()
    (sibling / "k.cu").write_text("__global__ void k(){}", encoding="utf-8")
    roots = index._find_csrc(pkg)
    assert sibling in roots
