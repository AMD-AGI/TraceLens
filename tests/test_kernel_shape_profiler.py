###############################################################################
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the no-patch kernel shape profiler.

Covers the launcher-wrapping engine in
``TraceLens/TraceUtils/kernel_shape_tool/kernel_shape_profiler.py``: schema
inference, custom-op registration and dispatch, reference patching, and the
``enable()`` / ``disable()`` lifecycle. All tensors are CPU-only so the suite
runs in CPU CI (no GPU / sglang / aiter install required).
"""

import functools
import inspect
import sys
import types
from pathlib import Path
from typing import Optional, Union

import pytest

torch = pytest.importorskip("torch")

# The tool is delivered on PYTHONPATH (see sitecustomize.py), so it is imported
# as a top-level module rather than through the TraceLens package.
_TOOL_DIR = (
    Path(__file__).parent.parent / "TraceLens" / "TraceUtils" / "kernel_shape_tool"
)
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))
import kernel_shape_profiler as _KSP  # noqa: E402

# Real submodule walker; the autouse fixture stubs the module global so
# enable() never imports real kernel packages in CI. Tests needing the real
# walker call this captured reference.
_REAL_FORCE_IMPORT = _KSP._force_import_submodules

# Serialise every test in this file onto a single xdist worker: the profiler
# keeps process-global state (a persistent Library, the ``_enabled`` flag), so
# the tests must not run concurrently in the same interpreter.
pytestmark = pytest.mark.xdist_group("kernel_shape_tool")


@pytest.fixture(scope="module")
def ksp():
    return _KSP


@pytest.fixture(autouse=True)
def _hermetic_and_disabled(ksp, monkeypatch):
    """Stub the package walker so enable() imports nothing; disable afterwards."""
    monkeypatch.setattr(ksp, "_force_import_submodules", lambda _prefix: None)
    # Discovery results are cached per process; each test discovers afresh.
    monkeypatch.setattr(ksp, "_discovered_entry_points", None)
    yield
    if ksp.is_enabled():
        ksp.disable()


def _param(annotation):
    return inspect.Parameter(
        "p", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation
    )


def _make_kernel_module(name):
    """A fake module exposing kernel-launcher-like callables to wrap."""
    mod = types.ModuleType(name)

    def my_kernel(x: torch.Tensor, weight: torch.Tensor, alpha: float = 1.0):
        return x + weight * alpha

    my_kernel.__module__ = name

    def my_norm(a, b):  # no annotations -> record_function fallback path
        return a + b

    my_norm.__module__ = name

    class MyLayer:
        def forward(self, x: torch.Tensor) -> None:  # method -> is_method path
            return x * 2

    MyLayer.__module__ = name

    mod.my_kernel = my_kernel
    mod.my_norm = my_norm
    mod.MyLayer = MyLayer
    return mod


# ---------------------------------------------------------------------------
# Schema-type inference
# ---------------------------------------------------------------------------


class TestInferSchemaType:
    def test_no_annotation_returns_none(self, ksp):
        assert ksp._infer_schema_type(_param(inspect._empty)) is None

    def test_real_types(self, ksp):
        assert ksp._infer_schema_type(_param(torch.Tensor)) == "Tensor"
        assert ksp._infer_schema_type(_param(Optional[torch.Tensor])) == "Tensor?"
        assert ksp._infer_schema_type(_param(int)) == "int"
        assert ksp._infer_schema_type(_param(float)) == "float"
        assert ksp._infer_schema_type(_param(bool)) == "bool"
        assert ksp._infer_schema_type(_param(str)) == "str"
        assert ksp._infer_schema_type(_param(torch.dtype)) == "ScalarType"

    def test_union_with_none(self, ksp):
        assert ksp._infer_schema_type(_param(Union[int, None])) == "int?"

    def test_unknown_real_type_returns_none(self, ksp):
        assert ksp._infer_schema_type(_param(list)) is None

    def test_string_annotations(self, ksp):
        assert ksp._infer_schema_type(_param("torch.Tensor")) == "Tensor"
        assert ksp._infer_schema_type(_param("Tensor")) == "Tensor"
        assert ksp._infer_schema_type(_param("Optional[torch.Tensor]")) == "Tensor?"
        assert ksp._infer_schema_type(_param("Optional[Tensor]")) == "Tensor?"
        assert ksp._infer_schema_type(_param("int")) == "int"
        assert ksp._infer_schema_type(_param("Optional[int]")) == "int?"

    def test_unknown_string_annotation_returns_none(self, ksp):
        assert ksp._infer_schema_type(_param("SomeCustomType")) is None
        assert ksp._infer_schema_type(_param("Optional[SomeCustomType]")) is None


# ---------------------------------------------------------------------------
# Schema building from a signature
# ---------------------------------------------------------------------------


class TestBuildSchemaFromSig:
    def test_tensor_and_non_tensor_params(self, ksp):
        def fn(x: torch.Tensor, weight: torch.Tensor, alpha: float):
            return None

        schema, tensor_params, non_tensor = ksp._build_schema_from_sig(
            inspect.signature(fn)
        )
        # Only tensor params appear in the op schema; non-tensor args travel
        # through the thread-local side channel instead.
        assert schema == "(Tensor x, Tensor weight) -> ()"
        assert tensor_params == ["x", "weight"]
        assert non_tensor == ["alpha"]

    def test_no_tensor_params_returns_none(self, ksp):
        def fn(a: int, b: float):
            return None

        assert ksp._build_schema_from_sig(inspect.signature(fn)) is None

    def test_var_args_returns_none(self, ksp):
        def fn(x: torch.Tensor, *args):
            return None

        assert ksp._build_schema_from_sig(inspect.signature(fn)) is None

    def test_skip_self_for_methods(self, ksp):
        def method(self, x: torch.Tensor):
            return None

        schema, tensor_params, non_tensor = ksp._build_schema_from_sig(
            inspect.signature(method), skip_self=True
        )
        assert schema == "(Tensor x) -> ()"
        assert tensor_params == ["x"]
        assert non_tensor == []

    def test_hints_type_unannotated_tensors(self, ksp):
        def fn(q, k, sinks=None, flag: bool = False):
            return None

        schema, tensor_params, non_tensor = ksp._build_schema_from_sig(
            inspect.signature(fn),
            hints={"q": "Tensor", "k": "Tensor", "sinks": "Tensor?"},
        )
        assert schema == "(Tensor q, Tensor k, Tensor? sinks) -> ()"
        assert tensor_params == ["q", "k", "sinks"]
        assert non_tensor == ["flag"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_next_op_name_sanitises_and_increments(self, ksp):
        a = ksp._next_op_name("mod.sub::fn-name")
        b = ksp._next_op_name("mod.sub::fn-name")
        assert a.startswith("mod_sub_fn_name_")
        assert a != b  # monotonic counter

    def test_non_tensor_arg_stash_roundtrip(self, ksp):
        ksp._stash_non_tensor_args("op1", {"alpha": 2.0})
        assert ksp._pop_non_tensor_args("op1") == {"alpha": 2.0}
        # A second pop returns the empty default.
        assert ksp._pop_non_tensor_args("op1") == {}

    def test_return_value_stash_roundtrip(self, ksp):
        sentinel = object()
        ksp._stash_return_value("op2", sentinel)
        assert ksp._pop_return_value("op2") is sentinel
        assert ksp._pop_return_value("op2") is None

    def test_pop_helpers_empty_in_fresh_thread(self, ksp):
        import threading

        results = {}

        def worker():
            # A thread that never stashed sees empty thread-local defaults.
            results["nt"] = ksp._pop_non_tensor_args("never")
            results["ret"] = ksp._pop_return_value("never")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert results["nt"] == {}
        assert results["ret"] is None

    def test_get_or_create_lib_is_cached(self, ksp):
        lib = ksp._get_or_create_lib()
        assert lib is ksp._get_or_create_lib()

    def test_register_op_returns_none_on_bad_schema(self, ksp):
        def fn(x: torch.Tensor):
            return x

        # A malformed schema string makes ``lib.define`` raise -> None.
        result = ksp._register_op(
            "bad_op", " -> nonsense (", fn, ["x"], [], inspect.signature(fn)
        )
        assert result is None

    def test_active_default_device_override_default_none(self, ksp):
        assert ksp._active_default_device_override() is None

    def test_active_default_device_override_reimports(self, ksp, monkeypatch):
        # Force the ``sys.modules`` miss so the helper re-imports the device mod.
        monkeypatch.delitem(sys.modules, "torch.utils._device", raising=False)
        assert ksp._active_default_device_override() is None

    def test_preserve_global_torch_state_restores_dtype(self, ksp):
        original = torch.get_default_dtype()
        try:
            with ksp._preserve_global_torch_state():
                torch.set_default_dtype(torch.float64)
                assert torch.get_default_dtype() == torch.float64
            assert torch.get_default_dtype() == original
        finally:
            torch.set_default_dtype(original)

    def test_preserve_global_torch_state_swallows_restore_errors(
        self, ksp, monkeypatch
    ):
        def boom(*args, **kwargs):
            raise RuntimeError("cannot restore")

        monkeypatch.setattr(torch, "set_default_device", boom)
        monkeypatch.setattr(torch, "set_default_dtype", boom)
        # Restore failures on exit must be swallowed, not propagated.
        with ksp._preserve_global_torch_state():
            pass


# ---------------------------------------------------------------------------
# Kernel-launcher heuristics
# ---------------------------------------------------------------------------


class TestLauncherHeuristics:
    def test_source_launches_kernel_detects_marker(self, ksp):
        def launcher(x):
            return torch.ops.aten.relu(x)  # 'torch.ops.' marker

        assert ksp._source_launches_kernel(launcher) is True

    def test_source_launches_kernel_no_marker(self, ksp):
        def plain(x):
            return x + 1

        assert ksp._source_launches_kernel(plain) is False

    def test_source_launches_kernel_no_source(self, ksp):
        # Builtins have no retrievable source.
        assert ksp._source_launches_kernel(len) is False

    def test_tensor_annotation_is_launcher(self, ksp):
        def fn(x: torch.Tensor):
            return x

        assert ksp._is_likely_kernel_launcher(fn, inspect.signature(fn)) is True

    def test_non_tensor_annotation_excluded(self, ksp):
        def fn(a: int, b: float):
            return a

        assert ksp._is_likely_kernel_launcher(fn, inspect.signature(fn)) is False

    def test_unannotated_falls_back_to_source(self, ksp):
        def fn(a, b):
            return torch.ops.aten.add(a, b)

        assert ksp._is_likely_kernel_launcher(fn, inspect.signature(fn)) is True

        def plain(a, b):
            return a + b

        assert ksp._is_likely_kernel_launcher(plain, inspect.signature(plain)) is False

    def test_partial_annotations_fall_back_to_source(self, ksp):
        # Bare tensors next to an annotated flag: the flag alone must not exclude it.
        def launcher(x, flag: bool = False):
            return torch.ops.aten.relu(x)

        assert ksp._is_likely_kernel_launcher(launcher, inspect.signature(launcher))

        def plain(x, flag: bool = False):
            return x

        assert not ksp._is_likely_kernel_launcher(plain, inspect.signature(plain))

    def test_hinted_function_is_launcher(self, ksp, monkeypatch):
        def bare(q, k, flag: bool = False):
            return q

        monkeypatch.setitem(
            ksp._TENSOR_PARAM_HINTS,
            (bare.__module__, bare.__qualname__),
            {"q": "Tensor", "k": "Tensor"},
        )
        assert ksp._is_likely_kernel_launcher(bare, inspect.signature(bare))


# ---------------------------------------------------------------------------
# Target resolution & reference patching
# ---------------------------------------------------------------------------


class TestResolveTarget:
    def test_resolves_module_function(self, ksp):
        import math

        result = ksp._resolve_target("math", "sqrt")
        assert result == (math, "sqrt", math.sqrt, False)

    def test_resolves_class_method(self, ksp, monkeypatch):
        name = "fake_resolve_mod"
        mod = _make_kernel_module(name)
        monkeypatch.setitem(sys.modules, name, mod)

        container, attr, fn, is_method = ksp._resolve_target(name, "MyLayer.forward")
        assert container is mod.MyLayer
        assert attr == "forward"
        assert fn is mod.MyLayer.forward
        assert is_method is True

    def test_missing_module_returns_none(self, ksp):
        assert ksp._resolve_target("no_such_module_xyz", "foo") is None

    def test_missing_attr_returns_none(self, ksp):
        assert ksp._resolve_target("math", "definitely_not_here") is None

    def test_missing_class_returns_none(self, ksp):
        assert ksp._resolve_target("math", "Nope.method") is None

    def test_missing_method_returns_none(self, ksp, monkeypatch):
        name = "fake_resolve_mod2"
        mod = _make_kernel_module(name)
        monkeypatch.setitem(sys.modules, name, mod)
        assert ksp._resolve_target(name, "MyLayer.no_such_method") is None


class TestPatchAllReferences:
    def test_rebinds_every_reference(self, ksp, monkeypatch):
        def original():
            return "orig"

        def wrapper():
            return "wrapped"

        mod_a = types.ModuleType("fake_ref_a")
        mod_b = types.ModuleType("fake_ref_b")
        mod_a.f = original
        mod_b.g = original
        monkeypatch.setitem(sys.modules, "fake_ref_a", mod_a)
        monkeypatch.setitem(sys.modules, "fake_ref_b", mod_b)

        index = ksp._index_module_references({id(original)})
        patches = ksp._patch_all_references(original, wrapper, index)

        assert mod_a.f is wrapper
        assert mod_b.g is wrapper
        # Restoration data is returned for later undo.
        restored = {(m, a) for (m, a, _orig) in patches}
        assert (mod_a, "f") in restored
        assert (mod_b, "g") in restored

    def test_skips_none_and_non_dict_modules(self, ksp, monkeypatch):
        def original():
            return "orig"

        def wrapper():
            return "wrapped"

        # ``None`` placeholders and non-module objects can live in sys.modules.
        monkeypatch.setitem(sys.modules, "fake_none_mod", None)
        monkeypatch.setitem(sys.modules, "fake_int_mod", 42)
        # Must scan without raising despite the odd entries.
        index = ksp._index_module_references({id(original)})
        ksp._patch_all_references(original, wrapper, index)

    def test_skips_reference_rebound_after_indexing(self, ksp, monkeypatch):
        def original():
            return "orig"

        def wrapper():
            return "wrapped"

        def other():
            return "other"

        mod = types.ModuleType("fake_ref_stale")
        mod.f = original
        monkeypatch.setitem(sys.modules, "fake_ref_stale", mod)
        index = ksp._index_module_references({id(original)})
        mod.f = other

        assert ksp._patch_all_references(original, wrapper, index) == []
        assert mod.f is other


class TestPatchDecoratorClosures:
    def test_rebinds_and_restores_wraps_closure(self, ksp, monkeypatch):
        # ``self.fn = decorator(fn)`` keeps fn only in a closure cell, never in a
        # module global, so only the closure pass can reach it.
        name, mod = "fake_closure_mod", _make_kernel_module("fake_closure_mod")
        monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(ksp, "_KERNEL_ENTRY_POINTS", [(name, "my_kernel")])
        target = mod.my_kernel

        def decorate(fn):
            @functools.wraps(fn)
            def inner(*args, **kwargs):
                return fn(*args, **kwargs)

            return inner

        held = decorate(target)
        assert held.__closure__[0].cell_contents is target

        ksp.enable()
        assert getattr(
            held.__closure__[0].cell_contents, "_kernel_shape_wrapper", False
        )
        x, w = torch.ones(2), torch.ones(2)
        assert torch.allclose(held(x, w, alpha=2.0), x + w * 2.0)

        ksp.disable()
        assert held.__closure__[0].cell_contents is target


# ---------------------------------------------------------------------------
# record_function fallback wrapper
# ---------------------------------------------------------------------------


class TestRecordFunctionWrapper:
    def test_passthrough_when_disabled(self, ksp):
        def original(x):
            return x + 1

        wrapper = ksp._make_record_function_wrapper("mod.fn", original)
        assert getattr(wrapper, "_kernel_shape_wrapper", False) is True
        # Profiler is disabled -> wrapper is a transparent passthrough.
        assert wrapper(torch.zeros(2)).tolist() == [1.0, 1.0]

    def test_emits_event_when_enabled(self, ksp, monkeypatch):
        calls = {}

        def original(x, scale):
            calls["ran"] = True
            return x

        wrapper = ksp._make_record_function_wrapper("mod.fn", original)
        monkeypatch.setattr(ksp, "_enabled", True)
        out = wrapper(torch.ones(3), scale=torch.ones(1))
        assert calls["ran"] is True
        assert torch.allclose(out, torch.ones(3))

    def test_event_without_tensor_args_when_enabled(self, ksp, monkeypatch):
        def original(flag):
            return flag

        wrapper = ksp._make_record_function_wrapper("mod.noargs", original)
        monkeypatch.setattr(ksp, "_enabled", True)
        assert wrapper(True) is True


# ---------------------------------------------------------------------------
# enable() / disable() lifecycle
# ---------------------------------------------------------------------------


class TestEnableDisable:
    def _install(self, ksp, monkeypatch, entry_points):
        name = "fake_kernel_mod"
        mod = _make_kernel_module(name)
        monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(ksp, "_KERNEL_ENTRY_POINTS", entry_points(name))
        return name, mod

    def test_wrap_and_restore(self, ksp, monkeypatch):
        name, mod = self._install(
            ksp,
            monkeypatch,
            lambda n: [(n, "my_kernel"), (n, "my_norm"), (n, "MyLayer.forward")],
        )
        orig_kernel = mod.my_kernel
        orig_norm = mod.my_norm
        orig_method = mod.MyLayer.forward

        ksp.enable()
        assert ksp.is_enabled() is True
        assert getattr(mod.my_kernel, "_kernel_shape_wrapper", False) is True
        assert getattr(mod.my_norm, "_kernel_shape_wrapper", False) is True
        assert getattr(mod.MyLayer.forward, "_kernel_shape_wrapper", False) is True

        x = torch.ones(2, 3)
        w = torch.full((2, 3), 2.0)

        # Custom-op dispatch path returns the same value as the original.
        assert torch.allclose(mod.my_kernel(x, w, alpha=1.5), x + w * 1.5)
        # record_function fallback path.
        assert torch.allclose(mod.my_norm(x, w), x + w)
        # Wrapped bound method still returns the correct result.
        assert torch.allclose(mod.MyLayer().forward(x), x * 2)

        ksp.disable()
        assert ksp.is_enabled() is False
        assert mod.my_kernel is orig_kernel
        assert mod.my_norm is orig_norm
        assert mod.MyLayer.forward is orig_method

    def test_dispatch_falls_back_on_bind_error(self, ksp, monkeypatch):
        name, mod = self._install(ksp, monkeypatch, lambda n: [(n, "my_kernel")])
        ksp.enable()
        wrapper = mod.my_kernel
        x = torch.ones(2)
        w = torch.ones(2)
        # An unbindable call routes through the original, which then raises.
        with pytest.raises(TypeError):
            wrapper(x, w, not_a_real_kwarg=1)

    def test_dispatch_falls_back_when_no_tensor_args(self, ksp, monkeypatch):
        name, mod = self._install(ksp, monkeypatch, lambda n: [(n, "my_kernel")])
        ksp.enable()
        wrapper = mod.my_kernel
        # All-None inputs cannot dispatch; the original is called instead.
        with pytest.raises(TypeError):
            wrapper(None, None)

    def test_leaked_wrapper_is_noop_after_disable(self, ksp, monkeypatch):
        name, mod = self._install(ksp, monkeypatch, lambda n: [(n, "my_kernel")])
        ksp.enable()
        leaked = mod.my_kernel  # capture the live wrapper
        ksp.disable()

        x = torch.ones(2)
        w = torch.full((2,), 3.0)
        # A reference that escaped disable() must fall straight through.
        assert torch.allclose(leaked(x, w, alpha=1.0), x + w)

    def test_wrapper_reused_across_cycles(self, ksp, monkeypatch):
        name, mod = self._install(ksp, monkeypatch, lambda n: [(n, "my_kernel")])
        ksp.enable()
        first = mod.my_kernel
        ksp.disable()
        ksp.enable()
        second = mod.my_kernel
        # Same underlying function => cached wrapper is reused.
        assert first is second

    def test_duplicate_entry_points_wrap_once(self, ksp, monkeypatch):
        name, mod = self._install(
            ksp, monkeypatch, lambda n: [(n, "my_kernel"), (n, "my_kernel")]
        )
        ksp.enable()
        assert getattr(mod.my_kernel, "_kernel_shape_wrapper", False) is True

    def test_enable_and_disable_are_idempotent(self, ksp, monkeypatch):
        name, mod = self._install(ksp, monkeypatch, lambda n: [(n, "my_kernel")])
        ksp.enable()
        ksp.enable()  # second call is a no-op
        assert ksp.is_enabled() is True
        ksp.disable()
        ksp.disable()  # second call is a no-op
        assert ksp.is_enabled() is False

    def test_unresolvable_entry_points_are_skipped(self, ksp, monkeypatch):
        monkeypatch.setattr(
            ksp,
            "_KERNEL_ENTRY_POINTS",
            [("module_that_does_not_exist", "nope")],
        )
        ksp.enable()
        assert ksp.is_enabled() is True  # enables cleanly with nothing patched

    def test_already_wrapped_entry_is_skipped(self, ksp, monkeypatch):
        name = "fake_prewrapped_mod"
        mod = types.ModuleType(name)

        def already(x: torch.Tensor):
            return x

        already._kernel_shape_wrapper = True
        already.__module__ = name
        mod.already = already
        monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(ksp, "_KERNEL_ENTRY_POINTS", [(name, "already")])

        ksp.enable()
        assert ksp.is_enabled() is True
        # A target that is already our wrapper is left untouched (no re-wrap).
        assert mod.already is already

    def test_enable_setattr_fallback_without_references(self, ksp, monkeypatch):
        name, mod = self._install(ksp, monkeypatch, lambda n: [(n, "my_kernel")])
        orig = mod.my_kernel
        # Simulate a launcher with no discoverable ``from x import y`` refs so
        # enable() falls back to a direct attribute rebind.
        monkeypatch.setattr(ksp, "_patch_all_references", lambda _o, _w, _i: [])

        ksp.enable()
        assert getattr(mod.my_kernel, "_kernel_shape_wrapper", False) is True
        ksp.disable()
        assert mod.my_kernel is orig

    def test_uninspectable_entry_is_skipped(self, ksp, monkeypatch):
        name = "fake_uninspectable_mod"
        mod = types.ModuleType(name)
        mod.weird = object()  # has no introspectable signature
        monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(ksp, "_KERNEL_ENTRY_POINTS", [(name, "weird")])

        ksp.enable()
        assert ksp.is_enabled() is True


class TestPrepare:
    def _install(self, ksp, monkeypatch):
        name = "fake_prepare_mod"
        mod = _make_kernel_module(name)
        monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(ksp, "_KERNEL_ENTRY_POINTS", [])
        calls = []

        def discover():
            calls.append(1)
            return [(name, "my_kernel")]

        monkeypatch.setattr(ksp, "_discover_kernel_entry_points", discover)
        return mod, calls

    def test_discovery_runs_once_across_windows(self, ksp, monkeypatch):
        mod, calls = self._install(ksp, monkeypatch)
        for _ in range(2):
            ksp.enable()
            assert getattr(mod.my_kernel, "_kernel_shape_wrapper", False) is True
            ksp.disable()
        assert len(calls) == 1

    def test_prepare_builds_without_patching(self, ksp, monkeypatch):
        mod, calls = self._install(ksp, monkeypatch)
        orig = mod.my_kernel

        ksp.prepare()
        assert len(calls) == 1
        assert mod.my_kernel is orig
        assert ksp.is_enabled() is False

        ksp.enable()
        # The window reuses the prepared discovery and wrapper.
        assert len(calls) == 1
        assert mod.my_kernel is ksp._built_wrappers["fake_prepare_mod.my_kernel"][0]
        ksp.disable()
        assert mod.my_kernel is orig

    def test_prepare_is_noop_while_enabled(self, ksp, monkeypatch):
        mod, calls = self._install(ksp, monkeypatch)
        ksp.enable()
        wrapper = mod.my_kernel
        ksp.prepare()
        assert len(calls) == 1
        assert mod.my_kernel is wrapper

    def test_hinted_launcher_records_shapes(self, ksp, monkeypatch):
        name = "fake_hinted_mod"
        mod = types.ModuleType(name)

        def bare_attn(q, k, causal: bool = True):
            return q + k

        bare_attn.__module__ = name
        bare_attn.__qualname__ = "bare_attn"
        mod.bare_attn = bare_attn
        monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(ksp, "_KERNEL_ENTRY_POINTS", [(name, "bare_attn")])
        monkeypatch.setitem(
            ksp._TENSOR_PARAM_HINTS, (name, "bare_attn"), {"q": "Tensor", "k": "Tensor"}
        )

        ksp.enable()
        q, k = torch.ones(4, 8), torch.ones(4, 8)
        with torch.profiler.profile(record_shapes=True) as prof:
            out = mod.bare_attn(q, k, causal=False)
        assert torch.allclose(out, q + k)
        events = [e for e in prof.events() if "fake_hinted_mod_bare_attn" in e.name]
        assert events, "hinted launcher was not registered as a custom op"
        assert events[0].input_shapes[:2] == [[4, 8], [4, 8]]


# ---------------------------------------------------------------------------
# Auto-discovery of kernel launchers under target namespaces
# ---------------------------------------------------------------------------


class TestAutoDiscovery:
    def test_force_import_submodules_missing_pkg(self, ksp):
        # Non-existent package: returns without raising.
        _REAL_FORCE_IMPORT("no_such_pkg_abcxyz")

    def test_force_import_submodules_non_package(self, ksp):
        # A plain module has no ``__path__`` to walk: returns without raising.
        _REAL_FORCE_IMPORT("math")

    def test_force_import_submodules_walks_and_skips_tests(
        self, ksp, tmp_path, monkeypatch
    ):
        pkg = tmp_path / "fakewalkpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "kernel_ops.py").write_text("def go():\n    return 1\n")
        (pkg / "test_skip.py").write_text("raise RuntimeError('must not import')\n")
        # A non-test module that fails to import must be swallowed.
        (pkg / "badmod.py").write_text("raise ImportError('boom')\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        _REAL_FORCE_IMPORT("fakewalkpkg")
        # A second pass finds every submodule already imported and skips it.
        _REAL_FORCE_IMPORT("fakewalkpkg")

        assert "fakewalkpkg.kernel_ops" in sys.modules
        # ``test_``-prefixed leaves are filtered out before import.
        assert "fakewalkpkg.test_skip" not in sys.modules
        # Import failures are swallowed, leaving the module unloaded.
        assert "fakewalkpkg.badmod" not in sys.modules

    def test_discover_finds_launchers_under_prefix(self, ksp, tmp_path, monkeypatch):
        # Build a fake ``aiter.ops`` package (matches _AUTO_DISCOVER_PREFIXES),
        # exercising every candidate-filtering branch of the discovery scan.
        root = tmp_path / "disc"
        ops = root / "aiter" / "ops"
        ops.mkdir(parents=True)
        (root / "aiter" / "__init__.py").write_text("")
        (ops / "__init__.py").write_text("")
        (ops / "mykernels.py").write_text(
            "from json import dumps\n"  # function from another module -> skipped
            "X = 5\n"  # non-function attribute -> skipped
            "def my_launch(a, b):\n"
            "    return torch.ops.aten.add(a, b)\n"  # source marker -> launcher
            "alias = my_launch\n"  # duplicate object id -> skipped
            "def noparams():\n    return 1\n"  # no params -> skipped
            "def varargs(*a):\n    return a\n"  # *args -> skipped
            "def helper(a: int):\n    return a\n"  # non-tensor annotation -> skipped
        )
        monkeypatch.syspath_prepend(str(root))
        for mod_name in ("aiter", "aiter.ops", "aiter.ops.mykernels"):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
        # Odd entries under the scanned prefix must be tolerated.
        monkeypatch.setitem(sys.modules, "aiter.ops.none_entry", None)
        monkeypatch.setitem(sys.modules, "aiter.ops.weird_entry", 42)
        # Real walker, restricted to our fake package only.
        monkeypatch.setattr(ksp, "_force_import_submodules", _REAL_FORCE_IMPORT)
        monkeypatch.setattr(ksp, "_AUTO_DISCOVER_PREFIXES", ("aiter.ops",))

        discovered = ksp._discover_kernel_entry_points()

        assert ("aiter.ops.mykernels", "my_launch") in discovered
        # The filtered-out candidates must not appear.
        names = {attr for _mod, attr in discovered}
        assert names.isdisjoint({"dumps", "noparams", "varargs", "helper", "alias"})
