###############################################################################
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the auto-loaded kernel-shape ``sitecustomize`` shim.

Covers ``TraceLens/TraceUtils/kernel_shape_tool/sitecustomize.py``: env-flag
gating, lazy profiler resolution, the ``torch.profiler`` start/stop wrapping and
``record_shapes`` forcing, and the pending-patch import hook. The module is
loaded under a private name (never the real ``sitecustomize``) and every global
it patches (``builtins.__import__``, ``torch.profiler`` internals) is captured
and restored so the shim cannot leak into the rest of the test session.
"""

import builtins
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_TOOL_DIR = (
    Path(__file__).parent.parent / "TraceLens" / "TraceUtils" / "kernel_shape_tool"
)
_SITE_PATH = _TOOL_DIR / "sitecustomize.py"

# The shim mutates process-global torch state; keep its tests on one worker.
pytestmark = pytest.mark.xdist_group("kernel_shape_tool")


@pytest.fixture(scope="module")
def site():
    """Load the shim under a private name, restoring all globals afterwards."""
    if str(_TOOL_DIR) not in sys.path:
        sys.path.insert(0, str(_TOOL_DIR))

    import torch.cuda.profiler as tcp
    import torch.profiler as tp
    from torch.profiler.profiler import _KinetoProfile

    saved = {
        "import": builtins.__import__,
        "p_start": tp.profile.start,
        "p_stop": tp.profile.stop,
        "k_init": _KinetoProfile.__init__,
        "c_start": getattr(tcp, "start", None),
        "c_stop": getattr(tcp, "stop", None),
        "bootstrapped": getattr(sys, "_tracelens_shape_bootstrapped", None),
        "hook_flag": getattr(sys, "_tracelens_shape_import_hook", None),
    }

    spec = importlib.util.spec_from_file_location(
        "tracelens_site_under_test", _SITE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # runs _bootstrap()

    yield module

    # Restore every global the shim may have patched.
    builtins.__import__ = saved["import"]
    tp.profile.start = saved["p_start"]
    tp.profile.stop = saved["p_stop"]
    _KinetoProfile.__init__ = saved["k_init"]
    if saved["c_start"] is not None:
        tcp.start = saved["c_start"]
    if saved["c_stop"] is not None:
        tcp.stop = saved["c_stop"]
    for obj, attr in (
        (tp.profile, "_tracelens_shape_patched"),
        (_KinetoProfile, "_tracelens_record_shapes_patched"),
        (tcp, "_tracelens_shape_patched"),
    ):
        if hasattr(obj, attr):
            try:
                delattr(obj, attr)
            except (AttributeError, TypeError):
                pass


@pytest.fixture(autouse=True)
def _clean_flag(monkeypatch):
    """Default every test to shape-discovery disabled unless it opts in."""
    monkeypatch.delenv("TRACELENS_SHAPE_DISCOVERY", raising=False)
    yield


@pytest.fixture
def hermetic_engine(monkeypatch):
    """Neuter the engine so enable() is a pure state flip (imports nothing)."""
    import kernel_shape_profiler as ksp

    monkeypatch.setattr(ksp, "_force_import_submodules", lambda _prefix: None)
    monkeypatch.setattr(ksp, "_KERNEL_ENTRY_POINTS", [])
    return ksp


# ---------------------------------------------------------------------------
# Env-flag parsing
# ---------------------------------------------------------------------------


class TestFlagParsing:
    def test_flag_on_truthy_values(self, site, monkeypatch):
        for value in ("1", "true", "TRUE", "yes", "on", " 1 "):
            monkeypatch.setenv("SOME_FLAG", value)
            assert site._flag_on("SOME_FLAG") is True

    def test_flag_on_falsy_values(self, site, monkeypatch):
        for value in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("SOME_FLAG", value)
            assert site._flag_on("SOME_FLAG") is False

    def test_flag_on_default(self, site, monkeypatch):
        monkeypatch.delenv("SOME_FLAG", raising=False)
        assert site._flag_on("SOME_FLAG", "0") is False
        assert site._flag_on("SOME_FLAG", "1") is True

    def test_shape_discovery_flag(self, site, monkeypatch):
        assert site._shape_discovery_on() is False
        monkeypatch.setenv("TRACELENS_SHAPE_DISCOVERY", "1")
        assert site._shape_discovery_on() is True


# ---------------------------------------------------------------------------
# Lazy profiler resolution and gating
# ---------------------------------------------------------------------------


class TestProfilerGating:
    def test_get_profiler_returns_engine(self, site):
        import kernel_shape_profiler as ksp

        assert site._get_profiler() is ksp

    def test_get_profiler_inserts_own_dir(self, site, monkeypatch):
        # Force a cold resolve with the tool dir absent from sys.path so the
        # lazy import re-inserts it.
        monkeypatch.setattr(site, "_profiler", None)
        tool_dir = str(_TOOL_DIR)
        monkeypatch.setattr(sys, "path", [p for p in sys.path if p != tool_dir])
        prof = site._get_profiler()
        assert hasattr(prof, "enable")
        assert tool_dir in sys.path

    def test_enable_disable_gated_on_flag(self, site, monkeypatch, hermetic_engine):
        ksp = hermetic_engine

        # Flag off -> enable is a no-op.
        site._enable_profiler()
        assert ksp.is_enabled() is False

        # Flag on -> enable/disable drive the real engine.
        monkeypatch.setenv("TRACELENS_SHAPE_DISCOVERY", "1")
        try:
            site._enable_profiler()
            assert ksp.is_enabled() is True
            site._disable_profiler()
            assert ksp.is_enabled() is False
        finally:
            if ksp.is_enabled():
                ksp.disable()

    def test_enable_swallows_exceptions(self, site, monkeypatch):
        monkeypatch.setenv("TRACELENS_SHAPE_DISCOVERY", "1")

        def boom():
            raise RuntimeError("profiler import failed")

        monkeypatch.setattr(site, "_get_profiler", boom)
        # Must not propagate.
        site._enable_profiler()
        site._disable_profiler()


# ---------------------------------------------------------------------------
# torch.profiler patching
# ---------------------------------------------------------------------------


class TestProfilerPatching:
    def test_patches_are_idempotent(self, site):
        # Two consecutive calls: the second exercises the already-patched guard.
        assert site._patch_torch_profiler_profile() is True
        assert site._patch_torch_profiler_profile() is True
        assert site._patch_kineto_record_shapes() is True
        assert site._patch_kineto_record_shapes() is True
        assert site._patch_torch_profiler_both() is True
        assert site._patch_torch_cuda_profiler() is True

    def test_record_shapes_forced_when_enabled(self, site, monkeypatch):
        from torch.profiler.profiler import _KinetoProfile

        monkeypatch.setenv("TRACELENS_SHAPE_DISCOVERY", "1")
        kp = _KinetoProfile(record_shapes=False)
        assert kp.record_shapes is True

    def test_record_shapes_not_forced_when_disabled(self, site):
        from torch.profiler.profiler import _KinetoProfile

        kp = _KinetoProfile(record_shapes=False)
        assert kp.record_shapes is False

    def test_record_shapes_opt_out(self, site, monkeypatch):
        from torch.profiler.profiler import _KinetoProfile

        monkeypatch.setenv("TRACELENS_SHAPE_DISCOVERY", "1")
        monkeypatch.setenv("TRACELENS_SHAPE_FORCE_RECORD_SHAPES", "0")
        kp = _KinetoProfile(record_shapes=False)
        assert kp.record_shapes is False

    def test_cuda_profiler_wrappers_toggle_engine(
        self, site, monkeypatch, hermetic_engine
    ):
        import torch.cuda.profiler as tcp

        ksp = hermetic_engine

        # Substitute innocuous start/stop (the real ones need a CUDA device) and
        # force a fresh patch over them so the wrappers can be exercised on CPU.
        started, stopped = [], []
        monkeypatch.setattr(tcp, "start", lambda *a, **k: started.append(1))
        monkeypatch.setattr(tcp, "stop", lambda *a, **k: stopped.append(1))
        monkeypatch.setattr(tcp, "_tracelens_shape_patched", False, raising=False)

        assert site._patch_torch_cuda_profiler() is True
        # A second call short-circuits via the already-patched guard.
        assert site._patch_torch_cuda_profiler() is True

        monkeypatch.setenv("TRACELENS_SHAPE_DISCOVERY", "1")
        try:
            tcp.start()
            assert started == [1]
            assert ksp.is_enabled() is True
            tcp.stop()
            assert stopped == [1]
            assert ksp.is_enabled() is False
        finally:
            if ksp.is_enabled():
                ksp.disable()

    def test_profiler_window_toggles_engine(self, site, monkeypatch, hermetic_engine):
        ksp = hermetic_engine

        monkeypatch.setenv("TRACELENS_SHAPE_DISCOVERY", "1")
        assert ksp.is_enabled() is False
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ):
                # start() enabled the launcher-wrapping engine...
                assert ksp.is_enabled() is True
            # ...and stop() disabled it once the last window closed.
            assert ksp.is_enabled() is False
        finally:
            if ksp.is_enabled():
                ksp.disable()


# ---------------------------------------------------------------------------
# Pending-patch registry and import hook
# ---------------------------------------------------------------------------


class TestImportHook:
    def test_try_pending_applies_and_pops(self, site, monkeypatch):
        calls = []
        monkeypatch.setitem(
            site._PENDING_PATCHES, "os", lambda: (calls.append(1), True)[1]
        )
        site._try_pending()
        assert calls == [1]
        assert "os" not in site._PENDING_PATCHES

    def test_try_pending_pops_on_exception(self, site, monkeypatch):
        def boom():
            raise RuntimeError("patch failed")

        monkeypatch.setitem(site._PENDING_PATCHES, "sys", boom)
        site._try_pending()
        assert "sys" not in site._PENDING_PATCHES

    def test_try_pending_skips_none_fn(self, site, monkeypatch):
        monkeypatch.setitem(site._PENDING_PATCHES, "os", None)
        site._try_pending()
        # A ``None`` patch fn is skipped without being popped.
        assert "os" in site._PENDING_PATCHES

    def test_install_import_hook_processes_pending(self, site, monkeypatch):
        orig_import = builtins.__import__
        monkeypatch.setattr(sys, "_tracelens_shape_import_hook", False, raising=False)
        try:
            calls = []
            monkeypatch.setitem(
                site._PENDING_PATCHES, "os", lambda: (calls.append(1), True)[1]
            )
            site._install_import_hook()
            assert builtins.__import__ is not orig_import
            # A subsequent import runs the wrapped __import__ -> _try_pending.
            builtins.__import__("math")
            assert calls == [1]
            assert "os" not in site._PENDING_PATCHES
            # A further import now short-circuits (no pending patches left).
            builtins.__import__("math")
            assert calls == [1]
            # Re-installing is a no-op while the flag is set.
            hooked = builtins.__import__
            site._install_import_hook()
            assert builtins.__import__ is hooked
        finally:
            builtins.__import__ = orig_import

    def test_import_hook_is_reentrancy_safe(self, site, monkeypatch):
        orig_import = builtins.__import__
        monkeypatch.setattr(sys, "_tracelens_shape_import_hook", False, raising=False)
        try:

            def nested_patch():
                # Importing from inside the hook must be short-circuited by the
                # reentrancy guard rather than recursing.
                builtins.__import__("math")
                return True

            monkeypatch.setitem(site._PENDING_PATCHES, "os", nested_patch)
            site._install_import_hook()
            builtins.__import__("math")
            assert "os" not in site._PENDING_PATCHES
        finally:
            builtins.__import__ = orig_import

    def test_bootstrap_installs_import_hook_when_pending(self, site, monkeypatch):
        orig_import = builtins.__import__
        monkeypatch.setattr(sys, "_tracelens_shape_bootstrapped", False, raising=False)
        monkeypatch.setattr(sys, "_tracelens_shape_import_hook", False, raising=False)
        try:
            # A pending patch whose module is not yet imported keeps the registry
            # non-empty, so _bootstrap installs the import hook.
            monkeypatch.setitem(
                site._PENDING_PATCHES, "module.not.imported.yet", lambda: True
            )
            site._bootstrap()
            assert builtins.__import__ is not orig_import
        finally:
            builtins.__import__ = orig_import

    def test_bootstrap_is_idempotent(self, site):
        # The fixture already bootstrapped; a second call must return early.
        assert getattr(sys, "_tracelens_shape_bootstrapped", False) is True
        site._bootstrap()
