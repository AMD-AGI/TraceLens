"""
Auto-loaded shim that turns on kernel-shape annotation without patching the
server. CPython imports ``sitecustomize`` at startup when it is on ``sys.path``.

It installs the ``generic_shape_profiler`` hooks as their libraries load, and
drives ``enable()`` / ``disable()`` from the torch-profiler window.

Activation:
    export PYTHONPATH=/path/to/TraceLens/TraceLens/TraceUtils/generic_shape_profiler:$PYTHONPATH
    export TRACELENS_SHAPE_DISCOVERY=1

With ``TRACELENS_SHAPE_DISCOVERY`` unset every hook short-circuits, so the
directory can stay on ``PYTHONPATH``. All patches are idempotent and wrapped in
``try/except``.
"""

import os
import sys
import threading

_ENV_FLAG = "TRACELENS_SHAPE_DISCOVERY"
_FORCE_RECORD_SHAPES_ENV = "TRACELENS_SHAPE_FORCE_RECORD_SHAPES"


def _flag_on(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return val.strip().lower() not in ("", "0", "false", "no", "off")


def _shape_discovery_on() -> bool:
    return _flag_on(_ENV_FLAG, "0")


# Imported lazily: the core imports torch, which must not load at startup.
_tracer = None


def _get_tracer():
    global _tracer
    if _tracer is None:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import generic_shape_profiler as _gsp  # noqa: E402

        _tracer = _gsp
    return _tracer


def _enable_tracer() -> None:
    try:
        if _shape_discovery_on():
            _get_tracer().enable()
    except Exception:
        pass


def _disable_tracer() -> None:
    try:
        tracer = _get_tracer()
        if tracer.is_enabled():
            tracer.disable()
    except Exception:
        pass


# Open profiler windows; the tracer turns off when the last one closes.
_profiler_active = [0]


# ---------------------------------------------------------------------------
# JIT hooks. Inert until enable(), so they are installed as early as possible
# (but never before torch is loaded).
# ---------------------------------------------------------------------------
def _install_jit_hooks() -> bool:
    if "torch" not in sys.modules:
        return False
    try:
        return bool(_get_tracer().install_hooks())
    except Exception:
        return False


def _installer(fn_name: str):
    """Pending patch for one hook that must land before its first kernel loads."""

    def install() -> bool:
        if "torch" not in sys.modules:
            return False
        try:
            return bool(getattr(_get_tracer(), fn_name)())
        except Exception:
            return False

    return install


# ---------------------------------------------------------------------------
# torch.profiler.profile.start / stop drive enable() / disable().
# ---------------------------------------------------------------------------
def _patch_torch_profiler_profile() -> bool:
    try:
        import torch.profiler as tp
    except Exception:
        return False

    cls = getattr(tp, "profile", None)
    if cls is None or not isinstance(cls, type):
        return False
    if getattr(cls, "_tracelens_shape_patched", False):
        return True

    orig_start = cls.start
    orig_stop = cls.stop

    def _start(self, *a, **kw):
        if _shape_discovery_on():
            _enable_tracer()
            _profiler_active[0] += 1
            self._tracelens_incremented = True
        return orig_start(self, *a, **kw)

    def _stop(self, *a, **kw):
        try:
            return orig_stop(self, *a, **kw)
        finally:
            if getattr(self, "_tracelens_incremented", False):
                self._tracelens_incremented = False
                _profiler_active[0] = max(0, _profiler_active[0] - 1)
                if _profiler_active[0] == 0:
                    _disable_tracer()

    cls.start = _start
    cls.stop = _stop
    cls._tracelens_shape_patched = True
    return True


# ---------------------------------------------------------------------------
# Force record_shapes=True: Input Dims only surface when shapes are recorded.
# ---------------------------------------------------------------------------
def _patch_kineto_record_shapes() -> bool:
    try:
        from torch.profiler.profiler import _KinetoProfile
    except Exception:
        return False

    if getattr(_KinetoProfile, "_tracelens_record_shapes_patched", False):
        return True

    orig_init = _KinetoProfile.__init__

    def _patched_init(self, *args, **kwargs):
        if _shape_discovery_on() and _flag_on(_FORCE_RECORD_SHAPES_ENV, "1"):
            if kwargs.get("record_shapes") is False:
                sys.stderr.write(
                    "[tracelens-shape] forcing record_shapes=True "
                    "(needed for kernel shape annotation)\n"
                )
            kwargs["record_shapes"] = True
        orig_init(self, *args, **kwargs)

    _KinetoProfile.__init__ = _patched_init
    _KinetoProfile._tracelens_record_shapes_patched = True
    return True


def _patch_torch_profiler_both() -> bool:
    a = _patch_torch_profiler_profile()
    b = _patch_kineto_record_shapes()
    return a and b


# ---------------------------------------------------------------------------
# torch.cuda.profiler.start / stop (legacy API).
# ---------------------------------------------------------------------------
def _patch_torch_cuda_profiler() -> bool:
    try:
        import torch.cuda.profiler as tcp
    except Exception:
        return False
    if getattr(tcp, "_tracelens_shape_patched", False):
        return True
    if not (hasattr(tcp, "start") and hasattr(tcp, "stop")):
        return False

    orig_start = tcp.start
    orig_stop = tcp.stop

    def _start(*a, **kw):
        if _shape_discovery_on():
            _enable_tracer()
            _profiler_active[0] += 1
        return orig_start(*a, **kw)

    def _stop(*a, **kw):
        try:
            return orig_stop(*a, **kw)
        finally:
            if _profiler_active[0] > 0:
                _profiler_active[0] = max(0, _profiler_active[0] - 1)
                if _profiler_active[0] == 0:
                    _disable_tracer()

    tcp.start = _start
    tcp.stop = _stop
    tcp._tracelens_shape_patched = True
    return True


# ---------------------------------------------------------------------------
# Pending patches, applied when their module finishes importing.
# ---------------------------------------------------------------------------
_PENDING_PATCHES = {
    "torch.profiler": _patch_torch_profiler_both,
    "torch.cuda.profiler": _patch_torch_cuda_profiler,
    "triton.runtime.jit": _install_jit_hooks,
    "triton.runtime.autotuner": _install_jit_hooks,
    "flydsl.compiler.jit_function": _install_jit_hooks,
    "triton.compiler.compiler": _installer("install_triton_compiled_hooks"),
    "tvm_ffi.module": _installer("install_tvm_ffi_hooks"),
    "csrc.cpp_itfs.utils": _installer("install_aiter_cpp_itfs_hooks"),
    "aiter.jit.core": _installer("install_aiter_compile_ops_hooks"),
    "aiter.ops.flydsl.kernels.tensor_shim": _installer("install_aiter_flydsl_hooks"),
    "aiter.ops.flydsl.moe_kernels": _installer("install_aiter_flydsl_moe_hooks"),
    "tilelang.jit": _installer("install_tilelang_jit_hooks"),
    "tilelang.jit.kernel": _installer("install_tilelang_kernel_hooks"),
}


_tls = threading.local()


def _try_pending() -> None:
    if getattr(_tls, "in_hook", False):
        return
    _tls.in_hook = True
    try:
        for mod_name in list(_PENDING_PATCHES.keys()):
            if mod_name in sys.modules:
                fn = _PENDING_PATCHES.get(mod_name)
                if fn is None:
                    continue
                try:
                    if fn():
                        _PENDING_PATCHES.pop(mod_name, None)
                except Exception:
                    _PENDING_PATCHES.pop(mod_name, None)
    finally:
        _tls.in_hook = False
    if not _PENDING_PATCHES:
        _remove_finder()


class _PatchOnLoad:
    """Loader wrapper that applies pending patches once its module has run."""

    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        try:
            self._loader.exec_module(module)
        finally:
            # Restore the real loader on the module and its spec.
            if getattr(module, "__loader__", None) is self:
                module.__loader__ = self._loader
            spec = getattr(module, "__spec__", None)
            if spec is not None and spec.loader is self:
                spec.loader = self._loader
        _try_pending()

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _PostImportFinder:
    """``sys.meta_path`` finder that observes the first import of watched modules.

    Only consulted for modules not yet in ``sys.modules``, so hot-path imports
    cost nothing and add no ``with_stack`` frames.
    """

    def find_spec(self, fullname, path, target=None):
        if fullname not in _PENDING_PATCHES and fullname != "torch":
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is not None and hasattr(spec.loader, "exec_module"):
            spec.loader = _PatchOnLoad(spec.loader)
        return spec

    def invalidate_caches(self):
        pass


_finder = _PostImportFinder()


def _install_finder() -> None:
    if _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)


def _remove_finder() -> None:
    try:
        sys.meta_path.remove(_finder)
    except ValueError:
        pass


def _bootstrap() -> None:
    # Hooks are installed even with the flag off; they short-circuit on it.
    if getattr(sys, "_tracelens_shape_bootstrapped", False):
        return
    sys._tracelens_shape_bootstrapped = True
    _try_pending()
    if _PENDING_PATCHES:
        _install_finder()


_bootstrap()
