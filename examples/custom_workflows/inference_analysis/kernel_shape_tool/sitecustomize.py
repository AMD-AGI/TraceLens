"""
Auto-loaded shim that turns on kernel-shape annotation without patching the
inference server (SGLang / vLLM / any torch workload).

CPython imports a top-level module named ``sitecustomize`` automatically at
interpreter startup for every process, as long as its directory is on
``sys.path`` (i.e. on ``PYTHONPATH``). This shim uses that hook to:

  1. install the Triton / FlyDSL / tvm-ffi / aiter cpp_itfs hooks from
     ``jit_shape_tracer`` as soon as those libraries are imported, and
  2. drive ``enable()`` / ``disable()`` from the torch-profiler window, so
     shape annotation costs nothing outside a profiling run.

Activation (both required):
    export PYTHONPATH=/path/to/kernel_shape_tool:$PYTHONPATH
    export TRACELENS_SHAPE_DISCOVERY=1

Behaviour when ``TRACELENS_SHAPE_DISCOVERY`` is unset/false: this shim installs
nothing observable -- every hook short-circuits, so it is safe to leave the
directory on ``PYTHONPATH`` permanently.

torch is usually not imported yet when ``sitecustomize`` runs, so patches are
registered as *pending* and applied by a ``sys.meta_path`` finder the moment
the target module finishes importing. All hooks are idempotent (guarded by a
sentinel attribute) and crash-proof (wrapped in ``try/except``) so they can
never break the workload.
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


# ---------------------------------------------------------------------------
# Lazy handle to the co-located tracer.
#
# We intentionally do NOT import jit_shape_tracer (which imports torch) at
# sitecustomize time -- that would force a heavy torch import at interpreter
# startup and run torch's import-time side effects too early. Instead we import
# it lazily, once a hook actually needs it and torch is already loaded.
# ---------------------------------------------------------------------------
_tracer = None


def _get_tracer():
    global _tracer
    if _tracer is None:
        # Make sure this file's own directory is importable even if only the
        # parent ended up on sys.path.
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import jit_shape_tracer as _jst  # noqa: E402

        _tracer = _jst
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


# Number of torch-profiler windows currently open. Nested/overlapping windows
# must not disable the tracer until the last one closes.
_profiler_active = [0]


# ---------------------------------------------------------------------------
# Patch: install the JIT hooks.
#   The hooks themselves are inert until enable() is called, so installing them
#   eagerly (at Triton/FlyDSL import) is free and guarantees they are in place
#   before the first profiler window opens. Deferred until torch is loaded so
#   we never drag torch into a process that does not already use it.
# ---------------------------------------------------------------------------
def _install_jit_hooks() -> bool:
    if "torch" not in sys.modules:
        return False
    try:
        return bool(_get_tracer().install_hooks())
    except Exception:
        return False


def _installer(fn_name: str):
    """Pending patch for one tracer hook, so it is retried until that hook lands.

    These must go in before the first kernel is loaded: tvm-ffi modules and
    aiter's run_lib cache the functions they hand out, so a late hook would
    miss every kernel resolved before it.
    """

    def install() -> bool:
        if "torch" not in sys.modules:
            return False
        try:
            return bool(getattr(_get_tracer(), fn_name)())
        except Exception:
            return False

    return install


# ---------------------------------------------------------------------------
# Patch: torch.profiler.profile.start / stop
#   enable() runs just before the profiler starts recording so the first
#   captured launch already carries shape metadata; stop() turns it back off.
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
# Patch: _KinetoProfile.__init__  -> force record_shapes=True
#   The annotation ops only surface "Input Dims" / "Input type" in the trace
#   when the profiler was constructed with record_shapes=True. Force it on (opt
#   out with TRACELENS_SHAPE_FORCE_RECORD_SHAPES=0) so shape annotation "just
#   works" regardless of how the server configured its profiler.
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
# Patch: torch.cuda.profiler.start / stop  (legacy profiling API)
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
# Pending-patch registry + import hook.
#
# Kernels launched during CUDA-graph replay need no special handling: their
# shapes are recorded when the graph is *captured*, and capture that happens
# inside a profiler window is already covered by the flag being on.
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
            # Hand the module back its real loader; only the post-load
            # callback needed this wrapper.
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

    Python only consults ``sys.meta_path`` for modules not yet in
    ``sys.modules``, so -- unlike a ``builtins.__import__`` wrapper -- this
    costs nothing for the ``import`` statements a server executes on its hot
    paths, and adds no frames to a ``with_stack`` profiler trace. Watched:
    every pending patch target, plus ``torch``, which most patches need fully
    loaded before they can apply.
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
    # Even when the flag is off we still install the (cheap) hooks: they all
    # short-circuit via _shape_discovery_on(), and installing unconditionally
    # keeps behaviour stable if the flag is toggled between fork/exec boundaries.
    if getattr(sys, "_tracelens_shape_bootstrapped", False):
        return
    sys._tracelens_shape_bootstrapped = True
    _try_pending()
    if _PENDING_PATCHES:
        _install_finder()


_bootstrap()
