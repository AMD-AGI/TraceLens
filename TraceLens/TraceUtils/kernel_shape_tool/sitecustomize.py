"""Auto-loaded shim that drives kernel-shape annotation without patching the server.

CPython auto-imports ``sitecustomize`` at interpreter startup for any process
whose ``sys.path`` (``PYTHONPATH``) includes this directory. The shim wraps
``torch.profiler`` start/stop to call ``kernel_shape_profiler.enable()`` /
``disable()`` around each profiling window, so launcher wrapping costs nothing
outside a profiling run. Gated on ``TRACELENS_SHAPE_DISCOVERY``; when unset,
every hook short-circuits, so it is safe to leave on ``PYTHONPATH`` permanently.

torch is usually not imported yet when this runs, so profiler patches are
registered as *pending* and applied by an ``__import__`` hook once the target
module loads. All hooks are idempotent and wrapped in ``try/except``.
"""

import builtins
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


# Lazy handle to the co-located profiler. We do NOT import kernel_shape_profiler
# (which imports torch) at sitecustomize time -- that would force a heavy torch
# import at startup. Import it lazily once a hook needs it and torch is loaded.
_profiler = None


def _get_profiler():
    global _profiler
    if _profiler is None:
        # Ensure this file's directory is importable even if only the parent
        # ended up on sys.path.
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import kernel_shape_profiler as _ksp  # noqa: E402

        _profiler = _ksp
    return _profiler


def _enable_profiler() -> None:
    try:
        if _shape_discovery_on():
            _get_profiler().enable()
    except Exception:
        pass


def _disable_profiler() -> None:
    try:
        profiler = _get_profiler()
        if profiler.is_enabled():
            profiler.disable()
    except Exception:
        pass


# Number of torch-profiler windows currently open. Nested/overlapping windows
# must not disable the profiler until the last one closes.
_profiler_active = [0]


# Patch torch.profiler.profile.start/stop: enable() just before recording starts
# (so the first captured launch carries shapes), disable() on the last stop().
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
            _enable_profiler()
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
                    _disable_profiler()

    cls.start = _start
    cls.stop = _stop
    cls._tracelens_shape_patched = True
    return True


# Patch _KinetoProfile.__init__ to force record_shapes=True -- "Input Dims" only
# surface when shapes are recorded (opt out with TRACELENS_SHAPE_FORCE_RECORD_SHAPES=0).
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


# Patch torch.cuda.profiler.start/stop (legacy profiling API).
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
            _enable_profiler()
            _profiler_active[0] += 1
        return orig_start(*a, **kw)

    def _stop(*a, **kw):
        try:
            return orig_stop(*a, **kw)
        finally:
            if _profiler_active[0] > 0:
                _profiler_active[0] = max(0, _profiler_active[0] - 1)
                if _profiler_active[0] == 0:
                    _disable_profiler()

    tcp.start = _start
    tcp.stop = _stop
    tcp._tracelens_shape_patched = True
    return True


# Pending-patch registry + import hook. Only profiler entry points are patched
# here; launcher wrapping is done by kernel_shape_profiler.enable() on demand.
_PENDING_PATCHES = {
    "torch.profiler": _patch_torch_profiler_both,
    "torch.cuda.profiler": _patch_torch_cuda_profiler,
}


def _try_pending() -> None:
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


def _install_import_hook() -> None:
    if getattr(sys, "_tracelens_shape_import_hook", False):
        return
    sys._tracelens_shape_import_hook = True

    orig_import = builtins.__import__
    tls = threading.local()

    def _wrapped(name, globals=None, locals=None, fromlist=(), level=0):
        module = orig_import(name, globals, locals, fromlist, level)
        if not _PENDING_PATCHES:
            return module
        if getattr(tls, "in_hook", False):
            return module
        tls.in_hook = True
        try:
            _try_pending()
        finally:
            tls.in_hook = False
        return module

    builtins.__import__ = _wrapped


def _bootstrap() -> None:
    # Install the cheap profiler patches even when the flag is off: they all
    # short-circuit via _shape_discovery_on(), keeping behaviour stable if the
    # flag is toggled across fork/exec boundaries.
    if getattr(sys, "_tracelens_shape_bootstrapped", False):
        return
    sys._tracelens_shape_bootstrapped = True
    _try_pending()
    if _PENDING_PATCHES:
        _install_import_hook()


_bootstrap()
