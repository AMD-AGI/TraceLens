"""
Operand shapes for GPU kernels that bypass the PyTorch dispatcher.

JIT-compiled kernels (Triton, FlyDSL, TileLang, tvm-ffi, aiter ctypes/hsaco)
launch straight from Python, so profiler traces show them with no operand
metadata. Each hooked launch calls a no-op dispatcher op with one Tensor
parameter per operand, so the profiler records its shapes, strides and dtypes:

``kernel_shapes::<kernel>``    named after the kernel
``launcher_shapes::<caller>``  named after the Python function that launched it

Both carry the same work; pick one granularity when summing FLOPs.
See README.md for the hooked paths and design.

Public API: ``install_hooks()``, ``enable()`` / ``disable()``,
``mark_launch(name, tensors)``.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import re
import sys
import threading
import types
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import torch
from torch.library import Library

logger = logging.getLogger("generic_shape_profiler")

NAMESPACE = "kernel_shapes"

# Kept separate from NAMESPACE so the same work is not counted at both levels.
LAUNCHER_NAMESPACE = "launcher_shapes"

SPAN_PREFIX = "jit::"


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


_SPANS_ON = _env_on("TRACELENS_SHAPE_SPANS")
_LAUNCHER_OPS_ON = _env_on("TRACELENS_SHAPE_LAUNCHER_OPS")
# Run discovered launchers inside their op, so it encloses all their kernels.
_PROMOTE_ON = _env_on("TRACELENS_SHAPE_LAUNCHER_PROMOTE")

# Debug: breakpoint() at hooked launches ("1"/"all" or hook-id prefixes),
# optionally on one rank only.
_BREAK_SPEC = os.environ.get("TRACELENS_SHAPE_BREAK", "").strip().lower()
_BREAK_RANK = os.environ.get("TRACELENS_SHAPE_BREAK_RANK", "").strip()

# Frames in these packages are JIT machinery, never the launcher.
_JIT_MODULES = ("triton", "flydsl", "tvm_ffi", "tilelang", "tvm")

# Launch helpers inside otherwise ordinary packages.
_JIT_HELPER_MODULES = (
    "csrc.cpp_itfs.utils",
    "aiter.ops.flydsl.kernels.tensor_shim",
    "aiter.ops.flydsl.moe_kernels",
)

# aiter suffixes cpp_itfs entry points with an md5 of their template args.
_HASH_SUFFIX = re.compile(r"_[0-9a-f]{32}$")

# Too central to swap code objects safely.
_NO_PROMOTE_MODULES = _JIT_MODULES + ("torch",)

# Injected into a promoted launcher's globals for its trampoline to call.
_TRAMPOLINE_GLOBAL = "__generic_shape_profiler_call__"

_CO_NO_PROMOTE = (
    inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
)

_MAX_TENSORS = 32

_lock = threading.RLock()
_tls = threading.local()

_enabled = False
_libs: Dict[str, Library] = {}  # namespace -> Library
_ops: Dict[tuple, Any] = {}  # (namespace, name, arity) -> op callable / _BROKEN
_taken: Dict[str, Set[str]] = {}  # namespace -> op names already defined
_installed: Dict[str, str] = {}  # hook id -> what was patched
_break_hooks: Set[str] = set()  # hook ids that call breakpoint(); clear() to stop
_promoted: Dict[Any, "_Promoted"] = {}  # launcher code object -> promotion
_promoted_by_key: Dict[int, "_Promoted"] = {}
_unpromotable: Set[Any] = set()  # launcher code objects that cannot be promoted
_launches = 0
_window_start = 0  # value of _launches when the current window opened

_BROKEN = object()  # sentinel: op could not be defined/called, stop retrying


# ---------------------------------------------------------------------------
# Enable / disable: flip the flag and swap promoted launchers in and out.
# ---------------------------------------------------------------------------
def enable() -> None:
    global _enabled, _window_start
    with _lock:
        if _enabled:
            return
        install_hooks()
        _window_start = _launches
        _enabled = True
        for entry in _promoted.values():
            entry.swap_in()
    logger.info(
        "generic_shape_profiler enabled (hooks: %s)",
        ", ".join(sorted(_installed.values())) or "none available",
    )


def disable() -> None:
    global _enabled
    with _lock:
        if not _enabled:
            return
        _enabled = False
        for entry in _promoted.values():
            entry.swap_out()
        window = _launches - _window_start
    logger.info(
        "generic_shape_profiler disabled (%d launches this window, %d total, "
        "%d distinct kernels, %d distinct launchers, %d promoted)",
        window,
        _launches,
        len({n for ns, n, _ in _ops if ns == NAMESPACE}),
        len({n for ns, n, _ in _ops if ns == LAUNCHER_NAMESPACE}),
        len(_promoted),
    )


def is_enabled() -> bool:
    return _enabled


def stats() -> Dict[str, Any]:
    return {
        "enabled": _enabled,
        "hooks": dict(_installed),
        "kernels": sorted({n for ns, n, _ in _ops if ns == NAMESPACE}),
        "launchers": sorted({n for ns, n, _ in _ops if ns == LAUNCHER_NAMESPACE}),
        "promoted_launchers": sorted(e.name for e in _promoted.values()),
        "launches": _launches,
    }


# ---------------------------------------------------------------------------
# Dispatcher ops: <namespace>::<name>(Tensor a0, Tensor a1, ...) -> ()
# ---------------------------------------------------------------------------
def _noop(*args: torch.Tensor) -> None:
    return None


def _launcher_impl(*args: torch.Tensor) -> None:
    """Run the launcher call queued by _call_launcher, nesting it under the op."""
    run = getattr(_tls, "pending", None)
    if run is not None:
        _tls.pending = None
        run()


def _sanitize(name: str) -> str:
    """Valid op name, kept equal to the kernel name (leading ``_`` included)."""
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name).rstrip("_")
    if not out.strip("_"):
        return "kernel"
    if out[0].isdigit():
        out = "k_" + out
    return out[:200]


def _op_for(namespace: str, name: str, arity: int):
    key = (namespace, name, arity)
    op = _ops.get(key)
    if op is not None:
        return op

    with _lock:
        op = _ops.get(key)
        if op is not None:
            return op

        # Schemas have fixed arity: other arities get a "__<N>t" suffix.
        taken = _taken.setdefault(namespace, set())
        base = _sanitize(name)
        op_name = base if base not in taken else f"{base}__{arity}t"
        suffix = 1
        while op_name in taken:
            suffix += 1
            op_name = f"{base}__{arity}t_{suffix}"

        try:
            lib = _libs.get(namespace)
            if lib is None:
                # Never torn down: wrappers on hot paths may still call its ops.
                lib = Library(namespace, "FRAGMENT")
                _libs[namespace] = lib
            params = ", ".join(f"Tensor a{i}" for i in range(arity))
            lib.define(f"{op_name}({params}) -> ()")
            impl = _launcher_impl if namespace == LAUNCHER_NAMESPACE else _noop
            lib.impl(op_name, impl, "CompositeExplicitAutograd")
            op = getattr(getattr(torch.ops, namespace), op_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("could not define %s::%s: %s", namespace, name, exc)
            op = _BROKEN

        taken.add(op_name)
        _ops[key] = op
        return op


def _emit(namespace: str, name: str, tensors: Sequence[torch.Tensor]) -> bool:
    key = (namespace, name, len(tensors))
    op = _op_for(*key)
    if op is _BROKEN:
        return False
    try:
        op(*tensors)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("%s::%s failed, disabling it: %s", namespace, name, exc)
        _ops[key] = _BROKEN
        return False


def mark_launch(kernel_name: str, tensors: Sequence[torch.Tensor]) -> None:
    """Annotate a launch by hand, for paths that are not hooked. No-op when off."""
    global _launches
    if not _enabled or not tensors:
        return
    if _emit(NAMESPACE, kernel_name, tensors):
        _launches += 1


def _record_launch(
    kernel_name: str,
    launcher_name: Optional[str],
    tensors: Sequence[torch.Tensor],
) -> None:
    global _launches
    if launcher_name:
        _emit(LAUNCHER_NAMESPACE, launcher_name, tensors)
    if _emit(NAMESPACE, kernel_name, tensors):
        _launches += 1


# ---------------------------------------------------------------------------
# Launch introspection.
# ---------------------------------------------------------------------------
def _collect_tensors(args: Sequence[Any], kwargs: Dict[str, Any]) -> List[torch.Tensor]:
    """Tensors in the arguments, up to two list/tuple levels deep, in call order.

    Flat loops: under ``with_stack`` every Python call here is a trace event.
    """
    found: List[torch.Tensor] = []
    seq = (list, tuple)
    for values in (args, kwargs.values()):
        for value in values:
            if isinstance(value, torch.Tensor):
                found.append(value)
            elif isinstance(value, seq):
                for item in value:
                    if isinstance(item, torch.Tensor):
                        found.append(item)
                    elif isinstance(item, seq):
                        for inner in item:
                            if isinstance(inner, torch.Tensor):
                                found.append(inner)
    return found[:_MAX_TENSORS]


def _frame_tensors(frame) -> List[torch.Tensor]:
    """Tensor parameters of the function in ``frame``, for pointer-only launches."""
    code = frame.f_code
    flags = code.co_flags
    count = code.co_argcount + code.co_kwonlyargcount
    count += bool(flags & inspect.CO_VARARGS)
    names = code.co_varnames[:count]
    local_vars = frame.f_locals
    values = [local_vars[n] for n in names if n in local_vars]
    kwargs: Dict[str, Any] = {}
    if flags & inspect.CO_VARKEYWORDS:
        extra = local_vars.get(code.co_varnames[count])
        if isinstance(extra, dict):
            kwargs = extra
    return _collect_tensors(values, kwargs)


def _name_of(*candidates: Any) -> Optional[str]:
    """First ``__name__`` on the candidates or their ``fn`` / ``func``."""
    for obj in candidates:
        if obj is None:
            continue
        for target in (obj, getattr(obj, "fn", None), getattr(obj, "func", None)):
            name = getattr(target, "__name__", None)
            if isinstance(name, str) and name:
                return name
    return None


def _calling_launcher():
    """First caller frame outside the JIT machinery, skipping ``<lambda>`` etc."""
    try:
        frame = sys._getframe(2)
    except ValueError:  # pragma: no cover - stack shallower than expected
        return None
    depth = 0
    while frame is not None and depth < 20:
        module = frame.f_globals.get("__name__") or ""
        root = module.split(".", 1)[0]
        if (
            root not in _JIT_MODULES
            and module != __name__
            and module not in _JIT_HELPER_MODULES
        ):
            if not frame.f_code.co_name.startswith("<"):
                return frame
        frame = frame.f_back
        depth += 1
    return None


def _launcher_name(code: types.CodeType) -> str:
    # co_qualname keeps two classes' forward() distinct.
    return getattr(code, "co_qualname", None) or code.co_name


# ---------------------------------------------------------------------------
# Launcher promotion: swap a launcher's __code__ for a trampoline so later
# calls run inside launcher_shapes::<launcher>. Swapping in place covers every
# reference to the function without scanning modules.
# ---------------------------------------------------------------------------
def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    check = getattr(compiler, "is_compiling", None)
    try:
        return bool(check()) if check is not None else False
    except Exception:  # pragma: no cover - defensive
        return False


class _Promoted:
    """A launcher whose code object can be swapped for a trampoline."""

    def __init__(self, func: types.FunctionType, name: str, key: int) -> None:
        self.func = func
        self.name = name
        self.code = func.__code__
        self.defaults = func.__defaults__
        self.kwdefaults = func.__kwdefaults__
        self.had_signature = "__signature__" in func.__dict__

        orig = types.FunctionType(
            self.code, func.__globals__, func.__name__, self.defaults
        )
        orig.__kwdefaults__ = self.kwdefaults
        orig.__qualname__ = func.__qualname__
        self.orig = orig
        try:
            self.signature = inspect.signature(orig)
        except (TypeError, ValueError):  # pragma: no cover - exotic signatures
            self.signature = None

        src = (
            f"def {self.code.co_name}(*args, **kwargs):\n"
            f"    return {_TRAMPOLINE_GLOBAL}({key}, args, kwargs)\n"
        )
        ns: Dict[str, Any] = {}
        exec(compile(src, "<generic_shape_profiler>", "exec"), {}, ns)
        trampoline = ns[self.code.co_name].__code__
        if hasattr(trampoline, "co_qualname"):
            trampoline = trampoline.replace(co_qualname=self.code.co_qualname)
        self.trampoline = trampoline
        self.active = False

    def swap_in(self) -> None:
        if self.active:
            return
        func = self.func
        func.__globals__.setdefault(_TRAMPOLINE_GLOBAL, _call_launcher)
        # Keep inspect.signature() reporting the real parameters.
        if not self.had_signature and self.signature is not None:
            func.__signature__ = self.signature
        func.__code__ = self.trampoline
        func.__defaults__ = None
        func.__kwdefaults__ = None
        self.active = True

    def swap_out(self) -> None:
        if not self.active:
            return
        func = self.func
        func.__code__ = self.code
        func.__defaults__ = self.defaults
        func.__kwdefaults__ = self.kwdefaults
        if not self.had_signature:
            func.__dict__.pop("__signature__", None)
        self.active = False


def _function_for(frame) -> Optional[types.FunctionType]:
    """Function running ``frame``: module globals (via ``__wrapped__``), then self/cls."""
    code = frame.f_code

    def match(obj: Any) -> Optional[types.FunctionType]:
        for _ in range(8):
            if isinstance(obj, (staticmethod, classmethod)):
                obj = obj.__func__
            if isinstance(obj, types.FunctionType) and obj.__code__ is code:
                return obj
            obj = getattr(obj, "__wrapped__", None)
            if obj is None:
                return None
        return None

    found = match(frame.f_globals.get(code.co_name))
    if found is not None:
        return found
    local_vars = frame.f_locals
    for owner_name in ("self", "cls"):
        owner = local_vars.get(owner_name)
        if owner is None:
            continue
        klass = owner if isinstance(owner, type) else type(owner)
        for base in klass.__mro__:
            found = match(base.__dict__.get(code.co_name))
            if found is not None:
                return found
    return None


def _maybe_promote(frame, name: str) -> bool:
    """Promote the launcher in ``frame``; True only if this call promoted it."""
    code = frame.f_code
    if not _PROMOTE_ON or code in _promoted or code in _unpromotable:
        return False
    with _lock:
        if code in _promoted or code in _unpromotable:
            return False
        root = (frame.f_globals.get("__name__") or "").split(".", 1)[0]
        func = None
        if (
            root not in _NO_PROMOTE_MODULES
            and not code.co_freevars
            and not (code.co_flags & _CO_NO_PROMOTE)
        ):
            try:
                func = _function_for(frame)
            except Exception:  # pragma: no cover - defensive
                func = None
        if func is None or func.__closure__:
            _unpromotable.add(code)
            return False
        try:
            entry = _Promoted(func, name, len(_promoted_by_key))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("could not promote %s: %s", name, exc)
            _unpromotable.add(code)
            return False
        _promoted[code] = entry
        _promoted_by_key[len(_promoted_by_key)] = entry
        if _enabled:
            entry.swap_in()
        return True


def _launcher_stack() -> List[Any]:
    stack = getattr(_tls, "launcher_stack", None)
    if stack is None:
        stack = _tls.launcher_stack = []
    return stack


def _call_launcher(key: int, args: tuple, kwargs: dict) -> Any:
    """Trampoline target: run a promoted launcher inside its launcher op."""
    entry = _promoted_by_key[key]
    orig = entry.orig
    if not _enabled or _is_compiling():
        return orig(*args, **kwargs)
    try:
        tensors = _collect_tensors(args, kwargs)
        op = (
            _op_for(LAUNCHER_NAMESPACE, entry.name, len(tensors))
            if tensors
            else _BROKEN
        )
    except Exception:  # pragma: no cover - never break the workload
        op = _BROKEN
    if op is _BROKEN:
        return orig(*args, **kwargs)

    box: Dict[str, Any] = {}

    def run() -> None:
        box["ran"] = True
        try:
            box["result"] = orig(*args, **kwargs)
        except BaseException as exc:  # re-raised below, outside the dispatcher
            box["exc"] = exc

    stack = _launcher_stack()
    stack.append(entry.code)
    _tls.pending = run
    try:
        op(*tensors)
    except Exception as exc:
        if "ran" in box:
            raise
        logger.debug("launcher_shapes::%s failed, running it bare: %s", entry.name, exc)
    finally:
        _tls.pending = None
        stack.pop()

    if "ran" not in box:
        return orig(*args, **kwargs)
    if "exc" in box:
        raise box["exc"]
    return box["result"]


def _trace_launch(
    hook_id: str,
    name: str,
    tensors: Optional[List[torch.Tensor]],
    launch: Callable[[], Any],
) -> Any:
    """Annotate one launch and perform it with ``launch()``.

    ``tensors=None`` (pointer-only launch) records the launcher's tensor
    arguments instead. Only the outermost hooked launch on a thread is
    recorded, so ``Autotuner.run`` -> ``JITFunction.run`` counts once.
    """
    if getattr(_tls, "depth", 0):
        return launch()
    _tls.depth = 1
    try:
        frame = _calling_launcher() if _LAUNCHER_OPS_ON or tensors is None else None
        if tensors is None:
            tensors = _frame_tensors(frame) if frame is not None else []
            if not tensors:
                return launch()
        launcher = None
        if frame is not None and _LAUNCHER_OPS_ON:
            # Promoted launchers get their own enclosing op, including on the
            # call that promotes them, so no leaf op here.
            if frame.f_code not in _promoted:
                launcher_name = _launcher_name(frame.f_code)
                if not _maybe_promote(frame, launcher_name):
                    launcher = launcher_name
        frame = None
        if hook_id in _break_hooks and _break_rank_matches():
            # Inspect: hook_id, name, launcher, tensors; `up` for the hooked
            # call's args, kwargs and self.
            breakpoint()
        if not _SPANS_ON:
            _record_launch(name, launcher, tensors)
            return launch()
        # The GPU kernel launches inside the span, so it nests under it.
        with torch.profiler.record_function(SPAN_PREFIX + name):
            _record_launch(name, launcher, tensors)
            return launch()
    finally:
        _tls.depth = 0


# ---------------------------------------------------------------------------
# Hook installation. Each install_* returns False until its module is loaded.
# ---------------------------------------------------------------------------
def _wrap_launch(
    cls: type,
    method: str,
    name_of: Callable[[Any], Optional[str]],
    fallback_name: str,
    hook_id: str,
) -> bool:
    original = getattr(cls, method, None)
    if original is None:
        return False

    sentinel = f"_jit_shape_traced_{method.strip('_')}"
    if getattr(cls, sentinel, False):
        return True

    @functools.wraps(original)
    def traced(self, *args, **kwargs):
        if not _enabled:
            return original(self, *args, **kwargs)
        try:
            tensors = _collect_tensors(args, kwargs)
            if not tensors:
                return original(self, *args, **kwargs)
            name = name_of(self) or fallback_name
        except Exception:  # pragma: no cover - never break the workload
            return original(self, *args, **kwargs)
        return _trace_launch(
            hook_id, name, tensors, lambda: original(self, *args, **kwargs)
        )

    setattr(cls, method, traced)
    setattr(cls, sentinel, True)
    _installed[hook_id] = f"{cls.__module__}.{cls.__qualname__}.{method}"
    if _should_break(hook_id):
        _break_hooks.add(hook_id)
    return True


def _should_break(hook_id: str) -> bool:
    if _BREAK_SPEC in ("", "0", "false", "no", "off"):
        return False
    if _BREAK_SPEC in ("1", "true", "yes", "on", "all"):
        return True
    return any(
        hook_id.startswith(p.strip()) for p in _BREAK_SPEC.split(",") if p.strip()
    )


def _break_rank_matches() -> bool:
    if not _BREAK_RANK:
        return True
    try:
        dist = torch.distributed
        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank()) == _BREAK_RANK
    except Exception:
        pass
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == _BREAK_RANK


def install_triton_hooks() -> bool:
    """``JITFunction.run``, ``Autotuner.run``, ``Heuristics.run``."""
    ok = False

    jit_mod = sys.modules.get("triton.runtime.jit")
    if jit_mod is not None:
        cls = getattr(jit_mod, "JITFunction", None)
        if isinstance(cls, type):
            ok |= _wrap_launch(cls, "run", _name_of, "triton_kernel", "triton.jit")

    tuner_mod = sys.modules.get("triton.runtime.autotuner")
    if tuner_mod is not None:
        for cls_name, hook_id in (
            ("Autotuner", "triton.autotuner"),
            ("Heuristics", "triton.heuristics"),
        ):
            cls = getattr(tuner_mod, cls_name, None)
            if isinstance(cls, type):
                ok |= _wrap_launch(cls, "run", _name_of, "triton_kernel", hook_id)

    return ok


def install_flydsl_hooks() -> bool:
    """FlyDSL ``JitFunction.__call__``."""
    mod = sys.modules.get("flydsl.compiler.jit_function")
    if mod is None:
        return False
    cls = getattr(mod, "JitFunction", None)
    if not isinstance(cls, type):
        return False
    return _wrap_launch(cls, "__call__", _name_of, "flydsl_kernel", "flydsl.jit")


class _TracedForeignCall:
    """Annotating proxy for a tvm-ffi / ctypes entry point.

    Attribute access is forwarded, so ``argtypes`` / ``restype`` still work.
    """

    __slots__ = ("_func", "_name", "_hook_id", "_pointer_args")

    def __init__(self, func: Any, name: str, hook_id: str, pointer_args: bool) -> None:
        object.__setattr__(self, "_func", func)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_hook_id", hook_id)
        # Raw-pointer call: operands come from the caller's tensors.
        object.__setattr__(self, "_pointer_args", pointer_args)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        func = self._func
        if not _enabled:
            return func(*args, **kwargs)
        try:
            tensors = None if self._pointer_args else _collect_tensors(args, kwargs)
        except Exception:  # pragma: no cover - never break the workload
            return func(*args, **kwargs)
        if tensors is not None and not tensors:
            return func(*args, **kwargs)
        return _trace_launch(
            self._hook_id, self._name, tensors, lambda: func(*args, **kwargs)
        )

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._func, attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        setattr(self._func, attr, value)

    def __repr__(self) -> str:
        return f"<traced {self._func!r}>"


def install_tvm_ffi_hooks() -> bool:
    """``tvm_ffi.Module.__getattr__``, which backs SGLang's ``load_jit`` kernels.

    It caches each function it resolves, so this must precede the first lookup.
    """
    mod = sys.modules.get("tvm_ffi.module")
    cls = getattr(mod, "Module", None) if mod is not None else None
    if not isinstance(cls, type):
        return False
    if getattr(cls, "_jit_shape_traced_getattr", False):
        return True
    original = getattr(cls, "__getattr__", None)
    if original is None:
        return False

    @functools.wraps(original)
    def __getattr__(self, name):
        func = original(self, name)
        if name.startswith("__") or not callable(func):
            return func
        traced = _TracedForeignCall(func, name, "tvm_ffi", pointer_args=False)
        try:
            self.__dict__[name] = traced
        except Exception:  # pragma: no cover - read-only instance dict
            pass
        return traced

    cls.__getattr__ = __getattr__
    cls._jit_shape_traced_getattr = True
    _installed["tvm_ffi"] = f"{cls.__module__}.{cls.__qualname__}.__getattr__"
    if _should_break("tvm_ffi"):
        _break_hooks.add("tvm_ffi")
    return True


def install_aiter_cpp_itfs_hooks() -> bool:
    """aiter ``csrc.cpp_itfs``: ``run_lib`` (ctypes) and ``run_hsaco``.

    ``run_lib`` entry points take raw pointers, so operands come from the
    calling wrapper. False until both are defined.
    """
    mod = sys.modules.get("csrc.cpp_itfs.utils")
    if mod is None:
        return False
    run_lib = getattr(mod, "run_lib", None)
    run_hsaco = getattr(mod, "run_hsaco", None)

    if callable(run_lib) and not getattr(run_lib, "_jit_shape_traced", False):
        wrapped: Dict[tuple, _TracedForeignCall] = {}

        def traced_run_lib(func_name, folder=None):
            func = run_lib(func_name, folder)
            key = (func_name, folder)
            entry = wrapped.get(key)
            if entry is None or entry._func is not func:
                name = _HASH_SUFFIX.sub("", str(func_name))
                entry = wrapped[key] = _TracedForeignCall(
                    func, name, "aiter.cpp_itfs", pointer_args=True
                )
            return entry

        functools.update_wrapper(traced_run_lib, run_lib)
        for attr in ("cache_clear", "cache_info"):
            if hasattr(run_lib, attr):
                setattr(traced_run_lib, attr, getattr(run_lib, attr))
        traced_run_lib._jit_shape_traced = True
        mod.run_lib = traced_run_lib
        _installed["aiter.cpp_itfs"] = f"{mod.__name__}.run_lib"

    if callable(run_hsaco) and not getattr(run_hsaco, "_jit_shape_traced", False):

        @functools.wraps(run_hsaco)
        def traced_run_hsaco(func_name, *args, **kwargs):
            if not _enabled:
                return run_hsaco(func_name, *args, **kwargs)
            try:
                tensors = _collect_tensors(args, {})
            except Exception:  # pragma: no cover - never break the workload
                tensors = []
            if not tensors:
                return run_hsaco(func_name, *args, **kwargs)
            return _trace_launch(
                "aiter.hsaco",
                str(func_name),
                tensors,
                lambda: run_hsaco(func_name, *args, **kwargs),
            )

        traced_run_hsaco._jit_shape_traced = True
        mod.run_hsaco = traced_run_hsaco
        _installed["aiter.hsaco"] = f"{mod.__name__}.run_hsaco"

    for hook_id in ("aiter.cpp_itfs", "aiter.hsaco"):
        if hook_id in _installed and _should_break(hook_id):
            _break_hooks.add(hook_id)
    return callable(run_lib) and callable(run_hsaco)


def _traced_tensor_call(func: Callable, name: str, hook_id: str) -> Callable:
    """Wrap a launcher whose arguments carry the operand tensors."""

    @functools.wraps(func)
    def traced(*args, **kwargs):
        if not _enabled:
            return func(*args, **kwargs)
        try:
            tensors = _collect_tensors(args, kwargs)
        except Exception:  # pragma: no cover - never break the workload
            return func(*args, **kwargs)
        if not tensors:
            return func(*args, **kwargs)
        return _trace_launch(hook_id, name, tensors, lambda: func(*args, **kwargs))

    traced._jit_shape_traced = True
    return traced


def install_triton_compiled_hooks() -> bool:
    """``CompiledKernel.__getitem__``: ``compiled[grid](*args)`` relaunches.

    These skip ``JITFunction.run`` (e.g. ATOM's packed H2D scatter).
    """
    mod = sys.modules.get("triton.compiler.compiler")
    cls = getattr(mod, "CompiledKernel", None) if mod is not None else None
    if not isinstance(cls, type):
        return False
    if getattr(cls, "_jit_shape_traced_getitem", False):
        return True
    original = getattr(cls, "__getitem__", None)
    if original is None:
        return False

    @functools.wraps(original)
    def __getitem__(self, grid):
        runner = original(self, grid)
        if not _enabled:
            return runner
        return _traced_tensor_call(
            runner, getattr(self, "name", None) or "triton_kernel", "triton.compiled"
        )

    cls.__getitem__ = __getitem__
    cls._jit_shape_traced_getitem = True
    _installed["triton.compiled"] = f"{cls.__module__}.{cls.__qualname__}.__getitem__"
    if _should_break("triton.compiled"):
        _break_hooks.add("triton.compiled")
    return True


def install_aiter_flydsl_hooks() -> bool:
    """aiter ``tensor_shim._run_compiled``, which launches cached FlyDSL executables.

    Arguments are raw pointers, so operands come from the calling wrapper.
    Callers import it by name, so this must run when ``tensor_shim`` loads.
    """
    mod = sys.modules.get("aiter.ops.flydsl.kernels.tensor_shim")
    original = getattr(mod, "_run_compiled", None) if mod is not None else None
    if not callable(original):
        return False
    if getattr(original, "_jit_shape_traced", False):
        return True

    @functools.wraps(original)
    def _run_compiled(exe, *args, **kwargs):
        if not _enabled:
            return original(exe, *args, **kwargs)
        name = _name_of(exe) or "flydsl_kernel"
        return _trace_launch(
            "aiter.flydsl", name, None, lambda: original(exe, *args, **kwargs)
        )

    _run_compiled._jit_shape_traced = True
    mod._run_compiled = _run_compiled
    _installed["aiter.flydsl"] = f"{mod.__name__}._run_compiled"
    if _should_break("aiter.flydsl"):
        _break_hooks.add("aiter.flydsl")
    return True


def install_aiter_flydsl_moe_hooks() -> bool:
    """aiter ``moe_kernels._run_compiled`` (FP4 MoE), same pattern as ``tensor_shim``."""
    mod = sys.modules.get("aiter.ops.flydsl.moe_kernels")
    original = getattr(mod, "_run_compiled", None) if mod is not None else None
    if not callable(original):
        return False
    if getattr(original, "_jit_shape_traced", False):
        return True

    @functools.wraps(original)
    def _run_compiled(exe, *args, **kwargs):
        if not _enabled:
            return original(exe, *args, **kwargs)
        name = _name_of(exe) or "flydsl_kernel"
        return _trace_launch(
            "aiter.flydsl.moe", name, None, lambda: original(exe, *args, **kwargs)
        )

    _run_compiled._jit_shape_traced = True
    mod._run_compiled = _run_compiled
    _installed["aiter.flydsl.moe"] = f"{mod.__name__}._run_compiled"
    if _should_break("aiter.flydsl.moe"):
        _break_hooks.add("aiter.flydsl.moe")
    return True


def install_aiter_compile_ops_hooks() -> bool:
    """aiter ``compile_ops``, for the ``NONE_WRAPPED_OP`` ops it keeps off the dispatcher.

    The decorator runs as ``aiter.ops.*`` import, so this must run when
    ``aiter.jit.core`` loads.
    """
    core = sys.modules.get("aiter.jit.core")
    original = getattr(core, "compile_ops", None) if core is not None else None
    if not callable(original):
        return False
    if getattr(original, "_jit_shape_traced", False):
        return True
    guard = getattr(core, "torch_compile_guard", None)
    guard_mod = sys.modules.get(getattr(guard, "__module__", "") or "")
    none_wrapped = frozenset(getattr(guard_mod, "NONE_WRAPPED_OP", ()) or ())
    if not none_wrapped:
        return False

    @functools.wraps(original)
    def compile_ops(*args, **kwargs):
        decorator = original(*args, **kwargs)

        def traced_decorator(func):
            op = decorator(func)
            name = getattr(func, "__name__", "")
            if name in none_wrapped and callable(op):
                return _traced_tensor_call(op, name, "aiter.compile_ops")
            return op

        return traced_decorator

    compile_ops._jit_shape_traced = True
    core.compile_ops = compile_ops
    _installed["aiter.compile_ops"] = f"{core.__name__}.compile_ops"
    if _should_break("aiter.compile_ops"):
        _break_hooks.add("aiter.compile_ops")
    return True


def _tilelang_name(kernel: Any) -> Optional[str]:
    """``@tilelang.jit`` factory name, else the entry symbol (always ``main``)."""
    name = getattr(kernel, "_jit_shape_name", None)
    if name:
        return name
    func = getattr(kernel, "prim_func", None)
    try:
        return str(func.attrs["global_symbol"])
    except Exception:
        return None


def install_tilelang_kernel_hooks() -> bool:
    """TileLang ``JITKernel.__call__``. Kernel-allocated outputs are not recorded."""
    mod = sys.modules.get("tilelang.jit.kernel")
    cls = getattr(mod, "JITKernel", None) if mod is not None else None
    if not isinstance(cls, type):
        return False
    return _wrap_launch(
        cls, "__call__", _tilelang_name, "tilelang_kernel", "tilelang.jit"
    )


def install_tilelang_jit_hooks() -> bool:
    """Tag ``@tilelang.jit`` kernels with their factory name at compile time.

    Kernels compile during warmup, so this must run when ``tilelang.jit`` loads.
    """
    mod = sys.modules.get("tilelang.jit")
    cls = getattr(mod, "JITImpl", None) if mod is not None else None
    original = getattr(cls, "compile", None) if isinstance(cls, type) else None
    if not callable(original):
        return False
    if getattr(cls, "_jit_shape_traced_compile", False):
        return True

    @functools.wraps(original)
    def compile(self, *args, **kwargs):
        kernel = original(self, *args, **kwargs)
        try:
            name = _name_of(self.func, getattr(self.func, "func", None))
            if name and kernel is not None:
                kernel._jit_shape_name = name
        except Exception:  # pragma: no cover - never break the workload
            pass
        return kernel

    cls.compile = compile
    cls._jit_shape_traced_compile = True
    _installed["tilelang.compile"] = f"{cls.__module__}.{cls.__qualname__}.compile"
    return True


def install_hooks() -> bool:
    """Install every hook whose library is loaded. Idempotent."""
    triton = install_triton_hooks()
    compiled = install_triton_compiled_hooks()
    flydsl = install_flydsl_hooks()
    tvm_ffi = install_tvm_ffi_hooks()
    cpp_itfs = install_aiter_cpp_itfs_hooks()
    aiter_flydsl = install_aiter_flydsl_hooks()
    compile_ops = install_aiter_compile_ops_hooks()
    tilelang = install_tilelang_kernel_hooks()
    tilelang_names = install_tilelang_jit_hooks()
    return (
        triton
        or compiled
        or flydsl
        or tvm_ffi
        or cpp_itfs
        or aiter_flydsl
        or compile_ops
        or tilelang
        or tilelang_names
    )
