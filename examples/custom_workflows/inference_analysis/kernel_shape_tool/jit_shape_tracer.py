"""
Framework-agnostic kernel-shape tracing for JIT-compiled GPU kernels.

Why this exists
---------------
A torch profiler trace already carries full shape/dtype metadata for anything
that goes through the PyTorch dispatcher: aten ops and every ``torch.ops.*``
custom op (aiter, FlashInfer, sgl-kernel, DeepGEMM wrappers, ...) are recorded
with ``Input Dims`` / ``Input type`` for free.

Triton and FlyDSL kernels are the exception. They are launched straight from
Python onto the stream, bypassing the dispatcher entirely, so they show up in
the trace as bare GPU kernels with no operand information -- which makes them
impossible to roofline.

This module closes exactly that gap, and nothing else. It hooks the JIT launch
entry points and, for every launch, calls a no-op dispatcher op named after the
kernel whose single argument is the list of tensors being launched on. The
profiler then records that op like any other, so the kernel's operand shapes and
dtypes land in the trace.

Design
------
* **No framework knowledge.** The hooks are installed on Triton's and FlyDSL's
  own classes, so every launch is observed regardless of who calls it. There is
  no registry of entry points, no module scanning, no source-code heuristics,
  and no awareness of SGLang / vLLM / any serving stack.
* **Install once, gate with a flag.** Hooks are installed permanently the first
  time the JIT modules appear, and short-circuit on a boolean. ``enable()`` /
  ``disable()`` only flip that boolean, so there is never any patch/restore
  churn on the hot path and no risk of stale wrappers.
* **One op per kernel, one Tensor parameter per operand.** The op is defined
  lazily on first launch as ``kernel_shapes::<kernel>(Tensor a0, Tensor a1, ...)``,
  so the profiler records ``Input Dims`` / ``Input Strides`` / ``Input type``
  for it exactly as it does for a native op -- a ``Tensor[]`` parameter would
  have collapsed every dtype to the string ``TensorList``. The op name *is* the
  kernel name, and the ``record_function`` span around the launch nests the GPU
  kernel under the annotation for an exact association.
* **Outermost launch only.** ``Autotuner.run`` calls ``JITFunction.run``
  internally; a thread-local depth counter means the launch is recorded once.
* **Launcher level too.** A Triton kernel is usually reached through a Python
  wrapper (aiter's ``gemm_a8w8_blockscale``, say). With ``with_stack`` that
  wrapper shows up only as a ``python_function`` frame, which carries no
  operand metadata, so each launch is *also* annotated as
  ``launcher_shapes::<wrapper>``. The wrapper is found by walking out of the
  JIT machinery by module, which needs no registry and no name heuristics.

Namespaces
----------
``kernel_shapes::<kernel>``    named after the Triton/FlyDSL kernel
``launcher_shapes::<caller>``  named after the Python function that launched it

Both carry the same operands, so pick one granularity when summing FLOPs.

Public API
----------
``install_hooks()``          install whatever JIT hooks are available now
``enable()`` / ``disable()`` start / stop recording (driven by the profiler)
``mark_launch(name, args)``  annotate a launch by hand, for launchers that are
                             neither Triton nor FlyDSL
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import torch
from torch.library import Library

logger = logging.getLogger("jit_shape_tracer")

# Dispatcher namespace for kernel-level annotations (named after the kernel).
NAMESPACE = "kernel_shapes"

# Dispatcher namespace for launcher-level annotations (named after the Python
# function that issued the launch, e.g. aiter's ``gemm_a8w8_blockscale``). Kept
# separate from NAMESPACE so a roofline pass can pick one granularity and not
# double-count the same work at both levels.
LAUNCHER_NAMESPACE = "launcher_shapes"

# Prefix for the record_function span wrapped around each launch.
SPAN_PREFIX = "jit::"


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


# Emit the enclosing record_function span (set to 0 for shapes only).
_SPANS_ON = _env_on("TRACELENS_SHAPE_SPANS")

# Also annotate the calling Python launcher as a dispatcher op.
_LAUNCHER_OPS_ON = _env_on("TRACELENS_SHAPE_LAUNCHER_OPS")

# Debug only: drop into breakpoint() at each hooked launch. "1"/"all" breaks on
# every hook; otherwise a comma-separated list of hook-id prefixes, e.g.
# "triton", "flydsl", "triton.autotuner". Honours PYTHONBREAKPOINT.
_BREAK_SPEC = os.environ.get("TRACELENS_SHAPE_BREAK", "").strip().lower()

# Debug only: restrict the breakpoint to one torch.distributed rank, so a TP=N
# server stops in a single process instead of N.
_BREAK_RANK = os.environ.get("TRACELENS_SHAPE_BREAK_RANK", "").strip()

# Module prefixes that belong to the JIT machinery itself. A frame in one of
# these is never "the launcher" -- notably Triton's own KernelInterface
# __getitem__ lambda sits between the caller and .run().
_JIT_MODULES = ("triton", "flydsl")

# Cap on tensors recorded per launch, so a pathological kernel cannot blow up
# the trace. Triton kernels realistically take fewer than a dozen.
_MAX_TENSORS = 32

_lock = threading.RLock()
_tls = threading.local()

_enabled = False
_libs: Dict[str, Library] = {}  # namespace -> Library
_ops: Dict[tuple, Any] = {}  # (namespace, name, arity) -> op callable / _BROKEN
_taken: Dict[str, Set[str]] = {}  # namespace -> op names already defined
_installed: Dict[str, str] = {}  # hook id -> what was patched
_break_hooks: Set[str] = set()  # hook ids that call breakpoint(); clear() to stop
_launches = 0
_window_start = 0  # value of _launches when the current window opened

_BROKEN = object()  # sentinel: op could not be defined/called, stop retrying


# ---------------------------------------------------------------------------
# Enable / disable. Only flips a flag -- hooks stay installed.
# ---------------------------------------------------------------------------
def enable() -> None:
    global _enabled, _window_start
    with _lock:
        if _enabled:
            return
        install_hooks()
        _window_start = _launches
        _enabled = True
    logger.info(
        "jit_shape_tracer enabled (hooks: %s)",
        ", ".join(sorted(_installed.values())) or "none available",
    )


def disable() -> None:
    global _enabled
    with _lock:
        if not _enabled:
            return
        _enabled = False
        window = _launches - _window_start
    logger.info(
        "jit_shape_tracer disabled (%d launches this window, %d total, "
        "%d distinct kernels, %d distinct launchers)",
        window,
        _launches,
        len({n for ns, n, _ in _ops if ns == NAMESPACE}),
        len({n for ns, n, _ in _ops if ns == LAUNCHER_NAMESPACE}),
    )


def is_enabled() -> bool:
    return _enabled


def stats() -> Dict[str, Any]:
    return {
        "enabled": _enabled,
        "hooks": dict(_installed),
        "kernels": sorted({n for ns, n, _ in _ops if ns == NAMESPACE}),
        "launchers": sorted({n for ns, n, _ in _ops if ns == LAUNCHER_NAMESPACE}),
        "launches": _launches,
    }


# ---------------------------------------------------------------------------
# Dispatcher ops: kernel_shapes::<kernel>(Tensor a0, Tensor a1, ...) -> ()
#
# The implementation is a no-op; the op exists purely so the profiler records
# the operand shapes, strides and dtypes of the launch it precedes.
# ---------------------------------------------------------------------------
def _noop(*args: torch.Tensor) -> None:
    return None


def _sanitize(name: str) -> str:
    """Turn a kernel name into a valid op name, preserving it where possible.

    Leading underscores are kept: Triton kernels are very commonly named
    ``_foo_kernel``, and the op name is meant to equal the kernel name so the
    annotation can be matched to the GPU kernel by name.
    """
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

        # A schema's arity is fixed, so the same name launched with a different
        # number of tensors (optional operands) gets its own op. The first
        # arity seen keeps the bare name.
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
                # FRAGMENT so the namespace can be extended incrementally, and
                # never torn down: released ops would break any wrapper that is
                # still on a hot path.
                lib = Library(namespace, "FRAGMENT")
                _libs[namespace] = lib
            params = ", ".join(f"Tensor a{i}" for i in range(arity))
            lib.define(f"{op_name}({params}) -> ()")
            lib.impl(op_name, _noop, "CompositeExplicitAutograd")
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
    """Record one kernel launch's operands into the dispatcher.

    Safe to call from anywhere; a no-op unless tracing is enabled. Use this to
    annotate launchers that are neither Triton nor FlyDSL (a hand-written HIP
    extension, a CuTe kernel, ...) without teaching this module about them.
    """
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
    """Annotate one launch at kernel level and, when known, launcher level."""
    global _launches
    if launcher_name:
        _emit(LAUNCHER_NAMESPACE, launcher_name, tensors)
    if _emit(NAMESPACE, kernel_name, tensors):
        _launches += 1


# ---------------------------------------------------------------------------
# Launch introspection.
# ---------------------------------------------------------------------------
def _collect_tensors(args: Sequence[Any], kwargs: Dict[str, Any]) -> List[torch.Tensor]:
    """Tensors reachable from a launch's arguments, in call order.

    Recurses one level into lists/tuples (kernels that take tensor lists) and
    ignores everything else -- grids, constexprs, num_warps, streams.
    """
    found: List[torch.Tensor] = []

    def walk(value: Any, depth: int) -> None:
        if len(found) >= _MAX_TENSORS:
            return
        if isinstance(value, torch.Tensor):
            found.append(value)
        elif depth < 2 and isinstance(value, (list, tuple)):
            for item in value:
                walk(item, depth + 1)

    for value in args:
        walk(value, 0)
    for value in kwargs.values():
        walk(value, 0)
    return found


def _name_of(*candidates: Any) -> Optional[str]:
    """First usable ``__name__`` among the candidates or their inner functions.

    Triton wraps the user function differently across versions (``JITFunction``
    exposes ``__name__``, ``Autotuner``/``Heuristics`` nest it under ``fn``,
    FlyDSL under ``func``), so probe rather than assume a layout.
    """
    for obj in candidates:
        if obj is None:
            continue
        for target in (obj, getattr(obj, "fn", None), getattr(obj, "func", None)):
            name = getattr(target, "__name__", None)
            if isinstance(name, str) and name:
                return name
    return None


def _calling_launcher() -> Optional[str]:
    """Name of the Python function that issued this launch, or None.

    Walks out of the JIT machinery to the first frame in caller code. Triton
    reaches ``.run()`` through its own ``KernelInterface.__getitem__`` lambda,
    so skipping frames by *module* (rather than by name) is what lands on the
    real launcher -- e.g. ``gemm_a8w8_blockscale`` for aiter. Frames whose name
    is bracketed (``<lambda>``, ``<genexpr>``, ``<module>``) are skipped too,
    since they make useless op names.
    """
    try:
        frame = sys._getframe(2)
    except ValueError:  # pragma: no cover - stack shallower than expected
        return None
    depth = 0
    while frame is not None and depth < 20:
        module = frame.f_globals.get("__name__") or ""
        root = module.split(".", 1)[0]
        if root not in _JIT_MODULES and module != __name__:
            code = frame.f_code
            if not code.co_name.startswith("<"):
                # co_qualname (3.11+) keeps methods distinct: two classes with
                # a forward() would otherwise share one op.
                return getattr(code, "co_qualname", None) or code.co_name
        frame = frame.f_back
        depth += 1
    return None


@contextmanager
def _outermost_launch():
    """Yield True only for the outermost hooked launch on this thread.

    ``Autotuner.run`` -> ``JITFunction.run`` would otherwise record the same
    launch twice, with the inner call seeing fewer of the caller's arguments.
    """
    depth = getattr(_tls, "depth", 0)
    _tls.depth = depth + 1
    try:
        yield depth == 0
    finally:
        _tls.depth = depth


# ---------------------------------------------------------------------------
# Hook installation.
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

        with _outermost_launch() as outermost:
            if not outermost:
                return original(self, *args, **kwargs)
            # Keep this call in this frame: _calling_launcher() walks out from
            # here to find the caller.
            launcher = _calling_launcher() if _LAUNCHER_OPS_ON else None
            if hook_id in _break_hooks and _break_rank_matches():
                # Inspect: hook_id, name, launcher, tensors, args, kwargs, self.
                breakpoint()
            if not _SPANS_ON:
                _record_launch(name, launcher, tensors)
                return original(self, *args, **kwargs)
            # The span makes the association exact: the GPU kernel is launched
            # inside it, so it nests under the annotation in the trace.
            with torch.profiler.record_function(SPAN_PREFIX + name):
                _record_launch(name, launcher, tensors)
                return original(self, *args, **kwargs)

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
    return any(hook_id.startswith(p.strip()) for p in _BREAK_SPEC.split(",") if p.strip())


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
    """Hook Triton's launch paths. False if Triton is not imported yet."""
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
                ok |= _wrap_launch(
                    cls, "run", _name_of, "triton_kernel", hook_id
                )

    return ok


def install_flydsl_hooks() -> bool:
    """Hook FlyDSL's launch path. False if FlyDSL is not imported yet."""
    mod = sys.modules.get("flydsl.compiler.jit_function")
    if mod is None:
        return False
    cls = getattr(mod, "JitFunction", None)
    if not isinstance(cls, type):
        return False
    return _wrap_launch(cls, "__call__", _name_of, "flydsl_kernel", "flydsl.jit")


def install_hooks() -> bool:
    """Install every JIT hook whose library is currently importable.

    Idempotent and safe to call repeatedly -- the shim calls it again each time
    a new JIT module shows up in ``sys.modules``.
    """
    triton = install_triton_hooks()
    flydsl = install_flydsl_hooks()
    return triton or flydsl
