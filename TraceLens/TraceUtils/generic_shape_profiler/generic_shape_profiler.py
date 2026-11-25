###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Operand shapes for GPU kernels that bypass the PyTorch dispatcher.

JIT-compiled kernels (Triton, FlyDSL, TileLang, tvm-ffi, aiter ctypes/hsaco)
launch straight from Python, so profiler traces show them with no operand
metadata. Each hooked launch calls a no-op dispatcher op whose arguments are
the launch's, so the profiler records their shapes, strides, dtypes and scalar
values:

``kernel_shapes::<kernel>``    named after the kernel
``launcher_shapes::<caller>``  named after the Python function that launched it

Both carry the same work; pick one granularity when summing FLOPs.
Each parameter has a fixed slot in signature order; names are in the trace
metadata under ``SCHEMA_METADATA_KEY``. See README.md for the design.

Public API: ``install_hooks()``, ``enable()`` / ``disable()``,
``mark_launch(name, tensors)``, ``schema_table()``.
"""

from __future__ import annotations

import enum
import functools
import inspect
import json
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

SCHEMA_METADATA_KEY = "generic_shape_profiler_schemas"


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
_SCALARS_ON = _env_on("TRACELENS_SHAPE_SCALARS")

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

_MAX_SLOTS = 64
_MAX_LIST = 64


class SlotKind(str, enum.Enum):
    """A slot's kind; the value is the schema type its op declares.

    The ``str`` base keeps hashing and comparison in C: a plain Enum's Python
    ``__hash__`` would be a with_stack event per slot per launch.
    """

    TENSOR = "Tensor?"
    SCALAR = "Scalar?"
    DTYPE = "ScalarType?"
    TENSOR_LIST = "Tensor[]?"
    INT_LIST = "int[]?"
    NONE = ""  # no slot


# Membership tests add no with_stack events; isinstance() would.
_TENSOR_TYPES = {torch.Tensor, torch.nn.Parameter}
_SCALAR_TYPES = {int, float, bool}
_SEQ_TYPES = {list, tuple}
_I64 = 1 << 63

_UNSET = object()  # parameter neither passed nor defaulted

_POS_NAMES = tuple(f"a{i}" for i in range(_MAX_SLOTS))

_lock = threading.RLock()
_tls = threading.local()

_enabled = False
_libs: Dict[str, Library] = {}  # namespace -> Library
_ops: Dict[tuple, Any] = {}  # (namespace, name, spec) -> op callable / _BROKEN
_taken: Dict[str, Set[str]] = {}  # namespace -> op names already defined
_schemas: Dict[str, List[List[str]]] = {}  # "ns::op" -> [[param, schema type], ...]
_layouts: Dict[Any, "_Layout"] = {}  # code object -> layout
_obj_layouts: Dict[int, tuple] = {}  # id(obj) -> (obj, layout) when obj has no __dict__
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


def schema_table() -> Dict[str, List[List[str]]]:
    """``{"ns::op": [[param name, schema type], ...]}`` for every op defined."""
    with _lock:
        return {op: [list(p) for p in params] for op, params in _schemas.items()}


def add_schema_metadata(prof: Any) -> bool:
    """Write ``schema_table()`` into ``prof``'s trace; call before it stops."""
    if not _schemas:
        return False
    try:
        prof.add_metadata_json(SCHEMA_METADATA_KEY, json.dumps(schema_table()))
        return True
    except Exception as exc:  # pragma: no cover - defensive  # noqa: BLE001
        logger.debug("could not add schema metadata: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Dispatcher ops. A record is (spec, values); spec ((param, kind), ...) fixes
# the op's schema.
# ---------------------------------------------------------------------------
def _noop(*args: Any) -> None:
    return None


def _launcher_impl(*args: Any) -> None:
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


def _define_op(namespace: str, name: str, spec: tuple):
    with _lock:
        key = (namespace, name, spec)
        if key in _ops:
            return _ops[key]

        # Another spec under the same name gets a "__<N>t" suffix.
        taken = _taken.setdefault(namespace, set())
        base = _sanitize(name)
        n = len(spec)
        op_name = base if base not in taken else f"{base}__{n}t"
        suffix = 1
        while op_name in taken:
            suffix += 1
            op_name = f"{base}__{n}t_{suffix}"

        try:
            lib = _libs.get(namespace)
            if lib is None:
                # Never torn down: wrappers on hot paths may still call its ops.
                lib = Library(namespace, "FRAGMENT")
                _libs[namespace] = lib
            params = ", ".join(f"{kind.value} a{i}" for i, (_, kind) in enumerate(spec))
            lib.define(f"{op_name}({params}) -> ()")
            impl = _launcher_impl if namespace == LAUNCHER_NAMESPACE else _noop
            lib.impl(op_name, impl, "CompositeExplicitAutograd")
            op = getattr(getattr(torch.ops, namespace), op_name)
            _schemas[f"{namespace}::{op_name}"] = [
                [param, kind.value] for param, kind in spec
            ]
        except Exception as exc:  # pragma: no cover - defensive  # noqa: BLE001
            logger.debug("could not define %s::%s: %s", namespace, name, exc)
            op = _BROKEN

        taken.add(op_name)
        _ops[key] = op
        return op


def _op_for(namespace: str, name: str, spec: tuple):
    try:
        return _ops[(namespace, name, spec)]
    except KeyError:
        return _define_op(namespace, name, spec)


def _emit(namespace: str, name: str, record: tuple) -> bool:
    spec, values = record
    try:
        op = _ops[(namespace, name, spec)]
    except KeyError:
        op = _define_op(namespace, name, spec)
    if op is _BROKEN:
        return False
    try:
        op(*values)
        return True
    except Exception as exc:  # pragma: no cover - defensive  # noqa: BLE001
        logger.debug("%s::%s failed, disabling it: %s", namespace, name, exc)
        _ops[(namespace, name, spec)] = _BROKEN
        return False


def mark_launch(kernel_name: str, tensors: Sequence[torch.Tensor]) -> None:
    """Annotate a launch by hand, for paths that are not hooked. No-op when off."""
    global _launches
    if not _enabled or not tensors:
        return
    record = _positional_record(tuple(tensors), {})
    if record is not None and _emit(NAMESPACE, kernel_name, record):
        _launches += 1


def _record_launch(
    kernel_name: str,
    record: tuple,
    launcher_name: Optional[str],
    launcher_record: Optional[tuple],
) -> None:
    global _launches
    if launcher_name and launcher_record is not None:
        _emit(LAUNCHER_NAMESPACE, launcher_name, launcher_record)
    if _emit(NAMESPACE, kernel_name, record):
        _launches += 1


# ---------------------------------------------------------------------------
# Argument binding. Builtin calls are with_stack events too, so the per-call
# paths avoid them.
# ---------------------------------------------------------------------------
class _Layout:
    """A function's parameter slots; kinds are fixed by the first call."""

    __slots__ = (
        "names",
        "npos",
        "index",
        "defaults",
        "varargs",
        "varkw",
        "hints",
        "slots",
        "spec",
    )

    def __init__(self, code: types.CodeType) -> None:
        n = code.co_argcount + code.co_kwonlyargcount
        self.names = code.co_varnames[:n]
        self.npos = code.co_argcount
        self.index = {name: i for i, name in enumerate(self.names)}
        flags = code.co_flags
        self.varargs = code.co_varnames[n] if flags & inspect.CO_VARARGS else None
        k = n + (self.varargs is not None)
        self.varkw = code.co_varnames[k] if flags & inspect.CO_VARKEYWORDS else None
        self.defaults = [_UNSET] * n
        self.hints: Dict[str, str] = {}
        self.slots: Optional[tuple] = None  # ((param index, kind), ...)
        self.spec: Optional[tuple] = None

    def add_function(self, func: types.FunctionType) -> None:
        defaults = func.__defaults__ or ()
        first = self.npos - len(defaults)
        for i, value in enumerate(defaults):
            self.defaults[first + i] = value
        for name, value in (func.__kwdefaults__ or {}).items():
            if name in self.index:
                self.defaults[self.index[name]] = value
        for name, hint in (getattr(func, "__annotations__", None) or {}).items():
            self.hints[name] = hint if isinstance(hint, str) else repr(hint)


def _layout_for(code: types.CodeType, func: Optional[types.FunctionType] = None):
    layout = _layouts.get(code)
    if layout is None:
        with _lock:
            layout = _layouts.get(code)
            if layout is None:
                layout = _layouts[code] = _Layout(code)
                if func is not None:
                    layout.add_function(func)
    return layout


def _new_frame_layout(frame) -> _Layout:
    try:
        func = _function_for(frame)
    except Exception:  # noqa: BLE001 - annotations are only hints
        func = None
    return _layout_for(frame.f_code, func)


def _function_of(obj: Any) -> Optional[types.FunctionType]:
    """Innermost function behind ``obj`` via ``__wrapped__`` / ``fn`` / ``func``."""
    cur = obj
    for _ in range(8):
        inner = getattr(cur, "__wrapped__", None)
        if inner is None and not isinstance(cur, types.FunctionType):
            inner = getattr(cur, "fn", None) or getattr(cur, "func", None)
        if inner is None or inner is cur:
            return cur if isinstance(cur, types.FunctionType) else None
        cur = inner
    return None


def _callable_layout(obj: Any) -> Optional[_Layout]:
    """None means bind positionally, as for a bare ``(*args, **kwargs)`` wrapper."""
    func = _function_of(obj)
    if func is None:
        return None
    code = func.__code__
    if code.co_argcount + code.co_kwonlyargcount == 0:
        return None
    return _layout_for(code, func)


def _object_layout(obj: Any) -> Optional[_Layout]:
    entry = _obj_layouts.get(id(obj))
    if entry is not None and entry[0] is obj:
        return entry[1]
    layout = _callable_layout(obj)
    try:
        obj._gsp_layout = layout
    except Exception:  # noqa: BLE001 - no instance __dict__; keep obj alive instead
        _obj_layouts[id(obj)] = (obj, layout)
    return layout


def _hint_kind(hint: Optional[str]) -> SlotKind:
    """Kind for a parameter first seen as None: unannotated is a tensor, other types no slot."""
    h = hint or ""
    if not h or "Any" in h:
        return SlotKind.TENSOR
    seq = any(s in h for s in ("List[", "list[", "Sequence[", "Tuple[", "tuple["))
    if "Tensor" in h:
        return SlotKind.TENSOR_LIST if seq else SlotKind.TENSOR
    scalar = "int" in h or "float" in h or "bool" in h
    if seq:
        return SlotKind.INT_LIST if scalar and _SCALARS_ON else SlotKind.NONE
    if scalar:
        return SlotKind.SCALAR if _SCALARS_ON else SlotKind.NONE
    return SlotKind.DTYPE if "dtype" in h and _SCALARS_ON else SlotKind.NONE


def _seq_kind(value: Any) -> SlotKind:
    if all(v is None or isinstance(v, torch.Tensor) for v in value):
        return SlotKind.TENSOR_LIST
    if all(isinstance(v, int) for v in value):
        return SlotKind.INT_LIST if _SCALARS_ON else SlotKind.NONE
    return SlotKind.NONE


def _kind_of(value: Any, hint: Optional[str] = None) -> SlotKind:
    if value is None or value is _UNSET:
        return _hint_kind(hint)
    if isinstance(value, torch.Tensor):
        return SlotKind.TENSOR
    if isinstance(value, (bool, int, float)):
        return SlotKind.SCALAR if _SCALARS_ON else SlotKind.NONE
    if isinstance(value, torch.dtype):
        return SlotKind.DTYPE if _SCALARS_ON else SlotKind.NONE
    if isinstance(value, (list, tuple)):
        return _seq_kind(value)
    return SlotKind.NONE


def _decide_slots(layout: _Layout, vals: list) -> None:
    with _lock:
        if layout.slots is not None:
            return
        slots = []
        for i, name in enumerate(layout.names):
            kind = _kind_of(vals[i], layout.hints.get(name))
            if kind != SlotKind.NONE:
                slots.append((i, kind))
        if layout.varargs is not None:
            slots.append((-1, SlotKind.TENSOR_LIST))
        if layout.varkw is not None:
            slots.append((-2, SlotKind.TENSOR_LIST))
        slots = tuple(slots[:_MAX_SLOTS])
        names = []
        for i, _ in slots:
            if i >= 0:
                names.append(layout.names[i])
            else:
                names.append("*" + layout.varargs if i == -1 else "**" + layout.varkw)
        layout.spec = tuple(zip(names, (kind for _, kind in slots)))
        layout.slots = slots


def _convert(
    layout: _Layout, vals: list, extra_args: tuple, extra_kwargs
) -> Optional[tuple]:
    """One value per slot, or None when no tensor was passed."""
    out = [None] * _MAX_SLOTS
    has_tensor = False
    s = 0
    for i, kind in layout.slots:
        v = vals[i] if i >= 0 else _extra_value(i, extra_args, extra_kwargs)
        if v is None or v is _UNSET:
            pass
        elif kind is SlotKind.TENSOR:
            if type(v) in _TENSOR_TYPES or isinstance(v, torch.Tensor):
                out[s] = v
                has_tensor = True
        elif kind is SlotKind.SCALAR:
            out[s] = v if type(v) in _SCALAR_TYPES else _plain_number(v)
            # Scalar? only takes int64-range values (nan fails this too).
            if out[s] is not None and not -_I64 <= out[s] < _I64:
                out[s] = None
        elif kind is SlotKind.DTYPE:
            if type(v) is torch.dtype:
                out[s] = v
        elif type(v) in _SEQ_TYPES:
            out[s] = _seq_value(kind, v)
            has_tensor = has_tensor or (
                kind is SlotKind.TENSOR_LIST and out[s] is not None
            )
        s += 1
    if not has_tensor:
        return None
    return layout.spec, out[:s]


def _plain_number(value: Any) -> Optional[Any]:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return None


def _extra_value(i: int, extra_args: tuple, extra_kwargs) -> tuple:
    """Slot -1 is ``*args``, slot -2 is ``**kwargs``."""
    if i == -1:
        return extra_args
    if not extra_kwargs:
        return ()
    return tuple(extra_kwargs[k] for k in sorted(extra_kwargs))


def _seq_value(kind: SlotKind, value: Any) -> Optional[list]:
    items = []
    for item in value[:_MAX_LIST]:
        if kind is SlotKind.TENSOR_LIST:
            if type(item) in _TENSOR_TYPES or isinstance(item, torch.Tensor):
                items += (item,)
        elif type(item) is int or type(item) is bool:
            items += (item,)
        else:
            return None
    return items or None


def _bind_record(layout: _Layout, args: tuple, kwargs: dict) -> Optional[tuple]:
    vals = layout.defaults[:]
    npos = layout.npos
    extra_args = ()
    i = 0
    for a in args:
        if i == npos:
            extra_args = args[i:]
            break
        vals[i] = a
        i += 1
    extra_kwargs = None
    index = layout.index
    for k in kwargs:
        if k in index:
            vals[index[k]] = kwargs[k]
        else:
            if extra_kwargs is None:
                extra_kwargs = {}
            extra_kwargs[k] = kwargs[k]
    if layout.slots is None:
        _decide_slots(layout, vals)
    return _convert(layout, vals, extra_args, extra_kwargs)


def _frame_record(frame) -> Optional[tuple]:
    code = frame.f_code
    layout = _layouts[code] if code in _layouts else _new_frame_layout(frame)
    local_vars = frame.f_locals
    vals = layout.defaults[:]
    i = 0
    for name in layout.names:
        if name in local_vars:
            vals[i] = local_vars[name]
        i += 1
    extra_args = ()
    if layout.varargs is not None and layout.varargs in local_vars:
        extra_args = local_vars[layout.varargs]
    extra_kwargs = None
    if layout.varkw is not None and layout.varkw in local_vars:
        extra_kwargs = local_vars[layout.varkw]
    if layout.slots is None:
        _decide_slots(layout, vals)
    return _convert(layout, vals, extra_args, extra_kwargs)


def _positional_record(args: tuple, kwargs: dict) -> Optional[tuple]:
    """For callees with no Python signature: call order, then kwargs by name."""
    vals = args
    if kwargs:
        vals = args + tuple(kwargs[k] for k in sorted(kwargs))
    out = [None] * _MAX_SLOTS
    kinds = [SlotKind.TENSOR] * _MAX_SLOTS
    has_tensor = False
    s = 0
    for v in vals:
        if s == _MAX_SLOTS:
            break
        t = type(v)
        if v is None:
            pass
        elif t in _TENSOR_TYPES or isinstance(v, torch.Tensor):
            out[s] = v
            has_tensor = True
        elif t in _SEQ_TYPES:
            kinds[s] = _seq_kind(v)
            out[s] = (
                _seq_value(kinds[s], v)
                if kinds[s] in (SlotKind.TENSOR_LIST, SlotKind.INT_LIST)
                else None
            )
            if out[s] is None:
                kinds[s] = SlotKind.TENSOR
            has_tensor = has_tensor or kinds[s] is SlotKind.TENSOR_LIST
        elif not _SCALARS_ON:
            pass
        elif t in _SCALAR_TYPES and -_I64 <= v < _I64:
            out[s] = v
            kinds[s] = SlotKind.SCALAR
        elif t is torch.dtype:
            out[s] = v
            kinds[s] = SlotKind.DTYPE
        s += 1
    if not has_tensor:
        return None
    return tuple(zip(_POS_NAMES[:s], kinds[:s])), out[:s]


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
    except Exception:  # pragma: no cover - defensive  # noqa: BLE001
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
        self.layout = _layout_for(self.code)
        # orig, rebuilt from the code, has no annotations.
        self.layout.add_function(func)
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
            except Exception:  # pragma: no cover - defensive  # noqa: BLE001
                func = None
        if func is None or func.__closure__:
            _unpromotable.add(code)
            return False
        try:
            entry = _Promoted(func, name, len(_promoted_by_key))
        except Exception as exc:  # pragma: no cover - defensive  # noqa: BLE001
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
        record = _bind_record(entry.layout, args, kwargs)
        op = (
            _op_for(LAUNCHER_NAMESPACE, entry.name, record[0])
            if record is not None
            else _BROKEN
        )
    except Exception:  # pragma: no cover - never break the workload  # noqa: BLE001
        op = _BROKEN
    if op is _BROKEN:
        return orig(*args, **kwargs)
    values = record[1]

    box: Dict[str, Any] = {}

    def run() -> None:
        box["ran"] = True
        try:
            box["result"] = orig(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["exc"] = exc

    stack = _launcher_stack()
    stack.append(entry.code)
    _tls.pending = run
    try:
        op(*values)
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
    record: Optional[tuple],
    launch: Callable[[], Any],
) -> Any:
    """Annotate one launch and perform it with ``launch()``.

    ``record=None`` (pointer-only launch) records the launcher's parameters
    instead. Only the outermost hooked launch on a thread is recorded, so
    ``Autotuner.run`` -> ``JITFunction.run`` counts once.
    """
    if getattr(_tls, "depth", 0):
        return launch()
    _tls.depth = 1
    try:
        frame = _calling_launcher() if _LAUNCHER_OPS_ON or record is None else None
        frame_record = None
        if record is None:
            frame_record = record = _frame_record(frame) if frame is not None else None
            if record is None:
                return launch()
        launcher = None
        launcher_record = None
        if frame is not None and _LAUNCHER_OPS_ON:
            # Promoted launchers get their own enclosing op, including on the
            # call that promotes them, so no leaf op here.
            if frame.f_code not in _promoted:
                launcher_name = _launcher_name(frame.f_code)
                if not _maybe_promote(frame, launcher_name):
                    launcher = launcher_name
                    launcher_record = (
                        frame_record
                        if frame_record is not None
                        else _frame_record(frame)
                    )
        frame = None
        if hook_id in _break_hooks and _break_rank_matches():
            # Inspect: hook_id, name, launcher, record; `up` for the hooked
            # call's args, kwargs and self.
            breakpoint()
        if not _SPANS_ON:
            _record_launch(name, record, launcher, launcher_record)
            return launch()
        # The GPU kernel launches inside the span, so it nests under it.
        with torch.profiler.record_function(SPAN_PREFIX + name):
            _record_launch(name, record, launcher, launcher_record)
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
    named: bool = True,
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
            layout = None
            if named:
                try:
                    layout = self._gsp_layout
                except AttributeError:
                    layout = _object_layout(self)
            if layout is None:
                record = _positional_record(args, kwargs)
            else:
                record = _bind_record(layout, args, kwargs)
            if record is None:
                return original(self, *args, **kwargs)
            name = name_of(self) or fallback_name
        except Exception:  # pragma: no cover - never break the workload  # noqa: BLE001
            return original(self, *args, **kwargs)
        return _trace_launch(
            hook_id, name, record, lambda: original(self, *args, **kwargs)
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
    except Exception:  # noqa: BLE001
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
        if self._pointer_args:
            record = None
        else:
            try:
                record = _positional_record(args, kwargs)
            except Exception:  # noqa: BLE001  # pragma: no cover
                return func(*args, **kwargs)
            if record is None:
                return func(*args, **kwargs)
        return _trace_launch(
            self._hook_id, self._name, record, lambda: func(*args, **kwargs)
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
        except Exception:  # pragma: no cover - read-only instance dict  # noqa: BLE001
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
                record = _positional_record(args, {})
            except Exception:  # noqa: BLE001  # pragma: no cover
                record = None
            if record is None:
                return run_hsaco(func_name, *args, **kwargs)
            return _trace_launch(
                "aiter.hsaco",
                str(func_name),
                record,
                lambda: run_hsaco(func_name, *args, **kwargs),
            )

        traced_run_hsaco._jit_shape_traced = True
        mod.run_hsaco = traced_run_hsaco
        _installed["aiter.hsaco"] = f"{mod.__name__}.run_hsaco"

    for hook_id in ("aiter.cpp_itfs", "aiter.hsaco"):
        if hook_id in _installed and _should_break(hook_id):
            _break_hooks.add(hook_id)
    return callable(run_lib) and callable(run_hsaco)


def _traced_tensor_call(
    func: Callable, name: str, hook_id: str, named: bool = True
) -> Callable:
    """Wrap a launcher whose arguments carry the operands."""
    layout = _callable_layout(func) if named else None

    @functools.wraps(func)
    def traced(*args, **kwargs):
        if not _enabled:
            return func(*args, **kwargs)
        try:
            if layout is None:
                record = _positional_record(args, kwargs)
            else:
                record = _bind_record(layout, args, kwargs)
        except Exception:  # pragma: no cover - never break the workload  # noqa: BLE001
            return func(*args, **kwargs)
        if record is None:
            return func(*args, **kwargs)
        return _trace_launch(hook_id, name, record, lambda: func(*args, **kwargs))

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
            runner,
            getattr(self, "name", None) or "triton_kernel",
            "triton.compiled",
            named=False,
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
    except Exception:  # noqa: BLE001
        return None


def install_tilelang_kernel_hooks() -> bool:
    """TileLang ``JITKernel.__call__``. Kernel-allocated outputs are not recorded."""
    mod = sys.modules.get("tilelang.jit.kernel")
    cls = getattr(mod, "JITKernel", None) if mod is not None else None
    if not isinstance(cls, type):
        return False
    return _wrap_launch(
        cls,
        "__call__",
        _tilelang_name,
        "tilelang_kernel",
        "tilelang.jit",
        named=False,
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
        except Exception:  # pragma: no cover - never break the workload  # noqa: BLE001
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
