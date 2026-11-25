###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The GPU, its runtime, and its kernel launches, as interpreted source sees them.

Interpreted engine code (:mod:`TraceLens.ModelUtils.source_interpreter`)
expects a GPU. This module gives it the one the trace recorded, built from
the trace's ``deviceProperties`` and process groups, with no GPU present:

- A tensor on the GPU is a ``meta`` tensor: it has a shape, a dtype and
  strides, and no data. A CPU tensor stays a real CPU tensor, so values the
  host code computes are real.
- Every aten op on a GPU tensor runs its meta kernel. While a graph is being
  captured, each op that launches a kernel is recorded with the argument
  record the PyTorch profiler writes for it.
- ``torch.cuda``, ``torch.distributed``, ``torch.library`` and ``triton``
  are models of the runtime: streams, graphs, process groups and custom-op
  registrations live in this run only.
- A module with no source (a compiled extension the source loads by name) is
  a :class:`NativeModule`. Calling one of its functions is a kernel launch.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import os
import re
import logging
import functools
import inspect
import types
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

logger = logging.getLogger(__name__)

_GPU_DEVICE_TYPES = ("cuda", "hip")
_GPU = torch.device("cuda", 0)


@dataclass(frozen=True)
class DeviceFacts:
    """The GPU a trace ran on, as its ``deviceProperties`` record it."""

    name: str = ""
    total_memory: int = 0
    compute_units: int = 0
    major: int = 0
    minor: int = 0
    warp_size: int = 64
    shared_memory_per_block: int = 0
    shared_memory_per_cu: int = 0
    registers_per_block: int = 0
    gfx_target: str = ""
    count: int = 1
    hip_version: str | None = None
    # Device memory held outside the run's tensors (the runtime's context,
    # library workspaces): the trace does not record it.
    reserved_memory: int = 0

    @classmethod
    def from_trace(cls, metadata: dict | None, gfx_target: str = "") -> DeviceFacts:
        devices = [
            device
            for device in (metadata or {}).get("deviceProperties") or ()
            if isinstance(device, dict)
        ]
        first = devices[0] if devices else {}
        hip = (metadata or {}).get("hip_runtime_version")
        return cls(
            name=str(first.get("name", "")),
            total_memory=int(first.get("totalGlobalMem", 0)),
            compute_units=int(first.get("numSms", 0)),
            major=int(first.get("computeMajor", 0)),
            minor=int(first.get("computeMinor", 0)),
            warp_size=int(first.get("warpSize", 64)),
            shared_memory_per_block=int(first.get("sharedMemPerBlock", 0)),
            shared_memory_per_cu=int(first.get("maxSharedMemoryPerMultiProcessor", 0)),
            registers_per_block=int(first.get("regsPerBlock", 0)),
            gfx_target=gfx_target,
            count=max(len(devices), 1),
            hip_version=hip_version_string(hip),
        )


def hip_version_string(encoded) -> str | None:
    """``hipRuntimeGetVersion``'s integer as ``major.minor.patch``.

    HIP encodes ``major * 10_000_000 + minor * 100_000 + patch``.
    """
    try:
        value = int(encoded)
    except (TypeError, ValueError):
        return None
    major, rest = divmod(value, 10_000_000)
    minor, patch = divmod(rest, 100_000)
    return f"{major}.{minor}.{patch}"


@functools.cache
def scalar_type_codes() -> dict[torch.dtype, int]:
    """Each dtype's ``c10::ScalarType`` number, which the profiler records.

    TorchScript treats a dtype as that integer, so a scripted identity
    returns it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        @torch.jit.script
        def code(dtype: torch.dtype) -> int:
            return dtype

    codes = {}
    for dtype in _all_dtypes():
        try:
            codes[dtype] = int(code(dtype))
        except RuntimeError:
            continue
    return codes


@functools.cache
def enum_codes() -> dict:
    """``torch.memory_format`` and ``torch.layout`` values as the profiler's ints.

    Read from the profiler itself: it records each as its c10 enum integer
    in ``Concrete Inputs``.
    """
    from torch.profiler import ProfilerActivity, profile

    codes = {}
    tensor = torch.empty(1)
    for value in list(vars(torch).values()):
        if isinstance(value, torch.memory_format):
            run, position, name = (
                lambda v=value: torch.ops.aten.clone(tensor, memory_format=v),
                1,
                "aten::clone",
            )
        elif isinstance(value, torch.layout) and value is torch.strided:
            run, position, name = (
                lambda v=value: torch.ops.aten.empty.memory_format([1], layout=v),
                2,
                "aten::empty",
            )
        else:
            continue
        try:
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                run()
        except RuntimeError:
            continue
        for event in prof.events():
            concrete = getattr(event, "concrete_inputs", None) or []
            if event.name == name and len(concrete) > position:
                try:
                    codes[value] = int(concrete[position])
                except (TypeError, ValueError):
                    pass
                break
    return codes


def _all_dtypes() -> list[torch.dtype]:
    return [value for value in vars(torch).values() if isinstance(value, torch.dtype)]


@functools.cache
def profiler_type_names() -> dict[torch.dtype, str]:
    """The name the profiler's ``Input type`` gives each dtype.

    Read from the profiler itself: one CPU op on a tensor of each dtype.
    """
    names = {}
    from torch.profiler import ProfilerActivity, profile

    for dtype in _all_dtypes():
        try:
            with warnings.catch_warnings():
                # Experimental and deprecated dtypes announce themselves.
                warnings.simplefilter("ignore", UserWarning)
                tensor = torch.empty(1, dtype=dtype)
        except (RuntimeError, TypeError):
            continue
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            try:
                torch.ops.aten.alias(tensor)
            except RuntimeError:
                continue
        for event in prof.events():
            if event.name == "aten::alias" and event.input_dtypes:
                names[dtype] = event.input_dtypes[0]
                break
    return names


def _format_float(value: float) -> str:
    return "%.17g" % value  # noqa: UP031 - the profiler's own format


# Storages of host-resident tensors with no values: a checkpoint's weights,
# read from disk into host memory. They are meta like GPU tensors, but to the
# source they are on the CPU.
_HOST_STORAGES: set[int] = set()


def _is_host_meta(tensor) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_meta
        and tensor.untyped_storage()._cdata in _HOST_STORAGES
    )


def _mark_host(tensor) -> None:
    """``tensor`` lives in host memory (its storage, and so every view of it)."""
    import weakref

    storage = tensor.untyped_storage()
    key = storage._cdata
    if key not in _HOST_STORAGES:
        _HOST_STORAGES.add(key)
        weakref.finalize(storage, _HOST_STORAGES.discard, key)


def _is_gpu(tensor) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_meta
        and not _is_host_meta(tensor)
    )


@dataclass(frozen=True)
class ArgumentRecord:
    """The profiler's per-argument fields for one call."""

    dims: tuple
    types: tuple[str, ...]
    strides: tuple
    concrete: tuple[str, ...]


def argument_record(values: Iterable[Any]) -> ArgumentRecord:
    """``Input Dims``/``Input type``/``Input Strides``/``Concrete Inputs``."""
    names = profiler_type_names()
    codes = scalar_type_codes()
    dims, kinds, strides, concrete = [], [], [], []
    for value in values:
        if isinstance(value, torch.Tensor):
            dims.append(tuple(value.shape))
            kinds.append(names.get(value.dtype, str(value.dtype)))
            strides.append(tuple(value.stride()))
            concrete.append("")
        elif isinstance(value, _WrappedNumber):
            dims.append(())
            kinds.append(names.get(value.dtype, str(value.dtype)))
            strides.append(())
            concrete.append("")
        elif (
            isinstance(value, list | tuple)
            and value
            and all(isinstance(item, torch.Tensor) for item in value)
        ):
            dims.append(tuple(tuple(item.shape) for item in value))
            kinds.append("TensorList")
            strides.append(tuple(tuple(item.stride()) for item in value))
            concrete.append("")
        elif value is None:
            dims.append(())
            kinds.append("")
            strides.append(())
            concrete.append("")
        else:
            dims.append(())
            strides.append(())
            if isinstance(value, bool):
                kinds.append("Scalar")
                concrete.append(str(value))
            elif isinstance(value, int):
                kinds.append("Scalar")
                concrete.append(str(value))
            elif isinstance(value, float):
                kinds.append("Scalar")
                concrete.append(_format_float(value))
            elif isinstance(value, torch.dtype):
                kinds.append("Scalar")
                concrete.append(str(codes.get(value, "")))
            elif isinstance(value, torch.memory_format | torch.layout):
                kinds.append("Scalar")
                concrete.append(str(enum_codes().get(value, "")))
            elif isinstance(value, list | tuple) and all(
                isinstance(item, int | float | bool) for item in value
            ):
                kinds.append("ScalarList")
                concrete.append(
                    "["
                    + ", ".join(
                        _format_float(item) if isinstance(item, float) else str(item)
                        for item in value
                    )
                    + "]"
                )
            else:
                # The profiler records neither type nor value of anything
                # else, strings included.
                kinds.append("")
                concrete.append("")
    return ArgumentRecord(tuple(dims), tuple(kinds), tuple(strides), tuple(concrete))


@dataclass(frozen=True)
class _CallerRecord:
    """An interpreted call a launch was made within, with what it was passed."""

    name: str
    record: ArgumentRecord
    # Positions of the arguments passed as None.
    nones: tuple[int, ...] = ()
    # Names of the trailing arguments passed by keyword.
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecordedLaunch:
    """One kernel-launching call made while a graph was being captured."""

    op_name: str
    record: ArgumentRecord
    kernels: tuple[str, ...] = ()
    scope: tuple[str, ...] = ()
    # The interpreted calls it was made within that were handed tensors,
    # innermost first.
    callers: tuple[_CallerRecord, ...] = ()
    # Made by the same native call as the launch before it.
    same_call: bool = False
    # The settings (environment variables, ``--options``) that decided the
    # branches it was made within, innermost last; and those its innermost
    # interpreted caller's arguments were computed from.
    control: tuple[str, ...] = ()
    shaped_by: tuple[str, ...] = ()


@dataclass
class CapturedGraph:
    """One graph capture: where it began and the launches made inside it."""

    label: str
    launches: list[RecordedLaunch] = field(default_factory=list)
    # The calls the capture began within, outermost first, with the integers
    # each was passed (a batch size, a token count).
    context: tuple[tuple[str, tuple[int, ...]], ...] = ()


@dataclass
class LaunchLog:
    """Every graph captured during the run, each with its launches in order."""

    graphs: list[CapturedGraph] = field(default_factory=list)
    capturing: int = 0
    scope: list[str] = field(default_factory=list)
    # Describes where a capture begins (the interpreted call stack).
    label: Callable[[], str] = lambda: ""
    shadows: Shadows | None = None
    # The innermost interpreted call holding tensors: (name, arguments).
    caller: Callable[[], tuple | None] | None = None
    # Every such call, innermost first: [(name, arguments passed)].
    callers: Callable[[], list] | None = None
    # The calls on the stack, outermost first: [(name, integers passed)].
    context: Callable[[], list] | None = None
    # The settings deciding the branches being run, and those the innermost
    # call's arguments came from (see RecordedLaunch).
    control: Callable[[], tuple] | None = None
    shaping: Callable[[], tuple] | None = None
    # The launches made outside any capture since the last one began (the
    # warm-up forward a capture follows), the latest at most EAGER_KEPT.
    eager: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=LaunchLog.EAGER_KEPT)
    )
    EAGER_KEPT: ClassVar[int] = 20000

    @property
    def launches(self) -> list[RecordedLaunch]:
        return self.graphs[-1].launches if self.graphs else []

    def begin_capture(self):
        if not self.capturing:
            context = tuple(self.context()) if self.context is not None else ()
            self.graphs.append(CapturedGraph(self.label(), context=context))
            self.eager.clear()
        self.capturing += 1

    def end_capture(self):
        self.capturing -= 1

    def record(
        self, op_name, values, kernels=(), native=True, written=None, same_call=False
    ):
        """A launch with ``values`` as arguments.

        A native kernel may write any tensor it is handed, so their shadow
        copies no longer hold; ``written`` narrows that to what it writes.
        """
        if native and self.shadows is not None:
            # What an enclosing op declares it only reads stays as it was.
            kept = {id(tensor) for reads in self._scope_reads for tensor in reads}
            for value in values if written is None else written:
                items = value if isinstance(value, list | tuple) else (value,)
                for item in items:
                    if isinstance(item, torch.Tensor) and id(item) not in kept:
                        self.shadows.drop(item)
        if self.capturing or self.control is not None:
            # Outside a capture only what says where a stopped start was.
            callers = (
                self.callers() if self.callers is not None and self.capturing else ()
            )
            (self.graphs[-1].launches if self.capturing else self.eager).append(
                RecordedLaunch(
                    op_name,
                    argument_record(values),
                    tuple(kernels),
                    tuple(self.scope),
                    tuple(
                        _CallerRecord(
                            name,
                            argument_record(passed),
                            tuple(i for i, value in enumerate(passed) if value is None),
                            keywords,
                        )
                        for name, passed, keywords in callers
                    ),
                    same_call,
                    self.control() if self.control is not None else (),
                    self.shaping() if self.shaping is not None else (),
                )
            )

    @contextlib.contextmanager
    def op(self, name, values=None, reads=()):
        """An op scope: a launch made inside it is a launch of this op.

        ``values`` are the op's own arguments, which is what the profiler
        records for it; a kernel launched within takes them. ``reads`` are
        the tensors its schema declares it does not write.
        """
        self.scope.append(name)
        self._scope_values.append(values)
        self._scope_reads.append(tuple(reads))
        try:
            yield
        finally:
            self.scope.pop()
            self._scope_values.pop()
            self._scope_reads.pop()

    def innermost(self, default):
        return self.scope[-1] if self.scope else default

    def innermost_values(self, default):
        if self._scope_values and self._scope_values[-1] is not None:
            return self._scope_values[-1]
        return default

    def __post_init__(self):
        self._scope_values: list = []
        self._scope_reads: list = []


# --------------------------------------------------------------- dispatch


class _Unrecorded:
    """An argument the profiler records neither the type nor the value of."""

    def __init__(self, value):
        self.value = value


class _WrappedNumber:
    """A Python number passed for a ``Tensor`` argument: the dispatcher's
    0-dim tensor of the number's own type."""

    def __init__(self, value):
        self.value = value
        self.dtype = (
            torch.bool
            if isinstance(value, bool)
            else (
                torch.int64
                if isinstance(value, int)
                else torch.complex128 if isinstance(value, complex) else torch.float64
            )
        )


def _tensor_typed(argument) -> bool:
    kind = argument.type
    if isinstance(kind, torch.OptionalType):
        kind = kind.getElementType()
    return isinstance(kind, torch.TensorType)


def _profiled_as_other(argument) -> bool:
    # A ``Tensor?[]`` is a generic list to the profiler, unlike a ``Tensor[]``.
    kind = argument.type
    return isinstance(kind, torch.ListType) and isinstance(
        kind.getElementType(), torch.OptionalType
    )


def _schema_values(func, args, kwargs) -> list:
    """Every schema argument, defaults filled in, in schema order."""
    schema = func._schema
    values = []
    positional = list(args)
    for index, argument in enumerate(schema.arguments):
        if argument.kwarg_only:
            if argument.name in kwargs:
                value = kwargs[argument.name]
            else:
                value = argument.default_value
        elif index < len(positional):
            value = positional[index]
        elif argument.name in kwargs:
            value = kwargs[argument.name]
        else:
            value = argument.default_value
        if _profiled_as_other(argument):
            value = _Unrecorded(value)
        elif isinstance(value, int | float | complex) and _tensor_typed(argument):
            value = _WrappedNumber(value)
        values.append(value)
    return values


# Factories whose tensors hold whatever the allocator left there.
_UNDEFINED_CONTENTS = frozenset(
    {"empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided"}
)

_NO_KERNEL = frozenset(
    {
        "empty",
        "empty_like",
        "empty_strided",
        "new_empty",
        "new_empty_strided",
        "resize_",
        "set_",
        "detach",
        "lift_fresh",
        "_local_scalar_dense",
        "sym_size",
        "sym_stride",
        "sym_numel",
        "sym_storage_offset",
    }
)


def launches_kernel(func, args=(), result=None) -> bool:
    """Whether an aten op runs a kernel, or only builds a tensor's metadata.

    An op that writes an argument (``Tensor(a!)``) runs a kernel. Otherwise a
    view returns an alias of an input: its schema says so, or its output
    shares an input's storage. An allocation only reserves memory.
    """
    schema = func._schema
    if schema.name.split("::")[-1] in _NO_KERNEL:
        return False
    if any(
        argument.alias_info is not None and argument.alias_info.is_write
        for argument in schema.arguments
    ):
        return True
    for returned in schema.returns:
        info = returned.alias_info
        if info is not None and not info.is_write:
            return False
    inputs = {_storage_of(value) for value in args if isinstance(value, torch.Tensor)}
    outputs = result if isinstance(result, tuple | list) else (result,)
    return not any(
        isinstance(out, torch.Tensor) and _storage_of(out) in inputs for out in outputs
    )


def _storage_of(tensor) -> int:
    return tensor.untyped_storage()._cdata


class _UnknownContents(RuntimeError):
    """Deferred contents turned out not to be known when read."""


class Shadows:
    """CPU copies of the contents of small GPU tensors.

    A ``meta`` tensor has no values, yet host code reads small GPU buffers
    (a length, a fill value, a sum) to decide what it does. A GPU tensor of
    at most :attr:`LIMIT` elements keeps a real CPU tensor with its contents:
    a factory makes it, every op recomputes it on the CPU from its inputs'
    copies, and a view shares its base's copy so an in-place write reaches
    both. A write whose values are unknown (a native kernel, an input without
    a copy) drops it.

    Contents may also be lazy, computed only when something reads them (a
    checkpoint tensor fetched on demand): a tensor may have a function making
    its copy, and a storage a log of the writes made into it, replayed from
    zeros. An op on lazy contents makes its outputs lazy in turn; it keeps
    readers of its inputs' contents, never the tensors themselves, so it
    holds no device memory alive.
    """

    LIMIT = 1 << 25

    def __init__(self):
        # id(tensor) -> (weak reference, copy). Keyed by identity: tensors
        # compare elementwise, so they cannot be dictionary keys themselves.
        self._copies: dict[int, tuple] = {}
        # id(tensor) -> (weak reference, function making its copy)
        self._lazy: dict[int, tuple] = {}
        # storage -> (bytes, [((size, stride, offset, dtype), write)])
        self._logs: dict[int, tuple[int, list]] = {}
        # Storages an unknown write reached.
        self._unknown: set[int] = set()
        self._tracked: set[int] = set()
        # storage -> {geometry: copy}: every tensor over a storage (a view, a
        # Parameter wrapping it, ``.data``) finds the copy of its geometry.
        self._by_storage: dict[int, dict[tuple, torch.Tensor]] = {}
        # storage -> ids of the tensors over it with a copy or a lazy copy.
        self._ids: dict[int, set[int]] = {}

    def _peek(self, tensor):
        entry = self._copies.get(id(tensor))
        if entry is not None and entry[0]() is tensor:
            return entry[1]
        held = self._by_storage.get(self._storage_key(tensor))
        if held:
            return held.get(self._geometry(tensor))
        return None

    def _pending(self, tensor):
        entry = self._lazy.get(id(tensor))
        if entry is None or entry[0]() is not tensor:
            return None
        return entry[1]

    @staticmethod
    def _storage_key(tensor) -> int:
        return tensor.untyped_storage()._cdata

    @staticmethod
    def _geometry(tensor) -> tuple:
        return (
            tuple(tensor.size()),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.dtype,
        )

    def _logged(self, tensor):
        return self._logs.get(self._storage_key(tensor))

    def get(self, tensor):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_meta:
            return None
        copy = self._peek(tensor)
        if copy is None:
            make = self._pending(tensor)
            if make is not None:
                self._lazy.pop(id(tensor), None)
                try:
                    copy = make()
                except _UnknownContents:
                    self.drop(tensor)
                    return None
                self.set(tensor, copy)
            elif self._logged(tensor) is not None:
                try:
                    return self._reader(tensor)()
                except _UnknownContents:
                    # A deferred write left them unknown.
                    self.drop(tensor)
                    return None
        return copy

    def set_lazy(self, tensor, make) -> None:
        """``tensor``'s copy is ``make()``, made when it is first read."""
        import weakref

        if isinstance(tensor, torch.Tensor) and tensor.is_meta:
            key = id(tensor)
            lazy = self._lazy
            self._lazy[key] = (
                weakref.ref(tensor, lambda _ref, key=key: lazy.pop(key, None)),
                make,
            )
            self._ids.setdefault(self._track(tensor), set()).add(key)

    def set(self, tensor, value):
        import weakref

        if isinstance(tensor, torch.Tensor) and tensor.is_meta:
            key = id(tensor)
            copies = self._copies
            self._copies[key] = (
                weakref.ref(tensor, lambda _ref, key=key: copies.pop(key, None)),
                value,
            )
            storage = self._track(tensor)
            self._by_storage.setdefault(storage, {})[self._geometry(tensor)] = value
            self._ids.setdefault(storage, set()).add(key)

    def drop(self, tensor):
        """``tensor``'s contents are unknown now: an unknown write reached them."""
        if isinstance(tensor, torch.Tensor) and tensor.is_meta:
            key = self._track(tensor)
            self._logs.pop(key, None)
            # Any tensor over the storage may overlap what was written.
            self._by_storage.pop(key, None)
            self._unknown.add(key)
        while isinstance(tensor, torch.Tensor):
            self._copies.pop(id(tensor), None)
            self._lazy.pop(id(tensor), None)
            tensor = tensor._base

    def defer_write(self, tensor, write) -> None:
        """``tensor``, all of its storage, is written by ``write(region)`` when read.

        What was known of the storage is replaced: every copy of it (by any
        tensor over it) goes, and its contents are the write's.
        """
        key = self._track(tensor)
        self._logs[key] = (
            tensor.untyped_storage().nbytes(),
            [(self._geometry(tensor), write)],
        )
        self._by_storage.pop(key, None)
        self._unknown.discard(key)
        for entry_key in self._ids.pop(key, ()):
            for table in (self._copies, self._lazy):
                entry = table.get(entry_key)
                if entry is not None:
                    other = entry[0]()
                    if other is None or self._storage_key(other) == key:
                        table.pop(entry_key, None)

    def forget(self, tensor) -> None:
        """``tensor`` is replaced wholesale: what was known of it no longer applies."""
        self._copies.pop(id(tensor), None)
        self._lazy.pop(id(tensor), None)
        held = self._by_storage.get(self._storage_key(tensor))
        if held:
            held.pop(self._geometry(tensor), None)

    def _track(self, tensor) -> int:
        """``tensor``'s storage key; what is kept for it goes with the storage."""
        import weakref

        storage = tensor.untyped_storage()
        key = storage._cdata
        if key not in self._tracked:
            self._tracked.add(key)

            def release(
                key=key,
                logs=self._logs,
                unknown=self._unknown,
                tracked=self._tracked,
                by_storage=self._by_storage,
                ids=self._ids,
            ):
                logs.pop(key, None)
                unknown.discard(key)
                tracked.discard(key)
                by_storage.pop(key, None)
                ids.pop(key, None)

            weakref.finalize(storage, release)
        return key

    def _reader(self, value):
        """A function giving ``value``'s contents on the CPU, or None if unknown.

        It holds snapshots and lazy functions, not the tensor.
        """
        if isinstance(value, torch.Tensor) and value.is_meta:
            copy = self._peek(value)
            if copy is not None:
                snapshot = copy.clone()
                return lambda: snapshot
            make = self._pending(value)
            if make is not None:
                return make
            logged = self._logged(value)
            if logged is not None:
                nbytes, entries = logged
                entries = list(entries)
                geometry = self._geometry(value)
                return lambda: _replay(nbytes, entries, geometry)
            return None
        if isinstance(value, list | tuple):
            readers = [self._reader(item) for item in value]
            if any(reader is None for reader in readers):
                return None
            kind = type(value)
            return lambda: kind(reader() for reader in readers)
        if isinstance(value, torch.device) and value.type == "meta":
            return lambda: torch.device("cpu")
        if isinstance(value, str) and value == "meta":
            return lambda: "cpu"
        return lambda: value

    def follow(self, func, args, kwargs, result) -> None:
        """Recompute the copies ``func`` affects, on the CPU."""
        outputs = result if isinstance(result, tuple | list) else (result,)
        if func._schema.name.split("::")[-1] in _UNDEFINED_CONTENTS:
            # Allocated, not written: there are no values to copy yet.
            return
        written = _written_arguments(func, args, kwargs)
        written_ids = {id(tensor) for tensor in written}
        small = all(
            not isinstance(out, torch.Tensor)
            or not out.is_meta
            or out.numel() <= self.LIMIT
            for out in outputs
        )
        missing = False
        lazy = False

        def substitute(value):
            nonlocal missing, lazy
            if isinstance(value, torch.Tensor) and value.is_meta:
                copy = self._peek(value)
                if copy is None:
                    if (
                        self._pending(value) is not None
                        or self._logged(value) is not None
                    ):
                        lazy = True
                    elif (
                        id(value) in written_ids
                        and self._storage_key(value) not in self._unknown
                    ):
                        # Written before anything was known of it: what is
                        # written is all there will be.
                        lazy = True
                    else:
                        missing = True
                return copy
            if isinstance(value, list | tuple):
                return type(value)(substitute(item) for item in value)
            if isinstance(value, torch.device) and value.type == "meta":
                return torch.device("cpu")
            if isinstance(value, str) and value == "meta":
                return "cpu"
            return value

        cpu_args = substitute(list(args))
        cpu_kwargs = {key: substitute(value) for key, value in kwargs.items()}
        if missing or not small:
            for target in written:
                self.drop(target)
            return
        if lazy:
            self._follow_lazily(func, args, kwargs, outputs, written)
            return
        try:
            cpu_result = func(*cpu_args, **cpu_kwargs)
        except (RuntimeError, NotImplementedError, TypeError, ValueError, IndexError):
            for target in written:
                self.drop(target)
            return
        cpu_outputs = (
            cpu_result if isinstance(cpu_result, tuple | list) else (cpu_result,)
        )
        for out, cpu_out in zip(outputs, cpu_outputs, strict=False):
            if isinstance(out, torch.Tensor) and isinstance(cpu_out, torch.Tensor):
                self.set(out, cpu_out)

    def _follow_lazily(self, func, args, kwargs, outputs, written) -> None:
        """An op on lazy contents: its results are lazy too.

        A new output gets a function making its copy. A write into a tensor
        is logged against its storage, to be replayed when read; into a
        tensor with a known copy, it is applied now.
        """
        written_ids = {id(tensor) for tensor in written}

        def slot(value):
            """``(written tensor id, None)`` or ``(None, reader)``: never the tensor."""
            if isinstance(value, torch.Tensor) and id(value) in written_ids:
                return id(value), None
            return None, self._reader(value)

        arg_slots = [slot(value) for value in args]
        kwarg_slots = {key: slot(value) for key, value in kwargs.items()}
        if any(
            target is None and found is None
            for target, found in (*arg_slots, *kwarg_slots.values())
        ):
            for target in written:
                self.drop(target)
            return

        def call(targets):
            """Run ``func`` on the CPU, the written tensors replaced by ``targets``."""
            call_args = [
                targets[target] if target is not None else found()
                for target, found in arg_slots
            ]
            call_kwargs = {
                key: targets[target] if target is not None else found()
                for key, (target, found) in kwarg_slots.items()
            }
            result = func(*call_args, **call_kwargs)
            return result if isinstance(result, tuple | list) else (result,)

        for target in written:
            copy = self._peek(target)
            if copy is not None:
                try:
                    call({id(target): copy})
                except (
                    RuntimeError,
                    NotImplementedError,
                    TypeError,
                    ValueError,
                    IndexError,
                ):
                    self.drop(target)
                continue
            key = self._track(target)
            nbytes = target.untyped_storage().nbytes()
            entries = self._logs.setdefault(key, (nbytes, []))[1]
            self.forget(target)
            target_id = id(target)
            entries.append(
                (
                    self._geometry(target),
                    lambda region, target_id=target_id: call({target_id: region}),
                )
            )
            self._unknown.discard(key)
        computed = []

        def compute():
            if not computed:
                computed.append(call({}))
            return computed[0]

        for index, out in enumerate(outputs):
            if (
                isinstance(out, torch.Tensor)
                and out.is_meta
                and id(out) not in written_ids
            ):
                self.set_lazy(out, lambda index=index: compute()[index])


def _view_bytes(buffer, geometry):
    """The tensor with ``geometry`` over a byte buffer."""
    size, stride, offset, dtype = geometry
    return torch.empty(0, dtype=dtype).set_(
        buffer.untyped_storage(), offset, size, stride
    )


def _replay(nbytes, entries, geometry):
    """A storage's logged writes, replayed from zeros; the part ``geometry`` covers."""
    buffer = torch.zeros(nbytes, dtype=torch.uint8)
    for where, write in entries:
        region = _view_bytes(buffer, where)
        written = write(region.clone())
        region.copy_(written[0] if isinstance(written, tuple | list) else written)
    return _view_bytes(buffer, geometry).clone()


def _written_arguments(func, args, kwargs) -> list:
    """The tensor arguments an op writes (``Tensor(a!)`` in its schema)."""
    written = []
    for index, argument in enumerate(func._schema.arguments):
        info = argument.alias_info
        if info is None or not info.is_write:
            continue
        value = (
            args[index]
            if index < len(args) and not argument.kwarg_only
            else kwargs.get(argument.name)
        )
        if isinstance(value, torch.Tensor):
            written.append(value)
    return written


# Tensor methods that hand GPU contents to the host.
_READS_VALUES = (
    torch.Tensor.item,
    torch.Tensor.tolist,
    torch.Tensor.__bool__,
    torch.Tensor.__int__,
    torch.Tensor.__float__,
    torch.Tensor.__index__,
    torch.Tensor.numpy,
)


class DeviceMemory:
    """The GPU memory the run's live tensors hold, as the allocator sees it.

    A storage is held from the op that makes it until its last tensor goes;
    the caching allocator rounds each block up to :attr:`BLOCK` bytes.
    ``torch.cuda.mem_get_info`` reports the device's memory less this.
    """

    BLOCK = 512
    # Where the device's addresses start, as a HIP/CUDA process sees them.
    BASE = 0x7F0000000000

    def __init__(self):
        # storage address -> bytes held
        self._held: dict[int, int] = {}
        # storage -> its device address; a bump allocator never reuses one.
        self._addresses: dict[int, int] = {}
        self._next = self.BASE
        self.allocated = 0
        self.peak = 0
        # Bytes allocated outside the caching allocator (``hipMalloc``), by address.
        self._raw: dict[int, int] = {}
        self.raw = 0

    def allocate_raw(self, nbytes: int) -> int:
        """A ``hipMalloc`` of ``nbytes``: the device address it returns."""
        address = self._next
        size = max(-(-nbytes // self.BLOCK), 1) * self.BLOCK
        self._next += size
        self._raw[address] = size
        self.raw += size
        return address

    def free_raw(self, address: int) -> None:
        self.raw -= self._raw.pop(address, 0)

    def address(self, tensor) -> int:
        """``tensor.data_ptr()``: its storage's device address plus its offset."""
        storage = tensor.untyped_storage()
        key = storage._cdata
        base = self._addresses.get(key)
        if base is None:
            base = self._addresses[key] = self._next
            self._next += max(-(-storage.nbytes() // self.BLOCK), 1) * self.BLOCK
        return base + tensor.storage_offset() * tensor.element_size()

    def follow(self, result) -> None:
        import weakref

        outputs = result if isinstance(result, tuple | list) else (result,)
        for out in outputs:
            if not _is_gpu(out):
                continue
            storage = out.untyped_storage()
            key = storage._cdata
            size = -(-storage.nbytes() // self.BLOCK) * self.BLOCK
            held = self._held.get(key)
            if held is None:
                # The storage object lives exactly as long as the storage.
                weakref.finalize(storage, self._release, key)
            elif held == size:
                continue
            self._held[key] = size
            self.allocated += size - (held or 0)
            self.peak = max(self.peak, self.allocated)

    def _release(self, key) -> None:
        self.allocated -= self._held.pop(key, 0)
        self._addresses.pop(key, None)

    def forget(self, tensor) -> None:
        """``tensor``'s storage is not device memory after all."""
        self._release(tensor.untyped_storage()._cdata)


class _GpuAsMeta(TorchFunctionMode):
    """Places every tensor bound for the GPU on the ``meta`` device.

    Reading a GPU tensor's values on the host (``.item()``, ``.cpu()``)
    reads its shadow copy.
    """

    def __init__(self, shadows: Shadows, memory: DeviceMemory):
        super().__init__()
        self.shadows = shadows
        self.memory = memory

    def __torch_function__(self, func, types_, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        # A GPU tensor is meta underneath; to the source it is on the GPU.
        owner = getattr(func, "__self__", None)
        if args and _is_host_meta(args[0]):
            handled, value = self._host_tensor(func, owner, args, kwargs)
            if handled:
                return value
        args, kwargs = self._host_operands(args, kwargs)
        if args and _is_gpu(args[0]):
            if owner is torch.Tensor.is_cuda:
                return True
            if owner is torch.Tensor.device:
                return _GPU
            if func is torch.Tensor.get_device:
                return 0
            if func is torch.Tensor.data_ptr:
                return self.memory.address(args[0])
        device = kwargs.get("device")
        if func in (torch.tensor, torch.as_tensor) and _names_gpu(device):
            # Built from data: make it on the CPU, then move it, so the move
            # (a dispatched copy) carries the data into the shadow.
            return func(*args, **{**kwargs, "device": "cpu"}).to("meta")
        if _names_gpu(device):
            kwargs["device"] = "meta"
        if func in (torch.Tensor.cuda,):
            return args[0].to("meta")
        if func in (torch.Tensor.to,) and args:
            args = tuple("meta" if _names_gpu(arg) else arg for arg in args)
        if func in (torch.Tensor.pin_memory, torch.Tensor.is_pinned):
            return args[0] if func is torch.Tensor.pin_memory else True
        if args and _is_gpu(args[0]):
            to_host = func is torch.Tensor.cpu or (
                func is torch.Tensor.to
                and any(
                    (isinstance(arg, str) and arg == "cpu")
                    or (isinstance(arg, torch.device) and arg.type == "cpu")
                    for arg in (*args[1:], kwargs.get("device"))
                )
            )
            if func in _READS_VALUES or to_host:
                copy = self.shadows.get(args[0])
                if copy is None:
                    raise RuntimeError(
                        "the run reads GPU values its source does not determine "
                        f"({getattr(func, '__name__', func)} of a "
                        f"{tuple(args[0].shape)} {args[0].dtype} tensor)"
                    )
                if to_host:
                    return copy.clone()
                return func(copy, *args[1:], **kwargs)
        return func(*args, **kwargs)

    def _host_tensor(self, func, owner, args, kwargs):
        tensor = args[0]
        if owner is torch.Tensor.is_cuda:
            return True, False
        if owner is getattr(torch.Tensor, "is_cpu", None):
            return True, True
        if owner is torch.Tensor.device:
            return True, torch.device("cpu")
        if func is torch.Tensor.get_device:
            return True, -1
        if func in (torch.Tensor.pin_memory,):
            return True, tensor
        if func is torch.Tensor.is_pinned:
            return True, False
        targets = (*args[1:], kwargs.get("device"))
        to_gpu = func is torch.Tensor.cuda or (
            func is torch.Tensor.to and any(_names_gpu(target) for target in targets)
        )
        to_cpu = func is torch.Tensor.cpu or (
            func is torch.Tensor.to
            and any(
                (isinstance(target, str) and target == "cpu")
                or (isinstance(target, torch.device) and target.type == "cpu")
                for target in targets
            )
        )
        if to_gpu:
            # Uploaded: a new device tensor with the same contents.
            moved = (
                func(
                    *(("meta" if _names_gpu(a) else a) for a in args),
                    **{
                        k: ("meta" if k == "device" and _names_gpu(v) else v)
                        for k, v in kwargs.items()
                    },
                )
                if func is torch.Tensor.to
                else tensor
            )
            moved = moved.clone() if moved is tensor else moved
            _HOST_STORAGES.discard(moved.untyped_storage()._cdata)
            self.memory.follow(moved)
            return True, moved
        if to_cpu and func is torch.Tensor.cpu:
            return True, tensor
        if func in _READS_VALUES:
            copy = self.shadows.get(tensor)
            if copy is None:
                raise RuntimeError(
                    "the run reads host values its source does not determine "
                    f"({getattr(func, '__name__', func)} of a "
                    f"{tuple(tensor.shape)} {tensor.dtype} tensor)"
                )
            return True, func(copy, *args[1:], **kwargs)
        return False, None

    def _host_operands(self, args, kwargs):
        """With a host-resident meta tensor, real CPU operands join it on meta."""
        values = (*args, *kwargs.values())
        flat = [
            item
            for value in values
            for item in (value if isinstance(value, list | tuple) else (value,))
        ]
        if not any(_is_host_meta(item) for item in flat):
            return args, kwargs
        if not any(
            isinstance(item, torch.Tensor) and not item.is_meta for item in flat
        ):
            return args, kwargs

        def lift(value):
            if isinstance(value, torch.Tensor) and not value.is_meta:
                lifted = value.to("meta")
                self.shadows.set(lifted, value)
                _mark_host(lifted)
                return lifted
            if isinstance(value, list | tuple):
                return type(value)(lift(item) for item in value)
            return value

        return tuple(lift(arg) for arg in args), {
            key: lift(value) for key, value in kwargs.items()
        }


def _names_gpu(device) -> bool:
    if isinstance(device, torch.device):
        return device.type in _GPU_DEVICE_TYPES
    if isinstance(device, str):
        return device.split(":")[0] in _GPU_DEVICE_TYPES
    if isinstance(device, int) and not isinstance(device, bool):
        return True
    return False


def _host_results(args, kwargs, result) -> None:
    """An op on host-resident tensors alone makes host-resident tensors."""
    inputs = [
        item
        for value in (*args, *kwargs.values())
        for item in (value if isinstance(value, list | tuple) else (value,))
        if isinstance(item, torch.Tensor) and item.is_meta
    ]
    if not inputs or not all(_is_host_meta(item) for item in inputs):
        return
    for out in result if isinstance(result, tuple | list) else (result,):
        if isinstance(out, torch.Tensor) and out.is_meta:
            _mark_host(out)


class _Recorder(TorchDispatchMode):
    """Records each kernel-launching aten op on a GPU tensor."""

    def __init__(self, log: LaunchLog, shadows: Shadows, memory: DeviceMemory):
        super().__init__()
        self.log = log
        self.shadows = shadows
        self.memory = memory

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        _host_results(args, kwargs, result)
        self.shadows.follow(func, args, kwargs, result)
        self.memory.follow(result)
        if self.log.capturing and launches_kernel(func, args, result):
            values = _schema_values(func, args, kwargs)
            touches_gpu = any(
                _is_gpu(value)
                or (
                    isinstance(value, list | tuple)
                    and any(_is_gpu(item) for item in value)
                )
                for value in values
            ) or _is_gpu(result)
            if touches_gpu:
                name = func._schema.name
                for conversion in _index_conversions(func, args):
                    self.log.record("aten::copy_", conversion, native=False)
                self.log.record(name, values, native=False)
        return result


# Ops that index by tensors: on the GPU each converts an index of a narrower
# integer type to int64 first, a ``copy_`` of its own within the op.
_INDEXING_OPS = frozenset(
    {"aten::index", "aten::index_put_", "aten::_index_put_impl_", "aten::index_put"}
)
_NARROW_INDEX_DTYPES = frozenset({torch.int32, torch.int16, torch.int8})


def _index_conversions(func, args) -> list[list]:
    """The ``copy_`` launches an indexing op makes of its narrow integer indices.

    The indices are broadcast together, and each narrower than int64 is
    copied to an int64 tensor of that shape (``to`` -> ``_to_copy`` ->
    ``copy_``) before the op's own kernel.
    """
    if (
        func._schema.name not in _INDEXING_OPS
        or len(args) < 2
        or not isinstance(args[1], list | tuple)
    ):
        return []
    indices = [index for index in args[1] if isinstance(index, torch.Tensor)]
    narrow = [
        index
        for index in indices
        if index.dtype in _NARROW_INDEX_DTYPES and _is_gpu(index)
    ]
    if not narrow:
        return []
    try:
        shape = torch.broadcast_shapes(*(index.shape for index in indices))
    except RuntimeError:
        return []
    copies = [
        [
            torch.empty(shape, dtype=torch.int64, device="meta"),
            index.expand(shape),
            False,
        ]
        for index in narrow
    ]
    # With them, an int64 index whose broadcast is a stretched view
    # (``rows[:, None]`` against ``(n, k)`` offsets) is made whole too.
    for index in indices:
        if (
            index.dtype == torch.int64
            and _is_gpu(index)
            and not index.expand(shape).is_contiguous()
        ):
            copies.append(
                [
                    torch.empty(shape, dtype=torch.int64, device="meta"),
                    index.expand(shape),
                    False,
                ]
            )
    return copies


# ---------------------------------------------------------- native modules


class NativeFunction:
    """A function of a compiled extension.

    Called with a tensor, it is a kernel launch. Called with scalars only, it
    is host computation: its C/C++ body is evaluated when the source states
    it in a form :mod:`TraceLens.ModelUtils.native_host` reads, and its
    value is otherwise unknown.
    """

    def __init__(self, module: NativeModule, name: str):
        self._module = module
        self.__name__ = name
        self.__qualname__ = name
        self.__module__ = module.__name__
        self.__doc__ = module.signature_doc(name)
        # ctypes declares these on a loaded symbol.
        self.argtypes = None
        self.restype = _C_INT_DEFAULT

    def __call__(self, *args, **kwargs):
        values = list(args) + list(kwargs.values())
        module = self._module
        fact = (module.machine_functions or {}).get(self.__name__)
        if fact is not None:
            return fact(*args)
        host_only = not any(
            _holds_tensor(value) or (module.c_abi and _device_pointer(value))
            for value in values
        )
        if host_only:
            host = self._module.host
            if host is not None:
                try:
                    return host(self.__name__, tuple(_scalar(value) for value in args))
                except LookupError:
                    pass
            if module.c_abi and not module.kernels_of(self.__name__):
                # A C call handed only host memory: host code, with its status.
                return _c_result(self.restype)
            raw = module.raw_memory_call(self.__name__, args)
            if raw is not None:
                return raw
            if not values or self._module.declares_host_only(self.__name__):
                return NATIVE_RESULT
            scalar = self._module.scalar_return(self.__name__)
            if scalar is not None:
                # Handed no tensor and returning a number: host bookkeeping.
                return scalar
        log = self._module.log
        name = log.innermost(None)
        recorded = log.innermost_values(None)
        if name is None:
            # No op scope: a kernel called through raw pointers. The call is
            # the interpreted function that handed over the tensors.
            caller = log.caller() if log.caller is not None else None
            name, recorded = caller or (
                f"{self._module.__name__}::{self.__name__}",
                values,
            )
        # One launch per kernel the host function launches; one that launches
        # through the functions it calls takes the kernels it may launch, for
        # the replay's kernel names to choose among.
        launched = self._module.kernels_of(self.__name__)
        sequence = [(kernel,) for kernel in launched or ()] or [
            self._module.kernel_alternatives(self.__name__)
        ]
        for position, kernels in enumerate(sequence):
            log.record(
                name,
                recorded if recorded is not None else values,
                kernels,
                same_call=position > 0,
            )
        if module.c_abi:
            return _c_result(self.restype)
        allocated = module.tensor_result(self.__name__, args)
        return allocated if allocated is not None else NATIVE_RESULT

    def __repr__(self):
        return f"<native {self._module.__name__}.{self.__name__}>"


def _native_class(name: str, module: str) -> type:
    """A class a native module binds: it holds what it is constructed with."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.__dict__.update(kwargs)

    return type(
        name,
        (),
        {"__init__": __init__, "__module__": module, "__native_class__": True},
    )


def _holds_tensor(value) -> bool:
    """A tensor, or a native struct a binding takes in a tensor's place."""
    if isinstance(value, torch.Tensor) or getattr(value, "__native_class__", False):
        return True
    if isinstance(value, list | tuple):
        return any(_holds_tensor(item) for item in value)
    return False


def _scalar(value):
    """A ctypes scalar (``c_int(4)``) as the Python value it wraps."""
    return getattr(value, "value", value)


# ctypes gives an undeclared result type ``c_int``.
_C_INT_DEFAULT = object()


def _device_pointer(value) -> bool:
    """An address in device memory, as the run's allocator hands them out, or a
    pointer to a C structure holding one (a tensor descriptor a C ABI takes)."""
    import ctypes

    address = getattr(value, "value", value)
    if isinstance(address, int) and not isinstance(address, bool):
        return DeviceMemory.BASE <= address < DeviceMemory.BASE + (1 << 44)
    target = getattr(value, "_obj", None)
    if target is None and isinstance(value, ctypes._Pointer) and value:
        target = value.contents
    if isinstance(target, ctypes.Structure):
        return any(
            _device_pointer(getattr(target, field[0])) for field in target._fields_
        )
    return False


def _c_result(restype):
    """What a C function returns when it succeeds, by its declared type.

    A status code is ``0`` (success); a string is empty; a pointer is NULL.
    """
    import ctypes

    if restype is _C_INT_DEFAULT:
        return 0
    if restype is None:
        return None
    if restype in (ctypes.c_char_p,):
        return b""
    if restype in (ctypes.c_wchar_p,):
        return ""
    if restype in (ctypes.c_void_p,):
        return None
    if restype in (ctypes.c_float, ctypes.c_double, ctypes.c_longdouble):
        return 0.0
    if restype in (ctypes.c_bool,):
        return False
    if isinstance(restype, type) and issubclass(restype, ctypes._SimpleCData):
        return 0
    if isinstance(restype, type):
        try:
            return restype()
        except TypeError:
            return None
    return 0


class _NativeResult:
    """What a native function returns: its outputs are computed elsewhere."""

    def __repr__(self):
        return "<native result>"

    def __bool__(self):
        return True


NATIVE_RESULT = _NativeResult()

# A raw device allocation in a host function's body: its size argument.
_RAW_MALLOC = re.compile(
    r"\b(?:hip|cuda)(?:Ext)?Malloc(?:WithFlags|Managed)?\s*\([^,]+,(?P<size>[^,)]+)"
)
# A raw device free of an address: ``hipFree((void*)ptr)``.
_RAW_FREE = re.compile(
    r"\b(?:hip|cuda)Free\s*\(\s*(?:\([^)]*\)\s*)?(?P<address>\w+)\s*\)"
)


_GENERATED_SOURCES = (".cpp", ".cc", ".cu", ".hip", ".cxx")


class NativeModule(types.ModuleType):
    """A compiled extension the source loads by name and has no source for."""

    def __init__(
        self, name: str, log: LaunchLog, index=None, host=None, generated=None
    ):
        super().__init__(name)
        self.log = log
        self.index = index
        self.host = host
        # The C/C++ sources the run itself wrote (a library rendering one
        # from a template, then building it), by path.
        self.generated = generated
        # Loaded through ctypes: a C ABI, whose calls return their declared
        # result type. Functions the traced machine answers itself, by name.
        self.c_abi = False
        self.machine_functions: dict | None = None
        # The run's device memory, for the raw allocations host functions make.
        self.memory: DeviceMemory | None = None
        self._definitions: dict = {}
        self.__file__ = f"{name}.so"
        if index is not None:
            # ``export_values()`` puts every member on the module too.
            for enum_class in index.enum_classes.values():
                setattr(self, enum_class.__name__, enum_class)
                for member in enum_class:
                    setattr(self, member.name, member)

    def raw_memory_call(self, name, args):
        """A host function that allocates or frees device memory by hand, called.

        Its body names the allocator (``hipMalloc``, ``hipExtMallocWithFlags``)
        and the parameter giving the size; it returns the new address. A body
        that frees a parameter's address releases it.
        """
        function = (getattr(self.host, "functions", None) or {}).get(name)
        if function is None or self.memory is None:
            return None
        parameters = function.parameters
        allocated = _RAW_MALLOC.search(function.body)
        if allocated is not None and self.scalar_return(name) == 0:
            size = allocated.group("size").strip()
            if size in parameters and parameters.index(size) < len(args):
                return self.memory.allocate_raw(
                    int(_scalar(args[parameters.index(size)]))
                )
            return None
        freed = _RAW_FREE.search(function.body)
        if freed is not None and freed.group("address") in parameters:
            position = parameters.index(freed.group("address"))
            if position < len(args):
                self.memory.free_raw(int(_scalar(args[position])))
                return NATIVE_RESULT
        return None

    def scalar_return(self, name):
        """What ``name`` returns when it succeeds, if it is declared to return a number."""
        signature = (getattr(self.index, "signatures", None) or {}).get(name)
        return {"int": 0, "float": 0.0, "bool": False}.get(
            getattr(signature, "returns", None)
        )

    def signature_doc(self, name) -> str | None:
        if self.index is None or name not in self.index.signatures:
            return None
        return self.index.signatures[name].docstring()

    def kernel_alternatives(self, name) -> tuple[str, ...]:
        """Kernels ``name`` may launch through the functions it calls (see the index)."""
        if self.index is None or not hasattr(self.index, "kernel_alternatives"):
            return ()
        return self.index.kernel_alternatives(name)

    def kernels_of(self, name) -> tuple[str, ...]:
        """The kernels host function ``name`` launches, in order."""
        from TraceLens.ModelUtils.native_index import _strip, launch_sequences

        launched = (getattr(self.index, "launches", None) or {}).get(name)
        if launched is None and self.generated is not None:
            for path, text in self.generated().items():
                if path.endswith(_GENERATED_SOURCES) and name in text:
                    launched = launch_sequences(_strip(text)).get(name)
                    if launched:
                        break
        if launched is None:
            # A generated wrapper exporting a function its included sources define.
            definition = self.exported_definition(name)
            if definition is not None:
                cpp_name, texts = definition
                for text in texts:
                    launched = launch_sequences(text).get(cpp_name)
                    if launched:
                        break
        return tuple(launched or ())

    def exported_definition(self, name):
        """``(C++ function, [source texts])`` a generated wrapper exports as ``name``.

        A JIT build writes a wrapper that includes the library's sources and
        exports their functions by name; the build file lists where its
        ``#include <...>`` paths resolve.
        """
        from TraceLens.ModelUtils.native_index import _strip

        if self.generated is None:
            return None
        cached = self._definitions.get(name)
        if cached is not None:
            return cached or None
        generated = self.generated()
        export = re.compile(
            rf"TVM_FFI_DLL_EXPORT_TYPED_FUNC\(\s*{re.escape(name)}\s*,\s*\(?\s*([\w:]+)"
        )
        include_dirs = [
            directory
            for path, text in generated.items()
            if isinstance(text, str) and path.endswith(".ninja")
            for directory in re.findall(r"-I\s*(\S+)", text)
        ]
        for path, text in generated.items():
            if not isinstance(text, str) or not path.endswith(_GENERATED_SOURCES):
                continue
            found = export.search(text)
            if found is None:
                continue
            texts = [_strip(text)]
            pending = list(re.findall(r'#include\s*[<"]([^">]+)[">]', text))
            seen: set[str] = set()
            while pending and len(seen) < 64:
                include = pending.pop()
                if include in seen:
                    continue
                seen.add(include)
                for candidate in [
                    include,
                    *(os.path.join(root, include) for root in include_dirs),
                ]:
                    if os.path.isabs(candidate) and os.path.isfile(candidate):
                        with open(candidate, errors="ignore") as handle:
                            source = handle.read()
                        texts.append(_strip(source))
                        pending.extend(
                            re.findall(r'#include\s*[<"]([^">]+)[">]', source)
                        )
                        break
            definition = (found.group(1).split("::")[-1], texts)
            self._definitions[name] = definition
            return definition
        self._definitions[name] = ()
        return None

    def tensor_result(self, name, args):
        """The tensor host function ``name`` allocates and returns, or None."""
        from TraceLens.ModelUtils.native_host import (
            TensorResults,
            struct_sizes,
            tensor_host_functions,
        )

        definition = self.exported_definition(name)
        if definition is None or self.host is None:
            return None
        cpp_name, texts = definition
        function = next(
            (
                found[cpp_name]
                for found in map(tensor_host_functions, texts)
                if cpp_name in found
            ),
            None,
        )
        if function is None:
            return None
        output = TensorResults(self.host.interpreter, struct_sizes(texts)).output(
            function, args
        )
        if output is None:
            return None
        if isinstance(output, list):
            # A tuple or struct of tensors returns as a tuple.
            return tuple(
                torch.empty(shape, dtype=getattr(torch, dtype), device="meta")
                for shape, dtype in output
            )
        shape, dtype = output
        return torch.empty(shape, dtype=getattr(torch, dtype), device="meta")

    def declares_host_only(self, name) -> bool:
        """The binding of ``name`` takes no tensor: it sets up, it does not launch."""
        signature = self.index.signatures.get(name) if self.index is not None else None
        if signature is None:
            return False
        return not any("Tensor" in kind for _, kind, *_ in signature.parameters)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name in ("index", "host", "log"):
            raise AttributeError(name)
        exported = (
            self.index.symbols_of(self.__name__) if self.index is not None else None
        )
        if exported is not None and name not in exported:
            raise AttributeError(f"{self.__name__} has no symbol {name!r}")
        if self.index is not None and name in self.index.classes:
            value = _native_class(name, self.__name__)
        else:
            value = NativeFunction(self, name)
        setattr(self, name, value)
        return value


# ----------------------------------------------------------- custom ops


@dataclass
class _CustomOp:
    namespace: str
    name: str
    schema: str = ""
    impls: dict[str, Callable] = field(default_factory=dict)
    fake: Callable | None = None

    def _parsed(self):
        if not self.schema:
            return None
        text = (
            self.schema
            if self.schema.startswith(self.name)
            else f"{self.name}{self.schema}"
        )
        try:
            return torch._C.parse_schema(f"{self.namespace}::{text}")
        except RuntimeError:
            return None

    def record_values(self, args, kwargs) -> list:
        """The op's arguments in schema order, defaults filled, as profiled."""
        parsed = self._parsed()
        if parsed is not None:
            try:
                return _schema_values(
                    types.SimpleNamespace(_schema=parsed), args, kwargs
                )
            except IndexError:
                pass
        return list(args) + list(kwargs.values())

    def reads(self, args, kwargs) -> list:
        """The tensors the op's schema declares it does not write."""
        parsed = self._parsed()
        if parsed is None:
            return []
        try:
            values = _schema_values(types.SimpleNamespace(_schema=parsed), args, kwargs)
        except IndexError:
            return []
        return _schema_tensors(parsed, values, written=False)


class OpRegistry:
    """``torch.library`` registrations made by the interpreted source.

    They live in this run: the host's dispatcher never sees them. Calling one
    is a profiler op scope named ``namespace::name``; its implementation runs,
    and a result a native launch left open comes from its fake kernel.
    """

    _IMPL_ORDER = ("CUDA", "PrivateUse1", "CompositeExplicitAutograd", "Autograd")

    def __init__(self, log: LaunchLog):
        self.log = log
        self.ops: dict[tuple[str, str], _CustomOp] = {}

    def get(self, namespace, name) -> _CustomOp:
        key = (namespace, name.split(".")[0])
        op = self.ops.get(key)
        if op is None:
            op = self.ops[key] = _CustomOp(*key)
        return op

    def call(self, namespace, name, args, kwargs):
        op = self.ops.get((namespace, name))
        if op is None:
            raise AttributeError(f"'_OpNamespace' '{namespace}' has no op {name!r}")
        impl = next(
            (op.impls[key] for key in self._IMPL_ORDER if key in op.impls),
            None,
        ) or next(iter(op.impls.values()), None)
        with self.log.op(
            f"{namespace}::{name}",
            op.record_values(args, kwargs),
            op.reads(args, kwargs),
        ):
            result = impl(*args, **kwargs) if impl is not None else NATIVE_RESULT
        if _left_open(result) and op.fake is not None:
            try:
                try:
                    faked = op.fake(*args, **_accepted(op.fake, kwargs))
                except TypeError:
                    if not kwargs:
                        raise
                    # A fake written for fewer arguments than the op takes (a
                    # kernel-name hint, say): its output does not depend on them.
                    faked = op.fake(*args)
            except (
                Exception
            ) as error:  # noqa: BLE001 - a fake the run itself never calls
                # The engine runs the op's kernel, never its fake: a fake
                # that cannot run (written for other arguments) leaves the
                # result as the kernel left it.
                logger.debug("%s::%s: fake not run: %s", namespace, name, error)
            else:
                result = _filled(result, faked, op._parsed())
        return result


def _filled(result, faked, schema):
    """``result`` with what the native launch left open taken from its fake.

    A number the fake does not give (a stub whose body is ``...``) is its
    schema type's default, as a native call's is.
    """
    if (
        not isinstance(result, tuple)
        or not isinstance(faked, tuple | list)
        or len(faked) != len(result)
    ):
        result, faked = (result,), (faked,)
        single = True
    else:
        single = False
    kinds = (
        [str(returned.type) for returned in schema.returns]
        if schema is not None
        else []
    )
    values = []
    for position, (value, fake) in enumerate(zip(result, faked, strict=True)):
        if _left_open(value):
            value = fake
        kind = (
            kinds[position]
            if position < len(kinds) and not single
            else (kinds[0] if len(kinds) == 1 else "")
        )
        if (value is None or value is NATIVE_RESULT) and kind in _SCHEMA_DEFAULTS:
            value = _SCHEMA_DEFAULTS[kind]
        values.append(value)
    return values[0] if single else tuple(values)


def _schema_tensors(schema, values, written: bool) -> list:
    """The tensor arguments the schema marks written (``Tensor(a!)``), or not."""
    found = []
    for argument, value in zip(schema.arguments, values, strict=False):
        info = argument.alias_info
        is_written = info is not None and info.is_write
        if is_written != written:
            continue
        items = value if isinstance(value, list | tuple) else (value,)
        found.extend(item for item in items if isinstance(item, torch.Tensor))
    return found


# What a host-side native op returns when its C++ body cannot be evaluated.
_SCHEMA_DEFAULTS = {"int": 0, "float": 0.0, "bool": False}


def _native_op(log: LaunchLog, schema, host=None, function=None):
    """The launch a natively implemented op makes, per its schema.

    Handed no tensor, an op whose C++ ``function`` launches no kernel is host
    bookkeeping: ``host`` evaluates it where it can, and it launches nothing.
    """
    holder = types.SimpleNamespace(_schema=schema)

    def launch(*args, **kwargs):
        values = _schema_values(holder, args, kwargs)
        if function is not None and not any(_holds_tensor(value) for value in values):
            if host is not None:
                try:
                    return host(function, tuple(_scalar(value) for value in values))
                except LookupError:
                    pass
            results = [
                _SCHEMA_DEFAULTS.get(str(result.type), NATIVE_RESULT)
                for result in schema.returns
            ]
            if not results:
                return None
            return results[0] if len(results) == 1 else tuple(results)
        log.record(
            log.innermost(schema.name),
            values,
            written=_schema_tensors(schema, values, written=True),
        )
        results = []
        for result in schema.returns:
            info = result.alias_info
            written = None
            if info is not None and info.is_write:
                for argument, value in zip(schema.arguments, values, strict=False):
                    other = argument.alias_info
                    if other is not None and set(other.before_set) & set(
                        info.before_set
                    ):
                        written = value
                        break
            results.append(written if written is not None else NATIVE_RESULT)
        if not results:
            return None
        return results[0] if len(results) == 1 else tuple(results)

    return launch


def _accepted(function, kwargs: dict) -> dict:
    """The keyword arguments ``function`` declares (all of them if it takes ``**``)."""
    import inspect

    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs
    names = {parameter.name for parameter in parameters}
    return {key: value for key, value in kwargs.items() if key in names}


def _left_open(result) -> bool:
    if result is NATIVE_RESULT:
        return True
    if isinstance(result, tuple | list):
        return any(_left_open(item) for item in result)
    return False


class _OpPacket:
    def __init__(self, registry, namespace, name, host):
        self._registry = registry
        self._namespace = namespace
        self._name = name
        self._host = host

    def __call__(self, *args, **kwargs):
        if self._host is not None:
            return self._host(*args, **kwargs)
        return self._registry.call(self._namespace, self._name, args, kwargs)

    def __getattr__(self, overload):
        if self._host is not None:
            return getattr(self._host, overload)
        return self


class _OpNamespace:
    def __init__(self, registry, namespace):
        self._registry = registry
        self._namespace = namespace

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if (self._namespace, name) in self._registry.ops:
            return _OpPacket(self._registry, self._namespace, name, None)
        try:
            host = getattr(getattr(torch.ops, self._namespace), name)
        except (AttributeError, RuntimeError):
            host = None
        if host is None:
            raise AttributeError(
                f"'_OpNamespace' '{self._namespace}' has no op {name!r}"
            )
        return _OpPacket(self._registry, self._namespace, name, host)


class _Ops(types.ModuleType):
    def __init__(self, registry):
        super().__init__("torch.ops")
        self._registry = registry

    def __getattr__(self, namespace):
        if namespace.startswith("__"):
            raise AttributeError(namespace)
        if namespace in ("aten", "prims", "_c10d_functional", "c10d"):
            return getattr(torch.ops, namespace)
        return _OpNamespace(self._registry, namespace)

    def load_library(self, path):
        """A native op library: its ops are declared by its C++ source."""
        self.loaded_libraries.append(str(path))

    @property
    def loaded_libraries(self) -> list:
        return self.__dict__.setdefault("_loaded", [])


class _LibraryModule(types.ModuleType):
    """``torch.library`` for the run: registrations it does not model
    (autograd formulas, vmap rules) are accepted and change nothing."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        real = getattr(torch.library, name, None)
        if isinstance(real, type):
            return real

        def accept(*args, **kwargs):
            function = next((arg for arg in reversed(args) if callable(arg)), None)
            return function if function is not None else _identity_decorator

        return accept


def _library_module(registry: OpRegistry) -> types.ModuleType:
    module = _LibraryModule("torch.library")

    class Library:
        def __init__(self, namespace, kind, dispatch_key=""):
            self.ns = namespace
            self.dispatch_key = dispatch_key

        def define(self, schema, *args, **kwargs):
            name = schema.split("(")[0].strip()
            registry.get(self.ns, name).schema = schema
            return name

        def impl(self, name, fn, dispatch_key="", *args, **kwargs):
            op_name = name if isinstance(name, str) else str(name)
            registry.get(self.ns, op_name.split("::")[-1]).impls[
                dispatch_key or self.dispatch_key or "CompositeExplicitAutograd"
            ] = fn

        def _register_fake(self, name, fn, *args, **kwargs):
            registry.get(self.ns, name.split("::")[-1]).fake = fn

    def _split(qualname):
        namespace, _, name = qualname.partition("::")
        return namespace, name.split(".")[0]

    def register_fake(qualname, fn=None, *args, **kwargs):
        def register(function):
            registry.get(*_split(qualname)).fake = function
            return function

        return register(fn) if fn is not None else register

    def impl(qualname, types_=None, fn=None, *args, **kwargs):
        def register(function):
            registry.get(*_split(qualname)).impls[str(types_ or "CUDA")] = function
            return function

        return register(fn) if fn is not None else register

    def define(qualname, schema, *args, **kwargs):
        registry.get(*_split(qualname)).schema = schema

    def custom_op(
        qualname, fn=None, *, mutates_args=(), device_types=None, schema=None
    ):
        def register(function):
            op = registry.get(*_split(qualname))
            op.impls["CUDA"] = function
            packet = _OpPacket(registry, op.namespace, op.name, None)

            def register_fake_on(fake):
                op.fake = fake
                return fake

            packet.register_fake = register_fake_on  # type: ignore[attr-defined]
            packet.register_kernel = lambda *a, **k: (lambda f: f)  # type: ignore[attr-defined]
            return packet

        return register(fn) if fn is not None else register

    module.Library = Library
    module.register_fake = register_fake
    module.impl_abstract = register_fake
    module.impl = impl
    module.define = define
    module.custom_op = custom_op
    module.infer_schema = torch.library.infer_schema
    module.opcheck = lambda *args, **kwargs: None
    return module


class _ModelModule(types.ModuleType):
    """A model of a torch runtime module over the real one.

    What the model defines wins. Anything else is the real module's: its
    classes and constants are plain Python, and a function that would need
    the device fails the way it does on a machine without one.
    """

    def __init__(self, name, real):
        super().__init__(name)
        self._real = real

    def __getattr__(self, name):
        if name == "_real":
            raise AttributeError(name)
        return getattr(self._real, name)


# ------------------------------------------------------------- cuda model


class _Stream:
    def __init__(self, device=None, priority=0, **kwargs):
        # A stream belongs to a GPU: the current one unless given.
        self.device = _GPU if device is None else torch.device(device)
        if isinstance(device, int):
            self.device = torch.device("cuda", device)
        self.priority = priority
        self.cuda_stream = 0

    def wait_stream(self, other):
        pass

    def wait_event(self, event):
        pass

    def record_event(self, event=None):
        return event or _Event()

    def synchronize(self):
        pass

    def query(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Event:
    def __init__(self, *args, **kwargs):
        pass

    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass

    def synchronize(self):
        pass

    def query(self):
        return True

    def elapsed_time(self, other):
        return 0.0


class _Graph:
    """A CUDA graph. Capturing it records the launches made inside."""

    def __init__(self, log: LaunchLog):
        self._log = log

    def capture_begin(self, *args, **kwargs):
        self._log.begin_capture()

    def capture_end(self):
        self._log.end_capture()

    def replay(self):
        pass

    def reset(self):
        pass

    def pool(self):
        return object()

    def enable_debug_mode(self):
        pass

    def debug_dump(self, path):
        pass


def _cuda_module(
    device: DeviceFacts, log: LaunchLog, memory: DeviceMemory
) -> types.ModuleType:
    module = _ModelModule("torch.cuda", torch.cuda)
    default_stream = _Stream()

    properties = types.SimpleNamespace(
        name=device.name,
        total_memory=device.total_memory,
        multi_processor_count=device.compute_units,
        major=device.major,
        minor=device.minor,
        warp_size=device.warp_size,
        gcnArchName=device.gfx_target,
        max_shared_memory_per_block_optin=device.shared_memory_per_block,
        shared_memory_per_block=device.shared_memory_per_block,
        shared_memory_per_multiprocessor=device.shared_memory_per_cu,
        regs_per_multiprocessor=device.registers_per_block,
        is_integrated=False,
        L2_cache_size=0,
        uuid="0",
    )

    @contextlib.contextmanager
    def graph(cuda_graph, pool=None, stream=None, capture_error_mode="global"):
        cuda_graph.capture_begin()
        try:
            yield cuda_graph
        finally:
            cuda_graph.capture_end()

    @contextlib.contextmanager
    def stream_context(stream=None):
        yield stream

    @contextlib.contextmanager
    def device_context(index=None):
        yield

    module.is_available = lambda: True
    module.is_initialized = lambda: True
    module.init = lambda: None
    module.device_count = lambda: device.count
    module.current_device = lambda: 0
    module.set_device = lambda index: None
    module.get_device_name = lambda index=None: device.name
    module.get_device_properties = lambda index=None: properties
    module.get_device_capability = lambda index=None: (device.major, device.minor)
    module.get_arch_list = lambda: [device.gfx_target] if device.gfx_target else []
    module.synchronize = lambda device=None: None
    module.empty_cache = lambda: None
    module.ipc_collect = lambda: None
    module.mem_get_info = lambda index=None: (
        device.total_memory - device.reserved_memory - memory.allocated - memory.raw,
        device.total_memory,
    )
    module.memory_allocated = lambda index=None: memory.allocated
    module.max_memory_allocated = lambda index=None: memory.peak
    module.memory_reserved = lambda index=None: memory.allocated
    module.max_memory_reserved = lambda index=None: memory.peak

    def reset_peak(index=None):
        memory.peak = memory.allocated

    module.reset_peak_memory_stats = reset_peak
    module.reset_max_memory_allocated = reset_peak
    module.memory_stats = lambda index=None: {}
    module.memory_snapshot = lambda: []
    module.set_per_process_memory_fraction = lambda fraction, index=None: None
    module.Stream = _Stream
    module.ExternalStream = _Stream
    module.Event = _Event
    module.current_stream = lambda device=None: default_stream
    module.default_stream = lambda device=None: default_stream
    module.set_stream = lambda stream: None
    module.stream = stream_context
    module.device = device_context
    module.CUDAGraph = lambda *args, **kwargs: _Graph(log)
    module.graph = graph
    module.graph_pool_handle = lambda: object()
    module.is_current_stream_capturing = lambda: bool(log.capturing)
    module.manual_seed = lambda seed: None
    module.manual_seed_all = lambda seed: None
    module.get_rng_state = lambda device=None: torch.zeros(16, dtype=torch.uint8)
    module.set_rng_state = lambda state, device=None: None
    module.nvtx = types.SimpleNamespace(
        range_push=lambda *a, **k: None,
        range_pop=lambda *a, **k: None,
        range=lambda *a, **k: contextlib.nullcontext(),
        mark=lambda *a, **k: None,
    )
    module.amp = types.SimpleNamespace(
        autocast=lambda *a, **k: contextlib.nullcontext()
    )
    module._lazy_init = lambda: None
    module.is_bf16_supported = lambda *args, **kwargs: True
    module.Device = torch.device
    module.OutOfMemoryError = torch.cuda.OutOfMemoryError
    module.memory = types.SimpleNamespace(
        mem_get_info=module.mem_get_info,
        empty_cache=module.empty_cache,
        CUDAPluggableAllocator=lambda *a, **k: object(),
        MemPool=lambda *a, **k: object(),
        use_mem_pool=lambda *a, **k: contextlib.nullcontext(),
    )
    return module


# --------------------------------------------------------- distributed


class _ProcessGroup:
    """A process group as rank 0 sees it: its ranks and its backend."""

    def __init__(self, ranks, backend="nccl"):
        self.ranks = list(ranks)
        self.backend = str(backend or "nccl")

    def size(self):
        return len(self.ranks)

    def rank(self):
        return self.ranks.index(0) if 0 in self.ranks else -1

    def name(self):
        return "pg"

    def group_name(self):
        return "pg"

    def _get_backend_name(self):
        return self.backend.split(":")[-1].split(",")[0]


def _gathers_by_rank(data, size) -> bool:
    """A per-rank table: one slot per rank, rank 0's set and the others empty."""
    if data.dim() == 0 or data.shape[0] != size or size < 2:
        return False
    return bool((data[1:] == 0).all()) and bool((data[0] != 0).any())


def _symmetric_collective(name, args, kwargs, size, shadows) -> bool:
    """Apply collective ``name`` as every rank holding the same values would.

    Returns whether the outputs' values are known. Contents are the tensors
    themselves on the CPU, or their copies.
    """

    def values(tensor):
        if not isinstance(tensor, torch.Tensor):
            return None
        return (
            tensor if not tensor.is_meta else shadows.get(tensor) if shadows else None
        )

    op = kwargs.get(
        "op", args[1] if name in ("all_reduce", "reduce") and len(args) > 1 else None
    )
    op_name = str(getattr(op, "name", op) or "SUM").upper()
    try:
        if name in ("all_reduce", "reduce"):
            data = values(args[0] if args else kwargs.get("tensor"))
            if data is None:
                return False
            if "SUM" in op_name and _gathers_by_rank(data, size):
                # Each rank fills its own slot: every slot gets what this
                # rank put in its own.
                data.copy_(data[0:1].expand_as(data))
            elif "SUM" in op_name:
                data.mul_(size)
            elif "PRODUCT" in op_name:
                data.pow_(size)
            return True
        if name in ("broadcast", "barrier"):
            return True
        if name == "all_gather":
            outputs, source = args[0], values(args[1])
            if source is None:
                return False
            for output in outputs:
                target = values(output)
                if target is None:
                    return False
                target.copy_(source)
            return True
        if name == "all_gather_into_tensor":
            target, source = values(args[0]), values(args[1])
            if target is None or source is None:
                return False
            target.copy_(
                source.repeat(*([size] + [1] * (source.dim() - 1))).reshape(
                    target.shape
                )
            )
            return True
    except (RuntimeError, IndexError, TypeError):
        return False
    return False


class _Store:
    """``torch.distributed``'s key-value store, as every rank's writes leave it.

    The ranks run the same code, so a key another rank writes holds what
    this rank wrote under its own name: a read of a key no one here wrote
    finds the one that differs from it only by a number (a rank). Counters
    are bumped by every rank.
    """

    def __init__(self, world_size: int = 1):
        self._values: dict[str, bytes] = {}
        self._world_size = world_size

    @staticmethod
    def _bytes(value) -> bytes:
        return value if isinstance(value, bytes) else str(value).encode()

    def set(self, key, value):
        self._values[key] = self._bytes(value)

    def get(self, key):
        if key in self._values:
            return self._values[key]
        pattern = re.sub(r"\d+", r"\\d+", re.escape(key))
        for stored, value in reversed(list(self._values.items())):
            if re.fullmatch(pattern, stored):
                return value
        raise KeyError(key)

    def add(self, key, amount):
        total = int(self._values.get(key, b"0")) + int(amount) * self._world_size
        self._values[key] = str(total).encode()
        return total

    def compare_set(self, key, expected, desired):
        current = self._values.get(key)
        if current is None or current == self._bytes(expected):
            self._values[key] = self._bytes(desired)
        return self._values[key]

    def check(self, keys):
        return True

    def wait(self, *args, **kwargs):
        return None

    def delete_key(self, key):
        return self._values.pop(key, None) is not None

    def num_keys(self):
        return len(self._values)

    def set_timeout(self, timeout):
        return None


class _TCPStore(_Store):
    def __init__(self, *args, world_size=None, **kwargs):
        super().__init__(world_size or 1)


class _PrefixStore(_Store):
    def __init__(self, prefix, store):
        super().__init__(getattr(store, "_world_size", 1))
        self.underlying_store = store
        self._prefix = prefix
        self._values = store._values if isinstance(store, _Store) else {}


def _distributed_module(world_size: int, log: LaunchLog) -> types.ModuleType:
    module = _ModelModule("torch.distributed", torch.distributed)
    world = _ProcessGroup(range(world_size))
    state = {"initialized": False}

    def collective(name):
        def call(*args, **kwargs):
            values = [*args, *kwargs.values()]
            group = kwargs.get("group") or next(
                (value for value in args if isinstance(value, _ProcessGroup)), world
            )
            known = _symmetric_collective(name, args, kwargs, group.size(), log.shadows)
            on_device = any(
                _is_gpu(item)
                for value in values
                for item in (value if isinstance(value, list | tuple) else (value,))
            )
            if on_device:
                log.record(log.innermost(f"c10d::{name}"), values, native=not known)
            return None

        return call

    module.is_available = lambda: True
    module.is_initialized = lambda: state["initialized"]

    def init_process_group(backend=None, *args, **kwargs):
        state["initialized"] = True
        world.backend = str(backend or "nccl")

    module.init_process_group = init_process_group
    module.destroy_process_group = lambda *args, **kwargs: None
    module.get_world_size = lambda group=None: (group or world).size()
    module.get_rank = lambda group=None: 0
    module.get_backend = lambda group=None: (group or world)._get_backend_name()
    module.new_group = (
        lambda ranks=None, timeout=None, backend=None, *args, **kwargs: _ProcessGroup(
            ranks if ranks is not None else range(world_size), backend or world.backend
        )
    )
    module.new_subgroups = lambda *args, **kwargs: (world, [world])
    module.get_process_group_ranks = lambda group=None: list((group or world).ranks)
    module.get_group_rank = lambda group, global_rank: (group or world).ranks.index(
        global_rank
    )
    module.get_global_rank = lambda group, group_rank: (group or world).ranks[
        group_rank
    ]
    module.barrier = lambda *args, **kwargs: None
    module.monitored_barrier = lambda *args, **kwargs: None
    for name in (
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "all_to_all",
        "all_to_all_single",
        "broadcast",
        "send",
        "recv",
        "isend",
        "irecv",
        "reduce",
        "gather",
        "scatter",
    ):
        setattr(module, name, collective(name))
    module.broadcast_object_list = lambda *args, **kwargs: None
    module.all_gather_object = lambda output, obj, group=None: output.__setitem__(
        slice(None), [obj] * len(output)
    )
    module.ProcessGroup = _ProcessGroup
    module.GroupMember = types.SimpleNamespace(WORLD=world, NON_GROUP_MEMBER=None)
    module.group = types.SimpleNamespace(WORLD=world)
    module.ReduceOp = (
        torch.distributed.ReduceOp
        if torch.distributed.is_available()
        else (types.SimpleNamespace(SUM=0, MAX=1, MIN=2, PRODUCT=3, AVG=4))
    )
    module.Backend = types.SimpleNamespace(NCCL="nccl", GLOO="gloo")
    module.Store = _Store
    module.TCPStore = _TCPStore
    module.PrefixStore = _PrefixStore
    store = _TCPStore(world_size=world_size)
    module.distributed_c10d = types.SimpleNamespace(
        _get_default_store=lambda: store,
        _get_default_group=lambda: world,
        _world=types.SimpleNamespace(pg_map={}, pg_names={}),
        GroupMember=module.GroupMember,
        _get_pg_default_device=lambda group=None: torch.device("meta"),
        get_world_size=module.get_world_size,
        get_rank=module.get_rank,
    )
    return module


# -------------------------------------------------------------- triton


_TRITON_WRITES = frozenset(
    {
        "store",
        "atomic_add",
        "atomic_and",
        "atomic_cas",
        "atomic_max",
        "atomic_min",
        "atomic_or",
        "atomic_xchg",
        "atomic_xor",
    }
)


# What reads through a pointer: atomics read what they update.
_TRITON_READS = frozenset(
    {"load", *(name for name in _TRITON_WRITES if name != "store")}
)


def stored_parameters(node) -> frozenset[str] | None:
    """The parameters of the Triton kernel ``node`` its body may write through.

    A parameter is written when a pointer derived from it (by any
    assignment) reaches ``tl.store``/``tl.atomic_*``, the receiver of a
    ``.store`` method (a block or tensor descriptor), or a call to anything
    other than a ``tl`` builtin (another kernel function may store). None
    when the body is not available.
    """
    return _accessed_parameters(node, _TRITON_WRITES)


def loaded_parameters(node) -> frozenset[str] | None:
    """The parameters of the Triton kernel ``node`` its body may read through:
    as :func:`stored_parameters`, for ``tl.load`` and the atomics (which read
    what they update)."""
    if not hasattr(node, "_tracelens_loaded"):
        node._tracelens_loaded = _accessed_parameters(node, _TRITON_READS)
    return node._tracelens_loaded


def _accessed_parameters(node, operations) -> frozenset[str] | None:
    """The parameters a pointer derived from reaches one of ``operations``
    (or a call to anything other than a ``tl`` builtin)."""
    import ast

    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return None
    params = [
        arg.arg
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    ]
    sources: dict[str, set[str]] = {name: {name} for name in params}

    def origins(expr) -> set[str]:
        # A loaded value is data, not a pointer: an offset read from one
        # buffer does not make that buffer a store target.
        found: set[str] = set()
        pending = [expr]
        while pending:
            item = pending.pop()
            if isinstance(item, ast.Name):
                found |= sources.get(item.id, set())
            elif (
                isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute)
                and item.func.attr == "load"
            ):
                continue
            pending.extend(ast.iter_child_nodes(item))
        return found

    # Assignments may refer to later ones in loops: iterate to a fixed point.
    changed = True
    while changed:
        changed = False
        for item in ast.walk(node):
            targets, value = (), None
            if isinstance(item, ast.Assign):
                targets, value = item.targets, item.value
            elif (
                isinstance(item, ast.AugAssign | ast.AnnAssign)
                and item.value is not None
            ):
                targets, value = (item.target,), item.value
            elif isinstance(item, ast.For):
                targets, value = (item.target,), item.iter
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                        before = sources.setdefault(name.id, set())
                        after = before | origins(value)
                        if after != before:
                            sources[name.id] = after
                            changed = True

    accessed: set[str] = set()
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        function = item.func
        if isinstance(function, ast.Attribute) and function.attr in operations:
            receiver = function.value
            if isinstance(receiver, ast.Name) and receiver.id not in sources:
                # ``tl.store(pointer, ...)``: the pointer is the first argument.
                pointer = (
                    item.args[0]
                    if item.args
                    else next(
                        (kw.value for kw in item.keywords if kw.arg == "pointer"), None
                    )
                )
                if pointer is not None:
                    accessed |= origins(pointer)
            else:
                accessed |= origins(receiver)
        elif (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id not in sources
        ):
            continue  # a ``tl``/``libdevice`` builtin: computes, accesses nothing
        else:
            for argument in (*item.args, *(kw.value for kw in item.keywords)):
                accessed |= origins(argument)
    return frozenset(accessed & set(params))


_TRITON_ACCESSES = frozenset({"load", "store", *_TRITON_WRITES})
_TRITON_LOOPS = frozenset({"range", "static_range"})


def content_controlled_parameters(node) -> frozenset[str] | None:
    """See :func:`_content_controlled_parameters`; read once per kernel's source."""
    if not hasattr(node, "_tracelens_steering"):
        node._tracelens_steering = _content_controlled_parameters(node)
    return node._tracelens_steering


def _content_controlled_parameters(node) -> frozenset[str] | None:
    """Parameters of Triton kernel ``node`` whose loaded contents steer it.

    The values loaded through a parameter's pointer are its contents. They
    steer the kernel when they reach a pointer, a mask, a loop bound or a
    branch; a parameter whose contents are only computed with and stored is
    data. None when the body is not available.
    """
    import ast

    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return None
    params = [
        arg.arg
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    ]
    pointers: dict[str, set[str]] = {name: {name} for name in params}
    contents: dict[str, set[str]] = {}

    def is_access(call, names):
        function = call.func
        return (
            isinstance(function, ast.Attribute)
            and function.attr in names
            and isinstance(function.value, ast.Name)
            and function.value.id not in pointers
        )

    def pointer_origins(expr):
        found = set()
        for item in ast.walk(expr):
            if isinstance(item, ast.Name):
                found |= pointers.get(item.id, set())
        return found

    def content_of(expr) -> set[str]:
        found = set()
        for item in ast.walk(expr):
            if isinstance(item, ast.Name):
                found |= contents.get(item.id, set())
            elif isinstance(item, ast.Call) and is_access(
                item, {"load", *_TRITON_WRITES}
            ):
                pointer = item.args[0] if item.args else None
                if pointer is not None:
                    found |= pointer_origins(pointer)
        return found

    changed = True
    while changed:
        changed = False
        for item in ast.walk(node):
            targets, value = (), None
            if isinstance(item, ast.Assign):
                targets, value = item.targets, item.value
            elif (
                isinstance(item, ast.AugAssign | ast.AnnAssign)
                and item.value is not None
            ):
                targets, value = (item.target,), item.value
            elif isinstance(item, ast.For):
                targets, value = (item.target,), item.iter
            for target in targets:
                for name in ast.walk(target):
                    if not (
                        isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
                    ):
                        continue
                    pointed = pointers.setdefault(name.id, set()) | pointer_origins(
                        value
                    )
                    held = contents.setdefault(name.id, set()) | content_of(value)
                    if pointed != pointers[name.id] or held != contents[name.id]:
                        pointers[name.id], contents[name.id] = pointed, held
                        changed = True

    steering: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Call) and is_access(item, _TRITON_ACCESSES):
            # The pointer, and the mask (``mask=`` or its positional place).
            steering |= content_of(item.args[0]) if item.args else set()
            mask_position = 1 if item.func.attr == "load" else 2
            mask = next((kw.value for kw in item.keywords if kw.arg == "mask"), None)
            if mask is None and len(item.args) > mask_position:
                mask = item.args[mask_position]
            if mask is not None:
                steering |= content_of(mask)
        elif isinstance(item, ast.Call) and (
            (isinstance(item.func, ast.Name) and item.func.id in _TRITON_LOOPS)
            or (
                isinstance(item.func, ast.Attribute) and item.func.attr in _TRITON_LOOPS
            )
        ):
            for argument in item.args:
                steering |= content_of(argument)
        elif isinstance(item, ast.If | ast.While | ast.IfExp):
            steering |= content_of(item.test)
    return frozenset(steering & set(params))


class TritonKernel:
    """A ``@triton.jit`` function. Launching it records its own name."""

    def __init__(self, fn, log: LaunchLog, **options):
        self.fn = fn
        self._log = log
        self.__name__ = getattr(fn, "__name__", "triton_kernel")
        self.__qualname__ = getattr(fn, "__qualname__", self.__name__)
        self.__module__ = getattr(fn, "__module__", "triton")
        self.options = options
        try:
            self.arg_names = list(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            self.arg_names = []
        self._stored = stored_parameters(getattr(fn, "_node", None))
        if self._stored is not None:
            node = fn._node
            self.arg_names = [
                arg.arg
                for arg in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
            ]

    def _constants(self, args, kwargs) -> dict:
        """The launch's ``constexpr`` arguments by name, defaults included."""
        node = getattr(self.fn, "_node", None)
        if node is None:
            return {}
        params = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        constant = {
            param.arg
            for param in params
            if param.annotation is not None
            and "constexpr" in ast.unparse(param.annotation)
        }
        bound = dict(zip(self.arg_names, args)) | dict(kwargs)
        try:
            defaults = {
                name: parameter.default
                for name, parameter in inspect.signature(self.fn).parameters.items()
                if parameter.default is not inspect.Parameter.empty
            }
        except (TypeError, ValueError):
            defaults = {}
        return {
            name: bound.get(name, defaults.get(name))
            for name in constant
            if name in bound or name in defaults
        }

    def kernel_name(self, args, kwargs) -> str:
        """The compiled kernel's symbol: what the ``repr`` it is given makes of
        the launch's constants, else the function's name."""
        named = self.options.get("repr")
        if callable(named):
            try:
                name = named(
                    types.SimpleNamespace(constants=self._constants(args, kwargs))
                )
            except Exception:  # noqa: BLE001 - a name the launch cannot make
                name = None
            if isinstance(name, str) and name:
                return name
        return self.__name__

    def _written(self, args, kwargs):
        """The arguments this launch may write, or None for all of them."""
        if self._stored is None:
            return None
        bound = dict(zip(self.arg_names, args))
        bound.update(kwargs)
        return [value for name, value in bound.items() if name in self._stored]

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            values = list(args) + [
                value for key, value in kwargs.items() if key in self.arg_names
            ]
            log = self._log
            name = log.innermost(None)
            recorded = log.innermost_values(None)
            if name is None:
                # Outside a torch op the profiler annotates the launch by
                # the Python function that made it, with its arguments.
                caller = log.caller() if log.caller is not None else None
                name, recorded = caller or (self.__name__, values)
            # Run on the CPU copies when it can: what it writes stays known.
            unknown = _emulate_triton(self, grid, args, kwargs)
            log.record(
                name,
                recorded if recorded is not None else values,
                (self.kernel_name(args, kwargs),),
                written=self._written(args, kwargs) if unknown is None else unknown,
            )
            return None

        return launch

    def run(self, *args, grid=None, **kwargs):
        return self[grid](*args, **kwargs)

    def __call__(self, *args, **kwargs):
        # Called from another kernel: inlined device code. It only runs while
        # a launch is being emulated.
        if _EMULATING[0]:
            return self.fn(*args, **kwargs)
        return None


# Kernels are emulated one at a time; a device function called meanwhile runs.
_EMULATING = [False]
# The most programs and elements a launch may have to be emulated.
_EMULATED_PROGRAMS = 4096


class _NotEmulable(Exception):
    pass


def _emulate_triton(kernel: TritonKernel, grid, args, kwargs) -> list | None:
    """Make a Triton launch's writes computable on the CPU, when they are read.

    Returns the tensors whose contents the launch leaves unknown, or None
    when it cannot be run. Nothing runs at the launch: it keeps how to read
    each argument's contents as they are now (see :class:`_TritonLaunch`),
    never computing them, and runs when what it writes is first read. A
    launch whose results only other kernels read never runs, nor do the
    launches before it. A launch writing part of a storage runs at once.
    """
    shadows = kernel._log.shadows
    node = getattr(kernel.fn, "_node", None)
    if shadows is None or node is None or _EMULATING[0] or kernel._stored is None:
        return None
    if getattr(kernel.fn, "_frame", None) is None:
        # Not the source interpreter's function: nothing runs its body.
        return None
    try:
        from triton.runtime.interpreter import GridExecutor  # noqa: F401
    except ImportError:
        return None
    names = kernel.arg_names
    if not all(name.isidentifier() for name in names):
        return None
    # A launch the emulator could not run fails alike for the same kernel,
    # sizes and dtypes: it is not tried again.
    failing = (
        id(kernel.fn),
        tuple(_launch_form(value) for value in args),
        tuple(sorted((name, _launch_form(value)) for name, value in kwargs.items())),
    )
    if failing in shadows.__dict__.get("_not_emulable", ()):
        return None
    written = [
        tensor
        for value in kernel._written(args, kwargs) or ()
        for tensor in (value if isinstance(value, tuple | list) else (value,))
        if isinstance(tensor, torch.Tensor) and tensor.is_meta
    ]
    if not written:
        return []
    try:
        shape = grid(dict(zip(names, args)) | dict(kwargs)) if callable(grid) else grid
    except (
        Exception
    ) as error:  # noqa: BLE001 - a grid it cannot work out is left unknown
        logger.debug("%s: not emulated: %s", kernel.__name__, error)
        return None
    shape = tuple(shape) if isinstance(shape, tuple | list) else (shape,)
    if not all(isinstance(size, int) for size in shape):
        return None
    programs = 1
    for size in shape:
        programs *= size
    if programs > _EMULATED_PROGRAMS:
        logger.debug("%s: not emulated: %d programs", kernel.__name__, programs)
        return None
    try:
        launch = _TritonLaunch(kernel, shape, node, args, kwargs, written, shadows)
    except _NotEmulable as reason:
        logger.debug("%s: not emulated: %s", kernel.__name__, reason)
        return None
    if all(_whole_storage(tensor) for tensor in written):
        for position, tensor in enumerate(written):
            shadows.defer_write(
                tensor, lambda region, position=position: launch.result(position)
            )
        return []
    # Part of a storage: what it writes joins what is there now.
    left_unknown = []
    for position, tensor in enumerate(written):
        try:
            result = launch.result(position)
        except _UnknownContents:
            left_unknown.append(tensor)
            continue
        copy = shadows.get(tensor)
        if copy is None:
            shadows.set(tensor, result)
        else:
            copy.copy_(result)
    return left_unknown


def _whole_storage(tensor) -> bool:
    """Whether ``tensor`` is all of its storage, laid out in order."""
    return (
        tensor.storage_offset() == 0
        and tensor.is_contiguous()
        and tensor.numel() * tensor.element_size() == tensor.untyped_storage().nbytes()
    )


def _over(storage, geometry) -> torch.Tensor:
    """The tensor with ``geometry`` over a CPU storage."""
    size, stride, offset, dtype = geometry
    return torch.empty(0, dtype=dtype).set_(storage, offset, size, stride)


class _Contents:
    """A device tensor argument of a Triton launch: its geometry, and how to
    read its contents as they were at the launch, never the tensor itself.

    A copy is snapshotted (views of one storage keep sharing the snapshot);
    lazy contents, or a storage's logged writes, are read when the launch
    runs, so nothing is computed until then.
    """

    def __init__(self, name, tensor, shadows, storages):
        self.name = name
        self.geometry = Shadows._geometry(tensor)
        self.numel = tensor.numel()
        self._read = None
        copy = shadows._peek(tensor)
        if copy is not None:
            storage = copy.untyped_storage()
            key = ("copy", storage._cdata)
            if key not in storages:
                storages[key] = storage.clone()
            snapshot, geometry = storages[key], Shadows._geometry(copy)
            self._read = lambda: _over(snapshot, geometry)
            return
        make = shadows._pending(tensor)
        if make is not None:
            self._read = make
            return
        logged = shadows._logged(tensor)
        if logged is not None:
            key = ("log", shadows._storage_key(tensor))
            if key not in storages:
                nbytes, entries = logged
                entries, whole = list(entries), []

                def replayed(nbytes=nbytes, entries=entries, whole=whole):
                    if not whole:
                        whole.append(
                            _replay(nbytes, entries, ((nbytes,), (1,), 0, torch.uint8))
                        )
                    return whole[0]

                storages[key] = replayed
            replayed, geometry = storages[key], self.geometry
            self._read = lambda: _over(replayed().untyped_storage(), geometry)

    @property
    def readable(self) -> bool:
        return self._read is not None

    def read(self):
        """The contents on the CPU, or None if they are unknown."""
        if self._read is None:
            return None
        try:
            return self._read()
        except _UnknownContents:
            return None


class _TritonLaunch:
    """A Triton launch, run on the CPU when what it writes is first read.

    Triton's own interpreter executes each program, with ``triton.language``
    replaced by its NumPy implementation; the kernel body itself is run by
    the source interpreter (the stub below only carries its signature).

    An argument without contents whose contents are only data (see
    :func:`content_controlled_parameters`) is filled twice, with different
    values, and the launch run once per filling: a written tensor that comes
    out the same both times does not depend on it, so an output made by
    ``torch.empty`` and written whole is known. An argument whose contents
    steer the kernel, or holding device addresses the CPU cannot follow,
    must have contents. The same launch on the same contents writes the
    same: a layer's repeat of the one before it is not run again.
    """

    def __init__(self, kernel, shape, node, args, kwargs, written, shadows):
        self.kernel, self.shape, self.node = kernel, shape, node
        self.steering = content_controlled_parameters(node) or frozenset()
        self.memo = shadows.__dict__.setdefault("_emulated", {})
        self.failures = shadows.__dict__.setdefault("_not_emulable", set())
        self.failing = (
            id(kernel.fn),
            tuple(_launch_form(value) for value in args),
            tuple(
                sorted((name, _launch_form(value)) for name, value in kwargs.items())
            ),
        )
        contents: dict[int, _Contents] = {}
        storages: dict = {}

        def capture(name, value):
            if isinstance(value, torch.Tensor) and value.is_meta:
                if id(value) not in contents:
                    contents[id(value)] = _Contents(name, value, shadows, storages)
                return contents[id(value)]
            if isinstance(value, tuple | list):
                return type(value)(capture(name, item) for item in value)
            return value

        names = kernel.arg_names
        self.args = [
            capture(names[index] if index < len(names) else "", value)
            for index, value in enumerate(args)
        ]
        self.kwargs = {name: capture(name, value) for name, value in kwargs.items()}
        self.written = [contents[id(tensor)] for tensor in written]
        self.arguments = list(contents.values())
        for argument in self.arguments:
            if not argument.readable and (
                argument.name in self.steering or argument.numel > Shadows.LIMIT
            ):
                raise _NotEmulable(f"{argument.name} has no copy")
        self._results: list | None = None

    def result(self, position) -> torch.Tensor:
        """What the launch writes into its ``position``-th written tensor."""
        if self._results is None:
            self._results = self._run()
        if self._results[position] is None:
            raise _UnknownContents(
                f"{self.kernel.__name__} writes what its unknown arguments decide"
            )
        return self._results[position].clone()

    def _run(self) -> list:
        """What the launch writes into each written tensor, None where unknown."""
        kernel = self.kernel
        contents = {id(argument): argument.read() for argument in self.arguments}
        unknown = {key for key, copy in contents.items() if copy is None}
        for argument in self.arguments:
            copy = contents[id(argument)]
            if copy is None:
                if argument.name in self.steering or argument.numel > Shadows.LIMIT:
                    raise _UnknownContents(f"{argument.name} has no contents")
            elif copy.dtype in (torch.int64, torch.uint64) and copy.numel():
                signed = copy.view(torch.int64)
                if bool(((signed >= DeviceMemory.BASE) | (signed < 0)).any()):
                    raise _UnknownContents(f"{argument.name} holds device addresses")
        key = _emulation_key(
            kernel,
            self.shape,
            self.args,
            self.kwargs,
            lambda argument: contents[id(argument)],
        )
        if key is not None and key in self.memo:
            return self.memo[key]

        # An argument the kernel only stores to (an output made by
        # ``torch.empty``) does not change what it writes: filled with values
        # a write is unlikely to repeat, one run shows whether it is written
        # whole, as an element still holding its filling was not written.
        loaded = loaded_parameters(self.node)
        outputs = {
            id(argument)
            for argument in self.arguments
            if id(argument) in unknown
            and loaded is not None
            and argument.name not in loaded
        }
        inputs = unknown - outputs

        def host(value, run, bases):
            """``value`` on the CPU for filling ``run``; views of one base share it."""
            if isinstance(value, _Contents):
                copy = contents[id(value)]
                if copy is None:
                    fill = ("fill", id(value))
                    if fill not in bases:
                        bases[fill] = _made_up_contents(
                            value.geometry, run, wide=id(value) in outputs
                        )
                    return bases[fill]
                if not unknown:
                    return copy
                # Each filling works on its own copy of the storage; views of
                # one storage keep sharing it.
                storage = copy.untyped_storage()
                if storage._cdata not in bases:
                    bases[storage._cdata] = storage.clone()
                return _over(bases[storage._cdata], Shadows._geometry(copy))
            if isinstance(value, tuple | list):
                return type(value)(host(item, run, bases) for item in value)
            return value

        bases: list[dict] = [{}, {}]

        def filling(run):
            return (
                [host(value, run, bases[run]) for value in self.args],
                {
                    name: host(value, run, bases[run])
                    for name, value in self.kwargs.items()
                },
            )

        runs = [0, 1] if inputs else [0]
        fillings = [filling(run) for run in runs]
        filled = {key: bases[0][("fill", key)].clone() for key in outputs}
        try:
            _interpret(kernel, self.node, self.shape, fillings)
            if not inputs and any(
                bool((bases[0][("fill", key)] == filled[key]).any()) for key in outputs
            ):
                # Perhaps not written whole: run again on other fillings.
                runs.append(1)
                _interpret(kernel, self.node, self.shape, [filling(1)])
        except Exception as error:  # noqa: BLE001 - what it cannot run is unknown
            self.failures.add(self.failing)
            raise _UnknownContents(
                f"{kernel.__name__} could not be emulated: {error}"
            ) from error
        results = []
        for argument in self.written:
            first = host(argument, 0, bases[0])
            if len(runs) == 2 and not torch.equal(first, host(argument, 1, bases[1])):
                results.append(None)
            else:
                results.append(first.clone())
        if key is not None:
            self.memo[key] = results
        return results


def _made_up_contents(geometry, run, wide=False) -> torch.Tensor:
    """Made-up contents for filling ``run`` of an argument without any; spread
    over the dtype's whole range when ``wide``."""
    size, stride, _offset, dtype = geometry
    generator = torch.Generator().manual_seed(run + 1)
    filler = torch.empty_strided(size, stride, dtype=dtype)
    if wide and dtype.is_floating_point:
        bound = min(torch.finfo(dtype).max / 2, 1e6)
        filler.copy_(
            (torch.rand(size, generator=generator, dtype=torch.float64) * 2 - 1) * bound
        )
    elif wide and dtype != torch.bool and not dtype.is_complex:
        info = torch.iinfo(dtype)
        low, high = max(info.min, -(1 << 63)), min(info.max, (1 << 63) - 1)
        filler.copy_(
            torch.randint(low, high, size, generator=generator, dtype=torch.int64)
        )
    elif dtype.is_floating_point or dtype.is_complex:
        filler.copy_(torch.rand(size, generator=generator) * 997 + run * 13)
    elif dtype == torch.bool:
        filler.copy_(torch.rand(size, generator=generator) > 0.5)
    else:
        filler.copy_(torch.randint(0, 127, size, generator=generator) + run * 3)
    return filler


def _interpret(kernel, node, shape, fillings) -> None:
    """Run a Triton launch over each filling of its arguments, in Triton's interpreter."""
    import importlib

    from triton.runtime.interpreter import GridExecutor
    from triton.runtime.jit import JITFunction

    names = kernel.arg_names
    real = {
        name: importlib.import_module(name)
        for name in (
            "triton",
            "triton.language",
            "triton.language.core",
            "triton.language.math",
        )
    }
    real_tl = real["triton.language"]
    module = kernel.fn._frame.module.__dict__
    swapped = {
        key: value
        for key, value in module.items()
        if isinstance(value, types.ModuleType)
        and value.__name__ in real
        and value is not real[value.__name__]
    }

    def run(arguments):
        kernel.fn(**arguments)

    # A stub with the kernel's own signature: Triton reads its parameters,
    # their constexpr annotations and the triton.language it sees.
    constexprs = {
        arg.arg
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        if arg.annotation is not None and ast_text(arg.annotation).endswith("constexpr")
    }
    positional = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args)]
    defaults = dict(zip(reversed(positional), reversed(kernel.fn.__defaults__ or ())))
    defaults.update(kernel.fn.__kwdefaults__ or {})
    defaults = {key: getattr(value, "value", value) for key, value in defaults.items()}
    stub_globals = {"tl": real_tl, "__run": run, "__defaults": defaults}
    code = "def stub({}):\n    __run(dict(locals()))\n".format(
        ", ".join(
            f"{name}=__defaults[{name!r}]" if name in defaults else name
            for name in names
        )
    )
    exec(  # noqa: S102 - names checked by the caller
        compile(code, "<triton signature stub>", "exec"), stub_globals
    )
    stub = stub_globals["stub"]
    stub.__annotations__ = {name: real_tl.constexpr for name in constexprs}

    # triton.language's own @jit helpers (cdiv, max, ...) are plain Python
    # over the patched language once the interpreter is in place.
    def call_helper(self, *call_args, **call_kwargs):
        return self.fn(*call_args, **call_kwargs)

    original_call = JITFunction.__call__
    module.update({key: real[value.__name__] for key, value in swapped.items()})
    JITFunction.__call__ = call_helper
    _EMULATING[0] = True
    try:
        for call_args, call_kwargs in fillings:
            GridExecutor(stub, names, shape)(*call_args, **call_kwargs)
    finally:
        _EMULATING[0] = False
        JITFunction.__call__ = original_call
        module.update(swapped)


def _launch_form(value):
    """What of a launch argument decides how its kernel runs: a tensor's dtype
    and rank, a scalar's value."""
    if isinstance(value, torch.Tensor):
        return ("tensor", str(value.dtype), value.dim())
    if isinstance(value, tuple | list):
        return tuple(_launch_form(item) for item in value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return type(value).__name__


def _emulation_key(kernel, grid, args, kwargs, contents):
    """What decides a Triton launch's writes: the kernel, its grid, and each
    argument (a tensor's geometry and its contents, as ``contents`` gives a
    :class:`_Contents`), or None."""
    import hashlib

    def part(value):
        if isinstance(value, _Contents | torch.Tensor):
            if isinstance(value, _Contents):
                copy = contents(value)
                size, stride, offset, dtype = value.geometry
                geometry = (str(dtype), size, stride, offset)
            else:
                copy = value
                geometry = (
                    str(value.dtype),
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.storage_offset(),
                )
            if copy is None:
                return ("unknown", *geometry)
            digest = hashlib.blake2b(
                copy.contiguous().view(torch.uint8).numpy().tobytes(), digest_size=16
            )
            return ("tensor", *geometry, digest.hexdigest())
        if isinstance(value, tuple | list):
            return tuple(part(item) for item in value)
        if value is None or isinstance(value, bool | int | float | str):
            return value
        return (
            "object",
            type(value).__name__,
            (
                getattr(value, "value", None)
                if isinstance(getattr(value, "value", None), int | float | bool | str)
                else id(value)
            ),
        )

    try:
        return (
            id(kernel.fn),
            grid,
            tuple(part(value) for value in args),
            tuple(sorted((name, part(value)) for name, value in kwargs.items())),
        )
    except Exception:  # noqa: BLE001 - an argument it cannot read: run the launch
        return None


def ast_text(node) -> str:
    import ast

    return ast.unparse(node)


# ------------------------------------------------------- kernel DSLs

# Compiler packages of Python GPU-kernel DSLs: their ``jit``/``kernel``
# decorators compile a function to a kernel when the host calls it.
DSL_COMPILERS = ("flydsl.compiler",)


class _DslTensor:
    """A tensor handed to a DSL (``from_dlpack``); layout marks change nothing."""

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self


def _dsl_values(values):
    return [
        value.tensor if isinstance(value, _DslTensor) else value for value in values
    ]


class DslKernel:
    """A DSL-compiled function. A host call launches its kernel.

    A host launcher (``@jit``) runs its body, whose kernels (``@kernel``)
    launch named as the DSL's compiler names them: the name the kernel
    states, else its function's name and its number among the kernels the
    launcher compiles (``moe_reduction_kernel_0``). A body that cannot run
    is one launch, named by the op it is part of.
    """

    def __init__(
        self, fn, log: LaunchLog, deferred: bool = False, name: str | None = None
    ):
        self._func = fn
        self._log = log
        self._deferred = deferred
        self._name = name
        self._known_block_size = None
        self.__name__ = getattr(fn, "__name__", "kernel")
        self.__qualname__ = getattr(fn, "__qualname__", self.__name__)
        self.__module__ = getattr(fn, "__module__", "dsl")
        self.__wrapped__ = fn

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return functools.partial(self, instance)

    def _launch(self, args, kwargs, kernels=()):
        values = _dsl_values([*args, *kwargs.values()])
        log = self._log
        name = log.innermost(None)
        recorded = log.innermost_values(None)
        if name is None:
            caller = log.caller() if log.caller is not None else None
            name, recorded = caller or (self.__name__, values)
        log.record(name, recorded if recorded is not None else values, kernels)
        return None

    def __call__(self, *args, **kwargs):
        if self._deferred:
            # ``kernel(...)`` inside a launcher: ``.launch(grid=...)`` runs it.
            compiling = _DSL_COMPILING[-1] if _DSL_COMPILING else None
            if compiling is None:
                return types.SimpleNamespace(
                    launch=lambda **_: self._launch(args, kwargs)
                )
            kernel_id = compiling["next"]
            compiling["next"] += 1
            name = self._name or f"{self.__name__}_{kernel_id}"
            # Made in a launcher, it is launched: ``.launch(grid=...)``'s own
            # arguments are the DSL's types, which need not be read here.
            compiling["launches"].append((args, kwargs, (name,), self))
            return types.SimpleNamespace(launch=lambda *_, **__: None)
        compiling = {"next": 0, "launches": []}
        _DSL_COMPILING.append(compiling)
        try:
            self._func(*args, **kwargs)
        except (
            Exception
        ) as error:  # noqa: BLE001 - the kernels it made before stopping still launch
            logger.debug("%s: launcher stopped: %s", self.__name__, error)
        finally:
            _DSL_COMPILING.pop()
        ran = bool(compiling["launches"])
        if not ran:
            return self._launch(args, kwargs)
        for kernel_args, kernel_kwargs, kernels, kernel in compiling["launches"]:
            kernel._launch(kernel_args, kernel_kwargs, kernels)
        return None


# The DSL launchers being run, innermost last: each numbers the kernels it compiles.
_DSL_COMPILING: list = []


def _dsl_compiler_module(name: str, log: LaunchLog) -> types.ModuleType:
    module = _InertTail(name)

    def jit(fn=None, **options):
        if fn is None:
            return lambda function: DslKernel(function, log)
        return DslKernel(fn, log)

    def kernel(fn=None, **options):
        if fn is None:
            return lambda function: DslKernel(
                function, log, deferred=True, name=options.get("name")
            )
        return DslKernel(fn, log, deferred=True, name=options.get("name"))

    class Compile:
        """``compile(launcher, *args)``: compiles it and, handed arguments,
        runs it once as the DSL does; ``compile[hints]`` takes compile hints."""

        def __getitem__(self, hints):
            return self

        def __call__(self, fn, *args, **kwargs):
            compiled = fn if isinstance(fn, DslKernel) else DslKernel(fn, log)
            if args:
                compiled(*args)
            return compiled

    module.jit = jit
    module.kernel = kernel
    module.compile = Compile()
    module.CompiledFunction = DslKernel
    module.from_dlpack = module.from_torch_tensor = lambda tensor, *a, **k: _DslTensor(
        tensor
    )
    module.from_c_void_p = lambda *args, **kwargs: _DslTensor(None)
    return module


class _InertTail(types.ModuleType):
    """A module whose names beyond those it sets are not modelled."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        from TraceLens.ModelUtils.source_interpreter import Inert

        return Inert(f"{self.__name__}.{name}")


class _HeuristicKernel:
    """``@triton.heuristics``: constants computed from a launch's arguments."""

    def __init__(self, kernel, values):
        self.kernel = kernel
        self.values = dict(values)
        self.fn = getattr(kernel, "fn", kernel)
        self.arg_names = list(getattr(kernel, "arg_names", ()))
        self.__name__ = getattr(kernel, "__name__", "triton_kernel")

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            bound = {**dict(zip(self.arg_names, args)), **kwargs}
            for name, rule in self.values.items():
                kwargs[name] = bound[name] = rule(bound)
            return launch(*args, **kwargs)

        return run

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.kernel, name)


class _AstSource:
    """``triton.compiler.ASTSource``: a kernel and the constants it is compiled for."""

    def __init__(self, fn, signature=None, constexprs=None, attrs=None):
        self.fn = fn
        self.signature = signature
        self.constexprs = dict(constexprs or {})
        self.attrs = attrs


class _CompiledTritonKernel:
    """What ``triton.compile`` makes: launched with the kernel's other arguments."""

    def __init__(self, source):
        self._source = source
        self.name = getattr(source.fn, "__name__", "triton_kernel")

    def __getitem__(self, grid):
        kernel = self._source.fn
        if not isinstance(kernel, TritonKernel):
            return lambda *args, **kwargs: None
        constants = self._source.constexprs
        names = [name for name in kernel.arg_names if name not in constants]

        def launch(*args, **kwargs):
            return kernel[grid](**dict(zip(names, args)), **kwargs, **constants)

        return launch

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        from TraceLens.ModelUtils.source_interpreter import Inert

        return Inert(f"{self.name}.{name}")


def _triton_module(
    log: LaunchLog, device: DeviceFacts, version: str
) -> dict[str, types.ModuleType]:
    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")
    runtime = types.ModuleType("triton.runtime")

    def jit(fn=None, **options):
        if fn is None:
            return lambda function: TritonKernel(function, log, **options)
        return TritonKernel(fn, log, **options)

    def autotune(configs=None, key=None, **options):
        return lambda kernel: kernel

    def heuristics(values):
        return lambda kernel: _HeuristicKernel(kernel, values)

    class Config:
        def __init__(
            self,
            kwargs=None,
            num_warps=4,
            num_stages=2,
            num_ctas=1,
            maxnreg=None,
            pre_hook=None,
            **options,
        ):
            self.kwargs = dict(kwargs or {})
            self.num_warps = num_warps
            self.num_stages = num_stages
            self.num_ctas = num_ctas
            self.maxnreg = maxnreg
            self.pre_hook = pre_hook
            self.options = options

        def all_kwargs(self):
            return {
                **self.kwargs,
                **self.options,
                "num_warps": self.num_warps,
                "num_stages": self.num_stages,
                "num_ctas": self.num_ctas,
                "maxnreg": self.maxnreg,
            }

        def __repr__(self):
            return f"Config({self.all_kwargs()})"

    class constexpr:  # noqa: N801 - the name source code uses
        def __init__(self, value=None):
            self.value = value

        def __class_getitem__(cls, item):
            return cls

    def cdiv(x, y):
        return (x + y - 1) // y

    def next_power_of_2(n):
        return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()

    triton.jit = jit
    triton.autotune = autotune
    triton.heuristics = heuristics
    triton.Config = Config
    triton.cdiv = cdiv
    triton.next_power_of_2 = next_power_of_2
    triton.__version__ = version
    triton.language = language
    triton.runtime = runtime
    language.constexpr = constexpr
    language.tensor = object
    language.core = language
    for name in (
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint32",
        "float8e4nv",
        "float8e5",
        "float8e4b8",
        "float8e5b16",
        "float8e4b15",
    ):
        setattr(language, name, name)
    language.dtype = str
    driver = types.SimpleNamespace(
        active=types.SimpleNamespace(
            get_current_target=lambda: types.SimpleNamespace(
                backend="hip", arch=device.gfx_target, warp_size=device.warp_size
            ),
            utils=types.SimpleNamespace(
                get_device_properties=lambda index=0: {
                    "multiprocessor_count": device.compute_units,
                    "max_shared_mem": device.shared_memory_per_block,
                    "warpSize": device.warp_size,
                    "max_num_regs": device.registers_per_block,
                    "arch": device.gfx_target,
                }
            ),
            get_current_device=lambda: 0,
        )
    )
    runtime.driver = driver
    triton.runtime.jit = types.SimpleNamespace(JITFunction=TritonKernel)
    runtime.Autotuner = object
    runtime.Heuristics = object
    # Kernels compiled ahead of their launch, and Gluon's kernels: launched
    # like ``@triton.jit`` ones.
    compiler = _InertTail("triton.compiler")
    compiler.ASTSource = _AstSource
    triton.compile = lambda source, target=None, options=None: _CompiledTritonKernel(
        source
    )
    triton.compiler = compiler
    experimental = _InertTail("triton.experimental")
    gluon = _InertTail("triton.experimental.gluon")
    gluon_language = _InertTail("triton.experimental.gluon.language")
    gluon.jit = jit
    gluon_language.constexpr = constexpr
    gluon.language = gluon_language
    experimental.gluon = gluon
    triton.experimental = experimental
    return {
        "triton": triton,
        "triton.language": language,
        "triton.runtime": runtime,
        "triton.language.core": language,
        "triton.compiler": compiler,
        "triton.experimental": experimental,
        "triton.experimental.gluon": gluon,
        "triton.experimental.gluon.language": gluon_language,
    }


# ------------------------------------------------------------------ run


class TorchRuntime:
    """The torch, GPU and kernel libraries an interpreted run sees.

    ``modules`` are the host modules to hand
    :class:`~TraceLens.ModelUtils.source_interpreter.SourceInterpreter`.
    Use the runtime as a context manager around the interpretation: GPU
    tensors are meta tensors and recorded launches go to ``log`` only inside.
    """

    def __init__(
        self, device: DeviceFacts, world_size: int = 1, triton_version: str = ""
    ):
        # These tables are read from torch itself, outside the run's modes.
        profiler_type_names()
        scalar_type_codes()
        enum_codes()
        self.device = device
        self.triton_version = triton_version
        self.shadows = Shadows()
        self.memory = DeviceMemory()
        # The sources the run wrote, by path (set by whoever runs the source).
        self.generated_sources: Callable[[], dict] | None = None
        # The checkpoint file a weight file path stands for, if any.
        self.checkpoint_for: Callable[[object], object] | None = None
        self.log = LaunchLog(shadows=self.shadows)
        self.registry = OpRegistry(self.log)
        self.modules = self._modules(world_size)
        self._stack = contextlib.ExitStack()

    def _modules(self, world_size) -> dict[str, types.ModuleType]:
        proxy = types.ModuleType("torch")
        proxy.__dict__.update(
            {key: value for key, value in vars(torch).items() if key != "__builtins__"}
        )
        cuda = _cuda_module(self.device, self.log, self.memory)
        host_load = torch.load

        def load(f, *args, **kwargs):
            # A weight file of the run holds no weights, only the entries the
            # run states it holds: shapes come from the model.
            checkpoint = (
                self.checkpoint_for(f) if self.checkpoint_for is not None else None
            )
            if checkpoint is None:
                return host_load(f, *args, **kwargs)
            return dict(checkpoint.tensors)

        proxy.load = load
        host_from_dlpack = torch.from_dlpack

        def from_dlpack(external, *args, **kwargs):
            # A tensor a native function returned is already the run's tensor.
            if isinstance(external, torch.Tensor):
                return external
            return host_from_dlpack(external, *args, **kwargs)

        proxy.from_dlpack = from_dlpack
        distributed = _distributed_module(world_size, self.log)
        library = _library_module(self.registry)
        ops = _Ops(self.registry)
        version = types.SimpleNamespace(
            __version__=torch.__version__,
            hip=self.device.hip_version or "7.0",
            cuda=None,
            git_version="",
            debug=False,
        )
        proxy.cuda = cuda
        proxy.distributed = distributed
        proxy.library = library
        proxy.ops = ops
        proxy.version = version
        proxy.compile = _identity_decorator

        def get_device_module(device=None):
            kind = (
                device.type
                if isinstance(device, torch.device)
                else str(device or "cuda")
            )
            if kind.split(":")[0] in _GPU_DEVICE_TYPES:
                return cuda
            return torch.get_device_module(device)

        proxy.get_device_module = get_device_module

        class Generator(torch.Generator):
            """A generator on the GPU is a CPU one: random values never reach a shape."""

            def __new__(cls, device=None):
                kind = (
                    device.type
                    if isinstance(device, torch.device)
                    else str(device or "cpu")
                )
                if kind.split(":")[0] in (*_GPU_DEVICE_TYPES, "meta"):
                    return torch.Generator()
                return torch.Generator(device)

        proxy.Generator = Generator
        # Compilers would compile the interpreted functions; they run as is.
        jit = _ModelModule("torch.jit", torch.jit)
        for name in (
            "script",
            "ignore",
            "unused",
            "export",
            "trace",
            "script_if_tracing",
        ):
            setattr(jit, name, _identity_decorator)
        jit.is_scripting = lambda: False
        jit.is_tracing = lambda: False
        proxy.jit = jit
        proxy.accelerator = types.SimpleNamespace(
            current_accelerator=lambda *args, **kwargs: torch.device("cuda"),
            is_available=lambda: True,
            device_count=cuda.device_count,
            current_device_index=lambda: 0,
            set_device_index=lambda index: None,
            synchronize=lambda *args, **kwargs: None,
            current_stream=cuda.current_stream,
        )
        dynamo = _ModelModule("torch._dynamo", torch._dynamo)
        dynamo.disable = _identity_decorator
        dynamo.mark_dynamic = lambda *args, **kwargs: None
        dynamo.mark_static = lambda *args, **kwargs: None
        dynamo.is_compiling = lambda: False
        dynamo.config = types.SimpleNamespace()
        dynamo.allow_in_graph = _identity_decorator
        compiler = _ModelModule("torch.compiler", torch.compiler)
        compiler.is_compiling = lambda: False
        compiler.is_dynamo_compiling = lambda: False
        compiler.disable = _identity_decorator
        compiler.allow_in_graph = _identity_decorator
        proxy._dynamo = dynamo
        proxy.compiler = compiler
        modules = {
            "torch": proxy,
            "torch.cuda": cuda,
            "torch.distributed": distributed,
            "torch.library": library,
            "torch._dynamo": dynamo,
            "torch.compiler": compiler,
            "torch.jit": proxy.jit,
            "torch.version": version,
        }
        modules.update(_triton_module(self.log, self.device, self.triton_version))
        for name in DSL_COMPILERS:
            modules[name] = _dsl_compiler_module(name, self.log)
        return modules

    def native_module(self, name, index=None, host=None, version=None) -> NativeModule:
        """The stand-in for compiled extension ``name``, declared by ``index``.

        ``host`` evaluates the extension's scalar host functions by name, and
        ``version`` is that of the distribution it ships in.
        """
        generated = None
        if self.generated_sources is not None:
            # What the build wrote is read now: it may clear its staging after
            # loading the module, before any function is called.
            snapshot = {
                path: text
                for path, text in self.generated_sources().items()
                if isinstance(text, str)
                and path.endswith((*_GENERATED_SOURCES, ".ninja", ".h", ".cuh"))
            }
            generated = lambda: snapshot  # noqa: E731
        module = NativeModule(name, self.log, index, host, generated)
        module.memory = self.memory
        if version is not None:
            module.__version__ = version
        return module

    def register_native_ops(self, index, host=None) -> None:
        """Make the ops a native library's C++ registers callable as launches.

        Each ``namespace::name(schema)`` is an op whose call records one
        launch with its schema-ordered arguments. An output the schema marks
        as written into an input (``Tensor(a!)``) is that input; any other
        output is left to a fake kernel the Python side registers. ``host``
        evaluates the C++ of an op bound to a function that launches no kernel.
        """
        for (namespace, name), text in index.op_schemas.items():
            try:
                schema = torch._C.parse_schema(f"{namespace}::{text}")
            except RuntimeError:
                continue
            op = self.registry.get(namespace, name)
            op.schema = text
            function = index.op_functions.get((namespace, name))
            if function is not None and index.launches.get(function):
                function = None
            op.impls.setdefault("CUDA", _native_op(self.log, schema, host, function))

    def __enter__(self):
        self._stack.enter_context(_GpuAsMeta(self.shadows, self.memory))
        self._stack.enter_context(_Recorder(self.log, self.shadows, self.memory))
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


def _identity_decorator(fn=None, *args, **kwargs):
    if callable(fn):
        return fn
    return lambda function: function
