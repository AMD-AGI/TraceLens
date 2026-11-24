###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run Python source by walking its syntax tree, without executing the source.

The served model's behaviour is in source code: the checkpoint's modeling
file, the inference engine, and the kernel libraries the engine calls. That
code is read here, statement by statement, from checkouts on disk. Nothing in
those checkouts is imported, compiled, or executed by Python itself: every
function body, class body and module body is evaluated by this module from its
``ast``.

The objects the source builds are ordinary Python objects. A ``class`` in the
source becomes a real class whose methods are :class:`SourceFunction` objects,
so the standard library (``dataclasses``, ``enum``, ``functools``,
``contextlib``) works on them unchanged. Only modules on an explicit allow
list (the parts of the standard library with no side effects, and the host
packages a caller names, such as ``torch``) are real modules. Everything the
source asks of the outside world goes through :class:`Environment`: the
environment variables it reads, the files it opens, and any import that has
no source here.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import logging
import operator
import os
import re
import sys
import types
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Standard-library modules that compute without touching the world. The
# source gets the real module.
_PURE_STDLIB = frozenset(
    {
        "__future__",
        "abc",
        "argparse",
        "array",
        "ast",
        "base64",
        "binascii",
        "bisect",
        "builtins",
        "calendar",
        "cmath",
        "codecs",
        "collections",
        "collections.abc",
        "colorsys",
        "contextlib",
        "contextvars",
        "copy",
        "copyreg",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "difflib",
        "dis",
        "enum",
        "errno",
        "fnmatch",
        "fractions",
        "functools",
        "gc",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "importlib.abc",
        "importlib.machinery",
        "importlib.metadata",
        "importlib.resources",
        "inspect",
        "io",
        "itertools",
        "json",
        "keyword",
        "linecache",
        "locale",
        "logging",
        "logging.handlers",
        "math",
        "numbers",
        "operator",
        "pathlib",
        "glob",
        "platform",
        "sysconfig",
        "urllib",
        "urllib.parse",
        "ipaddress",
        "pickle",
        "pprint",
        "queue",
        "random",
        "re",
        "reprlib",
        "secrets",
        "shlex",
        "signal",
        "statistics",
        "string",
        "struct",
        "textwrap",
        "threading",
        "time",
        "timeit",
        "token",
        "tokenize",
        "traceback",
        "types",
        "typing",
        "unicodedata",
        "unittest",
        "unittest.mock",
        "uuid",
        "warnings",
        "weakref",
        "zlib",
    }
)

logger = logging.getLogger(__name__)


# Standard-library modules that reach the machine: processes, sockets,
# native libraries, the file system's writable side. The source gets an inert
# stand-in, so a module that only names them still loads.
_INERT_STDLIB = frozenset(
    {
        "asyncio",
        "atexit",
        "concurrent",
        "concurrent.futures",
        "faulthandler",
        "fcntl",
        "getpass",
        "grp",
        "http",
        "http.client",
        "http.server",
        "mmap",
        "multiprocessing",
        "multiprocessing.connection",
        "multiprocessing.pool",
        "multiprocessing.shared_memory",
        "multiprocessing.reduction",
        "pwd",
        "pty",
        "resource",
        "select",
        "selectors",
        "signal",
        "socket",
        "socketserver",
        "sqlite3",
        "ssl",
        "subprocess",
        "tempfile",
        "termios",
        "tracemalloc",
        "tty",
        "urllib.request",
        "webbrowser",
        "zipfile",
        "tarfile",
    }
)


import collections

# Calls into stand-ins, by the name the source reached them under.
INERT_CALLS: collections.Counter = collections.Counter()


class _InertParameters(Mapping):
    """The parameters of a stand-in: whichever is asked for is one."""

    def __init__(self, name):
        self._name = name

    def __getitem__(self, key):
        return Inert(f"{self._name}.parameters[{key!r}]")

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class _InertSignature(inspect.Signature):
    """What ``inspect.signature`` finds on a stand-in: nothing it can say."""

    def __init__(self, name):
        super().__init__()
        self._name = name

    @property
    def parameters(self):
        return _InertParameters(self._name)


class _SignatureOfInert:
    def __get__(self, instance, owner=None):
        return None if instance is None else _InertSignature(instance._name)


class Inert:
    """A stand-in for something the source names but the run model omits.

    Every attribute and call gives another stand-in, it can be a base class,
    and it is falsy, so code that only sets things up around it proceeds.
    Each call is counted in :data:`INERT_CALLS`, so a launch that went
    nowhere shows up instead of vanishing.
    """

    # Code that reads a signature (to test a library's version) gets one
    # whose parameters are stand-ins too.
    __signature__ = _SignatureOfInert()

    def __init__(self, name="inert"):
        object.__setattr__(self, "_name", name)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return Inert(f"{self._name}.{name}")

    def __call__(self, *args, **kwargs):
        INERT_CALLS[self._name] += 1
        if (
            self._name.endswith(")")
            and len(args) == 1
            and not kwargs
            and isinstance(args[0], type)
        ):
            # What a decorator factory (``@registry.register(key)``) makes
            # registers the class it is handed. A function decorator may
            # compile its function, so stays inert.
            return args[0]
        return Inert(f"{self._name}()")

    def __mro_entries__(self, bases):
        others = [base for base in bases if isinstance(base, type)]
        if any(type(base) is not type for base in others):
            # Another base brings its own metaclass; this one adds nothing.
            return ()
        return (InertBase,)

    def __bool__(self):
        return False

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __repr__(self):
        return f"<inert {self._name}>"

    def _combine(self, *_args):
        return Inert(self._name)

    # Arithmetic on a stand-in (flags or-ed together, sizes added) gives one.
    __or__ = __ror__ = __and__ = __rand__ = __xor__ = __rxor__ = _combine
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _combine
    __truediv__ = __rtruediv__ = __floordiv__ = __rfloordiv__ = _combine
    __mod__ = __rmod__ = __pow__ = __rpow__ = __neg__ = __invert__ = _combine
    __lshift__ = __rlshift__ = __rshift__ = __rrshift__ = _combine

    def __getitem__(self, key):
        return Inert(f"{self._name}[]")

    def _never(self, other):
        return False

    # An ordering against a stand-in holds for no value.
    __lt__ = __le__ = __gt__ = __ge__ = _never

    def __int__(self):
        return 0

    __index__ = __int__

    def __float__(self):
        return 0.0

    def __hash__(self):
        return id(self)


def _functools_module(interpreter) -> types.ModuleType:
    """``functools``, whose caches keep what each value was computed from.

    A cached helper's later calls run nothing, so the settings its value
    came from are remembered with the value and reach every caller.
    """
    import functools

    module = types.ModuleType("functools")
    module.__dict__.update(
        {key: value for key, value in vars(functools).items() if key != "__builtins__"}
    )

    class _Cached:
        def __init__(self, function, maxsize=128):
            self.__wrapped__ = function
            self._maxsize = maxsize
            self._values: dict = {}
            functools.update_wrapper(self, function)

        def __call__(self, *args, **kwargs):
            try:
                key = (args, tuple(sorted(kwargs.items())))
                hash(key)
            except TypeError:
                return self.__wrapped__(*args, **kwargs)
            if key not in self._values:
                interpreter._return_taint = _NO_SETTINGS
                value = self.__wrapped__(*args, **kwargs)
                if self._maxsize is not None and len(self._values) >= self._maxsize > 0:
                    self._values.pop(next(iter(self._values)))
                self._values[key] = (value, interpreter._return_taint)
            value, taint = self._values[key]
            interpreter._note(taint)
            return value

        def __get__(self, instance, owner=None):
            return self if instance is None else functools.partial(self, instance)

        def cache_clear(self):
            self._values.clear()

        def cache_info(self):
            return functools._CacheInfo(0, 0, self._maxsize, len(self._values))

    def lru_cache(maxsize=128, typed=False):
        if callable(maxsize) and not isinstance(maxsize, bool):
            return _Cached(maxsize)
        return lambda function: _Cached(function, maxsize)

    module.lru_cache = lru_cache
    module.cache = lambda function: _Cached(function, None)
    return module


def _argparse_module(environment: Environment) -> types.ModuleType:
    """``argparse``, noting each option a parser declares in ``environment.options``."""
    import argparse

    module = types.ModuleType("argparse")
    module.__dict__.update(
        {key: value for key, value in vars(argparse).items() if key != "__builtins__"}
    )

    class ArgumentParser(argparse.ArgumentParser):
        def parse_known_args(self, *args, **kwargs):
            # Every option, its groups' too, is the parser's by now.
            for action in self._actions:
                long = [
                    option
                    for option in action.option_strings
                    if option.startswith("--")
                ]
                if long:
                    environment.options.setdefault(
                        action.dest, (long[0], action.nargs == 0)
                    )
                    if action.default is None or isinstance(
                        action.default, bool | int | float | str
                    ):
                        environment.option_defaults.setdefault(
                            action.dest, action.default
                        )
                    if action.choices is not None:
                        environment.option_choices.setdefault(
                            action.dest, tuple(str(item) for item in action.choices)
                        )
                    if action.type in (int, float):
                        environment.option_numbers.add(action.dest)
            return super().parse_known_args(*args, **kwargs)

    module.ArgumentParser = ArgumentParser
    return module


def _shutil_module(environment: Environment) -> types.ModuleType:
    """``shutil`` on the run's machine: it finds the machine's tools, changes nothing."""
    import shutil

    module = types.ModuleType("shutil")

    def refused(*_args, **_kwargs):
        raise PermissionError("the interpreted source may not change the machine")

    def which(command, mode=None, path=None):
        command = os.fspath(command)
        if os.path.isabs(command):
            return command if environment.virtual_file(command) is not None else None
        search = path or environment.getenv("PATH") or ""
        for directory in str(search).split(os.pathsep):
            candidate = os.path.join(directory, command)
            if directory and environment.virtual_file(candidate) is not None:
                return candidate
        return None

    module.which = which
    module.disk_usage = shutil.disk_usage
    module.get_terminal_size = lambda *args, **kwargs: os.terminal_size((80, 24))
    # Copies stage a library's own build inputs; only the build reads them.
    for name in ("copy", "copy2", "copyfile", "copytree"):
        setattr(module, name, lambda source, target, *args, **kwargs: target)
    module.rmtree = lambda path, ignore_errors=False, *args, **kwargs: (
        environment.fs.remove_tree(path, ignore_errors)
    )
    module.move = lambda source, target, *args, **kwargs: environment.fs.move_tree(
        source, target
    )
    module.chown = refused
    module.Error = shutil.Error
    return module


def _ctypes_modules(interpreter) -> dict[str, types.ModuleType]:
    """``ctypes`` whose shared libraries are the run's native stand-ins.

    Declaring structures and types is ordinary Python; loading a library is
    the native boundary, so a loaded library is
    :attr:`SourceInterpreter.native_module_factory`'s module for its name.
    """
    import ctypes
    import ctypes.util

    module = types.ModuleType("ctypes")
    module.__dict__.update(
        {key: value for key, value in vars(ctypes).items() if not key.startswith("__")}
    )
    util = types.ModuleType("ctypes.util")
    util.find_library = lambda name: f"lib{name}.so"

    def load(name, *args, **kwargs):
        stem = os.path.basename(str(name)).split(".")[0].removeprefix("lib")
        factory = interpreter.native_module_factory
        native = factory(stem) if factory is not None else None
        if native is None:
            raise OSError(f"{name}: cannot load a native library")
        native.c_abi = True
        return native

    loader = types.SimpleNamespace(LoadLibrary=load)
    for name in ("CDLL", "PyDLL", "WinDLL", "OleDLL"):
        setattr(module, name, load)
    module.cdll = module.pydll = loader
    module.util = util
    return {"ctypes": module, "ctypes.util": util}


class _InertMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return Inert(f"{cls.__name__}.{name}")


class InertBase(metaclass=_InertMeta):
    """The base of a class whose real base the run model omits.

    What the omitted base would have provided reads as a stand-in.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return Inert(f"{type(self).__name__}.{name}")


def _threading_module() -> types.ModuleType:
    """``threading`` whose threads never start.

    A forward runs on one thread; what the source starts beside it
    (watchdogs, pollers, background loaders) is service work, and an
    interpreted body running concurrently would share this interpreter.
    Locks, events and conditions are real.
    """
    import threading

    module = types.ModuleType("threading")
    module.__dict__.update(
        {
            key: value
            for key, value in vars(threading).items()
            if not key.startswith("__")
        }
    )

    class Thread(threading.Thread):
        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

    class Timer(Thread):
        def __init__(self, interval, function, args=None, kwargs=None):
            super().__init__()

        def cancel(self):
            pass

    module.Thread = Thread
    module.Timer = Timer
    return module


def _pkgutil_module(interpreter) -> types.ModuleType:
    """``pkgutil`` listing the modules of the readable source directories."""
    import pkgutil

    module = types.ModuleType("pkgutil")
    module.__dict__.update(
        {key: value for key, value in vars(pkgutil).items() if not key.startswith("__")}
    )
    environment = interpreter.environment

    def iter_modules(path=None, prefix=""):
        for directory in path or ():
            directory = Path(str(directory))
            if not environment.may_read(directory) or not directory.is_dir():
                continue
            for entry in sorted(directory.iterdir()):
                if entry.suffix == ".py" and entry.stem != "__init__":
                    yield pkgutil.ModuleInfo(None, prefix + entry.stem, False)
                elif entry.is_dir() and (entry / "__init__.py").is_file():
                    yield pkgutil.ModuleInfo(None, prefix + entry.name, True)

    def walk_packages(path=None, prefix="", onerror=None):
        for info in iter_modules(path, prefix):
            yield info
            if info.ispkg:
                try:
                    package = interpreter.import_module(info.name)
                except Exception:  # noqa: BLE001 - pkgutil reports through onerror
                    if onerror is not None:
                        onerror(info.name)
                    continue
                yield from walk_packages(
                    getattr(package, "__path__", None) or (), info.name + ".", onerror
                )

    module.iter_modules = iter_modules
    module.walk_packages = walk_packages
    return module


_EXTENSION_SUFFIXES = (".so", ".pyd", ".dll")


def _is_extension_file(path: str) -> bool:
    return any(path.endswith(suffix) for suffix in _EXTENSION_SUFFIXES)


def _glob_module(environment: Environment) -> types.ModuleType:
    """``glob`` over the readable checkouts and the run machine's files."""
    import fnmatch
    import glob

    module = types.ModuleType("glob")
    module.__dict__.update(
        {key: value for key, value in vars(glob).items() if not key.startswith("__")}
    )

    def matches(pathname, *args, recursive=False, **kwargs):
        found = []
        directory = os.path.dirname(os.fspath(pathname))
        if environment.may_read(directory or "."):
            found = glob.glob(pathname, recursive=recursive)
        found += [
            path
            for path in environment.files
            if fnmatch.fnmatch(path, os.fspath(pathname))
        ]
        return sorted(set(found))

    module.glob = matches
    module.iglob = lambda pathname, *args, **kwargs: iter(
        matches(pathname, *args, **kwargs)
    )
    return module


def _json_module() -> types.ModuleType:
    """``json`` that writes a stand-in as its name instead of failing."""
    import json

    module = types.ModuleType("json")
    module.__dict__.update(
        {key: value for key, value in vars(json).items() if not key.startswith("__")}
    )

    def fallback(value):
        if isinstance(value, Inert):
            return repr(value)
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )

    module.dumps = lambda obj, *args, default=None, **kwargs: json.dumps(
        obj, *args, default=default or fallback, **kwargs
    )
    module.dump = lambda obj, fp, *args, default=None, **kwargs: json.dump(
        obj, fp, *args, default=default or fallback, **kwargs
    )
    return module


# A module's global naming the inert module it star-imported.
_INERT_STAR = "__tracelens_inert_star__"


class InertModule(types.ModuleType):
    """A module the source requires that the run model does not provide."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        value = Inert(f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


_PIN = re.compile(r"^\s*\"?([A-Za-z0-9_.\-]+)\s*==\s*([A-Za-z0-9_.+\-]+)")


def requirement_pins(roots: Iterable[Path]) -> dict[str, str]:
    """``{distribution: version}`` pinned with ``==`` in the roots' requirements.

    Read from ``requirements*.txt`` and ``pyproject.toml`` at each root's top
    level, the files a build installs its dependencies from. Distribution
    names are normalized to their import spelling (``-`` and ``.`` to ``_``).
    """
    pins: dict[str, str] = {}
    for root in roots:
        root = Path(root)
        for path in [*root.glob("requirements*.txt"), root / "pyproject.toml"]:
            if not path.is_file():
                continue
            for line in path.read_text(errors="ignore").splitlines():
                match = _PIN.match(line.strip().strip(","))
                if match:
                    name = re.sub(r"[-.]", "_", match.group(1)).lower()
                    pins.setdefault(name, match.group(2))
    return pins


class SourceUnavailable(ImportError):
    """A module the run imports has no source here and is not a host module."""


class InterpreterError(RuntimeError):
    """The source did something this interpreter does not model."""


# ``break``/``continue`` leave a block as a value its statements hand back,
# not an exception: they run once per loop iteration.
_BREAK = "break"
_CONTINUE = "continue"


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Return(Exception):
    def __init__(self, value, taint=frozenset()):
        self.value = value
        self.taint = taint


@dataclass
class Environment:
    """What the source can see of the world it runs in.

    ``settings`` are the environment variables the run is known to have set.
    Every variable the source reads is recorded in ``reads`` with the value it
    got, ``None`` when unset, so a caller can see which unknown settings the
    run depended on. ``readable`` are the directories whose files the source
    may open (its own checkouts).
    """

    settings: dict[str, str] = field(default_factory=dict)
    readable: tuple[Path, ...] = ()
    reads: dict[str, str | None] = field(default_factory=dict)
    # Files of the machine the run used, by absolute path: what the trace
    # says about its installation (for instance the HIP version file).
    files: dict[str, str] = field(default_factory=dict)
    # Command-line options the source declares: destination -> (option, and
    # whether it takes no value: a switch).
    options: dict[str, tuple[str, bool]] = field(default_factory=dict)
    # Destination -> the option's default, where it is a plain value, and the
    # values it accepts, where it names them.
    option_defaults: dict[str, object] = field(default_factory=dict)
    option_choices: dict[str, tuple] = field(default_factory=dict)
    # Destinations whose option takes a number.
    option_numbers: set[str] = field(default_factory=set)
    # Called with the name of each variable the source reads.
    on_read: Callable[[str], None] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def fs(self):
        from TraceLens.ModelUtils.sandbox_fs import ScratchFS

        scratch = self.__dict__.get("_fs")
        if scratch is None:
            scratch = self.__dict__["_fs"] = ScratchFS(self.files, self.may_read)
        return scratch

    def virtual_file(self, path) -> str | None:
        try:
            return self.files.get(os.path.normpath(os.fspath(path)))
        except TypeError:
            return None

    def virtual_dir(self, path) -> bool:
        try:
            prefix = os.path.normpath(os.fspath(path)).rstrip("/") + "/"
        except TypeError:
            return False
        return any(name.startswith(prefix) for name in self.files)

    def getenv(self, name, default=None):
        value = self.settings.get(name)
        self.reads.setdefault(name, value)
        if self.on_read is not None:
            self.on_read(name)
        return default if value is None else value

    def may_read(self, path) -> bool:
        try:
            resolved = Path(path).resolve()
        except (OSError, TypeError, ValueError):
            return False
        return any(resolved.is_relative_to(root) for root in self.readable)


class _Environ(dict):
    """``os.environ`` as the source sees it: the run's known settings only."""

    def __init__(self, environment: Environment):
        super().__init__()
        self._environment = environment

    def __getitem__(self, name):
        value = self._environment.getenv(name)
        if value is None:
            raise KeyError(name)
        return value

    def get(self, name, default=None):
        return self._environment.getenv(name, default)

    def __contains__(self, name):
        return self._environment.getenv(name) is not None

    def __setitem__(self, name, value):
        self._environment.settings[name] = str(value)

    def setdefault(self, name, default=None):
        if self._environment.getenv(name) is None and default is not None:
            self._environment.settings[name] = str(default)
        return self._environment.getenv(name)

    def pop(self, name, *default):
        value = self._environment.settings.pop(name, None)
        if value is None:
            if default:
                return default[0]
            raise KeyError(name)
        return value

    def __delitem__(self, name):
        self._environment.settings.pop(name, None)

    def keys(self):
        return self._environment.settings.keys()

    def items(self):
        return self._environment.settings.items()

    def __iter__(self):
        return iter(self._environment.settings)

    def __len__(self):
        return len(self._environment.settings)

    def copy(self):
        return dict(self._environment.settings)


def _sandboxed_os(environment: Environment) -> types.ModuleType:
    """``os`` with the run's environment and no way to change the machine."""
    module = types.ModuleType("os")
    for name in dir(os):
        if not name.startswith("__"):
            setattr(module, name, getattr(os, name))
    environ = _Environ(environment)
    module.environ = environ
    module.getenv = environment.getenv
    module.putenv = lambda name, value: environ.__setitem__(name, value)
    module.unsetenv = lambda name: environ.pop(name, None)

    def refused(*_args, **_kwargs):
        raise PermissionError("the interpreted source may not change the machine")

    for name in (
        "system",
        "popen",
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
        "rename",
        "replace",
        "chmod",
        "chown",
        "kill",
        "killpg",
        "fork",
        "forkpty",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "spawnv",
        "spawnve",
        "symlink",
        "link",
        "truncate",
        "write",
    ):
        if hasattr(module, name):
            setattr(module, name, refused)
    module.makedirs = lambda *args, **kwargs: None
    module.mkdir = lambda *args, **kwargs: None
    module.getpid = lambda: 1
    path = types.ModuleType("os.path")
    for name in dir(os.path):
        if not name.startswith("__"):
            setattr(path, name, getattr(os.path, name))
    for check in ("exists", "isfile", "isdir", "lexists"):
        real = getattr(os.path, check)
        is_dir = check == "isdir"
        is_file = check == "isfile"

        def checked(value, _real=real, _is_dir=is_dir, _is_file=is_file):
            if environment.virtual_file(value) is not None:
                return not _is_dir
            if environment.virtual_dir(value):
                return not _is_file
            return environment.may_read(value) and _real(value)

        setattr(path, check, checked)
    module.path = path
    environment.fs.install_os(module)
    return module


def _sandboxed_open(environment: Environment):
    return environment.fs.open


class _Cell:
    """A variable a nested function shares with the function that defines it."""

    __slots__ = ("value", "bound")

    def __init__(self):
        self.value = None
        self.bound = False


class _Frame:
    """One activation: its own variables, and the cells of enclosing ones."""

    __slots__ = (
        "variables",
        "module",
        "enclosing",
        "declared_global",
        "declared_nonlocal",
        "function",
        "is_class",
        "class_cell",
        "import_guards",
        "taint",
    )

    def __init__(self, module, enclosing=None, function=None, is_class=False):
        self.variables: dict[str, Any] = {}
        self.module = module
        self.enclosing = enclosing
        self.declared_global: set[str] = set()
        self.declared_nonlocal: set[str] = set()
        self.function = function
        self.is_class = is_class
        self.class_cell = None
        # ``try`` blocks of this body, around the statement now running,
        # whose handlers catch an ImportError.
        self.import_guards = 0
        # Variable -> the settings its value was computed from (see
        # :meth:`SourceInterpreter.control_settings`).
        self.taint: dict[str, frozenset] | None = None


_MISSING = object()


def _function_scope(frame):
    """The nearest enclosing frame that is not a class body."""
    while frame is not None and frame.is_class:
        frame = frame.enclosing
    return frame


class SourceFunction:
    """A function defined in interpreted source.

    It is a descriptor like a Python function, so it binds as a method,
    and ``functools.wraps``, ``inspect.signature`` and ``dataclasses`` treat
    it like one.
    """

    # What the interpreter needs lives in slots, not ``__dict__``:
    # ``functools.wraps`` copies the wrapped function's ``__dict__`` onto the
    # wrapper, which must keep its own body.
    __slots__ = (
        "_interpreter",
        "_node",
        "_frame",
        "_defaults",
        "_kw_defaults",
        "_class_cell",
        "_is_generator",
        "__dict__",
        "__weakref__",
    )

    def __init__(self, interpreter, node, frame, defaults, kw_defaults, name):
        self._interpreter = interpreter
        self._node = node
        self._frame = frame  # enclosing frame (None for module level)
        self._defaults = defaults
        self._kw_defaults = kw_defaults
        self.__name__ = name
        self.__qualname__ = name
        self.__module__ = frame.module.__name__
        self.__doc__ = None if isinstance(node, ast.Lambda) else ast.get_docstring(node)
        self.__defaults__ = tuple(defaults) or None
        self.__kwdefaults__ = kw_defaults or None
        self.__annotations__ = {}
        self.__type_params__ = ()
        self._class_cell = None
        self._is_generator = _contains_yield(node)

    @property
    def __signature__(self):
        import inspect

        assigned = self.__dict__.get("_assigned_signature")
        if assigned is not None:
            return assigned
        args = self._node.args
        annotations = self.__annotations__
        empty = inspect.Parameter.empty
        parameters = []
        positional = args.posonlyargs + args.args
        first_default = len(positional) - len(self._defaults)
        for index, argument in enumerate(positional):
            kind = (
                inspect.Parameter.POSITIONAL_ONLY
                if index < len(args.posonlyargs)
                else inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
            default = (
                self._defaults[index - first_default]
                if index >= first_default
                else inspect.Parameter.empty
            )
            parameters.append(
                inspect.Parameter(
                    argument.arg,
                    kind,
                    default=default,
                    annotation=annotations.get(argument.arg, empty),
                )
            )
        if args.vararg:
            parameters.append(
                inspect.Parameter(
                    args.vararg.arg,
                    inspect.Parameter.VAR_POSITIONAL,
                    annotation=annotations.get(args.vararg.arg, empty),
                )
            )
        for argument in args.kwonlyargs:
            default = self._kw_defaults.get(argument.arg, inspect.Parameter.empty)
            parameters.append(
                inspect.Parameter(
                    argument.arg,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotations.get(argument.arg, empty),
                )
            )
        if args.kwarg:
            parameters.append(
                inspect.Parameter(
                    args.kwarg.arg,
                    inspect.Parameter.VAR_KEYWORD,
                    annotation=annotations.get(args.kwarg.arg, empty),
                )
            )
        return inspect.Signature(
            parameters,
            return_annotation=annotations.get("return", inspect.Signature.empty),
        )

    @__signature__.setter
    def __signature__(self, value):
        self.__dict__["_assigned_signature"] = value

    @property
    def __globals__(self):
        return self._frame.module.__dict__

    @property
    def __closure__(self):
        return None

    @property
    def __builtins__(self):
        return self._interpreter.builtins

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return types.MethodType(self, instance)

    def __repr__(self):
        return f"<source function {self.__qualname__}>"

    def __call__(self, *args, **kwargs):
        return self._interpreter.call_function(self, args, kwargs)


_NO_SETTINGS: frozenset = frozenset()


def _assigned_names(statements) -> tuple[str, ...]:
    """The local names a block of statements assigns (not in nested scopes)."""
    if not statements:
        return ()
    found = getattr(statements[0], "_assigned_names", None)
    if found is None:
        names: dict[str, None] = {}
        pending = list(statements)
        while pending:
            node = pending.pop()
            if isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
            ):
                continue
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names[node.id] = None
            pending.extend(ast.iter_child_nodes(node))
        found = tuple(names)
        statements[0]._assigned_names = found
    return found


def _exits(statements) -> bool:
    return bool(statements) and isinstance(
        statements[-1], ast.Return | ast.Raise | ast.Continue | ast.Break
    )


def _interpreted(function) -> bool:
    """Whether calling ``function`` runs interpreted source (and binds its parameters)."""
    if isinstance(function, SourceFunction) or isinstance(
        getattr(function, "__func__", None), SourceFunction
    ):
        return True
    return isinstance(function, type) and isinstance(
        getattr(function, "__init__", None), SourceFunction
    )


def _contains_yield(node) -> bool:
    if isinstance(node, ast.Lambda):
        return False

    def walk(children):
        for child in children:
            if isinstance(child, ast.Yield | ast.YieldFrom):
                return True
            if isinstance(
                child,
                ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef,
            ):
                continue
            if walk(ast.iter_child_nodes(child)):
                return True
        return False

    return walk(node.body)


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.MatMult: operator.matmul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
}
_INPLACE = {
    ast.Add: operator.iadd,
    ast.Sub: operator.isub,
    ast.Mult: operator.imul,
    ast.MatMult: operator.imatmul,
    ast.Div: operator.itruediv,
    ast.FloorDiv: operator.ifloordiv,
    ast.Mod: operator.imod,
    ast.Pow: operator.ipow,
    ast.LShift: operator.ilshift,
    ast.RShift: operator.irshift,
    ast.BitOr: operator.ior,
    ast.BitXor: operator.ixor,
    ast.BitAnd: operator.iand,
}
_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


@dataclass
class _Location:
    file: str
    line: int
    function: str
    globals: dict = field(default_factory=dict, repr=False)
    locals: dict = field(default_factory=dict, repr=False)
    parameters: list = field(default_factory=list, repr=False)
    # The values the call was made with, positional then keyword, and the
    # names of the keyword ones.
    passed: tuple = field(default=(), repr=False)
    keywords: tuple = field(default=(), repr=False)
    frame: Any = field(default=None, repr=False)


class _FrameView:
    """What ``sys._getframe`` returns: one interpreted activation."""

    def __init__(self, stack: list[_Location], index: int):
        location = stack[index]
        self._stack = stack
        self._index = index
        self.f_globals = location.globals
        self.f_locals = location.locals
        self.f_lineno = location.line
        self.f_code = types.SimpleNamespace(
            co_name=location.function.rpartition(".")[2],
            co_qualname=location.function,
            co_filename=location.file,
            co_firstlineno=location.line,
        )

    @property
    def f_back(self):
        return _FrameView(self._stack, self._index - 1) if self._index > 0 else None


class SourceInterpreter:
    """Evaluates modules from source roots, one statement at a time.

    ``roots`` maps a top-level package name to the directory that holds it
    (``{"sglang": Path("checkout/python")}``). ``host_modules`` are real
    modules handed to the source in place of anything with that name: the
    caller's model of ``torch``, for instance. A module neither in a root nor
    a host module nor pure standard library raises :class:`SourceUnavailable`,
    which is an ``ImportError``, so the source's own fallback for a missing
    optional package runs.
    """

    def __init__(
        self,
        roots: Mapping[str, Path],
        environment: Environment | None = None,
        host_modules: Mapping[str, types.ModuleType] | None = None,
        host_packages: Iterable[str] = (),
    ):
        self.roots = {name: Path(path) for name, path in roots.items()}
        self.environment = environment or Environment(
            readable=tuple(Path(path).resolve() for path in self.roots.values())
        )
        self.modules: dict[str, types.ModuleType] = {}
        self.host_modules = dict(host_modules or {})
        self.host_packages = frozenset(host_packages)
        self.modules["os"] = _sandboxed_os(self.environment)
        self.modules["os.path"] = self.modules["os"].path
        self.modules["sys"] = self._sandboxed_sys()
        self.modules["json"] = _json_module()
        self.modules["shutil"] = _shutil_module(self.environment)
        self.modules["argparse"] = _argparse_module(self.environment)
        self.modules["functools"] = _functools_module(self)
        importlib_module, util = self._importlib_modules()
        self.modules["importlib"] = importlib_module
        self.modules["importlib.util"] = util
        # Called with the name of a module the source loads dynamically and
        # has no source for (a compiled extension); returns its stand-in.
        self.native_module_factory = None
        # Called with a top-level module name no root holds; returns the
        # directory holding it (fetched on demand) or None.
        self.resolver = None
        self._extensions: set[str] = set()
        # Distributions the run's builds pin, ``{import name: version}``.
        self.pins = requirement_pins(self.roots.values()) | requirement_pins(
            Path(path).parent for path in self.roots.values()
        )
        self.modules.update(_ctypes_modules(self))
        self.modules["glob"] = _glob_module(self.environment)
        self.modules["threading"] = _threading_module()
        self.modules["pkgutil"] = _pkgutil_module(self)
        self.modules["pathlib"] = self.environment.fs.pathlib_module()
        import contextlib

        self._host_io = contextlib.ExitStack()
        self._trees: dict[Path, ast.Module] = {}
        self._published: set[str] = set()
        # Libraries fetched on first import, and why any could not load.
        self._fetched: set[str] = set()
        self.unmodelled: dict[str, str] = {}
        # The latest error raised afresh in interpreted code, and where, and
        # the settings that decided the branches it was raised within.
        self.last_error: tuple[BaseException, str] | None = None
        self.last_error_control: tuple[str, ...] = ()
        # ``(file, function name)`` of every function the run called.
        self.executed: set[tuple[str, str]] = set()
        self._import_guards = 0
        self.stack: list[_Location] = []
        self.builtins = self._builtins()
        self.max_depth = 400
        # Settings provenance: which environment variables and command-line
        # options each value, branch and launch depends on. ``_reads`` holds
        # what the expressions being evaluated have read; variables and
        # attributes keep what their values were computed from; ``_control``
        # holds the tests of the branches being run, outermost first.
        self._reads: list[set] = []
        self._module_taint: dict[int, dict[str, frozenset]] = {}
        self._attribute_taint: dict[tuple[int, object], frozenset] = {}
        self._tainted_owners: set[int] = set()
        # Each node type's handler, looked up once.
        self._statement_handlers: dict[type, Any] = {}
        self._expression_handlers: dict[type, Any] = {}
        self._control: list[tuple[_Frame, frozenset]] = []
        self._pending_arguments = None
        self._return_taint: frozenset = frozenset()
        self._early_exit = None
        self.environment.on_read = self._read_setting

    # ------------------------------------------------------------------ world

    def _sandboxed_sys(self) -> types.ModuleType:
        module = types.ModuleType("sys")
        for name in (
            "version",
            "version_info",
            "hexversion",
            "implementation",
            "platform",
            "byteorder",
            "maxsize",
            "float_info",
            "int_info",
            "getrecursionlimit",
            "exc_info",
            "exception",
            "getsizeof",
            "intern",
            "flags",
            "float_repr_style",
            "getdefaultencoding",
            "getfilesystemencoding",
        ):
            setattr(module, name, getattr(sys, name))
        module.modules = self.modules
        module.meta_path = []
        module.path_hooks = []
        module.path_importer_cache = {}
        module.argv = ["interpreted"]
        module.path = [str(path) for path in self.roots.values()]
        module.executable = "python"
        module.stdout = module.stderr = _Sink()
        module.stdin = None
        module.setrecursionlimit = lambda limit: None
        # Hooks the source installs live in this module, never the host's.
        module.excepthook = module.__excepthook__ = sys.__excepthook__
        module.unraisablehook = module.__unraisablehook__ = sys.__unraisablehook__
        module.displayhook = module.__displayhook__ = sys.__displayhook__
        module.audit = lambda *args: None
        module.addaudithook = lambda hook: None

        def getframe(depth=0):
            # Depth 0 is the interpreted function that called _getframe.
            index = len(self.stack) - 1 - depth
            if index < 0:
                raise ValueError("call stack is not deep enough")
            return _FrameView(self.stack, index)

        module._getframe = getframe
        module.exit = _refuse_exit
        module.prefix = module.base_prefix = module.exec_prefix = "/"
        return module

    def _builtins(self) -> dict[str, Any]:
        namespace = dict(vars(builtins))
        namespace["__import__"] = self._import_hook
        namespace["open"] = _sandboxed_open(self.environment)
        namespace["print"] = lambda *args, **kwargs: None
        namespace["input"] = _refuse_exit
        namespace["exit"] = namespace["quit"] = _refuse_exit
        namespace["exec"] = self._exec_builtin
        namespace["eval"] = self._eval_builtin
        namespace["breakpoint"] = lambda *args, **kwargs: None
        return namespace

    # --------------------------------------------------------------- imports

    def _importlib_modules(self):
        """``importlib`` and ``importlib.util`` over this interpreter's modules."""
        import importlib
        import importlib.machinery

        module = types.ModuleType("importlib")
        util = types.ModuleType("importlib.util")

        def import_module(name, package=None):
            if name.startswith("."):
                level = len(name) - len(name.lstrip("."))
                name = _resolve_relative(name.lstrip("."), package or "", level)
            try:
                return self.import_module(name)
            except SourceUnavailable:
                if self.native_module_factory is None:
                    raise
                native = self.native_module_factory(name)
                if native is None:
                    raise
                self.modules[name] = native
                return native

        def find_spec(name, package=None):
            if name in self.modules:
                return getattr(self.modules[name], "__spec__", None) or (
                    importlib.machinery.ModuleSpec(name, None)
                )
            located = self._module_file(name)
            if located is not None:
                path, is_package = located
                spec = importlib.machinery.ModuleSpec(
                    name,
                    None,
                    origin=str(path) if path.suffix == ".py" else None,
                    is_package=is_package,
                )
                if is_package:
                    spec.submodule_search_locations = [
                        str(path.parent if path.suffix else path)
                    ]
                return spec
            top = name.partition(".")[0]
            if name in _PURE_STDLIB or top in self.host_packages:
                return importlib.util.find_spec(name)
            return None

        interpreter = self

        class NativeLoader:
            """Loads a compiled extension file: its stand-in, by module name."""

            def __init__(self, name):
                self.name = name

            def create_module(self, spec):
                return interpreter._native(self.name)

            def exec_module(self, module):
                pass

        def spec_from_file_location(name, location=None, *args, **kwargs):
            if location is not None and _is_extension_file(str(location)):
                return importlib.machinery.ModuleSpec(
                    name, NativeLoader(name), origin=str(location)
                )
            return importlib.util.spec_from_file_location(
                name, location, *args, **kwargs
            )

        def module_from_spec(spec):
            if isinstance(spec.loader, NativeLoader):
                return spec.loader.create_module(spec)
            return importlib.util.module_from_spec(spec)

        module.import_module = import_module
        module.reload = lambda target: target
        module.invalidate_caches = lambda: None
        module.util = util
        module.machinery = importlib.machinery
        util.find_spec = find_spec
        util.module_from_spec = module_from_spec
        util.spec_from_file_location = spec_from_file_location
        util.LazyLoader = importlib.util.LazyLoader
        return module, util

    def _compiled(self, name: str) -> bool:
        """Whether a compiled extension file for ``name`` sits in its package."""
        package, _, leaf = name.rpartition(".")
        if not package:
            return False
        located = self._module_file(package)
        if located is None:
            return False
        directory = located[0].parent if located[0].suffix else located[0]
        return any(
            entry.name.startswith(f"{leaf}.") and entry.suffix in _EXTENSION_SUFFIXES
            for entry in directory.glob(f"{leaf}.*")
        )

    def _native(self, name):
        factory = self.native_module_factory
        native = factory(name) if factory is not None else None
        if native is None:
            raise SourceUnavailable(f"No native module {name!r}")
        self.modules.setdefault(name, native)
        return self.modules[name]

    def declare_extensions(self, names: Iterable[str]) -> None:
        """Compiled extensions the run's build installed, by dotted name.

        Each is a file in its package directory, and importing it by either
        name gives the native stand-in.
        """
        for name in names:
            self._extensions.add(name)
            self._extensions.add(name.rpartition(".")[2])
            package, _, leaf = name.rpartition(".")
            located = self._module_file(package) if package else None
            if located is not None:
                directory = located[0].parent if located[0].suffix else located[0]
                self.environment.files[
                    str(directory / f"{leaf}.cpython-312-x86_64-linux-gnu.so")
                ] = ""

    def _module_file(self, name: str) -> tuple[Path, bool] | None:
        top, _, rest = name.partition(".")
        root = self.roots.get(top)
        if root is None:
            # A directory the source itself put on sys.path.
            for entry in self.modules["sys"].path:
                candidate = Path(str(entry))
                if not self.environment.may_read(candidate):
                    continue
                if (
                    (candidate / top / "__init__.py").is_file()
                    or (candidate / f"{top}.py").is_file()
                    or (candidate / top).is_dir()  # a namespace package
                ):
                    root = candidate
                    break
            else:
                return None
        relative = Path(top, *rest.split(".")) if rest else Path(top)
        package = root / relative / "__init__.py"
        if package.is_file():
            return package, True
        module = (root / relative).with_suffix(".py")
        if module.is_file():
            return module, False
        if (root / relative).is_dir():
            return root / relative, True  # namespace package
        return None

    def import_module(self, name: str) -> types.ModuleType:
        found = self.modules.get(name)
        if found is not None:
            return found
        if name in self.host_modules:
            self.modules[name] = self.host_modules[name]
            return self.host_modules[name]
        top = name.partition(".")[0]
        if top in self.unmodelled:
            # A library that could not load is a stand-in throughout.
            module = InertModule(name)
            self.modules[name] = module
            self._bind_submodule(name, module)
            return module
        if top in self.host_modules and top not in self.host_packages:
            # A part of a modelled package the model does not spell out.
            parent = self.import_module(name.rpartition(".")[0])
            module = InertModule(name)
            self.modules[name] = module
            self._bind_submodule(name, module)
            return module
        if name in _INERT_STDLIB or top in _INERT_STDLIB:
            module = InertModule(name)
            self.modules[name] = module
            if "." in name:
                self.import_module(name.rpartition(".")[0])
                self._bind_submodule(name, module)
            return module
        if name in _PURE_STDLIB or top in self.host_packages:
            import importlib

            try:
                module = importlib.import_module(name)
            except ImportError as error:
                raise SourceUnavailable(str(error)) from None
            self.modules[name] = module
            parent, _, child = name.rpartition(".")
            if parent in self.modules and not hasattr(self.modules[parent], child):
                self._bind_submodule(name, module)
            return module
        if "." in name:
            parent = self.import_module(name.rpartition(".")[0])
            found = self.modules.get(name)
            if found is not None:
                return found
        else:
            parent = None
        located = self._module_file(name)
        if located is None and self.resolver is not None and top not in self.roots:
            # The run's sources may hold it: fetch that checkout now.
            found_root = self.resolver(top)
            if found_root is not None:
                found_root = Path(found_root)
                self.roots[top] = found_root
                self._fetched.add(top)
                self.environment.readable = (
                    *self.environment.readable,
                    found_root.resolve(),
                )
                for pinned, version in requirement_pins(
                    (found_root, found_root.parent)
                ).items():
                    self.pins.setdefault(pinned, version)
                located = self._module_file(name)
        if located is None and (name in self._extensions or self._compiled(name)):
            return self._native(name)
        if located is None:
            # A finder the source installed may supply it (an alias, a lazy
            # module); it registers what it finds in sys.modules.
            for finder in list(self.modules["sys"].meta_path):
                find = getattr(finder, "find_spec", None)
                if find is not None and find(name, None) is not None:
                    found = self.modules.get(name)
                    if found is not None:
                        return found
            raise SourceUnavailable(f"No module named {name!r}")
        path, is_package = located
        module = types.ModuleType(name)
        module.__file__ = str(path) if path.suffix == ".py" else None
        module.__package__ = name if is_package else name.rpartition(".")[0]
        if is_package:
            module.__path__ = [str(path.parent if path.suffix else path)]
        module.__builtins__ = self.builtins
        module.__loader__ = None
        import importlib.machinery

        module.__spec__ = importlib.machinery.ModuleSpec(
            name, None, origin=module.__file__, is_package=is_package
        )
        if is_package:
            module.__spec__.submodule_search_locations = module.__path__
        self.modules[name] = module
        self._publish(name, module)
        if path.suffix == ".py":
            try:
                self.run_module(module, self.parse(path))
            except BaseException as error:
                self.modules.pop(name, None)
                if name in self._published:
                    self._published.discard(name)
                    sys.modules.pop(name, None)
                if (
                    name != top
                    or top not in self._fetched
                    or not isinstance(error, Exception)
                ):
                    raise
                return self._unmodelled_library(top, error)
        # Python binds a submodule to its package once it has loaded.
        if parent is not None:
            self._bind_submodule(name, self.modules.get(name, module))
        return self.modules.get(name, module)

    def _unmodelled_library(self, top, error):
        """An inert ``top``: a fetched library whose own import fails here.

        Its import needs what the checkout does not hold (a compiled runtime,
        a git submodule); anything the run calls in it is counted as inert.
        """
        reason = f"{type(error).__name__}: {str(error)[:200]}"
        self.unmodelled[top] = reason
        logger.warning("library %s is not modelled: %s", top, reason)
        for loaded in [key for key in self.modules if key.startswith(top + ".")]:
            self.modules.pop(loaded)
            if loaded in self._published:
                self._published.discard(loaded)
                sys.modules.pop(loaded, None)
        module = InertModule(top)
        if top in self.pins:
            module.__version__ = self.pins[top]
        self.modules[top] = module
        return module

    def _bind_submodule(self, name, module):
        """Bind ``module`` to its package as an attribute, as Python does.

        A package the host process owns is never changed: the interpreter
        swaps in its own copy of it and binds the submodule there, so host
        code keeps reading the real submodule.
        """
        parent_name, _, child = name.rpartition(".")
        parent = self.modules.get(parent_name)
        if parent is None:
            return
        if (
            sys.modules.get(parent_name) is parent
            and parent_name not in self._published
        ):
            local = types.ModuleType(parent_name)
            local.__dict__.update(vars(parent))
            self.modules[parent_name] = local
            parent = local
            if "." in parent_name:
                self._bind_submodule(parent_name, local)
        setattr(parent, child, module)

    def _publish(self, name, module):
        """Make ``module`` findable by the standard library while it runs.

        ``dataclasses`` and ``typing`` look a class's module up in
        ``sys.modules`` by name. Only names no real module holds are added,
        and :meth:`close` removes them.
        """
        if name not in sys.modules:
            sys.modules[name] = module
            self._published.add(name)

    def close(self) -> None:
        """Withdraw the interpreted modules from ``sys.modules``."""
        for name in self._published:
            if sys.modules.get(name) is self.modules.get(name):
                sys.modules.pop(name, None)
        self._published.clear()
        self._host_io.close()

    def __enter__(self):
        self._host_io.enter_context(self.environment.fs.host_libraries())
        return self

    def __exit__(self, *exc):
        self.close()

    def parse(self, path: Path) -> ast.Module:
        tree = self._trees.get(path)
        if tree is None:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            self._trees[path] = tree
        return tree

    def _import_hook(self, name, globals_=None, locals_=None, fromlist=(), level=0):
        package = (globals_ or {}).get("__package__") or ""
        absolute = _resolve_relative(name, package, level)
        module = self.import_module(absolute)
        if fromlist:
            for item in fromlist:
                if item != "*" and not hasattr(module, item):
                    try:
                        self.import_module(f"{absolute}.{item}")
                    except SourceUnavailable:
                        pass
            return module
        return self.modules[absolute.partition(".")[0]]

    def _from_import(self, module_name, item):
        module = self.import_module(module_name)
        try:
            return getattr(module, item)
        except AttributeError:
            pass
        try:
            return self.import_module(f"{module_name}.{item}")
        except SourceUnavailable:
            raise ImportError(
                f"cannot import name {item!r} from {module_name!r}"
            ) from None

    # ------------------------------------------------------------- execution

    def run_module(self, module: types.ModuleType, tree: ast.Module) -> None:
        frame = _Frame(module)
        frame.variables = module.__dict__
        self.stack.append(
            _Location(
                module.__file__ or "?",
                0,
                f"<module {module.__name__}>",
                module.__dict__,
                module.__dict__,
            )
        )
        try:
            self._exec_block(tree.body, frame)
        finally:
            self.stack.pop()

    def _exec_builtin(self, source, globals_=None, locals_=None):
        if not isinstance(source, str):
            raise InterpreterError("exec of a code object")
        module = types.ModuleType("<exec>")
        if globals_ is not None:
            module.__dict__.update(globals_)
        module.__builtins__ = self.builtins
        frame = _Frame(module)
        frame.variables = module.__dict__ if locals_ is None else locals_
        self._exec_block(ast.parse(source).body, frame)
        if globals_ is not None:
            globals_.update(
                {k: v for k, v in module.__dict__.items() if k != "__builtins__"}
            )

    def _eval_builtin(self, source, globals_=None, locals_=None):
        module = types.ModuleType("<eval>")
        if globals_ is not None:
            module.__dict__.update(globals_)
        module.__builtins__ = self.builtins
        frame = _Frame(module)
        frame.variables = dict(locals_ or {})
        return self.eval(ast.parse(source, mode="eval").body, frame)

    def call_function(self, function: SourceFunction, args, kwargs):
        if len(self.stack) > self.max_depth:
            raise RecursionError("interpreted call depth")
        node = function._node
        frame = _Frame(
            function._frame.module, enclosing=function._frame, function=function
        )
        self._bind_arguments(function, node.args, args, kwargs, frame.variables)
        self._bind_argument_taints(function, node, args, kwargs, frame)
        if not isinstance(node, ast.Lambda):
            self.executed.add(
                (function._frame.module.__dict__.get("__file__") or "?", node.name)
            )
        self.stack.append(
            _Location(
                function._frame.module.__dict__.get("__file__") or "?",
                node.lineno,
                function.__qualname__,
                function._frame.module.__dict__,
                frame.variables,
                [argument.arg for argument in node.args.posonlyargs + node.args.args]
                + [argument.arg for argument in node.args.kwonlyargs],
                (*args, *kwargs.values()),
                tuple(kwargs),
                frame,
            )
        )
        # What the callee reads reaches the caller only through what it returns.
        reads, self._reads = self._reads, []
        self._return_taint = frozenset()
        try:
            if isinstance(node, ast.Lambda):
                value, self._return_taint = self._tainted(node.body, frame)
                return value
            if function._is_generator:
                return _GeneratorRun(self, node.body, frame)
            try:
                self._exec_block(node.body, frame)
            except _Return as returned:
                self._return_taint = returned.taint
                return returned.value
            self._return_taint = frozenset()
            return None
        except Exception as error:
            if self.last_error is None or self.last_error[0] is not error:
                # Where an error began, before a caller's handler swallows it.
                if getattr(error, "__context__", None) is None:
                    self.last_error = (error, self.where())
                    self.last_error_control = self.control_settings()
                    # A traceback the run logs names the interpreted source too.
                    error.add_note(f"interpreted at {self.last_error[1]}")
            raise
        finally:
            self._reads = reads
            self.stack.pop()

    def call_path(self) -> list[tuple[str, tuple[int, ...]]]:
        """The interpreted calls on the stack, outermost first, with the integers passed."""
        path = []
        for location in self.stack:
            module = location.globals.get("__name__", "")
            integers = tuple(
                value
                for value in location.passed
                if isinstance(value, int) and not isinstance(value, bool)
            )
            path.append((f"{module}.{location.function}".lstrip("."), integers))
        return path

    # ------------------------------------------------------------ provenance

    def _read_setting(self, name: str) -> None:
        if self._reads:
            self._note((name,))

    def _note(self, taint) -> None:
        # An evaluation's set is made only once it reads a setting.
        if taint and self._reads:
            reads = self._reads[-1]
            if reads is None:
                self._reads[-1] = set(taint)
            else:
                reads.update(taint)

    def _tainted(self, node, frame, propagate: bool = False):
        """``(value, settings it was computed from)`` of evaluating ``node``."""
        self._reads.append(None)
        try:
            value = self.eval(node, frame)
        finally:
            reads = self._reads.pop()
        if reads is None:
            return value, _NO_SETTINGS
        if propagate:
            self._note(reads)
        return value, frozenset(reads)

    def _local_control(self, frame) -> frozenset:
        """What decided the branches of ``frame`` now being run."""
        found = frozenset()
        for owner, taint in self._control:
            if owner is frame:
                found |= taint
        return found

    def control_settings(self) -> tuple[str, ...]:
        """The settings that decided every branch now being run, innermost last."""
        seen: dict[str, None] = {}
        for _, taint in self._control:
            for name in sorted(taint):
                seen.pop(name, None)
                seen[name] = None
        return tuple(seen)

    def argument_settings(self) -> tuple[str, ...]:
        """The settings the values of the statement being run came from.

        Inside a call the statement makes to host code (a kernel launch),
        these are what its arguments were computed from.
        """
        return tuple(sorted(self._reads[-1])) if self._reads and self._reads[-1] else ()

    def _name_taint(self, name, frame):
        if name in frame.declared_global:
            return self._module_taint.get(id(frame.module.__dict__), {}).get(name)
        current = frame
        while current is not None:
            if current is frame or (
                not current.is_class and current.function is not None
            ):
                variables = current.variables
                if variables is current.module.__dict__:
                    break
                if dict.__contains__(variables, name):
                    return current.taint.get(name) if current.taint else None
            current = current.enclosing
        return self._module_taint.get(id(frame.module.__dict__), {}).get(name)

    def _store_taint(self, name, frame, taint) -> None:
        if (
            not taint
            and frame.taint is None
            and (
                frame.variables is not frame.module.__dict__
                or id(frame.variables) not in self._module_taint
            )
            and name not in frame.declared_global
            and name not in frame.declared_nonlocal
        ):
            # Nothing to record, and nothing recorded to clear.
            return
        scope = self._scope_of(name, frame)
        if scope is frame.module.__dict__:
            table = self._module_taint.get(id(scope))
            if table is None:
                if not taint:
                    return
                table = self._module_taint[id(scope)] = {}
        else:
            owner = frame
            while owner is not None and owner.variables is not scope:
                owner = owner.enclosing
            if owner is None:
                return
            if owner.taint is None:
                if not taint:
                    return
                owner.taint = {}
            table = owner.taint
        if taint:
            table[name] = taint
        else:
            table.pop(name, None)

    def _store_attribute_taint(self, owner, key, taint, accumulate=False) -> None:
        if not taint and id(owner) not in self._tainted_owners:
            return
        slot = (id(owner), key)
        if accumulate:
            taint = taint | self._attribute_taint.get(slot, frozenset())
        if taint:
            self._attribute_taint[slot] = taint
            self._tainted_owners.add(id(owner))
        else:
            self._attribute_taint.pop(slot, None)

    def _bind_argument_taints(self, function, node, args, kwargs, frame) -> None:
        """Give each parameter the settings its argument was computed from."""
        pending, self._pending_arguments = self._pending_arguments, None
        if pending is None:
            return
        passed, passed_kwargs, positional, keyword = pending
        # A method gets its instance first: the caller's arguments end the list.
        offset = len(args) - len(passed)
        if offset < 0 or any(a is not b for a, b in zip(args[offset:], passed)):
            return
        names = [argument.arg for argument in node.args.posonlyargs + node.args.args]
        for index, taint in enumerate(positional):
            if taint and index + offset < len(names):
                self._store_taint(names[index + offset], frame, taint)
        for name, taint in keyword.items():
            if taint and name in kwargs:
                self._store_taint(name, frame, taint)

    def _bind_arguments(self, function, spec: ast.arguments, args, kwargs, target):
        positional = spec.posonlyargs + spec.args
        names = [argument.arg for argument in positional]
        defaults = function._defaults
        first_default = len(names) - len(defaults)
        kwargs = dict(kwargs)
        for index, name in enumerate(names):
            if index < len(args):
                target[name] = args[index]
            elif name in kwargs and index >= len(spec.posonlyargs):
                target[name] = kwargs.pop(name)
            elif index >= first_default:
                target[name] = defaults[index - first_default]
            else:
                raise TypeError(
                    f"{function.__qualname__}() missing required argument {name!r}"
                )
        extra = args[len(names) :]
        if spec.vararg:
            target[spec.vararg.arg] = tuple(extra)
        elif extra:
            raise TypeError(
                f"{function.__qualname__}() takes {len(names)} positional "
                f"arguments but {len(args)} were given"
            )
        for argument in spec.kwonlyargs:
            name = argument.arg
            if name in kwargs:
                target[name] = kwargs.pop(name)
            elif name in function._kw_defaults:
                target[name] = function._kw_defaults[name]
            else:
                raise TypeError(
                    f"{function.__qualname__}() missing keyword argument {name!r}"
                )
        if spec.kwarg:
            target[spec.kwarg.arg] = kwargs
        elif kwargs:
            raise TypeError(
                f"{function.__qualname__}() got an unexpected keyword argument "
                f"{next(iter(kwargs))!r}"
            )

    def where(self) -> str:
        """The interpreted call stack, innermost last, for error messages."""
        return " <- ".join(
            f"{Path(loc.file).name}:{loc.line} {loc.function}"
            for loc in reversed(self.stack[-12:])
        )

    def caller_holding(self, holds: Callable[[object], bool]):
        """The innermost interpreted call with an argument ``holds`` accepts.

        Returns ``(qualified name, [argument values])`` or ``None``.
        """
        for location in reversed(self.stack):
            values = [location.locals.get(name) for name in location.parameters]
            if any(holds(value) for value in values):
                module = location.globals.get("__name__", "")
                return f"{module}.{location.function}".lstrip("."), values
        return None

    def callers_holding(self, holds: Callable[[object], bool]) -> list[tuple]:
        """Every interpreted call made with an argument ``holds`` accepts.

        Innermost first, as ``(qualified name, values passed, keyword names)``:
        the last ``len(keyword names)`` values were passed by those names.
        """
        found = []
        for location in reversed(self.stack):
            if any(holds(value) for value in location.passed):
                module = location.globals.get("__name__", "")
                found.append(
                    (
                        f"{module}.{location.function}".lstrip("."),
                        location.passed,
                        location.keywords,
                    )
                )
        return found

    # ------------------------------------------------------------ statements

    def _exec_block(self, body, frame):
        """Run statements; hand back ``_BREAK``/``_CONTINUE`` if one left the block."""
        for index, statement in enumerate(body):
            signal = self._exec(statement, frame)
            if signal is not None:
                return signal
            if self._early_exit is not None:
                # A branch not taken would have left the block: the rest of
                # it runs because of that test.
                self._control.append(self._early_exit)
                self._early_exit = None
                try:
                    return self._exec_block(body[index + 1 :], frame)
                finally:
                    self._control.pop()
        return None

    def _exec(self, node, frame):
        handler = self._statement_handlers.get(type(node))
        if handler is None:
            handler = self._statement_handlers[type(node)] = getattr(
                self, f"_stmt_{type(node).__name__}", None
            )
        if handler is None:
            raise InterpreterError(
                f"statement {type(node).__name__} at line {node.lineno}"
            )
        if self.stack:
            self.stack[-1].line = node.lineno
        try:
            return handler(node, frame)
        except (_Return, _Break, _Continue):
            raise
        except BaseException as error:
            if getattr(error, "_where", None) is None:
                try:
                    error._where = self.where()
                except AttributeError:
                    pass
            raise

    def _stmt_Expr(self, node, frame):
        self._tainted(node.value, frame)

    def _stmt_Pass(self, node, frame):
        pass

    def _stmt_Return(self, node, frame):
        if node.value is None:
            raise _Return(None, self._local_control(frame))
        value, taint = self._tainted(node.value, frame)
        raise _Return(value, taint | self._local_control(frame))

    def _stmt_Break(self, node, frame):
        return _BREAK

    def _stmt_Continue(self, node, frame):
        return _CONTINUE

    def _stmt_Assign(self, node, frame):
        value, taint = self._tainted(node.value, frame)
        taint |= self._local_control(frame)
        for target in node.targets:
            self.assign(target, value, frame, taint)

    def _stmt_AnnAssign(self, node, frame):
        if frame.is_class or frame.function is None:
            annotations = frame.variables.setdefault("__annotations__", {})
            if isinstance(node.target, ast.Name):
                annotations[node.target.id] = self._annotation(node.annotation, frame)
        if node.value is not None:
            value, taint = self._tainted(node.value, frame)
            self.assign(node.target, value, frame, taint | self._local_control(frame))

    def _annotation(self, node, frame):
        if frame.module.__dict__.get(_FUTURE_ANNOTATIONS):
            return ast.unparse(node)
        try:
            return self.eval(node, frame)
        except Exception:  # noqa: BLE001 - an annotation is documentation here
            return ast.unparse(node)

    def _stmt_AugAssign(self, node, frame):
        operation = _INPLACE[type(node.op)]
        target = node.target
        if isinstance(target, ast.Name):
            current = self.lookup(target.id, frame)
            value, taint = self._tainted(node.value, frame)
            self.store(target.id, operation(current, value), frame)
            taint |= self._name_taint(target.id, frame) or frozenset()
            self._store_taint(target.id, frame, taint | self._local_control(frame))
        elif isinstance(target, ast.Attribute):
            owner = self.eval(target.value, frame)
            value = operation(getattr(owner, target.attr), self.eval(node.value, frame))
            setattr(owner, target.attr, value)
        elif isinstance(target, ast.Subscript):
            owner = self.eval(target.value, frame)
            key = self.eval(target.slice, frame)
            owner[key] = operation(owner[key], self.eval(node.value, frame))
        else:
            raise InterpreterError("augmented assignment target")

    def _branch(self, frame, taint, body):
        """Run ``body`` under a branch whose test read ``taint``."""
        if not taint:
            return self._exec_block(body, frame)
        self._control.append((frame, taint))
        try:
            return self._exec_block(body, frame)
        finally:
            self._control.pop()

    def _stmt_If(self, node, frame):
        test, taint = self._tainted(node.test, frame)
        taken, skipped = (
            (node.body, node.orelse)
            if self.truth(test, node.test, frame)
            else (node.orelse, node.body)
        )
        signal = self._branch(frame, taint, taken)
        if signal is not None:
            return signal
        if taint:
            # What the other arm would have set holds what it holds because
            # the test went this way.
            for name in _assigned_names(skipped):
                self._store_taint(
                    name, frame, taint | (self._name_taint(name, frame) or frozenset())
                )
            if _exits(skipped):
                self._early_exit = (frame, taint)
        return None

    def _stmt_While(self, node, frame):
        while True:
            test, taint = self._tainted(node.test, frame)
            if not self.truth(test, node.test, frame):
                break
            if self._branch(frame, taint, node.body) is _BREAK:
                return None
        return self._exec_block(node.orelse, frame)

    def _stmt_For(self, node, frame):
        iterable, taint = self._tainted(node.iter, frame)
        item_taint = taint | self._local_control(frame)
        for item in iterable:
            self.assign(node.target, item, frame, item_taint)
            if self._branch(frame, taint, node.body) is _BREAK:
                return None
        return self._exec_block(node.orelse, frame)

    def _stmt_Raise(self, node, frame):
        if node.exc is None:
            raise  # noqa: PLE0704 - re-raise the exception being handled
        exception = self.eval(node.exc, frame)
        cause = self.eval(node.cause, frame) if node.cause is not None else None
        if isinstance(exception, type):
            exception = exception()
        if node.cause is not None:
            raise exception from cause
        raise exception

    def _guards_imports(self, node, frame) -> bool:
        for handler in node.handlers:
            if handler.type is None:
                return True
            names = (
                handler.type.elts
                if isinstance(handler.type, ast.Tuple)
                else [handler.type]
            )
            for name in names:
                text = ast.unparse(name)
                if text.split(".")[-1] in (
                    "ImportError",
                    "ModuleNotFoundError",
                    "Exception",
                    "BaseException",
                ):
                    return True
        return False

    def _stmt_Try(self, node, frame):
        guarded = self._guards_imports(node, frame)
        signal = None
        try:
            try:
                frame.import_guards += guarded
                try:
                    signal = self._exec_block(node.body, frame)
                finally:
                    frame.import_guards -= guarded
            except (_Break, _Continue, _Return):
                raise
            except (
                BaseException
            ) as error:  # noqa: BLE001 - the source's own handlers decide
                for handler in node.handlers:
                    if handler.type is None or isinstance(
                        error, self.eval(handler.type, frame)
                    ):
                        if handler.name:
                            self.store(handler.name, error, frame)
                        signal = self._exec_block(handler.body, frame)
                        break
                else:
                    raise
            else:
                if signal is None:
                    signal = self._exec_block(node.orelse, frame)
        finally:
            final = self._exec_block(node.finalbody, frame)
        return final if final is not None else signal

    def _stmt_With(self, node, frame):
        managers = []
        try:
            for item in node.items:
                manager = self.eval(item.context_expr, frame)
                entered = type(manager).__enter__(manager)
                managers.append(manager)
                if item.optional_vars is not None:
                    self.assign(item.optional_vars, entered, frame)
            signal = self._exec_block(node.body, frame)
        except (_Break, _Continue, _Return):
            for manager in reversed(managers):
                type(manager).__exit__(manager, None, None, None)
            raise
        except BaseException as error:  # noqa: BLE001 - each manager may suppress it
            pending = error
            for manager in reversed(managers):
                if pending is not None and type(manager).__exit__(
                    manager, type(pending), pending, pending.__traceback__
                ):
                    pending = None
                elif pending is None:
                    type(manager).__exit__(manager, None, None, None)
            if pending is not None:
                raise pending from None
        else:
            for manager in reversed(managers):
                type(manager).__exit__(manager, None, None, None)
            return signal

    def _stmt_Assert(self, node, frame):
        if not self.truth(self.eval(node.test, frame), node.test, frame):
            message = self.eval(node.msg, frame) if node.msg is not None else None
            raise AssertionError(message)

    def _stmt_Delete(self, node, frame):
        for target in node.targets:
            if isinstance(target, ast.Name):
                scope = self._scope_of(target.id, frame)
                scope.pop(target.id, None)
            elif isinstance(target, ast.Attribute):
                delattr(self.eval(target.value, frame), target.attr)
            elif isinstance(target, ast.Subscript):
                del self.eval(target.value, frame)[self.eval(target.slice, frame)]

    def _stmt_Global(self, node, frame):
        frame.declared_global.update(node.names)

    def _stmt_Nonlocal(self, node, frame):
        frame.declared_nonlocal.update(node.names)

    def _required(self, name, frame):
        """``name``, or an inert module when nothing provides it.

        An import statement the source wraps in a ``try`` that catches
        ``ImportError`` is an optional dependency, and its absence is the
        source's own fallback. An unguarded one is a dependency the run had
        installed.
        """
        try:
            return self.import_module(name)
        except SourceUnavailable:
            top = name.partition(".")[0]
            # A dependency the build pins was installed, guarded or not.
            if frame.import_guards and top not in self.pins:
                raise
            module = InertModule(name)
            if top in self.pins and "." not in name:
                module.__version__ = self.pins[top]
            self.modules[name] = module
            if "." in name:
                self._bind_submodule(name, module)
            return module

    def _stmt_Import(self, node, frame):
        for alias in node.names:
            module = self._required(alias.name, frame)
            if alias.asname:
                self.store(alias.asname, module, frame)
            else:
                top = alias.name.partition(".")[0]
                self.store(
                    top, self.modules.get(top) or self._required(top, frame), frame
                )

    def _stmt_ImportFrom(self, node, frame):
        package = frame.module.__dict__.get("__package__") or ""
        module_name = _resolve_relative(node.module or "", package, node.level)
        if module_name == "__future__":
            import __future__

            for alias in node.names:
                frame.module.__dict__[alias.asname or alias.name] = getattr(
                    __future__, alias.name
                )
                if alias.name == "annotations":
                    frame.module.__dict__[_FUTURE_ANNOTATIONS] = True
            return
        module = self._required(module_name, frame)
        for alias in node.names:
            if alias.name == "*":
                if isinstance(module, InertModule):
                    self._star_from_inert(module, frame)
                exported = getattr(module, "__all__", None)
                names = (
                    exported
                    if exported is not None
                    else [key for key in vars(module) if not key.startswith("_")]
                )
                for name in names:
                    self.store(name, getattr(module, name), frame)
                continue
            try:
                value = self._from_import(module_name, alias.name)
            except ImportError:
                if frame.import_guards or not isinstance(module, InertModule):
                    raise
                value = getattr(module, alias.name)
            self.store(alias.asname or alias.name, value, frame)

    def _star_from_inert(self, module, frame):
        """``from <inert> import *``: any name may have come from ``module``.

        The importing module's missing names, read as its globals or as its
        attributes (PEP 562 ``__getattr__``), are ``module``'s stand-ins.
        """
        namespace = frame.module.__dict__
        if _INERT_STAR in namespace:
            return
        namespace[_INERT_STAR] = module

        def missing(name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return getattr(module, name)

        namespace.setdefault("__getattr__", missing)

    def _stmt_FunctionDef(self, node, frame):
        function = self.make_function(node, frame)
        value = function
        for decorator in reversed(node.decorator_list):
            value = self.eval(decorator, frame)(value)
        self.store(node.name, value, frame)

    def _stmt_AsyncFunctionDef(self, node, frame):
        # A coroutine never runs on the path a forward takes; keep the name.
        self.store(node.name, self.make_function(node, frame), frame)

    def _stmt_ClassDef(self, node, frame):
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Starred):
                bases.extend(self.eval(base.value, frame))
            else:
                bases.append(self.eval(base, frame))
        keywords = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                keywords.update(self.eval(keyword.value, frame))
            else:
                keywords[keyword.arg] = self.eval(keyword.value, frame)
        resolved = types.resolve_bases(tuple(bases))
        metaclass, namespace, keywords = types.prepare_class(
            node.name, resolved, keywords
        )
        namespace["__module__"] = frame.module.__name__
        qualname = node.name
        owner = frame.function
        if owner is not None:
            qualname = f"{owner.__qualname__}.<locals>.{node.name}"
        namespace["__qualname__"] = qualname
        body = _Frame(frame.module, enclosing=frame, is_class=True)
        body.variables = namespace
        body.class_cell = _Cell()
        self._exec_block(node.body, body)
        # type() makes these implicit class/static methods of a plain function.
        for implicit, wrapper in (
            ("__init_subclass__", classmethod),
            ("__class_getitem__", classmethod),
            ("__new__", staticmethod),
        ):
            if isinstance(namespace.get(implicit), SourceFunction):
                namespace[implicit] = wrapper(namespace[implicit])
        if resolved is not tuple(bases):
            namespace["__orig_bases__"] = tuple(bases)
        cls = metaclass(node.name, resolved, namespace, **keywords)
        body.class_cell.value = cls
        body.class_cell.bound = True
        value = cls
        for decorator in reversed(node.decorator_list):
            value = self.eval(decorator, frame)(value)
        self.store(node.name, value, frame)

    def _stmt_Match(self, node, frame):
        subject = self.eval(node.subject, frame)
        for case in node.cases:
            if self._match(case.pattern, subject, frame) and (
                case.guard is None
                or self.truth(self.eval(case.guard, frame), case.guard, frame)
            ):
                return self._exec_block(case.body, frame)
        return None

    def _match(self, pattern, subject, frame) -> bool:
        if isinstance(pattern, ast.MatchValue):
            return subject == self.eval(pattern.value, frame)
        if isinstance(pattern, ast.MatchSingleton):
            return subject is pattern.value
        if isinstance(pattern, ast.MatchAs):
            if pattern.pattern is not None and not self._match(
                pattern.pattern, subject, frame
            ):
                return False
            if pattern.name:
                self.store(pattern.name, subject, frame)
            return True
        if isinstance(pattern, ast.MatchOr):
            return any(self._match(item, subject, frame) for item in pattern.patterns)
        if isinstance(pattern, ast.MatchClass):
            cls = self.eval(pattern.cls, frame)
            if not isinstance(subject, cls):
                return False
            match_args = getattr(cls, "__match_args__", ())
            for index, item in enumerate(pattern.patterns):
                if not self._match(item, getattr(subject, match_args[index]), frame):
                    return False
            for name, item in zip(pattern.kwd_attrs, pattern.kwd_patterns, strict=True):
                if not self._match(item, getattr(subject, name), frame):
                    return False
            return True
        if isinstance(pattern, ast.MatchSequence):
            items = list(subject) if isinstance(subject, list | tuple) else None
            if items is None or len(items) != len(pattern.patterns):
                return False
            return all(
                self._match(item, value, frame)
                for item, value in zip(pattern.patterns, items, strict=True)
            )
        raise InterpreterError(f"match pattern {type(pattern).__name__}")

    # ------------------------------------------------------------- variables

    def _scope_of(self, name, frame):
        if name in frame.declared_global:
            return frame.module.__dict__
        if name in frame.declared_nonlocal:
            outer = frame.enclosing
            while outer is not None:
                if (
                    not outer.is_class
                    and outer.function is not None
                    and (name in outer.variables)
                ):
                    return outer.variables
                outer = outer.enclosing
            raise SyntaxError(f"no binding for nonlocal {name!r}")
        return frame.variables

    def store(self, name, value, frame):
        self._scope_of(name, frame)[name] = value

    def lookup(self, name, frame):
        if name in frame.declared_global:
            return self._global(name, frame)
        value = frame.variables.get(name, _MISSING)
        if value is not _MISSING:
            return value
        if name == "__class__":
            cell = self._class_cell(frame)
            if cell is not None:
                return cell.value
        outer = frame.enclosing
        while outer is not None:
            if not outer.is_class and outer.function is not None:
                value = outer.variables.get(name, _MISSING)
                if value is not _MISSING:
                    return value
            outer = outer.enclosing
        return self._global(name, frame)

    def _global(self, name, frame):
        value = frame.module.__dict__.get(name, _MISSING)
        if value is not _MISSING:
            return value
        value = self.builtins.get(name, _MISSING)
        if value is not _MISSING:
            if name == "super":
                return self._super_for(frame)
            return value
        star = frame.module.__dict__.get(_INERT_STAR)
        if star is not None:
            return getattr(star, name)
        raise NameError(f"name {name!r} is not defined ({self.where()})")

    def _class_cell(self, frame):
        function_frame = frame
        while function_frame is not None and function_frame.function is None:
            function_frame = function_frame.enclosing
        if function_frame is None:
            return None
        return function_frame.function._class_cell

    def _super_for(self, frame):
        def zero_argument_super(*args):
            if args:
                return super(*args)
            cell = self._class_cell(frame)
            function_frame = frame
            while function_frame is not None and function_frame.function is None:
                function_frame = function_frame.enclosing
            if cell is None or function_frame is None:
                raise RuntimeError("super(): no class cell")
            node = function_frame.function._node
            first = (node.args.posonlyargs + node.args.args)[0].arg
            return super(cell.value, function_frame.variables[first])

        return zero_argument_super

    def assign(self, target, value, frame, taint=_NO_SETTINGS):
        if type(target) is ast.Name:
            name = target.id
            if (
                name not in frame.declared_global
                and name not in frame.declared_nonlocal
            ):
                variables = frame.variables
                variables[name] = value
                if (
                    taint
                    or frame.taint is not None
                    or (
                        variables is frame.module.__dict__
                        and id(variables) in self._module_taint
                    )
                ):
                    self._store_taint(name, frame, taint)
                return
            self.store(name, value, frame)
            self._store_taint(name, frame, taint)
        elif isinstance(target, ast.Name):
            self.store(target.id, value, frame)
            self._store_taint(target.id, frame, taint)
        elif isinstance(target, ast.Attribute):
            owner = self.eval(target.value, frame)
            setattr(owner, target.attr, value)
            self._store_attribute_taint(owner, target.attr, taint)
        elif isinstance(target, ast.Subscript):
            owner = self.eval(target.value, frame)
            owner[self.eval(target.slice, frame)] = value
            if taint:
                self._store_attribute_taint(owner, "[]", taint, accumulate=True)
        elif isinstance(target, ast.Tuple | ast.List):
            items = list(value)
            starred = getattr(target, "_starred", None)
            if starred is None:
                starred = target._starred = [
                    index
                    for index, element in enumerate(target.elts)
                    if isinstance(element, ast.Starred)
                ]
            if starred:
                at = starred[0]
                after = len(target.elts) - at - 1
                if len(items) < at + after:
                    raise ValueError("not enough values to unpack")
                head, middle, tail = (
                    items[:at],
                    items[at : len(items) - after],
                    items[len(items) - after :],
                )
                for element, item in zip(target.elts[:at], head, strict=True):
                    self.assign(element, item, frame, taint)
                self.assign(target.elts[at].value, middle, frame, taint)
                for element, item in zip(target.elts[at + 1 :], tail, strict=True):
                    self.assign(element, item, frame, taint)
            else:
                if len(items) != len(target.elts):
                    raise ValueError(
                        f"cannot unpack {len(items)} values into {len(target.elts)}"
                    )
                for element, item in zip(target.elts, items, strict=True):
                    self.assign(element, item, frame, taint)
        elif isinstance(target, ast.Starred):
            self.assign(target.value, value, frame, taint)
        else:
            raise InterpreterError(f"assignment target {type(target).__name__}")

    def truth(self, value, node, frame) -> bool:
        """Whether ``value`` is true. The run's model may decide unknowns."""
        return bool(value)

    # ----------------------------------------------------------- functions

    def make_function(self, node, frame, name=None):
        args = node.args
        defaults = [self.eval(default, frame) for default in args.defaults]
        kw_defaults = {
            argument.arg: self.eval(default, frame)
            for argument, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
            if default is not None
        }
        function = SourceFunction(
            self,
            node,
            frame,
            defaults,
            kw_defaults,
            name or getattr(node, "name", "<lambda>"),
        )
        owner = frame
        prefix = []
        while owner is not None:
            if owner.is_class:
                prefix.append(owner.variables.get("__qualname__", ""))
                break
            if owner.function is not None:
                prefix.append(f"{owner.function.__qualname__}.<locals>")
                break
            owner = owner.enclosing
        if prefix and prefix[0]:
            function.__qualname__ = f"{prefix[0]}.{function.__name__}"
        if frame.is_class:
            function._class_cell = frame.class_cell
        else:
            function._class_cell = self._class_cell(frame)
        if not isinstance(node, ast.Lambda):
            annotations = {}
            for argument in args.posonlyargs + args.args + args.kwonlyargs:
                if argument.annotation is not None:
                    annotations[argument.arg] = self._annotation(
                        argument.annotation, frame
                    )
            if node.returns is not None:
                annotations["return"] = self._annotation(node.returns, frame)
            function.__annotations__ = annotations
        return function

    # ---------------------------------------------------------- expressions

    def eval(self, node, frame):
        handler = self._expression_handlers.get(type(node))
        if handler is None:
            handler = self._expression_handlers[type(node)] = getattr(
                self, f"_expr_{type(node).__name__}", None
            )
        if handler is None:
            raise InterpreterError(
                f"expression {type(node).__name__} at line {getattr(node, 'lineno', '?')}"
            )
        return handler(node, frame)

    def _expr_Constant(self, node, frame):
        return node.value

    def _expr_Name(self, node, frame):
        value = self.lookup(node.id, frame)
        if self._reads and (
            frame.taint is not None
            or frame.enclosing is not None
            or id(frame.module.__dict__) in self._module_taint
        ):
            self._note(self._name_taint(node.id, frame))
        return value

    def _expr_Attribute(self, node, frame):
        owner = self.eval(node.value, frame)
        value = getattr(owner, node.attr)
        if self._reads:
            if id(owner) in self._tainted_owners:
                self._note(self._attribute_taint.get((id(owner), node.attr)))
            option = self.environment.options.get(node.attr)
            if option is not None:
                # A parsed option read by its destination's name.
                self._note((option[0],))
        return value

    def _expr_Subscript(self, node, frame):
        owner = self.eval(node.value, frame)
        value = owner[self.eval(node.slice, frame)]
        if self._reads and id(owner) in self._tainted_owners:
            self._note(self._attribute_taint.get((id(owner), "[]")))
        return value

    def _expr_Slice(self, node, frame):
        return slice(
            None if node.lower is None else self.eval(node.lower, frame),
            None if node.upper is None else self.eval(node.upper, frame),
            None if node.step is None else self.eval(node.step, frame),
        )

    def _expr_Tuple(self, node, frame):
        return tuple(self._elements(node.elts, frame))

    def _expr_List(self, node, frame):
        return list(self._elements(node.elts, frame))

    def _expr_Set(self, node, frame):
        return set(self._elements(node.elts, frame))

    def _elements(self, elements, frame):
        for element in elements:
            if isinstance(element, ast.Starred):
                yield from self.eval(element.value, frame)
            else:
                yield self.eval(element, frame)

    def _expr_Dict(self, node, frame):
        result = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                result.update(self.eval(value, frame))
            else:
                result[self.eval(key, frame)] = self.eval(value, frame)
        return result

    def _expr_BinOp(self, node, frame):
        return _BINARY[type(node.op)](
            self.eval(node.left, frame), self.eval(node.right, frame)
        )

    def _expr_UnaryOp(self, node, frame):
        operand = self.eval(node.operand, frame)
        if isinstance(node.op, ast.Not):
            return not self.truth(operand, node.operand, frame)
        return _UNARY[type(node.op)](operand)

    def _expr_BoolOp(self, node, frame):
        is_and = isinstance(node.op, ast.And)
        value = None
        for operand in node.values:
            value = self.eval(operand, frame)
            if self.truth(value, operand, frame) != is_and:
                return value
        return value

    def _expr_Compare(self, node, frame):
        left = self.eval(node.left, frame)
        last = len(node.ops) - 1
        for index, (op, comparator) in enumerate(
            zip(node.ops, node.comparators, strict=True)
        ):
            right = self.eval(comparator, frame)
            result = _COMPARE[type(op)](left, right)
            # Only a chain (``a < b < c``) tests a link's truth.
            if index < last and not self.truth(result, node, frame):
                return result
            left = right
        return result

    def _expr_IfExp(self, node, frame):
        if self.truth(self.eval(node.test, frame), node.test, frame):
            return self.eval(node.body, frame)
        return self.eval(node.orelse, frame)

    def _expr_Call(self, node, frame):
        function = self.eval(node.func, frame)
        if (
            self._reads
            and _interpreted(function)
            and not any(isinstance(argument, ast.Starred) for argument in node.args)
        ):
            # Each argument's settings go to the parameter it binds; the
            # call's value carries what the callee returns it from.
            args, positional = [], []
            for argument in node.args:
                value, taint = self._tainted(argument, frame)
                args.append(value)
                positional.append(taint)
            kwargs, keyword = {}, {}
            for item in node.keywords:
                value, taint = self._tainted(item.value, frame)
                if item.arg is None:
                    kwargs.update(value)
                else:
                    kwargs[item.arg] = value
                    keyword[item.arg] = taint
            self._pending_arguments = (tuple(args), kwargs, positional, keyword)
            try:
                result = self.call(function, args, kwargs, node, frame)
            finally:
                self._pending_arguments = None
            self._note(self._return_taint)
            return result
        args = list(self._elements(node.args, frame))
        kwargs = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                kwargs.update(self.eval(keyword.value, frame))
            else:
                kwargs[keyword.arg] = self.eval(keyword.value, frame)
        if not self._reads:
            return self.call(function, args, kwargs, node, frame)
        # Host code (a decorator's wrapper) may run interpreted source: what
        # that returned its value from comes back with the result.
        self._return_taint = _NO_SETTINGS
        result = self.call(function, args, kwargs, node, frame)
        self._note(self._return_taint)
        return result

    def call(self, function, args, kwargs, node, frame):
        """Call ``function``. A caller's model of the world may intercept it."""
        if function is globals:
            return frame.module.__dict__
        if function is locals or (function is vars and not args):
            return frame.variables
        if function in (self._eval_builtin, self._exec_builtin) and len(args) < 2:
            # Without a namespace, eval/exec see the caller's own names.
            return function(
                args[0] if args else kwargs.pop("source"),
                frame.module.__dict__,
                frame.variables,
            )
        return function(*args, **kwargs)

    def _expr_Lambda(self, node, frame):
        return self.make_function(node, frame, "<lambda>")

    def _expr_JoinedStr(self, node, frame):
        return "".join(str(self.eval(value, frame)) for value in node.values)

    def _expr_FormattedValue(self, node, frame):
        value = self.eval(node.value, frame)
        if node.conversion == ord("r"):
            value = repr(value)
        elif node.conversion == ord("s"):
            value = str(value)
        elif node.conversion == ord("a"):
            value = ascii(value)
        spec = "" if node.format_spec is None else self.eval(node.format_spec, frame)
        return format(value, spec)

    def _expr_NamedExpr(self, node, frame):
        value = self.eval(node.value, frame)
        scope = frame
        while isinstance(scope, _ComprehensionFrame):
            scope = scope.enclosing
        self.store(node.target.id, value, scope)
        return value

    def _expr_Starred(self, node, frame):
        return self.eval(node.value, frame)

    def _comprehension(self, generators, frame, produce):
        # Comprehension variables are its own; other names read through.
        inner = _ComprehensionFrame(frame)

        def walk(index):
            if index == len(generators):
                produce(inner)
                return
            generator = generators[index]
            iterable = self.eval(generator.iter, frame if index == 0 else inner)
            for item in iterable:
                self.assign(generator.target, item, inner)
                if all(
                    self.truth(self.eval(test, inner), test, inner)
                    for test in generator.ifs
                ):
                    walk(index + 1)

        walk(0)

    def _expr_ListComp(self, node, frame):
        result = []
        self._comprehension(
            node.generators, frame, lambda f: result.append(self.eval(node.elt, f))
        )
        return result

    def _expr_SetComp(self, node, frame):
        result = set()
        self._comprehension(
            node.generators, frame, lambda f: result.add(self.eval(node.elt, f))
        )
        return result

    def _expr_DictComp(self, node, frame):
        result = {}

        def produce(f):
            result[self.eval(node.key, f)] = self.eval(node.value, f)

        self._comprehension(node.generators, frame, produce)
        return result

    def _expr_GeneratorExp(self, node, frame):
        # Produced eagerly: a forward only ever consumes these whole.
        return iter(self._expr_ListComp(node, frame))

    def _expr_Await(self, node, frame):
        raise InterpreterError("await")

    def _expr_Yield(self, node, frame):
        raise InterpreterError("yield outside a statement this interpreter resumes")

    _expr_YieldFrom = _expr_Yield


class _ComprehensionFrame(_Frame):
    """A comprehension's own variables over the enclosing frame's."""

    __slots__ = ()

    def __init__(self, outer):
        super().__init__(outer.module, enclosing=outer, function=outer.function)
        self.variables = _ChainVariables(outer)


class _ChainVariables(dict):
    """Comprehension variables; a miss reads the enclosing function's names."""

    def __init__(self, outer):
        super().__init__()
        self._outer = outer

    def get(self, name, default=None):
        if dict.__contains__(self, name):
            return dict.__getitem__(self, name)
        outer = self._outer
        while outer is not None and outer.is_class:
            outer = outer.enclosing
        if outer is None:
            return default
        return outer.variables.get(name, default)


# Module-namespace key marking ``from __future__ import annotations``.
_FUTURE_ANNOTATIONS = "__tracelens_postponed_annotations__"


def _resolve_relative(name: str, package: str, level: int) -> str:
    if level == 0:
        return name
    parts = package.split(".") if package else []
    if level > 1:
        parts = parts[: len(parts) - (level - 1)]
    base = ".".join(parts)
    return f"{base}.{name}" if name else base


class _Sink:
    def write(self, text):
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


def _refuse_exit(*_args, **_kwargs):
    raise InterpreterError("the interpreted source tried to exit or read input")


class _GeneratorRun:
    """An interpreted generator: the body runs lazily, one ``yield`` at a time.

    The body is evaluated by a Python generator that walks the same
    statements, so ``contextlib.contextmanager`` and ``for`` loops see an
    ordinary generator.
    """

    def __init__(self, interpreter, body, frame):
        self._generator = _walk_generator(interpreter, body, frame)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._generator)

    def send(self, value):
        return self._generator.send(value)

    def throw(self, *args):
        return self._generator.throw(*args)

    def close(self):
        return self._generator.close()


def _walk_generator(interpreter, body, frame):
    try:
        yield from _gen_block(interpreter, body, frame)
    except _Return as returned:
        return returned.value


def _gen_block(interpreter, body, frame):
    for statement in body:
        yield from _gen_statement(interpreter, statement, frame)


def _yield_value(node):
    """The ``Yield``/``YieldFrom`` a statement consists of, if it is one."""
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Yield | ast.YieldFrom):
        return node.value, None
    if isinstance(node, ast.Assign) and isinstance(
        node.value, ast.Yield | ast.YieldFrom
    ):
        return node.value, node.targets
    return None, None


def _gen_statement(interpreter, node, frame):
    if not _contains_yield_statement(node):
        signal = interpreter._exec(node, frame)
        if signal is _BREAK:
            raise _Break
        if signal is _CONTINUE:
            raise _Continue
        return
    expression, targets = _yield_value(node)
    if expression is not None:
        if isinstance(expression, ast.YieldFrom):
            sent = yield from interpreter.eval(expression.value, frame)
        else:
            value = (
                None
                if expression.value is None
                else interpreter.eval(expression.value, frame)
            )
            sent = yield value
        for target in targets or ():
            interpreter.assign(target, sent, frame)
        return
    if isinstance(node, ast.If):
        branch = (
            node.body
            if interpreter.truth(interpreter.eval(node.test, frame), node.test, frame)
            else node.orelse
        )
        yield from _gen_block(interpreter, branch, frame)
        return
    if isinstance(node, ast.For):
        for item in interpreter.eval(node.iter, frame):
            interpreter.assign(node.target, item, frame)
            try:
                yield from _gen_block(interpreter, node.body, frame)
            except _Break:
                return
            except _Continue:
                continue
        yield from _gen_block(interpreter, node.orelse, frame)
        return
    if isinstance(node, ast.While):
        while interpreter.truth(interpreter.eval(node.test, frame), node.test, frame):
            try:
                yield from _gen_block(interpreter, node.body, frame)
            except _Break:
                return
            except _Continue:
                continue
        yield from _gen_block(interpreter, node.orelse, frame)
        return
    if isinstance(node, ast.Try):
        try:
            try:
                yield from _gen_block(interpreter, node.body, frame)
            except (_Break, _Continue, _Return, GeneratorExit):
                raise
            except (
                BaseException
            ) as error:  # noqa: BLE001 - the source's handlers decide
                for handler in node.handlers:
                    if handler.type is None or isinstance(
                        error, interpreter.eval(handler.type, frame)
                    ):
                        if handler.name:
                            interpreter.store(handler.name, error, frame)
                        yield from _gen_block(interpreter, handler.body, frame)
                        break
                else:
                    raise
            else:
                yield from _gen_block(interpreter, node.orelse, frame)
        finally:
            for statement in node.finalbody:
                if _contains_yield_statement(statement):
                    raise InterpreterError("yield in finally")
                interpreter._exec(statement, frame)
        return
    if isinstance(node, ast.With):
        managers = []
        for item in node.items:
            manager = interpreter.eval(item.context_expr, frame)
            entered = type(manager).__enter__(manager)
            managers.append(manager)
            if item.optional_vars is not None:
                interpreter.assign(item.optional_vars, entered, frame)
        try:
            yield from _gen_block(interpreter, node.body, frame)
        except BaseException as error:
            pending = error
            for manager in reversed(managers):
                if pending is not None and type(manager).__exit__(
                    manager, type(pending), pending, pending.__traceback__
                ):
                    pending = None
            if pending is not None:
                raise
        else:
            for manager in reversed(managers):
                type(manager).__exit__(manager, None, None, None)
        return
    raise InterpreterError(
        f"yield inside {type(node).__name__} at line {getattr(node, 'lineno', '?')}"
    )


def _contains_yield_statement(node) -> bool:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Yield | ast.YieldFrom):
            return True
    return False


__all__ = [
    "Environment",
    "InterpreterError",
    "SourceFunction",
    "SourceInterpreter",
    "SourceUnavailable",
]
