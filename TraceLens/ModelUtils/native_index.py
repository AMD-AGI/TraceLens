###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""What a kernel library's compiled extensions expose, read from their C++ source.

A compiled extension module is native code, so it never runs here. Its Python
face is declared in C++ binding code, and that is what is read:

- ``pybind11::enum_<T>(m, "Name").value("a", ...)`` declares an enum class,
  whose member values come from the C++ ``enum`` definition of ``T``;
- ``m.def("name", &function, py::arg("x"), ...)`` declares a function, whose
  parameter types come from the C++ declaration of ``function``;
- ``__global__`` functions are the device kernels the library defines.

Python code that inspects a native function (its pybind docstring) or uses a
native enum sees what the C++ source declares.
"""

from __future__ import annotations

import enum
import functools
import re
from dataclasses import dataclass, field
from pathlib import Path

_SOURCES = (".cpp", ".cc", ".cu", ".cuh", ".h", ".hpp", ".hip", ".inc")
_ENUM_BINDING = re.compile(
    r"(?:pybind11|py)::enum_<\s*([\w:]+)\s*>\s*\(\s*\w+\s*,\s*\"(\w+)\"(.*?);",
    re.DOTALL,
)
_ENUM_VALUE = re.compile(r"\.value\(\s*\"(\w+)\"\s*,\s*([\w:]+)\s*\)")
_ENUM_DEFINITION = re.compile(
    r"\benum\s+(?:class\s+|struct\s+)?(\w+)\s*(?::\s*[\w:\s]+)?\{([^}]*)\}",
    re.DOTALL,
)
_DEF = re.compile(r"\bm\.def\(\s*\"(\w+)\"\s*,\s*&?\s*([\w:]+)?(.*?)\)\s*;", re.DOTALL)
_ARG = re.compile(r"py(?:bind11)?::arg\(\s*\"(\w+)\"\s*\)(\s*=\s*([^,]+?))?\s*(?=,|$)")
_LAUNCH_BOUNDS = re.compile(r"__launch_bounds__\s*\([^)]*\)")
# How far after ``__global__`` a kernel's name is looked for.
_KERNEL_HEAD_SPAN = 600


def _kernel_names(text: str) -> list[str]:
    """Names of the ``__global__`` functions ``text`` declares.

    ``__global__ void __launch_bounds__(512, 1) name(``: the launch bounds
    are not the name; the name is the last word before the parameters.
    """
    names = []
    for match in re.finditer(r"\b__global__\b", text):
        head = _LAUNCH_BOUNDS.sub(
            " ", text[match.end() : match.end() + _KERNEL_HEAD_SPAN]
        )
        before, parenthesis, _ = head.partition("(")
        if not parenthesis or any(mark in before for mark in ";{}"):
            continue
        # A specialization's template arguments (``name<float>(``) are not its name.
        while True:
            stripped = re.sub(r"<[^<>]*>", " ", before)
            if stripped == before:
                break
            before = stripped
        words = re.findall(r"\w+", before)
        if words:
            names.append(words[-1])
    return names


@dataclass
class NativeSignature:
    name: str
    # (name, Python type, default as Python source or None)
    parameters: list[tuple] = field(default_factory=list)
    returns: str = "None"

    def docstring(self) -> str:
        """The first line pybind writes for a bound function."""
        params = ", ".join(
            f"{name}: {kind}" + ("" if default is None else f" = {default}")
            for name, kind, *rest in self.parameters
            for default in (rest[0] if rest else None,)
        )
        return f"{self.name}({params}) -> {self.returns}\n"


def _python_default(cpp: str) -> str:
    """A ``py::arg(...) = <C++ expression>`` default as pybind shows it."""
    text = cpp.strip()
    if text in (
        "py::none()",
        "pybind11::none()",
        "std::nullopt",
        "nullptr",
        "c10::nullopt",
    ):
        return "None"
    if text in ("true", "false"):
        return text.capitalize()
    number = re.fullmatch(r"-?\d+(?:\.\d*)?(?:[eE]-?\d+)?[fFuUlL]*", text)
    if number:
        return re.sub(r"[fFuUlL]+$", "", text)
    if re.fullmatch(r"\"[^\"]*\"", text):
        return repr(text[1:-1])
    return "None"


@dataclass
class NativeIndex:
    """Enums, functions, ops and kernels a library's C++ source declares."""

    enums: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    signatures: dict[str, NativeSignature] = field(default_factory=dict)
    kernels: dict[str, set[str]] = field(default_factory=dict)
    # ``TORCH_LIBRARY(namespace, m) { m.def("schema") }``: (namespace, name) -> schema
    op_schemas: dict[tuple[str, str], str] = field(default_factory=dict)
    # ``m.def("name", &function)``: (namespace, name) -> the C++ function it binds.
    op_functions: dict[tuple[str, str], str] = field(default_factory=dict)
    # Compiled extension modules a build script declares, by dotted name.
    extensions: set[str] = field(default_factory=set)
    # Scalar host functions, by name (see native_host).
    host_functions: dict = field(default_factory=dict)
    # Symbols each source file exports (pybind names, extern "C" functions).
    exports: dict[str, set[str]] = field(default_factory=dict)
    # Compiled module name -> path suffixes of the sources its build lists.
    module_sources: dict[str, list[str]] = field(default_factory=dict)
    # Classes the bindings declare (``pybind11::class_<T>(m, "Name")``).
    classes: set[str] = field(default_factory=set)
    # Host function -> the kernels its body launches, in order.
    launches: dict[str, list[str]] = field(default_factory=dict)
    # Function -> the identifiers its body names (the functions it calls, the
    # kernels it may launch through a macro), as a ROCm build reads it.
    named: dict[str, set[str]] = field(default_factory=dict)

    def merge(self, other: NativeIndex) -> None:
        """Add ``other``'s declarations; what this index already holds stays."""
        for name in (
            "enums",
            "signatures",
            "kernels",
            "op_schemas",
            "op_functions",
            "host_functions",
            "exports",
            "module_sources",
            "launches",
            "named",
        ):
            mine = getattr(self, name)
            for key, value in getattr(other, name).items():
                mine.setdefault(key, value)
        self.extensions |= other.extensions
        self.classes |= other.classes
        self.__dict__.pop("enum_classes", None)

    def kernel_alternatives(self, name: str) -> tuple[str, ...]:
        """Kernels host function ``name`` may launch through the functions it calls.

        A function that launches no kernel itself reaches one through calls
        (a member template choosing among kernels by size). Which one it
        chooses is decided where its conditions are evaluated; here the
        kernels named nearest it, in the fewest calls, are the candidates,
        and the replay's kernel names pick among them. Too many to tell
        apart are none.
        """
        kernels = {
            kernel
            for found in self.kernels.values()
            for kernel in found
            if re.search("[A-Za-z]", kernel)
        }
        seen, frontier = {name}, [name]
        for _ in range(_CALL_DEPTH):
            following = []
            found = set()
            for function in frontier:
                identifiers = self.named.get(function, set())
                found |= identifiers & kernels
                following.extend(sorted(identifiers - seen - kernels))
            if found:
                return tuple(sorted(found)) if len(found) <= _MOST_ALTERNATIVES else ()
            seen.update(following)
            frontier = [function for function in following if function in self.named]
        return ()

    def symbols_of(self, module: str) -> set[str] | None:
        """What compiled module ``module`` exports, or None when unknown."""
        stem = module.rpartition(".")[2]
        suffixes = self.module_sources.get(stem)
        if not suffixes:
            return None
        names: set[str] = set()
        for path, exported in self.exports.items():
            if any(path.endswith(suffix) for suffix in suffixes):
                names |= exported
        return names

    @functools.cached_property
    def enum_classes(self) -> dict[str, type]:
        classes = {}
        for name, members in self.enums.items():
            if not members:
                continue
            cls = enum.IntEnum(name, dict(members))
            # A pybind11 enum member reads as ``Name.member`` (an IntEnum's
            # reads as its number), in ``str`` and in formatting alike.
            cls.__str__ = lambda self: f"{type(self).__name__}.{self.name}"
            cls.__format__ = lambda self, spec: format(str(self), spec)
            # The docstring pybind11 writes for an enum.
            cls.__doc__ = "Members:\n\n" + "\n\n".join(
                f"  {member}" for member, _ in members
            )
            classes[name] = cls
        return classes


def _strip(text: str) -> str:
    """Source without comments, line continuations, or ``__attribute__((...))``."""
    text = re.sub(r"\\\s*\n", " ", text)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    return re.sub(r"__attribute__\s*\(\((?:[^()]|\([^()]*\))*\)\)", " ", text)


def _enum_values(body: str) -> list[tuple[str, int]]:
    members = []
    value = 0
    for item in body.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, explicit = item.partition("=")
        name = name.strip()
        if explicit.strip():
            try:
                value = int(explicit.strip().rstrip("uUlL"), 0)
            except ValueError:
                pass
        members.append((name, value))
        value += 1
    return members


_CPP_TO_PYTHON = (
    (
        re.compile(r"optional<\s*(?:const\s+)?(?:torch|at)::Tensor"),
        "Optional[torch.Tensor]",
    ),
    (re.compile(r"vector<\s*(?:const\s+)?(?:torch|at)::Tensor"), "list[torch.Tensor]"),
    (re.compile(r"(?:torch|at)::Tensor"), "torch.Tensor"),
    (re.compile(r"optional<\s*(?:int|int64_t|long|int32_t|uint32_t)"), "Optional[int]"),
    (re.compile(r"optional<\s*(?:float|double)"), "Optional[float]"),
    (re.compile(r"optional<\s*bool"), "Optional[bool]"),
    (re.compile(r"optional<\s*std::string"), "Optional[str]"),
    (re.compile(r"vector<\s*(?:int|int64_t|long|int32_t)|IntArrayRef"), "list[int]"),
    (re.compile(r"vector<\s*(?:float|double)"), "list[float]"),
    (re.compile(r"\b(?:int|int64_t|long|int32_t|uint32_t|size_t|uint64_t)\b"), "int"),
    (re.compile(r"\b(?:float|double)\b"), "float"),
    (re.compile(r"\bbool\b"), "bool"),
    (re.compile(r"std::string"), "str"),
)


def _python_type(cpp: str, enums: dict[str, list]) -> str:
    for pattern, python in _CPP_TO_PYTHON:
        if pattern.search(cpp):
            return python
    bare = re.sub(r"[&*]|\bconst\b", " ", cpp).split()
    if bare and bare[-1].split("::")[-1] in enums:
        return bare[-1].split("::")[-1]
    return "object"


def _declarations(text: str) -> dict[str, tuple[str, list[str]]]:
    """``name -> (return type, [parameter declarations])`` of C++ functions."""
    found = {}
    for match in _DECLARED.finditer(text):
        name, params = match.groups()
        # The return type is the run of type characters just before the name,
        # read backwards (a pattern reading it forwards backtracks badly).
        end = match.start()
        if (
            name in ("if", "for", "while", "switch", "return")
            or not end
            or not text[end - 1].isspace()
        ):
            continue
        begin = end
        while (
            begin > max(0, end - _RETURN_TYPE_SPAN)
            and text[begin - 1] in _TYPE_CHARACTERS
        ):
            begin -= 1
        returns = text[begin:end].split()
        if not returns:
            continue
        parts = [part.strip() for part in params.split(",") if part.strip()]
        found.setdefault(name, (returns[-1], parts))
    return found


_DECLARED = re.compile(r"\b(\w+)\s*\(([^;{}()]*(?:\([^)]*\)[^;{}()]*)*)\)\s*[;{]")
_TYPE_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:<>, \t\n\r"
)
_RETURN_TYPE_SPAN = 400


_TORCH_LIBRARY = re.compile(
    r"\bTORCH_LIBRARY(?:_EXPAND|_FRAGMENT)?\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\{"
)
_SCHEMA_DEF = re.compile(r"\b(\w+)\.def\(\s*((?:\"(?:[^\"\\]|\\.)*\"\s*)+)")
_EXTENSION = re.compile(
    r"\b(?:CUDAExtension|CppExtension|Extension|HIPExtension)\(\s*(?:name\s*=\s*)?\"([\w.]+)\""
)


def _block(text: str, start: int) -> str:
    """The brace-balanced block whose ``{`` precedes ``start``."""
    depth = 1
    index = start
    while index < len(text) and depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    return text[start:index]


_FUNCTION_START = re.compile(
    r"\b([A-Za-z_]\w*)\s*\((?:[^;{}()]|\([^()]*\))*\)\s*(?:const\s*)?(?:noexcept\s*)?\{"
)
_NOT_FUNCTIONS = frozenset(
    {"if", "for", "while", "switch", "catch", "return", "sizeof", "decltype"}
)
_GGL = re.compile(
    r"\bhipLaunchKernelGGL\s*\(\s*(?:HIP_KERNEL_NAME\s*\(\s*)?([A-Za-z_][\w:]*)"
)
# ``LaunchKernel(grid, block, device)(kernel, args...)``: a launcher object called with the kernel.
_LAUNCHER = re.compile(
    r"\bLaunchKernel\s*\((?:[^()]|\([^()]*\))*\)\s*\(\s*([A-Za-z_][\w:]*)"
)


def _launch_names(body: str) -> list[str]:
    """Kernels ``body`` launches (``k<...><<<...>>>``, ``hipLaunchKernelGGL``), in order."""
    found = [
        (match.start(), match.group(1).split("::")[-1])
        for pattern in (_GGL, _LAUNCHER)
        for match in pattern.finditer(body)
    ]
    for match in re.finditer(r"<<<", body):
        index = match.start() - 1
        while index >= 0 and body[index].isspace():
            index -= 1
        if index >= 0 and body[index] == ">":
            # Template arguments: skip them, balanced.
            depth = 0
            while index >= 0:
                if body[index] == ">":
                    depth += 1
                elif body[index] == "<":
                    depth -= 1
                    if depth == 0:
                        index -= 1
                        break
                index -= 1
            while index >= 0 and body[index].isspace():
                index -= 1
        end = index + 1
        while index >= 0 and (body[index].isalnum() or body[index] in "_:"):
            index -= 1
        name = body[index + 1 : end].split("::")[-1]
        if name:
            found.append((match.start(), name))
    return [name for _, name in sorted(found)]


def launch_sequences(text: str) -> dict[str, list[str]]:
    """Host function -> the kernels it launches, for each function in ``text``."""
    sequences: dict[str, list[str]] = {}
    if (
        "<<<" not in text
        and "hipLaunchKernelGGL" not in text
        and "LaunchKernel" not in text
    ):
        return sequences
    # A parameter list split by ``#if`` arms would otherwise run a function's
    # body on into the next ones; a string's words are no code.
    text = _STRING.sub('""', _rocm_source(text))
    for match in _FUNCTION_START.finditer(text):
        name = match.group(1)
        if name in _NOT_FUNCTIONS:
            continue
        # A parameter or a local launched (``kernelName<<<...>>>``,
        # ``auto kernel = select<...>(); LaunchKernel(...)(kernel, ...)``)
        # holds whichever kernel was chosen: it names none.
        body = _block(text, match.end())
        holders = set(_IDENTIFIER.findall(match.group(0).partition("(")[2])) | set(
            _LOCAL.findall(body)
        )
        launched = [
            kernel
            for kernel in _launch_names(body)
            if kernel != _MACRO_ARGUMENT and kernel not in holders
        ]
        if launched:
            sequences.setdefault(name, launched)
    return sequences


# How many calls deep a host function's kernels are looked for, and how many
# candidates still tell a launch apart.
_CALL_DEPTH = 4
_MOST_ALTERNATIVES = 16
# What a ROCm build of the libraries defines.
_ROCM_DEFINES = {"USE_ROCM": 1, "__HIP_PLATFORM_AMD__": 1, "__HIPCC__": 1}
# Macros a build defines or not by platform: ``#ifdef`` on one is decided.
_PLATFORM_MACROS = {
    *_ROCM_DEFINES,
    "__CUDACC__",
    "__NVCC__",
    "USE_CUDA",
    "__HIP_PLATFORM_NVIDIA__",
    "__HIP_PLATFORM_NVCC__",
    # The device pass's: host code (its launches) is read as the host pass reads it.
    "__HIP_DEVICE_COMPILE__",
    "__CUDA_ARCH__",
}
_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
# A local a body declares with a value: ``const auto kernel = ...``.
_LOCAL = re.compile(r"\b(?:auto|[\w:]+(?:<[^<>;{}]*>)?\s*[*&]?)\s+(\w+)\s*=(?!=)")
_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')


# What a macro's parameter reads as in its body: no kernel's name.
_MACRO_ARGUMENT = "__macro_argument__"


def _rocm_source(text: str) -> str:
    """``text`` as a ROCm build reads it; an ``#if`` it cannot decide keeps its first arm.

    A ``#define``'s body stays where it is defined, its parameters blanked:
    ``KL(name)`` launching ``name<<<...>>>`` launches whatever it is handed.
    """
    from TraceLens.ModelUtils.native_host import _directive_value

    lines, taking, decided = [], [True], [True]
    for line in text.splitlines():
        directive = re.match(r"\s*#\s*(\w+)\s*(.*)", line)
        word, rest = directive.groups() if directive else ("", "")
        if word in ("if", "ifdef", "ifndef"):
            if word == "if":
                value = _directive_value(rest, _ROCM_DEFINES)
            elif rest.split() and rest.split()[0] in _PLATFORM_MACROS:
                value = (rest.split()[0] in _ROCM_DEFINES) == (word == "ifdef")
            else:
                # A switch the including file sets (an implementation
                # switch, an include guard): this file cannot decide it.
                value = None
            value = True if value is None else bool(value)
            taking.append(value)
            decided.append(value)
        elif word == "elif" and len(taking) > 1:
            value = _directive_value(rest, _ROCM_DEFINES)
            taking[-1] = not decided[-1] and (value is None or bool(value))
            decided[-1] = decided[-1] or taking[-1]
        elif word == "else" and len(taking) > 1:
            taking[-1] = not decided[-1]
        elif word == "endif" and len(taking) > 1:
            taking.pop()
            decided.pop()
        elif all(taking) and not word:
            lines.append(line)
        elif all(taking) and word == "define":
            # A macro defined in a function body is part of what it names.
            macro = re.match(r"\w+\(([^)]*)\)(.*)", rest)
            if macro is not None:
                parameters = [
                    name.strip() for name in macro.group(1).split(",") if name.strip()
                ]
                rest = macro.group(2)
                for parameter in parameters:
                    rest = re.sub(rf"\b{re.escape(parameter)}\b", _MACRO_ARGUMENT, rest)
            lines.append(rest)
    return "\n".join(lines)


def _named_identifiers(text: str) -> dict[str, set[str]]:
    """Function -> the identifiers its body names, for each function ``text`` defines."""
    if "(" not in text:
        return {}
    source = _rocm_source(text)
    found: dict[str, set[str]] = {}
    for match in _FUNCTION_START.finditer(source):
        name = match.group(1)
        if name in _NOT_FUNCTIONS:
            continue
        found.setdefault(name, set()).update(
            _IDENTIFIER.findall(_block(source, match.end()))
        )
    return found


def _op_schemas(text: str) -> dict[tuple[str, str], str]:
    schemas = {}
    for match in _TORCH_LIBRARY.finditer(text):
        namespace, handle = match.groups()
        body = _block(text, match.end())
        for receiver, literals in _SCHEMA_DEF.findall(body):
            if receiver != handle:
                continue
            schema = "".join(re.findall(r"\"((?:[^\"\\]|\\.)*)\"", literals))
            if "(" in schema:
                name = schema.split("(", 1)[0].strip().split(".")[0]
                schemas.setdefault((namespace, name), schema)
    return schemas


# ``m.def("name", &function)``: an op whose schema torch infers from the C++ signature.
_POINTER_DEF = re.compile(r"\b(\w+)\.def\(\s*\"(\w+)\"\s*,\s*&\s*([\w:]+)\s*\)")
_TYPE_ALIAS = re.compile(
    r"\busing\s+(\w+)\s*=\s*([\w:]+)\s*;|\btypedef\s+([\w:]+)\s+(\w+)\s*;"
)
# The schema type torch infers for each C++ scalar type.
_SCHEMA_SCALARS = {
    "Tensor": "Tensor",
    "int64_t": "int",
    "int": "int",
    "long": "int",
    "double": "float",
    "float": "float",
    "bool": "bool",
    "string": "str",
    "string_view": "str",
    "IntArrayRef": "int[]",
    "ScalarType": "ScalarType",
    "Scalar": "Scalar",
    "Device": "Device",
}


def _schema_type(cpp: str, aliases: dict[str, str]) -> str | None:
    """The schema type torch infers for C++ type ``cpp`` (``None``: one it does not)."""
    text = re.sub(r"\bconst\b|&", " ", cpp)
    text = re.sub(r"\b(?:std|c10|at|torch)::", "", text).replace(" ", "")
    text = aliases.get(text, text)
    inner = re.fullmatch(r"(optional|vector|ArrayRef|tuple)<(.*)>", text)
    if inner is None:
        return _SCHEMA_SCALARS.get(text.split("::")[-1])
    kind, argument = inner.groups()
    if kind == "tuple":
        parts = [_schema_type(part, aliases) for part in _split_top_level(argument)]
        return None if None in parts else f"({', '.join(parts)})"
    element = _schema_type(argument, aliases)
    if element is None or element.endswith(("]", "?", ")")):
        return None
    return f"{element}?" if kind == "optional" else f"{element}[]"


def _split_top_level(text: str) -> list[str]:
    parts, depth, current = [], 0, ""
    for character in text:
        depth += character == "<"
        depth -= character == ">"
        if character == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += character
    return [*parts, current]


def _pointer_ops(text: str) -> dict[tuple[str, str], str]:
    """``(namespace, op) -> C++ function`` of ops registered by function pointer."""
    found = {}
    for match in _TORCH_LIBRARY.finditer(text):
        namespace, handle = match.groups()
        for receiver, name, function in _POINTER_DEF.findall(_block(text, match.end())):
            if receiver == handle:
                found.setdefault((namespace, name), function.split("::")[-1])
    return found


def _inferred_schema(
    name: str, declared: tuple[str, list[str]], aliases: dict[str, str]
) -> str | None:
    """The schema torch infers for an op bound to a function declared ``declared``."""

    def schema_type(cpp: str) -> str | None:
        words = re.sub(
            r"\b(?:\w+::)*(\w+)\b", lambda m: aliases.get(m.group(1), m.group(0)), cpp
        )
        return _schema_type(words, {})

    returns, parameters = declared
    arguments = []
    for position, parameter in enumerate(parameters):
        # The declared name and default, when there are any, are not part of the type.
        words = parameter.split("=", 1)[0].rsplit(None, 1)
        cpp = (
            words[0]
            if len(words) == 2 and not words[1].endswith((">", "&", "*"))
            else parameter
        )
        kind = schema_type(cpp)
        if kind is None:
            return None
        arguments.append(f"{kind} _{position}")
    if returns.split("::")[-1] == "void":
        result = "()"
    else:
        result = schema_type(returns)
        if result is None:
            return None
    return f"{name}({', '.join(arguments)}) -> {result}"


def declared_extensions(roots: list[Path]) -> set[str]:
    """Dotted names of the compiled extensions the roots' build scripts declare."""
    names = set()
    for root in roots:
        for path in Path(root).rglob("setup*.py"):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            names.update(_EXTENSION.findall(text))
    return names


_EXTERN_C = re.compile(r"extern\s+\"C\"[^;{]*?\b(\w+)\s*\([^;{]*\)\s*\{")
_C_ITFS = re.compile(r"\b[A-Z_]*ITFS\b[^;{(]*?\b(\w+)\s*\(")
_MACRO = re.compile(r"^\s*#\s*define\s+(\w+)(?:\([^)]*\))?\s+(.*)$", re.MULTILINE)
_MACRO_PARAMETERS = re.compile(
    r"^\s*#\s*define\s+(\w+)\(([^)]*)\)\s+(.*)$", re.MULTILINE
)


def exporting_macros(texts) -> dict[str, int]:
    """Function-like macros that export a C function named by a parameter.

    ``#define ENTRY(func, ARGS, CALL) ... ITFS int func ARGS {...}``: each
    ``ENTRY(name, ...)`` exports ``name``. Maps the macro to that argument's
    position.
    """
    found: dict[str, int] = {}
    for text in texts:
        for name, parameters, body in _MACRO_PARAMETERS.findall(text):
            params = [param.strip() for param in parameters.split(",")]
            for position, param in enumerate(params):
                if not re.fullmatch(r"\w+", param):
                    continue
                # ``ITFS int func ARGS``: the declarator names the function.
                declarator = (
                    rf"(?:\b[A-Z_]*ITFS\b|extern\s+\"C\")[^;{{]*?\b{param}\b(?!##)"
                )
                if re.search(declarator, body):
                    found.setdefault(name, position)
                    break
    return found


def _macro_exports(text: str, macros: dict[str, int]) -> set[str]:
    """Names ``text`` exports by invoking the macros of :func:`exporting_macros`."""
    names = set()
    for macro, position in macros.items():
        for match in re.finditer(rf"\b{macro}\s*\(", text):
            depth, start, args = 1, match.end(), []
            index = start
            while index < len(text) and depth:
                char = text[index]
                if char in "([{":
                    depth += 1
                elif char in ")]}":
                    depth -= 1
                if (char == "," and depth == 1) or depth == 0:
                    args.append(text[start:index].strip())
                    start = index + 1
                index += 1
            if position < len(args) and re.fullmatch(r"\w+", args[position]):
                names.add(args[position])
    return names


_SOURCE_LIST = re.compile(r"sources\s*=\s*\[([^\]]*)\]", re.DOTALL)


_CLASS_BINDING = re.compile(
    r"(?:pybind11|py)::class_<\s*[\w:]+[^>]*>\s*\(\s*\w+\s*,\s*\"(\w+)\""
)


def _invokes(text: str, macro: str) -> bool:
    """Whether ``text`` invokes ``macro`` (on a line of its own, or inline:
    ``PYBIND11_MODULE(name, m) { MACRO; }``), not where it is defined."""
    for match in re.finditer(rf"\b{macro}\b", text):
        line = text[text.rfind("\n", 0, match.start()) + 1 : match.start()]
        if not re.match(r"\s*#", line):
            return True
    return False


def _file_exports(text: str, macros: dict[str, str]) -> set[str]:
    names = set(re.findall(r"\bm\.def\(\s*\"(\w+)\"", text))
    names |= {name for _, name, _ in _ENUM_BINDING.findall(text)}
    names |= set(_CLASS_BINDING.findall(text))
    names |= set(_EXTERN_C.findall(text)) | set(_C_ITFS.findall(text))
    used = [macro for macro in macros if _invokes(text, macro)]
    seen: set[str] = set()
    while used:
        macro = used.pop()
        if macro in seen:
            continue
        seen.add(macro)
        body = macros[macro]
        names |= set(_C_ITFS.findall(body)) | set(_EXTERN_C.findall(body))
        names |= set(re.findall(r"\bm\.def\(\s*\"(\w+)\"", body))
        names |= set(_CLASS_BINDING.findall(body))
        names |= {name for _, name, _ in _ENUM_BINDING.findall(body)}
        # A binding macro may expand others.
        used.extend(
            name for name in re.findall(r"\b([A-Z_]{4,})\b", body) if name in macros
        )
    return names


def build_manifests(roots: list[Path]) -> dict[str, list[str]]:
    """Compiled module -> source path suffixes, from the roots' build files.

    A ``setup*.py`` ``Extension(name, sources=[...])`` lists its sources, and
    a JSON build manifest maps module names to entries with a ``srcs`` list.
    A source given relative to a directory variable is kept as the path after
    its last placeholder, which is enough to find it in the checkout.
    """
    import json

    manifests: dict[str, list[str]] = {}

    def suffix(entry: str) -> str:
        entry = entry.strip().strip("'\"").removeprefix("f").strip("'\"")
        return entry.rpartition("}")[2] if "}" in entry else "/" + entry.lstrip("./")

    for root in roots:
        for path in Path(root).rglob("*.json"):
            try:
                data = json.loads(path.read_text(errors="ignore"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            for name, entry in data.items():
                if isinstance(entry, dict) and isinstance(entry.get("srcs"), list):
                    manifests.setdefault(name, []).extend(
                        suffix(str(item)) for item in entry["srcs"]
                    )
        for path in Path(root).rglob("setup*.py"):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            for name, sources in re.findall(
                r"Extension\(\s*(?:name\s*=\s*)?\"([\w.]+)\".*?" + _SOURCE_LIST.pattern,
                text,
                re.DOTALL,
            ):
                manifests.setdefault(name.rpartition(".")[2], []).extend(
                    suffix(item) for item in re.findall(r"\"([^\"]+)\"", sources)
                )
    return manifests


def build_native_index(roots: list[Path], build_roots: list[Path] = ()) -> NativeIndex:
    index = NativeIndex()
    index.extensions = declared_extensions(list(build_roots) or list(roots))
    index.module_sources = build_manifests(list(build_roots) or list(roots))
    texts = []
    for root in roots:
        for path in Path(root).rglob("*"):
            if path.suffix in _SOURCES and path.is_file():
                try:
                    texts.append((path, _strip(path.read_text(errors="ignore"))))
                except OSError:
                    continue
    definitions: dict[str, list[tuple[str, int]]] = {}
    # Every declaration of a name: overloads share it.
    declarations: dict[str, list[tuple[str, list[str]]]] = {}
    pointer_ops: dict[tuple[str, str], str] = {}
    aliases: dict[str, str] = {}
    from TraceLens.ModelUtils.native_host import host_functions

    macros = {}
    for path, text in texts:
        if path.suffix in (".h", ".hpp", ".cuh", ".inc"):
            for name, body in _MACRO.findall(text):
                if any(
                    marker in body
                    for marker in (
                        "ITFS",
                        "extern",
                        "m.def(",
                        "class_<",
                        "enum_<",
                        "PYBIND",
                    )
                ):
                    macros.setdefault(name, body)
    exporters = exporting_macros(
        [text for path, text in texts if path.suffix in (".h", ".hpp", ".cuh", ".inc")]
    )
    for path, text in texts:
        index.exports[str(path)] = _file_exports(text, macros) | _macro_exports(
            text, exporters
        )
        index.classes.update(_CLASS_BINDING.findall(text))
    for body in macros.values():
        index.classes.update(_CLASS_BINDING.findall(body))
    for path, text in texts:
        for name, function in host_functions(text).items():
            index.host_functions.setdefault(name, function)
        if "TORCH_LIBRARY" in text:
            for key, schema in _op_schemas(text).items():
                index.op_schemas.setdefault(key, schema)
            for key, function in _pointer_ops(text).items():
                pointer_ops.setdefault(key, function)
        for alias, target, typedef_target, typedef_alias in _TYPE_ALIAS.findall(text):
            aliases.setdefault(
                alias or typedef_alias, (target or typedef_target).split("::")[-1]
            )
        for name, body in _ENUM_DEFINITION.findall(text):
            definitions.setdefault(name, _enum_values(body))
        if (
            "m.def(" in text
            or "__global__" in text
            or path.suffix in (".h", ".hpp", ".cuh")
        ):
            # Bindings name functions a header declares.
            for name, value in _declarations(text).items():
                if value not in declarations.setdefault(name, []):
                    declarations[name].append(value)
        for kernel in _kernel_names(text):
            index.kernels.setdefault(str(path), set()).add(kernel)
        for name, launched in launch_sequences(text).items():
            index.launches.setdefault(name, launched)
        for name, identifiers in _named_identifiers(text).items():
            index.named.setdefault(name, set()).update(identifiers)
    for path, text in texts:
        for cpp_type, python_name, chain in _ENUM_BINDING.findall(text):
            values = dict(definitions.get(cpp_type.split("::")[-1], []))
            members = []
            for member, qualified in _ENUM_VALUE.findall(chain):
                key = qualified.split("::")[-1]
                members.append((member, values.get(key, len(members))))
            index.enums.setdefault(python_name, members)
    for path, text in texts:
        for name, function, rest in _DEF.findall(text):
            arguments = _ARG.findall(rest)
            overloads = declarations.get((function or "").split("::")[-1]) or [None]
            # The overload a binding names takes as many parameters as it lists.
            declared = next(
                (
                    value
                    for value in overloads
                    if value is not None and len(value[1]) == len(arguments)
                ),
                overloads[0],
            )
            parameters = []
            for position, (argument, assigned, default) in enumerate(arguments):
                cpp = (
                    declared[1][position]
                    if declared and position < len(declared[1])
                    else ""
                )
                parameters.append(
                    (
                        argument,
                        _python_type(cpp, index.enums),
                        _python_default(default) if assigned else None,
                    )
                )
            if not arguments and declared:
                for position, cpp in enumerate(declared[1]):
                    parameters.append(
                        (f"arg{position}", _python_type(cpp, index.enums))
                    )
            returns = "None"
            if declared:
                returns = {
                    "void": "None",
                }.get(declared[0], _python_type(declared[0], index.enums))
            index.signatures.setdefault(
                name, NativeSignature(name, parameters, returns)
            )
    index.op_functions.update(pointer_ops)
    for (namespace, name), function in pointer_ops.items():
        if (namespace, name) in index.op_schemas or function not in declarations:
            continue
        schema = _inferred_schema(name, declarations[function][0], aliases)
        if schema is not None:
            index.op_schemas[(namespace, name)] = schema
    return index
