###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Host functions in a library's C/C++ source, evaluated from their text.

A compiled extension also exports small host functions that compute a value
the Python side branches on: a padded size, an ABI version, a split count.
Their bodies are arithmetic over scalars. A body in that subset (declarations,
assignments, ``if``/``else``, ``return``, arithmetic, calls to other such
functions) is translated to Python source and evaluated by the source
interpreter; any other body is left alone and its value stays unknown.
"""

from __future__ import annotations

import builtins
import functools
import re
from dataclasses import dataclass

_SCALAR_TYPES = (
    r"(?:unsigned\s+|signed\s+|const\s+)*(?:\w+::)*"
    r"(?:int|long|long\s+long|short|float|double|bool|size_t|"
    r"u?int(?:8|16|32|64)_t|index_t|auto)"
)
# C functions Python's own builtins compute the same way: a library's
# overload of one (often a device helper) never stands in for them.
_PYTHON_SAME = frozenset({"min", "max", "abs"})
# ``#define NAME <integer>``: a value a body's ``#if`` may test.
_INTEGER_DEFINE = re.compile(
    r"^\s*#\s*define\s+(\w+)\s+\(?\s*(-?\d+)\s*\)?\s*$", re.MULTILINE
)
_DEFINITION = re.compile(
    rf"(?:^|[\s;}}])(?:extern\s+\"C\"\s+)?(?:[A-Z_]+\s+)?(?:static\s+|inline\s+|constexpr\s+|__host__\s+|__device__\s+)*"
    rf"({_SCALAR_TYPES})\s+(\w+)\s*\(([^()]*)\)\s*\{{"
)
_TOKEN = re.compile(
    r"\s*(?:(\d+\.\d*(?:[eE][-+]?\d+)?[fFlL]?|\d+[uUlL]*|0[xX][0-9a-fA-F]+[uUlL]*)"
    r"|(\w+)|(&&|\|\||==|!=|<=|>=|<<|>>|\+=|-=|\*=|/=|%=|\+\+|--|::|\"(?:[^\"\\]|\\.)*\"|[-+*/%<>=!?:;,(){}&|^~\[\]]))"
)


@dataclass
class HostFunction:
    name: str
    parameters: list[str]
    body: str
    # Declared C type of each parameter, for ``sizeof``.
    types: dict = None


def _block(text: str, start: int) -> str:
    depth = 1
    index = start
    while index < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        index += 1
    return text[start : index - 1]


def _preprocessed(body: str, defines: dict[str, int]) -> str | None:
    """``body`` with its ``#if`` arms decided, or None if one cannot be.

    A condition reads the file's integer ``#define``s; a name with none is 0,
    as the C preprocessor has it.
    """
    lines, taking, decided = [], [True], [True]
    for line in body.splitlines():
        directive = re.match(r"\s*#\s*(\w+)\s*(.*)", line)
        if directive is None:
            if all(taking):
                lines.append(line)
            continue
        word, rest = directive.groups()
        if word in ("if", "ifdef", "ifndef"):
            if word == "if":
                value = _directive_value(rest, defines)
            else:
                value = (
                    (rest.split()[0] in defines) == (word == "ifdef")
                    if rest.split()
                    else None
                )
            if value is None:
                return None
            taking.append(bool(value))
            decided.append(bool(value))
        elif word == "elif":
            value = _directive_value(rest, defines)
            if value is None or len(taking) < 2:
                return None
            taking[-1] = not decided[-1] and bool(value)
            decided[-1] = decided[-1] or bool(value)
        elif word == "else":
            if len(taking) < 2:
                return None
            taking[-1] = not decided[-1]
        elif word == "endif":
            if len(taking) < 2:
                return None
            taking.pop()
            decided.pop()
    return "\n".join(lines)


def _directive_value(condition: str, defines: dict[str, int]):
    """An ``#if`` condition's value, or None when it is not simple arithmetic."""
    import ast as python_ast

    text = re.sub(r"//.*|/\*.*?\*/", "", condition)
    text = re.sub(
        r"\bdefined\s*\(?\s*(\w+)\s*\)?",
        lambda m: "1" if m.group(1) in defines else "0",
        text,
    )
    text = text.replace("&&", " and ").replace("||", " or ")
    text = re.sub(r"!(?!=)", " not ", text)
    text = re.sub(
        r"\b[A-Za-z_]\w*\b",
        lambda m: (
            m.group(0)
            if m.group(0) in ("and", "or", "not")
            else str(defines.get(m.group(0), 0))
        ),
        text,
    )
    try:
        tree = python_ast.parse(text.strip() or "0", mode="eval")
    except SyntaxError:
        return None
    allowed = (
        python_ast.Expression,
        python_ast.BoolOp,
        python_ast.BinOp,
        python_ast.UnaryOp,
        python_ast.Compare,
        python_ast.Constant,
        python_ast.operator,
        python_ast.unaryop,
        python_ast.cmpop,
        python_ast.boolop,
    )
    if not all(isinstance(node, allowed) for node in python_ast.walk(tree)):
        return None
    try:
        return eval(  # noqa: S307 - integers only, checked above
            compile(tree, "<if>", "eval"), {"__builtins__": {}}
        )
    except Exception:  # noqa: BLE001 - not an integer condition
        return None


def host_functions(text: str) -> dict[str, HostFunction]:
    """Scalar-returning functions with scalar parameters defined in ``text``."""
    found = {}
    defines = {name: int(value) for name, value in _INTEGER_DEFINE.findall(text)}
    for match in _DEFINITION.finditer(text):
        _, name, params = match.groups()
        parameters = []
        types: dict[str, str] = {}
        ok = True
        params = re.sub(r"/\*.*?\*/|//[^\n]*", " ", params, flags=re.DOTALL)
        for position, param in enumerate(
            part.strip() for part in params.split(",") if part.strip()
        ):
            if param == "void":
                continue
            if re.fullmatch(_SCALAR_TYPES, param):
                # An unnamed parameter: passed, never read.
                param = f"{param} _unnamed{position}"
            if not re.fullmatch(rf"{_SCALAR_TYPES}\s+\w+", param):
                ok = False
                break
            words = param.split()
            parameters.append(words[-1])
            types[words[-1]] = words[-2].rpartition("::")[2]
        body = _preprocessed(_block(text, match.end()), defines) if ok else None
        if body is not None and name not in _PYTHON_SAME:
            found.setdefault(name, HostFunction(name, parameters, body, types))
    return found


class _Untranslatable(Exception):
    pass


def _tokens(body: str) -> list[str]:
    tokens = []
    position = 0
    body = re.sub(r"//[^\n]*|/\*.*?\*/", " ", body, flags=re.DOTALL).strip()
    while position < len(body):
        match = _TOKEN.match(body, position)
        if match is None or match.end() == position:
            raise _Untranslatable(body[position : position + 20])
        number, word, symbol = match.groups()
        if number:
            tokens.append(re.sub(r"[uUlLfF]+$", "", number))
        else:
            tokens.append(word or symbol)
        position = match.end()
    return tokens


class _Translator:
    """C statements and expressions to Python, one function body at a time."""

    _OPERATORS = {"&&": " and ", "||": " or ", "!": " not "}

    def __init__(self, tokens: list[str], types: dict[str, str] | None = None):
        self.tokens = tokens
        self.index = 0
        self.types = dict(types or {})

    def peek(self, offset=0):
        position = self.index + offset
        return self.tokens[position] if position < len(self.tokens) else None

    def take(self, expected=None):
        token = self.peek()
        if expected is not None and token != expected:
            raise _Untranslatable(f"expected {expected}, got {token}")
        self.index += 1
        return token

    def block(self, indent: int) -> list[str]:
        lines = []
        while self.peek() is not None and self.peek() != "}":
            lines.extend(self.statement(indent))
        return lines

    def body(self, indent: int) -> list[str]:
        if self.peek() == "{":
            self.take("{")
            lines = self.block(indent)
            self.take("}")
            return lines or ["    " * indent + "pass"]
        return self.statement(indent)

    def statement(self, indent: int) -> list[str]:
        pad = "    " * indent
        token = self.peek()
        if token == "return":
            self.take()
            value = self.expression(";")
            self.take(";")
            return [f"{pad}return {value}"]
        if token == "if":
            self.take()
            self.take("(")
            condition = self.expression(")")
            self.take(")")
            lines = [f"{pad}if {condition}:"] + self.body(indent + 1)
            if self.peek() == "else":
                self.take()
                if self.peek() == "if":
                    nested = self.statement(indent)
                    lines.append(f"{pad}el{nested[0].lstrip()}")
                    lines.extend(nested[1:])
                else:
                    lines.append(f"{pad}else:")
                    lines.extend(self.body(indent + 1))
            return lines
        if token == ";":
            self.take()
            return []
        if token == "throw":
            # The host code refuses: so does its translation.
            while self.peek() not in (";", None):
                self.take()
            self.take(";")
            return [f"{pad}raise RuntimeError('the host function throws')"]
        # Declaration: [qualifiers] type name [= expr] {, name [= expr]} ;
        words = []
        while (
            self.peek() is not None
            and re.fullmatch(r"\w+", self.peek())
            and self.peek(1) is not None
            and re.fullmatch(r"\w+|::", self.peek(1))
        ):
            words.append(self.take())
            while self.peek() == "::":
                self.take()
                words.append(self.take())
        lines = []
        while True:
            target = self.take()
            if not re.fullmatch(r"\w+", target or ""):
                raise _Untranslatable(f"statement at {target}")
            operator = self.peek()
            if operator == "=" and self.peek(1) == "[":
                # ``x = [&](){ ... }();``: a lambda called where it is made.
                self.take()
                lines.extend(self._lambda_call(target, indent))
            elif operator in ("=", "+=", "-=", "*=", "/=", "%="):
                self.take()
                value = self.expression(";,")
                if operator == "/=":
                    lines.append(f"{pad}{target} = _cdiv({target}, {value})")
                elif operator == "=":
                    lines.append(f"{pad}{target} = {value}")
                else:
                    lines.append(f"{pad}{target} {operator} {value}")
            elif operator in ("++", "--"):
                self.take()
                lines.append(f"{pad}{target} {'+=' if operator == '++' else '-='} 1")
            elif words:
                lines.append(f"{pad}{target} = 0")
            else:
                raise _Untranslatable(f"statement {target} {operator}")
            if self.peek() == ",":
                self.take()
                continue
            self.take(";")
            return lines

    def _lambda_call(self, target: str, indent: int) -> list[str]:
        pad = "    " * indent
        self.take("[")
        while self.peek() != "]":
            self.take()
        self.take("]")
        self.take("(")
        if self.peek() != ")":
            raise _Untranslatable("a lambda with parameters")
        self.take(")")
        self.take("{")
        body = self.block(indent + 1) or ["    " * (indent + 1) + "pass"]
        self.take("}")
        self.take("(")
        self.take(")")
        name = f"_lambda_{target}"
        return [f"{pad}def {name}():", *body, f"{pad}{target} = {name}()"]

    def expression(self, stop: str) -> str:
        """Tokens up to an unnested stop token, as a Python expression."""
        parts: list[str] = []
        depth = 0
        while self.peek() is not None:
            token = self.peek()
            if depth == 0 and token in stop:
                break
            if token == "(":
                depth += 1
            elif token == ")":
                depth -= 1
                if depth < 0:
                    break
            self.take()
            parts.append(token)
        return self._python(parts)

    def _python(self, parts: list[str]) -> str:
        parser = _Expression(parts, self.types)
        text = parser.ternary()
        if parser.position != len(parts):
            raise _Untranslatable(f"trailing {parts[parser.position:]}")
        return text

    @staticmethod
    def _top_level(parts, symbol, last=False):
        depth = 0
        found = None
        for index, token in enumerate(parts):
            if token == "(":
                depth += 1
            elif token == ")":
                depth -= 1
            elif token == symbol and depth == 0:
                found = index
                if not last:
                    return index
        return found


class _Expression:
    """A C expression, by precedence climbing, as parenthesized Python."""

    _LEVELS = (
        ("||",),
        ("&&",),
        ("|",),
        ("^",),
        ("&",),
        ("==", "!="),
        ("<", "<=", ">", ">="),
        ("<<", ">>"),
        ("+", "-"),
        ("*", "/", "%"),
    )
    _PYTHON = {"||": "or", "&&": "and"}
    _CAST_TYPES = (
        "int",
        "unsigned",
        "long",
        "short",
        "char",
        "float",
        "double",
        "bool",
    )

    def __init__(self, parts: list[str], types: dict[str, str]):
        self.parts = parts
        self.position = 0
        self.types = types

    def peek(self):
        return self.parts[self.position] if self.position < len(self.parts) else None

    def take(self, expected=None):
        token = self.peek()
        if expected is not None and token != expected:
            raise _Untranslatable(f"expected {expected}, got {token}")
        self.position += 1
        return token

    def ternary(self) -> str:
        condition = self.binary(0)
        if self.peek() == "?":
            self.take()
            when = self.ternary()
            self.take(":")
            otherwise = self.ternary()
            return f"(({when}) if ({condition}) else ({otherwise}))"
        return condition

    def binary(self, level: int) -> str:
        if level == len(self._LEVELS):
            return self.unary()
        left = self.binary(level + 1)
        while self.peek() in self._LEVELS[level]:
            operator = self.take()
            right = self.binary(level + 1)
            if operator == "/":
                left = f"_cdiv({left}, {right})"
            elif operator == "%":
                left = f"_cmod({left}, {right})"
            else:
                left = f"({left} {self._PYTHON.get(operator, operator)} {right})"
        return left

    def unary(self) -> str:
        token = self.peek()
        if token in ("-", "+", "~"):
            self.take()
            return f"({token}{self.unary()})"
        if token == "!":
            self.take()
            return f"(not {self.unary()})"
        if token == "(" and self._is_cast():
            self.take("(")
            kind = []
            while self.peek() != ")":
                kind.append(self.take())
            self.take(")")
            return self._cast(kind, self.unary())
        return self.postfix()

    def _is_cast(self) -> bool:
        index = self.position + 1
        while index < len(self.parts) and self.parts[index] != ")":
            if self.parts[index] not in (*self._CAST_TYPES, "const", "signed"):
                return False
            index += 1
        return index > self.position + 1

    def _cast(self, kind: list[str], value: str) -> str:
        if any(word in ("float", "double") for word in kind):
            return f"float({value})"
        if "bool" in kind:
            return f"bool({value})"
        return f"int({value})"

    def postfix(self) -> str:
        token = self.take()
        if token is None:
            raise _Untranslatable("empty expression")
        if token == "(":
            inner = self.ternary()
            self.take(")")
            return f"({inner})"
        if token == "sizeof":
            self.take("(")
            words = []
            while self.peek() != ")":
                words.append(self.take())
            self.take(")")
            declared = self.types.get(words[-1], words[-1]) if words else "int"
            return str(_SIZES.get(declared, 4))
        if token in ("true", "false"):
            return token.capitalize()
        if token in ("static_cast", "reinterpret_cast"):
            self.take("<")
            kind = []
            while self.peek() != ">":
                kind.append(self.take())
            self.take(">")
            self.take("(")
            value = self.ternary()
            self.take(")")
            return self._cast(kind, value)
        name = token
        while self.peek() == "::":
            self.take()
            name = self.take()  # std::min -> min
        if not re.fullmatch(r"\w+(\.\d*)?([eE][-+]?\d+)?", name):
            raise _Untranslatable(f"token {name}")
        if (
            name == "get"
            and self.peek() == "<"
            and self.position + 3 < len(self.parts)
            and self.parts[self.position + 1].isdigit()
            and self.parts[self.position + 2] == ">"
            and self.parts[self.position + 3] == "("
        ):
            # ``get<N>(tuple)``: one element of a tuple.
            self.take("<")
            element = self.take()
            self.take(">")
            self.take("(")
            value = self.ternary()
            self.take(")")
            return f"({value})[{element}]"
        if self.peek() == "(":
            self.take()
            arguments = []
            while self.peek() != ")":
                arguments.append(self.ternary())
                if self.peek() == ",":
                    self.take()
            self.take(")")
            return f"{name}({', '.join(arguments)})"
        return name


def translate(function: HostFunction) -> str | None:
    """Python source for ``function``, or None outside the supported subset."""
    try:
        translator = _Translator(_tokens(function.body), function.types)
        lines = translator.block(1)
        if translator.peek() is not None:
            return None
    except (_Untranslatable, IndexError):
        return None
    header = f"def {function.name}({', '.join(function.parameters)}):"
    return "\n".join([header, *(lines or ["    return None"])]) + "\n"


class HostEvaluator:
    """Evaluates the library's host functions on scalar arguments.

    Each function and the host functions it calls are translated once and
    evaluated by ``interpreter`` (a :class:`SourceInterpreter`), so no
    library code is compiled or executed by Python itself.
    """

    def __init__(self, functions: dict[str, HostFunction], interpreter):
        self.functions = functions
        self.interpreter = interpreter
        self.namespace: dict = {}
        self._loaded: set[str] = set()
        self.interpreter._exec_builtin(PRELUDE, self.namespace)

    def _load(self, name: str) -> bool:
        if name in self._loaded:
            return True
        function = self.functions.get(name)
        if function is None:
            return False
        source = translate(function)
        if source is None:
            return False
        self._loaded.add(name)
        for called in set(
            re.findall(r"^\s*def \w+|\b(\w+)\(", source, re.MULTILINE)
        ) - {"", name}:
            if called in self.functions:
                if self._load(called):
                    continue
            elif (
                called in self.namespace
                or hasattr(builtins, called)
                or called.startswith("_lambda_")
            ):
                continue
            # A call it cannot make: its value stays unknown too.
            self._loaded.discard(name)
            return False
        self.interpreter._exec_builtin(source, self.namespace)
        return True

    def __call__(self, name: str, args: tuple):
        if not self._load(name):
            raise LookupError(name)
        # A C call may pass more arguments than the function declares (a
        # stream appended by the caller); the callee never sees them.
        declared = len(self.functions[name].parameters)
        return self.namespace[name](*args[:declared])


PRELUDE = '''
import math

CHAR_BIT = 8
INT_MAX = 2147483647
INT_MIN = -2147483648


def _cdiv(a, b):
    """C division: integers truncate toward zero."""
    if isinstance(a, int) and isinstance(b, int):
        quotient = abs(a) // abs(b)
        return quotient if (a >= 0) == (b >= 0) else -quotient
    return a / b


def _cmod(a, b):
    """C remainder: its sign follows the dividend."""
    if isinstance(a, int) and isinstance(b, int):
        return a - b * _cdiv(a, b)
    return math.fmod(a, b)


def __builtin_clz(x):
    return 32 - (x & 0xFFFFFFFF).bit_length()


def __builtin_clzll(x):
    return 64 - (x & 0xFFFFFFFFFFFFFFFF).bit_length()


__builtin_clzl = __builtin_clzll


def __builtin_ctz(x):
    return (x & -x).bit_length() - 1


__builtin_ctzll = __builtin_ctzl = __builtin_ctz


def __builtin_popcount(x):
    return bin(x & 0xFFFFFFFF).count("1")


def ceil(x):
    return math.ceil(x)


def floor(x):
    return math.floor(x)


def sqrt(x):
    return math.sqrt(x)


def log2(x):
    return math.log2(x)


def pow(x, y):
    return x**y


def div_ceil(a, b):
    return -(-a // b)


def make_tuple(*values):
    return values
'''

# Bytes of each C scalar type, for ``sizeof`` (LP64, as the libraries build).
_SIZES = {
    "char": 1,
    "bool": 1,
    "short": 2,
    "int": 4,
    "unsigned": 4,
    "float": 4,
    "long": 8,
    "double": 8,
    "size_t": 8,
    "int8_t": 1,
    "uint8_t": 1,
    "int16_t": 2,
    "uint16_t": 2,
    "int32_t": 4,
    "uint32_t": 4,
    "int64_t": 8,
    "uint64_t": 8,
    "index_t": 4,
}


# ------------------------------------------------------------ tensor results

# A host function that returns a tensor it allocates: ``ffi::Tensor f(...)``.
# A function definition, whatever it returns (a tensor, or a struct or tuple of them).
_TENSOR_DEFINITION = re.compile(
    r"(?:^|[\s;}])(?:inline\s+|static\s+)*[\w:]+(?:<[^<>{};()]*>)?\s+(\w+)\s*\(([^()]*)\)\s*\{"
)
_STATIC_SIZE = re.compile(
    r"static_assert\s*\(\s*sizeof\s*\(\s*([\w:]+)\s*\)\s*==\s*(\d+)\s*\)"
)
_STRUCT = re.compile(r"\bstruct\s+(?:alignas\s*\(\s*(\d+)\s*\)\s*)?(\w+)\s*\{")
_ALIAS = re.compile(r"\busing\s+(\w+)\s*=\s*([\w:]+)\s*;")
_DLPACK_DTYPES = {
    "kDLUInt8": "uint8",
    "kDLInt8": "int8",
    "kDLInt16": "int16",
    "kDLInt32": "int32",
    "kDLInt64": "int64",
    "kDLUInt16": "uint16",
    "kDLUInt32": "uint32",
    "kDLUInt64": "uint64",
    "kDLFloat16": "float16",
    "kDLFloat32": "float32",
    "kDLFloat64": "float64",
    "kDLBfloat16": "bfloat16",
    "kDLBool": "bool",
}


@dataclass
class TensorHostFunction:
    """A host function returning a tensor it allocates."""

    name: str
    # ``(declared type, name)`` of each parameter, in order.
    parameters: list[tuple[str, str]]
    body: str


@functools.lru_cache(maxsize=4096)
def tensor_host_functions(text: str) -> dict[str, TensorHostFunction]:
    """Tensor-returning host functions defined in ``text``."""
    found = {}
    for match in _TENSOR_DEFINITION.finditer(text):
        name, params = match.groups()
        parameters = []
        for param in [part.strip() for part in params.split(",") if part.strip()]:
            words = re.sub(r"/\*.*?\*/|//.*", "", param).replace("&", " ").split()
            if words:
                parameters.append((" ".join(words[:-1]), words[-1]))
        found.setdefault(
            name, TensorHostFunction(name, parameters, _block(text, match.end()))
        )
    return found


def struct_sizes(texts) -> dict[str, int]:
    """``sizeof`` of the structs ``texts`` define, as their C layout gives it.

    A ``static_assert(sizeof(S) == N)`` states it outright; otherwise fields of
    scalar, pointer and array type are laid out with natural alignment, and
    ``alignas`` raises the struct's alignment. ``using A = B`` aliases follow.
    """
    stated: dict[str, int] = {}
    layouts: dict[str, tuple[int, int]] = {}
    aliases: dict[str, str] = {}
    for text in texts:
        for name, size in _STATIC_SIZE.findall(text):
            stated.setdefault(name.split("::")[-1], int(size))
        for alias, target in _ALIAS.findall(text):
            aliases.setdefault(alias, target.split("::")[-1])
        for match in _STRUCT.finditer(text):
            align, name = match.groups()
            layout = _struct_layout(_block(text, match.end()), int(align or 1), layouts)
            if layout is not None:
                layouts.setdefault(name, layout)
    sizes = {name: size for name, (size, _) in layouts.items()}
    sizes.update(stated)
    for alias, target in aliases.items():
        if target in sizes:
            sizes.setdefault(alias, sizes[target])
    return sizes


def _struct_layout(body: str, align: int, known: dict) -> tuple[int, int] | None:
    """``(size, alignment)`` of a struct body of plain fields, or None."""
    offset = 0
    for field in [
        part.strip() for part in re.sub(r"//.*|/\*.*?\*/", "", body).split(";")
    ]:
        if (
            not field
            or "(" in field
            or field.startswith(("static", "using", "template", "public", "private"))
        ):
            continue
        array = re.search(r"\[(\d+)\]\s*$", field)
        count = int(array.group(1)) if array else 1
        words = re.sub(r"\[\d+\]\s*$", "", field).split()
        if len(words) < 2:
            return None
        kind = " ".join(words[:-1]).replace("const ", "")
        if "*" in field:
            size = field_align = 8
        elif kind.split("::")[-1] in _SIZES:
            size = field_align = _SIZES[kind.split("::")[-1]]
        elif kind.split("::")[-1] in known:
            size, field_align = known[kind.split("::")[-1]]
        else:
            return None
        offset = -(-offset // field_align) * field_align + size * count
        align = max(align, field_align)
    return -(-offset // align) * align, align


class TensorResults:
    """The tensor a host function returns, from its body: shape and dtype.

    Sizes come from what the body binds of its tensor arguments
    (``TensorMatcher({B, -1}).verify(t)`` binds ``B`` to ``t``'s first size,
    ``t.size(1)`` reads one) and from local scalars; the returned tensor is
    the one an ``empty({dims}, dtype, device)`` made. Expressions are
    translated as host functions are and evaluated by the interpreter.
    """

    def __init__(self, interpreter, struct_sizes: dict[str, int]):
        self.interpreter = interpreter
        self.sizes = struct_sizes
        self.namespace: dict = {}
        interpreter._exec_builtin(PRELUDE, self.namespace)

    def output(self, function: TensorHostFunction, args: tuple):
        """``(shape, dtype name)`` of the returned tensor, a list of them for a
        returned tuple or struct of tensors (``Plan{std::move(c), std::move(w)}``),
        or None. Of several returns (one per path), the first whose tensors are
        all known is taken."""
        bound = {name: value for (_, name), value in zip(function.parameters, args)}
        scope = dict(self.namespace)
        statements = _flat_statements(function.body)
        for statement in statements:
            matcher = re.match(r"TensorMatcher\s*\(\s*\{([^}]*)\}\s*\)", statement)
            verified = re.search(r"\.verify\s*\(\s*(\w+)\s*\)", statement)
            if matcher and verified and hasattr(bound.get(verified.group(1)), "shape"):
                shape = bound[verified.group(1)].shape
                for index, element in enumerate(
                    part.strip() for part in matcher.group(1).split(",")
                ):
                    if (
                        re.fullmatch(r"\w+", element)
                        and not element.lstrip("-").isdigit()
                        and index < len(shape)
                    ):
                        scope[f"__sym_{element}"] = int(shape[index])
        for name, value in bound.items():
            if not hasattr(value, "shape"):
                scope[name] = getattr(value, "value", value)
        allocations: dict[str, tuple[str, str]] = {}
        for statement in statements:
            found = re.match(
                r"(?:const\s+)?(?:auto|[\w:]+)\s+(\w+)\s*=\s*(?:[\w:]+::)?empty\s*\(\s*\{(.*)\}\s*,\s*([\w:]+)",
                statement,
                re.S,
            )
            if found:
                allocations[found.group(1)] = (
                    found.group(2),
                    found.group(3).split("::")[-1],
                )
                continue
            local = re.match(
                r"(?:const\s+|constexpr\s+|static\s+)*[\w:<>]+\s+(\w+)\s*=\s*(.+)$",
                statement,
                re.S,
            )
            if local and "{" not in local.group(2):
                value = self._evaluate(local.group(2), bound, scope)
                if value is not None:
                    scope[local.group(1)] = value
            returned = re.match(r"return\s+(.+)$", statement, re.S)
            if returned is None:
                continue
            packed = re.fullmatch(
                r"(?:[\w:]+(?:<[^<>]*>)?)?\s*\{(.*)\}", returned.group(1).strip(), re.S
            )
            items = (
                _split_top(packed.group(1)) if packed else [returned.group(1).strip()]
            )
            names = [
                re.sub(r"^std::move\s*\(\s*(\w+)\s*\)$", r"\1", item.strip())
                for item in items
            ]
            if not names or any(name not in allocations for name in names):
                continue
            outputs = []
            for name in names:
                dims, dtype = allocations[name]
                values = [self._evaluate(dim, bound, scope) for dim in _split_top(dims)]
                if (
                    any(value is None for value in values)
                    or dtype not in _DLPACK_DTYPES
                ):
                    return None
                outputs.append(
                    (tuple(int(value) for value in values), _DLPACK_DTYPES[dtype])
                )
            return outputs if packed else outputs[0]
        return None

    def _evaluate(self, expression: str, bound: dict, scope: dict):
        text = re.sub(r"(\w+)\.unwrap\s*\(\s*\)", r"__sym_\1", expression)
        text = re.sub(
            r"(\w+)\.size\s*\(\s*(-?\d+)\s*\)",
            lambda m: (
                str(int(bound[m.group(1)].shape[int(m.group(2))]))
                if hasattr(bound.get(m.group(1)), "shape")
                else m.group(0)
            ),
            text,
        )
        text = re.sub(
            r"sizeof\s*\(\s*([\w:]+)\s*\)",
            lambda m: str(
                self.sizes.get(m.group(1).split("::")[-1], _SIZES.get(m.group(1), -1))
            ),
            text,
        )
        try:
            python = _Translator(_tokens(text)).expression(";")
            return self.interpreter._eval_builtin(python, dict(scope))
        except Exception:  # noqa: BLE001 - outside the subset: unknown
            return None


def _statements(body: str) -> list[str]:
    """Top-level statements of a function body (nested blocks are skipped)."""
    statements, current, depth = [], [], 0
    for char in body:
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
        if char == ";" and depth == 0:
            statements.append(re.sub(r"\s+", " ", "".join(current)).strip())
            current = []
        else:
            current.append(char)
    return [statement for statement in statements if statement]


def _flat_statements(body: str) -> list[str]:
    """Every statement of a function body, those in nested blocks too, in order.

    A block's brace (after ``if (...)``, ``else``, ``;``) opens statements of
    its own; an initializer's (``Plan{a, b}``, ``empty({n, m}, ...)``) is part
    of its statement.
    """
    statements, current, parens, braces = [], [], 0, []

    def flush():
        text = re.sub(r"\s+", " ", "".join(current)).strip()
        if text:
            statements.append(text)
        current.clear()

    for char in body:
        if char in "([":
            parens += 1
        elif char in ")]":
            parens -= 1
        if parens == 0 and char == "{":
            before = "".join(current).rstrip()
            block = not before or before.endswith((")", "else", "do", "try"))
            braces.append(block)
            if block:
                # The header (``if (...)``) is no statement of its own.
                current.clear()
                continue
        elif parens == 0 and char == "}" and braces:
            if braces.pop():
                flush()
                continue
        if char == ";" and parens == 0 and not any(not block for block in braces):
            flush()
        else:
            current.append(char)
    flush()
    return statements


def _split_top(text: str) -> list[str]:
    parts, current, depth = [], [], 0
    for char in text:
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if "".join(current).strip():
        parts.append("".join(current).strip())
    return parts
