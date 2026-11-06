###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Find the tensor a size read in a ``torch.export`` graph was taken from.

``torch.export`` turns every ``.shape`` read into a symbol shared by all
tensors of that size and reads it back off whichever model input carries it,
so ``arange(N)`` after ``B, N = W.shape`` is recorded as reading some other
input, and a read of an intermediate tensor (``x.shape`` inside a layer) as
reading an input. The source still says which tensor was read: this module
follows the consumer's own source line back through its size names
(``N`` <- ``B, N = W.shape``, ``maxpos`` <- ``max(N, 4096)``, a parameter
<- its caller's argument) to a tensor variable, then to the graph node that
variable held at that point: the last node emitted at one of its assignments,
in the same module call, before the consumer. Graph order is execution order,
so a loop resolves to the current iteration's value.

Torch-free: it reads stack-trace text, ``nn_module_stack`` call keys and the
model's source files, so it is unit-tested without torch.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

# Constants
_FRAME_RE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<name>\S+)')
# Tensor methods whose result is a size of the tensor they are called on.
_SIZE_METHODS = frozenset({"size", "numel"})
# Calls that only combine sizes; anything else ends a size expression.
_SIZE_BUILTINS = frozenset({"max", "min", "int", "abs", "round"})
_SIZE_MODULES = frozenset({"math", "operator"})


@dataclass(frozen=True)
class Frame:
    """One source frame of a node's stack trace."""

    file: str
    line: int
    name: str


@dataclass(frozen=True)
class TracedNode:
    """One graph node: its user-code frames (outermost first) and module-call keys."""

    name: str
    frames: tuple[Frame, ...]
    module_calls: tuple[str, ...] = ()


def parse_stack_trace(
    text: str | None, files: set[str] | None = None
) -> tuple[Frame, ...]:
    """The model's frames in a ``node.meta["stack_trace"]`` string, from its root ``forward`` in.

    Only frames in ``files`` are kept when given; frames before the first
    ``forward`` belong to whatever called the model, not to the model.
    """
    frames = [
        Frame(match["file"], int(match["line"]), match["name"])
        for match in _FRAME_RE.finditer(text or "")
        if files is None or match["file"] in files
    ]
    root = next((i for i, frame in enumerate(frames) if frame.name == "forward"), None)
    return () if root is None else tuple(frames[root:])


class SizeReadResolver:
    """Resolve the tensors whose sizes a graph node reads, from the model's source."""

    def __init__(self, nodes: list[TracedNode], root_inputs: dict[str, str]):
        """``nodes`` in execution order; ``root_inputs`` maps a top-level
        ``forward`` parameter name to its input node."""
        self._nodes = nodes
        self._index = {node.name: i for i, node in enumerate(nodes)}
        self._root_inputs = root_inputs

    def sources_of(self, consumer: str) -> list[str]:
        """Graph nodes whose sizes ``consumer``'s source line reads; empty if unresolved."""
        index = self._index[consumer]
        node = self._nodes[index]
        if not node.frames:
            return []
        depth = len(node.frames) - 1
        function = _function_at(node.frames[depth])
        if function is None:
            return []
        statement = function.statement_at(node.frames[depth].line)
        if statement is None:
            return []
        variables: set[tuple[int, str]] = set()
        for expr in _statement_exprs(statement):
            for sub in ast.walk(expr):
                variables |= self._size_reads(
                    sub, node.frames, depth, statement.lineno, set(), direct=True
                )
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    variables |= self._resolve_name(
                        sub.id, node.frames, depth, statement.lineno, set()
                    )
        found: list[str] = []
        for var_depth, var in sorted(variables):
            source = self._value_of(var, var_depth, node, index)
            if source is not None and source not in found:
                found.append(source)
        return found

    def _size_reads(self, expr, frames, depth, line, seen, *, direct=False):
        """``(depth, tensor variable)`` pairs a size expression on ``line`` reads."""
        if isinstance(expr, ast.Attribute) and expr.attr == "shape":
            var = _tensor_variable(expr.value)
            return {(depth, var)} if var else set()
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute)
            and expr.func.attr in _SIZE_METHODS
        ):
            var = _tensor_variable(expr.func.value)
            return {(depth, var)} if var else set()
        if direct:
            return set()  # the consumer's line is walked node by node by the caller
        if isinstance(expr, ast.Name):
            return self._resolve_name(expr.id, frames, depth, line, seen)
        if isinstance(expr, ast.Subscript):
            return self._size_reads(expr.value, frames, depth, line, seen)
        if isinstance(expr, ast.Call) and _is_size_combinator(expr.func):
            return self._union(expr.args, frames, depth, line, seen)
        if isinstance(expr, ast.BinOp):
            return self._union([expr.left, expr.right], frames, depth, line, seen)
        if isinstance(expr, ast.UnaryOp):
            return self._size_reads(expr.operand, frames, depth, line, seen)
        if isinstance(expr, ast.IfExp):
            return self._union([expr.body, expr.orelse], frames, depth, line, seen)
        return set()

    def _union(self, exprs, frames, depth, line, seen):
        out: set[tuple[int, str]] = set()
        for expr in exprs:
            out |= self._size_reads(expr, frames, depth, line, seen)
        return out

    def _resolve_name(self, name, frames, depth, before_line, seen):
        """Size reads behind ``name`` as bound in frame ``depth`` before ``before_line``."""
        key = (frames[: depth + 1], name)
        if key in seen:
            return set()
        seen.add(key)
        function = _function_at(frames[depth])
        if function is None:
            return set()
        binding = function.binding_before(name, before_line)
        if binding is not None:
            line, target, value = binding
            if isinstance(target, ast.Name):
                return self._size_reads(value, frames, depth, line, seen)
            position = _tuple_position(target, name)
            if position is None:
                return set()
            if isinstance(value, (ast.Tuple, ast.List)):
                return self._size_reads(value.elts[position], frames, depth, line, seen)
            return self._size_reads(value, frames, depth, line, seen, direct=True)
        if depth == 0 or name not in function.params:
            return set()
        argument = _caller_argument(frames, depth, name)
        if argument is None:
            return set()
        return self._size_reads(
            argument, frames, depth - 1, frames[depth - 1].line, seen
        )

    def _value_of(self, var, depth, consumer, consumer_index):
        """The graph node ``var`` held in the consumer's frame ``depth``."""
        frames = consumer.frames
        function = _function_at(frames[depth])
        if function is None:
            return None
        spans = function.assignment_spans(var)
        calls = consumer.module_calls[: _module_depth(frames, depth)]
        # Skip the statement the consumer is part of: in ``x = layer(x)`` the
        # call's own nodes sit on an assignment line of ``x`` but come after
        # the value it was passed.
        start = consumer_index
        current = consumer.module_calls[: len(calls) + 1]
        while start > 0:
            node = self._nodes[start - 1]
            if (
                node.frames[: depth + 1] != frames[: depth + 1]
                or node.module_calls[: len(current)] != current
            ):
                break
            start -= 1
        if spans:
            for node in reversed(self._nodes[:start]):
                if len(node.frames) <= depth or node.frames[:depth] != frames[:depth]:
                    continue
                frame = node.frames[depth]
                if (frame.file, frame.name) != (frames[depth].file, frames[depth].name):
                    continue
                if node.module_calls[: len(calls)] != calls:
                    continue
                if any(first <= frame.line <= last for first, last in spans):
                    return node.name
        if var not in function.params:
            return None
        if depth == 0:
            return self._root_inputs.get(var)
        argument = _caller_argument(frames, depth, var)
        if isinstance(argument, ast.Name):
            return self._value_of(argument.id, depth - 1, consumer, consumer_index)
        return None


@dataclass
class _Function:
    node: ast.FunctionDef | ast.AsyncFunctionDef
    params: list[str]

    def statement_at(self, line: int) -> ast.stmt | None:
        """The innermost simple statement spanning ``line`` (a compound one's header)."""
        best = None
        for stmt in _own_statements(self.node):
            spans_line = stmt.lineno <= line <= (stmt.end_lineno or stmt.lineno)
            if spans_line and (best is None or stmt.lineno >= best.lineno):
                best = stmt
        return best

    def binding_before(self, name: str, line: int):
        """``(line, target, value)`` of the last plain assignment to ``name`` before ``line``."""
        best = None
        for stmt in _own_statements(self.node):
            if stmt.lineno >= line or not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                continue
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                later = best is None or stmt.lineno >= best[0]
                if _binds(target, name) and stmt.value is not None and later:
                    best = (stmt.lineno, target, stmt.value)
        return best

    def assignment_spans(self, name: str) -> list[tuple[int, int]]:
        spans = []
        for stmt in _own_statements(self.node):
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                )
                if any(_writes(target, name) for target in targets):
                    spans.append((stmt.lineno, stmt.end_lineno or stmt.lineno))
        return spans


@cache
def _parsed(file: str) -> ast.Module | None:
    try:
        return ast.parse(Path(file).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


@cache
def _function_at(frame: Frame) -> _Function | None:
    """The innermost function named ``frame.name`` spanning ``frame.line``."""
    tree = _parsed(frame.file)
    if tree is None:
        return None
    best = None
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
            if (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == frame.name
                and child.lineno <= frame.line <= (child.end_lineno or child.lineno)
                and (best is None or child.lineno >= best.lineno)
            ):
                best = child
    if best is None:
        return None
    params = [arg.arg for arg in best.args.posonlyargs + best.args.args]
    params += [arg.arg for arg in best.args.kwonlyargs]
    is_method = isinstance(parents.get(best), ast.ClassDef) and not any(
        isinstance(dec, ast.Name) and dec.id in ("staticmethod", "classmethod")
        for dec in best.decorator_list
    )
    if is_method and params:
        params = params[1:]
    return _Function(best, params)


def _own_statements(function: ast.AST):
    """Statements of ``function``, not of functions or classes nested in it."""
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        if isinstance(node, ast.stmt):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _statement_exprs(statement: ast.stmt) -> list[ast.expr]:
    """The expressions a statement itself evaluates (a compound one's header only)."""
    if isinstance(statement, (ast.If, ast.While)):
        return [statement.test]
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        return [statement.iter]
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return [item.context_expr for item in statement.items]
    return [
        child
        for child in ast.iter_child_nodes(statement)
        if isinstance(child, ast.expr)
    ]


def _caller_argument(
    frames: tuple[Frame, ...], depth: int, param: str
) -> ast.expr | None:
    """The argument the caller (frame ``depth - 1``) passed for ``param``."""
    caller = _function_at(frames[depth - 1])
    callee = _function_at(frames[depth])
    if caller is None or callee is None:
        return None
    statement = caller.statement_at(frames[depth - 1].line)
    if statement is None:
        return None
    calls = [
        node
        for expr in _statement_exprs(statement)
        for node in ast.walk(expr)
        if isinstance(node, ast.Call)
    ]
    named = [call for call in calls if _callee_name(call) == frames[depth].name]
    # A module is called as ``layer(x)``, which runs its ``forward``.
    if not named and frames[depth].name == "forward":
        named = [call for call in calls if not _is_size_combinator(call.func)]
    if len(named) != 1:
        return None
    call = named[0]
    for keyword in call.keywords:
        if keyword.arg == param:
            return keyword.value
    position = callee.params.index(param)
    if position < len(call.args) and not any(
        isinstance(arg, ast.Starred) for arg in call.args[: position + 1]
    ):
        return call.args[position]
    return None


def _callee_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_size_combinator(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id in _SIZE_BUILTINS
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id in _SIZE_MODULES
    )


def _tensor_variable(expr: ast.expr) -> str | None:
    """The variable a size is read from: ``x`` for ``x.shape`` and ``x[:, a:b].shape``."""
    while isinstance(expr, ast.Subscript):
        expr = expr.value
    return expr.id if isinstance(expr, ast.Name) else None


def _binds(target: ast.expr, name: str) -> bool:
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_binds(element, name) for element in target.elts)
    return False


def _writes(target: ast.expr, name: str) -> bool:
    """Whether assigning to ``target`` changes ``name``'s value, in place included."""
    while isinstance(target, (ast.Subscript, ast.Attribute)):
        target = target.value
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_writes(element, name) for element in target.elts)
    return isinstance(target, ast.Name) and target.id == name


def _tuple_position(target: ast.expr, name: str) -> int | None:
    if isinstance(target, (ast.Tuple, ast.List)):
        for position, element in enumerate(target.elts):
            if isinstance(element, ast.Name) and element.id == name:
                return position
    return None


def _module_depth(frames: tuple[Frame, ...], depth: int) -> int:
    """How many module calls frame ``depth`` sits inside (one per ``forward`` frame)."""
    return sum(1 for frame in frames[: depth + 1] if frame.name == "forward")
