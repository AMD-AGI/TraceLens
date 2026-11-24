###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Settings a traced run used that the trace does not state.

A setting is an environment variable (``NAME``) or a command-line option
(``--option``). Two sources say what they were:

* The trace. A profile records the engine's Python frames: which functions
  ran. A function the trace shows but a run from source never reached was
  called behind a condition the run decided the other way, at a call site
  the run did reach. Where that condition tests a setting
  (``envs.NAME.get() == "value"``, ``os.environ.get("NAME") == "value"``, a
  switch, an option's destination read off the parsed arguments, or a local,
  attribute or helper holding one of those), the trace says what the setting
  was (:func:`inferred_settings`).
* The engine's own launch recipes for the model (its tests, CI scripts and
  docs) give the values a setting is run with (:func:`recipe_values`): what
  to try when the kernels a run launched are not the trace's.
"""

from __future__ import annotations

import ast
import functools
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# How deep a condition is followed through locals, attributes and helpers.
_DEPTH = 4
# A function called from more places than this has too common a name to
# say which call ran.
_MOST_CALL_SITES = 4
# Files a launch recipe is written in.
_RECIPE_SUFFIXES = (".py", ".sh", ".yaml", ".yml", ".md", ".mdx", ".jsx", ".toml")
# A model name is shortened, a ``-`` part at a time, to no fewer parts than
# this when no recipe names it in full (``Org-V4-Pro-0813`` -> ``Org-V4-Pro``).
_FEWEST_NAME_PARTS = 2
# ``"NAME": "value"`` in a mapping, and ``NAME=value`` on a command line
# (``-e NAME=value``, ``export NAME=value``, ``NAME=value command``).
_MAPPED = re.compile(r"""["']([A-Z][A-Z0-9_]{2,})["']\s*:\s*["']([^"'$]*)["']""")
# ``--option value`` and ``--switch`` on a command line or in a list of
# arguments (``"--option", "value"``).
_OPTION = re.compile(
    r"""(?<![\w-])(--[a-z][a-z0-9-]+)(?:["'`]?,?\s+["'`]?([^\s"'`$\\,\])-][^\s"'`$\\,\]]*))?"""
)
_ASSIGNED = re.compile(
    r"""(?:^|\s)([A-Z][A-Z0-9_]{2,})=([^\s"'$;`()]+|"[^"$]*"|'[^']*')""", re.M
)


# How a switch's value is spelled.
_SWITCH_VALUES = {"0", "1", "true", "false", "True", "False", "on", "off", "yes", "no"}


def _read_as_switch(node: ast.Call) -> bool:
    """Whether an environment read reads a switch: a helper for one, or an on/off default."""
    if "bool" in ast.unparse(node.func).lower():
        return True
    default = node.args[1] if len(node.args) > 1 else None
    return isinstance(default, ast.Constant) and (
        isinstance(default.value, bool) or str(default.value) in _SWITCH_VALUES
    )


def _env_reference(node: ast.AST) -> str | None:
    """The variable a read names: ``envs.NAME.get()``, ``os.environ.get("NAME")``, ..."""
    if not isinstance(node, ast.Call):
        return None
    function = node.func
    if (
        isinstance(function, ast.Attribute)
        and function.attr in ("get", "value")
        and isinstance(function.value, ast.Attribute)
        and isinstance(function.value.value, ast.Name)
        and function.value.value.id == "envs"
    ):
        return function.value.attr
    name = ast.unparse(function)
    if name.endswith(("environ.get", "getenv")) or "env_var" in name:
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


def _compared(node: ast.AST) -> str | None:
    """A variable's name, or an attribute's (an option's destination)."""
    return _env_reference(node) or (
        node.attr if isinstance(node, ast.Attribute) else None
    )


def _called_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Subscript):
        # ``kernel[grid](...)``: a kernel launched over a grid.
        function = function.value
    return (
        function.attr
        if isinstance(function, ast.Attribute)
        else getattr(function, "id", None)
    )


def _negated(test: ast.expr) -> ast.expr:
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return test.operand
    return ast.UnaryOp(ast.Not(), test)


def _exits(statements) -> bool:
    return bool(statements) and isinstance(
        statements[-1], ast.Return | ast.Raise | ast.Continue | ast.Break
    )


@dataclass(frozen=True)
class _Site:
    """A call: the conditions it needs, the locals around it, where it is."""

    conditions: tuple[ast.expr, ...]
    # Single assignments of each enclosing function, innermost first.
    scopes: tuple[dict, ...]
    # The enclosing functions' names, innermost first.
    functions: tuple[str, ...]


class _Source:
    """The engine's Python source: its functions, and where each is called."""

    def __init__(self, roots: tuple[str, ...]):
        # function name -> [single return expression of a definition]
        self.returns: dict[str, list[ast.expr]] = {}
        # attribute name -> [expressions assigned to it (``x.name = ...``)]
        self.attributes: dict[str, list[ast.expr]] = {}
        # called name -> its call sites
        self.guards: dict[str, list[_Site]] = {}
        # variable -> the values the source compares it with
        self.literals: dict[str, set[str]] = {}
        # variables the source reads as a number
        self.numbers: set[str] = set()
        # variables the source reads as a switch (on or off)
        self.switches: set[str] = set()
        for path in sorted(path for root in roots for path in Path(root).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(errors="ignore"))
            except (SyntaxError, ValueError):
                continue
            self._read(tree)

    def _read(self, tree: ast.Module) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                returns = [
                    item
                    for item in ast.walk(node)
                    if isinstance(item, ast.Return) and item.value is not None
                ]
                if len(returns) == 1:
                    self.returns.setdefault(node.name, []).append(returns[0].value)
            elif (
                isinstance(node, ast.Call)
                and _called_name(node) in ("int", "float")
                and node.args
                and _env_reference(node.args[0])
            ):
                self.numbers.add(_env_reference(node.args[0]))
            elif (
                isinstance(node, ast.Call)
                and _env_reference(node)
                and _read_as_switch(node)
            ):
                self.switches.add(_env_reference(node))
            elif isinstance(node, ast.Compare) and len(node.ops) == 1:
                left, right = node.left, node.comparators[0]
                variable = _compared(left) or _compared(right)
                constant = right if _compared(left) else left
                values = (
                    constant.elts
                    if isinstance(constant, ast.Tuple | ast.List | ast.Set)
                    else [constant]
                )
                for value in values:
                    if (
                        variable
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str | int)
                    ):
                        self.literals.setdefault(variable, set()).add(
                            _text(value.value)
                        )
            elif (
                isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None
            ):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Attribute):
                        self.attributes.setdefault(target.attr, []).append(node.value)
                    elif (
                        isinstance(target, ast.Name)
                        and target.id.isupper()
                        and isinstance(node.value, ast.Call)
                        and node.value.args
                        and isinstance(node.value.args[0], ast.Constant)
                        and isinstance(node.value.args[0].value, bool)
                    ):
                        # A variable declared with a switch's default: ``NAME = EnvBool(False)``.
                        self.switches.add(target.id)
        self._visit(tree.body, [], (), ())

    def _visit(
        self, statements, conditions: list[ast.expr], scopes: tuple, functions: tuple
    ) -> None:
        conditions = list(conditions)
        for statement in statements:
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                self._visit(
                    statement.body,
                    conditions,
                    (_locals(statement), *scopes),
                    (statement.name, *functions),
                )
                continue
            if isinstance(statement, ast.If):
                self._calls(statement.test, conditions, scopes, functions)
                self._visit(
                    statement.body, [*conditions, statement.test], scopes, functions
                )
                self._visit(
                    statement.orelse,
                    [*conditions, _negated(statement.test)],
                    scopes,
                    functions,
                )
                if _exits(statement.body) and not statement.orelse:
                    # ``if not ready: return``: what follows needs ``ready``.
                    conditions.append(_negated(statement.test))
                continue
            for value in ast.iter_fields(statement):
                value = value[1]
                if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                    self._visit(value, conditions, scopes, functions)
                elif isinstance(value, ast.AST):
                    self._calls(value, conditions, scopes, functions)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self._calls(item, conditions, scopes, functions)

    def _calls(self, expression: ast.AST, conditions, scopes, functions) -> None:
        if isinstance(expression, ast.IfExp):
            self._calls(expression.test, conditions, scopes, functions)
            self._calls(
                expression.body, [*conditions, expression.test], scopes, functions
            )
            self._calls(
                expression.orelse,
                [*conditions, _negated(expression.test)],
                scopes,
                functions,
            )
            return
        if isinstance(
            expression,
            ast.Lambda | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        ):
            return
        self._record(expression, conditions, scopes, functions)
        for child in ast.iter_child_nodes(expression):
            self._calls(child, conditions, scopes, functions)

    def _record(self, node: ast.AST, conditions, scopes, functions) -> None:
        if isinstance(node, ast.Call):
            name = _called_name(node)
            if name:
                self.guards.setdefault(name, []).append(
                    _Site(tuple(conditions), scopes, functions)
                )

    def assignments(
        self, condition: ast.expr, scopes: tuple = (), options=None, depth: int = 0
    ) -> list[tuple[str, str]]:
        """``(setting, value)`` pairs that make ``condition`` hold.

        ``scopes`` are the locals around the condition; ``options`` maps a
        parsed argument's destination to its option and whether it is a
        switch.
        """
        options = options or {}
        if depth > _DEPTH:
            return []
        found: list[tuple[str, str]] = []
        if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.And):
            for value in condition.values:
                found.extend(self.assignments(value, scopes, options, depth))
            return found
        if isinstance(condition, ast.Compare) and len(condition.ops) == 1:
            left, right = condition.left, condition.comparators[0]
            setting = self._setting(left, options) or self._setting(right, options)
            constant = right if self._setting(left, options) else left
            if setting and isinstance(condition.ops[0], ast.Eq):
                if isinstance(constant, ast.Constant) and constant.value is not None:
                    found.append((setting, _text(constant.value)))
            elif (
                setting
                and isinstance(condition.ops[0], ast.In)
                and isinstance(constant, ast.Tuple | ast.List | ast.Set)
            ):
                values = [
                    item.value
                    for item in constant.elts
                    if isinstance(item, ast.Constant)
                ]
                if len(values) == 1:
                    found.append((setting, _text(values[0])))
            return found
        setting = self._setting(condition, options)
        if setting:
            # A switch read for its truth: set.
            switch = (
                not setting.startswith("--")
                or options.get(condition.attr, ("", False))[1]
            )
            return (
                [(setting, "1" if not setting.startswith("--") else "")]
                if switch
                else []
            )
        if isinstance(condition, ast.Name):
            for scope in scopes:
                if condition.id in scope:
                    return self.assignments(
                        scope[condition.id], scopes, options, depth + 1
                    )
            return []
        if isinstance(condition, ast.Attribute):
            assigned = self.attributes.get(condition.attr, ())
            if len(assigned) == 1:
                return self.assignments(assigned[0], (), options, depth + 1)
            return []
        if isinstance(condition, ast.Call):
            for returned in self.returns.get(_called_name(condition), ()):
                found.extend(self.assignments(returned, (), options, depth + 1))
        return found

    @staticmethod
    def _setting(node: ast.AST, options) -> str | None:
        variable = _env_reference(node)
        if variable:
            return variable
        if isinstance(node, ast.Attribute) and node.attr in options:
            return options[node.attr][0]
        return None


def _locals(function: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, ast.expr]:
    """The names a function assigns exactly once, and what to."""
    assigned: dict[str, list[ast.expr]] = {}
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assigned.setdefault(node.targets[0].id, []).append(node.value)
    return {name: values[0] for name, values in assigned.items() if len(values) == 1}


def _text(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


@functools.lru_cache(maxsize=None)
def _source(roots: tuple[str, ...]) -> _Source:
    return _Source(roots)


def _roots(engine_roots) -> tuple[str, ...]:
    """One source directory, or several read as one source."""
    if isinstance(engine_roots, str | Path):
        return (str(engine_roots),)
    return tuple(sorted(str(root) for root in engine_roots))


def inferred_settings(
    engine_roots,
    traced: set[tuple[str, str]],
    executed: set[tuple[str, str]],
    known: dict[str, str],
    options: dict[str, tuple[str, bool]] | None = None,
) -> dict[str, str]:
    """Settings that would reach the functions ``traced`` holds and the run did not.

    ``engine_roots`` are the source directories the run read (one or
    several, read as one). ``traced`` and ``executed`` are ``(path, function
    name)``; only names
    compare, as the two paths are of different machines. ``options`` are
    the command-line options the engine declares (destination -> option,
    switch). The trace records every frame, so a traced function was called
    from a function the trace also shows; what every such call needs is a
    setting the run must have had. A setting already ``known`` is kept.
    Each candidate value is weighed by how many missing functions it would
    reach; the most supported wins.
    """
    source = _source(_roots(engine_roots))
    ran = {name for _, name in executed}
    shown = {name for _, name in traced}
    votes: dict[str, Counter] = {}
    for name in {name for _, name in traced}:
        if name in ran or name.startswith("<"):
            continue
        sites = source.guards.get(name, ())
        if not sites or len(sites) > _MOST_CALL_SITES:
            continue
        # The trace records every frame: the call was made from a function
        # it shows running.
        sites = [
            site
            for site in sites
            if any(function in shown for function in site.functions)
        ]
        needed = None
        for site in sites:
            found = {
                pair
                for condition in site.conditions
                for pair in source.assignments(condition, site.scopes, options)
            }
            needed = found if needed is None else needed & found
        for setting, value in needed or ():
            if setting not in known:
                votes.setdefault(setting, Counter())[value] += 1
    return {setting: counter.most_common(1)[0][0] for setting, counter in votes.items()}


def _recipe_files(engine_root: Path) -> list[str]:
    texts = []
    for path in sorted(Path(engine_root).rglob("*")):
        if path.suffix in _RECIPE_SUFFIXES and path.is_file():
            try:
                texts.append(path.read_text(errors="ignore"))
            except OSError:
                continue
    return texts


def _stated_settings(text: str) -> dict[str, str]:
    stated: dict[str, str] = {}
    for name, value in [*_MAPPED.findall(text), *_ASSIGNED.findall(text)]:
        stated.setdefault(name, value.strip("\"'"))
    for name, value in _OPTION.findall(text):
        stated.setdefault(name, value.strip("\"'"))
    return stated


@functools.lru_cache(maxsize=8)
def recipe_values(engine_root: Path, model: str) -> dict[str, set[str]]:
    """The values the engine's own recipes for ``model`` give each variable.

    ``model`` is a repository id or directory; its last part names it. A
    recipe is a file of the engine's checkout that names the model; when none
    names it in full, the name is shortened a part at a time (a dated or
    quantized variant of a model runs as the model).
    """
    texts = _recipe_files(engine_root)
    parts = Path(model.rstrip("/")).name.split("-")
    recipes: list[str] = []
    while len(parts) >= _FEWEST_NAME_PARTS and not recipes:
        name = "-".join(parts)
        recipes = [text for text in texts if name in text]
        parts = parts[:-1]
    values: dict[str, set[str]] = {}
    for text in recipes:
        for setting, value in _stated_settings(text).items():
            values.setdefault(setting, set()).add(value)
    return values


def compared_values(engine_roots) -> dict[str, set[str]]:
    """The constants the source compares each variable or attribute with."""
    return _source(_roots(engine_roots)).literals


def switch_variables(engine_roots) -> set[str]:
    """The variables the source reads as a switch: declared or read with an on/off
    default, read through a helper for one, or compared with an on/off value."""
    source = _source(_roots(engine_roots))
    compared = {
        name for name, values in source.literals.items() if values & _SWITCH_VALUES
    }
    return source.switches | compared


def number_variables(engine_roots) -> set[str]:
    """The variables the source reads as a number."""
    return _source(_roots(engine_roots)).numbers


# A setting naming a checkpoint entry a loader looks for: ``checkpoint:<file>:<function>:<entry>``.
CHECKPOINT_ENTRY = "checkpoint:"


def _tested_literal(test: ast.expr, names: set[str]) -> str | None:
    """The string ``test`` looks for in an entry name: ``"x" in name``, ``name == "x"``,
    ``name.startswith("x")``, ``name.endswith("x")``."""
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left, right = test.left, test.comparators[0]
        if (
            isinstance(test.ops[0], ast.In)
            and isinstance(right, ast.Name)
            and right.id in names
        ):
            literal = left
        elif (
            isinstance(test.ops[0], ast.Eq)
            and isinstance(left, ast.Name)
            and left.id in names
        ):
            literal = right
        else:
            return None
        return (
            literal.value
            if isinstance(literal, ast.Constant) and isinstance(literal.value, str)
            else None
        )
    if (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Attribute)
        and test.func.attr in ("startswith", "endswith")
        and isinstance(test.func.value, ast.Name)
        and test.func.value.id in names
        and len(test.args) == 1
        and isinstance(test.args[0], ast.Constant)
        and isinstance(test.args[0].value, str)
    ):
        return test.args[0].value
    return None


def _sets_own_state(statements) -> bool:
    return any(
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
    )


def checkpoint_entries(executed: set[tuple[str, str]]) -> list[str]:
    """Checkpoint entries whose loading would make model state the run never made.

    The run's weight files hold no tensor, so a loader's loop over their
    entries runs no step. Where a function the run executed loops over the
    entries it is handed (``for name, weight in weights``) and, for an entry
    of a name it looks for, sets state of its own rather than a parameter
    (``self.hot_token_id = ...``), that entry decides what the model does:
    whether the checkpoint holds it is a setting, ``checkpoint:<file>:<function>:<name>``,
    whose value is the entry's length.
    """
    found = []
    for path, name in sorted(executed):
        try:
            tree = ast.parse(Path(path).read_text(errors="ignore"))
        except (OSError, SyntaxError, ValueError):
            continue
        for function in ast.walk(tree):
            if (
                not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
                or function.name != name
            ):
                continue
            parameters = {
                argument.arg
                for argument in function.args.args + function.args.kwonlyargs
            }
            for loop in ast.walk(function):
                if not (
                    isinstance(loop, ast.For)
                    and isinstance(loop.iter, ast.Name)
                    and loop.iter.id in parameters
                    and isinstance(loop.target, ast.Tuple)
                    and loop.target.elts
                    and isinstance(loop.target.elts[0], ast.Name)
                ):
                    continue
                names = {loop.target.elts[0].id}
                for test in ast.walk(loop):
                    if isinstance(test, ast.If) and _sets_own_state(test.body):
                        literal = _tested_literal(test.test, names)
                        setting = (
                            f"{CHECKPOINT_ENTRY}{Path(path).name}:{name}:{literal}"
                        )
                        if literal and setting not in found:
                            found.append(setting)
    return found
