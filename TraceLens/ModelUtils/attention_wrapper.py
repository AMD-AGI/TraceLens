###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Introspect a dispatched attention kernel's Python wrapper into its live ops.

A model that runs attention through ``ALL_ATTENTION_FUNCTIONS[impl]`` dispatches
to a small wrapper (SDPA's ``sdpa_attention_forward``) that repeats key/value
heads for grouped-query attention, calls the compiled
``scaled_dot_product_attention`` primitive, then transposes and makes the result
contiguous. Rendering that wrapper as one opaque leaf hides the post-kernel layout
ops; this module resolves the wrapper's tensor-input ports (mapping the caller's
positional arguments to the wrapper's parameters) and the ordered tail method
chain so the block tree can materialise the primitive as an atomic leaf inside a
visible ``sdpa -> transpose -> contiguous`` subtree. The compiled primitive itself
is never introspected -- it is the atomic boundary.

Only structural facts are read (parameter order, the caller's argument order, the
tail method chain and its literal int arguments); no wrapper body is executed and
no op allow-list gates the extraction.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from TraceLens.ModelUtils.ast_analyze import (
    _HostSourceResolver,
    _SYNTHETIC_ATTENTION_NAMES,
    attention_wrapper_expand_location,
    attention_wrapper_gqa_groups,
)


@dataclass
class WrapperExpansion:
    """Resolved plan for expanding a dispatched attention wrapper."""

    kernel: str
    gqa_groups: int
    # Ordered ``(wrapper_param, caller_var)`` tensor-input ports, e.g.
    # ``[("query", "q"), ("key", "kv"), ("value", "kv"), ("attention_mask", "mask")]``.
    # Two params sharing one caller var (``key``/``value`` <- ``kv``) is how a
    # grouped-query call feeds both key and value from a single tensor.
    port_map: list[tuple[str, str]]
    # Wrapper params re-materialised through a grouped-query head repeat
    # (``key = repeat_kv(key, ...)``); read structurally from the wrapper body,
    # not a param-name allow-list. Empty when the wrapper never repeats heads.
    repeat_params: frozenset[str]
    # Ordered ``(method, int_args)`` tensor methods applied to the kernel result
    # before the wrapper returns (``attn_output.transpose(1, 2).contiguous()`` ->
    # ``(("transpose", (1, 2)), ("contiguous", ()))``). Each becomes a visible op
    # after the atomic kernel; the literal int args (a ``transpose``'s two dims)
    # are carried so the op's shape rule permutes the correct axes.
    tail_ops: tuple[tuple[str, tuple[int, ...]], ...]


def _resolve_wrapper_def(module: str, symbol: str) -> ast.FunctionDef | None:
    """Locate the wrapper's ``FunctionDef`` by source AST (no execution).

    Mirrors ``_HostSourceResolver.runs_on_host``'s resolution: load the module,
    return its local definition, else follow a re-export binding to the defining
    module. ``None`` when the source cannot be located.
    """
    resolver = _HostSourceResolver()
    seen: set[tuple[str, str]] = set()
    cur_module, cur_symbol = module, symbol
    while (cur_module, cur_symbol) not in seen:
        seen.add((cur_module, cur_symbol))
        loaded = resolver._load(cur_module)
        if loaded is None:
            return None
        funcs, imports = loaded
        func = funcs.get(cur_symbol)
        if func is not None:
            return func
        binding = imports.get(cur_symbol)
        if not binding:
            return None
        dest, _, dest_symbol = binding.partition("#")
        if not dest or (dest == cur_module and dest_symbol == cur_symbol):
            return None
        cur_module, cur_symbol = dest, dest_symbol
    return None


def _wrapper_param_order(func: ast.FunctionDef) -> list[str]:
    """Ordered parameter names of the resolved wrapper (``module``/``self`` dropped)."""
    params = [arg.arg for arg in func.args.args]
    # The first parameter is the attention module handle (``module``/``self``); the
    # tensor inputs the caller passes positionally start after it.
    return params[1:] if params else []


def _wrapper_repeat_params(func: ast.FunctionDef) -> frozenset[str]:
    """Params re-materialised through a call on themselves (grouped-query repeat).

    ``key = repeat_kv(key, ...)`` reassigns ``key`` from a call whose first positional
    argument is ``key`` itself -- the structural signature of a head-repeat that
    replaces a param with an expanded copy. Keyed on that self-reassignment shape
    alone (single ``Name`` target equal to the call's first ``Name`` arg), so no
    function-name or param-name allow-list is needed; a layout op written as
    ``x = x.transpose(...)`` (method call, non-``Name`` first arg) or a slice
    ``x = x[...]`` (subscript, not a call) never matches.
    """
    repeated: set[str] = set()
    for stmt in ast.walk(func):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        value = stmt.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Call)
            and not isinstance(value.func, ast.Attribute)
            and value.args
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == target.id
        ):
            repeated.add(target.id)
    return frozenset(repeated)


def _int_call_args(call: ast.Call) -> tuple[int, ...]:
    """Literal ``int`` positional args of a method call (``.transpose(1, 2)`` -> ``(1, 2)``).

    Stops at the first non-int positional so a mixed call never yields a partial
    misaligned tuple; a call with no int args (``.contiguous()``) returns ``()``.
    """
    args: list[int] = []
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int) and not isinstance(
            arg.value, bool
        ):
            args.append(arg.value)
        else:
            break
    return tuple(args)


def _wrapper_tail_ops(func: ast.FunctionDef) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Ordered ``(method, int_args)`` tensor methods applied to the kernel result.

    Finds the variable the wrapper returns (first element of a ``return (x, ...)``
    tuple, or a bare ``return x``) and reads the method chain of the assignment that
    re-materialises it (``attn_output = attn_output.transpose(1, 2).contiguous()``),
    returning the methods in application order with each call's literal int args
    (``(("transpose", (1, 2)), ("contiguous", ()))``). Structural: the base of the
    chain must be the returned name itself, so an unrelated assignment is never
    mistaken for the output post-processing.
    """
    returned: str | None = None
    for stmt in ast.walk(func):
        if not isinstance(stmt, ast.Return) or stmt.value is None:
            continue
        expr = stmt.value
        if isinstance(expr, ast.Tuple) and expr.elts:
            expr = expr.elts[0]
        if isinstance(expr, ast.Name):
            returned = expr.id
        break
    if returned is None:
        return ()

    for stmt in ast.walk(func):
        if (
            not isinstance(stmt, ast.Assign)
            or len(stmt.targets) != 1
            or not isinstance(stmt.targets[0], ast.Name)
            or stmt.targets[0].id != returned
        ):
            continue
        methods: list[tuple[str, tuple[int, ...]]] = []
        node: ast.AST = stmt.value
        while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            methods.append((node.func.attr, _int_call_args(node)))
            node = node.func.value
        if isinstance(node, ast.Name) and node.id == returned and methods:
            methods.reverse()
            return tuple(methods)
    return ()


def _interface_call_args(cls_node: ast.AST) -> list[str] | None:
    """Ordered positional caller variables of the first attention-interface call."""
    for call in ast.walk(cls_node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name not in _SYNTHETIC_ATTENTION_NAMES:
            continue
        start = (
            1
            if call.args
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "self"
            else 0
        )
        args: list[str] = []
        for arg in call.args[start:]:
            if isinstance(arg, ast.Name):
                args.append(arg.id)
            else:
                # A non-Name positional (a literal / attribute) is not a wired
                # tensor input; stop so the positional zip stays aligned.
                break
        return args
    return None


def introspect_attention_wrapper(
    details: list[str], cls_node: ast.AST
) -> WrapperExpansion | None:
    """Resolve the wrapper-expansion plan from a step's stamped details.

    Returns ``None`` when the wrapper source cannot be located or the caller's
    interface call cannot be read -- the caller then keeps the atomic leaf.
    """
    location = attention_wrapper_expand_location(details)
    if location is None:
        return None
    func = _resolve_wrapper_def(*location)
    if func is None:
        return None
    params = _wrapper_param_order(func)
    if not params:
        return None
    caller_args = _interface_call_args(cls_node)
    if not caller_args:
        return None

    # The dispatched interface passes tensor inputs positionally in wrapper-param
    # order (query, key, value, attention_mask, ...); zip them so each wrapper
    # parameter learns which caller variable feeds it. Extra positional caller
    # args beyond the named params (or extra params beyond the caller's args) are
    # dropped by the shortest-sequence zip.
    port_map = list(zip(params, caller_args))
    if not port_map:
        return None

    kernel = location[1]
    for line in details:
        if line.startswith("kernel:"):
            kernel = line.split(":", 1)[1].strip()
            break

    # ``gqa_groups`` is resolved best-effort and carried on the expansion, but is
    # not consumed until ``repeat_kv`` is modelled as its ``expand``/``reshape``
    # primitives (a later commit). Deliberately silent here: an ambiguous factor
    # only matters once it would gate emitting those primitives, so warning now
    # -- on every affected layer, for a feature this expansion does not yet emit
    # -- is premature noise. The tail/kernel expansion is shape-correct without
    # it (``repeat_kv`` grows the head *count* from the query side; the sdpa
    # output rule reads the value's last dim, unaffected by the repeat).
    groups = attention_wrapper_gqa_groups(details)
    return WrapperExpansion(
        kernel=kernel,
        gqa_groups=groups if (groups and groups > 1) else 1,
        port_map=port_map,
        repeat_params=_wrapper_repeat_params(func),
        tail_ops=_wrapper_tail_ops(func),
    )
