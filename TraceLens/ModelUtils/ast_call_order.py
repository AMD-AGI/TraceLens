###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Static (no-execution) extraction of a module's child-call order.

Ported from the pre-``torch_trace`` AST-based graph builder
(``ast_analyze.py``, see git history commit ``c7ba9477`` and earlier), which
had well-developed, battle-tested logic for walking a ``forward()`` method's
AST and recovering the true evaluation order of ``self.<child>(...)`` calls —
including nested/method-chained calls (``self.a(self.b(x))`` evaluates ``b``
before ``a``), functional calls, and rope-style free functions.

This is used by ``torch_trace.py`` purely as an ordering fallback for
composite modules that have no runtime call-graph signal at all (e.g. an
optional modality branch, like a vision encoder, whose real inputs were
omitted from the dummy forward pass). Real tensor-ID tracing always takes
priority when available; this only kicks in to disambiguate declaration
order (``named_children()``) from actual execution order, which can differ
(e.g. a norm declared last in ``__init__`` but applied before a later-
declared submodule in ``forward()``).
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap

import torch

# ─────────────────────────────────────────────────────────────────────────
# Constants (ported as-is from ast_analyze.py)
# ─────────────────────────────────────────────────────────────────────────

_METHOD_CHAIN_OPS = {
    "view",
    "transpose",
    "reshape",
    "contiguous",
    "type",
    "float",
    "squeeze",
    "unsqueeze",
    "expand",
    "split",
    "mul",
    "mul_",
    "sum",
    "sigmoid",
}

SYNTHETIC_ATTENTION = "@attention"
_SYNTHETIC_ATTENTION_NAMES = {
    "eager_attention_forward",
    "flash_attention_forward",
    "sdpa_attention_forward",
    "attention_interface",
}

FUNCTIONAL_SYNTHETIC_PREFIX = "@functional_"
POSITIONAL_SYNTHETIC_PREFIX = "@positional_"
_POSITIONAL_SOURCE_POS_RE = re.compile(
    rf"^{re.escape(POSITIONAL_SYNTHETIC_PREFIX)}l(\d+)_"
)
POSITIONAL_ATTR_RE = re.compile(r"(rotary|rope|pos_emb)", re.I)

_DATA_MOVEMENT_NAMES = frozenset(
    {
        "cat",
        "stack",
        "split",
        "view",
        "reshape",
        "transpose",
        "permute",
        "contiguous",
        "squeeze",
        "unsqueeze",
        "flatten",
        "pad",
        "index_select",
        "gather",
        "rearrange",
        "index_first_axis",
        "pad_input",
        "get_unpad_data",
        "unpad_input",
    }
)
_KERNEL_MERGE_NAME_RE = re.compile(
    r"(attention|attn|recurrent|flash|sdpa|linear_attn|kernel|chunk)",
    re.IGNORECASE,
)


def positional_synthetic_attr(func_name: str, lineno: int) -> str:
    return f"{POSITIONAL_SYNTHETIC_PREFIX}l{lineno}_{func_name}"


def functional_synthetic_attr(op_name: str) -> str:
    return f"{FUNCTIONAL_SYNTHETIC_PREFIX}{op_name}"


def _expr_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _expr_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Subscript):
        return _expr_name(node.value)
    if isinstance(node, ast.Call):
        return _expr_name(node.func)
    return None


def _is_self_attr(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == attr
    )


def _unwrap_expr(node: ast.AST) -> ast.AST:
    while isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        owner = _expr_name(node.func.value)
        if node.func.attr in _METHOD_CHAIN_OPS and owner not in {
            "torch",
            "F",
            "torch.nn.functional",
            "nn.functional",
        }:
            return _unwrap_expr(node.func.value)
    return node


def _functional_call_name(func: ast.AST) -> str | None:
    """Return the op name for F.<op>(...) and torch.nn.functional.<op>(...)."""
    if not isinstance(func, ast.Attribute):
        return None
    op_name = func.attr
    value = func.value
    if isinstance(value, ast.Name) and value.id == "F":
        return op_name
    if isinstance(value, ast.Attribute) and value.attr == "functional":
        base = value.value
        if isinstance(base, ast.Attribute) and base.attr == "nn":
            if isinstance(base.value, ast.Name) and base.value.id == "torch":
                return op_name
    return None


def _is_functional_linear_call(func: ast.AST) -> bool:
    return _functional_call_name(func) == "linear"


def _is_data_movement_call(func: ast.AST) -> bool:
    name = _expr_name(func)
    if not name:
        return False
    base = name.split(".")[-1]
    return base in _DATA_MOVEMENT_NAMES


def _is_kernel_merge_call(func: ast.AST) -> bool:
    if _is_data_movement_call(func):
        return False
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
    ):
        return False
    if _is_functional_linear_call(func):
        return False
    name = _expr_name(func) or ""
    base = name.split(".")[-1]
    if base in _SYNTHETIC_ATTENTION_NAMES:
        return True
    return bool(_KERNEL_MERGE_NAME_RE.search(base))


def _is_positional_function_call(func: ast.AST, target: str) -> bool:
    """True for a bare call to a rope helper such as `apply_rotary_emb(q, freqs)`."""
    if not isinstance(func, ast.Name):
        return False
    return bool(POSITIONAL_ATTR_RE.search(target))


def _append_forward_call(calls: list[str], attr: str) -> None:
    if calls and calls[-1] == attr:
        return
    calls.append(attr)


def _extract_self_calls_ordered(node: ast.AST, out: list[str]) -> None:
    """Collect self.module(...) calls in approximate evaluation order (inner-first)."""
    node = _unwrap_expr(node)
    if isinstance(node, ast.Call):
        for arg in node.args:
            _extract_self_calls_ordered(arg, out)
        for keyword in node.keywords:
            _extract_self_calls_ordered(keyword.value, out)

        func = node.func
        if isinstance(func, ast.Attribute) and _is_self_attr(func, func.attr):
            _append_forward_call(out, func.attr)
            return
        functional_op = _functional_call_name(func)
        if functional_op:
            _append_forward_call(out, functional_synthetic_attr(functional_op))
            return

        target = _expr_name(func)
        if target in _SYNTHETIC_ATTENTION_NAMES or _is_kernel_merge_call(func):
            _append_forward_call(out, SYNTHETIC_ATTENTION)
            return
        if target and _is_positional_function_call(func, target):
            _append_forward_call(out, positional_synthetic_attr(target, node.lineno))
            return
        return

    if isinstance(node, ast.BinOp):
        _extract_self_calls_ordered(node.left, out)
        _extract_self_calls_ordered(node.right, out)
        return

    if isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            _extract_self_calls_ordered(elt, out)
        return

    if isinstance(node, ast.IfExp):
        _extract_self_calls_ordered(node.body, out)
        _extract_self_calls_ordered(node.orelse, out)
        return

    if isinstance(node, ast.Subscript):
        _extract_self_calls_ordered(node.value, out)
        return

    if isinstance(node, ast.Compare):
        _extract_self_calls_ordered(node.left, out)
        for comparator in node.comparators:
            _extract_self_calls_ordered(comparator, out)
        return


def _resolve_local_module_alias_calls(func: ast.AST) -> ast.AST:
    """Rewrite ``expert(...)`` aliases of ``self.experts[i]`` as module calls.

    Expert loops commonly bind one entry from a ModuleList to a local variable
    before invoking it.  Resolving that alias lets the call-order walk retain
    the real routed-expert branch instead of silently dropping it.
    """
    import copy

    aliases: dict[str, str] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Attribute)
            and _is_self_attr(value.value, value.value.attr)
        ):
            aliases[target.id] = value.value.attr
    if not aliases:
        return func

    resolved = copy.deepcopy(func)

    class AliasCallResolver(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in aliases:
                node.func = ast.copy_location(
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr=aliases[node.func.id],
                        ctx=ast.Load(),
                    ),
                    node.func,
                )
            return node

    return AliasCallResolver().visit(resolved)


# ─────────────────────────────────────────────────────────────────────────
# New glue: walk a full forward() body in source order, extending the old
# expression-level extractor with ModuleList for-loop detection (the old
# pipeline didn't need this since it tracked individual layers elsewhere;
# here we need the *container* itself as a single ordering position).
# ─────────────────────────────────────────────────────────────────────────

_STMT_VALUE_FIELDS: tuple[type, ...] = (
    ast.Assign,
    ast.AnnAssign,
    ast.Expr,
    ast.AugAssign,
    ast.Return,
)


def _walk_stmts(
    stmts: list[ast.stmt],
    calls: list[str],
    loop_var_to_child: dict[str, str],
    child_names: set[str],
) -> None:
    for node in stmts:
        if isinstance(node, ast.For):
            it = node.iter
            if (
                isinstance(node.target, ast.Name)
                and isinstance(it, ast.Attribute)
                and isinstance(it.value, ast.Name)
                and it.value.id == "self"
                and it.attr in child_names
            ):
                loop_var_to_child[node.target.id] = it.attr
                _append_forward_call(calls, it.attr)
            _walk_stmts(node.body, calls, loop_var_to_child, child_names)
            _walk_stmts(node.orelse, calls, loop_var_to_child, child_names)
            continue
        if isinstance(node, ast.While):
            _walk_stmts(node.body, calls, loop_var_to_child, child_names)
            _walk_stmts(node.orelse, calls, loop_var_to_child, child_names)
            continue
        if isinstance(node, ast.If):
            _walk_stmts(node.body, calls, loop_var_to_child, child_names)
            _walk_stmts(node.orelse, calls, loop_var_to_child, child_names)
            continue
        if isinstance(node, ast.With):
            _walk_stmts(node.body, calls, loop_var_to_child, child_names)
            continue
        if isinstance(node, ast.Try):
            _walk_stmts(node.body, calls, loop_var_to_child, child_names)
            for handler in node.handlers:
                _walk_stmts(handler.body, calls, loop_var_to_child, child_names)
            _walk_stmts(node.orelse, calls, loop_var_to_child, child_names)
            _walk_stmts(node.finalbody, calls, loop_var_to_child, child_names)
            continue
        if isinstance(node, _STMT_VALUE_FIELDS):
            value = getattr(node, "value", None)
            if value is None:
                continue
            stmt_calls: list[str] = []
            _extract_self_calls_ordered(value, stmt_calls)
            for attr in stmt_calls:
                _append_forward_call(calls, attr)
            # Bare loop-variable calls, e.g. `hidden_states = blk(...)`, aren't
            # caught by `_extract_self_calls_ordered` (it only resolves
            # `self.<attr>` — the loop already recorded the container above).
            for call_node in ast.walk(value):
                if (
                    isinstance(call_node, ast.Call)
                    and isinstance(call_node.func, ast.Name)
                    and call_node.func.id in loop_var_to_child
                ):
                    _append_forward_call(calls, loop_var_to_child[call_node.func.id])


def forward_call_order(mod: torch.nn.Module) -> list[str] | None:
    """Best-effort order that ``mod.forward`` invokes its direct children.

    Returns a list of attribute names (a subset of ``mod.named_children()``)
    in the order they're first invoked in ``forward()``'s source, or ``None``
    if the source is unavailable/unparseable or no direct children were
    found being called (callers should keep declaration order in that case).
    """
    child_names = {n for n, _ in mod.named_children()}
    if not child_names:
        return None
    try:
        src = inspect.getsource(type(mod).forward)
    except (OSError, TypeError):
        return None
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return None
    if not tree.body or not isinstance(
        tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        return None
    func = _resolve_local_module_alias_calls(tree.body[0])

    calls: list[str] = []
    loop_var_to_child: dict[str, str] = {}
    _walk_stmts(func.body, calls, loop_var_to_child, child_names)

    seen: set[str] = set()
    result: list[str] = []
    for c in calls:
        if c in child_names and c not in seen:
            seen.add(c)
            result.append(c)
    return result or None
