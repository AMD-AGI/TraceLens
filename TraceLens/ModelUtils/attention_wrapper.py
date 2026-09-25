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
import logging
from dataclasses import dataclass, field

from TraceLens.ModelUtils.ast_analyze import (
    _HostSourceResolver,
    _SYNTHETIC_ATTENTION_NAMES,
    attention_wrapper_expand_location,
    attention_wrapper_gqa_groups,
)

_log = logging.getLogger(__name__)

# One derived op of a head-repeat expansion: the display label the shape rule
# dispatches on (``Unsqueeze``/``Expand``/``Reshape``) and the ordered detail
# lines that drive that rule (``raw_op:``/``dim:``/``shape:``). Read structurally
# from the resolved repeat callee's body -- never a hardcoded op sequence.
RepeatOp = tuple[str, tuple[str, ...]]


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
    # The atomic torch primitive the wrapper calls to produce its result
    # (``scaled_dot_product_attention``), read structurally from the body as the
    # free-function/callable whose result becomes the returned var before the tail
    # method chain. Carried so the core leaf can stamp it as its ``raw_op``, letting
    # the type-check resolve the primitive's real required tensor-operand contract
    # (query/key/value) from its aten schema -- never a kernel-name allow-list.
    # ``None`` when the primitive call cannot be identified from the body.
    kernel_primitive: str | None = None
    # Ordered introspected ops that re-materialise a grouped-query head repeat
    # (``key = repeat_kv(key, n)``), DERIVED from the resolved repeat callee's real
    # body -- HF ``repeat_kv`` is ``x[:, :, None, :, :].expand(...).reshape(...)`` ->
    # ``(("Unsqueeze", ...), ("Expand", ...), ("Reshape", ...))``. Each entry's
    # detail lines drive the existing unsqueeze/expand/reshape shape rules, with the
    # grown axis + factor read from the body's own ``expand``/``reshape`` arguments
    # (the factor resolved from the live ``gqa_groups`` the module reports). The same
    # chain applies to every param in ``repeat_params`` (they share the callee). Empty
    # when the wrapper never repeats heads, the live factor is 1, or the callee body
    # is not a resolvable shape-op chain (the branch then keeps a bare port).
    repeat_ops: tuple[RepeatOp, ...] = field(default_factory=tuple)
    # Host module + symbol of the head-repeat callee (``repeat_kv``), carried raw so
    # a consumer can RE-derive :data:`repeat_ops` for a live factor that is not yet
    # known when the plan is first built. The grouped-query factor is stamped onto
    # the attention step only later (``reconcile_live_attention_groups`` runs in the
    # export build, after the block tree is constructed), so the block tree records
    # this context and a post-reconcile pass resolves the callee and derives the ops
    # with the real factor. ``None`` when the wrapper has no resolvable repeat call.
    repeat_module: str | None = None
    repeat_callee: str | None = None


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
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, int)
            and not isinstance(arg.value, bool)
        ):
            args.append(arg.value)
        else:
            break
    return tuple(args)


def _wrapper_returned_name(func: ast.FunctionDef) -> str | None:
    """The variable the wrapper returns (first element of ``return (x, ...)`` or a
    bare ``return x``), or ``None``."""
    for stmt in ast.walk(func):
        if not isinstance(stmt, ast.Return) or stmt.value is None:
            continue
        expr = stmt.value
        if isinstance(expr, ast.Tuple) and expr.elts:
            expr = expr.elts[0]
        if isinstance(expr, ast.Name):
            return expr.id
        break
    return None


def _wrapper_kernel_primitive(func: ast.FunctionDef) -> str | None:
    """The atomic torch primitive the wrapper calls to produce its result.

    The returned var (``attn_output``) is assigned twice: once from the primitive
    call (``attn_output = F.scaled_dot_product_attention(...)``) and once from the
    tail method chain rooted on itself (``attn_output = attn_output.transpose(...)``
    -- handled by :func:`_wrapper_tail_ops`). This returns the callee name of the
    FIRST assignment whose value is a call NOT rooted on the returned name -- the
    primitive that feeds the tail. Structural: the tail chain (whose call base
    resolves back to the returned name) is skipped, so a layout re-materialisation
    is never mistaken for the primitive. ``None`` when no such call is found.
    """
    returned = _wrapper_returned_name(func)
    if returned is None:
        return None
    for stmt in ast.walk(func):
        if (
            not isinstance(stmt, ast.Assign)
            or len(stmt.targets) != 1
            or not isinstance(stmt.targets[0], ast.Name)
            or stmt.targets[0].id != returned
            or not isinstance(stmt.value, ast.Call)
        ):
            continue
        call = stmt.value
        base = call.func
        while isinstance(base, ast.Attribute):
            base = base.value
        if isinstance(base, ast.Name) and base.id == returned:
            # The tail re-materialisation (``attn_output.transpose(...)...``); the
            # primitive is a different assignment to the same name.
            continue
        func_node = call.func
        if isinstance(func_node, ast.Attribute):
            return func_node.attr
        if isinstance(func_node, ast.Name):
            return func_node.id
        return None
    return None


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
    returned = _wrapper_returned_name(func)
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
        name = (
            func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        )
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


def _repeat_call_callee(func: ast.FunctionDef) -> str | None:
    """The callee name of a grouped-query head-repeat re-assignment.

    ``key = repeat_kv(key, n)`` re-binds a param from a call whose first positional
    argument is the param itself (the same structural signature
    :func:`_wrapper_repeat_params` keys on). This returns that call's callee name
    (a bare ``Name`` -- ``repeat_kv``), so its body can be resolved and introspected.
    ``None`` when no such self-reassigning call is found. Keyed on the structure,
    never on the callee's name.
    """
    for stmt in ast.walk(func):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        value = stmt.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.args
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == target.id
        ):
            return value.func.id
    return None


def _shape_unpack_axes(callee: ast.FunctionDef, param0: str) -> dict[str, int]:
    """Map each local unpacked from ``param0.shape`` to its axis position.

    ``batch, num_key_value_heads, slen, head_dim = hidden_states.shape`` yields
    ``{"batch": 0, "num_key_value_heads": 1, "slen": 2, "head_dim": 3}`` -- the
    names the body then uses to name the ``expand``/``reshape`` target axes.
    """
    for stmt in callee.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Tuple)
            and isinstance(stmt.value, ast.Attribute)
            and stmt.value.attr == "shape"
            and isinstance(stmt.value.value, ast.Name)
            and stmt.value.value.id == param0
        ):
            axes: dict[str, int] = {}
            for index, elt in enumerate(stmt.targets[0].elts):
                if isinstance(elt, ast.Name):
                    axes[elt.id] = index
            return axes
    return {}


def _none_index_positions(subscript: ast.Subscript) -> list[int]:
    """Positions of a ``None`` (``x[:, :, None, :, :]``) in a subscript's slice.

    Each ``None`` inserts a size-1 axis at that position (an ``unsqueeze``). Only
    a subscript whose other elements are plain full slices (``:``) is a pure
    unsqueeze; an integer index would drop an axis, so such a subscript returns an
    empty list and the caller treats the chain as non-derivable.
    """
    sl = subscript.slice
    elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
    positions: list[int] = []
    for index, elt in enumerate(elts):
        if isinstance(elt, ast.Constant) and elt.value is None:
            positions.append(index)
        elif isinstance(elt, ast.Slice) and elt.lower is None and elt.upper is None:
            continue
        else:
            # A non-trivial index (select / bounded slice) is not a pure
            # unsqueeze; signal non-derivable.
            return []
    return positions


def _unroll_transform_chain(
    expr: ast.expr, base_name: str
) -> list[tuple[str, ast.AST]] | None:
    """Ordered ``(kind, node)`` ops of a method/subscript chain rooted on ``base_name``.

    ``hidden_states[:, :, None, :, :].expand(...)`` unrolls (innermost-first) to
    ``[("subscript", <Subscript>), ("method:expand", <Call>)]``. Returns ``None``
    when the chain's innermost base is not the tracked variable, so an unrelated
    expression is never mistaken for the transform.
    """
    ops: list[tuple[str, ast.AST]] = []
    node: ast.AST = expr
    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            ops.append((f"method:{node.func.attr}", node))
            node = node.func.value
        elif isinstance(node, ast.Subscript):
            ops.append(("subscript", node))
            node = node.value
        else:
            break
    if isinstance(node, ast.Name) and node.id == base_name:
        ops.reverse()
        return ops
    return None


def _derive_repeat_ops(callee: ast.FunctionDef, factor: int) -> tuple[RepeatOp, ...]:
    """Introspect a head-repeat callee's body into ordered visible shape ops.

    Reads the callee's real body (no execution): the ``param.shape`` unpack names
    each source axis; the transform chain re-assigning ``param`` (``x[:, :, None,
    :, :].expand(...).reshape(...)``) is unrolled into an ordered op list. Each op
    becomes a ``(label, details)`` pair whose details drive the existing shape rule:

    * a ``None``-index subscript -> ``Unsqueeze`` (``dim:`` = the ``None`` position),
    * ``.expand(...)`` -> ``Expand`` (``shape:`` = ``-1`` per kept axis, the literal
      *factor* at the axis whose arg is the callee's repeat-count parameter),
    * ``.reshape(...)`` / ``.view(...)`` -> ``Reshape`` (``shape:`` = a source-axis
      reference ``x.shape[i]`` per bare name kept, a single ``-1`` for the merged
      axis, resolved by element conservation).

    The grown axis and factor are DERIVED from the body's own ``expand``/``reshape``
    arguments -- not assumed to be the kv-head axis or ``num_key_value_groups``.
    Returns ``()`` when the body is not a resolvable pure shape-op chain, so the
    caller keeps a bare port (the introspect-everything fallback, logged).
    """
    params = callee.args.args
    if len(params) < 2:
        return ()
    param0 = params[0].arg
    factor_param = params[1].arg
    axis_of = _shape_unpack_axes(callee, param0)
    if not axis_of:
        return ()

    # Running axis layout: each entry is the local name that names that axis (or an
    # inserted-axis placeholder). Lets a later ``reshape`` arg naming a source axis
    # resolve to that axis's CURRENT position after any unsqueeze shifted it.
    layout: list[str | None] = [None] * (max(axis_of.values()) + 1)
    for name, index in axis_of.items():
        layout[index] = name

    ops: list[RepeatOp] = []

    def _emit_chain(expr: ast.expr, base: str) -> bool:
        chain = _unroll_transform_chain(expr, base)
        if chain is None:
            return False
        for kind, node in chain:
            if kind == "subscript":
                positions = _none_index_positions(node)  # type: ignore[arg-type]
                if not positions:
                    return False
                for pos in positions:
                    ops.append(("Unsqueeze", ("raw_op: unsqueeze", f"dim: {pos}")))
                    layout.insert(pos, None)
            elif kind == "method:expand":
                call = node  # type: ignore[assignment]
                parts: list[str] = []
                for arg in call.args:  # type: ignore[attr-defined]
                    if isinstance(arg, ast.Name) and arg.id == factor_param:
                        parts.append(str(factor))
                    else:
                        # Every non-repeat axis is kept (``-1`` = keep, matching a
                        # bare source-axis name arg's meaning under ``expand``).
                        parts.append("-1")
                if len(parts) != len(layout):
                    return False
                ops.append(("Expand", ("raw_op: expand", f"shape: {', '.join(parts)}")))
            elif kind in ("method:reshape", "method:view"):
                call = node  # type: ignore[assignment]
                method = kind.split(":", 1)[1]
                parts = []
                merged = 0
                for arg in call.args:  # type: ignore[attr-defined]
                    if isinstance(arg, ast.Name) and arg.id in layout:
                        parts.append(f"x.shape[{layout.index(arg.id)}]")
                    elif isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                        parts.append(str(arg.value))
                    else:
                        # A product/merge of source axes (``kv * n_rep``) collapses
                        # to one axis; element conservation fills it.
                        parts.append("-1")
                        merged += 1
                if merged > 1:
                    return False
                ops.append(
                    (
                        method.capitalize(),
                        (f"raw_op: {method}", f"shape: {', '.join(parts)}"),
                    )
                )
            else:
                return False
        return True

    cur = param0
    for stmt in callee.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            if _emit_chain(stmt.value, cur):
                cur = stmt.targets[0].id
        elif isinstance(stmt, ast.Return) and stmt.value is not None:
            if not _emit_chain(stmt.value, cur):
                # A bare ``return hidden_states`` (the ``n_rep == 1`` guard, nested
                # in an ``if``) is not a transform chain and is never the top-level
                # return here; the real return carries the reshape.
                continue
            break
    return tuple(ops)


def _wrapper_repeat_ops(
    wrapper_module: str, func: ast.FunctionDef, factor: int | None
) -> tuple[RepeatOp, ...]:
    """Resolve the head-repeat callee and derive its visible shape ops.

    Gated on the LIVE repeat factor: ``repeat_kv`` short-circuits (returns its input
    unchanged) when ``n_rep == 1``, so a factor of 1 (GLM) emits no ops. Resolves the
    callee's ``FunctionDef`` through the same source-AST machinery as the wrapper
    itself (following a re-export binding), then derives the ops from its body.
    Returns ``()`` -- keeping a bare port -- when the factor is not >1, the callee
    cannot be resolved, or its body is not a pure shape-op chain (logged: the
    introspect-everything fallback, so a genuinely un-introspectable repeat is
    reported rather than silently dropped).
    """
    if factor is None or factor <= 1:
        return ()
    callee_symbol = _repeat_call_callee(func)
    if callee_symbol is None:
        return ()
    callee = _resolve_wrapper_def(wrapper_module, callee_symbol)
    if callee is None:
        _log.warning(
            "attention wrapper repeat callee %r could not be resolved from %r; "
            "keeping a bare key/value port (repeat not materialised)",
            callee_symbol,
            wrapper_module,
        )
        return ()
    ops = _derive_repeat_ops(callee, factor)
    if not ops:
        _log.warning(
            "attention wrapper repeat callee %r body is not a resolvable shape-op "
            "chain; keeping a bare key/value port (repeat not materialised)",
            callee_symbol,
        )
    return ops


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

    # ``gqa_groups`` is the LIVE grouped-query repeat factor the module reports
    # (harvested from the meta tree). It gates and sizes the ``repeat_kv``
    # expansion below: a factor of 1 (GLM) emits no repeat ops (``repeat_kv``
    # short-circuits), a factor >1 (DeepSeek 64, MiniMax 16) grows the key/value
    # head axis by it. Left silent when unresolved -- the branch then keeps a bare
    # port rather than guessing a factor.
    groups = attention_wrapper_gqa_groups(details)
    factor = groups if (groups and groups > 1) else 1
    return WrapperExpansion(
        kernel=kernel,
        gqa_groups=factor,
        port_map=port_map,
        repeat_params=_wrapper_repeat_params(func),
        tail_ops=_wrapper_tail_ops(func),
        kernel_primitive=_wrapper_kernel_primitive(func),
        repeat_ops=_wrapper_repeat_ops(location[0], func, factor),
        repeat_module=location[0],
        repeat_callee=_repeat_call_callee(func),
    )
