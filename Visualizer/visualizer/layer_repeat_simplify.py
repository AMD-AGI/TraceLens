"""Substitute config values and simplify layer-repeat fact sheet expressions."""

from __future__ import annotations

import ast
import json
import operator
import re
from typing import Any

_BRANCH_LINE_RE = re.compile(
    r"^(?P<prefix>.+) \((?P<branch>if|elif|else)(?: (?P<expr>.*))?\)$"
)
_LOOP_LINE_RE = re.compile(
    r"^(?P<head>\d+|N) × (?P<class_name>[^(]+) \((?P<loop_var>\w+) in (?P<range_expr>.+)\)$"
)

def simplify_layer_repeat_lines(
    lines: list[str],
    config: dict[str, Any],
) -> list[str]:
    """Return fact-sheet layer-repeat lines with config literals and simpler conditions."""
    if not config:
        return list(lines)
    return [_simplify_line(line, config) for line in lines]


def layer_condition_matches(layer_idx: int, condition: str, config: dict[str, Any]) -> bool:
    """Return True when a decoder __init__ branch condition matches ``layer_idx``."""
    expr = condition.strip()
    if expr == "else":
        return True
    if expr.startswith("if "):
        expr = expr[3:].strip()
    elif expr.startswith("elif "):
        expr = expr[5:].strip()

    expanded = _substitute_config_attrs(_expand_config_calls(expr, config), config)
    decided = _decide_condition(expanded, layer_idx)
    if decided is not None:
        return decided
    expanded = _simplify_boolean_expr(expanded)
    if not expanded:
        return False
    normalized = re.sub(r"\s+", "", expanded)
    if normalized.lower() == "true":
        return True
    if normalized.lower() == "false":
        return False

    match = re.fullmatch(
        r"\(layer_idx\+1\)in(\[[^\]]+\]|\w+_layers(?:\(\d+layers\))?)",
        normalized,
    )
    if match:
        payload = match.group(1)
        if "_layers" in payload and not payload.startswith("["):
            stem = payload.split("_layers", 1)[0]
            one_based = {idx + 1 for idx in _layer_index_list(config, stem, zero_based=False)}
            return (layer_idx + 1) in one_based
        indices = _parse_layer_index_set(payload)
        return (layer_idx + 1) in indices

    match = re.fullmatch(r"layer_idx>=(\d+)", normalized)
    if match:
        return layer_idx >= int(match.group(1))

    match = re.fullmatch(r"layer_idx%(\d+)==0", normalized)
    if match:
        return layer_idx % int(match.group(1)) == 0

    match = re.fullmatch(r"\(layer_idx%(\d+)==0\)", normalized)
    if match:
        return layer_idx % int(match.group(1)) == 0

    return False


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class _UndecidableCondition(Exception):
    """The condition needs more than ``layer_idx`` and config literals to decide."""


def _decide_condition(expr: str, layer_idx: int) -> bool | None:
    """Decide a condition whose config references are already literals.

    Returns None when something in the expression stayed symbolic, which leaves the
    caller's pattern matching to handle the shapes config substitution cannot reach.
    The expression is walked rather than executed: modeling source is only ever read.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return None
    try:
        return bool(_condition_value(tree.body, layer_idx))
    except (_UndecidableCondition, ArithmeticError, TypeError, ValueError):
        return None


def _condition_value(node: ast.AST, layer_idx: int) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id != "layer_idx":
            raise _UndecidableCondition(node.id)
        return layer_idx
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_condition_value(element, layer_idx) for element in node.elts]
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not _condition_value(node.operand, layer_idx)
        if isinstance(node.op, ast.USub):
            return -_condition_value(node.operand, layer_idx)
        raise _UndecidableCondition(type(node.op).__name__)
    if isinstance(node, ast.BoolOp):
        values = (_condition_value(value, layer_idx) for value in node.values)
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.BinOp):
        apply = _BINARY_OPS.get(type(node.op))
        if apply is None:
            raise _UndecidableCondition(type(node.op).__name__)
        return apply(
            _condition_value(node.left, layer_idx),
            _condition_value(node.right, layer_idx),
        )
    if isinstance(node, ast.Compare):
        left = _condition_value(node.left, layer_idx)
        for op_node, comparator in zip(node.ops, node.comparators):
            right = _condition_value(comparator, layer_idx)
            if isinstance(op_node, ast.In):
                matched = left in right
            elif isinstance(op_node, ast.NotIn):
                matched = left not in right
            else:
                compare = _COMPARE_OPS.get(type(op_node))
                if compare is None:
                    raise _UndecidableCondition(type(op_node).__name__)
                matched = compare(left, right)
            if not matched:
                return False
            left = right
        return True
    raise _UndecidableCondition(type(node).__name__)


def _simplify_line(line: str, config: dict[str, Any]) -> str:
    match = _BRANCH_LINE_RE.match(line)
    if match:
        prefix = match.group("prefix")
        branch = match.group("branch")
        expr = match.group("expr")
        if branch == "else":
            return f"{prefix} (else)"
        simplified = _simplify_condition(expr or "", config)
        if branch == "elif" and simplified.lower() == "true":
            return f"{prefix} (else)"
        return f"{prefix} ({branch} {simplified})"

    loop_match = _LOOP_LINE_RE.match(line)
    if loop_match:
        head = loop_match.group("head")
        class_name = loop_match.group("class_name").strip()
        loop_var = loop_match.group("loop_var")
        range_expr = _substitute_config_attrs(loop_match.group("range_expr"), config)
        range_expr = _simplify_boolean_expr(range_expr)
        return f"{head} × {class_name} ({loop_var} in {range_expr})"

    return line


def _simplify_condition(expr: str, config: dict[str, Any]) -> str:
    expr = _expand_config_calls(expr, config)
    expr = _substitute_config_attrs(expr, config)
    expr = _simplify_boolean_expr(expr)
    return expr.strip()


def _expand_config_calls(expr: str, config: dict[str, Any]) -> str:
    expr = re.sub(
        r"config\.is_(\w+)_layer\(\s*layer_idx\s*\)",
        lambda match: _layer_index_predicate(config, match.group(1)),
        expr,
    )
    expr = re.sub(
        r"config\.is_(\w+)\b",
        lambda match: _config_is_named_flag(config, match.group(1)),
        expr,
    )
    expr = re.sub(
        r"getattr\(\s*config\s*,\s*['\"](\w+)['\"]\s*,\s*([^)]+?)\s*\)",
        lambda match: _format_literal(
            _config_get(config, match.group(1), _parse_default(match.group(2)))
        ),
        expr,
    )
    return expr


def _substitute_config_attrs(expr: str, config: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in config:
            return match.group(0)
        return _format_literal(config[key])

    return re.sub(r"config\.(\w+)", repl, expr)


def _config_get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in config:
        return config[key]
    for value in config.values():
        if isinstance(value, dict) and key in value:
            return value[key]
    return default


def _config_containers(config: dict[str, Any]) -> list[dict[str, Any]]:
    containers = [config]
    for value in config.values():
        if isinstance(value, dict):
            containers.append(value)
    return containers


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _layer_index_list(
    config: dict[str, Any],
    stem: str,
    *,
    zero_based: bool = False,
) -> list[int]:
    """Return layer indices listed under ``{stem}_layers`` anywhere in the config tree."""
    keys = (f"{stem}_layers", f"{stem}_layer_indices", f"{stem}_layer_ids")
    for container in _config_containers(config):
        for key in keys:
            raw = container.get(key)
            if not isinstance(raw, list):
                continue
            values: list[int] = []
            for item in raw:
                idx = _as_int(item)
                if idx is None:
                    continue
                if not zero_based and idx > 0:
                    idx -= 1
                values.append(idx)
            if values:
                return sorted(set(values))
    return []


def _layer_index_predicate(config: dict[str, Any], stem: str) -> str:
    layers = _layer_index_list(config, stem, zero_based=False)
    if not layers:
        return f"(layer_idx + 1) in {stem}_layers"
    display_layers = [idx + 1 for idx in layers]
    if len(display_layers) <= 12:
        return f"(layer_idx + 1) in {_format_layer_index_set(display_layers)}"
    return f"(layer_idx + 1) in {stem}_layers ({len(display_layers)} layers)"


def _parse_layer_index_set(payload: str) -> set[int]:
    """Parse ``[1, 2]`` or ``[1–5]`` style index lists (1-based)."""
    payload = payload.strip()
    if not payload.startswith("[") or not payload.endswith("]"):
        return set()
    inner = payload[1:-1].strip()
    if not inner:
        return set()
    values: set[int] = set()
    for part in inner.split(","):
        token = part.strip()
        if not token:
            continue
        if "–" in token or "-" in token:
            sep = "–" if "–" in token else "-"
            start_text, end_text = token.split(sep, 1)
            start = _as_int(start_text)
            end = _as_int(end_text)
            if start is None or end is None:
                continue
            values.update(range(start, end + 1))
            continue
        idx = _as_int(token)
        if idx is not None:
            values.add(idx)
    return values


def _parse_default(raw: str) -> Any:
    raw = raw.strip()
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def _config_is_mla(config: dict[str, Any]) -> bool:
    return any(
        [
            config.get("q_lora_rank") is not None,
            config.get("kv_lora_rank") is not None,
            config.get("qk_nope_head_dim") is not None,
            config.get("qk_rope_head_dim") is not None,
            config.get("v_head_dim") is not None,
            config.get("mla_use_nope") is True,
        ]
    )


def _config_is_moe(config: dict[str, Any]) -> bool:
    num_experts = config.get("num_experts")
    return num_experts is not None


def _config_is_named_flag(config: dict[str, Any], name: str) -> str:
    lowered = name.lower()
    if lowered == "mla":
        return "True" if _config_is_mla(config) else "False"
    if lowered == "moe":
        return "True" if _config_is_moe(config) else "False"

    for container in _config_containers(config):
        for key in (name, f"is_{name}", f"use_{name}", f"{name}_enabled"):
            if key not in container:
                continue
            value = container[key]
            if isinstance(value, bool):
                return "True" if value else "False"
    return "False"


def _format_layer_index_set(values: list[int]) -> str:
    ordered = sorted(set(values))
    compact = _compact_ranges(ordered)
    if len(ordered) <= 12:
        return f"[{compact}]"
    return f"[{compact}, … ({len(ordered)} layers)]"


def _compact_ranges(values: list[int]) -> str:
    if not values:
        return ""
    parts: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        parts.append(f"{start}–{prev}" if start != prev else str(start))
        start = prev = value
    parts.append(f"{start}–{prev}" if start != prev else str(start))
    return ", ".join(parts)


def _format_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value)
    return str(value)


def _simplify_boolean_expr(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return expr

    while True:
        previous = expr
        expr = re.sub(r"\bTrue\s+and\s+", "", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\s+and\s+True\b", "", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\b(\d+)\s+is\s+not\s+None\b", "", expr)
        expr = re.sub(r"\bNone\s+is\s+not\s+None\b", "", expr)
        expr = re.sub(r"\(?\s*layer_idx\s*%\s*1\s*==\s*0\s*\)?", "", expr)
        expr = re.sub(r"\(\s*\)", "", expr)
        expr = re.sub(r"\s+and\s+and\s+", " and ", expr)
        expr = re.sub(r"^\s*and\s+", "", expr)
        expr = re.sub(r"\s+and\s*$", "", expr)
        expr = re.sub(r"\s{2,}", " ", expr)
        expr = expr.strip()
        if expr == previous:
            break

    return expr.strip()
