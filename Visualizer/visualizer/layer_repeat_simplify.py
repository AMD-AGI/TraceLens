"""Substitute config values and simplify layer-repeat fact sheet expressions."""

from __future__ import annotations

import ast
import json
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
        r"config\.is_kda_layer\(\s*layer_idx\s*\)",
        lambda _match: _kda_layer_predicate(config),
        expr,
    )
    expr = re.sub(
        r"config\.is_mla\b",
        lambda _match: "True" if _config_is_mla(config) else "False",
        expr,
    )
    expr = re.sub(
        r"config\.is_moe\b",
        lambda _match: "True" if _config_is_moe(config) else "False",
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
    nested = config.get("linear_attn_config")
    if isinstance(nested, dict) and key in nested:
        return nested[key]
    return default


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


def _kda_layer_indices(config: dict[str, Any]) -> list[int]:
    linear_cfg = config.get("linear_attn_config")
    if not isinstance(linear_cfg, dict):
        return []
    raw_layers = linear_cfg.get("kda_layers")
    if not isinstance(raw_layers, list):
        return []
    return [int(layer) for layer in raw_layers if isinstance(layer, int) or str(layer).isdigit()]


def _kda_layer_predicate(config: dict[str, Any]) -> str:
    layers = _kda_layer_indices(config)
    if not layers:
        return "(layer_idx + 1) in kda_layers"
    if len(layers) <= 12:
        formatted = _format_layer_index_set(layers)
        return f"(layer_idx + 1) in {formatted}"
    return f"(layer_idx + 1) in kda_layers ({len(layers)} layers)"


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
