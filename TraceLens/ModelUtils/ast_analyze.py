###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inspect Hugging Face modeling code via Python AST (CPU-only)."""

from __future__ import annotations

import ast
import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from TraceLens.ModelUtils.blocks import BlockComponent, CodeAnalysis

DECODER_CLASS_RE = re.compile(
    r"(DecoderLayer|DecoderBlock|TransformerBlock|ModelBlock|Block)$",
    re.IGNORECASE,
)
MODEL_CLASS_RE = re.compile(r"(ForCausalLM|Model|PreTrainedModel)$", re.IGNORECASE)

ATTENTION_CLASS_RE = re.compile(r"(Attention|Attn|MLA|LatentAttention)", re.IGNORECASE)
MOE_CLASS_RE = re.compile(r"(MoE|Moe|Expert|SparseMoe|SharedExpert)", re.IGNORECASE)
FFN_CLASS_RE = re.compile(r"(MLP|Mlp|FeedForward|FFN|SwiGLU|GatedMLP)", re.IGNORECASE)
NORM_CLASS_RE = re.compile(r"(RMSNorm|LayerNorm|Norm)", re.IGNORECASE)

ATTR_ROLE_HINTS: dict[str, str] = {
    "embed_tokens": "embedding",
    "word_embeddings": "embedding",
    "wte": "embedding",
    "lm_head": "head",
    "output": "head",
    "embed_out": "head",
    "rotary_emb": "positional",
    "rotary_pos_emb": "positional",
    "rotary_embedding": "positional",
    "self_attn": "attention",
    "self_attention": "attention",
    "attn": "attention",
    "attention": "attention",
    "mlp": "ffn",
    "feed_forward": "ffn",
    "ffn": "ffn",
    "block_sparse_moe": "moe",
    "moe": "moe",
    "experts": "moe",
    "router": "router",
    "gate": "router",
    "input_layernorm": "norm",
    "post_attention_layernorm": "norm",
    "pre_feedforward_layernorm": "norm",
    "post_feedforward_layernorm": "norm",
    "post_norm": "norm",
    "pre_norm": "norm",
    "norm": "norm",
}


SYNTHETIC_ATTENTION = "@attention"
FUNCTIONAL_SYNTHETIC_PREFIX = "@functional_"
SYNTHETIC_FUNCTIONAL_LINEAR = f"{FUNCTIONAL_SYNTHETIC_PREFIX}linear"
POSITIONAL_SYNTHETIC_PREFIX = "@positional_"
_POSITIONAL_SOURCE_POS_RE = re.compile(
    rf"^{re.escape(POSITIONAL_SYNTHETIC_PREFIX)}l(\d+)_"
)


def positional_synthetic_attr(func_name: str, lineno: int) -> str:
    """Synthetic attr for a rope helper called as a plain function in a forward.

    The line number keeps each application site distinct, since a forward commonly
    rotates queries and keys with separate calls to the same helper.
    """
    return f"{POSITIONAL_SYNTHETIC_PREFIX}l{lineno}_{func_name}"


def is_positional_synthetic(attr_name: str) -> bool:
    return attr_name.startswith(POSITIONAL_SYNTHETIC_PREFIX)


def positional_synthetic_source_pos(attr_name: str) -> tuple[int, int] | None:
    """Source position of a traced rope call, for ordering it among sibling steps."""
    match = _POSITIONAL_SOURCE_POS_RE.match(attr_name)
    if match is None:
        return None
    return int(match.group(1)), 0


def positional_display_label(attr_name_or_func: str) -> str:
    """Display label for a traced rope function (apply_rotary_emb -> Apply rotary emb)."""
    name = attr_name_or_func
    if name.startswith(POSITIONAL_SYNTHETIC_PREFIX):
        name = _POSITIONAL_SOURCE_POS_RE.sub("", name)
    text = name.replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else name


def functional_synthetic_attr(op_name: str) -> str:
    """Synthetic attr for a torch.nn.functional call (e.g. linear -> @functional_linear)."""
    return f"{FUNCTIONAL_SYNTHETIC_PREFIX}{op_name}"


def is_functional_synthetic(attr_name: str) -> bool:
    return attr_name.startswith(FUNCTIONAL_SYNTHETIC_PREFIX)


def functional_display_label(op_name_or_attr: str) -> str:
    """Display label for a functional op (e.g. linear -> Linear, @functional_softmax -> Softmax)."""
    name = op_name_or_attr
    if name.startswith(FUNCTIONAL_SYNTHETIC_PREFIX):
        name = name[len(FUNCTIONAL_SYNTHETIC_PREFIX) :]
    return "".join(part.capitalize() for part in name.split("_") if part)


def first_functional_synthetic_index(forward_calls: list[str]) -> int | None:
    for index, call in enumerate(forward_calls):
        if is_functional_synthetic(call):
            return index
    return None


SYNTHETIC_GATE_ACTIVATION = "@gate_activation"
SYNTHETIC_GATE_RESHAPE = "@gate_reshape"
_GATE_ACTIVATION_NAMES = {
    "sigmoid": "Sigmoid",
    "softmax": "Softmax",
    "tanh": "Tanh",
}
# Modeling code binds its activation from a registry keyed by config
# (`self.act_fn = ACT2FN[config.hidden_act]`) instead of constructing it, so the
# activation the checkpoint actually runs is only knowable from the config.
_ACTIVATION_REGISTRY_NAMES = frozenset(
    {"ACT2FN", "ACT2CLS", "ACT_FN", "ACTIVATION_REGISTRY"}
)
_ACTIVATION_DISPLAY_NAMES = {
    "silu": "SiLU",
    "swish": "SiLU",
    "gelu": "GELU",
    "gelu_new": "GELU",
    "gelu_pytorch_tanh": "GELU",
    "quick_gelu": "GELU",
    "relu": "ReLU",
    "relu6": "ReLU6",
    "sigmoid": "Sigmoid",
    "tanh": "Tanh",
    "mish": "Mish",
    "elu": "ELU",
    "selu": "SELU",
    "leaky_relu": "LeakyReLU",
    "prelu": "PReLU",
    "hardswish": "Hardswish",
    "hardsigmoid": "Hardsigmoid",
    "identity": "Identity",
    "linear": "Identity",
}
_ACTIVATION_LEAF_CLASS_NAMES = frozenset(_ACTIVATION_DISPLAY_NAMES.values())
FORWARD_OPERATION_PREFIX = "@op_"
# Stands for the value a helper method receives, so operations reading its parameter
# resolve to whatever feeds the chain the method is inlined into.
FORWARD_METHOD_INPUT = "@method_input"
_SYNTHETIC_ATTENTION_NAMES = {
    "eager_attention_forward",
    "flash_attention_forward",
    "sdpa_attention_forward",
    "attention_interface",
}
# Locals a forward assigns an attention implementation to before calling it, so the
# call site names the variable rather than the kernel that actually runs.
_ATTENTION_DISPATCH_NAMES = {
    "attention_interface",
    "attention_fn",
    "attn_interface",
    "all_attention_functions",
}
_KERNEL_MERGE_NAME_RE = re.compile(
    r"(attention|attn|recurrent|flash|sdpa|linear_attn|kernel|chunk)",
    re.IGNORECASE,
)
_SKIP_INIT_CLASS_NAMES = frozenset({"Parameter", "Buffer", "getattr"})
_SKIP_INIT_FORWARD_ATTRS = frozenset(
    {
        "config",
        "layer_idx",
        "layer_id",
        "layers",
        "layer",
        "module",
        "modules",
        "training",
        "gradient_checkpointing",
        "gradient_checkpointing_func",
        "device",
        "dtype",
    }
)
_SKIP_INIT_FORWARD_CLASS_NAMES = frozenset({"ModuleList", "Sequential", "ModuleDict"})


def _append_forward_call(calls: list[str], attr: str) -> None:
    if calls and calls[-1] == attr:
        return
    calls.append(attr)


def _is_positional_function_call(func: ast.AST, target: str) -> bool:
    """True for a bare call to a rope helper such as `apply_rotary_emb(q, freqs)`."""
    if not isinstance(func, ast.Name):
        return False
    return bool(POSITIONAL_ATTR_RE.search(target))


def _is_literal_true(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _call_undoes_rotation(node: ast.Call) -> bool:
    """True when a rope call unrotates its input rather than rotating it."""
    for keyword in node.keywords:
        if keyword.arg == "inverse":
            return _is_literal_true(keyword.value)
    return any(_is_literal_true(arg) for arg in node.args[2:])


def _positional_helper_functions(tree: ast.AST) -> list[str]:
    """Module-level rope helpers defined in the source, e.g. `apply_rotary_emb`.

    Their presence shows the architecture rotates positions even when the analyzer
    cannot place the call site.
    """
    names: list[str] = []
    for node in getattr(tree, "body", []):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if POSITIONAL_ATTR_RE.search(node.name):
            names.append(node.name)
    return names


def _positional_step_details(func: ast.FunctionDef) -> dict[str, list[str]]:
    """Detail lines for traced rope calls, so an inverse rotation reads differently."""
    details: dict[str, list[str]] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if not _is_positional_function_call(node.func, node.func.id):
            continue
        if _call_undoes_rotation(node):
            details[positional_synthetic_attr(node.func.id, node.lineno)] = [
                "inverse rotation"
            ]
    return details


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
        "chunk",
        "concat",
        "where",
        "masked_fill",
        "softmax",
        "dropout",
        "clone",
        "detach",
        "to",
        "expand",
        "repeat",
        "roll",
        "triu",
        "tril",
        "matmul",
        "bmm",
        "einsum",
    }
    | _METHOD_CHAIN_OPS
)


def _assign_target(stmt: ast.AST) -> str | None:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def _if_has_competing_assigns(if_node: ast.If) -> bool:
    if_targets = {
        _assign_target(stmt)
        for branch_stmt in if_node.body
        for stmt in ast.walk(branch_stmt)
        if isinstance(stmt, ast.Assign)
    }
    if_targets.discard(None)
    for branch_stmt in if_node.orelse:
        for stmt in ast.walk(branch_stmt):
            if not isinstance(stmt, ast.Assign):
                continue
            target = _assign_target(stmt)
            if target is not None and target in if_targets:
                return True
    return False


def _alternate_forward_dispatches(func: ast.FunctionDef) -> set[str]:
    """Private helpers invoked only via early-return branches (alternate forward paths)."""
    dispatches: set[str] = set()
    for node in func.body:
        if not isinstance(node, ast.If) or node.orelse:
            continue
        return_calls: list[str] = []
        for stmt in node.body:
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                _extract_self_calls_ordered(stmt.value, return_calls)
        if len(return_calls) != 1:
            continue
        call = return_calls[0]
        if call.startswith("_"):
            dispatches.add(call)
    return dispatches


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
            # Rope helpers live at module level, so the block that applies them is
            # the only place the diagram can show the rotation happening.
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


def _resolve_local_module_alias_calls(func: ast.FunctionDef) -> ast.FunctionDef:
    """Rewrite ``expert(...)`` aliases of ``self.experts[i]`` as module calls.

    Expert loops commonly bind one entry from a ModuleList to a local variable before
    invoking it.  Resolving that alias lets the normal forward parser retain the real
    routed-expert branch instead of silently dropping it.
    """
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
    """True for F.linear(...) and torch.nn.functional.linear(...)."""
    return _functional_call_name(func) == "linear"


def _is_moe_gate_class(class_name: str, forward_calls: list[str]) -> bool:
    if first_functional_synthetic_index(forward_calls) is None:
        return False
    if re.search(r"(?:Gate|Router)$", class_name):
        return True
    return bool(
        MOE_CLASS_RE.search(class_name) and re.search(r"gate|router", class_name, re.I)
    )


def _stmt_value(stmt: ast.AST) -> ast.AST | None:
    if isinstance(stmt, ast.Assign):
        return stmt.value
    if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        return stmt.value
    if isinstance(stmt, ast.Return) and stmt.value is not None:
        return stmt.value
    return None


COMBINE_DETAIL_PREFIX = "combine:"
MOE_AGGREGATION_LABEL = "MoE aggregation"


def combine_op_from_step_details(details: list[str] | None) -> str | None:
    """Return a combine-operator symbol recorded by AST analysis (e.g. Σ)."""
    if not details:
        return None
    prefix = f"{COMBINE_DETAIL_PREFIX} "
    for item in details:
        if item.startswith(prefix):
            symbol = item[len(prefix) :].strip()
            if symbol:
                return symbol
    return None


def _subexpr_has_multiplication(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if sub.func.attr in {"mul", "mul_", "multiply"}:
                return True
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, (ast.Mult, ast.MatMult)):
            return True
    return False


def _expr_is_weighted_sum(node: ast.AST) -> bool:
    """True when an expression reduces a weighted tensor via sum()."""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Attribute):
            continue
        if sub.func.attr != "sum":
            continue
        if _subexpr_has_multiplication(sub.func.value):
            return True
    return False


def _detect_method_combine_op(
    func: ast.FunctionDef, *, class_name: str = ""
) -> str | None:
    """Infer a combine-operator symbol from a helper method body."""
    weighted = False
    for node in ast.walk(func):
        value: ast.AST | None = None
        if isinstance(node, ast.Return):
            value = node.value
        elif isinstance(node, ast.Assign):
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            value = node.value
        if value is not None and _expr_is_weighted_sum(value):
            weighted = True
            break
    moe_like = bool(re.search(r"(?i)moe", func.name) or MOE_CLASS_RE.search(class_name))
    if moe_like and (
        weighted or re.search(r"(?i)(?:infer|combin|aggregat)", func.name)
    ):
        return MOE_AGGREGATION_LABEL
    if weighted:
        return "Σ"
    return None


def _method_forward_step_details(
    class_node: ast.ClassDef,
    forward_calls: list[str],
    init_assignments: dict[str, str],
) -> dict[str, list[str]]:
    """Attach AST-derived metadata to forward helper methods."""
    method_funcs = {
        item.name: item for item in class_node.body if isinstance(item, ast.FunctionDef)
    }
    details: dict[str, list[str]] = {}
    for call_attr in forward_calls:
        if call_attr in init_assignments:
            continue
        if call_attr.startswith("@") or call_attr == SYNTHETIC_ATTENTION:
            continue
        func = method_funcs.get(call_attr)
        if func is None:
            continue
        combine_op = _detect_method_combine_op(func, class_name=class_node.name)
        if combine_op is None:
            continue
        details[call_attr] = [
            f"method `{call_attr}()`",
            f"{COMBINE_DETAIL_PREFIX} {combine_op}",
        ]
    return details


def _single_op_forward_methods(
    class_node: ast.ClassDef,
    forward_calls: list[str],
    init_assignments: dict[str, str],
    *,
    self_values: dict[str, Any],
    all_tensor_ops: bool,
) -> dict[str, ForwardOperation]:
    """Forward helper methods whose body is one primitive op, keyed by method name.

    Such a method has no internals worth a frame of its own, so callers render the op
    it performs instead of an opaque tile named after the method.
    """
    method_funcs = {
        item.name: item for item in class_node.body if isinstance(item, ast.FunctionDef)
    }
    single: dict[str, ForwardOperation] = {}
    for call_attr in forward_calls:
        if (
            call_attr in init_assignments
            or call_attr.startswith("@")
            or call_attr == SYNTHETIC_ATTENTION
        ):
            continue
        func = method_funcs.get(call_attr)
        if func is None:
            continue
        # Combine-op methods drive side-input merge rendering, so leave them named.
        if _detect_method_combine_op(func, class_name=class_node.name) is not None:
            continue
        operations = _forward_operations_from_forward(
            func,
            self_values=self_values,
            all_tensor_ops=all_tensor_ops,
        )
        if len(operations.operations) == 1:
            single[call_attr] = operations.operations[0]
    return single


def _multi_op_forward_methods(
    class_node: ast.ClassDef,
    forward_calls: list[str],
    init_assignments: dict[str, str],
    *,
    self_values: dict[str, Any],
    all_tensor_ops: bool,
) -> dict[str, list[ForwardOperation]]:
    """Forward helper methods with enough tensor operations to expand as a subgraph."""
    method_funcs = {
        item.name: item for item in class_node.body if isinstance(item, ast.FunctionDef)
    }
    expanded: dict[str, list[ForwardOperation]] = {}
    for call_attr in forward_calls:
        if (
            call_attr in init_assignments
            or call_attr.startswith("@")
            or call_attr == SYNTHETIC_ATTENTION
        ):
            continue
        func = method_funcs.get(call_attr)
        if func is None:
            continue
        # Combine helpers (for example Kimi's moe_infer) are represented by their
        # semantic aggregation node and side inputs, not flattened tensor ops.
        if _detect_method_combine_op(func, class_name=class_node.name) is not None:
            continue
        operations = _forward_operations_from_forward(
            func,
            self_values=self_values,
            all_tensor_ops=all_tensor_ops,
        )
        if len(operations.operations) > 1:
            expanded[call_attr] = operations.operations
    return expanded


def _register_forward_calls(
    stmt_calls: list[str],
    calls: list[str],
    norm_before: list[str],
    pending_norm: str | None,
) -> str | None:
    for attr in stmt_calls:
        if attr == SYNTHETIC_ATTENTION:
            _append_forward_call(calls, attr)
            pending_norm = None
            continue

        role = _classify_role(attr, "")
        if role == "norm":
            _append_forward_call(calls, attr)
            pending_norm = attr
            continue

        if pending_norm is not None:
            norm_before.append(attr)
            pending_norm = None
        _append_forward_call(calls, attr)
    return pending_norm


def parse_python_ast(source: str, filename: str = "<model>") -> ast.Module:
    return ast.parse(source, filename=filename)


def dump_ast(source: str, filename: str = "<model>") -> str:
    tree = parse_python_ast(source, filename=filename)
    return ast.dump(tree, indent=2, include_attributes=False)


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


def _call_class_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_self_attr(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == attr
    )


POSITIONAL_ATTR_RE = re.compile(r"(rotary|rope|pos_emb)", re.I)
POSITIONAL_CLASS_RE = re.compile(r"(Rotary|RoPE|PosEmb|RotaryEmbedding)", re.I)
EMBEDDING_CLASS_RE = re.compile(r"Embedding", re.I)
# `Head$` catches inference-repo names like ParallelHead. Attention classes are
# classified earlier, so `AttentionHead` and friends never reach this test.
HEAD_CLASS_RE = re.compile(r"(?:^|_)head$|Head$|LMHead|CausalLMOutput", re.I)
_MOE_BLOCK_CLASS_RE = re.compile(
    r"(?i)(?:^moe$|sparse_?moe(?:_?block)?$|moe_?block$|experts$)"
)


def _classify_role(attr_name: str, class_name: str) -> str:
    attr_key = attr_name.lower()
    if _MOE_BLOCK_CLASS_RE.search(class_name):
        return "moe"
    if attr_key in ATTR_ROLE_HINTS:
        return ATTR_ROLE_HINTS[attr_key]
    tokens = [token for token in re.split(r"[_\W]+", attr_key) if token]
    # `attn_norm` / `ffn_norm` must be norms. Matching attn/ffn first left the
    # transformer overview with no (norm, module) pairs, so the block drew empty.
    if any("norm" in token for token in tokens):
        return "norm"
    for token in tokens:
        if token in {"attn", "attention"}:
            return "attention"
        if token in {"mlp", "ffn"}:
            return "ffn"
        if token in {"moe", "experts"}:
            return "moe"
        if token in {"router"}:
            return "router"
        if token in {"embed", "embedding"}:
            return "embedding"
    if attr_key in {"gate", "router"}:
        return "router"
    if POSITIONAL_ATTR_RE.search(attr_key):
        return "positional"

    if ATTENTION_CLASS_RE.search(class_name):
        return "attention"
    if MOE_CLASS_RE.search(class_name):
        return "moe"
    if FFN_CLASS_RE.search(class_name):
        return "ffn"
    if NORM_CLASS_RE.search(class_name):
        return "norm"
    if POSITIONAL_CLASS_RE.search(class_name):
        return "positional"
    if EMBEDDING_CLASS_RE.search(class_name) and "embed" in attr_key:
        return "embedding"
    if HEAD_CLASS_RE.search(class_name) or (
        re.match(r"(?i)^Linear$", class_name) and "head" in attr_key
    ):
        return "head"
    return "other"


def ffn_role_for_class(attr_name: str, class_name: str) -> str:
    """Tell an MoE block from a dense FFN when one attribute can hold either.

    ``self.mlp`` is bound to a sparse block as often as a dense one, so the attribute
    name cannot decide the role; a class named for the routed block does.
    """
    if _MOE_BLOCK_CLASS_RE.search(class_name):
        return "moe"
    if FFN_CLASS_RE.search(class_name):
        return "ffn"
    return _classify_role(attr_name, class_name)


def displays_as_linear(attr_name: str, class_name: str | None) -> bool:
    """True when a module should be drawn as a plain Linear op."""
    return bool(class_name and re.match(r"(?i)^Linear$", class_name))


def displays_as_pointwise_leaf(attr_name: str, class_name: str | None) -> bool:
    """True when a submodule is a leaf the parent's own tensor math flows through."""
    if displays_as_linear(attr_name, class_name):
        return True
    return bool(class_name) and class_name in _ACTIVATION_LEAF_CLASS_NAMES


def _forward_owns_tensor_math(
    forward_calls: list[str],
    init_assignments: dict[str, str],
) -> bool:
    """True when a module's forward does its own tensor math over plain projections.

    An MLP-style module computes its gating inline, so it has no submodule to carry
    that math: dropping the operations would leave the diagram showing its
    projections with nothing between them. A registered activation is pointwise and
    carries no gating either, so it counts as a projection here. Modules that call
    composite children (norms, attention, another MLP) leave the math to those
    children, and their own statements are residual plumbing represented elsewhere.
    """
    module_calls = [call for call in forward_calls if call in init_assignments]
    if not module_calls:
        return False
    return all(
        displays_as_pointwise_leaf(call, init_assignments[call])
        for call in module_calls
    )


def _forward_delegates_to_nothing(class_name: str, forward_calls: list[str]) -> bool:
    """True when a module computes everything in its own statements.

    Rotary embeddings, normalization layers, activations, and small collapse heads
    commonly own no child modules at all. Every computation they perform therefore
    lives inline; retaining those operations is the only way to render their real
    dataflow instead of an opaque class-name tile.
    """
    del class_name
    return not forward_calls


def _forward_mixes_modules_and_inline_ops(
    forward_calls: list[str],
    init_assignments: dict[str, str],
    parsed_operations: list[ForwardOperation],
) -> bool:
    """True when forward() calls submodules and also runs inline tensor math."""
    module_calls = [
        call
        for call in forward_calls
        if call in init_assignments
        or is_positional_synthetic(call)
        or is_functional_synthetic(call)
    ]
    if not module_calls or not parsed_operations:
        return False
    inline_ops = sum(
        1 for op in parsed_operations if is_forward_operation(op.attr_name)
    )
    return inline_ops >= 2


def _label_for(role: str, class_name: str, attr_name: str) -> str:
    if role == "embedding":
        if attr_name == "embed_tokens":
            return "Token Embedding"
        return class_name if len(class_name) <= 24 else attr_name
    if role == "head":
        if displays_as_linear(attr_name, class_name):
            return "Linear"
        return class_name if len(class_name) <= 24 else attr_name
    if role == "positional":
        if class_name in {"RotaryEmbedding", "RotaryEmbeddingModule"}:
            return class_name.replace("Embedding", " encoding").strip()
        return class_name if len(class_name) <= 24 else attr_name
    if displays_as_linear(attr_name, class_name):
        return "Linear"
    if role == "attention":
        if re.search(r"Gated", class_name, re.I):
            return "Gated Attention"
        if re.search(r"Sliding|Window", class_name, re.I):
            return "Sliding Window Attn"
        return class_name.replace("Attention", " Attn").strip()
    if role == "moe":
        return class_name if len(class_name) <= 22 else "MoE block"
    if role == "ffn":
        if "SwiGLU" in class_name or "Gated" in class_name:
            return "SwiGLU FFN"
        return class_name if len(class_name) <= 22 else "FFN"
    if role == "norm":
        if "RMS" in class_name:
            return "RMSNorm"
        if "Layer" in class_name:
            return "LayerNorm"
        return "Norm"
    if role == "router":
        return "Router"
    return class_name if len(class_name) <= 24 else attr_name


SideInputSource = Literal["forward_input", "prior_step"]


@dataclass
class SideInputSpec:
    """Extra argument feeding a forward call from an earlier step or the block input."""

    arg_name: str
    port_label: str
    source_chain: list[str]
    source_kind: SideInputSource = "prior_step"
    side_effect_call: bool = False


@dataclass(frozen=True)
class ForwardOperation:
    """One primitive tensor operation recovered from a forward expression."""

    attr_name: str
    label: str
    class_name: str
    predecessors: tuple[str, ...] = ()
    external_inputs: tuple[str, ...] = ()
    details: tuple[str, ...] = ()
    param_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoopCarriedSpec:
    """One value updated by a loop and consumed after its final iteration."""

    loop_id: str
    iteration_count: int | None
    variable: str
    initial_producer: str
    updated_producer: str
    operation_ids: tuple[str, ...]


@dataclass
class ForwardAnalysis:
    """Inline tensor ops recovered from one ``forward()`` plus return metadata."""

    operations: list[ForwardOperation]
    var_producer: dict[str, str]
    step_predecessors: dict[str, tuple[str, ...]]
    step_predecessor_args: dict[str, dict[str, str]]
    return_slots: dict[str, str]
    return_order: list[str]
    primary_return_slot: str | None
    loop_carried: list[LoopCarriedSpec]


@dataclass(frozen=True)
class StackEntryDataflow:
    """Source operations that transform embeddings into decoder-loop input."""

    operations: tuple[ForwardOperation, ...]
    output_producer: str


@dataclass
class ClassStructure:
    name: str
    node: ast.ClassDef
    init_assignments: dict[str, str]
    init_details: dict[str, list[str]]
    forward_calls: list[str]
    norm_before: list[str]
    attention_inputs: dict[str, list[str]] = field(default_factory=dict)
    parallel_gates: list[str] = field(default_factory=list)
    input_fed_calls: list[str] = field(default_factory=list)
    gate_activations: dict[str, str] = field(default_factory=dict)
    forward_step_details: dict[str, list[str]] = field(default_factory=dict)
    side_inputs: dict[str, list[SideInputSpec]] = field(default_factory=dict)
    init_assignment_options: dict[str, list[str]] = field(default_factory=dict)
    forward_input_name: str | None = None
    forward_operations: dict[str, ForwardOperation] = field(default_factory=dict)
    forward_step_predecessors: dict[str, tuple[str, ...]] = field(default_factory=dict)
    forward_step_predecessor_args: dict[str, dict[str, str]] = field(
        default_factory=dict
    )
    single_op_methods: dict[str, ForwardOperation] = field(default_factory=dict)
    multi_op_methods: dict[str, list[ForwardOperation]] = field(default_factory=dict)
    forward_return_slots: dict[str, str] = field(default_factory=dict)
    forward_return_order: list[str] = field(default_factory=list)
    primary_return_slot: str | None = None
    forward_call_output_names: dict[str, str] = field(default_factory=dict)
    referenced_return_producers: set[str] = field(default_factory=set)
    loop_carried: list[LoopCarriedSpec] = field(default_factory=list)
    forward_param_inputs: list[str] = field(default_factory=list)
    dataflow_expanded: bool = False


def stack_entry_dataflow(cls: ClassStructure) -> StackEntryDataflow | None:
    """Recover the exact tensor-method chain feeding an iterated decoder module."""
    forward = next(
        (
            item
            for item in cls.node.body
            if isinstance(item, ast.FunctionDef) and item.name == "forward"
        ),
        None,
    )
    if forward is None:
        return None

    decoder_loop: ast.For | None = None
    input_name: str | None = None
    for statement in forward.body:
        if not isinstance(statement, ast.For):
            continue
        loop_names = {
            node.id for node in ast.walk(statement.target) if isinstance(node, ast.Name)
        }
        loop_call = next(
            (
                call
                for call in ast.walk(statement)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in loop_names
                and call.args
                and isinstance(call.args[0], ast.Name)
            ),
            None,
        )
        if loop_call is not None:
            decoder_loop = statement
            input_name = loop_call.args[0].id
            break
    if decoder_loop is None or input_name is None:
        return None

    init_func = next(
        (
            item
            for item in cls.node.body
            if isinstance(item, ast.FunctionDef) and item.name == "__init__"
        ),
        None,
    )
    primary = _primary_forward_input_name(forward)
    extractor = _ForwardOperationExtractor(
        self_values=_self_config_values(init_func, {}),
        all_tensor_ops=True,
        param_names=_forward_input_names(forward) - {primary} if primary else set(),
    )
    if primary:
        extractor.var_producer[primary] = FORWARD_METHOD_INPUT
    loop_index = forward.body.index(decoder_loop)
    extractor.statements(forward.body[:loop_index])
    output_producer = extractor.var_producer.get(input_name)
    if output_producer is None or not is_forward_operation(output_producer):
        return None

    by_name = {operation.attr_name: operation for operation in extractor.operations}
    live = {output_producer}
    pending = [output_producer]
    while pending:
        producer = pending.pop()
        operation = by_name.get(producer)
        if operation is None:
            continue
        for predecessor in operation.predecessors:
            if predecessor in by_name and predecessor not in live:
                live.add(predecessor)
                pending.append(predecessor)
    operations = tuple(
        operation for operation in extractor.operations if operation.attr_name in live
    )
    return StackEntryDataflow(operations, output_producer)


def infer_forward_steps_from_init(cls: ClassStructure) -> list[str]:
    """Infer a sequential forward pipeline from ``__init__`` submodule assignments.

    Used when a class has submodule ``self.foo = ...`` assignments but no parsed
    ``forward()`` body (common in test fixtures and some wrapper modules).
    """
    steps: list[str] = []
    for attr, class_name in cls.init_assignments.items():
        if class_name in _SKIP_INIT_CLASS_NAMES:
            continue
        if class_name in _SKIP_INIT_FORWARD_CLASS_NAMES:
            continue
        if attr in _SKIP_INIT_FORWARD_ATTRS or attr.startswith("_"):
            continue
        steps.append(attr)
    return steps


def effective_forward_calls(cls: ClassStructure) -> list[str]:
    """Return parsed ``forward()`` module steps, falling back to inferred init order."""
    steps = [step for step in cls.forward_calls if step not in _SKIP_INIT_CLASS_NAMES]
    if not steps:
        return infer_forward_steps_from_init(cls)
    modules = [step for step in steps if not is_forward_operation(step)]
    return modules if modules else steps


_UNKNOWN = object()
_HOUSEKEEPING_METHODS = frozenset(
    {
        "view",
        "reshape",
        "flatten",
        "type",
        "float",
        "to",
        "unsqueeze",
        "squeeze",
        "expand",
        "contiguous",
        "transpose",
        "permute",
        "detach",
        "clone",
        "view_as_complex",
        "view_as_real",
    }
)
# Layout-only tensor methods: they rearrange or retype a tensor without computing
# new values, so the exporter renders them differently from real math.
_LAYOUT_ONLY_METHOD_LABELS = {
    "view": "View",
    "reshape": "Reshape",
    "flatten": "Flatten",
    "type": "Cast",
    "float": "Cast",
    "to": "Cast",
    "unsqueeze": "Unsqueeze",
    "squeeze": "Squeeze",
    "expand": "Expand",
    "contiguous": "Contiguous",
    "transpose": "Transpose",
    "permute": "Permute",
    "detach": "Detach",
    "clone": "Clone",
    "view_as_complex": "View as complex",
    "view_as_real": "View as real",
}

# Split / Concat rearrange tensors without computing new values. They stay
# visible in the graph (unlike the layout methods above, which are optional)
# but share the white data-movement fill.
LAYOUT_ONLY_LABELS = frozenset(_LAYOUT_ONLY_METHOD_LABELS.values()) | {
    "Split",
    "Concat",
}

# Keyed on the trailing call name, so `x.mean(...)` and `torch.mean(x)` both resolve.
_TENSOR_METHOD_LABELS = {
    # Reductions
    "amax": "Block max",
    "amin": "Block min",
    "sum": "Sum",
    "mean": "Mean",
    "prod": "Product",
    "cumsum": "Cumulative sum",
    "logsumexp": "LogSumExp",
    "argmax": "ArgMax",
    "argmin": "ArgMin",
    "max": "Max",
    "min": "Min",
    "norm": "Norm",
    "var": "Variance",
    "std": "Std",
    # Pointwise math
    "sigmoid": "Sigmoid",
    "softmax": "Softmax",
    "log_softmax": "LogSoftmax",
    "softplus": "Softplus",
    "tanh": "Tanh",
    "relu": "ReLU",
    "silu": "SiLU",
    "gelu": "GELU",
    "erf": "Erf",
    "exp": "Exp",
    "log": "Log",
    "log1p": "Log1p",
    "sqrt": "Sqrt",
    "rsqrt": "Reciprocal sqrt",
    "square": "Square",
    "pow": "Power",
    "abs": "Abs",
    "neg": "Negate",
    "reciprocal": "Reciprocal",
    "sign": "Sign",
    "clamp": "Clamp",
    "clip": "Clamp",
    "nan_to_num": "NaN to num",
    "maximum": "Maximum",
    "minimum": "Minimum",
    "where": "Where",
    "one_hot": "One hot",
    "cos": "Cosine",
    "sin": "Sine",
    # Indexing and assembly
    "gather": "Gather",
    "masked_fill": "Masked fill",
    "masked_scatter": "Masked scatter",
    "scatter": "Scatter",
    "scatter_": "Scatter",
    "index_add": "Index add",
    "index_add_": "Index add",
    "nonzero": "Nonzero",
    "split": "Split",
    "chunk": "Chunk",
    "unbind": "Unbind",
    "stack": "Stack",
    "repeat_interleave": "Repeat interleave",
    "roll": "Roll",
    "flip": "Flip",
    "tril": "Lower triangle",
    "triu": "Upper triangle",
    # Contractions
    "einsum": "Einsum",
    "bmm": "BatchMatMul",
    "mm": "MatMul",
    # Layout-only (suppressed unless every tensor op is requested)
    **_LAYOUT_ONLY_METHOD_LABELS,
}
_FUNCTION_LABELS = {
    "linear": "Linear",
    "matmul": "MatMul",
    "pad": "Pad",
    "topk": "TopK",
    "zeros_like": "Zeros like",
    "ones_like": "Ones like",
    "full_like": "Full like",
    "causal_conv1d_fn": "Causal Conv1D",
    "causal_conv1d_update": "Causal Conv1D update",
    "cat": "Concat",
    "outer": "Outer product",
    "polar": "Polar",
}
# Reductions whose axis decides the output shape, so the axis travels with the node.
_REDUCTION_METHODS = frozenset(
    {
        "sum",
        "mean",
        "prod",
        "amax",
        "amin",
        "max",
        "min",
        "cumsum",
        "logsumexp",
        "argmax",
        "argmin",
        "norm",
        "var",
        "std",
    }
)
_DIM_DETAIL_METHODS = _REDUCTION_METHODS | {"unsqueeze", "squeeze", "gather"}
_BINOP_LABELS = {
    ast.Add: "Add",
    ast.Sub: "Subtract",
    ast.Mult: "Multiply",
    ast.MatMult: "MatMul",
    ast.Div: "Divide",
    ast.FloorDiv: "Floor divide",
    ast.Pow: "Power",
}


def is_forward_operation(attr_name: str) -> bool:
    return attr_name.startswith(FORWARD_OPERATION_PREFIX)


def operation_display_label(label: str, *, class_name: str | None = None) -> str:
    """Human-facing operator name for graph tiles and exports."""
    text = (label or class_name or "").strip()
    return text or "Op"


def classify_matmul_label(*, external_inputs: list[str] | tuple[str, ...]) -> str:
    """Name a GEMM-like op from its operands: Linear when a weight is involved, else MatMul."""
    if external_inputs:
        return "Linear"
    return "MatMul"


def _inline_forward_step(attr_name: str) -> bool:
    return is_forward_operation(attr_name) or is_functional_synthetic(attr_name)


def _config_value(
    node: ast.AST, config: dict[str, Any], self_values: dict[str, Any]
) -> Any:
    """Evaluate the small literal/config expression subset used by model constructors."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "config":
            return config
        return _UNKNOWN
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "config":
            direct = config.get(node.attr, _UNKNOWN)
            if direct is not _UNKNOWN:
                return direct
            aliases = {
                "linear_lower_bound": ("linear_attn_config", "gate_lower_bound"),
            }
            nested_key = aliases.get(node.attr)
            if nested_key is not None:
                nested = config.get(nested_key[0])
                if isinstance(nested, dict):
                    return nested.get(nested_key[1], _UNKNOWN)
            return _UNKNOWN
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return self_values.get(node.attr, _UNKNOWN)
        base = _config_value(node.value, config, self_values)
        if isinstance(base, dict):
            return base.get(node.attr, _UNKNOWN)
        return _UNKNOWN
    if isinstance(node, ast.Call):
        name = _expr_name(node.func)
        if name == "getattr" and len(node.args) >= 2:
            base = _config_value(node.args[0], config, self_values)
            key = _config_value(node.args[1], config, self_values)
            default = (
                _config_value(node.args[2], config, self_values)
                if len(node.args) >= 3
                else _UNKNOWN
            )
            if isinstance(base, dict) and isinstance(key, str):
                return base.get(key, default)
        return _UNKNOWN
    if isinstance(node, ast.BinOp):
        left = _config_value(node.left, config, self_values)
        right = _config_value(node.right, config, self_values)
        if left is _UNKNOWN or right is _UNKNOWN:
            return _UNKNOWN
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
        except (TypeError, ValueError, ZeroDivisionError):
            return _UNKNOWN
        return _UNKNOWN
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _config_value(node.operand, config, self_values)
        return not value if value is not _UNKNOWN else _UNKNOWN
    if isinstance(node, ast.BoolOp):
        values = [_config_value(value, config, self_values) for value in node.values]
        if isinstance(node.op, ast.And):
            if any(value is False for value in values):
                return False
            return (
                all(values)
                if all(value is not _UNKNOWN for value in values)
                else _UNKNOWN
            )
        if isinstance(node.op, ast.Or):
            if any(value is True for value in values):
                return True
            return (
                any(values)
                if all(value is not _UNKNOWN for value in values)
                else _UNKNOWN
            )
    if isinstance(node, ast.Compare):
        left = _config_value(node.left, config, self_values)
        comparators = [
            _config_value(item, config, self_values) for item in node.comparators
        ]
        if left is _UNKNOWN or any(item is _UNKNOWN for item in comparators):
            return _UNKNOWN
        values = [left, *comparators]
        for index, op in enumerate(node.ops):
            a, b = values[index], values[index + 1]
            if isinstance(op, ast.Eq) and not (a == b):
                return False
            if isinstance(op, ast.NotEq) and not (a != b):
                return False
            if isinstance(op, ast.Gt) and not (a > b):
                return False
            if isinstance(op, ast.GtE) and not (a >= b):
                return False
            if isinstance(op, ast.Lt) and not (a < b):
                return False
            if isinstance(op, ast.LtE) and not (a <= b):
                return False
            if isinstance(op, ast.Is) and not (a is b):
                return False
            if isinstance(op, ast.IsNot) and not (a is not b):
                return False
        return True
    return _UNKNOWN


def _self_config_values(
    init_func: ast.FunctionDef | None, config: dict[str, Any]
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if init_func is None:
        return values
    for stmt in init_func.body:
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            continue
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        value_node = stmt.value
        if value_node is None:
            continue
        value = _config_value(value_node, config, values)
        for target in targets:
            if isinstance(target, ast.Attribute) and _is_self_attr(target, target.attr):
                if value is not _UNKNOWN or any(
                    isinstance(item, ast.Name) and item.id == "config"
                    for item in ast.walk(value_node)
                ):
                    values[target.attr] = value
    return values


class _ForwardOperationExtractor:
    """Recover primitive tensor operations and their data dependencies."""

    def __init__(
        self,
        *,
        self_values: dict[str, Any],
        all_tensor_ops: bool,
        param_names: set[str] | None = None,
    ) -> None:
        self.self_values = self_values
        self.all_tensor_ops = all_tensor_ops
        self.param_names = set(param_names or ())
        self.operations: list[ForwardOperation] = []
        self.var_producer: dict[str, str] = {}
        self.var_module_origin: dict[str, str] = {}
        self.step_predecessors: dict[str, tuple[str, ...]] = {}
        self.step_predecessor_args: dict[str, dict[str, str]] = {}
        self.loop_carried: list[LoopCarriedSpec] = []
        self._used_ids: set[str] = set()

    @staticmethod
    def _dedupe(values: list[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value for value in values if value))

    def _operation_id(self, node: ast.AST, label: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        line = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        base = f"{FORWARD_OPERATION_PREFIX}l{line}_c{col}_{slug}"
        candidate = base
        counter = 2
        while candidate in self._used_ids:
            candidate = f"{base}_{counter}"
            counter += 1
        self._used_ids.add(candidate)
        return candidate

    def _emit(
        self,
        node: ast.AST,
        label: str,
        predecessors: list[str],
        external_inputs: list[str],
        *,
        details: list[str] | None = None,
    ) -> str:
        attr_name = self._operation_id(node, label)
        if label.lower() in {"matmul", "matmull"}:
            display = classify_matmul_label(external_inputs=external_inputs)
        else:
            display = operation_display_label(label)
        self.operations.append(
            ForwardOperation(
                attr_name=attr_name,
                label=display,
                class_name=display,
                predecessors=self._dedupe(predecessors),
                external_inputs=self._dedupe(external_inputs),
                details=tuple(details or ()),
                param_inputs=self._param_refs(node),
            )
        )
        return attr_name

    def _param_refs(self, node: ast.AST) -> tuple[str, ...]:
        """Secondary forward parameters this operation's expression reads."""
        if not self.param_names:
            return ()
        return self._dedupe(
            [
                name.id
                for name in ast.walk(node)
                if isinstance(name, ast.Name) and name.id in self.param_names
            ]
        )

    def _call_step_producer(
        self, node: ast.Call, method_name: str | None
    ) -> str | None:
        """Chain step a call *is*, for calls the diagram turns into their own node.

        Mirrors the naming `_extract_self_calls_ordered` uses, so the producer recorded
        here refers to the same node the forward chain will hold.
        """
        func = node.func
        if method_name is not None and _is_self_attr(func, method_name):
            return method_name
        target = _expr_name(func)
        if target and (
            target in _SYNTHETIC_ATTENTION_NAMES
            or (isinstance(func, ast.Name) and _is_kernel_merge_call(func))
        ):
            return SYNTHETIC_ATTENTION
        if target and _is_positional_function_call(func, target):
            return positional_synthetic_attr(target, node.lineno)
        return None

    def _self_attr_input(self, node: ast.Attribute) -> tuple[str | None, list[str]]:
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            # Attributes with a known scalar value are settings, not tensor inputs;
            # ones that could not be evaluated (parameters, buffers) are inputs.
            if self.self_values.get(node.attr, _UNKNOWN) is not _UNKNOWN:
                return None, []
            return None, [node.attr]
        return None, []

    def expression(self, node: ast.AST) -> tuple[str | None, list[str]]:
        if isinstance(node, ast.Name):
            return self.var_producer.get(node.id), []
        if isinstance(node, ast.Attribute):
            producer, external = self._self_attr_input(node)
            if producer is not None or external:
                return producer, external
            # Preserve the computation behind result selectors such as
            # ``tensor.topk(...).indices`` and chained dtype/shape properties.
            return self.expression(node.value)
        if isinstance(node, ast.Constant):
            return None, []
        if isinstance(node, ast.Subscript):
            return self.expression(node.value)
        if isinstance(node, ast.UnaryOp):
            return self.expression(node.operand)
        if isinstance(node, ast.IfExp):
            left, left_external = self.expression(node.body)
            right, right_external = self.expression(node.orelse)
            return right or left, [*left_external, *right_external]
        if isinstance(node, (ast.Tuple, ast.List)):
            producers: list[str] = []
            external: list[str] = []
            for item in node.elts:
                producer, item_external = self.expression(item)
                if producer:
                    producers.append(producer)
                external.extend(item_external)
            return (producers[-1] if producers else None), external
        if isinstance(node, ast.BinOp):
            left, left_external = self.expression(node.left)
            right, right_external = self.expression(node.right)
            label = _BINOP_LABELS.get(type(node.op))
            if label is None:
                return right or left, [*left_external, *right_external]
            direct_module_predecessors = [
                operand.func.attr
                for operand in (node.left, node.right)
                if isinstance(operand, ast.Call)
                and isinstance(operand.func, ast.Attribute)
                and _is_self_attr(operand.func, operand.func.attr)
            ]
            if not left and not right and not direct_module_predecessors:
                return None, [*left_external, *right_external]
            producer = self._emit(
                node,
                label,
                [
                    *[value for value in (left, right) if value],
                    *direct_module_predecessors,
                ],
                [*left_external, *right_external],
            )
            return producer, []
        if not isinstance(node, ast.Call):
            return None, []

        method_name: str | None = None
        base_producer: str | None = None
        external: list[str] = []
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            owner_name = _expr_name(node.func.value)
            is_namespace_call = owner_name in {
                "torch",
                "F",
                "torch.nn.functional",
                "nn.functional",
            }
            if not is_namespace_call:
                base_producer, base_external = self.expression(node.func.value)
                external.extend(base_external)

        call_name = (_expr_name(node.func) or method_name or "").split(".")[-1]
        functional_name = _functional_call_name(node.func)
        registry_activation: str | None = None
        if isinstance(node.func, ast.Subscript):
            registry_name = _expr_name(node.func.value)
            activation_key = _config_value(node.func.slice, {}, self.self_values)
            if registry_name in _ACTIVATION_REGISTRY_NAMES and isinstance(
                activation_key, str
            ):
                registry_activation = _ACTIVATION_DISPLAY_NAMES.get(
                    activation_key.lower(), activation_key
                )
        # ``self.norm(x)`` runs a submodule that happens to share a tensor method's
        # name; it is a chain step, so it must not be relabelled as that method.
        submodule_call = isinstance(node.func, ast.Attribute) and _is_self_attr(
            node.func, method_name
        )
        label = (
            None
            if submodule_call
            else registry_activation
            or _FUNCTION_LABELS.get(functional_name or call_name)
            or _TENSOR_METHOD_LABELS.get(call_name)
        )
        housekeeping = not submodule_call and (
            call_name in _HOUSEKEEPING_METHODS or call_name == "zeros_like"
        )

        def _collect_call_arg_producers(arg: ast.AST) -> tuple[list[str], list[str]]:
            if isinstance(arg, (ast.List, ast.Tuple)):
                producers: list[str] = []
                external: list[str] = []
                for item in arg.elts:
                    producer, item_external = self.expression(item)
                    if producer:
                        producers.append(producer)
                    external.extend(item_external)
                return producers, external
            producer, arg_external = self.expression(arg)
            return ([producer] if producer else []), list(arg_external)

        arg_producers: list[str] = []
        arg_name_map: dict[str, str] = {}
        if not (housekeeping and method_name is not None):
            # Skip self/cls first positional arg for submodule calls.
            start = 0
            if (
                submodule_call
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
            ):
                start = 1
            for idx, arg in enumerate(node.args):
                producers, arg_external = _collect_call_arg_producers(arg)
                arg_producers.extend(producers)
                external.extend(arg_external)
                if submodule_call and idx >= start and len(producers) == 1:
                    name = _arg_name(arg, idx - start)
                    arg_name_map[name] = producers[0]
            for keyword in node.keywords:
                producer, arg_external = self.expression(keyword.value)
                if producer:
                    arg_producers.append(producer)
                    if submodule_call and keyword.arg:
                        arg_name_map[keyword.arg] = producer
                external.extend(arg_external)
        if label is None:
            # A call that becomes its own chain step is what later reads of its result
            # depend on; without this they resolve to whatever fed the call instead, and
            # an operation reading the result looks like it has no source at all.
            own_step = self._call_step_producer(node, method_name)
            if own_step is not None:
                self.step_predecessors[own_step] = self._dedupe(
                    [value for value in (base_producer, *arg_producers) if value]
                )
                if arg_name_map:
                    self.step_predecessor_args[own_step] = arg_name_map
                return own_step, external
            producers = [value for value in (base_producer, *arg_producers) if value]
            return (producers[-1] if producers else None), external

        if housekeeping and not self.all_tensor_ops:
            return (
                base_producer or (arg_producers[0] if arg_producers else None),
                external,
            )

        details: list[str] = []
        if call_name == "linear" and any(
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr in {"type", "float", "to"}
            for arg in node.args
            for item in ast.walk(arg)
        ):
            details.append("dtype: torch.float32")
        if call_name in {"view", "reshape", "expand"}:
            details.append("shape: " + ", ".join(ast.unparse(arg) for arg in node.args))
        if call_name in {"split", "chunk"}:
            # For torch.split(tensor, split_size, dim) the tensor is arg0;
            # for tensor.split(split_size, dim) there is no tensor arg.
            is_method = isinstance(node.func, ast.Attribute) and not (
                isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
            )
            size_idx = 0 if is_method else 1
            dim_idx = size_idx + 1
            if len(node.args) > size_idx:
                details.append(f"split_size: {ast.unparse(node.args[size_idx])}")
            if len(node.args) > dim_idx:
                details.append(f"dim: {ast.unparse(node.args[dim_idx])}")
            for keyword in node.keywords:
                if keyword.arg == "dim":
                    details.append(f"dim: {ast.unparse(keyword.value)}")
                elif keyword.arg in {"split_size_or_sections", "chunks"}:
                    details.append(f"split_size: {ast.unparse(keyword.value)}")
        if call_name == "transpose":
            is_method = isinstance(node.func, ast.Attribute) and not (
                isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
            )
            arg_start = 0 if is_method else 1
            if len(node.args) > arg_start + 1:
                details.append(f"dim0: {ast.unparse(node.args[arg_start])}")
                details.append(f"dim1: {ast.unparse(node.args[arg_start + 1])}")
        if call_name in _DIM_DETAIL_METHODS:
            if node.args:
                details.append(f"dim: {ast.unparse(node.args[0])}")
            for keyword in node.keywords:
                if keyword.arg in {"dim", "keepdim"}:
                    details.append(f"{keyword.arg}: {ast.unparse(keyword.value)}")
        if call_name in {"type", "float", "to"}:
            dtype = (
                ast.unparse(node.args[0])
                if node.args
                else ("float32" if call_name == "float" else "")
            )
            details.append(f"dtype: {dtype}" if dtype else "dtype cast")
        if (
            call_name.endswith("_")
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            details.append(f"mutates: {node.func.value.id}")
        producer = self._emit(
            node,
            label,
            [value for value in (base_producer, *arg_producers) if value],
            external,
            details=details,
        )
        return producer, []

    @staticmethod
    def _target_names(stmt: ast.Assign | ast.AnnAssign) -> list[str]:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        names: list[str] = []
        for target in targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                names.extend(
                    item.id for item in target.elts if isinstance(item, ast.Name)
                )
        return names

    def _bind(self, stmt: ast.Assign | ast.AnnAssign, producer: str | None) -> None:
        if producer is None:
            return
        for name in self._target_names(stmt):
            self.var_producer[name] = producer

    def _range_iteration_count(self, node: ast.For) -> int | None:
        """Resolve a small static ``range(...)`` loop from constructor/config values."""
        iterator = node.iter
        if (
            not isinstance(iterator, ast.Call)
            or _expr_name(iterator.func) != "range"
            or iterator.keywords
            or not 1 <= len(iterator.args) <= 3
        ):
            return None
        values = [_config_value(arg, {}, self.self_values) for arg in iterator.args]
        if not all(isinstance(value, int) for value in values):
            return None
        try:
            count = len(range(*values))
        except (TypeError, ValueError):
            return None
        # Keep generated graphs bounded for malformed or unexpectedly large configs.
        return count if 0 <= count <= 256 else None

    def _annotate_operations_since(self, start: int, detail: str) -> None:
        for index in range(start, len(self.operations)):
            operation = self.operations[index]
            self.operations[index] = ForwardOperation(
                **{
                    **operation.__dict__,
                    "details": (*operation.details, detail),
                }
            )

    def _inject_iterator_predecessor(self, before: int, iterable_producer: str) -> None:
        """Add the loop iterator as a predecessor of operations that use the loop var."""
        for index in range(before, len(self.operations)):
            op = self.operations[index]
            if iterable_producer not in op.predecessors:
                self.operations[index] = ForwardOperation(
                    **{
                        **op.__dict__,
                        "predecessors": (iterable_producer, *op.predecessors),
                    }
                )
                break

    @staticmethod
    def _assigned_names(statements: list[ast.stmt]) -> set[str]:
        names: set[str] = set()
        for statement in statements:
            for node in ast.walk(statement):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    for target in targets:
                        if isinstance(target, ast.Name):
                            names.add(target.id)
                        elif isinstance(target, (ast.Tuple, ast.List)):
                            names.update(
                                item.id
                                for item in target.elts
                                if isinstance(item, ast.Name)
                            )
                elif isinstance(node, ast.AugAssign) and isinstance(
                    node.target, ast.Name
                ):
                    names.add(node.target.id)
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr.endswith("_")
                    and isinstance(node.func.value, ast.Name)
                ):
                    names.add(node.func.value.id)
        return names

    @staticmethod
    def _statements_terminate(statements: list[ast.stmt]) -> bool:
        if not statements:
            return False
        final = statements[-1]
        if isinstance(final, (ast.Return, ast.Raise)):
            return True
        return (
            isinstance(final, ast.If)
            and bool(final.orelse)
            and _ForwardOperationExtractor._statements_terminate(final.body)
            and _ForwardOperationExtractor._statements_terminate(final.orelse)
        )

    def statements(
        self, statements: list[ast.stmt], *, condition: str | None = None
    ) -> None:
        for stmt in statements:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
                producer, _ = self.expression(stmt.value)
                value = stmt.value
                targets = (
                    stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                )
                direct_module = (
                    value.func.attr
                    if isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name)
                    and value.func.value.id == "self"
                    else None
                )
                if direct_module is not None:
                    for target in targets:
                        if isinstance(target, ast.Name):
                            self.var_module_origin[target.id] = direct_module
                if (
                    producer is None
                    and isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name)
                    and any(
                        isinstance(target, (ast.Tuple, ast.List)) for target in targets
                    )
                ):
                    producer = self.var_module_origin.get(value.func.value.id)
                self._bind(stmt, producer)
                continue
            if isinstance(stmt, ast.AugAssign):
                left, left_external = self.expression(stmt.target)
                right, right_external = self.expression(stmt.value)
                label = _BINOP_LABELS.get(type(stmt.op))
                if label:
                    producer = self._emit(
                        stmt,
                        label,
                        [value for value in (left, right) if value],
                        [*left_external, *right_external],
                    )
                    if isinstance(stmt.target, ast.Name):
                        self.var_producer[stmt.target.id] = producer
                continue
            if isinstance(stmt, ast.Expr):
                producer, _ = self.expression(stmt.value)
                if (
                    producer
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)
                ):
                    owner = stmt.value.func.value
                    if isinstance(owner, ast.Name):
                        self.var_producer[owner.id] = producer
                continue
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                self.expression(stmt.value)
                continue
            if isinstance(stmt, ast.If):
                outcome = _config_value(stmt.test, {}, self.self_values)
                if outcome is True:
                    self.statements(stmt.body, condition=condition)
                    if self._statements_terminate(stmt.body):
                        break
                elif outcome is False:
                    self.statements(stmt.orelse, condition=condition)
                    if self._statements_terminate(stmt.orelse):
                        break
                else:
                    test = ast.unparse(stmt.test)
                    before_env = dict(self.var_producer)
                    before = len(self.operations)
                    self.statements(stmt.body, condition=test)
                    body_env = dict(self.var_producer)
                    for index in range(before, len(self.operations)):
                        op = self.operations[index]
                        self.operations[index] = ForwardOperation(
                            **{
                                **op.__dict__,
                                "details": (*op.details, f"condition: {test}"),
                            }
                        )
                    self.var_producer = dict(before_env)
                    before_else = len(self.operations)
                    self.statements(stmt.orelse, condition=f"not ({test})")
                    else_env = dict(self.var_producer)
                    for index in range(before_else, len(self.operations)):
                        op = self.operations[index]
                        self.operations[index] = ForwardOperation(
                            **{
                                **op.__dict__,
                                "details": (*op.details, f"condition: not ({test})"),
                            }
                        )
                    self.var_producer = else_env if stmt.orelse else body_env
                continue
            if isinstance(stmt, ast.For):
                iteration_count = self._range_iteration_count(stmt)
                iterable_producer, _iterable_external = self.expression(stmt.iter)
                if iterable_producer is not None:
                    for index, operation in enumerate(self.operations):
                        if operation.attr_name != iterable_producer:
                            continue
                        self.operations[index] = ForwardOperation(
                            **{
                                **operation.__dict__,
                                "details": (*operation.details, "loop iterator"),
                            }
                        )
                        break
                    if isinstance(stmt.target, ast.Name):
                        self.var_producer[stmt.target.id] = iterable_producer
                    elif isinstance(stmt.target, (ast.Tuple, ast.List)):
                        for elt in stmt.target.elts:
                            if isinstance(elt, ast.Name):
                                self.var_producer[elt.id] = iterable_producer
                before_env = dict(self.var_producer)
                before = len(self.operations)
                self.statements(stmt.body, condition=condition)
                detail = (
                    f"loop: {iteration_count} iterations"
                    if iteration_count is not None
                    else "loop: repeated"
                )
                self._annotate_operations_since(before, detail)
                if iterable_producer is not None:
                    self._inject_iterator_predecessor(before, iterable_producer)
                operation_ids = tuple(
                    operation.attr_name for operation in self.operations[before:]
                )
                loop_id = f"loop_l{stmt.lineno}_c{stmt.col_offset}"
                for variable in sorted(self._assigned_names(stmt.body)):
                    initial = before_env.get(variable)
                    updated = self.var_producer.get(variable)
                    if initial and updated and initial != updated:
                        self.loop_carried.append(
                            LoopCarriedSpec(
                                loop_id=loop_id,
                                iteration_count=iteration_count,
                                variable=variable,
                                initial_producer=initial,
                                updated_producer=updated,
                                operation_ids=operation_ids,
                            )
                        )
                continue
            if isinstance(stmt, ast.With):
                self.statements(stmt.body, condition=condition)


_OPERATION_SOURCE_POS_RE = re.compile(r"^@op_l(\d+)_c(\d+)_")


def _self_call_source_positions(func: ast.FunctionDef) -> dict[str, tuple[int, int]]:
    """First source position each `self.<attr>(...)` call is made at."""
    positions: dict[str, tuple[int, int]] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if not isinstance(target, ast.Attribute):
            continue
        if not (isinstance(target.value, ast.Name) and target.value.id == "self"):
            continue
        where = (node.lineno, node.col_offset)
        if positions.get(target.attr, where) >= where:
            positions[target.attr] = where
    return positions


def _functional_synthetic_source_positions(
    func: ast.FunctionDef,
) -> dict[str, tuple[int, int]]:
    """First source position each ``F.<op>(...)`` maps to a functional synthetic attr."""
    positions: dict[str, tuple[int, int]] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        functional_op = _functional_call_name(node.func)
        if not functional_op:
            continue
        attr = functional_synthetic_attr(functional_op)
        where = (node.lineno, node.col_offset)
        if attr not in positions or where < positions[attr]:
            positions[attr] = where
    return positions


def _kernel_merge_source_position(func: ast.FunctionDef) -> tuple[int, int] | None:
    """First source position of a kernel represented by the synthetic merge node."""
    positions = [
        (node.lineno, node.col_offset)
        for node in ast.walk(func)
        if isinstance(node, ast.Call) and _is_kernel_merge_call(node.func)
    ]
    return min(positions) if positions else None


def _module_calls_for_forward_merge(
    forward_calls: list[str],
    init_assignments: dict[str, str],
    *,
    parsed_operations: list[ForwardOperation],
) -> list[str]:
    """Submodule/synthetic calls to interleave with parsed tensor ops.

    When inline ops were recovered from ``forward()``, drop redundant functional
    synthetics (``@functional_linear``) that duplicate the same ``F.linear(...)``.
    """
    drop_functional = bool(parsed_operations)
    return [
        call
        for call in forward_calls
        if call in init_assignments
        or is_positional_synthetic(call)
        or call == SYNTHETIC_ATTENTION
        or not call.startswith("@")
        or (is_functional_synthetic(call) and not drop_functional)
    ]


def _forward_calls_in_source_order(
    func: ast.FunctionDef,
    module_calls: list[str],
    operations: list[ForwardOperation],
) -> list[str]:
    """Merge submodule calls and parsed tensor ops into the order the forward runs them.

    Within one statement the nested expressions run first, and those sit further right,
    so a later column comes earlier in the chain.
    """

    def sort_key(where: tuple[int, int]) -> tuple[int, int]:
        line, col = where
        return (line, -col)

    ordered: list[tuple[tuple[int, int], str]] = []
    call_positions = _self_call_source_positions(func)
    functional_positions = _functional_synthetic_source_positions(func)
    kernel_position = _kernel_merge_source_position(func)
    fallback = 0
    for call in module_calls:
        where = (
            call_positions.get(call)
            or functional_positions.get(call)
            or positional_synthetic_source_pos(call)
        )
        if where is None and call == SYNTHETIC_ATTENTION:
            where = kernel_position
        if where is None:
            # A call the walk cannot place keeps its parsed order ahead of the ops.
            where = (0, -fallback)
            fallback += 1
        ordered.append((sort_key(where), call))
    for op in operations:
        match = _OPERATION_SOURCE_POS_RE.match(op.attr_name)
        where = (int(match.group(1)), int(match.group(2))) if match else (10**6, 0)
        ordered.append((sort_key(where), op.attr_name))
    ordered.sort(key=lambda item: item[0])
    source_order = [name for _where, name in ordered]
    operation_by_name = {operation.attr_name: operation for operation in operations}
    remaining = list(source_order)
    result: list[str] = []
    while remaining:
        ready = next(
            (
                name
                for name in remaining
                if all(
                    predecessor not in remaining
                    for predecessor in operation_by_name.get(
                        name,
                        ForwardOperation(name, name, name),
                    ).predecessors
                )
            ),
            remaining[0],
        )
        result.append(ready)
        remaining.remove(ready)
    return result


def _return_value_names(value: ast.AST) -> list[str]:
    if isinstance(value, ast.Tuple):
        return [elt.id for elt in value.elts if isinstance(elt, ast.Name)]
    if isinstance(value, ast.Name):
        return [value.id]
    return []


def _extract_forward_return_metadata(
    func: ast.FunctionDef,
    var_producer: dict[str, str],
) -> tuple[dict[str, str], list[str], str | None]:
    """Map ``return (a, b, c)`` names to the inline ops that produce them."""
    return_order: list[str] = []
    for stmt in reversed(func.body):
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            return_order = _return_value_names(stmt.value)
            break
    slots = {name: var_producer[name] for name in return_order if name in var_producer}
    input_name = _primary_forward_input_name(func)
    # A tuple return's last value is often the continuation (``post, comb,
    # collapsed``). The first value is the continuation when it is the module's
    # actual result (``attn_output, attn_weights``). Prefer a tensor the forward
    # names as the main hidden state before falling back to the last slot.
    main_names = {"hidden_states", "hidden_state", "attn_output", "output", "result"}
    primary = (
        input_name
        if input_name in return_order
        else next((name for name in return_order if name in main_names), None)
    )
    if primary is None:
        primary = return_order[-1] if return_order else None
    return slots, return_order, primary


def _live_forward_steps(
    *,
    operations: dict[str, ForwardOperation],
    return_slots: dict[str, str],
) -> set[str]:
    """Backward closure of ops and submodule steps that feed returned values."""
    if not return_slots:
        return set(operations.keys())
    live_ops: set[str] = set()
    pending = [producer for producer in return_slots.values() if producer in operations]
    while pending:
        step = pending.pop()
        if step in live_ops:
            continue
        live_ops.add(step)
        operation = operations.get(step)
        if operation is None:
            continue
        for pred in operation.predecessors:
            if pred in operations and pred not in live_ops:
                pending.append(pred)
    live = set(live_ops)
    for step in live_ops:
        operation = operations.get(step)
        if operation is None:
            continue
        for pred in operation.predecessors:
            if pred not in operations:
                live.add(pred)
    return live


def _prune_forward_pipeline(
    *,
    forward_calls: list[str],
    operations: dict[str, ForwardOperation],
    return_slots: dict[str, str],
) -> tuple[list[str], dict[str, ForwardOperation]]:
    if len(return_slots) < 2:
        return forward_calls, operations
    live = _live_forward_steps(operations=operations, return_slots=return_slots)
    pruned_operations = {name: op for name, op in operations.items() if name in live}
    pruned_calls = [
        step for step in forward_calls if step not in operations or step in live
    ]
    return pruned_calls, pruned_operations


def _self_module_call_attr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if _is_self_attr(node.func, node.func.attr):
            return node.func.attr
    return None


def _forward_call_output_names(forward_func: ast.FunctionDef) -> dict[str, str]:
    """Map ``module_attr -> local variable`` the caller binds its result to.

    Modules that return a bare expression (`return self.weight * hidden_states`)
    expose no tensor name of their own, so the name the caller gives the result is
    the only source-derived label available for that boundary.
    """
    names: dict[str, str] = {}
    for node in ast.walk(forward_func):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        attr = _self_module_call_attr(node.value) if node.value is not None else None
        if attr is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        # A reassigned spine variable keeps the first binding: later statements
        # rebind the same name for unrelated steps.
        names.setdefault(attr, targets[0].id)
    return names


def _module_return_unpacks(
    forward_func: ast.FunctionDef,
    init_assignments: dict[str, str],
    registry: dict[str, ClassStructure],
) -> dict[str, dict[str, str]]:
    """Map ``module_attr -> {local_var: producing_step}`` for tuple unpacks."""
    unpacks: dict[str, dict[str, str]] = {}
    for stmt in forward_func.body:
        if not isinstance(stmt, ast.Assign):
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Tuple) or not isinstance(stmt.value, ast.Call):
            continue
        module_attr = _self_module_call_attr(stmt.value)
        if module_attr is None or module_attr not in init_assignments:
            continue
        callee = registry.get(init_assignments[module_attr])
        if callee is None or not callee.forward_return_order:
            continue
        if len(target.elts) != len(callee.forward_return_order):
            continue
        mapping: dict[str, str] = {}
        for elt, slot_name in zip(target.elts, callee.forward_return_order):
            if not isinstance(elt, ast.Name):
                mapping = {}
                break
            producer = callee.forward_return_slots.get(slot_name)
            if producer is None:
                mapping = {}
                break
            mapping[elt.id] = producer
        if mapping:
            unpacks[module_attr] = mapping
    return unpacks


def _expression_at_operation_line(
    func: ast.FunctionDef,
    attr_name: str,
    operation_label: str,
) -> ast.AST | None:
    match = _OPERATION_SOURCE_POS_RE.match(attr_name)
    if match is None:
        return None
    target_line = int(match.group(1))
    target_col = int(match.group(2))
    for stmt in func.body:
        for node in ast.walk(stmt):
            if (
                getattr(node, "lineno", None) != target_line
                or getattr(node, "col_offset", None) != target_col
            ):
                continue
            label: str | None = None
            if isinstance(node, ast.BinOp):
                raw_label = _BINOP_LABELS.get(type(node.op))
                label = operation_display_label(raw_label) if raw_label else None
            elif isinstance(node, ast.Call):
                call_name = (_expr_name(node.func) or "").split(".")[-1]
                raw_label = _FUNCTION_LABELS.get(
                    _functional_call_name(node.func) or call_name
                ) or _TENSOR_METHOD_LABELS.get(call_name)
                label = operation_display_label(raw_label) if raw_label else None
            if label == operation_label:
                return node
        if getattr(stmt, "lineno", None) != target_line:
            continue
        if isinstance(stmt, ast.Return):
            return stmt.value
        if isinstance(stmt, ast.Assign):
            return stmt.value
        if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            return stmt.value
        if isinstance(stmt, ast.AugAssign):
            return stmt.value
    return None


def _vars_read_in_expr(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _latest_module_assignment_before(
    func: ast.FunctionDef,
    variable: str,
    *,
    line: int,
    column: int,
) -> str | None:
    latest: tuple[tuple[int, int], str | None] | None = None
    for node in ast.walk(func):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        position = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
        if position >= (line, column):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names: list[str] = []
        for target in targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                names.extend(
                    item.id for item in target.elts if isinstance(item, ast.Name)
                )
        if variable not in names:
            continue
        value = node.value
        module = _self_module_call_attr(value)
        if latest is None or position > latest[0]:
            latest = (position, module)
    return latest[1] if latest is not None else None


def _refine_forward_operation_predecessors(
    forward_func: ast.FunctionDef,
    forward_operations: dict[str, ForwardOperation],
    *,
    module_unpacks: dict[str, dict[str, str]],
) -> dict[str, ForwardOperation]:
    if not module_unpacks:
        return forward_operations
    refined: dict[str, ForwardOperation] = {}
    for name, operation in forward_operations.items():
        match = _OPERATION_SOURCE_POS_RE.match(name)
        position = (
            (int(match.group(1)), int(match.group(2)))
            if match is not None
            else (10**9, 10**9)
        )
        expr = _expression_at_operation_line(forward_func, name, operation.label)
        vars_read = _vars_read_in_expr(expr)
        predecessors: list[str] = []
        for pred in operation.predecessors:
            var_map = module_unpacks.get(pred)
            if var_map:
                mapped = [
                    producer
                    for var, producer in var_map.items()
                    if var in vars_read
                    and _latest_module_assignment_before(
                        forward_func,
                        var,
                        line=position[0],
                        column=position[1],
                    )
                    == pred
                ]
                if mapped:
                    predecessors.extend(mapped)
                    continue
            predecessors.append(pred)
        refined[name] = ForwardOperation(
            attr_name=operation.attr_name,
            label=operation.label,
            class_name=operation.class_name,
            predecessors=tuple(dict.fromkeys(predecessors)),
            external_inputs=operation.external_inputs,
            details=operation.details,
            param_inputs=operation.param_inputs,
        )
    return refined


def _apply_forward_analysis(
    forward_func: ast.FunctionDef,
    analysis: ForwardAnalysis,
    *,
    forward_calls: list[str],
    init_assignments: dict[str, str],
) -> tuple[
    list[str], dict[str, ForwardOperation], dict[str, str], list[str], str | None
]:
    operations = {op.attr_name: op for op in analysis.operations}
    pruned_calls, pruned_operations = _prune_forward_pipeline(
        forward_calls=forward_calls,
        operations=operations,
        return_slots=analysis.return_slots,
    )
    return (
        pruned_calls,
        pruned_operations,
        analysis.return_slots,
        analysis.return_order,
        analysis.primary_return_slot,
    )


def finalize_class_registry(registry: dict[str, ClassStructure]) -> None:
    """Resolve submodule return unpacks and refine inline op predecessors."""
    referenced: dict[str, set[str]] = {}
    for cls in registry.values():
        forward_func = next(
            (
                item
                for item in cls.node.body
                if isinstance(item, ast.FunctionDef) and item.name == "forward"
            ),
            None,
        )
        if forward_func is None:
            continue
        for module_attr, var_map in _module_return_unpacks(
            forward_func,
            cls.init_assignments,
            registry,
        ).items():
            callee_name = cls.init_assignments.get(module_attr)
            if callee_name is None:
                continue
            referenced.setdefault(callee_name, set()).update(var_map.values())

    for cls in registry.values():
        cls.referenced_return_producers = set(referenced.get(cls.name, set()))
        if not cls.forward_operations:
            continue
        forward_func = next(
            (
                item
                for item in cls.node.body
                if isinstance(item, ast.FunctionDef) and item.name == "forward"
            ),
            None,
        )
        if forward_func is None:
            continue
        module_unpacks = _module_return_unpacks(
            forward_func, cls.init_assignments, registry
        )
        cls.forward_operations = _refine_forward_operation_predecessors(
            forward_func,
            cls.forward_operations,
            module_unpacks=module_unpacks,
        )


def _forward_operations_from_forward(
    func: ast.FunctionDef,
    *,
    self_values: dict[str, Any],
    all_tensor_ops: bool,
) -> ForwardAnalysis:
    # The primary parameter is the main path, so only the extra ones can identify
    # which step consumes a side feed.
    primary = _primary_forward_input_name(func)
    extractor = _ForwardOperationExtractor(
        self_values=self_values,
        all_tensor_ops=all_tensor_ops,
        param_names=_forward_input_names(func) - {primary} if primary else set(),
    )
    # An operation reading the primary parameter partway through the forward reads the
    # value arriving at the chain, not the previous step. Naming it lets those reads
    # resolve to the chain input instead of silently inheriting the wrong producer.
    if primary:
        extractor.var_producer[primary] = FORWARD_METHOD_INPUT
    extractor.statements(func.body)
    return_slots, return_order, primary_return_slot = _extract_forward_return_metadata(
        func,
        extractor.var_producer,
    )
    return ForwardAnalysis(
        operations=extractor.operations,
        var_producer=dict(extractor.var_producer),
        step_predecessors=dict(extractor.step_predecessors),
        step_predecessor_args=dict(extractor.step_predecessor_args),
        return_slots=return_slots,
        return_order=return_order,
        primary_return_slot=primary_return_slot,
        loop_carried=list(extractor.loop_carried),
    )


def expand_class_forward_dataflow(
    cls: ClassStructure,
    registry: dict[str, ClassStructure],
) -> None:
    """Populate all source tensor-method steps for one selected class."""
    if cls.dataflow_expanded:
        return
    forward = next(
        (
            item
            for item in cls.node.body
            if isinstance(item, ast.FunctionDef) and item.name == "forward"
        ),
        None,
    )
    if forward is None:
        return
    cls.dataflow_expanded = True
    cls.forward_param_inputs = [
        arg.arg
        for arg in forward.args.posonlyargs + forward.args.args
        if arg.arg != "self"
    ]
    init_func = next(
        (
            item
            for item in cls.node.body
            if isinstance(item, ast.FunctionDef) and item.name == "__init__"
        ),
        None,
    )
    analysis = _forward_operations_from_forward(
        forward,
        self_values=_self_config_values(init_func, {}),
        all_tensor_ops=True,
    )
    if not analysis.operations:
        return
    module_calls = _module_calls_for_forward_merge(
        cls.forward_calls,
        cls.init_assignments,
        parsed_operations=analysis.operations,
    )
    merged_calls = _forward_calls_in_source_order(
        forward, module_calls, analysis.operations
    )
    (
        cls.forward_calls,
        cls.forward_operations,
        cls.forward_return_slots,
        cls.forward_return_order,
        cls.primary_return_slot,
    ) = _apply_forward_analysis(
        forward,
        analysis,
        forward_calls=merged_calls,
        init_assignments=cls.init_assignments,
    )
    cls.forward_step_predecessors = dict(analysis.step_predecessors)
    cls.forward_step_predecessor_args = dict(analysis.step_predecessor_args)
    cls.forward_operations = _refine_forward_operation_predecessors(
        forward,
        cls.forward_operations,
        module_unpacks=_module_return_unpacks(forward, cls.init_assignments, registry),
    )


# Backwards-compatible alias used internally.
_ClassInfo = ClassStructure


class _ModelAstVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        all_tensor_ops: bool = False,
    ) -> None:
        self.classes: dict[str, ClassStructure] = {}
        self.config = dict(config or {})
        self.all_tensor_ops = all_tensor_ops

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        init_assignments: dict[str, str] = {}
        init_details: dict[str, list[str]] = {}
        init_assignment_options: dict[str, list[str]] = {}
        forward_calls: list[str] = []
        norm_before: list[str] = []
        attention_inputs: dict[str, list[str]] = {}
        parallel_gates: list[str] = []
        input_fed_calls: list[str] = []
        gate_activations: dict[str, str] = {}
        forward_step_details: dict[str, list[str]] = {}
        side_inputs: dict[str, list[SideInputSpec]] = {}
        forward_input_name: str | None = None
        forward_operations: dict[str, ForwardOperation] = {}
        forward_step_predecessors: dict[str, tuple[str, ...]] = {}
        forward_step_predecessor_args: dict[str, dict[str, str]] = {}
        forward_return_slots: dict[str, str] = {}
        forward_return_order: list[str] = []
        primary_return_slot: str | None = None
        forward_call_output_names: dict[str, str] = {}
        forward_loop_carried: list[LoopCarriedSpec] = []
        single_op_methods: dict[str, ForwardOperation] = {}
        multi_op_methods: dict[str, list[ForwardOperation]] = {}
        init_func = next(
            (
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "__init__"
            ),
            None,
        )
        forward_func = next(
            (
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "forward"
            ),
            None,
        )

        if init_func is not None:
            init_assignments, init_details, init_assignment_options = _parse_init(
                init_func,
                config=self.config,
            )
        if forward_func is not None:
            forward_input_name = _primary_forward_input_name(forward_func)
            forward_call_output_names = _forward_call_output_names(forward_func)
            resolved_forward_func = _resolve_local_module_alias_calls(forward_func)
            (
                forward_calls,
                norm_before,
                attention_inputs,
                side_inputs,
                parsed_step_details,
            ) = _parse_forward(resolved_forward_func)
            alternate = _alternate_forward_dispatches(forward_func)
            if alternate:
                forward_calls = [
                    call for call in forward_calls if call not in alternate
                ]
            input_fed_calls = _input_fed_calls_from_forward(forward_func)
            parallel_gates = _parallel_gates_from_forward(forward_func)
            if forward_calls and parallel_gates:
                # Routers like MoE `gate` run on hidden_states as the main path, not in parallel.
                parallel_gates = [
                    gate for gate in parallel_gates if gate != forward_calls[0]
                ]
            gate_activations = _parallel_gate_activations_from_forward(
                forward_func, parallel_gates
            )
            forward_step_details = dict(parsed_step_details)
            if _is_moe_gate_class(node.name, forward_calls):
                values = _self_config_values(init_func, self.config)
                analysis = _forward_operations_from_forward(
                    forward_func,
                    self_values=values,
                    all_tensor_ops=self.all_tensor_ops,
                )
                if analysis.operations:
                    forward_step_predecessors = dict(analysis.step_predecessors)
                    forward_step_predecessor_args = dict(analysis.step_predecessor_args)
                    forward_loop_carried = list(analysis.loop_carried)
                    (
                        forward_calls,
                        forward_operations,
                        forward_return_slots,
                        forward_return_order,
                        primary_return_slot,
                    ) = _apply_forward_analysis(
                        forward_func,
                        analysis,
                        forward_calls=forward_calls,
                        init_assignments=init_assignments,
                    )
                    forward_step_details.update(
                        {
                            op.attr_name: list(op.details)
                            for op in forward_operations.values()
                        }
                    )
            forward_step_details.update(
                _method_forward_step_details(node, forward_calls, init_assignments)
            )
            single_op_methods = _single_op_forward_methods(
                node,
                forward_calls,
                init_assignments,
                self_values=_self_config_values(init_func, self.config),
                all_tensor_ops=self.all_tensor_ops,
            )
            multi_op_methods = _multi_op_forward_methods(
                node,
                forward_calls,
                init_assignments,
                self_values=_self_config_values(init_func, self.config),
                all_tensor_ops=self.all_tensor_ops,
            )
            delegates_inline = _forward_delegates_to_nothing(node.name, forward_calls)
            if (
                _forward_owns_tensor_math(forward_calls, init_assignments)
                or delegates_inline
            ):
                values = _self_config_values(init_func, self.config)
                analysis = _forward_operations_from_forward(
                    forward_func,
                    self_values=values,
                    all_tensor_ops=self.all_tensor_ops,
                )
                if analysis.operations:
                    forward_step_predecessors = dict(analysis.step_predecessors)
                    forward_step_predecessor_args = dict(analysis.step_predecessor_args)
                    forward_loop_carried = list(analysis.loop_carried)
                    module_calls = _module_calls_for_forward_merge(
                        forward_calls,
                        init_assignments,
                        parsed_operations=analysis.operations,
                    )
                    merged_calls = _forward_calls_in_source_order(
                        forward_func,
                        module_calls,
                        analysis.operations,
                    )
                    (
                        forward_calls,
                        forward_operations,
                        forward_return_slots,
                        forward_return_order,
                        primary_return_slot,
                    ) = _apply_forward_analysis(
                        forward_func,
                        analysis,
                        forward_calls=merged_calls,
                        init_assignments=init_assignments,
                    )
                    forward_step_details.update(
                        {
                            op.attr_name: list(op.details)
                            for op in forward_operations.values()
                        }
                    )
            elif forward_func is not None:
                values = _self_config_values(init_func, self.config)
                probed = _forward_operations_from_forward(
                    forward_func,
                    self_values=values,
                    all_tensor_ops=self.all_tensor_ops,
                )
                if _forward_mixes_modules_and_inline_ops(
                    forward_calls,
                    init_assignments,
                    probed.operations,
                ):
                    forward_step_predecessors = dict(probed.step_predecessors)
                    forward_step_predecessor_args = dict(probed.step_predecessor_args)
                    forward_loop_carried = list(probed.loop_carried)
                    module_calls = _module_calls_for_forward_merge(
                        forward_calls,
                        init_assignments,
                        parsed_operations=probed.operations,
                    )
                    merged_calls = _forward_calls_in_source_order(
                        forward_func,
                        module_calls,
                        probed.operations,
                    )
                    (
                        forward_calls,
                        forward_operations,
                        forward_return_slots,
                        forward_return_order,
                        primary_return_slot,
                    ) = _apply_forward_analysis(
                        forward_func,
                        probed,
                        forward_calls=merged_calls,
                        init_assignments=init_assignments,
                    )
                    forward_step_details.update(
                        {
                            op.attr_name: list(op.details)
                            for op in forward_operations.values()
                        }
                    )

        self.classes[node.name] = ClassStructure(
            name=node.name,
            node=node,
            init_assignments=init_assignments,
            init_details=init_details,
            init_assignment_options=init_assignment_options,
            forward_calls=forward_calls,
            norm_before=norm_before,
            attention_inputs=attention_inputs,
            parallel_gates=parallel_gates,
            input_fed_calls=input_fed_calls,
            gate_activations=gate_activations,
            forward_step_details=forward_step_details,
            side_inputs=side_inputs,
            forward_input_name=forward_input_name,
            forward_operations=forward_operations,
            forward_step_predecessors=forward_step_predecessors,
            forward_step_predecessor_args=forward_step_predecessor_args,
            single_op_methods=single_op_methods,
            multi_op_methods=multi_op_methods,
            forward_return_slots=forward_return_slots,
            forward_return_order=forward_return_order,
            primary_return_slot=primary_return_slot,
            forward_call_output_names=forward_call_output_names,
            loop_carried=forward_loop_carried,
            forward_param_inputs=(
                [
                    arg.arg
                    for arg in forward_func.args.posonlyargs + forward_func.args.args
                    if arg.arg != "self"
                ]
                if forward_func is not None
                else []
            ),
        )
        self.generic_visit(node)


def _parse_init(
    func: ast.FunctionDef,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
    assignments: dict[str, str] = {}
    details: dict[str, list[str]] = {}
    options: dict[str, list[str]] = {}

    def record_assignment(attr: str, value: ast.AST) -> None:
        class_names = _assignment_class_names(value, config=config)
        if not class_names:
            return
        # A registry lookup is the fallback arm of a config switch whose other arm
        # constructs a real module (`SituAndMul` vs `ACT2FN[...]`), so it must not
        # displace that module regardless of which arm the walk reaches last.
        if attr in assignments and _activation_registry_class_name(value, config):
            return
        attr_options = options.setdefault(attr, [])
        for class_name in class_names:
            if class_name not in attr_options:
                attr_options.append(class_name)
        assignments[attr] = class_names[0]
        details[attr] = _assignment_details(value, class_names[0])

    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and _is_self_attr(
                    target, target.attr
                ):
                    record_assignment(target.attr, node.value)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if (
                isinstance(target, ast.Attribute)
                and _is_self_attr(target, target.attr)
                and node.value is not None
            ):
                record_assignment(target.attr, node.value)

    return assignments, details, options


def _activation_registry_class_name(
    node: ast.AST,
    config: dict[str, Any] | None,
) -> str | None:
    """Resolve an activation-registry lookup to the activation the config selects."""
    if not isinstance(node, ast.Subscript):
        return None
    registry = (_expr_name(node.value) or "").rsplit(".", 1)[-1]
    if registry not in _ACTIVATION_REGISTRY_NAMES:
        return None
    key = node.slice
    name: object = None
    if isinstance(key, ast.Constant):
        name = key.value
    elif isinstance(key, ast.Attribute):
        name = (config or {}).get(key.attr)
    if not isinstance(name, str) or not name.strip():
        return None
    lowered = name.strip().lower()
    if lowered in _ACTIVATION_DISPLAY_NAMES:
        return _ACTIVATION_DISPLAY_NAMES[lowered]
    return lowered.replace("_", " ").title().replace(" ", "")


def _assignment_class_names(
    node: ast.AST,
    *,
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Return every constructible module class represented by an assignment."""
    if isinstance(node, ast.IfExp):
        names = _assignment_class_names(
            node.body, config=config
        ) + _assignment_class_names(node.orelse, config=config)
        return list(dict.fromkeys(names))
    if isinstance(node, ast.ListComp):
        return _assignment_class_names(node.elt, config=config)
    if isinstance(node, ast.Subscript):
        activation = _activation_registry_class_name(node, config)
        return [activation] if activation else []
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr == "Parameter":
            return []
        if isinstance(node.func, (ast.Name, ast.Attribute)) and _expr_name(
            node.func
        ) in {
            "ModuleList",
            "nn.ModuleList",
            "torch.nn.ModuleList",
        }:
            if node.args:
                return _assignment_class_names(node.args[0], config=config)
            return []
        class_name = _call_class_name(node)
        if class_name in _SKIP_INIT_CLASS_NAMES:
            return []
        return [class_name] if class_name else []
    if isinstance(node, (ast.List, ast.Tuple)):
        names: list[str] = []
        for item in node.elts:
            names.extend(_assignment_class_names(item, config=config))
        return list(dict.fromkeys(names))
    return []


def _assignment_class_name(
    node: ast.AST,
    *,
    config: dict[str, Any] | None = None,
) -> str | None:
    """Return the preferred module class for backwards-compatible callers."""
    names = _assignment_class_names(node, config=config)
    if names:
        return names[0]
    return None


def _assignment_details(node: ast.AST, class_name: str) -> list[str]:
    details: list[str] = []
    if not isinstance(node, ast.Call):
        return details

    for keyword in node.keywords:
        if keyword.arg in {"num_experts", "top_k", "num_experts_per_tok"}:
            value = (
                ast.literal_eval(keyword.value) if _is_literal(keyword.value) else None
            )
            if value is not None:
                details.append(f"{keyword.arg}={value}")
        if keyword.arg == "activation" and _is_literal(keyword.value):
            raw = ast.literal_eval(keyword.value)
            if isinstance(raw, str):
                details.append(
                    _GATE_ACTIVATION_NAMES.get(raw.lower(), raw.capitalize())
                )

    if re.search(r"SharedExpert|shared", class_name, re.I):
        details.append("shared expert path")
    return details


def _is_literal(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except Exception:
        return False


def _dedupe_chain(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _chains_from_expr(
    value: ast.AST, var_chains: dict[str, list[str]]
) -> list[list[str]]:
    """Collect provenance chains from variable references inside an expression."""
    chains: list[list[str]] = []
    if isinstance(value, ast.Name):
        chain = var_chains.get(value.id, [])
        if chain:
            chains.append(list(chain))
    elif isinstance(value, ast.Call):
        for arg in value.args:
            chains.extend(_chains_from_expr(arg, var_chains))
        for keyword in value.keywords:
            chains.extend(_chains_from_expr(keyword.value, var_chains))
        if isinstance(value.func, ast.Attribute):
            chains.extend(_chains_from_expr(value.func.value, var_chains))
    elif isinstance(value, (ast.Tuple, ast.List)):
        for elt in value.elts:
            chains.extend(_chains_from_expr(elt, var_chains))
    elif isinstance(value, ast.BinOp):
        chains.extend(_chains_from_expr(value.left, var_chains))
        chains.extend(_chains_from_expr(value.right, var_chains))
    return chains


def _merge_chains_from_value(
    value: ast.AST,
    var_chains: dict[str, list[str]],
    stmt_calls: list[str],
) -> list[str]:
    """Merge input-variable provenance with self-module calls from an assignment."""
    merged: list[str] = []
    for chain in _chains_from_expr(value, var_chains):
        merged.extend(chain)
    merged = _dedupe_chain(merged)
    for call in stmt_calls:
        if call not in merged:
            merged.append(call)
    if merged:
        return merged
    if stmt_calls:
        return list(stmt_calls)
    if isinstance(value, ast.Name):
        return list(var_chains.get(value.id, []))
    return []


def _trace_var_chain(
    value: ast.AST,
    var_chains: dict[str, list[str]],
    stmt_calls: list[str],
) -> list[str]:
    return _merge_chains_from_value(value, var_chains, stmt_calls)


def _tuple_source_names(value: ast.AST) -> list[str] | None:
    """Names of inputs when a tuple assignment maps 1:1 over an input tuple."""
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "map"
    ):
        if len(value.args) < 2 or not isinstance(value.args[1], (ast.Tuple, ast.List)):
            return None
        names: list[str] = []
        for elt in value.args[1].elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            else:
                return None
        return names
    if isinstance(value, (ast.Tuple, ast.List)):
        names = []
        for elt in value.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            else:
                return None
        return names
    return None


def _record_assign_targets(
    node: ast.Assign,
    stmt_calls: list[str],
    var_chains: dict[str, list[str]],
) -> None:
    chain = _merge_chains_from_value(node.value, var_chains, stmt_calls)

    def assign_one(target: ast.AST, provenance: list[str]) -> None:
        if not isinstance(target, ast.Name):
            return
        if provenance:
            var_chains[target.id] = list(provenance)
        elif isinstance(node.value, ast.Name):
            var_chains[target.id] = list(var_chains.get(node.value.id, []))

    target = node.targets[0]
    if isinstance(target, ast.Tuple):
        source_names = _tuple_source_names(node.value)
        if source_names is not None and len(source_names) == len(target.elts):
            zipped = True
            for elt, source_name in zip(target.elts, source_names):
                if not isinstance(elt, ast.Name):
                    zipped = False
                    break
                source_chain = list(var_chains.get(source_name, []))
                for call in stmt_calls:
                    if call not in source_chain:
                        source_chain.append(call)
                assign_one(elt, source_chain or list(stmt_calls))
            if zipped:
                return
        for elt in target.elts:
            assign_one(elt, chain)
        return
    assign_one(target, chain)


_KERNEL_PRODUCER_SKIP_KWARGS = frozenset(
    {
        "initial_state",
        "recurrent_state",
        "A_log",
        "dt_bias",
        "cu_seqlens",
        "cache",
        "output_final_state",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "use_beta_sigmoid_in_kernel",
        "safe_gate",
        "lower_bound",
        "transpose_state_layout",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "cache_params",
    }
)


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


def _collect_kernel_producers(
    call: ast.Call,
    var_chains: dict[str, list[str]],
) -> dict[str, list[str]]:
    producers: dict[str, list[str]] = {}

    def consider(label: str, arg: ast.AST) -> None:
        if not isinstance(arg, ast.Name):
            return
        chain = var_chains.get(arg.id, [])
        if chain:
            producers[label] = list(chain)

    for keyword in call.keywords:
        if keyword.arg in _KERNEL_PRODUCER_SKIP_KWARGS:
            continue
        if isinstance(keyword.value, ast.Attribute):
            continue
        if keyword.arg:
            consider(keyword.arg, keyword.value)

    args = call.args
    start = 1 if args and isinstance(args[0], ast.Name) and args[0].id == "self" else 0
    for index, arg in enumerate(args[start:], start=start):
        if isinstance(arg, ast.Name):
            consider(arg.id, arg)
        else:
            consider(f"in{index - start}", arg)

    return producers


_KERNEL_DETAIL_SKIP_KWARGS = frozenset(
    {
        "initial_state",
        "recurrent_state",
        "output_final_state",
        "cu_seqlens",
        "cu_seqlens_cpu",
        "cache",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "cache_params",
        "cp_context",
        "chunk_indices",
        "return_intermediate_states",
        "disable_recompute",
        "scale",
        "chunk_size",
        "state_v_first",
    }
)


def _collect_external_imports(tree: ast.AST) -> dict[str, str]:
    """Collect top-level imported names from a modeling module (including guarded imports)."""
    bindings: dict[str, str] = {}

    def register(name: str, module: str, symbol: str) -> None:
        bindings[name] = f"{module}#{symbol}" if module else symbol

    def walk_stmts(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.ImportFrom):
                module = stmt.module or ""
                for alias in stmt.names:
                    if alias.name == "*":
                        continue
                    register(alias.asname or alias.name, module, alias.name)
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    name = alias.asname or alias.name
                    register(name, alias.name, alias.name)
            elif isinstance(stmt, ast.Try):
                walk_stmts(stmt.body)
                for handler in stmt.handlers:
                    walk_stmts(handler.body)
                walk_stmts(stmt.orelse)
                walk_stmts(stmt.finalbody)

    if isinstance(tree, ast.Module):
        walk_stmts(tree.body)
    return bindings


def _enrich_kernel_import_details(
    classes: dict[str, ClassStructure],
    imports: dict[str, str],
) -> None:
    """Attach ``import:`` metadata to synthetic attention steps from modeling imports."""
    for cls in classes.values():
        details = cls.forward_step_details.get(SYNTHETIC_ATTENTION)
        if not details:
            continue
        if any(line.startswith("import:") for line in details):
            continue
        kernel = kernel_name_from_step_details(details)
        if not kernel:
            continue
        import_ref = imports.get(kernel)
        if import_ref:
            cls.forward_step_details[SYNTHETIC_ATTENTION] = [
                *details,
                f"import: {import_ref}",
            ]


def _resolve_dispatched_attention_kernel(
    classes: dict[str, ClassStructure],
    config: dict[str, Any] | None,
) -> None:
    """Name the kernel a dispatched attention call runs, from the checkpoint config.

    A forward that calls ``ALL_ATTENTION_FUNCTIONS[config._attn_implementation]``
    through a local variable leaves the AST with nothing but the variable's name.
    The checkpoint says which implementation that variable resolves to.
    """
    implementation = (config or {}).get("_attn_implementation")
    if not isinstance(implementation, str) or not implementation.strip():
        return
    resolved = implementation.strip()
    for cls in classes.values():
        details = cls.forward_step_details.get(SYNTHETIC_ATTENTION)
        if not details:
            continue
        kernel = kernel_name_from_step_details(details)
        if kernel is None or kernel.lower() not in _ATTENTION_DISPATCH_NAMES:
            continue
        cls.forward_step_details[SYNTHETIC_ATTENTION] = [
            f"kernel: {resolved}" if line.startswith("kernel:") else line
            for line in details
        ]


def _kernel_call_detail_lines(call: ast.Call) -> list[str]:
    """Capture kernel name and keyword arguments from a modeling forward call."""
    kernel_name = _expr_name(call.func) or "kernel"
    lines = [f"kernel: {kernel_name.split('.')[-1]}"]
    for keyword in call.keywords:
        if keyword.arg in _KERNEL_DETAIL_SKIP_KWARGS or keyword.arg is None:
            continue
        if isinstance(keyword.value, ast.Constant):
            value = repr(keyword.value.value)
        elif isinstance(keyword.value, ast.Name):
            value = keyword.value.id
        elif isinstance(keyword.value, ast.Attribute):
            value = _expr_name(keyword.value) or ast.unparse(keyword.value)
        else:
            value = ast.unparse(keyword.value)
        lines.append(f"kwarg: {keyword.arg}={value}")
    return lines


def _inject_kernel_merge(
    node: ast.AST,
    var_chains: dict[str, list[str]],
    stmt_calls: list[str],
    attention_inputs: dict[str, list[str]],
    forward_step_details: dict[str, list[str]],
) -> None:
    if len(attention_inputs) >= 2:
        return
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        producers = _collect_kernel_producers(call, var_chains)
        if not producers:
            continue
        is_named_kernel = _is_kernel_merge_call(call.func)
        is_generic_external = (
            not isinstance(call.func, ast.Attribute)
            and len(producers) >= 3
            and _expr_name(call.func) is not None
        )
        if not (is_named_kernel or is_generic_external):
            continue
        if _is_data_movement_call(call.func):
            continue
        if SYNTHETIC_ATTENTION not in stmt_calls:
            stmt_calls.append(SYNTHETIC_ATTENTION)
        attention_inputs.update(producers)
        forward_step_details[SYNTHETIC_ATTENTION] = _kernel_call_detail_lines(call)
        return


def _dedupe_kernel_merge_calls(calls: list[str]) -> list[str]:
    if SYNTHETIC_ATTENTION not in calls:
        return calls
    first = calls.index(SYNTHETIC_ATTENTION)
    without = [call for call in calls if call != SYNTHETIC_ATTENTION]
    without.insert(first, SYNTHETIC_ATTENTION)
    return without


def kernel_name_from_step_details(details: list[str]) -> str | None:
    for item in details:
        if item.startswith("kernel:"):
            return item.split(":", 1)[1].strip()
    return None


# Attention that torch itself provides: SDPA, torch.nn.attention, and the plain
# matmul/softmax eager path.
_TORCH_NATIVE_ATTENTION_MARKERS = (
    "sdpa",
    "scaled_dot_product",
    "eager",
    "flex_attention",
    "dot_product_attention",
    "multi_head_attention",
    "torch.nn.attention",
    "nn.attention",
)
# Attention from an outside library, which runs its own fused GPU kernel.
_LIBRARY_ATTENTION_MARKERS = (
    "flash_attn",
    "flash_attention",
    "transformer_engine",
    "transformerengine",
    "fused_attention",
    "fused_attn",
    "memory_efficient_attention",
    "paged_attention",
    "xformers",
)
_STANDARD_ATTENTION_MARKERS = (
    *_TORCH_NATIVE_ATTENTION_MARKERS,
    *_LIBRARY_ATTENTION_MARKERS,
    "attention_interface",
)


def is_standard_attention_kernel(kernel: str | None) -> bool:
    """True for kernels that delegate to a common attention library (SDPA, Flash, TE, …)."""
    if not kernel:
        return False
    lowered = kernel.lower()
    if lowered in _SYNTHETIC_ATTENTION_NAMES:
        return True
    return any(marker in lowered for marker in _STANDARD_ATTENTION_MARKERS)


def is_torch_native_attention_kernel(kernel: str | None) -> bool:
    """True only for attention torch ships itself, as opposed to a library kernel.

    Flash-attn, xformers and Transformer Engine are recognizable attention, but they
    are still outside fused kernels rather than torch operations.
    """
    if not kernel:
        return False
    lowered = kernel.lower()
    if any(marker in lowered for marker in _LIBRARY_ATTENTION_MARKERS):
        return False
    if lowered in {"eager_attention_forward", "sdpa_attention_forward"}:
        return True
    return any(marker in lowered for marker in _TORCH_NATIVE_ATTENTION_MARKERS)


def is_standard_attention_step(details: list[str]) -> bool:
    return is_standard_attention_kernel(kernel_name_from_step_details(details))


def is_kernel_pipeline_step(
    details: list[str],
    attention_inputs: dict[str, list[str]] | None = None,
) -> bool:
    """True when a synthetic attention step has an importable multi-input kernel pipeline."""
    from TraceLens.ModelUtils.kernel_pipeline import parse_kernel_import

    if not kernel_name_from_step_details(details):
        return False
    if parse_kernel_import(details) is None:
        return False
    kwarg_tensors = sum(
        1
        for line in details
        if line.startswith("kwarg:")
        and "=" in line
        and not line.split("=", 1)[1].strip().startswith("self.")
    )
    inputs = attention_inputs or {}
    return len(inputs) >= 2 or kwarg_tensors >= 2


def attention_kernel_label(details: list[str]) -> str:
    kernel = kernel_name_from_step_details(details)
    if is_standard_attention_kernel(kernel):
        return "Attention"
    if kernel:
        return kernel
    return "Attention"


def attention_kernel_details(
    details: list[str],
    attention_inputs: dict[str, list[str]] | None = None,
) -> list[str]:
    kernel = kernel_name_from_step_details(details)

    if kernel and kernel.lower() in _SYNTHETIC_ATTENTION_NAMES:
        return []

    if kernel:
        lines = [f"kernel: {kernel}"]
        if attention_inputs:
            lines.append(f"inputs: {','.join(attention_inputs.keys())}")
        return lines

    return []


def _capture_attention_inputs(
    node: ast.AST,
    var_chains: dict[str, list[str]],
    attention_inputs: dict[str, list[str]],
) -> None:
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        target = _expr_name(call.func)
        if target not in _SYNTHETIC_ATTENTION_NAMES:
            continue
        start = (
            1
            if call.args
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "self"
            else 0
        )
        for index, arg in enumerate(call.args[start:], start=start):
            if not isinstance(arg, ast.Name):
                continue
            chain = var_chains.get(arg.id, [])
            if chain:
                attention_inputs[arg.id] = list(chain)
        return


def _arg_name(arg: ast.AST, index: int) -> str:
    if isinstance(arg, ast.Subscript):
        return _arg_name(arg.value, index)
    if isinstance(arg, ast.Name):
        return arg.id
    return f"arg{index}"


def _is_forward_input_ref(
    name: str,
    var_chains: dict[str, list[str]],
    forward_input_names: set[str],
) -> bool:
    if name in forward_input_names:
        chain = var_chains.get(name)
        if chain:
            return False
        return True
    if name in var_chains and not var_chains[name]:
        return True
    return False


def _arg_provenance(
    arg: ast.AST,
    var_chains: dict[str, list[str]],
    forward_input_names: set[str],
) -> tuple[list[str], SideInputSource | None]:
    if isinstance(arg, ast.Subscript):
        return _arg_provenance(arg.value, var_chains, forward_input_names)
    if not isinstance(arg, ast.Name):
        return [], None
    if _is_forward_input_ref(arg.id, var_chains, forward_input_names):
        return [], "forward_input"
    chain = list(var_chains.get(arg.id, []))
    if chain:
        return chain, "prior_step"
    return [], None


def _side_port_label(
    arg: ast.AST,
    *,
    arg_index: int,
    source_chain: list[str],
    source_kind: SideInputSource,
    callee: str,
) -> str:
    if source_kind == "forward_input":
        return _arg_name(arg, arg_index)
    if isinstance(arg, ast.Name):
        lowered = arg.id.lower()
        if "topk" in lowered or lowered in {"topk_idx", "topk_weight"}:
            if "weight" in lowered:
                return "top_k_weights"
            return "top_k_index"
        if lowered == "router_logits":
            return "router_logits"
    if source_chain and _classify_role(source_chain[-1], "") == "router":
        return "router"
    return _arg_name(arg, arg_index)


def _capture_call_side_inputs(
    node: ast.AST,
    var_chains: dict[str, list[str]],
    forward_input_names: set[str],
    side_inputs: dict[str, list[SideInputSpec]],
    prior_calls: list[str],
) -> None:
    """Record non-primary arguments that bypass the sequential main path."""
    discarded_call = node.value if isinstance(node, ast.Expr) else None
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Attribute) or not _is_self_attr(
            call.func, call.func.attr
        ):
            continue
        callee = call.func.attr
        if callee.startswith("@") or callee in _SYNTHETIC_ATTENTION_NAMES:
            continue
        if not call.args:
            continue

        main_chain, main_kind = _arg_provenance(
            call.args[0], var_chains, forward_input_names
        )
        specs: list[SideInputSpec] = []
        seen: set[tuple[str, tuple[str, ...], SideInputSource]] = set()

        if main_kind == "forward_input" and prior_calls and callee != prior_calls[0]:
            arg0_name = _arg_name(call.args[0], 0)
            if arg0_name not in forward_input_names or call is discarded_call:
                key = (arg0_name, tuple(), "forward_input")
                if key not in seen:
                    seen.add(key)
                    specs.append(
                        SideInputSpec(
                            arg_name=arg0_name,
                            port_label=arg0_name,
                            source_chain=[],
                            source_kind="forward_input",
                            side_effect_call=call is discarded_call,
                        )
                    )

        for arg_index, arg in enumerate(call.args[1:], start=1):
            chain, source_kind = _arg_provenance(arg, var_chains, forward_input_names)
            if source_kind is None:
                continue
            if source_kind == "prior_step" and chain == main_chain:
                continue
            if source_kind == "forward_input" and main_kind == "forward_input":
                continue
            port_label = _side_port_label(
                arg,
                arg_index=arg_index,
                source_chain=chain,
                source_kind=source_kind,
                callee=callee,
            )
            key = (port_label, tuple(chain), source_kind)
            if key in seen:
                continue
            seen.add(key)
            specs.append(
                SideInputSpec(
                    arg_name=(
                        port_label
                        if port_label in {"top_k_index", "top_k_weights"}
                        else _arg_name(arg, arg_index)
                    ),
                    port_label=port_label,
                    source_chain=chain,
                    source_kind=source_kind,
                )
            )

        if not specs:
            continue
        existing = side_inputs.setdefault(callee, [])
        for spec in specs:
            duplicate = any(
                item.port_label == spec.port_label
                and item.source_chain == spec.source_chain
                and item.source_kind == spec.source_kind
                and item.side_effect_call == spec.side_effect_call
                for item in existing
            )
            if not duplicate:
                existing.append(spec)


def _capture_augassign_module_input(
    node: ast.AugAssign,
    var_chains: dict[str, list[str]],
    forward_input_names: set[str],
    side_inputs: dict[str, list[SideInputSpec]],
) -> None:
    """Keep a module's input branch when its output is accumulated with ``+=``."""
    for call in ast.walk(node.value):
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or not _is_self_attr(call.func, call.func.attr)
            or not call.args
        ):
            continue
        _chain, source_kind = _arg_provenance(
            call.args[0],
            var_chains,
            forward_input_names,
        )
        if source_kind != "forward_input":
            continue
        callee = call.func.attr
        arg_name = _arg_name(call.args[0], 0)
        spec = SideInputSpec(
            arg_name=arg_name,
            port_label=arg_name,
            source_chain=[],
            source_kind="forward_input",
        )
        existing = side_inputs.setdefault(callee, [])
        if not any(
            item.port_label == spec.port_label
            and item.source_chain == spec.source_chain
            and item.source_kind == spec.source_kind
            for item in existing
        ):
            existing.append(spec)


def _forward_input_names(func: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    args = func.args
    for arg in args.posonlyargs + args.args:
        if arg.arg != "self":
            names.add(arg.arg)
    return names


def _primary_forward_input_name(func: ast.FunctionDef) -> str | None:
    """Return the first forward parameter name (typically the hidden-state tensor)."""
    for arg in func.args.posonlyargs + func.args.args:
        if arg.arg != "self":
            return arg.arg
    return None


def kernel_kwarg_ports(details: list[str]) -> dict[str, str]:
    """Map kwarg parameter names to variable names from modeling AST kwarg lines."""
    ports: dict[str, str] = {}
    for line in details:
        if not line.startswith("kwarg:"):
            continue
        payload = line.split(":", 1)[1].strip()
        if "=" not in payload:
            continue
        param, value = payload.split("=", 1)
        param = param.strip()
        value = value.strip()
        if value and not value.startswith("self."):
            ports[param] = value
    return ports


def tensor_input_label_order(
    details: list[str],
    attention_inputs: dict[str, list[str]],
) -> list[str]:
    """Order tensor input labels from kwarg AST lines, then remaining provenance keys."""
    ordered: list[str] = []
    seen: set[str] = set()
    for line in details:
        if not line.startswith("kwarg:"):
            continue
        payload = line.split(":", 1)[1].strip()
        if "=" not in payload:
            continue
        param = payload.split("=", 1)[0].strip()
        if param in attention_inputs and param not in seen:
            ordered.append(param)
            seen.add(param)
    for key in attention_inputs:
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _parallel_gates_from_forward(func: ast.FunctionDef) -> list[str]:
    """Modules invoked directly on forward inputs (e.g. output gate from hidden_states)."""
    input_names = _forward_input_names(func)
    gates: list[str] = []
    for node in func.body:
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, ast.Attribute) or not _is_self_attr(
                call.func, call.func.attr
            ):
                continue
            if not call.args or not isinstance(call.args[0], ast.Name):
                continue
            if call.args[0].id not in input_names:
                continue
            attr = call.func.attr
            if attr in gates:
                continue
            if re.search(r"gate|g_proj", attr, re.I):
                gates.append(attr)
    return gates


def _input_fed_calls_from_forward(func: ast.FunctionDef) -> list[str]:
    """Submodule calls whose main argument is still the value the forward received.

    Such a call reads the forward input, not the result of the call before it, so the
    chain has to branch at the input rather than run the two steps in series. A name
    stops counting once it has been rebound to something a submodule produced;
    reshapes and views of the input still are the input.
    """
    pristine = set(_forward_input_names(func))
    if not pristine:
        return []
    fed: list[str] = []
    for stmt in func.body:
        for call in ast.walk(stmt):
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, ast.Attribute) or not _is_self_attr(
                call.func, call.func.attr
            ):
                continue
            if not call.args or not isinstance(call.args[0], ast.Name):
                continue
            if call.args[0].id in pristine and call.func.attr not in fed:
                fed.append(call.func.attr)
        # Read the arguments before the targets rebind, so `x = self.block(x)` still
        # counts as reading the input rather than the value it is about to hold.
        value = (
            _stmt_value(stmt) if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
        )
        if value is None:
            continue
        produced: list[str] = []
        _extract_self_calls_ordered(value, produced)
        if not produced:
            continue
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for target in targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name):
                    pristine.discard(name.id)
    return fed


def _parallel_gate_activation(func: ast.FunctionDef, gate_attr: str) -> str | None:
    """Detect activation applied to a parallel output gate (e.g. g_proj(...).sigmoid())."""
    gate_vars: set[str] = set()
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                if _is_self_attr(value.func, gate_attr):
                    gate_vars.add(target)
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
            if isinstance(target, ast.Name):
                value = node.value
                if isinstance(value, ast.Call) and isinstance(
                    value.func, ast.Attribute
                ):
                    if _is_self_attr(value.func, gate_attr):
                        gate_vars.add(target.id)

    for node in ast.walk(func):
        src = (
            _stmt_value(node)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return))
            else None
        )
        if src is None:
            continue
        if isinstance(src, ast.Call) and isinstance(src.func, ast.Attribute):
            activation = _GATE_ACTIVATION_NAMES.get(src.func.attr)
            if activation is None:
                continue
            inner = src.func.value
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                if _is_self_attr(inner.func, gate_attr):
                    return activation
            if isinstance(inner, ast.Name) and inner.id in gate_vars:
                return activation
    return None


def _parallel_gate_activations_from_forward(
    func: ast.FunctionDef,
    parallel_gates: list[str],
) -> dict[str, str]:
    activations: dict[str, str] = {}
    for gate_attr in parallel_gates:
        activation = _parallel_gate_activation(func, gate_attr)
        if activation:
            activations[gate_attr] = activation
    return activations


def _parse_forward(
    func: ast.FunctionDef,
) -> tuple[
    list[str],
    list[str],
    dict[str, list[str]],
    dict[str, list[SideInputSpec]],
    dict[str, list[str]],
]:
    calls: list[str] = []
    norm_before: list[str] = []
    pending_norm: str | None = None
    var_chains: dict[str, list[str]] = {}
    attention_inputs: dict[str, list[str]] = {}
    side_inputs: dict[str, list[SideInputSpec]] = {}
    forward_step_details: dict[str, list[str]] = {}
    forward_input_names = _forward_input_names(func)

    for node in func.body:
        pending_norm = _walk_forward_stmt(
            node,
            calls,
            norm_before,
            pending_norm,
            var_chains,
            attention_inputs,
            side_inputs,
            forward_input_names,
            forward_step_details,
        )
    forward_step_details.update(_positional_step_details(func))
    return (
        _dedupe_kernel_merge_calls(calls),
        norm_before,
        attention_inputs,
        side_inputs,
        forward_step_details,
    )


def _walk_forward_stmt(
    node: ast.AST,
    calls: list[str],
    norm_before: list[str],
    pending_norm: str | None,
    var_chains: dict[str, list[str]],
    attention_inputs: dict[str, list[str]],
    side_inputs: dict[str, list[SideInputSpec]],
    forward_input_names: set[str],
    forward_step_details: dict[str, list[str]],
) -> str | None:
    if isinstance(node, ast.Assign):
        stmt_calls: list[str] = []
        _extract_self_calls_ordered(node.value, stmt_calls)
        _inject_kernel_merge(
            node.value,
            var_chains,
            stmt_calls,
            attention_inputs,
            forward_step_details,
        )
        _capture_attention_inputs(node, var_chains, attention_inputs)
        _capture_call_side_inputs(
            node, var_chains, forward_input_names, side_inputs, calls
        )
        # Read RHS provenance before rebinding assignment targets. In
        # ``x = self.block(x, aux)`` the first ``x`` is still the forward input;
        # treating the newly produced chain as its source invents a residual merge.
        _record_assign_targets(node, stmt_calls, var_chains)
        return _register_forward_calls(stmt_calls, calls, norm_before, pending_norm)

    if isinstance(node, ast.AnnAssign) and node.value is not None:
        stmt_calls = []
        _extract_self_calls_ordered(node.value, stmt_calls)
        _inject_kernel_merge(
            node.value,
            var_chains,
            stmt_calls,
            attention_inputs,
            forward_step_details,
        )
        _capture_attention_inputs(node, var_chains, attention_inputs)
        _capture_call_side_inputs(
            node, var_chains, forward_input_names, side_inputs, calls
        )
        if isinstance(node.target, ast.Name):
            chain = _trace_var_chain(node.value, var_chains, stmt_calls)
            if chain:
                var_chains[node.target.id] = chain
        return _register_forward_calls(stmt_calls, calls, norm_before, pending_norm)

    if isinstance(node, ast.Expr):
        stmt_calls = []
        _extract_self_calls_ordered(node.value, stmt_calls)
        _inject_kernel_merge(
            node.value,
            var_chains,
            stmt_calls,
            attention_inputs,
            forward_step_details,
        )
        _capture_attention_inputs(node, var_chains, attention_inputs)
        _capture_call_side_inputs(
            node, var_chains, forward_input_names, side_inputs, calls
        )
        return _register_forward_calls(stmt_calls, calls, norm_before, pending_norm)

    if isinstance(node, ast.AugAssign):
        stmt_calls = []
        _extract_self_calls_ordered(node.value, stmt_calls)
        _inject_kernel_merge(
            node.value,
            var_chains,
            stmt_calls,
            attention_inputs,
            forward_step_details,
        )
        _capture_call_side_inputs(
            node, var_chains, forward_input_names, side_inputs, calls
        )
        _capture_augassign_module_input(
            node,
            var_chains,
            forward_input_names,
            side_inputs,
        )
        return _register_forward_calls(stmt_calls, calls, norm_before, pending_norm)

    if isinstance(node, ast.Return) and node.value is not None:
        stmt_calls = []
        _extract_self_calls_ordered(node.value, stmt_calls)
        _inject_kernel_merge(
            node.value,
            var_chains,
            stmt_calls,
            attention_inputs,
            forward_step_details,
        )
        _capture_attention_inputs(node, var_chains, attention_inputs)
        _capture_call_side_inputs(
            node, var_chains, forward_input_names, side_inputs, calls
        )
        return _register_forward_calls(stmt_calls, calls, norm_before, pending_norm)

    if isinstance(node, ast.If):
        branch = node.body + node.orelse
        for child in branch:
            pending_norm = _walk_forward_stmt(
                child,
                calls,
                norm_before,
                pending_norm,
                var_chains,
                attention_inputs,
                side_inputs,
                forward_input_names,
                forward_step_details,
            )
        return pending_norm

    if isinstance(node, ast.For):
        first_loop_call = len(calls)
        for child in node.body:
            pending_norm = _walk_forward_stmt(
                child,
                calls,
                norm_before,
                pending_norm,
                var_chains,
                attention_inputs,
                side_inputs,
                forward_input_names,
                forward_step_details,
            )
        # Tensor operations are annotated by _ForwardOperationExtractor, but
        # expanded helper calls (for example `_apply_gate()`) are not operations
        # in this method's graph. Preserve their call-site loop context too so
        # their expanded children remain inside the source loop.
        for call in calls[first_loop_call:]:
            details = forward_step_details.setdefault(call, [])
            if not any(detail.startswith("loop:") for detail in details):
                details.append("loop: repeated")
        return pending_norm

    if isinstance(node, ast.With):
        for child in node.body:
            pending_norm = _walk_forward_stmt(
                child,
                calls,
                norm_before,
                pending_norm,
                var_chains,
                attention_inputs,
                side_inputs,
                forward_input_names,
                forward_step_details,
            )
        return pending_norm

    return pending_norm


def _decoder_class_score(info: ClassStructure) -> int:
    score = 0
    if DECODER_CLASS_RE.search(info.name):
        score += 10
    if any(
        _classify_role(a, c) == "attention" for a, c in info.init_assignments.items()
    ):
        score += 5
    if any(
        _classify_role(a, c) in {"ffn", "moe"} for a, c in info.init_assignments.items()
    ):
        score += 3
    if info.forward_calls:
        score += 2
    return score


def _pick_decoder_class(classes: dict[str, ClassStructure]) -> ClassStructure | None:
    ranked: list[tuple[int, ClassStructure]] = []
    for info in classes.values():
        score = _decoder_class_score(info)
        if score > 0:
            ranked.append((score, info))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def _model_class_score(info: ClassStructure) -> int:
    """Rank stack/backbone classes so a vision tower cannot beat the language model.

    Name matching alone is not enough: multimodal repos name both the ViT and the
    text backbone ``*Model`` / ``*PreTrainedModel``, and dict order follows file
    order. Owning an embedding plus a decoder-layer child is the language stack.
    """
    if not info.init_assignments:
        return 0
    score = 0
    if MODEL_CLASS_RE.search(info.name):
        score += 10
    roles = {
        _classify_role(attr, class_name)
        for attr, class_name in info.init_assignments.items()
    }
    if "embedding" in roles:
        score += 5
    if "head" in roles:
        score += 3
    if "norm" in roles:
        score += 1
    if any(
        DECODER_CLASS_RE.search(class_name)
        for class_name in info.init_assignments.values()
    ):
        score += 8
    return score


def _pick_model_class(classes: dict[str, ClassStructure]) -> ClassStructure | None:
    ranked: list[tuple[int, ClassStructure]] = []
    for info in classes.values():
        score = _model_class_score(info)
        if score > 0:
            ranked.append((score, info))
    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]
    return _pick_model_class_by_structure(classes)


def _pick_model_class_by_structure(
    classes: dict[str, ClassStructure],
) -> ClassStructure | None:
    """Find the class that owns the stack when its name follows no known convention.

    Inference repos often name it plainly (`Transformer`), so the token embedding it
    owns, rather than its name, is what identifies it.
    """
    ranked: list[tuple[int, str, ClassStructure]] = []
    for info in classes.values():
        if DECODER_CLASS_RE.search(info.name) or not info.init_assignments:
            continue
        roles = {
            _classify_role(attr, class_name)
            for attr, class_name in info.init_assignments.items()
        }
        if "embedding" not in roles:
            continue
        ranked.append((int("head" in roles) + int("norm" in roles), info.name, info))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][2]


def _pick_causal_lm_class(classes: dict[str, ClassStructure]) -> ClassStructure | None:
    for info in classes.values():
        if info.name.endswith("ForCausalLM") or "ForCausalLM" in info.name:
            return info
    return None


def _pick_stack_model_class(
    classes: dict[str, ClassStructure],
    causal_lm: ClassStructure | None,
) -> ClassStructure | None:
    if causal_lm is not None:
        model_attr = causal_lm.init_assignments.get("model")
        if model_attr and model_attr in classes:
            return classes[model_attr]
        transform_attr = causal_lm.init_assignments.get("transformer")
        if transform_attr and transform_attr in classes:
            return classes[transform_attr]
    return _pick_model_class(classes)


def _is_positional_module(attr_name: str, class_name: str) -> bool:
    return _classify_role(attr_name, class_name) == "positional"


def _find_positional_module(
    registry: dict[str, ClassStructure],
    stack_model: ClassStructure | None,
    decoder: ClassStructure | None,
) -> tuple[str, str] | None:
    """Locate a rotary/positional submodule declared in modeling code."""
    search_roots: list[ClassStructure] = []
    if stack_model is not None:
        search_roots.append(stack_model)
    if decoder is not None:
        search_roots.append(decoder)

    for root in search_roots:
        order = set(root.forward_calls)
        for attr, class_name in root.init_assignments.items():
            if class_name in _SKIP_INIT_CLASS_NAMES:
                continue
            if not _is_positional_module(attr, class_name):
                continue
            if order and attr not in order:
                continue
            return attr, class_name

    for info in registry.values():
        for attr, class_name in info.init_assignments.items():
            if class_name in _SKIP_INIT_CLASS_NAMES:
                continue
            if _is_positional_module(attr, class_name):
                return attr, class_name
    return None


def _stack_component(
    *,
    attr_name: str,
    class_name: str,
    role: str,
    forward_order: int | None,
    details: list[str] | None = None,
) -> BlockComponent:
    return BlockComponent(
        attr_name=attr_name,
        class_name=class_name,
        role=role,
        label=_label_for(role, class_name, attr_name),
        forward_order=forward_order,
        details=list(details or []),
    )


def build_stack_components(
    *,
    stack_model: ClassStructure | None,
    causal_lm: ClassStructure | None,
    decoder: ClassStructure | None,
    registry: dict[str, ClassStructure],
) -> tuple[list[BlockComponent], list[BlockComponent]]:
    """Build pre-decoder and post-decoder stack segments from model AST."""
    pre: list[BlockComponent] = []
    tail: list[BlockComponent] = []

    if stack_model is not None:
        order = {attr: idx for idx, attr in enumerate(stack_model.forward_calls)}
        for attr, class_name in stack_model.init_assignments.items():
            if class_name in _SKIP_INIT_CLASS_NAMES:
                continue
            role = _classify_role(attr, class_name)
            if role != "embedding":
                continue
            pre.append(
                _stack_component(
                    attr_name=attr,
                    class_name=class_name,
                    role=role,
                    forward_order=order.get(attr, 0),
                    details=stack_model.init_details.get(attr, []),
                )
            )

        positional = _find_positional_module(registry, stack_model, decoder)
        if positional is not None:
            attr, class_name = positional
            pre.append(
                _stack_component(
                    attr_name=attr,
                    class_name=class_name,
                    role="positional",
                    forward_order=order.get(attr, 1),
                    details=registry.get(class_name, stack_model).init_details.get(
                        attr, []
                    ),
                )
            )

        if "norm" in stack_model.init_assignments:
            attr = "norm"
            class_name = stack_model.init_assignments[attr]
            tail.append(
                _stack_component(
                    attr_name=attr,
                    class_name=class_name,
                    role="norm",
                    forward_order=order.get(attr),
                    details=stack_model.init_details.get(attr, []),
                )
            )

    # Inference repos without a ForCausalLM wrapper hang the head off the stack itself.
    head_owner = causal_lm if causal_lm is not None else stack_model
    if head_owner is not None:
        order = {attr: idx for idx, attr in enumerate(head_owner.forward_calls)}
        for attr, class_name in head_owner.init_assignments.items():
            if class_name in _SKIP_INIT_CLASS_NAMES:
                continue
            if _classify_role(attr, class_name) != "head":
                continue
            tail.append(
                _stack_component(
                    attr_name=attr,
                    class_name=class_name,
                    role="head",
                    forward_order=order.get(attr),
                    details=head_owner.init_details.get(attr, []),
                )
            )

    pre.sort(
        key=lambda comp: (
            {"embedding": 0, "positional": 1}.get(comp.role, 99),
            comp.forward_order if comp.forward_order is not None else 999,
            comp.attr_name,
        )
    )
    tail.sort(
        key=lambda comp: (
            comp.forward_order is None,
            comp.forward_order if comp.forward_order is not None else 999,
            {"norm": 0, "head": 1}.get(comp.role, 99),
            comp.attr_name,
        )
    )
    return pre, tail


def _infer_attention_type_from_class(
    info: ClassStructure | None, all_classes: dict[str, ClassStructure]
) -> str | None:
    if info is None:
        return None

    attn_attr = next(
        (
            attr
            for attr, cls in info.init_assignments.items()
            if _classify_role(attr, cls) == "attention"
        ),
        None,
    )
    if not attn_attr:
        return None

    class_name = info.init_assignments[attn_attr]
    if re.search(r"Latent|MLA", class_name, re.I):
        return "MLA"

    attn_class = all_classes.get(class_name)
    if attn_class:
        joined = " ".join(
            f"{attr} {cls}" for attr, cls in attn_class.init_assignments.items()
        )
        if re.search(r"kv_lora|q_lora|latent", joined, re.I):
            return "MLA"
        if re.search(r"num_key_value_heads|k_proj", joined, re.I):
            # Can't know GQA vs MHA from AST alone unless config merged later.
            pass

    if re.search(r"Grouped|GQA", class_name, re.I):
        return "GQA"
    if re.search(r"MultiQuery|MQA", class_name, re.I):
        return "MQA"
    return None


def _infer_norm_from_ast(decoder: ClassStructure) -> tuple[str | None, str | None]:
    norm_classes = [
        cls
        for attr, cls in decoder.init_assignments.items()
        if _classify_role(attr, cls) == "norm"
    ]
    norm_type = None
    if any("RMS" in cls for cls in norm_classes):
        norm_type = "RMSNorm"
    elif norm_classes:
        norm_type = "LayerNorm"

    placement = None
    if decoder.norm_before:
        placement = "Pre-Norm"
    elif decoder.forward_calls and norm_classes:
        # If norms appear in init but never immediately precede modules in forward,
        # assume post-norm style wiring.
        norm_attrs = {
            a
            for a, c in decoder.init_assignments.items()
            if _classify_role(a, c) == "norm"
        }
        first_module = decoder.forward_calls[0] if decoder.forward_calls else None
        if first_module and first_module not in norm_attrs:
            placement = "Post-Norm (inside residual)"
        else:
            placement = "Pre-Norm"
    return norm_type, placement


def _build_components(decoder: ClassStructure) -> list[BlockComponent]:
    components: list[BlockComponent] = []
    order_map = {attr: idx for idx, attr in enumerate(decoder.forward_calls)}
    forward_attrs = set(decoder.forward_calls)

    for attr, class_name in decoder.init_assignments.items():
        if class_name in _SKIP_INIT_CLASS_NAMES:
            continue
        if attr not in forward_attrs:
            continue
        role = _classify_role(attr, class_name)
        if role in {"router"} and attr not in decoder.forward_calls:
            continue
        label = _label_for(role, class_name, attr)
        components.append(
            BlockComponent(
                attr_name=attr,
                class_name=class_name,
                role=role,
                label=label,
                forward_order=order_map.get(attr),
                details=decoder.init_details.get(attr, []),
            )
        )

    for index, attr in enumerate(decoder.forward_calls):
        if attr == SYNTHETIC_ATTENTION:
            step_details = decoder.forward_step_details.get(attr, [])
            components.append(
                BlockComponent(
                    attr_name=attr,
                    class_name="AttentionOp",
                    role="attention",
                    label=attention_kernel_label(step_details),
                    forward_order=index,
                    details=attention_kernel_details(
                        step_details, decoder.attention_inputs
                    ),
                )
            )
            continue
        if attr in decoder.init_assignments:
            continue
        if _inline_forward_step(attr):
            continue
        role = _classify_role(attr, attr)
        components.append(
            BlockComponent(
                attr_name=attr,
                class_name=attr,
                role=role,
                label=attr.replace("_", " "),
                forward_order=index,
                details=[f"method `{attr}()`"],
            )
        )

    components.sort(
        key=lambda comp: (
            comp.forward_order is None,
            comp.forward_order if comp.forward_order is not None else 999,
            comp.attr_name,
        )
    )
    return components


def decoder_type_for_components(components: list[BlockComponent]) -> str | None:
    """Name the decoder flavor implied by the roles of a layer's submodules."""
    roles = {comp.role for comp in components}
    if "moe" in roles:
        return "Sparse MoE"
    if len([comp for comp in components if comp.role == "attention"]) > 1:
        return "Hybrid"
    if "ffn" in roles:
        return "Dense"
    return None


def expand_conditional_block_components(
    decoder: ClassStructure,
    components: list[BlockComponent],
) -> list[BlockComponent]:
    """Include alternate submodule classes selected by conditional __init__ branches."""
    expanded: list[BlockComponent] = []
    seen: set[tuple[str, str]] = set()
    order_map = {comp.attr_name: comp.forward_order for comp in components}
    ffn_order = order_map.get("block_sparse_moe")
    if ffn_order is None:
        ffn_order = order_map.get("mlp")

    for comp in components:
        class_names = decoder.init_assignment_options.get(comp.attr_name) or [
            comp.class_name
        ]
        for class_name in class_names:
            key = (comp.attr_name, class_name)
            if key in seen:
                continue
            seen.add(key)
            if class_name == comp.class_name:
                expanded.append(comp)
                continue
            role = _classify_role(comp.attr_name, class_name)
            expanded.append(
                BlockComponent(
                    attr_name=comp.attr_name,
                    class_name=class_name,
                    role=role,
                    label=_label_for(role, class_name, comp.attr_name),
                    forward_order=order_map.get(comp.attr_name),
                    details=list(decoder.init_details.get(comp.attr_name, [])),
                )
            )

    for attr in ("mlp", "block_sparse_moe"):
        if attr in order_map:
            continue
        class_names = decoder.init_assignment_options.get(attr, [])
        for class_name in class_names:
            key = (attr, class_name)
            if key in seen:
                continue
            seen.add(key)
            role = _classify_role(attr, class_name)
            expanded.append(
                BlockComponent(
                    attr_name=attr,
                    class_name=class_name,
                    role=role,
                    label=_label_for(role, class_name, attr),
                    forward_order=ffn_order,
                    details=list(decoder.init_details.get(attr, [])),
                )
            )
    expanded.sort(
        key=lambda comp: (
            comp.forward_order is None,
            comp.forward_order if comp.forward_order is not None else 999,
            comp.attr_name,
            comp.class_name,
        )
    )
    return expanded


def _unparse_expr(node: ast.AST) -> str:
    if hasattr(ast, "unparse"):
        return ast.unparse(node)
    return ""


def _class_init_method(class_node: ast.ClassDef) -> ast.FunctionDef | None:
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            return item
    return None


def _is_module_list(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Name) and node.func.id == "ModuleList")
        or (isinstance(node.func, ast.Attribute) and node.func.attr == "ModuleList")
    )


def _parse_layer_module_list(value: ast.AST) -> tuple[str, str, str] | None:
    list_comp: ast.ListComp | None = None
    if (
        _is_module_list(value)
        and value.args
        and isinstance(value.args[0], ast.ListComp)
    ):
        list_comp = value.args[0]
    elif isinstance(value, ast.ListComp):
        list_comp = value
    if list_comp is None or not isinstance(list_comp.elt, ast.Call):
        return None
    decoder_class = _call_class_name(list_comp.elt)
    if not decoder_class or not list_comp.generators:
        return None
    gen = list_comp.generators[0]
    loop_var = gen.target.id if isinstance(gen.target, ast.Name) else None
    if not loop_var:
        return None
    count_expr = _unparse_expr(gen.iter)
    return decoder_class, loop_var, count_expr


def _find_layer_loop_in_init(
    init_func: ast.FunctionDef,
) -> tuple[str, str, str, str] | None:
    for node in init_func.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Attribute) or not _is_self_attr(
                target, target.attr
            ):
                continue
            parsed = _parse_layer_module_list(node.value)
            if parsed:
                decoder_class, loop_var, count_expr = parsed
                return target.attr, decoder_class, loop_var, count_expr
    return None


def _assignment_targets_role_attr(stmt: ast.Assign) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for target in stmt.targets:
        if isinstance(target, ast.Attribute) and _is_self_attr(target, target.attr):
            class_name = _assignment_class_name(stmt.value)
            if class_name:
                found.append((target.attr, class_name))
    return found


def _if_chain_nodes(if_node: ast.If) -> list[ast.If]:
    chain = [if_node]
    cursor = if_node
    while len(cursor.orelse) == 1 and isinstance(cursor.orelse[0], ast.If):
        cursor = cursor.orelse[0]
        chain.append(cursor)
    return chain


def _if_chain_references_layer_idx(if_node: ast.If) -> bool:
    for node in _if_chain_nodes(if_node):
        if "layer_idx" in _unparse_expr(node.test):
            return True
        for stmt in node.body:
            if isinstance(stmt, ast.If) and _if_chain_references_layer_idx(stmt):
                return True
    return False


def _collect_layer_init_conditionals(
    if_node: ast.If,
) -> list[tuple[str, str, str]]:
    if not _if_chain_references_layer_idx(if_node):
        return []

    results: list[tuple[str, str, str]] = []
    for index, node in enumerate(_if_chain_nodes(if_node)):
        branch = "elif" if index > 0 else "if"
        cond = _unparse_expr(node.test)
        condition_label = f"{branch} {cond}"
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for attr, class_name in _assignment_targets_role_attr(stmt):
                    if _classify_role(attr, class_name) in {"attention", "ffn", "moe"}:
                        results.append((attr, class_name, condition_label))
            elif isinstance(stmt, ast.If):
                results.extend(_collect_layer_init_conditionals(stmt))
    final_else = _if_chain_nodes(if_node)[-1].orelse
    if not (len(final_else) == 1 and isinstance(final_else[0], ast.If)):
        for stmt in final_else:
            if isinstance(stmt, ast.Assign):
                for attr, class_name in _assignment_targets_role_attr(stmt):
                    if _classify_role(attr, class_name) in {"attention", "ffn", "moe"}:
                        results.append((attr, class_name, "else"))
    return results


def _extract_decoder_layer_conditionals(
    decoder: ClassStructure,
) -> list[tuple[str, str, str]]:
    init_func = _class_init_method(decoder.node)
    if init_func is None:
        return []
    results: list[tuple[str, str, str]] = []
    for stmt in init_func.body:
        if isinstance(stmt, ast.If):
            results.extend(_collect_layer_init_conditionals(stmt))
    return results


def build_layer_repeat_lines(
    *,
    stack_model: ClassStructure | None,
    decoder: ClassStructure | None,
    num_layers: int | None = None,
) -> list[str]:
    """Summarize how decoder layers are constructed and selected in __init__."""
    if stack_model is None or decoder is None:
        return []
    init_func = _class_init_method(stack_model.node)
    if init_func is None:
        return []
    loop = _find_layer_loop_in_init(init_func)
    if loop is None:
        return []
    layer_attr, decoder_class, loop_var, count_expr = loop
    if num_layers is not None:
        count_display = str(num_layers)
        range_display = f"range({num_layers})"
    else:
        count_display = "N"
        range_display = count_expr
    lines = [f"{count_display} × {decoder_class} ({loop_var} in {range_display})"]
    del layer_attr
    for attr, class_name, condition in _extract_decoder_layer_conditionals(decoder):
        lines.append(f"{attr} → {class_name} ({condition})")
    return lines


def build_class_registry(
    source: str,
    *,
    filename: str = "<model>",
    config: dict[str, Any] | None = None,
    all_tensor_ops: bool = False,
) -> dict[str, ClassStructure]:
    """Return all class structures discovered in one modeling file."""
    tree = parse_python_ast(source, filename=filename)
    visitor = _ModelAstVisitor(config=config, all_tensor_ops=all_tensor_ops)
    visitor.visit(tree)
    return visitor.classes


def merge_class_registries(
    *registries: dict[str, ClassStructure]
) -> dict[str, ClassStructure]:
    merged: dict[str, ClassStructure] = {}
    for registry in registries:
        merged.update(registry)
    return merged


def analyze_source(
    source: str,
    *,
    filename: str = "<model>",
    config: dict[str, Any] | None = None,
    all_tensor_ops: bool = False,
) -> CodeAnalysis:
    """Analyze one modeling file and return extracted block structure."""
    tree = parse_python_ast(source, filename=filename)
    external_imports = _collect_external_imports(tree)
    visitor = _ModelAstVisitor(config=config, all_tensor_ops=all_tensor_ops)
    visitor.visit(tree)
    finalize_class_registry(visitor.classes)
    _enrich_kernel_import_details(visitor.classes, external_imports)
    _resolve_dispatched_attention_kernel(visitor.classes, config)

    decoder = _pick_decoder_class(visitor.classes)
    causal_lm = _pick_causal_lm_class(visitor.classes)
    stack_model = _pick_stack_model_class(visitor.classes, causal_lm)
    model = stack_model or _pick_model_class(visitor.classes)
    analysis = CodeAnalysis(source_files=[filename])
    analysis.class_registry = dict(visitor.classes)
    analysis.external_imports = dict(external_imports)
    analysis.positional_helpers = _positional_helper_functions(tree)

    if model is not None:
        analysis.model_class = model.name
    if stack_model is not None:
        analysis.stack_model_class = stack_model.name
    if causal_lm is not None:
        analysis.causal_lm_class = causal_lm.name

    if decoder is None:
        analysis.notes.append("No decoder layer class found in AST")
        if stack_model is not None or causal_lm is not None:
            analysis.stack_pre, analysis.stack_tail = build_stack_components(
                stack_model=stack_model,
                causal_lm=causal_lm,
                decoder=None,
                registry=visitor.classes,
            )
        return analysis

    analysis.decoder_class = decoder.name
    analysis.block_components = _build_components(decoder)
    analysis.forward_sequence = list(decoder.forward_calls)
    analysis.stack_pre, analysis.stack_tail = build_stack_components(
        stack_model=stack_model,
        causal_lm=causal_lm,
        decoder=decoder,
        registry=visitor.classes,
    )

    attn_type = _infer_attention_type_from_class(decoder, visitor.classes)
    if attn_type:
        analysis.attention_type = attn_type
        analysis.attention_class = next(
            (
                cls
                for attr, cls in decoder.init_assignments.items()
                if _classify_role(attr, cls) == "attention"
            ),
            None,
        )

    analysis.decoder_type = (
        decoder_type_for_components(analysis.block_components) or analysis.decoder_type
    )

    for comp in analysis.block_components:
        if comp.role == "ffn" and "SwiGLU" in comp.class_name:
            analysis.ffn_type = "SwiGLU"
        if comp.role == "other":
            analysis.custom_blocks.append(comp.class_name)

    norm_type, norm_placement = _infer_norm_from_ast(decoder)
    analysis.norm_type = norm_type
    analysis.norm_placement = norm_placement

    if analysis.custom_blocks:
        analysis.notes.append(
            "Custom blocks: " + ", ".join(sorted(set(analysis.custom_blocks)))
        )
    if analysis.forward_sequence:
        analysis.notes.append("Forward order: " + " → ".join(analysis.forward_sequence))

    analysis.layer_repeat_lines = build_layer_repeat_lines(
        stack_model=stack_model,
        decoder=decoder,
    )

    return analysis


def analyze_sources(
    sources: dict[Path, str],
    *,
    config: dict[str, Any] | None = None,
    all_tensor_ops: bool = False,
) -> CodeAnalysis:
    """Analyze multiple files and merge into one CodeAnalysis."""
    merged = CodeAnalysis()
    registries: list[dict[str, ClassStructure]] = []
    best_decoder_score = 0
    for path, text in sources.items():
        partial = analyze_source(
            text,
            filename=str(path),
            config=config,
            all_tensor_ops=all_tensor_ops,
        )
        registries.append(partial.class_registry)
        merged.source_files.extend(partial.source_files)
        merged.notes.extend(partial.notes)
        merged.external_imports.update(partial.external_imports)
        merged.positional_helpers.extend(partial.positional_helpers)

        # Multimodal repos ship several modeling files; the language decoder can live
        # in any of them, so rank candidates across files instead of taking the first.
        decoder_info = partial.class_registry.get(partial.decoder_class or "")
        decoder_score = _decoder_class_score(decoder_info) if decoder_info else 0
        if partial.decoder_class and decoder_score > best_decoder_score:
            best_decoder_score = decoder_score
            merged.decoder_class = partial.decoder_class
            merged.block_components = partial.block_components
            merged.forward_sequence = partial.forward_sequence
            merged.layer_repeat_lines = list(partial.layer_repeat_lines)
            merged.attention_class = partial.attention_class
            merged.attention_type = partial.attention_type
            merged.decoder_type = partial.decoder_type
            merged.ffn_type = partial.ffn_type
            merged.norm_type = partial.norm_type
            merged.norm_placement = partial.norm_placement
            merged.custom_blocks = list(partial.custom_blocks)

    merged.class_registry = merge_class_registries(*registries)
    # Re-pick graph-owning classes from the combined registry. First-file-wins
    # lets a vision modeling file stamp a ViT backbone (or leave these unset).
    causal_lm = _pick_causal_lm_class(merged.class_registry)
    stack_model = _pick_stack_model_class(merged.class_registry, causal_lm)
    model = stack_model or _pick_model_class(merged.class_registry)
    if causal_lm is not None:
        merged.causal_lm_class = causal_lm.name
    if stack_model is not None:
        merged.stack_model_class = stack_model.name
    if model is not None:
        merged.model_class = model.name
    merged.custom_blocks = sorted(set(merged.custom_blocks))
    merged.positional_helpers = sorted(set(merged.positional_helpers))
    return merged
