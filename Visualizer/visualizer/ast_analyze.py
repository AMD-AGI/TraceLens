"""Inspect Hugging Face modeling code via Python AST (CPU-only)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from visualizer.blocks import BlockComponent, CodeAnalysis

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
SYNTHETIC_FUNCTIONAL_LINEAR = "@functional_linear"
SYNTHETIC_ROUTER_ACTIVATION = "@router_activation"
SYNTHETIC_ROUTER_BIAS = "@router_bias"
SYNTHETIC_ROUTER_GROUP = "@router_group"
SYNTHETIC_ROUTER_TOPK = "@router_topk"
SYNTHETIC_ROUTER_GATHER = "@router_gather"
SYNTHETIC_ROUTER_RENORM = "@router_renorm"
SYNTHETIC_ROUTER_SCALE = "@router_scale"
SYNTHETIC_GATE_ACTIVATION = "@gate_activation"
SYNTHETIC_GATE_RESHAPE = "@gate_reshape"
_GATE_ACTIVATION_NAMES = {
    "sigmoid": "Sigmoid",
    "softmax": "Softmax",
    "tanh": "Tanh",
}
_ROUTER_SYNTHETICS = (
    SYNTHETIC_ROUTER_ACTIVATION,
    SYNTHETIC_ROUTER_BIAS,
    SYNTHETIC_ROUTER_GROUP,
    SYNTHETIC_ROUTER_TOPK,
    SYNTHETIC_ROUTER_GATHER,
    SYNTHETIC_ROUTER_RENORM,
    SYNTHETIC_ROUTER_SCALE,
)
_SYNTHETIC_ATTENTION_NAMES = {
    "eager_attention_forward",
    "flash_attention_forward",
    "sdpa_attention_forward",
    "attention_interface",
}
_KERNEL_MERGE_NAME_RE = re.compile(
    r"(attention|attn|kda|recurrent|flash|sdpa|linear_attn|delta)",
    re.IGNORECASE,
)
_SKIP_INIT_CLASS_NAMES = frozenset({"Parameter", "getattr"})


def _append_forward_call(calls: list[str], attr: str) -> None:
    if calls and calls[-1] == attr:
        return
    calls.append(attr)


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
    if_targets = {_assign_target(stmt) for stmt in if_node.body if isinstance(stmt, ast.Assign)}
    if_targets.discard(None)
    for stmt in if_node.orelse:
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
        if node.func.attr in _METHOD_CHAIN_OPS:
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
        if _is_functional_linear_call(func):
            _append_forward_call(out, SYNTHETIC_FUNCTIONAL_LINEAR)
            return

        target = _expr_name(func)
        if target in _SYNTHETIC_ATTENTION_NAMES or _is_kernel_merge_call(func):
            _append_forward_call(out, SYNTHETIC_ATTENTION)
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


def _is_functional_linear_call(func: ast.AST) -> bool:
    """True for F.linear(...) and torch.nn.functional.linear(...)."""
    if not isinstance(func, ast.Attribute) or func.attr != "linear":
        return False
    value = func.value
    if isinstance(value, ast.Name) and value.id == "F":
        return True
    if isinstance(value, ast.Attribute) and value.attr == "functional":
        base = value.value
        if isinstance(base, ast.Attribute) and base.attr == "nn":
            return isinstance(base.value, ast.Name) and base.value.id == "torch"
    return False


def _is_moe_gate_class(class_name: str, forward_calls: list[str]) -> bool:
    if SYNTHETIC_FUNCTIONAL_LINEAR not in forward_calls:
        return False
    if re.search(r"Gate$", class_name):
        return True
    return bool(MOE_CLASS_RE.search(class_name) and re.search(r"gate|router", class_name, re.I))


def _call_uses_attr(node: ast.AST, attr: str) -> bool:
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == attr:
            return True
    return False


def _call_uses_self_attr(node: ast.AST, attr: str) -> bool:
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Attribute) and _is_self_attr(func, attr):
            return True
    if isinstance(node, ast.BinOp):
        for side in (node.left, node.right):
            if isinstance(side, ast.Attribute) and _is_self_attr(side, attr):
                return True
    return False


def _detect_router_activation(func: ast.FunctionDef) -> str | None:
    for node in func.body:
        if isinstance(node, ast.If):
            for branch in (node.body, node.orelse):
                for stmt in branch:
                    src = _stmt_value(stmt)
                    if src is None:
                        continue
                    if _call_uses_attr(src, "sigmoid"):
                        return "Sigmoid"
                    if _call_uses_attr(src, "softmax"):
                        return "Softmax"
        src = _stmt_value(node)
        if src is not None:
            if _call_uses_attr(src, "sigmoid"):
                return "Sigmoid"
            if _call_uses_attr(src, "softmax"):
                return "Softmax"
    return None


def _stmt_value(stmt: ast.AST) -> ast.AST | None:
    if isinstance(stmt, ast.Assign):
        return stmt.value
    if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        return stmt.value
    if isinstance(stmt, ast.Return) and stmt.value is not None:
        return stmt.value
    return None


def _forward_has_expert_bias(func: ast.FunctionDef) -> bool:
    return any(
        isinstance(node, ast.Attribute) and _is_self_attr(node, "e_score_correction_bias")
        for node in ast.walk(func)
    )


def _forward_has_group_routing(func: ast.FunctionDef) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
        if "num_expert_group" not in test_src:
            continue
        if any(_call_uses_attr(stmt, "masked_fill") for stmt in node.body):
            return True
    return False


def _forward_has_topk(func: ast.FunctionDef) -> bool:
    return any(_call_uses_attr(node, "topk") for node in func.body)


def _forward_has_gather(func: ast.FunctionDef) -> bool:
    return any(_call_uses_attr(node, "gather") for node in func.body)


def _forward_has_renormalize(func: ast.FunctionDef) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
        if "moe_renormalize" in test_src or "renormalize" in test_src:
            return True
    return False


def _forward_has_route_scale(func: ast.FunctionDef) -> bool:
    return any(
        isinstance(node, ast.Attribute) and _is_self_attr(node, "routed_scaling_factor")
        for node in ast.walk(func)
    )


def _router_pipeline_from_forward(func: ast.FunctionDef) -> list[tuple[str, list[str]]]:
    """Synthetic MoE router steps after F.linear."""
    pipeline: list[tuple[str, list[str]]] = []

    activation = _detect_router_activation(func)
    if activation:
        pipeline.append((SYNTHETIC_ROUTER_ACTIVATION, [activation]))

    if _forward_has_expert_bias(func):
        pipeline.append((SYNTHETIC_ROUTER_BIAS, ["Expert bias"]))

    if _forward_has_group_routing(func):
        pipeline.append((SYNTHETIC_ROUTER_GROUP, ["Group routing"]))

    if _forward_has_topk(func):
        pipeline.append((SYNTHETIC_ROUTER_TOPK, ["Top-k experts"]))

    if _forward_has_gather(func):
        pipeline.append((SYNTHETIC_ROUTER_GATHER, ["Gather weights"]))

    if _forward_has_renormalize(func):
        pipeline.append((SYNTHETIC_ROUTER_RENORM, ["Renormalize"]))

    if _forward_has_route_scale(func):
        pipeline.append((SYNTHETIC_ROUTER_SCALE, ["Route scaling"]))

    return pipeline


def _router_forward_step_details(
    class_name: str,
    func: ast.FunctionDef,
    forward_calls: list[str],
) -> dict[str, list[str]]:
    if not _is_moe_gate_class(class_name, forward_calls):
        return {}

    pipeline = _router_pipeline_from_forward(func)
    if not pipeline:
        return {}

    linear_index = forward_calls.index(SYNTHETIC_FUNCTIONAL_LINEAR)
    details: dict[str, list[str]] = {}
    for offset, (step, step_details) in enumerate(pipeline, start=1):
        forward_calls.insert(linear_index + offset, step)
        details[step] = step_details
    return details


COMBINE_DETAIL_PREFIX = "combine:"


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


def _detect_method_combine_op(func: ast.FunctionDef) -> str | None:
    """Infer a combine-operator symbol from a helper method body."""
    for node in ast.walk(func):
        value: ast.AST | None = None
        if isinstance(node, ast.Return):
            value = node.value
        elif isinstance(node, ast.Assign):
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            value = node.value
        if value is not None and _expr_is_weighted_sum(value):
            return "Σ"
    return None


def _method_forward_step_details(
    class_node: ast.ClassDef,
    forward_calls: list[str],
    init_assignments: dict[str, str],
) -> dict[str, list[str]]:
    """Attach AST-derived metadata to forward helper methods."""
    method_funcs = {
        item.name: item
        for item in class_node.body
        if isinstance(item, ast.FunctionDef)
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
        combine_op = _detect_method_combine_op(func)
        if combine_op is None:
            continue
        details[call_attr] = [
            f"method `{call_attr}()`",
            f"{COMBINE_DETAIL_PREFIX} {combine_op}",
        ]
    return details


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
HEAD_CLASS_RE = re.compile(r"(?:^|_)head$|LMHead|CausalLMOutput", re.I)


def _classify_role(attr_name: str, class_name: str) -> str:
    attr_key = attr_name.lower()
    if attr_key in ATTR_ROLE_HINTS:
        return ATTR_ROLE_HINTS[attr_key]
    for token in re.split(r"[_\W]+", attr_key):
        if token in {"attn", "attention"}:
            return "attention"
        if token in {"mlp", "ffn"}:
            return "ffn"
        if token in {"moe", "experts"}:
            return "moe"
        if "norm" in token:
            return "norm"
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
    if HEAD_CLASS_RE.search(class_name) or (re.match(r"(?i)^Linear$", class_name) and "head" in attr_key):
        return "head"
    return "other"


def displays_as_linear(attr_name: str, class_name: str | None) -> bool:
    """True when a module should be drawn as a plain Linear op."""
    return bool(class_name and re.match(r"(?i)^Linear$", class_name))


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
        if re.search(r"Delta|KDA|LinearAttn|LinearAttention", class_name, re.I):
            return "KDA"
        if re.search(r"Latent|MLA", class_name, re.I):
            return "MLA"
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
    gate_activations: dict[str, str] = field(default_factory=dict)
    forward_step_details: dict[str, list[str]] = field(default_factory=dict)
    side_inputs: dict[str, list[SideInputSpec]] = field(default_factory=dict)
    init_assignment_options: dict[str, list[str]] = field(default_factory=dict)


# Backwards-compatible alias used internally.
_ClassInfo = ClassStructure


class _ModelAstVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.classes: dict[str, ClassStructure] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        init_assignments: dict[str, str] = {}
        init_details: dict[str, list[str]] = {}
        init_assignment_options: dict[str, list[str]] = {}
        forward_calls: list[str] = []
        norm_before: list[str] = []
        attention_inputs: dict[str, list[str]] = {}
        parallel_gates: list[str] = []
        gate_activations: dict[str, str] = {}
        forward_step_details: dict[str, list[str]] = {}
        side_inputs: dict[str, list[SideInputSpec]] = {}

        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                init_assignments, init_details, init_assignment_options = _parse_init(item)
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                (
                    forward_calls,
                    norm_before,
                    attention_inputs,
                    side_inputs,
                    parsed_step_details,
                ) = _parse_forward(item)
                alternate = _alternate_forward_dispatches(item)
                if alternate:
                    forward_calls = [call for call in forward_calls if call not in alternate]
                parallel_gates = _parallel_gates_from_forward(item)
                if forward_calls and parallel_gates:
                    # Routers like MoE `gate` run on hidden_states as the main path, not in parallel.
                    parallel_gates = [gate for gate in parallel_gates if gate != forward_calls[0]]
                gate_activations = _parallel_gate_activations_from_forward(item, parallel_gates)
                forward_step_details = _router_forward_step_details(node.name, item, forward_calls)
                forward_step_details.update(parsed_step_details)
                forward_step_details.update(
                    _method_forward_step_details(node, forward_calls, init_assignments)
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
            gate_activations=gate_activations,
            forward_step_details=forward_step_details,
            side_inputs=side_inputs,
        )
        self.generic_visit(node)


def _parse_init(func: ast.FunctionDef) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
    assignments: dict[str, str] = {}
    details: dict[str, list[str]] = {}
    options: dict[str, list[str]] = {}

    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and _is_self_attr(target, target.attr):
                    class_name = _assignment_class_name(node.value)
                    if class_name:
                        attr_options = options.setdefault(target.attr, [])
                        if class_name not in attr_options:
                            attr_options.append(class_name)
                        assignments[target.attr] = class_name
                        details[target.attr] = _assignment_details(node.value, class_name)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if (
                isinstance(target, ast.Attribute)
                and _is_self_attr(target, target.attr)
                and node.value is not None
            ):
                class_name = _assignment_class_name(node.value)
                if class_name:
                    attr_options = options.setdefault(target.attr, [])
                    if class_name not in attr_options:
                        attr_options.append(class_name)
                    assignments[target.attr] = class_name
                    details[target.attr] = _assignment_details(node.value, class_name)

    return assignments, details, options


def _assignment_class_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            return None
        if isinstance(node.func, ast.Attribute) and node.func.attr == "Parameter":
            return None
        class_name = _call_class_name(node)
        if class_name in _SKIP_INIT_CLASS_NAMES:
            return None
        return class_name
    if isinstance(node, ast.ListComp) and isinstance(node.elt, ast.Call):
        return _call_class_name(node.elt)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ModuleList":
        if node.args and isinstance(node.args[0], ast.ListComp):
            elt = node.args[0].elt
            if isinstance(elt, ast.Call):
                return _call_class_name(elt)
    return None


def _assignment_details(node: ast.AST, class_name: str) -> list[str]:
    details: list[str] = []
    if not isinstance(node, ast.Call):
        return details

    for keyword in node.keywords:
        if keyword.arg in {"num_experts", "top_k", "num_experts_per_tok"}:
            value = ast.literal_eval(keyword.value) if _is_literal(keyword.value) else None
            if value is not None:
                details.append(f"{keyword.arg}={value}")
        if keyword.arg == "activation" and _is_literal(keyword.value):
            raw = ast.literal_eval(keyword.value)
            if isinstance(raw, str):
                details.append(_GATE_ACTIVATION_NAMES.get(raw.lower(), raw.capitalize()))

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


def _chains_from_expr(value: ast.AST, var_chains: dict[str, list[str]]) -> list[list[str]]:
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
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "map":
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


def _kernel_input_label(name: str) -> str:
    normalized = {
        "q": "Q",
        "query": "Q",
        "query_states": "Q",
        "k": "K",
        "key": "K",
        "key_states": "K",
        "v": "V",
        "value": "V",
        "value_states": "V",
        "g": "G",
        "gate": "G",
        "beta": "β",
    }
    lowered = name.lower()
    if lowered in normalized:
        return normalized[lowered]
    if len(name) <= 4:
        return name.upper()
    return name


def _is_data_movement_call(func: ast.AST) -> bool:
    name = _expr_name(func)
    if not name:
        return False
    base = name.split(".")[-1]
    return base in _DATA_MOVEMENT_NAMES


def _is_kernel_merge_call(func: ast.AST) -> bool:
    if _is_data_movement_call(func):
        return False
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
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
            producers[_kernel_input_label(label)] = list(chain)

    for keyword in call.keywords:
        if keyword.arg in _KERNEL_PRODUCER_SKIP_KWARGS:
            continue
        if isinstance(keyword.value, ast.Attribute):
            continue
        if keyword.arg:
            consider(keyword.arg, keyword.value)

    args = call.args
    start = 1 if args and isinstance(args[0], ast.Name) and args[0].id == "self" else 0
    positional_labels = ["Q", "K", "V"]
    for index, arg in enumerate(args[start:], start=start):
        label = positional_labels[index - start] if (index - start) < len(positional_labels) else f"in{index - start}"
        consider(label, arg)

    return producers


def _inject_kernel_merge(
    node: ast.AST,
    var_chains: dict[str, list[str]],
    stmt_calls: list[str],
    attention_inputs: dict[str, list[str]],
    forward_step_details: dict[str, list[str]],
) -> None:
    if any(label in attention_inputs for label in ("Q", "K", "V")):
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
        kernel_name = _expr_name(call.func) or "kernel"
        forward_step_details[SYNTHETIC_ATTENTION] = [f"kernel: {kernel_name.split('.')[-1]}"]
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


_KDA_KERNEL_MARKERS = ("kda", "delta", "linear_attn", "linear_attention")

_STANDARD_ATTENTION_MARKERS = (
    "sdpa",
    "scaled_dot_product",
    "flash_attn",
    "flash_attention",
    "eager_attention",
    "attention_interface",
    "transformer_engine",
    "transformerengine",
    "fused_attention",
    "fused_attn",
    "memory_efficient_attention",
    "xformers",
    "dot_product_attention",
    "multi_head_attention",
)


def _is_kda_attention_kernel(kernel: str | None) -> bool:
    if not kernel:
        return False
    lowered = kernel.lower()
    return any(marker in lowered for marker in _KDA_KERNEL_MARKERS)


def is_standard_attention_kernel(kernel: str | None) -> bool:
    """True for kernels that delegate to a common attention library (SDPA, Flash, TE, …)."""
    if not kernel:
        return False
    if _is_kda_attention_kernel(kernel):
        return False
    lowered = kernel.lower()
    if lowered in _SYNTHETIC_ATTENTION_NAMES:
        return True
    return any(marker in lowered for marker in _STANDARD_ATTENTION_MARKERS)


def is_kda_attention_step(details: list[str]) -> bool:
    return _is_kda_attention_kernel(kernel_name_from_step_details(details))


def is_standard_attention_step(details: list[str]) -> bool:
    return is_standard_attention_kernel(kernel_name_from_step_details(details))


def attention_kernel_label(details: list[str]) -> str:
    kernel = kernel_name_from_step_details(details)
    if _is_kda_attention_kernel(kernel):
        return "KDA"
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
    extra_inputs: list[str] = []
    if attention_inputs:
        extra_inputs = [label for label in attention_inputs if label not in {"Q", "K", "V"}]

    if _is_kda_attention_kernel(kernel):
        summary = "recurrent delta rule (not softmax QKᵀV)"
        if kernel:
            summary = f"{summary} · {kernel}"
        lines = [summary]
        if extra_inputs:
            lines.append(f"inputs: Q,K,V,{','.join(extra_inputs)}")
        lines.append("S ← (I−βkkᵀ)Diag(α)S + βkvᵀ")
        return lines

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
        if len(call.args) < 4:
            continue
        for idx, label in enumerate(["Q", "K", "V"]):
            arg = call.args[idx + 1]
            if isinstance(arg, ast.Name):
                chain = var_chains.get(arg.id, [])
                if chain:
                    attention_inputs[label] = list(chain)
        return


def _arg_name(arg: ast.AST, index: int) -> str:
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
        return "residual"
    if isinstance(arg, ast.Name):
        lowered = arg.id.lower()
        if "topk" in lowered or lowered in {"topk_idx", "topk_weight", "router_logits"}:
            return "router"
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
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Attribute) or not _is_self_attr(call.func, call.func.attr):
            continue
        callee = call.func.attr
        if callee.startswith("@") or callee in _SYNTHETIC_ATTENTION_NAMES:
            continue
        if not call.args:
            continue

        main_chain, main_kind = _arg_provenance(call.args[0], var_chains, forward_input_names)
        specs: list[SideInputSpec] = []
        seen: set[tuple[str, tuple[str, ...], SideInputSource]] = set()

        if (
            main_kind == "forward_input"
            and prior_calls
            and callee != prior_calls[0]
        ):
            arg0_name = _arg_name(call.args[0], 0)
            if arg0_name not in forward_input_names:
                key = ("residual", tuple(), "forward_input")
                if key not in seen:
                    seen.add(key)
                    specs.append(
                        SideInputSpec(
                            arg_name=arg0_name,
                            port_label="residual",
                            source_chain=[],
                            source_kind="forward_input",
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
                    arg_name=_arg_name(arg, arg_index),
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
                for item in existing
            )
            if not duplicate:
                existing.append(spec)


def _forward_input_names(func: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    args = func.args
    for arg in args.posonlyargs + args.args:
        if arg.arg != "self":
            names.add(arg.arg)
    return names


def _parallel_gates_from_forward(func: ast.FunctionDef) -> list[str]:
    """Modules invoked directly on forward inputs (e.g. output gate from hidden_states)."""
    input_names = _forward_input_names(func)
    gates: list[str] = []
    for node in func.body:
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, ast.Attribute) or not _is_self_attr(call.func, call.func.attr):
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


def _parallel_gate_activation(func: ast.FunctionDef, gate_attr: str) -> str | None:
    """Detect activation applied to a parallel output gate (e.g. g_proj(...).sigmoid())."""
    gate_vars: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                if _is_self_attr(value.func, gate_attr):
                    gate_vars.add(target)
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
            if isinstance(target, ast.Name):
                value = node.value
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                    if _is_self_attr(value.func, gate_attr):
                        gate_vars.add(target.id)

    for node in ast.walk(func):
        src = _stmt_value(node) if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return)) else None
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
) -> tuple[list[str], list[str], dict[str, list[str]], dict[str, list[SideInputSpec]], dict[str, list[str]]]:
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
    return _dedupe_kernel_merge_calls(calls), norm_before, attention_inputs, side_inputs, forward_step_details


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
        _record_assign_targets(node, stmt_calls, var_chains)
        _capture_attention_inputs(node, var_chains, attention_inputs)
        _capture_call_side_inputs(node, var_chains, forward_input_names, side_inputs, calls)
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
        if isinstance(node.target, ast.Name):
            chain = _trace_var_chain(node.value, var_chains, stmt_calls)
            if chain:
                var_chains[node.target.id] = chain
        _capture_attention_inputs(node, var_chains, attention_inputs)
        _capture_call_side_inputs(node, var_chains, forward_input_names, side_inputs, calls)
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
        _capture_call_side_inputs(node, var_chains, forward_input_names, side_inputs, calls)
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
        _capture_call_side_inputs(node, var_chains, forward_input_names, side_inputs, calls)
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
        _capture_call_side_inputs(node, var_chains, forward_input_names, side_inputs, calls)
        return _register_forward_calls(stmt_calls, calls, norm_before, pending_norm)

    if isinstance(node, ast.If):
        branch = node.body if _if_has_competing_assigns(node) else node.body + node.orelse
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


def _pick_decoder_class(classes: dict[str, ClassStructure]) -> ClassStructure | None:
    ranked: list[tuple[int, ClassStructure]] = []
    for info in classes.values():
        score = 0
        name = info.name
        if DECODER_CLASS_RE.search(name):
            score += 10
        if any(_classify_role(a, c) == "attention" for a, c in info.init_assignments.items()):
            score += 5
        if any(_classify_role(a, c) in {"ffn", "moe"} for a, c in info.init_assignments.items()):
            score += 3
        if info.forward_calls:
            score += 2
        if score > 0:
            ranked.append((score, info))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def _pick_model_class(classes: dict[str, ClassStructure]) -> ClassStructure | None:
    for info in classes.values():
        if MODEL_CLASS_RE.search(info.name) and info.init_assignments:
            if any("embed" in attr.lower() for attr in info.init_assignments):
                return info
    return None


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


def _config_positional_component(positional_encoding: str) -> BlockComponent | None:
    if positional_encoding in {"NoPE", "none", "None", ""}:
        return None
    return BlockComponent(
        attr_name="rotary_emb",
        class_name="RotaryEmbedding",
        role="positional",
        label=positional_encoding,
        forward_order=1,
        details=[f"positional encoding ({positional_encoding})"],
    )


def build_stack_components(
    *,
    stack_model: ClassStructure | None,
    causal_lm: ClassStructure | None,
    decoder: ClassStructure | None,
    registry: dict[str, ClassStructure],
    positional_encoding: str,
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
                    details=registry.get(class_name, stack_model).init_details.get(attr, []),
                )
            )
        else:
            inferred = _config_positional_component(positional_encoding)
            if inferred is not None:
                pre.append(inferred)

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

    if causal_lm is not None:
        order = {attr: idx for idx, attr in enumerate(causal_lm.forward_calls)}
        for attr, class_name in causal_lm.init_assignments.items():
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
                    details=causal_lm.init_details.get(attr, []),
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
            {"norm": 0, "head": 1}.get(comp.role, 99),
            comp.forward_order if comp.forward_order is not None else 999,
            comp.attr_name,
        )
    )
    return pre, tail


def _infer_attention_type_from_class(info: ClassStructure | None, all_classes: dict[str, ClassStructure]) -> str | None:
    if info is None:
        return None

    attn_attr = next(
        (attr for attr, cls in info.init_assignments.items() if _classify_role(attr, cls) == "attention"),
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
        norm_attrs = {a for a, c in decoder.init_assignments.items() if _classify_role(a, c) == "norm"}
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
                    details=attention_kernel_details(step_details, decoder.attention_inputs),
                )
            )
            continue
        if attr in decoder.init_assignments:
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
        class_names = decoder.init_assignment_options.get(comp.attr_name) or [comp.class_name]
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
    return (
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "ModuleList")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "ModuleList")
        )
    )


def _parse_layer_module_list(value: ast.AST) -> tuple[str, str, str] | None:
    list_comp: ast.ListComp | None = None
    if _is_module_list(value) and value.args and isinstance(value.args[0], ast.ListComp):
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


def _find_layer_loop_in_init(init_func: ast.FunctionDef) -> tuple[str, str, str, str] | None:
    for node in init_func.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Attribute) or not _is_self_attr(target, target.attr):
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


def _extract_decoder_layer_conditionals(decoder: ClassStructure) -> list[tuple[str, str, str]]:
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


def build_class_registry(source: str, *, filename: str = "<model>") -> dict[str, ClassStructure]:
    """Return all class structures discovered in one modeling file."""
    tree = parse_python_ast(source, filename=filename)
    visitor = _ModelAstVisitor()
    visitor.visit(tree)
    return visitor.classes


def merge_class_registries(*registries: dict[str, ClassStructure]) -> dict[str, ClassStructure]:
    merged: dict[str, ClassStructure] = {}
    for registry in registries:
        merged.update(registry)
    return merged


def analyze_source(source: str, *, filename: str = "<model>") -> CodeAnalysis:
    """Analyze one modeling file and return extracted block structure."""
    tree = parse_python_ast(source, filename=filename)
    visitor = _ModelAstVisitor()
    visitor.visit(tree)

    decoder = _pick_decoder_class(visitor.classes)
    causal_lm = _pick_causal_lm_class(visitor.classes)
    stack_model = _pick_stack_model_class(visitor.classes, causal_lm)
    model = stack_model or _pick_model_class(visitor.classes)
    analysis = CodeAnalysis(source_files=[filename])
    analysis.class_registry = dict(visitor.classes)

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
                positional_encoding="RoPE",
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
        positional_encoding="RoPE",
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

    roles = {comp.role for comp in analysis.block_components}
    if "moe" in roles:
        analysis.decoder_type = "Sparse MoE"
    elif len([comp for comp in analysis.block_components if comp.role == "attention"]) > 1:
        analysis.decoder_type = "Hybrid"
    elif "ffn" in roles:
        analysis.decoder_type = "Dense"

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


def analyze_sources(sources: dict[Path, str]) -> CodeAnalysis:
    """Analyze multiple files and merge into one CodeAnalysis."""
    merged = CodeAnalysis()
    registries: list[dict[str, ClassStructure]] = []
    for path, text in sources.items():
        partial = analyze_source(text, filename=str(path))
        registries.append(partial.class_registry)
        merged.source_files.extend(partial.source_files)
        merged.notes.extend(partial.notes)

        if partial.decoder_class and not merged.decoder_class:
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
            merged.custom_blocks.extend(partial.custom_blocks)

        if partial.model_class and not merged.model_class:
            merged.model_class = partial.model_class

    merged.class_registry = merge_class_registries(*registries)
    merged.custom_blocks = sorted(set(merged.custom_blocks))
    return merged
