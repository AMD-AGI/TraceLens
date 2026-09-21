###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inspect Hugging Face modeling code via Python AST (CPU-only)."""

from __future__ import annotations

import ast
import copy
import importlib.util
import logging
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

_log = logging.getLogger(__name__)

from TraceLens.ModelUtils.blocks import BlockComponent, CodeAnalysis
from TraceLens.ModelUtils.config_resolve import apply_config_attribute_aliases

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
# A bare call to a module-level free function that is neither a recognised tensor
# op, a rope helper, nor an attention kernel (e.g. ``get_vision_position_ids(...)``).
# The forward still runs real computation there, so it must render as its own node
# rather than vanishing; the line number keeps repeated call sites distinct.
FUNCTION_SYNTHETIC_PREFIX = "@fn_"
_FUNCTION_SOURCE_POS_RE = re.compile(
    rf"^{re.escape(FUNCTION_SYNTHETIC_PREFIX)}l(\d+)_"
)
# Trailing per-element discriminator appended by ``function_synthetic_attr`` /
# ``positional_synthetic_attr`` / ``submodule_callsite_attr`` when one call site
# is cloned across a ``map(lambda x: ..., (a, b))`` tuple -- both clones share the
# lambda body's own line (and column), so the discriminator is what keeps them
# from colliding on one key.
_MAP_DISCRIMINATOR_RE = re.compile(r"@m\d+$")
# Python builtins and scalar constructors that appear in a forward but never carry
# a tensor the diagram should show as a computation node.
_NON_TENSOR_BUILTINS = frozenset(
    {
        "len", "int", "float", "bool", "str", "range", "enumerate", "zip",
        "super", "print", "isinstance", "issubclass", "getattr", "setattr",
        "hasattr", "delattr", "type", "min", "max", "sum", "abs", "round",
        "list", "tuple", "dict", "set", "frozenset", "sorted", "reversed",
        "map", "filter", "any", "all", "repr", "format", "iter", "next",
        "id", "hash", "vars", "dir", "callable", "slice", "object",
    }
)


def positional_synthetic_attr(
    func_name: str, lineno: int, discriminator: int | None = None
) -> str:
    """Synthetic attr for a rope helper called as a plain function in a forward.

    The line number keeps each application site distinct, since a forward commonly
    rotates queries and keys with separate calls to the same helper. ``discriminator``
    mirrors ``function_synthetic_attr``'s -- see there for when it is needed.
    """
    base = f"{POSITIONAL_SYNTHETIC_PREFIX}l{lineno}_{func_name}"
    return base if discriminator is None else f"{base}@m{discriminator}"


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
        name = _MAP_DISCRIMINATOR_RE.sub("", name)
    text = name.replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else name


def function_synthetic_attr(
    func_name: str, lineno: int, discriminator: int | None = None
) -> str:
    """Synthetic attr for a bare free-function call traced in a forward.

    Mirrors ``positional_synthetic_attr`` but for functions that are not rope
    helpers, so a computed side-input (``get_vision_position_ids(...)``) becomes a
    visible node instead of vanishing. The line number keeps each call site apart.

    ``discriminator`` disambiguates two applications of the SAME call that share
    one source line -- ``q, k = map(lambda x: rearrange(x, ...), (q, k))`` applies
    one lambda body to each tuple element, so both synthesized calls are clones of
    the exact same AST node (identical line AND column) and would otherwise
    collide on one key, silently dropping one element's producer.
    """
    base = f"{FUNCTION_SYNTHETIC_PREFIX}l{lineno}_{func_name}"
    return base if discriminator is None else f"{base}@m{discriminator}"


def is_function_synthetic(attr_name: str) -> bool:
    return attr_name.startswith(FUNCTION_SYNTHETIC_PREFIX)


def function_synthetic_source_pos(attr_name: str) -> tuple[int, int] | None:
    """Source position of a traced free-function call, for ordering among siblings."""
    match = _FUNCTION_SOURCE_POS_RE.match(attr_name)
    if match is None:
        return None
    return int(match.group(1)), 0


def function_display_label(attr_name_or_func: str) -> str:
    """Display label for a traced free function (get_vision_position_ids ->
    Get vision position ids)."""
    name = attr_name_or_func
    if name.startswith(FUNCTION_SYNTHETIC_PREFIX):
        name = _FUNCTION_SOURCE_POS_RE.sub("", name)
        name = _MAP_DISCRIMINATOR_RE.sub("", name)
    text = name.replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else name


# A submodule/method call's step key is normally the bare child attr
# (``self.norm(x)`` -> ``norm``). When the SAME child is called more than once in
# a single forward (``cos = self.recomposition_frequencies(cos)`` then
# ``sin = self.recomposition_frequencies(sin)``), that bare key collides: the
# second call overwrites the first's predecessors and ``var_producer`` binding, so
# one branch is dead-code-eliminated. Disambiguate repeated call sites with an
# ``@l{lineno}`` suffix (mirroring the synthetic kinds) so each call keeps its own
# wiring; strip it back to the base attr wherever the key indexes ``init_assignments``
# / ``multi_op_methods`` (the child-class join point).
_SUBMODULE_CALLSITE_RE = re.compile(r"@l(\d+)$")


def submodule_callsite_attr(attr: str, lineno: int) -> str:
    """Call-site-disambiguated step key for a repeated submodule/method call."""
    return f"{attr}@l{lineno}"


def submodule_callsite_source_pos(attr: str) -> tuple[int, int] | None:
    """Source position encoded in a call-site step key, for ordering among siblings."""
    match = _SUBMODULE_CALLSITE_RE.search(attr)
    if match is None:
        return None
    return int(match.group(1)), 0


def base_submodule_attr(attr: str) -> str:
    """Strip a call-site ``@l{lineno}`` suffix back to the base child attr.

    Safe for every key: the suffix is anchored to the end, synthetic attrs
    (``@op_l..``/``@positional_l..``/``@fn_l..``) end in their function/op name
    rather than a trailing ``@l\\d+``, and real submodule attrs are plain
    identifiers. Only keys produced by ``submodule_callsite_attr`` are affected.
    """
    return _SUBMODULE_CALLSITE_RE.sub("", attr)


def _is_emittable_free_function(func: ast.AST, target: str | None) -> bool:
    """True for a bare ``foo(...)`` call to a module-level function worth showing.

    Excludes Python builtins/scalar constructors (which never carry a tensor to
    diagram). The callers check submodule/functional/attention/positional first, so
    only genuinely unrecognised free functions reach this gate.
    """
    return (
        isinstance(func, ast.Name)
        and bool(target)
        and target not in _NON_TENSOR_BUILTINS
    )


# Tensor methods that force a host round-trip: they materialise tensor contents
# into Python objects, which only happens on CPU. A free function using any of
# these (directly, or via another free function it calls) runs host-side work.
_HOST_MATERIALIZE_METHODS = frozenset({"tolist", "item", "numpy", "cpu"})


def _call_forces_host(call: ast.Call) -> bool:
    """True when a call is a tensor->host materialisation (``.tolist()``/``.to('cpu')``)."""
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr in _HOST_MATERIALIZE_METHODS:
        return True
    if func.attr == "to":
        for arg in [*call.args, *(kw.value for kw in call.keywords)]:
            if isinstance(arg, ast.Constant) and arg.value == "cpu":
                return True
    return False


def _absolute_import_bindings(
    tree: ast.AST, current_module: str
) -> dict[str, str]:
    """Map imported names to ``absolute.module#symbol`` for one module's AST.

    Resolves relative imports (``from ...vision_utils import x``) to their absolute
    dotted module against *current_module* (the module the AST belongs to), so a
    cross-file callee can be located with ``importlib.util.find_spec``.
    """
    parts = current_module.split(".")
    bindings: dict[str, str] = {}
    if not isinstance(tree, ast.Module):
        return bindings
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            if stmt.level:
                base = parts[: -stmt.level] if len(parts) >= stmt.level else []
                module = ".".join(base + (stmt.module.split(".") if stmt.module else []))
            else:
                module = stmt.module or ""
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                bindings[alias.asname or alias.name] = f"{module}#{alias.name}"
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                bindings[alias.asname or alias.name] = f"{alias.name}#{alias.name}"
    return bindings


class _HostSourceResolver:
    """Detects whether a free function runs host/CPU work by reading source ASTs.

    Locates each callee's defining file with ``importlib.util.find_spec`` (no module
    execution — spec.origin + ``ast.parse`` only) and walks it for host-materialisation
    idioms, recursing into the free functions it calls. The analysed modeling file is
    seeded directly so its local helpers resolve without a spec lookup. General across
    models: any helper doing host-side index building is flagged, none are hardcoded.
    """

    def __init__(self) -> None:
        # module -> ({func_name: FunctionDef}, {imported_name: "module#symbol"}) | None
        self._modules: dict[str, tuple[dict[str, ast.FunctionDef], dict[str, str]] | None] = {}

    def seed(self, module: str | None, tree: ast.AST) -> None:
        if not module or not isinstance(tree, ast.Module):
            return
        funcs = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self._modules[module] = (funcs, _absolute_import_bindings(tree, module))

    def _load(
        self, module: str
    ) -> tuple[dict[str, ast.FunctionDef], dict[str, str]] | None:
        if module in self._modules:
            return self._modules[module]
        result: tuple[dict[str, ast.FunctionDef], dict[str, str]] | None = None
        origin: str | None = None
        try:
            spec = importlib.util.find_spec(module)
            origin = spec.origin if spec is not None else None
        except (ImportError, AttributeError, ValueError):
            origin = None
        if origin and Path(origin).is_file():
            try:
                tree = ast.parse(Path(origin).read_text(encoding="utf-8"))
            except (OSError, SyntaxError, ValueError):
                tree = None
            if isinstance(tree, ast.Module):
                funcs = {
                    node.name: node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                }
                result = (funcs, _absolute_import_bindings(tree, module))
        self._modules[module] = result
        return result

    def runs_on_host(
        self, module: str, name: str, _seen: set[tuple[str, str]] | None = None
    ) -> bool:
        seen = _seen if _seen is not None else set()
        key = (module, name)
        if key in seen:
            return False
        seen.add(key)
        loaded = self._load(module)
        if loaded is None:
            return False
        funcs, imports = loaded
        func = funcs.get(name)
        if func is None:
            # Imported (re-exported) here — follow it to the defining module.
            binding = imports.get(name)
            if binding:
                dest, _, symbol = binding.partition("#")
                if dest and dest != module:
                    return self.runs_on_host(dest, symbol, seen)
            return False
        if any(
            isinstance(node, ast.Call) and _call_forces_host(node)
            for node in ast.walk(func)
        ):
            return True
        for node in ast.walk(func):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            callee = node.func.id
            if callee in funcs:
                if self.runs_on_host(module, callee, seen):
                    return True
            elif callee in imports:
                dest, _, symbol = imports[callee].partition("#")
                if dest and self.runs_on_host(dest, symbol, seen):
                    return True
        return False


def _analyzed_base_module(config: dict[str, Any] | None) -> str | None:
    """Dotted module of the analysed modeling file, for resolving its own imports."""
    if not isinstance(config, dict):
        return None
    model_type = str(config.get("model_type") or "").strip().replace("-", "_")
    if not model_type:
        return None
    return f"transformers.models.{model_type}.modeling_{model_type}"


def _annotate_host_free_functions(
    classes: dict[str, "ClassStructure"],
    tree: ast.AST,
    config: dict[str, Any] | None,
) -> None:
    """Flag each traced free-function call that runs host/CPU work.

    A ``get_vision_position_ids(...)`` side-input node built from a synthetic
    function/positional attr gets ``forward_step_runs_on_host[attr] = True`` when the
    callee (or something it transitively calls) materialises a tensor on the host.
    Rope helpers (``apply_rotary_pos_emb_vision``) carry no such idiom -> stay False.
    """
    base_module = _analyzed_base_module(config)
    resolver = _HostSourceResolver()
    resolver.seed(base_module, tree)
    cache: dict[str, bool] = {}
    for cls in classes.values():
        attrs = set()
        for mapping in (
            cls.forward_step_details,
            cls.forward_operations,
            cls.forward_step_output_names,
            cls.multi_op_methods,
            cls.single_op_methods,
            cls.forward_step_predecessors,
        ):
            attrs.update(mapping.keys())
        attrs.update(cls.forward_calls)
        for attr in attrs:
            if not (is_function_synthetic(attr) or is_positional_synthetic(attr)):
                continue
            name = _synthetic_call_function_name(attr)
            if not name or base_module is None:
                continue
            if name not in cache:
                cache[name] = resolver.runs_on_host(base_module, name)
            if cache[name]:
                cls.forward_step_runs_on_host[attr] = True


def _registry_dict_literal(
    tree: ast.Module, name: str, _seen: set[str] | None = None
) -> ast.Dict | None:
    """Locate the dict literal a module-level registry name resolves to.

    Handles one level of indirection (``ACT2FN = ClassInstantier(ACT2CLS)``) by
    following a wrapper call's first argument back to its own module-level
    assignment, so a registry alias resolves to the same literal as the name it
    wraps.
    """
    seen = _seen if _seen is not None else set()
    if name in seen:
        return None
    seen.add(name)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Dict):
            return value
        if isinstance(value, ast.Call):
            for arg in value.args:
                if isinstance(arg, ast.Name):
                    found = _registry_dict_literal(tree, arg.id, seen)
                    if found is not None:
                        return found
    return None


def _registry_entry_class_name(value: ast.expr) -> str | None:
    """Return the class name a registry dict entry's value expression names.

    An entry is either a bare class reference (``SqrtSoftplusActivation``,
    ``nn.Sigmoid``) or a ``(cls, kwargs)`` tuple pairing one with constructor
    kwargs (``ClassInstantier`` convention); either way only the class name
    matters here.
    """
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    if isinstance(value, (ast.Tuple, ast.List)) and value.elts:
        return _registry_entry_class_name(value.elts[0])
    return None


def _resolve_activation_registry_class(
    registry_name: str,
    key: str,
    import_bindings: dict[str, str],
    *,
    all_tensor_ops: bool,
) -> "ClassStructure | None":
    """Load the concrete class an unrecognized activation-registry key selects.

    ``self.act = ACT2FN[key]`` (or ``ACT2CLS[key]``) freezes in a real
    ``nn.Module`` subclass; our curated ``_ACTIVATION_DISPLAY_NAMES`` table only
    covers the common ones. For any other key: follow the *modeling file's own
    import* of the registry name to its defining module (no hardcoded module
    path -- whatever the file actually imports from), read that module's
    registry-dict literal to find which class the key selects, and parse that
    class's own source into a :class:`ClassStructure` the same way any other
    submodule class is parsed, so its forward can be expanded instead of drawn
    as one opaque box. Returns ``None`` when any step is not resolvable (no
    import found, no importable source, key/class not found, or class defines
    no ``forward``); the caller then keeps its title-cased placeholder leaf and
    logs a warning.
    """
    binding = import_bindings.get(registry_name)
    if not binding:
        return None
    module, _, _symbol = binding.partition("#")
    if not module:
        return None
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, AttributeError, ValueError):
        return None
    origin = spec.origin if spec is not None else None
    if not origin or not Path(origin).is_file():
        return None
    try:
        source = Path(origin).read_text(encoding="utf-8")
        registry_tree = ast.parse(source, filename=origin)
    except (OSError, SyntaxError, ValueError):
        return None
    dict_literal = _registry_dict_literal(registry_tree, registry_name)
    if dict_literal is None:
        return None
    lowered = key.strip().lower()
    class_name: str | None = None
    for entry_key, entry_value in zip(dict_literal.keys, dict_literal.values):
        if (
            isinstance(entry_key, ast.Constant)
            and isinstance(entry_key.value, str)
            and entry_key.value.strip().lower() == lowered
        ):
            class_name = _registry_entry_class_name(entry_value)
            break
    if not class_name:
        return None
    try:
        external_registry = build_class_registry(
            source, filename=origin, all_tensor_ops=all_tensor_ops
        )
    except (SyntaxError, ValueError):
        return None
    resolved = external_registry.get(class_name)
    if resolved is None or not any(
        isinstance(item, ast.FunctionDef) and item.name == "forward"
        for item in resolved.node.body
    ):
        return None
    return resolved


def _expand_unresolved_activation_classes(
    classes: dict[str, "ClassStructure"],
    tree: ast.AST,
    config: dict[str, Any] | None,
    *,
    all_tensor_ops: bool,
) -> None:
    """Route an unrecognized activation-registry submodule through forward-expansion.

    A curated activation (SiLU/GELU/Sigmoid/...) stays an atomic leaf by design.
    Anything else selected through ``ACT2FN``/``ACT2CLS`` is a real ``nn.Module``
    whose forward we can read like any other submodule's -- so resolve it
    (structurally, via the modeling file's own imports; see
    ``_resolve_activation_registry_class``) and register its class so the normal
    block-tree expansion path picks it up instead of falling back to an opaque,
    title-cased ``OperationKind.UNKNOWN`` leaf. When resolution or parsing fails
    (dynamic key, no importable source, unparseable forward) the referencing
    class keeps its placeholder name and a warning is logged so the gap stays
    visible instead of silently mis-rendering.
    """
    base_module = _analyzed_base_module(config)
    if base_module is None:
        import_bindings: dict[str, str] = {}
    elif isinstance(tree, ast.Module):
        import_bindings = _absolute_import_bindings(tree, base_module)
    else:
        import_bindings = {}
    resolved_cache: dict[tuple[str, str], "ClassStructure | None"] = {}

    for cls in list(classes.values()):
        for attr, (registry_name, key) in list(cls.unresolved_activation_refs.items()):
            cache_key = (registry_name, key)
            if cache_key not in resolved_cache:
                resolved_cache[cache_key] = _resolve_activation_registry_class(
                    registry_name,
                    key,
                    import_bindings,
                    all_tensor_ops=all_tensor_ops,
                )
            resolved = resolved_cache[cache_key]
            if resolved is None:
                _log.warning(
                    "Could not resolve activation registry entry %s[%r] "
                    "(assigned to %s.%s) to an importable class with a "
                    "parseable forward; rendering it as an opaque leaf.",
                    registry_name,
                    key,
                    cls.name,
                    attr,
                )
                continue
            classes[resolved.name] = resolved
            placeholder = cls.init_assignments.get(attr)
            cls.init_assignments[attr] = resolved.name
            options = cls.init_assignment_options.get(attr)
            if options is not None:
                cls.init_assignment_options[attr] = [
                    resolved.name if option == placeholder else option
                    for option in options
                ]


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

# A gated norm's gate activation, once resolved generically from the module's
# constructor kwarg or its own init/config symbol table, is stored on the node as
# a tagged detail with this prefix. Consumers read the resolved value structurally
# from the tag instead of matching detail text against a hardcoded activation set.
GATE_ACTIVATION_DETAIL_PREFIX = "gate activation: "


def _display_activation_name(raw: str) -> str:
    """Canonical display name for an activation registry key (``silu`` -> ``SiLU``).

    Resolves through the shared activation registry so every path (constructor
    ``activation=`` kwarg, ``ACT2FN[self.x]`` forward reads) renders the same name;
    an unknown key title-cases as a best-effort label rather than being dropped.
    """
    lowered = raw.strip().lower()
    if lowered in _ACTIVATION_DISPLAY_NAMES:
        return _ACTIVATION_DISPLAY_NAMES[lowered]
    if lowered in _GATE_ACTIVATION_NAMES:
        return _GATE_ACTIVATION_NAMES[lowered]
    return raw.strip().replace("_", " ").title().replace(" ", "")
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
# Boolean helpers a forward branches on to pick a flash-attention code path
# (``if is_flash_attention_requested(self.config): ...``). We resolve them from
# the checkpoint's ``_attn_implementation`` so only the selected branch survives,
# instead of walking both and leaving duplicated/dangling kernel plumbing.
_FLASH_REQUEST_PREDICATES = {"is_flash_attention_requested"}
# ``_attn_implementation`` values that route to a flash code path.
_FLASH_IMPL_NAMES = {
    "flash_attention_2",
    "flash_attention_3",
    "flash_attention",
    "flash_attn",
    "flash_attn_2",
    "kernels-community/flash-attn",
}
_KERNEL_MERGE_NAME_RE = re.compile(
    r"(attention|attn|recurrent|flash|sdpa|linear_attn|kernel|chunk)",
    re.IGNORECASE,
)
# A metadata helper whose name merely *mentions* attention (e.g.
# ``get_vision_attention_seqlens``) computes cu_seqlens/masks, not the attention
# output. Its result head-noun (``seqlens``/``ids``/``mask``) or builder prefix
# (``get_``/``build_``) marks it as plumbing, so it must not be swept into the
# attention-kernel bucket by the substring match above.
_KERNEL_MERGE_HELPER_RE = re.compile(
    r"^(get|build|make|prepare|compute|create|update|_)_"
    r"|(_seqlens?|_ids?|_masks?|_lengths?|_indices|_index|_sizes?|"
    r"_positions?|_offsets?|_cache|_shapes?)$",
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


def _traced_free_function_arg_names(func: ast.FunctionDef) -> set[str]:
    """Names passed positionally into a traced free-function node in ``func``.

    A rope helper or other module-level free function (``get_vision_position_ids``)
    renders as its own node; the plain-``Name`` tensors it reads are that node's
    real sources. Collecting them lets a secondary forward input feeding one be
    seeded as the method boundary so the edge starts from the input.

    Calls nested inside a conditional are skipped: only unconditional free-function
    nodes render (see ``_extract_self_calls_ordered``'s ``skip_free_fn``), so seeding
    an input consumed only by a dropped-branch helper would resurrect otherwise-dead
    ops (e.g. a per-chunk ``lengths`` subtract) with no visible consumer.
    """
    conditional_calls: set[int] = set()
    for stmt in ast.walk(func):
        if isinstance(stmt, ast.If):
            for child in stmt.body + stmt.orelse:
                for sub in ast.walk(child):
                    if isinstance(sub, ast.Call):
                        conditional_calls.add(id(sub))
    names: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call) or id(node) in conditional_calls:
            continue
        callee = node.func
        target = _expr_name(callee)
        traced = (
            bool(target)
            and isinstance(callee, ast.Name)
            and _is_positional_function_call(callee, target)
        ) or _is_emittable_free_function(callee, target)
        if not traced:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Name):
                names.add(arg.id)
    return names


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


def _free_function_call_targets(funcs: dict[str, ast.FunctionDef]) -> set[str]:
    """Bare-name call targets referenced anywhere in a set of function bodies.

    Used to scope ``_imported_forward_functions``'s recursive import-following to
    names a harvested module's own functions actually call, rather than every
    name the module happens to import. A pure re-export module (no top-level
    ``def``s of its own -- e.g. a package ``__init__.py``) contributes nothing
    here, so none of its re-exports get chased just because one of them was
    independently resolved from somewhere else.
    """
    names: set[str] = set()
    for func in funcs.values():
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                names.add(node.func.id)
    return names


def _imported_forward_functions(
    tree: ast.AST, base_module: str | None
) -> dict[str, ast.FunctionDef]:
    """Resolve imported free functions' definitions from their defining files.

    A forward that calls a helper imported from a sibling file
    (``from ...vision_utils import get_vision_position_ids``) otherwise renders the
    call as one opaque tile, because the definition is not in this module's tree.
    Locating the source (``importlib.util.find_spec`` + ``ast.parse``, no module
    execution) lets the export inline the helper's computation like a local free
    function. General: every imported name is followed to its module, none are
    hardcoded.
    """
    if not base_module or not isinstance(tree, ast.Module):
        return {}
    # module -> ({func_name: FunctionDef}, {imported_name: "module#symbol"}) | None
    module_cache: dict[
        str, tuple[dict[str, ast.FunctionDef], dict[str, str]] | None
    ] = {}

    def _load(
        module: str,
    ) -> tuple[dict[str, ast.FunctionDef], dict[str, str]] | None:
        if module in module_cache:
            return module_cache[module]
        value: tuple[dict[str, ast.FunctionDef], dict[str, str]] | None = None
        origin: str | None = None
        try:
            spec = importlib.util.find_spec(module)
            origin = spec.origin if spec is not None else None
        except (ImportError, AttributeError, ValueError):
            origin = None
        if origin and Path(origin).is_file():
            try:
                mod_tree = ast.parse(Path(origin).read_text(encoding="utf-8"))
            except (OSError, SyntaxError, ValueError):
                mod_tree = None
            if isinstance(mod_tree, ast.Module):
                funcs = {
                    node.name: node
                    for node in mod_tree.body
                    if isinstance(node, ast.FunctionDef)
                }
                value = (funcs, _absolute_import_bindings(mod_tree, module))
        module_cache[module] = value
        return value

    resolved: dict[str, ast.FunctionDef] = {}
    seen_modules: set[str] = set()

    def _resolve_binding(local_name: str, binding: str, depth: int) -> None:
        dest, _, symbol = binding.partition("#")
        if not dest:
            return
        loaded = _load(dest)
        if loaded is None:
            return
        funcs, _ = loaded
        func = funcs.get(symbol)
        if isinstance(func, ast.FunctionDef):
            resolved.setdefault(local_name, func)
        _harvest_module(dest, depth)

    def _harvest_module(module: str, depth: int) -> None:
        # A resolved helper may call sibling free functions defined in (or imported
        # into) its own module; harvesting them lets those nested calls inline too.
        # Bounded depth keeps the pool from fanning out across the whole package.
        if depth <= 0 or module in seen_modules:
            return
        seen_modules.add(module)
        loaded = _load(module)
        if loaded is None:
            return
        funcs, bindings = loaded
        for fname, fdef in funcs.items():
            resolved.setdefault(fname, fdef)
        # Only chase bindings the module's own functions actually call. A module
        # can import (and re-export) far more names than its own code ever uses --
        # e.g. a package's ``__init__.py`` re-exporting its whole public surface,
        # or a large module importing dozens of symbols only a few of which are
        # referenced locally. Walking every binding regardless of use lets the
        # recursion wander into an unrelated dependency and, on a bare-name
        # collision, silently resolve one of *our* free-function names (like a
        # model's own ``rearrange`` helper) to a same-named symbol from a
        # completely different place -- with a different implementation and a
        # different calling convention. Restricting to call targets keeps the
        # harvest scoped to what the resolved helper(s) can actually reach.
        called = _free_function_call_targets(funcs)
        for name, binding in bindings.items():
            if name not in called:
                continue
            _resolve_binding(name, binding, depth - 1)

    for local_name, binding in _absolute_import_bindings(tree, base_module).items():
        _resolve_binding(local_name, binding, depth=2)
    return resolved


def _module_forward_functions(
    tree: ast.AST, config: dict[str, Any] | None = None
) -> dict[str, ast.FunctionDef]:
    """Module-level ``def``s keyed by name, for expanding traced free-function calls.

    A rope helper or other free function called from a forward
    (``apply_rotary_pos_emb_vision(q, k, cos, sin)``) renders as an opaque tile
    unless its body is available to inline. Collecting the definitions lets the
    export show the computation it performs, like a submodule's forward. Helpers
    imported from sibling files are resolved cross-file (``config`` supplies the
    analysed module's dotted path) so they expand the same way; a local definition
    always wins over an import on a name clash.
    """
    functions: dict[str, ast.FunctionDef] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.FunctionDef):
            functions.setdefault(node.name, node)
    for name, func in _imported_forward_functions(
        tree, _analyzed_base_module(config)
    ).items():
        functions.setdefault(name, func)
    return functions


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
    # Strip trailing *bare* attribute accesses (``x.T``/``x.mT``/``x.data``) to reach
    # the underlying call. Method *calls* in a chain (``x.reshape(...).unbind(0)``)
    # are peeled by ``_extract_self_calls_ordered`` itself, which descends a
    # non-producer method call's receiver as operand 0 — so chaining, a purely
    # syntactic convenience, never hides the base producer regardless of which
    # ``torch.Tensor`` methods appear in the chain (no method allowlist to keep up
    # to date; see the receiver descent below).
    while isinstance(node, ast.Attribute):
        node = node.value
    return node


def _extract_self_calls_ordered(
    node: ast.AST,
    out: list[str],
    skip_free_fn: bool = False,
    repeated_attrs: frozenset[str] = frozenset(),
) -> None:
    """Collect self.module(...) calls in approximate evaluation order (inner-first).

    ``skip_free_fn`` suppresses only the unrecognised free-function (``@fn_``) node
    emission. Set it when the call lives inside a conditional branch: ``forward_calls``
    is built branch-unaware (both arms of an ``if`` are walked), so an ``@fn_`` node
    from a branch that config-resolution later drops would leak in and scramble the
    surrounding wiring. Recognised ops/kernels/rope helpers are unaffected.

    ``repeated_attrs`` names the self-submodule attrs called more than once in this
    forward; each call to one gets a call-site ``@l{lineno}`` suffix so two calls to
    the same child stay distinct steps (see ``submodule_callsite_attr``). Empty by
    default, so single-call forwards keep bare keys (byte-identical).
    """
    node = _unwrap_expr(node)
    if isinstance(node, ast.Call):
        for arg in node.args:
            _extract_self_calls_ordered(arg, out, skip_free_fn, repeated_attrs)
        for keyword in node.keywords:
            _extract_self_calls_ordered(keyword.value, out, skip_free_fn, repeated_attrs)

        func = node.func
        if isinstance(func, ast.Attribute) and _is_self_attr(func, func.attr):
            attr = func.attr
            if attr in repeated_attrs:
                attr = submodule_callsite_attr(attr, node.lineno)
            _append_forward_call(out, attr)
            return
        functional_op = _functional_call_name(func)
        if functional_op:
            _append_forward_call(out, functional_synthetic_attr(functional_op))
            return

        target = _expr_name(func)
        if target in _SYNTHETIC_ATTENTION_NAMES or _is_kernel_merge_call(func):
            _append_forward_call(out, SYNTHETIC_ATTENTION)
            return
        # A ``map(lambda x: BODY(x), (a, b))`` idiom clones BODY's own call node
        # once per tuple element (see ``_expand_map_lambda_tuple``); every clone
        # shares BODY's original source position, so the discriminator stamped on
        # the clone is what keeps their synthetic keys from colliding (mirrors
        # ``_call_step_producer``/``_map_element_step_attr``).
        discriminator = getattr(node, "_tracelens_map_discriminator", None)
        if target and _is_positional_function_call(func, target):
            # Rope helpers live at module level, so the block that applies them is
            # the only place the diagram can show the rotation happening.
            _append_forward_call(
                out, positional_synthetic_attr(target, node.lineno, discriminator)
            )
            return
        if not skip_free_fn and _is_emittable_free_function(func, target):
            # Any other module-level free function still runs real computation the
            # forward feeds downstream (``get_vision_position_ids(...)``); show it as
            # its own node instead of dropping it.
            _append_forward_call(
                out, function_synthetic_attr(target, node.lineno, discriminator)
            )
            return
        # No producer form matched: this is a tensor-method chain link
        # (``.reshape(...)``/``.permute(...)``/``.unbind(0)``/``.to(dtype)`` — any
        # ``torch.Tensor`` method returning a tensor, at *any* position in a chain).
        # Chaining is a syntactic convenience: ``recv.op(*args)`` is equivalent to
        # ``op(recv, *args)`` with ``recv`` as operand 0. Descend into the receiver
        # so the base producer (``self.qkv(...)``) is still collected however long
        # the chain is and whatever methods it uses. The link's own args were already
        # visited above, so a producer passed as a method argument is not dropped
        # either. Free-function calls (``func`` is an ``ast.Name``) don't reach here.
        if isinstance(func, ast.Attribute):
            _extract_self_calls_ordered(func.value, out, skip_free_fn, repeated_attrs)
        return

    if isinstance(node, ast.BinOp):
        _extract_self_calls_ordered(node.left, out, skip_free_fn, repeated_attrs)
        _extract_self_calls_ordered(node.right, out, skip_free_fn, repeated_attrs)
        return

    if isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            _extract_self_calls_ordered(elt, out, skip_free_fn, repeated_attrs)
        return

    if isinstance(node, ast.IfExp):
        _extract_self_calls_ordered(node.body, out, skip_free_fn, repeated_attrs)
        _extract_self_calls_ordered(node.orelse, out, skip_free_fn, repeated_attrs)
        return

    if isinstance(node, ast.Subscript):
        _extract_self_calls_ordered(node.value, out, skip_free_fn, repeated_attrs)
        return

    if isinstance(node, ast.Compare):
        _extract_self_calls_ordered(node.left, out, skip_free_fn, repeated_attrs)
        for comparator in node.comparators:
            _extract_self_calls_ordered(comparator, out, skip_free_fn, repeated_attrs)
        return


def _self_call_sites_in_expr(node: ast.AST) -> dict[str, set[int]]:
    """Distinct source linenos calling each ``self.<attr>`` inside an expression."""
    sites: dict[str, set[int]] = {}
    for inner in ast.walk(node):
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and _is_self_attr(inner.func, inner.func.attr)
        ):
            sites.setdefault(inner.func.attr, set()).add(inner.lineno)
    return sites


def _merge_sequential_sites(
    acc: dict[str, set[int]], nxt: dict[str, set[int]]
) -> dict[str, set[int]]:
    """Union call sites that execute one-after-another on the same path."""
    out = {attr: set(sites) for attr, sites in acc.items()}
    for attr, sites in nxt.items():
        out.setdefault(attr, set()).update(sites)
    return out


def _merge_branch_sites(
    body: dict[str, set[int]], orelse: dict[str, set[int]]
) -> dict[str, set[int]]:
    """Pick the busier branch per attr; mutually-exclusive arms don't both count."""
    out: dict[str, set[int]] = {}
    for attr in set(body) | set(orelse):
        body_sites = body.get(attr, set())
        else_sites = orelse.get(attr, set())
        out[attr] = body_sites if len(body_sites) >= len(else_sites) else else_sites
    return out


def _path_max_self_call_sites(stmts: list[ast.stmt]) -> dict[str, set[int]]:
    """Max distinct co-executing ``self.<attr>`` call linenos along any one path.

    ``ast.walk`` alone over-counts a child invoked once in each arm of an
    ``if/else`` (only one arm runs), which would wrongly disambiguate a single
    logical call (GLM's ``self.self_attn`` in its linear/full branches). Walking
    the control flow — summing sequential statements but taking the *larger* arm of
    a branch — counts only calls that can truly coexist, so genuinely repeated
    straight-line calls (rotary ``recomposition_frequencies(cos)`` then ``(sin)``)
    are still caught while branch alternatives are not.
    """
    result: dict[str, set[int]] = {}
    for stmt in stmts:
        if isinstance(stmt, ast.If):
            stmt_sites = _merge_sequential_sites(
                _self_call_sites_in_expr(stmt.test),
                _merge_branch_sites(
                    _path_max_self_call_sites(stmt.body),
                    _path_max_self_call_sites(stmt.orelse),
                ),
            )
        elif isinstance(stmt, (ast.For, ast.While)):
            iter_or_test = getattr(stmt, "iter", None) or getattr(stmt, "test", None)
            stmt_sites = _merge_sequential_sites(
                _self_call_sites_in_expr(iter_or_test) if iter_or_test else {},
                _merge_sequential_sites(
                    _path_max_self_call_sites(stmt.body),
                    _path_max_self_call_sites(stmt.orelse),
                ),
            )
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            stmt_sites = _path_max_self_call_sites(stmt.body)
        elif isinstance(stmt, ast.Try):
            stmt_sites = _path_max_self_call_sites(
                stmt.body + stmt.orelse + stmt.finalbody
            )
        else:
            stmt_sites = _self_call_sites_in_expr(stmt)
        result = _merge_sequential_sites(result, stmt_sites)
    return result


def _repeated_self_call_attrs(body: list[ast.stmt]) -> frozenset[str]:
    """Self-submodule/method attrs called from >1 distinct site on one exec path.

    Only these attrs receive call-site-disambiguated step keys; every other
    forward keeps bare keys, so the common single-call case stays byte-identical.
    Mutually-exclusive branch arms are not treated as repeats (see
    ``_path_max_self_call_sites``).
    """
    sites = _path_max_self_call_sites(list(body))
    return frozenset(attr for attr, linenos in sites.items() if len(linenos) > 1)


def _self_attr_name(node: ast.AST | None) -> str | None:
    """Return ``attr`` for a ``self.<attr>`` expression, else ``None``."""
    if isinstance(node, ast.Attribute) and _is_self_attr(node, node.attr):
        return node.attr
    return None


def _resolve_local_module_alias_calls(func: ast.FunctionDef) -> ast.FunctionDef:
    """Rewrite ``expert(...)`` aliases of ``self.experts[i]`` (and ``for blk in
    self.blocks``) as module calls.

    Two idioms bind a ModuleList entry to a local variable before invoking it:

    - subscript: ``expert = self.experts[i]; expert(...)``
    - iteration: ``for blk in self.blocks: blk(...)`` (also
      ``for i, blk in enumerate(self.blocks):``)

    Resolving the alias to ``self.<attr>(...)`` lets the normal forward parser retain
    the real submodule branch (routed expert, or the vision/decoder block body) instead
    of collapsing the bare-name call into a phantom kernel tile and silently dropping it.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(func):
        # Subscript alias: ``expert = self.experts[i]``.
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
            if (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Subscript)
                and isinstance(value.value, ast.Attribute)
                and _is_self_attr(value.value, value.value.attr)
            ):
                aliases[target.id] = value.value.attr
            continue
        # Iteration alias: ``for blk in self.blocks`` / ``enumerate(self.blocks)``.
        if isinstance(node, ast.For):
            iterable = node.iter
            if (
                isinstance(iterable, ast.Call)
                and isinstance(iterable.func, ast.Name)
                and iterable.func.id in {"enumerate", "reversed"}
                and iterable.args
            ):
                iterable = iterable.args[0]
            attr = _self_attr_name(iterable)
            if attr is None:
                continue
            loop_var = node.target
            # ``for i, blk in enumerate(...)`` binds the element to the last element
            # of the tuple; ``for blk in ...`` binds it directly.
            if isinstance(loop_var, ast.Tuple) and loop_var.elts:
                loop_var = loop_var.elts[-1]
            if isinstance(loop_var, ast.Name):
                aliases[loop_var.id] = attr
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
    # A pure aggregation node returns the single combined tensor. A method that
    # returns a tuple of several tensors (``pool_keys, pool_indices, pool_valid``)
    # is doing more than a weighted sum — it merely *contains* one as an inner step,
    # so it must expand into its full computation, not collapse to one ``Σ`` tile.
    for node in reversed(func.body):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            if len(node.value.elts) > 1:
                return None
            break
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
        base = base_submodule_attr(call_attr)
        if base in init_assignments:
            continue
        if call_attr.startswith("@") or call_attr == SYNTHETIC_ATTENTION:
            continue
        func = method_funcs.get(base)
        if func is None:
            continue
        combine_op = _detect_method_combine_op(func, class_name=class_node.name)
        if combine_op is None:
            continue
        details[call_attr] = [
            f"method `{base}()`",
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
        base = base_submodule_attr(call_attr)
        if (
            base in init_assignments
            or call_attr.startswith("@")
            or call_attr == SYNTHETIC_ATTENTION
        ):
            continue
        func = method_funcs.get(base)
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
            # Keyed by the base method name; two call sites of the same repeated
            # method resolve here through ``base_submodule_attr`` in the block tree.
            single[base] = operations.operations[0]
    return single


def _multi_op_forward_methods(
    class_node: ast.ClassDef,
    forward_calls: list[str],
    init_assignments: dict[str, str],
    *,
    self_values: dict[str, Any],
    all_tensor_ops: bool,
) -> tuple[
    dict[str, list[ForwardOperation]],
    dict[str, tuple[dict[str, str], list[str], str | None]],
    dict[str, str],
    dict[str, dict[str, tuple[str, ...]]],
    dict[str, list[str]],
    dict[str, dict[str, dict[str, str]]],
]:
    """Forward helper methods with enough tensor operations to expand as a subgraph.

    Also returns, per method, its ``(return_slots, return_order,
    primary_return_slot)`` so a tuple-returning helper (``key_states,
    value_states = self.expand_kv(...)``) exposes every return slot as its own
    frame output — the consumer then docks the right slot onto each port instead
    of collapsing parallel returns onto the frame tail — and its primary
    parameter name so the frame's ``@input`` boundary is labelled after the
    method's own first parameter. General: read off the method's own signature
    and return statement, no class-name checks.

    Also returns, per method, its full ``step_predecessors`` map -- this covers
    step names the flattened operation list itself has no entry for (a
    submodule invoked mid-expression inside the method, recorded only as a bare
    predecessor NAME on whichever op reads its result), so the block tree can
    still wire that submodule's own input once it resolves the name against the
    class's submodule registry and builds it a sibling node.

    Also returns, per method, the TRUE EVALUATION ORDER of its steps (ops *and*
    any submodule call embedded mid-expression), merging the method's own flat
    op list with such submodule-call names via ``_forward_calls_in_source_order``
    -- the same general merge a top-level ``forward()`` gets for its own
    ``forward_calls`` -- so the block tree can place a materialised submodule
    child (``act_fn``) in its real position instead of arbitrarily first or
    last, and each per-step ``step_predecessor_args`` map (arg name -> producer)
    so that child's input edge resolves through the same
    ``forward_step_predecessor_args`` mechanism an ordinary nested submodule
    call already relies on (``_submodule_chain_input``).
    """
    method_funcs = {
        item.name: item for item in class_node.body if isinstance(item, ast.FunctionDef)
    }
    expanded: dict[str, list[ForwardOperation]] = {}
    returns: dict[str, tuple[dict[str, str], list[str], str | None]] = {}
    inputs: dict[str, str] = {}
    step_predecessors: dict[str, dict[str, tuple[str, ...]]] = {}
    step_order: dict[str, list[str]] = {}
    step_predecessor_args: dict[str, dict[str, dict[str, str]]] = {}
    for call_attr in forward_calls:
        base = base_submodule_attr(call_attr)
        if (
            base in init_assignments
            or call_attr.startswith("@")
            or call_attr == SYNTHETIC_ATTENTION
        ):
            continue
        func = method_funcs.get(base)
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
            # Keyed by the base method name; two call sites of the same repeated
            # method resolve here through ``base_submodule_attr`` in the block tree.
            expanded[base] = operations.operations
            if operations.step_predecessors:
                step_predecessors[base] = dict(operations.step_predecessors)
            if operations.step_predecessor_args:
                step_predecessor_args[base] = dict(operations.step_predecessor_args)
            # Submodule calls embedded mid-expression (``self.act_fn(gate)``) are
            # recorded in ``step_predecessors`` but have no entry of their own in
            # ``operations.operations`` (a submodule call is never a labelled
            # tensor op). Resolve each such name against this class's own
            # submodule registry and merge it into the method's op list in true
            # evaluation order, so the block tree can place the materialised
            # submodule child correctly relative to its producer/consumer ops.
            op_attrs = {op.attr_name for op in operations.operations}
            embedded_submodule_calls = [
                name
                for name in operations.step_predecessors
                if name not in op_attrs
                and base_submodule_attr(name) in init_assignments
                and init_assignments[base_submodule_attr(name)]
                not in _SKIP_INIT_CLASS_NAMES
            ]
            if embedded_submodule_calls:
                step_order[base] = _forward_calls_in_source_order(
                    func, embedded_submodule_calls, operations.operations
                )
            primary_input = _primary_forward_input_name(func)
            if primary_input is not None:
                inputs[base] = primary_input
            if len(operations.return_order) >= 2:
                op_attrs = {op.attr_name for op in operations.operations}
                if all(
                    producer in op_attrs
                    for producer in operations.return_slots.values()
                ):
                    returns[base] = (
                        dict(operations.return_slots),
                        list(operations.return_order),
                        operations.primary_return_slot,
                    )
    return expanded, returns, inputs, step_predecessors, step_order, step_predecessor_args


def _synthetic_call_function_name(call_attr: str) -> str | None:
    """Recover the source function name from a traced free-function synthetic attr.

    ``@positional_l1615_apply_rotary_pos_emb_vision`` ->
    ``apply_rotary_pos_emb_vision``.
    """
    if is_positional_synthetic(call_attr):
        return _POSITIONAL_SOURCE_POS_RE.sub("", call_attr) or None
    if is_function_synthetic(call_attr):
        return _FUNCTION_SOURCE_POS_RE.sub("", call_attr) or None
    return None


def _free_function_param_list(func: ast.FunctionDef) -> list[str]:
    return [arg.arg for arg in func.args.posonlyargs + func.args.args]


def _inline_nested_free_functions(
    analysis: "ForwardAnalysis",
    module_functions: dict[str, ast.FunctionDef],
    *,
    self_values: dict[str, Any],
    all_tensor_ops: bool,
    _seen: frozenset[str] = frozenset(),
    _depth: int = 0,
) -> list[ForwardOperation]:
    """Flatten a free function's nested free-function calls into visible ops.

    ``apply_rotary_pos_emb_vision`` calls ``rotate_half(q)``; the extractor
    records that call as a synthetic predecessor (``@fn_l1575_rotate_half``) with
    no operation of its own, so a naive expansion leaves the multiply that
    consumes it pointing at a node that never renders. Splice the callee's ops in
    (namespaced per call site so two calls do not collide), remap the callee's
    ``@method_input`` to the producer feeding that call's primary arg, and rename
    the callee's return op to the synthetic attr so the original consumer still
    resolves. General: recurses to a bounded depth for any known free function.
    """
    operations = list(analysis.operations)
    if _depth >= 8:
        return operations
    own = {op.attr_name for op in operations}
    arg_maps = analysis.step_predecessor_args

    # Synthetic predecessors that name a known free function and have no op yet.
    nested_calls: list[str] = []

    def _consider(pred: str) -> None:
        if pred in own or pred in nested_calls:
            return
        name = _synthetic_call_function_name(pred)
        if name and name in module_functions and name not in _seen:
            nested_calls.append(pred)

    for op in operations:
        for pred in op.predecessors:
            _consider(pred)
    # A thin dispatcher's body may be only free-function calls whose results are
    # returned or chained (``get_vision_attention_seqlens`` = cu_seqlens helper +
    # max_seqlen helper), so no in-body op references them. Seed from the call
    # chain (step predecessors, in dependency order) and the return producers so
    # those calls still inline instead of leaving the frame an opaque tile.
    for pred in analysis.step_predecessors:
        _consider(pred)
    for producer in analysis.return_slots.values():
        if producer:
            _consider(producer)
    if not nested_calls:
        return operations

    expansions: dict[str, list[ForwardOperation]] = {}
    for call_attr in nested_calls:
        name = _synthetic_call_function_name(call_attr)
        nested_func = module_functions[name]
        nested = _forward_operations_from_forward(
            nested_func,
            self_values=self_values,
            all_tensor_ops=all_tensor_ops,
            module_functions=module_functions,
            is_free_function_body=True,
        )
        nested_ops = _inline_nested_free_functions(
            nested,
            module_functions,
            self_values=self_values,
            all_tensor_ops=all_tensor_ops,
            _seen=_seen | {name},
            _depth=_depth + 1,
        )
        if not nested_ops:
            continue
        params = _free_function_param_list(nested_func)
        primary = params[0] if params else None
        arg_map = arg_maps.get(call_attr, {})
        nested_attrs = {op.attr_name for op in nested_ops}
        return_producer = None
        if nested.primary_return_slot is not None:
            return_producer = nested.return_slots.get(nested.primary_return_slot)
        if return_producer is None:
            return_producer = nested_ops[-1].attr_name
        namespace = f"{call_attr}::"

        def remap_attr(attr: str) -> str:
            return call_attr if attr == return_producer else namespace + attr

        def remap_pred(pred: str) -> str | None:
            if pred == FORWARD_METHOD_INPUT:
                # The callee's primary parameter is fed by this call's first arg.
                return arg_map.get(primary) if primary else None
            if pred in nested_attrs:
                return remap_attr(pred)
            return pred

        rewritten: list[ForwardOperation] = []
        for op in nested_ops:
            preds = tuple(
                p for p in (remap_pred(pred) for pred in op.predecessors) if p
            )
            # A callee secondary parameter (rare) is fed by a further call arg;
            # turn it into a predecessor when a producer is known, else drop it.
            extra_param_preds: list[str] = []
            remaining_params: list[str] = []
            for param in op.param_inputs:
                producer = arg_map.get(param)
                if producer:
                    extra_param_preds.append(producer)
                elif param == primary:
                    resolved = arg_map.get(primary) if primary else None
                    if resolved:
                        extra_param_preds.append(resolved)
                else:
                    remaining_params.append(param)
            ports = tuple(
                (remap_attr(attr) if attr in nested_attrs else attr, ordinal)
                for attr, ordinal in op.predecessor_ports
            )
            rewritten.append(
                replace(
                    op,
                    attr_name=remap_attr(op.attr_name),
                    predecessors=tuple((*preds, *extra_param_preds)),
                    param_inputs=tuple(remaining_params),
                    predecessor_ports=ports,
                )
            )
        expansions[call_attr] = rewritten

    if not expansions:
        return operations

    # Emit each callee's ops just before the first original op that consumes it,
    # so producers precede consumers in the rendered pipeline.
    result: list[ForwardOperation] = []
    emitted: set[str] = set()
    for op in operations:
        for pred in op.predecessors:
            if pred in expansions and pred not in emitted:
                result.extend(expansions[pred])
                emitted.add(pred)
        result.append(op)
    for call_attr, ops in expansions.items():
        if call_attr not in emitted:
            result.extend(ops)
    return result


def _multi_op_free_functions(
    module_functions: dict[str, ast.FunctionDef],
    forward_calls: list[str],
    *,
    self_values: dict[str, Any],
    all_tensor_ops: bool,
) -> tuple[dict[str, list[ForwardOperation]], dict[str, list[str]]]:
    """Traced free-function calls whose body expands into a visible sub-pipeline.

    Keyed by the synthetic call attr (``@positional_l1615_...``) so the block
    tree renders the helper's computation inline instead of one opaque tile.
    Mirrors ``_multi_op_forward_methods`` for module-level functions.

    Also returns, for a tuple-returning helper, the ordered internal producer
    attrs of its return slots (ordinal -> producer attr), so a consumer reading a
    specific slot (``query_states`` = ordinal 0 of
    ``apply_rotary_pos_emb_vision``) can dock onto the matching internal op
    instead of the frame's last op. General: derived from the helper's own
    ``return_order``/``return_slots``, no class-name checks.
    """
    expanded: dict[str, list[ForwardOperation]] = {}
    return_producers: dict[str, list[str]] = {}
    for call_attr in forward_calls:
        name = _synthetic_call_function_name(call_attr)
        if name is None:
            continue
        func = module_functions.get(name)
        if func is None:
            continue
        analysis = _forward_operations_from_forward(
            func,
            self_values=self_values,
            all_tensor_ops=all_tensor_ops,
            module_functions=module_functions,
            is_free_function_body=True,
        )
        operations = _inline_nested_free_functions(
            analysis,
            module_functions,
            self_values=self_values,
            all_tensor_ops=all_tensor_ops,
            _seen=frozenset({name}),
        )
        if len(operations) > 1:
            expanded[call_attr] = operations
            if len(analysis.return_order) >= 2:
                op_attrs = {op.attr_name for op in operations}
                producers = [
                    analysis.return_slots.get(slot)
                    for slot in analysis.return_order
                ]
                # Only publish the map when every slot resolves to an op that
                # survived inlining (else fall back to the default last-op wiring).
                if all(p is not None and p in op_attrs for p in producers):
                    return_producers[call_attr] = [p for p in producers if p]
    return expanded, return_producers


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


def _constructed_class_name(func: ast.expr) -> str | None:
    """The class a call constructs: ``X(...)`` and ``X._from_config(...)`` -> ``X``.

    Direct instantiation names the class on ``func`` itself; HF classmethod
    constructors (``X._from_config``/``X.from_config``) name it on the attribute
    receiver. ``pkg.Thing(...)`` (e.g. ``nn.Conv3d``) resolves to the package name,
    which is harmless here since callers filter against the local class registry.
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id
    return None


def _call_uses_vision_config(call: ast.Call) -> bool:
    """True when any argument references ``…vision_config`` (the nested sub-config)."""
    for arg in list(call.args) + [kw.value for kw in call.keywords]:
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Attribute) and sub.attr == "vision_config":
                return True
    return False


def _vision_scoped_class_names(tree: ast.AST, config: dict[str, Any] | None) -> set[str]:
    """Local classes constructed under the vision tower (built with ``vision_config``).

    A HF vision-language model instantiates its vision tower with the nested
    ``vision_config`` (e.g. ``self.visual = XVisionModel._from_config(config.vision_config)``),
    inside which ``hidden_size`` and the patch geometry differ from the text model.
    Every module the tower builds inherits that sub-config, so its ``self.<attr> =
    config.<attr>`` reads must resolve against ``vision_config``. Text-only repos
    have no ``vision_config`` -> empty set -> the text path is provably untouched.

    Scoping is by instantiation subtree, not class name: a shared class such as
    ``RMSNorm`` used in both towers is included only through the vision subtree here,
    and callers overlay the sub-config for those instances alone.
    """
    if not isinstance(config, dict) or not isinstance(config.get("vision_config"), dict):
        return set()
    class_defs: dict[str, ast.ClassDef] = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    if not class_defs:
        return set()
    # Roots: any local class whose (class)method is called with ``config.vision_config``.
    roots: list[str] = []
    for cls in class_defs.values():
        for call in ast.walk(cls):
            if isinstance(call, ast.Call) and _call_uses_vision_config(call):
                root = _constructed_class_name(call.func)
                if root in class_defs and root not in roots:
                    roots.append(root)
    # BFS through ``__init__`` constructor calls, restricted to local classes.
    scoped: set[str] = set()
    frontier = list(roots)
    while frontier:
        name = frontier.pop()
        if name in scoped or name not in class_defs:
            continue
        scoped.add(name)
        init = next(
            (
                item
                for item in class_defs[name].body
                if isinstance(item, ast.FunctionDef) and item.name == "__init__"
            ),
            None,
        )
        if init is None:
            continue
        for call in ast.walk(init):
            if isinstance(call, ast.Call):
                child = _constructed_class_name(call.func)
                if child in class_defs and child not in scoped:
                    frontier.append(child)
    return scoped


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
    module_calls = [
        call for call in forward_calls if base_submodule_attr(call) in init_assignments
    ]
    if not module_calls:
        return False
    return all(
        displays_as_pointwise_leaf(
            base_submodule_attr(call), init_assignments[base_submodule_attr(call)]
        )
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


def _forward_delegates_only_to_sibling_methods(
    forward_calls: list[str],
    init_assignments: dict[str, str],
    method_names: set[str],
) -> bool:
    """True when forward()'s only calls are helper methods on the same class.

    A module can factor part of its own tensor math into sibling methods — e.g. a
    rotary embedding whose ``forward`` computes ``cos``/``sin`` inline and then calls
    ``self.recomposition_frequencies(...)`` to reshape them. Those calls are not
    submodules, so the math does not live in a child; the ``forward`` still owns it.
    Treat this like a forward that delegates to nothing so its inline operations
    (the multiplies here) are retained instead of collapsing to an opaque tile.
    """
    if not forward_calls:
        return False
    for call in forward_calls:
        base = base_submodule_attr(call)
        if base in init_assignments:
            return False  # a real submodule carries that part of the math
        if is_positional_synthetic(call) or is_functional_synthetic(call):
            return False  # positional/functional child, handled elsewhere
        if base not in method_names:
            return False  # unknown free call — don't assume inline ownership
    return True


def _forward_mixes_modules_and_inline_ops(
    forward_calls: list[str],
    init_assignments: dict[str, str],
    parsed_operations: list[ForwardOperation],
) -> bool:
    """True when forward() calls submodules and also runs inline tensor math."""
    module_calls = [
        call
        for call in forward_calls
        if base_submodule_attr(call) in init_assignments
        or is_positional_synthetic(call)
        or is_functional_synthetic(call)
        or is_function_synthetic(call)
    ]
    if not module_calls or not parsed_operations:
        return False
    inline_ops = sum(
        1 for op in parsed_operations if is_forward_operation(op.attr_name)
    )
    # A single inline op is enough. When a forward calls a composite child and
    # then combines its result inline (``left, right = self.split(x); return
    # self.proj(right) + left``), that lone Add is real computation the module
    # owns and must stay visible — it is not residual plumbing represented
    # elsewhere. ``_forward_owns_tensor_math`` only fires when *every* module
    # call is a pointwise leaf, so the composite-child + inline-op case reaches
    # here as its sole retention path.
    return inline_ops >= 1


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
    # Ordered names a multi-output op was tuple-unpacked into
    # (``pre_w, post_w, comb_w = ...split([hc, hc, hc * hc])`` -> these three).
    # Non-empty only for split/chunk/unbind that feed distinct downstream reads;
    # each name becomes one named output port with its own slice shape.
    output_names: tuple[str, ...] = ()
    # For a consumer of a multi-output op: which output ordinal of each producer
    # this operation reads (``producer_attr -> ordinal``), so its edge can attach
    # to the matching output port rather than the whole split.
    predecessor_ports: tuple[tuple[str, int], ...] = ()
    # For a ``param_inputs`` entry that is a tuple-unpacked alias of a secondary
    # forward parameter (``cos, sin = position_embeddings``), the origin
    # parameter's unpack ordinal (``position_embeddings`` -> 0 for ``cos``).
    # Lets the boundary-input pass dock a per-slot edge instead of collapsing
    # every alias onto the whole tensor's first consumer.
    param_input_ordinals: tuple[tuple[str, int], ...] = ()


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
    step_predecessor_ordinals: dict[str, dict[str, int]]
    step_output_names: dict[str, list[str]]
    step_boundary_params: dict[str, tuple[str, ...]]
    step_boundary_arg_params: dict[str, dict[str, tuple[str, int | None]]]
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
    # Display name of the gate activation a *gated norm* applies to its gate input
    # (``normalized * ACT2FN[self.activation](gate)``), resolved generically from
    # this class's own init/config symbol table. ``None`` for a plain norm.
    gate_activation: str | None = None
    forward_step_details: dict[str, list[str]] = field(default_factory=dict)
    side_inputs: dict[str, list[SideInputSpec]] = field(default_factory=dict)
    init_assignment_options: dict[str, list[str]] = field(default_factory=dict)
    forward_input_name: str | None = None
    forward_operations: dict[str, ForwardOperation] = field(default_factory=dict)
    forward_step_predecessors: dict[str, tuple[str, ...]] = field(default_factory=dict)
    forward_step_predecessor_args: dict[str, dict[str, str]] = field(
        default_factory=dict
    )
    forward_step_predecessor_ordinals: dict[str, dict[str, int]] = field(
        default_factory=dict
    )
    forward_step_output_names: dict[str, list[str]] = field(default_factory=dict)
    forward_step_boundary_params: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    forward_step_boundary_arg_params: dict[
        str, dict[str, tuple[str, int | None]]
    ] = field(default_factory=dict)
    single_op_methods: dict[str, ForwardOperation] = field(default_factory=dict)
    multi_op_methods: dict[str, list[ForwardOperation]] = field(default_factory=dict)
    # For an inline-expanded forward *method* returning a tuple
    # (``key_states, value_states = self.expand_kv(...)``): base method name ->
    # ``(return_slots, return_order, primary_return_slot)``, so the method frame
    # exposes every return slot as its own output port and consumers dock onto
    # the matching slot instead of collapsing onto the frame tail.
    multi_op_method_returns: dict[
        str, tuple[dict[str, str], list[str], str | None]
    ] = field(default_factory=dict)
    # For an inline-expanded forward *method*: base method name -> its primary
    # (first non-``self``) parameter name, so the method frame's ``@input``
    # boundary is labelled after the method's own parameter
    # (``build_attention_mask_from_topk`` -> ``topk_indices``) instead of falling
    # back to the generic ``hidden_states``.
    multi_op_method_inputs: dict[str, str] = field(default_factory=dict)
    # For an inline-expanded forward *method*: base method name -> {step name ->
    # predecessor attrs}, covering every step the method's own body extraction
    # recorded a predecessor for -- including a submodule invoked mid-expression
    # inside the method (``self.act_fn(gate)`` in ``return self.act_fn(gate) *
    # up``), which the flattened ``ForwardOperation`` list never carries its own
    # entry for (the extractor only leaves its bare name on the CONSUMING op's
    # predecessors). The block tree resolves such a name against this class's own
    # submodule registry and, when found, builds it a real sibling child; this
    # map is what tells that child what feeds it, so it does not appear as a
    # sourceless node when its own gate/branch happens not to be otherwise
    # consumed.
    multi_op_method_step_predecessors: dict[
        str, dict[str, tuple[str, ...]]
    ] = field(default_factory=dict)
    # For an inline-expanded forward *method*: base method name -> the TRUE
    # EVALUATION ORDER of its steps, merging its flat op-attr list with the name
    # of any submodule call embedded mid-expression (``act_fn`` in the example
    # above) via ``_forward_calls_in_source_order`` -- the same general merge a
    # class's own top-level ``forward()`` gets for ``forward_calls``. Only
    # populated when the method has such an embedded submodule call; lets the
    # block tree place that call's materialised sibling node in its real
    # position (after its own producer, before its own consumer) instead of
    # arbitrarily first or last in the frame's children.
    multi_op_method_order: dict[str, list[str]] = field(default_factory=dict)
    # For an inline-expanded forward *method*: base method name -> {step name ->
    # {arg name -> producer attr}}, the method's own ``step_predecessor_args``.
    # Set as ``forward_step_predecessor_args`` on the method's frame so a
    # materialised submodule child's input resolves through
    # ``_submodule_chain_input`` -- the same mechanism an ordinary nested
    # submodule call already relies on -- instead of the frame's naive
    # previous-sibling chaining, which would be wrong once the child is not the
    # very first step.
    multi_op_method_step_predecessor_args: dict[
        str, dict[str, dict[str, str]]
    ] = field(default_factory=dict)
    # For an inline-expanded free function returning a tuple
    # (``q_embed, k_embed = apply_rotary_pos_emb_vision(...)``): call attr ->
    # ordered internal producer attrs, so a consumer reading a specific return
    # ordinal docks onto the matching internal op, not the frame's last op.
    forward_step_return_producers: dict[str, list[str]] = field(default_factory=dict)
    forward_return_slots: dict[str, str] = field(default_factory=dict)
    forward_return_order: list[str] = field(default_factory=list)
    primary_return_slot: str | None = None
    forward_call_output_names: dict[str, str] = field(default_factory=dict)
    referenced_return_producers: set[str] = field(default_factory=set)
    loop_carried: list[LoopCarriedSpec] = field(default_factory=list)
    forward_param_inputs: list[str] = field(default_factory=list)
    dataflow_expanded: bool = False
    # Synthetic free-function/positional call attr -> True when that call (or a
    # function it transitively calls) runs host/CPU work (``.tolist()``/``.item()``).
    forward_step_runs_on_host: dict[str, bool] = field(default_factory=dict)
    # Submodule attr -> (registry_name, key) for a ``self.attr = ACT2FN[key]``-style
    # assignment whose key is not one of the curated ``_ACTIVATION_DISPLAY_NAMES``
    # (so its concrete class is unknown until resolved cross-file). Consumed by
    # ``_expand_unresolved_activation_classes`` to chase the real class and expand
    # its forward instead of rendering an opaque, title-cased placeholder leaf.
    unresolved_activation_refs: dict[str, tuple[str, str]] = field(default_factory=dict)


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
# A constructor attribute that is scalar-*typed* (an int/float/str/bool setting
# built from config reads and arithmetic) but whose concrete value could not be
# resolved -- e.g. ``self.qkv_dim = self.head_dim * self.num_heads`` when the
# ``linear_head_dim`` config key is not serialized in a given checkpoint. Stored
# so callers can tell "scalar setting, value unknown" from "genuine tensor
# attribute (parameter/buffer), never recorded". ``_config_value`` normalizes it
# back to ``_UNKNOWN`` so arithmetic/comparison folding stays value-based.
_SCALAR_SETTING = object()
_HOUSEKEEPING_METHODS = frozenset(
    {
        "view",
        "reshape",
        "flatten",
        "type",
        "float",
        "to",
        "type_as",
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
    # ``x.type_as(y)`` casts ``x`` to ``y``'s dtype -- ``y`` only supplies a dtype
    # reference (never real data), exactly like ``.to(dtype=...)``. Missing this
    # entry left the call unrecognized (no label), which fell through to the
    # generic call fallback and could resolve the reference operand as the
    # producer instead of the receiver, orphaning the real computation feeding it.
    "type_as": "Cast",
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

# Split / Concat / Slice / Tile rearrange or replicate tensors without computing
# new values. They stay visible in the graph (unlike the layout methods above,
# which are optional) but share the white data-movement fill.
LAYOUT_ONLY_LABELS = frozenset(_LAYOUT_ONLY_METHOD_LABELS.values()) | {
    "Split",
    "Concat",
    "Slice",
    "Tile",
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
    # Elementwise comparisons (each returns a boolean tensor; written as the
    # method form ``a.ge(b)`` rather than the ``a >= b`` operator, so they reach
    # the tensor-method label table rather than the BinOp/Compare path).
    "ge": "Greater equal",
    "gt": "Greater",
    "le": "Less equal",
    "lt": "Less",
    "eq": "Equal",
    "ne": "Not equal",
    "cos": "Cosine",
    "sin": "Sine",
    # Indexing and assembly
    "gather": "Gather",
    "masked_fill": "Masked fill",
    "masked_scatter": "Masked scatter",
    "scatter": "Scatter",
    "scatter_": "Scatter",
    "scatter_add": "Scatter add",
    "scatter_add_": "Scatter add",
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
    # Bitwise operators combine two tensors element-wise (mask logic such as
    # ``pool_visible & pool_valid``); without a label they collapse to a
    # pass-through that silently drops one operand and its producer subgraph.
    ast.BitAnd: "Bitwise and",
    ast.BitOr: "Bitwise or",
    ast.BitXor: "Bitwise xor",
}


def is_forward_operation(attr_name: str) -> bool:
    # A free-function frame inlines its ops under a ``@fn_..::`` namespace
    # (see ``_inline_nested_free_functions``); the op's identity is the final
    # ``::``-separated segment, so a frame-scoped op is still a forward operation.
    segment = attr_name.rsplit("::", 1)[-1]
    return segment.startswith(FORWARD_OPERATION_PREFIX)


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


# Upper bound on a loop's resolved static trip count. Keeps generated graphs
# bounded for malformed or unexpectedly large configs while still admitting
# realistic expert counts (e.g. ``num_experts=288``).
_LOOP_COUNT_MAX = 100_000

# A config *object* attribute name (what modeling code reads) mapped to the
# serialized config-dict keys it may be aliased to. Mirrors the synonym lists in
# ``extract._infer_ffn_and_moe`` so ``config.<attr>`` resolves against the raw
# config dict for any model using these conventional names.
_CONFIG_ATTR_ALIASES: dict[str, tuple[str, ...]] = {
    "num_experts": ("n_routed_experts", "moe_num_experts", "num_local_experts"),
    "num_experts_per_tok": (
        "num_experts_per_token",
        "moe_top_k",
        "num_selected_experts",
    ),
    "num_local_experts": ("num_experts", "n_routed_experts", "moe_num_experts"),
}

# A config attribute mapped to a (nested-dict-key, inner-key) pair.
_CONFIG_NESTED_ALIASES: dict[str, tuple[str, str]] = {
    "linear_lower_bound": ("linear_attn_config", "gate_lower_bound"),
}


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
            # Modeling code reads the transformers config *object* attribute
            # (e.g. ``config.num_experts``), whose name a config class often
            # aliases to a different serialized key (``n_routed_experts``). The
            # raw config dict only has the serialized key, so consult the same
            # well-known synonym lists ``extract._infer_*`` uses.
            for alias in _CONFIG_ATTR_ALIASES.get(node.attr, ()):
                aliased = config.get(alias, _UNKNOWN)
                if aliased is not _UNKNOWN:
                    return aliased
            nested_key = _CONFIG_NESTED_ALIASES.get(node.attr)
            if nested_key is not None:
                nested = config.get(nested_key[0])
                if isinstance(nested, dict):
                    return nested.get(nested_key[1], _UNKNOWN)
            return _UNKNOWN
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            resolved = self_values.get(node.attr, _UNKNOWN)
            # A scalar-typed-but-unresolved attribute carries no usable value for
            # arithmetic/comparison folding; surface it as unknown here so the rest
            # of ``_config_value`` never operates on the sentinel object.
            return _UNKNOWN if resolved is _SCALAR_SETTING else resolved
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


def _expr_is_scalar_typed(node: ast.AST, values: dict[str, Any]) -> bool:
    """True when a constructor RHS builds a scalar setting, not a tensor/module.

    Recognizes the expression shapes model ``__init__``s use for int/float/str
    hyper-parameters: literals, ``config.<key>`` reads, references to other
    already-recorded scalar attributes, arithmetic / comparison / boolean over
    those, and host builtins (``len``/``int``/``getattr(config, ...)``). A tensor
    or submodule assignment (``nn.Linear(...)``, ``nn.Parameter(...)``,
    ``self.forget_gate(...)``, a bare passed-in ``weight`` name) matches none of
    these, so it is *not* scalar-typed and stays a genuine tensor attribute. Kept
    general -- structural, no attribute-name or config-key literals.
    """
    if isinstance(node, ast.Constant):
        return not isinstance(node.value, bytes)
    if isinstance(node, ast.Name):
        return node.id == "config"
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            if node.value.id == "config":
                return True
            if node.value.id == "self":
                return node.attr in values
        return _expr_is_scalar_typed(node.value, values)
    if isinstance(node, ast.BinOp):
        return _expr_is_scalar_typed(node.left, values) and _expr_is_scalar_typed(
            node.right, values
        )
    if isinstance(node, ast.UnaryOp):
        return _expr_is_scalar_typed(node.operand, values)
    if isinstance(node, (ast.BoolOp, ast.Compare)):
        return True
    if isinstance(node, ast.IfExp):
        return _expr_is_scalar_typed(node.body, values) and _expr_is_scalar_typed(
            node.orelse, values
        )
    if isinstance(node, ast.Call):
        return _expr_name(node.func) in {
            "len",
            "int",
            "float",
            "bool",
            "round",
            "abs",
            "min",
            "max",
            "sum",
            "getattr",
        }
    return False


def _flatten_control_flow_body(stmts: list[ast.stmt]) -> list[ast.stmt]:
    """Flatten ``if``/``for``/``while``/``with``/``try`` bodies into one ordered,
    straight-line statement list (never descending into a nested ``def``/class).

    A constructor commonly guards its config-derived scalar assignments in a
    ``try: self.x = config.x \\n except Exception: raise ...`` (or an
    ``if cond: self.x = a \\n else: self.x = b``) block -- a defensive pattern,
    not specific to any one model family. Walking only ``init_func.body``
    misses every assignment nested one level inside such a block, silently
    dropping it out of ``self_values`` -- which then makes the read-site
    treat that name as an unresolved external tensor operand instead of a
    scalar setting (see ``_self_attr_input``'s ``_SCALAR_SETTING`` handling).
    Mirrors the existing control-flow-aware walk in
    ``_path_max_self_call_sites`` above.
    """
    flat: list[ast.stmt] = []
    for stmt in stmts:
        flat.append(stmt)
        if isinstance(stmt, ast.If):
            flat.extend(_flatten_control_flow_body(stmt.body))
            flat.extend(_flatten_control_flow_body(stmt.orelse))
        elif isinstance(stmt, (ast.For, ast.While)):
            flat.extend(_flatten_control_flow_body(stmt.body))
            flat.extend(_flatten_control_flow_body(stmt.orelse))
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            flat.extend(_flatten_control_flow_body(stmt.body))
        elif isinstance(stmt, ast.Try):
            flat.extend(_flatten_control_flow_body(stmt.body))
            for handler in stmt.handlers:
                flat.extend(_flatten_control_flow_body(handler.body))
            flat.extend(_flatten_control_flow_body(stmt.orelse))
            flat.extend(_flatten_control_flow_body(stmt.finalbody))
    return flat


def _self_config_values(
    init_func: ast.FunctionDef | None, config: dict[str, Any]
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if init_func is None:
        return values
    for stmt in _flatten_control_flow_body(init_func.body):
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            continue
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        value_node = stmt.value
        if value_node is None:
            continue
        value = _config_value(value_node, config, values)
        for target in targets:
            if isinstance(target, ast.Attribute) and _is_self_attr(target, target.attr):
                if value is not _UNKNOWN:
                    values[target.attr] = value
                elif _expr_is_scalar_typed(value_node, values):
                    # Scalar-typed but unresolvable (config key absent / derived
                    # from such): record under the sentinel so it is treated as a
                    # setting, never fabricated as a tensor operand, yet is not
                    # mistaken for a concrete value by the folding paths.
                    values[target.attr] = _SCALAR_SETTING
    return values


def _range_iteration_count_of(node: ast.For, self_values: dict) -> int | None:
    """Resolve a small static ``range(...)`` loop from constructor/config values."""
    iterator = node.iter
    if (
        not isinstance(iterator, ast.Call)
        or _expr_name(iterator.func) != "range"
        or iterator.keywords
        or not 1 <= len(iterator.args) <= 3
    ):
        return None
    values = [_config_value(arg, {}, self_values) for arg in iterator.args]
    if not all(isinstance(value, int) for value in values):
        return None
    try:
        count = len(range(*values))
    except (TypeError, ValueError):
        return None
    # Keep generated graphs bounded for malformed or unexpectedly large configs.
    return count if 0 <= count <= _LOOP_COUNT_MAX else None


def _num_classes_arg_of(call: ast.Call, self_values: dict) -> int | None:
    """Resolve ``one_hot``'s ``num_classes`` (keyword or 2nd positional)."""
    for keyword in call.keywords:
        if keyword.arg == "num_classes":
            resolved = _config_value(keyword.value, {}, self_values)
            if isinstance(resolved, int) and not isinstance(resolved, bool):
                return resolved
    if len(call.args) >= 2:
        resolved = _config_value(call.args[1], {}, self_values)
        if isinstance(resolved, int) and not isinstance(resolved, bool):
            return resolved
    return None


def _one_hot_bound_of(
    expr: ast.expr | None,
    seen: set[str],
    name_value_ast: dict[str, ast.expr],
    self_values: dict,
) -> int | None:
    """Follow an iterable's provenance to a ``one_hot(num_classes=...)`` width."""
    if expr is None:
        return None
    if isinstance(expr, ast.Name):
        if expr.id in seen:
            return None
        seen.add(expr.id)
        return _one_hot_bound_of(
            name_value_ast.get(expr.id), seen, name_value_ast, self_values
        )
    if isinstance(expr, ast.Call):
        func = expr.func
        is_one_hot = (_expr_name(func) == "one_hot") or (
            isinstance(func, ast.Attribute) and func.attr == "one_hot"
        )
        if is_one_hot:
            width = _num_classes_arg_of(expr, self_values)
            if width is not None:
                return width
        candidates: list[ast.expr] = []
        if isinstance(func, ast.Attribute):
            candidates.append(func.value)
        candidates.extend(expr.args)
        for candidate in candidates:
            width = _one_hot_bound_of(candidate, seen, name_value_ast, self_values)
            if width is not None:
                return width
        return None
    if isinstance(expr, (ast.Attribute, ast.Subscript)):
        return _one_hot_bound_of(expr.value, seen, name_value_ast, self_values)
    return None


def _loop_iteration_count_of(
    node: ast.For,
    self_values: dict,
    name_value_ast: dict[str, ast.expr],
) -> int | None:
    """Resolve a loop's static trip count for the ``Loop_N_iterations`` label.

    A literal ``range(...)`` bound wins. Otherwise a data-dependent iterable
    (``for expert_idx in hit:`` where ``hit`` is a ``nonzero()`` selection) can
    still have a static *upper* bound when the selected tensor's width comes from
    config — e.g. an expert-dispatch loop over
    ``one_hot(top_k_index, num_classes=self.num_experts)``. Trace the iterable
    back through simple ``name = expr`` bindings to that ``one_hot`` and resolve
    ``num_classes``. Returns ``None`` for genuinely unbounded loops.
    """
    count = _range_iteration_count_of(node, self_values)
    if count is not None:
        return count
    bound = _one_hot_bound_of(node.iter, set(), name_value_ast, self_values)
    return bound if bound is not None and 0 <= bound <= _LOOP_COUNT_MAX else None


def _collect_name_value_ast(func: ast.FunctionDef) -> dict[str, ast.expr]:
    """Map each simple ``name = expr`` binding in a function to its value AST.

    Used to trace a loop iterable's provenance (across statement boundaries and
    nesting) back to the ``one_hot`` call that bounds an expert-dispatch loop.
    """
    name_value_ast: dict[str, ast.expr] = {}
    for sub in ast.walk(func):
        if isinstance(sub, ast.Assign):
            for target in sub.targets:
                if isinstance(target, ast.Name):
                    name_value_ast[target.id] = sub.value
    return name_value_ast


class _MapLambdaParamSubstituter(ast.NodeTransformer):
    """Replaces a lambda's own parameter Name with a caller-supplied expression."""

    def __init__(self, param_name: str, replacement: ast.expr):
        self.param_name = param_name
        self.replacement = replacement

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802 (ast API name)
        if node.id == self.param_name:
            return copy.deepcopy(self.replacement)
        return node


def _expand_map_lambda_tuple(value: ast.expr) -> ast.expr:
    """Rewrite ``map(lambda x: BODY(x), (a, b, ...))`` into ``BODY(a), BODY(b), ...``.

    ``q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=...), (q, k))``
    applies ONE lambda body to each element of a literal tuple/list. ``map`` is a
    plain Python builtin the extractor never emits as its own step, so the whole
    call resolves through the generic single-producer fallback, which takes the
    LAST resolved argument producer for the entire expression -- both ``q`` and
    ``k`` collapse onto ``k``'s producer, orphaning ``q``'s producer entirely.

    Expanding the call up front into one clone of the lambda body per element
    -- with the lambda's own parameter substituted by that element's actual
    expression -- lets each element flow through the ordinary
    ``a, b = expr(a), expr(b)`` parallel-tuple-assignment path with its own
    identity, keyed on its own producer. General: any ``map(lambda <param>:
    <body>, <tuple/list literal>)`` right-hand side, regardless of which
    function the lambda body calls.

    Returns *value* unchanged when it is not this exact shape (a single-param
    lambda mapped over a literal tuple/list).
    """
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "map"
        and len(value.args) == 2
        and not value.keywords
        and isinstance(value.args[0], ast.Lambda)
    ):
        return value
    lam = value.args[0]
    lam_args = lam.args
    if lam_args.vararg or lam_args.kwonlyargs or lam_args.kwarg or lam_args.defaults:
        return value
    if len(lam_args.posonlyargs) + len(lam_args.args) != 1:
        return value
    param_name = (lam_args.posonlyargs or lam_args.args)[0].arg
    iterable = value.args[1]
    if not isinstance(iterable, (ast.Tuple, ast.List)):
        return value

    elements: list[ast.expr] = []
    for index, item in enumerate(iterable.elts):
        substituted = _MapLambdaParamSubstituter(param_name, item).visit(
            copy.deepcopy(lam.body)
        )
        ast.fix_missing_locations(substituted)
        # Both clones share the lambda body's own source position (one textual
        # call site applied twice), so the free-function/positional synthetic
        # naming that keys on ``lineno`` alone would still collide. Stamp a
        # discriminator any downstream call-producer lookup can key on.
        substituted._tracelens_map_discriminator = index  # type: ignore[attr-defined]
        elements.append(substituted)
    return ast.Tuple(elts=elements, ctx=ast.Load())


class _ForwardOperationExtractor:
    """Recover primitive tensor operations and their data dependencies."""

    def __init__(
        self,
        *,
        self_values: dict[str, Any],
        all_tensor_ops: bool,
        param_names: set[str] | None = None,
        config: dict[str, Any] | None = None,
        module_functions: dict[str, ast.FunctionDef] | None = None,
        repeated_submodule_attrs: frozenset[str] | None = None,
        class_methods: dict[str, ast.FunctionDef] | None = None,
        is_free_function_body: bool = False,
    ) -> None:
        self.self_values = self_values
        self.all_tensor_ops = all_tensor_ops
        self.config = dict(config or {})
        # True only when the body being extracted is a module-level free
        # function's own definition (``apply_rotary_pos_emb``), never a
        # class's own top-level ``forward`` or a sibling class method. A
        # free function's non-primary parameter can be sliced with no
        # producer of its own at all (``k[..., rotary_dim:]`` where ``k`` is
        # seeded only as a boundary alias) -- that case must still emit a
        # visible Slice. A class's own forward reading a genuine forward
        # parameter the same way (``cu_seqlens[1:] - cu_seqlens[:-1]``, pure
        # host index bookkeeping) must NOT gain a new visible op from this,
        # so the allowance below is scoped to free-function bodies only.
        self.is_free_function_body = is_free_function_body
        self.param_names = set(param_names or ())
        # Plain instance methods defined in the SAME class as the forward being
        # traced (``append_visible_tail``, ``get_visible_tokens``, ...), keyed by
        # name. A ``self.<method>(...)`` call site's positional args are the
        # CALLER's local variable names; when the callee is one of these sibling
        # methods, its own declared parameter names are what its inlined body's
        # ops key their ``boundary_input``/entry-param lookups on. Resolving the
        # call against the callee's real signature (mirroring
        # ``_free_function_param_names`` below) keeps a caller/callee local-name
        # mismatch from stranding an argument onto the callee's default target.
        self.class_methods = dict(class_methods or {})
        # Self-submodule attrs called more than once in this forward. A call to one
        # gets a call-site ``@l{lineno}`` step key so two calls to the same child
        # (rotary ``recomposition_frequencies(cos)`` then ``(sin)``) keep distinct
        # predecessors/producer bindings instead of the second overwriting the first.
        self.repeated_submodule_attrs = frozenset(repeated_submodule_attrs or ())
        # Module-level free functions (``apply_rotary_pos_emb_vision``, ...) keyed
        # by name, so a traced synthetic call can map its positional args to the
        # callee's parameter names and route each to the producer feeding it.
        self.module_functions = dict(module_functions or {})
        self.operations: list[ForwardOperation] = []
        self.var_producer: dict[str, str] = {}
        self.var_module_origin: dict[str, str] = {}
        # ``pre_w, post_w, comb_w = ...split(...)`` binds each unpacked local to
        # the single split producer plus the output ordinal it selects. Consumers
        # read one of these locals; the ordinal lets their edge attach to the
        # matching named output port (see ForwardOperation.predecessor_ports).
        self.var_output_ordinal: dict[str, int] = {}
        # ``a, b = x.shape[:2]`` binds ``a``/``b`` to a source dim; record the
        # positional read token (``x.shape[0]``) so a later ``view``/``reshape``
        # arg naming ``a`` resolves to that axis instead of an opaque local.
        self.shape_unpack_tokens: dict[str, str] = {}
        # Locals holding a host-side integer (``number_of_pools = (seq_len + k - 1)
        # // k``) rather than a tensor. Arithmetic over only these (and shape ints /
        # config scalars / int literals) is index/shape bookkeeping — it must not
        # emit fake tensor ops (Add/FloorDivide/Multiply) that dangle when their
        # result feeds a size argument like ``torch.arange(n * k)``.
        self.host_scalar_vars: set[str] = set()
        # Host-scalar locals whose value folds to a concrete int (``output_width =
        # self.index_topk`` → 2048; ``output_width += self.index_kpool - 1``).
        # Lets a slice/pad-to-constant (``topk_indices[..., :output_width]``)
        # resize an axis to the folded width instead of aliasing through, and is
        # kept a superset of the names in ``host_scalar_vars`` that resolve.
        self.host_scalar_values: dict[str, int] = {}
        # Set while resolving the *base* of an in-place slice mutation
        # (``key_states[..., :n].copy_(src)``): that subscript is an lvalue, so the
        # range-slice must NOT materialise a resize ``Slice`` op — the base tensor
        # is written into, not sliced-then-read.
        self._suppress_slice_resize = False
        # Ordered return producers captured as the return statement is walked, so a
        # subscripted return element (``return pool_keys[:, keep], ...``) — which
        # produces a gather op but binds no name — still docks its consumer. Keyed
        # by the element's base name; the name-based ``_extract_forward_return_metadata``
        # misses these because there is no local for the sliced value.
        self.return_producer_order: list[str] = []
        self.return_producer_slots: dict[str, str] = {}
        # ``hidden_shape = (a, b, -1, self.head_dim)`` — a local tuple used as a
        # reshape target. Record the literal so ``view(hidden_shape)`` expands to
        # its dims rather than the un-resolvable variable name.
        self.shape_tuple_vars: dict[str, ast.Tuple] = {}
        # ``name = expr`` bindings kept as raw AST so a loop's dynamic iterable
        # (``for i in hit:`` where ``hit = ...nonzero()``) can be traced back to
        # a config-resolvable static bound (see ``_loop_iteration_count``).
        self._name_value_ast: dict[str, ast.expr] = {}
        self.step_predecessors: dict[str, tuple[str, ...]] = {}
        self.step_predecessor_args: dict[str, dict[str, str]] = {}
        # arg_name -> output ordinal, when a submodule call reads a specific slot
        # of a multi-output producer (``self.k_norm(key_states)`` where
        # ``key_states`` is ordinal 1 of an ``unbind``). Parallel to
        # ``step_predecessor_args``; lets the export fan the producer out into one
        # port per consumed slot instead of collapsing every consumer onto slot 0.
        self.step_predecessor_ordinals: dict[str, dict[str, int]] = {}
        # producer attr -> ordered output names, for a tuple-unpacked synthetic
        # call (``q_embed, k_embed = apply_rotary_pos_emb_vision(...)``) that is
        # not an inline ``self.operations`` entry. Lets the export fan the
        # positional/function node out into one named output port per slot.
        self.step_output_names: dict[str, list[str]] = {}
        # own-step attr -> forward parameter names it reads straight from the
        # module boundary (``apply_rotary_pos_emb_vision(q, k, cos, sin)`` where
        # ``cos, sin = position_embeddings``). These synthetics are their own
        # chain node but read forward params that have no internal producer, so
        # the cross-module predecessor pass needs them named to route the caller's
        # producer (``rotary_pos_emb``) onto this consumer. Origins are the
        # caller-visible parameter (``position_embeddings``), not the unpacked
        # local, so the arg-name keys match the call site's keyword.
        self.step_boundary_params: dict[str, tuple[str, ...]] = {}
        # own-step attr -> {callee-parameter -> (boundary origin, ordinal|None)}
        # for a traced free-function call whose body is inlined. Its ops read the
        # callee's parameter names (``cos``/``sin``); this maps each such param
        # back to the boundary forward input it was fed from at the call site
        # (``position_embeddings``) and, for a tuple-unpacked boundary
        # (``cos, sin = position_embeddings``), the ordinal so the boundary input
        # fans out one port per slot (port0->cos-op, port1->sin-op).
        self.step_boundary_arg_params: dict[
            str, dict[str, tuple[str, int | None]]
        ] = {}
        # unpacked-local -> the forward parameter it aliases
        # (``cos``/``sin`` -> ``position_embeddings``). Populated as
        # ``_propagate_param_alias`` registers the alias.
        self.param_alias_origin: dict[str, str] = {}
        # unpacked-local -> its ordinal within a tuple-unpacked boundary param
        # (``cos`` -> 0, ``sin`` -> 1 for ``cos, sin = position_embeddings``), so
        # a free-function frame can fan the boundary input out per slot.
        self.param_alias_ordinal: dict[str, int] = {}
        # When an ``if``/``else`` assigns the same variable to different producers
        # (e.g. ``attn_output`` = flash ``@attention`` in one branch, a manual
        # ``torch.cat`` in the other), only one survives ``var_producer`` after the
        # merge. Downstream consumers would then orphan the losing branch's
        # producer, turning a real runtime path into a dead-end node. Record the
        # alternatives here (survivor → {losers}) and fold them into consumers'
        # predecessors in a post-pass so both branches stay live and acyclic.
        self.branch_alternatives: dict[str, set[str]] = {}
        self.loop_carried: list[LoopCarriedSpec] = []
        self._used_ids: set[str] = set()
        # Subscript nodes materialised into their own op (``Slice``/``Unsqueeze``).
        # The op owns the boundary param it read, so an enclosing expression must
        # not re-attribute that param to itself (it reads the new op instead).
        self._materialized_subscripts: set[int] = set()

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
        raw_op: str | None = None,
    ) -> str:
        attr_name = self._operation_id(node, label)
        if label.lower() in {"matmul", "matmull"}:
            display = classify_matmul_label(external_inputs=external_inputs)
        else:
            display = operation_display_label(label)
        emitted_details = list(details or ())
        # Carry the underlying torch op name (the one the display label discards)
        # so the downstream type-check can resolve the op's real operand arity from
        # its actual function parameters -- keyed on the name the model itself
        # calls, never a static op-name list. Threaded via the details channel
        # (both build paths carry details) and lifted to a dedicated ``raw_op``
        # node attr in ``merge._annotate_op_input_signatures``.
        if raw_op:
            emitted_details.append(f"raw_op: {raw_op}")
        predecessor_ports = self._read_output_ports(node)
        # ``_param_refs`` returns the raw name this expression reads, which may be
        # a tuple-unpacked alias of a secondary forward parameter (``cos``, ``sin``
        # aliasing ``position_embeddings``) rather than the boundary's own literal
        # name. Downstream boundary-input wiring keys strictly on the class's own
        # forward-parameter names (``root.forward_param_inputs``), so translate
        # each alias back to its origin here -- the same translation already
        # applied to a call step's own ``step_boundary_params`` -- and carry the
        # alias's unpack ordinal alongside it so a per-slot edge can dock onto the
        # right port instead of every alias colliding on the whole tensor.
        raw_param_refs = self._param_refs(node)
        param_inputs = self._dedupe(
            self.param_alias_origin.get(name, name) for name in raw_param_refs
        )
        param_input_ordinals = tuple(
            (self.param_alias_origin.get(name, name), self.param_alias_ordinal[name])
            for name in raw_param_refs
            if name in self.param_alias_ordinal
        )
        self.operations.append(
            ForwardOperation(
                attr_name=attr_name,
                label=display,
                class_name=display,
                predecessors=self._dedupe_predecessors(predecessors, predecessor_ports),
                external_inputs=self._dedupe(external_inputs),
                details=tuple(emitted_details),
                param_inputs=param_inputs,
                predecessor_ports=predecessor_ports,
                param_input_ordinals=param_input_ordinals,
            )
        )
        return attr_name

    @staticmethod
    def _dedupe_predecessors(
        values: list[str], ports: tuple[tuple[str, int], ...]
    ) -> tuple[str, ...]:
        """Drop repeats, except a producer read at several distinct output
        ordinals within this same expression keeps one entry per ordinal.

        ``torch.cat((q_pass, q_rot), dim=-1)`` reassembling a ``torch.split``
        reads the SAME split producer twice, at two different output slots.
        A plain identity dedupe (as for every other predecessor list) would
        collapse that to a single entry and silently drop the second slice's
        edge. ``ports`` (see ``_read_output_ports``) records how many
        distinct ordinals each producer here is actually read at, so this
        keeps exactly that many copies -- one every other repeat of the same
        producer (not backed by a distinct ordinal) is still deduped away.
        """
        repeat_needed: dict[str, int] = {}
        for producer, _ordinal in ports:
            repeat_needed[producer] = repeat_needed.get(producer, 0) + 1
        kept: list[str] = []
        seen_count: dict[str, int] = {}
        for value in values:
            if not value:
                continue
            limit = max(repeat_needed.get(value, 0), 1)
            count = seen_count.get(value, 0)
            if count < limit:
                kept.append(value)
                seen_count[value] = count + 1
        return tuple(kept)

    def _emit_branch_select(
        self,
        node: ast.AST,
        survivor_producer: str,
        other_producer: str,
        test: str,
    ) -> str:
        """Emit an explicit Select (phi) node joining two mutually-exclusive branch
        producers of one reassigned variable, and return its id.

        The two producers come from the taken/not-taken arms of an ``if`` whose
        predicate could not be statically resolved, so exactly one runs per
        invocation. Rendering an explicit merge keeps both branch computations
        reachable while giving downstream consumers a single tensor to read. The
        node is built directly (not via ``_emit``) so its label stays ``Select``
        -- it must not be display-mapped onto ``Slice`` nor resolve to
        ``aten::select`` -- and it carries no ``raw_op``, so the arity type-check
        skips it (a phi legitimately takes N tensor operands).
        """
        attr_name = self._operation_id(node, "Select")
        self.operations.append(
            ForwardOperation(
                attr_name=attr_name,
                label="Select",
                class_name="Select",
                predecessors=self._dedupe([survivor_producer, other_producer]),
                details=(f"select: {test}",),
            )
        )
        return attr_name

    def _read_output_ports(self, node: ast.AST) -> tuple[tuple[str, int], ...]:
        """Producer→ordinal pairs for multi-output locals this expression reads.

        When an operation reads ``comb_w`` (unpacked as ordinal 2 of a split), it
        consumes that specific output port, not the whole split. Walk the
        expression for such locals so the graph can wire the edge to the matching
        port. An op that reassembles two different slices of the *same* split in
        one expression (``torch.cat((q_pass, q_rot), dim=-1)``) reads it at two
        distinct ordinals -- keep every distinct (producer, ordinal) pair, in
        read order, so each slice keeps its own port instead of the second read
        silently collapsing onto the first.
        """
        if not self.var_output_ordinal:
            return ()
        ports: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for current in ast.walk(node):
            if isinstance(current, ast.Name):
                ordinal = self.var_output_ordinal.get(current.id)
                if ordinal is None:
                    continue
                producer = self.var_producer.get(current.id)
                if not producer:
                    continue
                key = (producer, ordinal)
                if key not in seen:
                    seen.add(key)
                    ports.append(key)
        return tuple(ports)

    def _param_refs(self, node: ast.AST) -> tuple[str, ...]:
        """Secondary forward parameters this operation's expression reads.

        A nested call that becomes its own chain step (a submodule call such as
        ``self.attn(...)``, or an attention/positional kernel) owns the params
        passed to it: ``h + self.attn(norm1(h), cu_seqlens=cu_seqlens)`` must
        not attribute ``cu_seqlens`` to the residual ``Add``. So we do not
        descend into those sub-calls — only params read directly by this
        operation's own expression count.
        """
        if not self.param_names:
            return ()
        names: list[str] = []

        def _owns_own_step(call: ast.Call) -> bool:
            func = call.func
            if isinstance(func, ast.Attribute) and _is_self_attr(func, func.attr):
                return True
            method = func.attr if isinstance(func, ast.Attribute) else None
            return self._call_step_producer(call, method) is not None

        def _visit(current: ast.AST, is_root: bool) -> None:
            # A host-side shape/size read (``position_ids.shape[0]``) reads a
            # param only to compute a Python int, not as a tensor operand: the
            # param it names there must not count as a param this operation
            # *consumes* (which would wire the param onto the op as a spurious
            # extra tensor edge downstream). Mirrors the other subtree-ownership
            # skips below.
            if not is_root and self._is_host_scalar_expr(current):
                return
            if (
                not is_root
                and isinstance(current, ast.Call)
                and _owns_own_step(current)
            ):
                return
            # A subscript already emitted as its own Slice/Unsqueeze op owns the
            # boundary param it read; the enclosing op reads that op, not the param.
            if (
                not is_root
                and isinstance(current, ast.Subscript)
                and id(current) in self._materialized_subscripts
            ):
                return
            if isinstance(current, ast.Name) and current.id in self.param_names:
                names.append(current.id)
            for child in ast.iter_child_nodes(current):
                _visit(child, False)

        _visit(node, True)
        return self._dedupe(names)

    def _call_step_producer(
        self, node: ast.Call, method_name: str | None
    ) -> str | None:
        """Chain step a call *is*, for calls the diagram turns into their own node.

        Mirrors the naming `_extract_self_calls_ordered` uses, so the producer recorded
        here refers to the same node the forward chain will hold.
        """
        func = node.func
        if method_name is not None and _is_self_attr(func, method_name):
            if method_name in self.repeated_submodule_attrs:
                return submodule_callsite_attr(method_name, node.lineno)
            return method_name
        target = _expr_name(func)
        if target and (
            target in _SYNTHETIC_ATTENTION_NAMES
            or (isinstance(func, ast.Name) and _is_kernel_merge_call(func))
        ):
            return SYNTHETIC_ATTENTION
        # A ``map(lambda x: BODY(x), (a, b))`` idiom clones BODY's own call node
        # once per tuple element (see ``_expand_map_lambda_tuple``); every clone
        # shares BODY's original source position, so the discriminator stamped on
        # the clone is what keeps their synthetic keys from colliding.
        discriminator = getattr(node, "_tracelens_map_discriminator", None)
        if target and _is_positional_function_call(func, target):
            return positional_synthetic_attr(target, node.lineno, discriminator)
        if _is_emittable_free_function(func, target):
            return function_synthetic_attr(target, node.lineno, discriminator)
        return None

    def _free_function_param_names(self, node: ast.Call) -> list[str] | None:
        """Positional parameter names of the module-level function *node* calls.

        Lets a traced free-function call align its call-site args with the
        callee's signature (``apply_rotary_pos_emb_vision(q, k, cos, sin)``), so
        each argument routes to the parameter it feeds once the body is inlined.
        """
        func = node.func
        if not isinstance(func, ast.Name):
            return None
        definition = self.module_functions.get(func.id)
        if definition is None:
            return None
        return [arg.arg for arg in definition.args.posonlyargs + definition.args.args]

    def _class_method_param_names(self, method_name: str) -> list[str] | None:
        """Positional parameter names of a same-class sibling method's own ``def``.

        A ``self.append_visible_tail(topk_indices, visible_tokens, valid_keys)``
        call site names its args after the CALLER's locals; the callee's own
        parameter names (``topk_indices, token_visible, key_valid``) are what its
        inlined body actually keys on. Resolving against the callee's real
        signature — analogous to ``_free_function_param_names`` for module-level
        functions — keeps a caller/callee name mismatch from losing an argument.

        Deliberately scoped to a literal same-class sibling ``def`` only (its
        body is inlined into THIS frame, so the frame's own steps key on the
        callee's parameter names directly). A genuine ``nn.Module`` submodule
        attribute is NOT resolved here: that submodule keeps its own separate
        namespace/boundary, and OTHER machinery in this same extractor
        (``var_producer``-style lookups for a locally reassigned name such as
        ``query_states = self.q_norm(query_states)``) depends on the call's
        ``arg_name_map`` staying keyed by the CALLER's own local variable names.
        Resolving against the callee's own signature here would rename that key
        and sever those same-forward lookups. A submodule caller/callee name
        mismatch is instead resolved downstream, structurally, by ordinal
        position (see ``_lookup_param_entry`` in ``computation_graph.py``).
        """
        definition = self.class_methods.get(method_name)
        if definition is None:
            return None
        args = definition.args
        names = [arg.arg for arg in args.posonlyargs + args.args]
        if names and names[0] == "self":
            names = names[1:]
        return names

    def _self_attr_input(self, node: ast.Attribute) -> tuple[str | None, list[str]]:
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            # Attributes recorded by the constructor pass are settings, not tensor
            # inputs. This covers concretely-resolved scalars (``self.hidden_size``
            # = 4096) AND scalar-typed attributes whose value could not be resolved
            # because a config key was absent (``self.qkv_dim = self.head_dim *
            # self.num_heads`` when ``linear_head_dim`` is not serialized) -- both
            # are stored, the latter under the ``_SCALAR_SETTING`` sentinel. A
            # genuine tensor attribute (an ``nn.Parameter`` / ``register_buffer`` /
            # submodule assignment) is never a scalar-typed constructor expression,
            # so it is NOT recorded and remains a tensor input here (materialized as
            # a Constant leaf). This keeps a scalar size/axis attribute used as an
            # op argument (``torch.split(x, [self.qkv_dim] * 3, -1)``) off the
            # tensor edges instead of fabricating a spurious rank-1 Constant operand.
            if self.self_values.get(node.attr, _UNKNOWN) is not _UNKNOWN:
                return None, []
            return None, [node.attr]
        return None, []

    def _return_element_label(self, node: ast.AST) -> str | None:
        """Base name of a return element, seeing through a trailing subscript
        or a trailing housekeeping method call.

        ``pool_keys[:, keep]`` -> ``pool_keys``; a bare ``pool_keys`` -> ``pool_keys``;
        ``cos.to(dtype=x.dtype)`` -> ``cos``. Used to name the return slot a
        subscripted/cast return produces -- without seeing through the call, a
        tuple return like ``return cos.to(...), sin.to(...)`` (no bare-Name
        elements) gets no slot names at all, so neither element becomes an
        output port and its producer looks unconsumed.
        """
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            return self._return_element_label(node.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            return self._return_element_label(node.func.value)
        if isinstance(node, ast.BinOp):
            # A returned element can be a scaled/combined tensor
            # (``weights * self.routed_scaling_factor``): one side is the real
            # local tensor being returned, the other a bare ``self.<attr>``/
            # literal multiplier that has no base name of its own. See through
            # to whichever side resolves to a name so this element still gets
            # a return slot -- without one, ``_live_forward_steps`` never seeds
            # from whatever produced it, and prunes it (and everything only it
            # depends on) as if the return statement never read it at all.
            return self._return_element_label(
                node.left
            ) or self._return_element_label(node.right)
        return None

    def _is_host_scalar_expr(self, node: ast.AST) -> bool:
        """A pure host-side integer expression (shape math / index bookkeeping).

        True for int literals, shape-unpacked locals (``seq_len`` from
        ``x.shape[:2]``), integer config scalars (``self.index_kpool``), ``len(...)``,
        ``<tensor>.shape[i]``, and arithmetic combining only these. Such expressions
        compute Python ints for sizes/offsets, not tensors, so emitting them as
        tensor ops would leave dangling nodes. Requires *every* operand to be host
        scalar — a tensor operand (``pool_indices + offsets``) makes it False.
        """
        if isinstance(node, ast.Constant):
            return isinstance(node.value, int) and not isinstance(node.value, bool)
        if isinstance(node, ast.Name):
            return node.id in self.shape_unpack_tokens or node.id in self.host_scalar_vars
        if isinstance(node, ast.Attribute):
            value = _config_value(node, self.config, self.self_values)
            return isinstance(value, int) and not isinstance(value, bool)
        if isinstance(node, ast.Subscript):
            base = node.value
            return isinstance(base, ast.Attribute) and base.attr == "shape"
        if isinstance(node, ast.BinOp):
            return self._is_host_scalar_expr(node.left) and self._is_host_scalar_expr(
                node.right
            )
        if isinstance(node, ast.UnaryOp):
            return self._is_host_scalar_expr(node.operand)
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            # ``len(...)`` is always a Python int. ``min``/``max``/``int``/... over
            # host-scalar arguments (``select_k = min(self.index_topk //
            # self.index_kpool, scores.shape[-1])``) is size/budget bookkeeping,
            # not a tensor reduction -- a bare-builtin call whose every argument is
            # itself host-scalar stays host-scalar. A method / namespaced reduction
            # (``scores.min()``, ``torch.max(t)``) is NOT a bare Name and operates
            # on tensors, so it is excluded and still emits a real op.
            if func_name == "len":
                return True
            if func_name in {"min", "max", "int", "abs", "round", "sum"}:
                return bool(node.args) and all(
                    self._is_host_scalar_expr(arg) for arg in node.args
                )
            return False
        return False

    def _fold_host_int(self, node: ast.AST) -> int | None:
        """Fold a host-scalar expression to a concrete non-negative int, else None.

        Resolves int literals, previously-folded host-scalar locals
        (``host_scalar_values``), integer ``self.<attr>``/``config.<attr>``
        scalars (via the constructor-resolved ``self_values`` and the config
        dict), and ``+ - * // %`` / unary-sign arithmetic over those. Used to give
        a slice/pad-to-constant bound (``topk_indices[..., :output_width]``) its
        concrete width so shape inference can resize the axis. Returns *None* for
        anything not statically resolvable (kept general — no name allowlist).
        """
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int) and not isinstance(node.value, bool):
                return node.value
            return None
        if isinstance(node, ast.Name):
            return self.host_scalar_values.get(node.id)
        if isinstance(node, ast.Attribute):
            value = _config_value(node, self.config, self.self_values)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = self._fold_host_int(node.operand)
            if operand is None:
                return None
            return -operand if isinstance(node.op, ast.USub) else operand
        if isinstance(node, ast.BinOp):
            left = self._fold_host_int(node.left)
            right = self._fold_host_int(node.right)
            if left is None or right is None:
                return None
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
                return None
        return None

    def _record_host_scalar(self, target: ast.AST, value: ast.AST) -> None:
        """Fold ``target = value``/``target += …`` to an int and cache it, if host-scalar."""
        if not isinstance(target, ast.Name):
            return
        folded = self._fold_host_int(value)
        if folded is not None and folded >= 0:
            self.host_scalar_values[target.id] = folded
        else:
            # A reassignment to a now-unresolvable value must not keep the stale int.
            self.host_scalar_values.pop(target.id, None)

    def _subscript_resize_dims(self, index: ast.AST) -> list[tuple[int, int]]:
        """Axes a bounded range-slice (``x[..., :output_width]``) resizes to a constant.

        Returns ``(axis, size)`` pairs for each ``ast.Slice`` element whose bound(s)
        fold to a concrete int — ``:B`` → size ``B``; ``a:b`` → size ``b - a`` — so
        shape inference can set that axis to the folded width instead of passing the
        source axis through unchanged. Axes after an ``Ellipsis`` are numbered from
        the end (``x[..., :B]`` resizes ``-1``). Slices with a non-static or full
        (``:``) bound, and integer selects (handled by ``_subscript_select_dims``),
        are skipped. Returns ``[]`` when nothing resolves — the caller then aliases.
        """
        elts = index.elts if isinstance(index, ast.Tuple) else [index]
        ellipsis_at = next(
            (
                pos
                for pos, elt in enumerate(elts)
                if isinstance(elt, ast.Constant) and elt.value is Ellipsis
            ),
            None,
        )
        resize: list[tuple[int, int]] = []
        for pos, elt in enumerate(elts):
            if not isinstance(elt, ast.Slice) or elt.step is not None:
                continue
            if elt.upper is None:
                continue
            upper = self._fold_host_int(elt.upper)
            if upper is None:
                continue
            if elt.lower is None:
                size = upper
            else:
                lower = self._fold_host_int(elt.lower)
                if lower is None:
                    continue
                size = upper - lower
            if size < 0:
                continue
            if ellipsis_at is None or pos < ellipsis_at:
                axis = pos
            else:
                axis = -(len(elts) - pos)
            resize.append((axis, size))
        return resize

    @staticmethod
    def _shape_relative_expr(bound: ast.AST, base_ast: ast.AST) -> str | None:
        """Rewrite ``<base>.shape[k]`` refs in an arithmetic slice bound to a ``shape[k]`` symbol.

        ``rotate_half`` slices ``x[..., : x.shape[-1] // 2]``: the bound cannot fold
        to a static int at extraction (it reads ``x.shape``) but IS concrete at shape
        inference, which knows ``x``'s shape. Returns a normalized expression string
        (``shape[-1] // 2``) when ``bound`` is pure integer arithmetic over the sliced
        operand's OWN ``.shape[...]`` and int constants; ``None`` for any other
        reference (a config symbol, a different tensor's shape), leaving the slice
        descriptive/pass-through. General: only the operand's own shape is resolvable
        from the operand's inferred shape alone.
        """
        base_dump = ast.dump(base_ast)
        ok = True

        def _const_int_index(idx: ast.AST) -> int | None:
            if isinstance(idx, ast.Constant) and isinstance(idx.value, int):
                return idx.value
            if (
                isinstance(idx, ast.UnaryOp)
                and isinstance(idx.op, ast.USub)
                and isinstance(idx.operand, ast.Constant)
                and isinstance(idx.operand.value, int)
            ):
                return -idx.operand.value
            return None

        class _Rewriter(ast.NodeTransformer):
            def visit_Subscript(self, n: ast.Subscript):
                v = n.value
                if (
                    isinstance(v, ast.Attribute)
                    and v.attr == "shape"
                    and ast.dump(v.value) == base_dump
                ):
                    axis = _const_int_index(n.slice)
                    if axis is not None:
                        return ast.copy_location(
                            ast.Subscript(
                                value=ast.Name(id="shape", ctx=ast.Load()),
                                slice=ast.Constant(value=axis),
                                ctx=ast.Load(),
                            ),
                            n,
                        )
                return self.generic_visit(n)

        rewritten = _Rewriter().visit(ast.parse(ast.unparse(bound), mode="eval").body)

        # Validate: only ``shape[int]`` reads, int constants, and +/-/*//// arithmetic.
        for sub in ast.walk(rewritten):
            if isinstance(sub, ast.Subscript):
                if not (isinstance(sub.value, ast.Name) and sub.value.id == "shape"):
                    ok = False
            elif isinstance(sub, ast.Name):
                if sub.id != "shape":
                    ok = False
            elif isinstance(sub, ast.BinOp):
                if not isinstance(
                    sub.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div)
                ):
                    ok = False
            elif isinstance(sub, ast.Constant):
                if not isinstance(sub.value, int):
                    ok = False
            elif isinstance(
                sub,
                (
                    ast.UnaryOp,
                    ast.operator,
                    ast.unaryop,
                    ast.expr_context,
                    ast.Expression,
                ),
            ):
                # Operator / context marker nodes (``FloorDiv``, ``USub``, ``Load`` …)
                # yielded by ``ast.walk`` carry no operands to validate.
                continue
            else:
                ok = False
        if not ok:
            return None
        # Require at least one shape reference — a bound of pure constants would
        # already have folded via ``_subscript_resize_dims``.
        if not any(
            isinstance(s, ast.Name) and s.id == "shape" for s in ast.walk(rewritten)
        ):
            return None
        return ast.unparse(rewritten)

    def _subscript_shape_relative_dims(
        self, node: ast.Subscript
    ) -> list[tuple[int, str | None, str | None]]:
        """Axes a range-slice narrows using bounds over the operand's OWN shape.

        Returns ``(axis, lower_expr, upper_expr)`` per ``ast.Slice`` element whose
        present bound(s) are pure arithmetic over ``<this operand>.shape[k]`` (a
        ``None`` bound stays ``None``); shape inference evaluates the exprs against the
        operand's concrete shape to size the axis. Skips an element whose present bound
        is NOT shape-relative (a plain symbol we cannot size). ``[]`` when nothing is
        shape-relative — caller falls back to a descriptive pass-through slice.
        """
        base_ast = node.value
        index = node.slice
        elts = index.elts if isinstance(index, ast.Tuple) else [index]
        ellipsis_at = next(
            (
                pos
                for pos, elt in enumerate(elts)
                if isinstance(elt, ast.Constant) and elt.value is Ellipsis
            ),
            None,
        )
        out: list[tuple[int, str | None, str | None]] = []
        for pos, elt in enumerate(elts):
            if not isinstance(elt, ast.Slice) or elt.step is not None:
                continue
            if elt.lower is None and elt.upper is None:
                continue
            lower_expr = (
                self._shape_relative_expr(elt.lower, base_ast)
                if elt.lower is not None
                else None
            )
            upper_expr = (
                self._shape_relative_expr(elt.upper, base_ast)
                if elt.upper is not None
                else None
            )
            if elt.lower is not None and lower_expr is None:
                continue
            if elt.upper is not None and upper_expr is None:
                continue
            if lower_expr is None and upper_expr is None:
                continue
            if ellipsis_at is None or pos < ellipsis_at:
                axis = pos
            else:
                axis = -(len(elts) - pos)
            out.append((axis, lower_expr, upper_expr))
        return out

    @staticmethod
    def _subscript_narrows_range(index: ast.AST) -> bool:
        """True when a subscript contains a partial range slice on some axis.

        A partial ``ast.Slice`` -- one with a ``lower``, ``upper``, or ``step``
        bound (``x[:, :, -seq_len:]``, ``x[..., 1:]``, ``x[::2]``) -- narrows or
        strides that axis, so it is a real ``Slice`` op even when the bound is a
        symbol that cannot fold to a static width. A full ``:`` (all bounds
        ``None``) and a bare ellipsis carry no narrowing and stay pass-through.
        Foldable bounded slices are handled first by ``_subscript_resize_dims``;
        this catches the non-foldable remainder so the op is never dropped.
        """
        elts = index.elts if isinstance(index, ast.Tuple) else [index]
        return any(
            isinstance(elt, ast.Slice)
            and (elt.lower is not None or elt.upper is not None or elt.step is not None)
            for elt in elts
        )

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
            # A host-side shape read (``x.shape[i]``) is index bookkeeping, not a
            # tensor read: it computes a Python int, not data. Without this guard
            # the fallback below ("preserve the computation behind chained
            # property access", meant for selectors like ``.topk(...).indices``)
            # also fires for ``.shape[i]`` and hands back the BASE tensor's own
            # producer as if the shape read were that tensor itself -- so a
            # downstream host-int expression built from it (``num_key_blocks =
            # -(-k_len // self.block_size)``) fails its own host-scalar check and
            # gets materialized as a real op reading the base tensor as a bogus
            # operand. Mirrors the ``ast.BinOp`` guard above.
            if self._is_host_scalar_expr(node):
                return None, []
            base, base_external = self.expression(node.value)
            # Advanced indexing (``x[idx]`` where ``idx`` is a tensor, e.g.
            # ``pool_indices[batch_idx, selected]`` or a boolean mask) is a gather:
            # the index tensor is a genuine data consumer, not slicing. Plain
            # slices/ints/``None``/``...`` carry no producer and stay pass-through.
            index_producers: list[str] = []
            index_external: list[str] = []
            for operand in _subscript_index_operands(node.slice):
                producer, operand_external = self.expression(operand)
                if producer:
                    index_producers.append(producer)
                index_external.extend(operand_external)
            if index_producers:
                producer = self._emit(
                    node,
                    "Gather",
                    [value for value in (base, *index_producers) if value],
                    [*base_external, *index_external],
                )
                return producer, []
            # A pure slice that selects a single index along a non-sole axis
            # (``freq[:, 0]``) drops that axis: a real ``Slice`` op, not an alias.
            # Materialising it keeps a later ``cat([freq_h, freq_w])`` reading two
            # distinct producers instead of the same base twice. A boundary param
            # read directly (``position_ids[..., None]``) has no producer but wires
            # through ``param_inputs``; still worth its own op.
            reads_param = bool(self._param_refs(node))
            if base is not None or base_external or reads_param:
                base_predecessors = [base] if base else []
                select_dims = _subscript_select_dims(node.slice)
                if select_dims:
                    self._materialized_subscripts.add(id(node))
                    producer = self._emit(
                        node,
                        "Slice",
                        base_predecessors,
                        base_external,
                        details=[
                            "select_dim: " + ", ".join(str(dim) for dim in select_dims)
                        ],
                    )
                    return producer, []
                # A bounded range-slice to a config-derived constant width
                # (``topk_indices[..., :output_width]``) resizes that axis to the
                # folded size — a real ``Slice``, not a pass-through alias, so the
                # axis reads its true width even when the source axis was inflated
                # by proven data-dependent internals upstream. Skip a host-scalar
                # shape read (``hidden_states.shape[:2]``): it is index bookkeeping
                # and must emit no tensor op.
                resize_dims = (
                    []
                    if self._suppress_slice_resize or self._is_host_scalar_expr(node)
                    else self._subscript_resize_dims(node.slice)
                )
                if resize_dims:
                    self._materialized_subscripts.add(id(node))
                    producer = self._emit(
                        node,
                        "Slice",
                        base_predecessors,
                        base_external,
                        details=[
                            "resize_dim: "
                            + ", ".join(f"{axis}={size}" for axis, size in resize_dims)
                        ],
                    )
                    return producer, []
                # A range slice whose bound is arithmetic over the operand's OWN
                # shape (``rotate_half``'s ``x[..., : x.shape[-1] // 2]``) narrows the
                # axis to a width that is not a static int at extraction but IS
                # concrete at shape inference. Emit a ``shape_slice`` detail carrying
                # the per-axis lower/upper expressions (over a ``shape`` symbol) so the
                # inferencer sizes the axis instead of passing the shape through.
                shape_rel_dims = (
                    []
                    if self._suppress_slice_resize or self._is_host_scalar_expr(node)
                    else self._subscript_shape_relative_dims(node)
                )
                if shape_rel_dims:
                    self._materialized_subscripts.add(id(node))
                    producer = self._emit(
                        node,
                        "Slice",
                        base_predecessors,
                        base_external,
                        details=[
                            "shape_slice: "
                            + ", ".join(
                                f"{axis}={lo or ''}|{up or ''}"
                                for axis, lo, up in shape_rel_dims
                            )
                        ],
                    )
                    return producer, []
                # A partial range slice whose bound does not fold to a static width
                # (``mixed_qkv[:, :, -seq_len:]``) still narrows/strides an axis: a
                # real ``Slice`` op, not a silent alias. The bound is symbolic, so
                # the detail is descriptive only (shape inference cannot size it and
                # passes the shape through), but the op stays visible in the graph.
                # Require a real upstream tensor producer OR a genuine param read
                # (``reads_param``) in one of two shapes: a range slice over a host
                # read with no producer and no param binding (``cu_seqlens[1:] -
                # cu_seqlens[:-1]`` index bookkeeping, read directly off a class's
                # own top-level ``forward`` parameter) has nothing to slice and must
                # stay pass-through, not become an orphan op -- ``cu_seqlens``
                # genuinely is a real forward parameter there too, so ``reads_param``
                # alone cannot tell the two cases apart. Two structural shapes DO
                # still need the visible ``Slice``, though: (1) a free function's own
                # non-primary tensor parameter (``k_rot, k_pass = k[..., :rotary_dim],
                # k[..., rotary_dim:]`` in ``apply_rotary_pos_emb`` when only ``q`` is
                # seeded as the boundary input) has no producer of its own at this
                # point either, but IS a real tensor operand; and (2) a *secondary*
                # forward input's tuple-unpack alias sliced down before being passed
                # on (``cos, sin = position_embeddings; ... cos[..., :self.head_dim]``,
                # a smaller rotary width for one submodule) is likewise a real operand
                # with no producer of its own -- unlike ``cu_seqlens``, its own name
                # is never itself the function's top-level parameter, only an alias
                # registered by ``_propagate_param_alias``, which is exactly what
                # distinguishes it from the index-bookkeeping case. ``is_free_function_body``
                # covers (1); ``is_param_alias`` covers (2).
                # Skip host-scalar shape reads too, as the resize branch does.
                is_param_alias = (
                    isinstance(node.value, ast.Name)
                    and node.value.id in self.param_alias_origin
                )
                if (
                    (
                        base is not None
                        or (reads_param and (self.is_free_function_body or is_param_alias))
                    )
                    and not self._suppress_slice_resize
                    and not self._is_host_scalar_expr(node)
                    and self._subscript_narrows_range(node.slice)
                ):
                    self._materialized_subscripts.add(id(node))
                    producer = self._emit(
                        node,
                        "Slice",
                        base_predecessors,
                        base_external,
                        details=[f"slice: {ast.unparse(node.slice)}"],
                    )
                    return producer, []
                # ``x[..., None]`` / ``x[:, None]`` inserts a size-1 axis: an
                # unsqueeze, not a pass-through, so downstream broadcasting sees
                # the new axis.
                if _subscript_inserts_axis(node.slice):
                    unsqueeze_dim = _none_insert_dim(node.slice)
                    if unsqueeze_dim is not None:
                        self._materialized_subscripts.add(id(node))
                        producer = self._emit(
                            node,
                            "Unsqueeze",
                            base_predecessors,
                            base_external,
                            details=[f"dim: {unsqueeze_dim}"],
                        )
                        return producer, []
            return base, base_external
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
            # Host-side integer arithmetic (shape/index math) is not a tensor op.
            if self._is_host_scalar_expr(node):
                return None, []
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
        # A host-scalar builtin call (``min``/``max``/``len`` over shape/config
        # ints) computes a Python size, not a tensor: emitting it as an op would
        # dangle a fake reduction node and mis-wire its result onto whatever size
        # argument reads it (``scores.topk(select_k, ...)``). Mirrors the BinOp
        # host-scalar guard above.
        if self._is_host_scalar_expr(node):
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
        # A tensor-method label (``x.float()`` -> Cast, ``x.sum()`` -> Sum, ...)
        # only applies to an attribute-style dispatch on some tensor/namespace
        # value. A bare builtin ``Name`` call that merely shares its name with a
        # tensor method (``float("-inf")``, a Python scalar cast, not
        # ``x.float()``) must not be relabelled as that tensor op -- it has no
        # tensor operand at all, and would otherwise materialize a spurious
        # zero-input op node.
        label = (
            None
            if submodule_call
            else registry_activation
            or _FUNCTION_LABELS.get(functional_name or call_name)
            or (
                _TENSOR_METHOD_LABELS.get(call_name)
                if isinstance(node.func, ast.Attribute)
                else None
            )
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
        arg_ordinal_map: dict[str, int] = {}
        positional_producers: list[list[str]] = []

        def _record_arg_ordinal(name: str, producer: str, arg_node: ast.AST) -> None:
            # If this arg reads a specific output slot of its producer (an
            # unpacked ``unbind``/``split`` local), remember the slot against the
            # arg name so the module-call edge can start from that port.
            for prod, ordinal in self._read_output_ports(arg_node):
                if prod == producer:
                    arg_ordinal_map[name] = ordinal
                    break

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
            # ``self.append_visible_tail(topk_indices, visible_tokens, valid_keys)``
            # names its args after the CALLER's locals; when the callee is a
            # sibling method inlined from its own ``def``, its body keys entry
            # params on ITS OWN parameter names (``token_visible``, ``key_valid``,
            # ...). Resolve those names once so a caller/callee mismatch doesn't
            # strand an argument onto the callee's default target.
            callee_param_names = (
                self._class_method_param_names(method_name)
                if submodule_call and method_name is not None
                else None
            )
            for idx, arg in enumerate(node.args):
                # A host-scalar positional (``key_states.shape[2]``) is an int size
                # read, not a tensor operand: it must contribute no producer, or the
                # shape's base tensor is fabricated as a data dependency / @input.
                if self._is_host_scalar_expr(arg):
                    positional_producers.append([])
                    continue
                producers, arg_external = _collect_call_arg_producers(arg)
                positional_producers.append(producers)
                arg_producers.extend(producers)
                external.extend(arg_external)
                if submodule_call and idx >= start and len(producers) == 1:
                    positional_index = idx - start
                    name = None
                    if callee_param_names is not None and positional_index < len(
                        callee_param_names
                    ):
                        name = callee_param_names[positional_index]
                    if name is None:
                        name = _arg_name(arg, positional_index)
                    arg_name_map[name] = producers[0]
                    _record_arg_ordinal(name, producers[0], arg)
            for keyword in node.keywords:
                # Likewise skip a host-scalar keyword (``kv_length=key_states.shape[2]``):
                # mapping it to the shape's base tensor producer fabricates a phantom
                # @input port on the callee frame (the "key_states" defect).
                if self._is_host_scalar_expr(keyword.value):
                    continue
                producer, arg_external = self.expression(keyword.value)
                if producer:
                    arg_producers.append(producer)
                    if submodule_call and keyword.arg:
                        arg_name_map[keyword.arg] = producer
                        _record_arg_ordinal(keyword.arg, producer, keyword.value)
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
                # A dispatched attention interface call
                # (``attention_interface(self, q, k, v, mask, ...)``) is a bare
                # local, not a ``self.<attr>`` submodule call, so the positional
                # arg-name/ordinal capture above was skipped. Record each tensor
                # operand's name and output ordinal here so the kernel's ports
                # dock onto the correct producer slot — ``value_states`` = slot 1
                # of a tuple-returning ``expand_kv`` — instead of two operands
                # collapsing onto one producer under dedupe and the remaining
                # ports sliding onto the wrong sources. General: reads operands
                # straight off whichever call the source spells out.
                if own_step == SYNTHETIC_ATTENTION and not arg_name_map:
                    attn_start = (
                        1
                        if node.args
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "self"
                        else 0
                    )
                    for arg in node.args[attn_start:]:
                        if not isinstance(arg, ast.Name):
                            continue
                        arg_producer = self.var_producer.get(arg.id)
                        if arg_producer is None:
                            continue
                        # Only an operand that reads a *specific output slot* of a
                        # multi-output producer (``value_states`` = slot 1 of a
                        # tuple-returning ``expand_kv``) needs its port docked; a
                        # single-output operand (``g`` from ``self.forget_gate``)
                        # carries no slot and must not be published as a step
                        # predecessor arg, or it perturbs how the enclosing frame's
                        # child submodule (forget_gate) is expanded downstream.
                        recorded_before = arg.id in arg_ordinal_map
                        _record_arg_ordinal(arg.id, arg_producer, arg)
                        if arg.id in arg_ordinal_map and not recorded_before:
                            arg_name_map.setdefault(arg.id, arg_producer)
                # A traced free-function node (rope helper, ...) is expanded into
                # its body's ops when the callee is known; map each positional arg
                # to the callee's parameter name so the cross-module predecessor
                # pass can route producers onto the matching per-parameter entry of
                # the inlined pipeline (``q``->query_states, ``k``->key_states).
                if not arg_name_map and (
                    is_positional_synthetic(own_step)
                    or is_function_synthetic(own_step)
                ):
                    param_names = self._free_function_param_names(node)
                    boundary_arg_map: dict[str, tuple[str, int | None]] = {}
                    if param_names:
                        for idx, producers in enumerate(positional_producers):
                            if idx >= len(param_names):
                                continue
                            callee_param = param_names[idx]
                            if len(producers) == 1:
                                arg_name_map[callee_param] = producers[0]
                                _record_arg_ordinal(
                                    callee_param, producers[0], node.args[idx]
                                )
                                continue
                            # No internal producer: the arg reads straight from a
                            # boundary forward input (``cos``/``sin``, aliased to
                            # ``position_embeddings``). Map the callee parameter to
                            # that origin and its unpack ordinal so the inlined
                            # frame fans the boundary input out one port per slot.
                            # A unary wrapper (``-sin``) is a transparent
                            # pass-through here too, mirroring ``expression()``'s
                            # own ``ast.UnaryOp`` handling (which just recurses
                            # into the operand): the boundary alias underneath is
                            # still the same tensor, so unwrap it before checking
                            # whether it is a tracked param alias.
                            arg = node.args[idx]
                            while isinstance(arg, ast.UnaryOp):
                                arg = arg.operand
                            if (
                                isinstance(arg, ast.Name)
                                and arg.id in self.param_names
                            ):
                                boundary_arg_map[callee_param] = (
                                    self.param_alias_origin.get(arg.id, arg.id),
                                    self.param_alias_ordinal.get(arg.id),
                                )
                    if boundary_arg_map:
                        self.step_boundary_arg_params[own_step] = boundary_arg_map
                if arg_name_map:
                    self.step_predecessor_args[own_step] = arg_name_map
                if arg_ordinal_map:
                    self.step_predecessor_ordinals[own_step] = arg_ordinal_map
                # A synthetic that is its own node may read forward parameters that
                # have no internal producer (``apply_rotary_pos_emb_vision(..., cos,
                # sin)`` where ``cos, sin = position_embeddings``). Record the
                # caller-visible origin (``position_embeddings``) so the cross-module
                # predecessor pass can route the producer feeding that parameter onto
                # this consumer, the way ``cu_seqlens`` reaches the kernel.
                boundary = self._dedupe(
                    [
                        self.param_alias_origin.get(name, name)
                        for name in self._param_refs(node)
                    ]
                )
                if boundary:
                    self.step_boundary_params[own_step] = boundary
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
            details.append("shape: " + self._format_shape_args(node.args))
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
        if call_name == "permute":
            is_method = isinstance(node.func, ast.Attribute) and not (
                isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
            )
            arg_start = 0 if is_method else 1
            dims_args = node.args[arg_start:]
            # ``permute`` accepts either varargs (``x.permute(0, 2, 1, 3)``) or a
            # single tuple/list (``x.permute((0, 2, 1, 3))``); flatten both to a
            # comma-joined dims spec so shape inference can reorder the axes.
            if len(dims_args) == 1 and isinstance(
                dims_args[0], (ast.Tuple, ast.List)
            ):
                dims_args = list(dims_args[0].elts)
            if dims_args:
                details.append(
                    "dims: " + ", ".join(ast.unparse(arg) for arg in dims_args)
                )
        if call_name == "einsum":
            if node.args and isinstance(node.args[0], ast.Constant):
                details.append(f"equation: {node.args[0].value}")
        if call_name == "flatten":
            # ``Tensor.flatten(start_dim=0, end_dim=-1)`` /
            # ``torch.flatten(t, start_dim, end_dim)`` collapse the axes in
            # ``[start_dim, end_dim]`` into one. Record both so shape inference
            # can compute the merged extent instead of passing the tensor
            # through unchanged (which left phantom rank, e.g. topk_indices).
            is_method = isinstance(node.func, ast.Attribute) and not (
                isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
            )
            arg_start = 0 if is_method else 1
            flatten_args = node.args[arg_start:]
            if len(flatten_args) >= 1:
                details.append(f"start_dim: {ast.unparse(flatten_args[0])}")
            if len(flatten_args) >= 2:
                details.append(f"end_dim: {ast.unparse(flatten_args[1])}")
            for keyword in node.keywords:
                if keyword.arg in {"start_dim", "end_dim"}:
                    details.append(f"{keyword.arg}: {ast.unparse(keyword.value)}")
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
        if label in {"Concat", "Stack"}:
            # The assembly axis drives the output width; record it so shape
            # inference sums (concat) or tiles along the right dim.
            dim_arg = next(
                (kw.value for kw in node.keywords if kw.arg == "dim"), None
            )
            if dim_arg is None and len(node.args) > 1:
                dim_arg = node.args[1]
            if dim_arg is not None:
                details.append(f"dim: {ast.unparse(dim_arg)}")
        emit_predecessors = [
            value for value in (base_producer, *arg_producers) if value
        ]
        if label == "Concat" and len(emit_predecessors) >= 2:
            # ``cat([freq_hw, freq_hw])`` concatenates one tensor with itself: the
            # deduped edge would collapse to a single-input concat that looks
            # inert. It is really a Tile (repeat k along the concat dim). A repeat
            # of the same producer *string* is not automatically a repeat of the
            # same value, though: reassembling two different slices of one
            # multi-output split (``torch.cat((q_pass, q_rot), dim=-1)``) shares a
            # producer but reads two distinct output ordinals from it, so it is a
            # genuine concatenation of two different tensors, not a self-repeat.
            distinct = dict.fromkeys(emit_predecessors)
            ordinal_counts: dict[str, int] = {}
            for producer, _ordinal in self._read_output_ports(node):
                ordinal_counts[producer] = ordinal_counts.get(producer, 0) + 1
            multi_ordinal_producers = {
                producer for producer, count in ordinal_counts.items() if count > 1
            }
            if len(distinct) == 1 and not (set(distinct) & multi_ordinal_producers):
                label = "Tile"
                details.append(f"repeat: {len(emit_predecessors)}")
        producer = self._emit(
            node,
            label,
            emit_predecessors,
            external,
            details=details,
            # The raw callable/method name the model itself calls at this site
            # (``transpose``/``cat``/``view``/...), read straight from the AST --
            # not a static op list. The display label discards it; the type-check
            # needs it to resolve the op's real operand arity from its parameters.
            raw_op=functional_name or call_name,
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
            # A reassignment drops any stale tuple-unpack ordinal: ``up`` bound to
            # chunk slice 1 by ``gate, up = x.chunk(2)`` becomes a fresh single-
            # output value after ``up = up.clamp(...)``. Without clearing, a later
            # read of ``up`` would still dock onto slice 1 of the chunk and the
            # real (reassigned) producer's edge would carry a dangling port.
            # ``_record_output_unpack`` re-stamps genuine unpack targets right
            # after this. General: any single-name reassignment.
            self.var_output_ordinal.pop(name, None)

    _MULTI_OUTPUT_LABELS = frozenset({"Split", "Chunk", "Unbind"})

    def _record_output_unpack(
        self, stmt: ast.Assign | ast.AnnAssign, producer: str | None
    ) -> None:
        """Record a ``a, b, c = <split/chunk/unbind>`` tuple-unpack.

        A single split op produces several tensors; unpacking names them. Tag each
        unpacked local with its output ordinal (so consumers wire to the matching
        port) and stamp the ordered names onto the producer op (so the graph can
        render one named output port per slice with its own shape). General: fires
        for any multi-output tensor call, not just the GLM hyperconnection splits.
        """
        if producer is None or not isinstance(stmt, ast.Assign):
            return
        if len(stmt.targets) != 1:
            return
        target = stmt.targets[0]
        if not isinstance(target, (ast.Tuple, ast.List)):
            return
        names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
        if len(names) < 2 or len(names) != len(target.elts):
            return
        # A tuple-returning positional kernel
        # (``q_embed, k_embed = apply_rotary_pos_emb_vision(...)``) is its own
        # chain node, not an inline ``self.operations`` entry. Record the ordered
        # output names against the producer attr (threaded to the block node) and
        # stamp each local's ordinal explicitly, so consumers wire to the matching
        # port and the node fans out one named output per slot. General: fires for
        # any multi-output positional synthetic, no class-name checks.
        if is_positional_synthetic(producer):
            self.step_output_names[producer] = names
            for ordinal, name in enumerate(names):
                self.var_output_ordinal[name] = ordinal
            return
        for index, operation in enumerate(self.operations):
            if operation.attr_name != producer:
                continue
            if operation.label not in self._MULTI_OUTPUT_LABELS:
                return
            self.operations[index] = replace(operation, output_names=tuple(names))
            for ordinal, name in enumerate(names):
                self.var_output_ordinal[name] = ordinal
            return
        # A tuple-unpack of a submodule/method call expanded as its own frame
        # (``key_states, value_states = self.expand_kv(...)``). The producer is
        # not an inline op in this scope, so tag each unpacked local with its
        # return ordinal and publish the ordered slot names. A consumer reading a
        # specific local (``value_states``) then docks onto the matching frame
        # return slot instead of collapsing every local onto the frame tail (and
        # being dropped by predecessor dedupe). General: fires for any
        # tuple-returning call, no class-name checks.
        if isinstance(stmt.value, ast.Call):
            self.step_output_names[producer] = names
            for ordinal, name in enumerate(names):
                self.var_output_ordinal[name] = ordinal

    def _propagate_param_alias(
        self, targets: list[ast.expr], value: ast.AST
    ) -> None:
        """Carry a secondary forward-input's param status onto unpacked locals.

        ``cos, sin = position_embeddings`` (and the plain ``x = position_embeddings``
        rename) binds new names that alias a forward parameter but are otherwise
        invisible to ``_param_refs`` — its gate only recognizes names literally in
        ``self.param_names``. Without this, downstream reads of ``cos``/``sin`` (the
        rotary path) resolve to nothing and the operation is dropped. When the RHS
        is itself a param (or an already-registered alias), register every unpacked
        target name as a param alias so ``_param_refs`` attributes it like the
        original forward input. This is general: any secondary forward input renamed
        or unpacked into locals is tracked.

        A forward input keyed by a config value before being unpacked
        (``cos, sin = position_embeddings[self.rope_layer_type]``, where
        ``position_embeddings`` is a ``{"main": (cos, sin), "compress": (cos, sin)}``
        dict from the model) still boils down to the same boundary crossing: the
        key is a host-side string/attribute read, never itself a tensor operand,
        so the unpack targets alias the OUTER parameter the same way they would if
        it had no dict layer at all. Recognize a single-level subscript of a
        tracked param the same as a bare name.
        """
        origin_name: str | None = None
        if isinstance(value, ast.Name) and value.id in self.param_names:
            origin_name = value.id
        elif (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and value.value.id in self.param_names
        ):
            origin_name = value.value.id
        if origin_name is None:
            return
        # The RHS may itself be an alias (``pe = position_embeddings; cos, sin = pe``);
        # resolve to the original forward parameter so every unpacked local points
        # back at the caller-visible name.
        origin = self.param_alias_origin.get(origin_name, origin_name)
        for target in targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                # ``cos, sin = position_embeddings`` -> cos is slot 0, sin slot 1.
                for ordinal, element in enumerate(target.elts):
                    if isinstance(element, ast.Name):
                        self.param_names.add(element.id)
                        self.param_alias_origin[element.id] = origin
                        self.param_alias_ordinal[element.id] = ordinal
            elif isinstance(target, ast.Name):
                # A plain rename (``pe = position_embeddings``) carries the RHS's
                # own ordinal forward, if it had one. A subscripted RHS
                # (``pe = position_embeddings[self.rope_layer_type]``) has no
                # ordinal of its own to inherit.
                self.param_names.add(target.id)
                self.param_alias_origin[target.id] = origin
                inherited = (
                    self.param_alias_ordinal.get(origin_name)
                    if isinstance(value, ast.Name)
                    else None
                )
                if inherited is not None:
                    self.param_alias_ordinal[target.id] = inherited

    def _track_shape_assignment(
        self, targets: list[ast.expr], value: ast.AST
    ) -> None:
        """Record shape-derived locals so reshape args resolve to real axes.

        Two patterns feed reshape targets: unpacking a tensor's shape
        (``a, b = x.shape[:2]``) and building a dim tuple from those unpacked
        names (``hidden_shape = (a, b, -1, self.head_dim)``). Neither is a tensor
        producer, so both are invisible to the data-flow tracking; capturing them
        here lets ``_format_shape_args`` expand ``view(hidden_shape)`` into
        ``x.shape[0], x.shape[1], -1, self.head_dim`` for the shape inferencer.
        """
        if len(targets) == 1 and isinstance(targets[0], (ast.Tuple, ast.List)):
            base = _shape_read_base(value)
            if base is not None:
                for index, elt in enumerate(targets[0].elts):
                    if isinstance(elt, ast.Name):
                        self.shape_unpack_tokens[elt.id] = f"{base}.shape[{index}]"
        if len(targets) == 1 and isinstance(targets[0], ast.Name):
            token = _single_shape_index_token(value)
            if token is not None:
                self.shape_unpack_tokens[targets[0].id] = token
        if isinstance(value, ast.Tuple):
            for target in targets:
                if isinstance(target, ast.Name):
                    self.shape_tuple_vars[target.id] = value

    def _format_shape_args(self, args: list[ast.expr]) -> str:
        """Render ``view``/``reshape``/``expand`` args, expanding shape locals."""
        parts: list[str] = []
        for arg in args:
            if isinstance(arg, ast.Name) and arg.id in self.shape_tuple_vars:
                parts.extend(self._expand_shape_tuple(self.shape_tuple_vars[arg.id]))
            else:
                parts.append(self._render_shape_dim(arg))
        return ", ".join(parts)

    def _expand_shape_tuple(self, tup: ast.Tuple) -> list[str]:
        return [self._render_shape_dim(elt) for elt in tup.elts]

    def _render_shape_dim(self, elt: ast.expr) -> str:
        """One reshape dim as a resolver-friendly token.

        Unpacked shape locals (``batch_size`` → ``x.shape[0]``) keep their source
        axis; ``self.head_dim``-style config attributes resolve to their concrete
        int (the global ``head_dim`` is a zero placeholder here); everything else
        is left as source text for the shape inferencer to interpret.
        """
        if isinstance(elt, ast.Name) and elt.id in self.shape_unpack_tokens:
            return self.shape_unpack_tokens[elt.id]
        resolved = _config_value(elt, {}, self.self_values)
        if isinstance(resolved, int) and not isinstance(resolved, bool):
            return str(resolved)
        return ast.unparse(elt)

    def _loop_iteration_count(self, node: ast.For) -> int | None:
        """Static trip count for the ``Loop_N_iterations`` label (see module helper)."""
        return _loop_iteration_count_of(node, self.self_values, self._name_value_ast)

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

    def _apply_branch_alternatives(self) -> None:
        """Fold ``if``/``else`` branch producers into their common consumers.

        After an if/else merge only the survivor's producer lives in
        ``var_producer``. A later statement consuming the merged variable then
        depends only on that survivor, orphaning the other branch's producer.
        Expand every consumer's predecessors to include the recorded
        alternatives so both runtime paths stay live (no dead-end promoted to a
        spurious output) and the graph stays acyclic.
        """
        if not self.branch_alternatives:
            return

        def expand(preds: tuple[str, ...]) -> tuple[str, ...]:
            result: list[str] = []
            for pred in preds:
                result.append(pred)
                for alt in sorted(self.branch_alternatives.get(pred, ())):
                    if alt != pred:
                        result.append(alt)
            return self._dedupe(result)

        for index, operation in enumerate(self.operations):
            new_preds = expand(operation.predecessors)
            if new_preds != operation.predecessors:
                self.operations[index] = ForwardOperation(
                    **{**operation.__dict__, "predecessors": new_preds}
                )
        for step, preds in list(self.step_predecessors.items()):
            new_preds = expand(preds)
            if new_preds != preds:
                self.step_predecessors[step] = new_preds

    def _reconstruct_attention_step(self, body: list[ast.stmt]) -> None:
        """Wire the attention kernel's q/k/v when the taken branch hid the call.

        The extractor records ``step_predecessors[@attention]`` when it visits the
        ``attention_interface(self, query, key, value, ...)`` call. But that call
        can live in a branch or comprehension the extractor does not descend into
        — a dispatched attention often runs the eager path as
        ``[interface(self, q, k, v, ...) for q, k, v in zip(*splits)]`` while the
        flash branch spells the operands out positionally. Either way the kernel
        collapses to one ``@attention`` node whose real inputs are the same three
        tensors, and ``var_producer`` already holds each one's final producer
        (through the trailing ``transpose``/``unsqueeze`` reshapes, and including a
        ``value`` tensor that flows only through such ops).

        When ``@attention`` is a forward call but carries no predecessors, recover
        them from the first attention-interface call in source order: resolve each
        positional tensor operand to its producer and record it, named after the
        operand so the kernel port is labelled ``query_states``/``key_states``/
        ``value_states``. General: no branch or comprehension shape is assumed —
        the operands are read straight off whichever call the source spells out.
        """
        existing = self.step_predecessors.get(SYNTHETIC_ATTENTION)
        if existing:
            return
        call = next(
            (
                node
                for stmt in body
                for node in ast.walk(stmt)
                if isinstance(node, ast.Call)
                and _expr_name(node.func) in _SYNTHETIC_ATTENTION_NAMES
            ),
            None,
        )
        if call is None:
            return
        start = (
            1
            if call.args
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "self"
            else 0
        )
        predecessors: list[str] = []
        arg_names: dict[str, str] = {}
        for arg in call.args[start:]:
            if not isinstance(arg, ast.Name):
                continue
            producer = self.var_producer.get(arg.id)
            if producer is None:
                continue
            predecessors.append(producer)
            arg_names.setdefault(arg.id, producer)
        if not predecessors:
            return
        self.step_predecessors[SYNTHETIC_ATTENTION] = self._dedupe(predecessors)
        if arg_names:
            self.step_predecessor_args[SYNTHETIC_ATTENTION] = arg_names

    def _drop_phantom_attention_steps(self) -> None:
        """Remove attention kernel steps that carry no predecessors.

        A real attention call always reads query/key/value tensors, so a
        ``@attention`` step with an empty predecessor list is never a genuine
        call at this scope. It appears when a forward loops over a submodule
        (``for blk in self.blocks: blk(...)``) whose *own* forward runs
        attention: analysing the loop body hoists that inner kernel up to the
        enclosing forward, where it has nothing to read. Left in place it becomes
        a phantom top-level attention node that the exporter later mirrors into
        the real nested attention diagram, wiring the kernel's own outputs back
        to its inputs and forming a cycle. The genuine attention still lives
        (with its q/k/v predecessors) inside the submodule's own analysis.
        """
        phantom = {
            step
            for step, preds in self.step_predecessors.items()
            if step == SYNTHETIC_ATTENTION and not preds
        }
        if not phantom:
            return
        for step in phantom:
            self.step_predecessors.pop(step, None)
            self.step_predecessor_args.pop(step, None)
        for step, preds in list(self.step_predecessors.items()):
            if any(pred in phantom for pred in preds):
                self.step_predecessors[step] = tuple(
                    pred for pred in preds if pred not in phantom
                )
        self.operations = [
            operation
            for operation in self.operations
            if operation.attr_name not in phantom
        ]
        for index, operation in enumerate(self.operations):
            if any(pred in phantom for pred in operation.predecessors):
                self.operations[index] = ForwardOperation(
                    **{
                        **operation.__dict__,
                        "predecessors": tuple(
                            pred
                            for pred in operation.predecessors
                            if pred not in phantom
                        ),
                    }
                )

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

    def _resolve_flash_request_predicate(self, test: ast.expr) -> bool | None:
        """Resolve ``if is_flash_attention_requested(config):`` from the checkpoint.

        A vision/attention forward commonly branches between a fused flash kernel
        and a per-chunk fallback on this transformers helper. Walking both branches
        leaves the graph with duplicated attention plumbing (a phantom ``cat`` from
        the branch that does not run). Resolve the predicate here from
        ``config._attn_implementation`` — defaulting to ``sdpa`` (the transformers
        default) when unset — so only the selected branch survives. Returns the
        boolean the predicate evaluates to, or ``None`` when *test* is not one of
        these predicates (leaving the general ``_config_value`` path in charge).

        General: keys off the shared transformers predicate name, not any model.
        """
        return _resolve_flash_predicate(test, self.config)

    def statements(
        self, statements: list[ast.stmt], *, condition: str | None = None
    ) -> None:
        for stmt in statements:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
                value = _expand_map_lambda_tuple(stmt.value)
                targets = (
                    stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                )
                # Parallel tuple assignment (``q, k = q.float(), k.float()``) binds
                # each target to its OWN right-hand element's producer. The generic
                # path takes a single producer for the whole RHS (the last element),
                # which collapses every target onto it — so ``q`` would wrongly point
                # at ``k.float()``. Element-wise binding keeps each name's true
                # source. General: any equal-arity ``a, b = x, y`` assignment.
                if (
                    isinstance(stmt, ast.Assign)
                    and len(targets) == 1
                    and isinstance(targets[0], (ast.Tuple, ast.List))
                    and isinstance(value, (ast.Tuple, ast.List))
                    and len(targets[0].elts) == len(value.elts)
                ):
                    element_producers = [
                        self.expression(element)[0] for element in value.elts
                    ]
                    self._track_shape_assignment(targets, value)
                    self._propagate_param_alias(targets, value)
                    for element_target, element_producer in zip(
                        targets[0].elts, element_producers
                    ):
                        if (
                            isinstance(element_target, ast.Name)
                            and element_producer is not None
                        ):
                            self.var_producer[element_target.id] = element_producer
                            # Parallel reassignment also drops any stale ordinal.
                            self.var_output_ordinal.pop(element_target.id, None)
                    continue
                producer, _ = self.expression(value)
                if producer is None and self._is_host_scalar_expr(value):
                    for target in targets:
                        if isinstance(target, ast.Name):
                            self.host_scalar_vars.add(target.id)
                for target in targets:
                    self._record_host_scalar(target, value)
                self._track_shape_assignment(targets, value)
                for target in targets:
                    if isinstance(target, ast.Name):
                        self._name_value_ast[target.id] = value
                self._propagate_param_alias(targets, value)
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
                self._record_output_unpack(stmt, producer)
                continue
            if isinstance(stmt, ast.AugAssign):
                # ``output_width += self.index_kpool - 1`` over host scalars is
                # index bookkeeping: fold the accumulation and emit no tensor op
                # (an empty-predecessor Add would dangle). Only a genuine tensor
                # accumuland (target bound to a real producer) emits an op.
                if (
                    isinstance(stmt.target, ast.Name)
                    and stmt.target.id not in self.var_producer
                    and self._is_host_scalar_expr(stmt.value)
                ):
                    combined = ast.BinOp(
                        left=stmt.target, op=stmt.op, right=stmt.value
                    )
                    self._record_host_scalar(stmt.target, combined)
                    self.host_scalar_vars.add(stmt.target.id)
                    continue
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
                value = stmt.value
                # An in-place write into a slice (``key_states[..., :n].copy_(src)``)
                # mutates the base tensor: its new value depends on ``src``. The base
                # is a Subscript, not a Name, so the plain-owner rebind below misses
                # it and ``src`` (e.g. an ``expand_kv`` Split) is dropped. Emit the
                # mutation as a real op reading [prior base, operands] and rebind the
                # root tensor name so downstream reads (and the return) carry it.
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and _is_inplace_method(value.func.attr)
                    and isinstance(value.func.value, ast.Subscript)
                ):
                    root = _subscript_root_name(value.func.value)
                    if root is not None:
                        self._suppress_slice_resize = True
                        try:
                            base_producer, base_external = self.expression(
                                value.func.value
                            )
                        finally:
                            self._suppress_slice_resize = False
                        operand_producers: list[str] = []
                        operand_external: list[str] = []
                        for operand in (*value.args, *(kw.value for kw in value.keywords)):
                            operand_producer, ext = self.expression(operand)
                            if operand_producer:
                                operand_producers.append(operand_producer)
                            operand_external.extend(ext)
                        producer = self._emit(
                            value,
                            _inplace_label(value.func.attr),
                            [
                                producer
                                for producer in (
                                    self.var_producer.get(root),
                                    base_producer,
                                    *operand_producers,
                                )
                                if producer
                            ],
                            [*base_external, *operand_external],
                        )
                        self.var_producer[root] = producer
                        continue
                producer, _ = self.expression(value)
                if (
                    producer
                    and isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                ):
                    owner = value.func.value
                    if isinstance(owner, ast.Name):
                        self.var_producer[owner.id] = producer
                continue
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                elements = (
                    stmt.value.elts
                    if isinstance(stmt.value, ast.Tuple)
                    else [stmt.value]
                )
                for element in elements:
                    producer, _ = self.expression(element)
                    label = self._return_element_label(element)
                    if label and producer and label not in self.return_producer_slots:
                        self.return_producer_slots[label] = producer
                        self.return_producer_order.append(label)
                continue
            if isinstance(stmt, ast.If):
                outcome = self._resolve_flash_request_predicate(stmt.test)
                if outcome is None:
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
                    before_host = dict(self.host_scalar_values)
                    before = len(self.operations)
                    self.statements(stmt.body, condition=test)
                    body_env = dict(self.var_producer)
                    body_host = dict(self.host_scalar_values)
                    for index in range(before, len(self.operations)):
                        op = self.operations[index]
                        self.operations[index] = ForwardOperation(
                            **{
                                **op.__dict__,
                                "details": (*op.details, f"condition: {test}"),
                            }
                        )
                    self.var_producer = dict(before_env)
                    self.host_scalar_values = dict(before_host)
                    before_else = len(self.operations)
                    self.statements(stmt.orelse, condition=f"not ({test})")
                    else_env = dict(self.var_producer)
                    else_host = dict(self.host_scalar_values)
                    for index in range(before_else, len(self.operations)):
                        op = self.operations[index]
                        self.operations[index] = ForwardOperation(
                            **{
                                **op.__dict__,
                                "details": (*op.details, f"condition: not ({test})"),
                            }
                        )
                    survivor_env = else_env if stmt.orelse else body_env
                    other_env = body_env if stmt.orelse else else_env
                    # A variable assigned in both mutually-exclusive branches to
                    # different producers is the output of exactly one branch per
                    # invocation. Join them with an explicit Select (phi) node so
                    # the merged variable's consumers read a single tensor while
                    # both branch computations stay reachable (they feed the
                    # Select). When the survivor branch merely passes a boundary
                    # parameter through (no op producer) but the other branch
                    # computes a real op (``topk_indices = self.indexer(...)`` vs
                    # ``= prev_topk_indices``), adopt the real producer so the
                    # merged variable wires to the visible computation.
                    for variable in set(survivor_env) | set(other_env):
                        survivor_producer = survivor_env.get(variable)
                        other_producer = other_env.get(variable)
                        if (
                            survivor_producer
                            and other_producer
                            and survivor_producer != other_producer
                        ):
                            survivor_env[variable] = self._emit_branch_select(
                                stmt,
                                survivor_producer,
                                other_producer,
                                test,
                            )
                        elif not survivor_producer and other_producer:
                            survivor_env[variable] = other_producer
                    self.var_producer = survivor_env
                    # Host scalars that survive with a single unambiguous value are
                    # kept; a name that folds to different constants on each branch
                    # is ambiguous, so drop it (a later slice bound then declines to
                    # fold rather than resize to a branch-specific width).
                    survivor_host = else_host if stmt.orelse else body_host
                    other_host = body_host if stmt.orelse else else_host
                    self.host_scalar_values = {
                        name: value
                        for name, value in survivor_host.items()
                        if other_host.get(name, value) == value
                    }
                continue
            if isinstance(stmt, ast.For):
                iteration_count = self._loop_iteration_count(stmt)
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
                        member_ids = operation_ids
                        if updated not in member_ids:
                            # A ``for blk in self.blocks: h = blk(h)`` loop carries
                            # its value through a ModuleList child, not an inline op,
                            # so the child never lands in ``operation_ids``. Register
                            # it as a loop member so the carried-in boundary gets a
                            # real consumer (mirroring inline-op loops).
                            member_ids = (*operation_ids, updated)
                        self.loop_carried.append(
                            LoopCarriedSpec(
                                loop_id=loop_id,
                                iteration_count=iteration_count,
                                variable=variable,
                                initial_producer=initial,
                                updated_producer=updated,
                                operation_ids=member_ids,
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
        or is_function_synthetic(call)
        or call == SYNTHETIC_ATTENTION
        or not call.startswith("@")
        or (is_functional_synthetic(call) and not drop_functional)
    ]


def _forward_node_eval_order(func: ast.FunctionDef) -> dict[tuple[int, int], int]:
    """Rank every source position by the order the forward actually evaluates it.

    Python evaluates a call's arguments before the call itself, so a nested
    ``self.norm1(x)`` inside ``self.attn(self.norm1(x), ...)`` runs first. A raw
    ``(line, col)`` sort assumes the nested call merely sits further right on the
    same line, which breaks when the enclosing call spans multiple lines and the
    argument lands on a *later* line than the call it feeds. A post-order walk
    (children before parent) captures the true evaluation order regardless of
    line breaks; visiting the parent last lets it win a shared position so an
    operation node outranks the operand Name it reuses the column of.
    """
    order: dict[tuple[int, int], int] = {}
    counter = 0

    def _walk(node: ast.AST) -> None:
        nonlocal counter
        for child in ast.iter_child_nodes(node):
            _walk(child)
        line = getattr(node, "lineno", None)
        if line is not None:
            order[(line, getattr(node, "col_offset", 0))] = counter
            counter += 1

    for stmt in func.body:
        _walk(stmt)
    return order


def _forward_calls_in_source_order(
    func: ast.FunctionDef,
    module_calls: list[str],
    operations: list[ForwardOperation],
) -> list[str]:
    """Merge submodule calls and parsed tensor ops into the order the forward runs them.

    Ordering follows true evaluation order (arguments before their enclosing
    call), so nested calls run first even when a multi-line call pushes them onto
    a later source line.
    """
    eval_order = _forward_node_eval_order(func)
    unplaceable = min(eval_order.values(), default=0) - 1

    # A synthetic call (rope helper, functional op, kernel merge) records only the
    # line it fired on with a placeholder column, so its exact ``(line, col)`` is
    # rarely a key in ``eval_order``. Falling back to the *last* evaluation rank on
    # that source line places the synthetic after its own arguments — the call
    # completes once its operands are ready — which keeps ``apply_rotary`` behind
    # the reshape/permute/q_norm it consumes instead of floating to the front.
    line_last_rank: dict[int, int] = {}
    for (line, _col), rank in eval_order.items():
        if rank > line_last_rank.get(line, -1):
            line_last_rank[line] = rank

    def _rank_for(where: tuple[int, int] | None) -> int | None:
        if where is None:
            return None
        rank = eval_order.get(where)
        if rank is not None:
            return rank
        return line_last_rank.get(where[0])

    ordered: list[tuple[float, str]] = []
    call_positions = _self_call_source_positions(func)
    functional_positions = _functional_synthetic_source_positions(func)
    kernel_position = _kernel_merge_source_position(func)
    fallback = 0
    for call in module_calls:
        where = (
            call_positions.get(call)
            or call_positions.get(base_submodule_attr(call))
            or functional_positions.get(call)
            or positional_synthetic_source_pos(call)
            or function_synthetic_source_pos(call)
            or submodule_callsite_source_pos(call)
        )
        if where is None and call == SYNTHETIC_ATTENTION:
            where = kernel_position
        rank = _rank_for(where)
        if rank is None:
            # A call the walk cannot place keeps its parsed order ahead of the ops.
            rank = unplaceable - fallback
            fallback += 1
        ordered.append((rank, call))
    tail = max(eval_order.values(), default=0) + 1
    for op in operations:
        match = _OPERATION_SOURCE_POS_RE.match(op.attr_name)
        where = (int(match.group(1)), int(match.group(2))) if match else None
        rank = _rank_for(where)
        ordered.append((tail if rank is None else rank, op.attr_name))
    ordered.sort(key=lambda item: item[0])
    source_order = [name for _rank, name in ordered]
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
    # ``return BaseModelOutputWithPooling(last_hidden_state=x, pooler_output=y)``
    # — a HuggingFace ``ModelOutput`` dataclass wrapper. Its keyword arguments
    # name the real tensor producers (dataclass fields are always built by
    # keyword in this codebase); without unwrapping it the forward looks like it
    # returns nothing, so ``forward_return_slots`` stays empty and the graph
    # falls back to exporting every dangling node as a spurious output.
    # POSITIONAL args are deliberately NOT read as field names here: a plain
    # method/function call used directly as the return value (``return
    # output.type_as(x)``, ``return rotate_half(x)``) passes its real operands
    # positionally, and treating the operand's own name as the return slot's
    # name would misattribute the module's output to that operand's producer
    # instead of the call's own result.
    if isinstance(value, ast.Call):
        names: list[str] = []
        for keyword in value.keywords:
            if isinstance(keyword.value, ast.Name):
                names.append(keyword.value.id)
        return names
    return []


def _extract_forward_return_metadata(
    func: ast.FunctionDef,
    var_producer: dict[str, str],
    step_predecessors: dict[str, tuple[str, ...]] | None = None,
    multi_output_slots: set[str] | None = None,
) -> tuple[dict[str, str], list[str], str | None]:
    """Map ``return (a, b, c)`` names to the inline ops that produce them."""
    return_order: list[str] = []
    returns_call_wrapper = False
    for stmt in reversed(func.body):
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            return_order = _return_value_names(stmt.value)
            # A ``ModelOutput``/dataclass wrapper (``return BaseModelOutputWithPooling(
            # last_hidden_state=x, pooler_output=y)``) is accessed by field, so the
            # caller typically reads one terminal. A bare ``return (a, b)`` tuple is
            # unpacked positionally, so every element is a real output and must not
            # be collapsed onto its most-downstream member.
            returns_call_wrapper = isinstance(stmt.value, ast.Call)
            break
    slots = {name: var_producer[name] for name in return_order if name in var_producer}
    input_name = _primary_forward_input_name(func)
    # When one returned tensor is data-derived from another — ``last_hidden_state``
    # feeds ``pooler_output = self.merger(last_hidden_state)`` in a vision tower's
    # ``BaseModelOutputWithPooling`` — the *downstream* slot is the module's real
    # result; the upstream one is an intermediate a caller may also expose. Prefer
    # the unique most-downstream returned tensor (the one no other returned tensor
    # descends from) so the parent wires the true output (the merger), not the
    # intermediate. Ambiguous fan-out (``hidden_states, past_key_values`` — a cache
    # side-channel not on the main chain) leaves several terminals; fall back then.
    terminal = (
        _sole_terminal_return_slot(
            slots, step_predecessors or {}, multi_output_slots or set()
        )
        if returns_call_wrapper
        else None
    )
    if terminal is not None:
        # The intermediate slots are subsumed by the terminal — they are ancestors
        # on its data chain, so they stay live via its producer and must not become
        # separate (dead, no-consumer) output ports. Expose the terminal alone.
        return {terminal: slots[terminal]}, [terminal], terminal
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


def _sole_terminal_return_slot(
    slots: dict[str, str],
    step_predecessors: dict[str, tuple[str, ...]],
    multi_output_slots: set[str] | None = None,
) -> str | None:
    """The one returned slot every other returned slot is a data-ancestor of.

    Returns ``None`` unless exactly one slot is downstream of all the others, so a
    genuine multi-output return (parallel tensors, or a cache side-channel) keeps
    the source-order heuristics instead of arbitrarily promoting one branch.
    """
    if len(slots) < 2:
        return None
    # A returned slot that names a slice of a multi-output op
    # (``k_nope, value_states = torch.split(...)`` -> ``return key_states,
    # value_states``) is a genuine parallel tensor, not an intermediate on
    # another slot's chain — even though it shares its producer op with a
    # sibling slice that a downstream slot *does* consume. Ancestry is tracked
    # per producer attr, which cannot tell the two slices apart, so never
    # collapse when any slot is such a slice; keep every slot as its own return.
    if multi_output_slots and any(name in multi_output_slots for name in slots):
        return None
    producers = {name: producer for name, producer in slots.items()}

    def _ancestors(producer: str) -> set[str]:
        seen: set[str] = set()
        stack = list(step_predecessors.get(producer, ()))
        while stack:
            step = stack.pop()
            if step in seen:
                continue
            seen.add(step)
            stack.extend(step_predecessors.get(step, ()))
        return seen

    other_producers = set(producers.values())
    terminals = [
        name
        for name, producer in producers.items()
        # A terminal is not an ancestor of any *other* returned producer.
        if not any(
            producer in _ancestors(other)
            for other in other_producers
            if other != producer
        )
    ]
    if len(terminals) != 1:
        return None
    # The lone terminal must actually sit downstream of the others, not merely be
    # disconnected from them — require every other producer among its ancestors.
    terminal = terminals[0]
    ancestors = _ancestors(producers[terminal])
    if all(
        producer == producers[terminal] or producer in ancestors
        for producer in other_producers
    ):
        return terminal
    return None


def _live_forward_steps(
    *,
    operations: dict[str, ForwardOperation],
    return_slots: dict[str, str],
    step_predecessors: dict[str, tuple[str, ...]] | None = None,
) -> set[str]:
    """Backward closure of ops and submodule steps that feed returned values."""
    if not return_slots:
        return set(operations.keys())
    step_predecessors = step_predecessors or {}

    def _seed(step: str, _seen: set[str] | None = None) -> list[str]:
        """Resolve a producer step to operation steps.

        A returned value may be produced by a non-op step — a sibling helper
        method such as a rotary embedding's ``recomposition_frequencies``. Such a
        step is not itself tensor math, but the ops feeding it (the inline
        multiplies) are live and must not be pruned, so bridge through the
        recorded ``step_predecessors``.
        """
        if step in operations:
            return [step]
        _seen = _seen or set()
        if step in _seen:
            return []
        _seen.add(step)
        seeds: list[str] = []
        for pred in step_predecessors.get(step, ()):
            seeds.extend(_seed(pred, _seen))
        return seeds

    live_ops: set[str] = set()
    pending = [
        seed for producer in return_slots.values() for seed in _seed(producer)
    ]
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
    step_predecessors: dict[str, tuple[str, ...]] | None = None,
) -> tuple[list[str], dict[str, ForwardOperation]]:
    if len(return_slots) < 2:
        return forward_calls, operations
    live = _live_forward_steps(
        operations=operations,
        return_slots=return_slots,
        step_predecessors=step_predecessors,
    )
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
            output_names=operation.output_names,
            predecessor_ports=operation.predecessor_ports,
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
        step_predecessors=analysis.step_predecessors,
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
    config: dict[str, Any] | None = None,
    module_functions: dict[str, ast.FunctionDef] | None = None,
    class_methods: dict[str, ast.FunctionDef] | None = None,
    is_free_function_body: bool = False,
) -> ForwardAnalysis:
    # The primary parameter is the main path, so only the extra ones can identify
    # which step consumes a side feed.
    primary = _primary_forward_input_name(func)
    extractor = _ForwardOperationExtractor(
        self_values=self_values,
        all_tensor_ops=all_tensor_ops,
        param_names=_forward_input_names(func) - {primary} if primary else set(),
        config=config,
        module_functions=module_functions,
        repeated_submodule_attrs=_repeated_self_call_attrs(func.body),
        class_methods=class_methods,
        is_free_function_body=is_free_function_body,
    )
    # An operation reading the primary parameter partway through the forward reads the
    # value arriving at the chain, not the previous step. Naming it lets those reads
    # resolve to the chain input instead of silently inheriting the wrong producer.
    if primary:
        extractor.var_producer[primary] = FORWARD_METHOD_INPUT
    # A secondary forward input consumed by a traced free-function node is that
    # node's real source; seed it as the method boundary so the edge starts from
    # the input instead of dangling. Gated to those args so ordinary side-inputs
    # (handed straight to a submodule) keep flowing through param attribution.
    # A param that is REASSIGNED in the body (``q, k = q.float(), k.float()``
    # inside ``apply_rotary_pos_emb_vision``) must NOT be seeded: the seed would
    # bind its first read to the shared ``@method_input`` (the *primary*'s
    # boundary), so ``k.float()`` would spuriously read the primary ``q`` before
    # ``k`` is rebound. Such a param already docks through its own boundary
    # (``param_names``) and flows normally after its assignment; only pure
    # pass-through params (never reassigned) need the seed.
    reassigned = _ForwardOperationExtractor._assigned_names(func.body)
    for name in (
        _traced_free_function_arg_names(func)
        & _forward_input_names(func)
    ) - reassigned:
        extractor.var_producer.setdefault(name, FORWARD_METHOD_INPUT)
    extractor.statements(func.body)
    extractor._apply_branch_alternatives()
    extractor._reconstruct_attention_step(func.body)
    extractor._drop_phantom_attention_steps()
    return_slots, return_order, primary_return_slot = _extract_forward_return_metadata(
        func,
        extractor.var_producer,
        extractor.step_predecessors,
        set(extractor.var_output_ordinal),
    )
    # Fill slots the name-based extraction missed — subscripted return elements
    # (``return pool_keys[:, keep], ...``) whose producer was captured while the
    # return statement was walked. Preserves source order so each consumer docks
    # onto its own slice op instead of the frame's last op (which would orphan the
    # others).
    for label in extractor.return_producer_order:
        producer = extractor.return_producer_slots.get(label)
        if producer and label not in return_slots:
            return_slots[label] = producer
            if label not in return_order:
                return_order.append(label)
    if primary_return_slot is None and return_order:
        primary_return_slot = return_order[-1]
    return ForwardAnalysis(
        operations=extractor.operations,
        var_producer=dict(extractor.var_producer),
        step_predecessors=dict(extractor.step_predecessors),
        step_predecessor_args=dict(extractor.step_predecessor_args),
        step_predecessor_ordinals=dict(extractor.step_predecessor_ordinals),
        step_output_names=dict(extractor.step_output_names),
        step_boundary_params=dict(extractor.step_boundary_params),
        step_boundary_arg_params=dict(extractor.step_boundary_arg_params),
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
    class_methods = {
        item.name: item
        for item in cls.node.body
        if isinstance(item, ast.FunctionDef)
    }
    analysis = _forward_operations_from_forward(
        forward,
        self_values=_self_config_values(init_func, {}),
        all_tensor_ops=True,
        class_methods=class_methods,
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
    cls.forward_step_predecessor_ordinals = dict(analysis.step_predecessor_ordinals)
    cls.forward_step_output_names = dict(analysis.step_output_names)
    cls.forward_step_boundary_params = dict(analysis.step_boundary_params)
    cls.forward_step_boundary_arg_params = dict(analysis.step_boundary_arg_params)
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
        activation_param_bindings: dict[str, dict[str, str]] | None = None,
        vision_scoped_classes: set[str] | None = None,
        vision_config: dict[str, Any] | None = None,
        module_functions: dict[str, ast.FunctionDef] | None = None,
    ) -> None:
        self.classes: dict[str, ClassStructure] = {}
        self.config = dict(config or {})
        self.all_tensor_ops = all_tensor_ops
        self.activation_param_bindings = activation_param_bindings or {}
        self.vision_scoped_classes = set(vision_scoped_classes or ())
        self.vision_config = dict(vision_config or {})
        # A multimodal composite config nests the decoder's own settings under
        # ``text_config`` (``index_topk``/``index_kpool`` live there, not at the
        # top level). Overlay it for the text path so a class resolves
        # ``config.index_topk`` to its real value instead of ``_UNKNOWN``. Derived
        # here (not a constructor arg) so both entry points pick it up. Empty for
        # a flat single-modality config, making the overlay a no-op.
        self.text_config = (
            dict((config or {}).get("text_config") or {})
            if isinstance(config, dict)
            else {}
        )
        self.module_functions = dict(module_functions or {})

    def _config_for_class(self, class_name: str) -> dict[str, Any]:
        """Config a class resolves ``self.<attr> = config.<attr>`` against.

        Vision-tower classes overlay ``vision_config`` (vision wins, because
        ``hidden_size`` exists at both levels: 4096 text vs 1024 vision). Every
        other class — the text path — overlays ``text_config`` when the composite
        config nests it, so decoder-only settings (``index_topk``) resolve; a flat
        config leaves the top level unchanged.
        """
        if class_name in self.vision_scoped_classes and self.vision_config:
            return {**self.config, **apply_config_attribute_aliases(self.vision_config)}
        if self.text_config:
            return {**self.config, **apply_config_attribute_aliases(self.text_config)}
        return self.config

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
        forward_step_predecessor_ordinals: dict[str, dict[str, int]] = {}
        forward_step_output_names: dict[str, list[str]] = {}
        forward_step_boundary_params: dict[str, tuple[str, ...]] = {}
        forward_step_boundary_arg_params: dict[
            str, dict[str, tuple[str, int | None]]
        ] = {}
        forward_return_slots: dict[str, str] = {}
        forward_return_order: list[str] = []
        primary_return_slot: str | None = None
        forward_call_output_names: dict[str, str] = {}
        forward_loop_carried: list[LoopCarriedSpec] = []
        single_op_methods: dict[str, ForwardOperation] = {}
        multi_op_methods: dict[str, list[ForwardOperation]] = {}
        multi_op_method_returns: dict[
            str, tuple[dict[str, str], list[str], str | None]
        ] = {}
        multi_op_method_inputs: dict[str, str] = {}
        multi_op_method_step_predecessors: dict[
            str, dict[str, tuple[str, ...]]
        ] = {}
        multi_op_method_order: dict[str, list[str]] = {}
        multi_op_method_step_predecessor_args: dict[
            str, dict[str, dict[str, str]]
        ] = {}
        forward_step_return_producers: dict[str, list[str]] = {}
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

        unresolved_activation_refs: dict[str, tuple[str, str]] = {}
        if init_func is not None:
            (
                init_assignments,
                init_details,
                init_assignment_options,
                unresolved_activation_refs,
            ) = _parse_init(
                init_func,
                config=self.config,
                param_bindings=self.activation_param_bindings.get(node.name),
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
            ) = _parse_forward(
                resolved_forward_func,
                self_values=_self_config_values(
                    init_func, self._config_for_class(node.name)
                ),
            )
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
            # Plain instance methods defined in this same class (not ``__init__``
            # submodule attrs). A ``self.<method>(...)`` call site's args are the
            # CALLER's local names; resolving against the callee's own signature
            # (when it is one of these) keeps a caller/callee name mismatch from
            # stranding an argument (see ``_class_method_param_names``).
            class_methods = {
                item.name: item
                for item in node.body
                if isinstance(item, ast.FunctionDef)
            }
            if _is_moe_gate_class(node.name, forward_calls):
                values = _self_config_values(init_func, self._config_for_class(node.name))
                analysis = _forward_operations_from_forward(
                    resolved_forward_func,
                    self_values=values,
                    all_tensor_ops=self.all_tensor_ops,
                    config=self._config_for_class(node.name),
                    module_functions=self.module_functions,
                    class_methods=class_methods,
                )
                if analysis.operations:
                    forward_step_predecessors = dict(analysis.step_predecessors)
                    forward_step_predecessor_args = dict(analysis.step_predecessor_args)
                    forward_step_predecessor_ordinals = dict(
                        analysis.step_predecessor_ordinals
                    )
                    forward_step_output_names = dict(analysis.step_output_names)
                    forward_step_boundary_params = dict(
                        analysis.step_boundary_params
                    )
                    forward_step_boundary_arg_params = dict(
                        analysis.step_boundary_arg_params
                    )
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
                self_values=_self_config_values(init_func, self._config_for_class(node.name)),
                all_tensor_ops=self.all_tensor_ops,
            )
            (
                multi_op_methods,
                multi_op_method_returns,
                multi_op_method_inputs,
                multi_op_method_step_predecessors,
                multi_op_method_order,
                multi_op_method_step_predecessor_args,
            ) = _multi_op_forward_methods(
                node,
                forward_calls,
                init_assignments,
                self_values=_self_config_values(init_func, self._config_for_class(node.name)),
                all_tensor_ops=self.all_tensor_ops,
            )
            # Traced free-function calls (rope helpers, ...) expand from their
            # module-level definition. Keys are synthetic attrs (``@positional_``/
            # ``@function_``), disjoint from method names, so they share the same
            # ``multi_op_methods`` rendering path in the block tree.
            free_fn_methods, free_fn_return_producers = _multi_op_free_functions(
                self.module_functions,
                forward_calls,
                self_values=_self_config_values(
                    init_func, self._config_for_class(node.name)
                ),
                all_tensor_ops=self.all_tensor_ops,
            )
            multi_op_methods.update(free_fn_methods)
            forward_step_return_producers.update(free_fn_return_producers)
            # A tuple-returning *method* expanded inline (``pool_keys,
            # pool_indices, pool_valid = self.get_pooled_states(...)``) exposes
            # the same ordinal→producer mapping as a free function: publish its
            # ordered internal producers so a consumer reading a specific return
            # ordinal docks onto that slot's producer instead of every consumer
            # collapsing onto the frame's last op (which strands the other slots
            # as dead nodes). General: driven off the method's own return tuple,
            # no class-name checks.
            for method_base, (
                ret_slots,
                ret_order,
                _ret_primary,
            ) in multi_op_method_returns.items():
                producers = [ret_slots.get(slot) for slot in ret_order]
                if len(producers) >= 2 and all(p is not None for p in producers):
                    forward_step_return_producers.setdefault(
                        method_base, [p for p in producers if p is not None]
                    )
            delegates_inline = _forward_delegates_to_nothing(node.name, forward_calls)
            method_names = {
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef)
            }
            delegates_to_siblings = _forward_delegates_only_to_sibling_methods(
                forward_calls, init_assignments, method_names
            )
            if (
                _forward_owns_tensor_math(forward_calls, init_assignments)
                or delegates_inline
                or delegates_to_siblings
            ):
                values = _self_config_values(init_func, self._config_for_class(node.name))
                analysis = _forward_operations_from_forward(
                    resolved_forward_func,
                    self_values=values,
                    all_tensor_ops=self.all_tensor_ops,
                    config=self._config_for_class(node.name),
                    module_functions=self.module_functions,
                    class_methods=class_methods,
                )
                if analysis.operations:
                    forward_step_predecessors = dict(analysis.step_predecessors)
                    forward_step_predecessor_args = dict(analysis.step_predecessor_args)
                    forward_step_predecessor_ordinals = dict(
                        analysis.step_predecessor_ordinals
                    )
                    forward_step_output_names = dict(analysis.step_output_names)
                    forward_step_boundary_params = dict(
                        analysis.step_boundary_params
                    )
                    forward_step_boundary_arg_params = dict(
                        analysis.step_boundary_arg_params
                    )
                    forward_loop_carried = list(analysis.loop_carried)
                    module_calls = _module_calls_for_forward_merge(
                        forward_calls,
                        init_assignments,
                        parsed_operations=analysis.operations,
                    )
                    merged_calls = _forward_calls_in_source_order(
                        resolved_forward_func,
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
                values = _self_config_values(init_func, self._config_for_class(node.name))
                probed = _forward_operations_from_forward(
                    resolved_forward_func,
                    self_values=values,
                    all_tensor_ops=self.all_tensor_ops,
                    config=self._config_for_class(node.name),
                    module_functions=self.module_functions,
                    class_methods=class_methods,
                )
                if _forward_mixes_modules_and_inline_ops(
                    forward_calls,
                    init_assignments,
                    probed.operations,
                ):
                    forward_step_predecessors = dict(probed.step_predecessors)
                    forward_step_predecessor_args = dict(probed.step_predecessor_args)
                    forward_step_predecessor_ordinals = dict(
                        probed.step_predecessor_ordinals
                    )
                    forward_step_output_names = dict(probed.step_output_names)
                    forward_step_boundary_params = dict(
                        probed.step_boundary_params
                    )
                    forward_step_boundary_arg_params = dict(
                        probed.step_boundary_arg_params
                    )
                    forward_loop_carried = list(probed.loop_carried)
                    module_calls = _module_calls_for_forward_merge(
                        forward_calls,
                        init_assignments,
                        parsed_operations=probed.operations,
                    )
                    merged_calls = _forward_calls_in_source_order(
                        resolved_forward_func,
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

        # Structural (never class-name keyed): resolves an activation only when the
        # forward actually gates the normalized result through an activation
        # registry; a plain norm returns ``None``.
        gate_activation = _gated_norm_activation_from_forward(
            forward_func, init_func, self._config_for_class(node.name)
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
            gate_activation=gate_activation,
            forward_step_details=forward_step_details,
            side_inputs=side_inputs,
            forward_input_name=forward_input_name,
            forward_operations=forward_operations,
            forward_step_predecessors=forward_step_predecessors,
            forward_step_predecessor_args=forward_step_predecessor_args,
            forward_step_predecessor_ordinals=forward_step_predecessor_ordinals,
            forward_step_output_names=forward_step_output_names,
            forward_step_boundary_params=forward_step_boundary_params,
            forward_step_boundary_arg_params=forward_step_boundary_arg_params,
            single_op_methods=single_op_methods,
            multi_op_methods=multi_op_methods,
            multi_op_method_returns=multi_op_method_returns,
            multi_op_method_inputs=multi_op_method_inputs,
            multi_op_method_step_predecessors=multi_op_method_step_predecessors,
            multi_op_method_order=multi_op_method_order,
            multi_op_method_step_predecessor_args=multi_op_method_step_predecessor_args,
            forward_step_return_producers=forward_step_return_producers,
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
            unresolved_activation_refs=unresolved_activation_refs,
        )
        self.generic_visit(node)


def _parse_init(
    func: ast.FunctionDef,
    *,
    config: dict[str, Any] | None = None,
    param_bindings: dict[str, str] | None = None,
) -> tuple[
    dict[str, str],
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, tuple[str, str]],
]:
    assignments: dict[str, str] = {}
    details: dict[str, list[str]] = {}
    options: dict[str, list[str]] = {}
    unresolved_activations: dict[str, tuple[str, str]] = {}

    def record_assignment(attr: str, value: ast.AST) -> None:
        class_names = _assignment_class_names(
            value, config=config, param_bindings=param_bindings
        )
        if not class_names:
            return
        # A registry lookup is the fallback arm of a config switch whose other arm
        # constructs a real module (`SituAndMul` vs `ACT2FN[...]`), so it must not
        # displace that module regardless of which arm the walk reaches last.
        if attr in assignments and _activation_registry_class_name(
            value, config, param_bindings
        ):
            return
        attr_options = options.setdefault(attr, [])
        for class_name in class_names:
            if class_name not in attr_options:
                attr_options.append(class_name)
        assignments[attr] = class_names[0]
        details[attr] = _assignment_details(value, class_names[0])
        # Track a registry-selected key our curated display-name table does not
        # recognize, so it can be chased to its real class after the fact instead
        # of staying a permanently opaque, title-cased placeholder leaf. A later
        # assignment to the same attr that resolves cleanly clears the tracking.
        lookup = _activation_registry_lookup_for_assignment(
            value, config, param_bindings
        )
        if lookup is not None and lookup[1] not in _ACTIVATION_DISPLAY_NAMES:
            unresolved_activations[attr] = lookup
        else:
            unresolved_activations.pop(attr, None)

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

    return assignments, details, options, unresolved_activations


def _subscript_index_operands(index: ast.AST) -> list[ast.AST]:
    """Slice operands that could be *tensor* indices (advanced indexing).

    Returns the per-axis operands of a subscript, dropping the ones that can never
    be a tensor: ``Slice`` (``a:b``), ``Constant`` (ints, ``None``, ``...``), and
    ``Starred``. A ``Tuple`` slice (``x[i, j]``) is unpacked to its axes. The caller
    resolves each survivor through ``expression`` — only those bound to a traced
    tensor become gather inputs; scalar names (``layer_idx``) resolve to nothing.
    """
    if isinstance(index, ast.Tuple):
        operands = list(index.elts)
    else:
        operands = [index]
    return [
        operand
        for operand in operands
        if not isinstance(operand, (ast.Slice, ast.Constant, ast.Starred))
    ]


def _is_int_index(node: ast.AST) -> bool:
    """True for a literal integer axis-index (``x[:, 0]`` or ``x[:, -1]``), not
    ``None``/``...``/bool.

    A negative literal parses as ``UnaryOp(USub, Constant(n))``, not a bare
    ``Constant`` — unwrap that one level so ``x[..., -1]`` is recognised as an
    integer select exactly like its positive-index sibling ``x[..., 1]``.
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


def _subscript_select_dims(index: ast.AST) -> list[int]:
    """Axes an integer *select* drops from a multi-axis subscript (``x[:, c]``).

    A single index alongside a range slice (``freq[:, 0]``) selects one position
    and drops that axis — a genuine ``Slice`` op. A bare leading index (``x[0]``)
    with no accompanying ``:``/``...`` carries no dropped axis and stays a
    pass-through alias, so this returns ``[]`` for that. ``Ellipsis`` unambiguously
    stands for "every preceding axis", so a bare trailing integer index after one
    (``packed_states[..., -1]``) is just as much a genuine drop-that-axis select
    as ``x[..., 0]`` even with no explicit ``:`` slice alongside it — the trailing
    axes are numbered from the end so ``x[..., 0]``/``x[..., -1]`` both drop ``-1``.
    """
    elts = index.elts if isinstance(index, ast.Tuple) else [index]
    if len(elts) < 2:
        return []
    ellipsis_at = next(
        (
            pos
            for pos, elt in enumerate(elts)
            if isinstance(elt, ast.Constant) and elt.value is Ellipsis
        ),
        None,
    )
    if ellipsis_at is None:
        if not any(isinstance(elt, ast.Slice) for elt in elts):
            return []
        return [pos for pos, elt in enumerate(elts) if _is_int_index(elt)]
    tail = elts[ellipsis_at + 1 :]
    return [
        -(len(tail) - offset) for offset, elt in enumerate(tail) if _is_int_index(elt)
    ]


def _subscript_inserts_axis(index: ast.AST) -> bool:
    """True when a subscript inserts a size-1 axis via ``None`` (``x[..., None]``)."""
    elts = index.elts if isinstance(index, ast.Tuple) else [index]
    return any(
        isinstance(elt, ast.Constant) and elt.value is None for elt in elts
    )


def _none_insert_dim(index: ast.AST) -> int | None:
    """Axis at which a single ``None`` inserts a size-1 dim, or None if ambiguous.

    ``x[..., None]`` appends (dim ``-1``); ``x[None]`` prepends (dim ``0``);
    ``x[:, None]`` inserts at that position. Only single-``None`` subscripts
    resolve; anything more elaborate falls back to a pass-through alias.
    """
    elts = index.elts if isinstance(index, ast.Tuple) else [index]
    none_positions = [
        pos
        for pos, elt in enumerate(elts)
        if isinstance(elt, ast.Constant) and elt.value is None
    ]
    if len(none_positions) != 1:
        return None
    pos = none_positions[0]
    has_ellipsis_before = any(
        isinstance(elt, ast.Constant) and elt.value is Ellipsis
        for elt in elts[:pos]
    )
    if has_ellipsis_before:
        return -1
    return pos


def _is_inplace_method(name: str) -> bool:
    """True for tensor in-place mutators (``copy_``, ``add_``, ``masked_fill_``…).

    These end in a single trailing underscore; dunders and private helpers are
    excluded so only genuine mutating tensor methods qualify.
    """
    return (
        len(name) > 1
        and name.endswith("_")
        and not name.endswith("__")
        and not name.startswith("_")
    )


def _subscript_root_name(expr: ast.AST) -> str | None:
    """Root local tensor name of a (possibly nested) subscript, else ``None``.

    ``key_states[..., :n]`` → ``"key_states"``; ``self.cache[i]`` (rooted at an
    attribute, not a local) → ``None`` so only local tensors get rebound.
    """
    while isinstance(expr, ast.Subscript):
        expr = expr.value
    return expr.id if isinstance(expr, ast.Name) else None


def _shape_read_base(value: ast.AST) -> str | None:
    """Source-tensor expression of a ``<tensor>.shape`` / ``.shape[:k]`` read.

    ``hidden_states.shape[:2]`` and ``hidden_states.shape`` both return
    ``"hidden_states"``; anything that is not a shape read returns ``None``.
    """
    node = value
    if isinstance(node, ast.Subscript):
        node = node.value
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "shape"
        and not _is_self_attr(node, node.attr)
    ):
        return ast.unparse(node.value)
    return None


def _single_shape_index_token(value: ast.AST) -> str | None:
    """Positional-axis read bound to a scalar local, as a resolver token.

    ``seq_length = hidden_states.shape[0]`` reads one axis into a plain name (not a
    tuple unpack), so it escapes ``_shape_read_base``'s ``a, b = x.shape[:2]``
    path. Returning ``"hidden_states.shape[0]"`` lets ``_render_shape_dim`` expand a
    later ``reshape(seq_length, 3, self.num_heads, -1)`` into
    ``hidden_states.shape[0], 3, 16, -1`` -- which the shape inferencer resolves by
    copying the source's leading axis. Returns ``None`` unless *value* is exactly a
    ``<tensor>.shape[<int>]`` read.
    """
    if not isinstance(value, ast.Subscript):
        return None
    base = value.value
    if not (
        isinstance(base, ast.Attribute)
        and base.attr == "shape"
        and not _is_self_attr(base, base.attr)
    ):
        return None
    index = value.slice
    if (
        isinstance(index, ast.Constant)
        and isinstance(index.value, int)
        and not isinstance(index.value, bool)
    ):
        return f"{ast.unparse(base.value)}.shape[{index.value}]"
    return None


def _inplace_label(method: str) -> str:
    """Display label for an in-place mutator, reusing the out-of-place op's label."""
    base = method[:-1]
    return (
        _TENSOR_METHOD_LABELS.get(base)
        or _FUNCTION_LABELS.get(base)
        or base.replace("_", " ").title()
    )


def _activation_registry_lookup(
    node: ast.AST,
    config: dict[str, Any] | None,
    param_bindings: dict[str, str] | None = None,
) -> tuple[str, str] | None:
    """Return ``(registry_name, key)`` for an ``ACT2FN[key]``-style subscript.

    Resolves the config/param-bound key to its lowercased string value but does
    not judge whether that key is one of the curated display names -- callers
    that only want the display name use :func:`_activation_registry_class_name`;
    callers that need to chase an *unrecognized* key to its real class (see
    ``_expand_unresolved_activation_classes``) use this directly.
    """
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
    elif isinstance(key, ast.Name):
        # ``self.act_fn = ACT2FN[hidden_act]`` where ``hidden_act`` is a constructor
        # parameter; the activation is only knowable from how the class is
        # instantiated (e.g. ``Merger(hidden_act=config.hidden_act)``).
        name = (param_bindings or {}).get(key.id)
    if not isinstance(name, str) or not name.strip():
        return None
    return registry, name.strip().lower()


def _activation_registry_class_name(
    node: ast.AST,
    config: dict[str, Any] | None,
    param_bindings: dict[str, str] | None = None,
) -> str | None:
    """Resolve an activation-registry lookup to the activation the config selects."""
    resolved = _activation_registry_lookup(node, config, param_bindings)
    if resolved is None:
        return None
    _registry, lowered = resolved
    if lowered in _ACTIVATION_DISPLAY_NAMES:
        return _ACTIVATION_DISPLAY_NAMES[lowered]
    return lowered.replace("_", " ").title().replace(" ", "")


def _activation_registry_lookup_for_assignment(
    value: ast.AST,
    config: dict[str, Any] | None,
    param_bindings: dict[str, str] | None,
) -> tuple[str, str] | None:
    """Recover the activation-registry lookup behind an init assignment's value.

    Mirrors just the shapes ``_assignment_class_names`` walks to reach a plain
    ``ACT2FN[key]`` subscript (a direct assignment, or one arm of a config-switch
    ``IfExp``/list of candidates) -- enough to let an unrecognized key be chased
    to its real class after the fact, without re-implementing that whole walk.
    """
    if isinstance(value, ast.Subscript):
        return _activation_registry_lookup(value, config, param_bindings)
    if isinstance(value, ast.IfExp):
        return _activation_registry_lookup_for_assignment(
            value.body, config, param_bindings
        ) or _activation_registry_lookup_for_assignment(
            value.orelse, config, param_bindings
        )
    if isinstance(value, (ast.List, ast.Tuple)):
        for item in value.elts:
            found = _activation_registry_lookup_for_assignment(
                item, config, param_bindings
            )
            if found is not None:
                return found
    return None


def _collect_activation_param_bindings(
    tree: ast.AST, config: dict[str, Any] | None
) -> dict[str, dict[str, str]]:
    """Map ``{class_name: {ctor_param: activation_key}}`` from instantiation sites.

    A submodule may select its activation from a constructor parameter, e.g.
    ``self.act_fn = ACT2FN[hidden_act]``. The activation is only knowable from how
    the class is *instantiated* — ``Merger(hidden_act=config.hidden_act)`` in some
    parent's ``__init__`` — and that site is often a different (later) class. This
    pre-pass walks every ``__init__`` and records, for each submodule constructed
    there, the config-resolved value of each keyword argument that names an
    activation, keyed by the submodule's parameter name.
    """
    config = dict(config or {})
    # Vision submodules resolve their config args against ``vision_config``; fall
    # back to it so ``hidden_act=config.hidden_act`` resolves even when the value
    # only lives in the nested sub-config.
    sub_config = config.get("vision_config")
    fallbacks = [config]
    if isinstance(sub_config, dict):
        fallbacks.append(sub_config)
    bindings: dict[str, dict[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        init_func = _class_init_method(node)
        if init_func is None:
            continue
        for call in ast.walk(init_func):
            if not isinstance(call, ast.Call):
                continue
            class_name = _call_class_name(call)
            if not class_name:
                continue
            for keyword in call.keywords:
                if keyword.arg is None:
                    continue
                resolved: Any = _UNKNOWN
                for cfg in fallbacks:
                    resolved = _config_value(keyword.value, cfg, {})
                    if resolved is not _UNKNOWN:
                        break
                if isinstance(resolved, str) and resolved.strip():
                    bindings.setdefault(class_name, {})[keyword.arg] = resolved
    return bindings


def _assignment_class_names(
    node: ast.AST,
    *,
    config: dict[str, Any] | None = None,
    param_bindings: dict[str, str] | None = None,
) -> list[str]:
    """Return every constructible module class represented by an assignment."""
    if isinstance(node, ast.IfExp):
        names = _assignment_class_names(
            node.body, config=config, param_bindings=param_bindings
        ) + _assignment_class_names(
            node.orelse, config=config, param_bindings=param_bindings
        )
        return list(dict.fromkeys(names))
    if isinstance(node, ast.ListComp):
        return _assignment_class_names(
            node.elt, config=config, param_bindings=param_bindings
        )
    if isinstance(node, ast.Subscript):
        activation = _activation_registry_class_name(node, config, param_bindings)
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
                return _assignment_class_names(
                    node.args[0], config=config, param_bindings=param_bindings
                )
            return []
        class_name = _call_class_name(node)
        if class_name in _SKIP_INIT_CLASS_NAMES:
            return []
        return [class_name] if class_name else []
    if isinstance(node, (ast.List, ast.Tuple)):
        names: list[str] = []
        for item in node.elts:
            names.extend(
                _assignment_class_names(
                    item, config=config, param_bindings=param_bindings
                )
            )
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
            if isinstance(raw, str) and raw.strip():
                # A gate activation selected by a constructor kwarg
                # (``FusedRMSNormGated(..., activation='sigmoid')``). Tag it so a
                # consumer recovers the resolved name structurally, without
                # re-matching the detail text against an activation name set.
                details.append(
                    f"{GATE_ACTIVATION_DETAIL_PREFIX}{_display_activation_name(raw)}"
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


def _map_element_step_attr(node: ast.AST) -> str | None:
    """Synthetic step key for one expanded ``map(lambda x: BODY(x), ...)`` element.

    Mirrors the relevant subset of ``_ForwardOperationExtractor._call_step_producer``
    (positional/free-function synthetic naming, including the per-element
    discriminator ``_expand_map_lambda_tuple`` stamps on each clone) for this
    module-level provenance tracker, which runs independently of that class and
    otherwise never sees inside a lambda body.
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    target = _expr_name(func)
    discriminator = getattr(node, "_tracelens_map_discriminator", None)
    if target and _is_positional_function_call(func, target):
        return positional_synthetic_attr(target, node.lineno, discriminator)
    if _is_emittable_free_function(func, target):
        return function_synthetic_attr(target, node.lineno, discriminator)
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
        # ``q, k = map(lambda x: rearrange(x, ...), (q, k))`` applies the SAME
        # lambda body once per element; that body's call becomes its own node
        # (``_expand_map_lambda_tuple``/``_call_step_producer``), so each
        # element's true provenance chain ends at THAT step, not at its
        # pre-map source. ``stmt_calls`` cannot see it (built by
        # ``_extract_self_calls_ordered``, which never descends into a lambda
        # body), so resolve it directly from the expanded per-element clones.
        # Only set (and only aligned with ``source_names``) when *node.value*
        # is actually this map idiom -- ``_expand_map_lambda_tuple`` returns
        # any other value unchanged.
        expanded = _expand_map_lambda_tuple(node.value)
        expanded_elements = (
            expanded.elts if isinstance(expanded, ast.Tuple) and expanded is not node.value else None
        )
        if source_names is not None and len(source_names) == len(target.elts):
            zipped = True
            for index, (elt, source_name) in enumerate(zip(target.elts, source_names)):
                if not isinstance(elt, ast.Name):
                    zipped = False
                    break
                source_chain = list(var_chains.get(source_name, []))
                for call in stmt_calls:
                    if call not in source_chain:
                        source_chain.append(call)
                if expanded_elements is not None:
                    own_step = _map_element_step_attr(expanded_elements[index])
                    if own_step and own_step not in source_chain:
                        source_chain.append(own_step)
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
    if _KERNEL_MERGE_HELPER_RE.search(base):
        return False
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

    # Collect in CALL-ARGUMENT order: positional args first, then keywords.
    # This order is the one the kernel's inputs are physically wired in (the
    # merged-graph predecessor pass links producers in call order), so the
    # declared ``inputs:`` list must match it — otherwise the positional
    # port-matcher scrambles labels whenever a kernel mixes positional tensors
    # with keyword tensors (e.g. ``kernel(query, key, value, g=g, beta=beta)``
    # would declare ``g,beta,query,key,value`` and mislabel every port).
    args = call.args
    start = 1 if args and isinstance(args[0], ast.Name) and args[0].id == "self" else 0
    for index, arg in enumerate(args[start:], start=start):
        if isinstance(arg, ast.Name):
            consider(arg.id, arg)
        else:
            consider(f"in{index - start}", arg)

    for keyword in call.keywords:
        if keyword.arg in _KERNEL_PRODUCER_SKIP_KWARGS:
            continue
        if isinstance(keyword.value, ast.Attribute):
            continue
        if keyword.arg:
            consider(keyword.arg, keyword.value)

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


def _resolve_flash_predicate(
    test: ast.expr, config: dict[str, Any] | None
) -> bool | None:
    """Evaluate ``is_flash_attention_requested(config)`` from the checkpoint.

    Module-level twin of ``_ForwardOperationExtractor._resolve_flash_request_predicate``
    so passes that only hold a ``config`` dict (not the extractor) can resolve the
    same predicate. Returns the boolean the predicate evaluates to, or ``None`` when
    *test* is not one of these flash-request predicates. General: keys off the shared
    transformers predicate name, not any model.
    """
    node = test
    negate = False
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        negate = not negate
        node = node.operand
    if not isinstance(node, ast.Call):
        return None
    if _expr_name(node.func) not in _FLASH_REQUEST_PREDICATES:
        return None
    impl = (config or {}).get("_attn_implementation")
    resolved = impl.strip().lower() if isinstance(impl, str) and impl.strip() else "sdpa"
    is_flash = resolved in _FLASH_IMPL_NAMES
    return (not is_flash) if negate else is_flash


def _split_resolved_dropped_stmts(
    stmts: list[ast.stmt], config: dict[str, Any] | None
) -> tuple[list[ast.stmt], list[ast.stmt]]:
    """Partition forward statements into the ones that run and the ones dropped.

    Resolves ``if is_flash_attention_requested(config):`` branches from the
    checkpoint (undecidable branches are conservatively treated as *resolved*, so
    nothing there is ever mislabelled dead). Only these flash-request predicates
    are resolved; every other ``If`` keeps both arms as resolved.
    """
    resolved: list[ast.stmt] = []
    dropped: list[ast.stmt] = []

    def walk(body: list[ast.stmt]) -> None:
        for stmt in body:
            if isinstance(stmt, ast.If):
                verdict = _resolve_flash_predicate(stmt.test, config)
                if verdict is None:
                    walk(stmt.body)
                    walk(stmt.orelse)
                    continue
                taken = stmt.body if verdict else stmt.orelse
                untaken = stmt.orelse if verdict else stmt.body
                dropped.extend(untaken)
                walk(taken)
            else:
                resolved.append(stmt)

    walk(stmts)
    return resolved, dropped


def _loaded_names(stmts: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
    return names


def _forward_func_of(cls_node: ast.ClassDef) -> ast.FunctionDef | None:
    return next(
        (
            item
            for item in cls_node.body
            if isinstance(item, ast.FunctionDef) and item.name == "forward"
        ),
        None,
    )


def _flag_unused_interface_inputs(
    classes: dict[str, ClassStructure],
    config: dict[str, Any] | None,
) -> None:
    """Flag interface inputs a module declares but its selected impl never reads.

    A dispatched-attention module takes packed-attention metadata by keyword
    (``cu_seqlens``, ``max_seqlen``) that only the flash code path consumes; the
    default sdpa/eager path ignores some of it. Once ``is_flash_attention_requested``
    resolves to the non-flash branch, such a parameter is a genuine part of the
    module's *interface* yet dead in *this* implementation. Rather than fabricate a
    live data edge for it, drop it from the wired kernel inputs and record it as an
    ``unused_interface_inputs`` detail so the graph stays honest about the interface.

    General: keys off forward-signature params referenced only in dropped branches,
    for any config-dispatched attention module — no model or parameter names baked in.
    """
    for cls in classes.values():
        details = cls.forward_step_details.get(SYNTHETIC_ATTENTION)
        if not details:
            continue
        forward = _forward_func_of(cls.node)
        if forward is None:
            continue
        resolved_stmts, dropped_stmts = _split_resolved_dropped_stmts(
            forward.body, config
        )
        if not dropped_stmts:
            continue
        interface = _forward_input_names(forward)
        referenced_resolved = _loaded_names(resolved_stmts)
        referenced_dropped = _loaded_names(dropped_stmts)
        dead = sorted(
            name
            for name in interface
            if name in referenced_dropped and name not in referenced_resolved
        )
        if not dead:
            continue
        for name in dead:
            cls.attention_inputs.pop(name, None)
        cls.forward_step_details[SYNTHETIC_ATTENTION] = [
            *details,
            f"unused_interface_inputs: {','.join(dead)}",
        ]


def _resolve_dispatched_attention_kernel(
    classes: dict[str, ClassStructure],
    config: dict[str, Any] | None,
) -> None:
    """Name the kernel a dispatched attention call runs, from the checkpoint config.

    A forward that calls ``ALL_ATTENTION_FUNCTIONS[config._attn_implementation]``
    through a local variable leaves the AST with nothing but the variable's name.
    The checkpoint says which implementation that variable resolves to.

    When the checkpoint leaves ``_attn_implementation`` unset we resolve it to
    ``"sdpa"`` — the transformers default when nothing is configured — so a
    dispatched-attention step still names the kernel that actually runs instead
    of leaving the call site's opaque dispatch variable.
    """
    implementation = (config or {}).get("_attn_implementation")
    if isinstance(implementation, str) and implementation.strip():
        resolved = implementation.strip()
    else:
        resolved = "sdpa"
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
    """Label an attention leaf with its real resolved kernel (``sdpa``,
    ``recurrent_kimi_delta_attention``, …), not the generic word "Attention".

    ``_resolve_dispatched_attention_kernel`` rewrites the dispatch call
    (``kernel: attention_interface``) into the concrete kernel it selected
    (``kernel: sdpa``), so by the time we label we usually have the real name --
    show it raw. Fall back to "Attention" only when no kernel resolves, or when
    the recorded "kernel" is still an unresolved dispatch *variable* / synthetic
    wrapper name rather than a kernel that actually runs.
    """
    kernel = kernel_name_from_step_details(details)
    if not kernel:
        return "Attention"
    if kernel.lower() in (_ATTENTION_DISPATCH_NAMES | _SYNTHETIC_ATTENTION_NAMES):
        return "Attention"
    return kernel


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
        # Interface inputs the module declares but this resolved kernel never reads
        # (e.g. ``max_seqlen`` under sdpa) are surfaced as a distinct flag rather than
        # a wired input port — see ``_flag_unused_interface_inputs``.
        for line in details:
            if line.startswith("unused_interface_inputs:"):
                lines.append(line)
        return lines

    return []


def _capture_attention_inputs(
    node: ast.AST,
    var_chains: dict[str, list[str]],
    attention_inputs: dict[str, list[str]],
    forward_input_names: set[str] | None = None,
) -> None:
    forward_input_names = forward_input_names or set()
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
        # Packed-attention metadata (``cu_seq_lens_q=cu_seqlens``,
        # ``max_length_q=max_seqlen``) reaches the kernel by keyword and is a real
        # kernel input, but it arrives as a forward parameter of the attention
        # module (empty provenance chain) rather than a prior op — the positional
        # scan above misses both facts. Record such keyword tensor arguments so the
        # kernel declares them as input ports; the empty chain marks a boundary
        # forward input that the cross-module predecessor pass threads back to its
        # producer. General: any attention-interface keyword whose value is a
        # module forward input or a prior-step tensor.
        for keyword in call.keywords:
            value = keyword.value
            if not isinstance(value, ast.Name):
                continue
            chain = var_chains.get(value.id, [])
            if chain:
                attention_inputs[value.id] = list(chain)
            elif value.id in forward_input_names:
                attention_inputs.setdefault(value.id, [])
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


def _gated_norm_activation_from_forward(
    forward: ast.FunctionDef | None,
    init_func: ast.FunctionDef | None,
    config: dict[str, Any] | None,
) -> str | None:
    """Resolve the gate activation of a *gated norm* module from its forward AST.

    Structural signal (no class-name matching): the forward multiplies its
    normalized result by a gate argument passed through an activation registry,
    i.e. ``normalized * ACT2FN[self.<x>](<gate>)``. The activation key ``self.<x>``
    is resolved generically against the module's own init/config symbol table
    (``self.activation = "silu"`` or ``self.activation = config.<y>``). Returns the
    display name, or ``None`` when the module is not a gated norm. When the gate
    pattern *is* present but the key cannot be resolved, warns rather than guessing.
    """
    if forward is None:
        return None
    self_values = _self_config_values(init_func, config or {})
    forward_params = _forward_input_names(forward)

    def _resolve_registry_call(call: ast.Call) -> tuple[bool, str | None]:
        """(*is_gate_activation_call*, *display_name_or_None*) for ``ACT2FN[...](gate)``."""
        func_node = call.func
        if not isinstance(func_node, ast.Subscript):
            return False, None
        registry = (_expr_name(func_node.value) or "").rsplit(".", 1)[-1]
        if registry not in _ACTIVATION_REGISTRY_NAMES:
            return False, None
        # The activated tensor must trace back to a forward argument (the gate),
        # not to the normalized main path -- that is what makes this a *gate*
        # activation rather than an ordinary activation on the hidden states.
        if not any(
            isinstance(name, ast.Name) and name.id in forward_params
            for arg in call.args
            for name in ast.walk(arg)
        ):
            return False, None
        resolved = _config_value(func_node.slice, config or {}, self_values)
        if isinstance(resolved, str) and resolved.strip():
            return True, _display_activation_name(resolved)
        return True, None

    pattern_present = False
    for node in ast.walk(forward):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
            continue
        for operand in (node.left, node.right):
            if not isinstance(operand, ast.Call):
                continue
            is_gate_call, display = _resolve_registry_call(operand)
            if not is_gate_call:
                continue
            if display is not None:
                return display
            pattern_present = True
    if pattern_present:
        _log.warning(
            "gated norm %s applies a gate activation whose key could not be "
            "resolved from its init/config symbol table; leaving it unlabelled",
            getattr(forward, "name", "forward"),
        )
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
    self_values: dict | None = None,
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
    self_values = self_values or {}
    name_value_ast = _collect_name_value_ast(func)
    repeated_attrs = _repeated_self_call_attrs(func.body)

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
            self_values,
            name_value_ast,
            repeated_attrs=repeated_attrs,
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
    self_values: dict | None = None,
    name_value_ast: dict[str, ast.expr] | None = None,
    in_conditional: bool = False,
    repeated_attrs: frozenset[str] = frozenset(),
) -> str | None:
    if isinstance(node, ast.Assign):
        stmt_calls: list[str] = []
        # ``q, k = map(lambda x: BODY(x), (q, k))`` applies one lambda body to
        # each tuple element; rewritten into an equivalent per-element ``Tuple``
        # (see ``_expand_map_lambda_tuple``) so the existing ``ast.Tuple``
        # handling below walks each clone's own call and materializes a real
        # step for it (each clone stays distinct via its stamped discriminator).
        # Every other consumer of ``node.value`` still reads the untouched map()
        # call -- only call *extraction* needs the expanded shape.
        _extract_self_calls_ordered(
            _expand_map_lambda_tuple(node.value), stmt_calls, in_conditional, repeated_attrs
        )
        _inject_kernel_merge(
            node.value,
            var_chains,
            stmt_calls,
            attention_inputs,
            forward_step_details,
        )
        _capture_attention_inputs(
            node, var_chains, attention_inputs, forward_input_names
        )
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
        _extract_self_calls_ordered(
            node.value, stmt_calls, in_conditional, repeated_attrs
        )
        _inject_kernel_merge(
            node.value,
            var_chains,
            stmt_calls,
            attention_inputs,
            forward_step_details,
        )
        _capture_attention_inputs(
            node, var_chains, attention_inputs, forward_input_names
        )
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
        _extract_self_calls_ordered(
            node.value, stmt_calls, in_conditional, repeated_attrs
        )
        _inject_kernel_merge(
            node.value,
            var_chains,
            stmt_calls,
            attention_inputs,
            forward_step_details,
        )
        _capture_attention_inputs(
            node, var_chains, attention_inputs, forward_input_names
        )
        _capture_call_side_inputs(
            node, var_chains, forward_input_names, side_inputs, calls
        )
        return _register_forward_calls(stmt_calls, calls, norm_before, pending_norm)

    if isinstance(node, ast.AugAssign):
        stmt_calls = []
        _extract_self_calls_ordered(
            node.value, stmt_calls, in_conditional, repeated_attrs
        )
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
        _extract_self_calls_ordered(
            node.value, stmt_calls, in_conditional, repeated_attrs
        )
        _inject_kernel_merge(
            node.value,
            var_chains,
            stmt_calls,
            attention_inputs,
            forward_step_details,
        )
        _capture_attention_inputs(
            node, var_chains, attention_inputs, forward_input_names
        )
        _capture_call_side_inputs(
            node, var_chains, forward_input_names, side_inputs, calls
        )
        return _register_forward_calls(stmt_calls, calls, norm_before, pending_norm)

    if isinstance(node, ast.If):
        # Both arms are walked into the same ordered ``calls`` list regardless of
        # which one actually executes. When there IS a second arm (``orelse``),
        # an unrecognised free-function (``@fn_``) call from either arm could
        # collide with the other arm's -- unlike a ``self.<attr>`` submodule
        # producer (joined by an explicit ``Select`` phi, see
        # ``_emit_branch_select``), a free-function node has no branch-select
        # mechanism, so both would otherwise leak into the sequence as if
        # unconditional. Suppressing them there is safe: `skip_free_fn` only
        # drops the ``@fn_`` node emission, not the op(s) the call's *result*
        # feeds (those still emit via ``_emit`` with a ``condition:`` detail).
        # A single-armed ``if cond: ...`` (no ``else``) has no alternative arm
        # to collide with -- it is exactly one, condition-tagged block, so a
        # free-function call inside it is as real as any other op there and
        # must not be suppressed (that previously orphaned the op reading its
        # result: the op kept its edge target, but the target node was never
        # built).
        branch_in_conditional = in_conditional or bool(node.orelse)
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
                self_values,
                name_value_ast,
                in_conditional=branch_in_conditional,
                repeated_attrs=repeated_attrs,
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
                self_values,
                name_value_ast,
                in_conditional=in_conditional,
                repeated_attrs=repeated_attrs,
            )
        # Tensor operations are annotated by _ForwardOperationExtractor, but
        # expanded helper calls (for example `_apply_gate()`) are not operations
        # in this method's graph. Preserve their call-site loop context too so
        # their expanded children remain inside the source loop. Resolve the same
        # static trip count the extractor uses so the helper ops share the loop's
        # `Loop_N_iterations` frame instead of fragmenting into `Loop_repeated`.
        count = _loop_iteration_count_of(
            node, self_values or {}, name_value_ast or {}
        )
        loop_detail = (
            f"loop: {count} iterations" if count is not None else "loop: repeated"
        )
        for call in calls[first_loop_call:]:
            details = forward_step_details.setdefault(call, [])
            if not any(detail.startswith("loop:") for detail in details):
                details.append(loop_detail)
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
                self_values,
                name_value_ast,
                in_conditional=in_conditional,
                repeated_attrs=repeated_attrs,
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


def _pick_causal_lm_class(
    classes: dict[str, ClassStructure],
    config: dict[str, Any] | None = None,
) -> ClassStructure | None:
    """Return the top-level checkpoint class (the one owning the output head).

    The config's ``architectures`` list names the concrete class the checkpoint
    instantiates -- ``LlamaForCausalLM`` for a plain LM, but ``*ForConditionalGeneration``
    (or any custom name) for a multimodal wrapper. Prefer that authoritative key:
    the ``ForCausalLM``-substring heuristic below silently returns ``None`` for every
    non-causal wrapper, which drops its ``lm_head`` from the exported stack. Fall back
    to the name heuristic only when the config does not name a parsed class.
    """
    architectures = (
        config.get("architectures") if isinstance(config, dict) else None
    ) or []
    for arch in architectures:
        info = classes.get(arch)
        if info is not None:
            return info
    for info in classes.values():
        if info.name.endswith("ForCausalLM") or "ForCausalLM" in info.name:
            return info
    return None


def _looks_like_stack(info: ClassStructure) -> bool:
    """True when a class is itself the language stack (owns embeddings/decoder layers).

    A multimodal wrapper's ``.model`` is often a container that builds its real
    sub-stacks through ``_from_config`` (opaque factory calls that leave no class
    name to follow). Such a container owns neither the token embedding nor a decoder
    layer directly, so it must not be mistaken for the stack -- the bottom-up
    structural pick finds the true language model instead.
    """
    for attr, class_name in info.init_assignments.items():
        if _classify_role(attr, class_name) == "embedding":
            return True
        if DECODER_CLASS_RE.search(class_name):
            return True
    return False


def _pick_stack_model_class(
    classes: dict[str, ClassStructure],
    causal_lm: ClassStructure | None,
) -> ClassStructure | None:
    if causal_lm is not None:
        for attr in ("model", "transformer", "language_model"):
            child = causal_lm.init_assignments.get(attr)
            if child and child in classes and _looks_like_stack(classes[child]):
                return classes[child]
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

    # Collect the output head(s). A ForCausalLM / ForConditionalGeneration wrapper
    # owns the vocab projection (``lm_head``); the stack model may ALSO own its own
    # head-role reduction (e.g. a hyper-connection head that runs inside the stack
    # before the wrapper's projection). Take head-role children from BOTH so neither
    # is dropped -- the stack's heads run first, then the wrapper's. Inference repos
    # without a wrapper hang the head off the stack itself, which the stack pass still
    # covers. ``owner_base`` keeps every wrapper head sorted after the stack heads
    # since forward orders from two different owners are not otherwise comparable.
    seen_heads: set[tuple[str, str]] = set()
    for owner_base, head_owner in ((0, stack_model), (1000, causal_lm)):
        if head_owner is None:
            continue
        order = {attr: idx for idx, attr in enumerate(head_owner.forward_calls)}
        for attr, class_name in head_owner.init_assignments.items():
            if class_name in _SKIP_INIT_CLASS_NAMES:
                continue
            if _classify_role(attr, class_name) != "head":
                continue
            if (attr, class_name) in seen_heads:
                continue
            seen_heads.add((attr, class_name))
            tail.append(
                _stack_component(
                    attr_name=attr,
                    class_name=class_name,
                    role="head",
                    forward_order=owner_base + order.get(attr, 0),
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
    activation_param_bindings = _collect_activation_param_bindings(tree, config)
    visitor = _ModelAstVisitor(
        config=config,
        all_tensor_ops=all_tensor_ops,
        activation_param_bindings=activation_param_bindings,
        module_functions=_module_forward_functions(tree, config),
    )
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
    activation_param_bindings = _collect_activation_param_bindings(tree, config)
    vision_scoped = _vision_scoped_class_names(tree, config)
    visitor = _ModelAstVisitor(
        config=config,
        all_tensor_ops=all_tensor_ops,
        activation_param_bindings=activation_param_bindings,
        vision_scoped_classes=vision_scoped,
        vision_config=(config or {}).get("vision_config")
        if isinstance(config, dict)
        else None,
        module_functions=_module_forward_functions(tree, config),
    )
    visitor.visit(tree)
    finalize_class_registry(visitor.classes)
    _annotate_host_free_functions(visitor.classes, tree, config)
    _enrich_kernel_import_details(visitor.classes, external_imports)
    _resolve_dispatched_attention_kernel(visitor.classes, config)
    _flag_unused_interface_inputs(visitor.classes, config)
    _expand_unresolved_activation_classes(
        visitor.classes, tree, config, all_tensor_ops=all_tensor_ops
    )

    decoder = _pick_decoder_class(visitor.classes)
    causal_lm = _pick_causal_lm_class(visitor.classes, config)
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
    causal_lm = _pick_causal_lm_class(merged.class_registry, config)
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
