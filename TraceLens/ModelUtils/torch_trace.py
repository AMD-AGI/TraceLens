###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PyTorch-based model graph builder.

Replaces the AST-based pipeline with runtime tracing:

1. Instantiate model on meta device (zero memory, no weights)
2. Walk ``named_modules()`` for the module hierarchy
3. ``torch.fx.symbolic_trace`` per composite module for tensor ops
4. Forward hooks for ground-truth output shapes
5. Convert to Model Explorer graph format
"""

from __future__ import annotations

import logging
import re
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.fx
import transformers
from transformers import AutoConfig

from TraceLens.ModelUtils.ast_call_order import forward_call_order

_log = logging.getLogger(__name__)

# ── Styling constants ────────────────────────────────────────────────────────

_DARK_TEXT = "#1a1a1a"
_WHITE_TEXT = "#ffffff"

_STYLE_INPUT = {"backgroundColor": "#d9e8f5", "textColor": _DARK_TEXT}
_STYLE_OUTPUT = {"backgroundColor": "#d5f5d9", "textColor": _DARK_TEXT}
_STYLE_FORK = {"backgroundColor": "#a29bfe", "textColor": _WHITE_TEXT}
_STYLE_JOIN = {"backgroundColor": "#a29bfe", "textColor": _WHITE_TEXT}
_STYLE_EMBEDDING = {"backgroundColor": "#27ae60", "textColor": _WHITE_TEXT}
_STYLE_LINEAR = {"backgroundColor": "#bdc3c7", "textColor": _DARK_TEXT}
_STYLE_NORM = {"backgroundColor": "#f0e68c", "textColor": _DARK_TEXT}
_STYLE_ATTENTION = {"backgroundColor": "#5dade2", "textColor": _WHITE_TEXT}
_STYLE_ACTIVATION = {"backgroundColor": "#e67e22", "textColor": _WHITE_TEXT}
_STYLE_DEFAULT = {"backgroundColor": "#bdc3c7", "textColor": _DARK_TEXT}
_STYLE_OP = {"backgroundColor": "#ecf0f1", "textColor": _DARK_TEXT}


# ── Config patching ──────────────────────────────────────────────────────────


def _patch_config(config) -> None:
    """Ensure custom HF configs have all attributes the model code expects."""
    raw = config.to_dict()
    for key, val in raw.items():
        if not hasattr(config, key):
            setattr(config, key, val)
    for attr, default in [("max_length", 131072), ("use_cache", True)]:
        if not hasattr(config, attr):
            setattr(config, attr, default)


# ── Meta-device model instantiation ──────────────────────────────────────────


def _resolve_auto_classes(config) -> list[type]:
    """Return a ranked list of Auto classes to try, based on model card metadata.

    Checks ``config.auto_map`` first (explicit mapping from the repo),
    then falls back to the ``config.architectures`` class-name suffix.
    """
    # 1. auto_map: explicit mapping in the repo's config.json
    auto_map = getattr(config, "auto_map", None) or {}
    _PREFERRED_ORDER = [
        "AutoModelForCausalLM",
        "AutoModelForSeq2SeqLM",
        "AutoModelForConditionalGeneration",
        "AutoModel",
    ]
    for cls_name in _PREFERRED_ORDER:
        if cls_name in auto_map:
            return [getattr(transformers, cls_name)]

    # 2. architectures: infer candidates from class name suffix
    architectures = getattr(config, "architectures", None) or []
    if architectures:
        arch = architectures[0]
        _SUFFIX_CANDIDATES: dict[str, list[str]] = {
            "ForCausalLM": [
                "AutoModelForCausalLM",
            ],
            "ForConditionalGeneration": [
                "AutoModelForCausalLM",
                "AutoModelForSeq2SeqLM",
                "AutoModel",
            ],
            "ForSeq2SeqLM": [
                "AutoModelForSeq2SeqLM",
            ],
        }
        for suffix, candidates in _SUFFIX_CANDIDATES.items():
            if arch.endswith(suffix):
                return [getattr(transformers, c) for c in candidates]

    # 3. Last resort
    return [transformers.AutoModel]


def _instantiate_meta(checkpoint: str | Path) -> tuple[Any, Any]:
    """Load config and instantiate model on meta device.

    Inspects the model card (``auto_map`` / ``architectures``) to pick
    the correct Auto class, then instantiates on the meta device.

    Returns (model, config).
    """
    # Suppress the "torch_dtype is deprecated" warning from transformers
    _tf_logger = logging.getLogger("transformers.configuration_utils")
    _prev_level = _tf_logger.level
    _tf_logger.setLevel(logging.ERROR)

    try:
        config = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=True)
        _patch_config(config)

        auto_classes = _resolve_auto_classes(config)
        last_err: Exception | None = None

        for auto_cls in auto_classes:
            try:
                _log.info("Trying %s for %s", auto_cls.__name__, checkpoint)
                with torch.device("meta"):
                    model = auto_cls.from_config(config, trust_remote_code=True)
                model.eval()
                _log.info("Instantiated with %s", auto_cls.__name__)
                return model, config
            except (ValueError, KeyError) as exc:
                last_err = exc
                continue

        raise ValueError(
            f"Could not instantiate {checkpoint} with any Auto class "
            f"({', '.join(c.__name__ for c in auto_classes)}): {last_err}"
        )
    finally:
        _tf_logger.setLevel(_prev_level)


# ── Rotary embedding patching ────────────────────────────────────────────────


def _patch_rotary_embeddings(model: torch.nn.Module) -> None:
    """Patch rotary embedding modules to stay on meta device.

    Many HF models create CPU tensors inside rotary embedding forward().
    This causes meta-device forward passes to fail.
    """

    def _rotary_forward_meta(self, *args, **kwargs):
        dim = getattr(self, "dim", 64)
        max_seq = args[0] if args and isinstance(args[0], int) else 4096
        return torch.empty(max_seq, dim // 2, 2, device="meta", dtype=torch.float32)

    for _name, mod in model.named_modules():
        if "rotary" in type(mod).__name__.lower():
            mod.forward = types.MethodType(_rotary_forward_meta, mod)

    # Patch apply_rotary_pos_emb if present (shape-preserving)
    model_module = sys.modules.get(type(model).__module__)
    if model_module is None:
        # Try the inner transformer module
        for child in model.children():
            model_module = sys.modules.get(type(child).__module__)
            if model_module:
                break
    if model_module and hasattr(model_module, "apply_rotary_pos_emb"):
        model_module.apply_rotary_pos_emb = lambda x, *a, **k: x


# ── Shape capture via forward hooks ──────────────────────────────────────────


def _capture_shapes(
    model: torch.nn.Module,
    *,
    seq_len: int = 128,
    batch_size: int = 1,
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Run a meta-device forward pass and capture per-module I/O shapes.

    Returns (output_shapes, input_shapes) dicts keyed by module path.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    input_shapes: dict[str, tuple[int, ...]] = {}

    def _make_hook(name: str):
        def hook(_mod, inp, output):
            try:
                # Capture first tensor from input
                if isinstance(inp, (tuple, list)):
                    for item in inp:
                        if isinstance(item, torch.Tensor):
                            input_shapes[name] = tuple(item.shape)
                            break
                elif isinstance(inp, torch.Tensor):
                    input_shapes[name] = tuple(inp.shape)

                # Capture first tensor from output
                if isinstance(output, torch.Tensor):
                    shapes[name] = tuple(output.shape)
                elif isinstance(output, (tuple, list)):
                    for item in output:
                        if isinstance(item, torch.Tensor):
                            shapes[name] = tuple(item.shape)
                            break
            except Exception:
                pass

        return hook

    handles = []
    for name, mod in model.named_modules():
        handles.append(mod.register_forward_hook(_make_hook(name)))

    dummy = torch.zeros(batch_size, seq_len, dtype=torch.long, device="meta")
    try:
        # Enable meta-device workarounds for ops that fail on meta tensors
        torch.fx.experimental._config.meta_nonzero_assume_all_nonzero = True
        # Cast model to bfloat16 so matmul ops succeed on meta device
        original_dtype = next(
            (p.dtype for p in model.parameters() if p.dtype.is_floating_point),
            None,
        )
        if original_dtype and original_dtype != torch.bfloat16:
            model.to(torch.bfloat16)
        with torch.no_grad():
            model(dummy, use_cache=False)
    except Exception:
        pass
    finally:
        if original_dtype and original_dtype != torch.bfloat16:
            model.to(original_dtype)
        for h in handles:
            h.remove()

    return shapes, input_shapes


def _infer_shapes_from_weights(
    model: torch.nn.Module,
    captured: dict[str, tuple[int, ...]],
    *,
    batch_size: int = 1,
    seq_len: int = 128,
) -> dict[str, tuple[int, ...]]:
    """Infer output shapes from module weight dimensions for modules
    that didn't get shapes from the forward pass (e.g. vision encoder)."""
    shapes = dict(captured)
    for name, mod in model.named_modules():
        if name in shapes or not name:
            continue
        if isinstance(mod, torch.nn.Linear):
            shapes[name] = (batch_size, seq_len, mod.out_features)
        elif isinstance(mod, torch.nn.Embedding):
            shapes[name] = (batch_size, seq_len, mod.embedding_dim)
        elif isinstance(mod, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
            ns = getattr(mod, "normalized_shape", None)
            if ns:
                shapes[name] = (batch_size, seq_len, *ns)
        elif isinstance(mod, torch.nn.Conv2d):
            shapes[name] = (batch_size, mod.out_channels, seq_len, seq_len)
        elif isinstance(mod, torch.nn.Conv1d):
            shapes[name] = (batch_size, mod.out_channels, seq_len)
        elif isinstance(mod, torch.nn.Conv3d):
            shapes[name] = (batch_size, mod.out_channels, seq_len, seq_len, seq_len)
        elif hasattr(mod, "weight") and hasattr(mod.weight, "shape"):
            # Generic: use last dim of weight as output feature dim
            w_shape = mod.weight.shape
            if len(w_shape) >= 1:
                shapes[name] = (batch_size, seq_len, w_shape[0])
        # For norm-like modules, try to infer from the hidden_size attr
        elif hasattr(mod, "hidden_size"):
            shapes[name] = (batch_size, seq_len, mod.hidden_size)
    return shapes


def _infer_input_shapes_from_weights(
    model: torch.nn.Module,
    captured: dict[str, tuple[int, ...]],
    *,
    batch_size: int = 1,
    seq_len: int = 128,
) -> dict[str, tuple[int, ...]]:
    """Infer input shapes from module weight dimensions for modules
    that didn't get input shapes from the forward pass."""
    shapes = dict(captured)
    for name, mod in model.named_modules():
        if name in shapes or not name:
            continue
        if isinstance(mod, torch.nn.Linear):
            shapes[name] = (batch_size, seq_len, mod.in_features)
        elif isinstance(mod, torch.nn.Embedding):
            shapes[name] = (batch_size, seq_len)
        elif isinstance(mod, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
            ns = getattr(mod, "normalized_shape", None)
            if ns:
                shapes[name] = (batch_size, seq_len, *ns)
        elif isinstance(mod, torch.nn.Conv2d):
            shapes[name] = (batch_size, mod.in_channels, seq_len, seq_len)
    return shapes


# ── Call-graph capture via forward hooks ─────────────────────────────────────


def _tensor_ids(x: Any) -> list[int]:
    """Extract Python id()s of all tensors in a nested structure."""
    if isinstance(x, torch.Tensor):
        return [id(x)]
    if isinstance(x, (tuple, list)):
        ids = []
        for item in x:
            ids.extend(_tensor_ids(item))
        return ids
    if isinstance(x, dict):
        ids = []
        for v in x.values():
            ids.extend(_tensor_ids(v))
        return ids
    return []


def _capture_call_graph(
    model: torch.nn.Module,
    composite_modules: set[str],
    *,
    seq_len: int = 128,
    batch_size: int = 1,
) -> dict[str, list[tuple[str, str]]]:
    """Capture dataflow edges between child modules of each composite.

    Runs a forward pass, tracking which tensor objects flow between
    children of each composite module. Returns a dict mapping composite
    module path → list of (source_child, target_child) edges.

    Children that receive the composite's own input (not a sibling's
    output) are marked as receiving from a virtual "@input" source.
    """
    # For each module, record ALL calls (modules may be called multiple
    # times, e.g. HyperConnections).  Each entry is a list of per-call
    # records: (call_index, tensor_ids).
    pre_inputs: dict[str, list[tuple[int, list[int]]]] = defaultdict(list)
    post_outputs: dict[str, list[tuple[int, list[int]]]] = defaultdict(list)
    counter = [0]

    def _pre_hook(name: str):
        def hook(_mod, args, kwargs):
            pre_inputs[name].append(
                (counter[0], _tensor_ids(args) + _tensor_ids(kwargs))
            )
            counter[0] += 1

        return hook

    def _post_hook(name: str):
        def hook(_mod, _inp, output):
            post_outputs[name].append((counter[0], _tensor_ids(output)))
            counter[0] += 1

        return hook

    handles = []
    for name, mod in model.named_modules():
        if not name:
            continue
        handles.append(mod.register_forward_pre_hook(_pre_hook(name), with_kwargs=True))
        handles.append(mod.register_forward_hook(_post_hook(name)))

    dummy = torch.zeros(batch_size, seq_len, dtype=torch.long, device="meta")
    try:
        # Enable meta-device workarounds for ops that fail on meta tensors
        torch.fx.experimental._config.meta_nonzero_assume_all_nonzero = True
        # Cast model to bfloat16 so matmul ops succeed on meta device
        original_dtype = next(
            (p.dtype for p in model.parameters() if p.dtype.is_floating_point),
            None,
        )
        if original_dtype and original_dtype != torch.bfloat16:
            model.to(torch.bfloat16)
        with torch.no_grad():
            model(dummy, use_cache=False)
    except Exception:
        pass
    finally:
        if original_dtype and original_dtype != torch.bfloat16:
            model.to(original_dtype)
        for h in handles:
            h.remove()

    # Build per-composite edge lists.
    #
    # Modules may be called multiple times (e.g. HyperConnections that
    # wrap both the pre- and post-residual path).  We flatten all calls
    # into a single ordered timeline and process them in call order so
    # that a second call to the same module can consume outputs from
    # children that ran between the two calls.
    # Identify ModuleList/ModuleDict containers whose children should be
    # promoted as direct children of the grandparent (these containers
    # are never called directly; their items are).
    _container_paths: set[str] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.ModuleDict)):
            _container_paths.add(name)

    edges: dict[str, list[tuple[str, str]]] = {}
    for comp_path in composite_modules:
        # Collect direct children (promoting ModuleList/Dict items)
        children: set[str] = set()
        prefix = comp_path + "."
        for name in pre_inputs:
            if name.startswith(prefix):
                suffix = name[len(prefix) :]
                if "." not in suffix:
                    children.add(name)
                elif suffix.count(".") == 1:
                    # Check if the intermediate is a container (ModuleList)
                    parent_path = comp_path + "." + suffix.split(".")[0]
                    if parent_path in _container_paths:
                        children.add(name)

        if len(children) < 2:
            continue

        # Build a timeline of all calls (pre + post) for direct children
        # Each event: (call_index, "pre"/"post", child_name, tensor_ids)
        events: list[tuple[int, str, str, list[int]]] = []
        for child in children:
            for call_idx, tids in pre_inputs.get(child, []):
                events.append((call_idx, "pre", child, tids))
            for call_idx, tids in post_outputs.get(child, []):
                events.append((call_idx, "post", child, tids))
        events.sort(key=lambda e: e[0])

        # Map tensor id → producing child (or "@input" for composite's own input)
        producer: dict[int, str] = {}
        # The composite's own input tensors are the "root" source
        for _call_idx, tids in pre_inputs.get(comp_path, []):
            for tid in tids:
                producer[tid] = "@input"

        child_edges: list[tuple[str, str]] = []
        untracked_consumers: dict[int, list[str]] = defaultdict(list)

        for _call_idx, event_type, child, tids in events:
            if event_type == "pre":
                # Check which producer this child's inputs come from
                sources: set[str] = set()
                for tid in tids:
                    if tid in producer:
                        sources.add(producer[tid])
                    else:
                        untracked_consumers[tid].append(child)
                for src in sorted(sources):
                    if src != child:
                        child_edges.append((src, child))
            else:  # "post"
                # Register this child's outputs
                for tid in tids:
                    producer[tid] = child

        # Sequential fallback: children with no incoming edges are
        # connected from the previous child in execution order.  The
        # very first child gets "@input".  This handles inline tensor ops
        # (e.g. residual combinations) and meta-device forward passes
        # where tensor-ID tracking produces no overlap.
        #
        # Run BEFORE untracked_consumers so sequential chaining takes
        # priority over spurious @input edges from shared tensor IDs.
        children_ordered = sorted(
            children,
            key=lambda c: (
                min((idx for idx, _ in pre_inputs.get(c, [(999,)])), default=999)
            ),
        )
        children_with_edges = {tgt for _, tgt in child_edges}
        last_child: str | None = None
        for child in children_ordered:
            if child not in children_with_edges:
                if last_child is not None:
                    child_edges.append((last_child, child))
                else:
                    child_edges.append(("@input", child))
                children_with_edges.add(child)
            # Update last_child to the most recently completed child
            last_child = child

        # Children sharing the same untracked input tensor are parallel.
        # Treat them as all coming from "@input" (only if still unconnected).
        children_with_edges = {tgt for _, tgt in child_edges}
        for _tid, consumers in untracked_consumers.items():
            if len(consumers) > 1:
                for child in consumers:
                    if child not in children_with_edges:
                        child_edges.append(("@input", child))
                        children_with_edges.add(child)

        if child_edges:
            # Deduplicate
            seen: set[tuple[str, str]] = set()
            unique = []
            for e in child_edges:
                if e not in seen:
                    seen.add(e)
                    unique.append(e)

            # Filter spurious @input edges: if a child has a sibling
            # source, the @input edge is likely a secondary/control input
            # (e.g. attention_mask, position_ids) — not the main data flow.
            children_with_sibling_src = {tgt for src, tgt in unique if src != "@input"}
            unique = [
                (src, tgt)
                for src, tgt in unique
                if src != "@input" or tgt not in children_with_sibling_src
            ]

            edges[comp_path] = unique

    # Multi-modal fallback: the root module itself is never traced (only
    # named composites are), so top-level wiring between e.g. a vision
    # encoder and a language model is otherwise left unresolved. Rather
    # than hardcoding module names like "visual"/"language_model", use
    # the ACTUAL traced call behavior: if exactly one top-level child was
    # invoked during the forward pass (because optional inputs like
    # pixel_values were omitted) and one or more siblings were never
    # invoked, treat the uninvoked sibling(s) as parallel input branches
    # that feed into the invoked one — the standard pattern for an
    # optional modality encoder whose output gets merged into the main
    # sequence before the primary model runs.
    top_children = [n for n, _ in model.named_children()]
    if len(top_children) >= 2:
        invoked = [n for n in top_children if pre_inputs.get(n) or post_outputs.get(n)]
        not_invoked = [n for n in top_children if n not in invoked]
        if len(invoked) == 1 and not_invoked:
            main = invoked[0]
            # List the uninvoked side-branch(es) before the main branch so
            # the exec-order topological sort (which processes "@input"
            # targets in list order) places them first — they feed INTO
            # the main branch, so they must execute first.
            root_edges = []
            for side in not_invoked:
                root_edges.append(("@input", side))
                root_edges.append((side, main))
            root_edges.append(("@input", main))
            edges[""] = root_edges

    return edges


def _symbolise(
    shape: tuple[int, ...], *, batch_size: int = 1, seq_len: int = 128
) -> str:
    """Convert shape tuple to symbolic string like ``B x S x 4096``."""
    parts = []
    for d in shape:
        if d == batch_size:
            parts.append("B")
        elif d == seq_len:
            parts.append("S")
        else:
            parts.append(str(d))
    return " x ".join(parts)


# ── torch.fx per-module tracing ──────────────────────────────────────────────


def _fx_trace_module(mod: torch.nn.Module) -> torch.fx.Graph | None:
    """Try to symbolically trace a module.  Returns None on failure."""
    try:
        traced = torch.fx.symbolic_trace(mod)
        return traced.graph
    except Exception:
        pass

    # Handle modules with unregistered nn.Module activations from dicts
    # like ACT2FN[self.activation] — build a mirror class with the
    # activation registered as a proper submodule.
    try:
        import inspect, re

        cls = type(mod)
        src = inspect.getsource(cls.forward)
        if "ACT2FN" not in src:
            return None

        act_name = getattr(mod, "activation", None)
        if not isinstance(act_name, str):
            return None

        from transformers.activations import ACT2FN

        act_mod = ACT2FN.get(act_name)
        if act_mod is None:
            return None
        # ACT2FN may return a class or an instance
        if isinstance(act_mod, type) and issubclass(act_mod, torch.nn.Module):
            act_mod = act_mod()
        if not isinstance(act_mod, torch.nn.Module):
            return None

        # Determine hidden_size from weight parameter
        w = getattr(mod, "weight", None)
        hidden_size = w.shape[0] if w is not None else 8
        eps = getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6))

        # Build a mirror class with act_fn registered
        class _Mirror(torch.nn.Module):
            def __init__(self_):
                super().__init__()
                self_.weight = torch.nn.Parameter(
                    torch.ones(hidden_size, device="meta")
                )
                self_.variance_epsilon = eps
                self_.act_fn = act_mod

        # Patch forward source: replace ACT2FN[self.activation] → self.act_fn
        new_src = re.sub(r"ACT2FN\[self\.activation\]", "self.act_fn", src)
        lines = new_src.split("\n")
        start = next(i for i, l in enumerate(lines) if "def forward" in l)
        lines = lines[start:]
        indent = len(lines[0]) - len(lines[0].lstrip())
        lines = [l[indent:] if len(l) > indent else l for l in lines]
        new_src = "\n".join(lines)

        ns: dict = {"torch": torch}
        exec(compile(new_src, "<mirror>", "exec"), ns)  # noqa: S102
        _Mirror.forward = ns["forward"]

        mirror = _Mirror()
        traced = torch.fx.symbolic_trace(mirror)
        return traced.graph
    except Exception:
        return None


_FX_OP_LABELS = {
    "silu": "SiLU",
    "relu": "ReLU",
    "gelu": "GELU",
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
    "softmax": "Softmax",
    "chunk": "Chunk",
    "split": "Split",
    "cat": "Concat",
    "stack": "Stack",
    "mul": "Multiply",
    "add": "Add",
    "matmul": "MatMul",
    "bmm": "BatchMatMul",
    "baddbmm": "BatchMatMul",
    "scaled_dot_product_attention": "SDPA",
    "layer_norm": "LayerNorm",
    "dropout": "Dropout",
    "getitem": "Index",
    "view": "View",
    "reshape": "Reshape",
    "transpose": "Transpose",
    "permute": "Permute",
    "contiguous": "Contiguous",
    "unsqueeze": "Unsqueeze",
    "squeeze": "Squeeze",
    "expand": "Expand",
    "mean": "Mean",
    "sum": "Sum",
}


def _fx_op_label(node: torch.fx.Node) -> str:
    """Human-readable label for an FX graph node."""
    if node.op == "call_module":
        return str(node.target).rsplit(".", 1)[-1]
    name = node.name
    # Strip trailing _N suffixes (getitem_1 → getitem)
    base = re.sub(r"_\d+$", "", name)
    return _FX_OP_LABELS.get(base, base.replace("_", " ").title())


def _is_interesting_op(node: torch.fx.Node) -> bool:
    """Keep tensor computation operations; skip weights, placeholders, dtype access."""
    if node.op in ("placeholder", "output", "get_attr"):
        return False
    # Filter getattr() calls — these just access tensor properties like dtype
    if node.op == "call_function":
        target = node.target
        if target is getattr or (
            hasattr(target, "__name__") and target.__name__ == "getattr"
        ):
            return False
    return True


# ── Module classification ────────────────────────────────────────────────────


def _classify_module(mod: torch.nn.Module) -> str:
    """Classify a module for styling and labeling."""
    name = type(mod).__name__
    if isinstance(mod, torch.nn.Embedding):
        return "embedding"
    if isinstance(mod, (torch.nn.Linear,)):
        return "linear"
    if isinstance(mod, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
        return "norm"
    if re.search(r"(?i)(RMSNorm|LayerNorm|GroupNorm)", name):
        return "norm"
    if re.search(r"(?i)(Attention)", name):
        return "attention"
    if re.search(r"(?i)(Embedding)", name):
        return "embedding"
    if isinstance(mod, (torch.nn.SiLU, torch.nn.ReLU, torch.nn.GELU, torch.nn.Tanh)):
        return "activation"
    return "default"


def _style_for(category: str) -> dict[str, str]:
    return {
        "embedding": _STYLE_EMBEDDING,
        "linear": _STYLE_LINEAR,
        "norm": _STYLE_NORM,
        "attention": _STYLE_ATTENTION,
        "activation": _STYLE_ACTIVATION,
        "op": _STYLE_OP,
        "input": _STYLE_INPUT,
    }.get(category, _STYLE_DEFAULT)


def _module_label(mod: torch.nn.Module) -> str:
    """Friendly label for a module."""
    cls = type(mod).__name__
    if isinstance(mod, torch.nn.Linear):
        return "Linear"
    if isinstance(mod, torch.nn.Embedding):
        return "Embedding"
    # For trivial activation wrappers (e.g. transformers SiLUActivation),
    # try FX tracing to get the real op name
    if not list(mod.children()):
        cls_mod = type(mod).__module__ or ""
        if not cls_mod.startswith("torch."):
            try:
                graph = torch.fx.symbolic_trace(mod).graph
                ops = [n for n in graph.nodes if n.op not in ("placeholder", "output")]
                if len(ops) == 1:
                    return _fx_op_label(ops[0])
            except Exception:
                pass
    return cls


# ── Layer deduplication ──────────────────────────────────────────────────────


@dataclass
class LayerGroup:
    """A group of structurally identical layers inside a ModuleList."""

    class_name: str
    indices: list[int]
    representative: int  # index to keep

    @property
    def count(self) -> int:
        return len(self.indices)


def _detect_repeated_layers(
    model: torch.nn.Module,
) -> dict[str, list[LayerGroup]]:
    """Find repeated layer blocks, grouping by structural type.

    For heterogeneous ModuleLists (e.g. 34 linear-attn + 11 sparse-attn
    layers), creates separate groups for each distinct layer type.

    Returns dict mapping container path to list of LayerGroups.
    """
    repeats: dict[str, list[LayerGroup]] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) <= 1:
            continue

        # Group children by their structural signature (class + child names)
        groups: dict[str, list[int]] = defaultdict(list)
        for i, child in enumerate(mod):
            # Signature = class name + sorted child module class names
            child_sig = type(child).__name__
            child_structure = tuple(
                (cname, type(cmod).__name__) for cname, cmod in child.named_children()
            )
            sig = f"{child_sig}:{child_structure}"
            groups[sig].append(i)

        layer_groups = []
        for _sig, indices in groups.items():
            cls_name = type(mod[indices[0]]).__name__
            layer_groups.append(
                LayerGroup(
                    class_name=cls_name,
                    indices=indices,
                    representative=indices[0],
                )
            )

        if any(g.count > 1 for g in layer_groups):
            repeats[name] = layer_groups

    return repeats


# ── Graph building ───────────────────────────────────────────────────────────


def _shape_attrs(shape_str: str, dtype: str = "bfloat16") -> list[dict]:
    """Build output shape attributes for a node."""
    return [
        {"key": "output_shape", "value": f"{shape_str} {dtype}"},
        {"key": "output_dtype", "value": dtype},
    ]


def _reorder_group_attrs(
    group_attrs: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Reorder each group's attributes so input_shape is first and
    output_shape is last, with all other keys in between."""
    result: dict[str, dict[str, str]] = {}
    for ns, attrs in group_attrs.items():
        ordered: dict[str, str] = {}
        if "input_shape" in attrs:
            ordered["input_shape"] = attrs["input_shape"]
        for k, v in attrs.items():
            if k not in ("input_shape", "output_shape"):
                ordered[k] = v
        if "output_shape" in attrs:
            ordered["output_shape"] = attrs["output_shape"]
        result[ns] = ordered
    return result


def _output_metadata(shape_str: str, dtype: str = "bfloat16") -> list[dict]:
    compact = shape_str.replace(" x ", "x")
    return [
        {
            "id": "0",
            "attrs": [
                {"key": "shape", "value": f"{shape_str} {dtype}"},
                {"key": "tensor_shape", "value": f"{compact} {dtype}"},
                {"key": "dtype", "value": dtype},
            ],
        }
    ]


def _build_fact_sheet(model_name: str, config) -> str:
    """Build an HTML fact sheet from the config, including sub-configs."""
    lines = [f"<b>{model_name}</b>"]

    # Collect all config dicts (top-level + sub-configs like text_config)
    top = config.to_dict()
    sub_configs: dict[str, dict] = {}
    for key, val in top.items():
        if isinstance(val, dict) and any(
            k in val for k in ("hidden_size", "num_hidden_layers", "vocab_size")
        ):
            sub_configs[key] = val

    _FACT_KEYS = [
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
        "dtype",
        "num_local_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    ]

    def _emit(d: dict, prefix: str = "") -> None:
        for key in _FACT_KEYS:
            val = d.get(key)
            if val is not None:
                label = f"{prefix}{key}" if prefix else key
                lines.append(f"  {label}: {val}")

    if sub_configs:
        # Multi-modal: show sub-configs with headers
        # Top-level keys first
        for key in ("model_type", "dtype"):
            val = top.get(key)
            if val is not None:
                lines.append(f"  {key}: {val}")
        for section, d in sub_configs.items():
            section_label = section.replace("_config", "").replace("_", " ").title()
            lines.append(f"\n<b>{section_label}</b>")
            _emit(d)
    else:
        _emit(top)

    # Layer type breakdown
    return "\n".join(lines)


def build_graph(
    checkpoint: str | Path,
    *,
    seq_len: int = 128,
    batch_size: int = 1,
    title: str | None = None,
) -> dict[str, Any]:
    """Build a complete Model Explorer payload from a checkpoint.

    This is the main entry point, replacing the entire AST pipeline.
    """
    model, config = _instantiate_meta(checkpoint)
    _patch_rotary_embeddings(model)

    # ── Capture shapes ───────────────────────────────────────────────────
    hook_shapes, hook_input_shapes = _capture_shapes(
        model, seq_len=seq_len, batch_size=batch_size
    )
    raw_shapes = _infer_shapes_from_weights(
        model, hook_shapes, batch_size=batch_size, seq_len=seq_len
    )
    shapes: dict[str, str] = {}
    for path, shape in raw_shapes.items():
        shapes[path] = _symbolise(shape, batch_size=batch_size, seq_len=seq_len)
    raw_input_shapes = _infer_input_shapes_from_weights(
        model, hook_input_shapes, batch_size=batch_size, seq_len=seq_len
    )
    input_shapes: dict[str, str] = {}
    for path, shape in raw_input_shapes.items():
        input_shapes[path] = _symbolise(shape, batch_size=batch_size, seq_len=seq_len)

    # ── Detect repeated layers ───────────────────────────────────────────
    repeats = _detect_repeated_layers(model)

    # ── Determine dtype ──────────────────────────────────────────────────
    dtype = getattr(config, "dtype", None)
    # Search sub-configs if top-level is None
    if dtype is None:
        for attr in ("text_config", "language_config", "decoder_config"):
            sub = getattr(config, attr, None)
            if sub is not None:
                dtype = getattr(sub, "dtype", None)
                if dtype is not None:
                    break
    dtype = str(dtype or "bfloat16").replace("torch.", "")

    # ── Build model name ─────────────────────────────────────────────────
    model_name = title or getattr(config, "_name_or_path", str(checkpoint))
    model_type = getattr(config, "model_type", "unknown")

    # ── Walk module tree and build nodes ──────────────────────────────────
    nodes: list[dict[str, Any]] = []
    group_attrs: dict[str, dict[str, str]] = {}
    group_configs: list[dict[str, Any]] = []
    edges_from: dict[str, str] = {}  # module_path → node_id

    # Input node
    nodes.append(
        {
            "id": "@input",
            "label": "Tokenized text",
            "namespace": "",
            "attrs": [
                {"key": "synthetic", "value": "@input"},
                *_shape_attrs("B x S", "int64"),
            ],
            "style": _STYLE_INPUT,
            "outputsMetadata": _output_metadata("B x S", "int64"),
        }
    )

    # ── Build skip set from repeated layer groups ────────────────────────
    # Keep one representative per *distinct layer type* in each ModuleList.
    # When there are multiple types (e.g. LinearAttn vs SparseAttn), show
    # them as parallel branches so the viewer renders them side-by-side.
    skip_layers: set[str] = set()
    layer_group_map: dict[str, list[LayerGroup]] = repeats
    container_total: dict[str, int] = {}
    # Set of representative layer paths to keep
    representative_paths: set[str] = set()

    for container_path, groups in layer_group_map.items():
        total = sum(g.count for g in groups)
        container_total[container_path] = total
        for group in groups:
            representative_paths.add(f"{container_path}.{group.representative}")
        # Skip all layers that aren't a representative
        for i in range(total):
            path = f"{container_path}.{i}"
            if path not in representative_paths:
                skip_layers.add(path)

    # ── Collect module info (preserving registration order) ──────────────
    module_children: dict[str, list[str]] = defaultdict(list)
    module_map: dict[str, torch.nn.Module] = {}
    module_order: list[str] = []
    # Also build attr_name map: path → attribute name used in parent
    attr_names: dict[str, str] = {}

    for name, mod in model.named_modules():
        module_map[name] = mod
        if name:
            module_order.append(name)
            # The last part of the dot-path is the attr name
            attr_names[name] = name.rsplit(".", 1)[-1]
        if "." in name:
            parent = name.rsplit(".", 1)[0]
            module_children[parent].append(name)
        elif name:
            module_children[""].append(name)

    # ── Reorder using AST-derived execution order ────────────────────────
    # named_modules() only reflects declaration order (the order
    # submodules were assigned in __init__), which can differ from the
    # order they're actually invoked in forward(). This matters most for
    # composites that are never invoked at all during the dummy trace
    # (e.g. an optional modality encoder branch whose inputs, like
    # pixel_values, were omitted) — there's no runtime call-graph signal
    # for their internals, so declaration order is the only fallback,
    # and it can be wrong (e.g. a norm declared last but applied before
    # a later-declared submodule). Fix this by statically reading each
    # module's own forward() source to learn the true call order of its
    # direct children — still "from the code", just read rather than run.
    # Reuses the well-developed call-order extraction from the pre-
    # torch_trace AST pipeline (ast_call_order.py), which correctly
    # handles nested/method-chained calls in true evaluation order.
    def _dfs_order(path: str, mod: torch.nn.Module, out: list[str]) -> None:
        declared = list(mod.named_children())
        call_order = forward_call_order(mod)
        if call_order:
            rank = {n: i for i, n in enumerate(call_order)}
            declared = sorted(declared, key=lambda np: rank.get(np[0], len(call_order)))
        for cname, cmod in declared:
            cpath = f"{path}.{cname}" if path else cname
            out.append(cpath)
            _dfs_order(cpath, cmod, out)

    _ast_ordered: list[str] = []
    _dfs_order("", model, _ast_ordered)
    if len(_ast_ordered) == len(module_order):
        module_order = _ast_ordered
        module_map = {"": module_map[""], **{p: module_map[p] for p in module_order}}

    # Identify leaf vs composite modules
    composite_modules: set[str] = set()
    for name, mod in model.named_modules():
        if name and list(mod.children()):
            composite_modules.add(name)

    # ── Capture call graph for dataflow-aware edge wiring ────────────────
    call_graph = _capture_call_graph(
        model, composite_modules, seq_len=seq_len, batch_size=batch_size
    )

    # Map skipped layer paths to their representative so the call graph
    # can resolve edges through collapsed layers.  E.g. if layers 0-2
    # collapse to rep 0, an edge "layers.2 → layers.3" maps to
    # "layers.0 → layers.3".
    _skip_to_rep: dict[str, str] = {}
    for container_path, groups in layer_group_map.items():
        for group in groups:
            rep_path = f"{container_path}.{group.representative}"
            for idx in group.indices:
                p = f"{container_path}.{idx}"
                if p != rep_path:
                    _skip_to_rep[p] = rep_path

    def _map_cg_path(p: str) -> str:
        """Map a possibly-skipped module path to its representative."""
        if p in _skip_to_rep:
            return _skip_to_rep[p]
        # Check if p is a child of a skipped path
        for skip, rep in _skip_to_rep.items():
            if p.startswith(skip + "."):
                return rep + p[len(skip) :]
        return p

    # Remap call_graph edges through collapsed layers
    remapped_cg: dict[str, list[tuple[str, str]]] = {}
    for comp_path, edges in call_graph.items():
        new_edges: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for src, tgt in edges:
            ms = _map_cg_path(src) if src != "@input" else src
            mt = _map_cg_path(tgt)
            if ms == mt:
                continue  # Skip self-edges from collapsed layers
            pair = (ms, mt)
            if pair not in seen:
                seen.add(pair)
                new_edges.append(pair)
        if new_edges:
            remapped_cg[_map_cg_path(comp_path)] = new_edges
    call_graph = remapped_cg

    # For multi-group containers, replace the sequential layer chain in
    # the parent's call graph with Fork/Join parallel edges.
    for container_path, groups in layer_group_map.items():
        if len(groups) <= 1:
            continue
        parent_path = container_path.rsplit(".", 1)[0] if "." in container_path else ""
        if parent_path not in call_graph:
            continue
        rep_paths = {f"{container_path}.{g.representative}" for g in groups}
        fork_path = container_path + ".@fork"
        join_path = container_path + ".@join"

        old_edges = call_graph[parent_path]
        new_edges = []
        # Find the predecessor and successor of the layer block
        # Predecessor: the source of edges targeting any rep layer
        predecessors: set[str] = set()
        successors: set[str] = set()
        for src, tgt in old_edges:
            if tgt in rep_paths and src not in rep_paths:
                predecessors.add(src)
            if src in rep_paths and tgt not in rep_paths:
                successors.add(tgt)

        # Keep non-layer edges, skip inter-layer edges
        for src, tgt in old_edges:
            if src in rep_paths or tgt in rep_paths:
                continue  # Skip all edges involving layers
            new_edges.append((src, tgt))

        # Add Fork/Join edges
        for pred in predecessors:
            new_edges.append((pred, fork_path))
        for rep in rep_paths:
            new_edges.append((fork_path, rep))
            new_edges.append((rep, join_path))
        for succ in successors:
            new_edges.append((join_path, succ))

        call_graph[parent_path] = new_edges

        # Also transform the container's own call_graph entry so that
        # the sequential layer chain is replaced with parallel branches.
        if container_path in call_graph:
            cont_old = call_graph[container_path]
            cont_new = []
            cont_preds: set[str] = set()
            cont_succs: set[str] = set()
            for src, tgt in cont_old:
                if tgt in rep_paths and src not in rep_paths:
                    cont_preds.add(src)
                if src in rep_paths and tgt not in rep_paths:
                    cont_succs.add(tgt)
            for src, tgt in cont_old:
                if src in rep_paths or tgt in rep_paths:
                    continue
                cont_new.append((src, tgt))
            for pred in cont_preds:
                cont_new.append((pred, fork_path))
            for rep in rep_paths:
                cont_new.append((fork_path, rep))
                cont_new.append((rep, join_path))
            for succ in cont_succs:
                cont_new.append((join_path, succ))
            call_graph[container_path] = cont_new

    def _should_skip(path: str) -> bool:
        for skip in skip_layers:
            if path == skip or path.startswith(skip + "."):
                return True
        return False

    # ── Build namespace labels using attr names + layer group labels ─────
    container_set: set[str] = set(container_total.keys())

    # Map each representative path → its group label (e.g. "3x LinearAttn+MLP")
    rep_group_label: dict[str, str] = {}
    for container_path, groups in layer_group_map.items():
        for group in groups:
            rep_path = f"{container_path}.{group.representative}"
            # Build a descriptive label for this layer type
            rep_mod = module_map.get(rep_path)
            if rep_mod:
                attn_type = ""
                mlp_type = ""
                for cname, cmod in rep_mod.named_children():
                    if "attn" in cname and not attn_type:
                        attn_type = type(cmod).__name__
                    if "mlp" in cname and not mlp_type:
                        mlp_type = type(cmod).__name__
                parts_label = attn_type
                if mlp_type:
                    parts_label += f" + {mlp_type}"
                rep_group_label[rep_path] = (
                    f"{group.count}x {group.class_name} ({parts_label})"
                )
            else:
                rep_group_label[rep_path] = f"{group.count}x {group.class_name}"

    def _namespace_for(path: str) -> str:
        """Build a namespace using attr names, with layer group labels."""
        parts = path.split(".")
        if len(parts) <= 1:
            return ""
        ns_parts = []
        i = 0
        while i < len(parts) - 1:
            prefix = ".".join(parts[: i + 1])

            if prefix in container_set:
                # ModuleList — next part is the layer index
                if i + 1 < len(parts) - 1:
                    layer_prefix = ".".join(parts[: i + 2])
                    # Use per-group label if this is a representative
                    label = rep_group_label.get(layer_prefix)
                    if label:
                        ns_parts.append(label)
                    else:
                        # Fallback: total count
                        total = container_total[prefix]
                        mod = module_map.get(layer_prefix)
                        cls_name = type(mod).__name__ if mod else "Layer"
                        ns_parts.append(f"{total}x {cls_name}")
                    i += 2  # skip container + index
                    continue
                i += 1
                continue

            # Use attr name for composites, class name for clarity
            if prefix in composite_modules:
                attr = attr_names.get(prefix, parts[i])
                mod = module_map.get(prefix)
                cls_name = type(mod).__name__ if mod else attr
                # Use attr name if it's informative, class name otherwise
                if attr.isdigit():
                    ns_parts.append(cls_name)
                else:
                    ns_parts.append(f"{attr} ({cls_name})")
            i += 1
        return "/".join(ns_parts)

    def _node_id(path: str) -> str:
        return path.replace(".", "/")

    # ── Try torch.fx on composite modules to get internal ops ────────────
    fx_graphs: dict[str, list[dict]] = {}
    # Children referenced by call_module in parent's FX graph
    fx_referenced_children: set[str] = set()
    # FX-derived incoming edges for call_module children
    fx_child_edges: dict[str, list[dict]] = {}  # child_path → [edge dicts]
    # FX graph output node IDs for each composite
    fx_output_ids: dict[str, list[str]] = {}  # comp_path → [output node IDs]

    for path in module_order:
        if path not in composite_modules:
            continue
        if _should_skip(path):
            continue
        mod = module_map[path]
        has_leaf = any(not list(c.children()) for c in mod.children())
        if not has_leaf:
            continue

        graph = _fx_trace_module(mod)
        if graph is None:
            continue

        namespace = _namespace_for(path) or type(mod).__name__
        op_nodes = []
        node_map: dict[str, str] = {}

        # Only the first placeholder (main data tensor) maps to @input.
        # Secondary placeholders (attention_mask, position_ids, etc.) are
        # control inputs that shouldn't create visible data-flow edges.
        first_placeholder_seen = False
        for fx_node in graph.nodes:
            if not _is_interesting_op(fx_node):
                if fx_node.op == "placeholder":
                    if not first_placeholder_seen:
                        node_map[fx_node.name] = "@input"
                        first_placeholder_seen = True
                    # else: skip — secondary control inputs
                # Capture output node args for composite output tracking
                if fx_node.op == "output":
                    out_ids = []
                    for arg in fx_node.all_input_nodes:
                        nid = node_map.get(arg.name)
                        if nid:
                            out_ids.append(nid)
                    if out_ids:
                        fx_output_ids[path] = out_ids
                continue

            if fx_node.op == "call_module":
                target = str(fx_node.target)
                child_path = f"{path}.{target}"
                node_map[fx_node.name] = _node_id(child_path)
                fx_referenced_children.add(child_path)
                # Record incoming edges from the FX graph — only from
                # sibling modules and inline ops, NOT from placeholders.
                # Placeholder-sourced edges are left to the call_graph /
                # sequential wiring to resolve correctly.
                child_incoming = []
                for arg in fx_node.all_input_nodes:
                    src = node_map.get(arg.name)
                    if src and src != "@input":
                        child_incoming.append({"sourceNodeId": src})
                if child_incoming:
                    fx_child_edges[child_path] = child_incoming
                continue

            label = _fx_op_label(fx_node)
            op_id = f"{_node_id(path)}/{fx_node.name}"
            node_map[fx_node.name] = op_id

            incoming = []
            for arg in fx_node.args:
                if isinstance(arg, torch.fx.Node) and arg.name in node_map:
                    incoming.append(
                        {
                            "sourceNodeId": node_map[arg.name],
                            "sourceNodeOutputId": "0",
                            "targetNodeInputId": str(len(incoming)),
                        }
                    )
                elif isinstance(arg, (tuple, list)):
                    for item in arg:
                        if isinstance(item, torch.fx.Node) and item.name in node_map:
                            incoming.append(
                                {
                                    "sourceNodeId": node_map[item.name],
                                    "sourceNodeOutputId": "0",
                                    "targetNodeInputId": str(len(incoming)),
                                }
                            )

            op_node = {
                "id": op_id,
                "label": label,
                "namespace": namespace,
                "attrs": [{"key": "operation", "value": "tensor_op"}],
                "style": _STYLE_OP,
            }
            op_node["incomingEdges"] = incoming
            op_nodes.append(op_node)

        if op_nodes:
            fx_graphs[path] = op_nodes

    # ── Try torch.fx on custom leaf modules to expand their ops ──────────
    for path in module_order:
        if path in composite_modules or path in fx_graphs:
            continue
        if _should_skip(path):
            continue
        # Skip if parent composite already has an FX graph (this leaf is
        # already referenced via call_module in the parent's graph)
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        if parent in fx_graphs:
            continue
        mod = module_map[path]
        cls = type(mod)
        # Only expand non-standard modules (not torch.nn builtins)
        if cls.__module__ and cls.__module__.startswith("torch."):
            continue
        if list(mod.children()):
            continue  # has children — handled as composite

        graph = _fx_trace_module(mod)
        if graph is None:
            continue

        namespace = _namespace_for(path)
        # Build a parent namespace for the expanded ops
        attr = attr_names.get(path, cls.__name__)
        parent_ns = (
            namespace + f"/{attr} ({cls.__name__})"
            if namespace
            else f"{attr} ({cls.__name__})"
        )

        op_nodes = []
        node_map: dict[str, str] = {}

        for fx_node in graph.nodes:
            if not _is_interesting_op(fx_node):
                if fx_node.op == "placeholder":
                    node_map[fx_node.name] = "@input"
                continue

            if fx_node.op == "call_module":
                target = str(fx_node.target)
                child_path = f"{path}.{target}"
                if child_path in module_map:
                    node_map[fx_node.name] = _node_id(child_path)
                    continue
                # Synthetic submodule (e.g. act_fn from mirror tracing)
                node_id = _node_id(child_path)
                node_map[fx_node.name] = node_id
                label = target.replace("_", " ").title()
                try:
                    owner = getattr(graph, "owning_module", None)
                    if owner:
                        sub = dict(owner.named_modules()).get(target)
                        if sub:
                            label = type(sub).__name__
                except Exception:
                    pass
                incoming = []
                for arg_node in fx_node.all_input_nodes:
                    src = node_map.get(arg_node.name, "@input")
                    incoming.append({"sourceNodeId": src})
                op_nodes.append(
                    {
                        "id": node_id,
                        "label": label,
                        "namespace": parent_ns,
                        "incomingEdges": incoming,
                    }
                )
                continue

            label = _fx_op_label(fx_node)
            op_id = f"{_node_id(path)}/{fx_node.name}"
            node_map[fx_node.name] = op_id

            incoming = []
            for arg in fx_node.args:
                if isinstance(arg, torch.fx.Node) and arg.name in node_map:
                    incoming.append(
                        {
                            "sourceNodeId": node_map[arg.name],
                            "sourceNodeOutputId": "0",
                            "targetNodeInputId": str(len(incoming)),
                        }
                    )
                elif isinstance(arg, (tuple, list)):
                    for item in arg:
                        if isinstance(item, torch.fx.Node) and item.name in node_map:
                            incoming.append(
                                {
                                    "sourceNodeId": node_map[item.name],
                                    "sourceNodeOutputId": "0",
                                    "targetNodeInputId": str(len(incoming)),
                                }
                            )

            op_node = {
                "id": op_id,
                "label": label,
                "namespace": parent_ns,
                "attrs": [{"key": "operation", "value": "tensor_op"}],
                "style": _STYLE_OP,
            }
            op_node["incomingEdges"] = incoming
            op_nodes.append(op_node)

        # If the module's entire computation is a single primitive op
        # (e.g. Glm5NextTextHyperHead's `hidden_streams.mean(dim=2)`),
        # showing a nested "hc_head (Glm5NextTextHyperHead)" box with its
        # own @input/mean/@output boundary just wraps one op in
        # pointless indirection. Leave it out of fx_graphs entirely so
        # it falls through to the regular leaf-module path below, which
        # already (a) picks the op's own name as the label for a
        # single-op leaf (see _module_label) and (b) uses the module's
        # own hook-captured output shape — correct even for shape-
        # changing ops like `mean`, unlike the op-node path which just
        # inherits the (pre-reduction) input's shape.
        if len(op_nodes) > 1:
            fx_graphs[path] = op_nodes

    # ── Add module nodes ─────────────────────────────────────────────────
    # Emits nodes (plain leaves, group attrs, and FX-expanded op nodes) in
    # a single pass over module_order, so relative ordering between e.g. a
    # composite-with-FX-graph child and a leaf-with-FX-graph sibling stays
    # correct (previously these were emitted in two separate later loops,
    # split by composite-vs-leaf classification rather than true call
    # order — which silently reordered custom leaf modules like a rotary
    # embedding helper to the very end, after every composite's FX ops).
    def _emit_fx_op_nodes(path: str) -> None:
        namespace = _namespace_for(path)
        mod = module_map[path]
        attr = attr_names.get(path, type(mod).__name__)
        parent_ns = (
            namespace + f"/{attr} ({type(mod).__name__})"
            if namespace
            else f"{attr} ({type(mod).__name__})"
        )
        for op_node in fx_graphs[path]:
            if not op_node["namespace"].startswith(parent_ns):
                op_node["namespace"] = parent_ns
            nodes.append(op_node)

    for path in module_order:
        if not path:
            continue
        if _should_skip(path):
            continue

        mod = module_map[path]
        category = _classify_module(mod)
        namespace = _namespace_for(path)
        node_id = _node_id(path)
        label = _module_label(mod)
        style = _style_for(category)

        if path in composite_modules and path not in fx_graphs:
            if path in container_set:
                continue  # ModuleList itself is not a node
            if namespace:
                group_id = (
                    namespace
                    + "/"
                    + f"{attr_names.get(path, '')} ({type(mod).__name__})"
                )
                attrs = {"class": type(mod).__name__}
                shape_str = shapes.get(path)
                if not shape_str:
                    # Derive output_shape from last child module that has a shape
                    for child_name in reversed(list(module_map)):
                        if child_name.startswith(path + ".") and child_name in shapes:
                            shape_str = shapes[child_name]
                            break
                if shape_str:
                    attrs["output_shape"] = f"{shape_str} {dtype}"
                inp_str = input_shapes.get(path)
                if not inp_str:
                    # Derive input_shape from first child module
                    for child_name in module_map:
                        if (
                            child_name.startswith(path + ".")
                            and child_name in input_shapes
                        ):
                            inp_str = input_shapes[child_name]
                            break
                if inp_str:
                    attrs["input_shape"] = f"{inp_str} {dtype}"
                group_attrs[group_id] = attrs
            continue

        if path in composite_modules and path in fx_graphs:
            _emit_fx_op_nodes(path)
            continue

        # Custom leaf module that was FX-expanded — emit its op nodes here
        # (in module_order position) instead of the single leaf node.
        if path in fx_graphs:
            _emit_fx_op_nodes(path)
            continue

        # Skip child modules not referenced by parent's FX graph — the FX
        # ops already cover their computation (e.g. act_fn inlined as silu)
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        if parent in fx_graphs and path not in fx_referenced_children:
            continue

        # Leaf module — add as a node
        attrs = [
            {"key": "class", "value": type(mod).__name__},
            {"key": "attr_name", "value": attr_names.get(path, "")},
        ]
        shape_str = shapes.get(path)
        if shape_str:
            attrs.extend(_shape_attrs(shape_str, dtype))

        node: dict[str, Any] = {
            "id": node_id,
            "label": label,
            "namespace": namespace,
            "attrs": attrs,
            "style": style,
        }
        if shape_str:
            node["outputsMetadata"] = _output_metadata(shape_str, dtype)

        # Apply FX-derived edges if this child is referenced by parent's
        # FX graph — these override the call_graph/sequential wiring.
        if path in fx_child_edges:
            node["incomingEdges"] = fx_child_edges[path]

        nodes.append(node)
        edges_from[path] = node_id

    # ── Build parallel branch info for edge wiring ─────────────────────
    # For containers with multiple layer types, collect the namespace
    # prefixes of each parallel branch so the wiring can fan them out
    # from the same predecessor instead of chaining them sequentially.
    parallel_ns_groups: list[set[str]] = []
    for container_path, groups in layer_group_map.items():
        if len(groups) <= 1:
            continue
        branch_namespaces = set()
        for group in groups:
            rep_path = f"{container_path}.{group.representative}"
            # _namespace_for(rep_path + ".dummy") gives the branch NS
            # because ".dummy" is the leaf (excluded from namespace)
            branch_ns = _namespace_for(rep_path + ".dummy")
            branch_namespaces.add(branch_ns)
        parallel_ns_groups.append(branch_namespaces)

    # ── Create Fork/Join nodes for parallel layer groups ─────────────────
    # When a container has multiple layer types (interleaved), create
    # Fork and Join nodes so all types fan out from Fork and merge at Join.
    fork_join_info: dict[str, dict] = (
        {}
    )  # container_path → {fork_id, join_id, branch_ns}
    for container_path, groups in layer_group_map.items():
        if len(groups) <= 1:
            continue
        parent_path = container_path.rsplit(".", 1)[0] if "." in container_path else ""
        parent_ns = _namespace_for(parent_path + ".dummy") if parent_path else ""
        fork_id = container_path.replace(".", "/") + "/@fork"
        join_id = container_path.replace(".", "/") + "/@join"

        # Determine shape for the fork/join (from first rep's input)
        rep_path = f"{container_path}.{groups[0].representative}"
        fork_shape = input_shapes.get(rep_path, "")
        if not fork_shape:
            for child_name in module_map:
                if child_name.startswith(rep_path + ".") and child_name in input_shapes:
                    fork_shape = input_shapes[child_name]
                    break
        join_shape = shapes.get(rep_path, "")
        if not join_shape:
            for child_name in reversed(list(module_map)):
                if child_name.startswith(rep_path + ".") and child_name in shapes:
                    join_shape = shapes[child_name]
                    break

        fork_node = {
            "id": fork_id,
            "label": "Fork",
            "namespace": parent_ns,
            "attrs": [{"key": "synthetic", "value": "fork"}],
            "style": _STYLE_FORK,
        }
        if fork_shape:
            fork_node["outputsMetadata"] = _output_metadata(fork_shape, dtype)

        join_node = {
            "id": join_id,
            "label": "Join",
            "namespace": parent_ns,
            "attrs": [{"key": "synthetic", "value": "join"}],
            "style": _STYLE_JOIN,
            "incomingEdges": [],  # Will be populated with branch outputs
        }
        if join_shape:
            join_node["outputsMetadata"] = _output_metadata(join_shape, dtype)

        fork_join_info[container_path] = {
            "fork_id": fork_id,
            "join_id": join_id,
            "fork_node": fork_node,
            "join_node": join_node,
            "branch_namespaces": set(),
        }
        # Collect branch namespace prefixes
        for group in groups:
            rep_path = f"{container_path}.{group.representative}"
            branch_ns = _namespace_for(rep_path + ".dummy")
            fork_join_info[container_path]["branch_namespaces"].add(branch_ns)

    # ── Output node ──────────────────────────────────────────────────────
    # Determine output shape from the model's last module
    output_shape = "B x S x V"
    # Check for lm_head or tied embeddings to get vocab size
    if hasattr(model, "lm_head") and isinstance(model.lm_head, torch.nn.Linear):
        output_shape = f"B x S x {model.lm_head.out_features}"
    elif hasattr(model, "language_model") and hasattr(
        model.language_model, "embed_tokens"
    ):
        vocab = model.language_model.embed_tokens.num_embeddings
        output_shape = f"B x S x {vocab}"
    elif hasattr(model, "embed_tokens"):
        vocab = model.embed_tokens.num_embeddings
        output_shape = f"B x S x {vocab}"

    nodes.append(
        {
            "id": "@output",
            "label": "Logits",
            "namespace": "",
            "attrs": [
                {"key": "synthetic", "value": "@output"},
                *_shape_attrs(output_shape, dtype),
            ],
            "style": _STYLE_OUTPUT,
            "outputsMetadata": _output_metadata(output_shape, dtype),
        }
    )

    # ── Wire edges ───────────────────────────────────────────────────────
    # Pre-compute alias mapping for FX-expanded leaf modules:
    # maps original module node_id → last FX op node_id
    _fx_leaf_aliases: dict[str, str] = {}
    _fx_leaf_first_map: dict[str, str] = {}
    for path in fx_graphs:
        if path in composite_modules:
            continue
        prefix = _node_id(path) + "/"
        first_op = last_op = None
        for n in nodes:
            if n["id"].startswith(prefix):
                if first_op is None:
                    first_op = n["id"]
                last_op = n["id"]
        if last_op:
            _fx_leaf_aliases[_node_id(path)] = last_op
        if first_op:
            _fx_leaf_first_map[path] = first_op

    # ── Reorder nodes by call-graph execution order ─────────────────────
    # FX-expanded op nodes may appear after leaf module nodes in the list,
    # but they may execute BEFORE them (e.g. input_layernorm ops must
    # precede self_attn's q_proj in the wiring traversal).  Build an
    # execution-order index from the call_graph, processing composites
    # from shallowest to deepest so top-level order takes precedence.
    _exec_order: dict[str, int] = {}
    _order_counter = 0
    for comp_path in sorted(call_graph.keys(), key=lambda p: p.count(".")):
        # Topologically sort edges so the first child (after @input)
        # gets the lowest execution order, regardless of edge list order.
        cg_edges = call_graph[comp_path]
        # Build adjacency: src → [tgt]
        _adj: dict[str, list[str]] = defaultdict(list)
        _all_nodes: set[str] = set()
        _has_incoming: set[str] = set()
        for src, tgt in cg_edges:
            _adj[src].append(tgt)
            _all_nodes.add(tgt)
            _has_incoming.add(tgt)
            if src != "@input":
                _all_nodes.add(src)
        # Start from nodes with no incoming (or @input targets)
        _roots = [n for n in _all_nodes if n not in _has_incoming]
        # Also add @input targets in edge order
        for src, tgt in cg_edges:
            if src == "@input" and tgt not in _roots:
                _roots.append(tgt)
        # BFS topological order
        _visited: set[str] = set()
        _queue = list(_roots)
        _topo: list[str] = []
        while _queue:
            node = _queue.pop(0)
            if node in _visited:
                continue
            _visited.add(node)
            _topo.append(node)
            for succ in _adj.get(node, []):
                if succ not in _visited:
                    _queue.append(succ)
        # Assign execution order
        for node in _topo:
            if node not in _exec_order:
                _exec_order[node] = _order_counter
                _order_counter += 1

    def _node_exec_key(n: dict) -> tuple:
        """Return a hierarchical sort key using the shallowest ancestor's
        exec_order as the primary key, then deeper ancestors as tie-breakers.
        This ensures that all nodes under ``layers.3`` (top-level order 2)
        sort before ``hc_head`` (top-level order 6), even if layers.3's
        internal children have higher absolute exec_order values."""
        nid = n["id"]
        path = nid.replace("/", ".")
        parts = path.split(".")
        # Collect all ancestor exec_orders from shallowest to deepest
        orders: list[int] = []
        for depth in range(1, len(parts) + 1):
            ancestor = ".".join(parts[:depth])
            if ancestor in _exec_order:
                orders.append(_exec_order[ancestor])
        if orders:
            return tuple(orders)
        return (999999,)

    # Preserve the root @input at position 0 and @output at the end
    root_nodes = [n for n in nodes if n["id"] in ("@input", "@output")]
    inner_nodes = [n for n in nodes if n["id"] not in ("@input", "@output")]
    # Stable sort: preserves relative order of nodes within the same
    # execution group (important for FX op chains within a module).
    inner_nodes.sort(key=_node_exec_key)
    nodes[:] = (
        [n for n in root_nodes if n["id"] == "@input"]
        + inner_nodes
        + [n for n in root_nodes if n["id"] == "@output"]
    )

    # Insert Fork/Join nodes at correct positions in the node list.
    # Fork goes before the first branch node; Join goes after the last.
    for container_path, fj in fork_join_info.items():
        container_prefix = container_path.replace(".", "/") + "/"
        first_branch_idx = None
        last_branch_idx = None
        for i, n in enumerate(nodes):
            if n["id"].startswith(container_prefix):
                if first_branch_idx is None:
                    first_branch_idx = i
                last_branch_idx = i
        if first_branch_idx is not None:
            nodes.insert(first_branch_idx, fj["fork_node"])
            # last_branch_idx shifted by 1 due to insert
            nodes.insert(last_branch_idx + 2, fj["join_node"])

    _wire_sequential_edges(
        nodes,
        model,
        module_map,
        shapes,
        {},
        skip_layers,
        parallel_ns_groups=parallel_ns_groups,
        call_graph=call_graph,
        fx_leaf_aliases=_fx_leaf_aliases,
        fork_join_info=fork_join_info,
    )

    # ── Fix up FX-expanded leaf module wiring ────────────────────────────
    # The _wire_sequential_edges pass uses aliases so downstream modules
    # connect from the original module id (which maps to the last FX op).
    # Now fix the FX ops themselves: resolve @input placeholders to the
    # actual predecessor.
    node_by_id = {n["id"]: n for n in nodes}

    # Also include the aliases so _find_last_node_for works
    for alias_id, target_id in _fx_leaf_aliases.items():
        if alias_id not in node_by_id and target_id in node_by_id:
            node_by_id[alias_id] = node_by_id[target_id]

    fx_leaf_first = _fx_leaf_first_map
    fx_leaf_last = {
        p: _fx_leaf_aliases[_node_id(p)]
        for p in _fx_leaf_first_map
        if _node_id(p) in _fx_leaf_aliases
    }

    # Find predecessor for each expanded leaf module.
    # Walk up the module hierarchy to find call_graph edges that tell us
    # what feeds into each expanded module.
    leaf_sources: dict[str, list[str]] = {}
    for comp_path, edges in call_graph.items():
        for src, tgt in edges:
            if tgt not in leaf_sources:
                leaf_sources[tgt] = []
            leaf_sources[tgt].append(src)

    # For single-child composites (not in call_graph), propagate: if a
    # composite is a target in leaf_sources, its only child inherits
    # the same source.
    for comp_path in sorted(composite_modules):
        if comp_path in call_graph:
            continue  # already handled
        mod = module_map.get(comp_path)
        if not mod:
            continue
        children = list(mod.named_children())
        if len(children) == 1:
            child_path = f"{comp_path}.{children[0][0]}"
            if comp_path in leaf_sources and child_path not in leaf_sources:
                leaf_sources[child_path] = leaf_sources[comp_path]

    def _find_last_node_for(mod_path: str) -> str | None:
        """Find the last node ID belonging to this module or its descendants."""
        # Check FX-expanded leaf
        if mod_path in fx_leaf_last:
            return fx_leaf_last[mod_path]
        # Check direct node
        candidate = mod_path.replace(".", "/")
        if candidate in node_by_id:
            return candidate
        # Check descendants (composite module)
        prefix = mod_path.replace(".", "/") + "/"
        last = None
        for n in nodes:
            if n["id"].startswith(prefix):
                last = n["id"]
        return last

    def _resolve_predecessor(mod_path: str, depth: int = 0) -> str | None:
        """Find the actual node ID that should feed into this module."""
        if depth > 10:
            return None  # prevent infinite recursion
        if mod_path in leaf_sources:
            for src_path in leaf_sources[mod_path]:
                if src_path == "@input":
                    # Composite's own input — resolve from parent
                    parent = mod_path.rsplit(".", 1)[0] if "." in mod_path else ""
                    if parent:
                        return _resolve_predecessor(parent, depth + 1)
                    return None
                else:
                    result = _find_last_node_for(src_path)
                    if result:
                        return result
        # Not in leaf_sources — this module might be a composite whose
        # predecessor is determined by sequential wiring. Check if any
        # node inside this module was wired from something outside.
        prefix = mod_path.replace(".", "/") + "/"
        for n in nodes:
            if n["id"].startswith(prefix) and "incomingEdges" in n:
                for edge in n["incomingEdges"]:
                    src = edge["sourceNodeId"]
                    if (
                        not src.startswith(prefix)
                        and src != "@input"
                        and src in node_by_id
                    ):
                        return src
        return None

    fx_leaf_predecessor: dict[str, str] = {}
    for path in fx_leaf_first:
        pred = _resolve_predecessor(path)
        if pred:
            fx_leaf_predecessor[path] = pred

    # For modules without a resolved predecessor, find the nearest
    # preceding sibling node in the ordered node list. Walk up through
    # successively wider ancestor scopes (immediate parent, grandparent,
    # ... root) rather than stopping at the immediate parent: for an
    # uninvoked composite (no call_graph data at any level), the first
    # child of a nested submodule (e.g. blocks.0.norm1, where norm1 is
    # blocks.0's first child) has no sibling within its immediate parent
    # scope at all — the true predecessor is a "cousin" module that ran
    # just before the immediate parent started (e.g. visual.rotary_pos_emb
    # feeding visual.blocks.0). Stopping at the immediate parent caused
    # such nodes to fall through to the literal "@input" placeholder,
    # which later got misinterpreted as the top-level graph input.
    for path, first_id in fx_leaf_first.items():
        if path in fx_leaf_predecessor:
            continue
        prefix = _node_id(path) + "/"
        path_parts = path.split(".")
        prev_id = None
        for depth in range(len(path_parts) - 1, -1, -1):
            scope_path = ".".join(path_parts[:depth])
            scope_prefix = _node_id(scope_path) + "/" if scope_path else ""
            for n in nodes:
                if n["id"] == first_id:
                    break
                nid = n["id"]
                # Only consider nodes within the current ancestor scope
                if scope_prefix and not nid.startswith(scope_prefix):
                    continue
                if not nid.startswith(prefix):
                    prev_id = nid
            if prev_id:
                break
        if prev_id and prev_id in node_by_id:
            fx_leaf_predecessor[path] = prev_id

    # 1. Resolve @input references in FX ops to actual predecessors
    for path, pred_id in fx_leaf_predecessor.items():
        prefix = _node_id(path) + "/"
        for n in nodes:
            if n["id"].startswith(prefix) and "incomingEdges" in n:
                for edge in n["incomingEdges"]:
                    if edge["sourceNodeId"] == "@input":
                        edge["sourceNodeId"] = pred_id

    # 2. Rewire downstream edges: replace alias references with actual
    #    last FX op node IDs so the viewer can find real nodes.
    for n in nodes:
        if "incomingEdges" in n:
            for edge in n["incomingEdges"]:
                if edge["sourceNodeId"] in _fx_leaf_aliases:
                    edge["sourceNodeId"] = _fx_leaf_aliases[edge["sourceNodeId"]]

    # 3. Wire missing consumers: if an FX-expanded leaf's last op has
    #    no consumer, find what should consume it from the call_graph
    #    and add the edge.
    all_sources_post = set()
    for n in nodes:
        for e in n.get("incomingEdges", []):
            all_sources_post.add(e["sourceNodeId"])

    def _find_entry_nodes(mod_path: str) -> list[dict]:
        """Find entry-point nodes for a module (first children receiving @input)."""
        mod_id = _node_id(mod_path)
        # Direct node?
        if mod_id in node_by_id and mod_id not in _fx_leaf_aliases:
            return [node_by_id[mod_id]]
        # FX-expanded leaf?
        if mod_path in fx_leaf_first:
            n = node_by_id.get(fx_leaf_first[mod_path])
            return [n] if n else []
        # Composite: find children that receive @input from call_graph
        if mod_path in call_graph:
            entries = []
            for src, tgt in call_graph[mod_path]:
                if src == "@input":
                    entries.extend(_find_entry_nodes(tgt))
            if entries:
                return entries
        # Fallback: first node inside this namespace
        prefix = mod_id + "/"
        for n in nodes:
            if n["id"].startswith(prefix):
                return [n]
        return []

    for path, last_id in fx_leaf_last.items():
        if last_id in all_sources_post:
            continue
        # Find downstream from call_graph
        for comp_path, edges in call_graph.items():
            for src, tgt in edges:
                if src == path:
                    for target_node in _find_entry_nodes(tgt):
                        if "incomingEdges" not in target_node:
                            target_node["incomingEdges"] = []
                        target_node["incomingEdges"].append({"sourceNodeId": last_id})
                        all_sources_post.add(last_id)

    # ── Add synthetic I/O nodes for composite modules ────────────────────
    # Rebuild node lookup after all wiring fixups
    node_by_id = {n["id"]: n for n in nodes}

    for comp_path in sorted(composite_modules, key=lambda p: (-p.count("."), p)):
        if _should_skip(comp_path):
            continue

        # Rebuild consumers_of each iteration so child I/O nodes
        # created in prior iterations are visible to parent composites
        consumers_of = {}
        for n in nodes:
            for e in n.get("incomingEdges", []):
                consumers_of.setdefault(e["sourceNodeId"], []).append((n, e))
        # Skip ModuleList/Sequential containers — they are structural,
        # not meaningful submodule boundaries for I/O.
        if comp_path in container_set:
            continue

        mod = module_map.get(comp_path)
        if not mod:
            continue

        comp_id = _node_id(comp_path)
        comp_ns = _namespace_for(comp_path + ".dummy")
        attr = attr_names.get(comp_path, type(mod).__name__)
        cls_name = type(mod).__name__

        # The module's own namespace (where internal nodes go)
        module_ns = (
            comp_ns + f"/{attr} ({cls_name})" if comp_ns else f"{attr} ({cls_name})"
        )

        # Find child node IDs inside this module
        child_prefix = comp_id + "/"
        child_nodes = [n for n in nodes if n["id"].startswith(child_prefix)]
        if not child_nodes:
            continue

        # Determine the namespace where children actually live.
        # For layer-indexed modules (e.g. layers.0), children are in the
        # layer group namespace, not inside module_ns.  Detect this by
        # checking the first child's namespace.
        children_ns = module_ns
        for cn in child_nodes:
            cn_ns = cn.get("namespace", "")
            if cn_ns and not cn_ns.startswith(module_ns):
                # Children are NOT inside module_ns — use comp_ns instead
                children_ns = comp_ns
                break

        # ── Determine input children ──────────────────────────────────
        # Children whose incoming edges come from OUTSIDE this module
        input_child_ids: list[str] = []
        external_sources: dict[str, list[str]] = {}  # child_id → [external_src_ids]
        for cn in child_nodes:
            ext_srcs = []
            for e in cn.get("incomingEdges", []):
                src = e["sourceNodeId"]
                if not src.startswith(child_prefix):
                    ext_srcs.append(src)
            if ext_srcs:
                input_child_ids.append(cn["id"])
                external_sources[cn["id"]] = ext_srcs

        # Deduplicate: unique external source sets define distinct inputs
        unique_ext_src_sets: list[set[str]] = []
        for cid in input_child_ids:
            s = set(external_sources[cid])
            if s not in unique_ext_src_sets:
                unique_ext_src_sets.append(s)

        n_inputs = len(unique_ext_src_sets)

        # ── Determine output children ─────────────────────────────────
        # Prefer FX graph output nodes (ground truth from tracing)
        output_child_ids: list[str] = []
        if comp_path in fx_output_ids:
            for out_id in fx_output_ids[comp_path]:
                if out_id in node_by_id and out_id not in output_child_ids:
                    output_child_ids.append(out_id)

        # Otherwise, find children consumed by nodes OUTSIDE this module
        if not output_child_ids:
            for cn in child_nodes:
                cid = cn["id"]
                if cid in consumers_of:
                    for consumer_node, edge in consumers_of[cid]:
                        # Guard: only count if edge still points to this node
                        # (a parent composite's output rewiring may have changed it)
                        if edge["sourceNodeId"] == cid and not consumer_node[
                            "id"
                        ].startswith(child_prefix):
                            if cid not in output_child_ids:
                                output_child_ids.append(cid)
                            break

            # Also check FX leaf aliases: if an alias maps to a child node
            for alias_id, target_id in _fx_leaf_aliases.items():
                if target_id.startswith(child_prefix) and alias_id in consumers_of:
                    for consumer_node, edge in consumers_of[alias_id]:
                        if edge["sourceNodeId"] == alias_id and not consumer_node[
                            "id"
                        ].startswith(child_prefix):
                            if target_id not in output_child_ids:
                                output_child_ids.append(target_id)
                            break

        # Also include orphan @output nodes — child synthetic output nodes
        # with no consumers (data exits via inline ops not tracked by
        # the call graph).
        for cn in child_nodes:
            cid = cn["id"]
            if cid in output_child_ids:
                continue
            cn_attrs = {a["key"]: a["value"] for a in cn.get("attrs", [])}
            if cn_attrs.get("synthetic") != "output":
                continue
            has_consumer = any(
                e["sourceNodeId"] == cid
                for n2 in nodes
                for e in n2.get("incomingEdges", [])
                if n2["id"] != cid
            )
            if not has_consumer:
                output_child_ids.append(cid)

        n_outputs = len(output_child_ids)

        # Last resort: use child nodes that have no consumers as outputs
        if n_outputs == 0 and child_nodes:
            for cn in reversed(child_nodes):
                cid = cn["id"]
                cn_attrs = {a["key"]: a["value"] for a in cn.get("attrs", [])}
                if cn_attrs.get("synthetic"):
                    continue
                has_consumer = any(
                    e["sourceNodeId"] == cid
                    for n2 in nodes
                    for e in n2.get("incomingEdges", [])
                    if n2["id"] != cid
                )
                if not has_consumer and cid not in output_child_ids:
                    output_child_ids.append(cid)
            n_outputs = len(output_child_ids)

        # Call-graph terminal override: use the terminal nodes from the
        # call graph (nodes that are targets but never sources) as the
        # authoritative output.  This overrides heuristic-based output
        # child detection which can pick intermediate modules when
        # wiring hasn't been finalized yet.
        if comp_path in call_graph:
            targets = {t for _, t in call_graph[comp_path]}
            sources = {s for s, _ in call_graph[comp_path] if s != "@input"}
            terminals = targets - sources
            if terminals:
                cg_output_ids: list[str] = []
                for term_path in terminals:
                    term_id = _node_id(term_path)
                    term_output_id = term_id + "/@output"
                    if term_output_id in node_by_id:
                        cg_output_ids.append(term_output_id)
                    elif term_id in node_by_id:
                        cg_output_ids.append(term_id)
                    else:
                        # Terminal might be an FX-expanded leaf whose
                        # @output doesn't exist yet.  Find the last
                        # FX op node under this module prefix.
                        prefix = term_id + "/"
                        last_fx_node = None
                        for cn in child_nodes:
                            if cn["id"].startswith(prefix):
                                last_fx_node = cn["id"]
                        if last_fx_node:
                            cg_output_ids.append(last_fx_node)
                if cg_output_ids:
                    output_child_ids = cg_output_ids
                    n_outputs = len(output_child_ids)

        # ── Create Input node(s) ──────────────────────────────────────
        if n_inputs >= 1:
            # Always create an internal input node inside the module
            input_id = comp_id + "/@input"
            input_node = {
                "id": input_id,
                "label": "Input",
                "namespace": children_ns,
                "attrs": [{"key": "synthetic", "value": "input"}],
                "style": _STYLE_INPUT,
                "incomingEdges": [],
            }

            # Collect all unique external sources
            all_ext_srcs: list[str] = []
            seen_srcs: set[str] = set()
            for cid in input_child_ids:
                for src in external_sources[cid]:
                    if src not in seen_srcs:
                        seen_srcs.add(src)
                        all_ext_srcs.append(src)

            input_node["incomingEdges"] = [{"sourceNodeId": s} for s in all_ext_srcs]

            nodes.append(input_node)
            node_by_id[input_id] = input_node
            for e in input_node["incomingEdges"]:
                consumers_of.setdefault(e["sourceNodeId"], []).append((input_node, e))

            # Rewire input children: replace external sources with input_id
            for cid in input_child_ids:
                cn = node_by_id.get(cid)
                if cn and "incomingEdges" in cn:
                    for e in cn["incomingEdges"]:
                        if not e["sourceNodeId"].startswith(child_prefix):
                            e["sourceNodeId"] = input_id
                    # Deduplicate edges after rewiring (multiple distinct
                    # external sources may collapse to the same @input)
                    seen_srcs: set[str] = set()
                    deduped: list[dict] = []
                    for e in cn["incomingEdges"]:
                        if e["sourceNodeId"] not in seen_srcs:
                            seen_srcs.add(e["sourceNodeId"])
                            deduped.append(e)
                    cn["incomingEdges"] = deduped

        # ── Create Output node(s) ─────────────────────────────────────
        if n_outputs >= 1:
            # Always create an internal output node inside the module
            output_id = comp_id + "/@output"
            output_node = {
                "id": output_id,
                "label": "Output",
                "namespace": children_ns,
                "attrs": [{"key": "synthetic", "value": "output"}],
                "style": _STYLE_OUTPUT,
                "incomingEdges": [{"sourceNodeId": oid} for oid in output_child_ids],
            }
            nodes.append(output_node)
            node_by_id[output_id] = output_node

            wire_output_id = output_id

            # Rewire consumers: nodes outside this module that consumed
            # ANY child node should now consume the composite's @output.
            # This prevents external nodes from bypassing the composite
            # boundary (e.g. after a child composite's @output was created
            # in a prior iteration and an external node was wired to it).
            for cn in child_nodes:
                cid = cn["id"]
                if cid in consumers_of:
                    for consumer_node, edge in consumers_of[cid]:
                        if (
                            not consumer_node["id"].startswith(child_prefix)
                            and edge["sourceNodeId"] == cid
                        ):
                            edge["sourceNodeId"] = wire_output_id
            # Also check alias consumers
            for alias_id, target_id in _fx_leaf_aliases.items():
                if target_id.startswith(child_prefix) and alias_id in consumers_of:
                    for consumer_node, edge in consumers_of[alias_id]:
                        if (
                            not consumer_node["id"].startswith(child_prefix)
                            and edge["sourceNodeId"] == alias_id
                        ):
                            edge["sourceNodeId"] = wire_output_id

    # ── Add I/O nodes for FX-expanded leaf modules ───────────────────
    # These aren't composite modules, but they have FX ops that need
    # input/output boundary nodes.
    node_by_id = {n["id"]: n for n in nodes}
    consumers_of = {}
    for n in nodes:
        for e in n.get("incomingEdges", []):
            consumers_of.setdefault(e["sourceNodeId"], []).append((n, e))

    for path in fx_graphs:
        if path in composite_modules:
            continue  # already handled above
        if _should_skip(path):
            continue

        prefix = _node_id(path) + "/"
        fx_nodes = [n for n in nodes if n["id"].startswith(prefix)]
        if not fx_nodes:
            continue

        # Find the namespace these FX ops live in
        fx_ns = fx_nodes[0].get("namespace", "")

        # Input node: find FX ops whose sources come from outside
        first_id = fx_nodes[0]["id"]
        ext_srcs: list[str] = []
        seen_ext: set[str] = set()
        for fn in fx_nodes:
            for e in fn.get("incomingEdges", []):
                src = e["sourceNodeId"]
                if not src.startswith(prefix) and src not in seen_ext:
                    seen_ext.add(src)
                    ext_srcs.append(src)

        if ext_srcs:
            input_id = _node_id(path) + "/@input"
            input_node = {
                "id": input_id,
                "label": "Input",
                "namespace": fx_ns,
                "attrs": [{"key": "synthetic", "value": "input"}],
                "style": _STYLE_INPUT,
                "incomingEdges": [{"sourceNodeId": s} for s in ext_srcs],
            }
            nodes.append(input_node)
            node_by_id[input_id] = input_node
            # Rewire FX ops to use input node
            for fn in fx_nodes:
                for e in fn.get("incomingEdges", []):
                    if not e["sourceNodeId"].startswith(prefix):
                        e["sourceNodeId"] = input_id

        # Output node: find the last FX op(s)
        last_id = fx_nodes[-1]["id"]
        # Check if the last op has external consumers
        has_ext_consumer = last_id in consumers_of
        if not has_ext_consumer:
            # Check aliases
            orig_id = _node_id(path)
            has_ext_consumer = orig_id in consumers_of

        output_id = _node_id(path) + "/@output"
        output_node = {
            "id": output_id,
            "label": "Output",
            "namespace": fx_ns,
            "attrs": [{"key": "synthetic", "value": "output"}],
            "style": _STYLE_OUTPUT,
            "incomingEdges": [{"sourceNodeId": last_id}],
        }
        nodes.append(output_node)
        node_by_id[output_id] = output_node

        # Rewire external consumers of last FX op to use output node
        if last_id in consumers_of:
            for consumer_node, edge in consumers_of[last_id]:
                if not consumer_node["id"].startswith(prefix):
                    edge["sourceNodeId"] = output_id
        # Also check alias
        orig_id = _node_id(path)
        if orig_id in consumers_of:
            for consumer_node, edge in consumers_of[orig_id]:
                if not consumer_node["id"].startswith(prefix):
                    edge["sourceNodeId"] = output_id

        # Add group attributes for the FX-expanded module's namespace
        mod = module_map.get(path)
        if mod and fx_ns:
            fx_attrs: dict[str, str] = {"class": type(mod).__name__}
            out_str = shapes.get(path)
            if out_str:
                fx_attrs["output_shape"] = f"{out_str} {dtype}"
            inp_str = input_shapes.get(path)
            if inp_str:
                fx_attrs["input_shape"] = f"{inp_str} {dtype}"
            group_attrs[fx_ns] = fx_attrs

    # ── Add layer group attributes ───────────────────────────────────────
    for container_path, groups in layer_group_map.items():
        total = container_total[container_path]
        rep_path = f"{container_path}.0"
        mod_0 = module_map.get(rep_path)
        cls_name = type(mod_0).__name__ if mod_0 else "Layer"
        ns = (
            _namespace_for(rep_path + ".dummy").rsplit("/", 1)[0]
            if "." in rep_path
            else ""
        )
        group_id = ns if ns else f"{total}x {cls_name}"
        # Summarize layer types
        type_summary = (
            ", ".join(f"{g.count}x {g.class_name}" for g in groups)
            if len(groups) > 1
            else f"{total}x {cls_name}"
        )
        layer_attrs: dict[str, str] = {
            "class": cls_name,
            "count": str(total),
            "layer_types": type_summary,
        }
        # Add shapes from the representative layer
        rep_shape = shapes.get(rep_path)
        if not rep_shape:
            for child_name in reversed(list(module_map)):
                if child_name.startswith(rep_path + ".") and child_name in shapes:
                    rep_shape = shapes[child_name]
                    break
        if rep_shape:
            layer_attrs["output_shape"] = f"{rep_shape} {dtype}"
        rep_inp = input_shapes.get(rep_path)
        if not rep_inp:
            for child_name in module_map:
                if child_name.startswith(rep_path + ".") and child_name in input_shapes:
                    rep_inp = input_shapes[child_name]
                    break
        if rep_inp:
            layer_attrs["input_shape"] = f"{rep_inp} {dtype}"
        group_attrs[group_id] = layer_attrs

        # For the container's parent module (e.g. "language_model"), derive
        # input/output shapes from the call-graph boundary children rather
        # than the representative layer — the parent's I/O may differ
        # (e.g. integer token IDs in, float hidden states out).
        parent_path = container_path.rsplit(".", 1)[0] if "." in container_path else ""
        if parent_path and parent_path in call_graph:
            cg_edges = call_graph[parent_path]
            # Find the first child (after @input) for input shape
            for src, tgt in cg_edges:
                if src == "@input":
                    first_inp = input_shapes.get(tgt)
                    if first_inp:
                        # Determine dtype for the parent's input
                        first_mod = module_map.get(tgt)
                        parent_inp_dtype = dtype
                        if first_mod and isinstance(first_mod, torch.nn.Embedding):
                            parent_inp_dtype = "int64"
                        parent_ns = _namespace_for(parent_path + ".dummy").rsplit(
                            "/", 1
                        )[0]
                        if parent_ns in group_attrs:
                            group_attrs[parent_ns][
                                "input_shape"
                            ] = f"{first_inp} {parent_inp_dtype}"
                    break
            # Find terminal nodes for output shape
            targets = {t for _, t in cg_edges}
            sources = {s for s, _ in cg_edges if s != "@input"}
            term_paths = targets - sources
            for term in term_paths:
                term_out = shapes.get(term)
                if term_out:
                    parent_ns = _namespace_for(parent_path + ".dummy").rsplit("/", 1)[0]
                    if parent_ns in group_attrs:
                        group_attrs[parent_ns]["output_shape"] = f"{term_out} {dtype}"
                    break
        elif parent_path:
            # No call-graph signal at all for this composite (e.g. an
            # optional modality branch — like a vision encoder — whose
            # own forward() never ran during the dummy trace because its
            # inputs were omitted). Fall back to the module's true
            # source-order direct children (see forward_call_order) to
            # find its actual first/last child for I/O shapes, instead of
            # leaving the ModuleList's own representative-layer shape
            # (e.g. a decoder block's output) misattributed as the whole
            # composite's boundary shape.
            parent_ns = _namespace_for(parent_path + ".dummy").rsplit("/", 1)[0]
            if parent_ns in group_attrs:
                direct_children = [
                    c for c in module_order if c.rsplit(".", 1)[0] == parent_path
                ]
                if direct_children:
                    first_child = direct_children[0]
                    first_inp = input_shapes.get(first_child)
                    if not first_inp:
                        for child_name in module_map:
                            if (
                                child_name.startswith(first_child + ".")
                                and child_name in input_shapes
                            ):
                                first_inp = input_shapes[child_name]
                                break
                    if first_inp:
                        first_mod = module_map.get(first_child)
                        parent_inp_dtype = dtype
                        if first_mod and isinstance(first_mod, torch.nn.Embedding):
                            parent_inp_dtype = "int64"
                        group_attrs[parent_ns][
                            "input_shape"
                        ] = f"{first_inp} {parent_inp_dtype}"

                    last_child = direct_children[-1]
                    last_out = shapes.get(last_child)
                    if not last_out:
                        for child_name in reversed(list(module_map)):
                            if (
                                child_name.startswith(last_child + ".")
                                and child_name in shapes
                            ):
                                last_out = shapes[child_name]
                                break
                    if last_out:
                        group_attrs[parent_ns]["output_shape"] = f"{last_out} {dtype}"

    # ── Build fact sheet ─────────────────────────────────────────────────
    fact_sheet = _build_fact_sheet(model_name, config)

    # Add layer type breakdown
    for container_path, groups in layer_group_map.items():
        if len(groups) > 1:
            fact_sheet += "\n\n<b>Layer Types</b>"
            for group in groups:
                rep = module_map.get(f"{container_path}.{group.representative}")
                if rep:
                    attn_type = ""
                    for cname, cmod in rep.named_children():
                        if "attn" in cname:
                            attn_type = type(cmod).__name__
                            break
                    mlp_type = ""
                    for cname, cmod in rep.named_children():
                        if "mlp" in cname:
                            mlp_type = type(cmod).__name__
                            break
                    desc = f"{attn_type}"
                    if mlp_type:
                        desc += f" + {mlp_type}"
                    fact_sheet += f"\n  {group.count}x {group.class_name}: {desc}"
                    fact_sheet += f"\n    layers: {group.indices}"

    # ── Final edge deduplication ────────────────────────────────────────
    for n in nodes:
        edges = n.get("incomingEdges")
        if edges and len(edges) > 1:
            seen_e: set[str] = set()
            deduped_e: list[dict] = []
            for e in edges:
                sid = e["sourceNodeId"]
                if sid not in seen_e:
                    seen_e.add(sid)
                    deduped_e.append(e)
            if len(deduped_e) < len(edges):
                n["incomingEdges"] = deduped_e

    # ── Propagate outputsMetadata to all nodes ──────────────────────────
    # Every node needs shape metadata so edges don't display as "?" in
    # the viewer.
    #
    # Strategy:
    # 1. Try inheriting from the source node's outputsMetadata.
    # 2. Fall back to the enclosing module's captured shape (from `shapes`).
    node_by_id = {n["id"]: n for n in nodes}

    def _shape_metadata_for_path(node_id: str) -> list[dict] | None:
        """Derive outputsMetadata from the shapes dict for a node's module."""
        # Strip synthetic suffixes to get the module path
        path = node_id.replace("/", ".")
        for suffix in (".@input", ".@output"):
            path = path.removesuffix(suffix)
        # Walk up the module hierarchy until we find a captured shape
        check = path
        while check:
            shape_str = shapes.get(check)
            if shape_str:
                return _output_metadata(shape_str, dtype)
            if "." in check:
                check = check.rsplit(".", 1)[0]
            else:
                break
        return None

    # For synthetic @input nodes, use the module's captured input_shape
    # instead of inheriting from the source (which may be a different
    # type, e.g. int64 tokens vs. float embeddings).
    for n in nodes:
        attrs = {a["key"]: a["value"] for a in n.get("attrs", [])}
        if attrs.get("synthetic") != "input" or n.get("outputsMetadata"):
            continue
        # Derive input shape from the module's captured input_shapes
        path = n["id"].replace("/", ".").removesuffix(".@input")
        inp_str = input_shapes.get(path)
        if not inp_str:
            # Try first child's input shape
            for child_name in module_map:
                if child_name.startswith(path + ".") and child_name in input_shapes:
                    inp_str = input_shapes[child_name]
                    break
        if inp_str:
            # Determine dtype: if the module's first child is an Embedding,
            # the input is integer token IDs, not floating-point tensors.
            inp_dtype = dtype
            mod = module_map.get(path)
            if mod is not None:
                for _child in mod.children():
                    if isinstance(_child, torch.nn.Embedding):
                        inp_dtype = "int64"
                    break
            n["outputsMetadata"] = _output_metadata(inp_str, inp_dtype)

    # For synthetic @output nodes, use the module's captured output shape
    # instead of inheriting from the last child (which may be an
    # intermediate operation, not the module's actual output).
    for n in nodes:
        attrs = {a["key"]: a["value"] for a in n.get("attrs", [])}
        if attrs.get("synthetic") != "output":
            continue
        path = n["id"].replace("/", ".").removesuffix(".@output")
        out_str = shapes.get(path)
        if out_str:
            n["outputsMetadata"] = _output_metadata(out_str, dtype)

    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n.get("outputsMetadata"):
                continue
            # Try inheriting from source node
            for e in n.get("incomingEdges", []):
                src = node_by_id.get(e["sourceNodeId"])
                if src and src.get("outputsMetadata"):
                    n["outputsMetadata"] = src["outputsMetadata"]
                    changed = True
                    break
            if n.get("outputsMetadata"):
                continue
            # Fall back to enclosing module's captured shape
            meta = _shape_metadata_for_path(n["id"])
            if meta:
                n["outputsMetadata"] = meta
                changed = True

    # ── Final node ordering pass ──────────────────────────────────────
    # The exec-order sort above ran before composite @input/@output nodes
    # (and some other synthetic nodes) were appended to `nodes`; those
    # later appends iterate `sorted(composite_modules)` alphabetically,
    # which can misorder top-level siblings whose real dataflow order
    # differs from alphabetical (e.g. "language_model" sorts before
    # "visual" alphabetically, but visual actually executes first and
    # feeds into language_model). Re-apply the same hierarchical
    # exec-order key now that every node has been created, so the final
    # list order matches true execution order throughout.
    _root_io = {n["id"] for n in nodes if n["id"] in ("@input", "@output")}
    _final_inner = [n for n in nodes if n["id"] not in _root_io]
    _final_inner.sort(key=_node_exec_key)
    nodes[:] = (
        [n for n in nodes if n["id"] == "@input"]
        + _final_inner
        + [n for n in nodes if n["id"] == "@output"]
    )

    return {
        "name": model_name,
        "model_type": model_type,
        "source": "tracelens-torch-trace",
        "tracelensViewer": {
            "factSheet": {
                "title": model_name,
                "body": fact_sheet,
                "bodyHtml": fact_sheet.replace("  ", "&nbsp;&nbsp;").replace(
                    "\n", "<br>\n"
                ),
            },
            "dtype": dtype,
        },
        "graphCollections": [
            {
                "label": model_name,
                "graphs": [
                    {
                        "id": "model",
                        "nodes": nodes,
                        "groupNodeAttributes": _reorder_group_attrs(group_attrs),
                        "groupNodeConfigs": group_configs,
                    }
                ],
            }
        ],
    }


def _wire_sequential_edges(
    nodes: list[dict],
    model: torch.nn.Module,
    module_map: dict[str, torch.nn.Module],
    shapes: dict[str, str],
    layer_containers: dict,
    skip_layers: set[str],
    *,
    parallel_ns_groups: list[set[str]] | None = None,
    call_graph: dict[str, list[tuple[str, str]]] | None = None,
    fx_leaf_aliases: dict[str, str] | None = None,
    fork_join_info: dict[str, dict] | None = None,
) -> None:
    """Wire edges between nodes that don't already have incoming edges.

    Uses call_graph (captured during forward pass) to wire dataflow-aware
    edges between child modules of composite modules, falling back to
    sequential wiring when call_graph data is unavailable.

    For parallel branches (multiple layer types in the same container),
    all branches fan out from the same predecessor instead of being
    chained sequentially.
    chained sequentially.
    """
    node_ids = {n["id"] for n in nodes}
    node_by_id: dict[str, dict] = {n["id"]: n for n in nodes}
    parallel_ns_groups = parallel_ns_groups or []
    call_graph = call_graph or {}
    fx_leaf_aliases = fx_leaf_aliases or {}
    fork_join_info = fork_join_info or {}

    # Build lookup: branch_ns → fork_id, join_id
    _branch_to_fork: dict[str, str] = {}  # branch_ns → fork_id
    _branch_to_join: dict[str, str] = {}  # branch_ns → join_id
    _fork_ids: set[str] = set()
    _join_ids: set[str] = set()
    for _cp, fj in fork_join_info.items():
        _fork_ids.add(fj["fork_id"])
        _join_ids.add(fj["join_id"])
        for bns in fj["branch_namespaces"]:
            _branch_to_fork[bns] = fj["fork_id"]
            _branch_to_join[bns] = fj["join_id"]

    # Add FX-expanded leaf module aliases so _cg_find_sources can resolve
    # them.  The alias maps orig_module_id → last_fx_op_id.
    for alias_id, target_id in fx_leaf_aliases.items():
        if alias_id not in node_by_id and target_id in node_by_id:
            node_ids.add(alias_id)
            node_by_id[alias_id] = node_by_id[target_id]

    # Build call-graph predecessor lookup at the node-ID level.
    # call_graph has paths like "a.b.c" → [("a.b.c.child1", "a.b.c.child2")].
    # We need to map node IDs (which use "/" and may be deeper) to their
    # dataflow predecessors.
    #
    # Strategy: for each composite's child-level edge (src_child → tgt_child),
    # find all nodes that belong to tgt_child (or are tgt_child itself),
    # and that are the FIRST node in tgt_child's subtree. Those should
    # have edges from the LAST node in src_child's subtree.
    #
    # We'll build this as: node_id → set of source child paths, and resolve
    # actual source node IDs during wiring.

    # Map: dotted child path → node ID prefix (using "/")
    def _path_to_id_prefix(dotted: str) -> str:
        return dotted.replace(".", "/")

    # For each composite, build set of children that have dataflow sources
    # (children that appear as targets in call_graph edges)
    cg_sources: dict[str, list[str]] = {}  # child_path → [source_child_paths]
    cg_children_with_no_sources: dict[str, set[str]] = (
        {}
    )  # comp → children with no incoming
    for comp_path, edges in call_graph.items():
        targets_seen: dict[str, list[str]] = defaultdict(list)
        all_children: set[str] = set()
        for src, tgt in edges:
            targets_seen[tgt].append(src)
            all_children.add(src)
            all_children.add(tgt)
        for tgt, srcs in targets_seen.items():
            cg_sources[tgt] = srcs
        # Children that only produce (never consume from a sibling)
        cg_children_with_no_sources[comp_path] = all_children - set(targets_seen.keys())

    # Build a lookup: namespace → set of sibling namespaces in parallel
    parallel_siblings: dict[str, set[str]] = {}
    for group in parallel_ns_groups:
        for ns in group:
            parallel_siblings[ns] = group

    # Track the last node seen at each namespace level.
    last_in_ns: dict[str, str] = {}

    # For parallel branches, snapshot the predecessor before the first
    # branch starts so all branches can connect from it.
    parallel_entry_point: dict[str, str] = {}  # branch_ns → predecessor_id
    parallel_entered: set[str] = set()  # branch namespaces we've seen

    def _update_last(ns: str, node_id: str, *, bubble: bool = True) -> None:
        if not bubble:
            last_in_ns[ns] = node_id
            return
        parts = ns.split("/") if ns else []
        while True:
            key = "/".join(parts)
            # Don't bubble up past a parallel branch boundary into sibling
            # territory — otherwise later parallel branches would see the
            # last node of a prior branch as their predecessor.
            if key in parallel_siblings and key != ns:
                # Only update this level, don't bubble further
                last_in_ns[key] = node_id
                break
            last_in_ns[key] = node_id
            if not parts:
                break
            parts.pop()

    def _find_branch_ns(ns: str) -> str | None:
        """Check if this namespace is inside a parallel branch."""
        for branch_ns in parallel_siblings:
            if ns == branch_ns or ns.startswith(branch_ns + "/"):
                return branch_ns
        return None

    # Track last node of each parallel branch for fan-in after branches
    branch_last_node: dict[str, str] = {}  # branch_ns → last node id

    # Track last node emitted per dotted module path (for call-graph wiring)
    last_node_for_path: dict[str, str] = {}
    # For @input call-graph sources, track the entry point of each composite.
    # Pre-compute from the call_graph: for each composite C, find its
    # predecessor in the parent's call_graph.  The @input of C resolves
    # to the last node of that predecessor.
    composite_entry: dict[str, str] = {}  # comp_path → predecessor node ID
    # Pre-populate from call_graph: if parent says "X → C", then C's
    # @input should resolve to the last node of X (set lazily below).
    _cg_predecessor: dict[str, list[str]] = {}
    _cg_has_root_input: set[str] = set()  # targets with @input source from root
    for comp_path, edges in call_graph.items():
        for src, tgt in edges:
            if src != "@input":
                if tgt not in _cg_predecessor:
                    _cg_predecessor[tgt] = []
                if src not in _cg_predecessor[tgt]:
                    _cg_predecessor[tgt].append(src)
            if src == "@input" and comp_path == "":
                _cg_has_root_input.add(tgt)

    def _node_id_to_path(nid: str) -> str:
        return nid.replace("/", ".")

    def _cg_find_sources(node_id: str, fallback_source: str | None) -> list[str] | None:
        """Check if this node is the first node of a child module that has
        call-graph predecessors. Returns list of source node IDs or None.

        ``fallback_source`` is the sequential predecessor — used when a
        call-graph edge points to "@input" (composite's own input).
        """
        path = _node_id_to_path(node_id)
        # Walk up the path to find a child that has call-graph sources
        parts = path.split(".")
        for depth in range(len(parts), 0, -1):
            child_path = ".".join(parts[:depth])
            if child_path in cg_sources:
                # Only apply to the FIRST node in this child's subtree
                if child_path in last_node_for_path:
                    return None  # already wired a node in this child

                # Determine the @input fallback: prefer the call-graph
                # predecessor's last node, then the saved composite entry
                # point, then the sequential fallback.
                comp_path = child_path.rsplit(".", 1)[0]
                input_sources: list[str] = []
                # Check if this composite has known predecessors from
                # the parent's call_graph
                if comp_path in _cg_has_root_input:
                    input_sources.append("@input")
                if comp_path in _cg_predecessor:
                    for pred_path in _cg_predecessor[comp_path]:
                        if pred_path in last_node_for_path:
                            input_sources.append(last_node_for_path[pred_path])
                input_source = input_sources[0] if input_sources else None
                if not input_source:
                    input_source = composite_entry.get(comp_path, fallback_source)

                # Find the last node of each source child
                src_ids = []
                for src_path in cg_sources[child_path]:
                    if src_path == "@input":
                        if input_sources:
                            src_ids.extend(input_sources)
                        elif input_source:
                            src_ids.append(input_source)
                    else:
                        if src_path in last_node_for_path:
                            src_ids.append(last_node_for_path[src_path])
                # Deduplicate while preserving order
                seen: set[str] = set()
                unique = []
                for s in src_ids:
                    if s not in seen:
                        seen.add(s)
                        unique.append(s)
                return unique if unique else None
        return None

    def _record_composite_entry(node_id: str, source_id: str | None) -> None:
        """Record the entry point for composites containing this node."""
        if not source_id:
            return
        path = _node_id_to_path(node_id)
        parts = path.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            comp_path = ".".join(parts[:depth])
            if comp_path in call_graph and comp_path not in composite_entry:
                composite_entry[comp_path] = source_id

    for node in nodes:
        if node["id"] == "@input":
            _update_last("", "@input")
            continue

        # Handle Fork nodes: wire from current predecessor in parent ns
        if node["id"] in _fork_ids:
            ns = node.get("namespace", "")
            source_id = None
            parts = ns.split("/") if ns else []
            while parts:
                parts.pop()
                parent = "/".join(parts)
                if parent in last_in_ns:
                    source_id = last_in_ns[parent]
                    break
            if source_id is None:
                source_id = last_in_ns.get("", "@input")
            if source_id and source_id in node_by_id:
                node["incomingEdges"] = [
                    {
                        "sourceNodeId": source_id,
                        "sourceNodeOutputId": "0",
                        "targetNodeInputId": "0",
                    }
                ]
            fork_path = _node_id_to_path(node["id"])
            last_node_for_path[fork_path] = node["id"]
            _update_last(ns, node["id"])
            continue

        # Handle Join nodes: wire from all branch last nodes
        if node["id"] in _join_ids:
            ns = node.get("namespace", "")
            fan_in = []
            for _cp, fj in fork_join_info.items():
                if fj["join_id"] == node["id"]:
                    for bns in fj["branch_namespaces"]:
                        if bns in branch_last_node:
                            fan_in.append(branch_last_node[bns])
                    break
            if fan_in:
                node["incomingEdges"] = [
                    {
                        "sourceNodeId": src,
                        "sourceNodeOutputId": "0",
                        "targetNodeInputId": str(i),
                    }
                    for i, src in enumerate(fan_in)
                    if src in node_by_id
                ]
            join_path = _node_id_to_path(node["id"])
            last_node_for_path[join_path] = node["id"]
            _update_last(ns, node["id"])
            continue

        if "incomingEdges" in node:
            ns = node.get("namespace", "")
            branch_ns = _find_branch_ns(ns)
            if branch_ns is not None:
                branch_last_node[branch_ns] = node["id"]
            # Track for call-graph
            path = _node_id_to_path(node["id"])
            parts = path.split(".")
            for depth in range(1, len(parts) + 1):
                last_node_for_path[".".join(parts[:depth])] = node["id"]
            _update_last(ns, node["id"])
            continue

        ns = node.get("namespace", "")
        source_id = None

        # Check if this node is in a parallel branch
        branch_ns = _find_branch_ns(ns)

        if branch_ns is not None and branch_ns not in parallel_entered:
            # First time entering this parallel branch.
            # If a Fork node exists for this branch, use it as the source.
            if branch_ns in _branch_to_fork:
                source_id = _branch_to_fork[branch_ns]
                for s in parallel_siblings.get(branch_ns, set()):
                    parallel_entry_point[s] = source_id
            elif not any(
                s in parallel_entered for s in parallel_siblings.get(branch_ns, set())
            ):
                # First branch in the group — find predecessor normally
                parts = ns.split("/") if ns else []
                while parts:
                    parts.pop()
                    parent = "/".join(parts)
                    if parent in last_in_ns:
                        source_id = last_in_ns[parent]
                        break
                if source_id is None:
                    source_id = last_in_ns.get("", "@input")
                # Save this as the entry point for all sibling branches
                for s in parallel_siblings.get(branch_ns, set()):
                    parallel_entry_point[s] = source_id
            else:
                # Subsequent branch — use the saved entry point
                source_id = parallel_entry_point.get(branch_ns)
            parallel_entered.add(branch_ns)
        elif branch_ns is not None and ns in last_in_ns:
            # Inside an already-entered parallel branch — wire sequentially
            source_id = last_in_ns[ns]
        elif branch_ns is not None:
            # Inside an already-entered branch, but first node in a new
            # sub-namespace — walk up to find predecessor within the branch
            parts = ns.split("/") if ns else []
            while parts:
                parts.pop()
                parent = "/".join(parts)
                if parent in last_in_ns:
                    source_id = last_in_ns[parent]
                    break
            if source_id is None:
                source_id = last_in_ns.get("", "@input")
        else:
            # Not in a parallel branch. Check if we just left one —
            # if so, fan in from all branches (or from the Join node
            # if Fork/Join nodes exist).
            fan_in_sources = []
            has_join = False
            for group in parallel_ns_groups:
                if group <= parallel_entered:
                    # Check if any branch in this group has a Join node
                    for sibling_ns in group:
                        if sibling_ns in _branch_to_join:
                            join_id = _branch_to_join[sibling_ns]
                            if join_id in node_by_id:
                                fan_in_sources = [join_id]
                                has_join = True
                            break
                    if not has_join:
                        for sibling_ns in group:
                            if sibling_ns in branch_last_node:
                                fan_in_sources.append(branch_last_node[sibling_ns])
            if fan_in_sources:
                node["incomingEdges"] = [
                    {
                        "sourceNodeId": src,
                        "sourceNodeOutputId": "0",
                        "targetNodeInputId": "0",
                    }
                    for src in fan_in_sources
                    if src in node_by_id
                ]
                # Clear the groups so we don't fan-in again for next node
                for group in parallel_ns_groups:
                    if group <= parallel_entered:
                        for sibling_ns in group:
                            branch_last_node.pop(sibling_ns, None)
                _update_last(ns, node["id"])
                continue

            if ns in last_in_ns:
                source_id = last_in_ns[ns]
            else:
                # First node in this namespace — walk up
                parts = ns.split("/") if ns else []
                while parts:
                    parts.pop()
                    parent = "/".join(parts)
                    if parent in last_in_ns:
                        source_id = last_in_ns[parent]
                        break
                if source_id is None:
                    source_id = last_in_ns.get("", "@input")

        if branch_ns is not None:
            branch_last_node[branch_ns] = node["id"]

        # ── Call-graph-aware wiring: override source_id if we have
        #    dataflow information for this node's parent composite ────────
        _record_composite_entry(node["id"], source_id)
        cg_srcs = _cg_find_sources(node["id"], source_id)
        if cg_srcs is not None:
            if len(cg_srcs) == 1:
                source_id = cg_srcs[0]
            elif len(cg_srcs) > 1:
                # Multiple dataflow predecessors
                node["incomingEdges"] = [
                    {
                        "sourceNodeId": src,
                        "sourceNodeOutputId": "0",
                        "targetNodeInputId": str(i),
                    }
                    for i, src in enumerate(cg_srcs)
                    if src in node_by_id
                ]
                # Update tracking and continue
                path = _node_id_to_path(node["id"])
                parts = path.split(".")
                for depth in range(1, len(parts) + 1):
                    last_node_for_path[".".join(parts[:depth])] = node["id"]
                _update_last(ns, node["id"])
                continue

        if source_id and source_id in node_by_id:
            node["incomingEdges"] = [
                {
                    "sourceNodeId": source_id,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            ]

        # Update last_node_for_path for call-graph tracking
        path = _node_id_to_path(node["id"])
        parts = path.split(".")
        for depth in range(1, len(parts) + 1):
            last_node_for_path[".".join(parts[:depth])] = node["id"]

        _update_last(ns, node["id"])
