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
from datetime import datetime
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
_STYLE_EMBEDDING = {"backgroundColor": "#27ae60", "textColor": _WHITE_TEXT}
_STYLE_LINEAR = {"backgroundColor": "#bdc3c7", "textColor": _DARK_TEXT}
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


# ── Synthetic attention-kernel node ──────────────────────────────────────────

# Attribute name used to attach a synthetic ``_AttentionKernel`` child to a
# real attention module the first time it's observed calling through
# transformers' attention dispatch. Deliberately unlikely to collide with a
# real submodule name.
_ATTN_KERNEL_ATTR = "_tracelens_attention_kernel"

_ATTN_IMPL_LABELS = {
    "sdpa": "SDPA",
    "eager": "Eager Attention",
    "flash_attention_2": "Flash Attention",
    "flash_attention_3": "Flash Attention",
    "flash_attention_4": "Flash Attention",
    "flex_attention": "Flex Attention",
}


class _AttentionKernel(torch.nn.Module):
    """Synthetic placeholder for an attention computation dispatched via a
    raw function call (e.g. transformers' ``ALL_ATTENTION_FUNCTIONS``
    sdpa/eager/flash dispatch, ``attention_interface(module, q, k, v,
    mask, ...)``) rather than through a real ``nn.Module`` child — and
    therefore otherwise entirely invisible to hook-based tracing.

    Without this, Q/K/V projection outputs that only ever feed into that
    raw call look like dead ends (nothing downstream ever appears to
    consume them), while whatever *unrelated* module happens to run next
    (e.g. an indexer with no real data dependency at all) gets a
    spurious "this is my input" edge from the positional sequential
    fallback instead — silently drawing a wrong, misleading dataflow
    edge and completely hiding the real attention computation.

    Instances are attached lazily (the first time the wrapped
    ``attention_interface`` call is observed for a given module) via
    ``setattr(module, _ATTN_KERNEL_ATTR, kernel)``, making them a genuine
    submodule that ``named_modules()``/hook registration naturally picks
    up from that point on — no special-casing needed anywhere else in
    the graph-building pipeline.
    """

    def __init__(self, impl_name: str = "") -> None:
        super().__init__()
        self._impl_name = impl_name


def _attention_kernel_label(impl_name: str) -> str:
    return _ATTN_IMPL_LABELS.get(impl_name, "Attention")


def _patch_attention_interface(get_or_create_kernel):
    """Monkeypatch ``transformers.modeling_utils.AttentionInterface`` (the
    ``ALL_ATTENTION_FUNCTIONS`` dispatch table almost all modern HF
    attention modules use) so every call through it is redirected
    through ``get_or_create_kernel(module, real_fn) -> callable``, which
    should return a callable taking the same ``(*args, **kwargs)`` and
    returning the same result as ``real_fn(module, *args, **kwargs)``,
    but wired up so the call becomes observable via hooks (see
    ``_AttentionKernel``).

    Returns a zero-arg ``restore()`` callback; safe to call even if
    ``transformers`` doesn't expose ``AttentionInterface`` (older/newer
    versions), in which case this is a no-op.
    """
    try:
        from transformers.modeling_utils import AttentionInterface
    except ImportError:
        return lambda: None

    orig_get_interface = AttentionInterface.get_interface

    def patched_get_interface(self, attn_implementation, default):
        real_fn = orig_get_interface(self, attn_implementation, default)

        def wrapped(module, *args, **kwargs):
            kernel_call = get_or_create_kernel(module, real_fn)
            if kernel_call is None:
                return real_fn(module, *args, **kwargs)
            return kernel_call(*args, **kwargs)

        return wrapped

    AttentionInterface.get_interface = patched_get_interface

    def restore() -> None:
        AttentionInterface.get_interface = orig_get_interface

    return restore


# ── Shape capture via forward hooks ──────────────────────────────────────────


def _capture_shapes(
    model: torch.nn.Module,
    *,
    seq_len: int = 128,
    batch_size: int = 1,
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]], dict[str, list[tuple[int, ...]]]]:
    """Run a meta-device forward pass and capture per-module I/O shapes.

    Returns (output_shapes, input_shapes, all_output_shapes) dicts keyed by
    module path. ``all_output_shapes`` additionally lists EVERY tensor shape
    found in a tuple/list return value (not just the first) — needed to
    disambiguate composites like a HyperConnection wrapper that return
    several tensors of genuinely different shapes (e.g. gating weights
    alongside the actual hidden-state that continues downstream), where
    naively picking "the first tensor" as *the* output shape can pick the
    wrong element.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    input_shapes: dict[str, tuple[int, ...]] = {}
    all_output_shapes: dict[str, list[tuple[int, ...]]] = {}

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
                    all_output_shapes[name] = [tuple(output.shape)]
                elif isinstance(output, (tuple, list)):
                    found: list[tuple[int, ...]] = []
                    for item in output:
                        if isinstance(item, torch.Tensor):
                            found.append(tuple(item.shape))
                    if found:
                        shapes[name] = found[0]
                        all_output_shapes[name] = found
            except Exception:
                pass

        return hook

    handles = []
    for name, mod in model.named_modules():
        handles.append(mod.register_forward_hook(_make_hook(name)))

    path_by_id = {id(m): n for n, m in model.named_modules() if n}
    # Kernels persist as real submodules across repeated calls to this
    # function (e.g. the two-probe shape disambiguation in `build_graph`
    # re-runs `_capture_shapes` on the same `model` instance), but hook
    # *handles* don't — they're removed in this function's own `finally`
    # block each time. Track which kernels already have a hook attached
    # for *this* call so a reused kernel still gets (re-)hooked, without
    # double-registering a hook within a single call.
    _hooked_kernel_ids: set[int] = set()

    def _get_or_create_kernel(module: torch.nn.Module, real_fn):
        path = path_by_id.get(id(module))
        if path is None:
            return None
        kernel = getattr(module, _ATTN_KERNEL_ATTR, None)
        if kernel is None:
            impl_name = getattr(getattr(module, "config", None), "_attn_implementation", "") or ""
            kernel = _AttentionKernel(impl_name)
            setattr(module, _ATTN_KERNEL_ATTR, kernel)
        kernel.forward = lambda *a, **kw: real_fn(module, *a, **kw)
        if id(kernel) not in _hooked_kernel_ids:
            _hooked_kernel_ids.add(id(kernel))
            child_path = f"{path}.{_ATTN_KERNEL_ATTR}"
            handles.append(kernel.register_forward_hook(_make_hook(child_path)))
        return kernel

    restore_attention_interface = _patch_attention_interface(_get_or_create_kernel)

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
        restore_attention_interface()
        if original_dtype and original_dtype != torch.bfloat16:
            model.to(original_dtype)
        for h in handles:
            h.remove()

    return shapes, input_shapes, all_output_shapes


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
        elif isinstance(mod, torch.nn.Conv1d):
            shapes[name] = (batch_size, mod.in_channels, seq_len)
        elif isinstance(mod, torch.nn.Conv3d):
            shapes[name] = (batch_size, mod.in_channels, seq_len, seq_len, seq_len)
        elif hasattr(mod, "weight") and hasattr(mod.weight, "shape"):
            # Generic fallback (e.g. custom norm classes that don't
            # subclass torch.nn.LayerNorm/RMSNorm, such as vision-encoder
            # RMSNorm variants): these are almost always shape-preserving,
            # so mirror the same heuristic used for their output shape.
            w_shape = mod.weight.shape
            if len(w_shape) >= 1:
                shapes[name] = (batch_size, seq_len, w_shape[0])
        # For norm-like modules, try to infer from the hidden_size attr
        elif hasattr(mod, "hidden_size"):
            shapes[name] = (batch_size, seq_len, mod.hidden_size)
    return shapes


# ── Call-graph capture via forward hooks ─────────────────────────────────────


def _tensor_ids(x: Any, keepalive: list[torch.Tensor] | None = None) -> list[int]:
    """Extract Python id()s of all tensors in a nested structure.

    ``id()`` is only a valid identity key for as long as the object is
    alive — CPython aggressively reuses the memory address (and hence
    the same ``id()``) of a garbage-collected object for the very next
    allocation. A deep model's forward pass creates and immediately
    drops a huge number of intermediate tensors (e.g. args to a
    function call that aren't stored anywhere else), so without holding
    a reference open, a *later, completely unrelated* tensor can easily
    get assigned the exact same ``id()`` as an earlier one — silently
    corrupting the call-graph's tensor-identity tracking with bogus
    "matches". Pass ``keepalive`` (a list held open for the whole trace)
    to pin every tensor we record an id for, so ids stay unique for the
    duration of the capture.
    """
    if isinstance(x, torch.Tensor):
        if keepalive is not None:
            keepalive.append(x)
        ids = [id(x)]
        # Also register the tensor's *view root* (``._base``), if any,
        # under the same alias set. ``.view()``/``.transpose()``/
        # ``.squeeze()``/etc. all return a NEW tensor object (different
        # ``id()``) that shares the same underlying storage as the
        # original — e.g. MLA attention's
        # ``self.q_b_proj(q_resid).view(...).transpose(1, 2)`` produces
        # a `query_states` tensor whose ``id()`` differs from
        # ``q_b_proj``'s raw output, even though it's really "the same
        # tensor" for dataflow purposes. Without this, a raw function
        # call downstream (e.g. an attention kernel) that only ever sees
        # the *reshaped* tensor could never be linked back to the real
        # producer via pure ``id()`` matching, making the producer look
        # like a dead end. PyTorch tracks this lineage for us (even on
        # meta tensors) via ``._base``, which always points straight to
        # the root non-view tensor regardless of how many views were
        # chained in between.
        base = getattr(x, "_base", None)
        if base is not None:
            if keepalive is not None:
                keepalive.append(base)
            ids.append(id(base))
        return ids
    if isinstance(x, (tuple, list)):
        ids = []
        for item in x:
            ids.extend(_tensor_ids(item, keepalive))
        return ids
    if isinstance(x, dict):
        ids = []
        for v in x.values():
            ids.extend(_tensor_ids(v, keepalive))
        return ids
    return []


def _capture_call_graph(
    model: torch.nn.Module,
    composite_modules: set[str],
    *,
    seq_len: int = 128,
    batch_size: int = 1,
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, set[str]]]:
    """Capture dataflow edges between child modules of each composite.

    Runs a forward pass, tracking which tensor objects flow between
    children of each composite module. Returns a dict mapping composite
    module path → list of (source_child, target_child) edges, plus a
    second dict mapping composite path → set of child paths whose
    output tensor is literally part of the composite's own returned
    output (ground truth, used to disambiguate the *real* output child
    when a composite has multiple call-graph "terminals" — e.g. when a
    raw function call, such as an attention kernel, isn't captured by
    module hooks and leaves some children's outputs as untracked
    dead-ends alongside the genuine output).

    Children that receive the composite's own input (not a sibling's
    output) are marked as receiving from a virtual "@input" source.
    """
    # For each module, record ALL calls (modules may be called multiple
    # times, e.g. HyperConnections).  Each entry is a list of per-call
    # records: (call_index, tensor_ids).
    pre_inputs: dict[str, list[tuple[int, list[int]]]] = defaultdict(list)
    post_outputs: dict[str, list[tuple[int, list[int]]]] = defaultdict(list)
    counter = [0]
    # Keep every tensor we ever id() alive for the whole trace — see
    # `_tensor_ids`'s docstring for why this matters (id() reuse after
    # garbage collection would otherwise silently create bogus identity
    # matches between unrelated tensors in a deep model).
    _keepalive: list[torch.Tensor] = []

    def _pre_hook(name: str):
        def hook(_mod, args, kwargs):
            pre_inputs[name].append(
                (
                    counter[0],
                    _tensor_ids(args, _keepalive) + _tensor_ids(kwargs, _keepalive),
                )
            )
            counter[0] += 1

        return hook

    def _post_hook(name: str):
        def hook(_mod, _inp, output):
            post_outputs[name].append((counter[0], _tensor_ids(output, _keepalive)))
            counter[0] += 1

        return hook

    handles = []
    for name, mod in model.named_modules():
        if not name:
            continue
        handles.append(mod.register_forward_pre_hook(_pre_hook(name), with_kwargs=True))
        handles.append(mod.register_forward_hook(_post_hook(name)))

    path_by_id = {id(m): n for n, m in model.named_modules() if n}
    # See the identical comment in `_capture_shapes`: kernels persist as
    # real submodules across repeated calls, but hook handles don't.
    _hooked_kernel_ids: set[int] = set()

    def _get_or_create_kernel(module: torch.nn.Module, real_fn):
        path = path_by_id.get(id(module))
        if path is None:
            return None
        kernel = getattr(module, _ATTN_KERNEL_ATTR, None)
        if kernel is None:
            impl_name = getattr(getattr(module, "config", None), "_attn_implementation", "") or ""
            kernel = _AttentionKernel(impl_name)
            setattr(module, _ATTN_KERNEL_ATTR, kernel)
        kernel.forward = lambda *a, **kw: real_fn(module, *a, **kw)
        if id(kernel) not in _hooked_kernel_ids:
            _hooked_kernel_ids.add(id(kernel))
            child_path = f"{path}.{_ATTN_KERNEL_ATTR}"
            handles.append(
                kernel.register_forward_pre_hook(_pre_hook(child_path), with_kwargs=True)
            )
            handles.append(kernel.register_forward_hook(_post_hook(child_path)))
        return kernel

    restore_attention_interface = _patch_attention_interface(_get_or_create_kernel)

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
        restore_attention_interface()
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
    real_output_children: dict[str, set[str]] = {}
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

        if not children:
            continue

        # Ground truth: which child (if any) produced a tensor that is
        # literally part of this composite's own returned output.  Used
        # later to pick the *real* output child when multiple children
        # end up as untracked dead-ends (e.g. an attention kernel that
        # consumes Q/K/V via a raw function call we can't hook, leaving
        # the Q/K/V-producing linears as spurious dead-ends alongside
        # the genuine output projection).
        #
        # Computed even for a single-child composite: a composite whose
        # sole child's output tensor is NOT part of the composite's own
        # returned output (e.g. Glm5NextTextHyperConnection, whose real
        # output is a weighted combination of its *raw* input that never
        # flows through its one child, `input_norm` — `input_norm`'s
        # output only feeds an untracked internal weight computation)
        # must NOT have that child treated as ground truth for its
        # output below — `matches` staying empty is exactly the signal
        # the later output-child heuristics need to avoid picking it.
        comp_output_tids = {
            tid for _, tids in post_outputs.get(comp_path, []) for tid in tids
        }
        if comp_output_tids:
            matches = {
                child
                for child in children
                if comp_output_tids
                & {tid for _, tids in post_outputs.get(child, []) for tid in tids}
            }
            real_output_children[comp_path] = matches

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

        # Children sharing the same untracked input tensor are parallel
        # (e.g. q_proj/k_proj/v_proj all consuming the same masked
        # hidden_states produced by an un-hooked helper function like
        # apply_mask_to_padding_states — a genuinely shared, untracked
        # tensor).  Treat them as all coming from "@input".  Must run
        # BEFORE the sequential fallback below: sequential fallback
        # unconditionally chains every still-unconnected child from
        # whichever child ran immediately before it, so if it ran first
        # it would serialize these parallel siblings into a bogus chain
        # (proj_a → proj_b → proj_c) before this rule ever got a chance
        # to recognize them as parallel.
        children_with_edges = {tgt for _, tgt in child_edges}
        for _tid, consumers in untracked_consumers.items():
            if len(consumers) > 1:
                for child in consumers:
                    if child not in children_with_edges:
                        child_edges.append(("@input", child))
                        children_with_edges.add(child)

        # Sequential fallback: children with no incoming edges are
        # connected from the previous child in execution order.  The
        # very first child gets "@input".  This handles inline tensor ops
        # (e.g. residual combinations) and meta-device forward passes
        # where tensor-ID tracking produces no overlap.
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
            #
            # NOTE: this is a known simplification — it also fires for
            # children with a genuine SECOND substantive input straight
            # from the composite's own input (e.g. GLM's DSA indexer,
            # called as `indexer(hidden_states=hidden_states,
            # q_resid=q_a_layernorm(...))`, truly consumes both the
            # composite's raw input AND a sibling's output). Dropping the
            # "@input" edge there under-represents that second input
            # rather than mis-attributing it, which downstream consumers
            # (single-port "@input" boundary nodes, `_resolve_predecessor`)
            # aren't equipped to render as two distinct edges anyway —
            # see the corresponding shape-audit note for this class of
            # multi-input leaf.
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
            # Determine where the side branch's output actually gets
            # consumed inside `main`. Naively wiring it straight to
            # `main`'s own boundary is wrong when `main`'s first child is
            # a token embedding lookup (nn.Embedding): that child only
            # accepts discrete token ids, never a side branch's continuous
            # features, so the edge would land on a node that can't (and
            # doesn't) consume it — appearing dead when `main` is expanded.
            # This mirrors the common VLM pattern (embed tokens, then
            # splice in modality features, then run the layer stack): route
            # the side branch to the first layer instead, right where the
            # merged embeddings actually enter computation.
            merge_target = main
            embed_path: str | None = None
            try:
                main_children = list(dict(model.named_modules())[main].named_children())
            except (KeyError, AttributeError):
                main_children = []
            if len(main_children) > 1 and isinstance(
                main_children[0][1], torch.nn.Embedding
            ):
                # Remember the token-embedding child so we can wire it as
                # a parallel predecessor of `merge_target` too (below):
                # real dataflow is text embeddings AND modality features
                # both feeding the merge, not just the modality branch —
                # otherwise the embedding lookup renders as a dead end
                # with no consumer.
                embed_path = f"{main}.{main_children[0][0]}"
                second_name, second_mod = main_children[1]
                if isinstance(second_mod, torch.nn.ModuleList) and len(second_mod) > 0:
                    merge_target = f"{main}.{second_name}.0"
                else:
                    merge_target = f"{main}.{second_name}"

            # List the uninvoked side-branch(es) before the main branch so
            # the exec-order topological sort (which processes "@input"
            # targets in list order) places them first — they feed INTO
            # the main branch, so they must execute first.
            root_edges = []
            for side in not_invoked:
                root_edges.append(("@input", side))
                if merge_target == main:
                    root_edges.append((side, main))
                else:
                    # Wire directly into `main`'s internal entry child
                    # rather than `main`'s own boundary, so the merge is
                    # visible when `main` is expanded.
                    edges.setdefault(main, []).append((side, merge_target))
                    if embed_path is not None:
                        # Also show the token-embedding path converging
                        # at the same merge point, so it isn't left as a
                        # disconnected dead end once the side branch's
                        # edge claims `merge_target` as its predecessor.
                        # Both are genuinely parallel sources of the
                        # merged embeddings.
                        edges.setdefault(main, []).append((embed_path, merge_target))
            root_edges.append(("@input", main))
            edges[""] = root_edges

    return edges, real_output_children


def _dim_role_map(
    shape1: tuple[int, ...],
    shape2: tuple[int, ...],
    *,
    batch_size: int,
    seq_len: int,
    batch_size2: int,
    seq_len2: int,
) -> tuple[str, ...] | None:
    """Classify each dim of a *real* captured shape as batch/seq/fixed by
    comparing two probe forward passes run with different concrete
    ``(batch_size, seq_len)`` values.

    A naive single-run heuristic (matching a dim's raw *value* against
    ``batch_size``/``seq_len``) gets fooled whenever an unrelated fixed
    weight dimension coincidentally equals the probed seq_len/batch_size —
    e.g. an attention ``head_dim`` of 128 gets mislabelled as the sequence
    dim "S" purely because tracing happened to use ``seq_len=128`` too,
    producing an impossible-looking "Linear turns 1536 into SxS" shape.
    Comparing two probes with *different* seq_len/batch_size values breaks
    that coincidence: a dim is only "S" if it tracks seq_len in BOTH probes
    (a fixed weight dim can't change across probes, since it doesn't
    depend on the runtime input shape at all).

    Returns ``None`` if the two shapes have different rank (e.g. dynamic,
    input-size-dependent control flow), in which case callers should fall
    back to naive single-shape matching.
    """
    if len(shape1) != len(shape2):
        return None
    roles = []
    for d1, d2 in zip(shape1, shape2):
        if d1 == batch_size and d2 == batch_size2:
            roles.append("B")
        elif d1 == seq_len and d2 == seq_len2:
            roles.append("S")
        else:
            roles.append("fixed")
    return tuple(roles)


def _symbolise(
    shape: tuple[int, ...],
    *,
    batch_size: int = 1,
    seq_len: int = 128,
    roles: tuple[str, ...] | None = None,
) -> str:
    """Convert shape tuple to symbolic string like ``B x S x 4096``.

    If ``roles`` (from `_dim_role_map`, derived from a two-probe
    comparison) is supplied and matches the shape's rank, it takes
    precedence over naive single-value matching — see `_dim_role_map`'s
    docstring for why that matters for correctness.
    """
    if not shape:
        # A genuine 0-d/scalar tensor (e.g. `self.scale.unbind(0)`'s
        # elements) — distinct from "no shape known", which callers
        # represent as `None`/`""`. Must NOT be an empty string: several
        # callers treat a falsy shape string as "unknown, fall back to
        # something else", which would silently mis-attribute a
        # DIFFERENT (wrong) shape to a real 0-d tensor.
        return "scalar"
    parts = []
    has_roles = bool(roles) and len(roles) == len(shape)
    for i, d in enumerate(shape):
        role = roles[i] if has_roles else None
        if role == "B" or (role is None and d == batch_size):
            parts.append("B")
        elif role == "S" or (role is None and d == seq_len):
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

    # Handle a common shape-unpack idiom that breaks FX tracing:
    # `x.view(*x.shape[:-1], a, b)` requires iterating a Proxy's
    # symbolic `.shape`, which FX can't do (Proxies don't support
    # `__iter__`/`__len__`) — but it's exactly equivalent to
    # `x.unflatten(-1, (a, b))` (reshape the trailing dim(s) only,
    # leaving every leading dim untouched), which traces fine. Without
    # this, tracing the WHOLE module aborts with no graph at all — even
    # for every op that ran fine *before* hitting this one line — so
    # e.g. `Glm5NextTextHyperConnection`'s real `hidden_streams.flatten(
    # start_dim=2)` feeding its `input_norm` becomes entirely invisible,
    # and the shape change across that edge looks unexplained.
    # `type(root).forward` is what `torch.fx` actually reads during
    # tracing (see `Tracer.trace`), so an instance-level override
    # wouldn't be picked up — patch the class itself, then always
    # restore it right after, whether tracing then succeeds or not.
    try:
        import inspect

        cls = type(mod)
        src = inspect.getsource(cls.forward)
        pattern = re.compile(r"(\w+)\.view\(\*\1\.shape\[:-1\],\s*(.+?)\)")
        if pattern.search(src):
            new_src = pattern.sub(r"\1.unflatten(-1, (\2,))", src)
            lines = new_src.split("\n")
            start = next(i for i, l in enumerate(lines) if "def forward" in l)
            lines = lines[start:]
            indent = len(lines[0]) - len(lines[0].lstrip())
            lines = [l[indent:] if len(l) > indent else l for l in lines]
            new_src = "\n".join(lines)

            ns: dict = dict(cls.forward.__globals__)
            exec(compile(new_src, "<shape_unpack_patch>", "exec"), ns)  # noqa: S102
            patched_forward = ns.get("forward")
            if patched_forward is not None:
                orig_forward = cls.forward
                cls.forward = patched_forward
                try:
                    traced = torch.fx.symbolic_trace(mod)
                    return traced.graph
                except Exception:
                    pass
                finally:
                    cls.forward = orig_forward
    except Exception:
        pass

    # Handle modules with unregistered nn.Module activations from dicts
    # like ACT2FN[self.activation] — build a mirror class with the
    # activation registered as a proper submodule.
    try:
        import inspect

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


def _propagate_fx_node_shapes(
    mod: torch.nn.Module,
    graph: torch.fx.Graph,
    input_shape: tuple[int, ...] | None,
    *,
    model_dtype: "torch.dtype | None" = None,
) -> dict[str, tuple[tuple[int, ...], "torch.dtype | None"]]:
    """Compute each FX node's REAL output shape (and dtype) by actually running the
    traced graph (on meta tensors) with ``input_shape`` as the module's
    genuine captured input shape.

    Without this, per-op shapes in an FX-expanded sequence (e.g.
    RMSNorm's ``to``/``pow``/``mean``/``add``/``rsqrt``/``mul`` chain)
    were rendered by blindly copying whatever shape was available from
    the nearest node with a known shape — which is simply WRONG for any
    op that actually changes shape: e.g. RMSNorm's
    ``variance = hidden_states.pow(2).mean(-1, keepdim=True)`` reduces
    the last dim to size 1, but the old fallback showed ``mean`` with
    the SAME (un-reduced) shape as its input. Every shape shown must be
    driven by the real op, not assumed to equal a neighbor's.

    Returns ``{}`` (meaning: caller should fall back to the old
    best-effort heuristic) if ``input_shape`` is unknown (the module was
    never actually invoked during tracing — no ground truth exists) or
    if propagation fails for any reason (e.g. a secondary placeholder
    like ``attention_mask``, stood in for with ``None`` below since its
    real value isn't tracked here, turns out to be genuinely used in a
    real tensor op) — never worse than the pre-existing behavior.
    """
    if input_shape is None:
        return {}
    try:
        from torch.fx.passes.shape_prop import ShapeProp

        gm = torch.fx.GraphModule(mod, graph)
        # The module's OWN parameters' dtype is NOT a reliable proxy for
        # its real runtime input dtype — e.g. `RMSNorm.weight` defaults
        # to float32 even inside a bfloat16 model, so guessing from it
        # would make `hidden_states.to(torch.float32)` look like a
        # (false) no-op cast against a float32 example input that was
        # never really the case (and a parameter-less module has no
        # such guess to make at all). Prefer the model's real compute
        # dtype (`model_dtype`) whenever the caller knows it; only fall
        # back to guessing from parameters (or float32) otherwise. If
        # the module's real input is actually integer (rare for a
        # custom class reaching this point), propagation below simply
        # fails and callers fall back to the old best-effort heuristic
        # — never worse than before.
        if model_dtype is not None:
            dtype = model_dtype
        else:
            dtype = next(
                (p.dtype for p in mod.parameters() if p.dtype.is_floating_point),
                torch.float32,
            )
        example = torch.zeros(*input_shape, dtype=dtype, device="meta")
        # Only the first placeholder (the main data tensor) gets the real
        # captured shape; secondary placeholders (attention_mask,
        # position_ids, etc.) are control inputs we don't track values
        # for — `None` is a safe stand-in: either they're genuinely
        # unused in a real tensor op (propagation just succeeds), or
        # they are, and propagation raises, which we catch below exactly
        # like any other failure.
        num_placeholders = sum(1 for n in graph.nodes if n.op == "placeholder")
        args = [example] + [None] * (num_placeholders - 1)
        ShapeProp(gm).propagate(*args)
    except Exception:
        return {}

    result: dict[str, tuple[tuple[int, ...], "torch.dtype | None"]] = {}
    for node in graph.nodes:
        tensor_meta = node.meta.get("tensor_meta")
        shape = getattr(tensor_meta, "shape", None)
        if shape is not None:
            try:
                result[node.name] = (tuple(shape), getattr(tensor_meta, "dtype", None))
            except Exception:
                pass
    return result


def _fx_noop_to_node_names(
    graph: torch.fx.Graph,
    node_metas: dict[str, tuple[tuple[int, ...], "torch.dtype | None"]],
) -> set[str]:
    """FX ``.to(...)`` calls that don't actually change anything: the
    output dtype equals the input dtype. ``.to()`` never changes shape,
    and device is meaningless here since every tensor traces on the
    meta device regardless of the original ``.to(device)`` argument —
    dtype is the only thing that could possibly differ.

    Real models are full of these (e.g. a defensive/no-op ``.to(dtype)``
    inside branches that don't apply to the traced dtype, or
    round-tripping through the same dtype for clarity in the source).
    Showing them as real graph nodes implies a data transformation that
    never actually happens for this trace.
    """
    noop: set[str] = set()
    for node in graph.nodes:
        if node.op != "call_method" or node.target != "to":
            continue
        if not node.args:
            continue
        src = node.args[0]
        if not isinstance(src, torch.fx.Node):
            continue
        self_meta = node_metas.get(node.name)
        src_meta = node_metas.get(src.name)
        if self_meta is None or src_meta is None:
            continue
        self_dtype, src_dtype = self_meta[1], src_meta[1]
        if self_dtype is not None and self_dtype == src_dtype:
            noop.add(node.name)
    return noop


def _fx_node_shape_strings(
    mod: torch.nn.Module,
    graph: torch.fx.Graph,
    path: str,
    *,
    hook_input_shapes: dict[str, tuple[int, ...]],
    hook_input_shapes2: dict[str, tuple[int, ...]],
    batch_size: int,
    seq_len: int,
    batch_size2: int,
    seq_len2: int,
    model_dtype: "torch.dtype | None" = None,
) -> tuple[dict[str, str], set[str]]:
    """Real, symbolised per-FX-node output shapes for ``path``'s traced
    graph (see `_propagate_fx_node_shapes`), disambiguated against a
    second probe the same way real hook-captured shapes are (see
    `_dim_role_map`) — a fixed weight dim inside the op sequence could
    just as easily coincide with the probed seq_len/batch_size as any
    other captured shape.

    Also returns the set of no-op ``.to(...)`` FX node names (see
    `_fx_noop_to_node_names`) found via the SAME (primary-probe)
    ShapeProp run, so callers can skip emitting a node for a cast that
    doesn't actually change anything.
    """
    shapes1 = _propagate_fx_node_shapes(
        mod, graph, hook_input_shapes.get(path), model_dtype=model_dtype
    )
    if not shapes1:
        return {}, set()
    noop_to_names = _fx_noop_to_node_names(graph, shapes1)
    shapes2 = _propagate_fx_node_shapes(
        mod, graph, hook_input_shapes2.get(path), model_dtype=model_dtype
    )
    result: dict[str, str] = {}
    for name, (shape, _dtype) in shapes1.items():
        shape2_entry = shapes2.get(name)
        shape2 = shape2_entry[0] if shape2_entry is not None else None
        roles = (
            _dim_role_map(
                shape,
                shape2,
                batch_size=batch_size,
                seq_len=seq_len,
                batch_size2=batch_size2,
                seq_len2=seq_len2,
            )
            if shape2 is not None
            else None
        )
        result[name] = _symbolise(
            shape, batch_size=batch_size, seq_len=seq_len, roles=roles
        )
    return result, noop_to_names


def _collapse_repeated_op_blocks(
    op_nodes: list[dict],
) -> tuple[list[dict], dict[str, str]]:
    """Collapse a straight-line chain of REPEATED, identically-labeled op
    blocks — the FX-unrolled form of a Python ``for _ in range(N): ...``
    loop over a loop-carried tensor (e.g. HyperConnection's Sinkhorn
    normalization, ``for _ in range(hc_sinkhorn_iters - 1): comb = comb /
    (...)``  ) — down to ONE representative iteration annotated with the
    real iteration count.

    This mirrors how the AST-based backend represents a ``for`` loop:
    trace the body ONCE and annotate it with the real iteration count
    (``ast_analyze.py``'s ``"loop: N iterations"`` detail), rather than
    literally duplicating the same op sequence dozens of times, which
    would bloat the graph with near-identical nodes without adding any
    real information (every repetition has the exact same op sequence
    and, since it's a genuine reduction/normalize-in-place loop, the
    exact same shape at each corresponding position).

    Detection is purely structural (label sequence + dataflow chaining),
    not tied to any specific model/module — a run of >=3 consecutive,
    equal-length, equal-label-sequence blocks where each block's first
    node consumes the PREVIOUS block's LAST node (the loop-carried
    tensor) qualifies, for any op sequence.

    Returns the (possibly shortened) op_nodes list plus a
    ``{removed_node_id: surviving_representative_id}`` remap the caller
    must apply to any other edges/lookups (e.g. `fx_output_ids`,
    `fx_child_edges`, `node_map`) that may have referenced a removed
    node.
    """
    n = len(op_nodes)
    id_remap: dict[str, str] = {}
    if n < 6:
        return op_nodes, id_remap

    def label_seq(start: int, length: int) -> tuple[str, ...]:
        return tuple(op_nodes[start + i]["label"] for i in range(length))

    def chains(prev_last_idx: int, next_first_idx: int) -> bool:
        edges = op_nodes[next_first_idx].get("incomingEdges", [])
        prev_id = op_nodes[prev_last_idx]["id"]
        return any(e.get("sourceNodeId") == prev_id for e in edges)

    result: list[dict] = []
    i = 0
    while i < n:
        collapsed = False
        max_block_len = min(8, (n - i) // 3)
        for block_len in range(1, max_block_len + 1):
            reps = 1
            while True:
                cur_start = i + reps * block_len
                if cur_start + block_len > n:
                    break
                if label_seq(i, block_len) != label_seq(cur_start, block_len):
                    break
                if not chains(cur_start - 1, cur_start):
                    break
                reps += 1
            if reps >= 3:
                rep_block = op_nodes[i : i + block_len]
                for r in range(1, reps):
                    removed_block = op_nodes[
                        i + r * block_len : i + (r + 1) * block_len
                    ]
                    for pos, removed_node in enumerate(removed_block):
                        id_remap[removed_node["id"]] = rep_block[pos]["id"]
                annotated_rep = [dict(node) for node in rep_block]
                first = dict(annotated_rep[0])
                first["attrs"] = [
                    *first.get("attrs", []),
                    {"key": "loop_iterations", "value": str(reps)},
                ]
                annotated_rep[0] = first
                result.extend(annotated_rep)
                i += reps * block_len
                collapsed = True
                break
        if not collapsed:
            result.append(op_nodes[i])
            i += 1
    return result, id_remap


def _apply_id_remap(nodes: list[dict], id_remap: dict[str, str]) -> None:
    """Redirect any incoming edge that pointed at a now-removed node
    (per `_collapse_repeated_op_blocks`) to its surviving representative.
    """
    if not id_remap:
        return
    for node in nodes:
        for edge in node.get("incomingEdges", []):
            sid = edge.get("sourceNodeId")
            if sid in id_remap:
                edge["sourceNodeId"] = id_remap[sid]


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
    if isinstance(mod, _AttentionKernel):
        return "attention"
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
    # "norm" is intentionally absent: every norm in practice is either a
    # custom RMSNorm subclass (FX-expanded into its raw ops, each styled
    # via _STYLE_OP) or a built-in nn.LayerNorm leaf that falls through
    # to _STYLE_DEFAULT below — giving it its own distinct color would
    # make it stand out as if it were structurally different from every
    # other norm in the graph, when it isn't.
    return {
        "embedding": _STYLE_EMBEDDING,
        "linear": _STYLE_LINEAR,
        "attention": _STYLE_ATTENTION,
        "activation": _STYLE_ACTIVATION,
        "op": _STYLE_OP,
        "input": _STYLE_INPUT,
    }.get(category, _STYLE_DEFAULT)


def _module_label(mod: torch.nn.Module) -> str:
    """Friendly label for a module."""
    cls = type(mod).__name__
    if isinstance(mod, _AttentionKernel):
        return _attention_kernel_label(mod._impl_name)
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
    """Build an HTML fact sheet from the config, including sub-configs.

    The model name is already shown as the fact-sheet title in the viewer
    header, so it isn't repeated here — the first line is a non-bold
    generation timestamp instead.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"Generated: {generated_at}"]

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
    hook_shapes, hook_input_shapes, hook_all_output_shapes = _capture_shapes(
        model, seq_len=seq_len, batch_size=batch_size
    )
    # Second probe pass with *different* concrete (batch_size, seq_len)
    # values, purely to disambiguate which dims of the shapes captured
    # above are actually batch-/sequence-derived vs. fixed weight dims
    # that happen to coincide numerically with the probed batch_size/
    # seq_len (see `_dim_role_map`). Meta-device tracing has no real
    # compute cost, so this second pass is cheap.
    seq_len2 = seq_len + 11
    batch_size2 = batch_size + 5
    hook_shapes2, hook_input_shapes2, hook_all_output_shapes2 = _capture_shapes(
        model, seq_len=seq_len2, batch_size=batch_size2
    )

    def _roles_for(path: str, shape: tuple[int, ...], probe: dict[str, tuple[int, ...]]):
        shape2 = probe.get(path)
        if shape2 is None:
            return None
        return _dim_role_map(
            shape,
            shape2,
            batch_size=batch_size,
            seq_len=seq_len,
            batch_size2=batch_size2,
            seq_len2=seq_len2,
        )

    raw_shapes = _infer_shapes_from_weights(
        model, hook_shapes, batch_size=batch_size, seq_len=seq_len
    )
    shapes: dict[str, str] = {}
    for path, shape in raw_shapes.items():
        roles = _roles_for(path, shape, hook_shapes2)
        shapes[path] = _symbolise(
            shape, batch_size=batch_size, seq_len=seq_len, roles=roles
        )
    raw_input_shapes = _infer_input_shapes_from_weights(
        model, hook_input_shapes, batch_size=batch_size, seq_len=seq_len
    )
    input_shapes: dict[str, str] = {}
    for path, shape in raw_input_shapes.items():
        roles = _roles_for(path, shape, hook_input_shapes2)
        input_shapes[path] = _symbolise(
            shape, batch_size=batch_size, seq_len=seq_len, roles=roles
        )
    # Every tensor shape found in each module's returned tuple/list
    # (symbolised), keyed by path — used to disambiguate which element of
    # a multi-tensor return value is the "real" one that continues into a
    # given downstream consumer (see `_pick_output_shape`).
    all_output_shapes: dict[str, list[str]] = {}
    for path, shape_list in hook_all_output_shapes.items():
        shape_list2 = hook_all_output_shapes2.get(path)
        symbolised: list[str] = []
        for i, s in enumerate(shape_list):
            s2 = (
                shape_list2[i]
                if shape_list2 is not None and i < len(shape_list2)
                else None
            )
            roles = (
                _dim_role_map(
                    s,
                    s2,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    batch_size2=batch_size2,
                    seq_len2=seq_len2,
                )
                if s2 is not None
                else None
            )
            symbolised.append(
                _symbolise(s, batch_size=batch_size, seq_len=seq_len, roles=roles)
            )
        all_output_shapes[path] = symbolised

    def _pick_output_shape(path: str) -> str | None:
        """Return the symbolised output shape to show for ``path``.

        Defaults to ``shapes.get(path)`` (the first tensor found in the
        module's return value). If the module actually returned SEVERAL
        differently-shaped tensors, and we know (from the call graph) which
        sibling it feeds directly, prefer whichever returned tensor's shape
        matches that successor's OWN captured input shape — e.g.
        Glm5NextTextHyperConnection returns ``(post, comb, collapsed)``;
        only ``collapsed`` (the last element, not the first) is what
        actually flows into the next module (``input_layernorm``), and
        picking the first tensor's shape would silently print a shape that
        doesn't match its successor's input, implying a bogus reshape.
        """
        candidates = all_output_shapes.get(path)
        if not candidates or len(candidates) < 2:
            return shapes.get(path)
        if len(set(candidates)) == 1:
            return candidates[0]
        successors: set[str] = set()
        for _comp_path, edges in call_graph.items():
            for src, tgt in edges:
                if src == path:
                    successors.add(tgt)
        for succ in successors:
            succ_in = input_shapes.get(succ)
            if succ_in and succ_in in candidates:
                return succ_in
        return shapes.get(path)

    # Leaf modules that were never actually invoked during the dummy
    # forward pass (no hook fired for them at all — distinct from
    # modules whose *tensor identity* just couldn't be tracked), yet
    # whose direct parent WAS invoked. This is the "read my .weight/
    # .bias directly, never call self.mod(x)" pattern (e.g. Kimi linear
    # attention's conv1d, whose kernel weights are passed straight into
    # a raw causal_conv1d_fn(...) call instead of `self.conv1d(x)`).
    # There's no real tensor to trace flowing in or out of these, so
    # they should render as standalone info boxes (shape inferred from
    # weights above) rather than getting a speculative positional
    # "@input" edge from the generic wiring fallback later — a half
    # wire (input but no consumer) looks like a broken/dead computation
    # step, when really the module just isn't part of the traced
    # dataflow at all.  This must NOT apply to leaves whose entire
    # ancestor chain was never invoked (e.g. an omitted vision branch)
    # — those still need the generic positional fallback to render
    # sensibly, since there's no live parent to contrast them against.
    def _was_invoked(path: str) -> bool:
        return path in hook_shapes or path in hook_input_shapes

    _dead_leaves_in_live_parents: set[str] = set()
    for _name, _mod in model.named_modules():
        if not _name or list(_mod.children()):
            continue  # only leaves
        if _was_invoked(_name):
            continue
        _parent = _name.rsplit(".", 1)[0] if "." in _name else ""
        if _was_invoked(_parent):
            _dead_leaves_in_live_parents.add(_name)

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

    # Representatives of containers with MULTIPLE distinct layer types
    # (interleaved). These render as parallel sibling groups (AST-style)
    # rather than a sequential chain, so any "previous representative
    # ran right before me" edge — e.g. from whole-module FX tracing,
    # which always unrolls a for-loop into a flat sequential chain — is
    # a misleading artifact of collapsing duplicates to one
    # representative and must be suppressed; the real fan-out/fan-in
    # wiring is instead derived from the call graph (see the Fork/Join
    # replacement below).
    multi_group_rep_paths: set[str] = set()
    # Node-ID prefixes of every multi-group container's *entire* index
    # range (reps AND skipped duplicates) — used to catch FX edges whose
    # source is "whichever duplicate happened to be unrolled last" (e.g.
    # a node right after the whole layer stack sourcing from the FX
    # graph's literal final iteration instead of fanning in from every
    # representative's own output).
    multi_group_container_id_prefixes: list[str] = []

    for container_path, groups in layer_group_map.items():
        total = sum(g.count for g in groups)
        container_total[container_path] = total
        for group in groups:
            representative_paths.add(f"{container_path}.{group.representative}")
        if len(groups) > 1:
            for group in groups:
                multi_group_rep_paths.add(f"{container_path}.{group.representative}")
            multi_group_container_id_prefixes.append(
                container_path.replace(".", "/") + "/"
            )
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

    # Populated later (after FX tracing) with composites that should be
    # rendered pass-through (see `_single_child_passthrough` below).
    # Declared up front — before `_namespace_for` even exists — because
    # `_namespace_for` is a closure invoked during FX tracing itself
    # (before this set's real contents are known); it must see an empty
    # set at that point rather than raise NameError.
    _single_child_passthrough: set[str] = set()

    # ── Capture call graph for dataflow-aware edge wiring ────────────────
    call_graph, real_output_children = _capture_call_graph(
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
    # the parent's call graph with direct parallel edges — every
    # predecessor feeds every layer-type representative directly, and
    # every representative feeds every successor directly. This mirrors
    # how the AST backend renders interleaved layer-type stacks: sibling
    # variant groups fanning out from (and back into) the surrounding
    # dataflow with no separate Fork/Join wiring nodes in between.
    for container_path, groups in layer_group_map.items():
        if len(groups) <= 1:
            continue
        parent_path = container_path.rsplit(".", 1)[0] if "." in container_path else ""
        if parent_path not in call_graph:
            continue
        rep_paths = {f"{container_path}.{g.representative}" for g in groups}

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

        # The generic "@input" sentinel is a weak, low-priority fallback
        # predecessor — some rep may have a stray "(@input, rep)" edge
        # (e.g. a secondary FX placeholder) alongside a genuinely
        # resolved real predecessor like a token-embedding lookup or a
        # side-modality merge. Fanning "@input" out to every
        # representative alongside real predecessors would dilute/
        # collide with those real, specific sources once flattened into
        # the target-keyed `cg_sources` map downstream. Prefer real
        # predecessors whenever any exist.
        real_predecessors = predecessors - {"@input"}
        fan_out_predecessors = real_predecessors or predecessors

        # Keep non-layer edges, skip inter-layer edges
        for src, tgt in old_edges:
            if src in rep_paths or tgt in rep_paths:
                continue  # Skip all edges involving layers
            new_edges.append((src, tgt))

        # Fan every predecessor out to every representative, and fan
        # every representative in to every successor — directly, with
        # no intermediary Fork/Join node.
        for pred in fan_out_predecessors:
            for rep in rep_paths:
                new_edges.append((pred, rep))
        for rep in rep_paths:
            for succ in successors:
                new_edges.append((rep, succ))

        call_graph[parent_path] = new_edges

        # Also transform the container's own call_graph entry so that
        # the sequential layer chain is replaced with parallel branches.
        # Reuse the SAME (fully-resolved) predecessors/successors as the
        # parent scope above rather than re-deriving them from the
        # container's own, more limited edge set: `cg_sources` (built
        # later from ALL call_graph scopes) is a flat, target-keyed map
        # with "last write wins" semantics, so a weaker container-scope
        # entry for the same rep target would silently clobber the
        # parent's more complete resolution (e.g. dropping a
        # side-modality merge the container itself has no knowledge of).
        if container_path in call_graph:
            cont_old = call_graph[container_path]
            cont_new = []
            for src, tgt in cont_old:
                if src in rep_paths or tgt in rep_paths:
                    continue
                cont_new.append((src, tgt))
            for pred in fan_out_predecessors:
                for rep in rep_paths:
                    cont_new.append((pred, rep))
            for rep in rep_paths:
                for succ in successors:
                    cont_new.append((rep, succ))
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

            # Use attr name for composites, class name for clarity.
            # Single-child pass-through composites (see
            # `_single_child_passthrough` below) don't get their own
            # namespace segment — their one child is promoted to render
            # directly inside the parent's namespace instead of being
            # nested inside a pointless wrapper box.
            if prefix in composite_modules and prefix not in _single_child_passthrough:
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
    # Raw op nodes that must be emitted right after a specific call_module
    # child (keyed by that child's dotted path) rather than all up-front at
    # the parent composite's own module_order position. This preserves true
    # execution order when a composite's forward() interleaves calls to
    # child submodules with inline tensor ops (e.g. proj -> norm -> act ->
    # gate_proj/up_proj -> (raw silu/mul ops) -> down_proj).
    fx_trailing_ops: dict[str, list[dict]] = {}  # child_path → [op nodes]
    # Composites where whole-module Path A tracing was attempted (has a
    # leaf child) but genuinely failed (e.g. data-dependent control flow
    # like a sinkhorn-iteration loop) — as opposed to composites that
    # simply have no raw ops of their own because tracing trivially
    # succeeded (e.g. a wrapper whose forward is just `self.proj(x)`).
    # Only the former should be considered for single-child pass-through
    # rendering below — the latter's own boundary is still meaningful.
    fx_trace_failed: set[str] = set()

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
            fx_trace_failed.add(path)
            continue

        fx_shape_strings, fx_noop_to_names = _fx_node_shape_strings(
            mod,
            graph,
            path,
            hook_input_shapes=hook_input_shapes,
            hook_input_shapes2=hook_input_shapes2,
            batch_size=batch_size,
            seq_len=seq_len,
            batch_size2=batch_size2,
            seq_len2=seq_len2,
            model_dtype=getattr(torch, dtype, None),
        )

        namespace = _namespace_for(path) or type(mod).__name__
        op_nodes = []
        node_map: dict[str, str] = {}
        # Tracks the most recently seen call_module child in the FX graph's
        # own node order. Raw tensor ops are tagged with this so that later
        # emission can interleave them AFTER that child (matching true
        # execution order) instead of dumping all raw ops before any child
        # — see `fx_trailing_ops` below.
        last_child_path: str | None = None

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
                # A layer-type representative's FX-graph predecessor is
                # always "whatever representative (or duplicate) was
                # unrolled right before it" — an artifact of the
                # for-loop unrolling, not real fan-out/fan-in semantics.
                # Leave it for the call-graph-based wiring pass to
                # resolve instead, both when THIS child is itself such a
                # representative, and when its FX-graph source lies
                # inside a multi-group container (e.g. the node right
                # after the whole interleaved layer stack, whose FX
                # source is literally the stack's final unrolled
                # duplicate rather than every representative's output).
                from_multi_group_container = any(
                    edge["sourceNodeId"].startswith(prefix)
                    or edge["sourceNodeId"].rstrip("/") == prefix.rstrip("/")
                    for edge in child_incoming
                    for prefix in multi_group_container_id_prefixes
                )
                if (
                    child_incoming
                    and child_path not in multi_group_rep_paths
                    and not from_multi_group_container
                ):
                    fx_child_edges[child_path] = child_incoming
                last_child_path = child_path
                continue

            if fx_node.name in fx_noop_to_names:
                # A `.to(...)` call that doesn't actually change dtype
                # (or anything else — shape never changes via `.to()`,
                # and device is meaningless on the meta device) — skip
                # emitting a node for it; downstream ops connect
                # straight through to its real predecessor instead of
                # implying a cast that never actually happens.
                src_arg = fx_node.args[0] if fx_node.args else None
                src_name = (
                    src_arg.name if isinstance(src_arg, torch.fx.Node) else None
                )
                node_map[fx_node.name] = node_map.get(src_name, "@input")
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
            op_node["_anchor_child"] = last_child_path
            shape_str = fx_shape_strings.get(fx_node.name)
            if shape_str:
                op_node["outputsMetadata"] = _output_metadata(shape_str, dtype)
            op_nodes.append(op_node)

        if op_nodes:
            op_nodes, _loop_id_remap = _collapse_repeated_op_blocks(op_nodes)
            if _loop_id_remap:
                _apply_id_remap(op_nodes, _loop_id_remap)
                for _k, _v in node_map.items():
                    if _v in _loop_id_remap:
                        node_map[_k] = _loop_id_remap[_v]
                if path in fx_output_ids:
                    fx_output_ids[path] = [
                        _loop_id_remap.get(nid, nid) for nid in fx_output_ids[path]
                    ]
                for _edges in fx_child_edges.values():
                    for _edge in _edges:
                        _sid = _edge.get("sourceNodeId")
                        if _sid in _loop_id_remap:
                            _edge["sourceNodeId"] = _loop_id_remap[_sid]
            fx_graphs[path] = op_nodes

    # ── Detect single-child pass-through composites ──────────────────────
    # A composite with exactly one direct child module and no computation
    # of its own — its whole-module Path A trace was attempted (it has a
    # leaf child) but genuinely failed (e.g. data-dependent control flow
    # like a sinkhorn-iteration loop), as opposed to a composite that
    # simply has no raw ops because tracing trivially succeeded (e.g. a
    # wrapper whose forward is just `self.proj(x)`) — contributes nothing
    # besides wrapping that one child in an extra "@input"/"@output" box.
    # Render such composites' children directly in the parent's namespace
    # instead of nesting them one level deeper.
    # Must run BEFORE the custom-leaf-expansion loop below, since that
    # loop bakes each leaf's `parent_ns` (namespace + own attr/class
    # label) into all of its emitted op nodes up front.
    #
    # Guard: only inline when the child's OWN captured I/O shapes match
    # the composite's OWN captured I/O shapes exactly. A genuine trivial
    # wrapper's shapes are identical to its child's (nothing happens
    # before/after the call). If they differ, the composite has real
    # extra computation around the child that we can't see — e.g.
    # Glm5NextTextHyperConnection normalizes a *flattened* view of its
    # input through `input_norm` purely to derive gating weights, then
    # separately returns a differently-shaped weighted combination of the
    # *original* (unnormalized) input. Inlining `input_norm` there would
    # make it look like the composite's real output IS `input_norm`'s
    # output, which has a different shape than what the next module
    # actually consumes — a bogus "shape changed for no reason" edge.
    for cpath in fx_trace_failed:
        if cpath in fx_graphs:
            continue
        direct_children = module_children.get(cpath, [])
        if len(direct_children) != 1:
            continue
        child_path = direct_children[0]
        # Compare against every tensor shape the composite's own hook saw
        # in its return value (not just the first) — a composite that
        # returns a tuple where *some* element matches the child's output
        # is still plausibly a genuine wrapper for that element.
        comp_out_candidates = all_output_shapes.get(cpath) or (
            [shapes[cpath]] if cpath in shapes else []
        )
        child_out = shapes.get(child_path)
        comp_in, child_in = input_shapes.get(cpath), input_shapes.get(child_path)
        out_mismatch = bool(comp_out_candidates) and child_out is not None and (
            child_out not in comp_out_candidates
        )
        in_mismatch = comp_in is not None and child_in is not None and comp_in != child_in
        if out_mismatch or in_mismatch:
            continue
        _single_child_passthrough.add(cpath)

    # The viewer groups nodes into visual boxes keyed by the raw
    # `namespace` string, so two DIFFERENT pass-through composites whose
    # namespace segment we're about to drop (e.g. sibling `attn_hc` and
    # `ffn_hc`, each wrapping a same-named/same-class `input_norm` child)
    # would otherwise produce an IDENTICAL namespace for their children
    # once stripped, silently merging two distinct instances into one
    # box. Disambiguate by labeling the inlined child with the DROPPED
    # composite's own attr name alone (e.g. "attn_hc", not
    # "attn_hc.input_norm") — the parent attr is already unique among
    # its own siblings, so it's sufficient on its own, and it avoids
    # echoing the child's generic name (often something like
    # "input_norm"), which visually reads as a near-duplicate of
    # unrelated same-layer nodes like "input_layernorm".
    for cpath in _single_child_passthrough:
        child_path = module_children[cpath][0]  # full dotted path
        parent_attr = attr_names.get(cpath, cpath.rsplit(".", 1)[-1])
        attr_names[child_path] = parent_attr

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

        fx_shape_strings, fx_noop_to_names = _fx_node_shape_strings(
            mod,
            graph,
            path,
            hook_input_shapes=hook_input_shapes,
            hook_input_shapes2=hook_input_shapes2,
            batch_size=batch_size,
            seq_len=seq_len,
            batch_size2=batch_size2,
            seq_len2=seq_len2,
            model_dtype=getattr(torch, dtype, None),
        )

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
                # Capture output node args for multi-return output-port
                # splitting below (mirrors the composite call site above)
                # — e.g. Glm5NextTextTopkRouter's `return router_logits,
                # topk_weights, topk_indices`.
                if fx_node.op == "output":
                    out_ids = []
                    for arg in fx_node.all_input_nodes:
                        nid = node_map.get(arg.name)
                        if nid and nid != "@input":
                            out_ids.append(nid)
                    if out_ids:
                        fx_output_ids[path] = out_ids
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

            if fx_node.name in fx_noop_to_names:
                # A `.to(...)` call that doesn't actually change dtype
                # (or anything else) — skip emitting a node for it; see
                # the identical check at the composite call site above.
                src_arg = fx_node.args[0] if fx_node.args else None
                src_name = (
                    src_arg.name if isinstance(src_arg, torch.fx.Node) else None
                )
                node_map[fx_node.name] = node_map.get(src_name, "@input")
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
            shape_str = fx_shape_strings.get(fx_node.name)
            if shape_str:
                op_node["outputsMetadata"] = _output_metadata(shape_str, dtype)
            op_nodes.append(op_node)

        # If the module's entire computation is a single primitive op
        # (e.g. Glm5NextTextHyperHead's `hidden_streams.mean(dim=2)`),
        # showing a nested "hc_head (Glm5NextTextHyperHead)" box with its
        # own @input/mean/@output boundary just wraps one op in
        # pointless indirection. Leave it out of fx_graphs entirely so
        # it falls through to the regular leaf-module path below, which
        # already picks the op's own name as the label for a single-op
        # leaf (see _module_label) and uses the module's own
        # hook-captured output shape directly.
        if len(op_nodes) > 1:
            op_nodes, _loop_id_remap = _collapse_repeated_op_blocks(op_nodes)
            if _loop_id_remap:
                _apply_id_remap(op_nodes, _loop_id_remap)
                for _k, _v in node_map.items():
                    if _v in _loop_id_remap:
                        node_map[_k] = _loop_id_remap[_v]
                if path in fx_output_ids:
                    fx_output_ids[path] = [
                        _loop_id_remap.get(nid, nid) for nid in fx_output_ids[path]
                    ]
            fx_graphs[path] = op_nodes

    # ── Add module nodes ─────────────────────────────────────────────────
    # Emits nodes (plain leaves, group attrs, and FX-expanded op nodes) in
    # a single pass over module_order, so relative ordering between e.g. a
    # composite-with-FX-graph child and a leaf-with-FX-graph sibling stays
    # correct (previously these were emitted in two separate later loops,
    # split by composite-vs-leaf classification rather than true call
    # order — which silently reordered custom leaf modules like a rotary
    # embedding helper to the very end, after every composite's FX ops).
    def _emit_fx_op_nodes(path: str, op_nodes: list[dict] | None = None) -> None:
        namespace = _namespace_for(path)
        mod = module_map[path]
        attr = attr_names.get(path, type(mod).__name__)
        parent_ns = (
            namespace + f"/{attr} ({type(mod).__name__})"
            if namespace
            else f"{attr} ({type(mod).__name__})"
        )
        for op_node in fx_graphs[path] if op_nodes is None else op_nodes:
            op_node.pop("_anchor_child", None)
            if not op_node["namespace"].startswith(parent_ns):
                op_node["namespace"] = parent_ns
            nodes.append(op_node)

    def _flush_trailing_ops(path: str) -> None:
        """Emit any raw ops that ran right after `path` (a call_module
        child) in its parent's FX graph, immediately after `path` itself
        has been emitted — preserving true execution order."""
        if path not in fx_trailing_ops:
            return
        parent_path = path.rsplit(".", 1)[0] if "." in path else ""
        _emit_fx_op_nodes(parent_path, fx_trailing_ops[path])

    for path in module_order:
        if not path:
            continue
        if _should_skip(path):
            continue
        if path in _single_child_passthrough:
            # No group box for this composite — its single child renders
            # directly in the parent's namespace (see
            # `_single_child_passthrough` above).
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
                shape_str = _pick_output_shape(path)
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
            _flush_trailing_ops(path)
            continue

        if path in composite_modules and path in fx_graphs:
            # Split into a leading segment (raw ops that ran before any
            # call_module child) emitted now, and trailing segments
            # (raw ops that ran right after a given child) deferred until
            # that child itself is emitted below — preserving true
            # execution order instead of dumping all raw ops up-front.
            leading = [n for n in fx_graphs[path] if n.get("_anchor_child") is None]
            _emit_fx_op_nodes(path, leading)
            for n in fx_graphs[path]:
                anchor = n.get("_anchor_child")
                if anchor is not None:
                    fx_trailing_ops.setdefault(anchor, []).append(n)
            _flush_trailing_ops(path)
            continue

        # Custom leaf module that was FX-expanded — emit its op nodes here
        # (in module_order position) instead of the single leaf node.
        if path in fx_graphs:
            _emit_fx_op_nodes(path)
            _flush_trailing_ops(path)
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
        elif path in _dead_leaves_in_live_parents:
            # Never actually invoked, even though its parent was — there's
            # no real tensor flowing in (or out). Pre-set an empty edge
            # list so the generic positional-fallback wiring pass below
            # leaves it alone instead of guessing a misleading "@input"
            # source for a module that isn't really part of the dataflow.
            node["incomingEdges"] = []

        nodes.append(node)
        edges_from[path] = node_id
        _flush_trailing_ops(path)

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
        if not orders:
            return (999999,)
        # A composite's own synthetic "@output" boundary genuinely runs
        # LAST within its subtree — but if a deeper descendant (a nested
        # composite with its own call-graph entry) has a *longer*
        # resolved order tuple, plain tuple comparison would rank the
        # @output's short tuple as "less than" (i.e. before) the
        # descendant's, since a tuple that's a strict prefix of another
        # always sorts first regardless of what follows. That silently
        # places e.g. "visual/@output" before "visual/merger/down_proj",
        # even though merger's down_proj is what visual/@output is
        # actually derived from. Pad with a large sentinel so it always
        # sorts after any sibling/descendant sharing the same prefix.
        if nid == "@output" or nid.endswith("/@output"):
            orders.append(999998)
        return tuple(orders)

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
        real_output_children=real_output_children,
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
        # Ground truth (from real tensor-identity tracking, see
        # `_capture_call_graph`): if NONE of this composite's direct
        # children are known to produce its real returned output (e.g.
        # Glm5NextTextHyperConnection/`attn_hc`, whose real computation
        # is an untracked sinkhorn-iteration loop — its one real
        # submodule, `input_norm`, only feeds an untracked internal
        # weight computation, never the composite's actual return
        # value), don't hand back that merely-orphaned descendant as if
        # it stood in for the composite's output. Point at the
        # composite's own (not-yet-created) "@output" boundary instead
        # — the composite-boundary pass later in `build_graph` creates
        # a real node with exactly this id (giving it correct shape
        # metadata from the composite's own captured hook shape,
        # regardless of internal wiring), so this reference resolves
        # correctly once that pass runs.
        if last is not None and real_output_children.get(mod_path) == set():
            return f"{prefix}@output"
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

    # Layer-type representatives of multi-group (interleaved) containers
    # each get their OWN "@input" boundary, carefully resolved straight
    # from the real call-graph (e.g. a side-modality branch like vision
    # features merging in ahead of a specific layer type). That source
    # may live OUTSIDE an ancestor composite even though the rep is
    # namespaced inside it — see the exclusion below.
    _layer_group_rep_input_ids: set[str] = set()
    for _container_path, _groups in layer_group_map.items():
        if len(_groups) <= 1:
            continue
        for _group in _groups:
            _rep_path = f"{_container_path}.{_group.representative}"
            _layer_group_rep_input_ids.add(_node_id(_rep_path) + "/@input")

    for comp_path in sorted(composite_modules, key=lambda p: (-p.count("."), p)):
        if _should_skip(comp_path):
            continue
        if comp_path in _single_child_passthrough:
            # No wrapper box for this composite — its single child's own
            # @input/@output (created in its own iteration of this loop,
            # or via the FX-expanded-leaf I/O pass) stand in for it.
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
        # Children whose incoming edges come from OUTSIDE this module.
        # A layer-type representative's own "@input" is excluded: its
        # predecessor(s) were already carefully resolved straight from
        # the real call-graph (e.g. a side-modality branch like vision
        # features merging in ahead of a specific layer type), and that
        # source may live OUTSIDE this ancestor composite even though
        # the representative is namespaced inside it. Treating its
        # "@input" as a generic "input child" would collapse its real,
        # specific predecessor into this ancestor's own blanket
        # "@input" — destroying the distinction between "the model's
        # own primary input" and "a side branch merging in partway
        # through".
        input_child_ids: list[str] = []
        external_sources: dict[str, list[str]] = {}  # child_id → [external_src_ids]
        for cn in child_nodes:
            if cn["id"] in _layer_group_rep_input_ids:
                continue
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

        # Ground truth (ordinary FX-graph output, or a real consumer
        # genuinely outside this module) is trustworthy on its own —
        # only the weaker heuristics below (orphan/last-resort guesses)
        # need the `real_output_children` sanity check applied later,
        # since an "empty means untracked" ground-truth reading is only
        # safe to *distrust a guess* with, not to override real
        # evidence (e.g. a residual-add composite legitimately has no
        # single child whose tensor IS its output, yet tier 1/2 above
        # can still correctly identify the right child via real
        # dataflow).
        _output_child_ids_are_weak_guess = not output_child_ids

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
            # Ground-truth disambiguation: if some terminals are known
            # (from real tensor-identity tracking) to literally be part
            # of the composite's own returned output, prefer those over
            # the full terminal set.  This filters out spurious dead-ends
            # caused by raw function calls (e.g. an attention kernel)
            # that aren't captured by module hooks — such calls leave
            # their inputs (Q/K/V projections) as untracked dead-ends
            # alongside the genuine output projection.
            real_terminals = terminals & real_output_children.get(comp_path, set())
            if real_terminals:
                terminals = real_terminals
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
            # Ground truth (from real tensor-identity tracking, see
            # `_capture_call_graph`): the set of this composite's DIRECT
            # children whose output is literally part of its own
            # returned output. When known (even as an empty set), don't
            # wire the boundary's @output from a candidate outside that
            # set — e.g. Glm5NextTextHyperConnection (`attn_hc`)'s real
            # computation (a sinkhorn-iteration loop) is untracked, so
            # nothing of "self" is visible except its one real
            # submodule, `input_norm` — whose own output only feeds an
            # untracked internal weight computation, never the
            # composite's actual returned value. Wiring @output from it
            # anyway would misrepresent input_norm's (differently-
            # shaped) output as attn_hc's real return value, making it
            # look like two RMSNorms run back-to-back with nothing in
            # between. Still create the @output node itself (with
            # correct shape metadata from the composite's own captured
            # hook shape, set later) so downstream consumers have a
            # boundary to be rewired onto below — just leave it with no
            # (known-wrong) incoming edge of its own.
            _known_real_outputs = (
                real_output_children.get(comp_path)
                if _output_child_ids_are_weak_guess
                else None
            )
            if _known_real_outputs is not None:

                def _direct_child_path(oid: str) -> str:
                    rel = oid[len(child_prefix) :]
                    return f"{comp_path}.{rel.split('/', 1)[0]}"

                _filtered_output_child_ids = [
                    oid
                    for oid in output_child_ids
                    if _direct_child_path(oid) in _known_real_outputs
                ]
            else:
                _filtered_output_child_ids = output_child_ids

            # Always create output node(s) inside the module. A composite
            # whose real forward() returns MULTIPLE distinct values (a
            # tuple — e.g. Glm5NextTextHyperConnection returning `(post,
            # comb, collapsed_hidden_states)`, three DIFFERENT shapes)
            # gets one @output PORT NODE per return value, each with
            # exactly the ONE real incoming edge (and, once shape
            # propagation runs below, the one real shape) that belongs
            # to it — mirroring how the AST backend splits a composite's
            # multi-value output (`_port_output_id`) rather than merging
            # distinct-shaped tensors into a single node whose one
            # declared shape can't represent all of them. The
            # input/output list for an expandable (composite) node must
            # match its actual edges, port for port — a single merged
            # node with 3 incoming edges but 1 declared shape doesn't.
            output_id = comp_id + "/@output"
            multi_port = len(_filtered_output_child_ids) > 1
            port_output_ids = [
                f"{output_id}:{idx}" if multi_port else output_id
                for idx in range(len(_filtered_output_child_ids))
            ] or [output_id]

            for port_id, oid in zip(port_output_ids, _filtered_output_child_ids):
                port_node = {
                    "id": port_id,
                    "label": "Output",
                    "namespace": children_ns,
                    "attrs": [{"key": "synthetic", "value": "output"}],
                    "style": _STYLE_OUTPUT,
                    "incomingEdges": [{"sourceNodeId": oid}],
                }
                nodes.append(port_node)
                node_by_id[port_id] = port_node
            if not _filtered_output_child_ids:
                # No known-real output child (e.g. attn_hc's real return
                # is untracked entirely) — still create an empty @output
                # boundary so downstream consumers have somewhere to be
                # rewired onto (with correct shape metadata from the
                # composite's own captured hook shape, set later).
                output_node = {
                    "id": output_id,
                    "label": "Output",
                    "namespace": children_ns,
                    "attrs": [{"key": "synthetic", "value": "output"}],
                    "style": _STYLE_OUTPUT,
                    "incomingEdges": [],
                }
                nodes.append(output_node)
                node_by_id[output_id] = output_node

            child_to_port = dict(zip(_filtered_output_child_ids, port_output_ids))
            # Fallback target for (a) a child that (rarely) has a real
            # external consumer despite not being among the known real
            # outputs, and (b) any STALE reference to the old bare
            # `output_id` placeholder — an earlier pass (`_resolve_
            # predecessor`/`_find_last_node_for`) may have pointed a
            # real successor straight at `comp_id + "/@output"` *before*
            # this composite-boundary pass ran, back when a single
            # merged node was going to be created there.
            default_wire_id = port_output_ids[-1]
            if multi_port:
                _candidates = all_output_shapes.get(comp_path)
                _picked_shape = _pick_output_shape(comp_path)
                if _picked_shape and _candidates and _picked_shape in _candidates:
                    default_wire_id = port_output_ids[_candidates.index(_picked_shape)]
                for n2 in nodes:
                    for e2 in n2.get("incomingEdges", []):
                        if e2["sourceNodeId"] != output_id or n2["id"].startswith(
                            child_prefix
                        ):
                            continue
                        # Prefer matching THIS specific consumer's own
                        # captured input shape directly against each
                        # port's real shape — more robust than
                        # `_pick_output_shape`'s `call_graph`-derived
                        # successor lookup, which needs a populated
                        # sequential-fallback entry that may not exist
                        # (e.g. when real tensor-identity tracking
                        # already resolved everything else and never
                        # recorded a call_graph edge for this pair).
                        wire_id = default_wire_id
                        if _candidates:
                            succ_in = input_shapes.get(
                                n2["id"].replace("/", ".")
                            )
                            if succ_in and succ_in in _candidates:
                                wire_id = port_output_ids[_candidates.index(succ_in)]
                        e2["sourceNodeId"] = wire_id

            # Rewire consumers: nodes outside this module that consumed
            # ANY child node should now consume the composite's @output
            # — its own specific port, for a known real-output child.
            # This prevents external nodes from bypassing the composite
            # boundary (e.g. after a child composite's @output was created
            # in a prior iteration and an external node was wired to it).
            for cn in child_nodes:
                cid = cn["id"]
                wire_id = child_to_port.get(cid, default_wire_id)
                if cid in consumers_of:
                    for consumer_node, edge in consumers_of[cid]:
                        if (
                            not consumer_node["id"].startswith(child_prefix)
                            and edge["sourceNodeId"] == cid
                        ):
                            edge["sourceNodeId"] = wire_id
            # Also check alias consumers
            for alias_id, target_id in _fx_leaf_aliases.items():
                if target_id.startswith(child_prefix) and alias_id in consumers_of:
                    wire_id = child_to_port.get(target_id, default_wire_id)
                    for consumer_node, edge in consumers_of[alias_id]:
                        if (
                            not consumer_node["id"].startswith(child_prefix)
                            and edge["sourceNodeId"] == alias_id
                        ):
                            edge["sourceNodeId"] = wire_id

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

        # Output node(s): a leaf whose real forward() returns MULTIPLE
        # distinct values (ground truth from `fx_output_ids`, e.g.
        # Glm5NextTextTopkRouter's `return router_logits, topk_weights,
        # topk_indices`) gets one @output PORT NODE per return value —
        # same rationale/mechanism as the composite case above: a
        # single merged node with one declared shape can't represent
        # several genuinely different per-port shapes, and the
        # input/output list for an expandable node must match its
        # actual edges. Fall back to the single last-FX-op node when
        # there's no multi-return ground truth (e.g. tracing failed to
        # capture the output node's args, or there's truly one return
        # value).
        _fx_out_ids = [
            oid for oid in fx_output_ids.get(path, []) if oid in node_by_id
        ]
        output_id = _node_id(path) + "/@output"
        if len(_fx_out_ids) > 1:
            port_ids = [f"{output_id}:{idx}" for idx in range(len(_fx_out_ids))]
            for port_id, oid in zip(port_ids, _fx_out_ids):
                port_node = {
                    "id": port_id,
                    "label": "Output",
                    "namespace": fx_ns,
                    "attrs": [{"key": "synthetic", "value": "output"}],
                    "style": _STYLE_OUTPUT,
                    "incomingEdges": [{"sourceNodeId": oid}],
                }
                nodes.append(port_node)
                node_by_id[port_id] = port_node
            child_to_port = dict(zip(_fx_out_ids, port_ids))
            _candidates = all_output_shapes.get(path)
            _picked_shape = _pick_output_shape(path)
            default_port_id = port_ids[-1]
            if _picked_shape and _candidates and _picked_shape in _candidates:
                default_port_id = port_ids[_candidates.index(_picked_shape)]
            for oid, port_id in zip(_fx_out_ids, port_ids):
                if oid in consumers_of:
                    for consumer_node, edge in consumers_of[oid]:
                        if (
                            not consumer_node["id"].startswith(prefix)
                            and edge["sourceNodeId"] == oid
                        ):
                            edge["sourceNodeId"] = port_id
            # Any stale reference to the bare placeholder id (from an
            # earlier pass that pointed a successor straight at
            # `_node_id(path) + "/@output"` before this ran) or to the
            # module's own alias id: prefer matching THIS specific
            # consumer's own captured input shape directly against each
            # port's real shape (more robust than `_pick_output_shape`'s
            # `call_graph`-derived successor lookup, which needs a
            # populated sequential-fallback entry that may not exist),
            # falling back to the ground-truth-matched default port.
            orig_id = _node_id(path)
            for stale_src in (output_id, orig_id):
                if stale_src in consumers_of:
                    for consumer_node, edge in consumers_of[stale_src]:
                        if (
                            not consumer_node["id"].startswith(prefix)
                            and edge["sourceNodeId"] == stale_src
                        ):
                            wire_id = default_port_id
                            if _candidates:
                                succ_in = input_shapes.get(
                                    consumer_node["id"].replace("/", ".")
                                )
                                if succ_in and succ_in in _candidates:
                                    wire_id = port_ids[_candidates.index(succ_in)]
                            edge["sourceNodeId"] = wire_id
        else:
            # find the last FX op(s)
            last_id = fx_nodes[-1]["id"]
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
        rep_shape = _pick_output_shape(rep_path)
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

    # ── Normalize edge output/input ids ─────────────────────────────────
    # Many edges are built throughout this function as bare
    # {"sourceNodeId": ...} dicts (composite boundary wiring, sequential
    # fallback, fx_child_edges, etc.) without `sourceNodeOutputId` /
    # `targetNodeInputId`. The viewer looks up
    # `sourceNode.outputsMetadata[edge.sourceNodeOutputId]` to render the
    # tensor shape on an edge/tooltip — when that id is missing, the
    # lookup fails and the shape displays as "?" even though the source
    # node's own outputsMetadata is well-defined. Default to single-output
    # convention ("0") and a positional target input id everywhere.
    for n in nodes:
        for idx, e in enumerate(n.get("incomingEdges", [])):
            e.setdefault("sourceNodeOutputId", "0")
            e.setdefault("targetNodeInputId", str(idx))

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
            shape_str = _pick_output_shape(check)
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
    #
    # Skip PORT-specific @output nodes (id ends in ":<idx>", from a
    # multi-value composite return split into one node per port above):
    # `_pick_output_shape` only knows the composite's OWN (single) captured
    # hook shape, which is at most one of several genuinely different
    # per-port shapes — applying it to every port would blow away the
    # correct, per-port shape the fixed-point inheritance loop below
    # would otherwise pull from each port's own (single) source edge.
    for n in nodes:
        attrs = {a["key"]: a["value"] for a in n.get("attrs", [])}
        if attrs.get("synthetic") != "output":
            continue
        if re.search(r"/@output:\d+$", n["id"]):
            continue
        path = n["id"].replace("/", ".").removesuffix(".@output")
        out_str = _pick_output_shape(path)
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
    real_output_children: dict[str, set[str]] | None = None,
) -> None:
    """Wire edges between nodes that don't already have incoming edges.

    Uses call_graph (captured during forward pass) to wire dataflow-aware
    edges between child modules of composite modules, falling back to
    sequential wiring when call_graph data is unavailable.

    For parallel branches (multiple layer types in the same container),
    all branches fan out from the same predecessor instead of being
    chained sequentially, and all branches fan back in to whatever
    consumes them next — with no separate Fork/Join wiring node.
    """
    node_ids = {n["id"] for n in nodes}
    node_by_id: dict[str, dict] = {n["id"]: n for n in nodes}
    parallel_ns_groups = parallel_ns_groups or []
    call_graph = call_graph or {}
    fx_leaf_aliases = fx_leaf_aliases or {}
    real_output_children = real_output_children or {}

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

    def _redirect_from_untracked_output(source_id: str) -> str | None:
        """If `source_id` sits inside a descendant subtree of some
        composite whose real output children are known (ground truth
        from tensor-identity tracking, see `_capture_call_graph`) and
        are known to NOT include that subtree, redirect to that
        composite's own (not-yet-created) "@output" boundary instead —
        mirrors the same redirect in `_find_last_node_for` inside
        `build_graph` (see its docstring for the motivating
        Glm5NextTextHyperConnection/attn_hc example). Returns None
        (no redirect) if `source_id` isn't affected.

        Only meant to veto a *positional guess* (callers only consult
        this when no call-graph-derived source was found) — an empty
        ground-truth set does NOT mean "definitely wrong" in general
        (e.g. a residual-add composite legitimately has no single
        child whose tensor IS its output either), so this must stay
        scoped to the weak positional-guess path only — anything with
        real, reliable dataflow evidence (an actual call-graph edge, a
        genuine outside consumer) is handled elsewhere and must never
        be overridden by this heuristic.
        """
        path = source_id.replace("/", ".")
        parts = path.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            anc = ".".join(parts[:depth])
            known = real_output_children.get(anc)
            if known is not None:
                direct_child = ".".join(parts[: depth + 1])
                if direct_child not in known:
                    return anc.replace(".", "/") + "/@output"
        return None

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

    def _resolve_nested_input(comp_path: str, _depth: int = 0) -> list[str]:
        """Resolve the real predecessor(s) of ``comp_path`` even when its
        own call-graph entry is just "@input" relative to its immediate
        parent — which itself may only be "@input" relative to ITS
        parent, and so on (e.g. a gate submodule nested inside an
        attention module, both of which receive the composite's own
        input directly, with no transforming sibling in between).

        Without this recursion, a nested child whose ONLY known source
        is "@input" (skipped by ``_cg_predecessor``, which only tracks
        *real* sibling predecessors) would have no known entry point and
        would silently fall back to whatever ran immediately before it
        in the arbitrary node list order — e.g. wiring a gate module
        from an unrelated sibling projection just because it happened to
        run first, rather than from the shared, genuine input.
        """
        if _depth > 20 or not comp_path:
            return []
        sources: list[str] = []
        if comp_path in _cg_has_root_input:
            sources.append("@input")
        if comp_path in _cg_predecessor:
            for pred_path in _cg_predecessor[comp_path]:
                if pred_path in last_node_for_path:
                    sources.append(last_node_for_path[pred_path])
        if sources:
            return sources
        if comp_path in composite_entry:
            return [composite_entry[comp_path]]
        if "." in comp_path and cg_sources.get(comp_path) == ["@input"]:
            parent_path = comp_path.rsplit(".", 1)[0]
            return _resolve_nested_input(parent_path, _depth + 1)
        return []

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
                # point, then (recursively) a grandparent's resolved
                # entry when this composite's own source is itself just
                # "@input", then the sequential fallback.
                comp_path = child_path.rsplit(".", 1)[0]
                input_sources: list[str] = _resolve_nested_input(comp_path)
                input_source = input_sources[0] if input_sources else None
                if not input_source:
                    input_source = fallback_source

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
            # First time entering this parallel branch — all branches
            # fan out from the same predecessor (no Fork node).
            if not any(
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
            # Not in a parallel branch. Check if we just left one — if
            # so, fan in directly from every branch's last node (no
            # Join node — matches the AST backend's rendering, where an
            # interleaved layer-type stack's tail simply gets multiple
            # incomingEdges, one per variant).
            fan_in_sources = []
            for group in parallel_ns_groups:
                if group <= parallel_entered:
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
        # IMPORTANT: resolve via the call graph BEFORE recording this
        # node as its ancestors' entry point. `_record_composite_entry`
        # is what lets a LATER sibling (e.g. a nested gate submodule
        # that only receives the composite's own "@input", recursively
        # resolved by `_cg_find_sources`) look up "what did this
        # composite's first node actually connect from?". If we recorded
        # the naive pre-correction `source_id` (the arbitrary sequential
        # guess) first, that wrong guess would get locked in as the
        # composite's entry point and poison every later sibling's
        # resolution before the call-graph override ever had a chance to
        # run — even though THIS node's own edge gets corrected fine.
        cg_srcs = _cg_find_sources(node["id"], source_id)
        source_is_weak_guess = cg_srcs is None
        if cg_srcs is not None:
            if len(cg_srcs) == 1:
                source_id = cg_srcs[0]
                _record_composite_entry(node["id"], source_id)
            elif len(cg_srcs) > 1:
                # Multiple dataflow predecessors
                _record_composite_entry(node["id"], cg_srcs[0])
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
        else:
            _record_composite_entry(node["id"], source_id)

        redirected_to_future_boundary = False
        if source_id and source_is_weak_guess:
            redirected = _redirect_from_untracked_output(source_id)
            if redirected:
                source_id = redirected
                redirected_to_future_boundary = True

        if source_id and (redirected_to_future_boundary or source_id in node_by_id):
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
