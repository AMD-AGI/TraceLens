###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Meta-device instantiation and per-module FX shape helpers.

These utilities support shape inference for the AST-based Model Explorer
pipeline (``shape_inference`` / ``meta_trace``): they instantiate a model on
the PyTorch ``meta`` device (zero memory, no weights) and symbolically trace
individual submodules to recover ground-truth tensor shapes.

The former PyTorch *figure-generation* backend (``build_graph`` and its
node/edge/boundary machinery) has been removed; the AST backend is now the
only graph builder. Only the tracing/shape helpers it depends on remain here.
"""

from __future__ import annotations

import logging
import re
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.fx
import transformers
from transformers import AutoConfig

_log = logging.getLogger(__name__)


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


# ── Per-module FX tracing + shape propagation ────────────────────────────────


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
