###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Optional meta-device shape tracing for ground-truth tensor shapes.

When ``torch`` and ``transformers`` are installed, this module can instantiate a
model on the PyTorch ``meta`` device (zero memory, no weights loaded) and run a
dummy forward pass to capture the real output shape of every submodule.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Distinctive probe dims for the per-op FX pass — primes, distinct from 1, so a
# materialized op's output dims don't collide with a config dim (e.g. head_dim)
# and get misread as batch/sequence when symbolising.
_OP_PROBE_BATCH = 2
_OP_PROBE_SEQ = 137


# ── Robust meta-device instantiation ─────────────────────────────────────────

# Bound on the config-attribute repair loop: a model whose modeling code demands
# a fresh missing attribute on every access can't spin here forever.
_MAX_CONFIG_ATTR_REPAIRS = 64
_MISSING_ATTR_RE = re.compile(r"object has no attribute '([^']+)'")


def _repair_missing_config_attr(config: Any, attr_name: str) -> bool:
    """Give a neutral default to a config attribute the modeling code demands
    but that the loaded config (and its sub-configs) do not define.

    Generic — the attribute name is parsed out of the raised ``AttributeError``,
    never hardcoded. If any sub-config already defines the attribute its real
    value is reused (so a genuine vision/text setting is preserved); otherwise a
    neutral ``1`` is used, a value that is safe wherever such an attribute is
    read as a size / stride / patch factor and that cannot divide-by-zero. The
    default is written on the top config and on every sub-config that lacks it,
    because the failing access may be on either. Returns *True* if anything was
    patched.
    """
    subconfigs = [v for v in vars(config).values() if hasattr(v, "to_dict")]
    value: Any = 1
    for candidate in [config, *subconfigs]:
        if hasattr(candidate, attr_name):
            value = getattr(candidate, attr_name)
            break
    patched = False
    for target in [config, *subconfigs]:
        if not hasattr(target, attr_name):
            setattr(target, attr_name, value)
            patched = True
    return patched


def _instantiate_meta_robust(checkpoint: str | Path) -> tuple[Any, Any] | None:
    """Instantiate the model on the ``meta`` device, tolerating the two common
    ways an otherwise-fine model kills the shape-inference backup.

    The torch backend's :func:`_instantiate_meta` lets an ``AttributeError`` from
    a config key the modeling code expects but that is absent (e.g. a VLM whose
    ``temporal_patch_size`` lives in neither the top nor a sub-config), and any
    ``ImportError`` from a missing *optional* dependency (e.g. ``einops``),
    propagate — taking the whole backup down. This wrapper instead:

    * retries instantiation, filling a neutral default for each missing config
      attribute in turn (generically, by parsing the name out of the error —
      no hardcoded key), and
    * degrades to *None* (no raise) for a genuinely uninstantiable model, so the
      caller simply falls back to whatever partial shape info it already has.

    It reuses the importable :func:`_patch_config` / :func:`_resolve_auto_classes`
    so the Auto-class selection stays identical to :func:`_instantiate_meta`.
    Returns ``(model, config)`` or *None*.
    """
    try:
        import torch
        from transformers import AutoConfig

        from TraceLens.ModelUtils.torch_trace import (
            _instantiate_meta,
            _patch_config,
            _resolve_auto_classes,
        )
    except Exception as exc:  # noqa: BLE001
        _log.info("meta-instantiation prerequisites unavailable: %s", exc)
        return None

    # Happy path: delegate to the canonical instantiator, which handles the
    # common case and whose Auto-class selection we want to match exactly. Only
    # when it raises do we attempt the generic config-attribute repair below, so
    # a model that already instantiates is completely untouched.
    try:
        return _instantiate_meta(checkpoint)
    except Exception as first_exc:  # noqa: BLE001
        _log.info(
            "Standard meta instantiation of %s failed (%s); attempting robust repair",
            checkpoint,
            first_exc,
        )

    try:
        config = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=True)
        _patch_config(config)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not load config for %s: %s", checkpoint, exc)
        return None

    auto_classes = _resolve_auto_classes(config)
    repaired: set[str] = set()

    for _attempt in range(_MAX_CONFIG_ATTR_REPAIRS + 1):
        attr_err: Exception | None = None
        other_err: Exception | None = None
        for auto_cls in auto_classes:
            try:
                with torch.device("meta"):
                    model = auto_cls.from_config(config, trust_remote_code=True)
                model.eval()
                return model, config
            except AttributeError as exc:
                # A missing config attribute — potentially repairable; keep
                # trying the remaining auto classes first in case one avoids it.
                attr_err = exc
                continue
            except (ValueError, KeyError) as exc:
                other_err = exc
                continue
            except Exception as exc:  # noqa: BLE001
                # Missing optional dependency or any other non-repairable
                # failure: this model is genuinely uninstantiable here.
                _log.warning(
                    "Could not instantiate %s on meta device: %s", checkpoint, exc
                )
                return None

        # No auto class succeeded this pass.
        if attr_err is None:
            _log.warning(
                "Could not instantiate %s on meta device: %s",
                checkpoint,
                other_err,
            )
            return None
        match = _MISSING_ATTR_RE.search(str(attr_err))
        if match is None:
            _log.warning(
                "Could not instantiate %s on meta device: %s", checkpoint, attr_err
            )
            return None
        attr_name = match.group(1)
        if attr_name in repaired or not _repair_missing_config_attr(
            config, attr_name
        ):
            _log.warning(
                "Could not repair missing config attribute %r for %s: %s",
                attr_name,
                checkpoint,
                attr_err,
            )
            return None
        repaired.add(attr_name)
        _log.info(
            "Set a neutral default for missing config attribute %r on %s; retrying",
            attr_name,
            checkpoint,
        )

    _log.warning(
        "Exhausted config-attribute repairs instantiating %s on meta device",
        checkpoint,
    )
    return None


@dataclass(frozen=True)
class MetaModuleGroup:
    """A repeated ``nn.ModuleList`` read structurally off the instantiated tree.

    ``signatures`` holds one structural signature per element (ordered tuple of the
    element's immediate child class names), so callers can bucket the elements into
    sub-variants with ``collections.Counter(signatures)``.
    """

    path: str
    length: int
    element_class: str
    signatures: tuple[str, ...]


def _element_signature(element: Any) -> str:
    """Structural signature of a ModuleList element.

    The ordered tuple of the element's immediate child module class names, rendered
    as a stable string. This separates e.g. an MoE-bearing decoder layer from a dense
    one (different ``mlp`` child class) without descending the whole subtree. Tuning
    knob: deepen one level if this under-splits a model's variants.
    """
    child_classes = [type(child).__name__ for _, child in element.named_children()]
    return "(" + ",".join(child_classes) + ")"


def walk_meta_module_tree(checkpoint: str | Path) -> list[MetaModuleGroup] | None:
    """Read the repeated-``ModuleList`` structure off the instantiated meta tree.

    Instantiates the model on the ``meta`` device and walks ``named_modules()`` for
    every non-empty ``nn.ModuleList``, recording its path, ``len()``, element class
    name, and per-element structural signature. Returns *None* when torch /
    transformers are unavailable or the model cannot be instantiated.

    Unlike :func:`trace_meta_shapes` this runs **no forward pass** and applies **no
    rotary patch**: building the module tree on the meta device allocates no storage,
    needs no GPU, and never touches the data-dependent ops (``.item()``/``.tolist()``,
    rotary CPU tensors, ``ACT2FN[...]``) that make meta *forwards* fragile. Reading
    structure needs only the reliable instantiation half.
    """
    try:
        import torch
    except ImportError:
        _log.info(
            "torch and/or transformers not installed; "
            "skipping meta-device module-tree walk"
        )
        return None

    # Instantiate our own clean model — no shared instance with the forward-based
    # tracers, so an in-place rotary patch there can never contaminate this walk.
    result = _instantiate_meta_robust(checkpoint)
    if result is None:
        return None
    model, _config = result

    try:
        groups: list[MetaModuleGroup] = []
        for path, mod in model.named_modules():
            if not isinstance(mod, torch.nn.ModuleList) or len(mod) == 0:
                continue
            groups.append(
                MetaModuleGroup(
                    path=path,
                    length=len(mod),
                    element_class=type(mod[0]).__name__,
                    signatures=tuple(_element_signature(element) for element in mod),
                )
            )
    finally:
        del model

    return groups


@dataclass(frozen=True)
class MetaTensorSpec:
    """Shape + dtype of an ``nn.Parameter`` / buffer read off the meta tree."""

    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class MetaTensorIndex:
    """Meta parameter/buffer tensors indexed for constant-operand resolution.

    ``by_qualified`` keys the fully-qualified name (``visual.rotary_pos_emb.inv_freq``)
    and ``by_class_attr`` keys ``(owner_class_name, leaf_attr)`` (``("Glm5NextVisionRotaryEmbedding", "inv_freq")``),
    mirroring how :meth:`ShapeInferencer._lookup_parameter_spec` resolves a
    parameter — but adding dtype and covering buffers, not just parameters.
    """

    by_qualified: dict[str, MetaTensorSpec]
    by_class_attr: dict[tuple[str, str], MetaTensorSpec]


def harvest_meta_tensors(checkpoint: str | Path) -> MetaTensorIndex | None:
    """Index every parameter and buffer of the meta-instantiated model.

    Instantiates the model on the ``meta`` device (zero memory, no weights) and
    walks ``named_parameters()`` + ``named_buffers()``, recording each tensor's
    shape and dtype. Runs **no forward pass** and applies **no rotary patch** —
    like :func:`walk_meta_module_tree`, reading tensor metadata needs only the
    reliable instantiation half. Returns *None* when torch/transformers are
    unavailable or the model cannot be instantiated.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        _log.info(
            "torch and/or transformers not installed; skipping meta-tensor harvest"
        )
        return None

    result = _instantiate_meta_robust(checkpoint)
    if result is None:
        return None
    model, _config = result

    def _dtype_name(tensor: Any) -> str:
        return str(tensor.dtype).replace("torch.", "")

    try:
        # Map each owning module path to its class so a qualified tensor name
        # (``visual.rotary_pos_emb.inv_freq``) can be keyed by ``(class, leaf)``.
        module_class: dict[str, str] = {
            path: type(mod).__name__ for path, mod in model.named_modules()
        }
        by_qualified: dict[str, MetaTensorSpec] = {}
        by_class_attr: dict[tuple[str, str], MetaTensorSpec] = {}
        for name, tensor in list(model.named_parameters()) + list(
            model.named_buffers()
        ):
            spec = MetaTensorSpec(
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=_dtype_name(tensor),
            )
            by_qualified[name] = spec
            owner_path, _, leaf = name.rpartition(".")
            owner_class = module_class.get(owner_path)
            if owner_class:
                by_class_attr.setdefault((owner_class, leaf), spec)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Meta-tensor harvest failed: %s", exc)
        return None
    finally:
        del model

    return MetaTensorIndex(by_qualified=by_qualified, by_class_attr=by_class_attr)


def trace_meta_shapes(
    checkpoint: str | Path,
    config: dict[str, Any] | None = None,
    *,
    seq_len: int = 128,
    batch_size: int = 1,
) -> dict[str, tuple[int, ...]] | None:
    """Run a forward pass on the ``meta`` device and return per-module output shapes.

    Parameters
    ----------
    checkpoint:
        Hugging Face model id or local path (used to resolve ``AutoConfig``).
    config:
        Pre-loaded config dict.  When *None*, loaded from *checkpoint*.
    seq_len:
        Sequence length for the dummy input tokens.
    batch_size:
        Batch size for the dummy input.

    Returns
    -------
    dict mapping ``model.named_modules()`` paths (e.g.
    ``"model.layers.0.self_attn.q_proj"``) to output shape tuples, or *None*
    when torch / transformers are not available or the model cannot be
    instantiated.
    """
    try:
        import torch
    except ImportError:
        _log.info(
            "torch and/or transformers not installed; "
            "skipping meta-device shape tracing"
        )
        return None

    # ---- instantiate on meta device ----------------------------------------
    # Reuse the torch backend's robust instantiation: it inspects the model
    # card (auto_map / architectures) to pick the right Auto class, so it also
    # handles conditional-generation / VLM configs that AutoModelForCausalLM
    # rejects (e.g. Glm5NextForConditionalGeneration).
    result = _instantiate_meta_robust(checkpoint)
    if result is None:
        return None
    model, _config = result
    try:
        from TraceLens.ModelUtils.torch_trace import _patch_rotary_embeddings

        _patch_rotary_embeddings(model)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not patch rotary embeddings on meta device: %s", exc)
        return None

    # ---- register hooks ----------------------------------------------------
    shapes: dict[str, tuple[int, ...]] = {}

    def _make_hook(name: str):
        def hook(_module, _input, output):
            try:
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

    # ---- run forward pass --------------------------------------------------
    try:
        dummy = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device="meta"
        )
        with torch.no_grad():
            model(dummy)
    except Exception:
        # Meta tensors fail on data-dependent ops (nonzero, item, etc.).
        # Hooks fire *before* the failing op, so we keep partial results.
        pass
    finally:
        for handle in handles:
            handle.remove()

    if shapes:
        _log.info(
            "Meta-device tracing captured shapes for %d modules", len(shapes)
        )
    else:
        _log.warning("Meta-device tracing captured no shapes")
        return None

    return shapes


def trace_meta_input_specs(
    checkpoint: str | Path,
    config: dict[str, Any] | None = None,
    *,
    seq_len: int = 130,
    batch_size: int = 2,
) -> (
    tuple[
        dict[str, tuple[tuple[Any, ...], str]],
        dict[tuple[str, str], tuple[tuple[Any, ...], str]],
    ]
    | None
):
    """Run a meta forward and return forward-parameter *input* shapes.

    Returns a pair ``(global_specs, class_specs)``:

    * ``global_specs`` maps ``param_name -> (symbolised_shape, dtype_str)`` for
      every forward parameter whose captured ``(shape, dtype)`` is **identical
      across every module that receives it**. Any parameter observed with
      conflicting shapes (e.g. ``hidden_states`` at several widths, or
      ``position_ids`` present in both the text ``[B, S]`` and the vision path)
      is dropped, so an ambiguous global seed is never emitted.
    * ``class_specs`` maps ``(class_name, param_name) -> (shape, dtype)`` for
      every parameter whose shape is unambiguous **within the modules of one
      class**. This resolves a boundary whose name is globally ambiguous but
      locally definite — most notably ``attention_mask``, which is a 4-D causal
      mask on the decoder attention yet a flat ``[B, S]`` padding mask on the
      sparse-attention indexer. Keyed by the receiving module's own class, so the
      indexer's ``@input:attention_mask`` seeds ``[B, S]`` while the decoder's
      keeps its causal shape.

    ``batch_size`` and ``seq_len`` are deliberately chosen distinct from common
    structural dims (head_dim 128, num_heads 64, …) so :func:`symbolise_meta_shape`
    does not alias a feature axis onto ``B``/``S`` (seq_len 128 == head_dim 128
    would corrupt a ``[B, S, 64, 128]`` gate into ``[B, S, 64, S]``).

    Returns *None* when torch is unavailable, the model cannot be instantiated, or
    nothing was captured. Best-effort, like :func:`trace_meta_shapes`.
    """
    try:
        import torch
    except ImportError:
        return None

    result = _instantiate_meta_robust(checkpoint)
    if result is None:
        return None
    model, _config = result
    try:
        from TraceLens.ModelUtils.torch_trace import _patch_rotary_embeddings

        _patch_rotary_embeddings(model)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not patch rotary embeddings on meta device: %s", exc)
        return None

    import inspect

    # param_name -> set of (symbolised_shape, dtype_str) seen across all modules.
    observed: dict[str, set[tuple[tuple[Any, ...], str]]] = {}
    # (class_name, param_name) -> set of (shape, dtype) seen within that class.
    observed_by_class: dict[tuple[str, str], set[tuple[tuple[Any, ...], str]]] = {}

    def _record(class_name: str, param: str, tensor: Any) -> None:
        shape = symbolise_meta_shape(
            tuple(int(d) for d in tensor.shape),
            batch_size=batch_size,
            seq_len=seq_len,
        )
        dtype = str(tensor.dtype).replace("torch.", "")
        observed.setdefault(param, set()).add((shape, dtype))
        observed_by_class.setdefault((class_name, param), set()).add((shape, dtype))

    def _make_hook(module: Any):
        class_name = type(module).__name__
        try:
            params = list(inspect.signature(module.forward).parameters)
        except (TypeError, ValueError):
            params = []

        def hook(_module, args, kwargs):
            for index, value in enumerate(args):
                if isinstance(value, torch.Tensor):
                    name = params[index] if index < len(params) else f"arg{index}"
                    _record(class_name, name, value)
            for name, value in (kwargs or {}).items():
                if isinstance(value, torch.Tensor):
                    _record(class_name, name, value)

        return hook

    handles = [
        mod.register_forward_pre_hook(_make_hook(mod), with_kwargs=True)
        for _name, mod in model.named_modules()
    ]
    try:
        dummy = torch.zeros(batch_size, seq_len, dtype=torch.long, device="meta")
        with torch.no_grad():
            model(dummy)
    except Exception:
        # Meta tensors fail on data-dependent ops; pre-hooks already fired for
        # every module reached before the failure (incl. the sparse indexer).
        pass
    finally:
        for handle in handles:
            handle.remove()

    # Keep only globally-unambiguous parameters (exactly one observed value).
    resolved = {
        param: next(iter(values))
        for param, values in observed.items()
        if len(values) == 1
    }
    # Per-class: keep parameters unambiguous within their own class.
    resolved_by_class = {
        key: next(iter(values))
        for key, values in observed_by_class.items()
        if len(values) == 1
    }
    if resolved or resolved_by_class:
        _log.info(
            "Meta-device tracing captured %d consistent input params "
            "(%d class-scoped)",
            len(resolved),
            len(resolved_by_class),
        )
        return resolved, resolved_by_class
    return None


def _norm_op(name: Any) -> str:
    """Normalize an op name for cross-source matching (AST id vs FX node)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _fx_op_base(node: Any) -> str:
    if node.op == "call_module":
        raw = str(node.target).rsplit(".", 1)[-1]
    else:
        raw = re.sub(r"_\d+$", "", node.name)
    return _norm_op(raw)


_STACK_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')


def _fx_source_line(node: Any, source_file: str | None) -> int | None:
    """Source line of an FX node's own frame, but only when that frame is in
    ``source_file`` — the file where the module's class (and thus the AST
    ``@op_l{line}...`` ids we join against) is defined.

    This deliberately rejects torch-internal frames (e.g. ``nn.Linear.forward``
    in torch's ``linear.py``) and recompiled/patched forwards
    (``<shape_unpack_patch>``, ``<mirror>``), whose line numbers would not line
    up with the modeling source and could false-match a real op on that line.
    """
    if not source_file:
        return None
    st = node.meta.get("stack_trace")
    if not st:
        return None
    match = _STACK_FRAME_RE.search(st)
    if not match:
        return None
    path, line = match.group(1), match.group(2)
    if path != source_file:
        return None
    return int(line)


# Sentinel: a (line, op, occurrence) key seen with two different shapes across
# modules is ambiguous — we drop it rather than guess.
_CONFLICT = object()


def trace_meta_op_shapes(
    checkpoint: str | Path,
    config: dict[str, Any] | None = None,
    *,
    seq_len: int = _OP_PROBE_SEQ,
    batch_size: int = _OP_PROBE_BATCH,
) -> dict[tuple[int, str, int], tuple[str | int, ...]] | None:
    """Ground-truth per-op output shapes via ``torch.fx`` + ``ShapeProp``.

    For every distinct submodule class actually invoked in a dummy meta-device
    forward, symbolically trace it, propagate the module's real captured input
    shape through the graph, and record each op node's true output shape keyed
    by ``(source_line, normalized_op_name, occurrence_on_that_line)``.

    The source line comes from the FX node's recorded stack trace, so it lines
    up with the ``@op_l{line}_..._{name}`` ids the AST graph builder emits —
    giving a robust join that needs no fragile class-name/path matching. Keys
    that resolve to conflicting shapes across modules are dropped.

    Returns *None* when torch is unavailable, the model can't be instantiated,
    or nothing usable was captured. Best-effort: this is a last-resort fallback
    for ops with no symbolic rule.
    """
    try:
        import torch
    except ImportError:
        return None

    result = _instantiate_meta_robust(checkpoint)
    if result is None:
        return None
    model, _config = result
    try:
        from TraceLens.ModelUtils.torch_trace import (
            _fx_trace_module,
            _patch_rotary_embeddings,
            _propagate_fx_node_shapes,
        )

        _patch_rotary_embeddings(model)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not patch rotary embeddings on meta device: %s", exc)
        return None

    # ---- capture each module's real input shape ----------------------------
    input_shapes: dict[str, tuple[int, ...]] = {}

    def _make_pre_hook(name: str):
        def hook(_module, inputs):
            for item in inputs:
                if isinstance(item, torch.Tensor):
                    input_shapes.setdefault(
                        name, tuple(int(d) for d in item.shape)
                    )
                    break
        return hook

    modules = dict(model.named_modules())
    handles = [
        mod.register_forward_pre_hook(_make_pre_hook(name))
        for name, mod in modules.items()
    ]
    try:
        dummy = torch.zeros(batch_size, seq_len, dtype=torch.long, device="meta")
        with torch.no_grad():
            model(dummy)
    except Exception:
        # Meta tensors fail on data-dependent ops; pre-hooks already fired.
        pass
    finally:
        for handle in handles:
            handle.remove()

    model_dtype = next(
        (p.dtype for p in model.parameters() if p.dtype.is_floating_point), None
    )

    # ---- FX-trace one representative per class, propagate shapes ------------
    import inspect

    tracer_base = torch.fx.proxy.TracerBase
    prev_record = getattr(tracer_base, "record_stack_traces", False)
    result: dict[tuple[int, str, int], Any] = {}
    seen_classes: set[str] = set()
    try:
        tracer_base.record_stack_traces = True
        for path, in_shape in input_shapes.items():
            mod = modules.get(path)
            if mod is None:
                continue
            cls = type(mod).__name__
            if cls in seen_classes:
                continue
            try:
                source_file = inspect.getsourcefile(type(mod))
            except (TypeError, OSError):
                source_file = None
            if source_file is None:
                continue
            try:
                graph = _fx_trace_module(mod)
            except Exception:  # noqa: BLE001
                graph = None
            if graph is None:
                continue
            node_metas = _propagate_fx_node_shapes(
                mod, graph, in_shape, model_dtype=model_dtype
            )
            if not node_metas:
                continue
            seen_classes.add(cls)
            occurrence: dict[tuple[int, str], int] = {}
            for node in graph.nodes:
                if node.op not in ("call_function", "call_method", "call_module"):
                    continue
                line = _fx_source_line(node, source_file)
                if line is None:
                    continue
                entry = node_metas.get(node.name)
                if entry is None:
                    continue
                base = _fx_op_base(node)
                idx = occurrence.get((line, base), 0)
                occurrence[(line, base)] = idx + 1
                sym = symbolise_meta_shape(
                    entry[0], batch_size=batch_size, seq_len=seq_len
                )
                key = (line, base, idx)
                if key in result and result[key] != sym:
                    result[key] = _CONFLICT
                elif key not in result:
                    result[key] = sym
    finally:
        tracer_base.record_stack_traces = prev_record

    clean = {k: v for k, v in result.items() if v is not _CONFLICT}
    if clean:
        _log.info("FX per-op tracing captured %d op shapes", len(clean))
        return clean
    return None


def symbolise_meta_shape(
    shape: tuple[int, ...],
    *,
    batch_size: int = 1,
    seq_len: int = 128,
) -> tuple[str | int, ...]:
    """Replace concrete batch and sequence dims with symbolic ``B`` / ``S``."""
    result: list[str | int] = []
    for dim in shape:
        if dim == batch_size:
            result.append("B")
        elif dim == seq_len:
            result.append("S")
        else:
            result.append(dim)
    return tuple(result)
