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
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Distinctive probe dims for the per-op FX pass — primes, distinct from 1, so a
# materialized op's output dims don't collide with a config dim (e.g. head_dim)
# and get misread as batch/sequence when symbolising.
_OP_PROBE_BATCH = 2
_OP_PROBE_SEQ = 137


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
    try:
        from TraceLens.ModelUtils.torch_trace import (
            _instantiate_meta,
            _patch_rotary_embeddings,
        )

        model, _config = _instantiate_meta(checkpoint)
        _patch_rotary_embeddings(model)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not instantiate model on meta device: %s", exc)
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

    try:
        from TraceLens.ModelUtils.torch_trace import (
            _fx_trace_module,
            _instantiate_meta,
            _patch_rotary_embeddings,
            _propagate_fx_node_shapes,
        )

        model, _config = _instantiate_meta(checkpoint)
        _patch_rotary_embeddings(model)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not instantiate model on meta device: %s", exc)
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
