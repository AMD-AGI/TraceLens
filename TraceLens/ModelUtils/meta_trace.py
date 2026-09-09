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
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


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
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        _log.info(
            "torch and/or transformers not installed; "
            "skipping meta-device shape tracing"
        )
        return None

    # ---- load config -------------------------------------------------------
    try:
        hf_config = AutoConfig.from_pretrained(
            str(checkpoint), trust_remote_code=True
        )
    except Exception as exc:
        _log.warning("Could not load config for meta tracing: %s", exc)
        return None

    # ---- instantiate on meta device ----------------------------------------
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                hf_config, trust_remote_code=True
            )
        model.eval()
    except Exception as exc:
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
