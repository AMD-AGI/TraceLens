###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Architecture facts for Loom's ``loom-port-llm`` ``cfg.json``, from an ``ArchitectureSpec``.

``loom-port-llm`` (https://github.com/loom-fun/loom-skills/tree/main/skills/loom-port-llm)
turns a declarative ``cfg.json`` into Loom IR kernels for the dense-transformer
family (``starter/gen_llm.py``). Today that ``cfg.json`` comes from
``inspect_gguf.py``, which reverse-engineers GGUF tensor names and either
hardcodes or leaves blank several fields (``rope_type``, ``embed_scale``,
``sliding_layers``, ``nope_layers``, ...) that its own "cfg.json review
checklist" tells a human to fix by reading ``llama.cpp``'s
``src/models/<arch>.cpp``.

:class:`TraceLens.ModelUtils.extract.ArchitectureSpec` already parses the
checkpoint's real ``config.json`` and modeling source -- ground truth, no GGUF
round-trip -- so this module derives the subset of ``cfg.json`` architecture
fields that can be read from it with confidence, and leaves the rest alone
rather than guessing: ``rope_type`` is a ``llama.cpp`` GGUF weight-permutation
convention, not a property of the HF source; ``attn_gate``/``post_norms``/
``embed_scale``/``embed_norm`` need submodule/weight-name AST inspection
``extract.py`` doesn't do today; ``tensors``/``gguf``/``chat``/``eos_id``/
``bos_id`` stay owned by ``inspect_gguf.py`` / ``chat_probe.py``.
"""

from __future__ import annotations

from typing import Any

from TraceLens.ModelUtils.extract import ArchitectureSpec

# Constants

# HF's own near-universal default when a RoPE model's config omits an
# explicit theta (every HF RoPE config class defaults to this).
_DEFAULT_ROPE_THETA = 10000.0

# config.json ``layer_types`` entries that name ordinary (full or windowed)
# attention; anything else (``linear_attention``, ``gated_delta_net``, ...)
# is treated as a non-attention (recurrent) layer kind.
_ATTENTION_LAYER_TYPES = frozenset({"full_attention", "sliding_attention"})


def build_loom_arch_facts(
    spec: ArchitectureSpec, *, name: str | None = None
) -> dict[str, Any]:
    """Map *spec* to the ``cfg.json`` architecture fields TraceLens can derive.

    ``name`` should be the checkpoint id/path as the caller passed it
    (``spec.checkpoint_source`` is a resolved ``config.json`` URI/path, e.g.
    ``hf://org/repo/config.json``, not the clean display name ``cfg.json``
    wants); falls back to ``spec.name`` when omitted.

    Core geometry fields (``hidden``, ``layers``, ``heads``, ``kv_heads``,
    ``head_dim``, ``ffn``, ``vocab``, ``context_length``, ``tied_embeddings``)
    are always present (``None`` when the checkpoint's config never declared
    one). Everything else is included only when derived with confidence; a
    field this module cannot derive is simply absent, never guessed.
    """
    config = spec.raw_config
    heads = spec.num_attention_heads
    kv_heads = spec.num_key_value_heads or heads
    head_dim = spec.head_dim
    if head_dim is None and spec.hidden_size and heads:
        head_dim = spec.hidden_size // heads

    facts: dict[str, Any] = {
        "name": name or spec.name,
        "arch": spec.model_type,
        "hidden": spec.hidden_size,
        "layers": spec.num_hidden_layers,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "ffn": spec.intermediate_size,
        "vocab": spec.vocab_size,
        "context_length": spec.max_position_embeddings,
        "tied_embeddings": spec.tie_word_embeddings,
    }

    rms_eps = _cfg(config, "rms_norm_eps", "layer_norm_eps")
    if rms_eps is not None:
        facts["rms_eps"] = rms_eps

    if spec.positional_encoding == "RoPE":
        facts["rope_theta"] = _cfg(config, "rope_theta") or _DEFAULT_ROPE_THETA
        rope_dims = _rope_dims(config, head_dim)
        if rope_dims is not None:
            facts["rope_dims"] = rope_dims

    qk_norm = _as_bool(_cfg(config, "qk_norm", "use_qk_norm"))
    if qk_norm is not None:
        facts["qk_norm"] = qk_norm

    sliding_layers, sliding_window = _sliding_layers(config, spec.num_hidden_layers)
    if sliding_layers is not None:
        facts["sliding_layers"] = sliding_layers
        facts["sliding_window"] = sliding_window

    recurrent_layers = _recurrent_layers(config)
    if recurrent_layers is not None:
        facts["recurrent_layers"] = recurrent_layers

    return facts


def coverage_notes(spec: ArchitectureSpec) -> list[str]:
    """Free-text caveats: HF architecture names, and families the starter kit (or
    this module) doesn't cover, for a human to act on rather than silently drop.
    """
    notes: list[str] = []
    if spec.architectures:
        notes.append(
            f"HF architectures: {spec.architectures} -- resolve against the "
            "GGUF 'arch' string by hand if it doesn't already match."
        )
    if spec.attention_type == "MLA":
        notes.append(
            "Multi-head Latent Attention: the starter kit's attention kernels "
            "assume standard (GQA/MHA) KV; see 'Beyond the starter' in loom-port-llm."
        )
    if spec.decoder_type == "Sparse MoE":
        notes.append(
            "Sparse MoE: the starter kit covers dense FFN only; see the MoE "
            "pattern under 'Beyond the starter' in loom-port-llm."
        )
    if _recurrent_layers(spec.raw_config) is not None:
        notes.append(
            "recurrent_layers inferred from config.layer_types values outside "
            "{full_attention, sliding_attention} -- verify against gen_gdn.py's "
            "expected geometry and fill cfg.json's gdn block by hand."
        )
    if _cfg(spec.raw_config, "no_rope_layers") is not None:
        notes.append(
            "config declares 'no_rope_layers'; not auto-translated to nope_layers "
            "here -- verify its 0/1 polarity against the model's own source and "
            "fill cfg.json's nope_layers by hand."
        )
    return notes


def diff_against_existing(
    derived: dict[str, Any], existing: dict[str, Any]
) -> list[str]:
    """Compare *derived* facts against an already-authored ``cfg.json``.

    Reports a mismatch for every key *derived* states that disagrees with
    *existing*, and an addition for every key *derived* states that
    *existing* lacks. Fields this module never derives are not in *derived*
    and so are never flagged.
    """
    lines: list[str] = []
    for key, value in derived.items():
        if key not in existing:
            lines.append(f"{key}: derived {value!r}, missing from existing cfg.json")
            continue
        existing_value = existing[key]
        if _normalize(value) != _normalize(existing_value):
            lines.append(f"{key}: derived {value!r} != existing {existing_value!r}")
    return lines


def _cfg(config: dict[str, Any], *keys: str) -> Any:
    """First present, non-null value among *keys* in a raw HF config dict."""
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _normalize(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(value, key=repr)
    return value


def _rope_dims(config: dict[str, Any], head_dim: int | None) -> int | None:
    """Partial rotary embedding width, when the config states a fraction < 1."""
    factor = _cfg(config, "partial_rotary_factor")
    if not factor or not head_dim:
        return None
    rope_dims = round(float(factor) * head_dim)
    return rope_dims if rope_dims != head_dim else None


def _sliding_layers(
    config: dict[str, Any], num_layers: int | None
) -> tuple[list[int] | None, int | None]:
    """Per-layer sliding-window layer indices, and the window size.

    Prefers an explicit per-layer ``layer_types`` list (so an empty result
    here means "no layer slides", not "unknown"); falls back to the
    Qwen2/Qwen3-style ``use_sliding_window``/``max_window_layers`` pair.
    """
    layer_types = _cfg(config, "layer_types")
    if isinstance(layer_types, list) and layer_types:
        kinds = {str(kind) for kind in layer_types}
        if kinds & _ATTENTION_LAYER_TYPES:
            indices = [
                i
                for i, kind in enumerate(layer_types)
                if str(kind) == "sliding_attention"
            ]
            return indices, _cfg(config, "sliding_window")

    if _as_bool(_cfg(config, "use_sliding_window")):
        max_window_layers = _cfg(config, "max_window_layers")
        if max_window_layers is not None and num_layers:
            return list(range(int(max_window_layers))), _cfg(config, "sliding_window")

    return None, None


def _recurrent_layers(config: dict[str, Any]) -> list[int] | None:
    """Layer indices whose ``layer_types`` entry names neither full nor sliding
    attention (a gated-delta-net/linear-attention/Mamba-style hybrid layer).
    """
    layer_types = _cfg(config, "layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        return None
    indices = [
        i
        for i, kind in enumerate(layer_types)
        if str(kind) not in _ATTENTION_LAYER_TYPES
    ]
    return indices or None
