"""Extract architecture metadata from Hugging Face configs (CPU-only, no weights)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visualizer.ast_analyze import analyze_sources, dump_ast
from visualizer.basic_ops import BasicOpFilter
from visualizer.block_tree import (
    BlockNode,
    build_decoder_block_trees,
    build_full_detailed_block_trees,
    partition_detail_trees,
)
from visualizer.blocks import BlockComponent, CodeAnalysis, LayerVariant
from visualizer.config_resolve import load_checkpoint_config
from visualizer.github import fetch_github_source, github_config_path, parse_github_url
from visualizer.source import read_sources, resolve_source_files

BYTES_PER_BF16 = 2


@dataclass
class ArchitectureSpec:
    """Normalized architecture description for diagram rendering."""

    name: str
    model_type: str
    architectures: list[str] = field(default_factory=list)

    # Scale
    total_params_hint: str | None = None
    active_params_hint: str | None = None
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    intermediate_size: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None

    # Attention
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    attention_type: str = "MHA"
    attention_notes: list[str] = field(default_factory=list)

    # Positional
    positional_encoding: str = "RoPE"

    # FFN / MoE
    decoder_type: str = "Dense"
    ffn_type: str = "SwiGLU"
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    num_shared_experts: int | None = None
    moe_intermediate_size: int | None = None
    moe_notes: list[str] = field(default_factory=list)

    # Norm / block layout
    norm_type: str = "RMSNorm"
    norm_placement: str = "Pre-Norm"
    norm_notes: list[str] = field(default_factory=list)

    # Layer mix (sliding window, hybrid, dense prefix, etc.)
    layer_mix: str | None = None
    layer_variants: list[LayerVariant] = field(default_factory=list)
    layer_repeat_lines: list[str] = field(default_factory=list)
    layer_notes: list[str] = field(default_factory=list)

    # Embeddings / output
    tie_word_embeddings: bool | None = None

    # Derived
    kv_cache_per_token_bf16: str | None = None
    highlights: list[str] = field(default_factory=list)
    source_path: str = ""
    checkpoint_source: str = ""
    github_source: str = ""
    raw_config: dict[str, Any] = field(default_factory=dict, repr=False)

    # AST-derived block graph
    block_components: list[BlockComponent] = field(default_factory=list)
    stack_pre: list[BlockComponent] = field(default_factory=list)
    stack_tail: list[BlockComponent] = field(default_factory=list)
    forward_sequence: list[str] = field(default_factory=list)
    decoder_class: str | None = None
    code_sources: list[str] = field(default_factory=list)
    analysis_notes: list[str] = field(default_factory=list)
    custom_blocks: list[str] = field(default_factory=list)
    detailed_block_trees: list[tuple[str, BlockNode]] = field(default_factory=list)
    export_block_trees: list[tuple[str, BlockNode]] = field(default_factory=list)
    class_registry: dict = field(default_factory=dict, repr=False)
    basic_ops: BasicOpFilter | None = None


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _get(config: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _human_bytes(num_bytes: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}".replace(".0 ", " ")
        size /= 1024
    return f"{size:.1f} TiB"


def load_config_dict(
    checkpoint: str | Path,
    *,
    config_path: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Load config.json from a HF checkpoint, subpath, or local directory."""
    return load_checkpoint_config(checkpoint, config_path=config_path)


def _resolve_checkpoint(
    *,
    checkpoint: str | Path | None,
    github: str | None,
    config_path: str | None = None,
) -> tuple[dict[str, Any], str]:
    if checkpoint is not None:
        return load_config_dict(checkpoint, config_path=config_path)

    if github:
        ref = parse_github_url(github)
        root = fetch_github_source(ref)
        discovered = github_config_path(root)
        if discovered is None:
            raise FileNotFoundError(
                "No checkpoint provided and no config.json found in GitHub source. "
                "Pass a Hugging Face checkpoint via SOURCE or --checkpoint."
            )
        config = json.loads(discovered.read_text(encoding="utf-8"))
        label = f"github-config://{ref.display}"
        from visualizer.config_resolve import normalize_config

        return normalize_config(config, source_label=label), label

    raise ValueError(
        "Provide a Hugging Face checkpoint (SOURCE or --checkpoint) "
        "or a GitHub repo that contains config.json."
    )


def _infer_attention(config: dict[str, Any], spec: ArchitectureSpec) -> None:
    model_type = (spec.model_type or "").lower()
    kv_lora_rank = _as_int(_get(config, "kv_lora_rank"))
    q_lora_rank = _as_int(_get(config, "q_lora_rank"))
    qk_nope_head_dim = _as_int(_get(config, "qk_nope_head_dim"))
    qk_rope_head_dim = _as_int(_get(config, "qk_rope_head_dim"))

    if kv_lora_rank or q_lora_rank or "deepseek" in model_type and qk_nope_head_dim:
        spec.attention_type = "MLA"
        spec.attention_notes.append("Multi-head Latent Attention (compressed KV)")
        if kv_lora_rank:
            spec.attention_notes.append(f"kv_lora_rank={kv_lora_rank}")
        return

    num_heads = spec.num_attention_heads
    num_kv = spec.num_key_value_heads

    if num_heads and num_kv:
        if num_kv == 1:
            spec.attention_type = "MQA"
        elif num_kv < num_heads:
            spec.attention_type = "GQA"
            group = max(1, num_heads // num_kv)
            spec.attention_notes.append(f"GQA group size ≈ {group}")
        else:
            spec.attention_type = "MHA"

    if _as_bool(_get(config, "use_sliding_window")):
        window = _as_int(_get(config, "sliding_window", "sliding_window_size"))
        max_window_layers = _as_int(_get(config, "max_window_layers"))
        if max_window_layers and spec.num_hidden_layers:
            global_layers = spec.num_hidden_layers - max_window_layers
            spec.layer_mix = f"{max_window_layers} sliding-window + {global_layers} global"
            spec.layer_notes.append(f"Sliding window size={window}")
        elif window:
            spec.layer_notes.append(f"Sliding window attention (window={window})")

    if _as_bool(_get(config, "attention_bias", "use_bias", "bias")):
        spec.attention_notes.append("Attention projections use bias")

    if _get(config, "qk_norm", "use_qk_norm") is True:
        spec.attention_notes.append("QK-Norm enabled")
        spec.norm_notes.append("QK-Norm inside attention")


def _infer_positional(config: dict[str, Any], spec: ArchitectureSpec) -> None:
    model_type = (spec.model_type or "").lower()
    rope = _get(config, "rope_parameters", "rope_scaling", "rope_theta")

    if _get(config, "position_embedding_type") == "nope" or "nope" in model_type:
        spec.positional_encoding = "NoPE"
        return

    alibi = _as_bool(_get(config, "alibi"))
    if alibi:
        spec.positional_encoding = "ALiBi"
        return

    if _get(config, "position_embedding_type") == "absolute":
        spec.positional_encoding = "Learned absolute"
        return

    if model_type in {"gpt2", "gpt_neox", "bloom", "opt"} and not rope:
        spec.positional_encoding = "Learned absolute"
        return

    if rope or spec.max_position_embeddings:
        spec.positional_encoding = "RoPE"
        theta = _get(config, "rope_theta")
        if theta:
            spec.attention_notes.append(f"RoPE theta={theta}")


def _infer_ffn_and_moe(config: dict[str, Any], spec: ArchitectureSpec) -> None:
    hidden_act = str(_get(config, "hidden_act", "activation_function") or "silu").lower()
    if hidden_act in {"silu", "swish"}:
        spec.ffn_type = "SwiGLU"
    elif hidden_act == "gelu":
        spec.ffn_type = "GeGLU" if _get(config, "gated_ffn") else "GELU"
    else:
        spec.ffn_type = hidden_act.upper()

    num_experts = _as_int(
        _get(
            config,
            "num_experts",
            "n_routed_experts",
            "moe_num_experts",
            "num_local_experts",
            "num_moe_experts",
        )
    )
    experts_per_tok = _as_int(
        _get(
            config,
            "num_experts_per_token",
            "num_experts_per_tok",
            "moe_k",
            "num_selected_experts",
            "top_k",
            "moe_top_k",
        )
    )
    shared_experts = _as_int(
        _get(
            config,
            "num_shared_experts",
            "moe_num_shared_experts",
            "n_shared_experts",
        )
    )

    spec.num_experts = num_experts
    spec.num_experts_per_tok = experts_per_tok
    spec.num_shared_experts = shared_experts
    spec.moe_intermediate_size = _as_int(
        _get(config, "moe_intermediate_size", "expert_intermediate_size")
    )

    if num_experts and num_experts > 1:
        spec.decoder_type = "Sparse MoE"
        if shared_experts:
            spec.moe_notes.append(f"{shared_experts} shared expert(s)")
        if experts_per_tok:
            active_ratio = experts_per_tok / num_experts * 100
            spec.moe_notes.append(
                f"{experts_per_tok}/{num_experts} experts active (~{active_ratio:.1f}%)"
            )
    elif _get(config, "layer_types", "block_types", "hybrid_block_types"):
        spec.decoder_type = "Hybrid"
    else:
        spec.decoder_type = "Dense"

    first_k_dense = _as_int(_get(config, "first_k_dense_replace", "num_dense_layers"))
    moe_layer_start = _as_int(_get(config, "moe_layer_start_index"))
    moe_layer_interval = _as_int(_get(config, "moe_layer_interval"))
    mlp_only_layers = _get(config, "mlp_only_layers")

    if first_k_dense:
        spec.moe_notes.append(f"First {first_k_dense} layers are dense FFN")
    if moe_layer_start is not None and moe_layer_interval:
        spec.moe_notes.append(
            f"MoE from layer {moe_layer_start} every {moe_layer_interval} layer(s)"
        )
    if isinstance(mlp_only_layers, list) and mlp_only_layers:
        spec.moe_notes.append(f"Dense FFN on layer indices: {mlp_only_layers}")

    layer_types = _get(config, "layer_types", "block_types")
    if isinstance(layer_types, list) and layer_types:
        from collections import Counter

        counts = Counter(layer_types)
        spec.decoder_type = "Hybrid"
        spec.layer_mix = ", ".join(f"{count} {kind}" for kind, count in counts.items())


def _config_has_per_layer_typing(config: dict[str, Any]) -> bool:
    """True when config encodes per-layer module selection beyond a flat layer_types list."""
    if isinstance(_get(config, "layer_types", "block_types"), list):
        return False
    for container in [config, *([value for value in config.values() if isinstance(value, dict)])]:
        for key, value in container.items():
            if "layer" in key.lower() and isinstance(value, list) and value:
                if all(_as_int(item) is not None for item in value):
                    return True
    return False


_ATTENTION_ATTRS = frozenset({"self_attn", "self_attention", "attn", "attention"})


def _resolve_conditional_class(
    layer_idx: int,
    rules: list[tuple[str, str]],
    config: dict[str, Any],
) -> str | None:
    from visualizer.layer_repeat_simplify import layer_condition_matches

    for class_name, condition in rules:
        if condition == "else":
            continue
        if layer_condition_matches(layer_idx, condition, config):
            return class_name
    for class_name, condition in rules:
        if condition == "else":
            return class_name
    return None


def _default_attention_class(
    class_registry: dict | None,
    decoder_class: str | None,
) -> str | None:
    if not class_registry or not decoder_class:
        return None
    from visualizer.ast_analyze import ClassStructure, _classify_role

    decoder = class_registry.get(decoder_class)
    if not isinstance(decoder, ClassStructure):
        return None
    for attr, class_name in decoder.init_assignments.items():
        if _classify_role(attr, class_name) == "attention":
            return class_name
    return None


def _resolve_ffn_for_layer(
    layer_idx: int,
    ffn_rules: list[tuple[str, str, str]],
    config: dict[str, Any],
) -> tuple[str | None, str | None]:
    from visualizer.layer_repeat_simplify import layer_condition_matches

    for attr, class_name, condition in ffn_rules:
        if condition != "else" and layer_condition_matches(layer_idx, condition, config):
            return attr, class_name
    for attr, class_name, condition in ffn_rules:
        if condition == "else":
            return attr, class_name
    if _config_moe_layer(layer_idx, config):
        for attr, class_name, _condition in ffn_rules:
            from visualizer.ast_analyze import _classify_role

            if _classify_role(attr, class_name) == "moe":
                return attr, class_name
        return "block_sparse_moe", None
    for attr, class_name, _condition in ffn_rules:
        from visualizer.ast_analyze import _classify_role

        if _classify_role(attr, class_name) == "ffn":
            return attr, class_name
    return "mlp", None


def _apply_uniform_ffn_component(
    spec: ArchitectureSpec,
    *,
    ffn_attr: str | None,
    ffn_class: str | None,
) -> None:
    """Name the spine's FFN tile after the branch every layer of this config takes.

    A decoder's ``__init__`` can bind one attribute to either a dense or a sparse
    class; the AST keeps whichever it saw last, which is not necessarily the one this
    checkpoint builds.
    """
    if not ffn_attr or not ffn_class:
        return
    from dataclasses import replace

    from visualizer.ast_analyze import _label_for, ffn_role_for_class

    for index, comp in enumerate(spec.block_components):
        if comp.attr_name != ffn_attr or comp.role not in {"ffn", "moe"}:
            continue
        if comp.class_name == ffn_class:
            return
        role = ffn_role_for_class(comp.attr_name, ffn_class)
        spec.block_components[index] = replace(
            comp,
            class_name=ffn_class,
            role=role,
            label=_label_for(role, ffn_class, comp.attr_name),
        )
        return


def _infer_layer_variants(
    config: dict[str, Any],
    spec: ArchitectureSpec,
    *,
    class_registry: dict | None = None,
    decoder_class: str | None = None,
) -> None:
    """Infer per-layer decoder templates when attention or FFN type varies by depth."""
    num_layers = spec.num_hidden_layers
    if not num_layers:
        return

    layer_types = _get(config, "layer_types", "block_types")
    if isinstance(layer_types, list) and layer_types:
        return

    conditionals: list[tuple[str, str, str]] = []
    if class_registry and decoder_class:
        from visualizer.ast_analyze import ClassStructure, _extract_decoder_layer_conditionals

        decoder = class_registry.get(decoder_class)
        if isinstance(decoder, ClassStructure):
            conditionals = _extract_decoder_layer_conditionals(decoder)

    if not conditionals and not _config_has_per_layer_typing(config):
        return

    from collections import Counter
    from visualizer.ast_analyze import _classify_role, _label_for, ffn_role_for_class

    attn_rules = [(cls, cond) for attr, cls, cond in conditionals if attr in _ATTENTION_ATTRS]
    ffn_rules = [
        (attr, cls, cond)
        for attr, cls, cond in conditionals
        if _classify_role(attr, cls) in {"ffn", "moe"}
    ]
    decoder_options: list[str] = []
    if class_registry and decoder_class:
        decoder = class_registry.get(decoder_class)
        if isinstance(decoder, ClassStructure):
            for attr in ("mlp", "block_sparse_moe"):
                decoder_options.extend(decoder.init_assignment_options.get(attr, []))
    decoder_options = list(dict.fromkeys(decoder_options))
    moe_option = next(
        (class_name for class_name in decoder_options if ffn_role_for_class("", class_name) == "moe"),
        None,
    )
    dense_option = next(
        (class_name for class_name in decoder_options if ffn_role_for_class("", class_name) == "ffn"),
        None,
    )

    buckets: Counter[tuple[str, str | None, str, str | None, str | None]] = Counter()
    for layer_idx in range(num_layers):
        attn_class = (
            _resolve_conditional_class(layer_idx, attn_rules, config)
            if attn_rules
            else None
        )
        if attn_class is None:
            attn_class = _default_attention_class(class_registry, decoder_class)

        ffn_attr, ffn_class = (
            _resolve_ffn_for_layer(layer_idx, ffn_rules, config)
            if ffn_rules
            else (None, None)
        )
        if ffn_class is None and ffn_attr is None:
            if _config_moe_layer(layer_idx, config):
                ffn_attr, ffn_class = "mlp", moe_option
            else:
                ffn_attr, ffn_class = "mlp", dense_option

        attn_label = _label_for("attention", attn_class or "Attention", "self_attn")
        ffn_role = ffn_role_for_class(ffn_attr or "mlp", ffn_class or "MLP")
        ffn_display = _label_for(ffn_role, ffn_class or "", ffn_attr or "")

        buckets[(attn_label, attn_class, ffn_display, ffn_class, ffn_attr)] += 1

    if len(buckets) <= 1:
        only = next(iter(buckets.items()), None)
        if only:
            (attn_label, attn_class, _ffn_display, ffn_class, ffn_attr), _count = only
            if attn_class:
                spec.attention_notes.append(f"Attention module: {attn_class}")
            if len({variant.attention_label for variant in spec.layer_variants}) <= 1:
                spec.attention_type = attn_label
            _apply_uniform_ffn_component(spec, ffn_attr=ffn_attr, ffn_class=ffn_class)
        return

    variants: list[LayerVariant] = []
    for (attn_label, attn_class, ffn_display, ffn_class, ffn_attr), count in sorted(
        buckets.items(),
        key=lambda item: (-item[1], item[0][0], item[0][2]),
    ):
        variants.append(
            LayerVariant(
                label=f"{attn_label} + {ffn_display}",
                count=count,
                attention_label=attn_label,
                attention_class=attn_class,
                ffn_label=ffn_display,
                ffn_class=ffn_class,
                ffn_attr=ffn_attr,
            )
        )

    spec.layer_variants = variants
    attn_labels = sorted({variant.attention_label for variant in variants})
    if len(attn_labels) > 1:
        spec.attention_type = "Hybrid"
        spec.attention_notes.append(" / ".join(attn_labels))
    elif len(attn_labels) == 1:
        spec.attention_type = attn_labels[0]
        if variants[0].attention_class:
            spec.attention_notes.append(f"Attention module: {variants[0].attention_class}")

    mix_parts = [f"{variant.count} {variant.label}" for variant in variants]
    spec.layer_mix = ", ".join(mix_parts)
    if len({variant.ffn_label for variant in variants}) > 1:
        spec.decoder_type = "Hybrid"
        spec.layer_notes.append("Per-layer module types from AST/config")


def _config_moe_layer(layer_idx: int, config: dict[str, Any]) -> bool:
    num_experts = _as_int(
        _get(
            config,
            "num_experts",
            "n_routed_experts",
            "moe_num_experts",
            "num_local_experts",
            "num_moe_experts",
        )
    )
    if not num_experts or num_experts <= 1:
        return False
    moe_pattern = _get(config, "moe_layer_freq")
    if isinstance(moe_pattern, list):
        if 0 <= layer_idx < len(moe_pattern):
            return bool(_as_int(moe_pattern[layer_idx]) or 0)
        return False
    first_k_dense = _as_int(_get(config, "first_k_dense_replace", "num_dense_layers")) or 0
    moe_freq = _as_int(_get(config, "moe_layer_freq", "moe_layer_interval")) or 1
    return layer_idx >= first_k_dense and layer_idx % moe_freq == 0


def _infer_norm(config: dict[str, Any], spec: ArchitectureSpec) -> None:
    model_type = (spec.model_type or "").lower()

    if _get(config, "rms_norm_eps") is not None:
        spec.norm_type = "RMSNorm"
    elif _get(config, "layer_norm_eps") is not None or model_type in {"gpt2", "gpt_neox", "opt", "bloom"}:
        spec.norm_type = "LayerNorm"

    if any(token in model_type for token in ("gpt2", "gpt_neox", "bloom", "opt", "llama", "mistral", "qwen")):
        spec.norm_placement = "Pre-Norm"

    if "olmo" in model_type:
        spec.norm_placement = "Post-Norm (inside residual)"
        spec.norm_notes.append("Sandwich / post-norm variant")

    if _get(config, "post_norm", "use_post_norm") is True:
        spec.norm_placement = "Post-Norm"


def _estimate_kv_cache(spec: ArchitectureSpec, config: dict[str, Any]) -> None:
    layers = spec.num_hidden_layers
    if not layers:
        return

    if spec.attention_type == "MLA" or _as_int(_get(config, "kv_lora_rank")):
        kv_lora_rank = _as_int(_get(config, "kv_lora_rank")) or 512
        kv_heads = spec.num_key_value_heads or spec.num_attention_heads or 1
        bytes_per_layer = kv_heads * kv_lora_rank * BYTES_PER_BF16
        spec.kv_cache_per_token_bf16 = _human_bytes(bytes_per_layer * layers)
        return

    head_dim = spec.head_dim
    if not head_dim and spec.hidden_size and spec.num_attention_heads:
        head_dim = spec.hidden_size // spec.num_attention_heads

    kv_heads = spec.num_key_value_heads or spec.num_attention_heads
    if not head_dim or not kv_heads:
        return

    # K and V tensors per layer.
    bytes_per_layer = 2 * kv_heads * head_dim * BYTES_PER_BF16
    spec.kv_cache_per_token_bf16 = _human_bytes(bytes_per_layer * layers)


def _estimate_param_hint(config: dict[str, Any], spec: ArchitectureSpec) -> None:
    """Best-effort parameter estimate from config when total size isn't annotated."""
    if spec.total_params_hint:
        return

    hidden = spec.hidden_size
    layers = spec.num_hidden_layers
    vocab = spec.vocab_size
    inter = spec.intermediate_size or spec.moe_intermediate_size
    if not all([hidden, layers, vocab]):
        return

    # Very rough: embeddings + transformer blocks + lm head.
    embed = vocab * hidden
    attn = layers * hidden * hidden * 4
    ffn_multiplier = 3 if spec.ffn_type == "SwiGLU" else 2
    if spec.decoder_type == "Sparse MoE" and spec.num_experts:
        # Each expert is as wide as one routed FFN, not as wide as the dense one.
        expert_inter = spec.moe_intermediate_size or inter or hidden * 4
        ffn = layers * spec.num_experts * hidden * expert_inter * ffn_multiplier
        active = spec.num_experts_per_tok or 1
        active_ffn = layers * active * hidden * expert_inter * ffn_multiplier
        total = embed * 2 + attn + ffn
        active_total = embed * 2 + attn + active_ffn
        spec.total_params_hint = _format_params(total)
        spec.active_params_hint = _format_params(active_total)
    else:
        ffn = layers * hidden * (inter or hidden * 4) * ffn_multiplier
        total = embed * 2 + attn + ffn
        spec.total_params_hint = _format_params(total)


def _format_params(count: float) -> str:
    if count >= 1e12:
        return f"{count / 1e12:.2f}T"
    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    if count >= 1e6:
        return f"{count / 1e6:.0f}M"
    return f"{int(count)}"


def _build_highlights(spec: ArchitectureSpec) -> None:
    highlights: list[str] = []
    if spec.attention_type != "MHA":
        highlights.append(spec.attention_type)
    if spec.decoder_type != "Dense":
        highlights.append(spec.decoder_type)
    if spec.layer_mix:
        highlights.append(spec.layer_mix)
    if spec.positional_encoding != "RoPE":
        highlights.append(spec.positional_encoding)
    for block in spec.custom_blocks[:1]:
        highlights.append(block)
    for note in spec.attention_notes[:1]:
        highlights.append(note)
    spec.highlights = highlights[:4]


def _rebuild_stack_components(
    spec: ArchitectureSpec,
    analysis: CodeAnalysis,
) -> None:
    from visualizer.ast_analyze import (
        ClassStructure,
        _pick_causal_lm_class,
        _pick_decoder_class,
        _pick_stack_model_class,
        build_stack_components,
    )

    registry: dict[str, ClassStructure] = analysis.class_registry
    if not registry:
        return

    decoder = registry.get(analysis.decoder_class) if analysis.decoder_class else _pick_decoder_class(registry)
    causal_lm = registry.get(analysis.causal_lm_class) if analysis.causal_lm_class else _pick_causal_lm_class(registry)
    stack_model = (
        registry.get(analysis.stack_model_class)
        if analysis.stack_model_class
        else _pick_stack_model_class(registry, causal_lm)
    )
    if stack_model is None and causal_lm is None:
        spec.stack_pre = []
        spec.stack_tail = []
        return
    spec.stack_pre, spec.stack_tail = build_stack_components(
        stack_model=stack_model,
        causal_lm=causal_lm,
        decoder=decoder,
        registry=registry,
    )


def _finalize_layer_repeat_lines(spec: ArchitectureSpec) -> None:
    """Fill in resolved layer counts on AST-derived repeat lines."""
    if not spec.layer_repeat_lines:
        return
    lines = list(spec.layer_repeat_lines)
    if spec.num_hidden_layers is not None and lines:
        first = lines[0]
        if first.startswith("N ×"):
            lines[0] = first.replace("N ×", f"{spec.num_hidden_layers} ×", 1)
        if " in range(" in lines[0]:
            prefix, _ = lines[0].split(" in range(", 1)
            lines[0] = f"{prefix} in range({spec.num_hidden_layers}))"
    if spec.layer_variants:
        existing = set(lines)
        for variant in spec.layer_variants:
            line = f"{variant.count} × {variant.label}"
            if line not in existing:
                lines.append(line)
                existing.add(line)
    if spec.raw_config:
        from visualizer.layer_repeat_simplify import simplify_layer_repeat_lines

        lines = simplify_layer_repeat_lines(lines, spec.raw_config)
    spec.layer_repeat_lines = lines


def _merge_code_analysis(spec: ArchitectureSpec, analysis: CodeAnalysis) -> None:
    spec.decoder_class = analysis.decoder_class
    spec.block_components = list(analysis.block_components)
    spec.forward_sequence = list(analysis.forward_sequence)
    spec.code_sources = list(analysis.source_files)
    spec.analysis_notes = list(analysis.notes)
    spec.custom_blocks = list(analysis.custom_blocks)
    spec.class_registry = dict(analysis.class_registry)
    spec.layer_repeat_lines = list(analysis.layer_repeat_lines)
    _rebuild_stack_components(spec, analysis)
    if spec.raw_config and spec.num_hidden_layers:
        _infer_layer_variants(
            spec.raw_config,
            spec,
            class_registry=spec.class_registry,
            decoder_class=spec.decoder_class,
        )
    _finalize_layer_repeat_lines(spec)

    if analysis.attention_type:
        mixed_attention = spec.layer_variants and len({variant.attention_label for variant in spec.layer_variants}) > 1
        if not mixed_attention:
            spec.attention_type = analysis.attention_type
        if analysis.attention_class:
            spec.attention_notes.insert(0, f"AST: {analysis.attention_class}")

    if analysis.decoder_type:
        from visualizer.ast_analyze import decoder_type_for_components

        # Read the flavor off the resolved spine, which may name a different FFN class
        # than the AST alone reported.
        spec.decoder_type = (
            decoder_type_for_components(spec.block_components) or analysis.decoder_type
        )

    if analysis.ffn_type:
        spec.ffn_type = analysis.ffn_type

    if analysis.norm_type:
        spec.norm_type = analysis.norm_type
    if analysis.norm_placement:
        spec.norm_placement = analysis.norm_placement

    moe_components = [c for c in analysis.block_components if c.role == "moe"]
    if moe_components and not spec.moe_notes:
        for comp in moe_components:
            spec.moe_notes.append(f"AST module: {comp.class_name}")
            spec.moe_notes.extend(comp.details)

    for comp in analysis.block_components:
        if comp.role == "other":
            spec.layer_notes.append(f"Custom block `{comp.attr_name}` ({comp.class_name})")


def _code_rotates_positions(analysis: CodeAnalysis) -> bool:
    """True when the modeling source shows positions being rotated somewhere."""
    from visualizer.ast_analyze import POSITIONAL_CLASS_RE, is_positional_synthetic

    if analysis.positional_helpers:
        return True
    if any(comp.role == "positional" for comp in analysis.stack_pre):
        return True
    for name, cls in analysis.class_registry.items():
        if POSITIONAL_CLASS_RE.search(name):
            return True
        if any(is_positional_synthetic(call) for call in cls.forward_calls):
            return True
    return False


def _refine_positional_from_code(spec: ArchitectureSpec, analysis: CodeAnalysis) -> None:
    """Correct a config-derived rope claim the modeling source contradicts.

    A config carrying `rope_theta` only means rope parameters exist; a checkpoint can
    still run without positional encoding, as MLA variants asserting NoPE do.
    """
    # Only a rope claim can be checked this way: ALiBi biases scores and learned
    # absolute encodings add an embedding, neither of which rotates anything.
    if spec.positional_encoding != "RoPE":
        return
    if _code_rotates_positions(analysis):
        return
    spec.positional_encoding = "NoPE"
    spec.attention_notes = [
        note for note in spec.attention_notes if not note.startswith("RoPE theta=")
    ]
    spec.layer_notes.append("No positional encoding applied in modeling code")


def _build_detailed_block_trees(spec: ArchitectureSpec, basic_ops: BasicOpFilter) -> None:
    if not spec.class_registry:
        spec.detailed_block_trees = []
        spec.export_block_trees = []
        return
    common_kwargs = dict(
        components=spec.block_components,
        registry=spec.class_registry,
        basic_ops=basic_ops,
        positional_encoding=spec.positional_encoding,
        norm_type=spec.norm_type,
        decoder_class=spec.decoder_class,
        stack_pre=spec.stack_pre,
        stack_tail=spec.stack_tail,
    )
    decoder_trees = build_decoder_block_trees(
        spec.block_components,
        spec.class_registry,
        basic_ops,
        decoder_class=spec.decoder_class,
        include_norms=False,
        infer_init_steps=False,
    )
    spec.detailed_block_trees = partition_detail_trees(decoder_trees)
    spec.export_block_trees = build_full_detailed_block_trees(
        **common_kwargs,
        partition=False,
        include_norms=True,
        infer_init_steps=True,
    )


def architecture_section_trees(spec: ArchitectureSpec) -> list[tuple[str, BlockNode]]:
    """Return block trees for detailed SVG sections and operator/graph export."""
    return spec.export_block_trees or spec.detailed_block_trees


def parse_architecture(
    config: dict[str, Any],
    source: str,
    name: str | None = None,
    *,
    code_analysis: CodeAnalysis | None = None,
) -> ArchitectureSpec:
    """Convert a config dict into an ArchitectureSpec."""
    model_type = str(_get(config, "model_type") or "unknown")
    architectures = _get(config, "architectures") or []
    if not isinstance(architectures, list):
        architectures = [str(architectures)]

    display_name = name or (
        architectures[0] if architectures else model_type.replace("_", " ").title()
    )

    spec = ArchitectureSpec(
        name=display_name,
        model_type=model_type,
        architectures=[str(a) for a in architectures],
        hidden_size=_as_int(_get(config, "hidden_size", "n_embd", "d_model")),
        num_hidden_layers=_as_int(_get(config, "num_hidden_layers", "n_layer", "num_layers")),
        intermediate_size=_as_int(_get(config, "intermediate_size", "n_inner", "ffn_dim")),
        vocab_size=_as_int(_get(config, "vocab_size")),
        max_position_embeddings=_as_int(
            _get(config, "max_position_embeddings", "max_seq_len", "seq_length")
        ),
        num_attention_heads=_as_int(_get(config, "num_attention_heads", "n_head")),
        num_key_value_heads=_as_int(_get(config, "num_key_value_heads", "num_kv_heads")),
        head_dim=_as_int(_get(config, "head_dim")),
        tie_word_embeddings=_as_bool(_get(config, "tie_word_embeddings")),
        source_path=source,
        raw_config=config,
    )

    total_params = _get(config, "total_params", "num_parameters")
    if total_params:
        spec.total_params_hint = str(total_params)

    _infer_attention(config, spec)
    _infer_positional(config, spec)
    _infer_ffn_and_moe(config, spec)
    _infer_norm(config, spec)

    if code_analysis is not None and code_analysis.has_block_graph():
        _merge_code_analysis(spec, code_analysis)
        _refine_positional_from_code(spec, code_analysis)

    _estimate_kv_cache(spec, config)
    _estimate_param_hint(config, spec)
    _build_highlights(spec)
    return spec


def load_architecture(
    source: str | Path | None = None,
    name: str | None = None,
    *,
    checkpoint: str | Path | None = None,
    github: str | None = None,
    config_path: str | None = None,
    code_path: str | Path | None = None,
    analyze_code: bool = True,
    detailed: bool = False,
    basic_ops: BasicOpFilter | None = None,
    all_tensor_ops: bool = False,
) -> ArchitectureSpec:
    """Load architecture metadata from an HF checkpoint and/or GitHub modeling code."""
    resolved_checkpoint = checkpoint or source
    config, config_label = _resolve_checkpoint(
        checkpoint=resolved_checkpoint,
        github=github,
        config_path=config_path,
    )
    code_analysis: CodeAnalysis | None = None
    code_labels: list[str] = []

    if analyze_code:
        source_files, code_labels = resolve_source_files(
            resolved_checkpoint,
            config,
            code_path=code_path,
            github=github,
        )
        if source_files:
            from visualizer.kernel_pipeline import register_kernel_search_root

            # The caller's own path first: resolving a checkpoint file follows HF's
            # symlink into its blob store, where sibling kernel modules do not exist.
            if code_path is not None:
                register_kernel_search_root(code_path)
            for source_file in source_files:
                register_kernel_search_root(source_file)
            code_analysis = analyze_sources(
                read_sources(source_files),
                config=config,
                all_tensor_ops=all_tensor_ops,
            )

    spec = parse_architecture(config, config_label, name=name, code_analysis=code_analysis)
    spec.checkpoint_source = config_label
    spec.source_path = config_label
    if config.get("_has_vision_tower"):
        spec.layer_notes.append("Multimodal wrapper includes a vision tower")
    if config.get("_wrapper_model_type"):
        spec.layer_notes.append(f"Loaded text backbone from {config['_wrapper_model_type']} wrapper")
    if github:
        spec.github_source = parse_github_url(github).display
    else:
        spec.github_source = next(
            (label for label in code_labels if label.startswith("github://")),
            spec.github_source,
        )
    spec.code_sources = code_labels or spec.code_sources
    if detailed:
        resolved_basic_ops = basic_ops or BasicOpFilter.for_detailed()
        spec.basic_ops = resolved_basic_ops
        _build_detailed_block_trees(spec, resolved_basic_ops)
    return spec


def dump_model_ast(
    source: str | Path | None = None,
    *,
    checkpoint: str | Path | None = None,
    github: str | None = None,
    config_path: str | None = None,
    code_path: str | Path | None = None,
) -> str:
    """Return a pretty-printed AST dump for the primary modeling file."""
    resolved_checkpoint = checkpoint or source
    config, _ = _resolve_checkpoint(
        checkpoint=resolved_checkpoint,
        github=github,
        config_path=config_path,
    )
    source_files, _ = resolve_source_files(
        resolved_checkpoint,
        config,
        code_path=code_path,
        github=github,
    )
    if not source_files:
        raise FileNotFoundError("No modeling source file found to parse")
    text = source_files[0].read_text(encoding="utf-8")
    return dump_ast(text, filename=str(source_files[0]))
