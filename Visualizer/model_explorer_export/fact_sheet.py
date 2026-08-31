"""Model Explorer fact sheet panel (mirrors ``visualizer.render._fact_lines``)."""

from __future__ import annotations

from typing import Any

from visualizer.extract import ArchitectureSpec

FACT_SHEET_NODE_ID = "@fact_sheet"
_FACT_SUBLINE_INDENT = "    "


def _fact_lines(spec: ArchitectureSpec) -> list[str]:
    """Keep in sync with ``visualizer.render._fact_lines``."""
    lines = [
        f"Model type: {spec.model_type}",
        f"Decoder: {spec.decoder_type}",
        f"Attention: {spec.attention_type}",
        f"Positional: {spec.positional_encoding}",
        f"Norm: {spec.norm_type} ({spec.norm_placement})",
    ]
    if spec.decoder_class:
        lines.append(f"Decoder class: {spec.decoder_class}")
    if spec.checkpoint_source:
        lines.append(f"Checkpoint: {spec.checkpoint_source}")
    if spec.github_source:
        lines.append(f"GitHub code: {spec.github_source}")
    if spec.num_hidden_layers is not None and not spec.layer_repeat_lines:
        lines.append(f"Layers: {spec.num_hidden_layers}")
    if spec.hidden_size is not None:
        lines.append(f"Hidden size: {spec.hidden_size:,}")
    if spec.num_attention_heads is not None:
        kv = spec.num_key_value_heads or spec.num_attention_heads
        lines.append(f"Heads: {spec.num_attention_heads} Q / {kv} KV")
    if spec.vocab_size is not None:
        lines.append(f"Vocab: {spec.vocab_size:,}")
    if spec.max_position_embeddings is not None:
        lines.append(f"Context: {spec.max_position_embeddings:,} tokens")
    if spec.total_params_hint:
        param_line = f"Params (est.): {spec.total_params_hint}"
        if spec.active_params_hint:
            param_line += f" ({spec.active_params_hint} active)"
        lines.append(param_line)
    if spec.kv_cache_per_token_bf16:
        lines.append(f"KV cache / token (bf16 est.): {spec.kv_cache_per_token_bf16}")
    if spec.layer_repeat_lines:
        lines.append(f"Layer repeat: {spec.layer_repeat_lines[0]}")
        lines.extend(
            f"{_FACT_SUBLINE_INDENT}{subline}" for subline in spec.layer_repeat_lines[1:]
        )
    elif spec.layer_mix:
        lines.append(f"Layer mix: {spec.layer_mix}")
    if spec.forward_sequence:
        lines.append("Forward: " + " → ".join(spec.forward_sequence))
    for note in spec.moe_notes[:2]:
        lines.append(f"MoE: {note}")
    for note in spec.layer_notes[:1]:
        lines.append(f"Layers: {note}")
    for note in spec.analysis_notes[:1]:
        lines.append(f"AST: {note}")
    return lines


def _highlight_lines(spec: ArchitectureSpec) -> list[str]:
    if not spec.highlights:
        return []
    return [f"Highlights: {'; '.join(spec.highlights)}"]


def _format_fact_sheet_label(spec: ArchitectureSpec) -> str:
    body: list[str] = []
    for line in _fact_lines(spec):
        if line.startswith(_FACT_SUBLINE_INDENT):
            body.append(f"  {line[len(_FACT_SUBLINE_INDENT) :].strip()}")
        else:
            body.append(f"• {line.strip()}")
    body.extend(line.strip() for line in _highlight_lines(spec))
    return "Fact sheet\n" + "\n".join(body)


def build_fact_sheet_node(spec: ArchitectureSpec) -> dict[str, Any]:
    """Return one always-visible fact sheet block with a multi-line label."""
    label = _format_fact_sheet_label(spec)
    return {
        "id": FACT_SHEET_NODE_ID,
        "label": label,
        "namespace": "",
        "attrs": [
            {"key": "synthetic", "value": "fact_sheet"},
            {"key": "class_name", "value": "FactSheet"},
            {"key": "title", "value": spec.name},
            {"key": "model_type", "value": spec.model_type},
        ],
        "style": {
            "backgroundColor": "#ffffff",
            "textColor": "#1a1a1a",
            "borderColor": "#d0d0d0",
        },
    }
