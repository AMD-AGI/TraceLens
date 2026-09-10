###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Model Explorer fact sheet panel."""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from datetime import datetime

from TraceLens.ModelUtils.ast_analyze import _classify_role, _label_for
from TraceLens.ModelUtils.extract import ArchitectureSpec
from TraceLens.ModelUtils.github import is_github_url, parse_github_url

from TraceLens.Visualizer.model_explorer_export.overview import format_forward_sequence

_FACT_SUBLINE_INDENT = "    "
_LAYER_REPEAT_BRANCH_RE = re.compile(r"^(?P<attr>\w+) → (?P<class>[^ (]+)(?P<rest>.*)$")
_HF_SOURCE_RE = re.compile(r"^hf://(?P<rest>.+)$")
_GITHUB_DISPLAY_RE = re.compile(
    r"^github://(?P<owner>[^/]+)/(?P<repo>[^@/]+)@(?P<ref>[^/]+)(?:/(?P<subpath>.+))?$",
)


def checkpoint_source_url(label: str) -> str | None:
    """Map TraceLens checkpoint labels to a public https URL when possible."""
    text = label.strip()
    if text.startswith(("http://", "https://")):
        return text
    match = _HF_SOURCE_RE.match(text)
    if match is None:
        return None
    rest = match.group("rest")
    parts = rest.split("/")
    if len(parts) <= 2:
        return f"https://huggingface.co/{rest}"
    model_id = "/".join(parts[:2])
    subpath = "/".join(parts[2:])
    return f"https://huggingface.co/{model_id}/blob/main/{subpath}"


def github_source_url(label: str) -> str | None:
    """Map TraceLens GitHub labels to a public https URL when possible."""
    text = label.strip()
    if text.startswith(("http://", "https://")):
        return text
    if is_github_url(text):
        ref = parse_github_url(text)
        if ref.subpath:
            return f"https://github.com/{ref.owner}/{ref.repo}/blob/{ref.ref}/{ref.subpath}"
        return f"https://github.com/{ref.owner}/{ref.repo}/tree/{ref.ref}"
    match = _GITHUB_DISPLAY_RE.match(text)
    if match is None:
        return None
    owner = match.group("owner")
    repo = match.group("repo")
    ref = match.group("ref")
    subpath = (match.group("subpath") or "").strip("/")
    if subpath:
        return f"https://github.com/{owner}/{repo}/blob/{ref}/{subpath}"
    return f"https://github.com/{owner}/{repo}/tree/{ref}"


def hf_model_id(label: str) -> str | None:
    """Extract the bare ``owner/name`` Hugging Face id from a checkpoint label."""
    text = label.strip()
    match = _HF_SOURCE_RE.match(text)
    if match is None:
        return None
    parts = [part for part in match.group("rest").split("/") if part]
    if len(parts) < 2:
        return None
    return "/".join(parts[:2])


def _model_display_name(spec: ArchitectureSpec) -> str | None:
    """The exact HF model id when known, else the architecture class name."""
    if spec.checkpoint_source:
        model_id = hf_model_id(spec.checkpoint_source)
        if model_id:
            return model_id
    return spec.name or None


def _format_source_line(prefix: str, label: str) -> str:
    url = (
        checkpoint_source_url(label)
        if prefix == "Checkpoint"
        else github_source_url(label) if prefix == "GitHub code" else None
    )
    display = url or label
    return f"{prefix}: {display}"


def _format_source_line_html(prefix: str, label: str) -> str:
    url = (
        checkpoint_source_url(label)
        if prefix == "Checkpoint"
        else github_source_url(label) if prefix == "GitHub code" else None
    )
    if url is None:
        return f"- {html.escape(prefix)}: {html.escape(label)}"
    safe_url = html.escape(url, quote=True)
    safe_text = html.escape(url)
    return (
        f"- {html.escape(prefix)}: "
        f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{safe_text}</a>'
    )


def _display_layer_repeat_subline(line: str) -> str:
    """Map conditional layer-repeat class names to graph tile labels."""
    match = _LAYER_REPEAT_BRANCH_RE.match(line.strip())
    if match is None:
        return line
    attr_name = match.group("attr")
    class_name = match.group("class")
    rest = match.group("rest")
    role = _classify_role(attr_name, class_name)
    display = _label_for(role, class_name, attr_name)
    return f"{attr_name} → {display}{rest}"


def _fact_lines(spec: ArchitectureSpec) -> list[str]:
    """Build fact-sheet bullet lines from architecture metadata."""
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
        lines.append(_format_source_line("Checkpoint", spec.checkpoint_source))
    if spec.github_source:
        lines.append(_format_source_line("GitHub code", spec.github_source))
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
            f"{_FACT_SUBLINE_INDENT}{_display_layer_repeat_subline(subline)}"
            for subline in spec.layer_repeat_lines[1:]
        )
    elif spec.layer_mix:
        lines.append(f"Layer mix: {spec.layer_mix}")
    if spec.forward_sequence:
        lines.append(f"Forward: {format_forward_sequence(spec, arrow=' -> ')}")
    for note in spec.moe_notes[:2]:
        lines.append(f"MoE: {note}")
    for note in spec.layer_notes[:1]:
        lines.append(f"Layers: {note}")
    lines.extend(_analysis_note_lines(spec))
    return lines


def _format_analysis_note(spec: ArchitectureSpec, note: str) -> str:
    """Use graph tile labels in AST notes instead of raw forward attr names."""
    if note.startswith("Forward order: "):
        return f"Forward order: {format_forward_sequence(spec)}"
    return note


def _analysis_note_lines(spec: ArchitectureSpec) -> list[str]:
    lines: list[str] = []
    for note in spec.analysis_notes[:1]:
        formatted = _format_analysis_note(spec, note)
        if spec.forward_sequence and formatted.startswith("Forward order: "):
            continue
        if "@op_" in formatted or formatted.startswith("@op "):
            continue
        lines.append(f"AST: {formatted}")
    return lines


def _highlight_lines(spec: ArchitectureSpec) -> list[str]:
    if not spec.highlights:
        return []
    return [f"Highlights: {'; '.join(spec.highlights)}"]


def _render_fact_sheet_html(spec: ArchitectureSpec) -> str:
    parts: list[str] = []
    name = _model_display_name(spec)
    if name:
        parts.append(f"<strong>{html.escape(name)}</strong>")
    for line in _fact_lines(spec):
        if line.startswith(_FACT_SUBLINE_INDENT):
            parts.append(f"  {html.escape(line[len(_FACT_SUBLINE_INDENT) :].strip())}")
            continue
        text = line.strip()
        if text.startswith("Checkpoint: "):
            parts.append(
                _format_source_line_html("Checkpoint", spec.checkpoint_source or "")
            )
            continue
        if text.startswith("GitHub code: "):
            parts.append(
                _format_source_line_html("GitHub code", spec.github_source or "")
            )
            continue
        parts.append(f"- {html.escape(text)}")
    parts.extend(html.escape(line.strip()) for line in _highlight_lines(spec))
    return "\n".join(parts)


def build_fact_sheet_viewer(spec: ArchitectureSpec) -> dict[str, str]:
    """Return left-aligned fact sheet text for the HTML viewer panel."""
    body: list[str] = []
    name = _model_display_name(spec)
    if name:
        body.append(f"**{name}**")
    for line in _fact_lines(spec):
        if line.startswith(_FACT_SUBLINE_INDENT):
            body.append(f"  {line[len(_FACT_SUBLINE_INDENT) :].strip()}")
        else:
            body.append(f"- {line.strip()}")
    body.extend(line.strip() for line in _highlight_lines(spec))
    return {
        "title": "Fact sheet",
        "body": "\n".join(body),
        "bodyHtml": _render_fact_sheet_html(spec),
    }


_GENERATED_LABEL = "Generated"


def format_generated_line(generated_at: datetime) -> str:
    """Render the fact-sheet 'Generated: <date time>' line for a timestamp."""
    return f"{_GENERATED_LABEL}: {generated_at.strftime('%Y-%m-%d %H:%M:%S')}"


def _insert_after_model_name(
    text: str, line: str, is_name: Callable[[str], bool]
) -> str:
    """Insert ``line`` right after a leading model-name line, else at the top."""
    if not text:
        return line
    existing = text.split("\n")
    at = 1 if existing and is_name(existing[0]) else 0
    existing.insert(at, line)
    return "\n".join(existing)


def with_generated_timestamp(
    viewer: dict[str, str], generated_at: datetime
) -> dict[str, str]:
    """Return a copy of a fact-sheet viewer dict with a generated-time line added.

    The 'Generated' line is placed right after the bold model-name line (so the
    fact sheet reads: model name, then time, then the metadata bullets). It is
    added only to the viewer (HTML/body) forms, never to the persisted payload
    JSON, so exported payloads stay byte-reproducible.
    """
    line = format_generated_line(generated_at)
    updated = dict(viewer)
    updated["body"] = _insert_after_model_name(
        viewer.get("body", ""),
        f"- {line}",
        lambda first: first.startswith("**") and first.endswith("**"),
    )
    updated["bodyHtml"] = _insert_after_model_name(
        viewer.get("bodyHtml", ""),
        f"- {html.escape(line)}",
        lambda first: first.startswith("<strong>"),
    )
    return updated


def build_fact_sheet_group_attributes(spec: ArchitectureSpec) -> dict[str, str]:
    """Mirror the HTML fact sheet in Model Explorer graph info metadata."""
    viewer = build_fact_sheet_viewer(spec)
    return {"architecture_fact_sheet": viewer["body"]}
