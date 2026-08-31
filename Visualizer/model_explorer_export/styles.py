"""Shared node styling helpers for Model Explorer export."""

from __future__ import annotations

_WHITE_TEXT = "#ffffff"
_DARK_TEXT = "#1a1a1a"

# Backgrounds that should always use white label text.
_WHITE_TEXT_BACKGROUNDS = frozenset(
    {
        "#3a4550",  # ffn
        "#566573",  # gpu_kernel / detail border gray
        "#85929e",  # torch_functional
        "#8e44ad",  # moe purple
        "#5dade2",  # attention blue
        "#95a5a6",  # residual gray
    }
)

ROLE_COLORS: dict[str, dict[str, str]] = {
    "embedding": {"backgroundColor": "#d9e8f5", "textColor": _DARK_TEXT},
    "attention": {"backgroundColor": "#5dade2", "textColor": _WHITE_TEXT},
    "ffn": {"backgroundColor": "#3a4550", "textColor": _WHITE_TEXT},
    "moe": {"backgroundColor": "#8e44ad", "textColor": _WHITE_TEXT},
    "norm": {"backgroundColor": "#f5b041", "textColor": _DARK_TEXT},
    "head": {"backgroundColor": "#d5dbdb", "textColor": _DARK_TEXT},
    "residual": {"backgroundColor": "#95a5a6", "textColor": _WHITE_TEXT},
    "positional": {"backgroundColor": "#d9e8f5", "textColor": _DARK_TEXT},
}

OPERATION_COLORS: dict[str, dict[str, str]] = {
    "gpu_kernel": {"backgroundColor": "#566573", "textColor": _WHITE_TEXT},
    "nn_module": {"backgroundColor": "#bdc3c7", "textColor": _DARK_TEXT},
    "torch_functional": {"backgroundColor": "#85929e", "textColor": _WHITE_TEXT},
    "composite": {"backgroundColor": "#f4f6f7", "textColor": _DARK_TEXT, "borderColor": "#566573"},
    "synthetic": {"backgroundColor": "#ecf0f1", "textColor": _DARK_TEXT},
}


def ensure_readable_text(style: dict[str, str]) -> dict[str, str]:
    """Force white text on dark gray and purple node fills."""
    resolved = dict(style)
    background = resolved.get("backgroundColor", "").lower()
    if background in _WHITE_TEXT_BACKGROUNDS:
        resolved["textColor"] = _WHITE_TEXT
    return resolved
