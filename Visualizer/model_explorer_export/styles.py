###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared node styling helpers for Model Explorer export."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from visualizer.block_tree import BlockNode

_WHITE_TEXT = "#ffffff"
_DARK_TEXT = "#1a1a1a"

# Detail-tile color palette for Model Explorer nodes.
_BASIC_OP = "#bdc3c7"
_ATTENTION = "#5dade2"
# Model Explorer WebGL labels use dark text on pale GPU kernel / MoE fills.
# Match input-tile lightness (#d9e8f5) so black labels stay readable.
_GPU_KERNEL = "#f5d9d9"
_GPU_KERNEL_BORDER = "#d98888"
_LEGACY_GPU_KERNEL = "#8e44ad"
_SYNTHETIC = "#ecf0f1"
_INPUT = "#d9e8f5"
_LAYOUT_ONLY = "#ffffff"

_GPU_KERNEL_FILLS = frozenset(
    {
        _GPU_KERNEL.lower(),
        _LEGACY_GPU_KERNEL.lower(),
    }
)

# Backgrounds that should always use white label text.
_WHITE_TEXT_BACKGROUNDS = frozenset(
    {
        "#3a4550",  # ffn
        "#566573",  # detail border gray
        "#85929e",  # legacy torch_functional
        "#5dade2",  # expanded composite / attention blue
        "#95a5a6",  # residual gray
    }
)

ROLE_COLORS: dict[str, dict[str, str]] = {
    "embedding": {"backgroundColor": _INPUT, "textColor": _DARK_TEXT},
    "attention": {"backgroundColor": _ATTENTION, "textColor": _WHITE_TEXT},
    "ffn": {"backgroundColor": "#3a4550", "textColor": _WHITE_TEXT},
    "moe": {"backgroundColor": _GPU_KERNEL, "textColor": _DARK_TEXT},
    "norm": {"backgroundColor": _BASIC_OP, "textColor": _DARK_TEXT},
    "head": {"backgroundColor": "#d5dbdb", "textColor": _DARK_TEXT},
    "residual": {"backgroundColor": "#95a5a6", "textColor": _WHITE_TEXT},
    "positional": {"backgroundColor": _INPUT, "textColor": _DARK_TEXT},
}

# Operation-kind colors for detail tiles.
OPERATION_COLORS: dict[str, dict[str, str]] = {
    "gpu_kernel": {"backgroundColor": _GPU_KERNEL, "textColor": _DARK_TEXT},
    "nn_module": {"backgroundColor": _BASIC_OP, "textColor": _DARK_TEXT},
    "torch_functional": {"backgroundColor": _BASIC_OP, "textColor": _DARK_TEXT},
    "composite": {
        "backgroundColor": _ATTENTION,
        "textColor": _WHITE_TEXT,
        "borderColor": "#566573",
    },
    "synthetic": {"backgroundColor": _SYNTHETIC, "textColor": _DARK_TEXT},
}


def ensure_readable_text(style: dict[str, str]) -> dict[str, str]:
    """Normalize fills and pick label colors Model Explorer can render legibly."""
    resolved = dict(style)
    background = resolved.get("backgroundColor", "").lower()
    if background in _GPU_KERNEL_FILLS:
        resolved["backgroundColor"] = _GPU_KERNEL
        resolved["textColor"] = _DARK_TEXT
        return resolved
    if background in _WHITE_TEXT_BACKGROUNDS:
        resolved["textColor"] = _WHITE_TEXT
    return resolved


def is_layout_only_label(label: str) -> bool:
    """True for ops that only rearrange or retype a tensor, computing no values."""
    from visualizer.ast_analyze import LAYOUT_ONLY_LABELS

    return (label or "").strip() in LAYOUT_ONLY_LABELS


def operation_tile_style(label: str) -> dict[str, str]:
    """Color an operation tile by whether it computes values or only moves them."""
    if is_layout_only_label(label):
        return {"backgroundColor": _LAYOUT_ONLY, "textColor": _DARK_TEXT}
    return {"backgroundColor": _BASIC_OP, "textColor": _DARK_TEXT}


def finalize_graph_node_styles(nodes: list[dict[str, Any]]) -> None:
    """Normalize label colors and give every operation tile a consistent fill.

    Nodes synthesized during merge carry no style of their own, so without this
    an identical op renders gray in one block and white in another.
    """
    for node in nodes:
        style = node.get("style")
        if not isinstance(style, dict):
            if style is None:
                node["style"] = ensure_readable_text(
                    operation_tile_style(str(node.get("label", "")))
                )
            continue
        if (
            is_layout_only_label(str(node.get("label", "")))
            and style.get("backgroundColor") == _BASIC_OP
        ):
            style = {**style, "backgroundColor": _LAYOUT_ONLY}
        node["style"] = ensure_readable_text(style)


def _exact_namespace_regex(namespace: str) -> str:
    return f"^{re.escape(namespace)}$"


def build_group_node_configs(
    *,
    decoder_namespace: str,
    group_node_attributes: dict[str, dict[str, str]],
    role_configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build namespace group styling with most-specific regexes first."""
    configs: list[dict[str, Any]] = []

    for namespace in sorted(group_node_attributes, key=len, reverse=True):
        if not namespace:
            continue
        attrs = group_node_attributes[namespace]
        operation = attrs.get("operation")
        if operation in {"gpu_kernel", "kernel pipeline"}:
            configs.append(
                {
                    "namespaceRegex": _exact_namespace_regex(namespace),
                    "backgroundColor": _GPU_KERNEL,
                    "textColor": _DARK_TEXT,
                    "layoutDirection": "TOP_BOTTOM",
                }
            )
            continue
        segment = namespace.rsplit("/", 1)[-1]
        if segment in {"KimiSparseMoeBlock", "KimiMoEGate"} or attrs.get(
            "label", ""
        ).endswith("MoEGate"):
            configs.append(
                {
                    "namespaceRegex": _exact_namespace_regex(namespace),
                    "backgroundColor": _GPU_KERNEL,
                    "textColor": _DARK_TEXT,
                    "borderColor": _GPU_KERNEL_BORDER,
                    "layoutDirection": "TOP_BOTTOM",
                }
            )
            continue
        if segment in {"KimiDeltaAttention", "KimiMLAAttention"}:
            configs.append(
                {
                    "namespaceRegex": _exact_namespace_regex(namespace),
                    "backgroundColor": _ATTENTION,
                    "textColor": _WHITE_TEXT,
                    "layoutDirection": "TOP_BOTTOM",
                }
            )

    anchored_role_configs: list[dict[str, Any]] = []
    for config in role_configs:
        regex = str(config.get("namespaceRegex", ""))
        if regex.startswith("^") and regex.endswith("$"):
            anchored_role_configs.append(config)
        else:
            anchored = dict(config)
            anchored["namespaceRegex"] = f"^{regex}$"
            anchored_role_configs.append(anchored)
    configs.extend(anchored_role_configs)

    configs.append(
        {
            "namespaceRegex": _exact_namespace_regex(decoder_namespace),
            "backgroundColor": "#fff5f4",
            "borderColor": "#c0392b",
            "textColor": _DARK_TEXT,
            "layoutDirection": "TOP_BOTTOM",
        }
    )
    return configs


def spine_tile_style() -> dict[str, str]:
    """Neutral gray fill for flat overview spine tiles."""
    return {"backgroundColor": _BASIC_OP, "textColor": _DARK_TEXT}


def input_port_style() -> dict[str, str]:
    """Synthetic @input ports on the model spine."""
    return {"backgroundColor": _INPUT, "textColor": _DARK_TEXT}


def output_port_style() -> dict[str, str]:
    """Synthetic @output ports on graph and subgraph boundaries."""
    return {"backgroundColor": "#d5f5e3", "textColor": _DARK_TEXT}


def detail_tile_style(
    block: BlockNode | None,
    *,
    synthetic: str | None = None,
    label: str = "",
) -> dict[str, str]:
    """Pick detail-tile colors from block role and operation kind."""
    from visualizer.model_graph import (
        OperationKind,
        _COMBINE_LABELS,
        classify_operation,
    )

    if synthetic == "@input":
        return input_port_style()
    if synthetic == "@output":
        return output_port_style()
    if synthetic == "@kernel_port":
        return {
            "backgroundColor": "#e8daef",
            "textColor": _DARK_TEXT,
            "borderColor": "#7d3c98",
        }
    if synthetic == "@kernel_port_in":
        return input_port_style()
    if synthetic == "@kernel_port_out":
        return output_port_style()
    if synthetic == "@loop_carried":
        return {
            "backgroundColor": "#f9e79f",
            "textColor": _DARK_TEXT,
            "borderColor": "#b7950b",
        }

    # Combine tiles (Multiply, Add, …) use the same gray as Linear/RMSNorm.
    if synthetic == "@combine" or label in _COMBINE_LABELS:
        return {"backgroundColor": _BASIC_OP, "textColor": _DARK_TEXT}

    operation = classify_operation(
        block,
        synthetic=synthetic,
        label=label or (block.label if block is not None else ""),
    )
    if operation == OperationKind.SYNTHETIC:
        return {"backgroundColor": _SYNTHETIC, "textColor": _DARK_TEXT}
    if operation == OperationKind.GPU_KERNEL:
        return {"backgroundColor": _GPU_KERNEL, "textColor": _DARK_TEXT}
    if operation == OperationKind.COMPOSITE and block is not None and block.children:
        return {"backgroundColor": _ATTENTION, "textColor": _WHITE_TEXT}
    return {"backgroundColor": _BASIC_OP, "textColor": _DARK_TEXT}
