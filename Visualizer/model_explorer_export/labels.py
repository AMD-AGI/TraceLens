###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Human-readable labels for kernel inline frames in Model Explorer exports."""

from __future__ import annotations

import re
from typing import Any

from visualizer.kernel_pipeline import (
    _OP_LABEL_OVERRIDES,
    kernel_op_display_label,
    tensor_port_kernel_frame_label,
)

# Model Explorer WebGL MSDF font lacks many Unicode math glyphs.
_WEBGL_CHAR_REPLACEMENTS = (
    ("× scale", "x scale"),
    ("×", "x"),
    ("÷", "/"),
    ("−", "-"),
    ("→", "->"),
    ("•", "-"),
)

# AST-derived kernel sub-op symbols → WebGL-safe operator notation.
_KERNEL_SUBOP_LABELS = {
    "÷": "1/x",
    "/": "1/x",
    "×": "X",
    "x": "X",
    "*": "X",
}


_TENSOR_PORT_FRAME_KEY_RE = re.compile(r"(?P<stem>.+)_fwd_(?P<port>[a-z])$")
_MERGED_TENSOR_PORT_PARENT_RE = re.compile(r"forward_(?P<stem>.+)_fwd_(?P<port>[a-z])")


def _node_attr(node: dict[str, Any], key: str) -> str | None:
    for attr in node.get("attrs", []):
        if attr.get("key") != key:
            continue
        value = attr.get("value")
        if isinstance(value, str):
            return value
    return None


def _namespace_segment(namespace: str) -> str:
    return namespace.rsplit("/", 1)[-1] if namespace else ""


def _frame_owner_attr_name(node: dict[str, Any]) -> str:
    attr_name = _node_attr(node, "attr_name") or ""
    if "_sub_" in attr_name:
        return attr_name.rsplit("_sub_", 1)[0]
    node_id = node.get("id", "")
    if ":pipeline:" in node_id:
        segment = node_id.rsplit(":pipeline:", 1)[-1]
        if "_sub_" in segment:
            return segment.rsplit("_sub_", 1)[0]
        if segment.endswith("/@input"):
            return segment[: -len("/@input")]
    return attr_name


def tensor_port_frame_key(node: dict[str, Any]) -> str | None:
    """Frame key for kernels instantiated once per tensor port (e.g. l2norm_fwd_q)."""
    for text in (_frame_owner_attr_name(node), node.get("id", "")):
        if not text:
            continue
        tail = text.rsplit(":", 1)[-1]
        if tail.endswith("/@input"):
            tail = tail[: -len("/@input")]
        label = tensor_port_kernel_frame_label(tail)
        if label is not None:
            return label
        match = _MERGED_TENSOR_PORT_PARENT_RE.search(tail)
        if match is not None:
            return f"{match.group('stem')}_fwd_{match.group('port')}"
    return None


def split_tensor_port_namespace(namespace: str, frame_key: str) -> str:
    """Split a shared parent namespace into a per-port child namespace."""
    match = _TENSOR_PORT_FRAME_KEY_RE.fullmatch(frame_key)
    if match is None:
        return namespace
    parent = f"{match.group('stem')}_fwd"
    if namespace.endswith(f"/{frame_key}"):
        return namespace
    if namespace.endswith(f"/{parent}"):
        return f"{namespace[: -len(parent)]}{frame_key}"
    if namespace == parent:
        return frame_key
    return namespace


def frame_group_label(frame_key: str) -> str:
    """Human-readable title for an expanded kernel inline frame."""
    port_match = _TENSOR_PORT_FRAME_KEY_RE.fullmatch(frame_key)
    if port_match is not None:
        base_key = f"{port_match.group('stem')}_fwd"
        return f"{kernel_op_display_label(base_key)} ({port_match.group('port')})"
    if frame_key in _OP_LABEL_OVERRIDES:
        return _OP_LABEL_OVERRIDES[frame_key]
    return kernel_op_display_label(frame_key)


def kernel_subop_display_label(label: str) -> str:
    """Map AST-derived kernel sub-op labels to WebGL-safe operator notation."""
    stripped = label.strip()
    if stripped in _KERNEL_SUBOP_LABELS:
        return _KERNEL_SUBOP_LABELS[stripped]
    sanitized = stripped
    for src, dst in _WEBGL_CHAR_REPLACEMENTS:
        sanitized = sanitized.replace(src, dst)
    return sanitized


def tensor_port_input_label(namespace: str) -> str | None:
    """Input-port label for a per-tensor kernel frame namespace."""
    match = _TENSOR_PORT_FRAME_KEY_RE.fullmatch(_namespace_segment(namespace))
    if match is not None:
        return match.group("port")
    return None


def skip_merged_tensor_port_parent(
    namespace: str, group_nodes: list[dict[str, Any]]
) -> bool:
    """Skip injecting a shared parent @input when port-specific frames exist."""
    leaf = _namespace_segment(namespace)
    if not leaf.endswith("_fwd") or _TENSOR_PORT_FRAME_KEY_RE.fullmatch(leaf):
        return False
    stem = leaf[: -len("_fwd")]
    pattern = re.compile(rf"forward_{re.escape(stem)}_fwd_(?P<port>[a-z])")
    return any(pattern.search(node.get("id", "")) for node in group_nodes)


def _kernel_frame_key(node: dict[str, Any], namespace: str) -> str | None:
    port_key = tensor_port_frame_key(node)
    if port_key is not None:
        return port_key

    segment = _namespace_segment(namespace)
    if segment in _OP_LABEL_OVERRIDES:
        return segment

    attr_name = _node_attr(node, "attr_name") or ""
    for key in _OP_LABEL_OVERRIDES:
        if key in attr_name:
            return key
    return None


def apply_kernel_frame_labels(
    nodes: list[dict[str, Any]],
    group_node_attributes: dict[str, dict[str, str]] | None = None,
) -> None:
    """Fix kernel inline-frame group titles and substep labels for Model Explorer."""
    for node in nodes:
        frame_key = tensor_port_frame_key(node)
        if frame_key is None:
            continue
        namespace = node.get("namespace", "")
        if namespace:
            node["namespace"] = split_tensor_port_namespace(namespace, frame_key)

    seen_namespaces: set[str] = set()
    for node in nodes:
        namespace = node.get("namespace", "")
        frame_key = _kernel_frame_key(node, namespace) if namespace else None

        if frame_key and namespace and namespace not in seen_namespaces:
            seen_namespaces.add(namespace)
            if group_node_attributes is not None:
                group_node_attributes.setdefault(
                    namespace,
                    {
                        "label": frame_group_label(frame_key),
                        "operation": "gpu_kernel",
                    },
                )

        if _node_class(node) != "KernelSubOp":
            continue

        label = node.get("label")
        if isinstance(label, str):
            node["label"] = kernel_subop_display_label(label)


def _node_class(node: dict[str, Any]) -> str | None:
    return _node_attr(node, "class_name")
