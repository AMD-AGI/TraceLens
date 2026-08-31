"""Human-readable labels for kernel inline frames in Model Explorer exports."""

from __future__ import annotations

import re
from typing import Any

# Matches visualizer.kernel_pipeline._OP_LABEL_OVERRIDES display names.
KERNEL_FRAME_LABELS: dict[str, str] = {
    "l2norm_fwd": "L2Norm",
    "kda_gate_chunk_cumsum": "Gate cumsum",
    "fused_beta_sigmoid": "Fused beta sigmoid",
    "chunk_kda_fwd_intra": "Intra-chunk WY",
    "chunk_gated_delta_rule_fwd_h": "Gated delta rule h",
    "chunk_gla_fwd_o_gk": "Output o",
    "chunk_local_cumsum": "Chunk local cumsum",
}

# fla.modules.l2norm.l2norm_fwd_kernel: rstd = 1/sqrt(sum(x^2)+eps); y = x*rstd
L2NORM_SUBSTEP_LABELS = ("Sum sq", "Sqrt", "Inv sqrt", "Normalize")

# fla.ops.kda.gate.kda_gate_chunk_cumsum_vector_kernel (default path):
# gate = -exp(A_log) * softplus(g [+ bias]); out = cumsum(gate)
KDA_GATE_SUBSTEP_LABELS = {
    0: "Exp",
    1: "Softplus",
    2: "Gate mul",
    3: "Gate",
    4: "Chunk cumsum",
}

_UNICODE_LABEL_REPLACEMENTS = {
    "÷": "Inv sqrt",
    "×": "Multiply",
    "−": "Subtract",
}

_SUBSTEP_INDEX_RE = re.compile(r"_sub_(\d+)$")
_L2NORM_ATTR_RE = re.compile(r"forward_l2norm_fwd_[a-z]_sub_\d+$")
_KDA_GATE_ATTR_RE = re.compile(r"kda_gate_chunk_cumsum.*_sub_\d+$")


def _substep_index(node: dict[str, Any]) -> int | None:
    attr_name = _node_attr(node, "attr_name")
    if not attr_name:
        return None
    match = _SUBSTEP_INDEX_RE.search(attr_name)
    if match:
        return int(match.group(1))
    return None


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


def _kernel_frame_key(node: dict[str, Any], namespace: str) -> str | None:
    segment = _namespace_segment(namespace)
    if segment in KERNEL_FRAME_LABELS:
        return segment

    attr_name = _node_attr(node, "attr_name") or ""
    if _L2NORM_ATTR_RE.search(attr_name):
        return "l2norm_fwd"
    if _KDA_GATE_ATTR_RE.search(attr_name):
        return "kda_gate_chunk_cumsum"
    return None


def _apply_substep_label(node: dict[str, Any], frame_key: str, index: int) -> None:
    if frame_key == "l2norm_fwd" and index < len(L2NORM_SUBSTEP_LABELS):
        node["label"] = L2NORM_SUBSTEP_LABELS[index]
    elif frame_key == "kda_gate_chunk_cumsum" and index in KDA_GATE_SUBSTEP_LABELS:
        node["label"] = KDA_GATE_SUBSTEP_LABELS[index]


def apply_kernel_frame_labels(
    nodes: list[dict[str, Any]],
    group_node_attributes: dict[str, dict[str, str]] | None = None,
) -> None:
    """Fix kernel inline-frame group titles and substep labels for Model Explorer."""
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
                        "label": KERNEL_FRAME_LABELS[frame_key],
                        "operation": "gpu_kernel",
                    },
                )

        if _node_class(node) != "KernelSubOp":
            label = node.get("label")
            if isinstance(label, str) and label in _UNICODE_LABEL_REPLACEMENTS:
                node["label"] = _UNICODE_LABEL_REPLACEMENTS[label]
            continue

        index = _substep_index(node)
        if index is None:
            continue

        resolved_frame = frame_key or _kernel_frame_key(node, namespace)
        if resolved_frame is None:
            continue
        _apply_substep_label(node, resolved_frame, index)


def _node_class(node: dict[str, Any]) -> str | None:
    return _node_attr(node, "class_name")
