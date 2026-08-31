"""Human-readable labels for kernel inline frames in Model Explorer exports."""

from __future__ import annotations

import re
from typing import Any

# Matches visualizer.kernel_pipeline._OP_LABEL_OVERRIDES display names.
KERNEL_FRAME_LABELS: dict[str, str] = {
    "l2norm_fwd": "L2Norm",
    "l2norm_fwd_q": "L2Norm (q)",
    "l2norm_fwd_k": "L2Norm (k)",
    "kda_gate_chunk_cumsum": "Gate cumsum",
    "fused_beta_sigmoid": "Fused beta sigmoid",
    "chunk_kda_fwd_intra": "Intra-chunk WY",
    "chunk_gated_delta_rule_fwd_h": "Gated delta rule h",
    "chunk_gla_fwd_o_gk": "Output o",
    "chunk_local_cumsum": "Chunk local cumsum",
}

# fla.modules.l2norm.l2norm_fwd_kernel: rstd = 1/sqrt(sum(x^2)+eps); y = x*rstd
L2NORM_SUBSTEP_LABELS = ("Sum sq", "Sqrt", "Inv sqrt", "X")

# fla.ops.kda.gate.kda_gate_chunk_cumsum_vector_kernel (default path):
# gate = -exp(A_log) * softplus(g [+ bias]); out = cumsum(gate)
KDA_GATE_SUBSTEP_LABELS = {
    0: "Exp",
    1: "Softplus",
    2: "Gate mul",
    3: "Gate",
    4: "Chunk cumsum",
}

FUSED_BETA_SUBSTEP_LABELS = {
    0: "Sigmoid",
    1: "x scale",
}

# Model Explorer WebGL MSDF font lacks many Unicode math glyphs.
_WEBGL_CHAR_REPLACEMENTS = (
    ("× scale", "x scale"),
    ("×", "x"),
    ("÷", "/"),
    ("−", "-"),
    ("→", "->"),
    ("•", "-"),
)

_UNICODE_LABEL_REPLACEMENTS = {
    "÷": "Inv sqrt",
    "×": "Multiply",
    "−": "Subtract",
}

_SUBSTEP_INDEX_RE = re.compile(r"_sub_(\d+)$")
_L2NORM_ATTR_RE = re.compile(r"forward_l2norm_fwd_[a-z]_sub_\d+$")
_KDA_GATE_ATTR_RE = re.compile(r"kda_gate_chunk_cumsum.*_sub_\d+$")
_FUSED_BETA_ATTR_RE = re.compile(r"forward_fused_beta_sigmoid.*_sub_\d+$")
_L2NORM_NAMESPACE_RE = re.compile(r"/l2norm_fwd$")


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


def _l2norm_frame_key(node: dict[str, Any]) -> str | None:
    attr_name = _node_attr(node, "attr_name") or ""
    node_id = node.get("id", "")
    if "forward_l2norm_fwd_q" in attr_name or "forward_l2norm_fwd_q" in node_id:
        return "l2norm_fwd_q"
    if "forward_l2norm_fwd_k" in attr_name or "forward_l2norm_fwd_k" in node_id:
        return "l2norm_fwd_k"
    if _L2NORM_ATTR_RE.search(attr_name):
        return "l2norm_fwd"
    return None


def _split_l2norm_namespace(namespace: str, frame_key: str) -> str:
    if frame_key not in {"l2norm_fwd_q", "l2norm_fwd_k"}:
        return namespace
    if namespace.endswith(f"/{frame_key}"):
        return namespace
    if _L2NORM_NAMESPACE_RE.search(namespace):
        return _L2NORM_NAMESPACE_RE.sub(f"/{frame_key}", namespace)
    if namespace == "l2norm_fwd":
        return frame_key
    return namespace


def _kernel_frame_key(node: dict[str, Any], namespace: str) -> str | None:
    l2norm_key = _l2norm_frame_key(node)
    if l2norm_key is not None:
        return l2norm_key

    segment = _namespace_segment(namespace)
    if segment in KERNEL_FRAME_LABELS:
        return segment

    attr_name = _node_attr(node, "attr_name") or ""
    if _FUSED_BETA_ATTR_RE.search(attr_name):
        return "fused_beta_sigmoid"
    if _KDA_GATE_ATTR_RE.search(attr_name):
        return "kda_gate_chunk_cumsum"
    return None


def _sanitize_webgl_label(label: str) -> str:
    if label in _UNICODE_LABEL_REPLACEMENTS:
        return _UNICODE_LABEL_REPLACEMENTS[label]
    sanitized = label
    for src, dst in _WEBGL_CHAR_REPLACEMENTS:
        sanitized = sanitized.replace(src, dst)
    return sanitized


def _apply_substep_label(node: dict[str, Any], frame_key: str, index: int) -> None:
    if frame_key in {"l2norm_fwd", "l2norm_fwd_q", "l2norm_fwd_k"} and index < len(L2NORM_SUBSTEP_LABELS):
        node["label"] = L2NORM_SUBSTEP_LABELS[index]
    elif frame_key == "kda_gate_chunk_cumsum" and index in KDA_GATE_SUBSTEP_LABELS:
        node["label"] = KDA_GATE_SUBSTEP_LABELS[index]
    elif frame_key == "fused_beta_sigmoid" and index in FUSED_BETA_SUBSTEP_LABELS:
        node["label"] = FUSED_BETA_SUBSTEP_LABELS[index]


def apply_kernel_frame_labels(
    nodes: list[dict[str, Any]],
    group_node_attributes: dict[str, dict[str, str]] | None = None,
) -> None:
    """Fix kernel inline-frame group titles and substep labels for Model Explorer."""
    for node in nodes:
        frame_key = _l2norm_frame_key(node)
        if frame_key in {"l2norm_fwd_q", "l2norm_fwd_k"}:
            namespace = node.get("namespace", "")
            if namespace:
                node["namespace"] = _split_l2norm_namespace(namespace, frame_key)

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

        label = node.get("label")
        if isinstance(label, str):
            node["label"] = _sanitize_webgl_label(label)

        if _node_class(node) != "KernelSubOp":
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
