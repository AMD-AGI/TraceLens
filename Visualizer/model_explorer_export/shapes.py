"""Attach symbolic tensor shapes from shape inference to Model Explorer nodes."""

from __future__ import annotations

from typing import Any

from visualizer.block_tree import BlockNode
from visualizer.shape_inference import ShapeInferencer, TensorSpec


def format_shape(spec: TensorSpec) -> str:
    """Human-readable shape for node attrs (Model Explorer font-safe)."""
    return " x ".join(str(dim) for dim in spec.shape)


def format_shape_tensor(spec: TensorSpec) -> str:
    """Compact ``BxTxH`` shape for outputsMetadata tensor_shape conversion."""
    return "x".join(str(dim) for dim in spec.shape)


def format_shape_bracket(spec: TensorSpec) -> str:
    """Legacy bracket form kept for operator JSON consumers."""
    inner = ", ".join(str(dim) for dim in spec.shape)
    return f"[{inner}]"


def _apply_shape_attrs(node: dict[str, Any], spec: TensorSpec) -> None:
    shape_text = format_shape(spec)
    if not shape_text:
        return
    tensor_shape = format_shape_tensor(spec)
    attrs = [item for item in node.get("attrs", []) if item.get("key") not in {"output_shape", "output_dtype"}]
    attrs.append({"key": "output_shape", "value": shape_text})
    attrs.append({"key": "output_dtype", "value": spec.dtype})
    node["attrs"] = attrs
    node["outputsMetadata"] = [
        {
            "id": "0",
            "attrs": [
                {"key": "shape", "value": shape_text},
                {"key": "tensor_shape", "value": tensor_shape},
                {"key": "dtype", "value": spec.dtype},
            ],
        }
    ]


def annotate_nodes_with_shapes(
    nodes: list[dict[str, Any]],
    shape_specs: dict[str, TensorSpec],
    *,
    id_prefix: str,
) -> None:
    """Match exported nodes to computation-graph keys under ``id_prefix``."""
    if not shape_specs:
        return
    prefix = f"{id_prefix}/" if id_prefix else ""
    for node in nodes:
        node_id = node.get("id", "")
        if prefix:
            if not node_id.startswith(prefix):
                continue
            local_key = node_id[len(prefix) :]
        else:
            local_key = node_id
        spec = shape_specs.get(local_key)
        if spec is None:
            continue
        _apply_shape_attrs(node, spec)


def infer_block_tree_shapes(
    inferencer: ShapeInferencer,
    block_tree: BlockNode,
    *,
    title: str,
) -> dict[str, TensorSpec]:
    return inferencer.infer_block_tree(block_tree, title=title)
