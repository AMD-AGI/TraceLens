"""Neutral model graph IR decoupled from layout and rendering.

The graph contains only structural information: node kind, label, edges, and
optional metadata. Rendering and Sugiyama layout operate on ``ComputationGraph``;
this module is the serializable export layer built from ``BlockNode`` trees.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

from visualizer.basic_ops import BasicOpFilter, introspect_is_modeling_operation
from visualizer.ast_analyze import (
    is_forward_operation,
    is_functional_synthetic,
    is_torch_native_attention_kernel,
    kernel_name_from_step_details,
)
from visualizer.extract import architecture_section_trees

if TYPE_CHECKING:
    from visualizer.block_tree import BlockNode
    from visualizer.computation_graph import ComputationGraph, GraphNodeSpec, InlineFrameSpec
    from visualizer.extract import ArchitectureSpec


class NodeKind(str, Enum):
    """Structural role of a node in the model graph."""

    LEAF = "leaf"
    BLOCK = "block"
    INLINE = "inline"
    SUBGRAPH = "subgraph"
    TOP_LEVEL = "top_level"


class OperationKind(str, Enum):
    """Reduced operation category for leaf compute nodes."""

    NN_MODULE = "nn_module"
    TORCH_FUNCTIONAL = "torch_functional"
    GPU_KERNEL = "gpu_kernel"
    SYNTHETIC = "synthetic"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"


_REDUCED_OPERATION_KINDS = frozenset(
    {
        OperationKind.NN_MODULE,
        OperationKind.TORCH_FUNCTIONAL,
        OperationKind.GPU_KERNEL,
    }
)

_SYNTHETIC_KEYS = frozenset(
    {"@input", "@output", "@hidden_states", "@tensor", "@combine"}
)
_COMBINE_LABELS = frozenset({"×", "+", "Elementwise ×", "Multiply", "Add"})
_KERNEL_CLASS_NAMES = frozenset(
    {
        "KernelOp",
        "KernelSubOp",
        "KernelOutput",
        "AttentionOp",
        "KernelPipeline",
        "AttentionMerge",
    }
)
_TORCH_FUNCTIONAL_RE = re.compile(
    r"(?i)(?:^F\.|torch\.nn\.functional\.|functional\.|@functional_)"
)
_NN_MODULE_CLASS_RE = re.compile(
    r"(?i)^(Linear|Embedding|Conv\d*d|Dropout|Identity|RMSNorm|LayerNorm|Parameter)$"
)
# Primitive torch.nn / torch.functional ops. Custom fused kernels whose names
# merely mention these words (e.g. "Fused beta sigmoid", "Gate cumsum") stay kernels.
_TORCH_PRIMITIVE_LABELS = frozenset(
    {
        "exp",
        "exp2",
        "softplus",
        "sigmoid",
        "hardsigmoid",
        "cumsum",
        "cumprod",
        "softmax",
        "logsoftmax",
        "tanh",
        "relu",
        "relu6",
        "gelu",
        "gelunew",
        "silu",
        "swish",
        "elu",
        "selu",
        "mish",
        "contiguous",
        "sqrt",
        "rsqrt",
        "sum",
        "mean",
        "log",
        "abs",
        "clamp",
        "dropout",
        "identity",
        "leakyrelu",
        "hardtanh",
    }
)
_TORCH_PRIMITIVE_SYMBOLS = frozenset({"×", "÷", "+", "−", "^", "× scale"})


@dataclass
class GraphEdge:
    source: str
    target: str
    style: str = "solid"
    label: str | None = None


@dataclass
class InlineFrame:
    frame_id: str
    label: str
    node_ids: list[str] = field(default_factory=list)
    sublabel: str | None = None


@dataclass
class ModelGraphNode:
    id: str
    kind: NodeKind
    label: str
    operation: OperationKind | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelGraph:
    title: str
    nodes: list[ModelGraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    inline_frames: list[InlineFrame] = field(default_factory=list)
    subgraphs: dict[str, ModelGraph] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "nodes": [
                {
                    "id": node.id,
                    "kind": node.kind.value,
                    "label": node.label,
                    **({"operation": node.operation.value} if node.operation is not None else {}),
                    **({"metadata": node.metadata} if node.metadata else {}),
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "style": edge.style,
                    **({"label": edge.label} if edge.label else {}),
                }
                for edge in self.edges
            ],
            "inline_frames": [
                {
                    "frame_id": frame.frame_id,
                    "label": frame.label,
                    "node_ids": frame.node_ids,
                    **({"sublabel": frame.sublabel} if frame.sublabel else {}),
                }
                for frame in self.inline_frames
            ],
            "subgraphs": {
                key: subgraph.to_dict() for key, subgraph in self.subgraphs.items()
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent) + "\n"


def classify_operation(
    block: BlockNode | None,
    *,
    synthetic: str | None = None,
    label: str = "",
) -> OperationKind:
    """Classify one graph vertex into a reduced operation category."""
    if synthetic in _SYNTHETIC_KEYS or label in _COMBINE_LABELS:
        return OperationKind.SYNTHETIC
    if block is None:
        return OperationKind.UNKNOWN

    details = list(block.details or [])
    # Expansion is the strongest visual signal: the parent represents a composite
    # module while each child keeps its own leaf classification.
    if block.children:
        return OperationKind.COMPOSITE

    if (
        is_functional_synthetic(block.attr_name)
        or is_forward_operation(block.attr_name)
        or any(_TORCH_FUNCTIONAL_RE.search(line) for line in details)
        or _is_torch_library_operation(block, details)
    ):
        return OperationKind.TORCH_FUNCTIONAL

    if block.class_name in _KERNEL_CLASS_NAMES or any(line.lower().startswith("kernel:") for line in details):
        return OperationKind.GPU_KERNEL

    if block.is_basic or _NN_MODULE_CLASS_RE.match(block.class_name or ""):
        return OperationKind.NN_MODULE

    if introspect_is_modeling_operation(block.class_name, block.attr_name, details):
        return OperationKind.COMPOSITE

    if _NN_MODULE_CLASS_RE.search(block.class_name or ""):
        return OperationKind.NN_MODULE

    return OperationKind.UNKNOWN


def _normalized_op_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def is_torch_primitive_label(label: str) -> bool:
    """True for a standalone torch.nn / torch.functional primitive name."""
    text = (label or "").strip()
    if text in _TORCH_PRIMITIVE_SYMBOLS:
        return True
    return _normalized_op_label(text) in _TORCH_PRIMITIVE_LABELS


def _is_torch_library_operation(block: BlockNode, details: list[str]) -> bool:
    """True for torch.nn / F.* ops, including torch's own attention, excluding library kernels.

    Attention counts as torch only when the source or the checkpoint says so. An
    attention step nobody could resolve is far more often a fused library kernel than
    a torch call, so it stays a kernel rather than being assumed into the torch bucket.
    """
    if is_torch_primitive_label(block.label or ""):
        return True
    if block.class_name == "AttentionOp":
        return is_torch_native_attention_kernel(kernel_name_from_step_details(details))
    return False


def _edge_style(
    graph: ComputationGraph,
    source: int,
    target: int,
) -> str:
    pair = (source, target)
    if pair in graph.dashed_links:
        return "dashed"
    return "solid"


def _minimal_metadata(spec: GraphNodeSpec) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    block = spec.block
    if block is not None and block.class_name and block.class_name not in {spec.label, block.label}:
        metadata["class_name"] = block.class_name
    if block is not None and is_forward_operation(block.attr_name):
        metadata["attr_name"] = block.attr_name
        if block.details:
            metadata["details"] = list(block.details)
        if block.external_inputs:
            metadata["external_inputs"] = list(block.external_inputs)
    if spec.port_label:
        metadata["port_label"] = spec.port_label
    if spec.port_style:
        metadata["port_style"] = spec.port_style
    if spec.synthetic:
        metadata["synthetic"] = spec.synthetic
    return metadata


def _node_kind_for_spec(
    spec: GraphNodeSpec,
    *,
    root: BlockNode,
    inline_member_indices: set[int],
    node_index: int,
    subgraph_keys: set[str],
) -> NodeKind:
    if node_index == 0 and spec.key in {root.attr_name, f"{root.attr_name}:0"}:
        return NodeKind.TOP_LEVEL
    block = spec.block
    if block is not None and block.attr_name in subgraph_keys:
        return NodeKind.SUBGRAPH
    if node_index in inline_member_indices:
        return NodeKind.INLINE
    if block is not None and block.is_basic and not block.children:
        return NodeKind.LEAF
    if spec.synthetic in _SYNTHETIC_KEYS or spec.label in _COMBINE_LABELS:
        return NodeKind.LEAF
    if block is not None and block.children:
        return NodeKind.BLOCK
    return NodeKind.LEAF


def _convert_inline_frames(
    frames: list[InlineFrameSpec],
    index_to_id: dict[int, str],
) -> list[InlineFrame]:
    converted: list[InlineFrame] = []
    for frame in frames:
        converted.append(
            InlineFrame(
                frame_id=frame.frame_id,
                label=frame.label,
                sublabel=frame.sublabel,
                node_ids=[index_to_id[index] for index in frame.node_indices if index in index_to_id],
            )
        )
    return converted


def build_model_graph(
    root: BlockNode,
    *,
    title: str = "",
    basic_ops: BasicOpFilter | None = None,
    include_subgraphs: bool = True,
) -> ModelGraph:
    """Build a serializable model graph from a block tree."""
    from visualizer.block_tree import collect_nested_diagrams
    from visualizer.computation_graph import build_computation_graph

    resolved_basic_ops = basic_ops or BasicOpFilter.for_detailed()
    computation_graph = build_computation_graph(root, basic_ops=resolved_basic_ops)

    subgraph_blocks: dict[str, BlockNode] = {}
    if include_subgraphs:
        for _, block in collect_nested_diagrams(root, basic_ops=resolved_basic_ops):
            subgraph_blocks[block.attr_name] = block

    inline_member_indices: set[int] = set()
    for frame in computation_graph.inline_frames:
        inline_member_indices.update(frame.node_indices)

    index_to_id: dict[int, str] = {}
    for index, spec in enumerate(computation_graph.nodes):
        index_to_id[index] = spec.key or f"node:{index}"

    nodes: list[ModelGraphNode] = []
    for index, spec in enumerate(computation_graph.nodes):
        node_id = index_to_id[index]
        kind = _node_kind_for_spec(
            spec,
            root=root,
            inline_member_indices=inline_member_indices,
            node_index=index,
            subgraph_keys=set(subgraph_blocks),
        )
        operation = classify_operation(spec.block, synthetic=spec.synthetic, label=spec.label)
        metadata = _minimal_metadata(spec)
        if spec.block is not None and spec.block.attr_name in subgraph_blocks:
            metadata["subgraph_key"] = spec.block.attr_name
        nodes.append(
            ModelGraphNode(
                id=node_id,
                kind=kind,
                label=spec.label or (spec.block.label if spec.block is not None else node_id),
                operation=operation,
                metadata=metadata,
            )
        )

    edges: list[GraphEdge] = []
    for source, target in computation_graph.links:
        if source not in index_to_id or target not in index_to_id:
            continue
        port_label = computation_graph.link_port_labels.get((source, target))
        edges.append(
            GraphEdge(
                source=index_to_id[source],
                target=index_to_id[target],
                style=_edge_style(computation_graph, source, target),
                label=port_label,
            )
        )

    inline_frames = _convert_inline_frames(computation_graph.inline_frames, index_to_id)
    subgraphs: dict[str, ModelGraph] = {}
    if include_subgraphs:
        for key, block in subgraph_blocks.items():
            subgraphs[key] = build_model_graph(
                block,
                title=block.label,
                basic_ops=resolved_basic_ops,
                include_subgraphs=True,
            )

    return ModelGraph(
        title=title or root.label,
        nodes=nodes,
        edges=edges,
        inline_frames=inline_frames,
        subgraphs=subgraphs,
    )


def build_architecture_model_graphs(
    spec: ArchitectureSpec,
    *,
    basic_ops: BasicOpFilter | None = None,
) -> dict[str, Any]:
    """Build model graphs for every detailed section in an architecture spec."""
    from visualizer.block_tree import subgraph_warrants_json_export

    resolved_basic_ops = basic_ops or spec.basic_ops
    sections: list[dict[str, Any]] = []
    for section_title, block_tree in architecture_section_trees(spec):
        if not subgraph_warrants_json_export(block_tree, basic_ops=resolved_basic_ops):
            continue
        graph = build_model_graph(
            block_tree,
            title=section_title,
            basic_ops=resolved_basic_ops,
        )
        sections.append(
            {
                "title": section_title,
                "graph": graph.to_dict(),
            }
        )
    return {
        "name": spec.name,
        "model_type": spec.model_type,
        "sections": sections,
    }


def save_model_graph(graph: ModelGraph, path: Path | str) -> Path:
    """Write one model graph to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(graph.to_json(), encoding="utf-8")
    return target


def save_architecture_model_graphs(payload: dict[str, Any], path: Path | str) -> Path:
    """Write architecture-level model graphs to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


@dataclass
class OperationValidationIssue:
    graph_title: str
    node_id: str
    label: str
    operation: OperationKind
    reason: str


def collect_non_reduced_operations(graph: ModelGraph, *, graph_title: str | None = None) -> list[OperationValidationIssue]:
    """Return leaf nodes that are not reduced to nn / torch functional / GPU kernel."""
    title = graph_title or graph.title
    issues: list[OperationValidationIssue] = []

    for node in graph.nodes:
        if node.kind == NodeKind.SUBGRAPH:
            subgraph_key = node.metadata.get("subgraph_key")
            if subgraph_key and subgraph_key in graph.subgraphs:
                issues.extend(
                    collect_non_reduced_operations(
                        graph.subgraphs[subgraph_key],
                        graph_title=f"{title}/{subgraph_key}",
                    )
                )
            continue

        if node.operation in {OperationKind.SYNTHETIC, None}:
            continue
        if node.operation in _REDUCED_OPERATION_KINDS:
            continue
        if node.kind in {NodeKind.BLOCK, NodeKind.TOP_LEVEL} and node.operation == OperationKind.COMPOSITE:
            continue

        reason = "composite block not expanded to leaf ops"
        if node.operation == OperationKind.UNKNOWN:
            reason = "could not classify operation from AST/block tree"
        issues.append(
            OperationValidationIssue(
                graph_title=title,
                node_id=node.id,
                label=node.label,
                operation=node.operation,
                reason=reason,
            )
        )

    for subgraph in graph.subgraphs.values():
        issues.extend(collect_non_reduced_operations(subgraph))

    return issues


def assert_operations_reduced(graph: ModelGraph) -> None:
    """Raise AssertionError when any operational leaf is not a reduced op kind."""
    issues = collect_non_reduced_operations(graph)
    if not issues:
        return
    lines = [
        f"{issue.graph_title}:{issue.node_id} ({issue.label!r}) -> {issue.operation.value}: {issue.reason}"
        for issue in issues
    ]
    raise AssertionError("Non-reduced operations found:\n" + "\n".join(lines))
