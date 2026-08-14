"""Symbolic tensor shape and dtype inference for model computation graphs."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

from visualizer.model_graph import (
    ModelGraph,
    ModelGraphNode,
    NodeKind,
    OperationKind,
    build_model_graph,
    classify_operation,
)

if TYPE_CHECKING:
    from visualizer.ast_analyze import ClassStructure
    from visualizer.block_tree import BlockNode
    from visualizer.extract import ArchitectureSpec


DimExpr = int | str


class Symbol(str, Enum):
    """Common symbolic dimensions propagated through the graph."""

    BATCH = "B"
    SEQ = "T"
    HIDDEN = "H"
    VOCAB = "V"
    HEADS = "N"
    KV_HEADS = "K"
    HEAD_DIM = "D"
    INTERMEDIATE = "I"
    EXPERTS = "E"
    EXPERTS_PER_TOK = "TopK"


@dataclass(frozen=True)
class TensorSpec:
    """Shape and dtype for one tensor."""

    shape: tuple[DimExpr, ...]
    dtype: str = "float16"

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": [_serialize_dim(dim) for dim in self.shape],
            "dtype": self.dtype,
        }


@dataclass
class ModuleLinearSpec:
    in_features: DimExpr
    out_features: DimExpr


@dataclass
class ModuleEmbeddingSpec:
    num_embeddings: DimExpr
    embedding_dim: DimExpr


@dataclass
class ShapeContext:
    """Resolved and symbolic dimensions derived from model config."""

    dims: dict[str, DimExpr] = field(default_factory=dict)
    dtype: str = "float16"

    @classmethod
    def from_spec(cls, spec: ArchitectureSpec) -> ShapeContext:
        config = spec.raw_config or {}
        dtype = _config_dtype(config)
        dims: dict[str, DimExpr] = {
            Symbol.BATCH.value: Symbol.BATCH.value,
            Symbol.SEQ.value: Symbol.SEQ.value,
        }
        if spec.hidden_size is not None:
            dims[Symbol.HIDDEN.value] = spec.hidden_size
        if spec.vocab_size is not None:
            dims[Symbol.VOCAB.value] = spec.vocab_size
        if spec.num_attention_heads is not None:
            dims[Symbol.HEADS.value] = spec.num_attention_heads
        if spec.num_key_value_heads is not None:
            dims[Symbol.KV_HEADS.value] = spec.num_key_value_heads
        if spec.head_dim is not None:
            dims[Symbol.HEAD_DIM.value] = spec.head_dim
        elif spec.hidden_size and spec.num_attention_heads:
            dims[Symbol.HEAD_DIM.value] = spec.hidden_size // spec.num_attention_heads
        if spec.intermediate_size is not None:
            dims[Symbol.INTERMEDIATE.value] = spec.intermediate_size
        if spec.moe_intermediate_size is not None:
            dims.setdefault(Symbol.INTERMEDIATE.value, spec.moe_intermediate_size)
        if spec.num_experts is not None:
            dims[Symbol.EXPERTS.value] = spec.num_experts
        if spec.num_experts_per_tok is not None:
            dims[Symbol.EXPERTS_PER_TOK.value] = spec.num_experts_per_tok

        for key, value in config.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                dims[key] = value
            elif isinstance(value, float) and value.is_integer():
                dims[key] = int(value)

        return cls(dims=dims, dtype=dtype)


@dataclass
class ModuleDimRegistry:
    """Linear and embedding constructor dimensions parsed from modeling AST."""

    linear: dict[tuple[str, str], ModuleLinearSpec] = field(default_factory=dict)
    linear_by_attr: dict[str, ModuleLinearSpec] = field(default_factory=dict)
    embedding: dict[tuple[str, str], ModuleEmbeddingSpec] = field(default_factory=dict)
    embedding_by_attr: dict[str, ModuleEmbeddingSpec] = field(default_factory=dict)

    @classmethod
    def from_registry(
        cls,
        class_registry: dict[str, ClassStructure],
        *,
        config: dict[str, Any],
        context: ShapeContext,
    ) -> ModuleDimRegistry:
        registry = cls()
        for class_name, structure in class_registry.items():
            init_func = _find_init_function(structure.node)
            if init_func is None:
                continue
            local_vars: dict[str, DimExpr] = {}
            for node in ast.walk(init_func):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute) and _is_self_attr(target):
                            resolved = _resolve_dim_expr(node.value, config=config, local_vars=local_vars, context=context)
                            if resolved is not None:
                                local_vars[target.attr] = resolved
                            spec = _parse_module_ctor(node.value, config=config, local_vars=local_vars, context=context)
                            if isinstance(spec, ModuleLinearSpec):
                                registry.linear[(class_name, target.attr)] = spec
                                registry.linear_by_attr[target.attr] = spec
                            elif isinstance(spec, ModuleEmbeddingSpec):
                                registry.embedding[(class_name, target.attr)] = spec
                                registry.embedding_by_attr[target.attr] = spec
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    target = node.target
                    if isinstance(target, ast.Attribute) and _is_self_attr(target):
                        resolved = _resolve_dim_expr(node.value, config=config, local_vars=local_vars, context=context)
                        if resolved is not None:
                            local_vars[target.attr] = resolved
                        spec = _parse_module_ctor(node.value, config=config, local_vars=local_vars, context=context)
                        if isinstance(spec, ModuleLinearSpec):
                            registry.linear[(class_name, target.attr)] = spec
                            registry.linear_by_attr[target.attr] = spec
                        elif isinstance(spec, ModuleEmbeddingSpec):
                            registry.embedding[(class_name, target.attr)] = spec
                            registry.embedding_by_attr[target.attr] = spec
        return registry


@dataclass
class OperatorRecord:
    """One compute step exported from a model graph."""

    name: str
    computation: str
    operation: str
    inputs: list[str]
    output: TensorSpec
    class_name: str | None = None
    node_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "computation": self.computation,
            "operation": self.operation,
            "inputs": self.inputs,
            "output": self.output.to_dict(),
        }
        if self.class_name:
            payload["class_name"] = self.class_name
        if self.node_id:
            payload["node_id"] = self.node_id
        return payload


class ShapeInferencer:
    """Infer symbolic/concrete tensor shapes for every node in a model graph."""

    def __init__(
        self,
        spec: ArchitectureSpec,
        *,
        context: ShapeContext | None = None,
        module_dims: ModuleDimRegistry | None = None,
    ) -> None:
        self.spec = spec
        self.context = context or ShapeContext.from_spec(spec)
        self.module_dims = module_dims or ModuleDimRegistry.from_registry(
            spec.class_registry,
            config=spec.raw_config or {},
            context=self.context,
        )
        self._tensor_names: dict[str, str] = {}
        self._tensor_specs: dict[str, TensorSpec] = {}

    def infer_model_graph(self, graph: ModelGraph, *, root: BlockNode | None = None) -> dict[str, TensorSpec]:
        """Infer output tensor specs for every node id in one model graph."""
        self._tensor_names = {}
        for node in graph.nodes:
            if node.metadata.get("synthetic") == "@input":
                self._tensor_names[node.id] = "input"
            else:
                self._tensor_names[node.id] = _output_tensor_name(node)
        self._tensor_specs = {}
        order = _topological_order(graph)
        node_by_id = {node.id: node for node in graph.nodes}

        for node_id in order:
            node = node_by_id[node_id]
            input_specs = self._gather_input_specs(graph, node_id)
            output = self._infer_node_output(node, input_specs, root=root)
            self._tensor_specs[node_id] = output

        for node in graph.nodes:
            if node.kind == NodeKind.SUBGRAPH:
                subgraph_key = node.metadata.get("subgraph_key")
                if subgraph_key and subgraph_key in graph.subgraphs:
                    self.infer_model_graph(graph.subgraphs[subgraph_key])

        return dict(self._tensor_specs)

    def infer_block_tree(self, root: BlockNode, *, title: str = "") -> dict[str, TensorSpec]:
        """Build a model graph from a block tree and infer all node shapes."""
        from visualizer.basic_ops import BasicOpFilter

        basic_ops = self.spec.basic_ops or BasicOpFilter.for_detailed()
        graph = build_model_graph(root, title=title or root.label, basic_ops=basic_ops)
        return self.infer_model_graph(graph, root=root)

    def export_operators(
        self,
        graph: ModelGraph,
        *,
        root: BlockNode | None = None,
        include_synthetic_inputs: bool = False,
    ) -> list[OperatorRecord]:
        """Export graph nodes as a flat operator list with inferred shapes."""
        specs = self.infer_model_graph(graph, root=root)
        operators: list[OperatorRecord] = []
        for node in _operational_node_order(graph):
            if node.kind == NodeKind.SUBGRAPH:
                subgraph_key = node.metadata.get("subgraph_key")
                if subgraph_key and subgraph_key in graph.subgraphs:
                    operators.extend(
                        self.export_operators(
                            graph.subgraphs[subgraph_key],
                            root=root,
                            include_synthetic_inputs=include_synthetic_inputs,
                        )
                    )
                continue

            if node.operation == OperationKind.SYNTHETIC and not include_synthetic_inputs:
                if node.metadata.get("synthetic") == "@input":
                    pass
                elif node.label not in {"×", "+", "Elementwise ×"} and node.metadata.get("synthetic") != "@combine":
                    continue

            output = specs.get(node.id)
            if output is None:
                continue
            inputs = [
                self._tensor_names[edge.source]
                for edge in graph.edges
                if edge.target == node.id and edge.source in self._tensor_names
            ]
            if node.metadata.get("synthetic") == "@input":
                operators.append(
                    OperatorRecord(
                        name="input",
                        computation="input",
                        operation="input",
                        inputs=[],
                        output=output,
                        class_name=node.label or "input",
                        node_id=node.id,
                    )
                )
                continue
            operators.append(
                OperatorRecord(
                    name=_operator_name(node),
                    computation=_low_level_computation(node),
                    operation=(node.operation.value if node.operation else "unknown"),
                    class_name=node.metadata.get("class_name"),
                    inputs=_dedupe_preserve(inputs),
                    output=output,
                    node_id=node.id,
                )
            )
        return operators

    def model_output_operator(self, *, input_tensor: str = "hidden_states") -> OperatorRecord | None:
        """Build the terminal output operator (typically LM head logits)."""
        vocab = self.context.dims.get(Symbol.VOCAB.value)
        if vocab is None:
            vocab = self.spec.vocab_size
        if vocab is None:
            return None
        hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
        return OperatorRecord(
            name="output",
            computation="output",
            operation="output",
            inputs=[input_tensor],
            output=TensorSpec(
                shape=(Symbol.BATCH.value, Symbol.SEQ.value, vocab),
                dtype=self.context.dtype,
            ),
            class_name="logits",
        )

    def export_architecture(
        self,
        *,
        include_model_output: bool = True,
    ) -> dict[str, Any]:
        """Export operators for every block-tree section in the loaded architecture."""
        from visualizer.basic_ops import BasicOpFilter

        block_trees = self.spec.export_block_trees or self.spec.detailed_block_trees
        basic_ops = self.spec.basic_ops or BasicOpFilter.for_detailed()
        sections: list[dict[str, Any]] = []
        lm_head_tensor: str | None = None

        for title, block_tree in block_trees:
            graph = build_model_graph(block_tree, title=title, basic_ops=basic_ops)
            operators = self.export_operators(graph, root=block_tree)
            for op in operators:
                if op.name == "lm_head":
                    lm_head_tensor = "lm_head"
            sections.append(
                {
                    "title": title,
                    "operators": [op.to_dict() for op in operators],
                }
            )

        if include_model_output:
            output_op = self.model_output_operator(
                input_tensor=lm_head_tensor or "hidden_states",
            )
            if output_op is not None:
                if sections:
                    sections[-1]["operators"].append(output_op.to_dict())
                else:
                    sections.append(
                        {
                            "title": "output",
                            "operators": [output_op.to_dict()],
                        }
                    )

        return {
            "name": self.spec.name,
            "model_type": self.spec.model_type,
            "checkpoint_source": self.spec.checkpoint_source,
            "code_sources": list(self.spec.code_sources),
            "dtype": self.context.dtype,
            "dimensions": {key: _serialize_dim(value) for key, value in self.context.dims.items()},
            "sections": sections,
        }

    def _gather_input_specs(self, graph: ModelGraph, node_id: str) -> list[TensorSpec]:
        specs: list[TensorSpec] = []
        for edge in graph.edges:
            if edge.target != node_id:
                continue
            source_spec = self._tensor_specs.get(edge.source)
            if source_spec is not None:
                specs.append(source_spec)
        return specs

    def _infer_node_output(
        self,
        node: ModelGraphNode,
        inputs: list[TensorSpec],
        *,
        root: BlockNode | None,
    ) -> TensorSpec:
        dtype = self.context.dtype
        synthetic = node.metadata.get("synthetic")
        class_name = (node.metadata.get("class_name") or node.label or "").strip()
        block_class = class_name or node.label

        if synthetic == "@input":
            if (node.label or "").lower() in {"input_ids", "input"}:
                return TensorSpec(shape=(Symbol.BATCH.value, Symbol.SEQ.value), dtype="int64")
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype)

        if node.label in {"×", "+", "Elementwise ×"} or synthetic == "@combine":
            if inputs:
                return inputs[0]
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype)

        linear_spec = self._lookup_linear_spec(node, root=root)
        if linear_spec is not None or _is_linear(node):
            in_shape = inputs[0].shape if inputs else _default_hidden_shape(self.context)
            out_features = (
                linear_spec.out_features
                if linear_spec is not None
                else _heuristic_linear_out_features(_node_attr_name(node), self.context)
            )
            if out_features is None and inputs:
                out_features = in_shape[-1]
            if out_features is None:
                out_features = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(shape=_replace_last_dim(in_shape, out_features), dtype=dtype)

        embedding_spec = self._lookup_embedding_spec(node, root=root)
        if embedding_spec is not None or _is_embedding(block_class, node):
            hidden = embedding_spec.embedding_dim if embedding_spec else self.context.dims.get(
                Symbol.HIDDEN.value, Symbol.HIDDEN.value
            )
            return TensorSpec(shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype)

        if _is_norm(block_class, node):
            if inputs:
                return inputs[0]
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype)

        if node.operation == OperationKind.GPU_KERNEL or class_name in {
            "AttentionOp",
            "KernelOp",
            "KernelOutput",
            "AttentionMerge",
        }:
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype)

        if _is_router(block_class, node):
            experts = self.context.dims.get(Symbol.EXPERTS.value, Symbol.EXPERTS.value)
            in_shape = inputs[0].shape if inputs else _default_hidden_shape(self.context)
            return TensorSpec(shape=_replace_last_dim(in_shape, experts), dtype=dtype)

        if node.operation == OperationKind.TORCH_FUNCTIONAL:
            if inputs:
                return inputs[0]
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype)

        if node.kind in {NodeKind.BLOCK, NodeKind.TOP_LEVEL} or node.operation == OperationKind.COMPOSITE:
            if inputs:
                return inputs[0]
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype)

        if inputs:
            return inputs[0]

        hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
        return TensorSpec(shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype)

    def _lookup_linear_spec(self, node: ModelGraphNode, *, root: BlockNode | None) -> ModuleLinearSpec | None:
        attr = _node_attr_name(node)
        class_name = node.metadata.get("class_name")
        if class_name and attr:
            spec = self.module_dims.linear.get((class_name, attr))
            if spec is not None:
                return spec
        if attr:
            return self.module_dims.linear_by_attr.get(attr)
        return None

    def _lookup_embedding_spec(self, node: ModelGraphNode, *, root: BlockNode | None) -> ModuleEmbeddingSpec | None:
        attr = _node_attr_name(node)
        class_name = node.metadata.get("class_name")
        if class_name and attr:
            spec = self.module_dims.embedding.get((class_name, attr))
            if spec is not None:
                return spec
        if attr:
            return self.module_dims.embedding_by_attr.get(attr)
        return None


def build_operator_export(
    spec: ArchitectureSpec,
    *,
    include_model_output: bool = True,
) -> dict[str, Any]:
    """Convenience wrapper: infer shapes and export operator lists for an architecture spec."""
    inferencer = ShapeInferencer(spec)
    return inferencer.export_architecture(include_model_output=include_model_output)


def save_operator_export(payload: dict[str, Any], path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def _serialize_dim(value: DimExpr) -> int | str:
    return value


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _default_hidden_shape(context: ShapeContext) -> tuple[DimExpr, ...]:
    hidden = context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
    return (Symbol.BATCH.value, Symbol.SEQ.value, hidden)


def _replace_last_dim(shape: tuple[DimExpr, ...], last: DimExpr) -> tuple[DimExpr, ...]:
    if not shape:
        return (Symbol.BATCH.value, Symbol.SEQ.value, last)
    return (*shape[:-1], last)


def _config_dtype(config: dict[str, Any]) -> str:
    raw = config.get("torch_dtype")
    if isinstance(raw, str):
        return raw.removeprefix("torch.").lower()
    return "float16"


def _is_self_attr(target: ast.Attribute) -> bool:
    return isinstance(target.value, ast.Name) and target.value.id == "self"


def _find_init_function(class_node: ast.ClassDef) -> ast.FunctionDef | None:
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            return item
    return None


def _call_class_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _resolve_dim_expr(
    node: ast.AST,
    *,
    config: dict[str, Any],
    local_vars: dict[str, DimExpr],
    context: ShapeContext,
) -> DimExpr | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return int(node.value)
    if isinstance(node, ast.Name):
        if node.id in local_vars:
            return local_vars[node.id]
        if node.id in context.dims:
            return context.dims[node.id]
        return None
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "config":
            value = config.get(node.attr)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
            if isinstance(value, str) and value.isdigit():
                return int(value)
            return node.attr
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return local_vars.get(node.attr)
    if isinstance(node, ast.BinOp):
        left = _resolve_dim_expr(node.left, config=config, local_vars=local_vars, context=context)
        right = _resolve_dim_expr(node.right, config=config, local_vars=local_vars, context=context)
        if isinstance(left, int) and isinstance(right, int):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Div):
                return left // right
        if isinstance(left, int) and isinstance(right, str):
            return f"{left}*{right}" if isinstance(node.op, ast.Mult) else None
        if isinstance(left, str) or isinstance(right, str):
            return f"{left}!{type(node.op).__name__}!{right}"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "int":
        if node.args:
            return _resolve_dim_expr(node.args[0], config=config, local_vars=local_vars, context=context)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _resolve_dim_expr(node.operand, config=config, local_vars=local_vars, context=context)
        if isinstance(inner, int):
            return -inner
    return None


def _parse_module_ctor(
    node: ast.AST,
    *,
    config: dict[str, Any],
    local_vars: dict[str, DimExpr],
    context: ShapeContext,
) -> ModuleLinearSpec | ModuleEmbeddingSpec | None:
    if not isinstance(node, ast.Call):
        return None
    class_name = _call_class_name(node) or ""
    args = list(node.args)
    if re.search(r"Linear$", class_name):
        in_features = _resolve_dim_expr(args[0], config=config, local_vars=local_vars, context=context) if len(args) >= 1 else None
        out_features = _resolve_dim_expr(args[1], config=config, local_vars=local_vars, context=context) if len(args) >= 2 else None
        for keyword in node.keywords:
            if keyword.arg == "in_features" and in_features is None:
                in_features = _resolve_dim_expr(keyword.value, config=config, local_vars=local_vars, context=context)
            if keyword.arg == "out_features" and out_features is None:
                out_features = _resolve_dim_expr(keyword.value, config=config, local_vars=local_vars, context=context)
        if in_features is not None and out_features is not None:
            return ModuleLinearSpec(in_features=in_features, out_features=out_features)
    if re.search(r"Embedding$", class_name):
        num_embeddings = _resolve_dim_expr(args[0], config=config, local_vars=local_vars, context=context) if len(args) >= 1 else None
        embedding_dim = _resolve_dim_expr(args[1], config=config, local_vars=local_vars, context=context) if len(args) >= 2 else None
        for keyword in node.keywords:
            if keyword.arg == "num_embeddings" and num_embeddings is None:
                num_embeddings = _resolve_dim_expr(keyword.value, config=config, local_vars=local_vars, context=context)
            if keyword.arg == "embedding_dim" and embedding_dim is None:
                embedding_dim = _resolve_dim_expr(keyword.value, config=config, local_vars=local_vars, context=context)
        if num_embeddings is not None and embedding_dim is not None:
            return ModuleEmbeddingSpec(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    return None


def _topological_order(graph: ModelGraph) -> list[str]:
    incoming = {node.id: 0 for node in graph.nodes}
    outgoing: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.source not in outgoing or edge.target not in incoming:
            continue
        outgoing[edge.source].append(edge.target)
        incoming[edge.target] += 1
    queue = [node_id for node_id, degree in incoming.items() if degree == 0]
    order: list[str] = []
    while queue:
        node_id = queue.pop(0)
        order.append(node_id)
        for target in outgoing.get(node_id, []):
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    if len(order) != len(incoming):
        remaining = [node_id for node_id in incoming if node_id not in order]
        order.extend(remaining)
    return order


def _operational_node_order(graph: ModelGraph) -> list[ModelGraphNode]:
    order = _topological_order(graph)
    node_by_id = {node.id: node for node in graph.nodes}
    return [node_by_id[node_id] for node_id in order if node_id in node_by_id]


def _output_tensor_name(node: ModelGraphNode) -> str:
    if node.metadata.get("synthetic") == "@input":
        return node.label
    if node.metadata.get("port_label"):
        return str(node.metadata["port_label"])
    attr = _node_attr_name(node)
    if attr:
        return attr
    return node.label or node.id


def _node_attr_name(node: ModelGraphNode) -> str | None:
    node_id = node.id
    if node_id.startswith("@"):
        return None
    parts = node_id.split(":")
    for part in reversed(parts):
        if part and not part.isdigit() and not re.match(r"^(fan\d+|merge|side|post|combine|node)$", part):
            return part
    return None


def _operator_name(node: ModelGraphNode) -> str:
    attr = _node_attr_name(node)
    if attr:
        return attr
    if node.metadata.get("synthetic") == "@input":
        return node.label
    if node.label in {"×", "+", "Elementwise ×"}:
        return node.label
    return node.label or node.id


def _low_level_computation(node: ModelGraphNode) -> str:
    class_name = node.metadata.get("class_name")
    if class_name:
        return str(class_name)
    if _is_linear(node):
        return "Linear"
    if node.operation == OperationKind.GPU_KERNEL:
        label = node.label or "gpu_kernel"
        return label
    if node.operation == OperationKind.TORCH_FUNCTIONAL:
        return node.label or "torch_functional"
    if node.metadata.get("synthetic") == "@input":
        return "input"
    if node.label in {"×", "+", "Elementwise ×"}:
        return "elementwise_mul" if "×" in node.label else "elementwise_add"
    return node.label or "unknown"


def _is_embedding(class_name: str, node: ModelGraphNode) -> bool:
    return bool(re.search(r"(?i)^Embedding$", class_name)) or node.label == "Embedding"


def _is_norm(class_name: str, node: ModelGraphNode) -> bool:
    return bool(re.search(r"(?i)(RMSNorm|LayerNorm|GroupNorm)$", class_name)) or node.label in {
        "RMSNorm",
        "LayerNorm",
    }


def _is_linear(node: ModelGraphNode) -> bool:
    if node.operation != OperationKind.NN_MODULE:
        return False
    class_name = node.metadata.get("class_name") or node.label or ""
    return bool(re.search(r"(?i)^Linear$", str(class_name)))


def _heuristic_linear_out_features(attr: str | None, context: ShapeContext) -> DimExpr | None:
    if not attr:
        return None
    lowered = attr.lower()
    if lowered in {"lm_head", "embed_out"}:
        return context.dims.get(Symbol.VOCAB.value)
    if lowered in {"gate_proj", "up_proj", "w1", "w3"}:
        return context.dims.get(Symbol.INTERMEDIATE.value)
    if lowered in {"down_proj", "w2", "o_proj", "q_proj", "k_proj", "v_proj"}:
        return context.dims.get(Symbol.HIDDEN.value)
    if lowered in {"router", "gate"} or lowered.endswith("_gate"):
        return context.dims.get(Symbol.EXPERTS.value)
    if "expert" in lowered and "proj" in lowered:
        return context.dims.get(Symbol.INTERMEDIATE.value)
    if lowered.endswith("_proj"):
        return context.dims.get(Symbol.HIDDEN.value)
    return None


def _is_router(class_name: str, node: ModelGraphNode) -> bool:
    return class_name == "RouterOp" or "router" in _operator_name(node).lower()


__all__ = [
    "DimExpr",
    "ModuleDimRegistry",
    "OperatorRecord",
    "ShapeContext",
    "ShapeInferencer",
    "Symbol",
    "TensorSpec",
    "build_operator_export",
    "save_operator_export",
]
