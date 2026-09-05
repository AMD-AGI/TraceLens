###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Symbolic tensor shape and dtype inference for model computation graphs."""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

from TraceLens.ModelUtils.extract import architecture_section_trees
from TraceLens.ModelUtils.ast_analyze import (
    analyze_source,
    is_forward_operation,
    operation_display_label,
)
from TraceLens.ModelUtils.kernel_pipeline import parse_kernel_import, _find_symbol_definition
from TraceLens.ModelUtils.model_graph import (
    ModelGraph,
    ModelGraphNode,
    NodeKind,
    OperationKind,
    build_model_graph,
)

if TYPE_CHECKING:
    from TraceLens.ModelUtils.ast_analyze import ClassStructure
    from TraceLens.ModelUtils.block_tree import BlockNode
    from TraceLens.ModelUtils.extract import ArchitectureSpec

_log = logging.getLogger(__name__)

DimExpr = int | str


class Symbol(str, Enum):
    """Common symbolic dimensions propagated through the graph."""

    BATCH = "B"
    SEQ = "S"
    HIDDEN = "H"
    VOCAB = "V"
    HEADS = "N"
    KV_HEADS = "K"
    HEAD_DIM = "D"
    INTERMEDIATE = "I"
    EXPERTS = "E"
    EXPERTS_PER_TOK = "TopK"


# Config attribute names modeling code reads for each symbolic dimension. Registered as
# fallbacks, so a key the checkpoint config actually defines always wins.
_SPEC_DIM_ALIASES: dict[Symbol, tuple[str, ...]] = {
    Symbol.HIDDEN: ("hidden_size", "hidden_dim", "d_model", "model_dim", "embed_dim"),
    Symbol.VOCAB: ("vocab_size",),
    Symbol.HEADS: ("num_attention_heads", "num_heads", "n_heads", "num_query_heads"),
    Symbol.KV_HEADS: ("num_key_value_heads", "num_kv_heads", "n_kv_heads"),
    Symbol.HEAD_DIM: ("head_dim", "attention_head_dim", "qk_head_dim"),
    Symbol.INTERMEDIATE: (
        "intermediate_size",
        "ffn_hidden_size",
        "ffn_dim",
        "moe_intermediate_size",
    ),
    Symbol.EXPERTS: (
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "num_routed_experts",
        "moe_num_experts",
    ),
    Symbol.EXPERTS_PER_TOK: (
        "num_experts_per_tok",
        "num_experts_per_token",
        "moe_top_k",
        "num_selected_experts",
    ),
}


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
class ModuleParameterSpec:
    """Shape of an ``nn.Parameter`` / raw tensor buffer declared in ``__init__``."""

    shape: tuple[DimExpr, ...]


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
        if spec.head_dim:
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

        # Sub-configs such as `linear_attn_config` are exposed on the config object as
        # flattened properties (`config.linear_head_dim`), so register those aliases too.
        for key, value in config.items():
            if not isinstance(value, dict):
                continue
            for alias, nested in _nested_dim_aliases(key, value):
                dims.setdefault(alias, nested)

        # Modeling code reads names that the checkpoint config spells differently
        # (`config.num_local_experts` against `n_routed_experts`, say).
        for symbol, aliases in _SPEC_DIM_ALIASES.items():
            resolved = dims.get(symbol.value)
            if not isinstance(resolved, int):
                continue
            for alias in aliases:
                dims.setdefault(alias, resolved)

        return cls(dims=dims, dtype=dtype)


@dataclass
class ModuleDimRegistry:
    """Linear and embedding constructor dimensions parsed from modeling AST."""

    linear: dict[tuple[str, str], ModuleLinearSpec] = field(default_factory=dict)
    linear_by_attr: dict[str, ModuleLinearSpec] = field(default_factory=dict)
    embedding: dict[tuple[str, str], ModuleEmbeddingSpec] = field(default_factory=dict)
    embedding_by_attr: dict[str, ModuleEmbeddingSpec] = field(default_factory=dict)
    parameter: dict[tuple[str, str], ModuleParameterSpec] = field(default_factory=dict)
    parameter_by_attr: dict[str, ModuleParameterSpec] = field(default_factory=dict)
    # Names like `weight` are declared by many modules with different shapes; guessing
    # across classes would be worse than having no shape at all.
    ambiguous_parameters: set[str] = field(default_factory=set)

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
                    targets = list(node.targets)
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    targets = [node.target]
                else:
                    continue
                for target in targets:
                    registry._record_assignment(
                        class_name,
                        target,
                        node.value,
                        config=config,
                        local_vars=local_vars,
                        context=context,
                    )
        return registry

    def _record_assignment(
        self,
        class_name: str,
        target: ast.AST,
        value: ast.AST,
        *,
        config: dict[str, Any],
        local_vars: dict[str, DimExpr],
        context: ShapeContext,
    ) -> None:
        if isinstance(target, ast.Name):
            # Plain locals feed later parameter shapes, e.g. `mix = (2 + hc) * hc`.
            resolved = _resolve_dim_expr(
                value, config=config, local_vars=local_vars, context=context
            )
            if resolved is not None:
                local_vars[target.id] = resolved
            return
        if not (isinstance(target, ast.Attribute) and _is_self_attr(target)):
            return

        resolved = _resolve_dim_expr(
            value, config=config, local_vars=local_vars, context=context
        )
        if resolved is not None:
            local_vars[target.attr] = resolved
        spec = _parse_module_ctor(
            value, config=config, local_vars=local_vars, context=context
        )
        if isinstance(spec, ModuleLinearSpec):
            self.linear[(class_name, target.attr)] = spec
            self.linear_by_attr[target.attr] = spec
        elif isinstance(spec, ModuleEmbeddingSpec):
            self.embedding[(class_name, target.attr)] = spec
            self.embedding_by_attr[target.attr] = spec
        elif isinstance(spec, ModuleParameterSpec):
            self.parameter[(class_name, target.attr)] = spec
            existing = self.parameter_by_attr.get(target.attr)
            if existing is not None and existing.shape != spec.shape:
                self.ambiguous_parameters.add(target.attr)
            self.parameter_by_attr[target.attr] = spec

    def lookup_parameter(
        self, attr: str | None, class_name: str | None
    ) -> ModuleParameterSpec | None:
        if not attr:
            return None
        if class_name:
            spec = self.parameter.get((class_name, attr))
            if spec is not None:
                return spec
        if attr in self.ambiguous_parameters:
            return None
        return self.parameter_by_attr.get(attr)


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


# Display labels (lowercased) whose op collapses one axis of its input.
_REDUCTION_LABELS = frozenset(
    {
        "sum",
        "mean",
        "product",
        "block max",
        "block min",
        "max",
        "min",
        "argmax",
        "argmin",
        "logsumexp",
        "norm",
        "variance",
        "std",
    }
)
# Display labels (lowercased) whose op keeps the shape of its widest operand.
_POINTWISE_LABELS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "floor divide",
        "power",
        "sigmoid",
        "softmax",
        "logsoftmax",
        "softplus",
        "tanh",
        "relu",
        "silu",
        "gelu",
        "erf",
        "exp",
        "log",
        "log1p",
        "sqrt",
        "reciprocal sqrt",
        "square",
        "abs",
        "negate",
        "reciprocal",
        "sign",
        "clamp",
        "nan to num",
        "maximum",
        "minimum",
        "where",
        "cumulative sum",
        "masked fill",
        "masked scatter",
        "scatter",
        "cosine",
        "sine",
        "index add",
        "roll",
        "flip",
        "lower triangle",
        "upper triangle",
        "zeros like",
        "ones like",
        "full like",
        "clone",
        "detach",
        "pad",
        "outer product",
        "polar",
        "view as complex",
        "view as real",
        "repeat interleave",
    }
)


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
        self._owner_classes: dict[int, dict[str, str]] = {}
        # Specs carrying the activation a block's forward receives.
        self._forward_input_specs: set[int] = set()
        # Guard against infinite recursion during forward introspection.
        self._introspecting: set[str] = set()
        # Meta-device traced shapes (module path -> symbolic shape).
        self._meta_shapes: dict[str, TensorSpec] = {}

    def load_meta_shapes(
        self,
        checkpoint: str | Path,
        *,
        seq_len: int = 128,
        batch_size: int = 1,
    ) -> bool:
        """Run a meta-device forward pass and store per-module shapes.

        Returns *True* when shapes were successfully captured.
        """
        from TraceLens.ModelUtils.meta_trace import trace_meta_shapes, symbolise_meta_shape

        raw = trace_meta_shapes(
            checkpoint,
            config=self.spec.raw_config,
            seq_len=seq_len,
            batch_size=batch_size,
        )
        if raw is None:
            return False
        for module_path, shape in raw.items():
            sym = symbolise_meta_shape(
                shape, batch_size=batch_size, seq_len=seq_len
            )
            self._meta_shapes[module_path] = TensorSpec(
                shape=sym, dtype=self.context.dtype
            )
        return bool(self._meta_shapes)

    def infer_model_graph(
        self, graph: ModelGraph, *, root: BlockNode | None = None
    ) -> dict[str, TensorSpec]:
        """Infer output tensor specs for every node id in one model graph."""
        self._tensor_names = {}
        for node in graph.nodes:
            if node.metadata.get("synthetic") == "@input":
                self._tensor_names[node.id] = "input"
            else:
                self._tensor_names[node.id] = _output_tensor_name(node)
        self._tensor_specs = {}
        self._forward_input_specs = set()
        order = _topological_order(graph)
        node_by_id = {node.id: node for node in graph.nodes}

        for node_id in order:
            node = node_by_id[node_id]
            input_specs = self._gather_input_specs(graph, node_id)
            output = self._infer_node_output(node, input_specs, root=root)
            self._tensor_specs[node_id] = output
            if node.metadata.get("synthetic") == "@input":
                self._forward_input_specs.add(id(output))

        if root is not None and "HyperConnection" in root.class_name:
            batch = Symbol.BATCH.value
            seq = Symbol.SEQ.value
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            streams = self.context.dims.get(
                "hc_mult", self.context.dims.get("text_hc_mult", "HC")
            )
            dtype = self.context.dtype
            slot_specs = {
                "post": TensorSpec((batch, seq, streams), "float32"),
                "comb": TensorSpec((batch, seq, streams, streams), "float32"),
                "collapsed": TensorSpec((batch, seq, hidden), dtype),
            }
            for node in graph.nodes:
                synthetic = node.metadata.get("synthetic")
                if synthetic == "@input":
                    self._tensor_specs[node.id] = TensorSpec(
                        (batch, seq, streams, hidden), dtype
                    )
                if synthetic == "@loop_carried":
                    self._tensor_specs[node.id] = slot_specs["comb"]
                attr_name = node.metadata.get("attr_name")
                for slot, producer in root.forward_return_slots.items():
                    if attr_name == producer and slot in slot_specs:
                        self._tensor_specs[node.id] = slot_specs[slot]
                if synthetic == "@output":
                    self._tensor_specs[node.id] = slot_specs.get(
                        root.primary_return_slot or "", slot_specs["collapsed"]
                    )

        for node in graph.nodes:
            if node.kind == NodeKind.SUBGRAPH:
                subgraph_key = node.metadata.get("subgraph_key")
                if subgraph_key and subgraph_key in graph.subgraphs:
                    self.infer_model_graph(graph.subgraphs[subgraph_key])

        return dict(self._tensor_specs)

    def infer_block_tree(
        self, root: BlockNode, *, title: str = ""
    ) -> dict[str, TensorSpec]:
        """Build a model graph from a block tree and infer all node shapes."""
        from TraceLens.ModelUtils.basic_ops import BasicOpFilter

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

            if (
                node.operation == OperationKind.SYNTHETIC
                and not include_synthetic_inputs
            ):
                if node.metadata.get("synthetic") == "@input":
                    pass
                elif (
                    node.label not in {"×", "+", "Elementwise ×", "Multiply", "Add"}
                    and node.metadata.get("synthetic") != "@combine"
                ):
                    continue

            output = specs.get(node.id)
            if output is None:
                continue
            inputs = [
                self._tensor_names[edge.source]
                for edge in graph.edges
                if edge.target == node.id and edge.source in self._tensor_names
            ]
            inputs.extend(
                str(item) for item in node.metadata.get("external_inputs", [])
            )
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
                    operation=_export_operation_kind(node),
                    class_name=node.metadata.get("class_name"),
                    inputs=_dedupe_preserve(inputs),
                    output=output,
                    node_id=node.id,
                )
            )
        return operators

    def model_output_operator(
        self, *, input_tensor: str = "hidden_states"
    ) -> OperatorRecord | None:
        """Build the terminal output operator (typically LM head logits)."""
        vocab = self.context.dims.get(Symbol.VOCAB.value)
        if vocab is None:
            vocab = self.spec.vocab_size
        if vocab is None:
            return None
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
        from TraceLens.ModelUtils.basic_ops import BasicOpFilter

        block_trees = architecture_section_trees(self.spec)
        basic_ops = self.spec.basic_ops or BasicOpFilter.for_detailed()
        sections: list[dict[str, Any]] = []
        lm_head_tensor: str | None = None

        from TraceLens.ModelUtils.block_tree import subgraph_warrants_json_export

        seen_shape_signatures: set[tuple[Any, ...]] = set()
        for title, block_tree in block_trees:
            graph = build_model_graph(block_tree, title=title, basic_ops=basic_ops)
            operators = self.export_operators(graph, root=block_tree)
            for op in operators:
                if op.name == "lm_head":
                    lm_head_tensor = "lm_head"
            if not subgraph_warrants_json_export(block_tree, basic_ops=basic_ops):
                continue
            signature = subgraph_boundary_signature(
                operators,
                class_name=block_tree.class_name,
            )
            if signature is not None:
                if signature in seen_shape_signatures:
                    continue
                seen_shape_signatures.add(signature)
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
            "dimensions": {
                key: _serialize_dim(value) for key, value in self.context.dims.items()
            },
            "sections": sections,
        }

    def _elementwise_operand(self, inputs: list[TensorSpec]) -> TensorSpec:
        """Operand an elementwise op takes its shape from.

        Broadcasting against the block's own forward input widens the result, which is
        how a stream-collapse multiply recovers the activation width from the mixing
        weights feeding it. Every other operand keeps chain order, since a step's width
        is often inherited rather than known.
        """
        widest = max(inputs, key=_broadcast_rank)
        if id(widest) in self._forward_input_specs and _broadcast_rank(
            widest
        ) > _broadcast_rank(inputs[0]):
            return widest
        return inputs[0]

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
                return TensorSpec(
                    shape=(Symbol.BATCH.value, Symbol.SEQ.value), dtype="int64"
                )
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(
                shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype
            )

        if synthetic in {"@output", "@loop_carried"}:
            if inputs:
                return inputs[-1]
            return TensorSpec(
                shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype
            )

        if synthetic == "@tensor":
            label = (node.metadata.get("port_label") or node.label or "").lower()
            experts = self.context.dims.get(Symbol.EXPERTS.value, Symbol.EXPERTS.value)
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            parameter = self._lookup_parameter_spec(node, root=root, names=[label])
            if parameter is not None:
                return TensorSpec(shape=parameter.shape, dtype=dtype)
            if "weight" in label:
                return TensorSpec(shape=(experts, hidden), dtype="float32")
            if "bias" in label:
                return TensorSpec(shape=(experts,), dtype=dtype)
            return TensorSpec(shape=(), dtype=dtype)

        if (
            node.label in {"×", "+", "Elementwise ×", "Multiply", "Add"}
            or synthetic == "@combine"
        ):
            if inputs:
                if node.label in {"+", "Add"}:
                    return max(inputs, key=_broadcast_rank)
                return self._elementwise_operand(inputs)
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(
                shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype
            )

        # Catch-all for any remaining synthetic wiring nodes (kernel ports,
        # hidden_states, etc.) — silent passthrough, no warning.
        if synthetic is not None and synthetic.startswith("@"):
            if inputs:
                return inputs[0]
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(
                shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype
            )

        # Meta-device ground-truth shapes (highest priority for real modules).
        meta_spec = self._lookup_meta_shape(node)
        if meta_spec is not None:
            return meta_spec

        operation_label = (node.label or class_name).strip().lower()
        details = [str(item) for item in node.metadata.get("details", [])]
        external_inputs = [
            str(item).lower() for item in node.metadata.get("external_inputs", [])
        ]

        def external_spec() -> TensorSpec | None:
            experts = self.context.dims.get(Symbol.EXPERTS.value, Symbol.EXPERTS.value)
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            parameter = self._lookup_parameter_spec(
                node, root=root, names=external_inputs
            )
            if parameter is not None:
                return TensorSpec(parameter.shape, dtype)
            if any("weight" in item for item in external_inputs):
                return TensorSpec((experts, hidden), "float32")
            if any("bias" in item for item in external_inputs):
                return TensorSpec((experts,), dtype)
            return None

        if operation_label in {"view", "reshape", "flatten"}:
            source = (
                inputs[0]
                if inputs
                else external_spec()
                or TensorSpec(_default_hidden_shape(self.context), dtype)
            )
            shape_detail = next(
                (
                    item.split(":", 1)[1].strip()
                    for item in details
                    if item.startswith("shape:")
                ),
                "",
            )
            # Try structured resolution first (handles starred prefixes and
            # symbolic dimension names from the model config).
            resolved = _resolve_view_shape(
                shape_detail, source, self.context.dims
            )
            if resolved is not None:
                return TensorSpec(shape=resolved, dtype=source.dtype)
            if "-1" in shape_detail:
                flattened = f"{Symbol.BATCH.value}*{Symbol.SEQ.value}"
                return TensorSpec(
                    shape=(flattened, source.shape[-1]), dtype=source.dtype
                )
            return source

        if operation_label == "unsqueeze":
            source = inputs[0] if inputs else external_spec() or TensorSpec((), dtype)
            return TensorSpec(shape=(1, *source.shape), dtype=source.dtype)

        if operation_label in {"split", "chunk", "unbind"}:
            source = (
                inputs[0]
                if inputs
                else TensorSpec(_default_hidden_shape(self.context), dtype)
            )
            dim_str = _detail_value(details, "dim")
            dim = _int_dim(dim_str) if dim_str is not None else -1
            if dim is None:
                dim = -1
            split_size = _detail_value(details, "split_size")
            resolved_dim = dim % len(source.shape) if source.shape else 0
            dim_val = source.shape[resolved_dim] if source.shape else None
            if isinstance(dim_val, int):
                if split_size is not None:
                    if operation_label == "chunk":
                        # For chunk, the recorded value is the number of chunks.
                        try:
                            n = int(split_size)
                            if n > 0:
                                return TensorSpec(
                                    shape=_replace_dim(
                                        source.shape, resolved_dim, dim_val // n
                                    ),
                                    dtype=source.dtype,
                                )
                        except (ValueError, TypeError):
                            pass
                    else:
                        sizes = _parse_split_sizes(split_size, self.context.dims)
                        if sizes:
                            out_size = sizes[0]
                            return TensorSpec(
                                shape=_replace_dim(
                                    source.shape, resolved_dim, out_size
                                ),
                                dtype=source.dtype,
                            )
                # Fallback: look at external_inputs for the split size name
                for ext in external_inputs:
                    resolved = _resolve_dim_name(ext, self.context.dims)
                    if resolved is not None and isinstance(resolved, int):
                        return TensorSpec(
                            shape=_replace_dim(source.shape, resolved_dim, resolved),
                            dtype=source.dtype,
                        )
            return source

        if operation_label in {"concat", "stack"}:
            if not inputs:
                return TensorSpec(_default_hidden_shape(self.context), dtype)
            if operation_label == "stack":
                base = inputs[0]
                return TensorSpec(shape=(len(inputs), *base.shape), dtype=base.dtype)
            # Concat: sum the last dim when all inputs have the same rank
            dim_str = _detail_value(details, "dim")
            dim = _int_dim(dim_str) if dim_str is not None else -1
            if dim is None:
                dim = -1
            base = max(inputs, key=_broadcast_rank)
            resolved_dim = dim % len(base.shape) if base.shape else 0
            concat_sizes = []
            for inp in inputs:
                if inp.shape and len(inp.shape) > resolved_dim:
                    concat_sizes.append(inp.shape[resolved_dim])
            if concat_sizes and all(isinstance(s, int) for s in concat_sizes):
                total = sum(concat_sizes)
                return TensorSpec(
                    shape=_replace_dim(base.shape, resolved_dim, total),
                    dtype=base.dtype,
                )
            return base

        if operation_label in {"transpose", "permute"}:
            source = (
                inputs[0]
                if inputs
                else TensorSpec(_default_hidden_shape(self.context), dtype)
            )
            if operation_label == "transpose" and len(source.shape) >= 2:
                dim0_str = _detail_value(details, "dim0")
                dim1_str = _detail_value(details, "dim1")
                dim0 = _int_dim(dim0_str) if dim0_str is not None else -2
                dim1 = _int_dim(dim1_str) if dim1_str is not None else -1
                if dim0 is not None and dim1 is not None:
                    n = len(source.shape)
                    dim0 = dim0 % n
                    dim1 = dim1 % n
                    shape = list(source.shape)
                    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
                    return TensorSpec(shape=tuple(shape), dtype=source.dtype)
            return source

        if operation_label in {"matmul", "batchmatmul", "mm", "bmm"}:
            if len(inputs) >= 2:
                a, b = inputs[0], inputs[1]
                if a.shape and b.shape:
                    out_shape = (*a.shape[:-1], b.shape[-1])
                    return TensorSpec(shape=out_shape, dtype=a.dtype)
            if inputs:
                return inputs[0]
            return TensorSpec(_default_hidden_shape(self.context), dtype)

        if operation_label == "einsum":
            equation = _detail_value(details, "equation")
            if equation and "->" in equation and inputs:
                out_shape = _infer_einsum_shape(equation, inputs)
                if out_shape is not None:
                    return TensorSpec(shape=out_shape, dtype=inputs[0].dtype)
            if inputs:
                return inputs[0]
            return TensorSpec(_default_hidden_shape(self.context), dtype)

        if operation_label == "nonzero":
            source = (
                inputs[0]
                if inputs
                else TensorSpec(_default_hidden_shape(self.context), dtype)
            )
            ndim = len(source.shape) if source.shape else 1
            return TensorSpec(shape=("nnz", ndim), dtype="int64")

        if operation_label == "one hot":
            source = (
                inputs[0]
                if inputs
                else TensorSpec(_default_hidden_shape(self.context), dtype)
            )
            num_classes = self.context.dims.get(
                Symbol.EXPERTS.value, Symbol.EXPERTS.value
            )
            return TensorSpec(shape=(*source.shape, num_classes), dtype="int64")

        if operation_label in {
            "causal conv1d",
            "causal conv1d update",
        }:
            if inputs:
                return inputs[0]
            return TensorSpec(_default_hidden_shape(self.context), dtype)

        if operation_label in {"cast", "contiguous", "squeeze", "expand"}:
            source = (
                inputs[0]
                if inputs
                else external_spec()
                or TensorSpec(_default_hidden_shape(self.context), dtype)
            )
            cast_dtype = source.dtype
            dtype_detail = next(
                (
                    item.split(":", 1)[1].strip()
                    for item in details
                    if item.startswith("dtype:")
                ),
                "",
            )
            if "float32" in dtype_detail:
                cast_dtype = "float32"
            return TensorSpec(shape=source.shape, dtype=cast_dtype)

        if operation_label == "topk":
            source = (
                inputs[0]
                if inputs
                else TensorSpec(_default_hidden_shape(self.context), dtype)
            )
            top_k = self.context.dims.get(
                Symbol.EXPERTS_PER_TOK.value, Symbol.EXPERTS_PER_TOK.value
            )
            return TensorSpec(
                shape=_replace_last_dim(source.shape, top_k), dtype="int64"
            )

        if operation_label == "gather":
            source = next(
                (item for item in inputs if item.dtype != "int64"),
                inputs[0] if inputs else None,
            )
            index = next((item for item in inputs if item.dtype == "int64"), None)
            if source is None:
                source = TensorSpec(_default_hidden_shape(self.context), dtype)
            shape = (
                index.shape
                if index is not None
                else _replace_last_dim(
                    source.shape,
                    self.context.dims.get(
                        Symbol.EXPERTS_PER_TOK.value, Symbol.EXPERTS_PER_TOK.value
                    ),
                )
            )
            return TensorSpec(shape=shape, dtype=source.dtype)

        if operation_label in _REDUCTION_LABELS:
            source = (
                inputs[0]
                if inputs
                else TensorSpec(_default_hidden_shape(self.context), dtype)
            )
            index_reduction = operation_label in {"argmax", "argmin"}
            reduced_dim = _detail_value(details, "dim")
            if reduced_dim is not None and reduced_dim != "-1":
                # Reductions over stream or head axes the symbolic (B, S, H) view omits.
                if not index_reduction:
                    return source
                return TensorSpec(shape=source.shape, dtype="int64")
            return TensorSpec(
                shape=_replace_last_dim(source.shape, 1),
                dtype="int64" if index_reduction else source.dtype,
            )

        if operation_label in _POINTWISE_LABELS:
            if inputs:
                source = max(inputs, key=_broadcast_rank)
                return TensorSpec(shape=source.shape, dtype=source.dtype)
            return TensorSpec(shape=_default_hidden_shape(self.context), dtype=dtype)

        linear_spec = self._lookup_linear_spec(node, root=root)
        if linear_spec is not None or _is_linear(node):
            hidden_dim = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            activation_input = next(
                (
                    item
                    for item in inputs
                    if item.shape
                    and item.shape[-1] == hidden_dim
                    and not (
                        len(item.shape) == 2
                        and item.shape[0] == self.context.dims.get(Symbol.EXPERTS.value)
                    )
                ),
                inputs[-1] if inputs else None,
            )
            in_shape = (
                activation_input.shape
                if activation_input is not None
                else _default_hidden_shape(self.context)
            )
            out_features = (
                linear_spec.out_features
                if linear_spec is not None
                else _heuristic_linear_out_features(_node_attr_name(node), self.context)
            )
            if (
                out_features is None
                and root is not None
                and re.search(r"(?i)(MoE)?Gate|Router", root.class_name)
            ):
                out_features = self.context.dims.get(
                    Symbol.EXPERTS.value, Symbol.EXPERTS.value
                )
            if out_features is None and operation_label == "linear":
                # `F.linear(x, w)` reads out features from w's row axis; stacked expert
                # weights (E, out, in) are indexed per expert before the call.
                parameter = self._lookup_parameter_spec(
                    node, root=root, names=external_inputs
                )
                if parameter is not None and len(parameter.shape) >= 2:
                    out_features = parameter.shape[-2]
            if out_features is None and inputs:
                out_features = in_shape[-1]
            if out_features is None:
                out_features = self.context.dims.get(
                    Symbol.HIDDEN.value, Symbol.HIDDEN.value
                )
            output_dtype = (
                "float32" if any("float32" in item for item in details) else dtype
            )
            return TensorSpec(
                shape=_replace_last_dim(in_shape, out_features), dtype=output_dtype
            )

        embedding_spec = self._lookup_embedding_spec(node, root=root)
        if embedding_spec is not None or _is_embedding(block_class, node):
            hidden = (
                embedding_spec.embedding_dim
                if embedding_spec
                else self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            )
            return TensorSpec(
                shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype
            )

        if _is_norm(block_class, node):
            if inputs:
                return inputs[0]
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(
                shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype
            )

        if node.operation == OperationKind.GPU_KERNEL or class_name in {
            "AttentionOp",
            "KernelOp",
            "KernelOutput",
            "AttentionMerge",
        }:
            introspected = self._introspect_forward_shape(
                node, inputs, root=root
            )
            if introspected is not None:
                return introspected
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(
                shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype
            )

        if _is_router(block_class, node):
            experts = self.context.dims.get(Symbol.EXPERTS.value, Symbol.EXPERTS.value)
            in_shape = (
                inputs[0].shape if inputs else _default_hidden_shape(self.context)
            )
            return TensorSpec(shape=_replace_last_dim(in_shape, experts), dtype=dtype)

        if node.operation == OperationKind.TORCH_FUNCTIONAL:
            introspected = self._introspect_forward_shape(
                node, inputs, root=root
            )
            if introspected is not None:
                return introspected
            _log.warning(
                "No shape inference rule for %s (label=%r, class=%r); "
                "passing through input shape",
                node.id,
                node.label,
                node.metadata.get("class_name"),
            )
            if inputs:
                return inputs[0]
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(
                shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype
            )

        if (
            node.kind in {NodeKind.BLOCK, NodeKind.TOP_LEVEL}
            or node.operation == OperationKind.COMPOSITE
        ):
            if inputs:
                return inputs[0]
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(
                shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype
            )

        introspected = self._introspect_forward_shape(
            node, inputs, root=root
        )
        if introspected is not None:
            return introspected

        if inputs:
            _log.warning(
                "No shape inference rule for %s (label=%r, class=%r, kind=%s); "
                "passing through input shape",
                node.id,
                node.label,
                node.metadata.get("class_name"),
                node.operation,
            )
            return inputs[0]

        _log.warning(
            "No shape inference rule for %s (label=%r, class=%r, kind=%s); "
            "defaulting to (B, S, H)",
            node.id,
            node.label,
            node.metadata.get("class_name"),
            node.operation,
        )
        hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
        return TensorSpec(
            shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype
        )

    # ------------------------------------------------------------------
    # Meta-device shape lookup
    # ------------------------------------------------------------------

    def _lookup_meta_shape(
        self, node: ModelGraphNode
    ) -> TensorSpec | None:
        """Return meta-traced shape if available for this node's module."""
        if not self._meta_shapes:
            return None
        attr = node.metadata.get("attr_name") or ""
        if attr and attr in self._meta_shapes:
            return self._meta_shapes[attr]
        # Try matching via class_name + layer index patterns.
        for part in node.id.split(":"):
            if part in self._meta_shapes:
                return self._meta_shapes[part]
        return None

    # ------------------------------------------------------------------
    # Forward introspection: simulate shape flow through forward_operations
    # ------------------------------------------------------------------

    def _introspect_forward_shape(
        self,
        node: ModelGraphNode,
        inputs: list[TensorSpec],
        *,
        root: BlockNode | None,
    ) -> TensorSpec | None:
        """Try to infer the output shape by simulating the module's ``forward()``."""
        # class_name may be absent when it equals the label (optimised away
        # by _minimal_metadata).  Try several sources.
        candidates = [
            node.metadata.get("class_name"),
            node.metadata.get("attr_name"),
        ]
        # Recover the original attr from the node id (``seq:9:get_pooled_states``).
        raw_id = node.id.rsplit("/", 1)[-1]
        id_parts = raw_id.split(":")
        if len(id_parts) >= 3:
            candidates.append(id_parts[2])
        # Also try the label with underscores restored.
        if node.label:
            candidates.append(node.label.replace(" ", "_"))

        class_name = ""
        for c in candidates:
            if c and c not in self._introspecting:
                class_name = c
                break
        if not class_name:
            return None

        structure = self.spec.class_registry.get(class_name)
        if structure is not None:
            return self._simulate_forward_ops(
                structure, inputs, root=root, guard_name=class_name
            )

        # class_name may be a method name (e.g. "get_pooled_states") rather
        # than a class.  Search the registry for a class that owns this method
        # and parse the method body for shape-bearing operations.
        method_result = self._introspect_method_shape(
            class_name, inputs, root=root
        )
        if method_result is not None:
            return method_result

        # For GPU kernels, try resolving the kernel source.
        details = [str(d) for d in node.metadata.get("details", [])]
        kernel_import = parse_kernel_import(details)
        if kernel_import is not None:
            return self._introspect_kernel_source(
                kernel_import, inputs, root=root
            )
        return None

    def _introspect_method_shape(
        self,
        method_name: str,
        inputs: list[TensorSpec],
        *,
        root: BlockNode | None,
    ) -> TensorSpec | None:
        """Find *method_name* on a registry class and parse its body for shapes."""
        from TraceLens.ModelUtils.ast_analyze import (
            _ForwardOperationExtractor,
            _extract_forward_return_metadata,
        )
        guard = f"method:{method_name}"
        if guard in self._introspecting:
            return None
        for _cls_name, structure in self.spec.class_registry.items():
            method_func = _find_method(structure.node, method_name)
            if method_func is None:
                continue
            # Determine the first parameter after self.
            input_name = (
                method_func.args.args[1].arg
                if len(method_func.args.args) >= 2
                else None
            )
            extractor = _ForwardOperationExtractor(
                self_values=structure.init_assignments,
                all_tensor_ops=True,
            )
            if input_name:
                extractor.var_producer[input_name] = input_name
            extractor.statements(method_func.body)
            if not extractor.operations:
                return None
            fwd_ops = {op.attr_name: op for op in extractor.operations}
            _slots, _order, primary = _extract_forward_return_metadata(
                method_func, extractor.var_producer
            )

            class _MethodProxy:
                forward_operations = fwd_ops
                forward_input_name = input_name
                primary_return_slot = (
                    extractor.var_producer.get(primary) if primary else None
                )

            return self._simulate_forward_ops(
                _MethodProxy(),  # type: ignore[arg-type]
                inputs,
                root=root,
                guard_name=guard,
            )
        return None

    def _simulate_forward_ops(
        self,
        structure: "ClassStructure",
        inputs: list[TensorSpec],
        *,
        root: BlockNode | None,
        guard_name: str,
    ) -> TensorSpec | None:
        """Walk ``forward_operations`` and propagate shapes op-by-op."""
        fwd_ops = structure.forward_operations
        if not fwd_ops:
            return None

        self._introspecting.add(guard_name)
        try:
            dtype = self.context.dtype
            input_spec = inputs[0] if inputs else TensorSpec(
                _default_hidden_shape(self.context), dtype
            )
            op_shapes: dict[str, TensorSpec] = {}
            input_name = structure.forward_input_name

            for op_id, op in fwd_ops.items():
                pred_specs: list[TensorSpec] = []
                for pred in op.predecessors:
                    if pred in op_shapes:
                        pred_specs.append(op_shapes[pred])
                    elif pred == input_name:
                        pred_specs.append(input_spec)
                if not pred_specs:
                    pred_specs = [input_spec]

                temp_node = ModelGraphNode(
                    id=op_id,
                    kind=NodeKind.LEAF,
                    label=op.label,
                    operation=OperationKind.TORCH_FUNCTIONAL,
                    metadata={
                        "class_name": op.class_name,
                        "details": list(op.details),
                        "external_inputs": list(op.external_inputs),
                    },
                )
                op_shapes[op_id] = self._infer_node_output(
                    temp_node, pred_specs, root=root
                )

            # Return the primary return producer's shape, or the last op.
            ret_slot = structure.primary_return_slot
            if ret_slot and ret_slot in op_shapes:
                return op_shapes[ret_slot]
            if op_shapes:
                return list(op_shapes.values())[-1]
        finally:
            self._introspecting.discard(guard_name)
        return None

    def _introspect_kernel_source(
        self,
        kernel_import: tuple[str, str],
        inputs: list[TensorSpec],
        *,
        root: BlockNode | None,
    ) -> TensorSpec | None:
        """Resolve a kernel's source, parse it, and simulate its forward ops."""
        module, symbol = kernel_import
        guard = f"{module}#{symbol}"
        if guard in self._introspecting:
            return None

        # Try the kernel's own class first.
        definition = _find_symbol_definition(module, symbol)
        if definition is not None:
            source, qualname, owning_module = definition
            analysis = analyze_source(source, filename=owning_module)
            # Look for the kernel class in the analysis registry.
            for cls_name, structure in analysis.class_registry.items():
                if qualname.startswith(cls_name) and structure.forward_operations:
                    result = self._simulate_forward_ops(
                        structure, inputs, root=root, guard_name=guard
                    )
                    if result is not None:
                        return result

        # Level 2: look for an eager/pure-torch fallback in the same module.
        eager_candidates = [
            f"eager_{symbol.lower()}",
            f"eager_attention_forward",
            symbol.replace("flash_", "eager_").replace("sdpa_", "eager_"),
        ]
        seen: set[str] = set()
        for candidate in eager_candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            eager_def = _find_symbol_definition(module, candidate)
            if eager_def is None:
                continue
            source, qualname, owning_module = eager_def
            analysis = analyze_source(
                source, filename=owning_module, all_tensor_ops=True
            )
            for cls_name, structure in analysis.class_registry.items():
                if structure.forward_operations:
                    result = self._simulate_forward_ops(
                        structure, inputs, root=root, guard_name=guard
                    )
                    if result is not None:
                        return result
            # Also check if it's a standalone function (not a class).
            # analyze_source treats top-level forward-like functions as class entries
            # only if they're in a class; for standalone functions we won't find them
            # in class_registry. This is a future enhancement.

        return None

    def _lookup_linear_spec(
        self, node: ModelGraphNode, *, root: BlockNode | None
    ) -> ModuleLinearSpec | None:
        attr = _node_attr_name(node)
        class_name = node.metadata.get("class_name")
        if class_name and attr:
            spec = self.module_dims.linear.get((class_name, attr))
            if spec is not None:
                return spec
        if attr:
            return self.module_dims.linear_by_attr.get(attr)
        return None

    def _lookup_parameter_spec(
        self,
        node: ModelGraphNode,
        *,
        root: BlockNode | None,
        names: Sequence[str],
    ) -> ModuleParameterSpec | None:
        candidates = [
            node.metadata.get("class_name"),
            self._owner_class_name(node, root),
            root.class_name if root is not None else None,
        ]
        for name in names:
            attr = str(name).split(".")[-1].strip()
            for class_name in candidates:
                spec = self.module_dims.lookup_parameter(attr, class_name)
                if spec is not None:
                    return spec
        return None

    def _owner_class_name(
        self, node: ModelGraphNode, root: BlockNode | None
    ) -> str | None:
        """Class of the submodule a functional op lives in, read off the node id path.

        Node ids keep the module path (``sideproducer:0:gate:@op_..._linear:0``), so the
        trailing module segment says which class declared the parameters the op reads.
        """
        if root is None:
            return None
        classes = self._owner_classes.get(id(root))
        if classes is None:
            classes = _descendant_classes(root)
            self._owner_classes[id(root)] = classes
        for segment in reversed(re.split(r"[:/]", node.id)):
            if segment.startswith("@"):
                # Operation segments name the op itself, not the module that owns it.
                continue
            owner = classes.get(segment)
            if owner:
                return owner
        return None

    def _lookup_embedding_spec(
        self, node: ModelGraphNode, *, root: BlockNode | None
    ) -> ModuleEmbeddingSpec | None:
        attr = _node_attr_name(node)
        class_name = node.metadata.get("class_name")
        if class_name and attr:
            spec = self.module_dims.embedding.get((class_name, attr))
            if spec is not None:
                return spec
        if attr:
            return self.module_dims.embedding_by_attr.get(attr)
        return None


def subgraph_boundary_signature(
    operators: list[OperatorRecord],
    *,
    class_name: str | None = None,
) -> tuple[Any, ...] | None:
    """Hashable input/output boundary signature for deduplicating exported subgraphs."""
    input_ops = [op for op in operators if op.operation == "input"]
    compute_ops = [op for op in operators if op.operation not in {"input", "output"}]
    if not compute_ops:
        return None
    identity = class_name or compute_ops[0].class_name or compute_ops[0].computation
    if input_ops:
        in_spec = input_ops[0].output
        return (
            identity,
            tuple(in_spec.shape),
            in_spec.dtype,
            tuple(compute_ops[-1].output.shape),
            compute_ops[-1].output.dtype,
        )
    return (
        identity,
        tuple(compute_ops[0].output.shape),
        compute_ops[0].output.dtype,
        tuple(compute_ops[-1].output.shape),
        compute_ops[-1].output.dtype,
        "no_input",
    )


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


def _resolve_dim_name(name: str, dims: dict[str, DimExpr]) -> DimExpr | None:
    """Resolve a symbolic dimension name against config dims.

    Tries direct lookup first, then ``self.xxx`` stripping, and finally
    common suffixed variants (``_mult``, ``_size``, ``_dim``, ``_count``).
    """
    # Direct hit
    val = dims.get(name)
    if val is not None:
        return val
    # ``self.xxx`` → ``xxx``
    bare = name
    if bare.startswith("self."):
        bare = bare[len("self."):]
        val = dims.get(bare)
        if val is not None:
            return val
    # Try common config suffixes: ``hc`` → ``hc_mult``
    for suffix in ("_mult", "_size", "_dim", "_count"):
        val = dims.get(bare + suffix)
        if val is not None:
            return val
    return None


def _resolve_view_shape(
    detail: str,
    source: TensorSpec,
    dims: dict[str, DimExpr],
) -> tuple[DimExpr, ...] | None:
    """Try to resolve symbolic view/reshape arguments into a concrete shape.

    Handles patterns like ``*x.shape[:-1], hc, hc`` by keeping leading dims
    from the source shape and resolving trailing symbolic names via *dims*.
    """
    if not detail:
        return None
    parts = [p.strip() for p in detail.split(",") if p.strip()]
    if not parts:
        return None

    leading: tuple[DimExpr, ...] = ()
    trailing_start = 0

    # Detect starred prefix like ``*foo.shape[:-1]`` or ``*foo.shape[:-N]``.
    first = parts[0]
    if first.startswith("*") and ".shape" in first:
        import re

        m = re.search(r"\.shape\[:\s*(-?\d+)\]", first)
        if m:
            cut = int(m.group(1))
            leading = source.shape[:cut] if cut < 0 else source.shape[:cut]
        else:
            leading = source.shape
        trailing_start = 1

    resolved: list[DimExpr] = list(leading)
    for part in parts[trailing_start:]:
        # Try literal int
        try:
            resolved.append(int(part))
            continue
        except ValueError:
            pass
        # Try resolving via config dims (with suffix heuristics)
        val = _resolve_dim_name(part, dims)
        if val is not None:
            resolved.append(val)
            continue
        # Cannot resolve — give up
        return None

    return tuple(resolved) if resolved else None


def _default_hidden_shape(context: ShapeContext) -> tuple[DimExpr, ...]:
    hidden = context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
    return (Symbol.BATCH.value, Symbol.SEQ.value, hidden)


def _replace_last_dim(shape: tuple[DimExpr, ...], last: DimExpr) -> tuple[DimExpr, ...]:
    if not shape:
        return (Symbol.BATCH.value, Symbol.SEQ.value, last)
    return (*shape[:-1], last)


def _replace_dim(
    shape: tuple[DimExpr, ...], dim: int, value: DimExpr
) -> tuple[DimExpr, ...]:
    """Return *shape* with position *dim* replaced by *value*."""
    lst = list(shape)
    if 0 <= dim < len(lst):
        lst[dim] = value
    return tuple(lst)


def _parse_split_sizes(
    text: str, dims: dict[str, DimExpr]
) -> list[DimExpr] | None:
    """Parse a split_size_or_sections detail like ``[qkv_dim] * 3`` or ``2048``."""
    text = text.strip()
    # "[name] * N" pattern
    m = re.match(r"\[(\w+)\]\s*\*\s*(\d+)", text)
    if m:
        name, count = m.group(1), int(m.group(2))
        resolved = _resolve_dim_name(name, dims)
        if resolved is not None:
            return [resolved] * count
        return None
    # Plain integer
    try:
        return [int(text)]
    except (ValueError, TypeError):
        pass
    # Comma-separated integers
    parts = text.split(",")
    if len(parts) > 1:
        try:
            return [int(p.strip()) for p in parts]
        except (ValueError, TypeError):
            pass
    return None


def _infer_einsum_shape(
    equation: str, inputs: list[TensorSpec]
) -> tuple[DimExpr, ...] | None:
    """Resolve output shape from an einsum equation and concrete input shapes.

    Supports the explicit ``"ij,jk->ik"`` form.  Each letter in the output
    subscript is mapped to the size it has in one of the input tensors.
    """
    parts = equation.replace(" ", "").split("->")
    if len(parts) != 2:
        return None
    input_subs = parts[0].split(",")
    output_sub = parts[1]
    if len(input_subs) != len(inputs):
        return None

    # Build letter → dimension size mapping from the inputs.
    letter_dim: dict[str, DimExpr] = {}
    for sub, spec in zip(input_subs, inputs):
        if len(sub) != len(spec.shape):
            return None
        for letter, dim_val in zip(sub, spec.shape):
            if letter not in letter_dim:
                letter_dim[letter] = dim_val

    out_shape = tuple(letter_dim.get(letter) for letter in output_sub)
    if any(d is None for d in out_shape):
        return None
    return out_shape  # type: ignore[return-value]


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


def _find_method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    """Find a method by *name* on *class_node*."""
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
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
            return _config_dim(node.attr, config=config, context=context)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return local_vars.get(node.attr)
    if isinstance(node, ast.Subscript):
        # `config.linear_attn_config["head_dim"]` and friends.
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            container = node.value
            if isinstance(container, ast.Attribute) and isinstance(
                container.value, ast.Name
            ):
                nested = config.get(container.attr)
                if isinstance(nested, dict):
                    return _int_dim(nested.get(key.value))
            return _config_dim(key.value, config=config, context=context)
    if isinstance(node, ast.BinOp):
        left = _resolve_dim_expr(
            node.left, config=config, local_vars=local_vars, context=context
        )
        right = _resolve_dim_expr(
            node.right, config=config, local_vars=local_vars, context=context
        )
        if left is None or right is None:
            return None
        if isinstance(left, int) and isinstance(right, int):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, (ast.FloorDiv, ast.Div)):
                return left // right if right else None
        return _symbolic_binop(left, right, node.op)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "int"
    ):
        if node.args:
            return _resolve_dim_expr(
                node.args[0], config=config, local_vars=local_vars, context=context
            )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
    ):
        if (
            node.args
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            return _config_dim(node.args[1].value, config=config, context=context)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _resolve_dim_expr(
            node.operand, config=config, local_vars=local_vars, context=context
        )
        if isinstance(inner, int):
            return -inner
    return None


_BINOP_SYMBOLS: dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.FloorDiv: "/",
    ast.Div: "/",
}


def _symbolic_binop(left: DimExpr, right: DimExpr, op: ast.operator) -> DimExpr | None:
    """Render a partially-resolved dimension as readable algebra (``4*H``), not a marker."""
    symbol = _BINOP_SYMBOLS.get(type(op))
    if symbol is None:
        return None
    return f"{_dim_term(left, symbol)}{symbol}{_dim_term(right, symbol)}"


def _dim_term(value: DimExpr, symbol: str) -> str:
    text = str(value)
    if symbol in {"*", "/"} and any(char in text for char in "+-"):
        return f"({text})"
    return text


def _broadcast_rank(spec: TensorSpec) -> tuple[int, float]:
    """Order operands of an elementwise op by what broadcasting keeps.

    Highest rank wins, then the widest trailing dimension; unresolved symbolic widths
    outrank concrete ones because they stand for the model's activation width.
    """
    last = spec.shape[-1] if spec.shape else 1
    width = float(last) if isinstance(last, int) else float("inf")
    return len(spec.shape), width


def _detail_value(details: Sequence[str], key: str) -> str | None:
    """Read a recorded call detail such as ``dim: -1``."""
    prefix = f"{key}:"
    for item in details:
        text = str(item).strip()
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return None


def _int_dim(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _config_dim(name: str, *, config: dict[str, Any], context: ShapeContext) -> DimExpr:
    """Resolve `config.<name>`, falling back to flattened sub-config aliases."""
    resolved = _int_dim(config.get(name))
    if resolved is not None:
        return resolved
    alias = context.dims.get(name)
    if isinstance(alias, int):
        return alias
    return name


def _nested_dim_aliases(name: str, mapping: dict[str, Any]):
    """Yield (alias, value) pairs for integer entries of a nested config dict.

    A sub-config named ``linear_attn_config`` is surfaced by HF config classes as both
    ``config.linear_attn_head_dim`` and ``config.linear_head_dim``, so register every
    leading-token prefix as well as the bare key.
    """
    tokens = [token for token in name.split("_") if token and token != "config"]
    prefixes = ["_".join(tokens[: index + 1]) for index in range(len(tokens))]
    for key, value in mapping.items():
        resolved = _int_dim(value)
        if resolved is None:
            continue
        yield key, resolved
        for prefix in prefixes:
            yield f"{prefix}_{key}", resolved


def _parse_module_ctor(
    node: ast.AST,
    *,
    config: dict[str, Any],
    local_vars: dict[str, DimExpr],
    context: ShapeContext,
) -> ModuleLinearSpec | ModuleEmbeddingSpec | ModuleParameterSpec | None:
    if not isinstance(node, ast.Call):
        return None
    class_name = _call_class_name(node) or ""
    args = list(node.args)
    if re.search(r"Linear$", class_name):
        in_features = (
            _resolve_dim_expr(
                args[0], config=config, local_vars=local_vars, context=context
            )
            if len(args) >= 1
            else None
        )
        out_features = (
            _resolve_dim_expr(
                args[1], config=config, local_vars=local_vars, context=context
            )
            if len(args) >= 2
            else None
        )
        for keyword in node.keywords:
            if keyword.arg == "in_features" and in_features is None:
                in_features = _resolve_dim_expr(
                    keyword.value, config=config, local_vars=local_vars, context=context
                )
            if keyword.arg == "out_features" and out_features is None:
                out_features = _resolve_dim_expr(
                    keyword.value, config=config, local_vars=local_vars, context=context
                )
        if in_features is not None and out_features is not None:
            return ModuleLinearSpec(in_features=in_features, out_features=out_features)
    if re.search(r"Embedding$", class_name):
        num_embeddings = (
            _resolve_dim_expr(
                args[0], config=config, local_vars=local_vars, context=context
            )
            if len(args) >= 1
            else None
        )
        embedding_dim = (
            _resolve_dim_expr(
                args[1], config=config, local_vars=local_vars, context=context
            )
            if len(args) >= 2
            else None
        )
        for keyword in node.keywords:
            if keyword.arg == "num_embeddings" and num_embeddings is None:
                num_embeddings = _resolve_dim_expr(
                    keyword.value, config=config, local_vars=local_vars, context=context
                )
            if keyword.arg == "embedding_dim" and embedding_dim is None:
                embedding_dim = _resolve_dim_expr(
                    keyword.value, config=config, local_vars=local_vars, context=context
                )
        if num_embeddings is not None and embedding_dim is not None:
            return ModuleEmbeddingSpec(
                num_embeddings=num_embeddings, embedding_dim=embedding_dim
            )
    if class_name == "Parameter":
        inner = args[0] if args else None
        shape = _parse_tensor_ctor_shape(
            inner, config=config, local_vars=local_vars, context=context
        )
        if shape:
            return ModuleParameterSpec(shape=shape)
    shape = _parse_tensor_ctor_shape(
        node, config=config, local_vars=local_vars, context=context
    )
    if shape:
        return ModuleParameterSpec(shape=shape)
    return None


_TENSOR_FACTORIES = {"empty", "zeros", "ones", "randn", "rand", "full", "tensor"}


def _parse_tensor_ctor_shape(
    node: ast.AST | None,
    *,
    config: dict[str, Any],
    local_vars: dict[str, DimExpr],
    context: ShapeContext,
) -> tuple[DimExpr, ...] | None:
    """Extract the shape from `torch.empty(a, b)` / `torch.zeros((a, b))` style calls."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr in _TENSOR_FACTORIES):
        return None
    args = list(node.args)
    if len(args) == 1 and isinstance(args[0], (ast.Tuple, ast.List)):
        args = list(args[0].elts)
    if func.attr == "full" and args:
        args = args[:1]
        if isinstance(args[0], (ast.Tuple, ast.List)):
            args = list(args[0].elts)
    dims: list[DimExpr] = []
    for arg in args:
        resolved = _resolve_dim_expr(
            arg, config=config, local_vars=local_vars, context=context
        )
        if resolved is None:
            return None
        dims.append(resolved)
    return tuple(dims) if dims else None


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
    if node.metadata.get("class_name") == "AttentionOp":
        return node.label or "Attention"
    attr = _node_attr_name(node)
    if attr:
        return attr
    return node.label or node.id


def _descendant_classes(root: BlockNode) -> dict[str, str]:
    classes: dict[str, str] = {}
    stack = [root]
    while stack:
        block = stack.pop()
        if block.attr_name and block.class_name:
            classes.setdefault(block.attr_name, block.class_name)
        stack.extend(block.children)
    return classes


def _node_attr_name(node: ModelGraphNode) -> str | None:
    metadata_attr = node.metadata.get("attr_name")
    if isinstance(metadata_attr, str) and metadata_attr:
        return metadata_attr
    node_id = node.id
    if node_id.startswith("@"):
        return None
    parts = node_id.split(":")
    for part in reversed(parts):
        if (
            part
            and not part.isdigit()
            and not re.match(r"^(fan\d+|merge|side|post|combine|node)$", part)
        ):
            return part
    return None


def _operator_name(node: ModelGraphNode) -> str:
    if node.metadata.get("class_name") == "AttentionOp":
        return node.label or "Attention"
    attr = _node_attr_name(node)
    if attr and is_forward_operation(attr):
        return operation_display_label(
            node.label or "", class_name=node.metadata.get("class_name")
        )
    if attr:
        return attr
    if node.metadata.get("synthetic") == "@input":
        return node.label
    if node.label in {"×", "+", "Elementwise ×", "Multiply", "Add"}:
        if node.label == "Multiply":
            return "×"
        if node.label == "Add":
            return "+"
        return node.label
    return node.label or node.id


def _export_operation_kind(node: ModelGraphNode) -> str:
    """Map exported operator kinds for shape-export consumers."""
    if node.operation is None:
        return "unknown"
    return node.operation.value


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
    if node.label in {"×", "+", "Elementwise ×", "Multiply", "Add"}:
        if node.label in {"×", "Multiply", "Elementwise ×"}:
            return "elementwise_mul"
        return "elementwise_add"
    return node.label or "unknown"


def _is_embedding(class_name: str, node: ModelGraphNode) -> bool:
    return bool(re.search(r"(?i)^Embedding$", class_name)) or node.label == "Embedding"


def _is_norm(class_name: str, node: ModelGraphNode) -> bool:
    return bool(
        re.search(r"(?i)(RMSNorm|LayerNorm|GroupNorm)$", class_name)
    ) or node.label in {
        "RMSNorm",
        "LayerNorm",
    }


def _is_linear(node: ModelGraphNode) -> bool:
    if node.operation not in {OperationKind.NN_MODULE, OperationKind.TORCH_FUNCTIONAL}:
        return False
    class_name = node.metadata.get("class_name") or node.label or ""
    return bool(re.search(r"(?i)^Linear$", str(class_name)))


def _heuristic_linear_out_features(
    attr: str | None, context: ShapeContext
) -> DimExpr | None:
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
    return "router" in class_name.lower() or "router" in _operator_name(node).lower()


def serialize_dim(value: DimExpr) -> int | str:
    """Serialize a dimension expression for JSON export."""
    return _serialize_dim(value)


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
    "serialize_dim",
]
