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
class ModuleConvSpec:
    """Channel dimensions of an ``nn.Conv{1,2,3}d`` declared in ``__init__``.

    The channel axis is always tracked: a conv maps ``(N, in_channels, *spatial)``
    to ``(N, out_channels, *spatial')``. When ``kernel_size``/``stride`` are
    captured as concrete ints, a *concrete* spatial extent is reduced via
    ``out = (in + 2*padding - kernel) // stride + 1``; symbolic spatial axes still
    pass through unchanged. ``None`` geometry means "kernel/stride unknown", in
    which case the spatial axes pass through as before.
    """

    in_channels: DimExpr | None
    out_channels: DimExpr
    kernel_size: tuple[int, ...] | None = None
    stride: tuple[int, ...] | None = None
    padding: tuple[int, ...] | None = None


def _conv_out_dim(in_dim: DimExpr, kernel: int, stride: int, padding: int) -> DimExpr:
    """Reduce one concrete spatial extent through a conv; pass symbolic dims through."""
    if not isinstance(in_dim, int) or stride <= 0:
        return in_dim
    return (in_dim + 2 * padding - kernel) // stride + 1


def _reduce_conv_spatial(
    shape: list[DimExpr],
    channel_axis: int,
    kernel: tuple[int, ...] | None,
    stride: tuple[int, ...] | None,
    padding: tuple[int, ...] | None,
) -> None:
    """In-place reduce the spatial axes (those after ``channel_axis``) of ``shape``.

    Applies ``_conv_out_dim`` per spatial axis when kernel/stride are known. The
    kernel/stride/padding tuples are broadcast (a length-1 tuple applies to every
    axis). No-ops when geometry is missing or spatial extents are symbolic.
    """
    if not kernel or not stride:
        return
    spatial_axes = range(channel_axis + 1, len(shape))

    def _at(values: tuple[int, ...] | None, idx: int, default: int) -> int:
        if not values:
            return default
        return values[idx] if idx < len(values) else values[-1]

    for idx, axis in enumerate(spatial_axes):
        shape[axis] = _conv_out_dim(
            shape[axis],
            _at(kernel, idx, 1),
            _at(stride, idx, 1),
            _at(padding, idx, 0),
        )


@dataclass
class ShapeContext:
    """Resolved and symbolic dimensions derived from model config."""

    dims: dict[str, DimExpr] = field(default_factory=dict)
    dtype: str = "float16"
    # Conv geometry keyed by constructor attr name (e.g. ``"downsample"``):
    # ``(kernel_size, stride, padding)`` as concrete-int tuples. Populated when the
    # module registry is built so the render-time fallback (which has no access to
    # the inferencer's ModuleConvSpec registry) can reduce spatial extents too.
    conv_geometry: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = (
        field(default_factory=dict)
    )
    # Weight storage dtype for quantized checkpoints (e.g. ``fp8_e4m3``); ``None``
    # when the model is not quantized. ``not_convert`` holds the normalized
    # ``modules_to_not_convert`` patterns (kept at the compute dtype).
    quant_dtype: str | None = None
    not_convert: tuple[tuple[str, ...], ...] = ()

    def weight_dtype(self, node_id: str) -> str:
        """Storage dtype of a weight node: quant dtype unless its module is kept
        at full precision (in ``modules_to_not_convert``)."""
        if not self.quant_dtype:
            return self.dtype
        segments = _module_path_segments(node_id)
        for pattern in self.not_convert:
            if _module_path_matches(segments, pattern):
                return self.dtype
        return self.quant_dtype

    @classmethod
    def from_spec(cls, spec: ArchitectureSpec) -> ShapeContext:
        config = spec.raw_config or {}
        dtype = _config_dtype(config)
        quant = config.get("quantization_config")
        quant_dtype: str | None = None
        not_convert: tuple[tuple[str, ...], ...] = ()
        if isinstance(quant, dict):
            quant_dtype = _quant_storage_dtype(quant)
            not_convert = _normalize_module_patterns(
                quant.get("modules_to_not_convert")
            )
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

        # Fold ``__init__`` self-attribute scalars (e.g.
        # ``self.qkv_dim = self.head_dim * self.num_heads``) so shape expressions
        # that reference them (``.split([self.qkv_dim] * 3)``) resolve.
        for name, value in _collect_init_scalar_attrs(spec, dims, config).items():
            dims.setdefault(name, value)

        # Fold forward-local scalar bindings (e.g. `hc = self.hc_mult`) so shape
        # expressions that use them (`.split([hc, hc, hc * hc])`) resolve.
        for name, value in _collect_forward_scalar_locals(spec, dims).items():
            dims.setdefault(name, value)

        return cls(
            dims=dims,
            dtype=dtype,
            quant_dtype=quant_dtype,
            not_convert=not_convert,
        )


@dataclass
class ModuleDimRegistry:
    """Linear and embedding constructor dimensions parsed from modeling AST."""

    linear: dict[tuple[str, str], ModuleLinearSpec] = field(default_factory=dict)
    linear_by_attr: dict[str, ModuleLinearSpec] = field(default_factory=dict)
    embedding: dict[tuple[str, str], ModuleEmbeddingSpec] = field(default_factory=dict)
    embedding_by_attr: dict[str, ModuleEmbeddingSpec] = field(default_factory=dict)
    parameter: dict[tuple[str, str], ModuleParameterSpec] = field(default_factory=dict)
    parameter_by_attr: dict[str, ModuleParameterSpec] = field(default_factory=dict)
    conv: dict[tuple[str, str], ModuleConvSpec] = field(default_factory=dict)
    conv_by_attr: dict[str, ModuleConvSpec] = field(default_factory=dict)
    # Scalar ``self.head_dim = config.linear_head_dim`` style dims, per class. The
    # global config dim of the same name can be a zero placeholder (GLM's
    # ``head_dim``), so a reshape naming ``self.head_dim`` must resolve against the
    # owning module's own constructor value.
    scalar_by_class: dict[str, dict[str, DimExpr]] = field(default_factory=dict)
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
            registry._walk_init_body(
                init_func.body,
                class_name=class_name,
                config=config,
                local_vars=local_vars,
                context=context,
            )
        return registry

    def _walk_init_body(
        self,
        stmts: list[ast.stmt],
        *,
        class_name: str,
        config: dict[str, Any],
        local_vars: dict[str, DimExpr],
        context: ShapeContext,
    ) -> None:
        """Walk __init__ body in statement order, evaluating ``if`` conditions
        against the model config so conditional assignments are resolved correctly.

        This replaces the previous ``ast.walk`` approach which processed
        assignments in BFS order, causing later-executed conditional
        overrides to be visited *after* constructor calls that depend on them.
        """
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    self._record_assignment(
                        class_name, target, stmt.value,
                        config=config, local_vars=local_vars, context=context,
                    )
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                self._record_assignment(
                    class_name, stmt.target, stmt.value,
                    config=config, local_vars=local_vars, context=context,
                )
            elif isinstance(stmt, ast.If):
                # Evaluate the condition against config and local_vars.
                branch = _eval_config_condition(
                    stmt.test, config=config, local_vars=local_vars
                )
                if branch is True:
                    self._walk_init_body(
                        stmt.body, class_name=class_name,
                        config=config, local_vars=local_vars, context=context,
                    )
                elif branch is False:
                    self._walk_init_body(
                        stmt.orelse, class_name=class_name,
                        config=config, local_vars=local_vars, context=context,
                    )
                else:
                    # Cannot evaluate condition — process both branches so
                    # we don't miss assignments.  Later branch wins.
                    self._walk_init_body(
                        stmt.body, class_name=class_name,
                        config=config, local_vars=local_vars, context=context,
                    )
                    self._walk_init_body(
                        stmt.orelse, class_name=class_name,
                        config=config, local_vars=local_vars, context=context,
                    )
            elif isinstance(stmt, (ast.For, ast.While, ast.With)):
                self._walk_init_body(
                    stmt.body, class_name=class_name,
                    config=config, local_vars=local_vars, context=context,
                )
            elif isinstance(stmt, ast.Try):
                self._walk_init_body(
                    stmt.body, class_name=class_name,
                    config=config, local_vars=local_vars, context=context,
                )

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
            self.scalar_by_class.setdefault(class_name, {})[target.attr] = resolved
        spec = _parse_module_ctor(
            value, config=config, local_vars=local_vars, context=context
        )
        if isinstance(spec, ModuleLinearSpec):
            self.linear[(class_name, target.attr)] = spec
            self.linear_by_attr[target.attr] = spec
        elif isinstance(spec, ModuleConvSpec):
            self.conv[(class_name, target.attr)] = spec
            self.conv_by_attr[target.attr] = spec
            if spec.kernel_size is not None and spec.stride is not None:
                # Expose geometry to the render-time fallback, which reduces
                # spatial extents without access to this registry.
                context.conv_geometry[target.attr] = (
                    spec.kernel_size,
                    spec.stride,
                    spec.padding or (0,),
                )
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
        # Bitwise/logical mask combinators and in-place copies keep the widest
        # operand's shape, like any other element-wise op.
        "bitwise and",
        "bitwise or",
        "bitwise xor",
        "copy",
    }
)


# ── Torch per-op shape execution (ground truth for shape-changing ops) ───────
# Distinctive probe values for the symbolic batch/sequence dims — primes chosen
# so a materialized op's output dims don't accidentally collide with a config
# dimension (e.g. head_dim=128), which would misread as the sequence length.
_TORCH_PROBE_BATCH = 2
_TORCH_PROBE_SEQ = 137


def _torch_dtype(torch: Any, name: str) -> Any:
    return getattr(torch, str(name).replace("torch.", ""), torch.float32)


def _first_tensor_shape(out: Any) -> tuple[int, ...] | None:
    import torch

    if isinstance(out, torch.Tensor):
        return tuple(int(d) for d in out.shape)
    if isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return tuple(int(d) for d in item.shape)
    return None


def _resolve_op_int(value: Any, dims: dict[str, Any]) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        resolved = dims.get(str(value))
        return resolved if isinstance(resolved, int) else None


def _torch_op_split(torch, metas, details, dims):
    dim = _int_dim(_detail_value(details, "dim") or "0") or 0
    size = _resolve_op_int(_detail_value(details, "split_size"), dims)
    if size is None or size <= 0:
        return None
    return torch.split(metas[0], size, dim=dim)[0]


def _torch_op_chunk(torch, metas, details, dims):
    dim = _int_dim(_detail_value(details, "dim") or "0") or 0
    n = _resolve_op_int(
        _detail_value(details, "split_size") or _detail_value(details, "chunks"), dims
    )
    if n is None or n <= 0:
        return None
    return torch.chunk(metas[0], n, dim=dim)[0]


def _torch_op_unbind(torch, metas, details, dims):
    dim = _int_dim(_detail_value(details, "dim") or "0") or 0
    pieces = torch.unbind(metas[0], dim=dim)
    return pieces[0] if pieces else None


def _torch_op_unflatten(torch, metas, details, dims):
    dim = _int_dim(_detail_value(details, "dim") or "0") or 0
    sizes_text = _detail_value(details, "sizes") or _detail_value(details, "shape")
    if not sizes_text:
        return None
    sizes: list[int] = []
    for part in sizes_text.strip().strip("()[]").split(","):
        part = part.strip()
        if not part:
            continue
        resolved = _resolve_op_int(part, dims)
        sizes.append(resolved if resolved is not None else -1)
    if not sizes or sizes.count(-1) > 1:
        return None
    return torch.unflatten(metas[0], dim, sizes)


def _normalize_op_name(name: Any) -> str:
    """Normalize an op name for cross-source matching (must stay identical to
    ``meta_trace._norm_op`` so AST ids and FX node names join)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Trailing ``@op_l{line}_c{col}_{name}[:idx]`` token of a graph node id, with
# everything before it captured as the block-instance prefix.
_LAST_OP_ID_RE = re.compile(
    r"^(?P<prefix>.*):@op_l(?P<line>\d+)_c(?P<col>\d+)_(?P<name>[a-z0-9_]+?)"
    r"(?::(?P<idx>\d+))?$"
)


# Non-parametric shape-transforming ops where torch execution gives the ground
# truth the symbolic arithmetic often can't resolve. Keyed by normalized label.
_TORCH_OP_BUILDERS = {
    "split": _torch_op_split,
    "chunk": _torch_op_chunk,
    "unbind": _torch_op_unbind,
    "unflatten": _torch_op_unflatten,
}


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
        # Checkpoint retained for the lazy per-op FX fallback (see below).
        self._meta_checkpoint: str | Path | None = None
        # Per-op FX ground-truth shapes, keyed by (line, op, occurrence).
        # None = not yet built; built lazily only if an op reaches the
        # "no symbolic rule" fallback (so models fully covered by symbolic
        # rules — e.g. GLM-5.3 — pay nothing for it).
        self._op_fx_shapes: dict[tuple[int, str, int], tuple[Any, ...]] | None = None
        # node.id -> occurrence index of this op on its source line within its
        # block instance (matches the FX-side occurrence counting).
        self._op_line_occ: dict[str, int] = {}

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

        # Retain for the lazy per-op FX fallback, even if module-level tracing
        # below captures nothing.
        self._meta_checkpoint = checkpoint
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
        self._register_op_line_occurrences(graph)
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
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(
                shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype
            )

        if synthetic == "@tensor":
            label = (node.metadata.get("port_label") or node.label or "").lower()
            experts = self.context.dims.get(Symbol.EXPERTS.value, Symbol.EXPERTS.value)
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            parameter = self._lookup_parameter_spec(node, root=root, names=[label])
            if parameter is not None:
                param_dtype = (
                    self.context.weight_dtype(str(node.id))
                    if "weight" in label
                    else dtype
                )
                return TensorSpec(shape=parameter.shape, dtype=param_dtype)
            if "weight" in label:
                return TensorSpec(
                    shape=(experts, hidden),
                    dtype=self.context.weight_dtype(str(node.id)),
                )
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
            has_weight = any("weight" in item for item in external_inputs)
            if parameter is not None:
                param_dtype = (
                    self.context.weight_dtype(str(node.id)) if has_weight else dtype
                )
                return TensorSpec(parameter.shape, param_dtype)
            if has_weight:
                return TensorSpec(
                    (experts, hidden), self.context.weight_dtype(str(node.id))
                )
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
            # symbolic dimension names from the model config). Overlay the owning
            # module's own scalar dims so ``self.head_dim`` resolves to that
            # module's value, not a global zero placeholder.
            view_dims = self.context.dims
            # The op may live directly in ``root``'s forward (block-local id with no
            # module segment), so fall back to the block's own class.
            owner = self._owner_class_name(node, root=root) or (
                root.class_name if root is not None else None
            )
            owner_scalars = (
                self.module_dims.scalar_by_class.get(owner) if owner else None
            )
            if owner_scalars:
                view_dims = {**self.context.dims, **owner_scalars}
            resolved = _resolve_view_shape(shape_detail, source, view_dims)
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
            if operation_label == "permute" and source.shape:
                dims_str = _detail_value(details, "dims")
                permuted = _permute_shape(source.shape, dims_str)
                if permuted is not None:
                    return TensorSpec(shape=permuted, dtype=source.dtype)
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
            if operation_label == "cast" and dtype_detail:
                cast_dtype = _resolve_cast_dtype(
                    dtype_detail, source.dtype, self.context.dtype
                )
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

        # An activation module resolved from a registry (e.g. ``act_fn = ACT2FN[...]``)
        # keeps its attribute name as the label (``act_fn``) while its class is the
        # concrete activation (``SiLU``); match on the class so it is treated as the
        # pointwise op it is rather than falling through to the shape warning.
        if (
            operation_label in _POINTWISE_LABELS
            or (class_name or "").strip().lower() in _POINTWISE_LABELS
        ):
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
                if parameter is None:
                    # The weight arg (`self.fn.float()`) is often not recorded as an
                    # external input, so fall back to the owning class's sole matrix
                    # Parameter (e.g. the mHC ``fn`` weight in HyperConnection).
                    parameter = self._unique_matrix_parameter(node, root)
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

        conv_spec = self._lookup_conv_spec(node, root=root)
        if conv_spec is not None or _is_conv(node):
            source = (
                inputs[0]
                if inputs
                else TensorSpec(_default_hidden_shape(self.context), dtype)
            )
            out_channels = (
                conv_spec.out_channels
                if conv_spec is not None
                else self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            )
            shape = list(source.shape)
            if not shape:
                return TensorSpec(shape=(out_channels,), dtype=source.dtype)
            # Channel-axis only: replace the axis carrying in_channels (or the
            # conventional channel axis 1) with out_channels; spatial axes pass through.
            channel_axis = 1 if len(shape) >= 2 else 0
            if conv_spec is not None and conv_spec.in_channels is not None:
                for axis, dim in enumerate(shape):
                    if dim == conv_spec.in_channels:
                        channel_axis = axis
                        break
            shape[channel_axis] = out_channels
            if conv_spec is not None:
                _reduce_conv_spatial(
                    shape,
                    channel_axis,
                    conv_spec.kernel_size,
                    conv_spec.stride,
                    conv_spec.padding,
                )
            return TensorSpec(shape=tuple(shape), dtype=source.dtype)

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
            fx_spec = self._fx_op_shape(node, inputs)
            if fx_spec is not None:
                return fx_spec
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

        # Fallback: attention modules produce (B, S, H) — derive from config.
        node_name = (
            node.metadata.get("attr_name") or node.id.rsplit(":", 1)[-1] or ""
        ).lower()
        if "attention" in node_name:
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            return TensorSpec(
                shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden), dtype=dtype
            )

        # Genuinely unknown op (no symbolic rule): get a ground-truth shape from
        # the per-module FX pass, or by running the op on the meta device,
        # before falling back to passing through / (B, S, H).
        fx_spec = self._fx_op_shape(node, inputs)
        if fx_spec is not None:
            return fx_spec
        torch_op_spec = self._torch_op_shape(node, inputs)
        if torch_op_spec is not None:
            return torch_op_spec

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

    def _torch_op_label(self, node: ModelGraphNode) -> str:
        raw = node.label or node.metadata.get("class_name") or ""
        return str(raw).strip().lower().replace(" ", "")

    def _concrete_dim(self, dim: DimExpr) -> int | None:
        if isinstance(dim, int):
            return dim
        text = str(dim)
        if text == Symbol.BATCH.value:
            return _TORCH_PROBE_BATCH
        if text == Symbol.SEQ.value:
            return _TORCH_PROBE_SEQ
        value = self.context.dims.get(text)
        return value if isinstance(value, int) else None

    def _concrete_shape(self, spec: TensorSpec) -> tuple[int, ...] | None:
        resolved: list[int] = []
        for dim in spec.shape:
            value = self._concrete_dim(dim)
            if value is None or value < 0:
                return None
            resolved.append(int(value))
        return tuple(resolved)

    def _symbolise_concrete(self, shape: tuple[int, ...]) -> tuple[Any, ...]:
        from TraceLens.ModelUtils.meta_trace import symbolise_meta_shape

        return symbolise_meta_shape(
            shape, batch_size=_TORCH_PROBE_BATCH, seq_len=_TORCH_PROBE_SEQ
        )

    def _torch_op_shape(
        self, node: ModelGraphNode, inputs: list[TensorSpec]
    ) -> TensorSpec | None:
        """Run a shape-changing op on the meta device to get its true output
        shape, when the symbolic arithmetic can't resolve it (e.g. a split
        whose size is a config-derived name). Best-effort: returns None on any
        gap (unmapped dim, unknown op, execution error) so the caller falls
        back to symbolic inference."""
        if not inputs:
            return None
        builder = _TORCH_OP_BUILDERS.get(self._torch_op_label(node))
        if builder is None:
            return None
        try:
            import torch
        except ImportError:
            return None
        metas = []
        for spec in inputs:
            concrete = self._concrete_shape(spec)
            if concrete is None:
                return None
            try:
                metas.append(
                    torch.zeros(
                        concrete, dtype=_torch_dtype(torch, spec.dtype), device="meta"
                    )
                )
            except Exception:  # noqa: BLE001
                return None
        details = [str(item) for item in node.metadata.get("details", [])]
        try:
            with torch.device("meta"):
                out = builder(torch, metas, details, self.context.dims)
        except Exception:  # noqa: BLE001
            return None
        if out is None:
            return None
        shape = _first_tensor_shape(out)
        if shape is None:
            return None
        return TensorSpec(self._symbolise_concrete(shape), inputs[0].dtype)

    # ------------------------------------------------------------------
    # Per-op FX fallback (ground truth for ops with no symbolic rule)
    # ------------------------------------------------------------------

    def _register_op_line_occurrences(self, graph: ModelGraph) -> None:
        """Assign each op node its occurrence index on its source line within
        its block instance, matching the FX-side occurrence counting used by
        :func:`trace_meta_op_shapes` so the two line up when joined."""
        groups: dict[str, list[tuple[int, int, str, str]]] = {}
        for node in graph.nodes:
            match = _LAST_OP_ID_RE.match(node.id)
            if match is None:
                continue
            groups.setdefault(match["prefix"], []).append(
                (
                    int(match["line"]),
                    int(match["col"]),
                    _normalize_op_name(match["name"]),
                    node.id,
                )
            )
        for items in groups.values():
            items.sort()
            counter: dict[tuple[int, str], int] = {}
            for line, _col, base, node_id in items:
                key = (line, base)
                occ = counter.get(key, 0)
                counter[key] = occ + 1
                self._op_line_occ[node_id] = occ

    def _op_line_key(
        self, node: ModelGraphNode
    ) -> tuple[int, str, int] | None:
        match = _LAST_OP_ID_RE.match(node.id)
        if match is None:
            return None
        return (
            int(match["line"]),
            _normalize_op_name(match["name"]),
            self._op_line_occ.get(node.id, 0),
        )

    def _ensure_op_fx_shapes(self) -> None:
        if self._op_fx_shapes is not None:
            return
        self._op_fx_shapes = {}
        if self._meta_checkpoint is None:
            return
        try:
            from TraceLens.ModelUtils.meta_trace import trace_meta_op_shapes

            captured = trace_meta_op_shapes(
                self._meta_checkpoint, config=self.spec.raw_config
            )
        except Exception as exc:  # noqa: BLE001
            _log.debug("Per-op FX shape tracing failed: %s", exc)
            return
        if captured:
            self._op_fx_shapes = captured

    def _fx_op_shape(
        self, node: ModelGraphNode, inputs: list[TensorSpec]
    ) -> TensorSpec | None:
        """Ground-truth output shape for an op with no symbolic rule, from a
        per-module FX + ShapeProp pass on the meta device (built lazily on
        first use). Returns None when unavailable or unmatched."""
        if self._meta_checkpoint is None:
            return None
        key = self._op_line_key(node)
        if key is None:
            return None
        self._ensure_op_fx_shapes()
        shape = (self._op_fx_shapes or {}).get(key)
        if shape is None:
            return None
        dtype = inputs[0].dtype if inputs else self.context.dtype
        return TensorSpec(tuple(shape), dtype)

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

        # Fuzzy match: convert snake_case attr name to CamelCase and search
        # the registry (e.g. "core_attention" → "CoreAttention").
        structure = self._fuzzy_registry_lookup(class_name)
        if structure is not None:
            # Attention modules use complex runtime reshapes (.size() calls,
            # multi-head view/transpose) that AST simulation cannot resolve.
            # Derive output from config: attention always produces (B, S, H).
            if "attention" in (structure.name or "").lower():
                hidden = self.context.dims.get(
                    Symbol.HIDDEN.value, Symbol.HIDDEN.value
                )
                return TensorSpec(
                    shape=(Symbol.BATCH.value, Symbol.SEQ.value, hidden),
                    dtype=self.context.dtype,
                )
            result = self._simulate_forward_ops(
                structure, inputs, root=root, guard_name=class_name
            )
            if result is not None and _looks_valid(result, inputs):
                return result

        # Try introspecting inline function definitions in parent classes'
        # __init__ (e.g. ``self.activation_func = swiglu`` where swiglu is
        # defined as a nested function).
        inline_result = self._introspect_inline_function(
            class_name, inputs, root=root
        )
        if inline_result is not None:
            return inline_result

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

    def _fuzzy_registry_lookup(self, name: str):
        """Try to match *name* against registry classes using common transforms.

        Handles snake_case → CamelCase (``core_attention`` → ``CoreAttention``)
        and partial suffix matches (``attention_func`` → ``CoreAttention``).
        """
        # snake_case → CamelCase
        camel = "".join(part.capitalize() for part in name.split("_"))
        structure = self.spec.class_registry.get(camel)
        if structure is not None:
            return structure
        # Try suffix match: find classes whose name ends with the camel form.
        for cls_name, structure in self.spec.class_registry.items():
            if cls_name.lower() == name.replace("_", "").lower():
                return structure
        return None

    def _introspect_inline_function(
        self,
        attr_name: str,
        inputs: list[TensorSpec],
        *,
        root: BlockNode | None,
    ) -> TensorSpec | None:
        """Introspect inline functions assigned in a parent class's ``__init__``.

        Handles patterns like::

            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]
            self.activation_func = swiglu
        """
        from TraceLens.ModelUtils.ast_analyze import (
            _ForwardOperationExtractor,
            _extract_forward_return_metadata,
        )
        guard = f"inline:{attr_name}"
        if guard in self._introspecting:
            return None

        for _cls_name, structure in self.spec.class_registry.items():
            func_node = self._find_init_inline_function(
                structure.node, attr_name
            )
            if func_node is None:
                continue
            # Parse the inline function body using the same approach as
            # _introspect_method_shape.
            try:
                input_name = (
                    func_node.args.args[0].arg
                    if func_node.args.args
                    else "x"
                )
                extractor = _ForwardOperationExtractor(
                    self_values=structure.init_assignments,
                    all_tensor_ops=True,
                )
                extractor.var_producer[input_name] = input_name
                extractor.statements(func_node.body)
                if not extractor.operations:
                    continue
                fwd_ops = {op.attr_name: op for op in extractor.operations}
                _slots, _order, primary = _extract_forward_return_metadata(
                    func_node, extractor.var_producer
                )

                class _InlineProxy:
                    forward_operations = fwd_ops
                    forward_input_name = input_name
                    primary_return_slot = (
                        extractor.var_producer.get(primary) if primary else None
                    )

                self._introspecting.add(guard)
                try:
                    result = self._simulate_forward_ops(
                        _InlineProxy(),  # type: ignore[arg-type]
                        inputs,
                        root=root,
                        guard_name=guard,
                    )
                finally:
                    self._introspecting.discard(guard)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    @staticmethod
    def _find_init_inline_function(
        class_node: ast.ClassDef, attr_name: str
    ) -> ast.FunctionDef | None:
        """Find an inline function assigned to ``self.<attr_name>`` in ``__init__``."""
        init_func = _find_method(class_node, "__init__")
        if init_func is None:
            return None
        # Collect nested function definitions in __init__.
        nested_funcs: dict[str, ast.FunctionDef] = {}
        for stmt in ast.walk(init_func):
            if isinstance(stmt, ast.FunctionDef):
                nested_funcs[stmt.name] = stmt
        # Find ``self.<attr_name> = <func_name>`` assignments.
        for stmt in ast.walk(init_func):
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == attr_name
                    and isinstance(stmt.value, ast.Name)
                    and stmt.value.id in nested_funcs
                ):
                    return nested_funcs[stmt.value.id]
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

    def _lookup_conv_spec(
        self, node: ModelGraphNode, *, root: BlockNode | None
    ) -> ModuleConvSpec | None:
        attr = _node_attr_name(node)
        class_name = node.metadata.get("class_name")
        if class_name and attr:
            spec = self.module_dims.conv.get((class_name, attr))
            if spec is not None:
                return spec
        if attr:
            return self.module_dims.conv_by_attr.get(attr)
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

    def _unique_matrix_parameter(
        self, node: ModelGraphNode, root: BlockNode | None
    ) -> ModuleParameterSpec | None:
        """The sole 2-D Parameter of the class owning a functional ``F.linear``.

        When the weight argument isn't captured as an external input we cannot
        name it, but a functional linear whose owning class declares exactly one
        matrix Parameter (``self.fn`` for the mHC mapping) has an unambiguous
        weight — use its row axis as ``out_features``.
        """
        candidates = [
            node.metadata.get("class_name"),
            self._owner_class_name(node, root),
            root.class_name if root is not None else None,
        ]
        for class_name in candidates:
            if not class_name:
                continue
            matrices = [
                spec
                for (cls, _attr), spec in self.module_dims.parameter.items()
                if cls == class_name and len(spec.shape) >= 2
            ]
            if len(matrices) == 1:
                return matrices[0]
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
    # Strip ``self.``/``config.``/``self.config.`` access prefixes, e.g.
    # ``self.config.out_hidden_size`` → ``out_hidden_size``.
    bare = name
    for prefix in ("self.config.", "config.", "self."):
        if bare.startswith(prefix):
            bare = bare[len(prefix):]
            break
    if bare != name:
        val = dims.get(bare)
        if val is not None:
            return val
    # Try common config suffixes: ``hc`` → ``hc_mult``
    for suffix in ("_mult", "_size", "_dim", "_count"):
        val = dims.get(bare + suffix)
        if val is not None:
            return val
    return None


def _dim_factors(dim: DimExpr) -> list[str]:
    """Split a possibly-merged dim like ``B*S`` into its individual factors."""
    return [token for token in str(dim).split("*") if token]


def _merge_flatten_dim(
    source_shape: tuple[DimExpr, ...], explicit_dims: list[DimExpr]
) -> str | None:
    """Compute the ``-1`` dim of a reshape by conservation of elements.

    The flattened axis equals ``prod(source) / prod(explicit target dims)``.
    Symbolic factors (``B``, ``S``) cancel against matching source factors and
    numeric factors divide, yielding a readable product like ``B*S`` (or ``H`` /
    ``4096``). Returns ``None`` when the numerics do not divide cleanly or a
    target symbol has no matching source factor, so the caller can fall back.
    """

    def collect(dimlist: list[DimExpr]) -> tuple[int, list[str]]:
        num = 1
        sym: list[str] = []
        for dim in dimlist:
            for token in _dim_factors(dim):
                if token == "-1":
                    # The flatten placeholder; not a real target factor.
                    continue
                if token.lstrip("-").isdigit():
                    num *= int(token)
                else:
                    sym.append(token)
        return num, sym

    src_num, src_sym = collect(list(source_shape))
    tgt_num, tgt_sym = collect(list(explicit_dims))

    remaining = list(src_sym)
    for symbol in tgt_sym:
        if symbol in remaining:
            remaining.remove(symbol)
        else:
            return None
    if tgt_num == 0:
        return None
    if src_num % tgt_num == 0:
        num = src_num // tgt_num
        factors = remaining + ([str(num)] if num != 1 else [])
        return "*".join(factors) if factors else "1"
    # The numeric part does not divide evenly (e.g. flattening ``[B*S, 4096]``
    # to ``[-1, 2, 2, 4096]`` leaves ``B*S/4``). When the target numeric is an
    # exact multiple of the source numeric, express the flatten dim as the
    # remaining symbolic product divided by that factor rather than giving up —
    # this keeps rank-preserving reshapes (vision spatial-merge, patch pooling)
    # from collapsing to a no-op pass-through.
    if remaining and tgt_num % src_num == 0:
        divisor = tgt_num // src_num
        base = "*".join(remaining)
        return f"{base}/{divisor}" if divisor != 1 else base
    return None


def _permute_shape(
    source_shape: tuple[DimExpr, ...], dims_str: str | None
) -> tuple[DimExpr, ...] | None:
    """Reorder ``source_shape`` by a captured ``permute`` dims spec.

    ``dims_str`` is a comma-joined axis list (``"0, 3, 1, 2"``). Returns the
    reordered shape only when the axes form a genuine permutation of the source
    rank; otherwise ``None`` so callers fall back to a pass-through rather than
    corrupting the rank.
    """
    if not source_shape or dims_str is None:
        return None
    n = len(source_shape)
    axes: list[int | None] = []
    for part in dims_str.split(","):
        try:
            axes.append(int(part.strip()))
        except ValueError:
            axes.append(None)
    if len(axes) != n or any(a is None for a in axes):
        return None
    resolved = [a % n for a in axes]  # type: ignore[operator]
    if sorted(resolved) != list(range(n)):
        return None
    return tuple(source_shape[a] for a in resolved)


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
        m = re.search(r"\.shape\[:\s*(-?\d+)\]", first)
        if m:
            cut = int(m.group(1))
            leading = source.shape[:cut] if cut < 0 else source.shape[:cut]
        else:
            leading = source.shape
        trailing_start = 1

    resolved: list[DimExpr] = list(leading)
    neg_index: int | None = None
    for part in parts[trailing_start:]:
        # A ``-1`` axis is resolved last, once every other dim is known.
        if part == "-1":
            if neg_index is not None:
                # More than one ``-1`` cannot be resolved by element conservation.
                return None
            neg_index = len(resolved)
            resolved.append(part)
            continue
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
        # A ``x.shape[i]`` axis (from an unpacked ``a, b = x.shape[:2]`` local)
        # copies that positional dim from the reshape's source — the leading
        # batch/seq axes a reshape preserves.
        shape_ref = re.match(r"^[\w.]+\.shape\[(-?\d+)\]$", part)
        if shape_ref is not None:
            axis = int(shape_ref.group(1))
            if -len(source.shape) <= axis < len(source.shape):
                resolved.append(source.shape[axis])
                continue
        # Cannot resolve — give up
        return None

    if neg_index is not None:
        explicit = [dim for index, dim in enumerate(resolved) if index != neg_index]
        merged = _merge_flatten_dim(source.shape, explicit)
        if merged is None:
            return None
        resolved[neg_index] = merged

    return tuple(resolved) if resolved else None


def _looks_valid(result: TensorSpec, inputs: list[TensorSpec]) -> bool:
    """Sanity-check an introspection result against its inputs.

    Returns *False* when the result looks like a simulation artifact — e.g. the
    batch/seq dimensions are in wrong positions or the output rank shrank
    suspiciously (indicating the AST simulation hit runtime-dependent reshapes
    it couldn't resolve).
    """
    if not result.shape:
        return False
    if not inputs:
        return True
    # If any input had symbolic B or S and the result lost them, likely wrong.
    has_batch = Symbol.BATCH.value in result.shape or any(
        isinstance(d, str) and "B" in str(d) for d in result.shape
    )
    input_has_batch = any(
        Symbol.BATCH.value in inp.shape or any(
            isinstance(d, str) and "B" in str(d) for d in inp.shape
        )
        for inp in inputs
    )
    if input_has_batch and not has_batch:
        return False
    return True


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


def _eval_dim_expr(node: Any, dims: dict[str, DimExpr]) -> int | None:
    """Evaluate a small integer dimension expression against the config dims.

    Handles the arithmetic that appears in real ``.split``/``.view`` sizes —
    ``hc``, ``hc * hc``, ``self.hc_mult``, ``hidden_size // 2`` — by resolving
    every name to a config value and folding ``+ - * //``. Returns None if any
    name is unresolved (so the caller can keep the size symbolic).
    """
    import ast as _pyast

    if isinstance(node, str):
        try:
            node = _pyast.parse(node.strip(), mode="eval")
        except SyntaxError:
            return None
    if isinstance(node, _pyast.Expression):
        return _eval_dim_expr(node.body, dims)
    if isinstance(node, _pyast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, _pyast.Name):
        value = dims.get(node.id)
        return value if isinstance(value, int) else None
    if isinstance(node, _pyast.Attribute):
        # self.hc_mult / config.hidden_size → resolve by the attribute name.
        value = dims.get(node.attr)
        return value if isinstance(value, int) else None
    if isinstance(node, _pyast.BinOp):
        left = _eval_dim_expr(node.left, dims)
        right = _eval_dim_expr(node.right, dims)
        if left is None or right is None:
            return None
        if isinstance(node.op, _pyast.Mult):
            return left * right
        if isinstance(node.op, _pyast.Add):
            return left + right
        if isinstance(node.op, _pyast.Sub):
            return left - right
        if isinstance(node.op, (_pyast.FloorDiv, _pyast.Div)):
            return left // right if right else None
    if isinstance(node, _pyast.UnaryOp) and isinstance(node.op, _pyast.USub):
        inner = _eval_dim_expr(node.operand, dims)
        return -inner if inner is not None else None
    return None


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current = ""
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current)
    return parts


def _parse_split_sizes(
    text: str, dims: dict[str, DimExpr]
) -> list[DimExpr] | None:
    """Resolve a split_size_or_sections detail to concrete sizes.

    Handles ``[qkv_dim] * 3``, ``2048``, and — critically — a list of arbitrary
    integer expressions such as ``[hc, hc, hc * hc]`` by evaluating each entry
    against the config dims (including forward-local scalar bindings folded in
    by ``ShapeContext.from_spec``).
    """
    text = text.strip()
    # "[name] * N" pattern — name may be dotted (``self.qkv_dim``).
    m = re.match(r"\[([\w.]+)\]\s*\*\s*(\d+)", text)
    if m:
        name, count = m.group(1), int(m.group(2))
        resolved = _resolve_dim_name(name, dims)
        if resolved is not None:
            return [resolved] * count
        return None
    # A bracketed list of expressions: [hc, hc, hc * hc], [head_dim, head_dim].
    if text.startswith("[") and text.endswith("]"):
        entries = _split_top_level_commas(text[1:-1])
        sizes = [_eval_dim_expr(entry, dims) for entry in entries if entry.strip()]
        if sizes and all(isinstance(s, int) for s in sizes):
            return sizes
        return None
    # Plain integer or single expression.
    single = _eval_dim_expr(text, dims)
    if single is not None:
        return [single]
    # Comma-separated expressions.
    parts = _split_top_level_commas(text)
    if len(parts) > 1:
        sizes = [_eval_dim_expr(p, dims) for p in parts]
        if all(isinstance(s, int) for s in sizes):
            return sizes
    return None


def _collect_init_scalar_attrs(
    spec: ArchitectureSpec, dims: dict[str, DimExpr], config: dict[str, Any]
) -> dict[str, int]:
    """Fold ``__init__`` self-attribute scalars (``self.qkv_dim = ...``) into dims.

    A forward often reads an integer attribute set in ``__init__`` from config
    values (``self.qkv_dim = self.head_dim * self.num_heads``) and then uses it
    in a shape expression (``.split([self.qkv_dim] * 3)``). Resolve those
    attributes to their integer values by walking each class's ``__init__`` in
    statement order, carrying forward earlier self-attrs and plain locals.
    """
    import ast as _pyast

    ctx = ShapeContext(dims=dict(dims))
    extra: dict[str, int] = {}
    for structure in (getattr(spec, "class_registry", None) or {}).values():
        class_node = getattr(structure, "node", None)
        if class_node is None:
            continue
        init_func = _find_init_function(class_node)
        if init_func is None:
            continue
        local_vars: dict[str, DimExpr] = {}

        def _walk(stmts: list[_pyast.stmt]) -> None:
            for stmt in stmts:
                targets: list[tuple[Any, Any]] = []
                if isinstance(stmt, _pyast.Assign):
                    targets = [(t, stmt.value) for t in stmt.targets]
                elif isinstance(stmt, _pyast.AnnAssign) and stmt.value is not None:
                    targets = [(stmt.target, stmt.value)]
                elif isinstance(stmt, _pyast.If):
                    branch = _eval_config_condition(
                        stmt.test, config=config, local_vars=local_vars
                    )
                    if branch is not False:
                        _walk(stmt.body)
                    if branch is not True:
                        _walk(stmt.orelse)
                    continue
                elif isinstance(stmt, (_pyast.For, _pyast.While, _pyast.With, _pyast.Try)):
                    _walk(stmt.body)
                    continue
                for target, value in targets:
                    resolved = _resolve_dim_expr(
                        value, config=config, local_vars=local_vars, context=ctx
                    )
                    if not isinstance(resolved, int):
                        continue
                    if isinstance(target, _pyast.Name):
                        local_vars[target.id] = resolved
                    elif isinstance(target, _pyast.Attribute) and _is_self_attr(target):
                        local_vars[target.attr] = resolved
                        extra.setdefault(target.attr, resolved)

        _walk(init_func.body)
    return extra


def _collect_forward_scalar_locals(
    spec: ArchitectureSpec, dims: dict[str, DimExpr]
) -> dict[str, int]:
    """Fold forward-local scalar bindings (``hc = self.hc_mult``) into dims.

    A forward often aliases a config value to a short local (``hc``) and then
    uses it in a shape expression (``.split([hc, hc, hc * hc])``). The config is
    available to us, so resolve those locals to their integer values by walking
    each class's ``forward`` for simple ``name = <int expr over config>``
    assignments.
    """
    import ast as _pyast

    extra: dict[str, int] = {}
    for structure in (getattr(spec, "class_registry", None) or {}).values():
        class_node = getattr(structure, "node", None)
        if class_node is None:
            continue
        for item in getattr(class_node, "body", []):
            if not (isinstance(item, _pyast.FunctionDef) and item.name == "forward"):
                continue
            for stmt in _pyast.walk(item):
                if (
                    isinstance(stmt, _pyast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], _pyast.Name)
                ):
                    value = _eval_dim_expr(stmt.value, {**dims, **extra})
                    if isinstance(value, int):
                        extra.setdefault(stmt.targets[0].id, value)
    return extra


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


# Concrete dtype tokens, checked longest-first so ``bfloat16`` wins over the
# ``float16`` substring it contains.
_CONCRETE_DTYPE_TOKENS: tuple[tuple[str, str], ...] = (
    ("bfloat16", "bfloat16"),
    # Low-precision quant dtypes. Placed before the wider float/int tokens so a
    # substring match resolves e.g. ``float8_e4m3fn`` -> ``fp8_e4m3`` rather than
    # hitting the generic ``float`` -> ``float32`` fallback. ``uint4`` precedes
    # ``int4`` because ``uint4`` contains ``int4`` as a substring.
    ("float8_e4m3", "fp8_e4m3"),
    ("float8_e5m2", "fp8_e5m2"),
    ("e4m3", "fp8_e4m3"),
    ("e5m2", "fp8_e5m2"),
    ("fp8", "fp8_e4m3"),
    ("nf4", "nf4"),
    ("fp4", "fp4"),
    ("uint4", "uint4"),
    ("int4", "int4"),
    ("float64", "float64"),
    ("float32", "float32"),
    ("float16", "float16"),
    ("complex64", "complex64"),
    ("complex128", "complex128"),
    ("int64", "int64"),
    ("int32", "int32"),
    ("int16", "int16"),
    ("uint8", "uint8"),
    ("int8", "int8"),
    ("bool", "bool"),
    ("double", "float64"),
    ("half", "float16"),
    ("long", "int64"),
    ("short", "int16"),
    ("float", "float32"),
)


def _resolve_cast_dtype(
    dtype_detail: str, source_dtype: str, working_dtype: str
) -> str:
    """Resolve the target dtype of a ``.to(...)`` cast from its recorded expr.

    ``dtype_detail`` is the raw argument text captured by the AST analyzer, e.g.
    ``torch.float32``, ``int32``, or a variable like ``dtype`` / ``x.dtype`` /
    ``input_dtype``. A concrete dtype token is honoured directly. A *variable*
    dtype reference is the "compute in float32, cast back" idiom — it restores
    the module's working dtype (``dtype = hidden_states.dtype``), so it resolves
    to ``working_dtype`` rather than silently keeping the float32 source dtype.
    """
    text = dtype_detail.strip().lower()
    if not text:
        return source_dtype
    for token, resolved in _CONCRETE_DTYPE_TOKENS:
        if token in text:
            return resolved
    # A non-concrete dtype expression (a captured local or ``<tensor>.dtype``)
    # names the module's working precision the float32 compute is cast back to.
    return working_dtype


def _clean_dtype(raw: str) -> str:
    """Normalize a config dtype token: drop a ``torch.`` prefix, lower-case."""
    return raw.removeprefix("torch.").lower()


def _config_dtype(config: dict[str, Any]) -> str:
    """Resolve the model's real compute/activation dtype.

    Newer HF configs renamed ``torch_dtype`` to ``dtype`` and multimodal models
    often set it only on ``text_config`` — so search the top level then the known
    sub-configs for either key. A quantized checkpoint's explicit compute dtype
    (``bnb_4bit_compute_dtype``) wins, since that is the precision the dequantized
    weights are computed in.
    """
    quant = config.get("quantization_config")
    if isinstance(quant, dict):
        compute = quant.get("bnb_4bit_compute_dtype")
        if isinstance(compute, str) and compute:
            return _clean_dtype(compute)
    scopes: list[dict[str, Any]] = [config]
    for key in ("text_config", "language_config", "vision_config"):
        nested = config.get(key)
        if isinstance(nested, dict):
            scopes.append(nested)
    for scope in scopes:
        for field_name in ("torch_dtype", "dtype"):
            raw = scope.get(field_name)
            if isinstance(raw, str) and raw:
                return _clean_dtype(raw)
    return "float16"


# FP8 storage-format tokens (``fmt`` in an fp8 quantization_config) -> display dtype.
_FP8_FMT_DTYPES: dict[str, str] = {"e4m3": "fp8_e4m3", "e5m2": "fp8_e5m2"}


def _quant_storage_dtype(quant: dict[str, Any]) -> str | None:
    """Weight *storage* dtype implied by a HF ``quantization_config``.

    Handles fp8 (e4m3/e5m2), bitsandbytes 4/8-bit, and gptq/awq bit widths.
    Returns ``None`` when the method/bit-width is unrecognized.
    """
    method = str(quant.get("quant_method") or "").lower()
    if method == "fp8" or quant.get("fmt"):
        fmt = str(quant.get("fmt") or "e4m3").lower()
        return _FP8_FMT_DTYPES.get(fmt, "fp8_e4m3")
    if (
        method == "bitsandbytes"
        or quant.get("load_in_4bit")
        or quant.get("load_in_8bit")
    ):
        if quant.get("load_in_8bit"):
            return "int8"
        qtype = str(quant.get("bnb_4bit_quant_type") or "").lower()
        return qtype if qtype in {"nf4", "fp4"} else "int4"
    bits = quant.get("bits")
    if method in {"gptq", "awq"} or bits in {4, 8}:
        return "int8" if bits == 8 else "int4"
    return None


# Structural / non-module tokens that appear in a node id but do not name a
# module attribute; dropped when reconstructing a module path.
_STRUCTURAL_ID_TOKENS: frozenset[str] = frozenset(
    {"seq", "sidefeed", "sideproducer", "side", "loop_repeated", "mirror"}
)


def _module_path_segments(node_id: str) -> list[str]:
    """Reconstruct a node's local module-attribute path from its graph id.

    Node ids look like ``decoder/11x_.../self_attn/seq:0:q_proj:q_proj:0`` — the
    module attributes (``self_attn``, ``q_proj``) are interleaved with structural
    tokens, repeat-group titles, op markers (``@op_*``) and indices, which are all
    dropped here so the result can be matched against ``modules_to_not_convert``.
    """
    segments: list[str] = []
    for token in re.split(r"[/:]", str(node_id)):
        token = token.strip().split("^", 1)[0].strip()
        if not token or token.startswith("@") or token.isdigit():
            continue
        if token.lower() in _STRUCTURAL_ID_TOKENS:
            continue
        if re.match(r"^\d+x_", token):  # repeat-group title, e.g. 45x_Decoder
            continue
        if segments and segments[-1] == token:  # collapse doubled attr (q_proj:q_proj)
            continue
        segments.append(token)
    return segments


def _module_path_matches(segments: list[str], pattern: tuple[str, ...]) -> bool:
    """True when *pattern* (a normalized not-convert tail) applies to *segments*.

    A single-segment pattern (``visual``, ``lm_head``) matches if it appears
    anywhere (covering all of that module's descendants). A multi-segment pattern
    must appear as a contiguous run, so ``mlp.shared_experts.down_proj`` never
    matches ``visual.merger.down_proj`` and vice versa.
    """
    if not pattern:
        return False
    if len(pattern) == 1:
        return pattern[0] in segments
    span = len(pattern)
    for start in range(len(segments) - span + 1):
        if tuple(segments[start : start + span]) == pattern:
            return True
    return False


def _normalize_module_patterns(modules: Any) -> tuple[tuple[str, ...], ...]:
    """Normalize a ``modules_to_not_convert`` list into matchable segment tuples.

    Drops a leading ``model.`` and any numeric index segments so per-layer entries
    like ``model.layers.0.self_attn.q_proj`` collapse to ``(self_attn, q_proj)``.
    """
    if not isinstance(modules, (list, tuple)):
        return ()
    patterns: list[tuple[str, ...]] = []
    for entry in modules:
        segs = [seg for seg in str(entry).split(".") if seg and not seg.isdigit()]
        if segs and segs[0] == "model":
            segs = segs[1:]
        if segs and segs[0] == "layers":
            segs = segs[1:]
        if segs:
            patterns.append(tuple(segs))
    return tuple(dict.fromkeys(patterns))


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


def _eval_config_condition(
    test: ast.AST,
    *,
    config: dict[str, Any],
    local_vars: dict[str, Any],
) -> bool | None:
    """Try to evaluate an ``if`` condition against *config* and *local_vars*.

    Returns ``True``/``False`` when the condition can be statically resolved,
    or ``None`` when it cannot.
    """
    # self.<attr> — look up in local_vars or config
    if isinstance(test, ast.Attribute) and _is_self_attr(test):
        val = local_vars.get(test.attr)
        if val is None:
            val = config.get(test.attr)
        if val is not None:
            return bool(val)
    # config.<attr>
    if (
        isinstance(test, ast.Attribute)
        and isinstance(test.value, ast.Name)
        and test.value.id == "config"
    ):
        val = config.get(test.attr)
        if val is not None:
            return bool(val)
    # not <expr>
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _eval_config_condition(
            test.operand, config=config, local_vars=local_vars
        )
        if inner is not None:
            return not inner
    # <a> is not None
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.IsNot)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    ):
        return _eval_config_condition(
            test.left, config=config, local_vars=local_vars
        )
    return None


def _resolve_int_tuple(
    node: ast.AST | None,
    *,
    config: dict[str, Any],
    local_vars: dict[str, DimExpr],
    context: ShapeContext,
) -> tuple[int, ...] | None:
    """Resolve a conv kernel/stride/padding arg to a tuple of concrete ints.

    Accepts a scalar (``kernel_size=2`` → ``(2,)``) or an ``ast.Tuple``/``List``
    (``kernel_size=(2, 2)`` → ``(2, 2)``). Returns ``None`` unless every element
    resolves to a concrete int, so symbolic geometry never fabricates a spatial
    reduction.
    """
    if node is None:
        return None
    if isinstance(node, (ast.Tuple, ast.List)):
        elems = node.elts
    else:
        elems = [node]
    resolved: list[int] = []
    for elem in elems:
        value = _resolve_dim_expr(
            elem, config=config, local_vars=local_vars, context=context
        )
        if not isinstance(value, int):
            return None
        resolved.append(value)
    return tuple(resolved) if resolved else None


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
) -> ModuleLinearSpec | ModuleConvSpec | ModuleEmbeddingSpec | ModuleParameterSpec | None:
    if not isinstance(node, ast.Call):
        return None
    class_name = _call_class_name(node) or ""
    args = list(node.args)
    if re.search(r"Conv(Transpose)?[123]d$", class_name):
        # nn.Conv{1,2,3}d(in_channels, out_channels, kernel_size, ...)
        in_channels = (
            _resolve_dim_expr(
                args[0], config=config, local_vars=local_vars, context=context
            )
            if len(args) >= 1
            else None
        )
        out_channels = (
            _resolve_dim_expr(
                args[1], config=config, local_vars=local_vars, context=context
            )
            if len(args) >= 2
            else None
        )
        # positional kernel_size/stride/padding: nn.Conv2d(in, out, k, stride, pad)
        geometry: dict[str, ast.AST] = {}
        for name, idx in (("kernel_size", 2), ("stride", 3), ("padding", 4)):
            if len(args) > idx:
                geometry[name] = args[idx]
        for keyword in node.keywords:
            if keyword.arg == "in_channels" and in_channels is None:
                in_channels = _resolve_dim_expr(
                    keyword.value, config=config, local_vars=local_vars, context=context
                )
            if keyword.arg == "out_channels" and out_channels is None:
                out_channels = _resolve_dim_expr(
                    keyword.value, config=config, local_vars=local_vars, context=context
                )
            if keyword.arg in {"kernel_size", "stride", "padding"}:
                geometry[keyword.arg] = keyword.value
        if out_channels is not None:
            kernel = _resolve_int_tuple(
                geometry.get("kernel_size"),
                config=config,
                local_vars=local_vars,
                context=context,
            )
            # When ``stride`` is omitted, nn.Conv2d defaults it to 1 (not to
            # kernel_size). Only reduce spatial when kernel is known; a missing
            # stride then means the default stride of 1.
            stride = _resolve_int_tuple(
                geometry.get("stride"),
                config=config,
                local_vars=local_vars,
                context=context,
            )
            if kernel is not None and stride is None:
                stride = (1,)
            padding = _resolve_int_tuple(
                geometry.get("padding"),
                config=config,
                local_vars=local_vars,
                context=context,
            )
            return ModuleConvSpec(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
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
    if bool(re.search(r"(?i)^Embedding$", class_name)) or node.label == "Embedding":
        return True
    # Some models wrap the embedding in a factory function (e.g. ``init_method``).
    # Fall back to the ``role`` metadata assigned by the block tree.
    return node.metadata.get("role") == "embedding"


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


def _is_conv(node: ModelGraphNode) -> bool:
    if node.operation not in {OperationKind.NN_MODULE, OperationKind.TORCH_FUNCTIONAL}:
        return False
    class_name = node.metadata.get("class_name") or node.label or ""
    return bool(re.search(r"(?i)^Conv(Transpose)?[123]d$", str(class_name)))


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
